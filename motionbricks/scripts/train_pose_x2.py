#!/usr/bin/env python3
"""Train MotionBricks pose backbone on X2 motion-lib clips.

Requires a previously trained VQVAE checkpoint
(``out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt`` by default).

Usage (smoke / dev — 5090 < 15 min):

    python scripts/train_pose_x2.py --max_clips 10 --max_steps 1000 --batch_size 2

Usage (full Path A)::

    python scripts/train_pose_x2.py --max_steps 200000 --batch_size 4
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

MB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MB_ROOT.parent
sys.path.insert(0, str(MB_ROOT))

from motionbricks.data.synthetic_dataset import collate_batch  # noqa: E402
from motionbricks.data.x2_category_dataset import (  # noqa: E402
  X2MotionDataset,
  build_finetune_sampler,
  default_cache_dir_for,
)
from motionbricks.data.x2_loco_filters import (  # noqa: E402
  DEFAULT_EXCLUDE_PATTERNS,
  DEFAULT_INCLUDE_PATTERNS,
)
from motionbricks.helper.pl_util import load_motion_rep  # noqa: E402


def load_config(
  result_dir: Path,
  max_steps: int,
  vqvae_ckpt: Path,
  devices: int = 1,
  num_nodes: int = 1,
  strategy: str = "auto",
  out_name: str = "motionbricks_pose_x2",
):
  version_dir = result_dir / out_name / "version_1"
  hparams_path = version_dir / "hparams.yaml"
  if not hparams_path.is_file():
    raise FileNotFoundError(
      f"Missing {hparams_path}. Run: python scripts/build_x2_skeleton_assets.py "
      f"(or scaffold the {out_name}/version_1 dir manually for an alt config)."
    )
  conf = OmegaConf.load(hparams_path)

  with open_dict(conf):
    conf.data = {"folder": str(version_dir), "text_embeddings": None}
    conf.skeleton.folder = str(version_dir / "skeleton")
    conf.motion_rep.stats.folder = str(version_dir / "stats" / "motion")
    conf.trainer.devices = devices
    conf.trainer.num_nodes = num_nodes
    conf.trainer.max_steps = max_steps
    conf.trainer.accelerator = "auto"
    conf.trainer.strategy = strategy
    conf.trainer.enable_progress_bar = True
    conf.trainer.log_every_n_steps = 10
    conf.trainer.val_check_interval = max_steps
    conf.trainer.num_sanity_val_steps = 0
    conf.model.scheduler.num_training_steps = max_steps
    conf.model.args.vqvae_model_ckpt_path = str(vqvae_ckpt)
    conf.id = "x2"
    conf.run_dir = "."
    conf.out_dir = str(result_dir / out_name)

  resolved = OmegaConf.to_container(conf, resolve=True)
  conf = OmegaConf.create(resolved)
  return conf, version_dir


def parse_pkl_args(values: List[str]) -> List[Path]:
  """Comma/space separated PKL paths, with DIRECTORIES expanded.

  A directory expands to every ``*.pkl`` under it (recursive, sorted for
  deterministic ordering and therefore a stable dataset cache identity). This
  lets a directory-mode corpus -- one pkl per clip, which is how the X2 direct
  retarget is produced -- be used directly, instead of first merging 134k clips
  into a single ~67 GB file. Applies to ``--finetune-pkl`` too, since that is
  parsed through here as well.

  ``metadata.pkl`` sidecars are skipped: they are corpus bookkeeping, not
  motions, and would fail FK extraction.
  """
  out: List[Path] = []
  for v in values:
    for piece in v.split(","):
      piece = piece.strip()
      if not piece:
        continue
      p = Path(piece)
      if p.is_dir():
        found = sorted(
          q for q in p.rglob("*.pkl") if q.name != "metadata.pkl"
        )
        if not found:
          raise SystemExit(f"--pkl directory has no *.pkl files: {p}")
        print(f"  expanded {p} -> {len(found)} pkl files")
        out.extend(found)
      else:
        out.append(p)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description="X2 pose backbone training")
  parser.add_argument("--result_dir", type=Path, default=MB_ROOT / "out")
  parser.add_argument(
    "--pkl",
    nargs="+",
    default=[str(REPO_ROOT / "gear_sonic/data/motions/x2_demo_bank.pkl")],
  )
  parser.add_argument("--filter", choices=["none", "loco"], default="none")
  parser.add_argument(
    "--include-keys-pkl", type=Path, default=None,
    help="Pickle of an explicit (windowed) key whitelist; AND-composed with "
    "pattern filters (same semantics as train_root_x2).",
  )
  parser.add_argument("--max_steps", type=int, default=1000)
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--num_workers", type=int, default=2)
  parser.add_argument("--max_clips", type=int, default=None)
  parser.add_argument("--min_frames", type=int, default=80)
  parser.add_argument("--max_frames", type=int, default=200)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--finetune-pkl", default=None,
    help="PKL of PRIORITY (new) motion files to oversample; its keys must also be "
         "present via --pkl. None -> uniform sampling.")
  parser.add_argument("--finetune-sample-rate", type=float, default=0.3,
    help="fraction of samples drawn from --finetune-pkl clips (default 0.3)")
  parser.add_argument("--save-every-n-steps", type=int, default=200)
  parser.add_argument("--save-top-k", type=int, default=-1)
  parser.add_argument(
    "--vqvae-ckpt",
    type=Path,
    default=MB_ROOT / "out/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt",
  )
  parser.add_argument(
    "--out-name",
    default="motionbricks_pose_x2",
    help=(
      "Output sub-directory under --result_dir. Reads "
      "<result_dir>/<out_name>/version_1/hparams.yaml and writes checkpoints "
      "next to it. Use a non-default value (e.g. motionbricks_pose_x2_v2) to "
      "run an alternate config without disturbing the main version_1 outputs."
    ),
  )
  parser.add_argument("--no-progress-bar", action="store_true")
  parser.add_argument(
    "--devices",
    type=int,
    default=1,
    help="Number of GPUs per node (>=2 enables DDP).",
  )
  parser.add_argument("--num-nodes", type=int, default=1)
  parser.add_argument(
    "--strategy",
    default=None,
    help="Lightning strategy. Defaults to 'auto' for 1 GPU, 'ddp' for >1.",
  )
  parser.add_argument(
    "--resume",
    type=Path,
    default=None,
    help="Resume training from a Lightning checkpoint.",
  )
  parser.add_argument(
    "--init-from",
    type=Path,
    default=None,
    help="Weights-only init from a Lightning checkpoint: loads model.state_dict "
    "but starts a FRESH optimizer, FRESH LR scheduler, and step counter at 0. "
    "Use for fine-tuning when the previous run's cosine schedule has bottomed "
    "out. Mutually exclusive with --resume. (Separate from --vqvae-ckpt, which "
    "loads the sub-VQVAE network.)",
  )
  parser.add_argument("--use-wandb", action="store_true")
  parser.add_argument("--wandb-project", default="TRL_X2Ultra_Planner")
  parser.add_argument("--wandb-name", default=None)
  args = parser.parse_args()

  if args.resume is not None and args.init_from is not None:
    raise SystemExit("--resume and --init-from are mutually exclusive")

  if not args.vqvae_ckpt.is_file():
    raise FileNotFoundError(
      f"X2 VQVAE checkpoint not found: {args.vqvae_ckpt}. "
      "Run scripts/train_vqvae_x2.py first."
    )

  pl.seed_everything(args.seed, workers=True)
  strategy = args.strategy or ("ddp" if args.devices > 1 else "auto")
  conf, version_dir = load_config(
    args.result_dir,
    args.max_steps,
    args.vqvae_ckpt,
    devices=args.devices,
    num_nodes=args.num_nodes,
    strategy=strategy,
    out_name=args.out_name,
  )

  motion_rep = load_motion_rep(conf)
  feat_dim = len(motion_rep.indices["all"])
  print(f"X2 motion_rep feat_dim={feat_dim} (skeleton={conf.skeleton.name})")
  print(f"VQVAE checkpoint: {args.vqvae_ckpt}")

  if args.filter == "loco":
    include_patterns = DEFAULT_INCLUDE_PATTERNS
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
  else:
    include_patterns = None
    exclude_patterns = None

  include_keys = None
  if args.include_keys_pkl is not None:
    import joblib
    include_keys = set(joblib.load(args.include_keys_pkl))
    print(f"X2 include-keys whitelist: {len(include_keys):,} keys from {args.include_keys_pkl}")

  pkls = parse_pkl_args(args.pkl)
  dataset = X2MotionDataset(
    pkls,
    motion_rep,
    cache_dir=default_cache_dir_for(version_dir, pkls),
    min_frames=args.min_frames,
    max_frames=args.max_frames,
    max_clips=args.max_clips,
    include_patterns=include_patterns,
    exclude_patterns=exclude_patterns,
    include_keys=include_keys,
  )
  print(f"X2 dataset: {len(dataset)} clips from {len(pkls)} PKL(s) (filter={args.filter})")
  if len(dataset) == 0:
    raise RuntimeError("Empty dataset after FK extraction")

  effective_batch = min(args.batch_size, len(dataset))
  # Priority sampling: oversample the new motion files (--finetune-pkl) to
  # --finetune-sample-rate. None -> uniform shuffle (legacy behaviour).
  ft_sampler = None
  if args.finetune_pkl:
    ft_sampler = build_finetune_sampler(
      dataset, parse_pkl_args([args.finetune_pkl]),
      args.finetune_sample_rate, seed=args.seed,
    )
  dataloader = DataLoader(
    dataset,
    batch_size=effective_batch,
    shuffle=ft_sampler is None,
    sampler=ft_sampler,
    num_workers=args.num_workers,
    collate_fn=collate_batch,
    persistent_workers=args.num_workers > 0,
    pin_memory=True,
    prefetch_factor=4 if args.num_workers > 0 else None,
  )

  model_conf = copy.deepcopy(conf.model)
  with open_dict(model_conf):
    pose_vqvae_net = instantiate(
      model_conf.pose_vqvae_network,
      motion_rep=motion_rep.dual_rep.local_motion_rep,
    )
    backbone_net = instantiate(
      model_conf.backbone_network,
      motion_rep=motion_rep,
      _recursive_=False,
    )
    optimizer_fn = instantiate(model_conf.optimizer)
    scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None
    model = instantiate(
      model_conf,
      pose_vqvae_network=pose_vqvae_net,
      root_vqvae_network=None,
      backbone_network=backbone_net,
      motion_rep=motion_rep,
      optimizer=optimizer_fn,
      scheduler=scheduler_fn,
      _recursive_=False,
    )

  ckpt_dir = version_dir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  ckpt_callback = ModelCheckpoint(
    dirpath=str(ckpt_dir),
    filename="model-{step:07d}",
    save_last=True,
    save_top_k=args.save_top_k,
    every_n_train_steps=args.save_every_n_steps,
    monitor=None,
    save_weights_only=False,
  )

  logger = False
  if args.use_wandb:
    from pytorch_lightning.loggers import WandbLogger  # noqa: WPS433
    logger = WandbLogger(
      project=args.wandb_project,
      name=args.wandb_name or f"pose_x2_d{args.devices}_b{effective_batch}_s{args.max_steps}",
      save_dir=str(version_dir),
    )

  trainer = pl.Trainer(
    max_steps=conf.trainer.max_steps,
    devices=conf.trainer.devices,
    num_nodes=conf.trainer.num_nodes,
    accelerator=conf.trainer.accelerator,
    strategy=conf.trainer.strategy,
    precision=conf.trainer.precision,
    gradient_clip_val=conf.trainer.gradient_clip_val,
    enable_progress_bar=not args.no_progress_bar,
    log_every_n_steps=conf.trainer.log_every_n_steps,
    num_sanity_val_steps=0,
    enable_checkpointing=True,
    callbacks=[ckpt_callback],
    logger=logger,
    use_distributed_sampler=ft_sampler is None,
  )

  resume_path = str(args.resume) if args.resume else None

  if args.init_from is not None:
    init_path = Path(args.init_from).resolve()
    if not init_path.is_file():
      raise FileNotFoundError(f"--init-from path not found: {init_path}")
    print(f"Loading weights-only from {init_path} (fresh optimizer + scheduler)")
    init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    state_dict = init_ckpt.get("state_dict", init_ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
      print(f"  WARN missing keys ({len(missing)}); first 5: {missing[:5]}")
    if unexpected:
      print(f"  WARN unexpected keys ({len(unexpected)}); first 5: {unexpected[:5]}")
    print("  loaded; resetting optimizer/scheduler/step_counter to step 0")
  print(
    f"Starting X2 pose training: max_steps={args.max_steps} "
    f"devices={args.devices} num_nodes={args.num_nodes} strategy={strategy} "
    f"batch_per_dev={effective_batch} ckpt_every={args.save_every_n_steps}"
    + (f" resume={resume_path}" if resume_path else "")
  )
  print(f"  VQVAE loaded: {model.vqvae_model_loaded}")
  trainer.fit(model, train_dataloaders=dataloader, ckpt_path=resume_path)
  print(f"Done. Final ckpt -> {ckpt_callback.last_model_path}")


if __name__ == "__main__":
  main()
