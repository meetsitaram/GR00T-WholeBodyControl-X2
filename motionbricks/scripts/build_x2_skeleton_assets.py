#!/usr/bin/env python3
"""Build X2 skeleton assets, motion stats, and bootstrap hparams.yaml for VQVAE/pose/root.

The G1 hparams (``out/motionbricks_{vqvae,pose,root}/version_1/hparams.yaml``)
are used as templates and patched in-place to reference X2 paths and skeleton.

Usage (from ``motionbricks/``)::

    # Smoke / dev: stats over a small subset (fast).
    python scripts/build_x2_skeleton_assets.py --max-clips-stats 200

    # Production: stats over the full SONIC corpus (filter off).
    python scripts/build_x2_skeleton_assets.py --max-clips-stats 0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

MB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MB_ROOT.parent
sys.path.insert(0, str(MB_ROOT))

from motionbricks.data.x2_category_dataset import (  # noqa: E402
  X2MotionDataset,
  build_x2_motion_rep,
)
from motionbricks.data.x2_pkl_to_motion import (  # noqa: E402
  assert_x2_mjcf_coherent,
  build_neutral_joints_from_mjcf,
  default_x2_mjcf_path,
)


SKELETON_TARGET = "motionbricks.motionlib.core.skeletons.x2.X2Skeleton34"
SKELETON_NAME = "x2skel34"


def _patch_hparams_x2(
  conf,
  *,
  variant: str,
  out_root: str,
  vqvae_ckpt_glob: Optional[str],
) -> None:
  """In-place: patch G1 hparams to point at X2 skeleton + X2 paths.

  ``variant`` is one of ``vqvae`` / ``pose`` / ``root``.
  """
  with open_dict(conf):
    conf.skeleton.base_name = SKELETON_NAME
    conf.skeleton.name = SKELETON_NAME
    conf.skeleton._target_ = SKELETON_TARGET
    conf.skeleton.folder = f"{out_root}/skeleton"
    conf.motion_rep.stats.folder = f"{out_root}/stats/motion"
    conf.out_dir = f"out/motionbricks_{variant}_x2"
    conf.wandb_run = f"motionbricks_{variant}_x2"
    if "loggers" in conf and "wandb_run_name" in conf.loggers:
      conf.loggers.wandb_run_name = f"motionbricks_{variant}_x2"
    if "data" in conf:
      if "name" in conf.data:
        conf.data.name = "motionbricks-X2"
      if "folder" in conf.data:
        conf.data.folder = "../datasets/motionbricks-X2"
    if (
      variant == "pose"
      and vqvae_ckpt_glob is not None
      and "args" in conf.model
      and "vqvae_model_ckpt_path" in conf.model.args
    ):
      conf.model.args.vqvae_model_ckpt_path = vqvae_ckpt_glob


def _bootstrap_variant_dir(
  *,
  variant: str,
  src_hparams: Path,
  dst_dir: Path,
  shared_skel_dir: Path,
  shared_stats_dir: Path,
  vqvae_ckpt_glob: Optional[str],
  overwrite_hparams: bool,
) -> None:
  dst_dir.mkdir(parents=True, exist_ok=True)
  hparams_dst = dst_dir / "hparams.yaml"

  if not src_hparams.is_file():
    raise FileNotFoundError(
      f"Missing template {src_hparams}. Pull G1 LFS checkpoints first "
      "(see motionbricks/README.md 'Setup')."
    )

  if hparams_dst.is_file() and not overwrite_hparams:
    print(f"  {variant}: hparams.yaml already exists at {hparams_dst} (skipped, use --overwrite-hparams to redo)")
  else:
    conf = OmegaConf.load(src_hparams)
    _patch_hparams_x2(
      conf,
      variant=variant,
      out_root=f"out/motionbricks_{variant}_x2/version_1",
      vqvae_ckpt_glob=vqvae_ckpt_glob,
    )
    OmegaConf.save(conf, hparams_dst)
    print(f"  {variant}: wrote {hparams_dst}")

  # Symlink shared skeleton + stats so all three variants share the same data
  # without copying ~tens of MB. Recreate symlinks on every run to handle moves.
  for sub_name, src_dir in (("skeleton", shared_skel_dir), ("stats/motion", shared_stats_dir)):
    target = dst_dir / sub_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
      if target.is_symlink():
        target.unlink()
      elif target == shared_skel_dir or target == shared_stats_dir:
        continue
      else:
        # already a real directory in the canonical (vqvae) variant dir
        continue
    target.symlink_to(src_dir.resolve())


def _expand_pkls(values):
  """Expand any directory in --pkl to the *.pkl files under it (sorted).

  The X2 direct-retarget corpus is directory-mode (one pkl per clip, 134,793 of
  them), so a bare joblib.load on the directory raises IsADirectoryError. Mirrors
  parse_pkl_args() in the three trainers.
  """
  from pathlib import Path as _P
  out = []
  for v in values:
    q = _P(str(v))
    if q.is_dir():
      found = sorted(x for x in q.rglob("*.pkl") if x.name != "metadata.pkl")
      if not found:
        raise SystemExit(f"--pkl directory has no *.pkl files: {q}")
      print(f"  expanded {q} -> {len(found)} pkl files")
      out.extend(found)
    else:
      out.append(q)
  return out


def main() -> None:
  p = argparse.ArgumentParser(description="Build X2 MotionBricks skeleton + stats + hparams")
  p.add_argument(
    "--out-root",
    type=Path,
    default=MB_ROOT / "out",
    help="Root containing motionbricks_{vqvae,pose,root}_x2/version_1/",
  )
  p.add_argument(
    "--pkl",
    type=Path,
    nargs="+",
    default=[REPO_ROOT / "gear_sonic/data/motions/x2_demo_bank.pkl"],
  )
  p.add_argument("--mjcf", type=Path, default=None)
  p.add_argument(
    "--max-clips-stats",
    type=int,
    default=0,
    help="Clips used for mean/std (0 = use all clips that pass min/max frames)",
  )
  p.add_argument(
    "--filter",
    choices=["none", "loco"],
    default="none",
    help="Apply walk/turn name filter when building stats (default off, matches SONIC)",
  )
  p.add_argument("--recompute-cache", action="store_true")
  p.add_argument(
    "--overwrite-hparams",
    action="store_true",
    help="Re-bootstrap pose/root hparams.yaml even if they already exist",
  )
  p.add_argument(
    "--g1-vqvae-hparams",
    type=Path,
    default=MB_ROOT / "out/motionbricks_vqvae/version_1/hparams.yaml",
  )
  p.add_argument(
    "--g1-pose-hparams",
    type=Path,
    default=MB_ROOT / "out/motionbricks_pose/version_1/hparams.yaml",
  )
  p.add_argument(
    "--g1-root-hparams",
    type=Path,
    default=MB_ROOT / "out/motionbricks_root/version_1/hparams.yaml",
  )
  args = p.parse_args()

  vqvae_dir = args.out_root / "motionbricks_vqvae_x2/version_1"
  pose_dir = args.out_root / "motionbricks_pose_x2/version_1"
  root_dir = args.out_root / "motionbricks_root_x2/version_1"
  vqvae_dir.mkdir(parents=True, exist_ok=True)

  skel_dir = vqvae_dir / "skeleton"
  stats_dir = vqvae_dir / "stats/motion"
  skel_dir.mkdir(parents=True, exist_ok=True)
  stats_dir.mkdir(parents=True, exist_ok=True)

  # Phase 1.1 — sphere-feet structural check on whichever MJCF we will use.
  mjcf = args.mjcf or default_x2_mjcf_path(REPO_ROOT)
  print(f"X2 MJCF: {mjcf}")
  assert_x2_mjcf_coherent(mjcf)
  print("  -> coherence OK (nq=38, njnt=32, sphere geoms >= 24)")

  # Skeleton joints + parents
  joints, parents = build_neutral_joints_from_mjcf(mjcf)
  torch.save(joints, skel_dir / "joints.p")
  torch.save(parents, skel_dir / "parents.p")
  print(f"Wrote skeleton assets to {skel_dir}  (nbjoints={joints.shape[1]})")

  # Stats — feature mean/std on the full corpus (filter off by default).
  #
  # CRITICAL: stats must be computed on the *dual rep* (418-D), not the global
  # rep (414-D). The dual rep is what the on-disk stats file represents:
  # ``stats.mean`` is later split into [global_root | local_root | body] via
  # the indices in DualRootLocalBody.__init__. If we save 414-D stats, the
  # `motion_rep_dim == stats.get_dim()` assert in SeparatedRootLocalBody fires.
  motion_rep = build_x2_motion_rep(skel_dir, stats_dir, load_stats=False)
  dual_rep = motion_rep.dual_rep
  if dual_rep is None:
    raise RuntimeError(
      "X2 motion_rep has no dual_rep; check skeleton.name contains 'dual_root_global_joints'"
    )

  if args.filter == "loco":
    from motionbricks.data.x2_loco_filters import (
      DEFAULT_EXCLUDE_PATTERNS,
      DEFAULT_INCLUDE_PATTERNS,
    )

    include_patterns = DEFAULT_INCLUDE_PATTERNS
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
  else:
    include_patterns = None
    exclude_patterns = None

  # We still need the FK extractor to produce input_tensor_dicts; bypass the
  # X2MotionDataset cache because that one runs through the *global* rep.
  from motionbricks.data.x2_pkl_to_motion import X2MujocoFkExtractor
  import joblib

  extractor = X2MujocoFkExtractor(mjcf)
  total_frames = 0
  mean = None
  m2 = None
  d: Optional[int] = None
  loaded_clips = 0

  for pkl_path in _expand_pkls(args.pkl):
    lib = joblib.load(pkl_path)
    keys = sorted(lib.keys())
    if include_patterns is not None or exclude_patterns is not None:
      from motionbricks.data.x2_loco_filters import filter_motion_keys
      keys = filter_motion_keys(
        keys,
        include_patterns=include_patterns or (r".",),
        exclude_patterns=exclude_patterns or (),
      )
    if args.max_clips_stats > 0:
      remaining = args.max_clips_stats - loaded_clips
      if remaining <= 0:
        break
      keys = keys[:remaining]

    for key in keys:
      clip = lib[key]
      n_raw = int(clip["dof"].shape[0])
      if n_raw < 60 or n_raw > 400:
        continue
      try:
        inp = extractor.clip_to_input_dict(clip, subsample=1)
        feat = dual_rep(inp, to_normalize=False, return_numpy=False)
        if feat.dim() == 3:
          feat = feat.squeeze(0)
      except Exception:
        continue
      f64 = feat.detach().to(torch.float64)
      n = f64.shape[0]
      if d is None:
        d = int(f64.shape[-1])
        mean = torch.zeros(d, dtype=torch.float64)
        m2 = torch.zeros(d, dtype=torch.float64)
      mean = mean + f64.sum(dim=0)
      total_frames += n
      loaded_clips += 1

    del lib

  if mean is None or d is None or total_frames == 0:
    raise RuntimeError("No clips passed FK extraction; check PKL / MJCF.")

  mean = mean / total_frames

  # Second pass for variance (streaming with single-pass would risk numerical issues).
  loaded_clips = 0
  for pkl_path in _expand_pkls(args.pkl):
    lib = joblib.load(pkl_path)
    keys = sorted(lib.keys())
    if include_patterns is not None or exclude_patterns is not None:
      from motionbricks.data.x2_loco_filters import filter_motion_keys
      keys = filter_motion_keys(
        keys,
        include_patterns=include_patterns or (r".",),
        exclude_patterns=exclude_patterns or (),
      )
    if args.max_clips_stats > 0:
      remaining = args.max_clips_stats - loaded_clips
      if remaining <= 0:
        break
      keys = keys[:remaining]
    for key in keys:
      clip = lib[key]
      n_raw = int(clip["dof"].shape[0])
      if n_raw < 60 or n_raw > 400:
        continue
      try:
        inp = extractor.clip_to_input_dict(clip, subsample=1)
        feat = dual_rep(inp, to_normalize=False, return_numpy=False)
        if feat.dim() == 3:
          feat = feat.squeeze(0)
      except Exception:
        continue
      f64 = feat.detach().to(torch.float64)
      diff = f64 - mean
      m2 = m2 + (diff * diff).sum(dim=0)
      loaded_clips += 1
    del lib

  std = (m2 / total_frames).sqrt().clamp_min(1e-6)
  np.save(stats_dir / "mean.npy", mean.float().numpy())
  np.save(stats_dir / "std.npy", std.float().numpy())
  print(f"Wrote stats to {stats_dir}  (dim={d}, frames={total_frames}, clips={loaded_clips})")

  # Phase 3 — bootstrap hparams for all three variants.
  print("Bootstrapping hparams.yaml for all three X2 variants:")
  vqvae_ckpt_glob = (
    f"{args.out_root}/motionbricks_vqvae_x2/version_1/checkpoints/last.ckpt"
  )
  for variant, src in (
    ("vqvae", args.g1_vqvae_hparams),
    ("pose", args.g1_pose_hparams),
    ("root", args.g1_root_hparams),
  ):
    dst = args.out_root / f"motionbricks_{variant}_x2/version_1"
    _bootstrap_variant_dir(
      variant=variant,
      src_hparams=src,
      dst_dir=dst,
      shared_skel_dir=skel_dir,
      shared_stats_dir=stats_dir,
      vqvae_ckpt_glob=vqvae_ckpt_glob,
      overwrite_hparams=args.overwrite_hparams,
    )

  print("\nDone. Next steps:")
  print(f"  python scripts/train_vqvae_x2.py --max_steps 200 --batch_size 2 --max_clips 10")


if __name__ == "__main__":
  main()
