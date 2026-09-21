"""Load X2 MotionBricks checkpoints into a ready-to-stream NeuralPlannerCore.

This is the X2 counterpart to the G1 demo's `navigation_demo._initialize_inference_modles`
([`motionbricks/motion_backbone/demo/utils.py`](../demo/utils.py)). The G1 demo
uses Hydra `experiment.test()` plus a pre-baked clip-holder pickle; the X2
planner needs neither (we drop the clip-holder entirely; see `neural_planner.py`).

The loader:

  1. Reads each stage's `hparams.yaml` (`out/motionbricks_{vqvae,pose,root}_x2/version_1/`).
  2. Wires the user-selected VQVAE checkpoint path into the pose / root model
     hparams so `MotionModel._load_vqvae_models` finds it.
  3. Instantiates the pose + root LightningModules via Hydra and pulls weights
     from the user-selected checkpoint files.
  4. Wraps them in `motion_inference` (the predict() entry point), pairs with
     the `X2MujocoQposConverter`, and returns the ready-to-stream
     `NeuralPlannerCore`.

Usage::

    from motionbricks.motion_backbone.inference.load_x2_planner import (
        load_x2_planner, X2PlannerPaths,
    )

    paths = X2PlannerPaths.default()
    planner = load_x2_planner(paths, device="cuda")
    planner.reset(init_qpos)
    planner.replan_with_velocity([0.0, 0.5, 0.0, 0.95])  # walk forward 0.5 m/s
    while True:
        qpos = planner.get_next_frame()  # [38] torch.Tensor on `device`
        if planner.should_replan():
            planner.replan_with_velocity(...)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from motionbricks.helper.mujoco_helper_x2 import X2MujocoQposConverter
from motionbricks.helper.pl_util import load_motion_rep
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.motion_backbone.inference.neural_planner import NeuralPlannerCore


# Default X2 result-dir layout (the training scripts' default out/ tree).
# load_x2_planner.py lives at motionbricks/motionbricks/motion_backbone/inference/
# parents[3] is the outer `motionbricks/` package root.
_DEFAULT_RESULT_DIR = (
    Path(__file__).resolve().parents[3] / "out"
)  # motionbricks/out

_DEFAULT_VQVAE_VERSION = _DEFAULT_RESULT_DIR / "motionbricks_vqvae_x2/version_1"
# Round 2 (8-GPU cloud run, 2026-06-24/25) re-trained the pose model end-to-end
# in this base directory, so we no longer hop through motionbricks_pose_x2_v2
# (which held the Round 1 post-100k keyframe-warmup pose).
_DEFAULT_POSE_VERSION = _DEFAULT_RESULT_DIR / "motionbricks_pose_x2/version_1"
_DEFAULT_ROOT_VERSION = _DEFAULT_RESULT_DIR / "motionbricks_root_x2/version_1"

# Default location for the baked pose-template clip library, produced by
# ``motionbricks/scripts/build_x2_planner_clips.py``. Optional — when this
# file is missing, ``load_x2_planner`` skips clip-library loading and the
# returned ``NeuralPlannerCore`` only exposes the velocity-only path.
_DEFAULT_CLIP_LIBRARY = _DEFAULT_RESULT_DIR / "X2-clip.ckpt"


@dataclass
class X2PlannerPaths:
    """Paths to the three X2 checkpoints + asset folders.

    Attributes:
        vqvae_ckpt: Path to the VQVAE checkpoint (e.g. last.ckpt).
        pose_ckpt: Path to the pose-model checkpoint.
        root_ckpt: Path to the root-model checkpoint.
        vqvae_version_dir: Dir holding hparams.yaml + skeleton/ + stats/.
        pose_version_dir: Dir holding the pose model's hparams.yaml.
        root_version_dir: Dir holding the root model's hparams.yaml.
        mjcf_path: Optional override for the X2 MJCF (defaults to the canonical
            sphere-feet asset shipped at `default_x2_mjcf_path_str()`).
    """

    vqvae_ckpt: Path
    pose_ckpt: Path
    root_ckpt: Path
    vqvae_version_dir: Path = field(default_factory=lambda: _DEFAULT_VQVAE_VERSION)
    pose_version_dir: Path = field(default_factory=lambda: _DEFAULT_POSE_VERSION)
    root_version_dir: Path = field(default_factory=lambda: _DEFAULT_ROOT_VERSION)
    mjcf_path: Optional[Path] = None

    @classmethod
    def default(cls) -> "X2PlannerPaths":
        """Default paths: pinned step checkpoints from each X2 ``version_1`` dir.

        Pinned to a known-good training step rather than ``last.ckpt`` so a
        fresh training run doesn't silently re-point inference at an
        unverified checkpoint. Override via the CLI / dataclass kwargs when
        switching to a newer checkpoint.
        """
        return cls(
            vqvae_ckpt=_DEFAULT_VQVAE_VERSION / "checkpoints/model-step=0500000.ckpt",
            pose_ckpt=_DEFAULT_POSE_VERSION / "checkpoints/model-step=0500000.ckpt",
            root_ckpt=_DEFAULT_ROOT_VERSION / "checkpoints/model-step=0315000.ckpt",
        )

    def validate(self) -> None:
        """Raise FileNotFoundError if any required path is missing."""
        missing = []
        for label, p in [
            ("vqvae_ckpt", self.vqvae_ckpt),
            ("pose_ckpt", self.pose_ckpt),
            ("root_ckpt", self.root_ckpt),
            ("vqvae_version_dir", self.vqvae_version_dir),
            ("pose_version_dir", self.pose_version_dir),
            ("root_version_dir", self.root_version_dir),
        ]:
            if not Path(p).exists():
                missing.append(f"{label}={p}")
        if missing:
            raise FileNotFoundError(
                "X2PlannerPaths missing: " + ", ".join(missing)
            )


def _patch_hparams(
    version_dir: Path, vqvae_ckpt: Path, result_dir: Path, *, stage: str
):
    """Mirror `train_{pose,root}_x2.load_config` minus the Trainer plumbing.

    We keep the original ``trainer`` block intact (only override max_steps and
    a few harmless fields) because other interpolations like
    ``loggers.log_nsteps -> trainer.log_every_n_steps`` resolve through it.
    """
    hparams_path = version_dir / "hparams.yaml"
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Missing {hparams_path}")
    conf = OmegaConf.load(hparams_path)
    with open_dict(conf):
        conf.data = {"folder": str(version_dir), "text_embeddings": None}
        conf.skeleton.folder = str(version_dir / "skeleton")
        conf.motion_rep.stats.folder = str(version_dir / "stats" / "motion")
        # Tame the trainer to a 1-step single-GPU shell. Keep all keys present
        # so any ${trainer.*} interpolations elsewhere still resolve.
        conf.trainer.devices = 1
        conf.trainer.num_nodes = 1
        conf.trainer.max_steps = 1
        conf.trainer.strategy = "auto"
        conf.trainer.enable_progress_bar = False
        if "val_check_interval" in conf.trainer:
            conf.trainer.val_check_interval = 1
        conf.model.scheduler.num_training_steps = 1
        if "args" in conf.model and "vqvae_model_ckpt_path" in conf.model.args:
            conf.model.args.vqvae_model_ckpt_path = str(vqvae_ckpt)
        conf.id = "x2"
        conf.run_dir = "."
        conf.out_dir = str(result_dir / f"motionbricks_{stage}_x2")
        conf.wandb_run = None  # silence ${wandb_run} references
    return OmegaConf.create(OmegaConf.to_container(conf, resolve=True))


def _load_pose_hparams(version_dir: Path, vqvae_ckpt: Path, result_dir: Path):
    return _patch_hparams(version_dir, vqvae_ckpt, result_dir, stage="pose")


def _load_root_hparams(version_dir: Path, vqvae_ckpt: Path, result_dir: Path):
    return _patch_hparams(version_dir, vqvae_ckpt, result_dir, stage="root")


def _instantiate_pose_model(conf, motion_rep):
    """Same pattern as train_pose_x2.main() (minus checkpoint / trainer)."""
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
    return model


def _instantiate_root_model(conf, motion_rep):
    """Same pattern as train_root_x2.main() (minus checkpoint / trainer)."""
    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        backbone_net = instantiate(
            model_conf.backbone_network,
            motion_rep=motion_rep,
            _recursive_=False,
        )
        optimizer_fn = instantiate(model_conf.optimizer)
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None
        model = instantiate(
            model_conf,
            pose_vqvae_network=None,
            root_vqvae_network=None,
            backbone_network=backbone_net,
            motion_rep=motion_rep,
            optimizer=optimizer_fn,
            scheduler=scheduler_fn,
            _recursive_=False,
        )
    return model


def _load_state_dict_into(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        raise RuntimeError(f"Checkpoint {ckpt_path} is missing 'state_dict'")
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    # vqvae weights live in the pose model's _supporting_networks and get
    # re-loaded by _load_vqvae_models from `args.vqvae_model_ckpt_path`;
    # they show up as 'missing' here in strict mode. Don't fail on those.
    truly_missing = [
        k for k in missing
        if not (k.startswith("pose_net.") or k.startswith("root_net."))
    ]
    if truly_missing:
        raise RuntimeError(
            f"Loading {ckpt_path}: {len(truly_missing)} unexpected missing key(s), "
            f"first 5: {truly_missing[:5]}"
        )


def load_x2_models(
    paths: X2PlannerPaths,
    device: str = "cuda",
) -> tuple[motion_inference, X2MujocoQposConverter]:
    """Load the pose+root X2 models and pair with the X2 MuJoCo converter.

    Returns:
        inferencer: `motion_inference` wrapper exposing `.predict(...)`.
        converter: `X2MujocoQposConverter` paired with the same motion_rep
            instance the inferencer was built against.
    """
    paths.validate()
    result_dir = _DEFAULT_RESULT_DIR

    # Pose: build from pose hparams + load weights.
    pose_conf = _load_pose_hparams(paths.pose_version_dir, paths.vqvae_ckpt, result_dir)
    pose_motion_rep = load_motion_rep(pose_conf)
    pose_model = _instantiate_pose_model(pose_conf, pose_motion_rep)
    _load_state_dict_into(pose_model, paths.pose_ckpt)
    pose_model = pose_model.eval().to(device)

    # Root: build from root hparams + load weights. (Uses its own motion_rep,
    # which should be identical structurally to the pose one but with its own
    # stats folder.)
    root_conf = _load_root_hparams(paths.root_version_dir, paths.vqvae_ckpt, result_dir)
    root_motion_rep = load_motion_rep(root_conf)
    root_model = _instantiate_root_model(root_conf, root_motion_rep)
    _load_state_dict_into(root_model, paths.root_ckpt)
    root_model = root_model.eval().to(device)

    # Mirror navigation_demo._initialize_inference_modles: wrap both in
    # motion_inference (uses pose's args dict for max_tokens etc.).
    inferencer = motion_inference(
        {"pose": pose_model, "root": root_model},
        pose_model._args,
        device=device,
    )

    # Build converter against the inferencer's motion_rep so the FK convention
    # (skeleton, motion_to_mujoco_matrix) matches what predict() decodes.
    converter = X2MujocoQposConverter(
        motion_rep=inferencer.motion_rep,
        xml_path=str(paths.mjcf_path) if paths.mjcf_path else None,
    ).to(device)
    return inferencer, converter


def load_x2_planner(
    paths: Optional[X2PlannerPaths] = None,
    device: str = "cuda",
    clip_library_ckpt: Optional[Path] = "auto",
    **kwargs,
) -> NeuralPlannerCore:
    """One-shot helper: load the X2 stack and wrap it in a NeuralPlannerCore.

    Args:
        paths: Optional path override; defaults to ``X2PlannerPaths.default()``
            (pinned step checkpoints in each ``version_1`` dir).
        device: torch device (cuda / cpu).
        clip_library_ckpt: Optional path to a baked pose-template clip
            library (see ``motionbricks/scripts/build_x2_planner_clips.py``).
            * ``"auto"`` (default): use the canonical ``out/X2-clip.ckpt``
              if it exists; otherwise skip clip loading silently. Lets
              the velocity-only path keep working when the library has
              not been baked yet.
            * ``Path``: explicit path; missing file raises FileNotFoundError.
            * ``None``: never load a clip library; only ``replan_with_velocity``
              will be available.
        **kwargs: Forwarded to NeuralPlannerCore (e.g. filter_qpos=True).

    Returns:
        Ready-to-stream NeuralPlannerCore (call .reset(init_qpos) before
        the first .replan_with_velocity(...) or .replan_with_pose_template(...)).
    """
    if paths is None:
        paths = X2PlannerPaths.default()
    inferencer, converter = load_x2_models(paths, device=device)

    resolved_clip_ckpt: Optional[Path]
    if clip_library_ckpt == "auto":
        if _DEFAULT_CLIP_LIBRARY.is_file():
            resolved_clip_ckpt = _DEFAULT_CLIP_LIBRARY
        else:
            resolved_clip_ckpt = None
    elif clip_library_ckpt is None:
        resolved_clip_ckpt = None
    else:
        resolved_clip_ckpt = Path(clip_library_ckpt)
        if not resolved_clip_ckpt.is_file():
            raise FileNotFoundError(
                f"clip_library_ckpt={resolved_clip_ckpt} not found. Run "
                "motionbricks/scripts/build_x2_planner_clips.py first."
            )

    return NeuralPlannerCore(
        inferencer,
        converter,
        device=device,
        clip_library_ckpt=resolved_clip_ckpt,
        **kwargs,
    )
