"""Load G1 MotionBricks reference checkpoints into a ready-to-stream NeuralPlannerCore.

Mirrors ``load_x2_planner.py`` but targets the NVIDIA-shipped G1 reference
checkpoints under ``motionbricks/out/motionbricks_{vqvae,pose,root}/``.
The model architecture is identical to the X2 stack; only the
checkpoint paths, skeleton, stats, and MuJoCo converter differ.

This loader exists so per-model diagnostic scripts (test_root_isolated,
test_e2e_velocity_tracking, etc.) can swap ``--ckpt-set x2`` vs
``--ckpt-set g1`` and run the same metric pipeline against the
known-good NVIDIA reference. The G1 numbers become the parity template
against which X2 dimensionless metrics are judged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from motionbricks.helper.mujoco_helper import mujoco_qpos_converter
from motionbricks.helper.pl_util import load_motion_rep
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.motion_backbone.inference.neural_planner import NeuralPlannerCore


# Default G1 result-dir layout. NVIDIA's checkpoints land here after
# `git lfs pull --include="motionbricks/out/**"`.
_DEFAULT_RESULT_DIR = Path(__file__).resolve().parents[3] / "out"
_DEFAULT_VQVAE_VERSION = _DEFAULT_RESULT_DIR / "motionbricks_vqvae/version_1"
_DEFAULT_POSE_VERSION = _DEFAULT_RESULT_DIR / "motionbricks_pose/version_1"
_DEFAULT_ROOT_VERSION = _DEFAULT_RESULT_DIR / "motionbricks_root/version_1"

# G1 MJCF for the mujoco converter. The XML defines the joint hierarchy
# the converter walks; meshes are not required for our diagnostic
# scripts (we never render). Use ``g1.xml`` (the demo default) -- it
# represents the root as ``<freejoint>``, which the converter skips
# correctly. ``g1_29dof.xml`` represents the root as ``<joint
# type="free">`` and trips a name-lookup in ``_prepare_transforms``.
_DEFAULT_G1_MJCF = (
    Path(__file__).resolve().parents[3]
    / "assets/skeletons/g1/g1.xml"
)


@dataclass
class G1PlannerPaths:
    """Paths to the three G1 reference checkpoints + asset folders."""

    vqvae_ckpt: Path
    pose_ckpt: Path
    root_ckpt: Path
    vqvae_version_dir: Path = field(default_factory=lambda: _DEFAULT_VQVAE_VERSION)
    pose_version_dir: Path = field(default_factory=lambda: _DEFAULT_POSE_VERSION)
    root_version_dir: Path = field(default_factory=lambda: _DEFAULT_ROOT_VERSION)
    mjcf_path: Optional[Path] = None

    @classmethod
    def default(cls) -> "G1PlannerPaths":
        """Default paths: NVIDIA's pinned step-2M checkpoints."""
        ckpt_name = "model-step=2000000.ckpt"
        return cls(
            vqvae_ckpt=_DEFAULT_VQVAE_VERSION / "checkpoints" / ckpt_name,
            pose_ckpt=_DEFAULT_POSE_VERSION / "checkpoints" / ckpt_name,
            root_ckpt=_DEFAULT_ROOT_VERSION / "checkpoints" / ckpt_name,
            mjcf_path=_DEFAULT_G1_MJCF,
        )

    def validate(self) -> None:
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
                "G1PlannerPaths missing: " + ", ".join(missing) +
                ". Did you run `git lfs pull --include=\"motionbricks/out/**\"`?"
            )


def _patch_hparams(
    version_dir: Path, vqvae_ckpt: Path, result_dir: Path, *, stage: str
):
    """G1 counterpart to load_x2_planner._patch_hparams.

    The G1 hparams.yaml may already reference the on-disk skeleton /
    stats dirs correctly, but for safety we force them to the
    version_dir's subfolders just like the X2 path does.
    """
    hparams_path = version_dir / "hparams.yaml"
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Missing {hparams_path}")
    conf = OmegaConf.load(hparams_path)
    with open_dict(conf):
        conf.data = {"folder": str(version_dir), "text_embeddings": None}
        conf.skeleton.folder = str(version_dir / "skeleton")
        conf.motion_rep.stats.folder = str(version_dir / "stats" / "motion")
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
        conf.id = "g1"
        conf.run_dir = "."
        conf.out_dir = str(result_dir / f"motionbricks_{stage}")
        conf.wandb_run = None
    return OmegaConf.create(OmegaConf.to_container(conf, resolve=True))


def _instantiate_pose_model(conf, motion_rep):
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
    truly_missing = [
        k for k in missing
        if not (k.startswith("pose_net.") or k.startswith("root_net."))
    ]
    if truly_missing:
        raise RuntimeError(
            f"Loading {ckpt_path}: {len(truly_missing)} unexpected missing key(s), "
            f"first 5: {truly_missing[:5]}"
        )


def load_g1_models(
    paths: G1PlannerPaths,
    device: str = "cuda",
) -> tuple[motion_inference, mujoco_qpos_converter]:
    paths.validate()
    result_dir = _DEFAULT_RESULT_DIR

    pose_conf = _patch_hparams(paths.pose_version_dir, paths.vqvae_ckpt, result_dir, stage="pose")
    pose_motion_rep = load_motion_rep(pose_conf)
    pose_model = _instantiate_pose_model(pose_conf, pose_motion_rep)
    _load_state_dict_into(pose_model, paths.pose_ckpt)
    pose_model = pose_model.eval().to(device)

    root_conf = _patch_hparams(paths.root_version_dir, paths.vqvae_ckpt, result_dir, stage="root")
    root_motion_rep = load_motion_rep(root_conf)
    root_model = _instantiate_root_model(root_conf, root_motion_rep)
    _load_state_dict_into(root_model, paths.root_ckpt)
    root_model = root_model.eval().to(device)

    inferencer = motion_inference(
        {"pose": pose_model, "root": root_model},
        pose_model._args,
        device=device,
    )

    mjcf = paths.mjcf_path if paths.mjcf_path else _DEFAULT_G1_MJCF
    converter = mujoco_qpos_converter(
        motion_rep=inferencer.motion_rep,
        xml_path=str(mjcf),
    ).to(device)
    return inferencer, converter


def load_g1_planner(
    paths: Optional[G1PlannerPaths] = None,
    device: str = "cuda",
    **kwargs,
) -> NeuralPlannerCore:
    """One-shot helper: load the G1 stack and wrap it in a NeuralPlannerCore."""
    if paths is None:
        paths = G1PlannerPaths.default()
    inferencer, converter = load_g1_models(paths, device=device)
    return NeuralPlannerCore(inferencer, converter, device=device, **kwargs)
