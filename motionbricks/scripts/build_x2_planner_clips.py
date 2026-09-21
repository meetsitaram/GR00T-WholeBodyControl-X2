"""Bake the X2 mode-template clip library used by the pose-template inference path.

This is the X2 counterpart to the G1 demo's pre-baked ``out/G1-clip.ckpt``
(produced by ``clip_holder_G1._preprocess_clips_from_dataloader`` in
``motionbricks/motion_backbone/demo/clips.py``). It pulls a small set of
canonical X2 PKL clips, runs them through the X2 MuJoCo->motion converter,
and stacks the result into a single ckpt that ``NeuralPlannerCore`` can
load at inference time.

Why this matters
----------------
The SONIC / MotionBricks papers describe inference as autoregressive
motion in-betweening between a context keyframe (past) and a *target
keyframe* (future) where the target is either a spring-derived position
or a sampled pose from a reference clip. The X2 planner has been running
without the latter, masking ``has_local_poses[:, -4:] = False`` and zeroing
out ``target_local_poses``. Filling those in with aligned poses from the
matching mode clip restores the paper's actual contract.

What this script writes
-----------------------
A single torch ``state_dict`` ckpt with stacked per-mode buffers::

    global_root_positions   [num_modes, max_frames, 3]   # motion-rep Y-up, Y=0
    global_joint_positions  [num_modes, max_frames, J, 3] # root-XZ-relative, full Y
    global_joint_rotations  [num_modes, max_frames, J, 3, 3]
    global_headings         [num_modes, max_frames]       # atan2 of forward dir
    mujoco_qpos             [num_modes, max_frames, 38]
    num_frames_per_clip     [num_modes]                   # int32

plus a Python sidecar ``X2-clip.modes.json`` mapping mode name -> integer
index (insertion order is the binding contract).

Default mode set (4 modes), all windows of the demo bank regenerated from the
upstream reference clips (``tools/build_demo_bank_from_upstream.sh`` ->
``gear_sonic/data/motions/x2_demo_bank.pkl``, 50 fps):

    0  idle       squat_001__A359                 [0:30)     standing lead-in, 0.0 m/s
    1  slow_walk  walking_quip_360_R_002__A428    [75:150)   gait onset, ~0.6 m/s
    2  walk       walking_quip_360_R_002__A428    [100:175)  steady straight walk, ~0.8 m/s
    3  run_proxy  walking_quip_360_R_002__A428    [275:350)  steady walk after the turn, ~0.8 m/s
                                                             (no faster gait in the bank)

Per-mode source PKL overrides are supported via ``ModeSpec.source_pkl``;
when None the mode falls back to the script's ``--motion-lib-pkl`` flag.
Point the flag (or ``source_pkl``) at your own X2 motion-lib pkl to bake a
richer template set (a dedicated walk loop with ~0 net yaw drift is the
best "walk" source: heading wobble in the template propagates through every
replan).

Planner checkpoints: ``--planner-dir`` (default ``$SONIC_HOME/x2/kplanner_torch``,
the layout ``download_from_hf.py`` fills: ``{vqvae,pose,root}/x2_kplanner_*.ckpt``
+ ``hparams.yaml`` / ``skeleton/`` / ``stats/`` sidecars); when that dir is
absent the training-tree defaults of ``X2PlannerPaths.default()`` are used.

Run::

    source .venv/bin/activate && \\
      PYTHONPATH="${PWD}/motionbricks:${PWD}" \\
      python motionbricks/scripts/build_x2_planner_clips.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "motionbricks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "motionbricks"))


# ---------------------------------------------------------------------------
# Mode table — the binding contract between the bake script and
# `_X2ClipLibrary` in neural_planner.py. KEEP IN SYNC.
# Insertion order defines the integer mode index.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeSpec:
    name: str
    clip_key: str
    start_frame: int
    end_frame: int
    # avg_root_vel is the spring model's default forward velocity for this
    # mode in m/s. The actual planner uses velocity_intent[2] (vel_z_forward)
    # for the target root xz integration; avg_root_vel is currently advisory
    # metadata. We still bake it so future spring-model upgrades (Phase 2.5
    # in the plan) can read it without rebaking.
    avg_root_vel: float
    # Optional override for the source PKL. When None, the bake script's
    # ``--motion-lib-pkl`` is used. Set this to source a single mode from
    # a different PKL (e.g. the "relaxed_walk" mode comes from
    # ``x2_ultra_relaxed_walk_forward_v1.pkl``).
    source_pkl: Path | None = None


_DEMO_BANK_PKL = REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_demo_bank.pkl"
_WALK_CLIP = "walking_quip_360_R_002__A428"   # 9.1 s walk with a 360 turn in the middle
_IDLE_CLIP = "squat_001__A359"                # standing lead-in before the squat


DEFAULT_MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        name="idle",
        clip_key=_IDLE_CLIP,
        start_frame=0,
        end_frame=30,
        avg_root_vel=0.0,
    ),
    ModeSpec(
        # Gait onset: root speed ramps 0.35 -> 0.78 m/s, heading steady.
        name="slow_walk",
        clip_key=_WALK_CLIP,
        start_frame=75,
        end_frame=150,
        avg_root_vel=0.58,
    ),
    ModeSpec(
        # Steady straight stretch before the turn (~0.83 m/s, yaw within 2 deg).
        # A template with heading wobble propagates it through every replan,
        # so the window is cut to the straight part of the clip.
        name="walk",
        clip_key=_WALK_CLIP,
        start_frame=100,
        end_frame=175,
        avg_root_vel=0.75,
    ),
    ModeSpec(
        # Steady walk after the turn. The bank has no faster gait, so the
        # run proxy is the same speed as walk; replace with a run clip from
        # your own corpus via --motion-lib-pkl / source_pkl.
        name="run_proxy",
        clip_key=_WALK_CLIP,
        start_frame=275,
        end_frame=350,
        avg_root_vel=0.80,
    ),
)


# ---------------------------------------------------------------------------
# G1-STYLE mode table — a faithful port of the G1 demo's clip_holder_G1
# (motionbricks/motion_backbone/demo/clips.py). On G1, idle/slow_walk/walk all
# share the SAME neutral-idle keyframe (30 frames of a standing clip) and
# differ ONLY by avg_root_vel (0.0 / 0.3 / 1.0 m/s, written *2 for the
# spring model's 0.5x factor). The gait is produced by the velocity target
# + backbone, not by a distinct walk clip. Select with ``--modes g1style``.
# run_proxy keeps the same idle keyframe at a higher velocity (G1 has no
# separate run clip; "run" == walk mode at higher target_vel).
# ---------------------------------------------------------------------------
_G1_IDLE_CLIP = _IDLE_CLIP

G1STYLE_MODES: tuple[ModeSpec, ...] = (
    ModeSpec(name="idle", clip_key=_G1_IDLE_CLIP, start_frame=0, end_frame=30, avg_root_vel=0.0),
    ModeSpec(name="slow_walk", clip_key=_G1_IDLE_CLIP, start_frame=0, end_frame=30, avg_root_vel=0.6),
    ModeSpec(name="walk", clip_key=_G1_IDLE_CLIP, start_frame=0, end_frame=30, avg_root_vel=2.0),
    ModeSpec(name="run_proxy", clip_key=_G1_IDLE_CLIP, start_frame=0, end_frame=30, avg_root_vel=3.0),
)

_MODE_TABLES: dict[str, tuple[ModeSpec, ...]] = {
    "default": DEFAULT_MODES,
    "g1style": G1STYLE_MODES,
}


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """PKL stores root_rot as xyzw; mujoco_qpos uses wxyz."""
    return np.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        axis=-1,
    )


def _build_clip_qpos(payload: dict, start: int, end: int) -> np.ndarray:
    """Slice a PKL payload [start:end] into an X2 MuJoCo qpos [T, 38] array."""
    trans = np.asarray(payload["root_trans_offset"], dtype=np.float32)
    rot_xyzw = np.asarray(payload["root_rot"], dtype=np.float32)
    dof = np.asarray(payload["dof"], dtype=np.float32)
    n = trans.shape[0]
    if end > n or start < 0 or end <= start:
        raise ValueError(
            f"frame range [{start},{end}) invalid for clip of length {n}"
        )
    trans = trans[start:end]
    rot_xyzw = rot_xyzw[start:end]
    dof = dof[start:end]
    rot_wxyz = _quat_xyzw_to_wxyz(rot_xyzw)
    qpos = np.concatenate([trans, rot_wxyz, dof], axis=-1)
    return qpos.astype(np.float32)


def _compute_clip_buffers(
    qpos_np: np.ndarray, converter, device: str
) -> dict:
    """Convert a [T, 38] qpos slice to the buffers the clip library needs.

    Mirrors the per-clip block in
    ``clip_holder._preprocess_clips_from_dataloader`` (G1) so the layout
    matches what the inference path in ``full_navigation_agent`` (and our
    upcoming ``_predict_with_pose_template``) expects.
    """
    qpos = torch.from_numpy(qpos_np).to(device)[None]  # [1, T, 38]

    # Motion-rep Y-up joint transforms.
    gjp, gjr = converter.convert_mujoco_qpos_to_motion_transforms(qpos)
    # gjp: [1, T, J, 3], gjr: [1, T, J, 3, 3]
    gjp = gjp[0]  # [T, J, 3]
    gjr = gjr[0]  # [T, J, 3, 3]

    # Root XZ position (Y zeroed; matches G1 bake at clips.py:93-94).
    root_pos = gjp[:, 0, :].clone()  # [T, 3]
    root_pos[:, 1] = 0.0  # zero Y (gravity axis in motion-rep is Y)
    global_root_positions = root_pos  # [T, 3]

    # Joint positions made root-XZ-relative (Y absolute), per G1 clips.py:95-96.
    global_joint_positions = gjp - global_root_positions[:, None, :]  # [T, J, 3]
    global_joint_rotations = gjr  # [T, J, 3, 3]

    # Heading angle: atan2 of root's forward direction in motion-rep.
    # G1 uses root_rot @ [0, 0, 1] = forward axis in motion-rep coords; the
    # heading is atan2(x, z). See clips.py:99-106.
    fwd = torch.matmul(
        gjr[:, 0, :, :], torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1)
    ).view(-1, 3)
    fwd_xz = fwd * torch.tensor([1.0, 0.0, 1.0], device=device).view(1, 3)
    norm = fwd_xz.norm(dim=1, keepdim=True)
    if (norm < 1e-5).any():
        bad = (norm < 1e-5).nonzero(as_tuple=False)
        raise ValueError(
            f"Clip has ill-defined heading at frames {bad.squeeze().tolist()}"
        )
    fwd_xz = fwd_xz / norm
    global_headings = torch.atan2(fwd_xz[:, 0], fwd_xz[:, 2])  # [T]

    return {
        "global_root_positions": global_root_positions.cpu(),
        "global_joint_positions": global_joint_positions.cpu(),
        "global_joint_rotations": global_joint_rotations.cpu(),
        "global_headings": global_headings.cpu(),
        "mujoco_qpos": qpos[0].cpu(),
    }


def _stack_modes(per_mode_buffers: Sequence[dict], num_joints: int) -> dict:
    """Stack per-mode buffers to [num_modes, max_frames, ...] padded tensors."""
    num_modes = len(per_mode_buffers)
    num_frames = [int(b["mujoco_qpos"].shape[0]) for b in per_mode_buffers]
    max_frames = max(num_frames)

    out = {
        "global_root_positions": torch.zeros(num_modes, max_frames, 3),
        "global_joint_positions": torch.zeros(num_modes, max_frames, num_joints, 3),
        "global_joint_rotations": torch.zeros(num_modes, max_frames, num_joints, 3, 3),
        "global_headings": torch.zeros(num_modes, max_frames),
        "mujoco_qpos": torch.zeros(num_modes, max_frames, 38),
        "num_frames_per_clip": torch.zeros(num_modes, dtype=torch.int32),
    }
    for i, b in enumerate(per_mode_buffers):
        n = num_frames[i]
        out["global_root_positions"][i, :n] = b["global_root_positions"]
        out["global_joint_positions"][i, :n] = b["global_joint_positions"]
        out["global_joint_rotations"][i, :n] = b["global_joint_rotations"]
        out["global_headings"][i, :n] = b["global_headings"]
        out["mujoco_qpos"][i, :n] = b["mujoco_qpos"]
        out["num_frames_per_clip"][i] = n
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--motion-lib-pkl",
        type=Path,
        default=_DEMO_BANK_PKL,
        help="Source PKL containing all mode clips (default: the regenerated "
        "demo bank gear_sonic/data/motions/x2_demo_bank.pkl).",
    )
    p.add_argument(
        "--planner-dir",
        type=Path,
        default=Path(os.environ.get("SONIC_HOME", Path.home() / ".cache/sonic")) / "x2" / "kplanner_torch",
        help="Planner checkpoint dir in the SONIC cache layout "
        "({vqvae,pose,root}/x2_kplanner_*.ckpt + sidecars). Falls back to "
        "X2PlannerPaths.default() (training out/ tree) when absent.",
    )
    p.add_argument(
        "--out-ckpt",
        type=Path,
        default=REPO_ROOT / "motionbricks" / "out" / "X2-clip.ckpt",
        help="Output path for the baked clip library ckpt.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--modes",
        choices=sorted(_MODE_TABLES.keys()),
        default="default",
        help="Which mode table to bake: 'default' (X2-curated walk clips) or "
        "'g1style' (faithful G1 port: shared standing keyframe for "
        "idle/slow_walk/walk, gait driven by velocity). Default: default.",
    )
    args = p.parse_args()

    modes = _MODE_TABLES[args.modes]
    print(f"  mode table      = {args.modes} ({len(modes)} modes)")

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print(f"[device] cuda unavailable, falling back to cpu")
        device = "cpu"
    print(f"  device          = {device}")

    # Bootstrap the X2 converter the same way load_x2_planner does. We
    # don't need the pose/root models for baking, only the converter,
    # but loading the whole stack is the simplest path and adds only a
    # few seconds.
    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths,
        load_x2_planner,
    )

    pd = args.planner_dir
    if (pd / "vqvae" / "x2_kplanner_vqvae.ckpt").is_file():
        paths = X2PlannerPaths(
            vqvae_ckpt=pd / "vqvae" / "x2_kplanner_vqvae.ckpt",
            pose_ckpt=pd / "pose" / "x2_kplanner_pose.ckpt",
            root_ckpt=pd / "root" / "x2_kplanner_root.ckpt",
            vqvae_version_dir=pd / "vqvae",
            pose_version_dir=pd / "pose",
            root_version_dir=pd / "root",
        )
        print(f"  planner dir     = {pd}")
    else:
        paths = X2PlannerPaths.default()
    print(f"  vqvae_ckpt      = {paths.vqvae_ckpt}")
    print(f"  pose_ckpt       = {paths.pose_ckpt}")
    print(f"  root_ckpt       = {paths.root_ckpt}")
    planner = load_x2_planner(paths, device=device)
    converter = planner._converter

    # Load each unique source PKL only once and cache them.
    pkl_cache: dict[Path, dict] = {}

    def _load_pkl(path: Path) -> dict:
        path = Path(path).resolve()
        if path not in pkl_cache:
            if not path.is_file():
                raise FileNotFoundError(f"source PKL not found: {path}")
            pkl_cache[path] = joblib.load(path)
        return pkl_cache[path]

    default_pkl = args.motion_lib_pkl

    # Probe the number of joints by running one clip frame through the
    # FIRST mode's source clip (which may or may not be the default PKL).
    probe_mode = modes[0]
    probe_pkl_path = probe_mode.source_pkl or default_pkl
    probe_raw = _load_pkl(probe_pkl_path)
    probe_key = probe_mode.clip_key
    if probe_key not in probe_raw:
        raise KeyError(f"probe clip {probe_key!r} not in {probe_pkl_path}")
    probe_qpos = _build_clip_qpos(probe_raw[probe_key], 0, 4)
    probe_gjp, _ = converter.convert_mujoco_qpos_to_motion_transforms(
        torch.from_numpy(probe_qpos).to(device)[None]
    )
    num_joints = int(probe_gjp.shape[2])
    print(f"  num_joints      = {num_joints}")

    # Process each mode in order, sourcing from per-mode PKL when set.
    per_mode_buffers = []
    per_mode_source_pkls: list[Path] = []
    for mode in modes:
        pkl_path = (mode.source_pkl or default_pkl).resolve()
        raw = _load_pkl(pkl_path)
        if mode.clip_key not in raw:
            raise KeyError(
                f"mode {mode.name!r}: clip {mode.clip_key!r} not in "
                f"{pkl_path}"
            )
        payload = raw[mode.clip_key]
        clip_len = payload["root_trans_offset"].shape[0]
        end = min(mode.end_frame, clip_len)
        qpos_np = _build_clip_qpos(payload, mode.start_frame, end)
        buf = _compute_clip_buffers(qpos_np, converter, device)
        per_mode_buffers.append(buf)
        per_mode_source_pkls.append(pkl_path)
        hr_xz = buf["global_root_positions"][:, [0, 2]]
        net_xz = (hr_xz[-1] - hr_xz[0]).tolist()
        hdg = buf["global_headings"]
        dhdg = float((hdg[-1] - hdg[0]).item())
        # Show ONLY the PKL filename so the per-mode source table fits.
        pkl_short = pkl_path.name
        print(
            f"    {mode.name:10s}  {mode.clip_key:48s}  "
            f"frames=[{mode.start_frame:4d},{end:4d})  "
            f"len={qpos_np.shape[0]:4d}  "
            f"net_xz=({net_xz[0]:+.2f},{net_xz[1]:+.2f})  "
            f"dheading={dhdg:+.3f}rad  src={pkl_short}"
        )

    out = _stack_modes(per_mode_buffers, num_joints)

    # Sidecar JSON capturing the mode-name -> index binding plus per-mode
    # metadata (avg_root_vel, source clip key, frame range). This is what
    # users / scripts read to discover the mode set without unpickling
    # the ckpt.
    sidecar = {
        "modes": [
            {
                "index": i,
                "name": m.name,
                "clip_key": m.clip_key,
                "start_frame": m.start_frame,
                "end_frame": m.end_frame,
                "avg_root_vel": m.avg_root_vel,
                "num_frames": int(out["num_frames_per_clip"][i].item()),
                "source_pkl": str(per_mode_source_pkls[i]),
            }
            for i, m in enumerate(modes)
        ],
        "num_joints": num_joints,
        "mode_table": args.modes,
        "default_source_pkl": str(args.motion_lib_pkl.resolve()),
    }
    sidecar_path = args.out_ckpt.with_suffix(".modes.json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  sidecar         = {sidecar_path}")

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_ckpt)
    print(f"  out_ckpt        = {args.out_ckpt}")
    size_mb = args.out_ckpt.stat().st_size / (1024 * 1024)
    print(f"  size            = {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
