"""Forward-kinematics helpers for X2 wrist trajectories from body_q_mj.

Centralises the small but non-trivial FK code that turns a series of
31-D MuJoCo-ordered body joint poses (``action.body_q_mj`` in the
recorded LeRobot datasets, or the bridge's wire) into pelvis-frame
wrist xyz trajectories.

Pelvis-frame means we hold the root freejoint at the world origin with
identity orientation, so the resulting wrist positions reflect *pure
arm + waist motion*. Base drift / walking is excluded by construction.
This is the right frame for:

  * comparing demonstrations to inference (visually & numerically),
  * phase / cycle segmentation (locomotion noise doesn't leak into
    "is the wrist near the table?"),
  * raw-policy-intent vs. delivered-wire diagnostics on chunk dumps.

Originally lived inside
the recorded-dataset viewer; extracted so other
analysis tools (segmentation pipeline, diagnostic CLIs, future replay
tooling) can reuse the same code path without copy-pasting joint name
tables.

The functions intentionally return ``None`` instead of raising when
MuJoCo or the MJCF can't be loaded, so callers can degrade gracefully
to scalar-only fallbacks (the viewer does this already).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# Default location of the canonical X2 MJCF shipped with the repo. Most
# callers should let the FK helpers pick this up automatically; pass an
# explicit ``mjcf_path`` to override (e.g. when testing a variant
# embodiment file).
DEFAULT_X2_MJCF: Path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "assets"
    / "robot_description"
    / "mjcf"
    / "x2_ultra.xml"
)


# 31 hinge / slide joints written in the exact order used by
# ``action.body_q_mj`` in the X2 LeRobot dataset format (and by the
# bridge's wire). The MJCF's root is a freejoint (qpos[0:7]); these 31
# joints follow it.
X2_BODY_JOINT_NAMES_31: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint",
    "head_pitch_joint",
]


# Body names whose ``data.xpos`` we read after each FK step. These are
# the end-of-chain frames closest to where the operator's hand attaches
# (matches the wrist origin used by the IK retarget too).
DEFAULT_WRIST_BODY_NAMES: tuple[str, str] = (
    "left_wrist_roll_link",
    "right_wrist_roll_link",
)


def compute_wrist_trajectories(
    body_q_mj_series: np.ndarray,
    *,
    mjcf_path: Path | str = DEFAULT_X2_MJCF,
    body_names: Iterable[str] = DEFAULT_WRIST_BODY_NAMES,
    verbose: bool = True,
    log_prefix: str = "[fk]",
) -> Optional[dict[str, np.ndarray]]:
    """Pelvis-frame FK on a series of 31-D body poses.

    Parameters
    ----------
    body_q_mj_series
        ``(N, >=31)`` array of MuJoCo-ordered body joints. Only the
        first 31 columns are used. Dtype is coerced to float64 inside.
    mjcf_path
        MJCF to load. Defaults to ``data/assets/robot_description/mjcf/
        x2_ultra.xml`` shipped with the repo.
    body_names
        Body names to read after each FK step. Bodies missing from the
        MJCF are skipped with a warning rather than raising; if NO body
        survives, returns ``None``.
    verbose
        When True, prints a one-line FK-done summary with per-axis
        ranges for each tracked body.
    log_prefix
        Prefix attached to all info / warning prints (e.g. ``"[view]"``
        for the rerun viewer, ``"[segment]"`` for the segmentation
        pipeline).

    Returns
    -------
    dict[body_name, (N, 3) float64] | None
        Pelvis-frame wrist positions. ``None`` if MuJoCo can't be
        imported, the MJCF can't be loaded, the MJCF lacks any of the
        expected body joints, or no tracked body is present in the MJCF
        — in all these cases a warning is printed so the caller can
        degrade gracefully without retrying.
    """
    try:
        import mujoco
    except ImportError as exc:
        print(
            f"{log_prefix} WARN: FK requested but mujoco import failed: "
            f"{exc}. Re-run install_scripts/install_viewer.sh (picks up "
            "the mujoco pin in requirements-viewer.txt) or skip the FK "
            "step.",
            flush=True,
        )
        return None

    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception as exc:
        print(
            f"{log_prefix} WARN: failed to load MJCF {mjcf_path} for "
            f"FK: {exc}.",
            flush=True,
        )
        return None

    data = mujoco.MjData(model)

    # Build a map joint_name -> qpos index. Free joints occupy 7 slots
    # (xyz + quat); hinge / slide joints occupy 1. The MJCF puts the
    # freejoint first so all 31 body joints start at qpos[7].
    try:
        qpos_indices: list[int] = []
        for jn in X2_BODY_JOINT_NAMES_31:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise KeyError(f"joint {jn!r} not found in {mjcf_path}")
            qpos_indices.append(int(model.jnt_qposadr[jid]))
    except KeyError as exc:
        print(
            f"{log_prefix} WARN: MJCF schema mismatch: {exc}.",
            flush=True,
        )
        return None

    # Hold root at identity so the wrist trajectory reflects pure
    # arm + waist motion (i.e. pelvis-relative). Otherwise base drift
    # / walking would dominate the result.
    qpos0 = np.zeros(model.nq, dtype=np.float64)
    qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

    tracked: dict[str, int] = {}
    for body_name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            print(
                f"{log_prefix} WARN: body {body_name!r} missing from "
                "MJCF; skipping that side.",
                flush=True,
            )
            continue
        tracked[body_name] = int(bid)

    if not tracked:
        return None

    n_frames = int(body_q_mj_series.shape[0])
    out: dict[str, np.ndarray] = {
        body_name: np.empty((n_frames, 3), dtype=np.float64)
        for body_name in tracked
    }
    qpos_idx_arr = np.asarray(qpos_indices, dtype=np.int64)

    for f in range(n_frames):
        np.copyto(data.qpos, qpos0)
        data.qpos[qpos_idx_arr] = body_q_mj_series[f, :31].astype(np.float64)
        mujoco.mj_kinematics(model, data)
        for body_name, bid in tracked.items():
            out[body_name][f] = data.xpos[bid]

    if verbose:
        print(
            f"{log_prefix} FK done: {n_frames} frames -> "
            + ", ".join(
                f"{body_name} (x,y,z range "
                f"{out[body_name].min(axis=0).round(3).tolist()} -> "
                f"{out[body_name].max(axis=0).round(3).tolist()})"
                for body_name in out
            ),
            flush=True,
        )
    return out


def compute_finger_curl(
    hand_joints_series: np.ndarray,
) -> np.ndarray:
    """Per-frame scalar "finger curl" magnitude for the omnihand.

    Computes ``||hand_joints||_2`` over each row. Open-hand poses (joint
    angles near zero) yield small values; closed-hand poses yield large
    values. The threshold between "open" and "closed" is empirically
    around 2.0-2.5 on this dataset (see the chunk-dump diagnostics:
    open-hand chunks read ~1.02, closed-hand chunks ~3.45) but is
    dataset-dependent — callers should compute thresholds from each
    episode's own min/max if precise transition detection matters.

    Parameters
    ----------
    hand_joints_series
        ``(N, 10)`` array of omnihand joint angles
        (``action.left_hand_joints`` or ``action.right_hand_joints``).

    Returns
    -------
    (N,) float64
        Per-frame L2 norm.
    """
    arr = np.asarray(hand_joints_series, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"hand_joints_series must be 2-D (N, D); got shape {arr.shape}"
        )
    return np.linalg.norm(arr, axis=1)


__all__ = [
    "DEFAULT_X2_MJCF",
    "DEFAULT_WRIST_BODY_NAMES",
    "X2_BODY_JOINT_NAMES_31",
    "compute_finger_curl",
    "compute_wrist_trajectories",
]
