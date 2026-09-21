"""Yaw-cylinder alignment + SLERP/LERP blend helpers for primitive stitching.

Lifted (with light renaming) from
the playlist stitcher (not shipped) so the curator and the runtime
planner share exactly the same math the existing playlist tooling uses for
seam blending. If you change one place, change both — or import from here
in the playlist module.

All quaternions are scipy ``xyzw`` order. All angles in radians.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def yaw_of_quat_xyzw(quat_xyzw: np.ndarray) -> float:
    """Yaw (rad) about world z. ``zyx`` Euler convention. Returns (-pi, pi]."""
    return float(Rot.from_quat(quat_xyzw).as_euler("zyx")[0])


def rotate_quats_yaw_only(
    quats_xyzw: np.ndarray, dyaw: float
) -> np.ndarray:
    """Left-multiply quaternions by ``Rz(dyaw)``. Preserves roll / pitch."""
    rz = Rot.from_euler("z", dyaw)
    return (rz * Rot.from_quat(quats_xyzw)).as_quat()


def slerp_quats(
    q_start_xyzw: np.ndarray, q_end_xyzw: np.ndarray, n: int
) -> np.ndarray:
    """Inclusive SLERP from ``q_start`` to ``q_end`` over ``n`` frames.

    Returns ``(n, 4)`` xyzw. ``n`` must be >= 2.
    """
    if n < 2:
        raise ValueError(f"slerp_quats needs n>=2, got {n}")
    times = np.array([0.0, 1.0])
    rots = Rot.from_quat(np.stack([q_start_xyzw, q_end_xyzw]))
    slerp = Slerp(times, rots)
    samples = np.linspace(0.0, 1.0, n)
    return slerp(samples).as_quat()


def lerp_dof(
    dof_start: np.ndarray, dof_end: np.ndarray, n: int
) -> np.ndarray:
    """Inclusive linear interpolation between two DOF vectors over ``n`` frames."""
    if n < 2:
        raise ValueError(f"lerp_dof needs n>=2, got {n}")
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return (1.0 - t) * dof_start[None, :] + t * dof_end[None, :]


def yaw_align_segment(
    dof: np.ndarray,
    root_rot_xyzw: np.ndarray,
    root_trans: np.ndarray,
    xy_world: np.ndarray,
    yaw_world: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Yaw-only rigid re-alignment so frame 0 lands at ``(xy_world, yaw_world)``.

    XY: rotate the segment's XY trajectory (relative to its own frame 0) by
    the heading delta, then translate to ``xy_world``.
    Z: untouched (preserves natural pelvis bob).
    Rotation: left-multiply every frame's quaternion by ``Rz(dyaw)``.

    Args:
      dof: ``(T, 31)`` joint positions (returned unchanged; passed through
           for symmetry with the stitcher segment dict).
      root_rot_xyzw: ``(T, 4)`` xyzw quaternions.
      root_trans: ``(T, 3)`` world-frame root translation.
      xy_world: ``(2,)`` target XY of frame 0 in world coordinates.
      yaw_world: target yaw (rad) at frame 0.

    Returns:
      ``(dof, new_root_rot_xyzw, new_root_trans)``. New arrays; inputs are
      not mutated. ``dof`` is returned by reference (no need to copy joints).
    """
    yaw0 = yaw_of_quat_xyzw(root_rot_xyzw[0])
    dyaw = yaw_world - yaw0
    rz = Rot.from_euler("z", dyaw)

    rel = root_trans - root_trans[0]
    rel_rot = rz.apply(rel)

    new_trans = np.empty_like(root_trans)
    new_trans[:, 0] = rel_rot[:, 0] + xy_world[0]
    new_trans[:, 1] = rel_rot[:, 1] + xy_world[1]
    # Rz only rotates about z so rel_rot[:, 2] == rel[:, 2]; preserve clip Z.
    new_trans[:, 2] = rel[:, 2] + root_trans[0, 2]

    new_rot = rotate_quats_yaw_only(root_rot_xyzw, dyaw)

    return dof, new_rot, new_trans


def build_blend_window(
    dof_start: np.ndarray,
    rot_start_xyzw: np.ndarray,
    trans_start_xyz: np.ndarray,
    dof_end: np.ndarray,
    rot_end_xyzw: np.ndarray,
    trans_end_xyz: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize an inclusive blend between two ``(dof, root_rot, root_trans)`` endpoints.

    XY is held at the start handover (no foot-skate during the blend). Z is
    LERP'd between the endpoints. Joints LERP, root rotation SLERPs.
    """
    if n_frames < 2:
        raise ValueError(f"build_blend_window needs n_frames>=2, got {n_frames}")

    dof_blend = lerp_dof(dof_start, dof_end, n_frames)
    rot_blend = slerp_quats(rot_start_xyzw, rot_end_xyzw, n_frames)
    trans_blend = np.empty((n_frames, 3), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_frames)
    trans_blend[:, 0] = trans_start_xyz[0]
    trans_blend[:, 1] = trans_start_xyz[1]
    trans_blend[:, 2] = (1.0 - t) * trans_start_xyz[2] + t * trans_end_xyz[2]
    return dof_blend, rot_blend, trans_blend


def resample_motion_30_to_50hz(
    dof: np.ndarray,
    rot_xyzw: np.ndarray,
    trans: np.ndarray,
    src_fps: float,
    target_fps: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear / SLERP resample from ``src_fps`` to ``target_fps`` (50 by default).

    Uses the same math as ``LocalMotionPlannerBase::ResampleGeneratedSequence50Hz``
    in the C++ G1 deploy: linear on positions, SLERP on root quat. Output
    length is ``floor(input_seconds * target_fps)``.
    """
    if dof.shape[0] < 2:
        raise ValueError("need at least 2 frames to resample")
    if src_fps <= 0 or target_fps <= 0:
        raise ValueError("fps must be positive")

    duration_s = dof.shape[0] / src_fps
    n_out = int(np.floor(duration_s * target_fps))
    if n_out < 2:
        return dof.copy(), rot_xyzw.copy(), trans.copy()

    # Output sample times in source frame index space.
    out_times = np.arange(n_out) / target_fps  # seconds
    src_idx_f = out_times * src_fps  # fractional source frame indices
    src_idx_f = np.clip(src_idx_f, 0.0, dof.shape[0] - 1 - 1e-9)
    src_lo = np.floor(src_idx_f).astype(np.int64)
    src_hi = np.minimum(src_lo + 1, dof.shape[0] - 1)
    w_hi = src_idx_f - src_lo
    w_lo = 1.0 - w_hi

    out_dof = (
        w_lo[:, None] * dof[src_lo] + w_hi[:, None] * dof[src_hi]
    ).astype(dof.dtype)
    out_trans = (
        w_lo[:, None] * trans[src_lo] + w_hi[:, None] * trans[src_hi]
    ).astype(trans.dtype)

    # SLERP root quaternion. Build a per-segment SLERP and sample.
    times = np.arange(rot_xyzw.shape[0], dtype=np.float64)
    rots = Rot.from_quat(rot_xyzw)
    slerp = Slerp(times, rots)
    out_rot = slerp(src_idx_f).as_quat().astype(rot_xyzw.dtype)

    return out_dof, out_rot, out_trans


__all__ = [
    "build_blend_window",
    "lerp_dof",
    "resample_motion_30_to_50hz",
    "rotate_quats_yaw_only",
    "slerp_quats",
    "yaw_align_segment",
    "yaw_of_quat_xyzw",
]
