"""Per-clip / per-window metrics for X2 planner primitive curation.

Inputs are always slices of a motion-lib clip:

  - ``dof``        : (T, 31) joint positions in MuJoCo order, radians
  - ``root_rot``   : (T, 4)  root orientation, scipy ``xyzw`` order
  - ``root_trans`` : (T, 3)  world-frame root translation, meters

Outputs are :class:`WindowMetrics` records summarising what kind of motion
the window contains. The curator then scores each window against a per-bin
spec from ``x2_planner_bins.yaml`` (see :mod:`gear_sonic.utils.planner.registry`).

We deliberately keep this MuJoCo-free: the curator must run anywhere the
motion-lib PKL can be loaded (laptops, CI, no GPU, no Isaac Lab). For the
"feet planted" / "end at square" judgements we use heuristics on the leg
DOFs themselves rather than full forward kinematics — this is good enough
to rank ~2,500 candidates and the runtime sim validation in D.6 is what
catches actual physics issues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .constants import (
    LEFT_ANKLE_PITCH_IDX,
    LEFT_HIP_PITCH_IDX,
    LEFT_KNEE_IDX,
    LEG_INDICES,
    NUM_BODY_DOFS,
    RIGHT_ANKLE_PITCH_IDX,
    RIGHT_HIP_PITCH_IDX,
    RIGHT_KNEE_IDX,
    WAIST_PITCH_IDX,
    WAIST_ROLL_IDX,
    WAIST_YAW_IDX,
)

# Pre-converted to a list for ndarray fancy indexing.
_LEG_INDICES_LIST: list[int] = list(LEG_INDICES)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowMetrics:
    """All metrics computed for a single (clip, start_frame, n_frames) window.

    Locomotion metrics:
      net_xy_body_m       : (2,) translation in start-frame body frame, meters.
                            x = forward, y = left.
      net_yaw_deg         : signed yaw delta (start -> end), degrees.
      cross_axis_xy_m     : magnitude of translation orthogonal to the bin's
                            intended axis, meters. Set by the bin scorer
                            after it knows the intent direction.
      pelvis_z_min_m      : min root z over the window, meters.
      pelvis_z_max_m      : max root z over the window, meters.
      pelvis_z_mean_m     : mean root z over the window, meters.
      stride_count        : number of full strides (left-right ankle-z zero
                            crossings / 2). 0 for single-step or static
                            primitives.

    End-at-square (foot symmetry at last frame):
      end_at_square_score : in [0, 1]; 1.0 = perfect L/R symmetry of leg DOFs
                            in the body frame. Used as a soft gate.
      end_hip_pitch_diff  : |L_hip_pitch[-1] - R_hip_pitch[-1]|, radians.
      end_knee_diff       : |L_knee[-1] - R_knee[-1]|, radians.
      end_ankle_pitch_diff: |L_ankle_pitch[-1] - R_ankle_pitch[-1]|, radians.

    Static upper-body metrics:
      waist_yaw_apex_deg   : peak |waist_yaw[t] - waist_yaw[0]|, degrees.
      waist_pitch_apex_deg : peak |waist_pitch[t] - waist_pitch[0]|, degrees.
      waist_roll_apex_deg  : peak |waist_roll[t] - waist_roll[0]|, degrees.
      end_at_apex_score    : in [0, 1]; 1.0 = clip ends at the apex pose
                             (so the lean / twist is held, not returned).
      feet_planted_score   : in [0, 1]; 1.0 = leg DOFs barely move over the
                             whole window (hip + knee + ankle range each
                             below the planted threshold).
      max_leg_dof_drift    : max over leg DOFs of (max(t) - min(t)), radians.

    Loop / idle:
      loop_dof_drift     : max |dof[-1] - dof[0]| over all DOFs, radians.
      loop_quat_distance : 1 - |dot(root_rot[-1], root_rot[0])|, unitless.

    Sanity:
      n_frames           : number of frames in the window (= dof.shape[0]).
      fps                : reference fps (passed through; used by the curator
                           for window-length suggestions).
    """

    n_frames: int
    fps: float
    # locomotion
    net_xy_body_m: np.ndarray
    net_yaw_deg: float
    cross_axis_xy_m: float
    pelvis_z_min_m: float
    pelvis_z_max_m: float
    pelvis_z_mean_m: float
    stride_count: int
    # end-at-square (locomotion-bin gate)
    end_at_square_score: float
    end_hip_pitch_diff: float
    end_knee_diff: float
    end_ankle_pitch_diff: float
    # static upper-body
    waist_yaw_apex_deg: float
    waist_pitch_apex_deg: float
    waist_roll_apex_deg: float
    end_at_apex_score: float
    feet_planted_score: float
    max_leg_dof_drift: float
    # loop / idle
    loop_dof_drift: float
    loop_quat_distance: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaw_of_quat_xyzw(quat: np.ndarray) -> float:
    """Yaw (rad) about world z, ``zyx`` Euler convention."""
    return float(Rot.from_quat(quat).as_euler("zyx")[0])


def _unwrap_signed_diff_rad(start_rad: float, end_rad: float) -> float:
    """Return signed yaw delta in (-pi, pi]."""
    diff = end_rad - start_rad
    return float(np.arctan2(np.sin(diff), np.cos(diff)))


def _world_to_body_xy(
    delta_xy_world: np.ndarray, start_yaw_rad: float
) -> np.ndarray:
    """Rotate a world-frame XY delta into the start-frame body frame.

    Body x = robot's forward at frame 0; body y = robot's left.
    """
    c = np.cos(-start_yaw_rad)
    s = np.sin(-start_yaw_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return rot @ delta_xy_world.astype(np.float64)


def _stride_count_from_legs(dof: np.ndarray) -> int:
    """Approximate stride count = #zero crossings of (L_ankle_z - R_ankle_z) / 2.

    We don't have foot-z directly without FK, so we use the proxy
    ``L_hip_pitch + 0.5 * L_knee`` vs the right-side analogue. This isn't
    physical foot height, but it is positively correlated with foot lift in
    a normal stride, which is all we need for "is this one stride or many".
    """
    if dof.shape[0] < 4:
        return 0
    proxy_left = dof[:, LEFT_HIP_PITCH_IDX] + 0.5 * dof[:, LEFT_KNEE_IDX]
    proxy_right = dof[:, RIGHT_HIP_PITCH_IDX] + 0.5 * dof[:, RIGHT_KNEE_IDX]
    diff = proxy_left - proxy_right
    diff_demean = diff - float(np.mean(diff))
    if float(np.std(diff_demean)) < 0.02:
        return 0
    signs = np.sign(diff_demean)
    crossings = int(np.sum((signs[1:] != signs[:-1]) & (signs[1:] != 0)))
    return crossings // 2


def _end_at_square_score(dof_last: np.ndarray) -> tuple[float, float, float, float]:
    """Score how mirror-symmetric the leg DOFs are in the last frame.

    Uses three axes (hip_pitch, knee, ankle_pitch). Score = exp(-max_diff/0.05)
    so a 0.05 rad (~3 deg) asymmetry maps to ~0.37, and 0.10 rad maps to ~0.13.
    """
    hip_diff = float(abs(dof_last[LEFT_HIP_PITCH_IDX] - dof_last[RIGHT_HIP_PITCH_IDX]))
    knee_diff = float(abs(dof_last[LEFT_KNEE_IDX] - dof_last[RIGHT_KNEE_IDX]))
    ankle_diff = float(abs(dof_last[LEFT_ANKLE_PITCH_IDX] - dof_last[RIGHT_ANKLE_PITCH_IDX]))
    max_diff = max(hip_diff, knee_diff, ankle_diff)
    score = float(np.exp(-max_diff / 0.05))
    return score, hip_diff, knee_diff, ankle_diff


def _feet_planted_score(
    dof: np.ndarray,
) -> tuple[float, float]:
    """Score "feet didn't move over the whole window".

    For each leg DOF, compute (max - min) over time (radians). Take the max
    across legs. Score = exp(-max_drift/0.03) so 0.03 rad (~1.7 deg per joint)
    maps to ~0.37, 0.06 rad maps to ~0.13. A clip with hip_yaw drifting by
    0.10 rad gets ~0.04 — clearly not "planted".
    """
    leg_dofs = dof[:, _LEG_INDICES_LIST]
    drift_per_dof = leg_dofs.max(axis=0) - leg_dofs.min(axis=0)
    max_drift = float(drift_per_dof.max())
    score = float(np.exp(-max_drift / 0.03))
    return score, max_drift


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_window_metrics(
    dof: np.ndarray,
    root_rot_xyzw: np.ndarray,
    root_trans: np.ndarray,
    fps: float,
) -> WindowMetrics:
    """Compute every metric for a single ``(start, n_frames)`` slice.

    Args:
      dof: ``(T, 31)`` joint positions in MuJoCo order.
      root_rot_xyzw: ``(T, 4)`` root quaternions in scipy xyzw order.
      root_trans: ``(T, 3)`` world-frame root translation.
      fps: reference fps for downstream window-length suggestions.

    Returns:
      :class:`WindowMetrics`.

    Raises:
      ValueError on shape mismatch or empty input.
    """
    if dof.ndim != 2 or dof.shape[1] != NUM_BODY_DOFS:
        raise ValueError(f"dof must be (T, {NUM_BODY_DOFS}), got {dof.shape}")
    if root_rot_xyzw.shape != (dof.shape[0], 4):
        raise ValueError(
            f"root_rot must be (T, 4) matching dof.shape[0]={dof.shape[0]}, "
            f"got {root_rot_xyzw.shape}"
        )
    if root_trans.shape != (dof.shape[0], 3):
        raise ValueError(
            f"root_trans must be (T, 3), got {root_trans.shape}"
        )
    if dof.shape[0] < 2:
        raise ValueError("need at least 2 frames to compute metrics")

    n_frames = int(dof.shape[0])

    # --- locomotion (XY + yaw)
    yaw0 = _yaw_of_quat_xyzw(root_rot_xyzw[0])
    yawN = _yaw_of_quat_xyzw(root_rot_xyzw[-1])
    net_yaw_rad = _unwrap_signed_diff_rad(yaw0, yawN)
    net_yaw_deg = float(np.degrees(net_yaw_rad))

    delta_xy_world = (root_trans[-1, :2] - root_trans[0, :2]).astype(np.float64)
    net_xy_body = _world_to_body_xy(delta_xy_world, yaw0)

    pelvis_z_min = float(root_trans[:, 2].min())
    pelvis_z_max = float(root_trans[:, 2].max())
    pelvis_z_mean = float(root_trans[:, 2].mean())

    # --- end-at-square
    sq_score, hip_diff, knee_diff, ankle_diff = _end_at_square_score(dof[-1])

    # --- stride count
    stride_count = _stride_count_from_legs(dof)

    # --- waist (static upper-body)
    waist_yaw_drift = dof[:, WAIST_YAW_IDX] - dof[0, WAIST_YAW_IDX]
    waist_pitch_drift = dof[:, WAIST_PITCH_IDX] - dof[0, WAIST_PITCH_IDX]
    waist_roll_drift = dof[:, WAIST_ROLL_IDX] - dof[0, WAIST_ROLL_IDX]

    waist_yaw_apex_deg = float(np.degrees(np.max(np.abs(waist_yaw_drift))))
    waist_pitch_apex_deg = float(np.degrees(np.max(np.abs(waist_pitch_drift))))
    waist_roll_apex_deg = float(np.degrees(np.max(np.abs(waist_roll_drift))))

    # end_at_apex: how close is the last-frame waist drift to the peak drift?
    # Use the dominant axis (whichever has the largest peak).
    apex_axis_amps = np.array(
        [
            np.max(np.abs(waist_yaw_drift)),
            np.max(np.abs(waist_pitch_drift)),
            np.max(np.abs(waist_roll_drift)),
        ]
    )
    last_axis_amps = np.array(
        [
            abs(float(waist_yaw_drift[-1])),
            abs(float(waist_pitch_drift[-1])),
            abs(float(waist_roll_drift[-1])),
        ]
    )
    dominant = int(np.argmax(apex_axis_amps))
    if apex_axis_amps[dominant] < 1e-3:
        # No discernible waist motion; trivially "ends at apex" of the
        # nothing-pose. Score 1.0 makes idle-loop bins happy.
        end_at_apex_score = 1.0
    else:
        ratio = float(last_axis_amps[dominant] / apex_axis_amps[dominant])
        end_at_apex_score = float(np.clip(ratio, 0.0, 1.0))

    # --- feet planted
    feet_score, max_leg_drift = _feet_planted_score(dof)

    # --- loopability (idle bin)
    loop_dof_drift = float(np.max(np.abs(dof[-1] - dof[0])))
    dot = float(abs(np.dot(root_rot_xyzw[-1], root_rot_xyzw[0])))
    loop_quat_distance = float(1.0 - dot)

    return WindowMetrics(
        n_frames=n_frames,
        fps=float(fps),
        net_xy_body_m=net_xy_body.astype(np.float64),
        net_yaw_deg=net_yaw_deg,
        cross_axis_xy_m=0.0,  # filled by bin scorer when intent is known
        pelvis_z_min_m=pelvis_z_min,
        pelvis_z_max_m=pelvis_z_max,
        pelvis_z_mean_m=pelvis_z_mean,
        stride_count=stride_count,
        end_at_square_score=sq_score,
        end_hip_pitch_diff=hip_diff,
        end_knee_diff=knee_diff,
        end_ankle_pitch_diff=ankle_diff,
        waist_yaw_apex_deg=waist_yaw_apex_deg,
        waist_pitch_apex_deg=waist_pitch_apex_deg,
        waist_roll_apex_deg=waist_roll_apex_deg,
        end_at_apex_score=end_at_apex_score,
        feet_planted_score=feet_score,
        max_leg_dof_drift=max_leg_drift,
        loop_dof_drift=loop_dof_drift,
        loop_quat_distance=loop_quat_distance,
    )


__all__ = [
    "WindowMetrics",
    "compute_window_metrics",
]
