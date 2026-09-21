"""Joint-order and default-pose constants for the X2 heuristic planner.

These are the canonical 31-DOF MuJoCo names + indices for AgiBot X2 Ultra.
They mirror ``MUJOCO_JOINT_NAMES`` in ``gear_sonic/scripts/eval_x2_mujoco.py``
and ``DEFAULT_STAND_POSE_MUJOCO_RAD`` in
``gear_sonic/scripts/vla/mock_vla_publish_stand_token.py``. We re-declare them
here so the planner library can be imported without pulling in MuJoCo, tyro,
or the full eval stack.

If either source-of-truth shifts, the test
``tests/test_x2_planner_curator.py::test_constants_match_eval_x2_mujoco`` and
``tests/test_x2_zmq_vla_smoke.py::test_default_stand_pose_constants_match_cpp_header``
will fail and force a re-sync.
"""

from __future__ import annotations

import numpy as np

NUM_BODY_DOFS: int = 31

MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
)
assert len(MUJOCO_JOINT_NAMES) == NUM_BODY_DOFS

# ---------------------------------------------------------------------------
# Joint-group indices (MuJoCo order)
# ---------------------------------------------------------------------------

LEFT_LEG_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
RIGHT_LEG_INDICES: tuple[int, ...] = (6, 7, 8, 9, 10, 11)
LEG_INDICES: tuple[int, ...] = LEFT_LEG_INDICES + RIGHT_LEG_INDICES

LEFT_HIP_PITCH_IDX: int = 0
LEFT_HIP_ROLL_IDX: int = 1
LEFT_HIP_YAW_IDX: int = 2
RIGHT_HIP_PITCH_IDX: int = 6
RIGHT_HIP_ROLL_IDX: int = 7
RIGHT_HIP_YAW_IDX: int = 8
LEFT_KNEE_IDX: int = 3
RIGHT_KNEE_IDX: int = 9
LEFT_ANKLE_PITCH_IDX: int = 4
LEFT_ANKLE_ROLL_IDX: int = 5
RIGHT_ANKLE_PITCH_IDX: int = 10
RIGHT_ANKLE_ROLL_IDX: int = 11

WAIST_YAW_IDX: int = 12
WAIST_PITCH_IDX: int = 13
WAIST_ROLL_IDX: int = 14
WAIST_INDICES: tuple[int, ...] = (WAIST_YAW_IDX, WAIST_PITCH_IDX, WAIST_ROLL_IDX)

LEFT_ARM_INDICES: tuple[int, ...] = (15, 16, 17, 18, 19, 20, 21)
RIGHT_ARM_INDICES: tuple[int, ...] = (22, 23, 24, 25, 26, 27, 28)
HEAD_INDICES: tuple[int, ...] = (29, 30)

# ---------------------------------------------------------------------------
# Default standing pose (radians, MuJoCo joint order)
# Mirrors ``mock_vla_publish_stand_token.DEFAULT_STAND_POSE_MUJOCO_RAD`` and
# ``policy_parameters.hpp::default_angles`` in the C++ deploy.
# ---------------------------------------------------------------------------

DEFAULT_STAND_POSE_MUJOCO_RAD: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # right leg
    0.0, 0.0, 0.0,                                # waist
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,           # left arm
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,          # right arm
    0.0, 0.0,                                     # head
)
assert len(DEFAULT_STAND_POSE_MUJOCO_RAD) == NUM_BODY_DOFS

DEFAULT_STAND_POSE_NP: np.ndarray = np.asarray(
    DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32
)

# Identity quaternion (xyzw order; matches the ``root_rot`` convention in the
# motion-lib PKL schema).
IDENTITY_ROOT_QUAT_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

# Reference body height used when no clip provides an explicit Z offset.
DEFAULT_PELVIS_Z_M: float = 0.788740


__all__ = [
    "NUM_BODY_DOFS",
    "MUJOCO_JOINT_NAMES",
    "LEFT_LEG_INDICES",
    "RIGHT_LEG_INDICES",
    "LEG_INDICES",
    "LEFT_HIP_PITCH_IDX",
    "LEFT_HIP_ROLL_IDX",
    "LEFT_HIP_YAW_IDX",
    "RIGHT_HIP_PITCH_IDX",
    "RIGHT_HIP_ROLL_IDX",
    "RIGHT_HIP_YAW_IDX",
    "LEFT_KNEE_IDX",
    "RIGHT_KNEE_IDX",
    "LEFT_ANKLE_PITCH_IDX",
    "LEFT_ANKLE_ROLL_IDX",
    "RIGHT_ANKLE_PITCH_IDX",
    "RIGHT_ANKLE_ROLL_IDX",
    "WAIST_YAW_IDX",
    "WAIST_PITCH_IDX",
    "WAIST_ROLL_IDX",
    "WAIST_INDICES",
    "LEFT_ARM_INDICES",
    "RIGHT_ARM_INDICES",
    "HEAD_INDICES",
    "DEFAULT_STAND_POSE_MUJOCO_RAD",
    "DEFAULT_STAND_POSE_NP",
    "IDENTITY_ROOT_QUAT_XYZW",
    "DEFAULT_PELVIS_Z_M",
]
