"""Forward kinematics for the AgiBot X2 arm chains.

Computes end-effector (wrist) position relative to ``torso_link`` using
the kinematic parameters extracted from
``X2_URDF-v1.3.0/x2_ultra.urdf``. Each arm is a 7-DOF revolute chain:

    shoulder_pitch -> shoulder_roll -> shoulder_yaw -> elbow ->
    wrist_yaw       -> wrist_pitch  -> wrist_roll

Vendored verbatim from
``agitbot-x2-record-and-replay/src/x2_recorder/forward_kinematics.py``
(branch ``quest3-bare-hand-control``). Kept in the gear_sonic tree so
the dataset recorder doesn't have to add a sibling-repo path at
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RevoluteJoint:
    """One revolute joint in a serial chain.

    All coordinates are in the parent link's frame. ``axis`` is a unit
    vector along the joint's rotational axis. ``rpy`` is the standard
    URDF roll/pitch/yaw at zero displacement.
    """

    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


# ── X2 URDF kinematic parameters ──────────────────────────────────────────
#
# Extracted from ``X2_URDF-v1.3.0/x2_ultra.urdf`` (the same URDF the
# C++ deploy uses). Both arms have a slight non-zero RPY at the
# shoulder_pitch joint (roughly ±12 deg roll) coming from the
# pelvis/torso mount geometry; we keep that exactly so wrist positions
# computed from FK match the deploy / sim.

_LEFT_ARM_CHAIN: list[RevoluteJoint] = [
    RevoluteJoint(
        xyz=(0.00329906676354264, 0.143031506663431, 0.241647721005027),
        rpy=(0.209439453777995, 0.0, 0.0),
        axis=(0, 1, 0),
    ),
    RevoluteJoint(
        xyz=(-0.000499991072907324, 0.0495, 0.0),
        rpy=(0.0, 0.0, 0.0),
        axis=(1, 0, 0),
    ),
    RevoluteJoint(
        xyz=(0.00122423638490546, 0.0, -0.121194848262102),
        rpy=(0.0, -0.00597565694965793, 0.0),
        axis=(0, 0, 1),
    ),
    RevoluteJoint(
        xyz=(0.0140000000488655, 0.000199986464082369, -0.0829500000002385),
        rpy=(0.0, 0.00597565694965793, 0.0),
        axis=(0, 1, 0),
    ),
    RevoluteJoint(
        xyz=(-0.0132797235062217, -9.75487744990788e-05, -0.120575783591803),
        rpy=(0.0, 0.0, 0.0),
        axis=(0, 0, 1),
    ),
    RevoluteJoint(
        xyz=(0.000490001227040848, 0.0, -0.0819985359552045),
        rpy=(0.0, -0.00597566028430406, 0.0),
        axis=(0, 1, 0),
    ),
    RevoluteJoint(
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        axis=(1, 0, 0),
    ),
]

_RIGHT_ARM_CHAIN: list[RevoluteJoint] = [
    RevoluteJoint(
        xyz=(0.00330093831251838, -0.143030971450747, 0.241648201390685),
        rpy=(-0.209442812383636, 0.0, 0.0),
        axis=(0, 1, 0),
    ),
    RevoluteJoint(
        xyz=(-0.000499999994070091, -0.0495000000000002, 0.0),
        rpy=(0.0, 0.000154003092926045, 0.0),
        axis=(1, 0, 0),
    ),
    RevoluteJoint(
        xyz=(0.000499999999999765, 0.0, -0.121199999999934),
        rpy=(0.0, 0.0, 0.0),
        axis=(0, 0, 1),
    ),
    RevoluteJoint(
        xyz=(0.0139999999999662, -0.0002000000000143, -0.0829500000000745),
        rpy=(0.0, 0.0, 0.0),
        axis=(0, 1, 0),
    ),
    RevoluteJoint(
        xyz=(-0.0132799999999916, 9.99999999858059e-05, -0.120580000000037),
        rpy=(0.0, 0.0, 0.0),
        axis=(0, 0, 1),
    ),
    RevoluteJoint(
        xyz=(0.000500000000020762, 0.0, -0.0819999999999927),
        rpy=(0.0, 0.0, 0.0),
        axis=(0, 1, 0),
    ),
    RevoluteJoint(
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        axis=(1, 0, 0),
    ),
]


# ── Rotation primitives ───────────────────────────────────────────────────


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float64)


def _rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    ax, ay, az = axis
    if ax:
        return _rot_x(angle * ax)
    if ay:
        return _rot_y(angle * ay)
    if az:
        return _rot_z(angle * az)
    return np.eye(3, dtype=np.float64)


# ── FK ────────────────────────────────────────────────────────────────────


def _chain_fk(chain: list[RevoluteJoint], joint_angles: np.ndarray) -> np.ndarray:
    """Compute end-effector position for one chain. Returns (3,) xyz."""
    T = np.eye(4, dtype=np.float64)
    for joint, q in zip(chain, joint_angles):
        R_fixed = _rpy_to_rotation(*joint.rpy)
        T_fixed = _make_transform(R_fixed, joint.xyz)
        R_joint = _axis_rotation(joint.axis, q)
        T_joint = _make_transform(R_joint, (0, 0, 0))
        T = T @ T_fixed @ T_joint
    return T[:3, 3].astype(np.float32)


def chain_fk_full(
    chain: list[RevoluteJoint], joint_angles: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Run FK and also return per-joint origins + axes in the base frame.

    The per-joint origins / axes are what an analytical Jacobian needs:

        J_lin[:, i] = z_i x (p_e - p_i)
        J_ang[:, i] = z_i

    where ``z_i`` is the joint axis and ``p_i`` is the joint origin
    both in the base (torso_link) frame after applying the chain
    transforms up to (but not through) joint i.

    Returns:
        T_e          -- 4x4 SE(3) transform of the wrist link in the
                        base frame.
        joint_origins -- list of 7 (3,) joint origins in the base frame.
        joint_axes    -- list of 7 (3,) joint axes in the base frame.
    """
    T = np.eye(4, dtype=np.float64)
    joint_origins: list[np.ndarray] = []
    joint_axes: list[np.ndarray] = []
    for joint, q in zip(chain, joint_angles):
        R_fixed = _rpy_to_rotation(*joint.rpy)
        T_fixed = _make_transform(R_fixed, joint.xyz)
        T_after_fixed = T @ T_fixed

        joint_origins.append(T_after_fixed[:3, 3].copy())
        body_axis = np.array(joint.axis, dtype=np.float64)
        joint_axes.append(T_after_fixed[:3, :3] @ body_axis)

        R_joint = _axis_rotation(joint.axis, q)
        T_joint = _make_transform(R_joint, (0, 0, 0))
        T = T_after_fixed @ T_joint
    return T, joint_origins, joint_axes


def left_arm_chain() -> list[RevoluteJoint]:
    """Return the left-arm kinematic chain (7 joints)."""
    return _LEFT_ARM_CHAIN


def right_arm_chain() -> list[RevoluteJoint]:
    """Return the right-arm kinematic chain (7 joints)."""
    return _RIGHT_ARM_CHAIN


def arm_fk(joint_angles, side: str = "left") -> np.ndarray:
    """End-effector xyz in torso_link frame for a single 7-DOF arm.

    Args:
        joint_angles: array-like of 7 joint positions in radians.
        side: ``"left"`` or ``"right"``.
    """
    q = np.asarray(joint_angles, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"Expected 7 joint angles, got shape {q.shape}")
    chain = _LEFT_ARM_CHAIN if side == "left" else _RIGHT_ARM_CHAIN
    return _chain_fk(chain, q)


def arm_fk_pose(joint_angles, side: str = "left") -> np.ndarray:
    """Same as :func:`arm_fk` but returns the full 4x4 SE(3) transform."""
    q = np.asarray(joint_angles, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"Expected 7 joint angles, got shape {q.shape}")
    chain = _LEFT_ARM_CHAIN if side == "left" else _RIGHT_ARM_CHAIN
    T, _, _ = chain_fk_full(chain, q)
    return T


def arms_fk(joint_positions_14) -> tuple[np.ndarray, np.ndarray]:
    """Both wrist xyz from a 14-DOF arm vector ``[left[0:7], right[7:14]]``."""
    q = np.asarray(joint_positions_14, dtype=np.float64)
    if q.shape != (14,):
        raise ValueError(f"Expected 14 joint positions, got shape {q.shape}")
    return arm_fk(q[:7], "left"), arm_fk(q[7:], "right")
