"""X2 arm inverse kinematics for VR teleop.

Pure-numpy DLS (damped least squares) IK + analytical FK. No
pinocchio / qpsolvers dependency: the X2 arm is a simple 7-DOF
revolute chain whose Jacobian we form analytically from the URDF in
``forward_kinematics``. The intended use is the VR-driven dataset
recorder ``gear_sonic.scripts.record_x2_dataset`` -- one DLS step per
50 Hz tick per arm, with the previous q as the seed.

Vendored from the sister repo ``agitbot-x2-record-and-replay`` (branch
``quest3-bare-hand-control``). Source-of-truth URDF in both repos is
``X2_URDF-v1.3.0/x2_ultra.urdf``.
"""

from gear_sonic.utils.teleop.solver.arm.x2_arm_fk import (
    chain_fk_full,
    arm_fk,
    arm_fk_pose,
    arms_fk,
    left_arm_chain,
    right_arm_chain,
    RevoluteJoint,
)
from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import (
    ArmIKSolver,
    IKResult,
    X2_ARM_JOINT_LIMITS_RAD,
    X2_ARM_JOINT_NAMES,
)

__all__ = [
    "ArmIKSolver",
    "IKResult",
    "RevoluteJoint",
    "X2_ARM_JOINT_LIMITS_RAD",
    "X2_ARM_JOINT_NAMES",
    "arm_fk",
    "arm_fk_pose",
    "arms_fk",
    "chain_fk_full",
    "left_arm_chain",
    "right_arm_chain",
]
