"""VR-driven arm teleoperation for the X2 dataset recorder (legacy).

.. deprecated::
   This engage-anchor implementation has been superseded by
   :mod:`gear_sonic.utils.teleop.vr_arm_teleop_v2`, which uses an
   offline 3-pose calibration to compute a stateless head-relative
   wrist mapping. The engage-anchor approach below couples retargeting
   to the play-area frame and to controller tracking quality at
   A-press time, which produced the "hands behind the body" symptom
   when the operator turned in place. New code should use
   :class:`vr_arm_teleop_v2.VRArmTeleopCalibrated`. This module is
   retained for tests + reproducing pre-calibration recordings.

Closed-loop pipeline that turns Quest 3 ``(3, 7)`` 3-point pose data
(left wrist, right wrist, head, all in robot frame) into 7-DOF arm
joint targets per side via the vendored DLS IK solver.

The retargeting is intentionally simple compared to the full
``decoupled_wbc.body_ik_solver`` path: it's a per-side pose tracker
with engage-time calibration. The X2 stays in the gantry profile so
we don't need head/waist tracking; only arm motion is driven.

Engagement model
----------------

* The operator stands in their preferred neutral pose and presses
  the engage button (``A`` on Quest 3 in our wrapper).
* :meth:`VRArmTeleop.engage` snapshots both wrist poses at that
  instant. Those become the *anchor*: any later wrist pose is taken
  *relative to the anchor*, then composed with a corresponding robot
  anchor (the wrist FK at the X2 stand pose).
* IK is run with the previous joint vector as the seed, so the arm
  smoothly tracks operator deltas without ever jumping.

Coordinate frames
-----------------

Quest 3 reports poses already in the robot frame thanks to
``transform_pose_to_robot`` in :mod:`vr.quest3_reader`. All vectors
in this module are therefore in the robot frame:

    +X = forward, +Y = left, +Z = up, quaternion = [w, x, y, z].

The X2 arm FK lives in the ``torso_link`` frame (pelvis-centric);
since the gantry profile keeps the pelvis upright we treat operator
+X / +Y / +Z as identical to torso_link's. If we ever drive a
moving torso this is where the world->torso transform would apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.solver.arm import ArmIKSolver, IKResult


# ── X2 stand-pose neutral arm angles (rad) ────────────────────────────────
#
# Same numbers used by the live VLA bridge / mock publisher. Layout is
# left[7] then right[7] in the canonical X2 arm joint order:
#   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#   wrist_yaw, wrist_pitch, wrist_roll
DEFAULT_LEFT_ARM_NEUTRAL_RAD: tuple[float, ...] = (
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
)
DEFAULT_RIGHT_ARM_NEUTRAL_RAD: tuple[float, ...] = (
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
)


@dataclass
class _ArmAnchor:
    """Per-side calibration anchor captured on engage."""

    op_pos: np.ndarray  # (3,) operator wrist position in robot frame
    op_rot_inv: sRot    # inverse of operator wrist orientation at engage
    robot_pos: np.ndarray  # (3,) robot wrist position from FK at neutral
    robot_rot: sRot        # robot wrist rotation from FK at neutral


@dataclass
class TeleopTickResult:
    """Returned by :meth:`VRArmTeleop.step`."""

    left_q: np.ndarray
    right_q: np.ndarray
    left_target_pos: np.ndarray
    right_target_pos: np.ndarray
    left_ik: IKResult
    right_ik: IKResult
    engaged: bool


class VRArmTeleop:
    """Quest 3 3-pt pose -> X2 dual-arm joint target stream.

    Args:
        left_neutral_q: 7-DOF left-arm angles at engage time. Defaults
            to the trained X2 stand pose; pass measured ``body_q`` if
            you want to re-anchor against the live deploy state.
        right_neutral_q: same for the right arm.
        damping: DLS damping (forwarded to :class:`ArmIKSolver`).
        rotation_weight: orientation weight for IK. 0 disables
            orientation tracking (position-only). v0 default 0.5
            gives smoother behaviour near the elbow-singular pose.
        per_tick_step_rad: per-joint upper bound on a single DLS step.
        position_scale: multiplicative gain on the operator delta. <1
            shrinks operator motions (helpful for first-time users);
            1.0 = 1:1 mapping which is the v0 default.
        recenter_on_engage: when True (default), :meth:`engage` resets
            the IK seed back to the neutral q so a fresh engagement
            doesn't carry over arm drift from a previous session.

    Public surface:
        :meth:`engage` — capture wrist anchors from a fresh 3-pt pose.
        :meth:`disengage` — drop anchors; subsequent :meth:`step`
            calls hold the last commanded q.
        :meth:`step` — run one DLS tick and return new arm targets.
        :attr:`is_engaged` — True iff anchors are present.
    """

    def __init__(
        self,
        *,
        left_neutral_q: np.ndarray | None = None,
        right_neutral_q: np.ndarray | None = None,
        damping: float = 0.08,
        rotation_weight: float = 0.5,
        per_tick_step_rad: float = 0.30,
        position_scale: float = 1.0,
        recenter_on_engage: bool = True,
    ) -> None:
        if left_neutral_q is None:
            left_neutral_q = np.asarray(DEFAULT_LEFT_ARM_NEUTRAL_RAD, dtype=np.float64)
        if right_neutral_q is None:
            right_neutral_q = np.asarray(DEFAULT_RIGHT_ARM_NEUTRAL_RAD, dtype=np.float64)

        if left_neutral_q.shape != (7,) or right_neutral_q.shape != (7,):
            raise ValueError(
                "neutral_q must be (7,) per side; got "
                f"left={left_neutral_q.shape} right={right_neutral_q.shape}"
            )

        self._left_solver = ArmIKSolver(
            side="left",
            damping=damping,
            rotation_weight=rotation_weight,
            max_per_tick_step_rad=per_tick_step_rad,
        )
        self._right_solver = ArmIKSolver(
            side="right",
            damping=damping,
            rotation_weight=rotation_weight,
            max_per_tick_step_rad=per_tick_step_rad,
        )
        self._left_neutral_q = left_neutral_q.copy()
        self._right_neutral_q = right_neutral_q.copy()
        self._left_q = left_neutral_q.copy()
        self._right_q = right_neutral_q.copy()
        self._position_scale = float(position_scale)
        self._recenter_on_engage = bool(recenter_on_engage)

        self._left_anchor: _ArmAnchor | None = None
        self._right_anchor: _ArmAnchor | None = None

    # -- properties -----------------------------------------------------------

    @property
    def is_engaged(self) -> bool:
        return self._left_anchor is not None and self._right_anchor is not None

    @property
    def left_q(self) -> np.ndarray:
        return self._left_q.copy()

    @property
    def right_q(self) -> np.ndarray:
        return self._right_q.copy()

    @property
    def left_neutral_q(self) -> np.ndarray:
        return self._left_neutral_q.copy()

    @property
    def right_neutral_q(self) -> np.ndarray:
        return self._right_neutral_q.copy()

    # -- lifecycle ------------------------------------------------------------

    def engage(self, vr_3pt_pose: np.ndarray) -> None:
        """Capture operator wrist anchors against the X2 neutral FK pose.

        Args:
            vr_3pt_pose: ``(3, 7)`` array
                ``[lwrist, rwrist, head] = [pos_xyz, quat_wxyz]`` in
                the robot frame (the same payload that
                ``Quest3Reader.get_3pt_pose`` returns).
        """
        vr = np.asarray(vr_3pt_pose, dtype=np.float64)
        if vr.shape != (3, 7):
            raise ValueError(f"vr_3pt_pose must be (3, 7); got {vr.shape}")

        if self._recenter_on_engage:
            self._left_q = self._left_neutral_q.copy()
            self._right_q = self._right_neutral_q.copy()

        l_robot_pos, l_robot_rot_mat = self._left_solver.fk(self._left_q)
        r_robot_pos, r_robot_rot_mat = self._right_solver.fk(self._right_q)

        l_op_pos = vr[0, :3].copy()
        l_op_rot = sRot.from_quat(vr[0, 3:], scalar_first=True)
        r_op_pos = vr[1, :3].copy()
        r_op_rot = sRot.from_quat(vr[1, 3:], scalar_first=True)

        self._left_anchor = _ArmAnchor(
            op_pos=l_op_pos,
            op_rot_inv=l_op_rot.inv(),
            robot_pos=l_robot_pos,
            robot_rot=sRot.from_matrix(l_robot_rot_mat),
        )
        self._right_anchor = _ArmAnchor(
            op_pos=r_op_pos,
            op_rot_inv=r_op_rot.inv(),
            robot_pos=r_robot_pos,
            robot_rot=sRot.from_matrix(r_robot_rot_mat),
        )

    def disengage(self) -> None:
        self._left_anchor = None
        self._right_anchor = None

    # -- per-tick step --------------------------------------------------------

    def step(self, vr_3pt_pose: np.ndarray) -> TeleopTickResult:
        """One DLS tick. Idempotent if ``vr_3pt_pose`` doesn't change.

        When not engaged, returns the last commanded q untouched (and
        reports zero IK residual + engaged=False).
        """
        if not self.is_engaged:
            zero_pos = np.zeros(3, dtype=np.float64)
            return TeleopTickResult(
                left_q=self._left_q.copy(),
                right_q=self._right_q.copy(),
                left_target_pos=zero_pos.copy(),
                right_target_pos=zero_pos.copy(),
                left_ik=IKResult(0.0, 0.0, 0, 0),
                right_ik=IKResult(0.0, 0.0, 0, 0),
                engaged=False,
            )

        vr = np.asarray(vr_3pt_pose, dtype=np.float64)
        if vr.shape != (3, 7):
            raise ValueError(f"vr_3pt_pose must be (3, 7); got {vr.shape}")

        l_target_pos, l_target_quat = self._compose_target(
            self._left_anchor, op_pos=vr[0, :3], op_quat_wxyz=vr[0, 3:]
        )
        r_target_pos, r_target_quat = self._compose_target(
            self._right_anchor, op_pos=vr[1, :3], op_quat_wxyz=vr[1, 3:]
        )

        self._left_q, l_info = self._left_solver.solve(
            q_seed=self._left_q,
            target_pos=l_target_pos,
            target_quat_wxyz=l_target_quat,
            max_iters=1,
        )
        self._right_q, r_info = self._right_solver.solve(
            q_seed=self._right_q,
            target_pos=r_target_pos,
            target_quat_wxyz=r_target_quat,
            max_iters=1,
        )

        return TeleopTickResult(
            left_q=self._left_q.copy(),
            right_q=self._right_q.copy(),
            left_target_pos=l_target_pos,
            right_target_pos=r_target_pos,
            left_ik=l_info,
            right_ik=r_info,
            engaged=True,
        )

    # -- internals ------------------------------------------------------------

    def _compose_target(
        self,
        anchor: _ArmAnchor,
        *,
        op_pos: np.ndarray,
        op_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(target_pos, target_quat_wxyz)`` in torso_link frame."""
        delta_pos = (op_pos - anchor.op_pos) * self._position_scale
        target_pos = anchor.robot_pos + delta_pos

        op_rot = sRot.from_quat(op_quat_wxyz, scalar_first=True)
        delta_rot = anchor.op_rot_inv * op_rot
        target_rot = anchor.robot_rot * delta_rot
        target_quat = target_rot.as_quat(scalar_first=True)
        return target_pos, target_quat


__all__ = [
    "DEFAULT_LEFT_ARM_NEUTRAL_RAD",
    "DEFAULT_RIGHT_ARM_NEUTRAL_RAD",
    "TeleopTickResult",
    "VRArmTeleop",
]
