"""Stateless head-relative VR arm teleop using OperatorCalibration.

The original :mod:`gear_sonic.utils.teleop.vr_arm_teleop` couples
operator-to-robot retargeting to two pieces of session state:

1. The wrist positions captured at A-press time (``_ArmAnchor``).
2. The play-area / world frame the WebXR session started in.

That coupling produces the "hands behind the body" symptom whenever the
operator turns their body 90 deg in place: the wrist sweeps a 60+ cm arc
in world frame while the engage-anchor IK sees that arc as a "delta"
and walks the robot wrist target around the side of the torso. The X2
waist is locked in v0, so the arm clamps and the wrist ends up behind
the back. See ``data/lerobot/x2_quest3_kinematic_v0`` analysis (2026-05-09).

The fix in this module is to map every tick's wrist position through a
fixed, anatomy-aware affine map computed offline from a 3-pose
calibration:

* ``op_wrist_in_head_yaw = R_yaw_inv @ (wrist_robot_frame - head_robot_frame)``
* ``robot_wrist_target = calibration.apply_to_wrist(op_wrist_in_head_yaw, side)``

There is no engage moment and no anchor: A toggles "active" (do the IK
solve) vs "idle" (hold last commanded q). The mapping is identical
whether the operator just started, has been driving for 10 minutes, or
has rotated 360 degrees in place.

Rotation tracking is intentionally optional. v0 caller passes
``rotation_weight=0`` so we do position-only IK; this keeps things
simple while we validate that the calibrated head-yaw mapping fixes the
position symptom. Wrist-orientation calibration can be layered on later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gear_sonic.utils.teleop.operator_calibration import (
    OperatorCalibration,
    _quat_multiply_wxyz,
    head_yaw_from_quat,
    wrist_quat_to_head_yaw_frame,
    wrist_to_head_yaw_frame,
)
from gear_sonic.utils.teleop.operator_calibration import (
    ARMS_DOWN_LEFT,
    ARMS_DOWN_RIGHT,
)
from gear_sonic.utils.teleop.solver.arm import ArmIKSolver, IKResult
from gear_sonic.utils.teleop.vr_arm_teleop import (
    DEFAULT_LEFT_ARM_NEUTRAL_RAD,
    DEFAULT_RIGHT_ARM_NEUTRAL_RAD,
)


_IDENTITY_QUAT_WXYZ: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _rpy_deg_to_quat_wxyz(rpy_deg: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """Convert intrinsic XYZ Tait-Bryan RPY (degrees) to a wxyz quaternion.

    Convention: ``q = q_x(roll) * q_y(pitch) * q_z(yaw)``, applied in the
    object's body-fixed frame. ``q_axis(theta)`` is the unit quaternion
    rotating ``theta`` radians about the named axis. This matches scipy's
    ``Rotation.from_euler('XYZ', rpy, degrees=True)`` (uppercase 'XYZ' =
    intrinsic) so callers can sanity-check from a python repl.

    Returns the identity quaternion ``[1, 0, 0, 0]`` for an all-zeros
    input (cheap fast path; the dataclass constructor uses this to
    detect "no offset" and skip the multiply in the hot path).
    """
    arr = np.asarray(rpy_deg, dtype=np.float64).reshape(3)
    if not np.any(arr):
        return _IDENTITY_QUAT_WXYZ.copy()
    r, p, y = np.deg2rad(arr)
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    qx = np.array([cr, sr, 0.0, 0.0], dtype=np.float64)
    qy = np.array([cp, 0.0, sp, 0.0], dtype=np.float64)
    qz = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
    return _quat_multiply_wxyz(_quat_multiply_wxyz(qx, qy), qz)


def _is_identity_quat(q_wxyz: np.ndarray) -> bool:
    """True if ``q_wxyz`` is exactly the identity (post-fast-path check)."""
    return bool(np.array_equal(q_wxyz, _IDENTITY_QUAT_WXYZ))


# The IK null-space "preferred" posture used while the operator is
# actively engaged. We pull toward the calibration ``arms_down``
# reference (fully-straight arms at the sides) -- NOT the SONIC stand
# pose (which has bent elbows). Otherwise the operator extending their
# arms fully would still produce a robot pose with bent elbows because
# the null-space bias was fighting the IK back toward the bent-arm
# posture. Disengaged behaviour still snaps to the (bent) stand
# neutral so SONIC's standing controller gets the reference it expects.
#
# IMPORTANT: ``ARMS_DOWN`` has ``elbow = 0`` exactly, which is the X2
# elbow's *upper* joint limit (range ``[-2.36, 0]`` rad). Using that
# as the null-space attractor pinned the elbow against the limit
# whenever the operator's hands were close to their torso, because the
# DLS gradient's "increase elbow" direction is physically impossible
# and the joint just clamps at 0. With the elbow locked the shoulder
# would then wind into its own joint limit trying to compensate,
# producing arm-flip artefacts. The 2026-06-07 NPZ replay showed 33 %
# of engaged left-arm ticks parked at elbow=0 with median 12 cm IK
# error (vs 1.1 cm when the elbow was free); the worst-case wrist
# placement was 78 cm off-target. Holding the preferred posture at a
# small flex (~17 deg) keeps the elbow off the limit so the gradient
# has signed motion in both directions. Replaying the same NPZ with
# the flexed preferred posture cut left-arm p95 IK error from 36.9 cm
# to 5.3 cm without changing the visual neutral pose noticeably.
_ELBOW_FLEX_PREFERRED_RAD = -0.30  # ~17 deg flex; keeps elbow off the upper limit
_LEFT_PREFERRED_Q = np.asarray(ARMS_DOWN_LEFT, dtype=np.float64).copy()
_LEFT_PREFERRED_Q[3] = _ELBOW_FLEX_PREFERRED_RAD
_RIGHT_PREFERRED_Q = np.asarray(ARMS_DOWN_RIGHT, dtype=np.float64).copy()
_RIGHT_PREFERRED_Q[3] = _ELBOW_FLEX_PREFERRED_RAD


# "Stuck-corner damper" thresholds. When the IK can't reach (residual
# position error exceeds ``_STUCK_ERR_M`` AND at least
# ``_STUCK_CLAMPED_COUNT`` joints are pinned to their joint limits),
# the DLS solver will thrash between joint-limit corners while chasing
# the unreachable target -- the operator sees the wrist visibly
# oscillate ("going crazy") even though the IK error itself stays
# roughly constant. Out-of-workspace targets the operator generates
# this way are common: wrist-to-shoulder, hands-meet-past-midline,
# folding both hands onto the chest, etc. The X2 arm cannot fold
# tightly enough (min wrist-to-shoulder distance is ~20 cm at max
# elbow flex) so these poses are geometrically unreachable.
#
# Fix: when the IK is in a "stuck corner", low-pass-filter the
# commanded q toward the solver's output instead of jumping. The
# wrist drifts smoothly toward the best-achievable joint configuration
# (which the IK is still finding correctly) instead of spazzing
# between corners. Replaying the 2026-06-07 helmet-mounted "hand-to-
# shoulder" episode through this filter cut LEFT-arm joint-space
# sign-flips during failure ticks from 180 -> 19 (10x smoother) and
# the same for the right arm (195 -> 14). Normal-operation episodes
# (where the filter only activates 4-7% of ticks) saw a 3x reduction
# in sign-flips with no perceptible lag.
_STUCK_ERR_M: float = 0.05
_STUCK_CLAMPED_COUNT: int = 2
_STUCK_LPF_ALPHA: float = 0.10  # tau ~= 0.18 s @ 50 Hz


_DROPOUT_POS_ORIGIN_THRESHOLD_M = 0.05
_DROPOUT_QUAT_IDENTITY_THRESHOLD = 0.01
_DROPOUT_TWIN_THRESHOLD_M = 0.05


def _is_controller_dropout(
    wrist_pos_robot_frame: np.ndarray,
    wrist_quat_wxyz: np.ndarray,
) -> bool:
    """True if a Quest 3 controller is reporting a "lost-tracking" sample.

    Two signatures from the WebXR API:

    * Position is exactly the origin (or within a few centimetres of it).
      WebXR returns ``(0, 0, 0)`` when ``getPose()`` cannot resolve the
      controller; our ``compute_3pt_pose_from_quest3`` floor-projects the
      head, so a dropped controller ends up at robot-frame origin too.
    * Orientation quaternion is exactly identity ``(1, 0, 0, 0)``.
      Same fallback behaviour for the rotation channel.

    Either condition by itself produces ~7 % of corrupt frames in the
    v2 recording; combined they're a reliable detector.
    """
    if np.linalg.norm(wrist_pos_robot_frame) < _DROPOUT_POS_ORIGIN_THRESHOLD_M:
        return True
    quat = np.asarray(wrist_quat_wxyz, dtype=np.float64)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    if np.linalg.norm(quat - identity) < _DROPOUT_QUAT_IDENTITY_THRESHOLD:
        return True
    return False


def _is_twin_dropout(
    left_wrist_pos: np.ndarray,
    right_wrist_pos: np.ndarray,
) -> bool:
    """True if both controllers report (nearly) identical positions.

    When *both* controllers lose tracking simultaneously, WebXR can fall
    back to the headset's pose for both — they'll report the same xyz.
    A real bimanual operator never has both wrists at the same point.
    """
    return float(np.linalg.norm(left_wrist_pos - right_wrist_pos)) < _DROPOUT_TWIN_THRESHOLD_M


@dataclass
class CalibratedTeleopTickResult:
    """Returned by :meth:`VRArmTeleopCalibrated.step`.

    Mirrors :class:`gear_sonic.utils.teleop.vr_arm_teleop.TeleopTickResult`
    so the calling script can swap implementations with no other changes,
    plus a couple of extra diagnostic fields (``op_wrist_in_head_yaw``)
    that are useful for the side-channel debug NPZ.
    """

    left_q: np.ndarray  # (7,) commanded left arm joints (rad)
    right_q: np.ndarray  # (7,)
    left_target_pos: np.ndarray  # (3,) torso-frame wrist target
    right_target_pos: np.ndarray
    left_op_wrist_head_yaw: np.ndarray  # (3,) operator wrist in head-yaw frame
    right_op_wrist_head_yaw: np.ndarray
    head_yaw_rad: float
    left_ik: IKResult
    right_ik: IKResult
    engaged: bool
    # New diagnostic fields for dropout handling.
    left_dropout: bool = False  # this tick's left wrist was a dropout sample
    right_dropout: bool = False
    left_target_held: bool = False  # IK target was held from a previous tick
    right_target_held: bool = False


class VRArmTeleopCalibrated:
    """Quest 3 3-pt pose -> X2 dual-arm joint target stream, calibrated.

    Args:
        calibration: pre-fitted :class:`OperatorCalibration` (typically
            loaded from YAML via ``OperatorCalibration.load_yaml``).
        left_neutral_q / right_neutral_q: 7-DOF arm angles to hold when
            ``engaged=False`` (idle). Defaults to the X2 stand pose.
        damping: DLS damping (forwarded to :class:`ArmIKSolver`).
        rotation_weight: orientation weight for IK. v0 default 0.0
            (position-only) since wrist orientation is not calibrated.
        per_tick_step_rad: per-joint upper bound on a single DLS step.

    Public surface:
        :meth:`set_engaged` -- toggle active vs idle.
        :meth:`step` -- run one DLS tick and return new arm targets.
        :attr:`is_engaged` -- read current state.
        :attr:`calibration` -- the loaded calibration (read-only).
    """

    def __init__(
        self,
        *,
        calibration: OperatorCalibration,
        left_neutral_q: np.ndarray | None = None,
        right_neutral_q: np.ndarray | None = None,
        damping: float = 0.08,
        rotation_weight: float = 0.0,
        per_tick_step_rad: float = 0.30,
        null_space_gain: float = 0.10,
        stuck_err_m: float = _STUCK_ERR_M,
        stuck_clamped_count: int = _STUCK_CLAMPED_COUNT,
        stuck_lpf_alpha: float = _STUCK_LPF_ALPHA,
        left_wrist_op_quat_offset_rpy_deg: (
            tuple[float, float, float] | np.ndarray | None
        ) = None,
        right_wrist_op_quat_offset_rpy_deg: (
            tuple[float, float, float] | np.ndarray | None
        ) = None,
    ) -> None:
        """``left_wrist_op_quat_offset_rpy_deg`` /
        ``right_wrist_op_quat_offset_rpy_deg``: optional 3-tuple of
        intrinsic Tait-Bryan ``(roll, pitch, yaw)`` degrees applied as a
        post-multiplication on the operator's wrist quaternion in the
        head-yaw frame, BEFORE the calibration's ``apply_to_wrist_quat``
        runs. Equivalent to "the controller is rigidly mounted at this
        rotation relative to the operator's actual wrist; pre-rotate it
        back". Use this as a stop-gap until ``vr_operator_calibrate.py``
        is re-run; once the YAML is refreshed the offsets should drop
        back to zero.

        The post-multiply ordering means the offset is applied in the
        controller's own (operator-wrist) local frame:

        * ``roll`` rotates about the operator's wrist long axis
          (pronation / supination).
        * ``pitch`` rotates about the operator's wrist lateral axis
          (flex / extend).
        * ``yaw`` rotates about the operator's wrist normal axis
          (ulnar / radial deviation).

        Defaults to ``None`` (== zero offset == identity quat == today's
        behaviour). The fast path skips the multiply when both sides are
        identity, so leaving these unset has zero per-tick cost.

        IMPORTANT: only takes effect when ``rotation_weight > 0``. When
        the IK is running position-only (``rotation_weight == 0`` --
        either explicitly or auto-clamped by the legacy-YAML detector),
        the IK never sees a rotation target and the offset is silently a
        no-op. This is logged at construction time when both offsets
        are non-zero but the effective rotation_weight is 0.
        """
        if calibration is None:
            raise ValueError("calibration is required")

        if left_neutral_q is None:
            left_neutral_q = np.asarray(DEFAULT_LEFT_ARM_NEUTRAL_RAD, dtype=np.float64)
        if right_neutral_q is None:
            right_neutral_q = np.asarray(DEFAULT_RIGHT_ARM_NEUTRAL_RAD, dtype=np.float64)
        if left_neutral_q.shape != (7,) or right_neutral_q.shape != (7,):
            raise ValueError(
                "neutral_q must be (7,) per side; got "
                f"left={left_neutral_q.shape} right={right_neutral_q.shape}"
            )

        self._calibration = calibration

        # Auto-disable orientation tracking if the loaded calibration
        # carries identity alignment quats. v0 calibration YAMLs only
        # capture wrist *position*, so their alignment defaults to
        # identity. Running the IK with ``rotation_weight > 0`` against
        # identity-aligned quats would twist the wrists toward an
        # arbitrary pose. The fix is to recalibrate (vr_operator_calibrate
        # now records wrist quats too); until then we silently fall back
        # to position-only IK to avoid making things worse.
        effective_rotation_weight = float(rotation_weight)
        if effective_rotation_weight > 0 and self._has_identity_alignment(calibration):
            print(
                "[VRArmTeleopCalibrated] WARNING: calibration has identity "
                "wrist alignment quats (legacy v0 YAML). Disabling wrist "
                "orientation IK. Re-run vr_operator_calibrate.py to enable "
                "wrist rotation tracking.",
                flush=True,
            )
            effective_rotation_weight = 0.0

        # Bias the redundant DOF toward the calibration arms-down
        # reference (FULLY-STRAIGHT arms hanging at the sides). This is
        # what fixes the "elbow flips backwards" failure mode for
        # unreachable IK targets, AND it ensures the operator extending
        # their arms fully produces straight robot arms (rather than
        # the bent-elbow stand-pose-default which the previous
        # implementation pulled toward and which made operators' fully
        # extended arms-down look bent on the robot).
        self._left_solver = ArmIKSolver(
            side="left",
            damping=damping,
            rotation_weight=effective_rotation_weight,
            max_per_tick_step_rad=per_tick_step_rad,
            null_space_gain=null_space_gain,
            q_preferred=_LEFT_PREFERRED_Q,
        )
        self._right_solver = ArmIKSolver(
            side="right",
            damping=damping,
            rotation_weight=effective_rotation_weight,
            max_per_tick_step_rad=per_tick_step_rad,
            null_space_gain=null_space_gain,
            q_preferred=_RIGHT_PREFERRED_Q,
        )
        self._left_neutral_q = left_neutral_q.copy()
        self._right_neutral_q = right_neutral_q.copy()
        self._left_q = left_neutral_q.copy()
        self._right_q = right_neutral_q.copy()
        self._engaged = False
        self._rotation_weight = effective_rotation_weight
        # Stuck-corner damper thresholds (see module-level constants for the
        # rationale). ``alpha == 1.0`` disables the damper (q jumps to the
        # solver's raw output each tick == today's pre-fix behaviour).
        self._stuck_err_m = float(stuck_err_m)
        self._stuck_clamped_count = int(stuck_clamped_count)
        self._stuck_lpf_alpha = float(stuck_lpf_alpha)
        # Last known-good IK target per side. Initialised to None so the
        # first dropout-only tick falls back to the neutral pose instead
        # of zeros. Updated every clean tick.
        self._last_left_target: np.ndarray | None = None
        self._last_right_target: np.ndarray | None = None

        # Pre-bake the optional operator-side wrist quat offsets to wxyz
        # quaternions once. Both default to identity == no-op.
        #
        # Resolution order (first non-None / non-zero wins):
        #   1. Explicit constructor kwarg (CLI override path).
        #   2. ``calibration.fit[side].op_quat_offset_rpy_deg`` from the
        #      YAML, so a single calibration file applies the same
        #      offset across every launcher (kinematic teleop, dataset
        #      recording, VLA bridge) without per-script CLI plumbing.
        #   3. Zero (no offset).
        def _resolve_offset(
            side: str,
            override: tuple[float, float, float] | np.ndarray | None,
        ) -> tuple[tuple[float, float, float] | np.ndarray, str]:
            if override is not None:
                return override, "constructor-kwarg"
            yaml_v = calibration.fit[side].op_quat_offset_rpy_deg
            if np.any(yaml_v):
                return tuple(float(v) for v in yaml_v), "calibration-yaml"
            return (0.0, 0.0, 0.0), "default-identity"

        l_off_resolved, l_src = _resolve_offset("left", left_wrist_op_quat_offset_rpy_deg)
        r_off_resolved, r_src = _resolve_offset("right", right_wrist_op_quat_offset_rpy_deg)
        self._left_op_quat_offset_wxyz = _rpy_deg_to_quat_wxyz(l_off_resolved)
        self._right_op_quat_offset_wxyz = _rpy_deg_to_quat_wxyz(r_off_resolved)
        self._has_left_op_offset = not _is_identity_quat(self._left_op_quat_offset_wxyz)
        self._has_right_op_offset = not _is_identity_quat(self._right_op_quat_offset_wxyz)
        if self._has_left_op_offset or self._has_right_op_offset:
            print(
                f"[VRArmTeleopCalibrated] wrist op-quat offsets active: "
                f"left_rpy_deg={tuple(float(v) for v in l_off_resolved)} "
                f"({l_src}); right_rpy_deg={tuple(float(v) for v in r_off_resolved)} "
                f"({r_src}).",
                flush=True,
            )
        if (
            (self._has_left_op_offset or self._has_right_op_offset)
            and self._rotation_weight == 0.0
        ):
            print(
                "[VRArmTeleopCalibrated] WARNING: wrist op-quat offsets "
                "configured but rotation_weight is 0. The IK runs "
                "position-only -- the offsets will be a no-op until "
                "rotation_weight > 0.",
                flush=True,
            )

    @staticmethod
    def _has_identity_alignment(calibration: OperatorCalibration) -> bool:
        """True if both arms' alignment quats look like the legacy default."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        for side in ("left", "right"):
            q = calibration.fit[side].wrist_alignment_quat
            # tolerance covers small numerical noise and intentional sign-flip
            if np.linalg.norm(q - identity) > 1e-3 and np.linalg.norm(q + identity) > 1e-3:
                return False
        return True

    def _apply_stuck_corner_damper(
        self,
        q_prev: np.ndarray,
        q_new: np.ndarray,
        info: IKResult,
    ) -> np.ndarray:
        """Smooth the commanded q toward ``q_new`` when the IK is stuck.

        See the module-level ``_STUCK_*`` constants for the rationale.
        Briefly: when the operator's wrist target is geometrically
        unreachable (e.g. wrist-to-shoulder), the DLS solver thrashes
        between joint-limit corners and the wrist visibly oscillates
        even though the IK error is roughly constant. Low-pass-filtering
        the commanded q during these "stuck corner" ticks keeps the
        motion smooth without changing normal-operation behaviour.

        The condition for "stuck" is conservative on purpose: a large
        residual error alone (e.g. mid-transition while chasing a far
        target) does NOT trigger the damper -- we also require at
        least ``stuck_clamped_count`` joints to be pinned at their
        limits in the same iteration, which is the actual signature of
        an unreachable target.
        """
        if self._stuck_lpf_alpha >= 1.0:
            return q_new
        if (
            info.pos_err_m > self._stuck_err_m
            and info.clamped_count >= self._stuck_clamped_count
        ):
            a = self._stuck_lpf_alpha
            return (1.0 - a) * q_prev + a * q_new
        return q_new

    # -- properties -----------------------------------------------------------

    @property
    def calibration(self) -> OperatorCalibration:
        return self._calibration

    @property
    def is_engaged(self) -> bool:
        return self._engaged

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

    def set_engaged(self, engaged: bool) -> None:
        """Toggle active vs idle. When transitioning back to active we
        keep the last commanded q as the IK seed; the calibrated mapping
        is stateless so there's nothing else to reset.
        """
        self._engaged = bool(engaged)

    def reset_to_neutral(self) -> None:
        """Snap commanded q back to neutral (and idle the solver).

        Useful when the operator wants to "park" the arms before
        stepping out of the play area.
        """
        self._left_q = self._left_neutral_q.copy()
        self._right_q = self._right_neutral_q.copy()
        self._engaged = False

    # -- per-tick step --------------------------------------------------------

    def step(self, vr_3pt_pose: np.ndarray) -> CalibratedTeleopTickResult:
        """One DLS tick. Idempotent if the input doesn't change.

        Args:
            vr_3pt_pose: ``(3, 7)`` array
                ``[lwrist, rwrist, head] = [pos_xyz, quat_wxyz]`` in the
                robot frame (same payload as
                ``Quest3Reader.get_3pt_pose``).

        Returns:
            :class:`CalibratedTeleopTickResult` with new arm joints,
            wrist targets, head-yaw-frame wrist diagnostics, IK
            residuals, and engage state.

        When ``set_engaged(False)`` (idle), we return the last
        commanded q untouched and report engaged=False with zero
        residuals -- so the upstream loop can hold the robot still
        without flicker.
        """
        vr = np.asarray(vr_3pt_pose, dtype=np.float64)
        if vr.shape != (3, 7):
            raise ValueError(f"vr_3pt_pose must be (3, 7); got {vr.shape}")

        head_pos = vr[2, :3]
        head_quat = vr[2, 3:]
        yaw = head_yaw_from_quat(head_quat)
        l_op = wrist_to_head_yaw_frame(vr[0, :3], head_pos, head_quat)
        r_op = wrist_to_head_yaw_frame(vr[1, :3], head_pos, head_quat)

        # Detect dropout samples per side BEFORE we apply any
        # calibration. ``compute_3pt_pose_from_quest3`` floor-projects
        # head x/y to zero, so a dropped controller arrives in the robot
        # frame either at the origin (pos lost) or with identity quat
        # (rotation lost). A "twin" dropout where both controllers
        # report the same position is also corrupted (WebXR sometimes
        # falls back to the headset pose for both).
        left_drop = _is_controller_dropout(vr[0, :3], vr[0, 3:])
        right_drop = _is_controller_dropout(vr[1, :3], vr[1, 3:])
        if _is_twin_dropout(vr[0, :3], vr[1, :3]):
            left_drop = True
            right_drop = True

        if not self._engaged:
            zero3 = np.zeros(3, dtype=np.float64)
            return CalibratedTeleopTickResult(
                left_q=self._left_q.copy(),
                right_q=self._right_q.copy(),
                left_target_pos=zero3.copy(),
                right_target_pos=zero3.copy(),
                left_op_wrist_head_yaw=l_op,
                right_op_wrist_head_yaw=r_op,
                head_yaw_rad=yaw,
                left_ik=IKResult(0.0, 0.0, 0, 0),
                right_ik=IKResult(0.0, 0.0, 0, 0),
                engaged=False,
                left_dropout=left_drop,
                right_dropout=right_drop,
            )

        # Compose IK targets per side, falling back to the last good
        # target whenever the live sample is a dropout. If we have no
        # last-good target yet (very first ticks), freeze the FK of the
        # current q ONCE -- otherwise subsequent dropout ticks would
        # recompute FK against a q that has drifted under null-space
        # bias, which is both visually annoying (arms slowly creep
        # toward the preferred posture during a long dropout) AND a
        # subtle break of the head-yaw invariance contract (target
        # changes even though the operator pose did not).
        left_target_held = False
        right_target_held = False
        if left_drop:
            left_target_held = True
            if self._last_left_target is None:
                self._last_left_target = self._left_solver.fk(self._left_q)[0].copy()
            l_target = self._last_left_target.copy()
        else:
            l_target = self._calibration.apply_to_wrist(l_op, "left")
            self._last_left_target = l_target.copy()

        if right_drop:
            right_target_held = True
            if self._last_right_target is None:
                self._last_right_target = self._right_solver.fk(self._right_q)[0].copy()
            r_target = self._last_right_target.copy()
        else:
            r_target = self._calibration.apply_to_wrist(r_op, "right")
            self._last_right_target = r_target.copy()

        # Wrist orientation: head-yaw-correct the operator quat, then
        # apply the per-arm alignment learned at calibration so the
        # wrist quaternion is expressed in the robot's torso frame.
        # Skip on dropout frames since the quat is identity / corrupt.
        #
        # Optional per-side ``_*_op_quat_offset_wxyz`` is post-multiplied
        # on the head-yaw-frame operator quat BEFORE the calibration
        # alignment. This is the "controller mount calibration" knob:
        # if the controller sits at a fixed rotation on the operator's
        # wrist (e.g. left controller mounted with ~30deg outward yaw on
        # the cuff), pre-rotating by the inverse of that mount alignment
        # makes the calibration see a "corrected" operator quat. The
        # offset is in the operator-wrist's local frame (post-mul); see
        # the ctor docstring for the (roll, pitch, yaw) axis convention.
        # Identity offsets are skipped via the ``_has_*_op_offset`` flag
        # so the hot path stays free of wasted multiplies.
        if self._rotation_weight == 0.0:
            l_quat_target: np.ndarray | None = None
            r_quat_target: np.ndarray | None = None
        else:
            if left_drop:
                l_quat_target = None
            else:
                l_op_quat = wrist_quat_to_head_yaw_frame(vr[0, 3:], head_quat)
                if self._has_left_op_offset:
                    l_op_quat = _quat_multiply_wxyz(
                        l_op_quat, self._left_op_quat_offset_wxyz,
                    )
                l_quat_target = self._calibration.apply_to_wrist_quat(l_op_quat, "left")
            if right_drop:
                r_quat_target = None
            else:
                r_op_quat = wrist_quat_to_head_yaw_frame(vr[1, 3:], head_quat)
                if self._has_right_op_offset:
                    r_op_quat = _quat_multiply_wxyz(
                        r_op_quat, self._right_op_quat_offset_wxyz,
                    )
                r_quat_target = self._calibration.apply_to_wrist_quat(r_op_quat, "right")

        # IMPORTANT: skip the IK solve entirely on dropout frames.
        # Holding only the IK *target* (above) is not enough -- the
        # solver still runs one Newton iteration each tick and the
        # null-space bias term ``q_preferred - q_current`` keeps
        # nudging q toward the preferred posture even when position
        # error is ~0. Over a 200 ms dropout (10 frames at 50 Hz)
        # the wrist roll DOF drifts ~50 deg, and the "real" frame
        # that ends the dropout sees a sudden recovery jolt as the
        # IK pulls q back toward the held target's null-space
        # solution. The user perceives this as a wrist orientation
        # flip whenever a controller momentarily loses tracking.
        # The fix: hold q outright, not just the target. We still
        # report a zero-error IKResult so the diagnostic stream
        # stays clean.
        if left_drop:
            l_info = IKResult(0.0, 0.0, 0, 0)
        else:
            q_new_left, l_info = self._left_solver.solve(
                q_seed=self._left_q,
                target_pos=l_target,
                target_quat_wxyz=l_quat_target,
                max_iters=1,
            )
            self._left_q = self._apply_stuck_corner_damper(
                self._left_q, q_new_left, l_info,
            )
        if right_drop:
            r_info = IKResult(0.0, 0.0, 0, 0)
        else:
            q_new_right, r_info = self._right_solver.solve(
                q_seed=self._right_q,
                target_pos=r_target,
                target_quat_wxyz=r_quat_target,
                max_iters=1,
            )
            self._right_q = self._apply_stuck_corner_damper(
                self._right_q, q_new_right, r_info,
            )

        return CalibratedTeleopTickResult(
            left_q=self._left_q.copy(),
            right_q=self._right_q.copy(),
            left_target_pos=l_target,
            right_target_pos=r_target,
            left_op_wrist_head_yaw=l_op,
            right_op_wrist_head_yaw=r_op,
            head_yaw_rad=yaw,
            left_ik=l_info,
            right_ik=r_info,
            engaged=True,
            left_dropout=left_drop,
            right_dropout=right_drop,
            left_target_held=left_target_held,
            right_target_held=right_target_held,
        )


__all__ = [
    "CalibratedTeleopTickResult",
    "VRArmTeleopCalibrated",
    "_is_controller_dropout",
    "_is_twin_dropout",
    "_rpy_deg_to_quat_wxyz",
]
