"""Per-tick X2 retargeting pipeline used by both the recorder and the manager.

This wraps the four pieces of the X2 Quest 3 retargeting stack so they
can be driven from a single ``step()`` call:

1. :class:`gear_sonic.utils.teleop.vr_arm_teleop_v2.VRArmTeleopCalibrated`
   — calibrated DLS arm IK with dropout handling and null-space biasing.
2. :class:`gear_sonic.utils.teleop.finger_signal_filter.FingerSignalFilter`
   — per-side EMA + deadband-hold smoother (one instance per hand).
3. :func:`gear_sonic.utils.teleop.x2_hand_retarget.per_finger_grasp_command_from_curls_and_oppose`
   — XRHand 5-finger curls + thumb opposition -> 10-DOF OmniHand command.
4. ``DEFAULT_STAND_POSE_MUJOCO_RAD`` overlay
   — pin lower body / waist / head to the trained stand pose; overlay
     the operator-driven 7+7 arm joints into MJ slots ``[15:22]`` and
     ``[22:29]``.

The dataset recorder and the new manager
(`quest3_manager_x2.py`) both call this so they cannot drift apart.
The blocking parity test
(`tests/test_quest3_manager_x2_retargeting_parity.py`) drives the same
``Retargeter`` from a recorded NPZ and diffs the outputs against the
parquet ground truth — that is the contract that pins the lift-and-shift.

Lift-and-shift provenance
-------------------------

Code originally factored out of the dataset recorder:

- ``_compose_body_q`` (recorder line ~1055) -> :func:`compose_body_q`.
- The hand-input dispatch around recorder lines ~778-845 (XRHand
  fast path + controller fallback + finger filter) -> :class:`Retargeter`.
  Optional ``left_hand_source`` / ``right_hand_source`` on
  :class:`RetargetTickInput` reset the filter on Quest ``hands.*.source``
  transitions without affecting NPZ replay (sources default to
  ``None``).
- Recorder lines ~847-872 (engaged -> compose body_q overlay; idle ->
  return DEFAULT_STAND_POSE) -> :class:`Retargeter.step`.

Behaviour is preserved bit-for-bit; the parity test enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Sentinel for "no prior Quest ``hands.*.source`` yet" (distinct from
# ``None`` meaning unknown / absent on the wire this frame).
_PREV_HAND_SRC_UNSET: Any = object()

from gear_sonic.utils.planner.constants import DEFAULT_STAND_POSE_MUJOCO_RAD
from gear_sonic.utils.teleop.finger_signal_filter import (
    FingerFilterParams,
    FingerSignalFilter,
)
from gear_sonic.utils.teleop.operator_calibration import OperatorCalibration
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import VRArmTeleopCalibrated
from gear_sonic.utils.teleop.x2_hand_retarget import (
    NUM_HAND_DOF_PER_SIDE,
    controller_grasp_ratio,
    grasp_command_from_ratio,
    per_finger_grasp_command_from_curls_and_oppose,
)

# Index ranges into the 31-DOF MuJoCo body_q vector. These MUST match
# the dataset recorder's retarget defaults.
LEFT_ARM_MJ_SLICE = slice(15, 22)
RIGHT_ARM_MJ_SLICE = slice(22, 29)


# ---------------------------------------------------------------------------
# Per-tick I/O dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetargetTickInput:
    """One frame of Quest 3 inputs needed for retargeting.

    All optional fields can be ``None`` when the headset isn't reporting
    them this frame; the retargeter falls back to the controller
    trigger/grip path for hands and to the dropout / hold logic inside
    ``VRArmTeleopCalibrated`` for arms.
    """

    vr_pose: np.ndarray
    """``(3, 7)`` array — ``[lwrist, rwrist, neck]`` rows, each
    ``[pos_xyz, quat_wxyz]`` in the robot frame. Same payload as
    ``Quest3Reader.get_3pt_pose()``."""

    triggers: tuple[float, float, float, float]
    """``(left_trigger, right_trigger, left_grip, right_grip)`` analog
    scalars in ``[0, 1]``. Used as the hand-input fallback when XRHand
    is not reporting."""

    left_curls: Optional[np.ndarray]
    """``(5,)`` per-finger curl in ``[0, 1]`` (``thumb, index, middle,
    ring, pinky``) from XRHand, or ``None``."""
    right_curls: Optional[np.ndarray]
    left_thumb_oppose: Optional[float]
    """Scalar in ``[0, 1]``; XRHand thumb-opposition signal."""
    right_thumb_oppose: Optional[float]
    left_finger_tip_oppose: Optional[np.ndarray]
    """``(4,)`` per-non-thumb finger -> thumb-tip proximity scalar; the
    "thumb-tip-touches-finger-tip" pinch signal."""
    right_finger_tip_oppose: Optional[np.ndarray]

    left_hand_source: Optional[str] = None
    """Quest ``hands.left.source`` (``"hand"`` / ``"controller"``) from
    :meth:`Quest3Reader.get_hand_curls`, or ``None`` when absent.

    Used only to reset :class:`FingerSignalFilter` on source transitions
    so the filter's NaN-holding EMA cannot freeze XRHand curls across a
    controller-only segment (which would strand dispatch on the XRHand
    path and ignore trigger/grip forever).
    """
    right_hand_source: Optional[str] = None
    """Same as ``left_hand_source`` for the right side."""


@dataclass
class RetargetTickOutput:
    """Outputs of one retargeting tick.

    ``body_q_mj`` is the FULL 31-DOF body command (legs + waist +
    arms + head); arms come from IK when ``is_engaged``, otherwise the
    full vector equals ``DEFAULT_STAND_POSE_MUJOCO_RAD``.
    """

    body_q_mj: np.ndarray  # (31,) float64
    left_hand_q: np.ndarray  # (10,) float64
    right_hand_q: np.ndarray  # (10,) float64

    is_engaged: bool

    # Diagnostics (for sidecar / parity test).
    left_arm_q: np.ndarray  # (7,)
    right_arm_q: np.ndarray
    left_curls_filtered: Optional[np.ndarray]
    right_curls_filtered: Optional[np.ndarray]
    left_thumb_oppose_filtered: Optional[float]
    right_thumb_oppose_filtered: Optional[float]
    left_finger_tip_oppose_filtered: Optional[np.ndarray]
    right_finger_tip_oppose_filtered: Optional[np.ndarray]
    left_dropout: bool
    right_dropout: bool


# ---------------------------------------------------------------------------
# Pure helper: arm-overlay onto stand pose.
# ---------------------------------------------------------------------------


def compose_body_q(
    *,
    left_arm_q: np.ndarray,
    right_arm_q: np.ndarray,
) -> np.ndarray:
    """Build a 31-DOF body_q with arms overlaid onto the trained stand pose.

    Pure function (no state). Identical math to
    ``X2DatasetRecorder._compose_body_q`` (recorder line ~1055).
    """
    if left_arm_q.shape != (7,) or right_arm_q.shape != (7,):
        raise ValueError(
            f"arm_q must be (7,) per side; got "
            f"L={left_arm_q.shape} R={right_arm_q.shape}"
        )
    body = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
    body[LEFT_ARM_MJ_SLICE] = left_arm_q
    body[RIGHT_ARM_MJ_SLICE] = right_arm_q
    return body


# ---------------------------------------------------------------------------
# Retargeter
# ---------------------------------------------------------------------------


class Retargeter:
    """Stateful X2 retargeter: VR + Quest 3 hands -> body_q + hand_q.

    Args:
        calibration: Operator calibration (head-yaw -> wrist mapping +
            optional hand_range for per-finger floor/ceiling).
        finger_filter_params: Finger filter config; pass ``None`` to
            disable smoothing (NOT recommended for live use).
        ik_damping: DLS damping factor for the arm IK.
        ik_rotation_weight: Wrist orientation weight in the IK cost.
            Auto-clamps to 0 if the calibration has identity alignment.
        ik_per_tick_step_rad: Maximum per-DOF step the IK can take in
            one tick. Limits rate of motion to keep deploy-side
            tracking stable.
        hand_input_mode: Controller-fallback dispatch:
            ``"trigger"`` / ``"grip"`` / ``"max"``. Selects which
            analog scalar feeds ``controller_grasp_ratio`` when XRHand
            is not reporting curls.
        apply_curl_compensation: Forwarded to
            ``per_finger_grasp_command_from_curls_and_oppose``.
        apply_oppose_compensation: ditto.
        left_wrist_op_quat_offset_rpy_deg /
        right_wrist_op_quat_offset_rpy_deg: optional 3-tuple of
            intrinsic Tait-Bryan ``(roll, pitch, yaw)`` degrees applied
            in the operator's wrist-local frame BEFORE the calibration
            is applied. Stop-gap fix for a controller mount that's
            slightly misaligned on the operator's wrist; rerun
            ``vr_operator_calibrate.py`` to drop these to zero. See
            :class:`VRArmTeleopCalibrated` for the axis convention.
    """

    def __init__(
        self,
        *,
        calibration: OperatorCalibration,
        finger_filter_params: Optional[FingerFilterParams] = None,
        ik_damping: float = 0.08,
        ik_rotation_weight: float = 0.3,
        ik_per_tick_step_rad: float = 0.30,
        hand_input_mode: str = "trigger",
        apply_curl_compensation: bool = False,
        apply_oppose_compensation: bool = False,
        left_neutral_q: Optional[np.ndarray] = None,
        right_neutral_q: Optional[np.ndarray] = None,
        left_wrist_op_quat_offset_rpy_deg: Optional[
            tuple[float, float, float]
        ] = None,
        right_wrist_op_quat_offset_rpy_deg: Optional[
            tuple[float, float, float]
        ] = None,
    ) -> None:
        if calibration is None:
            raise ValueError("calibration is required")

        self._calibration = calibration
        # ``left_neutral_q`` / ``right_neutral_q`` set the IK's initial
        # joint state. The parity test passes the recorded NPZ's frame-0
        # IK output so the test starts from the same IK state as the
        # original recording (otherwise we'd see a ~30-frame DLS
        # convergence transient on the test diff).
        self._teleop = VRArmTeleopCalibrated(
            calibration=calibration,
            damping=ik_damping,
            rotation_weight=ik_rotation_weight,
            per_tick_step_rad=ik_per_tick_step_rad,
            left_neutral_q=left_neutral_q,
            right_neutral_q=right_neutral_q,
            left_wrist_op_quat_offset_rpy_deg=left_wrist_op_quat_offset_rpy_deg,
            right_wrist_op_quat_offset_rpy_deg=right_wrist_op_quat_offset_rpy_deg,
        )

        if finger_filter_params is not None:
            finger_filter_params.validate()
            self._filt_left: Optional[FingerSignalFilter] = FingerSignalFilter(
                finger_filter_params
            )
            self._filt_right: Optional[FingerSignalFilter] = FingerSignalFilter(
                finger_filter_params
            )
        else:
            self._filt_left = None
            self._filt_right = None

        self._prev_left_hand_src: Any = _PREV_HAND_SRC_UNSET
        self._prev_right_hand_src: Any = _PREV_HAND_SRC_UNSET

        self._hand_input_mode = hand_input_mode
        self._apply_curl_comp = bool(apply_curl_compensation)
        self._apply_oppose_comp = bool(apply_oppose_compensation)

    # -- engage state ---------------------------------------------------------

    @property
    def is_engaged(self) -> bool:
        return self._teleop.is_engaged

    def set_engaged(self, on: bool) -> None:
        self._teleop.set_engaged(on)

    # -- finger filter --------------------------------------------------------

    def reset_finger_filter(self) -> None:
        """Clear the per-side finger filter state.

        Call this on episode boundaries so the warm-up window doesn't
        leak EMA state from the previous episode.
        """
        if self._filt_left is not None:
            self._filt_left.reset()
        if self._filt_right is not None:
            self._filt_right.reset()
        self._prev_left_hand_src = _PREV_HAND_SRC_UNSET
        self._prev_right_hand_src = _PREV_HAND_SRC_UNSET

    @property
    def calibration(self) -> OperatorCalibration:
        return self._calibration

    # -- main step ------------------------------------------------------------

    def step(self, inp: RetargetTickInput) -> RetargetTickOutput:
        """Run one tick of arm IK + hand retargeting, return the wire payload.

        Mirrors the per-tick pipeline in ``X2DatasetRecorder.run()``
        lines ~778-872. The blocking parity test verifies that this
        produces bit-equivalent outputs to the recorder.
        """
        # When the WebXR client flips ``hands.*.source`` between
        # ``"hand"`` and ``"controller"`` (operator toggles hand-tracking
        # vs controllers-only), clear the per-side finger filter. Without
        # this, ``FingerSignalFilter``'s NaN-tolerant EMA keeps outputting
        # the last finite XRHand curls while raw ``curls`` are ``None``,
        # so ``l_curls is not None`` stays true and trigger/grip never
        # reaches ``grasp_command_from_ratio`` again.
        if self._filt_left is not None and self._filt_right is not None:
            if (
                self._prev_left_hand_src is not _PREV_HAND_SRC_UNSET
                and inp.left_hand_source != self._prev_left_hand_src
            ):
                self._filt_left.reset()
            if (
                self._prev_right_hand_src is not _PREV_HAND_SRC_UNSET
                and inp.right_hand_source != self._prev_right_hand_src
            ):
                self._filt_right.reset()
            self._prev_left_hand_src = inp.left_hand_source
            self._prev_right_hand_src = inp.right_hand_source

            l_curls, l_oppose, l_tip = self._filt_left.update(
                inp.left_curls, inp.left_thumb_oppose, inp.left_finger_tip_oppose,
            )
            r_curls, r_oppose, r_tip = self._filt_right.update(
                inp.right_curls, inp.right_thumb_oppose, inp.right_finger_tip_oppose,
            )
        else:
            l_curls, l_oppose, l_tip = (
                inp.left_curls, inp.left_thumb_oppose, inp.left_finger_tip_oppose,
            )
            r_curls, r_oppose, r_tip = (
                inp.right_curls, inp.right_thumb_oppose, inp.right_finger_tip_oppose,
            )

        # Hand command dispatch — XRHand fast path when curls present,
        # else the controller-trigger/grip fallback.
        hr = self._calibration.hand_range
        l_hr = hr.left if hr is not None else None
        r_hr = hr.right if hr is not None else None

        if l_curls is not None:
            left_hand_q = per_finger_grasp_command_from_curls_and_oppose(
                "left", l_curls, l_oppose,
                finger_tip_oppose=l_tip,
                apply_curl_compensation=self._apply_curl_comp,
                apply_oppose_compensation=self._apply_oppose_comp,
                curl_floor=l_hr.floor if l_hr is not None else None,
                curl_ceiling=l_hr.ceiling if l_hr is not None else None,
                oppose_floor=l_hr.oppose_floor if l_hr is not None else None,
                oppose_ceiling=l_hr.oppose_ceiling if l_hr is not None else None,
            )
        else:
            l_ratio, _ = controller_grasp_ratio(
                left_trigger=inp.triggers[0],
                right_trigger=inp.triggers[1],
                left_grip=inp.triggers[2],
                right_grip=inp.triggers[3],
                mode=self._hand_input_mode,
            )
            left_hand_q = grasp_command_from_ratio("left", l_ratio)

        if r_curls is not None:
            right_hand_q = per_finger_grasp_command_from_curls_and_oppose(
                "right", r_curls, r_oppose,
                finger_tip_oppose=r_tip,
                apply_curl_compensation=self._apply_curl_comp,
                apply_oppose_compensation=self._apply_oppose_comp,
                curl_floor=r_hr.floor if r_hr is not None else None,
                curl_ceiling=r_hr.ceiling if r_hr is not None else None,
                oppose_floor=r_hr.oppose_floor if r_hr is not None else None,
                oppose_ceiling=r_hr.oppose_ceiling if r_hr is not None else None,
            )
        else:
            _, r_ratio = controller_grasp_ratio(
                left_trigger=inp.triggers[0],
                right_trigger=inp.triggers[1],
                left_grip=inp.triggers[2],
                right_grip=inp.triggers[3],
                mode=self._hand_input_mode,
            )
            right_hand_q = grasp_command_from_ratio("right", r_ratio)

        # Arm IK.
        tick = self._teleop.step(inp.vr_pose)

        if self._teleop.is_engaged:
            body_q_mj = compose_body_q(
                left_arm_q=tick.left_q, right_arm_q=tick.right_q,
            )
        else:
            body_q_mj = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)

        return RetargetTickOutput(
            body_q_mj=body_q_mj,
            left_hand_q=left_hand_q,
            right_hand_q=right_hand_q,
            is_engaged=self._teleop.is_engaged,
            left_arm_q=tick.left_q.copy(),
            right_arm_q=tick.right_q.copy(),
            left_curls_filtered=l_curls,
            right_curls_filtered=r_curls,
            left_thumb_oppose_filtered=l_oppose,
            right_thumb_oppose_filtered=r_oppose,
            left_finger_tip_oppose_filtered=l_tip,
            right_finger_tip_oppose_filtered=r_tip,
            left_dropout=tick.left_dropout,
            right_dropout=tick.right_dropout,
        )


__all__ = [
    "LEFT_ARM_MJ_SLICE",
    "RIGHT_ARM_MJ_SLICE",
    "RetargetTickInput",
    "RetargetTickOutput",
    "Retargeter",
    "compose_body_q",
    "NUM_HAND_DOF_PER_SIDE",
]
