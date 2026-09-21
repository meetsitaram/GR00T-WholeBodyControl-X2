"""Per-operator wrist position + orientation calibration for VR teleop.

Why this module exists
----------------------

VR arm teleop has to translate the operator's wrist position (somewhere
inside their reach envelope, in their head's local frame) into a robot
wrist target (somewhere inside the X2 arm's reach envelope, in
``torso_link``). Three things vary between operators that the robot
doesn't care about:

* Operator anatomy: arm length, shoulder width, height. A 1.6 m operator
  holding their arms straight out at shoulder height does not put their
  wrists at the same robot-frame position as a 1.9 m operator does.
* Operator pose habits: how relaxed the elbows are, whether the wrists
  flex inward, etc.
* Where the operator stands in the play area / where they were facing
  when the WebXR session started.

The previous engage-anchor solver (``vr_arm_teleop.py``) tried to handle
this by snapshotting wrist positions at A-press time and replaying
deltas. That coupled retargeting to play-area frame and to controller
tracking quality at engage time -- both of which failed in practice
("hands behind the body" when the operator turned their body 90 deg in
place; see ``data/lerobot/x2_quest3_kinematic_v0`` analysis).

The replacement approach is stateless:

1. Once per operator, run a guided 4-pose calibration in VR.
2. For each pose (arms-down / T-pose / arms-forward / hands-together),
   record the mean operator wrist position in the head-yaw frame.
3. Fit a per-arm per-axis affine map ``p_robot = diag(s) * p_op + t``
   from those four operator measurements against the robot's known FK
   wrist positions for the same poses.
4. At runtime, every tick: take the operator's wrist in the head-yaw
   frame and apply the affine map to get a torso-frame wrist target for
   IK. No state, no anchor, no engage moment.

The fit is closed-form and overdetermined (4 measurements per axis, 2
unknowns per axis), so we get a residual back that's the right metric
for "is this calibration sane?". Reject threshold is 5 cm per arm.

Why 4 poses
~~~~~~~~~~~

The earlier 3-pose calibration (arms-down, T-pose, arms-forward) all
kept the operator's wrists on their own side of the body centerline:
left wrist y > 0 in head-yaw frame, right wrist y < 0. The
least-squares y-axis fit was therefore unconstrained near ``op_y = 0``
and produced a positive y-intercept on the left arm and negative on
the right -- the model predicted robot wrists ~25 cm off-center even
when the operator brought their hands together. The 4th pose
(hands-together) puts ``op_y`` near zero and anchors the y-axis fit at
the centerline, which fixes the bias.

Limitations
-----------

* This v0 only fits the arm reach envelope. The X2 waist
  (``waist_pitch / waist_roll / waist_yaw``) is held at neutral; torso
  tilt is documented as a v0 limitation. Adding torso tracking requires
  either the Meta Movement SDK or a chest tracker.
* The fit is per-axis (``diag(s)``), not full 3x3 rotation. That's
  intentional: per-axis is enough to absorb anatomy + bias, and a full
  affine would over-fit 3 noisy measurements.
* Wrist *orientation* uses a single per-arm alignment quaternion derived
  from the arms-down pose. This is enough to give the operator natural
  pitch/roll/yaw of the wrist in the robot frame; finer per-pose
  orientation calibration would need more data than 3 poses can
  realistically provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gear_sonic.utils.teleop.solver.arm.x2_arm_fk import arm_fk, arm_fk_pose


# Schema version stamped into YAML files. Bump on any incompatible
# change to the on-disk format.
#
# v1: 3-pose calibration (arms_down, t_pose, arms_forward). Suffered
#     from a structural y-axis bias because all three poses kept the
#     operator's wrists on their own side of the body centerline -- the
#     least-squares fit's y-intercept was therefore unconstrained near
#     ``op_y = 0`` and predicted robot wrists ~25 cm off-center even
#     when the operator brought their hands together. See the v2 commit
#     message and ``docs/source/tutorials/x2_dataset_record_and_replay.md``.
# v2: adds a 4th ``hands_together`` pose (operator brings both hands
#     together at chest height, arms forward, elbows bent). Robot
#     reference for v2 is a feasible joint-limited pose where both
#     wrists are ~6 cm apart at chest height. The fit becomes a
#     well-posed linear regression with 4 data points spanning the
#     centerline, fixing the ``hands together stays apart`` bug.
SCHEMA_VERSION = 2


# ── Robot reference joint angles for each calibration pose ───────────────
#
# These are the X2 arm joint targets the operator's anatomy is fitted
# against. We pick them so:
#   1. The wrist position from FK is well inside the X2 arm's reach
#      envelope (no IK-clamped corner cases).
#   2. The operator can hit them naturally without cognitive load.
#
# Joint order (per arm, 7 DOF):
#   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#   wrist_yaw, wrist_pitch, wrist_roll
#
# Sign conventions match the X2 URDF (see solver/arm/x2_arm_fk.py):
#   - shoulder_pitch +ve = arm forward (away from body)
#   - shoulder_roll  +ve on LEFT = arm abducting (away from body),
#                    +ve on RIGHT = arm adducting (toward body)
#   - elbow          -ve = elbow flexion (bringing wrist toward body)
#
# Numbers below are picked to match approximate human poses; the actual
# wrist-position references come from FK so small adjustments don't
# break the math.

# Pose 1: arms relaxed at sides, hanging FULLY STRAIGHT down.
#
# We deliberately use ``q = 0`` (every joint at its zero) instead of
# the SONIC C++ deploy's stand-pose default ``(0.2, 0.2, 0, -0.6, ...)``
# which has bent elbows. The bent-arm reference made the calibration's
# z-axis fit predict a robot wrist target ~3.5 cm ABOVE where the
# operator's wrist actually was when they extended their arms fully
# down -- the IK then had to KEEP the elbow bent to lift the wrist up
# to the predicted target, even though the operator was clearly asking
# for fully-extended arms. ``q = 0`` puts the wrist 14.6 cm below the
# torso link which is where straight arms genuinely terminate, so the
# scale_z absorbs the human/robot length difference cleanly and full
# arm extension on the operator becomes full arm extension on the
# robot.
#
# This change is INDEPENDENT of the deploy stand pose: SONIC's standing
# behavior is loaded from ``stand_pose_loader.hpp`` on the C++ side. The
# Python ``DEFAULT_*_NEUTRAL_RAD`` (used as the IK snap target when
# disengaged) still matches the SONIC stand pose so the robot doesn't
# move when the operator releases A.
ARMS_DOWN_LEFT = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ARMS_DOWN_RIGHT = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Pose 2: T-pose. Arms straight out sideways, parallel to the floor.
# shoulder_roll is the abducting axis: +1.4 rad on left = abduction.
# shoulder_pitch ~0 keeps the arm horizontal; elbow ~0 keeps it
# straight.
T_POSE_LEFT = (0.0, 1.4, 0.0, -0.05, 0.0, 0.0, 0.0)
T_POSE_RIGHT = (0.0, -1.4, 0.0, -0.05, 0.0, 0.0, 0.0)

# Pose 3: arms straight forward at shoulder height, parallel.
# shoulder_pitch ~ -1.5 rad lifts the arm forward to roughly horizontal;
# elbow ~ 0 keeps it almost straight. ``shoulder_roll = 0`` (was 0.2)
# so the arms point straight forward at the natural shoulder offset
# rather than slightly abducted -- this matches what operators
# instinctively do when asked to "hold arms forward parallel" and
# avoids forcing the y-axis fit to absorb a 16 cm wider robot
# reference than the operator actually demonstrates.
ARMS_FORWARD_LEFT = (-1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ARMS_FORWARD_RIGHT = (-1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Pose 4: namaste. Both hands together at the chest centerline with
# the forearms angled vertically (palms touching). This is the data
# point that anchors the y-axis fit at ``op_y = 0`` so the operator
# can actually bring the robot's hands together when they bring their
# own hands together.
#
# Why "namaste" and not "hands together at chest height":
#   * The instruction is one word and instantly recognizable across
#     cultures -- operators don't have to puzzle out elbow angles or
#     hand orientation, they just put their palms together at chest.
#   * Vertical forearms are easier to repeat consistently than
#     horizontal ones, and the resulting wrist position is genuinely
#     at the body centerline (which is what the math needs).
#
# Joint-angle choice notes:
#   * The X2 ``shoulder_roll`` joints CANNOT cross the body centerline
#     (left limit ``[-0.061, 2.993]``, right ``[-2.993, 0.061]`` in
#     radians). So adduction has to come from ``shoulder_yaw`` rotating
#     the upper-arm axis inward + elbow flexion.
#   * ``shoulder_pitch = -1.1`` raises the upper arm forward to chest
#     level (any less and the wrists end up below mid-chest).
#   * ``shoulder_yaw = -1.2`` on the LEFT (and ``+1.2`` on the RIGHT)
#     rotates the upper arm strongly inward so the elbow points
#     sideways and the forearm crosses to the body centerline. The
#     joint limit on shoulder_yaw is ±2.556 rad, so 1.2 rad is well
#     inside.
#   * ``elbow = -1.60`` flexes the forearm to bring the wrist exactly
#     to the body centerline at chest height.
#   * Forward-kinematics check: left wrist at ``(0.228, -0.001, 0.186)``
#     and right wrist at ``(0.228, +0.001, 0.186)`` in torso-link frame
#     -- 0.2 cm apart (palms-touching), at chest height (Z = 0.19 m).
NAMASTE_LEFT = (-1.1, 0.0, -1.2, -1.6, 0.0, 0.0, 0.0)
NAMASTE_RIGHT = (-1.1, 0.0, +1.2, -1.6, 0.0, 0.0, 0.0)

# Backward-compat aliases. Older code paths and tests imported the
# v1 pose name; we kept the constants live so they can be removed
# without ripple.
HANDS_TOGETHER_LEFT = NAMASTE_LEFT
HANDS_TOGETHER_RIGHT = NAMASTE_RIGHT


CALIBRATION_POSE_IDS: tuple[str, ...] = (
    "arms_down",
    "t_pose",
    "arms_forward",
    "namaste",
)


# Per-pose residual rejection thresholds in metres.
#
# These are the defaults used by :func:`try_fit_calibration` /
# :func:`fit_calibration` if the caller doesn't override them. They are
# intentionally NOT all the same value:
#
# * The X2 has a fixed shoulder offset (~14 cm half-width) and joint
#   limits that prevent the arms crossing the body centerline; meanwhile
#   real operators have differently-proportioned shoulders and forearms.
#   So the per-axis affine fit on 4 poses inevitably leaves a few cm of
#   residual on every pose -- 5 cm everywhere is unattainable in
#   practice.
#
# * The ``namaste`` pose is the most permissive of the four because the
#   operator is *holding the controllers*. There is a structural ~5-7 cm
#   offset between the controller's reported position (the grip) and
#   where a "palm-touching" pose would put the actual palm. Quest 3
#   tracking jitter at close range adds another ~2 cm. Together that
#   makes 5 cm an unrealistic gate for namaste. We allow up to 18 cm
#   here, which empirically accepts captures from operators who are
#   genuinely doing their best to put hands together while gripping
#   the controllers.
#
# Tighten or loosen these via the CLI flags on
# ``vr_operator_calibrate.py`` if needed for a particular setup.
# Default per-pose residual rejection thresholds (meters).
#
# Empirically-derived floor for the per-axis affine model
# ``p_robot = diag(s) * p_op + t``. The model has 6 unknowns per arm
# (3 axes × 2 unknowns each), and human-vs-X2 anatomy ratios are
# rarely separable into per-axis scales, so even a clean, well-held
# 4-pose calibration produces 9-13 cm residuals on at least one
# (pose, side) cell. Setting the gate at 10 cm rejected
# well-executed captures (e.g. 9.6 cm L / 12.0 cm R on arms_down
# from a noise-free 1 mm-spread capture). 15 cm is the realistic
# floor; namaste gets 20 cm because both wrists meet near the body
# centerline (op_y ≈ 0) which is a structurally singular row in the
# y-axis fit and adds another ~5 cm bias on top of the per-axis
# anatomy mismatch.
#
# If/when we replace the per-axis fit with full Procrustes
# (rotation + uniform scale + translation), these gates can drop
# back to 5-8 cm because that model handles anatomy mismatch
# without per-axis degeneracy.
DEFAULT_POSE_RESIDUAL_REJECT_M: dict[str, float] = {
    "arms_down": 0.15,
    "t_pose": 0.15,
    "arms_forward": 0.15,
    "namaste": 0.20,
}


def _normalize_residual_reject(
    value: float | dict[str, float] | None,
) -> dict[str, float]:
    """Normalize the ``residual_reject_m`` argument to a per-pose dict.

    Accepts:

    * ``None`` -- use :data:`DEFAULT_POSE_RESIDUAL_REJECT_M` for every pose.
    * ``float`` -- single uniform threshold for every pose.
    * ``dict`` -- per-pose overrides; missing keys fall back to
      :data:`DEFAULT_POSE_RESIDUAL_REJECT_M`.

    Always returns a dict with one entry per pose in
    :data:`CALIBRATION_POSE_IDS`.
    """
    if value is None:
        return dict(DEFAULT_POSE_RESIDUAL_REJECT_M)
    if isinstance(value, (int, float)):
        return {p: float(value) for p in CALIBRATION_POSE_IDS}
    if isinstance(value, dict):
        out = dict(DEFAULT_POSE_RESIDUAL_REJECT_M)
        for k, v in value.items():
            if k not in CALIBRATION_POSE_IDS:
                raise ValueError(
                    f"residual_reject_m has key {k!r} which is not a "
                    f"calibration pose; valid keys: {CALIBRATION_POSE_IDS}"
                )
            out[k] = float(v)
        return out
    raise TypeError(
        f"residual_reject_m must be float | dict | None; got {type(value)}"
    )


ROBOT_REFERENCE_Q_RAD: dict[str, dict[str, np.ndarray]] = {
    "arms_down": {
        "left": np.array(ARMS_DOWN_LEFT, dtype=np.float64),
        "right": np.array(ARMS_DOWN_RIGHT, dtype=np.float64),
    },
    "t_pose": {
        "left": np.array(T_POSE_LEFT, dtype=np.float64),
        "right": np.array(T_POSE_RIGHT, dtype=np.float64),
    },
    "arms_forward": {
        "left": np.array(ARMS_FORWARD_LEFT, dtype=np.float64),
        "right": np.array(ARMS_FORWARD_RIGHT, dtype=np.float64),
    },
    "namaste": {
        "left": np.array(NAMASTE_LEFT, dtype=np.float64),
        "right": np.array(NAMASTE_RIGHT, dtype=np.float64),
    },
}


# Human-readable instructions surfaced by the calibration script (also
# spoken via TTS in the WebXR client).
CALIBRATION_POSE_INSTRUCTIONS: dict[str, str] = {
    "arms_down": (
        "Stand relaxed with both arms hanging fully straight down at your "
        "sides. Do not bend the elbows. Palms face inward toward your thighs. "
        "Hold the controllers the same way you will during teleop. "
        "Press A on either controller when ready."
    ),
    "t_pose": (
        "Raise both arms straight out sideways, parallel to the floor. "
        "Palms face down toward the floor. "
        "Press A when steady."
    ),
    "arms_forward": (
        "Hold both arms straight out forward at shoulder height. "
        "Keep your hands close together, about as wide as your shoulders. "
        "Palms face each other (inward). "
        "Press A when steady."
    ),
    "namaste": (
        "Bring both palms together at your chest in a namaste pose. "
        "Forearms vertical, palms touching. "
        "Press A when steady."
    ),
}


def robot_reference_wrist_positions() -> dict[str, dict[str, np.ndarray]]:
    """FK-computed wrist positions in ``torso_link`` for each calibration pose.

    Returned dict is keyed by ``pose_id -> side -> (3,) xyz``. These are
    the targets the operator's measured wrists are fitted against.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for pose_id, by_side in ROBOT_REFERENCE_Q_RAD.items():
        out[pose_id] = {
            side: arm_fk(q, side=side).astype(np.float64)
            for side, q in by_side.items()
        }
    return out


def robot_reference_wrist_quats() -> dict[str, dict[str, np.ndarray]]:
    """FK-computed wrist orientation quats (wxyz) in ``torso_link`` per pose.

    Used to fit an operator -> robot wrist orientation alignment from the
    arms-down pose. We only currently use ``arms_down`` for the fit but
    pre-compute all three so the YAML carries the full reference.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for pose_id, by_side in ROBOT_REFERENCE_Q_RAD.items():
        out[pose_id] = {}
        for side, q in by_side.items():
            T = arm_fk_pose(q, side=side)
            R = T[:3, :3]
            out[pose_id][side] = _matrix_to_quat_wxyz(R)
    return out


@dataclass
class PoseMeasurement:
    """Operator wrist samples captured during one calibration pose.

    Stored separately for left/right so we can reject one arm and
    re-capture it without redoing both.

    ``left_wrist_quat_head_yaw`` / ``right_wrist_quat_head_yaw`` carry
    the *median* wrist orientation, expressed in the operator's
    head-yaw frame. They're optional (default identity) for backward
    compat with v0 calibration YAMLs that did not record orientation.
    """

    pose_id: str
    left_wrist_mean: np.ndarray  # (3,) operator wrist in head-yaw frame
    right_wrist_mean: np.ndarray
    sample_count: int
    left_wrist_vel_rms_mps: float
    right_wrist_vel_rms_mps: float
    left_wrist_quat_head_yaw: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    right_wrist_quat_head_yaw: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "left_wrist": [float(v) for v in self.left_wrist_mean],
            "right_wrist": [float(v) for v in self.right_wrist_mean],
            "samples": int(self.sample_count),
            "left_vel_rms_mps": float(self.left_wrist_vel_rms_mps),
            "right_vel_rms_mps": float(self.right_wrist_vel_rms_mps),
            "left_wrist_quat_wxyz": [float(v) for v in self.left_wrist_quat_head_yaw],
            "right_wrist_quat_wxyz": [float(v) for v in self.right_wrist_quat_head_yaw],
        }

    @classmethod
    def from_yaml_dict(cls, pose_id: str, data: dict[str, Any]) -> "PoseMeasurement":
        return cls(
            pose_id=pose_id,
            left_wrist_mean=np.asarray(data["left_wrist"], dtype=np.float64),
            right_wrist_mean=np.asarray(data["right_wrist"], dtype=np.float64),
            sample_count=int(data.get("samples", 0)),
            left_wrist_vel_rms_mps=float(data.get("left_vel_rms_mps", 0.0)),
            right_wrist_vel_rms_mps=float(data.get("right_vel_rms_mps", 0.0)),
            left_wrist_quat_head_yaw=np.asarray(
                data.get("left_wrist_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            ),
            right_wrist_quat_head_yaw=np.asarray(
                data.get("right_wrist_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            ),
        )


@dataclass
class ArmFit:
    """Per-arm calibration fit.

    ``wrist_alignment_quat`` carries the rotation that takes an
    operator wrist quaternion (already expressed in the head-yaw frame)
    into the robot wrist's torso-frame orientation. It's derived from
    the arms-down pose: at that pose we know the robot wrist quat from
    FK, and the operator's wrist quat is whatever the controller / hand
    tracker reported. The product of the robot-quat and the inverse of
    the operator-quat is the alignment.

    ``op_quat_offset_rpy_deg`` is an optional per-operator constant
    rotation (intrinsic XYZ Tait-Bryan ``(roll, pitch, yaw)`` degrees)
    that compensates for the Quest 3 controller's fixed grip tilt: the
    way the controller body angles relative to the operator's actual
    wrist in a closed-fist grip. At runtime ``VRArmTeleopCalibrated``
    post-multiplies this offset on the operator's head-yaw-frame wrist
    quat **before** ``wrist_alignment_quat`` runs, so a single set of
    values applies to every launcher (kinematic teleop, dataset
    recording, VLA bridge) without per-script CLI plumbing.

    Defaults: identity ``wrist_alignment_quat`` and zero
    ``op_quat_offset_rpy_deg`` for backward compat with v0 / v1
    calibration YAMLs (which give correct *position* targets but
    identity orientation, so callers can still run with
    ``rotation_weight=0`` and no per-operator offset).
    """

    scale: np.ndarray  # (3,) per-axis multiplicative factor
    translation: np.ndarray  # (3,) per-axis offset (meters)
    residual_m: float  # max per-pose Euclidean error after fit
    wrist_alignment_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    op_quat_offset_rpy_deg: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def apply(self, op_wrist_in_head_yaw_frame: np.ndarray) -> np.ndarray:
        """Map an operator wrist position to a robot wrist target.

        Args:
            op_wrist_in_head_yaw_frame: ``(3,)`` operator wrist in the
                head-yaw frame (meters).

        Returns:
            ``(3,)`` robot wrist target in ``torso_link`` frame (meters).
        """
        v = np.asarray(op_wrist_in_head_yaw_frame, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError(f"expected (3,) wrist; got {v.shape}")
        return self.scale * v + self.translation

    def apply_quat(self, op_wrist_quat_in_head_yaw_frame: np.ndarray) -> np.ndarray:
        """Map an operator wrist quat to a robot wrist target quat.

        Args:
            op_wrist_quat_in_head_yaw_frame: ``(4,)`` operator wrist
                quaternion ``[w, x, y, z]`` in the head-yaw frame.

        Returns:
            ``(4,)`` target wrist quaternion ``[w, x, y, z]`` in
            ``torso_link`` frame.
        """
        q = np.asarray(op_wrist_quat_in_head_yaw_frame, dtype=np.float64)
        if q.shape != (4,):
            raise ValueError(f"expected (4,) quat; got {q.shape}")
        return _quat_multiply_wxyz(self.wrist_alignment_quat, q)

    def to_yaml_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scale": [float(v) for v in self.scale],
            "translation": [float(v) for v in self.translation],
            "residual_m": float(self.residual_m),
            "wrist_alignment_quat_wxyz": [
                float(v) for v in self.wrist_alignment_quat
            ],
        }
        # Omit the offset when zero so v1 YAMLs round-trip unchanged
        # (op_quat_offset_rpy_deg defaults to zeros on load).
        if np.any(self.op_quat_offset_rpy_deg):
            out["op_quat_offset_rpy_deg"] = [
                float(v) for v in self.op_quat_offset_rpy_deg
            ]
        return out

    @classmethod
    def from_yaml_dict(cls, data: dict[str, Any]) -> "ArmFit":
        return cls(
            scale=np.asarray(data["scale"], dtype=np.float64),
            translation=np.asarray(data["translation"], dtype=np.float64),
            residual_m=float(data["residual_m"]),
            wrist_alignment_quat=np.asarray(
                data.get("wrist_alignment_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            ),
            op_quat_offset_rpy_deg=np.asarray(
                data.get("op_quat_offset_rpy_deg", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ),
        )


@dataclass
class HandRangeFit:
    """Per-side, per-finger raw-curl ``(floor, ceiling)`` plus
    thumb-opposition ``(floor, ceiling)`` for one hand.

    Used by :func:`per_finger_grasp_command_from_curls_and_oppose` to
    affine-rescale Quest 3 curl signals onto the operator's actual
    ergonomic range. See the module docstring of
    ``gear_sonic.utils.teleop.x2_hand_retarget`` for why this is
    needed and how the ranges are interpreted.

    All five floor / ceiling values are in ``[0, 1]`` and ordered
    ``[thumb, index, middle, ring, pinky]`` to match the Quest 3
    ``XRHand`` curl array.
    """

    floor: np.ndarray  # (5,) float64 in [0, 1]
    ceiling: np.ndarray  # (5,) float64 in [0, 1], strictly > floor element-wise
    oppose_floor: float
    oppose_ceiling: float

    def __post_init__(self) -> None:
        self.floor = np.asarray(self.floor, dtype=np.float64).reshape(5)
        self.ceiling = np.asarray(self.ceiling, dtype=np.float64).reshape(5)
        if not (self.ceiling > self.floor).all():
            raise ValueError(
                f"need ceiling > floor element-wise; got "
                f"floor={self.floor}, ceiling={self.ceiling}"
            )
        if not (0.0 <= self.oppose_floor < self.oppose_ceiling <= 1.0):
            raise ValueError(
                f"need 0 <= oppose_floor < oppose_ceiling <= 1; got "
                f"floor={self.oppose_floor}, ceiling={self.oppose_ceiling}"
            )

    def to_yaml_dict(self) -> dict:
        return {
            "floor": [float(v) for v in self.floor],
            "ceiling": [float(v) for v in self.ceiling],
            "oppose_floor": float(self.oppose_floor),
            "oppose_ceiling": float(self.oppose_ceiling),
        }

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "HandRangeFit":
        return cls(
            floor=np.asarray(raw["floor"], dtype=np.float64),
            ceiling=np.asarray(raw["ceiling"], dtype=np.float64),
            oppose_floor=float(raw["oppose_floor"]),
            oppose_ceiling=float(raw["oppose_ceiling"]),
        )


@dataclass
class HandRangeCalibration:
    """Both hands' :class:`HandRangeFit` in one bundle, written into
    the operator-calibration YAML under the optional ``hand_range``
    section.
    """

    left: HandRangeFit
    right: HandRangeFit
    source: str = ""  # free-form provenance ("npz:<path>", "live-capture", ...)
    samples: int = 0  # number of frames the fit was estimated from

    def to_yaml_dict(self) -> dict:
        return {
            "left": self.left.to_yaml_dict(),
            "right": self.right.to_yaml_dict(),
            "source": self.source,
            "samples": int(self.samples),
        }

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "HandRangeCalibration":
        return cls(
            left=HandRangeFit.from_yaml_dict(raw["left"]),
            right=HandRangeFit.from_yaml_dict(raw["right"]),
            source=str(raw.get("source", "")),
            samples=int(raw.get("samples", 0)),
        )


@dataclass
class OperatorCalibration:
    """Complete per-operator calibration, ready to save / load / apply.

    ``measurements`` and ``fit`` are kept together so the YAML doubles as
    a debugging artifact -- you can re-fit later if the formulation
    changes without re-recording the operator.

    ``hand_range`` is an optional bundle of per-finger raw-curl ranges
    used by the OmniHand retargeting to remove Quest 3's resting-bias
    and partial-fist-ceiling artifacts. When absent (legacy YAMLs), the
    retargeting falls back to the linear pass-through (raw curl is
    already in ``[0, 1]``); when present, the retargeting affine-rescales
    each finger using its operator-specific ``(floor, ceiling)``. This
    field is opt-in -- adding it does NOT break older YAMLs that omit
    it.
    """

    operator_id: str
    measurements: dict[str, PoseMeasurement]
    fit: dict[str, ArmFit]  # keys: "left", "right"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = SCHEMA_VERSION
    units: str = "meters"
    notes: str = ""
    hand_range: HandRangeCalibration | None = None

    def apply_to_wrist(self, op_wrist_in_head_yaw_frame: np.ndarray, side: str) -> np.ndarray:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        return self.fit[side].apply(op_wrist_in_head_yaw_frame)

    def apply_to_wrist_quat(
        self,
        op_wrist_quat_in_head_yaw_frame: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """Map an operator wrist quat (head-yaw frame) to a robot target quat."""
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        return self.fit[side].apply_quat(op_wrist_quat_in_head_yaw_frame)

    def save_yaml(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "schema_version": self.schema_version,
            "operator_id": self.operator_id,
            "created_utc": self.created_utc,
            "units": self.units,
            "notes": self.notes,
            "poses": {
                pose_id: m.to_yaml_dict()
                for pose_id, m in self.measurements.items()
            },
            "robot_reference_q_rad": {
                pose_id: {
                    side: [float(v) for v in q]
                    for side, q in by_side.items()
                }
                for pose_id, by_side in ROBOT_REFERENCE_Q_RAD.items()
            },
            "fit": {side: f.to_yaml_dict() for side, f in self.fit.items()},
        }
        if self.hand_range is not None:
            payload["hand_range"] = self.hand_range.to_yaml_dict()
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return path

    @classmethod
    def load_yaml(cls, path: Path | str) -> "OperatorCalibration":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"calibration not found: {path}")
        data = yaml.safe_load(path.read_text())
        sv = int(data.get("schema_version", 0))
        if sv != SCHEMA_VERSION:
            raise ValueError(
                f"calibration {path} has schema_version={sv}, expected "
                f"{SCHEMA_VERSION}; recapture with vr_operator_calibrate.py."
            )
        measurements = {
            pose_id: PoseMeasurement.from_yaml_dict(pose_id, raw)
            for pose_id, raw in data["poses"].items()
        }
        fit = {side: ArmFit.from_yaml_dict(raw) for side, raw in data["fit"].items()}
        hand_range_raw = data.get("hand_range")
        hand_range = (
            HandRangeCalibration.from_yaml_dict(hand_range_raw)
            if hand_range_raw is not None
            else None
        )
        return cls(
            operator_id=str(data.get("operator_id", "default")),
            measurements=measurements,
            fit=fit,
            created_utc=str(data.get("created_utc", "")),
            schema_version=sv,
            units=str(data.get("units", "meters")),
            notes=str(data.get("notes", "")),
            hand_range=hand_range,
        )


def _fit_per_axis(
    op: np.ndarray, robot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Closed-form least-squares fit of ``robot = diag(s) * op + t`` per axis.

    Args:
        op:    ``(N, 3)`` operator wrist positions (meters).
        robot: ``(N, 3)`` matching robot FK wrist positions (meters).

    Returns:
        ``(scale (3,), translation (3,), max_residual_m, per_sample_residual_m)``
        where ``max_residual_m`` is the worst-case per-sample Euclidean
        distance between the predicted and target robot wrist after
        fitting, and ``per_sample_residual_m`` is the ``(N,)`` array of
        per-sample distances (so callers can identify *which* pose
        contributed the worst error).
    """
    if op.shape != robot.shape or op.ndim != 2 or op.shape[1] != 3:
        raise ValueError(f"op/robot must both be (N, 3); got {op.shape}, {robot.shape}")
    n = op.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 poses to fit; got {n}")

    scale = np.zeros(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for ax in range(3):
        # Solve robot[:, ax] = scale[ax] * op[:, ax] + translation[ax]
        # via numpy lstsq on the design matrix [op, 1].
        A = np.column_stack([op[:, ax], np.ones(n, dtype=np.float64)])
        b = robot[:, ax]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        scale[ax] = sol[0]
        translation[ax] = sol[1]

    pred = op * scale + translation  # (N, 3)
    resid = np.linalg.norm(pred - robot, axis=1)  # (N,)
    return scale, translation, float(resid.max()), resid.astype(np.float64)


@dataclass
class CalibrationFitResult:
    """Outcome of a non-raising calibration fit attempt.

    ``calibration`` is *always* populated -- even when the fit is
    rejected. That lets callers (the ``vr_operator_calibrate`` script)
    inspect the per-pose residual breakdown and pinpoint exactly which
    pose was the outlier so the operator can recapture only that one,
    instead of crashing the whole flow.

    ``per_pose_residual_m`` maps ``pose_id -> side -> residual_m`` so
    the worst pose can be identified (e.g. T-pose right-arm with the
    user's right wrist 17 cm forward will contribute the largest
    residual on ``side='right'`` in pose ``'t_pose'``).
    """

    calibration: "OperatorCalibration"
    per_pose_residual_m: dict[str, dict[str, float]]
    accepted: bool
    rejected_side: str | None  # "left", "right", or None
    rejected_residual_m: float | None
    rejected_pose: str | None  # which pose (if any) drove the rejection
    residual_reject_m: dict[str, float]  # per-pose threshold actually used

    def worst_pose_for_side(self, side: str) -> tuple[str, float]:
        """Return ``(pose_id, residual_m)`` of the worst pose on ``side``."""
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        worst_pose = max(
            CALIBRATION_POSE_IDS,
            key=lambda p: self.per_pose_residual_m.get(p, {}).get(side, 0.0),
        )
        return worst_pose, float(self.per_pose_residual_m[worst_pose][side])

    def worst_pose_overall(self) -> tuple[str, str, float]:
        """Return ``(pose_id, side, residual_m)`` of the largest-residual pose."""
        best_pose = CALIBRATION_POSE_IDS[0]
        best_side = "left"
        best_resid = -1.0
        for pose_id in CALIBRATION_POSE_IDS:
            for side in ("left", "right"):
                r = self.per_pose_residual_m.get(pose_id, {}).get(side, 0.0)
                if r > best_resid:
                    best_resid = r
                    best_pose = pose_id
                    best_side = side
        return best_pose, best_side, float(best_resid)


def try_fit_calibration(
    measurements: dict[str, PoseMeasurement],
    *,
    operator_id: str = "default",
    robot_ref: dict[str, dict[str, np.ndarray]] | None = None,
    robot_quat_ref_override: dict[str, dict[str, np.ndarray]] | None = None,
    residual_reject_m: float | dict[str, float] | None = None,
    notes: str = "",
) -> CalibrationFitResult:
    """Non-raising variant of :func:`fit_calibration`.

    ``residual_reject_m`` accepts:

    * ``None`` (default) -- use the per-pose defaults
      (:data:`DEFAULT_POSE_RESIDUAL_REJECT_M`); namaste is intentionally
      more permissive than the other poses because operators hold
      the controllers (palm-grip offset).
    * ``float`` -- one uniform threshold for every pose.
    * ``dict`` -- per-pose overrides; missing keys fall back to the
      per-pose defaults.

    Always returns a populated :class:`CalibrationFitResult` so callers
    can choose how to handle a high-residual fit. The
    :func:`fit_calibration` strict variant remains for tests and
    library callers that just want a yes/no answer.
    """
    missing = [p for p in CALIBRATION_POSE_IDS if p not in measurements]
    if missing:
        raise ValueError(
            f"missing calibration poses: {missing}. Required: "
            f"{list(CALIBRATION_POSE_IDS)}."
        )

    if robot_ref is None:
        robot_ref = robot_reference_wrist_positions()

    robot_quat_ref = (
        robot_quat_ref_override
        if robot_quat_ref_override is not None
        else robot_reference_wrist_quats()
    )

    threshold_by_pose = _normalize_residual_reject(residual_reject_m)

    fit: dict[str, ArmFit] = {}
    per_pose_residual_m: dict[str, dict[str, float]] = {
        p: {} for p in CALIBRATION_POSE_IDS
    }
    rejected_side: str | None = None
    rejected_residual_m: float | None = None
    rejected_pose: str | None = None

    for side in ("left", "right"):
        op_pts = np.stack(
            [
                measurements[p].left_wrist_mean if side == "left" else measurements[p].right_wrist_mean
                for p in CALIBRATION_POSE_IDS
            ],
            axis=0,
        )
        robot_pts = np.stack(
            [robot_ref[p][side] for p in CALIBRATION_POSE_IDS], axis=0
        )
        scale, translation, max_resid, per_sample = _fit_per_axis(op_pts, robot_pts)

        for i, pose_id in enumerate(CALIBRATION_POSE_IDS):
            per_pose_residual_m[pose_id][side] = float(per_sample[i])

        # Per-pose rejection: each pose has its own threshold so a
        # naturally-fuzzy pose (namaste) doesn't gate the whole fit
        # at the strictness of a precise pose (T-pose).
        for i, pose_id in enumerate(CALIBRATION_POSE_IDS):
            threshold = threshold_by_pose[pose_id]
            if per_sample[i] > threshold and rejected_side is None:
                rejected_side = side
                rejected_residual_m = float(per_sample[i])
                rejected_pose = pose_id
                break  # one rejection is enough; let the caller fix it

        # Wrist orientation alignment from the arms-down pose only.
        arms_down = measurements["arms_down"]
        op_quat = (
            arms_down.left_wrist_quat_head_yaw
            if side == "left"
            else arms_down.right_wrist_quat_head_yaw
        )
        robot_quat = robot_quat_ref["arms_down"][side]
        wrist_alignment = _quat_multiply_wxyz(robot_quat, _quat_inverse_wxyz(op_quat))

        fit[side] = ArmFit(
            scale=scale,
            translation=translation,
            residual_m=float(max_resid),
            wrist_alignment_quat=wrist_alignment,
        )

    cal = OperatorCalibration(
        operator_id=operator_id,
        measurements=dict(measurements),
        fit=fit,
        notes=notes,
    )
    return CalibrationFitResult(
        calibration=cal,
        per_pose_residual_m=per_pose_residual_m,
        accepted=(rejected_side is None),
        rejected_side=rejected_side,
        rejected_residual_m=rejected_residual_m,
        rejected_pose=rejected_pose,
        residual_reject_m=threshold_by_pose,
    )


def fit_calibration(
    measurements: dict[str, PoseMeasurement],
    *,
    operator_id: str = "default",
    robot_ref: dict[str, dict[str, np.ndarray]] | None = None,
    robot_quat_ref_override: dict[str, dict[str, np.ndarray]] | None = None,
    residual_reject_m: float | dict[str, float] | None = None,
    notes: str = "",
) -> OperatorCalibration:
    """Fit per-arm calibration from the canonical poses.

    Args:
        measurements: dict ``pose_id -> PoseMeasurement`` covering every
            pose in :data:`CALIBRATION_POSE_IDS`.
        operator_id: free-form label stamped into the YAML.
        robot_ref: pre-computed FK reference; overrideable for tests.
            Defaults to :func:`robot_reference_wrist_positions`.
        residual_reject_m: per-pose residual threshold(s). See
            :func:`try_fit_calibration` for the full accepted set
            (``None``, ``float``, ``dict``). ``None`` uses the
            per-pose defaults from
            :data:`DEFAULT_POSE_RESIDUAL_REJECT_M`.
        notes: optional free-form notes saved into the YAML (e.g.,
            "Quest 3 controllers, second attempt").

    Returns:
        Populated :class:`OperatorCalibration` ready for :meth:`save_yaml`.

    Raises:
        ValueError: if any per-arm residual exceeds the matching
            per-pose threshold. Use :func:`try_fit_calibration` for
            a non-raising variant that surfaces per-pose diagnostics
            so the caller can do a targeted recapture instead of
            bailing.
    """
    result = try_fit_calibration(
        measurements,
        operator_id=operator_id,
        robot_ref=robot_ref,
        robot_quat_ref_override=robot_quat_ref_override,
        residual_reject_m=residual_reject_m,
        notes=notes,
    )
    if not result.accepted:
        side = result.rejected_side or "?"
        residual = result.rejected_residual_m or 0.0
        pose = result.rejected_pose or "?"
        threshold = result.residual_reject_m.get(pose, 0.0)
        raise ValueError(
            f"{side}-arm calibration residual {residual:.3f} m on pose "
            f"{pose!r} exceeds threshold {threshold:.3f} m. Re-capture; "
            f"likely cause: operator moved during capture, controller "
            f"dropouts, or pose was wildly off-target. Per-pose "
            f"thresholds: {result.residual_reject_m}."
        )
    return result.calibration


# ── Small quaternion helpers (wxyz convention) ────────────────────────────


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a ``[w, x, y, z]`` quaternion."""
    from scipy.spatial.transform import Rotation as sRot

    return sRot.from_matrix(np.asarray(R, dtype=np.float64)).as_quat(scalar_first=True)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_inverse_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (conjugate / norm^2). Assumes wxyz order."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"expected (4,) quat; got {q.shape}")
    n2 = float(np.dot(q, q))
    if n2 < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64) / n2


def _quat_multiply_wxyz(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
    """Hamilton product: returns ``a * b`` in ``[w, x, y, z]`` order."""
    a = np.asarray(a_wxyz, dtype=np.float64)
    b = np.asarray(b_wxyz, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError(f"expected (4,) quats; got {a.shape}, {b.shape}")
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def head_yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    """Extract the yaw component (rotation about robot +Z) from a quaternion.

    We want to discard head pitch/roll so looking up/down or tilting the
    head sideways doesn't drag the robot wrist target. Yaw is the
    operator's body-facing direction.

    Args:
        quat_wxyz: ``(4,)`` quaternion ``[w, x, y, z]`` (robot frame).

    Returns:
        yaw angle in radians.
    """
    from scipy.spatial.transform import Rotation as sRot

    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"expected (4,) quat_wxyz; got {q.shape}")
    R = sRot.from_quat(q, scalar_first=True)
    yaw = R.as_euler("zyx", degrees=False)[0]
    return float(yaw)


def wrist_to_head_yaw_frame(
    wrist_robot_frame: np.ndarray,
    head_robot_frame: np.ndarray,
    head_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Transform a wrist position into the operator's head-yaw frame.

    The head-yaw frame is centered at ``head_robot_frame`` with its X
    axis along the operator's forward-facing direction (head yaw),
    discarding head pitch and roll. This is the frame the operator
    naturally moves their wrists in -- left/right of "where I'm
    looking" rather than world frame.

    Note: in the live pipeline ``head_robot_frame.xy`` is always 0 by
    construction (see :func:`compute_3pt_pose_from_quest3`), so the
    subtraction is a no-op for x/y. We still include it to stay correct
    if the upstream coordinate convention ever changes.

    Args:
        wrist_robot_frame: ``(3,)`` operator wrist xyz (robot frame).
        head_robot_frame:  ``(3,)`` operator head xyz (robot frame).
        head_quat_wxyz:    ``(4,)`` head orientation quaternion
                           ``[w, x, y, z]`` (robot frame).

    Returns:
        ``(3,)`` wrist position in the head-yaw frame.
    """
    yaw = head_yaw_from_quat(head_quat_wxyz)
    rel = np.asarray(wrist_robot_frame, dtype=np.float64) - np.asarray(
        head_robot_frame, dtype=np.float64
    )
    c, s = np.cos(-yaw), np.sin(-yaw)
    out = np.empty(3, dtype=np.float64)
    out[0] = c * rel[0] - s * rel[1]
    out[1] = s * rel[0] + c * rel[1]
    out[2] = rel[2]
    return out


def wrist_quat_to_head_yaw_frame(
    wrist_quat_wxyz_robot_frame: np.ndarray,
    head_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Re-express a wrist quaternion in the operator's head-yaw frame.

    Args:
        wrist_quat_wxyz_robot_frame: ``(4,)`` wrist orientation in robot
            frame (the way the WebXR client publishes it).
        head_quat_wxyz: ``(4,)`` head orientation in robot frame.

    Returns:
        ``(4,)`` wrist orientation in the head-yaw frame, i.e. with
        head pitch and roll discarded but head yaw factored out.
    """
    yaw = head_yaw_from_quat(head_quat_wxyz)
    half = 0.5 * yaw
    q_yaw = np.array(
        [np.cos(half), 0.0, 0.0, np.sin(half)],
        dtype=np.float64,
    )
    q_yaw_inv = _quat_inverse_wxyz(q_yaw)
    return _quat_multiply_wxyz(
        q_yaw_inv,
        _quat_normalize(np.asarray(wrist_quat_wxyz_robot_frame, dtype=np.float64)),
    )


__all__ = [
    "SCHEMA_VERSION",
    "CALIBRATION_POSE_IDS",
    "CALIBRATION_POSE_INSTRUCTIONS",
    "DEFAULT_POSE_RESIDUAL_REJECT_M",
    "ROBOT_REFERENCE_Q_RAD",
    "ARMS_DOWN_LEFT",
    "ARMS_DOWN_RIGHT",
    "T_POSE_LEFT",
    "T_POSE_RIGHT",
    "ARMS_FORWARD_LEFT",
    "ARMS_FORWARD_RIGHT",
    "NAMASTE_LEFT",
    "NAMASTE_RIGHT",
    "HANDS_TOGETHER_LEFT",
    "HANDS_TOGETHER_RIGHT",
    "PoseMeasurement",
    "ArmFit",
    "HandRangeFit",
    "HandRangeCalibration",
    "OperatorCalibration",
    "CalibrationFitResult",
    "robot_reference_wrist_positions",
    "robot_reference_wrist_quats",
    "fit_calibration",
    "try_fit_calibration",
    "head_yaw_from_quat",
    "wrist_to_head_yaw_frame",
    "wrist_quat_to_head_yaw_frame",
]
