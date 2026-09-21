"""X2 Ultra supplemental info: 31-DOF body + 20-DOF OmniHand action space.

Two hand variants are supported via the ``HandVariant`` enum:

- ``HandVariant.OMNIHAND_10`` -- full 10-DOF AgiBot OmniHand-2025 (per side),
  matching ``agitbot-x2-record-and-replay`` constants exactly. This is the
  primary v0 target for X2 manipulation.
- ``HandVariant.G1_COMPAT_7`` -- a 7-DOF subset (per side) that mirrors
  Unitree G1's three-finger hand action layout. Used for cross-embodiment
  evaluation against the SONIC reference embodiment ``unitree_g1_sonic``.

Important architectural note
----------------------------
Unlike Unitree G1, whose URDF (``g1_29dof_with_hand.urdf``) kinematically
models hand finger joints, the AgiBot X2 Ultra URDF
(``gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra.urdf``)
contains only the 31 body DOFs (legs, waist, arms, head). Finger control
on X2 goes through the AimDK HAL (``/aima/hal/joint/hand/command``) and
is opaque to Pinocchio FK. Therefore:

- ``body_actuated_joints`` enumerates exactly the 31 URDF joints so
  ``RobotModel.dof_index(name)`` resolves cleanly via Pinocchio.
- ``left_hand_actuated_joints`` and ``right_hand_actuated_joints`` are
  intentionally empty here. Hand joint *names* are still exposed via
  ``OMNIHAND_LEFT_FINGER_NAMES`` / ``OMNIHAND_RIGHT_FINGER_NAMES`` and
  ``HAND_JOINT_LIMITS`` for downstream consumers (LeRobot dataset
  encoders, ZMQ publishers, modality config builders) that operate on
  hand joints without needing kinematic FK.
- ``joint_groups["left_hand"]`` / ``joint_groups["right_hand"]`` deliberately
  return empty join lists, matching the empty hand actuated lists. Use
  the dedicated finger-name constants instead.

Joint limits and DOF ordering are sourced from:

- Body 31 DOFs: parsed from ``x2_ultra.urdf`` ``<limit lower upper effort>`` tags.
- Hand 20 DOFs: ``agitbot-x2-record-and-replay/src/x2_recorder/constants.py``
  (the production-tested teleop / replay codebase).

If you need to refresh either source, the parsing one-liner is documented in
``docs/source/references/x2_isaac_groot_data_contract.md``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np

from gear_sonic.data.robot_model.supplemental_info.robot_supplemental_info import (
    RobotSupplementalInfo,
)


class HandVariant(Enum):
    """Selects the OmniHand action layout exposed by the supplemental info."""

    OMNIHAND_10 = "omnihand_10"
    G1_COMPAT_7 = "g1_compat_7"


# ── Body joint ordering (31 DOFs, matches x2_ultra.urdf) ────────────────────

X2_BODY_JOINT_NAMES: List[str] = [
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

# Joint limits from x2_ultra.urdf (radians).
X2_BODY_JOINT_LIMITS: Dict[str, List[float]] = {
    "left_hip_pitch_joint": [-2.704, 2.556],
    "left_hip_roll_joint": [-0.235, 2.906],
    "left_hip_yaw_joint": [-1.684, 3.43],
    "left_knee_joint": [0.0, 2.4073],
    "left_ankle_pitch_joint": [-0.803, 0.453],
    "left_ankle_roll_joint": [-0.262, 0.262],
    "right_hip_pitch_joint": [-2.704, 2.556],
    "right_hip_roll_joint": [-2.906, 0.235],
    "right_hip_yaw_joint": [-3.43, 1.684],
    "right_knee_joint": [0.0, 2.4073],
    "right_ankle_pitch_joint": [-0.803, 0.453],
    "right_ankle_roll_joint": [-0.2625, 0.2625],
    "waist_yaw_joint": [-3.43, 2.382],
    "waist_pitch_joint": [-0.314, 0.314],
    "waist_roll_joint": [-0.488, 0.488],
    "left_shoulder_pitch_joint": [-3.08, 2.04],
    "left_shoulder_roll_joint": [-0.061, 2.993],
    "left_shoulder_yaw_joint": [-2.556, 2.556],
    "left_elbow_joint": [-2.3556, 0.0],
    "left_wrist_yaw_joint": [-2.556, 2.556],
    "left_wrist_pitch_joint": [-0.558, 0.558],
    "left_wrist_roll_joint": [-1.571, 0.724],
    "right_shoulder_pitch_joint": [-3.08, 2.04],
    "right_shoulder_roll_joint": [-2.993, 0.061],
    "right_shoulder_yaw_joint": [-2.556, 2.556],
    "right_elbow_joint": [-2.3556, 0.0],
    "right_wrist_yaw_joint": [-2.556, 2.556],
    "right_wrist_pitch_joint": [-0.558, 0.558],
    "right_wrist_roll_joint": [-0.724, 1.571],
    "head_yaw_joint": [-0.366, 0.366],
    "head_pitch_joint": [-0.3838, 0.3838],
}

# ── OmniHand 10-DOF per side (matches agitbot-x2-record-and-replay) ─────────
#
# Motor axis order from Omnihand-2025-SDK / API_PYTHON.md "Joint Angle I/O Order":
#   1 thumb_roll  2 thumb_abad  3 thumb_mcp
#   4 index_abad  5 index_pip
#   6 middle_pip
#   7 ring_abad   8 ring_pip
#   9 pinky_abad 10 pinky_pip
#
# Joint names are formatted ``{side}_{finger}_joint`` per the agitbot codebase.

OMNIHAND_FINGER_NAMES_PER_SIDE: List[str] = [
    "thumb_roll",
    "thumb_abad",
    "thumb_mcp",
    "index_abad",
    "index_pip",
    "middle_pip",
    "ring_abad",
    "ring_pip",
    "pinky_abad",
    "pinky_pip",
]

OMNIHAND_LEFT_FINGER_NAMES: List[str] = [
    f"left_{n}_joint" for n in OMNIHAND_FINGER_NAMES_PER_SIDE
]
OMNIHAND_RIGHT_FINGER_NAMES: List[str] = [
    f"right_{n}_joint" for n in OMNIHAND_FINGER_NAMES_PER_SIDE
]

# Per-finger joint angle ranges (radians). Values mirror the Omnihand-2025 SDK
# tables; these are the firmware-enforced clamps, asymmetric per side.
# Derived from agitbot-x2-record-and-replay/src/x2_recorder/constants.py.
HAND_JOINT_LIMITS: Dict[str, List[float]] = {
    # Left side
    "left_thumb_roll_joint": [-0.349, 0.349],
    "left_thumb_abad_joint": [-0.785, 0.785],
    "left_thumb_mcp_joint": [0.0, 1.571],
    "left_index_abad_joint": [-0.349, 0.349],
    "left_index_pip_joint": [0.0, 1.745],
    "left_middle_pip_joint": [0.0, 1.745],
    "left_ring_abad_joint": [-0.349, 0.349],
    "left_ring_pip_joint": [0.0, 1.745],
    "left_pinky_abad_joint": [-0.349, 0.349],
    "left_pinky_pip_joint": [0.0, 1.745],
    # Right side (mirror)
    "right_thumb_roll_joint": [-0.349, 0.349],
    "right_thumb_abad_joint": [-0.785, 0.785],
    "right_thumb_mcp_joint": [0.0, 1.571],
    "right_index_abad_joint": [-0.349, 0.349],
    "right_index_pip_joint": [0.0, 1.745],
    "right_middle_pip_joint": [0.0, 1.745],
    "right_ring_abad_joint": [-0.349, 0.349],
    "right_ring_pip_joint": [0.0, 1.745],
    "right_pinky_abad_joint": [-0.349, 0.349],
    "right_pinky_pip_joint": [0.0, 1.745],
}

# G1-compatible 7-DOF subset (per side). Picks one joint from each of the 7
# G1 hand DOFs (thumb_0/1/2, index_0/1, middle_0/1) so a 10-DOF teleop
# recording can be lossily down-projected for cross-embodiment evaluation
# against ``unitree_g1_sonic``. Down-projection logic lives in the dataset
# transform layer; this enum only declares the *layout*.
G1_COMPAT_FINGER_NAMES_PER_SIDE: List[str] = [
    "thumb_roll",  # ~ thumb_0
    "thumb_abad",  # ~ thumb_1
    "thumb_mcp",   # ~ thumb_2
    "index_abad",  # ~ index_0
    "index_pip",   # ~ index_1
    "ring_abad",   # ~ middle_0 (proxy: G1 middle finger maps to X2 ring)
    "ring_pip",    # ~ middle_1
]


def _select_finger_names(variant: HandVariant) -> List[str]:
    if variant == HandVariant.OMNIHAND_10:
        return OMNIHAND_FINGER_NAMES_PER_SIDE
    if variant == HandVariant.G1_COMPAT_7:
        return G1_COMPAT_FINGER_NAMES_PER_SIDE
    raise ValueError(f"Unknown HandVariant: {variant!r}")


@dataclass
class X2UltraSupplementalInfo(RobotSupplementalInfo):
    """
    Supplemental information for the AgiBot X2 Ultra robot.

    Body kinematics use the 31-DOF URDF (``x2_ultra.urdf``); finger control
    is delegated to the AimDK OmniHand HAL (out-of-band) and exposed via
    name lists rather than Pinocchio DOFs.

    Args:
        hand_variant: Which OmniHand action layout to expose. Defaults to
            full 10-DOF (the v0 target). Use ``G1_COMPAT_7`` for cross-
            embodiment evaluation.
    """

    def __init__(
        self,
        hand_variant: HandVariant = HandVariant.OMNIHAND_10,
    ):
        finger_names = _select_finger_names(hand_variant)
        left_finger_names = [f"left_{n}_joint" for n in finger_names]
        right_finger_names = [f"right_{n}_joint" for n in finger_names]

        if hand_variant == HandVariant.OMNIHAND_10:
            name = "X2Ultra_Omnihand10"
        else:
            name = "X2Ultra_G1Compat7"

        body_actuated_joints = list(X2_BODY_JOINT_NAMES)

        # Hands are not in the URDF kinematic chain; keep these empty so
        # ``RobotModel.__init__`` does not try to look them up via Pinocchio.
        left_hand_actuated_joints: List[str] = []
        right_hand_actuated_joints: List[str] = []

        joint_limits: Dict[str, List[float]] = dict(X2_BODY_JOINT_LIMITS)
        # Expose hand joint *limits* even though they're not in the URDF;
        # downstream tooling (dataset clipping, ZMQ payload validators)
        # uses these without going through Pinocchio.
        for name_ in left_finger_names + right_finger_names:
            if name_ in HAND_JOINT_LIMITS:
                joint_limits[name_] = list(HAND_JOINT_LIMITS[name_])

        joint_groups: Dict[str, Dict[str, List[str]]] = {
            "waist": {
                "joints": ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
                "groups": [],
            },
            "head": {
                "joints": ["head_yaw_joint", "head_pitch_joint"],
                "groups": [],
            },
            "left_leg": {
                "joints": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ],
                "groups": [],
            },
            "right_leg": {
                "joints": [
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ],
                "groups": [],
            },
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            "left_arm": {
                "joints": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_yaw_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_roll_joint",
                ],
                "groups": [],
            },
            "right_arm": {
                "joints": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_yaw_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_roll_joint",
                ],
                "groups": [],
            },
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            # Hands are NOT in URDF; these groups stay empty for Pinocchio
            # consumers. Direct-name consumers should use OMNIHAND_LEFT_FINGER_NAMES /
            # OMNIHAND_RIGHT_FINGER_NAMES instead.
            "left_hand": {"joints": [], "groups": []},
            "right_hand": {"joints": [], "groups": []},
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms", "head"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "hands"]},
        }

        joint_name_mapping = {
            "waist_pitch": "waist_pitch_joint",
            "waist_roll": "waist_roll_joint",
            "waist_yaw": "waist_yaw_joint",
            "head_pitch": "head_pitch_joint",
            "head_yaw": "head_yaw_joint",
            "shoulder_pitch": {
                "left": "left_shoulder_pitch_joint",
                "right": "right_shoulder_pitch_joint",
            },
            "shoulder_roll": {
                "left": "left_shoulder_roll_joint",
                "right": "right_shoulder_roll_joint",
            },
            "shoulder_yaw": {
                "left": "left_shoulder_yaw_joint",
                "right": "right_shoulder_yaw_joint",
            },
            "elbow_pitch": {
                "left": "left_elbow_joint",
                "right": "right_elbow_joint",
            },
            "wrist_yaw": {
                "left": "left_wrist_yaw_joint",
                "right": "right_wrist_yaw_joint",
            },
            "wrist_pitch": {
                "left": "left_wrist_pitch_joint",
                "right": "right_wrist_pitch_joint",
            },
            "wrist_roll": {
                "left": "left_wrist_roll_joint",
                "right": "right_wrist_roll_joint",
            },
        }

        # X2 URDF root link is "base_link".
        root_frame_name = "base_link"

        # X2 wrist roll link names follow the same convention as the URDF
        # (verified in ``gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra.urdf``).
        hand_frame_names = {
            "left": "left_wrist_roll_link",
            "right": "right_wrist_roll_link",
        }

        # Calibration / default poses: zero pose corresponds to standing
        # upright with arms at sides per the AimDK initialization. Operators
        # who need a posed default (e.g. arms up for piano teleop) can edit
        # ``home_pose.json`` in the recorder repo; the SONIC checkpoint
        # handles balance from any reasonable starting pose.
        calibration_joint_q = {"elbow_pitch": {"left": 0.0, "right": 0.0}}
        default_joint_q: Dict[str, float] = {}

        # 90 deg Y-axis rotation: aligns hand-tracking frame (palm-forward)
        # to robot wrist frame. Inherited from G1 convention; verify against
        # Quest3 OpenXR data when M8 lands.
        hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        teleop_upper_body_motion_scale = 1.0

        super().__init__(
            name=name,
            body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits,
            joint_groups=joint_groups,
            root_frame_name=root_frame_name,
            hand_frame_names=hand_frame_names,
            calibration_joint_q=calibration_joint_q,
            joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction,
            default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
        )

        # Make the variant + finger-name lists discoverable from the
        # supplemental info (downstream consumers read these to size
        # ZMQ payloads, dataset features, etc.).
        self.hand_variant = hand_variant
        self.left_finger_names = left_finger_names
        self.right_finger_names = right_finger_names
