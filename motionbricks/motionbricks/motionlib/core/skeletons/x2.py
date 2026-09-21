from .base import SkeletonBase

# AgiBot X2 Ultra — MuJoCo body order matches gear_sonic ``MUJOCO_JOINT_NAMES`` plus
# pelvis root and two dummy toe joints (same foot-contact pattern as G1Skeleton34).


class X2Skeleton(SkeletonBase):
    bone_order_names_with_parents = []
    name = "x2skel"
    right_hand_joint_names = ["right_wrist_roll_skel"]
    left_hand_joint_names = ["left_wrist_roll_skel"]
    hip_joint_names = [
        "right_hip_pitch_skel",
        "left_hip_pitch_skel",
    ]

    def get_skel_slice(self, skeleton: SkeletonBase):
        try:
            skel_slice = [skeleton.bone_index[x] for x in self.bone_order_names]
        except KeyError as exc:
            raise ValueError(
                "The current skeleton contain joints that are not in the input"
            ) from exc
        return skel_slice


class X2Skeleton34(X2Skeleton):
    """31 actuated hinge joints + pelvis + 2 dummy toe joints (34 total).

    Feature dimensionality matches G1Skeleton34 (418-dim dual-root global rep)
    so VQVAE / backbone widths can be reused when retraining from scratch on X2.
    """

    name = "x2skel34"

    bone_order_names_with_parents = [
        ("pelvis_skel", None),
        # left leg
        ("left_hip_pitch_skel", "pelvis_skel"),
        ("left_hip_roll_skel", "left_hip_pitch_skel"),
        ("left_hip_yaw_skel", "left_hip_roll_skel"),
        ("left_knee_skel", "left_hip_yaw_skel"),
        ("left_ankle_pitch_skel", "left_knee_skel"),
        ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
        ("left_toe_base", "left_ankle_roll_skel"),
        # right leg
        ("right_hip_pitch_skel", "pelvis_skel"),
        ("right_hip_roll_skel", "right_hip_pitch_skel"),
        ("right_hip_yaw_skel", "right_hip_roll_skel"),
        ("right_knee_skel", "right_hip_yaw_skel"),
        ("right_ankle_pitch_skel", "right_knee_skel"),
        ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
        ("right_toe_base", "right_ankle_roll_skel"),
        # waist (yaw → pitch → roll/torso)
        ("waist_yaw_skel", "pelvis_skel"),
        ("waist_pitch_skel", "waist_yaw_skel"),
        ("waist_roll_skel", "waist_pitch_skel"),
        # left arm
        ("left_shoulder_pitch_skel", "waist_roll_skel"),
        ("left_shoulder_roll_skel", "left_shoulder_pitch_skel"),
        ("left_shoulder_yaw_skel", "left_shoulder_roll_skel"),
        ("left_elbow_skel", "left_shoulder_yaw_skel"),
        ("left_wrist_yaw_skel", "left_elbow_skel"),
        ("left_wrist_pitch_skel", "left_wrist_yaw_skel"),
        ("left_wrist_roll_skel", "left_wrist_pitch_skel"),
        # right arm
        ("right_shoulder_pitch_skel", "waist_roll_skel"),
        ("right_shoulder_roll_skel", "right_shoulder_pitch_skel"),
        ("right_shoulder_yaw_skel", "right_shoulder_roll_skel"),
        ("right_elbow_skel", "right_shoulder_yaw_skel"),
        ("right_wrist_yaw_skel", "right_elbow_skel"),
        ("right_wrist_pitch_skel", "right_wrist_yaw_skel"),
        ("right_wrist_roll_skel", "right_wrist_pitch_skel"),
        # head
        ("head_yaw_skel", "waist_roll_skel"),
        ("head_pitch_skel", "head_yaw_skel"),
    ]

    right_foot_joint_names = ["right_ankle_roll_skel", "right_toe_base"]
    left_foot_joint_names = ["left_ankle_roll_skel", "left_toe_base"]


# MuJoCo body used for FK when converting X2 motion-lib PKL clips.
X2_SKEL_TO_MUJOCO_BODY = {
    "pelvis_skel": "pelvis",
    "left_hip_pitch_skel": "left_hip_pitch_link",
    "left_hip_roll_skel": "left_hip_roll_link",
    "left_hip_yaw_skel": "left_hip_yaw_link",
    "left_knee_skel": "left_knee_link",
    "left_ankle_pitch_skel": "left_ankle_pitch_link",
    "left_ankle_roll_skel": "left_ankle_roll_link",
    "left_toe_base": "left_ankle_roll_link",
    "right_hip_pitch_skel": "right_hip_pitch_link",
    "right_hip_roll_skel": "right_hip_roll_link",
    "right_hip_yaw_skel": "right_hip_yaw_link",
    "right_knee_skel": "right_knee_link",
    "right_ankle_pitch_skel": "right_ankle_pitch_link",
    "right_ankle_roll_skel": "right_ankle_roll_link",
    "right_toe_base": "right_ankle_roll_link",
    "waist_yaw_skel": "waist_yaw_link",
    "waist_pitch_skel": "waist_pitch_link",
    "waist_roll_skel": "torso_link",
    "left_shoulder_pitch_skel": "left_shoulder_pitch_link",
    "left_shoulder_roll_skel": "left_shoulder_roll_link",
    "left_shoulder_yaw_skel": "left_shoulder_yaw_link",
    "left_elbow_skel": "left_elbow_link",
    "left_wrist_yaw_skel": "left_wrist_yaw_link",
    "left_wrist_pitch_skel": "left_wrist_pitch_link",
    "left_wrist_roll_skel": "left_wrist_roll_link",
    "right_shoulder_pitch_skel": "right_shoulder_pitch_link",
    "right_shoulder_roll_skel": "right_shoulder_roll_link",
    "right_shoulder_yaw_skel": "right_shoulder_yaw_link",
    "right_elbow_skel": "right_elbow_link",
    "right_wrist_yaw_skel": "right_wrist_yaw_link",
    "right_wrist_pitch_skel": "right_wrist_pitch_link",
    "right_wrist_roll_skel": "right_wrist_roll_link",
    "head_yaw_skel": "head_yaw_link",
    "head_pitch_skel": "head_pitch_link",
}
