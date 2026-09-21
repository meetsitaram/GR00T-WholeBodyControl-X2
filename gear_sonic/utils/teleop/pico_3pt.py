"""Pico/XRoboToolkit 24-joint body pose -> 3-point VR pose reduction.

Extracted verbatim from ``gear_sonic/scripts/pico_manager_thread_server.py``
so the X2 teleop manager can consume the same reduction without importing
that script's torch/zmq/visualizer dependency stack. The G1 pico manager
imports these names back from here; both managers therefore share one
implementation of the Unity->robot transform and the SMPL keypoint offsets.

Only numpy + scipy are required.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as sRot

# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint
# frames with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15)
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's
#     axes): R_total = R_z(c) @ R_y(b) @ R_x(a)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes):
#     R_total = R_x(a) @ R_y(b) @ R_z(c)
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90 about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90 about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90 about fixed X, then yaw 180 about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90 about fixed Z
]


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """
    Transform a pose from Unity coordinate frame to robot coordinate frame.

    Args:
        pose: np.ndarray shape (7,) - [x, y, z, qx, qy, qz, qw] in Unity frame
        world_frame: np.ndarray shape (7,) - reference frame to compute relative transform
        scalar_first: bool - if True, quaternion is [qw, qx, qy, qz]; if False, [qx, qy, qz, qw]

    Returns:
        rel_pos: np.ndarray (3,) - position in robot frame
        rel_rot: np.ndarray (4,) - quaternion [qw, qx, qy, qz] in robot frame

    Coordinate transform matrix Q converts Unity (Y-up, left-handed) to Robot (Z-up, right-handed):
        Unity:  X-right, Y-up, Z-forward
        Robot:  X-forward, Y-left, Z-up
    """
    world_frame = np.asarray(world_frame, dtype=np.float64).copy()
    pose = np.asarray(pose, dtype=np.float64).copy()

    # Q transforms Unity coordinates to Robot coordinates
    # Unity [x, y, z] -> Robot [-x, z, y]
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def _process_3pt_pose(smpl_pose_np):
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.

    NOTE: We use Neck (joint 12) instead of Head (joint 15) because:
      - Neck is more rigidly coupled to the torso
      - Head has high DoF (looking around) which doesn't reflect body pose
      - Neck provides more stable tracking for upper body orientation

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)

                     IMPORTANT: Positions and orientations are RELATIVE TO ROOT (pelvis).

    Processing Steps:
        1. Transform all 24 joints from Unity frame to robot frame
        2. Extract 4 keypoints: Root(0), L-Wrist(22), R-Wrist(23), Neck(12)
        3. Apply per-joint rotation OFFSETS to align joint frames
        4. Make L-Wrist, R-Wrist, Neck relative to Root (both position and orientation)
        5. Return only the 3 non-root keypoints

    Note: Position calibration (wrist offsets, neck kinematic chain) is done in
          ThreePointPose.apply_calibration() to ensure consistency with calibrated
          orientations.
    """

    # Defensive copy: _compute_rel_transform modifies pose[:3] in-place, which would
    # corrupt the caller's array (e.g. PicoReader._latest) and cause wrong results
    # if the same sample is processed more than once.
    smpl_pose_np = np.asarray(smpl_pose_np, dtype=np.float64).copy()

    # STEP 1: Transform all joints from Unity frame to robot frame.
    # Input rows are [x, y, z, qx, qy, qz, qw] Unity scalar-last; output rows
    # are [x, y, z, qw, qx, qy, qz] robot-frame scalar-first.
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos
        body_poses[i, 3:] = orn

    # STEP 2 & 3: Extract the 4 keypoints and apply rotation OFFSETS.
    # rel_i: 0=Root, 1=L-Wrist, 2=R-Wrist, 3=Neck
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue

        pos = positions[i]
        rel_i = [0, 22, 23, 12].index(i)

        # pose[3:7] is [qw, qx, qy, qz]; post-multiply the offset
        # (intrinsic rotation).
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )

        kp_poses[rel_i, 3:] = rot_quat  # scalar-last temporarily for scipy
        kp_poses[rel_i, :3] = pos

    # STEP 4: Make positions and orientations RELATIVE TO ROOT.
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # still scalar-last for scipy

    for i in range(1, 4):
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # STEP 5: Return only L-Wrist, R-Wrist, Neck (skip Root). Each row:
    # [x, y, z, qw, qx, qy, qz] relative to root, scalar-first quaternion.
    return kp_poses[1:]


def _yaw_only_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Scalar-last quat -> scalar-last quat keeping only the Z (yaw) rotation."""
    yaw = sRot.from_quat(quat_xyzw).as_euler("zyx")[0]
    return sRot.from_euler("z", yaw).as_quat()


def _process_3pt_from_devices(head_pose, left_ctrl_pose, right_ctrl_pose):
    """Controller-pose fallback: 3-point VR pose without body tracking.

    Mirrors the Quest 3 reduction (``compute_3pt_pose_from_quest3``): the
    root is the floor projection of the HEADSET with yaw-only orientation,
    and the controllers stand in for the wrists — instead of the Pico body
    path's pelvis root / SMPL wrist joints. Operator calibration absorbs
    the controller-grip-vs-wrist offset, exactly as on Quest 3.

    Args:
        head_pose / left_ctrl_pose / right_ctrl_pose: (7,) arrays
            ``[x, y, z, qx, qy, qz, qw]`` in the XRoboToolkit (Unity,
            Y-up left-handed) world frame, as returned by
            ``xrt.get_headset_pose`` / ``get_*_controller_pose``.

    Returns:
        (3, 7) ``[left_wrist, right_wrist, head]`` rows, robot frame,
        RELATIVE TO ROOT, ``[xyz, quat_wxyz]`` scalar-first — the same
        contract as :func:`_process_3pt_pose`.
    """
    world = [0, 0, 0, 0, 0, 0, 1]
    robot = np.zeros((3, 7), dtype=np.float64)
    for i, pose in enumerate((left_ctrl_pose, right_ctrl_pose, head_pose)):
        pos, orn_wxyz = _compute_rel_transform(
            np.asarray(pose, dtype=np.float64), world, scalar_first=False
        )
        robot[i, :3] = pos
        robot[i, 3:] = orn_wxyz  # [qw, qx, qy, qz]

    # Root: floor projection of the head, yaw-only orientation.
    head_pos = robot[2, :3]
    root_pos = np.array([head_pos[0], head_pos[1], 0.0])
    head_quat_xyzw = np.concatenate([robot[2, 4:], robot[2, 3:4]])
    root_rot = sRot.from_quat(_yaw_only_quat_xyzw(head_quat_xyzw))

    out = np.zeros((3, 7), dtype=np.float32)
    for i in range(3):
        out[i, :3] = root_rot.inv().apply(robot[i, :3] - root_pos)
        row_rot = sRot.from_quat(
            np.concatenate([robot[i, 4:], robot[i, 3:4]]))  # wxyz -> xyzw
        out[i, 3:] = (root_rot.inv() * row_rot).as_quat(scalar_first=True)
    return out
