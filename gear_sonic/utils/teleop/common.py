"""Shared constants, enums, and utility classes for VR teleoperation.

Extracted from pico_manager_thread_server.py so that both PICO and Quest 3
managers can reuse them without pulling in device-specific dependencies.
"""

import os
import threading
import time
from enum import Enum, IntEnum

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller

try:
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )
except ImportError:
    G1GripperInverseKinematicsSolver = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
except ImportError:
    VR3PtPoseVisualizer = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
except ImportError:
    get_g1_key_frame_poses = None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LocomotionMode(IntEnum):
    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


# ---------------------------------------------------------------------------
# Constants + YawAccumulator (re-exported from vr.joystick_mapping)
# ---------------------------------------------------------------------------
#
# The actual implementation lives in
# ``gear_sonic.utils.teleop.vr.joystick_mapping`` so the new X2 manager
# (``quest3_manager_x2.py``) can import it without pulling in the G1-only
# ``LocomotionMode`` enum below. We keep the names re-exported here for
# back-compat with the PICO and G1 manager scripts that already do
# ``from gear_sonic.utils.teleop.common import JOYSTICK_DEADZONE,
# YawAccumulator``.

from gear_sonic.utils.teleop.vr.joystick_mapping import (  # noqa: E402
    JOYSTICK_DEADZONE,
    YawAccumulator,
)


# ---------------------------------------------------------------------------
# Hand IK helpers
# ---------------------------------------------------------------------------


def generate_finger_data(hand: str, trigger: float, grip: float) -> np.ndarray:
    fingertips = np.zeros([25, 4, 4])
    thumb = 0
    middle = 10
    fingertips[4 + thumb, 0, 3] = 1.0
    if trigger > 0.5:
        fingertips[4 + middle, 0, 3] = 1.0
    return fingertips


def init_hand_ik_solvers():
    if G1GripperInverseKinematicsSolver is not None:
        left_solver = G1GripperInverseKinematicsSolver(side="left")
        right_solver = G1GripperInverseKinematicsSolver(side="right")
        print("Hand IK solvers initialized")
        return left_solver, right_solver
    print("Warning: Hand IK solvers not available")
    return None, None


def compute_hand_joints_from_inputs(
    left_solver, right_solver, left_trigger, left_grip, right_trigger, right_grip
) -> tuple[np.ndarray, np.ndarray]:
    if left_solver is not None and right_solver is not None:
        left_finger_data = generate_finger_data("left", left_trigger, left_grip)
        right_finger_data = generate_finger_data("right", right_trigger, right_grip)
        left_hand_joints = left_solver({"position": left_finger_data})
        right_hand_joints = right_solver({"position": right_finger_data})
    else:
        left_hand_joints = np.zeros((1, 7), dtype=np.float32)
        right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    return left_hand_joints, right_hand_joints


# ---------------------------------------------------------------------------
# ThreePointPose  (device-agnostic calibration)
# ---------------------------------------------------------------------------


class ThreePointPose:
    """Calibration and optional visualization for VR 3-point pose data.

    Works with any source that produces a (3, 7) array of
    [left_wrist, right_wrist, neck/torso] in the robot coordinate frame
    (X-forward, Y-left, Z-up; quaternion scalar-first wxyz).
    """

    TORSO_LINK_OFFSET_Z = 0.05
    NECK_LINK_LENGTH = 0.35

    def __init__(
        self,
        enable_vis_vr3pt: bool = False,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        log_prefix: str = "ThreePointPose",
        robot_model=None,
    ):
        self.log_prefix = log_prefix
        self.with_g1_robot = with_g1_robot
        self.enable_waist_tracking = enable_waist_tracking
        self.enable_smpl_vis = enable_smpl_vis

        self._robot_model = robot_model
        if self._robot_model is None:
            from gear_sonic.data.robot_model.instantiation.g1 import (
                instantiate_g1_robot_model,
            )

            self._robot_model = instantiate_g1_robot_model()
            print(f"[{log_prefix}] Robot model loaded for FK calibration")

        self.vr3pt_visualizer = None
        if enable_vis_vr3pt:
            if VR3PtPoseVisualizer is None:
                raise ImportError(
                    "VR3PtPoseVisualizer could not be imported but --vis_vr3pt was requested. "
                    "Ensure pyvista is installed: pip install pyvista"
                )
            self.vr3pt_visualizer = VR3PtPoseVisualizer(
                axis_length=0.08,
                ball_radius=0.015,
                with_g1_robot=with_g1_robot,
                robot_model=self._robot_model,
                enable_waist_tracking=enable_waist_tracking,
                enable_smpl_vis=enable_smpl_vis,
            )
            self.vr3pt_visualizer.create_realtime_plotter(interactive=True)
            g1_str = " with G1 robot" if with_g1_robot else ""
            waist_str = " + waist tracking" if enable_waist_tracking else ""
            smpl_str = " + SMPL body" if enable_smpl_vis else ""
            print(f"[{log_prefix}] VR 3pt pose visualization enabled{g1_str}{waist_str}{smpl_str}")

        self._calibration_pending = False
        self._calibration_neck_quat_inv: np.ndarray | None = None
        self._calibration_lwrist_offset: np.ndarray | None = None
        self._calibration_rwrist_offset: np.ndarray | None = None
        self._calibration_lwrist_rot_offset: sRot | None = None
        self._calibration_rwrist_rot_offset: sRot | None = None
        self._override_robot_q: np.ndarray | None = None

    # -- public API -----------------------------------------------------------

    @property
    def is_pending(self) -> bool:
        return self._calibration_pending

    @property
    def is_calibrated(self) -> bool:
        return self._calibration_neck_quat_inv is not None

    def process(
        self,
        vr_3pt_pose_raw: np.ndarray,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calibrate and (optionally) visualize a raw 3-point pose.

        This is the device-agnostic entry point.  Both PICO and Quest 3
        managers should call this after converting their raw tracking data
        into a (3, 7) array [lwrist, rwrist, neck].

        Args:
            vr_3pt_pose_raw: (3, 7) ndarray — raw 3-pt pose (robot frame).
            smpl_joints_local: Optional (24, 3) for SMPL body visualization.

        Returns:
            vr_3pt_pose: (3, 7) calibrated pose.
        """
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_pose_raw)

        vr_3pt_pose = self._apply_calibration(vr_3pt_pose_raw)

        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            if smpl_joints_local is not None:
                self.vr3pt_visualizer.update_smpl_joints(smpl_joints_local)
            self.vr3pt_visualizer.render()

        return vr_3pt_pose

    def calibrate_now(self, vr_3pt_pose_raw: np.ndarray) -> bool:
        """Calibrate using the given raw 3-pt pose against FK of all-zero joints."""
        try:
            self._override_robot_q = np.zeros(29, dtype=np.float64)
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] Calibration completed (zero-pose reference)")
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def reset(self) -> None:
        self._clear_calibration()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Calibration reset, will re-calibrate on next frame")

    def reset_with_measured_q(self, body_q_measured: np.ndarray) -> None:
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = body_q_measured.copy()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Wrist recalibration pending (neck preserved, measured q)")

    def close(self) -> None:
        if self.vr3pt_visualizer is not None:
            try:
                self.vr3pt_visualizer.close()
            except Exception as e:
                print(f"[{self.log_prefix}] Warning: Error closing VR3pt visualizer: {e}")

    # -- internal -------------------------------------------------------------

    def _capture_calibration(self, vr_3pt_pose: np.ndarray) -> None:
        if self._calibration_neck_quat_inv is None:
            neck_quat_wxyz = vr_3pt_pose[2, 3:].copy()
            neck_rot = sRot.from_quat(neck_quat_wxyz, scalar_first=True)
            self._calibration_neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        lwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[0, :3].copy())
        rwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[1, :3].copy())
        lwrist_rot_corrected = calib_inv_rot * sRot.from_quat(
            vr_3pt_pose[0, 3:], scalar_first=True
        )
        rwrist_rot_corrected = calib_inv_rot * sRot.from_quat(
            vr_3pt_pose[1, 3:], scalar_first=True
        )

        if self._robot_model is None:
            raise RuntimeError("Robot model is required for calibration.")
        if get_g1_key_frame_poses is None:
            raise RuntimeError("get_g1_key_frame_poses could not be imported.")

        if self._override_robot_q is not None:
            robot_q = self._robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=self._override_robot_q[:29]
            )
        else:
            robot_q = None
        g1_poses = get_g1_key_frame_poses(self._robot_model, q=robot_q)

        g1_lwrist_pos = g1_poses["left_wrist"]["position"]
        g1_rwrist_pos = g1_poses["right_wrist"]["position"]
        g1_lwrist_rot = sRot.from_quat(
            g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True
        )
        g1_rwrist_rot = sRot.from_quat(
            g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True
        )

        self._calibration_lwrist_offset = lwrist_pos_corrected - g1_lwrist_pos
        self._calibration_rwrist_offset = rwrist_pos_corrected - g1_rwrist_pos
        self._calibration_lwrist_rot_offset = g1_lwrist_rot * lwrist_rot_corrected.inv()
        self._calibration_rwrist_rot_offset = g1_rwrist_rot * rwrist_rot_corrected.inv()

        self._calibration_pending = False
        self._override_robot_q = None

        source = "override q" if g1_lwrist_pos.any() else "default/zero"
        print(
            f"[{self.log_prefix}] Calibration captured (FK ref: {source}):\n"
            f"  L-Wrist pos offset: [{self._calibration_lwrist_offset[0]:.4f}, "
            f"{self._calibration_lwrist_offset[1]:.4f}, {self._calibration_lwrist_offset[2]:.4f}]\n"
            f"  R-Wrist pos offset: [{self._calibration_rwrist_offset[0]:.4f}, "
            f"{self._calibration_rwrist_offset[1]:.4f}, {self._calibration_rwrist_offset[2]:.4f}]"
        )

    def _apply_calibration(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        if self._calibration_neck_quat_inv is None:
            return vr_3pt_pose

        calibrated = vr_3pt_pose.copy()
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        neck_rot = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True)
        calibrated[2, 3:] = (calib_inv_rot * neck_rot).as_quat(scalar_first=True)

        if self._calibration_lwrist_offset is not None:
            calibrated[0, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[0, :3]) - self._calibration_lwrist_offset
            )
        if self._calibration_rwrist_offset is not None:
            calibrated[1, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[1, :3]) - self._calibration_rwrist_offset
            )

        if self._calibration_lwrist_rot_offset is not None:
            lw_corrected = calib_inv_rot * sRot.from_quat(
                vr_3pt_pose[0, 3:], scalar_first=True
            )
            calibrated[0, 3:] = (self._calibration_lwrist_rot_offset * lw_corrected).as_quat(
                scalar_first=True
            )
        if self._calibration_rwrist_rot_offset is not None:
            rw_corrected = calib_inv_rot * sRot.from_quat(
                vr_3pt_pose[1, 3:], scalar_first=True
            )
            calibrated[1, 3:] = (self._calibration_rwrist_rot_offset * rw_corrected).as_quat(
                scalar_first=True
            )

        neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0, 0, 1])
        calibrated[2, :3] = (
            np.array([0, 0, self.TORSO_LINK_OFFSET_Z]) + self.NECK_LINK_LENGTH * neck_z
        ).astype(np.float32)

        return calibrated

    def _clear_calibration(self):
        self._calibration_neck_quat_inv = None
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = None


# ---------------------------------------------------------------------------
# FeedbackReader
# ---------------------------------------------------------------------------


class FeedbackReader:
    """Reads feedback from robot via ZMQ to get measured upper body position."""

    def __init__(self, zmq_feedback_host: str = "localhost", zmq_feedback_port: int = 5557):
        self.poller = ZMQPoller(host=zmq_feedback_host, port=zmq_feedback_port, topic="g1_debug")
        self.upper_body_joint_indices = [
            12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
        ]

        self.upper_body_position_target = None
        self.left_hand_position_target = None
        self.right_hand_position_target = None
        self.full_body_q_measured: np.ndarray | None = None

    def poll_feedback(self):
        (
            self.upper_body_position_target,
            self.left_hand_position_target,
            self.right_hand_position_target,
            self.full_body_q_measured,
        ) = self._process_upper_body_position_targets()
        print("[FeedbackReader] Saved upper body position target:", self.upper_body_position_target)

    def _process_upper_body_position_targets(self):
        data = self.poller.get_data()
        if data is None:
            print("[FeedbackReader] No feedback data received")
            return None, None, None, None

        unpacked = msgpack.unpackb(data, raw=False)
        full_body_q = None
        if "body_q_measured" in unpacked:
            body_q_swizzled = unpacked["body_q_measured"]
            full_body_q = np.array(body_q_swizzled, dtype=np.float64)
            body_q = [body_q_swizzled[i] for i in self.upper_body_joint_indices]
        else:
            print("[FeedbackReader] body_q_measured not in feedback data")
            body_q = None

        left_hand_q = unpacked.get("left_hand_q_measured")
        if left_hand_q is None:
            print("[FeedbackReader] left_hand_q_measured not in feedback data")

        right_hand_q = unpacked.get("right_hand_q_measured")
        if right_hand_q is None:
            print("[FeedbackReader] right_hand_q_measured not in feedback data")

        return body_q, left_hand_q, right_hand_q, full_body_q
