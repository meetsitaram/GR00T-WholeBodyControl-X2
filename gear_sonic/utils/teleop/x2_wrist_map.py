"""SMPL wrist rotation -> X2 wrist chain angles (yaw, pitch, roll).

The Pico body stream gives the hand's rotation relative to the forearm,
expressed in the forearm's SMPL rest frame (x = bone, y = up, z = forward;
palm down in T-pose). The X2 wrist chain is wrist_yaw (link Z, along the
forearm) -> wrist_pitch (link Y) -> wrist_roll (link X):
R_hand = Rz(yaw) Ry(pitch) Rx(roll) in the forearm LINK frame.

Fixed SMPL -> X2 frame map, derived on the composed X2 + OmniHand model
(arms hanging, palms inward, thumbs forward; docs/source/references/
x2_whole_body_dual_head.md "Wrist axis convention"):
  left : x_s -> -Z_w, y_s -> +Y_w, z_s -> +X_w
  right: x_s -> +Z_w, y_s -> -Y_w, z_s -> +X_w        (both det +1)
and the forearm link frame is the world frame rolled 12.1 deg about X at
the zero pose (wrist_yaw axis (0,-0.21,0.98) left / (0,0.21,0.98) right).

Shared by pico_intent_sender.py (live) and wrist_calib_report.py (offline
verification of an operator calibration clip).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

LINK_TILT = float(np.arctan2(0.21, 0.98))
M_SMPL2WORLD = {
    0: np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),   # left
    1: np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),   # right
}
A_WORLD2LINK = {
    0: Rot.from_euler("x", -LINK_TILT).as_matrix(),
    1: Rot.from_euler("x", +LINK_TILT).as_matrix(),
}
# X2 ranges (x2_ultra.xml): yaw +-2.556, pitch +-0.558, roll L (-1.571, 0.724) / R mirrored
RANGES = {
    0: ((-2.556, 2.556), (-0.558, 0.558), (-1.571, 0.724)),
    1: ((-2.556, 2.556), (-0.558, 0.558), (-0.724, 1.571)),
}


# Empirical sign corrections on top of the derived frame map (operator
# calibration in the sim, 2026-09-04 17:09-17:15):
#  * roll (fwd/back bend): -1, FINAL, settled on the REAL ROBOT (operator
#    17:25: "90 deg is the forward direction -- palm turns forward, I
#    checked on the robot"). The 1.571 rad side of wrist_roll is palm-ward
#    flexion (left negative, right positive), and -1 sends the operator's
#    forward bend there. The derived +1 put it on the short 0.724 side.
#    NOTE: in the sim viewer the composed OmniHand made -1 LOOK mirrored;
#    if the real palms do not face inward at rest, the compose mount
#    (compose_x2_with_omnihand) is rotated, not this map.
#  * pitch (sideways tilt): operator "yaw not good" -> sign flipped (-1).
#  * yaw (pronation): "rotating good" -> unchanged.
# Env overrides for a live A/B: WRIST_ROLL_SIGN, WRIST_PITCH_SIGN,
# WRIST_YAW_SIGN_FRAME as "+1" / "-1". wrist_calib_report.py reads a
# scripted six-motion clip to settle all three at once.
import os as _os
ROLL_SIGN = float(_os.environ.get("WRIST_ROLL_SIGN", "+1"))
PITCH_SIGN = float(_os.environ.get("WRIST_PITCH_SIGN", "+1"))
YAW_SIGN = float(_os.environ.get("WRIST_YAW_SIGN_FRAME", "+1"))


def wrist_frame_ypr(aa: np.ndarray) -> np.ndarray:
    """aa (2,3) local wrist rotvecs [left, right] -> (2,3) [yaw, pitch, roll] rad
    (X2 wrist chain angles, sign corrections applied)."""
    out = np.zeros((2, 3), np.float64)
    for side in (0, 1):
        R_s = Rot.from_rotvec(np.asarray(aa[side], float)).as_matrix()
        T = A_WORLD2LINK[side] @ M_SMPL2WORLD[side]
        out[side] = Rot.from_matrix(T @ R_s @ T.T).as_euler("ZYX")   # intrinsic z, y', x''
    out[:, 0] *= YAW_SIGN
    out[:, 1] *= PITCH_SIGN
    out[:, 2] *= ROLL_SIGN
    return out


def wrist_frame_ypr_batch(aa: np.ndarray) -> np.ndarray:
    """aa (N,2,3) -> (N,2,3)."""
    return np.stack([wrist_frame_ypr(a) for a in aa])
