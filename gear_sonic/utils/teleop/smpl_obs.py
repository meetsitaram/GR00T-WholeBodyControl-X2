"""Torch/mujoco-free SMPL-observation assembly for the whole-body pipeline.

Extracted verbatim from live_pico_smpl_teleop.py (2026-08-29) so the PC2
token service can import it without torch/mujoco — the exact functions the
feel-tested sim path runs. Any change here changes BOTH consumers; keep
them in lock-step with the fused-smpl export contract (840 = 10 x
[joints 72 | root6d 6 | wrists 6]).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

SMPL_WINDOW = 10
SMPL_DT = 0.02
DELAY_S = (SMPL_WINDOW - 1) * SMPL_DT  # 0.18 s design latency


def _heading(rot: Rot) -> float:
    v = rot.apply([1.0, 0.0, 0.0])
    return float(np.arctan2(v[1], v[0]))


def build_smpl_obs(joints, quats, base_quat_wxyz, wrist_il6,
                   yaw_align: Rot | None = None,
                   ori_mode: str = "full") -> np.ndarray:
    cur = Rot.from_quat([base_quat_wxyz[1], base_quat_wxyz[2],
                         base_quat_wxyz[3], base_quat_wxyz[0]])
    if ori_mode == "heading":
        # v1.1 cores: robot-YAW-only anchoring (see eval_x2_mujoco note)
        cur = Rot.from_euler("z", _heading(cur))
    frames = []
    for f in range(SMPL_WINDOW):
        oq = quats[f]
        o = Rot.from_quat([oq[1], oq[2], oq[3], oq[0]])
        if yaw_align is not None:
            o = yaw_align * o
        rel = cur.inv() * o
        frames.append(np.concatenate([
            joints[f].reshape(72),
            rel.as_matrix()[:, :2].reshape(6).astype(np.float32),
            wrist_il6.astype(np.float32),
        ]))
    return np.concatenate(frames).astype(np.float32)


