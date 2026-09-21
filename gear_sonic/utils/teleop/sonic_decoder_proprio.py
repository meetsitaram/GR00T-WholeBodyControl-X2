"""Bridge-side 990-D proprioception buffer for the SONIC pose decoder.

The SONIC ``g1_dyn`` decoder (loaded on the laptop by
``live_vla_publish_motion_token`` to translate VLA ``motion_token`` chunks
into ``joint_pos_mj``) was trained against the full IsaacLab proprioception
vector: 10 frames of history per term, in the term order

    base_ang_vel  (3)
    joint_pos_rel (31)
    joint_vel     (31)
    last_action   (31)
    gravity_dir   (3)

Each term ships oldest-first within its 10-frame history block, giving
``10 * (3+31+31+31+3) = 990`` floats. Feeding all-zeros for this vector
(the bridge's v0 placeholder) was the dominant cause of the on-robot
jitter observed 2026-06-07: the decoder is OOD with zero proprio and
emits a ~0.46 rad-from-idle pose target on every chunk regardless of
the VLA's intent.

This module ports the canonical ``ProprioceptionBuffer`` from
``gear_sonic/scripts/eval_x2_mujoco.py`` so the bridge can assemble the
real 990-D vector from the ``x2_debug`` stream that's already arriving
~50 Hz.

Live-bridge wiring:

* Every fresh ``x2_debug`` frame carries ``body_q`` (31, MJ),
  ``body_dq`` (31, MJ), ``base_quat`` (4, wxyz), ``base_ang_vel`` (3,
  body-frame), and ``left_hand_q`` / ``right_hand_q`` (10 each).
* The publisher converts MJ-order joints to IL-order via
  :data:`IL_TO_MJ_DOF`, computes ``joint_pos_rel = jpos_il - default_il``,
  and rotates ``[0, 0, -1]`` by ``R(base_quat).T`` to get the body-frame
  projected gravity.
* The bridge tracks its OWN ``last_action_il`` -- i.e., the raw
  decoder-output residual from the previous publish tick, not the
  deploy's published ``last_action`` field (which is ``target_pos_mj``,
  a position, not the residual the proprio buffer expects).

This matches the IL ``proprioception`` term-by-term layout produced by
the deploy's C++ ``ProprioceptionBuffer`` (see
``gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/proprioception_buffer.cpp``).
"""

from __future__ import annotations

import collections
from typing import Optional

import numpy as np

# Re-export the canonical joint-order + default-pose + action-scale
# constants from the decoder module so callers only need to import from
# one place.
from gear_sonic.utils.teleop.sonic_token_to_pose_decoder import (
    IL_TO_MJ_DOF,
    MJ_TO_IL_DOF,
    NUM_BODY_DOFS,
    X2_ACTION_SCALE_MJ,
    X2_DEFAULT_ANGLES_MJ,
)

__all__ = [
    "HISTORY_LEN",
    "PROPRIO_TOTAL_DIM",
    "ProprioceptionBuffer",
    "build_proprio_990",
    "default_angles_il",
    "quat_rotate_inverse",
]

HISTORY_LEN: int = 10

# 10 frames * (3 base_ang_vel + 31 joint_pos_rel + 31 joint_vel + 31
# last_action + 3 gravity_dir) = 990
PROPRIO_TOTAL_DIM: int = HISTORY_LEN * (3 + NUM_BODY_DOFS + NUM_BODY_DOFS + NUM_BODY_DOFS + 3)


_DEFAULT_ANGLES_IL = X2_DEFAULT_ANGLES_MJ[IL_TO_MJ_DOF].astype(np.float32)


def default_angles_il() -> np.ndarray:
    """Return the IsaacLab-order default stand pose used to compute
    ``joint_pos_rel``. Pre-permuted from ``X2_DEFAULT_ANGLES_MJ`` so the
    publisher hot path can avoid the gather each tick.
    """
    return _DEFAULT_ANGLES_IL


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the INVERSE of quaternion ``q`` (wxyz).

    Matches IsaacLab's ``quat_apply_inverse`` (see
    ``gear_sonic/scripts/eval_x2_mujoco.py:482``):

        v_local = v - w * t + cross(u, t)    with t = 2 * cross(u, v)

    Used to compute ``projected_gravity = R(base_quat).T @ [0, 0, -1]``
    from the body's world-frame quaternion.
    """
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"q_wxyz must be (4,); got {q.shape}")
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    u = np.array([x, y, z], dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    if vv.shape[0] != 3:
        raise ValueError(f"v must be (3,); got {vv.shape}")
    t = 2.0 * np.cross(u, vv)
    return vv - w * t + np.cross(u, t)


class ProprioceptionBuffer:
    """Term-by-term IsaacLab proprioception with 10-frame history.

    Ported from ``gear_sonic/scripts/eval_x2_mujoco.py:493`` so the live
    bridge can assemble the same 990-D vector the SONIC ``g1_dyn``
    decoder was trained against. Mirrors IsaacLab's
    ``CircularBuffer`` semantics: at reset the buffer is empty, and the
    first :meth:`append` after reset broadcast-fills all
    ``HISTORY_LEN`` slots with the first observation (see IsaacLab
    ``circular_buffer.py::buffer`` -- the first sample is replicated
    until the buffer fills naturally).

    Without this priming the first inference would consume
    ``HISTORY_LEN - 1`` zeroed frames in every history slot, which is
    OOD for any policy trained with history (and the entire reason the
    bridge previously had to fall back to ``_PROPRIO_ZERO_990``).
    """

    def __init__(self) -> None:
        self.gravity_hist: "collections.deque[np.ndarray]" = collections.deque(maxlen=HISTORY_LEN)
        self.angvel_hist: "collections.deque[np.ndarray]" = collections.deque(maxlen=HISTORY_LEN)
        self.jpos_hist: "collections.deque[np.ndarray]" = collections.deque(maxlen=HISTORY_LEN)
        self.jvel_hist: "collections.deque[np.ndarray]" = collections.deque(maxlen=HISTORY_LEN)
        self.action_hist: "collections.deque[np.ndarray]" = collections.deque(maxlen=HISTORY_LEN)
        self._primed: bool = False

    @property
    def primed(self) -> bool:
        return self._primed

    def reset(self) -> None:
        self.gravity_hist.clear()
        self.angvel_hist.clear()
        self.jpos_hist.clear()
        self.jvel_hist.clear()
        self.action_hist.clear()
        self._primed = False

    def append(
        self,
        gravity: np.ndarray,
        angvel: np.ndarray,
        jpos_rel_il: np.ndarray,
        jvel_il: np.ndarray,
        action_il: np.ndarray,
    ) -> None:
        g = np.asarray(gravity, dtype=np.float32).reshape(-1)
        a = np.asarray(angvel, dtype=np.float32).reshape(-1)
        jp = np.asarray(jpos_rel_il, dtype=np.float32).reshape(-1)
        jv = np.asarray(jvel_il, dtype=np.float32).reshape(-1)
        ac = np.asarray(action_il, dtype=np.float32).reshape(-1)
        if g.shape[0] != 3 or a.shape[0] != 3:
            raise ValueError(
                f"gravity/angvel must be (3,); got {g.shape}/{a.shape}"
            )
        if jp.shape[0] != NUM_BODY_DOFS or jv.shape[0] != NUM_BODY_DOFS or ac.shape[0] != NUM_BODY_DOFS:
            raise ValueError(
                f"jpos_rel/jvel/action must be ({NUM_BODY_DOFS},); "
                f"got {jp.shape}/{jv.shape}/{ac.shape}"
            )
        if not self._primed:
            for _ in range(HISTORY_LEN):
                self.gravity_hist.append(g)
                self.angvel_hist.append(a)
                self.jpos_hist.append(jp)
                self.jvel_hist.append(jv)
                self.action_hist.append(ac)
            self._primed = True
        else:
            self.gravity_hist.append(g)
            self.angvel_hist.append(a)
            self.jpos_hist.append(jp)
            self.jvel_hist.append(jv)
            self.action_hist.append(ac)

    def get_flat(self) -> np.ndarray:
        """Return the 990-dim proprioception in IsaacLab term-by-term layout.

        Term order MUST match ``PolicyCfg`` dataclass attribute order:

            base_ang_vel, joint_pos, joint_vel, actions, gravity_dir.

        Within each term, frames are oldest-first (the IsaacLab
        ``CircularBuffer.buffer`` convention).
        """
        parts = []
        for hist in (
            self.angvel_hist,
            self.jpos_hist,
            self.jvel_hist,
            self.action_hist,
            self.gravity_hist,
        ):
            for frame in hist:
                parts.append(frame)
        out = np.concatenate(parts).astype(np.float32)
        if out.shape[0] != PROPRIO_TOTAL_DIM:
            raise RuntimeError(
                f"proprio assembly produced {out.shape[0]} floats; "
                f"expected {PROPRIO_TOTAL_DIM} (history not yet primed?)"
            )
        return out


def build_proprio_990(
    buf: ProprioceptionBuffer,
    *,
    body_q_mj: np.ndarray,
    body_dq_mj: np.ndarray,
    base_quat_wxyz: np.ndarray,
    base_ang_vel: np.ndarray,
    last_action_il: np.ndarray,
    default_angles_il_cached: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience: append one frame's worth of x2_debug + last-action,
    then return the flat 990-D proprio.

    All inputs are taken in the raw form they arrive on the wire:

    * ``body_q_mj``, ``body_dq_mj``: MuJoCo joint order, radians /
      radians-per-second.
    * ``base_quat_wxyz``: world->body quaternion, wxyz convention
      (matches the deploy's ``x2_debug`` field).
    * ``base_ang_vel``: BODY-LOCAL angular velocity, rad/s (matches
      the deploy's ``rs.base_ang_vel`` -- see the long comment in
      ``eval_x2_mujoco.py:876`` about why we don't double-rotate this).
    * ``last_action_il``: the bridge's most recent decoded residual
      action in IsaacLab DOF order. Defaults to zeros on the first
      tick (the C++ deploy primes the same way; see
      ``proprioception_buffer.cpp``).

    Returns the flat 990-D float32 vector ready to feed
    :meth:`SonicTokenToPoseDecoder.decode_chunk`.
    """
    default_il = (
        default_angles_il_cached
        if default_angles_il_cached is not None
        else _DEFAULT_ANGLES_IL
    )
    body_q = np.asarray(body_q_mj, dtype=np.float32).reshape(-1)
    body_dq = np.asarray(body_dq_mj, dtype=np.float32).reshape(-1)
    if body_q.shape[0] != NUM_BODY_DOFS or body_dq.shape[0] != NUM_BODY_DOFS:
        raise ValueError(
            f"body_q/body_dq must be ({NUM_BODY_DOFS},); "
            f"got {body_q.shape}/{body_dq.shape}"
        )
    jpos_il = body_q[IL_TO_MJ_DOF]
    jvel_il = body_dq[IL_TO_MJ_DOF]
    jpos_rel_il = jpos_il - default_il
    gravity = quat_rotate_inverse(base_quat_wxyz, np.array([0.0, 0.0, -1.0])).astype(np.float32)
    angvel = np.asarray(base_ang_vel, dtype=np.float32).reshape(-1)
    buf.append(gravity, angvel, jpos_rel_il, jvel_il, last_action_il)
    return buf.get_flat()
