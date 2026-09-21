"""Numerically tiny per-element rate clamps.

These are used by:

* :mod:`gear_sonic.scripts.x2_pose_mux` -- engagement slow-step ramp:
  the mux clamps the operator's first OVERRIDE frames per-element
  relative to the last forwarded VLA pose so the deploy doesn't see
  a single-tick jump from VLA-pose to operator-pose at the takeover
  edge.
* :mod:`gear_sonic.scripts.vla.live_vla_publish_motion_token` (tracking
  feedback, follow-up 11) -- per-joint variant for closed-loop wire
  step cap.

The semantics intentionally match ``_clamp_vector_step`` in the
bridge so a clamp applied at either end of the takeover handshake
behaves identically.
"""

from __future__ import annotations

import numpy as np


def clamp_vector_step_f32(
    target: np.ndarray,
    prev: np.ndarray | None,
    max_step: float,
) -> np.ndarray:
    """Per-element ``|target - prev| <= max_step`` rate clamp.

    Caps each element's step from ``prev`` to ``max_step`` while
    preserving the direction vector (so a large multi-joint step
    shrinks proportionally rather than slicing individual joints).
    Returns ``target`` unchanged when ``max_step <= 0`` or
    ``prev is None`` (cold-start tick).
    """
    tgt = np.asarray(target, dtype=np.float32)
    if max_step <= 0.0 or prev is None:
        return tgt.copy()
    prv = np.asarray(prev, dtype=np.float32)
    delta = tgt - prv
    peak = float(np.abs(delta).max())
    if peak <= max_step:
        return tgt.copy()
    return (prv + delta * (max_step / peak)).astype(np.float32)


def clamp_vector_step_per_joint_f32(
    target: np.ndarray,
    prev: np.ndarray | None,
    max_step_per_joint: np.ndarray,
) -> np.ndarray:
    """Per-element, per-joint independent ``|delta_i| <= cap_i`` clamp.

    Unlike :func:`clamp_vector_step_f32`, each joint is clamped
    independently rather than shrinking the whole vector by one
    scalar factor. ``max_step_per_joint`` must broadcast against
    ``target`` (same shape, or a 1-D cap of length ``target.shape[-1]``).

    Semantics for each element ``i``:

    * ``cap[i] > 0`` -> clamp |delta_i| to ``cap[i]``.
    * ``cap[i] == 0`` -> freeze the joint (delta_i forced to 0). Used
      by the tracking-feedback hard-cap: when measured joint error
      exceeds the hard threshold, the bridge passes a 0 cap to halt
      that joint until the measured pose catches up.
    * ``cap[i] < 0`` -> treat as "no cap" (+inf). Lets the caller
      mask out arm joints from the per-joint clamp by handing the
      non-arm slots a negative sentinel.

    Returns a fresh copy of ``target`` when ``prev is None`` (cold
    start; nothing to clamp against).
    """
    tgt = np.asarray(target, dtype=np.float32)
    if prev is None:
        return tgt.copy()
    prv = np.asarray(prev, dtype=np.float32)
    cap = np.asarray(max_step_per_joint, dtype=np.float32)
    if cap.shape != tgt.shape and cap.shape != tgt.shape[-1:]:
        raise ValueError(
            "max_step_per_joint shape mismatch: "
            f"got {cap.shape}, target {tgt.shape}"
        )
    delta = tgt - prv
    # cap < 0 -> +inf (no cap). cap == 0 -> 0 (freeze).
    effective = np.where(cap < 0.0, np.float32(np.inf), cap)
    clamped_delta = np.clip(delta, -effective, effective)
    return (prv + clamped_delta).astype(np.float32)
