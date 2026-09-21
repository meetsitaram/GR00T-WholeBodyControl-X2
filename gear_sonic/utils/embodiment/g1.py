"""Unitree G1 :class:`EmbodimentConfig` -- placeholder.

G1 is a planned future target for the X2 teleop / record / replay
stack. The :class:`EmbodimentConfig` instance registered here is a
*stub*: it advertises a plausible body / hand DOF count so the CLI's
``--robot g1`` flag completes argument parsing, but every factory it
exposes raises :class:`NotImplementedError` with a clear message.
This converts the "unknown robot" failure mode (``KeyError`` from the
registry) into a "robot not yet supported" failure mode pinpointed at
the place a real model would be needed.

To turn this into a real embodiment:

1. Replace the ``_g1_build_kinematic_model`` body with an MJCF loader
   for the G1 (likely wrapping a ``build_model_with_camera``-style
   helper analogous to X2's).
2. Replace ``_g1_apply_dexhand_fn`` with the G1-specific hand qpos
   applier (or set ``apply_omnihand_fn=None`` if G1 has no dexterous
   hand surface).
3. Update the constants below (DOF counts, pelvis pose, stand pose)
   to match the real G1 URDF / MJCF.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gear_sonic.utils.embodiment.config import EmbodimentConfig
from gear_sonic.utils.embodiment.registry import register_embodiment


__all__ = ["build_g1_embodiment_config"]


_G1_NOT_SUPPORTED_MSG = (
    "Unitree G1 embodiment is not yet supported. "
    "Replace the stub in gear_sonic/utils/embodiment/g1.py with a real "
    "EmbodimentConfig (MJCF builder + hand applier + stand pose) to "
    "enable --robot g1 for replay / teleop / record."
)


# Approximate placeholders so EmbodimentConfig.__post_init__ shape checks
# don't spuriously fail before the real factories are wired. These
# numbers are *not* authoritative for G1 -- update when G1 lands.
_G1_NUM_BODY_DOFS: int = 23
_G1_NUM_HAND_DOF_PER_SIDE: int = 7
_G1_PELVIS_POS_XYZ: tuple[float, float, float] = (0.0, 0.0, 0.793)
_G1_PELVIS_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _g1_build_kinematic_model(*, with_omnihand: bool) -> tuple[Any, Any, np.ndarray]:
    raise NotImplementedError(_G1_NOT_SUPPORTED_MSG)


def _g1_apply_dexhand_fn(
    data: Any,
    layout: Any,
    *,
    left_active: np.ndarray,
    right_active: np.ndarray,
) -> None:
    raise NotImplementedError(_G1_NOT_SUPPORTED_MSG)


def build_g1_embodiment_config() -> EmbodimentConfig:
    """Construct the G1 stub :class:`EmbodimentConfig`."""
    return EmbodimentConfig(
        name="g1",
        num_body_dofs=_G1_NUM_BODY_DOFS,
        num_hand_dof_per_side=_G1_NUM_HAND_DOF_PER_SIDE,
        pelvis_pos_xyz=_G1_PELVIS_POS_XYZ,
        pelvis_quat_wxyz=_G1_PELVIS_QUAT_WXYZ,
        default_stand_pose_mj=np.zeros(_G1_NUM_BODY_DOFS, dtype=np.float64),
        build_kinematic_model=_g1_build_kinematic_model,
        apply_omnihand_fn=_g1_apply_dexhand_fn,
    )


register_embodiment(build_g1_embodiment_config())
