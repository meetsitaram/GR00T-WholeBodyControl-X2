"""AgiBot X2 Ultra :class:`EmbodimentConfig`.

Wraps the existing X2 helpers (MJCF builder, OmniHand qpos applier,
stand-pose constants) into the embodiment-registry surface so the
kinematic-replay CLI can dispatch off ``--robot``. Everything here is a
thin adapter over modules that have already been the X2 source of
truth for a long time:

* MJCF + OmniHand augmentation:
  :func:`gear_sonic.scripts.compose_x2_with_omnihand.build_x2_with_omnihand_spec`.
* OmniHand qpos applier:
  :func:`gear_sonic.scripts.compose_x2_with_omnihand.apply_active_hand_qpos`.
* Default stand pose:
  :data:`gear_sonic.utils.planner.constants.DEFAULT_STAND_POSE_MUJOCO_RAD`.
* Pelvis pose: matches the gantry_hang firmware-stand entry in
  ``gear_sonic_deploy/configs/sim_init_poses.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.utils.embodiment.config import EmbodimentConfig
from gear_sonic.utils.embodiment.registry import register_embodiment


__all__ = ["build_x2_embodiment_config"]


# Pelvis pose for kinematic playback. Matches the on-feet stand entry
# the C++ deploy uses at boot.
_X2_PELVIS_POS_XYZ: tuple[float, float, float] = (0.0, 0.0, 0.665)
_X2_PELVIS_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


_X2_MJCF_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "assets" / "robot_description" / "mjcf" / "x2_ultra.xml"
)


def _x2_build_kinematic_model(*, with_omnihand: bool) -> tuple[Any, Any, np.ndarray]:
    """Load the X2 MJCF (optionally OmniHand-augmented) and return ``(model, layout, body_qposadr)``.

    * ``model``: the compiled :class:`mujoco.MjModel`.
    * ``layout``: a ``compose_x2_with_omnihand.HandQposLayout`` when
      ``with_omnihand`` is True, otherwise ``None``.
    * ``body_qposadr``: a length-31 ``np.ndarray[int64]`` mapping each
      slot of the canonical body trajectory (``X2_BODY_JOINT_NAMES``) to
      its ``qposadr`` in the compiled model.

    ``MjSpec.attach()`` inserts the OmniHand finger hinges immediately
    after the parent ``*_wrist_roll`` joint, so in the augmented model the
    right-arm joints are pushed past the left-hand finger qpos slots.
    Callers must therefore address body joints through the returned
    per-name table rather than assuming ``qpos[7:38]`` is contiguous.
    """
    import mujoco

    from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
        X2_BODY_JOINT_NAMES,
    )

    if with_omnihand:
        from gear_sonic.scripts.compose_x2_with_omnihand import (
            build_x2_with_omnihand_spec,
        )

        spec, _, layout = build_x2_with_omnihand_spec()
        model = spec.compile()
        if model is None:
            raise RuntimeError("augmented (X2 + OmniHand) MJCF failed to compile")
    else:
        spec = mujoco.MjSpec.from_file(str(_X2_MJCF_PATH))
        model = spec.compile()
        layout = None

    body_qposadr = np.empty(len(X2_BODY_JOINT_NAMES), dtype=np.int64)
    for i, name in enumerate(X2_BODY_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(
                f"X2 body joint {name!r} missing from compiled model "
                f"(with_omnihand={with_omnihand})"
            )
        body_qposadr[i] = int(model.jnt_qposadr[jid])
    return model, layout, body_qposadr


def _x2_apply_omnihand_fn(
    data: Any,
    layout: Any,
    *,
    left_active: np.ndarray,
    right_active: np.ndarray,
) -> None:
    """Lazy-import wrapper around ``apply_active_hand_qpos`` for X2's OmniHand."""
    from gear_sonic.scripts.compose_x2_with_omnihand import apply_active_hand_qpos

    apply_active_hand_qpos(
        data,
        layout,
        left_active=left_active,
        right_active=right_active,
    )


def build_x2_embodiment_config() -> EmbodimentConfig:
    """Construct the X2 :class:`EmbodimentConfig`.

    Pulls the canonical 31-D stand pose from
    :mod:`gear_sonic.utils.planner.constants` (the same constant the
    planner and the teleop scripts already use) so there is exactly one
    stand pose definition for X2 in the repo.
    """
    from gear_sonic.utils.embodiment.x2_constants import DEFAULT_HAND_DOF
    from gear_sonic.utils.planner.constants import (
        DEFAULT_STAND_POSE_MUJOCO_RAD,
        NUM_BODY_DOFS,
    )

    return EmbodimentConfig(
        name="x2",
        num_body_dofs=int(NUM_BODY_DOFS),
        num_hand_dof_per_side=int(DEFAULT_HAND_DOF),
        pelvis_pos_xyz=_X2_PELVIS_POS_XYZ,
        pelvis_quat_wxyz=_X2_PELVIS_QUAT_WXYZ,
        default_stand_pose_mj=np.array(
            DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64
        ),
        build_kinematic_model=_x2_build_kinematic_model,
        apply_omnihand_fn=_x2_apply_omnihand_fn,
    )


register_embodiment(build_x2_embodiment_config())
