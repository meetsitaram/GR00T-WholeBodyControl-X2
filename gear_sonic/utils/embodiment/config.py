"""Embodiment configuration dataclass.

An :class:`EmbodimentConfig` bundles the per-robot constants and
factories that the kinematic-replay CLI (and, in the future, other
verbs) need to load a robot into MuJoCo and pose it from a recorded
trajectory.

The config is intentionally minimal: just enough metadata to validate a
LeRobot parquet and stand the robot on its feet for kinematic playback.
Full proprio / IK / hand retargeting still lives in the per-robot
modules under ``gear_sonic.utils.teleop`` and ``gear_sonic.data``; the
config is only the connective tissue between a CLI flag (``--robot``)
and the robot-specific helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


__all__ = ["EmbodimentConfig"]


@dataclass(frozen=True)
class EmbodimentConfig:
    """Per-robot constants + factories consumed by the replay CLI.

    Attributes:
        name: Short canonical key used by the registry (e.g. ``"x2"``,
            ``"g1"``). Matches the value passed via ``--robot``.
        num_body_dofs: Number of body joints in the MuJoCo joint order
            used by ``action.body_q_mj`` (v1 schema; legacy v0 datasets
            still use ``action.commanded_body_q_mj``, which the replay
            CLI auto-falls-back to). The replay CLI validates the
            parquet column width against this.
        num_hand_dof_per_side: Number of OmniHand-style finger DOFs per
            side (or 0 if the embodiment has no dexterous hand).
            Validated against ``action.left_hand_joints`` /
            ``action.right_hand_joints`` widths when ``--with-omnihand``
            is set.
        pelvis_pos_xyz: Floating-base position written into
            ``data.qpos[0:3]`` for kinematic playback. Should match the
            on-feet stand pose for the robot.
        pelvis_quat_wxyz: Floating-base orientation written into
            ``data.qpos[3:7]`` (MuJoCo convention is ``wxyz``).
        default_stand_pose_mj: Length-``num_body_dofs`` array of
            initial joint positions in MuJoCo ordering. Used as the
            initial qpos before the first parquet frame is applied so
            the viewer opens on a calm robot.
        build_kinematic_model: Factory that returns
            ``(model, layout, body_qposadr)`` for kinematic playback.
            Accepts ``with_omnihand: bool`` keyword. ``layout`` is
            ``None`` when ``with_omnihand=False``.
        apply_omnihand_fn: Callable that writes per-finger active DOFs
            into ``data.qpos`` given the OmniHand layout. Signature:
            ``fn(data, layout, *, left_active, right_active)``. ``None``
            when the robot has no dexterous hand or when callers are
            expected to call ``build_kinematic_model(with_omnihand=False)``.
    """

    name: str
    num_body_dofs: int
    num_hand_dof_per_side: int
    pelvis_pos_xyz: tuple[float, float, float]
    pelvis_quat_wxyz: tuple[float, float, float, float]
    default_stand_pose_mj: np.ndarray
    build_kinematic_model: Callable[..., tuple[Any, Any, np.ndarray]]
    apply_omnihand_fn: Optional[Callable[..., None]]

    def __post_init__(self) -> None:
        pose = np.asarray(self.default_stand_pose_mj, dtype=np.float64)
        if pose.shape != (self.num_body_dofs,):
            raise ValueError(
                f"EmbodimentConfig({self.name!r}): default_stand_pose_mj has "
                f"shape {pose.shape}; expected ({self.num_body_dofs},)"
            )
        # frozen dataclass -> can't reassign via self.x = ...; use object.__setattr__.
        object.__setattr__(self, "default_stand_pose_mj", pose)
        if len(self.pelvis_pos_xyz) != 3:
            raise ValueError(
                f"EmbodimentConfig({self.name!r}): pelvis_pos_xyz must have "
                f"3 elements; got {self.pelvis_pos_xyz!r}"
            )
        if len(self.pelvis_quat_wxyz) != 4:
            raise ValueError(
                f"EmbodimentConfig({self.name!r}): pelvis_quat_wxyz must have "
                f"4 elements (wxyz); got {self.pelvis_quat_wxyz!r}"
            )
