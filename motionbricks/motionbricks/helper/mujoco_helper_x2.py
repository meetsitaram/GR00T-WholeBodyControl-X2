"""X2 Ultra MuJoCo qpos converter (38 cols: 7 root + 31 hinge).

Subclass of ``mujoco_helper.mujoco_qpos_converter`` that fixes three G1-isms:

1. ``qpos`` has 38 columns (3 root_pos + 4 root_quat + 31 hinge), not G1's 36.
2. Default ``xml_path`` points at the canonical sphere-feet X2 MJCF
   (``gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml``).
3. The dead-joint scheme uses only 2 dummy joints (``left_toe_base`` /
   ``right_toe_base``) — G1 has 4 because it also has dummy hand_roll joints.
   The base class's generic ``-1`` scan handles this automatically as long as
   the X2 skeleton's ``X2_SKEL_TO_MUJOCO_BODY`` map matches.

See ``docs/source/user_guide/sim2sim_ablation_study.md`` §1.4 for the
sphere-feet asset history this helper anchors to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch as t
from torch import nn

from motionbricks.helper.mujoco_helper import mujoco_qpos_converter
from motionbricks.geometry.quaternions import matrix_to_quaternion
from motionbricks.motionlib.core.motion_reps import MotionRepBase
from motionbricks.motionlib.core.motion_reps.tools.changing_t_pose import (
    global_mats_to_local_mats as _slow_global_mats_to_local_mats,  # noqa: F401
)
from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix


def default_x2_mjcf_path_str() -> str:
    """Convenience wrapper returning the canonical X2 MJCF path as a string."""
    from motionbricks.data.x2_pkl_to_motion import default_x2_mjcf_path

    return str(default_x2_mjcf_path())


class X2MujocoQposConverter(mujoco_qpos_converter):
    """X2-specific qpos converter (38-col qpos)."""

    def __init__(
        self,
        motion_rep: MotionRepBase,
        xml_path: Optional[str] = None,
        dead_joint_rotation_scheme: str = "dummy",
    ) -> None:
        super().__init__(
            motion_rep=motion_rep,
            xml_path=xml_path or default_x2_mjcf_path_str(),
            dead_joint_rotation_scheme=dead_joint_rotation_scheme,
        )
        # G1: 7 + 29 = 36 ; X2: 7 + 31 = 38. Use the parsed value.
        self._qpos_dim = self._nb_joints_mujoco - 1 + 7

    def convert_motion_features_to_mujoco_qpos(
        self,
        motion_features: t.Tensor,
        motion_rep: MotionRepBase,
        is_normalized: bool = True,
        root_quat_w_first: bool = False,
    ) -> t.Tensor:
        """Same algorithm as G1 helper but with parameterized qpos column count."""
        from motionbricks.helper.mujoco_helper import global_mats_to_local_mats

        batch_size, num_frames = motion_features.shape[0], motion_features.shape[1]
        nb_joints = motion_rep.skeleton.nbjoints

        motion_rep = motion_rep.to(motion_features.device)
        if is_normalized:
            motion_features = motion_rep.unnormalize(motion_features)
        root_translation, root_rot_quat = motion_rep.compute_root_pos_and_rot(motion_features)

        global_joints_ric_rot = motion_rep.slice(motion_features, "global_rot_data")
        global_rot = cont6d_to_matrix(
            global_joints_ric_rot.view([batch_size, num_frames, nb_joints, 6])
        )
        local_joint_rot = global_mats_to_local_mats(global_rot, motion_rep.skeleton)
        local_joint_rot = t.matmul(
            self._rot_offsets_f2q.to(motion_features.device), local_joint_rot
        )

        device, dtype = root_translation.device, root_translation.dtype
        motion_to_mujoco_matrix = self.motion_to_mujoco_matrix.to(device=device, dtype=dtype)

        qpos = t.zeros(
            (batch_size, num_frames, self._qpos_dim), dtype=dtype, device=device
        )

        root_translation_mujoco = t.matmul(
            motion_to_mujoco_matrix[None, None, ...], root_translation[..., None]
        )
        qpos[:, :, :3] = root_translation_mujoco.view(batch_size, num_frames, 3)

        root_rot = local_joint_rot[:, :, 0, :]
        mujoco_to_motion_matrix = motion_to_mujoco_matrix.T
        root_rot_mujoco = t.matmul(
            t.matmul(motion_to_mujoco_matrix[None, None, ...], root_rot),
            mujoco_to_motion_matrix[None, None, ...],
        )
        root_rot_quat = matrix_to_quaternion(root_rot_mujoco)  # [w, x, y, z]
        if root_quat_w_first:
            qpos[:, :, 3:7] = root_rot_quat[:, :, [0, 1, 2, 3]]
        else:
            qpos[:, :, 3:7] = root_rot_quat[:, :, [1, 2, 3, 0]]

        joint_rot_mujoco = local_joint_rot[
            :, :, self._mujoco_indices_to_motion_indices, :
        ]
        x_joint_dof = t.atan2(joint_rot_mujoco[..., 2, 1], joint_rot_mujoco[..., 2, 2])
        y_joint_dof = t.atan2(joint_rot_mujoco[..., 0, 2], joint_rot_mujoco[..., 0, 0])
        z_joint_dof = t.atan2(joint_rot_mujoco[..., 1, 0], joint_rot_mujoco[..., 1, 1])
        xyz_joint_dofs = t.stack([x_joint_dof, y_joint_dof, z_joint_dof], dim=-1)
        joint_dofs = (
            xyz_joint_dofs
            * self._mujoco_joint_axis_values_motion_space[None, None, :, :].to(device)
        ).sum(dim=-1)
        qpos[:, :, 7:] = joint_dofs

        return qpos


_global_x2_converter: Optional[X2MujocoQposConverter] = None


def get_x2_mujoco_converter(
    motion_rep: MotionRepBase, xml_path: Optional[str] = None
) -> X2MujocoQposConverter:
    """Cached X2 converter (mirrors G1 helper's caching pattern)."""
    global _global_x2_converter
    if _global_x2_converter is None:
        _global_x2_converter = X2MujocoQposConverter(motion_rep, xml_path=xml_path)
    return _global_x2_converter


def motion_feature_to_x2_mujoco_qpos(
    motion_features: t.Tensor,
    motion_rep: MotionRepBase,
    is_normalized: bool = False,
    xml_path: Optional[str] = None,
    root_quat_w_first: bool = True,
) -> t.Tensor:
    """One-shot conversion: motion features [B, T, D] -> X2 qpos [B, T, 38].

    The default ``root_quat_w_first=True`` matches MuJoCo's wxyz convention
    (consistent with ``play_x2_motion_mujoco.py`` / ``eval_x2_mujoco.py``).
    """
    converter = get_x2_mujoco_converter(motion_rep, xml_path=xml_path)
    return converter.convert_motion_features_to_mujoco_qpos(
        motion_features,
        motion_rep,
        is_normalized=is_normalized,
        root_quat_w_first=root_quat_w_first,
    )


def mujoco_qpos_to_x2_pose_msg(qpos: t.Tensor) -> dict:
    """Split an X2 qpos[T, 38] tensor into the heuristic-planner pose-msg fields.

    Returns dict with::

        root_xyz:   Tensor[T, 3]
        root_quat:  Tensor[T, 4]   # wxyz, MuJoCo convention
        joint_pos:  Tensor[T, 31]  # MJ joint order (MUJOCO_JOINT_NAMES)
    """
    if qpos.shape[-1] != 38:
        raise ValueError(
            f"X2 qpos expected last-dim=38, got {qpos.shape[-1]}; "
            "did you accidentally pass G1 qpos (36) here?"
        )
    return {
        "root_xyz": qpos[..., :3].contiguous(),
        "root_quat": qpos[..., 3:7].contiguous(),
        "joint_pos": qpos[..., 7:].contiguous(),
    }
