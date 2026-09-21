"""Generate IsaacLab<->MuJoCo index maps from name lists (never hand-type them).

The hand-typed per-robot index arrays are the #1 documented source of silent
porting bugs (docs/source/references/conventions.md#joint-ordering). This
module derives them from two NAME LISTS, which are auditable at a glance, and
provides the mapping dict in the exact legacy key convention the motion_lib
consumes.

Legacy key convention (established by G1, matched by X2 — see the comment at
x2_ultra.py "isaaclab_to_mujoco_mapping"): the VALUE under key
``"mujoco_to_isaaclab_dof"`` is the gather-source array that CONVERTS
MuJoCo-ordered data to IsaacLab order::

    dof_il = dof_mj[ mapping["mujoco_to_isaaclab_dof"] ]

i.e. ``mapping["mujoco_to_isaaclab_dof"][i]`` is the MuJoCo index of the joint
at IsaacLab position ``i`` (and symmetrically for the other three keys). Yes,
the key naming is confusing; it is load-bearing, so we reproduce it exactly
and verify against the existing G1/X2 constants in tests.
"""

from __future__ import annotations


def _src_map(dst_names: list[str], src_names: list[str]) -> list[int]:
    """out[i] = index in src_names of dst_names[i]  (dst = src[out])."""
    idx = {n: i for i, n in enumerate(src_names)}
    missing = [n for n in dst_names if n not in idx]
    if missing or len(dst_names) != len(src_names):
        raise ValueError(f"name lists differ: missing={missing}, "
                         f"len {len(dst_names)} vs {len(src_names)}")
    return [idx[n] for n in dst_names]


def build_isaaclab_mujoco_mapping(
    il_body_names: list[str], mj_body_names: list[str],
    il_joint_names: list[str], mj_joint_names: list[str],
) -> dict:
    """Return the mapping dict in the legacy motion_lib key convention."""
    return {
        "isaaclab_joints": il_body_names,
        # gather source arrays, named by the CONVERSION each performs:
        "mujoco_to_isaaclab_dof": _src_map(il_joint_names, mj_joint_names),
        "isaaclab_to_mujoco_dof": _src_map(mj_joint_names, il_joint_names),
        "mujoco_to_isaaclab_body": _src_map(il_body_names, mj_body_names),
        "isaaclab_to_mujoco_body": _src_map(mj_body_names, il_body_names),
    }
