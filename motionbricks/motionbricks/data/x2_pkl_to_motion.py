"""Convert X2 motion-lib PKL clips to MotionBricks ``input_tensor_dict`` tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from motionbricks.motionlib.core.skeletons.x2 import X2_SKEL_TO_MUJOCO_BODY, X2Skeleton34

# Same order as gear_sonic/scripts/eval_x2_mujoco.py
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint",
    "head_pitch_joint",
]

_TOE_FORWARD_OFFSET_M = 0.08

# MotionBricks uses a Y-up / Z-forward "motion space" while MuJoCo is Z-up /
# X-forward. The relation (see motionbricks/docs/motion_representation.md
# "Coordinate Systems") is:
#     motion[X, Y, Z] = mujoco[Y, Z, X]
# Equivalently, ``motion_v = MUJOCO_TO_MOTION @ mujoco_v``. This matches the
# constant used inside ``motionbricks/helper/mujoco_helper.py``.
_MUJOCO_TO_MOTION = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_MOTION_TO_MUJOCO = _MUJOCO_TO_MOTION.T


def default_x2_mjcf_path(repo_root: Optional[Path] = None) -> Path:
    """Canonical X2 MJCF (sphere-feet, 24 sphere geoms, nq=38).

    This is the same MJCF SONIC retargeted against (see
    ``gear_sonic/data_process/convert_soma_csv_to_motion_lib.py::X2_MJCF_FILE``)
    and trained against in IsaacLab via the sphere-feet config. The MotionBricks
    pipeline anchors here so the asset chain stays coherent end-to-end. See
    ``docs/source/user_guide/sim2sim_ablation_study.md`` §1.4 for the wrong-URDF
    history that motivates this single source of truth.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"


_EXPECTED_X2_NQ = 38       # 3 root pos + 4 quat + 31 hinge
_EXPECTED_X2_NJNT = 32     # 1 free + 31 hinge
_EXPECTED_X2_SPHERE_GEOMS = 24  # 12 per foot, sphere-feet collider


def assert_x2_mjcf_coherent(mjcf_path: Path) -> None:
    """Cheap structural check that ``mjcf_path`` is the canonical sphere-feet MJCF.

    Raises if the MJCF has the wrong qpos size, joint count, or is missing the
    24 sphere foot geoms. Catches accidental swaps to a mesh-feet variant
    (the failure mode from sim2sim_ablation_study.md §1.4).
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    sphere_count = int((m.geom_type == 2).sum())  # mjGEOM_SPHERE == 2
    problems: list[str] = []
    if m.nq != _EXPECTED_X2_NQ:
        problems.append(f"nq={m.nq} (expected {_EXPECTED_X2_NQ})")
    if m.njnt != _EXPECTED_X2_NJNT:
        problems.append(f"njnt={m.njnt} (expected {_EXPECTED_X2_NJNT})")
    if sphere_count < _EXPECTED_X2_SPHERE_GEOMS:
        problems.append(
            f"sphere_geoms={sphere_count} (expected >= {_EXPECTED_X2_SPHERE_GEOMS}); "
            f"likely a mesh-feet MJCF — see sim2sim_ablation_study.md §1.4"
        )
    if problems:
        raise AssertionError(
            f"X2 MJCF at {mjcf_path} failed coherence checks: {', '.join(problems)}"
        )


class X2MujocoFkExtractor:
  """Run MuJoCo FK on X2 motion-lib entries and pack skeleton-aligned tensors."""

  def __init__(self, mjcf_path: Path):
    self._model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    self._data = mujoco.MjData(self._model)
    self._body_ids = {
      name: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body)
      for name, body in X2_SKEL_TO_MUJOCO_BODY.items()
    }
    self._joint_qposadr = {
      name: self._model.jnt_qposadr[
        mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
      ]
      for name in MUJOCO_JOINT_NAMES
    }

  def clip_to_input_dict(
    self,
    clip: dict,
    *,
    subsample: int = 1,
  ) -> Dict[str, torch.Tensor]:
    """Build tensors for ``motion_rep(...)`` from one PKL motion entry.

    The output positions / rotations / translations are in **motion space**
    (Y-up, Z-forward) because that's what MotionBricks's
    ``compute_motion_features`` and the downstream ``mujoco_qpos_converter``
    assume. MuJoCo gives us positions in Z-up / X-forward, so we apply
    ``_MUJOCO_TO_MOTION`` before packing tensors. (Validated by the
    ``visualize_x2_round_trip.py`` smoke: max ``|raw - rt|`` should be near
    zero on root translation when this is wired correctly.)
    """
    dof = np.asarray(clip["dof"], dtype=np.float64)
    root_trans = np.asarray(clip["root_trans_offset"], dtype=np.float64)
    root_rot = np.asarray(clip["root_rot"], dtype=np.float64)  # xyzw scipy
    n = dof.shape[0]
    idx = np.arange(0, n, subsample, dtype=np.int64)
    dof = dof[idx]
    root_trans_mj = root_trans[idx]
    root_rot = root_rot[idx]
    t = len(idx)
    nb = len(X2_SKEL_TO_MUJOCO_BODY)

    posed_mj = np.zeros((t, nb, 3), dtype=np.float64)
    grot_mj = np.zeros((t, nb, 3, 3), dtype=np.float64)
    bone_names = list(X2_SKEL_TO_MUJOCO_BODY.keys())

    for fi in range(t):
      self._set_qpos(root_trans_mj[fi], root_rot[fi], dof[fi])
      mujoco.mj_forward(self._model, self._data)
      for j, skel_name in enumerate(bone_names):
        bid = self._body_ids[skel_name]
        pos = self._data.xpos[bid].copy()
        if skel_name in ("left_toe_base", "right_toe_base"):
          # Dummy toe: small forward offset from ankle in MuJoCo X-forward.
          root_q = root_rot[fi]
          yaw = Rot.from_quat(root_q).as_euler("zyx")[0]
          pos[0] += _TOE_FORWARD_OFFSET_M * np.cos(yaw)
          pos[1] += _TOE_FORWARD_OFFSET_M * np.sin(yaw)
        posed_mj[fi, j] = pos
        grot_mj[fi, j] = self._data.xmat[bid].reshape(3, 3).copy()

    # MuJoCo (Z-up) -> motion (Y-up):
    #   pos_motion  = M @ pos_mujoco
    #   R_motion    = M @ R_mujoco @ M.T
    posed = posed_mj @ _MUJOCO_TO_MOTION.T  # equivalent to (M @ pos.T).T
    grot = _MUJOCO_TO_MOTION @ grot_mj @ _MOTION_TO_MUJOCO
    translation_motion = root_trans_mj @ _MUJOCO_TO_MOTION.T

    return {
      "posed_joints": torch.from_numpy(posed.astype(np.float32)),
      "global_joint_rots": torch.from_numpy(grot.astype(np.float32)),
      "translation": torch.from_numpy(translation_motion.astype(np.float32)),
    }

  def _set_qpos(self, root_trans: np.ndarray, root_rot_xyzw: np.ndarray, dof: np.ndarray) -> None:
    qpos = self._data.qpos
    qpos[:] = 0.0
    qpos[0:3] = root_trans
    # MuJoCo free joint quaternion: w, x, y, z
    q_xyzw = np.asarray(root_rot_xyzw, dtype=np.float64)
    qpos[3:7] = np.array(
      [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
    )  # scipy xyzw → MuJoCo wxyz
    for jname, angle in zip(MUJOCO_JOINT_NAMES, dof, strict=True):
      qpos[self._joint_qposadr[jname]] = angle


def build_neutral_joints_from_mjcf(mjcf_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
  """T-pose joint positions (motion space, root at origin) + parent indices.

  Joint positions are returned in motion space (Y-up) so they line up with the
  features produced by :class:`X2MujocoFkExtractor`.
  """
  skel = X2Skeleton34()
  parents = torch.tensor(
    [-1 if skel.bone_parents[n] is None else skel.bone_index[skel.bone_parents[n]] for n in skel.bone_order_names],
    dtype=torch.long,
  )
  ext = X2MujocoFkExtractor(mjcf_path)
  ext._data.qpos[:] = 0.0
  ext._data.qpos[2] = 0.68  # nominal standing pelvis height (MuJoCo Z)
  mujoco.mj_forward(ext._model, ext._data)
  joints_mj = np.zeros((skel.nbjoints, 3), dtype=np.float64)
  for j, name in enumerate(skel.bone_order_names):
    bid = ext._body_ids[name]
    joints_mj[j] = ext._data.xpos[bid]
  joints = joints_mj @ _MUJOCO_TO_MOTION.T
  joints -= joints[skel.root_idx]
  return torch.from_numpy(joints.astype(np.float32)).unsqueeze(0), parents
