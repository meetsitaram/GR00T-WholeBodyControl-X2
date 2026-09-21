#!/usr/bin/env python3
"""Convert X2 recorder NPZ files into motion_lib PKL entries.

Sibling to ``convert_soma_csv_to_motion_lib.py``. That script turns
SOMA-retargeted mocap CSVs into motion_lib PKLs. THIS script does the same
for robot-side captures produced by
``gear_sonic_deploy/scripts/x2_record_real_run.py`` (which the
``x2_record_remote.sh`` wrapper drives over SSH). The result is a PKL with
the same schema (``root_trans_offset``, ``pose_aa``, ``dof``, ``root_rot``,
``fps``) and is consumable by:

  * ``gear_sonic/scripts/play_x2_motion_mujoco.py`` (kinematic playback)
  * motion_lib training loaders / ``train_agent_trl.py``
  * any other tool that reads a ``joblib.dump``'d
    ``{motion_key: entry_dict}`` PKL.

NPZ schema (input)
------------------

For each group g in {leg, waist, arm, head}::

    joint_names_<g>      object array, k strings (order MATCHES the X2 MJCF)
    t_state_<g>          (N,)   monotonic seconds
    state_pos_<g>        (N, k) rad
    state_vel_<g>        (N, k) rad/s
    state_eff_<g>        (N, k) Nm
    t_cmd_<g>            (N,)
    cmd_pos_<g>          (N, k) rad
    cmd_vel_<g>          (N, k) rad/s
    cmd_kp_<g>           (N, k)
    cmd_kd_<g>           (N, k)

IMU::

    t_imu                (N,)
    imu_quat_wxyz        (N, 4)   -- note: WXYZ, not XYZW
    imu_angvel           (N, 3)
    imu_linacc           (N, 3)

MC mode timeline (ignored by this converter; preserved as recorder
metadata only)::

    t_mc_mode            (M,)
    mc_mode_str          (M,)   object array of strings

Misc::

    meta_json            zero-dim string array, JSON of {note, hostname,
                         duration_s_actual, started_at_iso, git_sha, ...}

PKL schema (output)
-------------------

::

    {<motion_key>: {
        "root_trans_offset": (T, 3)     float32,  # meters, world xyz
        "pose_aa":           (T, 32, 3) float32,  # body-frame axis-angle/body
        "dof":               (T, 31)    float32,  # rad, MuJoCo X2 31-DOF order
        "root_rot":          (T, 4)     float32,  # quaternion xyzw (scipy)
        "smpl_joints":       (T, 24, 3) float32,  # placeholder zeros
        "fps":               int,
        # Non-canonical extras (motion_lib ignores unknown keys):
        "x2_record_meta":             dict,
        "x2_record_source_npz":       str,
        "x2_record_dof_source":       "state" | "cmd",
        "x2_record_root_rot_mode":    "identity" | "torso-imu",
    }, ...}

Joint order is the canonical X2 Ultra MuJoCo 31-DOF vector, which matches
the recorder's group concatenation in MJ order::

    [l_hip_pitch, l_hip_roll, l_hip_yaw, l_knee, l_ankle_pitch, l_ankle_roll,
     r_hip_pitch, r_hip_roll, r_hip_yaw, r_knee, r_ankle_pitch, r_ankle_roll,
     waist_yaw, waist_pitch, waist_roll,
     l_shoulder_pitch, l_shoulder_roll, l_shoulder_yaw, l_elbow,
     l_wrist_yaw, l_wrist_pitch, l_wrist_roll,
     r_shoulder_pitch, r_shoulder_roll, r_shoulder_yaw, r_elbow,
     r_wrist_yaw, r_wrist_pitch, r_wrist_roll,
     head_yaw, head_pitch]

Root pose synthesis
-------------------

The recorder has only joint encoders + torso IMU. There is no global pose
(no world XY, no foot-contact integration upstream). We therefore SYNTHESISE
the pelvis floating-base pose so the PKL is self-contained -- any
downstream consumer that reads ``root_trans_offset`` sees a properly
grounded robot without needing playback-time tricks. SOMA-retargeted PKLs
work the same way: their ``root_trans_offset`` is the real mocap world
position; ours is the foot-FK reconstructed equivalent.

  ``root_trans_offset[:, 0:2]``  When the NPZ has a SLAM odometry stream
  (``t_slam`` + ``slam_pose_xyz`` + ``slam_pose_xyzw``, written by the
  recorder when subscribed to ``/slam/localization/odometry``), we use
  the LiDAR-SLAM world position directly -- it's ground truth from the
  robot's localization stack. When the NPZ has no SLAM stream, we fall
  back to pinning XY at ``(0, 0)``: joints + a torso IMU alone don't
  carry enough information to recover global XY (IMU linacc integration
  drifts catastrophically, and an operator pushing the robot via gantry
  without changing joints is unrecoverable by definition). Foot-anchor
  transfer was tried but flapped on weight-shift recordings (the anchor
  side legitimately switches as the operator rocks weight, and small
  per-frame foot-offset deltas under large IMU yaw drift accumulated
  hundreds of mm of phantom XY motion). Use the LiDAR path for real
  world XY.

  ``root_trans_offset[:, 2]``  Per-frame forward kinematics. We load the
  X2 MJCF, set each frame's joints + root rot, find the LOWEST contact
  sphere across both feet (``class="foot"`` group in the MJCF, ~12 per
  foot), and write ``pelvis_z = default_pelvis_z - lowest_foot_z`` so
  that sphere sits at world z=0. Picking the lowest contact sphere (per
  side) handles foot tilt (heel-up, tip-toe, foot-edge), and picking the
  MIN of left vs right handles one foot lifted in the air automatically:
  the airborne foot's spheres are all higher than the grounded foot's,
  so it's never chosen as the anchor. ``--floor-anchor`` overrides the
  side selection (``lower-foot`` (default) / ``left-foot`` / ``right-foot``
  / ``none``). In ``left-foot`` / ``right-foot`` the XY anchor is
  hard-locked to that side for the whole take regardless of who's
  lower; useful for single-leg balance demos.

    ``root_rot`` Depends on ``--root-rot``:

    ``identity`` -- pelvis stays upright. Operator-induced world tilts
        are not visualised. Good when you want to inspect joint
        kinematics in isolation.

    ``torso-imu`` (default) -- the recorder subscribes to
        ``/aima/hal/imu/torso/state``, which is the **torso** IMU (MJCF
        site ``imu_1`` inside ``waist_roll_link``). The world rotation
        it reports is therefore ``R_world_torso``, not the pelvis we
        actually want in qpos[3:7]. We recover the pelvis rotation by
        inverting the waist chain (pelvis -> waist_yaw(Z) ->
        waist_pitch(Y) -> waist_roll(X) = torso)::

            R_world_pelvis(t) = (R_imu(t)   · R_imu(0)^-1)
                              · (R_waist(t) · R_waist(0)^-1)^-1

        We anchor ``R_world_pelvis(0) = identity`` which folds the
        constant IMU mounting offset (typ. ~90 deg about X on the X2
        body) and the initial-world-yaw into one lumped frame zero.
        Geometrically: pelvis rotation = (how the torso moved in world)
        minus (how the torso moved relative to the pelvis via waist).
        A squat where the operator bends waist +18 deg while the torso
        stays world-vertical now shows the pelvis tilting back -18 deg,
        which is the actual physical pose.

    ``torso-imu-raw`` -- legacy: paste the torso IMU quat straight into
        the pelvis slot with no waist correction. The robot will appear
        nearly always upright in the viewer (because the operator usually
        keeps the torso world-vertical while squatting / shifting weight)
        and the waist joints will animate on top, making the visible
        total tilt = imu_tilt + waist_dof, which is double-counted.
        Provided only for debugging the raw IMU stream.

    NOTE: the floor-anchor pass uses whatever ``root_rot`` you picked,
    so ``torso-imu`` produces a *tilted* pelvis whose lowest foot is
    still on the floor (the robot leans but doesn't sink through the
    ground); ``identity`` keeps the pelvis flat (the robot bobs up/down
    purely from leg flexion).

Multi-file mode
---------------

If multiple NPZ inputs are passed, each becomes one entry keyed by the
file's stem (or ``--name`` if exactly one input is given). The combined
PKL is what ``play_x2_motion_mujoco.py --motion-key STEM`` expects.

Usage
-----

Single take (default: floor-anchored, IMU-driven pelvis tilt)::

    python gear_sonic/data_process/convert_x2_record_to_motion_lib.py \
        scratch/runs/move_library_20260527/balanced_stand.npz \
        --output gear_sonic/data/motions/x2_recorded/balanced_stand.pkl \
        --root-rot torso-imu

Single-leg balance demo (force the right foot as the floor anchor so the
left foot stays visibly in the air even if its lowest sphere briefly dips
below the right's)::

    python gear_sonic/data_process/convert_x2_record_to_motion_lib.py \
        scratch/runs/move_library_20260527/right_leg_stand.npz \
        --output gear_sonic/data/motions/x2_recorded/right_leg_stand.pkl \
        --root-rot torso-imu --floor-anchor right-foot

Many takes -> one combined library::

    python gear_sonic/data_process/convert_x2_record_to_motion_lib.py \
        scratch/runs/move_library_20260527/*.npz \
        --output gear_sonic/data/motions/x2_recorded/move_library.pkl

Replay one entry in MuJoCo (no special flags -- the PKL already carries
the grounded root_trans_offset, just like SOMA-retargeted PKLs do)::

    conda run -n env_isaaclab --no-capture-output python \
        gear_sonic/scripts/play_x2_motion_mujoco.py \
        --motion gear_sonic/data/motions/x2_recorded/balanced_stand.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial import transform

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    X2_DOF_AXIS,
    X2_NUM_BODIES,
    X2_NUM_DOF,
)

MJCF_PATH = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"


# Order of groups in the canonical X2 MuJoCo 31-DOF vector. Each tuple is
# (group_name_in_npz, expected_dof_count, expected_joint_names_in_order).
# We assert these match the npz exactly before concatenating, so an upstream
# joint rename in the recorder (or a future firmware that adds/drops joints)
# is caught loudly here rather than silently producing a misaligned dof
# vector.
X2_GROUP_ORDER: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("leg", 12, (
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    )),
    ("waist", 3, (
        "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    )),
    ("arm", 14, (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_yaw_joint", "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_yaw_joint", "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    )),
    ("head", 2, (
        "head_yaw_joint", "head_pitch_joint",
    )),
)
EXPECTED_TOTAL_DOF = sum(count for _, count, _ in X2_GROUP_ORDER)
assert EXPECTED_TOTAL_DOF == X2_NUM_DOF, (EXPECTED_TOTAL_DOF, X2_NUM_DOF)


def _verify_group(npz: np.lib.npyio.NpzFile, group: str,
                  expected_names: tuple[str, ...]) -> None:
    """Bail loudly if the recorder's joint names don't match what we expect.
    This catches firmware-level joint renames that would otherwise produce a
    silently-misaligned dof vector."""
    key = f"joint_names_{group}"
    if key not in npz.files:
        raise KeyError(f"NPZ missing required field: {key}")
    names_arr = npz[key]
    actual = tuple(str(n) for n in names_arr)
    if actual != expected_names:
        raise ValueError(
            f"joint_names_{group} mismatch.\n"
            f"  expected: {expected_names}\n"
            f"  got:      {actual}\n"
        )


def _resample_to_grid(t_src: np.ndarray, x_src: np.ndarray,
                      t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation onto t_grid. ``x_src`` can be 1-D or 2-D
    (N,) or (N, k). NaN outside the source span (we always grid INSIDE
    [t_src[0], t_src[-1]] so this only fires when there's a per-column
    gap; out-of-window samples become NaN which downstream consumers can
    detect)."""
    if x_src.ndim == 1:
        return np.interp(t_grid, t_src, x_src, left=np.nan, right=np.nan)
    out = np.empty((len(t_grid), x_src.shape[1]), dtype=np.float64)
    for j in range(x_src.shape[1]):
        out[:, j] = np.interp(t_grid, t_src, x_src[:, j],
                              left=np.nan, right=np.nan)
    return out


def _slerp_to_grid_wxyz(t_src: np.ndarray, q_wxyz_src: np.ndarray,
                        t_grid: np.ndarray) -> np.ndarray:
    """Spherical interpolation of (wxyz) quaternions onto t_grid. Returns
    wxyz. scipy's Slerp wants xyzw, so we shuffle in and back out. Source
    quats are normalized first (the IMU sometimes drifts slightly non-unit)."""
    if q_wxyz_src.shape[0] == 0:
        return np.tile(np.array([1, 0, 0, 0], dtype=np.float64),
                       (len(t_grid), 1))
    q_xyzw = q_wxyz_src[:, [1, 2, 3, 0]].astype(np.float64)
    norms = np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    q_xyzw = q_xyzw / norms
    # Clip the grid to source span so Slerp can't extrapolate.
    t_grid_clip = np.clip(t_grid, float(t_src[0]), float(t_src[-1]))
    rots = transform.Rotation.from_quat(q_xyzw)
    slerp = transform.Slerp(t_src, rots)
    interp_xyzw = slerp(t_grid_clip).as_quat()
    return interp_xyzw[:, [3, 0, 1, 2]]  # back to wxyz


def _pelvis_rot_from_torso_imu(
    q_imu_xyzw: np.ndarray,
    waist_yaw: np.ndarray,
    waist_pitch: np.ndarray,
    waist_roll: np.ndarray,
) -> np.ndarray:
    """Reconstruct ``R_world_pelvis(t)`` from a torso-mounted IMU plus the
    waist joint chain.

    The recorder subscribes to ``/aima/hal/imu/torso/state`` which is the
    IMU at MJCF site ``imu_1`` (inside ``waist_roll_link`` = the torso),
    not the pelvis. So the quat it reports is ``R_world_torso`` (modulo a
    constant mounting offset). Putting that directly into ``qpos[3:7]``
    treats the torso reading as if it were the pelvis, which double-counts
    the waist chain when the operator squats / shifts weight while keeping
    the torso world-vertical.

    Kinematic relation (with constant IMU mounting offset ``R_torso_imu``)::

        R_world_imu(t) = R_world_pelvis(t)
                       · R_pelvis_torso(t)
                       · R_torso_imu

    where ``R_pelvis_torso(t)`` is the waist chain in MJCF body order
    (yaw about Z, then pitch about Y, then roll about X). Solving for the
    pelvis::

        R_world_pelvis(t) = R_world_imu(t)
                          · R_torso_imu^-1
                          · R_pelvis_torso(t)^-1

    We don't know ``R_torso_imu`` (it's a robot-firmware/IMU-driver-
    specific mounting; the raw quat starts at ``~(0.70, -0.72, 0, 0)``,
    which is a -90 deg X rotation that's almost certainly the mounting
    convention rather than a 90 deg pelvis lean). We also don't care
    about the absolute world yaw (the operator can stand the robot
    facing any direction). Both unknowns are folded into a single
    "frame-zero anchor" by demanding ``R_world_pelvis(0) = I``, which
    gives::

        R_torso_imu = R_pelvis_torso(0)^-1 · R_world_imu(0)

    Substituting back yields the "delta vs frame 0" form actually
    implemented below::

        R_world_pelvis(t) = (R_world_imu(t)   · R_world_imu(0)^-1)
                          · (R_pelvis_torso(t) · R_pelvis_torso(0)^-1)^-1

    Intuition check: if the IMU is stationary (torso fixed in world) and
    the waist pitches +30 deg, the pelvis must have rotated -30 deg in
    world (so the torso stays put). And it does: ``delta_imu = I`` and
    ``delta_waist = R_y(+30)``, giving
    ``R_pelvis = I · R_y(-30) = R_y(-30)``.
    """
    R_imu = transform.Rotation.from_quat(q_imu_xyzw)
    # Waist chain order follows the MJCF body tree:
    #   pelvis -> waist_yaw_link  (axis="0 0 1", Z)
    #          -> waist_pitch_link (axis="0 1 0", Y)
    #          -> waist_roll_link  (axis="1 0 0", X)  -- this is the torso
    # In scipy convention r1 * r2 means r1.apply(r2.apply(v)), so the
    # product below corresponds to R_pelvis_torso = R_z(qy) R_y(qp) R_x(qr)
    # which is the pelvis-frame expression of a vector originally in the
    # torso frame.
    # scipy >= 1.13 requires the angles array's last dim to match the
    # seq-axis count: with a single-axis seq like "z", the input must be
    # shape (T, 1), not (T,). [:, None] satisfies both old and new scipy
    # without changing the resulting rotations.
    Ry = transform.Rotation.from_euler("z", waist_yaw[:, None])
    Rp = transform.Rotation.from_euler("y", waist_pitch[:, None])
    Rr = transform.Rotation.from_euler("x", waist_roll[:, None])
    R_waist = Ry * Rp * Rr  # R_pelvis_torso(t)

    R_imu_0_inv = R_imu[0].inv()
    R_waist_0_inv = R_waist[0].inv()
    delta_imu = R_imu * R_imu_0_inv
    delta_waist = R_waist * R_waist_0_inv
    R_pelvis = delta_imu * delta_waist.inv()
    return R_pelvis.as_quat().astype(np.float64)  # xyzw


def _pelvis_rot_from_foot_flat(
    dof: np.ndarray,
    floor_anchor: str,
) -> np.ndarray:
    """Reconstruct ``R_world_pelvis(t)`` from leg kinematics under the
    constraint that the anchor foot stays at its frame-0 world
    orientation (i.e. flat on the ground, as it was at idle).

    Why not just use the torso IMU? For in-place gestures (hug, bow,
    reach) MC commands a balance-preserving posture: arms forward, hips
    push back, torso also tilts back so the COM stays over the feet.
    The torso IMU reads a slightly forward-tilted torso (because the
    waist chain doesn't compensate fully), and the IMU-derived pelvis
    pitch comes out forward too -- which makes the MuJoCo replay
    moonwalk the feet under a forward-leaning robot that would
    physically tip over.

    Foot-flat IK instead asks: *given the recorded leg joints, what
    pelvis rotation keeps the anchor foot at its initial orientation?*
    That's a closed-form FK problem:

        Define R_chain(t) := orientation of the anchor foot body in the
                              PELVIS frame, computed by running FK with
                              pelvis at world identity and DOFs at t.
        Foot world orientation = R_pelvis(t) · R_chain(t).
        Want foot world orientation == R_chain(0) (the "flat" reference).
        =>  R_pelvis(t) = R_chain(0) · R_chain(t)^{-1}

    Properties:
      * Frame 0 ⇒ R_pelvis(0) = I (matches the IMU-derived convention).
      * Both legs idle ⇒ pelvis stays at identity (no rotation).
      * Hip flexes 5° (knee forward) ⇒ pelvis rolls back 5° to keep the
        foot flat. Matches the operator's "body tilts back to balance
        arms-forward weight shift" expectation.

    Trade-off: this discards the IMU reading entirely. Drift, sensor
    bias, and any genuine pelvis motion that ISN'T captured by leg
    joints (e.g. the operator physically lifting the robot) is lost.
    For in-place MC gestures that's a feature, not a bug.

    Anchor side: with ``floor_anchor in {left-foot, right-foot}`` the
    side is fixed for the whole take. With ``lower-foot`` (default) we
    pick the side that's lowest at frame 0 and stick with it -- mid-
    take anchor swap would cause a pelvis rotation discontinuity.
    """
    ctx = _get_floor_anchor_ctx()
    mj_model = ctx.mj_model
    mj_data = ctx.mj_data
    n = int(dof.shape[0])

    def _set_provisional(f: int) -> None:
        mj_data.qpos[0] = 0.0
        mj_data.qpos[1] = 0.0
        mj_data.qpos[2] = ctx.default_pelvis_z  # any Z is fine, we
                                                # only read orientations
        mj_data.qpos[3] = 1.0  # wxyz identity
        mj_data.qpos[4] = 0.0
        mj_data.qpos[5] = 0.0
        mj_data.qpos[6] = 0.0
        mj_data.qpos[7:7 + X2_NUM_DOF] = dof[f]
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

    def _foot_world_rot(body_id: int) -> transform.Rotation:
        # MjData.xmat is (nbody, 9) row-major rotation; ndarray view is
        # fine. We copy to be safe against the in-place qpos updates.
        return transform.Rotation.from_matrix(
            mj_data.xmat[body_id].reshape(3, 3).copy()
        )

    # Pick anchor side at frame 0 and hold it.
    _set_provisional(0)
    if floor_anchor == "left-foot":
        anchor_body = ctx.left_ankle_body_id
    elif floor_anchor == "right-foot":
        anchor_body = ctx.right_ankle_body_id
    else:  # lower-foot or unknown -> default to lower foot at frame 0
        left_min = float(mj_data.geom_xpos[ctx.left_foot_geom_ids, 2].min())
        right_min = float(mj_data.geom_xpos[ctx.right_foot_geom_ids, 2].min())
        anchor_body = (ctx.left_ankle_body_id if left_min <= right_min
                       else ctx.right_ankle_body_id)

    R_ref = _foot_world_rot(anchor_body)  # R_chain(0)

    out_xyzw = np.zeros((n, 4), dtype=np.float64)
    for f in range(n):
        _set_provisional(f)
        R_chf = _foot_world_rot(anchor_body)
        R_pelvis = R_ref * R_chf.inv()
        out_xyzw[f] = R_pelvis.as_quat()  # xyzw
    return out_xyzw


class _FloorAnchorContext:
    """One-time MuJoCo model load + foot-sphere geom indexing. Reused across
    frames (and across NPZ inputs in a multi-file conversion) so we don't pay
    the MJCF parse cost per frame."""

    __slots__ = (
        "mj_model", "mj_data", "left_foot_geom_ids",
        "right_foot_geom_ids", "left_ankle_body_id",
        "right_ankle_body_id", "default_pelvis_z",
    )

    def __init__(self) -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.mj_data = mujoco.MjData(self.mj_model)
        sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        self.left_ankle_body_id = self.mj_model.body("left_ankle_roll_link").id
        self.right_ankle_body_id = self.mj_model.body("right_ankle_roll_link").id
        # The ankle_roll_link bodies hold both a visual mesh geom and the
        # ~12 small contact spheres. We restrict to type=sphere so we don't
        # accidentally pull the bounding box of the visual mesh, whose
        # "world Z" is the body origin and would miss tip-toe / heel-up
        # nuances. The mesh geom is type=mesh and is excluded.
        self.left_foot_geom_ids = np.array(
            [g for g in range(self.mj_model.ngeom)
             if self.mj_model.geom_bodyid[g] == self.left_ankle_body_id
             and int(self.mj_model.geom_type[g]) == sphere_type],
            dtype=np.int32,
        )
        self.right_foot_geom_ids = np.array(
            [g for g in range(self.mj_model.ngeom)
             if self.mj_model.geom_bodyid[g] == self.right_ankle_body_id
             and int(self.mj_model.geom_type[g]) == sphere_type],
            dtype=np.int32,
        )
        if (self.left_foot_geom_ids.size == 0
                or self.right_foot_geom_ids.size == 0):
            raise RuntimeError(
                "Couldn't find foot contact spheres in the MJCF "
                f"(left: {self.left_foot_geom_ids.size}, "
                f"right: {self.right_foot_geom_ids.size}). Check that "
                "left/right_ankle_roll_link have <geom class=\"foot\" .../>"
                " spheres in the model."
            )
        # Default standing-stance pelvis height: <body name="pelvis" pos="0 0 0.68">.
        self.default_pelvis_z = float(self.mj_model.body("pelvis").pos[2])


_FLOOR_ANCHOR_CTX: _FloorAnchorContext | None = None


def _get_floor_anchor_ctx() -> _FloorAnchorContext:
    global _FLOOR_ANCHOR_CTX
    if _FLOOR_ANCHOR_CTX is None:
        _FLOOR_ANCHOR_CTX = _FloorAnchorContext()
    return _FLOOR_ANCHOR_CTX


def _floor_anchored_root_trans(
    dof: np.ndarray,
    root_xyzw: np.ndarray,
    mode: str,
    anchor_xy: bool = False,
) -> np.ndarray:
    """Per-frame foot-FK pass returning (T, 3) world XYZ so the chosen
    anchor foot's lowest contact sphere sits at z=0. When ``anchor_xy``
    is False (legacy), pelvis XY stays pinned at (0, 0) -- adequate when
    the take really is in place and a downstream SLAM substitute is
    available for world XY. When ``anchor_xy`` is True, the anchor
    foot's ankle world XY is also held constant across the take (i.e.
    pelvis XY shifts so the planted foot stays put). Required for any
    motion that involves hip/knee weight shifts -- without it, joint
    motion makes the feet appear to slide under a locked pelvis.

    ``mode``:
        ``lower-foot``  pick MIN(left_min, right_min) Z per frame. The
                        airborne foot's spheres are higher than the
                        grounded foot's, so it's never the anchor (the
                        robot doesn't sink through the floor when one
                        leg lifts).
        ``left-foot``   force left foot as Z anchor for the whole take
                        (useful for single-leg balance demos so the
                        right leg stays visibly off the ground even when
                        joint noise briefly puts the right lower than
                        the left).
        ``right-foot``  symmetric.

    With ``anchor_xy=True`` + ``lower-foot`` mode, the XY lock follows
    the current anchor side: when the anchor swaps (foot transfer in
    walking), the new anchor's current world XY becomes the lock target,
    so the pelvis doesn't jump. For in-place gestures the anchor side
    typically doesn't switch, so the lock target is the frame-0 anchor
    XY (which is (0, 0) by construction).
    """
    ctx = _get_floor_anchor_ctx()
    mj_model = ctx.mj_model
    mj_data = ctx.mj_data
    left_ids = ctx.left_foot_geom_ids
    right_ids = ctx.right_foot_geom_ids
    left_ankle_id = ctx.left_ankle_body_id
    right_ankle_id = ctx.right_ankle_body_id
    z0 = ctx.default_pelvis_z

    n = int(dof.shape[0])
    out = np.zeros((n, 3), dtype=np.float32)
    locked_xy: np.ndarray | None = None
    last_anchor: str | None = None
    for f in range(n):
        # Run FK with pelvis at origin XY. Since the pelvis is the free
        # root, any later XY shift translates the whole tree rigidly --
        # we read the anchor ankle's "at-origin" XY and add the
        # correction at the end, no second FK pass needed.
        mj_data.qpos[0] = 0.0
        mj_data.qpos[1] = 0.0
        mj_data.qpos[2] = z0  # provisional; corrected below
        # PKL stores xyzw; MuJoCo qpos quat is wxyz. Reorder.
        mj_data.qpos[3] = root_xyzw[f, 3]
        mj_data.qpos[4] = root_xyzw[f, 0]
        mj_data.qpos[5] = root_xyzw[f, 1]
        mj_data.qpos[6] = root_xyzw[f, 2]
        mj_data.qpos[7:7 + X2_NUM_DOF] = dof[f]
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        left_min = float(mj_data.geom_xpos[left_ids, 2].min())
        right_min = float(mj_data.geom_xpos[right_ids, 2].min())
        if mode == "lower-foot":
            anchor_side = "left" if left_min <= right_min else "right"
        elif mode == "left-foot":
            anchor_side = "left"
        elif mode == "right-foot":
            anchor_side = "right"
        else:
            raise ValueError(f"unknown floor_anchor mode: {mode}")

        anchor_z = left_min if anchor_side == "left" else right_min
        out[f, 2] = z0 - anchor_z

        if anchor_xy:
            anchor_body = (left_ankle_id if anchor_side == "left"
                           else right_ankle_id)
            # MjData.xpos is (nbody, 3) world body-frame origins after
            # mj_forward; the attribute is `xpos`, NOT `body_xpos`
            # (which exists on MjModel for *default* poses, not on
            # MjData).
            ankle_xy_at_origin = mj_data.xpos[anchor_body, :2].copy()
            if locked_xy is None or anchor_side != last_anchor:
                # First frame OR foot transfer: re-anchor at the new
                # anchor's current world XY (with pelvis XY = 0 the
                # ankle world XY equals ankle_xy_at_origin, so the
                # initial lock is just where the anchor naturally is).
                locked_xy = ankle_xy_at_origin
            out[f, 0] = float(locked_xy[0] - ankle_xy_at_origin[0])
            out[f, 1] = float(locked_xy[1] - ankle_xy_at_origin[1])
            last_anchor = anchor_side

    return out


def _slam_root_trans_rot(
    npz: np.lib.npyio.NpzFile,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Use LiDAR-SLAM odometry as ground-truth root pose if present.

    Returns ``(root_trans_xyz, root_rot_xyzw)`` resampled to ``t_grid``,
    or ``None`` if the NPZ doesn't carry a SLAM stream (older recordings
    or recordings made while the SLAM stack was down). The recorder
    writes::

        t_slam              (N,)        seconds
        slam_pose_xyz       (N, 3)      meters, world frame
        slam_pose_xyzw      (N, 4)      scipy quat order

    from ``/slam/localization/odometry`` (``nav_msgs/Odometry``).
    """
    required = ("t_slam", "slam_pose_xyz", "slam_pose_xyzw")
    if not all(k in npz.files for k in required):
        return None
    t_src = np.asarray(npz["t_slam"], dtype=np.float64)
    xyz = np.asarray(npz["slam_pose_xyz"], dtype=np.float64)
    xyzw = np.asarray(npz["slam_pose_xyzw"], dtype=np.float64)
    if t_src.size == 0:
        return None
    # Resample position linearly, orientation via slerp. wxyz<->xyzw shuffle
    # mirrors _slerp_to_grid_wxyz; redo here to keep slam-side internal.
    pos = np.column_stack([
        _resample_to_grid(t_src, xyz[:, 0], t_grid),
        _resample_to_grid(t_src, xyz[:, 1], t_grid),
        _resample_to_grid(t_src, xyz[:, 2], t_grid),
    ]).astype(np.float64)
    # Slerp on xyzw quats.
    norms = np.linalg.norm(xyzw, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    q_xyzw = xyzw / norms
    t_clip = np.clip(t_grid, float(t_src[0]), float(t_src[-1]))
    slerp = transform.Slerp(t_src, transform.Rotation.from_quat(q_xyzw))
    rot_xyzw = slerp(t_clip).as_quat()
    return pos.astype(np.float32), rot_xyzw.astype(np.float64)


def convert_one(npz_path: Path, *, fps: int, source: str,
                root_rot_mode: str, anchor_z: float,
                trim_start: float, trim_end: float,
                floor_anchor: str, use_slam: str,
                anchor_xy: bool = False) -> dict:
    with np.load(npz_path, allow_pickle=True) as npz:
        # Joint-name sanity check before anything else.
        for g, _, names in X2_GROUP_ORDER:
            _verify_group(npz, g, names)

        # Pick the time window. We bound by joint + IMU streams only;
        # MC mode is too sparse (5 Hz) to be a useful bound, and it's not
        # consumed by motion_lib anyway.
        starts: list[float] = []
        ends: list[float] = []
        for g, _, _ in X2_GROUP_ORDER:
            t = npz[f"t_{source}_{g}"]
            if t.size:
                starts.append(float(t[0]))
                ends.append(float(t[-1]))
        t_imu = npz["t_imu"]
        if t_imu.size:
            starts.append(float(t_imu[0]))
            ends.append(float(t_imu[-1]))
        if not starts:
            raise ValueError(f"{npz_path}: empty recording, nothing to convert.")

        t0 = max(starts) + trim_start
        t1 = min(ends) - trim_end
        if t1 - t0 <= 0:
            raise ValueError(
                f"{npz_path}: after trim_start={trim_start}s + "
                f"trim_end={trim_end}s, no window left "
                f"(t0={t0:.3f} t1={t1:.3f})."
            )

        # Uniform target grid.
        n_frames = int(np.floor((t1 - t0) * fps)) + 1
        if n_frames < 2:
            raise ValueError(f"{npz_path}: target fps={fps} gives <2 frames.")
        t_grid = t0 + np.arange(n_frames) / fps

        # Resample joints group-by-group, concat to (T, 31) in MJ order.
        cols = []
        for g, count, _ in X2_GROUP_ORDER:
            t = npz[f"t_{source}_{g}"]
            x = npz[f"{source}_pos_{g}"]
            if t.size == 0 or x.shape[0] == 0:
                print(f"  WARN: group '{g}' has no samples; filling zeros",
                      file=sys.stderr)
                cols.append(np.zeros((n_frames, count), dtype=np.float64))
                continue
            cols.append(_resample_to_grid(t, x, t_grid))
        dof = np.concatenate(cols, axis=1).astype(np.float32)
        assert dof.shape[1] == X2_NUM_DOF, (dof.shape, X2_NUM_DOF)

        # If the NPZ has SLAM odometry and we're allowed to use it, that
        # supersedes both root_rot_mode and floor_anchor: LiDAR gives us
        # ground-truth world XYZ + orientation directly. Otherwise we
        # fall through to the synthesis path below.
        slam_result = None
        if use_slam in ("auto", "require"):
            slam_result = _slam_root_trans_rot(npz, t_grid)
            if slam_result is None and use_slam == "require":
                raise ValueError(
                    f"{npz_path}: --root-pose-source=require but the NPZ "
                    "has no t_slam/slam_pose_* fields. Was "
                    "/slam/localization/odometry publishing during the "
                    "recording? See recorder docstring for setup."
                )
        # use_slam == "off" -> slam_result stays None, falls through to
        # the synthesis path.

        # Root rotation (needed before the floor-anchor pass: pelvis tilt
        # changes which foot sphere is lowest in world frame).
        if root_rot_mode == "identity":
            root_xyzw = np.tile(np.array([0, 0, 0, 1], dtype=np.float64),
                                (n_frames, 1))
        elif root_rot_mode in ("torso-imu", "torso-imu-raw"):
            q_wxyz = npz["imu_quat_wxyz"]
            t_imu_src = npz["t_imu"]
            if q_wxyz.shape[0] == 0:
                print(f"  WARN: {root_rot_mode} requested but no IMU "
                      "samples; falling back to identity", file=sys.stderr)
                root_xyzw = np.tile(np.array([0, 0, 0, 1], dtype=np.float64),
                                    (n_frames, 1))
            else:
                q_wxyz_grid = _slerp_to_grid_wxyz(
                    t_imu_src.astype(np.float64),
                    q_wxyz.astype(np.float64),
                    t_grid,
                )
                imu_xyzw_grid = q_wxyz_grid[:, [1, 2, 3, 0]]
                if root_rot_mode == "torso-imu-raw":
                    # Legacy / debug: paste the torso IMU quat directly
                    # into the pelvis slot, ignoring the waist chain.
                    # The viewer will show the robot mostly upright because
                    # the torso typically stays world-vertical while the
                    # operator squats; waist joints animate on top.
                    root_xyzw = imu_xyzw_grid
                else:
                    # Canonical "torso-imu": invert the waist chain so the
                    # PKL's pelvis rotation reflects the actual pelvis lean.
                    # X2_GROUP_ORDER puts legs (12) before waist (3), so
                    # the waist columns in dof are 12, 13, 14 == yaw,
                    # pitch, roll respectively. See _pelvis_rot_from_torso_imu
                    # for the derivation.
                    root_xyzw = _pelvis_rot_from_torso_imu(
                        imu_xyzw_grid,
                        waist_yaw=dof[:, 12].astype(np.float64),
                        waist_pitch=dof[:, 13].astype(np.float64),
                        waist_roll=dof[:, 14].astype(np.float64),
                    )
        elif root_rot_mode == "foot-flat":
            # No IMU input; pelvis rotation derived purely from leg-chain
            # FK + foot-flat constraint. See _pelvis_rot_from_foot_flat
            # docstring for derivation.
            root_xyzw = _pelvis_rot_from_foot_flat(dof, floor_anchor)
        else:
            raise ValueError(f"unknown --root-rot mode: {root_rot_mode}")

        # Root translation + (optional) rotation override.
        # Priority order:
        #   1. SLAM odometry if present (ground truth from LiDAR
        #      localization: gives both XYZ and pelvis-quat with no
        #      synthesis tricks needed).
        #   2. floor_anchor=none: legacy fixed-Z behaviour.
        #   3. floor-anchored foot-FK: pelvis Z follows the lowest contact
        #      sphere; pelvis XY pinned at (0, 0).
        if slam_result is not None:
            slam_pos, slam_rot = slam_result
            root_trans = slam_pos
            # SLAM gives us a real world pelvis orientation; use it instead
            # of the synthesised one. The IMU-derived path (above) is now
            # only useful when SLAM is unavailable.
            root_xyzw = slam_rot
            slam_used = True
        elif floor_anchor == "none":
            root_trans = np.zeros((n_frames, 3), dtype=np.float32)
            root_trans[:, 2] = float(anchor_z)
            slam_used = False
        else:
            root_trans = _floor_anchored_root_trans(
                dof, root_xyzw, floor_anchor, anchor_xy=anchor_xy,
            )
            slam_used = False

        # pose_aa: body-frame axis-angle per body. Body 0 = root (from
        # root_rot as axis-angle); bodies 1..31 = DOF_AXIS * dof_value.
        pose_aa = np.zeros((n_frames, X2_NUM_BODIES, 3), dtype=np.float32)
        pose_aa[:, 1:X2_NUM_BODIES, :] = (
            X2_DOF_AXIS[None, :, :] * dof[:, :, None]
        ).astype(np.float32)
        pose_aa[:, 0, :] = transform.Rotation.from_quat(
            root_xyzw
        ).as_rotvec().astype(np.float32)

        # Preserve recorder meta as a sidecar so it survives the trip into
        # the PKL. motion_lib loaders ignore unknown keys.
        meta: dict = {}
        if "meta_json" in npz.files:
            try:
                meta = json.loads(str(npz["meta_json"]))
            except json.JSONDecodeError:
                pass

        return {
            "root_trans_offset": root_trans,
            "pose_aa": pose_aa,
            "dof": dof,
            "root_rot": root_xyzw.astype(np.float32),
            "smpl_joints": np.zeros((n_frames, 24, 3), dtype=np.float32),
            "fps": int(fps),
            # Recorder-side metadata (non-canonical extras).
            "x2_record_meta": meta,
            "x2_record_source_npz": str(npz_path),
            "x2_record_dof_source": source,
            "x2_record_root_rot_mode": root_rot_mode,
            "x2_record_floor_anchor": floor_anchor,
            "x2_record_anchor_xy": bool(anchor_xy),
            "x2_record_root_pose_source": (
                "slam" if slam_used
                else (f"foot-fk:{floor_anchor}+xy" if anchor_xy
                      else f"foot-fk:{floor_anchor}")
            ),
            "x2_record_window_s": (float(t0), float(t1)),
        }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", type=Path,
                   help="NPZ file(s) produced by x2_record_real_run.py")
    p.add_argument("--output", "-o", required=True, type=Path,
                   help="Output PKL path")
    p.add_argument("--fps", type=int, default=30,
                   help="Target uniform frame rate (default 30, same as SOMA "
                        "PKLs ship at).")
    p.add_argument("--source", choices=("state", "cmd"), default="state",
                   help="Joint position source. 'state' = robot's actual "
                        "encoder positions (what physically happened). "
                        "'cmd' = MC's commanded positions (what MC wanted "
                        "to happen). Default: state.")
    p.add_argument("--root-rot",
                   choices=("identity", "torso-imu", "torso-imu-raw",
                            "foot-flat"),
                   default="torso-imu",
                   help="Pelvis world rotation source. 'torso-imu' "
                        "(default) reconstructs the pelvis rotation by "
                        "inverting the waist chain through the recorded "
                        "waist joint angles -- right when the operator "
                        "squats / shifts weight without tilting the "
                        "torso. 'foot-flat' (recommended for in-place "
                        "MC gestures: hug, bow, reach) derives the "
                        "pelvis rotation entirely from leg kinematics "
                        "so the anchor foot stays at its frame-0 "
                        "(idle, flat) world orientation; this produces "
                        "physically-correct counter-balance pitch (e.g. "
                        "pelvis tilts back when arms reach forward), "
                        "vs the IMU which can read a tip-over forward "
                        "lean. 'identity' keeps the pelvis dead upright "
                        "(joint-only animation). 'torso-imu-raw' pastes "
                        "the torso IMU quat straight into the pelvis "
                        "slot (legacy / debug, double-counts the waist).")
    p.add_argument("--anchor-z", type=float, default=1.0,
                   help="Pelvis world Z (height) when --floor-anchor=none. "
                        "Ignored otherwise (foot FK computes it per frame). "
                        "Default 1.0 m.")
    p.add_argument("--floor-anchor",
                   choices=("lower-foot", "left-foot", "right-foot", "none"),
                   default="lower-foot",
                   help="How to ground the pelvis Z in world frame. "
                        "'lower-foot' (default) runs per-frame forward "
                        "kinematics on the X2 MJCF and pins the lowest "
                        "contact sphere across both feet to z=0. "
                        "'left-foot' / 'right-foot' hard-lock one side "
                        "as the Z anchor for the whole take (single-leg "
                        "balance demos so the lifted leg stays visibly "
                        "off the ground even with joint noise). 'none' "
                        "falls back to the locked-in-place (0, 0, "
                        "--anchor-z) behaviour with no FK. Use "
                        "--anchor-xy to additionally pin the anchor "
                        "foot's world XY (recommended for any take "
                        "with hip/knee weight shifts, including "
                        "in-place gestures that sit back, lean, or "
                        "squat -- otherwise the pelvis stays at (0, 0) "
                        "and joint motion looks like sliding feet).")
    p.add_argument("--anchor-xy", action="store_true", default=False,
                   help="When --floor-anchor uses foot-FK (lower/left/"
                        "right-foot), also shift pelvis XY so the "
                        "anchor foot's ankle body stays at its lock-in "
                        "world XY. The anchor side can swap (foot "
                        "transfer during walking); the lock target "
                        "re-bases to the new anchor's current world XY "
                        "to avoid a pelvis jump. Default off for "
                        "backward compatibility with older PKLs; "
                        "MC-gesture captures should always enable it.")
    p.add_argument("--root-pose-source",
                   choices=("auto", "require", "off"), default="auto",
                   help="How to derive the pelvis world pose. 'auto' "
                        "(default): use SLAM odometry from the NPZ if "
                        "present (t_slam + slam_pose_xyz + slam_pose_xyzw, "
                        "written by the recorder when subscribed to "
                        "/slam/localization/odometry), otherwise fall back "
                        "to torso-IMU + foot-FK synthesis. 'require': fail "
                        "if no SLAM stream is present (useful for catching "
                        "recordings made while the SLAM stack was down). "
                        "'off': ignore the SLAM stream even if present "
                        "(forces the synthesis path; useful for diffing).")
    p.add_argument("--trim-start", type=float, default=0.5,
                   help="Seconds to drop from the start of each recording "
                        "(removes recorder-attach transients). Default 0.5s.")
    p.add_argument("--trim-end", type=float, default=0.5,
                   help="Seconds to drop from the end of each recording. "
                        "Default 0.5s.")
    p.add_argument("--name", type=str, default=None,
                   help="Override the motion key for single-input mode. "
                        "Ignored when multiple inputs are passed.")
    args = p.parse_args()

    if args.name and len(args.inputs) > 1:
        print("ERROR: --name is only valid with a single input.",
              file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    for npz_path in args.inputs:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            print(f"WARN: skipping missing input {npz_path}", file=sys.stderr)
            continue
        if len(args.inputs) == 1 and args.name:
            key = args.name
        else:
            key = npz_path.stem
        print(f"Converting {npz_path.name} -> '{key}'  "
              f"(source={args.source}, root_rot={args.root_rot}, "
              f"root_pose_source={args.root_pose_source}, "
              f"floor_anchor={args.floor_anchor}"
              f"{'+xy' if args.anchor_xy else ''}, "
              f"fps={args.fps})",
              flush=True)
        try:
            entry = convert_one(
                npz_path,
                fps=args.fps,
                source=args.source,
                root_rot_mode=args.root_rot,
                anchor_z=args.anchor_z,
                trim_start=args.trim_start,
                trim_end=args.trim_end,
                floor_anchor=args.floor_anchor,
                use_slam=("off" if args.root_pose_source == "off"
                          else args.root_pose_source),
                anchor_xy=args.anchor_xy,
            )
        except (ValueError, KeyError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        if key in entries:
            print(f"  WARN: duplicate key '{key}'; previous take overwritten",
                  file=sys.stderr)
        entries[key] = entry
        n_fr = entry['dof'].shape[0]
        src = entry.get("x2_record_root_pose_source", "?")
        print(f"  -> {n_fr} frames @ {args.fps} fps = "
              f"{n_fr / args.fps:.2f}s  [root pose: {src}]")

    if not entries:
        print("ERROR: no entries to write", file=sys.stderr)
        return 1

    print(f"\nSaving {args.output} ({len(entries)} entries)")
    joblib.dump(entries, args.output, compress=3)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
