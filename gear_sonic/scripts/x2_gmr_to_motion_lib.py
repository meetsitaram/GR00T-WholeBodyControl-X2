#!/usr/bin/env python
"""Stage 2 of tape -> X2: GMR raw retarget npz -> motion-lib pkl (+ render).

Consumes the {root_pos, root_rot_xyzw, dof_pos, fps} npz written by
pico_tape_to_x2_gmr.py, then:
  1. ports the corpus driver's sanity gates (tilt / jump / hip_pitch /
     yaw_rate, from soma-retargeter batched_g1x2_driver.sanity_check) and
     reports them — a retarget that trips these produced garbage upstream
     of any policy;
  2. writes a motion-lib pkl in the schema of
     g1_captures_to_x2_motion_pkl._x2_csv_to_entry (root_trans_offset /
     root_rot xyzw / dof / pose_aa / smpl_joints zeros / fps);
  3. optionally renders the clip kinematically (MuJoCo offscreen EGL on the
     deploy x2_ultra.xml) to an MP4 for eyeballing — the mandatory
     validate-kinematically-first step before any policy tracking.

Runs in the repo .venv (mujoco + gear_sonic importable).

Usage:
  .venv/bin/python -m gear_sonic.scripts.x2_gmr_to_motion_lib \
      --raw <retarget dir>/walk_crouch_stand_002_x2_raw.npz \
      --key pico_walk_crouch_stand_002 \
      --pkl <retarget dir>/pico_walk_crouch_stand_002.pkl \
      --render <renders dir>/walk_crouch_stand_002_x2.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    X2_DOF_AXIS,
    X2_NUM_DOF,
)

X2_XML = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
# MuJoCo-order indices of the hip pitch joints (leg chains start with hip_pitch).
HIP_PITCH_IDX = (0, 6)


def sanity_report(root_pos, root_rot_xyzw, dof, fps) -> dict:
    rot = sRot.from_quat(root_rot_xyzw)
    rpy = rot.as_euler("xyz")
    tilt = np.abs(rpy[:, :2]).max()
    jump = np.abs(np.diff(dof, axis=0)).max() if len(dof) > 1 else 0.0
    hip_pitch = np.abs(dof[:, HIP_PITCH_IDX]).max()
    yaw = np.unwrap(rpy[:, 2])
    yaw_rate = np.abs(np.diff(yaw)).max() * fps if len(yaw) > 1 else 0.0
    checks = {
        "tilt_rad": (tilt, 1.0),
        "jump_rad": (jump, 1.0),
        "hip_pitch_rad": (hip_pitch, 2.0),
        "yaw_rate_rad_s": (yaw_rate, 6.0),
    }
    print("[gmr2lib] sanity gates (value / threshold):")
    ok = True
    for name, (v, thr) in checks.items():
        flag = "OK " if v <= thr else "FAIL"
        ok &= v <= thr
        print(f"  {flag} {name:16s} {v:7.3f} / {thr}")
    return {"pass": ok, **{k: float(v) for k, (v, _) in checks.items()}}


def despike(root_pos, root_rot_xyzw, dof, fps,
            dof_jump_thr=0.8, yaw_rate_thr=6.0) -> tuple:
    """Repair isolated single-frame IK glitches by re-interpolating the
    flagged frames from their neighbours (linear for dof/pos, slerp for the
    root quat). Only frames whose per-step delta exceeds the thresholds are
    touched; a systemic failure (many frames) is left alone and reported —
    despiking that would hide a real retarget defect."""
    T = len(dof)
    jump = np.abs(np.diff(dof, axis=0)).max(axis=1)
    rot = sRot.from_quat(root_rot_xyzw)
    yaw = np.unwrap(rot.as_euler("xyz")[:, 2])
    yrate = np.abs(np.diff(yaw)) * fps
    bad = np.zeros(T, dtype=bool)
    # a spike shows as two consecutive large deltas (in and out of the bad
    # frame); flag the frame after the first large delta
    bad[1:] |= (jump > dof_jump_thr)
    bad[1:] |= (yrate > yaw_rate_thr)
    n_bad = int(bad.sum())
    if n_bad == 0 or n_bad > 0.02 * T:
        if n_bad:
            print(f"[gmr2lib] despike SKIPPED: {n_bad} flagged frames "
                  f"(> 2% of clip — systemic, inspect instead)")
        return root_pos, root_rot_xyzw, dof, 0
    good = np.where(~bad)[0]
    idx = np.arange(T)
    dof_f = dof.copy()
    pos_f = root_pos.copy()
    for c in range(dof.shape[1]):
        dof_f[:, c] = np.interp(idx, good, dof[good, c])
    for c in range(3):
        pos_f[:, c] = np.interp(idx, good, root_pos[good, c])
    from scipy.spatial.transform import Slerp
    slerp = Slerp(good.astype(float), sRot.from_quat(root_rot_xyzw[good]))
    rot_f = slerp(idx.clip(good[0], good[-1]).astype(float)).as_quat()
    print(f"[gmr2lib] despiked {n_bad} isolated frame(s)")
    return pos_f, rot_f, dof_f, n_bad



def _load_tape_body(tape):
    """(body(T,24,>=3), t, M) from a RAW device tape or a PROCESSED smpl-obs
    tape. Raw: unity coords, M = the y-up->z-up hop. Processed (§31.6 calib
    tapes): smpl_joints are 24 SMPL-order joint positions ALREADY z-up, so
    M = identity; a zero quat column is padded so f[j, :3] indexing holds.
    Joint indices are SMPL-24 order in both layouts."""
    d = np.load(tape)
    if "body_ok" in d.files:
        ok = d["body_ok"].astype(bool)
        body = d["body"][ok].astype(np.float64)
        t = (d["stamp_ns"][ok] - d["stamp_ns"][ok][0]) * 1e-9
        M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    else:
        body = np.asarray(d["smpl_joints"], dtype=np.float64).reshape(-1, 24, 3)
        body = np.concatenate([body, np.zeros((len(body), 24, 4))], axis=2)
        t = np.asarray(d["t"], dtype=np.float64)
        t = t - t[0]
        M = np.eye(3)
    return body, t, M

def analytic_arms(root_rot_xyzw, dof, tape: Path, fps: float) -> np.ndarray:
    """Overwrite the 8 shoulder/elbow dofs with closed-form values from the
    tape geometry. IK arms kept finding contorted branches (arms-behind with
    perfect elbow flex — 2026-08-15); analytic is branch-free by
    construction. Legs/torso/root stay from GMR IK (excellent there).

    Chain per side (MJCF axes): pitch about +y, roll about +x, yaw about +z,
    elbow about +y in [-2.356, 0]; rest arm points -z in the mount frame.
      upper-arm dir (mount frame) = Ry(p) Rx(r) (0,0,-1)
        -> r = asin(u_y), p = atan2(-u_x, -u_z)
      forearm dir after p, r: f' = Rz(yw) Ry(-flex) (0,0,-1)
        -> yw = atan2(f'_y, f'_x) (when bent; hold when straight)
    Torso frame per frame from the tape: up = pelvis->neck, heading from the
    shoulder line. Mount rotation measured from the MJCF at rest."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(X2_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def bq(name):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return sRot.from_quat(data.xquat[b], scalar_first=True)

    r_torso_rest = bq("torso_link")
    mounts = {s: r_torso_rest.inv() * bq(f"{s}_shoulder_pitch_link")
              for s in ("left", "right")}

    body, t, M = _load_tape_body(tape)
    tape_idx = np.searchsorted(
        t, np.arange(len(dof)) / fps, side="left").clip(0, len(t) - 1)

    ARM0 = {"left": 15, "right": 22}
    J = {"left": (16, 18, 20), "right": (17, 19, 21)}
    out = dof.copy()
    last_yw = {"left": 0.0, "right": 0.0}
    for i, k in enumerate(tape_idx):
        f = body[k]
        pel = M @ f[0, :3]
        neck = M @ f[12, :3]
        ls = M @ f[16, :3]
        rs = M @ f[17, :3]
        up = neck - pel
        up = up / max(np.linalg.norm(up), 1e-9)
        lat = ls - rs
        lat -= up * np.dot(lat, up)
        lat = lat / max(np.linalg.norm(lat), 1e-9)
        fwd = np.cross(lat, up)
        r_torso = sRot.from_matrix(np.stack([fwd, lat, up], axis=1))
        for side in ("left", "right"):
            js, je, jw = J[side]
            S = M @ f[js, :3]
            E = M @ f[je, :3]
            W = M @ f[jw, :3]
            u = E - S
            u = u / max(np.linalg.norm(u), 1e-9)
            w = W - E
            wn = np.linalg.norm(w)
            w = w / max(wn, 1e-9)
            m_inv = (r_torso * mounts[side]).inv()
            um = m_inv.apply(u)
            r = np.arcsin(np.clip(um[1], -1.0, 1.0))
            p = np.arctan2(-um[0], -um[2])
            cosf = np.clip(np.dot(u, w), -1.0, 1.0)
            flex = np.arccos(cosf)  # 0 = straight
            fm = m_inv.apply(w)
            f_pr = (sRot.from_euler("y", p) * sRot.from_euler("x", r)).inv().apply(fm)
            if flex > 0.15:
                yw = np.arctan2(f_pr[1], f_pr[0])
                last_yw[side] = yw
            else:
                yw = last_yw[side]
            a0 = ARM0[side]
            out[i, a0 + 0] = p
            out[i, a0 + 1] = r
            out[i, a0 + 2] = yw
            out[i, a0 + 3] = np.clip(-flex, -2.3556, 0.0)
    # respect X2 joint limits on the analytic dofs
    lo = model.jnt_range[1:, 0][ARM0["left"]:ARM0["left"] + 4]
    hi = model.jnt_range[1:, 1][ARM0["left"]:ARM0["left"] + 4]
    for a0 in ARM0.values():
        out[:, a0:a0 + 4] = np.clip(
            out[:, a0:a0 + 4],
            model.jnt_range[1 + a0:1 + a0 + 4, 0],
            model.jnt_range[1 + a0:1 + a0 + 4, 1])
    return out


def analytic_feet(root_pos, root_rot_xyzw, dof, tape: Path,
                  fps: float) -> np.ndarray:
    """Overwrite the ankle-PITCH dofs with closed-form values from the tape
    foot geometry. The IK's toe orientation task (weight 10) loses to the
    position task: post-calibration foot-pitch corr was ~0, and BEFORE
    calibration the feet mirrored the operator ("resting on the ball
    instead of the toes", 2026-08-22) — same orientation-task failure
    family as the arms, same cure. The ankle-tracker signal itself is
    excellent (|corr| 0.99 position-vs-quat foot pitch).

    Per frame: desired world foot pitch = human ankle->toe pitch relative
    to the clip's calibrated flat-stance frame; solve the ankle-pitch dof
    by 2 Newton steps on FK (slope ~1: the pitch axis is lateral)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(X2_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ANK = {"left": (4, "left_ankle_roll_link"),
           "right": (10, "right_ankle_roll_link")}
    bids, axes = {}, {}
    for s, (di, link) in ANK.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        assert bid >= 0, link
        bids[s] = bid
        # foot-forward axis in link frame: world +x at rest (flat feet)
        r_rest = sRot.from_quat(data.xquat[bid], scalar_first=True)
        axes[s] = r_rest.inv().apply([1.0, 0.0, 0.0])

    body, t, M = _load_tape_body(tape)
    tape_idx = np.searchsorted(
        t, np.arange(len(dof)) / fps, side="left").clip(0, len(t) - 1)

    HJ = {"left": (7, 10), "right": (8, 11)}   # ankle, toe

    def hpitch(f, s):
        a, tj = HJ[s]
        v = M @ f[tj, :3] - M @ f[a, :3]
        return float(np.arctan2(v[2], max(np.hypot(v[0], v[1]), 1e-9)))

    # Flat-stance calibration: MEDIAN over the first second (the engage
    # gate guarantees a standing start). A single "stillest" frame up to
    # ~3 s in caught the operator already moving -> constant toes-down
    # bias that corr-based checks can't see (operator: "you messed up
    # big", 2026-08-22). Never calibrate on one frame.
    n_cal = max(int(np.searchsorted(t, t[0] + 1.0)), 5)
    rest_k = n_cal  # reported below
    rest = {s: float(np.median([hpitch(body[k], s)
                                for k in range(min(n_cal, len(body)))]))
            for s in ANK}

    out = dof.copy()
    achieved = {s: [] for s in ANK}
    targets = {s: [] for s in ANK}
    head_h = {s: [] for s in ANK}   # human foot heading rel pelvis heading
    head_r = {s: [] for s in ANK}   # robot foot heading rel root yaw
    for i, k in enumerate(tape_idx):
        data.qpos[0:3] = root_pos[i]
        q = root_rot_xyzw[i]
        data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        data.qpos[7:7 + X2_NUM_DOF] = out[i]
        for s, (di, _link) in ANK.items():
            target = hpitch(body[k], s) - rest[s]   # flat stance -> 0
            lo, hi = model.jnt_range[1 + di]

            def _pitch():
                mujoco.mj_forward(model, data)
                r_link = sRot.from_quat(data.xquat[bids[s]],
                                        scalar_first=True)
                fv = r_link.apply(axes[s])
                return np.arctan2(fv[2], max(np.hypot(fv[0], fv[1]), 1e-9))

            # Newton with MEASURED slope: the pitch/dof slope is NEGATIVE
            # in this chain (assumed +1 first — solver ran away to the
            # -46 deg limit on every frame, 2026-08-22).
            for _ in range(2):
                cur = _pitch()
                dq = 0.05
                data.qpos[7 + di] += dq
                slope = (_pitch() - cur) / dq
                data.qpos[7 + di] -= dq
                if abs(slope) < 1e-3:
                    break
                data.qpos[7 + di] = np.clip(
                    data.qpos[7 + di] + (target - cur) / slope, lo, hi)
            out[i, di] = data.qpos[7 + di]
            achieved[s].append(_pitch())
            targets[s].append(target)
            # leg-heading samples (every 5th frame): robot foot forward
            # vs root yaw; human ankle->toe vs shoulder-line heading
            if i % 5 == 0:
                r_link = sRot.from_quat(data.xquat[bids[s]],
                                        scalar_first=True)
                fv = r_link.apply(axes[s])
                ryaw = sRot.from_quat(q).as_euler("zyx")[0]
                head_r[s].append(np.arctan2(fv[1], fv[0]) - ryaw)
                f = body[k]
                a_j, t_j = HJ[s]
                hv = M @ f[t_j, :3] - M @ f[a_j, :3]
                sh = M @ f[17, :3] - M @ f[16, :3]
                hyaw = np.arctan2(sh[1], sh[0]) + np.pi / 2
                head_h[s].append(np.arctan2(hv[1], hv[0]) - hyaw)
    # Gate on ACHIEVED world pitch, not the dof (the dof anti-correlates
    # with world pitch by construction — negative chain slope; a dof-level
    # corr here misled the first diagnosis, 2026-08-22).
    for s in ANK:
        a = np.array(achieved[s]); tg = np.array(targets[s])
        c = np.corrcoef(tg - tg.mean(), a - a.mean())[0, 1]
        err = np.degrees(np.abs(a - tg).mean())
        # ABSOLUTE flat-stance check (no mean subtraction — a constant
        # bias must show here, not be normalized away): first-second
        # achieved world pitch should be ~0 (feet flat).
        n1s = max((np.array(range(len(a))) < 50).sum(), 1)
        bias = np.degrees(np.mean(a[:n1s]))
        flag = "" if abs(bias) < 6.0 else "  <-- FLAT-STANCE BIAS, check!"
        # LEG-HEADING gate: constant whole-chain hip-yaw twist is invisible
        # to corr-based gates (seg08 had BOTH legs yawed ~-70 deg with
        # knees corr still ~+1 — operator caught it visually, 2026-08-22).
        dh = np.unwrap(np.array(head_r[s])) - np.unwrap(np.array(head_h[s]))
        htw = np.degrees(np.arctan2(np.mean(np.sin(dh)), np.mean(np.cos(dh))))
        hflag = "" if abs(htw) < 25.0 else "  <-- LEG-HEADING TWIST, check!"
        print(f"[gmr2lib] analytic feet {s}: FK world-pitch corr {c:+.2f} "
              f"solve err {err:.1f} deg | stance bias {bias:+.1f} deg "
              f"(cal={n_cal} frames){flag} | leg-heading diff "
              f"{htw:+.1f} deg{hflag}")
    return out


def to_entry(root_pos, root_rot_xyzw, dof, fps) -> dict:
    T = len(dof)
    assert dof.shape[1] == X2_NUM_DOF, dof.shape
    pose_aa = np.zeros((T, X2_NUM_DOF + 1, 3), dtype=np.float32)
    pose_aa[:, 0] = sRot.from_quat(root_rot_xyzw).as_rotvec()
    for j in range(X2_NUM_DOF):
        pose_aa[:, j + 1] = np.asarray(X2_DOF_AXIS[j])[None] * dof[:, j:j + 1]
    return {
        "root_trans_offset": root_pos.astype(np.float32),
        "root_rot": root_rot_xyzw.astype(np.float32),
        "dof": dof.astype(np.float32),
        "pose_aa": pose_aa,
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": int(round(fps)),
    }


def fk_wrist_check(root_pos, root_rot_xyzw, dof, tape: Path, fps: float) -> bool:
    """FK acceptance gate for the arm mapping (operator-requested 2026-08-15):
    robot wrist positions via MuJoCo FK vs the tape's human wrist positions,
    both expressed pelvis-relative in the HEADING frame, compared by per-axis
    Pearson correlation. A flipped mapping (arms behind instead of ahead)
    shows as negative X correlation; a side swap as negative Y."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(X2_XML))
    data = mujoco.MjData(model)
    wid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{s}_wrist_yaw_link") for s in ("left", "right")}
    sid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{s}_shoulder_pitch_link") for s in ("left", "right")}

    def heading_from_shoulders(l_pos, r_pos):
        """Yaw of body facing from the shoulder line — convention-free:
        forward = up x (left_shoulder - right_shoulder)."""
        d = np.asarray(l_pos) - np.asarray(r_pos)
        fwd = np.cross([0.0, 0.0, 1.0], d)
        return float(np.arctan2(fwd[1], fwd[0]))

    eid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{s}_elbow_link") for s in ("left", "right")}
    T = len(dof)
    stride = max(T // 600, 1)
    idxs = np.arange(0, T, stride)
    quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    rob = {(s, lvl): [] for s in wid for lvl in ("wrist", "elbow")}
    for i in idxs:
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = quat_wxyz[i]
        data.qpos[7:7 + dof.shape[1]] = dof[i]
        mujoco.mj_forward(model, data)
        yaw = heading_from_shoulders(data.xpos[sid["left"]], data.xpos[sid["right"]])
        r_inv = sRot.from_euler("z", -yaw)
        for s in wid:
            rob[(s, "wrist")].append(r_inv.apply(data.xpos[wid[s]] - root_pos[i]))
            rob[(s, "elbow")].append(r_inv.apply(data.xpos[eid[s]] - root_pos[i]))

    body, t, M = _load_tape_body(tape)
    t_out = np.arange(0.0, t[-1], 1.0 / fps)
    tape_idx = np.searchsorted(t, t_out[idxs.clip(0, len(t_out) - 1)],
                               side="left").clip(0, len(t) - 1)
    hum = {(s, lvl): [] for s in ("left", "right") for lvl in ("wrist", "elbow")}
    # scale human arm vectors to robot proportions for a fair ABSOLUTE
    # comparison — measured here (config scales are unit now; the morph in
    # stage 1 owns proportions)
    f0 = body[tape_idx[0]]
    hum_arm = (np.linalg.norm(f0[16, :3] - f0[18, :3])
               + np.linalg.norm(f0[18, :3] - f0[20, :3]))
    rob_arm = (np.linalg.norm(data.xpos[sid["left"]] - data.xpos[eid["left"]])
               + np.linalg.norm(data.xpos[eid["left"]] - data.xpos[wid["left"]]))
    arm_scale = round(float(rob_arm / max(hum_arm, 1e-6)), 4)
    for k in tape_idx:
        f = body[k]
        pelvis = M @ f[0, :3]
        yaw = heading_from_shoulders(M @ f[16, :3], M @ f[17, :3])
        r_inv = sRot.from_euler("z", -yaw)
        for s, jw, je in (("left", 20, 18), ("right", 21, 19)):
            hum[(s, "wrist")].append(r_inv.apply(M @ f[jw, :3] - pelvis) * arm_scale)
            hum[(s, "elbow")].append(r_inv.apply(M @ f[je, :3] - pelvis) * arm_scale)

    # ---- ELBOW ANGLE gate (operator-caught defect 2026-08-15: elbow
    # POSITION correlates even when the joint is LOCKED straight, because
    # elbow position rides on the shoulder. Compare the actual flex.) ------
    # human elbow angle: pi - angle between (shoulder-elbow, wrist-elbow)
    # MJCF qpos order (verified via mj_id2name 2026-08-15): L-arm 15-21,
    # R-arm 22-28, head 29-30. Elbow = index 3 within each arm block.
    ELBOW_DOF = {"left": 18, "right": 25}
    print("[gmr2lib] elbow ANGLE gate (human flex vs robot elbow dof):")
    for s, js, je, jw in (("left", 16, 18, 20), ("right", 17, 19, 21)):
        ang = []
        for k in tape_idx:
            f = body[k]
            u = M @ f[js, :3] - (M @ f[je, :3])
            v = M @ f[jw, :3] - (M @ f[je, :3])
            c = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
            ang.append(np.pi - np.arccos(np.clip(c, -1, 1)))
        ang = np.array(ang)
        rdof = dof[idxs.clip(0, len(dof) - 1), ELBOW_DOF[s]]
        n = min(len(ang), len(rdof))
        corr = float(np.corrcoef(ang[:n], np.abs(rdof[:n]))[0, 1])
        hum_range = float(np.degrees(ang.max() - ang.min()))
        rob_range = float(np.degrees(np.ptp(rdof)))
        passed = rob_range > 0.4 * hum_range and corr > 0.5
        print(f"  {'OK ' if passed else 'FAIL'} {s:5s} human flex range "
              f"{hum_range:5.1f} deg | robot elbow range {rob_range:5.1f} deg "
              f"| corr {corr:+.2f}")

    ok = True
    print(f"[gmr2lib] FK arm gate (heading frame, human scaled x{arm_scale}):")
    for s in ("left", "right"):
        for lvl in ("wrist", "elbow"):
            R_ = np.array(rob[(s, lvl)])
            H_ = np.array(hum[(s, lvl)])
            n = min(len(R_), len(H_))
            corr = [float(np.corrcoef(R_[:n, a], H_[:n, a])[0, 1]) for a in range(3)]
            abs_err = np.abs(R_[:n] - H_[:n]).mean(axis=0)
            passed = min(corr) > 0.5 and abs_err.max() < 0.15
            ok &= passed
            print(f"  {'OK ' if passed else 'FAIL'} {s:5s} {lvl:5s} "
                  f"corr x/y/z {corr[0]:+.2f}/{corr[1]:+.2f}/{corr[2]:+.2f}  "
                  f"|err| x/y/z {abs_err[0]:.3f}/{abs_err[1]:.3f}/{abs_err[2]:.3f} m")
    return ok


def render(root_pos, root_rot_xyzw, dof, fps, out: Path,
           width=960, height=720) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(X2_XML))
    # the deploy MJCF's offscreen framebuffer defaults to 640x480
    model.vis.global_.offwidth = max(width, 640)
    model.vis.global_.offheight = max(height, 480)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.0, -12.0, 135.0
    quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    with imageio.get_writer(out, fps=int(round(fps)), quality=8) as w:
        for i in range(len(dof)):
            data.qpos[:3] = root_pos[i]
            data.qpos[3:7] = quat_wxyz[i]
            data.qpos[7:7 + X2_NUM_DOF] = dof[i]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [root_pos[i, 0], root_pos[i, 1], 0.6]
            renderer.update_scene(data, camera=cam)
            w.append_data(renderer.render())
    print(f"[gmr2lib] rendered {len(dof)} frames -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--key", required=True)
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--render", type=Path, default=None)
    ap.add_argument("--fk-check-tape", type=Path, default=None,
                    help="original pico tape; runs the FK wrist gate")
    args = ap.parse_args()

    d = np.load(args.raw)
    root_pos = d["root_pos"].astype(np.float64)
    root_rot = d["root_rot_xyzw"].astype(np.float64)
    dof = d["dof_pos"].astype(np.float64)
    fps = float(d["fps"])
    print(f"[gmr2lib] {args.raw.name}: {len(dof)} frames @ {fps:.0f} Hz")

    # Auto-ground: measure the standing sole clearance via FK (lower decile
    # of the min ankle height, minus the 0.078 m ankle-to-sole drop) and
    # subtract it from root z. GMR grounds the HUMAN's lowest JOINT, but a
    # joint center is not a sole — the residual is a constant (+5.0 cm on
    # the 2026-08-15 tape) that reads as "robot walks in mid air".
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as _mj
    _m = _mj.MjModel.from_xml_path(str(X2_XML))
    _dd = _mj.MjData(_m)
    _aid = [_mj.mj_name2id(_m, _mj.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
            for s in ("left", "right")]
    _lows = []
    for i in range(0, len(dof), 10):
        _dd.qpos[:3] = root_pos[i]
        _dd.qpos[3:7] = root_rot[i][[3, 0, 1, 2]]
        _dd.qpos[7:7 + dof.shape[1]] = dof[i]
        _mj.mj_forward(_m, _dd)
        _lows.append(min(_dd.xpos[b][2] for b in _aid))
    _lows = np.sort(np.asarray(_lows))
    ground_err = float(_lows[:max(len(_lows) // 10, 1)].mean() - 0.078)
    root_pos = root_pos.copy()
    root_pos[:, 2] -= ground_err
    print(f"[gmr2lib] auto-ground: sole clearance {ground_err:+.3f} m removed "
          f"from root z")

    sanity_report(root_pos, root_rot, dof, fps)
    root_pos, root_rot, dof, n_fixed = despike(root_pos, root_rot, dof, fps)
    if n_fixed:
        print("[gmr2lib] post-despike gates:")
        sanity_report(root_pos, root_rot, dof, fps)

    if args.fk_check_tape is not None:
        dof = analytic_arms(root_rot, dof, args.fk_check_tape, fps)
        dof = analytic_feet(root_pos, root_rot, dof, args.fk_check_tape, fps)
        print("[gmr2lib] arms replaced with analytic closed-form solution")
        fk_wrist_check(root_pos, root_rot, dof, args.fk_check_tape, fps)

    entry = to_entry(root_pos, root_rot, dof, fps)
    args.pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({args.key: entry}, args.pkl)
    print(f"[gmr2lib] wrote {args.pkl} (key={args.key})")

    if args.render is not None:
        args.render.parent.mkdir(parents=True, exist_ok=True)
        render(root_pos, root_rot, dof, fps, args.render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
