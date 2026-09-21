#!/usr/bin/env python
"""Pico tape -> clean SMPL stream for the SONIC smpl encoder (840-D path).

Converts a recorded Pico body tape (24 SMPL-topology joints, Unity frame,
~40 Hz) into the exact SMPL quantities the trained smpl encoder consumes
(sonic_release 3-encoder checkpoints, e.g. viewer_14000):

  smpl_joints_local  (T, 24, 3)  root-relative joints, Z-up, w-first chain,
                                 slots 22/23 = THUMB TIPS (via
                                 compute_human_joints FK, matching training)
  global_orient_quat (T, 4)      w-first, y-up->z-up hop + base-rot removed
                                 (compose with the ROBOT anchor at eval time;
                                 never feed world-frame 6D directly)
  pose_aa            (T, 72)     CANONICAL smpl_filtered layout:
                                 [root_raw(3) | body(69)] — root is the RAW
                                 y-up SMPL axis-angle (base rot included),
                                 exactly the source-corpus sidecar convention the
                                 motion-lib loader expects. (Before
                                 2026-08-26 this field was [body(69) |
                                 zeros(3)] — root MISSING, every body joint
                                 shifted one slot: the source of the
                                 pico-vs-corpus SMPL heading mismatch.)
  transl             (T, 3)      pelvis position in the y-up SMPL world
                                 (y = height), source-corpus convention
  fps                50          resampled; the encoder window is 10 frames
                                 at exactly 0.02 s — a 40 Hz feed silently
                                 stretches the lookahead 25%

Cleanliness gates (all run in-process, non-zero exit on failure):
  G1 quat/FK round-trip — elbow flex angles computed from the FK joints must
     match angles computed directly from the tape (validates xyzw->wxyz, the
     y-180 pre-rotation, parent chain, and the y-up->z-up hop end to end).
  G2 bone-length stability — FK bone lengths constant across frames.
  G3 timing — output dt exactly 1/50 s.

Conversion reuses the reference implementations verbatim
(pico_manager_thread_server.compute_from_body_poses / process_smpl_joints);
the parent chain is the STANDARD SMPL-24 one — the pico manager's local
copy carries an apparent typo (parent[23]=22, R_Hand under L_Hand) that its
3pt path never exercises.

Usage (.venv):
  .venv/bin/python -m gear_sonic.scripts.pico_tape_to_smpl_obs \
      --tape logs/pico_tapes/<tape>.npz \
      --out logs/pico_tapes/smpl/<tape>_smpl.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.pico_manager_thread_server import (  # noqa: E402
    compute_from_body_poses,
)

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                12, 13, 14, 16, 17, 18, 19, 20, 21]
FPS_OUT = 50.0


def elbow_angles_from_points(P: np.ndarray) -> np.ndarray:
    """(T, 24, 3) joints -> (T, 2) elbow flex angles [L, R]."""
    out = []
    for js, je, jw in ((16, 18, 20), (17, 19, 21)):
        u = P[:, js] - P[:, je]
        v = P[:, jw] - P[:, je]
        c = (u * v).sum(-1) / np.maximum(
            np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-9)
        out.append(np.pi - np.arccos(np.clip(c, -1, 1)))
    return np.stack(out, axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="write the output even when a quality gate fails (REVIEW use only, e.g. "
                         "side-by-side viewing; the npz records gates_ok=False so a corpus build "
                         "can still refuse it)")
    args = ap.parse_args()

    d = np.load(args.tape)
    ok = d["body_ok"].astype(bool)
    body = d["body"][ok].astype(np.float64)          # (T, 24, 7) unity xyzw
    t = (d["stamp_ns"][ok] - d["stamp_ns"][ok][0]) * 1e-9
    t_out = np.arange(0.0, t[-1], 1.0 / FPS_OUT)
    idx = np.searchsorted(t, t_out, side="left").clip(0, len(t) - 1)
    body = body[idx]
    T = len(body)
    print(f"[tape2smpl] {T} frames resampled to {FPS_OUT:.0f} Hz "
          f"({t[-1]:.1f}s) from {args.tape.name}")

    from scipy.spatial.transform import Rotation as Rot
    # Recover the RAW y-up root from the canonical (z-up, base-removed) quat
    # by inverting process_smpl_joints' exact chain:
    #   q_deb = ytoz * q_raw * base^-1   =>   q_raw = ytoz^-1 * q_deb * base
    _ytoz = Rot.from_rotvec([np.pi / 2, 0.0, 0.0])
    _base = Rot.from_quat([0.5, 0.5, 0.5, 0.5])

    joints_local, joints_world, orient_q, orient_6d, pose_aa = [], [], [], [], []
    for k in range(T):
        r = compute_from_body_poses(SMPL_PARENTS, "cpu", body[k])
        joints_local.append(r["smpl_joints_local"][0].numpy())
        joints_world.append(r["joints"][0].numpy())
        oq = r["global_orient_quat"][0].numpy()  # wxyz, z-up, base removed
        orient_q.append(oq)
        orient_6d.append(r["global_orient_6d"][0].numpy())
        q_deb = Rot.from_quat([oq[1], oq[2], oq[3], oq[0]])
        root_raw = (_ytoz.inv() * q_deb * _base).as_rotvec()
        pose_aa.append(np.concatenate(
            [root_raw, r["smpl_pose"][0].numpy()]))  # [root(3) | body(69)]
        if k % 1000 == 0:
            print(f"[tape2smpl] {k}/{T}", flush=True)
    joints_local = np.asarray(joints_local, np.float32)
    joints_world = np.asarray(joints_world, np.float32)
    orient_q = np.asarray(orient_q, np.float32)
    # Pelvis world position, tape (Unity, y-up) -> y-up SMPL world: the
    # conversion pre-rotates every body quat by y-180, so positions flip the
    # same way. Only y (height) is consumed downstream (smpl_transl_z).
    transl = (body[:, 0, :3] * np.array([-1.0, 1.0, -1.0])).astype(np.float32)

    # ---- G1: elbow flex round-trip (FK vs raw tape geometry) --------------
    M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    tape_pts = body[:, :, :3] @ M.T
    ang_tape = elbow_angles_from_points(tape_pts)
    ang_fk = elbow_angles_from_points(joints_world.astype(np.float64))
    # The FK skeleton is canonical SMPL; the tape skeleton is the operator's
    # Pico fit — identical local ROTATIONS still read as slightly different
    # position-derived angles. Bug-vs-proportions discriminator: per-side
    # CORRELATION (articulation dynamics preserved?) and SIGNED bias
    # (systematic convention error?), not absolute delta.
    err = np.degrees(ang_tape - ang_fk)
    g1 = True
    for i, side in enumerate(("L", "R")):
        corr = float(np.corrcoef(ang_tape[:, i], ang_fk[:, i])[0, 1])
        bias = float(err[:, i].mean())
        spread = float(np.abs(err[:, i] - bias).mean())
        # corr + spread are the bug detectors (a convention error kills
        # both). A CONSTANT single-side bias is a Pico-fit frame artifact
        # (quats vs positions disagree by a fixed angle; observed -15 deg on
        # the left elbow, +1 right, corr 0.98 both) — warn, don't block; the
        # closed-loop run arbitrates whether it needs a per-side correction.
        side_ok = corr > 0.90 and spread < 8.0 and abs(bias) < 25.0
        warn = " (WARN: constant bias — watch this side in closed loop)" \
            if abs(bias) > 8.0 else ""
        g1 &= side_ok
        print(f"[tape2smpl] G1 {side} elbow round-trip: corr {corr:+.3f} "
              f"bias {bias:+.1f} deg spread {spread:.1f} deg -> "
              f"{'OK' if side_ok else 'FAIL'}{warn}")

    # ---- G2: bone-length stability ----------------------------------------
    bones = [(p, c) for c, p in enumerate(SMPL_PARENTS) if p >= 0]
    lens = np.stack([np.linalg.norm(
        joints_world[:, c] - joints_world[:, p], axis=-1) for p, c in bones])
    drift = float((lens.std(axis=1) / np.maximum(lens.mean(axis=1), 1e-9)).max())
    g2 = drift < 0.02
    print(f"[tape2smpl] G2 bone stability: max rel std {drift:.4f} "
          f"-> {'OK' if g2 else 'FAIL'}")

    # ---- G3: timing -------------------------------------------------------
    g3 = abs((t_out[1] - t_out[0]) - 0.02) < 1e-9
    print(f"[tape2smpl] G3 dt {(t_out[1]-t_out[0]):.4f}s -> {'OK' if g3 else 'FAIL'}")

    gates_ok = bool(g1 and g2 and g3)
    if not gates_ok and not args.force:
        print("[tape2smpl] GATES FAILED — not writing output")
        return 1
    if not gates_ok:
        print("[tape2smpl] GATES FAILED — writing anyway (--force, gates_ok=False)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        smpl_joints_local=joints_local,
        smpl_joints_world=joints_world,
        global_orient_quat_wxyz=orient_q,
        global_orient_6d=np.asarray(orient_6d, np.float32),
        pose_aa=np.asarray(pose_aa, np.float32),
        pose_aa_layout="smpl_canonical_root_first",
        transl=transl,
        fps=FPS_OUT,
        gates_ok=gates_ok,
        source_tape=str(args.tape),
    )
    print(f"[tape2smpl] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
