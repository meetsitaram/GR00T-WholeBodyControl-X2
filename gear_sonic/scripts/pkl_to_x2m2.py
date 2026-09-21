#!/usr/bin/env python3
"""Window a motion-lib clip and write it as X2M2 for the dance/clip player.

WHY: to isolate SONIC from the kplanner. The dance path streams a FIXED
reference clip straight to SONIC, bypassing planner generation entirely. If
SONIC tracks a known-good walk cleanly but stumbles on planner output, the
problem is upstream; if it stumbles on both, it is the tracker.

Windowing matters on a gantry: full corpus clips run 10 s and travel metres.
A few seconds of a low-speed turning clip gives a couple of steps and a turn
without crossing the room.

The written file is round-trip verified with the repo's OWN loader
(``gear_sonic.utils.pose_pipeline.wire.load_x2m2``) before it is accepted --
a hand-written binary format that only this script can read would be worse
than useless.

    python gear_sonic/scripts/pkl_to_x2m2.py \
        --pkl gear_sonic/data/motions/x2_demo_bank.pkl \
        --key walking_quip_360_R_002__A428 --start-s 1.0 --dur-s 5.0 \
        --out /tmp/probe_walk_turn.x2m2
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gear_sonic.utils.pose_pipeline.wire import (  # noqa: E402
    NUM_BODY_DOFS, X2M2_MAGIC, load_x2m2)


def write_x2m2(path: Path, dof: np.ndarray, quat_xyzw: np.ndarray,
               fps: float) -> None:
    n, d = dof.shape
    if d != NUM_BODY_DOFS:
        raise ValueError(f"dof has {d} cols, need {NUM_BODY_DOFS}")
    if quat_xyzw.shape != (n, 4):
        raise ValueError(f"quat shape {quat_xyzw.shape}, need ({n}, 4)")
    flat = np.concatenate([dof, quat_xyzw], axis=1).astype(np.float64)
    with path.open("wb") as f:
        f.write(struct.pack("<III", X2M2_MAGIC, n, d))
        f.write(struct.pack("<d", float(fps)))
        f.write(flat.tobytes(order="C"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--key", required=True)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--dur-s", type=float, default=5.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-hand-sidecar", action="store_true",
                    help="skip <out stem>.hand.npz even when the clip carries hand_grip")
    ap.add_argument("--hand-close-on-grip", action="store_true",
                    help="grip 1.0 = CLOSED (sender ran --hand-close-on-grip); default: closed at rest, grip 1.0 = OPEN")
    args = ap.parse_args()

    lib = joblib.load(args.pkl)
    if args.key not in lib:
        cand = [k for k in lib if args.key in k]
        print(f"key {args.key!r} not found; close matches: {cand[:5]}",
              file=sys.stderr)
        return 1
    clip = lib[args.key]
    dof = np.asarray(clip["dof"], dtype=np.float64)
    rot = np.asarray(clip["root_rot"], dtype=np.float64)      # xyzw
    tr = np.asarray(clip["root_trans_offset"], dtype=np.float64)
    fps = float(clip.get("fps", 30.0))

    a = int(round(args.start_s * fps))
    b = min(dof.shape[0], a + int(round(args.dur_s * fps)))
    dof, rot, tr = dof[a:b], rot[a:b], tr[a:b]
    n = dof.shape[0]
    if n < 2:
        print("window too short", file=sys.stderr)
        return 1

    write_x2m2(args.out, dof, rot, fps)

    # OmniHand track (2026-09-05, manipulation recordings): the clip's
    # hand_grip [T,2] (Pico grip analog, teleop_session_to_clips.py) baked to
    # the 10-DOF per-side motor targets the pose wire carries
    # (left/right_hand_joints), as a numpy sidecar next to the x2m2. The x2m2
    # binary itself stays dof+quat (the C++ reader). The kplanner loads
    # <stem>.hand.npz when it plays the clip and feeds the frames through the
    # arm-ingest overlay's hand slots.
    hg = clip.get("hand_grip")
    if hg is not None and not args.no_hand_sidecar:
        from gear_sonic.utils.teleop.x2_hand_retarget import grasp_command_from_ratio
        g = np.asarray(hg, dtype=np.float32)[a:b]
        closed = g if args.hand_close_on_grip else 1.0 - g          # ratio 1 = closed
        lh = np.stack([grasp_command_from_ratio("left", float(r)) for r in closed[:, 0]]).astype(np.float32)
        rh = np.stack([grasp_command_from_ratio("right", float(r)) for r in closed[:, 1]]).astype(np.float32)
        side = args.out.with_suffix(".hand.npz")
        np.savez(side, left_hand_q=lh, right_hand_q=rh, grip=g, fps=np.float64(fps),
                 semantics=np.array("grip analog 0..1 -> closed ratio (%s); q = OmniHand motor targets rad, order HAND_FINGER_NAMES_PER_SIDE"
                                    % ("1=closed" if args.hand_close_on_grip else "closed at rest, 1=open")))
        sq = lambda x: int(np.sum(np.diff((x > 0.5).astype(int)) == 1))
        print(f"  hand sidecar {side.name}: {len(g)} frames, squeezes L {sq(g[:, 0])} R {sq(g[:, 1])}")

    # Round-trip through the REPO loader -- the only check that matters.
    rd, rq, rf = load_x2m2(args.out)
    ok = (rd.shape == dof.shape and rq.shape == rot.shape
          and abs(rf - fps) < 1e-9
          and np.allclose(rd, dof.astype(np.float32), atol=1e-6)
          and np.allclose(rq, rot.astype(np.float32), atol=1e-6))
    disp = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
    path_len = float(np.sum(np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1)))
    jump = float(np.abs(np.diff(dof, axis=0)).max())
    print(f"  {args.key}[{a}:{b}]  {n} frames @ {fps:g}fps = {n/fps:.1f}s")
    print(f"  travel: net {disp:.2f} m, path {path_len:.2f} m")
    print(f"  max per-frame joint jump: {jump:.3f} rad")
    print(f"  round-trip via load_x2m2: {'OK' if ok else '*** MISMATCH ***'}")
    print(f"  wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
