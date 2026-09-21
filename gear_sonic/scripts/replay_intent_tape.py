#!/usr/bin/env python3
"""Replay a captured intent tape through the deployed ONNX planner, offline.

Consumes the ``intent_tape.jsonl`` written by ``pc2_kplanner_onnx.py``
(harvested from the robot run log dir): applies the recorded intents at their recorded
monotonic times on a virtual 50 Hz clock, forces each replan to use the
recorded RNG seed, and dumps the published reference stream as a motion-lib
PKL — a bit-exact offline reconstruction of what the robot was asked to do.

View the result (e.g. to find a stumble seam) with::

    python gear_sonic/scripts/play_motion_mujoco.py \
        --motion out/frame_eval/tape_replay/<name>.pkl --speed 0.5

or side-by-side against the robot's actual motion with view_multi_qpos.py.

The backend, contract handling, blend and resample logic are pc2_kplanner's
own classes — imported, not reimplemented — so a divergence between this
replay and the robot means the inputs differed (timing/seed/intent), not the
code.

Usage::

    python gear_sonic/scripts/replay_intent_tape.py \
        --tape <run-log-dir>/intent_tape.jsonl \
        --onnx-dir $SONIC_HOME/x2/kplanner_onnx \
        --out out/frame_eval/tape_replay/<name>.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gear_sonic" / "scripts"))


def load_tape(path: Path) -> dict:
    events = [json.loads(line) for line in open(path) if line.strip()]
    intents = [e for e in events if e["ev"] == "intent_applied"]
    seeds = [e["seed"] for e in events if e["ev"] == "replan_prep"]
    lat_ms = [e["ms"] for e in events if e["ev"] == "replan_done"]
    ticks = [e for e in events if e["ev"] == "tick"]
    t0 = ticks[0]["tm"] if ticks else (intents[0]["tm"] if intents else 0.0)
    return {"events": events, "intents": intents, "seeds": seeds,
            "lat_ms": lat_ms, "ticks": ticks, "t0": t0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tape", required=True, type=Path)
    ap.add_argument("--onnx-dir", required=True, type=Path,
                    help="dir with the deployed planner graph + sidecar "
                         "(same files as PC2 planner_stack/models/planner_onnx)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--output-fps", type=float, default=50.0)
    ap.add_argument("--store-fps", type=float, default=50.0,
                    help="fps stamped into the output pkl")
    ap.add_argument("--max-s", type=float, default=0.0,
                    help="truncate replay after this many seconds (0 = full tape)")
    ap.add_argument("--replan-threshold", type=int, default=16,
                    help="must match the deployed value (config md5 48a8: 16)")
    ap.add_argument("--planner-mode", default="slow_walk",
                    help="must match the deployed mode table entry")
    ap.add_argument("--sim-latency-ms", type=float, default=450.0,
                    help="fallback inference latency when tape has no ms")
    ap.add_argument("--no-fastforward", action="store_true",
                    help="reproduce the ORIGINAL rewind-at-commit behavior")
    args = ap.parse_args()

    import pc2_kplanner_onnx as k  # the deployed runtime itself

    tape = load_tape(args.tape)
    if not tape["ticks"]:
        raise SystemExit("tape has no tick events — was the daemon patched?")
    n_ticks = len(tape["ticks"])
    if args.max_s > 0:
        n_ticks = min(n_ticks, int(args.max_s * args.output_fps))
    print(f"tape: {len(tape['intents'])} intents, {len(tape['seeds'])} replans, "
          f"{len(tape['ticks'])} ticks (replaying {n_ticks})")

    # mirror of pc2_kplanner_onnx.run()'s OnnxPlannerBackend construction
    onnx_path = next(iter(sorted(args.onnx_dir.glob("*.onnx"))), None)
    if onnx_path is None:
        raise SystemExit(f"no .onnx graph in {args.onnx_dir}")
    sidecars = sorted(args.onnx_dir.glob("*.json"))
    contract = k._load_onnx_contract(
        onnx_path, sidecars[0] if sidecars else None)
    backend = k.OnnxPlannerBackend(
        onnx_path=onnx_path,
        contract=contract,
        replan_threshold_frames=args.replan_threshold,
        planner_mode=args.planner_mode,
    )
    warm_path = Path('gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl')
    warm = k._load_warmup_qpos(warm_path if warm_path.exists() else None)
    backend.reset(warm)
    seeds = list(tape["seeds"])
    # deployed warm-the-model pattern: one idle replan (consumes the tape's
    # first recorded seed), then clean re-seed
    if seeds:
        backend._fixed_seed = seeds.pop(0)
    backend.replan((0.0, 0.0, 0.0, float(warm[2])))
    backend._fixed_seed = None
    backend.reset(warm)

    # virtual clock: tick i happens at tape.ticks[i]['tm']; intents apply when
    # their tm passes; each replan consumes the next recorded seed. The idle
    # GATE is mimicked: while idle, the daemon freezes the anchor and the
    # worker skips replans entirely -- so the replay must too, or it would
    # burn recorded seeds during idle stretches. On idle->walk the daemon
    # reseeds the ring at the current root (warm) before the forced replan.
    intents = list(tape["intents"])
    lat = list(tape["lat_ms"])
    pending = None
    backend.commit_fastforward = not args.no_fastforward
    idle = (0.0, 0.0, 0.0)
    target = (0.0, 0.0, 0.0, float(warm[2]))
    was_idle = True
    frames = []
    for i in range(n_ticks):
        now = tape["ticks"][i]["tm"]
        while intents and intents[0]["tm"] <= now:
            target = tuple(intents.pop(0)["target"])
        is_idle = tuple(target[:3]) == idle
        if is_idle:
            frames.append(warm.copy())
            was_idle = True
            continue
        if was_idle:
            backend.reset(warm)   # IDLE -> PLAYING warm reseed (daemon parity)
            backend._force_next_replan = True
            was_idle = False
        if pending is not None and i >= pending["commit_tick"]:
            pred, npf = backend.replan_infer(pending["prep"])
            backend.replan_commit(pred, npf)
            pending = None
        forced = getattr(backend, "_force_next_replan", False)
        if pending is None and (forced or backend.should_replan()):
            backend._force_next_replan = False
            if seeds:
                backend._fixed_seed = seeds.pop(0)
            prep = backend.replan_prepare(target)
            backend._fixed_seed = None
            ms = lat.pop(0) if lat else args.sim_latency_ms
            pending = {"prep": prep,
                       "commit_tick": i + max(1, int(round(ms / 20.0)))}
        # deployed serve path: 30 fps model buffer resampled to 50 Hz output
        # (get_next_frame_resampled advances read_pos and the blend window
        # exactly as on the robot)
        frames.append(backend.get_next_frame_resampled(args.output_fps))

    q = np.asarray(frames, dtype=np.float32)  # [T, 38] trans(3)+wxyz(4)+dof(31)
    quat_xyzw = np.stack([q[:, 4], q[:, 5], q[:, 6], q[:, 3]], 1)
    entry = {
        "root_trans_offset": q[:, :3],
        "root_rot": quat_xyzw,
        "dof": q[:, 7:38],
        "pose_aa": np.zeros((len(q), 32, 3), np.float32),
        "smpl_joints": np.zeros((len(q), 24, 3), np.float32),
        "fps": int(args.store_fps),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({f"TAPE_REPLAY_{args.tape.parent.name}": entry}, args.out)
    print(f"wrote {args.out} ({len(q)} frames @ {args.store_fps:g} fps)")
    print("view: python gear_sonic/scripts/play_motion_mujoco.py "
          f"--motion {args.out} --speed 0.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
