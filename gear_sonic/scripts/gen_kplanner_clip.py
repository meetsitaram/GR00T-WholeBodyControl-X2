#!/usr/bin/env python3
"""Generate a motion clip by driving the PLANNER ONNX with a scripted intent.

Used by ``deploy_regression_check.sh`` so a regression run validates the planner
models too -- not just sonic. The produced PKL is the planner's own output, which
is then played through the sonic ONNX, exercising the exact deployed pipeline
(planner ONNX -> resampled/blended handoff -> sonic).

The frames come from the same runtime the robot uses (``pc2_kplanner_onnx``'s
backend + resampled read), so the clip reflects the 30->50Hz handoff behaviour.

Usage:
    python gear_sonic/scripts/gen_kplanner_clip.py \
        --planner-onnx $SONIC_HOME/x2/kplanner_onnx/x2_kplanner_template.onnx \
        --out /tmp/kplanner_walk.pkl --seconds 8 --vel-z 0.5            # straight walk
    ... --yaw-rate 0.4                                                  # walk + turn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "motionbricks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUTPUT_FPS = 50.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--vel-z", type=float, default=0.5, help="forward m/s")
    ap.add_argument("--vel-x", type=float, default=0.0, help="lateral m/s")
    ap.add_argument("--yaw-rate", type=float, default=0.0, help="rad/s")
    ap.add_argument("--hip-h", type=float, default=0.687)
    ap.add_argument("--mode", default="slow_walk",
                    help="template mode: idle|slow_walk|walk|run_proxy")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    # Reuse the robot's own runtime backend so the clip reflects deployed behaviour.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pc2_kplanner_onnx", REPO / "gear_sonic" / "scripts" / "pc2_kplanner_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    # Must be registered before exec: the module defines dataclasses, whose type
    # resolution looks the module up in sys.modules.
    sys.modules["pc2_kplanner_onnx"] = mod
    spec.loader.exec_module(mod)  # type: ignore

    contract = mod._load_onnx_contract(args.planner_onnx, None)
    backend = mod.OnnxPlannerBackend(args.planner_onnx, contract, 16, args.mode)

    warm = mod._build_training_default_qpos()
    backend.reset(warm)
    intent = (float(args.yaw_rate), float(args.vel_x), float(args.vel_z), float(args.hip_h))
    backend.replan(intent)

    n = int(args.seconds * OUTPUT_FPS)
    frames = np.zeros((n, 38), dtype=np.float32)
    for i in range(n):
        frames[i] = backend.get_next_frame_resampled(OUTPUT_FPS)
        if backend.should_replan():
            backend.replan(intent)

    name = args.name or f"kplanner_{args.mode}_vz{args.vel_z}_yaw{args.yaw_rate}"
    clip = {
        name: {
            "root_trans_offset": frames[:, :3].copy(),
            "root_rot": frames[:, [4, 5, 6, 3]].copy(),   # wxyz -> xyzw
            "dof": frames[:, 7:].copy(),
            "fps": OUTPUT_FPS,
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clip, args.out)
    disp = float(np.linalg.norm(frames[-1, :2] - frames[0, :2]))
    print(f"wrote {args.out}: '{name}' {n} frames @ {OUTPUT_FPS:.0f}fps, "
          f"planar displacement {disp:.2f} m ({disp/args.seconds:.2f} m/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
