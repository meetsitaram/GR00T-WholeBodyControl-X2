#!/usr/bin/env python
"""Append operator-captured WALK-START modes to the G1 clip library.

Gait-initiation fix (any2any S1 notes, 2026-08-13): the frozen G1 core
authors a surging first stride from standstill (core-native; intent
ramps don't fix it — tau 0.6 vs 1.2 A/B, operator). Fix = template
injection, same mechanism as turn-v2: the operator RECORDED smooth
starts in sim (tau=1.2 session, 2026-08-12 ~22:43 tape), we lift the
best standstill->steady-gait windows, Phi-ENCODE them to the G1 side
(they were produced by the wrapped planner, so the encode is
self-consistent by construction), and append them as walk_start modes.
The frozen core's weights are untouched: the library is data.

Windows are chosen from the tape by timestamp (analyze_starts verdicts):
GOOD takes t = 118.2 (right lead), 300.3 (left lead, cleanest),
backups 136.5 / 202.5.

Run:
  PYTHONPATH="$PWD:$PWD/motionbricks" .venv/bin/python \
      motionbricks/scripts/build_g1_walkstart_clip_library.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "motionbricks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.build_g1_turn_clip_library import compute_clip_buffers  # noqa: E402
from scripts.export_g1core_x2_planner_onnx import PhiEncode  # noqa: E402

G1_QPOS = 36
FPS = 30.0
# The operator's 2026-08-12 sim session (first-party), shipped with the repo;
# the take timestamps below index THIS tape.
_DEFAULT_TAPE = REPO_ROOT / "gear_sonic/data/kplanner_tapes/walkstart_20260812.frames.f32"
# (t0_s, tail_s, name) — t0 = motion onset (relaxed detector); window =
# [t0 - lead_s, t0 + tail_s]. TAILS ARE PER-TAKE: the operator stopped
# quickly after each start, so the clip must END MID-GAIT (v ~0.3), not
# in the stop. TAPE IS THE SERVED 50 Hz STREAM — resampled to 30 fps
# below (the 1.67x fps trap; kplanner handoff incident precedent).
_TAKES = [(300.3, 1.2, "walk_start_left"), (118.2, 1.8, "walk_start_right")]
_LEAD_S = 0.8
_TAPE_FPS_TOL = 0.2


def _resample_qpos(q: np.ndarray, t: np.ndarray, fps_out: float) -> np.ndarray:
    """Linear resample [T,38] qpos onto a uniform fps_out grid; quat renorm."""
    tt = np.arange(t[0], t[-1], 1.0 / fps_out)
    out = np.stack([np.interp(tt, t, q[:, j]) for j in range(q.shape[1])], axis=1)
    n = np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    out[:, 3:7] /= np.clip(n, 1e-6, None)
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, default=_DEFAULT_TAPE)
    ap.add_argument("--base-ckpt", type=Path,
                    default=REPO_ROOT / "motionbricks/out/G1-clip-turns.ckpt")
    ap.add_argument("--out-ckpt", type=Path,
                    default=REPO_ROOT / "motionbricks/out/G1-clip-turns-walkstart.ckpt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from motionbricks.motion_backbone.inference.load_g1_planner import (
        G1PlannerPaths, load_g1_models)
    paths = G1PlannerPaths.default()
    paths.validate()
    _, converter = load_g1_models(paths, device=args.device)

    a = np.fromfile(args.tape, dtype=np.float32).reshape(-1, 40)
    # tape rec = [tm, branch, xy, z, quat_XYZW, jpos31] (pc2_kplanner_onnx
    # frame()); reorder quat to the qpos wxyz convention. (First build
    # read cols 5:9 AS wxyz — wrong root orientation in the clips;
    # caught 2026-08-13 via the float-compare FK giving 76cm "stance".)
    t = a[:, 0]
    x2_qpos = np.concatenate(
        [a[:, 2:5], a[:, 5:9][:, [3, 0, 1, 2]], a[:, 9:40]], axis=1)
    print(f"tape: {args.tape.name}, {len(a)} frames, {t[-1]:.0f}s")

    enc = PhiEncode()
    base = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    n_modes_old = int(base["num_frames_per_clip"].shape[0])
    max_old = int(base["mujoco_qpos"].shape[1])
    J = int(base["global_joint_positions"].shape[2])
    print(f"base library: {n_modes_old} modes, max_frames {max_old}, J={J}")

    fps_tape = 1.0 / float(np.median(np.diff(t)))
    print(f"tape fps: {fps_tape:.2f}")
    if abs(fps_tape - 50.0) > 50.0 * _TAPE_FPS_TOL:
        raise ValueError(f"unexpected tape rate {fps_tape:.1f} — check source")

    new_modes: list[tuple[str, dict]] = []
    for t0, tail_s, name in _TAKES:
        si = int(np.searchsorted(t, t0 - _LEAD_S))
        ei = int(np.searchsorted(t, t0 + tail_s))
        x2_30 = _resample_qpos(x2_qpos[si:ei], t[si:ei], FPS)
        win = torch.from_numpy(x2_30)[None]                      # [1,T,38]
        g1 = enc(win)[0].numpy()                                 # [T,36]
        if not np.isfinite(g1).all():
            raise ValueError(f"non-finite encode for {name}")
        # re-anchor: xy to origin at frame 0 (library clips are local)
        g1[:, 0:2] -= g1[0:1, 0:2]
        bufs = compute_clip_buffers(g1.astype(np.float32), converter, args.device)
        v = np.linalg.norm(np.diff(g1[:, 0:2], axis=0), axis=1) * FPS
        print(f"{name}: t=[{t[si]:.1f},{t[ei-1]:.1f}]s {len(g1)}f, "
              f"v end {v[-15:].mean():.2f} m/s (G1 frame)")
        new_modes.append((name, bufs))

    max_new = max(max_old, max(int(b["mujoco_qpos"].shape[0]) for _, b in new_modes))
    n_total = n_modes_old + len(new_modes)
    out = {
        "global_root_positions": torch.zeros(n_total, max_new, 3),
        "global_joint_positions": torch.zeros(n_total, max_new, J, 3),
        "global_joint_rotations": torch.zeros(n_total, max_new, J, 3, 3),
        "global_headings": torch.zeros(n_total, max_new),
        "mujoco_qpos": torch.zeros(n_total, max_new, G1_QPOS),
        "num_frames_per_clip": torch.zeros(
            n_total, dtype=base["num_frames_per_clip"].dtype),
    }
    for k in ("global_root_positions", "global_joint_positions",
              "global_joint_rotations", "global_headings", "mujoco_qpos"):
        out[k][:n_modes_old, :max_old] = base[k]
    out["num_frames_per_clip"][:n_modes_old] = base["num_frames_per_clip"]

    mode_idx = {}
    for i, (name, b) in enumerate(new_modes):
        mi = n_modes_old + i
        n = int(b["mujoco_qpos"].shape[0])
        for k in ("global_root_positions", "global_joint_positions",
                  "global_joint_rotations", "global_headings", "mujoco_qpos"):
            out[k][mi, :n] = b[k]
        out["num_frames_per_clip"][mi] = n
        mode_idx[name] = mi

    torch.save(out, args.out_ckpt)
    sidecar = args.out_ckpt.with_suffix(".modes.json")
    sidecar.write_text(json.dumps(
        {"base": str(args.base_ckpt.name), "source_tape": str(args.tape),
         "takes": _TAKES, "modes": mode_idx}, indent=1))
    print(f"wrote {args.out_ckpt} ({n_total} modes, max_frames {max_new})")
    print(f"walk_start modes: {mode_idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
