#!/usr/bin/env python
"""Build the S1 paired-chunk training cache (frozen-core stage S1, build step 1).

frozen-core S1 heads plan — heads train on PAIRED qpos:
G1 source-corpus CSV (120 fps, native G1 motion) <-> X2 retarget pkl (30 fps,
feasibility-filtered). Pairing is by filename stem; alignment verified
2026-08-12 (24 random pairs: exact length match after ::4 decimation,
lag 0, arm R^2 median 0.996 through the S0 calibration affines).

Per clip this writes  <out>/clips/<key>.pt  with
    g1_qpos  [T, 36] f32   (m, wxyz, rad, 30 fps)
    x2_qpos  [T, 38] f32   (m, wxyz, rad, 30 fps)
and a manifest.json with per-clip gate stats (NOT hard rejects — the
trainer thresholds at load time so gates stay tunable):
    n_frames, mean_speed / p95_speed (X2 root xy, m/s),
    arm_r2 / leg_r2  (pair agreement through analytic Phi — low values
                      flag IK-branch-flip / heavy-deviation retargets),
    jerk_p95         (X2 dof jerk, rad/s^3 — spiky-clip screen),
    split            (train/val by md5(stem), ~5% val, BY CLIP)

Chunking to MotionBricks' native granularity (4 frames/token, segments of
6-16 tokens, random start) happens at TRAIN time — the cache stores whole
aligned clips, mirroring the X2MotionDataset level-A / level-B split.

Run (background, ~10 min with 16 workers):
  PYTHONPATH="$PWD:$PWD/motionbricks" .venv/bin/python \
      motionbricks/scripts/build_frozen_core_pair_cache.py --workers 16
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "motionbricks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.export_g1core_x2_planner_onnx import _PHI_TABLE  # noqa: E402

DEG = np.pi / 180.0
G1_QPOS = 36


def load_g1_csv_qpos(path: Path) -> np.ndarray:
    """Soma G1 CSV -> qpos [T, 36] (m, wxyz quat, rad), 120 fps."""
    d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    T = len(d)
    q = np.zeros((T, G1_QPOS), np.float32)
    q[:, 0:3] = d[:, 1:4] / 100.0
    quat_xyzw = R.from_euler("xyz", d[:, 4:7], degrees=True).as_quat()
    q[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
    q[:, 7:36] = np.deg2rad(d[:, 7:36])
    if not np.isfinite(q).all():
        raise ValueError(f"non-finite values in {path}")
    return q


# Paired corpus (bring-your-own): FROZEN_CORE_X2_PKL = X2 retarget motion-lib pkl,
# FROZEN_CORE_G1_CSV_DIR = the matching native G1 CSVs (same clip keys).
PKL = Path(os.environ.get("FROZEN_CORE_X2_PKL", "")) if os.environ.get("FROZEN_CORE_X2_PKL") else None
G1CSV = Path(os.environ.get("FROZEN_CORE_G1_CSV_DIR", "")) if os.environ.get("FROZEN_CORE_G1_CSV_DIR") else None
FPS = 30.0
MIN_FRAMES = 24          # one min-length MotionBricks segment (6 tokens * 4)

# dof-index rows for pair-agreement stats (x2_idx, g1_idx, a, b_rad)
_ARM_ROWS = [(i, g, a, b * DEG) for i, (n, g, a, b) in enumerate(_PHI_TABLE)
             if g >= 0 and ("shoulder" in n or "elbow" in n)]
_LEG_ROWS = [(i, g, a, b * DEG) for i, (n, g, a, b) in enumerate(_PHI_TABLE)
             if g >= 0 and ("hip" in n or "knee" in n or "ankle" in n)]

_G1_INDEX: dict[str, Path] = {}
_X2_DATA: dict = {}


def _split_of(stem: str, val_frac: float = 0.05) -> str:
    h = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def _stem_of(key: str) -> str | None:
    if key in _G1_INDEX:
        return key
    tail = key.split("__", 1)[1] if "__" in key else key
    return tail if tail in _G1_INDEX else None


def _r2(rows, g1_dof: np.ndarray, x2_dof: np.ndarray) -> float:
    vals = []
    for xi, gi, a, b in rows:
        v = np.var(x2_dof[:, xi])
        if v < 1e-6:
            continue
        vals.append(1.0 - np.var(x2_dof[:, xi] - (a * g1_dof[:, gi] + b)) / v)
    return float(np.mean(vals)) if vals else float("nan")


def _process_one(item):
    key, out_dir = item
    c = _X2_DATA[key]
    stem = _stem_of(key)
    if stem is None:
        return key, None, "no_csv"
    try:
        g1 = load_g1_csv_qpos(_G1_INDEX[stem])[::4]
    except Exception as e:  # noqa: BLE001
        return key, None, f"g1_load: {e}"

    dof = np.asarray(c["dof"], np.float32)
    trans = np.asarray(c["root_trans_offset"], np.float32)
    quat_xyzw = np.asarray(c["root_rot"], np.float32)
    n = min(len(g1), len(dof))
    if n < MIN_FRAMES:
        return key, None, "too_short"
    x2 = np.concatenate(
        [trans[:n], quat_xyzw[:n][:, [3, 0, 1, 2]], dof[:n]], axis=1
    ).astype(np.float32)
    g1 = g1[:n]
    if not (np.isfinite(x2).all() and np.isfinite(g1).all()):
        return key, None, "non_finite"

    v = np.linalg.norm(np.diff(trans[:n, :2], axis=0), axis=1) * FPS
    jerk = np.diff(dof[:n], n=3, axis=0) * FPS**3 if n > 3 else np.zeros((1, 1))
    stats = {
        "n_frames": int(n),
        "mean_speed": round(float(v.mean()), 4),
        "p95_speed": round(float(np.percentile(v, 95)), 4),
        "arm_r2": round(_r2(_ARM_ROWS, g1[:, 7:], x2[:, 7:]), 4),
        "leg_r2": round(_r2(_LEG_ROWS, g1[:, 7:], x2[:, 7:]), 4),
        "jerk_p95": round(float(np.percentile(np.abs(jerk), 95)), 2),
        "split": _split_of(stem),
    }
    safe = key.replace("/", "__")
    torch.save({"g1_qpos": torch.from_numpy(g1.astype(np.float32)),
                "x2_qpos": torch.from_numpy(x2)},
               out_dir / f"{safe}.pt")
    return key, stats, None


def main() -> int:
    if PKL is None or G1CSV is None:
        raise SystemExit("build_frozen_core_pair_cache: set FROZEN_CORE_X2_PKL=<x2 motion-lib pkl> and "
                         "FROZEN_CORE_G1_CSV_DIR=<dir of native G1 CSVs with the same clip keys> "
                         "(paired corpus is bring-your-own; it is not part of this repo)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("X2_EVAL_ROOT", Path.home() / ".cache/sonic/x2_eval")) / "frozen_core_s1/pair_cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N keys")
    args = ap.parse_args()

    clips_dir = args.out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print("indexing G1 csvs ...", flush=True)
    for p in G1CSV.glob("*/*.csv"):
        _G1_INDEX[p.stem] = p
    print(f"  {len(_G1_INDEX)} stems", flush=True)

    print(f"loading {PKL.name} ...", flush=True)
    t0 = time.time()
    _X2_DATA.update(joblib.load(PKL))
    keys = sorted(_X2_DATA.keys())
    if args.limit:
        keys = keys[: args.limit]
    print(f"  {len(keys)} clips in {time.time()-t0:.0f}s", flush=True)

    manifest, errors = {}, {}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (key, stats, err) in enumerate(
            pool.imap_unordered(_process_one, ((k, clips_dir) for k in keys),
                                chunksize=16)):
            if err:
                errors[key] = err
            else:
                manifest[key] = stats
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(keys)}  ({rate:.0f} clips/s, "
                      f"{len(errors)} errors)", flush=True)

    (args.out / "manifest.json").write_text(json.dumps(
        {"source_pkl": str(PKL), "fps": FPS, "min_frames": MIN_FRAMES,
         "n_clips": len(manifest), "errors": errors, "clips": manifest},
        indent=0))
    n_val = sum(1 for s in manifest.values() if s["split"] == "val")
    arm = np.array([s["arm_r2"] for s in manifest.values() if np.isfinite(s["arm_r2"])])
    leg = np.array([s["leg_r2"] for s in manifest.values() if np.isfinite(s["leg_r2"])])
    print(f"\ndone: {len(manifest)} clips ({n_val} val), {len(errors)} errors "
          f"-> {args.out}")
    print(f"arm_r2: p5 {np.percentile(arm, 5):.3f}  p50 {np.percentile(arm, 50):.3f}")
    print(f"leg_r2: p5 {np.percentile(leg, 5):.3f}  p50 {np.percentile(leg, 50):.3f}")
    band = sum(1 for s in manifest.values() if 0.05 <= s["mean_speed"] < 0.6)
    print(f"demo-band clips (0.05-0.6 m/s): {band}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
