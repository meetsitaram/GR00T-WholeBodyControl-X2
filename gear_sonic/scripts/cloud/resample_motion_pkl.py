#!/usr/bin/env python3
"""Resample a motion_lib pkl to a target fps by INTERPOLATION.

Why not `downsample_sequence`: it strides with `jump = int(fps_source /
fps_target)`. For 120 -> 50 that truncates to 2, keeping 60 fps of frames while
stamping the label 50 -- the clip then plays 20% slow and its SMPL/robot
timebases diverge. Only integral ratios are safe there; 120/50 = 2.4 is not.

So we resample on a real time axis:
  * `dof` and `root_trans_offset` -> linear interpolation
  * `root_rot` (xyzw quaternions) -> SLERP, because linear interpolation of
    quaternions is not a rotation and denormalises through the arc
  * every other array with a matching leading time axis -> linear
  * `fps` is rewritten to the target so downstream duration math stays honest

Usage:
    python resample_motion_pkl.py --in x2_heldout500.pkl \
        --out x2_heldout500_50fps.pkl --fps 50
"""

from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def resample_entry(entry: dict, fps_dst: float) -> dict:
    fps_src = float(entry.get("fps", 0) or 0)
    if fps_src <= 0:
        raise ValueError("entry has no usable fps")
    # PRESERVE THE SOURCE DTYPE. Interpolation is done in float64 for accuracy,
    # but the corpus is float32 and the FK batch runs in float32 — handing it
    # float64 dies with "expected scalar type Float but found Double" deep in
    # forward_kinematics_batch, surfacing as an UnboundLocalError on `rot_mat`
    # rather than anything mentioning dtype.
    src_dtype = np.asarray(entry["dof"]).dtype
    dof = np.asarray(entry["dof"], dtype=np.float64)
    n_src = dof.shape[0]
    if n_src < 2:
        return entry
    if abs(fps_src - fps_dst) < 1e-9:
        return entry

    duration = (n_src - 1) / fps_src
    n_dst = int(np.floor(duration * fps_dst)) + 1
    t_src = np.arange(n_src) / fps_src
    t_dst = np.arange(n_dst) / fps_dst
    t_dst = np.clip(t_dst, t_src[0], t_src[-1])

    out = dict(entry)

    def lin(a):
        a = np.asarray(a, dtype=np.float64)
        flat = a.reshape(n_src, -1)
        res = np.empty((n_dst, flat.shape[1]), dtype=np.float64)
        for c in range(flat.shape[1]):
            res[:, c] = np.interp(t_dst, t_src, flat[:, c])
        return res.reshape((n_dst,) + a.shape[1:]).astype(src_dtype, copy=False)

    out["dof"] = lin(dof)
    if "root_trans_offset" in entry:
        out["root_trans_offset"] = lin(entry["root_trans_offset"])

    if "root_rot" in entry:
        q = np.asarray(entry["root_rot"], dtype=np.float64)  # xyzw
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        rot = R.from_quat(q / norms)
        out["root_rot"] = Slerp(t_src, rot)(t_dst).as_quat().astype(src_dtype,
                                                                     copy=False)

    for k, v in entry.items():
        if k in ("dof", "root_trans_offset", "root_rot", "fps"):
            continue
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n_src and arr.dtype.kind in "fiu":
            out[k] = lin(arr)

    # match the corpus convention: an integral rate is stored as int
    out["fps"] = int(fps_dst) if float(fps_dst).is_integer() else float(fps_dst)
    return out


def _one_file(job):
    src, dst, fps = job
    try:
        data = joblib.load(src)
        out = {k: resample_entry(v, fps) for k, v in data.items()}
        joblib.dump(out, dst)
        return (True, "")
    except Exception as e:  # one bad clip must not kill a 134k-file sweep
        return (False, f"{os.path.basename(src)}: {str(e)[:70]}")


def run_directory(src_dir: str, dst_dir: str, fps: float, jobs: int) -> int:
    paths = sorted(p for p in glob.glob(os.path.join(src_dir, "**", "*.pkl"),
                                        recursive=True)
                   if os.path.basename(p) != "metadata.pkl")
    if not paths:
        raise SystemExit(f"no *.pkl under {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    print(f"[resample] {len(paths)} source files found", flush=True)

    # On a network FS (virtiofs) a per-file makedirs + exists() is ~2 metadata
    # round-trips x 134k files before any work starts, which looks like a hang.
    # Instead: create each unique parent ONCE, and get the already-done set from
    # a single glob of the destination.
    done = {os.path.relpath(q, dst_dir)
            for q in glob.glob(os.path.join(dst_dir, "**", "*.pkl"), recursive=True)}
    print(f"[resample] {len(done)} already present in destination", flush=True)

    dirs = set()
    todo = []
    for p in paths:
        rel = os.path.relpath(p, src_dir)
        if rel in done:                    # resumable
            continue
        d = os.path.join(dst_dir, rel)
        dirs.add(os.path.dirname(d))
        todo.append((p, d, fps))
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[resample] {len(todo)} to do across {len(dirs)} dirs, "
          f"{jobs} workers", flush=True)
    ok = bad = 0
    errs = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, (good, msg) in enumerate(ex.map(_one_file, todo, chunksize=32), 1):
            if good:
                ok += 1
            else:
                bad += 1
                if len(errs) < 10:
                    errs.append(msg)
            if i % 10000 == 0:
                print(f"[resample] {i}/{len(todo)}  ok={ok} bad={bad}", flush=True)
    print(f"[resample] done: ok={ok} bad={bad}")
    for e in errs:
        print(f"   ! {e}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--jobs", type=int, default=16,
                    help="parallel workers for directory mode")
    args = ap.parse_args()

    if os.path.isdir(args.src):
        return run_directory(args.src, args.dst, args.fps, args.jobs)

    data = joblib.load(args.src)
    print(f"[resample] {len(data)} clips -> {args.fps:g} fps")
    out, changed, dur_err = {}, 0, 0.0
    for k, v in data.items():
        before = v["dof"].shape[0] / float(v.get("fps", args.fps))
        e = resample_entry(v, args.fps)
        after = e["dof"].shape[0] / float(e["fps"])
        dur_err = max(dur_err, abs(before - after))
        if e is not v:
            changed += 1
        out[k] = e
    joblib.dump(out, args.dst)
    print(f"[resample] rewrote {changed}/{len(data)} clips")
    # Duration must survive resampling; a drift here means the time axis is wrong.
    print(f"[resample] max duration drift: {dur_err * 1000:.1f} ms")
    if dur_err > 1.0 / args.fps + 1e-6:
        print("[resample] FAIL: duration drift exceeds one target frame")
        return 1
    print(f"[resample] wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
