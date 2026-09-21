"""Kinematic-sanity gate for motion-lib corpora (companion to the dynamic
`_feasible` filter, which executes clips but never checks kinematic continuity).

Drops clips that fail any of three gates:

  1. manufactured bad-branch residency: frames with root tilt >60 deg from
     upright OR |hip pitch| > 2.0 rad, when the counterpart clip in a
     reference (pre-retarget) corpus is clean at those frames' rate. Catches
     IK wrong-basin output while keeping
     genuine non-upright content (handstands, sitting) that is bad in BOTH.
     Without --ref-pkl the residency gate drops on the retarget side alone.

     *** ALWAYS PASS --ref-pkl. Running without it silently deletes every
     legitimately horizontal motion. Measured on a full corpus build:
     rejection rate tracked POSTURE rather than motion quality --
     Unusual Locomotion 59.3% rejected (1923/3242) and Stunts 34.3%, versus
     Gestures 0.8% and Baseline 1.3%. A 74x spread. ALL 1,874 crawl clips were
     dropped; a crawl-loop clip was then replayed in the MuJoCo
     kinematic viewer and is CLEAN (root z 0.136-0.217 m over 1,273 frames, no
     limb flips). The retargets were fine; this gate threw them away for being
     horizontal, which is what crawling IS. Consequence: the kplanner corpus has
     zero floor-level locomotion while the SONIC flagship demos crawling. ***
  2. kinematic continuity: max single-frame dof jump > --max-jump rad.
  3. root yaw rate: any frame with |yaw rate| > --max-yaw-rate rad/s
     (upright-preserving branch flips escape gate 1; this catches them).

Usage:
    python gear_sonic/data_process/filter_kinematic_continuity.py \
        --in-pkl <x2 retargeted corpus>.pkl \
        --ref-pkl <source (pre-retarget) corpus>.pkl \
        --out-pkl <x2 retargeted corpus>_kinclean.pkl

Reference keys are matched by stripping a trailing `_M` from the input key.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

L_HIP_PITCH, R_HIP_PITCH = 0, 6  # X2 dof indices


def _up_z(quat_xyzw: np.ndarray) -> np.ndarray:
    """World-z component of the body z-axis (1 = upright, <0 = inverted)."""
    x, y = quat_xyzw[:, 0], quat_xyzw[:, 1]
    return 1.0 - 2.0 * (x * x + y * y)


def _yaw(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _bad_frac(entry: dict) -> float:
    dof = np.asarray(entry["dof"])
    q = np.asarray(entry["root_rot"])
    bad = (_up_z(q) < 0.5) | (np.abs(dof[:, L_HIP_PITCH]) > 2.0) | (np.abs(dof[:, R_HIP_PITCH]) > 2.0)
    return float(bad.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-pkl", required=True, type=Path)
    ap.add_argument("--ref-pkl", type=Path, default=None,
                    help="pre-retarget corpus; keys matched by stripping trailing _M")
    ap.add_argument("--out-pkl", required=True, type=Path)
    ap.add_argument("--max-jump", type=float, default=1.0, help="rad, gate 2")
    ap.add_argument("--max-yaw-rate", type=float, default=6.0, help="rad/s, gate 3")
    ap.add_argument("--residency", type=float, default=0.01,
                    help="bad-frame fraction above which gate 1 fires")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    print(f"[kinclean] loading {args.in_pkl} ...", flush=True)
    data = joblib.load(args.in_pkl)
    ref = None
    if args.ref_pkl:
        print(f"[kinclean] loading reference {args.ref_pkl} ...", flush=True)
        ref = joblib.load(args.ref_pkl)

    kept, dropped = {}, Counter()
    drop_log = []
    for key, entry in data.items():
        dof = np.asarray(entry["dof"])
        q = np.asarray(entry["root_rot"])
        fps = float(entry.get("fps", 30))
        reasons = []

        fr = _bad_frac(entry)
        if fr > args.residency:
            ref_clean = True
            if ref is not None:
                rk = key[:-2] if key.endswith("_M") else key
                rv = ref.get(rk)
                ref_clean = rv is None or _bad_frac(rv) < args.residency
            if ref_clean:
                reasons.append("residency")

        if dof.shape[0] >= 2 and np.abs(np.diff(dof, axis=0)).max() > args.max_jump:
            reasons.append("jump")

        yr = np.abs(np.diff(np.unwrap(_yaw(q)))) * fps
        if yr.size and yr.max() > args.max_yaw_rate:
            reasons.append("yaw_rate")

        if reasons:
            dropped["+".join(reasons)] += 1
            drop_log.append((key, "+".join(reasons), fr))
        else:
            kept[key] = entry

    n_in, n_drop = len(data), len(drop_log)
    print(f"[kinclean] {n_in} clips in -> {len(kept)} kept, {n_drop} dropped "
          f"({n_drop / n_in * 100:.1f}%)")
    for combo, n in dropped.most_common():
        print(f"[kinclean]   {combo:24s} {n}")

    log_path = args.out_pkl.with_suffix(".dropped.txt")
    if not args.dry_run:
        with open(log_path, "w") as f:
            f.write("# reason<TAB>bad_frac<TAB>key\n")
            for key, why, fr in sorted(drop_log, key=lambda r: r[0]):
                f.write(f"{why}\t{fr:.3f}\t{key}\n")
        print(f"[kinclean] drop log -> {log_path}", flush=True)
        print(f"[kinclean] writing {args.out_pkl} ...", flush=True)
        joblib.dump(kept, args.out_pkl, compress=3)
        print(f"[kinclean] done: {args.out_pkl}")
    else:
        print("[kinclean] dry run, nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
