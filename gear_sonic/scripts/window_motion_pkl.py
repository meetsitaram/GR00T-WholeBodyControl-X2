"""Window long motion-lib clips into overlapping sub-clips for kplanner training.

The kplanner datasets (train_{root,vqvae,pose}_x2.py) filter clips by effective
frame count: a clip is DROPPED unless ``min_frames <= ceil(T/subsample) <= max_frames``
(defaults 80 / 200 / subsample=1). Our G1-teleop recordings are ~299 frames
(10 s @ 30 fps), so every one of them would be silently dropped -- and any
priority pkl built from them would then be empty, so the finetune sampler falls
back to a uniform shuffle (the whole point of the priority run is lost).

This tool slices each over-long clip into a few overlapping windows that each
land inside [min_frames, max_frames]. A ~299f clip becomes two ~160f halves
(≈20f overlap). Clips already inside the band are copied through unchanged.

The INPUT pkl is never modified -- a NEW pkl is written. Window keys are
``<origkey>_w0``, ``<origkey>_w1``, ... Clips that pass through keep their key.

    python gear_sonic/scripts/window_motion_pkl.py \
        --in  gear_sonic/data/motions/x2_demo_bank.pkl \
        --out gear_sonic/data/motions/x2_demo_bank_windowed.pkl \
        --max-frames 200 --min-frames 80 --overlap 20

Time-indexed arrays are sliced automatically: any ndarray whose axis-0 length
equals the clip length T is windowed; everything else (scalars like ``fps``) is
copied verbatim.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np


def plan_windows(T: int, max_frames: int, min_frames: int, overlap: int) -> list[tuple[int, int]]:
    """Return [(start, end), ...] covering [0, T) with each window in the band.

    Uses the fewest windows N such that windows of equal length L <= max_frames
    with >= ``overlap`` frames shared between neighbours span the whole clip.
    """
    if T <= max_frames:
        return [(0, T)]
    # Minimal window count so that N windows of length <= max_frames, overlapping
    # by `overlap`, cover T:  N*(L-overlap) + overlap >= T  with L <= max_frames.
    step_max = max_frames - overlap
    n = max(2, math.ceil((T - overlap) / step_max))
    # Equal window length that exactly tiles T with (n-1) overlaps of `overlap`.
    L = math.ceil((T + (n - 1) * overlap) / n)
    L = min(L, max_frames)
    if L < min_frames:
        L = min(min_frames, T)
    starts = [round(i * (T - L) / (n - 1)) for i in range(n)]
    wins = []
    for s in starts:
        s = max(0, min(s, T - L))
        wins.append((s, s + L))
    # Dedup identical windows (can happen for tiny T just over the band).
    out: list[tuple[int, int]] = []
    for w in wins:
        if w not in out:
            out.append(w)
    return out


def window_entry(entry: dict, start: int, end: int) -> dict:
    T = int(entry["root_trans_offset"].shape[0])
    new: dict = {}
    for k, v in entry.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            new[k] = v[start:end].copy()
        else:
            new[k] = v
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", dest="out", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--min-frames", type=int, default=80)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--subsample", type=int, default=1,
                    help="matches dataset subsample; effective frames = ceil(T/subsample)")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    data = joblib.load(args.inp)
    out: dict = {}
    n_pass = n_split = n_win = 0
    bad = []
    for key, entry in data.items():
        T = int(entry["root_trans_offset"].shape[0])
        wins = plan_windows(T, args.max_frames, args.min_frames, args.overlap)
        if len(wins) == 1 and wins[0] == (0, T):
            out[key] = entry
            n_pass += 1
            eff = (T + args.subsample - 1) // args.subsample
            if not (args.min_frames <= eff <= args.max_frames):
                bad.append((key, eff))
            continue
        n_split += 1
        for i, (s, e) in enumerate(wins):
            nk = f"{key}_w{i}"
            out[nk] = window_entry(entry, s, e)
            n_win += 1
            eff = ((e - s) + args.subsample - 1) // args.subsample
            if not (args.min_frames <= eff <= args.max_frames):
                bad.append((nk, eff))

    print(f"input clips:      {len(data)}")
    print(f"  passed through: {n_pass}")
    print(f"  split:          {n_split} -> {n_win} windows")
    print(f"output clips:     {len(out)}")
    if bad:
        print(f"  !! {len(bad)} clips OUT OF BAND [{args.min_frames},{args.max_frames}]:")
        for k, eff in bad[:20]:
            print(f"     {k}: eff={eff}")
    else:
        print(f"  all output clips within [{args.min_frames},{args.max_frames}] effective frames")

    if args.dry_run:
        print("dry-run: not writing")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, args.out, compress=3)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
