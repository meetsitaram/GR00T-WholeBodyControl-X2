#!/usr/bin/env python3
"""Pre-compute the X2 MotionBricks FK feature cache in parallel.

The default `X2MotionDataset.__init__` runs `X2MujocoFkExtractor` serially over
every clip in the source PKL on first construction; on a 38k-clip source
corpus that takes ~30+ minutes single-threaded. Running this once locally with
N CPU workers populates the on-disk cache so subsequent training launches (and
the cloud bundle that ships ``feature_cache/`` to the H200) hit hot data.

Output layout (matches what ``X2MotionDataset._load_from_cache`` expects)::

    <out_dir>/
      manifest.json   # {"keys": [...], "count": N}
      <safe_key>.pt   # torch.save'd Tensor[T_eff, D]

Usage (recommended — wipes any existing cache for the new PKL)::

    python scripts/build_feature_cache_x2.py \
      --pkl ../gear_sonic/data/motions/x2_demo_bank.pkl \
      --out-dir out/motionbricks_vqvae_x2/version_1/feature_cache \
      --workers 24 \
      --recompute

Symlink the populated cache into pose+root version_1/ to avoid 3x disk bloat
(``build_planner_bundle.sh`` already does this on the cloud node).
"""

from __future__ import annotations

import os

# CRITICAL: cap per-worker BLAS/OMP threads BEFORE importing torch/numpy.
# Each ProcessPool worker otherwise inherits torch's default thread pool sized
# to nproc. On a 128-vcpu cloud node with 21 workers, that's 21 * 64 = 1,344
# threads competing for 128 cores → kernel scheduler thrash, ~150x slowdown,
# zero progress for >10 min. Defaults below give 21 workers * 4 threads ≈ 84
# threads on 128 cores (clean) and 32 workers * 1 thread = 32 threads on a
# 32-core local box. Override OMP_NUM_THREADS if your node has very different
# topology. See docs/source/user_guide/train-planner-on-cloud.md §5a callout.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import joblib

from motionbricks.data.clip_split import slice_clip, window_bounds
import torch
from omegaconf import OmegaConf, open_dict

# Belt-and-suspenders: also cap torch's intra-op pool explicitly. The env vars
# above govern OpenMP/MKL/OpenBLAS; this line caps torch.set_num_threads which
# some torch builds use independently.
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

MB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MB_ROOT))

from motionbricks.data.x2_category_dataset import (  # noqa: E402
  _safe_key,
  default_cache_dir_for,
)
from motionbricks.data.x2_loco_filters import (  # noqa: E402
  DEFAULT_EXCLUDE_PATTERNS,
  DEFAULT_INCLUDE_PATTERNS,
  filter_motion_keys,
)
from motionbricks.data.x2_pkl_to_motion import (  # noqa: E402
  X2MujocoFkExtractor,
  default_x2_mjcf_path,
)
from motionbricks.helper.pl_util import load_motion_rep  # noqa: E402


# Module-level state populated lazily inside each worker process. Avoids
# pickling MuJoCo + motion_rep across the fork boundary.
_WORKER_STATE: dict = {}


def _init_worker(version_dir_str: str, mjcf_str: str) -> None:
  """Per-process init — load MJCF, motion_rep, and FK extractor once per worker.

  CRITICAL: workers must NOT load the source PKL — that would blow up memory by
  N_workers× (1.5 GB on locowalk → 36 GB at 24 workers, killed the box once).
  Instead, the main process loads the PKL once and pipes per-clip dicts in.
  """
  version_dir = Path(version_dir_str)
  hparams_path = version_dir / "hparams.yaml"
  conf = OmegaConf.load(hparams_path)
  with open_dict(conf):
    conf.skeleton.folder = str(version_dir / "skeleton")
    conf.motion_rep.stats.folder = str(version_dir / "stats" / "motion")

  motion_rep = load_motion_rep(conf)
  extractor = X2MujocoFkExtractor(Path(mjcf_str))
  _WORKER_STATE["motion_rep"] = motion_rep
  _WORKER_STATE["extractor"] = extractor


def _warmup_ping() -> str:
  """No-op task used to force eager fork of all workers BEFORE PKL load."""
  return "ready"


def _read_meminfo_gb() -> Tuple[float, float, float]:
  """Returns (total_gb, used_gb, available_gb) from /proc/meminfo (Linux)."""
  vals: dict = {}
  with open("/proc/meminfo") as f:
    for line in f:
      k, _, rest = line.partition(":")
      vals[k.strip()] = int(rest.split()[0])  # all values in kB
  total_kb = vals.get("MemTotal", 0)
  avail_kb = vals.get("MemAvailable", 0)
  used_kb = total_kb - avail_kb
  return total_kb / 1024 / 1024, used_kb / 1024 / 1024, avail_kb / 1024 / 1024


def _autoscale_workers(
  requested_workers: int,
  pkl_total_gb: float,
  max_mem_gb: int,
) -> int:
  """Probe RAM and clamp worker count so estimated peak stays under budget.

  Empirically, each worker process holds ~2.5 GB RSS (mostly torch + motion_rep
  + per-job clip dict — NOT the PKL, since we fork before loading it). Main
  holds the PKL plus ~1.5 GB Python/torch overhead.
  """
  total_gb, used_gb, _ = _read_meminfo_gb()
  WORKER_GB = 2.5
  MAIN_OVERHEAD_GB = 1.5

  est_peak = used_gb + MAIN_OVERHEAD_GB + pkl_total_gb + requested_workers * WORKER_GB
  print("Memory probe:")
  print(f"  total system    : {total_gb:.1f} GB")
  print(f"  used at start   : {used_gb:.1f} GB")
  print(f"  PKL(s) on disk  : {pkl_total_gb:.1f} GB")
  print(f"  est peak (W={requested_workers}): {est_peak:.1f} GB  (budget {max_mem_gb} GB)")

  if est_peak <= max_mem_gb:
    return requested_workers

  budget_for_workers = max_mem_gb - used_gb - MAIN_OVERHEAD_GB - pkl_total_gb
  scaled = max(1, int(budget_for_workers / WORKER_GB))
  scaled = min(scaled, requested_workers)
  print(
    f"  WOULD EXCEED budget -> auto-scaling workers {requested_workers} -> {scaled}"
  )
  if scaled < 2:
    raise SystemExit(
      f"FATAL: only room for {scaled} worker under {max_mem_gb} GB budget. "
      f"Free up RAM (kill Cursor/Chrome) or raise --max-mem-gb."
    )
  return scaled


def _spawn_mem_monitor(period_sec: int, max_mem_gb: int):
  """Daemon thread that logs memory every period_sec; aborts if used > budget+5GB."""
  import os
  import threading
  import time as _time

  if period_sec <= 0:
    return None
  stop = threading.Event()

  def _loop():
    while not stop.wait(period_sec):
      total, used, avail = _read_meminfo_gb()
      mark = "  " if used < max_mem_gb else "!!"
      print(f"  [mem] {mark} used={used:.1f}GB avail={avail:.1f}GB  (budget {max_mem_gb})")
      if used > max_mem_gb + 5:
        print("  [mem] !! EMERGENCY ABORT: used > budget+5GB; sending SIGTERM to self")
        os.kill(os.getpid(), 15)
        return

  t = threading.Thread(target=_loop, daemon=True)
  t.start()
  return stop


def _process_one(
  key: str,
  clip: dict,
  out_dir: str,
  subsample: int,
  min_frames: int,
  max_frames: int,
  normalize: bool,
  skip_if_exists: bool = False,
) -> Tuple[str, Optional[str]]:
  """Run FK + motion_rep on one already-loaded clip dict (~80 KB pickled)."""
  if skip_if_exists:
    pt_path = Path(out_dir) / f"{_safe_key(key)}.pt"
    if pt_path.is_file() and pt_path.stat().st_size > 0:
      return key, "ok_cached"
  n_raw = int(clip["dof"].shape[0])
  n_eff = (n_raw + subsample - 1) // subsample
  if n_eff < min_frames or n_eff > max_frames:
    return key, "out_of_band"

  try:
    extractor = _WORKER_STATE["extractor"]
    motion_rep = _WORKER_STATE["motion_rep"]
    inp = extractor.clip_to_input_dict(clip, subsample=subsample)
    feat = motion_rep(inp, to_normalize=normalize, return_numpy=False)
    if feat.dim() == 3:
      feat = feat.squeeze(0)
  except Exception as exc:  # noqa: BLE001
    return key, f"error:{exc.__class__.__name__}"

  torch.save(feat, Path(out_dir) / f"{_safe_key(key)}.pt")
  return key, "ok"


def main() -> None:
  parser = argparse.ArgumentParser(description="Parallel X2 FK feature-cache builder")
  parser.add_argument(
    "--pkl",
    nargs="+",
    required=True,
    help="One or more X2 motion-lib PKLs (each emits its own keyspace).",
  )
  parser.add_argument(
    "--version-dir",
    default=str(MB_ROOT / "out" / "motionbricks_vqvae_x2" / "version_1"),
    help="Path to motionbricks version_1 dir (must contain hparams.yaml + skeleton/ + stats/).",
  )
  parser.add_argument(
    "--out-dir",
    default=None,
    help="Cache output dir. Defaults to <version-dir>/feature_cache.",
  )
  parser.add_argument(
    "--mjcf",
    default=None,
    help="Path to X2 MJCF. Defaults to default_x2_mjcf_path().",
  )
  parser.add_argument("--workers", type=int, default=12)
  parser.add_argument(
    "--in-flight",
    type=int,
    default=128,
    help="Max number of submitted-but-not-completed jobs (caps pickled-arg memory).",
  )
  parser.add_argument(
    "--max-mem-gb",
    type=int,
    default=None,
    help="Total system-memory ceiling in GB. The builder probes /proc/meminfo "
    "at start and auto-scales --workers down so estimated peak (current_used + "
    "main_overhead + pkl_size + workers*2.5GB) stays under this. DEFAULT: 70%% "
    "of detected system RAM. The old fixed default of 70 GB was tuned for the "
    "91 GB workstation (~30 GB idle under Cursor+Chrome) and silently crippled "
    "cloud builds — on a 1,574 GB node it scaled 100 workers down to 12, an "
    "~8x slowdown for no reason (found 2026-08-07).",
  )
  parser.add_argument(
    "--mem-poll-sec",
    type=int,
    default=30,
    help="How often (sec) to log memory usage during the run; 0 to disable.",
  )
  parser.add_argument("--min-frames", type=int, default=80)
  parser.add_argument("--max-frames", type=int, default=300)
  parser.add_argument(
    "--subsample", type=int, default=1,
    help="Keep every Nth frame. DEFAULT 1 IS CORRECT — leave it alone. "
         "This flag is an X2-port artifact and does not exist upstream: "
         "MotionBricks states its data is 30 FPS and never resamples. It was "
         "added only because our direct retarget once landed at 120 FPS. That "
         "corpus has since been resampled once, offline, to a true 30 FPS "
         "(x2_direct_motionlib_30fps, via scripts/cloud/resample_motion_pkl.py), "
         "so clips now carry fps=30 and need no stride. "
         "DEPRECATED: remove this flag once no 120 FPS corpus is in use. "
         "Setting it >1 on an already-30 FPS corpus silently trains on motion "
         "N times too fast, and nothing downstream detects it.")
  parser.add_argument(
    "--filter",
    choices=["none", "loco"],
    default="none",
    help="Optional regex filter; default 'none' caches every clip in the PKLs.",
  )
  parser.add_argument(
    "--recompute",
    action="store_true",
    help="Wipe any existing manifest/.pt files before building.",
  )
  parser.add_argument(
    "--no-normalize",
    action="store_true",
    help="Disable motion_rep normalization (rarely useful — diagnostics only).",
  )
  args = parser.parse_args()

  version_dir = Path(args.version_dir)
  if not (version_dir / "hparams.yaml").is_file():
    raise FileNotFoundError(
      f"Missing {version_dir/'hparams.yaml'}. "
      "Run scripts/build_x2_skeleton_assets.py first."
    )

  # Per-PKL cache dir convention — keeps caches built from different PKLs in
  # different sibling directories so the dataset class can never silently use
  # a smoke-PKL cache for full-PKL training (the PKL stem is the dir name).
  out_dir = (
    Path(args.out_dir)
    if args.out_dir
    else default_cache_dir_for(version_dir, args.pkl)
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  manifest_path = out_dir / "manifest.json"
  if args.recompute:
    print(f"--recompute: wiping {out_dir}/")
    for p in out_dir.glob("*.pt"):
      p.unlink()
    if manifest_path.is_file():
      manifest_path.unlink()

  mjcf_path = Path(args.mjcf) if args.mjcf else default_x2_mjcf_path()

  if args.filter == "loco":
    include = DEFAULT_INCLUDE_PATTERNS
    exclude = DEFAULT_EXCLUDE_PATTERNS
  else:
    include = exclude = None

  pkl_total_gb = sum(Path(p).stat().st_size for p in args.pkl) / 1024**3
  max_mem_gb = args.max_mem_gb
  if max_mem_gb is None:
    # Scale the ceiling to the machine instead of the workstation it was first
    # written for. 70% leaves ample headroom on both a 91 GB laptop (63 GB) and
    # a 1,574 GB cloud node (1,101 GB).
    total_gb = 0.0
    try:
      with open("/proc/meminfo") as fh:
        for line in fh:
          if line.startswith("MemTotal:"):
            total_gb = int(line.split()[1]) / 1024**2
            break
    except OSError:
      pass
    max_mem_gb = int(total_gb * 0.7) if total_gb else 70
    print(f"  --max-mem-gb not set -> {max_mem_gb} GB (70% of {total_gb:.0f} GB)")
  workers = _autoscale_workers(args.workers, pkl_total_gb, max_mem_gb)

  results: List[str] = []
  status_counts: dict = {}
  mon_stop = _spawn_mem_monitor(args.mem_poll_sec, max_mem_gb)

  with ProcessPoolExecutor(
    max_workers=workers,
    initializer=_init_worker,
    initargs=(str(version_dir), str(mjcf_path)),
  ) as ex:
    # CRITICAL: warm up all workers BEFORE loading PKLs in main. This forks
    # the workers from a clean main heap, so the (potentially multi-GB)
    # joblib-loaded clip dicts never enter their address space via COW.
    # Without this, on a 1.5 GB locowalk PKL, 16 workers × ~3 GB de-shared
    # heap = ~48 GB unique RAM — risky on a 91 GB box.
    print(f"Warming up {workers} worker processes (forks BEFORE PKL load)...")
    warmup = [ex.submit(_warmup_ping) for _ in range(workers)]
    for f in warmup:
      f.result()
    print("  workers ready.")

    libs: dict[str, dict] = {}
    job_keys: List[Tuple[str, str]] = []
    for pkl in args.pkl:
      print(f"Scanning {pkl} ...")
      lib = joblib.load(pkl)
      libs[pkl] = lib
      raw_keys = list(lib.keys())
      if include is None and exclude is None:
        keys = raw_keys
      else:
        keys = filter_motion_keys(
          raw_keys,
          include_patterns=include or (r".",),
          exclude_patterns=exclude or (),
        )
      keys = sorted(keys)
      print(f"  {len(keys):,} / {len(raw_keys):,} keys after filter={args.filter}")
      # SONIC-style long-clip windowing (2026-08-02): clips longer than
      # max_frames are split into consecutive windows instead of being
      # discarded. job_keys carries (pkl, source_key, suffix, s, e).
      n_windowed = 0
      for k in keys:
        n_raw = int(libs[pkl][k]["dof"].shape[0])
        wins = window_bounds(n_raw, args.subsample, args.min_frames,
                             args.max_frames)
        if len(wins) > 1:
          n_windowed += 1
        for suffix, ws, we in wins:
          job_keys.append((pkl, k, suffix, ws, we))
      if n_windowed:
        print(f"  {n_windowed:,} long clips split into windows "
              f"(max_frames={args.max_frames})")

    total = len(job_keys)
    print(f"Total jobs queued : {total:,}")
    print(f"Workers           : {workers}")
    print(f"In-flight cap     : {args.in_flight}")
    print(f"Output            : {out_dir}")
    print(f"MJCF              : {mjcf_path}")

    job_iter = iter(job_keys)
    in_flight: dict = {}  # Future -> key

    def _submit_next() -> bool:
      try:
        pkl, key, suffix, ws, we = next(job_iter)
      except StopIteration:
        return False
      clip = libs[pkl][key]
      if suffix:
        clip = slice_clip(clip, ws, we)
      fut = ex.submit(
        _process_one,
        key + suffix,
        clip,
        str(out_dir),
        args.subsample,
        args.min_frames,
        args.max_frames,
        not args.no_normalize,
        not args.recompute,  # skip clips already cached on resume
      )
      in_flight[fut] = key + suffix
      return True

    for _ in range(min(args.in_flight, total)):
      if not _submit_next():
        break

    done = 0
    while in_flight:
      for fut in as_completed(list(in_flight.keys())):
        key = in_flight.pop(fut)
        try:
          rkey, status = fut.result()
        except Exception as exc:  # noqa: BLE001
          rkey, status = key, f"error:{exc.__class__.__name__}"
        done += 1
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in ("ok", "ok_cached"):
          results.append(rkey)
        if done % 500 == 0 or done == total:
          err_count = sum(
            v for k, v in status_counts.items() if k.startswith("error")
          )
          print(
            f"  [{done:>6,} / {total:,}]  ok={status_counts.get('ok', 0):,}  "
            f"out_of_band={status_counts.get('out_of_band', 0):,}  "
            f"errors={err_count:,}"
          )
        _submit_next()
        # break out of as_completed so we re-iterate over the freshly-topped-up
        # in_flight set (avoids stale-iterator issues from list mutation).
        break

  if mon_stop is not None:
    mon_stop.set()

  results.sort()
  # Record the windowing parameters ALONGSIDE the keys. Without this the cache
  # is opaque: the trainer's own --max_frames is only consulted on a cache MISS,
  # so a cache built at one window length silently overrides every later launch
  # command (found 2026-08-07 -- cache built at 300 while every launch said 200
  # and the model can consume at most 65). The dataset cross-checks this on HIT.
  manifest_path.write_text(
    json.dumps({
      "keys": results,
      "count": len(results),
      "build": {
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "subsample": args.subsample,
      },
    }, indent=2)
  )
  print(f"\nWrote {manifest_path}")
  print(f"Cached {len(results):,} / {total:,} clips")
  print("Status breakdown:")
  for k, v in sorted(status_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:,}")
  _, used_gb, _ = _read_meminfo_gb()
  print(f"Final mem: used={used_gb:.1f}GB")


if __name__ == "__main__":
  main()
