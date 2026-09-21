#!/usr/bin/env python3
"""Build the demo-v1 motion_lib PKL by merging everything under
``gear_sonic/data/motions/demo_v1_sources/<subdir>/*.pkl``.

Staging is uniform PKL (motion_lib schema). Each staged PKL is a
``{motion_key: motion_lib_entry}`` dict — could be single-entry (per-clip
PKL like ``squat_001__A359``) or multi-entry (bundle like
``x2_ultra_dances.pkl`` with 34 entries). This merger iterates
``dict.items()`` for every staged PKL, re-keys each entry with a short
subdir-derived prefix, and writes the merged dict to
``gear_sonic/data/motions/x2_ultra_demo_v1.pkl``.

Prefix policy (kept short so the merged keys stay legible):

  body_check/             ->  bodycheck__<orig_key>
  combat/                 ->  combat__<orig_key>
  dances/                 ->  dance__<orig_key>
  mc_gestures/            ->  gesture__<orig_key>
  retargeted/             ->  walk__<orig_key>
  sitstand/               ->  sitstand__<orig_key>
  teleop_kinematic/       ->  teleop__<orig_key>
  (any other subdir)      ->  <subdir>__<orig_key>   (fallback)

The merger asserts no post-prefix key collisions — if two staged PKLs
contribute the same fully-prefixed key, we bail loudly rather than
silently last-writer-wins.

Run from repo root:
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/data_process/build_x2_demo_motion_lib.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_STAGE = REPO / "gear_sonic" / "data" / "motions" / "demo_v1_sources"
DEFAULT_OUT = REPO / "gear_sonic" / "data" / "motions" / "x2_ultra_demo_v1.pkl"

SUBDIR_PREFIX = {
    "body_check": "bodycheck",
    "combat": "combat",
    "dances": "dance",
    "mc_gestures": "gesture",
    "retargeted": "walk",
    "sitstand": "sitstand",
    "teleop_kinematic": "teleop",
}

REQUIRED_ENTRY_KEYS = {"dof", "fps", "root_rot", "root_trans_offset"}

# Cached X2 axes (populated lazily on first synth call).
_X2_DOF_AXIS: np.ndarray | None = None
_X2_NUM_BODIES: int | None = None


def _entry_frames(entry: dict) -> int:
    """Best-effort frame count, tolerant of either dof or joint_pos field."""
    for k in ("dof", "joint_pos"):
        if k in entry:
            return int(entry[k].shape[0])
    raise ValueError(f"entry has neither 'dof' nor 'joint_pos': keys={sorted(entry)}")


def _ensure_x2_axes_loaded() -> None:
    """Pull DOF_AXIS / NUM_BODIES from convert_soma_csv_to_motion_lib for X2 Ultra.

    Must be called before any synthesize_pose_aa() invocation. Caches at module
    scope so multiprocessing workers (if ever used) won't re-import per call.
    """
    global _X2_DOF_AXIS, _X2_NUM_BODIES
    if _X2_DOF_AXIS is not None:
        return
    from gear_sonic.data_process import convert_soma_csv_to_motion_lib as csm
    csm.set_robot("x2_ultra")
    _X2_DOF_AXIS = np.asarray(csm.DOF_AXIS, dtype=np.float32)  # (NUM_DOF, 3)
    _X2_NUM_BODIES = int(csm.NUM_BODIES)


def synthesize_pose_aa_smpl_joints(entry: dict) -> dict:
    """Add missing `pose_aa` and `smpl_joints` fields, derived from
    ``(dof, root_rot[xyzw])`` using the same formula as
    ``convert_soma_csv_to_motion_lib.convert_sequence`` (X2 Ultra).

    Pure remapping — no FK, no extra data needed. Quaternion convention is
    xyzw (scipy.Rotation.from_quat default), confirmed empirically on the
    walk/gesture/combat PKLs already in the corpus.

    Mutates a copy; returns the augmented entry. Idempotent: if both fields
    already exist, the input is returned unchanged.
    """
    if "pose_aa" in entry and "smpl_joints" in entry:
        return entry
    _ensure_x2_axes_loaded()
    assert _X2_DOF_AXIS is not None and _X2_NUM_BODIES is not None

    dof = np.asarray(entry["dof"], dtype=np.float32)          # (T, NUM_DOF)
    root_rot = np.asarray(entry["root_rot"], dtype=np.float32)  # (T, 4) xyzw
    T = dof.shape[0]
    if dof.shape[1] != _X2_DOF_AXIS.shape[0]:
        raise ValueError(
            f"dof has {dof.shape[1]} dims but X2 Ultra has {_X2_DOF_AXIS.shape[0]}; "
            "this synthesizer is X2-specific."
        )

    augmented = dict(entry)
    if "pose_aa" not in augmented:
        pose_aa = np.zeros((T, _X2_NUM_BODIES, 3), dtype=np.float32)
        pose_aa[:, 1:_X2_NUM_BODIES, :] = _X2_DOF_AXIS[None, :, :] * dof[:, :, None]
        pose_aa[:, 0, :] = R.from_quat(root_rot).as_rotvec().astype(np.float32)
        augmented["pose_aa"] = pose_aa
    if "smpl_joints" not in augmented:
        # Placeholder — matches convert_sequence's behavior for non-SMPL retargets.
        augmented["smpl_joints"] = np.zeros((T, 24, 3), dtype=np.float32)
    return augmented


def merge_stage(stage_dir: Path, *, allow_collisions: bool = False) -> dict:
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"staging dir not found: {stage_dir}")

    merged: dict[str, dict] = {}
    seen_origin: dict[str, str] = {}
    per_subdir_counts: dict[str, tuple[int, int]] = {}  # subdir -> (n_pkls, n_entries)
    total_frames = 0
    synth_count = 0  # entries that needed pose_aa/smpl_joints synthesis

    for sub in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
        prefix = SUBDIR_PREFIX.get(sub.name, sub.name)
        pkls = sorted(sub.glob("*.pkl"))
        if not pkls:
            print(f"  {sub.name:<28}  (empty — skipped)")
            per_subdir_counts[sub.name] = (0, 0)
            continue

        sub_entries = 0
        sub_frames = 0
        for pkl in pkls:
            try:
                bundle = joblib.load(pkl)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"failed to load {pkl}: {e}") from e

            if not isinstance(bundle, dict):
                raise TypeError(
                    f"{pkl}: expected dict, got {type(bundle).__name__} — "
                    "motion_lib PKLs must be {motion_key: entry}"
                )

            for orig_key, entry in bundle.items():
                missing = REQUIRED_ENTRY_KEYS - set(entry.keys())
                if missing:
                    raise ValueError(
                        f"{pkl}:{orig_key} missing required keys {sorted(missing)}; "
                        f"present={sorted(entry.keys())}"
                    )

                # SONIC motion_lib_base.py:1769 indexes curr_file["pose_aa"]
                # unconditionally. For entries built by make_warehouse_motion.py
                # (which omits SMPL fields), synthesize them on the fly from
                # (dof, root_rot) so the merged corpus is uniformly trainable.
                needs_synth = ("pose_aa" not in entry) or ("smpl_joints" not in entry)
                if needs_synth:
                    entry = synthesize_pose_aa_smpl_joints(entry)
                    synth_count += 1

                merged_key = f"{prefix}__{orig_key}"
                if merged_key in merged:
                    msg = (
                        f"key collision on '{merged_key}': "
                        f"first from {seen_origin[merged_key]}, now from {pkl}"
                    )
                    if not allow_collisions:
                        raise RuntimeError(msg)
                    print(f"  WARN  {msg}  (overwriting)")
                merged[merged_key] = entry
                seen_origin[merged_key] = str(pkl.relative_to(REPO))
                sub_entries += 1
                sub_frames += _entry_frames(entry)

        per_subdir_counts[sub.name] = (len(pkls), sub_entries)
        total_frames += sub_frames
        dur_s = sub_frames / 30.0  # all entries are 30 fps
        print(
            f"  {sub.name:<28}  pkls={len(pkls):>2}  entries={sub_entries:>3}  "
            f"frames={sub_frames:>5}  ({dur_s:6.1f} s @ 30fps)"
        )

    total_dur = total_frames / 30.0
    print(
        f"  {'TOTAL':<28}  pkls={sum(c[0] for c in per_subdir_counts.values()):>2}  "
        f"entries={len(merged):>3}  frames={total_frames:>5}  ({total_dur:6.1f} s @ 30fps)"
    )
    if synth_count:
        print(
            f"  (synthesized pose_aa+smpl_joints for {synth_count} entries that "
            f"lacked them — pure remap of dof+root_rot, no FK)"
        )
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--allow-collisions", action="store_true",
        help="WARN instead of FAIL on key collisions (last writer wins).",
    )
    args = ap.parse_args()

    print(f"Staging dir: {args.stage_dir}")
    print(f"Output PKL:  {args.out}")
    print()

    merged = merge_stage(args.stage_dir, allow_collisions=args.allow_collisions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged, args.out, compress=3)
    sz_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {args.out}  ({sz_mb:.2f} MB, {len(merged)} entries)")


if __name__ == "__main__":
    main()
