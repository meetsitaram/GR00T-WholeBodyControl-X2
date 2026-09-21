"""Stage a retargeted Pico teleop corpus into the SONIC fine-tune set.

Produces two artifacts:

  1. a merged fine-tune pkl  = <base> PLUS the new clips, APPEND-ONLY
  2. SMPL sidecars           = one <key>.pkl per clip, in smpl_filtered format

APPEND-ONLY IS LOAD-BEARING, not tidiness. Adaptive-sampling bins are laid
out per-motion-per-frame-window, so the saved episode/failure statistics are
only restorable if the new corpus is the old one with clips appended at the
END. Rebuilding from scratch or re-sorting silently resets the sampler to
uniform (see motion_lib_base.load_state_dict). This script therefore writes
base keys first, in their original order, then the new keys -- and refuses to
overwrite an existing key, since an in-place replacement can change a clip's
frame count and shift every downstream bin.

SIDECAR CONTRACT (motion_lib_base ~line 2074): `fps` and `pose_aa` are read
UNCONDITIONALLY; `smpl_joints` and `transl` fall back to zeros if absent. All
of them are reconciled against the robot frame count with a HARD 2-frame
tolerance -- exceed it and the loader raises at startup and takes the whole
distributed run down (this is what killed v11 at iteration 2249).

Usage:
  python gear_sonic/data_process/build_pico_finetune_bundle.py \
      --corpus-dir <retargeted pico corpus dir (per-clip motion-lib pkls)> \
      --tape-dir   <pico tape sessions dir (*.npz)> \
      --base-pkl   <base fine-tune pkl> \
      --out-pkl    <merged fine-tune pkl, e.g. x2_ft_pico.pkl> \
      --sidecar-dir <SMPL sidecar output dir> \
      --keep-json  <validate output>.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np

SMPL_LEN_TOLERANCE_FRAMES = 2
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- despike -------------------------------------------------------------
# The retargeter runs its despiker BEFORE the analytic arm/foot passes, so
# wrap flips those passes introduce (a wrist yaw jumping ~pi between two
# frames, 100+ rad/s at 50 fps) survive into the per-clip pkls: 1-3 frame
# events on a handful of joints. Fixed here per joint by linear
# interpolation over the flagged frames; pose_aa is derived (X2_DOF_AXIS *
# dof) and regenerated from the corrected dof. Real teleop motion stays
# under ~20 rad/s, the artifacts run 26-256 rad/s, so 0.5 rad/frame
# (25 rad/s) separates them cleanly.
JUMP_THR_RAD_PER_FRAME = 0.5


def despike_dof(dof, jump_thr=JUMP_THR_RAD_PER_FRAME, max_passes=4):
    """Per-joint: flag frames whose step from the previous frame exceeds
    jump_thr rad and interpolate them from good neighbours. Per joint (not
    whole frame) so a shoulder-yaw flip does not resample the legs. Iterates
    because interpolating a flagged window can leave a just-over-threshold
    step at its edges. Returns (dof, n_frames_fixed)."""
    out = np.array(dof, dtype=np.float64, copy=True)
    n_fixed = 0
    t = np.arange(len(out), dtype=float)
    for _ in range(max_passes):
        fixed = 0
        for j in range(out.shape[1]):
            col = out[:, j]
            step = np.abs(np.diff(col)) > jump_thr
            if not step.any():
                continue
            bad = np.zeros(len(col), dtype=bool)
            bad[1:] |= step
            bad[:-1] |= step
            good = np.where(~bad)[0]
            if len(good) < 2:
                continue
            out[:, j] = np.interp(t, t[good], col[good])
            fixed += int(bad.sum())
        n_fixed += fixed
        if fixed == 0:
            break
    return out.astype(np.asarray(dof).dtype), n_fixed


def regen_pose_aa(dof):
    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import X2_DOF_AXIS, X2_NUM_DOF
    pose_aa = np.zeros((len(dof), X2_NUM_DOF + 1, 3), dtype=np.float32)
    pose_aa[:, 1:] = np.asarray(X2_DOF_AXIS, np.float32)[None] * dof[:, :, None].astype(np.float32)
    return pose_aa


def despike_clip(c, jump_thr=JUMP_THR_RAD_PER_FRAME):
    """Despike one motion-lib clip dict in place; returns frames fixed."""
    dof, n = despike_dof(np.asarray(c["dof"]), jump_thr)
    if n:
        c["dof"] = dof
        if "pose_aa" in c:
            c["pose_aa"] = regen_pose_aa(dof)
    return n


def tape_to_smpl(tape_path, python_exe, tmpdir):
    """Run the existing pico->SMPL converter. It owns the frame conventions
    (xyzw->wxyz, y-up->z-up, SMPL-24 parent chain) and self-validates with
    elbow round-trip / bone-stability / dt gates -- do not reimplement it."""
    out = os.path.join(tmpdir, os.path.basename(tape_path).replace(".npz", "_smpl.npz"))
    r = subprocess.run(
        [python_exe, "-m", "gear_sonic.scripts.pico_tape_to_smpl_obs",
         "--tape", tape_path, "--out", out],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        lines = (r.stdout + r.stderr).strip().splitlines()
        fails = [l for l in lines if "FAIL" in l] or lines[-2:]
        return None, fails
    return np.load(out, allow_pickle=True), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--tape-dir", required=True)
    ap.add_argument("--base-pkl", required=True)
    ap.add_argument("--out-pkl", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--keep-json", default=None,
                    help="validate_pico_corpus.py --json output; uses its keep-list")
    ap.add_argument("--python", default=os.path.join(REPO_ROOT, ".venv", "bin", "python"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-despike", action="store_true",
                    help="keep retarget wrap spikes as-is (default: despike each added clip)")
    ap.add_argument("--jump-thr", type=float, default=JUMP_THR_RAD_PER_FRAME,
                    help="despike threshold, rad per frame (default 0.5 = 25 rad/s at 50 fps)")
    a = ap.parse_args()

    keep = None
    if a.keep_json:
        keep = set(json.load(open(a.keep_json))["keep"])
        print(f"keep-list: {len(keep)} clips")

    print(f"loading base: {a.base_pkl}")
    base = joblib.load(a.base_pkl)
    base_keys = list(base.keys())
    print(f"  {len(base_keys)} base clips")

    os.makedirs(a.sidecar_dir, exist_ok=True)
    added, skipped, no_sidecar, sidecars = {}, [], [], 0
    tmpdir = tempfile.mkdtemp(prefix="pico_smpl_")

    files = sorted(f for f in os.listdir(a.corpus_dir) if f.endswith(".pkl"))
    for i, fn in enumerate(files, 1):
        clips = joblib.load(os.path.join(a.corpus_dir, fn))
        for key, c in clips.items():
            if keep is not None and key not in keep:
                skipped.append((key, "not in keep-list")); continue
            if key in base:
                # never silently replace: a same-key clip of a different length
                # shifts every downstream adaptive-sampling bin.
                skipped.append((key, "COLLIDES with a base key")); continue

            n_robot = np.asarray(c["dof"]).shape[0]
            tape = os.path.join(a.tape_dir, fn.replace("pico_", "session_").replace(".pkl", ".npz"))
            if not os.path.exists(tape):
                skipped.append((key, "no session tape")); continue

            # A SIDECAR FAILURE IS NOT A CLIP FAILURE. The robot track was
            # validated independently by the retargeter; the SMPL conversion
            # gates judge only the tape's body tracking. Directory mode appends
            # None for a missing sidecar, so a clip without one simply carries
            # no SMPL track and still trains in g1 mode. Dropping it instead
            # would throw away good motion over an unrelated problem.
            z, err = tape_to_smpl(tape, a.python, tmpdir)
            pose_aa, delta = None, None
            if z is None:
                no_sidecar.append((key, f"smpl gates: {'; '.join(e.replace(chr(91)+chr(116)+'ape2smpl'+chr(93)+' ','') for e in err[:2])}"))
            else:
                # The converter must emit the CANONICAL [root(3)|body(69)]
                # pose_aa (pose_aa_layout marker, 2026-08-26). The legacy
                # [body|pad] layout put the LEFT HIP in the root slot — the
                # source of a pico-vs-corpus SMPL heading mismatch.
                if str(z.get("pose_aa_layout", "")) != "smpl_canonical_root_first":
                    raise RuntimeError(
                        f"{key}: converter output lacks the canonical "
                        "pose_aa layout — update pico_tape_to_smpl_obs.py")
                pose_aa = np.asarray(z["pose_aa"], np.float32)
                delta = pose_aa.shape[0] - n_robot
                if abs(delta) > SMPL_LEN_TOLERANCE_FRAMES:
                    no_sidecar.append((key, f"pose_aa len delta {delta}"))
                    pose_aa = None

            if pose_aa is not None:
                # smpl_joints / transl in the corpus sidecar convention:
                # z-up world FK joints and y-up SMPL translation (y=height),
                # both from the tape conversion. (Earlier bundles wrote the
                # retarget corpus' smpl_joints — all zeros — and the ROBOT's
                # z-up root_trans_offset, whose y is not a height.)
                sj = np.asarray(z["smpl_joints_world"], np.float32)
                transl = np.asarray(z["transl"], np.float32)
                side = {"pose_aa": pose_aa, "transl": transl, "smpl_joints": sj,
                        "fps": 50.0,
                        "original_pose_aa": pose_aa, "original_fps": 50.0}
                if not a.dry_run:
                    joblib.dump(side, os.path.join(a.sidecar_dir, key + ".pkl"))
                sidecars += 1
            n_fix = 0 if a.no_despike else despike_clip(c, a.jump_thr)
            added[key] = c
            tag = f"smpl={pose_aa.shape[0]} delta={delta:+d}" if pose_aa is not None else "NO SIDECAR"
            if n_fix:
                tag += f" despiked={n_fix}f"
            print(f"  [{i:3d}/{len(files)}] {key}  robot={n_robot} {tag}")

    # APPEND-ONLY: base keys first in original order, then new keys.
    merged = {}
    for k in base_keys:
        merged[k] = base[k]
    for k, v in added.items():
        merged[k] = v

    assert list(merged.keys())[:len(base_keys)] == base_keys, \
        "base key order changed -- this would reset the adaptive sampler"

    print(f"\nbase {len(base_keys)} + new {len(added)} = {len(merged)} clips")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for k, why in skipped:
            print(f"  {k}: {why}")
    if no_sidecar:
        print(f"\nclips INCLUDED but with no SMPL sidecar ({len(no_sidecar)}) -- "
              "they train in g1 mode and are simply absent from smpl episodes:")
        for k, why in no_sidecar:
            print(f"  {k}: {why}")
    print(f"\nsidecars written: {sidecars} -> {a.sidecar_dir}")

    if not a.dry_run:
        joblib.dump(merged, a.out_pkl)
        print(f"wrote {a.out_pkl} ({os.path.getsize(a.out_pkl)/1e6:.1f} MB)")
    else:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    sys.exit(main())
