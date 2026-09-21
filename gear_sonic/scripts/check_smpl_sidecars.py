#!/usr/bin/env python
"""Gate for SMPL sidecar files before they are staged for training or SMPL-mode eval.

Why: a batch of freshly built sidecars once carried ROOT-LOCAL joints where the motion lib
expects the corpus convention (world joints, y-up SMPL pelvis offset). Field names and array
shapes were identical, so a name/shape check passed; the frame was wrong and a third of the
Pico fine-tune episodes trained on it; the same bug had hit an earlier corpus pipeline.

Checks per file (all must hold):
  * keys pose_aa (T,72) transl (T,3) smpl_joints (T,24,3) fps
  * corpus frame: pelvis joint at frame 0 sits at the SMPL rest offset ~(0, -0.35, 0)
    -- root-local files put it at the origin; that is the root-local failure signature
  * up axis: |transl_y| dominates |transl_z| (y-up) and the head sits above the pelvis
  * clip length matches the motion pkl entry with the same key (when --motion is given)
Reference values are read from a known-good corpus sidecar when --reference is given.

Usage:
  check_smpl_sidecars.py DIR [--motion clips.pkl] [--reference corpus/smpl_filtered/Idle_Left_001__A017.pkl]
Exit 1 on any FAIL. Print one line per failing file, a summary line always.
"""
import argparse, glob, os, sys
import joblib, numpy as np


def check_file(path, motion, ref):
    d = joblib.load(path)
    for k, shape in (("pose_aa", (None, 72)), ("transl", (None, 3)), ("smpl_joints", (None, 24, 3))):
        if k not in d:
            return f"missing field {k}"
        a = np.asarray(d[k])
        if a.ndim != len(shape) or any(s is not None and a.shape[i] != s for i, s in enumerate(shape)):
            return f"{k} shape {a.shape}"
    if "fps" not in d:
        return "missing fps"
    pa, tr, sj = np.asarray(d["pose_aa"]), np.asarray(d["transl"]), np.asarray(d["smpl_joints"])
    if not (len(pa) == len(tr) == len(sj)):
        return f"length mismatch pose {len(pa)} transl {len(tr)} joints {len(sj)}"
    pelvis0 = sj[0, 0]
    if ref is not None and np.linalg.norm(pelvis0 - ref["pelvis0"]) > 0.08:
        return f"pelvis[0] {pelvis0.round(3)} != corpus {ref['pelvis0'].round(3)} (root-local joints?)"
    if ref is None and np.linalg.norm(pelvis0) < 0.05:
        return f"pelvis[0] at the origin {pelvis0.round(3)} -> ROOT-LOCAL joints (viewer frame), not the corpus frame"
    # transl: the y (height) channel must not be flat at zero -- the clip splitter's re-anchoring zeroes it
    # when it is fed a y-up file; a walking clip's z drift can exceed |y|, so no z-vs-y ratio test.
    if np.abs(tr[:, 1]).max() < 0.05:
        return f"transl y is flat at zero (max |y| {np.abs(tr[:, 1]).max():.3f}) -> height lost (splitter re-anchoring?)"
    head_up = (sj[:, 15] - sj[:, 0])
    up_axis = int(np.argmax(np.abs(head_up).mean(0)))
    if ref is not None and up_axis != ref["up_axis"]:
        return f"head-pelvis up axis {up_axis} != corpus {ref['up_axis']}"
    if motion is not None:
        key = os.path.basename(path)[:-4]
        if key in motion and len(motion[key]["dof"]) != len(pa):
            return f"length {len(pa)} != motion clip {len(motion[key]['dof'])}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--motion", default=None, help="motion-lib pkl; sidecar length must match the clip with the same key")
    ap.add_argument("--reference", default=None, help="a known-good corpus sidecar (e.g. smpl_filtered/Idle_Left_001__A017.pkl)")
    ap.add_argument("--max-print", type=int, default=20)
    a = ap.parse_args()
    ref = None
    if a.reference:
        r = joblib.load(a.reference); sj = np.asarray(r["smpl_joints"])
        ref = {"pelvis0": sj[0, 0], "up_axis": int(np.argmax(np.abs(sj[:, 15] - sj[:, 0]).mean(0)))}
    motion = joblib.load(a.motion) if a.motion else None
    files = sorted(glob.glob(os.path.join(a.dir, "*.pkl")))
    fails = []
    for f in files:
        try:
            why = check_file(f, motion, ref)
        except Exception as e:  # noqa: BLE001
            why = f"unreadable: {e}"
        if why:
            fails.append((os.path.basename(f), why))
    for name, why in fails[: a.max_print]:
        print(f"FAIL {name}: {why}")
    print(f"[check_smpl_sidecars] {a.dir}: {len(files) - len(fails)} PASS / {len(fails)} FAIL of {len(files)}"
          + (" (reference frame from %s)" % os.path.basename(a.reference) if a.reference else ""))
    return 1 if fails or not files else 0


if __name__ == "__main__":
    sys.exit(main())
