"""Validate a retargeted Pico teleop corpus before it joins the fine-tune set.

Checks the data itself (authoritative) and reconciles it against the retarget
gate logs (context). Emits a keep-list and a reject-list.

What it checks, and why each one is here:
  fps            50 Hz, matching the SONIC corpus. A mislabelled fps silently
                 destroys SMPL/robot correspondence -- exactly the failure that
                 killed the v11 run at iteration 2249.
  finite         no NaN/Inf anywhere. Non-finite clips have poisoned this
                 corpus before (QUARANTINE_nonfinite).
  smpl_len       len(smpl_joints) vs len(dof), which motion_lib reconciles with
                 a HARD 2-frame tolerance. Beyond that it raises at startup and
                 takes the whole distributed run down with it.
  wrist_sat      fraction of frames each wrist joint sits within 1% of its X2
                 limit. This corpus is teleop, and pinned wrists are the known
                 pathology -- a reference that is itself saturated teaches the
                 policy to saturate.
  duration       very short clips are usually capture fragments, not motions.
  gates          final (post-despike) verdicts from the retarget log.

Usage:
  python gear_sonic/data_process/validate_pico_corpus.py <corpus_dir> [--json out.json]
"""

import argparse
import glob
import json
import os
import re
import sys

import joblib
import numpy as np

MIN_DURATION_S = 1.0
SMPL_LEN_TOLERANCE_FRAMES = 2  # must match motion_lib_base.SMPL_LEN_TOLERANCE_FRAMES
SAT_EPS_FRAC = 0.01            # within 1% of range counts as "at the stop"
SAT_WARN_FRAC = 0.25           # flag a joint pinned this often


def x2_joint_limits():
    """(names, lo, hi) in MJCF joint order. Kinematic-tree order, NOT actuator order."""
    import mujoco
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eval_x2_mujoco import MJCF_PATH
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    names, lo, hi = [], [], []
    for j in range(m.njnt):
        if m.jnt_type[j] in (0, 1):  # skip free/ball root
            continue
        names.append(mujoco.mj_name2id and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        lo.append(m.jnt_range[j][0]); hi.append(m.jnt_range[j][1])
    return names, np.array(lo), np.array(hi)


def parse_gates(log_path):
    """Return final-verdict failures. Pre-despike FAILs that a later block
    clears are NOT failures -- the despiker exists precisely to fix them."""
    if not os.path.exists(log_path):
        return {"gate_fail": ["<no log>"], "despiked": 0}
    txt = open(log_path, errors="replace").read()
    despiked = 0
    m = re.search(r"despiked (\d+) isolated frame", txt)
    if m:
        despiked = int(m.group(1))
    # sanity gates: keep only the LAST block, which is post-despike if one ran
    blocks = txt.split("sanity gates (value / threshold):")
    final_sanity = blocks[-1] if len(blocks) > 1 else ""
    sanity_fail = re.findall(r"FAIL\s+(\w+)", final_sanity.split("[gmr2lib] analytic")[0])
    # FK/elbow gates appear once and are final
    fk_section = txt.split("elbow ANGLE gate")[-1] if "elbow ANGLE gate" in txt else ""
    fk_fail = re.findall(r"FAIL\s+(left|right)\s+(\w+)", fk_section)
    return {"gate_fail": sanity_fail + [f"{a}_{b}" for a, b in fk_fail], "despiked": despiked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir")
    ap.add_argument("--json", default=None)
    ap.add_argument("--sat-warn", type=float, default=SAT_WARN_FRAC)
    a = ap.parse_args()

    names, lo, hi = x2_joint_limits()
    rng = hi - lo
    wrist_idx = [i for i, n in enumerate(names) if "wrist" in n]
    print(f"X2 joints: {len(names)}; wrist joints: {[names[i] for i in wrist_idx]}\n")

    rows = []
    for f in sorted(glob.glob(os.path.join(a.corpus_dir, "*.pkl"))):
        d = joblib.load(f)
        for key, c in d.items():
            dof = np.asarray(c["dof"]); T = dof.shape[0]
            fps = int(c.get("fps", -1))
            sm = np.asarray(c["smpl_joints"]) if "smpl_joints" in c else None
            problems = []

            if fps != 50:
                problems.append(f"fps={fps}")
            finite = all(np.isfinite(np.asarray(v)).all()
                         for v in c.values() if isinstance(v, np.ndarray))
            if not finite:
                problems.append("nonfinite")
            if dof.shape[1] != len(names):
                problems.append(f"dof_width={dof.shape[1]}!={len(names)}")
            smpl_delta = (sm.shape[0] - T) if sm is not None else None
            if sm is None:
                problems.append("no_smpl_joints")
            elif abs(smpl_delta) > SMPL_LEN_TOLERANCE_FRAMES:
                problems.append(f"smpl_len_delta={smpl_delta}")
            dur = T / fps if fps > 0 else 0.0
            if dur < MIN_DURATION_S:
                problems.append(f"short={dur:.2f}s")

            # wrist saturation against the real X2 limits
            sat = {}
            if dof.shape[1] == len(names):
                for i in wrist_idx:
                    tol = SAT_EPS_FRAC * rng[i]
                    frac = float(np.mean((dof[:, i] <= lo[i] + tol) |
                                         (dof[:, i] >= hi[i] - tol)))
                    sat[names[i]] = frac
                worst = max(sat, key=sat.get)
                if sat[worst] >= a.sat_warn:
                    problems.append(f"wrist_pinned:{worst}={sat[worst]*100:.0f}%")

            g = parse_gates(os.path.join(
                a.corpus_dir,
                os.path.basename(f).replace("pico_", "session_").replace(".pkl", "_gates.log")))

            rows.append({"key": key, "file": os.path.basename(f), "frames": T,
                         "fps": fps, "dur_s": round(dur, 2),
                         "smpl_delta": smpl_delta, "wrist_sat": sat,
                         "gate_fail": g["gate_fail"], "despiked": g["despiked"],
                         "problems": problems})

    keep = [r for r in rows if not r["problems"]]
    rej = [r for r in rows if r["problems"]]
    print(f"{'key':44s} {'dur':>6s} {'dsm':>4s} {'worst wrist sat':>22s}  problems")
    for r in sorted(rows, key=lambda r: (bool(r["problems"]), -r["dur_s"])):
        w = max(r["wrist_sat"], key=r["wrist_sat"].get) if r["wrist_sat"] else "-"
        ws = f"{w.replace('_joint','')}={r['wrist_sat'].get(w,0)*100:.0f}%" if r["wrist_sat"] else "-"
        print(f"{r['key']:44s} {r['dur_s']:6.1f} {str(r['smpl_delta']):>4s} {ws:>22s}  "
              f"{','.join(r['problems']) if r['problems'] else 'OK'}")

    tot = sum(r["dur_s"] for r in rows)
    ktot = sum(r["dur_s"] for r in keep)
    print(f"\n{len(rows)} clips, {tot/60:.1f} min total")
    print(f"KEEP   {len(keep):3d} clips, {ktot/60:.1f} min")
    print(f"REJECT {len(rej):3d} clips, {(tot-ktot)/60:.1f} min")
    if rej:
        from collections import Counter
        c = Counter(p.split(":")[0].split("=")[0] for r in rej for p in r["problems"])
        print("  reasons: " + ", ".join(f"{k}x{v}" for k, v in c.most_common()))
    gf = [r for r in rows if r["gate_fail"]]
    print(f"\nretarget gate failures (context, not auto-reject): {len(gf)} clips")
    from collections import Counter
    print("  " + ", ".join(f"{k}x{v}" for k, v in
                           Counter(g for r in gf for g in r["gate_fail"]).most_common()))

    if a.json:
        json.dump({"keep": [r["key"] for r in keep], "reject": rej, "all": rows},
                  open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
