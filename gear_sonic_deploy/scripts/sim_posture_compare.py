"""Side-by-side table of posture bench cases (metrics.json from sim_posture_metrics.py), with the robot's 2026-10-24
SuperMimic 320k numbers as the reference row where one exists.

    python sim_posture_compare.py OUT_ROOT LABEL [LABEL ...]
"""

from __future__ import annotations

import json
import pathlib
import sys

ROBOT = {  # 2026-10-24 robot session, SuperMimic 320k @ vendor plant (findings doc s9)
    "waist_pitch_q_minus_ref_deg": -15.2, "hip_roll_spread_deg": -9.0, "waist_pitch_target_deg": 19.5,
    "waist_pitch_tau_cmd_nm": 7.0, "ankle_roll_target_offsets_deg": "-40.5 / +21.4",
}
STAND_ROWS = [("waist pitch q-ref (deg)", "waist_pitch_q_minus_ref_deg"), ("waist roll q-ref (deg)", "waist_roll_q_minus_ref_deg"),
              ("waist pitch target (deg)", "waist_pitch_target_deg"), ("waist pitch torque (N*m)", "waist_pitch_tau_cmd_nm"),
              ("hip-roll spread L-R (deg)", "hip_roll_spread_deg"), ("ankle-roll target offsets L / R (deg)", "ankle_roll_target_offsets_deg"),
              ("mean |posture err| legs+waist (deg)", "mean_abs_posture_err_deg"), ("base pitch / roll (deg)", None)]  # fmt: skip
K_JOINTS = ["waist_pitch", "waist_roll", "left_hip_roll", "right_hip_roll", "left_ankle_roll", "right_ankle_roll",
            "left_ankle_pitch", "right_ankle_pitch", "left_knee", "right_knee"]  # fmt: skip


def get(root: pathlib.Path, label: str, probe: str):
    f = root / label / probe / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, list):
        return " / ".join(f"{x:+.1f}" for x in v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:+.1f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def table(header, rows):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"  # noqa: E731
    print(line(header)); print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print(line(r))
    print()


def main():
    root, labels = pathlib.Path(sys.argv[1]), sys.argv[2:]
    M = {lb: {p: get(root, lb, p) for p in ("stand", "nudge", "tilt_pitch", "tilt_roll", "push")} for lb in labels}
    print("### Standing idle (planner streams the stand anchor)\n")
    rows = []
    for name, key in STAND_ROWS:
        r = [name]
        for lb in labels:
            s = (M[lb]["stand"] or {}).get("stand", {})
            r.append(fmt([s.get("base_pitch_deg"), s.get("base_roll_deg")] if key is None else s.get(key)) if s else "-")
        r.append(fmt(ROBOT.get(key)) if key else "-")
        rows.append(r)
    rows.append(["fell / trips"] + [fmt((M[lb]["stand"] or {}).get("fell")) + f" / {len((M[lb]['stand'] or {}).get('trips', []))}" for lb in labels] + ["-"])
    table(["metric"] + labels + ["robot 320k"], rows)

    print("### Slow push at the neck (4 s ramp, 2 s hold, 1 s release; +x +y -x -y per level)\n")
    levels = sorted({lv for lb in labels for lv in (((M[lb]["push"] or {}).get("push") or {}).get("by_level") or {})}, key=float)
    rows = []
    for name, key in [("peak pelvis yield (cm)", "peak_disp_cm"), ("peak base tilt (deg)", "peak_tilt_deg"),
                      ("posture err vs stand, force just peaked (deg)", "err_ref_at_peak_deg"),
                      ("posture err vs stand, end of 2 s hold (deg)", "err_ref_end_hold_deg"),
                      ("posture err vs stand, 3 s after release (deg)", "err_ref_3s_deg"),
                      ("residual vs pre-push, 3 s after release (deg)", "residual_3s_deg"),
                      ("max foot movement (cm)", "foot_moved_cm"), ("pushes with a step (>3 cm)", "steps"),
                      ("min lateral foot separation (cm)", "min_foot_lat_sep_cm"), ("falls", "falls")]:  # fmt: skip
        for lv in levels:
            r = [f"{name} @ {lv} N"]
            for lb in labels:
                b = ((((M[lb]["push"] or {}).get("push") or {}).get("by_level") or {}).get(lv))
                r.append("-" if b is None else (f"{b[key]}/{b['n']}" if key in ("steps", "falls") else fmt(b[key])))
            rows.append(r)
    rows.append(["FELL ON push"] + [fmt(((M[lb]["push"] or {}).get("push") or {}).get("fell_on")) for lb in labels])
    rows.append(["stand lateral foot separation (cm)"] + [fmt(((M[lb]["push"] or {}).get("push") or {}).get("stand_foot_lat_sep_cm")) for lb in labels])
    table(["metric"] + labels, rows)

    print("### Chest shoves (80 N, 0.2 s; +x +y -x -y)\n")
    rows = []
    for name, key in [("recovered", "n_recovered"), ("mean peak base tilt (deg)", "mean_peak_tilt_deg"),
                      ("mean residual posture err 3 s after (deg)", "mean_residual_3s_deg")]:  # fmt: skip
        r = [name]
        for lb in labels:
            n = (M[lb]["nudge"] or {}).get("nudge")
            r.append("-" if not n else (f"{n['n_recovered']}/{n['n']}" if key == "n_recovered" else fmt(n[key])))
        rows.append(r)
    rows.append(["fell"] + [fmt((M[lb]["nudge"] or {}).get("fell")) for lb in labels])
    table(["metric"] + labels, rows)

    for probe in ("tilt_pitch", "tilt_roll"):
        print(f"### Slow torso push, {probe.split('_')[1]} (25 N*m): effective stiffness K_eff (N*m/rad) | posture gain\n")
        rows = []
        for j in K_JOINTS:
            r = [j]
            for lb in labels:
                k = ((M[lb][probe] or {}).get("tilt") or {}).get("K", {}).get(j)
                r.append("-" if not k else f"{k['K_eff']:7.1f} | {k['posture_gain']:+.2f}")
            rows.append(r)
        for name, key in [("snap after release, waist (deg/s)", "mean_snap_waist_dps"), ("residual err 3 s after release (deg)", "mean_residual_3s_deg")]:
            rows.append([name] + [fmt(((M[lb][probe] or {}).get("tilt") or {}).get(key)) for lb in labels])
        rows.append(["fell"] + [fmt((M[lb][probe] or {}).get("fell")) for lb in labels])
        table(["joint"] + labels, rows)


if __name__ == "__main__":
    main()
