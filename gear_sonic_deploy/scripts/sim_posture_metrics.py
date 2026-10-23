"""Posture bench metrics (2026-10-24) for sessions written by sim_posture_bench.sh.

Same definitions as the robot nudge analysis of 2026-10-24 so sim and robot numbers compare directly:
  reference q_ref = the kplanner idle stand anchor (what the planner streams while idle);
  commanded PD torque tau = kp*(target - q) - kd*v with the SESSION's plant gains (meta.txt -> x2_ultra_<plant>.yaml);
  effective stiffness K_eff = -dtau/dd over quasi-static samples (|v| < 0.3 rad/s), d = q - q_ref;
  posture gain = d(target - q_ref)/dd (-1 = target fully opposes the push, 0 = ignores it, > 0 = follows it).

Per probe:
  stand       posture in CONTROL (5 s after the ramp ends -> end): waist / hip-roll spread / ankle roll vs reference,
              ankle-roll target offsets (the robot's feet-fighting signature), base tilt, joint jitter.
  nudge       per shove: peak base tilt, peak leg+waist deviation, residual deviation 3 s later (vs the 1 s before the
              shove), recovered = residual < 1.5 deg and upright.
  tilt_*      K_eff / posture gain of waist, hips, knees, ankles while the torque is applied; after each release the
              peak waist speed (snap) and the residual deviation 3 s later.

    python sim_posture_metrics.py OUT_ROOT/LABEL/PROBE [...]       # prints + writes metrics.json next to each
"""

from __future__ import annotations

import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gear_sonic/envs/manager_env/robots"))
from plant_config import load_plant  # noqa: E402

ANCHOR = REPO / "gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl"
POSTURE_JOINTS = ["waist_pitch", "waist_roll", "waist_yaw", "left_hip_pitch", "right_hip_pitch", "left_hip_roll",
                  "right_hip_roll", "left_knee", "right_knee", "left_ankle_pitch", "right_ankle_pitch", "left_ankle_roll",
                  "right_ankle_roll"]  # fmt: skip
FALL_TILT = np.radians(45.0)


def load(sess: pathlib.Path) -> dict:
    meta = dict(line.split(" ", 1) for line in (sess / "meta.txt").read_text().splitlines() if " " in line)
    L = sess / "session/logs"
    q, v, tg = (pd.read_csv(L / f) for f in ("joint_pos.csv", "joint_vel.csv", "target_pos.csv"))
    imu, tick = pd.read_csv(L / "imu.csv"), pd.read_csv(L / "tick.csv")
    n = min(len(q), len(v), len(tg), len(imu), len(tick))
    names = [c[2:].replace("_joint", "") for c in q.columns[1:32]]
    plant = load_plant(meta["plant"].strip())
    kp = np.array([plant.kp_for(nm + "_joint") for nm in names])
    kd = np.array([plant.kd_for(nm + "_joint") for nm in names])
    eff = np.array([plant.effort_for(nm + "_joint") for nm in names])
    ref = np.asarray(joblib.load(ANCHOR)["idle_anchor_v3_graft"]["dof"][0], float)
    w, x, y, z = (imu[c].values[:n] for c in ("qw", "qx", "qy", "qz"))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    probe = pd.read_csv(L / "probe.csv") if (L / "probe.csv").exists() else None
    tripped = (sess / "trips.txt").exists() and bool((sess / "trips.txt").read_text().strip())
    return dict(meta=meta, names=names, t=q["t"].values[:n], Q=q.iloc[:n, 1:32].values, V=v.iloc[:n, 1:32].values,
                TG=tg.iloc[:n, 1:32].values, ramp=tick["ramp_alpha"].values[:n], roll=roll, pitch=pitch, yaw=yaw, kp=kp, kd=kd,
                eff=eff, ref=ref, probe=probe, tripped=tripped)  # fmt: skip


def control_mask(d: dict, settle: float = 5.0) -> np.ndarray:
    full = np.nonzero(d["ramp"] >= 0.999)[0]
    if len(full) == 0:
        return np.zeros(len(d["t"]), bool)
    return d["t"] >= d["t"][full[0]] + settle


def tau_cmd(d: dict) -> np.ndarray:
    return np.clip(d["kp"] * (d["TG"] - d["Q"]) - d["kd"] * d["V"], -d["eff"], d["eff"])


def deg(x):
    return float(np.degrees(x))


def stand_metrics(d: dict, m: np.ndarray) -> dict:
    ix = {j: d["names"].index(j) for j in POSTURE_JOINTS}
    dq = d["Q"][m] - d["ref"]
    off = d["TG"][m] - d["ref"]
    out = {f"{j}_q_minus_ref_deg": deg(dq[:, i].mean()) for j, i in ix.items()}
    out["hip_roll_spread_deg"] = deg((dq[:, ix["left_hip_roll"]] - dq[:, ix["right_hip_roll"]]).mean())
    out["ankle_roll_target_offsets_deg"] = [deg(off[:, ix["left_ankle_roll"]].mean()), deg(off[:, ix["right_ankle_roll"]].mean())]
    out["waist_pitch_target_deg"] = deg(d["TG"][m][:, ix["waist_pitch"]].mean())
    out["waist_pitch_tau_cmd_nm"] = float(tau_cmd(d)[m][:, ix["waist_pitch"]].mean())
    out["base_pitch_deg"], out["base_roll_deg"] = deg(d["pitch"][m].mean()), deg(d["roll"][m].mean())
    legs = [ix[j] for j in POSTURE_JOINTS if j != "waist_yaw"]
    out["leg_waist_jitter_dps"] = deg(d["V"][m][:, legs].std(0).mean())
    out["mean_abs_posture_err_deg"] = deg(np.abs(dq[:, legs]).mean())
    return out


def events(d: dict, mag: np.ndarray, thresh: float):
    """Rising and falling edges of |external wrench| > thresh, in deploy time."""
    p = d["probe"]
    on = mag > thresh
    rise = p["t"].values[1:][on[1:] & ~on[:-1]]
    fall = p["t"].values[1:][~on[1:] & on[:-1]]
    return rise, fall


def window_err(d: dict, legs, t0: float, t1: float, base: np.ndarray) -> np.ndarray:
    s = (d["t"] >= t0) & (d["t"] < t1)
    return np.abs(d["Q"][s][:, legs] - base).mean(1) if s.any() else np.array([np.nan])


def nudge_metrics(d: dict) -> dict:
    p = d["probe"]
    rise, fall = events(d, np.hypot(p["fx"].values, p["fy"].values), 1.0)
    legs = [d["names"].index(j) for j in POSTURE_JOINTS if j != "waist_yaw"]
    rows = []
    for k, t0 in enumerate(rise):
        pre = (d["t"] >= t0 - 1.0) & (d["t"] < t0)
        if not pre.any():
            continue
        base = d["Q"][pre][:, legs].mean(0)
        post = (d["t"] >= t0) & (d["t"] < t0 + 2.0)
        tilt = np.hypot(d["roll"][post] - d["roll"][pre].mean(), d["pitch"][post] - d["pitch"][pre].mean())
        pk = window_err(d, legs, t0, t0 + 2.0, base).max()
        res = window_err(d, legs, t0 + 3.0, t0 + 3.5, base).mean()
        upright = np.hypot(d["roll"], d["pitch"])[(d["t"] >= t0) & (d["t"] < t0 + 4.0)].max() < FALL_TILT
        fi = np.argmin(np.abs(p["t"].values - t0 - 0.05))
        direction = "+x" if p["fx"].values[fi] > 1 else "-x" if p["fx"].values[fi] < -1 else "+y" if p["fy"].values[fi] > 1 else "-y"
        rows.append(dict(t=float(t0 - d["t"][0]), dir=direction, peak_tilt_deg=deg(tilt.max()), peak_dev_deg=deg(pk),
                         residual_3s_deg=deg(res), recovered=bool(res < np.radians(1.5) and upright)))  # fmt: skip
    return {"pushes": rows, "n": len(rows), "n_recovered": sum(r["recovered"] for r in rows),
            "mean_residual_3s_deg": float(np.nanmean([r["residual_3s_deg"] for r in rows])) if rows else None,
            "mean_peak_tilt_deg": float(np.mean([r["peak_tilt_deg"] for r in rows])) if rows else None}  # fmt: skip


def tilt_metrics(d: dict) -> dict:
    p = d["probe"]
    tq = np.linalg.norm(p[["tx", "ty", "tz"]].values, axis=1)
    rise, fall = events(d, tq, 0.5)
    on = np.interp(d["t"], p["t"].values, (tq > 0.5).astype(float)) > 0.5
    tau = tau_cmd(d)
    K = {}
    for j in POSTURE_JOINTS:
        i = d["names"].index(j)
        s = on & (np.abs(d["V"][:, i]) < 0.3)
        dd = d["Q"][s, i] - d["ref"][i]
        if s.sum() < 50 or dd.std() < np.radians(0.3):
            K[j] = None
            continue
        K[j] = dict(K_eff=float(-np.polyfit(dd, tau[s, i], 1)[0]), kp=float(d["kp"][i]),
                    posture_gain=float(np.polyfit(dd, d["TG"][s, i] - d["ref"][i], 1)[0]),
                    disp_p5_p95_deg=[deg(np.percentile(dd, 5)), deg(np.percentile(dd, 95))])  # fmt: skip
    legs = [d["names"].index(j) for j in POSTURE_JOINTS if j != "waist_yaw"]
    iw = d["names"].index("waist_pitch" if "pitch" in d["meta"]["probe"] else "waist_roll")
    rel = []
    for t1, t0 in zip(fall, rise):
        pre = (d["t"] >= t0 - 1.0) & (d["t"] < t0)
        if not pre.any():
            continue
        base = d["Q"][pre][:, legs].mean(0)
        snap = (d["t"] >= t1) & (d["t"] < t1 + 1.0)
        rel.append(dict(snap_waist_dps=deg(np.abs(d["V"][snap, iw]).max()) if snap.any() else None,
                        residual_3s_deg=deg(window_err(d, legs, t1 + 3.0, t1 + 3.5, base).mean())))  # fmt: skip
    return {"K": K, "releases": rel,
            "mean_snap_waist_dps": float(np.nanmean([r["snap_waist_dps"] or np.nan for r in rel])) if rel else None,
            "mean_residual_3s_deg": float(np.nanmean([r["residual_3s_deg"] for r in rel])) if rel else None}  # fmt: skip


def push_metrics(d: dict) -> dict:
    """Slow neck push: per push, how far the body yields and whether the policy pulls back toward the stand while the force
    is held, then the error after release. Pelvis / feet positions come from the probe log (MuJoCo world)."""
    p = d["probe"]
    F = np.hypot(p["fx"].values, p["fy"].values)
    tp = p["t"].values
    rise, fall = events(d, F, 1.0)
    legs = [d["names"].index(j) for j in POSTURE_JOINTS if j != "waist_yaw"]
    yaw = np.interp(tp, d["t"], d["yaw"])
    lat = -(p["lfx"].values - p["rfx"].values) * np.sin(yaw) + (p["lfy"].values - p["rfy"].values) * np.cos(yaw)
    tilt_all = np.hypot(d["roll"], d["pitch"])
    fallen = np.nonzero(tilt_all > FALL_TILT)[0]
    t_fall = d["t"][fallen[0]] if len(fallen) else np.inf
    if d.get("tripped"):  # the deploy stops logging at a trip (SAFE_HOLD): its last tick is the fall
        t_fall = min(t_fall, d["t"][-1])
    rows = []
    for t0, t1 in zip(rise, fall):
        if t0 > t_fall:
            break  # down: later pushes are meaningless
        k0 = np.searchsorted(tp, t0)
        level = float(F[k0:np.searchsorted(tp, t1)].max())
        fx, fy = p["fx"].values[k0 + 25], p["fy"].values[k0 + 25]
        u = np.array([fx, fy]) / max(np.hypot(fx, fy), 1e-6)
        direction = ("+x" if fx > 0 else "-x") if abs(fx) > abs(fy) else ("+y" if fy > 0 else "-y")
        pre_p = (tp >= t0 - 1.0) & (tp < t0)
        win_p = (tp >= t0) & (tp < t1 + 1.0)
        P0 = np.array([p["px"].values[pre_p].mean(), p["py"].values[pre_p].mean()])
        disp = ((np.c_[p["px"].values[win_p], p["py"].values[win_p]] - P0) @ u)
        feet0 = np.c_[p["lfx"], p["lfy"], p["rfx"], p["rfy"]].astype(float)[pre_p].mean(0)
        feet = np.c_[p["lfx"], p["lfy"], p["rfx"], p["rfy"]].astype(float)[win_p | ((tp >= t1) & (tp < t1 + 4.0))]
        step = float(np.max(np.hypot(feet[:, [0, 2]] - feet0[[0, 2]], feet[:, [1, 3]] - feet0[[1, 3]])))
        pre = (d["t"] >= t0 - 1.0) & (d["t"] < t0)
        base = d["Q"][pre][:, legs].mean(0)
        ref = d["ref"][legs]
        hold_end = t1 - 1.0  # release starts after the hold; t1 = force back to ~0
        t_peak = hold_end - 2.0  # end of ramp (2 s hold)
        err_ref = lambda a, b: deg(np.abs(d["Q"][(d["t"] >= a) & (d["t"] < b)][:, legs] - ref).mean())  # noqa: E731
        tilt = np.hypot(d["roll"], d["pitch"])[(d["t"] >= t0) & (d["t"] < t1 + 1.0)]
        rows.append(dict(level=round(level), dir=direction, peak_disp_cm=float(100 * disp.max()),
                         peak_tilt_deg=deg(tilt.max()) if len(tilt) else None,
                         err_ref_at_peak_deg=err_ref(t_peak, t_peak + 0.3), err_ref_end_hold_deg=err_ref(hold_end - 0.3, hold_end),
                         residual_3s_deg=deg(window_err(d, legs, t1 + 3.0, t1 + 3.5, base).mean()),
                         err_ref_3s_deg=err_ref(t1 + 3.0, t1 + 3.5), foot_moved_cm=100 * step,
                         min_foot_lat_sep_cm=float(100 * lat[win_p].min()),
                         fell=bool(t0 <= t_fall < t1 + 4.0)))  # fmt: skip
    by = {}
    for r in rows:
        by.setdefault(r["level"], []).append(r)
    summ = {str(lv): {k: float(np.nanmean([np.nan if r[k] is None else r[k] for r in rs])) for k in ("peak_disp_cm", "peak_tilt_deg", "err_ref_at_peak_deg",
                                                                      "err_ref_end_hold_deg", "residual_3s_deg", "err_ref_3s_deg", "foot_moved_cm")}
            | {"steps": sum(r["foot_moved_cm"] > 3.0 for r in rs), "falls": sum(r["fell"] for r in rs), "n": len(rs),
               "min_foot_lat_sep_cm": float(min(r["min_foot_lat_sep_cm"] for r in rs))}
            for lv, rs in sorted(by.items())}  # fmt: skip
    first_fall = next((f"{r['level']} N {r['dir']}" for r in rows if r["fell"]), None)
    return {"pushes": rows, "by_level": summ, "fell_on": first_fall, "stand_foot_lat_sep_cm": float(100 * np.median(lat[tp < (rise[0] if len(rise) else tp[-1])]))}


def analyse(sess: pathlib.Path) -> dict:
    d = load(sess)
    m = control_mask(d)
    tilt_all = np.hypot(d["roll"], d["pitch"])
    out = {"label": d["meta"]["label"].strip(), "probe": d["meta"]["probe"].strip(), "plant": d["meta"]["plant"].strip(),
           "control_s": float(d["t"][m][-1] - d["t"][m][0]) if m.any() else 0.0,
           "fell": bool((tilt_all[m] > FALL_TILT).any()) if m.any() else None,
           "trips": (sess / "trips.txt").read_text().strip().splitlines() if (sess / "trips.txt").exists() else []}  # fmt: skip
    if not m.any():
        return out
    probe = out["probe"]
    if probe == "stand":
        out["stand"] = stand_metrics(d, m)
    else:
        pre = m & (d["t"] < (d["probe"]["t"].values[np.argmax(np.abs(d["probe"][["fx", "fy", "tx", "ty"]].values).max(1) > 0.5)]
                             if d["probe"] is not None else d["t"][-1]))  # fmt: skip
        if pre.sum() > 100:
            out["stand_before_probe"] = stand_metrics(d, pre)
        if probe == "push":
            out["push"] = push_metrics(d)
        else:
            out["nudge" if probe == "nudge" else "tilt"] = nudge_metrics(d) if probe == "nudge" else tilt_metrics(d)
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        sess = pathlib.Path(a)
        res = analyse(sess)
        (sess / "metrics.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1)[:4000])
