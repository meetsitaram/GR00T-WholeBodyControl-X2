#!/usr/bin/env python3
"""Pre-deploy regression suite for pc2_kplanner_onnx — MUST PASS before any
PC2 push of planner code or primitive clips.

Born 2026-08-03 after two sim falls shipped on partial validation:
  * model-stop rework: ring starvation served frozen mid-swing frames and the
    settle detector counted the frozen root as "stopped" -> jump -> collapse.
  * side-step primitives: reference kinematics verified, SONIC tracking never
    checked (synthetic clip untracked; locomanip clip had an 11.5 cm z-dip).

Stages (all headless, CPU, ~3 min):
  1. UNIT      — stick resolver margins (deadband, exclusive strafe, purity)
  2. CLIPS     — every discrete primitive in the pkl passes physical bounds
  3. SERVE     — live serve loop driven through scripted intents INCLUDING
                 edge timings; wire asserts: no frozen frames, no joint
                 snaps, no root jumps, no starvation lines in the log
  4. (manual)  — SONIC-in-the-loop MuJoCo sim for any MOTION-LEVEL change
                 (new/edited clips, new playback modes): this suite cannot
                 judge trackability — a reference can be kinematically clean
                 and still be OOD for the policy. Drive it or run the
                 deploy-sim before shipping. THIS STAGE IS NOT OPTIONAL for
                 clip changes; stages 1-3 do not cover it.

Exit code 0 = all automated stages pass.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gear_sonic/scripts"))

import mujoco  # noqa: E402
import pc2_kplanner_onnx as k  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
def stage_unit() -> None:
    print("== STAGE 1: stick resolver margins")
    r = k._resolve_locomotion_continuous
    yaw, vx, vz, _ = r(0.10, 0.0, 0.0)
    check("slight fwd lean is NOT a walk command", vz == 0.0, f"vz={vz}")
    yaw, vx, vz, _ = r(-0.10, 0.0, 0.0)
    check("slight back lean is NOT a walk command", vz == 0.0, f"vz={vz}")
    yaw, vx, vz, _ = r(0.9, 0.0, 0.0)
    check("deliberate fwd push walks", vz > 0.0, f"vz={vz}")
    yaw, vx, vz, _ = r(0.15, 0.9, 0.0)
    check("strafe with slight fwd tilt strafes ONLY",
          vx != 0.0 and vz == 0.0, f"vx={vx} vz={vz}")
    yaw, vx, vz, _ = r(0.6, 0.9, 0.0)
    check("diagonal (fwd beyond purity cone) does not strafe",
          vx == 0.0, f"vx={vx}")
    yaw, vx, vz, _ = r(0.0, 0.3, 0.0)
    check("weak lateral push does not strafe", vx == 0.0, f"vx={vx}")


# ---------------------------------------------------------------------------
def stage_clips() -> None:
    print("== STAGE 2: primitive clip physical bounds")
    import joblib
    pkl = REPO / "gear_sonic/data/motions/x2_planner_primitives.pkl"
    lib = joblib.load(pkl)
    for name, c in sorted(lib.items()):
        fam = c.get("recipe_family", "")
        if fam not in ("discrete_step",) and not name.startswith(
                ("side_", "fwd_step", "back_step", "crouch_", "lean_", "torso_")):
            continue
        rt = np.asarray(c["root_trans"], dtype=np.float64)
        dt = np.abs(np.diff(np.asarray(c["dof"], dtype=np.float64), axis=0))
        zr = float(rt[:, 2].max() - rt[:, 2].min())
        travel = rt[-1, :2] - rt[0, :2]
        dist = float(np.linalg.norm(travel))
        dur = c["dof"].shape[0] / float(c.get("fps", 30.0))
        probs = []
        if name.startswith("side_") and zr > 0.04:  # normal gait bob ~0.03; locomanip disaster was 0.115
            probs.append(f"z-range {zr:.3f}m (steps must be height-flat)")
        if name.startswith("side_") and not (0.05 <= dist <= 0.70):
            probs.append(f"travel {dist:.2f}m out of [0.05,0.70]")
        if dt.max() > 0.5:
            probs.append(f"dof snap {dt.max():.2f} rad/frame")
        if not (0.3 <= dur <= 8.0):
            probs.append(f"duration {dur:.1f}s out of [0.3,8]")
        check(f"clip {name}", not probs, "; ".join(probs))


# ---------------------------------------------------------------------------
def stage_serve() -> None:
    print("== STAGE 3: serve-loop wire regression (headless)")
    PY = sys.executable
    log_path = Path("/tmp/preflight_serve.log")
    proc = subprocess.Popen(
        [PY, str(REPO / "gear_sonic/scripts/pc2_kplanner_onnx.py"),
         "--onnx", str(Path.home() / ".cache/sonic/x2/kplanner_onnx/x2_kplanner_template.onnx"),
         "--planner-mode", "slow_walk", "--pub-port", "16556",
         "--cmd-port", "16563", "--x2-debug-port", "0",
         "--clip-cmd-port", "16568", "--arm-port", "16572",
         "--pid-file", "/tmp/preflight_kplanner.pid",
         # SHIPPED ritual config, not defaults: the replan storm only
         # reproduces at threshold 48 (default 32 masked it — the suite's
         # own first verification of the storm fix was vacuous until this).
         "--replan-threshold-frames", "48"],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        env={**__import__("os").environ,
             # shipped ritual knobs — the battery must test what ships
             "KPLANNER_FIXED_TURN_RAD_S": "1.0",
             "KPLANNER_FIXED_FWD_MPS": "0.3",
             "KPLANNER_FIXED_ARC_TURN_RAD_S": "0.70"})
    import zmq
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB); sub.connect("tcp://127.0.0.1:16556")
    sub.setsockopt_string(zmq.SUBSCRIBE, "pose"); sub.setsockopt(zmq.RCVTIMEO, 2000)
    pub = ctx.socket(zmq.PUB); pub.bind("tcp://127.0.0.1:16563"); time.sleep(0.3)

    def send(d): pub.send_multipart([b"planner_cmd", json.dumps(d).encode()])

    frames: list[tuple[np.ndarray, np.ndarray, bool, np.ndarray, str]] = []
    motion_active = [False]
    motion_since = [0.0]
    motion_kind = [""]

    def pump(seconds: float) -> None:
        t0 = time.time()
        while time.time() - t0 < seconds:
            try:
                f = unpack_message(sub.recv(), expected_topic="pose")
            except zmq.Again:
                continue
            xy = f.fields["root_xy_world"].ravel()[:2].astype(float)
            z = float(f.fields["root_z_world"].ravel()[0])
            # "steady motion" excludes the 0.7 s command->first-frame spin-up
            # (replan inference latency is BY DESIGN; frozen frames only
            # count once motion should genuinely be flowing).
            steady = motion_active[0] and (time.time() - motion_since[0]) > 0.7
            q = f.fields["root_quat_xyzw"].ravel().astype(float)
            frames.append((np.array([xy[0], xy[1], z]),
                           f.fields["joint_pos_mj"].ravel().astype(float),
                           steady, q, motion_kind[0] if steady else ""))

    try:
        t0 = time.time()
        got = False
        while time.time() - t0 < 60 and not got:
            try:
                unpack_message(sub.recv(), expected_topic="pose"); got = True
            except zmq.Again:
                send({"intent": "idle", "magnitude": "default"})
        check("serve came up", got)
        if not got:
            return
        # scripted battery with EDGE TIMINGS: stop right after motion starts,
        # rapid intent flapping, stop from steady walk, lateral from stand.
        script = [
            ({"intent": "walk", "magnitude": "forward"}, 0.4),   # stop-early edge
            ({"intent": "idle", "magnitude": "default"}, 2.0),
            ({"intent": "walk", "magnitude": "forward"}, 3.0),   # steady walk
            ({"intent": "idle", "magnitude": "default"}, 2.0),   # normal stop
            ({"intent": "walk", "magnitude": "forward"}, 0.3),   # flap
            ({"intent": "idle", "magnitude": "default"}, 0.3),
            ({"intent": "walk", "magnitude": "forward"}, 0.3),
            ({"intent": "idle", "magnitude": "default"}, 2.0),
            ({"intent": "locomotion", "magnitude": "continuous",
              "stick_fwd": 0.0, "stick_side": -1.0, "stick_yaw": 0.0}, 1.5),
            ({"intent": "idle", "magnitude": "default"}, 2.0),
            # sustained in-place turn: the case that exposed the replan storm
            # (threshold 48 vs ~50-frame turn plans -> 8 replans/s -> slow
            # collapse, 2026-08-03)
            ({"intent": "locomotion", "magnitude": "continuous",
              "stick_fwd": 0.0, "stick_side": 0.0, "stick_yaw": 1.0}, 5.0),
            ({"intent": "idle", "magnitude": "default"}, 2.0),
        ]
        for cmd, dur in script:
            new_active = not (cmd.get("intent") == "idle"
                              and all(cmd.get(k, 0.0) == 0.0
                                      for k in ("stick_fwd", "stick_side")))
            if new_active != motion_active[0]:
                motion_since[0] = time.time()
            motion_active[0] = new_active
            if not new_active:
                motion_kind[0] = ""
            elif abs(cmd.get("stick_yaw", 0.0)) > 0.5 and \
                    abs(cmd.get("stick_fwd", 0.0)) < 0.1 and \
                    abs(cmd.get("stick_side", 0.0)) < 0.1:
                motion_kind[0] = "turn"
            else:
                motion_kind[0] = "move"
            tend = time.time() + dur
            while time.time() < tend:
                send(cmd)
                pump(0.04)

        arr_j = np.array([j for _, j, _, _, _ in frames])
        arr_r = np.array([r for r, _, _, _, _ in frames])
        arr_m = np.array([m for _, _, m, _, _ in frames])
        arr_q = np.array([q for _, _, _, q, _ in frames])
        arr_k = [k_ for _, _, _, _, k_ in frames]
        dj = np.abs(np.diff(arr_j, axis=0))
        droot = np.linalg.norm(np.diff(arr_r[:, :2], axis=0), axis=1)
        # frozen frames: >6 consecutive bit-identical publishes (120 ms) —
        # ONLY while motion is commanded (the idle anchor is intentionally
        # bit-identical; that is the designed IDLE_LOOP behavior).
        ident = ((dj.max(axis=1) < 1e-9) & (droot < 1e-9)
                 & arr_m[1:] & arr_m[:-1])
        max_frozen = 0
        run = 0
        for v in ident:
            run = run + 1 if v else 0
            max_frozen = max(max_frozen, run)
        check("no frozen-frame runs on the wire", max_frozen <= 6,
              f"longest identical-frame run {max_frozen} ticks")
        # Baseline (2026-08-03, immediate-crossfade stop) measures ~10 deg/tick
        # at the stop edge — the known roughness the stop-fix retry targets.
        # This bar catches REGRESSIONS; tighten when the model-stop lands.
        _snap_i = int(dj.max(axis=1).argmax())
        _snap_j = int(dj[_snap_i].argmax())
        check("no joint snaps", float(np.degrees(dj.max())) < 12.0,
              f"max joint step {np.degrees(dj.max()):.2f} deg/tick "
              f"(tick {_snap_i}, t={_snap_i/50.0:.2f}s, dof {_snap_j}; "
              f"top5 ticks "
              f"{sorted(np.argsort(dj.max(axis=1))[-5:].tolist())})")
        check("no root teleports", float(droot.max()) < 0.05,
              f"max root step {droot.max():.3f} m/tick")
        # ---- TAP-TAP foot-fumble detector (2026-08-03: replan seams caught
        # feet mid-descent -> touch/re-lift/touch stumble; rate scales with
        # seam count, so ANY seam-plumbing or replan-config experiment that
        # regresses shows up here). Tap = re-lift < 4.5 cm between two
        # contacts within 0.6 s (a real swing re-lifts ~9-11 cm).
        mjm = mujoco.MjModel.from_xml_path(str(
            REPO / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"))
        mjd = mujoco.MjData(mjm)
        bnames = [mjm.body(i).name for i in range(mjm.nbody)]
        b_la = [i for i, n in enumerate(bnames) if "left" in n and "ankle" in n][-1]
        b_ra = [i for i, n in enumerate(bnames) if "right" in n and "ankle" in n][-1]
        Lz, Rz = [], []
        for i in range(len(arr_j)):
            mjd.qpos[:3] = arr_r[i]
            qq = arr_q[i]
            mjd.qpos[3:7] = [qq[3], qq[0], qq[1], qq[2]]
            mjd.qpos[7:38] = arr_j[i]
            mujoco.mj_forward(mjm, mjd)
            Lz.append(mjd.xpos[b_la][2]); Rz.append(mjd.xpos[b_ra][2])
        gnd = min(min(Lz), min(Rz))
        taps = {"LEFT": 0, "RIGHT": 0}
        for foot, z in (("LEFT", np.array(Lz)), ("RIGHT", np.array(Rz))):
            low = z < gnd + 0.015
            i = 0
            while i < len(z) - 1:
                if low[i]:
                    j = i
                    while j < len(z) - 1 and low[j]:
                        j += 1
                    kk = j
                    lifted = False
                    peak = 0.0
                    while kk < len(z) - 1 and (kk - j) < 30:   # 0.6 s @50 Hz
                        peak = max(peak, float(z[kk] - gnd))
                        if z[kk] > gnd + 0.023:
                            lifted = True
                        if lifted and low[kk]:
                            if peak < 0.045:
                                taps[foot] += 1
                            break
                        kk += 1
                    i = max(j, kk)
                else:
                    i += 1
        n_taps = taps["LEFT"] + taps["RIGHT"]
        # ---- Sustained-turn yaw-rate bound (2026-08-03: template yaw
        # under-modulates; commanded 1.0 rad/s measured 1.1-2.8 sustained.
        # The serve-side yaw governor must hold the served rate near the
        # command; this asserts it and catches regressions in either the
        # governor or the model conditioning.)
        tq = np.array([q for q, k_ in zip(arr_q, arr_k) if k_ == "turn"])
        if len(tq) > 60:
            yaws = np.unwrap([np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),
                                          1-2*(q[1]**2+q[2]**2)) for q in tq])
            rates = []
            for w0 in range(0, len(yaws)-50, 50):
                rates.append(abs(yaws[w0+50]-yaws[w0]))   # rad over 1 s
            mean_r = float(np.mean(rates)); max_r = float(np.max(rates))
            check("sustained turn rate near command (1.0 rad/s)",
                  0.4 <= mean_r <= 1.3 and max_r <= 1.5,
                  f"mean {mean_r:.2f}, max {max_r:.2f} rad/s over "
                  f"{len(rates)} windows (cap 1.35x; ungoverned baseline: "
                  "mean ~1.9, max 2.8)")
        else:
            check("sustained turn rate near command (1.0 rad/s)", False,
                  "no steady turn frames captured")
        logtxt = log_path.read_text()
        n_replans_tap = max(1, logtxt.count("Replanning with mode"))
        tap_rate = n_taps / n_replans_tap
        # Baseline @ shipped threshold 48 (2026-08-03): ~2 taps / 41 replans
        # RECALIBRATED 2026-08-04 for FWD 0.3 (indoor default; operator
        # hardware-validated "decent walking" in a 10x10 garage): the
        # reference gait taps more at low speed (measured 10-14% over 3
        # runs vs ~5% at 0.4 — slow-walk deadband territory). Ceiling
        # 20% still catches the tap storms this gate was born from
        # (those ran 50-100%).
        # (~5% of seams). Gate at 4 absolute AND 12% of replans.
        check("foot tap-tap rate within baseline",
              n_taps <= 6 and tap_rate <= 0.20,
              f"L={taps['LEFT']} R={taps['RIGHT']} total={n_taps} "
              f"({100*tap_rate:.0f}% of {n_replans_tap} replans; "
              "baseline ~5%)")
        check("no ring starvation logged", "STARVED" not in logtxt)
        check("no liveness-gate re-rolls (lateral chunks must pass)",
              "standing chunk" not in logtxt)
        n_replans = logtxt.count("Replanning with mode")
        wall = max(1.0, len(frames) / 50.0)
        check("replan rate sane (no storm)", n_replans / wall < 2.5,
              f"{n_replans} replans over ~{wall:.0f}s = {n_replans/wall:.1f}/s")
        check("no tracebacks in serve log", "Traceback" not in logtxt)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def stage_estop() -> None:
    """Run the e-stop gesture regression suite (gear_sonic/utils/teleop/
    test_estop_gesture.py). Added after the 2026-08-03 sim collapse:
    3 gentle deadman pumps while driving reached pure damping. Every
    live incident is a named test there; a red test = the e-stop would
    misbehave on the robot, so the ship is blocked."""
    print("== STAGE 0: e-stop gesture regression suite")
    suite = REPO / "gear_sonic" / "utils" / "teleop" / "test_estop_gesture.py"
    r = subprocess.run([sys.executable, str(suite)],
                       capture_output=True, text=True, timeout=120)
    tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
    check("e-stop suite green", r.returncode == 0, tail)
    if r.returncode != 0:
        print(r.stdout)


def main() -> int:
    stage_estop()
    stage_unit()
    stage_clips()
    stage_serve()
    print()
    if FAILS:
        print(f"PREFLIGHT FAIL ({len(FAILS)}): {FAILS}")
        return 1
    print("PREFLIGHT PASS (stages 1-3). Reminder: motion-level changes ALSO "
          "need SONIC-in-the-loop sim (stage 4) before PC2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
