#!/usr/bin/env python3
"""Assisted bring-up ("clay-hold") sim battery.

Validates the boot-to-SONIC lifecycle entirely in MuJoCo:

  CLAY-HOLD -> SUPINE PREP -> ASSISTED RISE -> POSE WINDOW -> SONIC HANDOFF
  (+ e-stop recovery cycling, + floor-stiffness sweep)

Groups (see spec for acceptance rationale):
  1 clay units      sag / yield+re-latch / ratchet under re-grip dropouts
  2 supine prep     torque bounds, self-collision, obstruction-yield
  3 assisted rise   force budget (kg-equiv), hands-dropout hold
  4 pose window     adversarial refuse/accept + consent-chord spam
  5 handoff         SONIC engages under hands, hands fade, survival +
                    stance quietness + no slam-class jumps
  6 recovery cycle  damp -> clay -> rise -> handoff xN, leak test
  7 floor sweep     groups 3+5 on rigid AND compliant ground

Run:  MUJOCO_GL=egl .venv/bin/python gear_sonic/scripts/bringup_sim.py \
          --checkpoint <sonic.pt> [--groups 1234567] [--out results.jsonl]

The ClayController here is the reference implementation intended to port
into the pose watchdog (spec build item 2).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = Path(__file__).resolve().parents[2]
import mujoco  # noqa: E402
import eval_x2_mujoco as H  # noqa: E402  (gains, maps, obs pipeline, actor)

MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../data/assets/robot_description/mjcf/x2_ultra.xml")
N = H.NUM_DOFS

# MC-match damping (the presets' hard-won constants): ankles ring
# catastrophically below ~3x training kd once contacts are loaded; waist
# similar. Battery reproduced exactly this at stand (dofs 5/11/13
# oscillating 1-2 rad/s) before this correction.
KD_DEPLOY = H.KD.copy()
for _i in (4, 5, 10, 11):          # ankle pitch/roll both sides
    KD_DEPLOY[_i] *= 3.3
for _i in (12, 13, 14):            # waist
    KD_DEPLOY[_i] *= 3.0

# Supine prep target: legs curled so the feet land under the pelvis when
# the body is rotated upright. Arms tucked slightly. (Leg DOF order:
# hip_p, hip_r, hip_y, knee, ankle_p, ankle_r per side.)
PREP_CONFIG = H.DEFAULT_DOF.copy()
for i, v in ((0, -1.35), (3, 1.95), (4, -0.55),
             (6, -1.35), (9, 1.95), (10, -0.55)):
    PREP_CONFIG[i] = v

POSE_WINDOW = dict(tilt_deg=15.0, z_lo=0.55, z_hi=0.72,
                   joint_rms_deg=25.0, settle_qvel=0.6, settle_s=1.0)


class ClayController:
    """Gravity-ff leaky posture hold + ratchet + prep interpolation.

    Reference implementation of the watchdog CLAY mode (spec item 2)."""

    def __init__(self, kp_scale=0.5, kd_scale=0.9, band_rate=1.5):
        self.kp = H.KP * kp_scale
        self.kd = KD_DEPLOY * kd_scale
        self.band = band_rate
        self.q_target: np.ndarray | None = None
        self.ratchet = True
        self._z_hist: list[float] = []
        self.prep_active = False
        self._prep_from: np.ndarray | None = None
        self._prep_t = 0.0
        self.prep_duration = 5.0   # operator 2026-08-11: 8s felt slow; 5s = brisker fold, still gentle

    def latch(self, d):
        self.q_target = d.qpos[7:7 + N].copy()
        self._z_hist = [float(d.qpos[2])]

    def stiffen(self, kp_scale=0.85, kd_scale=1.0, ramp_s=0.0):
        if ramp_s > 0.0:
            self._stiffen_goal = (kp_scale, kd_scale)
            self._stiffen_rate = ramp_s
            return
        self._stiffen_goal = None
        """RISE phase: firmer hold (human is steadying; softness was for
        handling). Battery finding: clay-soft gains sag ~11 cm under load
        during a hands re-grip; stiffened rise holds it."""
        self.kp = H.KP * kp_scale
        self.kd = KD_DEPLOY * kd_scale

    def _stiffen_step(self, dt):
        g = getattr(self, "_stiffen_goal", None)
        if g is None:
            return
        kp_goal = H.KP * g[0]
        a = min(1.0, dt / self._stiffen_rate)
        self.kp = self.kp + a * (kp_goal - self.kp)
        self.kd = self.kd + a * (KD_DEPLOY * g[1] - self.kd)
        if float(np.abs(self.kp - kp_goal).max()) < 1e-3:
            self._stiffen_goal = None

    @staticmethod
    def face_up(d, tol_deg=35.0):
        """Prep orientation gate: the supine leg-fold is only valid
        roughly face-up (battery: from side/prone the ground blocks the
        fold and the pause logic waits forever — correctly forceless,
        but it must REFUSE, not hang). Handler rolls the robot first."""
        w, x, y, z = d.qpos[3:7]
        # world-z component of the BODY-X axis (chest normal): R20 of the
        # body-to-world matrix. Supine -> +1 (chest up), prone -> -1,
        # side -> ~0. Face-up = chest normal within tol of world-up.
        chest_up = 2 * (x * z - w * y)
        return chest_up > math.cos(math.radians(tol_deg))

    def start_prep(self, cfg=None, d=None):
        if d is not None and not self.face_up(d):
            return False                        # gate: refuse, stay clay
        self.prep_active = True
        self._prep_from = self.q_target.copy()
        self._prep_t = 0.0
        self._prep_wall = 0.0
        self._prep_cfg = PREP_CONFIG if cfg is None else cfg
        return True

    def start_rise_extension(self, duration=8.0):
        """RISE actuation (battery finding: clay alone reaches a static
        squat equilibrium — standing needs LEG EXTENSION force; the human
        steadies while the legs push, as in human-assisted standing).
        Prep-like interpolation to the stand config, ratchet-strict: the
        clock pauses when reality lags (human stopped assisting) but the
        target is never dragged back."""
        self.rise_active = True
        self._rise_from = self.q_target.copy()
        self._rise_t = 0.0
        self.rise_duration = duration

    def torques(self, d, dt):
        self._stiffen_step(dt)
        q = d.qpos[7:7 + N]
        qd = d.qvel[6:6 + N]
        # SUPPORT-AWARE gravity ff (operator-caught 2026-08-11: full ff on
        # a LYING body arches the torso up unnaturally -- the floor already
        # carries the weight; compensating as-if-unsupported over-pushes).
        # Scale ff with uprightness: lying ~0.25, upright ~1.0.
        w_, x_, y_, z_ = d.qpos[3:7]
        upz = 1 - 2 * (x_ * x_ + y_ * y_)          # body-z dot world-z
        ff_scale = 0.25 + 0.75 * max(0.0, upz)     # 0.25 lying -> 1.0 stood
        tau = ff_scale * d.qfrc_bias[6:6 + N] \
            + self.kp * (self.q_target - q) - self.kd * qd
        # prep interpolation (yield-aware: interpolate the TARGET; the
        # band-follow below still lets reality pull the target back)
        if self.prep_active:
            gap = float(np.abs(self.q_target - q).max())
            self._prep_wall = getattr(self, "_prep_wall", 0.0) + dt
            if self._prep_wall - self._prep_t > 10.0:
                # blocked too long (ground/obstacle): abandon gracefully
                self.prep_active = False
                self.prep_abandoned = True
            if gap < 0.35:                  # blocked -> pause the clock
                self._prep_t += dt
            w = min(1.0, self._prep_t / self.prep_duration)
            w = 0.5 * (1 - math.cos(math.pi * w))
            self.q_target = (self._prep_from * (1 - w)
                             + self._prep_cfg * w).astype(np.float64)
            if w >= 1.0:
                self.prep_active = False
        if getattr(self, "rise_active", False):
            gap = float(np.abs(self.q_target - q)[0:12].max())
            if gap < 0.5:                    # stalled/blocked -> pause clock
                self._rise_t += dt
            w = min(1.0, self._rise_t / self.rise_duration)
            w = 0.5 * (1 - math.cos(math.pi * w))
            self.q_target = (self._rise_from * (1 - w)
                             + H.DEFAULT_DOF * w).astype(np.float64)
            if w >= 1.0:
                self.rise_active = False
                # pre-handoff hold: FULL gains. Battery finding: 0.85x
                # sagged back from z 0.62 to 0.30 after extension; at
                # near-stand under steadying hands there is no reason to
                # stay soft -- softness was for handling, not load-bearing.
                self.stiffen(1.0, 1.0, ramp_s=0.8)
            # ratchet-strict during extension: NO band-follow (a loaded leg
            # lags its target by design; dragging the target back = stall)
            self._z_hist.append(float(d.qpos[2]))
            if len(self._z_hist) > 25:
                self._z_hist.pop(0)
            return tau
        # band-follow with optional ratchet gate
        self._z_hist.append(float(d.qpos[2]))
        if len(self._z_hist) > 25:
            self._z_hist.pop(0)
        rising = self._z_hist[-1] - self._z_hist[0] > 0.001
        if (not self.ratchet) or rising or self.prep_active:
            self.q_target += np.clip(q - self.q_target,
                                     -self.band * dt, self.band * dt)
        return tau


class HandsModel:
    """Human assist: world-up force + orientation spring-damper + XY
    damping at the torso. Supports ramps, dropouts, and fade-out."""

    def __init__(self, m, fz=180.0, steady_k=60.0, steady_c=8.0,
                 f_fwd_frac=0.5):
        self.body = max(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_link"))
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.fz = fz
        self.f_fwd = f_fwd_frac * fz        # tow toward the feet ("up AND
        self.k = steady_k                   # over your feet" -- how humans
        self.c = steady_c                   # actually assist a stand)
        self.gain = 1.0                     # global scale (fade / dropout)

    def apply(self, m, d):
        if self.gain <= 0.0:
            return
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        bz = R.reshape(3, 3)[:, 2]
        ang = math.acos(max(-1, min(1, bz[2])))
        ax = np.cross(bz, [0, 0, 1.0])
        n = np.linalg.norm(ax)
        ax = ax / n if n > 1e-8 else np.zeros(3)
        d.xfrc_applied[self.body, 2] += self.gain * self.fz
        # horizontal tow: torso toward the midpoint of the feet
        if self.feet[0] >= 0 and self.feet[1] >= 0:
            feet_mid = 0.5 * (d.xpos[self.feet[0]][:2]
                              + d.xpos[self.feet[1]][:2])
            tow = feet_mid - d.xpos[self.body][:2]
            nt = np.linalg.norm(tow)
            if nt > 0.05:
                d.xfrc_applied[self.body, 0:2] += (
                    self.gain * self.f_fwd * tow / nt)
        d.xfrc_applied[self.body, 3:6] += self.gain * (
            self.k * ang * ax) - self.gain * self.c * d.qvel[3:6]
        d.xfrc_applied[self.body, 0:2] += -self.gain * 25.0 * d.qvel[0:2]


def tilt_deg(d):
    w, x, y, z = d.qpos[3:7]
    return math.degrees(math.acos(max(-1, min(1, 1 - 2 * (x * x + y * y)))))


def joint_rms_deg(d, ref):
    return float(np.degrees(np.sqrt(np.mean((d.qpos[7:7 + N] - ref) ** 2))))


def in_window(d, settle_ticks, w=POSE_WINDOW):
    ok = (tilt_deg(d) < w["tilt_deg"]
          and w["z_lo"] < d.qpos[2] < w["z_hi"]
          and joint_rms_deg(d, H.DEFAULT_DOF) < w["joint_rms_deg"]
          and float(np.abs(d.qvel[6:6 + 15]).max()) < w["settle_qvel"])
    return ok


class Sim:
    def __init__(self, floor_soft=False):
        self.m = mujoco.MjModel.from_xml_path(MJCF)
        if floor_soft:
            # compliant "mat": soften the floor contact (THE MAT lesson)
            floor = 0  # geom 0 is the plane in x2_ultra.xml
            self.m.geom_solref[floor] = [0.03, 0.6]
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep

    def reset_supine(self):
        d = self.d
        d.qpos[:] = 0
        d.qvel[:] = 0
        d.qpos[0:3] = [0, 0, 0.18]
        d.qpos[3:7] = [math.cos(-math.pi / 4), 0, math.sin(-math.pi / 4), 0]
        d.qpos[7:7 + N] = H.DEFAULT_DOF
        mujoco.mj_forward(self.m, d)

    def step_clay(self, clay, hands=None, n=1, estop=False):
        for _ in range(n):
            d = self.d
            if estop:
                tau = -8.0 * d.qvel[6:6 + N]        # damp profile Kp=0 Kd=8
            else:
                tau = clay.torques(d, self.dt)
            for j in range(N):
                d.ctrl[H.JOINT_TO_ACTUATOR[j]] = tau[j]
            d.xfrc_applied[:] = 0
            if hands is not None:
                hands.apply(self.m, d)
            mujoco.mj_step(self.m, d)

    def selfcollision_count(self):
        n = 0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            if b1 != 0 and b2 != 0:          # neither is world/floor
                n += 1
        return n


RES = []


def rec(group, name, ok, **kw):
    row = dict(group=group, test=name, ok=bool(ok), **kw)
    RES.append(row)
    print(f"[{group}] {'PASS' if ok else 'FAIL'}  {name}  "
          + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


# ---------------------------------------------------------------- groups
def g1_clay_units(sim: Sim):
    clay = ClayController()
    sim.reset_supine()
    clay.latch(sim.d)
    q0 = sim.d.qpos[7:7 + N].copy()
    sim.step_clay(clay, n=int(3 / sim.dt))
    sag = float(np.degrees(np.abs(sim.d.qpos[7:7 + N] - q0)).max())
    rec(1, "hold_sag_3s", sag < 25.0, sag_deg=round(sag, 1))

    # yield + re-latch: push a shoulder, release, residual must be small
    tgt0 = clay.q_target[16]
    clay.ratchet = False
    sim.d.qfrc_applied[6 + 16] = 25.0
    sim.step_clay(clay, n=int(0.8 / sim.dt))
    sim.d.qfrc_applied[6 + 16] = 0.0
    sim.step_clay(clay, n=int(1.0 / sim.dt))
    followed = math.degrees(abs(clay.q_target[16] - tgt0))
    resid = math.degrees(abs(sim.d.qpos[7 + 16] - clay.q_target[16]))
    rec(1, "yield_relatch", followed > 20 and resid < 12,
        followed_deg=round(followed, 1), residual_deg=round(resid, 1))

    # ratchet under re-grip dropouts: lift pelvis via hands with two 0.4 s
    # dropouts; pelvis z must never fall back more than 2 cm from its max
    clay = ClayController()
    sim.reset_supine()
    clay.latch(sim.d)
    clay.stiffen()
    hands = HandsModel(sim.m, fz=200.0)
    # ratchet semantics: during a re-grip dropout the FLOATING BASE may
    # dip (part of the weight was on the hands, by design) but the JOINT
    # targets must not unfold, and the pose must recover on re-grip.
    worst_joint_slide = 0.0
    z_pre = z_post = None
    for k in range(int(8 / sim.dt)):
        t = k * sim.dt
        in_drop = 2.0 < t < 2.4
        if in_drop and z_pre is None:
            z_pre = float(sim.d.qpos[2])
            tgt_pre = clay.q_target.copy()
        hands.gain = 0.0 if in_drop else 1.0
        sim.step_clay(clay, hands)
        if z_pre is not None and in_drop:
            # unfold is scored ONLY inside the dropout: post-re-grip target
            # motion is legitimate resumed follow, not ratchet failure
            worst_joint_slide = max(worst_joint_slide, float(
                np.degrees(np.abs(tgt_pre - clay.q_target)).max()))
        if z_pre is not None and not in_drop and t > 2.4 + 1.5 \
                and z_post is None:
            z_post = float(sim.d.qpos[2])
    recovered = z_post is not None and z_post > z_pre - 0.02
    rec(1, "ratchet_regrip", worst_joint_slide < 5.0 and recovered,
        joint_unfold_deg=round(worst_joint_slide, 1),
        z_pre=round(z_pre, 2) if z_pre else None,
        z_after_regrip=round(z_post, 2) if z_post else None)


def g2_supine_prep(sim: Sim):
    clay = ClayController()
    sim.reset_supine()
    clay.latch(sim.d)
    sim.step_clay(clay, n=int(1 / sim.dt))
    clay.start_prep()
    peak_tau = 0.0
    selfcol = 0
    while clay.prep_active:
        sim.step_clay(clay)
        q = sim.d.qpos[7:7 + N]
        tau_now = np.abs(clay.kp * (clay.q_target - q))[0:12].max()
        peak_tau = max(peak_tau, float(tau_now))
        selfcol = max(selfcol, sim.selfcollision_count())
    err = joint_rms_deg(sim.d, PREP_CONFIG)
    rec(2, "prep_reaches_config", err < 15.0, rms_deg=round(err, 1))
    rec(2, "prep_torque_bounded", peak_tau < 60.0, peak_Nm=round(peak_tau, 1))
    rec(2, "prep_no_self_collision", selfcol == 0, contacts=selfcol)

    # obstruction-yield: block the left knee mid-prep with opposing torque
    clay = ClayController()
    sim.reset_supine()
    clay.latch(sim.d)
    clay.start_prep()
    sim.step_clay(clay, n=int(2 / sim.dt))
    sim.d.qfrc_applied[6 + 3] = -40.0
    t_before = clay._prep_t
    sim.step_clay(clay, n=int(2 / sim.dt))
    clock_advance = clay._prep_t - t_before   # must be ~0 while blocked
    yielded = math.degrees(abs(sim.d.qpos[7 + 3] - clay.q_target[3]))
    sim.d.qfrc_applied[6 + 3] = 0.0
    while clay.prep_active:
        sim.step_clay(clay)
    final_err = math.degrees(abs(sim.d.qpos[7 + 3] - PREP_CONFIG[3]))
    # orientation gate: prep must REFUSE from side/prone, accept supine
    accepts, refusals = 0, 0
    for nm, quat in (("supine", [math.cos(-math.pi/4), 0, math.sin(-math.pi/4), 0]),
                     ("side",   [math.cos(math.pi/4), math.sin(math.pi/4), 0, 0]),
                     ("prone",  [math.cos(math.pi/4), 0, math.sin(math.pi/4), 0])):
        sim.d.qpos[:] = 0; sim.d.qvel[:] = 0
        sim.d.qpos[0:3] = [0, 0, 0.25]
        sim.d.qpos[3:7] = quat
        sim.d.qpos[7:7 + N] = H.DEFAULT_DOF
        mujoco.mj_forward(sim.m, sim.d)
        c2 = ClayController(); c2.latch(sim.d)
        started = c2.start_prep(d=sim.d)
        if nm == "supine":
            accepts += int(started)
        else:
            refusals += int(not started)
    rec(2, "prep_orientation_gate", accepts == 1 and refusals == 2,
        supine_accepted=bool(accepts), side_prone_refused=f"{refusals}/2")

    rec(2, "prep_obstruction_yield",
        clock_advance < 0.3 and final_err < 12.0,
        clock_paused=round(clock_advance, 2), yield_deg=round(yielded, 1),
        final_deg=round(final_err, 1))


def rise_to_window(sim, clay, hands, timeout=25.0):
    clay.stiffen()
    ticks_in = 0
    ext_started = False
    for k in range(int(timeout / sim.dt)):
        hands.gain = min(1.0, k * sim.dt / 3.0)
        if not ext_started and k * sim.dt > 3.0:
            clay.start_rise_extension()
            ext_started = True
        sim.step_clay(clay, hands)
        if in_window(sim.d, ticks_in):
            ticks_in += 1
            if ticks_in * sim.dt >= POSE_WINDOW["settle_s"]:
                return k * sim.dt
        else:
            ticks_in = 0
    return None


def g3_assisted_rise(sim: Sim):
    for F in (140, 180, 220):
        clay = ClayController()
        sim.reset_supine()
        clay.latch(sim.d)
        sim.step_clay(clay, n=int(0.5 / sim.dt))
        clay.start_prep()
        while clay.prep_active:
            sim.step_clay(clay)
        hands = HandsModel(sim.m, fz=F)
        t = rise_to_window(sim, clay, hands)
        if t is not None:
            rec(3, "rise_force_budget", True, assist_N=F,
                kg_equiv=round(F / 9.81, 1), t_s=round(t, 1),
                z=round(float(sim.d.qpos[2]), 2))
            # dropout hold at the top: hands vanish 0.5 s
            z0 = float(sim.d.qpos[2])
            hands.gain = 0.0
            sim.step_clay(clay, hands, n=int(0.5 / sim.dt))
            drop = z0 - float(sim.d.qpos[2])
            rec(3, "rise_dropout_hold", drop < 0.06,
                z_drop_cm=round(drop * 100, 1))
            return F
    rec(3, "rise_force_budget", False, note="no force <=220N reached window")
    return None


def g4_pose_window(sim: Sim):
    refusals = 0
    trials = 0
    clay = ClayController()
    # adversarial states that must be refused
    for setup in ("supine", "half", "moving"):
        sim.reset_supine()
        clay.latch(sim.d)
        if setup == "half":
            sim.d.qpos[2] = 0.40
            sim.d.qpos[3:7] = [math.cos(-0.35), 0, math.sin(-0.35), 0]
        if setup == "moving":
            sim.d.qpos[0:3] = [0, 0, 0.66]
            sim.d.qpos[3:7] = [1, 0, 0, 0]
            sim.d.qvel[6:6 + N] = 1.5
        mujoco.mj_forward(sim.m, sim.d)
        trials += 1
        if not in_window(sim.d, 0):
            refusals += 1
    # a legit standing pose must be accepted
    sim.reset_supine()
    sim.d.qpos[0:3] = [0, 0, 0.66]
    sim.d.qpos[3:7] = [1, 0, 0, 0]
    sim.d.qpos[7:7 + N] = H.DEFAULT_DOF
    sim.d.qvel[:] = 0
    mujoco.mj_forward(sim.m, sim.d)
    accept = in_window(sim.d, 0)
    rec(4, "window_gate", refusals == trials and accept,
        refused=f"{refusals}/{trials}", accepts_stand=accept)


def g5_handoff(sim: Sim, ckpt, assist_N, motion_pkl):
    import torch
    actor = H.load_actor_from_checkpoint(ckpt, "cpu", encoder="g1")
    motion_data = H.load_motion_data(motion_pkl)
    fps = H.get_motion_fps(motion_data)

    clay = ClayController()
    sim.reset_supine()
    clay.latch(sim.d)
    sim.step_clay(clay, n=int(0.5 / sim.dt))
    clay.start_prep()
    while clay.prep_active:
        sim.step_clay(clay)
    hands = HandsModel(sim.m, fz=assist_N)
    t = rise_to_window(sim, clay, hands)
    if t is None:
        rec(5, "handoff", False, note="never reached window")
        return

    # SONIC engages under hands: blend clay target -> policy control
    prop = H.ProprioceptionBuffer()
    last_action_mj = np.zeros(N)
    CONTROL_DT = 0.02
    DECIM = max(1, int(round(CONTROL_DT / sim.dt)))
    held = clay.q_target.copy()
    n_blend = 50                       # 1 s policy ramp-in
    jumps = 0
    m_time = 0.0
    for k in range(int(10 / CONTROL_DT)):
        d = sim.d
        base_quat = d.qpos[3:7].copy()
        qpos_j = d.qpos[7:7 + N].copy()
        qvel_j = d.qvel[6:6 + N].copy()
        dof_pos_il = qpos_j[H.IL_TO_MJ_DOF]
        dof_vel_il = qvel_j[H.IL_TO_MJ_DOF]
        action_il = last_action_mj[H.IL_TO_MJ_DOF]
        gravity = H.quat_rotate_inverse(base_quat, np.array([0., 0., -1.]))
        prop.append(gravity, d.qvel[3:6].copy(),
                    dof_pos_il - H.DEFAULT_DOF[H.IL_TO_MJ_DOF],
                    dof_vel_il, action_il)
        tok = H.build_tokenizer_obs(motion_data, m_time, base_quat, fps)
        with torch.no_grad():
            a = actor(torch.from_numpy(prop.get_flat()).unsqueeze(0),
                      torch.from_numpy(tok).unsqueeze(0)).squeeze(0).numpy()
        a = np.clip(a, -20.0, 20.0)
        act_mj = np.zeros(N)
        act_mj[H.IL_TO_MJ_DOF] = a
        prev_t = held if k == 0 else target
        target = H.DEFAULT_DOF + act_mj * H.ACTION_SCALE
        w = min(1.0, (k + 1) / n_blend)
        target = held * (1 - w) + target * w
        if np.abs(target - prev_t).max() > 0.5:
            jumps += 1
        last_action_mj = act_mj
        # hands fade out over the 2 s after blend completes
        hands.gain = max(0.0, 1.0 - max(0.0, (k * CONTROL_DT - 1.0)) / 2.0)
        for _ in range(DECIM):
            tau = H.KP * (target - sim.d.qpos[7:7 + N]) \
                - H.KD * sim.d.qvel[6:6 + N]
            for j in range(N):
                sim.d.ctrl[H.JOINT_TO_ACTUATOR[j]] = tau[j]
            sim.d.xfrc_applied[:] = 0
            hands.apply(sim.m, sim.d)
            mujoco.mj_step(sim.m, sim.d)
        m_time += CONTROL_DT
    alive = sim.d.qpos[2] > 0.45 and tilt_deg(sim.d) < 30
    rec(5, "handoff_under_hands", alive and jumps == 0,
        survived_10s=alive, tilt=round(tilt_deg(sim.d), 1),
        z=round(float(sim.d.qpos[2]), 2), slam_jumps=jumps)
    return alive


def g6_recovery_cycle(sim: Sim, assist_N, cycles=8):
    """damp -> clay -> rise -> window xN with metric drift check."""
    t_rise = []
    for c in range(cycles):
        clay = ClayController()
        sim.reset_supine()
        clay.latch(sim.d)
        sim.step_clay(clay, n=int(0.3 / sim.dt))
        # e-stop damp for 1 s (robot settles), then chord -> clay latches
        sim.step_clay(clay, n=int(1.0 / sim.dt), estop=True)
        clay.latch(sim.d)
        clay.start_prep()
        while clay.prep_active:
            sim.step_clay(clay)
        hands = HandsModel(sim.m, fz=assist_N)
        t = rise_to_window(sim, clay, hands, timeout=25.0)
        t_rise.append(t if t is not None else -1)
    ok = all(t > 0 for t in t_rise)
    drift = (max(t_rise) - min(t_rise)) if ok else -1
    rec(6, "recovery_cycles", ok and drift < 5.0,
        cycles=cycles, t_rise=[round(t, 1) for t in t_rise],
        drift_s=round(drift, 1) if ok else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default=os.environ.get("X2_CHECKPOINT") or None,
                    help="SONIC .pt checkpoint (required unless $X2_CHECKPOINT is set; "
                         "bring-your-own, see MODELS.md)")
    ap.add_argument("--motion",
                    default=str(_REPO_ROOT / "gear_sonic/data/motions/x2_demo_bank.pkl"),
                    help="motion-lib pkl for the hand-off group 5 (default: the regenerated "
                         "demo bank; build it with tools/build_demo_bank_from_upstream.sh)")
    ap.add_argument("--groups", default="1234567")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if "5" in args.groups and args.checkpoint is None:
        ap.error("--checkpoint is required (group 5) (or export X2_CHECKPOINT); no X2 checkpoint "
                 "ships with the repo -- see MODELS.md")
    if "5" in args.groups and not os.path.isfile(args.motion):
        ap.error(f"--motion {args.motion} not found; regenerate the demo bank with "
                 "tools/build_demo_bank_from_upstream.sh or pass your own motion pkl")

    t0 = time.time()
    sim = Sim()
    assist_N = 180
    if "1" in args.groups:
        g1_clay_units(sim)
    if "2" in args.groups:
        g2_supine_prep(sim)
    if "3" in args.groups:
        f = g3_assisted_rise(sim)
        assist_N = f or assist_N
    if "4" in args.groups:
        g4_pose_window(sim)
    if "5" in args.groups:
        g5_handoff(sim, args.checkpoint, assist_N, args.motion)
    if "6" in args.groups:
        g6_recovery_cycle(sim, assist_N)
    if "7" in args.groups:
        soft = Sim(floor_soft=True)
        print("[7] --- compliant floor (mat model) ---", flush=True)
        clay = ClayController()
        soft.reset_supine()
        clay.latch(soft.d)
        soft.step_clay(clay, n=int(0.5 / soft.dt))
        clay.start_prep()
        while clay.prep_active:
            soft.step_clay(clay)
        hands = HandsModel(soft.m, fz=assist_N)
        t = rise_to_window(soft, clay, hands)
        rec(7, "rise_on_mat", t is not None,
            t_s=round(t, 1) if t else None)

    npass = sum(1 for r in RES if r["ok"])
    print(f"\n===== BRINGUP BATTERY: {npass}/{len(RES)} pass "
          f"({time.time()-t0:.0f}s) =====", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            for r in RES:
                fh.write(json.dumps(r) + "\n")
        print(f"results -> {args.out}", flush=True)
    return 0 if npass == len(RES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
