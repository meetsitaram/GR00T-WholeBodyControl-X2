#!/usr/bin/env python3
"""Torch-free X2 kinematic planner runtime for PC2 (Jetson).

A slim port of ``gear_sonic/scripts/x2_kplanner.py``'s publish/replan loop
that runs the fused planner graph through **onnxruntime** (CPU EP) instead
of torch, so the whole planner stack can live on the robot's Jetson with
no laptop in the loop.

Architecture (all on PC2):

  - SUB ``planner_cmd``  on tcp://127.0.0.1:5563 (published by
    ``pad_locomotion_bridge --bind``; JSON payloads
    ``{"intent":"locomotion","magnitude":"continuous","stick_fwd",
    "stick_side","stick_yaw"[,"speed_delta"]}``).
  - Ring buffer of planner qpos frames ([T, 38] MuJoCo qpos, world frame,
    wxyz root quat) refilled by a worker thread whenever occupancy drops
    below ``--replan-threshold-frames`` -- scheduling / threshold /
    cadence logic ported from ``x2_kplanner._planner_worker`` +
    ``NeuralPlannerCore.get_next_frame``.
  - Replan backend: the ONNX fused graph exported by
    ``motionbricks/scripts/export_x2_planner_onnx.py``:

        inputs : context_mujoco_qpos f32 [1, 4, 38]
                 velocity_intent     f32 [1, 4]
                 (template graph)    mode i64 [1], random_seed i64 [1]
        outputs: mujoco_qpos         f32 [1, 64, 38] (padded)
                 num_pred_frames     i64 [1]

    Tensor names / graph flavour are data-driven from an optional JSON
    sidecar next to the .onnx (see ``_load_onnx_contract``); the graph
    bakes the FILTER_QPOS first-4-frame context blend in-graph
    (``x2_kplanner`` never applies it Python-side -- it lives inside
    ``NeuralPlannerCore._predict_*`` which the export traces), so the
    runtime must NOT reapply it unless the sidecar says
    ``"filter_qpos_baked": false``.
  - ``--backend torch`` (laptop A/B only): drives the *real*
    ``motionbricks`` ``load_x2_planner`` + ``NeuralPlannerCore`` exactly
    like ``x2_kplanner.run()`` does, proving this file's glue independent
    of the ONNX graph. torch is imported lazily inside that branch only.
  - 50 Hz publisher: PUB bind tcp://*:5556 topic ``pose``, VLA wire
    format (v4 packed message). The frame dict is constructed with the
    exact field set + insertion order of
    ``gear_sonic.utils.planner.state_machine.build_pose_payload`` (the
    consumer decodes by order); byte-encoded via
    ``gear_sonic.utils.pose_pipeline.wire.pack_pose_message`` which is
    byte-identical to the ``zmq_planner_sender`` encoder x2_kplanner uses
    when called with ``version=4``.
  - Frame pacing: identical to x2_kplanner -- the publish loop pops ONE
    ring-buffer frame per 50 Hz tick (``get_next_frame`` clamps at the
    buffer tail), i.e. the model's 30 fps frames are consumed at the
    50 Hz wire rate with no interpolation, and the 9-slot future window
    peeks at ``cursor + 5*(k+1) - 1`` (0.1 s spacing on the wire clock).
  - DANCE PLAYBACK: SUB bind tcp://*:5568 topic ``motion_clip_cmd``
    accepting the ``play_locomotion`` payload
    ``{"action":"play","pkl":...,"motion_key":k,"kind":"locomotion"}``
    and ``{"action":"stop"}``. On play the x2m2 bake
    ``<dances-dir>/<motion_key>.x2m2`` is streamed through the same
    50 Hz publisher (planner output preempted, ring paused); root quats
    are yaw-rebased onto the CURRENT streamed heading (delta-rebase via
    ``rebase_quats_xyzw_by_yaw``, mirroring the fallback ladder's
    ``build_idle_frame_msg`` rebase) so the dance never snaps the robot's
    heading. On clip end / stop, the idle anchor pose streams for
    ``--post-dance-idle-s`` (~2 s) before normal planner idle resumes.

Dependency budget (PC2 Jetson venv): stdlib + numpy + pyzmq + joblib +
onnxruntime + ``gear_sonic.utils.pose_pipeline.*`` (numpy-only). NO torch
import outside the ``--backend torch`` branch.

Known deviations from x2_kplanner (all intentional, documented):
  - No closed-loop robot_pose reseed / yaw refresh (x2_kplanner's default
    config also runs open-loop; real-robot deploys used reseed scope
    ``none`` anyway).
  - No waist-overlay path (``hold_torso`` waist_*_deg are ignored; the
    pad bridge never emits them). ``hold_torso`` still resolves to idle
    with the optional ``hip_height_m`` override, as upstream.
  - No scripted-YAML / keyboard command sources (ZMQ only).
  - No instant replan on command change. The MotionBricks paper replans
    on "C changed OR buffer running low" (Alg. 1; Appendix C "instant
    control reaction"; their Orin deploy replans "at 10 Hz or whenever
    commands change"). Here a mid-walk intent change only updates
    IntentState and is read at the NEXT drop-frame replan
    (``--replan-threshold-frames``); only IDLE->PLAYING forces one
    (``_force_replan``). Deliberate: PC2 CPU inference is 0.3-0.6 s and
    VR/pad sticks stream continuously-varying commands, so
    change-triggered replans would fire nonstop and churn seams.
    Consequence: the threshold is the ONLY mid-walk responsiveness
    lever -- raising 32->48 fires the replan 16 model frames (~0.53 s)
    earlier, measured turn response 0.68 s -> 0.22 s (see
    gear_sonic_deploy/configs/kplanner_tuning_history.md, 2026-08-02).

PC2 launch (defaults are the real PC2 ports/paths)::

    PYTHONPATH=/home/run/gear-sonic/planner_stack/gear_sonic \
    python pc2_kplanner_onnx.py \
        --onnx /home/run/gear-sonic/planner_stack/models/planner_onnx/x2_planner_velocity.onnx

Laptop A/B (live stack owns 5556/5563/5568 -- offset everything)::

    .venv/bin/python gear_sonic/scripts/pc2_kplanner_onnx.py \
        --backend torch --port-offset 100 --device cpu \
        --vqvae-ckpt ... --pose-ckpt ... --root-ckpt ... \
        --warmup-qpos gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import queue
import random
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Make ``gear_sonic`` importable when launched as a plain file with the
# repo (or the PC2 planner_stack copy) as the parent tree.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.pose_pipeline.wire import (  # noqa: E402
    DEFAULT_HAND_DOF,
    NUM_BODY_DOFS,
    SONIC_MOTION_TOKEN_DIM,
    decode_x2_debug_base_quat,
    decode_x2_debug_fields,
    load_x2m2,
    pack_pose_message,
    rebase_quats_xyzw_by_yaw,
    yaw_from_quat_wxyz,
)

log = logging.getLogger("pc2_kplanner_onnx")

# ---------------------------------------------------------------------------
# Wire / timing constants (mirror x2_kplanner + state_machine.OUTPUT_FPS).
# ---------------------------------------------------------------------------
OUTPUT_FPS: float = 50.0
MODEL_FPS: float = 30.0     # kplanner model output rate (resample source rate)

QPOS_DIM: int = 38
NUM_FUTURE: int = 9         # future-window slots on the wire
STEP_TICKS: int = 5         # 50 Hz ticks between future slots (= 0.1 s)
FUTURE_DT_S: float = 0.1

# PC2 defaults (document the real deployment surface).
DEFAULT_PUB_PORT: int = 5556           # PUB bind, topic "pose"
DEFAULT_CMD_PORT: int = 5563           # SUB connect, topic "planner_cmd"
DEFAULT_CLIP_CMD_PORT: int = 5568      # SUB bind, topic "motion_clip_cmd"
# PC2_PREFIX = the robot-side install prefix (deploy tree on PC2).
_PC2_PREFIX = Path(os.environ.get("PC2_PREFIX", "/home/run/gear-sonic"))
DEFAULT_WARMUP_PKL = (
    _PC2_PREFIX / "planner_stack/models/kplanner_idle_anchor_g1teleop_v3.pkl"
)
DEFAULT_DANCES_DIR = _PC2_PREFIX / "planner_stack/models/dances_x2m2"


def _slerp_wxyz_np(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    """Shortest-arc SLERP between two wxyz quaternions (numpy port).

    Mirrors ``NeuralPlannerCore._slerp_wxyz`` so the ONNX runtime resamples
    the root orientation identically to the torch stack.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + w * (q1 - q0)
        return (out / (np.linalg.norm(out) + 1e-8)).astype(np.float32)
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta0 * w
    sin0 = math.sin(theta0)
    s0 = math.sin(theta0 - theta) / sin0
    s1 = math.sin(theta) / sin0
    out = s0 * q0 + s1 * q1
    return (out / (np.linalg.norm(out) + 1e-8)).astype(np.float32)

# ---------------------------------------------------------------------------
# Intent -> velocity dispatcher. Ported verbatim (numbers + shaping) from
# x2_kplanner.py; see that file for the full channel-convention rationale.
# Velocity tuple layout: (yaw_rate_rad_s, vel_x=lateral, vel_z=forward, hip_h).
# ---------------------------------------------------------------------------
_WALK_SPEED_MPS: float = 0.6
_FAST_WALK_SPEED_MPS: float = 0.9
_SIDE_SPEED_MPS: float = 0.4
_BACK_SPEED_MPS: float = 0.45

_TEST_FIXED_BACK_MPS: float = 0.30
_TEST_FIXED_SIDE_MPS: float = 0.30
# Turn-rate setpoint. 0.30 was the original conservative test value; the
# 2026-07-21 sweep (kplanner_turnrate_sweep.py) showed the root head scales
# to 1.3 rad/s with NO still chunks at >=0.8 (strong conditioning escapes
# the idle attractor and the pose head generates real turn stepping), and
# the robot tracks the reference heading ~1:1. Override like the forward
# setpoint: env KPLANNER_FIXED_TURN_RAD_S.
_TEST_FIXED_TURN_RAD_S: float = float(
    os.environ.get("KPLANNER_FIXED_TURN_RAD_S") or 0.30)
# Arc-turn rate: yaw applied WHILE walking. At the standing-turn rate the
# arc is too tight ("turns much more than walking" -- operator, sim,
# 2026-07-21); walking + full yaw makes the heading win over travel. A
# separate, lower setpoint gives a good turning walk without giving up the
# brisk standing 360s.
_TEST_FIXED_ARC_TURN_RAD_S: float = float(
    os.environ.get("KPLANNER_FIXED_ARC_TURN_RAD_S") or 0.55)
# ---- Model-generated stop (2026-08-09 re-attempt of the 2026-08-03 revert).
# The immediate anchor blend halts the reference in 0.32 s with XY frozen at
# the stop instant. Above ~0.35 m/s the robot cannot null its momentum that
# fast: it overruns the frozen reference and collapses recovering (run), or
# hangs on its toes chasing a reference frozen AHEAD of it (walk lag).
# G1's stock stack has neither problem because its idle is a MODE: the model
# stays in the loop and plans the deceleration (demo/controllers.py:209).
# Re-attempt honors the revert note's guarantee: enter model-stop only as a
# STATE (zero-velocity replans, frames keep serving); ring starvation falls
# back to the immediate blend.
_MODEL_STOP: bool = os.environ.get(
    "KPLANNER_LEGACY_IMMEDIATE_STOP", "0") != "1"
_MODEL_STOP_MIN_MPS: float = float(
    os.environ.get("KPLANNER_MODEL_STOP_MIN_MPS") or 0.20)
_MODEL_STOP_SETTLED_MPS: float = float(
    os.environ.get("KPLANNER_MODEL_STOP_SETTLED_MPS") or 0.22)
_MODEL_STOP_TIMEOUT_S: float = float(
    os.environ.get("KPLANNER_MODEL_STOP_TIMEOUT_S") or 2.0)
# Below this, halved decel targets are pointless: slow_walk templates can't
# track small velocities (realized gait floors ~0.3 m/s), so a 0.2->0.1
# taper just keeps walking until the deadline (operator: "more than a meter
# of extra steps at 0.4"). Snap to the TRUE zero-velocity plan instead --
# the graph's actual stop content decelerates hard, and the starvation
# fallback covers its short plans.
_MODEL_STOP_ZERO_SNAP_MPS: float = float(
    os.environ.get("KPLANNER_MODEL_STOP_ZERO_SNAP_MPS") or 0.30)
# Foot-aligned stop (2026-08-09, operator requirement: "max one step to
# match the other" -- manipulation needs a bounded, precise stop, tighter
# than G1's learned 2-movement settle). Below the zero-snap speed the buffer
# ALREADY holds the future gait: scan it for the next L/R hip-pitch
# crossing -- the instant the swing foot passes the stance foot -- serve
# exactly to that frame and blend to stand from there. The swing foot sets
# down BESIDE the planted one: one movement, <= half a gait cycle of travel,
# zero replans (which also removes the buffered-frames flush delay).
# X2 DOF order: l_hip_pitch = 0, r_hip_pitch = 6 (convert_x2_record 73-74).
_STOP_FOOT_ALIGN: bool = os.environ.get(
    "KPLANNER_STOP_FOOT_ALIGN", "1") != "0"
# Approach slowdown (operator 2026-08-09: cut at full walk speed still
# overshoots -> transient toe-rise). Ease the buffer playback rate from 1.0
# down to this value as the read position approaches the cut: same
# trajectory, slower clock -- the closing step gentles out and the blend
# starts from ~rate x walk speed instead of full speed. 1.0 disables.
_STOP_ALIGN_SLOW_RATE: float = float(
    os.environ.get("KPLANNER_STOP_ALIGN_SLOW") or 0.35)
# Counter-brake pulse (operator-discovered 2026-08-09: "if i pull the stick
# back right before the final stop, it stops cleanly"). A pure decay never
# crosses zero -- it can only coast; a brief REVERSE-target plan actively
# cancels the remaining momentum (weight rocks back) before the foot-align
# cut. Magnitude reuses the stick's own back speed (the operator's exact,
# proven input); duration below. DEFAULT OFF (0) until the approach
# slowdown + XY momentum continuation are proven insufficient in the
# operator's sim test -- enable with KPLANNER_STOP_BRAKE_PULSE_S=0.35.
_STOP_BRAKE_PULSE_S: float = float(
    os.environ.get("KPLANNER_STOP_BRAKE_PULSE_S") or 0.0)
_HIP_PITCH_L_DOF: int = 0
_HIP_PITCH_R_DOF: int = 6
_ANKLE_PITCH_L_DOF: int = 4
_ANKLE_PITCH_R_DOF: int = 10
_KNEE_L_DOF: int = 3
_KNEE_R_DOF: int = 9
# Knee gate for the stop latch (2026-08-12): the ACTUAL tippy-toes
# mechanism is knee flexion at the cut (33-62 deg vs ~6 deg anchor)
# being extended over a fixed 0.32 s blend -> ~2-3 rad/s commanded
# extension -> vertical launch -> toe landing. Ankles were always
# neutral at cuts (a frames-tape column bug pointed at them).
_STOP_LATCH_KNEE_MAX_RAD: float = float(
    os.environ.get("KPLANNER_STOP_LATCH_KNEE_MAX_RAD") or 0.30)
# Ankle-neutral gate for the align cut (2026-08-12 tippy-toes regression
# with the OLD planner): the bare hip-pitch crossing is MID-SWING, and the
# old model's livelier gait carries 50-71 deg of toe-off plantarflexion at
# exactly that phase (tape 20260811_090002 stops 0/1/4/6) -- the stop blend
# then eased joints from a toe-extended pose and SONIC settled on its
# toes. The bigrun model's flat gait masked this because ITS crossing
# frames happened to be ankle-benign. Cut frames must now also have both
# ankles within this many rad of neutral (default ~17 deg).
_STOP_ALIGN_ANKLE_MAX_RAD: float = float(
    os.environ.get("KPLANNER_STOP_ALIGN_ANKLE_MAX_RAD") or 0.30)
# Worker-visible latch: while an align cut is armed, suppress cadence
# replans (a replan would swap the buffer out from under the cut point);
# forced replans (resume/cancel) still pass.
_STOP_ALIGN_ACTIVE: list = [False]


def _find_foot_align_cut(buf: "np.ndarray", from_pos: float) -> float:
    """Nearest native-frame index >= from_pos where the L/R hip-pitch
    difference changes sign (feet passing) AND both ankle pitches are
    near neutral (no toe-off at the cut -- see
    _STOP_ALIGN_ANKLE_MAX_RAD). Preference order:
      1. hip crossing with ankle-neutral frame at/adjacent to it;
      2. any ankle-neutral frame after the first crossing (the landing
         that follows the swing);
      3. bare hip crossing (legacy behaviour);
      4. last frame.
    """
    d = buf[:, 7 + _HIP_PITCH_L_DOF] - buf[:, 7 + _HIP_PITCH_R_DOF]
    la = np.abs(buf[:, 7 + _ANKLE_PITCH_L_DOF])
    ra = np.abs(buf[:, 7 + _ANKLE_PITCH_R_DOF])
    neutral = (la <= _STOP_ALIGN_ANKLE_MAX_RAD) & \
              (ra <= _STOP_ALIGN_ANKLE_MAX_RAD)
    start = max(1, int(math.ceil(from_pos)) + 1)
    first_cross = None
    for i in range(start, len(d)):
        if (d[i - 1] > 0.0) != (d[i] > 0.0):
            if first_cross is None:
                first_cross = i
            # accept the crossing if it (or a neighbour) is ankle-neutral
            for j in (i, i + 1, i - 1):
                if 0 <= j < len(d) and neutral[j]:
                    return float(j)
    if first_cross is not None:
        # no neutral crossing: take the first ankle-neutral frame AFTER
        # the crossing (the swing foot's landing) -- feet are adjacent
        # and planted there, which is the pose the align wants anyway.
        for j in range(first_cross, len(d)):
            if neutral[j]:
                return float(j)
        # Ankle drought (e.g. turn gait keeps ankles loaded for >1 s):
        # no neutral frame exists at all. Pick the LEAST-BAD frame after
        # the crossing instead of surrendering to the bare crossing.
        seg = np.maximum(la[first_cross:], ra[first_cross:])
        return float(first_cross + int(np.argmin(seg)))
    # NO hip crossing in the buffer at all -- arc/turn-walk keeps the
    # hip difference one-signed, and the legacy last-frame fallback cut
    # at an ARBITRARY pose (the actual source of the 55-66 deg toe-off
    # tail in the 400-stop replay; droughts were a red herring). Prefer
    # the LAST ankle-neutral frame (feet planted, close to the natural
    # end of served motion); else the least-bad frame.
    if start < len(d):
        neu_idx = np.where(neutral[start:])[0]
        if len(neu_idx):
            return float(start + int(neu_idx[-1]))
        seg = np.maximum(la[start:], ra[start:])
        return float(start + int(np.argmin(seg)))
    return float(len(d) - 1)


def _stop_blend_total_for(from_jpos, anchor_jpos) -> int:
    """Anchor-blend length scaled to the knee gap of the pose we blend
    FROM: STOP_BLEND_FRAMES (0.32 s) per 0.30 rad of worst-knee delta,
    clamped to [STOP_BLEND_FRAMES, 40] (0.32-0.80 s). Keeps commanded
    knee-extension rate roughly constant (~1 rad/s) regardless of how
    bent the cut pose is -- the constant-TIME blend was the tippy-toes
    launcher."""
    if from_jpos is None or anchor_jpos is None:
        return STOP_BLEND_FRAMES
    delta = max(
        abs(float(from_jpos[_KNEE_L_DOF]) - float(anchor_jpos[_KNEE_L_DOF])),
        abs(float(from_jpos[_KNEE_R_DOF]) - float(anchor_jpos[_KNEE_R_DOF])))
    # A crouch-mode exit blends from a DEEP cut pose (the G1-core planner's
    # crouch template serves 60-100 deg knees vs the 5 deg anchor). Under
    # the walking cap (0.8 s) that is a fast extension; the robot's 2026-09-09
    # crouch stops were clean with it, so it stays the default and the
    # rate-limited exit below is opt-in (KPLANNER_CROUCH_STAND_RATE_RAD_S).
    if delta <= _CROUCH_STAND_DELTA_RAD or CROUCH_STAND_RATE_RAD_S <= 0.0:
        return int(np.clip(round(STOP_BLEND_FRAMES * delta / 0.30),
                           STOP_BLEND_FRAMES, STOP_BLEND_MAX_FRAMES))
    # crouch exit: stand up at CROUCH_STAND_RATE_RAD_S of knee extension
    total = int(np.clip(round(delta / CROUCH_STAND_RATE_RAD_S * 50.0),
                        STOP_BLEND_FRAMES, CROUCH_STAND_BLEND_MAX_FRAMES))
    log.info("stop blend from a crouch: knee delta %.2f rad -> %d ticks (%.2f s at "
             "%.2f rad/s; KPLANNER_CROUCH_STAND_RATE_RAD_S, cap %d)",
             delta, total, total / 50.0, CROUCH_STAND_RATE_RAD_S,
             CROUCH_STAND_BLEND_MAX_FRAMES)
    return total
# Optional arc forward boost (default 1.0 = NO change). SONIC's obs carries
# no reference root translation -- forward progress is implied by gait
# joints while heading error is explicit -- so tracked arcs under-translate
# vs the reference (sim, 2026-07-21). A boost >1 over-commands forward
# during arcs to compensate AT THE INTENT LAYER. Deliberately opt-in: it is
# a tracker-era workaround that a better future model should not inherit.
_ARC_FWD_BOOST: float = float(
    os.environ.get("KPLANNER_ARC_FWD_BOOST") or 1.0)
# Arc turn rate follows the forward setpoint (constant turn radius) when the
# operator trims the speed up; KPLANNER_ARC_TURN_TRACKS_SPEED=0 restores the
# fixed rate. Cap keeps a trimmed-up walk from spinning faster than the
# in-place turn rate the gait was tuned for.
_ARC_TURN_TRACKS_SPEED: bool = os.environ.get("KPLANNER_ARC_TURN_TRACKS_SPEED", "1") == "1"
_ARC_TURN_MAX_RAD_S: float = float(os.environ.get("KPLANNER_ARC_TURN_MAX_RAD_S") or 1.2)
# Continuous yaw stick (2026-09-22, default ON; KPLANNER_YAW_PROPORTIONAL=0
# restores the fixed turn). The resolver used to be bang-bang on yaw: any
# deflection past the pad bridge's 0.15 deadzone commanded the FULL turn rate,
# so a slight turn was only possible by tapping (sim, frozen core at 0.35 m/s:
# stick 0.2 / 0.5 / 1.0 all -> -0.70 rad/s commanded, ~170 deg in 6 s
# executed). Now the deadzone is remapped to zero and the rate scales
# linearly from KPLANNER_YAW_PROP_MIN x full at the deadzone edge to the full
# rate at full deflection (walking arcs and in-place turns alike).
_YAW_PROPORTIONAL: bool = os.environ.get("KPLANNER_YAW_PROPORTIONAL", "1") == "1"
_YAW_STICK_DEADZONE: float = float(os.environ.get("KPLANNER_YAW_STICK_DEADZONE") or 0.15)
_YAW_PROP_MIN: float = float(os.environ.get("KPLANNER_YAW_PROP_MIN") or 0.25)
_FWD_LATERAL_DEADBAND: float = 0.35

# Strafe gating (2026-07-18). The resolver below is a SIGN function: any
# non-zero lateral component becomes a FULL _TEST_FIXED_SIDE_MPS strafe. With
# forward and lateral sharing one stick, a mild diagonal push therefore
# injected a full 0.3 m/s side-step into what the operator intended as a
# walking turn -- observed on hardware as intents (yaw=+0.3, vel_x=-0.3,
# vel_z=+0.3) and felt as "unclean steps". There is also no good side-step
# clip in the corpus, so an unintended strafe is worse than no strafe.
#
# The old magnitude-only rule (|side| < 0.35 while moving forward) was too
# permissive: a 45 deg push has |side| ~= 0.7 and sailed through. Replaced by
# an ANGLE rule -- lateral must dominate, i.e. the push must be close to pure
# sideways. Strafe engages only when BOTH hold:
#   * |side| >= _LATERAL_MIN_MAG        (a decisive push, not a lean)
#   * |fwd|  <= |side| * tan(theta_max) (within theta_max of the lateral axis)
# tan(25 deg) ~= 0.466, so a 45 deg diagonal (ratio 1.0) is rejected and only a
# near-90-degree push strafes. Raise _LATERAL_MAX_TAN to loosen.
_LATERAL_MIN_MAG: float = 0.60
_LATERAL_MAX_TAN: float = 0.466   # tan(25 deg)
# Forward/backward axis margin (2026-08-03, VR-band misfires): the fwd axis
# was BINARY — any nonzero tilt commanded the full speed setpoint, so a
# slight vertical lean during a strafe (or at rest) fired walk commands.
#   * |fwd| >= _FWD_MIN_MAG          (a deliberate push, not a lean)
#   * strafe active -> fwd forced 0  (strafing is exclusive; no diagonal creep)
_FWD_MIN_MAG: float = 0.20

_TURN_15_RAD_S: float = 0.5
_TURN_30_RAD_S: float = 1.0
_TURN_45_RAD_S: float = 1.5
_TURN_90_RAD_S: float = 3.0

# X2 stand hip height. Env override (2026-08-12) so sim sessions can command
# crouch-band intents (the wrapped G1-core planner routes hip < 0.60 to its
# crouch template); the deployed default is unchanged.
_HIP_HEIGHT_M: float = float(os.environ.get("KPLANNER_HIP_HEIGHT_M") or 0.687)
_IDLE_INTENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, _HIP_HEIGHT_M)

HOLD_TORSO_INTENT: str = "hold_torso"

# ---- CROUCH MODE (operator 2026-09-09: "L2+R2 + D-pad down -> crouch mode, D-pad up -> stand") ----------------
# The planner's own hip_h channel is dead, but a hip < 0.60 m on a LOCOMOTION command routes the wrapped G1-core
# planner to its crouch TEMPLATE, whose depth is its own (pelvis ~0.57 m executed, 10 cm below standing). Sim sweep
# 2026-09-09: >= 0.60 normal walk; 0.59 = the route seam, erratic; 0.55 consistent crouch walk. So the mode is a
# LATCH that stamps hip = _CROUCH_HIP_M on every locomotion target while on; idle stands up (the in-place
# hold_torso crouch is NOT used: below hip 0.597 it is a full-depth squat and walking out of it fell the sim robot).
# Safety: entry only when the robot is standing still (gate below), cleared by E-STOP and by a whole-body engage;
# a direct-velocity command with hip < 0.60 while the mode is OFF is clamped to the standing hip (belt and braces).
CROUCH_MODE_INTENT: str = "crouch_mode"
_CROUCH_HIP_M: float = float(os.environ.get("KPLANNER_CROUCH_HIP_M") or 0.55)
_CROUCH: dict = {"on": False, "since": 0.0, "ts": 0.0}
# Crouch-only sampler seed (2026-09-09, first robot crouch session): the template graph's random_seed is the
# segment-start PHASE of the mode clip (period 16), drawn fresh per replan by default; in the crouch band that made
# six walks six different walks (served pelvis 0.52-0.59, one shaky, some with pauses). KPLANNER_CROUCH_SEED pins
# the phase ONLY while crouch mode is ON (normal walking keeps the default draw / KPLANNER_FIXED_SEED). Unset = off.
_CROUCH_SEED: Optional[int] = (int(os.environ["KPLANNER_CROUCH_SEED"])
                               if os.environ.get("KPLANNER_CROUCH_SEED") not in (None, "") else None)
_CROUCH_KEEPALIVE_S: float = 0.6     # HOLD-TO-CROUCH (operator 2026-09-09 "safer this way"): the pad re-sends
                                     # enable=true while BOTH triggers stay held; no keepalive for this long -> OFF
_CROUCH_GATE_LEG_QD_RAD_S: float = 1.0     # every leg joint slower than this over the last ~0.4 s
_CROUCH_GATE_KNEE_MAX_RAD: float = math.radians(26.0)   # knees below this = standing (the policy stands at 12-15 deg; a crouch is > 40)
_CROUCH_GATE_QUIET_S: float = 1.5          # no non-zero locomotion target for this long
_CROUCH_LAST_MOVE: dict = {"ts": 0.0}
_CROUCH_CLAMP_LOGGED: dict = {"ts": 0.0}

# ---------------------------------------------------------------------------
# Waist overlay (v7.4 port from x2_kplanner). joint_pos_mj slot indices are
# the planner-side MuJoCo joint order (constants.py upstream); the slew rate
# matches the heuristic STATIC_HOLD path so operator feel is identical
# across planner backends. Self-contained on purpose: this file runs alone
# on PC2 with no gear_sonic package available.
WAIST_YAW_IDX: int = 12
WAIST_PITCH_IDX: int = 13
WAIST_ROLL_IDX: int = 14
HOLD_SLEW_DPS: float = 60.0
PRIMITIVE_INTENT: str = "primitive"

# Runtime forward-speed SETPOINT (m/s): any forward stick deflection past
# the deadzone commands exactly this speed. Nudged live via the payload's
# one-shot "speed_delta" field, clamped to [_SETPOINT_MIN, _SETPOINT_MAX].
_SETPOINT_MIN: float = float(
    os.environ.get("KPLANNER_SPEED_SETPOINT_MIN") or 0.2)
_SETPOINT_MAX: float = 1.0
_SPEED_SETPOINT: float = float(os.environ.get("KPLANNER_FIXED_FWD_MPS") or 0.30)

_DEFAULT_CONTINUOUS_FORWARD_MIN_MPS: float = float(
    os.environ.get("KPLANNER_CONTINUOUS_FWD_MIN_MPS") or 0.30)
_RUNTIME_CONTINUOUS_FORWARD_MIN_MPS: float = _DEFAULT_CONTINUOUS_FORWARD_MIN_MPS
_DEFAULT_STICK_SHAPING_EXPONENT: float = 1.0
_RUNTIME_STICK_SHAPING_EXPONENT: float = _DEFAULT_STICK_SHAPING_EXPONENT
_RUNTIME_TURN_LEFT_SCALE: float = 1.0
_RUNTIME_TURN_RIGHT_SCALE: float = 1.0
_RUNTIME_FORWARD_SCALE: float = 1.0
_RUNTIME_BACKWARD_SCALE: float = 1.0
_RUNTIME_LATERAL_SCALE: float = 1.0


# Nudge range (2026-08-12 D-pad speed control): live +-0.1 nudges move the
# setpoint within [NUDGE_MIN, NUDGE_MAX]; every locomotion stop resets it
# to the launch default. Distinct from _SETPOINT_MIN (the slow-walk floor
# machinery) on purpose -- the floor may sit at 0.2 while nudges stay in
# the demo-comfortable band.
_NUDGE_MIN: float = float(os.environ.get("KPLANNER_SPEED_NUDGE_MIN") or 0.3)
# 0.7 -> 0.5 (OPERATOR, hardware, 2026-08-22). The D-pad step is +0.2 from a
# 0.3 base, so the reachable rungs were 0.3 / 0.5 / 0.7. Only the FIRST press
# is stable on the robot -- "only one up increment performs well and others
# are unstable" -- so 0.7 was a rung the pad could always reach and the robot
# could not hold. Capping at 0.5 makes the stable rung the ceiling instead of
# the midpoint, which is the behaviour the demo actually wants: press up once,
# get the fast-but-safe gait, and further presses are a no-op rather than a
# fall. Raise deliberately with KPLANNER_SPEED_NUDGE_MAX once a faster gait is
# hardware-verified -- do NOT raise it because sim looked fine (see
# real_deploy_tuning/frozen_g1.yaml for what that costs).
_NUDGE_MAX: float = float(os.environ.get("KPLANNER_SPEED_NUDGE_MAX") or 0.5)

# Head-look overlay (2026-08-12 operator feature): with the deadman
# RELEASED, the pad's left-stick X streams head_yaw_stick; the daemon
# maps it to a slew-limited head-yaw joint overlay. ABSENT-FIELD-
# RECENTRE (the waist contract): any locomotion payload without the
# field zeroes the target, so the head eases home the moment driving
# resumes. Applied in the idle-anchor AND gait branches only -- dances
# and primitives keep their authored head motion.
_HEAD_YAW_DOF: int = 29
# Cap just under the physical head_yaw range (+-0.366 rad): commanding
# past the hard stop only winds up the PD against the limit, and the
# deploy's --max-target-dev-head clamp (0.40 in bypass mode) sits right
# above this.
_HEAD_YAW_MAX_RAD: float = float(
    os.environ.get("KPLANNER_HEAD_YAW_MAX_RAD") or 0.35)
_HEAD_YAW_SLEW_RAD_S: float = float(
    os.environ.get("KPLANNER_HEAD_YAW_SLEW_RAD_S") or 2.0)
_HEAD_YAW: dict = {"target": 0.0, "cur": 0.0}
# Absolute head-yaw feed from whole-body teleop (pico_intent_sender
# `head_targets`, the operator's SMPL neck+head, 2026-09-08). While it is
# fresh, locomotion payloads WITHOUT head_yaw_stick do not recentre the
# head -- the sender's own stick heartbeats carry no head field and would
# otherwise snap the gaze home 25 times a second. A stick deflection
# (head_yaw_stick non-zero) still wins outright.
_HEAD_SRC: dict = {"ts": 0.0}
_HEAD_SRC_TIMEOUT_S: float = 1.0


def _apply_head_yaw(jpos: "np.ndarray", advance: bool = True) -> "np.ndarray":
    """Overlay the slewed head-yaw target on a served frame (copy-on-
    write; returns the input untouched when centred)."""
    if advance:
        step = _HEAD_YAW_SLEW_RAD_S / OUTPUT_FPS
        d = _HEAD_YAW["target"] - _HEAD_YAW["cur"]
        if abs(d) > 1e-9:
            _HEAD_YAW["cur"] += max(-step, min(step, d))
    if abs(_HEAD_YAW["cur"]) < 1e-4:
        return jpos
    out = jpos.copy()
    out[_HEAD_YAW_DOF] = _HEAD_YAW["cur"]
    return out
_DEFAULT_SPEED_SETPOINT: float = _SPEED_SETPOINT


def _adjust_speed_setpoint(delta: float) -> float:
    """Nudge the runtime forward-speed setpoint; returns the new value."""
    global _SPEED_SETPOINT
    _SPEED_SETPOINT = max(_NUDGE_MIN, min(_NUDGE_MAX, _SPEED_SETPOINT + delta))
    log.info("speed setpoint %+0.1f -> %.1f m/s", delta, _SPEED_SETPOINT)
    return _SPEED_SETPOINT


def _reset_speed_setpoint() -> None:
    """Back to the launch default on every locomotion stop (operator
    spec: stick to neutral forgets the nudges)."""
    global _SPEED_SETPOINT
    if abs(_SPEED_SETPOINT - _DEFAULT_SPEED_SETPOINT) > 1e-6:
        log.info("speed setpoint reset %.1f -> %.1f m/s (stop)",
                 _SPEED_SETPOINT, _DEFAULT_SPEED_SETPOINT)
        _SPEED_SETPOINT = _DEFAULT_SPEED_SETPOINT


def _shape_stick(value: float) -> float:
    """sign(v) * |v|**exp -- port of x2_kplanner._shape_stick."""
    sign = 1.0 if value >= 0 else -1.0
    mag = abs(float(value))
    if mag == 0.0:
        return 0.0
    return sign * mag ** _RUNTIME_STICK_SHAPING_EXPONENT


def _resolve_locomotion_continuous(
    stick_fwd: float, stick_side: float, stick_yaw: float
) -> tuple[float, float, float, float]:
    """Continuous VR/pad stick resolver (port of x2_kplanner's)."""
    shaped_fwd = _shape_stick(stick_fwd)
    shaped_side = _shape_stick(stick_side)
    shaped_yaw = _shape_stick(stick_yaw)

    # Strafe requires a near-pure sideways push (see _LATERAL_MIN_MAG /
    # _LATERAL_MAX_TAN). Applies in BOTH travel directions -- the old rule only
    # gated lateral while moving forward, so a diagonal pull-back still strafed.
    if shaped_side != 0.0:
        if (abs(shaped_side) < _LATERAL_MIN_MAG
                or abs(shaped_fwd) > abs(shaped_side) * _LATERAL_MAX_TAN):
            shaped_side = 0.0

    # Fwd/back margin: deadzone a lean; and while a strafe is active the
    # fwd axis is suppressed entirely (exclusive strafing).
    if abs(shaped_fwd) < _FWD_MIN_MAG or shaped_side != 0.0:
        shaped_fwd = 0.0

    if shaped_fwd > 0.0:
        vel_z = _SPEED_SETPOINT           # deterministic setpoint mode
    elif shaped_fwd < 0.0:
        vel_z = -_TEST_FIXED_BACK_MPS
    else:
        vel_z = 0.0
    if shaped_side > 0.0:
        vel_x = -_TEST_FIXED_SIDE_MPS     # stick right -> side_right -> -vel_x
    elif shaped_side < 0.0:
        vel_x = _TEST_FIXED_SIDE_MPS
    else:
        vel_x = 0.0
    turn_mag = (_TEST_FIXED_ARC_TURN_RAD_S if vel_z != 0.0
                else _TEST_FIXED_TURN_RAD_S)
    if vel_z > 0.0 and _ARC_TURN_TRACKS_SPEED and _DEFAULT_SPEED_SETPOINT > 0.0:
        # Constant turn RADIUS across the operator's speed trim (2026-09-05,
        # operator: "the robot doesn't turn much when walking at 0.5-0.6"):
        # the X/Y trim nudges _SPEED_SETPOINT but the arc turn rate was a
        # fixed constant, so radius = v / omega grew with speed (0.35/0.70 =
        # 0.5 m at the default, 0.86 m at 0.6 m/s). Scale omega with v,
        # capped at KPLANNER_ARC_TURN_MAX_RAD_S.
        turn_mag = min(_ARC_TURN_MAX_RAD_S,
                       turn_mag * max(1.0, vel_z / _DEFAULT_SPEED_SETPOINT))
    if _YAW_PROPORTIONAL and shaped_yaw != 0.0:
        # deadzone edge -> floor x full, full deflection -> full rate
        u = (min(1.0, abs(shaped_yaw)) - _YAW_STICK_DEADZONE) / max(1e-6, 1.0 - _YAW_STICK_DEADZONE)
        turn_mag *= _YAW_PROP_MIN + (1.0 - _YAW_PROP_MIN) * max(0.0, u)
    if shaped_yaw > 0.0:
        yaw_rate = -turn_mag              # stick right -> turn-right -> -yaw
    elif shaped_yaw < 0.0:
        yaw_rate = turn_mag
    else:
        yaw_rate = 0.0
    if yaw_rate != 0.0 and vel_z > 0.0 and _ARC_FWD_BOOST != 1.0:
        vel_z = vel_z * _ARC_FWD_BOOST
    return (yaw_rate, vel_x, vel_z, _HIP_HEIGHT_M)


# Bucketed (legacy) table -- kept for scripted / manager vocabulary parity.
_BASE_VELOCITY: dict[str, tuple[float, float, float, float]] = {
    "idle":       (0.0,             0.0,              0.0,             _HIP_HEIGHT_M),
    "fwd_step":   (0.0,             0.0,              _WALK_SPEED_MPS, _HIP_HEIGHT_M),
    "back_step":  (0.0,             0.0,             -_BACK_SPEED_MPS, _HIP_HEIGHT_M),
    "side_left":  (0.0,             _SIDE_SPEED_MPS,  0.0,             _HIP_HEIGHT_M),
    "side_right": (0.0,            -_SIDE_SPEED_MPS,  0.0,             _HIP_HEIGHT_M),
    "turn_left":  ( _TURN_45_RAD_S, 0.0,              0.0,             _HIP_HEIGHT_M),
    "turn_right": (-_TURN_45_RAD_S, 0.0,              0.0,             _HIP_HEIGHT_M),
}
_TRANSLATIONAL_SCALE: dict[str, float] = {
    "default": 1.0, "stand": 0.0, "quarter_ft": 0.5, "half_ft": 1.0, "one_ft": 1.5,
}
_TURN_SCALE: dict[str, float] = {
    "default": 1.0,
    "deg_15": _TURN_15_RAD_S / _TURN_45_RAD_S,
    "deg_30": _TURN_30_RAD_S / _TURN_45_RAD_S,
    "deg_45": 1.0,
    "deg_90": _TURN_90_RAD_S / _TURN_45_RAD_S,
}
_ROTATIONAL_INTENTS: frozenset[str] = frozenset({"turn_left", "turn_right"})
_WALK_VELOCITY_BY_MAGNITUDE: dict[str, tuple[float, float, float, float]] = {
    "forward":  (0.0, 0.0,  _WALK_SPEED_MPS,      _HIP_HEIGHT_M),
    "backward": (0.0, 0.0, -_BACK_SPEED_MPS,      _HIP_HEIGHT_M),
    "fast":     (0.0, 0.0,  _FAST_WALK_SPEED_MPS, _HIP_HEIGHT_M),
}


@dataclass(frozen=True)
class LocomotionCommand:
    """Slim local mirror of state_machine.LocomotionCommand (fields we use)."""

    intent: str
    magnitude: str = "default"
    source: str = "zmq"
    stick_fwd: float = 0.0
    stick_side: float = 0.0
    stick_yaw: float = 0.0
    direct_velocity: Optional[tuple[float, float, float, float]] = None
    hip_height_m: Optional[float] = None


def _resolve_velocity(intent: str, magnitude: str) -> tuple[float, float, float, float]:
    if intent == "walk":
        return _WALK_VELOCITY_BY_MAGNITUDE.get(magnitude, _IDLE_INTENT)
    base = _BASE_VELOCITY.get(intent)
    if base is None:
        return _IDLE_INTENT
    yaw, vx, vy, hip_h = base
    if intent in _ROTATIONAL_INTENTS:
        return (yaw * _TURN_SCALE.get(magnitude, 1.0), vx, vy, hip_h)
    scale = _TRANSLATIONAL_SCALE.get(magnitude, 1.0)
    return (yaw, vx * scale, vy * scale, hip_h)


def _apply_runtime_scales(
    intent: str, velocity: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    yaw, vel_x, vel_z, hip_h = velocity
    if intent == "turn_left":
        yaw *= _RUNTIME_TURN_LEFT_SCALE
    elif intent == "turn_right":
        yaw *= _RUNTIME_TURN_RIGHT_SCALE
    elif intent in ("fwd_step",) or (intent == "walk" and vel_z > 0):
        vel_z *= _RUNTIME_FORWARD_SCALE
    elif intent in ("back_step",) or (intent == "walk" and vel_z < 0):
        vel_z *= _RUNTIME_BACKWARD_SCALE
    elif intent in ("side_left", "side_right"):
        vel_x *= _RUNTIME_LATERAL_SCALE
    return (yaw, vel_x, vel_z, hip_h)


def _apply_continuous_runtime_scales(
    velocity: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    yaw, vx, vz, hip_h = velocity
    if yaw > 0:
        yaw *= _RUNTIME_TURN_LEFT_SCALE
    elif yaw < 0:
        yaw *= _RUNTIME_TURN_RIGHT_SCALE
    if vz > 0:
        vz *= _RUNTIME_FORWARD_SCALE
        if _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS > 0.0:
            vz = max(vz, _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS)
    elif vz < 0:
        vz *= _RUNTIME_BACKWARD_SCALE
    vx *= _RUNTIME_LATERAL_SCALE
    return (yaw, vx, vz, hip_h)


# ---- forward-obstacle guard -------------------------------------------------
# Subscribes to scan_guard_pub.py over ZMQ rather than importing rclpy: this
# process runs in the gear_sonic venv, which has no ROS bindings, and the gait
# loop is the last place to add an import that can fail.
#
# Clamped HERE (planner_cmd ingest) rather than in the pad bridge so every
# input source is covered -- pad, Quest/VR, anything else that publishes to
# this socket.
#
# FAIL-OPEN: stale or absent guard data means no clamping. Freezing a walking
# biped because a sensor process died is worse than not clamping; the operator
# deadman is the real stop.
_GUARD_PORT = 5571
_GUARD_STALE_S = 0.5
_guard_state = {"blocked": False, "dist": float("inf"), "ts": 0.0}
# Latched stop: once an obstacle trips the guard, forward stays dead until
# the operator releases the deadman (all sticks zero). Auto-release on a
# clear path would let the robot resume walking without a human deciding to.
_guard_latched = {"on": False}

# ---- command-source ownership (pad vs VR mutual exclusion) ------------------
# Enforced in _zmq_command_thread. VR supersedes pad while engaged; release
# is explicit (VR idle on disengage) with a crash-safety silence timeout.
# The VR manager keepalives every 0.5 s while engaged, so 2.0 s can only
# expire when the manager is actually gone (crash / network loss).
_VR_OWNER_TIMEOUT_S = 2.0
_cmd_owner = {"src": None, "vr_ts": 0.0}

# ---- whole-body (Pico token) release stability -----------------------------
# While the deploy's token graph has authority the planner's pose-ref is
# ignored, so on ANY release (B, wifi drop, service crash) the deploy used
# to ramp 2 s toward this planner's CONSTANT idle anchor = a commanded
# stand-up from wherever the operator left the robot (deep crouch, 09-03
# incident). Now the idle anchor tick runs a small state machine:
#   off    -> normal anchor behaviour (feature disabled or never engaged)
#   shadow -> whole-body ENGAGED (token service keepalives on planner_cmd,
#             intent "wb_state"): the served reference SHADOWS the measured
#             joints (x2_debug body_q) so the wire is continuous at release
#   hold   -> link lost / operator released: FREEZE at the last shadowed
#             pose (the policy keeps balancing there; nothing moves), arms
#             + hands latched via the arm overlay
#   return -> first pad locomotion intent (operator decision: "once we get
#             commands from the gamepad we switch to the kplanner"): a
#             half-cosine, per-joint RATE-LIMITED blend to the anchor
#             (knees/hips/ankles <= WB_RETURN_LEG_RAD_S, waist <=
#             WB_RETURN_WAIST_RAD_S); the pad intent is deferred until the
#             blend completes; arms stay latched until an explicit
#             "arm_release" intent or the next whole-body engage.
# Enabled by KPLANNER_WB_RETURN=1 (sim stack sets it; robot_env.env opts
# the robot in after the operator feel-test).
_WB_ENABLED: bool = os.environ.get("KPLANNER_WB_RETURN", "0") == "1"
# Hold the LEGS/WAIST at the measured pose after a release (operator's
# option 1)? Sim 2026-09-04: the pose graph does not hold a static
# off-anchor stance -- it stood the twin up from a crouch in 2 s in the
# harness, and in the operator's session a B release from an idle-crept
# stand (ankle -12 deg) tipped it in 1.5 s -- while releases whose goal was
# the ANCHOR survived. A real leg hold needs the deploy to keep the token
# graph balancing (layer A, C++). Until then the default release = arms +
# wrists + head latched, legs + waist on the rate-limited RETURN at once.
_WB_HOLD_LEGS: bool = os.environ.get("KPLANNER_WB_HOLD_LEGS", "0") == "1"
# TODO(wb-latched-gait, 2026-09-05): walking WITH the arms held (carrying a
# manipulation pose) is a policy capability gap, not a kplanner problem --
# the walking policy needs gait references with pinned/held arms in its
# training corpus (kplanner served frames from robot sessions are recorded
# in log/kplanner_tape/ for exactly that). Until then the latch is released
# at gait start (below).
# KPLANNER_WB_ARM_LATCH_GAIT (2026-09-05): what the WB arm latch does when a
# locomotion GAIT starts. "release" (default) slews the arms back to the
# planner's own gait arms; "keep" pins them through the walk. Robot session
# 2026-09-05 09:16-09:18: walking with the arms latched forward skipped 56%
# of the steady steps (5/9) vs 2% (1/45) with the planner's arms -- the
# walking policy never trained on a gait reference with pinned arms. The
# latch still covers standing, HOLD and the rate-limited RETURN.
_WB_ARM_LATCH_GAIT: str = os.environ.get("KPLANNER_WB_ARM_LATCH_GAIT", "release")
_WB_LINK_TIMEOUT_S: float = 1.0        # token-service keepalive is 0.5 s
_WB_SHADOW_STEP_RAD: float = 0.06      # 3 rad/s @50 Hz, same as arm slew
_WB_RETURN_LEG_RAD_S: float = float(os.environ.get("WB_RETURN_LEG_RAD_S", "1.0"))
_WB_RETURN_WAIST_RAD_S: float = float(os.environ.get("WB_RETURN_WAIST_RAD_S", "0.5"))

# KPLANNER_HAND_DEFAULT (2026-09-05, operator: "on kplanner start, always start
# with omni hand fingers closed if we have a hand"): the OmniHand pose the
# planner puts on the wire when nothing else drives the hands -- "closed"
# (default; the sender's own rest state is closed too, squeeze = open) or
# "open" (the pre-09-05 zeros). Targets come from the same grasp table the
# sender uses (x2_hand_retarget, on PC2 under planner_stack); if that import
# is unavailable the wire falls back to zeros (open) with a warning.
_HAND_DEFAULT_MODE: str = os.environ.get("KPLANNER_HAND_DEFAULT", "closed")
_HAND_DEFAULT_Q: dict[str, "np.ndarray"] = {}


def _hand_default(side: str, hand_dof: int) -> "np.ndarray":
    q = _HAND_DEFAULT_Q.get(side)
    if q is not None and q.shape == (hand_dof,):
        return q.copy()
    q = np.zeros(hand_dof, dtype=np.float32)
    if _HAND_DEFAULT_MODE == "closed":
        try:
            from gear_sonic.utils.teleop.x2_hand_retarget import grasp_command_from_ratio
            cand = np.asarray(grasp_command_from_ratio(side, 1.0), dtype=np.float32)   # ratio 1 = closed
            if cand.shape == (hand_dof,):
                q = cand
            else:
                log.warning("hand default: grasp table has %d dof, wire expects %d -> open", cand.shape[0], hand_dof)
        except Exception as exc:  # planner_stack without the hand module
            log.warning("hand default: cannot build the CLOSED pose (%r) -> hands start OPEN", exc)
    _HAND_DEFAULT_Q[side] = q
    if side == "right":
        log.info("hand default on the wire: %s (KPLANNER_HAND_DEFAULT=%s)",
                 "CLOSED" if np.any(q) else "open/zeros", _HAND_DEFAULT_MODE)
    return q.copy()
_WB_RETURN_MIN_TICKS: int = 16          # 0.32 s floor (= STOP_BLEND_FRAMES)
_WB_RETURN_PREROLL_TICKS: int = int(os.environ.get("WB_RETURN_PREROLL_TICKS", "12"))  # 0.24 s
_WB_RETURN_MAX_TICKS: int = 200         # 4.0 s cap
_WB_LINK = {"engaged": False, "ts": 0.0}
_WB = {"mode": "off", "hold_jpos": None, "pending_cmd": None,
       "return_requested": False}
_WB_LEG_DOFS = slice(0, 12)             # MJ order: L leg 0-5, R leg 6-11
_WB_WAIST_DOFS = slice(12, 15)
_WB_HEAD_DOFS = slice(29, 31)


def _wb_return_total_for(from_jpos, anchor_jpos) -> int:
    """Return-blend length in output ticks so that NO leg/waist/head joint
    exceeds its rate limit over the half-cosine (peak rate of a half-cosine
    is pi/2 x the mean, folded in). Arms are excluded: they stay latched."""
    if from_jpos is None or anchor_jpos is None:
        return _WB_RETURN_MIN_TICKS
    d = np.abs(np.asarray(from_jpos, dtype=np.float64)
               - np.asarray(anchor_jpos, dtype=np.float64))
    t_leg = float(d[_WB_LEG_DOFS].max()) / _WB_RETURN_LEG_RAD_S
    t_waist = float(d[_WB_WAIST_DOFS].max()) / _WB_RETURN_WAIST_RAD_S
    t_head = float(d[_WB_HEAD_DOFS].max()) / _WB_RETURN_LEG_RAD_S
    t = max(t_leg, t_waist, t_head) * (math.pi / 2.0)
    return int(np.clip(math.ceil(t * OUTPUT_FPS),
                       _WB_RETURN_MIN_TICKS, _WB_RETURN_MAX_TICKS))
# Set by main() when --arm-port is active; ownership release clears it.
_ARM_INGEST_REF: list = [None]


def _scan_guard_thread(stop_event) -> None:
    import zmq as _zmq
    ctx = _zmq.Context.instance()
    sock = ctx.socket(_zmq.SUB)
    sock.setsockopt(_zmq.SUBSCRIBE, b"scan_guard")
    sock.setsockopt(_zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://127.0.0.1:{_GUARD_PORT}")
    log.info("scan guard: SUB tcp://127.0.0.1:%d", _GUARD_PORT)
    while not stop_event.is_set():
        try:
            _, payload = sock.recv_multipart()
        except Exception:  # noqa: BLE001 -- timeout is normal
            continue
        try:
            d = json.loads(payload)
            _guard_state["blocked"] = bool(d["blocked"])
            _guard_state["dist"] = float(d["dist"])
            _guard_state["ts"] = time.monotonic()
        except Exception:  # noqa: BLE001
            continue
    sock.close(linger=0)


def _guard_blocked() -> bool:
    if time.monotonic() - _guard_state["ts"] > _GUARD_STALE_S:
        return False
    return _guard_state["blocked"]


def intent_to_velocity(cmd: LocomotionCommand) -> tuple[float, float, float, float]:
    """LocomotionCommand -> 4-D velocity, then the CROUCH MODE hip policy (see _CROUCH)."""
    yaw, vx, vz, hip_h = _intent_to_velocity_raw(cmd)
    moving = (abs(vx) + abs(vz) + abs(yaw)) > 0.02
    if moving:
        _CROUCH_LAST_MOVE["ts"] = time.monotonic()
    if cmd.intent in ("idle", HOLD_TORSO_INTENT):
        return (yaw, vx, vz, hip_h)
    if _CROUCH["on"]:
        # Crouched WALKING only (operator 2026-09-09, sim fall): a zero-stick target at the crouch hip is the
        # template's idle = a deep squat (served pelvis 0.42-0.46 m) from which no walk starts and a hard push
        # tips the robot -> standing still keeps the STANDING hip. Forward + turn only until backward/lateral
        # crouched gaits are tested; backward and lateral components are zeroed (logged, rate-limited).
        if not moving:
            return (yaw, vx, vz, float(_HIP_HEIGHT_M))
        vx_c, vz_c = 0.0, max(0.0, vz)
        if (abs(vx) > 0.02 or vz < -0.02) and time.monotonic() - _CROUCH_CLAMP_LOGGED["ts"] > 3.0:
            _CROUCH_CLAMP_LOGGED["ts"] = time.monotonic()
            log.warning("crouch mode: backward/lateral stick ignored (vx %.2f vz %.2f) -- forward + turn only", vx, vz)
        if abs(vz_c) + abs(yaw) < 0.02:
            return (yaw, 0.0, 0.0, float(_HIP_HEIGHT_M))
        return (yaw, vx_c, vz_c, float(_CROUCH_HIP_M))
    if hip_h < 0.60 and cmd.direct_velocity is not None:
        if time.monotonic() - _CROUCH_CLAMP_LOGGED["ts"] > 5.0:
            _CROUCH_CLAMP_LOGGED["ts"] = time.monotonic()
            log.warning("crouch mode OFF: direct-velocity hip %.3f < 0.60 clamped to the standing hip %.3f "
                        "(enable the mode first)", hip_h, _HIP_HEIGHT_M)
        return (yaw, vx, vz, float(_HIP_HEIGHT_M))
    return (yaw, vx, vz, hip_h)


def _intent_to_velocity_raw(cmd: LocomotionCommand) -> tuple[float, float, float, float]:
    """LocomotionCommand -> 4-D velocity; port of x2_kplanner.intent_to_velocity."""
    if cmd.direct_velocity is not None:
        yaw, vx, vz, hip_h = cmd.direct_velocity
        return (float(yaw), float(vx), float(vz), float(hip_h))
    if cmd.intent == "locomotion" and cmd.magnitude == "continuous":
        result = _resolve_locomotion_continuous(
            cmd.stick_fwd, cmd.stick_side, cmd.stick_yaw
        )
        return _apply_continuous_runtime_scales(result)
    if cmd.intent == HOLD_TORSO_INTENT:
        yaw_idle, vx_idle, vz_idle, hip_idle = _IDLE_INTENT
        hip_h = (
            float(cmd.hip_height_m) if cmd.hip_height_m is not None
            else float(hip_idle)
        )
        return (float(yaw_idle), float(vx_idle), float(vz_idle), hip_h)
    result = _resolve_velocity(cmd.intent, cmd.magnitude)
    if result == _IDLE_INTENT and cmd.intent != "idle":
        log.debug("intent %s,%s has no velocity mapping; idling",
                  cmd.intent, cmd.magnitude)
        return result
    return _apply_runtime_scales(cmd.intent, result)


# ---------------------------------------------------------------------------
# Process hygiene (port of x2_kplanner's).
# ---------------------------------------------------------------------------


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
    except OSError:
        return True
    finally:
        s.close()
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PidFile:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> "PidFile":
        if self.path.exists():
            try:
                old = int(self.path.read_text().strip())
                if _pid_alive(old):
                    raise RuntimeError(
                        f"PID file {self.path} exists and PID {old} is alive -- "
                        f"another planner is running. `kill {old}` first."
                    )
                log.warning("stale PID file %s for dead PID %d, cleaning up",
                            self.path, old)
            except ValueError:
                log.warning("bad PID file %s, cleaning up", self.path)
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Warmup / idle anchor qpos loading (port of x2_kplanner._load_warmup_qpos).
# ---------------------------------------------------------------------------

_TRAINING_DEFAULT_ANGLES: np.ndarray = np.array((
    -0.312, 0.0,  0.0,  0.669, -0.363, 0.0,    # left  leg (6)
    -0.312, 0.0,  0.0,  0.669, -0.363, 0.0,    # right leg (6)
     0.0,  0.0,  0.0,                          # waist     (3)
     0.2,  0.2,  0.0, -0.6,  0.0,  0.0,  0.0,  # left  arm (7)
     0.2, -0.2,  0.0, -0.6,  0.0,  0.0,  0.0,  # right arm (7)
     0.0,  0.0,                                # head      (2)
), dtype=np.float32)
_TRAINING_DEFAULT_HIP_Z: float = 0.636


def _build_training_default_qpos() -> np.ndarray:
    qpos = np.zeros(QPOS_DIM, dtype=np.float32)
    qpos[2] = _TRAINING_DEFAULT_HIP_Z
    qpos[3] = 1.0  # wxyz identity
    qpos[7:38] = _TRAINING_DEFAULT_ANGLES
    return qpos


def _qpos_from_deploy_pkl_frame(obj_inner: dict) -> Optional[np.ndarray]:
    """Deploy-PKL schema frame 0 -> qpos[38] wxyz (root_rot stored xyzw)."""
    needed = {"root_trans_offset", "root_rot", "dof"}
    if not needed.issubset(obj_inner):
        return None
    root_trans = np.asarray(obj_inner["root_trans_offset"])
    root_rot = np.asarray(obj_inner["root_rot"])  # xyzw
    dof = np.asarray(obj_inner["dof"])
    f0_trans = root_trans[0] if root_trans.ndim == 2 else root_trans
    f0_rot_xyzw = root_rot[0] if root_rot.ndim == 2 else root_rot
    f0_dof = dof[0] if dof.ndim == 2 else dof
    if f0_trans.shape != (3,) or f0_rot_xyzw.shape != (4,) or f0_dof.shape != (31,):
        return None
    qpos = np.zeros(QPOS_DIM, dtype=np.float32)
    qpos[0:3] = f0_trans
    qpos[3:7] = [f0_rot_xyzw[3], f0_rot_xyzw[0], f0_rot_xyzw[1], f0_rot_xyzw[2]]
    qpos[7:38] = f0_dof
    return qpos


def _load_warmup_qpos(path: Optional[Path]) -> np.ndarray:
    """Load a [38]-D anchor qpos from a (joblib) PKL, with the same schema
    tolerance as x2_kplanner; falls back to training_default_angles."""
    if path is None:
        qpos = _build_training_default_qpos()
        log.info("warmup anchor: training_default_angles stand (hip_z=%.3fm)",
                 float(qpos[2]))
        return qpos
    if not path.is_file():
        log.warning("warmup-qpos %s not found; falling back to training default",
                    path)
        return _build_training_default_qpos()
    try:
        import joblib
        obj = joblib.load(path)
    except Exception:  # noqa: BLE001 -- fall through to raw pickle
        with path.open("rb") as f:
            obj = pickle.load(f)
    if isinstance(obj, dict):
        arr = obj.get("mujoco_qpos", obj.get("qpos", None))
        if arr is None:
            if len(obj) >= 1:
                inner = next(iter(obj.values()))
                if isinstance(inner, dict):
                    qpos = _qpos_from_deploy_pkl_frame(inner)
                    if qpos is not None:
                        log.info(
                            "warmup anchor: loaded from %s "
                            "(deploy-PKL schema, frame 0, hip_z=%.3fm)",
                            path, float(qpos[2]),
                        )
                        return qpos
            raise ValueError(
                f"warmup PKL {path} has no recognisable qpos / deploy-PKL "
                f"schema (keys={list(obj.keys())})"
            )
    else:
        arr = obj
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[-1] != QPOS_DIM:
        raise ValueError(f"warmup PKL {path} qpos shape {arr.shape} != [38]")
    log.info("warmup anchor: loaded from %s (shape=[38])", path)
    return arr


# ---------------------------------------------------------------------------
# Replan backends. Both implement the exact NeuralPlannerCore ring-buffer
# semantics x2_kplanner relies on:
#   * reset(qpos)      -> buffer := 64 tiled copies, cursor := 0
#   * get_next_frame() -> pop buf[clamp(cursor)], cursor := min(cursor+1, T-1)
#   * context          -> buf[[clamp(cursor + i) for i in 0..3]]
#                         (idx - NUM_FT + i + PRED_OFFSETS with the default
#                          PRED_OFFSETS == NUM_FT == 4)
#   * replan(target)   -> buffer := prediction[:num_pred_frames], cursor := 0
#                         (NUM_FT - PRED_OFFSETS)
#   * should_replan()  -> frames_remaining <= threshold
# ---------------------------------------------------------------------------

NUM_FRAMES_PER_TOKEN: int = 4
PRED_OFFSETS: int = 4
NUM_MIN_FRAMES_IN_BUFFER: int = 64

_PLANNER_MODE_NAMES: tuple[str, ...] = ("idle", "slow_walk", "walk", "run_proxy")

# Default ONNX runtime contract -- mirrors the interface documented in
# motionbricks/scripts/export_x2_planner_onnx.py. Every key can be
# overridden by a JSON sidecar next to the graph so a re-export with
# different tensor names / extra conditioning tensors needs no code change.
_DEFAULT_ONNX_CONTRACT: dict = {
    # role -> graph input tensor name
    "inputs": {
        "context": "context_mujoco_qpos",     # f32 [1, 4, 38]
        "velocity": "velocity_intent",        # f32 [1, 4]
        "mode": "mode",                       # i64 [1]   (template graph)
        "random_seed": "random_seed",         # i64 [1]   (template graph)
    },
    # role -> graph output tensor name
    "outputs": {
        "qpos": "mujoco_qpos",                # f32 [1, 64, 38]
        "num_pred_frames": "num_pred_frames", # i64 [1]
    },
    # The export bakes the FILTER_QPOS first-4-frame context blend
    # in-graph (it lives inside NeuralPlannerCore._predict_* which the
    # trace covers); x2_kplanner never applies it Python-side. Set false
    # in the sidecar only for a graph exported without the blend.
    "filter_qpos_baked": True,
    # mode-name -> index binding (build_x2_planner_clips DEFAULT_MODES).
    "modes": list(_PLANNER_MODE_NAMES),
    # Optional: dense constant extra inputs {tensor_name: {"dtype": "i64"
    # or "f32", "value": [...], "shape": [...]}} fed verbatim per replan.
    "extra_inputs": {},
    # Optional: fixed random seed (int) instead of per-replan random.
    "fixed_random_seed": None,
    # Optional: "template" / "velocity"; default auto-detect from the
    # session's input names (template iff the mode input exists).
    "graph_kind": None,
    # Optional default mode name for template graphs when --planner-mode
    # is not passed.
    "default_mode": "walk",
}


def _load_onnx_contract(onnx_path: Path, sidecar: Optional[Path]) -> dict:
    """Merge the JSON sidecar (if any) over the default contract.

    Search order when --onnx-sidecar is not given: ``<onnx>.json`` then
    ``<dir>/runtime_contract.json``. Missing sidecar -> pure defaults
    (matches the current export script's tensor names).
    """
    contract = json.loads(json.dumps(_DEFAULT_ONNX_CONTRACT))  # deep copy
    candidates = (
        [sidecar] if sidecar is not None
        else [onnx_path.with_suffix(onnx_path.suffix + ".json"),
              onnx_path.parent / "runtime_contract.json"]
    )
    for cand in candidates:
        if cand is not None and Path(cand).is_file():
            raw = json.loads(Path(cand).read_text())
            for key, val in raw.items():
                if key in ("inputs", "outputs") and isinstance(val, dict):
                    contract[key].update(val)
                else:
                    contract[key] = val
            log.info("onnx contract: merged sidecar %s", cand)
            return contract
    if sidecar is not None:
        raise FileNotFoundError(f"--onnx-sidecar {sidecar} not found")
    log.info("onnx contract: no sidecar found; using export-script defaults")
    return contract


class OnnxPlannerBackend:
    """Ring buffer + onnxruntime fused-graph replan (torch-free)."""

    # Flipped by --ort-gpu / --ort-trt in main(). Class-level so the flags
    # set once at startup reach every backend instance without threading.
    USE_GPU: bool = False
    USE_TRT: bool = False

    def __init__(
        self,
        onnx_path: Path,
        contract: dict,
        replan_threshold_frames: int,
        planner_mode: Optional[str],
    ) -> None:
        import onnxruntime as ort

        # Provider order: GPU (CUDA) first with CPU fallback when --ort-gpu is
        # set, else CPU-only (the safe default that has always shipped). ORT
        # silently drops any provider not present in the build, so on a CPU-only
        # onnxruntime this still runs on CPU -- the flag is a no-op until a
        # Jetson GPU build of onnxruntime is installed in the venv.
        # NOTE: CUDA kernels may sample differently from CPU for the same
        # random_seed. Fine for deploy; but an intent-tape replay must use the
        # SAME provider it was recorded under to stay bit-exact.
        providers: list = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                           if OnnxPlannerBackend.USE_GPU
                           else ["CPUExecutionProvider"])
        if OnnxPlannerBackend.USE_TRT:
            # TensorRT EP ahead of CUDA (2026-08-11 latency work: Jetson
            # ORT-GPU replans at p50 75 ms; TRT targets lower + less GPU
            # contention with the deploy's own 50 Hz inference). Engine
            # cache is MANDATORY on Jetson: first build takes minutes,
            # cached startups seconds. Ops TRT can't take (e.g. the
            # template sampler's random ops) partition to CUDA/CPU
            # automatically -- measure, don't assume the win.
            cache_dir = os.environ.get(
                "KPLANNER_TRT_CACHE",
                str(Path.home() / ".cache" / "kplanner_trt"))
            os.makedirs(cache_dir, exist_ok=True)
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_timing_cache_enable": True,
                    "trt_fp16_enable": bool(int(os.environ.get(
                        "KPLANNER_TRT_FP16", "0"))),
                }),
            ] + providers
        if OnnxPlannerBackend.USE_GPU and hasattr(ort, "preload_dlls"):
            # Desktop pip installs ship CUDA/cuDNN as nvidia-* wheels that
            # are not on the loader path; preload_dlls() (ORT >= 1.21) loads
            # them from site-packages. No-op where libs resolve system-wide
            # (Jetson) -- guarded so a CPU-only or old ORT is unaffected.
            try:
                ort.preload_dlls()
            except Exception as exc:  # never let GPU setup kill the daemon
                log.warning("ort.preload_dlls() failed (%s); continuing", exc)
        t0 = time.monotonic()
        self._sess = ort.InferenceSession(str(onnx_path), providers=providers)
        _active = self._sess.get_providers()
        log.info("onnxruntime providers requested=%s active=%s", providers, _active)
        if OnnxPlannerBackend.USE_GPU and "CUDAExecutionProvider" not in _active:
            log.warning("--ort-gpu set but CUDAExecutionProvider NOT active -- "
                        "onnxruntime build lacks CUDA; running on CPU.")
        if (OnnxPlannerBackend.USE_TRT
                and "TensorrtExecutionProvider" not in _active):
            log.warning("--ort-trt set but TensorrtExecutionProvider NOT "
                        "active -- falling through to %s.", _active)
        self._contract = contract
        sess_inputs = {i.name for i in self._sess.get_inputs()}
        sess_outputs = [o.name for o in self._sess.get_outputs()]
        roles = contract["inputs"]

        kind = contract.get("graph_kind")
        if kind is None:
            kind = "template" if roles["mode"] in sess_inputs else "velocity"
        self.graph_kind = kind

        # Validate the input roles we intend to feed actually exist.
        needed = [roles["context"], roles["velocity"]]
        if kind == "template":
            needed += [roles["mode"], roles["random_seed"]]
        missing = [n for n in needed if n not in sess_inputs]
        if missing:
            raise ValueError(
                f"ONNX graph {onnx_path} missing expected inputs {missing}; "
                f"graph has {sorted(sess_inputs)}. Fix the sidecar 'inputs' "
                f"mapping."
            )
        for out_name in contract["outputs"].values():
            if out_name not in sess_outputs:
                raise ValueError(
                    f"ONNX graph {onnx_path} missing output {out_name!r}; "
                    f"graph has {sess_outputs}. Fix the sidecar 'outputs'."
                )
        # Any session input not covered by roles/extra_inputs is an error
        # (a silent zero-feed would corrupt predictions).
        covered = set(needed) | set(contract.get("extra_inputs", {}))
        uncovered = sess_inputs - covered
        if uncovered:
            raise ValueError(
                f"ONNX graph has inputs {sorted(uncovered)} not covered by "
                f"the runtime contract; add them to the sidecar's "
                f"'extra_inputs'."
            )

        # Mode index resolution (template graphs only).
        self.mode_idx: Optional[int] = None
        if kind == "template":
            modes = list(contract.get("modes") or _PLANNER_MODE_NAMES)
            mode_name = planner_mode or contract.get("default_mode") or "walk"
            if mode_name not in modes:
                raise ValueError(
                    f"--planner-mode {mode_name!r} not in contract modes {modes}"
                )
            self.mode_idx = modes.index(mode_name)
            if planner_mode is None:
                log.warning(
                    "template graph with no --planner-mode; defaulting to "
                    "%r (idx=%d)", mode_name, self.mode_idx,
                )
            else:
                log.info("pose-template inference: mode=%s (idx=%d)",
                         mode_name, self.mode_idx)
        elif planner_mode is not None:
            log.warning(
                "--planner-mode=%s ignored: %s is a velocity-only graph",
                planner_mode, onnx_path.name,
            )

        self._fixed_seed = contract.get("fixed_random_seed")
        # Env override (2026-08-11 walk-seam sweep): pin the template
        # sampler's seed across replans. With ~2 replans/s while walking
        # and a FRESH seed each time (85 distinct seeds in an 85-replan
        # tape), the served walk is a patchwork of independently sampled
        # gaits -- plan-to-plan phase disagreement no seam blend can heal,
        # only smear. A pinned seed + near-identical context makes
        # consecutive plans coherent continuations instead.
        _env_seed = os.environ.get("KPLANNER_FIXED_SEED")
        if _env_seed:
            self._fixed_seed = int(_env_seed)
            log.info("template sampler seed PINNED via KPLANNER_FIXED_SEED=%d",
                     self._fixed_seed)
        self._filter_baked = bool(contract.get("filter_qpos_baked", True))
        self.REPLAN_THRESHOLD_FRAMES = int(replan_threshold_frames)
        self._buf: Optional[np.ndarray] = None  # [T, 38] f32
        self._cursor = 0
        # Output resampling (30 Hz model -> OUTPUT_FPS control loop) + an
        # 8-tick cross-fade blend across replan seams. numpy port of
        # NeuralPlannerCore.get_next_frame_resampled; inert until
        # get_next_frame_resampled() is first called.
        # Seam-blend knobs (2026-08-11 walk-smoothness sweep): length and
        # weight shape of the replan-seam crossfade. Linear weights are C0
        # only -- the crossfade of two moving trajectories puts a velocity
        # kink at BOTH seam ends every replan (~1/s while walking), the
        # prime suspect for the unwanted recovery steps during walk.
        # halfcos zeroes the weight derivative at the ends (C1 through the
        # seam). Defaults preserve the shipped behaviour exactly.
        self.BLEND_FRAMES = int(os.environ.get(
            "KPLANNER_SEAM_BLEND_FRAMES", "8"))
        self.BLEND_SHAPE = os.environ.get(
            "KPLANNER_SEAM_BLEND_SHAPE", "linear").strip().lower()
        if self.BLEND_SHAPE not in ("linear", "halfcos"):
            log.warning("KPLANNER_SEAM_BLEND_SHAPE=%r unknown; using linear",
                        self.BLEND_SHAPE)
            self.BLEND_SHAPE = "linear"
        if (self.BLEND_FRAMES, self.BLEND_SHAPE) != (8, "linear"):
            log.info("seam blend: %d ticks, %s weights (default 8, linear)",
                     self.BLEND_FRAMES, self.BLEND_SHAPE)
        self._model_fps = float(MODEL_FPS)
        self._resample_active = False
        self._resample_output_fps = float(OUTPUT_FPS)
        self._read_pos = 0.0
        self._blend_prev_buf: Optional[np.ndarray] = None
        self._blend_prev_pos = 0.0
        self._blend_remaining = 0
        log.info(
            "onnx backend ready in %.2fs: %s (kind=%s, filter_qpos_baked=%s)",
            time.monotonic() - t0, onnx_path, kind, self._filter_baked,
        )

    # --- ring-buffer surface (NeuralPlannerCore-equivalent) ---------------

    @property
    def current_frame_idx(self) -> int:
        return self._cursor

    @property
    def frames_remaining(self) -> int:
        if self._buf is None:
            return 0
        if self._resample_active:
            return int(np.floor(self._buf.shape[0] - self._read_pos))
        return int(self._buf.shape[0]) - self._cursor

    def should_replan(self) -> bool:
        if self._buf is None:
            return False
        # Effective threshold respects the committed plan's LENGTH: in-place
        # turn templates commit short plans (~50 native frames) and a fixed
        # threshold of 48 fired ~2 frames after every commit -> replan STORM
        # (up to 8/s), endless seam crossfades, robot slowly sank and
        # collapsed (2026-08-03 sim, in-place right turn). Invariant: at
        # least MIN_SERVED native frames of every plan are served before the
        # next replan fires; plans long enough for the configured threshold
        # keep the tuned responsiveness EXACTLY (64-frame walk plans with
        # threshold 48 still serve 16 = unchanged tuned cadence).
        MIN_SERVED = 16.0   # native frames (~0.53 s @30 fps)
        buf_len = float(self._buf.shape[0])
        thr_eff = max(4.0, min(float(self.REPLAN_THRESHOLD_FRAMES),
                               buf_len - MIN_SERVED))
        if self._resample_active:
            return (self._read_pos + thr_eff) >= buf_len
        return self.frames_remaining <= thr_eff

    def reset(self, init_qpos: np.ndarray) -> None:
        init_qpos = np.asarray(init_qpos, dtype=np.float32).reshape(-1)
        if init_qpos.shape[0] != QPOS_DIM:
            raise ValueError(f"init qpos must be [38], got {init_qpos.shape}")
        self._buf = np.tile(init_qpos[None, :], (NUM_MIN_FRAMES_IN_BUFFER, 1))
        self._cursor = 0
        self._read_pos = 0.0
        self._blend_prev_buf = None
        self._blend_prev_pos = 0.0
        self._blend_remaining = 0
        self._rate_scale = 1.0

    def get_next_frame(self) -> np.ndarray:
        if self._buf is None:
            raise RuntimeError("get_next_frame() called before reset()")
        idx = max(0, min(self._cursor, self._buf.shape[0] - 1))
        self._cursor = min(self._cursor + 1, self._buf.shape[0] - 1)
        return self._buf[idx].copy()

    def peek_frame(self, idx: int) -> np.ndarray:
        assert self._buf is not None
        return self._buf[max(0, min(int(idx), self._buf.shape[0] - 1))].copy()

    # --- resampled read + 8-tick cross-fade (numpy port) ------------------

    def _frame_at(self, buf: np.ndarray, pos: float) -> np.ndarray:
        """Interpolate a single [38] qpos row at fractional index ``pos``.

        Layout ``[trans(3), root_quat_wxyz(4), dof(31)]``: lerp 0:3 and 7:,
        slerp the root quat at 3:7.
        """
        T = int(buf.shape[0])
        if T <= 1:
            return buf[0].copy()
        pos = min(max(float(pos), 0.0), float(T - 1))
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, T - 1)
        frac = pos - i0
        f0 = buf[i0]
        if i1 == i0 or frac <= 0.0:
            return f0.copy()
        f1 = buf[i1]
        out = f0.copy()
        out[:3] = f0[:3] * (1.0 - frac) + f1[:3] * frac
        out[7:] = f0[7:] * (1.0 - frac) + f1[7:] * frac
        out[3:7] = _slerp_wxyz_np(f0[3:7], f1[3:7], frac)
        return out

    def _blend_frames(
        self, old: np.ndarray, new: np.ndarray, w_new: float
    ) -> np.ndarray:
        w_old = 1.0 - w_new
        out = new.copy()
        out[:3] = old[:3] * w_old + new[:3] * w_new
        out[7:] = old[7:] * w_old + new[7:] * w_new
        out[3:7] = _slerp_wxyz_np(old[3:7], new[3:7], w_new)
        return out

    def _resampled_output_frame(self, output_offset_ticks: float = 0.0) -> np.ndarray:
        assert self._buf is not None
        step = (self._model_fps / self._resample_output_fps
                * getattr(self, "_rate_scale", 1.0))
        new_pos = self._read_pos + step * output_offset_ticks
        frame = self._frame_at(self._buf, new_pos)
        if self._blend_prev_buf is not None:
            blend_left = self._blend_remaining - output_offset_ticks
            if blend_left > 0.0:
                old_pos = self._blend_prev_pos + step * output_offset_ticks
                old_frame = self._frame_at(self._blend_prev_buf, old_pos)
                w_new = 1.0 - (blend_left - 1.0) / float(self.BLEND_FRAMES)
                w_new = min(max(w_new, 0.0), 1.0)
                if self.BLEND_SHAPE == "halfcos":
                    w_new = 0.5 * (1.0 - math.cos(math.pi * w_new))
                frame = self._blend_frames(old_frame, frame, w_new)
        return frame

    def get_next_frame_resampled(
        self, output_fps: Optional[float] = None
    ) -> np.ndarray:
        if self._buf is None:
            raise RuntimeError("get_next_frame_resampled() called before reset()")
        if output_fps is not None:
            self._resample_output_fps = float(output_fps)
        self._resample_active = True
        frame = self._resampled_output_frame(0.0)
        step = (self._model_fps / self._resample_output_fps
                * getattr(self, "_rate_scale", 1.0))
        # Starvation telemetry: serving at/past the last buffered frame means
        # SONIC receives a frozen reference at full 50 Hz -- invisible to the
        # silence-based pose watchdog (tape 20260719: stumbles). Should be
        # unreachable with replan threshold 32; scream if it ever recurs.
        if self._read_pos >= self._buf.shape[0] - 1:
            self._starved_ticks = getattr(self, "_starved_ticks", 0) + 1
            if self._starved_ticks in (1, 25) or self._starved_ticks % 250 == 0:
                log.warning("ring STARVED: serving frozen end-of-buffer frame "
                            "(tick %d of this episode)", self._starved_ticks)
            _TAPE.ev("starved", n=self._starved_ticks)
        else:
            self._starved_ticks = 0
        self._read_pos += step
        self._cursor = int(np.floor(self._read_pos))
        if self._blend_prev_buf is not None:
            self._blend_prev_pos += step
            self._blend_remaining -= 1
            if self._blend_remaining <= 0:
                self._blend_prev_buf = None
                self._blend_remaining = 0
        return frame

    def peek_output_frame(self, output_offset_ticks: float) -> np.ndarray:
        if self._buf is None:
            raise RuntimeError("peek_output_frame() called before reset()")
        self._resample_active = True
        return self._resampled_output_frame(float(output_offset_ticks))

    def _get_context(self) -> np.ndarray:
        """buf[[clamp(cursor - 4 + i + PRED_OFFSETS)]] -> [1, 4, 38] f32."""
        assert self._buf is not None
        last = self._buf.shape[0] - 1
        indices = [
            max(0, min(self._cursor - NUM_FRAMES_PER_TOKEN + i + PRED_OFFSETS, last))
            for i in range(NUM_FRAMES_PER_TOKEN)
        ]
        return self._buf[indices][None, :, :].astype(np.float32)

    # --- replan ------------------------------------------------------------

    # -----------------------------------------------------------------
    # Replan is SPLIT into prepare / infer / commit so the ONNX inference
    # can run WITHOUT holding the publisher's lock.
    #
    # Why this matters: the publish loop takes replan_lock to read every
    # 50 Hz frame. When replan() ran wholly inside that lock, a 300-500 ms
    # inference blocked the pose stream for 15-25 frames. SONIC cannot hold
    # a frame -- a gap that long is a collapse. Observed on hardware as
    # repeated "loop fell behind by 300-550ms; resyncing" with the robot
    # nearly going down.
    #
    # Only prepare() and commit() touch shared state (_buf / _read_pos);
    # infer() is pure compute over a snapshot, so it is safe to run unlocked.
    #   with lock: prep = replan_prepare(t)
    #   (no lock): pred, npf = replan_infer(prep)
    #   with lock: replan_commit(pred, npf)
    # -----------------------------------------------------------------
    def replan_prepare(self, target: tuple[float, float, float, float]) -> dict:
        """Snapshot context + build ONNX feeds. CALLER MUST HOLD THE LOCK."""
        if self._buf is None:
            raise RuntimeError("replan() called before reset()")
        roles = self._contract["inputs"]
        context = self._get_context()
        feeds: dict = {
            roles["context"]: context,
            roles["velocity"]: np.asarray([list(target)], dtype=np.float32),
        }
        if self.graph_kind == "template":
            seed = (
                int(self._fixed_seed) if self._fixed_seed is not None
                else int(_CROUCH_SEED) if (_CROUCH_SEED is not None and _CROUCH.get("on"))
                else random.randint(0, 999_999)
            )
            feeds[roles["mode"]] = np.asarray([self.mode_idx], dtype=np.int64)
            feeds[roles["random_seed"]] = np.asarray([seed], dtype=np.int64)
            _TAPE.ev("replan_prep", seed=seed, mode=int(self.mode_idx),
                     target=list(target))
        # Snapshot the serve position so commit can fast-forward the new
        # buffer by whatever played during inference (see replan_commit).
        self._prep_read_pos = float(self._read_pos)
        self._prep_cursor = int(self._cursor)
        for name, spec in (self._contract.get("extra_inputs") or {}).items():
            dtype = np.int64 if spec.get("dtype", "f32") == "i64" else np.float32
            arr = np.asarray(spec["value"], dtype=dtype)
            if "shape" in spec:
                arr = arr.reshape(spec["shape"])
            feeds[name] = arr
        return {"feeds": feeds, "context": context}

    def replan_infer(self, prep: dict) -> tuple[np.ndarray, int]:
        """Run the planner graph. MUST be called WITHOUT the lock.

        This is the 300-500 ms step. It reads only the snapshot in ``prep``
        and mutates no shared state, so the publisher keeps streaming while
        it runs.
        """
        outs = self._contract["outputs"]
        context = prep["context"]

        qpos_out, npf_out = self._sess.run(
            [outs["qpos"], outs["num_pred_frames"]], prep["feeds"]
        )
        npf = int(np.asarray(npf_out).reshape(-1)[0])
        npf = max(1, min(npf, int(qpos_out.shape[1])))
        pred = np.asarray(qpos_out[0, :npf], dtype=np.float32).copy()

        if not self._filter_baked:
            # FILTER_QPOS context blend, numpy port of NeuralPlannerCore's
            # (linspace 0.3..0.7 over the 4 context frames; root quat rows
            # [3:7] untouched). Only used for graphs exported WITHOUT the
            # in-graph blend (sidecar filter_qpos_baked=false).
            ctx = context[0]  # [4, 38] raw context (pre-canonicalize copy)
            num_ctx = ctx.shape[0]
            n = min(num_ctx, pred.shape[0])
            blend = np.linspace(0.3, 0.7, num_ctx, dtype=np.float32)[:n, None]
            pred[:n, :3] = ctx[:n, :3] * (1 - blend) + pred[:n, :3] * blend
            pred[:n, 7:] = ctx[:n, 7:] * (1 - blend) + pred[:n, 7:] * blend
        return pred, npf

    def replan_commit(self, pred: np.ndarray, npf: int) -> int:
        """Arm the seam blend and swap in the new buffer.

        CALLER MUST HOLD THE LOCK. Pure bookkeeping -- microseconds, not the
        hundreds of milliseconds that inference costs.
        """
        # Arm the 8-tick cross-fade before swapping the buffer (snapshot the
        # still-playing old buffer + read cursor). No-op unless resampling
        # is active.
        if self._resample_active and self._buf is not None:
            self._blend_prev_buf = self._buf
            self._blend_prev_pos = float(self._read_pos)
            self._blend_remaining = self.BLEND_FRAMES
        # REWIND FIX (run 20260719_214150): the new chunk continues from the
        # PREP-time context, but the publisher kept serving the old buffer
        # during the 0.3-0.6 s inference. Restarting the new buffer at 0
        # therefore rewound the served reference by the frames consumed
        # during inference (~18 = half a gait cycle at walk cadence); the
        # 8-tick seam blend then averaged antiphase leg poses into a
        # near-still reference. Fast-forward the new buffer by exactly the
        # frames consumed since prep so served content stays continuous.
        base = NUM_FRAMES_PER_TOKEN - PRED_OFFSETS  # == 0
        consumed = 0.0
        if getattr(self, "commit_fastforward", True):
            if self._resample_active:
                consumed = max(0.0, float(self._read_pos)
                               - getattr(self, "_prep_read_pos", self._read_pos))
            else:
                consumed = float(max(0, self._cursor
                                     - getattr(self, "_prep_cursor", self._cursor)))
            consumed = min(consumed, max(0.0, float(npf - 2)))
        # ---- SEAM YAW ALIGNMENT (2026-08-09) --------------------------------
        # ROOT CAUSE of the reference yaw spikes: the fast-forward above fixes
        # the frame INDEX but not the HEADING. The new chunk is generated from
        # the PREP-time context, and during the 0.3-0.6 s of inference the
        # publisher keeps serving the old buffer -- which, mid-turn, rotates the
        # robot by 0.2-0.3 rad the new plan knows nothing about. Model drift
        # adds more. The 8-tick cross-fade then has to absorb that offset,
        # turning it into a 6-17 rad/s reference step.
        #
        # Measured (replay_intents_to_daemon.py on a real operator session):
        # 100 % of reference-yaw spikes >5 rad/s land within 100 ms of a
        # replan_done (baseline 43 %), median 46 ms vs 128 ms.
        #
        # Fix: rigidly rotate the incoming plan about its own seam frame so its
        # heading AND position match what is being served right now. This
        # removes only the absolute offset -- the plan's internal yaw profile
        # (i.e. the commanded turn) is preserved exactly, because a rigid
        # rotation cannot change frame-to-frame deltas.
        if (SEAM_YAW_ALIGN and self._buf is not None
                and pred is not None and pred.shape[0] > 0):
            seam = int(max(0, min(int(base + consumed), pred.shape[0] - 1)))
            if self._resample_active:
                served = self._frame_at(self._buf, float(self._read_pos))
            else:
                served = self._buf[max(0, min(self._cursor, self._buf.shape[0] - 1))]
            d = _wrap_pi(yaw_from_quat_wxyz(np.asarray(served[3:7], np.float64))
                         - yaw_from_quat_wxyz(np.asarray(pred[seam, 3:7], np.float64)))
            # Skip the no-op case: rotating every frame costs more than the
            # sub-milliradian it would remove, and it churns float noise.
            if abs(d) > 1e-4:
                c, s = math.cos(d), math.sin(d)
                pred = pred.copy()
                # quats: wxyz, pre-multiply by R_z(d)
                qd = np.array([math.cos(d * 0.5), 0.0, 0.0, math.sin(d * 0.5)])
                q = pred[:, 3:7].astype(np.float64)
                pred[:, 3:7] = np.stack([
                    qd[0]*q[:,0] - qd[3]*q[:,3],
                    qd[0]*q[:,1] - qd[3]*q[:,2],
                    qd[0]*q[:,2] + qd[3]*q[:,1],
                    qd[0]*q[:,3] + qd[3]*q[:,0]], axis=1).astype(pred.dtype)
                # XY: rotate the path about the seam point, then pin the seam to
                # the served XY so the swap is positionally continuous too.
                p0 = pred[seam, :2].astype(np.float64).copy()
                rel = pred[:, :2].astype(np.float64) - p0
                pred[:, 0] = (rel[:, 0]*c - rel[:, 1]*s + served[0]).astype(pred.dtype)
                pred[:, 1] = (rel[:, 0]*s + rel[:, 1]*c + served[1]).astype(pred.dtype)

        self._buf = pred
        self._cursor = int(base + consumed)
        if self._resample_active:
            self._read_pos = float(base) + consumed
        if consumed > 0:
            _TAPE.ev("commit_ff", consumed=round(consumed, 2), npf=int(npf))
        _TAPE.chunk(pred, npf)
        return npf

    def replan(self, target: tuple[float, float, float, float]) -> int:
        """Single-threaded convenience wrapper (offline clip generation, A/B).

        The live publisher must NOT use this -- it would hold the lock across
        inference again. Use prepare/infer/commit with the lock released around
        infer(). Safe here because these callers have no concurrent reader.
        """
        prep = self.replan_prepare(target)
        pred, npf = self.replan_infer(prep)
        return self.replan_commit(pred, npf)

    def describe(self) -> str:
        return f"onnx/{self.graph_kind}" + (
            "" if self.mode_idx is None else f"(mode_idx={self.mode_idx})"
        )


class TorchPlannerBackend:
    """A/B backend wrapping the real NeuralPlannerCore, called exactly the
    way ``x2_kplanner.run()`` calls it. torch/motionbricks imported lazily
    here so the ONNX path stays torch-free."""

    def __init__(
        self,
        vqvae_ckpt: Path,
        pose_ckpt: Path,
        root_ckpt: Path,
        device: str,
        replan_threshold_frames: int,
        planner_mode: Optional[str],
    ) -> None:
        import torch  # noqa: PLC0415 -- torch backend only
        from motionbricks.motion_backbone.inference.load_x2_planner import (
            X2PlannerPaths,
            load_x2_planner,
        )

        self._torch = torch
        default_paths = X2PlannerPaths.default()
        paths = X2PlannerPaths(
            vqvae_ckpt=vqvae_ckpt,
            pose_ckpt=pose_ckpt,
            root_ckpt=root_ckpt,
            vqvae_version_dir=default_paths.vqvae_version_dir,
            pose_version_dir=default_paths.pose_version_dir,
            root_version_dir=default_paths.root_version_dir,
        )
        log.info("loading X2 kplanner stack on device=%s ...", device)
        self._core = load_x2_planner(
            paths, device=device,
            replan_threshold_frames=replan_threshold_frames,
        )
        log.info("kplanner stack loaded.")

        self.graph_kind = "velocity" if planner_mode is None else "template"
        self.mode_idx: Optional[int] = None
        if planner_mode is not None:
            if self._core._clip_library is None:
                raise RuntimeError(
                    f"--planner-mode={planner_mode} requested but no clip "
                    f"library is loaded (bake out/X2-clip.ckpt first)."
                )
            self.mode_idx = _PLANNER_MODE_NAMES.index(planner_mode)
            log.info("pose-template inference: mode=%s (idx=%d)",
                     planner_mode, self.mode_idx)
        self.REPLAN_THRESHOLD_FRAMES = int(replan_threshold_frames)

    @property
    def current_frame_idx(self) -> int:
        return int(self._core.current_frame_idx)

    @property
    def frames_remaining(self) -> int:
        return int(self._core.frames_remaining)

    def should_replan(self) -> bool:
        return bool(self._core.should_replan())

    def reset(self, init_qpos: np.ndarray) -> None:
        self._core.reset(
            self._torch.from_numpy(np.asarray(init_qpos, dtype=np.float32))
        )

    def get_next_frame(self) -> np.ndarray:
        return self._core.get_next_frame().detach().cpu().numpy()

    def get_next_frame_resampled(
        self, output_fps: Optional[float] = None
    ) -> np.ndarray:
        return (
            self._core.get_next_frame_resampled(output_fps).detach().cpu().numpy()
        )

    def peek_output_frame(self, output_offset_ticks: float) -> np.ndarray:
        return (
            self._core.peek_output_frame(output_offset_ticks).detach().cpu().numpy()
        )

    def peek_frame(self, idx: int) -> np.ndarray:
        buf = self._core.frames["mujoco_qpos"]
        i = max(0, min(int(idx), buf.shape[1] - 1))
        return buf[0, i].detach().cpu().numpy()

    def replan(self, target: tuple[float, float, float, float]) -> int:
        if self.mode_idx is None:
            _, _, npf = self._core.replan_with_velocity(list(target))
        else:
            _, _, npf = self._core.replan_with_pose_template(
                list(target), mode_idx=self.mode_idx
            )
        return int(npf)

    # NOTE: deliberately does NOT implement the prepare/infer/commit split.
    # The torch core fuses inference and buffer swap, so there is no safe way
    # to run part of it unlocked. The worker feature-detects the split API and
    # falls back to the locked replan() here. That costs a stall, which is
    # acceptable because this backend is the offline A/B path and never runs
    # on the robot -- and a stall is far better than a half-applied replan.

    def describe(self) -> str:
        return f"torch/{self.graph_kind}" + (
            "" if self.mode_idx is None else f"(mode_idx={self.mode_idx})"
        )


# ---------------------------------------------------------------------------
# IntentState (port of x2_kplanner.IntentState).
# ---------------------------------------------------------------------------


class IntentState:
    def __init__(self, initial: tuple[float, float, float, float]) -> None:
        self._target = initial
        self._lock = threading.Lock()
        self._version = 0
        self._last_set_t = time.monotonic()

    def set(self, target: tuple[float, float, float, float]) -> None:
        with self._lock:
            self._target = tuple(target)
            self._version += 1
            self._last_set_t = time.monotonic()

    def get(self) -> tuple[tuple[float, float, float, float], int]:
        with self._lock:
            return self._target, self._version

    def seconds_since_last_set(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_set_t

    def force_idle_if_stale(
        self, max_age_s: float, idle_target: tuple[float, float, float, float]
    ) -> bool:
        with self._lock:
            age = time.monotonic() - self._last_set_t
            if age < max_age_s:
                return False
            if tuple(self._target) == tuple(idle_target):
                return False
            self._target = tuple(idle_target)
            self._version += 1
            return True


# ---------------------------------------------------------------------------
# Cold-start velocity ramp (port; default OFF, matching x2_kplanner
# 2026-07-16 default).
# ---------------------------------------------------------------------------


class ColdStartVelocityRamp:
    def __init__(self, tau_s: float = 0.0) -> None:
        self.tau_s = float(tau_s)
        self._smoothed = np.zeros(3, dtype=np.float64)
        self._last_was_idle = True

    @property
    def enabled(self) -> bool:
        return self.tau_s > 0.0

    def step(
        self, target: tuple[float, float, float, float], dt_s: float
    ) -> tuple[float, float, float, float]:
        yaw, vx, vz, hip = target
        if self._last_was_idle:
            self._smoothed.fill(0.0)
        target_vec = np.array([yaw, vx, vz], dtype=np.float64)
        if self.tau_s <= 0.0 or dt_s <= 0.0:
            self._smoothed = target_vec
        else:
            alpha = float(dt_s) / (self.tau_s + float(dt_s))
            self._smoothed += alpha * (target_vec - self._smoothed)
        self._last_was_idle = False
        return (float(self._smoothed[0]), float(self._smoothed[1]),
                float(self._smoothed[2]), float(hip))

    def reset_idle(self) -> None:
        self._smoothed.fill(0.0)
        self._last_was_idle = True


# ---------------------------------------------------------------------------
# Reference-step smoother (compact port of x2_kplanner._ReferenceStepSmoother;
# same defaults: 300 ms halfcos ramp, 0.05 rad trigger, lower-body joints).
# ---------------------------------------------------------------------------

_REF_SMOOTHER_JOINTS_PRESETS: dict[str, np.ndarray] = {
    "lower_body": np.arange(0, 15, dtype=np.int64),
    "legs_only":  np.arange(0, 12, dtype=np.int64),
    "all":        np.arange(0, 31, dtype=np.int64),
}
_REF_SMOOTHER_SHAPES: tuple[str, ...] = ("halfcos", "linear", "off")


class ReferenceStepSmoother:
    def __init__(
        self,
        ramp_duration_s: float = 0.300,
        trigger_rad: float = 0.05,
        shape: str = "halfcos",
        blend_indices: Optional[np.ndarray] = None,
    ) -> None:
        if shape not in _REF_SMOOTHER_SHAPES:
            raise ValueError(f"shape={shape!r} not in {_REF_SMOOTHER_SHAPES}")
        self.ramp_duration_s = float(ramp_duration_s)
        self.trigger_rad = float(trigger_rad)
        self.shape = shape
        self.blend_indices = (
            blend_indices if blend_indices is not None
            else _REF_SMOOTHER_JOINTS_PRESETS["lower_body"]
        )
        self._last_q: Optional[np.ndarray] = None
        self._ramp_active = False
        self._ramp_start_t = 0.0
        self._source_q: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        return self.shape != "off" and self.ramp_duration_s > 0.0

    def _alpha(self, t_in_ramp: float) -> float:
        x = max(0.0, min(1.0, t_in_ramp / self.ramp_duration_s))
        if self.shape == "halfcos":
            return 0.5 * (1.0 - math.cos(math.pi * x))
        return x  # linear

    def update(
        self, target_q: np.ndarray, t_now: float, allow_arm: bool = True
    ) -> np.ndarray:
        """``allow_arm=False`` suppresses NEW ramp arming (an active ramp
        still completes). Used by dance playback: the clip's own fast leg
        motion legitimately exceeds the step trigger every few ticks, and
        continuous re-arming would lag the choreography by up to one ramp
        duration -- only the entry step (anchor -> clip frame 0) should
        ramp."""
        target_q = np.asarray(target_q)
        if not self.enabled or self._last_q is None:
            self._last_q = target_q.astype(target_q.dtype, copy=True)
            return self._last_q.copy()
        bi = self.blend_indices
        delta_max = float(np.max(np.abs(target_q[bi] - self._last_q[bi])))
        if allow_arm and (not self._ramp_active) and delta_max > self.trigger_rad:
            self._ramp_active = True
            self._ramp_start_t = float(t_now)
            self._source_q = self._last_q.astype(target_q.dtype, copy=True)
            log.info(
                "ref-smoother: armed (delta=%.3f rad, trigger=%.3f rad, "
                "T=%.0f ms, shape=%s)",
                delta_max, self.trigger_rad,
                self.ramp_duration_s * 1000.0, self.shape,
            )
        if self._ramp_active and self._source_q is not None:
            t_in_ramp = float(t_now) - self._ramp_start_t
            if t_in_ramp >= self.ramp_duration_s:
                self._ramp_active = False
                self._source_q = None
                out = target_q.astype(target_q.dtype, copy=True)
            else:
                alpha = self._alpha(t_in_ramp)
                out = target_q.astype(target_q.dtype, copy=True)
                out[bi] = (
                    (1.0 - alpha) * self._source_q[bi] + alpha * target_q[bi]
                ).astype(target_q.dtype, copy=False)
        else:
            out = target_q.astype(target_q.dtype, copy=True)
        self._last_q = out
        return out.copy()


# ---------------------------------------------------------------------------
# Wire payload builder. EXACT field set + insertion order of
# state_machine.build_pose_payload (the consumer decodes by order):
#   joint_pos_mj, root_quat_xyzw, motion_token, left_hand_joints,
#   right_hand_joints, frame_index, root_xy_world, root_z_world,
#   joint_pos_mj_future, root_quat_xyzw_future, joint_vel_mj_future,
#   frame_index_future, future_dt_s
# ---------------------------------------------------------------------------


def build_pose_payload_np(
    jpos: np.ndarray,
    quat_xyzw: np.ndarray,
    root_xy: np.ndarray,
    root_z: float,
    frame_index: int,
    future_jpos: list[np.ndarray],
    future_quat: list[np.ndarray],
    motion_token_dim: int = SONIC_MOTION_TOKEN_DIM,
    hand_dof: int = DEFAULT_HAND_DOF,
    future_dt_s: float = FUTURE_DT_S,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "joint_pos_mj": np.asarray(jpos, dtype=np.float32),
        "root_quat_xyzw": np.asarray(quat_xyzw, dtype=np.float32),
        "motion_token": np.zeros(motion_token_dim, dtype=np.float32),
        # OmniHand rest pose on the wire (operator 2026-09-05: "on kplanner start,
        # always start with the fingers CLOSED if we have a hand"): the deploy hand
        # bridge follows these, the arm-ingest overlay overrides them whenever the
        # sender publishes grips. KPLANNER_HAND_DEFAULT=open restores the old zeros.
        "left_hand_joints": _hand_default("left", hand_dof),
        "right_hand_joints": _hand_default("right", hand_dof),
        "frame_index": np.array([frame_index], dtype=np.int64),
        "root_xy_world": np.asarray(root_xy, dtype=np.float32),
        "root_z_world": np.array([float(root_z)], dtype=np.float32),
    }
    if future_jpos:
        n_future = len(future_jpos)
        jpos_future = np.stack(
            [np.asarray(f, dtype=np.float32) for f in future_jpos]
        )
        rot_future = np.stack(
            [np.asarray(f, dtype=np.float32) for f in future_quat]
        )
        prev_jpos = np.asarray(jpos, dtype=np.float32)[None, :]
        all_jpos = np.concatenate([prev_jpos, jpos_future], axis=0)
        jvel_future = (
            (all_jpos[1:] - all_jpos[:-1]) / max(float(future_dt_s), 1e-6)
        ).astype(np.float32)
        step_ticks = int(round(future_dt_s * OUTPUT_FPS))
        frame_idx_future = np.array(
            [frame_index + (k + 1) * step_ticks for k in range(n_future)],
            dtype=np.int64,
        )
        payload["joint_pos_mj_future"] = jpos_future
        payload["root_quat_xyzw_future"] = rot_future
        payload["joint_vel_mj_future"] = jvel_future
        payload["frame_index_future"] = frame_idx_future
        payload["future_dt_s"] = np.array([float(future_dt_s)], dtype=np.float32)
    return payload


# ---------------------------------------------------------------------------
# VR arm/hand target ingest (hop-in manipulation, 2026-07-30).
#
# The tethered laptop stack merges operator arm IK into the pose wire via the
# dataset recorder. The onboard (ritual) stack has no recorder, so this class
# is the robot-side replacement: a SUB (bind tcp://*:ARM_TARGET_PORT) that the
# laptop manager PUB-connects to (--arm-connect), caching the operator's arm
# and hand targets, plus an overlay applied at the PosePublisher choke point:
#   * joint_pos_mj[15:22]/[22:29]      <- left/right arm q (7 DOF each)
#   * joint_pos_mj_future[:, slices]   <- same pose pinned across the window
#     (mirrors the recorder: arm_targets is a "current command", there is no
#     arm trajectory to look ahead with)
#   * left/right_hand_joints           <- hand q (the deploy hand bridge reads
#     fingers straight off these pose-wire fields)
# Semantics mirror the recorder exactly: the cache holds the last commanded
# pose until a passthrough message clears it (link loss => arms HOLD, never
# snap); a dance clip (dance_active) suspends the overlay so clips own the
# whole body. Fail-open: no manager connected -> planner behaves as before.
# ---------------------------------------------------------------------------

_LEFT_ARM_MJ = slice(15, 22)
_RIGHT_ARM_MJ = slice(22, 29)


class ArmTargetIngest:
    def __init__(self, port: int) -> None:
        import zmq
        self._lock = threading.Lock()
        self._left: Optional[np.ndarray] = None
        self._right: Optional[np.ndarray] = None
        self._vel_left: Optional[np.ndarray] = None
        self._vel_right: Optional[np.ndarray] = None
        self._arm_msg_t: float = 0.0
        self._left_hand: Optional[np.ndarray] = None
        self._right_hand: Optional[np.ndarray] = None
        self._last_msg_t = 0.0
        self._stop = threading.Event()
        self._port = int(port)
        self._thr = threading.Thread(
            target=self._loop, name="arm-ingest", daemon=True)
        self._thr.start()

    def _loop(self) -> None:
        import zmq
        try:
            import msgpack
        except ImportError:
            log.error("arm-ingest: msgpack unavailable; VR arm targets OFF")
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        for t in ("arm_targets", "hand_finger_cmd"):
            sock.setsockopt_string(zmq.SUBSCRIBE, t)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.bind(f"tcp://*:{self._port}")
        log.info("arm-ingest: SUB bind tcp://*:%d (arm_targets, "
                 "hand_finger_cmd); laptop manager PUB-connects here "
                 "(--arm-connect)", self._port)
        while not self._stop.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                continue
            if len(parts) < 2:
                continue
            try:
                topic = parts[0].decode("ascii", errors="replace")
                data = msgpack.unpackb(parts[1], raw=False)
                if topic == "arm_targets":
                    with self._lock:
                        if data.get("passthrough_arm_targets"):
                            self._left = None
                            self._right = None
                            self._vel_left = None
                            self._vel_right = None
                        else:
                            lq = np.asarray(
                                data.get("left_q_rad", ()), dtype=np.float32)
                            rq = np.asarray(
                                data.get("right_q_rad", ()), dtype=np.float32)
                            if lq.shape == (7,) and rq.shape == (7,):
                                now_m = time.monotonic()
                                # Per-joint target velocity from the last
                                # two arrivals — used to extrapolate through
                                # stream gaps (first-order hold).
                                prev = self._left
                                prev_t = self._arm_msg_t
                                if prev is not None and prev_t and                                         0.005 < now_m - prev_t < 0.5:
                                    dt = now_m - prev_t
                                    self._vel_left = np.clip(
                                        (lq - prev) / dt, -3.0, 3.0)
                                    self._vel_right = np.clip(
                                        (rq - self._right) / dt, -3.0, 3.0)
                                else:
                                    self._vel_left = None
                                    self._vel_right = None
                                self._left = lq
                                self._right = rq
                                self._wb_latched = False
                                self._arm_msg_t = now_m
                        self._last_msg_t = time.monotonic()
                elif topic == "hand_finger_cmd":
                    lh = np.asarray(
                        data.get("left_hand_q", ()), dtype=np.float32)
                    rh = np.asarray(
                        data.get("right_hand_q", ()), dtype=np.float32)
                    if lh.shape == (10,) and rh.shape == (10,):
                        with self._lock:
                            self._left_hand = lh
                            self._right_hand = rh
                            self._last_msg_t = time.monotonic()
            except Exception as exc:  # malformed frame must never kill pose
                log.warning("arm-ingest: bad frame dropped (%r)", exc)
        sock.close(linger=0)

    # Max per-published-frame target step (rad @ 50 Hz => 3.0 rad/s arm,
    # 7.5 rad/s fingers). Smooths the sample-and-hold chop from wifi /
    # headset micro-stalls (measured: 29% frozen frames + 0.23 rad jumps
    # during continuous motion) and doubles as blend-in on first engage
    # (slew seeds from the planner's current arm pose, so arms ramp from
    # stand to the operator pose instead of snapping).
    ARM_SLEW_RAD_PER_FRAME = 0.06
    HAND_SLEW_RAD_PER_FRAME = 0.15

    @staticmethod
    def _slew(cur: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
        d = target - cur
        return cur + np.clip(d, -step, step)

    def overlay(self, payload: dict) -> bool:
        """Apply cached targets to a pose payload. Returns True if the
        arm slices were overlaid (used for state-transition logging)."""
        with self._lock:
            left, right = self._left, self._right
            lh, rh = self._left_hand, self._right_hand
            releasing = getattr(self, "_releasing", False)
            vel_l, vel_r = self._vel_left, self._vel_right
            arm_t = self._arm_msg_t
        if left is not None and vel_l is not None and vel_r is not None:
            # First-order hold: extrapolate along the last observed target
            # velocity during stream gaps (wifi / headset micro-stalls), so
            # motion continues instead of freezing. Capped at 250 ms — a
            # genuinely dead stream degrades to a plain hold, and the
            # ownership silence-release clears it at 2 s.
            gap = time.monotonic() - arm_t
            if 0.0 < gap:
                h = min(gap, 0.25)
                left = left + vel_l * h
                right = right + vel_r * h
        if releasing and left is None:
            # Ownership released: slew back toward the planner's own arm
            # pose; deactivate once converged (or state was never seeded).
            jpos = payload.get("joint_pos_mj")
            if (getattr(self, "_slew_left", None) is None or jpos is None
                    or jpos.shape != (31,)):
                self._releasing = False
                self._slew_left = None
                self._slew_right = None
                self._slew_lh = None
                self._slew_rh = None
                return False
            tgt_l = np.asarray(jpos[_LEFT_ARM_MJ], dtype=np.float32)
            tgt_r = np.asarray(jpos[_RIGHT_ARM_MJ], dtype=np.float32)
            self._slew_left = self._slew(
                self._slew_left, tgt_l, self.ARM_SLEW_RAD_PER_FRAME)
            self._slew_right = self._slew(
                self._slew_right, tgt_r, self.ARM_SLEW_RAD_PER_FRAME)
            done = (np.abs(self._slew_left - tgt_l).max() < 0.02
                    and np.abs(self._slew_right - tgt_r).max() < 0.02)
            jpos = jpos.copy()
            jpos[_LEFT_ARM_MJ] = self._slew_left
            jpos[_RIGHT_ARM_MJ] = self._slew_right
            payload["joint_pos_mj"] = jpos
            fut = payload.get("joint_pos_mj_future")
            if fut is not None and fut.ndim == 2 and fut.shape[1] == 31:
                fut = fut.copy()
                fut[:, _LEFT_ARM_MJ] = self._slew_left
                fut[:, _RIGHT_ARM_MJ] = self._slew_right
                payload["joint_pos_mj_future"] = fut
            if done:
                self._releasing = False
                self._slew_left = None
                self._slew_right = None
                self._slew_lh = None
                self._slew_rh = None
            return not done
        if left is not None and right is not None:
            # COPY-ON-WRITE, never mutate in place: build_pose_payload_np's
            # np.asarray typically aliases the planner's own anchor/output
            # arrays — in-place writes would pollute planner state and the
            # arm pose would survive a passthrough clear.
            jpos = payload.get("joint_pos_mj")
            if jpos is not None and jpos.shape == (31,):
                # Slew toward the operator target from the last frame we
                # PUBLISHED (seeded from the planner's own arms on first
                # engage), never jumping more than ARM_SLEW_RAD_PER_FRAME.
                if getattr(self, "_slew_left", None) is None:
                    self._slew_left = np.asarray(
                        jpos[_LEFT_ARM_MJ], dtype=np.float32).copy()
                    self._slew_right = np.asarray(
                        jpos[_RIGHT_ARM_MJ], dtype=np.float32).copy()
                self._slew_left = self._slew(
                    self._slew_left, left, self.ARM_SLEW_RAD_PER_FRAME)
                self._slew_right = self._slew(
                    self._slew_right, right, self.ARM_SLEW_RAD_PER_FRAME)
                left = self._slew_left
                right = self._slew_right
                jpos = jpos.copy()
                jpos[_LEFT_ARM_MJ] = left
                jpos[_RIGHT_ARM_MJ] = right
                payload["joint_pos_mj"] = jpos
            fut = payload.get("joint_pos_mj_future")
            if fut is not None and fut.ndim == 2 and fut.shape[1] == 31:
                fut = fut.copy()
                fut[:, _LEFT_ARM_MJ] = left
                fut[:, _RIGHT_ARM_MJ] = right
                payload["joint_pos_mj_future"] = fut
            # Future joint velocities were finite-differenced BEFORE the
            # overlay pinned the arm slices; a pinned pose has zero arm
            # velocity, so zero those slices for consistency.
            jvel = payload.get("joint_vel_mj_future")
            if jvel is not None and jvel.ndim == 2 and jvel.shape[1] == 31:
                jvel = jvel.copy()
                jvel[:, _LEFT_ARM_MJ] = 0.0
                jvel[:, _RIGHT_ARM_MJ] = 0.0
                payload["joint_vel_mj_future"] = jvel
        if lh is not None and payload.get("left_hand_joints") is not None:
            if getattr(self, "_slew_lh", None) is None:
                self._slew_lh = np.asarray(
                    payload["left_hand_joints"], dtype=np.float32).copy()
            self._slew_lh = self._slew(
                self._slew_lh, lh, self.HAND_SLEW_RAD_PER_FRAME)
            payload["left_hand_joints"] = self._slew_lh
        if rh is not None and payload.get("right_hand_joints") is not None:
            if getattr(self, "_slew_rh", None) is None:
                self._slew_rh = np.asarray(
                    payload["right_hand_joints"], dtype=np.float32).copy()
            self._slew_rh = self._slew(
                self._slew_rh, rh, self.HAND_SLEW_RAD_PER_FRAME)
            payload["right_hand_joints"] = self._slew_rh
        return left is not None and right is not None

    def latch(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        """WB HOLD: pin the arms at the given (measured) joints as if they
        were operator targets, so the overlay keeps them on the wire through
        the hold and the rate-limited return (manipulation mid-task must not
        be disturbed by a link drop). Released by clear() -- explicit
        "arm_release" intent or the next whole-body engage -- which slews
        back to the planner's arms instead of snapping."""
        with self._lock:
            self._left = np.asarray(left_q, dtype=np.float32).copy()
            self._right = np.asarray(right_q, dtype=np.float32).copy()
            self._vel_left = None
            self._vel_right = None
            self._arm_msg_t = time.monotonic()
            self._last_msg_t = self._arm_msg_t
            self._releasing = False
            self._wb_latched = True
            # seed the slew AT the latched pose: no blend-in step
            self._slew_left = self._left.copy()
            self._slew_right = self._right.copy()

    def unlatch(self) -> None:
        """Drop a WB HOLD latch silently (no release slew): used when the
        whole-body link re-engages and the reference goes back to
        SHADOWING the measured joints, arms included."""
        with self._lock:
            if not getattr(self, "_wb_latched", False):
                return
            self._wb_latched = False
            self._left = None
            self._right = None
            self._vel_left = None
            self._vel_right = None
            self._releasing = False
            self._slew_left = None
            self._slew_right = None

    def clear(self, force: bool = False) -> None:
        """Drop cached arm + hand targets (planner arms take over).

        A WB HOLD latch (``latch``) outranks the VR-ownership housekeeping
        that calls this: the Pico thumbsticks publish ``source:"vr"``
        intents, and their 2 s silence release used to clear the latched
        arms 2 s after the operator let go of the stick (operator session
        2026-09-04 14:29: "hands must stay where they were after the
        switch"). Only ``force=True`` (explicit ``arm_release`` intent)
        drops a WB latch.

        Called when VR command ownership releases — explicit disengage or
        the crash-silence timeout — so a dead/disengaged manager can never
        leave the robot stuck holding a stale manipulation pose (incident
        2026-07-31: headset stream stalled mid-ARM_MAN, operator killed
        the manager, arms held the frozen pose with no reset path).
        """
        with self._lock:
            if getattr(self, "_wb_latched", False) and not force:
                return
            self._wb_latched = False
            self._left = None
            self._right = None
            self._left_hand = None
            self._right_hand = None
            self._vel_left = None
            self._vel_right = None
            # Blend-out: if we were overlaying, keep slewing toward the
            # PLANNER's arms until converged instead of snapping back in
            # one frame (release can happen with arms fully extended).
            self._releasing = getattr(self, "_slew_left", None) is not None

    def stop(self) -> None:
        self._stop.set()


class WaistOverlay:
    """v7.4 waist overlay: slew ``waist_*_deg`` targets at HOLD_SLEW_DPS and
    add the current deltas onto the outgoing waist DOF slots (current frame
    AND future window), copy-on-write like ArmTargetIngest.overlay. Only the
    3 waist slots are touched so planner leg/arm motion stays byte-identical.
    Applied in PosePublisher.publish -> active in every branch (idle anchor,
    PLAYING, dance, primitive)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.target = np.zeros(3, dtype=np.float64)   # pitch, roll, yaw deg
        self.current = np.zeros(3, dtype=np.float64)

    def set_target(self, pitch_deg: float, roll_deg: float, yaw_deg: float) -> None:
        with self._lock:
            self.target[:] = (float(pitch_deg), float(roll_deg), float(yaw_deg))

    def clear(self) -> None:
        with self._lock:
            self.target[:] = 0.0

    def step_and_apply(self, payload: dict, dt_s: float) -> bool:
        with self._lock:
            step = HOLD_SLEW_DPS * float(dt_s)
            delta = np.clip(self.target - self.current, -step, step)
            self.current += delta
            cur = self.current.copy()
        if not np.any(np.abs(cur) > 1e-6):
            return False
        rad = np.deg2rad(cur)
        jpos = payload.get("joint_pos_mj")
        if jpos is not None:
            jpos = np.asarray(jpos, dtype=np.float32).copy()
            jpos[WAIST_PITCH_IDX] += rad[0]
            jpos[WAIST_ROLL_IDX] += rad[1]
            jpos[WAIST_YAW_IDX] += rad[2]
            payload["joint_pos_mj"] = jpos
        fut = payload.get("joint_pos_mj_future")
        if fut is not None:
            fut = np.asarray(fut, dtype=np.float32).copy()
            fut[:, WAIST_PITCH_IDX] += rad[0]
            fut[:, WAIST_ROLL_IDX] += rad[1]
            fut[:, WAIST_YAW_IDX] += rad[2]
            payload["joint_pos_mj_future"] = fut
        return True


class WristOverlay:
    """Whole-body teleop wrist overlay (2026-09-04): ABSOLUTE targets for
    the four bypassed wrist DOFs (MJ 20/21 left pitch/roll, 27/28 right),
    written onto the outgoing pose-ref -- which is what the deploy's
    ``--wrist-bypass ik`` actually samples (NOT the token message). Fed by
    the PC2 token service (`wrist_targets` on planner_cmd) from the
    operator's SMPL wrist rotation. Slewed at WRIST_SLEW_RAD_PER_FRAME
    (3 rad/s @50 Hz) from the frame we last published, so engage/release
    never step. Latches the last targets when the stream stops (the
    manipulation pose survives a link drop); cleared by ``arm_release`` or
    ``clear()`` -> slews back to the planner's own wrists."""

    WRIST_SLEW_RAD_PER_FRAME = 0.06
    _IDX = {"lp": 20, "lr": 21, "rp": 27, "rr": 28}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target: Optional[np.ndarray] = None    # [lp, lr, rp, rr]
        self._cur: Optional[np.ndarray] = None
        self._releasing = False

    def set_target(self, lp: float, lr: float, rp: float, rr: float) -> None:
        with self._lock:
            self._target = np.array([lp, lr, rp, rr], dtype=np.float32)
            self._releasing = False
            self._wb_latched = False

    def latch(self, lp: float, lr: float, rp: float, rr: float) -> None:
        """WB HOLD: pin the four bypassed wrists at the given (measured)
        angles, seeded so the wire does not step. Needed on the dual-head
        path, where no token service feeds ``wrist_targets``: the deploy's
        wrist bypass reads the STREAM while the smpl head drives and the
        POSE-REF after the hand-over -- without this latch the wrists
        snapped to the planner's default at every release. Overridden by
        the next ``set_target`` (legacy token service) and dropped only by
        ``clear(force=True)`` (``arm_release``)."""
        with self._lock:
            self._target = np.array([lp, lr, rp, rr], dtype=np.float32)
            self._cur = self._target.copy()
            self._releasing = False
            self._wb_latched = True

    def clear(self, force: bool = False) -> None:
        with self._lock:
            if getattr(self, "_wb_latched", False) and not force:
                return
            self._wb_latched = False
            self._target = None
            self._releasing = self._cur is not None

    def active(self) -> bool:
        with self._lock:
            return self._target is not None or self._releasing

    def apply(self, payload: dict) -> bool:
        with self._lock:
            tgt = self._target
            releasing = self._releasing
        jpos = payload.get("joint_pos_mj")
        if jpos is None or jpos.shape != (31,):
            return False
        idx = [self._IDX[k] for k in ("lp", "lr", "rp", "rr")]
        planner = np.asarray(jpos[idx], dtype=np.float32)
        if tgt is None and not releasing:
            return False
        if self._cur is None:
            self._cur = planner.copy()
        goal = planner if tgt is None else tgt
        step = self.WRIST_SLEW_RAD_PER_FRAME
        self._cur = self._cur + np.clip(goal - self._cur, -step, step)
        if tgt is None and np.abs(self._cur - planner).max() < 0.01:
            with self._lock:
                self._releasing = False
            self._cur = None
            return False
        jpos = jpos.copy()
        jpos[idx] = self._cur
        payload["joint_pos_mj"] = jpos
        fut = payload.get("joint_pos_mj_future")
        if fut is not None and fut.ndim == 2 and fut.shape[1] == 31:
            fut = fut.copy()
            fut[:, idx] = self._cur
            payload["joint_pos_mj_future"] = fut
        return True


_WRIST_REF: list = [None]
_WAIST_REF: list = [None]
# Served yaw rate EMA (rad/s), written by the publisher loop, read by the
# replan worker's in-place yaw governor.
_SERVED_YAW_RATE_REF: list = [0.0]
# Terminal operator e-stop latch: once set, every outgoing pose payload
# carries an ``estop`` field (the deploy slams Kp=0/Kd=8 on it; unknown
# fields are ignored by older deploys). Cleared by a planner restart or by
# the operator repeating the full damping gesture >= 5 s later (phase
# "clear", 2026-09-23): the latch drops and every payload then carries an
# explicit ``estop = 0`` so the deploy unlatches too. An ABSENT field never
# clears it, so a wire flap cannot lift an e-stop.
_ESTOP_LATCH: list = [False]
_ESTOP_CLEARED: list = [False]
_DANCE_QUEUE_REF: list = [None]
# Persistent yaw-governor trim. Init = calibrated prior (measured 2026-08-03:
# templates over-turn ~1.85x commanded, so the FIRST plan of a turn is
# already compensated); adapts multiplicatively toward cmd/served each
# replan and persists across turns (the miscalibration is a model property,
# not per-episode).
_YAW_TRIM_STATE: list = [0.55]
# Model-stop decel target override (G1's fixed decay law, full_agent.py:214:
# idle-mode target = current speed / 2 per second, hard zero below the
# cutoff). While STOPPING the 50 Hz loop keeps this at half the served
# signed forward speed and the replan worker substitutes it for the idle
# intent, so each successive plan decelerates exponentially instead of
# slamming to zero. None = inactive (worker uses the intent verbatim).
_MODEL_STOP_DECEL_REF: list = [None]


def _pc3_announce_estop(phase: str) -> None:
    """Fire-and-forget spoken e-stop cue on the PC3 robot speaker.

    PC3_HOST (env; unset = no-op) names the speaker host the planner can
    reach directly from PC2; the WAVs are pre-staged at
    /opt/x2_interact/audio/ (48 kHz dmix format). PC3_SSH_PASSWORD (env)
    goes through sshpass when set; otherwise key-based ssh. Non-blocking
    Popen, every failure swallowed -- audio is decorative, the e-stop path
    must never wait on it. On a laptop sim run PC3_HOST is unset and
    nothing happens.
    """
    host = os.environ.get("PC3_HOST", "")
    if not host:
        return
    wav = ("estop_damping.wav" if phase == "damp"
           else "estop_activating.wav")
    try:
        import subprocess
        pw = os.environ.get("PC3_SSH_PASSWORD", "")
        subprocess.Popen(
            (["sshpass", "-p", pw] if pw else []) + ["ssh",
             "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             f"{os.environ.get('PC3_USER', 'agi')}@{host}",
             f"aplay -D playback_def /opt/x2_interact/audio/{wav}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        pass


class PosePublisher:
    """PUB bind + v4 packed encoder (port of x2_kplanner.PosePublisher,
    defaulting to bind-on-all-interfaces for the PC2 deployment)."""

    def __init__(self, host: str, port: int, topic: str = "pose") -> None:
        import zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{port}")
        self._topic = topic
        # Optional VR arm/hand overlay (hop-in manipulation). Wired by main
        # when --arm-port > 0; dance_active suspends it so clips own the
        # whole body (arms return to the held operator pose afterwards).
        self.arm_ingest: Optional["ArmTargetIngest"] = None
        self.dance_active: Optional[threading.Event] = None
        time.sleep(0.1)

    # NOTE (2026-08-12): a serve-path root-z offset was tried here to ground
    # the reference feet (operator design: shift SONIC's view, not the
    # planner loop). DEAD END: the C++ deploy never reads root_z_world —
    # reference height reaches SONIC only through LEG JOINT configuration.
    # Grounding therefore requires joint-space correction = the S1 learned
    # decoder's foot-band objective. Do not re-add a z knob here.

    def publish(self, payload: dict[str, np.ndarray]) -> None:
        if self.arm_ingest is not None and not (
            self.dance_active is not None and self.dance_active.is_set()
        ):
            applied = self.arm_ingest.overlay(payload)
            if applied != getattr(self, "_overlay_state", None):
                log.info("arm overlay %s",
                         "ACTIVE (operator arm/hand targets on the wire)"
                         if applied else "cleared (planner arms restored)")
                self._overlay_state = applied
        if _ESTOP_LATCH[0]:
            payload["estop"] = np.asarray([1.0], dtype=np.float32)
        elif _ESTOP_CLEARED[0]:
            payload["estop"] = np.asarray([0.0], dtype=np.float32)
        waist = _WAIST_REF[0]
        if waist is not None:
            w_applied = waist.step_and_apply(payload, 1.0 / OUTPUT_FPS)
            if w_applied != getattr(self, "_waist_state", None):
                log.info("waist overlay %s",
                         "ACTIVE" if w_applied else "cleared (waist restored)")
                self._waist_state = w_applied
        wrist = _WRIST_REF[0]
        if wrist is not None and not (
            self.dance_active is not None and self.dance_active.is_set()
        ):
            wr_applied = wrist.apply(payload)
            if wr_applied != getattr(self, "_wrist_state", None):
                log.info("wrist overlay %s",
                         "ACTIVE (operator SMPL wrists on the wire)"
                         if wr_applied else "cleared (planner wrists restored)")
                self._wrist_state = wr_applied
        # the frame as it went on the WIRE (after the arm / waist / wrist
        # overlays) -- the tape records this, not the planner's raw frame
        # (corpus capture 2026-09-05: the latched-arm walks must be recorded
        # with the arms the deploy actually saw)
        self.last_wire_jpos = payload.get("joint_pos_mj")
        self._sock.send(pack_pose_message(payload, topic=self._topic, version=4))
        self._tape_seq = getattr(self, "_tape_seq", -1) + 1
        _TAPE.ev("tick", seq=self._tape_seq)

    def close(self) -> None:
        self._sock.close(linger=0)


# ---------------------------------------------------------------------------
# Quaternion helpers (numpy; conventions match wire.py / blending.py).
# ---------------------------------------------------------------------------


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def _yaw_of_quat_xyzw(q: np.ndarray) -> float:
    return yaw_from_quat_wxyz(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))


# Last published reference yaw + its wall time, for the global rate clamp below.
_REF_YAW_PREV: list = [None, None]   # [yaw_rad, monotonic_s]


def _clamp_ref_yaw_step(q_xyzw: np.ndarray) -> np.ndarray:
    """Bound the tick-to-tick change of the PUBLISHED reference heading.

    The 2026-08-09 turn-burst incident: the stop-blend branch published
    reference yaw rates up to 28 rad/s (p99 23, 64% of its ticks over
    1.5 rad/s) because it eased joints but stepped the heading. SONIC chases
    ``rel = inv(measured) * reference``, so a heading step becomes a hip snap;
    repeated bursts compounded it into a fall.

    This is a LAST-RESORT clamp, not the fix -- the stop-blend SLERP is. It
    exists so any future path that steps the heading degrades into a fast slew
    instead of a snap. At the default 8.0 rad/s it is ~11x the fastest commanded
    turn (0.70), so legitimate motion never reaches it.

    Disabled by ``KPLANNER_MAX_REF_YAW_RAD_S=0``. First call is a no-op (no
    previous sample to rate-limit against).
    """
    if MAX_REF_YAW_RAD_S <= 0.0:
        return q_xyzw
    now = time.monotonic()
    yaw = _yaw_of_quat_xyzw(q_xyzw)
    prev_yaw, prev_t = _REF_YAW_PREV
    if prev_yaw is not None and prev_t is not None:
        dt = now - prev_t
        # MUST be called on EVERY publish path (PLAYING and IDLE), else this
        # state goes stale across a whole PLAYING segment and dt lands in the
        # seconds -- which is precisely the PLAYING -> IDLE transition this
        # clamp exists to bound. First cut only clamped the IDLE path and so
        # was bypassed at the transition: measured 3.7% of stop-blend ticks
        # still over 8 rad/s, peak 29.75 (2026-08-09 run 2).
        # 0.25 s = 12 ticks @ 50 Hz: past that we have lost continuity and a
        # "rate" is meaningless, so re-seed instead of inventing a limit.
        if 0.0 < dt <= 0.25:
            d = _wrap_pi(yaw - prev_yaw)
            lim = MAX_REF_YAW_RAD_S * dt
            if abs(d) > lim:
                yaw = _wrap_pi(prev_yaw + math.copysign(lim, d))
                q_xyzw = _wxyz_to_xyzw(_quat_wxyz_from_yaw(yaw))
    _REF_YAW_PREV[0] = yaw
    _REF_YAW_PREV[1] = now
    return q_xyzw


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _quat_wxyz_from_yaw(yaw_rad: float) -> np.ndarray:
    """``R_z(yaw)`` packed as (qw, qx, qy, qz).

    Yaw-only by construction: pitch/roll are dropped on purpose, mirroring
    ``x2_kplanner._yaw_only_wxyz_from_pelvis`` -- a transient leg lean (fall
    recovery, slip) must not bleed into the published reference and pull SONIC
    outside its upright-reference training distribution.
    """
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


# Closed-loop idle yaw resync (port of x2_kplanner.py:3094).
# Max age of a measured-yaw sample we will trust; matches x2_kplanner's
# ``pose_feedback_max_age_s`` default. Stale -> hold last good, never revert to
# identity (identity == world +X is a KNOWN-WRONG heading the deploy actively
# twists the body toward).
POSE_FEEDBACK_MAX_AGE_S: float = 0.5
# Rate cap on how fast the reference yaw may be re-anchored. Defence in depth:
# resync is an ASSIGNMENT (no feedback term, so it cannot wind up), but a cap
# means even a wrong sign or frame error can only leak slowly instead of
# spinning the robot. See the 2026-07-18 runaway-spin incident.
MAX_YAW_RESYNC_RAD_S: float = 1.5

# PLAYING -> IDLE stop blend length. 16 ticks @ 50 Hz = 320 ms -- long enough to
# take a 1.2 rad snap down to ~0.075 rad/tick, short enough that the robot still
# settles promptly when the operator releases.
# Env-overridable (2026-08-09) so the blend length can be A/B'd without an edit:
# the turn-burst incident needed exactly that and there was no knob.
STOP_BLEND_FRAMES: int = max(1, int(os.environ.get("KPLANNER_STOP_BLEND_FRAMES") or 16))
# Upper clamp of the knee-scaled anchor blend for ordinary walking stops
# (0.8 s), and the wider clamp used only when the cut pose is a crouch
# (worst-knee delta > _CROUCH_STAND_DELTA_RAD, i.e. deeper than any walking
# gait frame): 200 ticks = 4.0 s at 50 Hz, the ceiling of the rate-based
# crouch exit below.
STOP_BLEND_MAX_FRAMES: int = max(STOP_BLEND_FRAMES, int(os.environ.get("KPLANNER_STOP_BLEND_MAX_FRAMES") or 40))
CROUCH_STAND_BLEND_MAX_FRAMES: int = max(STOP_BLEND_MAX_FRAMES, int(os.environ.get("KPLANNER_CROUCH_STAND_BLEND_MAX_FRAMES") or 200))
# OPT-IN slower crouch exit: knee-extension rate in rad/s (0 = off, the
# walking blend cap applies). The robot's crouch stops on 2026-09-09
# (frozen core + G1-core planner, six walks) were all clean under the
# walking cap, so that stays the default; the sim twin tips at those exits
# regardless (it cannot hold the deep crouch the robot holds, see
# docs/x2/F01_gamepad.md), so the knob exists for sim work and for a robot
# trial from a deeper draw, e.g. KPLANNER_CROUCH_STAND_RATE_RAD_S=0.5.
CROUCH_STAND_RATE_RAD_S: float = float(os.environ.get("KPLANNER_CROUCH_STAND_RATE_RAD_S") or 0.0)
_CROUCH_STAND_DELTA_RAD: float = 0.6

# Optional hard clamp on how fast the PUBLISHED reference heading may move.
#
# DEFAULT OFF (0.0), and that default is load-bearing -- do not re-enable it
# without re-measuring. It first shipped at 8.0 on the reasoning that 8 rad/s is
# ~11x the fastest commanded turn (0.70) and so could only ever catch step
# discontinuities. **That reasoning was wrong.** Measured on a replayed operator
# session (replay_intents_to_daemon.py, 2026-08-09), the reference legitimately
# moves far faster than the COMMAND at replan seams:
#
#     planner arm            ring/planner p99   max
#     round-4 (pose 100k)          11.67       12.45
#     base    (pose 500k)          17.32       17.82
#
# So a 8.0 ceiling rate-limits ~9% of PLAYING ticks -- it mangles normal turning
# rather than guarding it, and the operator reported exactly that ("walking
# turns bad, in-place turns sluggish") on a command line they had not changed.
# The real transition-step fix is the stop-blend heading SLERP below; this clamp
# is not needed for it.
#
# Enable deliberately (e.g. KPLANNER_MAX_REF_YAW_RAD_S=25) only to catch the
# ~28-30 rad/s outliers, and only after checking it clears the current p99.
MAX_REF_YAW_RAD_S: float = float(
    os.environ.get("KPLANNER_MAX_REF_YAW_RAD_S") or 0.0)

# Rigidly yaw/XY-align each newly committed plan to the frame being served at
# the seam. This is the ROOT-CAUSE fix for the reference-yaw spikes (100% of
# >5 rad/s spikes land within 100 ms of a replan); the 8-tick cross-fade was
# being asked to absorb a heading offset instead of just smoothing content.
# Preserves the plan's internal yaw profile exactly -- a rigid rotation cannot
# change frame-to-frame deltas -- so commanded turns execute unchanged.
# DEFAULT ON -- this is the fix for the turn-burst collapse.
# Controlled live A/B, deployed HF models, the SAME recorded operator inptus
# replayed into both arms, only this flag differing:
#     ON   200 s, tilt max 12.2 deg, z_min 0.608  -> SURVIVED
#     OFF  127 s, tilt max 71.9 deg, z_min 0.368  -> COLLAPSED at 126 s
# Mechanism: a heading step at a replan seam is a PERMANENT offset, not a
# transient. SONIC chases it (rel = inv(measured) * reference), each burst
# forces another replan and another step, and the accumulated chase shows up
# as sustained ~2.3 rad/s executed yaw against a commanded 0.55 -- tilt then
# ratchets past 13 deg and the robot goes down.
# NOTE: an earlier OFFLINE A/B (replay_intents_to_daemon.py) called this WORSE
# in 3/3 reps. That harness runs the planner with no rebase, no SONIC and no
# closed loop, so nothing chases the reference and the mechanism is invisible.
# Do not trust that harness for anything involving tracking response.
# KPLANNER_SEAM_YAW_ALIGN=0 restores the old behaviour.
SEAM_YAW_ALIGN: bool = (os.environ.get("KPLANNER_SEAM_YAW_ALIGN", "1") != "0")

# Minimum wall-clock gap between FORCED replans (IDLE->PLAYING reseeds), in ms.
# Mid-walk intent changes do not replan (see the module docstring) -- but every
# stick press is an IDLE->PLAYING transition and DOES force one. A rapid burst
# sequence therefore forces one replan per press, and each replan lays down a
# seam. Measured 2026-08-09: 211 replans in 80 s (2.6/s, i.e. every ~380 ms,
# faster than the MIN_SERVED=16-native-frame floor of ~530 ms), and 100% of
# reference-yaw spikes >5 rad/s landed within 100 ms of a replan_done. Because
# a seam step is a PERMANENT heading offset rather than a transient, those
# offsets accumulate: executed yaw reached a sustained 2.3 rad/s against a
# commanded 0.55 before the robot tilted past 13 deg and fell.
#
# Debouncing trades responsiveness for seam churn: a suppressed force means the
# press waits up to this long before the plan starts. 0 = off (previous
# behaviour). Suppression is DEFERRED, never dropped -- the force flag is left
# set so the next loop iteration honours it.
MIN_FORCED_REPLAN_MS: float = float(
    os.environ.get("KPLANNER_MIN_FORCED_REPLAN_MS") or 0.0)

# Master gate for the 2026-08-09 stop-blend changes (heading SLERP + the
# blend-completes off-by-one). Both are believed-correct bug fixes, but they
# are UNCONDITIONAL and therefore also change the battle-tested
# 5252 + base-500k configuration -- so a clean regression baseline needs a way
# to switch them off. KPLANNER_LEGACY_STOP_BLEND=1 restores the exact
# pre-2026-08-09 behaviour (joints-only blend, w topping out at 1-1/N).
LEGACY_STOP_BLEND: bool = (os.environ.get("KPLANNER_LEGACY_STOP_BLEND") == "1")


# ---------------------------------------------------------------------------
# Dance playback
# ---------------------------------------------------------------------------


_PRIM_QUEUE: "queue.Queue" = queue.Queue()


class PrimitivePlayback:
    """Crouch/lean primitive playback: ENTER (play clip forward) -> HOLD
    (freeze at last frame) -> EXIT (play backward to frame 0) -> done.

    Primitives come from x2_planner_primitives.pkl (dof [T,31],
    root_rot_xyzw [T,4], root_trans [T,3], fps). Root handling differs
    from DancePlayback: xy holds at entry values (primitives are
    in-place), yaw is rebased like dance, and z streams as a DELTA from
    the clip's first frame on top of the serve-time root z -- crouches
    genuinely lower the pelvis without trusting the clip's absolute
    height calibration. The model's hip_h intent channel is dead
    (measured 2026-08-04: torch + both ONNX graphs root_z-invariant for
    hip 0.50-0.72), so clip playback is the only crouch path that works
    with the deployed models."""

    def __init__(
        self,
        name: str,
        clip: dict,
        current_yaw_rad: float,
        base_root_z: float,
        mode: str = "oneshot",
    ) -> None:
        self.name = name
        self.mode = mode          # "oneshot" | "scrub" | "step"
        self._depth = 0.0         # scrub target in [0,1] (0=stand, 1=peak)
        self._dof = np.ascontiguousarray(clip["dof"], dtype=np.float32)
        quat = np.ascontiguousarray(clip["root_rot_xyzw"], dtype=np.float32)
        clip_yaw0 = _yaw_of_quat_xyzw(np.asarray(quat[0], dtype=np.float64))
        delta = float(current_yaw_rad) - float(clip_yaw0)
        self._quat = rebase_quats_xyzw_by_yaw(quat, delta)
        z = np.asarray(clip["root_trans"], dtype=np.float64)[:, 2]
        self._z = (float(base_root_z) + (z - z[0])).astype(np.float64)
        # Root XY deltas (clip frame -> serve frame): rotate the clip's
        # translation track by the same yaw rebase applied to the quats.
        # In-place primitives (crouch/lean) have ~zero track; "step"
        # primitives (side steps) genuinely move the root and the loop
        # persists final_xy_delta into current_root_xy on completion.
        xy = np.asarray(clip["root_trans"], dtype=np.float64)[:, :2]
        rel = xy - xy[0]
        c_d, s_d = math.cos(delta), math.sin(delta)
        self._xy = np.stack(
            [c_d * rel[:, 0] - s_d * rel[:, 1],
             s_d * rel[:, 0] + c_d * rel[:, 1]], axis=1)
        self._fps = float(clip.get("fps", 30.0)) or 30.0
        self._n = int(self._dof.shape[0])
        self._phase = 0.0
        self.state = {"scrub": "scrub", "step": "step"}.get(mode, "enter")
        # Hold frame = peak displacement from the start pose, NOT the last
        # frame: the primitive bakes are full cycles (crouch down AND back
        # up; z returns to stand by the final frame), so ENTER plays to the
        # peak, HOLD freezes there, EXIT reverses peak -> frame 0.
        disp = np.linalg.norm(
            self._dof.astype(np.float64) - self._dof[0].astype(np.float64),
            axis=1,
        ) + 5.0 * np.abs(z - z[0])   # weight root drop so crouches pin to z-min
        self._hold_idx = int(np.argmax(disp))
        if self._hold_idx < 1:
            self._hold_idx = self._n - 1
        log.info("primitive %s: %d frames @ %.1f fps, hold@%d "
                 "(dz_hold=%+.3f m), yaw rebase %+.1f deg", name, self._n,
                 self._fps, self._hold_idx,
                 float(z[self._hold_idx] - z[0]), math.degrees(delta))

    @property
    def done(self) -> bool:
        return self.state == "done"

    def set_depth(self, depth: float) -> None:
        """Scrub mode: command the target depth in [0,1] (stick axis)."""
        self._depth = max(0.0, min(1.0, float(depth)))

    def xy_delta(self) -> np.ndarray:
        """Rotated root-XY offset of the current frame vs clip start."""
        return self._xy[self._idx()]

    def final_xy_delta(self) -> np.ndarray:
        return self._xy[-1]

    def exit(self) -> None:
        if self.mode == "scrub":
            self._depth = 0.0
        elif self.state in ("enter", "hold"):
            self.state = "exit"
            log.info("primitive %s: EXIT (reverse from frame %.1f)",
                     self.name, self._phase)

    def _idx(self) -> int:
        return max(0, min(int(round(self._phase)), self._n - 1))

    def tick(self) -> tuple[np.ndarray, np.ndarray, float,
                            list[np.ndarray], list[np.ndarray]]:
        """(jpos, quat_xyzw, root_z, future_jpos[9], future_quat[9]);
        advances one 50 Hz tick. Futures sample the remaining trajectory
        in the current direction, clamped at the trajectory end."""
        i = self._idx()
        jpos, quat, z = self._dof[i], self._quat[i], float(self._z[i])
        step_clip = self._fps / OUTPUT_FPS
        if self.state == "scrub":
            tgt_phase = self._depth * self._hold_idx
            direction = (0.0 if abs(tgt_phase - self._phase) < 1e-6
                         else (1.0 if tgt_phase > self._phase else -1.0))
            clamp_hi = self._hold_idx
        elif self.state == "step":
            direction = 1.0
            clamp_hi = self._n - 1
        else:
            direction = -1.0 if self.state == "exit" else 1.0
            clamp_hi = (self._hold_idx if self.state in ("enter", "hold")
                        else self._n - 1)
        fut_step = self._fps * FUTURE_DT_S * direction
        fut_j: list[np.ndarray] = []
        fut_q: list[np.ndarray] = []
        for k in range(1, NUM_FUTURE + 1):
            fi = max(0, min(int(round(self._phase + fut_step * k)), clamp_hi))
            fut_j.append(self._dof[fi])
            fut_q.append(self._quat[fi])
        if self.state == "scrub":
            if direction != 0.0:
                self._phase += step_clip * direction
                self._phase = max(0.0, min(self._phase, float(self._hold_idx)))
                if direction > 0 and self._phase > tgt_phase:
                    self._phase = tgt_phase
                elif direction < 0 and self._phase < tgt_phase:
                    self._phase = tgt_phase
            if self._depth <= 0.01 and self._phase <= 1e-6:
                self.state = "done"
                log.info("primitive %s: scrub returned to stand; planner "
                         "resumes", self.name)
        elif self.state == "step":
            self._phase += step_clip
            if self._phase >= self._n - 1:
                self._phase = float(self._n - 1)
                self.state = "done"
                log.info("primitive %s: step complete (root moved %+.3f, "
                         "%+.3f m)", self.name, self._xy[-1][0], self._xy[-1][1])
        elif self.state == "enter":
            self._phase += step_clip
            if self._phase >= self._hold_idx:
                self._phase = float(self._hold_idx)
                self.state = "hold"
                log.info("primitive %s: HOLD (crouch/lean held; send "
                         "release or any locomotion intent to exit)", self.name)
        elif self.state == "exit":
            self._phase -= step_clip
            if self._phase <= 0.0:
                self._phase = 0.0
                self.state = "done"
                log.info("primitive %s: exit complete; planner resumes",
                         self.name)
        return jpos, quat, z, fut_j, fut_q


def _load_primitives_pkl(path: Path) -> dict:
    try:
        import joblib  # noqa: WPS433
        return joblib.load(path)
    except ImportError as exc:
        raise RuntimeError(
            "x2_planner_primitives.pkl is joblib-format; joblib is not "
            "importable in this venv (pip install joblib)"
        ) from exc


class DancePlayback:
    """Streams a loaded x2m2 clip through the 50 Hz publisher clock.

    x2m2 bakes carry (dof [T,31], quat_xyzw [T,4], fps); no root
    translation, so root xy/z hold at the values streamed when the dance
    started (same convention as the fallback ladder's idle-clip replay).
    The clip's own fps may differ from 50 (some bakes are 120 fps); a
    float phase accumulator advances ``fps / OUTPUT_FPS`` clip frames per
    wire tick so playback stays real-time with nearest-frame sampling.

    Yaw rebase: all root quats are pre-multiplied by
    ``R_z(current_yaw - clip_frame0_yaw)`` at start (delta form of the
    fallback ladder's ``build_idle_frame_msg`` rebase, which assumes
    yaw-0-aligned clips; dance bakes may start at arbitrary yaw) so the
    dance's heading track starts exactly at the robot's current streamed
    heading and evolves with the clip's authored yaw motion.
    """

    def __init__(
        self,
        dof: np.ndarray,
        quat_xyzw: np.ndarray,
        fps: float,
        current_yaw_rad: float,
        name: str,
        entry_jpos: Optional[np.ndarray] = None,
        entry_blend_ticks: int = 0,
    ) -> None:
        self.name = name
        self._dof = np.ascontiguousarray(dof, dtype=np.float32)
        clip_yaw0 = _yaw_of_quat_xyzw(np.asarray(quat_xyzw[0], dtype=np.float64))
        delta = float(current_yaw_rad) - float(clip_yaw0)
        self._quat = rebase_quats_xyzw_by_yaw(
            np.ascontiguousarray(quat_xyzw, dtype=np.float32), delta
        )
        self._fps = float(fps) if fps and fps > 0 else OUTPUT_FPS
        self._phase = 0.0
        self._n = int(dof.shape[0])
        # Entry blend (2026-08-11 "transition click" fix): half-cosine the
        # JOINTS from the last served pose to clip frame 0 before the clip
        # rolls. The shared ReferenceStepSmoother only ramps the LOWER BODY,
        # so the arm step (idle anchor -> raised guard / dance opening) hit
        # the wire in one tick -- inaudible while the 8 Hz arm LPF smeared
        # it, an audible click since bigrun freed the arms. Heading needs no
        # blend here: the yaw rebase above already makes frame 0 continuous
        # with the served heading (pitch/roll of a standing clip start is
        # negligible). Exit uses the anchor tick's existing stop-blend.
        self._entry_from = (
            None if entry_jpos is None
            else np.asarray(entry_jpos, dtype=np.float32).copy())
        self._entry_total = max(1, int(entry_blend_ticks))
        self._entry_left = (int(entry_blend_ticks)
                            if self._entry_from is not None else 0)
        log.info(
            "dance %s: %d frames @ %.1f fps (%.1fs), yaw rebase %+.1f deg, "
            "entry blend %d ticks",
            name, self._n, self._fps, self._n / self._fps,
            math.degrees(delta), self._entry_left,
        )

    @property
    def finished(self) -> bool:
        return self._phase >= self._n - 1

    def _frame_at(self, clip_idx: float) -> tuple[np.ndarray, np.ndarray]:
        i = max(0, min(int(round(clip_idx)), self._n - 1))
        return self._dof[i], self._quat[i]

    def tick(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Emit (jpos, quat, future_jpos[9], future_quat[9]) and advance one
        50 Hz tick. Future slots sample the clip at +0.1 s spacing, clamped
        at the final frame (clip ends -> hold last pose in the horizon).
        While the entry blend runs, the clip clock is HELD at frame 0 and
        the served joints sweep last-served -> frame 0 on a half-cosine
        (inclusive schedule: w reaches exactly 1 on the final blend tick,
        same convention as the anchor stop-blend's 2026-08-09 fix)."""
        jpos, quat = self._frame_at(self._phase)
        if self._entry_left > 0 and self._entry_from is not None:
            done = 1.0 - ((self._entry_left - 1)
                          / float(max(1, self._entry_total - 1)))
            w = 0.5 * (1.0 - math.cos(math.pi * done))
            jpos = ((1.0 - w) * self._entry_from + w * jpos).astype(
                np.float32)
            self._entry_left -= 1
        step_clip = self._fps * FUTURE_DT_S
        fut_j: list[np.ndarray] = []
        fut_q: list[np.ndarray] = []
        for k in range(NUM_FUTURE):
            fj, fq = self._frame_at(self._phase + (k + 1) * step_clip)
            fut_j.append(fj)
            fut_q.append(fq)
        if self._entry_left <= 0:
            self._phase += self._fps / OUTPUT_FPS
        return jpos, quat, fut_j, fut_q


def _resolve_dance_x2m2(
    dances_dir: Path, pkl: Optional[str], motion_key: Optional[str]
) -> Optional[Path]:
    """<dances-dir>/<motion_key>.x2m2, falling back to the pkl stem."""
    candidates = []
    if motion_key:
        candidates.append(dances_dir / f"{motion_key}.x2m2")
    if pkl:
        candidates.append(dances_dir / f"{Path(pkl).stem}.x2m2")
    for cand in candidates:
        if cand.is_file():
            return cand
    log.error(
        "motion_clip_cmd play: no x2m2 bake found (tried %s)",
        [str(c) for c in candidates],
    )
    return None


# ---------------------------------------------------------------------------
# Command source threads
# ---------------------------------------------------------------------------


def _zmq_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
    bind: bool = False,
) -> None:
    """SUB planner_cmd (port of x2_kplanner._zmq_command_thread minus the
    waist fields the pad bridge never sends -- hip_height/direct_velocity
    passthroughs kept for x2_pkl_command_source compatibility).

    ``bind=True`` flips the SUB to bind so MULTIPLE command sources
    (pad bridge + quest3 manager, each PUB-connect) can coexist —
    two PUBs cannot share one bound port, but one bound SUB accepts
    any number of connected PUBs."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    if bind:
        sock.bind(f"tcp://*:{port}")
        log.info("planner_cmd source: SUB bind %r on tcp://*:%d", topic, port)
    else:
        sock.connect(f"tcp://{host}:{port}")
        log.info("planner_cmd source: SUB %r on tcp://%s:%d", topic, host, port)
    try:
        while not stop_event.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                # No message this cycle (200 ms tick). VR-silence release
                # must NOT depend on pad traffic: a killed/disengaged
                # manager with an idle pad would otherwise hold ownership
                # (and the arm overlay) forever.
                if (_cmd_owner["src"] == "vr"
                        and time.monotonic() - _cmd_owner["vr_ts"]
                        > _VR_OWNER_TIMEOUT_S):
                    log.warning(
                        "cmd owner: VR silent %.1fs > %.1fs — releasing to "
                        "idle (arms restored unless WB-latched); pad may re-acquire",
                        time.monotonic() - _cmd_owner["vr_ts"],
                        _VR_OWNER_TIMEOUT_S)
                    _cmd_owner["src"] = None
                    if _ARM_INGEST_REF[0] is not None:
                        _ARM_INGEST_REF[0].clear()
                    cmd_queue.put(LocomotionCommand(
                        intent="idle", magnitude="default"))
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
                intent = str(payload["intent"])
                if intent == "shutdown":
                    log.info("planner_cmd source: shutdown received")
                    stop_event.set()
                    continue
                if intent == "wb_state":
                    # Whole-body engagement keepalive from the PC2 token
                    # service.
                    # Pure state, no ownership change: the pad keeps its
                    # slot; the anchor tick decides shadow/hold/return.
                    eng = bool(int(payload.get("engaged", 0)))
                    if eng != _WB_LINK["engaged"]:
                        log.info("wb_state: whole-body %s (from %s)",
                                 "ENGAGED" if eng else "released",
                                 payload.get("source", "?"))
                    _WB_LINK["engaged"] = eng
                    _WB_LINK["ts"] = time.monotonic()
                    _WB_LINK["svc_ts"] = time.monotonic()   # token service present
                    continue
                if intent == "arm_release":
                    # Explicit operator release of arms latched by WB HOLD
                    # (slews back to the planner's arms, never snaps).
                    if _ARM_INGEST_REF[0] is not None:
                        _ARM_INGEST_REF[0].clear(force=True)
                        log.info("arm_release: latched arms slewing back to "
                                 "planner arms")
                    if _WRIST_REF[0] is not None:
                        _WRIST_REF[0].clear(force=True)
                    continue
                if intent == "wrist_targets":
                    # Whole-body teleop wrists from the PC2 token service
                    # (operator SMPL wrist rotation -> X2 pitch/roll). Pure
                    # overlay, no ownership change.
                    if _WRIST_REF[0] is not None:
                        if payload.get("release"):
                            _WRIST_REF[0].clear()
                        else:
                            _WRIST_REF[0].set_target(
                                float(payload.get("lp", 0.0)), float(payload.get("lr", 0.0)),
                                float(payload.get("rp", 0.0)), float(payload.get("rr", 0.0)))
                    continue
                if intent == "head_targets":
                    # Whole-body teleop head yaw (pico_intent_sender, the
                    # operator's SMPL neck+head, 2026-09-08): an ABSOLUTE
                    # target in rad on the same slew-limited overlay the pad
                    # head-look uses (DOF 29, cap _HEAD_YAW_MAX_RAD, applied
                    # in the idle-anchor / WB shadow-hold and gait branches).
                    # Pure overlay, no ownership change; release recentres.
                    if payload.get("release"):
                        _HEAD_YAW["target"] = 0.0
                        _HEAD_SRC["ts"] = 0.0
                    else:
                        y = float(payload.get("yaw_rad", 0.0))
                        _HEAD_YAW["target"] = max(-_HEAD_YAW_MAX_RAD,
                                                  min(_HEAD_YAW_MAX_RAD, y))
                        _HEAD_SRC["ts"] = time.monotonic()
                    continue
                if intent == "estop":
                    # E-STOP OUTRANKS SOURCE OWNERSHIP. Two phases (operator
                    # design 2026-08-04):
                    #   soft (first trip): abort dance/primitive/locomotion
                    #     -> idle stand. Recoverable; drive again after.
                    #   damp (gesture continued ~1s more): additionally latch
                    #     the wire estop flag -> rebuilt deploy slams
                    #     Kp=0/Kd=8. TERMINAL until stack restart.
                    phase = str(payload.get("phase",
                                payload.get("magnitude", "soft")))
                    if phase == "clear":
                        # Operator repeated the full gesture (pad bridge
                        # enforces the >= 5 s gap and a fresh chord). Lower
                        # the wire flag; nothing moves -- the deploy stays
                        # limp in SAFE_HOLD until its RECOVER gate sees the
                        # robot held upright and still, or the stack restarts.
                        log.critical("E-STOP CLEARED from %s -- wire flag "
                                     "lowered (was %s); deploy may RECOVER "
                                     "when held upright",
                                     payload.get("source", "?"),
                                     "latched" if _ESTOP_LATCH[0] else "clear")
                        _ESTOP_LATCH[0] = False
                        _ESTOP_CLEARED[0] = True
                        continue
                    if phase not in ("soft", "damp"):
                        phase = "soft"
                    log.critical("E-STOP (%s) from %s — aborting dance/"
                                 "primitive/locomotion%s", phase,
                                 payload.get("source", "?"),
                                 " + WIRE DAMP FLAG (terminal)"
                                 if phase == "damp" else "")
                    if phase == "damp":
                        _ESTOP_LATCH[0] = True
                        _ESTOP_CLEARED[0] = False
                    _pc3_announce_estop(phase)
                    if _DANCE_QUEUE_REF[0] is not None:
                        _DANCE_QUEUE_REF[0].put(("stop",))
                    _PRIM_QUEUE.put(("exit",))
                    cmd_queue.put(LocomotionCommand(
                        intent="idle", magnitude="default", source="estop"))
                    _CROUCH["on"] = False
                    continue
                if intent == CROUCH_MODE_INTENT:
                    # Latched by the pad chord (L2+R2 + D-pad down/up); the main loop gates the ENTRY on a
                    # still, standing robot and logs a refusal otherwise. Never a locomotion command by itself.
                    cmd_queue.put(LocomotionCommand(
                        intent=CROUCH_MODE_INTENT,
                        magnitude="on" if bool(payload.get("enable", False)) else "off",
                        source=str(payload.get("source", "pad"))))
                    continue
                # ---- source ownership (mutual exclusion) --------------------
                # Multiple PUBs may be connected (pad bridge + VR manager),
                # but exactly ONE source owns the robot at a time. VR
                # SUPERSEDES pad: any VR message takes ownership and pad
                # messages are dropped until VR explicitly releases (idle on
                # disengage) or goes silent past the crash timeout (the
                # manager keepalives every 0.5 s while engaged, so a quiet
                # held stick cannot false-expire). On timeout release the
                # planner idles; the pad re-acquires with its next message.
                src = str(payload.get("source", "pad"))
                now_own = time.monotonic()
                if src == "vr":
                    _cmd_owner["vr_ts"] = now_own
                    if payload.get("vr_release"):
                        # Explicit disengage from the manager: the ONLY
                        # VR message that hands control back. A plain VR
                        # idle is a stand-still command and keeps
                        # ownership (sticks centered != disengage).
                        if _cmd_owner["src"] == "vr":
                            log.warning("cmd owner: VR released — "
                                        "pad may re-acquire")
                            _cmd_owner["src"] = None
                            if _ARM_INGEST_REF[0] is not None:
                                _ARM_INGEST_REF[0].clear()
                        elif _cmd_owner["src"] == "pad":
                            continue  # stray release while pad drives
                    else:
                        if _cmd_owner["src"] != "vr":
                            log.warning("cmd owner: VR ENGAGED — pad input ignored")
                            _cmd_owner["src"] = "vr"
                else:  # pad (or untagged legacy = pad)
                    if _cmd_owner["src"] == "vr":
                        if now_own - _cmd_owner["vr_ts"] > _VR_OWNER_TIMEOUT_S:
                            log.warning(
                                "cmd owner: VR silent %.1fs > %.1fs — releasing "
                                "to idle; pad re-acquires",
                                now_own - _cmd_owner["vr_ts"], _VR_OWNER_TIMEOUT_S)
                            _cmd_owner["src"] = None
                            if _ARM_INGEST_REF[0] is not None:
                                _ARM_INGEST_REF[0].clear()
                            cmd_queue.put(LocomotionCommand(
                                intent="idle", magnitude="default"))
                            continue  # this pad msg is dropped; next one owns
                        continue  # VR owns: drop pad message
                    if _cmd_owner["src"] is None:
                        _cmd_owner["src"] = "pad"
                magnitude = str(payload.get("magnitude", "default"))
                stick_fwd = float(payload.get("stick_fwd", 0.0))
                stick_side = float(payload.get("stick_side", 0.0))
                stick_yaw = float(payload.get("stick_yaw", 0.0))
                # Forward-obstacle clamp. Yaw is deliberately untouched so the
                # operator can turn away instead of being stuck facing a wall.
                if _guard_blocked():
                    if not _guard_latched["on"]:
                        log.warning("OBSTACLE %.2fm -> LATCHED; release the "
                                    "deadman to reset", _guard_state["dist"])
                    _guard_latched["on"] = True
                elif (stick_fwd == 0.0 and stick_side == 0.0
                      and stick_yaw == 0.0 and _guard_latched["on"]):
                    # deadman released (bridge sends an all-zero frame) -> reset
                    log.info("guard latch reset")
                    _guard_latched["on"] = False
                if _guard_latched["on"]:
                    stick_fwd = 0.0
                    stick_side = 0.0
                sd = payload.get("speed_delta")
                if sd:
                    _adjust_speed_setpoint(float(sd))
                # Head look: absent field recentres (waist contract) --
                # unless the whole-body head_targets feed is fresh.
                hy = payload.get("head_yaw_stick")
                if hy:
                    _HEAD_YAW["target"] = (
                        max(-1.0, min(1.0, float(hy))) * _HEAD_YAW_MAX_RAD)
                elif time.monotonic() - _HEAD_SRC["ts"] > _HEAD_SRC_TIMEOUT_S:
                    _HEAD_YAW["target"] = 0.0
                hip_height_raw = payload.get("hip_height_m", None)
                hip_height_m: Optional[float] = (
                    float(hip_height_raw) if hip_height_raw is not None else None
                )
                # ---- Waist overlay targets (v7.4 semantics: every command
                # message re-samples the target; absent fields recentre --
                # matches x2_kplanner's payload.get(..., 0.0) contract).
                if _WAIST_REF[0] is not None:
                    _WAIST_REF[0].set_target(
                        float(payload.get("waist_pitch_deg", 0.0)),
                        float(payload.get("waist_roll_deg", 0.0)),
                        float(payload.get("waist_yaw_deg", 0.0)),
                    )
                # ---- Primitive intents (crouch_*/lean_*/torso_*): route to
                # the playback queue; anything else while a primitive is
                # active triggers its exit (handled loop-side).
                if intent == PRIMITIVE_INTENT:
                    prim_name = str(payload.get("name", "") or "")
                    if prim_name in ("", "release", "exit"):
                        _PRIM_QUEUE.put(("exit",))
                    else:
                        _PRIM_QUEUE.put(("enter", prim_name))
                    continue
                target_velocity = payload.get("target_velocity")
                direct_velocity: Optional[tuple[float, float, float, float]] = None
                if target_velocity is not None:
                    if (
                        not isinstance(target_velocity, (list, tuple))
                        or len(target_velocity) != 4
                    ):
                        log.warning(
                            "planner_cmd: target_velocity must be a 4-list; "
                            "got %r (ignoring)", target_velocity,
                        )
                    else:
                        direct_velocity = tuple(float(v) for v in target_velocity)
                        # VR/Quest sends velocity directly, bypassing sticks.
                        if _guard_latched["on"] and direct_velocity[0] > 0.0:
                            log.warning("OBSTACLE %.2fm -> direct vx held",
                                        _guard_state["dist"])
                            direct_velocity = (0.0, 0.0,
                                               direct_velocity[2],
                                               direct_velocity[3])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("planner_cmd: bad payload %r: %s", parts[1], exc)
                continue
            _TAPE.ev("intent_recv", intent=intent, magnitude=magnitude,
                     stick_fwd=stick_fwd, stick_side=stick_side,
                     stick_yaw=stick_yaw, direct_velocity=direct_velocity)
            cmd_queue.put(
                LocomotionCommand(
                    intent=intent,
                    magnitude=magnitude,
                    source="zmq",
                    stick_fwd=stick_fwd,
                    stick_side=stick_side,
                    stick_yaw=stick_yaw,
                    direct_velocity=direct_velocity,
                    hip_height_m=hip_height_m,
                )
            )
    finally:
        sock.close(linger=0)


class MeasuredYaw:
    """Thread-safe latch for the robot's live IMU yaw (from x2_debug).

    Mirrors the watchdog's yaw-rebase source: the C++ deploy PUBs
    ``x2_debug`` with the pelvis ``base_quat``; we decode it to a yaw so
    the planner can rebase its published root quats to the robot's
    actual heading (else SONIC twists the body to world +X -- the
    orientation snap). ``value`` is None until the first frame lands.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._yaw: Optional[float] = None
        self._ts: float = 0.0
        self._jpos: Optional[np.ndarray] = None   # measured body_q, MJ order
        self._jpos_ts: float = 0.0

    def set(self, yaw: float) -> None:
        with self._lock:
            self._yaw = float(yaw)
            self._ts = time.monotonic()

    def get(self, max_age_s: float = 1.0) -> Optional[float]:
        with self._lock:
            if self._yaw is None:
                return None
            if time.monotonic() - self._ts > max_age_s:
                return None
            return self._yaw

    def set_jpos(self, jpos: np.ndarray) -> None:
        with self._lock:
            self._jpos = np.asarray(jpos, dtype=np.float32).copy()
            self._jpos_ts = time.monotonic()

    def set_quat(self, quat_wxyz: np.ndarray) -> None:
        with self._lock:
            self._quat = np.asarray(quat_wxyz, dtype=np.float64).copy()
            self._quat_ts = time.monotonic()

    def get_quat_xyzw(self, max_age_s: float = 0.5) -> Optional[np.ndarray]:
        """Measured base orientation (xyzw) or None if absent/stale. The
        deploy's reference-orientation feature is inv(measured)*reference
        per future frame (tokenizer_obs.cpp:185), so a reference published
        WITH the measured orientation is identity for the policy -- the
        yaw-only idle quat is not when the robot stands pitched."""
        with self._lock:
            q = getattr(self, "_quat", None)
            if q is None or time.monotonic() - getattr(self, "_quat_ts", 0.0) > max_age_s:
                return None
            return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)

    def get_jpos(self, max_age_s: float = 0.5) -> Optional[np.ndarray]:
        """Measured joints (31, MJ order) or None if absent/stale."""
        with self._lock:
            if self._jpos is None:
                return None
            if time.monotonic() - self._jpos_ts > max_age_s:
                return None
            return self._jpos.copy()


def _x2_debug_thread(
    measured: "MeasuredYaw",
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
) -> None:
    """SUB the deploy's x2_debug PUB; latch the measured pelvis yaw.

    Best-effort: transient decode failures are ignored (a misshapen
    frame must never wedge this thread). Absent on the laptop/sim
    (no C++ deploy) -- the latch simply stays None and rebase is a
    no-op there."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    log.info("x2_debug source: SUB %r on tcp://%s:%d (measured-yaw rebase)",
             topic, host, port)
    try:
        while not stop_event.is_set():
            try:
                raw = sock.recv()
            except zmq.error.Again:
                continue
            fields = decode_x2_debug_fields(
                raw, ("base_quat", "body_q", "wb_engaged"), topic)
            if not fields or "base_quat" not in fields:
                continue
            try:
                measured.set(yaw_from_quat_wxyz(fields["base_quat"].ravel()[:4]))
                measured.set_quat(fields["base_quat"].ravel()[:4])
                bq = fields.get("body_q")
                if bq is not None and bq.size == NUM_BODY_DOFS:
                    measured.set_jpos(bq.ravel())
                # Dual-head deploy (2026-09-04): engagement comes from the
                # deploy itself (no token service). Same latch the
                # planner_cmd wb_state keepalive fills on the legacy path.
                we = fields.get("wb_engaged")
                # -1 = "no information" (deploy not in dual-head mode: the
                # token service's wb_state keepalive owns the flag). Trusting
                # a legacy 0.0 here fought that keepalive at 50 Hz and made
                # the WB state machine flap shadow->hold->return (operator
                # session 2026-09-04 14:26). Also ignored whenever a token
                # service keepalive was seen in the last 2 s (a deploy binary
                # predating the -1 default still publishes 0.0).
                svc_recent = (time.monotonic() - _WB_LINK.get("svc_ts", -1e9)) < 2.0
                if (we is not None and we.size >= 1
                        and float(we.ravel()[0]) >= 0.0 and not svc_recent):
                    eng = bool(float(we.ravel()[0]) > 0.5)
                    if eng != _WB_LINK["engaged"]:
                        log.info("wb_engaged (x2_debug): whole-body %s",
                                 "ENGAGED" if eng else "released")
                    _WB_LINK["engaged"] = eng
                    _WB_LINK["ts"] = time.monotonic()
            except Exception:  # noqa: BLE001 -- never wedge on a bad frame
                continue
    finally:
        sock.close(linger=0)


def _motion_clip_cmd_thread(
    dance_queue: "queue.Queue[tuple]",
    port: int,
    topic: str,
    dances_dir: Path,
    stop_event: threading.Event,
) -> None:
    """SUB bind motion_clip_cmd; loads the x2m2 in-thread and enqueues the
    arrays so the publish loop never blocks on disk I/O."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.bind(f"tcp://*:{port}")
    log.info("motion_clip_cmd source: SUB bind %r on tcp://*:%d (dances=%s)",
             topic, port, dances_dir)
    try:
        while not stop_event.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
                action = str(payload.get("action", ""))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                log.warning("motion_clip_cmd: bad payload %r: %s", parts[1], exc)
                continue
            if action == "stop":
                log.info("motion_clip_cmd: STOP")
                dance_queue.put(("stop",))
                continue
            if action != "play":
                log.warning("motion_clip_cmd: unknown action %r", action)
                continue
            kind = str(payload.get("kind", "locomotion"))
            if kind != "locomotion":
                log.warning(
                    "motion_clip_cmd: kind=%r unsupported here (locomotion "
                    "only); ignoring", kind,
                )
                continue
            motion_key = payload.get("motion_key")
            x2m2_path = _resolve_dance_x2m2(
                dances_dir, payload.get("pkl"), motion_key
            )
            if x2m2_path is None:
                continue
            try:
                dof, quat, fps = load_x2m2(x2m2_path)
            except (ValueError, OSError) as exc:
                log.error("motion_clip_cmd: cannot load %s: %s", x2m2_path, exc)
                continue
            log.info("motion_clip_cmd: PLAY %s (%d frames @ %.1f fps)",
                     x2m2_path.name, dof.shape[0], fps)
            dance_queue.put(("play", x2m2_path.stem, dof, quat, fps))
    finally:
        sock.close(linger=0)


# ---------------------------------------------------------------------------
# Worker thread: replan when the ring buffer dips below threshold
# (port of x2_kplanner._planner_worker; pose reseed omitted -- open loop,
# same as x2_kplanner's default config; dance_active pauses replans).
# ---------------------------------------------------------------------------


def _planner_worker(
    backend,
    intent: IntentState,
    replan_lock: threading.Lock,
    stop_event: threading.Event,
    replan_event: threading.Event,
    dance_active: threading.Event,
    cold_start_ramp_tau_s: float = 0.0,
) -> None:
    log.info("planner worker thread started (backend=%s)", backend.describe())
    cold_start_ramp = ColdStartVelocityRamp(tau_s=cold_start_ramp_tau_s)
    if cold_start_ramp.enabled:
        log.info("planner worker: cold-start velocity ramp ENABLED (tau=%.3fs)",
                 cold_start_ramp.tau_s)
    last_replan_mono: Optional[float] = None

    while not stop_event.is_set():
        forced = False
        if not replan_event.wait(timeout=0.05):
            with replan_lock:
                needs_replan = backend.should_replan()
        else:
            replan_event.clear()
            # Stale-event guard (tape 20260719_205421: every mid-walk replan
            # double-fired). The 50 Hz loop re-arms this event on every tick
            # the buffer is below threshold -- including the whole inference
            # window of the replan already refilling it -- so a fresh commit
            # was immediately followed by a redundant replan and a second
            # overlapping seam blend. Re-check the actual buffer state after
            # clearing; explicit forces (IDLE->PLAYING reseed) still pass via
            # the _force_replan flag because reset() leaves the ring full and
            # should_replan() alone would skip them.
            with replan_lock:
                forced = getattr(backend, "_force_replan", False)
                if forced and MIN_FORCED_REPLAN_MS > 0.0 and last_replan_mono is not None:
                    gap_ms = (time.monotonic() - last_replan_mono) * 1000.0
                    if gap_ms < MIN_FORCED_REPLAN_MS:
                        # DEFER, do not drop: leave _force_replan set so the very
                        # next iteration replans once the gap has elapsed. The
                        # press is honoured late, never lost.
                        _TAPE.ev("forced_replan_deferred", gap_ms=round(gap_ms, 1),
                                 min_ms=MIN_FORCED_REPLAN_MS)
                        forced = False
                if forced:
                    backend._force_replan = False
                needs_replan = forced or backend.should_replan()
        if needs_replan and _STOP_ALIGN_ACTIVE[0] and not forced:
            # A foot-align cut is armed on the CURRENT buffer; a cadence
            # replan would swap that buffer out from under the cut point.
            needs_replan = False
        if not needs_replan or stop_event.is_set():
            continue
        if dance_active.is_set():
            # Dance playback preempts the planner; ring is paused. Reset
            # the ramp so post-dance locomotion ramps from zero.
            cold_start_ramp.reset_idle()
            last_replan_mono = None
            continue
        target, ver = intent.get()
        if _MODEL_STOP_DECEL_REF[0] is not None:
            # STOPPING: replace the idle intent with the decayed velocity
            # target (see the decay-law comment at the declaration).
            target = _MODEL_STOP_DECEL_REF[0]
        if tuple(target) == _IDLE_INTENT:
            # Idle gate: publisher holds the frozen anchor; don't replan.
            cold_start_ramp.reset_idle()
            last_replan_mono = None
            continue

        now_mono = time.monotonic()
        if last_replan_mono is None:
            dt_s = 1.0 / OUTPUT_FPS
        else:
            dt_s = max(1e-3, now_mono - last_replan_mono)
        target = cold_start_ramp.step(tuple(target), dt_s)
        last_replan_mono = now_mono

        # ---- In-place yaw GOVERNOR (2026-08-03): template-mode yaw
        # conditioning under-modulates — sustained in-place turns commanded
        # at 1.0 rad/s measured 1.1-2.8 (mean ~1.9) on the wire. Closed
        # loop: the publisher maintains an EMA of the SERVED yaw rate in
        # _SERVED_YAW_RATE_REF; when turning in place (no translation) and
        # the served rate overshoots the command, attenuate the yaw intent
        # for this replan. Attenuate ONLY (cap 1.0) with a floor so turns
        # can never stall; translation intents are untouched.
        if (abs(target[0]) > 0.05
                and abs(target[1]) + abs(target[2]) < 0.10):
            served = abs(_SERVED_YAW_RATE_REF[0])
            cmd_mag = abs(float(target[0]))
            if os.environ.get("KPLANNER_YAW_GOV_V2", "0") == "1":
                # V2 (2026-08-09, behind flag until live-validated). Tape
                # analysis over 18 sessions: v1 never converges -- trim mean
                # wanders 0.34-0.68 around the ~0.54 ideal, hits the 0.30
                # floor in nearly every session, served/cmd std ~0.5 even
                # with seam-align ON. Causes: per-replan random seed makes
                # the model's yaw gain stochastic, and v1's ASYMMETRIC step
                # clamps (0.55x down vs 1.2x up) ratchet trim DOWN under
                # symmetric noise -> progressively sluggish long holds,
                # lurchy when a high-gain sample lands.
                #   1. symmetric log-clamps [1/1.2, 1.2]: unbiased under noise
                #   2. steady-state gate served > 0.5*cmd: don't adapt off a
                #      cold/decayed EMA (burst starts, post-idle)
                #   3. floor 0.45: worst case ~18% under ideal, not 45%
                if served > 0.5 * cmd_mag:
                    r = max(1.0 / 1.2, min(1.2, cmd_mag / served))
                    _YAW_TRIM_STATE[0] = max(0.45, min(1.0,
                                                       _YAW_TRIM_STATE[0] * r))
            elif served > 0.10:
                # V1: multiplicative adaptation toward cmd/served; step clamped
                # so one noisy EMA sample cannot slam the trim.
                r = max(0.55, min(1.2, cmd_mag / served))
                _YAW_TRIM_STATE[0] = max(0.30, min(1.0,
                                                   _YAW_TRIM_STATE[0] * r))
            trim = _YAW_TRIM_STATE[0]
            target = (float(target[0]) * trim, target[1],
                      target[2], target[3])
            _TAPE.ev("yaw_governor", served=round(served, 3),
                     cmd=round(cmd_mag, 3), trim=round(trim, 3))

        log.info(
            "Replanning with mode: %s, target_vel(fwd): %+.3f, lateral: %+.3f, "
            "yaw_rate: %+.3f, hip_h: %.3f  [v=%d]",
            ("velocity-only" if backend.mode_idx is None
             else f"template_idx={backend.mode_idx}"),
            float(target[2]), float(target[1]), float(target[0]),
            float(target[3]), ver,
        )
        t0 = time.monotonic()
        try:
            # Lock held only for the two cheap phases. Inference -- the
            # 300-500 ms step -- runs UNLOCKED so the 50 Hz publisher keeps
            # streaming. Holding the lock across inference starved SONIC for
            # 15-25 frames and nearly dropped the robot.
            #
            # Feature-detected: the ONNX backend splits, the torch A/B backend
            # cannot (it fuses inference with the buffer swap) and falls back
            # to the fully-locked path.
            if hasattr(backend, "replan_prepare"):
                # Chunk LIVENESS GATE (run 20260719_214150: the model emitted
                # a standing chunk mid-walk -> 2.0 s dead reference -> violent
                # catch-up). A committed standing chunk contaminates the next
                # replan's context, making the collapse self-sustaining. So:
                # while a walk is commanded, reject statistically-still chunks
                # BEFORE commit (old, still-walking buffer keeps streaming),
                # re-roll with a fresh seed; after N failures commit anyway
                # and scream -- a sliding reference beats a starved one.
                # Threshold from measured data: walking chunks show hip-pitch
                # std ~0.12 rad, the dead-window reference ~0.01.
                # TRANSLATION-only predicate (2026-08-03): the incident this
                # gate guards against was a dead reference while WALKING.
                # Yaw-only in-place turns legitimately produce low hip
                # pitch+roll variance (slow pivot, measured 0.017-0.033) and
                # were being re-rolled 3x per replan — each re-roll is a full
                # ~0.3 s inference, tripling turn replan latency for nothing.
                walk_cmded = (abs(target[1]) + abs(target[2])) > 0.10
                for attempt in range(3):
                    with replan_lock:
                        prep = backend.replan_prepare(target)
                    pred, npf = backend.replan_infer(prep)  # <-- no lock
                    if not walk_cmded:
                        break
                    # Axis-appropriate liveness: sagittal gait shows in hip
                    # PITCH std, lateral gait in hip ROLL std. The gate was
                    # pitch-only and re-rolled LEGIT side-step chunks as
                    # "standing" (2026-08-03 log: lateral chunks at pitch-std
                    # 0.027-0.038 rejected -> stutter, no lateral progress).
                    # qpos: 7/13 = L/R hip pitch, 8/14 = L/R hip roll.
                    hp_std = float(max(np.std(pred[:npf, 7]),
                                       np.std(pred[:npf, 13]),
                                       np.std(pred[:npf, 8]),
                                       np.std(pred[:npf, 14])))
                    if hp_std > 0.045:
                        break
                    _TAPE.ev("chunk_rejected", attempt=attempt + 1,
                             hip_pitch_std=round(hp_std, 4),
                             target=list(target))
                    log.warning(
                        "liveness gate: standing chunk while walk commanded "
                        "(hip_pitch_std=%.4f, attempt %d/3) -- re-rolling",
                        hp_std, attempt + 1)
                else:
                    log.error("liveness gate: 3 standing chunks in a row; "
                              "committing anyway (reference may slide)")
                with replan_lock:
                    backend.replan_commit(pred, npf)
                log.info("replan latency %.0f ms (npf=%d)",
                         (time.monotonic() - t0) * 1e3, int(npf))
            else:
                with replan_lock:
                    backend.replan(target)
        except Exception:
            log.exception("worker: replan failed; will retry next cycle")
            time.sleep(0.05)
            continue
        log.debug("worker: replan done in %.1fms (frames_remaining=%d)",
                  (time.monotonic() - t0) * 1000.0, backend.frames_remaining)
        _TAPE.ev("replan_done", ms=round((time.monotonic() - t0) * 1000.0, 1),
                 frames_remaining=int(backend.frames_remaining))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class _IntentTape:
    """Machine-readable replay tape: jsonl of every intent (received and
    applied), every replan (with its RNG seed -> bit-exact offline replay),
    and every published tick, each stamped with monotonic + wall time.

    Exists because the human-readable daemon log has no reliable per-event
    timing (kplanner_gen_from_log.py had to invent a timing template). One
    tape per daemon start; capture_robot_run.py harvests it next to the
    deploy telemetry. Never raises: a broken tape must not touch the robot.

    Env: KPLANNER_TAPE=0 disables; KPLANNER_TAPE_DIR overrides the default
    <PC2_PREFIX or .>/log/kplanner_tape/ location.
    """

    def __init__(self) -> None:
        self._fh = None
        self._t0 = time.monotonic()
        if os.environ.get("KPLANNER_TAPE", "1") == "0":
            return
        try:
            root = os.environ.get("KPLANNER_TAPE_DIR")
            if not root:
                # rituals launch us with cwd=/ and no env: derive from the
                # script's own home (on PC2 that is /home/run/gear-sonic, which
                # has log/); fall back to /tmp rather than dying.
                prefix = os.environ.get("PC2_PREFIX", "")
                script_home = os.path.dirname(os.path.abspath(__file__))
                for base in (prefix, script_home, "."):
                    if base and os.path.isdir(os.path.join(base, "log")):
                        root = os.path.join(base, "log", "kplanner_tape")
                        break
                else:
                    import tempfile
                    root = os.path.join(tempfile.gettempdir(), "kplanner_tape")
            os.makedirs(root, exist_ok=True)
            path = os.path.join(
                root, time.strftime("tape_%Y%m%d_%H%M%S.jsonl"))
            self._fh = open(path, "a", buffering=1)
            # FRAME TAPE (full-content observability, run 20260719_214150:
            # source of a dead reference could not be attributed because no
            # artifact records what the planner actually puts on the wire).
            # Binary f32 records, 40 per tick:
            #   [tm, branch, root_xy(2), root_z, quat_xyzw(4), jpos(31)]
            # branch: 0=ring/planner 1=idle-anchor 2=stop-blend 3=dance.
            # Committed chunks are dumped whole as <session>_chunks/*.npy.
            self._ffh = open(path.replace(".jsonl", ".frames.f32"), "ab")
            self._chunk_dir = path.replace(".jsonl", "_chunks")
            os.makedirs(self._chunk_dir, exist_ok=True)
            # Env audit: every KPLANNER_* knob in effect, stamped into the
            # tape so any session's exact configuration is reconstructable
            # (config-sprawl guard, 2026-08-09). Log too, for live eyes.
            _knobs = {k: v for k, v in sorted(os.environ.items())
                      if k.startswith("KPLANNER_")}
            self.ev("start", wall=time.time(), argv=sys.argv[1:],
                    env_knobs=_knobs)
            if _knobs:
                log.info("env knobs: %s",
                         " ".join(f"{k}={v}" for k, v in _knobs.items()))
            log.info("intent tape: %s (+frames.f32, +chunks/)", path)
        except Exception as exc:  # noqa: BLE001 - tape must never kill the daemon
            log.warning("intent tape disabled: %s", exc)
            self._fh = None
            self._ffh = None

    def frame(self, branch: float, xy, z: float, quat_xyzw, jpos) -> None:
        if getattr(self, "_ffh", None) is None:
            return
        try:
            rec = np.empty(40, dtype=np.float32)
            rec[0] = time.monotonic() - self._t0
            rec[1] = branch
            rec[2:4] = np.asarray(xy, dtype=np.float32)[:2]
            rec[4] = z
            rec[5:9] = np.asarray(quat_xyzw, dtype=np.float32)[:4]
            rec[9:40] = np.asarray(jpos, dtype=np.float32)[:31]
            rec.tofile(self._ffh)
            self._ffh.flush()
        except Exception:  # noqa: BLE001
            pass

    def chunk(self, pred: "np.ndarray", npf: int) -> None:
        if getattr(self, "_chunk_dir", None) is None:
            return
        try:
            tm = time.monotonic() - self._t0
            np.save(os.path.join(self._chunk_dir, f"chunk_{tm:09.3f}.npy"),
                    np.asarray(pred[:npf], dtype=np.float32))
        except Exception:  # noqa: BLE001
            pass

    def ev(self, kind: str, **kw) -> None:
        if self._fh is None:
            return
        try:
            kw["ev"] = kind
            kw["tm"] = round(time.monotonic() - self._t0, 6)
            kw["tw"] = round(time.time(), 3)
            self._fh.write(json.dumps(kw, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass


_TAPE = _IntentTape.__new__(_IntentTape)
_TAPE._fh = None   # inert until run() replaces it
_TAPE._ffh = None
_TAPE._chunk_dir = None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def run(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    OnnxPlannerBackend.USE_GPU = bool(getattr(args, "ort_gpu", False))
    OnnxPlannerBackend.USE_TRT = bool(getattr(args, "ort_trt", False))
    if OnnxPlannerBackend.USE_TRT:
        OnnxPlannerBackend.USE_GPU = True   # TRT implies the GPU chain
    global _TAPE
    _TAPE = _IntentTape()

    global _RUNTIME_TURN_LEFT_SCALE, _RUNTIME_TURN_RIGHT_SCALE
    global _RUNTIME_FORWARD_SCALE, _RUNTIME_BACKWARD_SCALE, _RUNTIME_LATERAL_SCALE
    global _RUNTIME_STICK_SHAPING_EXPONENT, _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS
    global _SPEED_SETPOINT
    _RUNTIME_TURN_LEFT_SCALE = float(args.turn_left_scale)
    _RUNTIME_TURN_RIGHT_SCALE = float(args.turn_right_scale)
    _RUNTIME_FORWARD_SCALE = float(args.forward_scale)
    _RUNTIME_BACKWARD_SCALE = float(args.backward_scale)
    _RUNTIME_LATERAL_SCALE = float(args.lateral_scale)
    if args.stick_shape_exp > 0:
        _RUNTIME_STICK_SHAPING_EXPONENT = float(args.stick_shape_exp)
    _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS = max(0.0, float(args.continuous_forward_min_mps))
    if args.speed_setpoint is not None:
        _SPEED_SETPOINT = max(_SETPOINT_MIN, min(_SETPOINT_MAX, float(args.speed_setpoint)))
    # The nudge-reset default must be the LAUNCH-EFFECTIVE setpoint, not
    # the module-import value: with --speed-setpoint set, the import-time
    # capture would make every idle tick "reset" to the wrong speed
    # (latent bug caught while chasing the 2026-08-12 preflight abort).
    global _DEFAULT_SPEED_SETPOINT
    _DEFAULT_SPEED_SETPOINT = _SPEED_SETPOINT
    log.info("forward speed setpoint: %.2f m/s (X/Y speed_delta nudges, clamp "
             "[%.1f, %.1f])", _SPEED_SETPOINT, _SETPOINT_MIN, _SETPOINT_MAX)

    # ---- Resolve ports (offset support for laptop A/B testing).
    off = int(args.port_offset)
    pub_port = int(args.pub_port) + off
    cmd_port = int(args.cmd_port) + off
    clip_port = int(args.clip_cmd_port) + off
    if off:
        log.info("port offset %+d: pub=%d cmd=%d clip=%d",
                 off, pub_port, cmd_port, clip_port)

    if _port_in_use(pub_port, "127.0.0.1") or _port_in_use(pub_port, "0.0.0.0"):
        log.error("publish port %d already in use.", pub_port)
        return 1
    if _port_in_use(clip_port, "0.0.0.0"):
        log.error("motion_clip_cmd port %d already in use.", clip_port)
        return 1

    # ---- Backend (the slow part).
    if args.backend == "torch":
        for label, p in (("vqvae-ckpt", args.vqvae_ckpt),
                         ("pose-ckpt", args.pose_ckpt),
                         ("root-ckpt", args.root_ckpt)):
            if p is None or not Path(p).is_file():
                log.error("--backend torch requires --%s (got %s)", label, p)
                return 1
        backend = TorchPlannerBackend(
            vqvae_ckpt=Path(args.vqvae_ckpt),
            pose_ckpt=Path(args.pose_ckpt),
            root_ckpt=Path(args.root_ckpt),
            device=args.device,
            replan_threshold_frames=args.replan_threshold_frames,
            planner_mode=args.planner_mode,
        )
    else:
        if args.onnx is None or not Path(args.onnx).is_file():
            log.error("--backend onnx requires --onnx <graph.onnx> (got %s)",
                      args.onnx)
            return 1
        contract = _load_onnx_contract(
            Path(args.onnx),
            Path(args.onnx_sidecar) if args.onnx_sidecar else None,
        )
        backend = OnnxPlannerBackend(
            onnx_path=Path(args.onnx),
            contract=contract,
            replan_threshold_frames=args.replan_threshold_frames,
            planner_mode=args.planner_mode,
        )

    # ---- Warmup anchor + first (warm-the-model) replan.
    warmup_qpos = _load_warmup_qpos(
        Path(args.warmup_qpos) if args.warmup_qpos else None
    )
    backend.reset(warmup_qpos)
    intent_state = IntentState(_IDLE_INTENT)
    t0 = time.monotonic()
    backend.replan(_IDLE_INTENT)
    log.info("first replan complete in %.2fs; ring buffer has %d frames",
             time.monotonic() - t0, backend.frames_remaining)
    # Re-seed so the publish loop starts from a clean anchor buffer (the
    # warm-up replan output is discarded; x2_kplanner keeps it, but its
    # idle gate never reads it either -- the anchor freeze wins while idle).
    backend.reset(warmup_qpos)

    ref_smoother = ReferenceStepSmoother(
        ramp_duration_s=float(args.ref_smoother_ms) / 1000.0,
        trigger_rad=float(args.ref_smoother_trigger_rad),
        shape=args.ref_smoother_shape,
        blend_indices=_REF_SMOOTHER_JOINTS_PRESETS[args.ref_smoother_joints],
    )
    log.info("ref-smoother: shape=%s T=%.0fms trigger=%.3frad joints=%s enabled=%s",
             ref_smoother.shape, ref_smoother.ramp_duration_s * 1000.0,
             ref_smoother.trigger_rad, args.ref_smoother_joints,
             ref_smoother.enabled)

    arm_ingest: Optional[ArmTargetIngest] = None
    if int(getattr(args, "arm_port", 0)) > 0:
        arm_ingest = ArmTargetIngest(int(args.arm_port))
        _ARM_INGEST_REF[0] = arm_ingest

    _WAIST_REF[0] = WaistOverlay()
    _WRIST_REF[0] = WristOverlay()

    publisher = PosePublisher(host=args.pub_host, port=pub_port,
                              topic=args.pub_topic)
    publisher.arm_ingest = arm_ingest
    log.info("publishing %r on tcp://%s:%d at %.1f Hz (backend=%s)",
             args.pub_topic, args.pub_host, pub_port, OUTPUT_FPS,
             backend.describe())

    cmd_queue: "queue.Queue[LocomotionCommand]" = queue.Queue()
    dance_queue: "queue.Queue[tuple]" = queue.Queue()
    _DANCE_QUEUE_REF[0] = dance_queue
    stop_event = threading.Event()
    replan_event = threading.Event()
    replan_lock = threading.Lock()
    dance_active = threading.Event()
    publisher.dance_active = dance_active
    threads: list[threading.Thread] = []

    thr = threading.Thread(
        target=_zmq_command_thread,
        args=(cmd_queue, args.cmd_host, cmd_port, args.cmd_topic, stop_event,
              bool(args.cmd_bind)),
        name="cmd-zmq", daemon=True,
    )
    thr.start()
    threads.append(thr)

    # Forward-obstacle guard feed (scan_guard_pub.py over ZMQ). Fails open:
    # if that process is not running, _guard_blocked() stays False and the
    # planner behaves exactly as before.
    thr = threading.Thread(
        target=_scan_guard_thread, args=(stop_event,),
        name="scan-guard", daemon=True,
    )
    thr.start()
    threads.append(thr)

    thr = threading.Thread(
        target=_motion_clip_cmd_thread,
        args=(dance_queue, clip_port, args.clip_cmd_topic,
              Path(args.dances_dir), stop_event),
        name="cmd-motion-clip", daemon=True,
    )
    thr.start()
    threads.append(thr)

    thr = threading.Thread(
        target=_planner_worker,
        args=(backend, intent_state, replan_lock, stop_event, replan_event,
              dance_active),
        kwargs={"cold_start_ramp_tau_s": float(args.cold_start_ramp_tau_s)},
        name="kplanner-worker", daemon=True,
    )
    thr.start()
    threads.append(thr)

    # Measured-yaw rebase (port of the watchdog's x2_debug yaw fix): the
    # planner is the LIVE producer, and the watchdog only rebases its OWN
    # fallback states -- so the planner must rebase its published root
    # quats to the robot's heading itself, else SONIC twists to world +X.
    measured_yaw = MeasuredYaw()
    # The x2_debug SUB feeds TWO consumers: the measured-yaw rebase (opt-out
    # with --no-yaw-rebase) and the measured JOINTS the crouch-mode entry gate
    # and the whole-body hold read. It used to start only when the rebase was
    # on, so every launcher that passed --no-yaw-rebase (the sim pad stack
    # unless KPLANNER_YAW_REBASE=1, e.g. the frozen-core profile) left the
    # gate with "no measured joints" and crouch mode could never engage. The
    # feed now runs whenever a debug port is configured; --no-yaw-rebase only
    # disables the rebase.
    feed_enabled = int(args.x2_debug_port) > 0
    yaw_rebase_enabled = (not args.no_yaw_rebase) and feed_enabled
    if feed_enabled:
        log.info("x2_debug feed: measured yaw + joints from tcp://%s:%d (yaw rebase %s)",
                 args.x2_debug_host, int(args.x2_debug_port),
                 "ON" if yaw_rebase_enabled else "OFF (--no-yaw-rebase)")
        thr = threading.Thread(
            target=_x2_debug_thread,
            args=(measured_yaw, args.x2_debug_host, int(args.x2_debug_port),
                  args.x2_debug_topic, stop_event),
            name="x2-debug-yaw", daemon=True,
        )
        thr.start()
        threads.append(thr)

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("signal %d -> shutting down", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    period_s = 1.0 / OUTPUT_FPS
    next_tick = time.monotonic()
    end_at = time.monotonic() + args.duration_s if args.duration_s > 0 else float("inf")

    anchor_jpos = warmup_qpos[7:].astype(np.float32).copy()
    # Integrated world root persisted across IDLE <-> PLAYING <-> DANCE.
    current_root_xy = warmup_qpos[:2].astype(np.float64).copy()
    current_root_wxyz = warmup_qpos[3:7].astype(np.float32).copy()
    current_root_z = float(warmup_qpos[2])

    # One-shot measured-yaw offset (captured at ignition, below). The
    # planner runs in a world +X frame; rotating every published root
    # quat by R_z(yaw_offset) aligns that frame to the robot's actual
    # heading so idle holds heading and drive/turns are relative to it.
    # Kept constant per session (not continuous) so it never double-
    # counts the planner's own yaw integration during PLAYING. None ==
    # no x2_debug (sim/laptop) -> rebase is a no-op.
    yaw_offset: list[Optional[float]] = [None]
    # PLAYING-scope heading trim (2026-07-20). The mid-walk complement of the
    # IDLE yaw resync: a slew-limited, deadbanded wire-frame rotation that
    # bleeds reference-vs-measured heading error DURING walks, so nudges and
    # the root model's open-loop yaw wander (measured 6-33 deg per walk;
    # reference-led whip at the worst stumble) never accumulate into a
    # violent SONIC correction. ASSIGNMENT-form servo on the published error
    # (converges; step -> 0 as published -> measured), never the multiply
    # form of the 2026-07-18 runaway. Gated off while a turn is commanded so
    # deliberate turns still lead the robot. Default OFF (--playing-yaw-
    # resync-dps 0); evaluate in the sim stack first.
    yaw_trim: list[float] = [0.0]

    def _eff_off() -> Optional[float]:
        if yaw_offset[0] is None:
            return None
        return yaw_offset[0] + yaw_trim[0]

    def _reb1(q_xyzw: np.ndarray) -> np.ndarray:
        off = _eff_off()
        if off is None:
            return q_xyzw
        return rebase_quats_xyzw_by_yaw(
            np.asarray(q_xyzw, dtype=np.float32).reshape(1, 4), off
        )[0]

    def _rebL(qs: list[np.ndarray]) -> list[np.ndarray]:
        if yaw_offset[0] is None:
            return qs
        return [_reb1(q) for q in qs]

    def _reb_xy(xy: np.ndarray) -> np.ndarray:
        """Rotate a planner-frame root XY into the wire frame.

        The quat has ALWAYS been rebased by R_z(yaw_offset) (_reb1) but the
        XY went out raw -- so published position and orientation were in
        frames that disagree by the ignition heading. Nothing on the robot
        consumed XY (SONIC's tokenizer obs is joints + relative orientation
        only), but every tape consumer (frame tape, overlay, gait metrics)
        saw forward walks as world-frame crab-walks (2026-07-20, operator-
        caught). Rotate XY by the same offset so the wire is self-consistent.
        """
        off = _eff_off()
        if off is None:
            return xy
        c = math.cos(off)
        s = math.sin(off)
        return np.array([c * xy[0] - s * xy[1],
                         s * xy[0] + c * xy[1]], dtype=np.float64)

    def _resync_idle_yaw_from_measured() -> None:
        """IDLE-only: re-anchor the reference heading to the MEASURED yaw.

        Port of ``x2_kplanner.py:3094`` ("Yaw-only resync from robot_pose
        feedback"), whose omission is listed as a known deviation at the top of
        this file. Without it ``current_root_wxyz`` is only ever written by the
        model's own predictions, so anything that moves the real robot off-yaw
        while the stick is centred -- a push, a slip, fall recovery -- leaves us
        publishing a stale ABSOLUTE yaw target. The C++ tokenizer feeds SONIC
        ``rel = inv(measured) * reference``, so a stale reference makes the policy
        twist the body back to the old heading: the "robot always tries to recover
        to the same world orientation" symptom.

        WHY THIS IS SAFE, where the 2026-07-18 runaway spin was not:
        that incident PRE-MULTIPLIED the published quat by a live-updating offset,
        so the planner's frozen -35 deg residual survived as a constant lead and
        the robot chased it forever. This REPLACES the planner's internal heading
        belief with the measurement, discarding the residual. At rest the
        commanded heading equals the measured heading, so the error is exactly
        zero and there is no constant to chase. Assignment, not feedback.

        FRAME: ``current_root_wxyz`` is in the PLANNER frame, but what reaches the
        wire is ``_reb1()`` = ``R_z(yaw_offset) (x) current_root_wxyz``, and
        ``measured_yaw`` is world-frame. So the planner-frame target is
        ``measured - yaw_offset``; the published yaw then comes out as exactly
        ``measured``. Writing ``R_z(measured)`` here instead would double-count
        the ignition offset.

        Also fixes turn-start drift: ``_build_warm_qpos()`` seeds the ring from
        ``current_root_wxyz`` on IDLE -> PLAYING, so keeping it truthful means each
        turn begins from where the robot ACTUALLY points rather than from
        accumulated open-loop error. (Within a single sustained turn PLAYING still
        publishes model-predicted yaw verbatim -- deliberately, so commanded turns
        execute as intended.)
        """
        nonlocal current_root_wxyz
        off = _eff_off()
        if off is None:
            return                      # rebase not armed (sim / no x2_debug)
        m = measured_yaw.get(max_age_s=POSE_FEEDBACK_MAX_AGE_S)
        if m is None:
            return                      # stale -> HOLD LAST GOOD, never identity
        target = _wrap_pi(float(m) - float(off))   # -> planner frame
        cur = _yaw_of_quat_xyzw(_idle_root_xyzw())
        step = _wrap_pi(target - cur)
        cap = MAX_YAW_RESYNC_RAD_S / OUTPUT_FPS
        if step > cap:
            step = cap
        elif step < -cap:
            step = -cap
        current_root_wxyz = _quat_wxyz_from_yaw(_wrap_pi(cur + step))

    global_tick = 0
    last_intent_log: tuple[float, float, float, float] = _IDLE_INTENT
    is_playing = False           # False == IDLE_LOOP (frozen anchor)
    dance: Optional[DancePlayback] = None
    dance_started_t = 0.0
    post_dance_hold_until: Optional[float] = None
    primitive: Optional[PrimitivePlayback] = None
    prim_lib: Optional[dict] = None
    prim_started_t = 0.0
    post_prim_hold_until: Optional[float] = None
    # Stick-scrubbed crouch: hold_torso hip_height_m below the default maps
    # to a depth in [0,1] that scrubs the crouch primitive's phase (the
    # model's own hip_h channel is dead, so the VR/pad squat axis routes
    # through clip playback instead). Stale commands auto-stand.
    crouch_depth = 0.0
    crouch_depth_ts = 0.0
    _crouch_jhist: list = []      # (t, measured jpos) samples for the CROUCH MODE entry gate

    def _resolve_primitives_pkl() -> Optional[Path]:
        cands = [
            Path(getattr(args, "primitives_pkl", "") or ""),
            Path(__file__).resolve().parent / "x2_planner_primitives.pkl",
            Path(__file__).resolve().parent
            / "planner_stack/models/x2_planner_primitives.pkl",
            _REPO_ROOT / "gear_sonic/data/motions/x2_planner_primitives.pkl"
            if "_REPO_ROOT" in globals() else Path("/nonexistent"),
        ]
        for c in cands:
            if str(c) and c.is_file():
                return c
        return None
    _watchdog_last_log_t = 0.0

    # ---- PLAYING -> IDLE stop blend (2026-07-19) --------------------------
    # Releasing the stick used to snap the reference from the mid-stride gait
    # frame straight to the frozen idle anchor in ONE 20 ms tick. Measured on
    # the real robot (yaw-oscillation fall incident):
    # single-tick reference jumps up to 1.213 rad (69.5 deg) on ankle_pitch --
    # ~3475 deg/s of commanded joint velocity, physically untrackable. Two of
    # those landed in the seconds before the robot lost balance.
    #
    # G1's stock stack never does this: its idle is a MODE, not a pose. The
    # controller keeps feeding the model the current velocity + heading and the
    # model generates a natural stop, which is why the G1 takes another step or
    # two after you release the key. We can't generate a stop without the model,
    # but we can stop TELEPORTING: cross-fade the last gait frame into the
    # anchor over STOP_BLEND_FRAMES with a half-cosine ease.
    stop_blend_from: Optional[np.ndarray] = None
    # Quaternion counterpart of ``stop_blend_from`` (2026-08-09). The blend used
    # to ease ONLY the joints, publishing the anchor heading unblended, so every
    # PLAYING -> IDLE transition stepped the reference yaw in a single tick.
    # Short turn bursts make one such transition per burst, which is what drove
    # the runaway (reference yaw rates to 28 rad/s vs 0.70 commanded).
    stop_blend_quat_from: Optional[np.ndarray] = None
    stop_blend_left: int = 0
    # Blend LENGTH is now dynamic (2026-08-12 tippy-toes true fix): the
    # cut pose's knees carry 33-62 deg of flexion vs the ~6 deg anchor
    # (ankles were always fine -- a tape-column bug had us chasing them);
    # extending that knee delta over the fixed 0.32 s commanded ~2-3
    # rad/s of extension and SONIC rode the vertical launch onto its
    # toes. Arming sites now scale the total with the knee delta
    # (16..40 ticks); this variable carries the armed total so the
    # half-cosine schedule uses the RIGHT denominator.
    stop_blend_total: int = STOP_BLEND_FRAMES
    last_gait_jpos: Optional[np.ndarray] = None
    # Heading counterpart of last_gait_jpos, for the stop-blend SLERP.
    last_gait_quat: Optional[np.ndarray] = None
    stopping: bool = False           # model-generated deceleration in flight
    stop_settle_count: int = 0
    # Ankle-neutral blend latch (2026-08-12 tippy-toes live-path fix):
    # every stop exit except deadline/starvation defers blend-arming
    # until the JUST-SERVED frame has both ankles near neutral, bounded
    # by this many extra ticks. Confirmed necessary by tape 092840: a
    # precomputed align cut is defeated by decel replans swapping the
    # buffer and by the settle exit that never consulted it (cut poses
    # p50 49 deg, 81% >30 deg with the selector-only fix).
    stop_latch_wait: int = 0
    _STOP_LATCH_MAX_TICKS: int = int(
        os.environ.get("KPLANNER_STOP_LATCH_MAX_TICKS") or 25)
    stopping_deadline: float = 0.0
    stop_align_cut: Optional[float] = None   # native-frame index to cut at
    stop_align_start_pos: float = 0.0        # read_pos when the cut armed
    stop_brake_until: float = 0.0            # counter-brake pulse deadline
    served_vel_xy: np.ndarray = np.zeros(2)  # last served root XY velocity
    stop_blend_xy_vel: np.ndarray = np.zeros(2)  # XY vel at blend entry
    prev_root_xy_tick: Optional[np.ndarray] = None
    served_speed_mps: float = 0.0    # EMA of published root XY speed
    served_fwd_mps: float = 0.0      # signed EMA along the published heading
    prev_served_yaw: Optional[float] = None   # yaw-governor rate tracking
    cap_prev_raw_yaw: Optional[float] = None  # instant yaw-rate cap state
    cap_yaw_offset: float = 0.0
    early_retrim_t: float = 0.0

    def _idle_root_xyzw() -> np.ndarray:
        return _wxyz_to_xyzw(current_root_wxyz)

    def _build_warm_qpos() -> np.ndarray:
        warm = np.empty_like(warmup_qpos)
        warm[0] = current_root_xy[0]
        warm[1] = current_root_xy[1]
        warm[2] = current_root_z
        warm[3:7] = current_root_wxyz
        # Joints from the LAST SERVED frame, not the anchor (2026-08-12
        # resume-snap fix): re-commanding walk MID-STOP-BLEND used to
        # seed the planner at the anchor pose while the wire was still
        # half-bent -- the new plan then stepped the knee up to 15-38
        # deg in one tick (intermittent preflight 'no joint snaps'
        # aborts; window widened by the dynamic blend lengths). After
        # long idle last_gait_jpos IS the anchor (the anchor tick keeps
        # it fresh), so steady-state behaviour is unchanged.
        warm[7:] = (last_gait_jpos if last_gait_jpos is not None
                    else warmup_qpos[7:])
        return warm

    def _publish_anchor_tick() -> None:
        """One idle-anchor frame (current + 9 identical futures). The
        smoother shapes only the CURRENT frame; the future window stays
        raw -- same as x2_kplanner (futures are built pre-smoother)."""
        nonlocal global_tick
        nonlocal last_gait_jpos, last_gait_quat
        # Speed-nudge safety reset (operator spec 2026-08-12: "reset to
        # default EVERY time the robot stops"): the anchor tick is the
        # one choke point every idle period flows through -- stick
        # release, dance/primitive preemption, e-stop recovery, watchdog
        # hold. Idempotent; logs only when it actually changes.
        _reset_speed_setpoint()
        # Close the yaw loop for IDLE only. This is the ASSIGNMENT form (replace
        # the planner's internal belief with the measurement), NOT the multiply
        # form that caused the 2026-07-18 runaway spin. Reached only from the
        # ``not is_playing`` branch, so PLAYING keeps publishing model-predicted
        # yaw verbatim and commanded turns still execute as intended.
        nonlocal stop_blend_left
        nonlocal stop_blend_from, stop_blend_quat_from, stop_blend_total
        nonlocal stop_blend_xy_vel
        nonlocal current_root_z
        nonlocal current_root_xy
        _resync_idle_yaw_from_measured()
        xyzw = _reb1(_idle_root_xyzw())

        # ---- whole-body release stability (see _WB_* at module scope) ----
        wb_target: Optional[np.ndarray] = None
        if _WB_ENABLED:
            now_wb = time.monotonic()
            link_engaged = (_WB_LINK["engaged"]
                            and now_wb - _WB_LINK["ts"] < _WB_LINK_TIMEOUT_S)
            jm = measured_yaw.get_jpos(POSE_FEEDBACK_MAX_AGE_S)
            if jm is not None:
                _crouch_jhist.append((now_wb, np.asarray(jm, dtype=np.float64).copy()))
                while _crouch_jhist and now_wb - _crouch_jhist[0][0] > 1.0:
                    _crouch_jhist.pop(0)
            mode = _WB["mode"]
            if link_engaged and mode != "shadow":
                if mode == "hold" or mode == "return":
                    log.info("WB: re-engaged from %s -> SHADOW", mode)
                else:
                    log.info("WB: whole-body engaged -> SHADOW (reference "
                             "follows measured joints)")
                if _CROUCH["on"]:
                    log.warning("CROUCH MODE OFF: whole-body engaged")
                    _CROUCH["on"] = False
                _WB["mode"] = mode = "shadow"
                _WB["return_requested"] = False
                _WB["pending_cmd"] = None
                _WB["shadow_drop_logged"] = False
                stop_blend_left = 0
                # The WB arm/wrist latches STAY through shadow (operator
                # session 2026-09-04 15:16, arms forward at chest level):
                # un-latching here handed the wire the planner's internal
                # anchor arms (the overlay pins copy-on-write; the smoother /
                # served-frame tracker never saw the latched values), so the
                # arms dropped for the deploy's 1 s engage warm-up (pose-ref
                # head) and came back up when the smpl head took over. The
                # next release re-latches at measured; arm_release clears.
            if mode == "shadow":
                if not link_engaged:
                    # Release of ANY kind (B, wifi drop, service crash):
                    # freeze where the robot is. The policy keeps
                    # balancing on a reference equal to its own pose.
                    base = (jm if jm is not None else
                            (last_gait_jpos if last_gait_jpos is not None
                             else anchor_jpos))
                    _WB["hold_jpos"] = np.asarray(base, dtype=np.float32).copy()
                    _WB["mode"] = mode = "hold"
                    if _ARM_INGEST_REF[0] is not None:
                        _ARM_INGEST_REF[0].latch(
                            _WB["hold_jpos"][_LEFT_ARM_MJ],
                            _WB["hold_jpos"][_RIGHT_ARM_MJ])
                    if _WRIST_REF[0] is not None:
                        hj = _WB["hold_jpos"]
                        _WRIST_REF[0].latch(float(hj[20]), float(hj[21]),
                                            float(hj[27]), float(hj[28]))
                    knee = max(float(_WB["hold_jpos"][_KNEE_L_DOF]),
                               float(_WB["hold_jpos"][_KNEE_R_DOF]))
                    if _WB_HOLD_LEGS:
                        log.warning("WB: link released -> HOLD at measured pose "
                                    "(worst knee %.2f rad vs anchor %.2f); arms "
                                    "latched; waiting for a pad intent",
                                    knee, float(anchor_jpos[_KNEE_L_DOF]))
                    else:
                        _WB["return_requested"] = True
                        log.warning("WB: link released -> arms/wrists/head "
                                    "latched; legs+waist RETURN to the anchor "
                                    "now (worst knee %.2f rad; leg hold needs "
                                    "the deploy WB_HOLD, KPLANNER_WB_HOLD_LEGS=1 "
                                    "to force it)", knee)
                    _TAPE.ev("wb_hold", knee=knee, hold_legs=int(_WB_HOLD_LEGS))
                else:
                    cur = (last_gait_jpos if last_gait_jpos is not None
                           else anchor_jpos)
                    if jm is not None:
                        step = np.clip(jm - cur, -_WB_SHADOW_STEP_RAD,
                                       _WB_SHADOW_STEP_RAD)
                        wb_target = (cur + step).astype(anchor_jpos.dtype)
                    else:
                        wb_target = np.asarray(cur, dtype=anchor_jpos.dtype)
            if mode == "hold":
                if _WB["return_requested"] and _WB.get("preroll_left", 0) == 0 \
                        and not _WB.get("preroll_done", False):
                    # PRE-ROLL (dual-head deploy hand-over, 2026-09-04): announce
                    # the return on the wire (wb_return=1) for
                    # _WB_RETURN_PREROLL_TICKS BEFORE the reference moves, so
                    # the deploy crossfades onto a static, zero-error reference
                    # and only then sees the rate-limited return.
                    _WB["preroll_left"] = _WB_RETURN_PREROLL_TICKS
                if _WB["return_requested"] and _WB.get("preroll_left", 0) > 0:
                    _WB["preroll_left"] -= 1
                    if _WB["preroll_left"] == 0:
                        _WB["preroll_done"] = True
                    wb_target = _WB["hold_jpos"]
                elif _WB["return_requested"]:
                    _WB["preroll_done"] = False
                    stop_blend_from = _WB["hold_jpos"].copy()
                    stop_blend_quat_from = (
                        None if last_gait_quat is None
                        else np.asarray(last_gait_quat, dtype=np.float64).copy())
                    stop_blend_total = _wb_return_total_for(
                        _WB["hold_jpos"], anchor_jpos)
                    stop_blend_left = stop_blend_total
                    stop_blend_xy_vel = np.zeros(2, dtype=np.float64)
                    _WB["mode"] = mode = "return"
                    _WB["return_requested"] = False
                    log.warning("WB: HOLD -> RETURN to anchor over %d ticks "
                                "(%.2f s; legs <= %.1f rad/s, waist <= %.1f "
                                "rad/s); arms stay latched",
                                stop_blend_total, stop_blend_total / OUTPUT_FPS,
                                _WB_RETURN_LEG_RAD_S, _WB_RETURN_WAIST_RAD_S)
                    _TAPE.ev("wb_return", ticks=int(stop_blend_total))
                else:
                    wb_target = _WB["hold_jpos"]
            # Orientation continuity (operator session 2026-09-04 12:51): the
            # deploy's reference-orientation feature is inv(measured) *
            # reference; the idle quat is yaw-only (upright), so a pitched
            # stance made the switch tick say "pitch up N deg" -> the waist
            # swing -> the fall. While the reference tracks the robot
            # (shadow / hold) publish the MEASURED orientation; the RETURN
            # then SLERPs from it to the upright anchor on the same
            # half-cosine (stop_blend_quat_from = last_gait_quat).
            if mode in ("shadow", "hold"):
                mq = measured_yaw.get_quat_xyzw(POSE_FEEDBACK_MAX_AGE_S)
                if mq is not None:
                    xyzw = mq
            if mode == "return" and stop_blend_left <= 0:
                _WB["mode"] = mode = "off"
                pend = _WB["pending_cmd"]
                _WB["pending_cmd"] = None
                log.info("WB: RETURN complete -> pad drives%s",
                         " (deferred intent re-queued)" if pend else "")
                if pend is not None:
                    cmd_queue.put(pend)

        # Stop blend: ease the last gait frame into the anchor instead of
        # snapping. w goes 0 -> 1 over STOP_BLEND_FRAMES (half-cosine, so the
        # derivative is zero at BOTH ends -- no velocity step at entry or exit).
        target_jpos = anchor_jpos
        # Root HEIGHT joins the blend (2026-08-09 "tippy-toes" fix): the idle
        # payload used to freeze current_root_z at whatever the last gait
        # frame had, for the WHOLE idle. Stop at a tall gait phase and SONIC
        # is handed a standing pose with a pelvis reference a few cm too
        # high -- the only way to satisfy rel-tracking is to rise onto the
        # toes (tape_20260809_202501: +1.0-1.4 cm held -> toe-stand; one stop
        # froze -4.6 cm for 35 s). Ease z to the anchor's stand height on the
        # same half-cosine as the joints, then bank it so the next warm
        # reseed also starts from stand height.
        anchor_root_z = float(warmup_qpos[2])
        pub_root_z = current_root_z if LEGACY_STOP_BLEND else anchor_root_z
        if stop_blend_left > 0 and stop_blend_from is not None:
            # done must sweep 0 -> 1 INCLUSIVE over the N ticks. The original
            # ``1 - left/N`` ran left N..1 and so topped out at 1 - 1/N (0.9375
            # at N=16): w never reached 1, the blend expired with a residual
            # gap, and the next tick snapped it to the anchor. That defeated the
            # "derivative is zero at BOTH ends" guarantee at the EXIT -- for the
            # joints as well as the heading. Measured: final tick 1.71 rad/s
            # instead of ~0. (Found 2026-08-09 while fixing the heading blend.)
            if LEGACY_STOP_BLEND:
                done = 1.0 - (stop_blend_left / float(stop_blend_total))
            else:
                done = 1.0 - ((stop_blend_left - 1)
                              / float(max(1, stop_blend_total - 1)))
            w = 0.5 * (1.0 - math.cos(math.pi * done))
            target_jpos = (stop_blend_from * (1.0 - w)
                           + anchor_jpos * w).astype(anchor_jpos.dtype)
            if not LEGACY_STOP_BLEND:
                # z rides the same schedule; current_root_z stays frozen at
                # the gait value while blending (it is the blend's start).
                pub_root_z = current_root_z * (1.0 - w) + anchor_root_z * w
                # XY momentum continuation (2026-08-09): the LAST unblended
                # channel. Freezing XY at the cut walls off the robot's
                # remaining momentum -> transient toe-rise at every stop.
                # Advance the reference with the handoff velocity decaying
                # on the same half-cosine (total extra travel ~ v0 * 0.16 s,
                # a few cm) so the robot decelerates WITH the reference.
                current_root_xy = (current_root_xy
                                   + stop_blend_xy_vel * (1.0 - w)
                                   / OUTPUT_FPS)
            # Blend the HEADING on the same schedule (2026-08-09 fix). Without
            # this the joints eased over 16 ticks while the reference yaw
            # stepped to the anchor in ONE tick -- the derivative-zero-at-both-
            # ends guarantee in the comment above only ever held for the joints.
            # SLERP in wxyz (shortest arc) to match the resampler's convention.
            if stop_blend_quat_from is not None and not LEGACY_STOP_BLEND:
                q_from = np.array([stop_blend_quat_from[3], stop_blend_quat_from[0],
                                   stop_blend_quat_from[1], stop_blend_quat_from[2]],
                                  dtype=np.float64)
                q_to = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
                xyzw = _wxyz_to_xyzw(_slerp_wxyz_np(q_from, q_to, w))
            stop_blend_left -= 1
        # Global rate clamp on the published heading (any branch, see
        # MAX_REF_YAW_RAD_S). The blend above fixes the known step source; this
        # bounds every other one, including paths added later.
        xyzw = _clamp_ref_yaw_step(xyzw)
        if wb_target is not None:
            # SHADOW / HOLD: the reference is the robot's own pose (all 31
            # dofs; the arm overlay re-pins arms during hold anyway).
            target_jpos = wb_target
        jpos = ref_smoother.update(target_jpos, time.monotonic())
        jpos = _apply_head_yaw(jpos)
        wire_xy = _reb_xy(current_root_xy)
        if not LEGACY_STOP_BLEND and stop_blend_left <= 0:
            # Blend done (or never armed): bank stand height so the next
            # IDLE->PLAYING warm reseed starts from the anchor z too.
            current_root_z = anchor_root_z
        payload = build_pose_payload_np(
            jpos, xyzw, wire_xy, pub_root_z, global_tick,
            # Future window: the anchor (the tracker anticipates the stop's
            # end pose) EXCEPT while the whole-body state machine serves a
            # held / returning reference -- then future == current frame.
            # Otherwise the deploy's pose head, the moment it takes over
            # after a hold, is told "you'll be at the STAND in 0.1..1.0 s"
            # while the current frame is the held crouch: a jump in the
            # future channel (dual-head hand-over spike, 2026-09-04:
            # waist roll to the +-25.8 deg clamp, ankle -17 deg in 0.3 s).
            future_jpos=([jpos] * NUM_FUTURE
                         if (_WB_ENABLED and _WB["mode"] != "off")
                         else [anchor_jpos] * NUM_FUTURE),
            future_quat=[xyzw] * NUM_FUTURE,
            hand_dof=args.hand_dof,
        )
        if _WB_ENABLED and _WB["mode"] in ("hold", "return") and (
                _WB.get("preroll_left", 0) > 0 or _WB["mode"] == "return"):
            # hand-over signal for the dual-head deploy (see pre-roll above)
            payload["wb_return"] = np.asarray([1.0], dtype=np.float32)
        publisher.publish(payload)
        # Served-frame trackers follow EVERY branch that touches the wire
        # (2026-08-11 click fix): without this, a dance started from idle
        # blended from the PREVIOUS clip's stale final frame instead of the
        # anchor actually being served -- reintroducing the entry step it
        # was meant to remove (caught by the local wire test, entry #2).
        last_gait_jpos = jpos.copy()
        last_gait_quat = np.asarray(xyzw, dtype=np.float64).copy()
        _TAPE.frame(2.0 if stop_blend_left > 0 else 1.0,
                    wire_xy, pub_root_z, xyzw,
                    getattr(publisher, "last_wire_jpos", None) if getattr(publisher, "last_wire_jpos", None) is not None else jpos)
        global_tick += 1

    try:
        with PidFile(Path(args.pid_file)):
            # ---- Yaw-capture gate (ordering fix). x2_debug comes from the
            # C++ deploy, which the ritual starts AFTER this planner (the
            # gate before deploy is satisfied by the watchdog's COLD_IDLE,
            # not by us). So stay SILENT until the first x2_debug frame
            # lands -- the watchdog holds the robot at its measured heading
            # (its own rebased idle clip) during the wait -- then latch the
            # ignition heading ONCE. Publishing LIVE identity frames before
            # capture would make SONIC twist to world +X (the snap). A
            # generous fail-safe timeout proceeds unrebased if x2_debug
            # never appears (deploy never started / regression escape).
            if yaw_rebase_enabled:
                # FAIL-STOP, NOT FAIL-OPEN (2026-07-18). This wait used to give
                # up after --yaw-capture-timeout-s and publish unrebased. That
                # is the WORST possible fallback: an unrebased publish is
                # identity == world +X, a KNOWN-WRONG heading SONIC actively
                # twists the body toward. Measured on hardware: commanded yaw
                # pinned at exactly 0.0 deg while a 40 deg hand-nudge was driven
                # back to -0.5 deg in ~1.5s.
                #
                # It also could not succeed on a clean start: x2_debug is
                # published by the DEPLOY, which the ritual starts AFTER this
                # planner, so a bounded wait always expired. It only ever armed
                # when a PREVIOUS deploy happened to still be alive.
                #
                # Waiting indefinitely IS the lazy-arm: we latch on the first
                # x2_debug frame whenever it arrives (seconds after the deploy
                # comes up). Staying silent meanwhile is safe and intended --
                # the watchdog holds the robot on its own measured-yaw-rebased
                # idle clip, and the ritual's pre-deploy pose gate is satisfied
                # by the watchdog's COLD_IDLE, not by us. No stream, no snap.
                warn_every_s = max(5.0, float(args.yaw_capture_timeout_s))
                next_warn = time.monotonic() + warn_every_s
                waited_s = 0.0
                log.info("waiting for x2_debug to capture ignition heading "
                         "(silent; watchdog holds; will NOT proceed unrebased)...")
                while not stop_event.is_set():
                    cap = measured_yaw.get(max_age_s=1.0)
                    if cap is not None:
                        yaw_offset[0] = cap
                        log.info("measured-yaw rebase ARMED after %.1fs: "
                                 "ignition heading %.1f deg "
                                 "(root quats -> robot frame)",
                                 waited_s, math.degrees(cap))
                        break
                    now_w = time.monotonic()
                    if now_w >= next_warn:
                        next_warn = now_w + warn_every_s
                        log.warning("still no x2_debug after %.0fs -- staying "
                                    "SILENT (watchdog holds the robot). This is "
                                    "expected until the deploy starts; it "
                                    "publishes x2_debug. Planner will arm and "
                                    "begin publishing automatically.",
                                    waited_s)
                    time.sleep(period_s)   # SILENT: do not publish pre-capture
                    waited_s += period_s

            # ---- Quiet-stand warmup (frozen anchor).
            warmup_n = int(round(max(0.0, args.warmup_quiet_stand_s) * OUTPUT_FPS))
            if warmup_n > 0:
                log.info("quiet-stand warmup: %d ticks (%.2fs) of frozen anchor",
                         warmup_n, args.warmup_quiet_stand_s)
                for _ in range(warmup_n):
                    if stop_event.is_set() or time.monotonic() >= end_at:
                        break
                    _publish_anchor_tick()
                    next_tick += period_s
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        next_tick = time.monotonic()
                log.info("quiet-stand warmup done; planner active.")

            while not stop_event.is_set() and time.monotonic() < end_at:
                # ---- Dance command queue (drain everything; last wins).
                while True:
                    try:
                        item = dance_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item[0] == "stop":
                        if dance is not None:
                            log.info("dance %s: STOP; idle hold %.1fs then "
                                     "planner resumes", dance.name,
                                     args.post_dance_idle_s)
                            dance = None
                            dance_active.clear()
                            post_dance_hold_until = (
                                time.monotonic() + args.post_dance_idle_s
                            )
                            # Exit blend (2026-08-11 click fix): ease the
                            # last served clip frame into the anchor via the
                            # anchor tick's existing stop-blend machinery
                            # (joints + heading SLERP + z, half-cosine) --
                            # previously the anchor pose hit the wire in one
                            # tick. XY momentum: zero (clips never advance
                            # the served XY).
                            if last_gait_jpos is not None:
                                stop_blend_from = last_gait_jpos.copy()
                                stop_blend_quat_from = (
                                    None if last_gait_quat is None
                                    else np.asarray(last_gait_quat,
                                                    dtype=np.float64).copy())
                                stop_blend_total = _stop_blend_total_for(
                                    last_gait_jpos, anchor_jpos)
                                stop_blend_left = stop_blend_total
                                stop_blend_xy_vel = np.zeros(
                                    2, dtype=np.float64)
                    else:
                        _, name, dof, quat, fps = item
                        cur_yaw = _yaw_of_quat_xyzw(_idle_root_xyzw())
                        # Entry blend starts from the frame SONIC actually
                        # last received (gait, anchor, or a previous dance --
                        # all update last_gait_jpos); anchor pose covers the
                        # cold path. Full-body: this is what the lower-body-
                        # only ref_smoother cannot do for the arms.
                        entry_from = (last_gait_jpos if last_gait_jpos
                                      is not None else anchor_jpos)
                        dance = DancePlayback(
                            dof, quat, fps, cur_yaw, name,
                            entry_jpos=entry_from,
                            entry_blend_ticks=STOP_BLEND_FRAMES)
                        dance_started_t = time.monotonic()
                        dance_active.set()
                        post_dance_hold_until = None
                        # Planner ring paused + cleared: freeze the FSM at
                        # IDLE so the post-dance resume path re-seeds the
                        # neural buffer at the (new) current root.
                        is_playing = False
                        stopping = False   # preemption cancels model-stop
                        with replan_lock:
                            backend.reset(_build_warm_qpos())
                        log.info("dance %s: START (planner preempted)", name)

                # ---- Drain primitive queue (crouch/lean playback control).
                while True:
                    try:
                        prim_cmd = _PRIM_QUEUE.get_nowait()
                    except queue.Empty:
                        break
                    if prim_cmd[0] == "exit":
                        if primitive is not None:
                            primitive.exit()
                    else:
                        _, prim_name = prim_cmd
                        if primitive is not None or dance is not None:
                            log.warning("primitive %s: ignored (playback "
                                        "already active)", prim_name)
                            continue
                        if is_playing:
                            log.warning("primitive %s: ignored (planner is "
                                        "walking; stop first)", prim_name)
                            continue
                        if prim_lib is None:
                            pkl_path = _resolve_primitives_pkl()
                            if pkl_path is None:
                                log.error("primitive %s: no "
                                          "x2_planner_primitives.pkl found",
                                          prim_name)
                                continue
                            prim_lib = _load_primitives_pkl(pkl_path)
                            log.info("primitives loaded: %d clips from %s",
                                     len(prim_lib), pkl_path)
                        clip = prim_lib.get(prim_name)
                        if clip is None:
                            log.error("primitive %r unknown; have: %s",
                                      prim_name, sorted(prim_lib)[:40])
                            continue
                        cur_yaw = _yaw_of_quat_xyzw(_idle_root_xyzw())
                        # "_step" anywhere in the name (fwd_step_quarter_ft,
                        # back_step_half_ft, side_left_step): play forward
                        # once and PERSIST the root XY -- the micro-step bank
                        # (2026-09-06). Oneshot would reverse the step.
                        prim_mode = ("step" if (
                            clip.get("recipe_family") == "discrete_step"
                            or "_step" in prim_name) else "oneshot")
                        primitive = PrimitivePlayback(
                            prim_name, clip, cur_yaw, float(current_root_z),
                            mode=prim_mode)
                        prim_started_t = time.monotonic()
                        post_prim_hold_until = None
                        is_playing = False
                        stopping = False   # preemption cancels model-stop
                        with replan_lock:
                            backend.reset(_build_warm_qpos())

                # ---- Drain planner_cmd queue; apply the latest intent.
                latest_cmd: Optional[LocomotionCommand] = None
                _mode_cmds: list = []
                while True:
                    try:
                        _c = cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    if _c.intent == CROUCH_MODE_INTENT:
                        _mode_cmds.append(_c)      # never dropped behind the 20 Hz stick frames
                    else:
                        latest_cmd = _c
                for latest_mode_cmd in _mode_cmds:
                  latest_cmd_saved = latest_cmd; latest_cmd = latest_mode_cmd
                  if latest_cmd is not None and latest_cmd.intent == CROUCH_MODE_INTENT:
                      want_on = latest_cmd.magnitude == "on"
                      if not want_on:
                          if _CROUCH["on"]:
                              log.warning("CROUCH MODE OFF (%s) -> next walk at the standing hip %.3f",
                                          latest_cmd.source, _HIP_HEIGHT_M)
                          _CROUCH["on"] = False
                      elif _CROUCH["on"]:
                          _CROUCH["ts"] = time.monotonic()     # keepalive
                      else:
                          now_c = time.monotonic()
                          jm_c = measured_yaw.get_jpos(POSE_FEEDBACK_MAX_AGE_S)
                          why = []
                          if jm_c is None:
                              why.append("no measured joints")
                          else:
                              _crouch_jhist.append((now_c, np.asarray(jm_c, dtype=np.float64).copy()))
                              while _crouch_jhist and now_c - _crouch_jhist[0][0] > 1.0:
                                  _crouch_jhist.pop(0)
                              if len(_crouch_jhist) >= 2 and now_c - _crouch_jhist[0][0] >= 0.3:
                                  t_a, q_a = _crouch_jhist[0]
                                  qd = np.abs(_crouch_jhist[-1][1][:12] - q_a[:12]) / max(1e-3, now_c - t_a)
                                  if float(qd.max()) > _CROUCH_GATE_LEG_QD_RAD_S:
                                      why.append("legs moving (%.2f rad/s)" % float(qd.max()))
                              else:
                                  why.append("no joint history yet (retry in a moment)")
                              kmax = max(float(jm_c[_KNEE_L_DOF]), float(jm_c[_KNEE_R_DOF]))
                              if kmax > _CROUCH_GATE_KNEE_MAX_RAD:
                                  why.append("knees at %.0f deg (not standing)" % math.degrees(kmax))
                          if now_c - _CROUCH_LAST_MOVE["ts"] < _CROUCH_GATE_QUIET_S:
                              why.append("locomotion %.1f s ago" % (now_c - _CROUCH_LAST_MOVE["ts"]))
                          if _WB.get("mode") == "shadow":
                              why.append("whole-body engaged")
                          if primitive is not None or dance is not None or is_playing:
                              why.append("clip/primitive playing")
                          if why:
                              log.warning("CROUCH MODE REFUSED (%s): %s", latest_cmd.source, "; ".join(why))
                          else:
                              _CROUCH["on"] = True
                              _CROUCH["since"] = now_c
                              _CROUCH["ts"] = now_c
                              log.warning("CROUCH MODE ON (%s): locomotion targets at hip %.3f (crouch template); "
                                          "idle stands; D-pad up / E-STOP / whole-body engage clears it",
                                          latest_cmd.source, _CROUCH_HIP_M)
                              if _CROUCH_SEED is not None:
                                  log.warning("crouch mode: sampler seed PINNED to %d (KPLANNER_CROUCH_SEED, phase %d of 16)", _CROUCH_SEED, _CROUCH_SEED % 16)
                      latest_cmd = None
                  latest_cmd = latest_cmd_saved
                if _CROUCH["on"] and time.monotonic() - _CROUCH["ts"] > _CROUCH_KEEPALIVE_S:
                    log.warning("CROUCH MODE OFF: keepalive lost (trigger released / pad silent %.1f s) -> standing hip",
                                time.monotonic() - _CROUCH["ts"])
                    _CROUCH["on"] = False
                if (latest_cmd is not None
                        and latest_cmd.intent == HOLD_TORSO_INTENT
                        and latest_cmd.hip_height_m is not None):
                    span = max(1e-3, float(args.crouch_hip_span))
                    crouch_depth = max(0.0, min(1.0,
                        (_HIP_HEIGHT_M - float(latest_cmd.hip_height_m)) / span))
                    crouch_depth_ts = time.monotonic()
                # ---- Side steps from stand -> clean lead-leg-first
                # primitive instead of the model's learned cross-step gait.
                # Triggers on ANY laterally-dominant command while idle:
                # discrete side_left/side_right intents AND the continuous
                # pad/VR stick (which is what operators actually use --
                # nothing in the current bridges emits discrete taps).
                # Held stick = repeated steps (each step completes, brief
                # settle, next fires). While already walking, the model's
                # continuous lateral gait keeps the stick.
                _side_tgt = (intent_to_velocity(latest_cmd)
                             if latest_cmd is not None
                             and args.side_step_primitive else None)
                if (_side_tgt is not None
                        and abs(_side_tgt[1]) > 0.05
                        and abs(_side_tgt[2]) < 0.10
                        and abs(_side_tgt[0]) < 0.20
                        and primitive is None and dance is None
                        and not is_playing):
                    step_name = ("side_left_step" if _side_tgt[1] > 0
                                 else "side_right_step")
                    if prim_lib is None:
                        pkl_path = _resolve_primitives_pkl()
                        if pkl_path is not None:
                            prim_lib = _load_primitives_pkl(pkl_path)
                            log.info("primitives loaded: %d clips from %s",
                                     len(prim_lib), pkl_path)
                    clip = (prim_lib or {}).get(step_name)
                    if clip is not None:
                        cur_yaw = _yaw_of_quat_xyzw(_idle_root_xyzw())
                        primitive = PrimitivePlayback(
                            step_name, clip, cur_yaw,
                            float(current_root_z), mode="step")
                        prim_started_t = time.monotonic()
                        post_prim_hold_until = None
                        is_playing = False
                        stopping = False   # preemption cancels model-stop
                        with replan_lock:
                            backend.reset(_build_warm_qpos())
                        latest_cmd = None
                    else:
                        log.warning("side step: %s not in primitives pkl; "
                                    "falling through to model gait", step_name)
                if latest_cmd is not None and primitive is not None:
                    # Locomotion with real motion cancels the primitive (walk
                    # away = stand up first); everything else (hold_torso
                    # lean updates, idle keepalives) passes through harmless
                    # -- intent is pinned to IDLE below while active.
                    tgt = intent_to_velocity(latest_cmd)
                    if (tgt[0] != 0.0 or tgt[1] != 0.0 or tgt[2] != 0.0):
                        primitive.exit()
                    intent_state.set(_IDLE_INTENT)
                    latest_cmd = None
                if (latest_cmd is not None and _WB_ENABLED
                        and _WB["mode"] in ("hold", "return")
                        and intent_to_velocity(latest_cmd) != _IDLE_INTENT):
                    # WB HOLD/RETURN: a pad locomotion intent is the
                    # operator's "switch to the kplanner" signal, but the
                    # gait must start from the STAND, not from a crouch
                    # seed: arm the rate-limited return (anchor tick) and
                    # defer the intent until it completes.
                    if _WB["mode"] == "hold" and not _WB["return_requested"]:
                        _WB["return_requested"] = True
                        log.warning("WB HOLD: pad intent %s -> returning to "
                                    "the anchor first (intent deferred)",
                                    latest_cmd.intent)
                    _WB["pending_cmd"] = latest_cmd
                    intent_state.set(_IDLE_INTENT)
                    latest_cmd = None
                if (latest_cmd is not None and _WB_ENABLED
                        and _WB["mode"] == "shadow"
                        and intent_to_velocity(latest_cmd) != _IDLE_INTENT):
                    # WB SHADOW: the operator's body drives the legs. A stick
                    # push (Pico thumbsticks publish source:"vr") must not
                    # start a gait under the shadowing reference: the WB
                    # state machine lives in the anchor tick, and a PLAYING
                    # gait would carry a moving reference into the next
                    # release / hand-over.
                    if not _WB.get("shadow_drop_logged", False):
                        log.warning("WB SHADOW: locomotion intent %s ignored "
                                    "while the operator's body drives the "
                                    "reference", latest_cmd.intent)
                        _WB["shadow_drop_logged"] = True
                    intent_state.set(_IDLE_INTENT)
                    latest_cmd = None
                if latest_cmd is not None:
                    target = intent_to_velocity(latest_cmd)
                    intent_state.set(target)
                    if target != last_intent_log:
                        log.info("intent applied (%s, %s, %s) -> target=%s",
                                 latest_cmd.intent, latest_cmd.magnitude,
                                 latest_cmd.source, target)
                        _TAPE.ev("intent_applied", target=list(target),
                                 intent=latest_cmd.intent,
                                 magnitude=latest_cmd.magnitude)
                        last_intent_log = target

                # ---- Stale-command watchdog (opt-in, default OFF).
                if args.command_watchdog_s > 0.0 and intent_state.force_idle_if_stale(
                    max_age_s=args.command_watchdog_s, idle_target=_IDLE_INTENT
                ):
                    last_intent_log = _IDLE_INTENT
                    now_t = time.monotonic()
                    if now_t - _watchdog_last_log_t > 1.0:
                        log.warning(
                            "command watchdog: no upstream intent for %.2fs "
                            "(threshold %.2fs); forcing IDLE",
                            intent_state.seconds_since_last_set(),
                            args.command_watchdog_s,
                        )
                        _watchdog_last_log_t = now_t

                # =========== DANCE branch (preempts planner output) ========
                if dance is not None:
                    jpos, quat, fut_j, fut_q = dance.tick()
                    # Persist the streamed heading so post-dance idle (and
                    # the next PLAYING seed) holds the dance-final yaw.
                    current_root_wxyz = np.array(
                        [quat[3], quat[0], quat[1], quat[2]], dtype=np.float32
                    )
                    now_mono = time.monotonic()
                    jpos_s = ref_smoother.update(
                        jpos.astype(np.float32), now_mono,
                        # Only the entry step (anchor -> clip frame 0) may
                        # arm a ramp; the clip's own fast motion must not.
                        allow_arm=(now_mono - dance_started_t)
                        <= ref_smoother.ramp_duration_s,
                    )
                    wire_quat = _reb1(quat)
                    payload = build_pose_payload_np(
                        jpos_s, wire_quat, _reb_xy(current_root_xy),
                        current_root_z,
                        global_tick, future_jpos=fut_j,
                        future_quat=_rebL(fut_q),
                        hand_dof=args.hand_dof,
                    )
                    publisher.publish(payload)
                    global_tick += 1
                    # Keep the served-frame trackers fresh during clips too:
                    # the exit stop-blend (and the NEXT clip's entry blend)
                    # must start from what SONIC actually last received.
                    last_gait_jpos = jpos_s.copy()
                    last_gait_quat = np.asarray(wire_quat,
                                                dtype=np.float64).copy()
                    if dance.finished:
                        log.info("dance %s: clip complete; idle hold %.1fs "
                                 "then planner resumes", dance.name,
                                 args.post_dance_idle_s)
                        dance = None
                        dance_active.clear()
                        post_dance_hold_until = (
                            time.monotonic() + args.post_dance_idle_s
                        )
                        # Exit blend (2026-08-11 click fix): arm the anchor
                        # tick's stop-blend from the clip's final served
                        # frame -- previously the idle anchor hit the wire
                        # in ONE tick (full-body jump = the audible click).
                        stop_blend_from = jpos_s.copy()
                        stop_blend_quat_from = np.asarray(
                            wire_quat, dtype=np.float64).copy()
                        stop_blend_total = _stop_blend_total_for(
                            jpos_s, anchor_jpos)
                        stop_blend_left = stop_blend_total
                        stop_blend_xy_vel = np.zeros(2, dtype=np.float64)
                    next_tick += period_s
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    elif -slack > 5 * period_s:
                        next_tick = time.monotonic()
                    continue

                # ---- Stick-scrubbed crouch: spawn/feed the scrub player.
                depth_now = (0.0 if time.monotonic() - crouch_depth_ts > 1.0
                             else crouch_depth)
                if (depth_now > 0.05 and primitive is None and dance is None
                        and not is_playing):
                    if prim_lib is None:
                        pkl_path = _resolve_primitives_pkl()
                        if pkl_path is not None:
                            prim_lib = _load_primitives_pkl(pkl_path)
                            log.info("primitives loaded: %d clips from %s",
                                     len(prim_lib), pkl_path)
                    clip = (prim_lib or {}).get(args.crouch_primitive)
                    if clip is not None:
                        cur_yaw = _yaw_of_quat_xyzw(_idle_root_xyzw())
                        primitive = PrimitivePlayback(
                            args.crouch_primitive, clip, cur_yaw,
                            float(current_root_z), mode="scrub")
                        prim_started_t = time.monotonic()
                        post_prim_hold_until = None
                        is_playing = False
                        stopping = False   # preemption cancels model-stop
                        with replan_lock:
                            backend.reset(_build_warm_qpos())
                        log.info("crouch scrub: START (depth %.2f)", depth_now)
                if primitive is not None and primitive.mode == "scrub":
                    primitive.set_depth(depth_now)

                # =========== PRIMITIVE branch (crouch/lean playback) =======
                if primitive is not None:
                    jpos, quat, prim_z, fut_j, fut_q = primitive.tick()
                    prim_xy = current_root_xy + primitive.xy_delta()
                    now_mono = time.monotonic()
                    jpos_s = ref_smoother.update(
                        jpos.astype(np.float32), now_mono,
                        allow_arm=(now_mono - prim_started_t)
                        <= ref_smoother.ramp_duration_s,
                    )
                    payload = build_pose_payload_np(
                        jpos_s, _reb1(quat), _reb_xy(prim_xy),
                        float(prim_z),
                        global_tick, future_jpos=fut_j,
                        future_quat=_rebL(fut_q),
                        hand_dof=args.hand_dof,
                    )
                    publisher.publish(payload)
                    global_tick += 1
                    if primitive.done:
                        mode_done = primitive.mode
                        if mode_done == "step":
                            current_root_xy = (
                                current_root_xy + primitive.final_xy_delta()
                            ).astype(current_root_xy.dtype)
                        primitive = None
                        if mode_done == "scrub":
                            post_prim_hold_until = None
                        elif mode_done == "step":
                            # short settle so a held stick chains steps at a
                            # controlled cadence instead of machine-gunning
                            post_prim_hold_until = time.monotonic() + 0.5
                        else:
                            post_prim_hold_until = (
                                time.monotonic() + args.post_dance_idle_s
                            )
                    next_tick += period_s
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    elif -slack > 5 * period_s:
                        next_tick = time.monotonic()
                    continue

                # =========== POST-PRIMITIVE idle hold ======================
                if post_prim_hold_until is not None:
                    if time.monotonic() < post_prim_hold_until:
                        _publish_anchor_tick()
                        next_tick += period_s
                        slack = next_tick - time.monotonic()
                        if slack > 0:
                            time.sleep(slack)
                        elif -slack > 5 * period_s:
                            next_tick = time.monotonic()
                        continue
                    post_prim_hold_until = None
                    log.info("post-primitive idle hold done; planner resumes")

                # =========== POST-DANCE idle hold ==========================
                if post_dance_hold_until is not None:
                    if time.monotonic() < post_dance_hold_until:
                        _publish_anchor_tick()
                        next_tick += period_s
                        slack = next_tick - time.monotonic()
                        if slack > 0:
                            time.sleep(slack)
                        elif -slack > 5 * period_s:
                            next_tick = time.monotonic()
                        continue
                    post_dance_hold_until = None
                    log.info("post-dance idle hold done; planner resumes")

                # =========== Normal IDLE_LOOP / PLAYING FSM ================
                current_target, _ = intent_state.get()
                is_idle = tuple(current_target) == _IDLE_INTENT
                if stopping and not is_idle:
                    # Operator resumed mid-deceleration: cancel the model
                    # stop and replan for the new intent; PLAYING never
                    # stopped serving frames, so this is seamless.
                    stopping = False
                    with replan_lock:
                        backend._force_replan = True
                    replan_event.set()
                    log.info("model-stop: cancelled (new intent %s)",
                             tuple(round(v, 2) for v in current_target))
                if stopping and is_idle:
                    # Deceleration in flight: intent stays idle and PLAYING
                    # stays true. Resolution (settle / deadline / starvation)
                    # lives in the PLAYING branch below.
                    pass
                elif (not is_idle) != is_playing:
                    if not is_idle:
                        # IDLE -> PLAYING: seed the ring at the CURRENT
                        # integrated root so velocity intents start from
                        # where the robot is (port of x2_kplanner's warm
                        # seed; no pose-feedback yaw refresh -- open loop).
                        warm = _build_warm_qpos()
                        # Cap-offset lifecycle fix (2026-08-09): the heading
                        # belief is stored POST-cap-rotation (see the
                        # current_root_wxyz update in the PLAYING branch), so
                        # the warm seed already banks cap_yaw_offset. Carrying
                        # the offset into the new episode applies it a SECOND
                        # time -- one instant +offset heading step per
                        # IDLE->PLAYING transition. Measured on
                        # tape_20260809_202501: +89.4 deg per turn tap,
                        # cumulative, and the collapse burst was exactly
                        # 7 x 89.4 deg for 7 reseeds. Zero it here; the cap
                        # keeps its intra-episode rate-shaping role.
                        cap_yaw_offset = 0.0
                        cap_prev_raw_yaw = None
                        with replan_lock:
                            backend.reset(warm)
                            backend._force_replan = True
                        replan_event.set()
                        log.info(
                            "state: IDLE_LOOP -> PLAYING (intent=%s); buffer "
                            "seeded at root_xy=%s yaw_wxyz=%s, replan queued",
                            current_target, warm[:2].tolist(),
                            warm[3:7].tolist(),
                        )
                        if (_WB_ENABLED and _WB_ARM_LATCH_GAIT != "keep"
                                and _ARM_INGEST_REF[0] is not None
                                and getattr(_ARM_INGEST_REF[0], "_wb_latched", False)):
                            _ARM_INGEST_REF[0].clear(force=True)
                            if _WRIST_REF[0] is not None:
                                _WRIST_REF[0].clear(force=True)
                            log.warning("WB: gait started with the arms LATCHED -> "
                                        "releasing the arm/wrist latch to the "
                                        "planner's gait arms (skipped steps on the "
                                        "robot with pinned arms, 2026-09-05; "
                                        "KPLANNER_WB_ARM_LATCH_GAIT=keep to pin)")
                    elif _MODEL_STOP and served_speed_mps > _MODEL_STOP_MIN_MPS:
                        # PLAYING -> STOPPING (2026-08-09): G1-style model
                        # stop. Intent went idle while the reference still
                        # moves; instead of freezing to the anchor, keep
                        # PLAYING and force a replan -- intent_state already
                        # reads _IDLE_INTENT (zero velocity), so the template
                        # graph plans the DECELERATION from the moving
                        # context. The anchor blend runs only after the
                        # served speed settles (or deadline / starvation
                        # fallback, resolved in the PLAYING branch).
                        stopping = True
                        stop_settle_count = 0
                        stop_latch_wait = 0
                        _reset_speed_setpoint()
                        # Deadline scales with entry speed: the decay law is
                        # exponential (half-life ~1 s), so each doubling of
                        # entry speed needs ~one more second, not two.
                        _halvings = max(1.0, math.log2(
                            max(served_speed_mps, _MODEL_STOP_SETTLED_MPS)
                            / _MODEL_STOP_SETTLED_MPS))
                        stopping_deadline = time.monotonic() + min(
                            5.0, max(_MODEL_STOP_TIMEOUT_S, 1.0 + 1.2 * _halvings))
                        if (_STOP_FOOT_ALIGN
                                and abs(0.5 * served_fwd_mps)
                                < _MODEL_STOP_ZERO_SNAP_MPS):
                            # Walk-speed stop: cut the ALREADY-generated
                            # buffer at the next feet-passing frame and blend
                            # there. No replan -- no flush delay, one
                            # movement, bounded travel.
                            with replan_lock:
                                stop_align_start_pos = float(backend._read_pos)
                                stop_align_cut = _find_foot_align_cut(
                                    backend._buf, stop_align_start_pos)
                                _cut_s = ((stop_align_cut
                                           - stop_align_start_pos)
                                          / max(1e-6, backend._model_fps))
                            _STOP_ALIGN_ACTIVE[0] = True
                            log.info("state: PLAYING -> STOPPING (foot-align "
                                     "cut in %.2fs, %.2f m/s)",
                                     _cut_s, served_speed_mps)
                        else:
                            _MODEL_STOP_DECEL_REF[0] = (
                                0.0, 0.0, 0.5 * served_fwd_mps, _HIP_HEIGHT_M)
                            with replan_lock:
                                backend._force_replan = True
                            replan_event.set()
                            log.info("state: PLAYING -> STOPPING (decay-law "
                                     "decel from %.2f m/s, deadline %.1fs)",
                                     served_speed_mps, stopping_deadline
                                     - time.monotonic())
                        # is_playing stays True: frames keep serving.
                    else:
                        # Immediate anchor blend -- the ONLY stop path until
                        # 2026-08-09 (see the model-stop state above; the
                        # 2026-08-03 revert note asked for exactly that
                        # ring-fill-guaranteed re-attempt). Still the
                        # terminal path for every stop (reached at settle /
                        # deadline / starvation, or directly below
                        # _MODEL_STOP_MIN_MPS where a 0.32 s fade is fine).
                        _reset_speed_setpoint()
                        if last_gait_jpos is not None:
                            stop_blend_from = last_gait_jpos.copy()
                            # Start the heading blend from the SAME frame SONIC
                            # last received, so the slerp is continuous with
                            # what was actually on the wire (mirrors the
                            # last_gait_jpos rationale above).
                            stop_blend_quat_from = (
                                None if last_gait_quat is None
                                else np.asarray(last_gait_quat, dtype=np.float64).copy())
                            stop_blend_total = _stop_blend_total_for(
                                last_gait_jpos, anchor_jpos)
                            stop_blend_left = stop_blend_total
                        stop_blend_xy_vel = served_vel_xy.copy()
                        _SERVED_YAW_RATE_REF[0] = 0.0
                        prev_served_yaw = None
                        cap_prev_raw_yaw = None
                        # Belief absorbed the cap rotation on the last PLAYING
                        # tick; a stale offset would otherwise freeze across
                        # IDLE (the 0.98/tick bleed only runs while PLAYING)
                        # and double-apply at the next reseed. See the
                        # IDLE->PLAYING twin of this reset above.
                        cap_yaw_offset = 0.0
                        log.info(
                            "state: PLAYING -> IDLE_LOOP (intent back to "
                            "idle); blending to anchor over %d ticks",
                            STOP_BLEND_FRAMES,
                        )
                        is_playing = False
                        served_speed_mps = 0.0
                        prev_root_xy_tick = None
                    if not is_idle:
                        is_playing = True
                # Failsafe: the decel override must never outlive STOPPING
                # (cancel, preemption, settle, and blend all clear the flag;
                # this guarantees the worker can't keep consuming a stale
                # decayed target as the next episode's intent).
                if not stopping:
                    if _MODEL_STOP_DECEL_REF[0] is not None:
                        _MODEL_STOP_DECEL_REF[0] = None
                    if _STOP_ALIGN_ACTIVE[0]:
                        _STOP_ALIGN_ACTIVE[0] = False
                        backend._rate_scale = 1.0
                    stop_align_cut = None
                    stop_brake_until = 0.0

                if not is_playing:
                    _publish_anchor_tick()
                else:
                    with replan_lock:
                        # Resampled read (30 Hz model -> OUTPUT_FPS) with an
                        # 8-tick cross-fade blend at each replan seam. Future
                        # slot k is +0.1*(k+1) s in REAL time (STEP_TICKS=5
                        # output ticks * 0.6 native/tick = 3 native = 0.1 s).
                        qpos_np = backend.get_next_frame_resampled(OUTPUT_FPS)
                        if backend.should_replan():
                            replan_event.set()
                        future_qposes: list[np.ndarray] = []
                        for k in range(NUM_FUTURE):
                            future_qposes.append(
                                backend.peek_output_frame(STEP_TICKS * (k + 1))
                            )

                    # ---- Instant yaw-rate CAP (2026-08-03): the trim
                    # governor acts only at replan boundaries, so the FIRST
                    # plan of a turn can serve a fast template for ~1-1.5 s
                    # before correction (user: "kicks in a bit late"). Cap
                    # the SERVED yaw increment at 1.35x the commanded rate,
                    # tick by tick, by accumulating a yaw offset applied to
                    # the current and future root quats (planner context is
                    # untouched — it continues its own frame, same contract
                    # as the wire rebase). Offset bleeds off outside turns.
                    _cap_on = (abs(float(current_target[0])) > 0.05
                               and abs(float(current_target[1]))
                               + abs(float(current_target[2])) < 0.10)
                    if _cap_on:
                        _raw_yaw = _yaw_of_quat_xyzw(
                            _wxyz_to_xyzw(qpos_np[3:7]))
                        if cap_prev_raw_yaw is not None:
                            _step = _wrap_pi(_raw_yaw - cap_prev_raw_yaw)
                            _lim = (1.35 * abs(float(current_target[0]))
                                    / OUTPUT_FPS)
                            _excess = _step - max(-_lim, min(_lim, _step))
                            cap_yaw_offset = _wrap_pi(cap_yaw_offset - _excess)
                        cap_prev_raw_yaw = _raw_yaw
                    else:
                        cap_prev_raw_yaw = None
                        cap_yaw_offset *= 0.98   # reconverge with planner frame
                    if abs(cap_yaw_offset) > 1e-6:
                        _c2 = math.cos(cap_yaw_offset / 2.0)
                        _s2 = math.sin(cap_yaw_offset / 2.0)

                        def _rz_wxyz(qw):
                            w_, x_, y_, z_ = float(qw[0]), float(qw[1]), \
                                float(qw[2]), float(qw[3])
                            return np.array(
                                [_c2 * w_ - _s2 * z_, _c2 * x_ - _s2 * y_,
                                 _c2 * y_ + _s2 * x_, _c2 * z_ + _s2 * w_],
                                dtype=qw.dtype)
                        qpos_np[3:7] = _rz_wxyz(qpos_np[3:7])
                        for _fq in future_qposes:
                            _fq[3:7] = _rz_wxyz(_fq[3:7])

                    # Early corrective replan: if the (capped) served rate
                    # still overshoots 1.4x mid-plan, pull the next replan
                    # forward instead of waiting for the threshold.
                    if (_cap_on
                            and abs(_SERVED_YAW_RATE_REF[0])
                            > 1.4 * abs(float(current_target[0]))
                            and time.monotonic() - early_retrim_t > 0.4):
                        early_retrim_t = time.monotonic()
                        with replan_lock:
                            backend._force_replan = True
                        replan_event.set()

                    # Yaw-lock mitigation (opt-in diagnostic; port).
                    yaw_locked = (
                        args.yaw_lock_epsilon > 0.0
                        and abs(float(current_target[0])) < args.yaw_lock_epsilon
                    )
                    if yaw_locked:
                        qpos_np[3:7] = current_root_wxyz.astype(qpos_np.dtype)
                        for fq in future_qposes:
                            fq[3:7] = current_root_wxyz.astype(fq.dtype)

                    current_root_xy = qpos_np[:2].astype(np.float64).copy()
                    current_root_z = float(qpos_np[2])
                    # Served root-speed EMAs (model-stop entry/settle signal;
                    # the SIGNED forward component drives the decay law).
                    if prev_root_xy_tick is not None:
                        _dxy = current_root_xy - prev_root_xy_tick
                        served_vel_xy = _dxy * OUTPUT_FPS
                        _inst_mps = float(np.hypot(_dxy[0], _dxy[1])) * OUTPUT_FPS
                        served_speed_mps = (0.8 * served_speed_mps
                                            + 0.2 * _inst_mps)
                        _hy = _yaw_of_quat_xyzw(_wxyz_to_xyzw(qpos_np[3:7]))
                        _inst_fwd = float(_dxy[0] * math.cos(_hy)
                                          + _dxy[1] * math.sin(_hy)) * OUTPUT_FPS
                        served_fwd_mps = 0.8 * served_fwd_mps + 0.2 * _inst_fwd
                    prev_root_xy_tick = current_root_xy.copy()
                    if not yaw_locked:
                        current_root_wxyz = qpos_np[3:7].astype(np.float32).copy()

                    # Served yaw-rate EMA for the in-place yaw governor.
                    _yaw_now = _yaw_of_quat_xyzw(
                        _wxyz_to_xyzw(current_root_wxyz))
                    if prev_served_yaw is not None:
                        _yr = _wrap_pi(_yaw_now - prev_served_yaw) * OUTPUT_FPS
                        _SERVED_YAW_RATE_REF[0] = (
                            0.9 * _SERVED_YAW_RATE_REF[0] + 0.1 * _yr)
                    prev_served_yaw = _yaw_now

                    jpos = ref_smoother.update(
                        qpos_np[7:].astype(np.float32), time.monotonic()
                    )
                    # Remember what SONIC actually last received, so the stop
                    # blend starts from the published frame (exact continuity)
                    # rather than the raw planner frame.
                    jpos = _apply_head_yaw(jpos)
                    last_gait_jpos = jpos.copy()
                    # PLAYING yaw resync (see yaw_trim above): bleed the
                    # published-vs-measured heading error, slew-limited and
                    # deadbanded, only while no turn is commanded.
                    if (args.playing_yaw_resync_dps > 0.0
                            and yaw_offset[0] is not None
                            and abs(float(current_target[0])) < 0.05):
                        m_yaw = measured_yaw.get(
                            max_age_s=POSE_FEEDBACK_MAX_AGE_S)
                        if m_yaw is not None:
                            pub_yaw = _wrap_pi(
                                _yaw_of_quat_xyzw(_wxyz_to_xyzw(qpos_np[3:7]))
                                + _eff_off())
                            err = _wrap_pi(float(m_yaw) - pub_yaw)
                            dead = math.radians(
                                args.playing_yaw_resync_deadband_deg)
                            if abs(err) > dead:
                                cap = (math.radians(
                                    args.playing_yaw_resync_dps) / OUTPUT_FPS)
                                yaw_trim[0] = float(np.clip(
                                    yaw_trim[0] + max(-cap, min(cap, err)),
                                    -0.6, 0.6))
                    wire_xyzw = _reb1(_wxyz_to_xyzw(qpos_np[3:7]))
                    # Clamp on the PLAYING path too, so the limiter's
                    # previous-sample state stays one tick fresh. Without
                    # this it only ticked during IDLE and the PLAYING ->
                    # IDLE step slipped through the dt guard unbounded.
                    # 8 rad/s is ~11x the fastest commanded turn, so this
                    # never shapes a legitimate turn.
                    wire_xyzw = _clamp_ref_yaw_step(wire_xyzw)
                    # Heading counterpart of last_gait_jpos above: the stop blend
                    # must slerp FROM the frame SONIC actually last received.
                    # Must live here, after wire_xyzw exists (it is computed below
                    # the jpos smoothing, not above it).
                    last_gait_quat = np.asarray(wire_xyzw, dtype=np.float64).copy()
                    wire_xy = _reb_xy(current_root_xy)
                    payload = build_pose_payload_np(
                        jpos,
                        wire_xyzw,
                        wire_xy,
                        current_root_z,
                        global_tick,
                        future_jpos=[fq[7:].astype(np.float32)
                                     for fq in future_qposes],
                        future_quat=_rebL([_wxyz_to_xyzw(fq[3:7])
                                           for fq in future_qposes]),
                        hand_dof=args.hand_dof,
                    )
                    publisher.publish(payload)
                    _TAPE.frame(0.0, wire_xy, current_root_z, wire_xyzw,
                                getattr(publisher, "last_wire_jpos", None) if getattr(publisher, "last_wire_jpos", None) is not None else jpos)
                    global_tick += 1

                    # ---- Model-stop resolution: hand off to the anchor
                    # blend once the DECELERATION has done its work. The
                    # starvation clause is the 2026-08-03 revert's ring-fill
                    # guarantee -- a dry ring serves frozen frames, which
                    # must never masquerade as a settle.
                    if stopping:
                        if stop_align_cut is None:
                            # Decay law: keep the worker's override at HALF
                            # the served forward speed (G1's per-second
                            # halving, continuously resampled). Once the
                            # halved target falls below the trackable floor,
                            # hand over to the foot-align cut on the current
                            # (last decel) buffer.
                            _decel = 0.5 * served_fwd_mps
                            if abs(_decel) < _MODEL_STOP_ZERO_SNAP_MPS:
                                if (_STOP_BRAKE_PULSE_S > 0.0
                                        and stop_brake_until == 0.0):
                                    # Counter-brake: one reverse-target plan
                                    # (the operator's stick-pull trick) to
                                    # actively cancel momentum before the
                                    # cut. The cut arms when the pulse
                                    # expires.
                                    _MODEL_STOP_DECEL_REF[0] = (
                                        0.0, 0.0,
                                        -math.copysign(_TEST_FIXED_BACK_MPS,
                                                       served_fwd_mps),
                                        _HIP_HEIGHT_M)
                                    with replan_lock:
                                        backend._force_replan = True
                                    replan_event.set()
                                    stop_brake_until = (time.monotonic()
                                                        + _STOP_BRAKE_PULSE_S)
                                    log.info("model-stop: counter-brake "
                                             "pulse %.2fs", _STOP_BRAKE_PULSE_S)
                                elif (stop_brake_until > 0.0
                                        and time.monotonic() < stop_brake_until):
                                    pass   # brake plan serving; hold target
                                elif _STOP_FOOT_ALIGN:
                                    _MODEL_STOP_DECEL_REF[0] = None
                                    with replan_lock:
                                        stop_align_start_pos = float(
                                            backend._read_pos)
                                        stop_align_cut = _find_foot_align_cut(
                                            backend._buf,
                                            stop_align_start_pos)
                                    _STOP_ALIGN_ACTIVE[0] = True
                                    log.info("model-stop: decay floor -> "
                                             "foot-align cut armed")
                                else:
                                    _MODEL_STOP_DECEL_REF[0] = (
                                        0.0, 0.0, 0.0, _HIP_HEIGHT_M)
                            else:
                                _MODEL_STOP_DECEL_REF[0] = (
                                    0.0, 0.0, _decel, _HIP_HEIGHT_M)
                        if (stop_align_cut is not None
                                and _STOP_ALIGN_SLOW_RATE < 1.0):
                            # Approach slowdown: ease playback toward the
                            # cut so the closing step decelerates instead of
                            # landing at full stride speed.
                            _span = max(1e-6, stop_align_cut
                                        - stop_align_start_pos)
                            _frac = max(0.0, min(1.0, (
                                stop_align_cut - float(backend._read_pos))
                                / _span))
                            backend._rate_scale = (
                                _STOP_ALIGN_SLOW_RATE
                                + (1.0 - _STOP_ALIGN_SLOW_RATE) * _frac)
                        _cut_hit = (stop_align_cut is not None
                                    and float(backend._read_pos)
                                    >= stop_align_cut)
                        if served_speed_mps < _MODEL_STOP_SETTLED_MPS:
                            stop_settle_count += 1
                        else:
                            stop_settle_count = 0
                        _starved = getattr(backend, "_starved_ticks", 0) > 2
                        _timed_out = time.monotonic() > stopping_deadline
                        _exit_wanted = (_cut_hit or stop_settle_count >= 8
                                        or _timed_out or _starved)
                        # Ankle-neutral blend latch: hold the exit (keep
                        # serving decel gait) until the served frame has
                        # both ankles near neutral, up to
                        # _STOP_LATCH_MAX_TICKS. Deadline/starvation arm
                        # immediately (safety). This is the gate that a
                        # precomputed cut index cannot provide: it reads
                        # the frame ACTUALLY served this tick, immune to
                        # buffer swaps and exit-path differences.
                        _neutral_now = (
                            last_gait_jpos is not None
                            and abs(float(last_gait_jpos[_ANKLE_PITCH_L_DOF]))
                            <= _STOP_ALIGN_ANKLE_MAX_RAD
                            and abs(float(last_gait_jpos[_ANKLE_PITCH_R_DOF]))
                            <= _STOP_ALIGN_ANKLE_MAX_RAD
                            and abs(float(last_gait_jpos[_KNEE_L_DOF]))
                            <= _STOP_LATCH_KNEE_MAX_RAD
                            and abs(float(last_gait_jpos[_KNEE_R_DOF]))
                            <= _STOP_LATCH_KNEE_MAX_RAD)
                        if (_exit_wanted and not (_timed_out or _starved)
                                and not _neutral_now
                                and stop_latch_wait < _STOP_LATCH_MAX_TICKS):
                            stop_latch_wait += 1
                            _exit_wanted = False
                        if _exit_wanted:
                            log.info(
                                "model-stop: %s at %.2f m/s -> anchor blend"
                                " (ankle latch %d ticks, neutral=%s)",
                                ("foot-align cut" if _cut_hit else
                                 "STARVED (fallback)" if _starved else
                                 "deadline" if _timed_out else "settled"),
                                served_speed_mps, stop_latch_wait,
                                _neutral_now,
                            )
                            _TAPE.ev("model_stop_done",
                                     cut=bool(_cut_hit),
                                     starved=bool(_starved),
                                     timed_out=bool(_timed_out),
                                     latch_ticks=int(stop_latch_wait),
                                     neutral=bool(_neutral_now),
                                     mps=round(served_speed_mps, 3))
                            stop_latch_wait = 0
                            stop_align_cut = None
                            _STOP_ALIGN_ACTIVE[0] = False
                            backend._rate_scale = 1.0
                            stop_brake_until = 0.0
                            stopping = False
                            is_playing = False
                            if last_gait_jpos is not None:
                                stop_blend_from = last_gait_jpos.copy()
                                stop_blend_quat_from = (
                                    None if last_gait_quat is None
                                    else np.asarray(last_gait_quat,
                                                    dtype=np.float64).copy())
                                stop_blend_total = _stop_blend_total_for(
                                    last_gait_jpos, anchor_jpos)
                                stop_blend_left = stop_blend_total
                            stop_blend_xy_vel = served_vel_xy.copy()
                            _SERVED_YAW_RATE_REF[0] = 0.0
                            prev_served_yaw = None
                            cap_prev_raw_yaw = None
                            cap_yaw_offset = 0.0
                            served_speed_mps = 0.0
                            prev_root_xy_tick = None

                next_tick += period_s
                slack = next_tick - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    if -slack > 5 * period_s:
                        log.warning("loop fell behind by %.0fms; resyncing",
                                    -slack * 1000)
                        next_tick = time.monotonic()
            log.info("main loop exited at tick %d", global_tick)
    finally:
        stop_event.set()
        replan_event.set()
        publisher.close()
        for thr in threads:
            thr.join(timeout=2.0)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pc2_kplanner_onnx",
        description=(
            "Torch-free X2 kinematic planner runtime for PC2 (onnxruntime "
            "fused graph); slim port of x2_kplanner's publish/replan loop "
            "with built-in x2m2 dance playback."
        ),
    )
    p.add_argument("--backend", choices=("onnx", "torch"), default="onnx",
                   help="onnx (PC2 default, torch-free) or torch (laptop A/B).")
    p.add_argument("--onnx", type=Path, default=None,
                   help="Fused planner graph (.onnx) for --backend onnx.")
    p.add_argument("--onnx-sidecar", type=Path, default=None,
                   help="JSON runtime contract (tensor names etc.); default "
                        "<onnx>.json or <dir>/runtime_contract.json, else "
                        "the export-script defaults.")
    p.add_argument("--planner-mode", choices=_PLANNER_MODE_NAMES, default=None,
                   help="Pose-template mode name (template graph / "
                        "replan_with_pose_template). Default None = "
                        "velocity-only path.")
    # torch backend checkpoints (laptop A/B only).
    p.add_argument("--vqvae-ckpt", type=Path, default=None)
    p.add_argument("--pose-ckpt", type=Path, default=None)
    p.add_argument("--root-ckpt", type=Path, default=None)
    p.add_argument("--device", default="cpu",
                   help="torch device for --backend torch (default cpu).")

    p.add_argument("--warmup-qpos", type=Path, default=DEFAULT_WARMUP_PKL,
                   help=f"Idle-anchor PKL (joblib; deploy-PKL or raw qpos "
                        f"schema). Default {DEFAULT_WARMUP_PKL}")
    p.add_argument("--warmup-quiet-stand-s", type=float, default=0.5)

    net = p.add_argument_group("network (real PC2 defaults; use "
                               "--port-offset for laptop testing)")
    net.add_argument("--pub-host", default="0.0.0.0",
                     help="PUB bind host (default 0.0.0.0 = tcp://*).")
    net.add_argument("--pub-port", type=int, default=DEFAULT_PUB_PORT)
    net.add_argument("--pub-topic", default="pose")
    net.add_argument("--cmd-host", default="127.0.0.1",
                     help="planner_cmd SUB connect host (pad bridge --bind).")
    net.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    net.add_argument("--cmd-topic", default="planner_cmd")
    net.add_argument("--cmd-bind", action="store_true",
                     help="bind the planner_cmd SUB instead of connecting, so "
                          "multiple sources (pad bridge + quest3 manager) can "
                          "PUB-connect into it; --cmd-host is ignored.")
    net.add_argument("--arm-port", type=int, default=5572,
                     help="VR arm/hand target ingest: SUB bind port the "
                          "laptop quest3 manager PUB-connects to with "
                          "--arm-connect (arm_targets + hand_finger_cmd; "
                          "overlaid onto the pose wire). 0 disables.")
    net.add_argument("--clip-cmd-port", type=int, default=DEFAULT_CLIP_CMD_PORT,
                     help="motion_clip_cmd SUB bind port.")
    net.add_argument("--clip-cmd-topic", default="motion_clip_cmd")
    net.add_argument("--x2-debug-host", default="127.0.0.1",
                     help="deploy x2_debug PUB host (measured-yaw rebase).")
    net.add_argument("--x2-debug-port", type=int, default=5557,
                     help="deploy x2_debug PUB port; <=0 disables the SUB.")
    net.add_argument("--x2-debug-topic", default="x2_debug")
    net.add_argument("--no-yaw-rebase", action="store_true",
                     help="disable measured-yaw rebase of published root "
                          "quats (published frames stay in world +X frame; "
                          "SONIC will twist to spawn heading -- regression "
                          "escape only).")
    net.add_argument("--yaw-capture-timeout-s", type=float, default=30.0,
                     help="how long to stay silent waiting for the first "
                          "x2_debug frame (deploy starts AFTER this planner) "
                          "before proceeding without rebase. The watchdog "
                          "holds the robot during the wait, so a generous "
                          "value is safe on the robot; sim has no x2_debug so "
                          "pass --no-yaw-rebase there.")
    net.add_argument("--port-offset", type=int, default=0,
                     help="Added to pub/cmd/clip ports (laptop testing; the "
                          "live stack owns 5556/5563/5568).")

    p.add_argument("--dances-dir", type=Path, default=DEFAULT_DANCES_DIR,
                   help=f"Directory of <motion_key>.x2m2 bakes. Default "
                        f"{DEFAULT_DANCES_DIR}")
    p.add_argument("--side-step-primitive", action="store_true",
                   default=os.environ.get(
                       "KPLANNER_SIDE_STEP_PRIMITIVE", "0") == "1",
                   help="EXPERIMENTAL: route standing lateral commands to the "
                   "side-step primitive clips. Default OFF -> model gait "
                   "(env KPLANNER_SIDE_STEP_PRIMITIVE=1 enables). Prior "
                   "bakes all failed SONIC-in-the-loop (synthetic: "
                   "untracked; locomanip: z-dip near-fall; lateral-gait "
                   "cycle: hopping; A038_M diagonal: cross-step scissor). "
                   "2026-08-09 lead-leg rebake from lateral_speed_step "
                   "mocap: operator validation PENDING.")
    p.add_argument("--stop-settle-timeout-s", type=float, default=2.5,
                   help="max seconds to let the model generate the stop "
                   "(zero-velocity replan) before forcing the anchor blend.")
    p.add_argument("--crouch-primitive", default="crouch_medium",
                   help="primitive clip scrubbed by the ARM_MAN squat axis "
                   "(hold_torso hip_height_m below default).")
    p.add_argument("--crouch-hip-span", type=float, default=0.09,
                   help="hip_height_m drop (m) that maps to FULL crouch "
                   "depth; matches the VR manager's intent_max_height_down_m.")
    p.add_argument("--primitives-pkl", type=str, default="",
                   help="x2_planner_primitives.pkl path (crouch_*/lean_*/"
                   "torso_* playback); auto-resolved next to the script or "
                   "the repo data dir when unset")
    p.add_argument("--post-dance-idle-s", type=float, default=2.0,
                   help="Idle-anchor stream duration after a clip ends or "
                        "is stopped, before the planner resumes.")

    p.add_argument(
        "--replan-threshold-frames", type=int, default=32,
        help="Replan when this many model frames (30fps) remain. Was 16 = "
             "0.53s, which PC2's 0.3-0.6s CPU inference consumed entirely, "
             "starving the ring at every mid-walk seam (tape 20260719). 32 "
             "gives ~0.4s commit margin at worst-case latency.")
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--hand-dof", type=int, default=DEFAULT_HAND_DOF)
    p.add_argument("--ort-gpu", action="store_true",
                   help="request CUDAExecutionProvider (CPU fallback). No-op "
                        "unless the venv has a Jetson GPU build of onnxruntime. "
                        "GPU inference (~tens of ms vs 0.3-0.6s CPU) would also "
                        "let --replan-threshold-frames drop back toward 16.")
    p.add_argument("--ort-trt", action="store_true",
                   help="put TensorrtExecutionProvider ahead of CUDA "
                        "(engine cache at ~/.cache/kplanner_trt or "
                        "KPLANNER_TRT_CACHE; KPLANNER_TRT_FP16=1 for fp16). "
                        "Jetson ORT-GPU baseline: replan p50 75 ms "
                        "(2026-08-11). STATUS: the CURRENT template graph "
                        "FAILS under TRT 10.7 -- its sampler ScatterND ops "
                        "are rejected by the importer and session creation "
                        "died on the Jetson (2026-08-11 test). The flag "
                        "becomes useful after a TRT-clean re-export "
                        "(sampler split out of the graph, host-side seed -> "
                        "noise; TRT engine for the dense core).")
    p.add_argument("--pid-file", type=Path,
                   default=Path("/tmp/pc2_kplanner_onnx.pid"))

    tune = p.add_argument_group("velocity tuning (ported from x2_kplanner)")
    tune.add_argument("--speed-setpoint", type=float, default=None,
                      help="Initial forward speed setpoint m/s (default 0.3 "
                           "or KPLANNER_FIXED_FWD_MPS env).")
    tune.add_argument("--turn-left-scale", type=float, default=1.0)
    tune.add_argument("--turn-right-scale", type=float, default=1.0)
    tune.add_argument("--forward-scale", type=float, default=1.0)
    tune.add_argument("--backward-scale", type=float, default=1.0)
    tune.add_argument("--lateral-scale", type=float, default=1.0)
    tune.add_argument("--stick-shape-exp", type=float,
                      default=_DEFAULT_STICK_SHAPING_EXPONENT)
    tune.add_argument("--continuous-forward-min-mps", type=float,
                      default=_DEFAULT_CONTINUOUS_FORWARD_MIN_MPS)
    tune.add_argument("--yaw-lock-epsilon", type=float, default=0.0)
    tune.add_argument("--playing-yaw-resync-dps", type=float, default=0.0,
                      help="If >0, bleed published-vs-measured heading error "
                           "DURING walks at this slew rate (deg/s), deadbanded, "
                           "gated off while a turn is commanded. Assignment-"
                           "form wire trim; 0 = off (legacy). Try ~10.")
    tune.add_argument("--playing-yaw-resync-deadband-deg", type=float,
                      default=2.0,
                      help="No resync while |error| is under this (noise).")
    tune.add_argument(
        "--cold-start-ramp-tau-s", type=float,
        default=float(os.environ.get("KPLANNER_COLD_START_RAMP_TAU_S") or 0.6),
        help="first-order ramp on the replan velocity target after idle "
             "(the start-side twin of the model-stop decay law; ~2-3 plans "
             "to full speed at 0.6). 0 disables = legacy instant-velocity "
             "starts. Env: KPLANNER_COLD_START_RAMP_TAU_S.")
    tune.add_argument("--command-watchdog-s", type=float, default=0.0)

    sm = p.add_argument_group("reference-step smoother")
    sm.add_argument("--ref-smoother-ms", type=float,
                    default=float(os.environ.get("KPLANNER_REF_SMOOTHER_MS") or 300.0))
    sm.add_argument("--ref-smoother-trigger-rad", type=float, default=0.05)
    # Default OFF: the 30->50 Hz output resampling + 8-tick cross-fade blend
    # now handles replan-seam discontinuities at the source; the ref-smoother
    # is opt-in (pass --ref-smoother-shape halfcos to re-enable).
    sm.add_argument("--ref-smoother-shape", choices=_REF_SMOOTHER_SHAPES,
                    default="off")
    sm.add_argument("--ref-smoother-joints",
                    choices=list(_REF_SMOOTHER_JOINTS_PRESETS),
                    default="lower_body")

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
