"""Neural kinematic planner daemon for AgiBot X2 Ultra.

Streams 31-DOF reference body poses + root quaternion at 50 Hz on the
``body_pose`` (or legacy ``pose``) ZMQ topic, the same wire format as
``x2_heuristic_planner.py`` -- so downstream subscribers (recorder, MuJoCo
viewer, C++ deploy) are protocol-compatible.

Source of motion: a trained MotionBricks stack
(``motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/``). Source
of velocity intent: hard-coded ``INTENT_VELOCITY_MAP`` keyed on ``(intent,
magnitude)`` from the Quest3 / scripted command sources.

Command surface (any combination, mirrors the heuristic daemon's UX):
  - ``--demo PATH.yaml`` : scripted command sequence
  - ``--keyboard``       : interactive keyboard mode (TTY only)
  - ``--zmq-cmd-port``   : ZMQ SUB on ``planner_cmd`` topic for external drive

Architecture (per
``docs/source/references/x2_quest3_planner_stack_architecture.md`` updated
for kplanner):

  - Main thread runs the 50 Hz publish loop (``PosePublisher.publish``),
    pulling one frame per tick out of the ``NeuralPlannerCore`` ring buffer.
  - Worker thread monitors the ring-buffer occupancy. When it drops below
    ``--replan-threshold-frames`` it calls
    ``NeuralPlannerCore.replan_with_velocity(target)`` to refill. ``predict()``
    can take 5--15 ms on GPU; running it in a worker keeps the publish thread
    on its 20 ms tick.
  - The current ``target_local_root_values`` is updated atomically from the
    latest received ``LocomotionCommand`` via ``INTENT_VELOCITY_MAP``.

Run from the repo root::

    .venv/bin/python -m gear_sonic.scripts.x2_kplanner \\
        --body-pose-port 5555 \\
        --zmq-cmd-host 127.0.0.1 --zmq-cmd-port 5557

See ``run_x2_quest3_planner_stack.sh --planner kplanner`` for the integrated
launcher that wires this up with the recorder + deploy + Quest3 stack.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import pickle
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw  # noqa: E402
from gear_sonic.utils.planner.state_machine import (  # noqa: E402
    HOLD_HIP_PITCH_SHARE,
    HOLD_HIP_YAW_SHARE,
    HOLD_ANKLE_PITCH_SHARE,
    HOLD_ANKLE_ROLL_SHARE,
    HOLD_SLEW_DPS,
    HOLD_TORSO_INTENT,
    LocomotionCommand,
    OUTPUT_FPS,
    PlannerState,
    StreamFrame,
    _HoldTracker,
    build_pose_payload,
    commands_from_yaml,
)
from gear_sonic.utils.planner.constants import (  # noqa: E402
    WAIST_PITCH_IDX,
    WAIST_ROLL_IDX,
    WAIST_YAW_IDX,
)
from gear_sonic.utils.planner.x2_recipes import make_waist_pose_frame  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402


log = logging.getLogger("x2_kplanner")


# ---------------------------------------------------------------------------
# Intent -> velocity dispatcher. The neural model consumes a continuous
# 4-D vector ``(yaw_rate_rad_s, vel_x_m_s, vel_z_m_s, hip_height_m)``.
#
# CHANNEL CONVENTION (verified against the X2 + G1 mujoco converter and
# the LocalRootLocalBody motion-rep, see
# ``motionbricks/scripts/probe_root_constraint_modes.py``):
#
#   * ``vel_x`` = motion-rep X = MuJoCo Y = **lateral** (positive = +X
#     in the motion-rep frame = robot's LEFT after canonicalization).
#   * ``vel_z`` = motion-rep Z = MuJoCo X = **forward** (positive = +Z
#     in the motion-rep frame = robot's FORWARD after canonicalization).
#
# The earlier docstring claim that "vel_z is lateral" was WRONG and the
# original ``_BASE_VELOCITY`` table fed forward speed into vel_x, which
# the model interpreted as a lateral command. Combined with the
# velocity-only target masking in NeuralPlannerCore (now fixed), this
# explains the historical "robot doesn't walk forward" symptom.
#
# Vocabulary contract with ``quest3_manager_x2``: the manager's
# ``IntentDecoder`` was authored for the heuristic planner's curated
# motion bins (``deg_15/30/45/90`` turns, ``quarter_ft / half_ft /
# one_ft`` strides). The kplanner does NOT honour those labels as
# literal angles or distances -- for the neural model they are
# intensity buckets scaling a continuous velocity. To avoid locking
# the kplanner into the heuristic's labels we split the table:
#
#   ``_BASE_VELOCITY``: direction-explicit intent -> 1× velocity vector.
#                       Direction is encoded in the intent name
#                       (``fwd_step`` vs ``back_step``, ``turn_left``
#                       vs ``turn_right``, etc.).
#   ``_TRANSLATIONAL_SCALE`` / ``_TURN_SCALE``: magnitude string ->
#                       scalar multiplier (``default`` = 1.0). Unknown
#                       magnitudes fall back to 1.0 so manager-side
#                       vocabulary additions don't silently idle.
#
# ``INTENT_VELOCITY_MAP`` is derived at import time from the dispatcher
# for docs / debug dumps. The two surfaces are guaranteed to agree
# because they go through the same ``_resolve_velocity()`` path.
#
# Numbers are intentionally conservative on the kinematic side; we can
# raise them once the deploy is verified stable on each direction.
# ---------------------------------------------------------------------------

_WALK_SPEED_MPS: float = 0.6
_FAST_WALK_SPEED_MPS: float = 0.9
_SIDE_SPEED_MPS: float = 0.4
_BACK_SPEED_MPS: float = 0.45

# Minimum forward velocity command (m/s) emitted whenever the operator
# commits any non-zero forward stick deflection in the
# continuous-locomotion path. Default 0.30 m/s lands the SONIC X2
# root model inside its in-distribution forward-walk band as soon as
# the operator pushes past the deadzone. Set to 0.0 to disable the
# floor entirely (pre-2026-05-31 legacy behaviour).
#
# Rationale: ``x2_ultra_locowalk.pkl`` forward-walk training samples
# are concentrated between roughly 0.3 - 1.0 m/s, with essentially no
# coverage below 0.3 m/s. Commanding 0 < vel_z < 0.3 puts the policy
# OOD -- empirically the robot wiggles its hips but never initiates
# a step, then snaps into an unstable stride once vel_z finally
# exceeds the training-band minimum (operator-reported 2026-05-31).
# The backward channel does not need this floor because the same
# corpus has dense backward samples all the way down to ~-0.08 m/s.
#
# 0.30 m/s was picked as the default because it sits just above the
# training-band p1 (~0.28 m/s) -- the smallest forward speed the
# policy has more than a handful of frames of. Operators who want
# more aggressive lift-off (faster, less smooth deadzone transition)
# can bump to ~0.40 m/s; operators who want to revert to raw
# proportional control (and accept the hip-wiggle below ~0.3 m/s)
# can set 0.0.
#
# The floor is applied **post** ``_RUNTIME_FORWARD_SCALE`` in
# ``_apply_continuous_runtime_scales`` so it acts as a true
# operator-felt minimum regardless of any scale override. Operators
# who want progressive forward control above the floor should keep
# ``floor < forward_scale * _WALK_SPEED_MPS`` (i.e.
# ``floor < 0.5 m/s`` at scale=1.0); otherwise every forward stick
# value collapses onto the floor. Tunable via
# ``--continuous-forward-min-mps`` (CLI) or
# ``KPLANNER_CONTINUOUS_FORWARD_MIN_MPS`` (env var on the
# Quest 3 wrapper). The bucketed forward intents and the PKL replay
# path are intentionally untouched -- the floor only fires when the
# operator is on the analog stick.
_DEFAULT_CONTINUOUS_FORWARD_MIN_MPS: float = 0.30

# Mutable runtime knob, set from ``run()`` per CLI flag. Reads in
# ``_apply_continuous_runtime_scales`` pick up the override on every
# dispatch call.
_RUNTIME_CONTINUOUS_FORWARD_MIN_MPS: float = _DEFAULT_CONTINUOUS_FORWARD_MIN_MPS

# --------------------------------------------------------------------------
# Fixed-speed movement test (continuous VR path).
#
# Temporary single-magnitude speeds so the operator can evaluate slow-
# movement feel with fully predictable velocities: any stick deflection
# past the deadzone produces the fixed magnitude below (direction from
# the stick sign), instead of analog scaling / discrete forward levels.
# To restore analog + discrete-forward behaviour, revert
# ``_resolve_locomotion_continuous`` and remove these constants.
# --------------------------------------------------------------------------
# Forward is ANALOG: the shaped forward stick scales from the in-distribution
# floor (0.30 m/s -- below this the corpus is empty and the robot just wiggles
# its hips without stepping) up to the walk peak (_WALK_SPEED_MPS = 0.60). The
# old fixed 0.30 "barely moved"; this lets the operator push into the 0.3-0.6
# walking band. Back / side / turn stay fixed-magnitude for now.
_FWD_MIN_MPS: float = 0.30
_FWD_MAX_MPS: float = _WALK_SPEED_MPS  # 0.60

# Runtime forward-speed SETPOINT (m/s): any forward stick deflection past the
# deadzone commands exactly this speed (deterministic; replaces the analog
# band). Adjustable live from the VR X/Y face buttons in LOCOMOTION mode
# (payload "speed_delta": X = -0.1, Y = +0.1), clamped to
# [_SETPOINT_MIN, _SETPOINT_MAX]. Default 0.3; KPLANNER_FIXED_FWD_MPS env var
# overrides the INITIAL value (kept for script compatibility).
_SETPOINT_MIN: float = 0.2
_SETPOINT_MAX: float = 1.0
_SPEED_SETPOINT: float = float(os.environ.get("KPLANNER_FIXED_FWD_MPS") or 0.30)


def _adjust_speed_setpoint(delta: float) -> float:
    """Nudge the runtime forward-speed setpoint; returns the new value."""
    global _SPEED_SETPOINT
    _SPEED_SETPOINT = max(_SETPOINT_MIN, min(_SETPOINT_MAX, _SPEED_SETPOINT + delta))
    log.info("speed setpoint %+0.1f -> %.1f m/s", delta, _SPEED_SETPOINT)
    return _SPEED_SETPOINT


# Back-compat alias: non-None means "fixed/setpoint mode" (always on now).
_FIXED_FWD_MPS: Optional[float] = _SPEED_SETPOINT
_TEST_FIXED_FORWARD_MPS: float = 0.30  # (superseded by the analog band above)
_TEST_FIXED_BACK_MPS: float = 0.30
_TEST_FIXED_SIDE_MPS: float = 0.30
# In-place / continuous-stick turn rate. Deploy default stays 0.30 (parity);
# KPLANNER_FIXED_TURN_RAD_S overrides (turnrate sweep certified up to 1.0,
# module guidance above puts 0.75 as the comfort ceiling — see the
# _CONTINUOUS_TURN_MAX_RAD_S discussion; the analog path that knob served
# was replaced by this fixed magnitude).
_TEST_FIXED_TURN_RAD_S: float = float(
    os.environ.get("KPLANNER_FIXED_TURN_RAD_S") or 0.30)
# Forward-dominance lateral deadband on the LEFT stick: while the
# operator is pushing forward, the shaped side-stick magnitude must
# exceed this (wider than the normal stick deadzone) before ANY lateral
# velocity is produced, so a slightly-canted left stick doesn't
# accidentally bleed into a strafe during forward walking. Pure lateral
# (forward stick centred) is unaffected -- it uses the normal deadzone.
_FWD_LATERAL_DEADBAND: float = 0.35

# Discrete VR forward-speed levels. The analog forward stick snaps to one of
# these two speeds instead of scaling continuously, with a hysteresis deadband
# around the edge so a stick hovering near the boundary doesn't flicker between
# levels (mirrors the G1 VR ``_discretize_slow_walk_speed`` behaviour). Only the
# continuous forward channel is affected; backward / lateral / yaw stay analog.
# NOTE: superseded by the fixed-speed test above -- kept for easy revert.
_VR_FORWARD_LEVELS: tuple[float, float] = (0.3, 0.5)
_VR_FORWARD_EDGE: float = 0.6      # shaped-forward-stick magnitude splitting the two levels
_VR_FORWARD_HYST: float = 0.04     # deadband around the edge (anti-flicker)
# Module state: the level currently latched. Reset to the low level whenever the
# operator is not pushing forward, so each new forward push starts slow.
_vr_forward_level: float = _VR_FORWARD_LEVELS[0]
# Per-step yaw rate baseline; magnitude scalars in ``_TURN_SCALE`` rescale.
_TURN_15_RAD_S: float = 0.5
_TURN_30_RAD_S: float = 1.0
_TURN_45_RAD_S: float = 1.5
_TURN_90_RAD_S: float = 3.0
# Yaw-rate ceiling for the **continuous-locomotion** path (Quest 3
# R-stick X axis after deadzone rescale + stick shaping). Decoupled
# from the bucketed ``_TURN_45_RAD_S`` because:
#
#   * The current X2 root model trained on a forward-walk loop
#     template, which contains essentially zero
#     yaw motion. Any non-zero yaw_rate is extrapolation; the larger
#     the yaw the more OOD the prediction. Empirically full-stick at
#     1.5 rad/s (~86 deg/s = a 90 deg turn in ~1 s) overdrives the
#     model and the operator reports "turns are too aggressive". A
#     90 deg turn in ~2 s (0.75 rad/s, ~43 deg/s) sits comfortably
#     inside what the policy can track.
#   * Bucketed callers (``turn_left / deg_45`` etc.) intentionally
#     want sharp pivots from a single button press -- those continue
#     to use the legacy ``_TURN_*_RAD_S`` table.
#
# Tunable at runtime via ``--continuous-turn-max-rad-s`` on the
# kplanner CLI, or ``KPLANNER_CONTINUOUS_TURN_MAX_RAD_S`` env var on
# the Quest 3 / PKL wrappers. The per-side ``_RUNTIME_TURN_LEFT_SCALE
# / _RIGHT_SCALE`` runtime scales still apply on top (default 1.0)
# so the operator can compensate for any L/R asymmetry independently.
_DEFAULT_CONTINUOUS_TURN_MAX_RAD_S: float = 0.50

# Mutable runtime knob, mutated by ``run()`` per CLI flag. Reads in
# ``_resolve_locomotion_continuous`` pick up the override at every
# dispatch call.
_CONTINUOUS_TURN_MAX_RAD_S: float = _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S
# Default hip-height TARGET (channel 3 of the velocity intent the model
# consumes). This is NOT a metadata field -- ``NeuralPlannerCore``
# wires it into ``implied_target_y`` (see
# ``motion_backbone/inference/neural_planner.py:414``), so the model
# treats it as the world-frame pelvis Y the predicted gait must drive
# toward.
#
# Must therefore match the **training distribution** of pelvis height.
# The X2 PKL corpus (``x2_ultra_locowalk.pkl``) has pelvis_z spanning
# roughly 0.595 - 0.726 m with a mean around 0.661 m; the working
# PKL-replay sweep (``x2_pkl_command_source --use-mean-intent``)
# computed 0.687 m and produced the only configuration where the
# deploy actually walked (deploy-integration diagnostics of the
# kplanner evaluation harness).
#
# The historical value (0.95 m, "matches idle_stand") was a stale
# carry-over from an earlier checkpoint whose stand pose sat ~25 cm
# higher than the current ``_TRAINING_DEFAULT_HIP_Z = 0.636 m``.
# Feeding 0.95 to the current model puts the target pelvis ~25 cm
# above any pose in its training distribution; the model produces
# OOD predictions and the policy can't track them, presenting to the
# operator as "robot won't walk forward even on full stick". This
# was the dominant failure mode on the Quest 3 stack before
# 2026-05-30. PKL replay was insulated because ``direct_velocity``
# carries hip_h verbatim from the clip (0.687 m).
#
# 2026-05-30 PM iteration log:
#   0.687 -> 0.636: dropped to _TRAINING_DEFAULT_HIP_Z after MC scan
#     showed the policy actively extending the legs at idle (knee
#     tgt-def ~-0.45 rad, ankle plantarflexed ~+0.28 rad) producing
#     a visible "stands taller than MC" posture + 1-2 Hz rocking sway.
#     Operator feedback: "height changes, but barely noticeable, and
#     still taller than MC height."
#   0.636 -> 0.55: cheap sensitivity probe. 0.55 m is 45 mm BELOW the
#     PKL training-distribution minimum (0.595), so it's intentionally
#     OOD on the low side. RESULT: stand height did not visibly change
#     -- a 14 cm reduction in implied_target_y produced ~no visible
#     pelvis drop. Conclusion: ``hip_h`` is largely ignored by the
#     policy at idle; the leg-extension / "tall stand" behavior is
#     driven by something else (most likely default_angles drift
#     between the policy's training-time defaults and the C++ deploy
#     binary, an IMU pitch bias, or the policy's learned neutral
#     pose). Investigation pivot: leave ``_HIP_HEIGHT_M`` at the
#     PKL-replay-validated 0.687 (only config where deploy actually
#     walks) and probe ``default_angles`` next.
#   0.55 -> 0.687: revert per operator decision -- back to last
#     known-good before pivoting to default_angles investigation.
_HIP_HEIGHT_M: float = 0.687

# Returned for any (intent, magnitude) the kplanner has no velocity
# meaning for (``hold_torso``, ``lean_*``, ``torso_*``, ``crouch``,
# unrecognised intents). The publisher's idle-gate compares against this
# tuple to decide whether to freeze on the static anchor.
_IDLE_INTENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, _HIP_HEIGHT_M)

# ==========================================================================
# DEPRECATED: bucketed motion stack.
#
# The bucketed (intent, magnitude) command path -- ``_BASE_VELOCITY``,
# ``_TURN_SCALE`` / ``_TRANSLATIONAL_SCALE``, ``_TURN_{15,30,45,90}_RAD_S``,
# ``_WALK_VELOCITY_BY_MAGNITUDE`` and ``_resolve_velocity`` -- drives the
# legacy single-button-press step / pivot commands. The live operator path
# is the continuous VR stick resolver (``_resolve_locomotion_continuous``).
# The bucketed stack is retained only for scripted demos / legacy manager
# vocabulary and should NOT be extended; prefer the continuous path for all
# new work. Do not wire new operator controls into this table.
# ==========================================================================

# Direction-explicit 1× velocity vector per intent. Magnitude is applied
# separately via ``_TRANSLATIONAL_SCALE`` / ``_TURN_SCALE`` below.
_BASE_VELOCITY: dict[str, tuple[float, float, float, float]] = {
    # (yaw_rate, vel_x=lateral, vel_z=forward, hip_h).
    "idle":       (0.0,             0.0,                  0.0,                _HIP_HEIGHT_M),
    "fwd_step":   (0.0,             0.0,                  _WALK_SPEED_MPS,    _HIP_HEIGHT_M),
    "back_step":  (0.0,             0.0,                 -_BACK_SPEED_MPS,    _HIP_HEIGHT_M),
    "side_left":  (0.0,             _SIDE_SPEED_MPS,      0.0,                _HIP_HEIGHT_M),
    "side_right": (0.0,            -_SIDE_SPEED_MPS,      0.0,                _HIP_HEIGHT_M),
    "turn_left":  ( _TURN_45_RAD_S, 0.0,                  0.0,                _HIP_HEIGHT_M),
    "turn_right": (-_TURN_45_RAD_S, 0.0,                  0.0,                _HIP_HEIGHT_M),
}

# Translational magnitude -> multiplier on (vx, vy). ``default`` is 1.0
# (what the L-stick single-press emits today); the ``*_ft`` rows are
# carry-over from the heuristic vocabulary and treated as intensity
# buckets, NOT literal foot distances. Unknown magnitudes default to 1.0.
_TRANSLATIONAL_SCALE: dict[str, float] = {
    "default":    1.0,
    "stand":      0.0,
    "quarter_ft": 0.5,
    "half_ft":    1.0,
    "one_ft":     1.5,
}

# Rotational magnitude -> multiplier on yaw_rate. Baseline = 1.0 = the
# ``deg_45`` bucket so the legacy ``_TURN_45_RAD_S`` constant remains
# the natural unit. ``deg_*`` labels are intensity buckets, NOT literal
# angles. Unknown magnitudes default to 1.0.
_TURN_SCALE: dict[str, float] = {
    "default": 1.0,
    "deg_15":  _TURN_15_RAD_S / _TURN_45_RAD_S,
    "deg_30":  _TURN_30_RAD_S / _TURN_45_RAD_S,
    "deg_45":  1.0,
    "deg_90":  _TURN_90_RAD_S / _TURN_45_RAD_S,
}

_ROTATIONAL_INTENTS: frozenset[str] = frozenset({"turn_left", "turn_right"})

# ---------------------------------------------------------------------------
# Continuous locomotion (analog Quest3 sticks)
# ---------------------------------------------------------------------------
#
# When the IntentDecoder is run with ``enable_continuous_locomotion=True``
# (the kplanner-mode wrapper flips this on), the manager publishes
# ``locomotion / continuous`` commands carrying three deadzone-rescaled
# stick deflections in ``[-1, 1]``:
#
#   stick_fwd  > 0 -> forward, < 0 -> backward
#   stick_side > 0 -> right,   < 0 -> left
#   stick_yaw  > 0 -> turn-right (negative yaw_rate), < 0 -> turn-left
#
# We shape each axis with a power curve (sign preserved) so the
# operator gets fine resolution near zero with full speed at full
# deflection -- this kills the "binary slam" failure mode that was
# making the SONIC controller fall when the bucketed path stepped the
# velocity 0 -> 0.5 m/s in a single tick. The peak velocities are
# pinned to the same constants the bucketed path uses
# (``_WALK_SPEED_MPS`` etc.) so scripted demos and the kplanner sweep
# harness stay calibrated.
#
# Default exponent = 1.0 (linear). Bigger -> harder to reach full
# speed (more fine control near zero, but operators report "robot
# struggles to move forward" because 50% stick deflection produces
# only 25% velocity at exp=2.0). Smaller -> easier to reach full
# speed (more bucketed-like). Override at runtime via ``--stick-shape-
# exp`` (CLI) or ``KPLANNER_STICK_SHAPE_EXP`` (env var on the
# wrapper). At exp=0.5 a 50% stick gives 71% of max velocity; at
# exp=1.0 a 50% stick gives 50%; at exp=2.0 a 50% stick gives 25%.

_DEFAULT_STICK_SHAPING_EXPONENT: float = 1.0

# Mutable runtime-tuned exponent, set from ``run()`` per CLI flag.
_RUNTIME_STICK_SHAPING_EXPONENT: float = _DEFAULT_STICK_SHAPING_EXPONENT


def _shape_stick(value: float) -> float:
    """Map ``[-1, 1]`` post-deadzone stick deflection to ``[-1, 1]`` velocity.

    Power curve with sign preserved: ``sign(v) * |v|**exp`` where ``exp``
    is the runtime-tuned ``_RUNTIME_STICK_SHAPING_EXPONENT``. Pure helper
    -- the IntentDecoder owns deadzone clamping, so values arrive here
    already in ``[-1, 1]`` (or 0 if inside the deadzone).
    """
    sign = 1.0 if value >= 0 else -1.0
    mag = abs(float(value))
    if mag == 0.0:
        return 0.0
    return sign * mag ** _RUNTIME_STICK_SHAPING_EXPONENT


def _discretize_vr_forward_speed(mag: float) -> float:
    """Snap the shaped forward-stick magnitude to one of ``_VR_FORWARD_LEVELS``
    with a hysteresis deadband so a stick hovering near ``_VR_FORWARD_EDGE``
    doesn't flicker between the two speeds. Latches the chosen level in the
    ``_vr_forward_level`` module global across ticks (mirrors the G1 VR
    ``_discretize_slow_walk_speed`` anti-flicker behaviour)."""
    global _vr_forward_level
    lo, hi = _VR_FORWARD_LEVELS
    if _vr_forward_level <= lo and mag >= _VR_FORWARD_EDGE + _VR_FORWARD_HYST:
        _vr_forward_level = hi
    elif _vr_forward_level >= hi and mag < _VR_FORWARD_EDGE - _VR_FORWARD_HYST:
        _vr_forward_level = lo
    return _vr_forward_level


def _reset_vr_forward_level() -> None:
    """Latch back to the low level; call when the operator is not pushing
    forward so each new forward push starts at the slow speed."""
    global _vr_forward_level
    _vr_forward_level = _VR_FORWARD_LEVELS[0]


def _resolve_locomotion_continuous(
    stick_fwd: float,
    stick_side: float,
    stick_yaw: float,
) -> tuple[float, float, float, float]:
    """Map continuous stick deflections to a 4-D velocity vector.

    Forward / backward use different peak speeds (``_WALK_SPEED_MPS``
    vs ``_BACK_SPEED_MPS``) because the model's training data has
    asymmetric forward / backward stride coverage. Lateral and turn
    axes are symmetric. Returns the ``(yaw_rate, vel_x, vel_z, hip_h)``
    tuple in the same convention as ``_BASE_VELOCITY``.

    The runtime-tuning scalars (``_RUNTIME_FORWARD_SCALE`` etc.) are
    intentionally NOT applied here -- ``_apply_runtime_scales`` handles
    them downstream, mirroring the bucketed path so a single
    ``--kplanner-forward-scale 0.6`` override caps both modes.
    """
    shaped_fwd  = _shape_stick(stick_fwd)
    shaped_side = _shape_stick(stick_side)
    shaped_yaw  = _shape_stick(stick_yaw)

    # Forward-dominance band on the LEFT stick: while pushing forward,
    # ignore small simultaneous lateral deflection so a slightly-canted
    # stick doesn't accidentally bleed into a strafe. Pure lateral
    # (forward centred) keeps the normal deadzone and is unaffected.
    if shaped_fwd > 0.0 and abs(shaped_side) < _FWD_LATERAL_DEADBAND:
        shaped_side = 0.0

    # Forward is ANALOG (floor 0.30 -> peak 0.60 m/s across the shaped
    # stick throw); backward / lateral snap to a single fixed magnitude
    # when engaged (direction from the stick sign); turn snaps to a fixed
    # yaw rate. Deadzone still applies (``_shape_stick`` returns 0 inside
    # it), so a centred stick idles.
    if shaped_fwd > 0.0:
        # Deterministic setpoint mode: ANY forward deflection past the
        # deadzone commands exactly the runtime setpoint (default 0.3 m/s,
        # X/Y buttons nudge it +/-0.1 in [0.2, 1.0]). Replaces the analog
        # 0.3-0.6 band; also reaches run-band speeds the band couldn't.
        vel_z = _SPEED_SETPOINT
    elif shaped_fwd < 0.0:
        vel_z = -_TEST_FIXED_BACK_MPS
    else:
        vel_z = 0.0
    # ``stick_side > 0`` (L-stick right, lx > 0) -> side_right ->
    # negative vel_x, matching ``_BASE_VELOCITY['side_right']``.
    if shaped_side > 0.0:
        vel_x = -_TEST_FIXED_SIDE_MPS
    elif shaped_side < 0.0:
        vel_x = _TEST_FIXED_SIDE_MPS
    else:
        vel_x = 0.0
    # ``stick_yaw > 0`` (R-stick right) -> turn-right -> negative
    # yaw_rate, matching ``_BASE_VELOCITY['turn_right']``.
    if shaped_yaw > 0.0:
        yaw_rate = -_TEST_FIXED_TURN_RAD_S
    elif shaped_yaw < 0.0:
        yaw_rate = _TEST_FIXED_TURN_RAD_S
    else:
        yaw_rate = 0.0
    return (yaw_rate, vel_x, vel_z, _HIP_HEIGHT_M)


# ``walk`` is a manager-side legacy where direction is in the magnitude
# (``forward`` / ``backward``) rather than the intent name. Resolve it
# explicitly so the rest of the dispatcher can assume direction-in-name
# semantics. Unknown ``walk`` magnitudes idle out (safe default for
# unrecognised direction).
_WALK_VELOCITY_BY_MAGNITUDE: dict[str, tuple[float, float, float, float]] = {
    # (yaw_rate, vel_x=lateral, vel_z=forward, hip_h).
    "forward":  (0.0, 0.0,  _WALK_SPEED_MPS,      _HIP_HEIGHT_M),
    "backward": (0.0, 0.0, -_BACK_SPEED_MPS,      _HIP_HEIGHT_M),
    "fast":     (0.0, 0.0,  _FAST_WALK_SPEED_MPS, _HIP_HEIGHT_M),
}

# Runtime tuning scalars applied on top of the resolved velocity. Defaults
# are 1.0 so the dispatcher behaves identically to the static table when
# no CLI overrides are passed. Mutated by ``run()`` when the user passes
# ``--turn-left-scale`` / ``--turn-right-scale`` / ``--walk-scale`` etc.
# to compensate for model-side L/R asymmetry or oversized magnitudes
# without rebuilding the static tables.
_RUNTIME_TURN_LEFT_SCALE: float = 1.0
_RUNTIME_TURN_RIGHT_SCALE: float = 1.0
_RUNTIME_FORWARD_SCALE: float = 1.0
_RUNTIME_BACKWARD_SCALE: float = 1.0
_RUNTIME_LATERAL_SCALE: float = 1.0


def _resolve_velocity(intent: str, magnitude: str) -> tuple[float, float, float, float]:
    """Pure ``(intent, magnitude)`` -> 4-D velocity resolver; idle on miss.

    .. deprecated::
        Bucketed motion stack (see the DEPRECATED banner above
        ``_BASE_VELOCITY``). The live operator path is the continuous VR
        resolver ``_resolve_locomotion_continuous``. Kept for scripted
        demos / legacy manager vocabulary only -- do not extend.

    Direction lives in the intent name for everything except ``walk``
    (legacy: ``walk/forward`` vs ``walk/backward``). Magnitude is a
    scalar multiplier (1.0 = ``default``) and unknown magnitudes fall
    back to 1.0 rather than idling so manager-side vocabulary
    additions don't silently freeze the planner.
    """
    if intent == "walk":
        return _WALK_VELOCITY_BY_MAGNITUDE.get(magnitude, _IDLE_INTENT)
    base = _BASE_VELOCITY.get(intent)
    if base is None:
        return _IDLE_INTENT
    yaw, vx, vy, hip_h = base
    if intent in _ROTATIONAL_INTENTS:
        scale = _TURN_SCALE.get(magnitude, 1.0)
        return (yaw * scale, vx, vy, hip_h)
    scale = _TRANSLATIONAL_SCALE.get(magnitude, 1.0)
    return (yaw, vx * scale, vy * scale, hip_h)


def _apply_runtime_scales(
    intent: str,
    velocity: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Apply runtime-tuning scalars to a resolved velocity.

    Per-direction scales let the operator compensate for model-side
    asymmetries (e.g. weaker left-turn coverage in the training data)
    or shave magnitudes globally without rebuilding the static tables.
    All scales default to 1.0 (no-op) so the existing unit-test
    invariants hold when nothing is overridden.

    Velocity tuple layout: ``(yaw_rate, vel_x=lateral, vel_z=forward,
    hip_h)`` -- matches ``_BASE_VELOCITY`` after the 2026-05-29
    channel-swap bugfix. Prior to that fix this helper scaled the
    wrong axis for ``--kplanner-{forward,backward,lateral}-scale``
    (the pre-fix layout had forward in ``vel_x`` and lateral in
    ``vel_z``); the swap is now applied here too so the CLI knobs
    target the right channel.
    """
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


def intent_to_velocity(cmd: LocomotionCommand) -> tuple[float, float, float, float]:
    """Translate ``LocomotionCommand`` -> 4-D velocity ``(yaw_rate, vx, vy, hip_h)``.

    Returns ``_IDLE_INTENT`` for any intent the kplanner has no velocity
    meaning for (``hold_torso``, ``lean_*``, ``torso_*``, ``crouch``,
    unrecognised intents). Logs a single DEBUG line per miss so missing
    vocabulary additions are visible in the planner log.

    Dispatch order (first match wins):

    1. ``cmd.direct_velocity is not None`` -> the recorded-velocity
       passthrough used by ``x2_pkl_command_source``. Returns the
       4-tuple verbatim with NO shaping and NO runtime scales applied,
       so the PKL replay path lands the exact (yaw, vel_x, vel_z,
       hip_h) extracted from the motion clip into the model. This
       isolates the kplanner -> deploy link from the analog-stick
       shaping curve and the operator's per-direction runtime scales.

    2. ``intent == "locomotion"`` paired with ``magnitude ==
       "continuous"`` -> analog Quest3 sticks. Reads ``cmd.stick_fwd /
       cmd.stick_side / cmd.stick_yaw`` and shapes them via
       ``_resolve_locomotion_continuous``. Runtime tuning scalars still
       apply post-shaping so ``--kplanner-forward-scale`` caps both
       bucketed and continuous modes consistently.

    3. Bucketed ``(intent, magnitude)`` lookup via ``_resolve_velocity``,
       with runtime tuning scalars applied via ``_apply_runtime_scales``.
    """
    if cmd.direct_velocity is not None:
        # PKL replay path: the source publishes a raw 4-D velocity
        # extracted from the recorded motion clip. We honour it
        # verbatim so the wire content matches what
        # _instant_intent_from_clip computed.
        yaw, vx, vz, hip_h = cmd.direct_velocity
        return (float(yaw), float(vx), float(vz), float(hip_h))
    if cmd.intent == "locomotion" and cmd.magnitude == "continuous":
        result = _resolve_locomotion_continuous(
            cmd.stick_fwd, cmd.stick_side, cmd.stick_yaw,
        )
        return _apply_continuous_runtime_scales(result)
    if cmd.intent == HOLD_TORSO_INTENT:
        # ``hold_torso`` keeps the velocity intent at idle (no walking,
        # no turning) so the SONIC policy plants the feet, but honours
        # any operator-supplied hip-height override on channel-3 of
        # the velocity tuple. ``cmd.hip_height_m is None`` -> no
        # override -> fall through to ``_IDLE_INTENT`` -> kplanner
        # default hip height (0.687m). When a finite value is
        # present, substitute it into the channel-3 slot. The waist
        # angles (pitch / roll / yaw) are NOT consumed here -- they
        # ride on a separate kinematic-overlay path applied to the
        # published frame after intent_to_velocity returns; see
        # ``_apply_waist_overlay`` in the publish loop. Without that
        # split the kplanner could not express the "lean while
        # squatting" combo the operator can drive with both sticks.
        yaw_idle, vx_idle, vz_idle, hip_idle = _IDLE_INTENT
        hip_h = (
            float(cmd.hip_height_m)
            if cmd.hip_height_m is not None
            else float(hip_idle)
        )
        return (float(yaw_idle), float(vx_idle), float(vz_idle), hip_h)
    result = _resolve_velocity(cmd.intent, cmd.magnitude)
    if result == _IDLE_INTENT and cmd.intent != "idle":
        log.debug("intent %s,%s has no velocity mapping; idling",
                  cmd.intent, cmd.magnitude)
        return result
    return _apply_runtime_scales(cmd.intent, result)


def _apply_continuous_runtime_scales(
    velocity: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Per-axis runtime scaling for the continuous-locomotion path.

    Mirrors ``_apply_runtime_scales`` but routes on the sign of each
    velocity axis (forward vs backward, left turn vs right turn, etc.)
    instead of the (now-absent) intent name. Same scalars, same defaults
    -- a single ``--kplanner-forward-scale 0.6`` flag caps continuous
    and bucketed modes identically.
    """
    yaw, vx, vz, hip_h = velocity
    if yaw > 0:
        yaw *= _RUNTIME_TURN_LEFT_SCALE
    elif yaw < 0:
        yaw *= _RUNTIME_TURN_RIGHT_SCALE
    if vz > 0:
        vz *= _RUNTIME_FORWARD_SCALE
        # Forward-velocity floor: if the operator has committed any
        # non-zero forward stick deflection (vz > 0 pre-scale), lift
        # the post-scale forward command up to the in-distribution
        # band so the SONIC root model commits to a stride instead of
        # hip-wiggling below the training minimum. See
        # ``_DEFAULT_CONTINUOUS_FORWARD_MIN_MPS``'s docstring for the
        # full rationale; default 0.0 = no-op so legacy invariants and
        # unit tests still hold when the operator does not opt in.
        if _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS > 0.0:
            vz = max(vz, _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS)
    elif vz < 0:
        vz *= _RUNTIME_BACKWARD_SCALE
    vx *= _RUNTIME_LATERAL_SCALE
    return (yaw, vx, vz, hip_h)


def _build_intent_velocity_map() -> dict[
    tuple[str, str], tuple[float, float, float, float]
]:
    """Precompute every (intent, magnitude) the dispatcher recognises.

    Used to populate ``INTENT_VELOCITY_MAP`` for docs / debug dumps.
    Unknown magnitudes still resolve correctly at runtime via
    ``intent_to_velocity`` (scale tables fall back to 1.0); this derived
    map is informative, not exhaustive.
    """
    out: dict[tuple[str, str], tuple[float, float, float, float]] = {
        ("idle", "default"): _resolve_velocity("idle", "default"),
        ("idle", "stand"):   _resolve_velocity("idle", "stand"),
    }
    for mag in _WALK_VELOCITY_BY_MAGNITUDE:
        out[("walk", mag)] = _resolve_velocity("walk", mag)
    for intent in _BASE_VELOCITY:
        if intent == "idle":
            continue
        scales = _TURN_SCALE if intent in _ROTATIONAL_INTENTS else _TRANSLATIONAL_SCALE
        for mag in scales:
            out[(intent, mag)] = _resolve_velocity(intent, mag)
    return out


INTENT_VELOCITY_MAP: dict[
    tuple[str, str], tuple[float, float, float, float]
] = _build_intent_velocity_map()


# ---------------------------------------------------------------------------
# Process hygiene (forked verbatim from x2_heuristic_planner)
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


class PidFile:
    """Context manager owning ``/tmp/...pid`` for the lifetime of the daemon."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> "PidFile":
        if self.path.exists():
            try:
                old = int(self.path.read_text().strip())
                if _pid_alive(old):
                    raise RuntimeError(
                        f"PID file {self.path} exists and PID {old} is alive — "
                        f"another planner is running. Run "
                        f"`kill {old}` or use the stack cleanup helper."
                    )
                log.warning(
                    "stale PID file %s for dead PID %d, cleaning up",
                    self.path, old,
                )
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Pose publisher (same wire format as heuristic)
# ---------------------------------------------------------------------------


class PosePublisher:
    """Publishes pose payloads on a configurable ZMQ topic."""

    def __init__(
        self,
        host: str,
        port: int,
        topic: str = "body_pose",
        version: int = 4,
        hand_dof: int = 10,
        motion_token_dim: int = 64,
    ) -> None:
        import zmq
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._version = version
        self._hand_dof = hand_dof
        self._motion_token_dim = motion_token_dim
        time.sleep(0.1)

    def publish(
        self,
        frame: StreamFrame,
        future_frames: list[StreamFrame] | None = None,
        future_dt_s: float = 0.1,
    ) -> None:
        payload = build_pose_payload(
            frame,
            motion_token_dim=self._motion_token_dim,
            hand_dof=self._hand_dof,
            future_frames=future_frames,
            future_dt_s=future_dt_s,
        )
        msg = pack_pose_message(payload, topic=self._topic, version=self._version)
        self._sock.send(msg)

    def close(self) -> None:
        self._sock.close(linger=0)


# ---------------------------------------------------------------------------
# qpos[T, 38] -> StreamFrame helpers
# ---------------------------------------------------------------------------


def _qpos_to_stream_frame(
    qpos: np.ndarray,
    frame_index: int,
    bin_name: str,
    state: PlannerState = PlannerState.IDLE_LOOP,
) -> StreamFrame:
    """Convert a single 38-D qpos vector into a wire-compatible StreamFrame.

    qpos layout (MuJoCo wxyz convention, set by NeuralPlannerCore):
        [0:3]   root xyz (world frame)
        [3:7]   root quaternion wxyz
        [7:38]  31 joint positions in MuJoCo joint order

    StreamFrame.root_quat_xyzw is xyzw, so we permute wxyz -> xyzw here.
    """
    if qpos.shape[-1] != 38:
        raise ValueError(f"qpos must be 38-D, got {qpos.shape}")
    root_xyz = qpos[:3].astype(np.float32)
    root_wxyz = qpos[3:7].astype(np.float32)
    root_xyzw = np.array(
        [root_wxyz[1], root_wxyz[2], root_wxyz[3], root_wxyz[0]],
        dtype=np.float32,
    )
    # Use MuJoCo's xy world position for the diagnostic field (it's advisory;
    # the wire-critical fields are joint_pos_mj + root_quat_xyzw).
    return StreamFrame(
        joint_pos_mj=qpos[7:].astype(np.float32),
        root_quat_xyzw=root_xyzw,
        root_xy_world=root_xyz[:2].astype(np.float64),
        yaw_world_deg=0.0,  # advisory only; not used by deploy
        state=state,
        bin_name=bin_name,
        frame_index=frame_index,
        seam_blend=False,
        # Carry the actual MuJoCo qpos[2] through to the wire so the
        # kinematic viewer + PKL recorder can reconstruct world-frame
        # pelvis height instead of pelvis-pinning at DEFAULT_PELVIS_Z_M.
        root_z_world=float(root_xyz[2]),
    )


# ---------------------------------------------------------------------------
# pad-vs-VR command ownership (see pc2_kplanner_onnx for rationale)
_VR_OWNER_TIMEOUT_S = 2.0
_cmd_owner = {"src": None, "vr_ts": 0.0}

# Command sources (forked verbatim from x2_heuristic_planner with light edits)
# ---------------------------------------------------------------------------


def _scripted_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    yaml_path: Path,
    stop_event: threading.Event,
) -> None:
    cmds = commands_from_yaml(yaml_path)
    log.info("scripted source: queueing %d commands from %s", len(cmds), yaml_path)
    for cmd in cmds:
        if stop_event.is_set():
            return
        cmd_queue.put(cmd)
    log.info("scripted source: done")


def _zmq_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
    bind: bool = False,
) -> None:
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    if bind:
        # Dual-source mode (pad bridge + quest3 manager): the SUB owns the
        # bind so any number of command PUBs can connect.
        sock.bind(f"tcp://*:{port}")
        log.info("zmq command source: SUB bind %s on tcp://*:%d", topic, port)
    else:
        sock.connect(f"tcp://{host}:{port}")
        log.info("zmq command source: SUB %s on tcp://%s:%d", topic, host, port)
    try:
        while not stop_event.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                # Clock-driven VR-silence release (parity with
                # pc2_kplanner_onnx): a dead manager must not hold
                # ownership until pad traffic happens to arrive.
                if (_cmd_owner["src"] == "vr"
                        and time.monotonic() - _cmd_owner["vr_ts"]
                        > _VR_OWNER_TIMEOUT_S):
                    log.warning(
                        "cmd owner: VR silent %.1fs > %.1fs — releasing to "
                        "idle; pad may re-acquire",
                        time.monotonic() - _cmd_owner["vr_ts"],
                        _VR_OWNER_TIMEOUT_S)
                    _cmd_owner["src"] = None
                    cmd_queue.put(LocomotionCommand(
                        intent="idle", magnitude="default"))
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
                intent = str(payload["intent"])
                if intent == "shutdown":
                    log.info("zmq command source: shutdown received")
                    stop_event.set()
                    continue
                # ---- source ownership (pad vs VR mutual exclusion) ----
                # Mirrors pc2_kplanner_onnx._zmq_command_thread so sim
                # dual-source behavior matches the robot: one owner at a
                # time, VR supersedes pad, release on VR idle or 2 s VR
                # silence (manager keepalives every 0.5 s while engaged).
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
                        elif _cmd_owner["src"] == "pad":
                            continue  # stray release while pad drives
                    else:
                        if _cmd_owner["src"] != "vr":
                            log.warning("cmd owner: VR ENGAGED — pad input ignored")
                            _cmd_owner["src"] = "vr"
                else:
                    if _cmd_owner["src"] == "vr":
                        if now_own - _cmd_owner["vr_ts"] > _VR_OWNER_TIMEOUT_S:
                            log.warning(
                                "cmd owner: VR silent %.1fs > %.1fs — releasing "
                                "to idle; pad re-acquires",
                                now_own - _cmd_owner["vr_ts"], _VR_OWNER_TIMEOUT_S)
                            _cmd_owner["src"] = None
                            cmd_queue.put(LocomotionCommand(
                                intent="idle", magnitude="default"))
                            continue
                        continue
                    if _cmd_owner["src"] is None:
                        _cmd_owner["src"] = "pad"
                magnitude = str(payload.get("magnitude", "default"))
                # Continuous locomotion (analog Quest3 sticks) carries
                # three deadzone-rescaled deflections in [-1, 1]. Missing
                # fields default to 0.0 so legacy bucketed payloads still
                # work for the same wire format.
                stick_fwd  = float(payload.get("stick_fwd",  0.0))
                stick_side = float(payload.get("stick_side", 0.0))
                stick_yaw  = float(payload.get("stick_yaw",  0.0))
                # One-shot X/Y speed-setpoint nudge from the VR manager.
                # Applied immediately to the module-level setpoint (the
                # continuous resolver reads it on every dispatch).
                sd = payload.get("speed_delta")
                if sd:
                    _adjust_speed_setpoint(float(sd))
                # ``hold_torso`` (continuous waist target) is consumed
                # by the kplanner's waist-overlay path (v7.4): the
                # tracker walks ``current_*_deg`` toward these targets
                # at HOLD_SLEW_DPS each tick and the publish loop adds
                # the deltas to the waist DOFs of the kplanner's
                # output frame. ``hip_height_m`` is an optional
                # absolute hip-height override (metres) consumed by
                # ``intent_to_velocity``: when present and the intent
                # is ``hold_torso`` it substitutes channel-3 of the
                # 4-D velocity intent so the operator's L-stick Y
                # squat / stand drives the model's continuous height
                # target. Missing field -> None -> kplanner default.
                waist_pitch_deg = float(payload.get("waist_pitch_deg", 0.0))
                waist_roll_deg  = float(payload.get("waist_roll_deg",  0.0))
                waist_yaw_deg   = float(payload.get("waist_yaw_deg",   0.0))
                hip_height_raw  = payload.get("hip_height_m", None)
                hip_height_m: Optional[float] = (
                    float(hip_height_raw)
                    if hip_height_raw is not None
                    else None
                )
                # Optional raw 4-D velocity passthrough used by
                # ``x2_pkl_command_source`` for replaying recorded motion
                # clips through the planner -> deploy chain without the
                # bucketed table / continuous shaping / runtime scales
                # distorting the recorded velocity. When present this is
                # ``[yaw_rate, vel_x, vel_z, hip_h]``; the dispatcher
                # short-circuits to it (see ``intent_to_velocity``).
                # Missing field -> ``None`` -> dispatcher follows its
                # normal bucketed / continuous path.
                target_velocity = payload.get("target_velocity")
                direct_velocity: Optional[tuple[float, float, float, float]] = None
                if target_velocity is not None:
                    if (
                        not isinstance(target_velocity, (list, tuple))
                        or len(target_velocity) != 4
                    ):
                        log.warning(
                            "zmq command source: target_velocity must be a "
                            "4-element list [yaw_rate, vel_x, vel_z, hip_h]; "
                            "got %r (ignoring)",
                            target_velocity,
                        )
                    else:
                        direct_velocity = (
                            float(target_velocity[0]),
                            float(target_velocity[1]),
                            float(target_velocity[2]),
                            float(target_velocity[3]),
                        )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("zmq command source: bad payload %r: %s", parts[1], exc)
                continue
            cmd_queue.put(
                LocomotionCommand(
                    intent=intent,
                    magnitude=magnitude,
                    source="zmq",
                    waist_pitch_deg=waist_pitch_deg,
                    waist_roll_deg=waist_roll_deg,
                    waist_yaw_deg=waist_yaw_deg,
                    stick_fwd=stick_fwd,
                    stick_side=stick_side,
                    stick_yaw=stick_yaw,
                    direct_velocity=direct_velocity,
                    hip_height_m=hip_height_m,
                )
            )
    finally:
        sock.close(linger=0)


# ---------------------------------------------------------------------------
# Closed-loop pose feedback. The kplanner's chain of self-conditioned
# replans accumulates yaw drift (deploy-integration diagnostics of the
# kplanner evaluation harness). To break the loop we
# subscribe to the sim bridge's ``robot_pose`` topic and, just before
# each replan, overwrite the root rows the planner is about to read as
# context with the robot's *actually observed* pelvis pose.
#
# Joint slots [7:] stay model-predicted on purpose: the policy's
# joint-level tracking error would inject high-frequency noise into the
# context if those were reseeded too. We only reseed the root because:
#   1. It is the only channel where compounding drift was measured.
#   2. Pelvis qpos is what robot_pose actually exposes.
#   3. Joint reseed would require a separate observability path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseObservation:
    """One ``robot_pose`` sample.

    ``pelvis_qpos_wxyz`` matches the bridge's wire format:
    ``[x, y, z, qw, qx, qy, qz]`` (length 7), the same wxyz quat
    convention NeuralPlannerCore stores in ``frames['mujoco_qpos']``.

    ``t_mono`` is ``time.monotonic()`` at receive time. Used for staleness
    checks so a frozen bridge doesn't keep feeding ancient observations
    into the planner forever.
    """

    t_mono: float
    pelvis_qpos_wxyz: np.ndarray  # shape (7,)


def _yaw_only_wxyz_from_pelvis(pelvis_qpos_wxyz: np.ndarray) -> np.ndarray:
    """Extract world-frame yaw from a pelvis ``(x,y,z, qw,qx,qy,qz)`` row
    and return ``R_z(yaw)`` packed as ``(qw, qx, qy, qz)``.

    Used by the publish loop's IDLE_LOOP branch to re-anchor the
    persisted reference yaw (``current_root_wxyz``) on every tick from
    the latest ``robot_pose`` observation. Pitch + roll are dropped on
    purpose so a transient leg lean (e.g. fall-recovery) doesn't bleed
    into the published reference and confuse SONIC's upright-reference
    training distribution.

    Yaw extraction goes through :func:`yaw_of_quat_xyzw` so the
    convention (lowercase ``"zyx"`` extrinsic, range ``(-pi, pi]``)
    matches every other yaw-touching site in the stack -- the gesture
    session yaw rebase, the recorder's snap quat, and the kplanner's
    own state-machine logging. A custom closed-form would silently
    drift from that convention under non-trivial pitch/roll.
    """
    qpos = np.asarray(pelvis_qpos_wxyz, dtype=np.float64).reshape(-1)
    if qpos.shape[0] < 7:
        raise ValueError(
            f"pelvis_qpos_wxyz must be >= 7 long (got {qpos.shape[0]})"
        )
    quat_xyzw = np.array(
        [qpos[4], qpos[5], qpos[6], qpos[3]],
        dtype=np.float64,
    )
    yaw = yaw_of_quat_xyzw(quat_xyzw)
    half = 0.5 * yaw
    return np.array(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        dtype=np.float32,
    )


def _pose_feedback_thread(
    pose_deque: "collections.deque[PoseObservation]",
    pose_lock: threading.Lock,
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
) -> None:
    """SUB on ``robot_pose:port`` and append observations to ``pose_deque``.

    Drains as fast as the bridge publishes (~50 Hz). Deque has a maxlen
    upstream so we don't grow unbounded if the worker stops draining.
    """
    import zmq
    from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    log.info("pose feedback source: SUB %r on tcp://%s:%d", topic, host, port)
    warn_once = False
    msg_count = 0
    try:
        while not stop_event.is_set():
            try:
                raw = sock.recv()
            except zmq.error.Again:
                continue
            try:
                payload = unpack_robot_pose(raw)
                qpos = payload.get("pelvis_qpos_wxyz")
                if not isinstance(qpos, list) or len(qpos) != 7:
                    continue
                obs = PoseObservation(
                    t_mono=time.monotonic(),
                    pelvis_qpos_wxyz=np.asarray(qpos, dtype=np.float32),
                )
                with pose_lock:
                    pose_deque.append(obs)
                msg_count += 1
                if msg_count == 1:
                    log.info(
                        "pose feedback: first observation received "
                        "(xyz=%.3f,%.3f,%.3f qw=%.3f)",
                        float(qpos[0]), float(qpos[1]), float(qpos[2]),
                        float(qpos[3]),
                    )
            except Exception as exc:
                if not warn_once:
                    log.warning("pose feedback: decode error %s (suppressed)", exc)
                    warn_once = True
    finally:
        sock.close(linger=0)


_RESEED_SCOPE_FULL_ROOT = "full_root"
_RESEED_SCOPE_QUAT_ONLY = "quat_only"
# ``none`` -> reseed disabled entirely (PLAYING side). Use this on real
# robot where the pose source is the IMU-only x2_debug bridge (no
# position measurement): ``full_root`` would overwrite the planner's
# xy/z history with zeros every replan tick (catastrophic instability),
# and ``quat_only`` would anchor the model's yaw integration to the
# lagging measured yaw, causing commanded turns to under-rotate (the
# model can never get more than one replan-tick ahead of the robot).
# With ``none`` the planner integrates yaw open-loop during PLAYING --
# snap-back protection still comes from the IDLE_LOOP yaw refresh and
# the new IDLE -> PLAYING transition seed (both yaw-only, both writing
# to ``current_root_wxyz`` only, never to the model's neural buffer).
_RESEED_SCOPE_NONE = "none"
_VALID_RESEED_SCOPES = (
    _RESEED_SCOPE_FULL_ROOT,
    _RESEED_SCOPE_QUAT_ONLY,
    _RESEED_SCOPE_NONE,
)


def _reseed_root_from_observations(
    planner_core,
    pose_deque: "collections.deque[PoseObservation]",
    pose_lock: threading.Lock,
    max_age_s: float,
    scope: str = _RESEED_SCOPE_FULL_ROOT,
) -> Optional[str]:
    """Overwrite the root rows the planner is about to read as context.

    Mirrors the index math in
    ``NeuralPlannerCore.get_context_mujoco_qpos`` so the context indices
    computed here match the indices that ``replan_with_velocity`` will
    read on its next call. Writes to ``frames['mujoco_qpos'][0, ctx_i,
    0:7]`` only -- joints (slots 7:) stay model-predicted.

    Caller MUST hold ``replan_lock`` because we mutate the in-place
    tensor that the next ``predict()`` reads.

    **Sampling strategy.** The model's 4-frame context window is
    *trained* at 30 fps spacing (1/30 = 33 ms between frames). The
    bridge's ``robot_pose`` stream runs at 50 Hz (20 ms spacing). Sampling
    "the last 4 observations" would feed the model a 60 ms window
    that it would interpret as a 100 ms window (with ~1.66x inflated
    implied velocity) -- and worse, near the buffer tail the planner's
    context indices collapse to ``[idx, last, last, last]`` so the
    second-through-fourth writes simply overwrite each other, giving
    the model a "robot teleported then froze" context that nothing in
    training resembles.

    Instead we walk backwards from "now" in 1/fps increments and pick
    the deque entry closest to each target timestamp. For duplicate
    context indices (common at the buffer tail) we let the per-slot
    selection naturally collapse to the same observation -- the model
    sees coherent "the robot is here, walking at the trained fps"
    context.

    Returns ``None`` on success, or a short skip-reason string.
    """
    # Short-circuit ``none`` BEFORE any planner_core / pose_deque access
    # so the call is cheap (a single string compare per replan tick) and
    # the disabled state is observable in reseed_stats without taking
    # the pose_lock. ``"disabled"`` is the canonical skip-reason; the
    # caller maps non-{insufficient,stale,buffer_uninit} reasons into
    # ``skipped_other`` for log accounting.
    if scope == _RESEED_SCOPE_NONE:
        return "disabled"

    import torch

    buf = planner_core.frames.get("mujoco_qpos")
    if buf is None:
        return "buffer_uninitialized"

    n_ft = int(planner_core.NUM_FRAMES_PER_TOKEN)

    with pose_lock:
        obs_list = list(pose_deque)

    if len(obs_list) < n_ft:
        return f"insufficient_obs ({len(obs_list)}/{n_ft})"

    newest = obs_list[-1]
    newest_age = time.monotonic() - newest.t_mono
    if newest_age > max_age_s:
        return f"stale_obs (age={newest_age:.3f}s > {max_age_s:.3f}s)"

    model_fps = float(getattr(planner_core, "fps", 30.0) or 30.0)
    target_spacing_s = 1.0 / model_fps

    target_times = [
        newest.t_mono - (n_ft - 1 - k) * target_spacing_s
        for k in range(n_ft)
    ]
    selected: list[PoseObservation] = []
    for tt in target_times:
        best = min(obs_list, key=lambda o, _tt=tt: abs(o.t_mono - _tt))
        selected.append(best)

    idx = int(planner_core._current_frame_idx)
    pred_off = int(planner_core.PRED_OFFSETS)
    last_idx = int(buf.shape[1]) - 1
    context_indices = [
        max(0, min(idx - n_ft + i + pred_off, last_idx))
        for i in range(n_ft)
    ]

    if scope not in _VALID_RESEED_SCOPES:
        raise ValueError(
            f"unknown reseed scope {scope!r}; want one of {_VALID_RESEED_SCOPES}"
        )

    device = buf.device
    dtype = buf.dtype
    # ``full_root`` rewrites xyz + quat; the open-loop diagnostic that
    # motivated this work showed the model's INTERNAL xy prediction
    # actually overshoots in a way that helps the deploy track forward
    # (the policy chases a slightly-ahead reference). Pinning xy to
    # the deploy's observed position removes that lure and forward
    # tracking regresses. ``quat_only`` preserves the helpful xy
    # overshoot while still anchoring the planner's heading to
    # observed reality.
    for ctx_i, obs in zip(context_indices, selected):
        obs_t = torch.from_numpy(np.asarray(obs.pelvis_qpos_wxyz, dtype=np.float64)).to(
            device=device, dtype=dtype,
        )
        if scope == _RESEED_SCOPE_FULL_ROOT:
            buf[0, ctx_i, 0:7] = obs_t
        else:
            buf[0, ctx_i, 3:7] = obs_t[3:7]

    return None


_KPLANNER_KEYBOARD_HELP = """
X2 kplanner — keyboard commands (mirrors heuristic):
   w        walk forward
   W        walk fast
   b        back step
   a / d    side left / right
   q / e    turn left / right 45 deg
   Q / E    turn left / right 90 deg
   1 / 3    turn left / right 15 deg
   2 / 4    turn left / right 30 deg
   space    idle
   x        quit
"""

_KPLANNER_KEYBOARD_MAP: dict[str, tuple[str, str]] = {
    "w": ("walk", "forward"),
    "W": ("walk", "fast"),
    "b": ("back_step", "half_ft"),
    "a": ("side_left", "default"),
    "d": ("side_right", "default"),
    "1": ("turn_left", "deg_15"),
    "2": ("turn_left", "deg_30"),
    "q": ("turn_left", "deg_45"),
    "Q": ("turn_left", "deg_90"),
    "3": ("turn_right", "deg_15"),
    "4": ("turn_right", "deg_30"),
    "e": ("turn_right", "deg_45"),
    "E": ("turn_right", "deg_90"),
    " ": ("idle", "default"),
}


def _keyboard_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    stop_event: threading.Event,
) -> None:
    if not sys.stdin.isatty():
        log.error("keyboard source: stdin is not a TTY; refusing to start")
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print(_KPLANNER_KEYBOARD_HELP, flush=True)
        while not stop_event.is_set():
            ch = sys.stdin.read(1)
            if ch in ("x", "\x03"):
                stop_event.set()
                return
            if ch in _KPLANNER_KEYBOARD_MAP:
                intent, mag = _KPLANNER_KEYBOARD_MAP[ch]
                cmd_queue.put(
                    LocomotionCommand(intent=intent, magnitude=mag, source="kbd")
                )
                print(f"  -> {intent} {mag}", flush=True)
            else:
                print(_KPLANNER_KEYBOARD_HELP, flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# T-pose / warmup anchor helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training-default stand pose. Source of truth for these numbers is
# ``policy_parameters.hpp::default_angles`` in the C++ deploy
# (mirrored in x2_capture_pose.py / x2_pelvis_z_from_capture.py /
# x2_action_monitor.py / x2_scan_mc_motors.py). When the deploy's
# pose-ref starvation watchdog trips it PD-controls toward this exact
# 31-DOF stand pose; using it as the kplanner's warmup qpos means
# SAFE_IDLE produces ZERO joint-target delta from the RSI'd spawn
# pose -- the robot sits perfectly still through the multi-second
# cold-start window where the planner is loading models and the
# recorder isn't yet merging pose:5556 frames.
#
# Floor-contact hip_z for this pose is 0.636 m (X2 with knees bent
# 0.669 rad / 38 deg, lowest foot geom just touches z=0). Computed via
# mujoco FK on the canonical sphere-feet x2_ultra MJCF; matches the
# Z that ``x2_pelvis_z_from_capture.py`` reports for DEFAULT_ANGLES.
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


# Canonical recorded stand pose (from real-robot IMU + state telemetry).
# Frame 0 is a stable bent-knee stand at hip_z=0.655m with the leg / arm
# joint angles a real X2 settles into. Used as the SECONDARY default
# (after training_default_angles) when the operator explicitly wants a
# real-recorded stand pose and is OK with SAFE_IDLE pulling the joints
# the ~0.15--0.57 rad toward training_default_angles during cold-start.
_BALANCED_STAND_PKL: Path = (
    Path(__file__).resolve().parent.parent
    / "data" / "motions" / "x2_recorded" / "balanced_stand.pkl"
)


def _build_training_default_qpos() -> np.ndarray:
    """qpos[38] (wxyz) at the deploy's training-default stand pose.

    This is the deploy's SAFE_IDLE PD target so RSI'ing the bridge to
    this pose means the watchdog watchdog-trip during cold-start
    produces ZERO joint-target delta -- the robot doesn't move even
    though no upstream pose stream has arrived yet.
    """
    qpos = np.zeros(38, dtype=np.float32)
    qpos[2] = _TRAINING_DEFAULT_HIP_Z
    qpos[3] = 1.0  # quaternion w = 1 (identity rotation, wxyz)
    qpos[7:38] = _TRAINING_DEFAULT_ANGLES
    return qpos


def _qpos_from_deploy_pkl_frame(obj_inner: dict) -> Optional[np.ndarray]:
    """Convert one frame of a deploy-PKL schema dict to qpos[38] (wxyz).

    The deploy-PKL schema (used by bake_planner_rsi_anchor.py,
    bake_kplanner_rsi_anchor.py, and the on-disk balanced_stand.pkl)
    stores the root rotation in **xyzw** convention; MuJoCo qpos uses
    **wxyz**. This helper reorders rather than re-normalizes so a
    "stationary" PKL stays stationary.
    """
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
    f0_rot_wxyz = np.array(
        [f0_rot_xyzw[3], f0_rot_xyzw[0], f0_rot_xyzw[1], f0_rot_xyzw[2]],
        dtype=np.float32,
    )
    qpos = np.zeros(38, dtype=np.float32)
    qpos[0:3] = f0_trans
    qpos[3:7] = f0_rot_wxyz
    qpos[7:38] = f0_dof
    return qpos


def _build_hardcoded_warmup_qpos() -> np.ndarray:
    """Hand-crafted T-pose-ish stand qpos[38] in MuJoCo wxyz convention.

    Fallback for when ``balanced_stand.pkl`` isn't on disk. The joint
    angles are ALL ZERO and the hip is at 0.95 m -- the feet are
    therefore NOT on the floor and the deploy's tracking policy will
    fight gravity from tick 0. Only use this if balanced_stand.pkl is
    absent; the wrapper bake step will warn loudly when this fires.

    qpos layout:
      - [0:3]   root xyz (origin + hip height)
      - [3:7]   wxyz quaternion (identity)
      - [7:38]  31 joint angles (rad), all zero
    """
    qpos = np.zeros(38, dtype=np.float32)
    qpos[2] = _HIP_HEIGHT_M  # hip height above floor (does NOT settle)
    qpos[3] = 1.0            # quaternion w = 1 (identity rotation)
    return qpos


def _build_default_warmup_qpos() -> np.ndarray:
    """Resolved-default warmup qpos[38] in MuJoCo wxyz convention.

    Defaults to the deploy's **training_default_angles** stand pose.
    This is the single safest choice for the parity RSI anchor because
    it's bit-identical to the pose the deploy's pose-ref starvation
    watchdog falls back to (SAFE_IDLE PD target). When the wrapper
    spawns the deploy ~5 s before the planner is ready and ~15 s
    before the recorder is merging pose:5556 frames, the watchdog
    will trip and SAFE_IDLE will hold the robot at training_default_
    angles. RSI'ing the bridge to the same pose means SAFE_IDLE
    produces ZERO joint-target delta and the robot doesn't drift
    through the entire cold-start window.

    Both the daemon's first publish tick AND the parity RSI anchor PKL
    (baked via ``bake_kplanner_rsi_anchor.py``) derive from this, so
    spawn-pose / wire-pose / SAFE_IDLE-target stay bit-identical.

    Operators who want a recorded real-robot stand instead can pass
    ``--warmup-qpos-path .../balanced_stand.pkl``; the bake step
    forwards it.
    """
    qpos = _build_training_default_qpos()
    log.info(
        "warmup anchor: using training_default_angles stand pose "
        "(hip_z=%.3fm) -- matches deploy SAFE_IDLE PD target",
        float(qpos[2]),
    )
    return qpos


def _load_balanced_stand_default() -> Optional[np.ndarray]:
    """Best-effort load of frame 0 from balanced_stand.pkl. Returns None
    if the schema doesn't match (caller falls back)."""
    try:
        import joblib  # local import: only needed for the recorded PKL
    except ImportError:  # pragma: no cover -- joblib ships with sklearn
        log.warning("joblib missing; cannot load balanced_stand.pkl")
        return None
    try:
        obj = joblib.load(_BALANCED_STAND_PKL)
    except Exception as exc:  # noqa: BLE001
        log.warning("balanced_stand.pkl unreadable: %r", exc)
        return None
    # Schema: {'balanced_stand': {'root_trans_offset', 'root_rot', 'dof', ...}}
    if isinstance(obj, dict) and len(obj) >= 1:
        inner = next(iter(obj.values()))
        if isinstance(inner, dict):
            return _qpos_from_deploy_pkl_frame(inner)
    return None


def _load_warmup_qpos(path: Optional[Path]) -> np.ndarray:
    """Load a [38]-D qpos vector from a pickled stand-pose, or fall back to
    the resolved default (balanced_stand.pkl -> hardcoded T-pose)."""
    if path is None:
        return _build_default_warmup_qpos()
    if not path.is_file():
        log.warning(
            "warmup-qpos-path %s not found; falling back to resolved default",
            path,
        )
        return _build_default_warmup_qpos()
    try:
        import joblib  # try joblib first since recorded PKLs are zlib-compressed
        obj = joblib.load(path)
    except Exception:  # noqa: BLE001 -- fall through to raw pickle
        with path.open("rb") as f:
            obj = pickle.load(f)
    # Accept several schemas:
    #   1. Raw [38] array or [T, 38] (take frame 0)
    #   2. {'mujoco_qpos': ...} or {'qpos': ...}
    #   3. Deploy-PKL nested: {'<name>': {'root_trans_offset', 'root_rot', 'dof', ...}}
    #      (this matches both x2_recorded/*.pkl and the anchor PKL schemas)
    if isinstance(obj, dict):
        # Case 2: explicit qpos field
        arr = obj.get("mujoco_qpos", obj.get("qpos", None))
        if arr is None:
            # Case 3: deploy-PKL nested
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
                f"warmup PKL {path} has no recognisable qpos / "
                f"deploy-PKL schema (keys={list(obj.keys())})"
            )
    else:
        arr = obj
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[-1] != 38:
        raise ValueError(
            f"warmup PKL {path} qpos shape {arr.shape} != [38]"
        )
    log.info("warmup anchor: loaded from %s (shape=[38])", path)
    return arr


# ---------------------------------------------------------------------------
# Worker thread: replan when ring buffer dips below threshold
# ---------------------------------------------------------------------------


# Default time constant for the cold-start velocity ramp (s). See
# ``_ColdStartVelocityRamp`` for the full rationale; 0.0 disables the
# ramp (the worker passes intent through verbatim, matching the
# pre-2026-05-30 behavior). Operators tune this via the kplanner CLI
# (``--cold-start-ramp-tau-s``) or the Quest 3 / PKL wrappers'
# ``KPLANNER_COLD_START_RAMP_TAU_S`` env var.
# 2026-07-16: default OFF (0.0). The ramp was a band-aid for models trained
# only on steady-state loops; the g1teleop fine-tuned roots train on real
# stand->walk transitions (idle lead-ins kept in the corpus), so deploy-side
# intent smoothing now just masks the model. Opt back in for legacy models
# via KPLANNER_COLD_START_RAMP_TAU_S / --cold-start-ramp-tau-s.
_DEFAULT_COLD_START_RAMP_TAU_S: float = 0.0


class _ColdStartVelocityRamp:
    """Per-channel EWMA velocity ramp to smooth idle -> playing transitions.

    The kplanner's neural root model trains exclusively on steady-state
    walking loops (the forward-walk template clip); it never sees
    a static-stand -> walking transition. When the operator's L-stick
    first crosses the deadzone, the state machine fires IDLE_LOOP ->
    PLAYING which:

    1. calls ``planner_core.reset(warm)``, wiping the 4-frame context
       ring buffer and refilling it with 4 identical static-stand
       poses;
    2. forwards the raw operator target (e.g. ``vel_z = +0.5 m/s``)
       to ``replan_with_velocity``.

    ``NeuralPlannerCore`` then builds an implied target position at
    ``current_xy + vel * TARGET_HORIZON_S`` (= ``current_xy + 0.5 *
    2.13 = current_xy + 1.07 m`` in this example), meaning the model is
    asked to project from "4 frames of static stand" to "1 m ahead in
    2.13 s" in a single replan. Because the model has no training
    coverage for that regime, the highest-likelihood prediction
    channel is **pelvis x-translation rather than leg swing**; the
    deploy policy faithfully tracks that and the operator sees
    "torso bends forward, no step". After ~2-4 replans the buffer is
    filled with the model's own (now non-static) predictions and the
    gait stabilises.

    The ramp keeps (yaw_rate, vel_x, vel_z) close to zero for the
    first few replans after PLAYING entry, so the implied target stays
    close to the robot's actual position and the model emits gentler
    forward motion the policy can step into. ``hip_h`` (channel 3) is
    NOT ramped -- it's a posture target, not a velocity, and the
    model needs the correct walking pelvis height from frame 1.

    Time constant defaults to 0.2 s. At a 200 ms replan period
    (``--replan-threshold-frames 2`` + 30 FPS output) that yields
    alpha ~= 0.5, reaching ~95% of the operator's target in ~3
    replans (600 ms). Operators wanting snappier acceleration can
    drop this; ``tau = 0`` disables the ramp entirely (pre-fix
    behaviour).

    Deceleration is handled by the IDLE gate, not by this ramp:
    releasing the stick makes ``intent_to_velocity`` resolve to
    ``_IDLE_INTENT``, the state machine fires PLAYING -> IDLE_LOOP,
    and the publisher freezes the anchor pose immediately. The
    ramper's idle-detection (``last_was_idle``) just makes sure the
    NEXT idle->playing transition starts the ramp from zero again.
    """

    def __init__(self, tau_s: float = _DEFAULT_COLD_START_RAMP_TAU_S) -> None:
        self.tau_s = float(tau_s)
        # (yaw_rate, vel_x, vel_z) -- hip_h passed through verbatim.
        self._smoothed = np.zeros(3, dtype=np.float64)
        self._last_was_idle = True

    @property
    def enabled(self) -> bool:
        return self.tau_s > 0.0

    def step(
        self,
        target: tuple[float, float, float, float],
        dt_s: float,
    ) -> tuple[float, float, float, float]:
        """Advance the ramp by ``dt_s`` and return the smoothed target.

        When the previous tick was idle (== the operator just started
        pushing the stick or we're in the very first PLAYING entry)
        the smoothed state is reset to zero so the new push gets the
        full ramp. ``hip_h`` is forwarded verbatim from ``target``.
        """
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
        return (
            float(self._smoothed[0]),
            float(self._smoothed[1]),
            float(self._smoothed[2]),
            float(hip),
        )

    def reset_idle(self) -> None:
        """Mark the next ``step()`` as a fresh idle->playing entry."""
        self._smoothed.fill(0.0)
        self._last_was_idle = True


# ---------------------------------------------------------------------------
# Reference-step smoother (publisher output, step-detection driven)
# ---------------------------------------------------------------------------


# Default ramp duration (s) used by ``_ReferenceStepSmoother`` when an
# inbound reference step is detected. 0.30 s sits at the natural period of
# the lower-body PD loop at deployed kp (hip/knee ~99 Nm/rad with leg
# inertia O(few kg.m^2)), so the joint actually has time to follow the
# rising reference instead of being thrown at the ramp's onset. Set 0.0 to
# fall back to passthrough (cheaper than the ``shape=off`` path). Operators
# tune via ``--ref-smoother-ms`` / ``KPLANNER_REF_SMOOTHER_MS``.
_DEFAULT_REF_SMOOTHER_MS: float = 300.0

# Default trigger threshold (rad) for the smoother. Per-tick reference
# delta on any blended-channel joint exceeding this value will arm a
# halfcos ramp. 0.05 rad ~= 3 deg cleanly separates the multi-degree
# step a stick push/release creates (frozen anchor <-> neural buffer pose)
# from the per-tick neural-buffer motion under steady walking (typ.
# < 0.01 rad/tick on the hip pitch leader). Operators tune via
# ``--ref-smoother-trigger-rad``; setting 0 makes every non-zero delta a
# candidate (debug only -- ramps will fire continuously during smooth
# walking and add steady-state lag).
_DEFAULT_REF_SMOOTHER_TRIGGER_RAD: float = 0.05

# Ramp shape options. ``halfcos`` is the operational default -- C^1 smooth
# at both endpoints, so neither the start nor the end of the ramp injects
# a velocity step into the PD law. ``linear`` is exposed for A/B
# comparison; it has a step in dq/dt at t=0 and t=T, which materially
# defeats the click-suppression goal but is useful for debugging the
# shape's contribution. ``off`` short-circuits the smoother to a pure
# passthrough -- byte-equivalent to the pre-2026-05-31 publisher output,
# usable as a single-flag revert in case the smoother regresses anything.
_REF_SMOOTHER_SHAPES: tuple[str, ...] = ("halfcos", "linear", "off")
# Default OFF as of the 30->50 Hz output-resampling + 8-frame cross-fade
# blend landing (see NeuralPlannerCore.get_next_frame_resampled). The
# ref-smoother was a reactive substitute for proper handoff blending
# (300 ms halfcos on >0.05 rad reference jumps); with the seam blend now
# handling replan discontinuities at the source it is opt-in only. Pass
# ``--ref-smoother-shape halfcos`` (or ``linear``) to re-enable.
_DEFAULT_REF_SMOOTHER_SHAPE: str = "off"

# Joint-mask presets. ``lower_body`` (default) ramps legs + waist only
# (MJ indices [0..14], confirmed against
# ``policy_parameters.hpp::x2_action_scale`` / ``default_angles``). Arms,
# wrists, and head pass through unchanged so manipulation tasks driven by
# the same body_pose stream (e.g. future bimanual reach) are not affected
# by any ramp -- the audible click is mechanical and lives in the leg
# drivetrain; arms run on smaller kp (~14 Nm/rad) and have no observed
# click symptom on the X2 hardware to date. ``legs_only`` skips the waist
# block ([12..14]) in case waist twist needs to remain maximally
# responsive. ``all`` blends all 31 DoFs (legacy debug A/B; the original
# plan formulation). Fingers / OmniHand commands are on a separate command
# path and never enter the kplanner body_pose stream, so they are
# unconditionally untouched by every preset.
_REF_SMOOTHER_JOINTS_PRESETS: dict[str, np.ndarray] = {
    "lower_body": np.arange(0, 15, dtype=np.int64),
    "legs_only":  np.arange(0, 12, dtype=np.int64),
    "all":        np.arange(0, 31, dtype=np.int64),
}
_DEFAULT_REF_SMOOTHER_JOINTS: str = "lower_body"


class _ReferenceStepSmoother:
    """One-shot halfcos ramp on detected per-tick reference steps.

    Why this exists
    ---------------

    The deploy publishes a 31-DoF position reference at 50 Hz. The PD law
    closing on each motor is
    ``tau = kp * (target_pos - q) + kd * (target_vel - dq) + effort_ff``,
    evaluated at 500 Hz. When the published ``target_pos`` jumps by
    several degrees in a single 20 ms tick (e.g. operator pushes the
    Quest 3 thumbstick and the kplanner switches its reference source
    from frozen ``default_angles`` to the live neural buffer; same in
    reverse on release), the lower-body kp (hip/knee ~99 Nm/rad,
    ankle ~21 Nm/rad after the 2026-05 stability bump) multiplies that
    step into a torque step on the wire. The motor current loop slams
    current to deliver it, and the real-hardware drivetrain absorbs the
    step as a percussive backlash / bearing snap -- the operator-reported
    "loud click" on every stick push and release. MuJoCo has no
    backlash / bearing-snap model, so the symptom is silent in sim.

    How it works (no FSM coupling)
    ------------------------------

    Every publish tick the caller hands the smoother the would-be
    ``target_q`` (31-vector) and the wall-clock tick time. The smoother:

    1. Computes ``delta_max = max(|target_q[i] - last_published_q[i]|)``
       over the channels in ``blend_indices`` only. Arms / head are
       excluded both from the step detection AND from any blending --
       they pass through byte-equivalent to the input.
    2. If no ramp is currently active AND ``delta_max > trigger_rad``,
       it arms a one-shot ramp: ``source_q = last_published_q.copy()``,
       ``ramp_start_t = t_now``, ``ramp_active = True``. The caller can
       attribute the trigger from the published log line (joint name +
       magnitude) but the smoother itself is intentionally agnostic to
       what produced the step. Operator stick push, stick release,
       primitive switch, future MC-handoff piped through this publisher
       -- all are handled uniformly because the step on the wire is the
       same signal.
    3. If a ramp is active, it computes
       ``alpha = blend_fn(min((t_now - ramp_start_t) / ramp_duration_s, 1.0))``
       where ``blend_fn`` is ``halfcos`` by default (``0.5 * (1 - cos(pi*x))``,
       ``C^1`` smooth at both endpoints). The output on blended channels is
       ``(1 - alpha) * source_q[i] + alpha * target_q[i]``; on excluded
       channels it's ``target_q[i]``. The target is sampled *live* every
       tick so the ramp ends exactly on whatever the publisher would
       have emitted anyway, eliminating a second click at the ramp's
       far end.
    4. When ``t_now - ramp_start_t >= ramp_duration_s``, the ramp clears
       (``ramp_active = False``). Subsequent ticks are passthrough until
       a new step is detected.

    Behaviour invariants tested in
    ``tests/test_x2_kplanner_reference_smoother.py``:

      * ``shape=off`` is byte-equivalent passthrough (no detection, no
        log lines, no state updates beyond ``last_published_q``).
      * ``joints=lower_body`` (default) NEVER modifies indices [15..30]
        even during an active ramp (manipulation-safety invariant).
      * An active ramp does not restart on subsequent large deltas (the
        smoother is one-shot per detected step; it must complete before
        re-arming).
      * Sub-trigger deltas pass through without arming a ramp (steady
        walking does not introduce ongoing lag).

    Hot-path cost is O(31) per tick (one np.abs + np.max on the blended
    slice, one np.where or pre-built mask multiply during ramps). The
    smoother is allocated once at startup and held by reference in the
    publish loop.
    """

    def __init__(
        self,
        *,
        ramp_duration_s: float = _DEFAULT_REF_SMOOTHER_MS / 1000.0,
        trigger_rad: float = _DEFAULT_REF_SMOOTHER_TRIGGER_RAD,
        shape: str = _DEFAULT_REF_SMOOTHER_SHAPE,
        blend_indices: Optional[np.ndarray] = None,
        num_dofs: int = 31,
        log_fn: Optional[callable] = None,  # type: ignore[valid-type]
    ) -> None:
        if shape not in _REF_SMOOTHER_SHAPES:
            raise ValueError(
                f"_ReferenceStepSmoother: shape={shape!r} not in "
                f"{_REF_SMOOTHER_SHAPES}"
            )
        self.ramp_duration_s = float(ramp_duration_s)
        self.trigger_rad = float(trigger_rad)
        self.shape = shape
        self.num_dofs = int(num_dofs)
        idx = (
            np.asarray(blend_indices, dtype=np.int64)
            if blend_indices is not None
            else _REF_SMOOTHER_JOINTS_PRESETS[_DEFAULT_REF_SMOOTHER_JOINTS]
        )
        if idx.ndim != 1:
            raise ValueError(
                f"_ReferenceStepSmoother: blend_indices must be 1-D, "
                f"got shape {idx.shape}"
            )
        if idx.size == 0:
            raise ValueError(
                "_ReferenceStepSmoother: blend_indices must be non-empty; "
                "use shape='off' to disable the smoother instead."
            )
        if int(idx.min()) < 0 or int(idx.max()) >= self.num_dofs:
            raise ValueError(
                f"_ReferenceStepSmoother: blend_indices out of range "
                f"[0, {self.num_dofs}); got "
                f"[{int(idx.min())}, {int(idx.max())}]"
            )
        self.blend_indices = idx
        self._log_fn = log_fn

        # Persistent state.
        self._last_published_q: Optional[np.ndarray] = None
        self._ramp_active: bool = False
        self._ramp_start_t: float = 0.0
        self._source_q: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        """True iff the smoother will actually shape the output.

        ``shape == 'off'`` and ``ramp_duration_s <= 0`` both short-
        circuit to passthrough; useful for callers that want to skip
        the per-tick subscribe / log overhead entirely.
        """
        return self.shape != "off" and self.ramp_duration_s > 0.0

    @staticmethod
    def _halfcos(x: float) -> float:
        """``0.5 * (1 - cos(pi * x))``, clamped to [0, 1] in input."""
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        return 0.5 * (1.0 - math.cos(math.pi * x))

    @staticmethod
    def _linear(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def _alpha(self, t_in_ramp: float) -> float:
        x = t_in_ramp / self.ramp_duration_s
        if self.shape == "halfcos":
            return self._halfcos(x)
        if self.shape == "linear":
            return self._linear(x)
        # 'off' is short-circuited in update(); arrive here only via tests
        # exercising _alpha directly.
        return 1.0 if x >= 1.0 else 0.0

    def _emit_arm_log(self, delta_max: float, worst_idx: int) -> None:
        if self._log_fn is None:
            return
        self._log_fn(
            "ref-smoother: armed (worst joint mj_idx=%d, delta=%.3f rad, "
            "trigger=%.3f rad, T=%.0f ms, shape=%s)",
            int(worst_idx),
            float(delta_max),
            float(self.trigger_rad),
            float(self.ramp_duration_s * 1000.0),
            self.shape,
        )

    def update(self, target_q: np.ndarray, t_now: float) -> np.ndarray:
        """Return the smoothed reference for this tick.

        Args:
            target_q: the would-be published joint reference, shape
                ``(num_dofs,)`` in MuJoCo joint order. Not modified.
            t_now: monotonic wall-clock seconds. Only differences are
                used; absolute origin is irrelevant.

        Returns:
            A new ``np.ndarray`` of dtype matching ``target_q``. On
            channels in ``blend_indices`` and during an active ramp,
            the value is the halfcos (or linear) blend of the snapshot
            source pose to the live target. Everywhere else, the value
            is ``target_q[i]`` verbatim. When ``shape='off'`` or
            ``ramp_duration_s <= 0`` the returned array is byte-equal
            to ``target_q``.
        """
        target_q = np.asarray(target_q)
        if target_q.shape != (self.num_dofs,):
            raise ValueError(
                f"_ReferenceStepSmoother.update: target_q shape "
                f"{target_q.shape} != ({self.num_dofs},)"
            )

        if not self.enabled:
            # Passthrough fast path; keep the last-published cache fresh
            # so the smoother can be re-enabled mid-run without seeding
            # a phantom step from a stale snapshot.
            self._last_published_q = target_q.astype(target_q.dtype, copy=True)
            return self._last_published_q.copy()

        if self._last_published_q is None:
            # First tick: nothing to detect a step against. Seed the
            # cache and pass the target through unchanged. The first
            # genuinely-detectable step lands on tick 2 at the earliest.
            self._last_published_q = target_q.astype(target_q.dtype, copy=True)
            return self._last_published_q.copy()

        # ---- Step detection (only on the blended channels).
        diff = target_q[self.blend_indices] - self._last_published_q[self.blend_indices]
        abs_diff = np.abs(diff)
        worst_local = int(np.argmax(abs_diff)) if abs_diff.size > 0 else 0
        delta_max = float(abs_diff[worst_local]) if abs_diff.size > 0 else 0.0
        worst_mj = int(self.blend_indices[worst_local]) if abs_diff.size > 0 else -1

        if (not self._ramp_active) and delta_max > self.trigger_rad:
            self._ramp_active = True
            self._ramp_start_t = float(t_now)
            self._source_q = self._last_published_q.astype(target_q.dtype, copy=True)
            self._emit_arm_log(delta_max, worst_mj)

        # ---- Build the output.
        if self._ramp_active and self._source_q is not None:
            t_in_ramp = float(t_now) - self._ramp_start_t
            if t_in_ramp >= self.ramp_duration_s:
                # Ramp complete -- emit the live target and clear state.
                self._ramp_active = False
                self._source_q = None
                out = target_q.astype(target_q.dtype, copy=True)
            else:
                alpha = self._alpha(t_in_ramp)
                out = target_q.astype(target_q.dtype, copy=True)
                bi = self.blend_indices
                out[bi] = (
                    (1.0 - alpha) * self._source_q[bi]
                    + alpha * target_q[bi]
                ).astype(target_q.dtype, copy=False)
        else:
            out = target_q.astype(target_q.dtype, copy=True)

        self._last_published_q = out
        return out.copy()

    def reset(self) -> None:
        """Clear all state. Safe to call from any caller-side reset path
        (planner restart, mode flip). The next ``update()`` call will
        re-seed the cache from its ``target_q`` and start fresh."""
        self._last_published_q = None
        self._ramp_active = False
        self._ramp_start_t = 0.0
        self._source_q = None


class IntentState:
    """Thread-safe holder for the current velocity-intent target.

    Tracks both the target value and the wall-clock time of the last
    update. The publish loop uses ``seconds_since_last_set`` to detect
    upstream silence (Quest 3 headset asleep, network drop, manager
    crash...) and force ``_IDLE_INTENT`` via ``force_idle_if_stale`` so
    the robot stops moving when commands stop arriving. Without this
    watchdog the planner would keep walking on the last non-idle intent
    indefinitely.
    """

    def __init__(self, initial: tuple[float, float, float, float]) -> None:
        self._target = initial
        self._lock = threading.Lock()
        self._version = 0  # bumped on every update; worker reads for logging
        # Initialise the "last set" timestamp to "now" so the watchdog
        # doesn't immediately trip during startup before any command
        # has arrived.
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
        self,
        max_age_s: float,
        idle_target: tuple[float, float, float, float],
    ) -> bool:
        """Snap target to ``idle_target`` if no command has arrived in
        ``max_age_s`` seconds AND the current target is non-idle.
        Returns True iff the watchdog tripped on this call.
        """
        with self._lock:
            age = time.monotonic() - self._last_set_t
            if age < max_age_s:
                return False
            if tuple(self._target) == tuple(idle_target):
                return False
            self._target = tuple(idle_target)
            self._version += 1
            # Do NOT bump ``_last_set_t``: we want subsequent ticks to
            # keep reporting the staleness so a periodic warning fires
            # while the upstream is silent. The next genuine command
            # from the manager will refresh both fields via ``set``.
            return True


def _planner_worker(
    planner_core,  # NeuralPlannerCore; not type-hinted to avoid eager import
    intent: IntentState,
    replan_lock: threading.Lock,
    stop_event: threading.Event,
    replan_event: threading.Event,
    pose_deque: "Optional[collections.deque[PoseObservation]]" = None,
    pose_lock: Optional[threading.Lock] = None,
    pose_max_age_s: float = 0.5,
    pose_reseed_scope: str = _RESEED_SCOPE_FULL_ROOT,
    cold_start_ramp_tau_s: float = _DEFAULT_COLD_START_RAMP_TAU_S,
    planner_mode_idx: Optional[int] = None,
) -> None:
    """Replan refill loop. Runs predict() each time the buffer drops below
    threshold, holding ``replan_lock`` only for the cursor swap (the predict
    call itself is unlocked so the publish thread can still pop frames).

    Idle gate: when the intent matches ``_IDLE_INTENT`` the publisher is
    holding the frozen default_angles anchor and never reads from the ring
    buffer, so running predict() here would just burn GPU cycles on
    frames nobody consumes -- and would also drift the buffer further
    from default_angles every call. Skip the replan while idle so the
    buffer's last-good state stays fresh for the next non-idle command.

    Closed-loop pose reseed: if ``pose_deque`` is provided, just before
    each replan we overwrite the 4 root rows the model will read as
    context with the robot's actually-observed pelvis qpos. Breaks the
    chain of self-conditioned predictions that compounds yaw drift.
    """
    log.info("planner worker thread started")
    feedback_enabled = pose_deque is not None and pose_lock is not None
    if feedback_enabled:
        log.info(
            "planner worker: closed-loop pose reseed ENABLED "
            "(scope=%s, max_age=%.3fs)",
            pose_reseed_scope, pose_max_age_s,
        )
    else:
        log.info("planner worker: closed-loop pose reseed DISABLED (open-loop)")

    cold_start_ramp = _ColdStartVelocityRamp(tau_s=cold_start_ramp_tau_s)
    if cold_start_ramp.enabled:
        log.info(
            "planner worker: cold-start velocity ramp ENABLED (tau=%.3fs); "
            "applies EWMA to (yaw_rate, vel_x, vel_z) channels on every "
            "idle -> playing entry so the model's implied target stays "
            "close to current pose for the first 2-3 replans",
            cold_start_ramp.tau_s,
        )
    else:
        log.info(
            "planner worker: cold-start velocity ramp DISABLED (tau=0); "
            "raw operator intent is forwarded verbatim (pre-fix behaviour)"
        )
    last_replan_mono: Optional[float] = None

    reseed_stats = {
        "applied": 0,
        "skipped_insufficient": 0,
        "skipped_stale": 0,
        "skipped_buffer_uninit": 0,
        "skipped_disabled": 0,
        "skipped_other": 0,
    }
    stats_log_every = 50

    while not stop_event.is_set():
        # Wait until the publisher signals it's draining the buffer, OR a
        # 50 ms timeout so we still check on stale buffers periodically.
        if not replan_event.wait(timeout=0.05):
            with replan_lock:
                needs_replan = planner_core.should_replan()
        else:
            replan_event.clear()
            needs_replan = True
        if not needs_replan or stop_event.is_set():
            continue
        target, ver = intent.get()
        if tuple(target) == _IDLE_INTENT:
            # Idle -- publisher holds the static anchor; no neural frames
            # are being consumed. Don't replan.
            #
            # Mark the cold-start ramp idle so the NEXT non-idle replan
            # starts ramping from zero again. Without this the ramp's
            # smoothed state would persist across idle gaps and the
            # operator's "release then re-push" pattern would skip the
            # ramp on the second push -- which IS what we want for
            # brief blips through idle, but breaks for sustained idle
            # where the buffer drifts away from walking context. The
            # idle gate in the worker doesn't track time-in-idle, so
            # we conservatively reset on every idle tick: a 1-tick
            # idle blip costs <50 ms of ramp time on resumption.
            cold_start_ramp.reset_idle()
            last_replan_mono = None
            continue

        # Cold-start velocity ramp: smooth (yaw_rate, vel_x, vel_z) on
        # idle -> playing transitions so the model's implied target
        # (= current_xy + vel * 2.13 s) doesn't jump 1 m+ ahead while
        # the context buffer still holds 4 frames of static stand
        # pose. hip_h (channel 3) passes through verbatim. See
        # ``_ColdStartVelocityRamp`` docstring for the full mechanism.
        now_mono = time.monotonic()
        if last_replan_mono is None:
            dt_s = 1.0 / OUTPUT_FPS  # first replan in this PLAYING segment
        else:
            dt_s = max(1e-3, now_mono - last_replan_mono)
        smoothed_target = cold_start_ramp.step(tuple(target), dt_s)
        last_replan_mono = now_mono
        if smoothed_target != tuple(target) and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "worker: cold-start ramp dt=%.3fs raw=%s -> smoothed=%s",
                dt_s, tuple(target), smoothed_target,
            )
        # Use the smoothed target from here on -- the model and the
        # closed-loop reseed both operate on the same vector.
        target = smoothed_target

        if feedback_enabled:
            with replan_lock:
                reason = _reseed_root_from_observations(
                    planner_core, pose_deque, pose_lock, pose_max_age_s,
                    scope=pose_reseed_scope,
                )
            if reason is None:
                reseed_stats["applied"] += 1
            elif reason.startswith("insufficient"):
                reseed_stats["skipped_insufficient"] += 1
            elif reason.startswith("stale"):
                reseed_stats["skipped_stale"] += 1
            elif reason.startswith("buffer_uninit"):
                reseed_stats["skipped_buffer_uninit"] += 1
            elif reason == "disabled":
                reseed_stats["skipped_disabled"] += 1
            else:
                reseed_stats["skipped_other"] += 1
            total = sum(reseed_stats.values())
            if total > 0 and total % stats_log_every == 0:
                log.info(
                    "reseed stats: total=%d applied=%d "
                    "insufficient=%d stale=%d buf_uninit=%d "
                    "disabled=%d other=%d",
                    total,
                    reseed_stats["applied"],
                    reseed_stats["skipped_insufficient"],
                    reseed_stats["skipped_stale"],
                    reseed_stats["skipped_buffer_uninit"],
                    reseed_stats["skipped_disabled"],
                    reseed_stats["skipped_other"],
                )

        # G1-demo-style replan line (post-ramp = the actual model input).
        # target = (yaw_rate, vel_x_lateral, vel_z_forward, hip_h).
        log.info(
            "Replanning with mode: %s, target_vel(fwd): %+.3f, lateral: %+.3f, "
            "yaw_rate: %+.3f, hip_h: %.3f  [v=%d]",
            ("velocity-only" if planner_mode_idx is None
             else f"template_idx={planner_mode_idx}"),
            float(target[2]), float(target[1]), float(target[0]),
            float(target[3]), ver,
        )
        t0 = time.monotonic()
        try:
            with replan_lock:
                if planner_mode_idx is None:
                    planner_core.replan_with_velocity(list(target))
                else:
                    planner_core.replan_with_pose_template(
                        list(target), mode_idx=planner_mode_idx,
                    )
        except Exception:
            log.exception("worker: replan failed; will retry next cycle")
            time.sleep(0.05)
            continue
        dt_ms = (time.monotonic() - t0) * 1000.0
        log.debug("worker: replan done in %.1fms (frames_remaining=%d)",
                  dt_ms, planner_core.frames_remaining)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def run(
    vqvae_ckpt: Path,
    pose_ckpt: Path,
    root_ckpt: Path,
    warmup_qpos_path: Optional[Path],
    pub_host: str,
    pub_port: int,
    pid_file: Path,
    demo_yaml: Optional[Path],
    enable_keyboard: bool,
    zmq_cmd_host: Optional[str],
    zmq_cmd_port: Optional[int],
    zmq_cmd_topic: str,
    zmq_cmd_bind: bool,
    duration_s: float,
    hand_dof: int,
    verbose: bool,
    warmup_quiet_stand_s: float,
    body_pose_port: Optional[int],
    device: str,
    replan_threshold_frames: int,
    planner_mode: Optional[str] = None,
    yaw_lock_epsilon_rad_s: float = 0.0,
    turn_left_scale: float = 1.0,
    turn_right_scale: float = 1.0,
    forward_scale: float = 1.0,
    backward_scale: float = 1.0,
    lateral_scale: float = 1.0,
    stick_shape_exp: float = _DEFAULT_STICK_SHAPING_EXPONENT,
    pose_feedback_host: Optional[str] = None,
    pose_feedback_port: Optional[int] = None,
    pose_feedback_topic: str = "robot_pose",
    pose_feedback_max_age_s: float = 0.5,
    pose_feedback_deque_maxlen: int = 32,
    pose_reseed_scope: str = _RESEED_SCOPE_FULL_ROOT,
    cold_start_ramp_tau_s: float = _DEFAULT_COLD_START_RAMP_TAU_S,
    continuous_turn_max_rad_s: float = _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S,
    continuous_forward_min_mps: float = _DEFAULT_CONTINUOUS_FORWARD_MIN_MPS,
    ref_smoother_ms: float = _DEFAULT_REF_SMOOTHER_MS,
    ref_smoother_trigger_rad: float = _DEFAULT_REF_SMOOTHER_TRIGGER_RAD,
    ref_smoother_shape: str = _DEFAULT_REF_SMOOTHER_SHAPE,
    ref_smoother_joints: str = _DEFAULT_REF_SMOOTHER_JOINTS,
    command_watchdog_s: float = 0.0,
) -> int:
    _setup_logging(verbose)

    # Stash runtime tuning scales on the module-level mutable singletons
    # used by ``intent_to_velocity``. Defaults are 1.0 so the static
    # unit-test invariants still hold when the user passes no override.
    global _RUNTIME_TURN_LEFT_SCALE, _RUNTIME_TURN_RIGHT_SCALE
    global _RUNTIME_FORWARD_SCALE, _RUNTIME_BACKWARD_SCALE, _RUNTIME_LATERAL_SCALE
    global _RUNTIME_STICK_SHAPING_EXPONENT
    global _CONTINUOUS_TURN_MAX_RAD_S
    global _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS
    _RUNTIME_TURN_LEFT_SCALE = float(turn_left_scale)
    _RUNTIME_TURN_RIGHT_SCALE = float(turn_right_scale)
    _RUNTIME_FORWARD_SCALE = float(forward_scale)
    _RUNTIME_BACKWARD_SCALE = float(backward_scale)
    _RUNTIME_LATERAL_SCALE = float(lateral_scale)
    if stick_shape_exp <= 0:
        log.error("--stick-shape-exp must be > 0 (got %s); using default %.2f",
                  stick_shape_exp, _DEFAULT_STICK_SHAPING_EXPONENT)
        stick_shape_exp = _DEFAULT_STICK_SHAPING_EXPONENT
    _RUNTIME_STICK_SHAPING_EXPONENT = float(stick_shape_exp)
    if continuous_turn_max_rad_s <= 0:
        log.error(
            "--continuous-turn-max-rad-s must be > 0 (got %s); using "
            "default %.3f rad/s",
            continuous_turn_max_rad_s, _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S,
        )
        continuous_turn_max_rad_s = _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S
    _CONTINUOUS_TURN_MAX_RAD_S = float(continuous_turn_max_rad_s)
    if continuous_forward_min_mps < 0.0:
        log.error(
            "--continuous-forward-min-mps must be >= 0 (got %s); "
            "disabling forward floor",
            continuous_forward_min_mps,
        )
        continuous_forward_min_mps = 0.0
    if continuous_forward_min_mps > _WALK_SPEED_MPS:
        log.warning(
            "--continuous-forward-min-mps (%.3f m/s) exceeds the forward "
            "stick cap _WALK_SPEED_MPS=%.3f m/s; every forward stick value "
            "will collapse onto the floor with no progressive control. "
            "Consider tuning down to <= %.3f m/s.",
            continuous_forward_min_mps, _WALK_SPEED_MPS, _WALK_SPEED_MPS,
        )
    _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS = float(continuous_forward_min_mps)

    # ---- Reference-step smoother validation + construction.
    # Built once here so the publish loop holds it by reference and the
    # config is logged exactly once. Validation is intentionally
    # permissive (clamp + warn rather than refuse) so a stale env-var
    # value can't take the whole stack down at startup.
    if ref_smoother_ms < 0.0:
        log.error(
            "--ref-smoother-ms must be >= 0 (got %s); disabling smoother",
            ref_smoother_ms,
        )
        ref_smoother_ms = 0.0
    if ref_smoother_trigger_rad < 0.0:
        log.error(
            "--ref-smoother-trigger-rad must be >= 0 (got %s); using default %.3f",
            ref_smoother_trigger_rad, _DEFAULT_REF_SMOOTHER_TRIGGER_RAD,
        )
        ref_smoother_trigger_rad = _DEFAULT_REF_SMOOTHER_TRIGGER_RAD
    if ref_smoother_shape not in _REF_SMOOTHER_SHAPES:
        log.error(
            "--ref-smoother-shape=%r not in %s; using default %r",
            ref_smoother_shape, _REF_SMOOTHER_SHAPES, _DEFAULT_REF_SMOOTHER_SHAPE,
        )
        ref_smoother_shape = _DEFAULT_REF_SMOOTHER_SHAPE
    if ref_smoother_joints not in _REF_SMOOTHER_JOINTS_PRESETS:
        log.error(
            "--ref-smoother-joints=%r not in %s; using default %r",
            ref_smoother_joints, tuple(_REF_SMOOTHER_JOINTS_PRESETS),
            _DEFAULT_REF_SMOOTHER_JOINTS,
        )
        ref_smoother_joints = _DEFAULT_REF_SMOOTHER_JOINTS
    ref_smoother = _ReferenceStepSmoother(
        ramp_duration_s=float(ref_smoother_ms) / 1000.0,
        trigger_rad=float(ref_smoother_trigger_rad),
        shape=ref_smoother_shape,
        blend_indices=_REF_SMOOTHER_JOINTS_PRESETS[ref_smoother_joints],
        log_fn=log.info,
    )
    log.info(
        "ref-smoother: shape=%s T=%.0f ms trigger=%.3f rad joints=%s (%d DoFs) enabled=%s",
        ref_smoother.shape,
        ref_smoother.ramp_duration_s * 1000.0,
        ref_smoother.trigger_rad,
        ref_smoother_joints,
        int(ref_smoother.blend_indices.size),
        ref_smoother.enabled,
    )

    if any(s != 1.0 for s in (
        turn_left_scale, turn_right_scale,
        forward_scale, backward_scale, lateral_scale,
    )):
        log.info(
            "velocity scales: turn_left=%.2f turn_right=%.2f "
            "forward=%.2f backward=%.2f lateral=%.2f",
            turn_left_scale, turn_right_scale,
            forward_scale, backward_scale, lateral_scale,
        )
    log.info("continuous-locomotion stick shape exponent: %.3f "
             "(1.0=linear, <1 closer-to-bucketed, >1 more deadzone-feel)",
             _RUNTIME_STICK_SHAPING_EXPONENT)
    log.info(
        "continuous-locomotion yaw ceiling: %.3f rad/s (~%.1f deg/s, "
        "90-deg turn in %.2f s); bucketed turn_*/deg_45 unchanged at "
        "%.3f rad/s",
        _CONTINUOUS_TURN_MAX_RAD_S,
        math.degrees(_CONTINUOUS_TURN_MAX_RAD_S),
        (math.pi / 2.0) / max(_CONTINUOUS_TURN_MAX_RAD_S, 1e-9),
        _TURN_45_RAD_S,
    )
    if _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS > 0.0:
        log.info(
            "continuous-locomotion forward floor: %.3f m/s "
            "(any post-deadzone forward stick lifts vel_z to this "
            "minimum; backward / lateral / yaw unaffected)",
            _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS,
        )
    else:
        log.info("continuous-locomotion forward floor: disabled (0.0 m/s)")
    log.info("yaw-lock epsilon: %.3f rad/s (0=disabled)",
             yaw_lock_epsilon_rad_s)

    # ---- Topic selection (mirrors heuristic daemon's two modes)
    if body_pose_port is not None:
        effective_port = body_pose_port
        effective_topic = "body_pose"
        log.info(
            "kplanner publish mode: 'body_pose' on tcp://%s:%d (Phase 0 recorder merge mode)",
            pub_host, effective_port,
        )
    else:
        effective_port = pub_port
        effective_topic = "pose"
        log.info(
            "kplanner publish mode: 'pose' on tcp://%s:%d (direct-to-deploy fallback)",
            pub_host, effective_port,
        )

    # ---- Pre-flight checks
    if _port_in_use(effective_port, "127.0.0.1") or _port_in_use(effective_port, "0.0.0.0"):
        log.error(
            "publish port %d already in use. Run the stack cleanup helper first.",
            effective_port,
        )
        return 1
    for label, p in (
        ("vqvae-ckpt", vqvae_ckpt),
        ("pose-ckpt", pose_ckpt),
        ("root-ckpt", root_ckpt),
    ):
        if not Path(p).is_file():
            log.error("%s not found: %s", label, p)
            return 1

    # ---- Load model + planner core (this is the slow part: ~5--10s on cold start)
    log.info("loading X2 kplanner stack on device=%s ...", device)
    log.info("  vqvae ckpt : %s", vqvae_ckpt)
    log.info("  pose  ckpt : %s", pose_ckpt)
    log.info("  root  ckpt : %s", root_ckpt)
    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths,
        load_x2_planner,
    )

    default_paths = X2PlannerPaths.default()
    paths = X2PlannerPaths(
        vqvae_ckpt=vqvae_ckpt,
        pose_ckpt=pose_ckpt,
        root_ckpt=root_ckpt,
        vqvae_version_dir=default_paths.vqvae_version_dir,
        pose_version_dir=default_paths.pose_version_dir,
        root_version_dir=default_paths.root_version_dir,
    )
    planner_core = load_x2_planner(
        paths,
        device=device,
        replan_threshold_frames=replan_threshold_frames,
    )
    log.info("kplanner stack loaded.")

    # ---- Pose-template inference path resolution.
    # ``--planner-mode`` maps the mode NAME to the integer index baked into
    # X2-clip.ckpt by motionbricks/scripts/build_x2_planner_clips.py
    # (DEFAULT_MODES insertion order = canonical binding). If the user
    # asked for pose-template mode but no clip library is loaded, fail
    # loudly here rather than silently falling back to velocity-only.
    _PLANNER_MODE_INDICES = ("idle", "slow_walk", "walk", "run_proxy")
    planner_mode_idx: Optional[int] = None
    if planner_mode is not None:
        if planner_core._clip_library is None:
            log.error(
                "--planner-mode=%s requested but NeuralPlannerCore has no "
                "clip library loaded. Run motionbricks/scripts/build_x2_planner_clips.py "
                "to bake out/X2-clip.ckpt first.",
                planner_mode,
            )
            return 2
        planner_mode_idx = _PLANNER_MODE_INDICES.index(planner_mode)
        log.info(
            "pose-template inference enabled: mode=%s (idx=%d, library has "
            "%d modes / %d joints)",
            planner_mode, planner_mode_idx,
            planner_core._clip_library.num_modes,
            planner_core._clip_library.num_joints,
        )
    else:
        log.info("planner path: legacy replan_with_velocity (velocity-only)")

    def _dispatch_replan(target_vec) -> None:
        """Call the right replan method based on --planner-mode."""
        if planner_mode_idx is None:
            planner_core.replan_with_velocity(list(target_vec))
        else:
            planner_core.replan_with_pose_template(
                list(target_vec), mode_idx=planner_mode_idx
            )

    # ---- Warmup anchor + first replan
    import torch
    warmup_qpos = _load_warmup_qpos(warmup_qpos_path)
    planner_core.reset(torch.from_numpy(warmup_qpos))
    intent_state = IntentState(_IDLE_INTENT)
    # First replan synchronously so the publish thread has frames immediately.
    _dispatch_replan(_IDLE_INTENT)
    log.info(
        "first replan complete; ring buffer has %d frames",
        planner_core.frames_remaining,
    )

    # ---- Publisher
    publisher = PosePublisher(
        host=pub_host,
        port=effective_port,
        topic=effective_topic,
        hand_dof=hand_dof,
    )
    log.info(
        "publishing %r on tcp://%s:%d at %.1f Hz",
        effective_topic, pub_host, effective_port, OUTPUT_FPS,
    )

    # ---- Command sources
    cmd_queue: "queue.Queue[LocomotionCommand]" = queue.Queue()
    stop_event = threading.Event()
    replan_event = threading.Event()
    replan_lock = threading.Lock()
    threads: list[threading.Thread] = []

    if demo_yaml is not None:
        thr = threading.Thread(
            target=_scripted_command_thread,
            args=(cmd_queue, demo_yaml, stop_event),
            name="cmd-scripted",
            daemon=True,
        )
        thr.start()
        threads.append(thr)

    if zmq_cmd_host is not None and zmq_cmd_port is not None:
        thr = threading.Thread(
            target=_zmq_command_thread,
            args=(cmd_queue, zmq_cmd_host, zmq_cmd_port, zmq_cmd_topic, stop_event,
                  zmq_cmd_bind),
            name="cmd-zmq",
            daemon=True,
        )
        thr.start()
        threads.append(thr)

    if enable_keyboard:
        thr = threading.Thread(
            target=_keyboard_command_thread,
            args=(cmd_queue, stop_event),
            name="cmd-kbd",
            daemon=True,
        )
        thr.start()
        threads.append(thr)

    pose_deque: "Optional[collections.deque[PoseObservation]]" = None
    pose_lock: Optional[threading.Lock] = None
    if pose_feedback_host is not None and pose_feedback_port is not None:
        pose_deque = collections.deque(maxlen=int(pose_feedback_deque_maxlen))
        pose_lock = threading.Lock()
        log.info(
            "closed-loop pose reseed: subscribing to %r on tcp://%s:%d "
            "(maxlen=%d, max_age=%.3fs)",
            pose_feedback_topic, pose_feedback_host, pose_feedback_port,
            int(pose_feedback_deque_maxlen), float(pose_feedback_max_age_s),
        )
        thr = threading.Thread(
            target=_pose_feedback_thread,
            args=(
                pose_deque, pose_lock,
                pose_feedback_host, pose_feedback_port, pose_feedback_topic,
                stop_event,
            ),
            name="pose-feedback",
            daemon=True,
        )
        thr.start()
        threads.append(thr)
    else:
        log.info(
            "closed-loop pose reseed: DISABLED "
            "(no --pose-feedback-host/port) -- running open-loop"
        )

    worker_thread = threading.Thread(
        target=_planner_worker,
        args=(planner_core, intent_state, replan_lock, stop_event, replan_event),
        kwargs={
            "pose_deque": pose_deque,
            "pose_lock": pose_lock,
            "pose_max_age_s": float(pose_feedback_max_age_s),
            "pose_reseed_scope": pose_reseed_scope,
            "cold_start_ramp_tau_s": float(cold_start_ramp_tau_s),
            "planner_mode_idx": planner_mode_idx,
        },
        name="kplanner-worker",
        daemon=True,
    )
    worker_thread.start()
    threads.append(worker_thread)

    # ---- Signal handlers
    def _on_signal(signum: int, _frame: object) -> None:
        log.info("signal %d -> draining and shutting down", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    period_s = 1.0 / OUTPUT_FPS
    next_tick = time.monotonic()
    end_at = (
        time.monotonic() + duration_s if duration_s > 0 else float("inf")
    )

    # The future window emitted alongside each frame (mirrors heuristic
    # daemon's tokenizer-obs contract). With OUTPUT_FPS=50 and
    # step_ticks=5, future[k] is at +(k+1)*0.1s from the current frame.
    num_future = 9
    step_ticks = 5

    global_tick = 0
    last_intent_log: tuple[float, float, float, float] = _IDLE_INTENT

    # ---- Continuous waist overlay (kplanner-side, v7.4).
    #
    # The neural model's intent space is a 4-D velocity vector
    # (yaw_rate, vel_x, vel_z, hip_h); it has no "command the body to
    # lean +15 deg" channel. Pre-v7.4 the manager could publish
    # ``hold_torso`` commands with non-zero waist_pitch / roll / yaw,
    # but the kplanner silently ignored them and the operator's stick
    # produced no waist motion.
    #
    # v7.4 closes that gap by maintaining a small slew-rate-limited
    # tracker (mirrors the heuristic planner's STATIC_HOLD path) and
    # adding a kinematic OVERLAY to the published joint vector each
    # tick: we add (target_pitch, target_roll, target_yaw) deltas to
    # the three waist joint slots in cur_frame.joint_pos_mj after the
    # neural model has produced its frame. This is intentionally
    # "pure waist" (no hip / ankle counter-balance shares) so the
    # overlay does NOT fight the model's leg motion when the
    # operator leans WHILE walking. The hip-share counter-balance the
    # heuristic planner uses is appropriate only for a feet-planted
    # static stand pose; during walking we want to leave the legs
    # entirely to the model.
    #
    # The tracker advances at OUTPUT_FPS (50 Hz) using HOLD_SLEW_DPS
    # (60 deg/s) -- same parameters as the heuristic STATIC_HOLD path,
    # so the operator-feel matches across planner backends.
    waist_tracker: _HoldTracker = _HoldTracker()
    last_waist_target_log: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Always-available frozen-anchor StreamFrame, shared by the warmup
    # ticks below AND the IDLE_LOOP branch of the main publish loop.
    # Built once from warmup_qpos so all three paths emit a bit-identical
    # pose (== deploy default_angles when the operator hasn't overridden
    # --kplanner-warmup-qpos). Hoisted out of the warmup-only branch so
    # the IDLE_LOOP gate works even when warmup_quiet_stand_s == 0
    # (e.g. tests that skip the warmup).
    anchor_frame = _qpos_to_stream_frame(
        warmup_qpos,
        frame_index=0,
        bin_name="kplanner_warmup_anchor",
        state=PlannerState.IDLE_LOOP,
    )

    # Integrated world pose persisted across IDLE_LOOP <-> PLAYING
    # transitions so the kplanner reference doesn't snap the robot back
    # to origin+identity quat every time the operator releases the stick.
    #
    # Without this, each new IDLE_LOOP entry freezes on the anchor's
    # root_quat = identity (= facing world +X), and the SONIC tracking
    # policy actively yaws the robot back to face forward after any
    # turn -- visible as "robot keeps coming back to the same direction".
    # Similarly each IDLE_LOOP -> PLAYING transition would reset
    # planner_core to the anchor (origin + identity), wiping the prior
    # translation / yaw so velocity intents start fresh from world (0,0,0)
    # -- visible as "robot stays fixed in place when commanded forward"
    # because the reference position keeps snapping back to origin.
    #
    # ``current_root_xy`` is in MuJoCo qpos units (m), ``current_root_wxyz``
    # in wxyz convention to match _BASE_VELOCITY's qpos layout. Joint
    # angles in IDLE_LOOP always snap to default_angles regardless;
    # only the root frame is preserved.
    current_root_xy = warmup_qpos[:2].astype(np.float32).copy()
    current_root_wxyz = warmup_qpos[3:7].astype(np.float32).copy()
    current_root_z = float(warmup_qpos[2])

    def _idle_root_xyzw() -> np.ndarray:
        """Current wxyz -> xyzw for the wire."""
        return np.array(
            [current_root_wxyz[1], current_root_wxyz[2],
             current_root_wxyz[3], current_root_wxyz[0]],
            dtype=np.float32,
        )

    def _build_warm_qpos() -> np.ndarray:
        """Construct a 38-D qpos at the current root frame with
        default_angles joints. Used to seed planner_core on
        IDLE_LOOP -> PLAYING so the neural buffer starts from the
        robot's current world pose rather than from origin.
        """
        warm = np.empty_like(warmup_qpos)
        warm[0] = current_root_xy[0]
        warm[1] = current_root_xy[1]
        warm[2] = current_root_z
        warm[3:7] = current_root_wxyz
        warm[7:] = warmup_qpos[7:]
        return warm

    try:
        with PidFile(pid_file):
            # ---- One-shot startup yaw seed from pose-feedback.
            #
            # The IDLE_LOOP yaw refresh (added 2026-06-01) only kicks in
            # once the main publish loop reaches its IDLE branch, but
            # the quiet-stand warmup runs FIRST and publishes
            # ``warmup_quiet_stand_s * OUTPUT_FPS`` frames at the
            # un-seeded root (= warmup_qpos[3:7] = identity quat / world
            # +X). The deploy treats the very first one of those as its
            # ``ZmqPoseInputSource`` bootstrap reference and the policy
            # twists the body to match -- this is the "VR planner stack
            # turns me back to default orientation as soon as I start
            # it" symptom on real robot.
            #
            # Fix: before publishing anything, block up to ~1s waiting
            # for a measured robot pose to land in the SUB queue, then
            # seed ``current_root_xy/z/wxyz`` from it. On sim runs the
            # MuJoCo bridge supplies robot_pose immediately; on real
            # robot the x2_debug -> robot_pose bridge (see
            # gear_sonic_deploy/scripts/x2_debug_to_robot_pose_bridge.py)
            # publishes within a few wifi RTTs. If neither is available
            # within the timeout we fall back to the warmup_qpos default
            # silently -- the IDLE_LOOP refresh will still correct it
            # eventually, and the C++ bootstrap fix protects against the
            # never-receive-a-frame case.
            if pose_deque is not None and pose_lock is not None:
                seed_timeout_s = 1.0
                seed_poll_s = 0.02
                seed_deadline = time.monotonic() + seed_timeout_s
                seeded = False
                while time.monotonic() < seed_deadline and not stop_event.is_set():
                    latest_seed_obs: Optional[PoseObservation] = None
                    with pose_lock:
                        if pose_deque:
                            latest_seed_obs = pose_deque[-1]
                    if latest_seed_obs is not None:
                        age_s = max(
                            0.0, time.monotonic() - latest_seed_obs.t_mono
                        )
                        if age_s <= float(pose_feedback_max_age_s):
                            # Yaw-only update. xy/z are NOT updated:
                            # the real-robot bridge publishes those as
                            # zeros (no IMU position measurement) and
                            # the warmup_qpos defaults (xy=0, z=hip_h
                            # from the PKL) are correct for both sim and
                            # real anyway. Matches the IDLE_LOOP refresh
                            # behaviour further down -- yaw is the only
                            # field that needs starvation-protection.
                            try:
                                current_root_wxyz = _yaw_only_wxyz_from_pelvis(
                                    latest_seed_obs.pelvis_qpos_wxyz
                                ).astype(np.float32)
                                log.info(
                                    "startup yaw seed: pose-feedback age=%.3fs, "
                                    "current_root_wxyz=%s (yaw-only re-projection "
                                    "of measured pelvis; xy/z left at warmup defaults)",
                                    age_s, current_root_wxyz.tolist(),
                                )
                                seeded = True
                                break
                            except (ValueError, TypeError) as exc:
                                log.warning(
                                    "startup yaw seed: rejected sample (%s); "
                                    "continuing to poll", exc,
                                )
                    time.sleep(seed_poll_s)
                if not seeded:
                    log.warning(
                        "startup yaw seed: TIMEOUT after %.2fs; falling back to "
                        "warmup_qpos[3:7]=%s. Quiet-stand warmup will publish "
                        "this stale reference; IDLE_LOOP refresh will correct "
                        "it once a measured pose lands. If you see snap-back, "
                        "check that the x2_debug -> robot_pose bridge is up "
                        "(--with-x2-debug-bridge on the planner stack launcher).",
                        seed_timeout_s, current_root_wxyz.tolist(),
                    )

            # ---- Optional quiet-stand warmup (publishes the frozen anchor qpos)
            warmup_n = (
                int(round(max(0.0, warmup_quiet_stand_s) * OUTPUT_FPS))
                if warmup_quiet_stand_s > 0 else 0
            )
            if warmup_n > 0:
                log.info(
                    "quiet-stand warmup: %d ticks (%.2fs) of frozen anchor qpos "
                    "(joint_pos=%s..., hip_h=%.2fm)",
                    warmup_n, warmup_quiet_stand_s,
                    warmup_qpos[7:10].tolist(), warmup_qpos[2],
                )
                for warm_idx in range(warmup_n):
                    if stop_event.is_set() or time.monotonic() >= end_at:
                        break
                    anchor_frame_idx = StreamFrame(
                        joint_pos_mj=anchor_frame.joint_pos_mj,
                        root_quat_xyzw=_idle_root_xyzw(),
                        root_xy_world=current_root_xy.astype(np.float64).copy(),
                        yaw_world_deg=anchor_frame.yaw_world_deg,
                        state=anchor_frame.state,
                        bin_name=anchor_frame.bin_name,
                        frame_index=warm_idx,
                        seam_blend=False,
                        root_z_world=float(current_root_z),
                    )
                    publisher.publish(anchor_frame_idx)
                    next_tick += period_s
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        next_tick = time.monotonic()
                global_tick = warmup_n
                log.info("quiet-stand warmup done; switching to neural planner.")

            # Track current high-level state for the idle-gate. When the
            # operator isn't pushing the stick (intent == _IDLE_INTENT) we
            # freeze on the warmup anchor (== deploy default_angles) so
            # the wire carries a perfectly stationary reference; the
            # neural planner only runs while a real locomotion command
            # is active. Without this gate the model freely predicts
            # walking-like motion in idle (visible in viewer as wide
            # arms / striding legs / slow yaw drift) since the planner
            # has no notion of "no motion intended" -- its training
            # distribution doesn't include a static-stand label.
            current_planner_state = PlannerState.IDLE_LOOP

            def _build_idle_future(start_idx: int) -> list[StreamFrame]:
                """num_future copies of the anchor pose at the CURRENT
                integrated world root, monotonically indexed. Mirrors
                the wire shape PLAYING emits so the C++ tokenizer's
                10x68 obs gather doesn't need a special idle-mode code
                path. Reading current_root_* fresh each call so a
                PLAYING -> IDLE_LOOP transition picks up the last
                published root pose for every future slot."""
                xyzw = _idle_root_xyzw()
                xy = current_root_xy.astype(np.float64).copy()
                z_world = float(current_root_z)
                return [
                    StreamFrame(
                        joint_pos_mj=anchor_frame.joint_pos_mj,
                        root_quat_xyzw=xyzw,
                        root_xy_world=xy.copy(),
                        yaw_world_deg=anchor_frame.yaw_world_deg,
                        state=PlannerState.IDLE_LOOP,
                        bin_name="kplanner_idle_future",
                        frame_index=start_idx + step_ticks * (k + 1),
                        seam_blend=False,
                        root_z_world=z_world,
                    )
                    for k in range(num_future)
                ]

            # Per-loop watchdog state -- only used when the upstream
            # source is the Quest 3 manager (or any other live source).
            # Throttled at ~1 Hz so an asleep/disconnected operator
            # doesn't flood the log.
            _watchdog_last_log_t = 0.0

            while not stop_event.is_set() and time.monotonic() < end_at:
                # ---- Drain command queue and apply the latest intent.
                latest_cmd: Optional[LocomotionCommand] = None
                while True:
                    try:
                        cmd = cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    latest_cmd = cmd
                if latest_cmd is not None:
                    target = intent_to_velocity(latest_cmd)
                    intent_state.set(target)
                    if target != last_intent_log:
                        log.info(
                            "intent applied (%s, %s, %s) -> target=%s",
                            latest_cmd.intent, latest_cmd.magnitude, latest_cmd.source,
                            target,
                        )
                        last_intent_log = target

                    # Waist-overlay target update. Honour every
                    # ``hold_torso`` command (continuous waist intent
                    # from the operator's R-stick / ARM_MAN L-stick);
                    # everything else implicitly resets the waist to
                    # neutral so a discrete ``walk / fwd_step / turn_*``
                    # command relaxes the lean cleanly. The tracker's
                    # slew limit handles the cross-tick smoothness;
                    # there's no need to debounce here because the
                    # decoder already throttles publishing.
                    #
                    # Scope: this branch only fires when a fresh
                    # ``latest_cmd`` was dequeued this tick. The
                    # watchdog branch below resets to neutral on
                    # upstream silence -- so we don't need (and must
                    # not duplicate) a NULL-cmd code path here.
                    # Regression note: pre-2026-06-26, this block was
                    # accidentally nested into the watchdog branch by
                    # the watchdog-introducing diff, causing a
                    # ``latest_cmd.intent`` AttributeError as soon as
                    # the operator went idle for >0.5 s.
                    if latest_cmd.intent == HOLD_TORSO_INTENT:
                        new_waist_target = (
                            float(latest_cmd.waist_pitch_deg),
                            float(latest_cmd.waist_roll_deg),
                            float(latest_cmd.waist_yaw_deg),
                        )
                    else:
                        new_waist_target = (0.0, 0.0, 0.0)
                    waist_tracker.set_target(*new_waist_target)
                    if new_waist_target != last_waist_target_log:
                        log.info(
                            "waist overlay target updated (%s, %s) -> "
                            "pitch=%.1f roll=%.1f yaw=%.1f deg",
                            latest_cmd.intent, latest_cmd.magnitude,
                            *new_waist_target,
                        )
                        last_waist_target_log = new_waist_target

                # ---- Stale-command watchdog (opt-in; default OFF).
                # If the upstream stops publishing, the last non-IDLE
                # intent would otherwise keep driving locomotion forever.
                # Snap to IDLE after ``command_watchdog_s`` seconds of
                # silence and reset the waist target to neutral so a
                # stuck ``hold_torso`` lean relaxes too.
                #
                # **Why default OFF**: the Quest 3 manager's IntentDecoder
                # is event-driven -- it publishes a new planner_cmd only
                # when the stick value changes meaningfully (see
                # ``intent_decoder._maybe_emit`` + ``_is_significant_change``).
                # A held stick can therefore produce inter-arrival gaps
                # of multiple seconds during NORMAL operation. We measured
                # gaps up to ~4 s with the operator actively turning at
                # constant rate (manager_sidecar.jsonl, 2026-06-26 stack).
                # A watchdog at 0.5 s would abort every locomotion gesture
                # after ~0.5 s -- exactly the bug observed on the real
                # robot before this knob was disabled.
                #
                # Upstream-silence detection therefore belongs in the
                # manager (``quest3_manager_x2.py --vr-input-max-age-s``),
                # which has access to the raw WebSocket packet rate and
                # forces sticks to neutral when the WS goes silent.
                # The decoder then emits an explicit idle command and
                # the kplanner stops cleanly.
                #
                # Enable this knob only for upstream sources that DO emit
                # a steady heartbeat at a known rate (e.g. PKL replay,
                # scripted demos publishing every tick).
                if command_watchdog_s > 0.0 and intent_state.force_idle_if_stale(
                    max_age_s=command_watchdog_s,
                    idle_target=_IDLE_INTENT,
                ):
                    waist_tracker.set_target(0.0, 0.0, 0.0)
                    last_intent_log = _IDLE_INTENT
                    last_waist_target_log = (0.0, 0.0, 0.0)
                    now_t = time.monotonic()
                    if now_t - _watchdog_last_log_t > 1.0:
                        log.warning(
                            "command watchdog: no upstream intent for "
                            "%.2fs (threshold %.2fs); forcing IDLE",
                            intent_state.seconds_since_last_set(),
                            command_watchdog_s,
                        )
                        _watchdog_last_log_t = now_t

                # ---- Resolve high-level state from current intent.
                # IDLE_LOOP <-> PLAYING transitions are edge-triggered
                # so we can log them and seed the neural buffer cleanly
                # on every entry to PLAYING.
                current_target, _ = intent_state.get()
                is_idle = tuple(current_target) == _IDLE_INTENT
                desired_state = (
                    PlannerState.IDLE_LOOP if is_idle else PlannerState.PLAYING
                )
                if desired_state != current_planner_state:
                    if desired_state == PlannerState.PLAYING:
                        # One-shot yaw refresh on IDLE -> PLAYING entry.
                        # The IDLE_LOOP branch below refreshes
                        # ``current_root_wxyz`` every tick while idle,
                        # so when the operator releases the stick after
                        # a gesture (the common workflow) the warm seed
                        # is already measured-aligned. But if the
                        # operator HOLDS the stick through a gesture --
                        # so the kplanner stays in PLAYING the whole
                        # time and gesture playback inside the recorder
                        # silently moves the robot to a new heading --
                        # there's no IDLE tick to refresh, and the
                        # next ``_build_warm_qpos()`` would seed the
                        # neural buffer at a stale model-integrated
                        # yaw. Snap-back symptom on PLAYING resume.
                        # Refresh here as a belt-and-braces measure;
                        # cost is one yaw extraction per state
                        # transition (~rare), and we never block more
                        # than the existing pose_feedback_max_age_s
                        # gate would allow.
                        if pose_deque is not None and pose_lock is not None:
                            latest_entry_obs: Optional[PoseObservation] = None
                            with pose_lock:
                                if pose_deque:
                                    latest_entry_obs = pose_deque[-1]
                            if latest_entry_obs is not None:
                                age_s = (
                                    time.monotonic() - latest_entry_obs.t_mono
                                )
                                if age_s <= float(pose_feedback_max_age_s):
                                    try:
                                        prev_wxyz = current_root_wxyz.copy()
                                        current_root_wxyz = (
                                            _yaw_only_wxyz_from_pelvis(
                                                latest_entry_obs.pelvis_qpos_wxyz
                                            ).astype(np.float32)
                                        )
                                        if not np.allclose(prev_wxyz, current_root_wxyz, atol=1e-4):
                                            log.info(
                                                "IDLE -> PLAYING yaw refresh: "
                                                "pose-feedback age=%.3fs, "
                                                "prev=%s -> measured=%s",
                                                age_s,
                                                prev_wxyz.tolist(),
                                                current_root_wxyz.tolist(),
                                            )
                                    except (ValueError, TypeError) as exc:
                                        log.debug(
                                            "IDLE -> PLAYING yaw refresh "
                                            "skipped: %s", exc,
                                        )

                        # Seed the ring buffer with default_angles at
                        # the ROBOT'S CURRENT integrated root frame
                        # (not at world origin / identity quat). This
                        # makes the neural planner's canonicalize step
                        # treat the operator's velocity intent as
                        # "starting from where the robot currently is",
                        # so a fwd_step after a 90-degree turn walks in
                        # the new heading direction, not in world +X.
                        # Without this seed the planner snapped the
                        # reference back to (0,0,0) + identity every
                        # PLAYING entry, fighting the robot's physical
                        # drift.
                        warm = _build_warm_qpos()
                        with replan_lock:
                            planner_core.reset(torch.from_numpy(warm))
                        replan_event.set()
                        log.info(
                            "state: IDLE_LOOP -> PLAYING (intent=%s); "
                            "buffer seeded at root_xy=%s yaw_wxyz=%s, replan queued",
                            current_target,
                            warm[:2].tolist(),
                            warm[3:7].tolist(),
                        )
                    else:
                        log.info(
                            "state: PLAYING -> IDLE_LOOP (intent back to idle); "
                            "freezing at root_xy=%s yaw_wxyz=%s",
                            current_root_xy.tolist(),
                            current_root_wxyz.tolist(),
                        )
                    current_planner_state = desired_state

                # ---- Build the published frame.
                if current_planner_state == PlannerState.IDLE_LOOP:
                    # Yaw-only resync from robot_pose feedback before we
                    # publish. Without this, ``current_root_wxyz`` is
                    # only ever updated by the model's own predictions
                    # (PLAYING branch below), so anything that moves
                    # the real robot off-yaw while the operator's stick
                    # is centred -- fall recovery, slip, push, or a
                    # recorder-side gesture override (sit_down /
                    # stand_up that turns the body) -- leaves us
                    # publishing a stale absolute yaw target. The C++
                    # tokenizer feeds the SONIC policy
                    # ``rel = inv(measured) * reference`` (see
                    # gear_sonic_deploy/src/x2/.../tokenizer_obs.cpp:114),
                    # so a stale reference causes the policy to twist
                    # the body back to the old heading -- the "robot
                    # always tries to recover to the same world
                    # orientation" symptom. Refreshing here closes that
                    # loop for the idle path while PLAYING continues
                    # to publish model-predicted yaw verbatim (so
                    # commanded turns still execute as intended).
                    if pose_deque is not None and pose_lock is not None:
                        latest_obs: Optional[PoseObservation] = None
                        with pose_lock:
                            if pose_deque:
                                latest_obs = pose_deque[-1]
                        if latest_obs is not None:
                            age_s = time.monotonic() - latest_obs.t_mono
                            if age_s <= float(pose_feedback_max_age_s):
                                try:
                                    current_root_wxyz = _yaw_only_wxyz_from_pelvis(
                                        latest_obs.pelvis_qpos_wxyz
                                    )
                                except (ValueError, TypeError) as exc:
                                    log.debug(
                                        "yaw refresh skipped: %s "
                                        "(continuing with stale current_root_wxyz)",
                                        exc,
                                    )

                    # Frozen-anchor branch: emit default_angles joints
                    # at the LAST integrated world root pose. Joint
                    # angles snap to anchor (default_angles); root XY
                    # and yaw carry over from the prior PLAYING session
                    # (or from the latest pose_deque refresh above)
                    # so a release-after-turn doesn't trigger a SONIC
                    # yaw-correction back toward identity.
                    cur_frame = StreamFrame(
                        joint_pos_mj=anchor_frame.joint_pos_mj,
                        root_quat_xyzw=_idle_root_xyzw(),
                        root_xy_world=current_root_xy.astype(np.float64).copy(),
                        yaw_world_deg=anchor_frame.yaw_world_deg,
                        state=PlannerState.IDLE_LOOP,
                        bin_name="kplanner_idle",
                        frame_index=global_tick,
                        seam_blend=False,
                        root_z_world=float(current_root_z),
                    )
                    future_frames = _build_idle_future(global_tick)
                else:
                    # PLAYING branch: sample from the neural ring buffer
                    # exactly as before, with the lookahead window the
                    # tokenizer expects.
                    with replan_lock:
                        # Resampled read: advance a float cursor by
                        # model_fps/OUTPUT_FPS (=0.6) native frames per tick
                        # and interpolate, instead of the legacy +1 whole
                        # 30 fps frame per 50 Hz tick (which played the
                        # reference 1.67x too fast). An 8-tick cross-fade
                        # blends successive planner outputs at each replan
                        # seam (both handled inside NeuralPlannerCore).
                        qpos_tensor = planner_core.get_next_frame_resampled(OUTPUT_FPS)
                        if planner_core.should_replan():
                            replan_event.set()
                        future_qposes: list[np.ndarray] = []
                        for k in range(num_future):
                            # Future slot k is +0.1*(k+1) s in REAL time:
                            # step_ticks(=5) output ticks * 0.6 native/tick =
                            # 3 native frames = 0.1 s. Time-correct spacing
                            # on the resampled timeline (was +5 native = 0.167s).
                            fq = planner_core.peek_output_frame(step_ticks * (k + 1))
                            future_qposes.append(fq.detach().cpu().numpy())
                    qpos_np = qpos_tensor.detach().cpu().numpy()

                    # Yaw-lock mitigation. When the operator's commanded
                    # yaw_rate is below threshold (pure forward / back /
                    # sidestep / idle), the model still predicts small
                    # per-frame yaw deltas that compound across replans
                    # into a visible spin. The deploy's SONIC policy
                    # tracks the published root_quat, so this drift
                    # makes the robot rotate instead of translate.
                    # Override the published yaw with the persisted
                    # wxyz when commanded yaw is near zero -- joint
                    # angles still produce stepping, only the root
                    # orientation reference is pinned. ``--yaw-lock-
                    # epsilon`` raises the threshold (0 disables, large
                    # values lock more aggressively).
                    yaw_locked = (
                        yaw_lock_epsilon_rad_s > 0.0
                        and abs(float(current_target[0])) < yaw_lock_epsilon_rad_s
                    )
                    if yaw_locked:
                        qpos_np[3:7] = current_root_wxyz.astype(qpos_np.dtype)
                        for fq in future_qposes:
                            fq[3:7] = current_root_wxyz.astype(fq.dtype)

                    # Persist the world-frame root for the next
                    # IDLE_LOOP freeze / PLAYING seed.
                    current_root_xy = qpos_np[:2].astype(np.float32).copy()
                    current_root_z = float(qpos_np[2])
                    if not yaw_locked:
                        # NeuralPlannerCore.replan_with_velocity packs the
                        # qpos's quat as wxyz in slots [3:7]; mirror that
                        # ordering when persisting (the xyzw permutation
                        # only happens in _qpos_to_stream_frame at publish).
                        # When yaw is locked we already overrode the
                        # published wxyz with the persisted value, so
                        # there's nothing to update.
                        current_root_wxyz = qpos_np[3:7].astype(np.float32).copy()
                    cur_frame = _qpos_to_stream_frame(
                        qpos_np,
                        frame_index=global_tick,
                        bin_name="kplanner",
                        state=PlannerState.PLAYING,
                    )
                    future_frames = [
                        _qpos_to_stream_frame(
                            future_qposes[k],
                            frame_index=global_tick + step_ticks * (k + 1),
                            bin_name="kplanner_future",
                            state=PlannerState.PLAYING,
                        )
                        for k in range(num_future)
                    ]

                # ---- Waist overlay (v7.4, kplanner-side STATIC_HOLD analogue).
                # Step the tracker so ``current_*`` walks toward the
                # operator-commanded target at HOLD_SLEW_DPS (60 deg/s),
                # then add the (current_pitch, current_roll, current_yaw)
                # deltas onto the published waist joint slots. We
                # operate on a copy of joint_pos_mj because the IDLE
                # branch shares ``anchor_frame.joint_pos_mj`` across
                # ticks; mutating it in place would compound the
                # overlay across consecutive idle ticks.
                #
                # PLAYING branch: only the 3 waist DOFs are touched so
                # the kplanner's leg / arm motion is preserved
                # byte-identical. IDLE_LOOP branch: same; the static
                # stand pose's hip / ankle joints stay at default. We
                # intentionally DO NOT apply the heuristic planner's
                # hip / ankle counter-balance shares here -- those make
                # sense for a feet-planted lean but would visually
                # fight the kplanner's stride during locomotion.
                # Operators wanting the "natural deadlift hinge" feel
                # can fall back to ``--planner heuristic`` for static
                # work; the kplanner trades that bit of ergonomic
                # realism for a unified walk + lean experience.
                waist_tracker.step(dt_s=period_s)
                if (
                    waist_tracker.current_pitch_deg != 0.0
                    or waist_tracker.current_roll_deg != 0.0
                    or waist_tracker.current_yaw_deg != 0.0
                ):
                    overlay_q = cur_frame.joint_pos_mj.copy()
                    overlay_q[WAIST_PITCH_IDX] += float(
                        np.deg2rad(waist_tracker.current_pitch_deg)
                    )
                    overlay_q[WAIST_ROLL_IDX] += float(
                        np.deg2rad(waist_tracker.current_roll_deg)
                    )
                    overlay_q[WAIST_YAW_IDX] += float(
                        np.deg2rad(waist_tracker.current_yaw_deg)
                    )
                    cur_frame = StreamFrame(
                        joint_pos_mj=overlay_q.astype(np.float32, copy=False),
                        root_quat_xyzw=cur_frame.root_quat_xyzw,
                        root_xy_world=cur_frame.root_xy_world,
                        yaw_world_deg=cur_frame.yaw_world_deg,
                        state=cur_frame.state,
                        bin_name=cur_frame.bin_name,
                        frame_index=cur_frame.frame_index,
                        seam_blend=cur_frame.seam_blend,
                        root_z_world=cur_frame.root_z_world,
                    )
                    if future_frames:
                        overlaid_futures: list[StreamFrame] = []
                        for ff in future_frames:
                            fq = ff.joint_pos_mj.copy()
                            fq[WAIST_PITCH_IDX] += float(
                                np.deg2rad(waist_tracker.current_pitch_deg)
                            )
                            fq[WAIST_ROLL_IDX] += float(
                                np.deg2rad(waist_tracker.current_roll_deg)
                            )
                            fq[WAIST_YAW_IDX] += float(
                                np.deg2rad(waist_tracker.current_yaw_deg)
                            )
                            overlaid_futures.append(StreamFrame(
                                joint_pos_mj=fq.astype(np.float32, copy=False),
                                root_quat_xyzw=ff.root_quat_xyzw,
                                root_xy_world=ff.root_xy_world,
                                yaw_world_deg=ff.yaw_world_deg,
                                state=ff.state,
                                bin_name=ff.bin_name,
                                frame_index=ff.frame_index,
                                seam_blend=ff.seam_blend,
                                root_z_world=ff.root_z_world,
                            ))
                        future_frames = overlaid_futures

                # ---- Reference-step smoother (publisher output, step-driven).
                # Applied at a single point AFTER both IDLE / PLAYING
                # branches have settled on ``cur_frame.joint_pos_mj`` so
                # the smoother sees whatever the FSM produced regardless
                # of which branch ran. The smoother is intentionally
                # agnostic to the FSM -- it watches per-tick reference
                # deltas on lower-body joints (legs + waist by default)
                # and arms a half-cosine ramp whenever the delta exceeds
                # ``--ref-smoother-trigger-rad``. Stick push, stick
                # release, primitive switches, future MC-handoff piping
                # -- all hit the same code path because they all create
                # a step on the wire. ``cur_frame`` is a dataclass; we
                # rebuild it with the smoothed joints, preserving every
                # other field byte-equivalent so root quat / xy / state
                # / bin_name semantics are untouched.
                smoothed_q = ref_smoother.update(
                    cur_frame.joint_pos_mj, time.monotonic()
                )
                cur_frame = StreamFrame(
                    joint_pos_mj=smoothed_q.astype(np.float32, copy=False),
                    root_quat_xyzw=cur_frame.root_quat_xyzw,
                    root_xy_world=cur_frame.root_xy_world,
                    yaw_world_deg=cur_frame.yaw_world_deg,
                    state=cur_frame.state,
                    bin_name=cur_frame.bin_name,
                    frame_index=cur_frame.frame_index,
                    seam_blend=cur_frame.seam_blend,
                    root_z_world=cur_frame.root_z_world,
                )

                publisher.publish(cur_frame, future_frames=future_frames, future_dt_s=0.1)
                global_tick += 1

                next_tick += period_s
                slack = next_tick - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    if -slack > 5 * period_s:
                        log.warning(
                            "loop fell behind by %.0fms; resyncing", -slack * 1000
                        )
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="x2_kplanner",
        description=(
            "Neural kinematic locomotion planner for AgiBot X2 — streams 31-DOF "
            "body refs + root quat over the 'body_pose' (or 'pose') ZMQ topic "
            "at 50 Hz, conditioned on a Quest3 / scripted velocity intent."
        ),
    )
    # Pinned step checkpoints (not last.ckpt) so a fresh training run
    # doesn't silently re-point inference at an unverified checkpoint.
    # Override via the CLI when promoting a newer checkpoint.
    p.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=(
            _REPO_ROOT
            / "motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0500000.ckpt"
        ),
    )
    p.add_argument(
        "--pose-ckpt",
        type=Path,
        default=(
            _REPO_ROOT
            / "motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/model-step=0500000.ckpt"
        ),
    )
    p.add_argument(
        "--root-ckpt",
        type=Path,
        default=(
            _REPO_ROOT
            / "motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0315000.ckpt"
        ),
    )
    p.add_argument(
        "--warmup-qpos-path",
        type=Path,
        default=None,
        help=(
            "Optional PKL with a stand qpos[38] (or [T, 38] -> frame 0). "
            "Used to seed the ring buffer and as the warmup-quiet-stand "
            "anchor. Falls back to a hand-crafted zero-joints stand with "
            f"hip_h={_HIP_HEIGHT_M:.2f}m + identity quat."
        ),
    )
    p.add_argument(
        "--pub-host", default="127.0.0.1",
        help="ZMQ publish bind host (default: 127.0.0.1).",
    )
    p.add_argument(
        "--pub-port", type=int, default=5556,
        help="ZMQ publish port (default: 5556).",
    )
    p.add_argument(
        "--body-pose-port", type=int, default=None,
        help=(
            "Publish under topic 'body_pose' on this port (Phase 0 recorder "
            "merge mode). Mutually exclusive with --pub-port's direct-to-deploy "
            "mode; when set, overrides --pub-port."
        ),
    )
    p.add_argument(
        "--pid-file",
        type=Path,
        default=Path("/tmp/x2_kplanner.pid"),
    )
    p.add_argument("--demo", type=Path, help="Scripted demo YAML.")
    p.add_argument("--keyboard", action="store_true",
                   help="Enable interactive keyboard input.")
    p.add_argument(
        "--zmq-cmd-host", default=None,
        help="If set, also subscribe for high-level commands on tcp://host:port",
    )
    p.add_argument("--zmq-cmd-port", type=int, default=None)
    p.add_argument(
        "--zmq-cmd-topic", default="planner_cmd",
        help="Topic prefix expected on the command SUB socket.",
    )
    p.add_argument(
        "--zmq-cmd-bind", action="store_true",
        help="Bind the command SUB instead of connecting, so multiple "
             "sources (pad bridge + quest3 manager) can PUB-connect into "
             "it; --zmq-cmd-host is then ignored (bind is on *).",
    )
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--hand-dof", type=int, default=10)
    p.add_argument(
        "--warmup-quiet-stand-s", type=float, default=0.5,
        help=(
            "Publish the warmup-qpos anchor frame for this many seconds at "
            "startup before the neural planner takes over."
        ),
    )
    p.add_argument(
        "--device", default="cuda",
        help="torch device (cuda / cpu). Falls back to cpu if cuda unusable.",
    )
    p.add_argument(
        "--replan-threshold-frames", type=int, default=16,
        help=(
            "Worker thread refills the ring buffer when frames_remaining < this "
            "value. With OUTPUT_FPS=50, 16 frames = 0.32 s of headroom -- safe "
            "even with 5--15 ms predict() jitter."
        ),
    )
    p.add_argument(
        "--planner-mode",
        type=str,
        default=None,
        choices=("idle", "slow_walk", "walk", "run_proxy"),
        help=(
            "When set, replan via NeuralPlannerCore.replan_with_pose_template() "
            "using the named mode from the baked X2-clip.ckpt library "
            "(see motionbricks/scripts/build_x2_planner_clips.py for the bake "
            "and the mode -> clip binding). Default None preserves the legacy "
            "velocity-only replan_with_velocity() path. Phase 2 of the "
            "x2_kplanner_pose-template_inference plan."
        ),
    )
    tune_grp = p.add_argument_group(
        "velocity tuning",
        "Runtime knobs for compensating model-side asymmetries and quelling "
        "low-frequency yaw drift without rebuilding the static intent tables.",
    )
    tune_grp.add_argument(
        "--yaw-lock-epsilon", type=float, default=0.0,
        help=(
            "Opt-in diagnostic mitigation. When > 0 and the commanded "
            "yaw_rate is below this threshold (rad/s) the kplanner "
            "overrides the published root_quat with the persisted yaw, "
            "suppressing cumulative yaw drift during translation intents. "
            "DEFAULT 0.0 (disabled) -- enabling this also kills the gait's "
            "natural yaw oscillation, which the SONIC policy needs as a "
            "phase signal to initiate stepping; the robot stops walking "
            "entirely. Use only for diagnostics (e.g. confirm the spin is "
            "from yaw drift rather than a joint-tracking failure)."
        ),
    )
    tune_grp.add_argument(
        "--turn-left-scale", type=float, default=1.0,
        help=(
            "Multiplier on turn_left yaw_rate intents only. >1.0 boosts left "
            "turns to compensate for training-data L/R asymmetry; <1.0 "
            "attenuates them. Default 1.0 (no-op)."
        ),
    )
    tune_grp.add_argument(
        "--turn-right-scale", type=float, default=1.0,
        help="Symmetric multiplier on turn_right yaw_rate. Default 1.0.",
    )
    tune_grp.add_argument(
        "--forward-scale", type=float, default=1.0,
        help="Multiplier on fwd_step / walk-forward vel_x. Default 1.0.",
    )
    tune_grp.add_argument(
        "--backward-scale", type=float, default=1.0,
        help="Multiplier on back_step / walk-backward vel_x. Default 1.0.",
    )
    tune_grp.add_argument(
        "--lateral-scale", type=float, default=1.0,
        help="Multiplier on side_left / side_right vel_y. Default 1.0.",
    )
    tune_grp.add_argument(
        "--stick-shape-exp", type=float,
        default=_DEFAULT_STICK_SHAPING_EXPONENT,
        help=(
            "Power-curve exponent applied to ``locomotion / continuous`` "
            "stick deflections before scaling by the base velocity. "
            "1.0 (default) = linear; >1.0 (e.g. 2.0) = more dead near zero, "
            "harder to reach full speed; <1.0 (e.g. 0.5) = closer to the "
            "bucketed feel (50%% stick already at 71%% velocity)."
        ),
    )
    fb_grp = p.add_argument_group(
        "closed-loop pose feedback",
        "Break the kplanner's chain of self-conditioned predictions by "
        "refreshing the context window from the robot's actually-observed "
        "pelvis pose before each replan (breaks the yaw-drift loop of "
        "self-conditioned replans).",
    )
    fb_grp.add_argument(
        "--pose-feedback-host", default=None,
        help=(
            "Sim bridge host that publishes the 'robot_pose' topic. "
            "When set together with --pose-feedback-port enables closed-loop "
            "pose reseed; default (None) keeps the open-loop behaviour."
        ),
    )
    fb_grp.add_argument(
        "--pose-feedback-port", type=int, default=None,
        help="ZMQ PUB port for 'robot_pose' (sim bridge default: 5570).",
    )
    fb_grp.add_argument(
        "--pose-feedback-topic", default="robot_pose",
        help="Topic name on the pose-feedback SUB socket. Default 'robot_pose'.",
    )
    fb_grp.add_argument(
        "--pose-feedback-max-age-s", type=float, default=0.5,
        help=(
            "If the newest pose observation is older than this many seconds "
            "we skip the reseed for this replan and let the planner fall "
            "back to its own predictions. Default 0.5 s (15 frames @ 30 Hz)."
        ),
    )
    fb_grp.add_argument(
        "--pose-feedback-deque-maxlen", type=int, default=32,
        help=(
            "Capacity of the rolling buffer of observations the feedback "
            "thread fills. Each replan reads the latest 4 entries. Default "
            "32 = ~640 ms of headroom at the bridge's 50 Hz publish rate."
        ),
    )
    fb_grp.add_argument(
        "--pose-reseed-scope",
        choices=list(_VALID_RESEED_SCOPES),
        default=_RESEED_SCOPE_FULL_ROOT,
        help=(
            "Which root channels the PLAYING-side reseed rewrites every "
            "replan tick. 'full_root' (default) overwrites xyz + quat -- "
            "best for sim, where the MuJoCo bridge supplies full "
            "ground-truth qpos. 'quat_only' overwrites just the quaternion "
            "-- preserves the planner's internal-model xy overshoot (which "
            "helps the policy track forward motion) while still anchoring "
            "heading to observed reality; use this with a pose source that "
            "has VALID xy/z but you want quat anchoring only. 'none' "
            "disables the PLAYING reseed entirely -- pose_deque still "
            "feeds the IDLE_LOOP yaw refresh and the IDLE -> PLAYING "
            "transition seed (both yaw-only, both writing to "
            "current_root_wxyz only), but the model's neural buffer is "
            "left untouched during PLAYING. REQUIRED on real robot with "
            "the IMU-only x2_debug bridge: 'full_root' would teleport the "
            "planner's xy history to zero every tick, and 'quat_only' "
            "would anchor yaw integration to the lagging measured yaw "
            "(commanded turns under-rotate)."
        ),
    )
    p.add_argument(
        "--continuous-turn-max-rad-s",
        type=float,
        default=_DEFAULT_CONTINUOUS_TURN_MAX_RAD_S,
        help=(
            "Yaw-rate ceiling (rad/s) at full R-stick deflection in the "
            "continuous-locomotion path. Default %.3f rad/s (~43 deg/s, a "
            "90-deg turn in ~2.1 s). The bucketed path keeps its own "
            "_TURN_45_RAD_S = 1.5 rad/s ceiling -- this knob only affects "
            "Quest 3 R-stick X analog turns. The per-side runtime turn "
            "scales (--turn-left-scale / --turn-right-scale) still apply "
            "on top of this ceiling for L/R asymmetry compensation."
        ) % _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S,
    )
    p.add_argument(
        "--continuous-forward-min-mps",
        type=float,
        default=_DEFAULT_CONTINUOUS_FORWARD_MIN_MPS,
        help=(
            "Minimum forward velocity (m/s) commanded whenever the operator "
            "commits any non-zero forward stick deflection in the "
            "continuous-locomotion path. Default %.3f m/s lands the SONIC "
            "X2 root model inside its in-distribution forward-walk band as "
            "soon as the operator pushes past the deadzone -- empirically "
            "the corpus has essentially no training samples below 0.3 m/s "
            "forward, so commanding 0 < vel_z < 0.3 produces hip-wiggle "
            "without stepping. Set 0.0 to disable the floor entirely "
            "(pre-2026-05-31 legacy behaviour). Backward / lateral / yaw "
            "are unaffected; bucketed forward intents and PKL replay are "
            "also untouched. Applied post --forward-scale, so keep "
            "``floor < forward_scale * 0.5 m/s`` to preserve progressive "
            "control above the floor."
        ) % _DEFAULT_CONTINUOUS_FORWARD_MIN_MPS,
    )
    p.add_argument(
        "--cold-start-ramp-tau-s",
        type=float,
        default=_DEFAULT_COLD_START_RAMP_TAU_S,
        help=(
            "Time constant (s) for the cold-start velocity ramp applied on "
            "every idle -> playing transition. Smooths (yaw_rate, vel_x, "
            "vel_z) via per-channel EWMA so the model's implied 2.13 s target "
            "doesn't jump 1 m+ ahead while the context buffer still holds 4 "
            "frames of static stand pose. Default 0.20 s ~= 95%% of operator "
            "target after ~3 replans at threshold=2. Set 0.0 to disable the "
            "ramp (raw intent verbatim; pre-fix behaviour). hip_h is never "
            "ramped -- it's a posture target, not a velocity."
        ),
    )
    p.add_argument(
        "--command-watchdog-s",
        type=float,
        default=0.0,
        help=(
            "Opt-in safety watchdog: when the upstream intent source "
            "stops publishing for this many seconds, the kplanner forces "
            "its internal intent state back to _IDLE_INTENT so the robot "
            "does not keep walking on a stale non-idle target. "
            "DISABLED BY DEFAULT (0.0) because the Quest 3 manager's "
            "IntentDecoder is event-driven: it emits a new planner_cmd "
            "only when the stick value changes, so a held stick produces "
            "inter-arrival gaps of multiple seconds during NORMAL "
            "operation. A low watchdog threshold here would abort "
            "locomotion every time the operator holds the stick steady. "
            "Upstream-silence detection belongs in the manager (see "
            "quest3_manager_x2.py --vr-input-max-age-s), which has "
            "access to the raw WebSocket liveness signal. Enable this "
            "knob only for upstream sources that DO emit a steady "
            "heartbeat at a known rate -- e.g. x2_pkl_command_source "
            "replaying recorded clips, or scripted demos that publish "
            "every tick. Set to ~3x the expected heartbeat period."
        ),
    )
    p.add_argument(
        "--ref-smoother-ms",
        type=float,
        default=_DEFAULT_REF_SMOOTHER_MS,
        help=(
            "Duration (ms) of the half-cosine ramp the publisher applies "
            "whenever it detects a step in the lower-body joint reference "
            "(legs + waist by default). Default %.0f ms ~= natural period "
            "of the leg PD loop at deployed kp; the joint follows the "
            "rising reference instead of being thrown by a 1-tick step. "
            "Eliminates the audible motor click on stick push and release "
            "(operator-reported on real hardware 2026-05-31; click is "
            "drivetrain backlash absorbing the torque step that high "
            "lower-body kp turns the reference step into). Set 0 to "
            "disable the smoother entirely (passthrough; pre-fix "
            "behaviour). Tune down to 150-200 ms if the ramp feels "
            "sluggish; up to 500 ms if residual clicks remain. The "
            "smoother is step-detection-driven, not FSM-driven -- it "
            "fires on any source of reference step, not just IDLE<->PLAYING."
        ) % _DEFAULT_REF_SMOOTHER_MS,
    )
    p.add_argument(
        "--ref-smoother-trigger-rad",
        type=float,
        default=_DEFAULT_REF_SMOOTHER_TRIGGER_RAD,
        help=(
            "Per-tick reference delta (rad) on any lower-body joint that "
            "arms the smoother's half-cosine ramp. Default %.3f rad ~= "
            "3 deg, cleanly separating the multi-degree jumps a stick "
            "push/release creates (frozen anchor <-> neural buffer pose) "
            "from the per-tick neural-buffer motion under steady walking "
            "(typ. < 0.01 rad/tick). Set 0 to make every non-zero delta "
            "a candidate (debug only; ramps will fire continuously during "
            "smooth walking)."
        ) % _DEFAULT_REF_SMOOTHER_TRIGGER_RAD,
    )
    p.add_argument(
        "--ref-smoother-shape",
        choices=list(_REF_SMOOTHER_SHAPES),
        default=_DEFAULT_REF_SMOOTHER_SHAPE,
        help=(
            "Ramp shape for the reference-step smoother. Default 'off' now "
            "that the 30->50 Hz output resampling + 8-frame cross-fade blend "
            "handles replan-seam discontinuities at the source. 'halfcos' "
            "(opt-in) is C^1 smooth at both endpoints -- no velocity step is "
            "injected into the PD law at the start OR end of the ramp; this "
            "is what the piano shoulder-click work used. 'linear' has a "
            "dq/dt step at the endpoints (worse for the click) and is "
            "exposed for A/B comparison. 'off' short-circuits the smoother "
            "to pure passthrough."
        ),
    )
    p.add_argument(
        "--ref-smoother-joints",
        choices=list(_REF_SMOOTHER_JOINTS_PRESETS),
        default=_DEFAULT_REF_SMOOTHER_JOINTS,
        help=(
            "Which MuJoCo joint indices participate in step detection AND "
            "blending. 'lower_body' (default) = legs 0-11 + waist 12-14; "
            "arms 15-28 and head 29-30 pass through unchanged so "
            "manipulation tasks driven by the same body_pose stream are "
            "not affected by any ramp. 'legs_only' = legs 0-11 only "
            "(waist twist remains maximally responsive). 'all' = all 31 "
            "DoFs (debug A/B against the legacy formulation). Fingers / "
            "OmniHand commands live on a separate command path and are "
            "never in this stream, so they are unconditionally untouched "
            "by every preset."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(
        vqvae_ckpt=args.vqvae_ckpt,
        pose_ckpt=args.pose_ckpt,
        root_ckpt=args.root_ckpt,
        warmup_qpos_path=args.warmup_qpos_path,
        pub_host=args.pub_host,
        pub_port=args.pub_port,
        pid_file=args.pid_file,
        demo_yaml=args.demo,
        enable_keyboard=args.keyboard,
        zmq_cmd_host=args.zmq_cmd_host,
        zmq_cmd_port=args.zmq_cmd_port,
        zmq_cmd_topic=args.zmq_cmd_topic,
        zmq_cmd_bind=args.zmq_cmd_bind,
        duration_s=args.duration_s,
        hand_dof=args.hand_dof,
        verbose=args.verbose,
        warmup_quiet_stand_s=args.warmup_quiet_stand_s,
        body_pose_port=args.body_pose_port,
        device=args.device,
        replan_threshold_frames=args.replan_threshold_frames,
        planner_mode=args.planner_mode,
        yaw_lock_epsilon_rad_s=args.yaw_lock_epsilon,
        turn_left_scale=args.turn_left_scale,
        turn_right_scale=args.turn_right_scale,
        forward_scale=args.forward_scale,
        backward_scale=args.backward_scale,
        lateral_scale=args.lateral_scale,
        stick_shape_exp=args.stick_shape_exp,
        pose_feedback_host=args.pose_feedback_host,
        pose_feedback_port=args.pose_feedback_port,
        pose_feedback_topic=args.pose_feedback_topic,
        pose_feedback_max_age_s=args.pose_feedback_max_age_s,
        pose_feedback_deque_maxlen=args.pose_feedback_deque_maxlen,
        pose_reseed_scope=args.pose_reseed_scope,
        cold_start_ramp_tau_s=args.cold_start_ramp_tau_s,
        continuous_turn_max_rad_s=args.continuous_turn_max_rad_s,
        continuous_forward_min_mps=args.continuous_forward_min_mps,
        ref_smoother_ms=args.ref_smoother_ms,
        ref_smoother_trigger_rad=args.ref_smoother_trigger_rad,
        ref_smoother_shape=args.ref_smoother_shape,
        ref_smoother_joints=args.ref_smoother_joints,
        command_watchdog_s=args.command_watchdog_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
