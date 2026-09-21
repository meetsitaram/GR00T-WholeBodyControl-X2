"""X2-specific Quest 3 manager — X2 counterpart of the G1 Quest manager.

This is the manager process for the Phase 0 planner-driven recorder
architecture. It owns the Quest 3 connection, runs the X2 retargeting
stack (calibrated arm IK + per-finger curl + finger filter + thumb
opposition), and publishes three ZMQ streams:

1. ``planner_cmd``  (JSON)    -> the planner (``pc2_kplanner_onnx.py`` / ``x2_kplanner.py``)
2. ``arm_targets`` + ``hand_finger_cmd`` + ``stream_mode``  (msgpack)
                              -> ``record_x2_dataset.py`` (subscribe-only mode)
3. ``recorder_cmd`` (JSON)    -> ``record_x2_dataset.py`` (episode control)

In Phase 0 the recorder also subscribes to the planner's ``body_pose``
topic and merges ``body_pose`` + ``arm_targets`` -> ``final_pose`` to
the deploy. This manager is the single source of truth for Quest 3
inputs across that pipeline; the recorder no longer touches the
headset.

See ``planner_driven_quest3_recorder_mvp_*.plan.md`` for the full
data-flow diagram.

Wire format
-----------

planner_cmd (ZMQ PUB, multipart):
    frame 0: ``b"planner_cmd"``
    frame 1: utf-8 JSON ``{"intent": str, "magnitude": str}``

arm_targets (ZMQ PUB, msgpack on topic ``arm_targets``):
    ``{"left_q_rad": [7], "right_q_rad": [7], "is_engaged": bool, "tick": int, "ts": float}``

hand_finger_cmd (ZMQ PUB, msgpack on topic ``hand_finger_cmd``):
    ``{"left_hand_q": [10], "right_hand_q": [10], "tick": int, "ts": float}``

stream_mode (ZMQ PUB, msgpack on topic ``stream_mode``):
    ``{"mode": str, "tick": int, "ts": float}``

recorder_cmd (ZMQ PUB, multipart):
    frame 0: ``b"recorder_cmd"``
    frame 1: utf-8 JSON ``{"action": str, "tick": int, "ts": float}``

Buttons (X2 manager-mediated mode)
----------------------------------

- ``A+B+X+Y`` chord: toggle OFF <-> LOCOMOTION (engage / E-stop)
- ``B`` single (only when not OFF): toggle LOCOMOTION <-> ARM_MANIPULATION
- ``A`` single (in ARM_MANIPULATION): toggle arm IK engaged
- ``X`` single (in ARM_MANIPULATION): start episode
  (forwarded to recorder; no-op when not in ARM_MANIPULATION)
- ``Y`` single (in ARM_MANIPULATION): stop & save episode
  (forwarded to recorder; no-op when not in ARM_MANIPULATION)
- Left stick (LOCOMOTION mode): ``fwd_step`` / ``back_step`` / ``side_*``
- Right stick (LOCOMOTION mode): ``turn_left`` / ``turn_right`` (deg_45)
- ``A`` held (LOCOMOTION mode): converts L-stick fwd/back into
  continuous ``walk / forward`` / ``walk / backward``
- ``X`` held (LOCOMOTION mode): upgrades hard R-stick rx into a
  90deg turn (``turn_left / deg_90`` / ``turn_right / deg_90``)
- Y held: was crouch; currently disabled in IntentDecoder

Note: the OmniHand fingers are driven from XRHand curls + thumb
opposition; per-side state freezes on a mode flip into LOCOMOTION so
the operator's relaxed grip during walking doesn't open the hand
mid-grasp.

Sidecar logging
---------------

Per-tick ``planner_cmd`` events are appended to a JSONL file (one line
per emit) so post-hoc analysis can correlate operator intent with the
recorded ``observation.*`` / ``action.*`` columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402
from gear_sonic.utils.teleop.estop_gesture import EstopGesture  # noqa: E402
from gear_sonic.utils.teleop.finger_signal_filter import FingerFilterParams  # noqa: E402
from gear_sonic.utils.teleop.operator_calibration import OperatorCalibration  # noqa: E402
from gear_sonic.utils.teleop.vr.button_state_machine import (  # noqa: E402
    ButtonEvents,
    ButtonStateMachine,
)
from gear_sonic.utils.teleop.vr.intent_decoder import (  # noqa: E402
    IntentDecoder,
    LocomotionCmd,
    ModeTransition,
    StreamMode,
)
from gear_sonic.utils.teleop.vr.quest3_reader import Quest3Reader  # noqa: E402
from gear_sonic.utils.teleop.vr.stick_smoother import (  # noqa: E402
    StickFilter,
    StickFilterConfig,
)
from gear_sonic.utils.teleop.x2_retarget_pipeline import (  # noqa: E402
    Retargeter,
    RetargetTickInput,
)
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)


log = logging.getLogger("quest3_manager_x2")


def _make_vr_reader(cfg: "ManagerConfig"):
    """Construct the VR input reader for ``cfg.input_source``.

    Both readers expose the same accessor surface (the de-facto device
    interface: ``get_3pt_pose`` and friends); everything downstream of this
    factory is device-agnostic. Kept as a module-level function so tests can
    exercise source selection without spinning up sockets or the xrt SDK.
    """
    if cfg.input_source == "pico":
        # Imported lazily: xrobotoolkit_sdk only exists in .venv_teleop.
        from gear_sonic.utils.teleop.vr.pico_x2_reader import (
            PicoX2Reader,
        )

        return PicoX2Reader()
    if cfg.input_source == "quest3":
        return Quest3Reader(
            ws_port=cfg.quest3_ws_port,
            http_port=cfg.quest3_http_port,
            use_ssl=cfg.quest3_use_ssl,
        )
    raise ValueError(
        f"unknown input_source {cfg.input_source!r} (expected 'quest3' or 'pico')"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ManagerConfig:
    # VR input source. "quest3" (default) spawns the WebXR WebSocket/HTTPS
    # servers and waits for the headset browser app; "pico" polls the
    # XRoboToolkit PC Service via xrobotoolkit_sdk (PicoX2Reader) — no
    # sockets, no headset app, but the PC Service daemon must be running
    # and the Pico's XRoboToolkit client connected to this machine.
    input_source: str = "quest3"

    # Quest 3 (ignored when input_source != "quest3")
    quest3_ws_port: int = 8765
    quest3_http_port: int = 8443
    quest3_use_ssl: bool = True

    # Calibration (default path resolves at construction-time so unit
    # tests can override REPO_ROOT without re-importing this module).
    calibration_path: Path = field(default_factory=lambda: _default_calibration_path())

    # Tick rate (Hz). The planner runs at 50 Hz; we match.
    publish_rate_hz: float = 50.0

    # Quest 3 input staleness gate. The Quest 3 browser app stops
    # streaming WebSocket packets when the headset goes to sleep, the
    # tab moves to the background, or the network drops. ``Quest3Reader``
    # would otherwise keep returning the *last* (lx, ly, rx, ry) for the
    # rest of the session, and the manager would happily decode that
    # into a locomotion command and publish it forever -- the robot
    # keeps walking on a stale stick value. When the most recent WS
    # packet is older than this threshold, the manager:
    #   - Forces the decoder's stick / button inputs to neutral so any
    #     emitted command collapses to idle.
    #   - Logs a throttled WARNING with the current age.
    # In addition, x2_kplanner.py has its own ``--command-watchdog-s``
    # that snaps its internal IntentState to _IDLE_INTENT when no
    # planner_cmd has arrived for the same duration; the two combine
    # into defence-in-depth.
    vr_input_max_age_s: float = 0.5

    # PUB sockets
    planner_cmd_host: str = "*"
    planner_cmd_port: int = 5563
    planner_cmd_topic: str = "planner_cmd"
    # Dual-source mode: PUB-connect to a kplanner that owns the SUB bind
    # (--cmd-bind / --zmq-cmd-bind), so the pad bridge can publish too.
    planner_cmd_connect: bool = False
    # Hop-in manipulation: "host:port" of the robot planner's arm-ingest
    # SUB (pc2_kplanner_onnx --arm-port, default 5566). Empty = classic
    # bind (tethered recorder topology).
    arm_connect: str = ""

    recorder_pub_host: str = "*"
    recorder_pub_port: int = 5564

    # Topic names on the recorder PUB socket
    arm_targets_topic: str = "arm_targets"
    hand_finger_cmd_topic: str = "hand_finger_cmd"
    stream_mode_topic: str = "stream_mode"
    recorder_cmd_topic: str = "recorder_cmd"
    # SMPL path (pico only): raw 24-joint body skeleton republished on the
    # same recorder PUB. Subscribers that don't SUB the topic never see it.
    smpl_body_topic: str = "smpl_body"

    # --- Split-topology SAFE_IDLE resume chord (Phase 3a-b) ----------------
    # PUB socket the deploy on PC2 subscribes to. The manager publishes a
    # single multipart message [topic, ts_ns_le_i64] whenever the operator
    # holds A+B (right controller) for >= resume_chord_hold_s seconds with
    # X+Y not pressed. The chord is recognised across ALL modes because it
    # operates at the safety layer (operator -> deploy "you can come back"),
    # not the policy layer.
    #
    # Defaults reflect "split topology by default": bind on 0.0.0.0:5566
    # but DON'T enable the PUB unless explicitly requested via CLI flag.
    # The wrapper (run_x2_quest3_planner_stack.sh --remote-deploy) sets the
    # flag automatically; manual launches that want it must pass
    # ``--resume-pub-enabled``.
    resume_pub_enabled: bool = False
    resume_pub_host: str = "*"        # bind address (0.0.0.0 in PUB land)
    resume_pub_port: int = 5566
    resume_pub_topic: str = "pose_resume"
    resume_chord_hold_s: float = 1.0  # hold-duration before first publish
    resume_chord_rep_s:  float = 0.5  # min republish interval during sustained hold

    # --- Motor monitor SUB (Phase 5c) -------------------------------------
    # SUB socket connecting to the x2_motor_monitor daemon running in a
    # tmux session on PC2 (port 5567). Each received message is JSON-
    # decoded and appended to ``sidecar_log_path`` (the same JSONL the
    # ``_sidecar_emit`` planner_cmd writer uses) under the key
    # ``motor_monitor`` so a single grep across the file surfaces both
    # operator intents and motor-side events on the same timeline.
    #
    # Disabled by default; the wrapper flips it on for --remote-deploy
    # runs with --pc2-host set. No WebXR UX, no terminal tail -- log-only
    # sink (matches the user's "minimal" preference).
    motor_monitor_sub_enabled: bool = False
    motor_monitor_sub_host: str = ""
    motor_monitor_sub_port: int = 5567
    motor_monitor_sub_topic: str = "motor_monitor"

    # --- Engage-pose SUB (2026-06-10 follow-up 10) -----------------------
    # When ``preserve_arms_on_engage`` is True, the manager spawns a SUB
    # that subscribes to the wire driving the deploy (the pose-proxy's
    # downstream PUB -- ``tcp://127.0.0.1:5558`` in SIM-on-PC1, or
    # ``tcp://<PC2_IP>:5558`` for split topology). A background thread
    # caches the latest decoded ``joint_pos_mj`` + ``left_hand_joints``
    # + ``right_hand_joints`` at ~50 Hz with negligible CPU cost (one
    # 1280-byte JSON header parse + a few np.frombuffer calls per
    # frame).
    #
    # On every OFF -> non-OFF transition, the snap-to-neutral default is
    # REPLACED by a snap-to-wire-pose: the manager extracts the arm
    # AND hand slices from the cached jpos and uses those as the new
    # ``_frozen_*_arm_q`` / ``_frozen_*_hand_q`` so the very next
    # ``arm_targets`` + ``hand_finger_cmd`` publish carries the
    # robot's CURRENT commanded arms AND hands. The recorder then
    # merges those into its ``pose`` PUB, the proxy engages override,
    # and the deploy sees a continuous wire from VLA's last pose to
    # the operator's first override pose (= the same VLA pose, by
    # construction). No jump on arms, no surprise drop / squeeze on
    # hands.
    #
    # The port + host + topic + freshness window are settable but
    # default to "the proxy's downstream on loopback" so the common
    # SIM-on-PC1 flow only needs ``--preserve-arms-on-engage``.
    #
    # ``engage_pose_sub_max_age_ms`` caps how stale a cached pose
    # can be before we fall back to the neutral snap. 200 ms (10
    # ticks at 50 Hz) tolerates a brief wire stall without
    # silently freezing arms at the prior session's last pose.
    preserve_arms_on_engage: bool = False
    engage_pose_sub_host: str = "127.0.0.1"
    engage_pose_sub_port: int = 5558
    engage_pose_sub_topic: str = "pose"
    engage_pose_sub_max_age_ms: int = 200

    # IntentDecoder
    intent_stick_deadzone: float = 0.30
    intent_repeat_interval_s: float = 0.0
    # Both default to False because the curated bins are *replay*
    # primitives that snap the body back to standing instead of
    # holding the static pose -- pushing the right stick fwd or
    # softly L/R looks like a flicker to the operator and isn't
    # useful for live teleop yet. The right-stick X-axis hard
    # deflection (turn_*) is unaffected. Re-enable here (or via
    # --enable-lean-fwd / --enable-torso) once the planner can
    # hold the static lean / torso pose without snapping back.
    intent_enable_lean_fwd: bool = False
    intent_enable_torso: bool = False
    # Continuous waist hold via the right stick: pitch (ry > 0), yaw
    # (rx). The roll axis was retired in v7.2 (the operator's right
    # thumb owns the R-stick, leaving A on the same controller
    # unreachable mid-lean -- so the legacy "A held + rx -> roll"
    # modifier was a footgun rather than a feature). Default True
    # because this is the primary VR teleop surface for static reach
    # now that the planner has STATIC_HOLD wired up. Set False to
    # fall back to the legacy
    # discrete soft-band torso bins.
    intent_enable_continuous_torso: bool = True
    # ``intent_enable_continuous_locomotion`` (default OFF) flips the
    # L/R-stick locomotion path from the bucketed
    # (``fwd_step / back_step / side_* / turn_*``) emit to a single
    # ``locomotion / continuous`` command carrying raw deadzone-rescaled
    # stick deflections in ``stick_fwd / stick_side / stick_yaw``. The
    # kplanner shapes those into a velocity vector for analog control;
    # the heuristic planner cannot consume the intent and treats it as
    # idle. ``run_x2_quest3_planner_stack.sh`` flips it ON automatically
    # when ``--planner kplanner`` is selected.
    intent_enable_continuous_locomotion: bool = False
    # Maximum R-stick X amplitude forwarded as ``stick_yaw`` to the
    # planner. Defaults to 0.5 so full-stick deflection commands HALF
    # the planner's continuous-mode yaw-rate ceiling; this empirically
    # matches what the current X2 root-model checkpoint can track
    # without overshoot. Lives on the teleop side (operator-feel
    # concern) rather than the planner (which only sees the post-
    # clamp value and treats it like any other intent). Set to 1.0
    # to restore the legacy "full stick = full planner ceiling"
    # mapping for A/B comparison.
    intent_continuous_yaw_max: float = 0.5
    # ARM_MANIPULATION L-stick mapping (v7.4). When True (default),
    # the left thumbstick decodes as roll (lx) + continuous hip
    # height (ly) inside ARM_MAN mode -- the analog "squat / stand"
    # axis the kplanner consumes via the hip_height_m wire field.
    # LOCOMOTION mode is unaffected (L-stick still owns step / side
    # / continuous-walk). Set False for ablation runs that want
    # ARM_MAN to behave exactly as in v7.3 (only R-stick lean / twist).
    intent_enable_arm_man_lstick: bool = True
    # Yaw-priority cone (R-stick): pitch is suppressed when |ry| <
    # ratio * |rx|. Default 0.4 lets the operator twist while leaning
    # slightly but blocks accidental lean from a yaw-intent stick
    # wobble. 0.0 disables the cone (every axis past the deadzone
    # fires independently); 1.0 is strict dominance.
    intent_pitch_dominance_ratio: float = 0.4
    # Roll-priority cone (L-stick, ARM_MAN): height is suppressed
    # when |ly| < ratio * |lx|. Same range / semantics as the yaw
    # priority cone above; 0.4 default.
    intent_height_dominance_ratio: float = 0.4
    # Continuous hip-height caps (metres, OFFSETS around the
    # kplanner's default hip height of 0.687 m). Asymmetric defaults
    # because the kplanner training distribution covers crouch-down
    # ~9 cm before going OOD but only stand-up ~4 cm. Operators
    # wanting a deeper / shallower envelope can override these on
    # the CLI; the kplanner-side overlay still respects the model's
    # in-distribution band so pushing past it gives degraded
    # tracking, not catastrophic failure.
    intent_max_height_down_m: float = 0.09
    intent_max_height_up_m: float = 0.04
    # Wire-side hysteresis for the hip-height channel (metres). The
    # decoder treats two ``hold_torso`` commands as identical when
    # their hip_height_m targets are within this many metres on the
    # current emit; below that thumbstick noise on ly is suppressed
    # at the publish layer. The kplanner's neural model has its own
    # smoothing so coarse waypoints are fine.
    intent_hold_height_threshold_m: float = 0.005
    # StickFilter config for taming raw VR stick step-inputs into a
    # band that matches the kplanner's training distribution.
    #
    # The kplanner + SONIC pair was trained on smooth human mocap; raw
    # Quest 3 thumbstick deflections are step inputs (operator's thumb
    # snaps to full deflection in 1-2 ticks of the 50 Hz loop). With
    # no filtering, the VR p99 |d(vel_z)/dt| is ~8 m/s^2 -- 2-3x the
    # ~3 m/s^2 carried by the curated PKL primitive subset. The robot
    # is stable when idle but lurches forward on every step input
    # because the model is asked to track an acceleration profile
    # outside its training distribution.
    #
    # Apply order in ``run()``:
    #   1. Read raw axes from the WebXR reader (post invert_lx/ly/rx/ry).
    #   2. If mode == LOCOMOTION and filter is set, pass (ly, lx, rx)
    #      as (stick_fwd, stick_side, stick_yaw) through the filter
    #      using the per-tick dt. The filter applies a slew clamp
    #      then a first-order LPF per channel.
    #   3. Pass the filtered axes to ``IntentDecoder.decode_locomotion``
    #      which deadzones + scales them and emits LocomotionCmd.
    #
    # Filter state is reset on mode transitions (LOCOMOTION <->
    # ARM_MAN, OFF, cold start) so the first tick after a fresh
    # LOCOMOTION engage is a pass-through. The decoder itself is
    # stateless and does not see the filter.
    #
    # ``None`` -> no filter, raw axes pass through unchanged (legacy
    # behaviour, kept as the safe default until the tuned defaults
    # are validated on the robot).
    stick_filter_config: Optional[StickFilterConfig] = None
    # ``intent_loco_decoupled_arms`` (default True) controls whether
    # the manager signals the recorder to *override* the kplanner's
    # predicted arm joints with the manager's frozen / IK arm pose
    # during LOCOMOTION mode.
    #
    # TRUE (default) -- current behaviour. Manager publishes
    # ``arm_targets`` with ``passthrough_arm_targets=False`` every
    # tick; the recorder applies the operator's last arm pose as a
    # hard override on the merged ``pose`` stream the deploy
    # consumes. This is intentional for the ARM_MANIPULATION ->
    # LOCOMOTION arm-hold workflow: the operator positions the arms
    # (e.g. holding a tool) in ARM_MAN, toggles to LOCOMOTION to walk
    # to a new spot, and the arms STAY LOCKED at the manipulation
    # pose for the duration of the walk instead of swinging back to a
    # neutral gait pose. Flipping this default to False would silently
    # break that workflow -- the moment the operator switches from
    # ARM_MAN to LOCOMOTION, the held pose would dissolve into the
    # planner's gait swing. Future contributors: do not flip the
    # default without coordinating with manipulation operators.
    #
    # FALSE -- opt-in for *whole-body locomotion* sessions (just
    # walking around, no payload to hold) and for diagnosing whether
    # the static-arms override hurts forward-walking quality. When
    # ``self._intent.mode is StreamMode.LOCOMOTION`` the manager
    # flips ``passthrough_arm_targets=True`` in the payload, the
    # recorder nulls its cached arm pose, and the existing
    # validity gate in the merge step falls through to the planner-
    # predicted arms (which carry natural gait-coupled swing from the
    # ``x2_ultra_locowalk`` training corpus). ARM_MAN and OFF still
    # publish ``passthrough_arm_targets=False`` -- the sentinel is
    # LOCOMOTION-only so other modes keep today's safety behaviour.
    #
    # Wired from CLI ``--loco-decoupled-arms`` /
    # ``--no-loco-decoupled-arms`` and wrapper env var
    # ``LOCO_DECOUPLED_ARMS`` (1/0).
    intent_loco_decoupled_arms: bool = True
    # Per-axis sign flips applied BEFORE the decoder sees the sticks.
    #
    # Operator UX contract: pushing the left stick AWAY from your body
    # must walk the robot forward in world; pulling it toward you must
    # walk the robot backward in world.
    #
    # Two layers shape the polarity:
    #   1. Hardware: Quest 3 controllers report ly < 0 when the stick
    #      is pushed forward (away from operator). The IntentDecoder
    #      is trained on the WebXR convention "+ly = forward push" and
    #      emits ``fwd_step`` on +ly, ``back_step`` on -ly.
    #   2. Bin world frame: the curated planner bins were authored in
    #      a body frame that is rotated 180 deg from the bridge's
    #      RSI init orientation, so the bin labelled ``fwd_step``
    #      actually translates the body BACKWARD in world (legs step
    #      forward in body-local frame, but the body's facing is
    #      reversed). Conversely the ``back_step_half_ft`` bin
    #      translates the body FORWARD in world.
    #
    # Net: leaving ``invert_ly = False`` lets the raw -ly from a
    # forward stick push fall through the decoder as ``back_step``,
    # which (after the world inversion above) actually moves the body
    # forward in world. That gives the operator the right end-to-end
    # behaviour without rebaking the bins or rotating the RSI anchor.
    # Operators who later fix the bin world frame (or use a different
    # RSI source) can restore the literal "+ly emits fwd_step" mapping
    # by passing ``--invert-ly``.
    invert_lx: bool = False
    invert_ly: bool = False
    invert_rx: bool = False
    invert_ry: bool = False

    # Retargeter
    ik_damping: float = 0.08
    ik_rotation_weight: float = 0.3
    ik_per_tick_step_rad: float = 0.30
    hand_input_mode: str = "trigger"
    apply_curl_compensation: bool = False
    apply_oppose_compensation: bool = False
    enable_finger_filter: bool = True

    # VR wrist quat offset (operator-side, applied in head-yaw frame
    # BEFORE the calibration alignment). Stop-gap "controller mount
    # calibration" knob: when one of the controllers is mounted at a
    # fixed rotation on the operator's wrist (e.g. left controller cuff
    # twisted ~30deg outward), this rotates the reported op-wrist quat
    # by the inverse of the mount alignment so the calibration sees a
    # corrected operator quat. Re-run ``vr_operator_calibrate.py`` to
    # drop these back to zero. Tuple is ``(roll, pitch, yaw)`` in
    # degrees, intrinsic XYZ Tait-Bryan; see ``VRArmTeleopCalibrated``
    # docstring for the axis convention. Defaults to no offset on
    # either side -- existing setups stay bit-exact.
    #
    # ``None`` means "use whatever the calibration YAML carries" (see
    # ``VRArmTeleopCalibrated`` constructor for the three-tier resolution
    # order). Passing an explicit tuple here -- including ``(0,0,0)`` to
    # force identity -- *overrides* the YAML value.
    left_wrist_op_quat_offset_rpy_deg: Optional[tuple[float, float, float]] = None
    right_wrist_op_quat_offset_rpy_deg: Optional[tuple[float, float, float]] = None

    # Sidecar
    sidecar_log_path: Optional[Path] = None
    """If set, append one JSONL line per emitted ``planner_cmd`` to this
    file. Useful for post-hoc analysis (which intent fired when)."""

    # Quest3 raw capture (input-smoothing tuning fixture)
    quest3_raw_log_path: Optional[Path] = None
    """If set, append one JSONL line per 50 Hz manager tick to this file,
    recording the raw Quest 3 inputs the manager loop actually consumed
    (post-invert sticks + buttons + 3pt-pose + reader timestamps). Used as
    a replayable fixture by ``Quest3Replayer`` so we can sweep
    ``StickFilter`` knobs headset-free against an identical input stream.

    Captures ONLY ticks where the reader has a sample (i.e. after the
    first WS packet has landed); waiting-for-Quest ticks are skipped so
    the file's row count equals the number of operator-driven ticks.
    Default off; the live recording session sets it via ``--quest3-record-to``."""

    # Episode lifecycle audio cues
    recorder_enabled: bool = False
    """When True, the manager plays the ``record_start`` / ``record_save``
    headset audio cues on X / Y press in ARM_MANIPULATION. When False
    (the default, matching ``--teleop-only`` recorder mode), those
    cues are suppressed so the operator doesn't get a false "Recording."
    / "Saved." ACK while no parquet is being written.

    The ``recorder_cmd`` ZMQ message is **always** published regardless
    of this flag: the recorder is the source of truth for whether the
    save actually landed and logs ``[recorder] [Y] ignored: ...`` when
    it can't honour the request. This flag only gates the *audio* path,
    not the wire path, so a future ACK-driven cue (recorder PUBs
    ``recorder_ack`` -> manager waits before playing) can land
    incrementally without breaking existing teleop-only sessions.

    The wrapper (``run_x2_quest3_planner_stack.sh``) sets this to True
    iff ``--with-record`` was passed; manual launches must set it
    explicitly with ``--recorder-enabled``. Note: this flag still
    leaves a known footgun -- in ``--with-record`` mode an X press
    while already recording, or a Y press with no active episode,
    will fire the cue (the recorder logs ``ignored`` but the manager
    doesn't see that). Tracked as the "ACK topic" follow-up."""

    # Camera cycler (left-stick-click -> ']' to deploy MuJoCo viewer)
    enable_viewer_camera_cycler: bool = True
    """When True (default), pressing the LEFT thumbstick click cycles
    the deploy MuJoCo viewer's fixed cameras via a synthesised ``]``
    keystroke (xdotool keysym ``bracketright``; that's the next-fixed-
    camera key in mujoco.viewer.launch_passive). Set to False on
    headless / CI runs where no GLFW window exists. Pre-v7.1 the
    binding was on the right click; pre-deploy-test we briefly used
    ``Tab`` here, but Tab only toggles the viewer's left UI panel and
    doesn't change cameras.

    TODO(unified-vr-input-topic): replace this entire xdotool path with
    a proper ZMQ ``vr_input`` topic that the manager publishes and any
    interested process (deploy viewer, recorder, future tools) can
    subscribe to. See ViewerCameraCycler.__doc__ for rationale."""
    viewer_window_pattern: str = "MuJoCo"
    """``xdotool search --name`` pattern used as a *fallback* to find
    the deploy viewer window. The cycler tries the more precise
    ``--classname`` search first (see ``viewer_window_classname``);
    this title-substring path only fires when the classname search
    returns nothing. Override if you have multiple MuJoCo windows
    open or if your deploy build sets a custom title (rare)."""
    viewer_window_classname: str = "MuJoCo"
    """``xdotool search --classname`` pattern -- the primary, more
    precise way to find the deploy viewer window. WM_CLASS is set
    by GLFW on the application window itself and is NOT inherited
    by the GNOME / mutter compositor's frame wrapper, so this filter
    avoids the "synthetic key vanishes into mutter-x11-frames"
    failure mode we hit before 2026-05-13. Only override if you've rebuilt
    MuJoCo with a custom WM_CLASS string (very rare)."""

    # Misc
    verbose: bool = False


def _default_calibration_path() -> Path:
    """Default to the repo's standard operator_calibrations path."""
    return REPO_ROOT / "data" / "operator_calibrations" / "default.yaml"


def _neutral_calibration_path() -> Path:
    """Shipped neutral (identity-fit) calibration used when no operator
    calibration exists yet. Same fallback order as the launcher
    (``run_x2_quest3_planner_stack.sh``)."""
    return REPO_ROOT / "gear_sonic" / "data" / "operator_calibrations" / "default.yaml"


# ---------------------------------------------------------------------------
# Helpers: ZMQ wire format
# ---------------------------------------------------------------------------


def _planner_cmd_payload(cmd: LocomotionCmd, release: bool = False) -> bytes:
    """Build the JSON payload the planner's _zmq_command_thread expects.

    For ``hold_torso`` commands we also serialize the continuous waist
    targets; the planner's ``_zmq_command_thread`` reads them as
    optional fields and feeds them into ``LocomotionCommand.waist_*_deg``.
    For ``locomotion`` commands (continuous L/R-stick teleop) we
    serialize the three stick deflections; the kplanner reads them as
    ``stick_fwd / stick_side / stick_yaw`` and shapes them into a 4-D
    velocity vector. For every other intent we omit both blocks
    (defaulting to 0.0 on the receiving end), which keeps wire payloads
    minimal and matches the pre-v7 wire format.
    """
    payload: dict[str, object] = {
        "intent": cmd.intent,
        "magnitude": cmd.magnitude,
        # Ownership tag: the planner's command SUB enforces single-source
        # control. Any unflagged "vr" message takes/holds ownership (a
        # stand-still idle included); ONLY the disengage idle carries
        # vr_release=True and hands control back to the pad.
        "source": "vr",
    }
    if release:
        payload["vr_release"] = True
    if cmd.intent == "hold_torso":
        payload["waist_pitch_deg"] = float(cmd.waist_pitch_deg)
        payload["waist_roll_deg"] = float(cmd.waist_roll_deg)
        payload["waist_yaw_deg"] = float(cmd.waist_yaw_deg)
        # ``hip_height_m`` is an optional channel-3 override for the
        # kplanner's 4-D velocity intent (squat / stand). Emit only
        # when the operator's L-stick Y is past the deadzone (decoder
        # leaves it ``None`` otherwise) so legacy hold_torso wire
        # payloads stay byte-identical and the kplanner default hip
        # height still applies for pure pitch / roll / yaw leans.
        if cmd.hip_height_m is not None:
            payload["hip_height_m"] = float(cmd.hip_height_m)
    elif cmd.intent == "locomotion":
        payload["stick_fwd"]  = float(cmd.stick_fwd)
        payload["stick_side"] = float(cmd.stick_side)
        payload["stick_yaw"]  = float(cmd.stick_yaw)
        # X/Y face-button speed-setpoint nudge (one-shot, +/-0.1 m/s);
        # consumed by the kplanner's runtime forward-speed setpoint.
        if getattr(cmd, "speed_delta", 0.0):
            payload["speed_delta"] = float(cmd.speed_delta)
    return json.dumps(payload).encode("utf-8")


def _recorder_cmd_payload(action: str, tick: int) -> bytes:
    return json.dumps({
        "action": action,
        "tick": tick,
        "ts": time.time(),
    }).encode("utf-8")


def _msgpack_payload(data: dict) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class Quest3ManagerX2:
    """X2-specific Quest 3 manager process.

    Runs a single 50 Hz loop. Owns:
    - :class:`Quest3Reader` (Quest 3 WebXR ingest)
    - :class:`IntentDecoder` (stick + button -> ``LocomotionCmd``)
    - :class:`Retargeter`    (VR + hand inputs -> arm + hand commands)
    - Two PUB sockets        (planner + recorder fan-out)
    """

    def __init__(self, cfg: ManagerConfig) -> None:
        self._cfg = cfg
        self._stop = threading.Event()

        self._calibration = self._resolve_calibration()
        self._retargeter = Retargeter(
            calibration=self._calibration,
            finger_filter_params=(
                FingerFilterParams() if cfg.enable_finger_filter else None
            ),
            ik_damping=cfg.ik_damping,
            ik_rotation_weight=cfg.ik_rotation_weight,
            ik_per_tick_step_rad=cfg.ik_per_tick_step_rad,
            hand_input_mode=cfg.hand_input_mode,
            apply_curl_compensation=cfg.apply_curl_compensation,
            apply_oppose_compensation=cfg.apply_oppose_compensation,
            left_wrist_op_quat_offset_rpy_deg=(
                tuple(cfg.left_wrist_op_quat_offset_rpy_deg)
                if cfg.left_wrist_op_quat_offset_rpy_deg is not None
                else None
            ),
            right_wrist_op_quat_offset_rpy_deg=(
                tuple(cfg.right_wrist_op_quat_offset_rpy_deg)
                if cfg.right_wrist_op_quat_offset_rpy_deg is not None
                else None
            ),
        )
        self._intent = IntentDecoder(
            stick_deadzone=cfg.intent_stick_deadzone,
            repeat_interval_s=cfg.intent_repeat_interval_s,
            enable_lean_fwd=cfg.intent_enable_lean_fwd,
            enable_torso=cfg.intent_enable_torso,
            enable_continuous_torso=cfg.intent_enable_continuous_torso,
            enable_continuous_locomotion=cfg.intent_enable_continuous_locomotion,
            continuous_yaw_max=cfg.intent_continuous_yaw_max,
            enable_arm_man_lstick=cfg.intent_enable_arm_man_lstick,
            pitch_dominance_ratio=cfg.intent_pitch_dominance_ratio,
            height_dominance_ratio=cfg.intent_height_dominance_ratio,
            max_height_down_m=cfg.intent_max_height_down_m,
            max_height_up_m=cfg.intent_max_height_up_m,
            hold_height_threshold_m=cfg.intent_hold_height_threshold_m,
        )

        # --- StickFilter (raw-axis smoother for continuous locomotion) ----
        # Filters the (ly, lx, rx) axes BEFORE the decoder sees them, but
        # only while in LOCOMOTION mode. The filter is reset on any
        # transition INTO LOCOMOTION so the first post-mode-flip tick is
        # a pass-through (no spurious slew-clamp ramp from a zero anchor).
        # ``None`` config preserves legacy behaviour exactly.
        if cfg.stick_filter_config is not None and not cfg.stick_filter_config.is_noop():
            self._stick_filter: Optional[StickFilter] = StickFilter(
                cfg.stick_filter_config
            )
            self._stick_filter_last_t: Optional[float] = None
            self._stick_filter_prev_in_loco: bool = False
            log.info(
                "[stick-filter] enabled: tau_fwd=%.3fs tau_side=%.3fs "
                "tau_yaw=%.3fs slew_fwd=%s slew_side=%s slew_yaw=%s "
                "(release_tau fwd/side/yaw = %s / %s / %s)",
                cfg.stick_filter_config.tau_lpf_fwd_s,
                cfg.stick_filter_config.tau_lpf_side_s,
                cfg.stick_filter_config.tau_lpf_yaw_s,
                cfg.stick_filter_config.slew_max_fwd_per_s,
                cfg.stick_filter_config.slew_max_side_per_s,
                cfg.stick_filter_config.slew_max_yaw_per_s,
                cfg.stick_filter_config.return_to_zero_tau_fwd_s,
                cfg.stick_filter_config.return_to_zero_tau_side_s,
                cfg.stick_filter_config.return_to_zero_tau_yaw_s,
            )
        else:
            self._stick_filter = None
            self._stick_filter_last_t = None
            self._stick_filter_prev_in_loco = False
        # Latched continuous waist target. Set in two situations:
        #   1) The operator presses B to flip LOCOMOTION ->
        #      ARM_MANIPULATION (existing behavior; ARM_MAN is implicitly
        #      a hold because the right stick is a no-op for waist there).
        #   2) The operator presses the right thumbstick CLICK while in
        #      LOCOMOTION (or ARM_MANIPULATION) to toggle ``_waist_frozen``
        #      ON; the live waist target is captured here so the planner
        #      can be re-pinned later if needed.
        # ``None`` means "no latch active" (live stick drives the waist).
        # Tuple layout: ``(pitch_deg, roll_deg, yaw_deg, hip_height_m)``;
        # the trailing element is None when the operator's L-stick Y is
        # neutral (== "no hip-height override; kplanner default
        # applies"). Pre-v7.4 this was a 3-tuple of angles only; the
        # height channel was added when ARM_MAN L-stick decoded as
        # squat / stand.
        self._latched_waist: (
            tuple[float, float, float, float | None] | None
        ) = None
        # R-thumbstick-click freeze toggle. Independent of mode: a press
        # in LOCOMOTION freezes the body so the operator can keep
        # leaning/twisting while walking with the L stick; the freeze
        # persists across B-press mode flips so the body stays leaned
        # through ARM_MANIPULATION and back. Toggled off by another
        # R-click. Reset on any transition to OFF.
        self._waist_frozen: bool = False
        self._button_sm = ButtonStateMachine(log_prefix="Input")
        self._estop_gesture = EstopGesture()

        # Stick-click rising-edge trackers. The WebXR client polls the
        # gamepad ~50 Hz so a click typically holds True for several
        # ticks. We only fire on the press transition (False -> True);
        # this mirrors the ``ButtonStateMachine`` debounce for the four
        # face buttons. Left click cycles deploy MuJoCo viewer cameras
        # (was on right click pre-v7); right click toggles waist freeze.
        self._prev_left_stick_click = False
        self._prev_right_stick_click = False

        # Camera cycler. Always constructed (even when xdotool isn't
        # installed) so the cycle() call path is exercised on every
        # press; the helper logs a one-shot warning and no-ops if its
        # prerequisites are missing. See ViewerCameraCycler docstring
        # for the planned vr_input-topic replacement.
        if cfg.enable_viewer_camera_cycler:
            from gear_sonic.utils.teleop.vr.viewer_camera_cycler import (
                ViewerCameraCycler,
            )
            self._viewer_cycler: Optional[ViewerCameraCycler] = (
                ViewerCameraCycler(
                    window_search_pattern=cfg.viewer_window_pattern,
                    window_class_name=cfg.viewer_window_classname,
                )
            )
        else:
            self._viewer_cycler = None

        self._quest = _make_vr_reader(cfg)

        self._ctx = zmq.Context.instance()
        self._planner_sock = self._ctx.socket(zmq.PUB)
        self._planner_sock.setsockopt(zmq.LINGER, 0)
        if cfg.planner_cmd_connect:
            _cmd_host = (cfg.planner_cmd_host
                         if cfg.planner_cmd_host not in ("*", "0.0.0.0")
                         else "127.0.0.1")
            self._planner_sock.connect(
                f"tcp://{_cmd_host}:{cfg.planner_cmd_port}"
            )
            log.info(
                "planner_cmd PUB connected to tcp://%s:%d (topic=%s; "
                "dual-source mode, kplanner owns the SUB bind)",
                _cmd_host, cfg.planner_cmd_port, cfg.planner_cmd_topic,
            )
        else:
            self._planner_sock.bind(
                f"tcp://{cfg.planner_cmd_host}:{cfg.planner_cmd_port}"
            )
            log.info(
                "planner_cmd PUB bound at tcp://%s:%d (topic=%s)",
                cfg.planner_cmd_host, cfg.planner_cmd_port, cfg.planner_cmd_topic,
            )
        self._recorder_sock = self._ctx.socket(zmq.PUB)
        self._recorder_sock.setsockopt(zmq.LINGER, 0)
        if cfg.arm_connect:
            # Hop-in manipulation: the ROBOT planner owns the arm-ingest
            # SUB bind (pc2_kplanner_onnx --arm-port); this PUB connects
            # out, mirroring --planner-cmd-connect. Same socket still
            # carries every recorder-wire topic; on the robot only
            # arm_targets + hand_finger_cmd have a subscriber.
            self._recorder_sock.connect(f"tcp://{cfg.arm_connect}")
            log.info(
                "recorder PUB connected to tcp://%s (hop-in arm targets; "
                "topics=%s, %s, %s, %s)",
                cfg.arm_connect,
                cfg.arm_targets_topic, cfg.hand_finger_cmd_topic,
                cfg.stream_mode_topic, cfg.recorder_cmd_topic,
            )
        else:
            self._recorder_sock.bind(
                f"tcp://{cfg.recorder_pub_host}:{cfg.recorder_pub_port}"
            )
            log.info(
                "recorder PUB bound at tcp://%s:%d (topics=%s, %s, %s, %s)",
                cfg.recorder_pub_host, cfg.recorder_pub_port,
                cfg.arm_targets_topic, cfg.hand_finger_cmd_topic,
                cfg.stream_mode_topic, cfg.recorder_cmd_topic,
            )

        # Last published arm + hand targets (used for freezing in
        # LOCOMOTION mode so the recorder gets a steady stream).
        self._frozen_left_arm_q = self._retargeter._teleop._left_q.copy()
        self._frozen_right_arm_q = self._retargeter._teleop._right_q.copy()
        self._frozen_left_hand_q = np.zeros(10, dtype=np.float64)
        self._frozen_right_hand_q = np.zeros(10, dtype=np.float64)

        # One-shot OFF-mode operator hint (re-armed on every successful
        # mode transition). Stops noisy spam every time the operator
        # presses a single button while still in OFF.
        self._off_mode_hint_logged = False

        if cfg.sidecar_log_path is not None:
            cfg.sidecar_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._sidecar = cfg.sidecar_log_path.open("a", buffering=1)
            self._sidecar_lock = threading.Lock()
            log.info("sidecar log -> %s", cfg.sidecar_log_path)
        else:
            self._sidecar = None
            self._sidecar_lock = threading.Lock()

        # --- Quest3 raw capture (input-smoothing tuning fixture) ----------
        # Independent file handle from the planner_cmd sidecar so the two
        # streams stay grep-able by file (raw quest dump = pre-decoder
        # inputs; planner sidecar = post-decoder emissions). Each row is
        # one manager tick that consumed a real Quest3 reader sample.
        self._quest3_raw_path = cfg.quest3_raw_log_path
        if self._quest3_raw_path is not None:
            self._quest3_raw_path.parent.mkdir(parents=True, exist_ok=True)
            self._quest3_raw_file = self._quest3_raw_path.open("a", buffering=1)
            self._quest3_raw_lock = threading.Lock()
            self._quest3_raw_count = 0
            self._quest3_raw_last_log = time.monotonic()
            log.info(
                "[quest3-raw] capture -> %s "
                "(one row per tick; replay with --quest3-replay-from)",
                self._quest3_raw_path,
            )
        else:
            self._quest3_raw_file = None
            self._quest3_raw_lock = threading.Lock()
            self._quest3_raw_count = 0
            self._quest3_raw_last_log = 0.0

        # --- Resume chord PUB (split-topology safety) ---------------------
        self._resume_sock = None
        if cfg.resume_pub_enabled:
            self._resume_sock = self._ctx.socket(zmq.PUB)
            self._resume_sock.setsockopt(zmq.LINGER, 0)
            self._resume_sock.bind(
                f"tcp://{cfg.resume_pub_host}:{cfg.resume_pub_port}"
            )
            log.info(
                "[safety] pose_resume PUB bound at tcp://%s:%d (topic=%s) -- "
                "operator chord A+B held for >=%.1fs republishes every %.2fs.",
                cfg.resume_pub_host, cfg.resume_pub_port, cfg.resume_pub_topic,
                cfg.resume_chord_hold_s, cfg.resume_chord_rep_s,
            )
        # Chord-hold state. ``_resume_chord_active_since`` is the
        # monotonic timestamp when A+B (without X+Y) was first seen;
        # cleared the moment the chord breaks. ``_resume_last_pub`` is
        # the last time we published a pose_resume frame (used to limit
        # republish rate while the operator continues to hold the chord).
        # ``_resume_press_count`` is for diagnostics on the periodic log line.
        self._resume_chord_active_since: Optional[float] = None
        self._resume_last_pub: float = -1.0
        self._resume_press_count: int = 0

        # --- Motor monitor SUB (Phase 5c forensic sidecar) ---------------
        self._motor_monitor_sock = None
        self._motor_monitor_thread = None
        self._motor_monitor_msg_count = 0
        if cfg.motor_monitor_sub_enabled and cfg.motor_monitor_sub_host:
            try:
                self._motor_monitor_sock = self._ctx.socket(zmq.SUB)
                self._motor_monitor_sock.setsockopt(zmq.LINGER, 0)
                self._motor_monitor_sock.setsockopt(zmq.RCVHWM, 10)
                self._motor_monitor_sock.setsockopt_string(
                    zmq.SUBSCRIBE, cfg.motor_monitor_sub_topic,
                )
                self._motor_monitor_sock.connect(
                    f"tcp://{cfg.motor_monitor_sub_host}:{cfg.motor_monitor_sub_port}"
                )
                self._motor_monitor_thread = threading.Thread(
                    target=self._motor_monitor_loop,
                    name="motor_monitor_sub",
                    daemon=True,
                )
                self._motor_monitor_thread.start()
                log.info(
                    "[safety] motor_monitor SUB connected to tcp://%s:%d "
                    "(topic=%s). Each message appended to sidecar JSONL.",
                    cfg.motor_monitor_sub_host, cfg.motor_monitor_sub_port,
                    cfg.motor_monitor_sub_topic,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[safety] failed to start motor_monitor SUB on "
                    "tcp://%s:%d: %s. Sidecar will not record motor-side events.",
                    cfg.motor_monitor_sub_host, cfg.motor_monitor_sub_port, exc,
                )
                self._motor_monitor_sock = None
                self._motor_monitor_thread = None

        # --- Engage-pose SUB (2026-06-10 follow-up 10) -------------------
        # Caches the latest decoded joint_pos_mj from the wire so
        # OFF -> non-OFF can snap the operator's freeze to "wherever
        # the robot currently is" instead of the legacy X2 neutral
        # stand pose. See ManagerConfig docstring for the why.
        self._engage_pose_sock = None
        self._engage_pose_thread = None
        self._engage_pose_lock = threading.Lock()
        self._engage_pose_jpos: Optional[np.ndarray] = None
        self._engage_pose_left_hand: Optional[np.ndarray] = None
        self._engage_pose_right_hand: Optional[np.ndarray] = None
        self._engage_pose_last_ts: float = -1.0
        self._engage_pose_msg_count = 0
        if (
            cfg.preserve_arms_on_engage
            and cfg.engage_pose_sub_port > 0
        ):
            try:
                self._engage_pose_sock = self._ctx.socket(zmq.SUB)
                self._engage_pose_sock.setsockopt(zmq.LINGER, 0)
                self._engage_pose_sock.setsockopt(zmq.RCVHWM, 4)
                self._engage_pose_sock.setsockopt_string(
                    zmq.SUBSCRIBE, cfg.engage_pose_sub_topic,
                )
                self._engage_pose_sock.connect(
                    f"tcp://{cfg.engage_pose_sub_host}:"
                    f"{cfg.engage_pose_sub_port}"
                )
                self._engage_pose_thread = threading.Thread(
                    target=self._engage_pose_loop,
                    name="engage_pose_sub",
                    daemon=True,
                )
                self._engage_pose_thread.start()
                log.info(
                    "[engage-pose] SUB connected to tcp://%s:%d "
                    "(topic=%s, max_age=%dms). On OFF->non-OFF the "
                    "arm + hand freeze will snap to the wire's last "
                    "commanded pose instead of X2 neutral / fingers-open.",
                    cfg.engage_pose_sub_host, cfg.engage_pose_sub_port,
                    cfg.engage_pose_sub_topic,
                    cfg.engage_pose_sub_max_age_ms,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[engage-pose] failed to start SUB on tcp://%s:%d: "
                    "%s. OFF->non-OFF will fall back to neutral snap.",
                    cfg.engage_pose_sub_host, cfg.engage_pose_sub_port,
                    exc,
                )
                self._engage_pose_sock = None
                self._engage_pose_thread = None

    # -- lifecycle ------------------------------------------------------------

    def _resolve_calibration(self) -> OperatorCalibration:
        cal_path = Path(self._cfg.calibration_path)
        if not cal_path.is_file():
            neutral = _neutral_calibration_path()
            if neutral.is_file():
                log.warning(
                    "calibration not found at %s; falling back to the shipped "
                    "neutral calibration %s (arm IK will be uncalibrated; run "
                    "`python -m gear_sonic.scripts.vr_operator_calibrate "
                    "--operator-id default` for a per-operator fit)",
                    cal_path,
                    neutral,
                )
                cal_path = neutral
            else:
                raise SystemExit(
                    f"calibration not found at {cal_path} (and no shipped neutral "
                    f"calibration at {neutral}). Run "
                    "`python -m gear_sonic.scripts.vr_operator_calibrate "
                    "--operator-id default` first."
                )
        cal = OperatorCalibration.load_yaml(cal_path)
        log.info(
            "loaded calibration %s (operator=%s, L_residual=%.1fcm, "
            "R_residual=%.1fcm)",
            cal_path,
            cal.operator_id,
            cal.fit["left"].residual_m * 100,
            cal.fit["right"].residual_m * 100,
        )
        return cal

    def start(self) -> None:
        self._quest.start()
        # Allow PUB-SUB to settle before flooding the wire.
        time.sleep(0.2)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._quest.stop()
        except Exception:
            pass
        try:
            self._planner_sock.close(linger=0)
        except Exception:
            pass
        try:
            self._recorder_sock.close(linger=0)
        except Exception:
            pass
        if self._motor_monitor_sock is not None:
            try:
                self._motor_monitor_sock.close(linger=0)
            except Exception:
                pass
        if self._motor_monitor_thread is not None:
            try:
                self._motor_monitor_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._engage_pose_sock is not None:
            try:
                self._engage_pose_sock.close(linger=0)
            except Exception:
                pass
        if self._engage_pose_thread is not None:
            try:
                self._engage_pose_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._resume_sock is not None:
            try:
                self._resume_sock.close(linger=0)
            except Exception:
                pass
        if self._sidecar is not None:
            try:
                self._sidecar.close()
            except Exception:
                pass
        if self._quest3_raw_file is not None:
            try:
                # Flush + fsync so the file is structurally complete even
                # if the operator Ctrl-C's mid-tick. The capture fixture
                # is the only artefact carrying the live session, so we
                # are deliberately paranoid here.
                with self._quest3_raw_lock:
                    self._quest3_raw_file.flush()
                    try:
                        import os as _os
                        _os.fsync(self._quest3_raw_file.fileno())
                    except OSError:
                        pass
                    self._quest3_raw_file.close()
                log.info(
                    "[quest3-raw] closed %s after %d rows",
                    self._quest3_raw_path, self._quest3_raw_count,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("[quest3-raw] close failed: %s", exc)

    # -- main loop ------------------------------------------------------------

    def run(self) -> int:
        period = 1.0 / max(self._cfg.publish_rate_hz, 1e-6)
        next_tick = time.monotonic()
        tick = 0
        wait_logged = False

        log.info(
            "[manager-x2] running. mode=OFF; press A+B+X+Y to engage. "
            "Publishing planner_cmd to tcp://%s:%d, "
            "arm_targets/hand_finger_cmd/stream_mode/recorder_cmd to "
            "tcp://%s:%d",
            self._cfg.planner_cmd_host, self._cfg.planner_cmd_port,
            self._cfg.recorder_pub_host, self._cfg.recorder_pub_port,
        )
        log.info(
            "[manager-x2] stick polarity: invert_lx=%s invert_ly=%s "
            "invert_rx=%s invert_ry=%s (push --no-invert-ly etc. to flip)",
            self._cfg.invert_lx, self._cfg.invert_ly,
            self._cfg.invert_rx, self._cfg.invert_ry,
        )
        log.info(
            "[manager-x2] recorder audio cues: %s "
            "(controlled by --recorder-enabled / --no-recorder-enabled; "
            "the wrapper sets it to ON iff --with-record was passed). "
            "When OFF, X/Y in ARM_MAN still publish recorder_cmd but "
            "skip the 'Recording.' / 'Saved.' headset cue so the "
            "operator doesn't get a false ACK in --teleop-only runs.",
            "ON" if self._cfg.recorder_enabled else "OFF",
        )

        vr_input_max_age_s = float(self._cfg.vr_input_max_age_s)
        vr_input_stale_active = False
        vr_input_last_warn_t = 0.0

        try:
            while not self._stop.is_set():
                tick_now = time.monotonic()
                vr_pose = self._quest.get_3pt_pose()
                # SMPL path: readers with full-body tracking (PicoX2Reader)
                # also expose the raw 24-joint skeleton; republish it on the
                # recorder wire so full-body retargeting / SMPL-conditioned
                # consumers get the high-fidelity human pose alongside the
                # reduced robot-frame 3pt path. Quest3Reader has no such
                # accessor and this is a no-op there.
                _body_getter = getattr(self._quest, "get_body_pose_24x7", None)
                if _body_getter is not None:
                    _body_24x7 = _body_getter()
                    if _body_24x7 is not None:
                        self._publish_smpl_body(_body_24x7, tick)
                buttons = self._quest.get_buttons()
                triggers = self._quest.get_controller_inputs()
                lx, ly, rx, ry = self._quest.get_controller_axes()
                if self._cfg.invert_lx:
                    lx = -lx
                if self._cfg.invert_ly:
                    ly = -ly
                if self._cfg.invert_rx:
                    rx = -rx
                if self._cfg.invert_ry:
                    ry = -ry

                # Quest 3 input freshness gate. When the WebSocket goes
                # silent (headset to sleep, tab in background, WS drop)
                # the Quest3Reader keeps returning the last cached
                # snapshot, so without this check the manager would
                # publish the stale stick values forever. We snap inputs
                # to neutral so the decoder emits idle; the kplanner's
                # own command-watchdog (--command-watchdog-s) is the
                # second line of defence. ``vr_pose is None`` (no
                # packet at all yet) is handled separately just below.
                # Log cadence: WARNING once on the rising edge of the
                # stale state (operator notices immediately), then drop
                # to a quiet INFO reminder at a much lower rate so a
                # long-idle session (headset on charger, lunch break)
                # doesn't paint the terminal red for 10+ minutes.
                vr_age_s = self._quest.get_last_message_age_s()
                if (
                    vr_pose is not None
                    and vr_input_max_age_s > 0.0
                    and vr_age_s > vr_input_max_age_s
                ):
                    lx = ly = rx = ry = 0.0
                    buttons = (False, False, False, False)
                    triggers = (0.0, 0.0, 0.0, 0.0)
                    if not vr_input_stale_active:
                        log.warning(
                            "[manager-x2] Quest 3 input stale: last WS "
                            "packet was %.2fs ago (threshold %.2fs); "
                            "forcing sticks/buttons to neutral. Headset "
                            "asleep, tab in background, or WS dropped?",
                            vr_age_s, vr_input_max_age_s,
                        )
                        vr_input_last_warn_t = tick_now
                    elif tick_now - vr_input_last_warn_t > 30.0:
                        log.info(
                            "[manager-x2] Quest 3 input still stale "
                            "(age %.1fs); awaiting WS reconnect.",
                            vr_age_s,
                        )
                        vr_input_last_warn_t = tick_now
                    vr_input_stale_active = True
                else:
                    if vr_input_stale_active:
                        log.info(
                            "[manager-x2] Quest 3 input recovered "
                            "(age %.2fs <= %.2fs); resuming normal input.",
                            vr_age_s, vr_input_max_age_s,
                        )
                    vr_input_stale_active = False

                # Quest3 raw capture sidecar -- one row per tick the
                # manager actually saw a Quest sample. We snapshot
                # the post-invert axes (what the IntentDecoder sees)
                # AND the raw payload (so Quest3Replayer can drive
                # every getter the manager calls). Skipping the
                # ``vr_pose is None`` ticks below means the file row
                # count equals operator-driven ticks.
                if self._quest3_raw_file is not None and vr_pose is not None:
                    self._quest3_raw_emit(
                        tick=tick,
                        tick_now=tick_now,
                        axes_post_invert=(lx, ly, rx, ry),
                        buttons=buttons,
                        triggers=triggers,
                        vr_pose=vr_pose,
                    )

                if vr_pose is None:
                    if not wait_logged:
                        if self._cfg.input_source == "pico":
                            log.info(
                                "waiting for first PICO body frame "
                                "(XRoboToolkit app: connect to this PC, "
                                "enable head/controller/body streaming, "
                                "full-body mode, headset awake) ..."
                            )
                        else:
                            log.info(
                                "waiting for first Quest 3 packet "
                                "(open WebXR app at "
                                "https://<HOST>:%d) ...",
                                self._cfg.quest3_http_port,
                            )
                        wait_logged = True
                    self._publish_stream_mode(tick)
                    self._sleep_until(next_tick + period)
                    next_tick += period
                    continue
                wait_logged = False

                ev = self._button_sm.tick(*buttons)
                a_held, _b_held, x_held, y_held = buttons
                # Defer single-press side effects by 0.15s: pressing
                # A+X for the e-stop chord lands across 1-2 ticks, and
                # the leading edge used to fire the buttons' normal
                # actions (A=arm toggle, B=mode flip, X=episode) — every
                # e-stop attempt scrambled modes (operator 2026-08-03).
                # If the chord forms within the window, pending singles
                # are dropped; chord events still fire instantly.
                ev = self._defer_button_singles(
                    ev, bool(a_held), bool(x_held), tick_now)  # estop chord

                # ---- Operator E-STOP gesture (v3 2026-08-04, operator
                # spec): A+X HELD (right thumb + left thumb) while BOTH
                # triggers fire rapidly (>=3 cycles in 1 s; keep going
                # ~1s more to escalate to damp). Identical shape to the
                # pad gesture (A+X + rapid L2/R2) — one gesture on every
                # surface. Known one-shot side effects at chord press
                # (tolerable mid-emergency): A = arm-tracking toggle,
                # X = episode ping (recorder ignores in teleop-only).
                _lt, _rt, _lg, _rg = triggers
                _estop_chord = bool(a_held) and bool(x_held)
                # NOTE (2026-08-03 incident): the both-sticks-pinned
                # alternative chord was REMOVED on every surface — sticks
                # forward + trigger activity is a normal driving /
                # manipulation posture and collapsed the robot in sim.
                # A+X is the sole chord: it cannot be held while a thumb
                # drives a stick.
                if _estop_chord and tick % 25 == 0:
                    log.warning("[manager-x2] estop chord armed (A+X): "
                                "lt=%.2f rt=%.2f pumps_1s=%d",
                                float(_lt), float(_rt),
                                len(self._estop_gesture._pump_win))
                # VR input-link health (2026-08-03): controllers kept
                # detaching at the WebXR level and ALL button/trigger
                # input (incl. this e-stop) silently died while pose
                # kept streaming — the operator pumped away at a dead
                # surface. Scream about it so nobody discovers a dead
                # e-stop mid-emergency. Pad e-stop is unaffected.
                _src_n, _src_zero_age = self._quest.get_gamepad_health()
                _unstable = _src_n == 0 or _src_zero_age < 10.0
                if not _unstable:
                    self._input_dead_since = None
                elif getattr(self, "_input_dead_since", None) is None:
                    self._input_dead_since = tick_now
                # Cadence decay: 5s while fresh (operator mid-session
                # must notice fast), 60s once it's clearly a parked /
                # sleeping headset (operator feedback 2026-08-03:
                # steady CRITICAL spam drowns the useful logs).
                _warn_gap = (5.0 if self._input_dead_since is None
                             or tick_now - self._input_dead_since < 30.0
                             else 60.0)
                if (_unstable
                        and tick_now - getattr(
                            self, "_input_dead_warn_t", 0.0) > _warn_gap):
                    self._input_dead_warn_t = tick_now
                    log.critical(
                        "[manager-x2] !!! VR CONTROLLER INPUT UNSTABLE "
                        "(sources=%d, last zero-drop %.1fs ago): buttons/"
                        "triggers — INCLUDING VR E-STOP — may be DEAD. "
                        "Use the GAMEPAD e-stop. Check controller "
                        "batteries / re-grip to wake controllers.",
                        _src_n, _src_zero_age)
                _estop_ph = self._estop_gesture.tick(_lt, _rt, _estop_chord)
                if _estop_ph == 1:
                    self._play_audio_prompt(
                        "estop_activating",
                        fallback="Emergency stop activating.")
                elif _estop_ph == 2:
                    self._play_audio_prompt(
                        "estop_damping",
                        fallback="Emergency stop. Pure damping engaged.")
                if _estop_ph == 1:
                    log.critical("[manager-x2] !!! E-STOP (SOFT: A+X + "
                                 "rapid triggers) — keep going ~1s more "
                                 "for PURE DAMPING")
                    self._publish_planner_cmd(
                        LocomotionCmd(intent="estop", magnitude="soft"))
                elif _estop_ph == 2:
                    log.critical("[manager-x2] !!! E-STOP ESCALATED: "
                                 "PURE DAMPING (terminal)")
                    self._publish_planner_cmd(
                        LocomotionCmd(intent="estop", magnitude="damp"))
                if _estop_ph:
                    _sentinel = os.environ.get("X2_SOFT_SHUTDOWN_SENTINEL", "")
                    if _sentinel:
                        try:
                            open(_sentinel, "w").close()
                            log.critical("[manager-x2] soft-shutdown "
                                         "sentinel touched: %s", _sentinel)
                        except OSError as exc:
                            log.error("[manager-x2] sentinel write failed: %s",
                                      exc)

                # Split-topology SAFE_IDLE resume: A+B (without X+Y) held
                # for >= resume_chord_hold_s. No-op unless --resume-pub-enabled.
                self._tick_resume_chord(buttons, tick_now)

                # Live continuous waist target derived from BOTH sticks
                # (R-stick = pitch/yaw, L-stick = roll/height in
                # ARM_MANIPULATION mode only -- the decoder gates the
                # L-stick path internally). We compute this BEFORE the
                # mode transition handler so a B-press into
                # ARM_MANIPULATION can latch exactly the pose the
                # operator was holding at the moment of the press,
                # rather than the pose from the previous tick (which
                # would drift by up to one 50 Hz interval). The
                # continuous target is well-defined in any mode (it's
                # a pure function of stick state) but only consumed on
                # the LOCOMOTION -> ARM_MANIPULATION transition. v7.2:
                # A-modifier removed from the waist path (right-thumb
                # ergonomics; see decoder docstring). v7.4: 4-tuple
                # adds hip_height_m so the latched pose can pin the
                # squat / stand target across the mode flip.
                live_waist_target = self._intent.continuous_waist_target(
                    rx=rx, ry=ry, lx=lx, ly=ly,
                )

                # 1) Mode transitions ---------------------------------------
                transition = self._intent.update_mode(ev, now=tick_now)
                if transition is not None:
                    self._on_mode_transition(
                        transition,
                        vr_pose=vr_pose,
                        tick=tick,
                        live_waist_target=live_waist_target,
                    )

                # 2) Operator-facing UX hint: in OFF mode, A/B/X/Y by
                #    themselves do nothing useful. Print a one-shot hint
                #    so newcomers know they need the A+B+X+Y chord to
                #    leave OFF, then B once to flip to ARM_MANIPULATION.
                #    Re-armed after every successful mode transition.
                if self._intent.mode == StreamMode.OFF and (
                    ev.a_pressed or ev.b_pressed or ev.x_pressed or ev.y_pressed
                ) and not ev.abxy_pressed and not self._off_mode_hint_logged:
                    log.info(
                        "Mode is OFF; single button press is a no-op. "
                        "Press A+B+X+Y SIMULTANEOUSLY to enter LOCOMOTION, "
                        "then press B alone to flip to ARM_MANIPULATION "
                        "(arms IK), then A to engage arm tracking."
                    )
                    self._off_mode_hint_logged = True

                # 3) Arm IK engage toggle (manager-internal; recorder
                #    does not need to know).
                if ev.a_pressed and self._intent.mode == StreamMode.ARM_MANIPULATION:
                    self._retargeter.set_engaged(not self._retargeter.is_engaged)
                    log.info(
                        "[A] arm tracking -> %s",
                        "ACTIVE" if self._retargeter.is_engaged else "IDLE",
                    )

                # 4) Episode lifecycle (ARM_MANIPULATION ONLY).
                #    The recorder consumes 'start' / 'save' / 'discard'
                #    actions; mismatched names silently no-op. We gate
                #    every recording trigger on ARM_MAN mode so:
                #      - A and X behave purely as locomotion modifiers
                #        in LOCOMOTION (A held = walk, X held + rx =
                #        90° turn) without ever poking the recorder.
                #      - The operator cannot accidentally start / save
                #        an episode while still positioning the robot.
                #
                # Mapping (deliberately single-button, no chord):
                #   X press -> 'start'  (only when not currently recording)
                #   Y press -> 'save'   (only when currently recording)
                #
                # The previous A+B chord for 'start' collided with the
                # B-single mode toggle: pressing them in the same tick
                # both started a recording AND flipped to LOCOMOTION,
                # immediately yanking the planner reference out from
                # under the freshly-opened episode. Splitting start /
                # save onto two distinct one-shot buttons removes the
                # chord entirely so B is always unambiguously a mode
                # toggle. Discard mid-episode is intentionally not
                # bound to a button right now (kill the manager and
                # delete the latest parquet/mp4 if you need to drop a
                # bad episode -- the recorder still understands the
                # 'discard' wire action if we re-bind it later).
                self._handle_episode_buttons(ev, tick)

                # 4b) Stick clicks ------------------------------------------
                #     LEFT thumbstick click  -> cycle deploy MuJoCo viewer
                #         cameras (xdotool ']' keypress -- mujoco's
                #         next-fixed-camera key). Pre-v7 this was
                #         the right click; moved here so the operator can
                #         keep their right thumb on the lean / twist stick
                #         while clicking the LEFT stick to re-frame.
                #     RIGHT thumbstick click -> toggle ``_waist_frozen``.
                #         While frozen, the right stick is suppressed for
                #         waist control: the planner stays in STATIC_HOLD
                #         at the pose the operator was holding when the
                #         click landed. Another R-click releases it.
                #
                #     Both are active in LOCOMOTION + ARM_MAN, idle in
                #     OFF (consistent with the rest of the manager:
                #     OFF means "ignore controller events"). Rising-edge
                #     tracked manually because ButtonStateMachine only
                #     handles the four face buttons today.
                #
                # TODO(unified-vr-input-topic): once the manager
                # publishes a unified ``vr_input`` ZMQ topic carrying
                # the full controller state, the deploy viewer should
                # subscribe directly and update mjvCamera in-process,
                # making the xdotool hack unnecessary. See
                # ViewerCameraCycler.__doc__ for the rationale.
                l_click, r_click = self._quest.get_stick_clicks()
                l_click_edge = l_click and not self._prev_left_stick_click
                r_click_edge = r_click and not self._prev_right_stick_click
                self._prev_left_stick_click = l_click
                self._prev_right_stick_click = r_click
                if l_click_edge and self._intent.mode != StreamMode.OFF:
                    # ALWAYS log the rising edge so we can tell whether
                    # the L-click event reached the manager at all,
                    # even when the cycler is disabled or fails. The
                    # cycler itself does its own one-shot WARN on the
                    # first failure (missing xdotool / DISPLAY / no
                    # MuJoCo window) and a periodic re-warn after that
                    # so a long session doesn't go silent if the viewer
                    # gets restarted under us. Crucial for diagnosing
                    # "I pressed the stick but nothing happened":
                    # without this line you can't tell stick-not-
                    # detected from cycler-failed-silently.
                    if self._viewer_cycler is None:
                        log.info(
                            "[L-click] camera cycler disabled "
                            "(--no-viewer-camera-cycler); ignoring."
                        )
                    else:
                        ok = self._viewer_cycler.cycle()
                        log.info(
                            "[L-click] camera cycle: %s",
                            "ok ('%s' dispatched to deploy viewer)" % (
                                self._viewer_cycler.CYCLE_KEYSYM,
                            )
                            if ok
                            else "no-op (cooldown or xdotool/window "
                            "unavailable; see preceding [viewer-cycler] "
                            "warning for the specific reason)",
                        )
                if r_click_edge and self._intent.mode != StreamMode.OFF:
                    self._toggle_waist_freeze(live_waist_target)

                # 3) Locomotion command ---------------------------------------
                # Held-button modifiers (LOCOMOTION-only via decoder
                # short-circuit). A held + ly = continuous walk;
                # X held + rx = 90° turn. Y held was crouch; currently
                # gated off in IntentDecoder. Buttons already destructured
                # above so the live_waist_target sample agrees with the
                # decoder's view of the held modifiers.
                #
                # StickFilter gating: when a smoother is configured and
                # the manager is in LOCOMOTION, route (ly, lx, rx) through
                # the filter so the decoder + downstream planner see the
                # band-limited stream rather than the raw step inputs the
                # VR controllers ship. Other modes (OFF / ARM_MAN) bypass
                # the filter so the right-stick lean / chord-detection
                # paths stay 1:1 with the raw axes. The filter is reset
                # on each LOCOMOTION re-entry so the first post-mode-flip
                # tick is a pass-through (no spurious slew clamp from
                # zero anchor).
                lx_for_decoder, ly_for_decoder, rx_for_decoder = lx, ly, rx
                if self._stick_filter is not None:
                    in_loco = self._intent.mode == StreamMode.LOCOMOTION
                    if in_loco:
                        if not self._stick_filter_prev_in_loco:
                            self._stick_filter.reset()
                            self._stick_filter_last_t = tick_now
                        dt = (
                            tick_now - self._stick_filter_last_t
                            if self._stick_filter_last_t is not None
                            else 0.02
                        )
                        self._stick_filter_last_t = tick_now
                        # Channel map: ly -> stick_fwd, lx -> stick_side,
                        # rx -> stick_yaw. The decoder applies the same
                        # mapping in ``_continuous_stick_targets``; we
                        # filter the raw axes (pre-deadzone) so the LPF
                        # tail can drain naturally through the deadzone
                        # boundary rather than getting truncated at the
                        # decoder edge.
                        ly_for_decoder, lx_for_decoder, rx_for_decoder = (
                            self._stick_filter.step(
                                stick_fwd=ly,
                                stick_side=lx,
                                stick_yaw=rx,
                                dt=dt,
                            )
                        )
                    else:
                        # Reset on exit from LOCOMOTION; on next re-entry
                        # the first tick is a pass-through. ``ry`` (lean)
                        # is never filtered because the decoder routes it
                        # through a different code path (hold_torso) that
                        # has its own internal hysteresis.
                        self._stick_filter.reset()
                        self._stick_filter_last_t = None
                    self._stick_filter_prev_in_loco = in_loco

                cmd = self._intent.decode_locomotion(
                    lx=lx_for_decoder, ly=ly_for_decoder,
                    rx=rx_for_decoder, ry=ry,
                    y_held=(y_held and self._intent.mode == StreamMode.LOCOMOTION),
                    a_held=(a_held and self._intent.mode == StreamMode.LOCOMOTION),
                    x_held=(x_held and self._intent.mode == StreamMode.LOCOMOTION),
                    now=tick_now,
                )
                if cmd is not None:
                    # Drop live ``hold_torso`` updates while the operator
                    # has the waist frozen (R-click toggle). The planner
                    # stays in STATIC_HOLD at its current target because
                    # no new hold_torso commands arrive; non-hold commands
                    # (walk / turn / idle) still flow through so the
                    # operator can keep walking with the body leaned.
                    if self._waist_frozen and cmd.intent == "hold_torso":
                        pass
                    else:
                        self._publish_planner_cmd(cmd)
                        self._sidecar_emit(cmd, tick)

                # Ownership keepalive: while engaged, re-assert the last
                # command every 0.5 s so a quiet held stick (the
                # IntentDecoder emits only on change) cannot trip the
                # planner's 2 s VR-silence release. OFF publishes idle on
                # entry, which explicitly releases ownership — no
                # keepalive there by design.
                if (self._intent.mode != StreamMode.OFF
                        and getattr(self, "_last_planner_cmd", None) is not None
                        and tick_now - getattr(self, "_last_planner_pub_ts", 0.0) > 0.5):
                    self._publish_planner_cmd(self._last_planner_cmd)

                # In ARM_MANIPULATION the per-tick decoder output above
                # is whatever ``hold_torso`` target the right stick is
                # currently demanding (or None if the operator is
                # pushing the L-stick / hard R-stick, both of which the
                # decoder filters out in ARM_MAN). The B-press into
                # ARM_MAN handler already emitted a one-shot
                # ``hold_torso(latched)`` so the planner is in
                # STATIC_HOLD with no jump on entry; subsequent ticks
                # just slew the target via the same path.

                # 4) Retargeting ---------------------------------------------
                # We RUN the retargeter every tick (even in LOCOMOTION
                # mode) so the IK doesn't snap when the operator
                # switches back to ARM. But the published values follow
                # the freeze rule below.
                inp = self._build_retarget_input(vr_pose=vr_pose, triggers=triggers)
                out = self._retargeter.step(inp)

                if self._intent.mode == StreamMode.ARM_MANIPULATION:
                    publish_left_arm = out.left_arm_q
                    publish_right_arm = out.right_arm_q
                    publish_left_hand = out.left_hand_q
                    publish_right_hand = out.right_hand_q
                    # Refresh the freeze cache so a flip back to
                    # LOCOMOTION continues from the latest pose.
                    self._frozen_left_arm_q = out.left_arm_q.copy()
                    self._frozen_right_arm_q = out.right_arm_q.copy()
                    self._frozen_left_hand_q = out.left_hand_q.copy()
                    self._frozen_right_hand_q = out.right_hand_q.copy()
                else:
                    # OFF or LOCOMOTION: hold last commanded arm + hand
                    # targets. Walking with VR-driven arms is unsafe
                    # because the operator's hands aren't visible in
                    # their HMD view while they're looking at the floor.
                    publish_left_arm = self._frozen_left_arm_q
                    publish_right_arm = self._frozen_right_arm_q
                    publish_left_hand = self._frozen_left_hand_q
                    publish_right_hand = self._frozen_right_hand_q

                self._publish_arm_targets(
                    left=publish_left_arm,
                    right=publish_right_arm,
                    is_engaged=self._retargeter.is_engaged,
                    tick=tick,
                )
                self._publish_hand_finger_cmd(
                    left=publish_left_hand,
                    right=publish_right_hand,
                    tick=tick,
                )
                self._publish_stream_mode(tick)

                # 2026-06-10 follow-up 5: periodic per-input telemetry
                # for the "fingers not responding" diagnosis. The
                # recorder telemetry (``hand|L|=0.000(manager)``)
                # tells us what the manager is PUBLISHING, but not
                # whether the operator's triggers reached the
                # manager. This log line closes that gap:
                #   - ``triggers=lt rt lg rg``: post-Quest3Reader,
                #     pre-retargeter operator input. If these are
                #     zero while the operator believes they're
                #     pulling triggers, the issue is upstream
                #     (controller battery, occluded sensor, wrong
                #     axis mapping, XRHand vs controller mode).
                #   - ``hand_q|L| |R|``: post-retargeter, what we
                #     just put on hand_finger_cmd. If triggers are
                #     non-zero but hand_q is ~0, the retargeter or
                #     calibration is broken. If triggers are
                #     non-zero AND hand_q is non-zero AND the
                #     OmniHand still doesn't move, the wire is
                #     broken (proxy override forwarding, OmniHand
                #     SUB host/port wiring).
                # Gated on ARM_MAN so it doesn't spam during
                # LOCOMOTION / OFF where the published hand_q is
                # the frozen last value and the recorder telemetry
                # is the better diagnostic anyway.
                if (
                    tick % 1500 == 0
                    and self._intent.mode == StreamMode.ARM_MANIPULATION
                ):
                    lt, rt, lg, rg = triggers
                    lh_norm = float(np.linalg.norm(publish_left_hand))
                    rh_norm = float(np.linalg.norm(publish_right_hand))
                    log.info(
                        "[manager-x2] tick=%d ARM_MAN triggers="
                        "lt=%.2f rt=%.2f lg=%.2f rg=%.2f "
                        "published hand_q|L|=%.3f |R|=%.3f",
                        tick, lt, rt, lg, rg, lh_norm, rh_norm,
                    )

                tick += 1
                next_tick += period
                self._sleep_until(next_tick)
        except KeyboardInterrupt:
            log.info("\n[manager-x2] interrupted")
        finally:
            self.stop()

        return tick

    # -- per-tick helpers -----------------------------------------------------

    def _build_retarget_input(
        self,
        *,
        vr_pose: np.ndarray,
        triggers: tuple[float, float, float, float],
    ) -> RetargetTickInput:
        l_curls, r_curls, l_src, r_src = self._quest.get_hand_curls()
        l_oppose, r_oppose = self._quest.get_thumb_opposition()
        l_tip, r_tip = self._quest.get_finger_tip_oppose()
        return RetargetTickInput(
            vr_pose=vr_pose,
            triggers=tuple(float(x) for x in triggers),
            left_curls=l_curls,
            right_curls=r_curls,
            left_thumb_oppose=None if l_oppose is None else float(l_oppose),
            right_thumb_oppose=None if r_oppose is None else float(r_oppose),
            left_finger_tip_oppose=l_tip,
            right_finger_tip_oppose=r_tip,
            left_hand_source=l_src,
            right_hand_source=r_src,
        )

    def _toggle_waist_freeze(
        self,
        live_waist_target: tuple[float, float, float, float | None],
    ) -> None:
        """R-thumbstick-click handler: toggle waist freeze on/off.

        On freeze ON: the live waist target at the moment of click is
        captured into ``_latched_waist`` so subsequent code paths (e.g.
        a B-press into ARM_MANIPULATION while frozen) source the
        latched pose instead of resampling. The decoder keeps emitting
        ``hold_torso`` updates internally, but the manager suppresses
        them at publish time (see the ``_waist_frozen`` check in the
        main loop), so the planner stays at its current STATIC_HOLD
        target even as the operator's right stick drifts.

        On freeze OFF: ``_latched_waist`` is cleared and the suppression
        lifts. The next decoder tick re-emits the live target, so the
        planner blends to whatever the operator is now holding (or to
        neutral, if the stick is centered).
        """
        if self._waist_frozen:
            self._waist_frozen = False
            self._latched_waist = None
            log.info("[R-click] waist freeze -> RELEASED")
            self._play_audio_prompt(
                "torso_released", fallback="Torso released.",
            )
        else:
            pitch, roll, yaw, hip_h = live_waist_target
            self._waist_frozen = True
            self._latched_waist = (pitch, roll, yaw, hip_h)
            log.info(
                "[R-click] waist freeze -> FROZEN at "
                "pitch=%+.1f roll=%+.1f yaw=%+.1f hip_h=%s",
                pitch, roll, yaw,
                f"{hip_h:.3f}m" if hip_h is not None else "default",
            )
            self._play_audio_prompt(
                "torso_frozen", fallback="Torso frozen.",
            )

    def _on_mode_transition(
        self,
        transition: ModeTransition,
        *,
        vr_pose: np.ndarray,
        tick: int,
        live_waist_target: tuple[float, float, float, float | None] = (
            0.0, 0.0, 0.0, None,
        ),
    ) -> None:
        log.info("[manager-x2] mode %s -> %s", transition.previous.name, transition.current.name)
        # Re-arm the OFF-mode hint so it fires once again on next entry
        # to OFF (helps if the operator forgets the chord mid-session).
        self._off_mode_hint_logged = False
        # Whenever we leave OFF we want a fresh idle planner_cmd so the
        # planner clears any stale queue entries and goes to idle_stand.
        if transition.previous == StreamMode.OFF:
            self._publish_planner_cmd(LocomotionCmd("idle", "default"))
            # Snap the arm + hand freeze caches back to neutral so the
            # very next arm_targets / hand_finger_cmd publish carries a
            # clean default pose. Without this, OFF held the LAST
            # commanded arm + hand targets from the prior session (eg.
            # mid-grasp, wrist twisted from a manipulation pose), and
            # re-entering LOCOMOTION via A+B+X+Y would continue to
            # publish that stale pose. The deploy's wrist-bypass=ik
            # would then drive the wrist motors to those stale IK
            # values -- they'd "stick" wherever the prior run left
            # them, even though the operator just engaged a fresh
            # session. Reset gives a predictable starting position
            # (X2 neutral stand-pose arms, fingers fully open) before
            # the operator presses B->ARM and A->engage to start
            # actively teleoperating.
            #
            # Safety: the C++ deploy slews PD targets via its soft-
            # start ramp + per-tick step clamp, so this neutral target
            # blends in smoothly rather than commanding an instant jump.
            # 2026-06-10 follow-up 10: when --preserve-arms-on-engage
            # is set AND a fresh wire frame is cached, snap arms + hands
            # to "wherever the robot currently is" instead of X2 neutral
            # / fingers-open. This is the VLA-takeover smoothness fix:
            # VLA was driving the arms to pose X just before the
            # operator chord; without this, the manager would publish
            # arm_targets at neutral, the recorder would forward those
            # into ``pose``, and the proxy's engagement ramp
            # (follow-up 9b) would then slowly walk the wire from
            # VLA's X down to neutral over ~5 s. Snapping the freeze
            # to the wire's current jpos collapses the operator's
            # first override frame onto VLA's last commanded pose --
            # the engagement ramp absorbs the residual delta and the
            # deploy sees a continuous wire. Hands ride along so a
            # mid-grasp doesn't get force-released across the
            # boundary; the operator can still open fingers normally
            # via the VR trigger once engaged.
            wire_left, wire_right, wire_lhand, wire_rhand, src = (
                self._resolve_engage_freeze()
            )
            if src == "wire" and wire_left is not None and wire_right is not None:
                self._frozen_left_arm_q = wire_left
                self._frozen_right_arm_q = wire_right
                self._frozen_left_hand_q = (
                    wire_lhand if wire_lhand is not None
                    else np.zeros(10, dtype=np.float64)
                )
                self._frozen_right_hand_q = (
                    wire_rhand if wire_rhand is not None
                    else np.zeros(10, dtype=np.float64)
                )
                hands_kind = (
                    "from-wire"
                    if (wire_lhand is not None and wire_rhand is not None)
                    else "partial-wire+fingers=open-fallback"
                    if (wire_lhand is not None or wire_rhand is not None)
                    else "fingers=open-fallback"
                )
                log.info(
                    "[manager-x2] OFF -> %s: snap arm+hand freeze to WIRE pose "
                    "(arms=jpos[15:29], hands=%s)",
                    transition.current.name,
                    hands_kind,
                )
            else:
                self._frozen_left_arm_q = self._retargeter._teleop.left_neutral_q
                self._frozen_right_arm_q = self._retargeter._teleop.right_neutral_q
                self._frozen_left_hand_q = np.zeros(10, dtype=np.float64)
                self._frozen_right_hand_q = np.zeros(10, dtype=np.float64)
                if self._cfg.preserve_arms_on_engage:
                    log.info(
                        "[manager-x2] OFF -> %s: engage-pose SUB stale/empty, "
                        "falling back to NEUTRAL snap "
                        "(arms=X2 stand pose, fingers=open)",
                        transition.current.name,
                    )
                else:
                    log.info(
                        "[manager-x2] OFF -> %s: snap arm + hand freeze to neutral "
                        "(arms=X2 stand pose, fingers=open)",
                        transition.current.name,
                    )

        # ----- LOCOMOTION <-> ARM_MANIPULATION transitions ---------------
        # Going INTO ARM_MANIPULATION: latch whatever pitch / roll / yaw
        # the operator was holding via the right stick at the moment of
        # B-press, and pin the planner's STATIC_HOLD to that pose. This
        # mirrors the existing arm-latch on the reverse transition: the
        # operator picks an upper-body pose in LOCOMOTION, B-clicks into
        # ARM_MANIPULATION, then drives arms via VR IK while the lower
        # body stays at the chosen lean / twist for extra reach.
        #
        # Going OUT of ARM_MANIPULATION (back to LOCOMOTION): clear the
        # latch and emit a single idle cmd so the planner cleanly
        # blends out of STATIC_HOLD; subsequent ticks resume normal
        # continuous emission from the right stick.
        if (
            transition.previous == StreamMode.LOCOMOTION
            and transition.current == StreamMode.ARM_MANIPULATION
        ):
            # Source the latch from whichever target is currently
            # authoritative: if the operator already R-clicked to freeze
            # in LOCOMOTION, use the frozen pose (so the B-press doesn't
            # snap to a slightly different live sample). Otherwise, take
            # the live continuous target as today.
            if self._waist_frozen and self._latched_waist is not None:
                pitch, roll, yaw, hip_h = self._latched_waist
            else:
                pitch, roll, yaw, hip_h = live_waist_target
                self._latched_waist = (pitch, roll, yaw, hip_h)
            self._publish_planner_cmd(
                LocomotionCmd(
                    intent="hold_torso",
                    magnitude="continuous",
                    waist_pitch_deg=pitch,
                    waist_roll_deg=roll,
                    waist_yaw_deg=yaw,
                    hip_height_m=hip_h,
                )
            )
            self._retargeter.reset_finger_filter()
            log.info(
                "[manager-x2] latched waist hold pitch=%+.1f roll=%+.1f yaw=%+.1f "
                "hip_h=%s%s",
                pitch, roll, yaw,
                f"{hip_h:.3f}m" if hip_h is not None else "default",
                " (R-click freeze active)" if self._waist_frozen else "",
            )
            # Audio cue: separate "torso_locked" prompt only when the
            # latched pose is meaningfully non-neutral (>= 1 deg on
            # any axis). For neutral poses the standard
            # "mode_arm_manipulation" cue below covers it.
            if max(abs(pitch), abs(roll), abs(yaw)) >= 1.0:
                self._play_audio_prompt(
                    "mode_torso_locked", fallback="Torso locked.",
                )
        elif (
            transition.previous == StreamMode.ARM_MANIPULATION
            and transition.current == StreamMode.LOCOMOTION
        ):
            # If the operator R-click-froze the waist before / during
            # ARM_MANIPULATION, KEEP the freeze across the transition
            # back to LOCOMOTION: don't clear ``_latched_waist`` and
            # don't blow the planner's STATIC_HOLD away with an idle
            # cmd. The operator can now walk / turn with the L stick
            # while the body stays at the locked pose. R-click again
            # to release. If the freeze flag is OFF, fall through to
            # the legacy release-into-idle path so the body smoothly
            # blends back to standing.
            if self._waist_frozen and self._latched_waist is not None:
                log.info(
                    "[manager-x2] ARM->LOCO with R-click freeze active; "
                    "keeping STATIC_HOLD at latched pose"
                )
            else:
                self._latched_waist = None
                self._publish_planner_cmd(LocomotionCmd("idle", "default"))
        elif transition.current == StreamMode.ARM_MANIPULATION:
            # Reached ARM_MANIPULATION not via LOCOMOTION (e.g. would
            # only happen if a future chord adds a direct OFF -> ARM
            # path). Keep the legacy "planner idles" semantics.
            self._publish_planner_cmd(LocomotionCmd("idle", "default"))
            self._retargeter.reset_finger_filter()

        # On leaving an active mode -> OFF: tell the recorder to drop
        # any in-progress episode (this is a hard E-stop semantic;
        # operator can re-arm and start fresh). Also drop the R-click
        # freeze so the next engagement starts with a clean slate.
        if transition.current == StreamMode.OFF and transition.previous != StreamMode.OFF:
            self._latched_waist = None
            self._waist_frozen = False
            # Disengage: the ONE place that releases command ownership
            # back to the pad (vr_release=True on the wire).
            self._publish_planner_cmd(
                LocomotionCmd("idle", "default"), release=True)
            self._publish_recorder_cmd("estop", tick)

        # Headset audio cue for the transition. Mapping is the same
        # vocabulary as the IntentDecoder.StreamMode enum so adding a
        # new mode automatically requires a matching prompt key (the
        # ``test_every_stream_mode_has_a_prompt`` contract test in
        # tests/test_quest3_audio_prompts.py guards against drift).
        _MODE_AUDIO_KEYS = {
            StreamMode.OFF:               ("mode_off",              "Off."),
            StreamMode.LOCOMOTION:        ("mode_locomotion",       "Locomotion."),
            StreamMode.ARM_MANIPULATION:  ("mode_arm_manipulation", "Arm manipulation."),
        }
        cue = _MODE_AUDIO_KEYS.get(transition.current)
        if cue is not None:
            key, fallback = cue
            self._play_audio_prompt(key, fallback=fallback)

    # -- publishers -----------------------------------------------------------

    def _publish_planner_cmd(
        self, cmd: LocomotionCmd, release: bool = False
    ) -> None:
        try:
            self._planner_sock.send_multipart(
                [
                    self._cfg.planner_cmd_topic.encode("ascii"),
                    _planner_cmd_payload(cmd, release=release),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass
        # Ownership keepalive bookkeeping: the planner releases VR
        # ownership after 2 s of VR silence (crash safety), and the
        # IntentDecoder only emits on change — so the tick loop
        # republishes the last command every 0.5 s while engaged.
        # A release is terminal: never keepalive it.
        self._last_planner_cmd = None if release else cmd
        self._last_planner_pub_ts = time.monotonic()
        if self._cfg.verbose:
            log.debug("planner_cmd <- intent=%s magnitude=%s", cmd.intent, cmd.magnitude)

    def _publish_arm_targets(
        self,
        *,
        left: np.ndarray,
        right: np.ndarray,
        is_engaged: bool,
        tick: int,
    ) -> None:
        # ``passthrough_arm_targets`` is the recorder-side sentinel
        # introduced for the LOCO_DECOUPLED_ARMS=0 path: True means
        # "treat this message as 'no operator arm override', let the
        # planner's predicted arms flow through the merge". Computed
        # here (and not at the call sites) so the same merge-mode is
        # applied consistently to every publish, including the
        # frozen-arms repeat publishes that the LOCOMOTION-mode
        # safety path emits at idle.
        #
        # Guarded on ``StreamMode.LOCOMOTION`` so ARM_MAN and OFF
        # always keep the legacy real-arms-override path (ARM_MAN
        # operator IK MUST win for manipulation; OFF preserves the
        # last-known safety pose held in the recorder cache).
        passthrough_arm_targets = (
            (not self._cfg.intent_loco_decoupled_arms)
            and (self._intent.mode is StreamMode.LOCOMOTION)
        )
        payload = {
            "left_q_rad": np.asarray(left, dtype=np.float32).tolist(),
            "right_q_rad": np.asarray(right, dtype=np.float32).tolist(),
            "is_engaged": bool(is_engaged),
            "passthrough_arm_targets": bool(passthrough_arm_targets),
            "tick": int(tick),
            "ts": time.time(),
        }
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.arm_targets_topic.encode("ascii"),
                    _msgpack_payload(payload),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _publish_smpl_body(self, body_24x7: np.ndarray, tick: int) -> None:
        """SMPL-path publish: raw 24-joint body skeleton on the recorder
        wire (topic ``smpl_body``). Rows are [x, y, z, qx, qy, qz, qw] in
        the Unity frame, untouched — consumers pick their own transform
        (pico_3pt for the robot frame, GMR's coordinate_transform for
        retargeting). ~1.4 kB/tick at 50 Hz."""
        payload = {
            "body_pose_24x7": np.asarray(body_24x7, dtype=np.float32).tolist(),
            "frame": "unity_xyz_quat_xyzw",
            "tick": int(tick),
            "ts": time.time(),
        }
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.smpl_body_topic.encode("ascii"),
                    _msgpack_payload(payload),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _publish_hand_finger_cmd(
        self,
        *,
        left: np.ndarray,
        right: np.ndarray,
        tick: int,
    ) -> None:
        payload = {
            "left_hand_q": np.asarray(left, dtype=np.float32).tolist(),
            "right_hand_q": np.asarray(right, dtype=np.float32).tolist(),
            "tick": int(tick),
            "ts": time.time(),
        }
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.hand_finger_cmd_topic.encode("ascii"),
                    _msgpack_payload(payload),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _publish_stream_mode(self, tick: int) -> None:
        payload = {
            "mode": self._intent.mode.name,
            "tick": int(tick),
            "ts": time.time(),
        }
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.stream_mode_topic.encode("ascii"),
                    _msgpack_payload(payload),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _publish_recorder_cmd(self, action: str, tick: int) -> None:
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.recorder_cmd_topic.encode("ascii"),
                    _recorder_cmd_payload(action, tick),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _handle_episode_buttons(self, ev: ButtonEvents, tick: int) -> None:
        """Translate X / Y rising edges in ARM_MANIPULATION into recorder
        commands and (optionally) headset audio cues.

        Splits cleanly into wire path + audio path:

        - **Wire** (always fires when the chord is right): publish a
          ``recorder_cmd`` so the recorder can decide to honour or
          ignore the action. Same in ``--with-record`` and
          ``--teleop-only`` -- the recorder is the source of truth.
        - **Audio** (gated on ``self._cfg.recorder_enabled``): play
          the ``record_start`` / ``record_save`` headset cue. When
          disabled, no cue plays so the operator doesn't get a false
          ACK in teleop-only sessions where no parquet is being
          written. Known footgun (still): in ``--with-record`` the
          cue still plays on X-while-recording / Y-with-no-episode
          because the manager has no ACK channel from the recorder.
          Tracked as the "recorder ACK topic" follow-up; the gate
          covers the most common foot-shoot (operator forgets the
          ``--with-record`` flag and trusts the headset).

        Args:
            ev: Rising-edge button events for this tick.
            tick: Current tick index, stamped into the ZMQ payload.
        """
        if self._intent.mode != StreamMode.ARM_MANIPULATION:
            return
        if ev.x_pressed:
            log.info("[X] start episode forwarded to recorder")
            self._publish_recorder_cmd("start", tick)
            if self._cfg.recorder_enabled:
                self._play_audio_prompt(
                    "record_start", fallback="Recording.",
                )
        if ev.y_pressed:
            log.info("[Y] save episode forwarded to recorder")
            self._publish_recorder_cmd("save", tick)
            if self._cfg.recorder_enabled:
                self._play_audio_prompt(
                    "record_save", fallback="Saved.",
                )

    def _play_audio_prompt(
        self, key: str, *, fallback: Optional[str] = None,
    ) -> None:
        """Push a ``play_audio`` message to the WebXR client.

        The client maps ``key`` to ``/audio/<key>.mp3`` (cached on disk
        by ``ensure_prompt_audio_files``) and falls back to
        ``speechSynthesis`` with ``fallback`` if the MP3 is missing.
        Silently no-ops if no headset is connected (e.g. during
        startup before the operator clicks "Enter VR") -- audio cues
        are best-effort feedback, never a hard dependency.
        """
        try:
            payload = {"_type": "play_audio", "key": key}
            if fallback:
                payload["fallback"] = fallback
            self._quest.send_message(payload)
        except Exception as exc:
            # Audio is decorative; never let a send failure crash the
            # control loop. Log at debug because the most common cause
            # (no client connected yet) is expected on startup.
            log.debug("[manager-x2] audio prompt %r send failed: %s", key, exc)

    # -- sidecar --------------------------------------------------------------

    def _sidecar_emit(self, cmd: LocomotionCmd, tick: int) -> None:
        if self._sidecar is None:
            return
        rec = {
            "tick": int(tick),
            "ts": time.time(),
            "intent": cmd.intent,
            "magnitude": cmd.magnitude,
            "stream_mode": self._intent.mode.name,
        }
        self._sidecar_write(rec)

    def _sidecar_write(self, rec: dict) -> None:
        """Thread-safe append to the manager sidecar JSONL.

        Used by ``_sidecar_emit`` (called from the 50 Hz main loop) AND by
        ``_motor_monitor_loop`` (background SUB thread). Without the lock
        a half-written line from one writer can be interleaved with a
        line from the other, breaking JSONL parsers downstream.
        """
        if self._sidecar is None:
            return
        try:
            line = json.dumps(rec) + "\n"
            with self._sidecar_lock:
                self._sidecar.write(line)
        except Exception as exc:  # noqa: BLE001
            log.debug("[sidecar] write failed: %s", exc)

    # -- quest3 raw capture --------------------------------------------------

    def _defer_button_singles(self, ev, chord_a: bool, chord_x: bool,
                              now: float):
        """Hold each single-press event for 0.15s; drop all pending
        singles the moment the A+X e-stop chord (or any 2/4-button
        chord event) engages. Chord events pass through untouched."""
        import dataclasses
        pend = getattr(self, "_btn_defer", None)
        if pend is None:
            pend = self._btn_defer = {}
        for nm in ("a", "b", "x", "y"):
            if getattr(ev, f"{nm}_pressed"):
                pend[nm] = now + 0.15
        if ((chord_a and chord_x) or ev.abxy_pressed or ev.ab_pressed
                or ev.xy_pressed or ev.ax_pressed or ev.by_pressed):
            pend.clear()
        fired = [nm for nm, ft in pend.items() if ft <= now]
        for nm in fired:
            del pend[nm]
        return dataclasses.replace(
            ev,
            a_pressed="a" in fired,
            b_pressed="b" in fired,
            x_pressed="x" in fired,
            y_pressed="y" in fired,
        )

    def _quest3_raw_emit(
        self,
        *,
        tick: int,
        tick_now: float,
        axes_post_invert: tuple[float, float, float, float],
        buttons: tuple[bool, bool, bool, bool],
        triggers: tuple[float, float, float, float],
        vr_pose,
    ) -> None:
        """Append one row to the Quest3 raw capture JSONL.

        Row contract (one JSON object per line; advance the cursor by
        the recorded ``t_mono`` deltas during replay):

          * ``t_mono``  : monotonic clock at consumption time (seconds)
          * ``wall_clock``: ``time.time()`` (for human-readable correlation)
          * ``tick``    : manager loop tick index
          * ``mode``    : ``OFF`` / ``LOCOMOTION`` / ``ARM_MANIPULATION``
          * ``axes_post_invert``: ``{lx, ly, rx, ry}`` after the four
            ``invert_*`` polarity flips (i.e. what the IntentDecoder sees)
          * ``buttons``: ``{a, b, x, y}`` face buttons + ``leftTrigger /
            rightTrigger / leftGrip / rightGrip`` analog values +
            ``leftStickClick / rightStickClick``
          * ``vr_3pt_pose``: ``[[lwrist 3+4], [rwrist 3+4], [neck 3+4]]``
            or ``null`` -- the same payload ``Quest3Reader.get_3pt_pose``
            exposes, materialised so the replayer can serve it verbatim
          * ``hand_curls``: ``{left, right, left_source, right_source}``
          * ``thumb_oppose``: ``{left, right}``
          * ``finger_tip_oppose``: ``{left, right}``
          * ``quest_fps``: WS arrival rate EMA from the reader (sanity)

        We pull from the reader's getters (rather than ``get_latest``)
        so the replayer can swap the reader 1:1 without any private
        struct coupling. NaN / None entries are preserved -- the
        replayer must accept the same nulls the live path tolerates.
        """
        if self._quest3_raw_file is None:
            return
        try:
            lx, ly, rx, ry = axes_post_invert
            a, b_btn, x_btn, y_btn = buttons
            lt, rt, lg, rg = triggers
            l_click, r_click = self._quest.get_stick_clicks()
            l_curls, r_curls, l_src, r_src = self._quest.get_hand_curls()
            l_oppose, r_oppose = self._quest.get_thumb_opposition()
            l_tip, r_tip = self._quest.get_finger_tip_oppose()
            try:
                quest_fps = float(getattr(self._quest, "_fps_ema", 0.0))
            except Exception:
                quest_fps = 0.0

            def _arr_to_list(arr):
                if arr is None:
                    return None
                try:
                    return np.asarray(arr).tolist()
                except Exception:
                    return None

            rec = {
                "t_mono": float(tick_now),
                "wall_clock": time.time(),
                "tick": int(tick),
                "mode": self._intent.mode.name,
                "axes_post_invert": {
                    "lx": float(lx), "ly": float(ly),
                    "rx": float(rx), "ry": float(ry),
                },
                "buttons": {
                    "a": bool(a), "b": bool(b_btn),
                    "x": bool(x_btn), "y": bool(y_btn),
                    "leftTrigger": float(lt), "rightTrigger": float(rt),
                    "leftGrip": float(lg), "rightGrip": float(rg),
                    "leftStickClick": bool(l_click),
                    "rightStickClick": bool(r_click),
                },
                "vr_3pt_pose": _arr_to_list(vr_pose),
                "hand_curls": {
                    "left": _arr_to_list(l_curls),
                    "right": _arr_to_list(r_curls),
                    "left_source": l_src,
                    "right_source": r_src,
                },
                "thumb_oppose": {"left": l_oppose, "right": r_oppose},
                "finger_tip_oppose": {
                    "left": _arr_to_list(l_tip),
                    "right": _arr_to_list(r_tip),
                },
                "quest_fps": quest_fps,
            }
            line = json.dumps(rec) + "\n"
            with self._quest3_raw_lock:
                self._quest3_raw_file.write(line)
                self._quest3_raw_count += 1

            # Periodic row-count beacon every ~5 s so the operator can
            # tail the manager log and see capture liveness without
            # opening the JSONL. First emission is always logged so a
            # silent first 5 s does not look like a hang.
            if (self._quest3_raw_count == 1
                or (tick_now - self._quest3_raw_last_log) >= 5.0):
                log.info(
                    "[quest3-raw] %d rows captured (mode=%s, fps=%.1f)",
                    self._quest3_raw_count, self._intent.mode.name,
                    quest_fps,
                )
                self._quest3_raw_last_log = tick_now
        except Exception as exc:  # noqa: BLE001
            log.debug("[quest3-raw] emit failed: %s", exc)

    # -- resume chord (split-topology safety) --------------------------------

    def _tick_resume_chord(self, buttons: tuple[bool, bool, bool, bool],
                           now_mono: float) -> None:
        """Detect A+B held >= ``resume_chord_hold_s`` and PUB pose_resume.

        Triggered every 50 Hz tick. Disambiguation rules (chosen to avoid
        colliding with the existing A+B+X+Y "engage" chord):

          * Requires A AND B held simultaneously.
          * Requires X AND Y NOT held (the engage chord includes both).
          * Requires continuous hold for ``resume_chord_hold_s`` seconds
            before the FIRST publish lands (default 1.0 s -- longer than
            any typical button press).
          * Sustained holds republish every ``resume_chord_rep_s`` seconds
            so a brief packet drop on wifi doesn't strand the deploy in
            SAFE_IDLE waiting for a frame that already passed.

        Chord lifecycle log lines fire only on rising edge (transitioning
        from "no chord" to "chord active") and on the first successful
        publish, to keep the per-tick log noise out of the steady state.
        """
        if self._resume_sock is None:
            return
        a_held, b_held, x_held, y_held = buttons
        chord_now = a_held and b_held and (not x_held) and (not y_held)
        if not chord_now:
            self._resume_chord_active_since = None
            return
        if self._resume_chord_active_since is None:
            self._resume_chord_active_since = now_mono
            log.debug("[safety] A+B chord rising edge; counting hold...")
            return
        held = now_mono - self._resume_chord_active_since
        if held < self._cfg.resume_chord_hold_s:
            return
        if (self._resume_last_pub > 0.0
                and (now_mono - self._resume_last_pub) < self._cfg.resume_chord_rep_s):
            return
        # Publish the resume frame. Payload is the publisher-side
        # monotonic ns timestamp (cosmetic only -- the deploy stamps its
        # own steady_clock on receipt for freshness calculations, see
        # ZmqResumeSubscriber).
        try:
            ts_ns = int(time.monotonic_ns())
            payload = ts_ns.to_bytes(8, byteorder="little", signed=False)
            self._resume_sock.send_multipart(
                [self._cfg.resume_pub_topic.encode("utf-8"), payload]
            )
            self._resume_last_pub = now_mono
            self._resume_press_count += 1
            if self._resume_press_count == 1 or self._resume_press_count % 10 == 0:
                log.info(
                    "[safety] published pose_resume (press #%d, held %.2fs).",
                    self._resume_press_count, held,
                )
            self._sidecar_write({
                "ts": time.time(),
                "_kind": "resume_chord",
                "press_count": self._resume_press_count,
                "held_s": held,
            })
            # One-shot audio cue for operator feedback. Suppressed when
            # the headset isn't connected (the helper is best-effort).
            self._play_audio_prompt(
                "resume_published",
                fallback="Resume sent.",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[safety] pose_resume send failed: %s", exc)

    # -- motor monitor SUB (background JSONL appender) -----------------------

    def _motor_monitor_loop(self) -> None:
        """Background loop draining the motor_monitor SUB into the sidecar.

        Each received frame is a multipart message [topic_bytes, json_bytes]
        where ``json_bytes`` is the UTF-8 JSON the ``x2_motor_monitor.py``
        daemon emits. We decode and append it under the ``motor_monitor``
        key so a single grep "motor_monitor" across the sidecar surfaces
        all motor-side events that landed during the session.

        Errors are logged at WARN once (subsequent failures are debug-
        suppressed via ``_motor_monitor_err_logged``) -- a flaky connection
        should NOT spam the operator's terminal.
        """
        err_logged = False
        sock = self._motor_monitor_sock
        if sock is None:
            return
        # Internal poller for short non-blocking waits; lets stop() join
        # within ~200 ms even if the publisher is silent.
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                events = dict(poller.poll(200))
                if sock not in events:
                    continue
                parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) < 2:
                    continue
                topic = parts[0].decode("utf-8", errors="replace")
                payload_bytes = parts[1]
                try:
                    payload = json.loads(payload_bytes.decode("utf-8"))
                except Exception:
                    payload = {"_decode_error": True,
                               "raw_len": len(payload_bytes)}
                self._motor_monitor_msg_count += 1
                self._sidecar_write({
                    "ts": time.time(),
                    "_kind": "motor_monitor",
                    "topic": topic,
                    "motor_monitor": payload,
                })
                if self._motor_monitor_msg_count == 1:
                    log.info(
                        "[safety] motor_monitor: first message received "
                        "(topic=%s, %d bytes).",
                        topic, len(payload_bytes),
                    )
            except zmq.Again:
                continue
            except Exception as exc:  # noqa: BLE001
                if not err_logged:
                    log.warning(
                        "[safety] motor_monitor receive error: %s. "
                        "Subsequent errors suppressed.", exc,
                    )
                    err_logged = True

    # ------------------------------------------------------------------
    # Engage-pose SUB (2026-06-10 follow-up 10)
    # ------------------------------------------------------------------
    def _engage_pose_loop(self) -> None:
        """Background loop caching the latest packed-pose frame from the wire.

        The SUB connects to the proxy's downstream PUB (or any compatible
        pose stream). For every frame we decode the JSON header, extract
        the ``joint_pos_mj`` payload (plus ``left_hand_joints`` /
        ``right_hand_joints`` when present), and stash a copy under
        ``self._engage_pose_lock``. ``_resolve_engage_freeze`` consumes
        the cache on OFF -> non-OFF transitions; everything else in the
        manager ignores it.

        Errors are logged once at WARN; subsequent failures are dropped
        to avoid spamming an operator who's running with a flaky SUB.
        """
        err_logged = False
        sock = self._engage_pose_sock
        if sock is None:
            return
        topic = self._cfg.engage_pose_sub_topic or "pose"
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                events = dict(poller.poll(200))
                if sock not in events:
                    continue
                raw = sock.recv(flags=zmq.NOBLOCK)
                decoded = unpack_message(raw, expected_topic=topic)
            except zmq.Again:
                continue
            except Exception as exc:  # noqa: BLE001
                if not err_logged:
                    log.warning(
                        "[engage-pose] decode error: %s. "
                        "Subsequent errors suppressed.", exc,
                    )
                    err_logged = True
                continue
            jpos = decoded.fields.get("joint_pos_mj")
            if jpos is None or jpos.size == 0:
                continue
            left_hand = decoded.fields.get("left_hand_joints")
            right_hand = decoded.fields.get("right_hand_joints")
            now = time.time()
            with self._engage_pose_lock:
                self._engage_pose_jpos = np.asarray(jpos, dtype=np.float64).copy()
                if left_hand is not None and left_hand.size > 0:
                    self._engage_pose_left_hand = np.asarray(
                        left_hand, dtype=np.float64,
                    ).copy()
                if right_hand is not None and right_hand.size > 0:
                    self._engage_pose_right_hand = np.asarray(
                        right_hand, dtype=np.float64,
                    ).copy()
                self._engage_pose_last_ts = now
                self._engage_pose_msg_count += 1
                first = self._engage_pose_msg_count == 1
            if first:
                log.info(
                    "[engage-pose] first pose frame received "
                    "(jpos_dim=%d, has_left_hand=%s, has_right_hand=%s).",
                    int(jpos.size),
                    left_hand is not None and left_hand.size > 0,
                    right_hand is not None and right_hand.size > 0,
                )

    def _resolve_engage_freeze(
        self,
    ) -> tuple[
        Optional[np.ndarray], Optional[np.ndarray],
        Optional[np.ndarray], Optional[np.ndarray],
        str,
    ]:
        """Return (left_arm, right_arm, left_hand, right_hand, source).

        Reads the engage-pose cache and slices out the arm + hand
        sub-vectors using the canonical MJ layout (the same one the
        recorder uses to write arms back into ``body_q_mj``):
        ``left_arm = jpos[15:22]``, ``right_arm = jpos[22:29]`` (7 DOF
        each). Hands come from the dedicated ``*_hand_joints`` fields
        published alongside ``joint_pos_mj``.

        Returns all-``None`` (and ``source="neutral"``) when:
        * preserve_arms_on_engage is False,
        * no frame has been received yet,
        * the latest frame is older than ``engage_pose_sub_max_age_ms``,
        * the jpos vector is too short to slice the arm joints.

        Hands ride along with arms: when the snap-to-wire path fires,
        we always return the cached hand joints (when present). The
        operator's reasoning is "I want the robot to stay exactly
        where it is across the takeover boundary" -- preserving the
        arms but force-opening the fingers would BREAK that promise
        on every transition where VLA was holding something. A
        hand-side cache miss (e.g., the wire frame omitted the
        ``*_hand_joints`` fields) silently falls back to fingers-open
        on a per-side basis; the caller fills in zeros for any None
        returned here.

        ``source`` is ``"wire"`` on success and ``"neutral"`` otherwise;
        the caller logs both branches so the operator can tell at a
        glance whether the snap-to-wire path actually fired.
        """
        if not self._cfg.preserve_arms_on_engage:
            return None, None, None, None, "neutral"
        max_age_s = max(0.0, float(self._cfg.engage_pose_sub_max_age_ms) / 1000.0)
        with self._engage_pose_lock:
            jpos = self._engage_pose_jpos
            left_hand_cached = self._engage_pose_left_hand
            right_hand_cached = self._engage_pose_right_hand
            last_ts = self._engage_pose_last_ts
            msg_count = self._engage_pose_msg_count
        if jpos is None or msg_count == 0:
            return None, None, None, None, "neutral"
        now = time.time()
        if last_ts < 0.0 or (now - last_ts) > max_age_s:
            return None, None, None, None, "neutral"
        # Canonical MJ layout: 15..22 = left arm, 22..29 = right arm.
        if jpos.shape[0] < 29:
            return None, None, None, None, "neutral"
        left_arm = jpos[15:22].astype(np.float64, copy=True)
        right_arm = jpos[22:29].astype(np.float64, copy=True)
        left_hand = None
        right_hand = None
        if left_hand_cached is not None and left_hand_cached.size >= 10:
            left_hand = left_hand_cached[:10].astype(np.float64, copy=True)
        if right_hand_cached is not None and right_hand_cached.size >= 10:
            right_hand = right_hand_cached[:10].astype(np.float64, copy=True)
        return left_arm, right_arm, left_hand, right_hand, "wire"

    @staticmethod
    def _sleep_until(deadline_mono: float) -> None:
        rem = deadline_mono - time.monotonic()
        if rem > 0:
            time.sleep(rem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="X2 Quest 3 manager — planner_cmd + arm_targets + hand_finger_cmd",
    )
    # VR input source
    p.add_argument(
        "--input-source",
        choices=("quest3", "pico"),
        default="quest3",
        help="VR device backend: 'quest3' = WebXR app over WebSocket "
        "(default); 'pico' = XRoboToolkit PC Service via xrobotoolkit_sdk "
        "(headset + wrist controllers + ankle motion trackers; requires "
        ".venv_teleop and a running PC Service).",
    )

    # Quest 3 (ignored with --input-source pico)
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--http-port", type=int, default=8443)
    p.add_argument("--no-ssl", action="store_true", help="disable TLS for WebXR")

    # Calibration
    p.add_argument(
        "--calibration",
        type=Path,
        default=_default_calibration_path(),
        help="Operator calibration YAML (default: data/operator_calibrations/default.yaml; "
             "falls back to the shipped neutral gear_sonic/data/operator_calibrations/default.yaml)",
    )

    # Planner output
    p.add_argument("--planner-cmd-host", default="*")
    p.add_argument("--planner-cmd-port", type=int, default=5563)
    p.add_argument("--planner-cmd-topic", default="planner_cmd")
    p.add_argument(
        "--planner-cmd-connect", action="store_true",
        help="PUB-connect the planner_cmd socket instead of binding it. "
             "Dual-source (pad + VR) mode: the kplanner runs with "
             "--cmd-bind/--zmq-cmd-bind and owns the SUB bind; both this "
             "manager and pad_locomotion_bridge PUB-connect into it.",
    )
    p.add_argument(
        "--arm-connect", default="",
        help="host:port of the ROBOT planner's arm-ingest SUB "
             "(pc2_kplanner_onnx --arm-port, default port 5572). PUB-connects "
             "the arm/hand target wire out to the robot instead of binding "
             "locally — hop-in manipulation against the onboard stack.",
    )

    # Recorder output
    p.add_argument("--recorder-pub-host", default="*")
    p.add_argument("--recorder-pub-port", type=int, default=5564)

    # Cadence
    p.add_argument("--rate", type=float, default=50.0, help="Publish rate Hz")
    p.add_argument(
        "--vr-input-max-age-s",
        type=float,
        default=0.5,
        help=(
            "Safety watchdog: when the Quest 3 WebSocket goes silent "
            "(headset asleep, browser tab in background, network drop) "
            "the WebXR app stops streaming and Quest3Reader keeps "
            "returning the last cached stick values. Without this gate "
            "the manager would happily publish those stale axes "
            "forever and the robot would keep walking on a stale "
            "input. When the most recent WS packet is older than this "
            "many seconds, sticks/buttons/triggers are snapped to "
            "neutral before they reach the decoder so the published "
            "intent collapses to idle. Pair with --command-watchdog-s "
            "on x2_kplanner.py for defence in depth. Default 0.5 s "
            "matches the kplanner pose-feedback staleness gate. Set "
            "0.0 to disable (pre-2026-06-26 behaviour)."
        ),
    )

    # IntentDecoder
    p.add_argument("--stick-deadzone", type=float, default=0.30)
    p.add_argument(
        "--repeat-interval", type=float, default=0.0,
        help="If > 0, re-emit the most recent planner_cmd at this cadence (s)",
    )
    # Right-stick primitives currently authored as replay-and-snap-back
    # bins, off by default. Flip on once the planner can hold a static
    # lean / torso pose without snapping.
    p.add_argument(
        "--enable-lean-fwd", dest="enable_lean_fwd", action="store_true",
        default=False,
        help="Re-enable graded lean_fwd_{small,medium,large} on R-stick "
             "fwd push (off by default; bin currently snaps back)",
    )
    p.add_argument(
        "--enable-torso", dest="enable_torso", action="store_true",
        default=False,
        help="Re-enable torso_left_30deg / torso_right_30deg on soft "
             "R-stick L/R (off by default; bin currently snaps back)",
    )
    # Continuous-locomotion: forwards raw stick deflections as
    # ``locomotion / continuous`` instead of bucketing to fwd_step /
    # back_step / side_* / turn_*. Off by default for back-compat
    # with the heuristic planner; the kplanner wrapper flips it on.
    p.add_argument(
        "--enable-continuous-locomotion",
        dest="enable_continuous_locomotion", action="store_true",
        default=False,
        help="Emit locomotion/continuous (analog stick deflection) "
             "instead of fwd_step / back_step / side_* / turn_*. "
             "Required for analog kplanner control; the heuristic "
             "planner ignores the intent.",
    )
    p.add_argument(
        "--no-enable-continuous-locomotion",
        dest="enable_continuous_locomotion", action="store_false",
        help="Force-disable continuous-locomotion (keeps bucketed "
             "fwd_step / turn_* even when paired with the kplanner). "
             "Useful for ablation / regression runs.",
    )
    p.add_argument(
        "--continuous-yaw-max",
        dest="continuous_yaw_max", type=float, default=0.5,
        help="Maximum R-stick X amplitude forwarded as stick_yaw to "
             "the planner in continuous-locomotion mode. Range (0, "
             "1]; default 0.5 commands HALF the planner's continuous "
             "yaw ceiling at full stick. Lower for gentler turns, "
             "1.0 to restore the legacy 'full stick = full ceiling' "
             "mapping. Bucketed turn_left / turn_right are unaffected.",
    )
    # ARM_MANIPULATION L-stick decoding (v7.4): roll (lx) + continuous
    # squat / stand (ly). LOCOMOTION is unaffected by this gate.
    p.add_argument(
        "--enable-arm-man-lstick",
        dest="enable_arm_man_lstick", action="store_true",
        default=True,
        help="(default) In ARM_MANIPULATION mode, decode the L-stick "
             "as roll (lx) + continuous hip height (ly, squat / "
             "stand). LOCOMOTION mode is unaffected (L-stick still "
             "owns step / side / continuous-walk).",
    )
    p.add_argument(
        "--no-enable-arm-man-lstick",
        dest="enable_arm_man_lstick", action="store_false",
        help="Disable v7.4 ARM_MAN L-stick decoding; restores the v7.3 "
             "behaviour where only the R-stick drives lean / twist in "
             "ARM_MANIPULATION (L-stick is idle).",
    )
    p.add_argument(
        "--pitch-dominance-ratio",
        dest="pitch_dominance_ratio", type=float, default=0.4,
        help="Yaw-priority cone (R-stick): pitch is suppressed when "
             "|ry| < ratio * |rx|. Range [0, 1]; default 0.4 lets the "
             "operator twist while leaning slightly but blocks "
             "accidental lean from a yaw-intent stick wobble. 0.0 "
             "disables the cone; 1.0 is strict dominance.",
    )
    p.add_argument(
        "--height-dominance-ratio",
        dest="height_dominance_ratio", type=float, default=0.4,
        help="Roll-priority cone (L-stick, ARM_MAN): height is "
             "suppressed when |ly| < ratio * |lx|. Same range / "
             "semantics as --pitch-dominance-ratio. Default 0.4.",
    )
    p.add_argument(
        "--max-height-down-m",
        dest="max_height_down_m", type=float, default=0.09,
        help="Maximum hip-height OFFSET (downward, m) at full L-stick "
             "Y forward push (squat). Default 0.09 m matches the "
             "kplanner's in-distribution crouch envelope; pushing "
             "past this gives degraded tracking, not catastrophic "
             "failure.",
    )
    p.add_argument(
        "--max-height-up-m",
        dest="max_height_up_m", type=float, default=0.04,
        help="Maximum hip-height OFFSET (upward, m) at full L-stick "
             "Y backward push (stand-up). Default 0.04 m matches the "
             "kplanner's in-distribution stand envelope; the "
             "asymmetry vs --max-height-down-m reflects the model's "
             "training-data coverage.",
    )
    p.add_argument(
        "--hold-height-threshold-m",
        dest="hold_height_threshold_m", type=float, default=0.005,
        help="Wire-side hysteresis for the hip-height channel "
             "(metres). Decoder treats two hold_torso commands as "
             "identical when their hip_height_m targets are within "
             "this many metres on the current emit. Default 0.005 m.",
    )
    p.add_argument(
        "--loco-decoupled-arms",
        dest="loco_decoupled_arms", action="store_true",
        default=True,
        help="LOCOMOTION-mode arms behaviour. TRUE (default): manager "
             "publishes the operator's last arm pose every tick and "
             "the recorder overrides the planner's predicted arms with "
             "it; required for the ARM_MAN -> LOCOMOTION arm-hold "
             "workflow (e.g. walking while holding a tool). FALSE: "
             "in LOCOMOTION the manager flags ``passthrough_arm_targets="
             "True`` so the recorder falls through to planner-"
             "predicted arms (natural gait-coupled swing from the "
             "training corpus). ARM_MAN and OFF are unaffected. "
             "Wrapper env var: LOCO_DECOUPLED_ARMS=1/0.",
    )
    p.add_argument(
        "--no-loco-decoupled-arms",
        dest="loco_decoupled_arms", action="store_false",
        help="Opt-in: let planner-predicted arms flow through to the "
             "deploy during LOCOMOTION (whole-body walking). Breaks "
             "the manipulation arm-hold workflow; use only when you "
             "are NOT carrying a payload between manipulation spots. "
             "Same as wrapper env var LOCO_DECOUPLED_ARMS=0.",
    )

    # Stick polarity (axis sign flips applied before the decoder).
    # All default to False: the operator-facing UX of "push the stick
    # the way you want the robot to move" is achieved by the natural
    # cancellation between the Quest 3's hardware sign convention
    # (-ly = stick fwd) and the curated bin world frame (the bin
    # labelled 'back_step' actually translates the body fwd in
    # world). See the ManagerConfig.invert_ly docstring for the full
    # picture. Flip per-axis here if you re-author the bins or use a
    # different RSI source.
    inv_grp = p.add_argument_group("stick polarity")
    inv_grp.add_argument("--invert-lx", dest="invert_lx", action="store_true",
                         default=False)
    inv_grp.add_argument("--no-invert-lx", dest="invert_lx",
                         action="store_false")
    inv_grp.add_argument("--invert-ly", dest="invert_ly", action="store_true",
                         default=False)
    inv_grp.add_argument("--no-invert-ly", dest="invert_ly",
                         action="store_false")
    inv_grp.add_argument("--invert-rx", dest="invert_rx", action="store_true",
                         default=False)
    inv_grp.add_argument("--no-invert-rx", dest="invert_rx",
                         action="store_false")
    inv_grp.add_argument("--invert-ry", dest="invert_ry", action="store_true",
                         default=False)
    inv_grp.add_argument("--no-invert-ry", dest="invert_ry",
                         action="store_false")

    # Retargeter knobs (mirror the dataset recorder defaults exactly)
    p.add_argument("--ik-damping", type=float, default=0.08)
    p.add_argument("--ik-rotation-weight", type=float, default=0.3)
    p.add_argument("--ik-per-tick-step-rad", type=float, default=0.30)
    p.add_argument("--hand-input-mode", choices=("trigger", "grip", "max"),
                   default="trigger")
    p.add_argument("--apply-curl-compensation", action="store_true")
    p.add_argument("--apply-oppose-compensation", action="store_true")
    p.add_argument("--no-finger-filter", action="store_true")
    # VR wrist orientation offsets -- stop-gap until the operator
    # re-runs vr_operator_calibrate.py. Three floats (roll, pitch, yaw)
    # in DEGREES, intrinsic XYZ Tait-Bryan, applied in the operator's
    # wrist-local frame BEFORE the calibration alignment (so the IK
    # sees a corrected operator quat). Convention reminder:
    #   roll  = pronation / supination (twist about forearm long axis)
    #   pitch = flex / extend
    #   yaw   = ulnar / radial deviation
    # Defaults to (0, 0, 0) on both sides == no-op == today's behaviour.
    p.add_argument(
        "--left-wrist-offset-rpy-deg",
        dest="left_wrist_offset_rpy_deg",
        type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"),
        default=None,
        help="Operator-side wrist quat offset for the LEFT controller "
             "(degrees, intrinsic XYZ in the operator's wrist-local "
             "frame). When omitted, the value from the calibration YAML "
             "(``fit.left.op_quat_offset_rpy_deg``, populated by "
             "``vr_operator_calibrate.py``) is used; pass a tuple here "
             "to override -- including ``0 0 0`` to force identity. "
             "Only takes effect when --ik-rotation-weight > 0. Example: "
             "'--left-wrist-offset-rpy-deg 0 0 -30' compensates a left "
             "controller that sits ~30deg outward (yaw) on the cuff.",
    )
    p.add_argument(
        "--right-wrist-offset-rpy-deg",
        dest="right_wrist_offset_rpy_deg",
        type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"),
        default=None,
        help="Operator-side wrist quat offset for the RIGHT controller "
             "(see --left-wrist-offset-rpy-deg for axis convention and "
             "calibration-YAML fallback).",
    )

    # Sidecar
    p.add_argument(
        "--sidecar-log", type=Path, default=None,
        help=(
            "Path to write a JSONL sidecar of emitted planner_cmds. When "
            "--motor-monitor-host is set the sidecar also receives one "
            "line per motor_monitor message (key=motor_monitor)."
        ),
    )
    p.add_argument(
        "--quest3-record-to", dest="quest3_record_to",
        type=Path, default=None,
        help=(
            "If set, append one JSONL row per manager tick capturing the "
            "raw Quest 3 inputs the manager consumed (post-invert axes, "
            "buttons, 3pt-pose, hand curls). Used as a replayable "
            "fixture by Quest3Replayer so we can sweep input-smoothing "
            "knobs headset-free against an identical operator stream. "
            "Default off."
        ),
    )

    # --- StickFilter (continuous-locomotion stick smoothing) -----------
    # The kplanner + SONIC pair was trained on smooth human mocap; raw
    # Quest 3 thumbstick step inputs put the model out of distribution
    # and cause forward-walk lurches. These flags expose the (tau, slew)
    # knobs that bring the live VR p99 |d(vel_z)/dt| into the PKL
    # training band. See ``docs/source/references/x2_quest3_stick_smoothing.md``.
    p.add_argument(
        "--stick-lpf-tau", type=float, default=0.0,
        help=(
            "First-order LPF time constant (s) applied uniformly to "
            "(stick_fwd, stick_side, stick_yaw) BEFORE the IntentDecoder "
            "deadzone, only while in LOCOMOTION mode. 0.0 = no LPF "
            "(default; legacy behaviour). Tuned default from the "
            "offline sweep: 0.10 s. Values above ~0.30 s introduce "
            "operator-noticeable lag."
        ),
    )
    p.add_argument(
        "--stick-slew-max", type=float, default=float("inf"),
        help=(
            "Slew-rate cap (stick-units/s) applied per-channel before "
            "the LPF. ``inf`` (default) disables. Useful as a backstop "
            "when an LPF tau alone is not enough to cap the d/dt; the "
            "offline sweep recommends ``inf`` (LPF-only) for the "
            "currently-tuned defaults."
        ),
    )
    p.add_argument(
        "--stick-return-tau", type=float, default=0.0,
        help=(
            "Optional asymmetric release LPF tau (s). When > 0, the "
            "filter uses this tau when the operator is releasing the "
            "stick toward zero (vs the engaged ``--stick-lpf-tau`` "
            "when the stick is being pushed away from zero). Useful "
            "for ``snappy push, gentle release`` operator-feel."
        ),
    )

    # Split-topology safety (Phase 3 + 5c)
    sft_grp = p.add_argument_group("split-topology safety (Phase 3 + 5c)")
    sft_grp.add_argument(
        "--resume-pub-enabled", dest="resume_pub_enabled",
        action="store_true", default=False,
        help=(
            "Bind a PUB socket on tcp://<--resume-pub-host>:<--resume-pub-port> "
            "and publish a pose_resume frame whenever the operator holds A+B "
            "(without X+Y) for >= --resume-chord-hold-s seconds. The deploy on "
            "PC2 SUBs to this to exit SAFE_IDLE. Default OFF; the wrapper "
            "(run_x2_quest3_planner_stack.sh --remote-deploy) flips this on "
            "automatically."
        ),
    )
    sft_grp.add_argument("--resume-pub-host", default="*")
    sft_grp.add_argument("--resume-pub-port", type=int, default=5566)
    sft_grp.add_argument("--resume-pub-topic", default="pose_resume")
    sft_grp.add_argument(
        "--resume-chord-hold-s", type=float, default=1.0,
        help="Continuous A+B hold (s) before the first pose_resume publish.",
    )
    sft_grp.add_argument(
        "--resume-chord-rep-s", type=float, default=0.5,
        help=(
            "Min interval (s) between republished pose_resume frames during a "
            "sustained chord hold. Compensates for ZMQ frame loss on flaky wifi."
        ),
    )
    sft_grp.add_argument(
        "--motor-monitor-host", dest="motor_monitor_sub_host", default="",
        help=(
            "If set, SUB to tcp://<host>:5567 topic 'motor_monitor' for the "
            "PC2 x2_motor_monitor daemon's compact summaries. Each message is "
            "decoded as JSON and appended to --sidecar-log under the "
            "'motor_monitor' key. Empty string disables (default)."
        ),
    )
    sft_grp.add_argument(
        "--motor-monitor-port", dest="motor_monitor_sub_port",
        type=int, default=5567,
    )
    sft_grp.add_argument(
        "--motor-monitor-topic", dest="motor_monitor_sub_topic",
        default="motor_monitor",
    )

    # --- Engage-pose preservation (2026-06-10 follow-up 10) ----------
    # Solves the "operator activates teleop mid-VLA-run and the arms
    # snap to default" UX. One boolean flag turns it on; the SUB
    # defaults to the canonical proxy-downstream wire on loopback
    # (tcp://127.0.0.1:5558), so the common SIM-on-PC1 case only
    # needs --preserve-arms-on-engage. Advanced overrides (host /
    # port / topic / freshness window) live in the same group but
    # rarely need to be touched.
    eng_grp = p.add_argument_group("engage-pose preservation")
    eng_grp.add_argument(
        "--preserve-arms-on-engage", dest="preserve_arms_on_engage",
        action="store_true", default=False,
        help=(
            "On every OFF -> non-OFF (A+B+X+Y chord) snap the arm + "
            "hand freeze to the robot's CURRENT commanded pose "
            "instead of X2 neutral / fingers-open. Use this when the "
            "operator takes over a live VLA run so the robot stays "
            "exactly where it is across the boundary (no arm drift "
            "to a parking pose, no surprise drop / squeeze if VLA "
            "was holding something). Hands ride along with arms; the "
            "operator can still open fingers normally via the VR "
            "trigger once engaged. SUBs to the proxy's downstream "
            "wire at tcp://127.0.0.1:5558 by default; pass "
            "--engage-pose-sub-host / --engage-pose-sub-port to "
            "override for split-topology / non-default ports."
        ),
    )
    eng_grp.add_argument(
        "--engage-pose-sub-host", dest="engage_pose_sub_host",
        default="127.0.0.1",
        help=(
            "Host the engage-pose SUB connects to. Default 127.0.0.1 "
            "for the SIM-on-one-machine setup; pass PC2's IP for "
            "split-topology / real-robot. No effect unless "
            "--preserve-arms-on-engage is set."
        ),
    )
    eng_grp.add_argument(
        "--engage-pose-sub-port", dest="engage_pose_sub_port",
        type=int, default=5558,
        help=(
            "Port the engage-pose SUB connects to. Default 5558 = the "
            "x2_pose_proxy downstream PUB the deploy listens on; pass "
            "an override only if the proxy was started with a "
            "non-default --downstream-port. No effect unless "
            "--preserve-arms-on-engage is set."
        ),
    )
    eng_grp.add_argument(
        "--engage-pose-sub-topic", dest="engage_pose_sub_topic",
        default="pose",
        help=(
            "ZMQ topic name on the engage-pose SUB. Default 'pose'. "
            "No effect unless --preserve-arms-on-engage is set."
        ),
    )
    eng_grp.add_argument(
        "--engage-pose-sub-max-age-ms", dest="engage_pose_sub_max_age_ms",
        type=int, default=200,
        help=(
            "Max age (ms) for a cached pose frame to count as 'fresh' "
            "for the OFF -> non-OFF snap. Older frames trigger the "
            "neutral-snap fallback. Default 200 ms (10 ticks @ 50 Hz)."
        ),
    )

    # Episode lifecycle audio cues
    rec_grp = p.add_argument_group("recorder audio cues")
    rec_grp.add_argument(
        "--recorder-enabled", dest="recorder_enabled",
        action="store_true", default=False,
        help=(
            "Play the 'Recording.' / 'Saved.' headset audio cues on "
            "X / Y press in ARM_MANIPULATION. Set this iff the "
            "downstream record_x2_dataset is running with "
            "--output-dir (i.e. NOT --teleop-only); the wrapper "
            "(run_x2_quest3_planner_stack.sh) sets it automatically "
            "when --with-record is passed. The recorder_cmd ZMQ "
            "message is published either way; this only gates the "
            "audio path so the operator doesn't get a false 'Saved.' "
            "ACK while no parquet is being written."
        ),
    )
    rec_grp.add_argument(
        "--no-recorder-enabled", dest="recorder_enabled",
        action="store_false",
        help="Suppress the X/Y audio cues (default; teleop-only safe).",
    )

    # Camera cycler (xdotool path; TODO replace with vr_input topic)
    cam_grp = p.add_argument_group("viewer camera cycler")
    cam_grp.add_argument(
        "--no-viewer-camera-cycler", dest="enable_viewer_camera_cycler",
        action="store_false", default=True,
        help=(
            "Disable the left-stick-click -> ']' camera cycler. "
            "Useful for headless / CI runs where no GLFW window exists."
        ),
    )
    cam_grp.add_argument(
        "--viewer-window-pattern", default="MuJoCo",
        help=(
            "xdotool search --name pattern -- FALLBACK only, used when "
            "the --classname search returns nothing. Override if your "
            "deploy build sets a non-default window title."
        ),
    )
    cam_grp.add_argument(
        "--viewer-window-classname", default="MuJoCo",
        help=(
            "xdotool search --classname pattern -- the PRIMARY way to "
            "find the deploy viewer window (avoids GNOME mutter's "
            "frame-wrapper that swallows synthetic key events). "
            "Only override if you've rebuilt MuJoCo with a custom "
            "WM_CLASS via glfwWindowHintString."
        ),
    )

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )

    # Build the optional StickFilter config from CLI flags. Uniform
    # tau/slew across the three channels for now; per-channel tuning
    # can land later if the offline sweep ever shows asymmetric needs.
    # ``--stick-lpf-tau 0`` with ``--stick-slew-max inf`` collapses to
    # the no-op identity filter so we don't pay the per-tick overhead
    # in the legacy code path.
    stick_filter_cfg: Optional[StickFilterConfig] = None
    if args.stick_lpf_tau > 0.0 or args.stick_slew_max != float("inf"):
        return_tau = args.stick_return_tau if args.stick_return_tau > 0 else None
        stick_filter_cfg = StickFilterConfig(
            tau_lpf_fwd_s=args.stick_lpf_tau,
            tau_lpf_side_s=args.stick_lpf_tau,
            tau_lpf_yaw_s=args.stick_lpf_tau,
            slew_max_fwd_per_s=args.stick_slew_max,
            slew_max_side_per_s=args.stick_slew_max,
            slew_max_yaw_per_s=args.stick_slew_max,
            return_to_zero_tau_fwd_s=return_tau,
            return_to_zero_tau_side_s=return_tau,
            return_to_zero_tau_yaw_s=return_tau,
        )

    cfg = ManagerConfig(
        input_source=args.input_source,
        quest3_ws_port=args.ws_port,
        quest3_http_port=args.http_port,
        quest3_use_ssl=not args.no_ssl,
        calibration_path=args.calibration,
        publish_rate_hz=args.rate,
        vr_input_max_age_s=args.vr_input_max_age_s,
        planner_cmd_host=args.planner_cmd_host,
        planner_cmd_port=args.planner_cmd_port,
        planner_cmd_topic=args.planner_cmd_topic,
        planner_cmd_connect=args.planner_cmd_connect,
        arm_connect=args.arm_connect,
        recorder_pub_host=args.recorder_pub_host,
        recorder_pub_port=args.recorder_pub_port,
        intent_stick_deadzone=args.stick_deadzone,
        intent_repeat_interval_s=args.repeat_interval,
        intent_enable_lean_fwd=args.enable_lean_fwd,
        intent_enable_torso=args.enable_torso,
        intent_enable_continuous_locomotion=args.enable_continuous_locomotion,
        intent_continuous_yaw_max=args.continuous_yaw_max,
        intent_enable_arm_man_lstick=args.enable_arm_man_lstick,
        intent_pitch_dominance_ratio=args.pitch_dominance_ratio,
        intent_height_dominance_ratio=args.height_dominance_ratio,
        intent_max_height_down_m=args.max_height_down_m,
        intent_max_height_up_m=args.max_height_up_m,
        intent_hold_height_threshold_m=args.hold_height_threshold_m,
        intent_loco_decoupled_arms=args.loco_decoupled_arms,
        invert_lx=args.invert_lx,
        invert_ly=args.invert_ly,
        invert_rx=args.invert_rx,
        invert_ry=args.invert_ry,
        enable_viewer_camera_cycler=args.enable_viewer_camera_cycler,
        viewer_window_pattern=args.viewer_window_pattern,
        viewer_window_classname=args.viewer_window_classname,
        ik_damping=args.ik_damping,
        ik_rotation_weight=args.ik_rotation_weight,
        ik_per_tick_step_rad=args.ik_per_tick_step_rad,
        hand_input_mode=args.hand_input_mode,
        apply_curl_compensation=args.apply_curl_compensation,
        apply_oppose_compensation=args.apply_oppose_compensation,
        enable_finger_filter=not args.no_finger_filter,
        left_wrist_op_quat_offset_rpy_deg=(
            tuple(float(v) for v in args.left_wrist_offset_rpy_deg)
            if args.left_wrist_offset_rpy_deg is not None
            else None
        ),
        right_wrist_op_quat_offset_rpy_deg=(
            tuple(float(v) for v in args.right_wrist_offset_rpy_deg)
            if args.right_wrist_offset_rpy_deg is not None
            else None
        ),
        sidecar_log_path=args.sidecar_log,
        quest3_raw_log_path=args.quest3_record_to,
        recorder_enabled=args.recorder_enabled,
        resume_pub_enabled=args.resume_pub_enabled,
        resume_pub_host=args.resume_pub_host,
        resume_pub_port=args.resume_pub_port,
        resume_pub_topic=args.resume_pub_topic,
        resume_chord_hold_s=args.resume_chord_hold_s,
        resume_chord_rep_s=args.resume_chord_rep_s,
        motor_monitor_sub_enabled=bool(args.motor_monitor_sub_host),
        motor_monitor_sub_host=args.motor_monitor_sub_host,
        motor_monitor_sub_port=args.motor_monitor_sub_port,
        motor_monitor_sub_topic=args.motor_monitor_sub_topic,
        preserve_arms_on_engage=bool(args.preserve_arms_on_engage),
        engage_pose_sub_host=args.engage_pose_sub_host,
        engage_pose_sub_port=int(args.engage_pose_sub_port),
        engage_pose_sub_topic=args.engage_pose_sub_topic,
        engage_pose_sub_max_age_ms=int(args.engage_pose_sub_max_age_ms),
        stick_filter_config=stick_filter_cfg,
        verbose=args.verbose,
    )

    mgr = Quest3ManagerX2(cfg)

    # Graceful shutdown on SIGINT / SIGTERM.
    def _sig_handler(signum, _frame):
        log.info("[manager-x2] signal %d received; stopping", signum)
        mgr._stop.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    mgr.start()
    n_ticks = mgr.run()
    log.info("[manager-x2] shutdown complete (%d ticks)", n_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
