#!/usr/bin/env python3
"""x2_pose_mux.py -- laptop-side N-to-1 pose multiplexer.

Sits between two pose sources colocated on the laptop:

* the VLA bridge (``live_vla_publish_motion_token.py``) publishing
  autonomous poses on loopback;
* the operator override stream (``record_x2_dataset`` driven by the
  Quest 3 manager) publishing teleop poses on a second loopback port.

The mux arbitrates between them (engage / release with hysteresis +
optional stream-mode strict gate), applies the engagement slow-step
ramp on the LIVE -> OVERRIDE edge, and PUBs the merged wire to the
PC2 ``x2_pose_watchdog`` over wifi. The watchdog forwards bytes
verbatim while we're alive, and falls back to its baked idle clip
ladder only if the laptop or wifi link dies.

This process is owned by ``run_x2_vla_runtime.sh`` (its lifetime ==
the VLA runtime's lifetime). It is NOT spawned by the Quest stack
launcher or the PC2 daemons script. When you're not running VLA, the
mux doesn't exist and the recorder publishes directly to the PC2
watchdog like any single-source deployment.

Architecture:

    LAPTOP                                            PC2
    ------                                            ---
    bridge      ----> mux  =======wifi=======>  pose_watchdog  --> deploy
    (loopback)        (this script)             (fallback only)
                       ^
    recorder --------- |  (operator override; loopback)
                       |
                       +---> vla_control PUB (loopback to bridge,
                              edge events for cold-restart on release)

Why the merge lives on the laptop instead of PC2:

* Crash isolation: any bug in the merge logic (the code that changes
  most) cannot crash the PC2 ``pose_watchdog`` that holds the safety
  fallback.
* Locality: both sources are on the laptop, so merging here means
  operator pose stops crossing wifi -- only the merged wire does.
* Unification: the sim path was already laptop-spawned; this makes
  the real path use the same pattern so sim and real share one
  topology.
* Scaling: the mux is architected for N sources today (primary +
  override). Adding a third source later (e.g. gesture-during-VLA)
  is a single SUB plus an arbitration-priority entry, all on the
  laptop without touching PC2.

Single-thread design: one ``zmq.Context``, two SUBs (primary +
override), one PUB (to watchdog), one optional PUB (vla_control for
bridge), one optional SUB (stream_mode for strict engage gate). The
50 Hz tick loop polls all SUBs non-blockingly, drains the queues
(keeps only the latest frame from each), runs the arbiter, and
forwards the winning bytes verbatim (or with a clamped jpos splice
during the engagement ramp). PUB-SUB is inherently lossy on slow
consumers; we set RCVHWM=SNDHWM=100 to keep buffer pressure bounded.

Dependencies: numpy, pyzmq, gear_sonic.utils.pose_pipeline. msgpack
is optional (only needed for the stream_mode strict engage gate; the
legacy motion-hysteresis path works without it).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Repo-root path injection so the script is runnable standalone
# (`python gear_sonic/scripts/x2_pose_mux.py …`) outside of a venv
# install. Mirrors what other top-level scripts in this repo do.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import zmq

from gear_sonic.utils.pose_pipeline.arbitrate import (
    ArbiterConfig,
    EDGE_ENGAGED,
    EDGE_RELEASED,
    SOURCE_OVERRIDE,
    SOURCE_PRIMARY,
    TakeoverArbiter,
)
from gear_sonic.utils.pose_pipeline.wire import (
    DEFAULT_PUB_RATE_HZ,
)

try:
    import msgpack as _msgpack
    _HAS_MSGPACK = True
except ImportError:
    _msgpack = None  # type: ignore[assignment]
    _HAS_MSGPACK = False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)

    # ----- Primary SUB (VLA bridge on laptop loopback) -----------------
    p.add_argument(
        "--primary-host",
        default="127.0.0.1",
        help="Host the VLA bridge publishes to (default 127.0.0.1).",
    )
    p.add_argument(
        "--primary-port",
        type=int,
        required=True,
        help="Port the VLA bridge publishes to (typically 5556 in sim "
             "or whatever the bridge's --pose-port is bound to).",
    )
    p.add_argument(
        "--primary-topic",
        default="pose",
        help="ZMQ topic prefix on the primary SUB (default 'pose').",
    )

    # ----- Override SUB (recorder on laptop loopback) ------------------
    p.add_argument(
        "--override-host",
        default="127.0.0.1",
        help="Host the recorder publishes the operator-override pose to "
             "(default 127.0.0.1; loopback when colocated on laptop).",
    )
    p.add_argument(
        "--override-port",
        type=int,
        required=True,
        help="Port the recorder publishes operator-override pose to "
             "(typically 5560).",
    )
    p.add_argument(
        "--override-topic",
        default="pose",
        help="Topic prefix on the override SUB (default 'pose').",
    )
    p.add_argument(
        "--override-stale-ms",
        type=int,
        default=200,
        help="Treat the override source as gone after this many ms of "
             "silence (default 200 = 10 ticks @ 50Hz). Acts as a "
             "debounce so a single dropped teleop frame doesn't flip "
             "the mux back to primary and trigger a spurious bridge "
             "cold-restart.",
    )

    # ----- Merged-output PUB (to PC2 watchdog over wifi) ---------------
    p.add_argument(
        "--out-host",
        default="*",
        help="Local bind iface for the merged-output PUB "
             "(default '*' = all; bind 0.0.0.0 explicitly if needed).",
    )
    p.add_argument(
        "--out-port",
        type=int,
        required=True,
        help="Port the merged-output PUB binds to. PC2 watchdog SUBs "
             "here (typically 5556 over wifi).",
    )
    p.add_argument(
        "--out-topic",
        default="pose",
        help="Topic prefix on the merged-output PUB (default 'pose'; "
             "must match the watchdog's --upstream-topic).",
    )

    # ----- vla_control PUB (loopback to bridge) ------------------------
    p.add_argument(
        "--vla-control-bind-host",
        default="127.0.0.1",
        help="Bind interface for the vla_control PUB (default "
             "127.0.0.1; bridge is colocated on the laptop).",
    )
    p.add_argument(
        "--vla-control-port",
        type=int,
        default=-1,
        help="Bind port for the vla_control edge-event PUB. Set to a "
             "positive int (e.g. 5559) to enable the bridge's cold-"
             "restart-on-release feedback loop. Default -1 = DISABLED "
             "(release edges still fire internally but nothing "
             "downstream reacts).",
    )
    p.add_argument(
        "--vla-control-topic",
        default="vla_control",
        help="Topic prefix on the vla_control PUB (default "
             "'vla_control').",
    )

    # ----- Frozen-frame release detection ------------------------------
    p.add_argument(
        "--override-frozen-ticks",
        type=int,
        default=10,
        help="Fire override_released after this many consecutive "
             "override frames within --override-frozen-l2-tol of the "
             "previous one (default 10 = 200ms @ 50Hz, matching "
             "--override-stale-ms semantics). The Quest3 manager "
             "publishes a frozen pose every tick in OFF/LOCOMOTION "
             "mode, so this is what actually fires release after the "
             "operator hits the A+B+X+Y disengage chord. Set to 0 to "
             "disable and rely on silence-based release only.",
    )
    p.add_argument(
        "--override-frozen-l2-tol",
        type=float,
        default=5e-3,
        help="L2 distance tolerance (rad) for two override frames to "
             "be considered 'frozen' (default 5e-3 ~ 0.3 deg of total "
             "joint-space motion). Small enough that any intentional "
             "teleop push trips above tol within one tick; large "
             "enough to absorb controller-rest jitter and IK "
             "retargeting flicker. Lower (e.g. 1e-4) for strict bytes-"
             "match detection only.",
    )
    p.add_argument(
        "--override-engage-motion-ticks",
        type=int,
        default=10,
        help="Require this many consecutive override frames with L2 "
             "delta ABOVE --override-frozen-l2-tol before firing "
             "override_engaged (default 10 = 200ms @ 50Hz, symmetric "
             "with --override-frozen-ticks). Prevents brief jitter "
             "from spurious engage/release cycles (each cycle "
             "triggers a heavy bridge cold-restart). Set to 0 for "
             "the legacy single-frame-engage behaviour (older smoke "
             "tests only).",
    )

    # ----- Engagement slow-step ramp -----------------------------------
    p.add_argument(
        "--engagement-max-wire-step",
        type=float,
        default=0.012,
        help="Per-element max joint-position step (rad) applied to "
             "the override frames forwarded right AFTER the LIVE -> "
             "OVERRIDE transition. Default 0.012 rad/tick (~36 deg/s "
             "per joint at 50 Hz). Without this clamp the operator's "
             "first OVERRIDE frame can step the wire ~3 rad away "
             "from VLA's last command in one tick and the deploy "
             "slams the body across the delta.",
    )
    p.add_argument(
        "--engagement-steady-wire-step",
        type=float,
        default=0.035,
        help="Per-element steady-state max joint-position step (rad) "
             "the engagement ramp converges to. Default 0.035 rad/"
             "tick (~100 deg/s per joint at 50 Hz, matches the "
             "bridge's --vla-max-wire-step default). After the "
             "engagement ramp completes the mux stops clamping "
             "override frames entirely.",
    )
    p.add_argument(
        "--engagement-step-ramp-ticks",
        type=int,
        default=250,
        help="Number of ticks over which to linearly ramp the "
             "engagement rate clamp from --engagement-max-wire-step "
             "(slow, applied at engagement) to --engagement-steady-"
             "wire-step (normal). Default 250 @ 50Hz = 5.0 s. Set "
             "to 0 to disable engagement clamping (operator frames "
             "forwarded verbatim from the very first OVERRIDE tick).",
    )

    # ----- Stream-mode strict engage gate (optional) -------------------
    p.add_argument(
        "--teleop-mode-host",
        default="127.0.0.1",
        help="Host where the manager's stream_mode PUB lives "
             "(default 127.0.0.1; colocated on the laptop with the "
             "mux + recorder).",
    )
    p.add_argument(
        "--teleop-mode-port",
        type=int,
        default=-1,
        help="Port of the manager's stream_mode PUB. The manager's "
             "default --recorder-pub-port is 5564. Set to a positive "
             "int to enable mode-gated engagement; set to -1 (default) "
             "to fall back to motion-hysteresis. STRICT MODE: when "
             "enabled and the mode signal goes stale "
             "(--teleop-mode-stale-ms), engagement is BLOCKED.",
    )
    p.add_argument(
        "--teleop-mode-topic",
        default="stream_mode",
        help="Topic prefix on the manager's PUB (default "
             "'stream_mode').",
    )
    p.add_argument(
        "--teleop-mode-stale-ms",
        type=int,
        default=1000,
        help="Treat the mode signal as gone after this many ms of "
             "silence (default 1000 = 50 ticks @ 50Hz). When stale, "
             "engagement is BLOCKED in strict mode.",
    )

    # ----- Loop knobs --------------------------------------------------
    p.add_argument(
        "--rate-hz",
        type=float,
        default=DEFAULT_PUB_RATE_HZ,
        help="Mux tick rate (default 50; matches deploy control loop). "
             "Forwarding is event-driven and inherits whatever cadence "
             "the source publishes at; this knob only governs the "
             "no-fresh-input idle.",
    )
    p.add_argument(
        "--status-every-s",
        type=float,
        default=5.0,
        help="Periodic status print interval (default 5s).",
    )

    args = p.parse_args(argv)

    # ----- Pre-flight knob validation ---------------------------------
    if int(args.teleop_mode_port) > 0 and not _HAS_MSGPACK:
        print(
            "[pose_mux] WARN: --teleop-mode-port is set but msgpack "
            "is not installed in this venv; falling back to legacy "
            "motion-hysteresis engagement (pip install msgpack to "
            "enable strict mode-gated engagement).",
            flush=True,
        )

    teleop_mode_enabled = int(args.teleop_mode_port) > 0 and _HAS_MSGPACK

    cfg = ArbiterConfig(
        upstream_topic=args.primary_topic,
        downstream_topic=args.out_topic,
        override_stale_s=max(args.override_stale_ms, 1) / 1000.0,
        frozen_ticks_threshold=max(int(args.override_frozen_ticks), 0),
        frozen_l2_tol=max(float(args.override_frozen_l2_tol), 0.0),
        engage_motion_threshold=max(
            int(args.override_engage_motion_ticks), 0
        ),
        teleop_mode_enabled=teleop_mode_enabled,
        teleop_mode_stale_s=(
            max(args.teleop_mode_stale_ms, 1) / 1000.0
        ),
        engagement_max_wire_step=max(
            float(args.engagement_max_wire_step), 0.0
        ),
        engagement_steady_wire_step=max(
            float(args.engagement_steady_wire_step), 0.0
        ),
        engagement_step_ramp_ticks=max(
            int(args.engagement_step_ramp_ticks), 0
        ),
    )
    arb = TakeoverArbiter(cfg)

    # ----- ZMQ wiring --------------------------------------------------
    ctx = zmq.Context.instance()

    primary_sub = ctx.socket(zmq.SUB)
    primary_sub.setsockopt(zmq.RCVHWM, 100)
    primary_url = f"tcp://{args.primary_host}:{args.primary_port}"
    primary_sub.connect(primary_url)
    primary_sub.setsockopt(
        zmq.SUBSCRIBE, args.primary_topic.encode("utf-8")
    )

    override_sub = ctx.socket(zmq.SUB)
    override_sub.setsockopt(zmq.RCVHWM, 100)
    override_url = f"tcp://{args.override_host}:{args.override_port}"
    override_sub.connect(override_url)
    override_sub.setsockopt(
        zmq.SUBSCRIBE, args.override_topic.encode("utf-8")
    )

    out_pub = ctx.socket(zmq.PUB)
    out_pub.setsockopt(zmq.SNDHWM, 100)
    out_url = f"tcp://{args.out_host}:{args.out_port}"
    out_pub.bind(out_url)

    vla_control_pub: zmq.Socket | None = None
    vla_control_enabled = int(args.vla_control_port) > 0
    if vla_control_enabled:
        vla_control_pub = ctx.socket(zmq.PUB)
        vla_control_pub.setsockopt(zmq.SNDHWM, 32)
        vla_control_url = (
            f"tcp://{args.vla_control_bind_host}:{args.vla_control_port}"
        )
        vla_control_pub.bind(vla_control_url)

    teleop_mode_sub: zmq.Socket | None = None
    if teleop_mode_enabled:
        teleop_mode_sub = ctx.socket(zmq.SUB)
        teleop_mode_sub.setsockopt(zmq.RCVHWM, 32)
        teleop_mode_url = (
            f"tcp://{args.teleop_mode_host}:{args.teleop_mode_port}"
        )
        teleop_mode_sub.connect(teleop_mode_url)
        teleop_mode_sub.setsockopt(
            zmq.SUBSCRIBE, args.teleop_mode_topic.encode("utf-8")
        )

    # ----- Startup banner ---------------------------------------------
    print(
        f"[pose_mux] primary SUB:    {primary_url} "
        f"topic={args.primary_topic!r}",
        flush=True,
    )
    print(
        f"[pose_mux] override SUB:   {override_url} "
        f"topic={args.override_topic!r} "
        f"(stale_ms={args.override_stale_ms} "
        f"frozen_ticks={cfg.frozen_ticks_threshold} "
        f"frozen_l2_tol={cfg.frozen_l2_tol:g} "
        f"engage_motion_ticks={cfg.engage_motion_threshold})",
        flush=True,
    )
    print(
        f"[pose_mux] out PUB:        {out_url} topic={args.out_topic!r}",
        flush=True,
    )
    if vla_control_enabled:
        print(
            f"[pose_mux] vla_control PUB: "
            f"tcp://{args.vla_control_bind_host}:{args.vla_control_port} "
            f"topic={args.vla_control_topic!r}",
            flush=True,
        )
    else:
        print(
            "[pose_mux] vla_control PUB: DISABLED "
            "(--vla-control-port not set; bridge won't cold-restart "
            "automatically on override release)",
            flush=True,
        )
    if teleop_mode_enabled:
        print(
            f"[pose_mux] teleop_mode SUB: "
            f"{teleop_mode_url} "
            f"topic={args.teleop_mode_topic!r} "
            f"(stale_ms={args.teleop_mode_stale_ms}) "
            f"-- STRICT mode-gated engage (motion-hysteresis bypassed)",
            flush=True,
        )
    else:
        print(
            "[pose_mux] teleop_mode SUB: DISABLED "
            "(--teleop-mode-port not set; legacy motion-hysteresis "
            "engage path -- will flicker if operator holds the "
            "controller still in ARM_MANIPULATION)",
            flush=True,
        )
    print(
        f"[pose_mux] engagement ramp: max_step={cfg.engagement_max_wire_step:.3f} -> "
        f"steady={cfg.engagement_steady_wire_step:.3f} rad/tick over "
        f"{cfg.engagement_step_ramp_ticks} ticks",
        flush=True,
    )

    period = 1.0 / max(args.rate_hz, 1e-6)
    primary_stale_s = cfg.override_stale_s  # share debounce for symmetry
    next_tick = time.monotonic()
    last_primary_s = -1.0
    last_primary_msg: bytes | None = None
    tick = 0
    primary_fwd = 0
    last_status_s = time.monotonic()

    print(
        "[pose_mux] starting (will forward primary frames as soon as "
        "the bridge publishes; engagement waits for override hysteresis)",
        flush=True,
    )

    try:
        while True:
            now = time.monotonic()

            # ----- Drain teleop_mode SUB -----------------------------
            if teleop_mode_sub is not None and _msgpack is not None:
                latest_mode_msg: list[bytes] | None = None
                while True:
                    try:
                        latest_mode_msg = teleop_mode_sub.recv_multipart(
                            zmq.NOBLOCK
                        )
                    except zmq.Again:
                        break
                if latest_mode_msg is not None:
                    payload_bytes: bytes | None = None
                    if len(latest_mode_msg) >= 2:
                        payload_bytes = latest_mode_msg[1]
                    elif len(latest_mode_msg) == 1:
                        payload_bytes = latest_mode_msg[0]
                    decoded_mode: str | None = None
                    if payload_bytes is not None:
                        try:
                            payload = _msgpack.unpackb(
                                payload_bytes, raw=False
                            )
                            if isinstance(payload, dict):
                                m = payload.get("mode")
                                if isinstance(m, str):
                                    decoded_mode = m
                        except Exception:
                            decoded_mode = None
                    if decoded_mode is not None:
                        arb.observe_teleop_mode(decoded_mode, now=now)
                    else:
                        arb.record_teleop_mode_decode_failure()

            # ----- Drain override SUB --------------------------------
            latest_override = None
            while True:
                try:
                    latest_override = override_sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break

            # ----- Drain primary SUB ---------------------------------
            latest_primary = None
            while True:
                try:
                    latest_primary = primary_sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break

            if latest_primary is not None:
                last_primary_msg = latest_primary
                last_primary_s = now
                arb.observe_primary(latest_primary, now=now)
            if latest_override is not None:
                arb.observe_override(latest_override, now=now)

            primary_fresh = (
                last_primary_s >= 0
                and (now - last_primary_s) <= primary_stale_s
            )

            decision = arb.decide(
                now=now,
                tick=tick,
                primary_fresh=primary_fresh,
                override_recvd_this_tick=(latest_override is not None),
            )

            # ----- Emit edge events on vla_control --------------------
            if vla_control_pub is not None:
                if decision.edge == EDGE_ENGAGED \
                        and decision.engage_event_payload is not None:
                    try:
                        vla_control_pub.send_multipart(
                            [args.vla_control_topic.encode("utf-8"),
                             decision.engage_event_payload],
                            zmq.NOBLOCK,
                        )
                    except zmq.Again:
                        pass
                elif decision.edge == EDGE_RELEASED \
                        and decision.release_event_payload is not None:
                    try:
                        vla_control_pub.send_multipart(
                            [args.vla_control_topic.encode("utf-8"),
                             decision.release_event_payload],
                            zmq.NOBLOCK,
                        )
                    except zmq.Again:
                        pass

            # ----- Forward bytes -------------------------------------
            if decision.source == SOURCE_OVERRIDE and latest_override is not None:
                fwd_msg, _ = arb.maybe_clamp_override(latest_override)
                try:
                    out_pub.send(fwd_msg, zmq.NOBLOCK)
                    arb.record_forwarded_override()
                except zmq.Again:
                    pass
            elif decision.source == SOURCE_PRIMARY and latest_primary is not None:
                try:
                    out_pub.send(latest_primary, zmq.NOBLOCK)
                    primary_fwd += 1
                except zmq.Again:
                    pass
            # Source == NEITHER: don't publish. The PC2 watchdog
            # owns the fallback ladder; sending nothing from here just
            # tells the watchdog "no fresh upstream this tick" and it
            # runs its GAP/HOLD/BLEND/IDLE_CLIP machine.

            # ----- Edge log line --------------------------------------
            if decision.edge == EDGE_ENGAGED:
                print(
                    f"[pose_mux] OVERRIDE engaged "
                    f"(slow-step ramp armed window="
                    f"{cfg.engagement_step_ramp_ticks} ticks; "
                    f"max_step {cfg.engagement_max_wire_step:.3f} -> "
                    f"{cfg.engagement_steady_wire_step:.3f} rad/tick)",
                    flush=True,
                )
            elif decision.edge == EDGE_RELEASED:
                print(
                    "[pose_mux] OVERRIDE released "
                    "(bridge cold-restart event sent on vla_control)",
                    flush=True,
                )

            tick += 1

            if now - last_status_s >= args.status_every_s:
                if last_primary_s < 0:
                    pri_age_str = "never"
                else:
                    pri_age_str = (
                        f"{(now - last_primary_s) * 1000:.0f}ms"
                    )
                if arb.last_override_s < 0:
                    ovr_age_str = "never"
                else:
                    ovr_age_str = (
                        f"{(now - arb.last_override_s) * 1000:.0f}ms"
                    )
                if teleop_mode_enabled:
                    if arb.last_teleop_mode_s < 0:
                        mode_age_str = "never"
                    else:
                        mode_age_str = (
                            f"{(now - arb.last_teleop_mode_s) * 1000:.0f}ms"
                        )
                    if not arb.teleop_mode_fresh(now):
                        stale_tag = " STALE"
                    elif arb.current_teleop_mode == "OFF":
                        stale_tag = " OFF"
                    else:
                        stale_tag = ""
                    gate_str = (
                        f" gate(mode={arb.current_teleop_mode} "
                        f"age={mode_age_str} "
                        f"msgs={arb.teleop_mode_msgs} "
                        f"fail={arb.teleop_mode_decode_failures}"
                        f"{stale_tag})"
                    )
                else:
                    if cfg.frozen_ticks_threshold > 0:
                        frz_str = (
                            f" frozen(det={arb.override_frozen_detected} "
                            f"streak={arb.override_frozen_count}/"
                            f"{cfg.frozen_ticks_threshold} "
                            f"rel={arb.override_frozen_release_events})"
                        )
                    else:
                        frz_str = " frozen(disabled)"
                    if cfg.engage_motion_threshold > 0:
                        mot_str = (
                            f" moving(streak={arb.override_motion_count}/"
                            f"{cfg.engage_motion_threshold})"
                        )
                    else:
                        mot_str = " moving(legacy:immediate)"
                    gate_str = f"{frz_str}{mot_str}"
                print(
                    f"[pose_mux] tick={tick} "
                    f"primary(age={pri_age_str} fwd={primary_fwd}) "
                    f"override(active={arb.override_active} "
                    f"age={ovr_age_str} fwd={arb.override_frames_forwarded} "
                    f"eng={arb.override_engage_events} "
                    f"rel={arb.override_release_events})"
                    f"{gate_str}",
                    flush=True,
                )
                last_status_s = now

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We're behind schedule; reset baseline rather than spinning.
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("[pose_mux] SIGINT received; tearing down", flush=True)
    finally:
        primary_sub.close(linger=0)
        override_sub.close(linger=0)
        out_pub.close(linger=0)
        if teleop_mode_sub is not None:
            teleop_mode_sub.close(linger=0)
        if vla_control_pub is not None:
            vla_control_pub.close(linger=0)
        ctx.term()

    print(
        f"[pose_mux] done. total_ticks={tick} "
        f"primary_fwd={primary_fwd} "
        f"override_fwd={arb.override_frames_forwarded} "
        f"engaged={arb.override_engage_events} "
        f"released={arb.override_release_events}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
