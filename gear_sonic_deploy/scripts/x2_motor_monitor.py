#!/usr/bin/env python3
"""Continuous motor-state monitoring daemon for the X2 split-topology deploy.

Runs as the third tmux session on PC2 (alongside the C++ deploy and the hand
bridge); subscribes to MC's joint state + command topics, computes per-tick
signals (tracking error, position-vs-soft-limit proximity, velocity / torque
saturation), polls the MC action mode service at 1 Hz, and:

  * writes one JSONL line per signal sample to ``--jsonl`` (default
    ``/var/log/x2/motor_monitor.jsonl`` with daily rotation);
  * detects edge events (MC mode change, drive-fault flips, tracking-error
    spikes crossing the configured threshold) and writes those as
    ``"kind": "event"`` JSONL records;
  * publishes a compact summary on ZMQ PUB ``tcp://*:5567`` topic
    ``motor_monitor`` for the operator-side sidecar (the manager on the
    laptop SUBs to this and appends each frame to manager_sidecar.jsonl
    so a single grep across the laptop logs surfaces every motor-side
    event the deploy run touched).

**Log-only policy.** This daemon does NOT publish any joint commands, does
NOT issue SetMcAction calls, and does NOT modify the deploy's behaviour in
any way. It only observes. The operator can correlate alerts post-hoc via
``x2_freeze_postmortem.py``, but real-time intervention is intentionally
out of scope -- the deploy's pose-ref starvation watchdog + SAFE_IDLE
state owns the real-time safety path.

Topic layout
------------

The daemon subscribes to::

    /aima/hal/joint/{leg,waist,arm,head}/{state,command}

and matches each joint by name against the canonical 31-DOF list (same
source-of-truth as ``x2_scan_mc_motors.py``). Joints not in the list (e.g.
hand DOFs) are silently ignored.

MC mode is polled via the AimRT service::

    /aimdk_5Fmsgs/srv/GetMcAction        (response: McActionInfo)

at 1 Hz. Mode transitions are surfaced both as JSONL events and on the
next ZMQ summary so the operator can see "MC just switched JOINT_DEFAULT
-> SOFT_EMERGENCY_STOP" within ~1 s of it happening.

Output schema
-------------

Two record kinds land in the JSONL:

1. ``"kind": "sample"`` -- one per second, aggregated per-group stats.
   Includes max abs tracking error per group, joints currently in the
   top-5 tracking-error list (with absolute rad values + nearest limit
   proximity), MC mode + status, command/state staleness counters.

2. ``"kind": "event"`` -- emitted on rising-edge conditions:
   * MC mode transition (``mc_mode_change``)
   * Tracking error crossing ``--tracking-error-warn-rad`` on any joint
     (``tracking_error_spike``)
   * Joint exceeding ``--limit-margin-rad`` of its soft limit
     (``limit_proximity``)
   * Persistent command staleness (``command_staleness``)

The ZMQ summary payload (JSON-encoded bytes) is a compact representation
of the latest sample + any events fired in the last cycle.

CLI
---

::

    x2_motor_monitor.py [--jsonl PATH] [--zmq-port PORT] [--zmq-topic NAME]
        [--mc-mode-poll-s SEC] [--summary-rate-hz HZ]
        [--tracking-error-warn-rad RAD] [--limit-margin-rad RAD]
        [--stale-state-s SEC] [--no-rotate]

All thresholds default to safe values; see ``--help``. Designed to be
restarted at will -- it has no in-memory state worth preserving, and the
JSONL is append-only with timestamps that let a postmortem stitch
multiple runs together.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import os
import pathlib
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional

# rclpy / aimdk_msgs / cppzmq imports
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
except Exception as _e:  # noqa: BLE001
    print(
        f"ERROR: rclpy not available ({_e}). Run inside the agibot docker on "
        "PC2 (or any environment with ROS 2 Humble + sourced overlay).",
        file=sys.stderr,
    )
    raise SystemExit(1)

try:
    from aimdk_msgs.msg import (
        CommonRequest,
        JointCommandArray,
        JointStateArray,
    )
    from aimdk_msgs.srv import GetMcAction
except ImportError as _e:
    print(
        f"ERROR: aimdk_msgs not on the Python path ({_e}). Source the colcon "
        "overlay that built aimdk_msgs (typically /ros2_ws/install/setup.bash).",
        file=sys.stderr,
    )
    raise SystemExit(1)

try:
    import zmq
except ImportError as _e:
    print(
        f"ERROR: pyzmq not installed ({_e}). pip install pyzmq.",
        file=sys.stderr,
    )
    raise SystemExit(1)


log = logging.getLogger("x2_motor_monitor")


# Canonical 31-DOF MJ joint order. Matches x2_scan_mc_motors.py /
# policy_parameters.hpp; kept inline so this daemon has zero deploy-
# package dependencies (it can run on PC2 without the gear_sonic_deploy
# python tree mounted).
MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
)
NAME_SET = set(MUJOCO_JOINT_NAMES)
NUM_DOFS = len(MUJOCO_JOINT_NAMES)

JOINT_GROUPS = ("leg", "waist", "arm", "head")
STATE_TOPIC = "/aima/hal/joint/{group}/state"
COMMAND_TOPIC = "/aima/hal/joint/{group}/command"
MC_GET_ACTION_SVC = "/aimdk_5Fmsgs/srv/GetMcAction"

# Best-effort QoS profile -- matches MC's HAL publishers (verified via
# x2_scan_mc_motors.py probe_publishers).
_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


# ---------------------------------------------------------------------------
# Per-joint rolling buffers
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class JointSample:
    t: float
    pos: float
    vel: float
    eff: float


@dataclasses.dataclass
class JointCommand:
    t: float
    target: float
    kp: float
    kd: float
    vel: float = 0.0
    eff: float = 0.0


# Rolling buffer size: 60 s @ 50 Hz = 3000 samples. Enough to detect
# oscillation cycles up to ~5 Hz with FFT (we don't do FFT here, but the
# buffers are sized to allow it via the postmortem tool).
ROLL_BUFFER_LEN = 3000


class JointBuffers:
    """Per-joint rolling state and command buffers (lock-protected)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, deque[JointSample]] = defaultdict(
            lambda: deque(maxlen=ROLL_BUFFER_LEN)
        )
        self._commands: dict[str, deque[JointCommand]] = defaultdict(
            lambda: deque(maxlen=ROLL_BUFFER_LEN)
        )
        # Most recent sample timestamps for staleness checks.
        self._last_state_t: dict[str, float] = {}
        self._last_command_t: dict[str, float] = {}

    def push_state(self, name: str, sample: JointSample) -> None:
        with self._lock:
            self._states[name].append(sample)
            self._last_state_t[name] = sample.t

    def push_command(self, name: str, cmd: JointCommand) -> None:
        with self._lock:
            self._commands[name].append(cmd)
            self._last_command_t[name] = cmd.t

    def snapshot(self) -> tuple[dict, dict, dict, dict]:
        """Return shallow copies of the latest-per-joint state + command."""
        with self._lock:
            latest_state = {n: self._states[n][-1] for n in self._states if self._states[n]}
            latest_cmd = {n: self._commands[n][-1] for n in self._commands if self._commands[n]}
            last_state_t = dict(self._last_state_t)
            last_cmd_t = dict(self._last_command_t)
        return latest_state, latest_cmd, last_state_t, last_cmd_t


# ---------------------------------------------------------------------------
# Monitor ROS node
# ---------------------------------------------------------------------------

class MonitorNode(Node):
    """ROS node owning the topic subscriptions + MC mode service client."""

    def __init__(self, buffers: JointBuffers, mc_poll_s: float) -> None:
        super().__init__("x2_motor_monitor")
        self._buffers = buffers
        self._t0 = time.monotonic()
        self._mc_action_mode = -1
        self._mc_action_desc = ""
        self._mc_action_status = -1
        self._mc_request_in_flight = False
        self._mc_lock = threading.Lock()

        for group in JOINT_GROUPS:
            self.create_subscription(
                JointStateArray,
                STATE_TOPIC.format(group=group),
                lambda msg, g=group: self._on_state(g, msg),
                _QOS,
            )
            self.create_subscription(
                JointCommandArray,
                COMMAND_TOPIC.format(group=group),
                lambda msg, g=group: self._on_command(g, msg),
                _QOS,
            )

        self._mc_client = self.create_client(GetMcAction, MC_GET_ACTION_SVC)
        if mc_poll_s > 0.0:
            period = max(mc_poll_s, 0.1)
            self.create_timer(period, self._poll_mc_mode)

    @property
    def t0(self) -> float:
        return self._t0

    def latest_mc_mode(self) -> tuple[int, str, int]:
        with self._mc_lock:
            return self._mc_action_mode, self._mc_action_desc, self._mc_action_status

    def _on_state(self, group: str, msg: JointStateArray) -> None:
        ts = time.monotonic() - self._t0
        for js in msg.joints:
            name = str(js.name)
            if name not in NAME_SET:
                continue
            self._buffers.push_state(name, JointSample(
                t=ts,
                pos=float(js.position),
                vel=float(js.velocity),
                eff=float(js.effort),
            ))

    def _on_command(self, group: str, msg: JointCommandArray) -> None:
        ts = time.monotonic() - self._t0
        for jc in msg.joints:
            name = str(jc.name)
            if name not in NAME_SET:
                continue
            self._buffers.push_command(name, JointCommand(
                t=ts,
                target=float(jc.position),
                kp=float(jc.stiffness),
                kd=float(jc.damping),
                vel=float(jc.velocity),
                eff=float(jc.effort),
            ))

    def _poll_mc_mode(self) -> None:
        """1 Hz best-effort poll of GetMcAction."""
        if not self._mc_client.service_is_ready():
            return
        if self._mc_request_in_flight:
            return
        req = GetMcAction.Request()
        req.request = CommonRequest()
        req.request.header.stamp = self.get_clock().now().to_msg()
        future = self._mc_client.call_async(req)
        self._mc_request_in_flight = True

        def _done(fut):
            try:
                resp = fut.result()
                if resp is not None:
                    with self._mc_lock:
                        self._mc_action_mode = int(resp.info.current_action.value)
                        self._mc_action_desc = str(resp.info.action_desc)
                        self._mc_action_status = int(resp.info.status.value)
            except Exception as exc:  # noqa: BLE001
                log.debug("MC mode poll future threw: %s", exc)
            finally:
                self._mc_request_in_flight = False
        future.add_done_callback(_done)


# ---------------------------------------------------------------------------
# Summary computer + event detector
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MonitorThresholds:
    """Operator-tunable thresholds for edge-event detection."""

    tracking_error_warn_rad: float = 0.30
    """Trip the ``tracking_error_spike`` event when |target - position| on any
    joint exceeds this. Default 0.30 rad (~17 deg) -- generous enough to
    avoid spurious alerts on normal teleop transients, tight enough to
    catch a stuck-against-limit or fighting-MC condition."""

    limit_margin_rad: float = 0.05
    """Trip ``limit_proximity`` when a joint's commanded target is within this
    distance of either soft limit. The default 0.05 rad (~3 deg) flags
    approaches before they become hard stops."""

    stale_state_s: float = 0.5
    """Trip ``state_staleness`` when no state sample for a joint has been
    received in this many seconds. 0.5 s = ~25 missed frames at the
    nominal 50 Hz publish rate."""

    stale_command_s: float = 1.0
    """Trip ``command_staleness`` when MC's command stream has been silent
    on a joint for this long. MC publishes its commands continuously
    while in CONTROL / STAND_DEFAULT modes."""


# Default soft-limit estimates (radians) per joint. These are conservative
# guesses pulled from the X2 URDF + sim configs; they are used ONLY for
# the limit_proximity alert, which is intentionally generous. Override at
# runtime via --soft-limits-json. Missing entries are treated as
# "limit unknown" -- the limit_proximity check is skipped on that joint.
DEFAULT_SOFT_LIMITS: dict[str, tuple[float, float]] = {
    # Hips: -1.0 .. 1.0 rad covers most teleop poses.
    "left_hip_pitch_joint":  (-2.4, 2.4),
    "right_hip_pitch_joint": (-2.4, 2.4),
    "left_hip_roll_joint":   (-0.8, 0.8),
    "right_hip_roll_joint":  (-0.8, 0.8),
    "left_hip_yaw_joint":    (-1.2, 1.2),
    "right_hip_yaw_joint":   (-1.2, 1.2),
    # Knees: 0 .. ~2.5 rad (flexion only).
    "left_knee_joint":  (-0.1, 2.6),
    "right_knee_joint": (-0.1, 2.6),
    # Ankles: typical sagittal +/- 0.8 rad; roll +/- 0.5 rad.
    "left_ankle_pitch_joint":  (-1.0, 1.0),
    "right_ankle_pitch_joint": (-1.0, 1.0),
    "left_ankle_roll_joint":   (-0.5, 0.5),
    "right_ankle_roll_joint":  (-0.5, 0.5),
    # Waist.
    "waist_yaw_joint":   (-1.5, 1.5),
    "waist_pitch_joint": (-0.9, 0.9),
    "waist_roll_joint":  (-0.6, 0.6),
    # Arms (per-joint; wrist limits are intentionally generous since the
    # smallmotor wrists have a wider mechanical range than the IK seeds).
    "left_shoulder_pitch_joint":  (-3.14, 3.14),
    "right_shoulder_pitch_joint": (-3.14, 3.14),
    "left_shoulder_roll_joint":   (-1.5, 1.5),
    "right_shoulder_roll_joint":  (-1.5, 1.5),
    "left_shoulder_yaw_joint":    (-3.14, 3.14),
    "right_shoulder_yaw_joint":   (-3.14, 3.14),
    "left_elbow_joint":  (-2.6, 0.1),
    "right_elbow_joint": (-2.6, 0.1),
    "left_wrist_yaw_joint":    (-2.0, 2.0),
    "right_wrist_yaw_joint":   (-2.0, 2.0),
    "left_wrist_pitch_joint":  (-1.0, 1.0),
    "right_wrist_pitch_joint": (-1.0, 1.0),
    "left_wrist_roll_joint":   (-1.0, 1.0),
    "right_wrist_roll_joint":  (-1.0, 1.0),
    # Head.
    "head_yaw_joint":   (-1.5, 1.5),
    "head_pitch_joint": (-0.6, 0.6),
}


def _joint_group(name: str) -> str:
    """Map a joint name to its publishing group (matches MC topology)."""
    if "hip" in name or "knee" in name or "ankle" in name:
        return "leg"
    if "waist" in name:
        return "waist"
    if "head" in name:
        return "head"
    return "arm"  # shoulders / elbows / wrists


class SummaryComputer:
    """Aggregates per-joint snapshots into a per-cycle summary + events."""

    def __init__(self,
                 thresholds: MonitorThresholds,
                 soft_limits: dict[str, tuple[float, float]]) -> None:
        self.thresholds = thresholds
        self.soft_limits = soft_limits
        # Last-known per-joint flags so we only emit RISING-edge events
        # (e.g. tracking_error_spike fires once when the joint crosses the
        # threshold, and again only after it dropped below and re-crossed).
        self._track_err_armed: dict[str, bool] = {}
        self._limit_proximity_armed: dict[str, bool] = {}
        self._state_stale_armed: dict[str, bool] = {}
        self._command_stale_armed: dict[str, bool] = {}
        self._last_mc_mode: int = -2  # sentinel that differs from -1 "unknown"
        self._last_mc_desc: str = ""

    def compute(self,
                buffers: JointBuffers,
                mc_mode: int,
                mc_desc: str,
                mc_status: int,
                now_rel_s: float) -> tuple[dict, list[dict]]:
        """Compute one summary + any rising-edge events."""
        latest_state, latest_cmd, last_state_t, last_cmd_t = buffers.snapshot()
        events: list[dict] = []

        # ---- MC mode transition (rising-edge across runs) -----------------
        if (self._last_mc_mode != -2
                and (mc_mode != self._last_mc_mode or mc_desc != self._last_mc_desc)):
            events.append({
                "kind": "event",
                "ts": time.time(),
                "type": "mc_mode_change",
                "previous_mode": self._last_mc_mode,
                "previous_desc": self._last_mc_desc,
                "current_mode": mc_mode,
                "current_desc": mc_desc,
                "current_status": mc_status,
            })
        self._last_mc_mode = mc_mode
        self._last_mc_desc = mc_desc

        # ---- Per-joint scans (tracking error, limit proximity, staleness) -
        joint_records: list[dict] = []
        top_err: list[tuple[float, str, dict]] = []
        for name in MUJOCO_JOINT_NAMES:
            s = latest_state.get(name)
            c = latest_cmd.get(name)
            rec: dict[str, Any] = {"name": name, "group": _joint_group(name)}
            if s is not None:
                rec.update(pos=s.pos, vel=s.vel, eff=s.eff)
            if c is not None:
                rec.update(target=c.target, kp=c.kp, kd=c.kd)
            if s is not None and c is not None:
                err = c.target - s.pos
                rec["tracking_err"] = err
                rec["tracking_err_abs"] = abs(err)
                # Rising-edge tracking error spike.
                armed = self._track_err_armed.get(name, True)
                if abs(err) >= self.thresholds.tracking_error_warn_rad:
                    if armed:
                        events.append({
                            "kind": "event",
                            "ts": time.time(),
                            "type": "tracking_error_spike",
                            "joint": name,
                            "tracking_err": err,
                            "threshold": self.thresholds.tracking_error_warn_rad,
                            "pos": s.pos,
                            "target": c.target,
                            "vel": s.vel,
                            "eff": s.eff,
                            "kp": c.kp,
                            "kd": c.kd,
                        })
                        self._track_err_armed[name] = False
                elif abs(err) <= 0.5 * self.thresholds.tracking_error_warn_rad:
                    # Hysteresis: re-arm once the error settles to <= half
                    # the trip threshold. Prevents flapping when the joint
                    # bounces around at the threshold.
                    self._track_err_armed[name] = True
                top_err.append((abs(err), name, rec))
            # Limit proximity (on TARGET, since pos can lag a divergent
            # target by enough to mask the trip).
            if c is not None and name in self.soft_limits:
                lo, hi = self.soft_limits[name]
                margin = self.thresholds.limit_margin_rad
                near_low = (c.target - lo) <= margin
                near_high = (hi - c.target) <= margin
                rec["limit_lo"] = lo
                rec["limit_hi"] = hi
                rec["limit_margin_lo"] = c.target - lo
                rec["limit_margin_hi"] = hi - c.target
                armed = self._limit_proximity_armed.get(name, True)
                if near_low or near_high:
                    if armed:
                        events.append({
                            "kind": "event",
                            "ts": time.time(),
                            "type": "limit_proximity",
                            "joint": name,
                            "side": "low" if near_low else "high",
                            "target": c.target,
                            "limit_lo": lo,
                            "limit_hi": hi,
                            "margin_rad": min(
                                c.target - lo if near_low else float("inf"),
                                hi - c.target if near_high else float("inf"),
                            ),
                        })
                        self._limit_proximity_armed[name] = False
                else:
                    self._limit_proximity_armed[name] = True
            # State staleness.
            last_t = last_state_t.get(name, -1.0)
            state_stale = (last_t > 0.0 and (now_rel_s - last_t) >= self.thresholds.stale_state_s)
            rec["state_age_s"] = (now_rel_s - last_t) if last_t > 0.0 else -1.0
            armed = self._state_stale_armed.get(name, True)
            if state_stale and armed:
                events.append({
                    "kind": "event",
                    "ts": time.time(),
                    "type": "state_staleness",
                    "joint": name,
                    "age_s": now_rel_s - last_t,
                    "threshold_s": self.thresholds.stale_state_s,
                })
                self._state_stale_armed[name] = False
            elif not state_stale:
                self._state_stale_armed[name] = True
            # Command staleness.
            last_t = last_cmd_t.get(name, -1.0)
            cmd_stale = (last_t > 0.0 and (now_rel_s - last_t) >= self.thresholds.stale_command_s)
            rec["command_age_s"] = (now_rel_s - last_t) if last_t > 0.0 else -1.0
            armed = self._command_stale_armed.get(name, True)
            if cmd_stale and armed:
                events.append({
                    "kind": "event",
                    "ts": time.time(),
                    "type": "command_staleness",
                    "joint": name,
                    "age_s": now_rel_s - last_t,
                    "threshold_s": self.thresholds.stale_command_s,
                })
                self._command_stale_armed[name] = False
            elif not cmd_stale:
                self._command_stale_armed[name] = True
            joint_records.append(rec)

        # ---- Per-group aggregates ----------------------------------------
        group_stats: dict[str, dict[str, Any]] = {}
        for group in JOINT_GROUPS:
            joints = [r for r in joint_records if r.get("group") == group]
            if not joints:
                group_stats[group] = {"count": 0}
                continue
            errs = [abs(r["tracking_err"]) for r in joints if "tracking_err" in r]
            vels = [abs(r["vel"]) for r in joints if "vel" in r]
            effs = [abs(r["eff"]) for r in joints if "eff" in r]
            kps = [r["kp"] for r in joints if "kp" in r]
            stats: dict[str, Any] = {"count": len(joints)}
            if errs:
                stats["max_tracking_err"] = max(errs)
                stats["mean_tracking_err"] = sum(errs) / len(errs)
            if vels:
                stats["max_abs_vel"] = max(vels)
            if effs:
                stats["max_abs_eff"] = max(effs)
            if kps:
                stats["max_kp"] = max(kps)
            group_stats[group] = stats

        top_err.sort(key=lambda x: x[0], reverse=True)
        top5 = [
            {"joint": name, "abs_err": err,
             "target": rec.get("target"), "pos": rec.get("pos")}
            for err, name, rec in top_err[:5]
        ]

        sample = {
            "kind": "sample",
            "ts": time.time(),
            "rel_t": now_rel_s,
            "mc_action_mode": mc_mode,
            "mc_action_desc": mc_desc,
            "mc_action_status": mc_status,
            "groups": group_stats,
            "top_tracking_err": top5,
        }
        return sample, events


# ---------------------------------------------------------------------------
# JSONL writer with daily rotation
# ---------------------------------------------------------------------------

class JsonlWriter:
    """Append-only JSONL writer with optional daily rotation."""

    def __init__(self, path: pathlib.Path, rotate_daily: bool) -> None:
        self.base_path = path
        self.rotate_daily = rotate_daily
        self._current_day: Optional[str] = None
        self._fh = None
        self._lock = threading.Lock()
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self) -> None:
        day = _dt.date.today().isoformat()
        if self.rotate_daily:
            path = self.base_path.with_suffix(f".{day}.jsonl")
        else:
            path = self.base_path
        self._fh = path.open("a", buffering=1)
        self._current_day = day
        log.info("JSONL writer opened: %s", path)

    def write(self, record: dict) -> None:
        with self._lock:
            if self.rotate_daily:
                today = _dt.date.today().isoformat()
                if today != self._current_day:
                    if self._fh is not None:
                        try:
                            self._fh.close()
                        except Exception:
                            pass
                    self._open()
            if self._fh is None:
                return
            self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None


# ---------------------------------------------------------------------------
# ZMQ publisher (motor_monitor summary)
# ---------------------------------------------------------------------------

class SummaryPublisher:
    """PUB socket emitting compact JSON summaries on the operator wire."""

    def __init__(self, port: int, topic: str) -> None:
        self._topic = topic
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SNDHWM, 10)
        self._sock.bind(f"tcp://*:{port}")
        log.info("ZMQ PUB bound on tcp://*:%d (topic=%s).", port, topic)

    def publish(self, sample: dict, events: list[dict]) -> None:
        try:
            payload = {
                "sample": sample,
                "events": events,
            }
            self._sock.send_multipart(
                [self._topic.encode("utf-8"),
                 json.dumps(payload).encode("utf-8")],
                flags=zmq.NOBLOCK,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("ZMQ publish failed: %s", exc)

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--jsonl",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("PC2_PREFIX", "/home/run/gear-sonic")) / "log" / "motor_monitor.jsonl",
        help="Path to the JSONL output. Parent dir is created if missing.",
    )
    p.add_argument(
        "--no-rotate", action="store_true",
        help="Disable daily rotation; always write to --jsonl as-is.",
    )
    p.add_argument(
        "--zmq-port", type=int, default=5567,
        help="ZMQ PUB port for the compact summary (default 5567).",
    )
    p.add_argument(
        "--zmq-topic", default="motor_monitor",
        help="ZMQ topic name (default 'motor_monitor').",
    )
    p.add_argument(
        "--summary-rate-hz", type=float, default=1.0,
        help="Sample + publish cadence in Hz (default 1.0).",
    )
    p.add_argument(
        "--mc-mode-poll-s", type=float, default=1.0,
        help="MC GetMcAction poll interval (s). Set <=0 to disable.",
    )
    p.add_argument(
        "--tracking-error-warn-rad", type=float, default=0.30,
        help="Tracking-error spike threshold (rad). Default 0.30.",
    )
    p.add_argument(
        "--limit-margin-rad", type=float, default=0.05,
        help="Soft-limit proximity margin (rad). Default 0.05.",
    )
    p.add_argument(
        "--stale-state-s", type=float, default=0.5,
        help="State staleness threshold (s). Default 0.5.",
    )
    p.add_argument(
        "--stale-command-s", type=float, default=1.0,
        help="Command staleness threshold (s). Default 1.0.",
    )
    p.add_argument(
        "--soft-limits-json", type=pathlib.Path, default=None,
        help=(
            "Optional JSON file overriding the built-in DEFAULT_SOFT_LIMITS "
            "table. Must map joint name -> [low, high] in radians."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s %(levelname)s x2_motor_monitor] %(message)s",
    )

    # Load soft-limit overrides.
    soft_limits = dict(DEFAULT_SOFT_LIMITS)
    if args.soft_limits_json is not None:
        try:
            data = json.loads(args.soft_limits_json.read_text())
            for k, v in data.items():
                if isinstance(v, list) and len(v) == 2:
                    soft_limits[k] = (float(v[0]), float(v[1]))
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to load --soft-limits-json: %s", exc)

    thresholds = MonitorThresholds(
        tracking_error_warn_rad=args.tracking_error_warn_rad,
        limit_margin_rad=args.limit_margin_rad,
        stale_state_s=args.stale_state_s,
        stale_command_s=args.stale_command_s,
    )

    rclpy.init(args=[])
    buffers = JointBuffers()
    node = MonitorNode(buffers, args.mc_mode_poll_s)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin, name="rclpy_spin", daemon=True,
    )
    spin_thread.start()

    writer = JsonlWriter(args.jsonl, rotate_daily=not args.no_rotate)
    publisher = SummaryPublisher(args.zmq_port, args.zmq_topic)
    computer = SummaryComputer(thresholds, soft_limits)

    stop_requested = threading.Event()

    def _on_signal(signum, _frame):  # type: ignore[unused-argument]
        log.info("signal %d received, shutting down...", signum)
        stop_requested.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    period = 1.0 / max(args.summary_rate_hz, 0.1)
    next_tick = time.monotonic()
    cycle = 0
    try:
        # Boot record so postmortem can see when this daemon started even if
        # the very first cycle is empty.
        writer.write({
            "kind": "boot",
            "ts": time.time(),
            "summary_rate_hz": args.summary_rate_hz,
            "tracking_error_warn_rad": thresholds.tracking_error_warn_rad,
            "limit_margin_rad": thresholds.limit_margin_rad,
            "stale_state_s": thresholds.stale_state_s,
            "stale_command_s": thresholds.stale_command_s,
            "zmq_port": args.zmq_port,
            "zmq_topic": args.zmq_topic,
            "jsonl_path": str(args.jsonl),
            "rotate_daily": not args.no_rotate,
            "pid": os.getpid(),
        })

        while not stop_requested.is_set():
            mc_mode, mc_desc, mc_status = node.latest_mc_mode()
            now_rel = time.monotonic() - node.t0
            sample, events = computer.compute(
                buffers, mc_mode, mc_desc, mc_status, now_rel,
            )
            writer.write(sample)
            for ev in events:
                writer.write(ev)
            publisher.publish(sample, events)

            cycle += 1
            if cycle == 1 or cycle % 30 == 0:
                log.info(
                    "cycle=%d mc_mode=%d desc=%r status=%d events_this_cycle=%d",
                    cycle, mc_mode, mc_desc, mc_status, len(events),
                )

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Overrun: re-anchor so we don't burn CPU catching up.
                next_tick = time.monotonic()
    finally:
        log.info("shutting down: closing JSONL + ZMQ + ROS node...")
        writer.write({
            "kind": "shutdown",
            "ts": time.time(),
            "cycles_completed": cycle,
        })
        writer.close()
        publisher.close()
        try:
            executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
