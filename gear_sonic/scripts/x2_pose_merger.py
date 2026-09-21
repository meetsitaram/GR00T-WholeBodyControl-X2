#!/usr/bin/env python3
"""x2_pose_merger.py -- laptop-side pose merger for the X2 planner stacks.

Minimal stand-in for the dataset recorder's *subscribe-mode, teleop-only*
role (``record_x2_dataset --body-pose-source zmq --arm-targets-source zmq
--teleop-only``), which is what ``run_x2_quest3_planner_stack.sh`` (step
4/4) and ``run_x2_pkl_direct_stack.sh`` spawn as the last process of the
stack. The full recorder (LeRobot writer, MuJoCo renderer, SONIC
tokenizer, RoboCasa mirror, VLA plumbing) is not part of this port; this
script reproduces only the wire-level behaviour the deploy depends on:

    planner  body_pose  @ :5565 ─┐
    manager  arm_targets /       ├─► merge ─► ``pose`` PUB @ :5556 ─► deploy
             hand_finger_cmd     │              (pack_pose_message v4, 50 Hz)
             @ :5564 ────────────┘
    play_gesture / play_locomotion / pad bridge  motion_clip_cmd @ :5568
             (SUB *bind*; PKL takeover of the wire)
    deploy   x2_debug   @ :5557 (base_quat -> yaw-rebased idle + clip seed)

Merge semantics (identical to the recorder's ``_run_subscribe_mode``):

* Body (legs / waist / head) = planner ``joint_pos_mj``. Arm slices
  ``[15:22]`` / ``[22:29]`` are overwritten by the manager's
  ``arm_targets`` when a valid ``(7,)`` command is cached; a
  ``passthrough_arm_targets`` message clears the cache so the planner's
  arms flow through. The planner's v5 future window is forwarded with
  the same arm command pinned across every future slot and
  ``joint_vel_mj_future`` recomputed by finite difference; the wire
  ``motion_token`` is forwarded when present, else zeros.
* Hands = manager ``hand_finger_cmd`` (zeros until the first message).
* ``root_quat_xyzw`` / ``root_xy_world`` / ``root_z_world`` pass through
  from the planner; an ``estop`` flag on ``body_pose`` is latched and
  forwarded on every merged frame (terminal).
* Before the first ``body_pose``: publish the trained stand pose (yaw
  rebased to the live / last-seen ``x2_debug`` ``base_quat``) with a
  stationary future window, unless ``--no-idle-publish``.
* ``motion_clip_cmd`` play / stop: a PKL clip takes over the wire
  (full 31-DOF body from the clip, zero hands / token, yaw-rebased to
  the robot's current heading); ``hold_after`` parks the last frame;
  stop / completion without an upstream ``body_pose`` ramps back to the
  stand pose over 1 s (smootherstep) instead of snapping.
* Shutdown (SIGINT / SIGTERM): 0.8 s ramp from the last published pose
  to stand + 0.4 s hold before the wire goes silent.

Deliberately dropped from the recorder: dataset episodes
(``recorder_cmd`` start/save/discard are logged and ignored), the
SONIC tokenizer (``action.motion_token``), rendering, head cameras,
RoboCasa scene mirroring and the obs-parity dump. Recording-only CLI
flags are accepted and ignored with a warning so both launchers can
pass their existing argument lists unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    pack_pose_message,
)

# ── Constants (mirror the recorder / deploy wire contract) ────────────────

NUM_BODY_DOFS = 31
NUM_HAND_DOF_PER_SIDE = 10
SONIC_MOTION_TOKEN_DIM = 64
_LEFT_ARM_MJ_SLICE = slice(15, 22)   # left shoulder/elbow/wrist (7 joints)
_RIGHT_ARM_MJ_SLICE = slice(22, 29)  # right arm (7 joints)

# Trained stand pose (radians, MuJoCo body order). Mirrors
# ``policy_parameters.hpp`` / ``gear_sonic.utils.planner.constants``.
DEFAULT_STAND_POSE_MUJOCO_RAD: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # left leg (6)
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # right leg (6)
    0.0, 0.0, 0.0,                                # waist (3)
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,           # left arm (7)
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,          # right arm (7)
    0.0, 0.0,                                     # head (2)
)
assert len(DEFAULT_STAND_POSE_MUJOCO_RAD) == NUM_BODY_DOFS

DEPLOY_ALIVE_STALE_THRESHOLD_S = 1.0   # x2_debug liveness window
_DEPLOY_SILENT_REWARN_S = 30.0
_SHUTDOWN_SETTLE_RAMP_S = 0.8
_SHUTDOWN_SETTLE_HOLD_S = 0.4
_RETURN_TO_STAND_RAMP_S = 1.0
_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

LOG = "[merger]"


def _log(msg: str) -> None:
    print(f"{LOG} {msg}", flush=True)


def yaw_of_quat_xyzw(q: np.ndarray) -> float:
    """Yaw (rad) about world z of an ``xyzw`` quaternion (zyx Euler)."""
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quat_xyzw(yaw: float) -> np.ndarray:
    """Pure ``R_z(yaw)`` as an ``xyzw`` quaternion."""
    return np.array(
        [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)], dtype=np.float32
    )


def _smootherstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ── Upstream state caches ─────────────────────────────────────────────────


class _DeployState:
    """Latest ``x2_debug`` frame (base_quat + liveness), lock-protected."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.received_any = False
        self.last_update_monotonic = 0.0

    def update(self, base_quat_wxyz: np.ndarray) -> None:
        with self.lock:
            self.base_quat_wxyz = base_quat_wxyz.astype(np.float64, copy=True)
            self.received_any = True
            self.last_update_monotonic = time.monotonic()

    def snapshot(self) -> tuple[np.ndarray, bool, bool, float]:
        with self.lock:
            alive = (
                self.received_any
                and (time.monotonic() - self.last_update_monotonic)
                <= DEPLOY_ALIVE_STALE_THRESHOLD_S
            )
            return (
                self.base_quat_wxyz.copy(),
                self.received_any,
                alive,
                self.last_update_monotonic,
            )

    def yaw(self) -> Optional[float]:
        """Yaw of the cached base_quat, or None if no packet ever arrived."""
        quat, received_any, _alive, _t = self.snapshot()
        if not received_any:
            return None
        return yaw_of_quat_xyzw(np.array([quat[1], quat[2], quat[3], quat[0]]))


class _UpstreamState:
    """Latest planner ``body_pose`` + manager arm / hand snapshot."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.body_pose_q_mj: Optional[np.ndarray] = None
        self.root_quat_xyzw: Optional[np.ndarray] = None
        self.root_xy_world: Optional[np.ndarray] = None
        self.root_z_world: Optional[float] = None
        self.joint_pos_mj_future: Optional[np.ndarray] = None
        self.root_quat_xyzw_future: Optional[np.ndarray] = None
        self.frame_index_future: Optional[np.ndarray] = None
        self.future_dt_s: Optional[float] = None
        self.wire_motion_token: Optional[np.ndarray] = None
        self.arm_left_q: Optional[np.ndarray] = None
        self.arm_right_q: Optional[np.ndarray] = None
        self.arm_engaged = False
        self.left_hand_q: Optional[np.ndarray] = None
        self.right_hand_q: Optional[np.ndarray] = None
        self.stream_mode = "OFF"
        self.pending_recorder_cmds: list[tuple[str, int]] = []
        self.estop_wire = False

    def update_body_pose(self, **kw: Any) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def update_arm_targets(
        self, left: np.ndarray, right: np.ndarray, engaged: bool, passthrough: bool
    ) -> None:
        with self.lock:
            if passthrough:
                self.arm_left_q = None
                self.arm_right_q = None
                self.arm_engaged = False
            else:
                self.arm_left_q = left.copy()
                self.arm_right_q = right.copy()
                self.arm_engaged = bool(engaged)

    def update_hands(self, left: np.ndarray, right: np.ndarray) -> None:
        with self.lock:
            self.left_hand_q = left.copy()
            self.right_hand_q = right.copy()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            cp = lambda a: None if a is None else np.array(a, copy=True)  # noqa: E731
            return {
                "body_pose_q_mj": cp(self.body_pose_q_mj),
                "root_quat_xyzw": cp(self.root_quat_xyzw),
                "root_xy_world": cp(self.root_xy_world),
                "root_z_world": self.root_z_world,
                "joint_pos_mj_future": cp(self.joint_pos_mj_future),
                "root_quat_xyzw_future": cp(self.root_quat_xyzw_future),
                "frame_index_future": cp(self.frame_index_future),
                "future_dt_s": self.future_dt_s,
                "wire_motion_token": cp(self.wire_motion_token),
                "arm_left_q": cp(self.arm_left_q),
                "arm_right_q": cp(self.arm_right_q),
                "left_hand_q": cp(self.left_hand_q),
                "right_hand_q": cp(self.right_hand_q),
                "stream_mode": self.stream_mode,
            }

    def drain_recorder_cmds(self) -> list[tuple[str, int]]:
        with self.lock:
            out, self.pending_recorder_cmds = self.pending_recorder_cmds, []
        return out


# ── Message decoders (same field gating as the recorder) ──────────────────


def handle_body_pose_msg(raw: bytes, state: _UpstreamState, *, topic: str) -> bool:
    """Decode a planner ``body_pose`` frame into ``state``. Returns True on accept."""
    try:
        decoded = unpack_message(raw, expected_topic=topic)
    except ValueError:
        return False
    fields = decoded.fields
    if "estop" in fields:
        try:
            if float(np.asarray(fields["estop"]).reshape(-1)[0]) >= 0.5:
                if not state.estop_wire:
                    _log("!!! E-STOP flag on body_pose wire -- forwarding to "
                         "deploy on every merged pose frame (terminal)")
                state.estop_wire = True
        except (IndexError, TypeError, ValueError):
            pass
    if "joint_pos_mj" not in fields:
        return False
    q = np.asarray(fields["joint_pos_mj"], dtype=np.float64).reshape(-1)
    if q.shape != (NUM_BODY_DOFS,):
        return False

    root_quat = None
    if "root_quat_xyzw" in fields:
        rq = np.asarray(fields["root_quat_xyzw"], dtype=np.float32).reshape(-1)
        if rq.shape == (4,):
            root_quat = rq
    root_xy = None
    if "root_xy_world" in fields:
        rxy = np.asarray(fields["root_xy_world"], dtype=np.float32).reshape(-1)
        if rxy.shape == (2,):
            root_xy = rxy
    root_z = None
    if "root_z_world" in fields:
        rz = np.asarray(fields["root_z_world"], dtype=np.float32).reshape(-1)
        if rz.shape == (1,):
            root_z = float(rz[0])
    wire_mt = None
    if "motion_token" in fields:
        mt = np.asarray(fields["motion_token"], dtype=np.float32).reshape(-1)
        if mt.shape == (SONIC_MOTION_TOKEN_DIM,):
            wire_mt = mt

    jpos_future = fields.get("joint_pos_mj_future")
    rot_future = fields.get("root_quat_xyzw_future")
    fidx_future = fields.get("frame_index_future")
    fdt_field = fields.get("future_dt_s")
    have_window = (
        jpos_future is not None and rot_future is not None
        and jpos_future.ndim == 2 and jpos_future.shape[1] == NUM_BODY_DOFS
        and rot_future.ndim == 2 and rot_future.shape == (jpos_future.shape[0], 4)
    )
    if have_window:
        jpos_arr = np.asarray(jpos_future, dtype=np.float32)
        fidx_arr = (
            None if fidx_future is None
            else np.asarray(fidx_future, dtype=np.int64).reshape(-1)
        )
        if fidx_arr is not None and fidx_arr.shape != (jpos_arr.shape[0],):
            fidx_arr = None
        fdt_val: Optional[float] = None
        if fdt_field is not None:
            try:
                fdt_val = float(np.asarray(fdt_field).reshape(-1)[0])
            except (IndexError, ValueError):
                fdt_val = None
        state.update_body_pose(
            body_pose_q_mj=q, root_quat_xyzw=root_quat, root_xy_world=root_xy,
            root_z_world=root_z, joint_pos_mj_future=jpos_arr,
            root_quat_xyzw_future=np.asarray(rot_future, dtype=np.float32),
            frame_index_future=fidx_arr, future_dt_s=fdt_val,
            wire_motion_token=wire_mt,
        )
    else:
        state.update_body_pose(
            body_pose_q_mj=q, root_quat_xyzw=root_quat, root_xy_world=root_xy,
            root_z_world=root_z, joint_pos_mj_future=None,
            root_quat_xyzw_future=None, frame_index_future=None,
            future_dt_s=None, wire_motion_token=wire_mt,
        )
    return True


def handle_manager_msg(parts: list[bytes], state: _UpstreamState) -> None:
    """Decode one manager multipart ``[topic, payload]`` message."""
    if len(parts) < 2:
        return
    topic = parts[0].decode("ascii", errors="replace")
    payload = parts[1]
    if topic == "recorder_cmd":
        try:
            d = json.loads(payload.decode("utf-8"))
            with state.lock:
                state.pending_recorder_cmds.append(
                    (str(d["action"]), int(d.get("tick", -1)))
                )
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError):
            pass
        return
    import msgpack  # arm_targets / hand_finger_cmd / stream_mode are msgpack

    try:
        msg = msgpack.unpackb(payload, raw=False)
    except Exception:  # noqa: BLE001 - malformed wire payload
        return
    if topic == "arm_targets":
        state.update_arm_targets(
            np.asarray(msg["left_q_rad"], dtype=np.float64),
            np.asarray(msg["right_q_rad"], dtype=np.float64),
            bool(msg.get("is_engaged", False)),
            bool(msg.get("passthrough_arm_targets", False)),
        )
    elif topic == "hand_finger_cmd":
        state.update_hands(
            np.asarray(msg["left_hand_q"], dtype=np.float64),
            np.asarray(msg["right_hand_q"], dtype=np.float64),
        )
    elif topic == "stream_mode":
        with state.lock:
            state.stream_mode = str(msg.get("mode", "OFF"))


# ── SUB threads ───────────────────────────────────────────────────────────


def _upstream_sub_thread(
    *, body_pose_url: str, body_pose_topic: str, manager_url: str,
    state: _UpstreamState, stop: threading.Event,
) -> None:
    ctx = zmq.Context.instance()
    sub_planner = ctx.socket(zmq.SUB)
    sub_planner.setsockopt(zmq.LINGER, 0)
    sub_planner.setsockopt_string(zmq.SUBSCRIBE, body_pose_topic)
    sub_planner.connect(body_pose_url)
    sub_mgr = ctx.socket(zmq.SUB)
    sub_mgr.setsockopt(zmq.LINGER, 0)
    for t in ("arm_targets", "hand_finger_cmd", "stream_mode", "recorder_cmd"):
        sub_mgr.setsockopt_string(zmq.SUBSCRIBE, t)
    sub_mgr.connect(manager_url)
    poller = zmq.Poller()
    poller.register(sub_planner, zmq.POLLIN)
    poller.register(sub_mgr, zmq.POLLIN)
    try:
        while not stop.is_set():
            events = dict(poller.poll(timeout=50))
            if sub_planner in events:
                try:
                    parts = sub_planner.recv_multipart(flags=zmq.NOBLOCK)
                    if parts:
                        handle_body_pose_msg(parts[0], state, topic=body_pose_topic)
                except zmq.error.Again:
                    pass
            if sub_mgr in events:
                try:
                    handle_manager_msg(sub_mgr.recv_multipart(flags=zmq.NOBLOCK), state)
                except zmq.error.Again:
                    pass
    finally:
        sub_planner.close(linger=0)
        sub_mgr.close(linger=0)


def _x2_debug_sub_thread(
    *, url: str, topic: str, state: _DeployState, stop: threading.Event,
) -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVHWM, 5)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(url)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        while not stop.is_set():
            if sock not in dict(poller.poll(200)):
                continue
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            try:
                msg = unpack_message(raw, expected_topic=topic)
            except ValueError:
                continue
            bq = np.asarray(
                msg.fields.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64
            ).reshape(-1)
            if bq.shape[0] != 4:
                bq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            state.update(bq)
    finally:
        sock.close(linger=0)


def _clip_cmd_sub_thread(
    *, url: str, topic: str, q: "queue.Queue[Any]", stop: threading.Event,
    parse_cmd: Any,
) -> None:
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVTIMEO, 100)
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    try:
        sub.bind(url)  # SUB binds: the trigger scripts are the transient side
    except zmq.error.ZMQError as exc:
        # Another process (typically the kplanner daemon) already owns this
        # port in the planner stacks; clip playback through the merger is then
        # simply unavailable -- the kplanner plays clips itself.
        print(f"[merger] WARNING: motion_clip_cmd SUB could not bind {url} ({exc}); "
              "clip playback through the merger DISABLED (kplanner owns the port)", flush=True)
        sub.close(linger=0)
        return
    try:
        while not stop.is_set():
            try:
                parts = sub.recv_multipart()
            except zmq.error.Again:
                continue
            if len(parts) < 2:
                continue
            try:
                q.put(parse_cmd(json.loads(parts[1].decode("utf-8"))))
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                _log(f"motion_clip_cmd: ignoring malformed payload: {exc}")
    finally:
        sub.close(linger=0)


# ── Merger ────────────────────────────────────────────────────────────────


class PoseMerger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.a = args
        self.rate = float(args.rate)
        self.period = 1.0 / max(self.rate, 1e-6)
        self.n_future = int(args.clip_future_window_frames)
        self.future_dt = float(args.clip_future_dt_s)
        self.step_ticks = max(1, int(round(self.future_dt * self.rate)))
        self.zero_token = np.zeros(SONIC_MOTION_TOKEN_DIM, dtype=np.float64)
        self.zero_hand = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
        self.stand = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
        self.stop = threading.Event()
        self.upstream = _UpstreamState()
        self.deploy = _DeployState()
        self.last_body: Optional[np.ndarray] = None
        self.last_root_quat: Optional[np.ndarray] = None
        self.in_shutdown_settle = False
        self.settle_ramp: Optional[dict[str, Any]] = None
        self.active_clip: Any = None
        self.active_clip_hold_after = False
        self.held_frame: Optional[dict[str, np.ndarray]] = None
        self.catalog: dict[str, Any] = {}
        self.clip_q: Optional["queue.Queue[Any]"] = None
        self._clip_api: Any = None
        self._deploy_silent_warn_t = 0.0
        self._last_status_t = 0.0
        self._idle_yaw_logged = False

        ctx = zmq.Context.instance()
        self.pub = ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 10)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://{args.pub_host}:{args.pub_port}")
        _log(f"pose PUB bound on tcp://{args.pub_host}:{args.pub_port} "
             f"topic={args.pub_topic!r} v{args.protocol_version} @ {self.rate:g} Hz")

        self.threads = [
            threading.Thread(
                target=_upstream_sub_thread, daemon=True, name="merger-upstream",
                kwargs=dict(
                    body_pose_url=f"tcp://{args.body_pose_sub_host}:{args.body_pose_sub_port}",
                    body_pose_topic=args.body_pose_sub_topic,
                    manager_url=f"tcp://{args.arm_and_hands_sub_host}:{args.arm_and_hands_sub_port}",
                    state=self.upstream, stop=self.stop,
                ),
            ),
            threading.Thread(
                target=_x2_debug_sub_thread, daemon=True, name="merger-x2_debug",
                kwargs=dict(
                    url=f"tcp://{args.sub_host}:{args.sub_port}", topic=args.sub_topic,
                    state=self.deploy, stop=self.stop,
                ),
            ),
        ]
        _log(f"subscribe-mode SUBs:\n"
             f"  planner   tcp://{args.body_pose_sub_host}:{args.body_pose_sub_port} "
             f"topic={args.body_pose_sub_topic!r}\n"
             f"  manager   tcp://{args.arm_and_hands_sub_host}:{args.arm_and_hands_sub_port} "
             f"topics=['arm_targets', 'hand_finger_cmd', 'stream_mode', 'recorder_cmd']\n"
             f"  x2_debug  tcp://{args.sub_host}:{args.sub_port} topic={args.sub_topic!r}")
        self._wire_motion_clips()

    # -- motion-clip wiring (lazy import: needs joblib/yaml/scipy) ---------

    def _wire_motion_clips(self) -> None:
        a = self.a
        if a.motion_clip_cmd_port <= 0:
            return
        try:
            from gear_sonic.utils.teleop import motion_clip_session as mcs
        except Exception as exc:  # noqa: BLE001
            _log(f"WARNING: motion-clip playback DISABLED (import failed: {exc}); "
                 f"motion_clip_cmd SUB not bound")
            return
        self._clip_api = mcs
        cat = str(a.gesture_catalog).strip()
        if cat:
            try:
                self.catalog = mcs.load_catalog(Path(cat))
                _log(f"gesture catalog: {cat} ({len(self.catalog)} entries)")
            except (FileNotFoundError, ValueError) as exc:
                _log(f"gesture catalog disabled: {exc} (ad-hoc --pkl play still works)")
        self.clip_q = queue.Queue()
        url = f"tcp://{a.motion_clip_cmd_host}:{a.motion_clip_cmd_port}"
        self.threads.append(threading.Thread(
            target=_clip_cmd_sub_thread, daemon=True, name="merger-clip-cmd",
            kwargs=dict(url=url, topic=a.motion_clip_cmd_topic, q=self.clip_q,
                        stop=self.stop, parse_cmd=mcs.parse_motion_clip_command),
        ))
        # Marker consumed by run_x2_pkl_direct_stack.sh ("motion_clip_cmd wired:").
        _log(f"motion_clip_cmd wired: SUB {url} topic={a.motion_clip_cmd_topic!r}")

    # -- publish helpers ---------------------------------------------------

    def _publish(
        self, *, body_q_mj: np.ndarray, motion_token: np.ndarray,
        left_hand_q: np.ndarray, right_hand_q: np.ndarray, tick: int,
        root_quat_xyzw: Optional[np.ndarray] = None,
        root_xy_world: Optional[np.ndarray] = None,
        root_z_world: Optional[float] = None,
        joint_pos_mj_future: Optional[np.ndarray] = None,
        root_quat_xyzw_future: Optional[np.ndarray] = None,
        frame_index_future: Optional[np.ndarray] = None,
        future_dt_s: Optional[float] = None,
    ) -> None:
        """Pack + send one ``pose`` frame (field order == recorder)."""
        if self.stop.is_set() and not self.in_shutdown_settle:
            return
        payload: dict[str, np.ndarray] = {
            "joint_pos_mj": np.asarray(body_q_mj, dtype=np.float32),
            "root_quat_xyzw": (
                _IDENTITY_QUAT_XYZW.copy() if root_quat_xyzw is None
                else np.asarray(root_quat_xyzw, dtype=np.float32).reshape(4)
            ),
            "motion_token": np.asarray(motion_token, dtype=np.float32),
            "left_hand_joints": np.asarray(left_hand_q, dtype=np.float32),
            "right_hand_joints": np.asarray(right_hand_q, dtype=np.float32),
            "frame_index": np.array([tick], dtype=np.int64),
        }
        if root_xy_world is not None:
            rxy = np.asarray(root_xy_world, dtype=np.float32).reshape(-1)
            if rxy.shape == (2,):
                payload["root_xy_world"] = rxy
        if root_z_world is not None:
            payload["root_z_world"] = np.array([float(root_z_world)], dtype=np.float32)
        if joint_pos_mj_future is not None and root_quat_xyzw_future is not None:
            jf = np.asarray(joint_pos_mj_future, dtype=np.float32)
            rf = np.asarray(root_quat_xyzw_future, dtype=np.float32)
            if jf.ndim == 2 and jf.shape[1] == NUM_BODY_DOFS and rf.shape == (jf.shape[0], 4):
                dt = float(future_dt_s) if future_dt_s is not None and future_dt_s > 1e-6 else 0.1
                all_j = np.concatenate([payload["joint_pos_mj"][None, :], jf], axis=0)
                payload["joint_pos_mj_future"] = jf
                payload["root_quat_xyzw_future"] = rf
                payload["joint_vel_mj_future"] = ((all_j[1:] - all_j[:-1]) / dt).astype(np.float32)
                if frame_index_future is not None:
                    fi = np.asarray(frame_index_future, dtype=np.int64).reshape(-1)
                    if fi.shape == (jf.shape[0],):
                        payload["frame_index_future"] = fi
                payload["future_dt_s"] = np.array([dt], dtype=np.float32)
        self.last_body = payload["joint_pos_mj"].astype(np.float64, copy=True)
        self.last_root_quat = payload["root_quat_xyzw"].astype(np.float32, copy=True)
        if self.upstream.estop_wire:
            payload["estop"] = np.asarray([1.0], dtype=np.float32)
        msg = pack_pose_message(
            payload, topic=self.a.pub_topic, version=int(self.a.protocol_version)
        )
        try:
            self.pub.send(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _future_frame_index(self, tick: int) -> np.ndarray:
        return np.array(
            [tick + (k + 1) * self.step_ticks for k in range(self.n_future)], dtype=np.int64
        )

    def _publish_stationary(
        self, *, body: np.ndarray, root_quat: Optional[np.ndarray], tick: int,
    ) -> None:
        """Held pose broadcast across the future window (keeps the policy balanced)."""
        body32 = np.asarray(body, dtype=np.float32)
        rot = _IDENTITY_QUAT_XYZW if root_quat is None else np.asarray(root_quat, dtype=np.float32).reshape(4)
        if self.n_future > 0:
            jf = np.broadcast_to(body32, (self.n_future, NUM_BODY_DOFS)).copy()
            rf = np.broadcast_to(rot, (self.n_future, 4)).copy()
            fi = self._future_frame_index(tick)
        else:
            jf = rf = fi = None
        self._publish(
            body_q_mj=np.asarray(body, dtype=np.float64), motion_token=self.zero_token,
            left_hand_q=self.zero_hand, right_hand_q=self.zero_hand, tick=tick,
            root_quat_xyzw=rot, joint_pos_mj_future=jf, root_quat_xyzw_future=rf,
            frame_index_future=fi, future_dt_s=self.future_dt,
        )

    def _idle_root_quat(self) -> Optional[np.ndarray]:
        """Yaw-rebased idle quat from x2_debug (hold-last-good); None = identity."""
        yaw = self.deploy.yaw()
        if yaw is None:
            return None
        if not self._idle_yaw_logged:
            _log(f"idle yaw-rebase ACTIVE: root_quat = R_z({math.degrees(yaw):.1f} deg) "
                 f"from x2_debug base_quat (held across stalls)")
            self._idle_yaw_logged = True
        return _yaw_quat_xyzw(yaw)

    def _publish_idle(self, tick: int) -> None:
        self._publish_stationary(body=self.stand, root_quat=self._idle_root_quat(), tick=tick)

    # -- return-to-stand ramp ----------------------------------------------

    def _begin_return_to_stand(self, reason: str) -> None:
        if self.last_body is None or self.last_body.shape != self.stand.shape:
            self.settle_ramp = None
            return
        rq = self.last_root_quat
        if rq is None:
            rq = self._idle_root_quat()
        total = max(1, int(round(self.rate * _RETURN_TO_STAND_RAMP_S)))
        self.settle_ramp = {"from": self.last_body.copy(), "rq": rq, "total": total, "elapsed": 0}
        _log(f"return-to-stand: ramping to idle stand over {total} ticks "
             f"(~{total / self.rate:.2f}s) [{reason}]")

    def _publish_return_to_stand_frame(self, tick: int) -> None:
        ramp = self.settle_ramp
        assert ramp is not None
        ramp["elapsed"] += 1
        total, frm = int(ramp["total"]), ramp["from"]

        def pose(step: float) -> np.ndarray:
            a = _smootherstep(step / total)
            return (1.0 - a) * frm + a * self.stand

        rot = _IDENTITY_QUAT_XYZW if ramp["rq"] is None else np.asarray(ramp["rq"], dtype=np.float32).reshape(4)
        if self.n_future > 0:
            jf = np.stack([pose(ramp["elapsed"] + (k + 1) * self.step_ticks).astype(np.float32)
                           for k in range(self.n_future)], axis=0)
            rf = np.broadcast_to(rot, (self.n_future, 4)).copy()
            fi = self._future_frame_index(tick)
        else:
            jf = rf = fi = None
        self._publish(
            body_q_mj=pose(ramp["elapsed"]), motion_token=self.zero_token,
            left_hand_q=self.zero_hand, right_hand_q=self.zero_hand, tick=tick,
            root_quat_xyzw=rot, joint_pos_mj_future=jf, root_quat_xyzw_future=rf,
            frame_index_future=fi, future_dt_s=self.future_dt,
        )
        if ramp["elapsed"] >= total:
            _log("return-to-stand: reached idle stand; holding (idle publish resumes)")
            self.settle_ramp = None

    # -- motion-clip takeover ----------------------------------------------

    def _resolve_clip_entry(self, req: Any) -> Any:
        mcs = self._clip_api
        if req.pkl_path is not None:
            return mcs.MotionClipEntry(
                name=f"adhoc:{req.pkl_path.name}", source=req.pkl_path,
                motion_key=req.motion_key, start_frame=req.start_frame,
                n_frames=req.n_frames, kind=req.kind,
            )
        if req.name not in self.catalog:
            _log(f"motion-clip PLAY: unknown name {req.name!r} "
                 f"({len(self.catalog)} catalog entries; use --pkl for ad-hoc)")
            return None
        base = self.catalog[req.name]
        if req.motion_key is None and req.start_frame == 0 and req.n_frames is None and req.kind == base.kind:
            return base
        return mcs.MotionClipEntry(
            name=base.name, source=base.source,
            motion_key=base.motion_key if req.motion_key is None else req.motion_key,
            start_frame=req.start_frame if req.start_frame else base.start_frame,
            n_frames=base.n_frames if req.n_frames is None else req.n_frames,
            hold_after=base.hold_after, kind=req.kind,
        )

    def _drain_clip_commands(self, snap: dict[str, Any]) -> None:
        if self.clip_q is None:
            return
        mcs = self._clip_api
        while True:
            try:
                req = self.clip_q.get_nowait()
            except queue.Empty:
                return
            if isinstance(req, mcs.MotionClipStopRequest):
                had_ref = self.active_clip is not None or self.held_frame is not None
                if self.active_clip is not None:
                    _log(f"motion-clip STOP (was playing {self.active_clip.entry.name!r} at "
                         f"frame {self.active_clip.current_index}/{self.active_clip.n_frames})")
                elif self.held_frame is not None:
                    _log("motion-clip STOP (releasing held pose; ramping back to idle stand)")
                self.active_clip = None
                self.active_clip_hold_after = False
                self.held_frame = None
                if had_ref:
                    self._begin_return_to_stand("clip-stop")
                continue
            entry = self._resolve_clip_entry(req)
            if entry is None:
                continue
            # Yaw seed: held frame > live x2_debug base_quat > planner snap > 0.
            if self.held_frame is not None:
                yaw, src = yaw_of_quat_xyzw(self.held_frame["root_quat_xyzw"]), "held-frame"
            else:
                _q, _any, alive, _t = self.deploy.snapshot()
                live_yaw = self.deploy.yaw() if alive else None
                if live_yaw is not None:
                    yaw, src = live_yaw, "x2_debug-base_quat"
                elif snap["root_quat_xyzw"] is not None:
                    yaw, src = yaw_of_quat_xyzw(snap["root_quat_xyzw"]), "planner-snap-fallback"
                else:
                    yaw, src = 0.0, "none(0)"
            try:
                self.active_clip = mcs.MotionClipSession(
                    entry=entry, target_rate_hz=self.rate,
                    robot_root_yaw_rad=yaw, future_dt_s=self.future_dt,
                )
            except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
                _log(f"motion-clip PLAY failed for {entry.name!r}: {exc}")
                self.active_clip = None
                self.active_clip_hold_after = False
                self.held_frame = None
                continue
            self.active_clip_hold_after = bool(
                entry.hold_after if req.hold_after is None else req.hold_after
            )
            self.held_frame = None
            self.settle_ramp = None
            _log(f"motion-clip PLAY (kind={entry.kind}) {entry.name!r}: "
                 f"{self.active_clip.n_frames} frames @ {self.rate:g} Hz "
                 f"(~{self.active_clip.duration_s:.1f}s) rebased_yaw={math.degrees(yaw):.1f}deg "
                 f"[{src}]{' (will HOLD on completion)' if self.active_clip_hold_after else ''}")

    def _publish_clip_frame(self, tick: int) -> None:
        body, rot = self.active_clip.next_frame()
        if self.n_future > 0:
            jf, rf = self.active_clip.future_window(self.n_future)
            fi = self._future_frame_index(tick)
        else:
            jf = rf = fi = None
        self._publish(
            body_q_mj=body.astype(np.float64), motion_token=self.zero_token,
            left_hand_q=self.zero_hand, right_hand_q=self.zero_hand, tick=tick,
            root_quat_xyzw=rot, joint_pos_mj_future=jf, root_quat_xyzw_future=rf,
            frame_index_future=fi, future_dt_s=self.future_dt,
        )

    # -- main loop -----------------------------------------------------------

    def run(self) -> int:
        for t in self.threads:
            t.start()
        time.sleep(0.2)  # let PUB/SUB wire up before the first frame
        a = self.a
        next_tick = time.monotonic()
        tick = 0
        wait_logged = False
        first_logged = False
        while not self.stop.is_set():
            snap = self.upstream.snapshot()
            for action, sub_tick in self.upstream.drain_recorder_cmds():
                _log(f"recorder_cmd {action!r} (tick={sub_tick}) ignored: dataset "
                     f"recording is not shipped in this port")

            self._drain_clip_commands(snap)
            if self.active_clip is not None and not self.active_clip.is_done():
                self._publish_clip_frame(tick)
                if self.active_clip.is_done():
                    if self.active_clip_hold_after:
                        self.held_frame = {
                            "body_q_mj": self.active_clip.body_q_mj[-1].astype(np.float64, copy=True),
                            "root_quat_xyzw": self.active_clip.root_quat_xyzw[-1].astype(np.float32, copy=True),
                        }
                        _log(f"motion-clip {self.active_clip.entry.name!r} completed; HOLDING "
                             f"last frame (send 'stop' or another 'play' to release)")
                    else:
                        _log(f"motion-clip {self.active_clip.entry.name!r} completed; ramping "
                             f"back to idle stand (or resuming kplanner if present)")
                        self._begin_return_to_stand("clip-completed")
                    self.active_clip = None
                    self.active_clip_hold_after = False
                tick += 1
                next_tick = self._sleep(next_tick)
                continue
            if self.held_frame is not None:
                self._publish_stationary(
                    body=self.held_frame["body_q_mj"], root_quat=self.held_frame["root_quat_xyzw"], tick=tick,
                )
                tick += 1
                next_tick = self._sleep(next_tick)
                continue
            if self.settle_ramp is not None:
                if snap["body_pose_q_mj"] is not None:
                    self.settle_ramp = None  # planner owns continuity again
                else:
                    self._publish_return_to_stand_frame(tick)
                    tick += 1
                    next_tick = self._sleep(next_tick)
                    continue

            body_pose = snap["body_pose_q_mj"]
            if body_pose is None:
                if not wait_logged:
                    _log(f"subscribe-mode: waiting for first body_pose on "
                         f"tcp://{a.body_pose_sub_host}:{a.body_pose_sub_port} "
                         f"topic={a.body_pose_sub_topic!r} ..."
                         + ("" if not a.no_idle_publish else "  [no-idle-publish: pose wire stays SILENT]"))
                    wait_logged = True
                if not a.no_idle_publish:
                    self._publish_idle(tick)
                    tick += 1
                next_tick = self._sleep(next_tick)
                continue
            if not first_logged:
                # Marker consumed by run_x2_quest3_planner_stack.sh.
                _log("subscribe-mode: first body_pose received; entering merge+publish loop")
                first_logged = True

            # Merge: planner body + manager arms (when valid) + manager hands.
            body_q = np.asarray(body_pose, dtype=np.float64).copy()
            la, ra = snap["arm_left_q"], snap["arm_right_q"]
            arm_dof = _LEFT_ARM_MJ_SLICE.stop - _LEFT_ARM_MJ_SLICE.start
            l_ok = la is not None and la.shape == (arm_dof,)
            r_ok = ra is not None and ra.shape == (arm_dof,)
            if l_ok:
                body_q[_LEFT_ARM_MJ_SLICE] = la
            if r_ok:
                body_q[_RIGHT_ARM_MJ_SLICE] = ra
            jf_planner = snap["joint_pos_mj_future"]
            rf_planner = snap["root_quat_xyzw_future"]
            jf_out: Optional[np.ndarray] = None
            if (
                jf_planner is not None and rf_planner is not None
                and jf_planner.ndim == 2 and jf_planner.shape[1] == NUM_BODY_DOFS
                and rf_planner.shape == (jf_planner.shape[0], 4)
            ):
                jf_out = jf_planner.astype(np.float32, copy=True)
                if l_ok:
                    jf_out[:, _LEFT_ARM_MJ_SLICE] = la.astype(np.float32, copy=False)
                if r_ok:
                    jf_out[:, _RIGHT_ARM_MJ_SLICE] = ra.astype(np.float32, copy=False)
            lh = snap["left_hand_q"] if snap["left_hand_q"] is not None else self.zero_hand
            rh = snap["right_hand_q"] if snap["right_hand_q"] is not None else self.zero_hand
            wt = snap["wire_motion_token"]
            token = wt.astype(np.float64, copy=False) if wt is not None and wt.shape == (SONIC_MOTION_TOKEN_DIM,) else self.zero_token
            self._publish(
                body_q_mj=body_q, motion_token=token, left_hand_q=lh, right_hand_q=rh,
                tick=tick, root_quat_xyzw=snap["root_quat_xyzw"],
                root_xy_world=snap["root_xy_world"], root_z_world=snap["root_z_world"],
                joint_pos_mj_future=jf_out, root_quat_xyzw_future=rf_planner,
                frame_index_future=snap["frame_index_future"], future_dt_s=snap["future_dt_s"],
            )

            now = time.monotonic()
            self._maybe_warn_deploy_silent(now)
            if now - self._last_status_t >= 5.0:
                self._last_status_t = now
                _log(f"status: tick={tick} mode={snap['stream_mode']} "
                     f"arms={'manager' if (l_ok or r_ok) else 'planner'} "
                     f"hand|L|={float(np.linalg.norm(lh)):.3f} hand|R|={float(np.linalg.norm(rh)):.3f}")
            tick += 1
            next_tick = self._sleep(next_tick)
        return tick

    def _sleep(self, next_tick: float) -> float:
        next_tick += self.period
        slack = next_tick - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        return next_tick

    def _maybe_warn_deploy_silent(self, now: float) -> None:
        _q, received_any, alive, last_rx = self.deploy.snapshot()
        if not received_any or alive:
            self._deploy_silent_warn_t = 0.0
            return
        if self._deploy_silent_warn_t == 0.0 or now - self._deploy_silent_warn_t >= _DEPLOY_SILENT_REWARN_S:
            self._deploy_silent_warn_t = now
            _log(f"WARNING: deploy x2_debug silent for {now - last_rx:.1f}s "
                 f"(tcp://{self.a.sub_host}:{self.a.sub_port}); pose wire keeps publishing")

    # -- shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        if self.stop.is_set():
            return
        self.stop.set()
        if self.a.no_idle_publish or self.last_body is None:
            return
        try:
            last = self.last_body if self.last_body.shape == self.stand.shape else self.stand
            rq = self.last_root_quat if self.last_root_quat is not None else self._idle_root_quat()
            ramp_ticks = max(1, int(round(self.rate * _SHUTDOWN_SETTLE_RAMP_S)))
            hold_ticks = max(1, int(round(self.rate * _SHUTDOWN_SETTLE_HOLD_S)))
            _log(f"shutdown: settling to stand pose over {ramp_ticks + hold_ticks} ticks "
                 f"(~{(ramp_ticks + hold_ticks) * self.period:.2f}s) before the wire goes silent")
            self.in_shutdown_settle = True
            next_tick = time.monotonic()
            for k in range(ramp_ticks + hold_ticks):
                a = _smootherstep((k + 1) / ramp_ticks) if k < ramp_ticks else 1.0
                self._publish_stationary(body=(1.0 - a) * last + a * self.stand, root_quat=rq, tick=k)
                next_tick = self._sleep(next_tick)
        except Exception as exc:  # noqa: BLE001 - never block teardown
            _log(f"shutdown: settle-to-stand FAILED ({exc!r}); proceeding with teardown")
        finally:
            self.in_shutdown_settle = False


# ── CLI ───────────────────────────────────────────────────────────────────

# Recorder-only flags: accepted for launcher compatibility, ignored with a
# warning. (value-taking flags listed with nargs=1 via the tuple form)
_IGNORED_FLAGS: dict[str, bool] = {
    "--sonic-checkpoint": True, "--sonic-tokenizer-device": True,
    "--encoder-config": True, "--obs-dump-recorder": True,
    "--render-width": True, "--render-height": True, "--no-omnihand": False,
    "--hand-input": True, "--no-finger-filter": False,
    "--finger-filter-alpha": True, "--finger-filter-hold-window": True,
    "--finger-filter-hold-std": True, "--apply-curl-compensation": False,
    "--apply-oppose-compensation": False, "--sonic-correction-warn-rad": True,
    "--no-sonic-correction-log": False, "--ik-damping": True,
    "--ik-rotation-weight": True, "--ik-per-tick-step-rad": True,
    "--calibration": True, "--recalibrate": False, "--operator-id": True,
    "--front-cam": False, "--no-front-cam": False, "--head-cameras": False,
    "--no-head-cameras": False, "--camera-host": True, "--camera-port": True,
    "--camera-warmup-timeout": True, "--camera-max-staleness": True,
    "--ready-file": True, "--embodiment-tag": True,
    "--quest3-ws-port": True, "--quest3-http-port": True, "--quest3-no-ssl": False,
}
_UNSUPPORTED_FLAGS = (
    "--output-dir", "--task", "--robocasa-env", "--scene-xml-path",
    "--scene-state-sub-host", "--scene-state-sub-port",
    "--scene-reset-pub-host", "--scene-reset-pub-port", "--episode-seed",
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--teleop-only", action="store_true",
                   help="accepted for compatibility (always teleop-only)")
    p.add_argument("--no-idle-publish", action="store_true",
                   help="do not publish the stand pose before the first body_pose "
                        "(and skip the shutdown settle ramp)")
    p.add_argument("--body-pose-source", choices=("internal", "zmq", "vla"), default="zmq")
    p.add_argument("--arm-targets-source", choices=("internal", "zmq", "vla"), default="zmq")
    p.add_argument("--body-pose-sub-host", default="localhost")
    p.add_argument("--body-pose-sub-port", type=int, default=5565)
    p.add_argument("--body-pose-sub-topic", default="body_pose")
    p.add_argument("--arm-and-hands-sub-host", default="localhost")
    p.add_argument("--arm-and-hands-sub-port", type=int, default=5564)
    p.add_argument("--pub-host", default="*", help="bind iface for the pose PUB")
    p.add_argument("--pub-port", type=int, default=5556)
    p.add_argument("--pub-topic", default="pose")
    p.add_argument("--sub-host", default="localhost", help="deploy x2_debug host")
    p.add_argument("--sub-port", type=int, default=5557)
    p.add_argument("--sub-topic", default="x2_debug")
    p.add_argument("--protocol-version", type=int, choices=(3, 4), default=4)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--motion-clip-cmd-host", default="*")
    p.add_argument("--motion-clip-cmd-port", type=int, default=5568,
                   help="motion_clip_cmd SUB bind port (0 = disable clip playback)")
    p.add_argument("--motion-clip-cmd-topic", default="motion_clip_cmd")
    p.add_argument("--gesture-catalog", type=str,
                   default=str(REPO_ROOT / "gear_sonic/data/motions/gestures/gestures_v1.yaml"),
                   help="gesture catalog YAML ('' disables; ad-hoc --pkl plays still work)")
    p.add_argument("--clip-future-dt-s", type=float, default=0.1)
    p.add_argument("--clip-future-window-frames", type=int, default=9)
    p.add_argument("--quiet", action="store_true")
    for flag, takes_value in _IGNORED_FLAGS.items():
        if takes_value:
            p.add_argument(flag, default=None, help=argparse.SUPPRESS)
        else:
            p.add_argument(flag, action="store_true", help=argparse.SUPPRESS)
    for flag in _UNSUPPORTED_FLAGS:
        p.add_argument(flag, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    bad = [f for f in _UNSUPPORTED_FLAGS if getattr(args, f[2:].replace("-", "_")) is not None]
    if bad:
        raise SystemExit(f"{LOG} ERROR: {', '.join(bad)} not supported: dataset recording "
                         f"(LeRobot writer / RoboCasa mirror) is not shipped in this port. "
                         f"Run teleop-only (drop the flags or pass --teleop-only).")
    if args.body_pose_source != "zmq" or args.arm_targets_source != "zmq":
        raise SystemExit(f"{LOG} ERROR: only --body-pose-source zmq / --arm-targets-source zmq "
                         f"(planner + manager subscribe mode) are supported.")
    ignored = [f for f, tv in _IGNORED_FLAGS.items()
               if (getattr(args, f[2:].replace("-", "_")) not in (None, False))]
    if ignored:
        _log(f"WARNING: recorder-only flags ignored (no dataset / tokenizer / renderer "
             f"in this port): {' '.join(ignored)}")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    merger = PoseMerger(args)

    def _on_signal(signum: int, _frame: Any) -> None:
        _log(f"caught signal {signum}, shutting down ...")
        merger.shutdown()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        merger.run()
    finally:
        merger.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
