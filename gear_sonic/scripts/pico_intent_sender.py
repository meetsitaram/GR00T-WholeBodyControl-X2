#!/usr/bin/env python3
"""LAPTOP intent sender — whole-body teleop, Quest-split architecture.

Streams only STATE-INDEPENDENT human intent to the PC2 token service
(pc2_pico_token_service.py): per-frame SMPL joints + global orient, the
engage flag, and controller-derived wrist targets. No model, no
robot state, nothing that can go stale into an observation — encoding
happens robot-side. Wifi loss simply stops the stream and the hybrid
deploy falls back to the pad.

Dual-mode controls (2026-09-02, Quest control grammar — same chords as
quest3_manager_x2 / IntentDecoder):

    A+B+X+Y chord   master toggle OFF <-> LOCOMOTION
    B single        LOCOMOTION <-> WHOLE_BODY (Quest's ARM_MAN slot)
    sticks (LOCO)   kplanner locomotion, identical to the Quest stack
                    (L-stick fwd/side, R-stick yaw, A/X modifiers,
                    X/Y speed trim) -> planner_cmd{source:"vr"} :5563
    A single (WB)   engage / disengage the whole-body token stream
                    (body-freshness gated, same guard as before)
    A+X + rapid both-trigger pumps   E-STOP (soft, ~1s more -> damp)
    grips           RESERVED for finger open/close (phase 2) — the old
                    dual-grip engage toggle is REMOVED (accidental
                    engages, operator 2026-09-02)

Locomotion authority: kplanner (pad primary, vr planner_cmd overlay).
Whole-body authority: the deploy's per-tick token arbiter — engage
publishes one idle+vr_release planner_cmd so the planner drops to
idle under the token overlay; disengage simply drops the engaged flag
and the deploy soft-ramps back to the kplanner reference.

    .venv_teleop/bin/python gear_sonic/scripts/pico_intent_sender.py \
        --pc2-host ${PC2_HOST}            # robot session
    ... --pc2-host 127.0.0.1              # sim rehearsal (same service)
    ... --tape-replay <npz> --auto-engage # headset-free
"""
from __future__ import annotations

import argparse
import math
import os
import dataclasses
import json
import threading
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
# Session / intent tapes are runtime OUTPUT (not shipped). Override with
# PICO_TAPES_DIR; default <repo>/logs/pico_tapes.
_PICO_TAPES_DIR = Path(os.environ.get("PICO_TAPES_DIR", str(SCRIPTS.parents[1] / "logs/pico_tapes")))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[1]))

from scipy.spatial.transform import Rotation as Rot  # noqa: E402
from live_pico_smpl_teleop import LiveSmplSource, _audio_cue  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402
from gear_sonic.utils.teleop.vr.button_state_machine import ButtonStateMachine  # noqa: E402
from gear_sonic.utils.teleop.vr.intent_decoder import (  # noqa: E402
    IntentDecoder, LocomotionCmd, StreamMode)
from gear_sonic.utils.teleop.estop_gesture import EstopGesture  # noqa: E402
import zmq  # noqa: E402

CONTROL_DT = 0.02
# X2 wrist ranges (x2_ultra.xml): pitch +-0.558 both sides; roll is
# MIRRORED: left (-1.571, 0.724), right (-0.724, 1.571).
WRIST_RANGE = {"p": (-0.56, 0.56), "r": (-1.57, 0.72), "r_right": (-0.72, 1.57)}


class _PadAbortWatch:
    """Replay abort from the GAMEPAD (operator 2026-09-05: "we need the gamepad
    to stop the replay at any time"): SUB to the pad daemon's ``pad_state``
    feed (PC2 :5569, JSON {axes, buttons, hats}) and raise ``fired`` on a
    chord -- LB+RB (the pad's stop chord, "L1+R1") or RB+B (mirrors the
    Pico's right-trigger + B). Button indices per pad_locomotion_bridge.py
    (SDL Xbox: A=0 B=1 X=2 Y=3 LB=4 RB=5). Silent if no feed (sim without
    the pad daemon): Ctrl-C is the stop there."""

    BTN = {"a": 0, "b": 1, "x": 2, "y": 3, "lb": 4, "rb": 5}

    def __init__(self, host: str, port: int, chords: list[tuple[str, ...]]):
        self.fired = threading.Event()
        self.chord = ""
        self.seen = False
        self._chords = chords
        self._sub = zmq.Context.instance().socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"pad_state")
        self._sub.setsockopt(zmq.RCVTIMEO, 200)
        self._sub.connect(f"tcp://{host}:{port}")
        self._stop = threading.Event()
        threading.Thread(target=self._run, name="pad-abort-watch", daemon=True).start()

    def _run(self) -> None:
        import json as _json
        while not self._stop.is_set():
            try:
                _, payload = self._sub.recv_multipart()
            except zmq.Again:
                continue
            except Exception:
                break
            try:
                bt = _json.loads(payload).get("buttons", [])
            except Exception:
                continue
            self.seen = True
            down = {k for k, i in self.BTN.items() if i < len(bt) and bt[i]}
            for ch in self._chords:
                if all(k in down for k in ch):
                    self.chord = "+".join(k.upper() for k in ch)
                    self.fired.set()
                    return

    def stop(self) -> None:
        self._stop.set()


class _X2DebugWatch:
    """Latest MEASURED robot joints from the deploy's ``x2_debug`` PUB (PC2 :5557,
    packed frame with topic prefix; ``body_q`` f64[31] MuJoCo order). Used to
    snapshot the ARM pose at an X/Y record start (``arms0``) so a replay can
    put the arms back there before the tape rolls (operator 2026-09-05: the
    replay "is just going with whatever the current arm state is")."""

    def __init__(self, host: str, port: int):
        from gear_sonic.utils.pose_pipeline.wire import decode_x2_debug_fields
        self._decode = decode_x2_debug_fields
        self.body_q = None
        self.t_rx = 0.0
        self._sub = zmq.Context.instance().socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"x2_debug")
        self._sub.setsockopt(zmq.RCVTIMEO, 200)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.connect(f"tcp://{host}:{port}")
        self._stop = threading.Event()
        threading.Thread(target=self._run, name="x2-debug-watch", daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._sub.recv()
            except zmq.Again:
                continue
            except Exception:
                break
            # DRAIN to the newest frame: the deploy publishes at 50 Hz and a
            # subscriber that lags accumulates a queue -- on 2026-09-05 the
            # arms0 snapshot came from a frame 18.8 s old while "arrival
            # time" said fresh. Freshness is judged by the FRAME's own
            # ros_timestamp below, never by arrival.
            while True:
                try:
                    msg = self._sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
            f = self._decode(msg, ("body_q", "ros_timestamp"))
            if f and "body_q" in f:
                self.body_q = np.asarray(f["body_q"], np.float32)
                self.t_rx = time.monotonic()
                ts = f.get("ros_timestamp")
                if ts is not None:
                    ts = float(np.asarray(ts).ravel()[0])
                    if self._ts0 is None:
                        self._ts0, self._rx0 = ts, self.t_rx
                    # ros_timestamp is the DEPLOY's clock (not wall time): the
                    # lag of this frame = how far its clock has fallen behind
                    # our arrival clock since the first frame.
                    self.lag_s = max(0.0, (self.t_rx - self._rx0) - (ts - self._ts0))

    _ts0 = None; _rx0 = 0.0; lag_s: float = float("inf")

    def stop(self) -> None:
        self._stop.set()

    def frame_age_s(self) -> float:
        if self.body_q is None:
            return float("inf")
        return self.lag_s + (time.monotonic() - self.t_rx)

    def fresh_arms(self, max_age_s: float = 1.0):
        """arm joints MJ 15..28 (7 left + 7 right, wrists included) or None
        unless the newest frame is < max_age_s old by ITS OWN timestamp."""
        if self.body_q is None or self.frame_age_s() > max_age_s:
            return None
        return self.body_q[15:29].copy()


class _IntentTapeSource:
    """Replays an intent tape (pico_intent_sender --record output) with the
    LiveSmplSource.window() contract the send loop expects. Holds the final
    frame after the tape ends. replay_engaged/replay_wrist expose the
    recorded flags so the sim session mirrors the robot session exactly."""

    def __init__(self, path: str):
        z = np.load(path)
        self.joints = np.asarray(z["smpl_joints"], np.float32).reshape(-1, 24, 3)
        self.quats = np.asarray(z["human_quat"], np.float32)
        self.engaged = np.asarray(z["engaged"], np.float32)
        self.wrist = np.asarray(z["wrist_pr"], np.float32)
        self.rel_t = np.asarray(z["t"], np.float64)
        self.rel_t = self.rel_t - self.rel_t[0]
        self.t0 = time.monotonic()
        print(f"[intent] replaying intent tape {path}: "
              f"{len(self.rel_t)} frames, "
              f"{int((self.engaged > 0.5).sum())} engaged", flush=True)

    def _idx(self, now: float) -> int:
        h = getattr(self, "_hold", None)
        if h is not None:
            start, end, idx = h
            if now < end:
                return idx                   # frozen at the engage frame (settle)
            self.t0 += (end - start)         # resume from the same frame, no jump
            self._hold = None
        return int(min(np.searchsorted(self.rel_t, now - self.t0),
                       len(self.rel_t) - 1))

    def finished(self, now: float) -> bool:
        """True once the tape's last frame has been reached (hold shifts included) -- lets --tape-once end an
        INTENT-tape replay too (before 2026-09-08 intent tapes streamed their last frame until the timeout)."""
        h = getattr(self, "_hold", None)
        if h is not None and now < h[1]:
            return False
        return (now - self.t0) >= float(self.rel_t[-1])

    def hold(self, now: float, secs: float) -> None:
        """Settle (2026-09-08, operator protocol: engage whole-body on a STILL operator, let the robot settle,
        THEN the motion): freeze the tape at the current frame for ``secs`` — the engaged flag and posture of that
        frame keep streaming, so the deploy engages and the robot settles on it before the recorded motion rolls."""
        if secs > 0:
            self._hold = (now, now + secs, self._idx(now))

    def window(self, now: float):
        i = self._idx(now)
        return (self.joints[max(0, i - 9):i + 1],
                self.quats[max(0, i - 9):i + 1])

    def replay_engaged(self, now: float) -> bool:
        return bool(self.engaged[self._idx(now)] > 0.5)

    def replay_wrist(self, now: float) -> np.ndarray:
        return self.wrist[self._idx(now)]

    def grips(self):
        return (0.0, 0.0)

    def stop(self) -> None:
        pass


from gear_sonic.utils.teleop.x2_hand_retarget import (  # noqa: E402
    controller_grasp_ratio, grasp_command_from_ratio)
import msgpack  # noqa: E402


def _pr(q_xyzw):
    e = Rot.from_quat(q_xyzw).as_euler("ZYX")
    return float(e[1]), float(e[2])


def _parse_wrist_map(spec: str) -> dict:
    """'lp:+z,lr:-y,rp:+z,rr:+y' -> {key: (sign, axis_idx)}; axis in the
    SMPL wrist's local (forearm) frame, x=0 y=1 z=2."""
    out = {}
    for item in spec.split(","):
        k, v = item.strip().split(":")
        sign = -1.0 if v.strip().startswith("-") else 1.0
        out[k.strip()] = (sign, "xyz".index(v.strip()[-1].lower()))
    for k in ("lp", "lr", "rp", "rr"):
        if k not in out:
            raise ValueError(f"WRIST_SMPL_MAP missing {k}")
    return out


_WRIST_MAP = _parse_wrist_map(
    __import__("os").environ.get("WRIST_SMPL_MAP", "lp:+y,lr:+z,rp:-y,rr:+z"))


# Pronation sign per side (left, right): the wrist_yaw axis runs along the
# bone, opposite sense per side. Override for a live A/B: WRIST_YAW_SIGN="+1,-1".
_YAW_SIGN = tuple(float(v) for v in __import__("os").environ.get("WRIST_YAW_SIGN", "-1,+1").split(","))


# FRAME-BASED wrist conversion (2026-09-04 16:50, default). The SMPL wrist
# rotation is the hand relative to the forearm, expressed in the forearm's
# rest frame (SMPL: x = bone, y = up, z = forward; palm down in T-pose).
# The X2 wrist chain is wrist_yaw (link Z, along the forearm) -> wrist_pitch
# (link Y) -> wrist_roll (link X), i.e. R_hand = Rz(yaw) Ry(pitch) Rx(roll)
# in the forearm LINK frame. The fixed SMPL->link map (composed OmniHand
# model, arms hanging, palms inward, thumbs forward):
#   left : x_s -> -Z_w, y_s -> +Y_w, z_s -> +X_w ; right: x_s -> +Z_w,
#          y_s -> -Y_w, z_s -> +X_w (both det +1), and the link frame is the
# world frame rolled 12.1 deg about X (wrist_yaw axis = (0,-0.21,0.98) L,
# (0,0.21,0.98) R at the zero pose). A rotvec COMPONENT is only an
# approximation of a chain angle once pronation and flexion happen
# together; this decomposes the actual rotation. WRIST_SMPL_MAP (set
# explicitly) falls back to the old per-axis pick.
from gear_sonic.utils.teleop.x2_wrist_map import wrist_frame_ypr as _wrist_frame_ypr  # noqa: E402

# ---- Head yaw from the operator's SMPL neck + head (2026-09-08) ------------
# Route = the pad head-look path (commit 123e1b2), no deploy change:
#   planner_cmd {"intent":"head_targets","yaw_rad":..}
#   -> kplanner slew-limited overlay on DOF 29 (cap KPLANNER_HEAD_YAW_MAX_RAD)
#   -> deploy --head-bypass ref (samples the pose-ref every tick, also under
#      token authority since the BeginTick fix 2026-09-04).
# SMPL frames: Y up, +Z forward, +X = the body's LEFT. Yaw = rotation of the
# head's forward vector about +Y, positive = look LEFT. X2 head_yaw_joint
# axis is +Z (up), so positive = left there too -> default sign +1.
_HEAD_YAW_SIGN: float = float(os.environ.get("HEAD_YAW_SIGN", "1.0"))
_HEAD_YAW_GAIN: float = float(os.environ.get("HEAD_YAW_GAIN", "1.0"))
_HEAD_YAW_MAX: float = float(os.environ.get("HEAD_YAW_MAX_RAD", "0.35"))
_HEAD_YAW_DEADBAND: float = math.radians(
    float(os.environ.get("HEAD_YAW_DEADBAND_DEG", "2.0")))
# "zero" (default): straight ahead relative to the torso = 0, the robot
# looks where the operator looks. "engage": the head pose at engage is the
# neutral (wrist parity) -- for sessions where the operator watches the
# robot off to one side (09-08 seg08 sat at +27 deg for the whole clip).
_HEAD_YAW_NEUTRAL: str = os.environ.get("HEAD_YAW_NEUTRAL", "zero")
HEAD_TX_PERIOD_S: float = 0.04          # 25 Hz, wrist_targets parity


def _smpl_head_yaw(aa: "np.ndarray") -> float:
    """Head yaw relative to the torso (spine3 frame) from SMPL local
    rotvecs [neck(12), head(15)]: compose parent-first, take the forward
    vector, atan2 in the horizontal plane. Radians, +left."""
    r = Rot.from_rotvec(np.asarray(aa[0], np.float64)) * \
        Rot.from_rotvec(np.asarray(aa[1], np.float64))
    f = r.apply([0.0, 0.0, 1.0])
    return float(math.atan2(f[0], f[2]))


def _head_yaw_cmd(raw: float) -> float:
    """Sign/gain/deadband/clamp -> the value put on the wire."""
    y = _HEAD_YAW_SIGN * _HEAD_YAW_GAIN * raw
    if abs(y) < _HEAD_YAW_DEADBAND:
        y = 0.0
    return float(max(-_HEAD_YAW_MAX, min(_HEAD_YAW_MAX, y)))


_USE_FRAME_MAP = "WRIST_SMPL_MAP" not in __import__("os").environ


def _wrist_map(aa: np.ndarray, key: str) -> float:
    """aa: (2,3) local wrist rotvecs [left, right]; key lp/lr/rp/rr."""
    sign, axis = _WRIST_MAP[key]
    side = 0 if key[0] == "l" else 1
    return float(sign * aa[side, axis])


def _planner_cmd_payload(cmd: LocomotionCmd, release: bool = False) -> bytes:
    """kplanner planner_cmd JSON — same wire the Quest manager builds
    (quest3_manager_x2._planner_cmd_payload); source:"vr" takes stick
    ownership from the pad, vr_release=True hands it back."""
    payload: dict = {"intent": cmd.intent, "magnitude": cmd.magnitude,
                     "source": "vr"}
    if release:
        payload["vr_release"] = True
    if cmd.intent == "hold_torso":
        payload["waist_pitch_deg"] = float(cmd.waist_pitch_deg)
        payload["waist_roll_deg"] = float(cmd.waist_roll_deg)
        payload["waist_yaw_deg"] = float(cmd.waist_yaw_deg)
        if cmd.hip_height_m is not None:
            payload["hip_height_m"] = float(cmd.hip_height_m)
    elif cmd.intent == "locomotion":
        payload["stick_fwd"] = float(cmd.stick_fwd)
        payload["stick_side"] = float(cmd.stick_side)
        payload["stick_yaw"] = float(cmd.stick_yaw)
        if getattr(cmd, "speed_delta", 0.0):
            payload["speed_delta"] = float(cmd.speed_delta)
    return json.dumps(payload).encode("utf-8")


class _ButtonDefer:
    """Hold single-press events 0.15 s; drop them if a chord forms
    (the A+X e-stop chord lands across 1-2 ticks and its leading edge
    used to fire A/B singles — operator incident 2026-08-03). Chord
    events pass through untouched. Port of the Quest manager's
    _defer_button_singles."""

    def __init__(self):
        self._pend: dict = {}

    def tick(self, ev, chord_ax_held: bool, now: float):
        for nm in ("a", "b", "x", "y"):
            if getattr(ev, f"{nm}_pressed"):
                self._pend[nm] = now + 0.15
        # Only the chords we ACTUALLY use suppress singles: A+B+X+Y (mode
        # master) and A+X (e-stop). A+B / X+Y / B+Y are NOT ours, so a
        # sloppy two-button press must still deliver its single (robot
        # 2026-09-02: A+B landing on the same tick ate the B mode-toggle).
        if chord_ax_held or ev.abxy_pressed or ev.ax_pressed:
            self._pend.clear()
        fired = [nm for nm, ft in self._pend.items() if ft <= now]
        for nm in fired:
            del self._pend[nm]
        return dataclasses.replace(
            ev,
            a_pressed="a" in fired, b_pressed="b" in fired,
            x_pressed="x" in fired, y_pressed="y" in fired,
        )


class _ButtonDebounce:
    """Filters phantom multi-presses from the Pico SDK: a button's NEW
    state must persist ``ticks`` consecutive polls before it is accepted
    (robot 2026-09-02: a single physical A press reported transient
    B/X/Y for 1-2 ticks, forming accidental chords). Real presses hold
    far longer than the phantoms."""

    def __init__(self, ticks: int = 3):
        self.ticks = ticks
        self._stable = [False, False, False, False]
        self._cand = [False, False, False, False]
        self._cnt = [0, 0, 0, 0]

    def update(self, raw, now=None):
        out = []
        for i, r in enumerate(raw):
            if r == self._stable[i]:
                self._cnt[i] = 0
                self._cand[i] = r
            else:
                if r != self._cand[i]:
                    self._cand[i] = r
                    self._cnt[i] = 1
                else:
                    self._cnt[i] += 1
                if self._cnt[i] >= self.ticks:
                    self._stable[i] = r
                    self._cnt[i] = 0
            out.append(self._stable[i])
        return tuple(out)


def _poll_controllers(xrt):
    """One tick of stick/button/trigger state from the XRoboToolkit SDK.
    Returns (lx, ly, rx, ry, a, b, x, y, lt, rt) or None when the SDK
    read fails (dead link — caller treats as all-neutral)."""
    try:
        la, ra = xrt.get_left_axis(), xrt.get_right_axis()
        return (float(la[0]), float(la[1]), float(ra[0]), float(ra[1]),
                bool(xrt.get_A_button()), bool(xrt.get_B_button()),
                bool(xrt.get_X_button()), bool(xrt.get_Y_button()),
                float(xrt.get_left_trigger()), float(xrt.get_right_trigger()))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pc2-host", required=True,
                    help="token service host (robot: PC2; sim: 127.0.0.1)")
    ap.add_argument("--intent-port", type=int, default=5573)
    ap.add_argument("--hand-source", choices=["grip", "trigger", "max", "off"], default="grip",
                    help="OmniHand fingers from the controllers: 'grip' (default; the "
                         "triggers are the stick deadman / e-stop chord), 'trigger', "
                         "'max', or 'off'. Published as hand_finger_cmd "
                         "{left_hand_q[10], right_hand_q[10]} (the Quest manager's wire) "
                         "to the kplanner's arm-ingest port; the kplanner overlays them "
                         "onto the pose wire -> sim bridge (--with-omnihand) / PC2 "
                         "x2_hand_zmq_to_aimdk_bridge.py drive the fingers.")
    ap.add_argument("--hand-close-on-grip", action="store_true",
                    help="legacy sense: fingers OPEN at rest, squeeze to close. Default (operator "
                         "2026-09-04): fingers CLOSED at rest, squeeze the grip to OPEN.")
    ap.add_argument("--hand-port", type=int, default=5572,
                    help="kplanner --arm-port (SUB bind) for hand_finger_cmd (default 5572)")
    ap.add_argument("--hand-hz", type=float, default=25.0)
    ap.add_argument("--planner-cmd-port", type=int, default=None,
                    help="kplanner planner_cmd SUB-bind port (PC2 ritual "
                         "runs pc2_kplanner_onnx --cmd-bind); locomotion-"
                         "mode sticks publish here as source:'vr'. Default "
                         "5563 (robot); 5663 when --pc2-host is localhost "
                         "(PORT_REGISTRY: sim renames the core chain +100)")
    ap.add_argument("--micro-keys", default=os.environ.get("MICRO", ""),
                    help="8-way MICRO-STEP replay bank on the LEFT stick (operator 2026-09-05 "
                         "backup plan): comma separated motion keys in the order F,B,L,R,FL,FR,BL,BR "
                         "(x2m2 bakes in the stack's dances dir). Fires once per 45-deg sector entry "
                         "while the walk deadman (L trigger) is RELEASED, gated by --micro-gate. "
                         "Default from $MICRO (x2_pc2/pad_bindings.env); empty = off.")
    ap.add_argument("--micro-gate", choices=["right-trigger", "right-click", "none"], default="right-trigger",
                    help="'right-trigger' (default): hold the RIGHT index-finger trigger (the operator's "
                         "'R1'; the left one is the walk deadman) while flicking the left stick; "
                         "'right-click': press the right stick instead; 'none': bare stick.")
    ap.add_argument("--micro-deflect", type=float, default=0.6)
    ap.add_argument("--micro-cooldown-s", type=float, default=3.5)
    ap.add_argument("--clip-port", type=int, default=5568,
                    help="kplanner motion_clip_cmd SUB-bind port (PC2 ritual 5568; local sim stack 5668)")
    ap.add_argument("--no-locomotion", action="store_true",
                    help="disable the dual-mode controller layer entirely "
                         "(no planner_cmd socket, no mode machine; whole-"
                         "body only, --auto-engage to engage)")
    ap.add_argument("--bucketed-locomotion", action="store_true",
                    help="legacy Quest bucketed step/turn bins instead of "
                         "continuous analog sticks. Continuous is the DEFAULT "
                         "(robot test 2026-09-02: bucketed 45-deg turn bins "
                         "felt wildly oversensitive vs the pad; the kplanner "
                         "path is built for continuous stick_fwd/side/yaw)")
    ap.add_argument("--stick-deadzone", type=float, default=0.30)
    ap.add_argument("--invert-lx", action="store_true")
    ap.add_argument("--invert-ly", action="store_true")
    ap.add_argument("--invert-rx", action="store_true")
    ap.add_argument("--invert-ry", action="store_true")
    ap.add_argument("--tape-replay", default=None)
    ap.add_argument("--auto-engage", action="store_true")
    ap.add_argument("--wb-toggle-gate", choices=["right-trigger", "none"], default="right-trigger",
                    help="deadman for the B mode switch (LOCOMOTION <-> WHOLE_BODY): 'right-trigger' "
                         "(default) = B counts only while the right trigger is held; 'none' = bare B "
                         "(pre-2026-09-05 behaviour). Guards against an inadvertent release mid-walk.")
    ap.add_argument("--x2-debug-port", type=int, default=5557,
                    help="deploy x2_debug PUB on <pc2-host>: an X/Y record snapshots the measured ARM joints "
                         "(arms0) at start; a replay sends them to the kplanner arm overlay before the tape "
                         "rolls (0 = off)")
    ap.add_argument("--arms0-settle-s", type=float, default=1.5,
                    help="replay: seconds to slew the arms to the record's arms0 before the tape starts")
    ap.add_argument("--pad-abort-port", type=int, default=5569,
                    help="with --tape-replay: SUB the pad daemon's pad_state feed on <pc2-host>:PORT and abort the "
                         "replay on a gamepad chord (0 = off)")
    ap.add_argument("--pad-abort-chords", default="lb+rb,rb+b",
                    help="gamepad chords that abort a replay (comma list of '+'-joined a/b/x/y/lb/rb); "
                         "default: LB+RB (the pad's stop chord) or RB+B (mirror of the Pico's R-trigger + B)")
    ap.add_argument("--tape-exit-s", type=float, default=5.0,
                    help="with --tape-once: keep streaming idle this long after the tape ends (the deploy "
                         "finishes its release/return on a live stream), then EXIT. <0 = stream until Ctrl-C "
                         "(operator 2026-09-05: 'the replay command never terminated').")
    ap.add_argument("--tape-settle-s", type=float, default=0.0,
                    help="intent-tape replay: after each ENGAGE, hold the engage frame this long (engaged flag + "
                         "posture keep streaming) before the recorded motion rolls -- the operator's protocol "
                         "(locomotion first, engage on a still operator, settle, then move); 0 = verbatim tape")
    ap.add_argument("--tape-once", action="store_true",
                    help="play the raw tape ONCE (no loop-wrap snap): at the end of the tape the sender "
                         "RELEASES whole-body (engaged=0 on the wire -> deploy hold -> kplanner return) and "
                         "keeps streaming idle until Ctrl-C. Use for every replay onto the ROBOT.")
    ap.add_argument("--no-wrist-follow", action="store_true",
                    help="alias for --wrist-source off")
    ap.add_argument("--wrist-source", choices=("smpl", "controller", "off"),
                    default="smpl",
                    help="where the X2 wrist pitch/roll targets come from "
                         "(2026-09-04): 'smpl' = the operator's OWN wrist "
                         "rotation from Pico body tracking (local wrist "
                         "rotvec in the forearm frame, neutral captured at "
                         "engage); 'controller' = the pre-09-04 controller "
                         "orientation deltas; 'off' = zeros. Axis/sign map "
                         "for 'smpl' via WRIST_SMPL_MAP (default "
                         "'%s')." % "lp:+z,lr:-y,rp:+z,rr:+y")
    ap.add_argument("--head-source", choices=("smpl", "off"), default="smpl",
                    help="robot head YAW while whole-body is engaged "
                         "(2026-09-08, operator: 'hook up the smpl head "
                         "motion to the robot head, left to right'): 'smpl' "
                         "= the operator's own neck+head rotation from Pico "
                         "body tracking, sent as planner_cmd head_targets "
                         "and served through the kplanner's head-look "
                         "overlay + the deploy head bypass (the pad "
                         "head-look route, commit 123e1b2). Knobs: "
                         "HEAD_YAW_SIGN (+1), HEAD_YAW_GAIN (1.0), "
                         "HEAD_YAW_MAX_RAD (0.35), HEAD_YAW_DEADBAND_DEG (2).")
    ap.add_argument("--rate-report-s", type=float, default=10.0)
    ap.add_argument("--no-record", action="store_true",
                    help="disable the intent tape. DEFAULT IS RECORD for live "
                         "sessions (sim2real plan phase 0: every session must "
                         "be replayable); tape-replay never re-records.")
    ap.add_argument("--record-dir",
                    default=str(_PICO_TAPES_DIR / "intent_sessions"))
    ap.add_argument("--session-dir",
                    default=str(_PICO_TAPES_DIR / "sessions"),
                    help="RAW body-tape clips (pico_tape format, world-frame "
                         "24x7 joints), one file per ENGAGED stretch, same "
                         "layout as live_pico_smpl_teleop.py's session clips. "
                         "The intent tape above is heading-normalized and is "
                         "NOT retargetable for walking/turning (2026-09-03); "
                         "these raw clips are what pico_tape_to_x2_gmr.py "
                         "consumes. --no-record disables both.")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    # BOUNDED QUEUES (2026-09-06 robot session on a jittery AP: "at least 5 second lag from when I
    # press locomotion to seeing it in the logs"). A PUB socket keeps up to SNDHWM messages (default
    # 1000 = 20 s at 50 Hz) plus the kernel's send buffer when the link stalls, then delivers them
    # IN ORDER — a stall becomes a permanent lag of stale frames instead of a gap. Keep only a few
    # frames in flight: a stall now costs a short freeze (the deploy holds after 0.5 s anyway), never
    # seconds of latency. Applied to every live-control PUB below.
    _LIVE_SNDHWM, _LIVE_SNDBUF = 4, 64 * 1024        # ~80 ms of intents / ~0.7 s of bytes at 94 KB/s
    def _bounded(s: "zmq.Socket", hwm: int = _LIVE_SNDHWM) -> "zmq.Socket":
        s.setsockopt(zmq.SNDHWM, hwm)
        s.setsockopt(zmq.SNDBUF, _LIVE_SNDBUF)
        s.setsockopt(zmq.LINGER, 0)
        return s
    sock = _bounded(ctx.socket(zmq.PUB))
    sock.connect(f"tcp://{args.pc2_host}:{args.intent_port}")
    print(f"intent PUB -> tcp://{args.pc2_host}:{args.intent_port} "
          f"topic 'pico_intent'", flush=True)

    # Tape kinds: an INTENT tape (this script's --record output: smpl_joints +
    # engaged flags), a per-engage RAW body clip (body + grips, controls
    # zero), or a FULL-SESSION tape (X long-press .. Y long-press: body +
    # every control at the device rate, operator 2026-09-05 "the replay has
    # to capture all the activity"). A full-session tape drives the SAME
    # decoder as a live headset (chord, sticks, B, grips), so it runs in
    # dual mode like a live session; the other two are stick-less.
    _tape_files = set(np.load(args.tape_replay).files) if args.tape_replay else set()
    full_tape = "full_session" in _tape_files
    dual_mode = not args.no_locomotion and (not args.tape_replay or full_tape)
    # --tape-once on a RAW body clip = a scripted B-in / B-out on a robot that
    # sits in LOCOMOTION: the planner_cmd and hand sockets the B path uses
    # stay open (tape mode otherwise runs stick-less, so dual_mode is off).
    raw_tape_once = bool(args.tape_replay and args.tape_once and not full_tape
                         and "smpl_joints" not in _tape_files)
    planner_sock = None
    if dual_mode or raw_tape_once:
        pport = args.planner_cmd_port
        if pport is None:
            pport = 5663 if args.pc2_host in ("127.0.0.1", "localhost") else 5563
        planner_sock = _bounded(ctx.socket(zmq.PUB), hwm=20)   # events, not a stream: a little slack
        planner_sock.connect(f"tcp://{args.pc2_host}:{pport}")
        print(f"planner_cmd PUB -> tcp://{args.pc2_host}:{pport} "
              f"(source:'vr')", flush=True)

    # keep empty entries: the list is POSITIONAL (F,B,L,R,FL,FR,BL,BR); an empty slot = direction unbound
    micro_list = [k.strip() for k in args.micro_keys.split(",")] if args.micro_keys.strip(", ") else []
    clip_sock = None
    if micro_list:
        if len(micro_list) != 8:
            raise SystemExit("--micro-keys needs exactly 8 keys: F,B,L,R,FL,FR,BL,BR")
        clip_sock = ctx.socket(zmq.PUB); clip_sock.setsockopt(zmq.SNDHWM, 10)
        clip_sock.connect(f"tcp://{args.pc2_host}:{args.clip_port}")
        print(f"[micro] 8-way MICRO-STEP bank armed on the LEFT stick (gate={args.micro_gate}, deadman released): "
              f"{micro_list} -> motion_clip_cmd tcp://{args.pc2_host}:{args.clip_port}", flush=True)
    micro_prev_dir, micro_cooldown_until = None, 0.0
    hand_sock = None
    if (dual_mode or raw_tape_once) and args.hand_source != "off":
        hand_sock = _bounded(ctx.socket(zmq.PUB))
        hand_sock.connect(f"tcp://{args.pc2_host}:{args.hand_port}")
        print(f"hand_finger_cmd PUB -> tcp://{args.pc2_host}:{args.hand_port} "
              f"(source: controller {args.hand_source}; OmniHand 10-DOF per side; "
              f"{'OPEN at rest, squeeze to close' if args.hand_close_on_grip else 'CLOSED at rest, squeeze to OPEN'})", flush=True)
    hand_tick = 0
    hand_next_t = 0.0
    hand_last = None

    LOCO_HEARTBEAT_S = 0.2          # == pad_locomotion_bridge.HEARTBEAT_S
    loco_last, loco_last_t = None, 0.0

    planner_tx = {"t": 0.0}

    def _send_planner(cmd: LocomotionCmd, release: bool = False) -> None:
        if planner_sock is not None:
            planner_sock.send_multipart(
                [b"planner_cmd", _planner_cmd_payload(cmd, release=release)])
            planner_tx["t"] = time.monotonic()

    # An intent tape (this script's own --record output) replays directly —
    # wall-clock paced raw frames, no XRT conversion, engaged flags replayed
    # verbatim so sim reproduces the exact engage segments of the robot
    # session (sim2real plan phase 1). Session npz tapes keep the old path.
    if args.tape_replay and "smpl_joints" in np.load(args.tape_replay):
        src = _IntentTapeSource(args.tape_replay)
    else:
        src = LiveSmplSource(args.tape_replay, loop=not args.tape_once)
        if args.tape_once:
            print(f"[tape] ONCE mode: {args.tape_replay} plays a single pass "
                  f"({src.xrt._span:.1f} s), then whole-body is released and the "
                  f"sender idles. Pad e-stop stays the safety (no controller chords on a tape).", flush=True)
    print("Waiting for body stream ...", flush=True)
    try:
        while src.window(time.monotonic()) is None:
            time.sleep(0.2)
    except KeyboardInterrupt:
        # Ctrl-C before the stream is live: same linger-free teardown as the main loop's finally,
        # otherwise PUB sockets with unreachable peers + SDK threads hold the interpreter open
        # ("can't terminate", 2026-09-08)
        print("\nintent sender stopped (no stream yet).", flush=True)
        src.stop()
        try:
            zmq.Context.instance().destroy(linger=0)
        except Exception:
            pass
        return 130
    print("stream live.", flush=True)

    engaged = bool(args.auto_engage)
    tape_done = False; tape_done_t = 0.0
    dbg_watch = None
    if not args.tape_replay and not args.no_record and args.x2_debug_port > 0:
        try:
            dbg_watch = _X2DebugWatch(args.pc2_host, args.x2_debug_port)
            print(f"[session] x2_debug watch on tcp://{args.pc2_host}:{args.x2_debug_port} (arms0 snapshot at record start)", flush=True)
        except Exception as e:
            print(f"[session] x2_debug watch unavailable ({e!r}) — records will carry no arms0", flush=True)
    if full_tape:
        # CANONICAL PRE-ROLL, step 1: planner through its idle gate (cold-start
        # ramp reset, any WB hold -> return), exactly what the record start did.
        _send_planner(LocomotionCmd("idle", "default"), release=True)
        src.xrt._t0 = time.monotonic() + 0.6                  # tape held at frame 0 meanwhile
        print("[tape] PRE-FLIGHT 1/2: planner -> idle/release (canonical cold start, 0.6 s) — the record is NOT playing yet", flush=True)
        time.sleep(0.6)
    if full_tape and hand_sock is not None:
        # ARMS0 (operator 2026-09-05): put the arms where they were when the
        # record started, BEFORE the tape rolls, via the kplanner's arm
        # overlay (same arm_targets wire the Quest manager uses; the overlay
        # slews at 3 rad/s and, with KPLANNER_WB_ARM_LATCH_GAIT=keep, holds
        # them through the gait exactly like the original session).
        meta = getattr(src.xrt, "meta", {})
        arms0 = meta.get("arms0")
        if arms0 is not None and np.asarray(arms0).shape == (14,) and args.arms0_settle_s > 0:
            arms0 = np.asarray(arms0, np.float32)
            # SMOOTH (operator: "has to be smooth though"): blend from the robot's
            # CURRENT measured arms (x2_debug) to arms0 on a half-cosine over
            # the settle time, streamed at 20 Hz; the overlay's 3 rad/s slew
            # never binds on that path. No fresh x2_debug -> the overlay slews
            # from wherever it is (still rate-limited, just less gentle).
            cur = None
            if args.x2_debug_port > 0:
                try:
                    w = _X2DebugWatch(args.pc2_host, args.x2_debug_port)
                    t_w = time.monotonic() + 0.8
                    while time.monotonic() < t_w and w.fresh_arms() is None:
                        time.sleep(0.05)
                    cur = w.fresh_arms()
                except Exception as e:
                    print(f"[tape] arms0: x2_debug unavailable ({e!r})", flush=True)
            settle = float(args.arms0_settle_s)
            src.xrt._t0 = time.monotonic() + settle      # hold the tape at frame 0 meanwhile
            print(f"[tape] PRE-FLIGHT 2/2: arms -> the record's start pose over {settle:.1f} s "
                  f"({'half-cosine blend from the measured arms' if cur is not None else 'overlay slew, no measured arms'}, "
                  f"max delta {np.degrees(np.abs(arms0 - (cur if cur is not None else arms0)).max()):.0f} deg) — the record starts after", flush=True)
            t_s = time.monotonic()
            while True:
                u = min(1.0, (time.monotonic() - t_s) / settle)
                w_hc = 0.5 - 0.5 * math.cos(math.pi * u)
                tgt = arms0 if cur is None else (1.0 - w_hc) * cur + w_hc * arms0
                hand_sock.send_multipart([b"arm_targets", msgpack.packb({
                    "left_q_rad": tgt[:7].tolist(), "right_q_rad": tgt[7:].tolist(), "ts": time.time()})])
                if u >= 1.0:
                    break
                time.sleep(0.05)
        elif arms0 is None:
            print("[tape] no arms0 in this record (recorded without the x2_debug watch) — arms stay where they are", flush=True)
    pad_abort = None
    if args.tape_replay and args.pad_abort_port > 0:
        chords = [tuple(c.strip().lower().split("+")) for c in args.pad_abort_chords.split(",") if c.strip()]
        pad_abort = _PadAbortWatch(args.pc2_host, args.pad_abort_port, chords)
        print(f"[tape] gamepad abort armed: {' or '.join('+'.join(k.upper() for k in c) for c in chords)} "
              f"on pad_state tcp://{args.pc2_host}:{args.pad_abort_port} -> release whole-body, planner idle, exit "
              "(no feed = no abort; Ctrl-C)", flush=True)
    REC_HOLD_S = 0.8
    b_hint_t = -1e9
    x_hold = y_hold = 0.0; rec_x_fired = rec_y_fired = False; rec_mode0 = "OFF"; rec_engaged0 = False; rec_t0_wall = 0.0; rec_arms0 = None
    if engaged and raw_tape_once:
        # mirror the B press into WHOLE_BODY: planner to idle + release
        # BEFORE the engaged flag rises (the deploy hands authority to the
        # token stream; at tape end the flag drops and the kplanner returns).
        _send_planner(LocomotionCmd("idle", "default"), release=True)
        _audio_cue("replay_started")
        print("[tape] planner -> idle/release (as on B); whole-body ENGAGES on the first frame", flush=True)
    body_ok_prev = True
    live_since = -1.0
    last_dead_cue = -1e9
    wrist_base = None
    wrist_yaw_base = None
    sm = ButtonStateMachine(log_prefix="Input")  # prints every rising edge:
    # "[Input] B pressed" etc. — keep ON until the Pico face-button naming
    # is field-verified (2026-09-02: B-in-LOCOMOTION did nothing on robot;
    # chord worked, so suspect SDK A/B/X/Y != physical labels)
    defer = _ButtonDefer()
    deadman_prev = False
    _debounce = _ButtonDebounce(ticks=3)
    decoder = IntentDecoder(
        stick_deadzone=args.stick_deadzone,
        enable_continuous_locomotion=not args.bucketed_locomotion,
    )
    estop = EstopGesture()
    _MODE_NAME = {StreamMode.OFF: "OFF",
                  StreamMode.LOCOMOTION: "LOCOMOTION",
                  StreamMode.ARM_MANIPULATION: "WHOLE_BODY"}
    if full_tape:
        # Replay starts in the mode the recording started in, so the tape's
        # own chords/B presses reproduce the same transitions.
        meta = getattr(src.xrt, "meta", {})
        mode0 = str(meta.get("mode0", "OFF"))
        decoder._mode = {"OFF": StreamMode.OFF, "LOCOMOTION": StreamMode.LOCOMOTION,
                         "WHOLE_BODY": StreamMode.ARM_MANIPULATION}.get(mode0, StreamMode.OFF)
        if args.auto_engage:
            print("[tape] FULL SESSION tape: --auto-engage ignored (the tape's own B engages)", flush=True)
        # a record started while whole-body was ENGAGED replays engaged from
        # frame 0 (else its first RT+B would disengage instead of engaging)
        engaged0 = bool(meta.get("engaged0", mode0 == "WHOLE_BODY"))
        engaged = engaged0
        if engaged0:
            _send_planner(LocomotionCmd("idle", "default"), release=True)
            print("[tape] record started ENGAGED -> replay engages on the first frame (planner idle/release as on B)", flush=True)
        _audio_cue("replay_started")
        print("[tape] REPLAY START — pre-flight done, the record's inputs play from here", flush=True)
        print(f"[tape] FULL SESSION tape: {src.xrt._span:.1f} s, starts in {mode0}; chord / sticks / "
              f"B toggles / grips replay through the live decoder"
              + (" (once: controls go neutral at the end, planner idle/release)" if args.tape_once else " (LOOPING)"),
              flush=True)
    frame_idx, n_sent, t_rep = 0, 0, time.monotonic()
    recording = not args.no_record and not args.tape_replay
    rec = {"t": [], "smpl_joints": [], "human_quat": [],
           "engaged": [], "wrist_pr": []} if recording else None
    wrist_source = "off" if args.no_wrist_follow else args.wrist_source
    wrist_dbg = None
    print(f"[wrist] conversion={'FRAME (SMPL->X2 chain yaw/pitch/roll)' if _USE_FRAME_MAP else 'per-axis WRIST_SMPL_MAP'}"
          f"  yaw sign L/R={_YAW_SIGN if not _USE_FRAME_MAP else (1.0, 1.0)}", flush=True)
    print(f"[wrist] source={wrist_source}"
          + (f" map={_WRIST_MAP}" if wrist_source == "smpl" else ""), flush=True)
    head_source = args.head_source
    if head_source == "smpl" and planner_sock is None:
        head_source = "off"
        print("[head] no planner_cmd socket in this mode -> head yaw OFF", flush=True)
    head_tx = {"t": 0.0, "active": False, "base": None}
    head_dbg = None
    print(f"[head] source={head_source}"
          + (f" sign={_HEAD_YAW_SIGN:+.0f} gain={_HEAD_YAW_GAIN:.2f} "
             f"max={math.degrees(_HEAD_YAW_MAX):.0f}deg "
             f"deadband={math.degrees(_HEAD_YAW_DEADBAND):.1f}deg "
             f"neutral={_HEAD_YAW_NEUTRAL} "
             f"-> planner_cmd head_targets @25Hz" if head_source == "smpl" else ""),
          flush=True)
    if recording:
        print(f"[intent] RECORDING intent tape -> {args.record_dir}", flush=True)
    # Raw body-tape clips (world-frame 24x7), gated on the SAME engaged
    # flag the intent tape carries: B press = clip start, B press = clip
    # end, flushed to disk at each disengage (crash-safe) — mirrors
    # live_pico_smpl_teleop.py. Only the live source records.
    raw_rec = recording and isinstance(src, LiveSmplSource)
    raw_seg = {"n": 0, "t0": None}
    raw_prev = False

    def _raw_flush() -> None:
        raw_seg["n"] += 1
        sdir = Path(args.session_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        p = sdir / f"session_{raw_seg['t0']}_x2_seg{raw_seg['n']:02d}.npz"
        cnt = src.save_session(p, clear=True)
        if cnt:
            print(f"[clip] auto per-engage clip (corpus pipeline; NOT an X/Y session record): "
                  f"saved {cnt} raw frames -> {p}", flush=True)
    if raw_rec:
        print(f"[clip] RECORDING raw body clips per engage -> "
              f"{args.session_dir}", flush=True)
    if engaged:
        print("ENGAGED (auto).", flush=True)
    elif dual_mode:
        print("mode OFF — A+B+X+Y chord to enter LOCOMOTION; B toggles "
              "LOCOMOTION <-> WHOLE_BODY (whole-body ENGAGES on entry, "
              "disengages on exit — no separate button); LT = walk deadman; "
              "A+X + rapid trigger pumps = E-STOP.", flush=True)
    else:
        print("DISENGAGED — controller layer off (--no-locomotion/tape); "
              "--auto-engage or tape flags control engage.", flush=True)
    def _publish_hands(now: float, lt: float, rt: float) -> None:
        """OmniHand fingers: grips -> grasp ratio -> 10-DOF motor targets on
        the hand_finger_cmd wire, at --hand-hz, whether or not whole-body is
        engaged (hands are usable in kplanner mode too; the kplanner latches
        the last targets on a link drop). Called from the dual-mode
        controller layer AND from the stick-less tape replay (2026-09-05:
        window-tape replays published no hands -- "fingers not opening in
        sim" -- because this lived inside the dual-mode block)."""
        nonlocal hand_next_t, hand_tick, hand_last
        if hand_sock is None or now < hand_next_t:
            return
        hand_next_t = now + 1.0 / max(args.hand_hz, 1.0)
        try:
            lg = float(src.xrt.get_left_grip()); rg = float(src.xrt.get_right_grip())
        except Exception:
            lg = rg = 0.0
        l_ratio, r_ratio = controller_grasp_ratio(
            left_trigger=lt, right_trigger=rt, left_grip=lg, right_grip=rg,
            mode=args.hand_source)
        if not args.hand_close_on_grip:
            # closed at rest, squeeze to open (ratio 1 = closed)
            l_ratio, r_ratio = 1.0 - l_ratio, 1.0 - r_ratio
        lhq = grasp_command_from_ratio("left", l_ratio)
        rhq = grasp_command_from_ratio("right", r_ratio)
        hand_tick += 1
        hand_sock.send_multipart([b"hand_finger_cmd", msgpack.packb({
            "left_hand_q": lhq.astype(np.float32).tolist(),
            "right_hand_q": rhq.astype(np.float32).tolist(),
            "tick": int(hand_tick), "ts": time.time()})])
        if hand_last is None or abs(l_ratio - hand_last[0]) > 0.25 or abs(r_ratio - hand_last[1]) > 0.25:
            print(f"[hand] grasp L {l_ratio:.2f} R {r_ratio:.2f}", flush=True)
            hand_last = (l_ratio, r_ratio)

    next_t = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(next_t - now, 0.005))
                continue
            next_t += CONTROL_DT
            if not dual_mode:
                # stick-less tape replay (single engage-window clip): the
                # tape's grips still drive the hands (triggers are zero)
                _publish_hands(now, 0.0, 0.0)

            # ---- dual-mode controller layer (live sessions only) ----
            # Quest control grammar: A+B+X+Y = OFF<->LOCOMOTION, B =
            # LOCOMOTION<->WHOLE_BODY, A in WHOLE_BODY = engage toggle,
            # A+X + rapid trigger pumps = e-stop. The old dual-grip
            # engage is gone (accidental engages); grips are reserved
            # for finger control.
            if dual_mode:
                cs = _poll_controllers(src.xrt)
                if cs is None:
                    lx = ly = rx = ry = lt = rt = 0.0
                    a = b = x = y = False
                else:
                    lx, ly, rx, ry, a, b, x, y, lt, rt = cs
                # ---- FULL-SESSION recording (operator 2026-09-05, same
                # grammar as the Quest manager's X=start / Y=stop): X held
                # >= REC_HOLD_S starts, Y held >= REC_HOLD_S stops and writes
                # session_<stamp>_full.npz (every control + body, device
                # rate). Long-press so the single-press speed trim keeps
                # working (the one trim step at press time cancels over a
                # start/stop pair). Live sessions only.
                if raw_rec and (a or b):
                    # a chord (A+B+X+Y master, A+X e-stop) is in progress: X/Y belong
                    # to it, not to the recorder (operator 2026-09-05: "a-b-x-y now
                    # thinks i am trying to record"); drop the hold timers.
                    x_hold = y_hold = 0.0
                    rec_x_fired = rec_y_fired = True     # re-armed on release below
                if raw_rec:
                    x_prev_hold, y_prev_hold = (0.0, 0.0) if (a or b) else (x_hold, y_hold)
                    x_hold = 0.0 if (a or b) else (x_hold if (x and x_hold) else (now if x else 0.0))
                    y_hold = 0.0 if (a or b) else (y_hold if (y and y_hold) else (now if y else 0.0))
                    if x_hold and now - x_hold >= REC_HOLD_S and not src.full_on and not rec_x_fired:
                        rec_x_fired = True
                        rec_mode0 = _MODE_NAME[decoder.mode]
                        rec_engaged0 = bool(engaged)
                        rec_t0_wall = time.time()
                        rec_arms0 = dbg_watch.fresh_arms() if dbg_watch is not None else None
                        if rec_arms0 is None:
                            print(f"[session] arms0: no fresh x2_debug frame (newest is {dbg_watch.frame_age_s() if dbg_watch else float('inf'):.1f} s old) "
                                  "— this record will not restore the arm pose on replay", flush=True)
                        else:
                            print(f"[session] arms0 snapshot taken (x2_debug frame {dbg_watch.frame_age_s():.2f} s old)", flush=True)
                        # CANONICAL START (operator 2026-09-05): a take and its replay must
                        # begin from the same planner state. The planner's cold-start
                        # velocity ramp is warm or cold depending on whether it crossed
                        # its idle gate since the last gait -- unobservable from here --
                        # so put it through the idle gate NOW (cold) and the replay does
                        # the same in its pre-roll. Hold X while standing, not mid-walk.
                        _send_planner(LocomotionCmd("idle", "default"), release=True)
                        loco_last = None
                        print("[session] planner -> idle/release at record start (canonical cold start; the replay pre-roll does the same)", flush=True)
                        src.start_full()
                        _audio_cue("record_started")
                        print(f"[session] ● RECORDING full session (mode {rec_mode0}) — Y long-press stops", flush=True)
                    # wrong-state presses are logged explicitly (operator 2026-09-05:
                    # "check if i forgot to start recording and tried stopping
                    # something that hasn't started? we should log that")
                    if x_hold and now - x_hold >= REC_HOLD_S and src.full_on and not rec_x_fired:
                        rec_x_fired = True
                        _audio_cue("already_recording")
                        print(f"[session] X held but a recording is ALREADY running ({src.full_frames()} samples) — hold Y to stop it", flush=True)
                    if y_hold and now - y_hold >= REC_HOLD_S and not src.full_on and not rec_y_fired:
                        rec_y_fired = True
                        _audio_cue("not_recording")
                        print("[session] Y held but NO recording is in progress — nothing stopped; hold X to start one", flush=True)
                    if not x:
                        # released early: say so, else a tap looks like a dead button
                        if x_prev_hold and not rec_x_fired and not src.full_on and now - x_prev_hold < REC_HOLD_S:
                            print(f"[session] X released after {now - x_prev_hold:.1f} s — hold X {REC_HOLD_S:.1f} s to START recording", flush=True)
                        rec_x_fired = False
                    if y_hold and now - y_hold >= REC_HOLD_S and src.full_on and not rec_y_fired:
                        rec_y_fired = True
                        import datetime as _dt
                        _stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
                        # X/Y session RECORDINGS live apart from the per-engage auto
                        # clips (operator 2026-09-05: "the normal recording should be
                        # separate from explicit x/y recording sessions"):
                        # $PICO_TAPES_DIR/records/record_<stamp>.npz
                        _rdir = Path(args.session_dir).parent / "records"
                        _out = _rdir / f"record_{_stamp}.npz"
                        _rdir.mkdir(parents=True, exist_ok=True)
                        _meta = dict(mode0=rec_mode0, engaged0=rec_engaged0, t0_wall_unix_s=rec_t0_wall,
                                     hand_close_on_grip=bool(args.hand_close_on_grip))
                        if rec_arms0 is not None:
                            _meta["arms0"] = np.asarray(rec_arms0, np.float32)
                        _n = src.stop_full(_out, **_meta)
                        _audio_cue("record_stopped")
                        print(f"[session] ■ STOPPED — {_n} samples -> {_out}", flush=True)
                    if not y:
                        if y_prev_hold and not rec_y_fired and src.full_on and now - y_prev_hold < REC_HOLD_S:
                            print(f"[session] Y released after {now - y_prev_hold:.1f} s — hold Y {REC_HOLD_S:.1f} s to STOP recording", flush=True)
                        rec_y_fired = False
                # ---- OmniHand fingers (2026-09-04): grips -> grasp ratio ->
                # 10-DOF motor targets, the Quest manager's hand_finger_cmd
                # wire. Published whether or not whole-body is engaged (the
                # hands are usable in kplanner mode too); the kplanner
                # latches the last targets on a link drop.
                _publish_hands(now, lt, rt)
                if args.invert_lx:
                    lx = -lx
                if args.invert_ly:
                    ly = -ly
                if args.invert_rx:
                    rx = -rx
                if args.invert_ry:
                    ry = -ry
                # LOCOMOTION deadman (robot session 2026-09-02, pad parity:
                # the ritual runs the pad bridge with --deadman left = L2
                # alone). LEFT TRIGGER held = sticks live; released = sticks
                # forced neutral, so the decoder emits the zero/idle command
                # exactly like the pad bridge's all-zero release frame. Only
                # gates the sticks — buttons/chords/e-stop stay live.
                loco_live = lt >= 0.6
                # MICRO-STEP bank: left stick sector while the deadman is RELEASED
                # (deadman held = the stick is kplanner locomotion), right stick pressed
                # unless --micro-gate none; whole-body engaged -> the sticks belong to the
                # SMPL session, never fire.
                if clip_sock is not None and not loco_live and not engaged:
                    if args.micro_gate == "right-trigger":
                        gate_ok = rt >= 0.6
                    elif args.micro_gate == "right-click":
                        try:
                            gate_ok = bool(xrt.get_right_axis_click())
                        except Exception:
                            gate_ok = False
                    else:
                        gate_ok = True
                    mdir = None
                    if gate_ok and max(abs(lx), abs(ly)) >= args.micro_deflect:
                        ang = (math.degrees(math.atan2(lx, ly)) + 360.0) % 360.0     # 0 = forward, 90 = right
                        mdir = ("F", "FR", "R", "BR", "B", "BL", "L", "FL")[int(((ang + 22.5) % 360.0) // 45.0)]
                    if mdir is None:
                        micro_prev_dir = None
                    elif mdir != micro_prev_dir:
                        micro_prev_dir = mdir
                        if now >= micro_cooldown_until:
                            key = micro_list[("F", "B", "L", "R", "FL", "FR", "BL", "BR").index(mdir)]
                            if not key:
                                print(f"[micro] {mdir}: no clip bound", flush=True)
                            elif key.startswith("prim:") and planner_sock is not None:
                                # kplanner PRIMITIVE step (root XY persists) instead of the
                                # x2m2 clip player, which pins the pelvis (2026-09-06).
                                planner_sock.send_multipart([b"planner_cmd", json.dumps(
                                    {"intent": "primitive", "name": key[5:], "magnitude": "default",
                                     "source": "vr"}).encode("utf-8")])
                            else:
                                clip_sock.send_multipart([b"motion_clip_cmd", json.dumps(
                                    {"action": "play", "pkl": "micro", "kind": "locomotion", "motion_key": key}).encode("utf-8")])
                            micro_cooldown_until = now + args.micro_cooldown_s
                            print(f"[micro] {mdir} -> {key}", flush=True)
                if loco_live != deadman_prev:
                    if decoder.mode == StreamMode.LOCOMOTION:
                        print("DEADMAN ENGAGED (L-trigger) — sticks live"
                              if loco_live else
                              "deadman released — sticks neutral", flush=True)
                    deadman_prev = loco_live
                if not loco_live:
                    lx = ly = rx = ry = 0.0
                # RAW-BUTTON DEBOUNCE (robot 2026-09-02): the Pico SDK
                # reports phantom multi-presses — a single A (engage) drags
                # in B/X/Y for a tick or two, which formed accidental
                # A+B+X+Y chords that kicked the session OUT of WHOLE_BODY
                # before engage could take (token svc: tokens 0Hz,
                # disengaged, despite the laptop printing ENGAGED). Require
                # each face button to hold a NEW state for DEBOUNCE_TICKS
                # consecutive polls before it counts. Real presses hold far
                # longer than the 1-2 tick phantoms; latency added is
                # DEBOUNCE_TICKS*20ms ~= 40ms, imperceptible.
                a, b, x, y = _debounce.update((a, b, x, y), now)
                ev = defer.tick(sm.tick(a, b, x, y), a and x, now)
                # DEADMAN FOR THE MODE SWITCH (operator 2026-09-05, after the
                # 19:01:50Z fall: an inadvertent B released whole-body while
                # SONIC was stepping on uneven ground -> frozen legs -> fall):
                # B toggles LOCOMOTION <-> WHOLE_BODY only with the RIGHT
                # TRIGGER held. A bare B is ignored with a hint + cue. The
                # A+B+X+Y master chord and the A+X e-stop are separate events
                # and unaffected.
                if (ev.b_pressed and args.wb_toggle_gate == "right-trigger"
                        and rt < 0.6):
                    ev = dataclasses.replace(ev, b_pressed=False)
                    if now - b_hint_t > 1.5:
                        b_hint_t = now
                        _audio_cue("b_needs_trigger")
                        print("B ignored — hold the RIGHT TRIGGER and press B to switch "
                              "LOCOMOTION <-> WHOLE_BODY (deadman for the mode switch)", flush=True)
                tr = decoder.update_mode(ev, now)
                if tr is not None:
                    print(f"MODE {_MODE_NAME[tr.previous]} -> "
                          f"{_MODE_NAME[tr.current]}", flush=True)
                    # B directly toggles whole-body engage (robot 2026-09-02,
                    # operator: one button, no separate A step). Entering
                    # WHOLE_BODY ENGAGES immediately (freshness-gated);
                    # leaving it DISENGAGES.
                    if tr.current == StreamMode.ARM_MANIPULATION:
                        if not src.body_fresh(now):
                            _audio_cue("disengage")
                            age = now - src._last_append_t
                            print(f"*** ENGAGE REFUSED: body stream dead/"
                                  f"frozen (last new frame {age:.1f}s ago, "
                                  f"identical-frame streak {src._same_streak})"
                                  f" -> RECALIBRATE in headset. ***",
                                  flush=True)
                        else:
                            engaged, wrist_base = True, None
                            _send_planner(LocomotionCmd("idle", "default"),
                                          release=True)
                            loco_last = None
                            _audio_cue("mode_whole_body")
                            print("WHOLE-BODY ENGAGED — token stream has "
                                  "authority", flush=True)
                    else:
                        # left WHOLE_BODY (to LOCOMOTION or OFF)
                        if engaged:
                            engaged, wrist_base, wrist_yaw_base = False, None, None
                            print("WHOLE-BODY DISENGAGED (kplanner has "
                                  "authority)", flush=True)
                        # spoken mode cue (Quest grammar): "Locomotion" / "Off"
                        _audio_cue("mode_locomotion" if tr.current == StreamMode.LOCOMOTION else "mode_off")
                        if tr.current == StreamMode.OFF:
                            _send_planner(LocomotionCmd("idle", "default"),
                                          release=True)
                            loco_last = None

                # E-STOP gesture (same shape as pad + Quest surfaces):
                # A+X held + >=3 paired trigger pumps/1s -> soft; keep
                # pumping ~1s more -> pure damping. Also force-drops the
                # whole-body engage so the token overlay releases and
                # the planner owns the stop.
                ph = estop.tick(lt, rt, a and x, now)
                if ph:
                    engaged = False
                    _send_planner(LocomotionCmd(
                        "estop", "soft" if ph == 1 else "damp"))
                    _audio_cue("estop_activating" if ph == 1 else "estop_damping")
                    if ph == 1:
                        print("!!! E-STOP SOFT — keep pumping ~1s more "
                              "for PURE DAMPING !!!", flush=True)
                    else:
                        print("!!! E-STOP ESCALATED: PURE DAMPING "
                              "(terminal) !!!", flush=True)

                # LOCOMOTION mode: sticks -> planner_cmd, identical
                # vocabulary to the Quest stack (decoder owns deadzone,
                # chord-quiet, emit-on-change).
                if decoder.mode == StreamMode.LOCOMOTION:
                    cmd = decoder.decode_locomotion(
                        lx, ly, rx, ry, y_held=y, now=now,
                        a_held=a, x_held=x)
                    if cmd is not None:
                        _send_planner(cmd)
                        loco_last, loco_last_t = cmd, now
                    elif (loco_last is not None
                          and getattr(loco_last, "intent", "idle") != "idle"
                          and now - loco_last_t >= LOCO_HEARTBEAT_S):
                        # HEARTBEAT (2026-09-04): emit-on-change alone let the
                        # kplanner's 2 s VR-ownership silence timeout fire
                        # with the stick HELD -> "VR silent 2.0s -> releasing
                        # to idle" -> the robot stopped every 2 s ("Pico
                        # forward motion sluggish vs the gamepad"). The pad
                        # bridge re-publishes every 0.2 s while driving; do
                        # the same.
                        _send_planner(loco_last)
                        loco_last_t = now

            win = src.window(now)
            if win is None:
                continue
            joints, quats = win

            # BODY-STREAM GUARD (2026-09-01): live mode only. If the Pico
            # body solve freezes (cached pose, dead trackers) the ring goes
            # stale/identical while grips + wrists stay live -- and the
            # robot would track a frozen mannequin (26deg-pitched ghost;
            # every sim engage fell). Force engaged=0 until the stream is
            # genuinely alive again; the token service then auto-releases
            # and the robot holds pad authority instead of falling.
            body_ok = True
            if not isinstance(src, _IntentTapeSource) and not args.tape_replay:
                raw_ok = src.body_fresh(now)
                # HYSTERESIS (2026-09-01): going DEAD is immediate, coming
                # back LIVE needs 1.0 s of sustained health -- Wi-Fi flaps
                # around the threshold were kicking the operator out and
                # looping the disengage audio mid-session.
                if raw_ok:
                    if live_since < 0.0:
                        live_since = now
                    body_ok = body_ok_prev or (now - live_since) >= 1.0
                else:
                    live_since = -1.0
                    body_ok = False
                if body_ok != body_ok_prev:
                    if not body_ok:
                        if now - last_dead_cue > 5.0:   # cue at most 1/5s
                            _audio_cue("disengage")
                            last_dead_cue = now
                        print("*** BODY STREAM DEAD/FROZEN — intent forced "
                              "DISENGAGED. Check Pico body tracking "
                              "(trackers awake? calibration done? PC "
                              "Service in body mode?) ***", flush=True)
                    else:
                        print("body stream LIVE again — press A (in "
                              "WHOLE_BODY mode) to re-engage.", flush=True)
                        # do NOT auto-re-engage: operator confirms.
                        engaged = False
                    body_ok_prev = body_ok

            # --tape-once: end of the tape = operator's B. Release once, keep
            # streaming (the deploy needs a continuous stream; a dead sender
            # is a stream-lost hold, not a clean return).
            if (args.tape_once and not tape_done
                    and (src.finished(now) if isinstance(src, _IntentTapeSource) else src.xrt.finished())):
                tape_done = True
                if engaged:
                    engaged = False
                _audio_cue("replay_done")
                # controls are neutral from here (TapeXrt.finished); hand the
                # planner an explicit idle so nothing recorded stays "held"
                _send_planner(LocomotionCmd("idle", "default"), release=True)
                tape_done_t = now
                _span = float(src.rel_t[-1]) if isinstance(src, _IntentTapeSource) else src.xrt._span
                print(f"[tape] DONE ({_span:.1f} s) — whole-body RELEASED, planner idle; "
                      + (f"streaming idle for {args.tape_exit_s:.0f} s, then exiting." if args.tape_exit_s >= 0
                         else "streaming idle. Ctrl-C to quit."), flush=True)
            # REPLAY OWNERSHIP HEARTBEAT (operator 2026-09-05, "the change should be
            # in the replay path"): the kplanner drops the VR arm overlay (arms0)
            # when VR ownership times out after 2 s of planner_cmd silence; a
            # plain idle keeps ownership without moving anything, so while the
            # tape is not engaged send one every second. The final DONE / pad
            # abort still sends idle+RELEASE, which hands the pad back.
            if (args.tape_replay and planner_sock is not None and not engaged and not tape_done
                    and now - planner_tx["t"] > 1.0):
                _send_planner(LocomotionCmd("idle", "default"))
            if pad_abort is not None and pad_abort.fired.is_set() and not tape_done:
                # gamepad chord: stop the tape NOW -- controls neutral from here
                # (the source keeps serving the last body frame), whole-body
                # released, planner idle/release so the pad owns locomotion.
                tape_done = True; tape_done_t = now
                if hasattr(src, "xrt") and hasattr(src.xrt, "_t0"):
                    src.xrt._t0 = -1e9        # TapeXrt.finished() -> True: neutral controls
                if engaged:
                    engaged = False
                _send_planner(LocomotionCmd("idle", "default"), release=True)
                _audio_cue("mode_locomotion")
                print(f"[tape] ABORTED by gamepad {pad_abort.chord} — whole-body RELEASED, planner idle; "
                      f"pad drives. Exiting in {max(args.tape_exit_s, 0):.0f} s.", flush=True)
            if tape_done and args.tape_exit_s >= 0 and now - tape_done_t >= args.tape_exit_s:
                print("[tape] exit (replay complete)", flush=True)
                break

            wr = np.zeros(4, np.float32)
            wy = np.zeros(2, np.float32)
            if isinstance(src, _IntentTapeSource):
                # replay the recorded session verbatim: engage segments and
                # wrist targets come from the tape, not grips/--auto-engage
                was = engaged
                engaged = src.replay_engaged(now)
                wr = src.replay_wrist(now).astype(np.float32)
                if engaged != was:
                    print("ENGAGED (tape)" if engaged else "DISENGAGED (tape)",
                          flush=True)
                    if engaged and args.tape_settle_s > 0:
                        src.hold(now, args.tape_settle_s)
                        print(f"[tape] settle: holding the engage frame for {args.tape_settle_s:.1f} s", flush=True)
            elif (engaged and wrist_source == "smpl"
                  and not isinstance(src, _IntentTapeSource)):
                # live headset OR a raw session-clip replay (LiveSmplSource
                # over TapeXrt): the wrist rotvecs are recomputed from the
                # recorded body poses, so a replay exercises the CURRENT
                # WRIST_SMPL_MAP (operator review 2026-09-04 16:10)
                # SMPL wrists (2026-09-04): the operator's own wrist rotation
                # from body tracking. Local rotvec of SMPL joints 20/21 in
                # the forearm frame; the forearm bone is the frame's X axis
                # (measured on the 08-29 raw tapes: +-0.98), so X = pronation
                # (X2 wrist_yaw, left to SONIC) and Y/Z = flexion/deviation.
                # WRIST_SMPL_MAP assigns those to X2 pitch/roll per side.
                # X2 geometry (x2_ultra.xml, arms hanging, palms inward):
                # wrist_pitch axis = world lateral -> the hand swings
                # fore/aft = anatomical DEVIATION (SMPL y); wrist_roll axis
                # = world forward -> the hand swings sideways = FLEXION
                # (SMPL z, the big asymmetric range). The first default
                # had them swapped (operator 2026-09-04 16:00: "only roll
                # responds"). Default now lp:+y lr:+z rp:-y rr:+z (right
                # side: SMPL y maps to -world-lateral).
                # Neutral = the wrist pose at engage (like the controller path).
                aa = src.latest_wrist_aa() if hasattr(src, "latest_wrist_aa") else None
                if aa is not None:
                    if _USE_FRAME_MAP:
                        ypr = _wrist_frame_ypr(aa)
                        raw = np.array([ypr[0, 1], ypr[0, 2], ypr[1, 1], ypr[1, 2]], np.float32)
                        raw_yaw = (float(ypr[0, 0]), float(ypr[1, 0]))
                    else:
                        raw = np.array([_wrist_map(aa, k) for k in ("lp", "lr", "rp", "rr")],
                                       np.float32)
                        raw_yaw = (-float(aa[0, 0]), float(aa[1, 0]))
                    if wrist_base is None:
                        wrist_base = tuple(float(x) for x in raw)
                    wr = np.array([
                        np.clip(raw[0] - wrist_base[0], *WRIST_RANGE["p"]),
                        np.clip(raw[1] - wrist_base[1], *WRIST_RANGE["r"]),
                        np.clip(raw[2] - wrist_base[2], *WRIST_RANGE["p"]),
                        np.clip(raw[3] - wrist_base[3], *WRIST_RANGE["r_right"]),
                    ], np.float32)
                    # PRONATION -> X2 wrist_yaw (axis along the forearm):
                    # SMPL x rotvec; left yaw axis is the negative bone
                    # direction, right the positive (composed-model
                    # geometry 2026-09-04). Operator: "wrist roll not
                    # working" == pronation, which nothing drove before.
                    if wrist_yaw_base is None:
                        wrist_yaw_base = raw_yaw
                    # frame map: the chain yaw already carries the per-side
                    # sense; _YAW_SIGN stays a live A/B knob (default +1,+1
                    # under the frame map, -1,+1 under WRIST_SMPL_MAP)
                    ys = (1.0, 1.0) if _USE_FRAME_MAP and "WRIST_YAW_SIGN" not in __import__("os").environ else _YAW_SIGN
                    wy = np.array([
                        np.clip(ys[0] * (raw_yaw[0] - wrist_yaw_base[0]), -2.5, 2.5),
                        np.clip(ys[1] * (raw_yaw[1] - wrist_yaw_base[1]), -2.5, 2.5),
                    ], np.float32)
                    wrist_dbg = (aa, raw, wr, wy)
            elif (engaged and wrist_source == "controller" and not args.tape_replay):
                try:
                    l = np.asarray(src.xrt.get_left_controller_pose(), np.float32)
                    r = np.asarray(src.xrt.get_right_controller_pose(), np.float32)
                    (lp, lr), (rp, rr) = _pr(l[3:7]), _pr(r[3:7])
                    if wrist_base is None:
                        wrist_base = (lp, lr, rp, rr)
                    wr = np.array([
                        np.clip(lp - wrist_base[0], *WRIST_RANGE["p"]),
                        np.clip(lr - wrist_base[1], *WRIST_RANGE["r"]),
                        np.clip(rp - wrist_base[2], *WRIST_RANGE["p"]),
                        np.clip(-(rr - wrist_base[3]), *WRIST_RANGE["r"]),
                    ], np.float32)
                except Exception:
                    pass

            # ---- Head yaw (operator's SMPL neck+head -> X2 head_yaw via
            # the kplanner head-look overlay). Engaged only; one release
            # message on disengage recentres the head (slewed, 2 rad/s).
            if (head_source == "smpl" and engaged
                    and hasattr(src, "latest_head_aa")):
                haa = src.latest_head_aa()
                if haa is not None:
                    raw_h = _smpl_head_yaw(haa)
                    if _HEAD_YAW_NEUTRAL == "engage":
                        if head_tx["base"] is None:
                            head_tx["base"] = raw_h
                        raw_h -= head_tx["base"]
                    y_cmd = _head_yaw_cmd(raw_h)
                    head_dbg = (raw_h, y_cmd)
                    if now - head_tx["t"] >= HEAD_TX_PERIOD_S:
                        planner_sock.send_multipart([b"planner_cmd", json.dumps({
                            "intent": "head_targets", "source": "vr",
                            "yaw_rad": round(y_cmd, 4)}).encode("utf-8")])
                        head_tx["t"] = now
                        head_tx["active"] = True
            elif head_tx["active"] and not engaged:
                planner_sock.send_multipart([b"planner_cmd", json.dumps({
                    "intent": "head_targets", "source": "vr",
                    "release": 1}).encode("utf-8")])
                head_tx["active"] = False
                head_tx["base"] = None
                head_dbg = None

            msg = pack_pose_message({
                "smpl_joints": joints[-1].reshape(72).astype(np.float32),
                "human_quat": quats[-1].astype(np.float32),
                "engaged": np.array([1.0 if (engaged and body_ok) else 0.0],
                                    np.float32),
                "wrist_pr": wr,
                "wrist_yaw": wy,
                "frame_index": np.array([frame_idx], np.int64),
            }, topic="pico_intent", version=4)
            sock.send(msg)
            if rec is not None:
                rec["t"].append(now)
                rec["smpl_joints"].append(joints[-1].reshape(72).copy())
                rec["human_quat"].append(quats[-1].copy())
                rec["engaged"].append(1.0 if (engaged and body_ok) else 0.0)
                rec["wrist_pr"].append(wr.copy())
            if raw_rec:
                raw_now = bool(engaged and body_ok)
                if raw_now != raw_prev:
                    if raw_now:
                        raw_seg["t0"] = time.strftime("%Y%m%d_%H%M%SZ",
                                                      time.gmtime())
                        src.rec_gate = True
                    else:
                        src.rec_gate = False
                        _raw_flush()
                    raw_prev = raw_now
            frame_idx += 1
            n_sent += 1
            if now - t_rep >= args.rate_report_s:
                if isinstance(src, _IntentTapeSource) or args.tape_replay:
                    body_rep = "tape"
                else:
                    # Verdict up front (2026-09-02: a 26k-frame frozen
                    # streak hid in the raw numbers — the one-shot ***
                    # transition warning had long scrolled away).
                    verdict = ("body OK" if src.body_fresh(now) else
                               "body FROZEN/DEAD — engage will refuse; "
                               "recalibrate in headset")
                    body_rep = (f"{verdict} (last new frame "
                                f"{now - src._last_append_t:.1f}s ago, "
                                f"identical-streak {src._same_streak})")
                print(f"[intent] {n_sent / (now - t_rep):.1f} Hz  "
                      f"{'ENGAGED' if engaged else 'disengaged'}  "
                      f"[{body_rep}]", flush=True)
                if wrist_dbg is not None:
                    # calibration aid: raw local wrist rotvecs (deg, forearm
                    # frame x/y/z) and the mapped pitch/roll deltas on the wire
                    aa_d, raw_d, wr_d, wy_d = wrist_dbg
                    print(f"[wrist] L aa(deg) {np.degrees(aa_d[0]).round(0)} "
                          f"R aa(deg) {np.degrees(aa_d[1]).round(0)} -> wire "
                          f"lp/lr/rp/rr(deg) {np.degrees(wr_d).round(0)} "
                          f"yaw L/R(deg) {np.degrees(wy_d).round(0)}",
                          flush=True)
                if head_dbg is not None:
                    print(f"[head] smpl yaw {math.degrees(head_dbg[0]):+5.1f} deg "
                          f"-> wire {math.degrees(head_dbg[1]):+5.1f} deg "
                          f"(+ = left)", flush=True)
                n_sent, t_rep = 0, now
    except KeyboardInterrupt:
        print("\nintent sender stopped.", flush=True)
    finally:
        src.stop()
        if raw_rec and raw_prev:
            # session ended while still engaged: flush the final clip
            src.rec_gate = False
            _raw_flush()
        if rec is not None and rec["t"]:
            import datetime
            Path(args.record_dir).mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
            out = Path(args.record_dir) / f"intent_{stamp}.npz"
            np.savez_compressed(
                out,
                t=np.asarray(rec["t"], np.float64),
                smpl_joints=np.asarray(rec["smpl_joints"], np.float32),
                human_quat=np.asarray(rec["human_quat"], np.float32),
                engaged=np.asarray(rec["engaged"], np.float32),
                wrist_pr=np.asarray(rec["wrist_pr"], np.float32),
                quat_convention="wxyz (canonical, matches sidecar "
                                "global_orient_quat_wxyz)",
                fps=50.0)
            n_eng = int((np.asarray(rec["engaged"]) > 0.5).sum())
            print(f"[intent] tape saved: {out}  "
                  f"({len(rec['t'])} frames, {n_eng} engaged)", flush=True)
        # linger-free teardown: PUB sockets with unreachable peers otherwise
        # hold the interpreter open for seconds after a replay exits
        # (2026-09-05: the next launch's one-sender guard saw the ghost).
        try:
            zmq.Context.instance().destroy(linger=0)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        rc = 130
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)   # never wait on the XRT SDK's native threads at interpreter shutdown
