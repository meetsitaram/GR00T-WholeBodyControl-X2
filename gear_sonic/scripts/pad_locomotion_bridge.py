#!/usr/bin/env python3
"""Gamepad -> kplanner continuous-locomotion bridge (PS/Xbox pad).

Runs ALONGSIDE ``play_xbox_controller.py`` (gestures/dances/e-stop): this
script owns ONLY the analog sticks -> ``planner_cmd`` continuous locomotion,
mirroring the Quest3 manager's payloads so the kplanner needs zero changes.

Controls (consistent with play_xbox_controller's conventions):
  L2+R2 BOTH held   = deadman: sticks are live
    left stick      = forward/back + lateral
    right stick X   = yaw
    L1 (tap)        = speed setpoint -0.1   (kplanner-side, 0.2..1.0)
    R1 (tap)        = speed setpoint +0.1
  triggers released = one zero-stick command (robot idles), sticks dead
  L1+R1+L2+R2      = e-stop chord (owned by play_xbox_controller; we also
                      emit a zero-cmd for good measure)

Wire: PUB-connect to the stack's planner_cmd SUB (default 127.0.0.1:5563),
multipart [b"planner_cmd", json] -- same frames as quest3_manager_x2.py.

    python gear_sonic/scripts/pad_locomotion_bridge.py            # defaults
    python gear_sonic/scripts/pad_locomotion_bridge.py --probe    # mapping check
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pygame
import zmq

import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))
try:
    from gear_sonic.utils.teleop.estop_gesture import EstopGesture  # noqa: E402
except ImportError:
    # PC2 runs this file standalone from the PC2 install prefix with
    # estop_gesture.py shipped alongside it.
    from estop_gesture import EstopGesture  # noqa: E402

# Pad mapping (Xbox-style SDL defaults; PS DualSense/DS4 map the same core
# indices under SDL2 for sticks/shoulders; verify with --probe if unsure).
AXIS_LX, AXIS_LY = 0, 1
AXIS_RX = 3          # right stick X (some pads: 2 -- use --probe / --axis-rx)
AXIS_RY = 4          # right stick Y (SDL Xbox: 0=LX 1=LY 2=LT 3=RX 4=RY 5=RT)
AXIS_LT, AXIS_RT = 2, 5
BTN_LB, BTN_RB = 4, 5   # L1 / R1

DEADZONE = 0.15
RATE_HZ = 50.0
HEARTBEAT_S = 0.2        # re-publish while driving even if sticks unchanged
CHANGE_EPS = 0.02


def _dz(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else max(-1.0, min(1.0, v))


class _ZmqPad:
    """Duck-typed pygame.Joystick reading pc2_pad_daemon's pad_state feed.

    SAFETY: if no pad_state arrives for ``stale_s`` the state zeroes out --
    triggers read released -> the bridge's deadman drops -> robot idles.
    """

    def __init__(self, host: str, port: int, stale_s: float = 0.5) -> None:
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"pad_state")
        self._sub.setsockopt(zmq.RCVTIMEO, 0)   # non-blocking drain
        self._sub.setsockopt(zmq.CONFLATE, 0)
        self._sub.connect(f"tcp://{host}:{port}")
        self._axes = [0.0] * 8
        # triggers rest at -1 in SDL convention
        self._axes[AXIS_LT] = -1.0
        self._axes[AXIS_RT] = -1.0
        self._buttons = [0] * 16
        # D-pad arrives as hats over ZMQ (see pc2_pad_daemon). Without
        # this the bridge reports zero hats and any D-pad binding is a
        # silent no-op.
        self._hats = []
        self._last_rx = 0.0
        self._stale_s = stale_s

    def pump(self) -> None:
        while True:
            try:
                _, payload = self._sub.recv_multipart()
            except zmq.Again:
                break
            d = json.loads(payload)
            ax = d.get("axes", [])
            bt = d.get("buttons", [])
            self._axes[:len(ax)] = ax
            self._buttons[:len(bt)] = bt
            self._hats = [tuple(h) for h in d.get("hats", [])]
            self._last_rx = time.monotonic()
        if self._last_rx and time.monotonic() - self._last_rx > self._stale_s:
            # feed lost -> fail safe to released
            self._axes = [0.0] * len(self._axes)
            self._axes[AXIS_LT] = -1.0
            self._axes[AXIS_RT] = -1.0
            self._buttons = [0] * len(self._buttons)

    def get_axis(self, i: int) -> float:
        return self._axes[i] if i < len(self._axes) else 0.0

    def get_numhats(self) -> int:
        return len(self._hats)

    def get_hat(self, i: int):
        return self._hats[i] if i < len(self._hats) else (0, 0)

    def get_button(self, i: int) -> int:
        return self._buttons[i] if i < len(self._buttons) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5563)
    ap.add_argument("--topic", default="planner_cmd")
    ap.add_argument("--axis-rx", type=int, default=AXIS_RX)
    ap.add_argument("--bind", action="store_true",
                    help="bind the PUB (pad-only mode, no quest3 manager); "
                         "default is connect (manager owns the bind)")
    ap.add_argument("--source", choices=("local", "zmq"), default="local",
                    help="'local' = pygame reads a directly-attached pad; "
                         "'zmq' = consume pad_state events from a "
                         "pc2_pad_daemon (pad lives on the robot)")
    ap.add_argument("--pad-host", default=os.environ.get("PC2_HOST", ""),
                    help="(--source zmq) host running pc2_pad_daemon "
                         "(default: $PC2_HOST; required with --source zmq)")
    ap.add_argument("--pad-port", type=int, default=5569)
    ap.add_argument("--deadman", choices=("both", "left"), default="both",
                    help="'both' = L2+R2 held (two-handed); 'left' = L2 only "
                         "(one-handed drive, right hand free to spot the robot)")
    ap.add_argument("--lock-speed", action="store_true",
                    help="disable L1/R1 speed nudges: the kplanner setpoint "
                         "stays at its launch value (constant-speed demo mode)")
    ap.add_argument("--clip-pkl", default=None,
                    help="enable the dance chord: L1+Y (Triangle on PS) plays "
                         "this motion PKL via the recorder's motion_clip_cmd "
                         "wire; L1+B (Circle) stops it. Only fires while the "
                         "deadman is RELEASED (no dancing mid-drive).")
    ap.add_argument("--clip-key", default=None,
                    help="motion_key inside --clip-pkl (default: first key)")
    ap.add_argument("--clip-keys", default=None,
                    help="comma-separated motion_keys: L1+Y cycles FORWARD "
                         "through the list, L1+A cycles BACKWARD (each press "
                         "plays the next/prev clip). Overrides --clip-key.")
    ap.add_argument("--clip-host", default="127.0.0.1",
                    help="recorder host owning the motion_clip_cmd SUB bind")
    ap.add_argument("--clip-port", type=int, default=5568)
    ap.add_argument("--clip-topic", default="motion_clip_cmd")
    ap.add_argument("--no-log-buttons", dest="log_buttons",
                    action="store_false",
                    help="Silence per-press button logging (ON by default "
                         "since 2026-08-03 e-stop debugging).")
    ap.add_argument("--log-buttons", action="store_true",
                    help="log every button-index rising edge (pad mapping aid)")
    ap.set_defaults(log_buttons=True)
    ap.add_argument("--probe", action="store_true",
                    help="print axis/button events and exit on ctrl-c")
    ap.add_argument("--pad-name", default=os.environ.get("PAD_NAME", ""),
                    help="Bind the gamepad whose name contains this substring "
                         "(case-insensitive). Required (with "
                         "--allow-multiple-pads) when >1 pad is connected.")
    ap.add_argument("--allow-multiple-pads", action="store_true",
                    help="Override the exclusive-access refusal when multiple "
                         "gamepads are connected; --pad-name must uniquely "
                         "select one.")
    ap.add_argument("--invert-ly", action="store_true", default=True,
                    help="stick up (-1 in SDL) = forward (default on)")
    ap.add_argument("--clip-key-dpad-up", default="",
                    help="single motion_key fired by L1 + D-pad UP. Separate from "
                         "the two cycling banks: this is a fixed showcase clip, so "
                         "it is always the same motion regardless of bank state.")
    ap.add_argument("--clip-keys-b", default="",
                    help="second bank, cycled with L1+X (next) / L1+B (prev). "
                         "STOP moves to L1+R1 when this is used.")
    ap.add_argument("--clip-keys-m", default="",
                    help="MEDIUM-dance bank, cycled forward with L1+B. Setting "
                         "this takes L1+B over from 'bank B previous', so every "
                         "face button owns one bank: Y=dances X=combat "
                         "A=gestures B=medium.")
    ap.add_argument("--clip-keys-turn", default="",
                    help="RIGHT-STICK 4-way clip bank, comma separated in the "
                         "order LEFT,RIGHT,UP,DOWN. Fires only while the deadman "
                         "is RELEASED (the stick is yaw when it is held). "
                         "LEFT/RIGHT fire immediately -- they are in-place turns "
                         "that do not travel. UP/DOWN require a hold "
                         "(--turn-hold-s) because those clips cover ground.")
    ap.add_argument("--clip-keys-micro", default="",
                    help="LEFT-STICK 8-way MICRO-STEP clip bank (operator 2026-09-05: replay "
                         "motions instead of relying on the kplanner), comma separated in "
                         "the order F,B,L,R,FL,FR,BL,BR. Fires only while the deadman is "
                         "RELEASED and the RIGHT TRIGGER is held (left trigger + sticks stays navigation); each "
                         "sector fires once per stick entry, then --micro-cooldown-s.")
    ap.add_argument("--micro-deflect", type=float, default=0.6)
    ap.add_argument("--micro-cooldown-s", type=float, default=3.5,
                    help="seconds before another micro-step clip may fire (clips are ~2-3 s)")
    ap.add_argument("--turn-hold-s", type=float, default=2.0,
                    help="dwell required for the travelling UP/DOWN clips")
    ap.add_argument("--turn-deflect", type=float, default=0.6,
                    help="stick magnitude that counts as a deflection")
    ap.add_argument("--clip-keys-g", default="",
                    help="GESTURE bank, cycled forward with L1+A. Setting this "
                         "takes L1+A over from 'bank A previous' -- bank A then "
                         "only advances (L1+Y).")
    ap.add_argument("--rumble-port", type=int, default=5570,
                    help="pc2_pad_daemon PULL port for haptics")
    args = ap.parse_args()

    if args.source == "zmq":
        js = _ZmqPad(args.pad_host, args.pad_port)
        print(f"[pad-bridge] pad source: pc2_pad_daemon at "
              f"tcp://{args.pad_host}:{args.pad_port}", flush=True)
    else:
        pygame.init()
        pygame.joystick.init()
        n_pads = pygame.joystick.get_count()
        if n_pads == 0:
            print("[pad-bridge] no gamepad found"); return 1
        pads = [pygame.joystick.Joystick(i) for i in range(n_pads)]
        names = [p.get_name() for p in pads]
        # EXCLUSIVE-ACCESS GUARD (2026-07-29 incident): binding Joystick(0)
        # with multiple pads connected let a stray paired controller (with a
        # different driver axis layout) become the robot's input. Rules:
        #   - exactly one pad connected -> use it;
        #   - multiple pads -> REFUSE unless --pad-name uniquely selects one
        #     AND --allow-multiple-pads is explicitly passed.
        if n_pads > 1 and not args.allow_multiple_pads:
            print(f"[pad-bridge] REFUSING to start: {n_pads} gamepads connected "
                  f"({', '.join(names)}). Disconnect/unpair the extras, or pass "
                  f"--pad-name <substring> --allow-multiple-pads to bind one "
                  f"explicitly.", flush=True)
            return 1
        idx = 0
        if args.pad_name:
            want = args.pad_name.lower()
            matches = [i for i, nm in enumerate(names) if want in nm.lower()]
            if len(matches) != 1:
                print(f"[pad-bridge] REFUSING: --pad-name '{args.pad_name}' "
                      f"matches {len(matches)} of {names}", flush=True)
                return 1
            idx = matches[0]
        js = pads[idx]; js.init()
        print(f"[pad-bridge] pad: {js.get_name()} "
              f"(axes={js.get_numaxes()} buttons={js.get_numbuttons()}); "
              f"{n_pads} pad(s) enumerated, bound index {idx}", flush=True)
        # DEADMAN-AXIS SANITY (2026-07-29 incident, second layer): SDL triggers
        # rest at -1.0; stick axes rest near 0. If the configured deadman axis
        # does not rest deep-negative, the driver's axis layout is not what we
        # assume (e.g. a stick landed on the trigger slot) and rest-noise can
        # phantom-engage the deadman. Refuse rather than guess.
        pygame.event.pump()
        _lt_rest = js.get_axis(AXIS_LT)
        _rt_rest = js.get_axis(AXIS_RT)
        if _lt_rest > -0.5 or (args.deadman != "left" and _rt_rest > -0.5):
            print(f"[pad-bridge] REFUSING: deadman axis rest values LT={_lt_rest:+.2f} "
                  f"RT={_rt_rest:+.2f} (expected < -0.5 for triggers). Wrong "
                  f"controller/driver axis layout — do NOT trust this mapping. "
                  f"(Release the triggers and restart if you were holding them.)",
                  flush=True)
            return 1

    if args.probe:
        prev = None
        while True:
            pygame.event.pump()
            state = ([round(js.get_axis(i), 1) for i in range(js.get_numaxes())],
                     [js.get_button(i) for i in range(js.get_numbuttons())])
            if state != prev:
                print(f"axes={state[0]} buttons={state[1]}", flush=True)
                prev = state
            time.sleep(0.05)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 10)
    if args.bind:
        # Pad-only sessions: the quest3 manager (the usual :5563 binder) is
        # not running, so WE bind and the kplanner's connected SUB attaches.
        sock.bind(f"tcp://*:{args.port}")
        print(f"[pad-bridge] PUB bind tcp://*:{args.port} "
              f"topic={args.topic!r}; hold L2+R2 to drive", flush=True)
    else:
        sock.connect(f"tcp://{args.host}:{args.port}")
        print(f"[pad-bridge] PUB connect tcp://{args.host}:{args.port} "
              f"topic={args.topic!r}; hold L2+R2 to drive", flush=True)
    time.sleep(0.3)  # slow-joiner

    def send(payload: dict) -> None:
        # Ownership tag: the planner's command SUB enforces single-source
        # control (VR supersedes pad); untagged messages count as pad.
        payload.setdefault("source", "pad")
        sock.send_multipart([args.topic.encode("ascii"),
                             json.dumps(payload).encode("utf-8")])

    # Dance chord wire (optional): separate PUB to the recorder's
    # motion_clip_cmd SUB. Payload mirrors play_locomotion.py exactly.
    clip_sock = None
    # TWO independent dance banks, each with its own next/prev pair:
    #   L1+Y / L1+A  -> bank A ("easy")
    #   L1+X / L1+B  -> bank B ("medium")
    # STOP moved to L1+R1 because B is now bank-B-previous. Keeping a stop is
    # worth a chord: some clips can tip the robot (a wide-swing dance clip
    # fell in sim under BOTH .pt and ONNX), so aborting mid-clip matters.
    btn_a_next, btn_a_prev = 3, 0            # Xbox: Y / A
    btn_b_next, btn_b_prev = 2, 1            # Xbox: X / B
    clip_list: list[str] = []                # bank A
    clip_list_b: list[str] = []              # bank B
    clip_list_g: list[str] = []              # bank G (gestures, L1+A)
    clip_list_m: list[str] = []              # bank M (medium dances, L1+B)
    clip_cur = -1
    clip_cur_b = -1
    clip_cur_g = -1
    clip_cur_m = -1
    turn_list: list[str] = []                # [LEFT, RIGHT, UP, DOWN]
    turn_prev_dir = None                     # debounce: fire once per entry
    turn_hold_start = None
    micro_list: list[str] = []
    micro_prev_dir = None
    micro_cooldown_until = 0.0
    if args.clip_pkl:
        pad_name = js.get_name().lower() if hasattr(js, "get_name") else ""
        if any(k in pad_name for k in ("dualsense", "sony", "playstation")):
            # PS face cluster: Triangle / Cross for bank A, Square / Circle for B
            btn_a_next, btn_a_prev = 2, 0
            btn_b_next, btn_b_prev = 3, 1
        if args.clip_keys:
            clip_list = [k for k in args.clip_keys.split(",") if k.strip()]
        elif args.clip_key:
            clip_list = [args.clip_key]
        if getattr(args, "clip_keys_b", ""):
            clip_list_b = [k for k in args.clip_keys_b.split(",") if k.strip()]
        if getattr(args, "clip_keys_g", ""):
            clip_list_g = [k for k in args.clip_keys_g.split(",") if k.strip()]
        if getattr(args, "clip_keys_m", ""):
            clip_list_m = [k for k in args.clip_keys_m.split(",") if k.strip()]
        if getattr(args, "clip_keys_turn", ""):
            turn_list = [k.strip() for k in args.clip_keys_turn.split(",")]
            if len(turn_list) != 4:
                raise SystemExit("--clip-keys-turn needs exactly 4 keys: "
                                 "LEFT,RIGHT,UP,DOWN")
        if getattr(args, "clip_keys_micro", ""):
            micro_list = [k.strip() for k in args.clip_keys_micro.split(",")]
            if len(micro_list) != 8:
                raise SystemExit("--clip-keys-micro needs exactly 8 keys: F,B,L,R,FL,FR,BL,BR")
            print(f"[pad-bridge] MICRO-STEP bank armed on the LEFT stick (right trigger held, left-trigger deadman released): "
                  f"{micro_list}", flush=True)
        clip_sock = ctx.socket(zmq.PUB)
        clip_sock.setsockopt(zmq.SNDHWM, 10)
        clip_sock.connect(f"tcp://{args.clip_host}:{args.clip_port}")
        print(f"[pad-bridge] dance chord armed: L1+Y next / L1+A prev of "
              f"{len(clip_list)} clip(s), L1+B stops; deadman must be "
              f"released. pkl={args.clip_pkl}", flush=True)

    def send_clip(payload: dict) -> None:
        clip_sock.send_multipart([args.clip_topic.encode("ascii"),
                                  json.dumps(payload).encode("utf-8")])

    # Rumble path for the ZMQ-fed pad. This process reads the pad but never
    # HOLDS it -- pc2_pad_daemon owns the pygame joystick, so it owns rumble.
    # Before this socket existed, every buzz() on the robot was a silent no-op.
    rumble_sock = None
    if args.source == "zmq":
        rumble_sock = ctx.socket(zmq.PUSH)
        rumble_sock.setsockopt(zmq.SNDHWM, 10)
        rumble_sock.setsockopt(zmq.LINGER, 0)
        rumble_sock.setsockopt(zmq.SNDTIMEO, 0)     # never block the 50Hz loop
        rumble_sock.connect(f"tcp://{args.pad_host}:{args.rumble_port}")
        print(f"[pad-bridge] rumble PUSH -> "
              f"tcp://{args.pad_host}:{args.rumble_port}", flush=True)

    def buzz(strength: float, ms: int, count: int = 1, gap_ms: int = 180) -> None:
        """Haptic cue. Multi-pulse patterns are scheduled by the DAEMON, not
        slept through here -- a sleep would stall intent publishing."""
        if rumble_sock is not None:
            try:
                rumble_sock.send_json({"strength": strength, "ms": ms,
                                       "count": count, "gap_ms": gap_ms})
            except Exception:  # noqa: BLE001 -- haptics are best-effort
                pass
            return
        rumble = getattr(js, "rumble", None)      # locally-attached pad
        if rumble is None:
            return
        try:
            for _ in range(count):
                rumble(strength, strength, ms)
        except Exception:  # noqa: BLE001
            pass

    tick = 1.0 / RATE_HZ
    last_sent: tuple | None = None
    last_pub_ts = 0.0
    was_live = False
    prev_lb = prev_rb = False
    prev_a_next = prev_a_prev = prev_b_next = prev_b_prev = prev_rb = False
    prev_hat_up = False
    prev_hat_y = 0
    crouch_armed = False; crouch_ka_ts = 0.0     # hold-to-crouch state
    last_head, last_head_ts = 0.0, 0.0
    showcase_hold_start = None      # L2+R2+Y hold timer
    showcase_buzzed = False
    clip_cooldown_until = 0.0
    pending_delta = 0.0
    estop_gesture = EstopGesture()
    _estop_armed_log_t = 0.0

    _dbg_prev_btns: set[int] = set()
    while True:
        if isinstance(js, _ZmqPad):
            js.pump()
        else:
            pygame.event.pump()
        # Button-index debug: log rising edges so pad layouts can be
        # mapped from a live session (harmless, low-rate).
        if args.log_buttons and not isinstance(js, _ZmqPad):
            cur_btns = {b for b in range(js.get_numbuttons()) if js.get_button(b)}
            for b in sorted(cur_btns - _dbg_prev_btns):
                _ts = time.strftime("[%H:%M:%S]")
                print(f"{_ts} [pad-bridge] button {b} pressed", flush=True)
            _dbg_prev_btns = cur_btns
        lt = js.get_axis(AXIS_LT) > 0.0   # SDL triggers: -1 rest -> +1 pressed
        rt = js.get_axis(AXIS_RT) > 0.0
        lb, rb = bool(js.get_button(BTN_LB)), bool(js.get_button(BTN_RB))
        deadman_held = lt if args.deadman == "left" else (lt and rt)
        live = deadman_held and not (lb and rb)   # all-four = e-stop chord: go dead

        # ---- Operator E-STOP gesture (design 2026-08-04): rapid L2/R2
        # firing while A and X are HELD, >=3 trigger cycles inside 1 s.
        # On trip: broadcast estop intent (planner idles immediately; the
        # deploy's damping-slam handler consumes the same message once the
        # rebuilt deploy ships) + touch the soft-shutdown sentinel if
        # X2_SOFT_SHUTDOWN_SENTINEL is set (graceful RAMP_OUT today).
        lt_lvl = (js.get_axis(AXIS_LT) + 1.0) / 2.0
        rt_lvl = (js.get_axis(AXIS_RT) + 1.0) / 2.0
        estop_chord = bool(js.get_button(0)) and bool(js.get_button(2))  # A+X
        # NOTE (2026-08-03 incident): the both-sticks-pinned alternative
        # chord was REMOVED. Left stick forward (walking) + right stick
        # forward (no-op) is a normal driving posture, and L2/R2 are the
        # deadman — an operator re-gripping mid-walk pumped 3 times and
        # collapsed the robot. A+X is the sole chord: it cannot be held
        # while a thumb drives a stick.
        if estop_chord and time.monotonic() - _estop_armed_log_t > 0.5:
            _estop_armed_log_t = time.monotonic()
            print(f"[pad-bridge {time.strftime('%H:%M:%S')}] estop chord "
                  f"armed (A+X): lt={lt_lvl:.2f} rt={rt_lvl:.2f} "
                  f"pumps_1s={len(estop_gesture._pump_win)}", flush=True)
        _ph = estop_gesture.tick(lt_lvl, rt_lvl, estop_chord)
        if _ph == 1:
            print("[pad-bridge] !!! E-STOP (SOFT): abort to idle stand — "
                  "keep the gesture going ~1s more for PURE DAMPING",
                  flush=True)
            send({"intent": "estop", "phase": "soft",
                  "magnitude": "default"})
        elif _ph == 2:
            print("[pad-bridge] !!! E-STOP ESCALATED: PURE DAMPING (terminal)",
                  flush=True)
            send({"intent": "estop", "phase": "damp",
                  "magnitude": "default"})
        if _ph:
            sentinel = os.environ.get("X2_SOFT_SHUTDOWN_SENTINEL", "")
            if sentinel:
                try:
                    open(sentinel, "w").close()
                    print(f"[pad-bridge] soft-shutdown sentinel touched: "
                          f"{sentinel}", flush=True)
                except OSError as exc:
                    print(f"[pad-bridge] sentinel write failed: {exc}",
                          flush=True)

        # (Showcase chord L2+R2+Y REMOVED 2026-07-30 per operator: it committed
        # the robot to ~10 s of continuous walking with ~293 deg of turning —
        # too much autonomous motion for one chord. Play showcase clips
        # deliberately via play_locomotion.py instead.)
        # ---- RIGHT-STICK 4-way clip bank -----------------------------
        # Deadman RELEASED only: with it held the right stick is yaw, so the
        # two uses can never collide. LEFT/RIGHT are in-place turns (~0.09 m
        # of travel) and fire on entry -- no dwell, nothing to guard against.
        # UP/DOWN cover ground (3.5 m / 5.1 m) so they need a deliberate hold.
        # MICRO-STEP bank: RIGHT TRIGGER held + LEFT stick, 8 sectors of 45 deg,
        # with the walk deadman (LEFT TRIGGER, the operator's "L1") released.
        # Operator 2026-09-05: "I press L1 and move the sticks to navigate; R1 +
        # left stick in 8 directions for the micro-steps" -- his L1/R1 are the
        # TRIGGERS (code: lt/rt), not the bumpers (lb/rb). With --deadman left
        # the right trigger is otherwise unused; with --deadman both the bank
        # cannot fire because rt alone never makes the sticks live.
        # (crouch_armed: the first tick after L2 is released during a crouch is exactly "R2 + stick" -- the
        #  crouch block below clears the arm and sets the bank's cooldown, so the bank must skip this tick)
        if clip_sock is not None and micro_list and not live and rt and not lb and not crouch_armed:   # rt is a bool (axis > 0)
            lx = js.get_axis(AXIS_LX)
            ly = js.get_axis(AXIS_LY)
            ly = -ly if args.invert_ly else ly          # stick up = +ly (forward)
            mdir = None
            if max(abs(lx), abs(ly)) >= args.micro_deflect:
                import math as _m
                ang = (_m.degrees(_m.atan2(lx, ly)) + 360.0) % 360.0   # 0 = forward, 90 = right
                mdir = ("F", "FR", "R", "BR", "B", "BL", "L", "FL")[int(((ang + 22.5) % 360.0) // 45.0)]
            now_m = time.monotonic()
            if mdir is None:
                micro_prev_dir = None
            elif mdir != micro_prev_dir:
                micro_prev_dir = mdir                   # fresh entry into a sector
                if now_m >= micro_cooldown_until:
                    key = micro_list[("F", "B", "L", "R", "FL", "FR", "BL", "BR").index(mdir)]
                    if not key:
                        print(f"[pad-bridge] MICRO {mdir}: no clip bound", flush=True)
                    elif key.startswith("prim:"):
                        # kplanner PRIMITIVE (x2_planner_primitives.pkl, "_step" names play
                        # forward once and persist the root XY). 2026-09-06 sim: the x2m2
                        # clip player pins the pelvis, so stepping clips became foot slides;
                        # the mocap step primitives (fwd_step_1ft, back_step_half_ft,
                        # side_*_step) execute real swings under the 40.5k policy.
                        send({"intent": "primitive", "name": key[5:], "magnitude": "default"})
                    else:
                        send_clip({"action": "play", "pkl": args.clip_pkl,
                                   "kind": "locomotion", "motion_key": key})
                    micro_cooldown_until = now_m + args.micro_cooldown_s
                    buzz(0.7, 120)
                    print(f"[pad-bridge] MICRO {mdir} -> {key}", flush=True)
        else:
            # bank inactive (R2 released, sticks live, or a crouch exit in progress): forget the sector so the
            # next R2 + stick is a fresh entry (before 2026-09-09 a sector recorded during an inactive spell
            # stuck until the stick was centred WITH R2 held -- the bank looked dead after a crouch exit)
            micro_prev_dir = None

        # L1 must be held, same as every other clip trigger -- an
        # ungated stick is the one surface a knock can fire.
        if clip_sock is not None and turn_list and not live and lb:
            rx = js.get_axis(args.axis_rx)
            ry = js.get_axis(AXIS_RY)
            ry = -ry if args.invert_ly else ry          # stick up = +ry
            d_ = args.turn_deflect
            tdir = None
            if max(abs(rx), abs(ry)) >= d_:             # dominant axis wins
                tdir = ("R" if rx > 0 else "L") if abs(rx) >= abs(ry) else \
                       ("U" if ry > 0 else "D")
            now_t = time.monotonic()
            if tdir is None:
                turn_prev_dir = None
                turn_hold_start = None
            elif tdir != turn_prev_dir:
                turn_prev_dir = tdir                    # fresh entry
                turn_hold_start = now_t
                if tdir in ("U", "D"):
                    # tell the operator the hold timer is COUNTING; without
                    # this a 2 s wait is indistinguishable from a dead binding.
                    buzz(0.5, 90)
                if tdir in ("L", "R") and now_t >= clip_cooldown_until:
                    key = turn_list[0 if tdir == "L" else 1]
                    send_clip({"action": "play", "pkl": args.clip_pkl,
                               "kind": "locomotion", "motion_key": key})
                    clip_cooldown_until = now_t + 5.5   # clip is ~4.8 s
                    buzz(0.7, 120)
                    print(f"[pad-bridge] TURN {tdir} -> {key}", flush=True)
                    turn_hold_start = None              # consumed
            elif (tdir in ("U", "D") and turn_hold_start is not None
                  and now_t - turn_hold_start >= args.turn_hold_s
                  and now_t >= clip_cooldown_until):
                key = turn_list[2 if tdir == "U" else 3]
                send_clip({"action": "play", "pkl": args.clip_pkl,
                           "kind": "locomotion", "motion_key": key})
                clip_cooldown_until = now_t + 3.0
                buzz(0.8, 180, count=2, gap_ms=180)
                print(f"[pad-bridge] TRAVEL {tdir} (held "
                      f"{args.turn_hold_s:g}s) -> {key}", flush=True)
                turn_hold_start = None                  # consumed

        # Speed nudges: shoulder taps while deadman held (rising edges).
        # Suppressed entirely in --lock-speed mode (constant-speed demos:
        # stray shoulder presses while gripping the triggers are easy).
        if not args.lock_speed:
            if live and lb and not prev_lb:
                pending_delta -= 0.1
            if live and rb and not prev_rb:
                pending_delta += 0.1
        prev_lb, prev_rb = lb, rb

        # D-pad speed nudges WHILE DRIVING (2026-08-12 operator feature):
        # up/down = +-0.2 m/s on the daemon's forward setpoint (daemon
        # clamps to [0.3, 0.7] and resets to default on every stop).
        # Deliberate presses, so NOT suppressed by --lock-speed (that
        # guard exists for accidental shoulder grips). No conflict with
        # the L1+dpad clip binding: that one requires deadman RELEASED.
        hat_y = 0
        try:
            if getattr(js, "get_numhats", None) and js.get_numhats() > 0:
                hat_y = js.get_hat(0)[1]
        except Exception:
            hat_y = 0
        # ---- CROUCH MODE, HOLD-TO-CROUCH (operator 2026-09-09: "should go back to standing as soon as L2 or
        # R2 are released, instead of waiting for the dpad up command -- safer this way"). ARM with BOTH triggers
        # held + D-pad DOWN (sticks centred, rising edge); while armed and both triggers stay held the bridge
        # re-sends enable=true at 5 Hz (the kplanner drops the mode after 0.6 s without it); releasing EITHER
        # trigger (or D-pad UP) sends enable=false at once and disarms. The kplanner owns the entry gate
        # (standing still) and stamps the crouch hip only on forward/turn targets. L1+R1 rejected as the key:
        # both bumpers held make the sticks go dead (half of the four-button e-stop chord).
        both_triggers = lt and rt
        sticks_centred = (abs(js.get_axis(AXIS_LX)) < 0.3 and abs(js.get_axis(AXIS_LY)) < 0.3
                          and abs(js.get_axis(args.axis_rx)) < 0.3)
        now_cr = time.monotonic()
        if crouch_armed and (not both_triggers or (hat_y > 0 and hat_y != prev_hat_y)):
            crouch_armed = False
            send({"intent": "crouch_mode", "enable": False, "magnitude": "default"})
            buzz(0.5, 120)
            # Exit interference (operator 2026-09-09): letting go of L2 first while the left stick is still
            # deflected leaves "R2 held, L2 released" = the MICRO-STEP bank's arming state, and the stick
            # sector would fire a step primitive on the way out of a crouch. Hold the bank off for a moment.
            micro_cooldown_until = max(micro_cooldown_until, now_cr + 1.5)
            micro_prev_dir = "exit"
            print("[pad-bridge] CROUCH MODE OFF (" + ("trigger released" if not both_triggers else "D-pad UP") + "); micro bank held off 1.5 s", flush=True)
        elif (not crouch_armed and both_triggers and sticks_centred and hat_y < 0 and hat_y != prev_hat_y):
            crouch_armed = True; crouch_ka_ts = 0.0
            buzz(0.6, 150, count=2)
            print("[pad-bridge] CROUCH MODE ARMED (L2+R2 + D-pad DOWN): crouched while both triggers stay held; "
                  "kplanner gates the entry (see its log)", flush=True)
        if crouch_armed and now_cr - crouch_ka_ts >= 0.2:
            send({"intent": "crouch_mode", "enable": True, "magnitude": "default"})   # arm + keepalive
            crouch_ka_ts = now_cr
        # D-pad speed nudges only when the chord is NOT held (both triggers = mode chord)
        if live and not rt and hat_y != prev_hat_y:
            if hat_y > 0:
                pending_delta += 0.2
                buzz(0.4, 80)
            elif hat_y < 0:
                pending_delta -= 0.2
                buzz(0.25, 80)
        prev_hat_y = hat_y

        # Dance chord: L1 + play-button, rising edge, deadman released,
        # 2s cooldown. Stop chord: L1 + stop-button. Both no-ops mid-drive
        # so a dance can never preempt active locomotion.
        if clip_sock is not None:
            # D-pad: pygame exposes it as a hat, (x, y) with y=+1 up.
            # Guard on get_numhats -- the ZMQ-fed pad and some layouts have none.
            hat_up = False
            try:
                if getattr(js, "get_numhats", None) and js.get_numhats() > 0:
                    hat_up = js.get_hat(0)[1] > 0
            except Exception:
                hat_up = False
            a_next_b = bool(js.get_button(btn_a_next))
            a_prev_b = bool(js.get_button(btn_a_prev))
            b_next_b = bool(js.get_button(btn_b_next))
            b_prev_b = bool(js.get_button(btn_b_prev))
            rb_b     = bool(js.get_button(BTN_RB))
            now_c = time.monotonic()
            ready = (not live) and now_c >= clip_cooldown_until
            bank = None; step = 0
            if lb and a_next_b and not prev_a_next and ready:
                bank, step = "A", 1          # L1+Y
            elif lb and a_prev_b and not prev_a_prev and ready:
                # L1+A = GESTURE bank, cycles forward. It is deliberately NOT
                # "bank A previous" any more -- with 8 gestures a prev button
                # is worth less than a third group.
                bank, step = "G", 1          # L1+A
            elif lb and b_next_b and not prev_b_next and ready:
                bank, step = "B", 1          # L1+X
            elif lb and b_prev_b and not prev_b_prev and ready:
                # L1+B = MEDIUM dances, cycles forward. One bank per face
                # button beats next/prev on two of them.
                bank, step = "M", 1          # L1+B

            lst = {"A": clip_list, "B": clip_list_b,
                   "G": clip_list_g, "M": clip_list_m}.get(bank, [])
            if bank and lst:
                if bank == "A":
                    clip_cur = (clip_cur + step) % len(lst); idx = clip_cur
                elif bank == "G":
                    clip_cur_g = (clip_cur_g + step) % len(lst); idx = clip_cur_g
                elif bank == "M":
                    clip_cur_m = (clip_cur_m + step) % len(lst); idx = clip_cur_m
                else:
                    clip_cur_b = (clip_cur_b + step) % len(lst); idx = clip_cur_b
                key = lst[idx]
                send_clip({"action": "play", "pkl": args.clip_pkl,
                           "kind": "locomotion", "motion_key": key})
                clip_cooldown_until = now_c + 2.0
                buzz(0.7, 200)   # dance fired: single firm pulse
                print(f"[pad-bridge] DANCE[{bank}] {idx + 1}/{len(lst)} "
                      f"-> {key}", flush=True)
            elif (lb and hat_up and not prev_hat_up and ready
                    and args.clip_key_dpad_up):
                # L1 + D-pad UP -> fixed showcase clip
                send_clip({"action": "play", "pkl": args.clip_pkl,
                           "kind": "locomotion",
                           "motion_key": args.clip_key_dpad_up})
                clip_cooldown_until = now_c + 2.0
                buzz(0.7, 200)
                print(f"[pad-bridge] DPAD-UP -> {args.clip_key_dpad_up}", flush=True)
            elif lb and rb_b and not prev_rb and not live:
                # L1+R1 = STOP (moved off B, which is now bank-B-previous)
                send_clip({"action": "stop"})
                clip_cooldown_until = now_c + 1.0
                buzz(0.3, 100)   # stop: weak blip
                print("[pad-bridge] dance STOP chord (L1+R1) -> idle", flush=True)
            prev_a_next, prev_a_prev = a_next_b, a_prev_b
            prev_b_next, prev_b_prev = b_next_b, b_prev_b
            prev_rb = rb_b
            prev_hat_up = hat_up

        # Input-change logging: every deadman transition and (throttled)
        # stick change is printed so the operator can see what the pad
        # is sending without watching the planner log.
        if live and not was_live:
            print(f"[pad-bridge] DEADMAN ENGAGED (L2+R2) -> sticks live", flush=True)

        if live:
            fwd = _dz(-js.get_axis(AXIS_LY)) if args.invert_ly else _dz(js.get_axis(AXIS_LY))
            side = _dz(js.get_axis(AXIS_LX))
            yaw = _dz(js.get_axis(args.axis_rx))
            cur = (round(fwd, 3), round(side, 3), round(yaw, 3))
            now = time.monotonic()
            if (last_sent is None
                    or any(abs(a - b) > CHANGE_EPS for a, b in zip(cur, last_sent))
                    or pending_delta != 0.0
                    or now - last_pub_ts > HEARTBEAT_S):
                payload = {
                    "intent": "locomotion",
                    "magnitude": "continuous",
                    "stick_fwd": cur[0],
                    "stick_side": cur[1],
                    "stick_yaw": cur[2],
                }
                if pending_delta:
                    payload["speed_delta"] = round(pending_delta, 2)
                    print(f"[pad-bridge] speed_delta {pending_delta:+.1f}")
                    pending_delta = 0.0
                send(payload)
                if last_sent is None or any(
                    abs(a - b) > 0.1 for a, b in zip(cur, last_sent or (9, 9, 9))
                ):
                    print(f"[pad-bridge] sticks fwd={cur[0]:+.2f} "
                          f"side={cur[1]:+.2f} yaw={cur[2]:+.2f}", flush=True)
                last_sent, last_pub_ts = cur, now
            was_live = True
        elif was_live:
            # Deadman released (or e-stop chord): one zero-stick cmd -> idle.
            send({"intent": "locomotion", "magnitude": "continuous",
                  "stick_fwd": 0.0, "stick_side": 0.0, "stick_yaw": 0.0})
            print("[pad-bridge] deadman released -> idle")
            last_sent, was_live = None, False
            last_head = 0.0
        else:
            # HEAD LOOK (2026-08-12 operator feature): deadman RELEASED,
            # left-stick X drives head yaw. Zero-stick locomotion payload
            # + head_yaw_stick field; the daemon maps stick -> slew-
            # limited head-yaw overlay (idle + gait branches only) with
            # ABSENT-FIELD-RECENTRE semantics (the waist contract), so
            # live-drive payloads (no field) ease the head home. Send on
            # change or 0.25 s heartbeat while deflected; one final 0 on
            # recentre.
            hx = _dz(js.get_axis(AXIS_LX))
            now_h = time.monotonic()
            if (abs(hx - last_head) > 0.02
                    or (hx != 0.0 and now_h - last_head_ts > 0.25)):
                send({"intent": "locomotion", "magnitude": "continuous",
                      "stick_fwd": 0.0, "stick_side": 0.0,
                      "stick_yaw": 0.0, "head_yaw_stick": round(hx, 3)})
                if (hx == 0.0) != (last_head == 0.0):
                    print(f"[pad-bridge] head look "
                          f"{'CENTER' if hx == 0.0 else f'{hx:+.2f}'}",
                          flush=True)
                last_head, last_head_ts = hx, now_h

        time.sleep(tick)


if __name__ == "__main__":
    raise SystemExit(main())
