#!/usr/bin/env python3
"""PC2-resident gamepad daemon: pad events broadcast + SAFE sonic-start ritual.

Runs ON PC2 with the pad connected to PC2 (USB or BT). Two jobs:

1. **Pad-event broadcast** — publishes the full pad state (axes+buttons) as
   JSON on ``tcp://*:5569`` topic ``pad_state`` at 50 Hz (on change) so any
   consumer on the LAN can use the robot's own pad:
   laptop ``pad_locomotion_bridge --source zmq --pad-host <PC2>`` today;
   a PC2-resident kplanner later.

2. **SAFE START ritual** for the sonic daemons (safety-critical: sonic start
   triggers MC shutdown while a human steadies the robot — must be impossible
   to fire accidentally):

       HOLD L1+R1+L2+R2 together for >= 3.0 s   ->  ARMED (8 s window)
       then press Y within the window            ->  START daemons
       anything else / timeout / early release   ->  DISARMED

   On confirm it runs ``x2_pc2_daemons.sh start --no-confirm ...`` (the pad
   ritual REPLACES the interactive Y/n gate). All transitions are logged
   loudly; hook a speaker cue here later (PC3).

Usage (on PC2):
    python3 pc2_pad_daemon.py \
        --start-cmd "${PC2_PREFIX:-/home/run/gear-sonic}/gear_sonic_deploy/scripts/x2_pc2_daemons.sh start \
                     --model ${PC2_PREFIX:-/home/run/gear-sonic}/policies/<x2_sonic>.onnx \
                     --tuning .../walking_recovery_loose.yaml --lock-head-straight --no-confirm"

Deps on PC2: python3 + pygame + pyzmq (pip install pygame pyzmq if missing).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import pygame
import zmq

BTN_LB, BTN_RB = 4, 5   # L1/R1: same index on Xbox and DualSense
AXIS_LT, AXIS_RT = 2, 5  # triggers: same on both (rest -1)

HOLD_S = 3.0        # all-four hold to arm
WINDOW_S = 8.0      # confirm window after arming


def face_buttons(pad_name: str) -> tuple[int, tuple[int, ...]]:
    """(confirm_btn, disarm_btns) for this pad's face-button layout.

    Confirm = the TOP face button (Xbox Y = 3; DualSense Triangle = 2).
    Anything else in the face cluster disarms.
    """
    if any(k in pad_name.lower() for k in ("dualsense", "sony", "playstation")):
        return 2, (0, 1, 3)   # Triangle; Cross/Circle/Square disarm
    return 3, (0, 1, 2)       # Y; A/B/X disarm


def wait_for_pad() -> "pygame.joystick.Joystick":
    """Block until a joystick exists (hot-plug: BT drops are a thing)."""
    while True:
        pygame.joystick.quit(); pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            js = pygame.joystick.Joystick(0); js.init()
            print(f"[pc2-pad] pad: {js.get_name()}", flush=True)
            return js
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub-port", type=int, default=5569)
    ap.add_argument("--rumble-port", type=int, default=5570,
                    help="PULL port for rumble requests from the bridge")
    ap.add_argument("--start-cmd", required=True,
                    help="shell command to run on ritual confirm")
    ap.add_argument("--rate", type=float, default=50.0)
    args = ap.parse_args()

    pygame.init(); pygame.joystick.init()
    js = wait_for_pad()
    btn_confirm, btns_disarm = face_buttons(js.get_name())

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 10)
    pub.bind(f"tcp://*:{args.pub_port}")
    print(f"[pc2-pad] pad_state PUB bind tcp://*:{args.pub_port}", flush=True)
    # Rumble command channel. The bridge reads the pad over ZMQ but never HOLDS
    # it -- only this daemon owns the pygame joystick, so only this daemon can
    # rumble. Without this channel every buzz() in the bridge is a silent no-op.
    rum = ctx.socket(zmq.PULL)
    rum.setsockopt(zmq.RCVTIMEO, 0)      # non-blocking drain
    rum.setsockopt(zmq.RCVHWM, 20)
    rum.bind(f"tcp://*:{args.rumble_port}")
    print(f"[pc2-pad] rumble PULL bind tcp://*:{args.rumble_port}", flush=True)
    print(f"[pc2-pad] START ritual: hold L1+R1+L2+R2 {HOLD_S:.0f}s -> ARMED, "
          f"then Y within {WINDOW_S:.0f}s", flush=True)

    def rumble(strength: float, ms: int) -> None:
        """Haptic cue; best-effort (rumble support varies by transport)."""
        try:
            ok = js.rumble(strength, strength, ms)
            print(f"[pc2-pad {time.strftime('%H:%M:%S')}] rumble "
                  f"{strength:.1f}/{ms}ms -> {'OK' if ok else 'UNSUPPORTED'}",
                  flush=True)
        except Exception as e:
            print(f"[pc2-pad] rumble failed: {e}", flush=True)

    pulse_q: list[tuple[float, float, int]] = []   # (fire_at, strength, ms)
    hold_start = None
    hold_pulses = 0     # arming-progress pulses sent this hold (1/sec, 3 total)
    armed_until = 0.0
    cooldown_until = 0.0
    last_state = None
    tick = 1.0 / args.rate

    while True:
        pygame.event.pump()
        # --- rumble requests from the bridge -------------------------------
        # Multi-pulse patterns are SCHEDULED, never slept through: a sleep here
        # would stall pad_state publishing and the bridge would hit its 0.5s
        # stale watchdog and failsafe to idle mid-gesture.
        while True:
            try:
                req = json.loads(rum.recv())
            except zmq.Again:
                break
            except Exception:            # noqa: BLE001 -- haptics are best-effort
                break
            st = float(req.get("strength", 0.8))
            ms = int(req.get("ms", 180))
            cnt = max(1, min(5, int(req.get("count", 1))))
            gap = max(0.05, float(req.get("gap_ms", 180)) / 1000.0)
            base = time.monotonic()
            for k in range(cnt):
                pulse_q.append((base + k * (ms / 1000.0 + gap), st, ms))
        if pulse_q:
            _now = time.monotonic()
            due = [q for q in pulse_q if q[0] <= _now]
            pulse_q[:] = [q for q in pulse_q if q[0] > _now]
            for _, st, ms in due:
                rumble(st, ms)
        # Hot-plug: BT drops are a thing. On device loss publish nothing
        # (bridge's 0.5s stale watchdog -> failsafe idle), reset ritual
        # state, and block until the pad returns.
        if pygame.joystick.get_count() == 0:
            print("[pc2-pad] pad LOST -> waiting for reconnect...", flush=True)
            hold_start, armed_until = None, 0.0
            js = wait_for_pad()
            btn_confirm, btns_disarm = face_buttons(js.get_name())
            rumble(0.6, 300)   # reconnect cue
            continue
        now = time.monotonic()
        axes = [round(js.get_axis(i), 3) for i in range(js.get_numaxes())]
        btns = [js.get_button(i) for i in range(js.get_numbuttons())]
        # Hats (D-pad). On Xbox pads via xpadneo the D-pad is a HAT, not
        # buttons, so without this it never reaches ZMQ consumers at all --
        # a D-pad binding in pad_locomotion_bridge silently never fires.
        try:
            hats = [list(js.get_hat(i)) for i in range(js.get_numhats())]
        except Exception:
            hats = []

        # 1) broadcast on change (and 5 Hz heartbeat)
        state = (axes, btns, hats)
        if state != last_state or (last_state is not None and now % 0.2 < tick):
            pub.send_multipart([b"pad_state", json.dumps(
                {"axes": axes, "buttons": btns, "hats": hats,
                 "ts": time.time()}
            ).encode()])
            last_state = state

        # 2) start ritual (disabled during post-fire cooldown)
        all_four = (btns[BTN_LB] and btns[BTN_RB]
                    and axes[AXIS_LT] > 0.5 and axes[AXIS_RT] > 0.5)
        if now >= cooldown_until:
            if now < armed_until:
                if btns[btn_confirm]:
                    print(f"[pc2-pad] *** CONFIRMED -> starting sonic daemons ***",
                          flush=True)
                    # 30s lockout, then the ritual can fire again (needed for
                    # failed-preflight retries); the ritual itself is the
                    # safety, the cooldown just prevents double-launch.
                    cooldown_until = now + 30.0
                    armed_until = 0.0
                    rumble(1.0, 800)   # ignition: long strong buzz
                    subprocess.Popen(args.start_cmd, shell=True)
                elif any(btns[i] for i in btns_disarm):
                    print("[pc2-pad] other button during window -> DISARMED",
                          flush=True)
                    rumble(0.3, 100)   # weak blip: disarmed
                    armed_until = 0.0
            elif all_four:
                if hold_start is None:
                    hold_start = now
                    hold_pulses = 0
                    print("[pc2-pad] all-four hold detected; keep holding "
                          f"{HOLD_S:.0f}s to arm...", flush=True)
                else:
                    # Haptic countdown: one pulse per held second (3 in 3s);
                    # the 3rd coincides with ARMED.
                    elapsed = now - hold_start
                    if hold_pulses < int(elapsed) and hold_pulses < HOLD_S:
                        hold_pulses += 1
                        rumble(0.6, 150)
                    if elapsed >= HOLD_S:
                        armed_until = now + WINDOW_S
                        hold_start = None
                        print(f"[pc2-pad] >>> ARMED for {WINDOW_S:.0f}s -- press "
                              f"TOP face button (Y/Triangle) to START SONIC "
                              f"(anything else disarms) <<<",
                              flush=True)
            else:
                if hold_start is not None:
                    print("[pc2-pad] hold released early -> reset", flush=True)
                hold_start = None
                hold_pulses = 0

        time.sleep(tick)


if __name__ == "__main__":
    raise SystemExit(main())
