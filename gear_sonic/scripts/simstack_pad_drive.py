#!/usr/bin/env python3
"""Scripted gamepad for the local sim stack: sends exactly what
pad_locomotion_bridge.py puts on the planner_cmd wire (same topic, payload
keys and 20 Hz cadence), so a walk or a crouch walk can be reproduced
without touching the pad.

  simstack_pad_drive.py walk       [secs] [fwd]   deadman + left stick forward
  simstack_pad_drive.py crouchwalk [secs] [fwd]   L2+R2 + D-pad DOWN (crouch_mode
                                                  keepalives), then forward, then release
  simstack_pad_drive.py idle                      one zero-stick frame

The planner must OWN the command socket (SUB bind), which is the case for
`sim_onnx_planner.sh --vr` (pad-and-VR) and simstack_local.sh; in the
pad-only stack the pad bridge binds the PUB and nothing else can inject.
Port: SIM_CMD_PORT (default 5563, the stack's planner_cmd port; 5663 for the
simstack_local map), topic SIM_CMD_TOPIC (planner_cmd).
"""
import json
import os
import sys
import time

import zmq

PORT = int(os.environ.get("SIM_CMD_PORT") or 5563)
TOPIC = os.environ.get("SIM_CMD_TOPIC") or "planner_cmd"
mode = sys.argv[1] if len(sys.argv) > 1 else "walk"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
fwd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6

sock = zmq.Context.instance().socket(zmq.PUB)
sock.connect(f"tcp://127.0.0.1:{PORT}")
time.sleep(0.4)                      # PUB-connect needs a beat before the first send


def send(p):
    sock.send_multipart([TOPIC.encode("ascii"), json.dumps(p).encode("utf-8")])


def stick(f, y=0.0):
    return {"intent": "locomotion", "magnitude": "continuous",
            "stick_fwd": f, "stick_side": 0.0, "stick_yaw": y}


def crouch(on):
    return {"intent": "crouch_mode", "enable": bool(on), "magnitude": "default"}


if mode == "idle":
    send(stick(0.0)); print("idle sent"); sys.exit(0)

t0 = time.monotonic()
if mode == "crouchwalk":
    # arm: both triggers + D-pad DOWN while standing still; the bridge re-sends
    # enable=true at 5 Hz and the kplanner gates the entry on a still robot
    print("CROUCH arm: keepalives 2.0 s", flush=True)
    ka = 0.0
    while time.monotonic() - t0 < 2.0:
        if time.monotonic() - ka >= 0.2:
            send(crouch(True)); ka = time.monotonic()
        time.sleep(0.02)

print(f"deadman ENGAGED -> stick_fwd={fwd:+.2f} for {secs:.1f}s @20Hz", flush=True)
t1 = time.monotonic(); n = 0; ka = 0.0
while time.monotonic() - t1 < secs:
    if mode == "crouchwalk" and time.monotonic() - ka >= 0.2:
        send(crouch(True)); ka = time.monotonic()
    send(stick(fwd)); n += 1
    time.sleep(0.05)
send(stick(0.0))                     # deadman released -> one zero-stick cmd -> stop
print(f"deadman RELEASED after {n} frames", flush=True)
if mode == "crouchwalk":
    time.sleep(0.5)
    send(crouch(False)); print("CROUCH off (trigger released)", flush=True)
time.sleep(0.3)
