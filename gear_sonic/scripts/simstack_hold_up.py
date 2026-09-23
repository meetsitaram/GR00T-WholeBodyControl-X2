#!/usr/bin/env python3
"""Sim-only: play the helper in the SAFE_HOLD -> RECOVER rehearsal.

Drives the MuJoCo bridge's elastic band through its scene_reset hook (the
bridge SUB-connects to localhost:5560, CONFLATE), which stands in for a
person holding the fallen robot up. Pelvis target height = 1.0 m + length.

  simstack_hold_up.py on  [length_m]   # hold up (e.g. -0.45 = pelvis 0.55 m, knees bent)
  simstack_hold_up.py off              # let go

Rehearsal: launch simstack_local.sh with the band available but released and
a scheduled shove, e.g.
  SIMSTACK_SIM_FLAGS="--sim-init-pose stand-settled --sim-band-release-after-s 0" \\
  SIM_BRIDGE_EXTRA="--tilt-torque 200 --tilt-start 30 --tilt-ramp 1.5 --tilt-hold 3 --tilt-release 0.1 --tilt-rest 100000" \\
  ./gear_sonic/scripts/simstack_local.sh
then after the tilt trip: `on -0.45`, watch deploy.log for RECOVER, `on -0.33`
(ease to stand height), `off`. See docs/x2/F10_real_robot_colcon.md,
"Fall recovery". The band's orientation PD spins a fallen body upright,
which a person would not do; that spin is a sim artifact.
"""
import sys
import time

import zmq

sys.path.insert(0, ".")
from gear_sonic.utils.teleop.zmq.scene_state_zmq import (  # noqa: E402
    SCENE_RESET_TOPIC, pack_json)

on = (sys.argv[1] if len(sys.argv) > 1 else "on") == "on"
payload = {"band_enable": on}
if len(sys.argv) > 2:
    payload["band_length"] = float(sys.argv[2])
sock = zmq.Context.instance().socket(zmq.PUB)
sock.bind("tcp://*:5560")
time.sleep(0.8)                      # let the CONFLATE SUB reconnect
for _ in range(3):
    sock.send(pack_json(SCENE_RESET_TOPIC, payload))
    time.sleep(0.2)
print("sent", payload)
time.sleep(0.3)
