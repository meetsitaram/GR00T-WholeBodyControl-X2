#!/usr/bin/env python3
"""Standalone Pico face-button / trigger probe. NO robot, NO teleop — just
reads the XRoboToolkit SDK and prints the raw button state whenever it
changes, so we can verify the physical-label -> SDK-getter mapping and
catch phantom multi-presses.

    .venv_teleop/bin/python gear_sonic/scripts/pico_button_probe.py

Press ONE physical button at a time and note what prints. Ctrl-C to stop.
"""
import time
import xrobotoolkit_sdk as xrt

xrt.init()
print("probe live. press one button at a time; watch the raw states.",
      flush=True)
prev = None
try:
    while True:
        raw = {
            "A": bool(xrt.get_A_button()),
            "B": bool(xrt.get_B_button()),
            "X": bool(xrt.get_X_button()),
            "Y": bool(xrt.get_Y_button()),
            "LT": round(float(xrt.get_left_trigger()), 2),
            "RT": round(float(xrt.get_right_trigger()), 2),
            "LG": round(float(xrt.get_left_grip()), 2),
            "RG": round(float(xrt.get_right_grip()), 2),
        }
        if raw != prev:
            on = [k for k in ("A", "B", "X", "Y") if raw[k]]
            an = {k: raw[k] for k in ("LT", "RT", "LG", "RG") if raw[k] > 0.1}
            print(f"[{time.strftime('%H:%M:%S')}] buttons={on or '-'}  "
                  f"analog={an or '-'}", flush=True)
            prev = raw
        time.sleep(0.02)
except KeyboardInterrupt:
    print("\nprobe stopped.", flush=True)
