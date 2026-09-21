#!/usr/bin/env python3
"""PC2 boot daemon: auto-pair game controllers + hand off to the sonic ritual.

Runs from robot power-on (systemd). The vendor MC boots as normal; this
daemon owns the operator-input side:

  1. Ensures the BT adapter is powered/pairable with a NoInputNoOutput agent.
  2. Reconnects already-BONDED pads first (covers mid-demo BT drops: no
     re-pairing, just press the pad's power button), then scans for NEW
     pads in pairing mode and pairs+trusts+connects them.
  3. Spawns ``pc2_pad_daemon.py`` (the L1+R1+L2+R2 hold -> Y ritual) ONCE
     and keeps it alive. It is never killed on pad loss -- the ritual
     daemon handles joystick hot-plug itself, so sonic handover state
     survives connection drops. Downstream safety on loss is the bridge's
     0.5s stale -> failsafe (robot idles).

Demo-day UX: power the robot -> walk up with a blinking pad -> it pairs
itself -> perform the ritual -> sonic takes over. Pad drops mid-demo ->
this loop reconnects it; worst case press the pad's power button.

    python3 x2_pad_autopair.py --start-cmd "<local sonic start script>"
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

# PC2 install prefix (robot-side).
_PC2_PREFIX = os.environ.get("PC2_PREFIX", "/home/run/gear-sonic")

PAD_NAME_RE = re.compile(
    r"xbox|wireless controller|dualsense|dual ?shock|gamepad|8bitdo", re.I)
MAC_RE = re.compile(r"([0-9A-F]{2}:){5}[0-9A-F]{2}", re.I)


def sh(cmd: str, timeout: int = 25) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return ""


def pad_present() -> bool:
    """A joystick-capable input device exists (BT or USB)."""
    return bool(PAD_NAME_RE.search(sh("cat /proc/bus/input/devices")))


def bt_prepare() -> None:
    sh("bluetoothctl power on")
    sh("bluetoothctl pairable on")
    sh("bluetoothctl agent NoInputNoOutput; bluetoothctl default-agent")


def bonded_pads() -> list[str]:
    """MACs of already-paired pad-like devices."""
    macs = []
    for line in sh("bluetoothctl devices Paired").splitlines():
        if PAD_NAME_RE.search(line):
            m = MAC_RE.search(line)
            if m:
                macs.append(m.group(0))
    return macs


def try_reconnect_bonded() -> bool:
    """Mid-demo drop recovery: connect known pads, no re-pairing needed."""
    for mac in bonded_pads():
        out = sh(f"bluetoothctl connect {mac}", timeout=15)
        if "successful" in out.lower():
            print(f"[autopair] reconnected bonded pad {mac}", flush=True)
            return True
    return False


def try_pair_new(scan_s: int = 12) -> bool:
    """One scan window; pair anything pad-like in pairing mode."""
    out = sh(f"timeout {scan_s + 2} bluetoothctl --timeout {scan_s} scan on",
             timeout=scan_s + 6)
    macs = {MAC_RE.search(l).group(0)
            for l in out.splitlines()
            if PAD_NAME_RE.search(l) and MAC_RE.search(l)}
    for mac in macs:
        print(f"[autopair] candidate {mac}; pairing...", flush=True)
        pair_out = sh(f"bluetoothctl pair {mac}", timeout=30)
        sh(f"bluetoothctl trust {mac}")
        conn_out = sh(f"bluetoothctl connect {mac}", timeout=20)
        p_ok = "successful" in pair_out.lower()
        c_ok = "successful" in conn_out.lower()
        print(f"[autopair] {mac}: pair={'OK' if p_ok else 'no'} "
              f"connect={'OK' if c_ok else 'no'}", flush=True)
        if c_ok:
            return True
    return False


def ensure_ritual(ritual, args):
    """Spawn the ritual daemon if not running; NEVER kill it on pad loss."""
    if ritual is not None and ritual.poll() is None:
        return ritual
    if ritual is not None:
        print(f"[autopair] ritual daemon exited rc={ritual.returncode}; "
              "respawning", flush=True)
    else:
        print("[autopair] launching ritual daemon", flush=True)
    return subprocess.Popen(
        [args.python, args.pad_daemon, "--start-cmd", args.start_cmd],
        stdout=sys.stdout, stderr=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-cmd", required=True,
                    help="LOCAL sonic start command handed to pc2_pad_daemon")
    ap.add_argument("--pad-daemon", default=os.path.join(_PC2_PREFIX, "pc2_pad_daemon.py"))
    ap.add_argument("--python", default=os.path.join(_PC2_PREFIX, "venv/bin/python"))
    args = ap.parse_args()

    print("[autopair] up; supervising pad connectivity...", flush=True)
    bt_prepare()
    ritual = None
    was_present = False
    while True:
        ritual = ensure_ritual(ritual, args)
        present = pad_present()
        if present != was_present:
            print(f"[autopair] pad {'PRESENT' if present else 'LOST'}",
                  flush=True)
            was_present = present
        if present:
            time.sleep(5)
            continue
        # pad missing: bonded reconnect first (fast path), then scan for new
        bt_prepare()
        if not try_reconnect_bonded():
            try_pair_new()
        time.sleep(3)


if __name__ == "__main__":
    raise SystemExit(main())
