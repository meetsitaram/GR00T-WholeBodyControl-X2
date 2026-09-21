#!/usr/bin/env bash
# Pad emulator for the local sim stack (gear_sonic/scripts/simstack_local.sh).
#
# This sends EXACTLY what pad_locomotion_bridge.py puts on the wire when the
# operator holds the deadman (L2+R2) and pushes the right stick sideways --
# same topic, same payload keys, same 20 Hz cadence, same zero-stick release.
# The in-place turn that produced the 2026-08-25 hip-roll burst came through
# this path, so anything else is a different experiment.
#
#   usage: simstack_turn.sh {right|left} [seconds] [magnitude]
set -euo pipefail

# Any python with pyzmq (the repo .venv has it); override with PYTHON=...
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$REPO/.venv/bin/python" ]]; then PY="$REPO/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
PORT="${SIM_CMD_PORT:-5663}"
TOPIC="${SIM_CMD_TOPIC:-planner_cmd}"

case "${1:-right}" in
  right) SIGN=1.0  ;;
  left)  SIGN=-1.0 ;;
  *) echo "usage: $0 {right|left} [seconds] [magnitude]" >&2; exit 2 ;;
esac
SECS="${2:-15}"
MAG="${3:-1.0}"

"$PY" - "$SIGN" "$SECS" "$MAG" "$PORT" "$TOPIC" <<'PY'
import json, sys, time, zmq

sign, secs, mag, port, topic = (
    float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
    int(sys.argv[4]), sys.argv[5])
yaw = round(sign * mag, 3)

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PUB)
sock.connect(f"tcp://127.0.0.1:{port}")
time.sleep(0.4)                      # PUB-connect needs a beat before the first send

def send(p):
    sock.send_multipart([topic.encode("ascii"), json.dumps(p).encode("utf-8")])

def stick(y):
    return {"intent": "locomotion", "magnitude": "continuous",
            "stick_fwd": 0.0, "stick_side": 0.0, "stick_yaw": y}

print(f"  deadman ENGAGED -> stick_yaw={yaw:+.2f} held {secs:.1f}s @20Hz", flush=True)
n, t_end = 0, time.monotonic() + secs
while time.monotonic() < t_end:
    send(stick(yaw)); n += 1
    time.sleep(0.05)
send(stick(0.0))                     # deadman released -> one zero-stick cmd -> idle
print(f"  deadman RELEASED after {n} stick frames", flush=True)
time.sleep(0.3)
PY
