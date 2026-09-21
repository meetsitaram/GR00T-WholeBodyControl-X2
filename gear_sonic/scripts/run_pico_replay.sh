#!/usr/bin/env bash
# Replay a captured Pico SMPL recording (raw body tape, session_*_x2_segNN.npz) onto the
# SIM stack or the ROBOT through the exact live-teleop path (sender -> SMPL stream -> dual
# deploy), ONCE, with whole-body released at the end of the tape. Operator 2026-09-05:
# "lets make a proper plan for replaying captured smpl motions on to the robot" (same-input
# A/B of checkpoints, e.g. the waist snap-back on 39k vs 41k; demo playback of captured
# manipulation motions incl. the OmniHand grips, which the tape carries).
#
#   ./gear_sonic/scripts/run_pico_replay.sh <tape.npz>                              # SIM: no --pc2-host = local sim stack, always
#   ./gear_sonic/scripts/run_pico_replay.sh <tape.npz> --pc2-host <PC2_IP>          # ROBOT: an ip = the real robot, always
#   ./gear_sonic/scripts/run_pico_replay.sh <tape.npz> --loop                       # loop (sim-style; never on the robot: wrap snaps the pose)
#   ./gear_sonic/scripts/run_pico_replay.sh <tape.npz> -- --hand-close-on-grip      # extra sender args after --
#
# TARGET RULE (operator 2026-09-05): no --pc2-host = sim, an ip = the robot, no robot
# default, no detection — same rule as run_pico_teleop.sh. Robot mode prints the checklist
# and asks for a typed REPLAY confirmation; the robot moves by itself once the tape engages. The sender keeps streaming idle after the tape (the deploy needs a
# continuous stream); Ctrl-C ends the session. The pad e-stop is the only e-stop during a
# replay: a tape has no controller chords.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAPE=""; HOST=""; LOOP=0; ENGAGE=1; EXTRA=()
while (( $# )); do
    case "$1" in
        --pc2-host) HOST="${2:-}"; shift 2 ;;
        --loop)  LOOP=1; shift ;;
        --no-engage) ENGAGE=0; shift ;;
        --)      shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n 2,19p "$0"; exit 0 ;;
        *)       if [[ -z "$TAPE" ]]; then TAPE="$1"; shift; else echo "unknown arg $1 (extra sender args go after --)" >&2; exit 2; fi ;;
    esac
done
[[ -n "$TAPE" && -f "$TAPE" ]] || { echo "usage: run_pico_replay.sh <tape.npz> [--pc2-host <robot ip>] [--loop] [--no-engage] [-- sender args]" >&2; exit 2; }
if [[ -z "$HOST" ]]; then TARGET=sim; else TARGET=robot; fi
# ONE SENDER PER TARGET, checked BEFORE the checklist/prompt (run_pico_teleop.sh
# checks again): a live teleop sender left running next to a replay sender
# interleaves engaged/disengaged frames on the wire (robot 2026-09-05 20:05Z).
CHK_HOST="${HOST:-127.0.0.1}"
for pid in $(pgrep -f "pico_intent_sender[.]py" 2>/dev/null); do
    argv0=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 1p); argv1=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 2p)
    [[ "$argv0" == *python* && "$argv1" == *pico_intent_sender.py ]] || continue     # real senders only
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--pc2-host $CHK_HOST"; then
        echo "[replay] REFUSING: a sender already targets $CHK_HOST — pid $pid, started $(ps -o lstart= -p "$pid" 2>/dev/null):" >&2
        echo "[replay]     $(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE '(--tape-replay [^ ]+|--pc2-host [^ ]+)' | tr '\n' ' ')" >&2
        echo "[replay]   teleop and replay are exclusive on a target: Ctrl-C that sender first." >&2
        exit 5
    fi
done
PY="${SONIC_PYTHON:-${REPO_ROOT}/.venv/bin/python}"   # SONIC_PYTHON overrides; default the repo .venv, else python3
command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3 || true)"
[[ -n "$PY" ]] || { echo "ERROR: no python interpreter (create ${REPO_ROOT}/.venv or set SONIC_PYTHON=/path/to/python)" >&2; exit 1; }
INFO=$("$PY" - "$TAPE" <<'PYEOF'
import sys, numpy as np, datetime as dt
z = np.load(sys.argv[1])
if "smpl_joints" in z.files and "engaged" in z.files:
    # INTENT tape (pico_intent_sender.py --record output): the sender replays it directly, engaged flags
    # verbatim (2026-09-08: same-tape A/B of checkpoints on yesterday's robot session tapes)
    t = z["t"]; dur = float(t[-1] - t[0]); e = z["engaged"] > 0.5
    segs = int(np.sum(np.diff(np.r_[0, e.astype(int)]) == 1))
    print(f"tape {sys.argv[1].split('/')[-1]}: INTENT tape ({segs} engage segments replayed verbatim, engaged {e.mean()*100:.0f} %); "
          f"{len(t)} frames, {dur:.1f} s at {len(t)/max(dur,1e-6):.0f} Hz")
    if dur < 1.0: print("BAD tape: shorter than 1 s"); sys.exit(3)
    sys.exit(0)
need = ("stamp_ns", "body", "body_ok", "analog")
missing = [k for k in need if k not in z.files]
if missing: print(f"BAD tape: missing {missing} (expected a session_*_x2_segNN.npz raw body tape or an intent_*.npz tape)"); sys.exit(3)
s = z["stamp_ns"].astype(np.int64); dur = (s[-1] - s[0]) / 1e9; g = z["analog"][:, 2:4]
sq = lambda x: int(np.sum(np.diff((x > 0.5).astype(int)) == 1))
full = "full_session" in z.files
if full:
    t0 = dt.datetime.utcfromtimestamp(float(z["t0_wall_unix_s"])).strftime("%Y-%m-%d %H:%M:%SZ") if "t0_wall_unix_s" in z.files and float(z["t0_wall_unix_s"]) > 0 else "?"
    btn = z["buttons"]; presses = lambda j: int(np.sum(np.diff(btn[:, j].astype(int)) == 1))
    kind = f"FULL SESSION (starts in {z['mode0']}; B presses {presses(1)}, chords ~{presses(0)}, deadman pulls {sq(z['analog'][:,0])})"
else:
    t0 = dt.datetime.utcfromtimestamp(s[0] / 1e9).strftime("%Y-%m-%d %H:%M:%SZ")
    kind = "single engage window (body + grips only; replays as B-in .. B-out)"
print(f"tape {sys.argv[1].split('/')[-1]}: {kind}; {len(s)} frames, {dur:.1f} s at {len(s)/max(dur,1e-6):.0f} Hz, recorded {t0}, body_ok {z['body_ok'].mean():.2f}, grips L {sq(g[:,0])} / R {sq(g[:,1])} squeezes")
if dur < 1.0: print("BAD tape: shorter than 1 s"); sys.exit(3)
PYEOF
) || { echo "$INFO" >&2; exit 3; }
echo "[replay] $INFO"
KIND=$("$PY" -c "import numpy as np,sys; f=np.load(sys.argv[1]).files; print('FULL' if 'full_session' in f else ('INTENT' if 'smpl_joints' in f else 'WINDOW'))" "$TAPE")
ARGS=(--tape-replay "$TAPE")
(( LOOP )) || ARGS+=(--tape-once)
# a single engage-window clip needs --auto-engage (it has no B on it); a full
# session tape engages with its own recorded B presses
if [[ "$KIND" == WINDOW ]]; then (( ENGAGE )) && ARGS+=(--auto-engage); fi
if [[ "$TARGET" == sim ]]; then
    ss -tln 2>/dev/null | grep -q ':5573 ' || { echo "[replay] no --pc2-host = SIM, but no local sim stack on :5573 — start simstack_local.sh / sim_onnx_planner.sh --whole-body-teleop first (robot = --pc2-host <ip>)" >&2; exit 4; }
    echo "[replay] target: LOCAL SIM STACK (no --pc2-host given)"
else
    ARGS+=(--pc2-host "$HOST")
    echo "[replay] target: REAL ROBOT PC2 $HOST"
    ssh -o ConnectTimeout=5 -o BatchMode=yes "run@$HOST" 'true' 2>/dev/null || echo "[replay] WARNING: $HOST not reachable over ssh right now"
    cat <<TXT
[replay] ROBOT CHECKLIST — the robot will move on its own once the tape engages:
         1. deploy ritual up, "Loaded ONNX ..." line read (which checkpoint?), kplanner idle-standing
         2. clear floor around the robot for the whole tape (${INFO#*: })
         3. pad in hand: LB+RB stop / e-stop chord ready — the tape carries NO controller chords
         4. whole-body must be DISENGAGED before starting (the replay engages itself)
         5. at the end of the tape the sender releases whole-body -> deploy hold -> kplanner return; Ctrl-C afterwards
TXT
    (( LOOP )) && echo "[replay] WARNING: --loop on the robot: the wrap snaps the human pose to frame 0 in one tick"
    read -r -p "[replay] type replay to start (anything else aborts): " ans
    [[ "${ans,,}" == "replay" ]] || { echo "[replay] aborted"; exit 0; }
    for i in 3 2 1; do echo "[replay] engaging in $i ..."; sleep 1; done
fi
LOGF="${REPO_ROOT}/logs/run_pico_replay_$(date -u +%Y%m%dT%H%M%SZ)_${TARGET}.log"; mkdir -p "${REPO_ROOT}/logs"
echo "[replay] log: $LOGF"
echo "[replay] $(date -u +%FT%TZ) tape=$TAPE target=$TARGET loop=$LOOP engage=$ENGAGE args=${ARGS[*]} ${EXTRA[*]}" | tee "$LOGF"
exec "${REPO_ROOT}/gear_sonic/scripts/run_pico_teleop.sh" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$LOGF"
