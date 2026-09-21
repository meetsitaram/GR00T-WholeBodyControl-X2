#!/usr/bin/env bash
# LAPTOP half of whole-body teleop -- the "second command", as a script.
# QUEST-SPLIT ARCHITECTURE (2026-08-29): this laptop streams only human
# INTENT (SMPL joints + engage + wrist targets) to the PC2-side token
# service; no model, no robot state on this side. Wraps
# pico_intent_sender.py: picks the venv, auto-starts the XRoboToolkit PC
# Service for live sessions.
#
#   ./gear_sonic/scripts/run_pico_teleop.sh                          # SIM (no --pc2-host = local sim stack, always)
#   ./gear_sonic/scripts/run_pico_teleop.sh --pc2-host ${PC2_HOST}   # ROBOT (an ip = the real robot, always)
#   ./gear_sonic/scripts/run_pico_teleop.sh --tape-replay <npz> --auto-engage
#   ./gear_sonic/scripts/run_pico_teleop.sh --cam orbbec --headset-ip ${HEADSET_IP} --pc2-host ${PC2_HOST}
#       --cam <orbbec|stereo_left|stereo_right|head_front|sim|off>  also stream the robot's (or the
#       sim's) camera into the headset's Camera panel (press Listen there). Robot cameras need the
#       PC2 bridge running (x2_pc2_camera_zmq_publisher.py --streams <key>, see pico_vr_setup.md);
#       no --pc2-host = the sim ego view regardless of the key. --headset-ip (or $HEADSET_IP) =
#       the Pico's LAN address; otherwise taken from the headset app's own connection to the PC Service
#       (the established socket peer), polled in the background so the app may connect after launch.
#       --cam-preview also opens a window on this PC with the frame being sent (q closes it).
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PICO_PC_SERVICE_DIR="${PICO_PC_SERVICE_DIR:-/opt/apps/roboticsservice}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

TAPE_MODE=0
HAVE_HOST=0
CAM="${CAM:-off}"
HEADSET_IP="${HEADSET_IP:-}"
PASS=()
while (( $# )); do
    case "$1" in
        --model)       echo "[pico-teleop] note: --model is obsolete (encoding runs on PC2)"; shift 2 ;;
        --cam)         if [[ -n "${2:-}" && "$2" != --* ]]; then CAM="$2"; shift 2; else CAM=orbbec; shift; fi ;;   # --cam alone = orbbec
        --cam-preview) CAM_PREVIEW=1; shift ;;
        --headset-ip)  HEADSET_IP="$2"; shift 2 ;;
        --pc2-host)    HAVE_HOST=1; PASS+=("$1" "$2"); shift 2 ;;
        --tape-replay) TAPE_MODE=1; PASS+=("$1" "$2"); shift 2 ;;
        *)             PASS+=("$1"); shift ;;
    esac
done
# TARGET RULE (operator 2026-09-05): no --pc2-host = the LOCAL SIM STACK, always.
# An IP = the real robot, always. There is NO robot default and NO detection —
# "someone expected they are testing sim and instead end up sending commands
# to the real robot" is the failure this rule exists to prevent.
if (( ! HAVE_HOST )); then
    echo "[pico-teleop] no --pc2-host -> LOCAL SIM STACK (127.0.0.1). Robot = pass --pc2-host <PC2 ip> explicitly."
    if ! ss -tln 2>/dev/null | grep -q ':5573 '; then
        # the dual-head deploy binds :5573 LAST in the stack launch; if a
        # stack is coming up, wait for the port (up to 120 s), else refuse.
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'x2sim' \
           || pgrep -f "pc2_kplanner_onnx[.]py" >/dev/null 2>&1 \
           || pgrep -f "simstack_loca[l].sh" >/dev/null 2>&1; then
            echo "[pico-teleop] local sim stack detected but :5573 not bound yet -- waiting (up to 120 s) ..."
            for _ in $(seq 1 120); do
                ss -tln 2>/dev/null | grep -q ':5573 ' && break
                sleep 1
            done
        fi
        if ! ss -tln 2>/dev/null | grep -q ':5573 '; then
            echo "[pico-teleop] ERROR: no local sim stack listening on :5573 (start simstack_local.sh / sim_onnx_planner.sh --whole-body-teleop first)." >&2
            echo "[pico-teleop]        NOT falling back to the robot. For the robot: --pc2-host <PC2 ip>." >&2
            exit 2
        fi
    fi
    PASS+=(--pc2-host 127.0.0.1 --clip-port 5668)   # sim stack: kplanner motion_clip_cmd on 5668 (PC2 ritual: 5568)
else
    echo "[pico-teleop] --pc2-host given -> REAL ROBOT target"
fi
# ONE SENDER PER TARGET (2026-09-05 20:05Z robot replay): a live teleop sender
# left running while a replay sender started for the same PC2 -> the deploy
# saw engaged=1 / engaged=0 frames interleaved on :5573, its engage flag
# flapped at several Hz and the 1 s warm-up never completed (no whole-body).
# Refuse to start when another sender already targets this host.
TARGET_HOST=""; for ((i = 0; i < ${#PASS[@]}; i++)); do [[ "${PASS[$i]}" == "--pc2-host" ]] && TARGET_HOST="${PASS[$((i+1))]}"; done
if [[ -n "$TARGET_HOST" ]]; then
    for pid in $(pgrep -f "pico_intent_sender[.]py" 2>/dev/null); do
        # only a REAL sender: argv[0] is a python and argv[1] the sender script
        # (a shell whose command line merely mentions both strings must not match)
        argv1=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 2p)
        argv0=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 1p)
        [[ "$argv0" == *python* && "$argv1" == *pico_intent_sender.py ]] || continue
        if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--pc2-host $TARGET_HOST"; then
            echo "[pico-teleop] REFUSING: another sender (pid $pid, started $(ps -o lstart= -p "$pid" 2>/dev/null)) already targets $TARGET_HOST:" >&2
            echo "[pico-teleop]     $(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE '(--tape-replay [^ ]+|--pc2-host [^ ]+)' | tr '\n' ' ')" >&2
            echo "[pico-teleop]   two senders on one wire interleave engaged/disengaged frames (robot 2026-09-05 20:05Z). Ctrl-C it first." >&2
            exit 3
        fi
    done
fi
set -- ${PASS[@]+"${PASS[@]}"}

# Tape replay is headset-free (.venv); live needs the SDK (.venv_teleop)
# and the PC Service -- same split as run_x2_pico_wbc.sh.
if (( TAPE_MODE )); then
    PY="${SONIC_PYTHON:-${REPO_ROOT}/.venv/bin/python}"   # SONIC_PYTHON overrides; default the repo .venv, else python3
    command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3 || true)"
    [[ -n "$PY" ]] || { echo "ERROR: no python interpreter (create ${REPO_ROOT}/.venv or set SONIC_PYTHON=/path/to/python)" >&2; exit 1; }
else
    PY="${REPO_ROOT}/.venv_teleop/bin/python"
    if pgrep -x RoboticsService >/dev/null 2>&1; then
        echo "[pico-teleop] XRoboToolkit PC Service already running"
    else
        if [[ ! -x "${PICO_PC_SERVICE_DIR}/RoboticsServiceProcess" ]]; then
            echo "ERROR: PC Service missing (${PICO_PC_SERVICE_DIR})." >&2
            exit 1
        fi
        echo "[pico-teleop] starting PC Service -> ${LOG_DIR}/roboticsservice.log"
        LD_LIBRARY_PATH="${PICO_PC_SERVICE_DIR}:${PICO_PC_SERVICE_DIR}/lib:${PICO_PC_SERVICE_DIR}/SDK/x64:${LD_LIBRARY_PATH:-}" \
            nohup "${PICO_PC_SERVICE_DIR}/RoboticsServiceProcess" \
            >"${LOG_DIR}/roboticsservice.log" 2>&1 &
        disown
        for _ in $(seq 1 20); do
            ss -tln 2>/dev/null | grep -q ':60061 ' && break
            sleep 0.5
        done
        ss -tln 2>/dev/null | grep -q ':60061 ' \
            && echo "[pico-teleop] PC Service READY (gRPC :60061)" \
            || { echo "ERROR: PC Service did not open :60061" >&2; exit 1; }
    fi
    IP="${LAPTOP_HOST:-$(hostname -I | awk '{print $1}')}"
    echo "[pico-teleop] headset connects to: ${IP:-<laptop LAN IP>} (override with LAPTOP_HOST)"
fi

# ---- Optional camera feed into the headset (x2_pico_video_sender.py, verified 2026-09-07).
# The headset LISTENS on :12345 after Camera -> Listen; the sender CONNECTS to it. Mono streams
# go out side-by-side duplicated (the client always splits per eye). Runs beside the intent
# sender and is killed when it exits.
VIDEO_PID=""
if [[ "$CAM" != "off" ]]; then
    [[ -z "$HEADSET_IP" ]] && echo "[pico-teleop] --cam: the video sender dials the headset at the address its app connects to the PC Service from (or --headset-ip <ip>); once connected, press Listen in the Camera panel"
    VPY="${SONIC_PYTHON:-${REPO_ROOT}/.venv/bin/python}"; command -v "$VPY" >/dev/null 2>&1 || VPY="${ISAACLAB_PYTHON:-python3}"
    # Headset decode budget (2026-09-08): 960x540x2 @60 fps queued 3-4 s inside the client decoder;
    # 640x480x2 @30 fps / 4 Mbit (the sim session) was live. Tune with CAM_FPS / CAM_KBPS / CAM_W / CAM_H.
    VARGS=(--mode connect --port 12345 --fps "${CAM_FPS:-30}" --bitrate-kbps "${CAM_KBPS:-4000}" --sbs-duplicate)
    (( ${CAM_PREVIEW:-0} )) && VARGS+=(--preview)   # --cam-preview: the same frame in a window on this PC
    (( ${CAM_STAMP:-0} )) && VARGS+=(--stamp)       # CAM_STAMP=1: burn the laptop clock into the frame (latency ruler)
    if (( ! HAVE_HOST )); then
        VARGS+=(--source sim --state-port "${CAM_STATE_PORT:-5659}" --width "${CAM_W:-640}" --height "${CAM_H:-480}")
        echo "[pico-teleop] camera: SIM ego view -> headset ${HEADSET_IP:-<auto>}:12345"
    else
        # udp-cam (2026-09-08): chunked datagrams from the camera callback, newest complete frame wins, nothing
        # queues (the TCP/ZMQ path built 2-3 s of backlog on wifi). :5555 stays the "bridge is up" probe.
        VARGS+=(--source udp-cam --cam-host "$TARGET_HOST" --cam-port 5556 --keys "$CAM")
        case "$CAM" in
            orbbec) VARGS+=(--rotate180 --width "${CAM_W:-640}" --height "${CAM_H:-360}") ;;   # inverted mount, 16:9
            *)      VARGS+=(--width "${CAM_W:-640}" --height "${CAM_H:-480}") ;;
        esac
        if ! timeout 3 bash -c "</dev/tcp/$TARGET_HOST/5555" 2>/dev/null; then
            echo "[pico-teleop] WARNING: no camera bridge on $TARGET_HOST:5555 -- start it on PC2:" >&2
            echo "[pico-teleop]   PYTHONPATH=/usr/local/lib/python3.10/dist-packages:\$PYTHONPATH python3 gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py --port 5555 --streams $CAM --width 1280 --height 720 --jpeg-quality 90 --publish-rate 20" >&2
        fi
        echo "[pico-teleop] camera: PC2 '$CAM' -> headset ${HEADSET_IP:-<auto>}:12345"
    fi
    # The video sender starts NOW (preview from the first robot frame); it dials the headset itself,
    # deterministically: --host auto = the peer of the established TCP connection the XRoboToolkit
    # PC Service (RoboticsServiceProcess) holds from the headset app, re-resolved every 2 s until the
    # app connects. --headset-ip overrides. No scanning, no ARP guessing. Teleop never waits for this.
    VPIDF="${LOG_DIR}/video_sender.pid"; rm -f "$VPIDF"
    "$VPY" -m gear_sonic.scripts.x2_pico_video_sender "${VARGS[@]}" --host "${HEADSET_IP:-auto}" > "${LOG_DIR}/video_sender.log" 2>&1 &
    VIDEO_PID=$!; echo $VIDEO_PID > "$VPIDF"
    echo "[pico-teleop] video sender pid $VIDEO_PID (log ${LOG_DIR}/video_sender.log); streams once the headset app connects to the PC Service (then press Listen in the Camera panel)"
    # Ctrl-C: a background child ignores SIGINT (bash), so kill the video sender BY PID ourselves,
    # on INT/TERM and on exit (the foreground intent sender gets the Ctrl-C directly).
    stop_video() { kill "$VIDEO_PID" 2>/dev/null; rm -f "$VPIDF"; }
    trap 'stop_video' INT TERM
    trap 'stop_video' EXIT
fi

echo "[pico-teleop] intent sender starting (streams to the PC2 token service)"
if [[ -n "$VIDEO_PID" ]]; then
    "$PY" "${REPO_ROOT}/gear_sonic/scripts/pico_intent_sender.py" "$@"
else
    exec "$PY" "${REPO_ROOT}/gear_sonic/scripts/pico_intent_sender.py" "$@"
fi
