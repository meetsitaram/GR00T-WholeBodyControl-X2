#!/usr/bin/env bash
# X2 direct PKL-to-SONIC stack runner.
#
# Sister wrapper of ``run_x2_quest3_planner_stack.sh``: same single-shell
# ergonomics, but the foreground client is the pose merger running in
# subscribe-mode with NO upstream planner / manager. The operator
# triggers motion-clip playback from another terminal via
# :file:`gear_sonic/scripts/play_xbox_controller.py` (gamepad chords ->
# demo-bank clips on the ``motion_clip_cmd`` wire). To drive a PKL
# through the kplanner instead, use run_x2_quest3_planner_stack.sh with
# :file:`gear_sonic/scripts/x2_pkl_command_source.py` as the
# ``planner_cmd`` source.
#
#     teleop          run_x2_quest3_planner_stack.sh
#     pkl-via-kplanner  run_x2_quest3_planner_stack.sh + x2_pkl_command_source.py
#     direct PKL      run_x2_pkl_direct_stack.sh   (this file)
#
# Lightweight topology (no manager, no planner, no policy, no parquet):
#
#                       +----------------------------+
#                       |  play_xbox_controller.py    |  PUBs motion_clip_cmd
#                       |   (operator, another shell) |  -> merger takeover
#                       +----------+------------------+
#                                  | motion_clip_cmd (5568)
#                                  v
#                       +-----------------------------+
#                       |   x2_pose_merger.py        |  subscribe mode, no
#                       |   (this script's job)      |  body_pose upstream
#                       |                            |  -> idle stand on
#                       |                            |     no-clip ticks;
#                       |                            |     PKL frames during
#                       |                            |     a play.
#                       +-----------+----------------+
#                                   | pose (5556)
#                                   v
#                       +-----------------------------+
#                       |    deploy_x2.sh sim         |  SONIC + MuJoCo,
#                       |        (optional)           |  or assume PC2 deploy
#                       +-----------------------------+
#
# The recorder publishes the v5-promoted ``pose`` envelope (current
# joint_pos_mj + 9-slot future window) directly to ``:5556``. The
# deploy is spawned with ``--disable-pose-ref-watchdog`` for the
# local-sim case (no kplanner upstream, so the cold-start gap before the first idle-stand
# tick would otherwise trip the 0.5 s SAFE_IDLE watchdog).
#
# Three modes:
#
#   * Sim (default)        - spawns ``deploy_x2.sh sim --vla`` on
#                            localhost, then runs the recorder against it.
#                            Safe place to validate the motion-clip wire
#                            end-to-end before powering the real robot.
#
#   * Real robot           - pass ``--pc2-host <PC2_IP>``. Skips spawning a
#                            sim deploy; assumes ``x2_pc2_daemons.sh start``
#                            is already running on PC2. The recorder's PUB
#                            binds on the LAN-visible interface and PC2's
#                            pose proxy connects out to it.
#
#   * External deploy      - pass ``--no-deploy``. Like real-robot mode but
#                            without the PC2_HOST annotation; useful when
#                            you brought up your own deploy in another shell
#                            and just want this wrapper to manage the
#                            recorder's lifecycle.
#
# Usage:
#   gear_sonic/scripts/run_x2_pkl_direct_stack.sh
#       [--with-deploy | --no-deploy | --pc2-host HOST]
#       [--no-sim-viewer] [--sim-profile {handoff,parity,manual}]
#       [--model PATH]
#       [--rate HZ]
#       [--gesture-catalog PATH]
#       [--motion-clip-port PORT] [--motion-clip-topic NAME]
#       [--no-idle-publish]
#       [--duration N]
#       [--cleanup-only]
#       [--log-dir PATH]
#
# Examples:
#   # Sim smoke: bring up sim deploy + recorder. Then from another
#   # terminal:
#   #   python -m gear_sonic.scripts.play_xbox_controller --list   # chord -> clip table
#   #   python -m gear_sonic.scripts.play_xbox_controller
#   ./run_x2_pkl_direct_stack.sh
#
#   # Real robot, PC2 daemons already running:
#   ./run_x2_pkl_direct_stack.sh --pc2-host ${PC2_HOST}
#
#   # External deploy you brought up elsewhere; just manage the recorder:
#   ./run_x2_pkl_direct_stack.sh --no-deploy
#
#   # Free ports + kill orphan deploy/recorder processes from a crashed run.
#   ./run_x2_pkl_direct_stack.sh --cleanup-only

set -u
set -o pipefail

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_SH="${REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh"
# Stack interpreter: SONIC_PYTHON overrides; default the repo .venv, else python3.
PYTHON="${SONIC_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: no python interpreter found (create ${REPO_ROOT}/.venv or set SONIC_PYTHON=/path/to/python)" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Port + topic contract. Mirrors run_x2_quest3_planner_stack.sh so the
# C++ deploy needs no special configuration to talk to this stack.
# --------------------------------------------------------------------------

POSE_PORT=5556          # recorder PUB (pose topic) -> deploy SUB
POSE_TOPIC="pose"
DEBUG_PORT=5557         # deploy PUB -> recorder SUB (x2_debug)
DEBUG_TOPIC="x2_debug"
# Recorder body_pose / arm_and_hands SUBs are wired to ports that
# nothing publishes on in this stack -- they just sit empty and the
# recorder's idle-stand fallback (RecorderConfig.idle_publish_enabled
# = True) keeps :5556 fed. We give them their nominal port numbers so
# a stray kplanner / VR bridge running on the same host doesn't
# accidentally drive the recorder.
BODY_POSE_PORT=5560     # nominal kplanner body_pose PUB port (unused here)
BODY_POSE_TOPIC="body_pose"
ARM_HANDS_PORT=5562     # nominal manager arm_and_hands port (unused here)
MOTION_CLIP_CMD_PORT=5568
MOTION_CLIP_CMD_TOPIC="motion_clip_cmd"

# --------------------------------------------------------------------------
# CLI defaults
# --------------------------------------------------------------------------

RATE=50

WITH_DEPLOY=1
SIM_VIEWER=1
# ``handoff``: bridge starts
# at DEFAULT_DOF (matches real-robot MC handoff), elastic band stays
# on through soft-start ramp so the body can't tip while the
# recorder's idle-stand fallback warms up. ``parity`` would 502 here
# because it requires ``--motion`` for RSI; this stack doesn't bake
# an anchor PKL up-front.
SIM_PROFILE="handoff"
# Canonical SONIC model cache (multi-embodiment): $SONIC_HOME (default
# ~/.cache/sonic) with X2 artifacts under $SONIC_HOME/x2; the X2 subtree
# alone is overridable via $SONIC_X2_MODELS. Populated by
# install_scripts/setup_x2.sh (download_from_hf.py --robot x2).
SONIC_X2_MODELS_DIR="${SONIC_X2_MODELS:-${SONIC_HOME:-$HOME/.cache/sonic}/x2}"
# SIM_MODEL default resolution order:
#   --model CLI arg > MODEL env > X2_PLANNER_SMOKE_MODEL env > model cache
#   ($SONIC_X2_MODELS > $SONIC_HOME/x2). No machine-local fallback: with
#   none of these set the deploy preflight fails with a clear message
#   (bring your own X2 SONIC ONNX -- see MODELS.md).
if [[ -n "${MODEL:-}" ]]; then
    SIM_MODEL="${MODEL}"
elif [[ -n "${X2_PLANNER_SMOKE_MODEL:-}" ]]; then
    SIM_MODEL="${X2_PLANNER_SMOKE_MODEL}"
elif [[ -f "${SONIC_X2_MODELS_DIR}/sonic_policy/x2_sonic_policy.onnx" ]]; then
    SIM_MODEL="${SONIC_X2_MODELS_DIR}/sonic_policy/x2_sonic_policy.onnx"
else
    SIM_MODEL=""
fi
SIM_CAM_TRACK_BODY="pelvis"
SIM_CAM_DISTANCE="3.5"
SIM_CAM_ELEVATION="-12"
SIM_CAM_AZIMUTH="135"
PC2_HOST=""

GESTURE_CATALOG="${REPO_ROOT}/gear_sonic/data/motions/gestures/gestures_v1.yaml"
NO_IDLE_PUBLISH=0

DURATION_S=0            # 0 = unlimited (run until Ctrl-C)
LOG_DIR=""
CLEANUP_ONLY=0

usage() {
    awk '/^# Usage:/,/^[^#]/{ if ($0 ~ /^[^#]/) exit; sub(/^# ?/, ""); print }' "$0" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rate) RATE="$2"; shift 2 ;;
        --with-deploy) WITH_DEPLOY=1; PC2_HOST=""; shift ;;
        --no-deploy) WITH_DEPLOY=0; PC2_HOST=""; shift ;;
        --pc2-host) PC2_HOST="$2"; WITH_DEPLOY=0; shift 2 ;;
        --no-sim-viewer) SIM_VIEWER=0; shift ;;
        --sim-viewer) SIM_VIEWER=1; shift ;;
        --sim-profile) SIM_PROFILE="$2"; shift 2 ;;
        --model) SIM_MODEL="$2"; shift 2 ;;
        --gesture-catalog) GESTURE_CATALOG="$2"; shift 2 ;;
        --motion-clip-port) MOTION_CLIP_CMD_PORT="$2"; shift 2 ;;
        --motion-clip-topic) MOTION_CLIP_CMD_TOPIC="$2"; shift 2 ;;
        --no-idle-publish) NO_IDLE_PUBLISH=1; shift ;;
        --duration) DURATION_S="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --cleanup-only) CLEANUP_ONLY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="/tmp/x2_pkl_direct_stack-$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# LAN isolation for the recorder pose PUB.
#
# PC2's x2_pose_proxy is a long-lived daemon that SUBs
# ``<LAPTOP_IP>:${POSE_PORT}`` over wifi. Binding ``*`` on the
# recorder's pose PUB would silently send the wire to PC2 even in
# sim mode. Gate on PC2_HOST: with --pc2-host bind '*', without it
# bind loopback so the wire is unreachable from PC2. Override-able
# via ``PUB_BIND_HOST=*`` env for the rare cross-host sim-mode debug.
if [[ -n "${PC2_HOST}" ]]; then
    : "${PUB_BIND_HOST:=*}"
else
    : "${PUB_BIND_HOST:=127.0.0.1}"
fi

# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_RESET=$'\e[0m'
log()  { printf '%s[pkl-direct-stack %s]%s %s\n' "${C_GREEN}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
warn() { printf '%s[pkl-direct-stack %s WARN]%s %s\n' "${C_YELLOW}" "$(date +%H:%M:%S)" "${C_RESET}" "$*"; }
err()  { printf '%s[pkl-direct-stack %s ERROR]%s %s\n' "${C_RED}" "$(date +%H:%M:%S)" "${C_RESET}" "$*" >&2; }

# --------------------------------------------------------------------------
# Process / port helpers (shared verbatim with run_x2_quest3_planner_stack.sh
# so behaviour stays in lockstep across the wrapper family).
# --------------------------------------------------------------------------

port_in_use() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nPi ":${port}" >/dev/null 2>&1
        return $?
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -n tcp "${port}" >/dev/null 2>&1
        return $?
    fi
    "${PYTHON}" - <<EOF
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", ${port}))
    sys.exit(1)
except OSError:
    sys.exit(0)
finally:
    s.close()
EOF
}

kill_pid_quiet() {
    local pid="$1"
    [[ -z "$pid" ]] && return 0
    kill -0 "$pid" 2>/dev/null || return 0
    local label="${2:-pid $pid}"
    log "  killing ${label} (pid=${pid})"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    warn "  force-killing ${label} (pid=${pid})"
    kill -KILL "$pid" 2>/dev/null || true
}

free_port() {
    local port="$1"
    if ! port_in_use "${port}"; then
        return 0
    fi
    log "freeing port ${port}..."
    if command -v fuser >/dev/null 2>&1; then
        fuser -k -TERM -n tcp "${port}" 2>/dev/null || true
        sleep 0.5
        fuser -k -KILL -n tcp "${port}" 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
        local pids
        pids="$(lsof -nPiTCP:"${port}" -sTCP:LISTEN -t || true)"
        for p in ${pids}; do kill_pid_quiet "${p}" "stale on :${port}"; done
    fi
}

stop_deploy_container() {
    local deploy_log="$1"
    [[ -z "${deploy_log}" || ! -f "${deploy_log}" ]] && return 0
    if ! command -v docker >/dev/null 2>&1; then
        return 0
    fi
    local deploy_container
    deploy_container="$(grep -oE 'docker_x2-x2sim-run-[a-f0-9]+' "${deploy_log}" 2>/dev/null | tail -1 || true)"
    if [[ -z "${deploy_container}" ]]; then
        return 0
    fi
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${deploy_container}"; then
        return 0
    fi
    log "  docker stop ${deploy_container} (--timeout 5)"
    docker stop --timeout 5 "${deploy_container}" >/dev/null 2>&1 || true
}

preflight_docker_cleanup() {
    if ! command -v docker >/dev/null 2>&1; then
        return 0
    fi
    local stale
    stale="$(
        {
            docker ps --filter name=x2sim-run --format '{{.Names}}' 2>/dev/null
            docker ps --filter ancestor=gr00t-x2sim --format '{{.Names}}' 2>/dev/null
            docker ps --filter ancestor=x2sim --format '{{.Names}}' 2>/dev/null
        } | sort -u
    )"
    if [[ -z "${stale}" ]]; then
        return 0
    fi
    warn "stale x2sim container(s) detected from previous run; stopping:"
    while IFS= read -r c; do
        [[ -z "$c" ]] && continue
        warn "  - ${c}"
        docker stop --timeout 5 "${c}" >/dev/null 2>&1 || true
    done <<< "${stale}"
}

cleanup_stale() {
    log "cleanup_stale: ports=${POSE_PORT},${MOTION_CLIP_CMD_PORT}"
    free_port "${POSE_PORT}"
    free_port "${MOTION_CLIP_CMD_PORT}"
    preflight_docker_cleanup
}

wait_for_log_marker() {
    local log_path="$1"
    local pid="$2"
    local marker="$3"
    local timeout_s="${4:-60}"
    local label="${5:-process}"
    local start_ts
    start_ts=$(date +%s)
    while :; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            err "${label} died during bring-up (pid=${pid}); tail of log:"
            tail -n 40 "${log_path}" >&2 || true
            return 1
        fi
        if [[ -f "${log_path}" ]] && grep -F -q "${marker}" "${log_path}" 2>/dev/null; then
            return 0
        fi
        local now elapsed
        now=$(date +%s)
        elapsed=$((now - start_ts))
        if (( elapsed > timeout_s )); then
            err "${label} did not log '${marker}' within ${timeout_s}s; tail:"
            tail -n 40 "${log_path}" >&2 || true
            return 1
        fi
        sleep 0.5
    done
}

# --------------------------------------------------------------------------
# Cleanup trap (set BEFORE any child is spawned). Reverse-order shutdown:
# recorder first (so any in-flight motion-clip session gets the SIGINT
# and the publish loop can drain cleanly), THEN deploy.
# --------------------------------------------------------------------------

DEPLOY_PID=""
DEPLOY_LOG=""
RECORDER_PID=""
RECORDER_PGID=""
RECORDER_LOG=""
RESET_WATCHER_PID=""

cleanup_children() {
    log "shutting down children (reverse spawn order)..."
    if [[ -n "${RESET_WATCHER_PID}" ]] && kill -0 "${RESET_WATCHER_PID}" 2>/dev/null; then
        kill -TERM "${RESET_WATCHER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${RECORDER_PID}" ]] && kill -0 "${RECORDER_PID}" 2>/dev/null; then
        log "  SIGINT recorder (pid=${RECORDER_PID}, pgid=${RECORDER_PGID:-?})"
        if [[ -n "${RECORDER_PGID}" ]]; then
            kill -INT -- "-${RECORDER_PGID}" 2>/dev/null || true
        else
            kill -INT "${RECORDER_PID}" 2>/dev/null || true
        fi
        local i=0
        while (( i < 10 )); do
            kill -0 "${RECORDER_PID}" 2>/dev/null || break
            sleep 0.5
            i=$((i + 1))
        done
        kill_pid_quiet "${RECORDER_PID}" "recorder"
    fi
    if [[ -n "${DEPLOY_PID}" ]] && kill -0 "${DEPLOY_PID}" 2>/dev/null; then
        log "  SIGINT deploy host-side bash (pid=${DEPLOY_PID})"
        kill -INT "${DEPLOY_PID}" 2>/dev/null || true
        local i=0
        while (( i < 10 )); do
            kill -0 "${DEPLOY_PID}" 2>/dev/null || break
            sleep 0.5
            i=$((i + 1))
        done
        kill_pid_quiet "${DEPLOY_PID}" "deploy host-bash"
    fi
    stop_deploy_container "${DEPLOY_LOG}"
}
trap 'cleanup_children; exit 130' INT TERM
trap 'cleanup_children' EXIT

# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

if [[ "${CLEANUP_ONLY}" -eq 1 ]]; then
    cleanup_stale
    log "cleanup-only: done"
    trap - EXIT
    exit 0
fi

# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------

if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    if [[ ! -x "${DEPLOY_SH}" ]]; then
        err "deploy_x2.sh not found / not executable: ${DEPLOY_SH}"
        exit 1
    fi
    if [[ -z "${SIM_MODEL}" || ! -f "${SIM_MODEL}" ]]; then
        err "deploy needs --model PATH (or MODEL / X2_PLANNER_SMOKE_MODEL env; see MODELS.md)."
        err "tried: ${SIM_MODEL:-<unset>}"
        exit 1
    fi
fi

if [[ -n "${GESTURE_CATALOG}" && ! -f "${GESTURE_CATALOG}" ]]; then
    warn "--gesture-catalog ${GESTURE_CATALOG} does not exist; passing"
    warn "  empty so recorder falls back to ad-hoc --pkl plays only."
    GESTURE_CATALOG=""
fi

# Sweep stale containers BEFORE probing DEBUG_PORT.
if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    preflight_docker_cleanup
fi

# Port pre-check.
if port_in_use "${POSE_PORT}"; then
    err "port ${POSE_PORT} (pose PUB) is in use. Run: $0 --cleanup-only"
    exit 1
fi
if [[ "${WITH_DEPLOY}" -eq 1 ]] && port_in_use "${DEBUG_PORT}"; then
    err "port ${DEBUG_PORT} (deploy x2_debug PUB) is in use."
    err "Run: $0 --cleanup-only or shut down the existing deploy first."
    exit 1
fi
if port_in_use "${MOTION_CLIP_CMD_PORT}"; then
    err "port ${MOTION_CLIP_CMD_PORT} (motion_clip_cmd SUB) is in use."
    err "Run: $0 --cleanup-only or shut down the existing recorder first."
    exit 1
fi

# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------

if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_DESC="ON  (sim --vla, profile=${SIM_PROFILE}, viewer=$([[ "${SIM_VIEWER}" -eq 1 ]] && echo on || echo off))"
elif [[ -n "${PC2_HOST}" ]]; then
    DEPLOY_DESC="OFF -- real-robot mode (PC2 deploy at ${PC2_HOST} SUBs pose:${POSE_PORT})"
else
    DEPLOY_DESC="OFF -- assume external deploy SUBs pose:${POSE_PORT}"
fi

if [[ "${NO_IDLE_PUBLISH}" -eq 1 ]]; then
    IDLE_DESC="OFF -- pose wire stays SILENT between clip plays (watchdog will trip)"
else
    IDLE_DESC="ON  -- recorder publishes idle stand on every no-clip tick"
fi

cat <<EOF
${C_GREEN}+----------------------------------------------------------------------+
|  X2 direct PKL-to-SONIC stack runner                                 |
|  (recorder -> deploy; no planner, no manager, no VLA, no parquet)    |
+----------------------------------------------------------------------+${C_RESET}
  log dir          : ${LOG_DIR}
  duration         : $([[ "${DURATION_S}" -eq 0 ]] && echo "unlimited (run until Ctrl-C)" || echo "${DURATION_S}s wall-clock cap")
  rate             : ${RATE} Hz
  deploy           : ${DEPLOY_DESC}
  idle publish     : ${IDLE_DESC}
  motion-clip wire : tcp://${PUB_BIND_HOST}:${MOTION_CLIP_CMD_PORT} topic=${MOTION_CLIP_CMD_TOPIC}
  gesture catalog  : ${GESTURE_CATALOG:-(disabled; --pkl plays only)}
  ports            : pose=${POSE_PORT}  x2_debug=${DEBUG_PORT}  motion_clip=${MOTION_CLIP_CMD_PORT}
  pose PUB bind    : ${PUB_BIND_HOST}:${POSE_PORT}$( [[ "${PUB_BIND_HOST}" == "127.0.0.1" ]] && echo "  (LAN-isolated: PC2 cannot SUB even if x2_pose_proxy is up)" || echo "  (LAN-visible: PC2 pose proxy SUBs this stream over wifi)" )

  Trigger clips from another terminal (gamepad chords -> demo-bank clips):
    python -m gear_sonic.scripts.play_xbox_controller --list   # chord -> clip table
    python -m gear_sonic.scripts.play_xbox_controller          # long-running dispatcher
  (PKL replay through the kplanner: run_x2_quest3_planner_stack.sh + x2_pkl_command_source.py)
EOF
echo

# --------------------------------------------------------------------------
# Step 1/2 -- Deploy (docker sim + ONNX). Skipped in --no-deploy /
# --pc2-host modes. We pass ``--disable-pose-ref-watchdog`` because the
# recorder publishes directly to pose:5556 (no kplanner merging idle
# frames during cold-start) and the watchdog would otherwise trip
# during the first second before the recorder's idle-stand fallback
# fills the wire. Same rationale the replay / pkl wrappers document.
# --------------------------------------------------------------------------

if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_LOG="${LOG_DIR}/deploy.log"
    if (( DURATION_S > 0 )); then
        DEPLOY_DURATION_S=$(( DURATION_S + 30 ))
    else
        DEPLOY_DURATION_S=0
    fi

    DEPLOY_ARGS=(
        sim
        --no-confirm
        --vla
        --vla-zmq-host 127.0.0.1
        --vla-zmq-port "${POSE_PORT}"
        --vla-zmq-topic "${POSE_TOPIC}"
        --vla-debug-port "${DEBUG_PORT}"
        --vla-debug-topic "${DEBUG_TOPIC}"
        --sim-profile "${SIM_PROFILE}"
        --sim-with-omnihand
        --wrist-bypass ik
        --model "${SIM_MODEL}"
        --autostart-after 0
        --sim-band-release-after-s "${SIM_BAND_RELEASE_S:-3}"
        --deploy-extra-arg --disable-pose-ref-watchdog
    )
    if (( DEPLOY_DURATION_S > 0 )); then
        DEPLOY_ARGS+=(--max-duration "${DEPLOY_DURATION_S}")
    fi
    if [[ "${SIM_VIEWER}" -eq 1 ]]; then
        DEPLOY_ARGS+=(--sim-viewer)
        if [[ -n "${SIM_CAM_TRACK_BODY}" ]]; then
            DEPLOY_ARGS+=(
                --sim-cam-track-body "${SIM_CAM_TRACK_BODY}"
                --sim-cam-distance "${SIM_CAM_DISTANCE}"
                --sim-cam-elevation "${SIM_CAM_ELEVATION}"
                --sim-cam-azimuth "${SIM_CAM_AZIMUTH}"
            )
        fi
    fi

    log "Step 1/2 -- spawning deploy_x2.sh sim --vla -> ${DEPLOY_LOG}"
    "${DEPLOY_SH}" "${DEPLOY_ARGS[@]}" >"${DEPLOY_LOG}" 2>&1 &
    DEPLOY_PID=$!

    log "  waiting for deploy 'Launching ...' marker (up to 180s)..."
    if ! wait_for_log_marker "${DEPLOY_LOG}" "${DEPLOY_PID}" "Launching ..." 180 "deploy"; then
        exit 1
    fi
    log "  deploy READY (pid=${DEPLOY_PID}); settle 2s before recorder ..."
    sleep 2.0
else
    if [[ -n "${PC2_HOST}" ]]; then
        log "Step 1/2 -- deploy spawn SKIPPED (--pc2-host ${PC2_HOST}). PC2 daemons must already be running."
    else
        log "Step 1/2 -- deploy spawn SKIPPED (--no-deploy). External deploy must SUB pose on :${POSE_PORT}."
    fi
fi

# --------------------------------------------------------------------------
# Step 2/2 -- Pose merger (x2_pose_merger.py: the teleop-only stand-in
# for the dataset recorder's subscribe mode; no dataset writes). It runs
# without a body_pose or arm_and_hands producer attached: those SUBs just
# sit empty and the idle-stand fallback keeps :5556 fed. The
# motion_clip_cmd SUB binds on the same wire play_xbox_controller
# connects to.
# --------------------------------------------------------------------------

RECORDER_LOG="${LOG_DIR}/recorder.log"
# When the laptop's pose PUB is loopback-only, the recorder's
# debug-port SUB should also stay loopback (we connect TO the deploy's
# debug PUB, which lives on the local sim docker). For real-robot /
# external-deploy modes the PC2's x2_debug needs reaching too; PC2's
# pose proxy will be the source. We mirror run_x2_quest3_planner_stack.sh's
# RECORDER_DEBUG_SUB_HOST logic.
if [[ -n "${PC2_HOST}" ]]; then
    RECORDER_DEBUG_SUB_HOST="${PC2_HOST}"
else
    RECORDER_DEBUG_SUB_HOST="localhost"
fi

RECORDER_ARGS=(
    -u
    -m gear_sonic.scripts.x2_pose_merger
    --body-pose-source zmq
    --arm-targets-source zmq
    --body-pose-sub-host localhost
    --body-pose-sub-port "${BODY_POSE_PORT}"
    --body-pose-sub-topic "${BODY_POSE_TOPIC}"
    --arm-and-hands-sub-host localhost
    --arm-and-hands-sub-port "${ARM_HANDS_PORT}"
    --pub-host "${PUB_BIND_HOST}"
    --pub-port "${POSE_PORT}"
    --pub-topic "${POSE_TOPIC}"
    --sub-host "${RECORDER_DEBUG_SUB_HOST}"
    --sub-port "${DEBUG_PORT}"
    --sub-topic "${DEBUG_TOPIC}"
    --motion-clip-cmd-host '*'
    --motion-clip-cmd-port "${MOTION_CLIP_CMD_PORT}"
    --motion-clip-cmd-topic "${MOTION_CLIP_CMD_TOPIC}"
    --rate "${RATE}"
    --teleop-only
)
if [[ -n "${GESTURE_CATALOG}" ]]; then
    RECORDER_ARGS+=(--gesture-catalog "${GESTURE_CATALOG}")
else
    # Explicit empty disables catalog loading; ad-hoc --pkl still works.
    RECORDER_ARGS+=(--gesture-catalog "")
fi
if [[ "${NO_IDLE_PUBLISH}" -eq 1 ]]; then
    RECORDER_ARGS+=(--no-idle-publish)
fi

log "Step 2/2 -- spawning x2_pose_merger (subscribe, --teleop-only) -> ${RECORDER_LOG}"
# setsid so the recorder is its own session leader; we SIGINT the
# pgid in cleanup so any in-flight motion-clip session gets a clean
# tear-down.
setsid "${PYTHON}" "${RECORDER_ARGS[@]}" >"${RECORDER_LOG}" 2>&1 < /dev/null &
RECORDER_PID=$!
RECORDER_PGID="${RECORDER_PID}"

# The recorder's loop never logs "first body_pose received" in this
# stack (no upstream). The earliest reliable ready marker is the
# motion_clip_cmd SUB bind log line, which fires once the SUB thread
# has bound and queue is ready -- exactly the point at which the
# operator can fire a play.
log "  waiting for recorder 'motion_clip_cmd wired:' marker (up to 60s)..."
if ! wait_for_log_marker "${RECORDER_LOG}" "${RECORDER_PID}" \
        "motion_clip_cmd wired:" 60 "recorder"; then
    exit 1
fi
log "  recorder READY (pid=${RECORDER_PID}); stack is live."
log "  follow recorder output with: tail -F ${RECORDER_LOG}"

# Surface the recorder's return-to-stand / shutdown-settle events in
# THIS terminal (the recorder's own logs go to recorder.log, which the
# operator may not be tailing). Background watcher: tail only NEW lines
# (-n0) and reformat the recorder's markers into a prominent
# launcher-level banner so the operator sees exactly when the robot is
# being reset to the stable stand pose. Killed in cleanup_children.
(
    tail -n0 -F "${RECORDER_LOG}" 2>/dev/null | while IFS= read -r _line; do
        case "${_line}" in
            *"return-to-stand: ramping"*)
                log "*** robot RESETTING to stable stand pose (clip stopped/ended) ***" ;;
            *"return-to-stand: reached idle stand"*)
                log "*** robot RESET to stable stand pose -- holding idle stand ***" ;;
            *"shutdown: settling to stand"*)
                log "*** shutdown: settling robot to stable stand pose before exit ***" ;;
        esac
    done
) &
RESET_WATCHER_PID=$!
log ""
log "  Operator hint -- in another terminal:"
log "    python -m gear_sonic.scripts.play_xbox_controller --list     # chord -> clip table"
log "    python -m gear_sonic.scripts.play_xbox_controller            # gamepad -> demo-bank clips"

# --------------------------------------------------------------------------
# Main wait: block until the recorder exits (Ctrl-C, error, or
# --duration cap on the deploy side fires first).
# --------------------------------------------------------------------------

__exit_rc=""
check_child_alive_or_finished() {
    local pid="$1"
    if kill -0 "${pid}" 2>/dev/null; then
        __exit_rc=""
        return 0
    fi
    wait "${pid}" 2>/dev/null
    local rc=$?
    if (( rc == 127 )); then
        __exit_rc=0
        return 1
    fi
    __exit_rc="${rc}"
    if (( rc == 0 )); then
        return 1
    fi
    return 2
}

main_wait_loop() {
    local end_ts="$1"  # 0 = run forever; non-zero = unix timestamp deadline
    local check_rc
    while :; do
        if (( end_ts > 0 )) && (( $(date +%s) >= end_ts )); then
            log "duration ${DURATION_S}s elapsed; shutting down."
            return 0
        fi

        if [[ -n "${DEPLOY_PID}" ]]; then
            check_child_alive_or_finished "${DEPLOY_PID}"
            check_rc=$?
            if (( check_rc == 1 )) && (( DURATION_S > 0 )); then
                log "deploy finished normally (pid=${DEPLOY_PID}, exit=${__exit_rc:-0}, --max-duration elapsed); ending session."
                return 0
            elif (( check_rc == 2 )); then
                err "deploy died (pid=${DEPLOY_PID}, exit=${__exit_rc}); see ${DEPLOY_LOG}"
                return 1
            fi
        fi

        check_child_alive_or_finished "${RECORDER_PID}"
        check_rc=$?
        if (( check_rc == 1 )); then
            log "recorder finished normally (pid=${RECORDER_PID}, exit=${__exit_rc:-0}); ending session."
            return 0
        elif (( check_rc == 2 )); then
            err "recorder died unexpectedly (pid=${RECORDER_PID}, exit=${__exit_rc}); see ${RECORDER_LOG}"
            return 1
        fi

        sleep 1
    done
}

if (( DURATION_S > 0 )); then
    log "running for up to ${DURATION_S}s wall-clock (Ctrl-C to stop early) ..."
    end_ts=$(( $(date +%s) + DURATION_S ))
    main_wait_loop "${end_ts}"
else
    log "running until Ctrl-C ..."
    main_wait_loop 0
fi
WAIT_LOOP_RC=$?

exit "${WAIT_LOOP_RC}"
