#!/usr/bin/env bash
# X2 planner smoke runner (kplanner; the legacy heuristic planner is not shipped).
#
# Spawns the planner, an optional debug capture, and (when --with-deploy is
# given) the X2 sim deploy as CHILD processes of this script. A trap-based
# cleanup ensures every child is killed on any exit path -- normal, SIGINT,
# SIGTERM, or shell error.
#
# Why this exists: iterating on the planner against the C++ deploy easily
# leaves zombie ros2/zmq/python processes if you Ctrl-C the wrong terminal.
# This wrapper is the canonical entry point for "policy in the loop"
# validation (planner plan section D.6) and the only thing the docs / tests
# tell operators to use.
#
# Usage:
#   gear_sonic/scripts/run_planner_smoke.sh [--demo PATH.yaml] [--duration N]
#                                           [--planner {heuristic,kplanner}]
#                                           [--kplanner-{vqvae,pose,root}-ckpt PATH]
#                                           [--kplanner-device {cuda,cpu}]
#                                           [--with-deploy] [--with-debug]
#                                           [--with-viewer] [--pub-port N]
#                                           [--cleanup-only] [--keyboard]
#                                           [--no-sim-viewer]
#                                           [--sim-profile {parity,manual,gantry,handoff}]
#                                           [--sim-rsi-pkl PATH]
#
# Default --sim-profile is `parity`: bridge RSIs from the canonical
# planner anchor PKL (auto-baked on first use), elastic band off,
# robot spawns on the floor. Pass `--sim-profile manual` if you need
# the legacy band-catch-from-air behaviour.
#
# Default --planner is `kplanner`: it drives the YAML demos through
# the trained neural daemon -> deploy bridge -> SONIC policy chain
# (with NO Quest3 manager in the loop). `--planner heuristic` needs
# the legacy x2_heuristic_planner.py, which is not shipped in this port. This is the canonical isolation test for "is the bug in
# the kplanner -> sonic link or in the Quest3 -> planner link?".
#
# Examples:
#   # Scripted demo end-to-end with SONIC physics + MuJoCo viewer:
#   ./run_planner_smoke.sh --demo gear_sonic/data/scripted_demos/forward_back_turn.yaml \
#                          --with-deploy --duration 30
#
#   # SAME demo but driven through the NEURAL planner (isolation test):
#   ./run_planner_smoke.sh --demo gear_sonic/data/scripted_demos/forward_back_turn.yaml \
#                          --with-deploy --duration 30 --planner kplanner
#
#   # Keyboard teleop with SONIC physics + MuJoCo viewer (W/A/S/D/Q/E/...):
#   ./run_planner_smoke.sh --with-deploy --keyboard --duration 120
#
#   # Lightweight: planner + standalone kinematic viewer (no physics, no SONIC):
#   ./run_planner_smoke.sh --demo gear_sonic/data/scripted_demos/forward_back_turn.yaml \
#                          --with-viewer
#
#   # Headless CI / regression run (no windows):
#   ./run_planner_smoke.sh --duration 5 --keyboard --with-deploy --no-sim-viewer
#
#   ./run_planner_smoke.sh --cleanup-only        # kill stale planners + free ports
#
# --with-viewer opens a MuJoCo window subscribing to the planner's pose
# stream so you can see the planner's KINEMATIC output (no physics).
#
# --with-deploy spawns the SONIC sim deploy in a docker container; it
# subscribes to the planner's pose topic and runs the trained policy
# closed-loop in MuJoCo with full physics. By default we also pass
# --sim-viewer + --autostart-after 0 so a window pops with the robot
# tracked by SONIC; pass --no-sim-viewer to suppress for CI / headless.
#
# Pre-flight:
#   - Verifies the publish port is free (refuses to start otherwise; the
#     planner CLI also refuses but we surface a clearer error).
#   - (heuristic only) Verifies the primitives PKL exists.
#
# All children inherit a fresh process group so SIGTERM cascades without
# orphaning grandchildren.

set -u
set -o pipefail

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRIMITIVES_PKL="${REPO_ROOT}/gear_sonic/data/motions/x2_planner_primitives.pkl"
BINS_YAML="${REPO_ROOT}/gear_sonic/data/motions/x2_planner_bins.yaml"
# Stack interpreter: SONIC_PYTHON overrides; default the repo .venv, else python3.
PYTHON="${SONIC_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: no python interpreter found (create ${REPO_ROOT}/.venv or set SONIC_PYTHON=/path/to/python)" >&2
    exit 1
fi

PID_FILE="/tmp/x2_heuristic_planner.pid"
# Resolved after --planner parsing below.
PID_FILE_KPLANNER="/tmp/x2_kplanner.pid"
LOG_DIR="${REPO_ROOT}/data/sim_to_real_anchors/planner_smoke"

# --------------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------------

DEMO=""
DURATION_S=20
PUB_PORT=5556
WITH_DEPLOY=0
WITH_DEBUG=0
WITH_VIEWER=0
KEYBOARD=0
CLEANUP_ONLY=0
SIM_VIEWER=1            # opens MuJoCo window when --with-deploy is set
# Planner backend selector. ``kplanner`` (default) is the neural daemon:
# it pipes a YAML demo straight through the trained models -> deploy
# bridge -> SONIC policy chain, bypassing Quest3 entirely. ``heuristic``
# is the historical planner the YAML demos under
# gear_sonic/data/scripted_demos/ were authored for; the intent /
# magnitude vocabulary is identical (the kplanner routes them through
# the same _BASE_VELOCITY / _WALK_VELOCITY_BY_MAGNITUDE tables), but the
# heuristic planner itself is NOT shipped in this port.
PLANNER_KIND="kplanner"
HEURISTIC_PLANNER_PY="${REPO_ROOT}/gear_sonic/scripts/x2_heuristic_planner.py"
# kplanner-specific paths. Defaults mirror the production wrapper
# (``run_x2_quest3_planner_stack.sh``); override with the env vars below
# or the matching CLI flag if you want to validate an alternate ckpt.
KPL_VQVAE_CKPT_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0200000.ckpt"
KPL_POSE_CKPT_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_pose_x2_v2/version_1/checkpoints/model-step=0250000.ckpt"
KPL_ROOT_CKPT_DEFAULT="${REPO_ROOT}/motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0235000.ckpt"
KPL_VQVAE_CKPT="${KPLANNER_VQVAE_CKPT:-}"
KPL_POSE_CKPT="${KPLANNER_POSE_CKPT:-}"
KPL_ROOT_CKPT="${KPLANNER_ROOT_CKPT:-}"
KPL_DEVICE="${KPLANNER_DEVICE:-cuda}"
KPL_REPLAN_THRESHOLD="${KPLANNER_REPLAN_THRESHOLD_FRAMES:-16}"
# Kplanner RSI anchor PKL (separate from the heuristic primitive PKL).
# Auto-baked on first use against the warmup qpos that the kplanner
# daemon will emit on tick 0, so the bridge RSI = planner wire content
# at boot -> no tracker-error fight on first frame.
KPL_SIM_RSI_PKL="${REPO_ROOT}/data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_kplanner_rsi_anchor.pkl"
# Default to PARITY profile (RSI from a stand-pose PKL, elastic band
# OFF, ramp 0, autostart 0) so the planner-driven sim mirrors
# `deploy_x2.sh sim --motion <pkl>` exactly: robot spawns ON THE FLOOR
# at the planner's canonical anchor frame, no air-drop, no band drama.
# `--sim-profile manual` is still available for legacy band-catch behaviour.
SIM_PROFILE="parity"
# RSI source PKL for parity profile. Bridge spawns at this PKL's frame 0
# pose. The default PKL is auto-baked from
# `gear_sonic.scripts.bake_planner_rsi_anchor` so frame 0 is bit-identical
# to the planner's canonical first-frame emit (joints = idle_stand[0],
# quat = identity yaw, pelvis_z from clip = ~0.685m). This guarantees:
#   bridge spawn pose == warmup wire content == state-machine first frame
# -> no tracker-error fight on tick 0.
SIM_RSI_PKL="${REPO_ROOT}/data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_rsi_anchor.pkl"
# ONNX checkpoint to feed --with-deploy. Override with --model PATH or
# the X2_PLANNER_SMOKE_MODEL env var (else MODEL, else the SONIC cache
# layout under $SONIC_HOME/x2 -- see MODELS.md / download_from_hf.py).
SIM_MODEL="${X2_PLANNER_SMOKE_MODEL:-${MODEL:-${SONIC_X2_MODELS:-${SONIC_HOME:-$HOME/.cache/sonic}/x2}/sonic_policy/x2_sonic_policy.onnx}}"
# When the deploy is run with --sim-viewer, lock the MuJoCo camera onto
# the robot's pelvis so a multi-primitive sequence (which can drift
# meters across the floor) stays in frame. Override with the matching
# CLI flags below if you want a different framing or no tracking.
SIM_CAM_TRACK_BODY="pelvis"
SIM_CAM_DISTANCE="3.5"
SIM_CAM_ELEVATION="-12"
SIM_CAM_AZIMUTH="135"

usage() {
    grep -E '^# (Usage|Examples)' "$0" | sed 's/^# //' >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --demo) DEMO="$2"; shift 2 ;;
        --duration) DURATION_S="$2"; shift 2 ;;
        --pub-port) PUB_PORT="$2"; shift 2 ;;
        --with-deploy) WITH_DEPLOY=1; shift ;;
        --with-debug) WITH_DEBUG=1; shift ;;
        --with-viewer) WITH_VIEWER=1; shift ;;
        --keyboard) KEYBOARD=1; shift ;;
        --cleanup-only) CLEANUP_ONLY=1; shift ;;
        --no-sim-viewer) SIM_VIEWER=0; shift ;;
        --sim-profile) SIM_PROFILE="$2"; shift 2 ;;
        --sim-rsi-pkl) SIM_RSI_PKL="$2"; shift 2 ;;
        --model) SIM_MODEL="$2"; shift 2 ;;
        --sim-cam-track-body) SIM_CAM_TRACK_BODY="$2"; shift 2 ;;
        --sim-cam-distance) SIM_CAM_DISTANCE="$2"; shift 2 ;;
        --sim-cam-elevation) SIM_CAM_ELEVATION="$2"; shift 2 ;;
        --sim-cam-azimuth) SIM_CAM_AZIMUTH="$2"; shift 2 ;;
        --no-sim-cam-track) SIM_CAM_TRACK_BODY=""; shift ;;
        --planner) PLANNER_KIND="$2"; shift 2 ;;
        --kplanner-vqvae-ckpt) KPL_VQVAE_CKPT="$2"; shift 2 ;;
        --kplanner-pose-ckpt) KPL_POSE_CKPT="$2"; shift 2 ;;
        --kplanner-root-ckpt) KPL_ROOT_CKPT="$2"; shift 2 ;;
        --kplanner-device) KPL_DEVICE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

case "${PLANNER_KIND}" in
    heuristic|kplanner) ;;
    *)
        echo "--planner must be 'heuristic' or 'kplanner' (got: ${PLANNER_KIND})" >&2
        exit 2
        ;;
esac

# Apply ckpt defaults only for kplanner mode (heuristic mode ignores
# these and the defaults reference cloud-trained PKLs that the user may
# not always have on disk).
if [[ "${PLANNER_KIND}" == "kplanner" ]]; then
    KPL_VQVAE_CKPT="${KPL_VQVAE_CKPT:-${KPL_VQVAE_CKPT_DEFAULT}}"
    KPL_POSE_CKPT="${KPL_POSE_CKPT:-${KPL_POSE_CKPT_DEFAULT}}"
    KPL_ROOT_CKPT="${KPL_ROOT_CKPT:-${KPL_ROOT_CKPT_DEFAULT}}"
    # Route SIM_RSI_PKL through the kplanner anchor since the bridge
    # RSI must match the kplanner's first emitted frame, not the
    # heuristic's curated stand pose. ``--sim-rsi-pkl`` overrides
    # both flavours when the user wants a custom RSI.
    if [[ "${SIM_RSI_PKL}" == "${REPO_ROOT}/data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_rsi_anchor.pkl" ]]; then
        SIM_RSI_PKL="${KPL_SIM_RSI_PKL}"
    fi
    # Use the kplanner-specific PID file so a stale heuristic PID file
    # doesn't false-positive the readiness check, and vice versa.
    PID_FILE="${PID_FILE_KPLANNER}"
fi

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

log() { printf '[planner-smoke %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
err() { printf '[planner-smoke %s ERROR] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

port_in_use() {
    # Returns 0 if port is in use, 1 if free.
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nPi ":${port}" >/dev/null 2>&1
        return $?
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -n tcp "${port}" >/dev/null 2>&1
        return $?
    fi
    # Fallback: probe via Python.
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
    log "  killing pid $pid"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    log "  force-killing pid $pid"
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
        for p in ${pids}; do kill_pid_quiet "${p}"; done
    fi
}

cleanup_stale() {
    log "cleanup_stale: PID file=${PID_FILE} port=${PUB_PORT}"
    if [[ -f "${PID_FILE}" ]]; then
        local stale_pid
        stale_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        kill_pid_quiet "${stale_pid}"
        rm -f "${PID_FILE}"
    fi
    free_port "${PUB_PORT}"
}

# --------------------------------------------------------------------------
# Cleanup trap (set BEFORE any child is spawned)
# --------------------------------------------------------------------------

CHILD_PIDS=()
cleanup_children() {
    log "shutting down children..."
    # Reverse-order so the deploy goes down before the planner that feeds it.
    for ((i=${#CHILD_PIDS[@]}-1; i>=0; i--)); do
        kill_pid_quiet "${CHILD_PIDS[$i]}"
    done
    CHILD_PIDS=()
    rm -f "${PID_FILE}"
}
trap 'cleanup_children; exit 130' INT TERM
trap 'cleanup_children' EXIT

# --------------------------------------------------------------------------
# wait_for_deploy_ready: tail $1 (deploy log) for the bridge's
# "Launching ..." marker that indicates the C++ deploy + bridge are up
# and ready to consume pose frames. Returns 0 on ready, 1 on timeout
# or early death.
#
# This mirrors the wait loop in record_x2_dataset.sh and
# browse_x2_planner_primitives._wait_for_deploy_ready, which has been
# the proven boot pattern for "land the robot gently on the floor"
# for months -- because nothing is on the ZMQ pose wire while the
# C++ deploy is booting + the band is releasing, the policy uses its
# internal default_angles target and the band drop is the ONLY
# transient the policy has to absorb. (vs. the old smoke flow:
# planner spawns first and starts streaming idle_stand BEFORE the
# deploy is up -> policy tries to track a moving reference during
# the band drop -> robot launches into the air.)
# --------------------------------------------------------------------------
wait_for_deploy_ready() {
    local deploy_log="$1"
    local deploy_pid="$2"
    local timeout_s="${3:-180}"
    local marker="${4:-Launching ...}"
    local start_ts
    start_ts=$(date +%s)
    while :; do
        if ! kill -0 "${deploy_pid}" 2>/dev/null; then
            err "deploy died during bring-up; tail of log:"
            tail -n 40 "${deploy_log}" >&2 || true
            return 1
        fi
        if [[ -f "${deploy_log}" ]] && grep -F -q "${marker}" "${deploy_log}" 2>/dev/null; then
            return 0
        fi
        local now elapsed
        now=$(date +%s)
        elapsed=$((now - start_ts))
        if (( elapsed > timeout_s )); then
            err "deploy did not reach '${marker}' within ${timeout_s}s; tail of log:"
            tail -n 40 "${deploy_log}" >&2 || true
            return 1
        fi
        sleep 0.5
    done
}

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

if [[ "${PLANNER_KIND}" == "heuristic" ]]; then
    if [[ ! -f "${HEURISTIC_PLANNER_PY}" ]]; then
        err "--planner heuristic: ${HEURISTIC_PLANNER_PY} is not shipped in this port."
        err "use --planner kplanner (the default)."
        exit 1
    fi
    if [[ ! -f "${PRIMITIVES_PKL}" ]]; then
        err "primitives PKL not found: ${PRIMITIVES_PKL}"
        err "(see manifest/REGENERATE.md / MODELS.md for the primitives bank)"
        exit 1
    fi
    if [[ ! -f "${BINS_YAML}" ]]; then
        err "bins YAML not found: ${BINS_YAML}"
        exit 1
    fi
else
    # kplanner mode: validate ckpts are reachable before spending time
    # on the deploy bring-up. The neural daemon will refuse to start
    # otherwise (with a less actionable error).
    for ck in "${KPL_VQVAE_CKPT}" "${KPL_POSE_CKPT}" "${KPL_ROOT_CKPT}"; do
        if [[ ! -f "${ck}" ]]; then
            err "kplanner ckpt not found: ${ck}"
            err "override with --kplanner-{vqvae,pose,root}-ckpt PATH or env"
            err "  KPLANNER_VQVAE_CKPT / KPLANNER_POSE_CKPT / KPLANNER_ROOT_CKPT"
            exit 1
        fi
    done
fi
if port_in_use "${PUB_PORT}"; then
    err "publish port ${PUB_PORT} is in use; run: $0 --cleanup-only"
    exit 1
fi
mkdir -p "${LOG_DIR}"

# --------------------------------------------------------------------------
# Spawn order (CRITICAL): when --with-deploy is set, the deploy MUST come
# up first, reach 'Launching ...' (bridge + C++ deploy alive, sim profile
# fully resolved, init pose applied, viewer window open), and then settle
# for ~2 s before the planner starts publishing on the ZMQ pose wire.
#
# What "good" looks like under --sim-profile parity (the default):
#   * Bridge RSIs from SIM_RSI_PKL frame 0 -> robot spawns ON THE FLOOR
#     in the planner's canonical anchor pose (joints from idle_stand[0],
#     identity-yaw quat, pelvis_z ~ 0.685m).
#   * Elastic band OFF, ramp_seconds 0, autostart 0 -- mirrors what
#     `deploy_x2.sh sim --motion <pkl>` does in legacy playback mode.
#   * Nothing on the ZMQ wire during boot.
#   * After "Launching ..." marker + 2s settle, planner starts publishing
#     the SAME anchor frame frozen for `--warmup-quiet-stand-s` seconds
#     (default 2.0 s), then transitions to live state-machine output.
#
# Bit-identical bridge spawn / warmup wire / state-machine first frame
# means there's no tracker-error fight on tick 0; the robot stays planted.
#
# Legacy behavior (--sim-profile manual) is preserved as an opt-in:
# spawns at sim_init_pose=default (pelvis 0.85m air-drop) with the
# elastic band catching the body, then auto-releasing 1s after the first
# deploy command. Tolerable for some debug flows but the robot visibly
# launches from the air.
#
# Standalone (no --with-deploy) keeps the legacy ordering: planner
# first, then optional viewer.
# --------------------------------------------------------------------------

DEPLOY_PID=""
if [[ "${WITH_DEPLOY}" -eq 1 ]]; then
    DEPLOY_SH="${REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh"
    if [[ ! -x "${DEPLOY_SH}" ]]; then
        err "deploy_x2.sh not found / not executable: ${DEPLOY_SH}"
        exit 1
    fi
    DEPLOY_LOG="${LOG_DIR}/deploy_$(date +%Y%m%d_%H%M%S).log"
    if [[ -z "${SIM_MODEL}" || ! -f "${SIM_MODEL}" ]]; then
        err "deploy needs --model PATH (or X2_PLANNER_SMOKE_MODEL env)."
        err "tried: ${SIM_MODEL:-<unset>}"
        exit 1
    fi
    # Parity profile requires --motion <pkl> for bridge RSI even in --vla
    # mode (the C++ deploy ignores --motion under --vla but the bridge
    # uses MOTION_SOURCE for RSI). Auto-bake the canonical anchor PKL
    # if it's missing so a fresh checkout (or one where primitives were
    # rebuilt) just works. The bake target differs by planner kind:
    # heuristic -> idle_stand[0] from the curated primitives PKL,
    # kplanner -> the warmup qpos the neural daemon emits on tick 0.
    # The RSI anchor bakers (bake_{planner,kplanner}_rsi_anchor.py) are not
    # shipped in this port. Without them (and without a pre-baked anchor
    # PKL) parity mode cannot start; drop to the manual profile with a warning.
    if [[ "${SIM_PROFILE}" == "parity" && ! -f "${SIM_RSI_PKL}" ]]; then
        if [[ "${PLANNER_KIND}" == "heuristic" ]]; then _BAKER="${REPO_ROOT}/gear_sonic/scripts/bake_planner_rsi_anchor.py"
        else _BAKER="${REPO_ROOT}/gear_sonic/scripts/bake_kplanner_rsi_anchor.py"; fi
        if [[ ! -f "${_BAKER}" ]]; then
            err "parity RSI anchor (${SIM_RSI_PKL}) missing and $(basename "${_BAKER}") is not shipped -> --sim-profile manual"
            err "(deploy spawns at its default pose; expect a settle transient at start)"
            SIM_PROFILE="manual"
        fi
    fi
    if [[ "${SIM_PROFILE}" == "parity" ]]; then
        if [[ "${PLANNER_KIND}" == "heuristic" ]]; then
            if [[ ! -f "${SIM_RSI_PKL}" ]]; then
                log "RSI anchor PKL not found at ${SIM_RSI_PKL}; baking now..."
                if ! "${PYTHON}" -m gear_sonic.scripts.bake_planner_rsi_anchor \
                        --primitives-pkl "${PRIMITIVES_PKL}" \
                        --bins-yaml "${BINS_YAML}" \
                        --out "${SIM_RSI_PKL}" \
                        >>"${LOG_DIR}/rsi_anchor_bake.log" 2>&1; then
                    err "failed to bake RSI anchor; see ${LOG_DIR}/rsi_anchor_bake.log"
                    exit 1
                fi
            fi
        elif [[ -f "${REPO_ROOT}/gear_sonic/scripts/bake_kplanner_rsi_anchor.py" ]]; then
            # Always re-bake the kplanner anchor (it's <300 ms and
            # depends only on the daemon's warmup qpos source). Mirrors
            # run_x2_quest3_planner_stack.sh's behaviour so changes to
            # the warmup qpos take effect without a manual rebuild.
            log "baking kplanner RSI anchor PKL -> ${SIM_RSI_PKL}"
            if ! "${PYTHON}" -m gear_sonic.scripts.bake_kplanner_rsi_anchor \
                    --out "${SIM_RSI_PKL}" \
                    >>"${LOG_DIR}/rsi_anchor_bake.log" 2>&1; then
                err "failed to bake kplanner RSI anchor; see ${LOG_DIR}/rsi_anchor_bake.log"
                exit 1
            fi
        fi
        log "parity RSI source: ${SIM_RSI_PKL}  (planner=${PLANNER_KIND})"
    fi
    # Deploy runs for the full duration plus a small headroom so it
    # doesn't terminate mid-clip if the planner takes a moment to come
    # up. The planner is the wall-clock keeper at the bottom of the
    # script.
    DEPLOY_DURATION_S=$(( DURATION_S + 5 ))
    DEPLOY_ARGS=(
        sim
        --no-confirm
        --vla
        --vla-zmq-host 127.0.0.1
        --vla-zmq-port "${PUB_PORT}"
        --sim-profile "${SIM_PROFILE}"
        --model "${SIM_MODEL}"
        --autostart-after 0
        --max-duration "${DEPLOY_DURATION_S}"
        # Disable the C++ deploy's pose-ref starvation watchdog. In the
        # smoke flow the planner takes ~3 s to load + warmup before it
        # streams the first pose frame, which is longer than the C++
        # binary's 0.5 s SAFE_IDLE trip. Once tripped, exit requires an
        # operator resume chord on tcp://localhost:5566 -- which nobody
        # publishes here (no Quest3 manager). The deploy then holds
        # default_angles with 4x kd, the spawn-pose <-> default_angles
        # delta tips the robot, and the kplanner+sonic combo gets blamed.
        # Same fix the production wrapper (run_x2_quest3_planner_stack.sh)
        # applies in local-sim mode via ``POSE_REF_WATCHDOG=auto -> off``.
        --deploy-extra-arg --disable-pose-ref-watchdog
    )
    if [[ "${SIM_PROFILE}" == "parity" ]]; then
        DEPLOY_ARGS+=(--motion "${SIM_RSI_PKL}")
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
            log "spawning deploy_x2.sh sim FIRST (viewer ON, cam->${SIM_CAM_TRACK_BODY}, profile=${SIM_PROFILE}) -> ${DEPLOY_LOG}"
        else
            log "spawning deploy_x2.sh sim FIRST (viewer ON, free cam, profile=${SIM_PROFILE}) -> ${DEPLOY_LOG}"
        fi
    else
        log "spawning deploy_x2.sh sim FIRST (viewer OFF, profile=${SIM_PROFILE}) -> ${DEPLOY_LOG}"
    fi
    "${DEPLOY_SH}" "${DEPLOY_ARGS[@]}" >"${DEPLOY_LOG}" 2>&1 &
    DEPLOY_PID=$!
    CHILD_PIDS+=("${DEPLOY_PID}")

    log "waiting for deploy to reach 'Launching ...' (up to 180s)..."
    if ! wait_for_deploy_ready "${DEPLOY_LOG}" "${DEPLOY_PID}" 180 "Launching ..."; then
        err "deploy did not become ready; aborting."
        exit 1
    fi
    log "deploy ready (pid=${DEPLOY_PID}); sleeping 2.0s settle before starting planner."
    sleep 2.0
fi

# --------------------------------------------------------------------------
# Spawn planner
# --------------------------------------------------------------------------

PLANNER_LOG="${LOG_DIR}/planner_$(date +%Y%m%d_%H%M%S).log"
log "spawning planner (kind=${PLANNER_KIND}) -> ${PLANNER_LOG}"
if [[ "${PLANNER_KIND}" == "heuristic" ]]; then
    PLANNER_ARGS=(
        -m gear_sonic.scripts.x2_heuristic_planner
        --primitives "${PRIMITIVES_PKL}"
        --bins "${BINS_YAML}"
        --pub-host "127.0.0.1"
        --pub-port "${PUB_PORT}"
        --pid-file "${PID_FILE}"
        --duration-s "${DURATION_S}"
    )
else
    # kplanner publishes directly on the deploy's pose topic at PUB_PORT
    # (no --body-pose-port -> no recorder-merge mode), so the scripted
    # YAML drives the deploy bridge in isolation from the Quest3 stack.
    PLANNER_ARGS=(
        -m gear_sonic.scripts.x2_kplanner
        --pub-host "127.0.0.1"
        --pub-port "${PUB_PORT}"
        --vqvae-ckpt "${KPL_VQVAE_CKPT}"
        --pose-ckpt "${KPL_POSE_CKPT}"
        --root-ckpt "${KPL_ROOT_CKPT}"
        --device "${KPL_DEVICE}"
        --replan-threshold-frames "${KPL_REPLAN_THRESHOLD}"
        --pid-file "${PID_FILE}"
        --duration-s "${DURATION_S}"
    )
fi
[[ -n "${DEMO}" ]] && PLANNER_ARGS+=(--demo "${DEMO}")
[[ "${KEYBOARD}" -eq 1 ]] && PLANNER_ARGS+=(--keyboard)

cd "${REPO_ROOT}"
# Inject the motionbricks editable install onto PYTHONPATH so the
# neural daemon (which imports motionbricks.motion_backbone.*) resolves
# regardless of whether the interpreter has run ``pip install -e
# motionbricks/``. Idempotent on .venv runs because the editable
# install masks the path entry. Mirrors run_x2_quest3_planner_stack.sh.
PLANNER_ENV_PYTHONPATH="${REPO_ROOT}/motionbricks:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${KEYBOARD}" -eq 1 ]]; then
    PYTHONPATH="${PLANNER_ENV_PYTHONPATH}" \
        "${PYTHON}" "${PLANNER_ARGS[@]}" 2>&1 | tee "${PLANNER_LOG}" &
else
    PYTHONPATH="${PLANNER_ENV_PYTHONPATH}" \
        "${PYTHON}" "${PLANNER_ARGS[@]}" >"${PLANNER_LOG}" 2>&1 &
fi
PLANNER_PID=$!
CHILD_PIDS+=("${PLANNER_PID}")

# Wait for the planner to come up (PID file written).
log "waiting for planner pid=${PLANNER_PID} to write PID file..."
for _ in $(seq 1 100); do
    [[ -f "${PID_FILE}" ]] && break
    if ! kill -0 "${PLANNER_PID}" 2>/dev/null; then
        err "planner exited early; tail of log:"
        tail -n 40 "${PLANNER_LOG}" >&2 || true
        exit 1
    fi
    sleep 0.05
done
if [[ ! -f "${PID_FILE}" ]]; then
    err "planner did not write ${PID_FILE} within 5s"
    exit 1
fi
log "planner up (pid=${PLANNER_PID})."

# --------------------------------------------------------------------------
# Optional: dump_x2_debug
# --------------------------------------------------------------------------

if [[ "${WITH_DEBUG}" -eq 1 && ! -f "${REPO_ROOT}/gear_sonic/scripts/dump_x2_debug.py" ]]; then
    err "--with-debug: gear_sonic/scripts/dump_x2_debug.py is not shipped in this port; drop --with-debug."
    exit 1
fi
if [[ "${WITH_DEBUG}" -eq 1 ]]; then
    DEBUG_LOG_DIR="${LOG_DIR}/debug_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${DEBUG_LOG_DIR}"
    log "spawning dump_x2_debug -> ${DEBUG_LOG_DIR}"
    "${PYTHON}" -m gear_sonic.scripts.dump_x2_debug \
        --output-dir "${DEBUG_LOG_DIR}" \
        --duration "${DURATION_S}" \
        >"${LOG_DIR}/debug.log" 2>&1 &
    DEBUG_PID=$!
    CHILD_PIDS+=("${DEBUG_PID}")
fi

# --------------------------------------------------------------------------
# Optional: live MuJoCo viewer (subscribes to the pose stream)
# --------------------------------------------------------------------------

if [[ "${WITH_VIEWER}" -eq 1 ]]; then
    VIEWER_LOG="${LOG_DIR}/viewer_$(date +%Y%m%d_%H%M%S).log"
    log "spawning live MuJoCo viewer -> ${VIEWER_LOG}"
    "${PYTHON}" -m gear_sonic.scripts.view_x2_planner_mujoco \
        --from-zmq "127.0.0.1:${PUB_PORT}" \
        --duration-s "${DURATION_S}" \
        >"${VIEWER_LOG}" 2>&1 &
    VIEWER_PID=$!
    CHILD_PIDS+=("${VIEWER_PID}")
fi

# --------------------------------------------------------------------------
# Wait for the planner to finish (it's the wall-clock keeper)
# --------------------------------------------------------------------------

log "waiting for planner to finish (duration=${DURATION_S}s)..."
wait "${PLANNER_PID}"
PLANNER_RC=$?
log "planner exited rc=${PLANNER_RC}"

# Trap-based cleanup will handle the rest.
exit "${PLANNER_RC}"
