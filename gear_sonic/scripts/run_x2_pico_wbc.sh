#!/usr/bin/env bash
# X2 whole-body Pico teleop launcher (SMPL path, live_pico_smpl_teleop.py).
#
#   ./run_x2_pico_wbc.sh                            # live headset, model from
#                                                   # $DEFAULT_ONNX / $MODEL
#   ./run_x2_pico_wbc.sh --model <checkpoint>       # explicit torch checkpoint
#   ./run_x2_pico_wbc.sh --onnx <graph.onnx>        # explicit fused SMPL ONNX
#   ./run_x2_pico_wbc.sh --tape-replay logs/pico_tapes/<tape>.npz \
#       --headless --seconds 125 --auto-engage      # headset-free tape replay
#
# --model follows run_x2_quest3_planner_stack.sh (--model PATH); the
# MODEL= env var also works, as in sim_onnx_planner.sh. Every other
# argument passes through to live_pico_smpl_teleop.py unchanged
# (--tape-replay / --smpl-replay additionally switch the launcher to
# headset-free mode: .venv instead of .venv_teleop, and no PC Service).
# --model + --onnx together = the driver's shadow-compare (torch drives,
# ONNX shadows on identical obs, per-step diff CSV).
#
# The model is a COMPLETE checkpoint path/string, passed verbatim to the
# driver's --checkpoint. Forms the driver accepts:
#   /path/to/model.pt                        X2-native 3-encoder ckpt
#   frozen-core-smpl:<ckpt.pt>                   composite on the original release
#   frozen-core-smpl:<ckpt.pt>:<release.pt>      composite w/ explicit release
#                                            (v1.1 lineage; LoRA folding +
#                                            heading obs handled in-loader)
#
# Operator controls once running:
#   both grips squeezed 0.5 s  -> engage / disengage (heading re-anchors)
#   SPACE (viewer window)      -> same toggle;  R -> reset robot
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Default model = a fused SMPL-encoder ONNX export of an X2 SONIC checkpoint
# (parity-gated export; in-loop shadow-compare ~1e-05 rad vs torch). This
# STANDALONE path has no shipped default: the shipped set
# (gear_sonic_deploy/models/, MODELS.md) carries the dual-head / pose / token
# graphs, not the fused *_smpl.onnx gate graph, and is driven by the
# deploy-faithful path instead (`sim_onnx_planner.sh --whole-body-teleop`
# with no MODEL= at all). Here, point DEFAULT_ONNX at a fused SMPL ONNX, or
# MODEL / --model at a torch checkpoint. Nothing is assumed on disk.
DEFAULT_ONNX="${DEFAULT_ONNX:-}"

# Resolution order (matches the stack script): --model/--onnx arg > MODEL env
# > default ONNX.
CKPT="${MODEL:-}"
EXPLICIT_MODEL=$([[ -n "${MODEL:-}" ]] && echo 1 || echo 0)
ONNX_MODE=0
TAPE_MODE=0
PASS_ARGS=()
while (( $# )); do
    case "$1" in
        --model)       CKPT="$2"; EXPLICIT_MODEL=1; shift 2 ;;
        --onnx)        ONNX_MODE=1; ONNX_PATH="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --tape-replay) TAPE_MODE=1; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --smpl-replay) TAPE_MODE=1; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        *)             PASS_ARGS+=("$1"); shift ;;
    esac
done

# No model given at all -> run $DEFAULT_ONNX (fused SMPL ONNX); refuse if unset.
if (( ! ONNX_MODE )) && (( ! EXPLICIT_MODEL )); then
    if [[ -z "${DEFAULT_ONNX}" ]]; then
        echo "ERROR: no model. Pass --onnx <fused_smpl.onnx> / --model <ckpt.pt>, or set" >&2
        echo "       DEFAULT_ONNX / MODEL (see MODELS.md for how to obtain X2 SONIC models)." >&2
        echo "       The shipped set needs no model flag on the deploy-faithful path:" >&2
        echo "       ./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop" >&2
        exit 2
    fi
    ONNX_MODE=1
    ONNX_PATH="${DEFAULT_ONNX}"
    PASS_ARGS+=(--onnx "${DEFAULT_ONNX}")
fi

# --onnx alone runs the fused SMPL ONNX export (no torch checkpoint). An
# EXPLICITLY given model alongside --onnx enables the driver's shadow-compare
# (torch drives, ONNX shadows). A torch checkpoint is passed only when the
# user named one — never a default alongside an ONNX (lineage mismatch).
CKPT_ARGS=()
if (( EXPLICIT_MODEL )); then
    CKPT_ARGS=(--checkpoint "${CKPT}")
fi

PICO_PC_SERVICE_DIR="${PICO_PC_SERVICE_DIR:-/opt/apps/roboticsservice}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

# Tape replay runs headset-free in .venv (no SDK needed); live needs the
# SDK, so use .venv_teleop (torch + mujoco + xrobotoolkit_sdk).
if (( TAPE_MODE )); then
    PY="${SONIC_PYTHON:-${REPO_ROOT}/.venv/bin/python}"   # SONIC_PYTHON overrides; default the repo .venv, else python3
    command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3 || true)"
    [[ -n "$PY" ]] || { echo "ERROR: no python interpreter (create ${REPO_ROOT}/.venv or set SONIC_PYTHON=/path/to/python)" >&2; exit 1; }
else
    PY="${REPO_ROOT}/.venv_teleop/bin/python"
    if pgrep -x RoboticsService >/dev/null 2>&1; then
        echo "[pico-wbc] XRoboToolkit PC Service already running"
    else
        if [[ ! -x "${PICO_PC_SERVICE_DIR}/RoboticsServiceProcess" ]]; then
            echo "ERROR: PC Service missing (${PICO_PC_SERVICE_DIR})." >&2
            echo "       Install via install_scripts/install_pico.sh (step 8)." >&2
            exit 1
        fi
        echo "[pico-wbc] starting PC Service -> ${LOG_DIR}/roboticsservice.log"
        LD_LIBRARY_PATH="${PICO_PC_SERVICE_DIR}:${PICO_PC_SERVICE_DIR}/lib:${PICO_PC_SERVICE_DIR}/SDK/x64:${LD_LIBRARY_PATH:-}" \
            nohup "${PICO_PC_SERVICE_DIR}/RoboticsServiceProcess" \
            >"${LOG_DIR}/roboticsservice.log" 2>&1 &
        disown
        for _ in $(seq 1 20); do
            ss -tln 2>/dev/null | grep -q ':60061 ' && break
            sleep 0.5
        done
        if ss -tln 2>/dev/null | grep -q ':60061 '; then
            echo "[pico-wbc] PC Service READY (gRPC :60061)"
        else
            echo "ERROR: PC Service did not open :60061 -- see ${LOG_DIR}/roboticsservice.log" >&2
            exit 1
        fi
    fi
    IP=$(ip -4 addr show scope global 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)
    echo "[pico-wbc] headset: open the XRoboToolkit app, connect to ${IP:-<this-PC-IP>},"
    echo "[pico-wbc]          enter the VR scene, keep the app foregrounded."
fi

if (( ${#CKPT_ARGS[@]} )); then
    echo "[pico-wbc] model: ${CKPT}"
fi
if (( ONNX_MODE )); then
    echo "[pico-wbc] onnx:  ${ONNX_PATH}"
fi
exec "${PY}" gear_sonic/scripts/live_pico_smpl_teleop.py \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
