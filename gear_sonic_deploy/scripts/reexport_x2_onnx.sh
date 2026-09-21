#!/usr/bin/env bash
# Re-export the fused X2 G1 deploy ONNX from a PyTorch checkpoint.
#
# Wraps the two Python entrypoints needed to produce a deploy-ready
# x2_sonic_16k.onnx:
#
#   1. gear_sonic.scripts.dump_isaaclab_step0   -- captures a step-0 ground
#      truth (encoder input + decoder action mean) from IsaacLab using the
#      SAME checkpoint we want to export. Without this, the validation step
#      below cannot tell "ONNX export is broken" apart from "you exported the
#      wrong checkpoint".
#
#   2. gear_sonic.scripts.reexport_x2_g1_onnx   -- builds a fused
#      encoder+FSQ+decoder torch.nn.Module that exactly mirrors training's
#      forward path, exports with FSQ-friendly tracing options, and validates
#      the freshly-exported ONNX against the dump from step 1. Refuses to
#      promote the artefact unless max|onnx - pt| < --max-action-diff
#      (default 1e-3 rad).
#
# We add this wrapper because we got bitten once by deploying an ONNX whose
# encoder input layout disagreed with the C++ deploy code, producing actions
# that diverged from the .pt by up to 9.77 rad per joint. The fused exporter
# fixes that by exporting a single (B, 1670) -> (B, 31) module whose layout
# is locked to dump_isaaclab_step0's "encoder_input_for_mlp_view".
#
# Usage:
#     ./gear_sonic_deploy/scripts/reexport_x2_onnx.sh \
#         <run-dir> [output.onnx] [extra hydra overrides...]
#
# Examples:
#     # Re-export from last.pt in the run dir, write to the deploy slot
#     ./gear_sonic_deploy/scripts/reexport_x2_onnx.sh \
#         $CKPT_ROOT/run-20260420_083925
#
#     # Re-export to a side path and force a specific checkpoint
#     ./gear_sonic_deploy/scripts/reexport_x2_onnx.sh \
#         $CKPT_ROOT/run-20260420_083925 \
#         /tmp/x2_sonic_16k_test.onnx \
#         +checkpoint=$CKPT_ROOT/run-20260420_083925/model_step_014000.pt
#
# Notes:
#   * Must be run from the repo root with the env_isaaclab conda env on PATH.
#     The script will activate env_isaaclab itself if not already active.
#   * Both inner Python entrypoints are imported as -m modules, so PYTHONPATH
#     is unaffected.
#   * The previous deploy ONNX is preserved at <output>.broken_export.<ts>
#     before being overwritten -- never silently destroys the running model.

set -euo pipefail

# --- defaults --------------------------------------------------------------
DEFAULT_OUTPUT="gear_sonic_deploy/models/x2_sonic_16k.onnx"
DEFAULT_DUMP="/tmp/x2_step0_isaaclab_lastpt.pt"
DEFAULT_CONDA_ENV="env_isaaclab"

usage() {
    cat <<EOF
Usage: $(basename "$0") <run-dir> [output.onnx] [extra hydra overrides...]

Required:
  <run-dir>        Training run directory containing last.pt (or model_step_*.pt)

Optional:
  output.onnx      Where to write the new ONNX
                   (default: ${DEFAULT_OUTPUT})

Extra args after output.onnx are forwarded to BOTH
gear_sonic.scripts.dump_isaaclab_step0 and
gear_sonic.scripts.reexport_x2_g1_onnx as Hydra overrides.
The most useful one is:
  +checkpoint=/abs/path/to/model_step_NNNN.pt   pin to a specific checkpoint

Environment:
  CONDA_ENV        conda env to activate (default: ${DEFAULT_CONDA_ENV})
  DUMP_PATH        where to write the IsaacLab step-0 dump
                   (default: ${DEFAULT_DUMP})
  MAX_DIFF         max allowed |onnx - pt| in radians (default: 1e-3)
  SKIP_DUMP=1      skip the dump step (assumes DUMP_PATH already exists for
                   this exact checkpoint -- only use if you know what you're
                   doing)

Outputs:
  <output>                                  the validated, deploy-ready ONNX
  <output>.broken_export.YYYYMMDD-HHMMSS    backup of the previous ONNX
EOF
}

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

RUN_DIR="$1"
shift

if [[ $# -gt 0 && "$1" != +* && "$1" != ++* ]]; then
    OUTPUT="$1"
    shift
else
    OUTPUT="${DEFAULT_OUTPUT}"
fi

EXTRA_HYDRA_ARGS=("$@")
DUMP_PATH="${DUMP_PATH:-${DEFAULT_DUMP}}"
MAX_DIFF="${MAX_DIFF:-1e-3}"
CONDA_ENV="${CONDA_ENV:-${DEFAULT_CONDA_ENV}}"

# --- sanity checks ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: run-dir does not exist: ${RUN_DIR}" >&2
    exit 1
fi

if [[ ! -f "${RUN_DIR}/last.pt" ]] && \
   ! ls "${RUN_DIR}"/model_step_*.pt >/dev/null 2>&1 && \
   ! printf '%s\n' "${EXTRA_HYDRA_ARGS[@]}" | grep -q '^+\?checkpoint='; then
    echo "ERROR: no last.pt or model_step_*.pt in ${RUN_DIR} and no +checkpoint= override" >&2
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${OUTPUT}")"

# Ensure conda is sourced and the right env is active. We allow the user to
# already be inside the env (CONDA_DEFAULT_ENV check) to avoid double-activate.
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if [[ -z "${CONDA_EXE:-}" ]]; then
        echo "ERROR: conda not found on PATH. Activate ${CONDA_ENV} manually first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

echo "================================================================"
echo "[reexport_x2_onnx] repo root  : ${REPO_ROOT}"
echo "[reexport_x2_onnx] run dir    : ${RUN_DIR}"
echo "[reexport_x2_onnx] output     : ${OUTPUT}"
echo "[reexport_x2_onnx] dump path  : ${DUMP_PATH}"
echo "[reexport_x2_onnx] max diff   : ${MAX_DIFF} rad"
echo "[reexport_x2_onnx] conda env  : ${CONDA_ENV}"
if [[ ${#EXTRA_HYDRA_ARGS[@]} -gt 0 ]]; then
    echo "[reexport_x2_onnx] extra args : ${EXTRA_HYDRA_ARGS[*]}"
fi
echo "================================================================"

# --- step 1: capture IsaacLab step-0 ground truth --------------------------
if [[ "${SKIP_DUMP:-0}" == "1" ]]; then
    if [[ ! -f "${DUMP_PATH}" ]]; then
        echo "ERROR: SKIP_DUMP=1 but ${DUMP_PATH} does not exist" >&2
        exit 1
    fi
    echo "[reexport_x2_onnx] SKIP_DUMP=1 -- using existing ${DUMP_PATH}"
else
    echo "[reexport_x2_onnx] Step 1/2: capturing IsaacLab step-0 dump..."
    python -m gear_sonic.scripts.dump_isaaclab_step0 \
        +checkpoint="${RUN_DIR}/last.pt" \
        +headless=True \
        ++num_envs=1 \
        ++eval_callbacks=im_eval \
        ++run_eval_loop=true \
        ++max_render_steps=1 \
        ++dump_path="${DUMP_PATH}" \
        "${EXTRA_HYDRA_ARGS[@]}"
fi

if [[ ! -f "${DUMP_PATH}" ]]; then
    echo "ERROR: dump step did not produce ${DUMP_PATH}" >&2
    exit 1
fi

# --- step 2: re-export and validate ---------------------------------------
# Back up the existing deploy ONNX before we touch anything.
if [[ -f "${OUTPUT}" ]]; then
    BACKUP="${OUTPUT}.broken_export.$(date +%Y%m%d-%H%M%S)"
    echo "[reexport_x2_onnx] Backing up existing ${OUTPUT} -> ${BACKUP}"
    cp -p "${OUTPUT}" "${BACKUP}"
fi

echo "[reexport_x2_onnx] Step 2/2: exporting + validating..."
python -m gear_sonic.scripts.reexport_x2_g1_onnx \
    --run-dir "${RUN_DIR}" \
    --output "${OUTPUT}" \
    --dump "${DUMP_PATH}" \
    --max-action-diff "${MAX_DIFF}" \
    ++num_envs=1 \
    ++eval_callbacks=im_eval \
    "${EXTRA_HYDRA_ARGS[@]}"

echo "================================================================"
echo "[reexport_x2_onnx] DONE."
echo "  New ONNX:  ${OUTPUT}"
sha256sum "${OUTPUT}" || true
echo "================================================================"
