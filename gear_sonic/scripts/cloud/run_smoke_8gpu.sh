#!/bin/bash
# 8-GPU smoke test for the X2 Ultra training stack.
#
# Verifies the full pipeline (Isaac Sim init -> env build -> Hydra config ->
# accelerate distributed launch -> PPO loop) on every GPU of a cloud node,
# without committing to a multi-hour real run.
#
# Defaults are tuned for an 8x H200 (or H100 80GB) SXM node:
#   - 200 PPO iterations  (~3 min wall-clock on H200)
#   - 4096 environments per GPU (32K total)
#   - W&B disabled
#   - motion library = MOTION_FILE PLUS the SMPL sidecar directory SMPL_MOTION_DIR;
#     both default to the shipped X2 Pico chores reference set
#     (gear_sonic/data/motions/x2_pico_chores/, 14 paired clips). The smoke runs the full 3-encoder config; it
#     refuses to start when sidecars are missing, zero-filled, in the wrong frame
#     or on the wrong timebase (gear_sonic/scripts/check_smpl_sidecars.py +
#     gear_sonic/scripts/cloud/preflight_train.py run first, fail-closed).
#
# Usage (run on the cloud node, from the repo root):
#
#   # optional: your own paired corpus + SMPL sidecars, e.g. from your Pico tapes:
#   python gear_sonic/data_process/build_pico_finetune_bundle.py --corpus-dir ... \
#       --tape-dir ... --base-pkl ... --out-pkl x2_ft.pkl --sidecar-dir smpl_sidecars/
#   export MOTION_FILE=x2_ft.pkl SMPL_MOTION_DIR=smpl_sidecars/
#
#   # launch the smoke (detached, in tmux):
#   tmux new -d -s smoke "bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh"
#   tmux a -t smoke         # attach to watch
#   tail -f ~/smoke.log     # ...or tail the log file
#
# Override knobs:
#   NUM_PROCESSES   number of GPUs                      (default: 8)
#   NUM_ENVS        envs per GPU                        (default: 4096)
#   NUM_ITERS       PPO iterations                      (default: 200)
#   MOTION_FILE     motion-lib PKL or per-clip dir      (default: gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl)
#   SMPL_MOTION_DIR SMPL sidecar dir (<key>.pkl per clip)  (default: gear_sonic/data/motions/x2_pico_chores/smpl_sidecars)
#   SKIP_GATES=1    skip the sidecar/preflight gates (debugging only)
#   USE_WANDB       True/False                          (default: False)
#   LOG_FILE        where to tee stdout                 (default: ~/smoke.log)
#   EXP_NAME        Hydra +exp= leaf                    (default: sonic_x2_ultra_smoke)
#                   Resolved as +exp=manager/universal_token/all_modes/$EXP_NAME
#   EXTRA_FLAGS     additional Hydra flags appended raw (default: empty)
#                   e.g. EXTRA_FLAGS="+checkpoint=/path/to/model_step_NNNNNN.pt"
#                        EXTRA_FLAGS="++use_wandb=True ++algo.config.lr=5e-5"
#   ISAACLAB_PYTHON python of the training env; when set, its sibling
#                   `accelerate` is used and no conda activation happens
#                   (default: activate conda env $CONDA_ENV, default env_isaaclab)
#   DRY_RUN=1 / --dry-run   print the accelerate command and exit (no GPU, no network)
#   --help                  print this header

set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
  --dry-run) DRY_RUN=1;;
esac
DRY_RUN=${DRY_RUN:-0}

NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_ENVS=${NUM_ENVS:-4096}
NUM_ITERS=${NUM_ITERS:-200}
MOTION_FILE=${MOTION_FILE:-gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl}
SMPL_MOTION_DIR=${SMPL_MOTION_DIR:-gear_sonic/data/motions/x2_pico_chores/smpl_sidecars}
USE_WANDB=${USE_WANDB:-False}
LOG_FILE=${LOG_FILE:-$HOME/smoke.log}
EXP_NAME=${EXP_NAME:-sonic_x2_ultra_smoke}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

# Repo root is three levels up from this script (gear_sonic/scripts/cloud/...).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${ISAACLAB_PYTHON:-}" ]]; then
  ACCELERATE="$(dirname "$ISAACLAB_PYTHON")/accelerate"
else
  ACCELERATE=accelerate
fi

# shellcheck disable=SC2086  # EXTRA_FLAGS is intentionally word-split.
CMD=("$ACCELERATE" launch "--num_processes=$NUM_PROCESSES"
  gear_sonic/train_agent_trl.py
  --config-name=base
  "+exp=manager/universal_token/all_modes/$EXP_NAME"
  "++num_envs=$NUM_ENVS"
  ++headless=True
  "++use_wandb=$USE_WANDB"
  "++algo.config.num_learning_iterations=$NUM_ITERS"
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION_FILE"
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL_MOTION_DIR"
  $EXTRA_FLAGS)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] cd $REPO_ROOT"
  printf '[dry-run] '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date) === SMOKE START"
echo "  num_processes : $NUM_PROCESSES"
echo "  num_envs/proc : $NUM_ENVS"
echo "  iterations    : $NUM_ITERS"
echo "  motion_file   : $MOTION_FILE"
echo "  smpl sidecars : ${SMPL_MOTION_DIR:-<UNSET>}"
echo "  use_wandb     : $USE_WANDB"
echo "  exp_name      : $EXP_NAME"
echo "  extra_flags   : ${EXTRA_FLAGS:-<none>}"
echo "  repo root     : $REPO_ROOT"

if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
  # Activate the training env (as installed by bootstrap_fresh_node.sh).
  CONDA_PREFIX_DIR=${CONDA_PREFIX_DIR:-$HOME/miniconda3}
  # shellcheck disable=SC1091
  source "$CONDA_PREFIX_DIR/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-env_isaaclab}"
fi

if [[ ! -e "$MOTION_FILE" ]]; then
  echo "ERROR: motion file not found: $MOTION_FILE"
  echo "       the shipped default is gear_sonic/data/motions/x2_pico_chores/ (git lfs pull)"
  echo "       or set MOTION_FILE=/path/to/your/corpus.pkl"
  exit 1
fi
if [[ -z "$SMPL_MOTION_DIR" || ! -d "$SMPL_MOTION_DIR" ]]; then
  echo "ERROR: SMPL_MOTION_DIR is unset or not a directory (${SMPL_MOTION_DIR:-<unset>})."
  echo "       The smoke trains the SMPL encoder too; it needs one <key>.pkl sidecar per clip"
  echo "       (build_pico_finetune_bundle.py --sidecar-dir ...). No zero-filled fallback exists."
  exit 1
fi
if [[ "${SKIP_GATES:-0}" != "1" ]]; then
  PYBIN="${ISAACLAB_PYTHON:-python}"
  echo "=== gate 1/2: SMPL sidecar frame/timebase check ==="
  "$PYBIN" gear_sonic/scripts/check_smpl_sidecars.py "$SMPL_MOTION_DIR" \
      $([[ -f "$MOTION_FILE" ]] && echo --motion "$MOTION_FILE") || { echo "ERROR: sidecar gate FAILED -- not launching."; exit 2; }
  echo "=== gate 2/2: config + corpus + SMPL coverage preflight ==="
  MOTION_FILE="$MOTION_FILE" SMPL_MOTION_DIR="$SMPL_MOTION_DIR" "$PYBIN" gear_sonic/scripts/cloud/preflight_train.py \
      "+exp=manager/universal_token/all_modes/$EXP_NAME" \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION_FILE" \
      "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL_MOTION_DIR" \
      || { echo "ERROR: preflight FAILED -- not launching."; exit 3; }
fi

# Mitigate fragmentation OOMs and pre-accept Omniverse EULAs (no-op if already
# accepted; safe to set every launch).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

"${CMD[@]}"

echo "=== $(date) === SMOKE DONE"
