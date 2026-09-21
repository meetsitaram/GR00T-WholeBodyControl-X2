#!/bin/bash
# Local single-GPU fine-tune of the X2 Ultra SONIC policy on a demo motion
# corpus. Continues from a checkpoint (CHECKPOINT, required) and adds another
# 4k PPO iterations.
#
# Sized for a 32 GB card (e.g. RTX 5090):
#   - num_envs       = 3072
#   - num_iterations = 4000
#   - num_processes  = 1 (single-GPU)
#   - wandb          = True
#
# Env:
#   CHECKPOINT   warm-start checkpoint model_step_NNNNNN.pt (required)
#   MOTION_FILE      motion-lib pkl or per-clip dir (default: the shipped chores set
#                    gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl)
#   SMPL_MOTION_DIR  SMPL sidecar dir (default: .../x2_pico_chores/smpl_sidecars)
#   EXP_NAME     Hydra +exp= leaf under manager/universal_token/all_modes (default: sonic_x2_ultra)
#   --dry-run    print the launch command (via run_smoke_8gpu.sh) and exit
#
# Usage (detached, no tmux needed):
#
#   CHECKPOINT=/path/model_step_NNNNNN.pt \
#   setsid nohup bash gear_sonic/scripts/run_local_finetune_demo_v2.sh \
#       </dev/null >/dev/null 2>&1 &
#   echo $! > ~/sonic_demo_v2.pid
#   tail -f ~/sonic_demo_v2.log
#
# The training script handles its own logging via tee to LOG_FILE inside
# run_smoke_8gpu.sh, so don't redirect again from outside (would double
# every line in the log file).
#
# Stop with: kill $(cat ~/sonic_demo_v2.pid)

set -euo pipefail

DRY_RUN=
case "${1:-}" in
  -h|--help) sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
  --dry-run) DRY_RUN=1;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LAUNCH_TS=$(date +%Y%m%d_%H%M%S)

CHECKPOINT="${CHECKPOINT:-${DRY_RUN:+<model_step_NNNNNN.pt>}}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=<warm-start model_step_NNNNNN.pt>}"

if [[ "${DRY_RUN:-}" != 1 && ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: warm-start checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

export MOTION_FILE="${MOTION_FILE:-gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl}"
export SMPL_MOTION_DIR="${SMPL_MOTION_DIR:-gear_sonic/data/motions/x2_pico_chores/smpl_sidecars}"
export EXP_NAME="${EXP_NAME:-sonic_x2_ultra}"
export DRY_RUN
export NUM_PROCESSES=1
export NUM_ENVS=3072
export NUM_ITERS=4000
export USE_WANDB=True
export EXTRA_FLAGS="+checkpoint=$CHECKPOINT"
export LOG_FILE="$HOME/sonic_demo_v2_${LAUNCH_TS}.log"

ln -sfn "$LOG_FILE" "$HOME/sonic_demo_v2.log"

exec bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
