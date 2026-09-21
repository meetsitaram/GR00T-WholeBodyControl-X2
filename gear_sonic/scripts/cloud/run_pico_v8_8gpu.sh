#!/bin/bash
# PICO FINE-TUNE LEG -- single 8-GPU node, resume from an incumbent checkpoint.
# Companion config: sonic_x2_leg5_pico_combined_ft.yaml; staging: stage_pico_v8.sh.
#
# ITERATIONS ARE RELATIVE ON RESUME. ppo_trainer runs
# `for batch_idx in range(1, num_total_batches+1)` fresh each launch while
# global_step continues from the checkpoint -- STEPS=3000 on a 14000-step
# checkpoint trains 14000 -> 17000.
#
# Env:
#   CKPT             warm-start checkpoint model_step_NNNNNN.pt (required)
#   X2_FINETUNE_PKL  fine-tune pkl (build_pico_finetune_bundle.py) (required)
#   SMPL_MOTION_DIR  SMPL sidecar dir            (default: gear_sonic/data/motions/x2_pico_chores/smpl_sidecars)
#   MOTION_FILE      base corpus pkl/dir; MODE=faithful requires it, MODE=fallback
#                    defaults to the shipped chores set gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl
#   MODE             faithful | fallback                         (default: faithful)
#   EXP              Hydra +exp= path   (default: manager/universal_token/all_modes/sonic_x2_leg5_pico_combined_ft)
#   STEPS            iterations to add (RELATIVE)                (default: 3000)
#   NUM_ENVS         envs per GPU                                (default: 12288; 4096 on a 32 GB card)
#   EXPECT_CLIPS     "<total> <pico>" fingerprint of the fine-tune pkl; asserted when set
#   ISAACLAB_PYTHON  python of the training env (default: $HOME/miniconda3/envs/env_isaaclab/bin/python)
#   LOG              log file                                    (default: $HOME/pico_ft.log)
#   --dry-run        print the accelerate command and exit; --help prints this.
#
# MODE=fallback CAVEAT: with the demo bank as the base corpus the breadth half
# of training sees a different distribution than the checkpoint's original
# corpus -- a breadth-decay eval against a held-out set is mandatory before
# trusting any comparison to the incumbent.
set -euo pipefail
DRY_RUN=
case "${1:-}" in
  -h|--help) sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
  --dry-run) DRY_RUN=1;;
esac

MODE=${MODE:-faithful}
STEPS=${STEPS:-3000}          # RELATIVE
NUM_ENVS=${NUM_ENVS:-12288}   # H100/H200 value; 4096 on a 32 GB card
EXP=${EXP:-manager/universal_token/all_modes/sonic_x2_leg5_pico_combined_ft}
LOG=${LOG:-$HOME/pico_ft.log}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/env_isaaclab/bin/python}
ACCELERATE=${ACCELERATE:-$(dirname "$ISAACLAB_PYTHON")/accelerate}
if [[ "${DRY_RUN:-}" == 1 ]]; then
  CKPT=${CKPT:-<model_step_NNNNNN.pt>}; X2_FINETUNE_PKL=${X2_FINETUNE_PKL:-<ft.pkl>}
  SMPL_MOTION_DIR=${SMPL_MOTION_DIR:-<smpl_dir>}; MOTION_FILE=${MOTION_FILE:-<corpus.pkl>}
fi
CKPT=${CKPT:?set CKPT=<model_step_NNNNNN.pt>}
FT_PKL=${X2_FINETUNE_PKL:?set X2_FINETUNE_PKL=<fine-tune pkl>}
SMPL_DIR=${SMPL_MOTION_DIR:-$REPO_ROOT/gear_sonic/data/motions/x2_pico_chores/smpl_sidecars}

cd "$REPO_ROOT"

EXTRA_ARGS=()
if [[ "$MODE" == "faithful" ]]; then
  MAIN_PKL=${MOTION_FILE:?MODE=faithful: set MOTION_FILE=<base corpus> (or MODE=fallback)}
  REQUIRED=("$CKPT" "$FT_PKL")
elif [[ "$MODE" == "fallback" ]]; then
  MAIN_PKL=${MOTION_FILE:-$REPO_ROOT/gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl}
  REQUIRED=("$CKPT" "$FT_PKL")
  # The demo bank is already curated; an exclusion list from the original
  # corpus would match none of its keys and the loud-fail guard would refuse
  # to launch -- clear it rather than let it kill the run.
  EXTRA_ARGS+=("++manager_env.commands.motion.motion_lib_cfg.remove_motion_keys_file=null")
else
  echo "FATAL: MODE must be faithful|fallback" >&2; exit 1
fi
export MOTION_FILE="$MAIN_PKL" SMPL_MOTION_DIR="$SMPL_DIR" X2_FINETUNE_PKL="$FT_PKL"

CMD=("$ACCELERATE" launch
  --num_processes=8
  gear_sonic/train_agent_trl.py
  "+exp=$EXP"
  "+checkpoint=$CKPT"
  +resume=true
  headless=True "num_envs=$NUM_ENVS"
  ++use_wandb=True
  "++algo.config.num_learning_iterations=$STEPS"
  "${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN:-}" == 1 ]]; then
  echo "[dry-run] MOTION_FILE=$MOTION_FILE SMPL_MOTION_DIR=$SMPL_MOTION_DIR X2_FINETUNE_PKL=$X2_FINETUNE_PKL"
  printf '[dry-run] '; printf '%q ' "${CMD[@]}"; echo; exit 0
fi

for f in "${REQUIRED[@]}" "$MAIN_PKL"; do
  [[ -e "$f" ]] || { echo "FATAL: missing required file: $f" >&2; exit 1; }
done
[[ -d "$SMPL_DIR" ]] || { echo "FATAL: $SMPL_DIR missing" >&2; exit 1; }

# fine-tune corpus fingerprint (see build_pico_finetune_bundle.py)
python3 - "$FT_PKL" "${EXPECT_CLIPS:-}" <<'PY'
import sys, joblib
d = joblib.load(sys.argv[1])
n = len(d); pico = sum(1 for k in d if k.startswith("pico_"))
if sys.argv[2]:
    want = tuple(int(x) for x in sys.argv[2].split())
    assert (n, pico) == want, f"fine-tune pkl fingerprint mismatch: {n} clips / {pico} pico != {want}"
print(f"[pico-ft] corpus ok: {n} clips ({pico} pico)")
PY

export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== pico fine-tune leg ($MODE) ==="
echo "  ckpt : $CKPT"
echo "  steps: +$STEPS (relative)"
echo "  envs : $NUM_ENVS    log: $LOG"

# checkpoints save NEXT TO the warm-start ckpt (experiment_dir = ckpt parent);
# wandb starts a fresh run with the x-axis continuing at the checkpoint step.
nohup "${CMD[@]}" > "$LOG" 2>&1 &
echo "launched accelerate PID $!  ->  tail -f $LOG"
