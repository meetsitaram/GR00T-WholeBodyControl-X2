#!/bin/bash
# SONIC big run -- N nodes x 8 GPUs under accelerate (multi-machine).
# Run this on EVERY node with a distinct NODE_RANK; rank 0 is the rendezvous host.
#
#   NODE_RANK=0 MASTER=<rank-0 private IP> bash launch_bigrun.sh    # on node 0
#   NODE_RANK=1 MASTER=<rank-0 private IP> bash launch_bigrun.sh    # on node 1 ...
#
# Env:
#   MASTER           rank-0 node private IP (required)
#   NODE_RANK        this node's rank (required)
#   NNODES           number of nodes                       (default: 4)
#   EXP              Hydra +exp= path                      (default: manager/universal_token/all_modes/sonic_x2_ultra)
#   CKPT             optional warm-start checkpoint (+checkpoint=)
#   STEPS            num_learning_iterations, ABSOLUTE     (default: 20000)
#   NUM_ENVS         envs per GPU                          (default: 12288; 4096 on a 32 GB card)
#   ISAACLAB_PYTHON  python of the training env           (default: $HOME/miniconda3/envs/env_isaaclab/bin/python)
#   MOTION_FILE      base corpus (read by the exp config; default = demo bank)
#   LOG              log file                              (default: $HOME/bigrun.log)
#   --dry-run        print the accelerate command and exit; --help prints this.
#
# Checkpoints land every save_interval steps (default 500), so quality-vs-step
# can be evaluated over the next days and the run paused or resumed at will.
set -u
DRY_RUN=
case "${1:-}" in
  -h|--help) sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
  --dry-run) DRY_RUN=1;;
esac
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-${DRY_RUN:+0}}
NODE_RANK=${NODE_RANK:?set NODE_RANK per node}
MASTER=${MASTER:-${DRY_RUN:+127.0.0.1}}
MASTER=${MASTER:?set MASTER=<rank-0 node private IP>}
STEPS=${STEPS:-20000}
NUM_ENVS=${NUM_ENVS:-12288}   # H100/H200 value; 4096 is the upstream / 32 GB-card default
EXP=${EXP:-manager/universal_token/all_modes/sonic_x2_ultra}
CKPT=${CKPT:-}
LOG=${LOG:-$HOME/bigrun.log}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/env_isaaclab/bin/python}
ACCELERATE=${ACCELERATE:-$(dirname "$ISAACLAB_PYTHON")/accelerate}

export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$MASTER MASTER_PORT=29500
cd "$REPO_ROOT" || exit 1

CMD=("$ACCELERATE" launch
  --multi_gpu --num_machines "$NNODES" --num_processes $((NNODES * 8))
  --machine_rank "$NODE_RANK" --main_process_ip "$MASTER" --main_process_port 29500
  gear_sonic/train_agent_trl.py
  "+exp=$EXP"
  ${CKPT:+"+checkpoint=$CKPT"}
  headless=True "num_envs=$NUM_ENVS"
  "++algo.config.num_learning_iterations=$STEPS")

if [ "${DRY_RUN:-}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec > "$LOG" 2>&1
echo "[bigrun] node_rank=$NODE_RANK/$NNODES master=$MASTER steps=$STEPS envs=$NUM_ENVS exp=$EXP $(date -u +%H:%M:%S)"
echo "[bigrun] repo=$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
"${CMD[@]}"
echo "[bigrun] exit=$? $(date -u +%H:%M:%S)"
