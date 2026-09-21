#!/bin/bash
# Stage the pico fine-tune inputs onto a training node, md5-verified both sides.
#   ./stage_pico_v8.sh ubuntu@<node-ip> [ssh-port]
# Companion: run_pico_v8_8gpu.sh (launch), sonic_x2_leg5_pico_combined_ft.yaml (config).
#
# Env (all local paths):
#   PICO_FT_PKL       fine-tune pkl from build_pico_finetune_bundle.py (required)
#   PICO_SIDECAR_DIR  its SMPL sidecar dir                             (required)
#   CKPT              warm-start checkpoint model_step_NNNNNN.pt       (required)
#   CFG               config.yaml saved next to that checkpoint       (required)
#   REMOTE_REPO       repo dir on the node, relative to $HOME  (default: GR00T-WholeBodyControl)
#
# What goes up (node-local -- the share, if any, is for checkpoints, not bulk
# reads):
#   $PICO_FT_PKL          -> ~/train_data/               (fine-tune set)
#   $PICO_SIDECAR_DIR/*   -> ~/train_data/smpl_filtered/  (per-clip SMPL pkls;
#                            motion_lib directory mode matches by key, so they
#                            simply join an existing sidecar set)
#   $CKPT                 -> ~/run_pico/                  (warm start;
#                            checkpoints will save next to it)
#   $CFG                  -> alongside, for provenance
# In fallback mode (no base corpus on the node) additionally:
#   STAGE_FALLBACK=1 stages MAIN_PKL (default: the regenerated demo bank
#   gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl) into the node's repo checkout.
set -euo pipefail

NODE=${1:?usage: stage_pico_v8.sh ubuntu@<node-ip> [ssh-port]}
PORT=${2:-22}
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new "$NODE")
RS=(rsync -a --info=progress2 -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE_REPO=${REMOTE_REPO:-GR00T-WholeBodyControl}
V8="${PICO_FT_PKL:?set PICO_FT_PKL=<fine-tune pkl>}"
SIDE="${PICO_SIDECAR_DIR:?set PICO_SIDECAR_DIR=<SMPL sidecar dir>}"
CKPT="${CKPT:?set CKPT=<model_step_NNNNNN.pt>}"
CFG="${CFG:?set CFG=<config.yaml next to the checkpoint>}"

for f in "$V8" "$CKPT" "$CFG"; do
  [[ -f "$f" ]] || { echo "FATAL: missing local file: $f" >&2; exit 1; }
done
[[ -d "$SIDE" ]] || { echo "FATAL: missing $SIDE" >&2; exit 1; }

"${SSH[@]}" "mkdir -p ~/train_data ~/train_data/smpl_filtered ~/run_pico"

"${RS[@]}" "$V8" "$NODE:train_data/"
"${RS[@]}" "$SIDE/" "$NODE:train_data/smpl_filtered/"
"${RS[@]}" "$CKPT" "$CFG" "$NODE:run_pico/"

if [[ "${STAGE_FALLBACK:-0}" == "1" ]]; then
  EF="${MAIN_PKL:-$REPO/gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl}"
  [[ -f "$EF" ]] || { echo "FATAL: missing $EF" >&2; exit 1; }
  "${SSH[@]}" "mkdir -p ~/$REMOTE_REPO/gear_sonic/data/motions"
  "${RS[@]}" "$EF" "$NODE:$REMOTE_REPO/gear_sonic/data/motions/"
fi

echo "=== md5 verification ==="
locsum=$(md5sum "$V8" "$CKPT" | awk '{print $1}' | paste -sd' ')
remsum=$("${SSH[@]}" "md5sum train_data/$(basename "$V8") run_pico/$(basename "$CKPT") | awk '{print \$1}' | paste -sd' '")
echo "local : $locsum"
echo "remote: $remsum"
[[ "$locsum" == "$remsum" ]] || { echo "FATAL: md5 mismatch" >&2; exit 1; }
n_side=$("${SSH[@]}" "ls train_data/smpl_filtered/*.pkl 2>/dev/null | wc -l")
echo "sidecars on node: $n_side (local: $(ls "$SIDE"/*.pkl 2>/dev/null | wc -l))"
echo "staging OK"
