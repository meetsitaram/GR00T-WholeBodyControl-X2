#!/bin/bash
# Stage the SONIC training payload to node-local disk so training does not
# read 100k+ small files off a shared network mount from 32 GPUs.
# Nodes WITH the shared mount copy from it; nodes without it pull over ssh
# from a node that already staged.
#
# Env:
#   SHARE           shared mount root                (default: /mnt/checkpoints)
#   SRC_CORPUS      corpus dir on the share          (default: $SHARE/corpus)
#   SRC_SMPL        SMPL sidecar dir on the share    (default: $SHARE/data/smpl_filtered)
#   SRC_CKPT        warm-start checkpoint dir        (default: $SHARE/checkpoints/current)
#   STAGE_SRC_HOST  user@host to rsync from when the share is absent (required then)
#   SSH_KEY         key for that pull               (default: ~/.ssh/id_ed25519)
#   DEST            node-local destination          (default: $HOME/train_data)
set -u
case "${1:-}" in
  -h|--help) sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
esac
LOG=${LOG:-$HOME/stage_local.log}
exec > "$LOG" 2>&1
SHARE=${SHARE:-/mnt/checkpoints}
DEST=${DEST:-$HOME/train_data}
mkdir -p "$DEST"
echo "[stage] start $(date -u +%H:%M:%S) on $(hostname)"

if [ -d "$SHARE" ]; then
  SRC_CORPUS=${SRC_CORPUS:-$SHARE/corpus}
  SRC_SMPL=${SRC_SMPL:-$SHARE/data/smpl_filtered}
  SRC_CKPT=${SRC_CKPT:-$SHARE/checkpoints/current}
  rsync -a "$SRC_CORPUS/"  "$DEST/corpus/"
  rsync -a "$SRC_SMPL/"    "$DEST/smpl_filtered/"
  rsync -a "$SRC_CKPT/"    "$DEST/ckpt/"
else
  : "${STAGE_SRC_HOST:?no $SHARE mount; set STAGE_SRC_HOST=user@<node that already staged>}"
  R="ssh -i ${SSH_KEY:-$HOME/.ssh/id_ed25519} -o StrictHostKeyChecking=accept-new"
  rsync -a -e "$R" "$STAGE_SRC_HOST:train_data/corpus/"        "$DEST/corpus/"
  rsync -a -e "$R" "$STAGE_SRC_HOST:train_data/smpl_filtered/" "$DEST/smpl_filtered/"
  rsync -a -e "$R" "$STAGE_SRC_HOST:train_data/ckpt/"          "$DEST/ckpt/"
fi

echo "[stage] DONE $(date -u +%H:%M:%S)"
echo "  corpus:   $(find "$DEST/corpus" -name '*.pkl' | wc -l) pkls"
echo "  sidecars: $(find "$DEST/smpl_filtered" -name '*.pkl' | wc -l) pkls"
echo "  ckpt:     $(ls "$DEST/ckpt" | tr '\n' ' ')"
