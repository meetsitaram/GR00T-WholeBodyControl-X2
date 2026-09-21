#!/usr/bin/env bash
# THE sanctioned path for pushing planner changes to the robot (PC2).
# Anything else (bare scp/rsync) bypasses the safety gates -- don't.
#
#   ./x2_pc2/push_to_pc2.sh [--manifest FILE] --pc2 user@host [--dry-run]
#
# The PC2 target has NO default: pass --pc2 run@<PC2_IP> or export
# PC2_HOST=<PC2_IP> (PC2_USER defaults to 'run'). Model lines in the
# manifests use ${X2_MODELS}/<name>.onnx -- export X2_MODELS to the
# directory holding your exported ONNX files (see MODELS.md).
#
# Gates, in order -- the push ABORTS at the first failure:
#   1. PREFLIGHT   preflight_planner.py (e-stop suite, stick margins,
#                  primitive bounds, serve-loop wire regression) must PASS.
#   2. COMMITTED   every local file in the manifest must have NO uncommitted
#                  diff ("if we are shipping any code changes, lets make
#                  sure they are all committed" -- operator 2026-08-09).
#   3. BACKUP      current PC2 copies -> backups/<ts>/ with an md5 MANIFEST
#                  before anything is overwritten.
#   4. VERIFY      md5 of every pushed file compared PC2-side vs local;
#                  any mismatch is a loud failure.
#
# This script NEVER restarts anything on PC2. Restarts happen at the robot,
# by the operator, after the ignition checklist -- never remotely
# (feedback_no_robot_commands / sonic kill safety).
#
# Manifest format (default: x2_pc2/push_manifest.txt), one entry per line:
#   <local path relative to repo root><TAB or spaces><absolute PC2 path>
# Lines starting with # are comments. ${VAR} references in either column
# are expanded from the environment (X2_MODELS, PC2_PREFIX, HOME, ...);
# an unset variable aborts the push before any PC2 contact.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MANIFEST="x2_pc2/push_manifest.txt"
PC2="${PC2_HOST:+${PC2_USER:-run}@${PC2_HOST}}"
PC2_PREFIX="${PC2_PREFIX:-/home/run/gear-sonic}"
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --pc2)      PC2="$2"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$PC2" ]] || { echo "ABORT: no PC2 target -- pass --pc2 run@<PC2_IP> or export PC2_HOST=<PC2_IP>" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "ABORT: manifest $MANIFEST not found" >&2; exit 1; }
mapfile -t RAW_ENTRIES < <(grep -vE '^\s*(#|$)' "$MANIFEST")
[[ ${#RAW_ENTRIES[@]} -gt 0 ]] || { echo "ABORT: manifest is empty" >&2; exit 1; }
# Expand ${VAR} placeholders (X2_MODELS, PC2_PREFIX, HOME, ...). Unset -> abort.
ENTRIES=()
for e in "${RAW_ENTRIES[@]}"; do
  for v in $(grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' <<<"$e" | tr -d '${}' | sort -u); do
    [[ -n "${!v:-}" ]] || { echo "ABORT: manifest line references \${$v} but it is unset: $e" >&2; exit 2; }
  done
  for v in $(grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' <<<"$e" | tr -d '${}' | sort -u); do
    e="${e//\$\{${v}\}/${!v}}"
  done
  ENTRIES+=("$e")
done

echo "== push_to_pc2: ${#ENTRIES[@]} files -> $PC2 (manifest: $MANIFEST)"

# ---- gate 1: preflight ----------------------------------------------------
echo "== GATE 1: preflight_planner.py"
REPO_PY="${REPO_PYTHON:-.venv/bin/python}"
[[ -x "$REPO_PY" ]] || REPO_PY="$(command -v python3)"
if ! "$REPO_PY" gear_sonic/scripts/preflight_planner.py; then
  echo "ABORT: preflight FAILED -- nothing was pushed." >&2
  exit 1
fi

# ---- gate 2: everything committed ----------------------------------------
echo "== GATE 2: manifest files committed"
dirty=0
for e in "${ENTRIES[@]}"; do
  local_f="$(awk '{print $1}' <<<"$e")"
  [[ -f "$local_f" ]] || { echo "ABORT: $local_f does not exist" >&2; exit 1; }
  if [[ "$local_f" == /* ]]; then
    # Absolute path = external artifact (model checkpoint etc.) -- not
    # git-tracked; integrity is covered by the md5 gate instead.
    echo "  external artifact (md5-gated): $local_f"
    continue
  fi
  if ! git ls-files --error-unmatch -- "$local_f" >/dev/null 2>&1; then
    # Untracked by design (per-robot files such as x2_pc2/robot_env.env,
    # created from the shipped template): not git-gated, md5-gated instead.
    echo "  untracked per-robot file (md5-gated): $local_f"
    continue
  fi
  if [[ -n "$(git status --porcelain -- "$local_f")" ]]; then
    echo "  DIRTY: $local_f has uncommitted changes" >&2
    dirty=1
  fi
done
[[ $dirty -eq 0 ]] || { echo "ABORT: commit first -- nothing was pushed." >&2; exit 1; }
echo "  all committed (HEAD $(git rev-parse --short HEAD))"

if [[ $DRY -eq 1 ]]; then
  echo "== DRY RUN: gates passed; stopping before any PC2 contact."
  exit 0
fi

# ---- gate 3: backup on PC2 ------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
BK="${PC2_PREFIX}/backups/${TS}_preship"
echo "== GATE 3: PC2 backup -> $BK"
ssh "$PC2" "mkdir -p '$BK'"
for e in "${ENTRIES[@]}"; do
  remote_f="$(awk '{print $2}' <<<"$e")"
  ssh "$PC2" "if [ -f '$remote_f' ]; then cp -a '$remote_f' '$BK/' \
      && md5sum '$remote_f' >> '$BK/MANIFEST.md5'; \
      else echo 'new file (no prior copy): $remote_f' >> '$BK/MANIFEST.md5'; fi"
done
echo "  backup written; manifest at $BK/MANIFEST.md5"

# ---- push -----------------------------------------------------------------
echo "== PUSH"
for e in "${ENTRIES[@]}"; do
  local_f="$(awk '{print $1}' <<<"$e")"
  remote_f="$(awk '{print $2}' <<<"$e")"
  echo "  $local_f -> $remote_f"
  ssh "$PC2" "mkdir -p \"\$(dirname '$remote_f')\""
  scp -q "$local_f" "$PC2:$remote_f"
done

# ---- gate 4: md5 verify both sides ----------------------------------------
echo "== GATE 4: md5 verify"
fail=0
for e in "${ENTRIES[@]}"; do
  local_f="$(awk '{print $1}' <<<"$e")"
  remote_f="$(awk '{print $2}' <<<"$e")"
  lm="$(md5sum "$local_f" | awk '{print $1}')"
  rm_="$(ssh "$PC2" "md5sum '$remote_f'" | awk '{print $1}')"
  if [[ "$lm" == "$rm_" ]]; then
    echo "  OK   $lm  $remote_f"
  else
    echo "  MISMATCH local=$lm remote=$rm_  $remote_f" >&2
    fail=1
  fi
done
[[ $fail -eq 0 ]] || { echo "ABORT: md5 mismatch -- INVESTIGATE (backup: $BK)" >&2; exit 1; }

echo "== DONE. Pushed at HEAD $(git rev-parse --short HEAD); backup: $BK"
echo "   Restart is YOURS, at the robot, after the ignition checklist."
echo "   Rollback: copy files back from $BK on PC2."
