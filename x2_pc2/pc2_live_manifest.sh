#!/bin/bash
# What the robot ACTUALLY runs: parse PC2's ritual scripts for every python script / env file / model they launch
# or source, and print md5 + mtime of those exact files, next to the laptop-side counterpart's md5 when one is
# known. Lesson learned: the ritual runs ${PC2_PREFIX}/pc2_kplanner_onnx.py (a top-level copy), not
# gear_sonic/scripts/pc2_kplanner_onnx.py -- a push to the repo-shaped path changed nothing on the robot.
# Usage: x2_pc2/pc2_live_manifest.sh [run@<PC2_IP>]      (read-only; safe any time; default from PC2_HOST)
PC2=${1:-${PC2_HOST:+run@${PC2_HOST}}}; G=${PC2_PREFIX:-/home/run/gear-sonic}
[ -n "$PC2" ] || { echo "usage: $0 run@<PC2_IP>   (or export PC2_HOST=<PC2_IP>)" >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
echo "== files the rituals launch on PC2 (from ritual_start_*.sh, start_x2_deploy_ritual.sh)"
ssh -o ConnectTimeout=8 -o BatchMode=yes $PC2 "cd $G; for f in ritual_start_sonic.sh ritual_start_demo.sh start_x2_deploy_ritual.sh; do [ -f \$f ] || continue; grep -vE '^\s*#' \$f | grep -oE '($G|\\\$PS|\\\$G|\\\$PREFIX)?[A-Za-z0-9_./-]+\.(py|env|onnx|yaml|pkl|x2m2)' ; done | sed 's#^\\\$PS#planner_stack#; s#^\\\$G#.#; s#^\\\$PREFIX#.#' | sort -u | while read p; do q=\$p; [ -e \"\$q\" ] || q=\"$G/\$p\"; [ -e \"\$q\" ] && printf '%s  %s  %s\n' \"\$(md5sum \"\$q\" | cut -c1-8)\" \"\$(stat -c %y \"\$q\" | cut -c1-16)\" \"\$q\" || printf 'MISSING  %s\n' \"\$p\"; done" 2>&1
echo "== laptop counterparts (repo working tree)"
for f in gear_sonic/scripts/pc2_kplanner_onnx.py gear_sonic/scripts/pad_locomotion_bridge.py gear_sonic/utils/teleop/estop_gesture.py x2_pc2/pad_bindings.env x2_pc2/robot_env.env; do
  [ -f "$REPO/$f" ] && printf '%s  %s  %s\n' "$(md5sum "$REPO/$f" | cut -c1-8)" "$(git -C "$REPO" log -1 --format=%h -- "$f" 2>/dev/null)" "$f"
done
echo "(a PC2 file whose md5 matches neither the working tree nor a committed version is an untracked robot-only edit)"
