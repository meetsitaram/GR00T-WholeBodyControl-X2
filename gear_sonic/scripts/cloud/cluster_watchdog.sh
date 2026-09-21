#!/bin/bash
# Auto-heal loop for the preemptible GPU cluster. Runs on the RESERVED node.
#
# Nebius does NOT auto-restart preempted instances, so this loop keeps trying
# to start any STOPPED worker every POLL_S seconds (start requests fail while
# capacity is unavailable — that is expected; we just keep asking). Once an
# instance boots, its own boot-time elastic-supervisor unit rejoins training;
# nothing else on this side is needed.
#
#   tmux new -d -s watchdog "NEBIUS_PROJECT=project-... bash gear_sonic/scripts/cloud/cluster_watchdog.sh"
#
# Env:
#   NEBIUS_PROJECT nebius project id (required; PARENT_ID is accepted as an alias)
#   NODE_PREFIX    instance-name prefix to watch (default: gpu-cluster-node)
#   EXCLUDE        instance names to skip, space-separated (default: none)
#   POLL_S         poll interval seconds (default: 120)

set -uo pipefail
case "${1:-}" in
  -h|--help) sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
esac
PARENT_ID=${PARENT_ID:-${NEBIUS_PROJECT:-}}
: "${PARENT_ID:?set NEBIUS_PROJECT=<nebius project id>}"
NODE_PREFIX=${NODE_PREFIX:-gpu-cluster-node}
EXCLUDE=${EXCLUDE:-}
POLL_S=${POLL_S:-120}
NEBIUS=${NEBIUS:-$HOME/.nebius/bin/nebius}
LOG=${LOG:-$HOME/cluster_watchdog.log}

echo "$(date -u) watchdog up: prefix=$NODE_PREFIX poll=${POLL_S}s" | tee -a "$LOG"
while true; do
  "$NEBIUS" compute instance list --parent-id "$PARENT_ID" --format json 2>/dev/null |
  python3 - "$NODE_PREFIX" "$EXCLUDE" <<'EOF' |
import json, sys
prefix, exclude = sys.argv[1], set(sys.argv[2].split())
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for i in d.get("items", []):
    name = i["metadata"]["name"]
    state = i.get("status", {}).get("state", "")
    if name.startswith(prefix) and name not in exclude and state == "STOPPED":
        print(i["metadata"]["id"], name)
EOF
  while read -r iid name; do
    [ -z "$iid" ] && continue
    echo "$(date -u) $name is STOPPED -> start attempt" | tee -a "$LOG"
    "$NEBIUS" compute instance start --id "$iid" --parent-id "$PARENT_ID" >> "$LOG" 2>&1 \
      && echo "$(date -u) $name start request accepted" | tee -a "$LOG" \
      || echo "$(date -u) $name start failed (likely no capacity yet), will retry" | tee -a "$LOG"
  done
  sleep "$POLL_S"
done
