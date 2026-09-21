#!/usr/bin/env bash
# Background watcher: polls Nebius capacity every INTERVAL seconds, logs the
# 8-GPU table to logs/nebius_capacity_watch.log, and emits ALERT lines when
# on-demand availability for 8xH200 or 8xH100 in any region changes
# meaningfully (newly available, newly empty, or count change).
#
# Usage:
#   bash gear_sonic/scripts/cloud/watch_8gpu_capacity.sh &
#   tail -f logs/nebius_capacity_watch.log
#
# Stop with: kill $(cat logs/nebius_capacity_watch.pid)

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

INTERVAL="${INTERVAL:-120}"
LOG="${LOG:-logs/nebius_capacity_watch.log}"
PIDFILE="${PIDFILE:-logs/nebius_capacity_watch.pid}"
mkdir -p "$(dirname "$LOG")"
echo $$ > "$PIDFILE"

# Track previous on-demand totals per (platform,region) so we can fire ALERTs
# only on meaningful state changes (not every poll).
declare -A PREV_OD

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] === watcher started  pid=$$  interval=${INTERVAL}s  log=$LOG ===" | tee -a "$LOG"

# Redirect everything below to the LOG (no pipe -> no subshell -> PREV_OD persists).
exec >> "$LOG" 2>&1

while true; do
    echo
    echo "[$(ts)] --- poll ---"
    OUT="$(python gear_sonic/scripts/cloud/nebius_gpu_scan.py --gpus 8 2>&1)"
    echo "$OUT"

    # Parse the table lines for ALERTs. Each data row looks like:
    #    8 | gpu-h200-sxm   | eu-north1    | 8gpu-128vcpu-1600gb    |    1/4     MEDIUM  | ...
    # We aggregate on-demand available per (platform,region) across AZ rows.
    unset CUR_OD
    declare -A CUR_OD
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]*8[[:space:]]*\|[[:space:]]*([a-z0-9-]+)[[:space:]]*\|[[:space:]]*([a-z0-9-]+)[[:space:]]*\|[[:space:]]*[^|]+\|[[:space:]]*([0-9]+)/[0-9]+ ]]; then
            plat="${BASH_REMATCH[1]}"
            region="${BASH_REMATCH[2]}"
            od="${BASH_REMATCH[3]}"
            key="${plat}@${region}"
            CUR_OD[$key]=$(( ${CUR_OD[$key]:-0} + od ))
        fi
    done <<< "$OUT"

    for key in "${!CUR_OD[@]}"; do
        cur="${CUR_OD[$key]}"
        prev="${PREV_OD[$key]:-INIT}"
        if [[ "$prev" == "INIT" ]]; then
            if (( cur > 0 )); then
                echo "[$(ts)] ALERT BASELINE ${key} on-demand=${cur}"
            fi
        elif [[ "$cur" != "$prev" ]]; then
            if (( cur > prev )); then
                echo "[$(ts)] ALERT UP ${key} on-demand ${prev} -> ${cur}"
            else
                echo "[$(ts)] ALERT DOWN ${key} on-demand ${prev} -> ${cur}"
            fi
        fi
        PREV_OD[$key]="$cur"
    done
    for key in "${!PREV_OD[@]}"; do
        if [[ -z "${CUR_OD[$key]+x}" && "${PREV_OD[$key]}" != "INIT" ]]; then
            echo "[$(ts)] ALERT GONE ${key} (no row this poll, was ${PREV_OD[$key]})"
            PREV_OD[$key]=0
        fi
    done

    sleep "$INTERVAL"
done
