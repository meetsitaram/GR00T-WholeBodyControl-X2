#!/usr/bin/env bash
# POSTURE BENCH (2026-10-24): one SIM session of the full stack (C++ deploy + MuJoCo bridge + kplanner idling on the stand
# anchor) with a scripted external disturbance, logs collected for sim_posture_metrics.py. SIM ONLY -- never touches the
# robot; the disturbances are MuJoCo xfrc_applied inside the bridge, the deploy gets no extra flags.
#
# Settings = the SuperMimic robot env (x2_pc2/robot_env.env.supermimic_320k.template, PC2_PREFIX -> this repo) except
# the model and the plant, which are what is being compared:
#
#   sim_posture_bench.sh LABEL MODEL.onnx PLANT PROBE OUT_ROOT
#     PLANT  vendor_20260823 | mc_20261024        (X2_PLANT: the gains the deploy sends)
#     PROBE  stand       45 s idle, no disturbance
#            nudge       80 N, 0.2 s chest shoves every 5 s, +x +y -x -y, from sim t=20 s
#            tilt_pitch  25 N*m slow torso pitch push (3 s ramp, 1.5 s hold, 0.1 s release, 4 s rest), from t=20 s
#            tilt_roll   same about roll
#            push        SLOW horizontal push at the neck (4 s ramp, 2 s hold, 1 s release, 4 s rest), +x +y -x -y at
#                        PUSH_LEVELS (default 10,20,30 N), from t=20 s (operator 2026-10-24: steady, slowly increasing,
#                        at the neck). ~40 N at the neck is the feet-planted limit (44 N*m about the ankles vs ~50 N*m
#                        of weight x toe lever): at >= 40 N every policy steps / walks with the push.
#   -> OUT_ROOT/LABEL/PROBE/{logs/*.csv, logs/probe.csv, deploy.log, meta.txt}
#
# Serialized on a flock (one stack at a time). Stops the stack at the end with simstack_local.sh --stop.
# GPU-free: no viewer, CUDA hidden from the host-side helpers; the deploy's onnxruntime is CPU-only.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LABEL=$1; MODEL=$(realpath "$2"); PLANT=$3; PROBE=$4; OUT_ROOT=$5
[[ -n "$OUT_ROOT" && -f "$MODEL" ]] || { sed -n '2,20p' "$0"; exit 2; }
[[ -f "$REPO/gear_sonic/config/robot_plant/x2_ultra_${PLANT}.yaml" ]] || { echo "no plant $PLANT" >&2; exit 2; }
OUT="$OUT_ROOT/$LABEL/$PROBE"
mkdir -p "$OUT"
exec 9>"$OUT_ROOT/.bench.lock"
flock -n 9 || { echo "another posture bench session holds $OUT_ROOT/.bench.lock" >&2; exit 3; }

case "$PROBE" in
  stand)      EXTRA="";                                                                           DUR=45 ;;
  nudge)      EXTRA="--nudge-force 80 --nudge-axis xy --nudge-period 5 --nudge-duration 0.2 --nudge-start 20"; DUR=62 ;;
  tilt_pitch) EXTRA="--tilt-torque 25 --tilt-axis pitch --tilt-start 20";                         DUR=62 ;;
  tilt_roll)  EXTRA="--tilt-torque 25 --tilt-axis roll --tilt-start 20";                          DUR=62 ;;
  push)       EXTRA="--push-levels ${PUSH_LEVELS:-10,20,30} --push-start 20 --push-point neck";   DUR=170 ;;
  *) echo "unknown probe $PROBE" >&2; exit 2 ;;
esac

export PC2_PREFIX="$REPO" X2_PLANT="$PLANT"
set -a
# shellcheck disable=SC1091
. "$REPO/x2_pc2/robot_env.env.supermimic_320k.template"
set +a
export X2_PLANT="$PLANT"  # the template's ':=' keeps it; re-export for clarity
export MODEL="$MODEL" SONIC_MODEL="$MODEL"
export PLANNER_MODEL="${PLANNER_MODEL:-$HOME/x2_cloud_checkpoints/planner_onnx_g1core_s1d_a05_ws/x2_planner_template.onnx}"
export SIMSTACK_DIR="$OUT/session" SIMSTACK_NO_VIEWER=1 SIM_VIEWER_ARGS="" CUDA_VISIBLE_DEVICES=""
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-x2sim_supermimic}"   # this worktree's own colcon volumes
export SIM_BRIDGE_EXTRA="$EXTRA --probe-log $SIMSTACK_DIR/logs/probe.csv"
mkdir -p "$SIMSTACK_DIR/logs"

{
  echo "label $LABEL"; echo "model $MODEL md5 $(md5sum "$MODEL" | cut -c1-12)"; echo "plant $PLANT"
  echo "probe $PROBE"; echo "bridge_extra $SIM_BRIDGE_EXTRA"; echo "tuning $X2_RITUAL_TUNING"
  echo "planner $PLANNER_MODEL"; echo "start_utc $(date -u +%FT%TZ)"
} > "$OUT/meta.txt"

stop_stack() { "$REPO/gear_sonic/scripts/simstack_local.sh" --stop > "$OUT/stop.log" 2>&1 || true; }
trap stop_stack EXIT

echo "[bench] $LABEL / $PROBE: launching (plant $PLANT)"
"$REPO/gear_sonic/scripts/simstack_local.sh" > "$OUT/launch.log" 2>&1 || { echo "[bench] LAUNCH FAILED (see $OUT/launch.log)"; tail -5 "$OUT/launch.log"; exit 1; }
grep -m1 "RUNTIME PLANT LOADED\|COMPILED plant" "$SIMSTACK_DIR/logs/deploy.log" | tee -a "$OUT/meta.txt"
grep -q "RUNTIME PLANT LOADED: $PLANT " "$SIMSTACK_DIR/logs/deploy.log" || { echo "[bench] deploy did not load plant $PLANT -- aborting"; exit 1; }
sleep "$DUR"
cp "$SIMSTACK_DIR/logs/deploy.log" "$OUT/deploy.log"
grep -E "E-STOP ->|-> SAFE_HOLD|\[FATAL\]|TILT TRIP" "$OUT/deploy.log" | head -5 > "$OUT/trips.txt" || true
echo "end_utc $(date -u +%FT%TZ)" >> "$OUT/meta.txt"
echo "[bench] $LABEL / $PROBE: done -> $OUT ($(wc -l < "$SIMSTACK_DIR/logs/tick.csv") ticks, trips: $(wc -l < "$OUT/trips.txt"))"
