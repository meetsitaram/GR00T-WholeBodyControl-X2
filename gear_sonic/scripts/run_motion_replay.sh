#!/usr/bin/env bash
# RAW motion replay of a robot-joint clip, the run_pico_replay way: the stack is ALREADY UP
# (sim dual stack or the robot ritual) and this streams a clip into it. Operator 2026-09-09:
# "we should already have the stack up and running and then launch something like
# run_pico_replay or run_motion_replay".
#
#   ./gear_sonic/scripts/run_motion_replay.sh <clip.pkl|clip.x2m2>                          # SIM: no --pc2-host = the local sim stack, always
#   ./gear_sonic/scripts/run_motion_replay.sh <clip.pkl|clip.x2m2> --pc2-host <PC2_IP>       # ROBOT: an ip = the real robot, always
#   options: --key <motion_key>        (multi-clip pkl; default first key)
#            --start-s S --dur-s D     (window of the clip to bake; default whole clip)
#            --own-deploy              (see below) [--model *_g1.onnx] [--max-duration S] [--no-viewer]
#
# DEFAULT = INTO THE RUNNING STACK. The clip is baked to X2M2 (joints + root quat @ fps),
# dropped into the kplanner's dances dir, and played with one motion_clip_cmd. The kplanner
# is only the wire relay here (DancePlayback streams the frames verbatim, no planning):
# SONIC tracks the joints and a walking clip WALKS (39k on the fixed-retarget relaxed walk:
# 6.4 m in 15 s, MuJoCo ground truth 2026-09-09). The pad's LB+RB / a second run with
# --stop / Ctrl-C here stop it. Nothing is restarted; the file is resolved at play time.
#
# --own-deploy = the kplanner-free path: deploy_x2.sh --input-type motion_file, the deploy
# reads the bake itself. SIM: deploy_x2.sh sim (refuses to stack on a running sim stack).
# ROBOT: deploy_x2.sh onbot in tmux 'x2_deploy' with the ritual DOWN; the launcher only
# stages files and opens the pane, the operator answers the MC handoff / Y/n gate there.
#
# TARGET RULE (operator 2026-09-05): no --pc2-host = sim, an ip = the robot, no robot
# default, no detection -- same rule as run_pico_replay.sh / run_pico_teleop.sh.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIP=""; HOST=""; MODEL=""; KEY=""; START_S=0; DUR_S=""; MAXDUR=""; VIEWER=1; OWN=0; STOP=0
while (( $# )); do
    case "$1" in
        --pc2-host) HOST="${2:-}"; shift 2 ;;
        --model) MODEL="${2:-}"; shift 2 ;;
        --key) KEY="${2:-}"; shift 2 ;;
        --start-s) START_S="${2:-0}"; shift 2 ;;
        --dur-s) DUR_S="${2:-}"; shift 2 ;;
        --max-duration) MAXDUR="${2:-}"; shift 2 ;;
        --no-viewer) VIEWER=0; shift ;;
        --own-deploy) OWN=1; shift ;;
        --stop) STOP=1; shift ;;
        -h|--help) sed -n 2,26p "$0"; exit 0 ;;
        *) if [[ -z "$CLIP" ]]; then CLIP="$1"; shift; else echo "unknown arg $1" >&2; exit 2; fi ;;
    esac
done
[[ -n "$CLIP" && -f "$CLIP" ]] || { echo "usage: run_motion_replay.sh <clip.pkl|clip.x2m2> [--pc2-host <robot ip>] [--key K] [--start-s S --dur-s D] [--stop] [--own-deploy [--model *_g1.onnx] [--max-duration S] [--no-viewer]]" >&2; exit 2; }
if [[ -z "$HOST" ]]; then TARGET=sim; else TARGET=robot; fi
PY="${SONIC_PYTHON:-${REPO_ROOT}/.venv/bin/python}"   # SONIC_PYTHON overrides; default the repo .venv, else python3
command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3 || true)"
[[ -n "$PY" ]] || { echo "ERROR: no python interpreter (create ${REPO_ROOT}/.venv or set SONIC_PYTHON=/path/to/python)" >&2; exit 1; }
# Model cache (MODELS.md): CKPT_ROOT (alias X2_CHECKPOINTS_DIR) defaults to $SONIC_HOME/x2.
SONIC_X2="${SONIC_X2_MODELS:-${SONIC_HOME:-$HOME/.cache/sonic}/x2}"
CKPT_ROOT="${CKPT_ROOT:-${X2_CHECKPOINTS_DIR:-$SONIC_X2}}"
PC2_PREFIX="${PC2_PREFIX:-/home/run/gear-sonic}"
# --own-deploy only: the single-head pose graph. MODEL= wins, else the shipped
# default set (gear_sonic_deploy/models/, git-lfs), else the cache layout.
SHIPPED_G1="${REPO_ROOT}/gear_sonic_deploy/models/x2_sonic_v16ft8_45000_g1.onnx"
if [[ -z "${MODEL:-}" ]] && [[ -f "$SHIPPED_G1" ]] && ! head -c 40 "$SHIPPED_G1" | grep -q 'git-lfs'; then
    MODEL="$SHIPPED_G1"
fi
MODEL="${MODEL:-$SONIC_X2/sonic_policy/x2_sonic_policy.onnx}"
# Tuning preset for --own-deploy (sim and robot): the shipped default, same
# as the robot ritual (X2_RITUAL_TUNING / SIM_TUNING_YAML override it).
TUNING_YAML="${X2_RITUAL_TUNING:-${SIM_TUNING_YAML:-configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml}}"
# Sim kplanner dances dir (must match the running stack's KPLANNER_DANCES_DIR).
DANCES_SIM="${KPLANNER_DANCES_DIR:-${REPO_ROOT}/gear_sonic_deploy/data/motions_x2m2/demo_bank}"
WORK_DIR="${SONIC_HOME:-$HOME/.cache/sonic}/motion_replay"
MODEL="$(readlink -f "$MODEL" 2>/dev/null || echo "$MODEL")"
case "$MODEL" in
    *_dual.onnx|*_token.onnx|*smpl*.onnx)
        echo "[replay] REFUSING model $MODEL: motion-file replay needs the single-head pose graph (*_g1.onnx)." >&2
        echo "[replay]   (a dual-head graph dies on the first control tick: 'use InferObs()' -> SAFE_HOLD, sim 2026-09-09 09:25)" >&2; exit 4 ;;
esac
if (( OWN )); then [[ -f "$MODEL" ]] || { echo "[replay] model not found: $MODEL (set MODEL=<*_g1.onnx>, see MODELS.md)" >&2; exit 2; }; fi

# ---- bake: pkl -> x2m2 (the deploy's format; identical layout in pkl_to_x2m2.py and
# reference_motion.hpp: magic X2M2, n, 31, fps, per frame dof[31]+quat_xyzw[4]) --------
BAKE_DIR="${WORK_DIR}/bakes"; mkdir -p "$BAKE_DIR"
CLIP="$(readlink -f "$CLIP")"
if [[ "$CLIP" == *.x2m2 ]]; then
    X2M2="$CLIP"
else
    STEM="$(basename "${CLIP%.pkl}")"; [[ -n "$KEY" ]] && STEM="${STEM}__${KEY}"
    [[ -n "$DUR_S" ]] && STEM="${STEM}__${START_S}s_${DUR_S}s"
    X2M2="${BAKE_DIR}/${STEM}.x2m2"
    if [[ -z "$KEY" ]]; then KEY="$("$PY" -c "import joblib,sys; print(next(iter(joblib.load(sys.argv[1]))))" "$CLIP")"; fi
    if [[ -z "$DUR_S" ]]; then DUR_S="$("$PY" -c "import joblib,sys; d=joblib.load(sys.argv[1])[sys.argv[2]]; print(len(d['dof'])/float(d.get('fps',30.0)))" "$CLIP" "$KEY")"; fi
    "$PY" "${REPO_ROOT}/gear_sonic/scripts/pkl_to_x2m2.py" --pkl "$CLIP" --key "$KEY" --start-s "$START_S" --dur-s "$DUR_S" --out "$X2M2" --no-hand-sidecar
fi
INFO="$("$PY" - "$X2M2" "$REPO_ROOT" <<'PYEOF'
import sys; sys.path.insert(0, sys.argv[2])
from pathlib import Path; import numpy as np
from gear_sonic.utils.pose_pipeline.wire import load_x2m2
dof, quat, fps = load_x2m2(Path(sys.argv[1]))
print(f"{dof.shape[0]} frames @ {fps:.0f} fps = {dof.shape[0]/fps:.1f} s; max joint jump {np.degrees(np.abs(np.diff(dof,axis=0)).max()):.1f} deg/frame; knee range L {np.degrees(np.ptp(dof[:,3])):.0f} deg")
PYEOF
)"
CLIP_S="$(echo "$INFO" | grep -oE '= [0-9.]+ s' | grep -oE '[0-9.]+')"
MAXDUR="${MAXDUR:-$(python3 -c "print(int(float('$CLIP_S'))+2)")}"
echo "[replay] bake  : $X2M2"
echo "[replay] clip  : $INFO"
echo "[replay] model : $MODEL"
echo "[replay] target: $TARGET   max-duration ${MAXDUR}s"

if (( ! OWN )); then
    # ------------------------- INTO THE RUNNING STACK (default) -------------------------
    STEM_KEY="$(basename "${X2M2%.x2m2}")"
    if [[ "$TARGET" == sim ]]; then
        pgrep -f "pc2_kplanner_onnx[.]py" >/dev/null || { echo "[replay] no sim kplanner running -- bring the sim stack up first (sim_onnx_planner.sh ... --whole-body-teleop)" >&2; exit 5; }
        DANCES="$DANCES_SIM"; mkdir -p "$DANCES"; CLIP_HOST=127.0.0.1; CLIP_PORT="${CLIP_PORT:-5668}"
        cp -f "$X2M2" "${DANCES}/${STEM_KEY}.x2m2"
    else
        PC2="run@${HOST}"; DANCES="${PC2_PREFIX}/planner_stack/models/dances_x2m2"; CLIP_HOST="$HOST"; CLIP_PORT="${CLIP_PORT:-5568}"
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$PC2" "pgrep -f pc2_kplanner_onnx.py >/dev/null" \
            || { echo "[replay] the ritual's kplanner is not running on PC2 -- ignite the ritual first (at the robot)." >&2; exit 5; }
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$PC2" "mkdir -p '${DANCES}'"
        scp -o ConnectTimeout=5 "$X2M2" "${PC2}:${DANCES}/${STEM_KEY}.x2m2" >/dev/null
        LMD5=$(md5sum "$X2M2" | cut -c1-32); RMD5=$(ssh -o BatchMode=yes "$PC2" "md5sum '${DANCES}/${STEM_KEY}.x2m2' | cut -c1-32")
        [[ "$LMD5" == "$RMD5" ]] || { echo "[replay] md5 mismatch after scp ($LMD5 vs $RMD5)" >&2; exit 6; }
        echo "[replay] staged ${DANCES}/${STEM_KEY}.x2m2 on PC2 (md5 $LMD5)"
        echo "[replay] ROBOT CHECKLIST: kplanner idle-standing, whole-body NOT engaged, floor clear for the clip's travel, pad in hand (LB+RB stops the clip)."
        read -r -p "[replay] type 'replay' to play ${STEM_KEY} on the robot: " ANS
        [[ "$ANS" == "replay" ]] || { echo "[replay] aborted"; exit 1; }
    fi
    "$PY" - "$CLIP_HOST" "$CLIP_PORT" "$STEM_KEY" "$CLIP_S" "$STOP" <<'PYEOF'
import sys, time, json, zmq
host, port, key, dur, stop = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5] == "1"
ctx = zmq.Context.instance(); s = ctx.socket(zmq.PUB); s.connect(f"tcp://{host}:{port}"); time.sleep(0.3)
def send(p): s.send_multipart([b"motion_clip_cmd", json.dumps(p).encode()])
if stop:
    send({"action": "stop"}); print("[replay] STOP sent", flush=True); time.sleep(0.2); sys.exit(0)
send({"action": "play", "motion_key": key, "kind": "locomotion", "source": "motion_replay"})
print(f"[replay] PLAY {key} -> tcp://{host}:{port}  ({dur:.1f} s; Ctrl-C = stop)", flush=True)
try:
    time.sleep(dur + 1.0); print("[replay] clip done (kplanner returns to its idle anchor)", flush=True)
except KeyboardInterrupt:
    send({"action": "stop"}); print("\n[replay] STOP sent", flush=True); time.sleep(0.2); sys.exit(130)
PYEOF
    exit 0
fi


# ------------------------------- --own-deploy -----------------------------------
if [[ "$TARGET" == sim ]]; then
    if docker ps -q --filter name=x2sim 2>/dev/null | grep -q .; then
        echo "[replay] REFUSING: a sim stack (docker x2sim) is running -- ./gear_sonic/scripts/simstack_local.sh --stop first." >&2; exit 5
    fi
    # the sim deploy resolves models under /workspace/checkpoints (= $CKPT_ROOT); stage by basename
    C_MODEL="${CKPT_ROOT}/$(basename "$MODEL")"
    if [[ ! -f "$C_MODEL" ]]; then cp -f "$MODEL" "$C_MODEL"; echo "[replay] staged model -> $C_MODEL"; fi
    LOG="${WORK_DIR}/sim_$(date -u +%Y%m%d_%H%M%SZ)_$(basename "${X2M2%.x2m2}")"; mkdir -p "$LOG"
    cd "${REPO_ROOT}/gear_sonic_deploy"
    _T="$TUNING_YAML"; [[ "$_T" == /* || -f "$_T" ]] || _T="${REPO_ROOT}/${_T}"
    TUNING_FLAGS="$(python3 scripts/tuning_yaml_to_sim_flags.py "$_T" | sed 's/--ramp-seconds 2.0 //')"
    VIEW=(); (( VIEWER )) && VIEW=(--sim-viewer)
    echo "[replay] sim log dir: $LOG"
    X2_PLANT=vendor_20260823 X2_ANCHOR_ORI_MODE=body ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=73 \
      ./deploy_x2.sh sim --no-stop-mc --no-confirm --autostart-after 3 "${VIEW[@]}" \
        --sim-mjcf /workspace/sonic/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml \
        --model "/workspace/checkpoints/$(basename "$MODEL")" --motion "$X2M2" \
        --log-dir "$LOG" --max-duration "$MAXDUR" $TUNING_FLAGS 2>&1 | tee "$LOG/launch.log"
    grep -E "Loaded ONNX|plant\] RUNTIME|FATAL|SAFE_HOLD|Max duration|max-duration" "$LOG/launch.log" | cut -c1-160 || true
    exit 0
fi

# ---------------------------------- ROBOT ------------------------------------------
PC2="run@${HOST}"; PREFIX="$PC2_PREFIX"; RDIR="${PREFIX}/motions_replay"
R_MODEL="${PREFIX}/policies/$(basename "$MODEL")"
R_TUNING="${PREFIX}/gear_sonic_deploy/configs/real_deploy_tuning/$(basename "$TUNING_YAML")"
ssh_pc2() { ssh -o ConnectTimeout=5 -o BatchMode=yes "$PC2" "$@"; }
echo "[replay] ROBOT ${HOST}: checking PC2 state ..."
if ssh_pc2 "tmux has-session -t x2_deploy 2>/dev/null"; then
    echo "[replay] REFUSING: tmux 'x2_deploy' exists on PC2 (the ritual's deploy is up). Stop the ritual first; raw replay owns the bus." >&2; exit 5
fi
if ssh_pc2 "pgrep -f pc2_kplanner_onnx.py >/dev/null"; then
    echo "[replay] REFUSING: pc2_kplanner_onnx is running on PC2. Raw replay does not use the kplanner; stop the ritual stack first." >&2; exit 5
fi
ssh_pc2 "test -f '$R_MODEL'" || { echo "[replay] model missing on PC2: $R_MODEL (ship the *_g1.onnx sibling first)" >&2; exit 2; }
ssh_pc2 "test -f '$R_TUNING'" || { echo "[replay] tuning yaml missing on PC2: $R_TUNING" >&2; exit 2; }
ssh_pc2 "mkdir -p '$RDIR' '${PREFIX}/log/motion_replay'"
scp -o ConnectTimeout=5 "$X2M2" "${PC2}:${RDIR}/" >/dev/null
R_X2M2="${RDIR}/$(basename "$X2M2")"
LMD5=$(md5sum "$X2M2" | cut -c1-32); RMD5=$(ssh_pc2 "md5sum '$R_X2M2' | cut -c1-32")
[[ "$LMD5" == "$RMD5" ]] || { echo "[replay] md5 mismatch after scp ($LMD5 vs $RMD5)" >&2; exit 6; }
STAMP=$(date -u +%Y%m%d_%H%M%SZ); R_LOG="${PREFIX}/log/motion_replay/replay_${STAMP}"
R_SCRIPT="${RDIR}/run_$(basename "${X2M2%.x2m2}")_${STAMP}.sh"
# The ritual's deploy launch line (x2_pc2/start_x2_deploy_ritual.sh) minus --vla/zmq,
# wrist/head bypass (zmq-only), and --no-confirm: the Y/n gate is the operator's.
cat > /tmp/run_motion_replay_remote.sh <<REOF
#!/usr/bin/env bash
export X2_PLANT=vendor_20260823
mkdir -p "$R_LOG"
cd "${PREFIX}/gear_sonic_deploy" && \\
  "${PREFIX}/gear_sonic_deploy/deploy_x2.sh" onbot --no-docker \\
    --model "$R_MODEL" --motion "$R_X2M2" --max-duration "$MAXDUR" \\
    --intra-op-threads 3 --log-dir "$R_LOG" \\
    --deploy-extra-arg --flight-recorder-seconds --deploy-extra-arg 30 \\
    --onbot-prefix "$PREFIX" --onbot-ws "${PREFIX}/ws" --onbot-venv "${PREFIX}/venv" \\
    --onbot-onnxruntime "${PREFIX}/onnxruntime" \\
    --onbot-aimdk-prefix /agibot/software/housekeeper/bin/aimdk_msgs \\
    --tuning-config "$R_TUNING"
rc=\$?; echo "[motion_replay] deploy exited rc=\$rc"; exec bash -l
REOF
scp -o ConnectTimeout=5 /tmp/run_motion_replay_remote.sh "${PC2}:${R_SCRIPT}" >/dev/null
echo
echo "[replay] ROBOT CHECKLIST -- raw replay of $(basename "$X2M2") ($INFO)"
echo "[replay]   * ritual DOWN (checked), robot on the STAND / gantry, floor clear for the clip's travel"
echo "[replay]   * pad in hand: the deploy's tilt trip + --max-duration ${MAXDUR}s are the only automatic stops"
echo "[replay]   * the deploy does the MC stop_app/start_app handoff itself; you answer the Y/n gate in the pane"
read -r -p "[replay] type 'replay' to open the deploy pane on PC2: " ANS
[[ "$ANS" == "replay" ]] || { echo "[replay] aborted"; exit 1; }
ssh_pc2 "tmux new-session -d -s x2_deploy 'bash $R_SCRIPT'"
echo "[replay] opened tmux 'x2_deploy' on PC2 running $R_SCRIPT"
echo "[replay] attach and answer the gate:   ssh -t ${PC2} tmux attach -t x2_deploy"
echo "[replay] logs: ${R_LOG}   (pull afterwards for the arm/travel analysis)"
