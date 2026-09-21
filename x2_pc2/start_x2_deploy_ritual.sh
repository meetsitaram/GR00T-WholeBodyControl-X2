#!/usr/bin/env bash
# PC2-resident deploy launcher for GAMEPAD RITUAL ignition. Checked into the
# repo at x2_pc2/ and staged to /home/run/gear-sonic/ by pc2_bringup.sh.
#
# WHY THIS FILE EXISTS (2026-07-29 blocked-ignition incident):
# the ritual scripts used to launch the deploy via log/start_x2_deploy.sh,
# which x2_pc2_daemons.sh REGENERATES from the laptop on every `start` with
# whatever flags that session used. A tethered session that omitted
# --no-confirm left a body behind whose deploy sat forever at the
# "Stop MC and launch policy? [y/N]" gate inside a detached tmux pane --
# unanswerable from a gamepad. The robot's ignition path must never depend
# on what the last laptop command happened to be.
#
# SAFETY: --no-confirm is baked in HERE, and only here, because on this
# path the operator confirmation is the ritual chord (L1+L2+R1+R2+Y,
# deliberately hard to press by accident) plus the pose-stream gate the
# ritual scripts run before this file is allowed to execute. Laptop-driven
# starts via x2_pc2_daemons.sh keep their interactive y/N gate.
#
# Arg set mirrors the 2026-07-28 hardware-verified deploy body, minus
# --vla-resume-host (that's the laptop/Quest3 SAFE_IDLE resume PUB; empty
# disables the socket, and the deploy still self-exits SAFE_IDLE on the
# first non-stale pose frame -- no laptop involved).
#
# Overridables (env), defaults = current deployed state:
#   X2_RITUAL_MODEL        sonic ONNX   (REQUIRED; robot_env.env sets it -- the shipped
#                                        template points at policies/x2_sonic_v16ft8_45000_dual.onnx)
#   X2_RITUAL_TUNING       tuning YAML  (default trained_gains_s0_inc_waistmc.yaml; robot_env.env sets it)
#   X2_RITUAL_HEAD_BYPASS  {ref,off}    (default ref -- pad head-look; SONIC's
#                                        own head tracking saturates ~+-10 deg)
#   X2_RITUAL_HEAD_DEV     rad          head dev clamp. Defaults: 0.40 when
#                                        bypass=ref (~= full joint range,
#                                        yaw +-0.366 / pitch +-0.384), 0.01
#                                        when off (legacy head-straight pin)
#
# TUNING DEFAULT = trained_gains_s0_inc_waistmc.yaml (the shipped preset:
# unity kp/kd scales for vendor-plant models plus MC-stiffness waist pitch);
# robot_env.env normally pins X2_RITUAL_TUNING explicitly. The
# 2026-07-29 waist-wobble incident (battery pull) ran the launcher's
# original default, walking_recovery_loose.yaml -- inherited blindly from
# the previous night's laptop session body. recovery_loose is soft_kp's
# ANCESTOR: waist_pr kp 2.81 (vs 2.00) and untrimmed kd, i.e. the exact
# action-jump-times-kp slam mode soft_kp's header documents. The ritual
# default must be the ritual-proven config, never whatever a laptop
# experiment last used.

# X2_RITUAL_INTRA_OP {1,2,3,...}  (default 3)
# ONNX Runtime threads INSIDE one inference call.
#
# 1 (the binary's own default) IS NOT ENOUGH FOR v13. Set to 1 on 2026-08-25
# for era-consistency in the armature matrix; the very next run failed. The
# 50 Hz control loop ran 22-36 ms against its 20 ms budget, and because every
# callback shared one MutuallyExclusive group, that saturated the executor:
# the 500 Hz writer fell to 3.4 Hz and the state subscriptions starved until
# all five sources read stale -> SAFE_HOLD. Meanwhile the policy integrated
# open-loop on frozen observations and drove the waist command to 129/162 deg
# (clamped at 0.45 rad, so the robot never moved).
#
# 2 was added 2026-08-24 because v13 (166 MB, ~2.85x the v0-era 58 MB models)
# was visibly slower -- correct reasoning. Even at 2 the >25 ms tail was
# 0.6% -> 3.7%, i.e. already near the edge. 3 buys headroom on 8 cores while
# leaving room for the executor and the 500 Hz writer.
#
# Ignition is the pad chord, so there is no shell to export from -- change the
# number on the --intra-op-threads line below to A/B it, and compare tick.csv
# period tails on the SAME model.
PREFIX="${PC2_PREFIX:-/home/run/gear-sonic}"
# Source the robot profile FIRST: it carries X2_RITUAL_TUNING (the tuning
# switch) so the pad-CHORD ignition selects the preset -- the operator
# holds the robot at SONIC start and can never be at a keyboard
# (2026-08-11). ":=" idiom in the profile means a pre-set env var still
# overrides the file.
# ONE env file (2026-09-02). This used to source kplanner_profile.env, the
# pre-08-25 name kept as a symlink to robot_env.env. On 09-01 the symlink
# became a real file (a snapshot of that day's env, pinning s1it10000) and
# every model push into robot_env.env after that was silently inert -- the
# 19000 token test and today's 36000 test both ran S1-10k. No optional
# sources, no silent defaults: the file must exist and must set the model.
if [ ! -f "${PREFIX}/robot_env.env" ]; then
    echo "[ritual-launch] FATAL: ${PREFIX}/robot_env.env missing -- refusing to launch" >&2
    echo "                 On the laptop: cp x2_pc2/robot_env.env.template x2_pc2/robot_env.env" >&2
    echo "                 (or robot_env.env.frozen.template), edit the X2_RITUAL_MODEL /" >&2
    echo "                 X2_WB_* model lines, then re-run pc2_bringup.sh (or push_to_pc2.sh)." >&2
    exit 1
fi
if [ -e "${PREFIX}/kplanner_profile.env" ] && [ ! -L "${PREFIX}/kplanner_profile.env" ]; then
    echo "[ritual-launch] FATAL: ${PREFIX}/kplanner_profile.env is a REAL FILE (stale env snapshot);" >&2
    echo "                 robot_env.env is the only env. Retire it (mv) before launching." >&2
    exit 1
fi
. "${PREFIX}/robot_env.env"
MODEL="${X2_RITUAL_MODEL:?robot_env.env must set X2_RITUAL_MODEL}"
echo "[ritual-launch] resolved: X2_RITUAL_MODEL=${MODEL}"
echo "[ritual-launch] resolved: X2_WB_TOKEN_MODEL=${X2_WB_TOKEN_MODEL:-<unset>}  X2_WB_TOKENIZER=${X2_WB_TOKENIZER:-<unset>}  X2_PLANT=${X2_PLANT:-<unset>}"
TUNING="${X2_RITUAL_TUNING:-${PREFIX}/gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml}"
if [ ! -f "${TUNING}" ]; then
    echo "[ritual-launch] FATAL: tuning preset ${TUNING} not found on PC2." >&2
    echo "                 Set X2_RITUAL_TUNING in robot_env.env (see x2_pc2/robot_env.env.template) to a" >&2
    echo "                 preset staged under ${PREFIX}/gear_sonic_deploy/configs/real_deploy_tuning/ (pc2_bringup.sh step 7)." >&2
    exit 1
fi
# Head bypass: force-write head targets from the ZMQ reference (pad
# head-look overlay in the kplanner daemon). The old hard pin
# (--max-target-dev-head 0.01) defeats it, so the dev clamp opens to
# ~full joint range whenever the bypass is on. Kill switch:
# X2_RITUAL_HEAD_BYPASS=off restores the exact legacy behaviour.
HEAD_BYPASS="${X2_RITUAL_HEAD_BYPASS:-ref}"
if [ "${HEAD_BYPASS}" = "ref" ]; then
    HEAD_DEV="${X2_RITUAL_HEAD_DEV:-0.40}"
else
    HEAD_DEV="${X2_RITUAL_HEAD_DEV:-0.01}"
fi
# --- whole-body teleop toggle (2026-08-29) -----------------------------------
# Env-driven like every other session knob (pad-chord ignition, no keyboard):
# X2_WHOLE_BODY_TELEOP=1 in robot_env.env repoints the pose/token source at
# the LAPTOP's pico_token_sender.py and adds --whole-body-teleop. The binary
# refuses a non-token model in this mode (and a token model outside it), so a
# half-flipped env fails loudly at start, before the robot moves. The
# pose-ref starvation watchdog is FORCED ON here -- stream loss -> SAFE_IDLE
# is the whole-body safety story; the pad session keeps its legacy disable.
# HYBRID (2026-08-29): pad/kplanner stays PRIMARY in every mode — same
# localhost pose chain, same watchdog posture as every demo to date. The
# whole-body flag only ARMS the overlay: token graph loaded alongside,
# deploy connects to the laptop sender, and the token drives ONLY while
# its stream is fresh AND the operator is engaged (grips). Stream loss or
# disengage = pad has authority again, no SAFE_IDLE theatrics.
VLA_HOST=localhost
VLA_PORT=5558
WATCHDOG_ARGS="--deploy-extra-arg --disable-pose-ref-watchdog"
# IGNITION GATE (2026-08-30 battery-pull incident): the deploy must never
# ignite while the kplanner is dead. The kplanner's pose PUB binds :5556;
# no listener there = kplanner crashed or still loading = pad chain AND the
# pad e-stop chord are dead. The old pose-frame gate passed on the
# watchdog's idle fallback, hiding exactly this. Wait for the bind (ONNX
# load takes ~5 s), then refuse loudly.
KP_OK=""
for _ in $(seq 1 30); do
    if ss -tln 2>/dev/null | grep -q ':5556 '; then KP_OK=1; break; fi
    sleep 1
done
if [ -z "${KP_OK}" ]; then
    echo "[ritual-launch] FATAL: kplanner pose PUB (:5556) never bound — kplanner dead or crashed. REFUSING IGNITION (pad + pad-e-stop would be dead). Check log/pc2_kplanner.log"
    exit 1
fi

WB="${X2_WHOLE_BODY_TELEOP:-0}"
DUAL_MODEL=""
case "${MODEL}" in
    *_dual.onnx)
        # NATIVE DUAL-HEAD (2026-09-04): X2_RITUAL_MODEL is the dual graph
        # (metadata graph_kind=native_dual_head); its <name>_g1.onnx
        # sibling from the SAME .pt fills the deploy's primary slot. Same
        # MODEL= convention as the sim launcher; no token service.
        DUAL_MODEL="${MODEL}"
        MODEL="${MODEL%_dual.onnx}_g1.onnx"
        [ -f "${MODEL}" ] || { echo "[ritual-launch] FATAL: dual graph needs its pose-graph sibling ${MODEL}"; exit 1; }
        WB=1
        # onnxruntime (present in the PC2 venv), NOT the onnx package (absent
        # there -- the first version of this check died silently, 2026-09-05).
        PAIR_CHECK=$("${PREFIX}/venv/bin/python" - "$MODEL" "$DUAL_MODEL" 2>&1 <<'PYEOF'
import sys
try:
    import onnxruntime as ort
    def meta(p):
        so = ort.SessionOptions(); so.log_severity_level = 3
        return dict(ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map)
    a, d = meta(sys.argv[1]), meta(sys.argv[2])
except Exception as exc:  # noqa: BLE001
    print(f"CHECK-ERROR {exc!r}"); sys.exit(0)
fa, fd = a.get("codec_fingerprint", ""), d.get("codec_fingerprint", "")
key = lambda f: f.split(":")[0] + ":" + f.split(":")[-1][:8]
if d.get("graph_kind") != "native_dual_head":
    print(f"MISMATCH dual graph_kind={d.get('graph_kind')}")
elif not fa or not fd:
    print("UNSTAMPED")
elif key(fa) != key(fd):
    print(f"MISMATCH pose={fa[:20]} dual={fd[:20]} (different checkpoints)")
else:
    print(f"OK {d.get('source_checkpoint','?')} {fd[:20]}")
PYEOF
)
        case "$PAIR_CHECK" in
            MISMATCH*) echo "[ritual-launch] FATAL: dual/pose pairing $PAIR_CHECK — REFUSING (wrong ONNX swap)"; exit 1 ;;
            UNSTAMPED*) echo "[ritual-launch] FATAL: dual/pose graphs unstamped — re-export with export_native_all.sh"; exit 1 ;;
            OK*) echo "[ritual-launch] dual/pose provenance: $PAIR_CHECK" ;;
            *) echo "[ritual-launch] FATAL: dual/pose pairing check could not run: $PAIR_CHECK"; exit 1 ;;
        esac
        ;;
esac
if [ -n "${DUAL_MODEL}" ]; then
    WB_ARGS="--deploy-extra-arg --whole-body-teleop --deploy-extra-arg --dual-model --deploy-extra-arg ${DUAL_MODEL} --deploy-extra-arg --smpl-intent-port --deploy-extra-arg 5573 --deploy-extra-arg --wb-switch-ramp-s --deploy-extra-arg ${X2_WB_SWITCH_RAMP_S:-0.4} ${X2_WB_OPERATOR_ROOT_LEVEL:+--deploy-extra-arg --operator-root-level --deploy-extra-arg ${X2_WB_OPERATOR_ROOT_LEVEL}}"
    pkill -f "pc2_pico_token_servic[e]" 2>/dev/null || true
    echo "[ritual-launch] WHOLE-BODY DUAL-HEAD armed: ${DUAL_MODEL} (deploy binds the SMPL intent stream on :5573; no token service); pad remains primary"
elif [ "${WB}" = "1" ]; then
    # QUEST-SPLIT (2026-08-29): the token service runs HERE on PC2 —
    # local debug echo, local token publish; the laptop streams only
    # intent to the service's :5573. Nothing state-critical on wifi.
    # No silent defaults: the token graph + SMPL tokenizer are bring-your-own
    # (MODELS.md); robot_env.env (from x2_pc2/robot_env.env.template) sets
    # both next to X2_RITUAL_MODEL.
    WB_TOKEN="${X2_WB_TOKEN_MODEL:?robot_env.env must set X2_WB_TOKEN_MODEL (see x2_pc2/robot_env.env.template)}"
    WB_TOKENIZER="${X2_WB_TOKENIZER:?robot_env.env must set X2_WB_TOKENIZER (see x2_pc2/robot_env.env.template)}"
    [ -f "${WB_TOKEN}" ] || { echo "[ritual-launch] FATAL: X2_WB_TOKEN_MODEL not found: ${WB_TOKEN}"; exit 1; }
    [ -f "${WB_TOKENIZER}" ] || { echo "[ritual-launch] FATAL: X2_WB_TOKENIZER not found: ${WB_TOKENIZER}"; exit 1; }
    # CODEC-FINGERPRINT PREFLIGHT (2026-09-01). The tokenizer and the actor
    # must share one codec generation (phi-tables sha stamped as
    # 'codec_fingerprint' in ONNX metadata). A silent default put the
    # armA-era v11release tokenizer in front of the S1-10k actor and the
    # robot ignored the operator's pose on engage — no check in the chain
    # could see it. REFUSE when both stamps exist and differ; WARN (only)
    # when either artifact predates stamping. X2_ALLOW_CODEC_MISMATCH=1
    # overrides for deliberate cross-era experiments.
    # Read the stamps through onnxruntime session metadata (present in the
    # PC2 venv), NOT the onnx package (absent there) -- same as the
    # dual-head pairing check above. A check that cannot run is FATAL.
    if [ "${X2_ALLOW_CODEC_MISMATCH:-0}" != "1" ]; then
        FPR_CHECK=$("${PREFIX}/venv/bin/python" - "$MODEL" "$WB_TOKENIZER" 2>&1 <<'PYEOF'
import sys
try:
    import onnxruntime as ort
    def fpr(p):
        so = ort.SessionOptions(); so.log_severity_level = 3
        meta = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]).get_modelmeta()
        return dict(meta.custom_metadata_map).get("codec_fingerprint", "")
    a, t = fpr(sys.argv[1]), fpr(sys.argv[2])
except Exception as exc:  # noqa: BLE001
    print(f"CHECK-ERROR {exc!r}"); sys.exit(0)
if a and t and a != t:
    print(f"MISMATCH actor={a[:12]} tokenizer={t[:12]}")
elif not a or not t:
    print("UNSTAMPED")
else:
    print(f"OK {a[:12]}")
PYEOF
)
        case "$FPR_CHECK" in
            MISMATCH*)
                echo "FATAL: codec fingerprint $FPR_CHECK" >&2
                echo "       tokenizer and actor are DIFFERENT codec generations;" >&2
                echo "       the robot would not track the operator. Fix" >&2
                echo "       X2_WB_TOKENIZER / X2_RITUAL_MODEL to an era-matched" >&2
                echo "       pair, or set X2_ALLOW_CODEC_MISMATCH=1 deliberately." >&2
                exit 1 ;;
            UNSTAMPED*)
                echo "WARNING: codec fingerprint unstamped on actor and/or tokenizer" >&2
                echo "         (pre-2026-09-01 export) — era match NOT verified." >&2 ;;
            OK*)
                echo "codec fingerprint preflight: $FPR_CHECK" ;;
            *)
                echo "FATAL: codec fingerprint preflight could not run: $FPR_CHECK" >&2
                echo "       (onnxruntime missing in ${PREFIX}/venv, or unreadable ONNX);" >&2
                echo "       set X2_ALLOW_CODEC_MISMATCH=1 only for a deliberate cross-era run." >&2
                exit 1 ;;
        esac
    fi
    WB_ARGS="--deploy-extra-arg --whole-body-teleop --deploy-extra-arg --token-model --deploy-extra-arg ${WB_TOKEN} --deploy-extra-arg --token-zmq-host --deploy-extra-arg localhost --deploy-extra-arg --token-zmq-port --deploy-extra-arg 5574"
    # Sweep any stale service AND wait for :5574/:5573 to free — spawning
    # right after pkill races the dying process's port release: the fresh
    # service crashes on EADDRINUSE while the zombie keeps serving OLD code
    # into the same log (sim rehearsal falls #3/#4, 2026-08-29).
    pkill -f "pc2_pico_token_servic[e]" 2>/dev/null || true
    for _ in $(seq 1 25); do
        ss -tln 2>/dev/null | grep -qE ':5574 |:5573 ' || break
        sleep 0.2
    done
    if ss -tln 2>/dev/null | grep -qE ':5574 |:5573 '; then
        echo "[ritual-launch] FATAL: ports 5574/5573 still bound after stale token-service sweep — refusing whole-body launch"
        exit 1
    fi
    nohup "${PREFIX}/venv/bin/python" "${PREFIX}/gear_sonic/scripts/pc2_pico_token_service.py" \
        --tokenizer "${WB_TOKENIZER}" --debug-port 5557 \
        --planner-cmd-port 5563 \
        ${X2_WB_OPERATOR_ROOT_LEVEL:+--operator-root-level "${X2_WB_OPERATOR_ROOT_LEVEL}"} \
        > "${PREFIX}/log/pico_token_service.log" 2>&1 &
    WB_SVC_PID=$!
    sleep 1.5
    if ! kill -0 "${WB_SVC_PID}" 2>/dev/null; then
        echo "[ritual-launch] FATAL: token service died at startup — log follows:"
        sed 's/^/    | /' "${PREFIX}/log/pico_token_service.log"
        exit 1
    fi
    echo "[ritual-launch] WHOLE-BODY OVERLAY armed: token service pid ${WB_SVC_PID} (intent :5573, tokens localhost:5574); pad remains primary"
else
    WB_ARGS=""
fi

LOG_DIR="${PREFIX}/log/deploy_ritual_$(date +%Y%m%d_%H%M%S)"

# Same post-mortem trap as the autogenerated bodies: keep the tmux pane
# attachable after a crash so the traceback survives `tmux attach`.
_keep_alive() {
    local rc=$?
    echo
    echo "[ritual-launch] cmd exited with status=${rc}"
    if [[ "${rc}" -ne 0 ]]; then
        echo "[ritual-launch] NON-ZERO EXIT -- inspect scrollback above for the real error."
    fi
    echo "[ritual-launch] keeping pane alive (read scrollback above; Ctrl-D to close)"
    exec bash -l
}
trap _keep_alive EXIT
echo "[ritual-launch] pid=$$ model=${MODEL}"
echo "[ritual-launch] tuning=${TUNING}"
echo "[ritual-launch] head_bypass=${HEAD_BYPASS} head_dev=${HEAD_DEV}"
echo "[ritual-launch] log_dir=${LOG_DIR}"
echo "[ritual-launch] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[ritual-launch] ---- deploy body (--no-confirm: ritual chord + pose gate are the operator confirmation) ----"

cd "${PREFIX}/gear_sonic_deploy" && \
    "${PREFIX}/gear_sonic_deploy/deploy_x2.sh" onbot --no-docker --vla \
        --vla-zmq-host "${VLA_HOST}" --vla-zmq-port "${VLA_PORT}" --vla-zmq-topic pose \
        --vla-debug-port 5557 --vla-debug-topic x2_debug \
        --wrist-bypass ik \
        --head-bypass "${HEAD_BYPASS}" \
        --model "${MODEL}" \
        --intra-op-threads "${X2_RITUAL_INTRA_OP:-3}" \
        --log-dir "${LOG_DIR}" \
        --deploy-extra-arg --flight-recorder-seconds \
        --deploy-extra-arg "${X2_FLIGHT_SECONDS:-30}" \
        --onbot-prefix "${PREFIX}" \
        --onbot-ws "${PREFIX}/ws" \
        --onbot-venv "${PREFIX}/venv" \
        --onbot-onnxruntime "${PREFIX}/onnxruntime" \
        --onbot-aimdk-prefix /agibot/software/housekeeper/bin/aimdk_msgs \
        ${WATCHDOG_ARGS} ${WB_ARGS} \
        --tuning-config "${TUNING}" \
        --max-target-dev-head "${HEAD_DEV}" \
        --no-confirm
