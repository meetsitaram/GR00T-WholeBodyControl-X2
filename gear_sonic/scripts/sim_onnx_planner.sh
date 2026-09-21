#!/bin/bash
# Standard SIM launch, gated against the robot: ONNX planner + MuJoCo sim +
# local gamepad. SIM-ONLY -- no --pc2-host, so the pose wire binds 127.0.0.1
# and the robot cannot subscribe.
#
# KPLANNER_PROFILE=<file.env>: source a versioned knob profile (see
# gear_sonic/config/kplanner_profiles/). Profiles use the ":=" idiom, so any
# env you export explicitly before launching still wins. This replaces the
# 14-line env incantation with ONE reviewable, md5-able file per deployment.
if [[ -n "${KPLANNER_PROFILE:-}" ]]; then
  if [[ ! -f "$KPLANNER_PROFILE" ]]; then
    echo "ERROR: KPLANNER_PROFILE=$KPLANNER_PROFILE not found" >&2; exit 1
  fi
  # shellcheck disable=SC1090
  source "$KPLANNER_PROFILE"
  echo "[profile] $KPLANNER_PROFILE (md5 $(md5sum "$KPLANNER_PROFILE" | cut -c1-8))"
fi

# Snapshot X2_*/KPLANNER_* variable NAMES at this exact point: after the
# operator's env and the profile, BEFORE this script computes any of its own
# (KPLANNER_ONNX / KPLANNER_DANCES_DIR / the FWD default are set ~line 440).
# The arg-parity check's "EXTRA" test compares against THIS set, so it flags a
# setting you or the profile introduced -- never the launcher's own plumbing,
# which is local paths by construction and can't match a robot value.
_ENV_AT_ENTRY=" $(compgen -v 2>/dev/null | grep -E '^(X2_|KPLANNER_)' | sort -u | tr '\n' ' ')"
#
#   ./gear_sonic/scripts/sim_onnx_planner.sh <PC2_IP>
#
# The PC2 IP is the ONLY argument, and it is required. Before launching, this
# md5-compares every model you are about to play with against the ones actually
# deployed on the robot, and reports each file's last-modified time on the robot
# so a stale artifact is visible. It then auto-selects the local sonic ONNX whose
# md5 matches the robot's, so "what you test" IS "what the robot runs" by
# construction rather than by hope.
#
# Why the gate exists: it once caught the robot running ft_2082_g1.onnx while
# every sim launcher used walkft_3065_g1.onnx -- i.e. sim had been validating a
# different policy than the robot. A model can pass every numeric and visual
# check locally and still not be the model on the robot.
#
# ONNX-ONLY, to match the robot: PC2 has no .pt files at all. The .pt the stack
# would otherwise auto-resolve is the *recorder's* VLA tokenizer, not the deploy
# policy, and the robot never runs the recorder -- so opting out is the
# robot-faithful configuration.
#
# DEFAULTS (no env at all): the SHIPPED set gear_sonic_deploy/models/
# x2_sonic_v16ft8_45000_{g1,dual,g1_token,smpl_tokenizer}.onnx (git-lfs; run
# `git lfs pull` if they are pointer files) and the shipped tuning preset
# gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml.
#   ./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop   # dual-head, shipped set
#   ./gear_sonic/scripts/sim_onnx_planner.sh                       # pad / planner, shipped _g1
#
# Overrides (env, for pre-deploy candidates). Either one intentionally trips the
# gate, and a tripped gate now ABORTS unless ALLOW_MISMATCH=1 is also set --
# testing a candidate must be deliberate, never something you drift into.
#   SONIC_MODEL=<path_g1.onnx>   candidate sonic policy       (alias: MODEL=)
#   PLANNER_MODEL=<dir|file>     candidate planner graphs (dir holding
#                                x2_planner_{template,velocity}.onnx, or one of them)
#   ALLOW_MISMATCH=1             acknowledge the gate and run anyway
#   PLANNER=velocity             use the velocity graph (default: template)
#   SONIC_CKPT=<path.pt>         ONLY when deliberately recording a VLA dataset
#
# Both models are gated independently, so you can vary one and hold the other at
# the robot's version -- which is what makes a candidate result attributable:
#   ALLOW_MISMATCH=1 SONIC_MODEL=<cand>   ./sim_onnx_planner.sh <ip>   # sonic only
#   ALLOW_MISMATCH=1 PLANNER_MODEL=<dir>  ./sim_onnx_planner.sh <ip>   # planner only
#
# Controls: L2 = deadman drive (locked 0.3), L1+Y/A = cycle dances, L1+B = stop.
# Needs a pad visible to pygame BEFORE launch (--pad-only tears the stack down
# otherwise) and the env_isaaclab conda env.
#
# --vr: dual-source input (pad + Quest 3). Spawns quest3_manager_x2 alongside
# the pad bridge (stack --pad-and-vr); the kplanner SUB binds :5563 and both
# sources PUB-connect. Open https://<laptop-ip>:8443 in the Quest 3 browser,
# A+B+X+Y to engage; the pad keeps working whenever its deadman is held.
#
# --whole-body-teleop: hands off to the WHOLE-BODY Pico SMPL teleop
# (--full-body is the deprecated alias, kept so older runbooks keep working;
#  the name matches the deploy-side --whole-body-teleop mode, 2026-08-29)
# (run_x2_pico_wbc.sh) so the familiar command shape launches it — no
# kplanner/PC2 in that path; planner-stack knobs are ignored with a notice.
# Args after --whole-body-teleop pass through to the launcher (e.g. --tape-replay).
set -euo pipefail
cd "$(dirname "$0")/../.."
# Shipped default model set (gear_sonic_deploy/models/README.md): used when no
# MODEL= / SONIC_MODEL= / WHOLE_BODY_MODEL= is given. A clone without
# `git lfs pull` holds ~130-byte pointer files there, which are NOT usable.
SHIPPED_MODELS_DIR="$PWD/gear_sonic_deploy/models"
SHIPPED_STEM="$SHIPPED_MODELS_DIR/x2_sonic_v16ft8_45000"
SHIPPED_TUNING="gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml"
shipped_usable() { [[ -f "$1" ]] && ! head -c 40 "$1" | grep -q 'git-lfs'; }
shipped_note() {   # shipped_note <file>: why the shipped file is not usable
  if [[ -f "$1" ]]; then echo "LFS pointer (run: git lfs pull)"; else echo "missing"; fi
}
# kplanner interpreter: KPLANNER_PYTHON, else ISAACLAB_PYTHON, else the repo .venv, else python3.
KPY="${KPLANNER_PYTHON:-${ISAACLAB_PYTHON:-$PWD/.venv/bin/python}}"
command -v "$KPY" >/dev/null 2>&1 || KPY="$(command -v python3 || true)"
[[ -n "$KPY" ]] || { echo "ERROR: no python interpreter for the kplanner (create $PWD/.venv or set KPLANNER_PYTHON)" >&2; exit 1; }
KPY="$(command -v "$KPY")"

# Accept a bare IP or the --pc2-host/--pc2 flag form: the flag form is what the
# demo command sheet documents, and silently treating "--pc2-host" as a hostname
# produces a baffling "cannot reach run@--pc2-host".
PC2_IP=""
VR_MODE=0
FULL_BODY=0
DRY_RUN=0
FB_EXTRA=()
ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)        sed -n '2,73p' "$0"; echo; echo "  --dry-run   resolve models/dances/pad banks, print them, exit (no gate, no launch)"; exit 0 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --pc2-host|--pc2) PC2_IP="${2:-}"; shift 2 ;;
    --vr)             VR_MODE=1; shift ;;
    --vr-only)        VR_MODE=2; shift ;;   # Quest 3 only, NO pad bridge (no gamepad needed)
    --whole-body-teleop) FULL_BODY=1; shift; FB_EXTRA=("$@"); set -- ;;
    --full-body)      echo "[whole-body] note: --full-body is the old name; use --whole-body-teleop" >&2
                      FULL_BODY=1; shift; FB_EXTRA=("$@"); set -- ;;
    --*)              echo "unknown option: $1" >&2; shift ;;
    *)                PC2_IP="$1"; shift ;;
  esac
done

# --whole-body-teleop: hand off to the WHOLE-BODY Pico SMPL teleop (run_x2_pico_wbc.sh)
# so the familiar stack command shape launches it. That stack is STANDALONE and
# SIM-ONLY: no kplanner, no PC2 — the planner-stack knobs (--pc2-host,
# PLANNER_MODEL, KPLANNER_PROFILE, ALLOW_MISMATCH, PC2_PREFIX) do not apply and
# are ignored here. MODEL= maps through only when it is usable by the smpl
# path: a *smpl*.onnx goes to --onnx, a .pt/composite string to --model; a
# g1-encoder planner graph (any other .onnx) CANNOT drive the smpl encoder, so
# it falls back to the full-body default with a warning. Args after
# --whole-body-teleop pass straight through to run_x2_pico_wbc.sh.
if (( FULL_BODY )); then
  # DEPLOY-FAITHFUL rehearsal (2026-08-29): a *_token.onnx MODEL= (or
  # WHOLE_BODY_MODEL=) routes to the FULL-LOOP simstack — the actual deploy
  # binary in docker with --whole-body-teleop, connecting to the SAME
  # pico_token_sender.py command the robot session uses. Two commands,
  # mirroring real deployment exactly. Composite .pt / *smpl*.onnx models
  # keep the lightweight in-process stack below.
  REH_MODEL="${WHOLE_BODY_MODEL:-${FULL_BODY_MODEL:-${MODEL:-}}}"
  # DEFAULT (no MODEL= at all): the shipped dual-head graph. Its _g1 sibling
  # sits next to it in the same directory, so the *_dual.onnx case below
  # resolves the pair exactly as for a bring-your-own set.
  if [[ -z "$REH_MODEL" ]]; then
    if shipped_usable "${SHIPPED_STEM}_dual.onnx"; then
      REH_MODEL="${SHIPPED_STEM}_dual.onnx"
      echo "[whole-body] no MODEL= -> shipped default set (gear_sonic_deploy/models/README.md):"
      echo "[whole-body]   MODEL=$REH_MODEL"
    else
      echo "[whole-body] shipped default ${SHIPPED_STEM}_dual.onnx is $(shipped_note "${SHIPPED_STEM}_dual.onnx"); set MODEL= (MODELS.md)" >&2
    fi
  fi
  # --dry-run given before --whole-body-teleop: forward it to simstack_local.sh
  # (which resolves every path, prints them and exits without launching).
  if (( DRY_RUN )); then
    _has_dry=0; for _a in ${FB_EXTRA[@]+"${FB_EXTRA[@]}"}; do [[ "$_a" == "--dry-run" ]] && _has_dry=1; done
    (( _has_dry )) || FB_EXTRA=(--dry-run ${FB_EXTRA[@]+"${FB_EXTRA[@]}"})
  fi
  # convenience: MODEL= pointing at the planner graph auto-derives its
  # sibling token graph (same exported dir, same checkpoint) so the SAME
  # MODEL= line works for --vr and --whole-body-teleop sessions.
  if [[ "$REH_MODEL" == *.onnx && "$REH_MODEL" != *_token.onnx && "$REH_MODEL" != *_dual.onnx && "$REH_MODEL" != *smpl*.onnx ]]; then
    SIB="${REH_MODEL%.onnx}_token.onnx"
    if [[ -f "$SIB" ]]; then
      echo "[whole-body] MODEL is the planner graph; using its token sibling:"
      echo "[whole-body]   $SIB"
      REH_MODEL="$SIB"
    fi
  fi
  # NATIVE 3-encoder lineage (native_token_onnx_export.py, 2026-09-02): the
  # token graph ships with its OWN smpl tokenizer sibling
  # (<stem>_smpl_tokenizer.onnx next to <stem>_g1_token.onnx). A native
  # decoder fed the default v11release (frozen-G1) tokens tests the wrong
  # policy, so the sibling is auto-selected; explicit SIMSTACK_TOKENIZER wins.
  if [[ "$REH_MODEL" == *_g1_token.onnx && -z "${SIMSTACK_TOKENIZER:-}" ]]; then
    TOKZ="${REH_MODEL%_g1_token.onnx}_smpl_tokenizer.onnx"
    if [[ -f "$TOKZ" ]]; then
      echo "[whole-body] native lineage: using the token graph's own tokenizer:"
      echo "[whole-body]   $TOKZ"
      export SIMSTACK_TOKENIZER="$TOKZ"
    fi
  fi
  # The planner-stack knobs on the operator's command sheet MUST mean the
  # same thing here (operator ask 2026-09-02 "make sure those args are
  # respected"): simstack_local.sh sources robot_env.env, whose ':=' defaults
  # win unless the profile is sourced FIRST — so source it here. PLANNER_MODEL
  # passes through (simstack_local.sh honors it). ALLOW_MISMATCH / --pc2-host
  # are robot-gate knobs with nothing to gate in a sim-only stack: say so.
  if [[ -n "${KPLANNER_PROFILE:-}" ]]; then
    [[ -f "$KPLANNER_PROFILE" ]] || { echo "[whole-body] ERROR: KPLANNER_PROFILE=$KPLANNER_PROFILE not found" >&2; exit 2; }
    # shellcheck disable=SC1090
    . "$KPLANNER_PROFILE"
    echo "[whole-body] kplanner profile: $KPLANNER_PROFILE (turn=${KPLANNER_FIXED_TURN_RAD_S:-?} rad/s fwd=${KPLANNER_FIXED_FWD_MPS:-?} m/s replan=${KPLANNER_REPLAN_THRESHOLD_FRAMES:-?})"
  fi
  [[ -n "${PLANNER_MODEL:-}" ]] && echo "[whole-body] planner: PLANNER_MODEL=$PLANNER_MODEL"
  [[ -n "$PC2_IP" || "${ALLOW_MISMATCH:-0}" == "1" ]] && \
    echo "[whole-body] note: --pc2-host / ALLOW_MISMATCH are robot-gate knobs; sim-only stack, nothing to gate (accepted, no effect)."
  case "$REH_MODEL" in
    *_dual.onnx)
      # NATIVE DUAL-HEAD (2026-09-04): MODEL= is the dual graph itself
      # (metadata graph_kind=native_dual_head, exported next to its
      # <name>_g1.onnx sibling from the SAME .pt). No second variable.
      # simstack_local.sh proves the pairing (onnx_provenance.py) before
      # the deploy starts.
      DUAL_ABS="${REH_MODEL/#\~/$HOME}"
      SIB_G1="${DUAL_ABS%_dual.onnx}_g1.onnx"
      [[ -f "$SIB_G1" ]] || { echo "[whole-body] ERROR: dual graph needs its pose-graph sibling next to it: $SIB_G1" >&2; exit 2; }
      echo "[whole-body] NATIVE DUAL-HEAD graph — one SONIC session, both reference heads"
      echo "[whole-body]   dual: $DUAL_ABS"
      echo "[whole-body]   pose sibling (deploy primary slot): $SIB_G1"
      echo "[whole-body] second command (laptop half, run alongside):"
      echo "[whole-body]   ./gear_sonic/scripts/run_pico_teleop.sh"
      export X2_WHOLE_BODY_TELEOP=1
      export SIMSTACK_DUAL_MODEL="$DUAL_ABS"
      export SIMSTACK_SONIC="$SIB_G1"
      exec ./gear_sonic/scripts/simstack_local.sh ${FB_EXTRA[@]+"${FB_EXTRA[@]}"}
      ;;
    *_token.onnx)
      echo "[whole-body] TOKEN GRAPH — deploy-faithful rehearsal via simstack_local.sh"
      echo "[whole-body] second command (laptop half, run alongside):"
      echo "[whole-body]   ./gear_sonic/scripts/run_pico_teleop.sh"
      export X2_WHOLE_BODY_TELEOP=1
      export SIMSTACK_TOKEN="$REH_MODEL"
      # primary = the pose/planner graph (the original MODEL= if it was one)
      if [[ -n "${MODEL:-}" && "${MODEL}" != *_token.onnx ]]; then
        export SIMSTACK_SONIC="${MODEL/#\~/$HOME}"
      fi
      exec ./gear_sonic/scripts/simstack_local.sh ${FB_EXTRA[@]+"${FB_EXTRA[@]}"}
      ;;
  esac
  echo "[whole-body] whole-body Pico SMPL teleop — standalone SIM-ONLY stack;"
  echo "[whole-body] planner/PC2 settings in this command are ignored (no kplanner in this path)."
  FB_ARGS=()
  FB_MODEL="${WHOLE_BODY_MODEL:-${FULL_BODY_MODEL:-${MODEL:-}}}"
  if [[ -n "$FB_MODEL" ]]; then
    case "$FB_MODEL" in
      *smpl*.onnx) FB_ARGS=(--onnx "$FB_MODEL") ;;
      *.onnx)
        # NO silent fallback (operator ruling 2026-08-30: a substituted
        # default model tests the WRONG POLICY and reads as a pass).
        echo "[whole-body] ERROR: MODEL=$FB_MODEL is a g1-encoder planner graph — the" >&2
        echo "[whole-body]        smpl path cannot use it. Refusing to substitute a default." >&2
        echo "[whole-body]        Pass a *smpl*.onnx, a 3-encoder .pt, or an frozen-core-smpl:<merged.pt>" >&2
        echo "[whole-body]        composite via WHOLE_BODY_MODEL= (or drop MODEL= for the documented default)." >&2
        exit 2
        ;;
      *) FB_ARGS=(--model "$FB_MODEL") ;;
    esac
  fi
  exec env -u MODEL ./gear_sonic/scripts/run_x2_pico_wbc.sh \
      ${FB_ARGS[@]+"${FB_ARGS[@]}"} ${FB_EXTRA[@]+"${FB_EXTRA[@]}"}
fi

# TARGET RULE (operator 2026-09-05): no --pc2-host = SIM-ONLY, always (no robot
# identity comparison, results are CANDIDATE-ONLY); an ip = compare against
# that real robot (read-only ssh). Same rule as run_pico_teleop.sh /
# run_pico_replay.sh: never a robot default, never detection.
GATE_OFFLINE=0
if [[ -z "$PC2_IP" ]]; then
  echo "[sim-stack] no --pc2-host -> SIM-ONLY run: no robot identity gate, results are CANDIDATE-ONLY."
  echo "[sim-stack] certification against the robot = pass --pc2-host <PC2 ip> (read-only compare)."
  GATE_OFFLINE=1
fi

# Canonical SONIC model cache (multi-embodiment, MODELS.md): $SONIC_HOME
# (default ~/.cache/sonic) with X2 artifacts under $SONIC_HOME/x2; the X2
# subtree alone is overridable via $SONIC_X2_MODELS. CKPT_ROOT (extra
# exports / cloud checkpoints, searched for md5 matches) defaults to the
# same X2 subtree. Identity gating vs the robot is unchanged: explicit
# SONIC_MODEL/PLANNER_MODEL env > md5 auto-match (CKPT_ROOT AND the cache).
SONIC_X2_MODELS_DIR="${SONIC_X2_MODELS:-${SONIC_HOME:-$HOME/.cache/sonic}/x2}"
CKPT_ROOT="${CKPT_ROOT:-$SONIC_X2_MODELS_DIR}"
GRAPH="${PLANNER:-template}"   # template | velocity
# Dance / clip bank (x2m2) for the kplanner: regenerated on a clean clone by
# tools/build_demo_bank_from_upstream.sh.
DANCES="${KPLANNER_DANCES_DIR:-$PWD/gear_sonic_deploy/data/motions_x2m2/demo_bank}"
LOCAL_PLANNER_DIR="$SONIC_X2_MODELS_DIR/kplanner_onnx"
[[ -d "$CKPT_ROOT/planner_onnx" ]] && LOCAL_PLANNER_DIR="$CKPT_ROOT/planner_onnx"
# Robot-side install prefix (the PC2 deploy tree).
PC2_PREFIX="${PC2_PREFIX:-/home/run/gear-sonic}"

# The HF cache names planner graphs x2_kplanner_{template,velocity}.onnx while
# the robot/cloud dirs use x2_planner_*.onnx. Resolve whichever exists.
planner_graph() {  # planner_graph <dir> <template|velocity> -> path (or empty)
  local d="$1" g="$2"
  if [[ -f "$d/x2_planner_$g.onnx" ]]; then echo "$d/x2_planner_$g.onnx"
  elif [[ -f "$d/x2_kplanner_$g.onnx" ]]; then echo "$d/x2_kplanner_$g.onnx"
  fi
}
R_SONIC=$PC2_PREFIX/policies/agibot_x2_sonic.onnx
R_PLAN=$PC2_PREFIX/planner_stack/models/planner_onnx
R_RUNTIME=$PC2_PREFIX/pc2_kplanner_onnx.py
# The deploy BINARY. It HARDCODES the plant (kps/kds/action scales/defaults in
# policy_parameters.hpp) and applies it to WHATEVER model it is handed. The gate
# was blind to it until 2026-08-24, when a robot binary built before the vendor
# plant change was fed a vendor-exported ONNX: 29/31 action scales disagreed, up
# to x2.86, and the waist pinned at its +-0.45 clamp on 26/45 ticks. The gate
# reported CLEAN throughout, because every file it DID hash matched.
R_BIN=$PC2_PREFIX/ws/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref
R_RITUAL=$PC2_PREFIX/start_x2_deploy_ritual.sh
R_PROFILE=$PC2_PREFIX/robot_env.env
# The IGNITION script -- the parent caller. It starts the kplanner daemon and
# owns its entire argv (planner-mode, replan cadence, warmup anchor, dance bank,
# yaw resync). The gate hashed the planner GRAPHS but never how the planner is
# INVOKED, so a sim could drive the right model through a different daemon.
R_IGNITION=$PC2_PREFIX/ritual_start_sonic.sh
L_IGNITION=x2_pc2/ritual_start_sonic.sh
# The repo's committed mirror of the ritual (per the "robot files live in repo"
# convention). The gate compares the two: if the robot's copy has drifted, every
# argument derived from the local mirror below is fiction.
L_RITUAL=x2_pc2/start_x2_deploy_ritual.sh

if (( DRY_RUN )); then
  _ok() { [[ -e "$1" ]] && echo "ok " || echo "MISSING"; }
  _cand="${SONIC_MODEL:-${MODEL:-}}"
  [[ -z "$_cand" ]] && shipped_usable "${SHIPPED_STEM}_g1.onnx" && _cand="${SHIPPED_STEM}_g1.onnx"
  [[ -z "$_cand" && -f "$SONIC_X2_MODELS_DIR/sonic_policy/x2_sonic_policy.onnx" ]] && _cand="$SONIC_X2_MODELS_DIR/sonic_policy/x2_sonic_policy.onnx"
  [[ -z "$_cand" ]] && _cand="$(find "$CKPT_ROOT" -name '*_g1.onnx' -type f 2>/dev/null | head -1)"
  _tmpl="$(planner_graph "$LOCAL_PLANNER_DIR" "$GRAPH")"
  echo "=== DRY RUN (nothing launched) ==="
  echo "  repo root        : $PWD"
  echo "  SONIC_HOME/x2    : $SONIC_X2_MODELS_DIR"
  echo "  CKPT_ROOT        : $CKPT_ROOT"
  echo "  PC2_PREFIX       : $PC2_PREFIX   (pc2 host: ${PC2_IP:-<none: sim-only>})"
  echo "  sonic candidate  : ${_cand:-<none: set MODEL= / SONIC_MODEL=, see MODELS.md>}  $([[ -n "$_cand" ]] && _ok "$_cand")"
  echo "  shipped set      : ${SHIPPED_STEM}_{g1,dual,g1_token,smpl_tokenizer}.onnx  ($(shipped_usable "${SHIPPED_STEM}_g1.onnx" && echo "ok " || shipped_note "${SHIPPED_STEM}_g1.onnx"))"
  echo "  tuning (sim)     : ${SIM_TUNING_YAML-$SHIPPED_TUNING (shipped default)}"
  echo "  planner dir      : $LOCAL_PLANNER_DIR  ($(_ok "$LOCAL_PLANNER_DIR"))"
  echo "  planner graph    : ${_tmpl:-$LOCAL_PLANNER_DIR/x2_planner_$GRAPH.onnx}  ($(_ok "${_tmpl:-/nonexistent}"))"
  echo "  dances (x2m2)    : $DANCES  ($(_ok "$DANCES"), $(ls "$DANCES"/*.x2m2 2>/dev/null | wc -l) clips)"
  echo "  pad clip pkl     : ${PAD_CLIP_PKL:-gear_sonic/data/motions/x2_demo_bank.pkl}  ($(_ok "${PAD_CLIP_PKL:-gear_sonic/data/motions/x2_demo_bank.pkl}"))"
  echo "  warmup anchor    : gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl  ($(_ok gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl))"
  echo "  primitives pkl   : gear_sonic/data/motions/x2_planner_primitives.pkl  ($(_ok gear_sonic/data/motions/x2_planner_primitives.pkl))"
  echo "  stack script     : gear_sonic/scripts/run_x2_quest3_planner_stack.sh  ($(_ok gear_sonic/scripts/run_x2_quest3_planner_stack.sh))"
  echo "  kplanner python  : $KPY"
  echo "  regenerate banks : tools/build_demo_bank_from_upstream.sh ; models: MODELS.md"
  exit 0
fi

# ---------------------------------------------------------------------------
# Fetch md5 + mtime for every robot-side artifact in ONE ssh round trip.
# ---------------------------------------------------------------------------
if [[ -z "$PC2_IP" ]]; then
  echo "=== PC2 IDENTITY GATE: SKIPPED (sim-only, no --pc2-host) ==="
  REMOTE=""; REMOTE_RC=0
else
echo "=== PC2 IDENTITY GATE ($PC2_IP) ==="
REMOTE=$(ssh -o ConnectTimeout=8 "run@$PC2_IP" "
  for f in $R_SONIC $R_PLAN/x2_planner_template.onnx $R_PLAN/x2_planner_velocity.onnx; do
    if [ -f \"\$f\" ]; then
      echo \"\$f \$(md5sum \"\$f\" | cut -d' ' -f1) \$(date -r \"\$f\" '+%Y-%m-%d_%H:%M:%S')\"
    else
      echo \"\$f MISSING -\"
    fi
  done
  echo \"handoff \$(grep -c get_next_frame_resampled $R_RUNTIME 2>/dev/null || echo 0) \$(date -r $R_RUNTIME '+%Y-%m-%d_%H:%M:%S' 2>/dev/null || echo -)\"
  if [ -f $R_BIN ]; then
    echo \"bin \$(md5sum $R_BIN | cut -d' ' -f1) \$(date -r $R_BIN '+%Y-%m-%d_%H:%M:%S')\"
  else echo \"bin MISSING -\"; fi
  if [ -f $R_RITUAL ]; then
    echo \"ritual \$(md5sum $R_RITUAL | cut -d' ' -f1) \$(date -r $R_RITUAL '+%Y-%m-%d_%H:%M:%S')\"
  else echo \"ritual MISSING -\"; fi
  if [ -f $R_IGNITION ]; then
    echo \"ignition \$(md5sum $R_IGNITION | cut -d' ' -f1) \$(date -r $R_IGNITION '+%Y-%m-%d_%H:%M:%S')\"
  else echo \"ignition MISSING -\"; fi
  # Resolve the ritual's BEHAVIOURAL argv exactly the way the ritual itself does:
  # source kplanner_profile.env, then apply the ritual's own defaults. Read-only --
  # this sources an env file and echoes; it starts nothing.
  ( P=$PC2_PREFIX
    [ -f $R_PROFILE ] && . $R_PROFILE
    T=\"\${X2_RITUAL_TUNING:-\$P/gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml}\"
    HB=\"\${X2_RITUAL_HEAD_BYPASS:-ref}\"
    if [ \"\$HB\" = ref ]; then HD=\"\${X2_RITUAL_HEAD_DEV:-0.40}\"; else HD=\"\${X2_RITUAL_HEAD_DEV:-0.01}\"; fi
    echo \"rtuning \$T -\"
    echo \"rheadbypass \$HB -\"
    echo \"rheaddev \$HD -\"
    echo \"rmodel \$(basename \"\${X2_RITUAL_MODEL:-\$P/policies/x2_sonic_v16ft8_45000_dual.onnx}\") -\"
    if [ -f \"\$T\" ]; then
      echo \"rtunefp \$(grep -vE '^[[:space:]]*(#|$)' \"\$T\" | sed 's/#.*//; s/[[:space:]]*$//' | grep -v '^$' | sort | md5sum | cut -d' ' -f1) -\"
    else echo \"rtunefp MISSING -\"; fi
  )
  # Every variable the robot's env file actually resolves to, by name. Emitting
  # them generically (rather than a hard-coded list) means a variable added to
  # the robot later is compared automatically instead of silently escaping the
  # gate -- which is exactly how X2_ANCHOR_ORI_MODE went unchecked for two days.
  ( P=$PC2_PREFIX
    E=\$P/robot_env.env
    [ -f \"\$E\" ] && . \"\$E\"
    [ -e \$P/kplanner_profile.env ] && [ ! -L \$P/kplanner_profile.env ] && echo \"rstaleenv kplanner_profile.env-is-a-real-file\"
    echo \"renvfile \$E \$(md5sum \"\$E\" 2>/dev/null | cut -d' ' -f1)\"
    for v in \$(grep -oE '^[[:space:]]*:[[:space:]]*\"\\\$\{[A-Z_0-9]+' \"\$E\" | grep -oE '[A-Z_0-9]+\$' | sort -u); do
      eval \"echo \\\"renv \$v \\\${\$v}\\\"\"
    done
  )
" 2>/dev/null) && REMOTE_RC=0 || REMOTE_RC=$?
fi
if [[ $REMOTE_RC -ne 0 ]]; then
  # PC2 unreachable. A sim-only validation run must NOT require the robot to be
  # online -- the whole point is to test a candidate BEFORE deploying. So:
  #   ALLOW_MISMATCH=1 -> warn and proceed with NO identity comparison.
  #   otherwise        -> abort (a certification run must confirm the robot).
  if [[ "${ALLOW_MISMATCH:-0}" == "1" ]]; then
    echo "  WARN: cannot reach run@$PC2_IP -- no robot identity comparison possible."
    echo "        Proceeding validates the CANDIDATE with NO check against the robot."
    # This USED to also demand a manual y/N keypress on /dev/tty, reasoning
    # that skipping the robot check should be deliberate every single run.
    # Dropped 2026-08-23: ALLOW_MISMATCH=1 IS already the deliberate act --
    # you have to type it, it is never a default, and it is not something you
    # drift into. The second confirmation added no real safety (nobody sets
    # ALLOW_MISMATCH=1 then reconsiders at the prompt) while making the
    # launcher unusable non-interactively: it blocks scripted A/B sweeps, and
    # during the 2026-08-23 model comparison it forced a human keypress on
    # every one of ~20 runs.
    #
    # The safety that MATTERS is kept and made louder: the banner states
    # plainly that no robot comparison happened, and GATE_OFFLINE=1 is
    # threaded to the run summary so an offline result can never be mistaken
    # for one certified against the robot.
    echo "  OFFLINE MODE -- no robot comparison was performed."
    echo "  Results are CANDIDATE-ONLY and are NOT certified against the robot."
    REMOTE=""
    GATE_OFFLINE=1
  else
    echo "  ERROR: cannot reach run@$PC2_IP" >&2
    echo "         (set ALLOW_MISMATCH=1 to validate a candidate without the robot)" >&2
    exit 3
  fi
fi

RITUAL_BYPASS_FLAGS=()   # filled by the arg-parity check when the robot is reachable
r_field() { echo "$REMOTE" | awk -v k="$1" '$1 ~ k {print $2; exit}'; }
r_time()  { echo "$REMOTE" | awk -v k="$1" '$1 ~ k {print $3; exit}'; }

REMOTE_SONIC_MD5=$(r_field "agibot_x2_sonic.onnx"); REMOTE_SONIC_T=$(r_time "agibot_x2_sonic.onnx")
HF=$(r_field "^handoff$");                          HF_T=$(r_time "^handoff$")

# --- sonic: prefer the local ONNX that MATCHES the robot (unless overridden) ---
# SONIC_MODEL is the preferred name; MODEL kept as a backward-compatible alias.
SONIC_ONNX="${SONIC_MODEL:-${MODEL:-}}"
AUTO_NOTE=""
if [[ -z "$SONIC_ONNX" ]]; then
  # Find the local export whose bytes equal the robot's. This is the whole point:
  # select by identity, never by a hardcoded path that can silently drift.
  while IFS= read -r f; do
    if [[ "$(md5sum "$f" | cut -d' ' -f1)" == "$REMOTE_SONIC_MD5" ]]; then
      SONIC_ONNX="$f"; AUTO_NOTE=" (auto-selected: matches robot)"; break
    fi
  done < <(
    shipped_usable "${SHIPPED_STEM}_g1.onnx" && echo "${SHIPPED_STEM}_g1.onnx"
    find "$CKPT_ROOT" -name '*_g1.onnx' -type f 2>/dev/null
    [[ -f "$SONIC_X2_MODELS_DIR/sonic_policy/x2_sonic_policy.onnx" ]] && \
      echo "$SONIC_X2_MODELS_DIR/sonic_policy/x2_sonic_policy.onnx"
  )
  # Sim-only (no robot to match against): the shipped default set.
  if [[ -z "$SONIC_ONNX" && "$GATE_OFFLINE" -eq 1 ]]; then
    if shipped_usable "${SHIPPED_STEM}_g1.onnx"; then
      SONIC_ONNX="${SHIPPED_STEM}_g1.onnx"; AUTO_NOTE=" (shipped default set)"
      echo "  sonic     : no MODEL= -> shipped default $(basename "$SONIC_ONNX")"
    else
      echo "  sonic     : shipped default ${SHIPPED_STEM}_g1.onnx is $(shipped_note "${SHIPPED_STEM}_g1.onnx")"
    fi
  fi
fi

SONIC_OK=0
if [[ -n "$SONIC_ONNX" && -f "$SONIC_ONNX" ]]; then
  LOCAL_SONIC_MD5=$(md5sum "$SONIC_ONNX" | cut -d' ' -f1)
  if [[ "$LOCAL_SONIC_MD5" == "$REMOTE_SONIC_MD5" ]]; then
    echo "  sonic     : MATCH    $(basename "$SONIC_ONNX")  [robot mtime $REMOTE_SONIC_T]"
    SONIC_OK=1
  else
    echo "  sonic     : *** MISMATCH ***  local=${LOCAL_SONIC_MD5:0:12} robot=${REMOTE_SONIC_MD5:0:12}"
    echo "              testing $(basename "$SONIC_ONNX") -- NOT what the robot runs"
    echo "              [robot artifact mtime $REMOTE_SONIC_T]"
  fi
else
  echo "  sonic     : *** NO LOCAL MATCH *** for robot md5 ${REMOTE_SONIC_MD5:0:12}"
  echo "              [robot mtime $REMOTE_SONIC_T] -- robot runs something not present locally."
  echo "              Pass MODEL=<path_g1.onnx> to choose explicitly."
  exit 4
fi

# --- planner graphs: select the local dir by IDENTITY, not by a fixed path ---
# A hardcoded dir drifts the moment new graphs are shipped to the robot. This
# once left every sim run driving planner_onnx/ while the robot ran the FT trio
# in planner_onnx_ft/ -- so sim was validating a different planner than deploy.
# Prefer whichever local dir actually matches the robot.
R_TMPL_MD5=$(r_field "x2_planner_template")
PLANNER_NOTE=""
if [[ -n "${PLANNER_MODEL:-}" ]]; then
  # Explicit candidate planner. Accept either the directory holding the graphs
  # or one of the graph files (we need the dir, since both graphs are checked).
  if [[ -d "$PLANNER_MODEL" ]]; then
    LOCAL_PLANNER_DIR="$PLANNER_MODEL"
  elif [[ -f "$PLANNER_MODEL" ]]; then
    LOCAL_PLANNER_DIR="$(dirname "$PLANNER_MODEL")"
  else
    echo "  ERROR: PLANNER_MODEL=$PLANNER_MODEL is neither a file nor a directory" >&2
    exit 7
  fi
  PLANNER_NOTE=" (explicit PLANNER_MODEL)"
  echo "  planner dir: $(basename "$LOCAL_PLANNER_DIR")/ -- explicit override, not auto-selected"
else
  for d in "$CKPT_ROOT"/planner_onnx* "$SONIC_X2_MODELS_DIR/kplanner_onnx"; do
    [[ -d "$d" ]] || continue
    c="$(planner_graph "$d" template)"; [[ -n "$c" ]] || continue
    if [[ "$(md5sum "$c" | cut -d' ' -f1)" == "$R_TMPL_MD5" ]]; then
      if [[ "$d" != "$LOCAL_PLANNER_DIR" ]]; then
        echo "  planner dir: using $(basename "$d")/ (matches robot; $(basename "$LOCAL_PLANNER_DIR")/ is stale)"
      fi
      LOCAL_PLANNER_DIR="$d"; break
    fi
  done
fi

PLAN_OK=1
declare -A PLAN_T=()
for g in x2_planner_template x2_planner_velocity; do
  R=$(r_field "$g"); T=$(r_time "$g"); PLAN_T[$g]="$T"
  LG="$(planner_graph "$LOCAL_PLANNER_DIR" "${g#x2_planner_}")"
  L=$([[ -n "$LG" ]] && md5sum "$LG" | cut -d' ' -f1 || echo "")
  if [[ -n "$L" && "$L" == "$R" ]]; then
    echo "  $g: MATCH    [robot mtime $T]"
  else
    echo "  $g: MISMATCH local=${L:0:12} robot=${R:0:12}  [robot mtime $T]"; PLAN_OK=0
  fi
done

# --- handoff fix present in the robot's runtime? ---
if [[ "${HF:-0}" -gt 0 ]]; then
  echo "  handoff fix: PRESENT ($HF markers)  [robot mtime $HF_T]"; HF_OK=1
else
  echo "  handoff fix: *** ABSENT *** -- robot runtime lacks the 30->50Hz resample!"
  echo "               [robot mtime $HF_T]"; HF_OK=0
fi

  # --- deploy BINARY: the plant it was compiled with ----------------------
  # Cannot be md5-compared against anything local: it is an ARM build made ON
  # PC2, so there is no host-side twin. Report it, and SHOUT if it predates the
  # model -- that ordering is precisely the 2026-08-24 failure (vendor ONNX on a
  # binary built before the vendor plant landed).
  R_BINMD=$(r_field "^bin$"); R_BIN_T=$(r_time "^bin$")
  R_SONIC_T=$(r_time "agibot_x2_sonic.onnx")
  echo "  deploy bin : md5=${R_BINMD:0:12}  built=$R_BIN_T"
  if [[ -n "$R_BIN_T" && -n "$R_SONIC_T" && "$R_BIN_T" < "$R_SONIC_T" ]]; then
    echo "               *** BINARY PREDATES THE MODEL ***"
    echo "               binary $R_BIN_T  <  sonic $R_SONIC_T"
    echo "               The binary hardcodes kps/kds/action-scales and applies"
               echo "               them to whatever model it is given. If the plant changed"
    echo "               since this build, EVERY command is mis-scaled. Rebuild on"
    echo "               PC2 (onbot) before igniting, or push a model exported"
    echo "               against the plant this binary was built with."
    BIN_STALE=1
  else
    BIN_STALE=0
  fi

  # --- LOCAL C++ AHEAD OF THE ROBOT BUILD? ------------------------------
  # Operator ask 2026-08-24: "flag if any c++ related code changes on local
  # are ahead of the robot build time". The binary is aarch64 and built ON
  # PC2, so there is no md5 to compare against -- but the SOURCE is shared,
  # and a local source file newer than the robot binary means the robot is
  # running code we have already changed here. That is exactly how the
  # 2026-08-24 vendor-plant mismatch shipped: policy_parameters.hpp was
  # regenerated locally at 17:00 while PC2 still ran a 13:27 build.
  SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref"
  if [[ -n "$R_BIN_T" && -d "$SRC_DIR" ]]; then
    # robot build stamp is YYYY-MM-DD_HH:MM:SS in the ROBOT clock; compare via epoch
    _bin_epoch=$(date -d "$(echo "$R_BIN_T" | tr _ " ")" +%s 2>/dev/null || echo 0)
    _newer=$(find "$SRC_DIR" \( -name "*.cpp" -o -name "*.hpp" \) -newermt "@$_bin_epoch" 2>/dev/null)
    if [[ -n "$_newer" && "$_bin_epoch" != "0" ]]; then
      echo "               *** LOCAL C++ IS AHEAD OF THE ROBOT BUILD ***"
      echo "$_newer" | sed "s|$SRC_DIR/|                 |" | head -8
      echo "               These are modified locally but NOT in the robot binary"
      echo "               (built $R_BIN_T). Rebuild on PC2 or you are testing"
               echo "               code the robot does not run."
      BIN_STALE=1
    else
      echo "               local C++ sources: none newer than the robot build  OK"
    fi
  fi

if [[ "$SONIC_OK" -eq 1 && "$PLAN_OK" -eq 1 && "${HF_OK:-0}" -eq 1 ]]; then
  GATE="MATCHED"; echo "  => GATE MATCHED: you are testing exactly what the robot runs."
else
  GATE="MISMATCHED"
  echo "  => GATE MISMATCHED. This run does NOT certify the robot's current build."
  # FAIL CLOSED. A gate you can drive past is not a gate: the whole failure mode
  # it exists to prevent is someone eyeballing a clean sim run and concluding the
  # robot is good, when sim was driving different artifacts. Testing a candidate
  # pre-deploy is legitimate, but it must be a deliberate act, never the default.
  if [[ "${ALLOW_MISMATCH:-0}" != "1" ]]; then
    echo
    echo "  ABORTING: refusing to launch against artifacts the robot does not run."
    echo "  Whatever you saw in the viewer would NOT be evidence about the robot."
    echo
    echo "  If this is deliberate (validating a candidate before deploying it):"
    echo "      ALLOW_MISMATCH=1 $0 ${ORIG_ARGS[*]}"
    echo "  Otherwise sync the mismatched artifact(s) to the robot and re-run."
    exit 6
  fi
  echo "  ALLOW_MISMATCH=1 set -- proceeding as an explicit CANDIDATE test."
  echo "  Results describe the candidate ONLY. Do not record them as a deploy GO."
fi
echo

PLANNER_ONNX="$(planner_graph "$LOCAL_PLANNER_DIR" "$GRAPH")"
[[ -n "$PLANNER_ONNX" ]] || PLANNER_ONNX="$LOCAL_PLANNER_DIR/x2_planner_${GRAPH}.onnx"

# ---------------------------------------------------------------------------
# Auto-cleanup of a previous LOCAL sim, so stale ports never block a launch.
#
# *** THIS MUST NEVER BE ABLE TO TOUCH THE ROBOT. ***
# Killing a live sonic drops the robot -- it cannot hold a frame, so losing its
# control process means it collapses. Safety rests on three facts, asserted here
# rather than assumed:
#   1. run_x2_quest3_planner_stack.sh --cleanup-only resolves victims ONLY via
#      lsof/fuser on LOCAL tcp ports and kills LOCAL pids. It contains no ssh or
#      scp in any kill path (verified).
#   2. This launcher is SIM-ONLY: it never passes --pc2-host, so the pose wire
#      binds 127.0.0.1 and the robot cannot even subscribe.
#   3. The PC2 IP above is used for the gate ONLY, over read-only ssh
#      (md5sum / date / grep). Nothing in this script writes to or signals PC2.
# The one way local cleanup could escape this machine is a remote Docker daemon,
# so refuse to run if DOCKER_HOST points off-box.
# ---------------------------------------------------------------------------
if [[ -n "${DOCKER_HOST:-}" && "$DOCKER_HOST" != unix://* ]]; then
  echo "REFUSING to auto-clean: DOCKER_HOST=$DOCKER_HOST is not a local socket." >&2
  echo "Cleanup stops containers; against a remote daemon that could reach the robot." >&2
  exit 5
fi
echo "=== local sim cleanup (frees stale ports; robot is NOT touched) ==="
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --cleanup-only 2>&1 | sed 's/^/  /' || true
echo

# ---------------------------------------------------------------------------
# End-of-run summary. Printed on ANY exit so the models under test are always
# recorded next to the verdict -- a visual result with no model provenance is
# not evidence of anything.
# ---------------------------------------------------------------------------
summary() {
  echo
  echo "==================== SIM RUN SUMMARY ===================="
  echo "  pc2                : $PC2_IP"
  echo "  gate verdict       : $GATE"
  echo "  --- models used ----------------------------------------"
  printf '  sonic              : %s\n' "$(basename "$SONIC_ONNX")$AUTO_NOTE"
  printf '                       md5 %s  robot mtime %s\n' "${LOCAL_SONIC_MD5:0:12}" "$REMOTE_SONIC_T"
  printf '  planner graph      : %s (%s)%s\n' "$(basename "$PLANNER_ONNX")" "$GRAPH" "$PLANNER_NOTE"
  printf '                       dir %s\n' "$LOCAL_PLANNER_DIR"
  printf '                       md5 %s  robot mtime %s\n' \
      "$([[ -f "$PLANNER_ONNX" ]] && md5sum "$PLANNER_ONNX" | cut -c1-12 || echo n/a)" \
      "${PLAN_T[x2_planner_$GRAPH]:-n/a}"
  printf '  planner runtime    : handoff fix %s  robot mtime %s\n' \
      "$([[ "${HF:-0}" -gt 0 ]] && echo PRESENT || echo ABSENT)" "$HF_T"
  printf '  dances             : %s\n' "$DANCES"
  printf '  warmup qpos        : kplanner_idle_anchor_g1teleop_v3.pkl\n'
  printf '  sonic .pt          : none (ONNX-only, matches robot)\n'
  printf '  input mode         : %s\n' "$([[ "$VR_MODE" -eq 1 ]] && echo 'pad + Quest 3 VR (dual-source)' || echo 'pad only')"
  # Offline banner LAST, so it is the final thing on screen when the robot was
  # never consulted. Replaces the old start-of-run y/N keypress: a reminder
  # after the run is what actually prevents an uncertified result being quoted
  # as a certified one, since by then you have numbers in hand worth quoting.
  if [[ "${GATE_OFFLINE:-0}" == "1" ]]; then
    printf '\n  *** OFFLINE RUN -- robot was NOT reachable; NO identity check ***\n'
    printf '      These are CANDIDATE results. Do not quote them as certified\n'
    printf '      against the robot until re-run with PC2 online.\n'
  fi
  echo "  --------------------------------------------------------"
  if [[ "$GATE" != "MATCHED" ]]; then
    echo "  REMINDER: gate was $GATE -- results do NOT certify the robot's build."
  fi
  echo "========================================================="
}
trap summary EXIT

export KPLANNER_ONNX="$PLANNER_ONNX"
export KPLANNER_DANCES_DIR="$DANCES"
# Respect a caller-provided speed (was a hard export that silently clobbered
# any KPLANNER_FIXED_FWD_MPS the operator set -- "no matter what fwd mps I
# set, it walks the same", 2026-07-21).
export KPLANNER_FIXED_FWD_MPS="${KPLANNER_FIXED_FWD_MPS:-0.3}"
export PAD_LOCK_SPEED=1
export PAD_DEADMAN=left
# Pad clip banks: the regenerated demo bank (tools/build_demo_bank_from_upstream.sh
# -> x2_demo_bank.pkl + the x2m2 bakes in $DANCES). Keys are the upstream
# reference clip names; the button->bank mapping mirrors the robot ritual
# (Y easy dances, X combat, B medium dances, A gestures, TURN turn clips).
# Override any PAD_CLIP_KEYS* from the environment / KPLANNER_PROFILE.
export PAD_CLIP_PKL="${PAD_CLIP_PKL:-gear_sonic/data/motions/x2_demo_bank.pkl}"
export PAD_CLIP_KEYS="${PAD_CLIP_KEYS:-}"          # dance bank (L1+Y/A): empty by default -- no bundled dance clips
export PAD_CLIP_KEYS_X="${PAD_CLIP_KEYS_X:-}"      # combat bank (L1+X): empty by default
export PAD_CLIP_KEYS_M="${PAD_CLIP_KEYS_M:-}"      # medium-dance bank (L1+B): empty by default
export PAD_CLIP_KEYS_G="${PAD_CLIP_KEYS_G:-right_wave_001,right_kiss_001,right_five_001,right_shake_001,turn_wave_right_001,turn_wave_left_001}"   # gesture bank (L1+A): X2 MC stock gestures
export PAD_CLIP_KEYS_TURN="${PAD_CLIP_KEYS_TURN:-}" # 4-way turn bank (right stick): empty by default

# ── SIM_TUNING_YAML: validate the full stack with the ROBOT's tuning ──────
# Without this the stack runs deploy_x2.sh on its BUILT-IN DEFAULTS -- every
# max_target_dev DISABLED, kp/kd_scale_waist_pr 1.0 (robot ships 2.00/5.51),
# output LPF off. That is a materially softer, unclamped machine than the one
# on the gantry, so "fine in sim" was never evidence about the robot.
# Discovered 2026-08-23 chasing v12_6500's waist saturation: quiet in the
# full-stack sim, pinned at the clamp on hardware.
#
#   SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/bigrun.yaml \
#     ./gear_sonic/scripts/sim_onnx_planner.sh --pc2-host <ip>
#
# DEFAULT = the shipped preset (trained_gains_s0_inc_waistmc.yaml, what the
# robot ritual runs when robot_env.env is the shipped template). Parity runs
# that must stay on the deploy's built-in defaults (mirroring
# eval_x2_mujoco.py, which is what makes the C++<->Python check meaningful)
# opt out explicitly with SIM_TUNING_YAML=none (or an empty value).
# What the OPERATOR set by hand, before the yaml appends its flags. The arg
# parity check below adopts the ritual's CLI overrides silently, but must never
# overwrite a value the operator typed deliberately -- that it flags instead.
_SDA_USER="${SIM_DEPLOY_ARGS:-}"
if [[ -z "${SIM_TUNING_YAML+x}" ]]; then
  SIM_TUNING_YAML="$SHIPPED_TUNING"
  echo "[tuning] no SIM_TUNING_YAML -> shipped default $SIM_TUNING_YAML (SIM_TUNING_YAML=none for deploy defaults)"
elif [[ "${SIM_TUNING_YAML}" == "none" ]]; then
  SIM_TUNING_YAML=""
fi
if [[ -n "${SIM_TUNING_YAML:-}" ]]; then
  if [[ ! -f "$SIM_TUNING_YAML" ]]; then
    echo "ERROR: SIM_TUNING_YAML=$SIM_TUNING_YAML not found" >&2; exit 1
  fi
  _flags="$(python3 gear_sonic_deploy/scripts/tuning_yaml_to_sim_flags.py "$SIM_TUNING_YAML")" || {
    echo "ERROR: could not convert $SIM_TUNING_YAML to sim flags" >&2; exit 1; }
  export SIM_DEPLOY_ARGS="${SIM_DEPLOY_ARGS:-} $_flags"
  echo "[tuning] $SIM_TUNING_YAML (md5 $(md5sum "$SIM_TUNING_YAML" | cut -c1-8))"
  echo "[tuning]   -> $_flags"
else
  echo "[tuning] NONE -- sim runs deploy DEFAULTS (clamps disabled, waist kp/kd x1.0)."
  echo "[tuning]   This is NOT the robot's config. Set SIM_TUNING_YAML=<preset.yaml>"
  echo "[tuning]   to validate against what the robot actually runs."
fi

# ---------------------------------------------------------------------------
# RITUAL ARG PARITY
#
# The identity gate above hashes MODELS and the BINARY. It never looked at how
# that binary gets INVOKED -- and every knob that decides how the robot feels
# (gains, dev clamps, output LPF, head/wrist bypass) lives in the argv, not in
# the weights. So the gate could report "you are testing exactly what the robot
# runs" while the sim ran a different tuning file, a different head clamp, or
# no tuning at all. That is exactly what happened on 2026-08-24: a run with
# SIM_TUNING_YAML=bigrun.yaml passed every line of the gate while the robot's
# ritual was resolving frozen_g1.yaml. Harmless that day only because the two
# files happened to be functionally identical -- one day earlier, frozen_g1
# carried a 3 Hz waist LPF and a 4x kd cut that bigrun did not.
#
# So: compare the ritual's EFFECTIVE behavioural argv against the sim's.
#   * tuning is compared by CONTENT, not md5 -- comments and key order must not
#     raise a false alarm, and two differently-named files with the same
#     settings ARE the same machine.
#   * the ritual's own script is compared against the repo mirror first; if the
#     robot's copy has drifted, everything derived from the mirror is fiction.
# Plumbing args (paths, zmq ports, --no-docker, --log-dir) are deliberately NOT
# compared: they must differ between a laptop sim and the robot.
# ---------------------------------------------------------------------------
_fp() {  # effective-content fingerprint of a tuning yaml: no comments, no order
  [[ -f "$1" ]] || { echo MISSING; return; }
  grep -vE '^[[:space:]]*(#|$)' "$1" | sed 's/#.*//; s/[[:space:]]*$//' \
    | grep -v '^$' | sort | md5sum | cut -d' ' -f1
}
_last_flag() {  # last value of flag $1 in the arg string $2 ("" if absent)
  local want="$1"; shift
  local out="" prev=""
  for tok in $*; do
    [[ "$prev" == "$want" ]] && out="$tok"
    prev="$tok"
  done
  echo "$out"
}

if [[ -n "$REMOTE" ]]; then
  echo "=== RITUAL ARG PARITY ==="
  ARG_OK=1; ARG_FIXED=0
  # ALLOW_MISMATCH=1 means "I am deliberately testing something the robot is
  # not running". Silently substituting the robot's values into THAT run defeats
  # the point twice over: the operator does not get the machine they asked for,
  # and the divergence they chose is hidden from the log. So adoption is for
  # certification runs only -- under ALLOW_MISMATCH the sim keeps what it was
  # given and the gate merely reports.
  _ADOPT=1; [[ "${ALLOW_MISMATCH:-0}" == "1" ]] && _ADOPT=0

  # --- the ritual script itself: robot vs repo mirror ---
  R_RITUAL_MD5=$(r_field "^ritual$"); R_RITUAL_T=$(r_time "^ritual$")
  if [[ "$R_RITUAL_MD5" == "MISSING" || -z "$R_RITUAL_MD5" ]]; then
    echo "  ritual    : *** NOT FOUND on robot *** ($R_RITUAL)"; ARG_OK=0
  elif [[ ! -f "$L_RITUAL" ]]; then
    echo "  ritual    : robot ${R_RITUAL_MD5:0:12} ($R_RITUAL_T); no repo mirror at $L_RITUAL"
  elif [[ "$(md5sum "$L_RITUAL" | cut -d' ' -f1)" == "$R_RITUAL_MD5" ]]; then
    echo "  ritual    : MATCH ${R_RITUAL_MD5:0:12} (robot == $L_RITUAL)"
  else
    echo "  ritual    : *** DRIFT *** robot=${R_RITUAL_MD5:0:12} ($R_RITUAL_T)"
    echo "              repo=$(md5sum "$L_RITUAL" | cut -c1-12) $L_RITUAL"
    # Announcing drift without showing it just moves the work to the operator at
    # the worst possible moment. Fetch the robot's copy (read-only cat -- this
    # NEVER writes to or signals PC2) and print the diff, so what actually
    # differs is on screen before launch. The robot's file is the source of
    # truth here: the repo mirror is a mirror, so drift is resolved by pulling
    # the robot's copy INTO the repo, never by pushing the mirror out.
    _RRIT="${TMPDIR:-/tmp}/x2_ritual_robot_${R_RITUAL_MD5:0:12}.sh"
    if timeout 20 ssh -o ConnectTimeout=8 "run@$PC2_IP" "cat $R_RITUAL" > "$_RRIT" 2>/dev/null \
       && [[ -s "$_RRIT" ]]; then
      # Behavioural lines are the ones that change how the binary is invoked.
      # A comment-only drift is worth knowing about but is not a different machine.
      _BEHAV=$(diff -u "$L_RITUAL" "$_RRIT" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' \
               | grep -E '(^[+-][[:space:]]*--|[A-Z_]+=)' | grep -vE '^[+-][[:space:]]*#' | wc -l)
      _TOTAL=$(diff -u "$L_RITUAL" "$_RRIT" | grep -cE '^[+-]' )
      if [[ "$_BEHAV" -gt 0 ]]; then
        echo "              $_BEHAV BEHAVIOURAL line(s) differ -- the robot invokes the"
        echo "              binary differently than the mirror says. Everything"
        echo "              compared below is read from the mirror, so treat it as"
        echo "              unverified until this is reconciled."
        ARG_OK=0
      else
        echo "              No behavioural lines differ (comments/whitespace only),"
        echo "              so the argv comparisons below still hold. The mirror is"
        echo "              stale and should be refreshed, but this is the same machine."
        ARG_FIXED=$((ARG_FIXED + 1))
      fi
      echo "              ---- diff  repo(-)  vs  robot(+) ----"
      diff -u "$L_RITUAL" "$_RRIT" | tail -n +3 | head -60 | sed 's/^/              /'
      [[ "$_TOTAL" -gt 60 ]] && echo "              ... truncated; full copy: $_RRIT"
      echo "              -------------------------------------"
      echo "              Robot's copy saved: $_RRIT"
      echo "              Adopt it into the repo (the robot is the source of truth):"
      echo "                  cp $_RRIT $L_RITUAL"
    else
      ARG_OK=0
      echo "              (could not fetch the robot's copy to diff it)"
      echo "              The robot's launcher is not the one in the repo, and the"
      echo "              arguments compared below are read from the REPO copy."
    fi
  fi

  # --- tuning config: compared by effective CONTENT ---
  R_TUNING=$(r_field "^rtuning$"); R_TUNEFP=$(r_field "^rtunefp$")
  L_TUNEFP=$(_fp "${SIM_TUNING_YAML:-}")
  if [[ -z "${SIM_TUNING_YAML:-}" ]]; then
    echo "  tuning    : *** SIM HAS NONE *** robot runs $(basename "$R_TUNING")"
    echo "              Deploy falls back to BUILT-IN defaults: every"
    echo "              max_target_dev disabled, waist kp/kd x1.0, LPF off."
    echo "              That is a softer, unclamped machine than the gantry."
    echo "              Fix:  SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/$(basename "$R_TUNING")"
    ARG_OK=0
  elif [[ "$R_TUNEFP" == "MISSING" ]]; then
    echo "  tuning    : *** robot's $R_TUNING is unreadable ***"; ARG_OK=0
  elif [[ "$L_TUNEFP" == "$R_TUNEFP" ]]; then
    if [[ "$(basename "$SIM_TUNING_YAML")" == "$(basename "$R_TUNING")" ]]; then
      echo "  tuning    : MATCH $(basename "$R_TUNING") (content ${R_TUNEFP:0:12})"
    else
      echo "  tuning    : MATCH by content ${R_TUNEFP:0:12}"
      echo "              sim=$(basename "$SIM_TUNING_YAML")  robot=$(basename "$R_TUNING")  -- different"
      echo "              filenames, identical settings. Same machine."
    fi
  else
    echo "  tuning    : *** MISMATCH ***"
    echo "              sim  =$SIM_TUNING_YAML  (${L_TUNEFP:0:12})"
    echo "              robot=$R_TUNING  (${R_TUNEFP:0:12})"
    echo "              These set different gains/clamps/LPF. Nothing you see in"
    echo "              the viewer is evidence about the robot."
    ARG_OK=0
  fi

  # --- head/wrist bypass: inherit the ritual's, don't hope the defaults agree ---
  # These reach the C++ binary through the stack, not through SIM_DEPLOY_ARGS.
  # Today the stack's defaults (ik/ref) happen to equal the ritual's -- but
  # "happens to equal" is the whole bug class here, so pass them explicitly.
  # wrist-bypass is hard-coded in the ritual, so read it from the mirror; the
  # robot-vs-mirror drift check above is what makes that safe.
  R_HB=$(r_field "^rheadbypass$"); R_HD=$(r_field "^rheaddev$")
  R_WB=$(grep -oE '\-\-wrist-bypass +[a-z]+' "$L_RITUAL" 2>/dev/null | head -1 | awk '{print $2}')
  R_WB="${R_WB:-ik}"
  S_STACK=gear_sonic/scripts/run_x2_quest3_planner_stack.sh
  L_WB=$(grep -m1 '^WRIST_BYPASS=' "$S_STACK" | cut -d'"' -f2)
  L_HB=$(grep -m1 '^HEAD_BYPASS='  "$S_STACK" | cut -d'"' -f2)
  if [[ "$_ADOPT" -eq 1 ]]; then
    RITUAL_BYPASS_FLAGS=(--wrist-bypass "$R_WB" --head-bypass "$R_HB")
  else
    # Candidate run: leave the stack on its own defaults rather than forcing the
    # robot's. Reported below either way.
    RITUAL_BYPASS_FLAGS=()
  fi
  if [[ "$L_WB" == "$R_WB" && "$L_HB" == "$R_HB" ]]; then
    echo "  bypass    : MATCH wrist=$R_WB head=$R_HB (passed explicitly)"
  elif [[ "$_ADOPT" -eq 0 ]]; then
    echo "  bypass    : DIVERGES -- KEEPING YOURS (sim wrist=$L_WB head=$L_HB;"
    echo "              robot wrist=$R_WB head=$R_HB). ALLOW_MISMATCH=1."
    ARG_OK=0
  else
    echo "  bypass    : !! DIVERGED -- CORRECTED"
    [[ "$L_WB" == "$R_WB" ]] || echo "              wrist: sim default $L_WB -> ritual $R_WB"
    [[ "$L_HB" == "$R_HB" ]] || echo "              head : sim default $L_HB -> ritual $R_HB"
    ARG_FIXED=$((ARG_FIXED + 1))
  fi

  # --- the ritual's CLI overrides: ADOPT them, don't just complain ---
  # deploy_x2.sh applies --tuning-config first and explicit flags after, so on the
  # robot the ritual's CLI value WINS over the yaml. frozen_g1.yaml says
  # max_target_dev_head 0.5; the ritual passes 0.40. Loading the yaml alone
  # therefore does NOT reproduce the robot -- which is precisely the silent
  # divergence this section exists to kill. So take the robot's value.
  # Exception: a value the operator typed by hand is never overwritten; that is a
  # deliberate act and gets flagged instead.
  _adopt() {  # $1 flag, $2 ritual value, $3 label
    local mine; mine=$(_last_flag "$1" "$_SDA_USER")
    if [[ -n "$mine" ]]; then
      if awk -v a="$mine" -v b="$2" 'BEGIN{exit !(a==b || (a-b<1e-9 && b-a<1e-9))}'; then
        echo "  ${3}: MATCH $2 (you set it explicitly)"
      else
        echo "  ${3}: *** MISMATCH *** you set $1 $mine, ritual passes $2"
        echo "              SIM_DEPLOY_ARGS wins here, so the sim would run YOUR"
        echo "              value, not the robot's. Drop it to inherit the ritual's."
        ARG_OK=0
      fi
      return
    fi
    local from_yaml; from_yaml=$(_last_flag "$1" "${SIM_DEPLOY_ARGS:-}")
    if [[ "$_ADOPT" -eq 0 ]]; then
      if [[ -n "$from_yaml" ]] && ! awk -v a="$from_yaml" -v b="$2" 'BEGIN{exit !(a==b || (a-b<1e-9 && b-a<1e-9))}'; then
        echo "  ${3}: DIVERGES -- KEEPING YOURS ($1 $from_yaml; robot passes $2)"
        echo "              ALLOW_MISMATCH=1, so the sim is NOT corrected to the"
        echo "              robot's value. This run is your candidate, not the robot."
        ARG_OK=0
      else
        echo "  ${3}: MATCH ${from_yaml:-$2}"
      fi
      return
    fi
    export SIM_DEPLOY_ARGS="${SIM_DEPLOY_ARGS:-} $1 $2"
    if [[ -n "$from_yaml" ]] && ! awk -v a="$from_yaml" -v b="$2" 'BEGIN{exit !(a==b || (a-b<1e-9 && b-a<1e-9))}'; then
      echo "  ${3}: !! DIVERGED -- CORRECTED"
      echo "              sim would have run $1 $from_yaml (from"
      echo "              $(basename "${SIM_TUNING_YAML:-yaml}")); the ritual passes $2 on the"
      echo "              CLI, which overrides the yaml on the robot. Using $2."
      ARG_FIXED=$((ARG_FIXED + 1))
    elif [[ -z "$from_yaml" ]]; then
      echo "  ${3}: !! DIVERGED -- CORRECTED (sim had no $1; ritual passes $2)"
      ARG_FIXED=$((ARG_FIXED + 1))
    else
      echo "  ${3}: MATCH $2"
    fi
  }
  _adopt --max-target-dev-head "$R_HD" "dev-head  "

  # --- ENV PARITY: every variable the robot's env file resolves ---------------
  # The gate used to compare models, the binary and the tuning yaml, and stop.
  # It never compared the ENVIRONMENT -- so a sim sourcing a different profile
  # ran different kplanner speeds and, worse, a different anchor-orientation
  # convention, and reported a clean sheet. Found 2026-08-25: sim ran
  # bigrun_teleop_v1.env (turn 0.55, fwd 0.35, X2_ANCHOR_ORI_MODE unset -> the
  # C++ default "b") against a robot on 0.65 / 0.30 / "heading". Every value
  # the robot resolves is now compared by name, so a variable added to the
  # robot later is checked automatically instead of silently escaping.
  R_ENVFILE=$(r_field "^renvfile$"); R_ENVFP=$(r_time "^renvfile$")
  echo "  env file  : robot $R_ENVFILE (${R_ENVFP:0:12})"
  # 2026-09-02: a REAL kplanner_profile.env (the pre-08-25 name, meant to be
  # a symlink) shadowed robot_env.env for two days -- S1-10k ran while the
  # env file said 19000 / 36000. The ritual now refuses to launch on it;
  # surface it here too so a push is never "verified" against a dead file.
  if [[ -n "$(r_field "^rstaleenv$")" ]]; then
    echo "  *** STALE ENV ON ROBOT: kplanner_profile.env is a real file, not a symlink ***"
    echo "      The deploy ritual will REFUSE to launch. Retire it: mv kplanner_profile.env kplanner_profile.env.stale_<date>"
    ARG_OK=0
  fi
  while read -r _v _rval; do
    [[ -n "$_v" ]] || continue
    case "$_v" in
      # Compared by CONTENT above -- the robot's absolute path can never equal
      # the sim's repo-relative one, and the fingerprint check is stronger.
      X2_RITUAL_TUNING) continue ;;
    esac
    _mine="${!_v-}"
    # X2_ANCHOR_ORI_MODE is the dangerous one: unset is not "no opinion", it is
    # "b" (tokenizer_obs.cpp -- e != nullptr && e == "heading"). Resolve it the
    # way the binary does, or an unset sim silently reads as agreeing.
    if [[ "$_v" == "X2_ANCHOR_ORI_MODE" && -z "$_mine" ]]; then
      _mine="b (UNSET -> C++ default)"
    fi
    if [[ -z "$_mine" ]]; then
      echo "  env       : *** $_v UNSET in sim *** robot=$_rval"
      ARG_OK=0
    elif [[ "$_mine" == "$_rval" ]]; then
      printf "  env       : MATCH %-32s %s\n" "$_v" "$_rval"
    else
      echo "  env       : *** MISMATCH *** $_v"
      echo "              sim  =$_mine"
      echo "              robot=$_rval"
      ARG_OK=0
    fi
  done < <(echo "$REMOTE" | awk '$1=="renv"{print $2, $3}')

  # --- the REVERSE direction: variables the SIM sets that the robot does NOT ---
  # Comparing only the robot's variables leaves the obvious hole: anything you
  # export that the robot has never heard of sails through. That is not a
  # harmless extra -- it is either a knob the robot does not have (so the run
  # is not the robot) or a no-op that reads like configuration and buys false
  # confidence. X2_PLANT is the standing example: it selects a plant for the
  # PYTHON/MuJoCo tooling, but deploy_x2.sh never forwards it and the binary's
  # plant is compile-time (policy_parameters.hpp), so passing it to a full-stack
  # run changes nothing while looking like it changes everything.
  _ROBOT_VARS=" $(echo "$REMOTE" | awk '$1=="renv"{printf "%s ", $2}')"
  for _v in $_ENV_AT_ENTRY; do
    # The launcher's OWN inputs -- these select what to test, they are not
    # robot settings, and each is verified by a stronger check above.
    case "$_v" in
      KPLANNER_PROFILE|KPLANNER_PYTHON|KPLANNER_WARMUP_QPOS) continue ;;
    esac
    [[ "$_ROBOT_VARS" == *" $_v "* ]] && continue
    echo "  env       : *** EXTRA *** $_v=${!_v}"
    echo "              The robot's env file does not define this. Either it is"
    echo "              a setting the robot does not have (so this run is not"
    echo "              the robot), or it is inert here and misleading."
    if [[ "$_v" == "X2_PLANT" ]]; then
      echo "              X2_PLANT specifically: NOT forwarded by deploy_x2.sh and"
      echo "              NOT read by the binary (plant is compile-time). It steers"
      echo "              only Python/MuJoCo tooling -- it does nothing here."
    fi
    ARG_OK=0
  done

  # --- the IGNITION script: drift + the kplanner daemon's argv -----------------
  R_IGN_MD5=$(r_field "^ignition$"); R_IGN_T=$(r_time "^ignition$")
  if [[ "$R_IGN_MD5" == "MISSING" || -z "$R_IGN_MD5" ]]; then
    echo "  ignition  : *** NOT FOUND on robot *** ($R_IGNITION)"; ARG_OK=0
  elif [[ ! -f "$L_IGNITION" ]]; then
    echo "  ignition  : robot ${R_IGN_MD5:0:12}; no repo mirror at $L_IGNITION"; ARG_OK=0
  elif [[ "$(md5sum "$L_IGNITION" | cut -d' ' -f1)" == "$R_IGN_MD5" ]]; then
    echo "  ignition  : MATCH ${R_IGN_MD5:0:12} (robot == $L_IGNITION)"
  else
    echo "  ignition  : *** DRIFT *** robot=${R_IGN_MD5:0:12} ($R_IGN_T)"
    echo "              repo=$(md5sum "$L_IGNITION" | cut -c1-12) $L_IGNITION"
    _RIGN="${TMPDIR:-/tmp}/x2_ignition_robot_${R_IGN_MD5:0:12}.sh"
    if timeout 20 ssh -o ConnectTimeout=8 "run@$PC2_IP" "cat $R_IGNITION" > "$_RIGN" 2>/dev/null \
       && [[ -s "$_RIGN" ]]; then
      echo "              ---- diff  repo(-)  vs  robot(+) ----"
      diff -u "$L_IGNITION" "$_RIGN" | tail -n +3 | head -40 | sed 's/^/              /'
      echo "              Robot's copy: $_RIGN"
    fi
    ARG_OK=0
  fi

  # The daemon's behavioural flags. Read from the MIRROR (the drift check above is
  # what makes that safe) and compared against the sim's effective values. These
  # decide how the planner behaves; the graph md5 says nothing about them.
  _ign_flag() { sed -n '/pc2_kplanner_onnx.py/,/tee -a/p' "$L_IGNITION" 2>/dev/null \
                | grep -oE -- "--$1 [^ \\\\]+" | head -1 | awk '{print $2}'; }
  R_PMODE=$(_ign_flag planner-mode)
  R_WARMUP=$(basename "$(_ign_flag warmup-qpos)" 2>/dev/null)
  R_YAWRE=$(_ign_flag playing-yaw-resync-dps)
  L_PMODE="${MODE:-slow_walk}"
  L_WARMUP=$(basename gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl)
  _cmp_ign() {  # $1 label  $2 sim  $3 robot
    if [[ -z "$3" ]]; then printf "  ignition  : (%s not found in the mirror -- not compared)\n" "$1"; return; fi
    if [[ "$2" == "$3" ]]; then printf "  ignition  : MATCH %-24s %s\n" "$1" "$3"
    else
      echo "  ignition  : *** MISMATCH *** $1"
      echo "              sim  =$2"
      echo "              robot=$3"
      ARG_OK=0
    fi
  }
  _cmp_ign planner-mode "$L_PMODE"  "$R_PMODE"
  _cmp_ign warmup-anchor "$L_WARMUP" "$R_WARMUP"
  # playing-yaw-resync-dps has no sim-side flag in this launcher; report the
  # robot's so a divergence is at least visible rather than silently absent.
  [[ -n "$R_YAWRE" ]] && echo "  ignition  : robot --playing-yaw-resync-dps $R_YAWRE (sim launcher passes none)"

  # --- informational: covered elsewhere or resolved inside the stack ---
  echo "  model     : ritual -> $(r_field '^rmodel$')  (md5 already gated above)"
  echo "  watchdog  : ritual always forwards --disable-pose-ref-watchdog; the"
  echo "              stack resolves its own (prints 'pose-ref watchdog:' at launch)"

  if [[ "$ARG_OK" -eq 1 && "$ARG_FIXED" -eq 0 ]]; then
    echo "  => ARG PARITY OK: the sim invokes the binary the way the ritual does."
  elif [[ "$ARG_OK" -eq 1 ]]; then
    # The run IS faithful now -- but say plainly that it only became faithful
    # because the gate corrected it. Silently fixing this would hide the very
    # drift the check exists to surface, and the same divergence is still live
    # for anyone launching deploy_x2.sh by hand.
    echo "  => ARG PARITY CORRECTED ($ARG_FIXED arg(s)): the sim's own defaults did"
    echo "     NOT match the ritual. They have been overridden with the robot's"
    echo "     values, so THIS run is faithful -- but the divergence above is real"
    echo "     and still applies to any hand-rolled deploy_x2.sh invocation."
  else
    echo "  => ARG PARITY FAILED. Same models, DIFFERENT machine."
    if [[ "${ALLOW_MISMATCH:-0}" != "1" ]]; then
      echo
      echo "  ABORTING: the weights match but the invocation does not, so a clean"
      echo "  run here would not be evidence about the robot -- which is the exact"
      echo "  failure the gate exists to prevent."
      echo "  Deliberate?   ALLOW_MISMATCH=1 $0 ${ORIG_ARGS[*]}"
      exit 7
    fi
    echo "  ALLOW_MISMATCH=1 set -- proceeding. Results describe THIS argv only."
  fi
  echo
fi

# ── SIM_PROFILE / SIM_INIT_POSE: reproduce the MC->sonic HANDOFF ─────────
# The stack hardcodes --sim-profile parity, which RSIs the robot ONTO the
# reference -- so sim starts from a PERFECT pose and the handoff error that
# exists on hardware is exactly zero. That is why sim has never reproduced the
# instant failure seen when sonic launches on the robot (2026-08-23).
#
# On the robot: MC holds its OWN stand pose (knees ~0.48 bent), MC stops, and
# sonic inherits that pose while the reference asks for idle_stand (knees
# 0.096, near-straight) -- a ~0.39 rad knee error on tick one.
#
# To recreate that:
#   SIM_PROFILE=handoff SIM_DEPLOY_ARGS="--sim-init-pose gantry-hang" ...
# gantry_hang IS the captured MC-stand pose (see sim_init_poses.yaml), so the
# robot spawns where MC would have left it, not where the policy wants it.
SIM_PROFILE_FLAG=()
if [[ -n "${SIM_PROFILE:-}" ]]; then
  SIM_PROFILE_FLAG=(--sim-profile "$SIM_PROFILE")
  echo "[sim-profile] $SIM_PROFILE (overriding the stack default 'parity')"
  [[ "$SIM_PROFILE" != "parity" ]] &&     echo "[sim-profile]   NOTE: non-parity profile -- this is NOT the C++<->Python parity check."
fi

MODE_FLAG=()
[[ "$GRAPH" == "template" ]] && MODE_FLAG=(--kplanner-planner-mode "${MODE:-slow_walk}")
SONIC_CKPT_FLAG=(--no-sonic-checkpoint)
[[ -n "${SONIC_CKPT:-}" ]] && SONIC_CKPT_FLAG=(--sonic-checkpoint "$SONIC_CKPT")

# --vr swaps --pad-only for --pad-and-vr: quest3_manager_x2 + pad bridge both
# feed the kplanner's bound planner_cmd SUB (dual-source; see stack script).
# VR device backend: the stack inherits INPUT_SOURCE from the environment —
# INPUT_SOURCE=pico ./sim_onnx_planner.sh --vr   drives the manager from the
# Pico headset + trackers via the XRoboToolkit PC Service (no WebXR app).
INPUT_FLAG=(--pad-only)
[[ "$VR_MODE" -eq 1 ]] && INPUT_FLAG=(--pad-and-vr)
# --vr-only: no input flag at all = the stack's native manager-only mode
# (quest3_manager_x2 is the sole planner_cmd source; no pygame/pad needed).
[[ "$VR_MODE" -eq 2 ]] && INPUT_FLAG=()

# NOT exec'd: we must survive the stack to print the summary trap.
# --no-record: the stack defaults WITH_RECORD=1 (2026-07-29), but this
# launcher is teleop-only validation by design (robot runs no recorder).
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
  --duration 0 --no-record "${INPUT_FLAG[@]}" "${MODE_FLAG[@]}" "${SONIC_CKPT_FLAG[@]}" \
  "${SIM_PROFILE_FLAG[@]}" "${RITUAL_BYPASS_FLAGS[@]}" \
  --kplanner-warmup-qpos gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl \
  --model "$SONIC_ONNX" \
  --kplanner-python "$KPY" || true
