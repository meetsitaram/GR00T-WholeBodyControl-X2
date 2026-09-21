#!/usr/bin/env bash
# One-shot bring-up: build the X2 container in REAL-ROBOT mode AND run the
# deploy node in --dry-run, with operator-gated safety prompts the whole way.
# This is the single command the operator on the gantry should need to take
# a freshly-flashed laptop + robot pair from cold to "policy is ticking,
# observation tensor is well-formed, no joint commands are leaving the box."
#
# The script is intentionally chatty -- it prints what it's about to do and
# asks before each step that touches the network, the docker daemon, or the
# robot. Add `--yes` to skip all the operator prompts once you trust the flow.
#
# Pre-conditions on the host (NOT auto-configured -- they require root and we
# want the operator to make the choice explicitly):
#   1. SDK ethernet cable in one of PC2/PC3's rear RJ45 dev ports (PC1 = the
#      compute box that hosts the Environment Manager (EM) HTTP API at
#      <PC1_IP>, PC2 = the dev unit at <PC2_IP> we DDS-discover the robot
#      through, PC3 = the redundant dev unit when fitted). Export
#      PC2_HOST=<PC2_IP> PC1_HOST=<PC1_IP> (x2_discover_network.sh writes
#      ~/.x2/env.wired with them); X2_SDK_NIC / X2_SDK_HOST_IP name the
#      laptop's SDK NIC and its static address (default enp10s0, unchecked
#      address unless set).
#   2. Host NIC configured static on the SDK subnet, e.g.:
#        sudo ip addr add <LAPTOP_IP>/24 dev $X2_SDK_NIC
#        sudo ip link set $X2_SDK_NIC up
#   3. ping $PC2_HOST succeeds from the host.
#
# Default behavior:
#   1. Host pre-flight: NIC, ping, PC1 Environment Manager (EM) HTTP API,
#      stale container check, --model file resolution.
#   2. docker compose build (no-op if cached).
#   3. Inside the container, layer docker-compose.real.yml (DDS unquarantined,
#      ROS_DOMAIN_ID=0, FastDDS multicast discovery on the SDK NIC) and run:
#        ./gear_sonic_deploy/deploy_x2.sh --dry-run --no-stop-mc <your extra flags>
#      We auto-inject --no-stop-mc so the gantry-held robot stays actively
#      PD-held by Motion Control (MC) for the smoke test (forgetting that
#      flag and watching the robot drop into the gantry straps is exactly
#      the surprise we want to engineer out of the bring-up tool). Pass
#      --stop-mc on the wrapper if you want the legacy "stop MC, run policy
#      with zero stiffness/damping, robot is fully passive" behavior.
#      deploy_x2.sh still owns its own ping + DDS pre-flight and the final
#      "ARE YOU SURE" launch gate.
#
# Required:
#   --model PATH  ONNX checkpoint to load (container-side path). We don't
#                 default this on purpose -- silently picking a stale or
#                 partial checkpoint behind the operator's back is exactly
#                 the bug that's painful to debug at the gantry. Even in
#                 --dry-run the C++ deploy loads the ONNX session at startup
#                 (--dry-run only zeroes the stiffness/damping in the
#                 published joint commands), so the model is non-optional.
#
#                 Two container bind mounts are available for the path:
#                   /workspace/sonic/        -> repo root (this repo)
#                   /workspace/checkpoints/  -> $X2_CHECKPOINTS_DIR (default gear_sonic_deploy/models)
#                 Stage long-lived bring-up checkpoints under
#                 gear_sonic_deploy/models/ (git-ignored via the top-level
#                 `models/` rule) for stable, short paths.
#
# Examples:
#   # Bring-up dry run with the in-repo checkpoint (model is required):
#   ./get_x2_sonic_ready.sh \
#     --model /workspace/sonic/gear_sonic_deploy/models/x2_sonic_16k.onnx
#
#   # Same, but auto-flip from WAIT to CONTROL after 5s instead of waiting
#   # for stdin "go" (fine in --dry-run; do NOT use for first powered run):
#   ./get_x2_sonic_ready.sh \
#     --model /workspace/sonic/gear_sonic_deploy/models/x2_sonic_16k.onnx \
#     --autostart-after 5
#
#   # Use a checkpoint from $X2_CHECKPOINTS_DIR without staging it in-repo first:
#   ./get_x2_sonic_ready.sh \
#     --model /workspace/checkpoints/<run>/exported/model_step_016000_g1.onnx
#
#   # Skip all the operator prompts (you've done this many times):
#   ./get_x2_sonic_ready.sh --yes --model ...
#
#   # Drop the safe default and let deploy_x2.sh stop MC (robot goes passive
#   # on the gantry; useful when you want to feel/observe gravity-only
#   # behavior under a zero-torque policy):
#   ./get_x2_sonic_ready.sh --stop-mc --model ...
#
#   # Just drop me into a shell -- I'll drive things by hand:
#   ./get_x2_sonic_ready.sh --shell
#
#   # One-shot command (skip the deploy entirely):
#   ./get_x2_sonic_ready.sh -- ros2 topic list
#   ./get_x2_sonic_ready.sh -- python3 gear_sonic_deploy/scripts/x2_preflight.py
set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# Style + helpers
# ----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ASSUME_YES=false

confirm() {
    # confirm "Question?" [default Y|N]
    local prompt="$1"
    local default="${2:-Y}"
    local hint
    if [[ "$default" == "Y" ]]; then hint="[Y/n]"; else hint="[y/N]"; fi
    if $ASSUME_YES; then
        echo -e "${CYAN}? $prompt $hint${NC} -- ${GREEN}assuming Y (--yes)${NC}"
        return 0
    fi
    local answer
    read -r -p "$(echo -e "${CYAN}? $prompt $hint${NC} ")" answer
    answer="${answer:-$default}"
    case "$answer" in
        Y|y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

abort() { echo -e "${RED}Aborted.${NC}"; exit 1; }

# ----------------------------------------------------------------------------
# Arg parsing -- consume our own flags, pass the rest to deploy_x2.sh
# ----------------------------------------------------------------------------
RUN_MODE="dryrun"   # dryrun | shell | oneshot
ONESHOT_CMD=()
PASSTHROUGH=()
# Wrapper default: in --dry-run we DO NOT stop MC, so the gantry-held robot
# stays actively PD-held throughout the smoke test. Forgetting --no-stop-mc
# is too easy and the failure mode (MC released, robot goes passive, gravity
# pulls it into the gantry straps) is too startling for a bring-up tool.
# Operator opts back into the legacy "stop MC then run" path with --stop-mc.
INJECT_NO_STOP_MC=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)
            ASSUME_YES=true; shift
            ;;
        --shell)
            RUN_MODE="shell"; shift
            ;;
        --stop-mc)
            # Operator explicitly wants the deploy to POST stop_app to EM
            # before launching the policy. In --dry-run that means MC
            # releases its PD-hold and our zero-stiffness/damping commands
            # take over -> robot ends up fully passive on the gantry. Use
            # this when you want to feel/observe gravity-only behavior.
            INJECT_NO_STOP_MC=false; shift
            ;;
        --)
            RUN_MODE="oneshot"; shift; ONESHOT_CMD=("$@"); break
            ;;
        -h|--help)
            sed -n '2,55p' "$0"; exit 0
            ;;
        *)
            PASSTHROUGH+=("$1"); shift
            ;;
    esac
done

COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.real.yml)
# No baked-in robot addresses. PC2_HOST is required; PC1_HOST is optional
# (EM HTTP liveness probe only). x2_discover_network.sh writes ~/.x2/env.wired.
NIC="${X2_SDK_NIC:-enp10s0}"
NIC_EXPECT_IP="${X2_SDK_HOST_IP:-}"        # e.g. <LAPTOP_IP>/24; empty = don't check the value
ROBOT="${PC2_HOST:-}"
if [[ -z "$ROBOT" ]]; then
    echo -e "${RED}error:${NC} PC2_HOST is not set. export PC2_HOST=<PC2_IP> (the robot dev unit on the SDK ethernet)." >&2
    exit 2
fi
PC1_EM="${X2_MC_EM_URL:-${PC1_HOST:+http://${PC1_HOST}:${PC1_EM_PORT:-50080}}}"
HOST_CKPT_DIR="${X2_CHECKPOINTS_DIR:-$(realpath ../models)}"
# Repo root on the host -- bind-mounted into the container at /workspace/sonic
# (see docker-compose.yml `../..:/workspace/sonic`). We computed pwd already
# via the `cd "$(dirname "$0")"` at the top of the script, so two levels up
# is unambiguously the repo root regardless of how the script was invoked
# (relative path, absolute path, symlink, from any cwd).
HOST_REPO_DIR="$(realpath ../..)"

# --model is required for the deploy run. We do NOT default it on purpose:
# silently picking a checkpoint behind the operator's back is exactly the
# class of mistake (wrong run, stale export, partial download) that is
# painful to debug at the gantry. Make the choice visible at every call.
#
# Convenient host-side staging dir for in-repo models, bind-mounted into the
# container at /workspace/sonic/gear_sonic_deploy/models/ via the
# `../..:/workspace/sonic` mount in docker-compose.yml. Listed in the help
# below so the operator can see what's already on hand without rummaging.
HOST_MODELS_DIR="${HOST_REPO_DIR}/gear_sonic_deploy/models"

HAS_NO_STOP_MC=false
for arg in "${PASSTHROUGH[@]}"; do
    [[ "$arg" == "--no-stop-mc" ]] && HAS_NO_STOP_MC=true && break
done

# Apply the wrapper-level safe default. Only inject in dryrun mode (shell /
# oneshot do not run deploy_x2.sh at all), and only if the operator
# didn't either: (a) already pass --no-stop-mc themselves, or (b) opt out
# of the safe default with --stop-mc.
if [[ "$RUN_MODE" == "dryrun" ]] && $INJECT_NO_STOP_MC && ! $HAS_NO_STOP_MC; then
    PASSTHROUGH+=(--no-stop-mc)
    HAS_NO_STOP_MC=true
    NO_STOP_MC_INJECTED=true
else
    NO_STOP_MC_INJECTED=false
fi

if [[ "$RUN_MODE" == "dryrun" ]]; then
    HAS_MODEL_FLAG=false
    for arg in "${PASSTHROUGH[@]}"; do
        if [[ "$arg" == "--model" ]]; then HAS_MODEL_FLAG=true; break; fi
    done
    if ! $HAS_MODEL_FLAG; then
        echo
        echo -e "${RED}error:${NC} --model is required."
        echo
        echo -e "Pass an ONNX checkpoint path (resolved inside the container)."
        echo -e "Two bind mounts are available:"
        echo -e "  ${BOLD}/workspace/sonic/${NC}       <- repo root (host: $HOST_REPO_DIR)"
        echo -e "  ${BOLD}/workspace/checkpoints/${NC} <- ${HOST_CKPT_DIR}"
        echo
        if [[ -d "$HOST_MODELS_DIR" ]] && compgen -G "$HOST_MODELS_DIR/*.onnx" >/dev/null; then
            echo -e "Models already staged in-repo (gear_sonic_deploy/models/):"
            for f in "$HOST_MODELS_DIR"/*.onnx; do
                echo -e "  ${GREEN}/workspace/sonic/gear_sonic_deploy/models/$(basename "$f")${NC}"
            done
            echo
        fi
        echo -e "Examples:"
        echo -e "  $0 --model /workspace/sonic/gear_sonic_deploy/models/x2_sonic_16k.onnx"
        echo -e "  $0 --model /workspace/checkpoints/run-.../exported/model_step_016000_g1.onnx \\\\"
        echo -e "       --autostart-after 5"
        echo
        exit 2
    fi
fi

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------
echo
echo -e "${BOLD}${CYAN}=============================================================${NC}"
echo -e "${BOLD}${CYAN}        X2 SONIC -- get-ready (REAL ROBOT bring-up)${NC}"
echo -e "${BOLD}${CYAN}=============================================================${NC}"
echo
case "$RUN_MODE" in
    dryrun)
        echo -e "Mode:   ${GREEN}deploy_x2.sh --dry-run${NC} (policy ticks but publishes stiffness=damping=0)"
        echo -e "Args:   ${PASSTHROUGH[*]}"
        if $HAS_NO_STOP_MC; then
            if $NO_STOP_MC_INJECTED; then
                echo -e "        ${GREEN}--no-stop-mc auto-injected${NC} (wrapper default for --dry-run):"
                echo -e "        ${GREEN}  Motion Control (MC) stays up, robot stays actively PD-held.${NC}"
                echo -e "        ${GREEN}  Override with --stop-mc if you want the deploy to POST stop_app${NC}"
                echo -e "        ${GREEN}  to the Environment Manager (EM) and let the robot go passive.${NC}"
            else
                echo -e "        ${GREEN}--no-stop-mc set${NC}: MC stays up, robot stays actively PD-held."
            fi
        else
            echo -e "        ${YELLOW}--stop-mc requested${NC}: deploy_x2.sh will POST stop_app to EM."
            echo -e "        ${YELLOW}  MC releases the PD-hold and motors go passive (gravity acts;${NC}"
            echo -e "        ${YELLOW}  expect a small settle on the gantry). Our policy adds zero${NC}"
            echo -e "        ${YELLOW}  torque -- robot ends up fully passive.${NC}"
        fi
        if ! printf '%s\n' "${PASSTHROUGH[@]}" | grep -q '^--autostart-after$'; then
            echo -e "        ${YELLOW}(no --autostart-after -- deploy will block on stdin 'go' before CONTROL)${NC}"
        fi
        ;;
    shell)
        echo -e "Mode:   ${YELLOW}interactive shell${NC} inside container (no deploy)"
        ;;
    oneshot)
        echo -e "Mode:   ${YELLOW}one-shot command:${NC} ${ONESHOT_CMD[*]}"
        ;;
esac
echo

# ----------------------------------------------------------------------------
# Operator gate 1: robot physically safe?
# ----------------------------------------------------------------------------
echo -e "${BOLD}[gate] Robot physical state${NC}"
echo -e "  REAL-ROBOT mode means DDS is unquarantined on the SDK ethernet."
echo -e "  Even in --dry-run, the container will be on the same DDS domain"
echo -e "  as the robot and you'll see /aima/* topics flowing."
confirm "Robot is powered, secured on the gantry, E-stop within reach?" Y || abort
echo

# ----------------------------------------------------------------------------
# Host pre-flight 1: NIC config
# ----------------------------------------------------------------------------
echo -e "${BOLD}[host preflight] NIC ${NIC}${NC}"
NIC_IP=$(ip -4 addr show "$NIC" 2>/dev/null | awk '/inet /{print $2; exit}' || true)
if [[ -n "$NIC_IP" && ( -z "$NIC_EXPECT_IP" || "$NIC_IP" == "$NIC_EXPECT_IP" ) ]]; then
    echo -e "  ${GREEN}OK${NC}  $NIC = $NIC_IP"
else
    echo -e "  ${RED}WRONG${NC}  $NIC = ${NIC_IP:-<unconfigured>} (expected ${NIC_EXPECT_IP:-a static address on the SDK subnet})"
    echo -e "  ${YELLOW}Fix:  sudo ip addr add <LAPTOP_IP>/24 dev $NIC && sudo ip link set $NIC up${NC}"
    confirm "Continue anyway? (ping will almost certainly fail)" N || abort
fi
echo

# ----------------------------------------------------------------------------
# Host pre-flight 2: ping the robot
# ----------------------------------------------------------------------------
echo -e "${BOLD}[host preflight] ping ${ROBOT}${NC}"
if ping -c1 -W2 "$ROBOT" &>/dev/null; then
    echo -e "  ${GREEN}OK${NC}  $ROBOT reachable"
else
    echo -e "  ${RED}FAIL${NC}  $ROBOT unreachable"
    echo -e "  ${YELLOW}Check the SDK ethernet cable and that the robot is fully booted.${NC}"
    confirm "Continue anyway?" N || abort
fi
echo

# ----------------------------------------------------------------------------
# Host pre-flight 3: PC1 Environment Manager (EM) HTTP API
# ----------------------------------------------------------------------------
# EM is the on-robot process supervisor that lives on PC1 (<PC1_IP>:50080).
# It's the same daemon `aima em start-app/stop-app` talks to under the hood.
# We need it later because deploy_x2.sh stops the Motion Control (MC) app
# via a POST to EM so HAL joint commands take effect.
#
# EM has no documented status / list endpoint -- the only routes used by the
# reference impl (AgiBot's record-and-replay mc_control.py) are
# /json/stop_app and /json/start_app, and we very much do not want to hit
# those just to check liveness. Instead, we GET the root path and accept
# ANY HTTP response code (including 404) as proof of life: a 404 from EM
# means the daemon is alive, parsing HTTP, and would happily accept a POST
# to the real routes. Only timeout / connection-refused (curl returns "000")
# counts as fail.
echo -e "${BOLD}[host preflight] PC1 Environment Manager (EM) HTTP API ${PC1_EM:-<PC1_HOST unset>}${NC}"
if [[ -z "$PC1_EM" ]]; then
    EM_HTTP_CODE="000"
    echo -e "  ${YELLOW}skipped${NC} (export PC1_HOST=<PC1_IP> or X2_MC_EM_URL to probe EM)"
else
    EM_HTTP_CODE=$(curl -sS --max-time 2 -o /dev/null -w "%{http_code}" "$PC1_EM/" 2>/dev/null || echo "000")
fi
if [[ "$EM_HTTP_CODE" =~ ^[1-5][0-9][0-9]$ ]]; then
    echo -e "  ${GREEN}OK${NC}  EM alive (HTTP $EM_HTTP_CODE on /)"
else
    echo -e "  ${YELLOW}WARN${NC}  EM (Environment Manager) on PC1 not responding (curl returned no HTTP code)"
    echo -e "  ${YELLOW}MC (Motion Control) stop will fail; harmless in --dry-run.${NC}"
    # In dry-run, deploy_x2.sh STILL POSTs stop_app unless --no-stop-mc was
    # passed -- so a dead EM is fatal even in dry-run unless the operator
    # confirmed --no-stop-mc explicitly.
    if [[ "$RUN_MODE" == "dryrun" ]] && $HAS_NO_STOP_MC; then
        confirm "Continue anyway? (--no-stop-mc means deploy will not POST stop_app)" Y || abort
    else
        confirm "Continue anyway?" N || abort
    fi
fi
echo

# ----------------------------------------------------------------------------
# Host pre-flight 4: stale x2sim containers
#
# Only do this in the default deploy flow. In --shell / -- <cmd> modes the
# operator is EXPECTLY launching a second container to run alongside an
# existing one (e.g. a topic echo while a deploy is already up),
# and any "Up <seconds>" container we'd flag as stale is in fact the thing
# they want to coexist with. Killing it would torpedo the active deploy.
# ----------------------------------------------------------------------------
if [[ "$RUN_MODE" == "dryrun" ]]; then
    echo -e "${BOLD}[host preflight] stale x2sim containers${NC}"
    STALE=$(docker ps -a --filter "name=docker_x2-x2sim" --format "{{.Names}}\t{{.Status}}" 2>/dev/null || true)
    if [[ -n "$STALE" ]]; then
        echo -e "  ${YELLOW}Found:${NC}"
        echo "$STALE" | sed 's/^/    /'
        # Default to N here -- if any of these are "Up <N> seconds" they may
        # be a deploy from another terminal. Operator should look before
        # blowing them away.
        if confirm "Remove them before starting a fresh container? (CAREFUL: kills any running deploy)" N; then
            docker ps -a --filter "name=docker_x2-x2sim" -q | xargs -r docker rm -f >/dev/null
            echo -e "  ${GREEN}removed${NC}"
        fi
    else
        echo -e "  ${GREEN}OK${NC}  none"
    fi
    echo
else
    echo -e "${BOLD}[host preflight] stale x2sim containers${NC}  ${GREEN}skipped${NC} ($RUN_MODE mode -- intentionally coexisting)"
    echo
fi

# ----------------------------------------------------------------------------
# Host pre-flight 5: --model file exists on host
# (resolves /workspace/checkpoints/* back to $X2_CHECKPOINTS_DIR/* via
# the bind mount declared in docker-compose.yml)
# ----------------------------------------------------------------------------
if [[ "$RUN_MODE" == "dryrun" ]]; then
    MODEL_PATH=""
    for ((i=0; i<${#PASSTHROUGH[@]}; i++)); do
        if [[ "${PASSTHROUGH[$i]}" == "--model" ]]; then
            MODEL_PATH="${PASSTHROUGH[$((i+1))]:-}"
            break
        fi
    done

    if [[ -n "$MODEL_PATH" ]]; then
        echo -e "${BOLD}[host preflight] model file${NC}"
        # Translate in-container paths back to host paths for the file
        # check. Both bind mounts come from docker-compose.yml:
        #   /workspace/sonic       -> repo root (../.. on host)
        #   /workspace/checkpoints -> ${X2_CHECKPOINTS_DIR:-gear_sonic_deploy/models}
        HOST_MODEL="$MODEL_PATH"
        case "$MODEL_PATH" in
            /workspace/sonic/*)
                HOST_MODEL="${HOST_REPO_DIR}${MODEL_PATH#/workspace/sonic}"
                ;;
            /workspace/checkpoints/*)
                HOST_MODEL="${HOST_CKPT_DIR}${MODEL_PATH#/workspace/checkpoints}"
                ;;
        esac
        if [[ -f "$HOST_MODEL" ]]; then
            SIZE=$(du -h "$HOST_MODEL" | cut -f1)
            MTIME=$(stat -c '%y' "$HOST_MODEL" | cut -d. -f1)
            echo -e "  ${GREEN}OK${NC}  $(basename "$HOST_MODEL") ($SIZE, modified $MTIME)"
            echo -e "  host:      $HOST_MODEL"
            echo -e "  container: $MODEL_PATH"
        else
            echo -e "  ${RED}MISSING${NC}  $HOST_MODEL"
            echo -e "  ${YELLOW}deploy_x2.sh will fail at the asset check.${NC}"
            confirm "Continue anyway?" N || abort
        fi
        echo
    fi
fi

# ----------------------------------------------------------------------------
# Build (cached layers are 0.0s; the new ping layer added ~7s last time)
# ----------------------------------------------------------------------------
echo -e "${BOLD}[docker] compose build${NC}"
docker compose "${COMPOSE_FILES[@]}" build
echo

# ----------------------------------------------------------------------------
# X11 forwarding (best-effort -- only matters if a later --shell or one-shot
# wants to render anything graphical)
# ----------------------------------------------------------------------------
if command -v xhost >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
    xhost +SI:localuser:root >/dev/null || \
        echo -e "${YELLOW}[xhost] +SI:localuser:root failed; X11 apps in the container will not be able to open a window.${NC}"
fi

# ----------------------------------------------------------------------------
# Operator gate 2: final go/no-go before entering the container
# ----------------------------------------------------------------------------
echo -e "${BOLD}[gate] About to enter container${NC}"
echo -e "  network_mode: host"
echo -e "  DDS:          ROS_DOMAIN_ID=0, ROS_LOCALHOST_ONLY=0 (FastDDS multicast discovery over $NIC)"
case "$RUN_MODE" in
    dryrun)
        echo -e "  command:      ${BOLD}./gear_sonic_deploy/deploy_x2.sh --dry-run ${PASSTHROUGH[*]:-}${NC}"
        if $HAS_NO_STOP_MC; then
            echo -e "  ${GREEN}NOTE:${NC} --no-stop-mc set -- MC stays up, robot stays actively held."
            echo -e "        deploy_x2.sh will skip the MC stop step and only ask for the"
            echo -e "        final launch gate."
        else
            echo -e "  ${YELLOW}NOTE:${NC} deploy_x2.sh will POST stop_app to EM (motors go passive,"
            echo -e "        gravity acts -- our policy adds zero torque) and then ask for"
            echo -e "        a final launch gate. This wrapper's job ends once you say go."
        fi
        ;;
    shell)
        echo -e "  command:      ${BOLD}bash${NC} (interactive)"
        ;;
    oneshot)
        echo -e "  command:      ${BOLD}${ONESHOT_CMD[*]}${NC}"
        ;;
esac
confirm "Proceed?" Y || abort
echo

# ----------------------------------------------------------------------------
# Hand off to the container
# ----------------------------------------------------------------------------
case "$RUN_MODE" in
    dryrun)
        DEPLOY_ARGS=(--dry-run "${PASSTHROUGH[@]}")
        QUOTED=$(printf ' %q' "${DEPLOY_ARGS[@]}")
        exec docker compose "${COMPOSE_FILES[@]}" run --rm x2sim bash -c \
            "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && cd /workspace/sonic && exec ./gear_sonic_deploy/deploy_x2.sh${QUOTED}"
        ;;
    shell)
        exec docker compose "${COMPOSE_FILES[@]}" run --rm x2sim
        ;;
    oneshot)
        QUOTED=$(printf ' %q' "${ONESHOT_CMD[@]}")
        exec docker compose "${COMPOSE_FILES[@]}" run --rm x2sim bash -c \
            "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec${QUOTED}"
        ;;
esac
