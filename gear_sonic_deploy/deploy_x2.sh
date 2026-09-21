#!/bin/bash
set -e

# ============================================================================
# X2 Ultra Deploy - Deployment Script
# ============================================================================
# Companion to deploy.sh (which targets G1). End-to-end build + pre-flight +
# launch for the agi_x2_deploy_onnx_ref ROS 2 package on AgiBot X2 Ultra.
#
# Topologies (per docs/source/user_guide/x2_sonic_deploy_real.md):
#   local  - Build and run on this machine (your laptop). DDS auto-discovers
#            the robot via the wired SDK ethernet (laptop NIC at <LAPTOP_IP>/24,
#            PC2 dev unit at <PC2_IP>; set PC2_HOST). The C++ deploy, MC stop/start
#            helpers, RAMP_OUT trap, hand bridge and recorder all live on
#            the laptop. Requires the wired link.
#   onbot  - Run NATIVELY on PC2 (the robot's onboard Jetson Orin NX). This
#            script is expected to be executed on PC2 itself -- launched
#            via ssh from the laptop, typically inside a tmux session
#            (see scripts/x2_pc2_daemons.sh). Everything bash-side
#            (Y/n safety gate, MC stop_app/start_app via PC1 EM HTTP
#            or `aima em` CLI fallback, sentinel handoff, RAMP_OUT trap,
#            HOLD_FOR_MC handoff) AND the C++ deploy run on the same
#            shell on PC2, so the orchestration survives a laptop-side
#            WiFi disconnect mid-run. Assumes ``pc2_bringup.sh`` has
#            already staged the colcon workspace, Python venv, ONNX
#            Runtime and ONNX policies under ``$ONBOT_PREFIX`` (default
#            /home/run/gear-sonic). The hand bridge and motor monitor are
#            launched separately by x2_pc2_daemons.sh (so they share
#            the same lifecycle as the deploy but in their own tmux
#            sessions). The recorder lives on the laptop with the
#            planner stack and is not touched here.
#   sim    - Build + run locally against a MuJoCo physics sim, on isolated
#            loopback DDS (ROS_LOCALHOST_ONLY=1 + private ROS_DOMAIN_ID).
#            The sibling Python bridge scripts/x2_mujoco_ros_bridge.py is
#            launched in the background; it steps physics at 1 kHz, publishes
#            joint state + IMU on the same /aima/* topics the deploy
#            subscribes to, and applies the deploy's PD commands as MuJoCo
#            torques. This is the X2 analogue of G1's sim mode (which
#            selects a loopback DDS interface and pairs with the
#            unitree_sdk2py MuJoCo bridge).
#
# Local/onbot pre-flight:
#   1. Ping PC2 to confirm we can reach the robot (local mode only;
#      onbot mode is already ON PC2 so this is skipped).
#   2. Verify ROS 2 topics are visible (joint state for each group + IMU).
#   3. Stop the MC module by POSTing stop_app to PC1's Environment Manager
#      HTTP API (the same mechanism `aima em stop-app mc` uses underneath).
#      On firmware where the HTTP endpoint has been removed (the EM service
#      moved to ROS 2 services in late 2026), `mc_em_post` falls back to
#      `aima em stop-app mc` / `aima em start-app mc` on PC2's PATH.
#      Use --no-stop-mc to skip.
#
# Sim pre-flight:
#   1. Verify the bridge script + MJCF + Python deps are importable.
#   2. Isolate DDS to loopback so a sim run on the SDK subnet cannot fight a
#      real robot (sets ROS_LOCALHOST_ONLY=1 and ROS_DOMAIN_ID).
#
# Usage:
#   ./deploy_x2.sh [OPTIONS] [local|onbot|sim]
#
# Examples:
#   ./deploy_x2.sh --model /opt/x2_models/model_step_016000_g1.onnx --dry-run
#   ./deploy_x2.sh --model ./model.onnx --motion ./standing.x2m2 onbot
#   ./deploy_x2.sh sim --model ./model.onnx --sim-viewer --autostart-after 5
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
DIM='\033[2;37m'
NC='\033[0m' # No Color

# Millisecond-precision wall-clock timestamp for operational log lines.
# Format: HH:MM:SS.mmm (24h, local TZ). Kept short so the colored tag
# stays close to the message. Wrap usage as: echo -e "$(ts) ${BLUE}[tag]${NC} message".
ts() {
    printf '%s[%s]%s' "$DIM" "$(date +'%H:%M:%S.%3N')" "$NC"
}

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Save the user's original CWD BEFORE we cd into SCRIPT_DIR so abspath() can
# resolve relative paths the way the user expects (against where they ran the
# command, not against gear_sonic_deploy/). Without this, a repo-rooted
# invocation like ``./gear_sonic_deploy/deploy_x2.sh ... --motion
# gear_sonic/data/motions/x2_recorded/mc_gestures/bow_001.pkl`` silently resolves the
# motion path against gear_sonic_deploy/ and fails to find the file.
USER_CWD="$(pwd)"

# Repo root (parent of gear_sonic_deploy/). Prefer ``.venv/bin/python`` for
# auxiliary Python (hand bridge, tuning YAML translator, sim MuJoCo bridge,
# --record helper, MC escalator) so we do not silently use the host's
# unconfigured ``python3``. Inside docker_x2, the repo mount may or may not
# include a .venv; when absent we fall back to ``python3`` (ROS image).
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_REPO_VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "$DEPLOY_REPO_VENV_PYTHON" ]]; then
    DEPLOY_AUX_PYTHON="$DEPLOY_REPO_VENV_PYTHON"
else
    DEPLOY_AUX_PYTHON="python3"
fi

# ============================================================================
# Auto-relaunch inside the docker_x2/x2sim container if invoked from a host
# shell that doesn't have ROS / aimdk_msgs sourced. Lets the user run
#
#   $ ./gear_sonic_deploy/deploy_x2.sh local --model $CKPT_ROOT/.../m.onnx ...
#
# directly from the repo root without first doing the
# `cd docker_x2 && docker compose run --rm --service-ports x2sim bash -c ...`
# dance.
#
# Detection: we treat "running on the host" as "/workspace/sonic doesn't
# exist" (the compose file binds the repo there inside the container).
# Override with --no-docker for advanced users who already have ROS sourced
# on the host, and respect $X2_DEPLOY_IN_DOCKER as a re-entry guard so the
# in-container invocation doesn't loop.
#
# Mount strategy: bind-mount $HOME at the same path inside the container.
# The compose file already binds the repo at /workspace/sonic and
# $X2_CHECKPOINTS_DIR at /workspace/checkpoints, but those are CONTAINER
# paths -- if the user passes $CKPT_ROOT/.../m.onnx (the natural
# host path) we want it to resolve identically inside without the wrapper
# having to translate every PATH-bearing flag. Mounting $HOME -> $HOME at
# the same absolute path solves both directions in one line.
#
# Real-mode env: the compose file pins ROS_LOCALHOST_ONLY=1 / ROS_DOMAIN_ID=73
# (sim DDS isolation). Real-robot modes need to talk to the actual robot, so
# we override both unless the user has set X2_REAL_DOMAIN_ID. Sim mode keeps
# the compose defaults.
maybe_relaunch_in_docker() {
    # Already inside the container, or invoked from a host shell that already
    # has ROS sourced and the user opted out via --no-docker.
    if [[ -d /workspace/sonic ]] || [[ -n "${X2_DEPLOY_IN_DOCKER:-}" ]]; then
        return 0
    fi
    for a in "$@"; do
        case "$a" in
            # User opt-out, or "no-ROS-needed" invocations: --help prints
            # usage and exits, --build-only doesn't talk to robot or ROS at
            # all (just colcon). We don't want to spin up a docker container
            # just to print --help text or rebuild C++.
            --no-docker|-h|--help|--build-only) return 0 ;;
            # `onbot` mode runs natively on PC2 (the robot's Jetson Orin NX).
            # PC2 has no docker_x2 image, no /workspace/sonic bind mount, and
            # ROS is sourced from the host /opt/ros/humble. Trying to relaunch
            # under docker would fail with "x2sim image not found" at best,
            # or eat several seconds and emit a confusing error at worst.
            # Treat onbot exactly like --no-docker.
            onbot) return 0 ;;
        esac
    done

    # docker not available -> let the caller fail naturally (e.g. with the
    # rclpy ModuleNotFoundError that the preflight will throw). We'd rather
    # fail loudly than silently no-op.
    if ! command -v docker &>/dev/null; then
        echo -e "\033[1;33mNote: 'docker' not in PATH and ROS doesn't look sourced;\033[0m" >&2
        echo -e "\033[1;33m       deploy_x2.sh is going to fail on rclpy/aimdk_msgs imports.\033[0m" >&2
        echo -e "\033[1;33m       Either install docker + run the docker_x2 container, or${NC}" >&2
        echo -e "\033[1;33m       source ROS 2 + aimdk_msgs and re-run with --no-docker.${NC}" >&2
        return 0
    fi

    local compose_dir="$SCRIPT_DIR/docker_x2"
    if [[ ! -f "$compose_dir/docker-compose.yml" ]]; then
        return 0
    fi

    # Sniff mode (first non-flag positional arg). Defaults to "local" to match
    # the deploy_x2.sh default further down.
    local mode="local"
    for a in "$@"; do
        case "$a" in
            sim|local|onbot)
                mode="$a"; break ;;
            -*) ;;  # flag, skip
            *) break ;;  # unknown positional -> stop sniffing
        esac
    done

    local env_overrides=()
    if [[ "$mode" != "sim" ]]; then
        env_overrides+=(
            "-e" "ROS_LOCALHOST_ONLY=0"
            "-e" "ROS_DOMAIN_ID=${X2_REAL_DOMAIN_ID:-0}"
        )
    fi

    local tty_args=("-T")  # no TTY by default (works in scripts/CI)
    if [[ -t 0 && -t 1 ]]; then
        tty_args=()  # interactive shell -> let docker compose allocate TTY
    fi

    echo -e "\033[0;34m[auto-docker]\033[0m re-exec inside docker_x2/x2sim ($mode mode)"
    echo -e "\033[0;34m[auto-docker]\033[0m mounting \$HOME ($HOME) at $HOME inside container"
    if [[ "$mode" != "sim" ]]; then
        echo -e "\033[0;34m[auto-docker]\033[0m clearing sim DDS isolation (ROS_LOCALHOST_ONLY=0, ROS_DOMAIN_ID=${X2_REAL_DOMAIN_ID:-0})"
    fi
    echo -e "\033[0;34m[auto-docker]\033[0m use --no-docker to bypass (requires ROS sourced on host)"
    echo ""

    local pwd_abs="$USER_CWD"
    # Use the CONTAINER script path so SCRIPT_DIR inside the re-exec resolves
    # to /workspace/sonic/gear_sonic_deploy. That path has the colcon
    # install/build/log docker volumes attached (see docker-compose.yml);
    # the host-side $HOME mount we add below points at the same source
    # files but does NOT carry the colcon volumes, so a SCRIPT_DIR resolved
    # against the host path would yield "Package agi_x2_deploy_onnx_ref not
    # found" at ros2 run time.
    local script_in_container="/workspace/sonic/gear_sonic_deploy/$(basename "${BASH_SOURCE[0]}")"

    cd "$compose_dir"
    export X2_DEPLOY_IN_DOCKER=1
    exec docker compose run --rm --service-ports \
        -e X2_WORLD -e X2_ISAAC_HOST -e X2_ANCHOR_ORI_MODE -e X2_PLANT -e X2_PLANT_YAML -e X2_REPO_ROOT \
          -e SIM_BRIDGE_EXTRA \
        "${tty_args[@]}" \
        "${env_overrides[@]}" \
        -e "X2_DEPLOY_IN_DOCKER=1" \
        -e "X2_HOST_HOME=$HOME" \
        -v "$HOME:$HOME:rw" \
        -w "$pwd_abs" \
        x2sim \
        bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec "$@"' \
        bash "$script_in_container" "$@"
}

maybe_relaunch_in_docker "$@"

cd "$SCRIPT_DIR"

PKG_NAME="agi_x2_deploy_onnx_ref"
PKG_DIR_REL="src/x2/agi_x2_deploy_onnx_ref"
SONIC_COMMON_REL="src/common"

# ============================================================================
# Defaults
# ============================================================================

MODE_DEFAULT="local"
# PC2 (robot dev unit) address. No baked-in default: set PC2_HOST in the
# environment (see x2_discover_network.sh) or pass --robot-host. Local mode
# refuses to run with it unset; onbot mode ignores it (already on PC2).
ROBOT_HOST_DEFAULT="${PC2_HOST:-}"
ROBOT_USER_DEFAULT="agi"
# ONBOT prefix conventions (must match pc2_bringup.sh layout). Everything
# the bringup script stages on PC2 lives under ONBOT_PREFIX, so flipping
# --onbot-prefix relocates the entire install in one shot:
#   $ONBOT_PREFIX/
#     ws/install/setup.bash       <- colcon overlay  (--onbot-ws)
#     venv/bin/python3            <- python with pyzmq + system rclpy
#                                    + aimdk_msgs           (--onbot-venv)
#     onnxruntime/lib/            <- libonnxruntime.so for the deploy
#                                                          (--onbot-onnxruntime)
#     policies/*.onnx             <- pre-staged ONNX checkpoints
#     log/                        <- per-run CSV dirs, monitor JSONLs,
#                                    tmux launch scripts
ONBOT_PREFIX_DEFAULT="/home/run/gear-sonic"
ONBOT_WS_DEFAULT="${ONBOT_PREFIX_DEFAULT}/ws"
ONBOT_VENV_DEFAULT="${ONBOT_PREFIX_DEFAULT}/venv"
ONBOT_ONNXRUNTIME_DEFAULT="${ONBOT_PREFIX_DEFAULT}/onnxruntime"
# aimdk_msgs (Python bindings, cmake config, libs) live in the AgiBot
# system tree on PC2. /opt/ros/humble/setup.bash does NOT add this to
# AMENT_PREFIX_PATH so the deploy + python helpers can't see it without
# us doing it explicitly. Same path used by x2_pc2_daemons.sh.
ONBOT_AIMDK_PREFIX_DEFAULT="/agibot/software/housekeeper/bin/aimdk_msgs"
ONNXRUNTIME_ROOT_DEFAULT="/opt/onnxruntime"
SIM_DOMAIN_ID_DEFAULT="73"
# PC1 (<PC1_IP>) Environment Manager HTTP API. This is the underlying
# mechanism `aima em start-app/stop-app` uses; talking to it directly avoids
# requiring an ssh key into the robot and works from inside docker as long
# as the host has a route to PC1 (true once the SDK NIC is on <LAPTOP_IP>/24
# and `network_mode: host` is in effect). Vetted in the AgiBot record-and-
# replay reference (mc_control.py).
# Resolved from PC1_HOST (+ PC1_EM_PORT) or X2_MC_EM_URL; empty when neither
# is set, in which case mc_em_post skips HTTP and goes straight to the
# `aima em` CLI fallback (which only exists on PC2). Pass --mc-em-url to
# override.
if [[ -n "${X2_MC_EM_URL:-}" ]]; then
    MC_EM_URL_DEFAULT="${X2_MC_EM_URL}"
elif [[ -n "${PC1_HOST:-}" ]]; then
    MC_EM_URL_DEFAULT="http://${PC1_HOST}:${PC1_EM_PORT:-50080}"
else
    MC_EM_URL_DEFAULT=""
fi

MODE="$MODE_DEFAULT"
ROBOT_HOST="$ROBOT_HOST_DEFAULT"
ROBOT_USER="$ROBOT_USER_DEFAULT"
ONBOT_PREFIX="$ONBOT_PREFIX_DEFAULT"
ONBOT_WS="$ONBOT_WS_DEFAULT"
ONBOT_VENV="$ONBOT_VENV_DEFAULT"
ONBOT_ONNXRUNTIME="$ONBOT_ONNXRUNTIME_DEFAULT"
ONBOT_AIMDK_PREFIX="$ONBOT_AIMDK_PREFIX_DEFAULT"
ONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT_DEFAULT"
MC_EM_URL="$MC_EM_URL_DEFAULT"
# Set to true once we've successfully POSTed stop_app, so the cleanup trap
# knows it has to POST start_app on exit. Never set true if MC was already
# down when we started.
MC_STOPPED_BY_US=false

# Deploy CLI passthrough flags
MODEL=""
MOTION=""
# VLA mode flips the deploy from "motion file replay" to "ZMQ pose subscriber".
# When VLA_MODE=true, --motion is ignored and the deploy subscribes to
# VLA_ZMQ_HOST:VLA_ZMQ_PORT instead. VLA_DEBUG_PORT (when non-zero) makes the
# deploy also bind a PUB socket so dump_x2_debug.py can tail the telemetry.
# Defaults match docs/source/references/x2_zmq_protocol.md.
VLA_MODE=false
VLA_ZMQ_HOST="localhost"
VLA_ZMQ_PORT="5556"
VLA_ZMQ_TOPIC="pose"
VLA_DEBUG_PORT="5557"
VLA_DEBUG_TOPIC="x2_debug"
# SAFE_IDLE recovery channel. Quest 3 A+B chord publishes to this SUB
# on the laptop side; the C++ deploy SUBs to it from PC2 (onbot) or
# from the local laptop (local). Empty host = leave the resume socket
# disabled (the deploy still self-exits SAFE_IDLE on first non-stale
# pose; the chord just makes it explicit + faster).
VLA_RESUME_HOST=""
VLA_RESUME_PORT="5566"
VLA_RESUME_TOPIC="pose_resume"
# Generic raw passthrough to the C++ binary. Accumulates across
# multiple --deploy-extra-arg invocations so callers can add new
# CliArgs flags without having to plumb a new --foo flag through
# this wrapper every time. Order is preserved; appended after
# everything else so it wins last-write-wins parsing.
DEPLOY_EXTRA_ARGS=()

# Wrist bypass: forwarded to the deploy binary as --wrist-bypass {off,ik}.
# When set to "ik" (and --vla is also set), the C++ deploy overwrites
# target_pos_mj for the 4 broken wrist DOFs (left/right wrist_pitch +
# wrist_roll, MJ indices {20,21,27,28}) with the latest IK reference from
# the ZMQ pose feed. wrist_yaw is left under SONIC because it tracks fine.
# Empty string = leave it disabled (legacy behaviour); the C++ binary
# defaults to "off" so sim-to-real fidelity tests are unchanged. See
# x2_deploy_onnx_ref.cpp::CliArgs::WristBypass for the full rationale.
WRIST_BYPASS=""
# Head bypass: forwarded to the deploy binary as --head-bypass {off,ref}.
# When "ref" (and the pose feed is ZMQ), the C++ deploy overwrites
# target_pos_mj for the 2 head DOFs (head_yaw MJ 29, head_pitch MJ 30)
# with the ZMQ reference BEFORE the safety stack. SONIC's head tracking
# saturates at ~+-10 deg (static-head training corpus), so the pad's
# head-look overlay needs this to reach the joint. Empty = binary default
# ("off"). See head_bypass.hpp for the full rationale.
HEAD_BYPASS=""
# Set when --motion is a PKL or YAML and we bake an x2m2 on the fly. Holds
# the absolute path to the original PKL/YAML so the wrapper can route it to
# the MuJoCo bridge as --sim-motion (matching what eval_x2_mujoco_onnx.py
# RSIs from). Empty when --motion was passed as an .x2m2 (legacy passthrough)
# or omitted entirely.
MOTION_SOURCE=""
# Per-run tempdir we bake the x2m2 into so it tears down on script exit.
MOTION_BAKE_TMPDIR=""
LOG_DIR=""
AUTOSTART=""
# Auto-shutdown N seconds after entering CONTROL state. Empty string =
# unbounded (run until Ctrl-C). Useful for bounded dry-run smoke tests where
# you don't want to babysit Ctrl-C.
MAX_DURATION=""
TILT_COS=""
RAMP_SECONDS=""
# Per-joint hard clamp on |target - default_angles|, in radians. Empty string
# = leave it disabled (legacy behaviour). Historically grounded values from
# this repo's recorded runs (see docs/source/user_guide/x2_sim_to_real_wip.md
# §"Recordings inventory" and the milestones under
# docs/source/user_guide/milestones/):
#   * 0.30  -- "first powered run with a new checkpoint OR new motion playlist"
#              (the conservative.yaml preset value). This is what every actual
#              first-powered run on this robot has used: iter-4k minimal_v1
#              stand (2026-05-02), all early take_a_sip shakeouts, and is the
#              right starting point for the first powered Quest 3 teleop run
#              (operator-driven motion = "new motion playlist" by the rule).
#   * 0.80  -- the expressive.yaml preset; switch to this once conservative
#              has cleared on the same model+motion class (paired with
#              --target-lpf-hz 8 for real-sensor jitter tame).
#   * 1.50  -- the value used for the first iter-22000 powered walk
#              (2026-05-03 casual_walk_v1) and shipped in the published
#              sim-to-real anchors b/c/d. Settled "play freely" value once
#              the full pipeline is trusted.
# Cleaner than inlining: pass --tuning-config configs/real_deploy_tuning/
# {conservative,expressive}.yaml (also gives the matching ramp_seconds /
# return_seconds / tilt_cos). See policy_parameters.hpp for the trained
# standing pose this clamps around.
MAX_TARGET_DEV=""

# Per-group max_target_dev overrides, forwarded to the deploy binary as
# --max-target-dev-{leg,waist,arm,head}. Empty = inherit MAX_TARGET_DEV
# (the global) for that group; positive number = clamp this group at the
# given radian deviation. Designed for the case where one joint family
# can safely take more travel than another -- e.g. on X2 Ultra the legs
# run kp ~99 Nm/rad and the arms kp ~14 Nm/rad, so a 1.50 rad uniform
# clamp produces ~7x the leg torque for the same nominal travel as the
# arms. Tight legs/waist + wide arms is the natural setting for any
# teleop-driven session (see configs/real_deploy_tuning/expressive.yaml,
# which now sets leg=0.30, arm=1.50 by default).
MAX_TARGET_DEV_LEG=""
MAX_TARGET_DEV_WAIST=""
MAX_TARGET_DEV_WAIST_PITCH=""  # per-JOINT (MJ 13); wins over the group
MAX_TARGET_DEV_ARM=""
MAX_TARGET_DEV_HEAD=""

# Deployment-time PD trim, forwarded to the binary as --kp-scale[-FAMILY]
# and --kd-scale[-FAMILY]. Empty string = use binary default (1.0, no
# trim) unless the tuning config sets a value. Real-deploy only -- the
# wrapper does NOT block these flags in sim mode the way it blocks
# --tuning-config, because the operator might want to A/B a PD bump in
# MuJoCo before pushing to robot; just be aware that doing so DEFEATS
# sim-profile parity with eval_x2_mujoco.py for the duration of that run.
#
# Why this exists: IsaacLab integrates PD implicitly against the joint-
# space inertia + armature at training, while MuJoCo and real X2 apply
# explicit ctrl-driven torque, so the same numerical KP behaves softer
# at deploy than at training. Symptom: robot stands fine but wobbles on
# nudge. Standard fix is to bump deployed PD per joint family: G16b for
# X2 has set ankle=1.5 since November 2025 (in eval_x2_mujoco.py); real-
# robot operators commonly need waist=1.5 on top of that. See
# configs/real_deploy_tuning/expressive.yaml.
KP_SCALE=""
KP_SCALE_HIP=""
KP_SCALE_KNEE=""
# Ankle alias + split. The alias (KP_SCALE_ANKLE) multiplies both
# subgroups for backward compat; KP_SCALE_ANKLE_PITCH and
# KP_SCALE_ANKLE_ROLL target the sagittal vs frontal axes respectively.
# MC publishes asymmetric PD (pitch 40/3.0, roll 30/2.0), so the split
# knobs are the only way to match MC exactly.
KP_SCALE_ANKLE=""
KP_SCALE_ANKLE_PITCH=""
KP_SCALE_ANKLE_ROLL=""
# Waist alias + split. The alias (KP_SCALE_WAIST) multiplies both
# subgroups for backward compat; KP_SCALE_WAIST_YAW and KP_SCALE_WAIST_PR
# target waist_yaw_joint vs waist_pitch_joint + waist_roll_joint
# respectively. See expressive.yaml for the MC-matched values.
KP_SCALE_WAIST=""
KP_SCALE_WAIST_YAW=""
KP_SCALE_WAIST_PR=""
KP_SCALE_SHOULDER=""
KP_SCALE_ELBOW=""
KP_SCALE_WRIST=""
KP_SCALE_HEAD=""
KD_SCALE=""
KD_SCALE_HIP=""
KD_SCALE_KNEE=""
KD_SCALE_ANKLE=""
KD_SCALE_ANKLE_PITCH=""
KD_SCALE_ANKLE_ROLL=""
KD_SCALE_WAIST=""
KD_SCALE_WAIST_YAW=""
KD_SCALE_WAIST_PR=""
KD_SCALE_SHOULDER=""
KD_SCALE_ELBOW=""
KD_SCALE_WRIST=""
KD_SCALE_HEAD=""

# Output-side target LPF cutoff in Hz, forwarded to the deploy binary as
# --target-lpf-hz. Empty string = use binary default (0 = bypass = parity-
# safe). Real-deploy only -- enabling this in sim mode will diverge from
# eval_x2_mujoco.py. Typically set via the tuning config preset, but
# exposed as a passthrough flag so operators can do quick A/B sweeps off
# a known-good preset (the binary's CLI parser is last-write-wins, so
# this overrides whatever the preset put there).
TARGET_LPF_HZ=""
# Symmetric clip on the raw ONNX action (action_il) BEFORE x2_action_scale.
# Empty string = let the C++ binary use its compiled-in default of 20.0,
# which matches IsaacLab training-time config.action_clip_value. Set to a
# negative number to disable (only useful for parity-vs-old-behavior tests).
ACTION_CLIP=""
# Soft-EXIT ramp duration. When --max-duration trips, lerp target_pos back
# from the last policy command to default_angles over this many seconds
# before shutting down. Empty = let the C++ binary use its compiled-in
# default of 2.0s; set "0" to disable (legacy immediate-shutdown). See
# x2_deploy_onnx_ref.cpp::CliArgs::return_seconds for the gory details.
RETURN_SECONDS=""
IMU_TOPIC=""
INTRA_OP_THREADS=""
# Optional one-shot debug capture: write the first CONTROL-tick obs (tokenizer
# 680 + proprioception 990 + raw policy output + robot state) to PATH and
# exit immediately (passthrough to the C++ binary's --obs-dump).
OBS_DUMP=""
DRY_RUN=false

# ────────────────────────────────────────────────────────────────────────
# End-of-run smooth handoff (HOLD_FOR_MC).
#
# The deploy node has a HOLD_FOR_MC state that, after RAMP_OUT lands the
# joints at MC's STAND_DEFAULT pose, keeps publishing that pose at the
# captured MC kp/kd until MC retakes the joint command bus. The bash
# wrapper sequences this end-of-run dance:
#
#   1. RAMP_OUT completes ->
#      deploy node touches HOLD_FOR_MC_SENTINEL and enters HOLD_FOR_MC.
#   2. The polling loop below sees the sentinel and POSTs start_app +
#      SetMcAction(STAND_DEFAULT) WHILE the deploy node is still actively
#      holding the pose. No zero-torque window.
#   3. MC boots and starts publishing on /aima/hal/joint/{leg,waist}/command.
#   4. The deploy node's MC-takeover detector sees a non-self publisher
#      and exits cleanly. We `wait` on its PID for the final exit code.
#
# The sentinel file lives in $RUN_LOG_DIR (or /tmp as fallback). It is
# unlinked on every entry to HOLD_FOR_MC by the deploy node and on every
# exit from this script (cleanup trap), so a stale file from a previous
# crashed run cannot mis-trigger the next run.
#
# STAND_POSE_YAML defaults to gear_sonic_deploy/configs/x2_stand_default_pose.yaml,
# a capture of MC's STAND_DEFAULT joint targets + kp/kd (re-capture from
# /aima/hal/joint/*/command while MC stands if the firmware changes).
# HOLD_FOR_MC_TIMEOUT_S is the cap on how
# long the deploy node will keep holding the pose if MC never comes back
# (covers a failed start_app POST). Both are passthroughs to the C++
# binary's --stand-default-pose / --hold-for-mc-timeout-s / --hold-for-mc-sentinel
# flags. Set HOLD_FOR_MC_TIMEOUT_S=0 to disable HOLD_FOR_MC entirely
# (legacy path: deploy exits on RAMP_OUT, bash starts MC after).
STAND_POSE_YAML=""
# 45 s default: MC's start_app + service rebind + PASSIVE-stable takes
# ~10-12 s on PC1; we leave ~3x headroom so a slow boot doesn't trip the
# safety cap. The hold itself is cheap (50 Hz writer, MC-stand gains),
# so leaving it active longer has no cost.
HOLD_FOR_MC_TIMEOUT_S="45"
HOLD_FOR_MC_SENTINEL=""
HOLD_FOR_MC_EXIT_SENTINEL=""
MC_FIRST_PUBLISH_SENTINEL=""
ESCALATOR_OK_SENTINEL=""
ESCALATOR_PID=""
# Soft-shutdown trigger sentinel. On Ctrl-C, bash touches this file to
# tell deploy "please RAMP_OUT + HOLD_FOR_MC gracefully" instead of
# exiting immediately. Empty disables the graceful path (legacy
# behaviour: Ctrl-C -> immediate exit -> ~1-2 s zero-torque drop while
# MC reboots through PASSIVE_DEFAULT). Auto-populated alongside the
# other HOLD_FOR_MC sentinels in the launch-prep block below.
SOFT_SHUTDOWN_SENTINEL=""
# Max seconds to wait in the cleanup trap for deploy to acknowledge
# the soft-shutdown trigger and reach HOLD_FOR_MC before we POST
# start_app. Should be larger than --return-seconds (RAMP_OUT
# duration). Default is generous; the trap exits early as soon as
# HOLD_FOR_MC_SENTINEL appears.
SOFT_SHUTDOWN_WAIT_S="6"
# Set to true once we've touched the soft-shutdown trigger, so a
# second Ctrl-C inside the wait loop knows to abort the wait and fall
# through to the legacy hard-exit path. (Escape hatch in case the
# graceful path hangs.)
SOFT_SHUTDOWN_TRIGGERED=false
SOFT_SHUTDOWN_ABORTED=false
# Soft-shutdown opt-in. DEFAULT IS DISABLED (2026-05-15) -- field
# testing showed the graceful RAMP_OUT path was producing 4-6 s of
# motor whir on Ctrl-C and the robot still collapsed at the end
# (because MC restarts in PASSIVE_DEFAULT regardless of our graceful
# exit). The hard-exit path is shorter in total motor activity and
# therefore safer for the actuators until the bash<->deploy<->MC
# handoff is reworked to avoid the dual-publisher window during
# MC's PASSIVE->JOINT->STAND boot. Pass --enable-soft-shutdown to
# opt back in (e.g. for development of the handoff fix). The legacy
# --no-soft-shutdown flag is retained as a no-op for any scripts
# already passing it.
SOFT_SHUTDOWN_DISABLED=true
# When true, abort with a friendly error if MC is not currently in
# STAND_DEFAULT mode at script entry. The smooth handoff assumes MC is
# alive and balancing the robot before we take the bus -- if MC is in
# PASSIVE_DEFAULT or DAMPING_DEFAULT, the robot is already on the gantry
# / floor with zero torque and there is nothing to "hand off" from.
# Operator override: --no-require-stand-default.
REQUIRE_STAND_DEFAULT=true

# Behaviour toggles
NO_STOP_MC=false
# OBSERVE-ONLY mode. Passes --no-publish to the deploy node, which then never
# writes to the motor bus in ANY state. Setting this FORCES NO_STOP_MC=true:
# a muted deploy plus a stopped MC means nothing at all is commanding the
# joints, i.e. zero torque and the robot drops. The two must never coexist.
NO_PUBLISH=false
NO_CONFIRM=false
NO_BUILD=false
BUILD_ONLY=false
# Wire-freshness probe before the MC-stop safety gate. Opens a ZMQ SUB on
# the deploy's actual --vla-zmq-host/-port (= localhost:proxy-port in
# split-topology onbot mode; = laptop:5556 in local mode without proxy)
# and counts frames received in --wire-probe-secs (default 1.0). Aborts
# the launch BEFORE stop_app if zero frames -- catches the "proxy not
# running" / "planner stack not started" / "wrong wire host" failure
# modes that would otherwise drop MC onto a silent wire and lean the
# robot. Default ON; pass --no-wire-probe for unattended CI runs or
# when intentionally testing a starvation path.
NO_WIRE_PROBE=false
WIRE_PROBE_SECS="1.0"
# Real-robot pre-flight (gear_sonic_deploy/scripts/x2_preflight.py): the
# gantry-aware safety check that validates topics + MC presence + joint
# pose/vel/effort + IMU upright/quiet. Runs after the lightweight ros2
# topic visibility check and BEFORE Safety Gate 1/2, so a failing
# preflight aborts while MC is still holding the robot. Default ON for
# real-robot modes, force-skipped in sim.
NO_PREFLIGHT_PY=false
# Pass --strict-pose --strict-effort to the preflight (promotes pose /
# effort violations from WARN to FAIL). Default OFF for gantry bring-up
# where any natural rest pose should not gate; flip ON for floor-stand
# powered runs where the IC must be tight.
PREFLIGHT_STRICT=false
# Free-form extra args appended to the preflight invocation, e.g.
#   --preflight-args "--max-effort 10 --imu-tilt-deg 12"
# for an aggressive floor-stand bring-up. Empty by default (use
# preflight script defaults). Quoted as a single string and word-split
# at invocation time, so multi-flag passthroughs work without further
# escaping.
PREFLIGHT_ARGS=""

# Sim mode (MuJoCo bridge is the only sim driver)
SIM_DOMAIN_ID="$SIM_DOMAIN_ID_DEFAULT"
SIM_BRIDGE_REL="scripts/x2_mujoco_ros_bridge.py"
SIM_PYTHON="${SIM_PYTHON:-$DEPLOY_AUX_PYTHON}"
SIM_MJCF=""
SIM_MOTION=""
SIM_INIT_FRAME=""
SIM_VIEWER=false
# Optional viewer-camera tracking. When SIM_CAM_TRACK_BODY is non-empty
# the bridge sets cam.type=mjCAMERA_TRACKING and locks the framing onto
# that body (e.g. "pelvis"). The other three knobs are forwarded as-is.
SIM_CAM_TRACK_BODY=""
SIM_CAM_DISTANCE=""
SIM_CAM_ELEVATION=""
SIM_CAM_AZIMUTH=""
SIM_IMU_FROM=""
SIM_HOLD_STIFFNESS_MULT=""
# Bridge init-pose selector: empty = bridge default ('default'). 'gantry-hang'
# triggers the bent-knee crouch + lowered pelvis + standby PD seeded to that
# pose. See x2_mujoco_ros_bridge.py:GANTRY_HANG_OFFSETS_MJ for the details.
SIM_INIT_POSE=""
# ElasticBand suspension knobs: how far below the world [0,0,1] anchor the
# band's pull-target sits, and a stiffness multiplier on the (kp,kd) gains.
# Both empty = bridge defaults (length=0, kp_mult=1.0).
SIM_BAND_LENGTH=""
SIM_BAND_KP_MULT=""
SIM_NO_ELASTIC_BAND=false
# Set true to force the band ON even when --motion would auto-disable it.
SIM_KEEP_ELASTIC_BAND=false
SIM_BAND_RELEASE_AFTER_S=""
# --sim-profile selects between two distinct closed-loop sim test modes that
# validate DIFFERENT invariants (both must pass before going to hardware):
#   parity   -- bridge RSIs to motion frame 0, elastic band off, --ramp-seconds 0.
#               Mirrors gear_sonic/scripts/eval_x2_mujoco_onnx.py exactly, so a
#               working C++ deploy MUST hold 30s clean here. Validates the C++
#               obs assembly + action application against the same ground truth
#               Python uses for sim-to-sim eval. Default when --motion is a
#               PKL or YAML (i.e. RSI is even possible).
#   handoff  -- bridge starts at DEFAULT_DOF (matches real-robot MC handoff),
#               soft-start ramp is left at the default (2.0s), elastic band
#               stays on through the ramp + a buffer to proxy the gantry/MC
#               support that holds the real robot upright until policy has
#               full authority. Validates that the deploy can take over from
#               default angles without tipping -- which is the actual
#               sequence on the robot. Recommended as the FINAL sim gate
#               before powered runs.
#   manual   -- no auto-magic; whatever explicit flags the user passes are
#               left alone. Default when --motion is a .x2m2 (legacy passthrough)
#               or omitted entirely.
SIM_PROFILE=""
SIM_DT=""
SIM_PRINT_SCENE=false
SIM_RECORD_COMMANDS=""
# OmniHand augmented MJCF + finger-command ZMQ subscriber. When the live
# VLA bridge is publishing left_hand_joints / right_hand_joints on the
# pose topic, --sim-with-omnihand makes the bridge's MuJoCo viewer load
# the augmented spec and animate the fingers in step with the policy.
# Defaults are off so the existing M2 / M5 acceptance gates (which pin
# the bare X2 MJCF) keep passing without changes.
SIM_WITH_OMNIHAND=false
SIM_HAND_ZMQ_HOST=""
SIM_HAND_ZMQ_PORT=""
SIM_HAND_ZMQ_TOPIC=""
SIM_NO_HAND_ZMQ=false

# Bookkeeping for child PIDs we must clean up on exit
SIM_BRIDGE_PID=""
SIM_RECORD_PID=""

# Hand bridge (REAL-ROBOT only). Republishes ZMQ pose
# left_hand_joints / right_hand_joints onto /aima/hal/joint/hand/command
# (aimdk_msgs/HandCommandArray) so the OmniHand HAL drives finger
# motors. The C++ deploy harness reads these wire fields into
# x2_debug echoes only -- it does NOT publish them to AimDK -- so
# without this bridge real-robot teleop fingers never close even
# though sim does (sim's MuJoCo bridge has its own
# _omnihand_zmq_thread that writes finger qpos directly; this script
# is the real-robot equivalent of that thread).
#
# Only auto-spawned in MODE=local: it's started AFTER MC has been
# verified silent (so its engage burst isn't eaten by mc's own hand
# republish loop) and reaped via the same restart_mc_on_exit trap
# the deploy uses. sim mode uses the existing --with-omnihand path
# instead. onbot users should launch the bridge manually -- the SSH
# topology there has no laptop-side handoff hook to wire it into.
HAND_BRIDGE_ENABLED=true
HAND_BRIDGE_SIDES="auto"
HAND_BRIDGE_ENGAGE_SHOTS=3
HAND_BRIDGE_MAX_STALE_S="0.20"
HAND_BRIDGE_PUBLISH_HZ="50"
HAND_BRIDGE_PYTHON=""
HAND_BRIDGE_SCRIPT_REL="scripts/x2_hand_zmq_to_aimdk_bridge.py"
HAND_BRIDGE_PID=""
HAND_BRIDGE_LOG=""

# Run recorder (--record) -- a sibling background process that subscribes to
# /aima/hal/joint/{leg,waist,arm,head}/{state,command} + the IMU and dumps
# everything to an .npz so we can do post-mortem cmd/state tracking-error
# analysis on real-robot runs without two-terminal coordination. Started at
# launch, stopped via SIGINT in the cleanup traps so the npz is finalised
# even on Ctrl-C / abort.
RECORD_RUN=false
RECORD_OUT=""
RUN_RECORD_PID=""

# Real-deploy tuning config (--tuning-config PATH.yaml). Expanded to
# CLI flags via gear_sonic_deploy/scripts/tuning_config_to_args.py and
# prepended to ROS2_ARGS so explicit per-flag overrides on the command
# line still win. REJECTED in sim modes -- tuning configs are real-robot
# only by design, to keep the C++<->Python parity surface bit-exact.
# See gear_sonic_deploy/configs/real_deploy_tuning/README.md.
TUNING_CONFIG=""

# ============================================================================
# Usage
# ============================================================================

show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [local|onbot|sim]

Deploy the agi_x2_deploy_onnx_ref ROS 2 node onto an AgiBot X2 Ultra.

Modes:
  local           Build + run on this machine; talk to robot via DDS over
                  the wired SDK ethernet (default).
  onbot           Run NATIVELY on PC2 (the robot's onboard Jetson Orin
                  NX). Expects pc2_bringup.sh to have staged the
                  workspace + venv + ONNX Runtime under \$ONBOT_PREFIX
                  (default $ONBOT_PREFIX_DEFAULT). Bash orchestration,
                  MC stop/start, sentinel handoff and the deploy all
                  run on PC2 in the same shell, so the lifecycle
                  survives a laptop-side WiFi disconnect. Recommended
                  for split-topology production runs; pair with
                  scripts/x2_pc2_daemons.sh as the launcher.
  sim             Build + run locally on isolated loopback DDS, paired with
                  the MuJoCo bridge (scripts/x2_mujoco_ros_bridge.py)
                  launched as a background child. Bridge steps physics at
                  1 kHz, publishes joint state + IMU on the same /aima
                  topics the deploy subscribes to, and applies the deploy's
                  PD commands as MuJoCo torques. Closed-loop. Mirrors what
                  G1's sim mode does via unitree_sdk2py_bridge.

Required:
  --model PATH                Fused g1+g1_dyn ONNX (e.g. *_g1.onnx)

Optional deploy flags (forwarded to ros2 run):
  --motion PATH               Reference motion. Accepts .pkl (motion-lib)
                              or .x2m2 (deploy runtime format). A PKL is
                              baked on the fly to a tempdir .x2m2 the deploy
                              binary loads; the same source PKL is auto-passed to the
                              MuJoCo bridge as --sim-motion (in sim mode) so
                              both consumers see bit-identical motion data
                              with no risk of drift. Pass an .x2m2 directly
                              for legacy/already-baked artefacts. Default:
                              StandStillReference (NOT recommended for
                              policies trained on motion).
  --log-dir PATH              Per-tick CSV log directory
  --record                    Background-record the full robot run (state +
                              command + IMU on /aima/hal/*) to an .npz for
                              post-mortem analysis. Recorder starts before
                              the deploy launches, stops on script exit
                              (Ctrl-C-safe: trap finalises the npz). Output
                              defaults to --log-dir/run.npz, or
                              <repo>/scratch/runs/x2_run_<ts>/run.npz when
                              --log-dir is unset (host bind-mounted so it
                              survives container --rm; never /tmp inside
                              docker -- that's the ephemeral writable layer).
                              Use scripts/x2_record_real_run.py --summarize
                              PATH.npz to print the cmd/state/track-err
                              tables.
  --record-out PATH           Implies --record. Override the .npz output path.
  --tuning-config PATH        REAL-ROBOT ONLY (rejected in sim mode). Loads a
                              YAML preset from
                              gear_sonic_deploy/configs/real_deploy_tuning/
                              and expands it to deploy-binary CLI flags
                              (--max-target-dev, --target-lpf-hz, etc.).
                              Explicit flags on this command line still win
                              over the preset (they appear later in argv and
                              the binary's parser is last-write-wins). Sim
                              profiles deliberately refuse this flag so the
                              C++<->Python parity surface against
                              eval_x2_mujoco.py cannot be silently changed.
                              Shipped presets: conservative.yaml,
                              expressive.yaml. See README.md in that folder.
  --autostart-after SECONDS         Auto-transition WAIT->CONTROL after N seconds
                              (default: -1, wait for stdin 'go')
  --max-duration SECONDS      Auto-shutdown N seconds after entering CONTROL
                              (default: unbounded). Useful for bounded dry-run
                              smoke tests so the operator isn't expected to
                              babysit Ctrl-C.
  --dry-run                   Publish stiffness=0/damping=0 (no torque).
                              MANDATORY for first power-on on the real robot.
  --tilt-cos COS              Tilt watchdog threshold (default: -0.3)
  --ramp-seconds SECONDS      Soft-start ramp duration (default: 2.0)
  --max-target-dev RAD        Per-joint hard clamp on |target - default_angles|,
                              in radians. Negative/omitted = disabled. Use a
                              small value (e.g. 0.05 ~= 3 deg) for first
                              powered bring-up runs so a divergent policy or
                              obs-construction bug cannot drive any joint
                              more than RAD away from the trained standing
                              pose, regardless of what the ONNX session emits.
                              Acts as the GLOBAL default for joints with no
                              per-group override below.
  --max-target-dev-leg RAD    Per-group override (MJ joints 0..11 = both
                              hips, knees, ankles). >0 wins over the global
                              --max-target-dev for these joints; omitted =
                              inherit the global. Typical pairing for VR
                              teleop: leg=0.30 (~17 deg), arm=1.50 (~86 deg).
                              Legs need to stay tight: kp ~99 Nm/rad means
                              the same nominal travel produces ~7x the
                              torque arms do.
  --max-target-dev-waist RAD  Per-group override (MJ joints 12..14 = waist
                              yaw/pitch/roll). Same semantics as --leg.
  --max-target-dev-arm RAD    Per-group override (MJ joints 15..28 = both
                              shoulders, elbows, wrists). Set this loose
                              (e.g. 1.50) for teleop where IK can drive
                              the wrist far from default.
  --max-target-dev-head RAD   Per-group override (MJ joints 29..30 = head
                              yaw, pitch). Same semantics as --leg.
  --kp-scale FACTOR           REAL-DEPLOY ONLY (parity-breaking in sim).
                              Multiplicative trim on every joint's KP.
                              Default 1.0 = ship trained kps[] from
                              policy_parameters.hpp as-is. The trained
                              KP is the value IsaacLab used at training;
                              IsaacLab integrates PD implicitly against
                              joint inertia + armature, so the same
                              numerical KP behaves softer at deploy
                              (MuJoCo + real X2 both apply explicit
                              ctrl). Symptom of leaving this at 1.0 on
                              the real robot: stands fine static, wobbles
                              when nudged. Standard fix: per-family bumps
                              below.
  --kp-scale-FAMILY FACTOR    Per-family override; FAMILY is one of
                              hip / knee / ankle / waist / shoulder /
                              elbow / wrist / head (substring match on
                              joint name; mirrors eval_x2_mujoco.py).
                              Effective scale on a joint = global *
                              family. The ankle and waist families have
                              FURTHER subgroup knobs (see below).
  --kp-scale-ankle-pitch FACTOR ankle_pitch_joint ONLY. Trained kp=21.38;
                              1.87 matches MC's 40 N*m/rad sagittal KP.
                              The G16b-validated default for fwd/back
                              recovery.
  --kp-scale-ankle-roll FACTOR ankle_roll_joint ONLY. Trained kp=21.38;
                              1.40 matches MC's 30 N*m/rad frontal KP.
                              Deliberately softer than pitch so sideways
                              disturbances absorb without snapping the
                              foot.
  --kp-scale-waist-yaw FACTOR waist_yaw_joint ONLY. Trained kp=40.18
                              matches MC's published 40 exactly, so the
                              typical value is 1.00 -- bumping is more
                              likely to ring than to help.
  --kp-scale-waist-pr FACTOR  waist_pitch + waist_roll. Trained kp=14.25
                              but MC publishes 40 N*m/rad -> 2.81 matches
                              MC exactly. This is the recommended knob
                              if you're chasing forward/back nudge
                              wobble. Pre-set in expressive.yaml.
  --kd-scale FACTOR           Same as --kp-scale but for damping.
                              Default 1.0 (no trim). Watch the kp/kd
                              ratio: bumping kp without kd lowers the
                              effective damping ratio. If a kp-bumped
                              run rings, try matching kd-scale 1.2 as
                              a first knob before backing off kp.
  --kd-scale-FAMILY FACTOR    Per-family kd override; same FAMILY set
                              as --kp-scale-FAMILY. The ankle and waist
                              families have split subgroup knobs (below).
  --kd-scale-ankle-pitch FACTOR ankle_pitch_joint ONLY. MC publishes
                              kd=3.0 vs trained 0.907 -> 3.31 matches
                              MC. Under-damped pitch is the usual cause
                              of 'foot-feels-springy' on fwd/back
                              ankle-direct nudges.
  --kd-scale-ankle-roll FACTOR ankle_roll_joint ONLY. MC publishes
                              kd=2.0 vs trained 0.907 -> 2.20 matches
                              MC. Less damping than pitch (frontal plane
                              is intrinsically more rigid).
  --kd-scale-waist-yaw FACTOR waist_yaw_joint ONLY. MC publishes kd=8.0
                              vs trained 2.56 -> 3.13 matches MC exactly.
  --kd-scale-waist-pr FACTOR  waist_pitch + waist_roll. MC publishes
                              kd=5.0 vs trained 0.907 -> 5.51 matches MC.
                              This is the SINGLE biggest knob for
                              closing the fwd/back nudge gap on the real
                              robot. Pre-set in expressive.yaml.
  --target-lpf-hz HZ          REAL-DEPLOY ONLY. First-order EMA cutoff (Hz)
                              applied to the published joint targets AFTER the
                              safety stack and BEFORE the bus, to tame jitter
                              caused by noisy real sensor obs. 0 = disabled
                              (default; preserves sim parity). Typically set
                              via --tuning-config; this CLI flag exists so
                              operators can override the preset for quick A/B
                              sweeps. The C++ binary's parser is
                              last-write-wins, so an explicit --target-lpf-hz
                              here always trumps whatever a tuning config set.
  --action-clip RAD           Symmetric clip on the raw ONNX action (action_il)
                              BEFORE x2_action_scale. Default in the C++ binary
                              is 20.0, matching the training-time
                              config.action_clip_value in
                              gear_sonic/config/manager_env/base_env.yaml.
                              Pass a negative value to disable (parity tests
                              only). Without this clip a saturated policy
                              produces O(100 rad) targets which the deploy
                              safety stack truncates -- silently breaking
                              parity with what training observed.
  --return-seconds SECONDS    Soft-EXIT ramp duration (default 2.0). When
                              --max-duration trips, lerp target_pos from the
                              last policy command back to default_angles over
                              SECONDS (deploy-mode kp/kd active) before
                              shutdown. Prevents MC from snapping joints back
                              at handoff (which can red-fault the X2 Ultra
                              MC unit if the policy left limbs mid-motion --
                              e.g. arm extended for take_a_sip). Set 0 to
                              disable (legacy immediate-shutdown).
  --imu-topic NAME            Override IMU topic; use this if the firmware
                              ships with the SDK-example typo
                              (/aima/hal/imu/torse/state)
  --intra-op-threads N        ONNX session threads (default: 1)
  --obs-dump PATH             DEBUG: capture the first CONTROL-tick inference
                              payload (tokenizer + proprioception + raw action
                              + robot state) to PATH and exit. Pair with
                              --dry-run + --autostart-after for a deterministic
                              snapshot from a known robot pose.

Robot connection (onbot mode):
  --robot-host HOST           PC2 IP/hostname (default: $ROBOT_HOST_DEFAULT).
                              IGNORED in the current onbot mode -- onbot
                              runs natively on PC2, so there is no ssh
                              target. Kept for backward compatibility.
  --robot-user USER           PC2 ssh user (default: $ROBOT_USER_DEFAULT).
                              Same: ignored in onbot mode.
  --onbot-prefix DIR          PC2 install prefix (default: $ONBOT_PREFIX_DEFAULT).
                              All other onbot paths default to subdirs of
                              this; pass once to relocate the whole install
                              (matching pc2_bringup.sh --prefix).
  --onbot-ws PATH             colcon workspace on PC2 (default:
                              \$ONBOT_PREFIX/ws). Must contain
                              install/setup.bash built by pc2_bringup.sh.
  --onbot-venv PATH           Python venv on PC2 (default: \$ONBOT_PREFIX/venv).
                              Used for all auxiliary python (tuning YAML
                              translator, MC escalator, preflight). Created
                              with --system-site-packages so it inherits
                              the system rclpy and aimdk_msgs bindings.
  --onbot-onnxruntime PATH    Prebuilt ONNX Runtime install on PC2 (default:
                              \$ONBOT_PREFIX/onnxruntime). lib/ is added to
                              LD_LIBRARY_PATH so the deploy can dlopen
                              libonnxruntime.so.
  --onbot-aimdk-prefix PATH   aimdk_msgs install on PC2 (default:
                              $ONBOT_AIMDK_PREFIX_DEFAULT). cmake config,
                              libs and python bindings live here; added
                              to AMENT_PREFIX_PATH, LD_LIBRARY_PATH and
                              PYTHONPATH at launch.

MC stop / restart (local + onbot modes):
  --mc-em-url URL             PC1 Environment Manager HTTP API URL.
                              We POST {stop,start}_app to {URL}/json/...
                              instead of using ssh. Reachable from inside
                              docker_x2/ as long as the host can route to
                              PC1. Default: http://\$PC1_HOST:\${PC1_EM_PORT:-50080}
                              (or \$X2_MC_EM_URL); empty -> `aima em` CLI only

Sim mode (only applies when 'sim' is selected; all optional):
  --sim-profile {parity,handoff,gantry,gantry-dangle,manual}
                              Two distinct closed-loop sim tests that
                              validate DIFFERENT invariants (BOTH should
                              pass before going to powered hardware):

                                parity  -- bridge RSIs to motion frame 0,
                                           elastic band off, --ramp-seconds 0.
                                           Mirrors eval_x2_mujoco_onnx.py
                                           exactly. A correct C++ deploy
                                           MUST hold 30s clean here. Tests
                                           the C++ obs assembly + action
                                           pipeline against the Python
                                           ground truth. (Default when
                                           --motion is a PKL or YAML.)

                                handoff -- bridge starts at DEFAULT_DOF
                                           (matches what the real robot
                                           looks like at the moment the MC
                                           controller releases the joints),
                                           soft-start ramp at the deploy
                                           default (2.0s), elastic band ON
                                           through the ramp + a buffer to
                                           proxy the gantry support that
                                           keeps the real robot upright
                                           until the policy has full
                                           authority. Tests that the deploy
                                           can take over from default
                                           angles without tipping the
                                           robot, which is the actual
                                           bring-up sequence on hardware.

                                gantry  -- bridge starts in a bent-knee
                                           crouch (~10 cm pelvis drop),
                                           elastic band ON for the entire
                                           run with its pull-target ~3 cm
                                           above the bent pelvis. The band
                                           takes ~85-90 % of body weight,
                                           ~10-15 % goes through the legs
                                           into ground contact. This is
                                           the EXACT pose the operator
                                           holds the X2 in during gantry-
                                           supported powered runs, so sim
                                           and the real Phase 7-9 powered
                                           bring-up test the same operating
                                           point. Recommended as the FINAL
                                           sim gate before powered runs;
                                           closest sim analogue of the
                                           gantry-on-the-real-robot test.

                                manual  -- no auto-magic; whatever explicit
                                           flags the user passes are left
                                           alone. Default when --motion is
                                           a .x2m2 or omitted entirely.

  --sim-mjcf PATH             Override MJCF path (default: x2_ultra.xml).
  --sim-motion PATH           [DEPRECATED] RSI source for the bridge.
                              Prefer just passing --motion <pkl|yaml>; this
                              wrapper now auto-derives the bridge's RSI source
                              from --motion when it's a PKL or YAML. This
                              flag is retained for back-compat / explicit
                              overrides only.
  --sim-init-frame N          Motion frame to RSI from (default 0).
  --sim-viewer                Open the MuJoCo passive viewer window.
  --sim-cam-track-body NAME   Lock the passive viewer's camera onto the named
                              body (e.g. "pelvis") so the robot stays framed
                              as it walks. Default: free camera.
  --sim-cam-distance M        Tracking-camera distance in meters
                              (default: bridge picks 3.5).
  --sim-cam-elevation DEG     Tracking-camera elevation in degrees, negative
                              looks down (default: bridge picks -12).
  --sim-cam-azimuth DEG       Tracking-camera azimuth in degrees
                              (0=+X, 90=+Y, 180=-X, 270=-Y; default 135).
  --sim-with-omnihand         Make the MuJoCo bridge load the X2 + OmniHand
                              augmented MJCF (programmatically composed via
                              gear_sonic.scripts.compose_x2_with_omnihand)
                              instead of the bare x2_ultra.xml. Replaces the
                              dummy wrist stubs with the 10-active-DOF (per
                              side) OmniHand fingers. The body joints +
                              actuators are unchanged so the deploy harness's
                              joint-name handshake still passes.
  --sim-hand-zmq-host HOST    Host the bridge SUBs on for OmniHand finger
                              setpoints when --sim-with-omnihand is set.
                              Default 'localhost' (live VLA bridge).
  --sim-hand-zmq-port PORT    Port for the OmniHand SUB. Default 5556 -- the
                              same port the SONIC C++ harness already
                              subscribes to (PUB/SUB is multi-subscriber-
                              safe so they ride alongside).
  --sim-hand-zmq-topic STR    ZMQ topic prefix for the OmniHand SUB. Must
                              match the live VLA bridge's --pub-topic
                              (default 'pose').
  --sim-no-hand-zmq           Skip the OmniHand ZMQ subscriber even when
                              --sim-with-omnihand is set. Useful for static
                              viewer screenshots: fingers stay at the
                              MJCF rest pose.
  --sim-imu-from {pelvis,torso}
                              Body to read IMU from (default: pelvis,
                              matches MJCF live sensor at imu_0).
  --sim-hold-stiffness-mult X
                              Multiplier on policy_parameters.kps used to
                              hold the default standing pose BEFORE the
                              deploy connects (default: 1.0).
  --sim-init-pose {default,gantry-hang}
                              Bridge init pose when --motion is NOT set.
                              'default' = upright stand; 'gantry-hang' =
                              bent-knee crouch with pelvis_z=0.75m. Auto-
                              set by --sim-profile gantry; pass directly
                              for ad-hoc experimentation.
  --sim-band-length M         ElasticBand suspension length (m). Pull-target
                              sits at world [0, 0, 1.0 - LENGTH]. Default 0
                              (target above standing pelvis). Auto-set to
                              0.22 by --sim-profile gantry (target ~3 cm
                              above the bent pelvis at z=0.75).
  --sim-band-kp-mult X        Multiplier on the ElasticBand's kp_pos /
                              kd_pos. 1.0 = stiff (10000 / 1000); lower
                              values (0.3-0.5) make the suspension softer
                              / more forgiving. kd is scaled by sqrt(mult)
                              to preserve the damping ratio.
  --sim-no-elastic-band       Disable the virtual ElasticBand. The band is
                              ON by default and hangs the pelvis from world
                              [0,0,1] so the robot stays upright while the
                              policy spins up. With viewer, press 9 to drop;
                              headless, see --sim-band-release-after-s.
                              NOTE: when --motion resolves to a PKL/YAML
                              (i.e. the bridge can RSI from frame 0), this
                              wrapper auto-disables the band so the robot
                              spawns in stable contact equilibrium instead
                              of being suspended-then-dropped (the latter
                              produces a transient the policy was never
                              trained on; see x2_deploy_architecture.md).
                              Use --sim-keep-elastic-band to override.
  --sim-keep-elastic-band     Force the elastic band ON even when --motion
                              would otherwise auto-disable it. Useful for
                              testing the suspended-then-dropped scenario
                              explicitly.
  --sim-band-release-after-s SECS
                              When headless, auto-release the band SECS
                              seconds after the deploy's first command.
                              Default 1.0. Negative = never (unsafe).
  --sim-dt SECS               Physics step (default: 0.001 = 1 kHz).
  --sim-print-scene           Dump bodies/joints/actuators/sensors on start.
  --sim-python PATH           Python interpreter for the bridge
                              (default: repo .venv/bin/python when that
                              file is executable, else python3).
  --sim-record-commands PATH  Record the deploy's command topics to this
                              rosbag2 directory (handy for diffing sim/real).
  --sim-domain-id N           ROS_DOMAIN_ID to isolate the sim from any real
                              robot on the same subnet (default: $SIM_DOMAIN_ID_DEFAULT)

VLA / ZMQ input source (M2 acceptance gate, see
docs/source/references/x2_zmq_protocol.md):
  --vla                       Switch the deploy from "motion file replay" to
                              "ZMQ pose subscriber". Ignores --motion. Pair
                              with the kplanner/pose-watchdog chain (PC2
                              ritual) or the laptop planner stack; any
                              publisher of the x2_zmq_protocol pose frame works.
  --vla-zmq-host HOST         Host of the VLA ZMQ pose publisher
                              (default: $VLA_ZMQ_HOST).
  --vla-zmq-port PORT         Port of the VLA ZMQ pose publisher
                              (default: $VLA_ZMQ_PORT).
  --vla-zmq-topic TOPIC       Topic prefix to subscribe to (default: $VLA_ZMQ_TOPIC).
  --vla-debug-port PORT       Port for the deploy's x2_debug PUB socket
                              (default: $VLA_DEBUG_PORT). Set 0 to disable.
                              dump_x2_debug.py is the reference subscriber.
  --vla-resume-host HOST      Host of the SAFE_IDLE resume PUB (the laptop
                              manager publishes when the Quest 3 operator
                              hits the A+B chord). Empty = leave the
                              resume socket disabled.
  --vla-resume-port PORT      Port for the resume PUB (default: $VLA_RESUME_PORT).
  --vla-resume-topic TOPIC    Topic prefix for resume frames
                              (default: $VLA_RESUME_TOPIC).
  --deploy-extra-arg ARG      Forward ARG verbatim to the C++ deploy binary.
                              Repeat for multiple args. Useful for new C++
                              CliArgs flags before they get a dedicated
                              wrapper flag here.
  --vla-debug-topic TOPIC     Topic prefix for x2_debug frames
                              (default: $VLA_DEBUG_TOPIC).
  --wrist-bypass MODE         {off, ik}. Forwarded to the deploy binary as
                              --wrist-bypass. When 'ik' (and --vla is set),
                              the deploy overwrites target_pos_mj for the
                              4 broken wrist DOFs (left/right wrist_pitch +
                              wrist_roll) with the latest IK reference from
                              the ZMQ pose feed BEFORE the safety stack.
                              SONIC still drives every other DOF including
                              wrist_yaw (which tracks correctly today).
                              Use 'ik' for VR teleop / VLA dataset recording
                              where SONIC's wrist attractor masks the
                              operator's hand pose; keep unset / 'off' for
                              sim-to-real fidelity tests. Default: empty
                              (= binary default of 'off').
  --head-bypass MODE          {off, ref}. Forwarded to the deploy binary as
                              --head-bypass. When 'ref' (ZMQ pose feed
                              required), the deploy overwrites target_pos_mj
                              for the 2 head DOFs (head_yaw/head_pitch) with
                              the ZMQ reference BEFORE the safety stack.
                              SONIC's head tracking saturates at ~+-10 deg;
                              use 'ref' so the pad head-look overlay reaches
                              the joint. Default: empty (= binary 'off').

OmniHand bridge (REAL ROBOT, MODE=local only -- sim uses --with-omnihand instead):
  --no-hand-bridge            Skip auto-spawning the ZMQ -> AimDK
                              HandCommandArray bridge. The C++ deploy
                              never publishes /aima/hal/joint/hand/command
                              on its own, so without the bridge real-robot
                              teleop cannot close fingers (wire fields land
                              in x2_debug only). Default: bridge is ON.
                              Pass this for arm/leg-only sessions where
                              the hands aren't connected.
  --hand-bridge-sides STR     {auto,left,right,both,off}. 'auto' detects
                              attached sides from the latched HandStateArray
                              on /aima/hal/joint/hand/state. (default: auto)
  --hand-bridge-engage-shots N
                              Number of position=0 shots to fire at startup
                              so the OmniHand HAL exits its 'no command yet'
                              state and enables motors. Set 0 only when
                              piggy-backing on an already-engaged HAL.
                              (default: 3, 1Hz spacing)
  --hand-bridge-max-stale-s S Wire frame is considered stale after this
                              many seconds with no update. Stale -> the
                              bridge republishes last-good positions
                              instead of dribbling zeros. (default: 0.20s)
  --hand-bridge-publish-hz HZ Publish loop rate (default: 50, matches the
                              deploy CONTROL tick).
  --hand-bridge-python PATH   Python interpreter to launch the bridge with.
                              Default: same as other auxiliary Python in
                              this script (repo .venv/bin/python when that
                              file is executable, otherwise python3; in
                              docker_x2 the image python3 carries rclpy).

Pre-flight + behaviour toggles:
  --no-wire-probe             Skip the pre-MC-stop ZMQ wire-freshness probe.
                              The default probe SUBs to the deploy's actual
                              --vla-zmq-host/-port for --wire-probe-secs and
                              aborts if zero frames arrive (catches dead
                              pose proxy / missing planner stack BEFORE MC
                              is stopped). Disable for unattended CI / when
                              intentionally testing starvation.
  --wire-probe-secs SECONDS   How long the wire probe listens (default: 1.0).
  --no-publish                OBSERVE-ONLY. The deploy node runs the policy and
                              writes every CSV, but NEVER publishes to the motor
                              bus in any state. FORCES --no-stop-mc, so MC keeps
                              the robot throughout. Use this to capture what the
                              policy WOULD command, with zero risk. NOT the same
                              as --dry-run, which still publishes (kp=0/kd=0) and
                              would fight MC for the bus.
  --no-stop-mc                Skip the stop_app POST (assume MC is already
                              stopped, or you're using JOINT_DEFAULT mode).
                              The cleanup trap that restarts MC on exit is
                              also skipped in that case.
  --no-require-stand-default  Bypass the pre-handoff gate that refuses to
                              start unless MC is currently in STAND_DEFAULT.
                              Use only if you know what you're doing -- the
                              smooth handoff (RAMP_OUT to STAND_DEFAULT pose
                              + HOLD_FOR_MC) assumes MC was actively
                              balancing the robot before bash takes the bus.
  --stand-pose-yaml PATH      YAML file capturing MC's STAND_DEFAULT pose
                              (default: configs/x2_stand_default_pose.yaml).
                              Forwarded to the deploy binary as
                              --stand-default-pose; controls the RAMP_OUT
                              + HOLD_FOR_MC target. Pass an empty string
                              to fall back to default_angles (legacy snap-
                              on-takeover behaviour).
  --hold-for-mc-timeout-s N   Maximum time the deploy node will keep
                              holding MC's STAND_DEFAULT pose after
                              RAMP_OUT, waiting for MC to take back over
                              the joint command bus (default 45). 0
                              disables HOLD_FOR_MC entirely (legacy:
                              deploy exits on RAMP_OUT, then bash starts
                              MC -- there will be a zero-torque window).
  --soft-shutdown-wait-s N    (Only relevant with --enable-soft-shutdown.)
                              Seconds the bash cleanup trap waits for
                              deploy to reach HOLD_FOR_MC after Ctrl-C
                              before falling through and POSTing
                              start_app anyway (default 6). Should be
                              larger than --return-seconds (RAMP_OUT
                              duration) plus a small DDS-discovery
                              margin. The wait is interruptible: a
                              second Ctrl-C bails immediately to the
                              legacy hard-exit path.
  --enable-soft-shutdown      OPT IN to the graceful Ctrl-C / RAMP_OUT
                              path. DISABLED BY DEFAULT as of
                              2026-05-15: field testing showed the
                              graceful path causes 4-6 s of motor whir
                              on Ctrl-C and the robot still collapses
                              at the end (because MC restarts in
                              PASSIVE_DEFAULT regardless). The hard-
                              exit path is shorter in total motor
                              activity and therefore safer for the
                              actuators. Use this flag only when
                              actively developing the soft-shutdown
                              handoff.
  --no-soft-shutdown          Legacy no-op (kept for backward compat).
                              Soft-shutdown is now off by default.
  --no-confirm                Skip the final "proceed?" prompt (for CI)
  --no-build                  Skip the colcon build step (use the existing
                              install/ tree as-is)
  --build-only                Build, don't run
  --no-docker                 Do NOT auto-relaunch inside the docker_x2/x2sim
                              container. Default behaviour is: if
                              /workspace/sonic doesn't exist (i.e. you're on
                              a host shell), the script re-execs itself
                              inside the container with \$HOME mounted at
                              \$HOME so all your paths just work. Pass this
                              flag if you've already sourced ROS 2 +
                              aimdk_msgs on the host and want to run
                              natively. Also honoured via the
                              X2_DEPLOY_IN_DOCKER=1 env var (set
                              automatically by the auto-relaunch).
  --no-preflight-py           Skip the gantry-aware Python preflight
                              (gear_sonic_deploy/scripts/x2_preflight.py).
                              Default: preflight runs in local/onbot mode
                              and is force-skipped in sim. Failures abort
                              BEFORE MC stop, so the robot stays held.
  --preflight-strict          Run the Python preflight with --strict-pose
                              --strict-effort (promotes pose/effort WARNs
                              to FAIL). Recommended for floor-stand powered
                              runs; leave off for gantry bring-up.
  --preflight-args "..."      Extra args appended verbatim to the Python
                              preflight invocation (e.g.
                              "--max-effort 10 --imu-tilt-deg 12").

ONNX Runtime:
  --onnxruntime-root PATH     ORT install prefix (default: $ONNXRUNTIME_ROOT_DEFAULT)

  -h, --help                  Show this help

Examples:
  # Bring-up dry run from your laptop (most common first command):
  $0 --model /opt/x2_models/model_step_016000_g1.onnx --dry-run --autostart-after 5

  # On-bot powered run (split-topology). Must be invoked ON PC2 after
  # pc2_bringup.sh has staged the workspace. Typically launched via
  # scripts/x2_pc2_daemons.sh from the laptop; this is what ends up
  # inside the x2_deploy tmux session:
  $0 onbot \\
      --vla --vla-zmq-host \${LAPTOP_HOST} --vla-zmq-port 5556 \\
      --model /home/run/gear-sonic/policies/agibot_x2_sonic.onnx \\
      --tuning-config /tmp/expressive.yaml --wrist-bypass ik \\
      --log-dir /home/run/gear-sonic/log/x2_powered_\$(date +%Y%m%d_%H%M%S)

  # Closed-loop sim in MuJoCo with viewer (no robot needed):
  $0 sim \\
      --model /opt/x2_models/model_step_016000_g1.onnx \\
      --sim-viewer --autostart-after 5

  # Just rebuild, no robot, no sim:
  $0 --model dummy --build-only --no-stop-mc local

For full background, see:
  docs/source/user_guide/x2_sonic_deploy_real.md
  docs/source/user_guide/x2_first_real_robot.md
  docs/source/references/x2_deployment_code.md
EOF
}

# ============================================================================
# Parse arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_usage; exit 0 ;;
        --model)              MODEL="$2"; shift 2 ;;
        --motion)             MOTION="$2"; shift 2 ;;
        --log-dir)            LOG_DIR="$2"; shift 2 ;;
        --record)             RECORD_RUN=true; shift ;;
        --record-out)         RECORD_RUN=true; RECORD_OUT="$2"; shift 2 ;;
        --tuning-config)      TUNING_CONFIG="$2"; shift 2 ;;
        --autostart-after)          AUTOSTART="$2"; shift 2 ;;
        --max-duration)       MAX_DURATION="$2"; shift 2 ;;
        --tilt-cos)           TILT_COS="$2"; shift 2 ;;
        --ramp-seconds)       RAMP_SECONDS="$2"; shift 2 ;;
        --max-target-dev)     MAX_TARGET_DEV="$2"; shift 2 ;;
        --max-target-dev-leg)   MAX_TARGET_DEV_LEG="$2"; shift 2 ;;
        --max-target-dev-waist) MAX_TARGET_DEV_WAIST="$2"; shift 2 ;;
        --max-target-dev-waist-pitch) MAX_TARGET_DEV_WAIST_PITCH="$2"; shift 2 ;;
        --max-target-dev-arm)   MAX_TARGET_DEV_ARM="$2"; shift 2 ;;
        --max-target-dev-head)  MAX_TARGET_DEV_HEAD="$2"; shift 2 ;;
        --kp-scale)             KP_SCALE="$2"; shift 2 ;;
        --kp-scale-hip)         KP_SCALE_HIP="$2"; shift 2 ;;
        --kp-scale-knee)        KP_SCALE_KNEE="$2"; shift 2 ;;
        --kp-scale-ankle)       KP_SCALE_ANKLE="$2"; shift 2 ;;
        --kp-scale-ankle-pitch) KP_SCALE_ANKLE_PITCH="$2"; shift 2 ;;
        --kp-scale-ankle-roll)  KP_SCALE_ANKLE_ROLL="$2"; shift 2 ;;
        --kp-scale-waist)       KP_SCALE_WAIST="$2"; shift 2 ;;
        --kp-scale-waist-yaw)   KP_SCALE_WAIST_YAW="$2"; shift 2 ;;
        --kp-scale-waist-pr)    KP_SCALE_WAIST_PR="$2"; shift 2 ;;
        --kp-scale-shoulder)    KP_SCALE_SHOULDER="$2"; shift 2 ;;
        --kp-scale-elbow)       KP_SCALE_ELBOW="$2"; shift 2 ;;
        --kp-scale-wrist)       KP_SCALE_WRIST="$2"; shift 2 ;;
        --kp-scale-head)        KP_SCALE_HEAD="$2"; shift 2 ;;
        --kd-scale)             KD_SCALE="$2"; shift 2 ;;
        --kd-scale-hip)         KD_SCALE_HIP="$2"; shift 2 ;;
        --kd-scale-knee)        KD_SCALE_KNEE="$2"; shift 2 ;;
        --kd-scale-ankle)       KD_SCALE_ANKLE="$2"; shift 2 ;;
        --kd-scale-ankle-pitch) KD_SCALE_ANKLE_PITCH="$2"; shift 2 ;;
        --kd-scale-ankle-roll)  KD_SCALE_ANKLE_ROLL="$2"; shift 2 ;;
        --kd-scale-waist)       KD_SCALE_WAIST="$2"; shift 2 ;;
        --kd-scale-waist-yaw)   KD_SCALE_WAIST_YAW="$2"; shift 2 ;;
        --kd-scale-waist-pr)    KD_SCALE_WAIST_PR="$2"; shift 2 ;;
        --kd-scale-shoulder)    KD_SCALE_SHOULDER="$2"; shift 2 ;;
        --kd-scale-elbow)       KD_SCALE_ELBOW="$2"; shift 2 ;;
        --kd-scale-wrist)       KD_SCALE_WRIST="$2"; shift 2 ;;
        --kd-scale-head)        KD_SCALE_HEAD="$2"; shift 2 ;;
        --target-lpf-hz)      TARGET_LPF_HZ="$2"; shift 2 ;;
        --action-clip)        ACTION_CLIP="$2"; shift 2 ;;
        --return-seconds)     RETURN_SECONDS="$2"; shift 2 ;;
        --stand-pose-yaml)    STAND_POSE_YAML="$2"; shift 2 ;;
        --hold-for-mc-timeout-s) HOLD_FOR_MC_TIMEOUT_S="$2"; shift 2 ;;
        --soft-shutdown-wait-s) SOFT_SHUTDOWN_WAIT_S="$2"; shift 2 ;;
        --no-soft-shutdown)
            # Legacy no-op: soft-shutdown is now DISABLED by default
            # (see SOFT_SHUTDOWN_DISABLED definition above). Kept so
            # existing CI / wrapper scripts that already pass this
            # flag don't break.
            SOFT_SHUTDOWN_DISABLED=true
            shift ;;
        --enable-soft-shutdown)
            # Opt back IN to the graceful Ctrl-C / RAMP_OUT path. Use
            # only when actively developing / debugging the soft-shutdown
            # handoff -- field-tested behaviour as of 2026-05-15 is
            # WORSE than hard-exit (4-6 s motor whir + still collapses
            # on MC PASSIVE_DEFAULT boot). See deploy_x2.sh head comment.
            SOFT_SHUTDOWN_DISABLED=false
            shift ;;
        --no-require-stand-default) REQUIRE_STAND_DEFAULT=false; shift ;;
        --imu-topic)          IMU_TOPIC="$2"; shift 2 ;;
        --intra-op-threads)   INTRA_OP_THREADS="$2"; shift 2 ;;
        --obs-dump)           OBS_DUMP="$2"; shift 2 ;;
        --dry-run)            DRY_RUN=true; shift ;;
        --robot-host)         ROBOT_HOST="$2"; shift 2 ;;
        --robot-user)         ROBOT_USER="$2"; shift 2 ;;
        --onbot-prefix)
            ONBOT_PREFIX="$2"
            # Re-derive any unset-by-flag downstream paths so --onbot-prefix
            # actually relocates the whole install in one shot, matching the
            # pc2_bringup.sh / x2_pc2_daemons.sh --prefix behaviour. Anything
            # the user passed explicitly via --onbot-ws / --onbot-venv /
            # --onbot-onnxruntime survives because those CLI flags come
            # after the case branch consumes them; we only override the
            # default-derived ones here.
            ONBOT_WS="${ONBOT_PREFIX}/ws"
            ONBOT_VENV="${ONBOT_PREFIX}/venv"
            ONBOT_ONNXRUNTIME="${ONBOT_PREFIX}/onnxruntime"
            shift 2 ;;
        --onbot-ws)           ONBOT_WS="$2"; shift 2 ;;
        --onbot-venv)         ONBOT_VENV="$2"; shift 2 ;;
        --onbot-onnxruntime)  ONBOT_ONNXRUNTIME="$2"; shift 2 ;;
        --onbot-aimdk-prefix) ONBOT_AIMDK_PREFIX="$2"; shift 2 ;;
        --onnxruntime-root)   ONNXRUNTIME_ROOT="$2"; shift 2 ;;
        --mc-em-url)          MC_EM_URL="$2"; shift 2 ;;
        --no-stop-mc)         NO_STOP_MC=true; shift ;;
        # OBSERVE-ONLY. Forces NO_STOP_MC=true unconditionally -- this is an
        # interlock, not a default, so no flag ordering or combination can
        # produce "deploy muted AND MC stopped" (= nothing commanding the
        # joints = zero torque = the robot drops).
        --no-publish)         NO_PUBLISH=true; NO_STOP_MC=true; shift ;;
        --no-confirm)         NO_CONFIRM=true; shift ;;
        --no-wire-probe)      NO_WIRE_PROBE=true; shift ;;
        --wire-probe-secs)    WIRE_PROBE_SECS="$2"; shift 2 ;;
        --no-build)           NO_BUILD=true; shift ;;
        --build-only)         BUILD_ONLY=true; shift ;;
        --no-preflight-py)    NO_PREFLIGHT_PY=true; shift ;;
        --preflight-strict)   PREFLIGHT_STRICT=true; shift ;;
        --preflight-args)     PREFLIGHT_ARGS="$2"; shift 2 ;;
        --no-docker)          shift ;;  # consumed by maybe_relaunch_in_docker
        --sim-mjcf)               SIM_MJCF="$2"; shift 2 ;;
        --sim-motion)             SIM_MOTION="$2"; shift 2 ;;
        --sim-init-frame)         SIM_INIT_FRAME="$2"; shift 2 ;;
        --sim-viewer)             SIM_VIEWER=true; shift ;;
      --sim-cam-track-body)     SIM_CAM_TRACK_BODY="$2"; shift 2 ;;
      --sim-cam-distance)       SIM_CAM_DISTANCE="$2"; shift 2 ;;
      --sim-cam-elevation)      SIM_CAM_ELEVATION="$2"; shift 2 ;;
      --sim-cam-azimuth)        SIM_CAM_AZIMUTH="$2"; shift 2 ;;
        --sim-imu-from)           SIM_IMU_FROM="$2"; shift 2 ;;
        --sim-hold-stiffness-mult) SIM_HOLD_STIFFNESS_MULT="$2"; shift 2 ;;
        --sim-init-pose)          SIM_INIT_POSE="$2"; shift 2 ;;
        --sim-band-length)        SIM_BAND_LENGTH="$2"; shift 2 ;;
        --sim-band-kp-mult)       SIM_BAND_KP_MULT="$2"; shift 2 ;;
        --sim-no-elastic-band)    SIM_NO_ELASTIC_BAND=true; shift ;;
        --sim-keep-elastic-band)  SIM_KEEP_ELASTIC_BAND=true; shift ;;
        --sim-band-release-after-s) SIM_BAND_RELEASE_AFTER_S="$2"; shift 2 ;;
        --sim-profile)            SIM_PROFILE="$2"; shift 2 ;;
        --sim-dt)                 SIM_DT="$2"; shift 2 ;;
        --sim-print-scene)        SIM_PRINT_SCENE=true; shift ;;
        --sim-python)             SIM_PYTHON="$2"; shift 2 ;;
        --sim-record-commands)    SIM_RECORD_COMMANDS="$2"; shift 2 ;;
        --sim-domain-id)          SIM_DOMAIN_ID="$2"; shift 2 ;;
        --sim-with-omnihand)      SIM_WITH_OMNIHAND=true; shift ;;
        --sim-hand-zmq-host)      SIM_HAND_ZMQ_HOST="$2"; shift 2 ;;
        --sim-hand-zmq-port)      SIM_HAND_ZMQ_PORT="$2"; shift 2 ;;
        --sim-hand-zmq-topic)     SIM_HAND_ZMQ_TOPIC="$2"; shift 2 ;;
        --sim-no-hand-zmq)        SIM_NO_HAND_ZMQ=true; shift ;;
        --vla)                VLA_MODE=true; shift ;;
        --vla-zmq-host)       VLA_ZMQ_HOST="$2"; shift 2 ;;
        --vla-zmq-port)       VLA_ZMQ_PORT="$2"; shift 2 ;;
        --vla-zmq-topic)      VLA_ZMQ_TOPIC="$2"; shift 2 ;;
        --vla-debug-port)     VLA_DEBUG_PORT="$2"; shift 2 ;;
        --vla-debug-topic)    VLA_DEBUG_TOPIC="$2"; shift 2 ;;
        --vla-resume-host)    VLA_RESUME_HOST="$2"; shift 2 ;;
        --vla-resume-port)    VLA_RESUME_PORT="$2"; shift 2 ;;
        --vla-resume-topic)   VLA_RESUME_TOPIC="$2"; shift 2 ;;
        --deploy-extra-arg)   DEPLOY_EXTRA_ARGS+=("$2"); shift 2 ;;
        --wrist-bypass)       WRIST_BYPASS="$2"; shift 2 ;;
        --head-bypass)        HEAD_BYPASS="$2"; shift 2 ;;
        --no-hand-bridge)             HAND_BRIDGE_ENABLED=false; shift ;;
        --hand-bridge-sides)          HAND_BRIDGE_SIDES="$2"; shift 2 ;;
        --hand-bridge-engage-shots)   HAND_BRIDGE_ENGAGE_SHOTS="$2"; shift 2 ;;
        --hand-bridge-max-stale-s)    HAND_BRIDGE_MAX_STALE_S="$2"; shift 2 ;;
        --hand-bridge-publish-hz)     HAND_BRIDGE_PUBLISH_HZ="$2"; shift 2 ;;
        --hand-bridge-python)         HAND_BRIDGE_PYTHON="$2"; shift 2 ;;
        local|onbot|sim)      MODE="$1"; shift ;;
        *)
            echo -e "${RED}Error: unknown argument: $1${NC}" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

if [[ -z "$MODEL" ]]; then
    echo -e "${RED}Error: --model is required${NC}" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

if [[ "$MODE" != "local" && "$MODE" != "onbot" && "$MODE" != "sim" ]]; then
    echo -e "${RED}Error: mode must be one of: local, onbot, sim (got '$MODE')${NC}" >&2
    exit 1
fi

# ============================================================================
# Onbot runtime env (hoisted to run BEFORE preflight)
# ----------------------------------------------------------------------------
# The preflight phase shells out to x2_preflight.py (which imports rclpy)
# and probes `ros2 topic list`, the tuning-config translator runs another
# python helper, and the launch dispatch sources the colcon overlay. In
# `local` mode all of that just inherits the user's shell env (laptop has
# ROS sourced via docker_x2 or .bashrc, and DEPLOY_AUX_PYTHON points at
# the repo .venv). In `onbot` mode we are on PC2 in a vanilla shell, so
# we have to set up the env OURSELVES before the first phase that needs
# it -- otherwise preflight aborts with "ModuleNotFoundError: rclpy" /
# "ros2 not in PATH" and we never get to the launch dispatch where the
# env used to be sourced.
#
# Previously this block lived inside the launch dispatch (way too late);
# moving it here means preflight, tuning translator AND the launch all
# see a fully-sourced PC2 env. Layout matches pc2_bringup.sh + the
# x2_pc2_daemons.sh env_prelude so all three stay byte-identical.
if [[ "$MODE" == "onbot" ]]; then
    if [[ ! -f "$ONBOT_WS/install/setup.bash" ]]; then
        echo -e "${RED}Error: onbot workspace missing at $ONBOT_WS/install/setup.bash${NC}" >&2
        echo -e "${YELLOW}       Run pc2_bringup.sh from the laptop first:${NC}" >&2
        echo -e "${YELLOW}         ./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host <PC2_IP>${NC}" >&2
        exit 1
    fi
    ros_setup="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
    if [[ ! -f "$ros_setup" ]]; then
        echo -e "${RED}Error: ROS 2 setup.bash missing at $ros_setup${NC}" >&2
        echo -e "${YELLOW}       deploy_x2.sh onbot requires ROS 2 ${ROS_DISTRO:-humble} on PC2.${NC}" >&2
        exit 1
    fi
    # aimdk_msgs first so its cmake/lib show up before ros2 sets up the
    # rest of the chain.
    export AMENT_PREFIX_PATH="${ONBOT_AIMDK_PREFIX}:${AMENT_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="${ONBOT_AIMDK_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    # shellcheck disable=SC1090
    source "$ros_setup"
    # shellcheck disable=SC1091
    source "$ONBOT_WS/install/setup.bash"
    export LD_LIBRARY_PATH="${ONBOT_ONNXRUNTIME}/lib:${LD_LIBRARY_PATH:-}"
    # PYTHONPATH so the venv python sees aimdk_msgs (no env hook ships
    # it; it lives at <aimdk_prefix>/local/lib/python3.10/dist-packages)
    # AND so SCRIPT_DIR-relative imports inside x2_preflight.py /
    # tuning_config_to_args.py resolve.
    export PYTHONPATH="${ONBOT_AIMDK_PREFIX}/local/lib/python3.10/dist-packages:${ONBOT_WS}/src:${PYTHONPATH:-}"
    # Re-point every python invocation in this script at the bringup
    # venv. CRITICAL: SIM_PYTHON was captured from DEPLOY_AUX_PYTHON
    # near the top of the script (long before MODE was parsed) and is
    # what the preflight + sim-bridge launches actually use. Update
    # BOTH so preflight no longer falls back to the system python3
    # (which has no rclpy / no pyzmq / no aimdk_msgs).
    if [[ -x "$ONBOT_VENV/bin/python3" ]]; then
        DEPLOY_AUX_PYTHON="$ONBOT_VENV/bin/python3"
        SIM_PYTHON="$ONBOT_VENV/bin/python3"
    fi
    echo -e "${BLUE}[onbot]${NC} runtime env ready"
    echo -e "${BLUE}[onbot]${NC}   ros:        $ros_setup"
    echo -e "${BLUE}[onbot]${NC}   overlay:    $ONBOT_WS/install/setup.bash"
    echo -e "${BLUE}[onbot]${NC}   aimdk:      $ONBOT_AIMDK_PREFIX"
    echo -e "${BLUE}[onbot]${NC}   onnxruntime: $ONBOT_ONNXRUNTIME"
    echo -e "${BLUE}[onbot]${NC}   aux python: $DEPLOY_AUX_PYTHON"
    unset ros_setup
fi

# Resolve absolute paths for local artefacts. We resolve relative paths
# against $USER_CWD (the directory the user ran us from, captured BEFORE
# we cd into SCRIPT_DIR) -- this matches how every other CLI tool in the
# world treats paths. Failing the resolution loudly (rather than silently
# producing /$basename) catches typos before they turn into "file not
# found" deep inside a phase that already started.
abspath() {
    local p="$1"
    if [[ -z "$p" ]]; then
        echo ""
        return 0
    elif [[ "$p" = /* ]]; then
        echo "$p"
        return 0
    fi
    local dir
    dir="$(dirname "$p")"
    # First try as-is from USER_CWD (the operator's original directory).
    if [[ -d "$USER_CWD/$dir" ]]; then
        echo "$(cd "$USER_CWD/$dir" && pwd)/$(basename "$p")"
        return 0
    fi
    # Fall back to SCRIPT_DIR-relative for back-compat with the old
    # behaviour (some doc invocations assume cwd == gear_sonic_deploy/).
    if [[ -d "$SCRIPT_DIR/$dir" ]]; then
        echo "$(cd "$SCRIPT_DIR/$dir" && pwd)/$(basename "$p")"
        return 0
    fi
    # Neither directory exists -- emit a non-empty marker so downstream
    # validation (-f checks) fail with a helpful path instead of swallowing
    # the input. Prefix with USER_CWD/ so the error message points at where
    # we looked.
    echo "$USER_CWD/$p"
}

# Returns 0 if the (already-absolute) path lives on a host bind-mount that
# survives container exit; 1 otherwise. Outside docker, every path is
# persistent. Inside docker (X2_DEPLOY_IN_DOCKER=1), only the bind-mounts
# established by docker_x2/docker-compose.yml + maybe_relaunch_in_docker
# survive --rm:
#   /workspace/sonic   <- repo root (../..)
#   $HOME / $X2_HOST_HOME <- operator home (deploy_x2.sh -v; host path carried in as X2_HOST_HOME)
# /tmp, /var/tmp, /root/anything-else, etc. are reaped on container exit.
is_host_persistent_path() {
    local p="$1"
    [[ "${X2_DEPLOY_IN_DOCKER:-0}" != "1" ]] && return 0
    [[ -z "$p" ]] && return 1
    # Inside the container HOME is /root, but the operator's host home is
    # bind-mounted at its own path (-v "$HOME:$HOME"); X2_HOST_HOME carries it in.
    local hh="${X2_HOST_HOME:-}"
    case "$p" in
        /workspace/sonic|/workspace/sonic/*) return 0 ;;
        "$HOME"|"$HOME"/*)                   return 0 ;;
        *)
            if [[ -n "$hh" && ( "$p" == "$hh" || "$p" == "$hh"/* ) ]]; then return 0; fi
            return 1 ;;
    esac
}

# Aborts with a helpful error if $2 is on the container's ephemeral
# writable layer. $1 is a human-readable label for the offending knob.
assert_host_persistent_path() {
    local label="$1"
    local p="$2"
    if is_host_persistent_path "$p"; then
        return 0
    fi
    echo -e "${RED}ERROR: $label resolves to '$p' inside the docker container.${NC}" >&2
    echo -e "${RED}       That path is on the container's ephemeral writable layer${NC}" >&2
    echo -e "${RED}       (--rm reaps it on exit), so the recording / CSVs would be${NC}" >&2
    echo -e "${RED}       lost the moment the run finishes.${NC}" >&2
    echo "" >&2
    echo -e "${YELLOW}       Host bind-mounts that DO persist inside this container:${NC}" >&2
    echo -e "${YELLOW}         /workspace/sonic/...   (repo root, recommended for run logs)${NC}" >&2
    echo -e "${YELLOW}         $HOME/...              (operator home)${NC}" >&2
    echo "" >&2
    echo -e "${YELLOW}       Re-run with e.g.:${NC}" >&2
    echo -e "${YELLOW}         --log-dir scratch/runs/x2_run_\$(date +%Y%m%d_%H%M%S)${NC}" >&2
    echo -e "${YELLOW}         --record-out scratch/runs/my_run/run.npz${NC}" >&2
    exit 1
}

if [[ "$MODE" == "local" || "$MODE" == "sim" ]]; then
    [[ -n "$MODEL" ]] && MODEL="$(abspath "$MODEL")"
    [[ -n "$MOTION" ]] && MOTION="$(abspath "$MOTION")"
    [[ -n "$LOG_DIR" ]] && LOG_DIR="$(abspath "$LOG_DIR")"

    # Default --record output: alongside the per-tick CSVs in --log-dir if
    # set, otherwise an auto-stamped dir under the bind-mounted repo so
    # the .npz survives `--rm` container teardown. Only computed if the
    # operator asked for --record but didn't override via --record-out.
    # NOTE: we deliberately do NOT default to /tmp -- that's the trap that
    # cost us the iter-16k real run on 2026-05-02. See
    # is_host_persistent_path() above.
    if $RECORD_RUN && [[ -z "$RECORD_OUT" ]]; then
        if [[ -n "$LOG_DIR" ]]; then
            RECORD_OUT="$LOG_DIR/run.npz"
        else
            RECORD_OUT="$(cd "$SCRIPT_DIR/.." && pwd)/scratch/runs/x2_run_$(date +%Y%m%d_%H%M%S)/run.npz"
        fi
    fi
    [[ -n "$RECORD_OUT" ]] && RECORD_OUT="$(abspath "$RECORD_OUT")"

    # Refuse to run if --log-dir / --record-out would land on the
    # container's ephemeral layer. Better to fail before the docker
    # spin-up than to discover afterwards that the run is lost.
    [[ -n "$LOG_DIR" ]]    && assert_host_persistent_path "--log-dir"    "$LOG_DIR"
    [[ -n "$RECORD_OUT" ]] && assert_host_persistent_path "--record-out" "$RECORD_OUT"
fi
# onbot is now PC2-native (no rsync), so MOTION resolution is just
# abspath against the current cwd -- matches local mode behavior.
if [[ "$MODE" == "onbot" && -n "$MOTION" ]]; then
    MOTION="$(abspath "$MOTION")"
fi

# ---------------------------------------------------------------------------
# --motion source-of-truth normalization
# ---------------------------------------------------------------------------
# If --motion is a PKL or YAML, bake to a per-run tempdir x2m2 so:
#   * the deploy binary always loads an x2m2 derived from THIS exact source
#     (no chance of a stale gear_sonic_deploy/data/motions_x2m2/<x>.x2m2
#     being silently consumed when the upstream PKL changed),
#   * the bridge's RSI init reads the SAME source PKL, so init pose
#     and reference-motion playback are bit-identical to what
#     eval_x2_mujoco_onnx.py --motion sees in Python sim-to-sim eval.
# Pass an .x2m2 directly to keep legacy behaviour (no bake, no auto sim-motion).
if [[ -n "$MOTION" ]]; then
    case "${MOTION,,}" in
        *.pkl|*.yaml|*.yml)
            if [[ ! -f "$MOTION" ]]; then
                echo -e "${RED}Error: --motion source does not exist: $MOTION${NC}" >&2
                exit 1
            fi
            MOTION_SOURCE="$MOTION"
            MOTION_BAKE_TMPDIR="$(mktemp -d -t x2_motion_bake.XXXXXX)"
            MOTION_BAKED="$MOTION_BAKE_TMPDIR/motion.x2m2"
            echo -e "${BLUE}[motion]${NC} baking $(basename "$MOTION_SOURCE") -> $MOTION_BAKED"
            if ! "$SIM_PYTHON" "$SCRIPT_DIR/scripts/export_motion_for_deploy.py" \
                    --in "$MOTION_SOURCE" --out "$MOTION_BAKED" --quiet; then
                echo -e "${RED}Error: motion bake failed for $MOTION_SOURCE${NC}" >&2
                rm -rf "$MOTION_BAKE_TMPDIR"
                exit 1
            fi
            MOTION="$MOTION_BAKED"
            # Make sure the tempdir is cleaned up on script exit. We append
            # to any existing EXIT trap so the sim-mode cleanup_sim trap
            # (installed later) still fires.
            trap 'rm -rf "$MOTION_BAKE_TMPDIR"' EXIT
            ;;
        *.x2m2)
            : # legacy passthrough; no MOTION_SOURCE, no auto sim-motion
            ;;
        *)
            echo -e "${YELLOW}Warning: --motion has unrecognised extension: $MOTION${NC}" >&2
            echo -e "${YELLOW}         Expected .pkl, .yaml, .yml, or .x2m2.${NC}" >&2
            ;;
    esac
fi

if [[ "$MODE" == "sim" ]]; then
    [[ -n "$SIM_MJCF" ]] && SIM_MJCF="$(abspath "$SIM_MJCF")"
    [[ -n "$SIM_MOTION" ]] && SIM_MOTION="$(abspath "$SIM_MOTION")"
    [[ -n "$SIM_RECORD_COMMANDS" ]] && SIM_RECORD_COMMANDS="$(abspath "$SIM_RECORD_COMMANDS")"

    # ----------------------------------------------------------------
    # --sim-profile resolution
    # ----------------------------------------------------------------
    # Resolve the default profile if the caller didn't pick one:
    #   --motion <pkl|yaml>    -> default 'parity'  (validate C++ vs Python)
    #   --motion <x2m2>        -> default 'manual'  (legacy / explicit flags)
    #   --motion omitted       -> default 'manual'
    if [[ -z "$SIM_PROFILE" ]]; then
        if [[ -n "$MOTION_SOURCE" ]]; then
            SIM_PROFILE="parity"
        else
            SIM_PROFILE="manual"
        fi
    fi
    case "$SIM_PROFILE" in
        parity|handoff|gantry|gantry-dangle|manual) : ;;
        *)
            echo -e "${RED}Error: --sim-profile must be one of: parity, handoff, gantry, gantry-dangle, manual (got '$SIM_PROFILE')${NC}" >&2
            exit 1
            ;;
    esac
    echo -e "${BLUE}[sim]${NC} sim profile: ${GREEN}$SIM_PROFILE${NC}"

    case "$SIM_PROFILE" in
        parity)
            # ===== Profile A: parity =====
            # Goal: validate the C++ deploy's obs assembly + action pipeline
            # bit-for-bit against eval_x2_mujoco_onnx.py. A correct C++
            # deploy MUST hold 30s clean here -- if it doesn't, the bug is
            # in the C++ binary or the bridge's obs publication, NOT the
            # policy or the motion data (since Python eval works under the
            # same init).
            #
            # Setup:
            #   * RSI from motion frame 0 (bridge needs source PKL/YAML).
            #   * Elastic band off (no transient release into gravity).
            #   * --ramp-seconds 0 (deploy at full alpha=1 from tick 0; the
            #     ramp blends toward default_angles and would yank joints
            #     away from the RSI'd pose for ~2s and tip the robot).
            if [[ -z "$MOTION_SOURCE" ]]; then
                echo -e "${RED}Error: --sim-profile parity requires --motion <pkl|yaml>${NC}" >&2
                echo -e "${RED}       (RSI needs the source motion the C++ x2m2 was baked from).${NC}" >&2
                exit 1
            fi
            if [[ -z "$SIM_MOTION" ]]; then
                SIM_MOTION="$MOTION_SOURCE"
                echo -e "${BLUE}[sim:parity]${NC} bridge RSI source: $(basename "$SIM_MOTION")"
            fi
            if [[ "$SIM_NO_ELASTIC_BAND" != "true" ]] \
                    && [[ "$SIM_KEEP_ELASTIC_BAND" != "true" ]] \
                    && [[ -z "$SIM_BAND_RELEASE_AFTER_S" ]]; then
                SIM_NO_ELASTIC_BAND=true
                echo -e "${BLUE}[sim:parity]${NC} elastic band: disabled (RSI gives stable ground contact at t=0)"
            fi
            if [[ -z "$RAMP_SECONDS" ]]; then
                RAMP_SECONDS="0"
                echo -e "${BLUE}[sim:parity]${NC} --ramp-seconds: 0 (mirrors Python eval; full alpha=1 from tick 0)"
            fi
            if [[ -z "$AUTOSTART" ]]; then
                # Python eval has no WAIT phase: it RSIs to motion frame 0 and
                # immediately ticks the policy. The longer the WAIT, the more
                # opportunity for the bridge's standby PD to drift the body
                # away from a clean RSI'd state (small numerical errors,
                # imperfect joint vel cancellation, etc.) before the policy
                # ever runs. Set autostart=0 so INIT->WAIT->CONTROL fires the
                # instant the bridge has published a fresh state.
                AUTOSTART="0"
                echo -e "${BLUE}[sim:parity]${NC} --autostart-after: 0 (no WAIT; Python eval has none)"
            fi
            ;;
        handoff)
            # ===== Profile B: handoff =====
            # Goal: validate the bring-up sequence the deploy actually
            # executes on the real robot. Real bring-up is:
            #   1. Robot in firmware-stand on the gantry (knees ~+28 deg,
            #      elbows ~-67 deg, gantry strap takes ~88% body weight,
            #      feet just barely touching ground -- this is the
            #      gantry_hang capture, NOT DEFAULT_DOF).
            #   2. Operator stops MC; deploy starts.
            #   3. Soft-start ramp blends commands from default_angles
            #      toward policy output over ~2s.
            #   4. Operator loosens the gantry strap; body now supports
            #      its own weight against the policy's commands.
            #   5. Policy holds the body upright on its own.
            #
            # Sim mirror:
            #   * --init-pose=gantry_hang  (firmware-stand pose, feet at
            #     floor, pelvis 0.665 m -- the actual real-robot start
            #     state, captured live from the X2).
            #   * Standard soft-start ramp (default 2.0s) unless overridden.
            #   * Elastic band ON at gantry_hang's ~88% support, auto-
            #     released ramp_seconds + 2.0s after the first deploy
            #     command -- proxies the operator releasing the gantry
            #     strap once the policy has full alpha and 2s of fresh
            #     in-control observations in the proprioception buffer.
            #
            # Compare to --sim-profile gantry, which is the same start pose
            # but holds the band on FOREVER (gantry-supported powered run,
            # the test we're actually allowed to run on hardware before we
            # bless the policy for free-standing operation).
            if [[ -n "$SIM_MOTION" ]]; then
                echo -e "${YELLOW}[sim:handoff] ignoring --sim-motion (handoff starts from a fixed gantry_hang pose, not RSI)${NC}"
                SIM_MOTION=""
            fi
            if [[ -z "$SIM_INIT_POSE" ]]; then
                SIM_INIT_POSE="gantry_hang"
                echo -e "${BLUE}[sim:handoff]${NC} init pose: gantry_hang (firmware-stand, feet on floor, ~88% on band)"
            fi
            if [[ "$SIM_KEEP_ELASTIC_BAND" == "true" ]] \
                    || { [[ "$SIM_NO_ELASTIC_BAND" != "true" ]] \
                         && [[ -z "$SIM_BAND_RELEASE_AFTER_S" ]]; }; then
                # Default ramp_seconds is 2.0 in the C++ deploy; we use
                # whatever the caller picked, falling back to 2.0.
                local_ramp="${RAMP_SECONDS:-2.0}"
                # Schedule band release for: ramp + 2s settle buffer.
                SIM_BAND_RELEASE_AFTER_S="$(awk "BEGIN { print $local_ramp + 2.0 }")"
                SIM_NO_ELASTIC_BAND=false
                echo -e "${BLUE}[sim:handoff]${NC} elastic band: ON at gantry_hang's band_length, auto-release ${SIM_BAND_RELEASE_AFTER_S}s after first deploy command (ramp + 2s settle)"
            fi
            if [[ -z "$RAMP_SECONDS" ]]; then
                echo -e "${BLUE}[sim:handoff]${NC} --ramp-seconds: deploy default (2.0s)"
            fi
            ;;
        gantry)
            # ===== Profile C: gantry =====
            # Goal: mirror the EXACT physical state the operator keeps the
            # X2 in during gantry-supported powered runs (deploy plan
            # Phase 7-9). The real robot is in a bent-knee crouch with
            # pelvis ~10 cm below standing height, the gantry strap takes
            # ~85-90 % of body weight, the feet just barely touch the
            # ground transmitting ~10-15 %. This profile sets the sim up
            # the same way so the closed-loop sim test and the powered
            # bring-up test the same operating point.
            #
            # Setup (driven by the bridge's new --init-pose / --band-*
            # knobs):
            #   * --init-pose=gantry-hang      MC-stand pose (captured live
            #                                  from the real X2 in firmware-
            #                                  stand mode; pelvis_z=0.665 m)
            #   * --band-length=GANTRY_HANG_BAND_LENGTH (~0.305 m) auto-
            #                                  picked by the bridge when
            #                                  init-pose=gantry-hang. Puts
            #                                  the band pull-target ~3 cm
            #                                  above the pelvis for ~88 %
            #                                  body weight on the band.
            #   * elastic band ON for the WHOLE run (no auto-release; the
            #     operator decides when to lower the gantry on hardware,
            #     so in sim we never auto-drop it)
            #   * --ramp-seconds at deploy default (2.0 s)
            #   * NO motion-RSI -- we want the policy to track the
            #     reference (e.g. minimal_v1's standing frames) FROM the
            #     captured firmware-stand pose, mirroring what happens
            #     when MC stops on the real robot.
            if [[ -n "$SIM_MOTION" ]]; then
                echo -e "${YELLOW}[sim:gantry] ignoring --sim-motion (gantry profile starts at gantry-hang pose, not RSI)${NC}"
                SIM_MOTION=""
            fi
            if [[ -z "$SIM_INIT_POSE" ]]; then
                SIM_INIT_POSE="gantry_hang"
                echo -e "${BLUE}[sim:gantry]${NC} init pose: gantry_hang (firmware-stand pose, pelvis_z=0.665m, from sim_init_poses.yaml)"
            fi
            if [[ -z "$SIM_BAND_LENGTH" ]]; then
                # Leave SIM_BAND_LENGTH empty so the bridge picks the
                # YAML's `band_length` for whichever pose was selected.
                # The operator can still override via --sim-band-length.
                echo -e "${BLUE}[sim:gantry]${NC} band length: pulled from sim_init_poses.yaml entry for this pose"
            fi
            if [[ "$SIM_NO_ELASTIC_BAND" == "true" ]]; then
                echo -e "${YELLOW}[sim:gantry] --sim-no-elastic-band conflicts with this profile; re-enabling band${NC}"
                SIM_NO_ELASTIC_BAND=false
            fi
            # Keep band ON forever -- the gantry doesn't auto-release in
            # reality. Operator picks when to lower; sim mirrors that.
            if [[ -z "$SIM_BAND_RELEASE_AFTER_S" ]]; then
                SIM_BAND_RELEASE_AFTER_S="-1"
                echo -e "${BLUE}[sim:gantry]${NC} band release: never (sim runs until --max-duration, mirroring an operator who keeps the gantry up)"
            fi
            if [[ -z "$RAMP_SECONDS" ]]; then
                echo -e "${BLUE}[sim:gantry]${NC} --ramp-seconds: deploy default (2.0s)"
            fi
            ;;
        gantry-dangle)
            # ===== Profile C2: gantry-dangle =====
            # Stress test: zero-torque hanging pose (MC fully stopped, robot
            # passive on the gantry). Pelvis sags ~4 cm below DEFAULT_DOF as
            # legs collapse into a deep crouch under gravity. Less common
            # in real bring-up than `gantry`, but useful for testing the
            # policy's recovery from off-distribution starting poses.
            # Captured from real X2 via x2_capture_pose.py (capture C).
            if [[ -n "$SIM_MOTION" ]]; then
                echo -e "${YELLOW}[sim:gantry-dangle] ignoring --sim-motion (profile starts at gantry_dangle pose, not RSI)${NC}"
                SIM_MOTION=""
            fi
            if [[ -z "$SIM_INIT_POSE" ]]; then
                SIM_INIT_POSE="gantry_dangle"
                echo -e "${BLUE}[sim:gantry-dangle]${NC} init pose: gantry_dangle (zero-torque hanging, pelvis_z=0.603m, from sim_init_poses.yaml)"
            fi
            if [[ -z "$SIM_BAND_LENGTH" ]]; then
                echo -e "${BLUE}[sim:gantry-dangle]${NC} band length: pulled from sim_init_poses.yaml entry for this pose"
            fi
            if [[ "$SIM_NO_ELASTIC_BAND" == "true" ]]; then
                echo -e "${YELLOW}[sim:gantry-dangle] --sim-no-elastic-band conflicts with this profile; re-enabling band${NC}"
                SIM_NO_ELASTIC_BAND=false
            fi
            if [[ -z "$SIM_BAND_RELEASE_AFTER_S" ]]; then
                SIM_BAND_RELEASE_AFTER_S="-1"
                echo -e "${BLUE}[sim:gantry-dangle]${NC} band release: never"
            fi
            if [[ -z "$RAMP_SECONDS" ]]; then
                echo -e "${BLUE}[sim:gantry-dangle]${NC} --ramp-seconds: deploy default (2.0s)"
            fi
            ;;
        manual)
            # ===== Profile D: manual =====
            # No automatic policy. Catch the deprecated explicit
            # --sim-motion case so users see the new one-arg ergonomics,
            # but don't override anything they set.
            if [[ -n "$SIM_MOTION" && -z "$MOTION_SOURCE" ]]; then
                echo -e "${YELLOW}[sim:manual] --sim-motion is deprecated; pass --motion <pkl|yaml> for the new auto-bake/source-of-truth path.${NC}" >&2
            fi
            ;;
    esac

    BRIDGE_PATH="$SCRIPT_DIR/$SIM_BRIDGE_REL"
    if [[ ! -f "$BRIDGE_PATH" ]]; then
        echo -e "${RED}Error: MuJoCo bridge script not found: $BRIDGE_PATH${NC}" >&2
        exit 1
    fi
    if [[ -n "$SIM_MJCF" ]] && [[ ! -f "$SIM_MJCF" ]]; then
        echo -e "${RED}Error: --sim-mjcf does not exist: $SIM_MJCF${NC}" >&2
        exit 1
    fi
    if [[ -n "$SIM_MOTION" ]] && [[ ! -f "$SIM_MOTION" ]]; then
        echo -e "${RED}Error: --sim-motion does not exist: $SIM_MOTION${NC}" >&2
        exit 1
    fi
    if [[ -n "$SIM_IMU_FROM" ]] && \
            [[ "$SIM_IMU_FROM" != "pelvis" && "$SIM_IMU_FROM" != "torso" ]]; then
        echo -e "${RED}Error: --sim-imu-from must be 'pelvis' or 'torso'${NC}" >&2
        exit 1
    fi
fi

# ============================================================================
# Header
# ============================================================================

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                       X2 ULTRA DEPLOY LAUNCHER                       ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}[Mode]${NC}                $MODE"
echo -e "${BLUE}[Package]${NC}             $PKG_NAME"
if [[ "$MODE" == "onbot" ]]; then
    echo -e "${BLUE}[Onbot prefix]${NC}        $ONBOT_PREFIX"
    echo -e "${BLUE}[Onbot workspace]${NC}     $ONBOT_WS"
    echo -e "${BLUE}[Onbot venv]${NC}          $ONBOT_VENV"
    echo -e "${BLUE}[Onbot onnxruntime]${NC}   $ONBOT_ONNXRUNTIME"
    echo -e "${BLUE}[Onbot aimdk_msgs]${NC}    $ONBOT_AIMDK_PREFIX"
    if [[ -n "$ROBOT_HOST" || -n "$ROBOT_USER" ]]; then
        echo -e "${BLUE}[Robot host]${NC}          $ROBOT_USER@$ROBOT_HOST ${YELLOW}(ignored: PC2-native)${NC}"
    fi
fi
echo ""

# ============================================================================
# Build command-line argument list for ros2 run
# ============================================================================

ROS2_ARGS=("--model" "$MODEL")
# VLA / ZMQ input source. When VLA_MODE=true (set via --vla), the deploy
# binary subscribes to a ZMQ pose topic instead of replaying a motion
# file. On the robot the publisher is the PC2 pose watchdog (fed by the
# kplanner daemon); in sim it is the laptop planner stack. See
# `docs/source/references/x2_zmq_protocol.md` for the wire format.
if [[ "${VLA_MODE:-false}" == "true" ]]; then
    ROS2_ARGS+=("--input-type" "zmq")
    ROS2_ARGS+=("--zmq-pose-host" "${VLA_ZMQ_HOST:-localhost}")
    ROS2_ARGS+=("--zmq-pose-port" "${VLA_ZMQ_PORT:-5556}")
    ROS2_ARGS+=("--zmq-pose-topic" "${VLA_ZMQ_TOPIC:-pose}")
elif [[ -n "$MOTION" ]]; then
    ROS2_ARGS+=("--motion" "$MOTION")
fi
# x2_debug telemetry PUB: enabled regardless of input source (VLA or
# --motion playback) so dump_x2_debug.py works for BOTH paths. Without
# this, A/B comparing PklMotionReference vs ZmqPoseInputSource is
# blind on the --motion side: no per-tick body_q_target /
# body_q_measured / safety_event stream. Set VLA_DEBUG_PORT=0 to
# disable.
if [[ -n "${VLA_DEBUG_PORT:-}" && "${VLA_DEBUG_PORT}" != "0" ]]; then
    ROS2_ARGS+=("--zmq-debug-port" "${VLA_DEBUG_PORT}")
    ROS2_ARGS+=("--zmq-debug-topic" "${VLA_DEBUG_TOPIC:-x2_debug}")
fi
# SAFE_IDLE resume socket (Quest 3 A+B chord). Wired when VLA mode is on
# AND a resume host was passed (we don't want the deploy to fail to
# subscribe to a phantom socket in legacy --motion runs).
if [[ "${VLA_MODE:-false}" == "true" && -n "${VLA_RESUME_HOST}" ]]; then
    ROS2_ARGS+=("--zmq-resume-host"  "${VLA_RESUME_HOST}")
    ROS2_ARGS+=("--zmq-resume-port"  "${VLA_RESUME_PORT}")
    ROS2_ARGS+=("--zmq-resume-topic" "${VLA_RESUME_TOPIC}")
fi
[[ -n "$LOG_DIR" ]]           && ROS2_ARGS+=("--log-dir" "$LOG_DIR")
[[ -n "$AUTOSTART" ]]         && ROS2_ARGS+=("--autostart-after" "$AUTOSTART")
[[ -n "$MAX_DURATION" ]]      && ROS2_ARGS+=("--max-duration" "$MAX_DURATION")
[[ -n "$TILT_COS" ]]          && ROS2_ARGS+=("--tilt-cos" "$TILT_COS")
[[ -n "$RAMP_SECONDS" ]]      && ROS2_ARGS+=("--ramp-seconds" "$RAMP_SECONDS")
[[ -n "$MAX_TARGET_DEV" ]]       && ROS2_ARGS+=("--max-target-dev" "$MAX_TARGET_DEV")
[[ -n "$MAX_TARGET_DEV_LEG" ]]   && ROS2_ARGS+=("--max-target-dev-leg"   "$MAX_TARGET_DEV_LEG")
[[ -n "$MAX_TARGET_DEV_WAIST" ]] && ROS2_ARGS+=("--max-target-dev-waist" "$MAX_TARGET_DEV_WAIST")
[[ -n "$MAX_TARGET_DEV_WAIST_PITCH" ]] && ROS2_ARGS+=("--max-target-dev-waist-pitch" "$MAX_TARGET_DEV_WAIST_PITCH")
[[ -n "$MAX_TARGET_DEV_ARM" ]]   && ROS2_ARGS+=("--max-target-dev-arm"   "$MAX_TARGET_DEV_ARM")
[[ -n "$MAX_TARGET_DEV_HEAD" ]]  && ROS2_ARGS+=("--max-target-dev-head"  "$MAX_TARGET_DEV_HEAD")
[[ -n "$KP_SCALE" ]]             && ROS2_ARGS+=("--kp-scale"             "$KP_SCALE")
[[ -n "$KP_SCALE_HIP" ]]         && ROS2_ARGS+=("--kp-scale-hip"         "$KP_SCALE_HIP")
[[ -n "$KP_SCALE_KNEE" ]]        && ROS2_ARGS+=("--kp-scale-knee"        "$KP_SCALE_KNEE")
[[ -n "$KP_SCALE_ANKLE" ]]       && ROS2_ARGS+=("--kp-scale-ankle"       "$KP_SCALE_ANKLE")
[[ -n "$KP_SCALE_ANKLE_PITCH" ]] && ROS2_ARGS+=("--kp-scale-ankle-pitch" "$KP_SCALE_ANKLE_PITCH")
[[ -n "$KP_SCALE_ANKLE_ROLL" ]]  && ROS2_ARGS+=("--kp-scale-ankle-roll"  "$KP_SCALE_ANKLE_ROLL")
[[ -n "$KP_SCALE_WAIST" ]]       && ROS2_ARGS+=("--kp-scale-waist"       "$KP_SCALE_WAIST")
[[ -n "$KP_SCALE_WAIST_YAW" ]]   && ROS2_ARGS+=("--kp-scale-waist-yaw"   "$KP_SCALE_WAIST_YAW")
[[ -n "$KP_SCALE_WAIST_PR" ]]    && ROS2_ARGS+=("--kp-scale-waist-pr"    "$KP_SCALE_WAIST_PR")
[[ -n "$KP_SCALE_SHOULDER" ]]    && ROS2_ARGS+=("--kp-scale-shoulder"    "$KP_SCALE_SHOULDER")
[[ -n "$KP_SCALE_ELBOW" ]]       && ROS2_ARGS+=("--kp-scale-elbow"       "$KP_SCALE_ELBOW")
[[ -n "$KP_SCALE_WRIST" ]]       && ROS2_ARGS+=("--kp-scale-wrist"       "$KP_SCALE_WRIST")
[[ -n "$KP_SCALE_HEAD" ]]        && ROS2_ARGS+=("--kp-scale-head"        "$KP_SCALE_HEAD")
[[ -n "$KD_SCALE" ]]             && ROS2_ARGS+=("--kd-scale"             "$KD_SCALE")
[[ -n "$KD_SCALE_HIP" ]]         && ROS2_ARGS+=("--kd-scale-hip"         "$KD_SCALE_HIP")
[[ -n "$KD_SCALE_KNEE" ]]        && ROS2_ARGS+=("--kd-scale-knee"        "$KD_SCALE_KNEE")
[[ -n "$KD_SCALE_ANKLE" ]]       && ROS2_ARGS+=("--kd-scale-ankle"       "$KD_SCALE_ANKLE")
[[ -n "$KD_SCALE_ANKLE_PITCH" ]] && ROS2_ARGS+=("--kd-scale-ankle-pitch" "$KD_SCALE_ANKLE_PITCH")
[[ -n "$KD_SCALE_ANKLE_ROLL" ]]  && ROS2_ARGS+=("--kd-scale-ankle-roll"  "$KD_SCALE_ANKLE_ROLL")
[[ -n "$KD_SCALE_WAIST" ]]       && ROS2_ARGS+=("--kd-scale-waist"       "$KD_SCALE_WAIST")
[[ -n "$KD_SCALE_WAIST_YAW" ]]   && ROS2_ARGS+=("--kd-scale-waist-yaw"   "$KD_SCALE_WAIST_YAW")
[[ -n "$KD_SCALE_WAIST_PR" ]]    && ROS2_ARGS+=("--kd-scale-waist-pr"    "$KD_SCALE_WAIST_PR")
[[ -n "$KD_SCALE_SHOULDER" ]]    && ROS2_ARGS+=("--kd-scale-shoulder"    "$KD_SCALE_SHOULDER")
[[ -n "$KD_SCALE_ELBOW" ]]       && ROS2_ARGS+=("--kd-scale-elbow"       "$KD_SCALE_ELBOW")
[[ -n "$KD_SCALE_WRIST" ]]       && ROS2_ARGS+=("--kd-scale-wrist"       "$KD_SCALE_WRIST")
[[ -n "$KD_SCALE_HEAD" ]]        && ROS2_ARGS+=("--kd-scale-head"        "$KD_SCALE_HEAD")
[[ -n "$TARGET_LPF_HZ" ]]     && ROS2_ARGS+=("--target-lpf-hz" "$TARGET_LPF_HZ")
[[ -n "$ACTION_CLIP" ]]       && ROS2_ARGS+=("--action-clip" "$ACTION_CLIP")
[[ -n "$RETURN_SECONDS" ]]    && ROS2_ARGS+=("--return-seconds" "$RETURN_SECONDS")
[[ -n "$IMU_TOPIC" ]]         && ROS2_ARGS+=("--imu-topic" "$IMU_TOPIC")
[[ -n "$INTRA_OP_THREADS" ]]  && ROS2_ARGS+=("--intra-op-threads" "$INTRA_OP_THREADS")
[[ -n "$OBS_DUMP" ]]          && ROS2_ARGS+=("--obs-dump" "$OBS_DUMP")
[[ -n "$WRIST_BYPASS" ]]      && ROS2_ARGS+=("--wrist-bypass" "$WRIST_BYPASS")
[[ -n "$HEAD_BYPASS" ]]       && ROS2_ARGS+=("--head-bypass" "$HEAD_BYPASS")
$DRY_RUN                      && ROS2_ARGS+=("--dry-run")
$NO_PUBLISH                   && ROS2_ARGS+=("--no-publish")

# ────────────────────────────────────────────────────────────────────────
# OBSERVE-ONLY needs an autostart, because it cannot use the stdin prompt.
#
# --no-publish forces NO_STOP_MC=true, which skips the start-trigger-sentinel
# setup (that lives inside `if ! $NO_STOP_MC`). Without a sentinel AND without
# --autostart-after, the node falls back to "type 'go' + Enter" on stdin -- but
# the ritual launches it with stdin on /dev/null, so it waits in
# WAIT_FOR_CONTROL forever. Observed 2026-08-23 10:19: deploy alive, 0 ticks,
# prompt printed, unreachable.
#
# Autostarting is safe here precisely BECAUSE --no-publish is set: entering
# CONTROL cannot move the robot, and MC still holds it.
# ────────────────────────────────────────────────────────────────────────
if $NO_PUBLISH && [[ -z "$AUTOSTART" ]]; then
    echo -e "${YELLOW}[observe-only] --autostart-after not set; defaulting to 5s${NC}"
    echo -e "${YELLOW}               (stdin is /dev/null under the ritual, so the${NC}"
    echo -e "${YELLOW}                interactive 'go' prompt can never be answered).${NC}"
    ROS2_ARGS+=("--autostart-after" "5")
fi

# ────────────────────────────────────────────────────────────────────────
# INTERLOCK: --no-publish must never arrive via --deploy-extra-arg.
#
# --deploy-extra-arg strings are appended to ROS2_ARGS verbatim WITHOUT this
# script inspecting them, so smuggling --no-publish through that path would
# mute the deploy while leaving the MC-stop logic untouched -- the one
# combination that leaves nothing commanding the joints. Refuse it outright
# and point at the first-class flag, which carries the NO_STOP_MC interlock.
#
# (This is the same class of bug as --obs-dump, whose "requesting shutdown"
# side effect the wrapper could not see: a flag whose consequences are
# invisible to the layer above it.)
# ────────────────────────────────────────────────────────────────────────
if [[ "${#DEPLOY_EXTRA_ARGS[@]}" -gt 0 ]] \
   && printf '%s\n' "${DEPLOY_EXTRA_ARGS[@]}" | grep -qx -- '--no-publish'; then
    echo -e "${RED}ERROR: --no-publish was passed via --deploy-extra-arg.${NC}"
    echo -e "${YELLOW}       Use the first-class flag instead:  deploy_x2.sh --no-publish ...${NC}"
    echo -e "${YELLOW}       Only the first-class flag carries the interlock that keeps MC${NC}"
    echo -e "${YELLOW}       running. Via --deploy-extra-arg the deploy would be muted while${NC}"
    echo -e "${YELLOW}       MC is still stopped, leaving NOTHING commanding the joints.${NC}"
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────
# Stand-default pose YAML (RAMP_OUT / HOLD_FOR_MC target in the C++ node).
#
# Real-robot modes (local/onbot): always forward when the file exists.
#
# Sim: historically omitted under the assumption HOLD_FOR_MC is unused.
# Sim + --vla still runs the handoff RAMP_OUT path against ``default_angles``
# unless we pass the captured MC stand YAML — which logs a loud warning
# and can snap elbows vs gantry_hang. Forward the shipped YAML for VLA sim
# only; pure --motion sim stays unchanged.
#
# HOLD_FOR_MC sentinels + timeout remain local/onbot-only (no MC bus in sim).
# ────────────────────────────────────────────────────────────────────────
if [[ "$MODE" != "sim" ]] || [[ "${VLA_MODE:-false}" == "true" ]]; then
    if [[ -z "$STAND_POSE_YAML" ]]; then
        # Default to the captured pose shipped in the repo. The C++ binary
        # falls back to default_angles if the file is missing, but it also
        # logs a loud warning -- so this default minimises the takeover
        # snap on the common case (operator just runs ./deploy_x2.sh local).
        STAND_POSE_YAML="$SCRIPT_DIR/configs/x2_stand_default_pose.yaml"
    fi
    if [[ -f "$STAND_POSE_YAML" ]]; then
        ROS2_ARGS+=("--stand-default-pose" "$STAND_POSE_YAML")
    else
        echo -e "${YELLOW}NOTE: --stand-pose-yaml '$STAND_POSE_YAML' not found;${NC}"
        echo -e "${YELLOW}      deploy node will fall back to default_angles for HOLD_FOR_MC.${NC}"
    fi
fi
if [[ "$MODE" != "sim" ]]; then
    if [[ -n "$HOLD_FOR_MC_TIMEOUT_S" && "$HOLD_FOR_MC_TIMEOUT_S" != "0" ]]; then
        ROS2_ARGS+=("--hold-for-mc-timeout-s" "$HOLD_FOR_MC_TIMEOUT_S")
        # Sentinel goes in $RUN_LOG_DIR if available (per-run dir; gets
        # cleaned up automatically by the run-recorder lifecycle), else
        # /tmp keyed by PID. Either way the deploy node's
        # ClearHoldForMcSentinel + this script's cleanup trap rm -f it.
        if [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]]; then
            HOLD_FOR_MC_SENTINEL="$RUN_LOG_DIR/hold_for_mc.sentinel"
            HOLD_FOR_MC_EXIT_SENTINEL="$RUN_LOG_DIR/hold_for_mc_exit.sentinel"
            MC_FIRST_PUBLISH_SENTINEL="$RUN_LOG_DIR/mc_first_publish.sentinel"
        else
            HOLD_FOR_MC_SENTINEL="/tmp/x2_hold_for_mc.$$.sentinel"
            HOLD_FOR_MC_EXIT_SENTINEL="/tmp/x2_hold_for_mc_exit.$$.sentinel"
            MC_FIRST_PUBLISH_SENTINEL="/tmp/x2_mc_first_publish.$$.sentinel"
        fi
        rm -f "$HOLD_FOR_MC_SENTINEL" "$HOLD_FOR_MC_EXIT_SENTINEL" \
              "$MC_FIRST_PUBLISH_SENTINEL"
        ROS2_ARGS+=("--hold-for-mc-sentinel" "$HOLD_FOR_MC_SENTINEL")
        ROS2_ARGS+=("--hold-for-mc-exit-sentinel" "$HOLD_FOR_MC_EXIT_SENTINEL")
        ROS2_ARGS+=("--mc-first-publish-sentinel" "$MC_FIRST_PUBLISH_SENTINEL")
        # Soft-shutdown trigger sentinel: touched on Ctrl-C by the cleanup
        # trap so deploy can enter RAMP_OUT -> HOLD_FOR_MC instead of
        # exiting immediately. Co-located with the other HOLD_FOR_MC
        # sentinels so the same RUN_LOG_DIR cleanup picks it up.
        # Skipped when --no-soft-shutdown is passed; in that case deploy
        # uses rclcpp's default SIGINT handler (legacy hard-exit path).
        if ! $SOFT_SHUTDOWN_DISABLED; then
            if [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]]; then
                SOFT_SHUTDOWN_SENTINEL="$RUN_LOG_DIR/soft_shutdown.sentinel"
            else
                SOFT_SHUTDOWN_SENTINEL="/tmp/x2_soft_shutdown.$$.sentinel"
            fi
            rm -f "$SOFT_SHUTDOWN_SENTINEL"
            ROS2_ARGS+=("--soft-shutdown-trigger-sentinel" "$SOFT_SHUTDOWN_SENTINEL")
        fi
    fi

    # ────────────────────────────────────────────────────────────────────
    # STANDBY pre-launch: spawn the C++ binary in a writer-suppressed
    # state BEFORE the safety gate so colcon build / DDS discovery /
    # ONNX load happen in parallel with the operator reading the prompt.
    # On 'Y', bash POSTs stop_app + verifies, then touches the trigger
    # sentinel; deploy advances STANDBY -> INIT -> WAIT -> CONTROL on
    # the next OnControl tick. End-to-end "Y -> CONTROL" lands at ~1 s
    # instead of ~14 s (1 s stop_app + 7 s build + 5 s autostart).
    # See the X2Deploy::State doc-comment for state-machine details.
    #
    # Skipped when --no-stop-mc is set (we never POST stop_app, so the
    # trigger sentinel would never fire and deploy would sit in STANDBY
    # forever). In that case fall back to the legacy boot-straight-to-
    # INIT path.
    # ────────────────────────────────────────────────────────────────────
    if ! $NO_STOP_MC; then
        if [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]]; then
            START_TRIGGER_SENTINEL="$RUN_LOG_DIR/start_trigger.sentinel"
            READY_SENTINEL="$RUN_LOG_DIR/ready.sentinel"
        else
            START_TRIGGER_SENTINEL="/tmp/x2_start_trigger.$$.sentinel"
            READY_SENTINEL="/tmp/x2_ready.$$.sentinel"
        fi
        rm -f "$START_TRIGGER_SENTINEL" "$READY_SENTINEL"
        ROS2_ARGS+=("--start-trigger-sentinel" "$START_TRIGGER_SENTINEL")
        ROS2_ARGS+=("--ready-sentinel" "$READY_SENTINEL")
        # Force --autostart-after to 0 in the bg-launch path: once the
        # trigger sentinel fires, deploy should fall through STANDBY ->
        # INIT (immediate, state already fresh from MC's continuous
        # broadcasts) -> WAIT_FOR_CONTROL -> CONTROL with no extra dwell.
        # Any user-supplied --autostart-after on the CLI still appears
        # earlier in ROS2_ARGS; the deploy parser is last-write-wins so
        # this 0 takes precedence.
        ROS2_ARGS+=("--autostart-after" "0")
    fi
fi

# ---------------------------------------------------------------------------
# Real-deploy tuning preset expansion (--tuning-config PATH.yaml).
#
# Sim profiles must NEVER load a tuning config: doing so would silently
# perturb the C++<->Python parity surface that eval_x2_mujoco.py establishes
# in MuJoCo. We hard-reject here with a friendly pointer to explicit CLI
# flags (which the operator can use if they really want to test a preset's
# effect in sim, knowing parity will diverge).
#
# In real-robot modes (local/onbot), the YAML keys are translated into
# deploy-binary flags by gear_sonic_deploy/scripts/tuning_config_to_args.py
# and PREPENDED to ROS2_ARGS (after the required --model). The deploy
# binary's CLI parser is last-write-wins on duplicate flags, so explicit
# --max-target-dev / --target-lpf-hz / etc. on this command line always
# override the preset. That ordering is intentional: it lets you do quick
# A/B sweeps off a known-good preset without copying the YAML.
# ---------------------------------------------------------------------------
if [[ -n "$TUNING_CONFIG" ]]; then
    if [[ "$MODE" == "sim" ]]; then
        echo -e "${RED}ERROR: --tuning-config is rejected in sim mode.${NC}" >&2
        echo -e "${RED}       Real-deploy tuning presets exist to mitigate real-robot${NC}" >&2
        echo -e "${RED}       sensor noise / hardware quirks. Loading them in sim would${NC}" >&2
        echo -e "${RED}       diverge from gear_sonic/scripts/eval_x2_mujoco.py and break${NC}" >&2
        echo -e "${RED}       the C++<->Python obs parity contract.${NC}" >&2
        echo -e "${YELLOW}       If you really want to test the preset's knobs in sim, copy${NC}" >&2
        echo -e "${YELLOW}       its values to explicit CLI flags (--max-target-dev,${NC}" >&2
        echo -e "${YELLOW}       --target-lpf-hz, ...) and accept that parity will diverge.${NC}" >&2
        exit 1
    fi
    TUNING_CONFIG="$(abspath "$TUNING_CONFIG")"
    if [[ ! -f "$TUNING_CONFIG" ]]; then
        echo -e "${RED}ERROR: --tuning-config file not found: $TUNING_CONFIG${NC}" >&2
        echo -e "${YELLOW}       Available presets:${NC}" >&2
        ls "$SCRIPT_DIR/configs/real_deploy_tuning/"*.yaml 2>/dev/null \
            | sed "s|^|         |" >&2
        exit 1
    fi
    TUNING_TRANSLATOR="$SCRIPT_DIR/scripts/tuning_config_to_args.py"
    if [[ ! -f "$TUNING_TRANSLATOR" ]]; then
        echo -e "${RED}ERROR: tuning translator missing: $TUNING_TRANSLATOR${NC}" >&2
        exit 1
    fi
    if ! mapfile -t TUNING_ARGS < <("$DEPLOY_AUX_PYTHON" "$TUNING_TRANSLATOR" "$TUNING_CONFIG"); then
        echo -e "${RED}ERROR: failed to parse tuning config $TUNING_CONFIG${NC}" >&2
        exit 1
    fi
    if [[ ${#TUNING_ARGS[@]} -gt 0 ]]; then
        # Prepend the tuning args after the mandatory --model so explicit
        # CLI flags (already in ROS2_ARGS) take precedence (last-write-wins
        # in the C++ ParseCli loop).
        ROS2_ARGS=("${ROS2_ARGS[0]}" "${ROS2_ARGS[1]}" \
                   "${TUNING_ARGS[@]}" \
                   "${ROS2_ARGS[@]:2}")
        echo -e "$(ts) ${GREEN}[tuning]${NC} loaded $(basename "$TUNING_CONFIG"): " \
                "${TUNING_ARGS[*]}"
    fi
fi

# Append any --deploy-extra-arg ... values at the very end so they win
# last-write-wins parsing inside the C++ binary, regardless of where
# they appear on the wrapper command line.
if [[ "${#DEPLOY_EXTRA_ARGS[@]}" -gt 0 ]]; then
    ROS2_ARGS+=("${DEPLOY_EXTRA_ARGS[@]}")
fi

# ============================================================================
# MC (Motion Control) HTTP helpers + cleanup trap
# ----------------------------------------------------------------------------
# `aima em start-app/stop-app mc` is a thin wrapper around an HTTP POST to
# PC1's Environment Manager. Talking to it directly avoids requiring an ssh
# key into the robot and makes the whole flow reachable from inside the
# docker_x2/ container. Verified against
# agitbot-x2-record-and-replay/src/x2_recorder/mc_control.py.
# ============================================================================

probe_pose_wire() {
    # Open a ZMQ SUB on the deploy's actual pose-input host:port for
    # WIRE_PROBE_SECS and count frames. Echoes a single line of the form
    #     OK frames=<N> dt_first_ms=<M>
    # on success (returns 0), or
    #     SILENT frames=0
    # on failure (returns 1). The caller is expected to wrap this with
    # operator-facing diagnostics. Idempotent and side-effect free:
    # no socket binds, only a connect+recv. Uses DEPLOY_AUX_PYTHON
    # (resolved in the onbot-env block) so the venv's pyzmq is in scope.
    local host="$1"; local port="$2"; local topic="$3"; local secs="$4"
    local py
    if [[ -n "${DEPLOY_AUX_PYTHON:-}" && -x "$DEPLOY_AUX_PYTHON" ]]; then
        py="$DEPLOY_AUX_PYTHON"
    elif command -v python3 &>/dev/null; then
        py="$(command -v python3)"
    else
        echo "ERROR_NO_PYTHON"
        return 1
    fi
    "$py" - "$host" "$port" "$topic" "$secs" <<'PYEOF'
import sys, time
try:
    import zmq
except ImportError as e:
    print(f"ERROR_NO_PYZMQ: {e}")
    sys.exit(2)
host, port, topic, secs = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.RCVHWM, 100)
sub.connect(f"tcp://{host}:{port}")
sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
poller = zmq.Poller()
poller.register(sub, zmq.POLLIN)
deadline = time.monotonic() + secs
n = 0
first_rx_s = None
while time.monotonic() < deadline:
    socks = dict(poller.poll(timeout=50))
    if sub in socks:
        try:
            sub.recv(zmq.NOBLOCK)
            if first_rx_s is None:
                first_rx_s = time.monotonic()
            n += 1
        except zmq.Again:
            pass
sub.close(linger=0)
if n > 0:
    dt = int((first_rx_s - (deadline - secs)) * 1000) if first_rx_s else -1
    print(f"OK frames={n} dt_first_ms={dt}")
    sys.exit(0)
else:
    print("SILENT frames=0")
    sys.exit(1)
PYEOF
}

dump_pose_proxy_log_tail() {
    # When the wire probe fails in split-topology (onbot) mode, the most
    # likely cause is that x2_pose_watchdog.py died at startup -- e.g.
    # because the laptop-side daemons wrapper passed a flag the PC2-side
    # proxy doesn't recognise (stale pc2_bringup), a Python import
    # error, or a port collision. The operator-facing failure banner
    # tells them WHERE the log lives, but they still have to leave the
    # tmux pane to read it. Inlining the last few lines of the proxy
    # log into the same scrollback makes the actual error obvious in
    # one place. Idempotent and safe to call even when no proxy log
    # exists (e.g. local-mode deploy with no proxy in the loop).
    local log_root="${PC2_LOG_ROOT:-/home/run/gear-sonic/log}"
    local newest
    newest="$(ls -t "${log_root}"/pose_proxy_*.log 2>/dev/null | head -n 1 || true)"
    if [[ -z "$newest" || ! -r "$newest" ]]; then
        echo -e "${YELLOW}  (no pose_proxy_*.log found under ${log_root}; proxy may never have started)${NC}"
        return 0
    fi
    echo ""
    echo -e "${YELLOW}── last 30 lines of ${newest} ──────────────────────${NC}"
    tail -n 30 "$newest" | sed 's/^/    /'
    echo -e "${YELLOW}── end of proxy log tail ────────────────────────────${NC}"
}

mc_em_post() {
    # $1 = action ("stop_app" or "start_app")
    #
    # Tries the legacy HTTP endpoint first (older PC1 firmware exposes
    # it on $MC_EM_URL/json/start_app | stop_app). Newer firmware (post
    # ~2026-05 on this fleet) dropped the HTTP wrapper entirely: the EM
    # service moved to ROS 2 services (/aimdk.protocol.EmAppService/
    # StartApp + StopApp), and the supported CLI on PC2 is `aima em
    # start-app mc` / `aima em stop-app mc`, which wraps those services.
    # When the HTTP POST fails AND we're on a host that has `aima` on
    # PATH (i.e. we're running on PC2 in onbot mode), fall back to the
    # CLI so the operator does not have to know which firmware they're
    # on. Useful for both onbot mode (PC2-native) and any future
    # firmware update on the local-mode laptop.
    local action="$1"
    local url="$MC_EM_URL/json/$action"
    local http_rc=0
    if [[ -z "$MC_EM_URL" ]]; then
        echo -e "$(ts) ${YELLOW}[mc-em]${NC} no EM HTTP URL (set PC1_HOST=<PC1_IP>, X2_MC_EM_URL or --mc-em-url); skipping HTTP, trying 'aima em' CLI ..." >&2
        http_rc=1
    elif command -v curl &>/dev/null; then
        if curl -fsS -X POST -H 'Content-Type: application/json' \
            --connect-timeout 3 --max-time 5 \
            -d '{"app_name":"mc"}' \
            "$url" >/dev/null 2>&1; then
            return 0
        fi
        http_rc=$?
    elif command -v python3 &>/dev/null; then
        if python3 - "$url" <<'PY' 2>/dev/null
import json, sys, urllib.request
url = sys.argv[1]
req = urllib.request.Request(
    url,
    data=json.dumps({"app_name": "mc"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        r.read()
except Exception:
    sys.exit(1)
PY
        then
            return 0
        fi
        http_rc=$?
    fi
    # HTTP failed. Try the `aima em` CLI fallback (PC2 only).
    local aima_subcmd=""
    case "$action" in
        stop_app)  aima_subcmd="stop-app" ;;
        start_app) aima_subcmd="start-app" ;;
    esac
    if [[ -n "$aima_subcmd" ]] && command -v aima &>/dev/null; then
        echo -e "$(ts) ${YELLOW}[mc-em]${NC} HTTP $url failed (rc=$http_rc); trying 'aima em $aima_subcmd mc' ..." >&2
        if aima em "$aima_subcmd" mc >/dev/null 2>&1; then
            echo -e "$(ts) ${GREEN}[mc-em]${NC} aima em $aima_subcmd mc OK." >&2
            return 0
        fi
        echo -e "$(ts) ${RED}[mc-em]${NC} aima em $aima_subcmd mc also failed." >&2
        return 1
    fi
    return $http_rc
}

# ────────────────────────────────────────────────────────────────────────
# MC mode helpers (smooth handoff). These wrap the existing stop_app /
# start_app HTTP path with a SetMcAction(DAMPING_DEFAULT) before stop and a
# SetMcAction(STAND_DEFAULT) after start, so the bus transitions are
# softer:
#
#   STAND  -> DAMPING -> stop_app -> deploy publishes -> ...
#   ... -> RAMP_OUT -> start_app -> wait MC up -> STAND
#
# Without these wraps:
#   - stop_app from STAND_DEFAULT cuts the PD-balancing instantly. Motors
#     drop to zero stiffness in one step => visible jolt on the gantry.
#   - start_app boots MC into whatever its default mode is; not deterministic.
#
# Helpers below use the MC service-call recipe: 250 ms
# timeout per service call, a few retries, parse action_desc out of the
# YAML response. Skipped if ros2 is unavailable on the current shell (host
# without ROS sourced); the deploy still works -- you just lose the smooth
# wrap.
# ────────────────────────────────────────────────────────────────────────
MC_SETMC_SVC="/aimdk_5Fmsgs/srv/SetMcAction"
MC_GETMC_SVC="/aimdk_5Fmsgs/srv/GetMcAction"
# Bumped from 0.5s to 3.0s after the iter-16k_180_lpf5 run on 2026-05-03
# logged "SetMcAction did not confirm" and "GetMcAction never responded
# after start_app". The empirical pnc-heartbeat probes used 3s and never
# saw a hang, so the original 0.5s was simply too tight for a freshly-
# booted MC where the service-discovery side of DDS hasn't fully
# propagated yet. 3s gives MC time to respond without making interactive
# waits feel sluggish.
MC_SET_TIMEOUT_S="${MC_SET_TIMEOUT_S:-3.0}"
MC_SET_RETRIES="${MC_SET_RETRIES:-6}"

mc_get_action() {
    # Print the current MC mode string on stdout, or empty on failure.
    #
    # The ROS 2 CLI may emit either YAML ('action_desc: STAND_DEFAULT') or
    # Python-repr ("action_desc='STAND_DEFAULT'") depending on rclpy
    # version. Parse both. (The pnc-heartbeat probe found this out the
    # hard way; see scratch/probes/mc_input_source_*/FINDINGS_addendum.md.)
    if ! command -v ros2 &>/dev/null; then
        return 1
    fi
    local out=""
    local n="${1:-$MC_SET_RETRIES}"
    for _ in $(seq 1 "$n"); do
        out=$(timeout "$MC_SET_TIMEOUT_S" \
                ros2 service call "$MC_GETMC_SVC" \
                aimdk_msgs/srv/GetMcAction \
                "{request: {}}" 2>/dev/null) || true
        if [[ -n "$out" ]]; then
            local mode=""
            # Python-repr form: action_desc='STAND_DEFAULT'
            mode=$(echo "$out" | grep -oE "action_desc='[^']*'" \
                   | head -1 | sed "s/.*action_desc='//; s/'$//")
            # YAML form: action_desc: STAND_DEFAULT
            if [[ -z "$mode" ]]; then
                mode=$(echo "$out" | awk -F': ' '
                    /action_desc:/ {
                        gsub(/[\r\n"\047]/, "", $2)
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
                        if ($2 != "") { print $2; exit }
                    }')
            fi
            if [[ -n "$mode" ]]; then
                echo "$mode"
                return 0
            fi
        fi
        sleep 0.2
    done
    return 1
}

mc_set_action() {
    # $1 = mode string (e.g. STAND_DEFAULT, DAMPING_DEFAULT). Returns 0 on
    # accepted reply (response.header.code == 0), 1 on exhausted retries /
    # no ros2.
    #
    # The actual aimdk_msgs/srv/SetMcAction schema is:
    #   Request: { header: RequestHeader, source: string,
    #              command: McActionCommand { action: McAction,
    #                                         action_desc: string } }
    #   Response: { response: CommonResponse }   (header.code = 0 on OK)
    #
    # The pre-2026-05-03 payload of {request: {action_desc: ...}} silently
    # NO-OP'd (the MC service ignored the unknown 'request' field, leaving
    # all required fields default-filled, which decoded to action_desc=''
    # and was rejected with a non-zero header.code). The deploy log then
    # showed "did not confirm" but the run continued. Verified against
    # /ros2_ws/install/aimdk_msgs/share/aimdk_msgs/srv/SetMcAction.srv.
    local mode="$1"
    if ! command -v ros2 &>/dev/null; then
        return 1
    fi
    local n="${2:-$MC_SET_RETRIES}"
    local payload
    payload=$(printf "{header: {}, source: deploy_x2, command: {action_desc: '%s'}}" "$mode")
    local out
    for _ in $(seq 1 "$n"); do
        out=$(timeout "$MC_SET_TIMEOUT_S" \
                ros2 service call "$MC_SETMC_SVC" \
                aimdk_msgs/srv/SetMcAction \
                "$payload" 2>/dev/null) || true
        # Success: response carries header.code=0. The python-repr ROS 2
        # CLI output includes that as 'code=0' on a single line; we also
        # accept the YAML form 'code: 0' for forward compat.
        if echo "$out" | grep -q -E '(code=0|code:\s*0)'; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

# Pre-handoff: bring MC down softly. Returns 0 if the SetMcAction call
# landed (or ros2 isn't available), so the caller can still proceed to
# stop_app even when the wrap is unsupported.
mc_pre_handoff_damp() {
    if ! command -v ros2 &>/dev/null; then
        echo -e "${YELLOW}  (smooth-handoff skipped: ros2 not on PATH)${NC}"
        return 0
    fi
    local from
    from=$(mc_get_action 1 || true)
    if [[ -n "$from" ]]; then
        echo -e "${BLUE}  Pre-handoff:${NC} MC mode = $from -> DAMPING_DEFAULT"
    else
        echo -e "${BLUE}  Pre-handoff:${NC} MC mode = (unknown) -> DAMPING_DEFAULT"
    fi
    if mc_set_action DAMPING_DEFAULT; then
        # Brief settle window for MC's internal mode transition. 500 ms is
        # plenty given the per-mode latencies we observed in mode-probe
        # (worst-case ~150 ms).
        sleep 0.5
        echo -e "${GREEN}  Pre-handoff: MC in DAMPING_DEFAULT.${NC}"
        return 0
    else
        echo -e "${YELLOW}  Pre-handoff: SetMcAction(DAMPING_DEFAULT) did not confirm.${NC}"
        echo -e "${YELLOW}  Falling back to direct stop_app (legacy behaviour).${NC}"
        return 1
    fi
}

# ────────────────────────────────────────────────────────────────────────
# Pre-handoff helper: drive MC to STAND_DEFAULT through the legal mode
# chain, asking the operator to confirm each transition.
#
# Mode chain (verified empirically against MC's SetMcAction service):
#
#   <unreachable>     ─ start_app via PC1 EM HTTP API
#         │             (use when a previous run crashed mid-handoff and
#         │              left MC's process down)
#         ▼
#   PASSIVE_DEFAULT   ─ SetMcAction(JOINT_DEFAULT)
#   (zero torque,       (gated by MC's posture detector! if the robot is
#    joints limp)        crouched into a 'sit' posture, the transition is
#         │              rejected. Operator fix: raise the gantry until
#         │              the robot is upright, then retry.)
#         ▼
#   JOINT_DEFAULT     ─ SetMcAction(STAND_DEFAULT)
#   (active joints,     (active balancing; resists sideways pushes)
#    no balancing)
#         │
#         ▼            ─ also reachable from DAMPING_DEFAULT
#   STAND_DEFAULT       (which is the mode mc_pre_handoff_damp transitions
#                        through on its way to stop_app)
#
# The function returns 0 on success (= MC is in STAND_DEFAULT), 1 if the
# operator declines or the chain breaks. --no-confirm skips the prompts
# but still walks the chain.
# ────────────────────────────────────────────────────────────────────────
escalate_mc_to_stand_default() {
    if ! command -v ros2 &>/dev/null; then
        echo -e "${YELLOW}  (mode escalation skipped: ros2 not on PATH)${NC}"
        return 1
    fi

    local current_mode=""
    current_mode="$(mc_get_action 3 2>/dev/null || true)"

    # ── Step 0: MC unreachable -> POST start_app, wait for services. ──
    if [[ -z "$current_mode" ]]; then
        echo -e "${YELLOW}  MC is not responding to GetMcAction on $MC_GETMC_SVC.${NC}"
        echo -e "${YELLOW}  This typically means the MC process is stopped (a previous run${NC}"
        echo -e "${YELLOW}  may have crashed mid-handoff). We can POST start_app to PC1${NC}"
        echo -e "${YELLOW}  ($MC_EM_URL) to bring it back. The robot stays at zero torque${NC}"
        echo -e "${YELLOW}  until MC re-attaches and you switch it to STAND_DEFAULT.${NC}"
        if ! $NO_CONFIRM; then
            read -p "$(echo -e ${YELLOW}Try POSTing start_app to bring MC back up? [y/N]: ${NC})" yn
            if [[ ! "$yn" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}  Declined. Resolve manually before re-running.${NC}"
                return 1
            fi
        fi
        if ! mc_em_post start_app; then
            echo -e "${RED}  start_app POST to $MC_EM_URL/json/start_app failed.${NC}"
            echo -e "${YELLOW}  Possible causes:${NC}"
            echo -e "${YELLOW}    - host has no route to PC1 (check the SDK NIC IP / PC1_HOST)${NC}"
            echo -e "${YELLOW}    - PC1 Environment Manager is down${NC}"
            return 1
        fi
        echo "  Waiting up to 30 s for MC services to come back ..."
        local i
        for i in $(seq 1 60); do
            current_mode="$(mc_get_action 1 2>/dev/null || true)"
            if [[ -n "$current_mode" ]]; then break; fi
            sleep 0.5
        done
        if [[ -z "$current_mode" ]]; then
            echo -e "${RED}  MC services never came up after start_app (waited 30 s).${NC}"
            echo -e "${YELLOW}  Check 'docker logs' on PC1 or ssh in and inspect MC.${NC}"
            return 1
        fi
        echo -e "${GREEN}  MC came back up in $current_mode.${NC}"
    fi

    # ── Already in STAND_DEFAULT: nothing to do. ──
    if [[ "$current_mode" == "STAND_DEFAULT" ]]; then
        return 0
    fi

    # ── Step 1: PASSIVE_DEFAULT -> JOINT_DEFAULT (gated by posture). ──
    if [[ "$current_mode" == "PASSIVE_DEFAULT" ]]; then
        echo -e "${YELLOW}  MC is in PASSIVE_DEFAULT (zero torque, joints fully limp).${NC}"
        echo -e "${YELLOW}  To resume active control we need to walk MC through:${NC}"
        echo -e "${YELLOW}    PASSIVE_DEFAULT -> JOINT_DEFAULT -> STAND_DEFAULT${NC}"
        echo -e "${YELLOW}  JOINT_DEFAULT activates the joints (no balancing). MC's posture${NC}"
        echo -e "${YELLOW}  detector will REJECT this transition if the robot is currently${NC}"
        echo -e "${YELLOW}  crouched into a 'sit' posture (knees deeply bent, pelvis low).${NC}"
        echo -e "${YELLOW}  Confirm BEFORE proceeding:${NC}"
        echo -e "${YELLOW}    [ ] Robot is firmly supported on the gantry${NC}"
        echo -e "${YELLOW}    [ ] Pelvis is roughly at standing height (knees not deeply bent)${NC}"
        echo -e "${YELLOW}    [ ] Joints will TENSE UP when JOINT_DEFAULT engages${NC}"
        if ! $NO_CONFIRM; then
            read -p "$(echo -e ${YELLOW}Switch MC to JOINT_DEFAULT? [y/N]: ${NC})" yn
            if [[ ! "$yn" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}  Declined. Robot stays in PASSIVE_DEFAULT.${NC}"
                return 1
            fi
        fi
        if ! mc_set_action JOINT_DEFAULT; then
            echo -e "${RED}  SetMcAction(JOINT_DEFAULT) was rejected.${NC}"
            echo -e "${YELLOW}  Most likely cause: MC's posture detector classified the robot${NC}"
            echo -e "${YELLOW}  as 'sit' (we have hit this exact failure before -- see${NC}"
            echo -e "${YELLOW}  scratch/probes/mc_input_source_*/FINDINGS_addendum.md).${NC}"
            echo -e "${YELLOW}  Recovery:${NC}"
            echo -e "${YELLOW}    1. Raise the gantry so the robot's pelvis is higher and${NC}"
            echo -e "${YELLOW}       the knees are less bent.${NC}"
            echo -e "${YELLOW}    2. Re-run this script (the gate will re-attempt).${NC}"
            return 1
        fi
        echo -e "${GREEN}  -> JOINT_DEFAULT.${NC}"
        sleep 0.5
        # Re-read so we walk the next step from the actual current mode.
        current_mode="$(mc_get_action 2 2>/dev/null || echo JOINT_DEFAULT)"
    fi

    # ── Step 2: anything-other-than-STAND -> STAND_DEFAULT. ──
    # Reachable from JOINT_DEFAULT (Step 1 fall-through) and from
    # DAMPING_DEFAULT (operator hit damping on the mobile app, etc.).
    if [[ "$current_mode" != "STAND_DEFAULT" ]]; then
        echo -e "${YELLOW}  MC is in $current_mode. Switching to STAND_DEFAULT for active${NC}"
        echo -e "${YELLOW}  balancing -- the robot will start resisting sideways pushes${NC}"
        echo -e "${YELLOW}  the moment this lands. Make sure the gantry has slack.${NC}"
        if ! $NO_CONFIRM; then
            read -p "$(echo -e ${YELLOW}Switch MC to STAND_DEFAULT? [y/N]: ${NC})" yn
            if [[ ! "$yn" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}  Declined. Robot stays in $current_mode.${NC}"
                return 1
            fi
        fi
        if ! mc_set_action STAND_DEFAULT; then
            echo -e "${RED}  SetMcAction(STAND_DEFAULT) was rejected.${NC}"
            echo -e "${YELLOW}  If you came from PASSIVE/JOINT, raise the gantry as above.${NC}"
            echo -e "${YELLOW}  If you came from DAMPING_DEFAULT, MC may be in a fault state${NC}"
            echo -e "${YELLOW}  -- check the mobile app for any red indicators.${NC}"
            return 1
        fi
        echo -e "${GREEN}  -> STAND_DEFAULT.${NC}"
        sleep 0.5
    fi

    # ── Step 3: confirm we landed where we expected. ──
    current_mode="$(mc_get_action 3 2>/dev/null || true)"
    if [[ "$current_mode" == "STAND_DEFAULT" ]]; then
        return 0
    fi
    echo -e "${RED}  After escalation MC is in '${current_mode:-<unknown>}', not STAND_DEFAULT.${NC}"
    return 1
}

# Non-interactive variant of escalate_mc_to_stand_default for use mid-
# handoff, when the deploy node is actively holding the robot in MC's
# stand pose and we just POSTed start_app. The deploy can hold for at
# most --hold-for-mc-timeout-s seconds (default 45) before it bails, so
# we cannot block on operator confirmations here -- the robot literally
# falls (well, the deploy stops driving it) if we sit at a y/N prompt.
# Walks PASSIVE -> JOINT_DEFAULT -> STAND_DEFAULT (or DAMPING -> STAND).
# Returns 0 when MC is in STAND_DEFAULT, 1 otherwise.
mc_post_policy_escalate_to_stand() {
    if ! command -v ros2 &>/dev/null; then return 1; fi
    # Wait up to 30 s for MC services to come back after start_app.
    local i mode=""
    for i in $(seq 1 60); do
        mode="$(mc_get_action 1 2>/dev/null || true)"
        [[ -n "$mode" ]] && break
        sleep 0.5
    done
    if [[ -z "$mode" ]]; then
        echo -e "$(ts) ${RED}[post-handoff]${NC} MC services never came up after start_app."
        return 1
    fi
    echo -e "$(ts) ${BLUE}[post-handoff]${NC} MC came back up in $mode."

    # PASSIVE -> JOINT_DEFAULT (the robot is being held upright by the
    # deploy node, so the posture detector should accept this).
    # NOTE: SetMcAction blocks until MC accepts/rejects, so no sleep is
    # needed afterwards. Skipping the sleep minimises the JOINT_DEFAULT
    # window where MC and deploy briefly publish in parallel (deploy kp
    # vs MC kp = motor whir if the gains differ). Goal is to be in
    # JOINT only as long as it takes to send the next service call.
    if [[ "$mode" == "PASSIVE_DEFAULT" ]]; then
        if mc_set_action JOINT_DEFAULT; then
            echo -e "$(ts) ${GREEN}[post-handoff]${NC} -> JOINT_DEFAULT."
            mode="JOINT_DEFAULT"
        else
            echo -e "$(ts) ${RED}[post-handoff]${NC} SetMcAction(JOINT_DEFAULT) was rejected."
            echo -e "${YELLOW}  Robot is currently held by deploy node; you have ~30 s${NC}"
            echo -e "${YELLOW}  before HOLD_FOR_MC times out. Switch via the mobile app.${NC}"
            return 1
        fi
    fi

    # JOINT_DEFAULT or DAMPING_DEFAULT -> STAND_DEFAULT (back-to-back
    # with the JOINT call -- no intervening sleep, see comment above).
    if [[ "$mode" != "STAND_DEFAULT" ]]; then
        if mc_set_action STAND_DEFAULT; then
            echo -e "$(ts) ${GREEN}[post-handoff]${NC} -> STAND_DEFAULT."
        else
            echo -e "$(ts) ${RED}[post-handoff]${NC} SetMcAction(STAND_DEFAULT) was rejected."
            return 1
        fi
    fi

    # Final verify (this is the only place we wait for MC to settle: the
    # SetMcAction service call returned, but MC's mode publisher catches
    # up a few ms later).
    mode="$(mc_get_action 3 2>/dev/null || true)"
    if [[ "$mode" == "STAND_DEFAULT" ]]; then
        return 0
    fi
    echo -e "$(ts) ${RED}[post-handoff]${NC} After escalation MC is in '${mode:-<unknown>}'."
    return 1
}

# Post-handoff: after start_app, wait for MC to expose its services again,
# then ask it to STAND_DEFAULT. Best-effort -- failure is logged but does
# not abort the cleanup. If ros2 isn't available we just skip the STAND
# request and rely on MC's default boot mode.
mc_post_handoff_stand() {
    if ! command -v ros2 &>/dev/null; then
        echo -e "${YELLOW}  (post-handoff skipped: ros2 not on PATH)${NC}"
        return 0
    fi
    # Poll until GetMcAction responds (MC has finished booting back up).
    local i mode=""
    for i in $(seq 1 30); do
        if mode=$(mc_get_action 1 2>/dev/null); then
            break
        fi
        sleep 0.5
    done
    if [[ -z "$mode" ]]; then
        echo -e "${YELLOW}  Post-handoff: GetMcAction never responded after start_app${NC}"
        echo -e "${YELLOW}  (waited ~15s). Robot left in whatever mode MC booted into.${NC}"
        return 1
    fi
    echo -e "${BLUE}  Post-handoff:${NC} MC came back up in $mode. Requesting STAND_DEFAULT ..."
    if mc_set_action STAND_DEFAULT; then
        echo -e "${GREEN}  Post-handoff: MC -> STAND_DEFAULT.${NC}"
        return 0
    else
        echo -e "${YELLOW}  Post-handoff: SetMcAction(STAND_DEFAULT) did not confirm.${NC}"
        echo -e "${YELLOW}  Robot stays in $mode; switch from the mobile app if needed.${NC}"
        return 1
    fi
}

start_run_recorder() {
    # Background the npz recorder. Called from the launch step (before the
    # mode branching) so the recording covers WAIT -> CONTROL -> RAMP_OUT.
    # Safe to call unconditionally -- no-op when --record wasn't passed.
    if ! $RECORD_RUN; then return 0; fi
    local recorder="$SCRIPT_DIR/scripts/x2_record_real_run.py"
    if [[ ! -f "$recorder" ]]; then
        echo -e "$(ts) ${YELLOW}[record]${NC} $recorder not found; skipping --record" >&2
        return 0
    fi
    mkdir -p "$(dirname "$RECORD_OUT")"
    echo -e "$(ts) ${BLUE}[record]${NC} backgrounding recorder -> $RECORD_OUT"
    # --quiet keeps the 1 Hz status line out of the deploy's terminal output.
    # Use scripts/x2_record_real_run.py --summarize PATH.npz after the run
    # to pull the analysis. Inheriting our shell's ROS env (sourced by the
    # docker auto-relaunch) means the recorder lands on the same domain as
    # the deploy with no extra setup.
    "$DEPLOY_AUX_PYTHON" "$recorder" \
        --out "$RECORD_OUT" \
        --note "deploy_x2.sh $MODE @ $(date -Iseconds)" \
        --quiet &
    RUN_RECORD_PID=$!
    sleep 0.5
    if ! kill -0 "$RUN_RECORD_PID" 2>/dev/null; then
        echo -e "$(ts) ${YELLOW}[record]${NC} recorder exited immediately; check rclpy/aimdk_msgs" >&2
        RUN_RECORD_PID=""
    fi
}

stop_run_recorder() {
    [[ -z "$RUN_RECORD_PID" ]] && return 0
    if kill -0 "$RUN_RECORD_PID" 2>/dev/null; then
        echo -e "$(ts) ${BLUE}[cleanup]${NC} stopping run recorder (pid $RUN_RECORD_PID) ..."
        kill -INT "$RUN_RECORD_PID" 2>/dev/null || true
        wait "$RUN_RECORD_PID" 2>/dev/null || true
        echo -e "$(ts) ${GREEN}[cleanup]${NC} recorder finalized: $RECORD_OUT"
    fi
    RUN_RECORD_PID=""
}

# ─────────────────────────────────────────────────────────────────────
# Hand bridge (ZMQ pose -> AimDK HandCommandArray) lifecycle
# ─────────────────────────────────────────────────────────────────────
# Spawned in MODE=local AFTER MC has been verified silenced (so its
# engage burst lands cleanly on a quiet bus). Reaped via the same
# restart_mc_on_exit trap that handles the deploy + run recorder, so
# Ctrl-C / abort / normal exit all converge on the same teardown path.
# Sim mode does NOT call this -- the MuJoCo bridge with --with-omnihand
# already writes finger qpos directly into MuJoCo and there's no AimDK
# HAL to publish to. onbot mode does not call this either -- the hand
# bridge is owned by scripts/x2_pc2_daemons.sh, which launches it in
# its own tmux session on PC2 alongside the deploy and motor monitor
# (so all three share the same WiFi-disconnect-tolerant lifecycle).
start_hand_bridge() {
    $HAND_BRIDGE_ENABLED || return 0
    if [[ "$MODE" != "local" ]]; then
        return 0
    fi
    if [[ "$HAND_BRIDGE_SIDES" == "off" ]]; then
        echo -e "$(ts) ${YELLOW}[hand-bridge]${NC} --hand-bridge-sides=off; not starting."
        return 0
    fi

    local script_path="$SCRIPT_DIR/$HAND_BRIDGE_SCRIPT_REL"
    if [[ ! -f "$script_path" ]]; then
        echo -e "$(ts) ${YELLOW}[hand-bridge]${NC} script not found: $script_path -- skipping." >&2
        return 0
    fi

    local py="${HAND_BRIDGE_PYTHON:-$DEPLOY_AUX_PYTHON}"
    if ! command -v "$py" &>/dev/null; then
        echo -e "$(ts) ${YELLOW}[hand-bridge]${NC} python interpreter '$py' not on PATH; skipping." >&2
        echo -e "$(ts) ${YELLOW}[hand-bridge]${NC} pass --hand-bridge-python /path/to/python or --no-hand-bridge to silence." >&2
        return 0
    fi

    if [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]]; then
        HAND_BRIDGE_LOG="$RUN_LOG_DIR/hand_bridge.log"
    else
        HAND_BRIDGE_LOG="/tmp/x2_hand_bridge.$$.log"
    fi

    local args=(
        "$script_path"
        --zmq-host "${VLA_ZMQ_HOST:-localhost}"
        --zmq-port "${VLA_ZMQ_PORT:-5556}"
        --zmq-topic "${VLA_ZMQ_TOPIC:-pose}"
        --sides "$HAND_BRIDGE_SIDES"
        --engage-shots "$HAND_BRIDGE_ENGAGE_SHOTS"
        --max-stale-s "$HAND_BRIDGE_MAX_STALE_S"
        --publish-hz "$HAND_BRIDGE_PUBLISH_HZ"
        --duration 0
    )

    echo -e "$(ts) ${BLUE}[hand-bridge]${NC} spawning: $py ${args[*]}"
    echo -e "$(ts) ${BLUE}[hand-bridge]${NC} stdout/stderr -> $HAND_BRIDGE_LOG"
    "$py" "${args[@]}" >"$HAND_BRIDGE_LOG" 2>&1 &
    HAND_BRIDGE_PID=$!

    # Best-effort liveness check: if the bridge died inside the
    # detect-timeout window (e.g. aimdk_msgs missing on PYTHONPATH or
    # zmq import failed), surface that immediately so the operator
    # isn't surprised at the first failed grasp.
    sleep 0.5
    if ! kill -0 "$HAND_BRIDGE_PID" 2>/dev/null; then
        echo -e "$(ts) ${YELLOW}[hand-bridge]${NC} exited immediately; tail of log:" >&2
        tail -n 20 "$HAND_BRIDGE_LOG" 2>/dev/null | sed 's/^/  /' >&2 || true
        echo -e "$(ts) ${YELLOW}[hand-bridge]${NC} continuing without hand publishing -- fingers will not move." >&2
        HAND_BRIDGE_PID=""
        return 0
    fi
    echo -e "$(ts) ${GREEN}[hand-bridge]${NC} pid $HAND_BRIDGE_PID up (sides=$HAND_BRIDGE_SIDES)."
}

stop_hand_bridge() {
    [[ -z "$HAND_BRIDGE_PID" ]] && return 0
    if kill -0 "$HAND_BRIDGE_PID" 2>/dev/null; then
        echo -e "$(ts) ${BLUE}[cleanup]${NC} stopping hand bridge (pid $HAND_BRIDGE_PID) ..."
        kill -INT "$HAND_BRIDGE_PID" 2>/dev/null || true
        # Bridge handles SIGINT cleanly; small grace then SIGTERM if it
        # didn't notice (engage timer mid-tick, etc).
        local i
        for i in 1 2 3 4 5; do
            kill -0 "$HAND_BRIDGE_PID" 2>/dev/null || break
            sleep 0.2
        done
        if kill -0 "$HAND_BRIDGE_PID" 2>/dev/null; then
            kill -TERM "$HAND_BRIDGE_PID" 2>/dev/null || true
            wait "$HAND_BRIDGE_PID" 2>/dev/null || true
        fi
        echo -e "$(ts) ${GREEN}[cleanup]${NC} hand bridge stopped."
    fi
    HAND_BRIDGE_PID=""
}

cleanup_sim() {
    local rc=$?
    if [[ -n "$SIM_BRIDGE_PID" ]] && kill -0 "$SIM_BRIDGE_PID" 2>/dev/null; then
        echo ""
        echo -e "$(ts) ${BLUE}[cleanup]${NC} stopping MuJoCo bridge (pid $SIM_BRIDGE_PID) ..."
        kill -INT "$SIM_BRIDGE_PID" 2>/dev/null || true
        wait "$SIM_BRIDGE_PID" 2>/dev/null || true
    fi
    if [[ -n "$SIM_RECORD_PID" ]] && kill -0 "$SIM_RECORD_PID" 2>/dev/null; then
        echo -e "$(ts) ${BLUE}[cleanup]${NC} stopping command bag recorder (pid $SIM_RECORD_PID) ..."
        kill -INT "$SIM_RECORD_PID" 2>/dev/null || true
        wait "$SIM_RECORD_PID" 2>/dev/null || true
    fi
    stop_run_recorder
    exit $rc
}

soft_shutdown_wait() {
    # Graceful Ctrl-C path. Cooperates with the C++ deploy's custom
    # SIGINT/SIGTERM handler (--soft-shutdown-trigger-sentinel) so the
    # robot stays under torque all the way through MC's PASSIVE_DEFAULT
    # boot -- no zero-torque drop.
    #
    # Flow:
    #   1. Touch SOFT_SHUTDOWN_SENTINEL (belt + suspenders: if deploy's
    #      in-process flag missed for any reason, the next OnControl
    #      tick will see the file and still RAMP_OUT).
    #   2. Wait up to SOFT_SHUTDOWN_WAIT_S for HOLD_FOR_MC_SENTINEL to
    #      appear, which means deploy has finished RAMP_OUT and is now
    #      publishing MC's STAND_DEFAULT pose with MC-stand gains.
    #   3. While waiting, a SECOND Ctrl-C sets SOFT_SHUTDOWN_ABORTED and
    #      we bail out early -- the caller falls through to the legacy
    #      hard-exit cleanup (POST start_app immediately).
    #
    # Safe no-op when:
    #   * soft-shutdown was never wired (SOFT_SHUTDOWN_SENTINEL empty)
    #   * deploy has already exited (no PID alive)
    #   * we've already run once this exit (SOFT_SHUTDOWN_TRIGGERED true)
    if [[ -z "${SOFT_SHUTDOWN_SENTINEL:-}" ]]; then
        return 0
    fi
    if $SOFT_SHUTDOWN_TRIGGERED; then
        return 0
    fi
    SOFT_SHUTDOWN_TRIGGERED=true
    if [[ -z "${DEPLOY_PID:-}" ]] || ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
        # Deploy already gone. Nothing to coordinate with; let the
        # legacy MC-restart path run.
        return 0
    fi
    # Re-trap SIGINT inside the wait loop so a second Ctrl-C trips
    # SOFT_SHUTDOWN_ABORTED instead of immediately re-firing the EXIT
    # trap (which would loop us back here). On exit from this function
    # the caller will install the next trap as appropriate.
    trap 'SOFT_SHUTDOWN_ABORTED=true; echo -e "$(ts) ${YELLOW}[soft-shutdown]${NC} second Ctrl-C -> aborting graceful wait, falling through to MC restart."' INT
    echo ""
    echo -e "$(ts) ${BLUE}[soft-shutdown]${NC} touching trigger sentinel '$SOFT_SHUTDOWN_SENTINEL' -> deploy RAMP_OUT -> HOLD_FOR_MC."
    echo -e "$(ts) ${BLUE}[soft-shutdown]${NC}   Press Ctrl-C again within ${SOFT_SHUTDOWN_WAIT_S}s to abort the graceful wait and force MC restart immediately."
    : > "$SOFT_SHUTDOWN_SENTINEL"
    # Poll for HOLD_FOR_MC_SENTINEL (deploy has reached HOLD_FOR_MC and
    # is publishing the stand pose). We sleep 0.1s between checks for
    # snappy response to the second-Ctrl-C abort. Total budget is
    # SOFT_SHUTDOWN_WAIT_S seconds.
    local deadline_ns
    deadline_ns=$(( $(date +%s%N) + SOFT_SHUTDOWN_WAIT_S * 1000000000 ))
    while ! $SOFT_SHUTDOWN_ABORTED; do
        if [[ -n "${HOLD_FOR_MC_SENTINEL:-}" && -f "$HOLD_FOR_MC_SENTINEL" ]]; then
            echo -e "$(ts) ${GREEN}[soft-shutdown]${NC} deploy reached HOLD_FOR_MC -> safe to restart MC."
            break
        fi
        # Deploy died mid-RAMP_OUT (uncaught crash, segfault, OOM, ...).
        # No point waiting further; fall through to MC restart.
        if ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
            echo -e "$(ts) ${YELLOW}[soft-shutdown]${NC} deploy exited before reaching HOLD_FOR_MC -> falling through to MC restart."
            break
        fi
        if [[ $(date +%s%N) -ge $deadline_ns ]]; then
            echo -e "$(ts) ${YELLOW}[soft-shutdown]${NC} timed out after ${SOFT_SHUTDOWN_WAIT_S}s waiting for HOLD_FOR_MC sentinel -> falling through to MC restart. (Deploy may still be in RAMP_OUT -- check the deploy log.)"
            break
        fi
        sleep 0.1
    done
}

restart_mc_on_exit() {
    # Always called via the trap once we've stopped MC. Idempotent + safe to
    # call multiple times. Preserves the original exit code so a failing
    # deploy run still surfaces its non-zero status to the caller / CI.
    local rc=$?
    # Graceful Ctrl-C: give deploy time to RAMP_OUT and reach HOLD_FOR_MC
    # BEFORE we POST start_app to MC. Without this step, deploy exits on
    # SIGINT (~50 ms) and the bus goes silent for the 1-2 s MC takes to
    # boot through PASSIVE_DEFAULT -- robot drops under gravity. With it,
    # deploy keeps publishing the stand pose with MC-stand gains
    # throughout MC's boot. No-op if --soft-shutdown-trigger-sentinel
    # wasn't wired (SOFT_SHUTDOWN_SENTINEL empty) or deploy already exited.
    soft_shutdown_wait
    # Stop the run recorder so it captures the deploy's RAMP_OUT and
    # the silence between deploy-exit and MC-restart in the same npz.
    stop_run_recorder
    # Stop the hand bridge BEFORE MC restarts, so MC's own hand
    # republish loop owns /aima/hal/joint/hand/command exclusively
    # again and we don't dual-publish for the brief window between
    # mc start_app and the bridge's SIGINT-driven exit.
    stop_hand_bridge
    # Clear all sentinels so stale files from a crashed run cannot
    # mis-trigger the next invocation. Best-effort.
    if [[ -n "${HOLD_FOR_MC_SENTINEL:-}" ]]; then
        rm -f "$HOLD_FOR_MC_SENTINEL"
    fi
    if [[ -n "${HOLD_FOR_MC_EXIT_SENTINEL:-}" ]]; then
        rm -f "$HOLD_FOR_MC_EXIT_SENTINEL"
    fi
    if [[ -n "${MC_FIRST_PUBLISH_SENTINEL:-}" ]]; then
        rm -f "$MC_FIRST_PUBLISH_SENTINEL"
    fi
    if [[ -n "${SOFT_SHUTDOWN_SENTINEL:-}" ]]; then
        rm -f "$SOFT_SHUTDOWN_SENTINEL"
    fi
    if [[ -n "${ESCALATOR_OK_SENTINEL:-}" ]]; then
        rm -f "$ESCALATOR_OK_SENTINEL"
    fi
    if [[ -n "${ESCALATOR_PID:-}" ]] && kill -0 "$ESCALATOR_PID" 2>/dev/null; then
        kill -TERM "$ESCALATOR_PID" 2>/dev/null || true
        wait "$ESCALATOR_PID" 2>/dev/null || true
    fi
    if [[ -n "${START_TRIGGER_SENTINEL:-}" ]]; then
        rm -f "$START_TRIGGER_SENTINEL"
    fi
    if [[ -n "${READY_SENTINEL:-}" ]]; then
        rm -f "$READY_SENTINEL"
    fi
    if $MC_STOPPED_BY_US; then
        # Mark first so a second SIGINT doesn't double-fire.
        MC_STOPPED_BY_US=false
        echo ""
        echo -e "$(ts) ${BLUE}[cleanup]${NC} restarting MC on $MC_EM_URL ..."
        if mc_em_post start_app; then
            echo -e "$(ts) ${GREEN}[cleanup]${NC} MC start_app POSTed."
            # Smooth-handoff: wait for MC services to come back, then ask
            # for STAND_DEFAULT so the robot ends up balancing again
            # without operator intervention. Non-fatal on failure.
            mc_post_handoff_stand || true
        else
            echo -e "$(ts) ${RED}[cleanup]${NC} MC start_app HTTP failed."
            echo -e "${YELLOW}  Manually restart with:${NC}"
            echo -e "  curl -X POST $MC_EM_URL/json/start_app \\"
            echo -e "       -H 'Content-Type: application/json' -d '{\"app_name\":\"mc\"}'"
        fi
    fi
    exit $rc
}

# ============================================================================
# Step 1: Pre-flight (robot for local/onbot, MuJoCo bridge + DDS for sim)
# ============================================================================

if [[ "$MODE" == "sim" ]]; then
    echo -e "$(ts) ${BLUE}[Step 1/4]${NC} Sim pre-flight"

    echo "  Bridge script:     $SCRIPT_DIR/$SIM_BRIDGE_REL"
    echo -n "  Python interpreter ($SIM_PYTHON) ... "
    if command -v "$SIM_PYTHON" &>/dev/null; then
        PYV="$($SIM_PYTHON -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown)"
        echo -e "${GREEN}found ($PYV)${NC}"
    else
        echo -e "${RED}not found${NC}"
        echo -e "${YELLOW}  Install Python 3 or pass --sim-python /path/to/python${NC}"
        exit 1
    fi

    echo -n "  Bridge import smoke test ... "
    if "$SIM_PYTHON" - <<'PY' >/dev/null 2>&1
import importlib
for m in ("mujoco", "rclpy", "aimdk_msgs.msg", "sensor_msgs.msg", "numpy"):
    importlib.import_module(m)
PY
    then
        echo -e "${GREEN}ok${NC}"
    else
        echo -e "${YELLOW}missing one of mujoco / rclpy / aimdk_msgs / sensor_msgs / numpy${NC}"
        echo -e "${YELLOW}  The bridge will fail on launch. Source your ROS 2 install/setup.bash"
        echo -e "  and pip install mujoco numpy if needed.${NC}"
    fi

    echo "  Isolating DDS to loopback (no traffic to/from real robot):"
    echo "    ROS_LOCALHOST_ONLY=1   ROS_DOMAIN_ID=$SIM_DOMAIN_ID"
    export ROS_LOCALHOST_ONLY=1
    export ROS_DOMAIN_ID="$SIM_DOMAIN_ID"
    echo ""
else
    echo -e "$(ts) ${BLUE}[Step 1/4]${NC} Robot pre-flight"

    if [[ -z "$ROBOT_HOST" ]]; then
        echo -e "${RED}no PC2 address${NC}"
        echo -e "${YELLOW}  Set PC2_HOST=<PC2_IP> in the environment (x2_discover_network.sh"
        echo -e "  writes ~/.x2/env.wired) or pass --robot-host <PC2_IP>.${NC}"
        exit 1
    fi
    echo -n "  Pinging $ROBOT_HOST ... "
    if ping -c 1 -W 2 "$ROBOT_HOST" &>/dev/null; then
        echo -e "${GREEN}reachable${NC}"
    else
        echo -e "${RED}unreachable${NC}"
        echo -e "${YELLOW}  Check the SDK ethernet cable and your laptop NIC IP."
        echo -e "  Per the AgiBot quick-start prerequisites: laptop should be"
        echo -e "  static <LAPTOP_IP>/24 on the SDK subnet, robot dev unit (PC2) at <PC2_IP>.${NC}"
        exit 1
    fi

    if command -v ros2 &>/dev/null; then
        # Cache the topic list once; the joint and IMU checks both grep it.
        TOPIC_LIST="$(timeout 5 ros2 topic list 2>/dev/null || true)"

        echo -n "  Checking ROS 2 joint topic visibility ... "
        if echo "$TOPIC_LIST" | grep -q "/aima/hal/joint/leg/state"; then
            echo -e "${GREEN}visible${NC}"
        else
            echo -e "${YELLOW}not visible (DDS may need a moment to discover)${NC}"
        fi

        echo -n "  Checking ROS 2 IMU topic visibility ... "
        if echo "$TOPIC_LIST" | grep -q "/aima/hal/imu/torso/state"; then
            echo -e "${GREEN}visible (torso)${NC}"
        elif echo "$TOPIC_LIST" | grep -q "/aima/hal/imu/torse/state"; then
            echo -e "${YELLOW}firmware uses 'torse' typo${NC}"
            echo -e "${YELLOW}    -> auto-adding --imu-topic /aima/hal/imu/torse/state${NC}"
            if [[ -z "$IMU_TOPIC" ]]; then
                IMU_TOPIC="/aima/hal/imu/torse/state"
                ROS2_ARGS+=("--imu-topic" "$IMU_TOPIC")
            fi
        else
            echo -e "${YELLOW}not visible (DDS may need a moment to discover)${NC}"
        fi
    else
        echo -e "${YELLOW}  ros2 not in PATH; skipping topic visibility check${NC}"
    fi

    # ────────────────────────────────────────────────────────────────
    # Gantry-aware Python pre-flight. Runs the dedicated x2_preflight.py
    # which audits joint pose/vel/effort, IMU upright/quiet, MC presence,
    # and topic publishers. Heavyweight version of the lightweight ros2
    # checks above. Placed BEFORE the MC stop on purpose: a failing
    # preflight aborts while the robot is still held by MC.
    # ────────────────────────────────────────────────────────────────
    PREFLIGHT_SCRIPT="$SCRIPT_DIR/scripts/x2_preflight.py"
    if $NO_PREFLIGHT_PY; then
        echo -e "${YELLOW}  --no-preflight-py: skipping gantry-aware preflight${NC}"
    elif [[ ! -f "$PREFLIGHT_SCRIPT" ]]; then
        echo -e "${YELLOW}  preflight script not found at $PREFLIGHT_SCRIPT; skipping${NC}"
    else
        echo ""
        echo -e "${BLUE}  Running gantry-aware preflight (x2_preflight.py) ...${NC}"
        PREFLIGHT_CMD=("$SIM_PYTHON" "$PREFLIGHT_SCRIPT")
        # Mirror IMU topic decision from the visibility check above so
        # firmware shipping with the 'torse' typo doesn't false-fail.
        if [[ -n "$IMU_TOPIC" ]]; then
            PREFLIGHT_CMD+=("--imu-topic" "$IMU_TOPIC")
        fi
        if $PREFLIGHT_STRICT; then
            PREFLIGHT_CMD+=("--strict-pose" "--strict-effort")
        fi
        if [[ -n "$PREFLIGHT_ARGS" ]]; then
            # shellcheck disable=SC2206  # intentional word-split for passthrough
            EXTRA_PREFLIGHT_ARGS=($PREFLIGHT_ARGS)
            PREFLIGHT_CMD+=("${EXTRA_PREFLIGHT_ARGS[@]}")
        fi
        echo "    \$ ${PREFLIGHT_CMD[*]}"
        if "${PREFLIGHT_CMD[@]}"; then
            echo -e "${GREEN}  Preflight PASS.${NC}"
        else
            PREFLIGHT_RC=$?
            echo -e "${RED}  Preflight FAILED (exit $PREFLIGHT_RC).${NC}"
            echo -e "${YELLOW}  Aborting BEFORE MC stop; robot is unchanged.${NC}"
            echo -e "${YELLOW}  Re-run with --preflight-args to relax thresholds, or${NC}"
            echo -e "${YELLOW}  --no-preflight-py to bypass entirely (operator override).${NC}"
            exit "$PREFLIGHT_RC"
        fi
        echo ""
    fi

    if ! $NO_STOP_MC; then
        # ────────────────────────────────────────────────────────────────
        # PRE-HANDOFF GATE: require MC in STAND_DEFAULT.
        #
        # Smooth handoff is built around the contract "we start in
        # STAND_DEFAULT and we end in STAND_DEFAULT". If MC is currently
        # in PASSIVE_DEFAULT / DAMPING_DEFAULT / a fault state, the
        # robot is already on the gantry / floor with zero or low
        # torque -- there is nothing to "hand off" from, and the
        # captured stand-pose YAML (which assumes MC was actively
        # balancing) is the wrong target. Refuse to proceed and tell
        # the operator how to recover. Operator override:
        # --no-require-stand-default.
        # ────────────────────────────────────────────────────────────────
        if $REQUIRE_STAND_DEFAULT; then
            echo ""
            echo -n "  Pre-handoff: MC mode = "
            CURRENT_MC_MODE="$(mc_get_action 3 2>/dev/null || true)"
            if [[ -z "$CURRENT_MC_MODE" ]]; then
                echo -e "${YELLOW}<unreachable>${NC}"
            else
                echo -e "${BLUE}$CURRENT_MC_MODE${NC}"
            fi

            if [[ "$CURRENT_MC_MODE" == "STAND_DEFAULT" ]]; then
                echo -e "${GREEN}  Smooth-handoff contract satisfied: start in STAND_DEFAULT,${NC}"
                echo -e "${GREEN}  end in STAND_DEFAULT (HOLD_FOR_MC bridges the gap).${NC}"
            else
                echo ""
                echo -e "${YELLOW}  MC is not in STAND_DEFAULT. The smooth handoff requires MC${NC}"
                echo -e "${YELLOW}  to be actively balancing the robot before we take the bus.${NC}"
                echo -e "${YELLOW}  Walking MC up the mode chain to STAND_DEFAULT now (each${NC}"
                echo -e "${YELLOW}  transition asks for confirmation).${NC}"
                echo ""
                if ! escalate_mc_to_stand_default; then
                    echo ""
                    echo -e "${RED}  ERROR: could not get MC into STAND_DEFAULT.${NC}" >&2
                    echo -e "${YELLOW}  Resolve the issue above (gantry height, MC fault, etc.)${NC}" >&2
                    echo -e "${YELLOW}  and re-run, OR bypass this gate with:${NC}" >&2
                    echo -e "${YELLOW}    --no-require-stand-default${NC}" >&2
                    echo -e "${YELLOW}  (only safe if you know the robot is in a recoverable state).${NC}" >&2
                    exit 1
                fi
                echo ""
                echo -e "${GREEN}  Smooth-handoff contract satisfied: start in STAND_DEFAULT,${NC}"
                echo -e "${GREEN}  end in STAND_DEFAULT (HOLD_FOR_MC bridges the gap).${NC}"
            fi
        else
            echo -e "${YELLOW}  --no-require-stand-default: skipping STAND_DEFAULT gate${NC}"
        fi

        # NOTE: the SAFETY GATE prompt + stop_app POST that used to live
        # here have moved to Step 4 (after the colcon build), so we can
        # spawn the deploy in STANDBY ahead of the prompt. With that
        # reordering "Y -> CONTROL" goes from ~14 s (1 s stop_app + 7 s
        # build + 5 s autostart) to ~1 s (stop_app + verify + sentinel
        # touch + one OnControl tick). See do_stop_mc_and_trigger_deploy
        # in this script for the post-build path.
        :  # no-op; pre-handoff gate above stays as is
    else
        echo -e "${YELLOW}  --no-stop-mc: skipping STAND_DEFAULT pre-handoff gate${NC}"
    fi
    echo ""
fi

# ============================================================================
# Step 2: Asset checks (--model and --motion exist where this script runs)
# ============================================================================

echo -e "$(ts) ${BLUE}[Step 2/4]${NC} Asset checks"

check_local_file() {
    if [[ -e "$1" ]]; then
        echo -e "  ${GREEN}✅${NC} $2: $1"
        return 0
    else
        echo -e "  ${RED}❌${NC} $2 not found: $1"
        return 1
    fi
}

MISSING=0
check_local_file "$MODEL" "Model"      || MISSING=$((MISSING+1))
# --motion is meaningless once VLA mode is on (deploy reads pose refs
# from ZMQ, not from the .x2m2 file); we tolerate it being absent then.
if [[ -n "$MOTION" ]] && ! $VLA_MODE; then
    check_local_file "$MOTION"  "Motion"  || MISSING=$((MISSING+1))
fi
if [[ "$MODE" == "sim" ]]; then
    check_local_file "$SCRIPT_DIR/$SIM_BRIDGE_REL" "MuJoCo bridge" \
        || MISSING=$((MISSING+1))
    [[ -n "$SIM_MJCF" ]] && { check_local_file "$SIM_MJCF" "MJCF override" \
        || MISSING=$((MISSING+1)); }
    [[ -n "$SIM_MOTION" ]] && { check_local_file "$SIM_MOTION" "Sim RSI motion" \
        || MISSING=$((MISSING+1)); }
fi
if [[ "$MODE" == "onbot" ]]; then
    # New onbot semantics: we ARE on PC2, so file paths are local. Verify
    # the bringup-staged install layout too so a missing pc2_bringup.sh
    # run fails fast with a clear hint instead of cryptically later.
    check_local_file "$ONBOT_WS/install/setup.bash" "onbot workspace install/setup.bash" \
        || MISSING=$((MISSING+1))
    check_local_file "$ONBOT_VENV/bin/python3" "onbot venv python" \
        || MISSING=$((MISSING+1))
    check_local_file "$ONBOT_ONNXRUNTIME/lib/libonnxruntime.so" "onbot ONNX Runtime lib" \
        || MISSING=$((MISSING+1))
    check_local_file "$ONBOT_AIMDK_PREFIX/local/lib/python3.10/dist-packages" \
        "onbot aimdk_msgs python bindings dir" \
        || MISSING=$((MISSING+1))
fi
if [[ $MISSING -gt 0 ]]; then
    echo -e "${RED}  $MISSING asset(s) missing. Aborting.${NC}"
    if [[ "$MODE" == "onbot" ]]; then
        echo -e "${YELLOW}  If onbot paths are missing, run pc2_bringup.sh from the${NC}"
        echo -e "${YELLOW}  laptop first: ./gear_sonic_deploy/scripts/pc2_bringup.sh${NC}"
        echo -e "${YELLOW}    --pc2-host <PC2_IP> --model /path/to/policy.onnx${NC}"
    fi
    exit 1
fi
echo ""

# ============================================================================
# Step 3: Build (colcon)
# ============================================================================

echo -e "$(ts) ${BLUE}[Step 3/4]${NC} Build"

build_local() {
    if $NO_BUILD; then
        echo -e "${YELLOW}  --no-build set; skipping colcon build.${NC}"
        return 0
    fi
    if ! command -v colcon &>/dev/null; then
        echo -e "${RED}  colcon not in PATH. Source your ROS 2 setup.bash first.${NC}"
        exit 1
    fi
    echo "  Building $PKG_NAME locally with colcon ..."
    # The root gear_sonic_deploy/CMakeLists.txt ('g1_deploy') is itself
    # discoverable by colcon and shadows everything beneath it (colcon does
    # not descend into a directory once it finds a package). Restrict the
    # search to the X2 package's tree so colcon actually finds it.
    colcon build --packages-select "$PKG_NAME" \
        --base-paths "$PKG_DIR_REL" \
        --cmake-args -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
    echo -e "${GREEN}  Local build OK.${NC}"
}

build_onbot() {
    # New onbot semantics (2026-05): this script runs natively on PC2,
    # so there is no rsync + ssh build path. The colcon workspace at
    # $ONBOT_WS/install/setup.bash is staged by pc2_bringup.sh and we
    # just verify it exists. To rebuild after C++ changes, re-run
    # pc2_bringup.sh from the laptop (which rsyncs + colcons on PC2)
    # rather than building from here.
    if [[ ! -f "$ONBOT_WS/install/setup.bash" ]]; then
        echo -e "${RED}  onbot workspace not found at $ONBOT_WS/install/setup.bash${NC}"
        echo -e "${YELLOW}  This script must be run on PC2 with the workspace pre-built.${NC}"
        echo -e "${YELLOW}  From the laptop run:${NC}"
        echo -e "${YELLOW}    ./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host <PC2_IP>${NC}"
        echo -e "${YELLOW}  then re-run this script ON PC2.${NC}"
        exit 1
    fi
    if $NO_BUILD; then
        echo -e "${YELLOW}  --no-build set; (onbot is build-free anyway -- noop).${NC}"
    else
        echo -e "${GREEN}  onbot workspace OK at $ONBOT_WS/install/setup.bash${NC}"
        echo "  (Skipping colcon build -- run pc2_bringup.sh to rebuild after C++ changes.)"
    fi
}

if [[ "$MODE" == "onbot" ]]; then
    build_onbot
    # New onbot: paths are LOCAL on PC2; no remote rewrite of --model / --motion.
else
    build_local
fi
echo ""

if $BUILD_ONLY; then
    echo -e "${GREEN}--build-only set; exiting without running.${NC}"
    exit 0
fi

# ============================================================================
# Step 4: Display configuration + confirm + run
# ============================================================================

echo -e "$(ts) ${BLUE}[Step 4/4]${NC} Ready to launch"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                       DEPLOYMENT CONFIGURATION                        ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Mode:               ${GREEN}$MODE${NC}"
echo -e "  Model:              ${GREEN}${ROS2_ARGS[1]}${NC}"
if [[ -n "$MOTION_SOURCE" ]]; then
    echo -e "  Motion source:      ${GREEN}$MOTION_SOURCE${NC}"
    echo -e "  Motion (baked):     ${GREEN}$MOTION${NC} ${CYAN}(per-run tempdir)${NC}"
elif [[ -n "$MOTION" ]]; then
    echo -e "  Motion:             ${GREEN}$MOTION${NC}"
fi
[[ -n "$LOG_DIR" ]]     && echo -e "  Log dir:            ${GREEN}$LOG_DIR${NC}"
$RECORD_RUN             && echo -e "  Record run npz:     ${GREEN}$RECORD_OUT${NC}"
[[ -n "$TUNING_CONFIG" ]] && echo -e "  Tuning preset:      ${GREEN}$(basename "$TUNING_CONFIG")${NC}"
[[ -n "$AUTOSTART" ]]   && echo -e "  Autostart (s):      ${GREEN}$AUTOSTART${NC}"
[[ -n "$MAX_DURATION" ]] && echo -e "  Max duration (s):   ${GREEN}$MAX_DURATION${NC}"
[[ -n "$TILT_COS" ]]    && echo -e "  Tilt cos thresh:    ${GREEN}$TILT_COS${NC}"
[[ -n "$RAMP_SECONDS" ]] && echo -e "  Ramp (s):           ${GREEN}$RAMP_SECONDS${NC}"
[[ -n "$MAX_TARGET_DEV" ]] && echo -e "  Max target dev (rad): ${GREEN}$MAX_TARGET_DEV${NC}"
[[ -n "$ACTION_CLIP" ]]    && echo -e "  Action clip (rad):    ${GREEN}$ACTION_CLIP${NC}"
[[ -n "$RETURN_SECONDS" ]] && echo -e "  Return ramp (s):      ${GREEN}$RETURN_SECONDS${NC}"
[[ -n "$IMU_TOPIC" ]]   && echo -e "  IMU topic:          ${GREEN}$IMU_TOPIC${NC}"
[[ -n "$OBS_DUMP" ]]    && echo -e "  ${YELLOW}OBS-DUMP${NC} -> ${GREEN}$OBS_DUMP${NC} (will exit after first tick)"
$DRY_RUN                && echo -e "  ${YELLOW}DRY-RUN${NC} (stiffness=damping=0; no torque)"
if [[ "$MODE" == "sim" ]]; then
    echo -e "  Sim driver:         ${GREEN}MuJoCo bridge (closed-loop)${NC}"
    echo -e "  Bridge:             ${GREEN}$SCRIPT_DIR/$SIM_BRIDGE_REL${NC}"
    [[ -n "$SIM_MJCF" ]]   && echo -e "  MJCF override:      ${GREEN}$SIM_MJCF${NC}"
    [[ -n "$SIM_MOTION" ]] && echo -e "  RSI motion:         ${GREEN}$SIM_MOTION${NC}"
    [[ -n "$SIM_INIT_FRAME" ]] && echo -e "  RSI frame:          ${GREEN}$SIM_INIT_FRAME${NC}"
    [[ -n "$SIM_IMU_FROM" ]] && echo -e "  IMU from:           ${GREEN}$SIM_IMU_FROM${NC}"
    [[ -n "$SIM_HOLD_STIFFNESS_MULT" ]] && \
        echo -e "  Hold stiffness x:   ${GREEN}$SIM_HOLD_STIFFNESS_MULT${NC}"
    [[ -n "$SIM_INIT_POSE" ]]   && echo -e "  Init pose:          ${GREEN}$SIM_INIT_POSE${NC}"
    [[ -n "$SIM_BAND_LENGTH" ]] && echo -e "  Band length:        ${GREEN}${SIM_BAND_LENGTH}m${NC}"
    [[ -n "$SIM_BAND_KP_MULT" ]] && echo -e "  Band kp mult:       ${GREEN}$SIM_BAND_KP_MULT${NC}"
    if $SIM_NO_ELASTIC_BAND; then
        echo -e "  ElasticBand:        ${YELLOW}disabled${NC}"
    elif $SIM_VIEWER; then
        echo -e "  ElasticBand:        ${GREEN}ON (viewer: 9 toggle, 7/8 raise/lower)${NC}"
    else
        echo -e "  ElasticBand:        ${GREEN}ON (auto-release ${SIM_BAND_RELEASE_AFTER_S:-1.0}s after 1st cmd)${NC}"
    fi
    [[ -n "$SIM_DT" ]] && echo -e "  Physics dt:         ${GREEN}${SIM_DT}s${NC}"
    $SIM_VIEWER         && echo -e "  Viewer:             ${GREEN}yes${NC}"
    $SIM_PRINT_SCENE    && echo -e "  Print scene:        ${GREEN}yes${NC}"
    [[ -n "$SIM_RECORD_COMMANDS" ]] && \
        echo -e "  Record commands:    ${GREEN}$SIM_RECORD_COMMANDS${NC}"
    echo -e "  DDS isolation:      ${GREEN}ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID=$SIM_DOMAIN_ID${NC}"
fi
if [[ "$MODE" == "local" ]]; then
    if $HAND_BRIDGE_ENABLED && [[ "$HAND_BRIDGE_SIDES" != "off" ]]; then
        echo -e "  Hand bridge:        ${GREEN}ON${NC} (py=${HAND_BRIDGE_PYTHON:-$DEPLOY_AUX_PYTHON}, sides=$HAND_BRIDGE_SIDES, engage=${HAND_BRIDGE_ENGAGE_SHOTS}x@1Hz, publish=${HAND_BRIDGE_PUBLISH_HZ}Hz, max_stale=${HAND_BRIDGE_MAX_STALE_S}s)"
    else
        echo -e "  Hand bridge:        ${YELLOW}OFF${NC} (real-robot hands will not move)"
    fi
elif [[ "$MODE" == "onbot" ]]; then
    echo -e "  Hand bridge:        ${YELLOW}skipped${NC} (onbot mode -- owned by x2_pc2_daemons.sh in its own tmux session)"
    echo -e "  Recorder:           ${YELLOW}skipped${NC} (onbot mode -- recorder lives on the laptop with the planner stack)"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}The following command will be executed:${NC}"
echo ""
if [[ "$MODE" == "onbot" ]]; then
    echo -e "${BLUE}source /opt/ros/humble/setup.bash${NC}"
    echo -e "${BLUE}source $ONBOT_WS/install/setup.bash${NC}"
    echo -e "${BLUE}ros2 run $PKG_NAME x2_deploy_onnx_ref \\"
    for ((i=0; i<${#ROS2_ARGS[@]}; i++)); do
        if [[ $((i+1)) -lt ${#ROS2_ARGS[@]} ]]; then
            echo -e "${BLUE}    ${ROS2_ARGS[i]} \\"
        else
            echo -e "${BLUE}    ${ROS2_ARGS[i]}${NC}"
        fi
    done
else
    echo -e "${BLUE}source install/setup.bash${NC}"
    echo -e "${BLUE}ros2 run $PKG_NAME x2_deploy_onnx_ref \\"
    for ((i=0; i<${#ROS2_ARGS[@]}; i++)); do
        if [[ $((i+1)) -lt ${#ROS2_ARGS[@]} ]]; then
            echo -e "${BLUE}    ${ROS2_ARGS[i]} \\"
        else
            echo -e "${BLUE}    ${ROS2_ARGS[i]}${NC}"
        fi
    done
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ "$MODE" != "sim" ]]; then
    if $DRY_RUN; then
        echo -e "$(ts) ${YELLOW}DRY-RUN${NC}: pipeline runs but no torque will be applied."
    else
        echo -e "$(ts) ${RED}WARNING${NC}: this will issue REAL torque commands to the X2 Ultra."
        echo -e "$(ts) ${RED}        ${NC} Robot must be on a gantry / supported. E-stop within reach."
    fi
else
    echo -e "${YELLOW}SIM mode: closed-loop MuJoCo. Bridge applies the deploy's PD${NC}"
    echo -e "${YELLOW}    commands as torques. Robot may fall if the policy isn't well-${NC}"
    echo -e "${YELLOW}    behaved -- that's the point. Use --sim-viewer to watch.${NC}"
fi
echo ""

# Sim mode keeps a one-line confirmation here (it has no MC to stop, so
# the merged gate that real-robot mode runs below doesn't apply).
# Real-robot mode (local/onbot) goes through the bg-launch + sentinel
# safety gate further down -- there is no separate confirmation here.
if ! $NO_CONFIRM; then
    if [[ "$MODE" == "sim" ]]; then
        read -p "$(echo -e ${GREEN}Proceed with sim launch? [Y/n]: ${NC})" confirm
        if [[ -n "$confirm" && ! "$confirm" =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Cancelled before launch.${NC}"
            exit 0
        fi
    fi
fi

echo ""
echo -e "$(ts) ${GREEN}Launching ...${NC}"
echo ""

# --record: spin the npz recorder up BEFORE the deploy / bridge so the
# recording covers the full STANDBY -> WAIT -> CONTROL -> RAMP_OUT window.
# The cleanup traps below (cleanup_sim for sim mode, restart_mc_on_exit
# for local/onbot) both call stop_run_recorder so a clean .npz is dumped
# even on Ctrl-C. In sim mode the bridge starts further down and the
# recorder picks up its /aima publishers as soon as they appear -- no
# need to interleave starts.
start_run_recorder

if [[ "$MODE" == "sim" ]]; then
    if [[ -f install/setup.bash ]]; then
        source install/setup.bash
    fi

    # Install the cleanup trap NOW (before the bridge launches) so the
    # already-running --record recorder is reaped on Ctrl-C in the gap
    # between recorder start and bridge start. cleanup_sim no-ops on an
    # empty SIM_BRIDGE_PID, so installing it early is safe.
    trap cleanup_sim INT TERM EXIT

    # Background the MuJoCo bridge first so it's ready before the deploy
    # starts polling /aima topics.
    BRIDGE_ARGS=()
    [[ -n "$SIM_MJCF" ]]                 && BRIDGE_ARGS+=("--mjcf" "$SIM_MJCF")
    [[ -n "$SIM_MOTION" ]]               && BRIDGE_ARGS+=("--motion" "$SIM_MOTION")
    [[ -n "$SIM_INIT_FRAME" ]]           && BRIDGE_ARGS+=("--init-frame" "$SIM_INIT_FRAME")
    [[ -n "$SIM_IMU_FROM" ]]             && BRIDGE_ARGS+=("--imu-from" "$SIM_IMU_FROM")
    [[ -n "$SIM_HOLD_STIFFNESS_MULT" ]]  && BRIDGE_ARGS+=("--hold-stiffness-mult" "$SIM_HOLD_STIFFNESS_MULT")
    [[ -n "$SIM_INIT_POSE" ]]            && BRIDGE_ARGS+=("--init-pose" "$SIM_INIT_POSE")
    [[ -n "$SIM_BAND_LENGTH" ]]          && BRIDGE_ARGS+=("--band-length" "$SIM_BAND_LENGTH")
    [[ -n "$SIM_BAND_KP_MULT" ]]         && BRIDGE_ARGS+=("--band-kp-mult" "$SIM_BAND_KP_MULT")
    $SIM_NO_ELASTIC_BAND                 && BRIDGE_ARGS+=("--no-elastic-band")
    [[ -n "$SIM_BAND_RELEASE_AFTER_S" ]] && BRIDGE_ARGS+=("--band-release-after-s" "$SIM_BAND_RELEASE_AFTER_S")
    [[ -n "$SIM_DT" ]]                   && BRIDGE_ARGS+=("--sim-dt" "$SIM_DT")
    BRIDGE_ARGS+=("--ros-domain-id" "$SIM_DOMAIN_ID")
    # SIM_BRIDGE_EXTRA: whitespace-separated extra flags for the MuJoCo bridge,
    # for probes that don't warrant a dedicated deploy_x2.sh flag. Intentionally
    # unquoted so multiple flags split. E.g. the chest-nudge disturbance probe:
    #   SIM_BRIDGE_EXTRA="--nudge-force 40 --nudge-period 4"
    # shellcheck disable=SC2206
    [[ -n "${SIM_BRIDGE_EXTRA:-}" ]] && BRIDGE_ARGS+=(${SIM_BRIDGE_EXTRA})
    $SIM_VIEWER                          && BRIDGE_ARGS+=("--viewer")
    [[ -n "$SIM_CAM_TRACK_BODY" ]]       && BRIDGE_ARGS+=("--cam-track-body" "$SIM_CAM_TRACK_BODY")
    [[ -n "$SIM_CAM_DISTANCE" ]]         && BRIDGE_ARGS+=("--cam-distance" "$SIM_CAM_DISTANCE")
    [[ -n "$SIM_CAM_ELEVATION" ]]        && BRIDGE_ARGS+=("--cam-elevation" "$SIM_CAM_ELEVATION")
    [[ -n "$SIM_CAM_AZIMUTH" ]]          && BRIDGE_ARGS+=("--cam-azimuth" "$SIM_CAM_AZIMUTH")
    $SIM_PRINT_SCENE                     && BRIDGE_ARGS+=("--print-scene")
    $SIM_WITH_OMNIHAND                   && BRIDGE_ARGS+=("--with-omnihand")
    [[ -n "$SIM_HAND_ZMQ_HOST" ]]        && BRIDGE_ARGS+=("--hand-zmq-host" "$SIM_HAND_ZMQ_HOST")
    [[ -n "$SIM_HAND_ZMQ_PORT" ]]        && BRIDGE_ARGS+=("--hand-zmq-port" "$SIM_HAND_ZMQ_PORT")
    [[ -n "$SIM_HAND_ZMQ_TOPIC" ]]       && BRIDGE_ARGS+=("--hand-zmq-topic" "$SIM_HAND_ZMQ_TOPIC")
    $SIM_NO_HAND_ZMQ                     && BRIDGE_ARGS+=("--no-hand-zmq")

    if [[ "${X2_WORLD:-mujoco}" == "isaaclab" ]]; then
        # IsaacLab world topology (physics outside this container, DDS<->ZMQ
        # adapter in here) is NOT part of this release: the adapter and the
        # IsaacLab bridge are not shipped. Only the MuJoCo world is supported.
        echo -e "$(ts) ${RED}[sim]${NC} X2_WORLD=isaaclab is not available in this release (x2_dds_zmq_adapter.py / x2_isaaclab_bridge.py not shipped); unset X2_WORLD to use MuJoCo." >&2
        exit 1
    else
    echo -e "$(ts) ${BLUE}[sim]${NC} backgrounding: $SIM_PYTHON $SIM_BRIDGE_REL ${BRIDGE_ARGS[*]}"
    "$SIM_PYTHON" "$SCRIPT_DIR/$SIM_BRIDGE_REL" "${BRIDGE_ARGS[@]}" &
    SIM_BRIDGE_PID=$!
    fi
    # cleanup_sim trap was installed above (before bridge launch).

    if [[ -n "$SIM_RECORD_COMMANDS" ]]; then
        echo -e "$(ts) ${BLUE}[sim]${NC} backgrounding: ros2 bag record -> $SIM_RECORD_COMMANDS"
        ros2 bag record -o "$SIM_RECORD_COMMANDS" \
            /aima/hal/joint/leg/command \
            /aima/hal/joint/waist/command \
            /aima/hal/joint/arm/command \
            /aima/hal/joint/head/command &
        SIM_RECORD_PID=$!
    fi

    # Give the bridge a moment to load MuJoCo + start publishing so the
    # deploy doesn't time out in INIT before the first state arrives.
    sleep 2
    if ! kill -0 "$SIM_BRIDGE_PID" 2>/dev/null; then
        echo -e "$(ts) ${RED}[sim]${NC} MuJoCo bridge exited immediately; aborting" >&2
        echo -e "${RED}      Re-run with --sim-print-scene for diagnostics.${NC}" >&2
        exit 1
    fi

    echo -e "$(ts) ${BLUE}[sim]${NC} starting deploy (cleanup trap installed; Ctrl-C to stop)"
    ros2 run "$PKG_NAME" x2_deploy_onnx_ref "${ROS2_ARGS[@]}"
else
    # local + onbot share the cold-warm orchestration below. The only
    # asymmetry is which colcon overlay we source: local uses the
    # repo-root install/ (built by build_local), onbot already sourced
    # the bringup-staged $ONBOT_WS/install/setup.bash way up at the
    # top of the script (right after mode validation, so preflight +
    # the tuning translator could see ros2/rclpy/aimdk_msgs). We just
    # verify the overlay registered the deploy node here.
    if [[ "$MODE" == "onbot" ]]; then
        if ! ros2 pkg prefix "$PKG_NAME" >/dev/null 2>&1; then
            echo -e "$(ts) ${RED}[onbot]${NC} colcon overlay missing $PKG_NAME at runtime."
            echo -e "$(ts) ${RED}[onbot]${NC} overlay was sourced from $ONBOT_WS/install/setup.bash;"
            echo -e "$(ts) ${RED}[onbot]${NC} re-run pc2_bringup.sh with --force-build to rebuild."
            exit 1
        fi
    else
        if [[ -f install/setup.bash ]]; then
            source install/setup.bash
        fi
    fi
    # NOT `exec` -- we need the EXIT trap (restart_mc_on_exit) to fire after
    # the deploy returns / Ctrl-C. The trap re-exit()s with the deploy's
    # status so callers / CI still see the right code.

    # ────────────────────────────────────────────────────────────────────
    # Cold-warm boot orchestration:
    #   1. Spawn deploy in background. It boots into STANDBY (writer
    #      suppressed; no joint commands published), so it does NOT
    #      contend with MC for the bus while the operator considers
    #      the safety prompt.
    #   2. Wait for the ready-sentinel that deploy touches once
    #      subscribers + ONNX + takeover detectors are armed.
    #   3. Show the merged safety gate prompt.
    #   4. On 'Y': stop_app POST + verify MC is silent + touch the
    #      start-trigger-sentinel. Deploy advances STANDBY -> INIT
    #      (state already fresh from MC's continuous broadcasts) ->
    #      WAIT_FOR_CONTROL -> CONTROL within ~20 ms.
    #   5. Fall through to the existing HOLD_FOR_MC handoff loop (poll
    #      for hold-for-mc sentinel, POST start_app, escalate to STAND,
    #      reap deploy).
    # ────────────────────────────────────────────────────────────────────
    DEPLOY_PID=""
    cleanup_bg_deploy_on_cancel() {
        # Used for the cancel paths (operator says 'n' or Ctrl-C's
        # before stop_app). At that point MC is still running and the
        # bg deploy is in STANDBY (silent) -- safe to SIGTERM it. Does
        # NOT touch MC.
        #
        # IMPORTANT: stop the run recorder FIRST so its npz is flushed
        # to disk before this shell exits. The recorder subscribes to
        # MC's /aima/hal/joint/*/state and /command topics from the
        # moment start_run_recorder runs (well before the safety gate),
        # so the npz captures the pre-'y' window where MC is the only
        # active publisher. Without this call, Ctrl-C at the gate kills
        # the recorder process without giving it a chance to save the
        # npz -- which is why /scratch/runs/x2_run_2026..../run.npz
        # was missing for runs the operator aborted at the gate on
        # 2026-05-03 (waist-runaway emergency-stop incident).
        stop_run_recorder
        if [[ -n "$DEPLOY_PID" ]] && kill -0 "$DEPLOY_PID" 2>/dev/null; then
            echo -e "$(ts) ${YELLOW}[handoff]${NC} cancelling: SIGTERM-ing background deploy (pid $DEPLOY_PID)."
            kill -TERM "$DEPLOY_PID" 2>/dev/null || true
            wait "$DEPLOY_PID" 2>/dev/null || true
        fi
        if [[ -n "${DEPLOY_TAIL_PID:-}" ]] && kill -0 "$DEPLOY_TAIL_PID" 2>/dev/null; then
            kill -TERM "$DEPLOY_TAIL_PID" 2>/dev/null || true
            wait "$DEPLOY_TAIL_PID" 2>/dev/null || true
        fi
    }

    if [[ -n "$START_TRIGGER_SENTINEL" && -n "$READY_SENTINEL" ]]; then
        echo -e "$(ts) ${BLUE}[handoff]${NC} STANDBY trigger sentinel:    $START_TRIGGER_SENTINEL"
        echo -e "$(ts) ${BLUE}[handoff]${NC} STANDBY ready sentinel:      $READY_SENTINEL"
        if [[ -n "$HOLD_FOR_MC_SENTINEL" ]]; then
            echo -e "$(ts) ${BLUE}[handoff]${NC} HOLD_FOR_MC sentinel:        $HOLD_FOR_MC_SENTINEL"
            echo -e "$(ts) ${BLUE}[handoff]${NC} HOLD_FOR_MC exit-sentinel:   $HOLD_FOR_MC_EXIT_SENTINEL"
            echo -e "$(ts) ${BLUE}[handoff]${NC} MC first-publish sentinel:   $MC_FIRST_PUBLISH_SENTINEL"
        fi
        # Tee deploy's stdout+stderr to a per-run log file AND back to
        # this terminal so its [INFO]/[WARN] lines don't corrupt the
        # safety-gate `read -p` prompt (the previous run had AimdkIo
        # joint-validation lines bleed into the prompt at 20:32:28).
        # The tee is line-buffered to keep ordering with bash echos.
        if [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]]; then
            DEPLOY_STDOUT_LOG="$RUN_LOG_DIR/deploy_stdout.log"
        else
            DEPLOY_STDOUT_LOG="/tmp/x2_deploy_stdout.$$.log"
        fi
        echo -e "$(ts) ${BLUE}[handoff]${NC} spawning deploy in STANDBY (writer suppressed; stdout -> $DEPLOY_STDOUT_LOG) ..."
        # 2>&1 first, then route via a coproc-style redirect so we
        # don't echo deploy's INFO chatter onto the controlling tty
        # while the operator is being prompted. After the trigger fires
        # we'll switch to a tail -f style follow so the operator sees
        # the CONTROL ticks live.
        ros2 run "$PKG_NAME" x2_deploy_onnx_ref "${ROS2_ARGS[@]}" \
            >"$DEPLOY_STDOUT_LOG" 2>&1 &
        DEPLOY_PID=$!

        # Trap to clean up the bg deploy if the user denies / Ctrl-Cs the
        # safety gate. Replaced with restart_mc_on_exit once stop_app fires.
        trap 'cleanup_bg_deploy_on_cancel; exit 130' INT TERM

        # Block until ready-sentinel appears (deploy's first STANDBY tick
        # touches it) OR deploy died early. 30 s upper bound is plenty:
        # ONNX load + DDS discovery + subscriber settle is ~0.5 s in
        # practice; the slack covers slow disks / cold caches.
        echo -e "$(ts) ${BLUE}[handoff]${NC} waiting for deploy ready-sentinel ..."
        READY_DEADLINE=$(( SECONDS + 30 ))
        while [[ ! -f "$READY_SENTINEL" ]]; do
            if ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
                echo -e "$(ts) ${RED}[handoff]${NC} deploy exited before reaching STANDBY-ready (pid $DEPLOY_PID)."
                wait "$DEPLOY_PID" 2>/dev/null || true
                exit 1
            fi
            if (( SECONDS >= READY_DEADLINE )); then
                echo -e "$(ts) ${RED}[handoff]${NC} timeout (30 s) waiting for ready-sentinel; aborting."
                cleanup_bg_deploy_on_cancel
                exit 1
            fi
            sleep 0.05
        done
        echo -e "$(ts) ${GREEN}[handoff]${NC} deploy is in STANDBY and ready."

        # ────────────────────────────────────────────────────────────────
        # WIRE-FRESHNESS PROBE -- catch "dead pose proxy / silent wire"
        # BEFORE we ask the operator to confirm MC-stop. Without this,
        # the operator can answer 'y' to the safety gate while the
        # pose proxy is dead (or the planner stack never started), MC
        # stops onto a silent wire, the deploy's ZmqPoseInputSource
        # returns its bootstrap default_angles forever, and the robot
        # ends up tracking an out-of-distribution reference. See the
        # 2026-05 whirring incident postmortem.
        #
        # Probe target = ${VLA_ZMQ_HOST}:${VLA_ZMQ_PORT}, which is what
        # the deploy will actually SUB to:
        #   * split-topology onbot mode: localhost:5558 (proxy output;
        #     proxy publishes idle_stand even when laptop is silent, so
        #     this gate only fires when the proxy itself is dead).
        #   * local mode without proxy: <laptop_ip>:5556 (planner stack
        #     output directly; gate fires if planner not running).
        # ────────────────────────────────────────────────────────────────
        if ! $NO_WIRE_PROBE; then
            echo -e "$(ts) ${BLUE}[handoff]${NC} probing pose wire on tcp://${VLA_ZMQ_HOST}:${VLA_ZMQ_PORT} (topic=${VLA_ZMQ_TOPIC}, ${WIRE_PROBE_SECS}s) ..."
            # NOTE: capture the substitution rc via `|| probe_rc=$?` so
            # `set -e` doesn't yank us out of the script at the
            # assignment line on probe failure -- otherwise the
            # diagnostic banner + proxy log tail below NEVER print and
            # the operator sees only a bare "status=1" with zero clues
            # about what actually went wrong (the symptom that ate
            # 2026-06-08 morning's debug session).
            probe_out=""
            probe_rc=0
            probe_out="$(probe_pose_wire "$VLA_ZMQ_HOST" "$VLA_ZMQ_PORT" "$VLA_ZMQ_TOPIC" "$WIRE_PROBE_SECS")" || probe_rc=$?
            if [[ $probe_rc -ne 0 ]]; then
                echo ""
                echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
                echo -e "${RED}  WIRE-FRESHNESS PROBE FAILED -- ABORTING BEFORE MC-STOP${NC}"
                echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
                echo ""
                echo -e "${RED}  probe result: ${probe_out}${NC}"
                echo -e "${RED}  No pose frames received on tcp://${VLA_ZMQ_HOST}:${VLA_ZMQ_PORT}${NC}"
                echo -e "${RED}  topic=${VLA_ZMQ_TOPIC} in ${WIRE_PROBE_SECS}s.${NC}"
                echo ""
                echo -e "${YELLOW}  Likely causes:${NC}"
                if [[ "${VLA_ZMQ_HOST}" == "localhost" || "${VLA_ZMQ_HOST}" == "127.0.0.1" ]]; then
                    echo -e "${YELLOW}    (split-topology) The PC2 pose proxy is not running. Check:${NC}"
                    echo -e "${YELLOW}      x2_pc2_daemons.sh logs proxy --pc2-host <pc2_ip>${NC}"
                    echo -e "${YELLOW}      x2_pc2_daemons.sh status --pc2-host <pc2_ip>${NC}"
                    echo -e "${YELLOW}    If the proxy session exited, look in ${PC2_LOG_ROOT:-/home/run/gear-sonic/log}/pose_proxy_*.log${NC}"
                    # Inline the actual proxy log tail so the operator
                    # doesn't have to leave the pane to figure out WHY
                    # the proxy died (argparse error from a stale
                    # bringup is the most common culprit).
                    dump_pose_proxy_log_tail
                else
                    echo -e "${YELLOW}    The laptop pose publisher is not running on ${VLA_ZMQ_HOST}. Start:${NC}"
                    echo -e "${YELLOW}      ./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --no-deploy${NC}"
                    echo -e "${YELLOW}    on the laptop and wait for 'PUB bound on tcp://*:${VLA_ZMQ_PORT}'.${NC}"
                fi
                echo -e "${YELLOW}    Or: --vla-zmq-host / --vla-zmq-port don't match what's publishing.${NC}"
                echo ""
                echo -e "${YELLOW}  To override (NOT recommended on a live robot): re-run with${NC}"
                echo -e "${YELLOW}    --no-wire-probe${NC}"
                echo -e "${YELLOW}  MC is still up; robot is unchanged.${NC}"
                cleanup_bg_deploy_on_cancel
                exit 1
            fi
            echo -e "$(ts) ${GREEN}[handoff]${NC} wire probe OK: ${probe_out}"
        else
            echo -e "$(ts) ${YELLOW}[handoff]${NC} --no-wire-probe: skipping pre-stop wire-freshness check."
        fi

        # ────────────────────────────────────────────────────────────────
        # SAFETY GATE -- merged stop_app + launch confirmation. The
        # robot is still under MC; the bg deploy is silent. Operator
        # has one decision to make.
        # ────────────────────────────────────────────────────────────────
        echo ""
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  SAFETY GATE -- STOP MC + LAUNCH POLICY${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -e "${YELLOW}  Deploy is loaded and waiting in STANDBY (writer suppressed).${NC}"
        echo -e "${YELLOW}  Confirming will: POST stop_app to ${MC_EM_URL} -> verify MC${NC}"
        echo -e "${YELLOW}  silent on bus -> touch trigger sentinel -> deploy enters${NC}"
        echo -e "${YELLOW}  CONTROL within ~20 ms. The robot stays under active control${NC}"
        echo -e "${YELLOW}  the whole time (never enters DAMPING / zero-torque).${NC}"
        echo ""
        if $DRY_RUN; then
            echo -e "${YELLOW}  Dry-run: pipeline runs but stiffness/damping are zero --${NC}"
            echo -e "${YELLOW}  no torque commanded. CSVs will log what WOULD have been sent.${NC}"
        else
            echo -e "${RED}  Powered: PD targets WILL be published to /aima/hal/joint/*/command${NC}"
            echo -e "${RED}  at ~500 Hz. Soft-start ramp blends policy in over --ramp-seconds.${NC}"
        fi
        echo ""
        echo -e "${YELLOW}  Confirm BEFORE proceeding:${NC}"
        echo -e "${YELLOW}    [ ] Robot is firmly supported (gantry / harness / hand-held)${NC}"
        echo -e "${YELLOW}    [ ] No personnel within arm or leg reach of robot${NC}"
        echo -e "${YELLOW}    [ ] You are ready for slight settling motion when MC releases${NC}"
        # Recorder liveness check. start_run_recorder ran several lines
        # back; this banner is the operator's last chance to confirm
        # that the npz pipeline is up before they Ctrl-C at the prompt
        # (which now flushes the recording on the way out -- see
        # cleanup_bg_deploy_on_cancel). If --record was not passed, fall
        # through silently.
        if $RECORD_RUN; then
            if [[ -n "$RUN_RECORD_PID" ]] && kill -0 "$RUN_RECORD_PID" 2>/dev/null; then
                echo -e "${GREEN}    [ ] Recorder LIVE (pid $RUN_RECORD_PID) -> $RECORD_OUT${NC}"
                echo -e "${GREEN}        (Ctrl-C at this prompt is also captured -- npz will be flushed)${NC}"
            else
                echo -e "${RED}    [!] Recorder is NOT running -- pre-'y' state will not be captured.${NC}"
            fi
        fi
        if ! $DRY_RUN; then
            echo -e "${RED}    [ ] E-stop is within reach (this is NOT a dry-run)${NC}"
        fi
        echo ""
        if [[ -n "$HOLD_FOR_MC_TIMEOUT_S" && "$HOLD_FOR_MC_TIMEOUT_S" != "0" ]]; then
            echo -e "${BLUE}  Smooth handoff: deploy will RAMP_OUT to MC's STAND_DEFAULT pose,${NC}"
            echo -e "${BLUE}  then HOLD that pose (MC-stand gains) while bash POSTs start_app${NC}"
            echo -e "${BLUE}  + SetMcAction(JOINT_DEFAULT -> STAND_DEFAULT). Deploy exits cleanly${NC}"
            echo -e "${BLUE}  once MC has the bus back. Run starts AND ends in STAND_DEFAULT.${NC}"
        fi
        echo ""
        if ! $NO_CONFIRM; then
            read -p "$(echo -e ${RED}Stop MC and launch policy? [y/N]: ${NC})" mc_confirm
            if [[ ! "$mc_confirm" =~ ^[Yy]$ ]]; then
                echo -e "$(ts) ${YELLOW}[handoff]${NC} cancelled before MC stop. Robot is unchanged."
                cleanup_bg_deploy_on_cancel
                exit 0
            fi
        else
            echo -e "$(ts) ${YELLOW}[handoff]${NC} --no-confirm: skipping safety gate."
        fi

        # Operator confirmed. From here we're committed -- swap traps so
        # an error/Ctrl-C will restart MC (cleanup_bg_deploy_on_cancel
        # is no longer the right thing because the deploy is what's
        # holding the robot).
        echo ""
        echo -e "$(ts) ${BLUE}[handoff]${NC} stopping MC via PC1 EM HTTP API ($MC_EM_URL) ..."
        if ! mc_em_post stop_app; then
            echo -e "$(ts) ${RED}[handoff]${NC} POST $MC_EM_URL/json/stop_app failed."
            echo -e "${YELLOW}  Possible causes:${NC}"
            echo -e "${YELLOW}    - host has no route to PC1 (check the SDK NIC IP / SDK cable / PC1_HOST)${NC}"
            echo -e "${YELLOW}    - PC1 EM is not running${NC}"
            echo -e "${YELLOW}    - MC is already stopped from a previous run (use --no-stop-mc)${NC}"
            cleanup_bg_deploy_on_cancel
            exit 1
        fi
        MC_STOPPED_BY_US=true
        # Replace the cancel trap with the cleanup trap. From now on
        # any unexpected exit must restart MC.
        trap restart_mc_on_exit EXIT INT TERM

        # Fire the start-trigger sentinel IMMEDIATELY after stop_app's
        # synchronous return. The HTTP POST to PC1's EM is itself the
        # confirmation that MC's process has stopped -- if it hadn't,
        # mc_em_post would have returned non-zero and we'd have bailed
        # already. Skipping the publisher-count verify shaves ~3 s off
        # the Y -> CONTROL latency. Note: in the STANDBY architecture
        # `ros2 topic info` would always report >= 1 publisher because
        # the deploy node's own AimdkIo publisher is alive throughout
        # STANDBY (just gated from publishing); the legacy "expected 0"
        # check was guaranteed to fail and waste ~3 s polling for an
        # impossible state.
        : > "$START_TRIGGER_SENTINEL"
        echo -e "$(ts) ${GREEN}[handoff]${NC} start-trigger sentinel touched -> deploy entering CONTROL."

        # Hand bridge: MC has been silenced and the deploy is about to
        # publish its first body command; this is the right moment to
        # bring the OmniHand HAL up. Spawning earlier would let MC
        # eat the engage burst (mc republishes its own HandCommandArray
        # at high rate); spawning later means the first user grasp
        # fires before motors are enabled. Reaped via stop_hand_bridge
        # in restart_mc_on_exit. Sim mode does not call this.
        start_hand_bridge

        # Now stream the deploy's pre-trigger output (everything from
        # boot through STANDBY) and tail-follow it for the rest of the
        # run. Tail is killed when the bg deploy exits.
        if [[ -f "$DEPLOY_STDOUT_LOG" ]]; then
            cat "$DEPLOY_STDOUT_LOG"
            tail -n 0 -F "$DEPLOY_STDOUT_LOG" 2>/dev/null &
            DEPLOY_TAIL_PID=$!
        fi

        # Best-effort sanity check, asynchronous to the trigger fire so
        # it doesn't add to the latency. Counts publishers on
        # /aima/hal/joint/arm/command: we expect 1 (the deploy node
        # itself). >= 2 means MC didn't actually stop and we're about
        # to dual-publish; surface that loudly (still proceed -- bailing
        # now would leave the robot half-committed).
        if command -v ros2 &>/dev/null; then
            (
                ARM_CMD_PUBS=$(timeout 1 ros2 topic info /aima/hal/joint/arm/command 2>/dev/null \
                    | awk '/Publisher count:/ {print $NF}')
                if [[ "${ARM_CMD_PUBS:-?}" == "1" ]]; then
                    echo -e "$(ts) ${GREEN}[handoff]${NC} bus check OK: 1 publisher (deploy) on /aima/hal/joint/arm/command."
                elif [[ "${ARM_CMD_PUBS:-?}" == "0" ]]; then
                    echo -e "$(ts) ${YELLOW}[handoff]${NC} bus check: 0 publishers (deploy may not have advanced past STANDBY yet)."
                else
                    echo -e "$(ts) ${RED}[handoff]${NC} bus check WARNING: ${ARM_CMD_PUBS:-?} publishers (expected 1). MC may not have actually stopped -- watch for motor whir."
                fi
            ) &
        fi
        echo ""
    elif [[ -n "$HOLD_FOR_MC_SENTINEL" ]]; then
        # Pre-cold-warm-boot fallback (shouldn't happen on real-robot
        # mode now that we always set the trigger/ready sentinels, but
        # kept defensively): launch in foreground-as-background, no
        # STANDBY pre-launch.
        echo -e "$(ts) ${YELLOW}[handoff]${NC} STANDBY sentinels not configured; using legacy stop_app -> launch flow."
        # Wire-freshness probe (same rationale as the STANDBY path).
        if ! $NO_WIRE_PROBE; then
            echo -e "$(ts) ${BLUE}[handoff]${NC} probing pose wire on tcp://${VLA_ZMQ_HOST}:${VLA_ZMQ_PORT} (topic=${VLA_ZMQ_TOPIC}, ${WIRE_PROBE_SECS}s) ..."
            # `|| probe_rc=$?`: suppress set -e on the failing
            # substitution so the diagnostic block below actually runs.
            probe_out=""
            probe_rc=0
            probe_out="$(probe_pose_wire "$VLA_ZMQ_HOST" "$VLA_ZMQ_PORT" "$VLA_ZMQ_TOPIC" "$WIRE_PROBE_SECS")" || probe_rc=$?
            if [[ $probe_rc -ne 0 ]]; then
                echo -e "${RED}wire probe FAILED (${probe_out}): no frames on ${VLA_ZMQ_HOST}:${VLA_ZMQ_PORT}. Aborting before MC-stop.${NC}"
                if [[ "${VLA_ZMQ_HOST}" == "localhost" || "${VLA_ZMQ_HOST}" == "127.0.0.1" ]]; then
                    dump_pose_proxy_log_tail
                fi
                echo -e "${YELLOW}Re-run with --no-wire-probe to override (NOT recommended on a live robot).${NC}"
                exit 1
            fi
            echo -e "$(ts) ${GREEN}[handoff]${NC} wire probe OK: ${probe_out}"
        fi
        # Operator confirmation BEFORE stop_app for the legacy path.
        if ! $NO_CONFIRM; then
            read -p "$(echo -e ${RED}Stop MC and launch policy? [y/N]: ${NC})" mc_confirm
            if [[ ! "$mc_confirm" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Cancelled before MC stop. Robot is unchanged.${NC}"
                exit 0
            fi
        fi
        if ! $NO_STOP_MC; then
            mc_em_post stop_app || { echo -e "${RED}stop_app failed${NC}"; exit 1; }
            MC_STOPPED_BY_US=true
            trap restart_mc_on_exit EXIT INT TERM
            sleep 1
        fi
        # Same rationale as the STANDBY-path call above: spawn the
        # hand bridge after MC is verified silenced so its engage
        # burst doesn't get clobbered by mc's own hand publish loop.
        start_hand_bridge
        echo -e "$(ts) ${BLUE}[handoff]${NC} HOLD_FOR_MC sentinel:        $HOLD_FOR_MC_SENTINEL"
        echo -e "$(ts) ${BLUE}[handoff]${NC} HOLD_FOR_MC exit-sentinel:   $HOLD_FOR_MC_EXIT_SENTINEL"
        echo -e "$(ts) ${BLUE}[handoff]${NC} MC first-publish sentinel:   $MC_FIRST_PUBLISH_SENTINEL"
        ros2 run "$PKG_NAME" x2_deploy_onnx_ref "${ROS2_ARGS[@]}" &
        DEPLOY_PID=$!
    else
        # Legacy path: HOLD_FOR_MC disabled. Run the deploy in the
        # foreground; the cleanup trap brings MC back after deploy
        # exits. There is a zero-torque window between deploy-exit
        # and MC-up here.
        # Wire-freshness probe (same rationale as the STANDBY path).
        if ! $NO_WIRE_PROBE; then
            echo -e "$(ts) ${BLUE}[handoff]${NC} probing pose wire on tcp://${VLA_ZMQ_HOST}:${VLA_ZMQ_PORT} (topic=${VLA_ZMQ_TOPIC}, ${WIRE_PROBE_SECS}s) ..."
            # `|| probe_rc=$?`: suppress set -e on the failing
            # substitution so the diagnostic block below actually runs.
            probe_out=""
            probe_rc=0
            probe_out="$(probe_pose_wire "$VLA_ZMQ_HOST" "$VLA_ZMQ_PORT" "$VLA_ZMQ_TOPIC" "$WIRE_PROBE_SECS")" || probe_rc=$?
            if [[ $probe_rc -ne 0 ]]; then
                echo -e "${RED}wire probe FAILED (${probe_out}): no frames on ${VLA_ZMQ_HOST}:${VLA_ZMQ_PORT}. Aborting before MC-stop.${NC}"
                if [[ "${VLA_ZMQ_HOST}" == "localhost" || "${VLA_ZMQ_HOST}" == "127.0.0.1" ]]; then
                    dump_pose_proxy_log_tail
                fi
                echo -e "${YELLOW}Re-run with --no-wire-probe to override (NOT recommended on a live robot).${NC}"
                exit 1
            fi
            echo -e "$(ts) ${GREEN}[handoff]${NC} wire probe OK: ${probe_out}"
        fi
        if ! $NO_CONFIRM; then
            read -p "$(echo -e ${RED}Stop MC and launch policy? [y/N]: ${NC})" mc_confirm
            if [[ ! "$mc_confirm" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Cancelled before MC stop. Robot is unchanged.${NC}"
                exit 0
            fi
        fi
        if ! $NO_STOP_MC; then
            mc_em_post stop_app || { echo -e "${RED}stop_app failed${NC}"; exit 1; }
            MC_STOPPED_BY_US=true
            trap restart_mc_on_exit EXIT INT TERM
            sleep 1
        fi
        ros2 run "$PKG_NAME" x2_deploy_onnx_ref "${ROS2_ARGS[@]}"
        DEPLOY_RC=$?
        exit "$DEPLOY_RC"
    fi

    # ────────────────────────────────────────────────────────────────────
    # HOLD_FOR_MC handoff loop (deploy is now running; we've already
    # POSTed stop_app + touched the start-trigger sentinel above for
    # the cold-warm path, or skipped that for legacy).
    # Poll for HOLD_FOR_MC_SENTINEL the deploy node touches on entering
    # HOLD_FOR_MC. The moment we see it, POST start_app + drive MC back
    # to STAND_DEFAULT WHILE the deploy is still actively holding the
    # bus (no zero-torque window). Deploy's MC-takeover detector then
    # exits cleanly within <= 20 ms; we wait on its PID for the final
    # exit code. If the sentinel never appears (deploy crashed in
    # CONTROL, or HOLD_FOR_MC was disabled), the cleanup trap's
    # restart_mc_on_exit still fires start_app as the fallback path.
    # ────────────────────────────────────────────────────────────────────
    if [[ -n "$DEPLOY_PID" && -n "$HOLD_FOR_MC_SENTINEL" ]]; then
        HANDOFF_DONE=false
        while kill -0 "$DEPLOY_PID" 2>/dev/null; do
            if ! $HANDOFF_DONE && [[ -f "$HOLD_FOR_MC_SENTINEL" ]]; then
                HANDOFF_DONE=true
                echo ""
                echo -e "$(ts) ${GREEN}[handoff]${NC} HOLD_FOR_MC sentinel detected -- deploy is in HOLD_FOR_MC."
                echo -e "$(ts) ${BLUE}[handoff]${NC} POSTing start_app to bring MC back ..."
                if mc_em_post start_app; then
                    echo -e "$(ts) ${GREEN}[handoff]${NC} MC start_app POSTed."
                    # MC is now running; clear MC_STOPPED_BY_US so the
                    # cleanup trap can't double-POST start_app later
                    # (HTTP 500 if MC is already up).
                    MC_STOPPED_BY_US=false
                    # ────────────────────────────────────────────────
                    # Inline escalation: PASSIVE_DEFAULT -> JOINT_DEFAULT
                    # -> exit deploy -> STAND_DEFAULT.
                    #
                    # Why split it this way (changed 2026-05-03 after
                    # operator observed ~3s of motor whirring during
                    # the JOINT->STAND window): the previous flow
                    # waited until STAND_DEFAULT before touching the
                    # exit-sentinel, so deploy + MC both published
                    # for the full ~1.3s of the JOINT->STAND escalation.
                    # That's the dual-publisher fight (deploy kp=24
                    # vs MC's JOINT kp=30 vs MC's STAND kp=30, all at
                    # 500 Hz on the same topic) the user heard as
                    # whirring.
                    #
                    # JOINT_DEFAULT is already an actively-balancing
                    # mode (kp_arm=30, eff_leg ~6 Nm, robot firmly
                    # held). Once MC is in JOINT_DEFAULT we can release
                    # deploy immediately -- MC alone handles the
                    # JOINT->STAND transition with no zero-torque
                    # window. Net result: ~20ms of dual-publisher
                    # overlap (sentinel-touch -> deploy's next
                    # OnControl tick) instead of ~1.3s.
                    # ────────────────────────────────────────────────
                    HANDOFF_OK=false
                    if command -v ros2 &>/dev/null; then
                        # ────────────────────────────────────────────────
                        # Spawn the persistent-client escalator BEFORE we
                        # start polling/waiting. It hammers SetMcAction(
                        # JOINT_DEFAULT) at 20 Hz with a held-open service
                        # client, so the very first attempt that lands
                        # AFTER MC's services come up succeeds within one
                        # tick (sub-ms RTT, no per-call rclpy startup).
                        #
                        # Pre-launch failed calls are harmless -- MC just
                        # returns code != 0 ("service not ready / mode
                        # rejected") and we retry.
                        #
                        # Compared to the old "wait for MC, then issue one
                        # `ros2 service call`" flow, this saves ~700 ms of
                        # cumulative bash + rclpy startup overhead, which
                        # directly cuts the dual-publisher whir window
                        # MC's PASSIVE_DEFAULT phase by the same amount.
                        # ────────────────────────────────────────────────
                        if [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]]; then
                            ESCALATOR_OK_SENTINEL="$RUN_LOG_DIR/mc_escalator_ok.sentinel"
                            ESCALATOR_LOG="$RUN_LOG_DIR/mc_escalator.log"
                        else
                            ESCALATOR_OK_SENTINEL="/tmp/x2_mc_escalator_ok.$$.sentinel"
                            ESCALATOR_LOG="/tmp/x2_mc_escalator.$$.log"
                        fi
                        rm -f "$ESCALATOR_OK_SENTINEL"
                        ESCALATOR_SCRIPT="$SCRIPT_DIR/scripts/x2_mc_escalator.py"
                        if [[ ! -x "$ESCALATOR_SCRIPT" && -f "$ESCALATOR_SCRIPT" ]]; then
                            chmod +x "$ESCALATOR_SCRIPT" 2>/dev/null || true
                        fi
                        ESCALATOR_PID=""
                        if [[ -f "$ESCALATOR_SCRIPT" ]]; then
                            "$DEPLOY_AUX_PYTHON" "$ESCALATOR_SCRIPT" \
                                --target JOINT_DEFAULT \
                                --rate-hz 20 \
                                --timeout-s 30 \
                                --call-timeout-s 0.4 \
                                --success-sentinel "$ESCALATOR_OK_SENTINEL" \
                                --log "$ESCALATOR_LOG" \
                                >>"$ESCALATOR_LOG" 2>&1 &
                            ESCALATOR_PID=$!
                            echo -e "$(ts) ${BLUE}[post-handoff]${NC} escalator launched (pid $ESCALATOR_PID): hammering SetMcAction(JOINT_DEFAULT) at 20Hz, ok-sentinel=$ESCALATOR_OK_SENTINEL"
                        else
                            echo -e "$(ts) ${YELLOW}[post-handoff]${NC} escalator script not found ($ESCALATOR_SCRIPT); falling back to single-shot mc_set_action."
                        fi

                        # Wait up to 30 s for either (a) the escalator to
                        # report success, or (b) deploy's first-publish
                        # sentinel to fire (= MC services are alive even
                        # if the escalator is still waiting on a clean
                        # accept). We poll at 50 ms granularity.
                        MC_BOOT_MODE=""
                        ESCALATOR_OK=false
                        for _i in $(seq 1 600); do
                            if [[ -f "$ESCALATOR_OK_SENTINEL" ]]; then
                                ESCALATOR_OK=true
                                MC_BOOT_MODE="JOINT_DEFAULT"
                                break
                            fi
                            if [[ -f "$MC_FIRST_PUBLISH_SENTINEL" && -z "$MC_BOOT_MODE" ]]; then
                                MC_BOOT_MODE="PASSIVE_DEFAULT"
                                # Don't break -- keep waiting for the
                                # escalator to confirm JOINT_DEFAULT.
                            fi
                            sleep 0.05
                        done

                        # If escalator timed out but first-publish fired,
                        # try one more bash-side mc_set_action as a
                        # fallback; this preserves the previous
                        # single-shot path's behaviour.
                        if ! $ESCALATOR_OK && [[ -n "$MC_BOOT_MODE" ]]; then
                            echo -e "$(ts) ${YELLOW}[post-handoff]${NC} escalator did not confirm (mode=$MC_BOOT_MODE); falling back to single-shot SetMcAction(JOINT_DEFAULT)."
                            if mc_set_action JOINT_DEFAULT; then
                                MC_BOOT_MODE="JOINT_DEFAULT"
                                ESCALATOR_OK=true
                            else
                                echo -e "$(ts) ${RED}[post-handoff]${NC} fallback SetMcAction(JOINT_DEFAULT) rejected."
                            fi
                        fi

                        if [[ -z "$MC_BOOT_MODE" ]]; then
                            echo -e "$(ts) ${RED}[post-handoff]${NC} MC services never came up after start_app (escalator timed out and first-publish never fired)."
                        else
                            if $ESCALATOR_OK; then
                                echo -e "$(ts) ${GREEN}[post-handoff]${NC} -> JOINT_DEFAULT confirmed by escalator (active PD; releasing deploy now)."
                            elif [[ -f "$MC_FIRST_PUBLISH_SENTINEL" ]]; then
                                echo -e "$(ts) ${BLUE}[post-handoff]${NC} MC came back up (first-publish detected on bus, mode=$MC_BOOT_MODE)."
                            else
                                echo -e "$(ts) ${BLUE}[post-handoff]${NC} MC came back up in $MC_BOOT_MODE."
                            fi
                            # CRITICAL: release deploy AS SOON AS MC is in
                            # JOINT_DEFAULT (or already STAND_DEFAULT). This
                            # is the single most important moment for the
                            # smooth handoff; everything below is async.
                            if [[ "$MC_BOOT_MODE" == "JOINT_DEFAULT" || "$MC_BOOT_MODE" == "STAND_DEFAULT" ]]; then
                                : > "$HOLD_FOR_MC_EXIT_SENTINEL"
                                echo -e "$(ts) ${BLUE}[handoff]${NC} exit-sentinel touched (MC is in $MC_BOOT_MODE); deploy will release the bus on its next tick."
                                HANDOFF_OK=true
                            fi
                            # JOINT_DEFAULT -> STAND_DEFAULT. Deploy is
                            # already exiting (or exited) at this point;
                            # MC handles the rest of the escalation alone.
                            #
                            # Verify the transition via mc_get_action.
                            # Like SetMcAction(JOINT_DEFAULT), the
                            # SetMcAction(STAND_DEFAULT) response code is
                            # not a reliable confirmation -- it returns
                            # code=0 even when MC is still booting and
                            # silently ignores the request. Confirmed via
                            # the 2026-05-03 run x2_run_20260503_213002,
                            # where bash logged "STAND_DEFAULT confirmed"
                            # but the recorder showed MC stayed in
                            # PASSIVE_DEFAULT throughout. Treat
                            # mc_get_action as ground truth.
                            if [[ "$MC_BOOT_MODE" == "JOINT_DEFAULT" ]]; then
                                if mc_set_action STAND_DEFAULT; then
                                    POST_STAND_OK=false
                                    POST_STAND_MODE=""
                                    for _i in $(seq 1 40); do
                                        POST_STAND_MODE="$(mc_get_action 0.3 2>/dev/null || true)"
                                        if [[ "$POST_STAND_MODE" == "STAND_DEFAULT" ]]; then
                                            POST_STAND_OK=true
                                            break
                                        fi
                                        sleep 0.05
                                    done
                                    if $POST_STAND_OK; then
                                        echo -e "$(ts) ${GREEN}[post-handoff]${NC} -> STAND_DEFAULT confirmed via mc_get_action."
                                    else
                                        echo -e "$(ts) ${RED}[post-handoff]${NC} SetMcAction(STAND_DEFAULT) returned code=0 but mc_get_action reports ${POST_STAND_MODE:-<empty>}, NOT STAND_DEFAULT. Robot may be in wrong mode -- check the mobile app."
                                    fi
                                else
                                    echo -e "$(ts) ${RED}[post-handoff]${NC} SetMcAction(STAND_DEFAULT) was rejected."
                                fi
                            fi
                        fi

                        # Reap escalator if still alive (it should have
                        # exited on success-sentinel).
                        if [[ -n "$ESCALATOR_PID" ]] && kill -0 "$ESCALATOR_PID" 2>/dev/null; then
                            kill -TERM "$ESCALATOR_PID" 2>/dev/null || true
                            wait "$ESCALATOR_PID" 2>/dev/null || true
                        fi
                    fi
                    if ! $HANDOFF_OK; then
                        echo -e "$(ts) ${RED}[handoff]${NC} MC escalation did not reach JOINT_DEFAULT."
                        echo -e "${YELLOW}  Deploy will keep holding the robot until${NC}"
                        echo -e "${YELLOW}  --hold-for-mc-timeout-s elapses. To recover:${NC}"
                        echo -e "${YELLOW}    1. From the mobile app, switch MC to Standing Default.${NC}"
                        echo -e "${YELLOW}    2. touch $HOLD_FOR_MC_EXIT_SENTINEL${NC}"
                        echo -e "${YELLOW}  to release the deploy.${NC}"
                    fi
                else
                    echo -e "$(ts) ${RED}[handoff]${NC} MC start_app POST failed."
                    echo -e "${YELLOW}  Deploy will hold STAND_DEFAULT pose until${NC}"
                    echo -e "${YELLOW}  --hold-for-mc-timeout-s elapses. Check the MC HTTP API.${NC}"
                fi
            fi
            sleep 0.1
        done
        wait "$DEPLOY_PID"
        DEPLOY_RC=$?
        if [[ -n "${DEPLOY_TAIL_PID:-}" ]] && kill -0 "$DEPLOY_TAIL_PID" 2>/dev/null; then
            kill -TERM "$DEPLOY_TAIL_PID" 2>/dev/null || true
            wait "$DEPLOY_TAIL_PID" 2>/dev/null || true
        fi
        rm -f "$HOLD_FOR_MC_SENTINEL" "$HOLD_FOR_MC_EXIT_SENTINEL" \
              "$MC_FIRST_PUBLISH_SENTINEL" \
              "$START_TRIGGER_SENTINEL" "$READY_SENTINEL"
        echo -e "$(ts) ${BLUE}[handoff]${NC} deploy exited with code $DEPLOY_RC."
        exit "$DEPLOY_RC"
    elif [[ -n "$DEPLOY_PID" ]]; then
        # No HOLD_FOR_MC; just wait on the bg deploy.
        wait "$DEPLOY_PID"
        DEPLOY_RC=$?
        if [[ -n "${DEPLOY_TAIL_PID:-}" ]] && kill -0 "$DEPLOY_TAIL_PID" 2>/dev/null; then
            kill -TERM "$DEPLOY_TAIL_PID" 2>/dev/null || true
            wait "$DEPLOY_TAIL_PID" 2>/dev/null || true
        fi
        rm -f "$START_TRIGGER_SENTINEL" "$READY_SENTINEL"
        exit "$DEPLOY_RC"
    fi
fi
