#!/usr/bin/env bash
# Build (idempotent) and drop into a shell inside the X2 sim container,
# with DDS isolated to loopback (ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID=73 from
# docker-compose.yml). Pair with `deploy_x2.sh sim`.
#
# For talking to a REAL robot from inside the container, use ./get_x2_sonic_ready.sh
# instead -- that one layers docker-compose.real.yml on top to un-quarantine
# DDS and pin it to the SDK ethernet NIC.
#
# Minimal wrapper -- all config lives in docker-compose.yml so `docker compose
# run` does the right thing whether you call it through this wrapper or
# directly. Mirrors AgiBot's record-and-replay reference start.sh.
#
# Usage:
#   ./enter_sim.sh                       # build + interactive shell
#   ./enter_sim.sh -- ros2 topic list    # build + run a one-shot command
#
# X11 forwarding for the mujoco viewer (--sim-viewer):
#   docker-compose.yml mounts /tmp/.X11-unix and forwards $DISPLAY into the
#   container, but the host's X server still rejects connections by default.
#   We grant the local root user (the container's uid) access via xhost.
#   This is the path of least resistance; a more locked-down alternative is
#   to copy the Xauth cookie into the container and run as a matching uid.
set -euo pipefail
cd "$(dirname "$0")"

docker compose build

# Allow the container's root to talk to the host X server. Skipped if there's
# no X session (e.g. a headless CI box) -- in that case --sim-viewer will not
# work, but everything else will.
if command -v xhost >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
    xhost +SI:localuser:root >/dev/null || \
        echo "[enter_sim.sh] WARNING: xhost +SI:localuser:root failed; --sim-viewer will not be able to open a window."
fi

if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
    exec docker compose run --rm x2sim bash -c \
        "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec $*"
else
    exec docker compose run --rm x2sim
fi
