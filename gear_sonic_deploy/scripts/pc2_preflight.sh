#!/usr/bin/env bash
# PC2 (Jetson Orin NX) preflight script for the X2 split-topology deploy.
#
# Verifies that PC2 is ready to host the long-running C++ deploy + hand
# bridge + motor monitor tmux sessions before x2_pc2_daemons.sh start
# is invoked from the laptop. Designed to be safe to run repeatedly:
# read-only checks, no MC interactions, no joint commands.
#
# Run from the laptop:
#
#     ./gear_sonic_deploy/scripts/pc2_preflight.sh \
#         [--pc2-host <PC2_IP>] [--pc2-user run] \
#         [--pc2-ws /home/run/gear-sonic/ws] \
#         [--pc1-host <PC1_IP>]        # optional: enables sections 2 + 6
#
# Each section is gated independently so a failure in one (e.g. tmux
# missing) doesn't suppress later checks; the script tallies all
# failures and exits non-zero if any check failed. Output is colour-
# coded so a pass-only run is a wall of green.
#
# Sections (in order):
#
#   1. PC2 reachability  -- ssh + ping + uname
#   2. PC1 reachability  -- ssh from PC2 -> PC1 (via SDK ethernet)
#   3. Required tools    -- ros2, tmux, python3, rsync, curl, pyzmq
#   4. ROS workspace     -- $PC2_WS/install/setup.bash + agi_x2_deploy_onnx_ref pkg
#   5. ROS topics        -- MC publishers visible (joint state + command)
#   6. EM HTTP API       -- /x2/em/status reachable from PC2 to PC1
#   7. Log root          -- $PC2_LOG_ROOT exists, writable
#   8. Free ports        -- 5557 (deploy x2_debug PUB) + 5567 (monitor PUB)
#                            currently free on PC2

set -u

PC2_USER="${PC2_USER:-run}"
PC2_HOST="${PC2_HOST:-}"      # REQUIRED: --pc2-host or env (no default)
PC1_HOST="${PC1_HOST:-}"
PC1_EM_PORT="${PC1_EM_PORT:-50080}"
PC2_PREFIX="${PC2_PREFIX:-/home/run/gear-sonic}"
PC2_WS="${PC2_WS:-${PC2_PREFIX}/ws}"
PC2_LOG_ROOT="${PC2_LOG_ROOT:-${PC2_PREFIX}/log}"
PC2_VENV="${PC2_VENV:-${PC2_PREFIX}/venv}"
PC2_PKG="${PC2_PKG:-agi_x2_deploy_onnx_ref}"
# The daemon wrapper places the C++ package one level deeper (under
# src/x2/) so the relative add_subdirectory(../../common) inside the
# package's CMakeLists resolves correctly; mirror that here when
# probing for the python scripts.
PC2_PKG_SRC_REL="${PC2_PKG_SRC_REL:-x2/${PC2_PKG}}"
ROS_DISTRO="${ROS_DISTRO:-humble}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pc2-host) PC2_HOST="$2"; shift 2 ;;
        --pc2-user) PC2_USER="$2"; shift 2 ;;
        --pc2-ws) PC2_WS="$2"; shift 2 ;;
        --pc1-host) PC1_HOST="$2"; shift 2 ;;
        --pc1-em-port) PC1_EM_PORT="$2"; shift 2 ;;
        --pc2-log-root) PC2_LOG_ROOT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${PC2_HOST}" ]]; then
    fail "PC2_HOST is not set: pass --pc2-host <PC2_IP> or export PC2_HOST (x2_discover_network.sh writes ~/.x2/env.wired)."
fi

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_BLUE=$'\e[34m'; C_RESET=$'\e[0m'

PASS=0
FAIL=0
WARN=0

ok()    { printf '  %s[ ok ]%s %s\n' "${C_GREEN}" "${C_RESET}" "$*"; PASS=$((PASS+1)); }
fail()  { printf '  %s[FAIL]%s %s\n' "${C_RED}"   "${C_RESET}" "$*"; FAIL=$((FAIL+1)); }
warn()  { printf '  %s[warn]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*"; WARN=$((WARN+1)); }
section() { printf '\n%s== %s ==%s\n' "${C_BLUE}" "$*" "${C_RESET}"; }

ssh_pc2() {
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes \
        "${PC2_USER}@${PC2_HOST}" "$@"
}

# -------------------------------------------------------------------------
# 1. PC2 reachability
# -------------------------------------------------------------------------
section "1. PC2 reachability (${PC2_USER}@${PC2_HOST})"
if ping -c1 -W2 "${PC2_HOST}" >/dev/null 2>&1; then
    ok "ping ${PC2_HOST} -> reachable"
else
    fail "ping ${PC2_HOST} -> no response"
fi
if ssh_pc2 "uname -a" >/dev/null 2>&1; then
    uname_out="$(ssh_pc2 'uname -a')"
    ok "ssh ${PC2_USER}@${PC2_HOST} -> ${uname_out}"
else
    fail "ssh ${PC2_USER}@${PC2_HOST} -- check key auth (BatchMode=yes)"
fi

# -------------------------------------------------------------------------
# 2. PC1 reachability from PC2
# -------------------------------------------------------------------------
section "2. PC1 reachability from PC2 (${PC1_HOST:-PC1_HOST unset})"
if [[ -z "${PC1_HOST}" ]]; then
    warn "PC1_HOST not set (pass --pc1-host <PC1_IP> or export PC1_HOST) -- skipping PC2 -> PC1 ping"
elif ssh_pc2 "ping -c1 -W2 ${PC1_HOST}" >/dev/null 2>&1; then
    ok "PC2 -> PC1 ping ok"
else
    fail "PC2 cannot ping PC1 (check SDK ethernet bridge)"
fi

# -------------------------------------------------------------------------
# 3. Required tools
# -------------------------------------------------------------------------
section "3. Required tools on PC2"
for tool in tmux rsync curl python3; do
    if ssh_pc2 "command -v ${tool}" >/dev/null 2>&1; then
        ok "${tool} present"
    else
        fail "${tool} not found on PC2 (install with apt-get install ${tool})"
    fi
done
# ros2 lives under /opt/ros and isn't on PATH by default; the daemon
# sources setup.bash at launch time, so probe the same way here.
if ssh_pc2 "source /opt/ros/${ROS_DISTRO}/setup.bash 2>/dev/null && command -v ros2" >/dev/null 2>&1; then
    ok "ros2 cli present (under /opt/ros/${ROS_DISTRO}/)"
else
    fail "ros2 cli not found (apt install ros-${ROS_DISTRO}-ros-base)"
fi
# pyzmq is installed inside the prefix venv (--system-site-packages),
# NOT in the system python3. The daemon launches python via
# ${PC2_VENV}/bin/python3, so probe the same binary.
if ssh_pc2 "${PC2_VENV}/bin/python3 -c 'import zmq; print(zmq.__version__)'" >/dev/null 2>&1; then
    pyzmq_ver="$(ssh_pc2 "${PC2_VENV}/bin/python3 -c 'import zmq; print(zmq.__version__)'" 2>/dev/null)"
    ok "pyzmq ${pyzmq_ver} (in ${PC2_VENV})"
else
    fail "pyzmq missing in ${PC2_VENV} (run: pc2_bringup.sh --pc2-host ${PC2_HOST})"
fi

# -------------------------------------------------------------------------
# 4. ROS workspace + agi_x2_deploy_onnx_ref pkg
# -------------------------------------------------------------------------
section "4. ROS workspace at ${PC2_WS}"
if ssh_pc2 "[ -f ${PC2_WS}/install/setup.bash ]" 2>/dev/null; then
    ok "${PC2_WS}/install/setup.bash exists"
else
    fail "${PC2_WS}/install/setup.bash missing -- run: cd ${PC2_WS} && colcon build --packages-select ${PC2_PKG}"
fi
if ssh_pc2 "source /opt/ros/${ROS_DISTRO}/setup.bash 2>/dev/null && \
            source ${PC2_WS}/install/setup.bash 2>/dev/null && \
            ros2 pkg prefix ${PC2_PKG}" >/dev/null 2>&1; then
    pkg_prefix="$(ssh_pc2 "source /opt/ros/${ROS_DISTRO}/setup.bash 2>/dev/null && \
                          source ${PC2_WS}/install/setup.bash 2>/dev/null && \
                          ros2 pkg prefix ${PC2_PKG}" 2>/dev/null)"
    ok "ros2 pkg ${PC2_PKG} -> ${pkg_prefix}"
else
    fail "ros2 pkg ${PC2_PKG} not found in workspace overlay"
fi
for script in x2_motor_monitor.py x2_hand_zmq_to_aimdk_bridge.py; do
    if ssh_pc2 "[ -f ${PC2_WS}/src/${PC2_PKG_SRC_REL}/scripts/${script} ]" 2>/dev/null; then
        ok "${script} present at src/${PC2_PKG_SRC_REL}/scripts/"
    else
        fail "${script} missing at ${PC2_WS}/src/${PC2_PKG_SRC_REL}/scripts/ -- run: pc2_bringup.sh --pc2-host ${PC2_HOST} --skip-onnx --skip-venv --skip-model --skip-build"
    fi
done
# The hand bridge does `from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import ...`
# from the rsynced compat shim at ws/src/gear_sonic/.
if ssh_pc2 "[ -f ${PC2_WS}/src/gear_sonic/utils/teleop/zmq/zmq_packed_message_decoder.py ]" 2>/dev/null; then
    ok "gear_sonic.utils.teleop.zmq compat shim present"
else
    fail "gear_sonic/utils/teleop/zmq/zmq_packed_message_decoder.py missing -- run pc2_bringup.sh"
fi

# -------------------------------------------------------------------------
# 5. MC publishers visible on PC2
# -------------------------------------------------------------------------
section "5. MC publishers (must be in STAND_DEFAULT or similar)"
for group in leg waist arm head; do
    info_out=$(ssh_pc2 "source /opt/ros/${ROS_DISTRO}/setup.bash 2>/dev/null && \
                       source ${PC2_WS}/install/setup.bash 2>/dev/null && \
                       timeout 3 ros2 topic info /aima/hal/joint/${group}/command 2>/dev/null" || true)
    if [[ -n "${info_out}" && "${info_out}" == *"Publisher count: "* ]]; then
        # Strip whitespace + extract publisher count.
        pub_count=$(echo "${info_out}" | awk '/Publisher count:/ { print $3 }')
        if [[ "${pub_count}" -gt 0 ]]; then
            ok "/aima/hal/joint/${group}/command  publishers=${pub_count}"
        else
            warn "/aima/hal/joint/${group}/command  publishers=0 (MC not active?)"
        fi
    else
        warn "could not query /aima/hal/joint/${group}/command (DDS isolation?)"
    fi
done

# -------------------------------------------------------------------------
# 6. EM service reachable (either HTTP or ROS 2)
#
# Older firmwares (<2026-04) shipped an HTTP front-end at
# http://${PC1_HOST}:50080/x2/em/{start,stop}_app and that's what
# x2_pc2_daemons.sh stop calls. Newer firmwares (this Orin NX, 2026-05+)
# drop the HTTP wrapper and expose the same surface as ROS 2 services
# under /aimdk.protocol.EmAppService/{Start,Stop}App. Probe both.
# -------------------------------------------------------------------------
section "6. EM service (HTTP on ${PC1_HOST:-PC1_HOST unset}:${PC1_EM_PORT} -or- ROS 2 EmAppService)"
if [[ -z "${PC1_HOST}" ]]; then
    warn "PC1_HOST not set (pass --pc1-host <PC1_IP> or export PC1_HOST) -- skipping EM service probe"
else
em_tcp=$(ssh_pc2 "curl -s -m 3 -o /dev/null -w '%{http_code}' \
                  'http://${PC1_HOST}:${PC1_EM_PORT}/'" 2>/dev/null || echo "000")
em_app=$(ssh_pc2 "curl -s -m 3 -o /dev/null -w '%{http_code}' \
                  'http://${PC1_HOST}:${PC1_EM_PORT}/x2/em/start_app?app=mc'" 2>/dev/null || echo "000")
em_ros_present=$(ssh_pc2 "source /opt/ros/${ROS_DISTRO}/setup.bash 2>/dev/null && \
                          timeout 3 ros2 service list 2>/dev/null | \
                          grep -c 'EmAppService/StartApp'" 2>/dev/null || echo 0)
if [[ "${em_tcp}" == "000" ]]; then
    fail "EM HTTP TCP unreachable on ${PC1_HOST}:${PC1_EM_PORT}"
elif [[ "${em_app}" == "200" || "${em_app}" == "204" ]]; then
    ok "EM HTTP /x2/em/start_app reachable (${em_app})"
elif [[ "${em_ros_present}" -ge 1 ]]; then
    ok "EM via ROS 2 service /aimdk.protocol.EmAppService/StartApp present (firmware drops the HTTP wrapper)"
    warn "x2_pc2_daemons.sh stop's MC-damp call uses the legacy HTTP path (${em_app}) -- use --no-mc-restart on stop, or rely on the deploy's own ramp-out for damping. Tracked as TODO."
else
    warn "EM HTTP server up (root -> ${em_tcp}) but neither /x2/em/start_app (${em_app}) nor ROS 2 EmAppService present"
fi
fi  # PC1_HOST set

# -------------------------------------------------------------------------
# 7. Log root
# -------------------------------------------------------------------------
section "7. Log root on PC2 (${PC2_LOG_ROOT})"
if ssh_pc2 "[ -d ${PC2_LOG_ROOT} ] && [ -w ${PC2_LOG_ROOT} ]" 2>/dev/null; then
    ok "${PC2_LOG_ROOT} exists and is writable by ${PC2_USER}"
elif ssh_pc2 "[ -d ${PC2_LOG_ROOT} ]" 2>/dev/null; then
    fail "${PC2_LOG_ROOT} exists but is not writable by ${PC2_USER}"
else
    warn "${PC2_LOG_ROOT} missing -- x2_pc2_daemons.sh start will mkdir at boot"
fi

# -------------------------------------------------------------------------
# 8. Free ports
# -------------------------------------------------------------------------
section "8. Free ports on PC2 (5557 deploy x2_debug; 5567 motor monitor)"
for port in 5557 5567; do
    in_use=$(ssh_pc2 "ss -tln 2>/dev/null | awk 'NR>1 && \$4 ~ /:${port}\$/ { print \$4 }'" 2>/dev/null || true)
    if [[ -z "${in_use}" ]]; then
        ok "port ${port} free"
    else
        warn "port ${port} appears bound: ${in_use} (x2_pc2_daemons.sh stop first?)"
    fi
done

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo
printf '%sSummary:%s pass=%d  warn=%d  fail=%d\n' "${C_BLUE}" "${C_RESET}" "${PASS}" "${WARN}" "${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
    exit 1
fi
exit 0
