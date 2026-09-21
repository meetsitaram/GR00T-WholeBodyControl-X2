#!/usr/bin/env bash
# x2_pc2_cameras.sh -- inspect / restart-hal / grab samples for the four
# head cameras that hang off PC2 (the Jetson Orin NX). Three of them only
# come up reliably after an `aima em` bounce of `hal_sensor_orin`, because
# at boot Argus has a known race with `orbbec_camera` and ends up with
# only one of the three IMX900 GMSL sensors registered. Bouncing after
# the Orbbec daemon is stable wins the race deterministically.
#
# Physical inventory on PC2 (per ``v4l2-ctl --list-devices`` + dmesg):
#   * 1x Orbbec Gemini 335 (USB 3, ``rgbd_head_front``) ->
#       /aima/hal/sensor/rgb_head_front_center/rgb_image{,/compressed,/camera_info}
#       Native RGB 2688x1944 @ 30 Hz, depth+IR streams via the
#       ``orbbec_camera`` em-app.
#   * 3x Sony IMX900 (CSI via MAX9295A/D GMSL serializers, mounted on
#     the head as ``stereo_head_front_{left,right}`` + ``rgb_head_rear``)
#       /aima/hal/sensor/stereo_head_front_left/rgb_image{,/compressed,/camera_info}
#       /aima/hal/sensor/stereo_head_front_right/rgb_image{,/compressed,/camera_info}
#       /aima/hal/sensor/rgb_head_rear/rgb_image{,/compressed,/camera_info}
#     Native Bayer 2064x1552 @ 30 Hz, debayered + published by
#     ``hal_sensor_orin``'s Argus client. Includes the stereo head IMU
#     under /aima/hal/sensor/stereo_head_front/imu.
#
# Usage:
#
#   ./gear_sonic_deploy/scripts/x2_pc2_cameras.sh status
#     -> Show what publishers exist for each of the 4 image topics and
#        which em-apps are running (orbbec_camera, hal_sensor_orin).
#
#   ./gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal [--deep]
#     -> Bounce `hal_sensor_orin` (`aima em stop-app` + `start-app`) to
#        re-arm the 3 IMX900 sensors after a boot where they came up
#        missing (typical symptom: status shows the Orbbec topic present
#        but the stereo + rear topics absent). Safe to run at any time --
#        it briefly drops joint state / IMU / LiDAR publishers for ~3 s
#        while the daemon restarts.
#        ``kick`` is accepted as a deprecated alias and prints a warning.
#
#        With ``--deep`` it also restarts ``nvargus-daemon`` between the
#        stop and start (requires passwordless sudo on PC2). Use this if
#        a plain ``restart-hal`` still leaves the stereo topics at
#        ``pubs=0`` and ``journalctl -u nvargus-daemon`` is full of
#        ``acquireBuffer() Error Timeout`` / ``ChanselFault`` / ``FALCON_ERROR``
#        lines. That means the libargus server itself is in a bad state
#        (often after a previous CameraProvider was force-destroyed) and
#        bouncing only the HAL client reconnects to the same stuck daemon.
#        Escalation ladder when stereo cameras won't come up:
#          1. ``restart-hal``         (cheap, fixes Argus boot race)
#          2. ``restart-hal --deep``  (resets libargus session state)
#          3. ``ssh run@PC2 sudo reboot`` (last resort -- needed when the
#             kernel reports ``imx900 sensor_recovery: Not Active`` +
#             ``ChanselFault`` after a deep bounce; the GMSL/CSI link
#             needs the camera reset lines re-asserted at boot).
#
#   ./gear_sonic_deploy/scripts/x2_pc2_cameras.sh grab [OUT_DIR]
#     -> Subscribe to each of the 4 compressed image topics on PC2 (using
#        AgiBot's FastDDS profile so discovery actually works), save one
#        JPEG per camera, and scp the four files back to OUT_DIR
#        (default: /tmp/x2_cam_samples on the laptop). Useful to confirm
#        each camera is alive and the framing/focus is good.
#
#   ./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve
#     -> Ship + launch the ROS->ZMQ camera bridge
#        (x2_pc2_camera_zmq_publisher.py) on PC2 in the background.
#        The bridge subscribes to the head_front Orbbec + stereo
#        left/right IMX900 topics, resizes to 640x480 at the source,
#        and republishes them as a merged ``ImageMessageSchema`` ZMQ
#        PUB on tcp://*:5555 (laptop-consumable via
#        ``ComposedCameraClientSensor(server_ip='<PC2_IP>', port=5555)``).
#        The recorder's ``--head-cameras`` flag connects to this.
#        Idempotent: bounces any prior instance before relaunching.
#
#   ./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-stop
#     -> Kill any running bridge on PC2.
#
#   ./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-log
#     -> Tail the bridge's log on PC2 (ctrl-c to detach; bridge keeps
#        running).
#
# Flags (all subcommands):
#
#   --host HOST     PC2 hostname/IP. Default: ${X2_PC2_HOST:-$PC2_HOST}; REQUIRED.
#   --user USER     SSH user. Default: ${X2_PC2_USER:-run}.
#   --port PORT     ZMQ PUB port (serve subcommand). Default 5555.
#   --width N       Bridge resize width (serve subcommand). Default 640.
#   --height N      Bridge resize height (serve subcommand). Default 480.
#   --include-rear  Include the rear head camera in the bridge output.
#   -h, --help      Show this help.
#
# Exit status: 0 on success, non-zero on SSH/EM/ROS failure.

set -euo pipefail

NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'

HOST="${X2_PC2_HOST:-${PC2_HOST:-}}"
USER_NAME="${X2_PC2_USER:-run}"
ACTION=""
GRAB_OUT_DIR=""
SERVE_PORT="${X2_PC2_CAM_PORT:-5555}"
SERVE_WIDTH="${X2_PC2_CAM_WIDTH:-640}"
SERVE_HEIGHT="${X2_PC2_CAM_HEIGHT:-480}"
SERVE_INCLUDE_REAR=false
RESTART_HAL_DEEP=false

REMOTE_BRIDGE_SCRIPT="/tmp/x2_pc2_camera_zmq_publisher.py"
REMOTE_BRIDGE_LOG="/tmp/x2_pc2_camera_zmq_publisher.log"
LOCAL_BRIDGE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/x2_pc2_camera_zmq_publisher.py"

print_help() {
    sed -n '/^# x2_pc2_cameras.sh/,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '$d' -e 's/^# \{0,1\}//'
}

while (( $# > 0 )); do
    case "$1" in
        status|restart-hal|grab|serve|serve-stop|serve-log)
            ACTION="$1"; shift ;;
        kick)
            echo "[x2_pc2_cameras] WARNING: 'kick' is deprecated; use" \
                 "'restart-hal' (does the same thing)." >&2
            ACTION="restart-hal"; shift ;;
        --host)           HOST="$2"; shift 2 ;;
        --user)           USER_NAME="$2"; shift 2 ;;
        --port)           SERVE_PORT="$2"; shift 2 ;;
        --width)          SERVE_WIDTH="$2"; shift 2 ;;
        --height)         SERVE_HEIGHT="$2"; shift 2 ;;
        --include-rear)   SERVE_INCLUDE_REAR=true; shift ;;
        --deep)           RESTART_HAL_DEEP=true; shift ;;
        -h|--help)        print_help; exit 0 ;;
        *)
            if [[ "$ACTION" == "grab" && -z "$GRAB_OUT_DIR" && "$1" != -* ]]; then
                GRAB_OUT_DIR="$1"; shift
            else
                echo "[x2_pc2_cameras] unknown arg: $1" >&2
                print_help >&2
                exit 2
            fi ;;
    esac
done

if [[ -z "${HOST}" ]]; then
    echo "[x2_pc2_cameras] PC2_HOST is not set: pass --host <PC2_IP> or export PC2_HOST / X2_PC2_HOST." >&2
    exit 2
fi

if [[ -z "$ACTION" ]]; then
    print_help
    exit 2
fi

SSH_OPTS=( -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR )

run_remote() {
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "bash -s" "$@"
}

# ────────────────────────────────────────────────────────────────────────
# Topics we care about
# ────────────────────────────────────────────────────────────────────────
TOPICS=(
    "/aima/hal/sensor/rgb_head_front_center/rgb_image"
    "/aima/hal/sensor/stereo_head_front_left/rgb_image"
    "/aima/hal/sensor/stereo_head_front_right/rgb_image"
    "/aima/hal/sensor/rgb_head_rear/rgb_image"
)
NAMES=(
    "rgbd_head_front_orbbec"
    "stereo_head_front_left"
    "stereo_head_front_right"
    "rgb_head_rear"
)

# ────────────────────────────────────────────────────────────────────────
# status
# ────────────────────────────────────────────────────────────────────────
do_status() {
    run_remote <<'REMOTE'
set -e
source /opt/ros/humble/setup.bash 2>/dev/null
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
echo "=== em-app state (orbbec_camera, hal_sensor_orin) ==="
aima em doctor 2>&1 | grep -E "orbbec_camera|hal_sensor_orin" || echo "(neither app running)"
echo
echo "=== publisher count per head image topic ==="
for t in /aima/hal/sensor/rgb_head_front_center/rgb_image \
         /aima/hal/sensor/stereo_head_front_left/rgb_image \
         /aima/hal/sensor/stereo_head_front_right/rgb_image \
         /aima/hal/sensor/rgb_head_rear/rgb_image; do
  cnt=$(timeout 3 ros2 topic info "$t" 2>&1 | awk -F: '/Publisher count/ {print $2}' | tr -d ' ')
  printf "  %-65s pubs=%s\n" "$t" "${cnt:-0}"
done
REMOTE
}

# ────────────────────────────────────────────────────────────────────────
# restart-hal: bounce hal_sensor_orin (+ optionally nvargus-daemon) to
# recover the 3 IMX900 GMSL sensors.
#
# Plain mode just stops/starts the HAL client (cheap, fixes the boot
# race). ``--deep`` additionally restarts nvargus-daemon between the
# stop and start, which is needed when libargus itself is in a bad
# state (acquireBuffer timeouts / ChanselFault / FALCON_ERROR in
# ``journalctl -u nvargus-daemon``).
# ────────────────────────────────────────────────────────────────────────
do_restart_hal() {
    local deep="${RESTART_HAL_DEEP:-false}"
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" \
        env DEEP="${deep}" bash -s <<'REMOTE'
set -e
deep="${DEEP:-false}"
echo "Stopping hal_sensor_orin ..."
aima em stop-app hal_sensor_orin 2>&1 | tail -1
sleep 3
if [[ "${deep}" == "true" ]]; then
    echo "Restarting nvargus-daemon (deep mode, requires sudo) ..."
    if sudo -n systemctl restart nvargus-daemon 2>&1; then
        sleep 3
        echo "  nvargus-daemon: $(systemctl is-active nvargus-daemon)"
    else
        echo "  ERROR: sudo systemctl restart nvargus-daemon failed."
        echo "  Hint: ensure NOPASSWD sudo for nvargus-daemon on PC2,"
        echo "        or run \"sudo systemctl restart nvargus-daemon\""
        echo "        manually and re-run restart-hal."
        exit 1
    fi
fi
echo "Starting hal_sensor_orin ..."
aima em start-app hal_sensor_orin 2>&1 | tail -1
# When we bounced nvargus too, give Argus + GMSL re-negotiation
# longer to settle before sampling publisher counts.
if [[ "${deep}" == "true" ]]; then sleep 12; else sleep 5; fi
echo
source /opt/ros/humble/setup.bash 2>/dev/null
echo "=== post-bounce: publisher count per image topic ==="
stereo_zero=0
for t in /aima/hal/sensor/rgb_head_front_center/rgb_image \
         /aima/hal/sensor/stereo_head_front_left/rgb_image \
         /aima/hal/sensor/stereo_head_front_right/rgb_image \
         /aima/hal/sensor/rgb_head_rear/rgb_image; do
  cnt=$(timeout 3 ros2 topic info "$t" 2>&1 | awk -F: '/Publisher count/ {print $2}' | tr -d ' ')
  printf "  %-65s pubs=%s\n" "$t" "${cnt:-0}"
  if [[ "$t" == *stereo_head_front* && "${cnt:-0}" -lt 1 ]]; then
    stereo_zero=1
  fi
done
if [[ "${stereo_zero}" -eq 1 ]]; then
  echo
  if [[ "${deep}" == "true" ]]; then
    echo "WARNING: stereo cameras still pubs=0 after --deep bounce."
    echo "  Likely a CSI/GMSL hardware-state issue (the IMX900 sensors"
    echo "  register but never go Active). Recovery path:"
    echo "    1. Check 'sudo dmesg | grep -iE imx900|max929' on PC2 for"
    echo "       'sensor_recovery: Not Active' + 'recovery_check err'."
    echo "    2. If those messages persist, only a PC2 reboot"
    echo "       (re-assert camera reset lines) fixes it:"
    echo "         ssh ${USER:-run}@<pc2> sudo reboot"
  else
    echo "WARNING: stereo cameras still pubs=0 after restart-hal."
    echo "  Try the deeper recovery that also bounces libargus:"
    echo "    $0 restart-hal --deep --host <pc2>"
  fi
fi
REMOTE
}

# ────────────────────────────────────────────────────────────────────────
# grab: capture one frame from each of the 4 cameras
# ────────────────────────────────────────────────────────────────────────
do_grab() {
    local out_dir="${GRAB_OUT_DIR:-/tmp/x2_cam_samples}"
    mkdir -p "$out_dir"
    rm -f "$out_dir"/*.jpg

    echo -e "${BLUE}[x2_pc2_cameras] grabbing one frame per camera ...${NC}"

    run_remote <<'REMOTE'
set -e
mkdir -p /tmp/x2_cam_samples
rm -f /tmp/x2_cam_samples/*.jpg
cat > /tmp/x2_grab_one.py <<'PYEOF'
import sys, time, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage

topic = sys.argv[1]
out_path = sys.argv[2]
rclpy.init()
n = Node("x2_oneshot_grabber")
qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                 reliability=QoSReliabilityPolicy.BEST_EFFORT)
got = []
def cb(msg):
    got.append(bytes(msg.data))
n.create_subscription(CompressedImage, topic, cb, qos)
t0 = time.time()
while not got and time.time() - t0 < 6.0:
    rclpy.spin_once(n, timeout_sec=0.2)
if got:
    with open(out_path, "wb") as f:
        f.write(got[0])
    print(f"OK {len(got[0])} bytes -> {out_path}")
else:
    print("NO_FRAME")
rclpy.shutdown()
PYEOF
source /opt/ros/humble/setup.bash 2>/dev/null
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
declare -A TOPIC_NAMES=(
  ["/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed"]="rgbd_head_front_orbbec"
  ["/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed"]="stereo_head_front_left"
  ["/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed"]="stereo_head_front_right"
  ["/aima/hal/sensor/rgb_head_rear/rgb_image/compressed"]="rgb_head_rear"
)
for topic in "${!TOPIC_NAMES[@]}"; do
  name=${TOPIC_NAMES[$topic]}
  out=/tmp/x2_cam_samples/${name}.jpg
  result=$(timeout 12 python3 /tmp/x2_grab_one.py "$topic" "$out" 2>&1 | tail -1)
  dims=""
  if [ -s "$out" ]; then
    dims=$(python3 -c "from PIL import Image; im=Image.open('$out'); print(im.size, im.mode)" 2>/dev/null)
  fi
  printf "  %-30s %s | dims=%s\n" "$name" "$result" "$dims"
done
REMOTE

    echo -e "${BLUE}[x2_pc2_cameras] copying samples back to ${out_dir} ...${NC}"
    scp "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}:/tmp/x2_cam_samples/*.jpg" "$out_dir/" 2>&1 \
        | grep -v "^$" || true

    echo
    echo -e "${GREEN}[x2_pc2_cameras] saved samples:${NC}"
    ls -la "$out_dir"/*.jpg 2>/dev/null || echo "  (no JPEGs landed)"
}

# ────────────────────────────────────────────────────────────────────────
# serve: ship + launch the ROS→ZMQ bridge in the background on PC2
# ────────────────────────────────────────────────────────────────────────
do_serve() {
    if [[ ! -f "${LOCAL_BRIDGE_SCRIPT}" ]]; then
        echo "[x2_pc2_cameras] ERROR: bridge script not found at" \
             "${LOCAL_BRIDGE_SCRIPT}" >&2
        exit 1
    fi

    echo -e "${BLUE}[x2_pc2_cameras] shipping bridge to PC2 ...${NC}"
    scp "${SSH_OPTS[@]}" "${LOCAL_BRIDGE_SCRIPT}" \
        "${USER_NAME}@${HOST}:${REMOTE_BRIDGE_SCRIPT}" >/dev/null

    echo -e "${BLUE}[x2_pc2_cameras] (re)launching bridge ...${NC}"
    local extra=""
    if ${SERVE_INCLUDE_REAR}; then extra="--include-rear"; fi

    # The bridge needs PC2 to have rclpy + pyzmq + msgpack-numpy on the
    # system Python (we pip-install --user the latter two on first
    # use). We re-run the install line every time so a wiped PC2 still
    # works on the next ``serve``.
    #
    # We pass the bridge script PATH into the remote heredoc via stdin
    # (read on the first ``read`` call), NOT via positional argv,
    # because the path string is what we ``pkill -f`` on -- and if we
    # put it in argv ssh embeds it in /proc/<sshpid>/cmdline, where
    # pkill matches the ssh session itself and SIGTERMs its own
    # parent.
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" \
        env REMOTE_BRIDGE_SCRIPT="${REMOTE_BRIDGE_SCRIPT}" \
            REMOTE_BRIDGE_LOG="${REMOTE_BRIDGE_LOG}" \
            REMOTE_BRIDGE_PORT="${SERVE_PORT}" \
            REMOTE_BRIDGE_WIDTH="${SERVE_WIDTH}" \
            REMOTE_BRIDGE_HEIGHT="${SERVE_HEIGHT}" \
            REMOTE_BRIDGE_EXTRA="${extra}" \
        bash -s <<'REMOTE'
# Drop ``-u``: /opt/ros/humble/setup.bash touches a handful of unset
# variables (CMAKE_PREFIX_PATH, AMENT_PREFIX_PATH, ...) and would
# otherwise SIGABRT here.
set -e
script="${REMOTE_BRIDGE_SCRIPT}"
log="${REMOTE_BRIDGE_LOG}"
port="${REMOTE_BRIDGE_PORT}"
width="${REMOTE_BRIDGE_WIDTH}"
height="${REMOTE_BRIDGE_HEIGHT}"
extra="${REMOTE_BRIDGE_EXTRA:-}"
# Kill any prior bridge instance. Match on the Python invocation
# (``python3 <path>``) so we don't also kill our own ssh session
# (whose argv may contain the script name when passed positionally).
# Use ``[[:space:]]+`` to tolerate one-or-many spaces between the
# interpreter and the script path.
pkill -f "python3[[:space:]]+${script}" 2>/dev/null || true
sleep 0.5

# Make sure the runtime deps are present; idempotent + fast when
# already installed.
python3 -c "import zmq, msgpack, msgpack_numpy" 2>/dev/null \
    || pip3 install --user --quiet pyzmq msgpack msgpack-numpy
# PC2's system cv2 / ROS stack is compiled against numpy 1.x. A stray
# user-site numpy 2.x shadows the system copy and kills the bridge at
# ``import cv2`` with "_ARRAY_API not found" (hit live 2026-07-28:
# numpy 2.2.6 in ~/.local). Prefer UNINSTALLING the user-site shadow
# (offline-safe; JetPack's system numpy 1.x underneath is the one cv2
# was built against); pip-install "numpy<2" only if numpy is then
# entirely missing. Idempotent no-op when already healthy.
if ! python3 -c 'import numpy, sys; sys.exit(0 if numpy.__version__.startswith("1.") else 1)' 2>/dev/null; then
    pip3 uninstall --yes --quiet numpy 2>/dev/null || true
    python3 -c 'import numpy' 2>/dev/null \
        || pip3 install --user --quiet "numpy<2"
fi

source /opt/ros/humble/setup.bash 2>/dev/null
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml

nohup python3 "$script" \
        --port "$port" --width "$width" --height "$height" $extra \
        > "$log" 2>&1 < /dev/null &
disown
sleep 1.8
if pgrep -f "python3[[:space:]]+${script}" > /dev/null; then
    pid=$(pgrep -f "python3[[:space:]]+${script}" | head -1)
    echo "[bridge] running, pid=${pid}, log=${log}"
    tail -5 "$log" || true
else
    echo "[bridge] FAILED to launch; recent log:" >&2
    tail -20 "$log" >&2 || true
    exit 1
fi
REMOTE

    echo
    echo -e "${GREEN}[x2_pc2_cameras] bridge serving on" \
            "tcp://${HOST}:${SERVE_PORT} (${SERVE_WIDTH}x${SERVE_HEIGHT}).${NC}"
    echo "  laptop recorder consumes via:"
    echo "    --head-cameras --camera-host ${HOST} --camera-port ${SERVE_PORT}"
    echo "  tail log:  $0 serve-log --host ${HOST}"
    echo "  stop:      $0 serve-stop --host ${HOST}"
}

# ────────────────────────────────────────────────────────────────────────
# serve-stop: kill any running bridge on PC2
# ────────────────────────────────────────────────────────────────────────
do_serve_stop() {
    # Match on the python invocation only (avoid killing our own ssh
    # session whose argv may also contain the script name).
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" \
        "pkill -f 'python3[[:space:]]+.*x2_pc2_camera_zmq_publisher.py' && \
         echo '[bridge] stopped' || echo '[bridge] (no instance found)'"
}

# ────────────────────────────────────────────────────────────────────────
# serve-log: tail the bridge log on PC2 (read-only follow)
# ────────────────────────────────────────────────────────────────────────
do_serve_log() {
    echo "[x2_pc2_cameras] tailing ${REMOTE_BRIDGE_LOG} on ${HOST}" \
         "(ctrl-C to detach; bridge keeps running)"
    ssh -t "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" \
        "tail -n 50 -f '${REMOTE_BRIDGE_LOG}'"
}

case "$ACTION" in
    status)      do_status ;;
    restart-hal) do_restart_hal ;;
    grab)        do_grab ;;
    serve)       do_serve ;;
    serve-stop)  do_serve_stop ;;
    serve-log)   do_serve_log ;;
    *)
        echo "[x2_pc2_cameras] unknown action: $ACTION" >&2
        exit 2 ;;
esac
