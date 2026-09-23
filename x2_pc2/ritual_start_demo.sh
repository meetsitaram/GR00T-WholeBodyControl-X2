#!/bin/bash
# LAPTOP-FREE demo ignition. Chain: local-upstream watchdog -> plumbing ->
# pc2 planner runtime (ONNX) -> pose-stream gate -> deploy -> pad bridge.
LOG=/home/run/gear-sonic/log/ritual_fired.log
PS=/home/run/gear-sonic/planner_stack
PY=/home/run/gear-sonic/venv/bin/python
echo "$(date +%F_%T) DEMO RITUAL START" >> $LOG
start_tmux() {  # name, command
  tmux has-session -t $1 2>/dev/null && { echo "$(date +%F_%T) $1: running -- skipped" >> $LOG; return 0; }
  # `; rc=$?` tail: record the child exit status in the ritual log so a
  # killed session is distinguishable from a crashed one. Without this the
  # only trace was the tmux pane, which dies with the session.
  tmux new-session -d -s $1 "$2 ; rc=\$?; echo \"$(date +%F_%T) $1: EXITED rc=\$rc\" >> /home/run/gear-sonic/log/ritual_fired.log"
  echo "$(date +%F_%T) $1: STARTED" >> $LOG
}
start_tmux x2_pose_watchdog "bash /home/run/gear-sonic/start_x2_pose_watchdog_local.sh"
start_tmux x2_hand_bridge   "bash /home/run/gear-sonic/log/start_x2_hand_bridge.sh"
start_tmux x2_motor_monitor "bash /home/run/gear-sonic/log/start_x2_motor_monitor.sh"
# Visualiser + scan pipeline, then the obstacle guard, both BEFORE kplanner
# so the scan is up before kplanner. The guard now starts AFTER kplanner:
# its ZMQ SUB reconnects when the publisher appears, so the clamp arms a few
# on /scan and the guard read inf. kplanner subscribes to ZMQ 5571 and zeroes
# forward/lateral while blocked, latched until the deadman is released.
# Toggle for the forward-obstacle guard (lumi scan pipeline + scan_guard_pub).
# 0 = OFF: kplanner's clamp is FAIL-OPEN, so with no publisher it never
# clamps — operator deadman is the only stop. Set to 1 to re-arm.
ENABLE_SCAN_GUARD=0   # owned by x2-scan-guard.service now

if [ "$ENABLE_SCAN_GUARD" = "1" ]; then
# lumi.sh (lidar scan pipeline) is not shipped in this release; stage your own.
[ -f /home/run/gear-sonic/lumi.sh ] && bash /home/run/gear-sonic/lumi.sh >> /home/run/gear-sonic/log/scan_guard.log 2>&1
sleep 8
fi

# Kplanner kitchen-teleop tuning (robot-verified sweep):
# FWD 0.5->0.4 (template floor ~0.45; 0.4 = smoothest), ARC_TURN default
# 0.55->0.70 (walking-turn radius ~1.2m -> 0.6-0.85m), replan threshold
# 32->48 (turn response at PC2 latency 0.68s -> 0.22s). Standing turn
# stays 1.0 (July sweep, robot-verified).
# Knobs live in ONE versioned profile now (repo: x2_pc2/robot_env.env,
# from robot_env.env.template -> ships as ${PC2_PREFIX}/robot_env.env).
# The daemon stamps every KPLANNER_* env into its tape start event, so
# each session records the exact configuration it ran.
# PYTHONPATH=$PS resolves gear_sonic.utils.* (bringup step 7b). The ONNX
# backend used here never imports motionbricks; only `--backend torch`
# does, and that path is not staged on PC2 (add $PS/motionbricks yourself
# if you stage the torch stack). x2_planner_primitives.pkl is picked up
# from $PS/models/ when bringup step 8b staged it (optional).
start_tmux pc2_kplanner "export PYTHONPATH=$PS && . /home/run/gear-sonic/robot_env.env && stdbuf -oL -eL $PY /home/run/gear-sonic/pc2_kplanner_onnx.py \
  --onnx $PS/models/planner_onnx/x2_planner_template.onnx --planner-mode slow_walk --cmd-bind --replan-threshold-frames \${KPLANNER_REPLAN_THRESHOLD_FRAMES:?robot_env.env must define it} \
  --warmup-qpos $PS/models/kplanner_idle_anchor_g1teleop_v3.pkl \
  --dances-dir $PS/models/dances_x2m2 --ort-gpu --playing-yaw-resync-dps 10 2>&1 | tee -a /home/run/gear-sonic/log/pc2_kplanner.log"
sleep 3

# SYSTEM python with absolute ROS paths: the gear_sonic venv has no rclpy and
# a fresh post-reboot shell has no ROS env, so without these spelled out the
# guard dies silently and the robot drives unguarded.
if [ "$ENABLE_SCAN_GUARD" = "1" ]; then
start_tmux scan_guard "LD_LIBRARY_PATH=/agibot/software/common/lib:/opt/ros/humble/lib \
  AMENT_PREFIX_PATH=/agibot/software/common:/opt/ros/humble \
  PYTHONPATH=/agibot/software/common/local/lib/python3.10/dist-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages \
  stdbuf -oL -eL python3 /home/run/gear-sonic/scan_guard_pub.py \
  2>&1 | tee -a /home/run/gear-sonic/log/scan_guard.log"
sleep 4
fi

# Clip banks -- FALLBACKS ONLY. x2_pc2/pad_bindings.env (sourced right after
# these, when staged alongside) is the single source of truth and overrides
# every string here. KEYS MUST MATCH THE CLIPS IN $PS/models/dances_x2m2/
# (<key>.x2m2); these are the upstream demo bank keys produced by
# tools/build_demo_bank_from_upstream.sh (manifest/REGENERATE.md).
#   L1+Y -> EASY   L1+B -> MEDIUM   L1+X -> COMBAT   L1+A -> GESTURES
#   L1+R1 -> STOP
EASY_DANCES=""
COMBAT=""
MEDIUM=""
# RIGHT-STICK 4-way (deadman RELEASED), positional: LEFT,RIGHT,UP,DOWN.
TURNS=""
GESTURES="right_wave_001,right_kiss_001,right_five_001,right_shake_001,turn_wave_right_001,turn_wave_left_001"
PAD_CLIP_DPAD_UP=""
# Shared bindings override the fallbacks above when shipped alongside.
[ -f "$(dirname "$0")/pad_bindings.env" ] && . "$(dirname "$0")/pad_bindings.env"

# PYTHONPATH/LD_LIBRARY_PATH include ROS humble dist-packages (rclpy for the
# bridge's ROS side) and gear-sonic/ itself; --cmd-bind on the planner means the
# bridge CONNECTS for planner_cmd (no --bind here). Back-ported 2026-07-29
# from the robot-verified PC2 copy that had drifted ahead of the repo.
# Optional add-on (not shipped in this release): a thermal notifier daemon.
# Only launched when the operator has staged one at ${PC2_PREFIX}/x2_thermal_notifier.py.
[ -f /home/run/gear-sonic/x2_thermal_notifier.py ] && \
start_tmux thermal_notifier "source /opt/ros/humble/setup.bash && export AMENT_PREFIX_PATH=/agibot/software/housekeeper/bin/aimdk_msgs:\$AMENT_PREFIX_PATH && export LD_LIBRARY_PATH=/agibot/software/housekeeper/bin/aimdk_msgs/lib:\$LD_LIBRARY_PATH && source /home/run/gear-sonic/ws/install/setup.bash && export PYTHONPATH=/agibot/software/housekeeper/bin/aimdk_msgs/local/lib/python3.10/dist-packages:\${PYTHONPATH:-} && $PY /home/run/gear-sonic/x2_thermal_notifier.py 2>&1 | tee -a /home/run/gear-sonic/log/thermal_notifier.log"
start_tmux pad_bridge "PYTHONPATH=/home/run/gear-sonic:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:$PS/gear_sonic LD_LIBRARY_PATH=/opt/ros/humble/lib:\$LD_LIBRARY_PATH stdbuf -oL -eL $PY /home/run/gear-sonic/pad_locomotion_bridge.py \
  --source zmq --pad-host 127.0.0.1 --lock-speed --deadman left \
  --clip-pkl $PS/models/dances_x2m2 \\
  --clip-keys \"$EASY_DANCES\" --clip-keys-b \"$COMBAT\" \
  --clip-keys-g \"$GESTURES\" --clip-keys-m \"$MEDIUM\" \
  --clip-keys-turn \"$TURNS\" --clip-keys-micro \"${MICRO:-}\" \
  --clip-key-dpad-up \"${PAD_CLIP_DPAD_UP:-}\" \\
  2>&1 | tee -a /home/run/gear-sonic/log/pad_bridge.log"
# gate: pose frames must flow (watchdog downstream :5558) before deploy exists
$PY - <<PYEOF
import zmq, sys, time
ctx = zmq.Context(); sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
sub.connect("tcp://127.0.0.1:5558"); sub.RCVTIMEO = 15000
try:
    m = sub.recv_multipart(); print(f"pose stream OK ({len(m[-1])}b)"); sys.exit(0)
except zmq.Again:
    print("NO POSE STREAM"); sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then
  echo "$(date +%F_%T) GATE FAILED -- DEPLOY NOT STARTED" >> $LOG; exit 1
fi
echo "$(date +%F_%T) GATE PASSED" >> $LOG
# PC2-RESIDENT ritual launcher (checked into repo, --no-confirm baked in),
# NEVER log/start_x2_deploy.sh: that body is regenerated by every laptop
# x2_pc2_daemons.sh start with that session's flags, and one without
# --no-confirm leaves the deploy stuck at a y/N prompt no gamepad can
# answer (2026-07-29 blocked-ignition incident).
# A DEAD pane must not block the next ignition. The old guard was
#   has-session || new-session
# so a pane left parked by a failed launch (the ritual keeps it alive on
# purpose, to preserve scrollback) made every later chord a silent no-op --
# while the line below still logged "STARTED". 2026-08-25: a bad deploy flag
# killed one launch, and the next two chords did nothing and said so in no way
# the operator could see. The log now reports what actually happened instead
# of always "STARTED".
if tmux has-session -t x2_deploy 2>/dev/null; then
    PANE="$(tmux capture-pane -p -t x2_deploy -S -400 2>/dev/null)"
    if echo "$PANE" | grep -q "cmd exited with status="; then
        tmux kill-session -t x2_deploy 2>/dev/null
        echo "$(date +%F_%T) x2_deploy: cleared post-mortem pane" >> $LOG
        tmux new-session -d -s x2_deploy "bash /home/run/gear-sonic/start_x2_deploy_ritual.sh"
        echo "$(date +%F_%T) x2_deploy: STARTED" >> $LOG
    elif echo "$PANE" | grep -q "stopping MC via PC1 EM HTTP API" \
         && ! echo "$PANE" | grep -q "start-trigger sentinel touched"; then
        # Stuck in the MC handoff (2026-09-23: with no PC1 address the
        # `aima em stop-app mc` fallback hung forever). The deploy is still
        # in STANDBY with its writer suppressed and MC owns the bus, so
        # sweeping it is safe; the chord then launches a fresh deploy.
        tmux kill-session -t x2_deploy 2>/dev/null
        echo "$(date +%F_%T) x2_deploy: cleared stuck MC handoff (never took the bus)" >> $LOG
        tmux new-session -d -s x2_deploy "bash /home/run/gear-sonic/start_x2_deploy_ritual.sh"
        echo "$(date +%F_%T) x2_deploy: STARTED" >> $LOG
    else
        echo "$(date +%F_%T) x2_deploy: already running -- SKIPPED (not started)" >> $LOG
    fi
else
    tmux new-session -d -s x2_deploy "bash /home/run/gear-sonic/start_x2_deploy_ritual.sh"
    echo "$(date +%F_%T) x2_deploy: STARTED" >> $LOG
fi

# Optional face-display reel (interact/x2_face.sh, not shipped in this release).
# Cosmetic only: runs AFTER the pose gate and after deploy start, and swallows
# every failure, so it can never delay or block ignition.
( /home/run/gear-sonic/interact/x2_face.sh on >/dev/null 2>&1 \
    && echo "$(date +%F_%T) face: logo reel ON" >> $LOG \
    || echo "$(date +%F_%T) face: logo reel FAILED (ignored)" >> $LOG ) &
