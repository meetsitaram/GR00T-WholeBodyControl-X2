# X2 Sim Architecture

How the X2 closed-loop MuJoCo simulation is split between **host** and
**container**, what runs where, and how data flows between the bridge and
the C++ deploy node. Mostly a snapshot of the conversation that built
`docker_x2/` so future-you doesn't have to re-derive it.

If you change `docker-compose.yml`, `Dockerfile`, `deploy_x2.sh sim`, or
`scripts/x2_mujoco_ros_bridge.py`, please update this doc too.

## TL;DR

- **Container** runs ROS 2 Humble + ONNX Runtime + colcon-built C++ deploy
  + the Python MuJoCo bridge. Two processes inside one container, talking
  over loopback DDS on `ROS_DOMAIN_ID=73`.
- **Host** owns the source tree (bind-mounted), the ONNX checkpoints
  (read-only mount), the GPU (passed in only when `--sim-viewer` is
  requested), and the X11 display.
- The container is a **necessary evil**: Ubuntu 24.04 (noble) doesn't ship
  ROS 2 Humble, and the deploy needs Humble + `aimdk_msgs` + ONNX Runtime
  1.16.3 in the same place CMake expects. The split-architecture
  alternative (bridge on host via Robostack conda, deploy in container) is
  documented at the bottom under "Future / Option B".

## Where things live

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ HOST   Ubuntu 24.04 (kernel 6.17, NVIDIA + AMD GPUs, X11 :1)        │
│                                                                              │
│  ┌──────────── filesystem (bind-mounted into container) ─────────────┐       │
│  │  <repo root>/  (monorepo, rw)                                      │       │
│  │      gear_sonic/scripts/eval_x2_mujoco.py     ── source of truth  │       │
│  │      gear_sonic/utils/mujoco_sim/...          ── G1 sim (band)    │       │
│  │      gear_sonic_deploy/                                           │       │
│  │          deploy_x2.sh                  ← user runs this           │       │
│  │          scripts/x2_mujoco_ros_bridge.py                          │       │
│  │          src/x2/agi_x2_deploy_onnx_ref/  (C++, colcon-built)      │       │
│  │          docker_x2/{Dockerfile,docker-compose*.yml,enter_*.sh}    │       │
│  │  $X2_CHECKPOINTS_DIR/  (ro mount → /workspace/checkpoints)        │       │
│  │      run-*/exported/model_step_*.onnx                             │       │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌──────────── conda env_isaaclab (NOT used by sim today) ───────────┐       │
│  │  python3.11 + mujoco 3.7.0 + torch+CUDA + Isaac Lab (training)    │       │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌──────────── docker engine ─────────────────────────────────────────┐      │
│  │  runtimes: runc (default), nvidia (used when --sim-viewer is on)   │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌──────────── X11 server :1 (mounted via /tmp/.X11-unix) ────────────┐      │
│  │  xhost +SI:localuser:root grants container root access             │      │
│  │  (run automatically by docker_x2/enter_{sim,robot}.sh)             │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  loopback DDS                                                                │
│  ┌──── network_mode: host ─────────────── shares ──────────────────┐         │
│                                                                    │         │
└────────────────────────────────────────────────────────────────────┼─────────┘
                                                                     │
┌────────────────────────────────────────────────────────────────────┼─────────┐
│ CONTAINER  gr00t-x2sim:latest                       (Ubuntu 22.04) │         │
│                                                                    │         │
│  /opt/ros/humble                ROS 2 Humble runtime              ─┘         │
│  /opt/onnxruntime               ONNX Runtime 1.16.3 (LD_LIBRARY_PATH set)    │
│  /ros2_ws/install/aimdk_msgs    vendored msgs (sourced)                      │
│  /workspace/sonic               ← bind mount of host monorepo                │
│  /workspace/checkpoints         ← bind mount of host $X2_CHECKPOINTS_DIR     │
│  /tmp/.X11-unix, /root/.Xauthority   ← bind mounts of host X11 socket+auth   │
│  named vols: pip cache + colcon build/install/log                            │
│                                                                              │
│  env: ROS_LOCALHOST_ONLY=1   ROS_DOMAIN_ID=73                                │
│       DISPLAY=$host           NVIDIA_DRIVER_CAPABILITIES=all                 │
│                                                                              │
│  When `deploy_x2.sh sim ...` runs, it spawns TWO processes inside this       │
│  container:                                                                  │
│                                                                              │
│  ┌────────────────── Process 1 ─────────────────┐                            │
│  │ python3 scripts/x2_mujoco_ros_bridge.py      │                            │
│  │                                              │                            │
│  │  Sim thread (1 kHz):                         │                            │
│  │    _apply_pd()        ← latest deploy cmd    │                            │
│  │    _apply_elastic_band() ← G1 ElasticBand    │                            │
│  │    mujoco.mj_step(model, data)               │                            │
│  │  ROS thread (rclpy spin):                    │                            │
│  │    pubs joint state @ 200 Hz                 │                            │
│  │    pubs IMU         @ 500 Hz                 │                            │
│  │    subs joint cmd  (per-group)               │                            │
│  │  Viewer (optional, --sim-viewer):            │                            │
│  │    mujoco.viewer.launch_passive              │                            │
│  │    keys: 9 toggle band, 7/8 raise/lower      │                            │
│  └──────────────────────────────────────────────┘                            │
│                                                                              │
│  ┌────────────────── Process 2 ─────────────────┐                            │
│  │ ros2 run agi_x2_deploy_onnx_ref              │                            │
│  │     x2_deploy_onnx_ref --model ... \         │                            │
│  │     --autostart-after 1 --tilt-cos 0.95            │                            │
│  │                                              │                            │
│  │  AimdkIo (rclcpp pubs/subs)                  │                            │
│  │  StandStillReference (default DOFs)          │                            │
│  │  OnnxActor (Ort::Session, 1670→31)           │                            │
│  │  Tilt watchdog → SAFE_HOLD                   │                            │
│  │  State machine: INIT → WAIT_FOR_CONTROL      │                            │
│  │                  → CONTROL → SAFE_HOLD       │                            │
│  └──────────────────────────────────────────────┘                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## ROS topology between bridge and deploy

Both processes share the host's loopback because `network_mode: host`, so
this is a single-machine DDS exchange even though the addresses are
ROS-style.

```text
                                ┌────────────────────────┐
   /aima/hal/joint/leg/state    │                        │
   /aima/hal/joint/waist/state  │                        │
   /aima/hal/joint/arm/state    │   x2_deploy_onnx_ref   │
   /aima/hal/joint/head/state   │   (C++, in container)  │
   /aima/hal/imu/torso/state    │                        │
   ──────────────────────────►  │  state in              │
                                │                        │
   ◄──────────────────────────  │  cmd out               │
   /aima/hal/joint/leg/command  │                        │
   /aima/hal/joint/waist/command│                        │
   /aima/hal/joint/arm/command  │  ONNX every 20 ms      │
   /aima/hal/joint/head/command │  obs[1670] → act[31]   │
                                └────────────────────────┘
                  ▲                          │
                  │                          │
                  │            ┌─────────────┘
                  │            ▼
                  │   ┌─────────────────────────────┐
                  │   │                             │
                  │   │  x2_mujoco_ros_bridge.py    │
                  │   │  (Python, in container)     │
                  │   │                             │
                  │   │   ┌───────────────────────┐ │
                  │   │   │  MuJoCo @ 1 kHz       │ │
                  │   │   │  x2_ultra.xml         │ │
                  └───┤   │  + ElasticBand        │ │
                      │   └───────────────────────┘ │
                      │  pubs joint state @ 200 Hz  │
                      │  pubs IMU         @ 500 Hz  │
                      └─────────────────────────────┘

   QoS for all topics: rclpy.qos.qos_profile_sensor_data
                       (BEST_EFFORT, VOLATILE, KEEP_LAST 10)
                       — matches aimdk_io.cpp exactly
```

## One closed-loop cycle in steady state

```text
   t = 0          MuJoCo step (1 ms)
                  bridge PD applies last action as torque
                  bridge: mj_step → qpos / qvel / xquat update

   t = 5 ms       bridge publishes joint state (every 5 ms = 200 Hz)
                  bridge publishes IMU         (every 2 ms = 500 Hz)
                       │
                       ▼
                  DDS over loopback
                       │
                       ▼
   t ≈ 5 ms +    deploy AimdkIo callback caches latest state
                  deploy spin loop @ 50 Hz (every 20 ms):
                      build obs[1670] from joint history + IMU
                      OnnxActor::Forward(obs) → act[31]   (~3-5 ms CPU)
                      ramp + clamp → JointCommandArray
                  deploy publishes 4 command msgs
                       │
                       ▼
                  DDS over loopback
                       │
                       ▼
   t + ε         bridge cmd callbacks update _target_pos/_target_vel
                  next mj_step uses the new PD setpoints
```

## Host vs container responsibility matrix

| Concern              | Host                                | Container                         |
| -------------------- | ----------------------------------- | --------------------------------- |
| MuJoCo physics       | mujoco 3.7.0 in conda (idle)        | **mujoco wheel in venv (active)** |
| MuJoCo viewer        | renders here via X11 (when enabled) | GLFW window forwarded over X11    |
| ROS 2 Humble runtime | none                                | **/opt/ros/humble**               |
| `aimdk_msgs`         | none                                | **/ros2_ws/install/aimdk_msgs**   |
| ONNX Runtime         | none                                | **/opt/onnxruntime (1.16.3)**     |
| C++ deploy build     | (deploy_x2.sh `local` would try)    | **colcon build via deploy_x2.sh** |
| ONNX checkpoints     | `$X2_CHECKPOINTS_DIR` (default `gear_sonic_deploy/models/`) | mounted ro at `/workspace/...` |
| Repo source          | `<repo root>`                       | bind-mounted rw at `/workspace`   |
| GPU                  | NVIDIA driver loaded                | injected via `runtime: nvidia`    |
| Display              | X11 `:1`                            | `/tmp/.X11-unix` + DISPLAY env    |

## How to enable the MuJoCo viewer (Option A, currently wired)

`docker_x2/enter_sim.sh` runs `xhost +SI:localuser:root` automatically before
`docker compose run`, and `docker-compose.yml` mounts the X11 socket + auth
cookie + uses the NVIDIA runtime. So the typical flow is:

```bash
cd gear_sonic_deploy/docker_x2
./enter_sim.sh                    # builds image, opens xhost, drops you into the container
# inside container:
cd /workspace/sonic/gear_sonic_deploy
./deploy_x2.sh sim \
    --model /workspace/checkpoints/run-XXXXXXXX/exported/model_step_XXXXX_g1.onnx \
    --sim-viewer \
    --autostart-after 5 --tilt-cos 0.95 \
    --no-build
```

Inside the viewer:

| Key         | Action                                     |
| ----------- | ------------------------------------------ |
| `9`         | Toggle the ElasticBand (suspend / drop)    |
| `7` / `8`   | Lower / raise the band's anchor height     |
| `Space`     | (mujoco built-in) pause physics            |
| `Tab`       | (mujoco built-in) cycle simulation panels  |

If the window doesn't appear:

1. Check `echo $DISPLAY` on the host before `enter_sim.sh` (must be set).
2. Check `xhost` output -- you should see `SI:localuser:root` listed.
3. Inside the container: `xeyes` (after `apt-get install -y x11-apps`) is a
   fast smoke test that doesn't depend on GLFW/MuJoCo.
4. If the host has no NVIDIA GPU, comment out `runtime: nvidia` in
   `docker-compose.yml`. GLFW will fall back to mesa software rendering.

## Future / Option B (bridge on host, deploy in container)

Cleaner architecturally but more setup. Move the bridge to a host-side
Robostack conda env and keep the deploy in the container:

- Host: install Robostack's `ros-humble-desktop` into a conda env on noble;
  build `aimdk_msgs` from source with `colcon` in that env; install the
  bridge's pip deps (`mujoco`, `numpy`, `scipy`, `joblib`).
- Container: shrink to ROS 2 Humble + ONNX Runtime + the C++ deploy. No
  more `mujoco` / `torch` / `scipy` in the image (saves ~1.5 GB).
- DDS still works via loopback because `network_mode: host` is already in
  the compose file.
- Pros: native viewer, no X11 plumbing, easier to attach a Python debugger
  to the bridge.
- Cons: two environments to keep in sync; Robostack is a moving target.

Don't migrate unless you start needing a Python debugger or you outgrow
the container for an unrelated reason. Option A is the pragmatic default.
