# X2 Docker container (sim + real robot)

Lightweight ROS 2 Humble + ONNX Runtime container for running the X2 Ultra
MuJoCo bridge (`scripts/x2_mujoco_ros_bridge.py`) AND the C++ deploy node
(`agi_x2_deploy_onnx_ref`), on hosts that don't natively have ROS 2 Humble
(e.g. Ubuntu 24.04 noble).

Two DDS modes are supported by layering compose files:

| Wrapper | Compose files | DDS mode | Pair with |
| --- | --- | --- | --- |
| `./enter_sim.sh` | `docker-compose.yml` | loopback (`ROS_LOCALHOST_ONLY=1`, `ROS_DOMAIN_ID=73`) | `deploy_x2.sh sim` |
| `./get_x2_sonic_ready.sh` | `docker-compose.yml` + `docker-compose.real.yml` | SDK ethernet (`ROS_DOMAIN_ID=0`, CycloneDDS pinned to `enp10s0`) | `deploy_x2.sh local` |

> **For the full architecture (host vs container split, ROS topology, data
> flow, viewer setup, future Option B), see [ARCHITECTURE.md](./ARCHITECTURE.md).**

The G1 deploy uses `gear_sonic_deploy/docker/` (CUDA + TensorRT + ONNX
Runtime, ~6 GB). That image is overkill for X2 sim because the bridge is
pure Python and the deploy here doesn't need GPU inference. This image
descends from `ros:humble-ros-base` plus `aimdk_msgs`, `mujoco`, and
CPU `torch` -- enough to bring up the bridge end of the loop in ~3 GB.

## Layout

| File | Purpose |
| --- | --- |
| `Dockerfile` | `ros:humble-ros-base` + `aimdk_msgs` + ONNX Runtime + bridge Python deps |
| `docker-compose.yml` | Bind-mounts the monorepo at `/workspace/sonic`, sets sim DDS env |
| `docker-compose.real.yml` | Overlay that flips DDS to the SDK ethernet for real-robot mode |
| `enter_sim.sh` | Build (idempotent) + drop into the container shell, sim DDS |
| `get_x2_sonic_ready.sh` | Build (idempotent) + drop into the container shell, real-robot DDS |

## Quick start (sim: bridge-only smoke test)

```bash
cd gear_sonic_deploy/docker_x2
./enter_sim.sh
```

That builds the image (~10 min first time, ~1 s once cached) and gives you
a bash shell inside the container with ROS 2 + `aimdk_msgs` overlaid. From
there:

```bash
# Inside the container:
cd /workspace/sonic/gear_sonic_deploy
python3 scripts/x2_mujoco_ros_bridge.py --print-scene
```

In a second host shell, attach to the same compose service to inspect
topics:

```bash
cd gear_sonic_deploy/docker_x2
docker compose exec x2sim bash -c \
    'source /opt/ros/humble/setup.bash && \
     source /ros2_ws/install/setup.bash && \
     ros2 topic list | grep aima'
```

You should see joint state for all 4 groups (`leg`, `waist`, `arm`, `head`)
plus `/aima/hal/imu/torso/state` published at the rates baked into the
bridge (200 Hz state, 500 Hz IMU).

## DDS isolation

`docker-compose.yml` exports `ROS_LOCALHOST_ONLY=1` and `ROS_DOMAIN_ID=73`
inside the container -- same defaults as `deploy_x2.sh sim`. Combined with
`network_mode: host`, that means the sim and any host-side `ros2` clients
on domain 73 can see each other on loopback, but cannot touch a real robot
on the SDK ethernet (which runs on domain 0).

For real-robot work, `docker-compose.real.yml` overlays the env to set
`ROS_LOCALHOST_ONLY=0`, `ROS_DOMAIN_ID=0`, and `CYCLONEDDS_URI` pinned to
`enp10s0`. Use `./get_x2_sonic_ready.sh` (which composes both files) instead of
calling `docker compose` by hand.

## Real-robot quick start

Pre-flight outside the container (operator does this once):

1. SDK cable plugged into one of PC2/PC3's rear RJ45 dev ports.
2. Host SDK NIC (`X2_SDK_NIC`, e.g. `enp10s0`) configured static on the SDK subnet (`<LAPTOP_IP>/24`).
3. `ping $PC2_HOST` succeeds (`export PC2_HOST=<PC2_IP>`; PC1's EM API is `PC1_HOST=<PC1_IP>`).

Then:

```bash
cd gear_sonic_deploy/docker_x2
./get_x2_sonic_ready.sh -- ros2 topic list | grep aima   # one-shot discovery smoke test
# Should list /aima/hal/joint/{leg,waist,arm,head}/{state,command} +
# /aima/hal/imu/torso/state.

./get_x2_sonic_ready.sh                                  # interactive shell
# Inside the container, run the deploy:
cd gear_sonic_deploy
./deploy_x2.sh local --model /workspace/checkpoints/run-XXXXXXXX/exported/model_step_XXXXXX_g1.onnx \
    --motion data/motions_x2m2/x2_ultra_idle_stand.x2m2 \
    --dry-run --autostart-after 5 \
    --log-dir /tmp/x2_dryrun_$(date +%Y%m%d_%H%M%S)
```
