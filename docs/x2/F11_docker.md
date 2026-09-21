# F11 — Docker test image

`gear_sonic_deploy/docker_x2/` is a ROS 2 Humble + ONNX Runtime + MuJoCo
image (~3 GB, from `ros:humble-ros-base`) that runs the C++ deploy node and
the Python MuJoCo bridge on hosts without native ROS 2 Humble. It is the
supported build/test environment for the X2 deploy package and the backend of
the sim stack ([`F09_mujoco_sim.md`](F09_mujoco_sim.md)). Design notes:
`gear_sonic_deploy/docker_x2/ARCHITECTURE.md`.

| Wrapper | Compose files | DDS | Pair with |
|---|---|---|---|
| `enter_sim.sh` | `docker-compose.yml` | loopback (`ROS_LOCALHOST_ONLY=1`, `ROS_DOMAIN_ID=73`) | `deploy_x2.sh sim` |
| `get_x2_sonic_ready.sh` | `docker-compose.yml` + `docker-compose.real.yml` | SDK ethernet (`ROS_DOMAIN_ID=0`, CycloneDDS on the wired NIC) | `deploy_x2.sh local` |

The compose file bind-mounts the repo at `/workspace/sonic` and
`$X2_CHECKPOINTS_DIR` (default `gear_sonic_deploy/models/`, see `MODELS.md`)
read-only at `/workspace/checkpoints`; `deploy_x2.sh` additionally mounts
`$HOME` at the same path so host model paths resolve unchanged.

## Prerequisites

- docker with the compose plugin; the user in the `docker` group.
- For the viewer: an X session (`enter_sim.sh` runs `xhost +SI:localuser:root`).

## Build the image and the deploy package

```bash
cd gear_sonic_deploy/docker_x2
./enter_sim.sh                                   # builds the image (~10 min first time), drops into a shell
./enter_sim.sh -- bash /workspace/sonic/gear_sonic_deploy/docker_x2/build_deploy_pkg.sh   # one-shot colcon build
```

Expected: `docker compose build` finishes, then
`colcon build --packages-select agi_x2_deploy_onnx_ref ...` and
`BUILD_DEPLOY_PKG_DONE rc=0`. The build output lands in
`gear_sonic_deploy/{build,install,log}` on the host (bind mount).

## Bridge-only smoke (no policy)

```bash
cd gear_sonic_deploy/docker_x2
./enter_sim.sh
# inside the container:
cd /workspace/sonic/gear_sonic_deploy
python3 scripts/x2_mujoco_ros_bridge.py --print-scene
```

Second host shell:

```bash
cd gear_sonic_deploy/docker_x2
docker compose exec x2sim bash -c 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 topic list | grep aima'
```

Expected: joint state for `leg`, `waist`, `arm`, `head` plus
`/aima/hal/imu/torso/state` (200 Hz state, 500 Hz IMU).

## Closed loop with a policy

From the host (auto-relaunches inside the container):

```bash
./gear_sonic_deploy/deploy_x2.sh sim --model $X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
    --motion gear_sonic_deploy/data/motions_x2m2/demo_bank/<clip>.x2m2 --sim-viewer --autostart-after 5
./gear_sonic_deploy/deploy_x2.sh --help
```

`deploy_x2.sh sim` builds the package (`build_local()`) on every launch
unless `--no-build`, verifies the bridge + MJCF + deps, isolates DDS, starts
the bridge, then the node. Expected: `Loaded ONNX: ...`, the tuning flags
from `--tuning-config` (translated by `scripts/tuning_yaml_to_sim_flags.py`),
and the robot tracking the clip in the viewer. Stop: Ctrl-C (RAMP_OUT), then
`./gear_sonic/scripts/simstack_local.sh --stop` sweeps any leftover node.

## Real robot from the container (wired SDK link)

Host NIC on the SDK subnet, cable in a PC2/PC3 dev port, then:

```bash
cd gear_sonic_deploy/docker_x2
./get_x2_sonic_ready.sh -- ros2 topic list | grep aima      # discovery smoke
./get_x2_sonic_ready.sh                                      # shell
# inside: dry run (gains muted), never the first thing you do on a standing robot without a spotter
cd gear_sonic_deploy && ./deploy_x2.sh local --model /workspace/checkpoints/<set>/exported/<name>_g1.onnx \
    --motion data/motions_x2m2/demo_bank/<clip>.x2m2 --dry-run --autostart-after 5 --log-dir /tmp/x2_dryrun_$(date +%Y%m%d_%H%M%S)
```

The PC2-native path ([`F10_real_robot_colcon.md`](F10_real_robot_colcon.md))
is the one used for demos; `local` mode requires the wired link.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `x2sim image not found` on PC2 | expected: `onbot` mode runs natively on PC2, not in docker |
| no viewer | run from an X session; `xhost +SI:localuser:root`; `DISPLAY` set |
| two containers running | `docker ps --filter name=x2sim`; `simstack_local.sh --stop` |
| `aimdk_msgs` not found in a custom image | the Dockerfile builds it into `/ros2_ws`; source `/ros2_ws/install/setup.bash` |
| colcon picks up the G1 root package | always pass `--base-paths src/x2/agi_x2_deploy_onnx_ref` (the build script does) |
