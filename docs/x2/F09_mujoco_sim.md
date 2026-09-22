# F09 — MuJoCo sim stack

Two launchers, one stack:

- `gear_sonic/scripts/simstack_local.sh` — the full loop on the workstation:
  ONNX kplanner -> pose watchdog -> the **real C++ deploy binary** in the
  `x2sim` container (`gear_sonic_deploy/docker_x2/`) -> `x2_mujoco_ros_bridge.py` (MuJoCo physics,
  `/aima/*` topics on loopback DDS, `ROS_DOMAIN_ID=73`). Isolated ports
  (5656/5658/5659/5663/5668). Sources `x2_pc2/robot_env.env`, the same file
  the robot ritual sources, so sim and robot agree on every knob.
- `gear_sonic/scripts/sim_onnx_planner.sh` — the operator front door: profile
  + model resolution + the robot identity gate, then `simstack_local.sh`.

Assets: MJCF `gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml`
(meshes via git-lfs), URDFs under `gear_sonic/data/assets/robot_description/urdf/x2_ultra/`,
plant yamls `gear_sonic/config/robot_plant/*.yaml`, sim init poses
`gear_sonic_deploy/configs/sim_init_poses.yaml`, stand pose
`gear_sonic_deploy/configs/x2_stand_default_pose.yaml`. The Python plant
(`gear_sonic/envs/manager_env/robots/x2_ultra.py`, `plant_config.py`) is the
training-side twin.

## Prerequisites

- Quickstart of [`README.md`](README.md) incl. the docker image
  ([`F11_docker.md`](F11_docker.md)) and `git lfs pull` (41 mesh refs).
- `CKPT_ROOT` under `$HOME` (mounted at `/workspace/checkpoints`).
- Demo motion bank regenerated: `bash tools/build_demo_bank_from_upstream.sh`
  (quickstart step 4; gesture bank by default, upstream clips opt-in). It writes `gear_sonic/data/motions/x2_demo_bank.pkl`,
  `x2_planner_primitives.pkl`, `x2_pad_banks.pkl` and
  `gear_sonic_deploy/data/motions_x2m2/demo_bank/*.x2m2`, which the launchers,
  the evaluations and `deploy_x2.sh --motion` read.
- The plant consistency gate must pass (it runs at every launch):

```bash
.venv/bin/python gear_sonic/scripts/check_plant_consistency.py
```

## Launch

```bash
# front door, shipped set + shipped tuning preset (whole-body rehearsal, dual graph)
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
# front door, shipped set (pad / planner only, the _g1 graph)
./gear_sonic/scripts/sim_onnx_planner.sh
# ... --dry-run on either prints every resolved path (models, tuning, banks) and exits

# front door with your own set (pad, template graph)
ALLOW_MISMATCH=1 \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh

# the stack directly (no gate), e.g. a whole-body rehearsal with your own dual graph
X2_WHOLE_BODY_TELEOP=1 SIMSTACK_OMNIHAND=1 \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_dual.onnx \
SIMSTACK_SONIC=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
SIMSTACK_DUAL_MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_dual.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
X2_RITUAL_TUNING=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml \
./gear_sonic/scripts/simstack_local.sh
```

Knobs read by `simstack_local.sh`: `SIMSTACK_SONIC`, `SIMSTACK_DUAL_MODEL`,
`SIMSTACK_TOKEN`, `SIMSTACK_TOKENIZER` (all default to the shipped
`gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*` set), `SIMSTACK_MJCF` (sim-only MJCF
override), `SIMSTACK_OMNIHAND=1` (compose the OmniHand:
`gear_sonic/scripts/compose_x2_with_omnihand.py`), `X2_RITUAL_TUNING` /
`SIM_TUNING_YAML` (default `trained_gains_s0_inc_waistmc.yaml`), `X2_PLANT`, `X2_ANCHOR_ORI_MODE`,
`KPLANNER_PYTHON` (the Isaac Lab python for the torch planner path),
`SIM_DEPLOY_ARGS` (extra `deploy_x2.sh` flags, e.g. `--sim-mjcf <xml>`).

Expected output: the `FULL-LOOP SIM -- WORKSTATION` banner with md5s of the
sonic, planner and MJCF files, `[1/4] .. [4/4]` step lines, the container
building (first time) then `Loaded ONNX`, the bridge publishing joint state
at 200 Hz / IMU at 500 Hz, watchdog `state=LIVE`, and the MuJoCo viewer.

The viewer is the host-side `gear_sonic/scripts/sim_mirror_viewer.py`
(read-only, fed by the deploy's telemetry). The bridge's own in-loop viewer
(`SIM_VIEWER_MODE=container`, `deploy_x2.sh --sim-viewer`) renders inside the
physics loop and halves the real-time factor (`[bridge] RTF 0.51`), which
plays every reference twice as fast into the physics and hides instabilities;
watch for `RTF 1.00` in the deploy log. `SIM_VIEWER=0` runs headless.

Stop (always, even after Ctrl-C):

```bash
./gear_sonic/scripts/simstack_local.sh --stop
```

It stops the kplanner, watchdog, bridge, token service, pad bridge, the
`x2sim` container and sweeps the deploy node. A relaunch over a live stack
runs this first automatically (two deploys must never drive one sim).

## Evaluations without the stack

```bash
python gear_sonic/scripts/play_x2_motion_mujoco.py --motion gear_sonic/data/motions/x2_demo_bank.pkl        # kinematic only
python gear_sonic/scripts/eval_x2_mujoco.py --checkpoint <ckpt.pt> --motion <clip.pkl>                       # torch actor, PD loop
python gear_sonic/scripts/eval_x2_mujoco_onnx.py --onnx <name>_g1.onnx --motion <clip.pkl>                  # ONNX actor, same loop
python gear_sonic/scripts/eval_x2_mujoco.py --checkpoint <ckpt.pt> --motions gear_sonic/data/motions/x2_demo_bank.pkl        # every bank clip, one by one
python gear_sonic/scripts/bringup_sim.py --help                                                              # assisted bring-up battery
```

`eval_x2_mujoco.py` knobs: `--freeze-wrist` / `--wrist-ref`, `--kp-scale`,
`--kd-scale`, `--assist-up`, `ACTION_CLIP=20` (env). The G1 side of the same
tools is `gear_sonic/scripts/play_motion_mujoco.py`.

Shipping gate for a new checkpoint ("battery"): run `eval_x2_mujoco.py
--motions gear_sonic/data/motions/x2_demo_bank.pkl` (every MC gesture, one by
one) and, for robustness, `--push-ladder`; a model must survive the MuJoCo battery before it
goes near the robot. Isaac Lab numbers alone never qualify a model.

`deploy_x2.sh sim` can also run alone (motion-file input, no kplanner):

```bash
./gear_sonic_deploy/deploy_x2.sh sim --model $X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
    --motion gear_sonic_deploy/data/motions_x2m2/demo_bank/<clip>.x2m2 --sim-viewer --autostart-after 5
```

## Rehearsing a fall and the recovery

The bridge can shove the robot over (`--tilt-torque`, a hand-tilt probe on the
torso) and its elastic band can play the helper that holds it up. Launch with
the band available but released and one shove scheduled at sim t = 30 s:

```bash
SIMSTACK_SIM_FLAGS="--sim-init-pose stand-settled --sim-band-release-after-s 0" \
SIM_BRIDGE_EXTRA="--tilt-torque 200 --tilt-start 30 --tilt-ramp 1.5 --tilt-hold 3 --tilt-release 0.1 --tilt-rest 100000" \
./gear_sonic/scripts/simstack_local.sh
# after "tilt watchdog tripped ... -> SAFE_HOLD" and the down-detect in deploy.log:
python gear_sonic/scripts/simstack_hold_up.py on -0.45     # hold the pelvis at 0.55 m, knees bent
# deploy.log: "SAFE_HOLD -> RECOVER", then "RECOVER complete ... -> CONTROL"
python gear_sonic/scripts/simstack_hold_up.py on -0.33     # ease up to stand height
python gear_sonic/scripts/simstack_hold_up.py off          # let go; the policy stands on its own
```

The band's orientation PD spins a fallen body upright before it lifts it;
a person would not do that, so ignore the spin. Verified 2026-09-22 on the
shipped set: gate armed once upright and quiet, stiffen 1.5 s, stand-up
8.7 s (one joint had 3.5 rad to travel), CONTROL re-entered, no trip after
release, and a scripted walk ran cleanly afterwards.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MISSING: <path>` at launch | one of MJCF / sonic / planner / tuning / python does not exist; check `MODEL`, `PLANNER_MODEL`, `KPLANNER_PYTHON` |
| `REFUSING LAUNCH: plant copies disagree` | regenerate the MJCF/header from the plant yaml ([`BUILD_CHAIN.md`](BUILD_CHAIN.md) C) |
| no viewer window | `xhost +SI:localuser:root` (done by `enter_sim.sh`), `DISPLAY` set; headless boxes cannot use `--sim-viewer` |
| second launch behaves oddly | a stale container: `simstack_local.sh --stop`, then `docker ps` must show no `x2sim` |
| model resolves to the wrong file | with no `MODEL=` the shipped set wins (sim-only) or the md5 match against the robot (`--pc2-host`) over candidates under `gear_sonic_deploy/models/`, `$CKPT_ROOT` and `$SONIC_HOME/x2`; pass `MODEL=` explicitly |
| shipped model files are ~130 bytes / `LFS pointer` in `--dry-run` | `git lfs install && git lfs pull` |
| `FileNotFoundError` in `tuning_yaml_to_sim_flags` | give an absolute or repo-relative `SIM_TUNING_YAML`; the launcher canonicalises it |
