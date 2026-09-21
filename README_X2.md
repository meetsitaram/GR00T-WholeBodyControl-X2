# AgiBot X2 Ultra support

This branch adds the AgiBot X2 Ultra (31 DOF, OmniHand) as a second
embodiment of GEAR-SONIC: training and finetuning (native 3-encoder policies
and a frozen-G1-core + LoRA recipe), a MuJoCo sim stack that runs the real
C++ deploy binary in docker, a PC2 (Jetson) colcon deploy with gamepad
ignition, and three operator inputs — gamepad, Quest 3 VR and Pico
whole-body teleop — with kplanner <-> whole-body switching, motion replay and
Pico-tape replay.

**What ships, what is bring-your-own.** One native dual-head SONIC set
(`gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*`, git-lfs) is the default
of every launcher (`./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop`
needs no model variables), together with a first-party Pico "chores" motion
corpus (`gear_sonic/data/motions/x2_pico_chores/`) as the default for smoke
and training runs and the AgiBot MC stock gestures for the pad. Every other
policy and planner graph is bring-your-own: earlier X2 sets are on Hugging
Face (`tinkerbuggy/sonic-x2`), the public `nvidia/GEAR-SONIC` G1 release
(`download_from_hf.py`) is the frozen core for finetuning, and
[`MODELS.md`](MODELS.md) lists every model variable, its file contract and how
to export it. No third-party motion data and no robot address ship here.

## Start here

- [`docs/x2/README.md`](docs/x2/README.md) — index and a 10-minute sim quickstart.
- [`MODELS.md`](MODELS.md) — model variables (`MODEL`, `SIMSTACK_*`, `PLANNER_MODEL`, `X2_RITUAL_*`, `SONIC_HOME`, `CKPT_ROOT`).
- [`docs/x2/ARCHITECTURE.md`](docs/x2/ARCHITECTURE.md) — stack diagrams: pose chain, sim launchers, PC2 bring-up, build chain, training loop.
- [`docs/x2/BUILD_CHAIN.md`](docs/x2/BUILD_CHAIN.md) — what to rebuild when a model, clip, C++ file or script changes.

## Runbooks

| | Runbook |
|---|---|
| F01 | [Gamepad control](docs/x2/F01_gamepad.md) · [cheat sheet](docs/x2/gamepad_cheatsheet.md) |
| F02 | [Quest 3 VR control](docs/x2/F02_quest_vr.md) |
| F03 | [Pico whole-body teleop](docs/x2/F03_pico_teleop.md) |
| F04 | [Motion replay (pose encoder)](docs/x2/F04_motion_replay.md) |
| F05 | [Pico-tape replay (SMPL encoder)](docs/x2/F05_pico_tape_replay.md) |
| F06 | [Native 3-encoder models](docs/x2/F06_incumbent_models.md) |
| F07 | [Frozen G1-core models](docs/x2/F07_frozen_g1_model.md) |
| F08 | [kplanner <-> whole-body switching](docs/x2/F08_kplanner_wholebody_switching.md) |
| F09 | [MuJoCo sim stack](docs/x2/F09_mujoco_sim.md) |
| F10 | [Real robot: PC2, colcon](docs/x2/F10_real_robot_colcon.md) |
| F11 | [Docker test image](docs/x2/F11_docker.md) |
| F12 | [Training and finetuning on Nebius](docs/x2/F12_training_nebius.md) |

## Where the X2 code lives

Everything the port added is under a handful of directories. One line per
entry point; the helper modules next to them are described in
[`X2_FILES.txt`](X2_FILES.txt).

### Launchers (`gear_sonic/scripts/`)

| Script | What it starts |
|---|---|
| `sim_onnx_planner.sh` | the sim stack for one session: pad (default), `--vr`, `--vr-only`, `--whole-body-teleop` |
| `simstack_local.sh` | the persistent docker sim stack the launcher above builds on; `--stop` tears everything down |
| `run_x2_quest3_planner_stack.sh` | planner + Quest manager + pose merger (also the pad-mode backend) |
| `run_pico_teleop.sh` / `run_x2_pico_wbc.sh` | Pico whole-body teleop against the sim or the robot |
| `run_pico_replay.sh` | a recorded Pico tape through the live whole-body path |
| `run_motion_replay.sh` / `run_x2_pkl_direct_stack.sh` | a motion clip (pkl / x2m2) into the running stack |
| `run_planner_smoke.sh` | headless planner regression (also `push_to_pc2.sh` gate 1) |
| `simstack_pad_drive.py`, `simstack_turn.sh`, `simstack_replay_tape.py` | scripted pad input and intent-tape replay on the wire |

### Runtime processes (`gear_sonic/scripts/`)

| Process | Role |
|---|---|
| `pc2_kplanner_onnx.py` | the ONNX planner daemon: intents, crouch mode, whole-body switching, stop chain |
| `pad_locomotion_bridge.py`, `pc2_pad_daemon.py`, `x2_pad_autopair.py` | gamepad to `planner_cmd`, robot-side pad daemon, Bluetooth pairing |
| `quest3_manager_x2.py`, `x2_pose_merger.py` | Quest headset to locomotion + arm IK; pose merging and gesture playback |
| `live_pico_smpl_teleop.py`, `pico_intent_sender.py`, `pico_token_sender.py`, `pc2_pico_token_service.py` | Pico body stream to SMPL intent, and the tokenizer service the deploy consumes |
| `play_xbox_controller.py` | pad-driven clip bank player |
| `sim_mirror_viewer.py` | host-side MuJoCo viewer fed by deploy telemetry (read-only) |

### Deploy (`gear_sonic_deploy/`)

| Path | Role |
|---|---|
| `src/x2/agi_x2_deploy_onnx_ref/` | the C++ deploy node (ROS 2 / colcon), shared by sim and robot |
| `deploy_x2.sh` | launches the node in docker (`sim`) or on the robot |
| `docker_x2/` | sim image, compose files, deploy package builder |
| `configs/` | tuning presets (`real_deploy_tuning/`), MC stand pose, sim init poses |
| `models/` | the shipped default ONNX set (git-lfs) |
| `scripts/x2_mujoco_ros_bridge.py` | the MuJoCo plant |
| `scripts/pc2_bringup.sh`, `pc2_preflight.sh`, `x2_pc2_daemons.sh` | robot bring-up (rsync + pip + colcon), preflight, daemon set |
| `scripts/x2_pose_watchdog.py`, `x2_motor_monitor.py`, `x2_hand_zmq_to_aimdk_bridge.py`, `x2_debug_to_robot_pose_bridge.py` | robot-side daemons |
| `scripts/reexport_x2_onnx.sh`, `export_motion_for_deploy.py`, `bake_idle_stand_x2m2.py` | planner graph re-export, clip to x2m2 |

### Robot rituals (`x2_pc2/`)

`ritual_start_sonic.sh` / `ritual_start_demo.sh` (boot daemons), `start_x2_*.sh`
(per-daemon starters), `push_to_pc2.sh` + `push_manifest*.txt` (file push with
md5 gates), `pc2_live_manifest.sh` (what the robot actually launches),
`robot_env.env.template` / `.frozen.template`, `pad_bindings.env`, systemd
services, [`PORT_REGISTRY.md`](x2_pc2/PORT_REGISTRY.md).

### Models: export and gates (`gear_sonic/scripts/`)

| Lineage | Scripts |
|---|---|
| native 3-encoder (F06) | `export_native_all.sh` -> `native_dual_head_onnx_export.py`, `native_token_onnx_export.py` |
| frozen G1 core (F07) | `make_g1_lora_warmstart.py`, `frozen_core_t2_export.py` (fold LoRA), `frozen_core_t2_onnx_export.py`, `frozen_core_token_onnx_export.py`, `frozen_core_sonic_codec.py` |
| gates | `onnx_provenance.py` (pairing / fingerprint), `onnx_obs_contract.py`, `preflight_deploy_contract.py`, `check_plant_consistency.py` |
| evals | `eval_x2_mujoco.py`, `eval_x2_mujoco_onnx.py`, `play_x2_motion_mujoco.py`, `bringup_sim.py` |

### Training (F12)

| Path | Role |
|---|---|
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_*.yaml` | the five X2 experiment configs (`ultra`, `ultra_smoke`, `leg5_pico_combined_ft`, `frozen_core_t2`, `s1_fresh_vendor_waistmap`) |
| `gear_sonic/config/manager_env/`, `gear_sonic/config/robot_plant/` | X2 observation / reward / event terms; plant variants |
| `gear_sonic/trl/modules/` | `frozen_core_g1_actor.py`, `lora.py`, `g1_lora_module.py`, `onnx_helpers.py` |
| `gear_sonic/data_process/`, `gear_sonic/scripts/check_smpl_sidecars.py`, `x2_gmr_to_motion_lib.py` | corpus building from Pico tapes, SMPL sidecar gate, GMR retarget |
| `gear_sonic/scripts/cloud/` | Nebius provisioning, node bootstrap, preflights, 8-GPU smoke, big run, Pico finetune |
| `gear_sonic/scripts/run_local_finetune_demo_v2.sh` | single-GPU local finetune template |

### Libraries and data

| Path | Contents |
|---|---|
| `gear_sonic/envs/manager_env/robots/x2_ultra.py`, `gear_sonic/utils/embodiment/` | the X2 embodiment |
| `gear_sonic/utils/teleop/`, `gear_sonic/utils/teleop/vr/`, `gear_sonic/utils/pose_pipeline/` | SMPL observation, wrist map, retarget pipeline, tape format, WebXR app, wire codecs |
| `gear_sonic/utils/planner/`, `gear_sonic/config/kplanner_profiles/` | planner blending / constants; the two planner knob profiles |
| `motionbricks/scripts/`, `motionbricks/motionbricks/motion_backbone/inference/` | planner graph export and the G1-core heads |
| `gear_sonic/data/assets/robot_description/` | X2 MJCF, URDF, meshes, OmniHand |
| `gear_sonic/data/motions/` | gesture catalog, 51 MC stock gestures, the Pico chores corpus + SMPL sidecars + tapes, planner primitive recipes |
| `gear_sonic/data/audio/`, `gear_sonic/data/operator_calibrations/`, `gear_sonic/data/scripted_demos/` | cue sounds, VR calibration, scripted planner demos |
| `tools/` | `build_demo_bank_from_upstream.sh` (pad gesture bank), `check_docs.py`, `check_x2_ownership.py`, `git-hooks/` |

Ports: [`x2_pc2/PORT_REGISTRY.md`](x2_pc2/PORT_REGISTRY.md). Safety: every
launcher treats "no `--pc2-host`" as the local sim and an IP as the real
robot; the pad e-stop is always live on the robot.

## Which files are X2, which are upstream

`X2_FILES.txt` lists every tracked file the port added (A), modified (M) or removed (D)
against upstream NVIDIA commit `4141c34`, one purpose per line. Anything not listed is
untouched upstream. `python tools/check_x2_ownership.py` fails when a non-upstream file
is missing from the list, when an upstream file was changed without being listed, or when
an entry went stale; `--write` regenerates the list and leaves `TODO: describe` for new
files. Run it before a rebase onto a newer upstream and after adding files.
