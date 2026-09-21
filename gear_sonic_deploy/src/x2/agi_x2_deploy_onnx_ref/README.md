# `agi_x2_deploy_onnx_ref`

Reference real-robot deploy harness for the AgiBot X2 Ultra (31 DOF).
Loads a fused encoder+decoder ONNX policy exported from the SONIC training
pipeline, builds the IsaacLab-ordered observations from live AimDK ROS 2
state, and publishes PD-target joint commands at 500 Hz.

This package is the deploy-harness deliverable of the X2 ONNX deploy plan.
ONNX export + parity, codegen of `policy_parameters.hpp`
(`scripts/codegen_x2_policy_parameters.py`, plant tables re-spliced by
`scripts/gen_policy_parameters.py --plant vendor_20260823` from
`gear_sonic/config/robot_plant/x2_ultra_vendor_20260823.yaml`), and the
`sonic_common` shared lib must already be in place; the safety scaffolding
is built into this package and active by default.

## What it does on every control tick

```
[AimDK joint state x4 + IMU] ───► RobotState (MJ-ordered, mutex-guarded)
                                      │
                       50 Hz control timer
                                      ▼
       IL-remap ► ProprioceptionBuffer.Append (990 D, oldest-first)
                                      │
                       BuildTokenizerObs (680 D, ONNX grouped layout)
                                      │
                  OnnxActor.Infer ──► action_il[31]
                                      │
            scale & MJ-remap ► target_pos_mj[31]
                                      │
                       ApplySafetyStack
                       ├─ tilt watchdog (cosine threshold)
                       ├─ soft-start ramp (default ↔ policy)
                       └─ dry-run gain mute
                                      ▼
                                SafeCommand (latched)
                                      │
                       500 Hz writer timer
                                      ▼
[JointCommandArray /aima/hal/joint/{leg,waist,arm,head}/command]
```

## Build prerequisites

The deploy machine must have:

1. **ROS 2** (Humble or newer) with `ament_cmake`, `rclcpp`, `sensor_msgs`.
2. **`aimdk_msgs`** built from AgiBot's AimDK SDK `src/aimdk_msgs/` (stage it
   under `gear_sonic_deploy/thirdparty/aimdk_msgs/`, see the README there; on
   PC2 it is already installed under `/agibot/software/housekeeper/bin/aimdk_msgs`)
   and sourced into the colcon workspace.
3. **ONNX Runtime** ≥ 1.16 installed under `/opt/onnxruntime` (default) or
   passed via `-DONNXRUNTIME_ROOT=/path/to/onnxruntime`. The C++ headers
   (`onnxruntime_cxx_api.h`) and `libonnxruntime.so` must both be findable.

The package also pulls in `sonic_common` from `gear_sonic_deploy/src/common/`
via `add_subdirectory`; nothing extra is required to make that work.

## Build

```bash
# Inside a colcon workspace that already has aimdk_msgs available:
colcon build \
    --packages-select agi_x2_deploy_onnx_ref \
    --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
```

### Offline syntax / unit-test build

If you only want to verify the C++ compiles (no ROS 2 install required):

```bash
cmake -S . -B build -DAGI_X2_OFFLINE_SYNTAX_CHECK=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

This compiles only the obs builders + a stand-alone unit test that covers
quat math, proprioception priming/aging, and tokenizer sizing/layout.

## Run

```bash
ros2 run agi_x2_deploy_onnx_ref x2_deploy_onnx_ref \
    --model /path/to/model_step_016000_g1.onnx \
    --dry-run \
    --autostart-after 5 \
    --log-dir /tmp/x2_deploy_log
```

### CLI flags

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--model PATH` | (required) | Fused `g1+g1_dyn` ONNX exported from the SONIC training pipeline. |
| `--motion PATH` | StandStill | Optional X2M2 reference motion (see "Reference motion file format" below). |
| `--autostart-after SECONDS` | -1 (operator) | Auto-transition `WAIT → CONTROL` after N seconds. With `<0`, type `go` on stdin instead. |
| `--dry-run` | off | Publish stiffness=0 / damping=0. **Use this for the very first robot test.** |
| `--tilt-cos COS` | -0.3 | Tilt-watchdog threshold on body-frame gravity[2]. Matches `eval_x2_mujoco --fall-tilt-cos`. |
| `--ramp-seconds SEC` | 2.0 | Soft-start blend duration from default-pose to policy. |
| `--log-dir PATH` | (off) | Per-tick CSVs; one file per channel. |
| `--intra-op-threads N` | 1 | ONNX session threads. |

### State machine

```
INIT --(all 5 sources fresh)--> WAIT_FOR_CONTROL --(operator/autostart)--> CONTROL
                                                                              │
                                                                       (tilt watchdog)
                                                                              ▼
                                                                          SAFE_HOLD
```

- **INIT**: nothing is published. Waiting for the first message on every
  joint-group `/state` topic plus the torso IMU.
- **WAIT_FOR_CONTROL**: still publishing nothing; waiting for operator GO.
  This is the safe place to power on the motors and verify the joint-name
  validation log lines all printed.
- **CONTROL**: 50 Hz inference, 500 Hz command publish, soft-start ramp
  active for the first `--ramp-seconds`.
- **SAFE_HOLD**: tilt watchdog tripped (or fatal ONNX error). Latches
  "hold default angles, 4× damping" and publishes that forever. Restart the
  binary to recover.

## Reference motion file format (`X2M2`)

The C++ deploy reads a deliberately compact little-endian binary format so it
has no Python dependency at runtime:

```
uint32_t  magic       == 0x58324D32   ("X2M2")
uint32_t  num_frames
uint32_t  num_dofs    (must equal 31)
double    fps
For each frame f in [0, num_frames):
  double  joint_pos_mj[31]
  double  root_quat_xyzw[4]
```

Joint velocities are reconstructed at runtime via finite difference. To
convert a SONIC training PKL into this format, run (script TBD):

```bash
python gear_sonic_deploy/scripts/export_motion_for_deploy.py \
    --in  gear_sonic/data/motions/x2_ultra_standing_only.pkl \
    --out /opt/x2_motions/x2_ultra_standing_only.x2m2
```

If `--motion` is omitted, the deploy uses `StandStillReference`, which
returns the trained default standing pose at every future-frame query. This
is the recommended starting point for the first dry-run.

## Safety defaults

The Phase 4 safety stack is wired up out-of-the-box:

1. **`--dry-run`**: every published `JointCommand` has `stiffness=0,
   damping=0`. The motors do **nothing**. Use this to verify topic wiring
   and joint-name validation logs.
2. **Soft-start ramp**: the first `--ramp-seconds` of CONTROL state blend
   the policy target with the default standing pose, so the motors don't
   slam into the policy's first inference output.
3. **Tilt watchdog**: monitors body-frame gravity. If the torso tilts past
   `--tilt-cos` (~72.5° default) the deploy latches into SAFE_HOLD: target
   = default angles, 4× damping, no policy commands. Operator must restart.

## Where things live

```
include/
├── policy_parameters.hpp     # autogen Phase-1 header (joint maps, kp/kd, scales)
├── math_utils.hpp            # quat_rotate_inverse, rot6d_from_quat_xyzw, ...
├── reference_motion.hpp      # StandStillReference + PklMotionReference
├── proprioception_buffer.hpp # 990-D IsaacLab CircularBuffer port
├── tokenizer_obs.hpp         # 680-D ONNX builder (IL buggy-reshape layout)
├── onnx_actor.hpp            # ORT session wrapper
├── aimdk_io.hpp              # ROS 2 subscribers/publishers
├── safety.hpp                # SoftStartRamp + TiltWatchdog + ApplySafetyStack
└── deploy_logger.hpp         # per-tick CSV logger (X2-sized; not the G1 logger)
src/
├── (one .cpp per header)
└── x2_deploy_onnx_ref.cpp    # main(): CLI + state machine + 50/500 Hz timers
config/
└── obs_config_x2_ultra.yaml  # documents obs layout (1670 D total)
test/
└── test_obs_builder.cpp      # CPU-only unit tests
```
