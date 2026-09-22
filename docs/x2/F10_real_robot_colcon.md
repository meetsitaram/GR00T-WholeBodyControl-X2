# F10 — Real robot: PC2 bring-up, colcon build, push, ritual

The robot side runs on PC2 (Jetson Orin NX, aarch64) under one prefix
`${PC2_PREFIX}` (default `/home/run/gear-sonic`): ONNX Runtime, a Python
venv, the colcon overlay with `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref`
(the C++ deploy node) + `gear_sonic_deploy/src/common`, the planner stack,
policies, logs. Nothing in the vendor tree is touched.

Address conventions: `PC2_HOST` (`<PC2_IP>`; on the wired SDK link PC2 is
usually reachable as well), `PC2_USER=run`, `LAPTOP_HOST` (your address as
PC2 sees it). `./gear_sonic_deploy/scripts/x2_discover_network.sh` discovers
both and writes env files you can source.

## Prerequisites

- ssh access `run@<PC2_IP>` with key auth; PC2 needs internet once
  (ONNX Runtime tarball + the venv pip deps: `pyzmq onnxruntime joblib numpy
  scipy pyyaml pygame msgpack`); later re-runs skip pip when everything
  imports, `--force-pip` re-runs the install.
- The demo bank generated once on the laptop
  (`tools/build_demo_bank_from_upstream.sh` ->
  `gear_sonic_deploy/data/motions_x2m2/demo_bank/*.x2m2`); bring-up copies it
  to PC2, without it the pad clip banks are empty.
- The docker image for the laptop-side sim ([`F11_docker.md`](F11_docker.md))
  is optional here but is where you rehearse every change first.
- The shipped model set `gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*`
  (`git lfs pull` on a fresh clone), or your own set in `$X2_MODELS`
  ([`MODELS.md`](../../MODELS.md)).

## 1. Bring-up (idempotent)

```bash
./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host <PC2_IP> --dry-run       # preview
./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host <PC2_IP>                 # full install; stages the shipped set (step 9)
./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host <PC2_IP> \
    --model $X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx              # ... or one policy file of your own
./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host <PC2_IP> --skip-build    # refresh sources only
./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host <PC2_IP> --force-build   # colcon --cmake-clean-cache
```

Steps: ORT extract -> venv (`--system-site-packages`, pip installs `pyzmq
onnxruntime joblib 'numpy<2' scipy pyyaml pygame msgpack`; numpy is pinned below 2 because the Orin image's system scipy is built against numpy 1.x) -> rsync
`src/x2/agi_x2_deploy_onnx_ref`, `src/common`, the teleop zmq decoders ->
`colcon build` **on PC2** -> `gear_sonic_deploy/{deploy_x2.sh,scripts,configs}`,
the ritual scripts, `x2_pc2/robot_env.env` + `pad_bindings.env` -> (7b) the
`gear_sonic/utils/{teleop,pose_pipeline,planner}` python modules under both
`${PC2_PREFIX}/gear_sonic/` and `${PC2_PREFIX}/planner_stack/gear_sonic/`,
`estop_gesture.py` beside the pad bridge, the systemd units + installers under
`${PC2_PREFIX}/x2_pc2/` -> (8) bake `${PC2_PREFIX}/data/idle_stand.x2m2`
from the regenerated `x2_planner_primitives.pkl` (a failing bake aborts
bring-up; with no PKL the committed `gear_sonic_deploy/data/idle_stand.x2m2`
is staged instead) -> (8b) demo bank `.x2m2` clips ->
`${PC2_PREFIX}/planner_stack/models/dances_x2m2/`,
`kplanner_idle_anchor_g1teleop_v3.pkl` and `x2_planner_primitives.pkl` ->
`planner_stack/models/` (both from `tools/build_demo_bank_from_upstream.sh`;
missing = warning, the kplanner's lean/crouch/torso pad gestures are then
unavailable) -> rsync the model into `${PC2_PREFIX}/policies/`.
Expected tail: `build ok`, `idle_stand.x2m2`, `rsync demo bank (N clips)`,
`PC2 is ready`. The planner/pad **scripts** (`pc2_kplanner_onnx.py`,
`pad_locomotion_bridge.py`, `pc2_pad_daemon.py`, `x2_pad_autopair.py`) and the
models are NOT staged by bring-up: they go through the md5-gated push (§3).

Rebuild whenever `.cpp/.hpp` or the plant header changes
([`BUILD_CHAIN.md`](BUILD_CHAIN.md) C). The plant is compile-time; a model
exported against a different plant than the binary silently runs with the
wrong action scales.

Preflight (read-only, safe any time):

```bash
./gear_sonic_deploy/scripts/pc2_preflight.sh --pc2-host <PC2_IP>
```

## 2. Gamepad on PC2 (once)

After bring-up (§1) and the first push (§3), on PC2 as `run` (sudo):

```bash
bash ${PC2_PREFIX:-/home/run/gear-sonic}/gear_sonic_deploy/scripts/pc2_pad_setup.sh      # xpadneo, hidraw, bluez policy
bash ${PC2_PREFIX:-/home/run/gear-sonic}/x2_pc2/install_pad_autopair_service.sh          # boot daemon
```

The installer copies `x2_pc2/x2-pad-autopair.service` (staged by bring-up
step 7b) into `/etc/systemd/system/`, rewrites the prefix if you relocated the
install, `daemon-reload`s and `enable --now`s it. The unit runs
`x2_pad_autopair.py --start-cmd ${PC2_PREFIX}/ritual_start_demo.sh`, so the
pad chord fires the demo ritual (§5). Pair the controller once
(`bluetoothctl`: `scan on; pair <MAC>; trust <MAC>; connect <MAC>`). If you
use the head cameras: `bash ${PC2_PREFIX}/x2_pc2/install_camera_bridge_service.sh`
(`x2_pc2/x2-camera-bridge.service`). Both `.service` units hardcode
`/home/run/gear-sonic` (systemd does not expand `${PC2_PREFIX}`): the pad
installer `sed`s the prefix into the unit it installs; the camera installer
and its unit assume the default, so edit their `ExecStart=` /
`WorkingDirectory=` / log paths by hand if you relocated the install.

## 3. Push the planner stack, rituals and models

```bash
./x2_pc2/push_to_pc2.sh --manifest x2_pc2/push_manifest.txt --pc2 run@<PC2_IP> --dry-run
./x2_pc2/push_to_pc2.sh --manifest x2_pc2/push_manifest.txt --pc2 run@<PC2_IP>
```

Gates in order: `preflight_planner.py` PASS (needs the regenerated
`gear_sonic/data/motions/x2_planner_primitives.pkl` from
`tools/build_demo_bank_from_upstream.sh`, and `$X2_MODELS` + `PC2_PREFIX`
exported for the manifest placeholders) -> every manifest file committed
(untracked per-robot files such as `x2_pc2/robot_env.env` are md5-gated
instead) -> PC2 backup (`backups/<ts>/` + md5 manifest) -> md5 verify both
sides. Expected tail: `VERIFY OK (N files)`. The push never restarts anything.

`push_manifest.txt` also carries the `gear_sonic/utils/{teleop,pose_pipeline}`
modules the PC2 scripts import (same files bring-up step 7b stages), so an
incremental push keeps them current. Extra `.x2m2` clips for the pad banks go
to `${PC2_PREFIX}/planner_stack/models/dances_x2m2/` (bring-up step 8b copies
the demo bank; `run_motion_replay.sh` stages single clips) -- the keys in
`x2_pc2/pad_bindings.env` must match the files there.

Manifests: `push_manifest.txt` (everything), `push_manifest_kplanner_only.txt`,
`push_manifest_ritual_only.txt`, `push_manifest_incumbent.txt` (pose graph +
planner into the default slots), `push_manifest_wholebody_dual.txt` (dual set
+ token service). The SONIC model lines name the shipped
`gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*` files (tracked, so gate 2
applies; `git lfs pull` first); for your own set edit the **local** column to
your `$X2_MODELS` paths (commented examples in each manifest). The PC2 column
is fixed (`${PC2_PREFIX}/policies/...`,
`${PC2_PREFIX}/planner_stack/models/planner_onnx/...`); the planner graph line
stays `${X2_MODELS}` (not part of the shipped set).

What the robot actually runs (md5 + mtime of every file the rituals launch,
next to the laptop copy):

```bash
./x2_pc2/pc2_live_manifest.sh run@<PC2_IP>
```

## 4. The robot environment file

`x2_pc2/robot_env.env` is the single injection point for both the kplanner
ritual and the deploy ritual. Start from the shipped templates
(`x2_pc2/robot_env.env.template` for a native dual-head set,
`x2_pc2/robot_env.env.frozen.template` for a frozen-G1 token set), which
differ in exactly the three model pointers. **After `pc2_bringup.sh`, an
unedited copy of `robot_env.env.template` makes the ritual default to the
shipped set** (`X2_RITUAL_MODEL=${PC2_PREFIX}/policies/x2_sonic_v16ft8_45000_dual.onnx`,
`X2_RITUAL_TUNING=${PC2_PREFIX}/gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml`); nothing to edit
unless you bring another model:

```bash
ssh run@<PC2_IP> 'cd ${PC2_PREFIX:-/home/run/gear-sonic} && grep -nE "^: .*(X2_RITUAL_MODEL|X2_WB_TOKEN_MODEL|X2_WB_TOKENIZER)" robot_env.env'
```

Switching models = editing those three lines (as a pair/triple from one
checkpoint) and restarting the ritual; the model is loaded at launch.
Verify by the deploy's own `Loaded ONNX: ...` line, never by the pushed file.
Other knobs there: `X2_RITUAL_TUNING`, `X2_ANCHOR_ORI_MODE`, `X2_PLANT`,
`KPLANNER_*`, `X2_WHOLE_BODY_TELEOP` (default 1), `X2_WB_LAPTOP_HOST`.

## 5. Ignition (laptop-free)

`x2_pc2/ritual_start_demo.sh` (pad demo) / `x2_pc2/ritual_start_sonic.sh`
chain: watchdog first (must own :5558) -> `pc2_kplanner_onnx.py` (streams the
idle anchor, `--cmd-bind`) -> **pose-stream gate** (SUB :5558 must deliver
frames, 12 s) -> `x2_pc2/start_x2_deploy_ritual.sh` (deploy, preflight posture
check **before** MC stop, MC handover) -> pad bridge. The pad ritual
(hold L1+L2+R1+R2 3 s, then Y) fires it. `ENABLE_SCAN_GUARD=0` is the shipped
default in both ritual scripts: the lidar obstacle guard (the lumi and
scan-guard publisher scripts) is not shipped, and the kplanner's guard SUB (:5571) is
fail-open, so the operator deadman is the only stop. Watch on PC2:

```bash
ssh run@<PC2_IP> 'tail -f ${PC2_PREFIX:-/home/run/gear-sonic}/log/ritual_fired.log'   # GATE PASSED -> x2_deploy STARTED
ssh run@<PC2_IP> 'tmux ls'                                                          # x2_pose_watchdog, pc2_kplanner, x2_deploy, pad_bridge, ...
```

Deploy pane must show `Loaded ONNX: ...` and `[plant] RUNTIME PLANT LOADED: vendor_20260823`.

## 6. Laptop-driven daemons (alternative to the pad ritual)

```bash
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh print-env --pc2-host <PC2_IP>
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --pc2-host <PC2_IP> \
    --model ${PC2_PREFIX:-/home/run/gear-sonic}/policies/x2_sonic_v16ft8_45000_g1.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh status --pc2-host <PC2_IP>
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh logs deploy --pc2-host <PC2_IP>
```

`start` spawns tmux sessions `x2_pose_watchdog`, `x2_deploy`
(`deploy_x2.sh onbot`, blocks on the Y/n gate in its pane), `x2_hand_bridge`,
`x2_motor_monitor`. Then a laptop stack with `--pc2-host <PC2_IP>` (F01/F02)
publishes poses to the watchdog.

## 7. Stop

```bash
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host <PC2_IP>
```

SIGINTs the deploy; `deploy_x2.sh` runs RAMP_OUT -> HOLD_FOR_MC -> MC
restart. **Never `tmux kill-session` a live deploy on a standing robot**
(instant torque drop). On the pad, **A+X** + trigger pumps is the e-stop
(soft stop, then full damping).

## Deploy binary reference

`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/README.md` documents the
node: 50 Hz control (IL-remap -> 990-D proprio buffer -> 680-D tokenizer obs
-> ONNX -> 31 actions -> safety stack) and a 500 Hz writer on
`/aima/hal/joint/{leg,waist,arm,head}/command`. Observation layout:
`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/config/obs_config_x2_ultra.yaml`.
Tuning presets: `gear_sonic_deploy/configs/real_deploy_tuning/README.md`
(`_schema.yaml`, `trained_gains_s0_inc_waistmc.yaml` (shipped default), `trained_gains_s0.yaml`, `bigrun.yaml`, `frozen_g1.yaml`,
`walk_101.yaml`, `walking_recovery*.yaml`, `conservative.yaml`, `expressive.yaml`).

## Fall recovery: assisted stand-up out of SAFE_HOLD

Before 2026-09-22 a tilt trip was terminal: the deploy went to SAFE_HOLD
(pure damping once the down-detect fired) and only a restart brought the
policy back. The deploy now has a `RECOVER` state:

1. **Gate.** In SAFE_HOLD, the IMU must read upright (`gravity_body[z] <
   --recover-upright-cos`, default -0.95, about 18 deg) and the body must be
   quiet (base |angular velocity| < 0.5 rad/s, every joint |qd| < 1.0 rad/s)
   for `--recover-dwell-s` (2.0 s). That is what "someone is holding it up"
   looks like; the legs can be anywhere between a crouch and standing.
   Each arming and disarming of the gate is logged.
2. **Phase A, stiffen in place** (`--recover-stiffen-s`, 1.5 s): the gains
   ramp from whatever is latched (the pure-damping profile, or the stand-pose
   hold of a false trip) to the deploy gains while the target stays at the
   measured pose. No position step, no kp step.
3. **Phase B, rise** (`--recover-standup-s` 4.0 s minimum, stretched so the
   fastest joint moves at most `--recover-max-rate` 0.4 rad/s): the target
   blends with a half-cosine from the measured pose to the stand pose
   (`x2_stand_default_pose.yaml`, else the trained default angles).
4. **Hand-off.** CONTROL re-entered exactly as from WAIT_FOR_CONTROL: soft-start
   ramp from the measured pose, watchdogs reset, reference re-anchored.
5. **Abort.** Any tilt past `--tilt-cos` during the stand-up, or stale state
   for 0.5 s, drops straight back to SAFE_HOLD pure damping; the gate can
   re-arm. `--no-safe-hold-recover` disables the path. `x2_debug` carries an
   `in_recover` flag; `tick.csv` reasons are `recover_stiffen` / `recover_standup`.

**Helper procedure on the robot (first trials on the gantry).** Lift the robot
upright and hold it still with the feet on the ground and the knees bent.
After 2 s the joints stiffen (1.5 s), then the legs straighten slowly (4 s or
more). Keep supporting until it is standing and the policy is in control,
then let go. If it tilts while rising it goes limp again at once. There is no
audible cue yet (the deploy node has no TTS hook), so watch the log line
`SAFE_HOLD -> RECOVER` in the ritual output.

Sim rehearsal (verified 2026-09-22 with the shipped set): see
[`F09_mujoco_sim.md`](F09_mujoco_sim.md), "Rehearsing a fall and the recovery".

## Troubleshooting

| Symptom | Fix |
|---|---|
| `push_to_pc2.sh` aborts at COMMITTED | commit the manifest files first; the gate is deliberate |
| `No module named gear_sonic...` on PC2 | re-run `pc2_bringup.sh` (step 7b stages the teleop, pose-pipeline and planner utils under `${PC2_PREFIX}/gear_sonic/` and `${PC2_PREFIX}/planner_stack/gear_sonic/`; the rituals set `PYTHONPATH` to those) |
| "No module named" onnxruntime, joblib, pygame or msgpack in the venv | `pc2_bringup.sh --force-pip` re-runs the venv pip install (needs PC2 internet); the default run skips pip when every dep already imports |
| `invalid load key` loading the anchor pkl | the laptop copy is an LFS pointer: `git lfs pull`, re-run bring-up (step 8b) |
| pad clip banks do nothing / `clip key not found` | `${PC2_PREFIX}/planner_stack/models/dances_x2m2/` empty or keys differ from `pad_bindings.env`; generate the demo bank (`tools/build_demo_bank_from_upstream.sh`) and re-run bring-up |
| `Address already in use :5558` | the watchdog must start first; stop whatever bound it |
| deploy `pose_ref_age=-1` | normal on the zmq input path; the health signal is watchdog `state=LIVE` and the ignition gate |
| service stuck `deactivating` | `systemctl kill -s SIGKILL x2-pad-autopair` (pygame ignores SIGTERM) |
| sim launcher exit 6 with `--pc2-host` | identity gate mismatch (model / binary mtime / ritual drift); fix or `ALLOW_MISMATCH=1` for a candidate run |
