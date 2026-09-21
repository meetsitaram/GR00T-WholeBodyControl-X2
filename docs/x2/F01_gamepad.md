# F01 — Gamepad control (sim + real)

The pad is the primary input on the robot and the default input in sim. It
drives the ONNX kplanner (sticks -> `planner_cmd` on :5563), fires clip banks
(`motion_clip_cmd` on :5568) and owns the e-stop gesture. Binding semantics
live in **one** file, `x2_pc2/pad_bindings.env`, sourced by both the robot
ritual and the sim stack. Full button map: [`gamepad_cheatsheet.md`](gamepad_cheatsheet.md).

## Prerequisites

- Quickstart steps 1–5 of [`README.md`](README.md) (venv, docker image,
  demo bank, `$X2_MODELS`).
- An Xbox-class or DualSense pad visible to `pygame` **before** launch
  (`python gear_sonic/scripts/pad_locomotion_bridge.py --probe` prints the
  axis/button map it sees).
- Models: a pose graph (`MODEL=`) and a kplanner directory (`PLANNER_MODEL=`),
  see [`MODELS.md`](../../MODELS.md).

## Sim

```bash
cd <repo>
ALLOW_MISMATCH=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh
```

`PLANNER=velocity` selects the velocity graph instead of the template graph.
`KPLANNER_FIXED_FWD_MPS=0.5 KPLANNER_FIXED_TURN_RAD_S=1.0` override the
profile's locked speed / turn rate (the profile uses the `:=` idiom, so an
exported env var wins).

Expected output:

- `[profile] gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env (md5 ...)`
- `=== PC2 IDENTITY GATE: SKIPPED (sim-only, no --pc2-host) ===`
- the sim stack banner (`FULL-LOOP SIM -- WORKSTATION`) with sonic/planner/mjcf md5s
- kplanner log `onnx backend ready`, watchdog `state=LIVE`, deploy `Loaded ONNX: .../x2_sonic_39000_g1.onnx`
- a MuJoCo window with the X2 standing; hold **L2** and push the left stick — it walks.

Stop: `./gear_sonic/scripts/simstack_local.sh --stop` (or Ctrl-C in the
launcher terminal, then the stop command to sweep the docker deploy node).

### Clip banks in sim

With the deadman released, `L1+Y` / `L1+X` / `L1+B` / `L1+A` cycle the banks
defined in `x2_pc2/pad_bindings.env` (`EASY_DANCES`, `COMBAT`, `MEDIUM`,
`GESTURES`); the keys must exist as `.x2m2` files under
`$CKPT_ROOT/dances_x2m2/` (the demo bank from
`tools/build_demo_bank_from_upstream.sh`, copied or symlinked there).
`L1+R1` stops a clip. Chord dispatch is `gear_sonic/scripts/play_xbox_controller.py`;
stick locomotion is `gear_sonic/scripts/pad_locomotion_bridge.py`.

### Crouch mode (L2+R2 + D-pad DOWN)

The chord latches `crouch_mode` in the kplanner; while it is on, forward and
turn targets are planned at hip 0.55 m (`KPLANNER_CROUCH_HIP_M`), standing
still keeps the standing hip, and releasing either trigger stands up. What
that produces depends on the planner graph:

| `PLANNER_MODEL` | crouch walk |
|---|---|
| G1-core planner (`tinkerbuggy/sonic-x2/kplanner_g1core/`, `g1core_bare_v1.env`) | hip < 0.60 routes to the core's **crouch template**: a deep squat-walk, ~100 deg reference knees |
| incumbent template planner (`tinkerbuggy/sonic-x2/kplanner_onnx/`, the default) | **no crouch template**: the hip is ignored, the reference is a normal walk (measured 2026-09-20: reference knees 5-22 deg either way) |

Status: **robot-verified with the frozen core** (2026-09-09, `frozen_g1_s1ft16000`
+ the G1-core planner (HF `tinkerbuggy/sonic-x2/kplanner_g1core/x2_planner_template_s1d_a05_ws.onnx`), six crouched walks at hip
0.55: measured knees 52-64 deg, IMU pitch 11-20 deg, every stop from the crouch
clean). The served depth varies run to run because the template sampler draws
a fresh segment phase per replan (served pelvis 0.52-0.59 m; one deep draw
gave 9 quick steps in 5.5 s); `KPLANNER_CROUCH_SEED=<0-15>` pins the phase
while the mode is on. The sim twin is **not** the qualifying test for crouch
walk: it cannot hold the deep crouches the robot holds (whole-body release
report, 2026-09-04), and at real time every policy (`v16ft8_45000`,
`s1ft16000`, `s1p4_35000`) tips at the crouch stop in sim, seed pinned or
not. Qualify on the gantry or hands-on, shallow first, as the report did.
Opt-in knob: `KPLANNER_CROUCH_STAND_RATE_RAD_S=0.5` rate-limits the stand-up
at the crouch exit (default off = the walking stop blend the robot was
verified with).

Reproduce without touching the pad (planner must own the command socket:
`sim_onnx_planner.sh --vr` or `simstack_local.sh`):

```bash
python gear_sonic/scripts/simstack_pad_drive.py walk 6 0.6          # deadman + stick forward
python gear_sonic/scripts/simstack_pad_drive.py crouchwalk 8 0.6    # chord, crouched walk, release
```

The entry gate reads the deploy's `x2_debug` joints; the kplanner now
subscribes to that feed whenever `--x2-debug-port` is set (it used to start
the subscriber only with the yaw rebase on, so every sim pad launch refused
the chord with "no measured joints").

## Real robot (laptop-free, pad on PC2)

One-time enablement on PC2 (xpadneo driver, hidraw permissions, bluez
policy), run **on PC2** as the `run` user:

```bash
bash ${PC2_PREFIX:-/home/run/gear-sonic}/gear_sonic_deploy/scripts/pc2_pad_setup.sh      # driver + permissions
bash ${PC2_PREFIX:-/home/run/gear-sonic}/x2_pc2/install_pad_autopair_service.sh          # boot ritual daemon
```

Both files are staged by `pc2_bringup.sh`, which also copies the demo bank
(`gear_sonic_deploy/data/motions_x2m2/demo_bank/*.x2m2`, generated by
`tools/build_demo_bank_from_upstream.sh`) to
`${PC2_PREFIX}/planner_stack/models/dances_x2m2/` and the kplanner idle anchor
to `${PC2_PREFIX}/planner_stack/models/`; the clip keys in `x2_pc2/pad_bindings.env` must
match the files in that directory.

The boot chain is `x2_pc2/x2-pad-autopair.service` ->
`gear_sonic/scripts/x2_pad_autopair.py` (pairs/reconnects the pad, spawns
`gear_sonic/scripts/pc2_pad_daemon.py`) -> the ignition ritual
(`x2_pc2/ritual_start_demo.sh`) -> watchdog -> `pc2_kplanner_onnx.py` ->
pose-stream gate -> deploy -> pad bridge. Install everything with
[`F10_real_robot_colcon.md`](F10_real_robot_colcon.md), then:

1. Robot standing under the vendor MC, operator beside it, pad blinking.
2. Hold **L1+L2+R1+R2** ~3 s (rumble countdown), release, press **Y**.
3. Watch the ritual log on PC2 (`${PC2_PREFIX}/log/`): `GATE PASSED` ->
   `x2_deploy STARTED`, then the deploy pane prints `Loaded ONNX: ...` and
   `[plant] RUNTIME PLANT LOADED: ...`.
4. Hold **L2** + left stick to drive; **L1+Y/X/B/A** clips; **A+X** then
   pump both triggers = e-stop.

Stop (robot): never kill the deploy tmux session on a standing robot. From
the laptop:

```bash
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host <PC2_IP>   # RAMP_OUT -> MC handover
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| launcher exits with `--pad-only tears the stack down` / no joystick | plug the pad before launch; `pad_locomotion_bridge.py --probe` must list it |
| sticks reversed | `--invert-ly` is on by default; adjust in `pad_bindings.env` or the bridge flags, not in code |
| a bank chord does nothing | the key is missing from `$CKPT_ROOT/dances_x2m2/`; check `pad_bindings.env` against the demo bank listing |
| pad pairs but no input on PC2 | wrong HID driver claimed it; `pc2_pad_setup.sh` installs the xpadneo udev rebind; restart `x2-pad-autopair` |
| `Address already in use :5558` on PC2 | something else bound the watchdog port first; stop the stray process, the ritual must start the watchdog first |
| planner ignores the deadman | `PAD_DEADMAN` in `pad_bindings.env` (`left` = L2 only, `both` = L2+R2) |
