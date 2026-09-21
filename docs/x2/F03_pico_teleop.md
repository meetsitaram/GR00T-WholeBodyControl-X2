# F03 — Pico whole-body teleop

Two architectures ship, both driven by the same headset:

1. **Deploy-faithful ("Quest split")** — `run_pico_teleop.sh` on the laptop
   streams human *intent* only (SMPL joints, engage flag, wrist targets) on
   :5573 to the deploy side, where the SMPL tokenizer runs
   (`pc2_pico_token_service.py` on PC2, or the dual-head deploy directly).
   Same command for sim and robot; the target is chosen by `--pc2-host`.
2. **Standalone sim (SMPL path)** — `run_x2_pico_wbc.sh` drives a MuJoCo sim
   in-process through the SMPL encoder. **SIM-ONLY**: no deploy safety stack;
   never point it at the robot. Used for model feel-tests and tape gates.

Controller grammar (both): **A+B+X+Y** held ~0.4 s = OFF <-> LOCOMOTION,
**B** = LOCOMOTION <-> WHOLE_BODY (engages on entry), sticks = kplanner walk /
turn with **LT held as deadman**, **X/Y** = speed setpoint -/+ 0.1 m/s, **A+X**
held + pumping both triggers = e-stop. Grips drive the OmniHand fingers.
`gear_sonic/scripts/pico_button_probe.py` prints raw button state if a
mapping looks wrong.

## Prerequisites

- Quickstart steps 1–5 of [`README.md`](README.md).
- `bash install_scripts/install_pico.sh` — builds `.venv_teleop`
  (`xrobotoolkit_sdk`, `isaacteleop`) and installs the XRoboToolkit PC
  Service (`SKIP_PC_SERVICE=1` to skip the deb). Never start the service
  from its desktop icon; the launchers start the bare daemon themselves
  (`pgrep -x RoboticsService` shows it).
- Headset (once): same Wi-Fi as the PC (`ping <headset-ip>` < 100 ms while
  worn), developer mode on, sideload the XRoboToolkit PICO client APK
  (`python3 -m http.server 8000` on the PC, open the APK URL in the headset
  browser). Motion Tracker app: set **max tracker usage = 3** *before*
  pairing waist + both ankles, select **Full body** mode, then **Calibrate**
  (a headset reboot wipes calibration — recalibrate after every reboot).
- Every session: open XRoboToolkit on the headset, connect to the PC IP,
  **send data ON**, enter the VR scene and keep the app foregrounded. Probe:

```bash
.venv_teleop/bin/python -c "import xrobotoolkit_sdk as xrt, time; xrt.init(); time.sleep(2); print('body:', xrt.is_body_data_available(), 'stamp:', xrt.get_time_stamp_ns(), 'trackers:', xrt.num_motion_data_available())"
# want: body: True, an advancing stamp, trackers: 2 (or 3)
```

- Models: the shipped native dual-head set (`gear_sonic_deploy/models/`, no
  variables needed) **or** your own dual-head / frozen-G1 token set
  ([`MODELS.md`](../../MODELS.md)).

## Sim — deploy-faithful whole-body teleop (acceptance commands)

Terminal 1, the stack. Shipped set + shipped tuning preset (canonical):

```bash
cd <repo> && ./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

Your own native dual-head set (explicit-env variant):

```bash
cd <repo> && TOKEN_SVC_OPERATOR_ROOT_LEVEL=zero ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
SIMSTACK_DUAL_MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_dual.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

Frozen-G1 token set (pose graph + token graph + release tokenizer):

```bash
cd <repo> && TOKEN_SVC_OPERATOR_ROOT_LEVEL=zero ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=$X2_MODELS/s1p4_35000/exported/x2_sonic_s1p4_35000_g1.onnx \
SIMSTACK_TOKEN=$X2_MODELS/s1p4_35000/exported/x2_sonic_s1p4_35000_g1_token.onnx \
SIMSTACK_TOKENIZER=$X2_MODELS/armA_14900/exported/x2_smpl_tokenizer_v11release.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

Terminal 2, the headset half (no `--pc2-host` = this sim stack):

```bash
./gear_sonic/scripts/run_pico_teleop.sh
./gear_sonic/scripts/run_pico_teleop.sh --cam sim --headset-ip <HEADSET_IP> --cam-preview   # + sim ego view in the headset
```

Expected output:

- Terminal 1: `[whole-body] NATIVE DUAL-HEAD graph — one SONIC session, both reference heads`
  (or `[whole-body] TOKEN GRAPH — deploy-faithful rehearsal via simstack_local.sh`),
  the `FULL-LOOP SIM` banner, deploy `Loaded ONNX ... kind=native_dual_head`
  (or the token service's `tokenizer ...` line), and the deploy binding :5573.
- Terminal 2: the sender waits for :5573, then prints rate lines with
  `body OK`; every mode flip, engage / `ENGAGE REFUSED ... RECALIBRATE in headset`,
  and deadman edge is logged.
- MuJoCo: A+B+X+Y -> LOCOMOTION (sticks walk), B -> WHOLE_BODY: the robot
  follows your torso, arms and legs; B again releases (hold -> rate-limited
  return to the kplanner).

Stop: Ctrl-C the sender, then `./gear_sonic/scripts/simstack_local.sh --stop`.

Knobs: `KPLANNER_WB_ARM_LATCH_GAIT=keep` (default) keeps the latched arm pose
through a gait; `WRIST_SMPL_MAP=lp:+y,lr:+z,rp:-y,rr:+z` flips a wrist sign;
`--head-source off` disables head-yaw follow; `HEAD_YAW_SIGN=-1` flips it.

## Sim — standalone SMPL path (feel-test, no docker)

```bash
# fused SMPL ONNX (obs 1830) from a native set:
./gear_sonic/scripts/run_x2_pico_wbc.sh --onnx $X2_MODELS/pico_39000/exported/x2_sonic_39000_smpl.onnx
# torch checkpoint (native 3-encoder .pt), or a frozen-G1 composite:
./gear_sonic/scripts/run_x2_pico_wbc.sh --model $X2_MODELS/pico_39000/model_step_039000.pt
./gear_sonic/scripts/run_x2_pico_wbc.sh --model "frozen-core-smpl:$X2_MODELS/armA_14900/armA_14900_merged.pt"
# the public G1 core on the G1 body (pair with --robot g1):
./gear_sonic/scripts/run_x2_pico_wbc.sh --robot g1 --model "smpl-g1:$SONIC_HOME/g1/sonic_v1_1/last.pt"
```

`--model` + `--onnx` together = torch drives, ONNX shadows, per-step diff
CSV. Controls: both grips 0.5 s = engage / disengage, SPACE in the viewer =
same, R = reset. Stop: Ctrl-C.

## Robot

Ritual up ([`F10_real_robot_colcon.md`](F10_real_robot_colcon.md)) with
`X2_WHOLE_BODY_TELEOP=1` in `x2_pc2/robot_env.env` (the default; the overlay
drives only while the laptop stream is fresh and the operator is engaged,
otherwise the session is a plain pad session). Then, on the laptop:

```bash
./gear_sonic/scripts/run_pico_teleop.sh --pc2-host <PC2_IP>
./gear_sonic/scripts/run_pico_teleop.sh --pc2-host <PC2_IP> --cam orbbec --cam-preview   # + robot camera in the headset
```

Health check before engaging (want intent 50 Hz + robot-state 50 Hz, no `[NO ...]` flags):

```bash
ssh run@<PC2_IP> 'tail -2 ${PC2_PREFIX:-/home/run/gear-sonic}/log/pico_token_service.log'
```

`--cam <orbbec|stereo_left|stereo_right|head_front>` needs the PC2 camera
bridge (`gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py --streams <key>`,
installed as a service by `x2_pc2/install_camera_bridge_service.sh`); on the
headset open Camera -> **Listen** when the script says so. Stop: Ctrl-C the
sender (the robot drops to the kplanner / pad within one tick); the pad
e-stop is always live.

## Recording a session

During a live `run_pico_teleop.sh` session hold **X** >= 0.8 s to start and
**Y** >= 0.8 s to stop; the sender writes `record_<stamp>.npz` into its
`--record-dir`, plus one `session_<stamp>_x2_segNN.npz` per whole-body engage
window. Replay them with [`F05_pico_tape_replay.md`](F05_pico_tape_replay.md).
Record for robustness: stick pushes >= 3 s, a pause between a turn and a
push, a real stop before engaging whole-body.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `--input-source pico needs .venv_teleop` | `bash install_scripts/install_pico.sh` |
| `xrt.init()` connects but everything is zero | headset asleep / app backgrounded / Full-body mode off / other Wi-Fi |
| `body: False` forever with trackers connected | calibration lost (button reads "Calibrate", not "Recalibrate"): recalibrate |
| only 2 of 3 trackers pair | Motion Tracker "max tracker usage" defaults to 2 — set 3 first |
| `ENGAGE REFUSED` | body solve frozen/dead: recalibrate, keep the app foregrounded |
| sender refuses: another sender targets this host | `run_pico_teleop.sh` and `run_pico_replay.sh` are exclusive; Ctrl-C the other one |
| `RoboticsServiceProcess: error while loading shared libraries` | do not start it by hand; the launcher sets `LD_LIBRARY_PATH` |
| `MODEL=` is a g1 planner graph on the standalone path | the SMPL path needs `*smpl*.onnx`, a `.pt`, or an `frozen-core-smpl:` composite |
