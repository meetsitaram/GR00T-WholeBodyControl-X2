# F02 — Quest 3 VR control

The Quest 3 drives the kplanner (locomotion, sticks) and the arm/hand IK
through a WebXR page served by `gear_sonic/scripts/quest3_manager_x2.py`
(HTTPS :8443, WebSocket :8765). The pad and the headset can coexist: the pad
publishes only while its deadman is held, the manager only while engaged.

## Prerequisites

- Quickstart steps 1–5 of [`README.md`](README.md).
- Teleop venv: `bash install_scripts/install_quest3.sh` (WebXR deps, no
  XRoboToolkit SDK).
- Headset on the same Wi-Fi as the laptop; firewall open for 8443/tcp and
  8765/tcp (`sudo ufw allow 8443/tcp 8765/tcp`).
- One-time Guardian setup on the headset (Settings -> Physical Space -> Set
  Floor + Create Boundary), otherwise `Start VR` fails with a reference-space error.
- Models: `MODEL=` pose graph + `PLANNER_MODEL=` kplanner dir ([`MODELS.md`](../../MODELS.md)).
- Demo motion bank regenerated: `bash tools/build_demo_bank_from_upstream.sh` (quickstart
  step 4; the planner stack loads `gear_sonic/data/motions/x2_planner_primitives.pkl` and
  `gear_sonic/data/motions/x2_pad_banks.pkl` from it).

## Sim, headset only (no pad)

```bash
cd <repo>
ALLOW_MISMATCH=1 \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --vr-only
```

## Sim, pad + headset (dual source)

```bash
cd <repo>
ALLOW_MISMATCH=1 \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --vr
```

`--vr` runs the stack in `--pad-and-vr` mode: the kplanner's `planner_cmd`
SUB binds :5563 (`pc2_kplanner_onnx.py --cmd-bind`), the pad bridge and the
manager both PUB-connect (`quest3_manager_x2.py --planner-cmd-connect`).

Headset side, in order:

1. Meta Quest Browser -> `https://<laptop-ip>:8443`, accept the self-signed
   certificate; also open `https://<laptop-ip>:8765` once and accept that
   certificate; back to :8443.
2. `Connect WS` (status turns green) -> `Start VR`.
3. **A+B+X+Y** = OFF <-> LOCOMOTION (sticks walk / turn). **B** = LOCOMOTION
   <-> ARM_MANIPULATION, **A** engages arm IK: hands drive the arms, triggers /
   grips close the fingers. Chord again = disengage.

Expected output: manager log `planner_cmd PUB connected`, kplanner log
`SUB bind` on :5563 (with `--vr`), pad log `PUB connect`; in the MuJoCo
window the robot walks on stick input and the arms follow the controllers in
ARM_MANIPULATION.

Stop: `./gear_sonic/scripts/simstack_local.sh --stop`.

## Operator calibration (recommended once per operator)

```bash
python -m gear_sonic.scripts.vr_operator_calibrate \
    --output data/operator_calibrations/<id>.yaml --operator-id <id>
```

The script boots the Quest server, walks through arms-down / T-pose /
arms-forward (press **A** at each pose) and writes a per-arm affine map.
Pass it to the stack with `--calibration data/operator_calibrations/<id>.yaml`
(`run_x2_quest3_planner_stack.sh` flag).

Without a headset (pad-only sim runs) no calibration is needed: when
`data/operator_calibrations/default.yaml` is absent the stack falls back to the
shipped neutral reference `gear_sonic/data/operator_calibrations/default.yaml`
and says so in a warning.

## Standalone manager (no sim, wire debugging)

The X2 Quest manager can run on its own to check the headset link and the
`planner_cmd` / `arm_targets` wires without a sim:

```bash
python -m gear_sonic.scripts.quest3_manager_x2 --help
python -m gear_sonic.scripts.quest3_manager_x2 --input-source quest3     # prints the https://<laptop-ip>:8443 URL to open on the headset
```

## Real robot

The Quest stack targets the robot through the same laptop launcher with
`--pc2-host <PC2_IP>` once the PC2 daemons are up
([`F10_real_robot_colcon.md`](F10_real_robot_colcon.md)); the pad on PC2
keeps e-stop authority. Arm/hand targets ride `arm_targets` /
`hand_finger_cmd` (:5564, :5572) into `pc2_kplanner_onnx.py`; physical
fingers go through `gear_sonic_deploy/scripts/x2_hand_zmq_to_aimdk_bridge.py`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| browser cannot reach :8443 | firewall (`ufw allow 8443/tcp 8765/tcp`), different subnet, or you launched with `INPUT_SOURCE=pico` (no WebXR server in Pico mode) |
| `Start VR` fails | Guardian/floor not set on the headset |
| WS connects, no tracking | accept the :8765 certificate too, then reconnect |
| sticks reversed | `quest3_manager_x2.py --invert-lx/-ly/-rx/-ry` (through the stack's pass-through flags), not code |
| arms track badly | run the operator calibration above and pass `--calibration` |
