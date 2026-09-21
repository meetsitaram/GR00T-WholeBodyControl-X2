# F08 — kplanner <-> whole-body teleop switching

One operator, one headset (or headset + pad), two authorities:

- **LOCOMOTION**: the kinematic planner (`gear_sonic/scripts/pc2_kplanner_onnx.py`,
  torch-free ONNX runtime; `gear_sonic/scripts/x2_kplanner.py` is the torch
  twin) turns stick intents into 50 Hz pose references. Dances, gestures and
  primitives play through the same wire.
- **WHOLE_BODY**: the operator's SMPL stream owns the command slot (token or
  dual head); the kplanner shadows, holds, and takes back on release.

The switch is behavioural, not a separate process: `B` on the Pico toggles
LOCOMOTION <-> WHOLE_BODY; the deploy's per-tick arbiter
(`gear_sonic/utils/pose_pipeline/arbitrate.py`, `fallback.py`, `wire.py`)
selects the source; the kplanner state machine
(`gear_sonic/utils/planner/state_machine.py`, recipes in
`gear_sonic/utils/planner/x2_recipes.py`) handles the return.

## Prerequisites

- [`F03_pico_teleop.md`](F03_pico_teleop.md) working in sim (either model set).
- Planner graphs `PLANNER_MODEL=<dir>` and a profile `KPLANNER_PROFILE=`
  ([`MODELS.md`](../../MODELS.md)).
- `gear_sonic/data/motions/x2_planner_primitives.pkl` from
  `tools/build_demo_bank_from_upstream.sh` (idle anchor and step primitives).

## Sim

```bash
cd <repo> && ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 KPLANNER_WB_ARM_LATCH_GAIT=keep \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_dual.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

then `./gear_sonic/scripts/run_pico_teleop.sh`. Walk-through:

1. **A+B+X+Y** -> LOCOMOTION. Hold LT, push the left stick: the kplanner
   walks the robot (log: `Replanning with mode ...`).
2. Release the stick and wait for a real stop (`PLAYING -> STOPPING`, foot
   align, anchor blend).
3. **B** -> WHOLE_BODY. The sender publishes one `idle + vr_release`; the
   kplanner drops to idle under the overlay; the deploy arbiter switches to
   the token/SMPL head. Your torso and arms move the robot.
4. **B** -> release. Deploy hold -> kplanner rate-limited return
   (`KPLANNER_WB_RETURN=1`, `KPLANNER_WB_HOLD_LEGS=1` in `x2_pc2/robot_env.env`)
   -> LOCOMOTION again; the sticks work immediately.
5. Re-engaging re-anchors your heading to the robot's, so there is no turn
   transient. Stream loss > 0.5 s freezes the robot in place; the pad (if
   present) has authority.

Expected log lines: sender `mode LOCOMOTION -> WHOLE_BODY (engaged)`,
`released`; kplanner `wb overlay: shadow`, `RETURN`; deploy `token authority`
/ `pose authority` transitions.

Planner-only sim (pad + kplanner, dances, primitives) is the F01 command;
`PLANNER=velocity` swaps the graph; `KPLANNER_PROFILE` picks the knob file
(`bigrun_teleop_v1.env` teleop profile, `g1core_bare_v1.env` for the frozen-G1
planner; the robot's knobs live in `x2_pc2/robot_env.env`, from `x2_pc2/robot_env.env.template`).

Stop: Ctrl-C the sender, `./gear_sonic/scripts/simstack_local.sh --stop`.

## Testing the planner without a headset

```bash
# scripted intents through the planner ONNX -> a pkl you can replay through SONIC (F04)
python gear_sonic/scripts/gen_kplanner_clip.py --planner-onnx $X2_MODELS/kplanner_g1core/x2_planner_template.onnx \
    --out /tmp/kplanner_walk.pkl --seconds 8 --vel-z 0.5 --mode slow_walk
# a pkl clip as the planner_cmd source instead of a headset (velocity intent per frame)
python gear_sonic/scripts/x2_pkl_command_source.py --help
# replay a recorded intent tape into the live sim stack at its recorded timings
python gear_sonic/scripts/simstack_replay_tape.py --tape <intents.jsonl>
# pad emulation on the wire (in-place turn, 20 Hz, same payload as the bridge)
bash gear_sonic/scripts/simstack_turn.sh right 3
# scripted walk / crouch walk (deadman + stick, or the crouch chord) on the same wire
python gear_sonic/scripts/simstack_pad_drive.py walk 6 0.6
python gear_sonic/scripts/simstack_pad_drive.py crouchwalk 8 0.6
```

Before any planner code or primitive push to the robot, the regression suite
must pass (headless, CPU, ~3 min; `push_to_pc2.sh` runs it as gate 1):

```bash
.venv/bin/python gear_sonic/scripts/preflight_planner.py
```

## Robot

`X2_WHOLE_BODY_TELEOP=1` in `x2_pc2/robot_env.env` (default) arms the
overlay at ignition; the ritual is otherwise a plain pad session. The
laptop half is `./gear_sonic/scripts/run_pico_teleop.sh --pc2-host <PC2_IP>`.
Ports: intent :5573, tokens :5574, planner_cmd :5563, pose :5556 -> watchdog
:5558 -> deploy ([`x2_pc2/PORT_REGISTRY.md`](../../x2_pc2/PORT_REGISTRY.md)).

## Troubleshooting

| Symptom | Fix |
|---|---|
| B does nothing | you are not in LOCOMOTION (chord first), or the body solve is dead (`ENGAGE REFUSED`) |
| robot twists toward one world direction at ignition | measured-yaw rebase is off; the kplanner needs the deploy's `x2_debug` (`--x2-debug-port 5557`, sim :5659); `KPLANNER_YAW_REBASE=1` in sim |
| return after release is abrupt | `KPLANNER_WB_RETURN=1` and `KPLANNER_WB_HOLD_LEGS=1` must be exported (they are in `robot_env.env`) |
| kplanner crashes on bind | a port collision; check the registry, never allocate a socket without a row there |
| `preflight_planner.py` fails | fix before pushing; the suite covers e-stop, stick margins, primitive bounds and the serve loop |
