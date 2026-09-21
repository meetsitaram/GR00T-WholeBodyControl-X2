# F04 — Motion replay through the pose encoder

`run_motion_replay.sh` streams a robot-joint clip (motion-lib `.pkl` or
baked `.x2m2`) into a stack that is **already up**: the clip is baked to
X2M2, dropped into the kplanner's dances dir and played with one
`motion_clip_cmd`. The kplanner is only the wire relay (frames stream
verbatim); SONIC tracks the joints, so a walking clip walks.

The only motion data that ships is the X2 Motion-Controller stock-gesture set
(`gear_sonic/data/motions/x2_recorded/mc_gestures/`, 51 clips, LFS). Build the bank
from it first (upstream G1 reference clips are opt-in via `--with-upstream-examples`):

```bash
bash tools/build_demo_bank_from_upstream.sh
# -> gear_sonic/data/motions/x2_demo_bank.pkl (multi-key), gear_sonic_deploy/data/motions_x2m2/demo_bank/*.x2m2
```

## Prerequisites

- Quickstart of [`README.md`](README.md); `CKPT_ROOT` set (bakes land in
  `$CKPT_ROOT/dances_x2m2/`).
- A sim stack up (F01 command, or the whole-body stack of F03), or the robot
  ritual up.
- `MODEL=` the pose graph the stack runs (used for the `--own-deploy` path and
  the identity print).

## Sim (into the running stack)

```bash
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
./gear_sonic/scripts/run_motion_replay.sh gear_sonic/data/motions/x2_demo_bank.pkl --key <clip key> --dur-s 5.5
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
./gear_sonic/scripts/run_motion_replay.sh gear_sonic/data/motions/x2_demo_bank.pkl --key <clip key> --start-s 12.5 --dur-s 4
./gear_sonic/scripts/run_motion_replay.sh gear_sonic_deploy/data/motions_x2m2/demo_bank/<clip>.x2m2
```

`--key` selects a key of a multi-clip pkl (default: first key);
`--start-s/--dur-s` window the clip (a full corpus clip travels metres — bake
a few seconds on a gantry). Clip keys: `python -c "import joblib;print(list(joblib.load('gear_sonic/data/motions/x2_demo_bank.pkl')))"`.

Expected output: `bake -> $CKPT_ROOT/dances_x2m2/<key>.x2m2 (N frames @ fps)`,
`motion_clip_cmd sent`, the kplanner log `clip complete` at the end and the
planner resuming idle. The robot in MuJoCo performs the clip.

Stop: the pad's **L1+R1** chord, a second run with `--stop`, or Ctrl-C.

## Sim (own deploy, kplanner-free)

```bash
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
./gear_sonic/scripts/run_motion_replay.sh gear_sonic/data/motions/x2_demo_bank.pkl --key <clip key> --own-deploy [--max-duration 20] [--no-viewer]
```

This runs `deploy_x2.sh sim --input-type motion_file` on the bake (refuses
to stack on a running sim stack). Stop: Ctrl-C.

## Robot

```bash
./gear_sonic/scripts/run_motion_replay.sh gear_sonic/data/motions/x2_demo_bank.pkl --key <clip key> --dur-s 5.5 --pc2-host <PC2_IP>
```

The bake is staged to PC2 and played into the ritual's kplanner. **L1+R1** on
the pad stops it; the pad e-stop stays live. `--own-deploy` on the robot
opens the deploy pane in tmux `x2_deploy` with the ritual **down** and leaves
the MC handoff / Y-N gate to the operator.

## Building your own clips

| Task | Command |
|---|---|
| bake a window by hand | `python gear_sonic/scripts/pkl_to_x2m2.py --pkl <clip.pkl> --key <key> --start-s 0 --dur-s 5 --out $CKPT_ROOT/dances_x2m2/<key>.x2m2` |
| convert a pkl for `deploy_x2.sh --motion` | `python gear_sonic_deploy/scripts/export_motion_for_deploy.py --in <clip.pkl> --out <clip.x2m2>` |
| window long clips for kplanner training | `python gear_sonic/scripts/window_motion_pkl.py --in <long.pkl> --out <windows.pkl> --max-frames 200 --min-frames 80` |
| kinematic preview, no policy | `python gear_sonic/scripts/play_x2_motion_mujoco.py --motion <clip.pkl> [--speed 0.5] [--start-sec S --end-sec E]` |
| closed-loop MuJoCo eval of a `.pt` | `python gear_sonic/scripts/eval_x2_mujoco.py --checkpoint <ckpt.pt> --motion <clip.pkl>` |
| closed-loop MuJoCo eval of the ONNX | `python gear_sonic/scripts/eval_x2_mujoco_onnx.py --onnx <name>_g1.onnx --motion <clip.pkl>` |
| retarget a SOMA CSV/PKL export | `python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py --robot x2_ultra ...` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MISSING: .../dances_x2m2` | set `CKPT_ROOT` (the sim stack passes it as its dances dir) |
| clip plays but feet slide | the clip player pins the pelvis; for stepping motions use a kplanner primitive (`x2_planner_primitives.pkl`) instead |
| robot falls on a fast clip | shorten with `--start-s/--dur-s`; check the deploy tuning preset matches the model's plant |
| nothing happens | the stack is not up, or a previous clip is still playing (`L1+R1`) |
