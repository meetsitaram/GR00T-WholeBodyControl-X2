# F05 — Pico-tape replay through the SMPL encoder

A recorded Pico tape replays through the **exact live path**:
sender -> intent stream -> tokenizer -> deploy. Nothing is restarted; the
stack is already up. Two tape kinds:

- full-session tape `record_<stamp>.npz` (all controls, starts in the
  recorded mode, reproduces every transition), and
- per-engage window clip `session_<stamp>_x2_segNN.npz` (one whole-body
  window; replays with `--auto-engage`).

Two first-party tapes ship with the repo (git-lfs; `git lfs pull` after a
fresh clone), recorded by the author on the Pico while doing household chores:

```
gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014256Z_x2_seg01.npz
gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz
```

`seg01` contains hard reaching/crouching pieces that can trip the tilt
watchdog on the shipped policy in sim after ~15 s; `seg02` is the calmer one
to start with. Record your own with the live Pico stack (F03; every session
writes a tape).

## Prerequisites

- A whole-body sim stack up ([`F03_pico_teleop.md`](F03_pico_teleop.md),
  terminal 1) or the robot ritual up with `X2_WHOLE_BODY_TELEOP=1`.
- The headset is **not** needed: tape mode runs in `.venv`.

## Sim

```bash
./gear_sonic/scripts/run_pico_replay.sh gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz
./gear_sonic/scripts/run_pico_replay.sh gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz --loop       # sim-style loop (never on the robot)
./gear_sonic/scripts/run_pico_replay.sh gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz -- --hand-close-on-grip   # extra sender args after --
```

Expected output: `run_pico_replay.sh` prints the tape kind (`WINDOW` or
full session), duration and grip counts, then the sender: planner
idle/release pre-roll (0.6 s) -> arms blend to the recorded start pose
(`--arms0-settle-s`, 1.5 s) -> `replay started` -> the tape rolls in its
recorded mode -> whole-body released -> `replay finished` -> 5 s of idle ->
the sender exits on its own (`--tape-once`; `--tape-exit-s <0` waits for
Ctrl-C). In MuJoCo the robot engages on the first engaged frame and follows
the tape.

Stop: Ctrl-C (sim has no pad daemon feed). Then
`./gear_sonic/scripts/simstack_local.sh --stop` if you are done with the stack.

## Robot

```bash
./gear_sonic/scripts/run_pico_replay.sh gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz --pc2-host <PC2_IP>
```

The script prints the checklist and asks you to type `replay`. Protocol:

1. Ritual up, kplanner idle-standing, whole-body **not** engaged; read the
   deploy's `Loaded ONNX ...` line — that is the checkpoint under test.
2. Floor clear for the whole tape; pad in hand — the tape carries no
   controller chords, the **pad e-stop is the only e-stop**.
3. Type `replay`. Stop at any time with the pad: **LB+RB** or **RB+B** (the
   replay sender subscribes to the pad daemon's `pad_state` on :5569,
   releases whole-body, idles the planner and exits; the pad then owns
   locomotion).

`run_pico_teleop.sh` and `run_pico_replay.sh` are exclusive (one sender per
target); both refuse to start while the other is running.

## Headset-free tape gates on the standalone SMPL path (sim only)

```bash
# watch it in the viewer (loops):
./gear_sonic/scripts/run_x2_pico_wbc.sh --onnx $X2_MODELS/pico_39000/exported/x2_sonic_39000_smpl.onnx \
    --tape-replay gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz --auto-engage
# headless regression gate (metrics from the qpos dump):
./gear_sonic/scripts/run_x2_pico_wbc.sh --onnx $X2_MODELS/pico_39000/exported/x2_sonic_39000_smpl.onnx \
    --tape-replay gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz \
    --headless --seconds 125 --auto-engage --qpos-dump /tmp/live_gate_qpos.npz
# the driver directly, e.g. the public G1 core on the G1 body:
.venv/bin/python gear_sonic/scripts/live_pico_smpl_teleop.py \
    --checkpoint "smpl-g1:$SONIC_HOME/g1/sonic_v1_1/last.pt" --robot g1 \
    --tape-replay gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz --auto-engage --no-record
```

The tape loops on this path and the wrap snaps the pose back to frame 0 in
one tick — a fall at the seam is a replay artifact, not a model property.

## Converting a tape for training / analysis

```bash
python gear_sonic/scripts/pico_tape_to_smpl_obs.py --tape gear_sonic/data/motions/x2_pico_chores/tapes/session_20260917_014453Z_x2_seg02.npz --out /tmp/chores_seg02_smpl.npz
```

produces the root-relative SMPL joints / global orientation stream the SMPL
encoder consumes (840-D path). Corpus building for finetuning is in
[`F12_training_nebius.md`](F12_training_nebius.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `usage: run_pico_replay.sh <tape.npz> ...` | the tape path is wrong; the file must exist |
| replay refuses to start: another sender | Ctrl-C the live `run_pico_teleop.sh` first |
| fewer steps in replay than live | planner cold-start ramp depends on its idle state; record with >= 3 s pushes and a real stop before engaging |
| robot never engages | the deploy needs `X2_WHOLE_BODY_TELEOP=1` and a paired token set / dual graph loaded (check `Loaded ONNX`) |
