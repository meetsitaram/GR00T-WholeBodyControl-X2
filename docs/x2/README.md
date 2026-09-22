# AgiBot X2 Ultra on GEAR-SONIC — runbook index

This directory documents the X2 feature set carried on this branch: gamepad
and VR control, Pico whole-body teleop, motion and tape replay, the two
model families, kplanner <-> whole-body switching, the MuJoCo sim stack, the
real-robot (PC2, colcon) deploy, the docker test image and Nebius training.

Everything here runs from a clean clone: one native dual-head SONIC set
(`gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*`, git-lfs) ships as the
default of every launcher and of the robot template, and any other policy is
**your own exported model** ([`../../MODELS.md`](../../MODELS.md)). No
captured motion data and no robot-specific address ships with the repository.

| # | Runbook | What it proves |
|---|---|---|
| F01 | [Gamepad control](F01_gamepad.md) | sticks drive the kplanner in sim and on the robot; clip banks; e-stop |
| F02 | [Quest 3 VR control](F02_quest_vr.md) | WebXR headset drives locomotion + arm IK, pad and VR coexist |
| F03 | [Pico whole-body teleop](F03_pico_teleop.md) | headset + trackers drive the whole body through the SMPL encoder |
| F04 | [Motion replay (pose encoder)](F04_motion_replay.md) | a pkl/x2m2 clip streams into the running stack |
| F05 | [Pico-tape replay (SMPL encoder)](F05_pico_tape_replay.md) | a recorded Pico SMPL tape (two ship) replays through the live path |
| F06 | [Native 3-encoder models](F06_incumbent_models.md) | export and run a native dual-head set |
| F07 | [Frozen G1-core models](F07_frozen_g1_model.md) | LoRA finetune of the public G1 core, token export path |
| F08 | [kplanner <-> whole-body switching](F08_kplanner_wholebody_switching.md) | one operator toggles locomotion and whole-body follow |
| F09 | [MuJoCo sim stack](F09_mujoco_sim.md) | `simstack_local.sh` / `sim_onnx_planner.sh`, plant checks, evals |
| F10 | [Real robot (PC2, colcon)](F10_real_robot_colcon.md) | bring-up, push, ritual ignition, model switch |
| F11 | [Docker test image](F11_docker.md) | build the deploy package and run the sim loop in a container |
| F12 | [Training and finetuning on Nebius](F12_training_nebius.md) | provision, smoke, full run, resume, export |

Reference pages: [`ARCHITECTURE.md`](ARCHITECTURE.md) (stack diagrams),
[`TRAINING_NOTES.md`](TRAINING_NOTES.md) (run history, reward deviations, traps),
[`BUILD_CHAIN.md`](BUILD_CHAIN.md) (what rebuilds when),
[`gamepad_cheatsheet.md`](gamepad_cheatsheet.md), [`../../MODELS.md`](../../MODELS.md),
[`../../x2_pc2/PORT_REGISTRY.md`](../../x2_pc2/PORT_REGISTRY.md) (every ZMQ port).

## 10-minute quickstart (sim, no robot)

Prerequisites: Ubuntu 22.04/24.04, Python 3.10, `git-lfs`, docker with the
compose plugin (for the deploy-parity sim), a gamepad if you want to drive.

```bash
# 1. clone with LFS (robot meshes)
git clone <your-fork-url> GR00T-WholeBodyControl && cd GR00T-WholeBodyControl
git lfs install && git lfs pull

# 2. python env (CPU torch first so the sim extra does not pull the CUDA wheel)
python3.10 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e "./gear_sonic[sim]" -e ./motionbricks
pip install -r requirements-x2.txt
#    or, all of the above in one idempotent step (also builds the docker image with --with-docker):
#    bash install_scripts/setup_x2.sh --skip-models

# 3. (optional) the public NVIDIA G1 release into the model cache ($SONIC_HOME, default ~/.cache/sonic).
#    Not needed to RUN anything here: both shipped sets are self-contained. It is the frozen
#    core the F07 export/finetune tools start from, and the standalone torch-checkpoint
#    Pico path (run_x2_pico_wbc.sh with a frozen-core-smpl:<ckpt>:<release.pt> composite).
pip install "huggingface_hub[cli]"
python download_from_hf.py                 # G1 core + G1 planner -> $SONIC_HOME/g1

# 4. build the motion banks: the pad/gesture bank from the shipped X2 MC stock gestures,
#    the planner primitives synthesized (add --with-upstream-examples to also retarget the
#    upstream G1 reference clips)
#    (writes gear_sonic/data/motions/x2_demo_bank.pkl, x2_planner_primitives.pkl,
#     x2_pad_banks.pkl and gear_sonic_deploy/data/motions_x2m2/demo_bank/*.x2m2)
bash tools/build_demo_bank_from_upstream.sh

# 5. models: the SONIC set SHIPS in-repo (gear_sonic_deploy/models/, git-lfs;
#    see MODELS.md "Shipped default set") -- fetch the LFS objects once.
#    Only the kplanner planner graphs (PLANNER_MODEL) are bring-your-own.
git lfs install && git lfs pull
#    (optional) your own exported X2 models (MODELS.md):
#    export X2_MODELS=$HOME/x2_models CKPT_ROOT=$HOME/x2_models

# 6. kinematic sanity check, no policy (renders a bank clip on the X2 MJCF)
python gear_sonic/scripts/play_x2_motion_mujoco.py --motion gear_sonic/data/motions/x2_demo_bank.pkl

# 7. full sim stack on the shipped set: ONNX kplanner + docker deploy + MuJoCo
#    + whole-body Pico teleop (second terminal: ./gear_sonic/scripts/run_pico_teleop.sh)
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
#    pad-only variant (hold L2 and push the left stick: the robot walks in the MuJoCo window):
#    ./gear_sonic/scripts/sim_onnx_planner.sh
#    another checkpoint (explicit env):
#    ALLOW_MISMATCH=1 MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
#    PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
#    KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
#    ./gear_sonic/scripts/sim_onnx_planner.sh

# 8. stop everything
./gear_sonic/scripts/simstack_local.sh --stop
```

Step 7 needs the docker image (`gear_sonic_deploy/docker_x2/enter_sim.sh`
builds it on first use, ~10 min) and a pad visible to `pygame` before launch.
Without a pad, use `--vr-only` (F02) or `--whole-body-teleop` (F03).

## Conventions used by every runbook

- `<PC2_IP>` is the robot's PC2 address on your network; there is no default.
  **No `--pc2-host` = the local sim stack, always; an IP = the real robot, always.**
- With no model variables every launcher and the robot template use the
  shipped set `gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*` and the shipped
  tuning preset `trained_gains_s0_inc_waistmc.yaml` ([`MODELS.md`](../../MODELS.md)).
- `$X2_MODELS` is your own model directory ([`MODELS.md`](../../MODELS.md)).
  Model file names in the commands (`x2_sonic_39000_*`, `x2_sonic_s1p4_35000_*`)
  are the example names produced by the exporters; substitute yours.
- `ALLOW_MISMATCH=1` is required for every sim run of a model that is not the
  one deployed on a reachable robot.
- Every launcher prints a stop command; `./gear_sonic/scripts/simstack_local.sh --stop`
  tears down the whole sim stack (docker container included).
- Ports are fixed and registered in [`x2_pc2/PORT_REGISTRY.md`](../../x2_pc2/PORT_REGISTRY.md);
  the sim stack renames the core chain +100 (5556 -> 5656 etc.).

## Tests

```bash
.venv/bin/python -m pytest gear_sonic/tests -q --ignore=gear_sonic/tests/test_input_readers.py
python tools/check_docs.py            # every path in these docs exists; no forbidden strings
python tools/check_x2_ownership.py    # every non-upstream file is listed in X2_FILES.txt
```

`test_input_readers.py` is an untouched upstream file that fails at collection
upstream as well (it imports a function `input_readers.py` no longer exports),
so it is ignored. The wrist-map
regression fixture `gear_sonic/tests/data/wrist_maxrange_20260905_002744Z.npz`
is an operator max-range capture shipped by the author (their own body, Pico
body tracking); it stands in for a synthetic-equivalent clip and contains no
third-party motion data.
