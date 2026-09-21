# F12 — Training and finetuning on Nebius

Training uses the upstream SONIC trainer (`gear_sonic/train_agent_trl.py`,
Isaac Lab) with the X2 embodiment (`gear_sonic/envs/manager_env/robots/x2_ultra.py`,
`gear_sonic/utils/embodiment/x2.py`) and the X2 experiment configs under
`gear_sonic/config/exp/manager/universal_token/all_modes/`. The cloud helpers
in `gear_sonic/scripts/cloud/` wrap node bootstrap, staging, smoke, multi-node
launch and monitoring. This page is self-contained; the upstream training
docs (`docs/source/user_guide/training.md`) cover the trainer itself.

The first-party Pico chores set ships as the default corpus
(`gear_sonic/data/motions/x2_pico_chores/`, 14 clips + SMPL sidecars; every
`MOTION_FILE` / `SMPL_MOTION_DIR` default points at it). For a real run bring
your own motion-lib pkl (SOMA / GMR retargets, your Pico recordings; the
corpus the shipped set was trained on came from an external G1-to-X2 retarget
pipeline over the G1 motion corpus, neither of which is part of this repo) or start from the bank built by
`tools/build_demo_bank_from_upstream.sh` (`gear_sonic/data/motions/x2_demo_bank.pkl`: the
shipped X2 MC stock gestures; `--with-upstream-examples` adds the upstream G1 reference clips).

## Prerequisites (local)

- `python download_from_hf.py --training` (public G1 checkpoint + SMPL data
  into `$SONIC_HOME/g1`); SMPL-X body models per the upstream docs.
- Nebius CLI authenticated; `NEBIUS_PROJECT` (parent id) exported; an SSH key.
- Your fork pushed (the node clones it; a private repo needs a read-only
  deploy key or a token in the node's git credential store — set that up
  before cloning, and verify with `git ls-remote` from the node).

## 1. Find capacity and provision a node

```bash
python gear_sonic/scripts/cloud/nebius_gpu_scan.py                   # where is an 8xH100/H200 available right now
python gear_sonic/scripts/cloud/nebius_parallel_provision.py --help  # fire several region/platform variants, first RUNNING+ssh wins
```

`nebius compute instance create` does not fail fast; the scan asks the
capacity API first and the parallel provisioner deletes the losers. A node
that reaches RUNNING but never opens TCP/22 within ~90 s is a husk — stop it.

Hardware preflight once you are on the node (NVLink, IB rails, idle GPUs):

```bash
bash gear_sonic/scripts/cloud/preflight_hw.sh
```

## 2. Bootstrap the node (idempotent)

```bash
ssh ubuntu@${NEBIUS_HOST}
git clone <your-fork-url> GR00T-WholeBodyControl && cd GR00T-WholeBodyControl
bash gear_sonic/scripts/cloud/bootstrap_fresh_node.sh
```

Installs conda + Isaac Lab (pinned isaacsim), `pip install -e "gear_sonic/[training]"`
plus the extras the training extra misses, git-lfs for the meshes, and
pre-accepts the Omniverse EULA. It does **not** copy your motion data or
checkpoints — stage them next.

## 3. Stage data

`stage_pico_v8.sh` reads four required exports and copies them to
`~/train_data/` (fine-tune pkl), `~/train_data/smpl_filtered/` (SMPL
sidecars) and `~/run_pico/` (warm-start checkpoint + its config) on the node,
md5-verified both sides:

```bash
export PICO_FT_PKL=<fine-tune motion-lib pkl>      # build_pico_finetune_bundle.py output
export PICO_SIDECAR_DIR=<its SMPL sidecar dir>     # per-clip SMPL pkls
export CKPT=<model_step_NNNNNN.pt>                 # your warm-start checkpoint (MODELS.md)
export CFG=<config.yaml next to that checkpoint>
bash gear_sonic/scripts/cloud/stage_pico_v8.sh ubuntu@${NEBIUS_HOST}
# node-local copy of the payload so 8-32 GPUs do not read small files off a shared mount
bash gear_sonic/scripts/cloud/stage_local.sh
```

`STAGE_FALLBACK=1` additionally stages a base corpus (`MAIN_PKL`, default the
regenerated demo bank `gear_sonic/data/motions/x2_demo_bank.pkl`) into the
node's checkout when none is on the node yet.

Corpus tools (local, `.venv`):

| Task | Command |
|---|---|
| SOMA retargeter CSV/PKL -> motion-lib pkl | `python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py --robot x2_ultra ...` |
| GMR raw retarget npz -> motion-lib pkl (+ sanity gates, render) | `python gear_sonic/scripts/x2_gmr_to_motion_lib.py --raw <retarget.npz> --key <key> --pkl <out.pkl>` |
| recorded robot sessions -> motion-lib | `python gear_sonic/data_process/convert_x2_record_to_motion_lib.py --help` |
| drop clips with IK basin jumps | `python gear_sonic/data_process/filter_kinematic_continuity.py --help` |
| validate a retargeted Pico corpus (fps, finite, saturation) | `python gear_sonic/data_process/validate_pico_corpus.py <corpus_dir> --json /tmp/keep.json` |
| append it to a fine-tune set + SMPL sidecars (append-only, sampler-safe; despikes retarget wrap flips per joint, `--no-despike` to keep them) | `python gear_sonic/data_process/build_pico_finetune_bundle.py --corpus-dir <corpus> --tape-dir <tapes> --base-pkl <base.pkl> --out-pkl <ft.pkl> --sidecar-dir <sidecars> --keep-json /tmp/keep.json` |
| resample to 50 fps by interpolation (SLERP on quats) | `python gear_sonic/scripts/cloud/resample_motion_pkl.py --in <a.pkl> --out <a_50fps.pkl> --fps 50` |

## 4. Preflight the config on CPU (no Isaac Sim)

```bash
python gear_sonic/scripts/cloud/preflight_train.py +exp=manager/universal_token/all_modes/sonic_x2_ultra \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<your.pkl>
```

Catches hydra resolution errors, motion-lib load failures, zero SMPL coverage
and reward/command body-name mismatches before a 5-minute Isaac Sim start.
`MOTION_FILE` and `SMPL_MOTION_DIR` default to the shipped reference set
`gear_sonic/data/motions/x2_pico_chores/` (14 paired clips, 878 s, first-party
Pico teleop recordings with 100% sidecar coverage). For your own data point both
at your corpus and run the frame/timebase gate first:

```bash
python gear_sonic/scripts/check_smpl_sidecars.py $SMPL_MOTION_DIR --motion <your.pkl>
```

Why both gates exist: a sidecar set with the right field names and shapes but
root-local joints, a zeroed height channel or a mismatched fps still loads, and
the SMPL encoder then trains on garbage while every run "passes".

## 5. Smoke on all GPUs (~3 min)

```bash
tmux new -d -s smoke "bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh"        # defaults: the shipped chores reference set
# your own data: MOTION_FILE=<your.pkl> SMPL_MOTION_DIR=<sidecar dir> bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
tmux a -t smoke        # or: tail -f ~/smoke.log
```

The smoke runs the full three-encoder config (`sonic_x2_ultra_smoke.yaml`
inherits `sonic_x2_ultra.yaml` with a 200-iteration schedule), so it needs the
SMPL sidecars like a real run; the shipped chores set provides them. It refuses to launch until
`check_smpl_sidecars.py` and `preflight_train.py` both pass (`SKIP_GATES=1`
bypasses them for debugging only).

Defaults: 200 PPO iterations, 4096 envs/GPU, W&B off. Expected: Isaac Sim
boots on every GPU, `accelerate launch --num_processes 8 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_x2_ultra ...`,
iteration lines with a finite reward, a checkpoint under the run dir.
For a laptop-only config check use `++num_envs=2` with the same variables.

## 6. Full run

Single node:

```bash
tmux new -d -s train "NUM_ENVS=16384 NUM_ITERS=20000 EXP_NAME=sonic_x2_ultra MOTION_FILE=<your.pkl> USE_WANDB=False LOG_FILE=\$HOME/train.log bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh"
```

Multi-node (4 x 8 GPUs, elastic): run on every node with a distinct
`NODE_RANK`, rank 0 is the rendezvous host:

```bash
NODE_RANK=0 MASTER=<rank0 private ip> bash gear_sonic/scripts/cloud/launch_bigrun.sh          # fresh
NODE_RANK=0 MASTER=<rank0 private ip> CKPT=$HOME/run/last.pt bash gear_sonic/scripts/cloud/launch_bigrun_resume.sh   # +resume
```

Edit `EXP=` at the top of the launch scripts to your config leaf. Preemptible
fleets: `bash gear_sonic/scripts/cloud/cluster_watchdog.sh` on the reserved
node restarts stopped workers (`PARENT_ID=${NEBIUS_PROJECT} NODE_PREFIX=...`);
`gear_sonic/scripts/cloud/elastic-supervisor.service` rejoins a rebooted
worker; `bash gear_sonic/scripts/cloud/watch_8gpu_capacity.sh` polls for a
free 8-GPU preset. The systemd unit ships no launcher: its `ExecStart` runs
`/home/ubuntu/launch_elastic_current.sh`, which you provide (a copy of
`launch_bigrun.sh` with your `EXP`, `MASTER` and `NODE_RANK` baked in; the
rendezvous host sets `RDZV_IS_HOST=1` in its copy).

Config knobs that matter for X2:

| Config | Use |
|---|---|
| `sonic_x2_ultra.yaml` | native 3-encoder X2 policy from scratch or warm start |
| `sonic_x2_ultra_smoke.yaml` | 2-env config smoke |
| `sonic_x2_frozen_core_t2.yaml` | frozen G1 core + Phi codec + decoder LoRA, base recipe ([`F07_frozen_g1_model.md`](F07_frozen_g1_model.md)) |
| `sonic_x2_s1_fresh_vendor_waistmap.yaml` | frozen-G1 recipe on the vendor plant with the wrist/waist range map (produced the published frozen-G1 set) |
| `sonic_x2_leg5_pico_combined_ft.yaml` | Pico-corpus finetune of a native checkpoint (`run_pico_v8_8gpu.sh`; the lineage of the shipped `v16ft8_45000` set) |
| `gear_sonic/config/callbacks/finetune_rate_schedule.yaml` | finetune LR schedule callback |
| `gear_sonic/config/manager_env/events/terms/randomize_actuator_gains.yaml`, `randomize_actuator_gains_deploy.yaml` | per-episode KP/KD randomisation (train vs deploy envelope) |
| `gear_sonic/config/robot_plant/x2_ultra_vendor_20260823.yaml` | the actuator plant the deploy binary is built against |

Resume rules: same schedule -> `++resume=True ++experiment_dir=<run dir>`
(optimizer + LR state); new budget or config -> `+checkpoint=<model_step_N.pt>`
(weights only). Iterations are **relative** on `+checkpoint` warm starts.

## 7. Monitor and pull back

```bash
python gear_sonic/scripts/cloud/wandb_health.py <entity/project/run_id>     # one line: HEALTHY / BOOTING / ALARM
rsync -az --partial --info=progress2 ubuntu@${NEBIUS_HOST}:~/<run_dir>/ $X2_MODELS/<run>/
python3 -c "import torch; torch.load('$X2_MODELS/<run>/last.pt', map_location='cpu'); print('INTACT')"
```

Never re-download over a good copy unverified. Then export
([`BUILD_CHAIN.md`](BUILD_CHAIN.md) A) and evaluate in MuJoCo
([`F09_mujoco_sim.md`](F09_mujoco_sim.md)) — Isaac Lab numbers alone never
qualify a model for the robot.

Local single-GPU finetune template (edit the checkpoint / corpus paths at the
top): `bash gear_sonic/scripts/run_local_finetune_demo_v2.sh`.

Isaac Lab tracking eval of a checkpoint (the same metrics the training
callback logs, on a fixed motion set):

```bash
python gear_sonic/eval_agent_trl.py checkpoint=$X2_MODELS/<run>/last.pt +headless=True +num_envs=1000 \
    +manager_env/terminations=tracking/eval eval_name=<tag> \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/motions/x2_demo_bank.pkl
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NotEnoughResources` after 5 min | use `nebius_gpu_scan.py` first; try another platform/region or the parallel provisioner |
| CUDA error 802 / "doesn't support bf16" | fabric manager down after a driver rebuild (`preflight_hw.sh` catches it) |
| XML parse error 8 from MuJoCo on the node | LFS pointer files: `git lfs pull` |
| reward NaN from iteration 1 | `preflight_train.py`: a reward body name not in the command set |
| SMPL losses flat at zero | `smpl_motion_file` is `dummy`/`null` or sidecars do not join by key; the preflight prints the coverage |
| distributed run dies at a random iteration on a sidecar length | frame-count mismatch > 2 frames between a clip and its SMPL sidecar; rebuild the bundle |
