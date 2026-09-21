# MODELS.md — the shipped default set, and bring-your-own models for the X2 stack

## Shipped default set (v16ft8_45000)

One native 3-encoder SONIC set ships in this repository under
`gear_sonic_deploy/models/` (git-lfs; run `git lfs pull` after a fresh clone
if the files are ~130-byte pointer files). It is the set the sim ritual
(`./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop`,
`simstack_local.sh`) and the robot ritual (`x2_pc2/robot_env.env.template`,
staged by `pc2_bringup.sh` step 9) pick automatically when no model variable
is set, paired with the shipped tuning preset
`gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml`
(`SIM_TUNING_YAML` / `X2_RITUAL_TUNING` default), `X2_PLANT=vendor_20260823`
and `X2_ANCHOR_ORI_MODE=body`.

| File (`gear_sonic_deploy/models/`) | Kind | Obs dim | md5 (first 12) | Env var it fills |
|---|---|---|---|---|
| `x2_sonic_v16ft8_45000_dual.onnx` | native dual-head (both reference heads, one session) | 2511 | `09d7b6971ee7` | `MODEL` / `SIMSTACK_DUAL_MODEL` (sim), `X2_RITUAL_MODEL` (robot) |
| `x2_sonic_v16ft8_45000_g1.onnx` | pose graph (deploy primary slot, pad / planner) | 1670 | `dae6b74a8d10` | `MODEL` / `SONIC_MODEL` / `SIMSTACK_SONIC` (sim), the ritual's `_g1` sibling (robot), `run_motion_replay.sh --own-deploy` |
| `x2_sonic_v16ft8_45000_g1_token.onnx` | token-input graph (legacy token path) | 1670 | `a7a82d3e3e17` | `SIMSTACK_TOKEN` (sim), `X2_WB_TOKEN_MODEL` (robot) |
| `x2_sonic_v16ft8_45000_smpl_tokenizer.onnx` | SMPL tokenizer `smpl_obs[840] -> motion_token[64]` | 840 | `e7fec5210ee0` | `SIMSTACK_TOKENIZER` (sim), `X2_WB_TOKENIZER` (robot) |

All four carry `codec_fingerprint native-3enc:52d8ca59...` (same checkpoint,
iteration 45000; `x2_sonic_v16ft8_45000.manifest.txt` next to them lists every
md5). Verify: `python gear_sonic/scripts/onnx_provenance.py check --pose
gear_sonic_deploy/models/x2_sonic_v16ft8_45000_g1.onnx --dual
gear_sonic_deploy/models/x2_sonic_v16ft8_45000_dual.onnx`. The export chain
that produced it is [`docs/x2/BUILD_CHAIN.md`](docs/x2/BUILD_CHAIN.md) A
(`export_native_all.sh`). The kplanner planner graphs (`PLANNER_MODEL`) are
**not** part of the shipped set; see the tables below.

## Shipped frozen G1-core set (s1ft16000)

The frozen-G1 lineage (F07) also ships, under `gear_sonic_deploy/models/`
(git-lfs): the set the robot ran on 2026-09-09 and the same bytes as
`tinkerbuggy/sonic-x2/demo_20260909/frozen_g1_s1ft16000/`. It is not the
launcher default (that stays `v16ft8_45000`); select it explicitly:

| File (`gear_sonic_deploy/models/`) | Kind | Obs dim | md5 (first 12) | Env var it fills |
|---|---|---|---|---|
| `x2_sonic_s1ft16000_g1.onnx` | pose graph (frozen G1 core + folded X2 LoRA) | 1670 | `5ae931344306` | `MODEL` (sim), `X2_RITUAL_MODEL` (robot) |
| `x2_sonic_s1ft16000_g1_token.onnx` | token-input graph (whole-body teleop) | 1670 | `438a14fea884` | `SIMSTACK_TOKEN` (sim), `X2_WB_TOKEN_MODEL` (robot) |
| `x2_smpl_tokenizer_v11release.onnx` | public G1 release SMPL tokenizer `smpl_obs[840] -> motion_token[64]` | 840 | `ee03f06e0211` | `SIMSTACK_TOKENIZER` (sim), `X2_WB_TOKENIZER` (robot) |

Pair it with `trained_gains_s0.yaml`, `X2_PLANT=vendor_20260823`, the G1-core
planner from HF `tinkerbuggy/sonic-x2/kplanner_g1core/` (`PLANNER_MODEL`, see the directory
convention below) and `gear_sonic/config/kplanner_profiles/g1core_bare_v1.env`.
`x2_pc2/robot_env.env.frozen.template` already points at these files. Sim:

```bash
ALLOW_MISMATCH=1 SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=gear_sonic_deploy/models/x2_sonic_s1ft16000_g1.onnx \
SIMSTACK_TOKEN=gear_sonic_deploy/models/x2_sonic_s1ft16000_g1_token.onnx \
SIMSTACK_TOKENIZER=gear_sonic_deploy/models/x2_smpl_tokenizer_v11release.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/g1core_bare_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop      # or without the flag: pad / planner only
```

## License of the shipped sets

Licensed by NVIDIA Corporation under the NVIDIA Open Model License: both
shipped sets are Derivative Models of the GEAR-SONIC release (Part 2 of
[`LICENSE`](LICENSE), sections 2 and 3). Redistributing them, or a model you
derive from them, requires a copy of that Agreement and the attribution notice
above; see `gear_sonic_deploy/models/README.md`.

## Bring your own

Any other X2 policy is bring-your-own. Every launcher resolves its models
through the environment variables below (an explicit variable always wins
over the shipped default); you point them at ONNX files you exported yourself
(from a checkpoint you trained or finetuned, see
[`docs/x2/F12_training_nebius.md`](docs/x2/F12_training_nebius.md)) or at the
public NVIDIA G1 release where one applies.

The only downloadable default is the public **`nvidia/GEAR-SONIC`** release
(G1 core + G1 planner), fetched with `download_from_hf.py`. It is the frozen
core that the frozen-G1 recipe (F07) finetunes from; it is **not** an X2
policy and will not drive the X2 sim by itself.

Earlier X2 sets are published on Hugging Face at
**`https://huggingface.co/tinkerbuggy/sonic-x2`** (public, no login):
the native incumbent `tinkerbuggy/sonic-x2/demo_20260909/incumbent_39k/` (F06), the frozen G1-core
set `tinkerbuggy/sonic-x2/demo_20260909/frozen_g1_s1ft16000/` (F07), the kplanner planner graphs
`tinkerbuggy/sonic-x2/kplanner_onnx/` (incumbent lineage) and `tinkerbuggy/sonic-x2/kplanner_g1core/` (frozen-core
lineage) for `PLANNER_MODEL` (F08), the torch planner checkpoints
`tinkerbuggy/sonic-x2/kplanner_torch/` for re-export, and the training configs those runs used.
Folder-by-folder pairing is in `gear_sonic_deploy/models/README.md`. Fetch a
folder with:

```bash
huggingface-cli download tinkerbuggy/sonic-x2 --include 'demo_20260909/incumbent_39k/*' --local-dir $X2_MODELS
```

## Directory convention used by every runbook

The runbooks use one shell variable, `X2_MODELS`, for "the folder where my
exported X2 models live". Keep it under `$HOME` so the docker sim (which
bind-mounts `$HOME` at the same path) can read it:

```bash
export X2_MODELS=$HOME/x2_models          # any writable dir under $HOME
export CKPT_ROOT=$X2_MODELS               # see CKPT_ROOT below
```

`export_native_all.sh <ckpt.pt> <name>` writes its five graphs to
`<ckpt dir>/exported/<name>_*.onnx`, so a typical tree looks like:

```
$X2_MODELS/
  pico_39000/exported/x2_sonic_39000_{g1,smpl,smpl_tokenizer,g1_token,dual}.onnx   # a native 3-encoder set
  s1p4_35000/exported/x2_sonic_s1p4_35000_{g1,g1_token}.onnx                      # a frozen-G1 set
  armA_14900/exported/x2_smpl_tokenizer_v11release.onnx                           # release SMPL tokenizer
  kplanner_g1core/x2_planner_{template,velocity}.onnx                             # kplanner graphs (a DIRECTORY, see below)
  dances_x2m2/*.x2m2                                                              # baked clips ($CKPT_ROOT/dances_x2m2)
```

`PLANNER_MODEL` is a directory that must hold `x2_planner_template.onnx`
(and `x2_planner_velocity.onnx` for the velocity graph). The Hugging Face
folder `tinkerbuggy/sonic-x2/kplanner_g1core/` ships the G1-core planners
under their variant names (`tinkerbuggy/sonic-x2/kplanner_g1core/x2_planner_template_s1d_a05_ws.onnx` is the one
the robot ran); copy or symlink the variant you want to
`$X2_MODELS/kplanner_g1core/x2_planner_template.onnx` (or rebuild the graph
yourself: F07 "G1-core planner graphs"). The HF folder
`tinkerbuggy/sonic-x2/kplanner_onnx/` holds the incumbent template / velocity pair under the plain names.

The set names above (`pico_39000`, `s1p4_35000`, ...) are only the example
names used by the acceptance commands; they are the `<name>` you pass to the
exporter. Substitute your own.

## Environment variables

Observation dims come from the `export_native_all.sh` header:
pose graph **1670**, fused SMPL **1830**, SMPL tokenizer **840**, token graph
**1670**, dual-head **2511**. Every graph outputs **31** actions.

| Variable | Read by | File kind / contract | How to produce | Public default |
|---|---|---|---|---|
| `MOTION_FILE` | every X2 training config (`motion_lib_cfg.motion_file`), `run_smoke_8gpu.sh`, `preflight_train.py` | motion-lib pkl (or per-clip dir): `dof (T,31)`, `root_trans_offset`, `root_rot`, `fps` | shipped chores set, or `tools/build_demo_bank_from_upstream.sh` (gesture bank), or your own corpus / Pico bundle | `gear_sonic/data/motions/x2_pico_chores/x2_pico_chores_0917.pkl` (14 first-party Pico teleop clips) |
| `SMPL_MOTION_DIR` | every X2 training config (`motion_lib_cfg.smpl_motion_file`), `run_smoke_8gpu.sh`, `preflight_train.py`, `check_smpl_sidecars.py` | directory of SMPL sidecars, one `<clip key>.pkl` per motion: `pose_aa (T,72)`, `transl (T,3)`, `smpl_joints (T,24,3)`, `fps`, corpus frame (y-up, pelvis rest offset) | `gear_sonic/data_process/build_pico_finetune_bundle.py --sidecar-dir` from Pico tapes; gate with `check_smpl_sidecars.py` | `gear_sonic/data/motions/x2_pico_chores/smpl_sidecars/` (100% coverage of the chores set); no zero-filled fallback exists |
| `MODEL` (alias `SONIC_MODEL`) | `sim_onnx_planner.sh`, `simstack_local.sh` (as `SIMSTACK_SONIC`), `run_x2_pico_wbc.sh`, `run_motion_replay.sh --model`, `deploy_x2.sh --model` | Pose-reference graph `<name>_g1.onnx`, obs 1670 -> 31. With `--whole-body-teleop` it may also be the `<name>_dual.onnx` (the launcher finds the `_g1` sibling) or, for the standalone SMPL sim path, a `*smpl*.onnx` (obs 1830) / a `.pt` composite string | Native set: `gear_sonic/scripts/export_native_all.sh <ckpt.pt> <name>`. Frozen-G1 set: `gear_sonic/scripts/frozen_core_t2_export.py` then `gear_sonic/scripts/frozen_core_t2_onnx_export.py --remap-waist --remap-wrist`. Legacy single graph: `gear_sonic_deploy/scripts/reexport_x2_onnx.sh <run-dir>` | shipped `x2_sonic_v16ft8_45000_g1.onnx` (`_dual` with `--whole-body-teleop`) |
| `SIMSTACK_DUAL_MODEL` | `simstack_local.sh` (set for you by `sim_onnx_planner.sh --whole-body-teleop` when `MODEL` is a `_dual.onnx`) | Native dual-head graph `<name>_dual.onnx`, obs 2511 (= 680 pose + 840 smpl + 990 proprio + head select) -> 31. Must sit next to its `<name>_g1.onnx` sibling from the same `.pt` (`onnx_provenance.py check` gates the pairing) | `export_native_all.sh` (internally `native_dual_head_onnx_export.py --checkpoint <ckpt.pt> --name <name> --out-dir <dir>`) | shipped `x2_sonic_v16ft8_45000_dual.onnx` (the `_dual` sibling of the pose graph when unset) |
| `SIMSTACK_TOKEN` | `simstack_local.sh` (set by `sim_onnx_planner.sh --whole-body-teleop` when `MODEL` is a `_token.onnx`, or auto-derived from the `_g1.onnx` sibling) | Token-input graph `<name>_g1_token.onnx`, obs 1670 (64-dim motion token in the first slots, then padding, then 990 proprio) -> 31 | Native: `export_native_all.sh` (`native_token_onnx_export.py`). Frozen-G1: `gear_sonic/scripts/frozen_core_token_onnx_export.py --checkpoint <merged.pt> --output <name>_g1_token.onnx --remap-waist --release` | shipped `x2_sonic_v16ft8_45000_g1_token.onnx` (explicit opt-in: the dual-head path is the default) |
| `SIMSTACK_TOKENIZER` | `simstack_local.sh` (token service in the sim stack); auto-selected as `<name>_smpl_tokenizer.onnx` next to a native `_g1_token.onnx` | SMPL tokenizer `smpl_obs[840] -> motion_token[64]`. Must pair with the token graph: native sets ship their own; frozen-G1 sets use the release tokenizer exported from the public G1 core | Native: `export_native_all.sh` (`native_token_onnx_export.py` writes `<name>_smpl_tokenizer.onnx`). Frozen-G1: the release SMPL tokenizer (`x2_smpl_tokenizer_v11release.onnx`), exported once from the public core `$SONIC_HOME/g1/sonic_v1_1/last.pt` with `native_token_onnx_export.py --checkpoint <release.pt> --fused-smpl <fused smpl graph> --name x2_smpl_tokenizer_v11release` | shipped `x2_sonic_v16ft8_45000_smpl_tokenizer.onnx` (auto-selected next to the token graph) |
| `PLANNER_MODEL` | `sim_onnx_planner.sh`, `simstack_local.sh`, `run_x2_quest3_planner_stack.sh` (as `KPLANNER_ONNX`) | A **directory** holding `x2_planner_template.onnx` (+ `x2_planner_velocity.onnx`), or one of those files. Inputs `context_mujoco_qpos [1,4,38]`, `velocity_intent [1,4]`; output `mujoco_qpos [1,64,38]` | Own trained trio: `motionbricks/scripts/export_x2_planner_onnx.py --vqvae-ckpt .. --pose-ckpt .. --root-ckpt .. --out-dir <dir> --mode both --verify` (ship only if every parity case passes). Frozen G1 planner core with analytic Phi heads: `motionbricks/scripts/export_g1core_x2_planner_onnx.py --out-dir <dir> --mode both` | none for X2 (`download_from_hf.py` fetches the **G1** planner into `$SONIC_HOME/g1`; the launcher also looks in `$SONIC_HOME/x2/kplanner_onnx/` and `$CKPT_ROOT/planner_onnx/`) |
| `X2_RITUAL_MODEL` | `x2_pc2/robot_env.env` -> `x2_pc2/start_x2_deploy_ritual.sh` (on PC2) | PC2-side absolute path of the pose graph (`<name>_g1.onnx`) or the dual graph (`<name>_dual.onnx`) under `${PC2_PREFIX}/policies/` | Same files as `MODEL`; copied to PC2 with `x2_pc2/push_to_pc2.sh --manifest ...` (or `pc2_bringup.sh --model`) | template: `${PC2_PREFIX}/policies/x2_sonic_v16ft8_45000_dual.onnx` (staged by `pc2_bringup.sh` step 9) |
| `X2_WB_TOKEN_MODEL` | `x2_pc2/robot_env.env` (whole-body overlay on PC2) | PC2-side path of `<name>_g1_token.onnx`; must pair with `X2_RITUAL_MODEL` (same checkpoint) | as `SIMSTACK_TOKEN` | template: `${PC2_PREFIX}/policies/x2_sonic_v16ft8_45000_g1_token.onnx` |
| `X2_WB_TOKENIZER` | `x2_pc2/robot_env.env` -> `gear_sonic/scripts/pc2_pico_token_service.py` on PC2 | PC2-side path of the SMPL tokenizer that pairs with the two above | as `SIMSTACK_TOKENIZER` | template: `${PC2_PREFIX}/policies/x2_sonic_v16ft8_45000_smpl_tokenizer.onnx` |
| `SONIC_HOME` | `download_from_hf.py`, `sim_onnx_planner.sh`, `run_x2_quest3_planner_stack.sh`, `install_scripts/setup_x2.sh` | Model cache root (default `~/.cache/sonic`). `$SONIC_HOME/g1` = public G1 artifacts; `$SONIC_HOME/x2` = your X2 artifacts (`$SONIC_HOME/x2/sonic_policy/x2_sonic_policy.onnx`, `$SONIC_HOME/x2/kplanner_onnx/x2_kplanner_{template,velocity}.onnx`). `SONIC_X2_MODELS` overrides the x2 subtree alone | `python download_from_hf.py` fills `$SONIC_HOME/g1`; copy your exports into `$SONIC_HOME/x2` by hand if you want auto-resolution without `MODEL=` | `nvidia/GEAR-SONIC` (G1 only) |
| `CKPT_ROOT` (alias `X2_CHECKPOINTS_DIR` in `run_motion_replay.sh` / `simstack_local.sh`) | `sim_onnx_planner.sh`, `simstack_local.sh`, `run_motion_replay.sh` | Root the sim stack reads staged artifacts from: `$CKPT_ROOT/dances_x2m2/` (baked clips), `$CKPT_ROOT/planner_onnx/` (planner graphs), and any `*_g1.onnx` found below it. The docker sim mounts it at `/workspace/checkpoints` | Set it to `$X2_MODELS` (or any dir under `$HOME`); `run_motion_replay.sh` bakes clips into it | none |

Related knobs that are **not** models but always travel with them:

| Variable | Meaning |
|---|---|
| `KPLANNER_PROFILE` | env file of planner runtime knobs, e.g. `gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env` (teleop profile) or `gear_sonic/config/kplanner_profiles/g1core_bare_v1.env` (frozen-G1 planner) |
| `SIM_TUNING_YAML` / `X2_RITUAL_TUNING` | deploy tuning preset; default (sim and robot) `gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml` (the shipped set's preset: unity gains plus MC-stiffness waist pitch). Others: `trained_gains_s0.yaml` (unity gains, for models trained on the vendor plant), `bigrun.yaml` (legacy scaled gains). `SIM_TUNING_YAML=none` = the deploy's built-in defaults (parity runs) |
| `X2_PLANT`, `X2_ANCHOR_ORI_MODE` | actuator plant name and reference-orientation convention the model was trained with; check with `python gear_sonic/scripts/onnx_obs_contract.py <model.onnx>` |
| `ALLOW_MISMATCH=1` | acknowledge that the model you test is not the one on the robot (required for every sim-only candidate run) |
| `SONIC_X2_HF_REPO` | OPTIONAL Hugging Face repo id of further X2 artifacts (planner graphs, other policy sets); `download_from_hf.py --robot x2` and `install_scripts/setup_x2.sh` snapshot it into `$SONIC_HOME/x2`. The shipped SONIC set needs no download; unset = nothing to fetch |
| `SONIC_RELEASE_CKPT` | path of the frozen public G1 release checkpoint (`<hf cache>/models--nvidia--GEAR-SONIC/snapshots/<sha>/sonic_release/last.pt`) read by `gear_sonic/scripts/frozen_core_sonic_codec.py` and `gear_sonic/scripts/frozen_core_t2_export.py`; default = the Hugging Face cache snapshot path, set it when your cache lives elsewhere |
| `X2_EVAL_ROOT` | your local eval scratch directory (checkpoints, merged `.pt`, exports) used in the example commands of `gear_sonic/scripts/frozen_core_t2_onnx_export.py` / `frozen_core_token_onnx_export.py`; any writable dir, nothing is read from it implicitly |
| `CKPT`, `CFG`, `PICO_FT_PKL`, `PICO_SIDECAR_DIR` | inputs of `gear_sonic/scripts/cloud/stage_pico_v8.sh` (your warm-start checkpoint + its `config.yaml`, fine-tune pkl, SMPL sidecar dir); see [`docs/x2/F12_training_nebius.md`](docs/x2/F12_training_nebius.md) |
| `SIMSTACK_DIR` | session dir of `simstack_local.sh` (logs, kplanner tapes, archived sessions); default `$SONIC_HOME/simstack_pc` |
| `KPLANNER_DANCES_DIR` | directory of baked `.x2m2` clips the sim kplanner serves (`run_motion_replay.sh`, `run_x2_quest3_planner_stack.sh`); default `gear_sonic_deploy/data/motions_x2m2/demo_bank/` from `tools/build_demo_bank_from_upstream.sh` |
| `PICO_TAPES_DIR` | where `gear_sonic/scripts/pico_intent_sender.py` / `live_pico_smpl_teleop.py` record Pico tapes and intent sessions; default `<repo>/logs/pico_tapes` |
| `ISAACLAB_PYTHON` | python of the Isaac Lab training env used by `launch_bigrun.sh` / `launch_bigrun_resume.sh` and the frozen-core export scripts; default `$HOME/miniconda3/envs/env_isaaclab/bin/python` |
| `KPLANNER_PYTHON` | interpreter for the kplanner process in `run_x2_quest3_planner_stack.sh` (`--kplanner-python`); default the launcher's `PYTHON` |
| `PC2_PREFIX` | robot-side install prefix of the deploy tree (`gear_sonic/scripts/pc2_kplanner_onnx.py`, `pc2_pad_daemon.py`, the PC2 launchers); default `/home/run/gear-sonic` |
| `PC1_HOST`, `X2_MC_EM_URL` | robot PC1 address (EM HTTP API at `http://$PC1_HOST:${PC1_EM_PORT:-50080}`) or the full EM URL, read by `gear_sonic_deploy/deploy_x2.sh` and `gear_sonic_deploy/docker_x2/get_x2_sonic_ready.sh`; unset = `aima em` CLI only |
| `X2_SDK_NIC`, `X2_SDK_HOST_IP` | host NIC on the robot SDK subnet (default `enp10s0`) and the static address it must carry (`<LAPTOP_IP>/24`; empty = not checked), read by `get_x2_sonic_ready.sh` |
| `X2_BAND_FORCE_LOG_DIR` | existing directory into which `gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py` logs the assist-band force CSV; unset = no log |

## Which set does what

| Set | Files | Sim launch | Robot pointers (`x2_pc2/robot_env.env`) |
|---|---|---|---|
| Shipped default (`x2_sonic_v16ft8_45000`, native 3-encoder, F06) | `gear_sonic_deploy/models/x2_sonic_v16ft8_45000_{g1,dual,g1_token,smpl_tokenizer}.onnx` | `./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop` (no env) | the shipped `x2_pc2/robot_env.env.template` as is |
| Native 3-encoder, your own (F06) | `<name>_g1.onnx` + `<name>_dual.onnx` (+ `_g1_token`, `_smpl_tokenizer` for the legacy token path) | `MODEL=<name>_dual.onnx ... sim_onnx_planner.sh --whole-body-teleop` (the `_g1` sibling is found next to it) | `X2_RITUAL_MODEL=<name>_dual.onnx` (legacy: `_g1` + `X2_WB_TOKEN_MODEL=_g1_token` + `X2_WB_TOKENIZER=_smpl_tokenizer`) |
| Frozen G1 core + X2 LoRA (F07) | `<name>_g1.onnx` + `<name>_g1_token.onnx` + `x2_smpl_tokenizer_v11release.onnx` | `MODEL=<name>_g1.onnx SIMSTACK_TOKEN=<name>_g1_token.onnx SIMSTACK_TOKENIZER=x2_smpl_tokenizer_v11release.onnx ... sim_onnx_planner.sh --whole-body-teleop` | `X2_RITUAL_MODEL=<name>_g1.onnx`, `X2_WB_TOKEN_MODEL=<name>_g1_token.onnx`, `X2_WB_TOKENIZER=x2_smpl_tokenizer_v11release.onnx` |
| kplanner (F08) | dir with `x2_planner_template.onnx` + `x2_planner_velocity.onnx` | `PLANNER_MODEL=<dir>` | `${PC2_PREFIX}/planner_stack/models/planner_onnx/` via `x2_pc2/push_manifest_kplanner_only.txt` |

## Verifying a set before you run it

```bash
# stamps, checkpoint fingerprint and pairing of a native set
python gear_sonic/scripts/onnx_provenance.py show $X2_MODELS/<set>/exported/*.onnx
python gear_sonic/scripts/onnx_provenance.py check \
  --pose $X2_MODELS/<set>/exported/<name>_g1.onnx \
  --dual $X2_MODELS/<set>/exported/<name>_dual.onnx
# which reference-orientation convention the graph was trained with
python gear_sonic/scripts/onnx_obs_contract.py $X2_MODELS/<set>/exported/<name>_g1.onnx
```

See [`docs/x2/BUILD_CHAIN.md`](docs/x2/BUILD_CHAIN.md) for what to rebuild
when a new model lands.
