# Build chain — what rebuilds when

Four kinds of change reach the X2 stack: a new **model**, new **motion data**,
new **C++/plant** code, and new **Python/scripts**. Each has exactly one
rebuild path. `sim_onnx_planner.sh` builds nothing; it only verifies that the
pieces you are about to run agree with each other (and, with `--pc2-host`,
with the robot).

## A. New model (`.pt` checkpoint) -> ONNX set

Run locally in the Isaac Lab conda env (`ISAACLAB_PYTHON`); `config.yaml`
must sit next to the `.pt`. Outputs go to `<ckpt dir>/exported/` unless you
pass an out dir.

| Lineage | Command | Produces |
|---|---|---|
| Native 3-encoder (dual-head sets, F06) | `./gear_sonic/scripts/export_native_all.sh <ckpt.pt> <name> [out-dir]` | `<name>_g1.onnx`, `<name>_smpl.onnx`, `<name>_smpl_tokenizer.onnx`, `<name>_g1_token.onnx`, `<name>_dual.onnx`, `<name>.manifest.txt` |
| (what it runs) | `gear_sonic/scripts/dump_isaaclab_step0.py` -> `gear_sonic/scripts/reexport_x2_g1_onnx.py` per encoder (step-0 fidelity gate) -> `gear_sonic/scripts/native_token_onnx_export.py` -> `gear_sonic/scripts/native_dual_head_onnx_export.py` (parity gates A/B) -> `gear_sonic/scripts/onnx_provenance.py check` | every file stamped with the checkpoint fingerprint and git sha |
| Frozen G1 core + LoRA (F07) | `python gear_sonic/scripts/frozen_core_t2_export.py <run_dir_or_ckpt> <merged.pt>` then `python -m gear_sonic.scripts.frozen_core_t2_onnx_export --checkpoint <merged.pt> --output <name>_g1.onnx --remap-waist --remap-wrist` then `python gear_sonic/scripts/frozen_core_token_onnx_export.py --checkpoint <merged.pt> --output <name>_g1_token.onnx --remap-waist --release` | `<name>_g1.onnx` + `<name>_g1_token.onnx`; tokenizer = the shared `x2_smpl_tokenizer_v11release.onnx` |
| Legacy single fused graph | `./gear_sonic_deploy/scripts/reexport_x2_onnx.sh <run-dir> [<out.onnx>]` | one `(B,1670)->(B,31)` graph, gated on max action diff vs the step-0 dump |
| kplanner graphs | `python motionbricks/scripts/export_x2_planner_onnx.py --vqvae-ckpt .. --pose-ckpt .. --root-ckpt .. --out-dir <dir> --mode both --verify` or `python motionbricks/scripts/export_g1core_x2_planner_onnx.py --out-dir <dir> --mode both` | `x2_planner_template.onnx` + `x2_planner_velocity.onnx`, consumed from `PLANNER_MODEL=<dir>`, `$CKPT_ROOT/planner_onnx/` or `$SONIC_HOME/x2/kplanner_onnx/` |

After exporting, check the set (`onnx_provenance.py show` / `check`,
`onnx_obs_contract.py`) as described in [`../../MODELS.md`](../../MODELS.md).
A native set is only valid as a whole: the launchers refuse a `_dual` /
`_g1` / `_token` / `_tokenizer` mix cut from different checkpoints.

## B. Motion data -> deploy format

| Step | Command | Output |
|---|---|---|
| Build the motion banks (once per clone): gesture bank from the shipped X2 MC stock gestures, synthesized planner primitives; `--with-upstream-examples` adds the upstream G1 reference clips | `bash tools/build_demo_bank_from_upstream.sh` | `gear_sonic/data/motions/x2_demo_bank.pkl`, `gear_sonic/data/motions/x2_planner_primitives.pkl`, `gear_sonic/data/motions/x2_pad_banks.pkl`, `gear_sonic_deploy/data/motions_x2m2/demo_bank/*.x2m2` |
| Bake one clip window for the clip player | `python gear_sonic/scripts/pkl_to_x2m2.py --pkl <clip.pkl> --key <key> --start-s S --dur-s D --out $CKPT_ROOT/dances_x2m2/<key>.x2m2` (done for you by `run_motion_replay.sh`) | `.x2m2` in `$CKPT_ROOT/dances_x2m2/` |
| Convert a pkl for `deploy_x2.sh --input-type motion_file` | `python gear_sonic_deploy/scripts/export_motion_for_deploy.py --in <clip.pkl or playlist.yaml> --out <clip.x2m2>` | `.x2m2` |
| Watchdog idle fallback | `python gear_sonic_deploy/scripts/bake_idle_stand_x2m2.py` (from `x2_planner_primitives.pkl`; `pc2_bringup.sh` also bakes it on PC2) | `gear_sonic_deploy/data/idle_stand.x2m2` |
| Planner primitives from recipes | `python gear_sonic/scripts/build_x2_planner_primitives.py --source <bank.pkl>` (recipes in `gear_sonic/data/motions/x2_planner_primitives_recipes.yaml`) | `gear_sonic/data/motions/x2_planner_primitives.pkl` |

## C. New C++ or plant code -> deploy binary

The plant (gains, action scales, default pose) is **compile-time**:
`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp`.
A model exported against a different plant than the binary was built with
runs, silently, with wrong action scales. Rebuild the binary whenever the
plant yaml (`gear_sonic/config/robot_plant/*.yaml`) or any `.cpp/.hpp` changes,
and check `X2_PLANT` in `x2_pc2/robot_env.env` matches.

| Target | Command | Notes |
|---|---|---|
| Sim (x86, docker) | `./gear_sonic/scripts/simstack_local.sh` -> `gear_sonic_deploy/deploy_x2.sh sim` runs `colcon build --packages-select agi_x2_deploy_onnx_ref --base-paths src/x2/agi_x2_deploy_onnx_ref -DONNXRUNTIME_ROOT=/opt/onnxruntime` inside the `x2sim` container on **every launch** unless `--no-build` | `gear_sonic_deploy/docker_x2/enter_sim.sh -- bash /workspace/sonic/gear_sonic_deploy/docker_x2/build_deploy_pkg.sh` is the standalone one-shot build (F11) |
| Robot (aarch64, PC2) | `./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host <PC2_IP>` rsyncs `src/x2/agi_x2_deploy_onnx_ref` + `src/common` and runs colcon **on PC2** (`--skip-build` to only refresh sources, `--force-build` for a clean cache) | `deploy_x2.sh onbot` never builds; it only checks `${PC2_PREFIX}/ws/install/setup.bash` exists |
| Regenerate the plant header | `python gear_sonic_deploy/scripts/codegen_x2_policy_parameters.py` / `python gear_sonic_deploy/scripts/gen_policy_parameters.py` from the plant yaml, then rebuild both targets | `python gear_sonic/scripts/check_plant_consistency.py` refuses a sim launch when the MJCF, plant yaml and header disagree |

## D. Python / script changes -> robot

`x2_pc2/push_to_pc2.sh --manifest x2_pc2/<manifest>.txt --pc2 run@<PC2_IP>` is
the only sanctioned path (no bare scp). Gates, in order: `preflight_planner.py`
PASS -> every manifest file committed -> PC2 backup with md5 manifest -> md5
verify both sides. Files only; C++ goes through `pc2_bringup.sh` (C above).
The push never restarts anything on PC2 — the operator does that at the robot.

| Manifest | Ships |
|---|---|
| `x2_pc2/push_manifest.txt` | the planner stack, daemons and ritual scripts |
| `x2_pc2/push_manifest_kplanner_only.txt` | planner runtime + graphs only |
| `x2_pc2/push_manifest_ritual_only.txt` | ritual scripts + `robot_env.env` |
| `x2_pc2/push_manifest_incumbent.txt` | a native pose graph + planner into the default slots |
| `x2_pc2/push_manifest_wholebody_dual.txt` | a native dual-head set + token service |

`x2_pc2/pc2_live_manifest.sh run@<PC2_IP>` prints md5 + mtime of every file the
robot's rituals actually launch, next to the laptop copy — run it after a push.

## E. What `sim_onnx_planner.sh` checks at launch (no build)

1. Resolves the model set from `MODEL=` (default: the shipped
   `gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*` set; auto-derives the `_token` / `_dual` /
   `_g1` / `_smpl_tokenizer` siblings) and, in `simstack_local.sh`, runs
   `onnx_provenance.py check` on the pairing before the deploy starts.
2. With `--pc2-host <PC2_IP>`: read-only ssh md5 of the robot's sonic ONNX,
   both planner graphs and `pc2_kplanner_onnx.py`; deploy **binary mtime vs
   model mtime**; local `*.cpp/*.hpp` newer than the robot build; ritual,
   ignition, tuning and env-file drift. Any mismatch exits 6 unless
   `ALLOW_MISMATCH=1`.
3. Without `--pc2-host`: sim-only, results are candidate-only.
4. Then `simstack_local.sh` -> `deploy_x2.sh sim` in `docker_x2` (the colcon
   build in C happens here), `check_plant_consistency.py`, the ONNX kplanner,
   the pose watchdog and the pad bridge.

## Regenerated artifacts (not in git)

| Artifact | Used by | How you get it |
|---|---|---|
| `gear_sonic/data/motions/x2_planner_primitives.pkl` | `pc2_kplanner_onnx.py`, `bake_idle_stand_x2m2.py`, `x2_pc2_daemons.sh` | `tools/build_demo_bank_from_upstream.sh` |
| `gear_sonic/data/motions/x2_demo_bank.pkl` | `sim_onnx_planner.sh`, `play_x2_motion_mujoco.py`, `run_motion_replay.sh` | `tools/build_demo_bank_from_upstream.sh` |
| `gear_sonic/data/motions/x2_pad_banks.pkl` | pad clip banks (`run_x2_quest3_planner_stack.sh`) | `tools/build_demo_bank_from_upstream.sh` |
| `gear_sonic_deploy/data/motions_x2m2/demo_bank/*.x2m2`, `$CKPT_ROOT/dances_x2m2/*.x2m2` | `simstack_local.sh --dances-dir`, `run_motion_replay.sh` | `tools/build_demo_bank_from_upstream.sh`; `pkl_to_x2m2.py` |
| `gear_sonic_deploy/data/idle_stand.x2m2` | `x2_pose_watchdog.py` fallback | `bake_idle_stand_x2m2.py` (also by `pc2_bringup.sh`) |
| `gear_sonic/data/motions/x2_pico_chores/tapes/*.npz` | `run_pico_replay.sh`, `run_x2_pico_wbc.sh --tape-replay` | shipped (git-lfs, first-party Pico chores session); new tapes are written by every live Pico session (F03) |
| `data/operator_calibrations/<id>.yaml` | `quest3_manager_x2.py --calibration` | `python -m gear_sonic.scripts.vr_operator_calibrate --output data/operator_calibrations/<id>.yaml --operator-id <id>` |
| `data/body_models/smplx/*.npz`, `data/human_joints_info.pkl` | training (`gear_sonic/trl/utils/smplx`) | `python download_from_hf.py --training` (SMPL-X per the upstream docs) |
| `gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*.onnx` (shipped SONIC set: `_g1`, `_dual`, `_g1_token`, `_smpl_tokenizer`) | every launcher, `x2_pc2/robot_env.env.template` | shipped (git-lfs, `git lfs pull`); re-export with `export_native_all.sh` (chain A) for a new checkpoint |
| `$X2_MODELS/**/*.onnx` (other sets, kplanner planner graphs) | every launcher | export chain A above; see `MODELS.md` |
| `motionbricks/out/X2-clip.ckpt` (torch-kplanner mode-template clips) | `x2_kplanner.py` torch path, `export_x2_planner_onnx.py --clip-ckpt` | `python motionbricks/scripts/build_x2_planner_clips.py` (from the demo bank); the ONNX kplanner path does not need it |
