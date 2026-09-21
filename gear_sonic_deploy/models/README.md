# Shipped default model set: `x2_sonic_v16ft8_45000` (native 3-encoder, dual-head)

Two sets ship here (git-lfs). The default set: the four ONNX graphs are the default the sim rituals (`sim_onnx_planner.sh --whole-body-teleop`, `simstack_local.sh`) and the robot ritual (`x2_pc2/robot_env.env.template`, `pc2_bringup.sh`) pick automatically: `_dual` (obs 2511, one SONIC session with both reference heads), `_g1` (obs 1670, pose graph / deploy primary slot), `_g1_token` (obs 1670, legacy token path) and `_smpl_tokenizer` (obs 840).
All four carry `codec_fingerprint native-3enc:52d8ca591e34a6a15f140c22aab957b8`: same checkpoint iteration 45000, cut from one `.pt`; `onnx_provenance.py check` gates the pairing. Pair with `gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml`, `X2_PLANT=vendor_20260823`, `X2_ANCHOR_ORI_MODE=body`.
`x2_sonic_v16ft8_45000.manifest.txt` lists the md5 of every graph of the export (the `_smpl.onnx` gate-only graph is not shipped). Export chain for a new checkpoint: `docs/x2/BUILD_CHAIN.md` (`export_native_all.sh`); other sets stay bring-your-own (`MODELS.md`). After a fresh clone run `git lfs pull` if these files are 130-byte pointers.

## Shipped frozen G1-core set: `x2_sonic_s1ft16000` (F07)

`x2_sonic_s1ft16000_g1.onnx` (pose graph, obs 1670), `x2_sonic_s1ft16000_g1_token.onnx` (token graph, obs 1670) and `x2_smpl_tokenizer_v11release.onnx` (the public G1 release's SMPL tokenizer, obs 840): the frozen NVIDIA G1 core with the X2 decoder LoRA folded in (`s1ft_it16000_merged.pt`, sha256 `f011417b...`). This is the set the robot ran on 2026-09-09 (six crouched walks, whole-body teleop) and is byte-identical to `tinkerbuggy/sonic-x2/demo_20260909/frozen_g1_s1ft16000/`. Pair with `trained_gains_s0.yaml`, `X2_PLANT=vendor_20260823`, the G1-core planner (HF `kplanner_g1core/`, see `MODELS.md`) and `gear_sonic/config/kplanner_profiles/g1core_bare_v1.env`; `x2_sonic_s1ft16000.manifest.txt` lists the md5s and `onnx_provenance.py check --pose ... --token ...` gates the pairing. The robot template for this lineage is `x2_pc2/robot_env.env.frozen.template`.

## Other sets (Hugging Face, public)

`https://huggingface.co/tinkerbuggy/sonic-x2` hosts the earlier sets referenced in `MODELS.md`; download with `huggingface-cli download tinkerbuggy/sonic-x2 --include '<folder>/*' --local-dir $X2_MODELS`:

| Folder | Contents | Pairs with |
|---|---|---|
| `tinkerbuggy/sonic-x2/demo_20260909/incumbent_39k/` | native 3-encoder set `x2_sonic_39000_{dual,g1,g1_token,smpl_tokenizer}.onnx` (obs 2511 / 1670 / 1670 / 840) | `MODEL=`/`X2_RITUAL_MODEL=` that `_dual` (F06); `robot_env/robot_env.env.demo_39k` |
| `tinkerbuggy/sonic-x2/demo_20260909/frozen_g1_s1ft16000/` | frozen G1-core set `x2_sonic_s1ft16000_{g1,g1_token}.onnx` + `x2_smpl_tokenizer_v11release.onnx` | token path (F07); `robot_env/robot_env.env.demo_frozen_s1ft16000` |
| `tinkerbuggy/sonic-x2/kplanner_onnx/` | `x2_kplanner_template.onnx`, `x2_kplanner_velocity.onnx` (planner graphs for `PLANNER_MODEL`) | incumbent policies, `bigrun_teleop_v1.env` profile |
| `tinkerbuggy/sonic-x2/kplanner_g1core/` | `x2_planner_frozen_g1core_v1.onnx` (+ provenance), template planners `x2_planner_template_s1d_*.onnx` | frozen G1-core policies, `g1core_bare_v1.env` profile (F08) |
| `tinkerbuggy/sonic-x2/kplanner_torch/` | torch planner checkpoints (`pose`, `root`, `vqvae`, `x2_clip.ckpt`) for `reexport_x2_onnx.sh` | planner re-export (`BUILD_CHAIN.md` C) |
| `tinkerbuggy/sonic-x2/sonic_onnx/`, `.../sonic_torch/`, `.../sonic_policy/` | frozen-G1 LoRA v1/v2 merged checkpoints and graphs, 14k native policy | F07 export chain examples |
| `tinkerbuggy/sonic-x2/training_configs/` | `sonic_x2_any2any_t2*.yaml` (the HF copy of this repo's `sonic_x2_frozen_core_t2.yaml`), `finetune_rate_schedule.yaml` as used for those runs | F12 |

## License

Licensed by NVIDIA Corporation under the NVIDIA Open Model License. The sets
in this directory are Derivative Models of the NVIDIA GEAR-SONIC release
(the frozen G1-core set folds the release core into its pose graph and ships
the release SMPL tokenizer as ONNX; the native set was trained with the
GEAR-SONIC code and recipe). The Agreement is Part 2 of the repository
[`LICENSE`](../../LICENSE); use is subject to NVIDIA's Trustworthy AI terms
(section 2.3) and the model's safety behaviour must not be circumvented
(section 2.1). Your own modifications may carry your own copyright (section
3.d).
