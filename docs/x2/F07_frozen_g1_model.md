# F07 — Frozen G1-core models (public core + X2 LoRA)

The frozen-G1 recipe keeps the public NVIDIA G1 SONIC core
(`nvidia/GEAR-SONIC`, `$SONIC_HOME/g1/sonic_release/last.pt`) frozen, maps X2 observations
and actions through the analytic Phi codec
(`gear_sonic/scripts/frozen_core_sonic_codec.py`) and trains a LoRA on the
decoder (`gear_sonic/trl/modules/lora.py`, `g1_lora_module.py`,
`frozen_core_g1_actor.py`). After training the LoRA is folded into a merged
checkpoint and exported to the **token path**: pose graph + token graph +
the release SMPL tokenizer.

## Prerequisites

- `python download_from_hf.py` (public G1 core in `$SONIC_HOME/g1`), and
  `python download_from_hf.py --training` for the training checkpoint + SMPL data.
- Isaac Lab conda env for training and export.

## Train (Nebius or a local GPU)

Warm-start file for the LoRA run (rank 16 by default):

```bash
python gear_sonic/scripts/make_g1_lora_warmstart.py $SONIC_HOME/g1/sonic_release/last.pt $X2_MODELS/warmstart/g1_lora_warmstart.pt 16
```

Experiment configs: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_frozen_core_t2.yaml`
(the Phi-codec LoRA recipe, base) and `sonic_x2_s1_fresh_vendor_waistmap.yaml`
(inherits it: vendor plant + wrist/waist range map + wrist tracking reward;
the recipe that produced the frozen-G1 set published on Hugging Face, see
`MODELS.md`). Launch as in [`F12_training_nebius.md`](F12_training_nebius.md) with
`EXP_NAME=sonic_x2_s1_fresh_vendor_waistmap`, e.g.
`++num_envs=12288 ++algo.config.num_learning_iterations=20000` (no
`+checkpoint`, no `+resume`: the config carries the warm start).

## Export (local, Isaac Lab env, `config.yaml` next to the checkpoint)

```bash
# 1. fold the LoRA into a merged checkpoint (verifies merged == LoRA forward < 1e-5)
python gear_sonic/scripts/frozen_core_t2_export.py $X2_MODELS/s1p4_35000/last.pt $X2_MODELS/s1p4_35000/s1p4_35000_merged.pt
# 2. pose graph (kplanner / pad slot), waist + wrist range maps baked in, parity-gated
python -m gear_sonic.scripts.frozen_core_t2_onnx_export \
    --checkpoint $X2_MODELS/s1p4_35000/s1p4_35000_merged.pt \
    --output $X2_MODELS/s1p4_35000/exported/x2_sonic_s1p4_35000_g1.onnx --remap-waist --remap-wrist
# 3. token graph (whole-body slot)
python gear_sonic/scripts/frozen_core_token_onnx_export.py \
    --checkpoint $X2_MODELS/s1p4_35000/s1p4_35000_merged.pt \
    --output $X2_MODELS/s1p4_35000/exported/x2_sonic_s1p4_35000_g1_token.onnx --remap-waist --release
```

Expected: each exporter prints its parity result (`max|onnx - torch|` under
`--max-action-diff`, default 1e-4) and refuses to write otherwise. The
tokenizer for this lineage is the release SMPL tokenizer
(`x2_smpl_tokenizer_v11release.onnx`, see [`MODELS.md`](../../MODELS.md));
keep it beside the set.

The frozen G1 **planner** core with Phi heads exports separately:
`python motionbricks/scripts/export_g1core_x2_planner_onnx.py --out-dir $X2_MODELS/kplanner_g1core --mode both`
(the directory `PLANNER_MODEL=` points at).

## Shipped set

The frozen-G1 set the robot ran (`x2_sonic_s1ft16000_g1.onnx`, its token graph
and the public release tokenizer) ships under `gear_sonic_deploy/models/`
(git-lfs; `MODELS.md` "Shipped frozen G1-core set"). The export steps above
are how to make a new one; the commands below run the shipped one.

## Run in sim (acceptance command, token path)

```bash
cd <repo> && TOKEN_SVC_OPERATOR_ROOT_LEVEL=zero ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=gear_sonic_deploy/models/x2_sonic_s1ft16000_g1.onnx \
SIMSTACK_TOKEN=gear_sonic_deploy/models/x2_sonic_s1ft16000_g1_token.onnx \
SIMSTACK_TOKENIZER=gear_sonic_deploy/models/x2_smpl_tokenizer_v11release.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

then `./gear_sonic/scripts/run_pico_teleop.sh` ([`F03_pico_teleop.md`](F03_pico_teleop.md)).
Expected: `[whole-body] TOKEN GRAPH — deploy-faithful rehearsal via simstack_local.sh`,
the token service's `tokenizer .../x2_smpl_tokenizer_v11release.onnx` line,
deploy `Loaded ONNX: .../x2_sonic_s1ft16000_g1.onnx`.

Planner-only (pad / VR) with the frozen-G1 planner profile:

```bash
cd <repo> && ALLOW_MISMATCH=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=gear_sonic_deploy/models/x2_sonic_s1ft16000_g1.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/g1core_bare_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh
```

Stop: `./gear_sonic/scripts/simstack_local.sh --stop`.

## Run on the robot

`x2_pc2/robot_env.env.frozen.template` carries the three pointers for this
lineage (`X2_RITUAL_MODEL` = pose graph, `X2_WB_TOKEN_MODEL` = token graph,
`X2_WB_TOKENIZER` = release tokenizer). Push with
`./x2_pc2/push_to_pc2.sh --manifest x2_pc2/push_manifest_incumbent.txt --pc2 run@<PC2_IP>`
after editing the manifest's local paths, then restart the ritual
([`F10_real_robot_colcon.md`](F10_real_robot_colcon.md)). Deploy default
keeps `--freeze-wrist`; the G1 core has no X2 wrist supervision.

## Troubleshooting

| Symptom | Fix |
|---|---|
| parity gate fails in `frozen_core_t2_onnx_export` | wrong `config.yaml` next to the checkpoint, or the merged file was made from a different run |
| robot / sim tracks arms but waist is weak | expected for the token path; the waist range map (`--remap-waist`) must be present in both graphs (`onnx_provenance.py show` prints `remap_waist`) |
| token service refuses the tokenizer | the tokenizer must be the release one for a frozen-G1 set (a native `_smpl_tokenizer` pairs only with its own native set) |
