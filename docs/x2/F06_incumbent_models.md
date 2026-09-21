# F06 — Native 3-encoder models (dual-head export path)

A native X2 checkpoint (trained with `sonic_x2_ultra*.yaml`-style configs:
g1 + smpl + teleop encoders, X2 decoder) exports to a **set** of five graphs
cut from one `.pt` and stamped with one fingerprint. The launchers accept the
set only as a whole. The **dual-head** graph runs both reference heads (pose
for kplanner / pad, SMPL token for whole-body) in one ONNX session, which is
what the robot ritual and the sim rehearsal use.

**Shipped default:** `gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*`
(git-lfs, [`../../MODELS.md`](../../MODELS.md) "Shipped default set") is such
a set, exported with the chain below; the launchers and the robot template
pick it with no model variables at all. The commands in this page show the
same flow for a set you export yourself.

## Prerequisites

- Isaac Lab conda env (`ISAACLAB_PYTHON`) for the exporter's step-0 dump;
  `.venv` for the ONNX gates.
- A checkpoint `<run>/last.pt` with `config.yaml` next to it (from
  [`F12_training_nebius.md`](F12_training_nebius.md)).

## Export

```bash
./gear_sonic/scripts/export_native_all.sh $X2_MODELS/pico_39000/last.pt x2_sonic_39000
# -> $X2_MODELS/pico_39000/exported/x2_sonic_39000_{g1,smpl,smpl_tokenizer,g1_token,dual}.onnx + x2_sonic_39000.manifest.txt
```

Internally: `dump_isaaclab_step0.py` -> `reexport_x2_g1_onnx.py` per encoder
(step-0 fidelity check) -> `native_token_onnx_export.py` (token graph +
tokenizer) -> `native_dual_head_onnx_export.py` (parity gates A/B against
the single-head graphs) -> `onnx_provenance.py check`. Expected last lines:
`onnx_provenance: OK` and the manifest with one md5 + provenance line per
file. Any gate failure aborts the export.

Verify a set later:

```bash
python gear_sonic/scripts/onnx_provenance.py check \
  --pose $X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
  --dual $X2_MODELS/pico_39000/exported/x2_sonic_39000_dual.onnx \
  --token $X2_MODELS/pico_39000/exported/x2_sonic_39000_g1_token.onnx \
  --tokenizer $X2_MODELS/pico_39000/exported/x2_sonic_39000_smpl_tokenizer.onnx
python gear_sonic/scripts/onnx_obs_contract.py $X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx
```

## Deploy tuning for this lineage

Models trained on the vendor plant use **unity gains** plus the MC-stiffness
waist pitch: `gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml`
(the default of `SIM_TUNING_YAML` in sim and of `X2_RITUAL_TUNING` in
`x2_pc2/robot_env.env`; `trained_gains_s0.yaml` is the same preset without the
waist-pitch stiffening). `bigrun.yaml` carries the legacy scaled gains for
older lineages; do not mix.
`X2_PLANT=vendor_20260823` and `X2_ANCHOR_ORI_MODE` must match what the
checkpoint was trained with (`onnx_obs_contract.py` prints the declared
term; `body` for `motion_anchor_ori_b_*`, `heading` for `*_heading_*`).

## Run in sim (acceptance command, dual-head)

Shipped set, shipped tuning preset (the canonical command):

```bash
cd <repo> && ./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

Another checkpoint (explicit-env variant; `MODEL=` may be the `_dual` or the
`_g1` graph, the sibling is found next to it):

```bash
cd <repo> && TOKEN_SVC_OPERATOR_ROOT_LEVEL=zero ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
SIMSTACK_DUAL_MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_dual.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

then `./gear_sonic/scripts/run_pico_teleop.sh` in a second terminal
([`F03_pico_teleop.md`](F03_pico_teleop.md)). Expected:
`[whole-body] no MODEL= -> shipped default set` (bare command only),
`[whole-body] NATIVE DUAL-HEAD graph — one SONIC session, both reference heads`,
`pose sibling (deploy primary slot): .../x2_sonic_v16ft8_45000_g1.onnx`, and the
deploy's `Loaded ONNX ... kind=native_dual_head`.

Pad / planner only (no whole-body): same env without `SIMSTACK_DUAL_MODEL`
and without `--whole-body-teleop` ([`F01_gamepad.md`](F01_gamepad.md)).

Legacy token path of the same set (pose graph + token service):

```bash
cd <repo> && ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

(`MODEL=` is the pose graph; the launcher finds `x2_sonic_39000_g1_token.onnx`
and its own `x2_sonic_39000_smpl_tokenizer.onnx` next to it. For the shipped
set: `SIMSTACK_TOKEN=gear_sonic_deploy/models/x2_sonic_v16ft8_45000_g1_token.onnx`
selects the token path instead of the default dual-head.)

Stop: `./gear_sonic/scripts/simstack_local.sh --stop`.

## Run on the robot

Push the set and point the ritual at it (see
[`F10_real_robot_colcon.md`](F10_real_robot_colcon.md)):

```bash
./x2_pc2/push_to_pc2.sh --manifest x2_pc2/push_manifest_wholebody_dual.txt --pc2 run@<PC2_IP>
```

`x2_pc2/robot_env.env.template` as shipped already points at the default set:
`X2_RITUAL_MODEL=${PC2_PREFIX}/policies/x2_sonic_v16ft8_45000_dual.onnx`,
`X2_RITUAL_TUNING=${PC2_PREFIX}/gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml`
(`pc2_bringup.sh` stages the files). For your own set, edit `X2_RITUAL_MODEL`
to `${PC2_PREFIX}/policies/<name>_dual.onnx`. The ritual finds the `_g1`
sibling for the primary slot and refuses a mismatched pairing. Verify at
ignition: `Loaded ONNX ... kind=native_dual_head` and
`[plant] RUNTIME PLANT LOADED: vendor_20260823`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `dual graph needs its pose-graph sibling next to it` | keep the whole exported directory together |
| `onnx_provenance` mismatch | the graphs come from different checkpoints; re-export the set |
| step-0 fidelity check fails in the exporter | a stale dump from another checkpoint; the script dumps per encoder per checkpoint — delete `<out-dir>/.step0_dumps` and retry |
| waist pins at its clamp | tuning preset / plant mismatch (`trained_gains_s0_inc_waistmc.yaml` + `vendor_20260823` for this lineage) |
| shipped model files are ~130 bytes | git-lfs pointers: `git lfs install && git lfs pull` |
