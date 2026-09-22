# Training notes: what changed over the X2 runs, and what we learned

A condensed, public version of the run history behind the shipped models,
kept so that anyone training on the X2 (or another new embodiment) does not
rediscover the same things. Dates are 2026. GPU-hours are training time
(iteration time x iterations x GPUs), not node billing.

## 1. Lineages of the shipped sets

Global batch was 8 x 12,288 environments for every stage except the 32-GPU
run (32 x 12,288). Corpus sizes are clip counts after retarget and
feasibility filtering.

**Native X2 line (incumbent), ending in the shipped `v16ft8_45000`**

| Stage | When | Compute | Corpus | Iterations | GPU-h |
|---|---|---|---|---|---|
| from scratch, mesh feet (bugged, never deployed) | Apr | 8 x H200 | 2,550 | 0 -> 18,350 | 294 |
| from scratch, sphere feet + root-rotation fix | May | 8 x H200 | 2,550 | 0 -> 25,000 | 442 |
| chain-matched continuation | Jun | 8 x H100 | 49,790 | 25,000 -> 1,376 (renumbered) | small |
| executed-feasible corpus | Jul | 8 x H100 | 35,974 (85.6 h) | 1,376 -> 2,000 | 26 |
| arm-dynamics / slow-walk / dance finetunes | Jul | 8 x H100 | 35,974 + overlays | <= 3,250 | ~440 |
| **32-GPU run with SMPL sidecars** | Aug 7-9 | 4 x 8 H100 | 121,607 -> 129,363 | 2,500 -> 15,050 | 942 |
| Pico finetune legs | Aug 26-29 | 8 x H100 | 129,489 + overlays | 14,000 -> 34,000 | 373 |
| vendor plant (24 N m waist) adaptation | Sep 2-3 | 8 x H100 | 124,970 + overlay | 34,000 -> 41,425 | 140 |
| Pico corpus finetune (the 39k snapshot) | Sep 5 | 8 x H100 | 125,002 + 204 clips | 39,000 -> 41,000 | 37 |
| waist penalty experiments (reverted) | Sep 6-7 | 8 x 1-GPU nodes | same | -> 44,200 | 142 |
| merged Pico corpus, equal encoders | Sep 13-14 | 8 x H100 | 125,032 | 39,000 -> 43,000 | 77 |
| waist range widened to 20 deg, all-Pico finetunes | Sep 16-17 | 8 x H100/H200 | 129,745 + 336-clip overlay | 43,200 -> **45,000** (shipped) | ~150 |

**Frozen G1-core line, ending in the shipped `s1ft16000`**

| Stage | When | Compute | Corpus | Iterations | GPU-h |
|---|---|---|---|---|---|
| fresh decoder LoRA on the frozen core (`sonic_x2_frozen_core_t2` -> `s1_fresh_vendor_waistmap`) | Aug 31 | 8 x H100 | 123,566 | 0 -> 13,385 | 190 (incl. next row) |
| teleop finetune, 290 clips at 5 % | Sep 1 | 8 x H100 | + 290 | 13,385 -> **16,000** (shipped) | |
| phases 1-5 (overlay on/off, SMPL sampling) | Sep 8-14 | 8 x H100 | 129,582 | 16,000 -> 40,000 | ~400 |

Totals: about 3,200 GPU-h on the native line, about 600 on the frozen-core
line, about 3,750 overall (roughly 790 H200-h and 2,950 H100-h). The single
largest item is the 32-GPU run at 942 GPU-h.

## 2. The reward configuration is not the paper's

`sonic_x2_leg5_pico_combined_ft.yaml`, the recipe every incumbent-line
checkpoint since July was trained with (and therefore the shipped
`v16ft8_45000`), overrides the SONIC paper's reward terms. The overrides
were tuned in July on small specialised corpora (34 to about 275 clips) and
then carried unchanged onto the 125,000-clip corpus, because they are
inherited down the Hydra chain and nobody noticed they were still there.
`sonic_x2_ultra.yaml` keeps the upstream kernels and weights.

Changed from the paper's values:

| Term | Paper | Ours | Why it was changed |
|---|---|---|---|
| `tracking_body_linvel` std | 1.0 | **0.25** | slow-walk dead-band: standing scored 0.986 of 1.0 at a 0.12 m/s target |
| `tracking_anchor_pos` weight / std | 0.5 / 0.3 | **2.0 / 0.15** | same, make world-position lag expensive |
| `tracking_anchor_ori` weight | 0.5 | **1.5** | pelvis-tilt terminations (61 % of terminations were anchor orientation) |
| `tracking_relative_body_ori` weight | 1.0 | **2.0** | same rebalance |
| `tracking_body_angvel` weight | 1.0 | **2.0** | same rebalance |
| `anti_shake_ang_vel` | -0.005 | **-0.015** | head and wrist chatter |
| `action_rate_l2` | -0.1 | **-0.08** | let the arms move at speed |
| `adaptive_lr_max` | 2e-4 | **1e-4** | gentler ceiling while warm-starting |

Added with no upstream counterpart: `tracking_arm_linvel` (arm-scoped
velocity reward), `feet_impact_vel` -1.0 and `feet_grf_overshoot` -0.4
(soft-landing penalties). Structural, no number moved but possibly the most
consequential: the tracked end-effector is the palm (`wrist_roll_link`)
instead of `wrist_yaw_link`, because the X2 wrist chain is elbow, yaw, pitch,
roll, the reverse of the G1. Reverted for the big run: `uniform_sampling_rate`
back to the paper's 0.1.

One more override lived only in the launcher of the shipped run, not in any
YAML: the waist pitch and roll joints were removed from the `joint_limit`
penalty (`++manager_env.rewards.joint_limit.params.asset_cfg.joint_names=[the
other 29 joints]`), because the penalty's soft limit (0.9 of range, 18 deg) sat
below the 20 deg the Pico references reach. Reproduce `v16ft8` with that
override; keep a launch record (resolved config, seed hash, corpus hashes,
every `++` override) next to every run directory.

**The open question.** Six of the eight changed values push tracking terms
the same way: sharper kernels and heavier weights. The plausible cost is a
dead-band at the fast end mirroring the slow one they fixed: a Gaussian
kernel at std 0.25 saturates near zero on fast motion where tracking error
is unavoidably large, and `tracking_anchor_pos` at 4x weight and 2x sharper
compounds it and also feeds a termination gate. Consistent with that (not
proof of it), the X2 is near parity with the stock G1 on the in-distribution
set but far behind on the hard tail and out of distribution, and the
frozen-core line, trained under the stock reward, beats the native line on
both (see `README_X2.md`, "How the X2 model compares"). Confounds: compute
(the G1 core had roughly ten times the GPU-hours), embodiment, the 24 N m
waist, the retarget chain.

The test that would settle it, now runnable from this repo: the same corpus
and warm start under `sonic_x2_ultra` (paper rewards) and under
`sonic_x2_leg5_pico_combined_ft` (ours), scored on the hard tail and the
out-of-distribution set. A few hours on one 8-GPU node. Caveat: the chain
above was traced by reading the YAMLs; Hydra composition order can silently
revert values, so confirm with a resolved-config dump (`preflight_train.py`
prints one) before relying on it.

## 3. Traps we fell into

- **SMPL encoder "enabled" but never sampled.** `encoder_sample_probs.smpl > 0`
  is not enough: the encoder is only sampled for clips that have a sidecar
  under `smpl_motion_file`. With the sidecar directory missing on the node,
  every clip falls back to the pose encoder while the config and the logger
  still say SMPL 1.0. Every run before the August big run trained without
  a working SMPL path this way. The shipped gates exist because of this:
  `check_smpl_sidecars.py` and `preflight_train.py` (coverage below 90 % fails),
  both run by `run_smoke_8gpu.sh` before launch, and the training log must not
  contain `0/N motions have SMPL data`.
- **Sidecar geometry bugs are silent.** One round of Pico sidecars carried
  root-local joints instead of world joints, another a wrong heading; both
  scored plausibly and were caught only by a render sign-off. Gate the
  sidecars (`check_smpl_sidecars.py`) and render a sample before training.
- **The plant was not measured before training on it.** Effort limits were
  stale against the motor datasheet, continuous and peak torque were
  confused, and the waist limit moved 48 -> 36 -> 24 N m over the programme.
  Policies trained at 32-36 N m on a waist the robot holds at 24 N m saturated
  on hardware at about 6 deg of pitch. Rule since then: measured beats
  datasheet beats borrowed; the vendor plant `x2_ultra_vendor_20260823` is the
  training and deploy plant, and `check_plant_consistency.py` runs at every
  sim launch. This cost about 280 GPU-h of re-adaptation and penalty
  experiments plus robot time.
- **Foot contact offset.** The X2 foot contact spheres sit about 8 cm above
  the sole and the retarget lifted the root, so executed steps floated. Fix:
  a root-unpinned, ground-anchored rebuild of the corpus.
- **Mesh feet and a 6-D root-rotation order bug** made the first from-scratch
  run collapse within 6 s on a natural walk at every checkpoint. Sphere feet
  and the rotation fix gave the first powered walk.
- **Joint-limit penalty versus the references.** With the waist range widened
  to 20 deg, the penalty's 0.9 soft limit (18 deg) fought references that reach
  20 deg; the fix was the launch override above, not a smoother margin term
  (tried, reverted).
- **Fine-tune rate and forgetting.** Routing 40 % of episodes to a small
  fine-tune overlay destroyed the model; 5-20 % exclusive routing with the
  overlay clips pinned resident worked. When the overlay leaves the sampler,
  the frozen-core LoRA lost half its Pico gain within 500 iterations while the
  native line kept it (its base corpus also contained the overlay clips).
  Treat a LoRA fine-tune's end checkpoint as the drop, not a later one.
- **Config lineage.** Overrides four levels down a Hydra chain outlive the
  experiments they were made for. Diff the resolved `config.yaml` against the
  warm-start checkpoint's own before every long run.

## 4. What the last finetune bought, measured

The final incumbent finetune (43,200 -> 45,000 on the all-Pico overlay) did
not change the general sets within noise (hard tail 216 / 300, in-distribution
485 / 500, out-of-distribution 60.7 %) while the Pico sets improved and real
falls on the operator's own tapes dropped (7 of 61 pieces at 44,500 versus 3
of the first 16 for the previous checkpoint). Every remaining fall was a deep
squat, a floor motion or a deep bend. On the deploy side, the waist pitch
gain preset (`trained_gains_s0_inc_waistmc.yaml`, kp x3.24 with matched kd)
cut the chores-tape event count from 67 to 25 without changing the fall
count, which is why it ships as the default preset.
