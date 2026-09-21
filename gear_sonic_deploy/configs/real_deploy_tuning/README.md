# Real-Deploy Tuning Configs

YAML presets for the post-policy tuning knobs `deploy_x2.sh` exposes when
running on a real X2 Ultra. Pulled into the deploy via:

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/conservative.yaml \
    --model ...   --motion ...
```

Each YAML key maps to a single `x2_deploy_onnx_ref` CLI flag. The
translator at `gear_sonic_deploy/scripts/tuning_config_to_args.py` reads the
file and emits the corresponding flags; `deploy_x2.sh` prepends them to its
arg list so anything the operator passes explicitly on the command line
still overrides the preset.

> **How were the numbers in `expressive.yaml` actually picked?**
> By scanning the motion controller for its per-joint kp/kd, running
> nudge tests, reading the oscillation summary, and translating each
> finding into a YAML knob. The condensed procedure is the "Tuning
> workflow at a glance" section below; the per-knob rationale is in the
> comments of each preset.

## Parity rule (read this first)

`--tuning-config` is **rejected in `sim` mode**.

Sim profiles (`parity`, `handoff`, `gantry`, `gantry-dangle`) exist to
guarantee bit-for-bit equivalence between the C++ deploy binary and the
Python reference (`gear_sonic/scripts/eval_x2_mujoco.py`) running in
MuJoCo. Allowing a tuning config to silently change safety clamps or
filtering inside a sim run would erode that parity surface. The wrapper
exits with an error if you try.

If you want to *test* a real-deploy preset's effect in MuJoCo, do it via
explicit CLI flags (`--max-target-dev`, `--target-lpf-hz`, `--kp-scale-*`,
etc.) and accept that the resulting trajectory **will not** match
`eval_x2_mujoco.py`.

The post-policy filters that ship today (e.g. `target_lpf_hz`) are
*architecturally* parity-safe even on the real robot:

* The policy receives the same observation it always has -- nothing the
  YAML can set is wired into the obs builder.
* `--obs-dump` returns from the OnControl tick **before** the LPF runs,
  so `compare_deploy_vs_python_obs.py` is bit-identical with the filter
  on or off.
* RAMP_OUT and SAFE_HOLD bypass the LPF -- those states already produce
  a deliberately shaped trajectory we don't want to attenuate.

## What the schema covers

There are five families of knobs the YAML can set today. Each maps to a
matching CLI flag on the deploy binary; `_schema.yaml` is the
authoritative listing.

| Family | Keys | Purpose |
|---|---|---|
| Action / target clipping | `action_clip`, `max_target_dev`, `max_target_dev_{leg,waist,arm,head}` | Hard ceilings on what the policy can ask the actuators to do. The per-group `max_target_dev_*` win over the global `max_target_dev` for joints in their group. |
| Soft-start / safety | `ramp_seconds`, `return_seconds`, `tilt_cos` | Alpha-blend ramp into / out of policy control, plus the tilt watchdog threshold. |
| **PD trim (kp)** | `kp_scale`, `kp_scale_{hip,knee,ankle,waist,shoulder,elbow,wrist,head}`, `kp_scale_ankle_{pitch,roll}`, `kp_scale_waist_{yaw,pr,pitch,roll}` | Multiplicative scale on the trained `kps[i]` per joint family. Defaults to 1.0; G16b-validated default is `kp_scale_ankle = 1.5`. The `_pitch` / `_roll` / `_yaw` / `_pr` knobs let you match MC's asymmetric per-axis PD exactly. `kp_scale_waist_pr` is the legacy combined knob (applies to BOTH waist_pitch and waist_roll); the per-axis `kp_scale_waist_pitch` and `kp_scale_waist_roll` compose multiplicatively on top, defaulting to 1.0, so existing presets keep working unchanged. |
| **PD trim (kd)** | `kd_scale`, `kd_scale_{hip,knee,ankle,waist,shoulder,elbow,wrist,head}`, `kd_scale_ankle_{pitch,roll}`, `kd_scale_waist_{yaw,pr,pitch,roll}` | Multiplicative scale on the trained `kds[i]` per joint family. KD bumps matter more than KP bumps for nudge rejection — the trained policy is critically-damped against IsaacLab's implicit integrator, which leaves it under-damped on real hardware. The waist pitch/roll split (2026-05-30) targets the asymmetric MC-vs-SONIC nudge response: pitch is +45% over MC, roll is -72% under MC, at numerically identical PD. The combined `kd_scale_waist_pr` cannot close the lateral gap without breaking the sagittal axis. |
| Post-policy filters | `target_lpf_hz`, `target_lpf_hz_{leg,waist,arm,head}` | First-order EMA cutoff on the published joint targets. The global value is the default for all joints; the four per-group keys override the global on their slice (same inherit/disable/explicit convention as `max_target_dev_*`). Use the per-group split when one group needs more aggressive smoothing than another — e.g. SONIC's waist shows a 2.0-2.5 Hz resonance under active walking that benefits from `target_lpf_hz_waist = 2-3` while the legs need their full 8 Hz bandwidth for heel-strike correction. |

For the full schema with field-by-field comments and CLI flag mapping,
read [`_schema.yaml`](_schema.yaml). For the rationale behind every
shipped default, read the inline comments in
[`expressive.yaml`](expressive.yaml).

## Shipped presets

The **shipped incumbent default** is `trained_gains_s0_inc_waistmc.yaml`: it
is what `x2_pc2/robot_env.env.template` sets as `X2_RITUAL_TUNING` and what
the sim launchers (`sim_onnx_planner.sh`, `simstack_local.sh`) use as
`SIM_TUNING_YAML` / `X2_RITUAL_TUNING` when unset, paired with the shipped
native dual-head set `gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*`.

| Preset | Use when | Highlights |
|---|---|---|
| `trained_gains_s0_inc_waistmc.yaml` | **Shipped default.** The native dual-head set (models trained on the vendor plant) in whole-body and pad sessions. | `trained_gains_s0.yaml` plus waist PITCH at the vendor MC's stiffness: `kp_scale_waist_pr 2.00 x kp_scale_waist_pitch 1.62 = 3.24` (= MC 40 N*m/rad), `kd_scale_waist_pr 5.51 x kd_scale_waist_pitch 1.27 = 7.0` (damping ratio kept); roll, hips, knees, ankles untouched. Roughly halves the waist-on-stop events under crouch / lean pieces in the sim battery; stiffening roll as well rang at 4 Hz in idle, so it is pitch-only. Watch actuator heat on long sessions (the waist now holds ~24 N*m through every lean). |
| `trained_gains_s0.yaml` | A vendor-plant (S0+) model without the waist-pitch stiffening: the graduation test for a mature checkpoint, or the A/B baseline for the shipped default. | Every kp/kd scale 1.0 except the waist training-wheels (`kp_scale_waist_pr 2.00`, `kd_scale_waist_pr 5.51`); bigrun's clamps / LPF / ramps otherwise. |
| `conservative.yaml` | First powered run with a new checkpoint OR new motion playlist. | `max_target_dev = 0.30` (uniform global clamp), `target_lpf_hz = 0`, all PD scales = 1.0 (trained defaults). The "if the policy is going to misbehave, the tight clamp catches it first" preset. |
| `expressive.yaml` | After conservative passes; teleop / VR / standing gestures where the operator needs full arm reach without leg torque spikes, AND the operator has done at least one nudge-test pass. | Per-group clamp: `leg = 0.50`, `waist = 0.30`, `arm = 1.50`, `head = 0.50`; `action_clip = 20`; `target_lpf_hz = 8`; full per-subgroup MC-match PD trim (split ankle pitch/roll, split waist yaw vs pitch/roll). Every constant has an inline comment explaining what scan it derived from and which nudge test produced the deviation from MC-match. The sagittal KD bumps (`kd_scale_ankle_pitch=5.0`, `kd_scale_knee=2.0`, `kd_scale_hip=2.0`) are tuned for STATIC nudge rejection; for active walking + forward push recovery, see `walking_recovery.yaml`. |
| `walking_recovery.yaml` | Active walking sessions where SONIC needs to initiate forward strides AND recover from forward chest-nudges. Use after `expressive.yaml` has demonstrated stable standing but operators observe "stiff sagittal, can't recover forward / can't stride out" OR 1-2 Hz rocking sway in idle stand. | **Phase 1b (current):** sagittal KD scales at FULL MC-match: `kd_scale_ankle_pitch = 3.31` (MC-match, was 5.00 in expressive), `kd_scale_knee = 0.792` (MC-match kd_eff = 5.0, was 2.00 in expressive), `kd_scale_hip = 0.476` (MC-match kd_eff = 3.0, was 2.00 in expressive). Phase 1 left hip+knee at trained baseline (1.00) which was still 2.1x and 1.26x over MC; the over-damping forced the policy into ankle-only ("ankle strategy") balance and the ankle pendulumed the upper body at 1-2 Hz = visible rocking sway. Phase 1b restores hip and knee compliance so the policy can use hip-strategy balance and forward push recovery again. Lateral chain unchanged so frontal-plane stability is preserved -- but note `kd_scale_hip` drops hip_roll + hip_yaw damping too, so side-to-side will feel less stiff. Watch for a hip-yaw shimmy at heel-strike. |
| `walking_recovery_loose.yaml` | Same as `walking_recovery.yaml` but for a MORE strongly-trained policy where the conservative Phase-1b clamps pinch the recovery stride / expressiveness. Validated 2026-07-14 on the real robot: recovers pushes in all directions with decisive hip-strategy steps (leg vel ~7-8 rad/s), within torque budget. | Deltas vs `walking_recovery.yaml` (all PD scales identical): `max_target_dev_leg 0.50 -> 0.70` (bigger stride / more push-recovery torque headroom), `max_target_dev_waist 0.30 -> 0.45` (mostly frees roll; pitch stays joint-limited), `target_lpf_hz_waist 16.0 -> 0.0` (DISABLED -- let the closed-loop policy handle the waist; removes command-path phase lag, the suspected ~2 Hz forward-wobble contributor). Trade-off: re-exposes >25 Hz waist command jitter; if it appears, walk the waist LPF back toward 16. `action_clip 20` + `tilt_cos -0.3` kept (training/safety floors, not weak-model clamps). |
| `walk_101.yaml` | A strong SONIC fine-tune that can play big dance moves and needs to FLEX -- when `walking_recovery_loose.yaml`'s clamps truncate dance excursions (kicks, single-leg stance, mambo pivots, 270/360 turns) or arm/head choreography feels mushy. **FIRST-PASS / UNVALIDATED** -- run `conservative` -> `walking_recovery_loose` on a fresh checkpoint first, then verify walk_101 on a familiar dance before pushing further. | Dance-FLEX (Phase 3). Deltas vs `walking_recovery_loose.yaml` (all PD scales identical, MC-match): `max_target_dev_leg 0.70 -> 1.00`, `max_target_dev_waist 0.45 -> 0.60`, `max_target_dev_arm 1.50 -> 2.00`, `max_target_dev_head 0.50 -> 0.70` (open the target clamps -- the PRIMARY flex knob AND the only divergence guard, so loosened not disabled), `target_lpf_hz_arm/_head 8.0 -> 16.0` (snappier arm/head; trade-off re-exposes >25 Hz command jitter). `action_clip 20` kept (training floor; a stronger model does NOT change it). PD left at MC-match on purpose: lowering kd re-introduces ankle resonance, lowering kp loses pose-tracking authority -- neither is the flex knob. |

## Adding a new preset

1. Copy `conservative.yaml` to `<your_preset>.yaml`.
2. Update the `description:` block so the next operator can tell at a
   glance why this preset exists.
3. Tweak knobs. Run
   `python3 gear_sonic_deploy/scripts/tuning_config_to_args.py
   <your_preset>.yaml --validate` to confirm the file parses.
4. Use it: `deploy_x2.sh local --tuning-config <your_preset>.yaml ...`.

If you need a knob that isn't in the schema, see "How to add a knob"
below. The translator rejects unknown keys explicitly so a typo in a
preset surfaces as a launch-time error rather than as silently-ignored
config.

## How presets compose with explicit CLI flags

```bash
deploy_x2.sh local \
    --tuning-config configs/real_deploy_tuning/expressive.yaml \
    --max-target-dev 0.50 \
    --kd-scale-waist-pr 4.5 \
    ...
```

Precedence rule: **last flag on the binary's arg list wins**, and
`deploy_x2.sh` prepends the YAML's flags before the operator's, so:

- The YAML's `max_target_dev: 0.30` is overridden by the explicit
  `--max-target-dev 0.50` on the command line for the GLOBAL default.
- The YAML's per-group `max_target_dev_arm: 1.50` STILL wins for arm
  joints, because per-group beats global at clamp-synthesis time. To
  override a per-group value, pass the matching `--max-target-dev-arm`
  on the command line.
- The YAML's `kd_scale_waist_pr: 5.51` is overridden by the explicit
  `--kd-scale-waist-pr 4.5`.

This is the intended pattern for quick A/B sweeps off a known-good
preset: keep the YAML pinned, add one or two CLI overrides per run,
and revert by removing the override (no YAML edit needed).

## Tuning workflow at a glance

The 60-second summary. The recorder is `scripts/x2_motor_monitor.py` (runs
on PC2; `x2_pc2/start_x2_motor_monitor.sh` launches it in the rituals): it
subscribes to `/aima/hal/joint/{leg,waist,arm,head}/{state,command}`, writes
one JSONL row per joint per sample (`--jsonl`, daily-rotated) with the MC's
published kp / kd / target next to the measured position, and publishes a
1 Hz summary (tracking-error spikes, soft-limit proximity, staleness) on
ZMQ :5567.

```bash
# 1. Capture MC's stock per-joint PD baseline (~30 s; nudge during the
#    window) with the MC in STAND_DEFAULT and the policy NOT running.
python3 gear_sonic_deploy/scripts/x2_motor_monitor.py --jsonl /tmp/mc_baseline.jsonl --no-rotate
#    Read the per-joint kp / kd off the command rows; eyeball ringing in the
#    position trace after each nudge (a damped 1-2 cycle settle is fine, a
#    sustained oscillation is not).

# 2. Pick scale factors so deployed PD = trained PD * scale = MC's PD.
#    E.g. ankle_pitch: MC kp 40 / trained kp 21.38 = kp_scale_ankle_pitch 1.87.
#    Encode in your preset YAML.

# 3. Run the policy with the preset, nudge again, record again.
./gear_sonic_deploy/deploy_x2.sh local \
    --tuning-config configs/real_deploy_tuning/<your_preset>.yaml ...
python3 gear_sonic_deploy/scripts/x2_motor_monitor.py --jsonl /tmp/mc_preset.jsonl --no-rotate

# 4. Bump KD on whichever joints still ring. Re-run. Stop when no joint
#    rings AND idle is silent (pos_p2p < 0.2 deg over 10 s).
```

## How to add a knob

1. Add a CLI flag to `x2_deploy_onnx_ref.cpp` (and wire it through
   `CliArgs` to whichever runtime config struct it lives on).
2. Add a one-line entry to the `KEY_TO_FLAG` table in
   `gear_sonic_deploy/scripts/tuning_config_to_args.py` (this is where
   the YAML key -> CLI flag mapping is enforced).
3. Add a documented entry in `_schema.yaml` so operators can discover
   it via `python3 tuning_config_to_args.py --schema`.
4. If the knob is parity-relevant (changes anything between obs builder
   and policy.run), also wire a corresponding flag into
   `eval_x2_mujoco.py` and confirm parity holds before merging.
5. Add a sentence to the family table at the top of this README.
