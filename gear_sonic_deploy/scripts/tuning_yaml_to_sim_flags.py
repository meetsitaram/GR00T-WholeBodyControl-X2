#!/usr/bin/env python3
"""Convert a real_deploy_tuning/*.yaml into deploy_x2.sh SIM flags.

WHY THIS EXISTS. ``deploy_x2.sh`` REFUSES ``--tuning-config`` in sim mode, on
the reasoning that real-deploy presets mitigate hardware quirks and would break
the C++<->Python parity check. The consequence, discovered 2026-08-23, is that
``sim_onnx_planner.sh`` -- the launcher used for pre-deploy validation -- runs
the full stack on the deploy binary's BUILT-IN DEFAULTS:

    max_target_dev        -1.0  (DISABLED)   robot: 0.30
    max_target_dev_waist  -1.0  (DISABLED)   robot: 0.45
    max_target_dev_leg    -1.0  (DISABLED)   robot: 0.70
    kp_scale_waist_pr      1.0               robot: 2.00
    kd_scale_waist_pr      1.0               robot: 5.51
    target_lpf_hz          0    (off)        robot: 8.0 / leg 16

i.e. every clamp disabled and the waist at half stiffness with a fifth of the
damping. A model could therefore pass full-stack sim and still meet clamp
behaviour on hardware that sim never exercised -- which is exactly what
happened with v12_6500: quiet in sim, waist pinned at the clamp on the robot.

This script closes that gap WITHOUT weakening the parity rule: parity runs
simply don't pass a YAML, so they still mirror eval_x2_mujoco.py exactly.

FLAG ROUTING. deploy_x2.sh proxies most knobs but NOT the per-group LPFs
(--target-lpf-hz-leg/-waist/-arm/-head). Those must go through
--deploy-extra-arg, one token per element. Getting this wrong is silent: the
wrapper rejects an unknown flag, but a MISROUTED one just never reaches the
binary and the run looks fine at a softer setting than you asked for.

Usage:
    tuning_yaml_to_sim_flags.py configs/real_deploy_tuning/bigrun.yaml
    -> prints a shell-quoted flag list on one line
"""

import os
import shlex
import sys

import yaml

# ── TWO CLASSES OF KNOB ───────────────────────────────────────────────────
# Learned the hard way 2026-08-23: applying a robot preset wholesale to sim does
# NOT reproduce robot behaviour, because the preset mixes two different things.
#
#   SAFETY ENVELOPE (transfers): clamps, action_clip, LPF, ramps, tilt_cos.
#     Hardware-independent limits. Running sim WITHOUT these was the real gap --
#     every full-stack validation before today ran with clamps DISABLED.
#
#   PD SCALES (kp_scale_* / kd_scale_*): APPLY THESE TOO. First cut of this
#     script made them opt-in, on the theory they were hardware compensation.
#     That was WRONG and operator testing showed it immediately: without them
#     the sim robot's legs are weaker still and it COLLAPSED on a dance move,
#     whereas with them it merely felt soft. The numbers say why --
#
#       eval_x2_mujoco.py baseline : KP ankle 1.5, all else 1.0;  KD all 1.0
#       bigrun.yaml (the robot)    : KP knee 1.20 ankle_pitch 1.40 waist_pr 2.00
#                                    KD ankle_pitch 2.00 ankle_roll 1.50 waist_pr 5.51
#
#     The ROBOT is stiffer than either sim default, so dropping the scales moves
#     sim further from the robot, not closer.
#
#     A residual gap remains even with them applied: MuJoCo's explicit
#     ctrl-driven torque loses the ~1.3-1.5x that IsaacLab's implicit PD
#     supplies, and eval_x2_mujoco.py compensates only partially (ankle KP 1.5,
#     nothing on KD). So sim stays somewhat soft vs hardware no matter what --
#     sim can validate the ENVELOPE and the policy's commands, but not feel.
#
# SIM_TUNING_NO_PD=1 omits them, for deliberately studying gain effects.
PD_PREFIXES = ("kp_scale", "kd_scale")

# Knobs deploy_x2.sh accepts directly (yaml key -> wrapper flag).
DIRECT = {
    "action_clip":          "--action-clip",
    "max_target_dev":       "--max-target-dev",
    "max_target_dev_leg":   "--max-target-dev-leg",
    "max_target_dev_waist": "--max-target-dev-waist",
    "max_target_dev_waist_pitch": "--max-target-dev-waist-pitch",
    "max_target_dev_arm":   "--max-target-dev-arm",
    "max_target_dev_head":  "--max-target-dev-head",
    "ramp_seconds":         "--ramp-seconds",
    "return_seconds":       "--return-seconds",
    "tilt_cos":             "--tilt-cos",
    "target_lpf_hz":        "--target-lpf-hz",
    "kp_scale":             "--kp-scale",
    "kp_scale_hip":         "--kp-scale-hip",
    "kp_scale_knee":        "--kp-scale-knee",
    "kp_scale_ankle":       "--kp-scale-ankle",
    "kp_scale_ankle_pitch": "--kp-scale-ankle-pitch",
    "kp_scale_ankle_roll":  "--kp-scale-ankle-roll",
    "kp_scale_waist":       "--kp-scale-waist",
    "kp_scale_waist_yaw":   "--kp-scale-waist-yaw",
    "kp_scale_waist_pr":    "--kp-scale-waist-pr",
    "kp_scale_shoulder":    "--kp-scale-shoulder",
    "kp_scale_elbow":       "--kp-scale-elbow",
    "kp_scale_wrist":       "--kp-scale-wrist",
    "kp_scale_head":        "--kp-scale-head",
    "kd_scale":             "--kd-scale",
    "kd_scale_hip":         "--kd-scale-hip",
    "kd_scale_knee":        "--kd-scale-knee",
    "kd_scale_ankle":       "--kd-scale-ankle",
    "kd_scale_ankle_pitch": "--kd-scale-ankle-pitch",
    "kd_scale_ankle_roll":  "--kd-scale-ankle-roll",
    "kd_scale_waist":       "--kd-scale-waist",
    "kd_scale_waist_yaw":   "--kd-scale-waist-yaw",
    "kd_scale_waist_pr":    "--kd-scale-waist-pr",
    "kd_scale_shoulder":    "--kd-scale-shoulder",
    "kd_scale_elbow":       "--kd-scale-elbow",
    "kd_scale_wrist":       "--kd-scale-wrist",
    "kd_scale_head":        "--kd-scale-head",
}

# Binary-only knobs: must be tunnelled via --deploy-extra-arg (flag and value
# are SEPARATE extra-args -- each --deploy-extra-arg takes exactly one token).
VIA_EXTRA = {
    # per-joint waist gains (deploy_x2.sh proxies only waist / waist_yaw /
    # waist_pr; the binary parses these itself -- 2026-09-05 waist snap-back
    # candidates use kd_scale_waist_pitch)
    "kp_scale_waist_pitch": "--kp-scale-waist-pitch",
    "kp_scale_waist_roll":  "--kp-scale-waist-roll",
    "kd_scale_waist_pitch": "--kd-scale-waist-pitch",
    "kd_scale_waist_roll":  "--kd-scale-waist-roll",
    "target_lpf_hz_leg":   "--target-lpf-hz-leg",
    "target_lpf_hz_waist": "--target-lpf-hz-waist",
    "target_lpf_hz_arm":   "--target-lpf-hz-arm",
    "target_lpf_hz_head":  "--target-lpf-hz-head",
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as fh:
        cfg = yaml.safe_load(fh) or {}

    want_pd = os.environ.get("SIM_TUNING_NO_PD", "0") != "1"
    out, unknown, skipped_pd = [], [], []
    for key, val in cfg.items():
        if key == "description":
            continue
        if key.startswith(PD_PREFIXES) and not want_pd:
            skipped_pd.append(key)
            continue
        if key in DIRECT:
            out += [DIRECT[key], f"{val}"]
        elif key in VIA_EXTRA:
            out += ["--deploy-extra-arg", VIA_EXTRA[key],
                    "--deploy-extra-arg", f"{val}"]
        else:
            unknown.append(key)

    if skipped_pd:
        print(f"[tuning->sim] SIM_TUNING_NO_PD=1: {len(skipped_pd)} PD-scale key(s) "
              f"OMITTED. Sim runs training-parity gains, which are WEAKER than "
              f"the robot -- expect soft legs and possible collapse on dynamic "
              f"moves. This is a diagnostic mode, not pre-deploy validation.",
              file=sys.stderr)

    if unknown:
        # Loud, not fatal: a preset may legitimately carry keys the sim path
        # has no equivalent for. Silence here would mean sim quietly running a
        # different configuration than the operator believes -- the exact
        # failure this script exists to prevent.
        print(f"[tuning->sim] WARNING: {len(unknown)} key(s) not mapped and "
              f"NOT applied in sim: {', '.join(sorted(unknown))}",
              file=sys.stderr)

    print(" ".join(shlex.quote(t) for t in out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
