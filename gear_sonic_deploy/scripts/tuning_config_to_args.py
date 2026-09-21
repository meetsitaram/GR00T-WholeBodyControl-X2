#!/usr/bin/env python3
"""YAML real-deploy tuning config -> CLI args translator for deploy_x2.sh.

Reads a YAML file from gear_sonic_deploy/configs/real_deploy_tuning/ and
prints the corresponding ``--flag VALUE`` tokens, one per line, on stdout.
The bash wrapper consumes them via ``mapfile -t`` and prepends them to the
ROS2_ARGS list -- explicit CLI flags on the deploy_x2.sh command line still
override anything the config file sets.

PARITY RULE
-----------
This translator is invoked ONLY from deploy_x2.sh in ``local`` / ``onbot``
mode. Sim profiles deliberately bypass it so the bit-exact C++<->Python
parity surface (eval_x2_mujoco.py vs the deploy binary in MuJoCo) cannot be
silently perturbed by a tuning preset. See gear_sonic_deploy/configs/
real_deploy_tuning/README.md for the rationale.

USAGE
-----
    python3 tuning_config_to_args.py PATH                # print flags
    python3 tuning_config_to_args.py PATH --validate     # parse + exit 0
    python3 tuning_config_to_args.py --schema            # print schema doc
    python3 tuning_config_to_args.py --list-keys         # print supported
                                                          # YAML keys
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

# YAML key -> (CLI flag name, value formatter, optional validator).
# Adding a new tuning knob is a one-line change here PLUS a one-line entry
# in _schema.yaml + the matching --flag in x2_deploy_onnx_ref.cpp /
# deploy_x2.sh forwarding. Keep this table sorted by CLI flag.
def _fmt_float(v: Any) -> str:
    return f"{float(v):.6g}"


def _fmt_int(v: Any) -> str:
    return str(int(v))


def _check_nonneg(name: str, v: Any) -> None:
    if float(v) < 0:
        raise ValueError(f"{name}: expected >= 0, got {v!r}")


def _check_pos(name: str, v: Any) -> None:
    if float(v) <= 0:
        raise ValueError(
            f"{name}: PD scales must be > 0 (got {v!r}); use 1.0 for no trim"
        )


KEY_TO_FLAG: dict[str, tuple[str, Callable[[Any], str], Callable[[str, Any], None] | None]] = {
    # Existing C++ binary flags. Pulling them into the YAML lets one preset
    # capture an entire real-deploy scenario in a single file.
    "action_clip":     ("--action-clip",      _fmt_float, None),
    "max_target_dev":  ("--max-target-dev",   _fmt_float, None),
    # Per-group max_target_dev overrides (added 2026-05). Each one wins
    # over the global ``max_target_dev`` for joints in its group:
    #   max_target_dev_leg   -> MJ joints  0..11 (hip/knee/ankle, both sides)
    #   max_target_dev_waist -> MJ joints 12..14 (yaw/pitch/roll)
    #   max_target_dev_arm   -> MJ joints 15..28 (shoulder/elbow/wrist, both)
    #   max_target_dev_head  -> MJ joints 29..30 (yaw/pitch)
    # Negative / null = inherit the global.  Typical tight-legs/wide-arms
    # teleop preset: max_target_dev: 0.30, max_target_dev_arm: 1.50.
    "max_target_dev_leg":   ("--max-target-dev-leg",   _fmt_float, None),
    "max_target_dev_waist": ("--max-target-dev-waist", _fmt_float, None),
    # Per-JOINT override inside the waist group (2026-08-31): waist_pitch's
    # rated range (0.314) needs a tighter clamp than roll/yaw can live with
    # (the 0.35-group experiment truncated roll and swayed; 0.45 lets pitch
    # overreach its stop). Wins over max_target_dev_waist for MJ 13 only.
    "max_target_dev_waist_pitch": ("--max-target-dev-waist-pitch", _fmt_float, None),
    "max_target_dev_arm":   ("--max-target-dev-arm",   _fmt_float, None),
    "max_target_dev_head":  ("--max-target-dev-head",  _fmt_float, None),
    "ramp_seconds":    ("--ramp-seconds",     _fmt_float, _check_nonneg),
    "return_seconds":  ("--return-seconds",   _fmt_float, _check_nonneg),
    "tilt_cos":        ("--tilt-cos",         _fmt_float, None),
    # Deployment-time PD trim (added 2026-05). Mirrors the
    # ``--kp-scale-*`` / ``--kd-scale-*`` flags in eval_x2_mujoco.py.
    # Each value is multiplicative; default 1.0 (no trim). The G16b-
    # validated default for X2 Ultra is ``kp_scale_ankle=1.5``;
    # operators often need ``kp_scale_waist=1.5`` on the real robot
    # to clear torso wobble on nudge (no Python-sim equivalent).
    # See gear_sonic/scripts/eval_x2_mujoco.py:155-200 for the full
    # IsaacLab-implicit-PD vs MuJoCo/real-robot-explicit-PD background.
    "kp_scale":          ("--kp-scale",          _fmt_float, _check_pos),
    "kp_scale_hip":      ("--kp-scale-hip",      _fmt_float, _check_pos),
    "kp_scale_knee":     ("--kp-scale-knee",     _fmt_float, _check_pos),
    "kp_scale_ankle":       ("--kp-scale-ankle",       _fmt_float, _check_pos),
    "kp_scale_ankle_pitch": ("--kp-scale-ankle-pitch", _fmt_float, _check_pos),
    "kp_scale_ankle_roll":  ("--kp-scale-ankle-roll",  _fmt_float, _check_pos),
    "kp_scale_waist":       ("--kp-scale-waist",       _fmt_float, _check_pos),
    "kp_scale_waist_yaw":   ("--kp-scale-waist-yaw",   _fmt_float, _check_pos),
    "kp_scale_waist_pr":    ("--kp-scale-waist-pr",    _fmt_float, _check_pos),
    "kp_scale_waist_pitch": ("--kp-scale-waist-pitch", _fmt_float, _check_pos),
    "kp_scale_waist_roll":  ("--kp-scale-waist-roll",  _fmt_float, _check_pos),
    "kp_scale_shoulder":  ("--kp-scale-shoulder",  _fmt_float, _check_pos),
    "kp_scale_elbow":    ("--kp-scale-elbow",    _fmt_float, _check_pos),
    "kp_scale_wrist":    ("--kp-scale-wrist",    _fmt_float, _check_pos),
    "kp_scale_head":     ("--kp-scale-head",     _fmt_float, _check_pos),
    "kd_scale":          ("--kd-scale",          _fmt_float, _check_pos),
    "kd_scale_hip":      ("--kd-scale-hip",      _fmt_float, _check_pos),
    "kd_scale_knee":     ("--kd-scale-knee",     _fmt_float, _check_pos),
    "kd_scale_ankle":       ("--kd-scale-ankle",       _fmt_float, _check_pos),
    "kd_scale_ankle_pitch": ("--kd-scale-ankle-pitch", _fmt_float, _check_pos),
    "kd_scale_ankle_roll":  ("--kd-scale-ankle-roll",  _fmt_float, _check_pos),
    "kd_scale_waist":       ("--kd-scale-waist",       _fmt_float, _check_pos),
    "kd_scale_waist_yaw":   ("--kd-scale-waist-yaw",   _fmt_float, _check_pos),
    "kd_scale_waist_pr":    ("--kd-scale-waist-pr",    _fmt_float, _check_pos),
    "kd_scale_waist_pitch": ("--kd-scale-waist-pitch", _fmt_float, _check_pos),
    "kd_scale_waist_roll":  ("--kd-scale-waist-roll",  _fmt_float, _check_pos),
    "kd_scale_shoulder":  ("--kd-scale-shoulder",  _fmt_float, _check_pos),
    "kd_scale_elbow":    ("--kd-scale-elbow",    _fmt_float, _check_pos),
    "kd_scale_wrist":    ("--kd-scale-wrist",    _fmt_float, _check_pos),
    "kd_scale_head":     ("--kd-scale-head",     _fmt_float, _check_pos),
    # New post-policy filters (parity-safe by construction; see C++ comment
    # next to CliArgs::target_lpf_hz for why this is dump-invisible).
    # ``target_lpf_hz`` is the global default; the four per-group keys
    # below override the global on their slice with the same
    # inherit/disable/explicit convention as max_target_dev_*:
    #   negative / null -> inherit global
    #   0.0             -> disabled on this group (passthrough)
    #   > 0.0           -> use this Hz cutoff on this group
    # Negatives ARE legal here (sentinel for "inherit"), which is why these
    # rows skip ``_check_nonneg``; the C++ binary handles the sentinel.
    "target_lpf_hz":       ("--target-lpf-hz",       _fmt_float, _check_nonneg),
    "target_lpf_hz_leg":   ("--target-lpf-hz-leg",   _fmt_float, None),
    "target_lpf_hz_waist": ("--target-lpf-hz-waist", _fmt_float, None),
    "target_lpf_hz_arm":   ("--target-lpf-hz-arm",   _fmt_float, None),
    "target_lpf_hz_head":  ("--target-lpf-hz-head",  _fmt_float, None),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required to parse real-deploy tuning configs. "
            "Inside docker_x2 it ships pre-installed; on the host run "
            "`pip install pyyaml`. Original error: " + str(e)
        ) from e
    if not path.is_file():
        raise SystemExit(f"tuning config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(
            f"tuning config {path}: top-level must be a YAML mapping, "
            f"got {type(data).__name__}"
        )
    return data


def translate(path: Path) -> list[str]:
    """Read PATH, return the list of CLI tokens to forward to the binary."""
    data = _load_yaml(path)
    args: list[str] = []
    unknown: list[str] = []
    for key, value in data.items():
        # Tolerate descriptive metadata that doesn't map to a flag.
        if key in ("description", "name", "notes", "_schema_version"):
            continue
        if value is None:
            # Explicit null = "leave this knob at the binary's default".
            continue
        if key not in KEY_TO_FLAG:
            unknown.append(key)
            continue
        flag, fmt, validator = KEY_TO_FLAG[key]
        if validator is not None:
            validator(key, value)
        args.append(flag)
        args.append(fmt(value))
    if unknown:
        raise SystemExit(
            f"tuning config {path}: unknown keys {unknown!r}. "
            f"Supported keys: {sorted(KEY_TO_FLAG)} (plus 'description', "
            f"'name', 'notes', '_schema_version' which are ignored). "
            f"See gear_sonic_deploy/configs/real_deploy_tuning/_schema.yaml."
        )
    return args


def _print_schema() -> None:
    print("# Supported real-deploy tuning keys")
    print("# Each maps to a single deploy binary CLI flag.")
    print("# null / unset = use binary default (which matches sim parity).")
    print()
    for key, (flag, _fmt, _v) in sorted(KEY_TO_FLAG.items()):
        print(f"{key}: <number>   # -> {flag}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("path", nargs="?", help="Path to a YAML tuning config")
    p.add_argument("--validate", action="store_true",
                   help="Parse and validate without printing flags")
    p.add_argument("--schema", action="store_true",
                   help="Print the supported key set and exit")
    p.add_argument("--list-keys", action="store_true",
                   help="Print supported keys (one per line) and exit")
    args = p.parse_args()

    if args.schema:
        _print_schema()
        return 0
    if args.list_keys:
        for k in sorted(KEY_TO_FLAG):
            print(k)
        return 0
    if args.path is None:
        p.error("path is required (unless --schema / --list-keys)")
    tokens = translate(Path(args.path))
    if args.validate:
        return 0
    for tok in tokens:
        print(tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
