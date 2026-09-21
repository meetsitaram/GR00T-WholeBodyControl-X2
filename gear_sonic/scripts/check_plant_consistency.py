#!/usr/bin/env python3
"""Assert the THREE copies of the X2 plant agree — yaml, training, MJCF.

The plant truth is hand-duplicated in:
  1. config/robot_plant/<ACTIVE>.yaml     (single source; training/eval/codec load it)
  2. envs/manager_env/robots/x2_ultra.py  (loads the yaml since 2026-08-30;
                                           era branches keep frozen literals)
  3. data/assets/.../mjcf/x2_ultra.xml    (MuJoCo attributes — hand copy)

They have drifted before (2026-08-23: referee vs training, would have
mis-scaled every ONNX export silently — armature_investigation.md §6-7).
This module makes divergence a loud failure AT POINT OF USE: the dynamics
consumers (eval_x2_mujoco referee, simstack launcher) call ``verify()``
before trusting the MJCF. Also runnable standalone:

    python gear_sonic/scripts/check_plant_consistency.py
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-9
KNOWN_ESTIMATES = {0.00425}   # wrist p/r + head: vendor data pending


def verify(repo: Path | None = None, plant_name: str | None = None) -> list[str]:
    """Return a list of consistency failures (empty = all three copies agree)."""
    repo = Path(repo) if repo else REPO
    sys.path.insert(0, str(repo / "gear_sonic/envs/manager_env/robots"))
    from plant_config import load_plant
    plant = load_plant(plant_name)
    y_arm = plant["armature"] if isinstance(plant, dict) else plant.armature
    arm_map = dict(y_arm)
    yaml_vals = sorted({round(float(v), 12) for v in arm_map.values()})

    failures: list[str] = []

    # training constants (import needs carb/isaaclab, so parse the source).
    # Two accepted forms: legacy literals (era branches) and the 2026-08-30
    # yaml-lookup form `ARMATURE_PFxx = _PLANT.armature["<key>"]`.
    src = (repo / "gear_sonic/envs/manager_env/robots/x2_ultra.py").read_text()
    py = {m.group(1): float(m.group(2))
          for m in re.finditer(r"^ARMATURE_(PF\d+)\s*=\s*([0-9.e-]+)", src, re.M)}
    for m in re.finditer(r'^ARMATURE_(PF\d+)\s*=\s*_PLANT\.armature\["(\w+)"\]', src, re.M):
        fam, key = m.group(1), m.group(2)
        if key not in arm_map:
            failures.append(f"training maps ARMATURE_{fam} to missing yaml key {key!r}")
            continue
        py[fam] = float(arm_map[key])
    if not py:
        failures.append("no ARMATURE_PF* definitions found in x2_ultra.py")

    # MJCF attributes
    tree = ET.parse(repo / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml")
    mj = sorted({round(float(j.get("armature")), 12)
                 for j in tree.iter("joint") if j.get("armature")})

    for fam, val in py.items():
        if not any(abs(val - v) < TOL for v in yaml_vals):
            failures.append(f"training ARMATURE_{fam}={val} not present in plant yaml")
    for v in mj:
        if v in KNOWN_ESTIMATES:
            continue
        if not any(abs(v - yv) < TOL for yv in yaml_vals):
            failures.append(f"MJCF armature {v} not present in plant yaml")
    for fam, val in {"PF52": 0.003132992, "PF70": 0.008840000,
                     "PF90": 0.029079810}.items():
        for name, pool in (("yaml", yaml_vals), ("mjcf", mj),
                           ("training", list(py.values()))):
            if not any(abs(val - v) < TOL for v in pool):
                failures.append(f"{fam} ({val}) missing from {name}")
    return failures


def main() -> int:
    failures = verify()
    if failures:
        print("PLANT CONSISTENCY: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PLANT CONSISTENCY: OK (yaml == training == mjcf, "
          f"estimates tolerated: {sorted(KNOWN_ESTIMATES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
