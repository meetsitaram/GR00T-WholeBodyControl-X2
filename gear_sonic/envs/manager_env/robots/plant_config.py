"""Load an X2 actuator plant (armature / effort / PD) from YAML.

("Plant" = the controlled physical system — standard control-theory
term, not project jargon; here it means the actuator dynamics
parameters the policy-as-controller acts against.)

WHY THIS EXISTS. The plant used to live in three independent Python copies —
``x2_ultra.py`` (training), ``eval_x2_mujoco.py`` (referee) and, transitively,
``frozen_core_sonic_codec.py`` (ONNX action scales). They drifted: on 2026-08-23 the
referee still held pre-vendor armature while training had moved to the datasheet
values, which would have baked wrong action scales into every exported ONNX with
no error anywhere.

TWO HARD REQUIREMENTS this module satisfies:

1. **Importable without IsaacSim.** MuJoCo-only consumers have no ``carb``, so
   this reads plain YAML and imports nothing from isaaclab. (A full
   ``from x2_ultra import X2_ULTRA_ACTION_SCALE`` fails with
   ``ModuleNotFoundError: carb`` because that constant is built from live cfg
   objects.)

2. **Both plants live at once.** An A/B across the vendor change must evaluate
   the OLD model under the OLD plant and the NEW model under the NEW one —
   ``action_scale = 0.25 * effort / (armature * w^2)``, so a checkpoint scored
   under the wrong plant is silently mis-scaled by up to 2.86x. A global switch
   cannot express that; a per-call selector can.

Usage:
    from plant_config import load_plant
    p = load_plant()                      # active (vendor)
    p = load_plant("legacy_pre20260823")  # a pre-change checkpoint's plant
    p.armature_for("left_ankle_pitch_joint")     -> 0.008840
    p.action_scale_for("waist_pitch_joint")      -> 0.25 * effort / kp
"""

from __future__ import annotations

import math
import os
import pathlib
from typing import Any

import yaml

_HERE = pathlib.Path(__file__).resolve()
# repo_root/gear_sonic/envs/manager_env/robots/plant_config.py -> repo_root
_REPO = _HERE.parents[4]
PLANT_DIR = _REPO / "gear_sonic" / "config" / "robot_plant"

#: Plant used by current training/eval unless overridden. Changing this
#: rescales the action space for every run AND every re-export — see §6.
ACTIVE_PLANT = "vendor_20260823"

#: Env override, for evaluating a pre-change checkpoint without editing code:
#:     X2_PLANT=legacy_pre20260823 python gear_sonic/scripts/eval_x2_mujoco.py ...
ENV_VAR = "X2_PLANT"


class Plant:
    """One actuator plant. Keys are SUBSTRINGS of the joint name, first match
    wins, so dict ORDER MATTERS — specific before general. ankle and shoulder
    each span two motor families; never collapse them."""

    def __init__(self, data: dict[str, Any], source: pathlib.Path):
        self.name: str = data["name"]
        self.source = source
        self.armature: dict[str, float] = {k: float(v) for k, v in data["armature"].items()}
        self.effort: dict[str, float] = {k: float(v) for k, v in data["effort"].items()}
        #: joint-name substring -> default position (rad). Actions are offsets
        #: FROM these. Lives in the plant YAML because a stub import cannot
        #: reach x2_ultra's init_state (isaaclab mocked -> MagicMock), and
        #: because eval_x2_mujoco kept its own copy (a 4th duplicate).
        self.default_joint_pos: dict[str, float] = {
            k: float(v) for k, v in (data.get("default_joint_pos") or {}).items()}
        self.natural_freq: float = 2.0 * math.pi * float(data.get("natural_freq_hz", 10.0))
        self.damping_ratio: float = float(data.get("damping_ratio", 2.0))

    def _match(self, table: dict[str, float], joint: str, what: str) -> float:
        for key, val in table.items():
            if key in joint:
                return val
        raise KeyError(f"{self.name}: no {what} entry matches joint {joint!r}")

    def default_pos_for(self, joint: str) -> float:
        """Default position, or 0.0 when no key matches (most joints)."""
        for key, val in self.default_joint_pos.items():
            if key in joint:
                return val
        return 0.0

    def armature_for(self, joint: str) -> float:
        return self._match(self.armature, joint, "armature")

    def effort_for(self, joint: str) -> float:
        return self._match(self.effort, joint, "effort")

    def kp_for(self, joint: str) -> float:
        """Training-equivalent implicit-PD stiffness: armature * w^2."""
        return self.armature_for(joint) * self.natural_freq ** 2

    def kd_for(self, joint: str) -> float:
        return 2.0 * self.damping_ratio * self.armature_for(joint) * self.natural_freq

    def action_scale_for(self, joint: str) -> float:
        """0.25 * effort / kp.

        NOTE this couples a SAFETY CEILING to the NOMINAL command mapping, which
        is a known design flaw (§6): changing an effort limit silently rescales
        the action space. Preserved here because it is what every existing
        checkpoint was trained with; fixing it means giving action_scale its own
        table and retraining.
        """
        return 0.25 * self.effort_for(joint) / self.kp_for(joint)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Plant {self.name} from {self.source.name}>"


def load_plant(name: str | None = None) -> Plant:
    """Load a plant. Order: explicit arg > $X2_PLANT > the DEFAULT FILE
    ``x2_ultra.yaml`` (> legacy ACTIVE_PLANT constant if that file is absent).

    The default file IS the active plant (operator directive 2026-08-30):
    edit/swap it to change what everything loads; the dated
    ``x2_ultra_<name>.yaml`` files are immutable named versions for A/B
    (a checkpoint must be scored under the plant it trained on)."""
    name = name or os.environ.get(ENV_VAR)
    if name is None:
        default = PLANT_DIR / "x2_ultra.yaml"
        if default.exists():
            return Plant(yaml.safe_load(default.read_text()), default)
        name = ACTIVE_PLANT
    path = PLANT_DIR / f"x2_ultra_{name}.yaml"
    if not path.exists():
        avail = sorted(p.stem.replace("x2_ultra_", "") for p in PLANT_DIR.glob("x2_ultra_*.yaml"))
        raise FileNotFoundError(f"no plant {name!r} at {path}; available: {avail}")
    return Plant(yaml.safe_load(path.read_text()), path)
