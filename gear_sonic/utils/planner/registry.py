"""Bin-spec / primitive-registry I/O for the X2 heuristic planner.

Three YAML files, separated by purpose:

  - ``x2_planner_bins.yaml`` — bin spec, hand-edited.
        Source of truth for what bins exist + their tolerances.
  - ``x2_planner_primitives.yaml`` — registry, written by curator.
        One row per bin: which (source_pkl, motion_key, start_frame,
        n_frames) was selected, plus measured metrics. ``pinned: true``
        flags rows the user has hand-edited; the curator validates pins
        but does not search alternatives.
  - ``x2_planner_primitives_report.md`` — markdown report, always
        regenerated. Top-N close candidates per bin with metrics. Read
        this before hand-pinning.

The PKL artifact lives next to the registry YAML and is the actual data
the runtime loads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Bin spec (hand-edited)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinSpec:
    """One row of ``x2_planner_bins.yaml``.

    Attributes:
      name: bin identifier, e.g. ``"fwd_step_half_ft"`` or ``"lean_fwd_medium"``.
      family: ``"locomotion"`` (feet move, end-at-square) or
              ``"static_upper_body"`` (feet planted, end-at-apex) or
              ``"idle"`` (loopable rest pose) or
              ``"continuous_walk"`` (multi-stride gait, no end-at-square).
      target_xy_m: ``[fwd_m, lat_m]`` body-frame translation target (locomotion only).
      target_yaw_deg: yaw delta target in degrees (locomotion only).
      target_waist_pitch_deg: peak waist pitch target (static only).
      target_waist_yaw_deg: peak waist yaw target (static only).
      target_waist_roll_deg: peak waist roll target (static only;
          lateral lean family).
      tol_xy_m: tolerance on net XY magnitude, meters (locomotion only).
      tol_yaw_deg: tolerance on net yaw, degrees (locomotion only).
      tol_waist_deg: tolerance on dominant waist axis, degrees (static only).
      cross_axis_max_m: hard cap on cross-axis bleed, meters (locomotion only).
      pelvis_z_band_m: ``[z_min, z_max]`` allowed root-z range (all families).
      end_at_square_min: minimum end-at-square score (locomotion only).
      end_at_apex_min: minimum end-at-apex score (static only).
      feet_planted_min: minimum feet-planted score (static only).
      stride_count_target: preferred stride count (soft; None = unconstrained).
      stride_count_min: hard minimum stride count (hard gate; None = no floor).
          When set, candidates with fewer detected strides are dropped before
          scoring. Use this for ``*_step_*`` and ``side_*`` bins where a
          body-sway clip with no actual foot-lift is useless to SONIC.
      freeze_arms_to_default: if True, the curator overwrites the arm + head
          DOFs of the curated slice with ``DEFAULT_STAND_POSE_MUJOCO_RAD``
          before writing the primitives PKL. Use this for ``static_upper_body``
          bins so the planner only commands the waist while VLA/teleop owns
          the arms downstream.
      window_frames_min: minimum window length in frames at source fps.
      window_frames_max: maximum window length in frames at source fps.
      name_regex: optional regex over motion_key for first-pass filtering.
      target_intent: command-side ``intent`` mapped to this bin (e.g. "fwd_step").
      target_magnitude: command-side ``magnitude`` (e.g. "half_ft").
    """

    name: str
    family: str
    target_intent: str
    target_magnitude: str
    target_xy_m: tuple[float, float] = (0.0, 0.0)
    target_yaw_deg: float = 0.0
    target_waist_pitch_deg: float = 0.0
    target_waist_yaw_deg: float = 0.0
    target_waist_roll_deg: float = 0.0
    tol_xy_m: float = 0.05
    tol_yaw_deg: float = 5.0
    tol_waist_deg: float = 5.0
    cross_axis_max_m: float = 0.05
    pelvis_z_band_m: tuple[float, float] = (0.60, 0.78)
    end_at_square_min: float = 0.0
    end_at_apex_min: float = 0.0
    feet_planted_min: float = 0.0
    stride_count_target: int | None = None
    stride_count_min: int | None = None
    freeze_arms_to_default: bool = False
    window_frames_min: int = 30
    window_frames_max: int = 90
    name_regex: str | None = None

    def is_locomotion(self) -> bool:
        return self.family in ("locomotion", "continuous_walk", "idle")

    def is_static(self) -> bool:
        return self.family == "static_upper_body"


def _as_tuple_float(v: Any, n: int, default: tuple) -> tuple:
    if v is None:
        return default
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ValueError(f"expected list of length {n}, got {v!r}")
    return tuple(float(x) for x in v)


def load_bin_specs(path: Path) -> dict[str, BinSpec]:
    """Parse ``x2_planner_bins.yaml`` into ``{bin_name: BinSpec}``.

    Schema::

        bins:
          - name: idle_stand
            family: idle
            target_intent: idle
            target_magnitude: default
            ...

    Unknown fields are tolerated (forward-compat). Missing required fields
    raise ``ValueError`` with the offending bin name.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "bins" not in raw:
        raise ValueError(f"{path}: top-level dict must have a 'bins' list")
    out: dict[str, BinSpec] = {}
    for entry in raw["bins"]:
        try:
            spec = BinSpec(
                name=str(entry["name"]),
                family=str(entry["family"]),
                target_intent=str(entry["target_intent"]),
                target_magnitude=str(entry.get("target_magnitude", "default")),
                target_xy_m=_as_tuple_float(entry.get("target_xy_m"), 2, (0.0, 0.0)),
                target_yaw_deg=float(entry.get("target_yaw_deg", 0.0)),
                target_waist_pitch_deg=float(entry.get("target_waist_pitch_deg", 0.0)),
                target_waist_yaw_deg=float(entry.get("target_waist_yaw_deg", 0.0)),
                target_waist_roll_deg=float(entry.get("target_waist_roll_deg", 0.0)),
                tol_xy_m=float(entry.get("tol_xy_m", 0.05)),
                tol_yaw_deg=float(entry.get("tol_yaw_deg", 5.0)),
                tol_waist_deg=float(entry.get("tol_waist_deg", 5.0)),
                cross_axis_max_m=float(entry.get("cross_axis_max_m", 0.05)),
                pelvis_z_band_m=_as_tuple_float(
                    entry.get("pelvis_z_band_m"), 2, (0.60, 0.78)
                ),
                end_at_square_min=float(entry.get("end_at_square_min", 0.0)),
                end_at_apex_min=float(entry.get("end_at_apex_min", 0.0)),
                feet_planted_min=float(entry.get("feet_planted_min", 0.0)),
                stride_count_target=(
                    int(entry["stride_count_target"])
                    if entry.get("stride_count_target") is not None
                    else None
                ),
                stride_count_min=(
                    int(entry["stride_count_min"])
                    if entry.get("stride_count_min") is not None
                    else None
                ),
                freeze_arms_to_default=bool(
                    entry.get("freeze_arms_to_default", False)
                ),
                window_frames_min=int(entry.get("window_frames_min", 30)),
                window_frames_max=int(entry.get("window_frames_max", 90)),
                name_regex=(
                    str(entry["name_regex"])
                    if entry.get("name_regex")
                    else None
                ),
            )
        except KeyError as exc:
            raise ValueError(
                f"{path}: bin entry {entry!r} missing required field {exc}"
            ) from exc
        if spec.name in out:
            raise ValueError(f"{path}: duplicate bin name {spec.name!r}")
        out[spec.name] = spec
    return out


# ---------------------------------------------------------------------------
# Primitive registry (curator-written, optionally hand-edited)
# ---------------------------------------------------------------------------


@dataclass
class PrimitiveEntry:
    """One row of ``x2_planner_primitives.yaml``.

    Curator-written fields update on each run (unless ``pinned: true``).
    Measured fields are recomputed every run from the resolved clip.
    """

    bin_name: str
    source_pkl: str  # repo-relative or absolute path
    motion_key: str
    start_frame: int
    n_frames: int
    fps: float
    # measured
    measured_xy_m: tuple[float, float] = (0.0, 0.0)
    measured_yaw_deg: float = 0.0
    measured_waist_pitch_deg: float = 0.0
    measured_waist_yaw_deg: float = 0.0
    end_at_square_score: float = 0.0
    end_at_apex_score: float = 0.0
    feet_planted_score: float = 0.0
    pelvis_z_min_m: float = 0.0
    pelvis_z_max_m: float = 0.0
    stride_count: int = 0
    # housekeeping
    partial: bool = False
    pinned: bool = False
    notes: str = ""


def load_primitive_registry(path: Path) -> dict[str, PrimitiveEntry]:
    """Parse ``x2_planner_primitives.yaml``. Missing file => empty dict."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict) or "primitives" not in raw:
        raise ValueError(f"{path}: top-level dict must have a 'primitives' list")
    out: dict[str, PrimitiveEntry] = {}
    for entry in raw["primitives"]:
        bin_name = str(entry["bin_name"])
        out[bin_name] = PrimitiveEntry(
            bin_name=bin_name,
            source_pkl=str(entry["source_pkl"]),
            motion_key=str(entry["motion_key"]),
            start_frame=int(entry["start_frame"]),
            n_frames=int(entry["n_frames"]),
            fps=float(entry.get("fps", 30.0)),
            measured_xy_m=_as_tuple_float(
                entry.get("measured_xy_m"), 2, (0.0, 0.0)
            ),
            measured_yaw_deg=float(entry.get("measured_yaw_deg", 0.0)),
            measured_waist_pitch_deg=float(entry.get("measured_waist_pitch_deg", 0.0)),
            measured_waist_yaw_deg=float(entry.get("measured_waist_yaw_deg", 0.0)),
            end_at_square_score=float(entry.get("end_at_square_score", 0.0)),
            end_at_apex_score=float(entry.get("end_at_apex_score", 0.0)),
            feet_planted_score=float(entry.get("feet_planted_score", 0.0)),
            pelvis_z_min_m=float(entry.get("pelvis_z_min_m", 0.0)),
            pelvis_z_max_m=float(entry.get("pelvis_z_max_m", 0.0)),
            stride_count=int(entry.get("stride_count", 0)),
            partial=bool(entry.get("partial", False)),
            pinned=bool(entry.get("pinned", False)),
            notes=str(entry.get("notes", "")),
        )
    return out


def write_primitive_registry(
    path: Path, entries: dict[str, PrimitiveEntry]
) -> None:
    """Serialize the registry to YAML in a stable, hand-editable format."""
    path = Path(path)
    rows: list[dict] = []
    for name in sorted(entries):
        e = entries[name]
        rows.append(
            {
                "bin_name": e.bin_name,
                "source_pkl": e.source_pkl,
                "motion_key": e.motion_key,
                "start_frame": int(e.start_frame),
                "n_frames": int(e.n_frames),
                "fps": float(e.fps),
                "measured_xy_m": [float(e.measured_xy_m[0]), float(e.measured_xy_m[1])],
                "measured_yaw_deg": float(e.measured_yaw_deg),
                "measured_waist_pitch_deg": float(e.measured_waist_pitch_deg),
                "measured_waist_yaw_deg": float(e.measured_waist_yaw_deg),
                "end_at_square_score": round(float(e.end_at_square_score), 4),
                "end_at_apex_score": round(float(e.end_at_apex_score), 4),
                "feet_planted_score": round(float(e.feet_planted_score), 4),
                "pelvis_z_min_m": round(float(e.pelvis_z_min_m), 4),
                "pelvis_z_max_m": round(float(e.pelvis_z_max_m), 4),
                "stride_count": int(e.stride_count),
                "partial": bool(e.partial),
                "pinned": bool(e.pinned),
                "notes": e.notes,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"primitives": rows}
    path.write_text(yaml.safe_dump(body, sort_keys=False, indent=2))


@dataclass
class CandidateRow:
    """One row of the per-bin candidate report (markdown)."""

    motion_key: str
    source_pkl: str
    start_frame: int
    n_frames: int
    score: float  # 0..1, higher is better; 1.0 = perfect
    measured_xy_m: tuple[float, float] = (0.0, 0.0)
    measured_yaw_deg: float = 0.0
    measured_waist_pitch_deg: float = 0.0
    measured_waist_yaw_deg: float = 0.0
    end_at_square_score: float = 0.0
    end_at_apex_score: float = 0.0
    feet_planted_score: float = 0.0
    pelvis_z_min_m: float = 0.0
    pelvis_z_max_m: float = 0.0
    stride_count: int = 0
    pass_fail: dict[str, bool] = field(default_factory=dict)


__all__ = [
    "BinSpec",
    "CandidateRow",
    "PrimitiveEntry",
    "load_bin_specs",
    "load_primitive_registry",
    "write_primitive_registry",
]
