"""Build ``x2_planner_primitives.pkl`` from a recipes YAML.

Recipe-driven replacement for the curator's "best-of-K window pick" PKL
writer. The curator still mines + reports candidates; this script is the
last mile that hand-edits / synthesizes / mirrors the picks into the
runtime PKL.

Re-runnable. Edit the recipes YAML, re-run, commit. The runtime planner
loads the resulting PKL through ``state_machine.load_primitives_pkl`` --
exactly the same loader the curator output already feeds.

Run from the repo root::

    .venv/bin/python -m gear_sonic.scripts.build_x2_planner_primitives \
        --source   gear_sonic/data/motions/x2_demo_bank.pkl \
        --recipes  gear_sonic/data/motions/x2_planner_primitives_recipes.yaml \
        --bins     gear_sonic/data/motions/x2_planner_bins.yaml \
        --out-pkl  gear_sonic/data/motions/x2_planner_primitives.pkl \
        --out-report gear_sonic/data/motions/x2_planner_primitives_recipes_report.md

For per-bin debugging::

    .venv/bin/python -m gear_sonic.scripts.build_x2_planner_primitives \
        --bins-only torso_left_15deg fwd_step_quarter_ft \
        --no-write-pkl   # just print recipe summaries

Output PKL schema (per bin) -- matches what the runtime ``Primitive`` loader
already expects::

    {
      "<bin_name>": {
        "dof":            float32 (T, 31),
        "root_rot_xyzw":  float32 (T, 4),
        "root_trans":     float32 (T, 3),
        "fps":            float,
        "source_pkl":     str,                # the source motion-library PKL
        "motion_key":     str,                # provenance label, e.g. "synth:..." or "loco_xxx"
        "start_frame":    int,                # 0 if synthesized
        "n_frames":       int,
        "partial":        bool,               # always False for built primitives
        "pinned":         bool,               # always True (recipe is a hand pin)
        "freeze_arms_to_default": bool,       # True if a freeze op covered both arms+head
        "recipe_family":  str,                # from the recipe entry
        "recipe_ops":     list[str],          # human-readable op trail
        "recipe_sources": list[str],          # buffer.sources at end of pipeline
      },
      ...
    }
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

# Make ``import gear_sonic.utils.planner`` work when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.planner.constants import (  # noqa: E402
    HEAD_INDICES,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
)
from gear_sonic.utils.planner.x2_recipes import (  # noqa: E402
    Buffer,
    Recipe,
    SourceClip,
    load_recipes,
    run_recipe,
)


_FREEZE_ARMS_HEAD_INDICES: frozenset[int] = frozenset(
    list(LEFT_ARM_INDICES) + list(RIGHT_ARM_INDICES) + list(HEAD_INDICES)
)


# ---------------------------------------------------------------------------
# Source library I/O
# ---------------------------------------------------------------------------


def load_source_clips(path: Path) -> dict[str, SourceClip]:
    """Load the source motion-library PKL into ``{motion_key: SourceClip}``.

    Mirrors ``curate_x2_primitives.load_motion_library`` so the recipe
    builder and the curator agree on field names / dtypes.
    """
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected dict, got {type(raw).__name__}")
    out: dict[str, SourceClip] = {}
    for key, entry in raw.items():
        try:
            dof = np.asarray(entry["dof"], dtype=np.float32)
            root_rot = np.asarray(entry["root_rot"], dtype=np.float32)
            root_trans = np.asarray(entry["root_trans_offset"], dtype=np.float32)
            fps = float(entry.get("fps", 30))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: clip {key!r} missing field {exc}") from exc
        if dof.ndim != 2 or dof.shape[1] != 31:
            raise ValueError(
                f"{path}: clip {key!r} dof shape {dof.shape}, expected (T, 31)"
            )
        if root_rot.shape[0] != dof.shape[0] or root_rot.shape[1] != 4:
            raise ValueError(
                f"{path}: clip {key!r} root_rot shape {root_rot.shape}, "
                f"expected ({dof.shape[0]}, 4)"
            )
        if root_trans.shape[0] != dof.shape[0] or root_trans.shape[1] != 3:
            raise ValueError(
                f"{path}: clip {key!r} root_trans shape {root_trans.shape}, "
                f"expected ({dof.shape[0]}, 3)"
            )
        out[str(key)] = SourceClip(
            motion_key=str(key),
            dof=dof,
            root_rot_xyzw=root_rot,
            root_trans=root_trans,
            fps=fps,
        )
    return out


# ---------------------------------------------------------------------------
# Bin family lookup (so the runtime loader maps recipe -> family)
# ---------------------------------------------------------------------------


def load_bin_family_lookup(bins_yaml: Path | None) -> dict[str, str]:
    """Best-effort ``{bin_name: family}`` lookup from the bins YAML.

    The runtime loader uses the family to pick blend windows. If a recipe
    declares a different family from the bins YAML, the recipe wins (and we
    print a one-line warning).
    """
    if bins_yaml is None or not Path(bins_yaml).exists():
        return {}
    raw = yaml.safe_load(Path(bins_yaml).read_text())
    if not isinstance(raw, dict) or "bins" not in raw:
        return {}
    out: dict[str, str] = {}
    for entry in raw["bins"]:
        try:
            out[str(entry["name"])] = str(entry["family"])
        except (KeyError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def build_all(
    recipes: dict[str, Recipe],
    source_clips: dict[str, SourceClip],
    only: set[str] | None = None,
    skip_missing_sources: bool = False,
) -> dict[str, tuple[Recipe, Buffer]]:
    """Run every recipe (or only the requested subset) and collect buffers.

    ``skip_missing_sources``: a recipe whose ``clip_window`` op names a
    ``motion_key`` absent from ``source_clips`` is skipped with a warning
    instead of aborting the build (the source library may be a smaller
    corpus than the one the recipes were curated on; synthesized bins
    still build).
    """
    out: dict[str, tuple[Recipe, Buffer]] = {}
    for name, recipe in recipes.items():
        if only is not None and name not in only:
            continue
        try:
            buf = run_recipe(recipe, recipes, source_clips)
        except KeyError as exc:
            if skip_missing_sources and "not in source library" in str(exc):
                print(f"[build] SKIP {name!r}: {exc.args[0]}", flush=True)
                continue
            raise RuntimeError(f"recipe {name!r}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"recipe {name!r}: {exc}") from exc
        out[name] = (recipe, buf)
    return out


def _recipe_op_trail(recipe: Recipe, recipes: dict[str, Recipe]) -> list[str]:
    """Human-readable op trail including derive_from chain (for the report)."""
    trail: list[str] = []
    if recipe.derive_from:
        trail.append(f"derive_from:{recipe.derive_from}")
        if recipe.derive_from in recipes:
            trail.extend(_recipe_op_trail(recipes[recipe.derive_from], recipes))
    for op in recipe.ops:
        op_name = next(iter(op))
        trail.append(op_name)
    return trail


def _has_arms_and_head_freeze(recipe: Recipe, recipes: dict[str, Recipe]) -> bool:
    """True if any freeze op in the recipe (incl. derived) covers both arms + head."""
    visited: set[str] = set()
    stack: list[Recipe] = [recipe]
    while stack:
        r = stack.pop()
        if r.bin_name in visited:
            continue
        visited.add(r.bin_name)
        for op in r.ops:
            if "freeze" in op:
                groups = {str(g).lower() for g in op["freeze"].get("groups", [])}
                covers_arms = (
                    "arms" in groups
                    or {"left_arm", "right_arm"}.issubset(groups)
                )
                covers_head = "head" in groups
                if covers_arms and covers_head:
                    return True
        if r.derive_from and r.derive_from in recipes:
            stack.append(recipes[r.derive_from])
    return False


def write_primitives_pkl(
    out_path: Path,
    built: dict[str, tuple[Recipe, Buffer]],
    recipes: dict[str, Recipe],
    source_pkl: str,
) -> int:
    """Write the runtime-ready primitives PKL. Returns number of bins written."""
    out: dict[str, dict[str, Any]] = {}
    for bin_name, (recipe, buf) in built.items():
        # Provenance: prefer the deepest source clip in the chain, fall back
        # to "synth:..." style markers from synthesize_* / op:* sources.
        clip_sources = [s for s in buf.sources if "[" in s and ":" not in s.split("[")[0]]
        if clip_sources:
            mk = clip_sources[0].split("[")[0]
            sf = int(clip_sources[0].split("[")[1].split(":")[0])
        else:
            mk = next(
                (s for s in buf.sources if s.startswith("synth:")),
                f"recipe:{bin_name}",
            )
            sf = 0
        out[bin_name] = {
            "dof": np.ascontiguousarray(buf.dof, dtype=np.float32),
            "root_rot_xyzw": np.ascontiguousarray(buf.root_rot_xyzw, dtype=np.float32),
            "root_trans": np.ascontiguousarray(buf.root_trans, dtype=np.float32),
            "fps": float(buf.fps),
            "source_pkl": source_pkl,
            "motion_key": str(mk),
            "start_frame": int(sf),
            "n_frames": int(buf.n_frames()),
            "partial": False,
            "pinned": True,
            "freeze_arms_to_default": _has_arms_and_head_freeze(recipe, recipes),
            "recipe_family": recipe.family,
            "recipe_ops": _recipe_op_trail(recipe, recipes),
            "recipe_sources": list(buf.sources),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, out_path)
    return len(out)


# ---------------------------------------------------------------------------
# Markdown report (parallel to the curator report)
# ---------------------------------------------------------------------------


def write_report(
    out_path: Path,
    built: dict[str, tuple[Recipe, Buffer]],
    recipes: dict[str, Recipe],
    recipes_yaml: Path,
    source_pkl: Path,
    started_at: float,
) -> None:
    lines: list[str] = []
    lines.append("# X2 planner primitive **recipe** build report")
    lines.append("")
    def _rel(p: Path) -> str:  # repo-relative when possible (no machine paths in the report)
        try:
            return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return Path(p).name
    lines.append(f"- Recipes: `{_rel(recipes_yaml)}`")
    lines.append(f"- Source: `{_rel(source_pkl)}`")
    lines.append(f"- Bins built: **{len(built)}**")
    lines.append(f"- Build wall time: {time.time() - started_at:.1f}s")
    lines.append("")
    lines.append("## Bin summary")
    lines.append("")
    lines.append(
        "| Bin | Family | Frames | fps | Recipe ops | Frozen arms+head |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name in sorted(built):
        recipe, buf = built[name]
        ops_str = " -> ".join(_recipe_op_trail(recipe, recipes))
        frozen = "yes" if _has_arms_and_head_freeze(recipe, recipes) else "no"
        lines.append(
            f"| `{name}` | {recipe.family} | {buf.n_frames()} | "
            f"{buf.fps:.1f} | `{ops_str}` | {frozen} |"
        )
    lines.append("")
    lines.append("## Per-bin sources")
    lines.append("")
    for name in sorted(built):
        recipe, buf = built[name]
        lines.append(f"### `{name}` ({recipe.family})")
        if recipe.notes:
            lines.append(f"- notes: {recipe.notes}")
        lines.append(f"- frames: {buf.n_frames()} @ {buf.fps:.1f} fps")
        lines.append("- sources:")
        for s in buf.sources:
            lines.append(f"  - `{s}`")
        lines.append("")
    lines.append("---")
    lines.append("Edit the recipes YAML and re-run "
                 "``gear_sonic.scripts.build_x2_planner_primitives`` to regenerate.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo_data = _REPO_ROOT / "gear_sonic" / "data" / "motions"
    p.add_argument(
        "--source", type=Path, required=True,
        help="Source motion library PKL ({motion_key: entry} with dof / "
             "root_rot / root_trans_offset / fps), e.g. the regenerated "
             "gear_sonic/data/motions/x2_demo_bank.pkl "
             "(tools/build_demo_bank_from_upstream.sh).",
    )
    p.add_argument(
        "--recipes", type=Path,
        default=repo_data / "x2_planner_primitives_recipes.yaml",
        help="Recipes YAML.",
    )
    p.add_argument(
        "--bins", type=Path,
        default=repo_data / "x2_planner_bins.yaml",
        help="Bins YAML (used only for cross-checking family labels).",
    )
    p.add_argument(
        "--out-pkl", type=Path,
        default=repo_data / "x2_planner_primitives.pkl",
        help="Where to write the runtime primitives PKL.",
    )
    p.add_argument(
        "--out-report", type=Path,
        default=repo_data / "x2_planner_primitives_recipes_report.md",
        help="Where to write the markdown build report.",
    )
    p.add_argument(
        "--bins-only", nargs="*", default=None,
        help="If provided, only build these bin names (rest skipped).",
    )
    p.add_argument(
        "--no-write-pkl", action="store_true",
        help="Skip writing the PKL (still prints summaries / writes report).",
    )
    p.add_argument(
        "--no-write-report", action="store_true",
        help="Skip writing the markdown report.",
    )
    p.add_argument(
        "--skip-missing-sources", action="store_true",
        help="Skip (with a warning) recipes whose clip_window motion_key is "
             "not in --source instead of aborting; synthesized bins still build.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    started_at = time.time()

    print(f"[build] loading recipes from {args.recipes}", flush=True)
    recipes = load_recipes(args.recipes)
    print(f"[build]   {len(recipes)} recipes", flush=True)

    print(f"[build] loading source library from {args.source}", flush=True)
    source_clips = load_source_clips(args.source)
    print(f"[build]   {len(source_clips)} clips", flush=True)

    family_lookup = load_bin_family_lookup(args.bins)
    if family_lookup:
        for name, recipe in recipes.items():
            if name in family_lookup and family_lookup[name] != recipe.family:
                print(
                    f"[build]   warn: recipe {name!r} family={recipe.family!r} "
                    f"differs from bins YAML family={family_lookup[name]!r} "
                    "(recipe wins)",
                    flush=True,
                )

    only = set(args.bins_only) if args.bins_only else None
    if only:
        unknown = only - set(recipes)
        if unknown:
            print(f"[build] error: unknown bins in --bins-only: {sorted(unknown)}",
                  file=sys.stderr)
            return 2

    built = build_all(recipes, source_clips, only=only,
                      skip_missing_sources=args.skip_missing_sources)
    print(f"[build] built {len(built)} bins in {time.time() - started_at:.1f}s",
          flush=True)
    for name in sorted(built):
        recipe, buf = built[name]
        ops_str = " -> ".join(_recipe_op_trail(recipe, recipes))
        print(
            f"[build]   {name:<24s} {recipe.family:<18s} "
            f"{buf.n_frames():>4d}f @ {buf.fps:>4.1f}fps  | {ops_str}",
            flush=True,
        )

    if not args.no_write_pkl:
        n = write_primitives_pkl(
            args.out_pkl, built, recipes, source_pkl=str(args.source)
        )
        print(f"[build] wrote {n} bins to {args.out_pkl}", flush=True)
    else:
        print("[build] --no-write-pkl: skipping PKL write", flush=True)

    if not args.no_write_report:
        write_report(
            args.out_report, built, recipes, args.recipes, args.source, started_at
        )
        print(f"[build] wrote report to {args.out_report}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
