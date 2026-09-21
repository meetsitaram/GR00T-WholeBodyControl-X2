#!/usr/bin/env python3
"""Ownership gate for the X2 port: every tracked file is either upstream or listed in X2_FILES.txt.

X2_FILES.txt names each file the port added (A), each upstream file it modified (M)
and each upstream file it removed (D), with a one-line purpose. This tool compares
that list against ``git diff --name-status <base> HEAD`` so the list can never
drift from the tree:

  * a tracked file that is not upstream and not listed      -> FAIL (unowned X2 file)
  * an upstream file changed on the branch but not listed   -> FAIL (silent upstream edit)
  * a listed file whose status no longer matches the diff   -> FAIL (stale entry)
  * a purpose that is empty or still "TODO"                 -> FAIL

    python tools/check_x2_ownership.py            # verify (exit 1 on any failure)
    python tools/check_x2_ownership.py --write    # regenerate, keeping known purposes,
                                                  # "TODO: describe" for new files

Base commit: the upstream NVIDIA commit the port branch was cut from (--base to override).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_BASE = "4141c34"
MANIFEST = "X2_FILES.txt"
HEADER = """# X2_FILES.txt -- ownership manifest of the AgiBot X2 port
#
# Every tracked file that is NOT upstream NVIDIA GR00T-WholeBodyControl (base {base}).
# Regenerate + verify: python tools/check_x2_ownership.py [--write]
# Format: <status>\\t<path>\\t<purpose>   status A = added by the port, M = upstream file modified, D = upstream file removed
#
# Anything tracked that is absent from this list is upstream and must be treated as such on a rebase.
"""


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def diff_status(root: Path, base: str) -> dict[str, str]:
    """{path: 'A'|'M'|'D'} vs base; renames count as D(old) + A(new)."""
    out: dict[str, str] = {}
    for line in git(root, "diff", "--name-status", "-M", base, "HEAD").splitlines():
        parts = line.split("\t")
        code = parts[0][0]
        if code == "R":
            out[parts[1]] = "D"
            out[parts[2]] = "A"
        elif code in "AMD":
            out[parts[1]] = code
        else:  # C, T, ...: treat as modified
            out[parts[-1]] = "M"
    return out


def load_manifest(path: Path) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return entries
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            sys.exit(f"{path}:{ln}: expected <status>\\t<path>\\t<purpose>")
        entries[parts[1]] = (parts[0], "\t".join(parts[2:]).strip())
    return entries


def write_manifest(path: Path, base: str, status: dict[str, str], known: dict[str, tuple[str, str]]) -> None:
    lines = [HEADER.format(base=base)]
    for code in "AMD":
        for f in sorted(p for p, s in status.items() if s == code):
            purpose = known.get(f, ("", ""))[1] or "TODO: describe"
            lines.append(f"{code}\t{f}\t{purpose}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip("\n") + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--write", action="store_true", help="regenerate the manifest (keeps known purposes)")
    a = ap.parse_args()

    status = diff_status(a.root, a.base)
    tracked = set(git(a.root, "ls-files").split("\n")) - {""}
    status = {p: s for p, s in status.items() if (s == "D" or p in tracked) and p != MANIFEST}
    manifest = a.root / MANIFEST
    known = load_manifest(manifest)

    if a.write:
        write_manifest(manifest, a.base, status, known)
        known = load_manifest(manifest)
        print(f"wrote {MANIFEST}: {sum(s == 'A' for s in status.values())} added, "
              f"{sum(s == 'M' for s in status.values())} modified, {sum(s == 'D' for s in status.values())} removed")

    fails: list[str] = []
    for f, s in sorted(status.items()):
        if f not in known:
            fails.append(f"UNLISTED {s} {f}" + ("   (X2 file with no owner entry)" if s == "A" else "   (upstream file changed silently)"))
        elif known[f][0] != s:
            fails.append(f"STATUS   {f}: listed {known[f][0]}, diff says {s}")
    for f, (s, purpose) in sorted(known.items()):
        if f not in status:
            fails.append(f"STALE    {f}: listed {s} but identical to upstream (or untracked)")
        if not purpose or purpose.upper().startswith("TODO"):
            fails.append(f"PURPOSE  {f}: missing description")

    n_up = len(tracked) - sum(1 for f, s in status.items() if s == "A")
    print(f"tracked {len(tracked)} = upstream {n_up} + X2-added {sum(s == 'A' for s in status.values())}; "
          f"upstream modified {sum(s == 'M' for s in status.values())}, removed {sum(s == 'D' for s in status.values())}; "
          f"listed {len(known)}")
    if fails:
        print("\n".join(fails))
        print(f"FAIL: {len(fails)} problem(s). Fix the entries or run: python tools/check_x2_ownership.py --write")
        return 1
    print("OK: every non-upstream file is owned and described")
    return 0


if __name__ == "__main__":
    sys.exit(main())
