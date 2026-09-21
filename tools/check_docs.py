#!/usr/bin/env python3
"""Check that every path / command target mentioned in the X2 docs exists.

Scans docs/x2/*.md, MODELS.md and README_X2.md for backticked spans and for
tokens inside fenced code blocks. Every path-like token must either

  * exist in the repository (relative to --root),
  * be an env-var / angle-bracket / glob placeholder (``$X2_MODELS/...``,
    ``<PC2_IP>``, ``*.x2m2``), a URL, a flag, or an absolute host path that is
    documented as living outside the repo (``/opt/...``, ``/workspace/...``,
    ``${PC2_PREFIX}``), or
  * be listed as a regenerated / bring-your-own artifact: the first column of
    the "Regenerated artifacts" table in docs/x2/BUILD_CHAIN.md, the model tree
    in MODELS.md, or any extra allow-list file passed with --allow (a markdown
    file whose backticked table cells are taken as allowed paths, e.g. a
    REGENERATE.md).

It also refuses the strings that must never appear in shipped docs.

    python tools/check_docs.py                # from the repo root
    python tools/check_docs.py --root <repo> --allow <extra.md> [--warn-only]

Exit status 1 when a path is missing or a forbidden string is found.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC_GLOBS = ["docs/x2/*.md", "MODELS.md", "README_X2.md"]
PATH_EXT = (
    ".sh", ".py", ".yaml", ".yml", ".onnx", ".pkl", ".x2m2", ".env", ".txt", ".md",
    ".html", ".npz", ".service", ".pt", ".json", ".csv", ".cpp", ".hpp", ".xml",
    ".urdf", ".apk", ".deb", ".jsonl", ".ckpt",
)
# absolute prefixes that live on another host / inside a container / are OS paths
EXTERNAL_PREFIXES = (
    "/opt/", "/workspace/", "/ros2_ws", "/aima/", "/etc/", "/dev/", "/lib/", "/usr/",
    "/home/run/", "/tmp/", "/sys/", "/mnt/", "http://", "https://", "tcp://", "git@",
)
FORBIDDEN = re.compile(
    r"boneseed|bones_seed|stickbot|sitaram|tinkerbuggy(?!/sonic-x2)|meetsitaram|"
    r"192\.168\.|10\.0\.1\.|195\.242\.|89\.169\.|10\.11[02]\.|"
    r"x2_upgraded_demo|data_local/|docs/experiments",
    re.IGNORECASE,
)
# Per-robot / per-operator instances that are created from a shipped template
# and never tracked (see F10): allowed even though absent from a clean clone.
BUILT_IN_ALLOW = {
    "x2_pc2/robot_env.env",          # from x2_pc2/robot_env.env.template
    "planner_stack/*",               # PC2-side deploy tree, relative to ${PC2_PREFIX} (F10)
}
INLINE_CODE = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^\s*```")
STRIP_CHARS = "'\"`,;:)(]}[{"


def classify(tok: str) -> str:
    """'skip' | 'placeholder' | 'path' for one whitespace-separated token."""
    t = tok.strip().strip(STRIP_CHARS)
    if not t or t.startswith(("-", "+", "#", ".", "_")):
        return "skip"           # flags, bare extensions / suffix mentions (.x2m2, _dual.onnx), venvs
    if t.startswith("...") or "](" in t or "(" in t or "'" in t or re.match(r"^\d+/", t) or t.startswith(("nvidia/", "tinkerbuggy/sonic-x2")):
        return "skip"           # markdown links, inline python, ratios like 1/4, HF repo ids / HF folders
    # ENV=value -> value ; a=b=c keeps the tail
    if re.match(r"^[A-Z_][A-Z0-9_]*=", t):
        t = t.split("=", 1)[1].strip(STRIP_CHARS)
        if not t:
            return "skip"
    # composite model strings: frozen-core-smpl:<ckpt>:<release>
    if re.match(r"^[a-z0-9-]+:", t) and not t.startswith(EXTERNAL_PREFIXES):
        t = t.split(":", 1)[1]
    if t.startswith(EXTERNAL_PREFIXES):
        return "placeholder"
    if any(c in t for c in "<>$*{}~@?") or t.startswith("~"):
        return "placeholder"
    if "@" in t or t.startswith("/"):
        return "placeholder"
    if t.startswith("gear_sonic.") or t.startswith("motionbricks."):
        # an elided module mention ("gear_sonic...") is prose, not a module
        return "skip" if ("..." in t or t.endswith(".")) else "module"
    if "/" in t or t.endswith(PATH_EXT):
        return "path"
    return "skip"


def normalize(tok: str) -> str:
    t = tok.strip().strip(STRIP_CHARS)
    if re.match(r"^[A-Z_][A-Z0-9_]*=", t):
        t = t.split("=", 1)[1].strip(STRIP_CHARS)
    if re.match(r"^[a-z0-9-]+:", t) and not t.startswith(EXTERNAL_PREFIXES):
        t = t.split(":", 1)[1]
    if t.startswith("./"):
        t = t[2:]
    if "[" in t:
        t = t.split("[", 1)[0]
    if t.startswith(("../", "../../")):
        # links from docs/x2/*.md are relative to that directory
        return t
    return t


def module_to_path(mod: str) -> str:
    return mod.replace(".", "/") + ".py"


def tokens_of(md_text: str):
    """Yield (line_no, token, in_fence) for inline code spans and fenced code lines."""
    in_fence = False
    in_mermaid = False
    for i, line in enumerate(md_text.splitlines(), 1):
        if FENCE.match(line):
            in_mermaid = (not in_fence) and line.strip().startswith("```mermaid")
            in_fence = not in_fence
            continue
        if in_fence:
            if in_mermaid:
                continue          # diagram node ids / labels are not paths
            body = line.split("#", 1)[0]  # drop trailing shell comments
            for tok in body.split():
                yield i, tok, True
        else:
            for span in INLINE_CODE.findall(line):
                for tok in span.split():
                    yield i, tok, False


def allowed_from_table(md_text: str, section: str | None) -> set[str]:
    """Backticked cells of table rows (optionally only under a '## section')."""
    out: set[str] = set()
    active = section is None
    for line in md_text.splitlines():
        if line.startswith("## "):
            active = section is None or line[3:].strip().lower().startswith(section.lower())
            continue
        if active and line.startswith("|"):
            first = line.split("|")[1] if line.count("|") >= 2 else ""
            for span in INLINE_CODE.findall(first):
                for tok in span.replace(",", " ").split():
                    out.add(normalize(tok))
    return out


def allowed_from_tree(md_text: str) -> set[str]:
    """Paths in the MODELS.md directory tree block become allowed patterns."""
    out: set[str] = set()
    for line in md_text.splitlines():
        m = re.match(r"^\s+(\S+\.(?:onnx|pkl|x2m2|pt))\s", line + " ")
        if m:
            out.add(m.group(1))
    return out


def build_index(root: Path) -> set[str]:
    """Every path relative to root (files and dirs), for suffix / basename matching."""
    out: set[str] = set()
    skip = {".git", ".venv", ".venv_teleop", "node_modules", "__pycache__", "build", "install", "log"}
    for p in root.rglob("*"):
        if any(part in skip for part in p.relative_to(root).parts):
            continue
        out.add(str(p.relative_to(root)) + ("/" if p.is_dir() else ""))
    return out


def suffix_match(path: str, index: set[str]) -> bool:
    """'scripts/x2_mujoco_ros_bridge.py' matches 'gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py';
    a bare basename matches any file with that name."""
    p = path.rstrip("/")
    for entry in index:
        e = entry.rstrip("/")
        if e == p or e.endswith("/" + p):
            return True
    return False


def load_ship(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def expand_braces(pat: str) -> list[str]:
    variants = [pat]
    while any("{" in v for v in variants):
        v = next(x for x in variants if "{" in x)
        variants.remove(v)
        pre, rest = v.split("{", 1)
        inner, post = rest.split("}", 1)
        variants += [pre + alt + post for alt in inner.split(",")]
    return variants


def is_allowed(path: str, allow: set[str]) -> bool:
    if path in allow:
        return True
    for pat in allow:
        if "*" in pat:
            if any(fnmatch.fnmatch(path, v) for v in expand_braces(pat)):
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=HERE.parent, help="repo root (default: the directory above tools/)")
    ap.add_argument("--allow", type=Path, action="append", default=[], help="extra markdown allow-list (table cells)")
    ap.add_argument("--ship", type=Path, default=None,
                    help="ship manifest (one repo path per line): paths listed there but not yet present are reported as PENDING, non-fatal")
    ap.add_argument("--warn-only", action="store_true", help="exit 0 even with missing paths")
    ap.add_argument("-v", "--verbose", action="store_true", help="print OK / placeholder tokens too")
    args = ap.parse_args()
    root: Path = args.root.resolve()

    docs: list[Path] = []
    for g in DOC_GLOBS:
        docs += sorted(root.glob(g))
    if not docs:
        print(f"no docs found under {root}", file=sys.stderr)
        return 2

    allow: set[str] = set()
    bc = root / "docs/x2/BUILD_CHAIN.md"
    if bc.exists():
        allow |= allowed_from_table(bc.read_text(), "Regenerated artifacts")
    models = root / "MODELS.md"
    if models.exists():
        allow |= allowed_from_tree(models.read_text())
    for extra in args.allow:
        allow |= allowed_from_table(extra.read_text(), None)
    allow |= BUILT_IN_ALLOW
    allow = {v for pat in allow for v in expand_braces(pat)}

    index = build_index(root)
    ship = load_ship(args.ship) if args.ship else set()
    missing: list[tuple[str, int, str]] = []
    pending: list[tuple[str, int, str]] = []
    forbidden: list[tuple[str, int, str]] = []
    counts = {"ok": 0, "placeholder": 0, "listed": 0, "pending": 0, "missing": 0}
    seen_missing: set[str] = set()
    seen_pending: set[str] = set()

    for doc in docs:
        rel_doc = doc.relative_to(root)
        text = doc.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            m = FORBIDDEN.search(line)
            if m:
                forbidden.append((str(rel_doc), i, m.group(0)))
        for lineno, tok, _fenced in tokens_of(text):
            kind = classify(tok)
            if kind == "skip":
                continue
            if kind == "placeholder":
                counts["placeholder"] += 1
                if args.verbose:
                    print(f"  placeholder {rel_doc}:{lineno}: {tok}")
                continue
            path = module_to_path(normalize(tok)) if kind == "module" else normalize(tok)
            # markdown links relative to the doc's directory
            candidate = (doc.parent / path) if path.startswith("../") else (root / path)
            if candidate.exists() or suffix_match(path, index):
                counts["ok"] += 1
                if args.verbose:
                    print(f"  ok          {rel_doc}:{lineno}: {path}")
                continue
            if is_allowed(path, allow) or suffix_match(path, allow):
                counts["listed"] += 1
                if args.verbose:
                    print(f"  listed      {rel_doc}:{lineno}: {path}")
                continue
            if ship and (path.rstrip("/") in ship or suffix_match(path, ship)
                         or any(sp.startswith(path.rstrip("/") + "/") or ("/" + path.rstrip("/") + "/") in sp for sp in ship)):
                counts["pending"] += 1
                if path not in seen_pending:
                    seen_pending.add(path)
                    pending.append((str(rel_doc), lineno, path))
                continue
            counts["missing"] += 1
            if path not in seen_missing:
                seen_missing.add(path)
                missing.append((str(rel_doc), lineno, path))

    print(f"docs checked: {len(docs)}; ok={counts['ok']} placeholder={counts['placeholder']} "
          f"listed={counts['listed']} pending={counts['pending']} (unique {len(pending)}) "
          f"missing={counts['missing']} (unique {len(missing)})")
    if pending:
        print("\nPENDING paths (in the ship manifest, not in this tree yet):")
        for d, i, p in sorted(pending, key=lambda x: x[2]):
            print(f"  {p}    (first: {d}:{i})")
    if forbidden:
        print("\nFORBIDDEN strings:")
        for d, i, s in forbidden:
            print(f"  {d}:{i}: {s}")
    if missing:
        print("\nMISSING paths (not in repo, not a placeholder, not a listed artifact):")
        for d, i, p in sorted(missing, key=lambda x: x[2]):
            print(f"  {p}    (first: {d}:{i})")
    if forbidden or (missing and not args.warn_only):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
