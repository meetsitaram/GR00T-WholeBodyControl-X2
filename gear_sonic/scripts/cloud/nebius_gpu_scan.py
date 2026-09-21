#!/usr/bin/env python3
"""Scan Nebius for real-time GPU capacity available to your tenant.

Wraps ``nebius capacity resource-advice list`` (the ResourceAdvisorService) so
you can answer "where can I actually launch an 8xH200 right now?" without
guessing regions and waiting 5 min for the scheduler to time out.

Why this exists:

  Nebius's ``compute instance create`` does NOT fail fast when a region/zone
  has no capacity for the requested platform/preset. It accepts the request,
  queues it in the scheduler, and waits. From the CLI side this looks
  identical to "the API is slow" until you eventually get NotEnoughResources.
  This script asks the capacity API up-front so you pick a (region, platform,
  preset) tuple that actually has on-demand or preemptible inventory.

Usage::

    # Default: all GPU presets, sorted by GPU count desc, with auto-recommendation
    python gear_sonic/scripts/cloud/nebius_gpu_scan.py

    # Only multi-GPU on-demand offerings with capacity right now
    python gear_sonic/scripts/cloud/nebius_gpu_scan.py --gpus 8 --min-on-demand 1

    # Filter by region and platform
    python gear_sonic/scripts/cloud/nebius_gpu_scan.py --region eu-west1 \\
                                                      --platform gpu-h200-sxm

    # Raw JSON for piping into jq / other tooling
    python gear_sonic/scripts/cloud/nebius_gpu_scan.py --format json

Requirements:

  - ``nebius`` CLI installed and authenticated (``nebius iam whoami`` works)
  - Active profile (``nebius profile list`` shows one with a ``[default]``)
  - Python 3.8+ (stdlib only)

Notes:

  - The capacity API returns one row per AZ × platform × preset. Multiple
    rows for the same platform in the same region usually mean the platform
    is offered in multiple AZs; the "available" numbers are per-AZ.
  - ``availability_level`` buckets the absolute count: HIGH / MEDIUM / LOW /
    LIMIT_REACHED. Use it as a tie-breaker.
  - ``data_state`` of ``STALE`` means the count was cached >5min ago; treat
    it as a hint only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class CapacityRow:
    gpus: int
    platform: str
    region: str
    preset: str
    od_available: int
    od_limit: int
    od_level: str
    od_stale: bool
    pe_available: int
    pe_limit: int
    pe_level: str
    pe_stale: bool


_LEVEL_RANK = {
    "HIGH": 4,
    "MEDIUM": 3,
    "LOW": 2,
    "LIMIT_REACHED": 1,
    "UNSPECIFIED": 0,
    "?": 0,
}


def _short_level(level: str) -> str:
    return (level or "?").replace("AVAILABILITY_LEVEL_", "") or "?"


def _run_nebius(args: list[str]) -> dict:
    if shutil.which("nebius") is None:
        sys.exit(
            "ERROR: 'nebius' CLI not found in PATH. Install per "
            "docs/source/user_guide/train-on-cloud.md Appendix A.1."
        )
    proc = subprocess.run(
        ["nebius", *args, "--format", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        sys.exit(f"ERROR: nebius {' '.join(args)} failed:\n{proc.stderr.strip()}")
    # The CLI may print an OAuth URL to stderr before the JSON arrives on
    # stdout; we only care about stdout. If stdout is empty something is
    # wrong with the auth state.
    out = proc.stdout.strip()
    if not out:
        sys.exit(
            "ERROR: nebius CLI returned empty stdout. Likely auth expired.\n"
            "Run 'nebius iam whoami' once to re-trigger the OAuth flow,\n"
            "then re-run this script."
        )
    return json.loads(out)


def _resolve_tenant_id(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    data = _run_nebius(["iam", "tenant", "list"])
    items = data.get("items", [])
    if not items:
        sys.exit("ERROR: no tenants visible to this profile. Run 'nebius iam whoami'.")
    if len(items) > 1:
        names = [it["metadata"]["id"] for it in items]
        print(
            f"WARNING: multiple tenants visible {names}; using first. "
            "Pass --tenant-id to override.",
            file=sys.stderr,
        )
    return items[0]["metadata"]["id"]


def _fetch_capacity(tenant_id: str) -> list[dict]:
    data = _run_nebius(
        ["capacity", "resource-advice", "list", "--parent-id", tenant_id, "--all"]
    )
    return data.get("items", [])


def _to_row(item: dict) -> Optional[CapacityRow]:
    spec = item.get("spec", {})
    ci = spec.get("compute_instance", {})
    if not ci:
        return None
    preset = ci.get("preset", {})
    gpus = int(preset.get("resources", {}).get("gpu_count", 0) or 0)
    status = item.get("status", {})
    od = status.get("on_demand", {}) or {}
    pe = status.get("preemptible", {}) or {}
    return CapacityRow(
        gpus=gpus,
        platform=ci.get("platform", "?"),
        region=spec.get("region", "?"),
        preset=preset.get("name", "?"),
        od_available=int(od.get("available", 0) or 0),
        od_limit=int(od.get("limit", 0) or 0),
        od_level=_short_level(od.get("availability_level", "?")),
        od_stale=od.get("data_state", "") == "DATA_STATE_STALE",
        pe_available=int(pe.get("available", 0) or 0),
        pe_limit=int(pe.get("limit", 0) or 0),
        pe_level=_short_level(pe.get("availability_level", "?")),
        pe_stale=pe.get("data_state", "") == "DATA_STATE_STALE",
    )


def _filter(rows: list[CapacityRow], args) -> list[CapacityRow]:
    out = []
    for r in rows:
        if not args.include_cpu and r.gpus == 0:
            continue
        if args.gpus is not None and r.gpus < args.gpus:
            continue
        if args.region and r.region != args.region:
            continue
        if args.platform and r.platform != args.platform:
            continue
        if args.min_on_demand is not None and r.od_available < args.min_on_demand:
            continue
        if args.min_preemptible is not None and r.pe_available < args.min_preemptible:
            continue
        out.append(r)
    return out


def _sort_for_display(rows: list[CapacityRow]) -> list[CapacityRow]:
    return sorted(
        rows,
        key=lambda r: (-r.gpus, r.platform, r.region, r.preset),
    )


def _recommend(rows: list[CapacityRow]) -> Optional[CapacityRow]:
    """Pick the most attractive row: prefer multi-GPU + on-demand + HIGH level."""
    candidates = [r for r in rows if r.gpus >= 1 and r.od_available > 0]
    if not candidates:
        candidates = [r for r in rows if r.gpus >= 1 and r.pe_available > 0]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (
            r.gpus,
            r.od_available,
            _LEVEL_RANK.get(r.od_level, 0),
            r.pe_available,
            _LEVEL_RANK.get(r.pe_level, 0),
        ),
    )


def _format_avail(available: int, limit: int, level: str, stale: bool) -> str:
    marker = "*" if stale else " "
    badge = " "
    if level == "HIGH" and available > 0:
        badge = "+"
    elif level in ("LIMIT_REACHED",) or available == 0:
        badge = "-"
    return f"{available:>4}/{limit:<4}{marker} {level:<6}{badge}"


def _print_table(rows: list[CapacityRow], tenant_id: str, args) -> None:
    if args.format == "json":
        json.dump(
            [r.__dict__ for r in rows],
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return

    print(f"Tenant: {tenant_id}")
    filt_parts = []
    if args.gpus is not None:
        filt_parts.append(f"gpus>={args.gpus}")
    if args.region:
        filt_parts.append(f"region={args.region}")
    if args.platform:
        filt_parts.append(f"platform={args.platform}")
    if args.min_on_demand is not None:
        filt_parts.append(f"on-demand>={args.min_on_demand}")
    if args.min_preemptible is not None:
        filt_parts.append(f"preemptible>={args.min_preemptible}")
    if filt_parts:
        print(f"Filter: {', '.join(filt_parts)}")
    print()

    if not rows:
        print("(no rows match)")
        return

    hdr = (
        f"{'GPUs':>4} | {'platform':<14} | {'region':<12} | {'preset':<22} "
        f"| {'on-demand (a/l, level)':<25} | preemptible (a/l, level)"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        od = _format_avail(r.od_available, r.od_limit, r.od_level, r.od_stale)
        pe = _format_avail(r.pe_available, r.pe_limit, r.pe_level, r.pe_stale)
        print(
            f"{r.gpus:>4} | {r.platform:<14} | {r.region:<12} | {r.preset:<22} | {od} | {pe}"
        )
    print()
    print("Legend: '+' high-confidence available, '-' empty/limit-reached, '*' STALE.")
    print("        Numbers are per-AZ; multiple rows per (platform, region) = multiple AZs.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Nebius capacity API for real-time GPU availability. "
            "Helps avoid 'create instance' calls that sit in the scheduler "
            "for 5+ min before failing on no-capacity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tenant-id",
        help="Tenant NID. Auto-discovered via `nebius iam tenant list` if omitted.",
    )
    parser.add_argument(
        "--region",
        help="Filter to one region (e.g. eu-north1, eu-west1, us-central1, me-west1).",
    )
    parser.add_argument(
        "--platform",
        help="Filter to one platform (e.g. gpu-h200-sxm, gpu-h100-sxm, gpu-b200-sxm).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        help="Minimum GPU count per preset (e.g. 8 to only see 8-GPU offerings).",
    )
    parser.add_argument(
        "--min-on-demand",
        type=int,
        help="Only show rows with on-demand 'available' >= this number.",
    )
    parser.add_argument(
        "--min-preemptible",
        type=int,
        help="Only show rows with preemptible 'available' >= this number.",
    )
    parser.add_argument(
        "--include-cpu",
        action="store_true",
        help="Include CPU-only presets (default: GPU only).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--no-recommend",
        action="store_true",
        help="Suppress the 'Recommended:' line at the bottom.",
    )
    args = parser.parse_args()

    tenant_id = _resolve_tenant_id(args.tenant_id)
    raw_items = _fetch_capacity(tenant_id)
    rows = [r for r in (_to_row(it) for it in raw_items) if r is not None]
    rows = _filter(rows, args)
    rows = _sort_for_display(rows)

    _print_table(rows, tenant_id, args)

    if args.format == "table" and not args.no_recommend:
        rec = _recommend(rows)
        if rec is not None:
            mode = "on-demand" if rec.od_available > 0 else "preemptible"
            avail = rec.od_available if rec.od_available > 0 else rec.pe_available
            limit = rec.od_limit if rec.od_available > 0 else rec.pe_limit
            level = rec.od_level if rec.od_available > 0 else rec.pe_level
            print()
            print(
                f"Recommended: {rec.gpus}x {rec.platform} in {rec.region} "
                f"({mode}, {avail}/{limit} {level}, preset {rec.preset})"
            )
        elif args.format == "table":
            print()
            print(
                "No on-demand or preemptible capacity matches the filter. "
                "Loosen --gpus / --region / --platform or wait + re-run."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
