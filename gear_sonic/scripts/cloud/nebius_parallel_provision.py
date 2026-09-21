#!/usr/bin/env python3
"""Parallel-provision an 8x GPU node on Nebius across multiple region/platform variants.

Why this exists
---------------

8-GPU presets (``gpu-h100-sxm``, ``gpu-h200-sxm``, ``gpu-b200-sxm``,
``gpu-b200-sxm-a``) are routinely fully booked. ``nebius compute instance create``
does *not* fail fast: it accepts the request, queues it in the scheduler,
sometimes returns ``state: STOPPED`` after a 5-minute timeout, sometimes goes
through ``STARTING`` → ``RUNNING`` but lands on a "husk" (no real node, TCP/22
never opens — see ``train-on-cloud.md`` Appendix B.4).

Strategy: fire off **multiple region/platform variants in parallel**, watch
each one's status, and the first variant that reaches ``RUNNING`` with a real
SSH-reachable IP wins. The losers get auto-deleted.

Default catalog (6 variants, all 8-GPU presets)::

    1. eu-west1     gpu-h200-sxm    8gpu-128vcpu-1600gb
    2. us-central1  gpu-b200-sxm    8gpu-160vcpu-1792gb
    3. us-central1  gpu-h200-sxm    8gpu-128vcpu-1600gb
    4. eu-north1    gpu-h100-sxm    8gpu-128vcpu-1600gb
    5. eu-north1    gpu-h200-sxm    8gpu-128vcpu-1600gb
    6. me-west1     gpu-b200-sxm-a  8gpu-160vcpu-1792gb

Concurrency caps
----------------

- ``--max-per-region`` (default 2): your tenant quota typically caps you at
  ~2 simultaneous instances per region. Variants beyond the cap queue and
  launch only after an earlier variant in the same region reaches a verdict.
- ``--max-total`` (default 6): overall cap. With the default catalog the
  effective max in-flight is min(cap, 2 + 1 + 3 + 1) = 6 anyway.

Verdict logic
-------------

For each variant, we poll ``compute instance get`` until we get a verdict:

- ``RUNNING`` + public IP + TCP/22 reachable within 60 s → **WIN**
- ``RUNNING`` + TCP/22 stays closed for 60 s → **FAIL** (husk, see B.4)
- ``STOPPED`` / ``ERROR`` → **FAIL** (Nebius gave up scheduling)

We deliberately **do not** treat ``STARTING`` as a failure on its own — we
wait until the scheduler resolves to ``RUNNING`` or ``STOPPED``.

Usage
-----

::

    # Default flow: launch 6 variants, wait for first winner, delete losers
    python gear_sonic/scripts/cloud/nebius_parallel_provision.py

    # Dry-run: show the exact `nebius` commands without running them
    python gear_sonic/scripts/cloud/nebius_parallel_provision.py --dry-run

    # Loop until success: probe nebius_gpu_scan.py every 20s for 8-GPU on-demand
    # capacity; when capacity appears, fire the parallel provisioner; on no
    # winner, cool down 120s and re-probe. Exits 0 on first winner.
    python gear_sonic/scripts/cloud/nebius_parallel_provision.py \\
        --retry-until-success --probe-interval 20

    # Cleanup after a previous run (resume from state file)
    python gear_sonic/scripts/cloud/nebius_parallel_provision.py \\
        --cleanup /tmp/nebius_parallel_<ts>/state.json

Requirements
------------

- ``nebius`` CLI installed + authenticated (``nebius iam whoami`` works).
- 4 default projects (one per region) discoverable under your tenant.
- Each region's project has at least one VPC subnet.
- ``~/.ssh/id_ed25519.pub`` (override with ``--ssh-key``).

Stdlib only — no extra deps. Spawns one thread per variant; each thread runs
``nebius`` subcommands serially within itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# -----------------------------------------------------------------------------#
# Defaults                                                                     #
# -----------------------------------------------------------------------------#

DEFAULT_VARIANTS: list[tuple[str, str, str]] = [
    # (region, platform, preset)  — ordered by today's capacity hint (HIGH first)
    ("eu-west1",    "gpu-h200-sxm",   "8gpu-128vcpu-1600gb"),
    ("us-central1", "gpu-b200-sxm",   "8gpu-160vcpu-1792gb"),
    ("us-central1", "gpu-h200-sxm",   "8gpu-128vcpu-1600gb"),
    ("eu-north1",   "gpu-h100-sxm",   "8gpu-128vcpu-1600gb"),
    ("eu-north1",   "gpu-h200-sxm",   "8gpu-128vcpu-1600gb"),
    ("me-west1",    "gpu-b200-sxm-a", "8gpu-160vcpu-1792gb"),
]

# Public-image projects are REGION-SCOPED: e00* serves the eu regions,
# u00* serves us-central1, m00* me-west1 (2026-08-30: us-central1 H200
# create failed with "no image of family ... in region" against the
# hardcoded eu project). Resolve by region prefix.
PUBLIC_IMAGES_PARENT_BY_PREFIX = {
    "eu": "project-e00public-images",
    "us": "project-u00public-images",
    "me": "project-m00public-images",
}
def public_images_parent(region: str) -> str:
    return PUBLIC_IMAGES_PARENT_BY_PREFIX.get(region[:2], "project-e00public-images")

DEFAULT_IMAGE_FAMILY = "ubuntu24.04-driverless"
DEFAULT_DISK_SIZE_GIB = 500
DEFAULT_DISK_TYPE = "network_ssd"
DEFAULT_USER = "ubuntu"
DEFAULT_INSTANCE_PREFIX = "x2-train"
DEFAULT_POLL_INTERVAL_S = 15
DEFAULT_SSH_GRACE_S = 60               # how long to wait for TCP/22 after RUNNING
NEBIUS_CMD_TIMEOUT_S = 120             # any single nebius CLI call


# -----------------------------------------------------------------------------#
# Data model                                                                   #
# -----------------------------------------------------------------------------#

VERDICT_PENDING = "pending"
VERDICT_WIN = "win"
VERDICT_FAIL = "fail"
VERDICT_CANCELLED = "cancelled"


@dataclass
class Variant:
    region: str
    platform: str
    preset: str
    name: str = ""
    project_id: str = ""
    subnet_id: str = ""
    disk_id: str = ""
    instance_id: str = ""
    state: str = ""
    public_ip: str = ""
    verdict: str = VERDICT_PENDING
    reason: str = ""
    created_at: str = ""
    won_at: str = ""

    @property
    def label(self) -> str:
        return f"{self.region}/{self.platform}"


@dataclass
class RunState:
    started_at: str
    workdir: str
    ssh_key_path: str
    ssh_pubkey: str
    cloud_init_path: str
    variants: list[Variant] = field(default_factory=list)


# -----------------------------------------------------------------------------#
# Nebius CLI helpers                                                           #
# -----------------------------------------------------------------------------#

_PRINT_LOCK = threading.Lock()


def log(msg: str, *, prefix: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {prefix}{msg}" if prefix else f"[{ts}] {msg}"
    with _PRINT_LOCK:
        print(line, flush=True)


def _run_nebius(args: list[str], *, timeout: int = NEBIUS_CMD_TIMEOUT_S) -> dict:
    if shutil.which("nebius") is None:
        raise RuntimeError(
            "nebius CLI not found in PATH. Install per "
            "docs/source/user_guide/train-on-cloud.md A.1."
        )
    proc = subprocess.run(
        ["nebius", *args, "--format", "json"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nebius {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    out = proc.stdout.strip()
    if not out:
        return {}
    return json.loads(out)


def _run_nebius_void(args: list[str], *, timeout: int = NEBIUS_CMD_TIMEOUT_S) -> str:
    """Run nebius command without --format json (e.g. delete returns plain text)."""
    if shutil.which("nebius") is None:
        raise RuntimeError("nebius CLI not found in PATH.")
    proc = subprocess.run(
        ["nebius", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nebius {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def _resolve_tenant_id(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    data = _run_nebius(["iam", "tenant", "list"])
    items = data.get("items", [])
    if not items:
        raise RuntimeError("no tenants visible to this profile. Run `nebius iam whoami`.")
    return items[0]["metadata"]["id"]


def _discover_projects_by_region(tenant_id: str) -> dict[str, str]:
    data = _run_nebius(["iam", "project", "list", "--parent-id", tenant_id])
    out: dict[str, str] = {}
    for it in data.get("items", []):
        region = it.get("spec", {}).get("region") or it.get("status", {}).get("region")
        if region:
            out[region] = it["metadata"]["id"]
    return out


def _resolve_tenant_id_t(timeout: int) -> str:
    data = _run_nebius(["iam", "tenant", "list"], timeout=timeout)
    items = data.get("items", [])
    if not items:
        raise RuntimeError("no tenants visible to this profile. Run `nebius iam whoami`.")
    return items[0]["metadata"]["id"]


def _discover_projects_by_region_t(tenant_id: str, timeout: int) -> dict[str, str]:
    data = _run_nebius(["iam", "project", "list", "--parent-id", tenant_id], timeout=timeout)
    out: dict[str, str] = {}
    for it in data.get("items", []):
        region = it.get("spec", {}).get("region") or it.get("status", {}).get("region")
        if region:
            out[region] = it["metadata"]["id"]
    return out


def _discover_first_subnet(project_id: str) -> str:
    data = _run_nebius(["vpc", "subnet", "list", "--parent-id", project_id])
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"no subnets in project {project_id}")
    return items[0]["metadata"]["id"]


def _instance_status(instance_id: str) -> tuple[str, str]:
    """Return (state, public_ip). Both may be ''."""
    data = _run_nebius(["compute", "instance", "get", "--id", instance_id])
    status = data.get("status", {})
    state = status.get("state", "") or ""
    nics = status.get("network_interfaces", []) or []
    pip = ""
    if nics:
        addr = (nics[0].get("public_ip_address") or {}).get("address") or ""
        pip = addr
    return state, pip


def _disk_state(disk_id: str) -> str:
    data = _run_nebius(["compute", "disk", "get", "--id", disk_id])
    return (data.get("status", {}).get("state", "") or "")


# -----------------------------------------------------------------------------#
# TCP/22 probe (husk discriminator from B.4)                                   #
# -----------------------------------------------------------------------------#


def _tcp_open(ip: str, port: int = 22, timeout_s: float = 4.0) -> bool:
    if not ip:
        return False
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return True
    except (socket.timeout, OSError):
        return False


# -----------------------------------------------------------------------------#
# Per-variant lifecycle                                                        #
# -----------------------------------------------------------------------------#


def _build_cloud_init(pubkey: str) -> str:
    return f"""#cloud-config
package_update: true
packages:
  - tmux
  - htop
  - rsync
  - jq
users:
  - name: {DEFAULT_USER}
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - {pubkey}
runcmd:
  - [ bash, -lc, "ulimit -n 65536" ]
"""


def _create_disk(v: Variant, disk_size_gib: int, image_family: str) -> str:
    args = [
        "compute", "disk", "create",
        "--parent-id", v.project_id,
        "--name", f"{v.name}-boot",
        "--type", DEFAULT_DISK_TYPE,
        "--size-gibibytes", str(disk_size_gib),
        "--source-image-family-image-family", image_family,
        "--source-image-family-parent-id", public_images_parent(v.region),
    ]
    data = _run_nebius(args)
    disk_id = data.get("metadata", {}).get("id", "") or data.get("resource_id", "")
    if not disk_id:
        # Fallback: lookup by name.
        data = _run_nebius([
            "compute", "disk", "get-by-name",
            "--parent-id", v.project_id,
            "--name", f"{v.name}-boot",
        ])
        disk_id = data.get("metadata", {}).get("id", "")
    if not disk_id:
        raise RuntimeError(f"could not determine disk id for {v.name}")
    return disk_id


def _wait_disk_ready(v: Variant, *, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = _disk_state(v.disk_id)
        if st == "READY":
            return
        if st in ("ERROR", "DELETED"):
            raise RuntimeError(f"disk for {v.label} entered terminal state {st}")
        time.sleep(5)
    raise RuntimeError(f"disk for {v.label} not READY after {timeout_s}s (last: {st})")


def _create_instance(v: Variant, cloud_init_text: str) -> str:
    nic = json.dumps([{
        "name": "eth0",
        "subnet_id": v.subnet_id,
        "ip_address": {},
        "public_ip_address": {},
    }])
    args = [
        "compute", "instance", "create",
        "--parent-id", v.project_id,
        "--name", v.name,
        "--resources-platform", v.platform,
        "--resources-preset", v.preset,
        "--boot-disk-existing-disk-id", v.disk_id,
        "--boot-disk-attach-mode", "read_write",
        "--network-interfaces", nic,
        "--cloud-init-user-data", cloud_init_text,
    ]
    data = _run_nebius(args)
    iid = data.get("metadata", {}).get("id", "") or data.get("resource_id", "")
    if not iid:
        # Fallback: lookup by name.
        data = _run_nebius([
            "compute", "instance", "get-by-name",
            "--parent-id", v.project_id,
            "--name", v.name,
        ])
        iid = data.get("metadata", {}).get("id", "")
    if not iid:
        raise RuntimeError(f"could not determine instance id for {v.name}")
    return iid


def _delete_instance_safe(v: Variant) -> None:
    if not v.instance_id:
        return
    try:
        _run_nebius_void(["compute", "instance", "delete", "--id", v.instance_id])
        log(f"deleted instance {v.instance_id} ({v.label})", prefix="[cleanup] ")
    except Exception as e:
        log(f"WARN: delete instance {v.instance_id} failed: {e}", prefix="[cleanup] ")


def _delete_disk_safe(v: Variant) -> None:
    if not v.disk_id:
        return
    try:
        _run_nebius_void(["compute", "disk", "delete", "--id", v.disk_id])
        log(f"deleted disk {v.disk_id} ({v.label})", prefix="[cleanup] ")
    except Exception as e:
        log(f"WARN: delete disk {v.disk_id} failed: {e}", prefix="[cleanup] ")


def _cleanup_variant(v: Variant) -> None:
    """Delete instance first, then disk (disk is bound to instance until then)."""
    _delete_instance_safe(v)
    # Give Nebius a moment to release the disk binding.
    time.sleep(3)
    _delete_disk_safe(v)


# -----------------------------------------------------------------------------#
# Per-variant worker                                                           #
# -----------------------------------------------------------------------------#

# Process-wide flag flipped when one variant has won; used by other workers
# to short-circuit their poll loops.
_WIN_EVENT = threading.Event()


def _worker(
    v: Variant,
    *,
    cloud_init_text: str,
    disk_size_gib: int,
    image_family: str,
    poll_interval_s: int,
    ssh_grace_s: int,
    dry_run: bool,
    state: RunState,
    save_state: callable,
) -> Variant:
    prefix = f"[{v.label:<28}] "

    if dry_run:
        log("DRY-RUN: would create disk + instance", prefix=prefix)
        v.verdict = VERDICT_PENDING
        v.reason = "dry-run"
        return v

    # ------- 1) Disk -------
    try:
        log("creating boot disk ...", prefix=prefix)
        v.disk_id = _create_disk(v, disk_size_gib, image_family)
        save_state()
        log(f"disk={v.disk_id}", prefix=prefix)
        _wait_disk_ready(v)
        log("disk READY", prefix=prefix)
    except Exception as e:
        v.verdict = VERDICT_FAIL
        v.reason = f"disk-create: {e}"
        log(f"FAIL during disk create: {e}", prefix=prefix)
        save_state()
        return v

    if _WIN_EVENT.is_set():
        v.verdict = VERDICT_CANCELLED
        v.reason = "another variant already won during disk create"
        log("cancelled (another variant won)", prefix=prefix)
        save_state()
        return v

    # ------- 2) Instance create -------
    try:
        log("creating instance ...", prefix=prefix)
        v.instance_id = _create_instance(v, cloud_init_text)
        v.created_at = datetime.now(timezone.utc).isoformat()
        save_state()
        log(f"instance={v.instance_id}", prefix=prefix)
    except Exception as e:
        v.verdict = VERDICT_FAIL
        v.reason = f"instance-create: {e}"
        log(f"FAIL during instance create: {e}", prefix=prefix)
        save_state()
        return v

    # ------- 3) Poll until verdict -------
    running_since: Optional[float] = None
    last_state = ""
    while True:
        if _WIN_EVENT.is_set() and v.verdict == VERDICT_PENDING:
            v.verdict = VERDICT_CANCELLED
            v.reason = "another variant won"
            log("cancelled (another variant won)", prefix=prefix)
            save_state()
            return v

        try:
            state_str, pip = _instance_status(v.instance_id)
        except Exception as e:
            log(f"WARN: status query failed: {e}", prefix=prefix)
            time.sleep(poll_interval_s)
            continue

        if state_str != last_state:
            log(f"state={state_str or '?'}  ip={pip or '-'}", prefix=prefix)
            last_state = state_str
            v.state = state_str
            v.public_ip = pip
            save_state()

        # Terminal failure states.
        if state_str in ("STOPPED", "ERROR"):
            v.verdict = VERDICT_FAIL
            v.reason = f"state={state_str} (Nebius could not allocate)"
            log(f"FAIL: {v.reason}", prefix=prefix)
            save_state()
            return v

        # Win path.
        if state_str == "RUNNING":
            if running_since is None:
                running_since = time.time()
                log(f"RUNNING — probing TCP/22 (up to {ssh_grace_s}s) ...", prefix=prefix)
            if pip and _tcp_open(pip):
                v.verdict = VERDICT_WIN
                v.reason = "RUNNING + TCP/22 open"
                v.won_at = datetime.now(timezone.utc).isoformat()
                v.public_ip = pip
                log(f"WIN  IP={pip}  TCP/22 open", prefix=prefix)
                _WIN_EVENT.set()
                save_state()
                return v
            if time.time() - running_since > ssh_grace_s:
                # Husk: RUNNING but never reachable.
                v.verdict = VERDICT_FAIL
                v.reason = f"husk (RUNNING but TCP/22 closed for {ssh_grace_s}s)"
                log(f"FAIL: {v.reason}", prefix=prefix)
                save_state()
                return v

        time.sleep(poll_interval_s)


# -----------------------------------------------------------------------------#
# Orchestration                                                                #
# -----------------------------------------------------------------------------#


def _validate_pubkey(path: str) -> tuple[str, str]:
    p = Path(path).expanduser()
    if not p.exists():
        raise RuntimeError(
            f"SSH pubkey not found at {p}. Use --ssh-key or generate one with "
            "`ssh-keygen -t ed25519`."
        )
    return str(p), p.read_text().strip()


def _build_variants(
    raw: list[tuple[str, str, str]],
    projects_by_region: dict[str, str],
    instance_prefix: str,
    run_ts: str,
) -> list[Variant]:
    variants: list[Variant] = []
    for region, platform, preset in raw:
        if region not in projects_by_region:
            log(
                f"WARN: skipping {region}/{platform} — no project found in tenant "
                f"for region {region}"
            )
            continue
        # Truncate platform suffix to keep instance name <40 chars.
        platform_short = (
            platform.replace("gpu-", "").replace("-sxm", "").replace("-", "")
        )
        v = Variant(
            region=region,
            platform=platform,
            preset=preset,
            name=f"{instance_prefix}-{region}-{platform_short}-{run_ts}",
            project_id=projects_by_region[region],
        )
        variants.append(v)
    return variants


def _enforce_per_region_cap(variants: list[Variant], cap: int) -> list[list[Variant]]:
    """Return waves of variants such that no wave has > cap variants in same region.

    Variants beyond the per-region cap are bumped to a later wave but their
    relative order in `variants` is preserved.
    """
    waves: list[list[Variant]] = []
    seen_region_count: dict[int, dict[str, int]] = {}
    for v in variants:
        placed = False
        for i, _wave in enumerate(waves):
            counts = seen_region_count.setdefault(i, {})
            if counts.get(v.region, 0) < cap:
                waves[i].append(v)
                counts[v.region] = counts.get(v.region, 0) + 1
                placed = True
                break
        if not placed:
            waves.append([v])
            seen_region_count[len(waves) - 1] = {v.region: 1}
    return waves


def _save_state(state: RunState, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": state.started_at,
        "workdir": state.workdir,
        "ssh_key_path": state.ssh_key_path,
        "cloud_init_path": state.cloud_init_path,
        "variants": [asdict(v) for v in state.variants],
    }
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(state_path)


def _load_state(state_path: Path) -> RunState:
    raw = json.loads(state_path.read_text())
    variants = [Variant(**v) for v in raw["variants"]]
    return RunState(
        started_at=raw["started_at"],
        workdir=raw["workdir"],
        ssh_key_path=raw["ssh_key_path"],
        ssh_pubkey="",
        cloud_init_path=raw["cloud_init_path"],
        variants=variants,
    )


# -----------------------------------------------------------------------------#
# Modes                                                                        #
# -----------------------------------------------------------------------------#


def cmd_provision(args: argparse.Namespace) -> int:
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = Path(f"/tmp/nebius_parallel_{run_ts}")
    workdir.mkdir(parents=True, exist_ok=True)
    state_path = workdir / "state.json"

    log(f"workdir={workdir}")

    # --- Auth + project discovery ---
    # --dry-run must work with no CLI / no network: fall back to placeholder
    # ids (short discovery timeout) so the plan and commands still print.
    offline = False
    try:
        disc_timeout = 15 if args.dry_run else NEBIUS_CMD_TIMEOUT_S
        tenant_id = args.tenant_id or _resolve_tenant_id_t(disc_timeout)
        projects_by_region = _discover_projects_by_region_t(tenant_id, disc_timeout)
    except Exception as e:  # noqa: BLE001
        if not args.dry_run:
            raise
        log(f"DRY-RUN: nebius discovery unavailable ({str(e)[:80]}); using placeholder ids")
        offline = True
        tenant_id = args.tenant_id or "<tenant>"
        projects_by_region = {r: f"<project:{r}>" for r, _, _ in DEFAULT_VARIANTS}
    log(f"tenant={tenant_id}")
    if not projects_by_region:
        log("ERROR: no projects under tenant.")
        return 1
    log(f"projects: " + ", ".join(f"{r}={p}" for r, p in projects_by_region.items()))

    ssh_key_path, ssh_pubkey = _validate_pubkey(args.ssh_key)
    cloud_init_text = _build_cloud_init(ssh_pubkey)
    cloud_init_path = workdir / "cloud-init.yaml"
    cloud_init_path.write_text(cloud_init_text)
    log(f"cloud-init written to {cloud_init_path}")

    # --- Build variants ---
    raw = DEFAULT_VARIANTS
    if args.variants_file:
        raw = [tuple(t) for t in json.loads(Path(args.variants_file).read_text())]
    variants = _build_variants(raw, projects_by_region, args.instance_prefix, run_ts)
    if not variants:
        log("ERROR: no variants resolved (all regions missing projects).")
        return 1

    # --- Resolve subnets up-front ---
    subnet_cache: dict[str, str] = {}
    for v in variants:
        if v.project_id not in subnet_cache:
            subnet_cache[v.project_id] = (
                f"<subnet:{v.region}>" if offline else _discover_first_subnet(v.project_id)
            )
        v.subnet_id = subnet_cache[v.project_id]

    # --- Cap to --max-total ---
    if args.max_total and len(variants) > args.max_total:
        log(f"capping {len(variants)} variants to --max-total={args.max_total}")
        variants = variants[: args.max_total]

    state = RunState(
        started_at=datetime.now(timezone.utc).isoformat(),
        workdir=str(workdir),
        ssh_key_path=ssh_key_path,
        ssh_pubkey=ssh_pubkey,
        cloud_init_path=str(cloud_init_path),
        variants=variants,
    )
    save_state = lambda: _save_state(state, state_path)
    save_state()

    # --- Plan ---
    waves = _enforce_per_region_cap(variants, args.max_per_region)
    log(f"plan: {len(variants)} variants in {len(waves)} wave(s)")
    for i, wave in enumerate(waves):
        log(f"  wave {i + 1}: " + ", ".join(v.label for v in wave))

    if args.dry_run:
        log("DRY-RUN — exiting without launching anything.")
        return 0

    # --- Signal handling for cleanup-on-Ctrl-C ---
    cancelled = threading.Event()

    def _sig(signum, frame):
        log(f"caught signal {signum}; cancelling and cleaning up ...")
        cancelled.set()
        _WIN_EVENT.set()  # short-circuit pollers

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # --- Launch waves ---
    winner: Optional[Variant] = None
    all_results: list[Variant] = []

    for wave_idx, wave in enumerate(waves):
        if winner is not None or cancelled.is_set():
            break
        log(f"=== launching wave {wave_idx + 1}/{len(waves)} ({len(wave)} variants) ===")
        with ThreadPoolExecutor(max_workers=len(wave)) as ex:
            futs = {
                ex.submit(
                    _worker,
                    v,
                    cloud_init_text=cloud_init_text,
                    disk_size_gib=args.disk_size_gib,
                    image_family=args.image_family,
                    poll_interval_s=args.poll_interval,
                    ssh_grace_s=args.ssh_grace,
                    dry_run=False,
                    state=state,
                    save_state=save_state,
                ): v
                for v in wave
            }
            for fut in as_completed(futs):
                v = fut.result()
                all_results.append(v)
                if v.verdict == VERDICT_WIN:
                    winner = v

    # --- Cleanup losers ---
    if winner is not None and not args.keep_losers:
        log(f"=== winner: {winner.label}  IP={winner.public_ip} ===")
        log("cleaning up losers ...")
        losers = [v for v in variants if v is not winner and v.instance_id]
        with ThreadPoolExecutor(max_workers=max(1, len(losers))) as ex:
            list(ex.map(_cleanup_variant, losers))

    save_state()

    # --- Report ---
    print()
    print("=" * 78)
    print(f"Final state file: {state_path}")
    print()
    print(f"{'variant':<32} | {'verdict':<10} | {'state':<10} | reason")
    print("-" * 78)
    for v in variants:
        print(
            f"{v.label:<32} | {v.verdict:<10} | {v.state or '-':<10} | {v.reason or '-'}"
        )
    print()
    if winner is not None:
        print(f"WINNER: {winner.label}")
        print(f"  instance:   {winner.instance_id}")
        print(f"  public ip:  {winner.public_ip}")
        print(f"  region:     {winner.region}")
        print(f"  platform:   {winner.platform}")
        print(f"  ssh:        ssh {DEFAULT_USER}@{winner.public_ip}")
        print()
        print("Stop the instance later with:")
        print(f"  nebius compute instance stop --id {winner.instance_id}")
        print("Delete with:")
        print(f"  nebius compute instance delete --id {winner.instance_id}")
        print(f"  nebius compute disk delete --id {winner.disk_id}")
        return 0

    print("NO WINNER — all variants failed.")
    print("To clean up any leftover resources from this run:")
    print(f"  python {sys.argv[0]} --cleanup {state_path}")
    return 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    state = _load_state(Path(args.cleanup))
    log(f"cleaning up {len(state.variants)} variants from state file")
    with ThreadPoolExecutor(max_workers=max(1, len(state.variants))) as ex:
        list(ex.map(_cleanup_variant, state.variants))
    log("cleanup complete.")
    return 0


def _scan_has_capacity(scan_script: Path, gpus: int, min_on_demand: int) -> tuple[bool, str]:
    """Run nebius_gpu_scan.py and return (has_capacity, summary_line).

    Treats output containing ``(no rows match)`` as no capacity. Otherwise
    returns the first non-header data row as a one-line summary so the loop
    log gives at-a-glance context on which platforms/regions are open.
    """
    proc = subprocess.run(
        [
            sys.executable, str(scan_script),
            "--gpus", str(gpus),
            "--min-on-demand", str(min_on_demand),
        ],
        capture_output=True, text=True, timeout=120,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if "(no rows match)" in out:
        return False, "no on-demand capacity"
    # Extract the first data row (begins with right-justified GPU count).
    for line in out.splitlines():
        s = line.strip()
        if s and s[0].isdigit() and "|" in s:
            return True, s
    # Fallback: scan produced no recognized rows — treat as no capacity to be safe.
    return False, "scan output unparseable"


def cmd_retry_provision(args: argparse.Namespace) -> int:
    """Loop: probe scan -> if capacity, call cmd_provision; else sleep and retry.

    Exits 0 on first winner, 1 on max-attempts exhaustion or interrupt.
    """
    scan_script = Path(args.scan_script).expanduser()
    if not scan_script.is_file():
        log(f"ERROR: scan script not found: {scan_script}")
        return 1

    log(
        f"=== RETRY-UNTIL-SUCCESS LOOP STARTED  "
        f"max_attempts={args.max_attempts}  probe_interval={args.probe_interval}s  "
        f"retry_cooldown={args.retry_cooldown}s  probe_gpus={args.probe_gpus}  "
        f"probe_min_od={args.probe_min_od} ==="
    )

    for i in range(1, args.max_attempts + 1):
        log(f"--- attempt {i}/{args.max_attempts} ---")

        # 1) Gate on scan unless explicitly disabled.
        if not args.no_scan_gate:
            try:
                has_cap, summary = _scan_has_capacity(
                    scan_script, args.probe_gpus, args.probe_min_od
                )
            except Exception as e:  # noqa: BLE001
                log(f"  scan error: {e} (treating as no-capacity)")
                has_cap, summary = False, str(e)
            if not has_cap:
                log(f"  scan: {summary}; sleeping {args.probe_interval}s")
                time.sleep(args.probe_interval)
                continue
            log(f"  scan: capacity detected -> {summary}")

        # 2) Reset the win-event between attempts (it's module-global).
        _WIN_EVENT.clear()

        rc = cmd_provision(args)
        if rc == 0:
            log(f"=== WINNER on attempt {i} ===")
            return 0

        log(f"  provision failed; sleeping {args.retry_cooldown}s before retry")
        time.sleep(args.retry_cooldown)

    log(f"=== EXHAUSTED {args.max_attempts} attempts; giving up ===")
    return 1


# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parallel-provision an 8x GPU node on Nebius across multiple "
            "region/platform variants. First variant to reach RUNNING with a "
            "reachable SSH wins; the others are auto-deleted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tenant-id",
        help="Tenant NID (auto-discovered via `nebius iam tenant list` if omitted).",
    )
    p.add_argument(
        "--ssh-key",
        default=os.path.expanduser("~/.ssh/id_ed25519.pub"),
        help="Path to SSH public key (default: ~/.ssh/id_ed25519.pub).",
    )
    p.add_argument(
        "--variants-file",
        help=(
            "Optional JSON file with a list of [region, platform, preset] triples "
            "to override the default 6-variant catalog."
        ),
    )
    p.add_argument(
        "--max-per-region",
        type=int,
        default=2,
        help="Max concurrent in-flight instances per region (default: 2 — your "
             "tenant quota cap).",
    )
    p.add_argument(
        "--max-total",
        type=int,
        default=6,
        help="Max total in-flight instances across all regions (default: 6).",
    )
    p.add_argument(
        "--disk-size-gib",
        type=int,
        default=DEFAULT_DISK_SIZE_GIB,
        help=f"Boot disk size in GiB (default: {DEFAULT_DISK_SIZE_GIB}).",
    )
    p.add_argument(
        "--image-family",
        default=DEFAULT_IMAGE_FAMILY,
        help=f"Boot image family (default: {DEFAULT_IMAGE_FAMILY}).",
    )
    p.add_argument(
        "--instance-prefix",
        default=DEFAULT_INSTANCE_PREFIX,
        help=f"Prefix for instance names (default: {DEFAULT_INSTANCE_PREFIX}).",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL_S,
        help=f"Status poll interval in seconds (default: {DEFAULT_POLL_INTERVAL_S}).",
    )
    p.add_argument(
        "--ssh-grace",
        type=int,
        default=DEFAULT_SSH_GRACE_S,
        help=(
            f"Seconds after RUNNING to wait for TCP/22 to open before declaring "
            f"a husk (default: {DEFAULT_SSH_GRACE_S})."
        ),
    )
    p.add_argument(
        "--keep-losers",
        action="store_true",
        help="Don't auto-delete losing variants (you'll clean them up manually).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without creating any resources.",
    )
    p.add_argument(
        "--cleanup",
        metavar="STATE_FILE",
        help="Cleanup mode: delete all instances+disks listed in a previous run's "
             "state.json (no provisioning).",
    )
    # ---- retry-until-success mode (loop scan -> provision -> retry) -------------#
    p.add_argument(
        "--retry-until-success",
        action="store_true",
        help="Loop forever (up to --max-attempts): probe nebius_gpu_scan.py for "
             "8-GPU on-demand capacity; when capacity exists, fire the parallel "
             "provisioner; on no winner, sleep --retry-cooldown and retry. Exits "
             "0 on first winner.",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=600,
        help="Cap for --retry-until-success (default 600 ≈ many hours at "
             "--probe-interval 20).",
    )
    p.add_argument(
        "--probe-interval",
        type=int,
        default=20,
        help="Seconds between scan probes when there's no on-demand capacity "
             "(default 20 = ~3 probes/min).",
    )
    p.add_argument(
        "--retry-cooldown",
        type=int,
        default=120,
        help="Seconds to wait after a failed provision attempt before re-probing "
             "(default 120). Lower than typical instance-create timeout to keep "
             "the loop moving.",
    )
    p.add_argument(
        "--probe-gpus",
        type=int,
        default=8,
        help="GPU count passed to scan as --gpus (default 8).",
    )
    p.add_argument(
        "--probe-min-od",
        type=int,
        default=1,
        help="Min on-demand availability passed to scan as --min-on-demand "
             "(default 1). Set to 0 to also accept preemptible-only inventory "
             "(NOTE: this script still creates on-demand instances; preempt "
             "is informational only).",
    )
    p.add_argument(
        "--scan-script",
        default=str(Path(__file__).resolve().parent / "nebius_gpu_scan.py"),
        help="Path to nebius_gpu_scan.py (default: sibling in same dir).",
    )
    p.add_argument(
        "--no-scan-gate",
        action="store_true",
        help="In --retry-until-success mode, skip the scan probe and always "
             "attempt provision. Useful when scan API itself is down.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cleanup:
        return cmd_cleanup(args)
    if args.retry_until_success:
        return cmd_retry_provision(args)
    return cmd_provision(args)


if __name__ == "__main__":
    raise SystemExit(main())
