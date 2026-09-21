#!/usr/bin/env python3
"""
Download GEAR-SONIC model checkpoints and training data from Hugging Face Hub.

Repository: https://huggingface.co/nvidia/GEAR-SONIC

The X2 SONIC policy set SHIPS in-repo (gear_sonic_deploy/models/, git-lfs;
run `git lfs pull` on a fresh clone) -- no download is required for it.
--robot x2 is an OPTIONAL extra: it snapshots $SONIC_X2_HF_REPO (planner
graphs, other policy sets; see MODELS.md) into the cache. Default download
root is the SONIC model cache: $SONIC_HOME
(~/.cache/sonic), with per-robot subtrees g1/, x2/ (and future
embodiments). The X2 subtree alone can be redirected via $SONIC_X2_MODELS.

Usage:
    python download_from_hf.py                    # G1 ONNX models for deployment
    python download_from_hf.py --low-latency      # Low-latency ONNX models
    python download_from_hf.py --training          # PyTorch checkpoint + SMPL data
    python download_from_hf.py --sample            # Sample data only (quick start)
    python download_from_hf.py --output-dir /path  # custom output directory
    python download_from_hf.py --no-planner        # skip planner model
    python download_from_hf.py --robot x2          # full X2 model set -> $SONIC_HOME/x2
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "nvidia/GEAR-SONIC"
# X2 models are bring-your-own (see MODELS.md): point $SONIC_X2_HF_REPO at a
# Hub repo laid out as sonic_policy/ + kplanner_onnx/ (the layout
# install_scripts/setup_x2.sh verifies).
X2_REPO_ID = os.environ.get("SONIC_X2_HF_REPO", "")


def sonic_home() -> Path:
    """Root of the multi-embodiment SONIC model cache (~/.cache/sonic)."""
    return Path(os.environ.get("SONIC_HOME", str(Path.home() / ".cache" / "sonic")))


def x2_models_dir() -> Path:
    """X2 subtree of the cache; $SONIC_X2_MODELS overrides just this subtree."""
    override = os.environ.get("SONIC_X2_MODELS")
    return Path(override) if override else sonic_home() / "x2"

# (filename in HF repo, local destination relative to output_dir)
POLICY_FILES = [
    ("model_encoder.onnx", "policy/release/model_encoder.onnx"),
    ("model_decoder.onnx", "policy/release/model_decoder.onnx"),
    ("observation_config.yaml", "policy/release/observation_config.yaml"),
]

LOW_LATENCY_POLICY_FILES = [
    ("low_latency/model_encoder.onnx", "policy/low_latency/model_encoder.onnx"),
    ("low_latency/model_decoder.onnx", "policy/low_latency/model_decoder.onnx"),
    ("low_latency/observation_config.yaml", "policy/low_latency/observation_config.yaml"),
]

PLANNER_FILE = ("planner_sonic.onnx", "planner/target_vel/V2/planner_sonic.onnx")

TRAINING_FILES = [
    ("sonic_release/last.pt", "sonic_release/last.pt"),
    ("sonic_release/config.yaml", "sonic_release/config.yaml"),
]

LOW_LATENCY_TRAINING_FILES = [
    ("low_latency/last.pt", "low_latency/last.pt"),
    ("low_latency/config.yaml", "low_latency/config.yaml"),
    ("low_latency/model_config.yaml", "low_latency/model_config.yaml"),
]

SMPL_TAR_PARTS_PREFIX = "bones_seed_smpl/bones_seed_smpl.tar.part_"
SMPL_TAR_PARTS = [f"{SMPL_TAR_PARTS_PREFIX}a{c}" for c in "abcdefg"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download GEAR-SONIC checkpoints from Hugging Face Hub"
    )
    parser.add_argument(
        "--robot",
        choices=("g1", "x2"),
        default="g1",
        help=(
            "Which embodiment's checkpoints to download. 'g1' (default) keeps "
            "the original nvidia/GEAR-SONIC behaviour; 'x2' fetches the full "
            "$SONIC_X2_HF_REPO snapshot into $SONIC_HOME/x2 "
            "(or $SONIC_X2_MODELS)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save files. Defaults to the SONIC model cache: "
            "$SONIC_HOME/g1 or $SONIC_HOME/x2 per --robot "
            "(SONIC_HOME defaults to ~/.cache/sonic; "
            "$SONIC_X2_MODELS overrides the x2 subtree)."
        ),
    )
    parser.add_argument(
        "--no-planner",
        action="store_true",
        help="Skip downloading the kinematic planner ONNX model",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Download training checkpoint + SMPL motion data (~30 GB)",
    )
    parser.add_argument(
        "--low-latency",
        action="store_true",
        help=(
            "Download the low-latency SONIC variant. For deployment, files are "
            "placed under gear_sonic_deploy/policy/low_latency/. With --training, "
            "downloads low_latency/last.pt and its configs."
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download sample motion data only (1 walking sequence, ~4 MB)",
    )
    parser.add_argument(
        "--no-smpl",
        action="store_true",
        help="With --training, skip SMPL data download (checkpoint only)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var / run `hf auth login`)",
    )
    return parser.parse_args()


def _ensure_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        return hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Install it with:")
        print("  pip install huggingface_hub")
        sys.exit(1)


def download_file(hf_hub_download, repo_id, hf_filename, local_dest, token=None):
    """Download hf_filename from the Hub and place it at local_dest."""
    print(f"  Downloading {hf_filename} ...", flush=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=hf_filename,
        token=token,
    )
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, local_dest)
    print(f"  -> {local_dest}")


def download_and_extract_smpl(hf_hub_download, repo_id, output_dir, token=None):
    """Download split tar parts and extract SMPL data."""
    parts_dir = output_dir / "bones_seed_smpl"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {len(SMPL_TAR_PARTS)} parts (~30 GB total) ...", flush=True)
    part_paths = []
    for hf_filename in SMPL_TAR_PARTS:
        local_name = Path(hf_filename).name
        local_dest = parts_dir / local_name
        if local_dest.exists():
            print(f"  (cached) {local_name}")
            part_paths.append(local_dest)
            continue
        cached = hf_hub_download(repo_id=repo_id, filename=hf_filename, token=token)
        shutil.copy2(cached, local_dest)
        part_paths.append(local_dest)
        print(f"  Downloaded {local_name}")

    # Reassemble and extract
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting to {data_dir}/smpl_filtered/ ...", flush=True)

    # cat parts | tar xf - -C data/
    cat_cmd = f"cat {parts_dir}/bones_seed_smpl.tar.part_*"
    tar_cmd = f"tar xf - -C {data_dir}"
    result = subprocess.run(
        f"{cat_cmd} | {tar_cmd}",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: Extraction failed: {result.stderr}")
        sys.exit(1)

    # Count extracted files
    smpl_dir = data_dir / "smpl_filtered"
    if smpl_dir.exists():
        n_files = sum(1 for f in smpl_dir.iterdir() if f.suffix == ".pkl")
        print(f"  -> {smpl_dir} ({n_files} PKL files)")
    else:
        print(f"  WARNING: Expected {smpl_dir} but directory not found")

    # Clean up tar parts
    print("  Cleaning up tar parts ...")
    shutil.rmtree(parts_dir)


def download_sample_data(snapshot_download, repo_id, output_dir, token=None):
    """Download sample motion data (1 walking sequence)."""
    print("  Downloading sample data ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns="sample_data/*",
        local_dir=str(output_dir),
        token=token,
    )
    sample_dir = output_dir / "sample_data"
    if sample_dir.exists():
        n_files = sum(1 for _ in sample_dir.rglob("*.pkl"))
        print(f"  -> {sample_dir} ({n_files} PKL files)")


def download_x2(snapshot_download, output_dir, token=None):
    """Download the full X2 model set (sonic policy + kplanner tiers)."""
    if not X2_REPO_ID:
        print(
            "Nothing to download: the X2 SONIC policy set ships in-repo under "
            "gear_sonic_deploy/models/ (git-lfs; run `git lfs pull` if the files "
            "are pointer files) and every launcher picks it automatically.\n"
            "--robot x2 is an optional extra: set $SONIC_X2_HF_REPO to a Hugging "
            "Face repo id holding further X2 artifacts (kplanner_onnx/*.onnx planner "
            "graphs, other sonic_policy/*.onnx sets) to snapshot it into "
            f"{output_dir}. See MODELS.md for the layout and how to produce them."
        )
        return
    print("=" * 60)
    print("  SONIC-X2 — Hugging Face Model Downloader")
    print(f"  Repository : {X2_REPO_ID}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)
    print(
        "\n  Note: if the repo is private and the download 401s, authenticate\n"
        "  via `hf auth login` (or pass --token).\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=X2_REPO_ID,
        local_dir=str(output_dir),
        token=token,
    )
    print("\n" + "=" * 60)
    print("  Done! Files saved under:")
    print(f"  {output_dir}")
    print("=" * 60)


def main():
    args = parse_args()
    if args.sample and args.low_latency:
        print("ERROR: --low-latency cannot be combined with --sample", file=sys.stderr)
        sys.exit(2)

    hf_hub_download, snapshot_download = _ensure_huggingface_hub()

    if args.robot == "x2":
        if args.training or args.sample or args.low_latency:
            print(
                "ERROR: --robot x2 downloads the full $SONIC_X2_HF_REPO "
                "snapshot; the G1 selection flags (--training/--sample/"
                "--low-latency) do not apply.",
                file=sys.stderr,
            )
            sys.exit(2)
        output_dir = args.output_dir if args.output_dir else x2_models_dir()
        download_x2(snapshot_download, output_dir, token=args.token)
        return

    output_dir = args.output_dir if args.output_dir else sonic_home() / "g1"

    print("=" * 60)
    print("  GEAR-SONIC — Hugging Face Model Downloader")
    print(f"  Repository : {REPO_ID}")
    print(f"  Output dir : {output_dir}")
    if args.training:
        if args.low_latency:
            print(f"  Mode       : low-latency training checkpoint")
        else:
            print(f"  Mode       : training (checkpoint + SMPL data)")
    elif args.sample:
        print(f"  Mode       : sample data (quick start)")
    elif args.low_latency:
        print(f"  Mode       : low-latency deployment (ONNX models)")
    else:
        print(f"  Mode       : deployment (ONNX models)")
    print("=" * 60)

    if args.sample:
        print("\n[Sample Data]")
        download_sample_data(snapshot_download, REPO_ID, output_dir, token=args.token)

    elif args.training:
        print("\n[Checkpoint]")
        training_files = LOW_LATENCY_TRAINING_FILES if args.low_latency else TRAINING_FILES
        for hf_filename, local_rel in training_files:
            download_file(
                hf_hub_download, REPO_ID, hf_filename,
                output_dir / local_rel, token=args.token,
            )

        if args.low_latency:
            print("\n[SMPL Motion Data] Skipped (not part of low-latency checkpoint download)")
        elif not args.no_smpl:
            print("\n[SMPL Motion Data]")
            download_and_extract_smpl(hf_hub_download, REPO_ID, output_dir, token=args.token)
        else:
            print("\n[SMPL Motion Data] Skipped (--no-smpl)")

    else:
        print("\n[Policy]")
        policy_files = LOW_LATENCY_POLICY_FILES if args.low_latency else POLICY_FILES
        for hf_filename, local_rel in policy_files:
            download_file(
                hf_hub_download, REPO_ID, hf_filename,
                output_dir / local_rel, token=args.token,
            )

        if not args.no_planner:
            print("\n[Planner]")
            hf_filename, local_rel = PLANNER_FILE
            download_file(
                hf_hub_download, REPO_ID, hf_filename,
                output_dir / local_rel, token=args.token,
            )

    print("\n" + "=" * 60)
    print("  Done! Files saved under:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
