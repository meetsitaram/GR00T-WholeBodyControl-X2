#!/usr/bin/env python
"""Append G1-native step-turn modes to the G1 clip library (turn design v2).

Closes the any2any orientation gap (operator tapes: yaw delivery -20..-32%
vs the old planner; in-place turns slide). The G1 stock library has no clean
in-place-turn template; the paired executed-feasible corpus has 891 bare
idle_turn clips whose G1 SIDE IS NATIVE G1 MOTION — perfectly in-distribution
for the frozen core's VQVAE. This script:

  1. loads two 120 fps G1 turn CSVs, a take and its mirror (soma convention:
     cm + euler-xyz degrees + 29 dof degrees), decimates x4 -> 30 fps,
  2. trims to the steady-turning window (|yaw rate| activity),
  3. computes the library buffers with the G1 converter (exact port of
     build_x2_planner_clips._compute_clip_buffers, qpos dim 36),
  4. appends turn_left / turn_right modes (direction measured from the
     data, not assumed from the _M suffix) to a COPY of G1-clip.ckpt ->
     out/G1-clip-turns.ckpt + a modes sidecar json.

The frozen core's weights are untouched: the library is data.

Source clip: NOT shipped. Pass two G1 SOMA-format CSVs (120 fps, cm +
euler-xyz degrees + 29 dof degrees) of one in-place turn and its mirror,
e.g. a BONES-SEED idle_turn take and its _M file after accepting that
dataset's license (the shipped planner graphs were built that way), or
your own G1 retarget of a turn.
Run:
  PYTHONPATH="$PWD:$PWD/motionbricks" .venv/bin/python \
      motionbricks/scripts/build_g1_turn_clip_library.py \
      --csv <turn.csv> --csv-mirror <turn_M.csv>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "motionbricks")):
    if p not in sys.path:
        sys.path.insert(0, p)

G1_QPOS = 36
SRC_FPS, DST_FPS = 120, 30


def load_g1_csv_qpos(path: Path) -> np.ndarray:
    """Soma G1 CSV -> qpos [T, 36] (m, wxyz quat, rad), 120 fps."""
    d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    T = len(d)
    q = np.zeros((T, G1_QPOS), np.float32)
    q[:, 0:3] = d[:, 1:4] / 100.0
    quat_xyzw = R.from_euler("xyz", d[:, 4:7], degrees=True).as_quat()
    q[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
    q[:, 7:36] = np.deg2rad(d[:, 7:36])
    if not np.isfinite(q).all():
        raise ValueError(f"non-finite values in {path}")
    return q


def steady_turn_window(qpos: np.ndarray, fps: int, thr_dps: float = 15.0,
                       pad_s: float = 0.2) -> tuple[int, int]:
    """Frames where |yaw rate| exceeds thr, padded — trims idle lead-in/tail."""
    yaw = np.unwrap(R.from_quat(qpos[:, [4, 5, 6, 3]]).as_euler("zyx")[:, 0])
    rate = np.abs(np.gradient(yaw) * fps)
    active = np.where(np.degrees(rate) > thr_dps)[0]
    if len(active) < fps // 2:
        raise ValueError("no steady turning found")
    pad = int(pad_s * fps)
    return max(0, active[0] - pad), min(len(qpos), active[-1] + pad)


def net_yaw_deg(qpos: np.ndarray) -> float:
    yaw = np.unwrap(R.from_quat(qpos[:, [4, 5, 6, 3]]).as_euler("zyx")[:, 0])
    return float(np.degrees(yaw[-1] - yaw[0]))


def compute_clip_buffers(qpos_np: np.ndarray, converter, device: str) -> dict:
    """Exact port of build_x2_planner_clips._compute_clip_buffers (qpos 36)."""
    qpos = torch.from_numpy(qpos_np).to(device)[None]
    gjp, gjr = converter.convert_mujoco_qpos_to_motion_transforms(qpos)
    gjp, gjr = gjp[0], gjr[0]
    root_pos = gjp[:, 0, :].clone()
    root_pos[:, 1] = 0.0
    gjp_rel = gjp - root_pos[:, None, :]
    fwd = torch.matmul(
        gjr[:, 0, :, :], torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1)
    ).view(-1, 3)
    fwd_xz = fwd * torch.tensor([1.0, 0.0, 1.0], device=device).view(1, 3)
    norm = fwd_xz.norm(dim=1, keepdim=True)
    if (norm < 1e-5).any():
        raise ValueError("ill-defined heading in turn clip")
    fwd_xz = fwd_xz / norm
    headings = torch.atan2(fwd_xz[:, 0], fwd_xz[:, 2])
    return {
        "global_root_positions": root_pos.cpu(),
        "global_joint_positions": gjp_rel.cpu(),
        "global_joint_rotations": gjr.cpu(),
        "global_headings": headings.cpu(),
        "mujoco_qpos": qpos[0].cpu(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="G1 SOMA-format CSV (120 fps) of an in-place turn")
    ap.add_argument("--csv-mirror", type=Path, required=True,
                    help="the mirrored take (the other turn direction)")
    ap.add_argument("--base-ckpt", type=Path,
                    default=REPO_ROOT / "motionbricks/out/G1-clip.ckpt")
    ap.add_argument("--out-ckpt", type=Path,
                    default=REPO_ROOT / "motionbricks/out/G1-clip-turns.ckpt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from motionbricks.motion_backbone.inference.load_g1_planner import (
        G1PlannerPaths, load_g1_models)
    paths = G1PlannerPaths.default()
    paths.validate()
    _, converter = load_g1_models(paths, device=args.device)

    base = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    n_modes_old = int(base["num_frames_per_clip"].shape[0])
    max_old = int(base["mujoco_qpos"].shape[1])
    J = int(base["global_joint_positions"].shape[2])
    print(f"base library: {n_modes_old} modes, max_frames {max_old}, J={J}")

    new_modes: list[tuple[str, dict]] = []
    for path in (args.csv, args.csv_mirror):
        q120 = load_g1_csv_qpos(path)
        q30 = q120[::4].copy()
        s, e = steady_turn_window(q30, DST_FPS)
        clip = q30[s:e]
        yaw = net_yaw_deg(clip)
        # measured direction, not assumed: positive net yaw = LEFT (ccw)
        name = "turn_left" if yaw > 0 else "turn_right"
        bufs = compute_clip_buffers(clip, converter, args.device)
        rate = yaw / (len(clip) / DST_FPS)
        print(f"{path.name}: {len(q120)}f@120 -> window [{s},{e}) = "
              f"{len(clip)}f@30, net yaw {yaw:+.0f} deg ({rate:+.0f} dps) -> {name}")
        new_modes.append((name, bufs))
    names = [n for n, _ in new_modes]
    assert set(names) == {"turn_left", "turn_right"}, f"both turns same sign: {names}"

    max_new = max(max_old, max(int(b["mujoco_qpos"].shape[0]) for _, b in new_modes))
    n_total = n_modes_old + len(new_modes)
    out = {
        "global_root_positions": torch.zeros(n_total, max_new, 3),
        "global_joint_positions": torch.zeros(n_total, max_new, J, 3),
        "global_joint_rotations": torch.zeros(n_total, max_new, J, 3, 3),
        "global_headings": torch.zeros(n_total, max_new),
        "mujoco_qpos": torch.zeros(n_total, max_new, G1_QPOS),
        "num_frames_per_clip": torch.zeros(n_total, dtype=base["num_frames_per_clip"].dtype),
    }
    for k in ("global_root_positions", "global_joint_positions",
              "global_joint_rotations", "global_headings", "mujoco_qpos"):
        out[k][:n_modes_old, :max_old] = base[k]
    out["num_frames_per_clip"][:n_modes_old] = base["num_frames_per_clip"]
    if "motion_feature" in base:
        print("note: base motion_feature buffer dropped (inference path "
              "recomputes features from transforms; verify if a consumer needs it)")

    mode_idx = {}
    for i, (name, b) in enumerate(new_modes):
        mi = n_modes_old + i
        n = int(b["mujoco_qpos"].shape[0])
        for k in ("global_root_positions", "global_joint_positions",
                  "global_joint_rotations", "global_headings", "mujoco_qpos"):
            out[k][mi, :n] = b[k]
        out["num_frames_per_clip"][mi] = n
        mode_idx[name] = mi

    torch.save(out, args.out_ckpt)
    sidecar = args.out_ckpt.with_suffix(".modes.json")
    sidecar.write_text(json.dumps(
        {"base": str(args.base_ckpt.name), "source_clip": [str(args.csv), str(args.csv_mirror)],
         "modes": mode_idx}, indent=1))
    print(f"wrote {args.out_ckpt} ({n_total} modes, max_frames {max_new}) + {sidecar.name}")
    print(f"turn modes: {mode_idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
