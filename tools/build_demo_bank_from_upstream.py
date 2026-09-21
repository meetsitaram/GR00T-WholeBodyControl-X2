#!/usr/bin/env python3
"""Build the X2 motion banks the launchers expect on a clean clone.

DEFAULT (no flags): the pad/dance bank holds ONLY the X2 Motion-Controller stock
gestures shipped under ``gear_sonic/data/motions/x2_recorded/mc_gestures/``
(vendor gestures recorded on the robot); the planner primitives are the
synthesized lean/crouch/torso bins plus a synthesized idle stand. No motion
derived from any third-party mocap corpus is produced unless you opt in.

``--with-upstream-examples`` ADDITIONALLY retargets the upstream G1 reference clips:

Upstream ships 11 NVIDIA reference motions as G1 body CSVs under
``gear_sonic_deploy/reference/example/<clip>/`` (the output format of
``gear_sonic_deploy/reference/convert_motions.py``: ``joint_pos.csv`` (T, 29),
``body_pos.csv`` (T, 14*3), ``body_quat.csv`` (T, 14*4, wxyz), 50 Hz). This
tool retargets them to the AgiBot X2 Ultra (31 DOF) and emits everything the
sim / deploy launchers expect on a clean clone (see manifest/REGENERATE.md):

  gear_sonic/data/motions/x2_demo_bank.pkl                 motion-lib pkl
  gear_sonic/data/motions/x2_pad_banks.pkl                 pad clip bank (same clips)
  gear_sonic_deploy/data/motions_x2m2/demo_bank/<key>.x2m2 baked clips (pkl_to_x2m2.py)
  gear_sonic/data/motions/x2_planner_primitives.pkl        planner primitives
      (gear_sonic/scripts/build_x2_planner_primitives.py --source x2_demo_bank.pkl)

Retargeting
-----------
The original X2 corpus was produced with an IK retargeter that is not part of
this repository. This tool ships a self-contained JOINT-MAP retarget instead:
G1 and X2 share the same 29 joint semantics with identical hinge axes
(pitch = y, roll = x, yaw = z); the X2 adds a 2-DOF head. Per frame:

  * joint angles are copied by NAME from the IsaacLab (breadth-first) column
    order of the CSVs into the X2 MuJoCo (depth-first) order, the elbow sign
    is flipped (X2 elbow flexes negative), the head is held at 0, every joint
    is clipped to the X2 range;
  * root heading/xy come from the G1 pelvis, xy scaled by the leg-length ratio;
  * root z is re-grounded with a forward-kinematics pass over the X2 MJCF so
    the stance foot sits at the corpus baseline (0.079 m link height) and
    sustained foot-float is removed (root-only correction, joints untouched);
  * the sanity gates of x2_gmr_to_motion_lib.py (tilt / jump / hip pitch /
    yaw rate) are reported per clip.

The result is a kinematically plausible X2 bank for the sim stack, pad
dances and planner primitives; it is NOT a dynamics-checked corpus. Clips
that trip a sanity gate are still written (flagged in the summary) unless
``--strict`` is passed.

Environment
-----------
Pure python: numpy, scipy, joblib, pyyaml (the repo ``.venv`` has them; no
MuJoCo needed — the FK reads the MJCF kinematic tree directly). The x2m2 and
primitives stages shell out to ``gear_sonic/scripts/pkl_to_x2m2.py`` and
``gear_sonic/scripts/build_x2_planner_primitives.py`` with the same
interpreter (``--python`` to override), so ``gear_sonic`` must be importable
from the repo root (run from a clone; the scripts insert the repo root on
``sys.path`` themselves).

Usage (repo root)::

    tools/build_demo_bank_from_upstream.sh             # everything
    .venv/bin/python tools/build_demo_bank_from_upstream.py --skip-primitives
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLES_DIR = REPO_ROOT / "gear_sonic_deploy/reference/example"
X2_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
G1_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
OUT_BANK = REPO_ROOT / "gear_sonic/data/motions/x2_demo_bank.pkl"
OUT_PAD = REPO_ROOT / "gear_sonic/data/motions/x2_pad_banks.pkl"
OUT_X2M2_DIR = REPO_ROOT / "gear_sonic_deploy/data/motions_x2m2/demo_bank"
OUT_PRIMS = REPO_ROOT / "gear_sonic/data/motions/x2_planner_primitives.pkl"
GESTURES_DIR = REPO_ROOT / "gear_sonic/data/motions/x2_recorded/mc_gestures"
RECIPES = REPO_ROOT / "gear_sonic/data/motions/x2_planner_primitives_recipes.yaml"
BINS = REPO_ROOT / "gear_sonic/data/motions/x2_planner_bins.yaml"

UPSTREAM_FPS = 50.0
# Stance-foot link z at standing in the X2 motion-lib convention (measured
# across the retargeted corpus; see gear_sonic/scripts/ground_feet_clips.py).
FOOT_BASELINE_Z = 0.079

# G1 29-DOF column order of the upstream reference CSVs. convert_motions.py
# dumps IsaacLab's ``joint_pos`` verbatim, and IsaacLab orders joints
# breadth-first over the kinematic tree (left/right/waist interleaved), NOT
# in MJCF depth-first order -- the list below is the G1 IsaacLab order from
# gear_sonic/envs/env_utils/joint_utils.py. (The ``_body_indexes`` in each
# clip's metadata.txt, [0 4 10 18 5 11 19 9 16 22 28 17 23 29], are the same
# breadth-first body numbering.)
G1_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
# X2 Ultra 31-DOF MuJoCo order (x2_ultra.xml).
X2_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
]
# Joints whose sign convention differs between the two robots (X2 range is the
# mirror of G1's: elbow [-2.36, 0] vs [-1.05, 2.09]).
SIGN_FLIP = {"left_elbow_joint": -1.0, "right_elbow_joint": -1.0}
X2_NUM_DOF = 31
X2_NUM_BODIES = 32
SANITY = {  # value thresholds ported from x2_gmr_to_motion_lib.sanity_report
    "tilt_rad": 1.0, "jump_rad": 1.0, "hip_pitch_rad": 2.0, "yaw_rate_rad_s": 6.0,
}


# ---------------------------------------------------------------------------
# Minimal MJCF kinematic tree + FK (hinge joints, no MuJoCo import)
# ---------------------------------------------------------------------------


def _vec(s: str | None, default) -> np.ndarray:
    return np.array([float(v) for v in s.split()], dtype=np.float64) if s else np.array(default, dtype=np.float64)


def _body_rot(el: ET.Element) -> R:
    if el.get("quat"):
        w, x, y, z = _vec(el.get("quat"), [1, 0, 0, 0])
        return R.from_quat([x, y, z, w])
    if el.get("euler"):
        return R.from_euler("xyz", _vec(el.get("euler"), [0, 0, 0]))
    if el.get("axisangle"):
        ax = _vec(el.get("axisangle"), [0, 0, 1, 0])
        return R.from_rotvec(ax[:3] / max(np.linalg.norm(ax[:3]), 1e-12) * ax[3])
    return R.identity()


class KinTree:
    """Bodies in MJCF document order with parent index, local pos/rot and the
    hinge joint (name, axis, range) attached to each body (at most one)."""

    def __init__(self, mjcf: Path):
        root = ET.parse(mjcf).getroot()
        world = root.find("worldbody")
        if world is None:
            raise ValueError(f"{mjcf}: no <worldbody>")
        self.names: list[str] = []
        self.parent: list[int] = []
        self.pos: list[np.ndarray] = []
        self.rot: list[R] = []
        self.joint: list[tuple[str, np.ndarray, tuple[float, float]] | None] = []
        self.joint_body: dict[str, int] = {}
        self.joint_range: dict[str, tuple[float, float]] = {}

        def visit(el: ET.Element, parent: int) -> None:
            for child in el:
                if child.tag != "body":
                    continue
                idx = len(self.names)
                self.names.append(child.get("name", f"body{idx}"))
                self.parent.append(parent)
                self.pos.append(_vec(child.get("pos"), [0, 0, 0]))
                self.rot.append(_body_rot(child))
                j = None
                for jel in child.findall("joint"):
                    if jel.get("type", "hinge") != "hinge":
                        continue
                    name = jel.get("name", "")
                    axis = _vec(jel.get("axis"), [0, 0, 1])
                    rng = tuple(_vec(jel.get("range"), [-np.pi, np.pi])) if jel.get("range") else (-np.inf, np.inf)
                    if j is not None:
                        raise ValueError(f"{mjcf}: body {self.names[idx]} has >1 hinge joint")
                    j = (name, axis / np.linalg.norm(axis), rng)
                    self.joint_body[name] = idx
                    self.joint_range[name] = rng
                self.joint.append(j)
                visit(child, idx)

        visit(world, -1)
        self.index = {n: i for i, n in enumerate(self.names)}

    def fk(self, root_pos: np.ndarray, root_rot: R, q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """World positions (T, 3) of every body for a (T,) joint-angle dict."""
        T = root_pos.shape[0]
        world_pos: list[np.ndarray] = []
        world_rot: list[R] = []
        for i, name in enumerate(self.names):
            p = self.parent[i]
            if p < 0:
                pos = root_pos
                rot = root_rot
            else:
                pos = world_pos[p] + world_rot[p].apply(np.broadcast_to(self.pos[i], (T, 3)))
                rot = world_rot[p] * self.rot[i]
            j = self.joint[i]
            if j is not None and j[0] in q:
                rot = rot * R.from_rotvec(np.outer(q[j[0]], j[1]))
            world_pos.append(pos)
            world_rot.append(rot)
        return {n: world_pos[i] for i, n in enumerate(self.names)}

    def zero_pose_distance(self, a: str, b: str) -> float:
        one = np.zeros((1, 3))
        out = self.fk(one, R.identity(), {})
        return float(np.linalg.norm(out[a][0] - out[b][0]))


# ---------------------------------------------------------------------------
# Upstream clip I/O
# ---------------------------------------------------------------------------


def load_upstream_clip(d: Path) -> dict:
    jp = np.loadtxt(d / "joint_pos.csv", delimiter=",", skiprows=1, dtype=np.float64)
    bp = np.loadtxt(d / "body_pos.csv", delimiter=",", skiprows=1, dtype=np.float64)
    bq = np.loadtxt(d / "body_quat.csv", delimiter=",", skiprows=1, dtype=np.float64)
    if jp.ndim != 2 or jp.shape[1] != 29:
        raise ValueError(f"{d.name}: joint_pos.csv shape {jp.shape}, expected (T, 29)")
    T = jp.shape[0]
    bp = bp.reshape(T, -1, 3)
    bq = bq.reshape(T, -1, 4)  # wxyz (IsaacLab)
    root_pos = bp[:, 0, :]
    root_quat_xyzw = bq[:, 0, [1, 2, 3, 0]]
    fps = UPSTREAM_FPS
    meta = d / "metadata.txt"
    if meta.is_file():
        for line in meta.read_text().splitlines():
            if line.lower().startswith("fps"):
                try:
                    fps = float(line.split(":")[1])
                except (IndexError, ValueError):
                    pass
    return {"dof": jp, "root_pos": root_pos, "root_quat_xyzw": root_quat_xyzw, "fps": fps}


# ---------------------------------------------------------------------------
# Retarget
# ---------------------------------------------------------------------------


def joint_map(g1_dof: np.ndarray, x2_ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    T = g1_dof.shape[0]
    out = np.zeros((T, X2_NUM_DOF), dtype=np.float64)
    g1_idx = {n: i for i, n in enumerate(G1_JOINTS)}
    for j, name in enumerate(X2_JOINTS):
        if name not in g1_idx:
            continue  # head joints: held at 0
        v = g1_dof[:, g1_idx[name]] * SIGN_FLIP.get(name, 1.0)
        lo, hi = x2_ranges.get(name, (-np.inf, np.inf))
        out[:, j] = np.clip(v, lo, hi)
    return out


def reground_root(tree: KinTree, dof: np.ndarray, root_pos: np.ndarray, root_rot: R,
                  fps: float) -> tuple[np.ndarray, dict]:
    """Shift root z so the stance foot rests at FOOT_BASELINE_Z; remove
    sustained float (rolling minimum over ~1 s) and sustained penetration."""
    q = {name: dof[:, j] for j, name in enumerate(X2_JOINTS)}
    feet = [n for n in ("left_ankle_roll_link", "right_ankle_roll_link",
                        "left_toe_link", "right_toe_link") if n in tree.index]
    out = tree.fk(root_pos, root_rot, q)
    minz = np.min(np.stack([out[n][:, 2] for n in feet], axis=1), axis=1)
    shift = float(np.percentile(minz, 5) - FOOT_BASELINE_Z)
    z = root_pos[:, 2] - shift
    minz = minz - shift
    win = max(1, int(round(1.0 * fps)))
    smooth = max(1, int(round(0.3 * fps)))
    float_ = np.clip(minz - FOOT_BASELINE_Z, 0.0, None)
    sustained = uniform_filter1d(minimum_filter1d(float_, size=win, mode="nearest"), size=smooth, mode="nearest")
    z = z - sustained
    minz = minz - sustained
    pen = np.clip((FOOT_BASELINE_Z - 0.02) - minz, 0.0, None)
    sustained_pen = uniform_filter1d(minimum_filter1d(pen, size=win, mode="nearest"), size=smooth, mode="nearest")
    z = z + sustained_pen
    rp = root_pos.copy()
    rp[:, 2] = z
    stats = {
        "z_shift_m": -shift,
        "float_removed_max_m": float(sustained.max()),
        "penetration_fixed_max_m": float(sustained_pen.max()),
    }
    return rp, stats


def sanity_report(root_rot: R, dof: np.ndarray, fps: float) -> dict:
    rpy = root_rot.as_euler("xyz")
    tilt = float(np.abs(rpy[:, :2]).max())
    jump = float(np.abs(np.diff(dof, axis=0)).max()) if len(dof) > 1 else 0.0
    hip_pitch = float(np.abs(dof[:, [0, 6]]).max())
    yaw = np.unwrap(rpy[:, 2])
    yaw_rate = float(np.abs(np.diff(yaw)).max() * fps) if len(yaw) > 1 else 0.0
    vals = {"tilt_rad": tilt, "jump_rad": jump, "hip_pitch_rad": hip_pitch, "yaw_rate_rad_s": yaw_rate}
    vals["pass"] = all(vals[k] <= thr for k, thr in SANITY.items())
    return vals


def make_entry(dof: np.ndarray, root_pos: np.ndarray, root_rot: R, fps: float, source: str) -> dict:
    """Motion-lib entry in the schema of g1_captures_to_x2_motion_pkl._x2_csv_to_entry."""
    T = dof.shape[0]
    axes = np.array(_X2_AXES, dtype=np.float32)  # filled from the MJCF in main()
    pa = np.zeros((T, X2_NUM_BODIES, 3), np.float32)
    pa[:, 1:X2_NUM_DOF + 1, :] = axes[None] * dof[:, :, None].astype(np.float32)
    pa[:, 0, :] = root_rot.as_rotvec().astype(np.float32)
    return {
        "root_trans_offset": root_pos.astype(np.float32),
        "root_rot": root_rot.as_quat().astype(np.float32),  # xyzw
        "dof": dof.astype(np.float32),
        "pose_aa": pa,
        "smpl_joints": np.zeros((T, 24, 3), np.float32),
        "fps": int(round(fps)),
        "source": source,
        "retarget": "tools/build_demo_bank_from_upstream.py joint_map_v1",
    }


_X2_AXES: list[np.ndarray] = []


def retarget_clip(name: str, clip: dict, x2: KinTree, xy_scale: float) -> tuple[dict, dict]:
    dof = joint_map(clip["dof"], x2.joint_range)
    rot = R.from_quat(clip["root_quat_xyzw"])
    root = clip["root_pos"].copy()
    root[:, :2] = (root[:, :2] - root[0, :2]) * xy_scale + root[0, :2]
    root, gstats = reground_root(x2, dof, root, rot, clip["fps"])
    entry = make_entry(dof, root, rot, clip["fps"], f"upstream:gear_sonic_deploy/reference/example/{name}")
    san = sanity_report(rot, dof, clip["fps"])
    deg = np.degrees
    j = {n: i for i, n in enumerate(X2_JOINTS)}
    legs = {
        "l_knee_deg": (float(deg(dof[:, j["left_knee_joint"]].min())), float(deg(dof[:, j["left_knee_joint"]].max()))),
        "r_knee_deg": (float(deg(dof[:, j["right_knee_joint"]].min())), float(deg(dof[:, j["right_knee_joint"]].max()))),
        "l_hip_pitch_deg": (float(deg(dof[:, j["left_hip_pitch_joint"]].min())), float(deg(dof[:, j["left_hip_pitch_joint"]].max()))),
        "r_hip_pitch_deg": (float(deg(dof[:, j["right_hip_pitch_joint"]].min())), float(deg(dof[:, j["right_hip_pitch_joint"]].max()))),
    }
    # Left/right knee excursions of a two-legged motion should be comparable;
    # a >3x asymmetry is the signature of a scrambled joint map.
    lk = legs["l_knee_deg"][1] - legs["l_knee_deg"][0]
    rk = legs["r_knee_deg"][1] - legs["r_knee_deg"][0]
    legs["knee_symmetric"] = bool(max(lk, rk) <= 3.0 * max(min(lk, rk), 1.0))
    return entry, {**gstats, **san, **legs, "frames": int(dof.shape[0]), "seconds": dof.shape[0] / clip["fps"]}


# ---------------------------------------------------------------------------
# Downstream stages
# ---------------------------------------------------------------------------


def stage_x2m2(python: str, bank_pkl: Path, bank: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "gear_sonic/scripts/pkl_to_x2m2.py"
    written = []
    for key, entry in bank.items():
        dur = entry["dof"].shape[0] / float(entry["fps"])
        out = out_dir / f"{key}.x2m2"
        cmd = [python, str(script), "--pkl", str(bank_pkl), "--key", key,
               "--start-s", "0", "--dur-s", f"{dur + 1.0:.3f}", "--out", str(out), "--no-hand-sidecar"]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        written.append(out)
    return written


def stage_primitives(python: str, bank_pkl: Path, out_pkl: Path, report: Path) -> None:
    cmd = [python, "-m", "gear_sonic.scripts.build_x2_planner_primitives",
           "--source", str(bank_pkl), "--recipes", str(RECIPES), "--bins", str(BINS),
           "--out-pkl", str(out_pkl), "--out-report", str(report), "--skip-missing-sources"]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    ensure_idle_stand(out_pkl)


def ensure_idle_stand(prims_pkl: Path, seconds: float = 1.5, fps: float = 50.0) -> None:
    """The recipes curate ``idle_stand`` from a mocap corpus clip; when that
    clip is not in the source bank the bin is skipped, but the planner state
    machine (``HeuristicPlanner``) requires it. Synthesize a static hold of
    the trained stand pose in its place (same schema as built primitives)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from gear_sonic.utils.planner.constants import (  # noqa: E402
        DEFAULT_PELVIS_Z_M, DEFAULT_STAND_POSE_NP, IDENTITY_ROOT_QUAT_XYZW,
    )
    prims = joblib.load(prims_pkl)
    if "idle_stand" in prims:
        return
    n = int(round(seconds * fps))
    prims["idle_stand"] = {
        "dof": np.repeat(DEFAULT_STAND_POSE_NP[None].astype(np.float32), n, axis=0),
        "root_rot_xyzw": np.repeat(np.asarray(IDENTITY_ROOT_QUAT_XYZW, np.float32)[None], n, axis=0),
        "root_trans": np.repeat(np.array([[0.0, 0.0, DEFAULT_PELVIS_Z_M]], np.float32), n, axis=0),
        "fps": float(fps),
        "source_pkl": "synth:default_stand_pose",
        "motion_key": "synth:idle_stand_hold",
        "start_frame": 0,
        "n_frames": n,
        "partial": False,
        "pinned": True,
        "freeze_arms_to_default": True,
        "recipe_family": "idle",
        "recipe_ops": ["synth:hold DEFAULT_STAND_POSE (tools/build_demo_bank_from_upstream.py)"],
        "recipe_sources": ["synth:idle_stand_hold"],
    }
    joblib.dump(prims, prims_pkl, compress=3)
    print(f"[bank] added synthesized idle_stand hold ({n}f @ {fps:g}fps) to {prims_pkl.name}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-dir", type=Path, default=EXAMPLES_DIR)
    ap.add_argument("--x2-mjcf", type=Path, default=X2_MJCF)
    ap.add_argument("--g1-mjcf", type=Path, default=G1_MJCF)
    ap.add_argument("--out-bank", type=Path, default=OUT_BANK)
    ap.add_argument("--out-pad", type=Path, default=OUT_PAD)
    ap.add_argument("--x2m2-dir", type=Path, default=OUT_X2M2_DIR)
    ap.add_argument("--out-primitives", type=Path, default=OUT_PRIMS)
    ap.add_argument("--with-upstream-examples", action="store_true",
                    help="also retarget the upstream G1 reference clips into the bank (off by default)")
    ap.add_argument("--gestures-dir", type=Path, default=GESTURES_DIR,
                    help="X2 MC stock-gesture captures merged into the banks (default: shipped dir; pass '' to skip)")
    ap.add_argument("--only", nargs="*", default=None, help="clip folder names to include (default: all)")
    ap.add_argument("--xy-scale", type=float, default=None,
                    help="root xy scale G1->X2 (default: X2/G1 hip-to-ankle length ratio from the MJCFs)")
    ap.add_argument("--skip-x2m2", action="store_true")
    ap.add_argument("--skip-primitives", action="store_true")
    ap.add_argument("--strict", action="store_true", help="drop clips that fail a sanity gate")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the pkl_to_x2m2 / primitives stages")
    args = ap.parse_args(argv)
    t0 = time.time()

    for p in (args.x2_mjcf, args.g1_mjcf):
        if not p.is_file():
            raise SystemExit(f"MJCF not found: {p}")
    clip_dirs: list[Path] = []
    if args.with_upstream_examples:
        if not args.examples_dir.is_dir():
            raise SystemExit(f"upstream example clips not found: {args.examples_dir}")
        clip_dirs = sorted(d for d in args.examples_dir.iterdir() if (d / "joint_pos.csv").is_file())
        if args.only:
            clip_dirs = [d for d in clip_dirs if d.name in set(args.only)]
        if not clip_dirs:
            raise SystemExit(f"no clips with joint_pos.csv under {args.examples_dir}")
    else:
        print("[bank] upstream reference clips NOT included (pass --with-upstream-examples to add them)")

    x2 = KinTree(args.x2_mjcf)
    g1 = KinTree(args.g1_mjcf)
    missing = [n for n in X2_JOINTS if n not in x2.joint_body]
    if missing:
        raise SystemExit(f"{args.x2_mjcf}: joints missing from the MJCF: {missing}")
    for name in X2_JOINTS:
        body = x2.joint_body[name]
        _X2_AXES.append(x2.joint[body][1])
    if args.xy_scale is None:
        leg_x2 = x2.zero_pose_distance("left_hip_pitch_link", "left_ankle_roll_link")
        leg_g1 = g1.zero_pose_distance("left_hip_pitch_link", "left_ankle_roll_link")
        xy_scale = leg_x2 / leg_g1
        print(f"[bank] leg length X2 {leg_x2:.3f} m / G1 {leg_g1:.3f} m -> root xy scale {xy_scale:.3f}")
    else:
        xy_scale = args.xy_scale

    bank: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    for d in clip_dirs:
        clip = load_upstream_clip(d)
        entry, st = retarget_clip(d.name, clip, x2, xy_scale)
        flag = "OK  " if st["pass"] else "GATE"
        print(f"[bank] {flag} {d.name:42s} {st['frames']:5d}f {st['seconds']:5.1f}s "
              f"tilt {st['tilt_rad']:.2f} jump {st['jump_rad']:.2f} hip {st['hip_pitch_rad']:.2f} "
              f"yawrate {st['yaw_rate_rad_s']:.2f}  z{st['z_shift_m']:+.3f} float-{st['float_removed_max_m']:.3f}")
        print(f"[bank]      knee L [{st['l_knee_deg'][0]:6.1f},{st['l_knee_deg'][1]:6.1f}] "
              f"R [{st['r_knee_deg'][0]:6.1f},{st['r_knee_deg'][1]:6.1f}]  "
              f"hip_pitch L [{st['l_hip_pitch_deg'][0]:6.1f},{st['l_hip_pitch_deg'][1]:6.1f}] "
              f"R [{st['r_hip_pitch_deg'][0]:6.1f},{st['r_hip_pitch_deg'][1]:6.1f}] deg"
              f"{'' if st['knee_symmetric'] else '  ** L/R KNEE ASYMMETRY **'}")
        if args.strict and not st["pass"]:
            print(f"[bank]      dropped (--strict)")
            continue
        bank[d.name] = entry
        stats[d.name] = st
    gdir = Path(args.gestures_dir) if str(args.gestures_dir) else None
    if gdir and gdir.is_dir():
        n_g = 0
        for f in sorted(gdir.glob("*.pkl")):
            for key, entry in joblib.load(f).items():
                if key in bank:
                    continue
                bank[key] = entry
                dof = np.asarray(entry["dof"]); fps = float(entry.get("fps", 30))
                n_g += 1
                print(f"[bank] GEST {key:42s} {dof.shape[0]:5d}f {dof.shape[0] / fps:5.1f}s @ {fps:g}fps  "
                      f"(X2 MC stock gesture, {f.name})")
        print(f"[bank] merged {n_g} X2 MC stock gestures from {gdir}")
    elif gdir:
        print(f"[bank] note: gesture captures not found at {gdir} (nothing merged)")
    if not bank:
        raise SystemExit("no clips survived")
    # Upstream ``<clip>_M`` folders are left/right mirrors: the mirrored clip's
    # LEFT leg must trace the original's RIGHT leg (and vice versa).
    for key in sorted(bank):
        if not key.endswith("_M") or key[:-2] not in stats:
            continue
        a, b = stats[key[:-2]], stats[key]
        ok = (abs(a["l_knee_deg"][1] - b["r_knee_deg"][1]) < 2.0
              and abs(a["r_knee_deg"][1] - b["l_knee_deg"][1]) < 2.0)
        print(f"[bank] mirror {key[:-2]} <-> _M: L/R knee swap {'consistent' if ok else '** INCONSISTENT **'}")

    args.out_bank.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bank, args.out_bank, compress=3)
    total_s = sum(v["dof"].shape[0] / v["fps"] for v in bank.values())
    print(f"[bank] wrote {args.out_bank} ({len(bank)} clips, {total_s:.0f}s, "
          f"{args.out_bank.stat().st_size / 1e6:.1f} MB)")
    # Pad bank: the same clips (every upstream reference is a return-to-stand
    # sequence short enough for a pad button); kept as a separate file so the
    # pad path can be curated independently of the planner source bank.
    joblib.dump(bank, args.out_pad, compress=3)
    print(f"[bank] wrote {args.out_pad} ({len(bank)} clips)")
    manifest = args.out_bank.with_suffix(".json")
    manifest.write_text(json.dumps({"clips": stats, "gestures": [k for k in bank if k not in stats],
                                    "xy_scale": xy_scale, "retarget": "joint_map_v1", "fps": UPSTREAM_FPS}, indent=2))

    if not args.skip_x2m2:
        written = stage_x2m2(args.python, args.out_bank, bank, args.x2m2_dir)
        print(f"[bank] wrote {len(written)} x2m2 clips -> {args.x2m2_dir}")
    if not args.skip_primitives:
        report = args.out_primitives.with_name(args.out_primitives.stem + "_recipes_report.md")
        stage_primitives(args.python, args.out_bank, args.out_primitives, report)
        print(f"[bank] wrote {args.out_primitives}")
    print(f"[bank] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
