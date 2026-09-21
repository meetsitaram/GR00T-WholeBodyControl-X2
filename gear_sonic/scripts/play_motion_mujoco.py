#!/usr/bin/env python3
"""Kinematic playback of a motion-lib PKL (or baked X2M2) in the MuJoCo viewer.

No physics, no policy: at each frame we write the recorded ``root_trans_offset``,
``root_rot``, and per-joint ``dof`` straight into ``mj_data.qpos``, call
``mj_kinematics`` to refresh body transforms, and sync the viewer. This is
the cleanest way to *see* what a reference motion actually contains
(useful for sanity-checking that a "standing" clip really stands still and
that joint ordering / quaternion convention are right).

Usage:
    # Source PKL (full motion-lib dict, has root translation):
    python gear_sonic/scripts/play_motion_mujoco.py \
        --motion gear_sonic/data/motions/x2_demo_bank.pkl

    # Baked X2M2 (the binary the C++ deploy actually loads). The X2M2
    # bake drops root_trans_offset (PklMotionReference in C++ doesn't
    # consume it), so we render every frame at a fixed standing-height
    # pelvis (override with --fixed-root-z 0.91 or whatever the MJCF
    # spawns at).
    python gear_sonic/scripts/play_motion_mujoco.py \
        --motion gear_sonic_deploy/data/motions_x2m2/demo_bank/<clip>.x2m2

Optional:
    --mjcf PATH        Override MJCF (defaults to gear_sonic x2_ultra.xml)
    --speed 1.0        Playback speed multiplier (e.g. 0.25 for slow-mo)
    --loop / --no-loop Loop at end (default: loop)
    --start-frame N    Start from frame N (default 0)
    --fixed-root-z Z   (X2M2 only) z-height of the floating-base anchor
                       in metres. Default 0.91 (canonical X2 stand
                       pelvis height). Pass --fixed-root-z auto to read
                       the MJCF's <freejoint> default pose, if present.

Controls:
    SPACE - viewer pause/resume (the script keeps stepping; you just freeze
            the camera ergonomics provided by mujoco.viewer).
    Esc   - close the viewer / exit.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MJCF = str(
    GEAR_SONIC_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
)

NUM_DOFS = 31

# Must match gear_sonic_deploy/scripts/export_motion_for_deploy.py +
# reference_motion.hpp. Keep this duplicated rather than importing the
# deploy script so this tool stays usable in a fresh checkout without
# colcon-built deploy aux deps.
X2M2_MAGIC = 0x58324D32  # "X2M2" little-endian
# Default pelvis z (m) used when an X2M2 is loaded; the bake drops
# root_trans_offset and the X2 normally spawns ~0.91 m up at MJCF init.
# Operators can override via --fixed-root-z.
X2M2_DEFAULT_PELVIS_Z = 0.91


def _load_x2m2(path: Path) -> dict:
    """Read an X2M2 binary into the same dict-of-arrays shape PKL loads use.

    The .x2m2 file format (see reference_motion.hpp / export_motion_for_deploy
    .py) is a deliberately compact little-endian binary. It does NOT carry
    root_trans_offset (PklMotionReference in C++ doesn't use it), so we
    synthesize a constant root translation at X2M2_DEFAULT_PELVIS_Z to keep
    the viewer happy. Caller can override via the wrapping playback loop.
    """
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(
            f"X2M2 {path} too small ({len(raw)} bytes); header alone is 16."
        )
    magic, n_frames, n_dofs = struct.unpack_from("<III", raw, 0)
    if magic != X2M2_MAGIC:
        raise ValueError(
            f"X2M2 {path}: bad magic 0x{magic:08X}, expected 0x{X2M2_MAGIC:08X}."
        )
    if n_dofs != NUM_DOFS:
        raise ValueError(
            f"X2M2 {path}: num_dofs={n_dofs} but expected {NUM_DOFS}."
        )
    (fps,) = struct.unpack_from("<d", raw, 12)
    per_frame_bytes = 8 * (NUM_DOFS + 4)
    expected = 4 * 3 + 8 + n_frames * per_frame_bytes
    if len(raw) != expected:
        raise ValueError(
            f"X2M2 {path}: byte count {len(raw)} != expected {expected} "
            f"(n_frames={n_frames}, fps={fps})."
        )
    dof = np.empty((n_frames, NUM_DOFS), dtype=np.float64)
    rot = np.empty((n_frames, 4), dtype=np.float64)
    offset = 20  # 12 (header ints) + 8 (fps)
    for i in range(n_frames):
        dof[i] = np.frombuffer(raw, dtype=np.float64,
                               count=NUM_DOFS, offset=offset)
        offset += 8 * NUM_DOFS
        rot[i] = np.frombuffer(raw, dtype=np.float64, count=4, offset=offset)
        offset += 8 * 4
    return {
        path.stem: {
            "dof": dof,
            "root_rot": rot,
            # Synthesized root translation -- constant z, zero x/y.
            "root_trans_offset": np.tile(
                np.array([0.0, 0.0, X2M2_DEFAULT_PELVIS_Z]),
                (n_frames, 1),
            ),
            "fps": fps,
        }
    }


def _take_first_motion(pkl_data: dict, key: str | None = None):
    """First clip, or the first whose name contains ``key``.

    Multi-clip corpora (33k+ entries) are unusable without selection: the
    caller cannot reach anything but entry 0.
    """
    if not isinstance(pkl_data, dict) or not pkl_data:
        raise ValueError("PKL is not a non-empty dict-of-motions")
    if key:
        hits = [k for k in pkl_data if key.lower() in k.lower()]
        if not hits:
            raise KeyError(
                f"no clip matching {key!r}; {len(pkl_data)} clips in file"
            )
        if len(hits) > 1:
            print(f"[play_motion] {len(hits)} clips match {key!r}; "
                  f"using {hits[0]}", flush=True)
        return hits[0], pkl_data[hits[0]]
    name = next(iter(pkl_data))
    return name, pkl_data[name]


def _validate(motion: dict) -> None:
    for key in ("dof", "root_rot", "root_trans_offset", "fps"):
        if key not in motion:
            raise KeyError(
                f"motion missing required key {key!r}; "
                f"present: {sorted(motion.keys())}"
            )
    dof = np.asarray(motion["dof"])
    if dof.ndim != 2:
        raise ValueError(f"dof must be 2-D (T, n_dofs); got {dof.shape}")
    # DOF count is validated against the loaded MJCF in play() (robot-agnostic).


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
    )


def play(
    motion_path: Path,
    mjcf_path: Path,
    speed: float = 1.0,
    loop: bool = True,
    start_frame: int = 0,
    fixed_root_z: float | None = None,
    key: str | None = None,
) -> int:
    print(f"[play_motion] loading {motion_path} ...", flush=True)
    suffix = motion_path.suffix.lower()
    if suffix == ".x2m2":
        data = _load_x2m2(motion_path)
        if fixed_root_z is not None:
            for m in data.values():
                m["root_trans_offset"][:, 2] = fixed_root_z
        print(
            f"[play_motion] X2M2 detected: no root_trans_offset in file; "
            f"rendering at pelvis_z="
            f"{next(iter(data.values()))['root_trans_offset'][0, 2]:.3f} m "
            f"(override via --fixed-root-z).",
            flush=True,
        )
    else:
        data = joblib.load(motion_path)
    name, motion = _take_first_motion(data, key)
    _validate(motion)

    dof = np.asarray(motion["dof"], dtype=np.float64)
    root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    fps = float(motion["fps"])
    n_frames = int(dof.shape[0])

    print(
        f"[play_motion] motion '{name}': {n_frames} frames @ {fps:.2f} fps "
        f"({n_frames / fps:.2f} s)",
        flush=True,
    )
    print(f"[play_motion] loading MJCF {mjcf_path}", flush=True)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    # Robot-agnostic: derive the actuated-DOF count from the MJCF (free root = 7).
    n_dofs = model.nq - 7
    if dof.shape[1] != n_dofs:
        print(
            f"[play_motion] ERROR: motion has {dof.shape[1]} DOF but MJCF "
            f"{mjcf_path.name} expects {n_dofs} (nq={model.nq} - 7 free root).",
            file=sys.stderr,
        )
        return 1

    data_mj = mujoco.MjData(model)
    frame_idx = max(0, min(start_frame, n_frames - 1))
    print(
        "[play_motion] opening viewer; press Esc in the window to exit.",
        flush=True,
    )

    target_dt = 1.0 / max(1e-6, fps * speed)
    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        while viewer.is_running():
            t_start = time.perf_counter()

            data_mj.qpos[:3] = root_pos[frame_idx]
            data_mj.qpos[3:7] = _xyzw_to_wxyz(root_quat_xyzw[frame_idx])
            data_mj.qpos[7 : 7 + n_dofs] = dof[frame_idx]
            data_mj.qvel[:] = 0.0
            mujoco.mj_kinematics(model, data_mj)
            mujoco.mj_comPos(model, data_mj)
            viewer.sync()

            frame_idx += 1
            if frame_idx >= n_frames:
                if loop:
                    frame_idx = 0
                else:
                    print("[play_motion] reached end (no-loop); exiting.")
                    break

            elapsed = time.perf_counter() - t_start
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    print("[play_motion] viewer closed; exiting.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True, type=Path,
                   help="Motion-lib .pkl path or baked .x2m2 path.")
    p.add_argument("--key", default=None,
                   help="substring match to pick a clip inside a multi-clip PKL")
    p.add_argument("--mjcf", type=Path, default=Path(DEFAULT_MJCF),
                   help=f"MJCF to load (default: {DEFAULT_MJCF})")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0).")
    p.add_argument("--no-loop", dest="loop", action="store_false",
                   help="Stop at the end instead of looping.")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Start from this frame (default 0).")
    p.add_argument(
        "--fixed-root-z", type=float, default=None,
        help=(
            "X2M2 only: render the floating base at this pelvis z (m). "
            f"Default {X2M2_DEFAULT_PELVIS_Z}; ignored when --motion is a PKL "
            "(PKL carries its own root_trans_offset)."
        ),
    )
    args = p.parse_args(argv)
    return play(
        motion_path=args.motion,
        mjcf_path=args.mjcf,
        speed=args.speed,
        loop=args.loop,
        start_frame=args.start_frame,
        fixed_root_z=args.fixed_root_z,
        key=args.key,
    )


if __name__ == "__main__":
    raise SystemExit(main())
