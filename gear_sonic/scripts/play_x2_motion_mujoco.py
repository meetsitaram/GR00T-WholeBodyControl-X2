#!/usr/bin/env python3
"""Kinematic playback of a motion_lib PKL on the X2 Ultra MuJoCo model.

No policy, no physics — just writes the motion's root pose + joint angles
straight into ``mj_data.qpos`` each frame and calls ``mj_forward``. Use this
to sanity-check that:

  * The 31-DOF joint ordering in the PKL matches the MJCF actuator order.
  * Root translation / orientation conventions match (xyz + wxyz quat).
  * The retargeted clip looks right kinematically (before involving the policy).

Pair with ``train_agent_trl.py ++replay=True`` (IsaacLab) on the same PKL
to confirm the same clip plays back identically across the two simulators.

Usage:
    conda run -n env_isaaclab --no-capture-output python \
        gear_sonic/scripts/play_x2_motion_mujoco.py \
        --motion gear_sonic/data/motions/x2_demo_bank.pkl

    # Pick a specific motion key from a multi-motion PKL
    conda run -n env_isaaclab --no-capture-output python \
        gear_sonic/scripts/play_x2_motion_mujoco.py \
        --motion gear_sonic/data/motions/x2_demo_bank.pkl \
        --motion-key <clip_name>

    # Slow-mo / faster
    --speed 0.25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np

GEAR_SONIC_ROOT = Path(__file__).resolve().parents[2]
MJCF_PATH = str(
    GEAR_SONIC_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
)
NUM_DOFS = 31


def load_motion(path: str, key: str | None) -> tuple[str, dict]:
    data = joblib.load(path)
    keys = list(data.keys())
    if not keys:
        raise RuntimeError(f"PKL has no motions: {path}")
    if key is None:
        key = keys[0]
    if key not in data:
        matches = [k for k in keys if key in k]
        if len(matches) == 1:
            key = matches[0]
        else:
            preview = ", ".join(keys[:6]) + (" ..." if len(keys) > 6 else "")
            raise KeyError(
                f"Motion key '{key}' not found in {path}\n"
                f"  {len(keys)} entries available, first few: {preview}"
            )
    return key, data[key]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--motion", required=True, help="Path to motion_lib PKL")
    parser.add_argument(
        "--motion-key",
        default=None,
        help="Motion entry key (substring match allowed). Default: first entry.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (default 1x)")
    parser.add_argument(
        "--no-loop", action="store_true", help="Stop after one playthrough"
    )
    parser.add_argument(
        "--anchor-xy",
        action="store_true",
        help="Anchor pelvis XY at origin (do not advance world translation). "
        "Useful for clips where the motion drifts off-screen.",
    )
    parser.add_argument("--world-cam", action="store_true",
                        help="Fixed world camera (not pelvis-tracking) so forward "
                        "translation is visible against the ground.")
    parser.add_argument("--start-sec", type=float, default=None,
                        help="Play window start in seconds (default: clip start).")
    parser.add_argument("--end-sec", type=float, default=None,
                        help="Play window end in seconds (default: clip end).")
    args = parser.parse_args()

    key, m = load_motion(args.motion, args.motion_key)
    fps = float(m["fps"])
    n_frames = m["dof"].shape[0]
    start_f = 0 if args.start_sec is None else max(0, int(args.start_sec * fps))
    end_f = n_frames if args.end_sec is None else min(n_frames, int(args.end_sec * fps))
    if end_f <= start_f:
        end_f = n_frames
    print(f"  play window: frames [{start_f}, {end_f}) = "
          f"{start_f / fps:.2f}s .. {end_f / fps:.2f}s", flush=True)
    print(f"Motion: {key}", flush=True)
    print(f"  {n_frames} frames @ {fps:g} fps = {n_frames / fps:.2f}s", flush=True)
    print(f"  dof shape={m['dof'].shape}  root_trans={m['root_trans_offset'].shape} "
          f"root_rot(xyzw)={m['root_rot'].shape}", flush=True)

    if m["dof"].shape[1] != NUM_DOFS:
        raise RuntimeError(
            f"Expected {NUM_DOFS} DOFs in PKL, got {m['dof'].shape[1]}. "
            "This script targets X2 Ultra; rebuild the motion lib with --robot x2_ultra."
        )

    print(f"Loading MuJoCo model: {MJCF_PATH}", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    pelvis_id = mj_model.body("pelvis").id

    dof = np.asarray(m["dof"], dtype=np.float64)               # (T, 31) MuJoCo order
    root_pos = np.asarray(m["root_trans_offset"], dtype=np.float64)  # (T, 3) world xyz
    root_xyzw = np.asarray(m["root_rot"], dtype=np.float64)    # (T, 4) scipy xyzw

    def apply_frame(f: int):
        f = int(f) % n_frames
        if args.anchor_xy:
            mj_data.qpos[0] = 0.0
            mj_data.qpos[1] = 0.0
        else:
            mj_data.qpos[0] = root_pos[f, 0]
            mj_data.qpos[1] = root_pos[f, 1]
        mj_data.qpos[2] = root_pos[f, 2]
        # MuJoCo qpos[3:7] is wxyz; PKL stores xyzw -> reorder.
        mj_data.qpos[3] = root_xyzw[f, 3]
        mj_data.qpos[4] = root_xyzw[f, 0]
        mj_data.qpos[5] = root_xyzw[f, 1]
        mj_data.qpos[6] = root_xyzw[f, 2]
        mj_data.qpos[7 : 7 + NUM_DOFS] = dof[f]
        mj_data.qvel[:] = 0.0
        mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(mj_model, mj_data)

    apply_frame(start_f)
    init_root_z = float(root_pos[start_f, 2])

    paused = [False]
    cur_frame = [start_f]
    # Mutable wall-clock anchor so key_callback can re-sync playback timing.
    clock = {"start": time.time(), "origin": start_f}

    def resync():
        clock["start"] = time.time()
        clock["origin"] = cur_frame[0]

    def key_callback(keycode):
        import glfw

        if keycode == glfw.KEY_SPACE:
            paused[0] = not paused[0]
            if not paused[0]:
                resync()  # don't fast-forward through paused wall time
            print("Paused" if paused[0] else "Resumed", flush=True)
        elif keycode == glfw.KEY_R:
            cur_frame[0] = start_f
            apply_frame(start_f)
            paused[0] = False
            resync()
            print(f"[replay] frame {start_f} ({start_f / fps:.2f}s)", flush=True)
        elif keycode == glfw.KEY_LEFT:
            cur_frame[0] = max(start_f, cur_frame[0] - 10)
            apply_frame(cur_frame[0])
            resync()
        elif keycode == glfw.KEY_RIGHT:
            cur_frame[0] = min(end_f - 1, cur_frame[0] + 10)
            apply_frame(cur_frame[0])
            resync()

    print(
        "\n=== X2 Kinematic Playback ===\n"
        "  SPACE pause | R restart | LEFT/RIGHT scrub ±10 frames\n",
        flush=True,
    )

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        if args.world_cam:
            # Fixed world viewpoint centered on the clip's mid-path so forward
            # translation is visible against the ground (not masked by tracking).
            mid_xy = 0.5 * (root_pos[start_f, :2] + root_pos[end_f - 1, :2])
            path_len = float(np.linalg.norm(root_pos[end_f - 1, :2] - root_pos[start_f, :2]))
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [mid_xy[0], mid_xy[1], init_root_z]
            viewer.cam.distance = max(4.0, path_len + 3.0)
        else:
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [0.0, 0.0, init_root_z]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_id

        frame_dt = 1.0 / (fps * max(args.speed, 1e-6))
        resync()

        while viewer.is_running():
            if paused[0]:
                viewer.sync()
                time.sleep(0.02)
                continue

            elapsed = time.time() - clock["start"]
            span = end_f - start_f
            target_frame = clock["origin"] + int(elapsed / frame_dt)

            if not args.no_loop:
                target_frame = start_f + ((target_frame - start_f) % span)
            elif target_frame >= end_f:
                print(f"End of window (frame {end_f}, {end_f / fps:.2f}s).", flush=True)
                paused[0] = True
                continue

            if target_frame != cur_frame[0]:
                cur_frame[0] = target_frame
                apply_frame(target_frame)

            viewer.sync()
            time.sleep(min(frame_dt, 0.02))


if __name__ == "__main__":
    main()
