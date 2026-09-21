#!/usr/bin/env python3
"""Host-side mirror viewer for the docker sim stack — stall-proof by design.

WHY THIS EXISTS (2026-08-29): running the MuJoCo viewer INSIDE the x2sim
container (`--sim-viewer`) needs NV-GLX on the forwarded display, which the
container's X client doesn't get; worse, the failed/stalled GL init blocked
the bridge's physics thread >500 ms and the deploy's staleness guard
E-STOPPED the run mid-rehearsal. This viewer instead renders on the HOST
GPU from the sim's existing read-only telemetry, so it can never touch the
control loop:

    robot_pose :5570  (bridge PUB)  -> pelvis qpos[0:7]  [x y z qw qx qy qz]
    x2_debug   :5659  (deploy PUB)  -> body_q (31, MJ order)

Full qpos = 7 + 31 = 38; we set it and mj_forward — pure kinematic mirror.

    python gear_sonic/scripts/sim_mirror_viewer.py            # sim defaults
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MJCF_DEFAULT = str(REPO / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mjcf", default=MJCF_DEFAULT)
    ap.add_argument("--pose-port", type=int, default=5570,
                    help="bridge robot_pose PUB (ground-truth pelvis qpos)")
    ap.add_argument("--debug-port", type=int, default=5659,
                    help="deploy x2_debug PUB (sim stack 5659)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--with-omnihand", action="store_true",
                    help="render the composed X2 + OmniHand model (what the bridge simulates "
                         "under --with-omnihand) with fingers mirrored from the kplanner pose wire")
    ap.add_argument("--hand-pose-port", type=int, default=5656,
                    help="kplanner pose PUB carrying left/right_hand_joints[10] (sim stack 5656)")
    args = ap.parse_args()

    import zmq
    import mujoco
    import mujoco.viewer as mjv
    from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose
    from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message

    ctx = zmq.Context.instance()
    pose = ctx.socket(zmq.SUB)
    pose.connect(f"tcp://{args.host}:{args.pose_port}")
    pose.setsockopt(zmq.SUBSCRIBE, b"robot_pose")
    pose.setsockopt(zmq.CONFLATE, 0)
    debug = ctx.socket(zmq.SUB)
    debug.connect(f"tcp://{args.host}:{args.debug_port}")
    debug.setsockopt_string(zmq.SUBSCRIBE, "x2_debug")

    hand_layout = None
    hand_sub = None
    if args.with_omnihand:
        sys.path.insert(0, str(REPO))
        from gear_sonic.scripts.compose_x2_with_omnihand import (
            build_x2_with_omnihand_spec, apply_active_hand_qpos)
        _, model, hand_layout = build_x2_with_omnihand_spec()
        hand_sub = ctx.socket(zmq.SUB)
        hand_sub.connect(f"tcp://{args.host}:{args.hand_pose_port}")
        hand_sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
        print(f"[mirror] OmniHand composed; fingers from pose wire :{args.hand_pose_port}", flush=True)
    else:
        model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # x2_debug body_q is in the plain x2_ultra.xml joint order (31). In the
    # composed model the hand joints sit INSIDE the tree (left hand before
    # the right arm), so map every body joint to its qpos address by name.
    plain = mujoco.MjModel.from_xml_path(args.mjcf)
    body_names = [plain.joint(i).name for i in range(1, plain.njnt)]   # skip the free joint
    nq_joints = len(body_names)
    body_qadr = np.array([model.joint(n).qposadr[0] for n in body_names], int)
    hand_q = {"left": None, "right": None}
    print(f"[mirror] {Path(args.mjcf).name}: nq={model.nq} "
          f"(7 free + {nq_joints} joints); robot_pose :{args.pose_port} "
          f"x2_debug :{args.debug_port} — read-only mirror, sim unaffected",
          flush=True)

    pelvis = None
    body_q = None
    t_last_msg = 0.0
    with mjv.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # drain both subs to newest
            while True:
                try:
                    payload = unpack_robot_pose(pose.recv(zmq.NOBLOCK))
                    pelvis = np.asarray(payload["pelvis_qpos_wxyz"], float)
                    t_last_msg = time.monotonic()
                except zmq.Again:
                    break
                except Exception:
                    pass
            while True:
                try:
                    f = unpack_message(debug.recv(zmq.NOBLOCK),
                                       expected_topic="x2_debug").fields
                    bq = f.get("body_q")
                    if bq is not None:
                        body_q = np.asarray(bq, float).ravel()[:nq_joints]
                        t_last_msg = time.monotonic()
                except zmq.Again:
                    break
                except Exception:
                    pass
            if hand_sub is not None:
                while True:
                    try:
                        f = unpack_message(hand_sub.recv(zmq.NOBLOCK), expected_topic="pose").fields
                        for side, key in (("left", "left_hand_joints"), ("right", "right_hand_joints")):
                            v = f.get(key)
                            if v is not None and np.asarray(v).size == 10:
                                hand_q[side] = np.asarray(v, float).ravel()
                    except zmq.Again:
                        break
                    except Exception:
                        pass
            if pelvis is not None:
                data.qpos[0:7] = pelvis
            if body_q is not None and body_q.shape[0] == nq_joints:
                data.qpos[body_qadr] = body_q
            if hand_layout is not None and (hand_q["left"] is not None or hand_q["right"] is not None):
                try:
                    apply_active_hand_qpos(data, hand_layout, left_active=hand_q["left"],
                                           right_active=hand_q["right"])
                except Exception:
                    pass
            mujoco.mj_forward(model, data)
            viewer.sync()
            stale = time.monotonic() - t_last_msg
            if t_last_msg and stale > 2.0:
                print(f"\r[mirror] telemetry stale {stale:.0f}s "
                      f"(stack down?)   ", end="", flush=True)
            time.sleep(1.0 / 60.0)
    print("\n[mirror] viewer closed (sim keeps running).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
