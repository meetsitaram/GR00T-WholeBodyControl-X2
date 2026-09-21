#!/usr/bin/env python3
"""x2_debug_to_robot_pose_bridge.py -- laptop-side measured-yaw republisher.

Sits between the C++ deploy's ``x2_debug:5557`` PUB on PC2 (packed
binary, includes ``base_quat`` from the IMU) and the laptop-side
consumers that want a measured pelvis pose -- principally the
``x2_kplanner.py`` closed-loop pose-feedback path (which reads JSON
``robot_pose`` on ``localhost:5570``) and the recorder's gesture yaw
rebase. Without this bridge, real-robot deployments have NO
measured-yaw source on the laptop, so:

  * the kplanner boots ``current_root_wxyz = R_z(0)`` from its
    warmup PKL and the very first frame it publishes hands the C++
    deploy a stale identity-yaw reference. The deploy then twists
    the body back to world +X. This is the "robot turns back to
    default orientation as soon as I start the VR planner stack"
    symptom.
  * IDLE_LOOP yaw refresh (added 2026-06-01 in x2_kplanner.py) is a
    no-op because pose_deque never receives anything.

Wire format on each side:

  * UPSTREAM (from C++ deploy on PC2):
        topic = "x2_debug"
        format = packed-binary (1280-byte JSON header + binary fields)
        decoder = gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder
        field of interest = "base_quat" (f64, length 4, IMU wxyz)

  * DOWNSTREAM (to kplanner / recorder on laptop):
        topic = "robot_pose"
        format = JSON-on-ZMQ ("robot_pose <json>")
        encoder = gear_sonic.utils.teleop.zmq.robot_pose_zmq.pack_robot_pose
        payload = {"sim_time": float, "pelvis_qpos_wxyz": [x,y,z,qw,qx,qy,qz]}

XY/Z are not measurable from the IMU on the real robot, so we publish
them as zeros. Every downstream consumer (kplanner, recorder gesture
rebase, eval tools) only uses the quat for yaw extraction; xy/z are
ignored by the gear_sonic stack on real-robot anyway.

This daemon is on the laptop (laptop -> PC2 over wifi/ethernet) and
imports gear_sonic helpers freely. It is the natural complement to
``gear_sonic_deploy/scripts/x2_pose_proxy.py`` (which lives on PC2
and now consumes x2_debug locally on loopback for its idle-frame
yaw rebase).

Example::

    python -m gear_sonic_deploy.scripts.x2_debug_to_robot_pose_bridge \\
        --x2-debug-host "${PC2_HOST}" \\
        --x2-debug-port 5557 \\
        --robot-pose-port 5570
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.zmq.robot_pose_zmq import (  # noqa: E402
    ROBOT_POSE_DEFAULT_PUB_PORT,
    ROBOT_POSE_TOPIC,
    pack_robot_pose,
)
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)


def _build_robot_pose_msg(base_quat_wxyz: np.ndarray) -> bytes:
    """Build a ``robot_pose`` JSON-on-ZMQ message from an IMU base_quat.

    XY/Z are not measurable from the IMU on real robot; we publish
    zeros so the wire shape (length-7 pelvis_qpos_wxyz) matches the
    MuJoCo bridge's contract. ``sim_time`` is monotonic seconds at
    publish time (consumers use it for staleness gating, not for
    physical interpretation).
    """
    pelvis_qpos_wxyz = [
        0.0, 0.0, 0.0,
        float(base_quat_wxyz[0]),
        float(base_quat_wxyz[1]),
        float(base_quat_wxyz[2]),
        float(base_quat_wxyz[3]),
    ]
    return pack_robot_pose(time.monotonic(), pelvis_qpos_wxyz)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--x2-debug-host",
        required=True,
        help="PC2 IP publishing x2_debug (the value of ${PC2_HOST}, e.g. <PC2_IP>).",
    )
    p.add_argument(
        "--x2-debug-port",
        type=int,
        default=5557,
        help="PC2 x2_debug PUB port (default 5557; deploy spawns this "
             "when --zmq-debug-port > 0, which is the deploy_x2.sh default).",
    )
    p.add_argument(
        "--x2-debug-topic",
        default="x2_debug",
        help="Topic prefix for x2_debug (default 'x2_debug').",
    )
    p.add_argument(
        "--robot-pose-bind",
        default="*",
        help="Local bind iface for the robot_pose PUB (default '*' = "
             "all). Bound on localhost so the kplanner can SUB it on "
             "tcp://127.0.0.1:<port>.",
    )
    p.add_argument(
        "--robot-pose-port",
        type=int,
        default=ROBOT_POSE_DEFAULT_PUB_PORT,
        help=f"Local PUB port for robot_pose (default "
             f"{ROBOT_POSE_DEFAULT_PUB_PORT}; matches the MuJoCo "
             f"bridge so kplanner config is the same across sim/real).",
    )
    p.add_argument(
        "--rate-cap-hz",
        type=float,
        default=200.0,
        help="Soft rate cap on republished frames (default 200 Hz). "
             "The deploy publishes x2_debug at 500 Hz which is way "
             "more than the kplanner needs; we coalesce to 1/cap "
             "seconds so we don't spend CPU repacking every frame. "
             "Set 0 to republish every received frame.",
    )
    p.add_argument(
        "--status-every-s",
        type=float,
        default=5.0,
        help="Periodic status print interval (default 5s).",
    )
    args = p.parse_args(argv)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 100)
    upstream_url = f"tcp://{args.x2_debug_host}:{args.x2_debug_port}"
    sub.connect(upstream_url)
    sub.setsockopt(zmq.SUBSCRIBE, args.x2_debug_topic.encode("utf-8"))

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 100)
    bind_url = f"tcp://{args.robot_pose_bind}:{args.robot_pose_port}"
    pub.bind(bind_url)

    print(
        f"[debug2pose] hostname={socket.gethostname()}", flush=True
    )
    print(
        f"[debug2pose] upstream SUB:   {upstream_url} "
        f"topic={args.x2_debug_topic!r}",
        flush=True,
    )
    print(
        f"[debug2pose] downstream PUB: {bind_url} topic={ROBOT_POSE_TOPIC!r}",
        flush=True,
    )

    min_period_s = (
        1.0 / float(args.rate_cap_hz) if args.rate_cap_hz > 0.0 else 0.0
    )
    last_pub_s = -1.0
    recv_frames = 0
    pub_frames = 0
    decode_failures = 0
    last_quat_wxyz: np.ndarray | None = None
    last_status_s = time.monotonic()

    print(
        "[debug2pose] starting; will publish robot_pose as soon as the "
        "first x2_debug frame is decoded",
        flush=True,
    )

    try:
        while True:
            try:
                raw = sub.recv()
            except KeyboardInterrupt:
                raise
            recv_frames += 1

            # Drain any backlog and only keep the latest frame -- yaw
            # doesn't accumulate, only the most recent IMU sample
            # matters. Keeps the bridge cheap even if upstream
            # bursts (e.g. wifi recovered and unblocked a queue).
            while True:
                try:
                    raw = sub.recv(zmq.NOBLOCK)
                    recv_frames += 1
                except zmq.Again:
                    break

            try:
                msg = unpack_message(raw, expected_topic=args.x2_debug_topic)
            except ValueError:
                decode_failures += 1
                continue

            quat = msg.fields.get("base_quat")
            if quat is None or quat.shape != (4,):
                decode_failures += 1
                continue
            quat_wxyz = quat.astype(np.float64).reshape(4).copy()
            last_quat_wxyz = quat_wxyz

            now = time.monotonic()
            if min_period_s > 0.0 and last_pub_s >= 0.0 and (
                now - last_pub_s
            ) < min_period_s:
                # Inside the rate cap window; skip republish. The
                # latest sample is cached in last_quat_wxyz; the next
                # tick that clears the cap will pick it up.
                pass
            else:
                try:
                    pub.send(_build_robot_pose_msg(quat_wxyz), zmq.NOBLOCK)
                    pub_frames += 1
                    last_pub_s = now
                except zmq.Again:
                    pass

            if now - last_status_s >= args.status_every_s:
                if last_quat_wxyz is None:
                    last_yaw_str = "n/a"
                else:
                    import math
                    qw, qx, qy, qz = last_quat_wxyz
                    yaw_rad = math.atan2(
                        2.0 * (qw * qz - qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz),
                    )
                    last_yaw_str = f"{math.degrees(yaw_rad):+.1f}deg"
                print(
                    f"[debug2pose] recv={recv_frames} pub={pub_frames} "
                    f"decode_fail={decode_failures} last_yaw={last_yaw_str}",
                    flush=True,
                )
                last_status_s = now
    except KeyboardInterrupt:
        print("[debug2pose] SIGINT received; tearing down", flush=True)
    finally:
        sub.close(linger=0)
        pub.close(linger=0)
        ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
