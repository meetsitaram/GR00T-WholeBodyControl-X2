"""JSON-on-ZMQ helpers for the sim bridge's ``robot_pose`` telemetry topic.

The bridge is the only process that knows the *ground-truth* pelvis pose
in MuJoCo (the deploy's ``x2_debug`` topic only carries IMU orientation
and joint state, not root XY/Z). This helper exposes a tiny PUB/SUB
contract any host-side tool can consume to evaluate primitives:

* distance traveled per primitive
* yaw delta per primitive
* drift detection / "is the robot tipping over" overlays

Wire format mirrors :mod:`scene_state_zmq`::

    b"robot_pose {\\"sim_time\\": ..., \\"pelvis_qpos_wxyz\\": [...]}"

``pelvis_qpos_wxyz`` is the MuJoCo free-joint qpos slice
``[x, y, z, qw, qx, qy, qz]`` (length 7). Browser side converts to
``xyzw`` before feeding scipy.
"""

from __future__ import annotations

import json
from typing import Any


_SEPARATOR = b" "

ROBOT_POSE_TOPIC: str = "robot_pose"
ROBOT_POSE_DEFAULT_PUB_PORT: int = 5570


def pack_robot_pose(
    sim_time: float,
    pelvis_qpos_wxyz: list[float] | tuple[float, ...],
    *,
    extra: dict[str, Any] | None = None,
) -> bytes:
    """Pack the standard ``robot_pose`` payload."""
    if len(pelvis_qpos_wxyz) != 7:
        raise ValueError(
            f"pelvis_qpos_wxyz must be length 7 [x,y,z,qw,qx,qy,qz], "
            f"got len={len(pelvis_qpos_wxyz)}"
        )
    payload: dict[str, Any] = {
        "sim_time": float(sim_time),
        "pelvis_qpos_wxyz": [float(v) for v in pelvis_qpos_wxyz],
    }
    if extra:
        payload.update(extra)
    return ROBOT_POSE_TOPIC.encode("ascii") + _SEPARATOR + json.dumps(payload).encode("utf-8")


def unpack_robot_pose(message: bytes) -> dict[str, Any]:
    """Inverse of :func:`pack_robot_pose`. Returns the JSON payload dict."""
    sep_idx = message.find(_SEPARATOR)
    if sep_idx < 0:
        raise ValueError("robot_pose message missing topic / separator")
    topic = message[:sep_idx].decode("ascii")
    if topic != ROBOT_POSE_TOPIC:
        raise ValueError(
            f"unexpected topic on robot_pose socket: got {topic!r}, "
            f"want {ROBOT_POSE_TOPIC!r}"
        )
    payload = json.loads(message[sep_idx + 1:].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"robot_pose payload must decode to a dict, got {type(payload).__name__}"
        )
    qpos = payload.get("pelvis_qpos_wxyz")
    if not isinstance(qpos, list) or len(qpos) != 7:
        raise ValueError(
            f"robot_pose payload missing/invalid pelvis_qpos_wxyz field: {qpos!r}"
        )
    return payload


__all__ = [
    "ROBOT_POSE_DEFAULT_PUB_PORT",
    "ROBOT_POSE_TOPIC",
    "pack_robot_pose",
    "unpack_robot_pose",
]
