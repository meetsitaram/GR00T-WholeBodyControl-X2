"""JSON-on-ZMQ helpers for scene_state / scene_reset traffic between the
deploy bridge and a scene-aware recorder (object poses / resets).

The two messages are small (a few floats per object, a handful of objects)
and infrequent (50 Hz scene_state, ad-hoc scene_reset) so JSON keeps the
wire format trivially debuggable -- you can ``zmq.recv()`` and print it.

Wire format (matches ``[topic][space][json_payload]`` sniff-pattern that
``unpack_message`` already uses for its packed binaries, except payload is
plain UTF-8 JSON with no fixed header)::

    b"scene_state {\\"sim_time\\": 1.234, ...}"

Topics:

* ``scene_state``  -- bridge -> recorder, 50 Hz
* ``scene_reset``  -- recorder -> bridge, episode-start ad-hoc

Default endpoints (override via the bridge / recorder CLI):

* ``scene_state``  PUB on tcp://*:5559   (recorder SUBs to tcp://localhost:5559)
* ``scene_reset``  PUB on tcp://*:5560   (bridge SUBs to tcp://localhost:5560)
"""

from __future__ import annotations

import json
from typing import Any


_SEPARATOR = b" "


def pack_json(topic: str, payload: dict[str, Any]) -> bytes:
    """Pack a topic + JSON dict into the canonical scene_* wire format."""
    if not topic:
        raise ValueError("topic must be non-empty")
    if " " in topic:
        raise ValueError(f"topic {topic!r} contains a space (reserved as separator)")
    return topic.encode("ascii") + _SEPARATOR + json.dumps(payload).encode("utf-8")


def unpack_json(
    message: bytes, expected_topic: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Inverse of :func:`pack_json`. Returns ``(topic, payload_dict)``.

    If *expected_topic* is set, raises ``ValueError`` when the wire topic
    differs.
    """
    sep_idx = message.find(_SEPARATOR)
    if sep_idx < 0:
        raise ValueError(
            "scene_* message missing topic / separator -- corrupt or wrong format"
        )
    topic = message[:sep_idx].decode("ascii")
    if expected_topic is not None and topic != expected_topic:
        raise ValueError(
            f"unexpected topic on scene_* socket: got {topic!r}, "
            f"want {expected_topic!r}"
        )
    payload = json.loads(message[sep_idx + 1:].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"scene_* payload must decode to a dict, got {type(payload).__name__}"
        )
    return topic, payload


# ── Topic + port defaults (importable so both sides stay in sync) ────────


SCENE_STATE_TOPIC: str = "scene_state"
SCENE_STATE_DEFAULT_PUB_PORT: int = 5559

SCENE_RESET_TOPIC: str = "scene_reset"
SCENE_RESET_DEFAULT_PUB_PORT: int = 5560


# ── Topic envelope helpers ────────────────────────────────────────────────


def serialize_scene_state(state) -> bytes:
    """Pack a scene-state object (anything with ``to_dict()``) for the ``scene_state`` topic."""
    return pack_json(SCENE_STATE_TOPIC, state.to_dict())


def serialize_reset_objects(payload) -> bytes:
    """Pack a reset-objects payload (anything with ``to_dict()``) for the ``scene_reset`` topic."""
    return pack_json(SCENE_RESET_TOPIC, payload.to_dict())
