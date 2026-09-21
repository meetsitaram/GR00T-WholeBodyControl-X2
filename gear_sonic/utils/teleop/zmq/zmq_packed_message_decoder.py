"""Python receive-side decoder for the packed-binary ZMQ wire format.

The wire format is the inverse of ``zmq_planner_sender.pack_pose_message``:

    [topic_bytes][1280-byte JSON header][concatenated binary fields]

The header is right-padded with NULs and (per the C++ subscriber implementation
at ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/
zmq_packed_message_subscriber.hpp``) describes:

    {
      "v": int,                 # protocol version (1, 3, 4)
      "endian": "le" | "be",
      "count": int,             # batch dimension
      "fields": [
        {"name": str, "dtype": "f32"|"f64"|"i32"|"i64"|"u8"|"bool",
         "shape": [..]},
        ...
      ]
    }

This module provides a tiny decoder so Python tools can subscribe to topics
published by either:

* the C++ deploy harness (``x2_debug`` on port 5557 — output handler), or
* another Python publisher built on top of ``pack_pose_message`` (``pose`` on
  port 5556 — VLA / mock-VLA stream).

It is **the missing complement** to ``pack_pose_message`` and is required for
the M2 mock-VLA acceptance gate (see ``gear_sonic/scripts/dump_x2_debug.py``
and ``tests/test_zmq_pose_loopback.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


HEADER_SIZE: int = 1280
"""Header size in bytes. Must match HEADER_SIZE in:

* ``gear_sonic/utils/teleop/zmq/zmq_planner_sender.py``
* ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/
  zmq_packed_message_subscriber.hpp``
"""


_DTYPE_TO_NUMPY: dict[str, np.dtype] = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("u1"),  # bools are encoded as 1-byte uints on the wire
}


@dataclass(frozen=True)
class DecodedMessage:
    """Result of unpacking a single ZMQ message.

    Attributes:
        topic: Topic prefix as received (or ``""`` if the caller stripped it).
        version: Protocol version from the header (``v`` field).
        endian: Endianness reported by the publisher (``"le"`` / ``"be"``).
        count: Batch dimension reported by the publisher.
        fields: ``{field_name: numpy.ndarray}``. Arrays are reshaped to the
            ``shape`` declared in the header. ``bool`` fields come back as
            ``numpy.bool_`` arrays.
        raw_header: Decoded JSON dict, useful for debugging.
    """

    topic: str
    version: int
    endian: str
    count: int
    fields: dict[str, np.ndarray]
    raw_header: dict[str, Any]


def unpack_message(message: bytes, expected_topic: str | None = None) -> DecodedMessage:
    """Decode one packed-format ZMQ message into named numpy arrays.

    Args:
        message: Single-part ZMQ payload as returned by ``socket.recv()``.
        expected_topic: If non-empty, the topic prefix the caller subscribed
            to. The bytes for ``expected_topic`` are stripped from the front
            of ``message`` before parsing. Pass ``""`` or ``None`` if the
            socket was subscribed to all topics OR if the caller has already
            stripped the prefix.

    Returns:
        A populated :class:`DecodedMessage`.

    Raises:
        ValueError: If the message is too short to contain a header, the JSON
            header is malformed, the declared fields don't fit the binary
            payload, or any field has an unsupported dtype.
    """
    if expected_topic:
        topic_bytes = expected_topic.encode("utf-8")
        if not message.startswith(topic_bytes):
            raise ValueError(
                f"message does not start with expected topic {expected_topic!r}; "
                f"first 32 bytes: {message[:32]!r}"
            )
        body = message[len(topic_bytes) :]
        topic = expected_topic
    else:
        body = message
        topic = ""

    if len(body) < HEADER_SIZE:
        raise ValueError(
            f"message body too small: got {len(body)} bytes, "
            f"need at least HEADER_SIZE={HEADER_SIZE}"
        )

    header_blob = body[:HEADER_SIZE].rstrip(b"\x00")
    payload = body[HEADER_SIZE:]

    try:
        header = json.loads(header_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"header is not valid JSON: {exc}") from exc

    fields_meta = header.get("fields")
    if not isinstance(fields_meta, list):
        raise ValueError(f"header missing 'fields' list, got: {header}")

    decoded: dict[str, np.ndarray] = {}
    cursor = 0
    for field in fields_meta:
        try:
            name = field["name"]
            dtype_str = field["dtype"]
            shape = tuple(int(s) for s in field["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed field entry {field!r}: {exc}") from exc

        if dtype_str not in _DTYPE_TO_NUMPY:
            raise ValueError(
                f"unsupported dtype {dtype_str!r} for field {name!r}; "
                f"supported: {sorted(_DTYPE_TO_NUMPY)}"
            )

        np_dtype = _DTYPE_TO_NUMPY[dtype_str]
        nelem = int(np.prod(shape)) if shape else 1
        nbytes = nelem * np_dtype.itemsize
        if cursor + nbytes > len(payload):
            raise ValueError(
                f"field {name!r}: payload truncated "
                f"(need {nbytes} bytes from offset {cursor}, "
                f"have {len(payload) - cursor})"
            )
        raw = payload[cursor : cursor + nbytes]
        cursor += nbytes

        arr = np.frombuffer(raw, dtype=np_dtype, count=nelem).reshape(shape if shape else (1,))
        if dtype_str == "bool":
            arr = arr.astype(np.bool_)
        decoded[name] = arr

    return DecodedMessage(
        topic=topic,
        version=int(header.get("v", 0)),
        endian=str(header.get("endian", "le")),
        count=int(header.get("count", 1)),
        fields=decoded,
        raw_header=header,
    )


__all__ = ["DecodedMessage", "HEADER_SIZE", "unpack_message"]
