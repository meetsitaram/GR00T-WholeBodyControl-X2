"""Packed pose-message wire format helpers shared by mux + watchdog.

All constants and (de)coders here mirror what
``gear_sonic/scripts/vla/live_vla_publish_motion_token.py`` emits and what
``gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/zmq/
zmq_pose_input_source.hpp`` consumes. Field order, dtype, and shape
must stay byte-compatible with both ends.

The format is the v4 wire layout used by every pose producer in the
stack:

    [topic_bytes][1280-byte JSON header (NUL-padded)][concatenated
    binary fields]

Header is JSON ``{"v": int, "endian": "le", "count": int, "fields":
[{name, dtype, shape}, ...]}``.

This module also carries the minimal slice of
``gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder`` we need for
the PC2 watchdog's x2_debug yaw-rebase path. The watchdog has a strict
numpy + pyzmq + stdlib dependency budget (it ships to PC2 by
``pc2_bringup.sh`` and runs in a venv that doesn't carry scipy), so we
inline rather than import the real decoder (which pulls scipy via the
planner's blending module).
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Wire-format constants (must mirror live_vla_publish_motion_token + the C++
# deploy's ZmqPoseInputSource).
# ---------------------------------------------------------------------------
NUM_BODY_DOFS: int = 31
SONIC_MOTION_TOKEN_DIM: int = 64
DEFAULT_HAND_DOF: int = 10
DEFAULT_PUB_RATE_HZ: float = 50.0
FUTURE_DT_S: float = 0.1
NUM_FUTURE_SLOTS: int = 9
# step_ticks at 50 Hz so adjacent future slots are 0.1 s apart -- matches
# planner.step_with_lookahead(step_ticks=5) and IdleStandReplay.future_window.
FUTURE_STEP_TICKS: int = int(round(DEFAULT_PUB_RATE_HZ * FUTURE_DT_S))

# X2M2 binary format (see bake_idle_stand_x2m2.py + reference_motion.hpp).
X2M2_MAGIC: int = 0x58324D32

# ZMQ wire header (matches gear_sonic.utils.teleop.zmq.zmq_planner_sender).
HEADER_SIZE: int = 1280

# Pre-allocated zero slices for idle frames (avoid per-tick allocation).
ZERO_MOTION_TOKEN = np.zeros(SONIC_MOTION_TOKEN_DIM, dtype=np.float32)
ZERO_HAND = np.zeros(DEFAULT_HAND_DOF, dtype=np.float32)
FUTURE_DT_FIELD = np.array([FUTURE_DT_S], dtype=np.float32)
ZERO_QVEL_FUTURE = np.zeros(
    (NUM_FUTURE_SLOTS, NUM_BODY_DOFS), dtype=np.float32
)
ZERO_QVEL_FUTURE.setflags(write=False)


# ---------------------------------------------------------------------------
# Inline x2_debug packed-binary decoder. The watchdog has a numpy +
# pyzmq + stdlib only dependency budget (no gear_sonic / scipy on PC2),
# so we duplicate the minimum slice of
# ``gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder`` we
# actually need (pulling ``base_quat`` out of one frame) rather than
# importing the real decoder and dragging in scipy via blending.py.
#
# Wire format: see zmq_packed_message_decoder.py header comment.
#   [topic_bytes][1280-byte JSON header (NUL-padded)][concatenated binary fields]
# Header is JSON {"v":int, "endian":"le", "count":int, "fields":[{name,dtype,shape},...]}
# ---------------------------------------------------------------------------
X2_DEBUG_HEADER_SIZE: int = 1280

_DTYPE_BPE: dict[str, int] = {
    "f32": 4, "f64": 8, "i32": 4, "i64": 8, "u8": 1, "bool": 1,
}


def decode_x2_debug_fields(
    msg: bytes, names: tuple[str, ...], topic: str = "x2_debug"
) -> dict[str, np.ndarray] | None:
    """Extract several named fields from one packed x2_debug frame.

    Generalisation of :func:`decode_x2_debug_base_quat` (2026-09-04,
    whole-body release stability): the kplanner needs the deploy's
    MEASURED joints (``body_q``, f64 [31], MuJoCo order) as well as the
    pelvis quat so it can shadow / hold / return-from the robot's actual
    pose instead of a constant anchor. Returns None on any decode
    failure; a field that is absent from the frame is simply missing
    from the dict (callers check). Stops walking once every requested
    field has been read.
    """
    topic_bytes = topic.encode("utf-8")
    if not msg.startswith(topic_bytes):
        return None
    body = msg[len(topic_bytes):]
    if len(body) < X2_DEBUG_HEADER_SIZE:
        return None
    header_blob = body[:X2_DEBUG_HEADER_SIZE].rstrip(b"\x00")
    payload = body[X2_DEBUG_HEADER_SIZE:]
    try:
        header = json.loads(header_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    fields = header.get("fields")
    if not isinstance(fields, list):
        return None
    want = set(names)
    out: dict[str, np.ndarray] = {}
    np_dtype = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8",
                "u8": "u1", "bool": "?"}
    cursor = 0
    for f in fields:
        try:
            name = str(f["name"])
            dtype = str(f["dtype"])
            shape = tuple(int(s) for s in f["shape"])
        except (KeyError, TypeError, ValueError):
            return None
        bpe = _DTYPE_BPE.get(dtype)
        if bpe is None:
            return None
        nelem = 1
        for s in shape:
            nelem *= s
        nbytes = nelem * bpe
        if cursor + nbytes > len(payload):
            return None
        if name in want:
            out[name] = np.frombuffer(
                payload[cursor:cursor + nbytes], dtype=np_dtype[dtype]
            ).reshape(shape).copy()
            if len(out) == len(want):
                break
        cursor += nbytes
    return out


def decode_x2_debug_base_quat(
    msg: bytes, topic: str = "x2_debug"
) -> np.ndarray | None:
    """Extract ``base_quat`` (wxyz, length 4, f64) from one x2_debug frame.

    Returns ``None`` on any decode failure (wrong topic, truncated
    payload, malformed header, missing/mistyped field). The watchdog's
    publish thread MUST survive transient decoder failures -- a
    misshapen frame can't be allowed to wedge the wire and trip the
    deploy's starvation watchdog. Callers should treat ``None`` as
    "fall back to last known measured yaw".

    Walks fields in order and bails out the moment we read ``base_quat``;
    we never need any later field, so this stays O(prefix_of_header).
    """
    topic_bytes = topic.encode("utf-8")
    if not msg.startswith(topic_bytes):
        return None
    body = msg[len(topic_bytes):]
    if len(body) < X2_DEBUG_HEADER_SIZE:
        return None
    header_blob = body[:X2_DEBUG_HEADER_SIZE].rstrip(b"\x00")
    payload = body[X2_DEBUG_HEADER_SIZE:]
    try:
        header = json.loads(header_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    fields = header.get("fields")
    if not isinstance(fields, list):
        return None
    cursor = 0
    for f in fields:
        try:
            name = str(f["name"])
            dtype = str(f["dtype"])
            shape = tuple(int(s) for s in f["shape"])
        except (KeyError, TypeError, ValueError):
            return None
        bpe = _DTYPE_BPE.get(dtype)
        if bpe is None:
            return None
        nelem = 1
        for s in shape:
            nelem *= s
        nbytes = nelem * bpe
        if cursor + nbytes > len(payload):
            return None
        if name == "base_quat":
            if dtype != "f64" or shape != (4,):
                return None
            return np.frombuffer(
                payload[cursor:cursor + nbytes], dtype="<f8"
            ).copy()
        cursor += nbytes
    return None  # field absent from this frame


# ---------------------------------------------------------------------------
# Surgical field-override helpers: splice new f32 bytes into an existing
# packed pose message in place, preserving every other field. Used by
# the mux's engagement slow-step ramp to clamp the operator's first
# OVERRIDE frames per-element relative to the last forwarded VLA pose
# without re-encoding the whole frame.
# ---------------------------------------------------------------------------
def rebuild_msg_with_field_overrides(
    msg: bytes,
    topic: str,
    overrides: dict[str, np.ndarray],
) -> bytes | None:
    """Surgically overwrite one or more f32 fields in a packed pose frame.

    Preserves every other field (root_quat, hand joints, motion_token,
    frame_index, ...) because the v4 header layout encodes each field
    as a fixed-length blob at a known cursor position. Same shape +
    same dtype = same byte count, so we splice the new bytes into
    the existing payload in place without re-packing the header or
    rewriting field order.

    ``overrides`` maps field names to numpy arrays. Each array must
    match the field's declared dtype + shape in the header (we
    enforce f32 here -- all jpos/jvel/quat fields are f32 on the
    wire). Returns ``None`` if any override mismatches or the
    header is malformed; callers should fall back to forwarding
    the original ``msg`` verbatim in that case so a corrupt frame
    never silently steals the wire.

    Used by the LIVE -> OVERRIDE engagement ramp: the mux clamps
    the operator's first few override frames to a slow per-tick step
    relative to the LAST FORWARDED jpos so the deploy doesn't see
    a single-tick step from VLA-pose to operator-pose. **Also flattens
    ``joint_pos_mj_future`` to the clamped current jpos and zeros
    ``joint_vel_mj_future``** -- without that, the deploy's window-
    mode policy reads the operator's untouched future window (which
    still encodes "go all the way to operator-pose in 0.9 s") and
    slams the body to follow the future even though the current
    frame is properly rate-limited.
    """
    topic_bytes = topic.encode("utf-8")
    if not msg.startswith(topic_bytes):
        return None
    body_start = len(topic_bytes)
    body = msg[body_start:]
    if len(body) < HEADER_SIZE:
        return None
    header_blob = body[:HEADER_SIZE].rstrip(b"\x00")
    payload_start = body_start + HEADER_SIZE
    payload_len = len(msg) - payload_start
    try:
        header = json.loads(header_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    fields = header.get("fields")
    if not isinstance(fields, list):
        return None
    splices: list[tuple[int, int, bytes]] = []
    matched: set[str] = set()
    cursor = 0
    for f in fields:
        try:
            fname = str(f["name"])
            dtype = str(f["dtype"])
            shape = tuple(int(s) for s in f["shape"])
        except (KeyError, TypeError, ValueError):
            return None
        bpe = _DTYPE_BPE.get(dtype)
        if bpe is None:
            return None
        nelem = 1
        for s in shape:
            nelem *= s
        nbytes = nelem * bpe
        if cursor + nbytes > payload_len:
            return None
        if fname in overrides:
            if dtype != "f32":
                return None
            arr = np.asarray(overrides[fname], dtype=np.float32)
            if arr.shape != shape:
                return None
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            new_bytes = arr.tobytes()
            if len(new_bytes) != nbytes:
                return None
            abs_start = payload_start + cursor
            splices.append((abs_start, abs_start + nbytes, new_bytes))
            matched.add(fname)
        cursor += nbytes
    if matched != set(overrides):
        return None
    if not splices:
        return msg
    splices.sort(key=lambda s: s[0])
    out = bytearray()
    cur = 0
    for start, end, blob in splices:
        out.extend(msg[cur:start])
        out.extend(blob)
        cur = end
    out.extend(msg[cur:])
    return bytes(out)


def rebuild_msg_with_jpos_override(
    msg: bytes, topic: str, new_jpos: np.ndarray
) -> bytes | None:
    """Back-compat single-field wrapper around the multi-field helper.

    Kept so older callers and regression pins that only need to clamp
    the current jpos keep working. Internally delegates to
    ``rebuild_msg_with_field_overrides`` so both code paths share the
    same byte-splice logic.
    """
    return rebuild_msg_with_field_overrides(
        msg, topic, {"joint_pos_mj": new_jpos}
    )


# ---------------------------------------------------------------------------
# Field decoders. The mux uses these to extract operator-commanded
# joint positions / hand vectors from incoming override frames so it
# can run the frozen-frame release detector + engagement-ramp clamp,
# and the watchdog uses joint_pos_mj to seed the HOLD/BLEND lerp
# anchor.
# ---------------------------------------------------------------------------
def decode_pose_joint_pos_mj(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``joint_pos_mj`` (f32, shape (NUM_BODY_DOFS,)) from a packed
    pose frame.

    Mirrors ``decode_x2_debug_base_quat``: tolerant of header noise,
    returns ``None`` on any decode failure rather than raising (the
    publish thread must survive a malformed cached frame -- the worst
    case is we lose the HOLD-vs-BLEND lerp anchor and fall back to the
    baked idle clip, which is no worse than the legacy behaviour).

    Called once per fresh upstream tick (~50 Hz) to snapshot the most
    recent operator-commanded joint targets. When upstream goes silent
    the snapshot is reused as (a) the byte payload to re-publish during
    HOLD and (b) the lerp anchor for BLEND.
    """
    return _decode_pose_field_f32(
        msg, topic, name="joint_pos_mj", expected_shape=(NUM_BODY_DOFS,)
    )


def decode_pose_left_hand(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``left_hand_joints`` (f32, shape (DEFAULT_HAND_DOF,))."""
    return _decode_pose_field_f32(
        msg, topic, name="left_hand_joints",
        expected_shape=(DEFAULT_HAND_DOF,),
    )


def decode_pose_right_hand(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``right_hand_joints`` (f32, shape (DEFAULT_HAND_DOF,))."""
    return _decode_pose_field_f32(
        msg, topic, name="right_hand_joints",
        expected_shape=(DEFAULT_HAND_DOF,),
    )


def decode_pose_joint_pos_mj_future(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``joint_pos_mj_future`` (f32, (NUM_FUTURE_SLOTS, NUM_BODY_DOFS)).

    Returns ``None`` if the field is absent (legacy v4 upstream like
    a legacy planner that never set the future window) or any
    decode failure. The watchdog uses this to snapshot the upstream
    policy's body-trajectory horizon at every fresh tick so the BLEND
    state can lerp the cached upstream future toward the idle clip
    future, eliminating the one-tick "policy planning horizon flips"
    snap that the original 2026-06-08 BLEND lerp left in place (it only
    interpolated the current ``joint_pos_mj``).
    """
    return _decode_pose_field_f32(
        msg, topic, name="joint_pos_mj_future",
        expected_shape=(NUM_FUTURE_SLOTS, NUM_BODY_DOFS),
    )


def decode_pose_root_quat_xyzw_future(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``root_quat_xyzw_future`` (f32, (NUM_FUTURE_SLOTS, 4)).

    See ``decode_pose_joint_pos_mj_future`` -- same purpose, same v4
    fallback semantics. The cached upstream future quats are assumed
    to already be in the live-heading frame (the VLA bridge applies
    yaw rebase pre-publish as of 2026-06-23), so the watchdog must
    NOT yaw-rebase them a second time before lerping.
    """
    return _decode_pose_field_f32(
        msg, topic, name="root_quat_xyzw_future",
        expected_shape=(NUM_FUTURE_SLOTS, 4),
    )


def decode_pose_joint_vel_mj_future(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``joint_vel_mj_future`` (f32, (NUM_FUTURE_SLOTS, NUM_BODY_DOFS)).

    Optional v5 field -- the C++ deploy backward-finite-diffs the
    future jpos array if this field is absent. Watchdog snapshots it
    so the BLEND lerp can decay velocities smoothly toward the idle
    clip's (mostly-zero) future jvel rather than letting them snap.
    """
    return _decode_pose_field_f32(
        msg, topic, name="joint_vel_mj_future",
        expected_shape=(NUM_FUTURE_SLOTS, NUM_BODY_DOFS),
    )


def decode_pose_motion_token(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``motion_token`` (f32, shape (SONIC_MOTION_TOKEN_DIM,)).

    Latched into the deploy's tokenizer obs at line 309 of
    ``zmq_pose_input_source.cpp``. When the bridge stops publishing,
    the original BLEND path forced this to ``ZERO_MOTION_TOKEN`` on
    the first BLEND tick -- a one-tick discontinuity in the policy's
    intent signal that contributed to the shoulder/elbow snap. The
    watchdog snapshots the latest upstream token so BLEND can decay
    it toward zero over the blend window instead of slamming it.
    """
    return _decode_pose_field_f32(
        msg, topic, name="motion_token",
        expected_shape=(SONIC_MOTION_TOKEN_DIM,),
    )


def _decode_pose_field_f32(
    msg: bytes,
    topic: str,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Generic packed-pose field extractor for f32 vectors.

    Walks the v4 header field list cursor-by-cursor (same scheme as
    the deploy's ZmqPoseInputSource) and returns the named field as
    a copied numpy array when the dtype + shape match, else ``None``.
    Tolerates header decode errors so the publish thread can survive
    a malformed cached frame.
    """
    topic_bytes = topic.encode("utf-8")
    if not msg.startswith(topic_bytes):
        return None
    body = msg[len(topic_bytes):]
    if len(body) < HEADER_SIZE:
        return None
    header_blob = body[:HEADER_SIZE].rstrip(b"\x00")
    payload = body[HEADER_SIZE:]
    try:
        header = json.loads(header_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    fields = header.get("fields")
    if not isinstance(fields, list):
        return None
    cursor = 0
    for f in fields:
        try:
            fname = str(f["name"])
            dtype = str(f["dtype"])
            shape = tuple(int(s) for s in f["shape"])
        except (KeyError, TypeError, ValueError):
            return None
        bpe = _DTYPE_BPE.get(dtype)
        if bpe is None:
            return None
        nelem = 1
        for s in shape:
            nelem *= s
        nbytes = nelem * bpe
        if cursor + nbytes > len(payload):
            return None
        if fname == name:
            if dtype != "f32" or shape != expected_shape:
                return None
            flat = np.frombuffer(
                payload[cursor:cursor + nbytes], dtype="<f4"
            ).copy()
            if expected_shape != (nelem,):
                flat = flat.reshape(expected_shape)
            return flat
        cursor += nbytes
    return None  # named field absent (e.g. token-only side-channel frame)


# ---------------------------------------------------------------------------
# Yaw extraction + yaw-rebase math. Closed-form so the watchdog stays
# scipy-free on PC2 (validated against scipy's Rotation.as_euler("zyx")[0]
# to 5.7e-14 deg over 2000 random rotations -- see unit tests). The
# convention matches every other yaw-touching site in the stack
# (gear_sonic.utils.planner.blending.yaw_of_quat_xyzw) so a yaw
# extracted here can be passed unchanged into kplanner / recorder /
# scipy code anywhere else.
# ---------------------------------------------------------------------------
def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Extract world-z yaw (rad, in ``(-pi, pi]``) from a wxyz quat.

    Equivalent to ``scipy.spatial.transform.Rotation.from_quat(
    [qx,qy,qz,qw]).as_euler("zyx")[0]`` (extrinsic ZYX, lowercase).
    Derivation: for ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)``,
    ``yaw = atan2(-R[0][1], R[0][0])``; substituting the quat->matrix
    closed form gives the expression below.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quat_wxyz must be length 4, got {q.shape[0]}")
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return float(math.atan2(
        2.0 * (qw * qz - qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def nlerp_quat_arrays_xyzw(
    q_from: np.ndarray, q_to: np.ndarray, alpha: float
) -> np.ndarray:
    """Normalized lerp between two batches of xyzw quats.

    Per-row normalized linear interpolation with dot-sign correction
    (flip ``q_to`` to ``-q_to`` when ``dot(q_from, q_to) < 0`` so the
    interpolation takes the short great-circle path; q and -q
    represent the same orientation so the sign flip is identity in
    rotation space but flips the lerp direction). Returns a new
    array of the same dtype + shape as ``q_from``.

    Why nlerp instead of full slerp: the watchdog's PC2 venv is
    scipy-free (numpy + pyzmq + stdlib budget) and slerp would
    require ``scipy.spatial.transform.Rotation`` or a hand-rolled
    sin/atan2 path. For the per-tick alpha increments we use
    (1 / (blend_secs * rate_hz) ~= 0.007 per tick at 50 Hz / 3 s)
    and the typical quat deltas across the BLEND window (~10-30 deg
    yaw, idle clip ~ identity), nlerp matches slerp to better than
    0.5 deg per tick -- well below the deploy's target_lpf_hz=8 Hz
    output bandwidth.

    Degenerate inputs (`q_from + q_to_aligned` near zero, i.e.
    truly antipodal even after sign correction) are extremely
    unlikely in practice but fall back to ``q_from`` rather than
    producing NaN. Either endpoint is a valid orientation; the
    cached upstream side is the safer choice if we're already
    seeing pathological data.
    """
    qf = np.asarray(q_from, dtype=np.float64)
    qt = np.asarray(q_to, dtype=np.float64)
    if qf.shape != qt.shape:
        raise ValueError(
            f"nlerp_quat_arrays_xyzw: shape mismatch q_from={qf.shape} "
            f"q_to={qt.shape}"
        )
    if qf.ndim != 2 or qf.shape[1] != 4:
        raise ValueError(
            f"nlerp_quat_arrays_xyzw: expected (N, 4); got {qf.shape}"
        )
    a = float(alpha)
    dot = np.sum(qf * qt, axis=1, keepdims=True)
    qt_aligned = np.where(dot < 0.0, -qt, qt)
    lerp = (1.0 - a) * qf + a * qt_aligned
    norm = np.linalg.norm(lerp, axis=1, keepdims=True)
    safe = norm > 1e-9
    out = np.where(safe, lerp / np.where(safe, norm, 1.0), qf)
    return out.astype(np.asarray(q_from).dtype, copy=False)


def rebase_quats_xyzw_by_yaw(
    quats_xyzw: np.ndarray, yaw_rad: float
) -> np.ndarray:
    """Pre-multiply a batch of xyzw quats by ``R_z(yaw)``.

    ``quats_xyzw`` is (N, 4); returns (N, 4) of the same dtype.
    Applied to the baked idle clip's (yaw=0-aligned) root quats so the
    published reference matches the robot's actual heading instead of
    world +X. Vectorized for batch efficiency since we rebase the
    current frame + 9 future-window slots on every idle tick.
    """
    q = np.asarray(quats_xyzw)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats_xyzw must be (N, 4); got {q.shape}")
    half = 0.5 * float(yaw_rad)
    rzx, rzy, rzz, rzw = 0.0, 0.0, math.sin(half), math.cos(half)
    qx = q[:, 0].astype(np.float64, copy=False)
    qy = q[:, 1].astype(np.float64, copy=False)
    qz = q[:, 2].astype(np.float64, copy=False)
    qw = q[:, 3].astype(np.float64, copy=False)
    out = np.empty_like(q, dtype=q.dtype)
    out[:, 0] = (rzw * qx + rzx * qw + rzy * qz - rzz * qy).astype(q.dtype)
    out[:, 1] = (rzw * qy - rzx * qz + rzy * qw + rzz * qx).astype(q.dtype)
    out[:, 2] = (rzw * qz + rzx * qy - rzy * qx + rzz * qw).astype(q.dtype)
    out[:, 3] = (rzw * qw - rzx * qx - rzy * qy - rzz * qz).astype(q.dtype)
    return out


# ---------------------------------------------------------------------------
# X2M2 loader (matches the format bake_idle_stand_x2m2.py emits, which
# is byte-compatible with PklMotionReference::Load on the C++ side).
# Used by the watchdog to load the baked idle_stand clip at startup.
# ---------------------------------------------------------------------------
def load_x2m2(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load an X2M2 binary into (dof, quat_xyzw, fps).

    ``dof`` is ``(num_frames, NUM_BODY_DOFS)`` f32; ``quat`` is
    ``(num_frames, 4)`` f32 xyzw. Raises ``ValueError`` on any header
    mismatch (bad magic, wrong dof count, payload length mismatch);
    the watchdog refuses to start without a valid idle clip.
    """
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"{path}: file too short ({len(raw)} bytes)")
    magic, num_frames, num_dofs = struct.unpack("<III", raw[:12])
    if magic != X2M2_MAGIC:
        raise ValueError(
            f"{path}: bad magic 0x{magic:08x}, expected 0x{X2M2_MAGIC:08x}"
        )
    if num_dofs != NUM_BODY_DOFS:
        raise ValueError(
            f"{path}: num_dofs={num_dofs}, expected {NUM_BODY_DOFS}"
        )
    fps = struct.unpack("<d", raw[12:20])[0]
    frame_bytes = (num_dofs + 4) * 8
    expected = 20 + num_frames * frame_bytes
    if len(raw) != expected:
        raise ValueError(
            f"{path}: payload len {len(raw)} != expected {expected}"
        )
    flat = np.frombuffer(raw[20:], dtype=np.float64).reshape(
        num_frames, num_dofs + 4
    )
    dof = np.ascontiguousarray(flat[:, :num_dofs], dtype=np.float32)
    quat = np.ascontiguousarray(flat[:, num_dofs:], dtype=np.float32)
    return dof, quat, fps


# ---------------------------------------------------------------------------
# Wire encoder (inlined copy of
# gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message so
# both processes stand alone without the full gear_sonic install on
# PYTHONPATH).
# ---------------------------------------------------------------------------
def pack_pose_message(
    pose_data: dict, topic: str = "pose", version: int = 4
) -> bytes:
    """Encode a pose dict into the v4 packed wire format.

    ``pose_data`` is a flat dict of ``{field_name: ndarray}``. Each
    value's dtype is preserved if it's one of (f32, f64, i32, i64,
    bool); anything else is cast to f32 (with a side effect on the
    dict's view via assignment, which is benign because we use a
    local ``value`` rebind). Field order in the on-wire payload
    matches Python's insertion order on the dict, which means
    callers must populate the dict in the same field order the
    consumer expects.
    """
    fields: list[dict] = []
    binary_data: list[bytes] = []
    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype == np.float32:
            dtype_str = "f32"
        elif value.dtype == np.float64:
            dtype_str = "f64"
        elif value.dtype == np.int32:
            dtype_str = "i32"
        elif value.dtype == np.int64:
            dtype_str = "i64"
        elif value.dtype == bool:
            dtype_str = "bool"
        else:
            dtype_str = "f32"
            value = value.astype(np.float32)
        fields.append(
            {"name": key, "dtype": dtype_str, "shape": list(value.shape)}
        )
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        binary_data.append(value.tobytes())
    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(
            f"Header too large: {len(header_json)} > {HEADER_SIZE}"
        )
    header_bytes = header_json.ljust(HEADER_SIZE, b"\x00")
    return topic.encode("utf-8") + header_bytes + b"".join(binary_data)
