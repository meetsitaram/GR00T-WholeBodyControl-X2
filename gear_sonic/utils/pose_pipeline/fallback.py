"""Staged HOLD/BLEND/IDLE_CLIP fallback ladder.

Used exclusively by :mod:`gear_sonic_deploy.scripts.x2_pose_watchdog`
on PC2. When the upstream wire stalls (laptop crash, wifi outage,
mux death), the watchdog runs this ladder to keep the C++ deploy's
wire alive WITHOUT stepping the commanded reference -- the
pre-2026-06-08 single-step-to-default behaviour slammed the arms
through their full ROM on every WiFi blip.

Ladder:

* ``LIVE``       -> upstream fresh this tick; forward bytes verbatim.
* ``COLD_IDLE``  -> proxy has NEVER seen upstream; publish baked idle
                    clip (startup; legacy behaviour preserved).
* ``HOLD``       -> upstream silent < hold_last_secs after stale
                    threshold; re-publish the LAST forwarded upstream
                    bytes. The deploy keeps tracking the operator's
                    last pose with jvel=0 (identical bytes, no
                    kinematic surprise).
* ``BLEND``      -> upstream silent past hold_last_secs but still
                    within blend_secs window; lerp joint_pos_mj from
                    cached upstream toward baked idle. The lerp is
                    monotonic so the deploy sees a smooth (~3 s)
                    glide rather than a step.
* ``IDLE_CLIP``  -> upstream silent past hold_last_secs + blend_secs;
                    publish baked idle clip indefinitely (legacy
                    destination behaviour, just reached gradually).

Mode selection via ``idle_mode``:

* ``blend``      -> the full ladder above (DEFAULT; safe).
* ``hold-last``  -> HOLD forever; never transitions to BLEND/IDLE_CLIP.
                    Operator-responsibility mode; use when you know
                    upstream WILL come back.
* ``idle-stand`` -> skip HOLD/BLEND entirely; jump to IDLE_CLIP on
                    first stale tick. Reproduces pre-2026-06-08
                    behaviour exactly. Regression escape only.
"""

from __future__ import annotations

import numpy as np

from .wire import (
    FUTURE_STEP_TICKS,
    FUTURE_DT_FIELD,
    HEADER_SIZE,  # noqa: F401  -- re-exported for backward compat
    NUM_BODY_DOFS,
    NUM_FUTURE_SLOTS,
    ZERO_HAND,
    ZERO_MOTION_TOKEN,
    ZERO_QVEL_FUTURE,
    pack_pose_message,
    rebase_quats_xyzw_by_yaw,
    rebuild_msg_with_field_overrides,
)


# ---------------------------------------------------------------------------
# State machine constants. Exported so the watchdog's status-line code
# and the test suite can refer to them by name rather than string
# literal.
# ---------------------------------------------------------------------------
STATE_LIVE: str = "LIVE"
STATE_COLD_IDLE: str = "COLD_IDLE"
STATE_HOLD: str = "HOLD"
STATE_BLEND: str = "BLEND"
STATE_IDLE_CLIP: str = "IDLE_CLIP"
# Silent < stale_s; don't publish (deploy holds last forwarded ref).
STATE_GAP: str = "GAP"

IDLE_MODE_BLEND: str = "blend"
IDLE_MODE_HOLD_LAST: str = "hold-last"
IDLE_MODE_IDLE_STAND: str = "idle-stand"
IDLE_MODES: tuple[str, ...] = (
    IDLE_MODE_BLEND,
    IDLE_MODE_HOLD_LAST,
    IDLE_MODE_IDLE_STAND,
)


# ---------------------------------------------------------------------------
# Idle-stand replay (matches _IdleStandLoop in
# gear_sonic/scripts/vla/live_vla_publish_motion_token.py, condensed).
# ---------------------------------------------------------------------------
class IdleStandReplay:
    """Looped playback over a baked X2M2 idle clip."""

    def __init__(self, dof: np.ndarray, quat: np.ndarray) -> None:
        if dof.ndim != 2 or dof.shape[1] != NUM_BODY_DOFS:
            raise ValueError(
                f"dof must be (T, {NUM_BODY_DOFS}); got {dof.shape}"
            )
        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"quat must be (T, 4) xyzw; got {quat.shape}")
        if dof.shape[0] != quat.shape[0]:
            raise ValueError(
                f"dof / quat length mismatch: {dof.shape[0]} vs "
                f"{quat.shape[0]}"
            )
        if dof.shape[0] < 1:
            raise ValueError("idle clip is empty")
        self._dof = np.ascontiguousarray(dof, dtype=np.float32)
        self._quat = np.ascontiguousarray(quat, dtype=np.float32)
        self._n = int(dof.shape[0])

    @property
    def n_frames(self) -> int:
        return self._n

    def current(self, tick: int) -> tuple[np.ndarray, np.ndarray]:
        i = int(tick) % self._n
        return self._dof[i].copy(), self._quat[i].copy()

    def future_window(
        self, tick: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.array(
            [
                (int(tick) + (k + 1) * FUTURE_STEP_TICKS) % self._n
                for k in range(NUM_FUTURE_SLOTS)
            ],
            dtype=np.int64,
        )
        return self._dof[idx].copy(), self._quat[idx].copy(), ZERO_QVEL_FUTURE


def build_idle_frame_msg(
    replay: IdleStandReplay,
    tick: int,
    topic: str,
    *,
    yaw_rebase_rad: float | None = None,
    joint_pos_mj_override: np.ndarray | None = None,
    joint_pos_mj_future_override: np.ndarray | None = None,
    root_quat_xyzw_future_override: np.ndarray | None = None,
    joint_vel_mj_future_override: np.ndarray | None = None,
    motion_token_override: np.ndarray | None = None,
) -> bytes:
    """Pack one idle-fallback frame, optionally yaw-rebased.

    ``yaw_rebase_rad`` (radians) is the current measured pelvis yaw the
    watchdog snapshot from the latest fresh ``x2_debug`` frame. When
    provided, every published quat (current frame + 9 future slots) is
    pre-multiplied by ``R_z(yaw_rebase_rad)`` so the deploy's tokenizer
    sees ``rel = inv(measured) * R_z(measured) ~= identity`` and the
    policy holds whatever heading the body is currently in instead of
    twisting back to world +X. ``None`` falls back to the legacy
    "publish baked yaw verbatim" behaviour (kept for the
    ``--no-x2-debug-yaw-track`` regression escape and the unit tests).

    ``joint_pos_mj_override`` (optional, shape (NUM_BODY_DOFS,) f32)
    lets the BLEND state machine substitute a lerp between cached-
    upstream and the baked idle clip for the CURRENT frame's joint
    targets while reusing the rest of the frame.

    The four ``*_future_override`` / ``motion_token_override`` kwargs
    (added 2026-06-23 in the BLEND future-window continuity fix) let
    the watchdog substitute lerps for the ENTIRE upstream-derived wire
    frame, not just the current ``joint_pos_mj``. This eliminates the
    one-tick discontinuity at the HOLD -> BLEND boundary that caused
    the policy's planning horizon (and intent token) to flip from the
    last VLA frame's values to the idle clip's values, producing the
    shoulder/elbow snap operators observed at ~T+stale+hold seconds
    after upstream went silent.

    Yaw-rebase frame convention for ``root_quat_xyzw_future_override``:
    the watchdog passes pre-rebased futures (the cached upstream futures
    are already in the live-heading frame thanks to the 2026-06-23 VLA
    bridge yaw-hold-last-good fix, and the idle-clip side is rebased
    by the watchdog BEFORE lerping so both lerp endpoints share a
    frame). When the override is set we therefore SKIP the second
    rebase that would otherwise apply via ``yaw_rebase_rad``. The
    same convention applies to ``root_quat_xyzw`` indirectly: the
    BLEND path passes ``yaw_rebase_rad`` for the current quat (the
    idle clip's current frame is still rebased), but the future
    override is treated as already-frame-correct.
    """
    cur_jpos, cur_quat = replay.current(tick)
    if joint_pos_mj_override is not None:
        override = np.asarray(joint_pos_mj_override, dtype=np.float32)
        if override.shape != (NUM_BODY_DOFS,):
            raise ValueError(
                "joint_pos_mj_override must have shape "
                f"({NUM_BODY_DOFS},); got {override.shape}"
            )
        cur_jpos = override
    jpos_future, quat_future, jvel_future = replay.future_window(tick)
    if yaw_rebase_rad is not None:
        cur_quat = rebase_quats_xyzw_by_yaw(
            cur_quat.reshape(1, 4), yaw_rebase_rad
        ).reshape(4)
        quat_future = rebase_quats_xyzw_by_yaw(quat_future, yaw_rebase_rad)
    if joint_pos_mj_future_override is not None:
        jpos_future = _validated_future_override(
            joint_pos_mj_future_override,
            name="joint_pos_mj_future_override",
            shape=(NUM_FUTURE_SLOTS, NUM_BODY_DOFS),
        )
    if root_quat_xyzw_future_override is not None:
        # The override is assumed already in the live-heading frame
        # (the BLEND lerp endpoints are both pre-rebased), so DO NOT
        # apply yaw_rebase_rad a second time here.
        quat_future = _validated_future_override(
            root_quat_xyzw_future_override,
            name="root_quat_xyzw_future_override",
            shape=(NUM_FUTURE_SLOTS, 4),
        )
    if joint_vel_mj_future_override is not None:
        jvel_future = _validated_future_override(
            joint_vel_mj_future_override,
            name="joint_vel_mj_future_override",
            shape=(NUM_FUTURE_SLOTS, NUM_BODY_DOFS),
        )
    motion_token_value = ZERO_MOTION_TOKEN
    if motion_token_override is not None:
        motion_token_value = _validated_future_override(
            motion_token_override,
            name="motion_token_override",
            shape=ZERO_MOTION_TOKEN.shape,
        )
    fidx_future = np.array(
        [tick + (k + 1) for k in range(NUM_FUTURE_SLOTS)],
        dtype=np.int64,
    )
    # Field order + dtypes mirror live_vla_publish_motion_token._publish_loop
    # so the deploy's ZmqPoseInputSource decodes our idle frames the same
    # way it decodes laptop bridge frames (no v5 fallback path).
    payload = {
        "joint_pos_mj": cur_jpos,
        "root_quat_xyzw": cur_quat,
        "motion_token": motion_token_value,
        "left_hand_joints": ZERO_HAND,
        "right_hand_joints": ZERO_HAND,
        "frame_index": np.array([tick], dtype=np.int64),
        "joint_pos_mj_future": jpos_future,
        "root_quat_xyzw_future": quat_future,
        "joint_vel_mj_future": jvel_future,
        "frame_index_future": fidx_future,
        "future_dt_s": FUTURE_DT_FIELD,
    }
    return pack_pose_message(payload, topic=topic, version=4)


def _yaw_to_quat_xyzw(yaw_rad: float) -> np.ndarray:
    """Build ``R_z(yaw)`` as a length-4 xyzw quaternion (f32).

    Matches the convention used by ``rebase_quats_xyzw_by_yaw``: a
    pure-yaw quaternion with zero pitch/roll. Centralised so the
    HOLD-rebase path and any future call sites agree on the exact
    half-angle / dtype semantics (no per-call-site drift in float
    precision or array layout).
    """
    import math as _math
    half = 0.5 * float(yaw_rad)
    return np.array(
        [0.0, 0.0, _math.sin(half), _math.cos(half)],
        dtype=np.float32,
    )


def rebase_hold_msg(
    last_upstream_msg: bytes,
    topic: str,
    yaw_rad: float,
) -> bytes | None:
    """Yaw-rebase a cached upstream HOLD frame to ``R_z(yaw_rad)``.

    The watchdog's HOLD state (originally introduced 2026-06-08 in the
    staged fallback ladder) re-publishes the LAST forwarded upstream
    bytes verbatim so the deploy keeps tracking the operator's final
    pose with zero joint velocity during a brief upstream silence.
    That verbatim re-publish has a known foot-gun: if the cached
    message's ``root_quat_xyzw`` (and future window) describe a
    yaw that no longer matches the robot's actual heading -- because
    the operator manually rotated the robot during HOLD, or the
    laptop stack was killed mid-clip -- then the SONIC policy fights
    the body back to the cached yaw with a noticeable restoring
    torque. Operators perceive this as "the robot snaps back to a
    previous orientation when I kill the stack".

    This helper produces a HOLD-friendly variant of the cached frame:
    every ``root_quat_xyzw`` field (the current frame's quat plus, if
    present, the 9 future-window slots) is REPLACED with a pure-yaw
    ``R_z(yaw_rad)`` quaternion derived from the latest measured
    ``x2_debug`` ``base_quat``. ``joint_pos_mj`` and the future joint
    windows are LEFT UNTOUCHED so the body pose still freezes exactly
    where upstream left it -- only the heading reference tracks the
    robot's actual yaw, freeing the operator to rotate the body by
    hand without the policy springing back.

    Returns the rebased bytes on success, or ``None`` if the splice
    failed (legacy v4 message missing a future window, corrupt
    header, etc.). Callers should fall back to publishing
    ``last_upstream_msg`` verbatim on ``None`` -- the worst case is
    today's pre-fix behaviour, which is strictly safer than dropping
    the frame.

    Splice strategy: we try BOTH overrides
    (``root_quat_xyzw`` + ``root_quat_xyzw_future``) first because
    the future window is the dominant policy input for the planning
    horizon. If that fails (header reports no future field, as in
    pre-v5 producer frames), we retry with just ``root_quat_xyzw``
    so the current-frame quat is still rebased. Only if BOTH
    attempts fail do we return ``None``.

    The future override sets all 9 slots to the SAME ``R_z(yaw_rad)``
    quat -- i.e. "no anticipated yaw motion" -- which is the correct
    HOLD semantic. The cached future window may have encoded
    anticipatory yaw deltas from a locomotion clip's last frame, but
    once upstream is silent we can't honour those deltas anyway, and
    flattening them avoids a yaw-cylinder seam at the HOLD entry.
    Joint-pos future slots are not touched; they keep the clip's
    last-frame anticipatory leg/arm motion intact (the body holds
    the final pose, not snaps to "all frames identical").
    """
    cur_quat = _yaw_to_quat_xyzw(yaw_rad)
    fut_quat = np.tile(cur_quat, (NUM_FUTURE_SLOTS, 1))
    # First try: rebase current + future. v5 producers (laptop bridge,
    # mock VLA in v5 mode, the recorder when a clip is active) include
    # the future window.
    out = rebuild_msg_with_field_overrides(
        last_upstream_msg,
        topic,
        {
            "root_quat_xyzw": cur_quat,
            "root_quat_xyzw_future": fut_quat,
        },
    )
    if out is not None:
        return out
    # Fallback: legacy v4 producer (recorder's _publish_idle path,
    # heuristic planner) that omits the future window. Still rebase
    # the current quat so the policy gets at least a fresh heading.
    return rebuild_msg_with_field_overrides(
        last_upstream_msg,
        topic,
        {"root_quat_xyzw": cur_quat},
    )


def _validated_future_override(
    arr: np.ndarray, *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    """Cast + shape-check a BLEND override array. Raises on mismatch.

    Centralised so every override kwarg in ``build_idle_frame_msg``
    has identical validation semantics and identical error messages
    (instead of growing four near-duplicate copies of the same
    five-line block).
    """
    out = np.asarray(arr, dtype=np.float32)
    if out.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}; got {out.shape}"
        )
    return out


def decide_fallback_state(
    *,
    have_upstream: bool,
    age_s: float,
    stale_s: float,
    hold_last_secs: float,
    blend_secs: float,
    idle_mode: str,
) -> tuple[str, float]:
    """Pure decision function for the no-fresh-upstream branch.

    Returns ``(target_state, blend_alpha)`` where ``blend_alpha`` is
    only meaningful for ``STATE_BLEND`` (0.0 = cached, 1.0 = idle).
    Split out as a pure function so the state transitions can be unit
    tested without spinning up ZMQ sockets.

    The "fallback clock" starts at ``stale_s``: ``HOLD`` runs from
    ``stale_s`` to ``stale_s + hold_last_secs``, ``BLEND`` from
    ``stale_s + hold_last_secs`` to ``stale_s + hold_last_secs +
    blend_secs``, and ``IDLE_CLIP`` afterwards. This way the on-the-
    wire behaviour matches what an operator would naively expect from
    the CLI knobs (e.g. ``--hold-last-secs=5`` means 5 s of HOLD,
    not "5 s minus stale threshold of HOLD").
    """
    if not have_upstream:
        return STATE_COLD_IDLE, 0.0
    if age_s <= stale_s:
        return STATE_GAP, 0.0
    fallback_age = age_s - stale_s
    if idle_mode == IDLE_MODE_IDLE_STAND:
        return STATE_IDLE_CLIP, 1.0
    if idle_mode == IDLE_MODE_HOLD_LAST:
        return STATE_HOLD, 0.0
    if fallback_age <= hold_last_secs:
        return STATE_HOLD, 0.0
    if fallback_age <= hold_last_secs + blend_secs:
        if blend_secs <= 0.0:
            return STATE_IDLE_CLIP, 1.0
        alpha = (fallback_age - hold_last_secs) / blend_secs
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        return STATE_BLEND, float(alpha)
    return STATE_IDLE_CLIP, 1.0
