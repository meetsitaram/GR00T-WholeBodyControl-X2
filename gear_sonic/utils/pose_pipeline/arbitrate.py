"""Stateful N-to-1 pose arbitration machine.

Used by :mod:`gear_sonic.scripts.x2_pose_mux`. Encapsulates the
takeover state machine that arbitrates between a primary autonomous
source (VLA bridge) and an operator override stream (Quest 3 manager
via the recorder).

State the arbiter owns:

* Override stream freshness (silence-debounce timer).
* Frozen-frame detection (the Quest 3 manager publishes the FROZEN
  last commanded pose every tick in OFF / LOCOMOTION mode, so the
  override SUB never goes silent across an A+B+X+Y disengage. Frame-
  equality detection catches this and fires release exactly once
  after N consecutive identical override frames).
* Motion-hysteresis engage gate (require N consecutive override
  frames with joint-space delta ABOVE tolerance before engaging --
  prevents brief jitter from spurious engage / release cycles, each
  of which kicks the bridge into a heavy cold-restart).
* Stream-mode strict engage gate (when the manager's stream_mode PUB
  is configured -- mode != "OFF" is the truth and motion-hysteresis
  is bypassed; the operator holding the controller still no longer
  flicker-releases the wire).
* Engagement slow-step ramp (rate-clamp the operator's first OVERRIDE
  frames per-element relative to the last forwarded autonomous pose
  so the deploy doesn't see a single-tick jump across the takeover
  edge).
* Edge event emission state (override_engaged on the engage edge,
  override_released on the release edge, each fired exactly once per
  edge with the operator's last commanded pose snapshot attached to
  the release payload).

Usage pattern (from the mux main loop):

    arb = TakeoverArbiter(... config ...)
    while True:
        now = time.monotonic()
        if got_primary_msg:
            arb.observe_primary(primary_msg, now=now)
        if got_override_msg:
            arb.observe_override(override_msg, now=now)
        if got_teleop_mode_msg:
            arb.observe_teleop_mode(mode_str, now=now)
        decision = arb.decide(now=now, tick=tick)
        # decision.edge in {"none", "engaged", "released"}
        # decision.source in {"primary", "override", "neither"}
        if decision.source == "override":
            msg, jpos = arb.maybe_clamp_override(latest_override_msg)
        ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .clamp import clamp_vector_step_f32
from .wire import (
    NUM_BODY_DOFS,
    NUM_FUTURE_SLOTS,
    ZERO_QVEL_FUTURE,
    decode_pose_joint_pos_mj,
    decode_pose_left_hand,
    decode_pose_right_hand,
    rebuild_msg_with_field_overrides,
    rebuild_msg_with_jpos_override,
)


# ---------------------------------------------------------------------------
# Decision protocol
# ---------------------------------------------------------------------------
EDGE_NONE: str = "none"
EDGE_ENGAGED: str = "engaged"
EDGE_RELEASED: str = "released"

SOURCE_PRIMARY: str = "primary"
SOURCE_OVERRIDE: str = "override"
SOURCE_NEITHER: str = "neither"


@dataclass
class ArbiterDecision:
    """One tick's verdict from the arbiter.

    * ``source`` -- which stream owns the wire this tick.
    * ``edge`` -- ``"engaged"`` / ``"released"`` / ``"none"``. Set on
      the tick where the engage / release transition fires.
    * ``release_event_payload`` -- when ``edge == "released"``, the
      pre-encoded JSON event bytes to send on the vla_control PUB
      (includes ``release_pose`` with the operator's last commanded
      body + hand vectors). ``None`` otherwise.
    * ``engage_event_payload`` -- when ``edge == "engaged"``, the
      pre-encoded JSON event bytes to send on vla_control. ``None``
      otherwise.
    """

    source: str
    edge: str
    release_event_payload: Optional[bytes] = None
    engage_event_payload: Optional[bytes] = None


@dataclass
class ArbiterConfig:
    """All knobs the arbiter cares about. Defaults match what the
    legacy x2_pose_proxy.py used in production (2026-06-10 -- the
    last "good" production tuning before the split)."""

    upstream_topic: str = "pose"
    downstream_topic: str = "pose"
    override_stale_s: float = 0.200
    frozen_ticks_threshold: int = 10
    frozen_l2_tol: float = 5e-3
    engage_motion_threshold: int = 10
    teleop_mode_enabled: bool = False
    teleop_mode_stale_s: float = 1.0
    engagement_max_wire_step: float = 0.012
    engagement_steady_wire_step: float = 0.035
    engagement_step_ramp_ticks: int = 250


# ---------------------------------------------------------------------------
# TakeoverArbiter
# ---------------------------------------------------------------------------
class TakeoverArbiter:
    """Stateful arbiter for primary VLA + operator override pose streams."""

    def __init__(self, cfg: ArbiterConfig) -> None:
        self.cfg = cfg

        # Override stream tracking.
        self._last_override_s: float = -1.0
        self._override_active: bool = False

        # Engage/release counters (observability + status line).
        self.override_engage_events: int = 0
        self.override_release_events: int = 0
        self.override_frozen_release_events: int = 0
        self.override_frames_forwarded: int = 0

        # Frozen-frame detection.
        self._prev_override_jpos: Optional[np.ndarray] = None
        self._override_frozen_count: int = 0
        self._override_frozen_detected: bool = False

        # Motion hysteresis on engage.
        self._override_motion_count: int = 0

        # Operator-pose handoff snapshots (for the release payload).
        self._last_override_left_hand: Optional[np.ndarray] = None
        self._last_override_right_hand: Optional[np.ndarray] = None

        # Stream-mode strict engage gate state.
        self._current_teleop_mode: Optional[str] = None
        self._last_teleop_mode_s: float = -1.0
        self.teleop_mode_msgs: int = 0
        self.teleop_mode_decode_failures: int = 0

        # Engagement ramp state.
        self._engagement_clamp_remaining: int = 0
        self._engagement_last_forwarded_jpos: Optional[np.ndarray] = None

        # Anchor of the last forwarded primary frame's joint_pos_mj.
        # Used to seed the engagement-ramp clamp at the LIVE -> OVERRIDE
        # edge: the clamp anchors at the last VLA pose so the
        # operator's first OVERRIDE frame can't jump the wire more
        # than ``engagement_max_wire_step`` per joint per tick.
        self._last_primary_jpos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------
    def observe_primary(self, msg: bytes, *, now: float) -> None:
        """Snapshot the primary frame's joint_pos_mj for the engagement
        ramp anchor. The frame itself is forwarded by the mux's main
        loop; the arbiter only cares about the jpos slice for the
        engage edge.
        """
        jpos = decode_pose_joint_pos_mj(msg, self.cfg.upstream_topic)
        if jpos is None:
            jpos = decode_pose_joint_pos_mj(msg, self.cfg.downstream_topic)
        if jpos is not None:
            self._last_primary_jpos = jpos.astype(np.float32, copy=True)

    def observe_override(self, msg: bytes, *, now: float) -> None:
        """Update override-freshness clock + run frozen / motion
        detectors against the previous override frame."""
        self._last_override_s = now

        ojpos = decode_pose_joint_pos_mj(msg, self.cfg.upstream_topic)
        if ojpos is None:
            ojpos = decode_pose_joint_pos_mj(msg, self.cfg.downstream_topic)

        # Cache operator-commanded hand joints alongside body so we
        # can pack them into the release event. Both fields are
        # optional in the wire format -- legacy token-only frames
        # omit them -- so we keep whichever we got and leave the rest
        # at the last seen value (initially None). The release-event
        # packer falls back to omission when None.
        olh = decode_pose_left_hand(msg, self.cfg.upstream_topic)
        if olh is None:
            olh = decode_pose_left_hand(msg, self.cfg.downstream_topic)
        if olh is not None:
            self._last_override_left_hand = olh
        orh = decode_pose_right_hand(msg, self.cfg.upstream_topic)
        if orh is None:
            orh = decode_pose_right_hand(msg, self.cfg.downstream_topic)
        if orh is not None:
            self._last_override_right_hand = orh

        if ojpos is None:
            return

        # Frozen-frame detector. The quest3_manager publishes the
        # frozen last commanded pose every tick in OFF / LOCOMOTION
        # (see manager lines 1221-1229), so the override SUB never
        # goes silent across an A+B+X+Y disengage gesture. Tracking
        # consecutive override frames whose joint_pos_mj delta from
        # the previous frame is within tolerance catches this. When
        # the streak crosses the threshold, latch
        # ``_override_frozen_detected`` so the engage gate trips
        # release exactly once.
        cfg = self.cfg
        if self._prev_override_jpos is None:
            self._override_frozen_count = 0
            self._override_motion_count = 0
        else:
            delta = float(np.linalg.norm(
                ojpos.astype(np.float64)
                - self._prev_override_jpos.astype(np.float64)
            ))
            if delta <= cfg.frozen_l2_tol:
                self._override_frozen_count += 1
                self._override_motion_count = 0
            else:
                # Operator is moving again. Clear the frozen latch
                # so future frozen streaks can re-fire release; bump
                # the motion streak so the engage-hysteresis can
                # eventually trip.
                self._override_frozen_count = 0
                self._override_frozen_detected = False
                self._override_motion_count += 1
        self._prev_override_jpos = ojpos.astype(np.float32, copy=True)

        if (
            cfg.frozen_ticks_threshold > 0
            and not self._override_frozen_detected
            and self._override_frozen_count >= cfg.frozen_ticks_threshold
        ):
            self._override_frozen_detected = True
            self.override_frozen_release_events += 1

    def observe_teleop_mode(self, mode: str, *, now: float) -> None:
        """Snapshot the manager's latest stream_mode value."""
        self._current_teleop_mode = mode
        self._last_teleop_mode_s = now
        self.teleop_mode_msgs += 1

    def record_teleop_mode_decode_failure(self) -> None:
        """Bump the decode-failure counter (status line)."""
        self.teleop_mode_decode_failures += 1

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def decide(
        self, *, now: float, tick: int,
        primary_fresh: bool,
        override_recvd_this_tick: bool,
    ) -> ArbiterDecision:
        """Per-tick verdict + edge events.

        ``primary_fresh`` is True iff the mux's primary SUB received
        at least one frame this tick (or held a fresh one in the
        debounce window). ``override_recvd_this_tick`` is True iff
        the mux's override SUB received at least one frame this tick;
        used by callers that prefer to take the "snapshot now" /
        "decide later" pattern.
        """
        cfg = self.cfg

        teleop_mode_fresh = (
            self._last_teleop_mode_s >= 0
            and (now - self._last_teleop_mode_s) <= cfg.teleop_mode_stale_s
        )

        if cfg.teleop_mode_enabled:
            teleop_engaged = (
                teleop_mode_fresh
                and self._current_teleop_mode is not None
                and self._current_teleop_mode != "OFF"
            )
            override_fresh = (
                self._last_override_s >= 0
                and (now - self._last_override_s) <= cfg.override_stale_s
                and teleop_engaged
            )
        else:
            override_motion_sustained = (
                cfg.engage_motion_threshold == 0
                or self._override_motion_count >= cfg.engage_motion_threshold
            )
            override_fresh = (
                self._last_override_s >= 0
                and (now - self._last_override_s) <= cfg.override_stale_s
                and not self._override_frozen_detected
                and override_motion_sustained
            )

        # ----- Engage edge -------------------------------------------
        edge = EDGE_NONE
        engage_payload: Optional[bytes] = None
        release_payload: Optional[bytes] = None
        if override_fresh and not self._override_active:
            self._override_active = True
            self.override_engage_events += 1
            edge = EDGE_ENGAGED
            # Arm the engagement slow-step ramp. Anchor the clamp at
            # the last successfully forwarded primary pose; under
            # cold-start with no primary anchor available we leave
            # the anchor None and the first override frame seeds it
            # (subsequent ticks clamp relative to that seed).
            if (
                cfg.engagement_step_ramp_ticks > 0
                and cfg.engagement_max_wire_step > 0.0
            ):
                self._engagement_clamp_remaining = (
                    cfg.engagement_step_ramp_ticks
                )
                if self._last_primary_jpos is not None:
                    self._engagement_last_forwarded_jpos = (
                        self._last_primary_jpos.astype(np.float32, copy=True)
                    )
                else:
                    self._engagement_last_forwarded_jpos = None
            engage_payload = json.dumps({
                "event": "override_engaged",
                "ts": now,
                "tick": tick,
            }).encode("utf-8")

        # ----- Release edge ------------------------------------------
        if (not override_fresh) and self._override_active:
            self._override_active = False
            self.override_release_events += 1
            edge = EDGE_RELEASED
            # Tear down the engagement ramp state on release so the
            # next engage edge re-arms cleanly from a fresh primary
            # anchor.
            self._engagement_clamp_remaining = 0
            self._engagement_last_forwarded_jpos = None
            evt_payload: dict[str, object] = {
                "event": "override_released",
                "ts": now,
                "tick": tick,
            }
            release_pose: dict[str, list[float]] = {}
            if self._prev_override_jpos is not None:
                release_pose["joint_pos_mj"] = (
                    self._prev_override_jpos.astype(float).tolist()
                )
            if self._last_override_left_hand is not None:
                release_pose["left_hand_joints"] = (
                    self._last_override_left_hand.astype(float).tolist()
                )
            if self._last_override_right_hand is not None:
                release_pose["right_hand_joints"] = (
                    self._last_override_right_hand.astype(float).tolist()
                )
            if release_pose:
                evt_payload["release_pose"] = release_pose
            release_payload = json.dumps(evt_payload).encode("utf-8")

        # ----- Source selection --------------------------------------
        if override_fresh:
            source = SOURCE_OVERRIDE
        elif primary_fresh:
            source = SOURCE_PRIMARY
        else:
            source = SOURCE_NEITHER

        return ArbiterDecision(
            source=source,
            edge=edge,
            release_event_payload=release_payload,
            engage_event_payload=engage_payload,
        )

    # ------------------------------------------------------------------
    # Engagement slow-step ramp
    # ------------------------------------------------------------------
    def maybe_clamp_override(
        self, msg: bytes
    ) -> tuple[bytes, Optional[np.ndarray]]:
        """Apply the engagement slow-step clamp to ``msg`` if active.

        Returns ``(forwarded_msg, decoded_jpos)``. When the engagement
        ramp is not armed (post-ramp window or cfg disables) this
        forwards ``msg`` verbatim and returns the decoded jpos if
        decodable, ``None`` otherwise.
        """
        cfg = self.cfg
        op_jpos = decode_pose_joint_pos_mj(msg, cfg.upstream_topic)
        if op_jpos is None:
            op_jpos = decode_pose_joint_pos_mj(msg, cfg.downstream_topic)

        if (
            self._engagement_clamp_remaining > 0
            and op_jpos is not None
            and self._engagement_last_forwarded_jpos is not None
        ):
            ramp_progress = 1.0 - float(
                self._engagement_clamp_remaining
            ) / float(max(cfg.engagement_step_ramp_ticks, 1))
            ramp_progress = min(max(ramp_progress, 0.0), 1.0)
            effective_max_step = (
                (1.0 - ramp_progress) * cfg.engagement_max_wire_step
                + ramp_progress * cfg.engagement_steady_wire_step
            )
            clamped_jpos = clamp_vector_step_f32(
                op_jpos,
                self._engagement_last_forwarded_jpos,
                effective_max_step,
            )
            # Flatten the future window: the deploy's window-mode
            # policy reads ``joint_pos_mj_future`` (9 slots, 0.1 s
            # apart) and would slam the body to the operator's
            # untouched future even when the current jpos is properly
            # rate-limited. Broadcasting the clamped current jpos to
            # all 9 slots + zeroing joint_vel_mj_future tells the
            # policy "operator wants to hold here" during the ramp.
            flat_future = np.broadcast_to(
                clamped_jpos, (NUM_FUTURE_SLOTS, NUM_BODY_DOFS)
            ).astype(np.float32, copy=True)
            zero_future_vel = ZERO_QVEL_FUTURE.copy()
            overrides_for_clamp = {
                "joint_pos_mj": clamped_jpos,
                "joint_pos_mj_future": flat_future,
                "joint_vel_mj_future": zero_future_vel,
            }
            rebuilt = rebuild_msg_with_field_overrides(
                msg, cfg.upstream_topic, overrides_for_clamp,
            )
            if rebuilt is None:
                rebuilt = rebuild_msg_with_field_overrides(
                    msg, cfg.downstream_topic, overrides_for_clamp,
                )
            if rebuilt is None:
                # The override frame likely doesn't carry the full v5
                # future window (e.g. a v4 token-only frame from a
                # legacy publisher). Fall back to clamping just the
                # current jpos -- still better than verbatim forward.
                rebuilt = rebuild_msg_with_jpos_override(
                    msg, cfg.upstream_topic, clamped_jpos,
                )
                if rebuilt is None:
                    rebuilt = rebuild_msg_with_jpos_override(
                        msg, cfg.downstream_topic, clamped_jpos,
                    )
            if rebuilt is not None:
                self._engagement_last_forwarded_jpos = (
                    clamped_jpos.copy()
                )
                self._engagement_clamp_remaining -= 1
                return rebuilt, clamped_jpos
            # Defensive: if rebuild failed, forward verbatim. Decrement
            # the clamp counter so we don't stay stuck in the ramp
            # window if every rebuild fails for some structural
            # reason.
            self._engagement_clamp_remaining -= 1
            return msg, op_jpos
        elif (
            self._engagement_clamp_remaining > 0
            and op_jpos is not None
            and self._engagement_last_forwarded_jpos is None
        ):
            # First override tick of the engagement ramp with NO
            # anchor (LIVE never ran on the bridge, so we don't have
            # a primary pose to clamp toward). Seed the anchor from
            # this operator pose and forward verbatim; subsequent
            # ticks will clamp relative to this anchor.
            self._engagement_last_forwarded_jpos = op_jpos.astype(
                np.float32, copy=True
            )
            self._engagement_clamp_remaining -= 1
            return msg, op_jpos

        return msg, op_jpos

    def record_forwarded_override(self) -> None:
        """Bump the forwarded-override counter (status line)."""
        self.override_frames_forwarded += 1

    # ------------------------------------------------------------------
    # Status-line accessors (read-only views into the internal state).
    # ------------------------------------------------------------------
    @property
    def override_active(self) -> bool:
        return self._override_active

    @property
    def last_override_s(self) -> float:
        return self._last_override_s

    @property
    def override_frozen_detected(self) -> bool:
        return self._override_frozen_detected

    @property
    def override_frozen_count(self) -> int:
        return self._override_frozen_count

    @property
    def override_motion_count(self) -> int:
        return self._override_motion_count

    @property
    def current_teleop_mode(self) -> Optional[str]:
        return self._current_teleop_mode

    @property
    def last_teleop_mode_s(self) -> float:
        return self._last_teleop_mode_s

    def teleop_mode_fresh(self, now: float) -> bool:
        return (
            self._last_teleop_mode_s >= 0
            and (now - self._last_teleop_mode_s) <= self.cfg.teleop_mode_stale_s
        )
