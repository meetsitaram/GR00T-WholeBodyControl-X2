"""X2 intent decoder: Quest 3 buttons + thumbsticks -> planner_cmd vocabulary.

This module is the X2 counterpart to the (G1-specific) joystick-to-mode
mapping inside the G1 Quest manager (not shipped).
It is intentionally pure logic — no Quest 3 connection, no ZMQ
dependency — so it can be unit-tested directly and reused by the new
``quest3_manager_x2.py``.

Vocabulary
----------

The output of :meth:`IntentDecoder.decode_locomotion` is the same
``(intent, magnitude)`` tuple format that
:class:`gear_sonic.utils.planner.state_machine.LocomotionCommand`
expects, and matches the ``KEYBOARD_MAP`` defined in
the X2 planner daemon (``pc2_kplanner_onnx.py``) so the planner consumes
manager-emitted commands with no wire-format change.

StreamMode
----------

The X2 manager only uses three modes in Phase 0:

- ``OFF`` — manager is idle; no ``planner_cmd`` is emitted; arms are
  held at their last commanded pose.
- ``LOCOMOTION`` — thumbsticks drive the planner; arm IK output is
  ignored and frozen to the last commanded pose.
- ``ARM_MANIPULATION`` — planner is held at ``idle_stand``; Quest 3
  hand poses drive the arm IK.

This is intentionally a *separate* enum from
:class:`gear_sonic.utils.teleop.common.StreamMode` (which carries
G1-specific PLANNER / PLANNER_FROZEN_UPPER_BODY / PLANNER_VR_3PT
values). Future modes (e.g. ``LOC_AND_ARM`` for parallel control) can
be added here without disturbing G1 callers.

Button vocabulary
-----------------

- ``A+B+X+Y`` chord — toggle ``OFF`` <-> ``LOCOMOTION``  (engage / E-stop)
- ``B`` single  — toggle ``LOCOMOTION`` <-> ``ARM_MANIPULATION`` (when not OFF)

Locomotion vocabulary (LOCOMOTION mode only)
--------------------------------------------

- Left stick forward          -> ``(fwd_step, default)``
- Left stick back             -> ``(back_step, default)``
- Left stick right            -> ``(side_right, default)``
- Left stick left             -> ``(side_left,  default)``
- Right stick hard left/right -> ``(turn_left/right, deg_45 [or deg_90])``
- Right stick (continuous)    -> ``(hold_torso, continuous, waist_*_deg)``
    * ``ry`` => signed ``waist_pitch_deg`` -- forward push (ry > 0)
      gives positive pitch (forward lean), backward push (ry < 0)
      gives negative pitch (backward lean). Bidirectional in v7.4;
      pre-v7.4 the negative side was clamped to 0.
    * ``rx``     => negative ``waist_yaw_deg`` for stick-right (twist)
    * Yaw-priority cone: when |rx| dominates |ry| past
      ``pitch_dominance_ratio``, the pitch axis is suppressed to 0
      so a near-pure twist doesn't accidentally lean the body. The
      symmetric "lean while only slightly twisting" case still
      works.
- Left stick (ARM_MANIPULATION only, v7.4)
                              -> ``(hold_torso, continuous, waist_*_deg)``
    * ``lx`` => signed ``waist_roll_deg`` (lateral lean). Signs
      match ``make_waist_pose_frame``: ``lx > 0`` => positive roll
      (lean to operator's right). Roll is hard-disabled in
      LOCOMOTION because L-stick X-axis owns side-step there.
    * ``ly`` => signed ``hip_height_m`` (squat / stand). Forward push
      (ly < 0 in Quest3 convention) pushes the hip DOWN (squat);
      backward push (ly > 0) raises the hip up. Asymmetric caps:
      down reach is larger than up reach because the kplanner's
      training distribution covers crouch-down ~9cm but only
      stand-up ~4cm before going OOD. The wire carries an absolute
      ``hip_height_m`` (default + offset) so the kplanner can pin
      the model's channel-3 target without tracking incremental
      deltas.
    * Roll-priority cone: when |lx| dominates |ly| past
      ``height_dominance_ratio``, the height axis is suppressed (no
      ``hip_height_m`` override emitted) so a near-pure roll
      doesn't accidentally squat. ``hold_height_threshold_m``
      hysteresis prevents tiny ly noise from spamming the wire.
- ``Y`` held                  -> ``(crouch, medium)`` (only when enabled)
- All sticks neutral          -> ``(idle, default)`` (or ``hold_torso`` at 0/0/0
                                  with no hip_height override while continuous
                                  mode is engaged)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from gear_sonic.utils.teleop.vr.button_state_machine import ButtonEvents


class StreamMode(Enum):
    """X2 manager operating modes for Phase 0.

    Distinct from :class:`gear_sonic.utils.teleop.common.StreamMode`
    (which is G1-shaped). See module docstring for details.
    """

    OFF = 0
    LOCOMOTION = 1
    ARM_MANIPULATION = 2


@dataclass(frozen=True)
class LocomotionCmd:
    """High-level command tuple: ``intent`` + ``magnitude``.

    Mirrors :class:`gear_sonic.utils.planner.state_machine.LocomotionCommand`
    but without the ``source`` field — the manager fills that in when
    serializing to JSON for the planner.

    The ``waist_*_deg`` fields carry continuous waist targets when
    ``intent == "hold_torso"`` (paired with ``magnitude == "continuous"``);
    they are ignored for every other intent so existing call sites remain
    valid. Defaults of 0.0 mean "neutral hold" -- a stick-released
    operator naturally produces a hold at the rest pose, which the
    planner blends back to ``idle_stand`` once a non-hold command lands.

    The ``stick_*`` fields carry continuous locomotion stick deflections
    in ``[-1, 1]`` when ``intent == "locomotion"`` paired with
    ``magnitude == "continuous"``. Replaces the legacy bucketed L/R-stick
    intents (``fwd_step / back_step / side_* / turn_*``) for the kplanner
    so the operator's thumb gets analog control with fine resolution
    near zero. Discrete bin intents ignore these fields, so the heuristic
    planner's primitives-based path remains valid.
    """

    intent: str
    magnitude: str
    waist_pitch_deg: float = 0.0
    waist_roll_deg: float = 0.0
    waist_yaw_deg: float = 0.0
    stick_fwd: float = 0.0
    stick_side: float = 0.0
    stick_yaw: float = 0.0
    # One-shot speed-setpoint adjustment (m/s), from X/Y face buttons in
    # LOCOMOTION mode: X = -0.1, Y = +0.1. Consumed by the kplanner's
    # runtime forward-speed setpoint (0.2..1.0, default 0.3). 0.0 = no
    # adjustment this command.
    speed_delta: float = 0.0
    # Continuous hip-height target in metres. Populated by the
    # ARM_MANIPULATION L-stick Y path (squat / stand) and read by the
    # kplanner's ``intent_to_velocity`` to override channel-3 of the
    # 4-D velocity intent when ``intent == "hold_torso"``. ``None``
    # means "no height delta requested" (kplanner uses its default
    # hip height); a finite value is the absolute target. The decoder
    # is asymmetric -- forward stick (squat) reaches further than
    # backward stick (rise) because the kplanner's training
    # distribution covers down ~9cm but only up ~4cm before going
    # OOD. See ``IntentDecoder.__init__`` for the per-direction caps.
    hip_height_m: float | None = None


@dataclass(frozen=True)
class ModeTransition:
    """Returned by :meth:`IntentDecoder.update_mode` when the mode flips."""

    previous: StreamMode
    current: StreamMode


class IntentDecoder:
    """Stateful Quest-3-button-and-stick to LocomotionCmd mapper.

    Args:
        stick_deadzone: Per-axis deflection threshold below which the
            stick is treated as neutral. Higher than the planner-side
            ``JOYSTICK_DEADZONE`` because operators rest their thumbs
            on the sticks (which would otherwise emit jittery
            ``fwd_step`` commands).
        repeat_interval_s: If > 0, re-emit the most recent command at
            this cadence even when the input has not changed; useful
            because the planner has a short queue and held thumbsticks
            will otherwise stall once the bin finishes. ``0.0``
            disables the repeat (commands fire only on change).
    """

    def __init__(
        self,
        stick_deadzone: float = 0.30,
        repeat_interval_s: float = 0.0,
        chord_debounce_s: float = 0.5,
        enable_crouch: bool = False,
        enable_lean_fwd: bool = False,
        enable_torso: bool = False,
        enable_continuous_torso: bool = False,
        enable_continuous_locomotion: bool = False,
        continuous_stick_threshold: float = 0.02,
        continuous_yaw_max: float = 0.5,
        turn_threshold: float = 0.75,
        lean_medium_threshold: float = 0.55,
        lean_large_threshold: float = 0.80,
        # Per-axis maximum deflection deg (clamps; defaults match the
        # planner's _WAIST_RAMP_CAP_DEG safety caps so the operator
        # cannot drive STATIC_HOLD past a trackable angle).
        max_waist_pitch_deg: float = 20.0,
        max_waist_roll_deg: float = 10.0,
        max_waist_yaw_deg: float = 40.0,
        # Continuous hip-height (squat / stand) caps. These are
        # OFFSETS in metres around ``default_hip_height_m`` (which
        # mirrors the kplanner's ``_HIP_HEIGHT_M = 0.687``). Defaults
        # are asymmetric to match the kplanner training distribution:
        # crouch-down survives ~9cm before going OOD, stand-up only
        # ~4cm. ``max_height_down_m`` is applied when ly < 0 (forward
        # push -> hip down); ``max_height_up_m`` is applied when
        # ly > 0 (backward push -> hip up).
        default_hip_height_m: float = 0.687,
        max_height_down_m: float = 0.09,
        max_height_up_m: float = 0.04,
        # Yaw-priority cone (R-stick): pitch is suppressed when
        # ``|ry| < pitch_dominance_ratio * |rx|`` (i.e. yaw dominates).
        # 0.0 disables the cone (pitch and yaw both fire whenever
        # their axis is past the deadzone). 1.0 means pitch only
        # fires when |ry| >= |rx| (strict dominance). The default
        # 0.4 lets the operator twist while leaning slightly, but
        # blocks accidental lean from a yaw-intent stick wobble.
        pitch_dominance_ratio: float = 0.4,
        # Roll-priority cone (L-stick, ARM_MANIPULATION only):
        # symmetric to ``pitch_dominance_ratio``. height is
        # suppressed when ``|ly| < height_dominance_ratio * |lx|``
        # (roll dominates). Same range / semantics; 0.4 default.
        height_dominance_ratio: float = 0.4,
        # ARM_MANIPULATION L-stick gate. When True (default), the
        # left stick decodes as roll (lx) + height (ly) in
        # ARM_MANIPULATION mode only. When False, the left stick is
        # idle in ARM_MAN exactly as in v7.3 -- only R-stick
        # lean / twist is active. LOCOMOTION mode is unaffected
        # by this flag (L-stick still owns step / side-step / walk).
        enable_arm_man_lstick: bool = True,
        # Hysteresis for continuous hold_torso emission. The decoder
        # only re-emits when ANY axis target moves by at least this
        # many degrees from the last sent target (or repeat_interval_s
        # elapses). Keeps the ZMQ topic from being spammed at every tick
        # by inevitable stick noise without losing perceived smoothness:
        # the planner-side _HoldTracker slews between coarse waypoints.
        hold_target_threshold_deg: float = 0.5,
        # Same idea for ``hip_height_m`` (metres). The decoder treats
        # two ``hold_torso`` commands as identical when their
        # hip-height targets are within this many metres on the
        # current emit; below that threshold thumbstick noise on ly
        # is suppressed at the wire layer. The kplanner's neural
        # model has its own internal smoothing so coarse waypoints
        # are fine.
        hold_height_threshold_m: float = 0.005,
    ) -> None:
        if stick_deadzone <= 0 or stick_deadzone >= 1:
            raise ValueError(f"stick_deadzone must be in (0, 1); got {stick_deadzone}")
        if repeat_interval_s < 0:
            raise ValueError(
                f"repeat_interval_s must be >= 0; got {repeat_interval_s}"
            )
        if chord_debounce_s < 0:
            raise ValueError(
                f"chord_debounce_s must be >= 0; got {chord_debounce_s}"
            )
        if not (stick_deadzone < turn_threshold < 1.0):
            raise ValueError(
                f"turn_threshold must satisfy stick_deadzone < t < 1.0; "
                f"got deadzone={stick_deadzone}, t={turn_threshold}"
            )
        if not (
            stick_deadzone < lean_medium_threshold < lean_large_threshold < 1.0
        ):
            raise ValueError(
                f"lean thresholds must satisfy "
                f"stick_deadzone < medium < large < 1.0; got "
                f"deadzone={stick_deadzone}, medium={lean_medium_threshold}, "
                f"large={lean_large_threshold}"
            )

        self._stick_deadzone = float(stick_deadzone)
        self._repeat_interval_s = float(repeat_interval_s)
        self._chord_debounce_s = float(chord_debounce_s)
        self._turn_threshold = float(turn_threshold)
        self._lean_medium_threshold = float(lean_medium_threshold)
        self._lean_large_threshold = float(lean_large_threshold)
        # Crouch is disabled by default: the X2 heuristic planner's
        # crouch primitive currently destabilizes the controller (the
        # SONIC tracking policy tips over partway through). Until the
        # planner-side fix lands, we drop Y-as-crouch on the floor and
        # log it as ignored. Pass enable_crouch=True to re-enable for
        # offline planner experiments.
        self._enable_crouch = bool(enable_crouch)
        # ``lean_fwd_*`` and ``torso_*_30deg`` are *replay* primitives:
        # the curated bins lean / twist into the pose and immediately
        # blend back to standing instead of holding. For interactive
        # teleop this looks like "the body flicks then snaps back",
        # which confuses operators (you push the stick and nothing
        # appears to stick). We disable both by default; the right-
        # stick X-axis hard-deflection still emits ``turn_*`` (which
        # the operator finds intuitive), and graded lean / soft torso
        # fall through to ``idle`` instead. Re-enable per primitive
        # once the planner can hold the static pose.
        #
        # ``enable_continuous_torso`` (default ON) supersedes the soft-
        # band discrete bins with a continuous hold_torso target. The
        # right-stick X-axis hard turn_* path is unchanged; pitch (ry)
        # / yaw (rx) / roll (A+rx) all map to a single hold_torso
        # command that the planner runs in STATIC_HOLD. Set False to
        # restore the legacy discrete-bin behavior.
        self._enable_lean_fwd = bool(enable_lean_fwd)
        self._enable_torso = bool(enable_torso)
        self._enable_continuous_torso = bool(enable_continuous_torso)
        # ``enable_continuous_locomotion`` (default OFF) replaces the
        # bucketed L-stick / R-stick locomotion intents (fwd_step,
        # back_step, side_*, turn_*) with a single ``locomotion /
        # continuous`` command that carries the raw stick deflections in
        # ``stick_fwd / stick_side / stick_yaw``. The kplanner shapes
        # those into a velocity vector so the operator's thumb gets
        # analog control; the heuristic planner cannot consume this
        # intent and treats it as idle. Set True only when the downstream
        # planner is the kplanner (``run_x2_quest3_planner_stack.sh``
        # passes the flag automatically in ``--planner kplanner`` mode).
        self._enable_continuous_locomotion = bool(enable_continuous_locomotion)
        if continuous_stick_threshold < 0 or continuous_stick_threshold >= 1:
            raise ValueError(
                "continuous_stick_threshold must be in [0, 1); "
                f"got {continuous_stick_threshold}"
            )
        # Per-stick-axis change threshold (post-deadzone) below which two
        # ``locomotion / continuous`` commands are considered identical
        # by ``_is_significant_change``. Mirrors the
        # ``hold_target_threshold_deg`` knob for ``hold_torso``: prevents
        # ZMQ spam from thumbstick noise without losing perceived
        # smoothness (kplanner's intent dispatcher slews internally).
        self._continuous_stick_threshold = float(continuous_stick_threshold)
        # Maximum R-stick X amplitude forwarded as ``stick_yaw`` to the
        # planner. Lives on the *teleop* side (this class) rather than
        # in the planner because turn aggressiveness is fundamentally an
        # operator-feel concern: the kplanner's yaw-rate ceiling
        # (``_CONTINUOUS_TURN_MAX_RAD_S``) sets the *physical* maximum
        # the robot can be asked to track; this knob caps how much of
        # that ceiling a full R-stick deflection actually reaches. With
        # the default 0.5, slamming the stick to the rail commands half
        # the planner's yaw-rate ceiling, which empirically gives a
        # turn rate the policy can track without overshoot. Operators
        # wanting sharper button-style pivots can leave the bucketed
        # path (which is unaffected by this clamp) or raise this to
        # 1.0. The clamp is symmetric -- the SAME ratio applies to
        # left and right turns; per-side asymmetry compensation lives
        # in the planner's runtime turn scales.
        if not (0.0 < continuous_yaw_max <= 1.0):
            raise ValueError(
                "continuous_yaw_max must be in (0, 1]; "
                f"got {continuous_yaw_max}"
            )
        self._continuous_yaw_max = float(continuous_yaw_max)
        if max_waist_pitch_deg < 0 or max_waist_roll_deg < 0 or max_waist_yaw_deg < 0:
            raise ValueError(
                "max_waist_*_deg must be non-negative; got "
                f"({max_waist_pitch_deg}, {max_waist_roll_deg}, {max_waist_yaw_deg})"
            )
        if hold_target_threshold_deg < 0:
            raise ValueError(
                f"hold_target_threshold_deg must be >= 0; "
                f"got {hold_target_threshold_deg}"
            )
        if default_hip_height_m <= 0:
            raise ValueError(
                f"default_hip_height_m must be > 0; got {default_hip_height_m}"
            )
        if max_height_down_m < 0 or max_height_up_m < 0:
            raise ValueError(
                "max_height_{down,up}_m must be non-negative; got "
                f"({max_height_down_m}, {max_height_up_m})"
            )
        if pitch_dominance_ratio < 0 or pitch_dominance_ratio > 1:
            raise ValueError(
                "pitch_dominance_ratio must be in [0, 1]; "
                f"got {pitch_dominance_ratio}"
            )
        if height_dominance_ratio < 0 or height_dominance_ratio > 1:
            raise ValueError(
                "height_dominance_ratio must be in [0, 1]; "
                f"got {height_dominance_ratio}"
            )
        if hold_height_threshold_m < 0:
            raise ValueError(
                f"hold_height_threshold_m must be >= 0; "
                f"got {hold_height_threshold_m}"
            )
        self._max_waist_pitch_deg = float(max_waist_pitch_deg)
        self._max_waist_roll_deg = float(max_waist_roll_deg)
        self._max_waist_yaw_deg = float(max_waist_yaw_deg)
        self._default_hip_height_m = float(default_hip_height_m)
        self._max_height_down_m = float(max_height_down_m)
        self._max_height_up_m = float(max_height_up_m)
        self._pitch_dominance_ratio = float(pitch_dominance_ratio)
        self._height_dominance_ratio = float(height_dominance_ratio)
        self._enable_arm_man_lstick = bool(enable_arm_man_lstick)
        self._hold_target_threshold_deg = float(hold_target_threshold_deg)
        self._hold_height_threshold_m = float(hold_height_threshold_m)

        self._mode = StreamMode.OFF
        self._last_emitted: Optional[LocomotionCmd] = None
        # Accumulated X/Y speed nudges awaiting attachment to the next
        # emitted locomotion command (see _apply_button_transitions).
        self._pending_speed_delta: float = 0.0
        self._last_emit_t: float = -1.0
        # Wall-clock deadline (monotonic) before which decode_locomotion
        # treats Y-as-crouch and all stick deflections as neutral. Armed
        # on any A+B+X+Y chord transition because the chord physically
        # requires Y to be held, and Y mapped naively to crouch was
        # tipping the robot over the instant the operator entered
        # LOCOMOTION mode.
        self._chord_quiet_until: float = -1.0

    # -- mode -----------------------------------------------------------------

    @property
    def mode(self) -> StreamMode:
        return self._mode

    def update_mode(
        self,
        ev: ButtonEvents,
        now: Optional[float] = None,
    ) -> Optional[ModeTransition]:
        """Apply the button-driven mode-toggle transitions.

        Returns the :class:`ModeTransition` when a transition fired this
        tick, or ``None`` when the mode is unchanged.

        ``now`` is the caller's monotonic clock; when supplied and the
        transition is driven by the A+B+X+Y chord, the decoder arms a
        short quiet window that suppresses Y-as-crouch and stick inputs
        so the operator's chord release does not immediately steer the
        planner.
        """
        prev = self._mode
        new = self._mode

        # X/Y single presses in LOCOMOTION = forward-speed setpoint nudges
        # (X -0.1, Y +0.1). Latched here (rising edges) and attached to the
        # next emitted locomotion command; the kplanner owns the actual
        # setpoint state/clamping. Chords (abxy/xy/...) below still win --
        # their handlers run after and a chord press also sets x/y flags,
        # so skip the nudge on any chord tick.
        if self._mode == StreamMode.LOCOMOTION and not (
            ev.abxy_pressed or ev.xy_pressed or ev.ax_pressed or ev.by_pressed
        ):
            if ev.x_pressed:
                self._pending_speed_delta -= 0.1
            if ev.y_pressed:
                self._pending_speed_delta += 0.1

        if ev.abxy_pressed:
            new = (
                StreamMode.LOCOMOTION
                if self._mode == StreamMode.OFF
                else StreamMode.OFF
            )
        elif ev.b_pressed and self._mode != StreamMode.OFF:
            # B-single only matters once we're already in an active mode.
            # In OFF mode, B is a no-op (avoids accidental engages).
            new = (
                StreamMode.ARM_MANIPULATION
                if self._mode == StreamMode.LOCOMOTION
                else StreamMode.LOCOMOTION
            )

        if new == prev:
            return None

        self._mode = new
        # On a mode change, drop the last-emitted memory so the very
        # first tick after the flip re-emits whatever the new mode
        # implies (e.g. an idle command when leaving LOCOMOTION, or a
        # held thumbstick re-firing on entering it).
        self._last_emitted = None

        # The A+B+X+Y chord physically holds Y down; without a debounce
        # the operator's chord-release fires a crouch the instant they
        # land in LOCOMOTION mode and the robot tips over. Arm a quiet
        # window so the next ~chord_debounce_s of decode_locomotion
        # sees Y as released and sticks as neutral. We also arm it on
        # OFF transitions to avoid the symmetric "chord into OFF then
        # straight back into LOCOMOTION" footgun.
        if ev.abxy_pressed and now is not None and self._chord_debounce_s > 0:
            self._chord_quiet_until = now + self._chord_debounce_s

        return ModeTransition(previous=prev, current=new)

    # -- locomotion -----------------------------------------------------------

    def decode_locomotion(
        self,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        y_held: bool,
        now: float,
        a_held: bool = False,
        x_held: bool = False,
    ) -> Optional[LocomotionCmd]:
        """Map this tick's stick + button state to a ``LocomotionCmd``.

        Returns ``None`` when the manager should NOT publish a command
        this tick (either because we're not in LOCOMOTION mode, or
        because the desired command is unchanged from the previous
        emit and the repeat interval has not elapsed).

        Modifier semantics (LOCOMOTION mode only):

        - ``a_held`` + forward/back stick → continuous walk
          (``walk / forward`` resp. ``walk / backward``) instead of
          a single-stride ``fwd_step`` / ``back_step``. Default
          (``a_held=False``) preserves the legacy single-step UX.
        - ``x_held`` + right-stick X → 90° turn
          (``turn_left / deg_90`` resp. ``turn_right / deg_90``)
          instead of the default 45° step.
        """
        if self._mode == StreamMode.OFF:
            return None

        if now < self._chord_quiet_until:
            # Operator is still releasing the A+B+X+Y chord that put us
            # here. Force-quiet inputs so we don't crouch on chord-Y or
            # walk on a thumb that grazed the stick during the chord.
            cmd = LocomotionCmd("idle", "default")
        else:
            cmd = self._cmd_for_inputs(
                lx=lx, ly=ly, rx=rx, ry=ry,
                y_held=y_held, a_held=a_held, x_held=x_held,
            )

        # In ARM_MANIPULATION the operator is doing VR-IK on the arms;
        # we MUST NOT pass through commands that move the robot's base
        # (walk / fwd_step / back_step / side_left / side_right /
        # turn_*) because the arm-IK targets are computed in torso
        # frame and a base translation slides the IK reference out
        # from under the operator's hands. Continuous waist hold
        # (hold_torso) is safe -- it only changes waist DOFs (pitch /
        # roll / yaw) and the IK actually rides along (torso lean
        # extends the arm reach envelope, which is the whole point
        # of v7.2 unlocking R-stick in ARM_MAN). Idle commands also
        # filtered so we don't wash away the planner's STATIC_HOLD
        # target with stale "go to idle_stand" pulses on every chord
        # quiet window.
        if self._mode == StreamMode.ARM_MANIPULATION and cmd.intent != "hold_torso":
            return None

        return self._maybe_emit(cmd, now)

    def _cmd_for_inputs(
        self,
        *,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        y_held: bool,
        a_held: bool = False,
        x_held: bool = False,
    ) -> LocomotionCmd:
        """Pure stick + button -> LocomotionCmd mapping (no state)."""
        if y_held and self._enable_crouch:
            return LocomotionCmd("crouch", "medium")

        # ---- continuous-locomotion path -------------------------------------
        # When enabled (kplanner mode), L/R-stick deflections are forwarded
        # raw (post-deadzone rescale) as ``locomotion / continuous`` so the
        # downstream planner gets analog control. The discrete fwd_step /
        # back_step / side_* / turn_* path below is skipped entirely.
        # ``hold_torso`` (right-stick lean) and ARM_MAN gating still apply
        # on the calling site; this branch only owns L-stick + R-stick X
        # axis (rx), leaving the ry-driven hold_torso to the legacy path.
        #
        # ARM_MANIPULATION carve-out (v7.4): when the operator is in
        # ARM_MAN, the L-stick decodes as roll (lx) + continuous hip
        # height (ly) inside ``_continuous_hold_cmd``, NOT as
        # ``locomotion / continuous`` (which would just be filtered
        # out by the ARM_MAN gate at the call site, silently dropping
        # the operator's L-stick movements). So skip both the
        # continuous-locomotion path AND the discrete L-stick path
        # when in ARM_MAN; fall straight through to the right-stick
        # / hold_torso branch which now owns both sticks.
        in_arm_man = self._mode == StreamMode.ARM_MANIPULATION
        if self._enable_continuous_locomotion and not in_arm_man:
            stick_fwd, stick_side, stick_yaw = self._continuous_stick_targets(
                lx=lx, ly=ly, rx=rx,
            )
            if (
                abs(stick_fwd) > 0.0
                or abs(stick_side) > 0.0
                or abs(stick_yaw) > 0.0
                or self._pending_speed_delta != 0.0
            ):
                # Attach (and clear) any accumulated X/Y speed nudge. The
                # dedup filter force-emits commands carrying a nonzero
                # speed_delta so a nudge is never lost to stick-noise
                # hysteresis.
                delta, self._pending_speed_delta = self._pending_speed_delta, 0.0
                return LocomotionCmd(
                    intent="locomotion",
                    magnitude="continuous",
                    stick_fwd=float(stick_fwd),
                    stick_side=float(stick_side),
                    stick_yaw=float(stick_yaw),
                    speed_delta=float(delta),
                )
            # All three locomotion sticks released. Fall through to the
            # ry-driven hold_torso path below so the right thumb can
            # still lean / twist while neither L-stick nor R-stick X
            # is deflected.

        # Left stick: cardinal direction wins (whichever axis dominates).
        # We bias toward forward/backward when the deflections are
        # roughly equal so a slight diagonal still produces forward
        # walking instead of jittering to side-step on a small lateral
        # component.
        #
        # ARM_MAN carve-out (v7.4): same rationale as the continuous-
        # locomotion branch above -- L-stick is owned by the
        # hold_torso / waist-overlay path in ARM_MAN, so don't return
        # discrete fwd_step / side_* / walk commands that would be
        # filtered out at the call site anyway.
        l_active = (
            not in_arm_man
            and (abs(ly) >= self._stick_deadzone
                 or abs(lx) >= self._stick_deadzone)
        )
        if l_active:
            if abs(ly) >= abs(lx):
                # A held promotes single-stride to continuous walk.
                # ``walk / forward`` resolves to ``fwd_walk_standard``
                # in the planner; ``walk / backward`` resolves to
                # ``back_walk_standard``. Both are loop primitives, so
                # holding the stick keeps the robot walking until the
                # operator releases (decoder then emits idle and the
                # planner blends back to standing).
                if a_held:
                    if ly > 0:
                        return LocomotionCmd("walk", "forward")
                    return LocomotionCmd("walk", "backward")
                if ly > 0:
                    return LocomotionCmd("fwd_step", "default")
                return LocomotionCmd("back_step", "default")
            if lx > 0:
                return LocomotionCmd("side_right", "default")
            return LocomotionCmd("side_left", "default")

        # Right stick: continuous-hold path wins when enabled, with
        # hard-turn deflection still pre-empting (operator wants to
        # PIVOT, not lean). Otherwise fall back to the legacy
        # dominant-axis decoder so existing scripted regression tests
        # (test_intent_decoder.py) keep their semantics.
        if self._enable_continuous_torso:
            rx_active = abs(rx) >= self._stick_deadzone
            if rx_active and abs(rx) >= self._turn_threshold:
                magnitude = "deg_90" if x_held else "deg_45"
                if rx > 0:
                    return LocomotionCmd("turn_right", magnitude)
                return LocomotionCmd("turn_left", magnitude)
            # v7.4: lx/ly forwarded so ARM_MANIPULATION mode can decode
            # roll (lx) + continuous hip height (ly) from the L-stick.
            # In LOCOMOTION mode ``_continuous_hold_cmd`` ignores
            # lx/ly because the L-stick already drove the discrete /
            # continuous-locomotion path above; here we're past that
            # branch (the L-stick was inactive this tick) so passing
            # the residual deflections through is harmless.
            #
            # ``a_held`` is intentionally not forwarded: A and the
            # R-stick share the operator's right thumb (v7.2).
            return self._continuous_hold_cmd(
                rx=rx, ry=ry, lx=lx, ly=ly,
            )

        # ----- Legacy discrete-bin path -----------------------------------
        # Y axis (forward) -> graded lean_fwd_{small,medium,large}
        #   (gated on ``enable_lean_fwd``).
        # X axis: hard -> turn_*; soft -> torso_* (gated on enable_torso).
        # Dominant-axis precedence preserves the legacy "ry > rx wins"
        # guard so a forward + slight-right push doesn't pivot.
        ry_active = abs(ry) >= self._stick_deadzone
        rx_active = abs(rx) >= self._stick_deadzone
        if ry_active and abs(ry) >= abs(rx):
            if ry > 0 and self._enable_lean_fwd:
                if ry >= self._lean_large_threshold:
                    return LocomotionCmd("lean_fwd", "large")
                if ry >= self._lean_medium_threshold:
                    return LocomotionCmd("lean_fwd", "medium")
                return LocomotionCmd("lean_fwd", "small")
            return LocomotionCmd("idle", "default")
        if rx_active:
            if abs(rx) >= self._turn_threshold:
                magnitude = "deg_90" if x_held else "deg_45"
                if rx > 0:
                    return LocomotionCmd("turn_right", magnitude)
                return LocomotionCmd("turn_left", magnitude)
            if self._enable_torso:
                if rx > 0:
                    return LocomotionCmd("torso_right", "deg_30")
                return LocomotionCmd("torso_left", "deg_30")
            return LocomotionCmd("idle", "default")

        return LocomotionCmd("idle", "default")

    def _continuous_stick_targets(
        self,
        *,
        lx: float,
        ly: float,
        rx: float,
    ) -> tuple[float, float, float]:
        """Return (stick_fwd, stick_side, stick_yaw) in ``[-1, 1]``.

        Each axis is deadzone-clamped to 0 below ``stick_deadzone`` and
        rescaled to ``[-1, 1]`` outside it (sign preserved). The kplanner
        squares the value to give the operator's thumb fine resolution
        near zero with full speed at full deflection.

        Notes on signs vs the legacy bucketed path:

        - ``ly > 0`` -> forward intent -> ``stick_fwd > 0``.
        - ``lx > 0`` -> right step -> ``stick_side > 0``.
        - ``rx > 0`` -> turn-right -> ``stick_yaw > 0``.

        These match the bucketed convention so the kplanner's
        ``_BASE_VELOCITY`` table and the new ``_resolve_locomotion``
        branch agree on direction sign.

        The yaw axis is additionally clamped to ``±continuous_yaw_max``
        before publishing. This caps the operator-commanded fraction of
        the planner's yaw-rate ceiling and lives here (teleop layer)
        rather than in the planner because turn aggressiveness is an
        operator-feel concern, not a planner capability. Forward and
        side axes are unaffected -- those don't suffer from the same
        "brief stick burst commits to a long turn" failure mode (the
        operator naturally holds the L-stick to drive).
        """
        dz = self._stick_deadzone
        return (
            self._rescaled_axis(ly, dz),
            self._rescaled_axis(lx, dz),
            self._rescaled_axis(rx, dz) * self._continuous_yaw_max,
        )

    @staticmethod
    def _rescaled_axis(value: float, deadzone: float) -> float:
        """Map raw stick deflection in ``[-1, 1]`` to deadzoned ``[-1, 1]``.

        Below ``deadzone`` -> 0. Above ``deadzone`` -> ``(|v| - dz) / (1 - dz)``
        clamped to 1, sign preserved. Pure helper for the continuous-
        locomotion path; kept separate from ``_scaled_axis`` (which is
        the waist-hold variant that also folds in a ``max_deg`` scale).
        """
        sign = 1.0 if value >= 0 else -1.0
        mag = abs(float(value))
        if mag <= deadzone:
            return 0.0
        denom = max(1.0 - deadzone, 1e-6)
        return sign * min(1.0, (mag - deadzone) / denom)

    @staticmethod
    def _scaled_axis(value: float, deadzone: float, max_deg: float) -> float:
        """Map a raw stick deflection in [-1, 1] to a clamped angle in degrees.

        Below |value| <= deadzone the result is 0 (so a thumb at rest
        produces a neutral target). Outside the deadzone (deadzone, 1)
        is rescaled to (0, 1) so that just past the deadzone yields a
        small angle and full deflection yields ``max_deg``. Sign is
        preserved.
        """
        sign = 1.0 if value >= 0 else -1.0
        mag = abs(float(value))
        if mag <= deadzone:
            return 0.0
        denom = max(1.0 - deadzone, 1e-6)
        norm = min(1.0, (mag - deadzone) / denom)
        return sign * norm * float(max_deg)

    def continuous_waist_target(
        self,
        rx: float,
        ry: float,
        a_held: bool = False,
        *,
        lx: float = 0.0,
        ly: float = 0.0,
    ) -> tuple[float, float, float, float | None]:
        """Compute the (pitch, roll, yaw, hip_height_m) continuous target.

        Public companion to :meth:`_continuous_hold_cmd` for callers
        (e.g. the Quest 3 manager) that need the *target value* without
        the dedup / throttle gating done by ``decode_locomotion``. Used
        to latch the current operator-driven hold pose at the moment
        the manager flips ``LOCOMOTION -> ARM_MANIPULATION``.

        Always returns clamped, deadzone-aware angles regardless of the
        decoder's mode or the ``enable_continuous_torso`` flag. The
        4-tuple's ``hip_height_m`` is ``None`` when the L-stick is
        within the height deadzone / the ARM_MAN gate is off (caller
        should treat this as "no override; use planner default").

        The ``a_held`` parameter is accepted for backward compatibility
        with v7.1 callers but is now ignored: see :meth:`_continuous_hold_cmd`
        for the ergonomic rationale (operator's right thumb cannot reach
        A while driving the R-stick on the same controller).

        ``lx`` / ``ly`` default to 0.0 so legacy v7.3 call-sites (which
        only sampled the right stick) keep their semantics: roll stays
        0.0 and ``hip_height_m`` stays ``None``.
        """
        del a_held  # v7.2: accepted but ignored
        cmd = self._continuous_hold_cmd(rx=rx, ry=ry, lx=lx, ly=ly)
        return (
            cmd.waist_pitch_deg,
            cmd.waist_roll_deg,
            cmd.waist_yaw_deg,
            cmd.hip_height_m,
        )

    def _continuous_hold_cmd(
        self,
        *,
        rx: float,
        ry: float,
        lx: float = 0.0,
        ly: float = 0.0,
    ) -> LocomotionCmd:
        """Build a continuous-hold LocomotionCmd from the stick state.

        Stick conventions (v7.4):

        - ``ry``                  -> signed pitch. ``ry > 0`` (forward
          push) gives positive pitch (forward lean); ``ry < 0`` gives
          negative pitch (backward lean). Bidirectional in v7.4;
          pre-v7.4 the negative side was clamped to 0.
        - ``rx``                  -> twist (yaw). Stick right (``rx > 0``)
          maps to a NEGATIVE ``waist_yaw_deg``, matching
          ``torso_right_*deg`` (negative peak_deg).
        - ``lx``                  -> signed roll (ARM_MANIPULATION only,
          gated on ``enable_arm_man_lstick``). ``lx > 0`` gives
          positive roll (lean to operator's right). LOCOMOTION leaves
          roll = 0 because the L-stick X-axis owns side-step there.
        - ``ly``                  -> signed continuous hip height
          (ARM_MANIPULATION only). Quest3 reports ``ly < 0`` for stick
          pushed forward (away from the operator) -> hip DOWN (squat).
          Asymmetric caps -- down reach is larger than up reach
          because the kplanner's training distribution covers
          crouch-down ~9cm but only stand-up ~4cm.

        Yaw-priority cone (R-stick): when the operator is making a
        predominantly-yaw stick movement (|rx| dominates |ry| past
        ``pitch_dominance_ratio``), the pitch axis is suppressed to 0
        so a near-pure twist doesn't accidentally lean the body. The
        symmetric "lean while only slightly twisting" case still
        works.

        Roll-priority cone (L-stick): when |lx| dominates |ly| past
        ``height_dominance_ratio``, the height axis is suppressed (no
        ``hip_height_m`` override emitted) so a near-pure roll
        doesn't accidentally squat.

        Released stick (in deadzone) yields a neutral hold target;
        the planner's STATIC_HOLD / waist-overlay slews back to
        neutral via the same ``_HoldTracker`` path that any other
        change would use.
        """
        yaw_deg = -self._scaled_axis(
            rx, self._stick_deadzone, self._max_waist_yaw_deg,
        )

        ry_active = abs(ry) >= self._stick_deadzone
        rx_active = abs(rx) >= self._stick_deadzone
        pitch_dominates = (
            self._pitch_dominance_ratio <= 0.0
            or not rx_active
            or abs(ry) >= self._pitch_dominance_ratio * abs(rx)
        )
        if ry_active and pitch_dominates:
            pitch_deg = self._scaled_axis(
                ry, self._stick_deadzone, self._max_waist_pitch_deg,
            )
        else:
            pitch_deg = 0.0

        roll_deg = 0.0
        hip_height_m: float | None = None
        if (
            self._mode == StreamMode.ARM_MANIPULATION
            and self._enable_arm_man_lstick
        ):
            roll_deg = self._scaled_axis(
                lx, self._stick_deadzone, self._max_waist_roll_deg,
            )

            ly_active = abs(ly) >= self._stick_deadzone
            lx_active = abs(lx) >= self._stick_deadzone
            height_dominates = (
                self._height_dominance_ratio <= 0.0
                or not lx_active
                or abs(ly) >= self._height_dominance_ratio * abs(lx)
            )
            if ly_active and height_dominates:
                # Quest3: ly < 0 = stick pushed forward (away from
                # operator) -> hip DOWN (squat). Asymmetric clamp:
                # down reach is larger than up reach. ``_scaled_axis``
                # is sign-preserving, so we feed it ``|ly|`` and a
                # direction-specific cap, then negate for the
                # forward-push branch so the absolute target sits
                # below the default.
                if ly < 0:
                    offset = -self._scaled_axis(
                        -ly, self._stick_deadzone, self._max_height_down_m,
                    )
                else:
                    offset = self._scaled_axis(
                        ly, self._stick_deadzone, self._max_height_up_m,
                    )
                hip_height_m = float(self._default_hip_height_m + offset)

        return LocomotionCmd(
            intent="hold_torso",
            magnitude="continuous",
            waist_pitch_deg=float(pitch_deg),
            waist_roll_deg=float(roll_deg),
            waist_yaw_deg=float(yaw_deg),
            hip_height_m=hip_height_m,
        )

    def _maybe_emit(
        self,
        cmd: LocomotionCmd,
        now: float,
    ) -> Optional[LocomotionCmd]:
        """Apply de-duplication + optional repeat-interval gating.

        Behaviour:

        - First tick a command appears: emit.
        - Subsequent ticks with the SAME command: emit only if
          ``repeat_interval_s > 0`` AND the elapsed time since the
          last emit reaches the threshold.
        - First tick a NEW command appears: emit (resets the timer).

        For ``hold_torso`` commands, "SAME" means the waist targets are
        within ``hold_target_threshold_deg`` of the previous emit on
        every axis -- otherwise inevitable thumbstick noise would
        retrigger the planner at every tick. The planner-side
        ``_HoldTracker`` slews between coarse waypoints, so this
        threshold is purely a wire-bandwidth optimisation.
        """
        if self._last_emitted is None or self._is_significant_change(
            cmd, self._last_emitted
        ):
            self._last_emitted = cmd
            self._last_emit_t = now
            return cmd

        if (
            self._repeat_interval_s > 0
            and (now - self._last_emit_t) >= self._repeat_interval_s
        ):
            self._last_emit_t = now
            return cmd

        return None

    def _is_significant_change(
        self,
        new: LocomotionCmd,
        old: LocomotionCmd,
    ) -> bool:
        """True iff ``new`` should be re-emitted relative to ``old``."""
        if new.intent != old.intent or new.magnitude != old.magnitude:
            return True
        if new.intent == "hold_torso":
            thresh = self._hold_target_threshold_deg
            angle_changed = (
                abs(new.waist_pitch_deg - old.waist_pitch_deg) >= thresh
                or abs(new.waist_roll_deg - old.waist_roll_deg) >= thresh
                or abs(new.waist_yaw_deg - old.waist_yaw_deg) >= thresh
            )
            # hip_height_m: None <-> finite is always a significant
            # change (operator engaged / released the squat axis); two
            # finite values count when their delta meets the metres
            # threshold.
            new_h = new.hip_height_m
            old_h = old.hip_height_m
            if (new_h is None) != (old_h is None):
                height_changed = True
            elif new_h is None:
                height_changed = False
            else:
                height_changed = (
                    abs(float(new_h) - float(old_h))
                    >= self._hold_height_threshold_m
                )
            return angle_changed or height_changed
        if new.intent == "locomotion":
            # Mirror the hold_torso hysteresis: only re-emit when the
            # operator has moved a stick by at least the configured
            # threshold (in normalised post-deadzone units). Without
            # this every tick of inevitable thumbstick noise would
            # republish at 50-90 Hz.
            thresh = self._continuous_stick_threshold
            # A speed-setpoint nudge must ALWAYS go out, even when the
            # sticks themselves haven't moved past the hysteresis.
            if new.speed_delta != 0.0:
                return True
            return (
                abs(new.stick_fwd  - old.stick_fwd)  >= thresh
                or abs(new.stick_side - old.stick_side) >= thresh
                or abs(new.stick_yaw  - old.stick_yaw)  >= thresh
            )
        # Discrete commands: any field change counts (intent/magnitude
        # already compared above; equality on the dataclass covers any
        # future fields too).
        return new != old

    # -- introspection (for sidecar logging) ----------------------------------

    def last_emitted(self) -> Optional[LocomotionCmd]:
        """Return the most recent emitted command, or ``None`` if never."""
        return self._last_emitted


__all__ = [
    "StreamMode",
    "LocomotionCmd",
    "ModeTransition",
    "IntentDecoder",
]
