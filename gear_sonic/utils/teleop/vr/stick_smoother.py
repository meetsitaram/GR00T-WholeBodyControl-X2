"""Per-channel low-pass + slew-rate filter for Quest 3 stick inputs.

Problem this fixes
------------------

The kplanner + SONIC stack was trained on smooth human mocap. Live Quest 3
VR teleop pushes raw thumbstick deflections at 50 Hz; a single thumb flick
is a step input into a model that expects sub-1 m/s^2 acceleration. The
p99 ``d(vel_z)/dt`` from a typical VR session is 8+ m/s^2 (peaking >20
m/s^2 on the worst tick) -- 5-20x what the training distribution carries
on the curated bins. The robot is therefore stable in idle but lurches
when commanded to walk forward.

The existing ``_ColdStartVelocityRamp`` in ``x2_kplanner.py`` (EWMA with
``tau ~= 0.2 s``) only fires on ``idle -> playing`` transitions, so steady-
state continuous-locomotion ticks pass through unfiltered. ``StickFilter``
is the missing piece: per-channel band-limiting + acceleration capping on
the manager side, BEFORE the IntentDecoder sees the sticks.

Signal-shaping contract
-----------------------

Each stick channel (``fwd``, ``side``, ``yaw``) gets:

1. **Slew limit** (rad/s for yaw, dimensionless deflection/s for translation
   sticks). The output cannot change by more than ``slew_max_per_s * dt``
   between ticks. This is the primary acceleration cap.
2. **First-order LPF** (tau in seconds). After slew clamping, the output is
   exponentially smoothed toward the slew-clamped target. ``alpha = 1 -
   exp(-dt / tau)`` is the discrete pole; ``tau=0`` disables LPF
   (pure slew).
3. **Optional asymmetric release tau** (``return_to_zero_tau``). When the
   raw input crosses through zero ("operator released the stick"), the LPF
   tau is replaced with ``return_to_zero_tau`` for that channel. Lets us
   keep the engaged-gait response snappy (small ``tau``) while still
   draining residual deflection slowly on release so the robot doesn't
   commit a half-stride after the operator stops asking for one.

Application order (slew first, then LPF):

  ``x_clamped = clamp(x_raw, y_prev - slew*dt, y_prev + slew*dt)``
  ``y         = y_prev + (x_clamped - y_prev) * alpha``

Slew-then-LPF (rather than LPF-then-slew) is the right composition because
the slew limit is the primary acceleration cap on the operator's COMMAND;
the LPF on top removes residual high-frequency jitter that doesn't violate
the slew cap. The other order treats slew as a backstop on the smoothed
signal, which means a fast step in raw input takes longer to register
(LPF delays it) AND then gets clipped (slew limits it) -- worst of both
worlds.

State + reset semantics
-----------------------

The filter holds per-channel ``y_prev`` (last output) and ``last_dt``. The
manager calls ``step()`` once per tick with the raw sticks + the wall-time
delta since the last call. ``reset()`` clears all internal state and is
called on mode transitions (LOCOMOTION enter/exit, OFF, mode flip into
ARM_MANIPULATION) so the operator's first tick after a mode change is
NOT slewed up from whatever stale value the filter happened to hold.

Defaults
--------

The defaults below are *placeholders* awaiting the offline (tau, slew)
sweep against the captured VR fixture. The expected ranges (informed by
PKL primitive analysis):

  tau in [0.10, 0.40] s -- shorter than the ~0.5 s lag operators notice
                            as "sluggish".
  slew in [2.0, 8.0]    -- one full deflection (1.0) takes 125 ms - 500 ms
                            to reach; matches PKL "operator pushes stick"
                            time constants from inspection of the curated
                            clips.

The actual chosen defaults will land in ``ManagerConfig`` after the sweep
runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class StickFilterConfig:
    """Per-channel tuning knobs for ``StickFilter``.

    All ``tau_*`` values are LPF time constants in seconds. ``tau == 0``
    disables the LPF on that channel (slew-only). Use ``math.inf`` to
    disable slew on a channel (LPF-only); the default of ``inf`` means
    "do nothing" so a default-constructed config is the no-op identity
    filter.
    """

    # Forward / backward (mapped from L-stick ly post-invert)
    tau_lpf_fwd_s: float = 0.0
    slew_max_fwd_per_s: float = math.inf

    # Lateral (mapped from L-stick lx)
    tau_lpf_side_s: float = 0.0
    slew_max_side_per_s: float = math.inf

    # Yaw rate proxy (mapped from R-stick rx)
    tau_lpf_yaw_s: float = 0.0
    slew_max_yaw_per_s: float = math.inf

    # Optional asymmetric release tau. When the RAW input crosses zero
    # (operator letting go of the stick), the channel's tau is replaced
    # with this value for the rest of the release. ``None`` disables and
    # the engaged-tau is used in both directions. Useful for "snappy
    # response, gentle release" feel.
    return_to_zero_tau_fwd_s: Optional[float] = None
    return_to_zero_tau_side_s: Optional[float] = None
    return_to_zero_tau_yaw_s: Optional[float] = None

    # Hard clamp on the time delta the filter will accept. A 50 Hz
    # manager tick is 20 ms; a long pause (e.g., debugger break) would
    # produce a single huge ``dt`` that lets the slew limiter swing the
    # output across its full range in one tick, defeating the cap.
    # Default 50 ms = 2.5 manager ticks worth of catch-up.
    max_dt_s: float = 0.05

    def is_noop(self) -> bool:
        """Return True if every channel is configured as identity.

        Used by the manager to skip filter wiring entirely when no
        smoothing is desired, so the unfiltered code path is exactly
        what it was before the filter landed (zero-perf-cost opt-out).
        """
        no_lpf = (
            self.tau_lpf_fwd_s <= 0.0
            and self.tau_lpf_side_s <= 0.0
            and self.tau_lpf_yaw_s <= 0.0
        )
        no_slew = (
            math.isinf(self.slew_max_fwd_per_s)
            and math.isinf(self.slew_max_side_per_s)
            and math.isinf(self.slew_max_yaw_per_s)
        )
        return no_lpf and no_slew


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class StickFilter:
    """Three-channel slew + LPF cascade for ``stick_fwd / side / yaw``.

    Stateful: holds the previous output per channel. Thread-safety is the
    caller's responsibility (the manager calls ``step`` and ``reset`` from
    the same 50 Hz loop, so no lock is required there).
    """

    def __init__(self, cfg: StickFilterConfig) -> None:
        self._cfg = cfg
        # Per-channel previous output. None until first ``step`` so that
        # the first tick is a pass-through (no spurious slew clamp from
        # an assumed-zero anchor).
        self._y_fwd: Optional[float] = None
        self._y_side: Optional[float] = None
        self._y_yaw: Optional[float] = None
        # Track most recent raw sign per channel for asymmetric tau.
        self._sign_fwd: int = 0
        self._sign_side: int = 0
        self._sign_yaw: int = 0

    @property
    def cfg(self) -> StickFilterConfig:
        return self._cfg

    def reset(self) -> None:
        """Clear all per-channel state. Call on mode transitions so the
        operator's first tick after entering LOCOMOTION is unfiltered.
        """
        self._y_fwd = None
        self._y_side = None
        self._y_yaw = None
        self._sign_fwd = 0
        self._sign_side = 0
        self._sign_yaw = 0

    def step(
        self,
        *,
        stick_fwd: float,
        stick_side: float,
        stick_yaw: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """Advance the filter by ``dt`` seconds and return smoothed sticks.

        Returns ``(fwd, side, yaw)`` in the same units as the input. ``dt``
        is clamped to ``cfg.max_dt_s`` to prevent a long pause from
        producing a single huge slew step.
        """
        dt_eff = max(0.0, min(float(dt), self._cfg.max_dt_s))

        fwd = self._step_channel(
            x=float(stick_fwd),
            y_prev_ref=("_y_fwd",),
            sign_attr="_sign_fwd",
            tau_engaged=self._cfg.tau_lpf_fwd_s,
            tau_release=self._cfg.return_to_zero_tau_fwd_s,
            slew=self._cfg.slew_max_fwd_per_s,
            dt=dt_eff,
        )
        side = self._step_channel(
            x=float(stick_side),
            y_prev_ref=("_y_side",),
            sign_attr="_sign_side",
            tau_engaged=self._cfg.tau_lpf_side_s,
            tau_release=self._cfg.return_to_zero_tau_side_s,
            slew=self._cfg.slew_max_side_per_s,
            dt=dt_eff,
        )
        yaw = self._step_channel(
            x=float(stick_yaw),
            y_prev_ref=("_y_yaw",),
            sign_attr="_sign_yaw",
            tau_engaged=self._cfg.tau_lpf_yaw_s,
            tau_release=self._cfg.return_to_zero_tau_yaw_s,
            slew=self._cfg.slew_max_yaw_per_s,
            dt=dt_eff,
        )
        return fwd, side, yaw

    # -- internals ----------------------------------------------------------

    def _step_channel(
        self,
        *,
        x: float,
        y_prev_ref: tuple[str],
        sign_attr: str,
        tau_engaged: float,
        tau_release: Optional[float],
        slew: float,
        dt: float,
    ) -> float:
        """Run one channel through (slew, LPF). Updates the per-channel
        state on ``self`` and returns the new output.
        """
        attr = y_prev_ref[0]
        y_prev = getattr(self, attr)

        if y_prev is None:
            # First tick: pass through verbatim and seed the state.
            setattr(self, attr, x)
            setattr(self, sign_attr, _sign(x))
            return x

        # Slew clamp against the previous output.
        if math.isfinite(slew):
            max_delta = slew * dt
            x_clamped = _clip(x, y_prev - max_delta, y_prev + max_delta)
        else:
            x_clamped = x

        # Pick LPF tau. "Release mode" is detected by comparing the raw
        # target to the current output: the operator is releasing if the
        # raw stick is closer to zero than the filter output (output is
        # still draining toward zero), or if the raw has crossed the
        # zero line relative to the output. Keying off ``y_prev`` rather
        # than the per-tick raw-sign trace means release_tau remains
        # active for the FULL drain trajectory, not just the first tick
        # where the raw crosses zero. The trailing ``sign_attr`` field
        # is kept for diagnostics / future hysteresis hooks.
        tau = tau_engaged
        if (
            tau_release is not None
            and tau_release > 0.0
            and y_prev != 0.0
        ):
            output_sign = 1.0 if y_prev > 0 else -1.0
            # Releasing toward zero: raw moved past or below the
            # current output in the direction of zero.
            is_releasing = (
                (x * output_sign) < (y_prev * output_sign) - 1e-9
            )
            if is_releasing:
                tau = tau_release

        # LPF. ``tau <= 0`` means "no LPF, output the slew-clamped target
        # verbatim". This lets a config disable LPF on one channel while
        # keeping it on the others.
        if tau > 0.0 and dt > 0.0:
            alpha = 1.0 - math.exp(-dt / tau)
            y = y_prev + (x_clamped - y_prev) * alpha
        else:
            y = x_clamped

        setattr(self, attr, y)
        setattr(self, sign_attr, _sign(x))
        return y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
