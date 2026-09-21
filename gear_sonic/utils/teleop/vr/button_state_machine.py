"""Edge-triggered button + chord state machine for VR controllers.

Both the G1 manager (upstream G1 Quest manager, not shipped) and the X2 manager
(`quest3_manager_x2.py`) need rising-edge detection on the four face buttons
plus a small set of chord combinations (A+B, X+Y, A+X, B+Y, A+B+X+Y).

This module owns the previous-state bookkeeping so each manager only has
to feed in the current-tick boolean tuple and read the events out.

Behavior is preserved bit-for-bit from the original inline state machines
in `Quest3PlannerStreamer.run_once()` (lines ~115-138) and the manager
loop (lines ~344-407) of the original G1 manager script. Each manager
constructs its own `ButtonStateMachine` instance, so per-button log
messages still appear once per tracking site (preserving the existing
double-log when the streamer and manager loop both observe the same
press; downstream tools rely on the log lines.)

Usage::

    sm = ButtonStateMachine(log_prefix="Manager")
    while True:
        a, b, x, y = reader.get_buttons()
        ev = sm.tick(a, b, x, y)
        if ev.abxy_pressed:
            ...  # start / stop chord
        if ev.ax_pressed:
            ...  # toggle PLANNER <-> PLANNER_VR_3PT
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ButtonEvents:
    """Rising-edge events for one tick.

    Each field is True iff the corresponding button or chord transitioned
    from "not pressed" on the previous tick to "pressed" on this tick.
    """

    # individual face buttons
    a_pressed: bool
    b_pressed: bool
    x_pressed: bool
    y_pressed: bool

    # two-button chords
    ab_pressed: bool
    xy_pressed: bool
    ax_pressed: bool
    by_pressed: bool

    # full chord
    abxy_pressed: bool


class ButtonStateMachine:
    """Edge-triggered detector for the four face buttons and common chords.

    Args:
        log_prefix: Prefix for per-button rising-edge log lines (e.g.
            "[Input] A pressed"). Set ``log_prefix=None`` to silence
            the per-button logs entirely.
    """

    def __init__(self, log_prefix: str | None = "Input") -> None:
        self._log_prefix = log_prefix
        self._prev_a = False
        self._prev_b = False
        self._prev_x = False
        self._prev_y = False
        self._prev_ab = False
        self._prev_xy = False
        self._prev_ax = False
        self._prev_by = False
        self._prev_abxy = False

    def tick(self, a: bool, b: bool, x: bool, y: bool) -> ButtonEvents:
        """Consume one frame of button state and return the rising edges."""
        a_pressed = a and not self._prev_a
        b_pressed = b and not self._prev_b
        x_pressed = x and not self._prev_x
        y_pressed = y and not self._prev_y

        if self._log_prefix is not None:
            # Wall-clock millis on every edge (2026-08-03: unstamped
            # edge lines made press timing unreconstructable during the
            # e-stop debug).
            _ts = time.strftime("[%H:%M:%S") + \
                f".{int(time.time() * 1000) % 1000:03d}]"
            if a_pressed:
                print(f"{_ts} [{self._log_prefix}] A pressed")
            if b_pressed:
                print(f"{_ts} [{self._log_prefix}] B pressed")
            if x_pressed:
                print(f"{_ts} [{self._log_prefix}] X pressed")
            if y_pressed:
                print(f"{_ts} [{self._log_prefix}] Y pressed")

        ab_now = a and b
        xy_now = x and y
        ax_now = a and x
        by_now = b and y
        abxy_now = a and b and x and y

        events = ButtonEvents(
            a_pressed=a_pressed,
            b_pressed=b_pressed,
            x_pressed=x_pressed,
            y_pressed=y_pressed,
            ab_pressed=ab_now and not self._prev_ab,
            xy_pressed=xy_now and not self._prev_xy,
            ax_pressed=ax_now and not self._prev_ax,
            by_pressed=by_now and not self._prev_by,
            abxy_pressed=abxy_now and not self._prev_abxy,
        )

        self._prev_a = a
        self._prev_b = b
        self._prev_x = x
        self._prev_y = y
        self._prev_ab = ab_now
        self._prev_xy = xy_now
        self._prev_ax = ax_now
        self._prev_by = by_now
        self._prev_abxy = abxy_now

        return events


__all__ = [
    "ButtonEvents",
    "ButtonStateMachine",
]
