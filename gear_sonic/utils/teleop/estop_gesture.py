"""Operator e-stop gesture detector (design: operator, 2026-08-04).

Gesture = RAPID PUMPING of both analog triggers TOGETHER while the A+X
chord is HELD (pad: L2+R2; VR: both index triggers). The stick-pin
alternative chord was REMOVED after a live incident: both-sticks-forward
is a normal driving posture (left stick = walk, right stick no-op) and
the triggers are the pad deadman, so an operator re-gripping mid-walk
collapsed the robot. A+X is the sole chord — it cannot be held while a
thumb is on a stick, so it cannot co-occur with driving.

A "press" is ONE PUMP: a press->release cycle on EACH trigger, paired
within ``pair_window_s`` (0.5 s). Squeezing both together = 1 pump;
alternating L,R,L,R = 1 pump per L+R pair (operators rapid-fire both
ways); hammering a single trigger pairs with nothing and counts ZERO.
The 2026-08-03 collapse happened because each pump counted twice
(L2 + R2 cycles summed), turning 3 gentle pumps into a 6-count damp
order; the min()-per-window fix that followed over-corrected and went
silent for alternating firing (each channel alone never reached 3
cycles/s). Pairing counts both styles correctly.

Two-phase trip rule (operator spec v3, after live incidents):

  1. SOFT  -- >= ``min_cycles`` (3) pumps inside the trailing
     ``window_s`` (1.0 s), chord held -> abort everything, idle stand
     (recoverable). 4-5 pumps stay HERE.
  2. DAMP  -- the last ``damp_cycles`` (6) pumps must form ONE RAPID
     BURST (spanning <= ``damp_span_s``, 1.5 s) AND at least
     ``escalate_after_s`` (1.0 s) must have passed
     since the FIRST press after chord engagement (fixed anchor -- a
     sliding anchor made damp unreachable under continuous pumping).
     Slow pumps (6 over 3-5 s) NEVER damp.

Releasing the chord resets everything and re-arms — but only a REAL
release: gaps shorter than ``chord_grace_s`` (0.8 s) are bridged, not
reset. Live finding 2026-08-03: Quest WebXR input sources flap for
~100-500 ms every few seconds; each flap zeroed the buttons for a beat
and wiped all pump progress, so the VR gesture could never complete
even though input flowed fine between flaps (robot fingers visibly
tracked the triggers). The pad never flaps — hence "pad works, VR
doesn't". Trigger cycles simply pause during a bridged gap.

Transport-agnostic: feed the two analog levels (normalized 0..1
pressed) + chord state each tick; ``tick`` returns 0 = nothing,
1 = SOFT fired (once), 2 = escalated to DAMP (once).

REGRESSION SUITE: test_estop_gesture.py next to this file encodes every
live incident; the preflight gate runs it before any ship.
"""
from __future__ import annotations

import time
from collections import deque


class EstopGesture:
    def __init__(self, window_s: float = 1.0, min_cycles: int = 3,
                 press_thresh: float = 0.6, release_thresh: float = 0.3,
                 escalate_after_s: float = 1.0, damp_cycles: int = 6,
                 damp_span_s: float = 1.5, pair_window_s: float = 0.5,
                 chord_grace_s: float = 0.8):
        self.window_s = float(window_s)
        self.min_cycles = int(min_cycles)
        self.press_t = float(press_thresh)
        self.release_t = float(release_thresh)
        self.escalate_after_s = float(escalate_after_s)
        self.damp_cycles = int(damp_cycles)
        self.damp_span_s = float(damp_span_s)
        self.pair_window_s = float(pair_window_s)
        self.chord_grace_s = float(chord_grace_s)
        self._last_chord_true_t: float | None = None
        self._pressed = [False, False]
        # Pump pairing: a cycle on one channel waits in _unmatched for
        # a cycle on the OTHER channel within pair_window_s; a match =
        # one pump. _pump_win (pruned to window_s) drives the soft
        # trip; _pump_recent (last damp_cycles pumps) drives the burst
        # check. Single-channel cycles never pair -> never count.
        self._unmatched: list[deque] = [deque(), deque()]
        self._pump_win: deque = deque()
        self._pump_recent: deque = deque(maxlen=int(damp_cycles))
        self._chord_since: float | None = None
        # Fixed anchor for the escalate floor: time of the FIRST press
        # cycle since chord engagement. Must NOT slide with the burst
        # window (a sliding anchor kept damp unreachable under
        # continuous fast pumping).
        self._first_cycle_t: float | None = None
        self._latched = False
        self._escalated = False

    def tick(self, a: float, b: float, chord_held: bool,
             now: float | None = None) -> int:
        """Feed one sample. Returns 0 = nothing, 1 = SOFT e-stop fired
        (once), 2 = ESCALATED to damp (once, only after a 1)."""
        now = time.monotonic() if now is None else now
        if chord_held:
            self._last_chord_true_t = now
        elif (self._last_chord_true_t is not None
                and self._chord_since is not None
                and now - self._last_chord_true_t <= self.chord_grace_s):
            # Micro-drop (WebXR source flap / WS hiccup): bridge it.
            # Cycles pause (levels read 0) but progress is kept.
            chord_held = True
        if not chord_held:
            self._chord_since = None
            self._last_chord_true_t = None
            self._pressed = [False, False]
            for d in self._unmatched:
                d.clear()
            self._pump_win.clear()
            self._pump_recent.clear()
            self._first_cycle_t = None
            self._latched = False
            self._escalated = False
            return 0
        if self._chord_since is None:
            self._chord_since = now
        for idx, level in ((0, a), (1, b)):
            if not self._pressed[idx] and level >= self.press_t:
                self._pressed[idx] = True
            elif self._pressed[idx] and level <= self.release_t:
                self._pressed[idx] = False
                if self._first_cycle_t is None:
                    self._first_cycle_t = now
                other = self._unmatched[1 - idx]
                while other and now - other[0] > self.pair_window_s:
                    other.popleft()
                if other:                       # L+R pair -> one pump
                    other.popleft()
                    self._pump_win.append(now)
                    self._pump_recent.append(now)
                else:
                    mine = self._unmatched[idx]
                    while mine and now - mine[0] > self.pair_window_s:
                        mine.popleft()
                    mine.append(now)
        while self._pump_win and now - self._pump_win[0] > self.window_s:
            self._pump_win.popleft()
        pumps_in_window = len(self._pump_win)
        if (not self._latched
                and pumps_in_window >= self.min_cycles
                and now - self._chord_since >= 0.15):
            self._latched = True
            return 1
        burst_ok = (
            len(self._pump_recent) >= self.damp_cycles
            and self._pump_recent[-1] - self._pump_recent[0]
            <= self.damp_span_s
        )
        if (self._latched and not self._escalated
                and burst_ok
                and self._first_cycle_t is not None
                and now - self._first_cycle_t >= self.escalate_after_s):
            self._escalated = True
            return 2
        return 0
