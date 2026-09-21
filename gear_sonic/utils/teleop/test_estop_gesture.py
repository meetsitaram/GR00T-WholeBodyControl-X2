"""Regression suite for the operator e-stop gesture detector.

Every test named ``test_incident_*`` encodes a REAL failure observed
live in sim on 2026-08-03. This suite is run by the preflight gate
(preflight_planner.py) before any planner/teleop file ships to PC2 —
a red test here means the e-stop would misbehave on the robot.

Run directly:  python3 gear_sonic/utils/teleop/test_estop_gesture.py
Or via pytest: pytest gear_sonic/utils/teleop/test_estop_gesture.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estop_gesture import EstopGesture  # noqa: E402


# ---------------------------------------------------------------------------
# Simulators for the two real-world firing styles.
# ---------------------------------------------------------------------------

def pump_both(g, t0, n_pumps, hz=5.0, chord=True):
    """The REAL gesture: L2+R2 (or both VR triggers) pumped TOGETHER.

    One pump = both channels press then release. Returns
    [(t, phase), ...] and the end time.
    """
    events = []
    t = t0
    for _ in range(n_pumps):
        events.append((t, g.tick(1.0, 1.0, chord, now=t)))
        t += 0.5 / hz
        events.append((t, g.tick(0.0, 0.0, chord, now=t)))
        t += 0.5 / hz
    return events, t


def tap_alternating(g, t0, n_taps, hz=6.0, chord=True):
    """Single-finger style: L2, then R2, then L2 ... one at a time."""
    events = []
    t = t0
    for i in range(n_taps):
        lt, rt = ((1.0, 0.0) if i % 2 == 0 else (0.0, 1.0))
        events.append((t, g.tick(lt, rt, chord, now=t)))
        t += 0.5 / hz
        events.append((t, g.tick(0.0, 0.0, chord, now=t)))
        t += 0.5 / hz
    return events, t


def hold_quiet(g, t, secs, chord=True):
    """Chord held, triggers idle."""
    return [(t + k * 0.02, g.tick(0.0, 0.0, chord, now=t + k * 0.02))
            for k in range(int(secs / 0.02))]


def phases(events):
    return [p for _, p in events]


# ---------------------------------------------------------------------------
# Incident regressions (2026-08-03, live sim).
# ---------------------------------------------------------------------------

def test_incident_three_gentle_pumps_must_not_damp():
    """THE COLLAPSE: walking with both sticks forward, operator gently
    pumped L2+R2 three times -> robot went straight to pure damping.
    Root cause: each pump counted as TWO cycles (L2 + R2). Three pumps
    must read as three presses: idle-stand at most, never damp."""
    g = EstopGesture()
    ev, t = pump_both(g, 0.2, 3, hz=2.5)          # "gentle" ~2.5 pumps/s
    ev += hold_quiet(g, t, 5.0)
    ph = phases(ev)
    assert 2 not in ph, "3 gentle pumps reached DAMP — collapse regression"
    assert 1 in ph, "3 pumps should still soft-trip to idle stand"


def test_incident_continuous_pumping_must_reach_damp():
    """Pad test: continuous fast pumping fired soft repeatedly but damp
    was unreachable (the 1s floor was anchored to a sliding window).
    Sustained deliberate pumping MUST damp, and not before 1.0s."""
    g = EstopGesture()
    ev, _ = pump_both(g, 0.2, 15, hz=5.0)          # 3s of deliberate pumping
    ph = phases(ev)
    assert 1 in ph and 2 in ph, "sustained pumping failed to damp"
    damp_t = next(t for t, p in ev if p == 2)
    assert damp_t - 0.2 >= 0.99, f"damp too early: +{damp_t - 0.2:.2f}s"


def test_incident_slow_presses_never_damp():
    """Operator question: chord held, 6 pumps trickling over 3-5s must
    never reach damping (accidental-collapse guard for demos)."""
    for total_s in (3.0, 5.0):
        g = EstopGesture()
        ev, t = pump_both(g, 0.2, 6, hz=6.0 / total_s)
        ev += hold_quiet(g, t, 3.0)
        assert 2 not in phases(ev), f"slow 6 pumps over {total_s}s damped"


def test_incident_first_vr_spec_three_presses_damped_too_easily():
    """VR test: 3 rapid presses in a second escalated straight to damp
    (old escalate-after-soft rule). 3-5 pumps must park at idle stand
    forever, even with the chord held for a long time."""
    for n in (3, 4, 5):
        g = EstopGesture()
        ev, t = pump_both(g, 0.2, n, hz=5.0)
        ev += hold_quiet(g, t, 5.0)
        ph = phases(ev)
        assert 1 in ph, f"{n} rapid pumps should soft-trip"
        assert 2 not in ph, f"{n} rapid pumps must NOT damp"


# ---------------------------------------------------------------------------
# Spec behaviors (operator design v3).
# ---------------------------------------------------------------------------

def test_six_rapid_pumps_damp_after_one_second_floor():
    g = EstopGesture()
    ev, t = pump_both(g, 0.2, 6, hz=5.0)
    ev += hold_quiet(g, t, 2.0)
    ph = phases(ev)
    assert 1 in ph and 2 in ph
    damp_t = next(tt for tt, p in ev if p == 2)
    assert damp_t - 0.2 >= 0.99


def test_ultra_fast_burst_still_held_to_floor():
    g = EstopGesture()
    ev, t = pump_both(g, 0.2, 10, hz=12.0)
    ev += hold_quiet(g, t, 1.5)
    damp_times = [tt for tt, p in ev if p == 2]
    assert damp_times, "fast burst never damped"
    assert damp_times[0] - 0.2 >= 0.99


def test_stragglers_after_soft_do_not_damp():
    g = EstopGesture()
    ev, t = pump_both(g, 0.2, 3, hz=5.0)
    ev2, t = pump_both(g, t + 1.2, 3, hz=0.8)      # slow stragglers
    ev2 += hold_quiet(g, t, 3.0)
    assert 2 not in phases(ev2)


def test_fresh_rapid_burst_after_pause_damps():
    g = EstopGesture()
    _, t = pump_both(g, 0.2, 3, hz=5.0)
    ev, t = pump_both(g, t + 2.0, 6, hz=5.0)
    ev += hold_quiet(g, t, 1.5)
    assert 2 in phases(ev)


def test_one_trigger_alone_never_counts():
    """min() across channels: hammering a SINGLE trigger (the other
    never pressed) accumulates zero pumps — nothing may ever fire."""
    g = EstopGesture()
    t = 0.2
    ev = []
    for _ in range(15):                     # 15 fast L2-only taps
        ev.append((t, g.tick(1.0, 0.0, True, now=t))); t += 0.0625
        ev.append((t, g.tick(0.0, 0.0, True, now=t))); t += 0.0625
    ev += hold_quiet(g, t, 2.0)
    ph = phases(ev)
    assert 1 not in ph and 2 not in ph, \
        "single-trigger hammering fired the e-stop"


def test_incident_vr_alternating_fire_must_trip():
    """VR incident #2 (2026-08-03): operator held A+X and rapid-fired
    both triggers ALTERNATELY — the min()-per-channel rule needed each
    trigger to reach 3 cycles/s alone, so a realistic alternating pace
    (6 taps/s total = 3/channel) never fired. Pair-matching must trip
    soft at that pace and at the faster 8 taps/s."""
    for hz in (6.0, 8.0):
        g = EstopGesture()
        ev, t = tap_alternating(g, 0.2, 12, hz=hz)
        assert 1 in phases(ev), \
            f"alternating fire at {hz} taps/s failed to soft-trip"


def test_alternating_sustained_reaches_damp():
    """Sustained alternating fire is just as deliberate as sustained
    pumping — it must escalate to damp too (after the 1s floor)."""
    g = EstopGesture()
    ev, _ = tap_alternating(g, 0.2, 30, hz=8.0)
    ph = phases(ev)
    assert 2 in ph, "sustained alternating fire never damped"
    damp_t = next(tt for tt, p in ev if p == 2)
    assert damp_t - 0.2 >= 0.99


def test_incident_source_flap_must_not_wipe_gesture():
    """VR incident #3 (2026-08-03): WebXR input sources flapped for
    100-500ms every few seconds; every flap read the chord as released
    and wiped all pump progress, so the gesture could NEVER complete on
    VR (robot fingers visibly tracked triggers between flaps; pad has
    no flaps and worked). Gaps under chord_grace_s must bridge: a
    pumping gesture interrupted by 0.3s input dropouts every ~1s must
    still reach SOFT and DAMP."""
    g = EstopGesture()
    t = 0.2
    ev = []
    for i in range(40):                      # ~4s of pumping at 5Hz
        # every 5th pump window, a 0.3s total input dropout (chord
        # False, triggers 0) split across 3 ticks
        if i % 10 in (4, 5, 6):
            ev.append((t, g.tick(0.0, 0.0, False, now=t))); t += 0.1
            continue
        ev.append((t, g.tick(1.0, 1.0, True, now=t))); t += 0.1
        ev.append((t, g.tick(0.0, 0.0, True, now=t))); t += 0.1
    ph = phases(ev)
    assert 1 in ph, "flap-chopped gesture never soft-tripped"
    assert 2 in ph, "flap-chopped gesture never damped"
    damp_t = next(tt for tt, p in ev if p == 2)
    assert damp_t - 0.2 >= 0.99, "floor violated"


def test_deliberate_chord_release_still_resets():
    """The grace bridges only MICRO-drops: a real release (> grace_s)
    resets everything and re-arms."""
    g = EstopGesture()
    _, t = pump_both(g, 0.2, 5, hz=5.0)
    for k in range(30):                      # 1.5s released >> 0.8s grace
        g.tick(0.0, 0.0, False, now=t + k * 0.05)
    ev, _ = pump_both(g, t + 1.6, 3, hz=5.0)
    ph = phases(ev)
    assert 1 in ph and 2 not in ph, "long release must reset the count"


def test_chord_release_resets_and_rearms():
    g = EstopGesture()
    _, t = pump_both(g, 0.2, 5, hz=5.0)
    for k in range(25):                            # 1.25s release > grace
        g.tick(0.0, 0.0, False, now=t + 0.05 + k * 0.05)
    ev, _ = pump_both(g, t + 1.4, 3, hz=5.0)
    ph = phases(ev)
    assert 1 in ph and 2 not in ph, "release must wipe the count"


def test_no_chord_means_nothing_ever_fires():
    g = EstopGesture()
    ev, _ = pump_both(g, 0.2, 20, hz=6.0, chord=False)
    assert phases(ev) == [0] * len(ev)


def test_soft_fires_once_and_damp_fires_once():
    g = EstopGesture()
    ev, t = pump_both(g, 0.2, 15, hz=5.0)
    ph = phases(ev)
    assert ph.count(1) == 1 and ph.count(2) == 1


# ---------------------------------------------------------------------------
# Wiring guards: the surfaces must not regrow removed/false chords.
# ---------------------------------------------------------------------------

def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def test_wiring_stick_chord_stays_removed():
    """2026-08-03 collapse: both-sticks-forward is a driving posture and
    must never arm the e-stop again on ANY surface."""
    for rel in ("gear_sonic/scripts/pad_locomotion_bridge.py",
                "gear_sonic/scripts/quest3_manager_x2.py"):
        path = os.path.join(_repo_root(), rel)
        src = open(path).read()
        assert "stick_chord =" not in src, \
            f"{rel}: stick chord re-appeared (collapse regression)"
        assert "estop_gesture2" not in src.lower().replace("self._", ""), \
            f"{rel}: second gesture detector re-appeared"


def test_wiring_a_plus_x_reserved_for_estop():
    """A+X is RESERVED for the e-stop chord (operator rule 2026-08-03).

    Any new binding that reads the A and X buttons together in a normal
    workflow would collide with the emergency gesture — an operator
    doing that workflow would be one trigger-pump away from an e-stop,
    or worse, the workflow would swallow the chord. Fail loudly on any
    line that combines both buttons unless it is explicitly the estop
    chord (marked by 'estop' or 'A+X' on the line)."""
    combos = {
        "gear_sonic/scripts/pad_locomotion_bridge.py":
            [("get_button(0)", "get_button(2)")],
        "gear_sonic/scripts/quest3_manager_x2.py":
            [("a_held", "x_held"), ("a_pressed", "x_pressed"),
             ("ev.a_", "ev.x_")],
    }
    for rel, pairs in combos.items():
        path = os.path.join(_repo_root(), rel)
        for lineno, line in enumerate(open(path).read().splitlines(), 1):
            if "= buttons" in line:
                continue        # tuple unpack of the button states, not a binding
            if all(t in line for t in ("a_", "b_", "x_", "y_")):
                continue        # any-button pattern (all four ORed), not a chord
            for tok_a, tok_x in pairs:
                if tok_a in line and tok_x in line:
                    low = line.lower()
                    assert "estop" in low or "a+x" in low, (
                        f"{rel}:{lineno} combines A and X buttons outside "
                        f"the e-stop chord — A+X is reserved for e-stop: "
                        f"{line.strip()!r}")


def test_wiring_both_surfaces_use_default_detector():
    """Threshold changes must go through EstopGesture defaults (covered
    by this suite), not per-surface constructor overrides."""
    for rel in ("gear_sonic/scripts/pad_locomotion_bridge.py",
                "gear_sonic/scripts/quest3_manager_x2.py"):
        path = os.path.join(_repo_root(), rel)
        src = open(path).read()
        assert "EstopGesture()" in src, \
            f"{rel}: detector constructed with overrides — suite blind"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
