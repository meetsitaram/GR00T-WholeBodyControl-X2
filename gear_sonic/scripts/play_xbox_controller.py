#!/usr/bin/env python3
"""Xbox-controller launcher for the X2 demo motion-clip wire.

Sibling of :file:`play_locomotion.py` and :file:`play_gesture.py`. All
three scripts PUB-connect to the same ``motion_clip_cmd`` ZMQ topic
(default port :data:`MOTION_CLIP_CMD_DEFAULT_PORT` = 5568) that the
recorder running inside
:file:`gear_sonic/scripts/run_x2_pkl_direct_stack.sh` or
:file:`gear_sonic/scripts/run_x2_quest3_planner_stack.sh` SUBs.

This script differs in that it is a **long-running foreground loop**
that reads an Xbox controller via :mod:`pygame.joystick` and dispatches
clips on chord edges instead of taking a single ``--pkl`` on the CLI.

Prereqs (assumed already running in other shells)
-------------------------------------------------

1. ``gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --pc2-host <PC2>``
2. Either ``gear_sonic/scripts/run_x2_pkl_direct_stack.sh --pc2-host
   <PC2>`` or ``gear_sonic/scripts/run_x2_quest3_planner_stack.sh
   --pc2-host <PC2>`` (both bind the ``motion_clip_cmd`` SUB on
   ``tcp://*:5568``).

Bindings (edit the BINDINGS_* dicts below to remap)
---------------------------------------------------

Locomotion (D-pad direction + L2+R2 held as deadman, L1+R1 released):

* D-pad up    -> relaxed walk forward
* D-pad left  -> relaxed walk one-left-turn
* D-pad right -> relaxed walk one-right-turn
* D-pad down  -> relaxed walk two-right-turns (about-face)

Gestures (A/B/X/Y + L1 OR R1 modifier, L2+R2 released):

* A+L1, A+R1, B+L1, B+R1, X+L1, X+R1 -> assorted demo / mc gestures
* Y+L1, Y+R1 -> FREE slots (set the values in BINDINGS_GESTURES below)

Emergency stop (all-trigger chord, always live, bypasses busy gate):

* L1+R1+L2+R2 (all four held) -> publish ``{"action": "stop"}`` and
  reset the local single-flight gate so the next chord can fire.

Single-flight gating
--------------------

The recorder side will *supersede* an in-flight session at frame 0 of
any new ``play`` payload (see ``_drain_clip_commands`` in
the dataset recorder). To avoid
mid-stride snap-cuts during the demo, this launcher gates new
locomotion / gesture chord edges on a local ``busy_until_ts`` timer
sized via :func:`estimate_duration_s` -- the same function
:file:`play_locomotion.py` and :file:`play_gesture.py` use to size
their own SIGINT-blockable sleeps. The e-stop chord remains live at
all times and resets the gate so the operator's "abort + redo"
pattern is always a hit-and-fire combo.

Usage::

    # Probe the connected pad's axes / buttons / hats and exit.
    .venv/bin/python -m gear_sonic.scripts.play_xbox_controller --list

    # Run the listener (assumes the stack is already up in another shell).
    .venv/bin/python -m gear_sonic.scripts.play_xbox_controller

    # Same, but never publish on the wire -- just print what *would* fire.
    .venv/bin/python -m gear_sonic.scripts.play_xbox_controller --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing pygame prints a banner on stderr; suppress it for cleaner
# operator output. Set before the import.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
import zmq  # noqa: E402


def _ensure_pygame_subsystems() -> None:
    """Init the pygame subsystems we depend on.

    ``pygame.event.pump`` (which we call every tick to drain SDL's
    internal event queue and keep joystick state fresh on macOS /
    Windows) raises ``video system not initialized`` unless the
    display subsystem has been initialized at least once. On a
    headless box (no ``DISPLAY``) we still need it; force the
    ``dummy`` SDL video driver so it succeeds without an X server.
    Safe to call repeatedly: pygame.*.init() is idempotent.
    """
    has_display = (
        "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
    )
    if not has_display:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        pygame.display.init()
    except pygame.error:
        # Last-resort: fall through with the warning. event.pump may
        # still raise; the operator can re-run with `DISPLAY=:0` set
        # or install a dummy SDL.
        print(
            "[xbox] WARN pygame.display.init() failed; "
            "event.pump may not work. Set DISPLAY or "
            "SDL_VIDEODRIVER=dummy.",
            flush=True,
            file=sys.stderr,
        )
    pygame.joystick.init()

from gear_sonic.utils.teleop.motion_clip_session import (  # noqa: E402
    MOTION_CLIP_CMD_DEFAULT_PORT,
    MOTION_CLIP_CMD_DEFAULT_TOPIC,
    MotionClipEntry,
    estimate_duration_s,
)


# ─── BINDINGS (edit these to remap / fill the FREE slots) ─────────────
#
# Motion clips are NOT shipped with the repository. Two ways to bind a
# slot (both resolved against ``REPO_ROOT`` when relative):
#
#   * ``X2_PAD_CLIPS_DIR=<dir>`` -- a directory of single-clip motion
#     PKLs (one clip per file, e.g. exported with
#     ``gear_sonic/scripts/window_motion_pkl.py``). The relative names
#     below are resolved inside it. Unset -> those slots stay FREE.
#   * ``"<bank.pkl>::<motion_key>"`` -- one key inside a multi-clip bank
#     PKL. Default bank: ``gear_sonic/data/motions/x2_demo_bank.pkl``
#     (regenerated from the upstream example clips by
#     ``tools/build_demo_bank_from_upstream.sh``; see REGENERATE.md).
#     Override with ``X2_CLIP_BANK=<path>``.
#
# Slots whose clip file is absent at startup are reported and left FREE;
# set ``X2_PAD_STRICT=1`` to make a missing clip a startup failure.

_CLIPS_DIR: str = os.environ.get("X2_PAD_CLIPS_DIR", "")
_CLIP_BANK: str = os.environ.get(
    "X2_CLIP_BANK", "gear_sonic/data/motions/x2_demo_bank.pkl"
)


def _clip_file(rel_name: str) -> Optional[str]:
    """Single-clip PKL under ``$X2_PAD_CLIPS_DIR`` (None when unset)."""
    return f"{_CLIPS_DIR}/{rel_name}" if _CLIPS_DIR else None


def _bank(motion_key: str) -> Optional[str]:
    """``<bank.pkl>::<key>`` reference into ``$X2_CLIP_BANK``."""
    return f"{_CLIP_BANK}::{motion_key}" if _CLIP_BANK else None


BINDINGS_LOCOMOTION: dict[str, Optional[str]] = {
    "UP": _clip_file("x2_ultra_relaxed_walk_forward_v1.pkl"),
    "LEFT": _clip_file("x2_ultra_relaxed_walk_one_left_turn_v1.pkl"),
    "RIGHT": _clip_file("x2_ultra_relaxed_walk_one_right_turn_v1.pkl"),
    "DOWN": _clip_file("x2_ultra_relaxed_walk_two_right_turns_v1.pkl"),
}

BINDINGS_GESTURES: dict[str, Optional[str]] = {
    # Bare face button (no shoulders, no triggers held): demo-bank keys.
    "A":    _bank("right_wave_001"),
    "B":    _bank("right_kiss_001"),
    "X":    _bank("right_five_001"),
    "Y":    _bank("right_shake_001"),
    # L1 modifier (L1 held alone -- not L2/R2, not R1).
    "A+L1": None,
    "B+L1": None,
    "X+L1": None,
    "Y+L1": _clip_file("mc_gestures/bow_001.pkl"),
    # R1 modifier.
    "A+R1": None,
    "B+R1": None,
    "X+R1": None,
    "Y+R1": _clip_file("mc_gestures/right_shake_001.pkl"),
    # L2 modifier (L2 held alone -- not R2; locomotion deadman needs BOTH).
    "A+L2": None,
    "B+L2": None,
    "X+L2": None,
    "Y+L2": _clip_file("demo_gestures/left_wave_high_001.pkl"),
    # R2 modifier.
    "A+R2": None,
    "B+R2": None,
    "X+R2": _clip_file("demo_gestures/chicken_001.pkl"),
    "Y+R2": _clip_file("demo_gestures/right_wave_001.pkl"),
}

# After publishing the e-stop ``{"action": "stop"}`` payload, the
# launcher also fires this PKL as a gesture so the operator gets a
# visible acknowledgment on the robot ("NO, that was aborted"). When
# the gesture finishes, the recorder's built-in
# ``DEFAULT_STAND_POSE_MUJOCO_RAD`` idle-stand fallback takes over
# automatically (the recorder's ``_publish_idle``).
# None (the default without $X2_PAD_CLIPS_DIR) skips the acknowledgment
# and snaps directly to idle. Override with X2_PAD_ESTOP_FOLLOWUP=<pkl>.
ESTOP_FOLLOWUP_PKL: Optional[str] = (
    os.environ.get("X2_PAD_ESTOP_FOLLOWUP")
    or _clip_file("mc_gestures/shake_head_001.pkl")
)


# ─── Controller mapping (standard Linux xpad / SDL2 Xbox) ─────────────
#
# Verify with `--list` before the demo if you swap pads. The defaults
# below match a wired Xbox One / Series controller on the xpad kernel
# driver and an Xbox 360 wireless dongle.

BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_LB = 4   # L1
BTN_RB = 5   # R1

# Triggers expose as analog axes. xpad reports rest=-1.0, fully
# pressed=+1.0. A threshold of >0.0 catches "any squeeze". 0.5 is
# safer against drift but requires a deliberate squeeze; matches
# typical deadman conventions.
AXIS_LT = 2  # L2
AXIS_RT = 5  # R2
TRIGGER_THRESHOLD = 0.5

HAT_DPAD = 0  # the D-pad usually lives on hat 0


# ─── Timing constants ─────────────────────────────────────────────────

TICK_HZ = 60.0
TICK_DT = 1.0 / TICK_HZ
STATUS_PRINT_INTERVAL_S = 1.0

# Duration buffer added to estimate_duration_s when arming the
# single-flight gate. Mirrors play_locomotion.py's +0.1 s nap but
# slightly longer to cover the recorder's snap-back-to-idle settle.
BUSY_TAIL_BUFFER_S = 0.4

# Once a chord fires we ignore subsequent edges for this long to
# guard against switch bounce / human double-press.
DEBOUNCE_AFTER_DISPATCH_S = 0.25

# After an e-stop the chord must be released and re-pressed before
# firing again. This cooldown stops a held 4-trigger chord from
# spamming stops every tick.
ESTOP_REARM_COOLDOWN_S = 0.5

# Haptic feedback on BUSY-ignored chord edges. Operator-asked
# behaviour: tactile reject when you press something while a clip is
# still in flight so you can keep your eyes on the robot. Tuned to
# medium strength (the Xbox Series X pad rumble is quite strong) and
# short enough to be unambiguously a "no" rather than a sustained buzz.
RUMBLE_BUSY_LOW = 0.55
RUMBLE_BUSY_HIGH = 0.55
RUMBLE_BUSY_MS = 150

# Hat -> direction lookup. Diagonals map to None so the operator must
# commit to one direction before a locomotion clip fires.
HAT_TO_DIR: dict[tuple[int, int], Optional[str]] = {
    (0, 0): None,
    (0, 1): "UP",
    (0, -1): "DOWN",
    (-1, 0): "LEFT",
    (1, 0): "RIGHT",
    (1, 1): None,
    (1, -1): None,
    (-1, 1): None,
    (-1, -1): None,
}


# ─── Chord state machine ──────────────────────────────────────────────


@dataclass
class ChordState:
    """Snapshot of all chord-relevant inputs for one tick.

    All ``bool`` fields are post-threshold; ``dpad_dir`` is a tag
    (None / "UP" / "DOWN" / "LEFT" / "RIGHT") for the hat 0 position.
    """

    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    l1: bool = False
    r1: bool = False
    l2: bool = False
    r2: bool = False
    dpad_dir: Optional[str] = None

    @property
    def estop_chord(self) -> bool:
        """L1+R1+L2+R2 all held. The dedicated all-trigger e-stop."""
        return self.l1 and self.r1 and self.l2 and self.r2

    @property
    def loco_deadman(self) -> bool:
        """L2+R2 held, L1+R1 released. Distinguishes locomotion fires
        from e-stop chord activations (which also have L2+R2).
        """
        return self.l2 and self.r2 and not self.l1 and not self.r1

    @property
    def gesture_modifier(self) -> Optional[str]:
        """Returns the chord-key suffix for face-button gesture
        bindings, or ``None`` when the modifier combination is
        ambiguous / unsafe.

        Suffixes (joined onto ``"A"`` / ``"B"`` / ``"X"`` / ``"Y"``
        to form ``BINDINGS_GESTURES`` keys):

        * ``""`` (empty)   - bare face button, no modifiers held
        * ``"+L1"``        - L1 held alone
        * ``"+R1"``        - R1 held alone
        * ``"+L2"``        - L2 held alone (not R2)
        * ``"+R2"``        - R2 held alone (not L2)

        Returns ``None`` (face buttons silent) when:

        * BOTH L2+R2 held  - locomotion deadman engaged; no gesture
          may fire while the operator is arming a walk.
        * BOTH L1+R1 held  - on the way to e-stop chord; no gesture.
        * Any mix of shoulder (L1/R1) + trigger (L2/R2) - reserved
          for future chord layouts; explicitly invalid today.
        """
        # Locomotion-armed: both triggers held -> face buttons silent.
        if self.l2 and self.r2:
            return None
        # E-stop transit: both shoulders held -> face buttons silent.
        if self.l1 and self.r1:
            return None
        # Mixed shoulder+trigger: reserved.
        if (self.l1 or self.r1) and (self.l2 or self.r2):
            return None
        if self.l1:
            return "+L1"
        if self.r1:
            return "+R1"
        if self.l2:
            return "+L2"
        if self.r2:
            return "+R2"
        return ""


def read_chord_state(js: "pygame.joystick.Joystick") -> ChordState:
    """Build a ChordState from the joystick's current axes/buttons/hat."""
    # Bounds-check button/axis indices so a non-standard pad doesn't
    # crash the loop -- treat missing inputs as released.
    n_buttons = js.get_numbuttons()
    n_axes = js.get_numaxes()
    n_hats = js.get_numhats()

    def btn(i: int) -> bool:
        return bool(js.get_button(i)) if 0 <= i < n_buttons else False

    def axis(i: int) -> float:
        return float(js.get_axis(i)) if 0 <= i < n_axes else -1.0

    if 0 <= HAT_DPAD < n_hats:
        hx, hy = js.get_hat(HAT_DPAD)
    else:
        hx, hy = (0, 0)

    return ChordState(
        a=btn(BTN_A),
        b=btn(BTN_B),
        x=btn(BTN_X),
        y=btn(BTN_Y),
        l1=btn(BTN_LB),
        r1=btn(BTN_RB),
        l2=axis(AXIS_LT) > TRIGGER_THRESHOLD,
        r2=axis(AXIS_RT) > TRIGGER_THRESHOLD,
        dpad_dir=HAT_TO_DIR.get((int(hx), int(hy)), None),
    )


# ─── Wire helpers ─────────────────────────────────────────────────────


@dataclass
class WirePub:
    """Lazy wrapper around the ZMQ PUB to the recorder. ``dry_run``
    suppresses the actual send (chord edges still log).
    """

    host: str
    port: int
    topic: str
    linger_ms: int
    dry_run: bool
    _sock: "Optional[zmq.Socket[Any]]" = field(default=None, init=False)

    def connect(self) -> None:
        if self.dry_run:
            print(
                f"[xbox] DRY-RUN: skipping PUB connect "
                f"(would have been tcp://{self.host}:{self.port} "
                f"topic={self.topic!r})",
                flush=True,
            )
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, max(0, self.linger_ms))
        url = f"tcp://{self.host}:{self.port}"
        sock.connect(url)
        self._sock = sock
        print(
            f"[xbox] PUB connect {url} topic={self.topic!r} "
            f"(slow-joiner sleep {self.linger_ms} ms)",
            flush=True,
        )
        # Slow-joiner mitigation -- same as play_locomotion.py.
        time.sleep(max(0.0, self.linger_ms / 1000.0))

    def send(self, payload: dict[str, Any]) -> None:
        if self.dry_run or self._sock is None:
            return
        self._sock.send_multipart([
            self.topic.encode("ascii"),
            json.dumps(payload).encode("utf-8"),
        ])

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=max(0, self.linger_ms))
            self._sock = None


# ─── Main loop ────────────────────────────────────────────────────────


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _split_clip_ref(p: Path) -> tuple[Path, Optional[str]]:
    """Split a ``<bank.pkl>::<motion_key>`` reference into (path, key).
    Plain paths come back as ``(p, None)`` (= first key in the PKL).
    """
    text = str(p)
    if "::" in text:
        src, key = text.rsplit("::", 1)
        return Path(src), (key or None)
    return p, None


def _resolve_paths(
    bindings: dict[str, Optional[str]],
) -> dict[str, Optional[Path]]:
    """Resolve repo-relative binding values to absolute paths. ``None``
    stays ``None`` (free slot).
    """
    out: dict[str, Optional[Path]] = {}
    for k, v in bindings.items():
        if v is None:
            out[k] = None
            continue
        src, key = _split_clip_ref(Path(v))
        if not src.is_absolute():
            src = REPO_ROOT / src
        out[k] = Path(f"{src}::{key}") if key else src
    return out


def _resolve_single_path(value: Optional[str]) -> Optional[Path]:
    """Same as ``_resolve_paths`` but for a single scalar string."""
    if value is None:
        return None
    src, key = _split_clip_ref(Path(value))
    if not src.is_absolute():
        src = REPO_ROOT / src
    return Path(f"{src}::{key}") if key else src


def _preflight_paths(
    loco: dict[str, Optional[Path]],
    gestures: dict[str, Optional[Path]],
    estop_followup: Optional[Path],
) -> tuple[dict[str, Optional[Path]], dict[str, Optional[Path]], Optional[Path]]:
    """Check every bound clip file exists. Missing clips are reported and
    their slots left FREE; with ``X2_PAD_STRICT=1`` a missing clip is a
    startup failure instead. Returns the (possibly pruned) bindings.
    """
    strict = os.environ.get("X2_PAD_STRICT", "0") == "1"
    missing: list[str] = []

    def _keep(label: str, p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        src, _ = _split_clip_ref(p)
        if src.is_file():
            return p
        missing.append(f"  {label}: {p}")
        return None

    loco = {k: _keep(k, p) for k, p in loco.items()}
    gestures = {k: _keep(k, p) for k, p in gestures.items()}
    estop_followup = _keep("ESTOP_FOLLOWUP", estop_followup)
    if missing:
        print(
            ("[xbox] STARTUP FAIL: " if strict else "[xbox] WARN: ")
            + "the following bound clips are missing"
            + ("" if strict else " (slots left FREE)")
            + ":\n" + "\n".join(missing)
            + "\n[xbox] Clips are not shipped with the repo: set X2_PAD_CLIPS_DIR"
            " (single-clip PKLs) and/or regenerate the demo bank"
            " (X2_CLIP_BANK, default gear_sonic/data/motions/x2_demo_bank.pkl;"
            " see REGENERATE.md).",
            flush=True,
            file=sys.stderr,
        )
        if strict:
            raise SystemExit(2)
    if not any(loco.values()) and not any(gestures.values()):
        print(
            "[xbox] NOTE: no clip bound to any chord -- only the e-stop chord"
            " will do anything. See the BINDINGS section of this script.",
            flush=True,
            file=sys.stderr,
        )
    return loco, gestures, estop_followup


def _print_bindings(
    loco: dict[str, Optional[Path]],
    gestures: dict[str, Optional[Path]],
    estop_followup: Optional[Path],
) -> None:
    print("[xbox] Bindings (edit BINDINGS_* in the script to remap):", flush=True)
    print("  Locomotion (D-pad + L2+R2 deadman, L1+R1 released):", flush=True)
    for k, p in loco.items():
        tag = p.name if p is not None else "(FREE)"
        print(f"    D-pad {k:<5} -> {tag}", flush=True)
    print(
        "  Gestures (A/B/X/Y bare or + single modifier L1|R1|L2|R2;"
        " any other combo is silent):",
        flush=True,
    )
    for k, p in gestures.items():
        tag = p.name if p is not None else "(FREE)"
        print(f"    {k:<5}        -> {tag}", flush=True)
    followup_tag = (
        estop_followup.name if estop_followup is not None
        else "(none; pure stop -> recorder idle stand)"
    )
    print(
        "  E-stop chord (always live, bypasses busy gate):\n"
        f"    L1+R1+L2+R2  -> stop + play {followup_tag}",
        flush=True,
    )


def _list_and_exit() -> int:
    """``--list``: enumerate connected joysticks + raw axes/buttons
    for the first one, then exit. No ZMQ traffic.
    """
    _ensure_pygame_subsystems()
    n = pygame.joystick.get_count()
    if n == 0:
        print(
            "[xbox] No joysticks detected. Plug in the Xbox pad "
            "and re-run.",
            flush=True,
        )
        return 1
    print(f"[xbox] Detected {n} joystick(s):", flush=True)
    for i in range(n):
        js = pygame.joystick.Joystick(i)
        try:
            js.init()
        except pygame.error:
            pass
        print(
            f"  [{i}] name={js.get_name()!r} "
            f"axes={js.get_numaxes()} "
            f"buttons={js.get_numbuttons()} "
            f"hats={js.get_numhats()}",
            flush=True,
        )
    js = pygame.joystick.Joystick(0)
    try:
        js.init()
    except pygame.error:
        pass
    # Map default constants -> friendly chord name so the delta print
    # below can suggest which chord the operator just hit.
    button_label: dict[int, str] = {
        BTN_A: "A", BTN_B: "B", BTN_X: "X", BTN_Y: "Y",
        BTN_LB: "LB(L1)", BTN_RB: "RB(R1)",
    }
    axis_label: dict[int, str] = {AXIS_LT: "LT(L2)", AXIS_RT: "RT(R2)"}
    print(
        "[xbox] Delta-only state stream for joystick 0 "
        "(Ctrl-C to exit). Press each chord button you plan to use; "
        "each press / release prints ONE line so you can verify "
        "the index matches the BTN_* / AXIS_* / HAT_DPAD constants.",
        flush=True,
    )
    try:
        n_btn = js.get_numbuttons()
        n_axis = js.get_numaxes()
        n_hat = js.get_numhats()
        # Treat small idle drift on the sticks (axes 0,1,3,4) as
        # "no change" so left-stick wobble doesn't spam the log.
        # Triggers (rest=-1.0) need a wider band; we threshold on
        # absolute change >= 0.5 to log only deliberate squeezes.
        AXIS_DELTA_THRESHOLD = 0.5
        prev_buttons = [js.get_button(i) for i in range(n_btn)]
        prev_hats = [js.get_hat(i) for i in range(n_hat)]
        # Quantize axes to the "press band" so floats don't spam.
        def _quantize_axis(v: float) -> int:
            if v >= AXIS_DELTA_THRESHOLD:
                return 1
            if v <= -AXIS_DELTA_THRESHOLD:
                return -1
            return 0
        prev_axes_q = [
            _quantize_axis(js.get_axis(i)) for i in range(n_axis)
        ]
        print(
            f"  start: buttons={prev_buttons} hats={prev_hats} "
            f"axes_q={prev_axes_q}",
            flush=True,
        )
        while True:
            pygame.event.pump()
            cur_buttons = [js.get_button(i) for i in range(n_btn)]
            cur_hats = [js.get_hat(i) for i in range(n_hat)]
            cur_axes_q = [
                _quantize_axis(js.get_axis(i)) for i in range(n_axis)
            ]
            # Buttons: log each press/release transition.
            for i in range(n_btn):
                if cur_buttons[i] != prev_buttons[i]:
                    label = button_label.get(i, "")
                    tag = f" ({label})" if label else ""
                    edge = "DOWN" if cur_buttons[i] else "up"
                    print(
                        f"  button[{i}]{tag:<8} {edge}",
                        flush=True,
                    )
            # Hats: log each transition (D-pad).
            for i in range(n_hat):
                if cur_hats[i] != prev_hats[i]:
                    hx, hy = cur_hats[i]
                    direction = {
                        (0, 0): "neutral",
                        (0, 1): "UP", (0, -1): "DOWN",
                        (-1, 0): "LEFT", (1, 0): "RIGHT",
                    }.get((hx, hy), f"diag({hx},{hy})")
                    note = (
                        " (HAT_DPAD)" if i == HAT_DPAD else ""
                    )
                    print(
                        f"  hat[{i}]{note} -> {direction} "
                        f"(value={cur_hats[i]})",
                        flush=True,
                    )
            # Axes: only log threshold-crossing edges (triggers go
            # -1 -> +1; sticks idle near 0 stay quiet).
            for i in range(n_axis):
                if cur_axes_q[i] != prev_axes_q[i]:
                    label = axis_label.get(i, "")
                    tag = f" ({label})" if label else ""
                    new_v = round(js.get_axis(i), 2)
                    band = (
                        "PRESSED" if cur_axes_q[i] > 0
                        else "released" if cur_axes_q[i] == 0
                        else "NEG"
                    )
                    print(
                        f"  axis[{i}]{tag:<10} {band:<8} "
                        f"raw={new_v:+.2f}",
                        flush=True,
                    )
            prev_buttons = cur_buttons
            prev_hats = cur_hats
            prev_axes_q = cur_axes_q
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[xbox] --list done", flush=True)
        return 0


def _calibration_snapshot(js: "pygame.joystick.Joystick", duration_s: float) -> None:
    """Print the raw axes/buttons/hats once at startup for ~duration_s
    so the operator can verify chord mapping before the demo. Quiet
    after the window expires -- the main loop's status line takes
    over.
    """
    print(
        f"[xbox] {duration_s:.0f}s calibration window. Press each "
        f"chord button you plan to use and verify it lights up in "
        f"the indices the BTN_* / AXIS_* constants expect.",
        flush=True,
    )
    end_ts = time.monotonic() + duration_s
    last_print = 0.0
    while time.monotonic() < end_ts:
        pygame.event.pump()
        now = time.monotonic()
        if now - last_print >= 0.25:
            axes = [round(js.get_axis(i), 2) for i in range(js.get_numaxes())]
            buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
            hats = [js.get_hat(i) for i in range(js.get_numhats())]
            print(
                f"  axes={axes} buttons={buttons} hats={hats}",
                flush=True,
            )
            last_print = now
        time.sleep(0.02)
    print("[xbox] calibration window done; entering main loop.", flush=True)


@dataclass
class _LoopState:
    """Mutable bookkeeping for the main loop."""

    prev: ChordState = field(default_factory=ChordState)
    busy_until_ts: float = 0.0
    busy_label: str = ""
    last_dispatch_ts: float = 0.0
    last_status_ts: float = 0.0
    last_estop_ts: float = -1e9


def _dispatch(
    *,
    chord_label: str,
    pkl_path: Path,
    kind: str,
    pub: WirePub,
    target_rate_hz: float,
    state: _LoopState,
) -> None:
    """Estimate the clip duration, send the play payload, and arm the
    single-flight gate. ``kind`` must be ``"gesture"`` or ``"locomotion"``.
    """
    clip_src, clip_key = _split_clip_ref(pkl_path)
    entry = MotionClipEntry(
        name=f"xbox:{pkl_path.name}",
        source=clip_src,
        motion_key=clip_key,
        kind=kind,  # type: ignore[arg-type]
    )
    try:
        duration_s = estimate_duration_s(entry, target_rate_hz=target_rate_hz)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(
            f"[xbox {_ts()}] DISPATCH FAIL {chord_label} -> "
            f"{pkl_path.name}: {exc}",
            flush=True,
            file=sys.stderr,
        )
        return

    payload: dict[str, Any] = {
        "action": "play",
        "pkl": str(pkl_path),
        "kind": kind,
    }
    pub.send(payload)
    now = time.monotonic()
    state.busy_until_ts = now + duration_s + BUSY_TAIL_BUFFER_S
    state.busy_label = pkl_path.name
    state.last_dispatch_ts = now
    print(
        f"[xbox {_ts()}] FIRE {kind:<10} {chord_label:<7} -> "
        f"{pkl_path.name} (~{duration_s:.1f}s; gate armed until "
        f"clip+buffer)",
        flush=True,
    )


def _rumble_busy(
    js: "pygame.joystick.Joystick", *, enabled: bool
) -> None:
    """Buzz the controller briefly when a chord is dropped because a
    clip is in flight. Silent no-op when ``enabled`` is False or the
    pad / driver doesn't support rumble (``Joystick.rumble`` returns
    False or raises). The rumble is fire-and-forget -- SDL queues it
    asynchronously and the main loop continues on the next tick.
    """
    if not enabled:
        return
    rumble = getattr(js, "rumble", None)
    if rumble is None:
        return
    try:
        rumble(RUMBLE_BUSY_LOW, RUMBLE_BUSY_HIGH, RUMBLE_BUSY_MS)
    except (pygame.error, OSError, ValueError):
        # Some pads / drivers accept rumble calls then fail silently.
        # Swallow so the launcher loop isn't taken down by a vibe
        # motor on a $40 third-party controller.
        pass


def _dispatch_estop(
    *,
    pub: WirePub,
    state: _LoopState,
    followup_pkl: Optional[Path],
    target_rate_hz: float,
) -> None:
    """Publish ``{"action": "stop"}`` to drop any in-flight clip, then
    optionally publish a follow-up gesture (e.g. shake_head) as a
    visible operator-facing acknowledgment. The follow-up arms the
    single-flight gate for its own duration so a stray chord during
    the acknowledgment doesn't trample it; a second e-stop chord
    bypasses the gate normally.

    ZMQ PUB guarantees in-order delivery on the same socket, so the
    ``stop`` lands at the recorder before the ``play`` -- the
    recorder's ``_drain_clip_commands`` processes them in order
    within a single tick.
    """
    pub.send({"action": "stop"})
    now = time.monotonic()
    state.last_dispatch_ts = now
    state.last_estop_ts = now

    if followup_pkl is None:
        # Pure stop -- recorder's DEFAULT_STAND_POSE_MUJOCO_RAD
        # idle-stand fallback takes over on the next tick.
        state.busy_until_ts = 0.0
        state.busy_label = ""
        print(
            f"[xbox {_ts()}] E-STOP L1+R1+L2+R2 -> stop "
            f"(gate cleared; re-arms after triggers release)",
            flush=True,
        )
        return

    # Followup play: estimate duration so we can arm the local gate,
    # then PUB the gesture payload. estimate_duration_s here doubles
    # as a pre-flight (a missing / broken follow-up PKL fails loud
    # rather than leaving the recorder mid-stop).
    followup_src, followup_key = _split_clip_ref(followup_pkl)
    entry = MotionClipEntry(
        name=f"xbox-estop:{followup_pkl.name}",
        source=followup_src,
        motion_key=followup_key,
        kind="gesture",
    )
    try:
        duration_s = estimate_duration_s(entry, target_rate_hz=target_rate_hz)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        # Fall back to pure stop on bad followup PKL -- the e-stop
        # primary objective (cancel in-flight clip) already
        # succeeded, so we don't want to crash the launcher loop.
        print(
            f"[xbox {_ts()}] E-STOP L1+R1+L2+R2 -> stop "
            f"(followup '{followup_pkl.name}' load failed: {exc}; "
            f"falling back to bare stop -> recorder idle stand)",
            flush=True,
            file=sys.stderr,
        )
        state.busy_until_ts = 0.0
        state.busy_label = ""
        return

    pub.send({
        "action": "play",
        "pkl": str(followup_pkl),
        "kind": "gesture",
    })
    state.busy_until_ts = now + duration_s + BUSY_TAIL_BUFFER_S
    state.busy_label = f"{followup_pkl.name} (estop ack)"
    print(
        f"[xbox {_ts()}] E-STOP L1+R1+L2+R2 -> stop + play "
        f"{followup_pkl.name} (~{duration_s:.1f}s acknowledgment; "
        f"then idle stand)",
        flush=True,
    )


def _maybe_print_status(state: _LoopState, *, has_pad: bool) -> None:
    now = time.monotonic()
    if now - state.last_status_ts < STATUS_PRINT_INTERVAL_S:
        return
    state.last_status_ts = now
    if not has_pad:
        print(f"[xbox {_ts()}] WAITING-FOR-PAD", flush=True)
        return
    if now < state.busy_until_ts:
        remaining = state.busy_until_ts - now
        print(
            f"[xbox {_ts()}] BUSY  {state.busy_label} "
            f"(t-{remaining:.1f}s)",
            flush=True,
        )
    else:
        print(f"[xbox {_ts()}] armed", flush=True)


def _open_joystick(
    index: int,
    *,
    wait_for_pad: bool = False,
    poll_interval_s: float = 1.0,
) -> "pygame.joystick.Joystick":
    """Open joystick ``index``. When ``wait_for_pad`` is true and no
    joystick is detected, poll once per ``poll_interval_s`` until one
    appears (Ctrl-C aborts). Useful for demo-day flaky cables where
    the operator might plug in the pad after launching the listener.
    """
    _ensure_pygame_subsystems()
    if pygame.joystick.get_count() == 0:
        if not wait_for_pad:
            raise SystemExit(
                "[xbox] No joysticks detected. Plug in the Xbox pad "
                "and re-run, or pass --wait-for-pad to have the "
                "listener poll until one appears."
            )
        print(
            f"[xbox] WAIT-FOR-PAD: no joystick detected. Polling "
            f"every {poll_interval_s:.1f}s (Ctrl-C to abort).",
            flush=True,
        )
        while pygame.joystick.get_count() == 0:
            time.sleep(poll_interval_s)
            # SDL caches the joystick list; re-init to pick up a new
            # plug-in. ``pygame.joystick.quit()`` then ``.init()`` is
            # the documented refresh pattern.
            pygame.joystick.quit()
            pygame.joystick.init()
        print("[xbox] WAIT-FOR-PAD: joystick appeared.", flush=True)
    n = pygame.joystick.get_count()
    if index < 0 or index >= n:
        raise SystemExit(
            f"[xbox] --joystick-index {index} out of range (count={n})"
        )
    js = pygame.joystick.Joystick(index)
    try:
        js.init()
    except pygame.error:
        pass
    print(
        f"[xbox] using joystick [{index}] name={js.get_name()!r} "
        f"axes={js.get_numaxes()} buttons={js.get_numbuttons()} "
        f"hats={js.get_numhats()}",
        flush=True,
    )
    return js


def _main_loop(
    *,
    js: "pygame.joystick.Joystick",
    pub: WirePub,
    loco: dict[str, Optional[Path]],
    gestures: dict[str, Optional[Path]],
    target_rate_hz: float,
    rumble_enabled: bool = True,
    estop_followup_pkl: Optional[Path] = None,
) -> int:
    state = _LoopState()
    print(
        "[xbox] listener live. Ctrl-C to exit (publishes a defensive "
        "stop on shutdown).",
        flush=True,
    )

    while True:
        pygame.event.pump()
        cur = read_chord_state(js)
        now = time.monotonic()

        # E-STOP -- always live, even while BUSY. Rising edge: chord
        # newly satisfied AND we're past the re-arm cooldown.
        if cur.estop_chord and not state.prev.estop_chord:
            if now - state.last_estop_ts >= ESTOP_REARM_COOLDOWN_S:
                _dispatch_estop(
                    pub=pub,
                    state=state,
                    followup_pkl=estop_followup_pkl,
                    target_rate_hz=target_rate_hz,
                )
                state.prev = cur
                _maybe_print_status(state, has_pad=True)
                time.sleep(TICK_DT)
                continue

        # Debounce: after any dispatch (including e-stop) we ignore
        # other chord edges for a short window. Required to keep a
        # finger-mash from triggering two clips back-to-back.
        if now - state.last_dispatch_ts < DEBOUNCE_AFTER_DISPATCH_S:
            state.prev = cur
            _maybe_print_status(state, has_pad=True)
            time.sleep(TICK_DT)
            continue

        # BUSY GATE -- single-flight enforcement. Locomotion + gesture
        # edges are dropped (with a log line) while a clip is in flight.
        busy = now < state.busy_until_ts

        # LOCOMOTION -- D-pad direction edge while L2+R2 deadman is
        # held and L1/R1 are released (the L1/R1-released check is
        # what distinguishes a locomotion chord from an e-stop activation).
        if (
            cur.loco_deadman
            and cur.dpad_dir is not None
            and cur.dpad_dir != state.prev.dpad_dir
        ):
            chord = cur.dpad_dir
            target = loco.get(chord)
            if target is None:
                print(
                    f"[xbox {_ts()}] WARN D-pad {chord} pressed but "
                    f"binding is empty (set BINDINGS_LOCOMOTION[{chord!r}])",
                    flush=True,
                )
            elif busy:
                remaining = state.busy_until_ts - now
                print(
                    f"[xbox {_ts()}] BUSY  t-{remaining:.1f}s -- "
                    f"ignored D-pad {chord} -> {target.name}",
                    flush=True,
                )
                _rumble_busy(js, enabled=rumble_enabled)
            else:
                _dispatch(
                    chord_label=f"D-pad-{chord}",
                    pkl_path=target,
                    kind="locomotion",
                    pub=pub,
                    target_rate_hz=target_rate_hz,
                    state=state,
                )

        # GESTURE -- face button edge while the modifier combo is
        # valid (see ChordState.gesture_modifier). Modifier suffix
        # already carries its own '+' prefix so we just concat.
        mod = cur.gesture_modifier
        if mod is not None:
            face_edges: list[tuple[str, bool, bool]] = [
                ("A", cur.a, state.prev.a),
                ("B", cur.b, state.prev.b),
                ("X", cur.x, state.prev.x),
                ("Y", cur.y, state.prev.y),
            ]
            for name, cur_v, prev_v in face_edges:
                if not (cur_v and not prev_v):
                    continue
                chord = f"{name}{mod}"  # e.g. "Y", "Y+L1", "Y+R2"
                target = gestures.get(chord)
                if target is None:
                    print(
                        f"[xbox {_ts()}] WARN {chord} pressed but "
                        f"binding is empty (set BINDINGS_GESTURES[{chord!r}])",
                        flush=True,
                    )
                    continue
                if busy:
                    remaining = state.busy_until_ts - now
                    print(
                        f"[xbox {_ts()}] BUSY  t-{remaining:.1f}s -- "
                        f"ignored {chord} -> {target.name}",
                        flush=True,
                    )
                    _rumble_busy(js, enabled=rumble_enabled)
                    continue
                _dispatch(
                    chord_label=chord,
                    pkl_path=target,
                    kind="gesture",
                    pub=pub,
                    target_rate_hz=target_rate_hz,
                    state=state,
                )
                break  # at most one face-button fire per tick

        state.prev = cur
        _maybe_print_status(state, has_pad=True)
        time.sleep(TICK_DT)


# ─── CLI ──────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Recorder host. Default 'localhost' matches the assumed "
             "run_x2_pkl_direct_stack / run_x2_quest3_planner_stack "
             "co-located setup.",
    )
    parser.add_argument(
        "--port", type=int, default=MOTION_CLIP_CMD_DEFAULT_PORT,
        help="motion_clip_cmd port. Must match the recorder's "
             "--motion-clip-cmd-port.",
    )
    parser.add_argument(
        "--topic", default=MOTION_CLIP_CMD_DEFAULT_TOPIC,
        help="motion_clip_cmd topic.",
    )
    parser.add_argument(
        "--joystick-index", type=int, default=0,
        help="Index of the joystick to listen on (default 0). "
             "Use --list to enumerate.",
    )
    parser.add_argument(
        "--target-rate-hz", type=float, default=50.0,
        help="Recorder publish rate. Used to estimate clip durations "
             "for the single-flight gate. Match the rate the "
             "recorder is running at (default 50 Hz in both stack "
             "wrappers).",
    )
    parser.add_argument(
        "--linger-ms", type=int, default=200,
        help="ZMQ PUB slow-joiner sleep after connect and linger on "
             "close. Same default as play_locomotion.py.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Enumerate joysticks, print live raw axes/buttons/hats "
             "for joystick 0, and exit. No ZMQ traffic.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log every chord fire but never PUB-send. Use to verify "
             "the binding map without the SONIC stack up.",
    )
    parser.add_argument(
        "--calibration-secs", type=float, default=2.0,
        help="Length of the on-startup raw-state print window for "
             "verifying chord mappings before the loop arms. Pass 0 "
             "to skip.",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip the PKL existence check (useful for --dry-run "
             "when paths are intentionally stubbed).",
    )
    parser.add_argument(
        "--wait-for-pad", action="store_true",
        help="If no joystick is detected at startup, poll once per "
             "second until one is plugged in instead of exiting. "
             "Handy when wrestling with flaky USB cables on demo day.",
    )
    parser.add_argument(
        "--no-rumble", action="store_true",
        help="Disable haptic feedback. By default the controller "
             "buzzes briefly when a chord is rejected because a clip "
             "is already in flight (BUSY gate).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list:
        # We don't need pygame.display; explicitly init only joystick.
        return _list_and_exit()

    loco = _resolve_paths(BINDINGS_LOCOMOTION)
    gestures = _resolve_paths(BINDINGS_GESTURES)
    estop_followup = _resolve_single_path(ESTOP_FOLLOWUP_PKL)
    _print_bindings(loco, gestures, estop_followup)
    if not args.skip_preflight:
        loco, gestures, estop_followup = _preflight_paths(
            loco, gestures, estop_followup
        )

    js = _open_joystick(
        args.joystick_index, wait_for_pad=args.wait_for_pad
    )
    if args.calibration_secs > 0:
        _calibration_snapshot(js, args.calibration_secs)

    pub = WirePub(
        host=args.host,
        port=args.port,
        topic=args.topic,
        linger_ms=args.linger_ms,
        dry_run=args.dry_run,
    )
    pub.connect()

    # Defensive: send a single stop on Ctrl-C so a half-fired chord
    # doesn't leave the recorder mid-clip when the launcher dies.
    exit_code = 0
    try:
        exit_code = _main_loop(
            js=js,
            pub=pub,
            loco=loco,
            gestures=gestures,
            target_rate_hz=float(args.target_rate_hz),
            rumble_enabled=not args.no_rumble,
            estop_followup_pkl=estop_followup,
        )
    except KeyboardInterrupt:
        print(
            f"\n[xbox {_ts()}] SIGINT -- sending defensive stop "
            f"and exiting",
            flush=True,
        )
        if not args.dry_run:
            try:
                pub.send({"action": "stop"})
                time.sleep(max(0.0, args.linger_ms / 1000.0))
            except Exception:
                pass
        exit_code = 130
    finally:
        pub.close()
        try:
            pygame.joystick.quit()
        except Exception:
            pass

    return exit_code


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
