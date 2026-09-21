"""Pico/XRoboToolkit stream tape: record live xrt samples, replay headset-free.

Two halves:

* **Recorder** (``python -m gear_sonic.utils.teleop.pico_tape record ...``):
  polls the live ``xrobotoolkit_sdk`` alongside whatever else is running
  (the PC Service serves multiple SDK clients) and writes an ``.npz`` tape
  of everything the X2 teleop path consumes: device timestamp, 24x7 body
  skeleton, headset/controller poses, sticks, buttons, triggers, clicks.

* **TapeXrt** — a drop-in stand-in for the ``xrobotoolkit_sdk`` module that
  replays a tape with original pacing (device-timestamp deltas). Activated
  in ``pico_x2_reader`` by setting ``PICO_TAPE=/path/to/tape.npz`` in the
  environment, so the ENTIRE stack (manager -> kplanner -> deploy) runs
  identically with zero headset — the offline harness for arm-retarget
  A/Bs and regression tests.

Typical loop:
    # record 60 s while the operator walks around (live headset):
    .venv_teleop/bin/python -m gear_sonic.utils.teleop.pico_tape record \
        --out logs/pico_tapes/walkaround_001.npz --seconds 60

    # replay through the full stack, no headset:
    PICO_TAPE=logs/pico_tapes/walkaround_001.npz INPUT_SOURCE=pico \
        ./gear_sonic/scripts/sim_onnx_planner.sh --vr-only ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

_FIELDS = ("stamp_ns", "body", "body_ok", "headset", "lctrl", "rctrl",
           "axes", "buttons", "analog")
# buttons: [A, B, X, Y, l_click, r_click]  (bools as uint8)
# analog:  [l_trigger, r_trigger, l_grip, r_grip]
# axes:    [lx, ly, rx, ry]


def record(out: Path, seconds: float, poll_hz: float = 100.0,
           xrt_module=None) -> int:
    """``xrt_module``: pass an ALREADY-INITIALIZED sdk module/instance to
    skip init — calling xrt.init() twice in one process aborts the SDK
    (learned 2026-08-15: the self-arming wrapper crashed exactly there)."""
    if xrt_module is not None:
        xrt = xrt_module
    else:
        import xrobotoolkit_sdk as xrt

        xrt.init()
    rows: dict[str, list] = {k: [] for k in _FIELDS}
    t_end = time.monotonic() + seconds
    last_stamp = None
    print(f"[pico-tape] recording {seconds:.0f}s -> {out}")
    while time.monotonic() < t_end:
        stamp = int(xrt.get_time_stamp_ns())
        if stamp == 0 or stamp == last_stamp:
            time.sleep(1.0 / poll_hz)
            continue
        last_stamp = stamp
        body_ok = bool(xrt.is_body_data_available())
        body = (np.asarray(xrt.get_body_joints_pose(), dtype=np.float32)
                if body_ok else np.zeros((24, 7), np.float32))
        lx, ly = xrt.get_left_axis()
        rx, ry = xrt.get_right_axis()
        rows["stamp_ns"].append(stamp)
        rows["body"].append(body)
        rows["body_ok"].append(body_ok)
        rows["headset"].append(np.asarray(xrt.get_headset_pose(), np.float32))
        rows["lctrl"].append(np.asarray(xrt.get_left_controller_pose(), np.float32))
        rows["rctrl"].append(np.asarray(xrt.get_right_controller_pose(), np.float32))
        rows["axes"].append(np.asarray([lx, ly, rx, ry], np.float32))
        rows["buttons"].append(np.asarray(
            [xrt.get_A_button(), xrt.get_B_button(), xrt.get_X_button(),
             xrt.get_Y_button(), xrt.get_left_axis_click(),
             xrt.get_right_axis_click()], np.uint8))
        rows["analog"].append(np.asarray(
            [xrt.get_left_trigger(), xrt.get_right_trigger(),
             xrt.get_left_grip(), xrt.get_right_grip()], np.float32))
        n = len(rows["stamp_ns"])
        if n % 250 == 0:
            print(f"[pico-tape] {n} samples, body_ok={body_ok}", flush=True)
        time.sleep(1.0 / poll_hz)

    n = len(rows["stamp_ns"])
    if n == 0:
        print("[pico-tape] NO samples captured (stream quiet?) — not writing")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **{k: np.asarray(v) for k, v in rows.items()})
    dur = (rows["stamp_ns"][-1] - rows["stamp_ns"][0]) * 1e-9
    print(f"[pico-tape] wrote {n} samples ({dur:.1f}s device time, "
          f"{sum(rows['body_ok'])} with body) -> {out}")
    return 0


class TapeXrt:
    """xrobotoolkit_sdk stand-in replaying a recorded tape.

    Same call surface as the pybind module (the subset PicoX2Reader and the
    recorder use). Pacing: sample i becomes current once wall time since
    ``init()`` exceeds its device-timestamp offset; ``loop=True`` (default)
    wraps around with a continuous stamp so downstream FPS math stays sane.
    """

    def __init__(self, tape_path: str | Path, loop: bool = True):
        d = np.load(Path(tape_path).expanduser())
        self._d = {k: d[k] for k in _FIELDS}
        # Session metadata beyond the frame fields (full-session tapes from
        # pico_intent_sender: full_session, mode0, t0_wall_unix_s, ...).
        self.meta = {k: (d[k].item() if d[k].shape == () else d[k]) for k in d.files if k not in _FIELDS}
        self._n = len(self._d["stamp_ns"])
        if self._n == 0:
            raise ValueError(f"empty tape: {tape_path}")
        self._offsets = (self._d["stamp_ns"] - self._d["stamp_ns"][0]) * 1e-9
        self._span = float(self._offsets[-1]) + 1e-3
        self._loop = loop
        self._t0 = None

    # -- module-surface ------------------------------------------------------
    def init(self):
        self._t0 = time.monotonic()

    def _idx(self) -> tuple[int, int]:
        """(sample index, completed loops)."""
        if self._t0 is None:
            self.init()
        el = time.monotonic() - self._t0
        loops, into = (int(el // self._span), el % self._span) if self._loop \
            else (0, min(el, self._span))
        i = int(np.searchsorted(self._offsets, into, side="right") - 1)
        return max(i, 0), loops

    def finished(self) -> bool:
        """True once a non-looping tape has played past its last sample
        (robot replays: the sender releases whole-body here instead of the
        loop-wrap snapping the human pose back to frame 0)."""
        if self._loop or self._t0 is None:
            return False
        return (time.monotonic() - self._t0) >= self._span

    def get_time_stamp_ns(self) -> int:
        i, loops = self._idx()
        base = int(self._d["stamp_ns"][0])
        return int(self._d["stamp_ns"][i]) + int(loops * self._span * 1e9)

    def is_body_data_available(self) -> bool:
        return bool(self._d["body_ok"][self._idx()[0]])

    def get_body_joints_pose(self):
        return self._d["body"][self._idx()[0]]

    def get_headset_pose(self):
        return self._d["headset"][self._idx()[0]]

    def get_left_controller_pose(self):
        return self._d["lctrl"][self._idx()[0]]

    def get_right_controller_pose(self):
        return self._d["rctrl"][self._idx()[0]]

    # Controls go NEUTRAL once a non-looping tape has finished (sticks 0,
    # buttons up, triggers/grips 0): the last sample must not keep a
    # deadman or a chord "held" on the robot forever. Body/poses hold.
    def get_left_axis(self):
        if self.finished():
            return [0.0, 0.0]
        return self._d["axes"][self._idx()[0]][:2].tolist()

    def get_right_axis(self):
        if self.finished():
            return [0.0, 0.0]
        return self._d["axes"][self._idx()[0]][2:].tolist()

    def _btn(self, j: int) -> bool:
        if self.finished():
            return False
        return bool(self._d["buttons"][self._idx()[0]][j])

    def get_A_button(self):
        return self._btn(0)

    def get_B_button(self):
        return self._btn(1)

    def get_X_button(self):
        return self._btn(2)

    def get_Y_button(self):
        return self._btn(3)

    def get_left_axis_click(self):
        return self._btn(4)

    def get_right_axis_click(self):
        return self._btn(5)

    def _ana(self, j: int) -> float:
        if self.finished():
            return 0.0
        return float(self._d["analog"][self._idx()[0]][j])

    def get_left_trigger(self):
        return self._ana(0)

    def get_right_trigger(self):
        return self._ana(1)

    def get_left_grip(self):
        return self._ana(2)

    def get_right_grip(self):
        return self._ana(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="record a live xrt tape")
    rec.add_argument("--out", required=True, type=Path)
    rec.add_argument("--seconds", type=float, default=60.0)
    rec.add_argument("--poll-hz", type=float, default=100.0)
    info = sub.add_parser("info", help="print tape stats")
    info.add_argument("tape", type=Path)
    args = ap.parse_args()
    if args.cmd == "record":
        return record(args.out, args.seconds, args.poll_hz)
    d = np.load(args.tape)
    n = len(d["stamp_ns"])
    dur = (d["stamp_ns"][-1] - d["stamp_ns"][0]) * 1e-9 if n else 0.0
    print(f"{args.tape}: {n} samples, {dur:.1f}s, "
          f"body_ok {int(d['body_ok'].sum())}/{n}, "
          f"buttons pressed {int(d['buttons'].any(axis=1).sum())} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
