"""Pico (XRoboToolkit) input reader for the X2 teleop manager.

Drop-in replacement for :class:`~gear_sonic.utils.teleop.vr.quest3_reader.
Quest3Reader` on the X2 path: it exposes the same accessor surface
``quest3_manager_x2`` consumes, but sources every signal from the
``xrobotoolkit_sdk`` (``xrt``) pybind module fed by the XRoboToolkit PC
Service instead of the Quest 3 WebXR/WebSocket app. No server sockets are
opened; the PC Service daemon owns the headset link and ``xrt`` reads
shared state in-process.

Signal mapping vs Quest 3:

- ``get_3pt_pose``: the Pico's 24-joint body-tracking skeleton (headset +
  2 wrist controllers + 2 ankle motion trackers) is reduced to the same
  root-relative (3, 7) ``[left_wrist, right_wrist, neck]`` contract via
  :func:`gear_sonic.utils.teleop.pico_3pt._process_3pt_pose` — identical
  math to the G1 pico manager. Root is the tracked pelvis (Quest 3 uses
  the headset floor projection); the operator calibration flow absorbs
  the residual offsets, so run a fresh calibration per device.
- ``get_buttons`` / ``get_controller_axes`` / ``get_controller_inputs`` /
  ``get_stick_clicks``: native controller state via ``xrt`` getters. Axis
  sign conventions differ per runtime — use the manager's existing
  ``--invert-lx/--invert-ly/--invert-rx/--invert-ry`` flags if a stick
  feels reversed rather than patching here.
- ``get_hand_curls``: always ``(None, None, "controller", "controller")``
  — the XRoboToolkit path carries no XRHand curls, which is exactly the
  Quest 3 controllers-only mode, so the downstream trigger/grip fallback
  drives the OmniHand unchanged.
- ``send_message``: no WebXR client exists to receive audio prompts or
  calibration payloads; logged at DEBUG and dropped.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from gear_sonic.utils.teleop.pico_3pt import _process_3pt_pose

import os as _os

if _os.environ.get("PICO_TAPE"):
    # Tape replay: drive the entire stack from a recorded xrt stream, no
    # headset needed (regression tests, arm-retarget A/Bs). Record tapes
    # with ``python -m gear_sonic.utils.teleop.pico_tape record``.
    from gear_sonic.utils.teleop.pico_tape import TapeXrt

    xrt = TapeXrt(_os.environ["PICO_TAPE"])
else:
    try:  # SDK only present in .venv_teleop (install_scripts/install_pico.sh)
        import xrobotoolkit_sdk as xrt
    except ImportError:  # pragma: no cover - tests inject a fake module
        xrt = None

logger = logging.getLogger(__name__)


class PicoX2Reader:
    """Threaded ``xrt`` poller exposing the Quest3Reader accessor surface."""

    # Flag the link dead when the device timestamp stops advancing for this
    # long (headset asleep, PC Service down, wifi drop). Mirrors
    # input_readers.PicoReader.STALE_TIMEOUT.
    STALE_TIMEOUT_S = 5.0

    def __init__(self, poll_hz: float = 200.0):
        if xrt is None:
            raise ImportError(
                "xrobotoolkit_sdk is not importable. Run "
                "install_scripts/install_pico.sh and launch from .venv_teleop, "
                "and make sure the XRoboToolkit PC Service is installed."
            )
        xrt.init()
        self._poll_period_s = 1.0 / max(poll_hz, 1.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="pico-x2-reader"
        )
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._fps_ema = 0.0
        self._last_stamp_ns: int | None = None
        self._last_new_data_mono: float | None = None
        # Gamepad-health bookkeeping (see get_gamepad_health): monotonic time
        # of the most recent fresh->stale transition, None if never stale.
        self._last_sources_zero_t: float | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # -- polling thread -----------------------------------------------------

    def _poll_once(self) -> dict[str, Any] | None:
        """One synchronous sweep of the xrt state. Returns a sample dict or
        None when the device timestamp has not advanced since last sweep."""
        stamp_ns = int(xrt.get_time_stamp_ns())
        if self._last_stamp_ns is not None and stamp_ns == self._last_stamp_ns:
            return None

        if self._last_stamp_ns is not None:
            dt = (stamp_ns - self._last_stamp_ns) * 1e-9
            if dt > 0.0:
                inst = 1.0 / dt
                self._fps_ema = (
                    inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst
                )
        self._last_stamp_ns = stamp_ns

        body_24x7 = None
        if xrt.is_body_data_available():
            try:
                body = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64)
                if body.shape == (24, 7):
                    body_24x7 = body
            except Exception:
                logger.exception("[PicoX2Reader] body pose read error")

        def _axis_pair(getter) -> tuple[float, float]:
            try:
                ax = getter()
            except Exception:
                return 0.0, 0.0
            try:
                return float(ax[0]), float(ax[1])
            except (TypeError, IndexError, ValueError):
                return 0.0, 0.0

        lx, ly = _axis_pair(xrt.get_left_axis)
        rx, ry = _axis_pair(xrt.get_right_axis)

        sample: dict[str, Any] = {
            "body_poses_np": body_24x7,
            "buttons": {
                "a": bool(xrt.get_A_button()),
                "b": bool(xrt.get_B_button()),
                "x": bool(xrt.get_X_button()),
                "y": bool(xrt.get_Y_button()),
                "leftTrigger": float(xrt.get_left_trigger()),
                "rightTrigger": float(xrt.get_right_trigger()),
                "leftGrip": float(xrt.get_left_grip()),
                "rightGrip": float(xrt.get_right_grip()),
                "leftStickClick": bool(xrt.get_left_axis_click()),
                "rightStickClick": bool(xrt.get_right_axis_click()),
            },
            "axes": {"lx": lx, "ly": ly, "rx": rx, "ry": ry},
            "timestamp_ns": stamp_ns,
            "timestamp_monotonic": time.monotonic(),
            "fps": self._fps_ema,
        }
        return sample

    def _run(self) -> None:
        last_report = time.monotonic()
        while not self._stop.is_set():
            try:
                sample = self._poll_once()
            except Exception:
                logger.exception("[PicoX2Reader] xrt poll error")
                sample = None
            now = time.monotonic()
            if sample is not None:
                if self._link_stale(now):
                    logger.info("[PicoX2Reader] fresh data, link restored")
                self._last_new_data_mono = now
                with self._lock:
                    self._latest = sample
            else:
                was_stale = self._link_stale(now - self._poll_period_s)
                if self._link_stale(now) and not was_stale:
                    self._last_sources_zero_t = now
                    logger.warning(
                        "[PicoX2Reader] no new xrt data for %.1fs — headset "
                        "asleep / PC Service down?",
                        self.STALE_TIMEOUT_S,
                    )
            if now - last_report >= 5.0:
                logger.info(
                    "[PicoX2Reader] fps=%.1f stale=%s body=%s",
                    self._fps_ema,
                    self._link_stale(now),
                    self.get_3pt_pose() is not None,
                )
                last_report = now
            time.sleep(self._poll_period_s)

    def _link_stale(self, now: float) -> bool:
        if self._last_new_data_mono is None:
            return True
        return (now - self._last_new_data_mono) > self.STALE_TIMEOUT_S

    # -- Quest3Reader-mirroring accessors ------------------------------------

    def get_latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest

    def disconnected(self) -> bool:
        return self._link_stale(time.monotonic())

    def get_timestamp_ns(self) -> int:
        sample = self.get_latest()
        return 0 if sample is None else int(sample["timestamp_ns"])

    def get_last_message_age_s(self) -> float:
        sample = self.get_latest()
        if sample is None:
            return float("inf")
        return time.monotonic() - float(sample["timestamp_monotonic"])

    def get_3pt_pose(self) -> np.ndarray | None:
        sample = self.get_latest()
        if sample is None or sample.get("body_poses_np") is None:
            return None
        return _process_3pt_pose(sample["body_poses_np"])

    def get_body_pose_24x7(self) -> np.ndarray | None:
        """Raw Pico body-tracking skeleton: (24, 7) rows of
        ``[x, y, z, qx, qy, qz, qw]`` in the Unity frame, exactly as xrt
        delivers it. This is the SMPL path — the full-fidelity human pose
        (headset + wrist controllers + ankle trackers) that rides alongside
        the reduced robot-frame 3pt path, for full-body retargeting (GMR)
        and SMPL-conditioned tracking. The manager republishes it on the
        recorder wire as the ``smpl_body`` topic."""
        sample = self.get_latest()
        if sample is None:
            return None
        body = sample.get("body_poses_np")
        return None if body is None else np.array(body, copy=True)

    def get_buttons(self) -> tuple[bool, bool, bool, bool]:
        sample = self.get_latest()
        if sample is None:
            return False, False, False, False
        b = sample["buttons"]
        return b["a"], b["b"], b["x"], b["y"]

    def get_controller_inputs(self) -> tuple[float, float, float, float]:
        """Returns (left_trigger, right_trigger, left_grip, right_grip)."""
        sample = self.get_latest()
        if sample is None:
            return 0.0, 0.0, 0.0, 0.0
        b = sample["buttons"]
        return b["leftTrigger"], b["rightTrigger"], b["leftGrip"], b["rightGrip"]

    def get_controller_axes(self) -> tuple[float, float, float, float]:
        """Returns (lx, ly, rx, ry); positive ly = forward. If the Pico
        runtime's stick sign disagrees, fix it with the manager's
        --invert-* flags, not here."""
        sample = self.get_latest()
        if sample is None:
            return 0.0, 0.0, 0.0, 0.0
        a = sample["axes"]
        return a["lx"], a["ly"], a["rx"], a["ry"]

    def get_stick_clicks(self) -> tuple[bool, bool]:
        sample = self.get_latest()
        if sample is None:
            return False, False
        b = sample["buttons"]
        return b["leftStickClick"], b["rightStickClick"]

    def get_hand_curls(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None, str | None]:
        """No XRHand data on this path: report controller-sourced hands with
        no per-finger curls, matching Quest 3's controllers-only mode so the
        trigger/grip fallback path drives the fingers."""
        if self.get_latest() is None:
            return None, None, None, None
        return None, None, "controller", "controller"

    def get_thumb_opposition(self) -> tuple[float | None, float | None]:
        return None, None

    def get_finger_tip_oppose(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return None, None

    def get_gamepad_health(self) -> tuple[int, float]:
        """(sources_count, seconds since sources last dropped to zero).

        Mirrors Quest3Reader semantics: count < 2 or a recent zero-drop means
        button/trigger input — INCLUDING the VR e-stop — is unreliable. Here
        the whole xrt link is one source-pair, so count is 2 when fresh and
        0 when the device timestamp has stalled past STALE_TIMEOUT_S. Before
        any data has arrived, report (-1, inf) like Quest3Reader does so the
        manager doesn't scream about a dead e-stop while the operator is
        still putting the headset on."""
        if self._last_new_data_mono is None:
            return -1, float("inf")
        stale = self._link_stale(time.monotonic())
        count = 0 if stale else 2
        zero_t = self._last_sources_zero_t
        age = float("inf") if zero_t is None else time.monotonic() - zero_t
        return count, age

    def send_message(self, payload: Any) -> None:
        """Quest 3 uses this to push audio prompts / calibration messages to
        the WebXR client; the Pico path has no such channel."""
        logger.debug("[PicoX2Reader] send_message dropped (no client): %r", payload)
