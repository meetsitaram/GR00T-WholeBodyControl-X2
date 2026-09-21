"""Cycle the deploy MuJoCo viewer's fixed cameras via xdotool.

Why this exists
---------------

The deploy's MuJoCo viewer (a Python ``mujoco.viewer.launch_passive``
window) cycles to the next fixed camera on the ``]`` key (and the
previous on ``[``); ``Esc`` returns to the free camera. The operator
wears the Quest 3 headset and can't reach the workstation keyboard,
so we let them tap the LEFT thumbstick click (in
:mod:`quest3_manager_x2`) and synthesise a ``]`` keypress targeted
at the viewer window via ``xdotool``.

We initially used ``Tab`` because the cheatsheet on hand at the time
said "Tab cycles cameras", but ``Tab`` actually toggles the viewer's
left UI panel and does not change the active camera at all. The
correct keysym is ``bracketright`` -- verified live on
mujoco==3.5.0's ``launch_passive`` and consistent with the official
docs (https://mujoco.readthedocs.io/en/2.3.6/programming/visualization.html
and the ``mjpython`` viewer source).

This is intentionally an MVP. The "right" architecture is documented
in the ``TODO`` comment on :class:`ViewerCameraCycler` -- a single
``vr_input`` ZMQ topic publishes the full controller state, the
deploy subscribes, and updates ``mjvCamera`` in-process without any
X11 round-trip. That removes the X11 / Wayland / multi-monitor /
window-name-collision footguns of this hack.

Until that lands, this module:

* Locates the viewer window once on first use (cached for the
  lifetime of the process; window IDs are stable as long as the
  viewer isn't restarted).
* Sends a single ``]`` keystroke per :meth:`cycle` call, with a
  short cooldown so a noisy thumbstick doesn't fire ten cycles and
  blow past the camera the operator wanted.
* Logs cleanly on every failure mode (missing ``xdotool``, no
  ``DISPLAY``, no MuJoCo window found) so the operator can diagnose
  without a debugger.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class ViewerCameraCycler:
    """Send ``]`` to the deploy MuJoCo viewer to cycle its fixed cameras.

    Thread-safe (all public methods take a lock); cheap to instantiate
    even when xdotool / DISPLAY are unavailable -- the failure surfaces
    on the first :meth:`cycle` call as a one-shot WARN log line. The
    manager constructs one of these unconditionally and binds it to a
    Quest 3 button; runtime conditions decide whether the keystroke
    actually goes anywhere.

    The configurable :attr:`CYCLE_KEYSYM` is the X11 keysym name xdotool
    expects (``bracketright`` for ``]``); change it to ``bracketleft``
    if you'd rather cycle backwards through cameras, or to ``Escape``
    to snap back to the free camera. ``Tab`` is NOT a valid choice --
    in mujoco's passive viewer Tab toggles the left UI panel and does
    not change cameras at all (we wasted some time on that one in v7.1
    -- preserved here as a regression breadcrumb).

    TODO(unified-vr-input-topic): Replace this entire xdotool path with
    a proper ZMQ ``vr_input`` topic published by the manager. Any
    process (deploy viewer, recorder, future logging tools) can then
    subscribe and react to controller events in-process, without the
    X11-forwarding hack. Tracked as a follow-up after the Phase 0 MVP
    settles. See run_x2_quest3_planner_stack.sh banner / cheatsheet
    for the user-facing rationale.
    """

    # X11 keysym name passed to ``xdotool key``. ``bracketright`` is the
    # X server's name for ``]`` -- the next-fixed-camera key in
    # mujoco.viewer.launch_passive (verified on mujoco==3.5.0; matches
    # the upstream visualization docs). Use ``bracketleft`` to cycle
    # backwards if your demo setup wants that direction. NEVER set this
    # to ``Tab`` -- Tab toggles the viewer's left UI panel and does
    # NOT change cameras (this was the v7.1 bug fixed at deploy time).
    CYCLE_KEYSYM: str = "bracketright"

    # How long to wait after a successful cycle before honouring the
    # next press. Stick clicks tend to come in pairs because the
    # gamepad polling loop sees the press for a couple of frames; a
    # 250 ms cooldown collapses those into one Tab without making the
    # operator wait a noticeable beat.
    DEFAULT_COOLDOWN_S: float = 0.25

    # Re-emit each one-shot failure WARN every ``WARN_REEMIT_PERIOD_S``
    # seconds while the failure persists. Without this, a long session
    # that started with the deploy viewer down (e.g. the operator hit
    # the click button before the deploy was up) goes silent forever
    # afterwards even if the situation eventually changes. With
    # periodic re-emission the operator gets a fresh log line every
    # ~30 s telling them exactly which precondition is still missing,
    # so they don't have to read the source to understand why nothing
    # happens. Set to 0 to fully suppress re-emission (one-shot only).
    WARN_REEMIT_PERIOD_S: float = 30.0

    # Window classes we always exclude from the candidate list. GNOME's
    # mutter compositor wraps every X11 client in a ``mutter-x11-frames``
    # window that ALSO carries the client's title (e.g. "MuJoCo : x2t2.5")
    # and ALSO matches ``xdotool search --name MuJoCo``. Tab synthesised
    # at that wrapper is silently dropped on the floor (the wrapper is
    # decorative -- it has no GLFW event loop). Excluding it removes
    # ~100% of the "the manager logs cycled but the camera doesn't move"
    # symptom on GNOME-Shell desktops.
    EXCLUDED_WINDOW_CLASSES: tuple[str, ...] = ("mutter-x11-frames",)

    def __init__(
        self,
        *,
        window_search_pattern: str = "MuJoCo",
        window_class_name: str = "MuJoCo",
        cooldown_s: float = DEFAULT_COOLDOWN_S,
    ) -> None:
        self._pattern = window_search_pattern
        self._classname = window_class_name
        self._cooldown_s = float(cooldown_s)
        self._lock = threading.Lock()
        self._cached_wid: Optional[str] = None
        self._last_cycle_t: float = 0.0
        # Per-failure-mode timestamps of the last warning we emitted
        # for that mode. ``cycle()`` re-emits the WARN whenever
        # (now - timestamp) >= WARN_REEMIT_PERIOD_S so a long session
        # that started in a bad state (deploy viewer down at first
        # press, xdotool not installed, etc.) gets a periodic
        # reminder instead of going permanently silent after the
        # first warn. ``0.0`` means "never warned"; the first press
        # always logs.
        self._last_warn_t_no_xdotool: float = 0.0
        self._last_warn_t_no_display: float = 0.0
        self._last_warn_t_no_window: float = 0.0

    def cycle(self) -> bool:
        """Send a single Tab keystroke to the deploy viewer.

        Returns True if the keystroke was dispatched (xdotool exited 0),
        False otherwise. Never raises -- camera cycling is decorative
        and must not crash the manager control loop.
        """
        with self._lock:
            now = time.monotonic()
            if (now - self._last_cycle_t) < self._cooldown_s:
                return False

            if shutil.which("xdotool") is None:
                if self._should_reemit(self._last_warn_t_no_xdotool, now):
                    log.warning(
                        "[viewer-cycler] xdotool not installed; install "
                        "with `sudo apt install xdotool` to enable "
                        "left-stick-click camera cycling. Press is a "
                        "no-op until then."
                    )
                    self._last_warn_t_no_xdotool = now
                return False

            if not os.environ.get("DISPLAY"):
                if self._should_reemit(self._last_warn_t_no_display, now):
                    log.warning(
                        "[viewer-cycler] no DISPLAY env var; this is "
                        "expected on headless runs (--no-sim-viewer). "
                        "Camera cycling will be a no-op."
                    )
                    self._last_warn_t_no_display = now
                return False

            wid = self._cached_wid or self._locate_window()
            if wid is None:
                if self._should_reemit(self._last_warn_t_no_window, now):
                    log.warning(
                        "[viewer-cycler] no window matching classname=%r "
                        "or name=%r found via xdotool. Is the deploy "
                        "MuJoCo viewer up? Re-run the planner stack "
                        "with viewer enabled (default) and click the "
                        "deploy window once. (DISPLAY=%s)",
                        self._classname,
                        self._pattern,
                        os.environ.get("DISPLAY", "(unset)"),
                    )
                    self._last_warn_t_no_window = now
                return False
            self._cached_wid = wid

            ok = self._send_camera_cycle_key(wid)
            if ok:
                self._last_cycle_t = now
            return ok

    def reset(self) -> None:
        """Clear the cached window ID so the next :meth:`cycle` re-searches.

        Call this if the deploy is restarted mid-session (the new
        viewer process gets a new X11 window ID; the cached one is
        stale and any Tab sent to it lands in the void).
        """
        with self._lock:
            self._cached_wid = None
            self._last_warn_t_no_window = 0.0

    def _should_reemit(self, last_warn_t: float, now: float) -> bool:
        """True if we should log this WARN now.

        Returns True on the first emission (``last_warn_t == 0.0``) and
        every ``WARN_REEMIT_PERIOD_S`` seconds thereafter while the
        failure persists. Returns False if re-emission is disabled
        (period <= 0) and we've already warned once.
        """
        if last_warn_t == 0.0:
            return True
        if self.WARN_REEMIT_PERIOD_S <= 0.0:
            return False
        return (now - last_warn_t) >= self.WARN_REEMIT_PERIOD_S

    # -- internals ---------------------------------------------------

    def _locate_window(self) -> Optional[str]:
        """Run ``xdotool search`` and return the most-likely window ID.

        Search strategy (most-precise first):

        1. ``xdotool search --classname <classname>``. WM_CLASS is set
           by GLFW directly on the application window and is *not*
           inherited by the GNOME / mutter compositor's frame wrapper,
           so this filter zeroes in on the real GLFW window without
           any post-filtering. This is the path that fixes the
           common "manager logs cycled but the camera doesn't move on
           GNOME-Shell" footgun (the wrapper window matches the title
           filter and silently swallows the synthetic Tab).

        2. ``xdotool search --name <pattern>``. Title-substring fallback
           used when the classname search returns nothing -- some
           builds of MuJoCo / GLFW set WM_CLASS to something other
           than the default ("python3", a custom value via
           ``glfwWindowHintString``, etc.). We post-filter the title
           hits to drop anything whose WM_CLASS sits in
           :attr:`EXCLUDED_WINDOW_CLASSES` (currently
           ``mutter-x11-frames``), so the wrapper window can never
           sneak in even if the operator narrowed the pattern.

        If either search returns multiple candidates after filtering
        we log them all and pick the first; the operator can always
        narrow further with ``--viewer-window-pattern`` /
        ``--viewer-window-classname`` on the manager CLI.
        """
        # 1) classname search — the precise path
        wids = self._xdotool_search(["--classname", self._classname])
        wids = self._drop_excluded_classes(wids)
        if wids:
            self._maybe_log_multiple(wids, source=f"--classname {self._classname!r}")
            return wids[0]

        # 2) name-substring fallback — the legacy path
        wids = self._xdotool_search(["--name", self._pattern])
        wids = self._drop_excluded_classes(wids)
        if not wids:
            return None
        self._maybe_log_multiple(wids, source=f"--name {self._pattern!r}")
        return wids[0]

    def _xdotool_search(self, search_args: list[str]) -> list[str]:
        """Run a single ``xdotool search ...`` and return the WIDs.

        Returns ``[]`` on failure / timeout / non-zero exit. Logs once
        on actual subprocess failure (timeout / OSError) but stays
        quiet on "no match" because the caller may have a fallback
        path it still wants to try.
        """
        try:
            out = subprocess.run(
                ["xdotool", "search", *search_args],
                capture_output=True, text=True, timeout=2.0,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning(
                "[viewer-cycler] xdotool search %s failed: %s",
                search_args, exc,
            )
            return []
        if out.returncode != 0 or not out.stdout.strip():
            return []
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    def _drop_excluded_classes(self, wids: list[str]) -> list[str]:
        """Filter out WIDs whose WM_CLASS is in
        :attr:`EXCLUDED_WINDOW_CLASSES`.

        We invoke ``xdotool getwindowclassname`` per WID rather than
        using ``xdotool search --class --classname`` jointly because
        ``--class`` and ``--classname`` are *intersected* with
        ``--name`` when given together, which would prevent the
        classname-first strategy from ever finding the GLFW window
        on builds where its title doesn't include "MuJoCo".
        """
        if not wids:
            return wids
        kept: list[str] = []
        dropped: list[str] = []
        for wid in wids:
            cls = self._get_window_class(wid)
            if cls in self.EXCLUDED_WINDOW_CLASSES:
                dropped.append(f"{wid} (class={cls!r})")
            else:
                kept.append(wid)
        if dropped:
            log.info(
                "[viewer-cycler] excluded %d compositor wrapper "
                "window(s) from candidate list: %s",
                len(dropped), dropped,
            )
        return kept

    @staticmethod
    def _get_window_class(wid: str) -> str:
        """Best-effort fetch of WM_CLASS for ``wid``.

        Returns ``""`` on any failure -- the caller treats unknown
        class as "not excluded" (defensive: we'd rather Tab a
        possibly-wrong window than skip the legitimate one because
        ``xprop`` was missing).
        """
        try:
            res = subprocess.run(
                ["xdotool", "getwindowclassname", wid],
                capture_output=True, text=True, timeout=1.0,
            )
        except (subprocess.TimeoutExpired, OSError):
            return ""
        if res.returncode != 0:
            return ""
        return res.stdout.strip()

    def _maybe_log_multiple(self, wids: list[str], *, source: str) -> None:
        """Log all candidate WIDs when the search returned >1, so
        the operator can narrow the pattern via the manager CLI if we
        picked the wrong one.
        """
        if len(wids) > 1:
            log.info(
                "[viewer-cycler] multiple windows match %s: %s; "
                "using the first (%s). If wrong, narrow with "
                "--viewer-window-classname / --viewer-window-pattern.",
                source, wids, wids[0],
            )

    def _send_camera_cycle_key(self, wid: str) -> bool:
        """Synthesize the configured camera-cycle keypress on ``wid``.

        Uses :attr:`CYCLE_KEYSYM` (defaults to ``bracketright`` -- the
        ``]`` key, which advances to the next fixed camera in the
        passive viewer). The instance attribute can be overridden for
        cycling the other direction (``bracketleft``) or snapping back
        to the free camera (``Escape``); see the class docstring for
        the rationale of why this is parameterised.
        """
        try:
            res = subprocess.run(
                ["xdotool", "key", "--window", wid, self.CYCLE_KEYSYM],
                capture_output=True, text=True, timeout=2.0,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("[viewer-cycler] xdotool key failed: %s", exc)
            return False
        if res.returncode != 0:
            # Most likely the window was closed under us (deploy
            # viewer killed). Drop the cache so the next press
            # re-locates whatever new window is up.
            log.warning(
                "[viewer-cycler] xdotool key exit=%d stderr=%r; "
                "window may have closed, will re-locate on next press.",
                res.returncode, res.stderr.strip(),
            )
            self._cached_wid = None
            self._last_warn_t_no_window = 0.0
            return False
        return True


__all__ = ["ViewerCameraCycler"]


def _diag_main(argv: Optional[list[str]] = None) -> int:
    """Standalone CLI: invoke cycle() once and print a verbose diagnosis.

    Run as:

        python -m gear_sonic.utils.teleop.vr.viewer_camera_cycler

    or via :mod:`gear_sonic.scripts.diag_viewer_camera_cycler`. Designed
    for debugging the xdotool path without putting the VR headset on
    -- if this prints "OK: cycled", the headset L-stick click should
    work too. If it prints anything else, the failure is in the
    xdotool / window / DISPLAY layer rather than the headset / WebXR
    / manager-edge-detection layer.

    Exit codes:
      * 0 -- cycle succeeded (camera-cycle key dispatched; viewer
        should have rotated to the next fixed camera; verify
        visually).
      * 1 -- cycle returned False (see WARN line just above for the
        specific reason: missing xdotool, no DISPLAY, no MuJoCo
        window found, or per-WID class lookup failures). Confirms
        the problem is host-side / X11-side, NOT manager-side.
      * 2 -- argument parsing failed.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Invoke ViewerCameraCycler.cycle() once and print whether "
            "it succeeded. Use to diagnose the xdotool path independent "
            "of the VR headset / WebXR client."
        ),
    )
    parser.add_argument(
        "--pattern", default="MuJoCo",
        help="Window title substring (xdotool search --name). Default: %(default)s.",
    )
    parser.add_argument(
        "--classname", default="MuJoCo",
        help="Window WM_CLASS (xdotool search --classname). Default: %(default)s.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of cycle() calls (with cooldown sleeps between). Default: %(default)d.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable INFO logging from the cycler (lists multiple matches, etc.).",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cycler = ViewerCameraCycler(
        window_search_pattern=args.pattern,
        window_class_name=args.classname,
    )

    # Print a quick environment summary so the operator doesn't have
    # to read the source to know what the cycler will see. This is
    # the most common cause of "I ran the diag and it said no window
    # found": DISPLAY=:0 when the deploy is on :1, or vice versa.
    print("=== viewer-cycler diagnosis ===")
    print(f"  DISPLAY        : {os.environ.get('DISPLAY', '(unset)')}")
    print(f"  XAUTHORITY     : {os.environ.get('XAUTHORITY', '(unset)')}")
    print(f"  xdotool path   : {shutil.which('xdotool') or '(NOT FOUND)'}")
    print(f"  search pattern : name={args.pattern!r} classname={args.classname!r}")
    print(f"  cycle keysym   : {cycler.CYCLE_KEYSYM!r} (next-fixed-camera in mujoco viewer)")
    print(f"  cooldown_s     : {cycler._cooldown_s}")
    print(f"  repeat         : {args.repeat}")
    print()

    overall_ok = True
    for i in range(args.repeat):
        ok = cycler.cycle()
        print(
            f"  cycle #{i + 1}: "
            f"{'OK (' + cycler.CYCLE_KEYSYM + ' dispatched)' if ok else 'no-op (see WARN above)'}"
        )
        overall_ok = overall_ok and ok
        if i + 1 < args.repeat:
            time.sleep(cycler._cooldown_s + 0.05)

    print()
    if overall_ok:
        print(
            "RESULT: keystroke dispatched successfully on every call. "
            "Look at the deploy MuJoCo viewer -- it should have rotated "
            "through {} cameras. If not, the keysym ({}) doesn't bind "
            "to camera-cycle in this build of mujoco.viewer; try "
            "another via --classname / inspecting MuJoCo source.".format(
                args.repeat, cycler.CYCLE_KEYSYM,
            )
        )
        return 0
    print(
        "RESULT: cycler returned False on at least one call. The xdotool "
        "path is the problem -- the headset/WebXR side is fine. Read the "
        "WARN line above for the specific failure mode."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(_diag_main())
