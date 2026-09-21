"""Quest 3 VR data reader via WebSocket.

Runs a WebSocket server that receives head + controller tracking data
from a WebXR app running in the Quest 3 browser, transforms the poses
into the robot coordinate frame, and exposes them through an API that
mirrors PicoReader so the rest of the pipeline can be device-agnostic.

WebXR coordinate system (right-handed, Y-up):
    X = right, Y = up, Z = backward (toward user)

Robot coordinate system (right-handed, Z-up):
    X = forward, Y = left, Z = up

Position transform:
    robot = WEBXR_TO_ROBOT @ webxr
    where WEBXR_TO_ROBOT swaps axes accordingly.

Quaternion transform:
    R_robot = Q * R_webxr * Q^{-1}   (basis change)
"""

import asyncio
import http.server
import json
import os
import socket as _socket
import ssl
import subprocess
import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as sRot


def _ts_print(*args, **kwargs):
    """Timestamped stand-in for print().

    2026-08-03 debugging lesson: the reader's diagnostics (input-source
    flaps, tracking transitions, heartbeats) carried no clock, so flap
    DURATION was unmeasurable from the logs and got guessed instead —
    a guess that then flowed into tuning. Every reader diagnostic now
    carries wall-clock millis; keep it that way.
    """
    import builtins
    builtins.print(time.strftime("[%H:%M:%S") +
                   f".{int(time.time() * 1000) % 1000:03d}]", *args, **kwargs)


print = _ts_print  # module-wide shadow: all reader diagnostics get a clock

# WebXR -> Robot basis-change matrix
# robot_x (forward) = -webxr_z, robot_y (left) = -webxr_x, robot_z (up) = webxr_y
WEBXR_TO_ROBOT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)

_Q_ROT = sRot.from_matrix(WEBXR_TO_ROBOT)


def _to_curl_array(raw: dict) -> np.ndarray | None:
    """Convert a hand-payload dict's ``curls`` field to a clipped (5,) array.

    Returns ``None`` when the field is missing or not the expected shape.
    """
    curls = raw.get("curls")
    if curls is None:
        return None
    arr = np.asarray(curls, dtype=np.float64)
    if arr.shape != (5,):
        return None
    return np.clip(arr, 0.0, 1.0)


def _get_lan_ip() -> str:
    """Return the LAN IP address of this machine (best-effort)."""
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _generate_self_signed_cert(cert_dir: str) -> tuple[str, str]:
    """Generate a self-signed TLS certificate for the WebSocket/HTTPS servers."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file

    print(f"[Quest3] Generating self-signed certificate in {cert_dir}")
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            key_file,
            "-out",
            cert_file,
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/CN=quest3-teleop",
        ],
        check=True,
        capture_output=True,
    )
    print(f"[Quest3] Certificate generated: {cert_file}")
    return cert_file, key_file


def transform_pose_to_robot(
    pos: list | np.ndarray,
    quat_xyzw: list | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a single pose from WebXR frame to robot frame.

    Args:
        pos: [x, y, z] in WebXR coordinates.
        quat_xyzw: [x, y, z, w] quaternion (WebXR / scipy default order).

    Returns:
        robot_pos: (3,) position in robot frame.
        robot_quat_wxyz: (4,) quaternion [w, x, y, z] in robot frame.
    """
    robot_pos = WEBXR_TO_ROBOT @ np.asarray(pos, dtype=np.float64)
    webxr_rot = sRot.from_quat(quat_xyzw)  # scipy: xyzw
    robot_rot = _Q_ROT * webxr_rot * _Q_ROT.inv()
    robot_quat_wxyz = robot_rot.as_quat(scalar_first=True)
    return robot_pos, robot_quat_wxyz


def compute_3pt_pose_from_quest3(data: dict) -> np.ndarray | None:
    """Convert Quest 3 tracking JSON to a (3, 7) VR 3-point pose array.

    The three points are [left_wrist, right_wrist, neck/head], each stored
    as [x, y, z, qw, qx, qy, qz] relative to an estimated root position
    (floor projection of the headset) in the robot coordinate frame.

    Args:
        data: Dict with keys ``head``, ``left``, ``right``, each containing
              ``position`` [x,y,z] and ``orientation`` [x,y,z,w] in WebXR frame.

    Returns:
        (3, 7) float32 ndarray, or *None* if required fields are missing.
    """
    head = data.get("head")
    left = data.get("left")
    right = data.get("right")

    if not head or not left or not right:
        return None
    for part in (head, left, right):
        if "position" not in part or "orientation" not in part:
            return None

    head_pos, head_quat = transform_pose_to_robot(head["position"], head["orientation"])
    left_pos, left_quat = transform_pose_to_robot(left["position"], left["orientation"])
    right_pos, right_quat = transform_pose_to_robot(right["position"], right["orientation"])

    # Estimate root at floor below headset (identity orientation)
    root_pos = np.array([head_pos[0], head_pos[1], 0.0])

    vr_3pt = np.zeros((3, 7), dtype=np.float32)

    vr_3pt[0, :3] = left_pos - root_pos
    vr_3pt[0, 3:] = left_quat

    vr_3pt[1, :3] = right_pos - root_pos
    vr_3pt[1, 3:] = right_quat

    vr_3pt[2, :3] = head_pos - root_pos
    vr_3pt[2, 3:] = head_quat

    return vr_3pt


class Quest3Reader:
    """Background reader for Quest 3 VR data via WebSocket.

    Starts a WebSocket server (optionally with TLS) that the WebXR app
    running in the Quest 3 browser connects to.  Also starts an HTTP(S)
    server to serve the WebXR client page.
    """

    def __init__(
        self,
        ws_host: str = "0.0.0.0",
        ws_port: int = 8765,
        http_port: int = 8443,
        use_ssl: bool = True,
        quiet_periodic: bool = False,
    ):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.http_port = http_port
        self.use_ssl = use_ssl
        # When True, suppress the per-100-msg "msgs=N fps=X idle" line.
        # The first-packet snapshot and one-shot XR / controller / hand
        # tracking events still log -- those are diagnostic gold.
        self.quiet_periodic = bool(quiet_periodic)

        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = False
        self._fps_ema = 0.0
        self._last_t: float | None = None

        self._ws_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Currently-connected WebXR client websocket (None if no client).
        # Used by ``send_message`` to push JSON payloads (e.g. calibration
        # overlays) from Python down to the browser.
        self._client_ws: Any = None

        self._cert_dir = os.path.join(os.path.dirname(__file__), "quest3_certs")

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        """Start WebSocket and HTTP servers in background threads."""
        # Generate calibration audio prompts on first run. The WebXR
        # client falls back to ``speechSynthesis`` for any prompt that
        # has no MP3 on disk, so failure here is non-fatal.
        try:
            from gear_sonic.utils.teleop.vr.quest3_audio_prompts import (
                ensure_prompt_audio_files,
            )

            ensure_prompt_audio_files()
        except Exception as exc:
            print(
                f"[Quest3Reader] WARNING: failed to materialise audio "
                f"prompts: {exc}. Calibration will fall back to browser TTS.",
                flush=True,
            )

        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()
        self._http_thread = threading.Thread(target=self._run_http, daemon=True)
        self._http_thread.start()

    def stop(self):
        self._stop.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._ws_thread:
            self._ws_thread.join(timeout=2.0)

    # -- public getters -------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_latest(self) -> dict | None:
        with self._lock:
            return self._latest

    def get_gamepad_health(self) -> tuple[int, float]:
        """(active input-source count, seconds since sources last hit 0).

        Second value is ``inf`` if sources never dropped this session.
        A recent zero-drop or count < 2 means button/trigger input —
        INCLUDING the VR e-stop — is unreliable or dead even though
        head/controller pose may still stream (heartbeats keep coming).
        Consumers should surface this loudly to the operator."""
        count = getattr(self, "_sources_count", -1)
        zero_t = getattr(self, "_last_sources_zero_t", None)
        age = float("inf") if zero_t is None else time.monotonic() - zero_t
        return count, age

    def get_last_message_age_s(self) -> float:
        """Seconds since the last WebSocket packet was ingested into
        ``_latest``. Returns ``math.inf`` while no message has ever
        arrived. Used by upstream consumers (e.g. quest3_manager_x2)
        to detect Quest 3 sleep / browser tab background / WS drop
        and force a safe idle command so the robot doesn't keep
        executing the last stale stick value forever.
        """
        with self._lock:
            sample = self._latest
        if sample is None:
            return float("inf")
        ts = sample.get("timestamp_monotonic")
        if ts is None:
            return float("inf")
        return time.monotonic() - float(ts)

    def get_3pt_pose(self) -> np.ndarray | None:
        sample = self.get_latest()
        if sample is None:
            return None
        return sample.get("vr_3pt_pose")

    def get_controller_inputs(self) -> tuple[float, float, float, float]:
        """Returns (left_trigger, right_trigger, left_grip, right_grip)."""
        sample = self.get_latest()
        if sample is None:
            return 0.0, 0.0, 0.0, 0.0
        buttons = sample.get("buttons", {})
        return (
            float(buttons.get("leftTrigger", 0.0)),
            float(buttons.get("rightTrigger", 0.0)),
            float(buttons.get("leftGrip", 0.0)),
            float(buttons.get("rightGrip", 0.0)),
        )

    def get_controller_axes(self) -> tuple[float, float, float, float]:
        """Returns (lx, ly, rx, ry) joystick values.

        Convention: positive lx = right, positive ly = forward.
        The WebXR app is expected to invert the Y axis before sending.
        """
        sample = self.get_latest()
        if sample is None:
            return 0.0, 0.0, 0.0, 0.0
        axes = sample.get("axes", {})
        return (
            float(axes.get("lx", 0.0)),
            float(axes.get("ly", 0.0)),
            float(axes.get("rx", 0.0)),
            float(axes.get("ry", 0.0)),
        )

    def get_buttons(self) -> tuple[bool, bool, bool, bool]:
        """Returns (a, b, x, y) face-button pressed states."""
        sample = self.get_latest()
        if sample is None:
            return False, False, False, False
        buttons = sample.get("buttons", {})
        return (
            bool(buttons.get("a", False)),
            bool(buttons.get("b", False)),
            bool(buttons.get("x", False)),
            bool(buttons.get("y", False)),
        )

    def get_stick_clicks(self) -> tuple[bool, bool]:
        """Returns ``(left_stick_click, right_stick_click)`` -- the
        thumbstick "click" buttons (``gpad.buttons[3]`` in the
        oculus-touch / standard gamepad mapping).

        These are independent of the face buttons surfaced by
        :meth:`get_buttons`, so consumers can bind them to actions
        without conflicting with the A/B/X/Y vocabulary. Currently
        used by :mod:`quest3_manager_x2` to cycle the deploy MuJoCo
        viewer's fixed cameras.

        Returns ``(False, False)`` when no sample is available, when
        the WebXR client is on a build that pre-dates the stick-click
        forwarding patch, or when the headset / browser doesn't expose
        ``gpad.buttons[3]`` (some Quest Browser versions skip it).
        """
        sample = self.get_latest()
        if sample is None:
            return False, False
        buttons = sample.get("buttons", {})
        return (
            bool(buttons.get("leftStickClick", False)),
            bool(buttons.get("rightStickClick", False)),
        )

    def get_hand_curls(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None, str | None]:
        """Per-finger curl from XRHand 25-joint poses, if available.

        Returns:
            ``(left_curls, right_curls, left_source, right_source)``.

            - ``*_curls`` are length-5 numpy arrays in ``[0, 1]``,
              ordered as ``[thumb, index, middle, ring, pinky]``. ``None``
              when the side has no XRHand input source for this frame.
            - ``*_source`` is one of ``"hand"`` (XRHand), ``"controller"``
              (analog buttons mapped, no per-finger detail), or ``None``
              (no input on that side).
        """
        sample = self.get_latest()
        if sample is None:
            return None, None, None, None
        hands = sample.get("hands") or {}
        left_raw = hands.get("left") or {}
        right_raw = hands.get("right") or {}

        left_src = left_raw.get("source")
        right_src = right_raw.get("source")
        return (
            _to_curl_array(left_raw),
            _to_curl_array(right_raw),
            left_src,
            right_src,
        )

    def get_thumb_opposition(self) -> tuple[float | None, float | None]:
        """Per-side thumb opposition score from XRHand, if available.

        Returns:
            ``(left_oppose, right_oppose)`` -- floats in ``[0, 1]``,
            independent of finger curl. ``0`` = thumb resting at the
            radial side of the palm (open hand); ``1`` = thumb fully
            opposed across the palm toward the pinky. ``None`` when
            the side has no XRHand data this frame.

        See ``computeThumbOpposition`` in the WebXR client for the
        exact geometric definition. This signal exists because
        thumb-finger touches are dominated by motion at the thumb
        CMC joint (which is NOT in the XRHand chain), so the
        per-finger curl array undershoots even when the operator's
        thumb is fully opposed.
        """
        sample = self.get_latest()
        if sample is None:
            return None, None
        hands = sample.get("hands") or {}
        left_raw = hands.get("left") or {}
        right_raw = hands.get("right") or {}

        def _to_oppose_scalar(raw: dict) -> float | None:
            v = raw.get("oppose")
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            if f < 0.0:
                f = 0.0
            elif f > 1.0:
                f = 1.0
            return f

        return _to_oppose_scalar(left_raw), _to_oppose_scalar(right_raw)

    def get_finger_tip_oppose(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Per-side, per-finger thumb-tip-to-fingertip proximity score.

        Returns:
            ``(left_finger_tip_oppose, right_finger_tip_oppose)`` --
            length-4 numpy arrays in ``[0, 1]``, ordered as
            ``[index, middle, ring, pinky]``. ``None`` when the side
            has no XRHand data this frame, or when the WebXR client
            is too old to emit the field (forwards-compat). Within
            the array, individual entries may be NaN if a specific
            fingertip joint dropped out for the frame -- callers
            should treat NaN as "fall back to the curl signal".

        Companion to :meth:`get_thumb_opposition` (which returns a
        single scalar = MIN over fingertips). This 4-vector splits
        that signal so each receiving finger gets its own dedicated
        proximity score, and the omnihand non-thumb pip motors can
        be driven on ``max(curls[i], finger_tip_oppose[i])``.

        See ``computeFingerTipOppose`` in the WebXR client for the
        exact geometric definition. Same touch (~0.5 cm) and far
        (~3.5 cm) thresholds as :meth:`get_thumb_opposition`.
        """
        sample = self.get_latest()
        if sample is None:
            return None, None
        hands = sample.get("hands") or {}
        left_raw = hands.get("left") or {}
        right_raw = hands.get("right") or {}

        def _to_tip_oppose_array(raw: dict) -> np.ndarray | None:
            v = raw.get("finger_tip_oppose")
            if v is None:
                return None
            try:
                arr = np.asarray(v, dtype=np.float32)
            except (TypeError, ValueError):
                return None
            if arr.shape != (4,):
                return None
            # NaN entries are kept; callers must handle them. Clamp
            # finite entries to [0, 1].
            finite = np.isfinite(arr)
            arr[finite] = np.clip(arr[finite], 0.0, 1.0)
            return arr

        return _to_tip_oppose_array(left_raw), _to_tip_oppose_array(right_raw)

    # -- WebSocket server -----------------------------------------------------

    def _make_ssl_context(self) -> ssl.SSLContext | None:
        if not self.use_ssl:
            return None
        try:
            cert_file, key_file = _generate_self_signed_cert(self._cert_dir)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert_file, key_file)
            return ctx
        except Exception as e:
            print(f"[Quest3Reader] SSL setup failed ({e}), falling back to WS")
            return None

    async def _handle_connection(self, websocket):
        self._connected = True
        self._client_ws = websocket
        addr = getattr(websocket, "remote_address", "unknown")
        print(f"[Quest3Reader] Client connected: {addr}")
        try:
            async for message in websocket:
                if self._stop.is_set():
                    break
                self._process_message(message)
        except Exception as e:
            if not self._stop.is_set():
                print(f"[Quest3Reader] Connection error: {e}")
        finally:
            self._connected = False
            if self._client_ws is websocket:
                self._client_ws = None
            print(f"[Quest3Reader] Client disconnected: {addr}")

    def send_message(self, payload: dict) -> bool:
        """Push a JSON payload to the connected WebXR client.

        Used by the calibration script to drive the on-headset overlay
        (pose instructions, capture confirmations). Returns True if the
        send was scheduled, False if no client is connected. Thread-safe;
        marshals to the asyncio loop running the websocket server.
        """
        if not self._connected or self._client_ws is None or self._loop is None:
            return False
        try:
            text = json.dumps(payload)
        except (TypeError, ValueError) as exc:
            print(f"[Quest3Reader] send_message: bad payload ({exc})")
            return False

        ws = self._client_ws
        loop = self._loop

        async def _send():
            try:
                await ws.send(text)
            except Exception as exc:  # client disconnected mid-send, etc.
                print(f"[Quest3Reader] send_message warn: {exc}")

        try:
            asyncio.run_coroutine_threadsafe(_send(), loop)
            return True
        except Exception as exc:
            print(f"[Quest3Reader] send_message threadsafe schedule failed: {exc}")
            return False

    def _process_message(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Handle status messages from the WebXR client
        if data.get("_type") == "status":
            event = data.get("event", "unknown")
            if event == "xr_session_started":
                ref = data.get("ref_space", "?")
                supported = ", ".join(data.get("supported", []))
                unsupported = ", ".join(data.get("unsupported", []))
                n_inputs = data.get("input_sources", 0)
                print(f"[Quest3Reader] XR session started!")
                print(f"[Quest3Reader]   Reference space: {ref}")
                print(f"[Quest3Reader]   Supported spaces: {supported}")
                if unsupported:
                    print(f"[Quest3Reader]   Unsupported spaces: {unsupported}")
                print(f"[Quest3Reader]   Input sources (controllers): {n_inputs}")
                if ref != "local-floor":
                    print(f"[Quest3Reader]   WARNING: 'local-floor' not available. Floor height may be wrong.")
                    print(f"[Quest3Reader]   TIP: Set up Guardian on Quest 3: Settings > Physical Space > Space Setup")
            elif event == "xr_ref_space_failed":
                unsupported = ", ".join(data.get("unsupported", []))
                print(f"[Quest3Reader] ERROR: No XR reference space available!")
                print(f"[Quest3Reader]   Unsupported: {unsupported}")
                print(f"[Quest3Reader]   FIX: Set up Guardian on Quest 3:")
                print(f"[Quest3Reader]     1. Press Meta button > Settings > Physical Space > Space Setup")
                print(f"[Quest3Reader]     2. Choose 'Roomscale' or 'Stationary'")
                print(f"[Quest3Reader]     3. Follow prompts to draw boundary, then reload the page")
            elif event == "input_sources_changed":
                count = data.get("count", 0)
                sources = data.get("sources", [])
                # The WebXR client emits an ``input_sources_changed``
                # event every time the headset toggles between hand-only
                # / controller / multimodal -- which on Quest 3 happens
                # constantly as the operator rests the controllers and
                # picks them back up. Build a stable signature of
                # (handedness, has_grip, has_hand) tuples so we only log
                # when the *actual* configuration changes, not on
                # idempotent re-broadcasts. Knocks the per-session log
                # volume from hundreds of lines down to a handful.
                sig = tuple(sorted(
                    (
                        s.get("handedness", "?"),
                        bool(s.get("has_gamepad", False)),
                        bool(s.get("has_grip", False)),
                        bool(s.get("has_hand", s.get("type") == "hand-tracking")),
                    )
                    for s in sources
                ))
                prev_sig = getattr(self, "_last_sources_sig", None)
                # Gamepad-health bookkeeping (2026-08-03): controllers
                # detaching mid-session silently killed ALL button and
                # trigger input (including the operator e-stop) while
                # pose kept streaming. Track the flap so consumers can
                # warn loudly instead of failing silent.
                self._sources_count = count
                if count == 0:
                    self._last_sources_zero_t = time.monotonic()
                if sig == prev_sig:
                    return
                self._last_sources_sig = sig
                print(f"[Quest3Reader] Input sources changed: {count} detected")
                # Quest 3 multimodal: a single source can carry BOTH
                # hand-tracking and a gripSpace controller. Inspect
                # `has_grip` and `has_hand` directly so we don't
                # spuriously warn "no controllers" on a multimodal
                # source whose gamepad is actually attached.
                has_controller = False
                has_hand_only = False
                for s in sources:
                    stype = s.get("type", "unknown")
                    hand = s.get("handedness", "?")
                    has_gpad = s.get("has_gamepad", False)
                    has_grip = s.get("has_grip", False)
                    has_hand_input = s.get("has_hand", stype == "hand-tracking")
                    print(
                        f"[Quest3Reader]   {hand}: {stype} "
                        f"(gamepad={'yes' if has_gpad else 'NO'} "
                        f"grip={'yes' if has_grip else 'NO'} "
                        f"hand={'yes' if has_hand_input else 'NO'})"
                    )
                    if has_grip:
                        has_controller = True
                    if has_hand_input and not has_grip:
                        has_hand_only = True
                if has_hand_only and not has_controller:
                    print(f"[Quest3Reader]   WARNING: Hand tracking only -- no controllers detected!")
                    print(f"[Quest3Reader]   FIX: Pick up the physical Quest 3 controllers")
                    print(f"[Quest3Reader]        (A/B/X/Y, joysticks, and stable wrist tracking")
                    print(f"[Quest3Reader]         all need the gripSpace from a controller).")
                elif has_controller:
                    if any(s.get("has_hand", False) for s in sources):
                        print(f"[Quest3Reader]   Multimodal: controllers + hand-tracking active. "
                              f"Pose from gripSpace, finger curls from XRHand.")
                    else:
                        print(f"[Quest3Reader]   Controllers detected — buttons and joysticks active.")
            elif event == "visibility":
                # XRSession visibility flip ("hidden" / "visible" /
                # "visible-blurred"). When this goes "hidden" the
                # compositor stops calling onXRFrame entirely, so
                # the operator perceives "controllers stopped
                # responding" even though the manager loop is fine.
                state = data.get("state", "?")
                prev = data.get("prev")
                if prev is not None:
                    print(f"[Quest3Reader] XR visibility: {prev} -> {state}")
                else:
                    print(f"[Quest3Reader] XR visibility: {state}")
                if state != "visible":
                    print(
                        f"[Quest3Reader]   WARNING: XR frames will pause "
                        f"while visibility != 'visible'. Operator likely "
                        f"removed the headset, opened the system menu, "
                        f"or stepped outside Guardian."
                    )
            elif event == "frame_stall":
                # WebXR rAF gap exceeded the browser-side threshold
                # (default 250ms). Indicates the compositor paused
                # frame delivery -- a much more reliable signal than
                # waiting for `[Quest3Reader] msgs` fps to dip,
                # because msgs/fps is averaged over 100-msg windows.
                gap = data.get("gap_ms", "?")
                vis = data.get("vis", "?")
                print(
                    f"[Quest3Reader] WARN: WebXR frame stall {gap}ms "
                    f"(visibility={vis})"
                )
            elif event == "heartbeat":
                # 2s pulse from the browser. Useful even when idle:
                # confirms the headset/browser is alive AND tells
                # us how many button rising edges happened in the
                # last 2s. The latter is the diagnostic that
                # catches "operator mashed B 12 times but no mode
                # flip happened in manager.log".
                vis = data.get("vis", "?")
                fps = data.get("fps", 0)
                ncon = data.get("n_controllers", "?")
                edges = data.get("btn_edges") or {}
                edge_str = " ".join(
                    f"{k}={v}" for k, v in edges.items() if v
                )
                if not edge_str:
                    edge_str = "(no btn edges)"
                left_emu = data.get("left_emulated")
                right_emu = data.get("right_emulated")
                emu_bits = []
                if left_emu is True:
                    emu_bits.append("L=IMU-only")
                if right_emu is True:
                    emu_bits.append("R=IMU-only")
                emu_str = (" [" + ",".join(emu_bits) + "]") if emu_bits else ""
                print(
                    f"[Quest3Reader] heartbeat vis={vis} fps={fps} "
                    f"controllers={ncon} edges=({edge_str}){emu_str}"
                )
                # Promote to WARN when the operator was clearly
                # acting but the gamepad path is the only thing
                # carrying it (no XR-level select/squeeze) AND
                # visibility is anything other than visible. This
                # is the smoking-gun pattern for the
                # "mode chord doesn't fire" footgun.
                if vis != "visible" and any(edges.values()):
                    print(
                        f"[Quest3Reader]   WARNING: button presses "
                        f"recorded but XR visibility is '{vis}'. "
                        f"The Python decoder likely never saw them."
                    )
            elif event == "tracking":
                # Per-side controller tracking source flipped
                # between camera-tracked and IMU dead-reckoning.
                side = data.get("side", "?")
                emu = data.get("emulated")
                print(
                    f"[Quest3Reader] {side} controller tracking -> "
                    f"{'IMU-only (camera lost it)' if emu else 'camera (recovered)'}"
                )
            elif event == "controller_pose":
                side = data.get("side", "?")
                has_pose = data.get("has_pose", True)
                # Tracking flaps (IMU-only controllers at the edge of the
                # camera cone) fire LOST/RECOVERED pairs several times a
                # minute — rate-limit to one pair per side per 10 s. The
                # heartbeat's controllers=N count stays authoritative.
                import time as _t
                if not hasattr(self, "_pose_log_ts"):
                    self._pose_log_ts = {}
                _suppress = (
                    _t.monotonic() - self._pose_log_ts.get(side, 0.0) < 10.0
                )
                if not _suppress:
                    self._pose_log_ts[side] = _t.monotonic()
                if _suppress:
                    pass
                elif has_pose:
                    print(
                        f"[Quest3Reader] {side} controller pose RECOVERED "
                        f"(gripSpace getPose() succeeding again)"
                    )
                else:
                    print(
                        f"[Quest3Reader] WARN: {side} controller pose LOST "
                        f"(gripSpace getPose() returning null -- controller "
                        f"in operator's hand but tracking system can't see it)"
                    )
            elif event == "xr_button":
                kind = data.get("kind", "?")
                handed = data.get("handedness") or "?"
                print(f"[Quest3Reader] XR {kind} ({handed})")
            elif event == "xr_session_ended":
                fc = data.get("frame_count", "?")
                print(
                    f"[Quest3Reader] XR session ended cleanly "
                    f"(browser-side frame_count={fc})"
                )
            elif event == "page_hidden":
                persisted = data.get("persisted", False)
                print(
                    f"[Quest3Reader] WARN: browser page hidden "
                    f"(persisted={persisted}) -- the WebXR client is "
                    f"about to disconnect."
                )
            elif event == "page_unload":
                print(f"[Quest3Reader] WARN: browser unload event received.")
            elif event == "doc_visibility":
                state = data.get("state", "?")
                # Document-level (not XR session-level) visibility:
                # fires when the user switches tabs in the browser.
                print(f"[Quest3Reader] document visibility -> {state}")
            else:
                print(f"[Quest3Reader] Status: {data}")
            return

        self._msg_count = getattr(self, "_msg_count", 0) + 1
        self._last_log_count = getattr(self, "_last_log_count", 0)

        # Log first message and then every 100th message
        if self._msg_count == 1:
            print(f"[Quest3Reader] First tracking data received! Keys: {list(data.keys())}")
            btns = data.get("buttons", {})
            axes = data.get("axes", {})
            print(f"[Quest3Reader]   buttons: {btns}")
            print(f"[Quest3Reader]   axes: {axes}")
        elif self._msg_count % 2000 == 0 and not self.quiet_periodic:
            btns = data.get("buttons", {})
            axes = data.get("axes", {})
            has_input = any(v for k, v in btns.items() if k in ("a", "b", "x", "y") and v)
            has_trigger = any(v > 0.1 for k, v in btns.items() if k not in ("a", "b", "x", "y"))
            has_stick = any(abs(v) > 0.05 for v in axes.values())
            status_parts = []
            if has_input:
                pressed = [k.upper() for k in ("a", "b", "x", "y") if btns.get(k)]
                status_parts.append(f"btns=[{'+'.join(pressed)}]")
            if has_trigger:
                status_parts.append("triggers=active")
            if has_stick:
                status_parts.append(f"sticks=active")
            status = " | ".join(status_parts) if status_parts else "idle"
            print(f"[Quest3Reader] msgs={self._msg_count} fps={self._fps_ema:.1f} {status}")

        vr_3pt_pose = compute_3pt_pose_from_quest3(data)

        now = time.time()
        dt = (now - self._last_t) if self._last_t else 0.0
        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps_ema = (
                inst_fps if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst_fps
            )
        self._last_t = now

        sample = {
            "vr_3pt_pose": vr_3pt_pose,
            "buttons": data.get("buttons", {}),
            "axes": data.get("axes", {}),
            "hands": data.get("hands", {}),
            "timestamp_realtime": now,
            "timestamp_monotonic": time.monotonic(),
            "dt": dt,
            "fps": self._fps_ema,
        }
        with self._lock:
            self._latest = sample

    def _run_ws(self):
        try:
            import websockets
        except ImportError:
            print(
                "[Quest3Reader] ERROR: 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        ssl_ctx = self._make_ssl_context()
        proto = "wss" if ssl_ctx else "ws"

        async def _serve():
            server = await websockets.serve(
                self._handle_connection,
                self.ws_host,
                self.ws_port,
                ssl=ssl_ctx,
            )
            lan_ip = _get_lan_ip()
            print(f"[Quest3Reader] WebSocket server on {proto}://{lan_ip}:{self.ws_port}")
            try:
                await asyncio.get_event_loop().create_future()
            finally:
                server.close()
                await server.wait_closed()

        try:
            self._loop.run_until_complete(_serve())
        except Exception as exc:
            # A dead WS server means the headset can load the page but
            # never stream — surface it loudly instead of dying silently.
            print(
                f"[Quest3Reader] FATAL: WebSocket server failed: {exc!r} "
                f"(host={self.ws_host} port={self.ws_port}); headset "
                "streaming is DOWN.",
                flush=True,
            )

    # -- HTTP server for the WebXR app ----------------------------------------

    def _run_http(self):
        app_dir = os.path.join(os.path.dirname(__file__), "quest3_webxr_app")
        if not os.path.isdir(app_dir):
            print(f"[Quest3Reader] WebXR app not found at {app_dir}")
            return

        class _Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=app_dir, **kwargs)

            def end_headers(self):
                # Prevent browser caching so code changes take effect immediately
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer((self.ws_host, self.http_port), _Handler)

        proto = "http"
        if self.use_ssl:
            try:
                cert_file, key_file = _generate_self_signed_cert(self._cert_dir)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert_file, key_file)
                server.socket = ctx.wrap_socket(server.socket, server_side=True)
                proto = "https"
            except Exception as e:
                print(f"[Quest3Reader] HTTPS setup failed ({e}), using HTTP")

        lan_ip = _get_lan_ip()
        print(f"[Quest3Reader] Serving WebXR app at {proto}://{lan_ip}:{self.http_port}")

        while not self._stop.is_set():
            server.timeout = 0.5
            server.handle_request()
