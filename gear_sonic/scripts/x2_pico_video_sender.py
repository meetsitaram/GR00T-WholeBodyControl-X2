#!/usr/bin/env python
"""X2 -> Pico headset video sender (XRoboToolkit Unity-client compatible).

Streams the robot's ego view (``rgbd_head_front`` — the one camera that
exists under the same name in the calibrated URDF, the MuJoCo sim, and the
PC2 HAL) to the XRoboToolkit Unity client's built-in H.264 video receiver,
so the operator sees what X2 sees while teleoperating with the Pico.

Wire protocol (reverse-engineered from XRoboToolkit-Orin-Video-Sender,
``main_web_gst.cpp`` ``on_new_sample`` + ``network_helper.hpp``): a plain
TCP stream of packets, each packet = 4-byte BIG-ENDIAN payload length
followed by exactly one H.264 access unit (Annex-B). The stereo convention
is ONE side-by-side frame (e.g. 2560x720 = left|right) that the client
splits into per-eye planes. No RTSP/WebRTC/depth channel exists.

Headset side: add/point a ``video_source.yml`` entry
(/sdcard/Android/data/com.xrobotoolkit.client/files/) at this machine's
IP, then Listen in the client.

Frame sources (``--source``):

  test     Moving synthetic pattern. No robot, no sim — loopback/bring-up.
  zmq-cam  The camera-bridge msgpack protocol on tcp://HOST:5555
           (``{"timestamps": {...}, "images": {key: jpeg-bytes}}`` — the
           PC2 publisher x2_pc2_camera_zmq_publisher.py and the G1 sim
           image publisher both speak it). ``--keys head_front`` for mono,
           ``--keys stereo_left,stereo_right`` for side-by-side stereo.
  sim      SUB the deploy's ``x2_debug:5557`` state stream and render the
           ``ego_view`` (= rgbd_head_front) camera with an in-process
           MuJoCo renderer — works against
           the local MuJoCo sim stack with zero deploy changes.

Encoding: ffmpeg subprocess, libx264 (``--encoder x264``, default;
zerolatency + repeat-headers so a late-joining client picks up at the next
IDR) or NVENC (``--encoder nvenc``). AUD NALs are forced so the Annex-B
stream can be split into one-access-unit packets, matching the reference
sender's per-GstBuffer framing.

Transport: ``--mode listen`` (default; bind and wait for the headset to
connect, re-accept on disconnect) or ``--mode connect HOST`` (push to a
listening receiver). Frames are dropped while no client is connected.

Examples:
  # Loopback smoke test (pair with tests/test_x2_pico_video_sender.py):
  python -m gear_sonic.scripts.x2_pico_video_sender --source test

  # Sim teleop ego view, headset connects to <this-pc>:12345:
  python -m gear_sonic.scripts.x2_pico_video_sender --source sim

  # Real robot head camera via the PC2 bridge:
  python -m gear_sonic.scripts.x2_pico_video_sender \\
      --source zmq-cam --cam-host $PC2_HOST --keys head_front

  # Side-by-side stereo from the IMX900 pair (needs rectification pass
  # before this is comfortable — see plan doc):
  python -m gear_sonic.scripts.x2_pico_video_sender \\
      --source zmq-cam --cam-host $PC2_HOST \\
      --keys stereo_left,stereo_right --width 1280 --height 480
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import shutil
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("x2_pico_video_sender")

# NAL unit types (H.264, nal_unit_type = byte & 0x1F)
_NAL_AUD = 9


# ---------------------------------------------------------------------------
# Annex-B access-unit splitter
# ---------------------------------------------------------------------------


class AnnexBAccessUnitSplitter:
    """Incremental splitter: raw Annex-B byte stream -> one access unit per
    emit. Relies on the encoder inserting an Access Unit Delimiter (AUD,
    nal_unit_type 9) at the start of every AU — both encoder configs below
    force that. Everything between consecutive AUD start codes is one AU
    (AUD + optional SPS/PPS + slices), which mirrors the reference sender's
    one-GstBuffer-per-packet framing."""

    def __init__(self) -> None:
        self._buf = bytearray()

    @staticmethod
    def _iter_aud_offsets(buf: bytes | bytearray) -> Iterator[int]:
        """Yield offsets of start codes whose NAL is an AUD."""
        i = 0
        n = len(buf)
        while True:
            j = buf.find(b"\x00\x00\x01", i)
            if j < 0 or j + 3 >= n:
                return
            # Prefer the 4-byte form's true start when preceded by a zero.
            start = j - 1 if j > 0 and buf[j - 1] == 0 else j
            nal_type = buf[j + 3] & 0x1F
            if nal_type == _NAL_AUD:
                yield start
            i = j + 3

    def push(self, data: bytes) -> list[bytes]:
        """Feed bytes; return zero or more complete access units."""
        self._buf.extend(data)
        offsets = list(self._iter_aud_offsets(self._buf))
        if len(offsets) < 2:
            return []
        units = [
            bytes(self._buf[offsets[k]:offsets[k + 1]])
            for k in range(len(offsets) - 1)
        ]
        del self._buf[: offsets[-1]]
        return units


# ---------------------------------------------------------------------------
# ffmpeg H.264 encoder subprocess
# ---------------------------------------------------------------------------


class FramedH264Encoder:
    """RGB frames in -> per-access-unit H.264 packets out (via callback).

    Wraps one ffmpeg subprocess: rawvideo rgb24 on stdin, Annex-B H.264 on
    stdout. A reader thread splits the stream on AUDs and invokes
    ``on_access_unit`` for each complete AU.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        bitrate_kbps: int,
        gop: int,
        encoder: str,
        on_access_unit: Callable[[bytes], None],
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        self._width = width
        self._height = height
        self._frame_nbytes = width * height * 3
        self._on_access_unit = on_access_unit

        common_in = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "pipe:0", "-an",
        ]
        if encoder == "x264":
            codec = [
                "-c:v", "libx264", "-preset", "ultrafast",
                "-tune", "zerolatency",
                # baseline profile: embedded/Android MediaCodec decoders
                # (the XRoboToolkit Unity client) are far happier with it
                # than x264's default High profile — the 2026-08-15 crash
                # repro died ~57 packets into a High-profile stream.
                "-profile:v", "baseline",
                # rgb24 input defaults libx264 to 4:4:4 chroma — baseline
                # rejects it (encoder fails to open) and Android MediaCodec
                # decoders don't take 4:4:4 either (the likely original
                # headset-crash cause). Force the universal 4:2:0.
                "-pix_fmt", "yuv420p",
                "-b:v", f"{bitrate_kbps}k", "-g", str(gop),
                # repeat-headers: SPS/PPS on every IDR so a late-joining
                # client decodes from the next keyframe. aud: AU delimiters
                # for the splitter.
                "-x264opts", f"repeat-headers=1:aud=1:keyint={gop}:min-keyint={gop}",
            ]
        elif encoder == "nvenc":
            codec = [
                "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll",
                "-b:v", f"{bitrate_kbps}k", "-g", str(gop),
                "-bsf:v", "h264_metadata=aud=insert",
            ]
        else:
            raise ValueError(f"unknown encoder {encoder!r}")
        # no output buffering anywhere: each encoded packet leaves ffmpeg immediately
        cmd = common_in + codec + ["-flush_packets", "1", "-fflags", "nobuffer", "-max_delay", "0", "-f", "h264", "pipe:1"]
        log.info("encoder: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,   # unbuffered pipes
        )
        self._splitter = AnnexBAccessUnitSplitter()
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name="h264-reader"
        )
        self._reader.start()

    def submit(self, frame_rgb: np.ndarray) -> None:
        """Write one (H, W, 3) uint8 RGB frame to the encoder."""
        data = np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes()
        if len(data) != self._frame_nbytes:
            raise ValueError(
                f"frame is {len(data)} bytes, expected {self._frame_nbytes} "
                f"({self._width}x{self._height}x3)"
            )
        assert self._proc.stdin is not None
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        while True:
            try:
                chunk = os.read(self._proc.stdout.fileno(), 1 << 16)   # available bytes, no fill wait
            except OSError:
                chunk = b""
            if not chunk:
                break
            for au in self._splitter.push(chunk):
                self._on_access_unit(au)

    def close(self) -> None:
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait(timeout=5.0)
        except Exception:
            self._proc.kill()
        if self._reader.is_alive():
            self._reader.join(timeout=2.0)


# ---------------------------------------------------------------------------
# TCP transport (4-byte big-endian length framing)
# ---------------------------------------------------------------------------


class FramedTcpTransport:
    """Sends length-prefixed packets to one peer; drops when none.

    listen mode: bind + accept in a background thread, re-accept after a
    client drops (operator can restart the headset app freely).
    connect mode: dial out with retry (for receivers that listen).
    """

    @staticmethod
    def headset_ip_from_service(exclude: set[str]) -> str:
        """The headset's address = the peer of the established TCP connection the XRoboToolkit PC
        Service (RoboticsServiceProcess) holds from the headset app (the app dials the service's
        device port when the operator taps connect). Deterministic: no scanning, no ARP guessing.
        Empty string until the app connects."""
        try:
            out = subprocess.run(["ss", "-tnp", "state", "established"], capture_output=True, text=True, timeout=2).stdout
        except Exception:
            return ""
        for line in out.splitlines():
            if "roboticsservice" not in line.lower():
                continue
            for tok in line.split():
                m = re.match(r"^\[::ffff:(\d+\.\d+\.\d+\.\d+)\]:\d+$|^(\d+\.\d+\.\d+\.\d+):\d+$", tok)
                if not m:
                    continue
                ip = m.group(1) or m.group(2)
                if ip not in exclude and not ip.startswith("127."):
                    return ip
        return ""

    def __init__(self, *, mode: str, host: str, port: int, exclude_ips: set[str] | None = None) -> None:
        self._mode = mode
        self._host = host          # connect mode: an IP, or "auto" = resolve from the PC Service socket
        self._exclude = set(exclude_ips or ())
        self._auto_logged = False
        self._port = port
        self._peer: socket.socket | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sent = 0
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._maintain_peer, daemon=True, name="tcp-peer"
        )
        self._server: socket.socket | None = None
        if mode == "listen":
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((host, port))
            self._server.listen(1)
            log.info("listening on %s:%d (waiting for headset client)", host, port)
        self._thread.start()

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._peer is not None

    @property
    def bound_port(self) -> int:
        """Actual bound port (listen mode; supports port 0 = ephemeral)."""
        if self._server is None:
            return self._port
        return self._server.getsockname()[1]

    def _maintain_peer(self) -> None:
        while not self._stop.is_set():
            if self.connected:
                time.sleep(0.1)
                continue
            try:
                if self._mode == "listen":
                    assert self._server is not None
                    self._server.settimeout(1.0)
                    try:
                        peer, addr = self._server.accept()
                    except socket.timeout:
                        continue
                    log.info("client connected: %s", addr)
                else:
                    host = self._host
                    if host == "auto":
                        host = self.headset_ip_from_service(self._exclude)
                        if not host:
                            if not self._auto_logged:
                                log.info("headset not connected to the PC Service yet; streaming starts when it is (preview runs meanwhile)")
                                self._auto_logged = True
                            time.sleep(2.0)
                            continue
                    peer = socket.create_connection((host, self._port), timeout=2.0)
                    log.info("connected to %s:%d", host, self._port)
                peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # small send buffer: back-pressure from a slow headset/wifi shows up within ~0.2 s
                # instead of after several seconds of queued video (5-10 s lag seen 2026-09-08)
                peer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)
                with self._lock:
                    self._peer = peer
            except OSError as exc:
                log.debug("peer attempt failed: %s", exc)
                time.sleep(1.0)

    def writable(self) -> bool:
        """True when the peer socket can take more data now (select, no wait). Used to DROP frames
        before encoding when the link is saturated, so latency stays bounded."""
        with self._lock:
            peer = self._peer
        if peer is None:
            return False
        try:
            _, w, _ = select.select([], [peer], [], 0)
            return bool(w)
        except (OSError, ValueError):
            return False

    def send_packet(self, payload: bytes) -> bool:
        with self._lock:
            peer = self._peer
        if peer is None:
            self._dropped += 1
            return False
        try:
            peer.sendall(struct.pack(">I", len(payload)) + payload)
            self._sent += 1
            return True
        except OSError as exc:
            log.warning("peer dropped (%s); waiting for reconnect", exc)
            with self._lock:
                self._peer = None
            try:
                peer.close()
            except OSError:
                pass
            return False

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._peer is not None:
                try:
                    self._peer.close()
                except OSError:
                    pass
                self._peer = None
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


class TestPatternSource:
    """Moving gradient + bouncing square; needs nothing but numpy."""

    __test__ = False  # not a pytest class despite the Test* name

    def __init__(self, width: int, height: int) -> None:
        self._w = width
        self._h = height
        self._n = 0
        xx = np.linspace(0, 255, width, dtype=np.uint8)
        yy = np.linspace(0, 255, height, dtype=np.uint8)
        self._gx, self._gy = np.meshgrid(xx, yy)

    def next_frame(self) -> np.ndarray:
        f = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        shift = (self._n * 3) % 256
        f[..., 0] = (self._gx + shift) % 256
        f[..., 1] = (self._gy + shift // 2) % 256
        f[..., 2] = 128
        s = max(8, self._h // 8)
        x = int((0.5 + 0.4 * np.sin(self._n / 15.0)) * (self._w - s))
        y = int((0.5 + 0.4 * np.cos(self._n / 11.0)) * (self._h - s))
        f[y:y + s, x:x + s] = (255, 255, 255)
        self._n += 1
        return f


class ZmqCameraSource:
    """SUB the camera-bridge msgpack protocol; mono or side-by-side."""

    def __init__(
        self, *, host: str, port: int, keys: list[str],
        width: int, height: int,
    ) -> None:
        import cv2  # noqa: F401  (lazy heavy imports)
        import msgpack
        import zmq

        self._cv2 = cv2
        self._msgpack = msgpack
        self._keys = keys
        self._w = width
        self._h = height
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        # Latest-frame-only: same CONFLATE semantics as the recorder's
        # ComposedCameraClientSensor — a viewer must never lag reality.
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sock.connect(f"tcp://{host}:{port}")
        self._last_ts: dict[str, float] = {}
        log.info("zmq-cam SUB tcp://%s:%d keys=%s", host, port, keys)

    def next_frame(self) -> np.ndarray | None:
        import zmq

        try:
            raw = self._sock.recv(flags=zmq.NOBLOCK)   # never block the send loop: no new
        except zmq.Again:                             # frame -> None, the caller repeats the last one
            return None
        msg = self._msgpack.unpackb(raw, raw=False)
        images = msg.get("images", {})
        ts = msg.get("timestamps", {})
        # The bridge republishes cached frames at 50 Hz; skip re-encoding
        # identical JPEGs (CONFLATE already dropped the backlog).
        if ts and all(ts.get(k) == self._last_ts.get(k) for k in self._keys):
            return None
        self._last_ts.update({k: ts.get(k) for k in self._keys if k in ts})

        panes = []
        for key in self._keys:
            jpeg = images.get(key)
            if jpeg is None:
                return None
            arr = np.frombuffer(
                jpeg if isinstance(jpeg, (bytes, bytearray)) else bytes(jpeg),
                dtype=np.uint8,
            )
            bgr = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            panes.append(self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB))
        frame = panes[0] if len(panes) == 1 else np.hstack(panes)
        if frame.shape[:2] != (self._h, self._w):
            frame = self._cv2.resize(
                frame, (self._w, self._h), interpolation=self._cv2.INTER_AREA
            )
        return frame


class UdpCamSource:
    """Frames from the PC2 camera bridge's UDP fan-out (x2_pc2_camera_zmq_publisher.py --udp-port).

    We send a hello datagram every second (that registers us; the bridge forgets peers silent for
    3 s), receive each frame as a burst of chunked datagrams and hand back the NEWEST complete
    frame. Nothing queues anywhere: a lost chunk drops that frame, a slow consumer just sees
    fewer frames. Replaces the TCP/ZMQ path that built seconds of backlog on wifi (2026-09-08).
    """

    HDR = struct.Struct("!4sIHHd16s")
    MAGIC = b"X2CF"

    def __init__(self, *, host: str, port: int, key: str, width: int, height: int) -> None:
        import cv2  # noqa: F401
        self._cv2 = cv2
        self._w, self._h = width, height
        self._key = key.encode()[:16].ljust(16, b"\0")
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self._sock.bind(("0.0.0.0", 0))
        self._sock.setblocking(False)
        self._last_hello = 0.0
        self._asm: dict[int, tuple[int, float, dict[int, bytes], float]] = {}   # fid -> (n, ts, chunks, t_first)
        self._done_fid = 0
        self.last_stamp = 0.0      # camera stamp of the last delivered frame (bridge clock)
        self.last_age = float("nan")
        self.frames = 0; self.incomplete = 0
        log.info("udp-cam hello -> %s:%d key=%s", host, port, key)

    def _hello(self) -> None:
        now = time.monotonic()
        if now - self._last_hello >= 1.0:
            self._last_hello = now
            try:
                self._sock.sendto(b"SUB", self._addr)
            except OSError:
                pass

    def next_frame(self) -> np.ndarray | None:
        self._hello()
        newest = None
        # drain everything that arrived since the last call
        while True:
            try:
                pkt = self._sock.recv(2048)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            if len(pkt) < self.HDR.size:
                continue
            magic, fid, n, i, ts, key = self.HDR.unpack_from(pkt)
            if magic != self.MAGIC or key != self._key or fid <= self._done_fid:
                continue
            ent = self._asm.get(fid)
            if ent is None:
                ent = (n, ts, {}, time.monotonic()); self._asm[fid] = ent
            ent[2][i] = pkt[self.HDR.size:]
            if len(ent[2]) == n and (newest is None or fid > newest):
                newest = fid
        if newest is None:
            # forget assemblies that can never complete (older than 1 s)
            now = time.monotonic()
            for fid in [f for f, e in self._asm.items() if now - e[3] > 1.0]:
                del self._asm[fid]; self.incomplete += 1
            return None
        n, ts, chunks, _ = self._asm[newest]
        jpg = b"".join(chunks[i] for i in range(n))
        # everything older than the frame we deliver is garbage now
        for fid in [f for f in self._asm if f <= newest]:
            if fid != newest: self.incomplete += 1
            del self._asm[fid]
        self._done_fid = newest
        arr = np.frombuffer(jpg, dtype=np.uint8)
        bgr = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        self.last_stamp = ts
        self.frames += 1
        frame = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (self._h, self._w):
            frame = self._cv2.resize(frame, (self._w, self._h), interpolation=self._cv2.INTER_AREA)
        return frame


# ---------------------------------------------------------------------------
# MuJoCo ego-view renderer for ``--source sim``
#
# Inlined (verbatim, head-camera subset) from the dataset-recorder's episode
# renderer so the sender stays self-contained: X2 head-camera mounting frames
# from the URDF, the URDF-rpy -> MuJoCo camera quaternion helper, and the
# per-frame render service. Spectator / world-fixed cameras and the static
# scene-XML path were not carried over (not used by the video sender).
# ---------------------------------------------------------------------------

from dataclasses import dataclass

MJCF_PATH = (
    REPO_ROOT / "gear_sonic" / "data" / "assets"
    / "robot_description" / "mjcf" / "x2_ultra.xml"
)


# Default floating-base pose for the offline dataset rendering path. Live
# callers pass real ``qpos[0:3]`` / ``qpos[3:7]`` so the rendered robot
# tips and translates the way it actually does in MuJoCo. Module-level
# constants keep ``render_frame`` allocation-free.
_DEFAULT_ROOT_POS_XYZ = np.array([0.0, 0.0, 0.793], dtype=np.float64)
_DEFAULT_ROOT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class HeadCameraSpec:
    """One head-mounted camera, parsed from the URDF and STL.

    The mounting frame fields ``pos`` and ``rpy_xyz`` come from the
    URDF joint origin. ``mesh_optical_axis_in_mesh_frame`` records
    which mesh axis is the optical forward; for the X2 head sensors
    that's the panel normal (mesh ``+Z``). Use :func:`build_camera_quat`
    to convert this spec into a MuJoCo camera quaternion.
    """

    name: str
    parent_link: str
    pos: tuple[float, float, float]
    rpy_xyz: tuple[float, float, float]
    fovy: float
    aliases: tuple[str, ...] = ()


# Camera mounting frames -- copy-pasted from
# gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra.urdf
# (lines 968-1029). Keeping them in this file rather than re-parsing
# the URDF at import time keeps the renderer dependency-light.
HEAD_CAMERAS: dict[str, HeadCameraSpec] = {
    "rgbd_head_front": HeadCameraSpec(
        name="rgbd_head_front",
        parent_link="head_pitch_link",
        pos=(0.05761, -0.011183, -0.04837),
        rpy_xyz=(2.2689, 0.0, 1.5708),
        # 60° vertical FoV approximates the AimDK RGB-D module
        # (Intel RealSense D435i family that the X2 head ships with).
        fovy=60.0,
        aliases=("ego_view", "rgbd"),
    ),
    "stereo_head_front": HeadCameraSpec(
        name="stereo_head_front",
        parent_link="head_pitch_link",
        pos=(0.067995, 0.029784, 0.05),
        rpy_xyz=(-1.5708, 0.0, -1.574),
        fovy=70.0,
        aliases=("stereo",),
    ),
    "rgb_head_center": HeadCameraSpec(
        name="rgb_head_center",
        parent_link="head_pitch_link",
        pos=(0.0684, -0.00021713, 0.05),
        rpy_xyz=(-1.5708, 0.0, -1.574),
        fovy=60.0,
        aliases=("rgb_center",),
    ),
    "rgb_head_rear": HeadCameraSpec(
        name="rgb_head_rear",
        parent_link="head_pitch_link",
        pos=(-0.0834, 0.00026495, 0.0),
        rpy_xyz=(-1.5708, 0.0, 1.5676),
        fovy=60.0,
        aliases=("rear",),
    ),
}


CameraSpec = HeadCameraSpec


def resolve_camera_spec(name_or_alias: str) -> CameraSpec:
    """Look up a head camera by canonical name or alias."""
    if name_or_alias in HEAD_CAMERAS:
        return HEAD_CAMERAS[name_or_alias]
    for spec in HEAD_CAMERAS.values():
        if name_or_alias in spec.aliases:
            return spec
    available = ", ".join(
        sorted(
            {n for spec in HEAD_CAMERAS.values()
             for n in (spec.name, *spec.aliases)}
        )
    )
    raise ValueError(
        f"Unknown camera name {name_or_alias!r}. Available: {available}"
    )


def build_camera_quat(rpy_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Compute a MuJoCo camera ``quat (wxyz)`` from a URDF mesh ``rpy``.

    The MuJoCo camera convention is ``-Z`` forward, ``+Y`` up. The X2
    URDF mounts every head sensor with a mesh whose ``+Z`` is the
    panel normal (the optical "forward" direction), so:

    1. Apply the URDF ``rpy`` (extrinsic XYZ) to get the panel normal
       in the parent link frame -- that is the desired ``-Z_cam``.
    2. Anchor ``+Y_cam`` to ``+Z`` of the parent link projected
       perpendicular to the look direction. This keeps the rendered
       image right-side-up regardless of the panel's roll.
    3. Build the rotation matrix ``[right | up | -look]`` and read off
       the quaternion.

    Returns: ``(w, x, y, z)`` -- MuJoCo's quaternion convention.
    """
    from scipy.spatial.transform import Rotation as R

    R_mesh = R.from_euler("xyz", list(rpy_xyz))
    look = R_mesh.apply([0.0, 0.0, 1.0])
    look /= np.linalg.norm(look)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(world_up, look)) > 0.999:
        # Degenerate: panel normal is nearly vertical. Fall back to
        # head_pitch_link +X (forward) as the "up reference" so the
        # cross product is well-defined.
        world_up = np.array([1.0, 0.0, 0.0])
    up = world_up - np.dot(look, world_up) * look
    up /= np.linalg.norm(up)

    right = np.cross(up, -look)
    right /= np.linalg.norm(right)

    M = np.column_stack([right, up, -look])
    q_xyzw = R.from_matrix(M).as_quat()
    return float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])


def add_camera_to_spec(spec, mjcf_camera: CameraSpec) -> None:
    """Programmatically add a named head camera to a loaded ``MjSpec``
    (rigidly attached to a head link with an explicit URDF rpy
    orientation).
    """
    if isinstance(mjcf_camera, HeadCameraSpec):
        parent = spec.body(mjcf_camera.parent_link)
        if parent is None:
            raise RuntimeError(
                f"parent body {mjcf_camera.parent_link!r} not found in MJCF"
            )
        cam = parent.add_camera()
        cam.name = mjcf_camera.name
        cam.pos = list(mjcf_camera.pos)
        cam.quat = list(build_camera_quat(mjcf_camera.rpy_xyz))
        cam.fovy = float(mjcf_camera.fovy)
        return

    raise TypeError(
        f"add_camera_to_spec: unsupported camera type {type(mjcf_camera).__name__}"
    )


def build_model_with_camera(
    camera: HeadCameraSpec,
    *,
    with_omnihand: bool = False,
    offwidth: int | None = None,
    offheight: int | None = None,
):
    """Load the X2 MJCF, attach the head camera, and optionally augment with OmniHand.

    Returns ``(model, layout, body_qposadr)``:

    * ``model``: the compiled :class:`mujoco.MjModel`.
    * ``layout``: a ``compose_x2_with_omnihand.HandQposLayout`` when
      ``with_omnihand`` is True, otherwise ``None``.
    * ``body_qposadr``: a length-31 ``np.ndarray[int64]`` mapping each
      slot of the canonical body trajectory (``X2_BODY_JOINT_NAMES``) to
      its ``qposadr`` in the compiled model.

    Why the per-name address table?  ``MjSpec.attach()`` inserts the
    OmniHand finger hinges immediately after the parent ``*_wrist_roll``
    joint -- which means in the augmented model the right-arm joints are
    pushed past the left-hand finger qpos slots.  Callers that assume
    ``qpos[7:38]`` is the 31 contiguous body slots silently corrupt the
    right arm into hand qpos slots and freeze the right arm in place.
    Returning a name-resolved address table forces the renderer to use
    per-joint addresses, regardless of how MuJoCo laid out the
    augmented model.
    """
    import mujoco

    from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
        X2_BODY_JOINT_NAMES,
    )

    if with_omnihand:
        from gear_sonic.scripts.compose_x2_with_omnihand import (
            build_x2_with_omnihand_spec,
        )

        spec, _, layout = build_x2_with_omnihand_spec()
        add_camera_to_spec(spec, camera)
        # MuJoCo's offscreen framebuffer defaults to 640x480; rendering at
        # higher resolutions silently fails on EGL with
        # ``ValueError: Image width N exceeds offwidth 640``. Resize the
        # spec's <visual><global offwidth=... offheight=.../> *before*
        # compile so the framebuffer matches the requested camera output.
        if offwidth is not None:
            spec.visual.global_.offwidth = int(offwidth)
        if offheight is not None:
            spec.visual.global_.offheight = int(offheight)
        model = spec.compile()
        if model is None:
            raise RuntimeError(
                "augmented (X2 + OmniHand) MJCF failed to compile after camera attach"
            )
    else:
        spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
        add_camera_to_spec(spec, camera)
        if offwidth is not None:
            spec.visual.global_.offwidth = int(offwidth)
        if offheight is not None:
            spec.visual.global_.offheight = int(offheight)
        model = spec.compile()
        layout = None

    body_qposadr = np.empty(len(X2_BODY_JOINT_NAMES), dtype=np.int64)
    for i, name in enumerate(X2_BODY_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(
                f"X2 body joint {name!r} missing from compiled model "
                f"(with_omnihand={with_omnihand})"
            )
        body_qposadr[i] = int(model.jnt_qposadr[jid])
    return model, layout, body_qposadr


class MujocoFrameRenderer:
    """Per-frame MuJoCo render service for the X2 + OmniHand spec.

    Build once, render N frames, close. Encapsulates the
    model+renderer+EGL setup.

    The renderer is purely kinematic: the floating base sits at the
    nominal stand pose ``(0, 0, 0.793)`` with identity orientation,
    body joints are written by *named* qposadr (the OmniHand augmented
    model fragments the contiguous body block), and finger mimic DOFs
    are projected on the fly via
    :func:`gear_sonic.scripts.compose_x2_with_omnihand.apply_active_hand_qpos`.
    """

    def __init__(
        self,
        *,
        camera: str | CameraSpec = "ego_view",
        width: int = 640,
        height: int = 480,
        with_omnihand: bool = True,
        egl: bool = True,
    ) -> None:
        if egl:
            os.environ.setdefault("MUJOCO_GL", "egl")

        import mujoco

        self._mujoco = mujoco
        self._cam_spec: CameraSpec = (
            camera if isinstance(camera, HeadCameraSpec)
            else resolve_camera_spec(str(camera))
        )
        self._with_omnihand = bool(with_omnihand)

        self._model, self._hand_layout, self._body_qposadr = (
            build_model_with_camera(
                self._cam_spec,
                with_omnihand=self._with_omnihand,
                offwidth=int(width),
                offheight=int(height),
            )
        )
        self._data = mujoco.MjData(self._model)
        self._cam_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, self._cam_spec.name
        )
        if self._cam_id < 0:
            raise RuntimeError(
                f"camera {self._cam_spec.name!r} did not survive MJCF compile"
            )

        self._width = int(width)
        self._height = int(height)
        self._renderer = mujoco.Renderer(self._model, height=self._height, width=self._width)

        self._apply_hand = None
        if self._with_omnihand:
            from gear_sonic.scripts.compose_x2_with_omnihand import (
                apply_active_hand_qpos as _apply,
            )
            self._apply_hand = _apply

    # ------------------------------------------------------------------
    # Read-only surface
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def with_omnihand(self) -> bool:
        return self._with_omnihand

    @property
    def camera_spec(self) -> CameraSpec:
        return self._cam_spec

    @property
    def body_qposadr(self) -> np.ndarray:
        """Length-31 ``int64`` array mapping each canonical body slot to a qposadr."""
        return self._body_qposadr

    # ------------------------------------------------------------------
    # Per-frame render
    # ------------------------------------------------------------------

    def render_frame(
        self,
        body_q: np.ndarray,
        *,
        left_active: np.ndarray | None = None,
        right_active: np.ndarray | None = None,
        root_pos_xyz: np.ndarray | None = None,
        root_quat_wxyz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render one frame and return a ``(H, W, 3)`` uint8 RGB array.

        Args:
            body_q: ``(31,)`` canonical X2 body joint vector
                (legs/waist/arms/head, MuJoCo joint order).
            left_active: ``(10,)`` active OmniHand joint vector for the
                left side. Required when ``with_omnihand=True``; ignored
                otherwise. Shape mismatches raise ``ValueError`` (via
                ``apply_active_hand_qpos``).
            right_active: ``(10,)`` active OmniHand joint vector for the
                right side. Same semantics as ``left_active``.
            root_pos_xyz: optional ``(3,)`` world-frame pelvis position.
                Defaults to ``(0, 0, 0.793)`` (the X2 nominal stand
                pose). Pass a live MuJoCo ``qpos[0:3]`` here to
                visualize translation.
            root_quat_wxyz: optional ``(4,)`` world-frame pelvis
                orientation in MuJoCo's ``wxyz`` order. Defaults to
                identity. Pass a live ``qpos[3:7]`` (or the deploy's
                ``base_quat`` from the ``x2_debug`` ZMQ stream) to
                visualize tilt / fall.

        Returns:
            ``np.ndarray`` of shape ``(self.height, self.width, 3)`` and
            dtype ``uint8``.
        """
        body_q = np.asarray(body_q, dtype=np.float64)
        if body_q.shape != (self._body_qposadr.shape[0],):
            raise ValueError(
                f"body_q must have shape ({self._body_qposadr.shape[0]},); got {body_q.shape}"
            )

        if root_pos_xyz is None:
            root_pos_xyz = _DEFAULT_ROOT_POS_XYZ
        else:
            root_pos_xyz = np.asarray(root_pos_xyz, dtype=np.float64).reshape(-1)
            if root_pos_xyz.shape != (3,):
                raise ValueError(
                    f"root_pos_xyz must have shape (3,); got {root_pos_xyz.shape}"
                )
        if root_quat_wxyz is None:
            root_quat_wxyz = _DEFAULT_ROOT_QUAT_WXYZ
        else:
            root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64).reshape(-1)
            if root_quat_wxyz.shape != (4,):
                raise ValueError(
                    f"root_quat_wxyz must have shape (4,); got {root_quat_wxyz.shape}"
                )
            n = float(np.linalg.norm(root_quat_wxyz))
            if n < 1e-9:
                root_quat_wxyz = _DEFAULT_ROOT_QUAT_WXYZ
            else:
                root_quat_wxyz = root_quat_wxyz / n

        d = self._data
        d.qpos[0] = float(root_pos_xyz[0])
        d.qpos[1] = float(root_pos_xyz[1])
        d.qpos[2] = float(root_pos_xyz[2])
        d.qpos[3] = float(root_quat_wxyz[0])
        d.qpos[4] = float(root_quat_wxyz[1])
        d.qpos[5] = float(root_quat_wxyz[2])
        d.qpos[6] = float(root_quat_wxyz[3])
        d.qpos[self._body_qposadr] = body_q

        if (
            self._with_omnihand
            and self._apply_hand is not None
            and self._hand_layout is not None
            and (left_active is not None or right_active is not None)
        ):
            self._apply_hand(
                d,
                self._hand_layout,
                left_active=left_active,
                right_active=right_active,
            )

        d.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, d)

        self._renderer.update_scene(d, camera=self._cam_id)
        return self._renderer.render()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the EGL render context. Idempotent."""
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except AttributeError:
                pass
            del self._renderer
            self._renderer = None  # type: ignore[assignment]

    def __enter__(self) -> "MujocoFrameRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class SimEgoSource:
    """SUB the deploy's x2_debug state stream; render ego_view
    (= rgbd_head_front) with the inlined MujocoFrameRenderer."""

    def __init__(
        self, *, host: str, port: int, topic: str,
        width: int, height: int,
    ) -> None:
        import zmq

        from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (
            unpack_message,
        )

        self._unpack = unpack_message
        self._topic = topic
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._sock.setsockopt(zmq.RCVHWM, 2)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sock.connect(f"tcp://{host}:{port}")
        self._renderer = MujocoFrameRenderer(
            camera="ego_view",
            width=width,
            height=height,
            egl=True,
        )
        log.info(
            "sim ego source: x2_debug SUB tcp://%s:%d, ego_view %dx%d",
            host, port, width, height,
        )

    def next_frame(self) -> np.ndarray | None:
        import zmq

        raw = None
        # Drain to the newest message; render at our own pace.
        while True:
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if raw is None:
            try:
                raw = self._sock.recv()  # blocking w/ RCVTIMEO
            except zmq.Again:
                return None
        try:
            msg = self._unpack(raw, expected_topic=self._topic)
        except ValueError:
            return None
        body_q = np.asarray(msg.fields.get("body_q", ()), dtype=np.float64).reshape(-1)
        if body_q.shape[0] != 31:
            return None
        base_quat = np.asarray(
            msg.fields.get("base_quat", (1.0, 0.0, 0.0, 0.0)), dtype=np.float64
        ).reshape(-1)
        if base_quat.shape[0] != 4:
            base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        base_pos = msg.fields.get("base_pos")
        root_pos = (
            np.asarray(base_pos, dtype=np.float64).reshape(-1)
            if base_pos is not None else None
        )
        return self._renderer.render_frame(
            body_q,
            root_pos_xyz=root_pos,
            root_quat_wxyz=base_quat,
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _local_ips() -> list[str]:
    try:
        return subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.split()
    except Exception:
        return []


def run(args: argparse.Namespace) -> int:
    transport = FramedTcpTransport(
        mode=args.mode,
        host=args.host if args.mode == "connect" else args.bind,
        port=args.port,
        exclude_ips=set(x for x in ([args.cam_host] + [ip for ip in _local_ips()]) if x),
    )
    # --sbs-duplicate: the XRoboToolkit client always splits the frame into
    # per-eye halves, so a MONO frame gives each eye half a picture (operator
    # 2026-09-07 on the first headset test: "i can't focus my eyes"). Send the
    # same image in both halves -> comfortable 2D view of a mono camera.
    sbs = bool(getattr(args, "sbs_duplicate", False))
    # --rotate180: the Orbbec (rgbd_head_front) raw stream is inverted (physical roll-180 mount).
    rot180 = bool(getattr(args, "rotate180", False))
    preview = bool(getattr(args, "preview", False))
    stamp = bool(getattr(args, "stamp", False))
    _cv2s = None
    if stamp:
        import cv2 as _cv2s
    preview_scale = float(getattr(args, "preview_scale", 2.0) or 1.0)
    preview_sized = False
    preview_xid = None   # X window id, resolved once by name (the title changes with --stamp)
    preview_last_q = 0.0
    preview_win_wh = None
    _cv2 = None
    _PREVIEW_WIN = "x2 cam -> headset"
    if preview:
        import cv2 as _cv2  # noqa: N813 — only needed for the local window
        # resizable window; highgui scales the image to the window (keep ratio). Maximize with the
        # window manager, 'f' toggles fullscreen, 'q' quits. Initial size = frame x --preview-scale.
        _cv2.namedWindow(_PREVIEW_WIN, _cv2.WINDOW_NORMAL | _cv2.WINDOW_KEEPRATIO | _cv2.WINDOW_GUI_NORMAL)
        preview_full = False
    encoder = FramedH264Encoder(
        width=args.width * 2 if sbs else args.width,
        height=args.height,
        fps=args.fps,
        bitrate_kbps=args.bitrate_kbps,
        gop=args.gop,
        encoder=args.encoder,
        on_access_unit=transport.send_packet,
    )

    if args.source == "test":
        source = TestPatternSource(args.width, args.height)
    elif args.source == "udp-cam":
        source = UdpCamSource(host=args.cam_host, port=args.cam_port, key=args.keys.split(",")[0],
                              width=args.width, height=args.height)
    elif args.source == "zmq-cam":
        source = ZmqCameraSource(
            host=args.cam_host, port=args.cam_port,
            keys=[k.strip() for k in args.keys.split(",") if k.strip()],
            width=args.width, height=args.height,
        )
    elif args.source == "sim":
        source = SimEgoSource(
            host=args.state_host, port=args.state_port, topic=args.state_topic,
            width=args.width, height=args.height,
        )
    else:
        raise ValueError(f"unknown source {args.source!r}")

    period = 1.0 / args.fps
    log.info(
        "streaming %s %dx%d@%g -> %s:%d (%s, %d kbps, gop %d)",
        args.source, args.width, args.height, args.fps,
        args.host if args.mode == "connect" else args.bind,
        args.port, args.encoder, args.bitrate_kbps, args.gop,
    )
    last_stat = time.monotonic()
    frames = 0
    backpressure_drops = 0
    t_src = t_prev = t_enc = 0.0; n_stage = 0
    last_frame = None; repeated = 0
    try:
        deadline = time.monotonic()
        while True:
            _t0 = time.monotonic()
            frame = source.next_frame()
            _t1 = time.monotonic(); t_src += _t1 - _t0
            # Constant output rate: when the camera is slower than --fps (Orbbec 10 Hz), repeat the
            # last frame so the headset decoder sees a steady stream. A decoder that buffers a fixed
            # number of frames turned 10 fps into ~9 s of lag (2026-09-08); duplicates are ~free in H.264.
            if frame is None and last_frame is not None:
                frame = last_frame; repeated += 1
            elif frame is not None:
                last_frame = frame
            if frame is not None:
                if rot180:
                    frame = np.ascontiguousarray(frame[::-1, ::-1])
                if stamp:
                    # latency ruler: the laptop clock burned into the frame; compare with the live clock in
                    # the preview title (or any clock) to read the true end-to-end delay in the headset
                    frame = np.ascontiguousarray(frame)
                    txt = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 10)}"
                    _cv2s.putText(frame, txt, (10, frame.shape[0] - 14), _cv2s.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, _cv2s.LINE_AA)
                    _cv2s.putText(frame, txt, (10, frame.shape[0] - 14), _cv2s.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, _cv2s.LINE_AA)
                if preview:
                    # local window with the camera image (before the side-by-side duplication,
                    # RGB -> BGR for cv2); resizable, 'f' toggles fullscreen, 'q' quits
                    shown = frame[:, :, ::-1]
                    if not preview_sized:
                        _cv2.resizeWindow(_PREVIEW_WIN, int(shown.shape[1] * preview_scale), int(shown.shape[0] * preview_scale))
                        preview_sized = True
                    # this highgui build paints the image at native size whatever the window flags, so
                    # scale the frame to the REAL window size (from the X server, polled every 0.5 s;
                    # OpenCV's own getWindowImageRect feeds back on itself and shrinks the image)
                    now_p = time.monotonic()
                    if now_p - preview_last_q > 0.5:
                        preview_last_q = now_p
                        try:
                            if preview_xid is None:
                                out0 = subprocess.run(["xwininfo", "-name", _PREVIEW_WIN], capture_output=True, text=True, timeout=1).stdout
                                preview_xid = next(l for l in out0.splitlines() if "Window id:" in l).split()[3]
                            out = subprocess.run(["xwininfo", "-id", preview_xid], capture_output=True, text=True, timeout=1).stdout
                            ww = int(next(l for l in out.splitlines() if "Width:" in l).split()[-1])
                            wh = int(next(l for l in out.splitlines() if "Height:" in l).split()[-1])
                            if ww > 0 and wh > 0: preview_win_wh = (ww, wh)
                        except Exception:
                            pass
                    if preview_win_wh:
                        ww, wh = preview_win_wh; sc = min(ww / shown.shape[1], wh / shown.shape[0])
                        fit = _cv2.resize(shown, (max(1, int(shown.shape[1] * sc)), max(1, int(shown.shape[0] * sc))), interpolation=_cv2.INTER_LINEAR)
                        canvas = np.full((wh, ww, 3), 32, dtype=fit.dtype)
                        y0 = (wh - fit.shape[0]) // 2; x0 = (ww - fit.shape[1]) // 2
                        canvas[y0:y0 + fit.shape[0], x0:x0 + fit.shape[1]] = fit; shown = canvas
                    if stamp:
                        _cv2.setWindowTitle(_PREVIEW_WIN, "x2 cam -> headset   laptop clock " + time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 10)}")
                    _cv2.imshow(_PREVIEW_WIN, shown)
                    k = _cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        raise KeyboardInterrupt
                    if k == ord("f"):
                        preview_full = not preview_full
                        _cv2.setWindowProperty(_PREVIEW_WIN, _cv2.WND_PROP_FULLSCREEN,
                                               _cv2.WINDOW_FULLSCREEN if preview_full else _cv2.WINDOW_NORMAL)
                _t2 = time.monotonic(); t_prev += _t2 - _t1
                if sbs:
                    frame = np.ascontiguousarray(np.hstack((frame, frame)))
                # Encode only when someone is watching: the encoder's GOP
                # clock keeps running, so a fresh client still gets an IDR
                # within gop/fps seconds of frames resuming.
                if transport.connected:
                    if transport.writable():
                        encoder.submit(frame)
                    else:
                        backpressure_drops += 1   # link full: skip this frame, keep latency bounded
                t_enc += time.monotonic() - _t2; n_stage += 1
                frames += 1
            now = time.monotonic()
            if now - last_stat >= 5.0:
                if hasattr(source, "last_stamp") and source.last_stamp:
                    # bridge-clock stamp vs laptop wall clock (PC2 measured +70 ms ahead on 2026-09-08)
                    log.info("cam: %d frames, %d incomplete, age of last frame %.0f ms (uncorrected clocks)",
                             source.frames, source.incomplete, (time.time() - source.last_stamp) * 1e3)
                log.info(
                    "frames=%d sent=%d dropped(no-client)=%d dropped(link-full)=%d connected=%s repeated=%d | ms/frame src %.0f preview %.0f encode %.0f",
                    frames, transport._sent, transport._dropped, backpressure_drops,
                    transport.connected, repeated, 1e3 * t_src / max(n_stage, 1), 1e3 * t_prev / max(n_stage, 1), 1e3 * t_enc / max(n_stage, 1),
                )
                t_src = t_prev = t_enc = 0.0; n_stage = 0
                last_stat = now
            deadline += period
            sleep = deadline - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        log.info("stopping")
        return 0
    finally:
        encoder.close()
        transport.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=("test", "zmq-cam", "udp-cam", "sim"), default="test")
    p.add_argument("--mode", choices=("listen", "connect"), default="listen")
    p.add_argument("--bind", default="0.0.0.0", help="listen-mode bind address")
    p.add_argument("--host", default="", help="connect-mode peer (headset) IP, or 'auto' = the peer of the XRoboToolkit PC Service socket, re-resolved until the headset app connects")
    p.add_argument("--port", type=int, default=12345,
                   help="video TCP port (XRoboToolkit convention: 12345)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--bitrate-kbps", type=int, default=4000)
    p.add_argument("--gop", type=int, default=15,
                   help="keyframe interval in frames (reference uses 15)")
    p.add_argument("--encoder", choices=("x264", "nvenc"), default="x264")
    p.add_argument("--stamp", action="store_true",
                   help="burn the laptop wall clock (HH:MM:SS.t) into every frame: a latency ruler for the headset")
    p.add_argument("--preview", action="store_true",
                   help="also show the outgoing frame in a local window on this PC (q closes and stops)")
    p.add_argument("--preview-scale", type=float, default=2.0,
                   help="preview window size = frame size x this (default 2.0 -> 1920x1080 for a 960x540 stream)")
    p.add_argument("--rotate180", action="store_true",
                   help="rotate every frame 180 deg (the Orbbec rgbd_head_front stream is inverted)")
    p.add_argument("--sbs-duplicate", action="store_true",
                   help="send the mono frame twice side by side (2*width) so the client's "
                        "per-eye split shows the full picture to both eyes (2D view)")
    # zmq-cam source
    p.add_argument("--cam-host", default="127.0.0.1")
    p.add_argument("--cam-port", type=int, default=5555)
    p.add_argument("--keys", default="head_front",
                   help="comma-separated image keys; two keys => side-by-side")
    # sim source
    p.add_argument("--state-host", default="127.0.0.1")
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--state-topic", default="x2_debug")
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    if args.mode == "connect" and not args.host:
        print("--mode connect requires --host", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
