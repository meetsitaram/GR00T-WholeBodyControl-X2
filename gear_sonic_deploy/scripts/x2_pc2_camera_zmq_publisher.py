#!/usr/bin/env python3
"""Bridge AgiBot HAL camera ROS topics → ``ComposedCameraClientSensor`` ZMQ.

Runs on PC2 (the Jetson Orin NX). Subscribes to one or more
``/aima/hal/sensor/.../rgb_image/compressed`` ROS 2 topics, optionally
resizes the JPEG frames, and re-publishes them as a single
``ImageMessageSchema`` payload over a ZMQ ``PUB`` socket.

This is the **server side** of the same wire protocol the existing
:class:`gear_sonic.camera.composed_camera.ComposedCameraClientSensor`
consumes. The X2 dataset recorder (laptop side) connects to it via
``ComposedCameraClientSensor(server_ip="<PC2_IP>", port=5555)``.

Default plumbing matches the four AgiBot head cameras that come up
after ``./gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal``:

* ``head_front``   ← ``/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed``
                     (Orbbec Gemini 335 RGB, native 2688×1944)
* ``stereo_left``  ← ``/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed``
                     (IMX900, native 2064×1552)
* ``stereo_right`` ← ``/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed``
                     (IMX900, native 2064×1552)

These mount-key names line up 1:1 with the
``observation.images.{head_front,stereo_left,stereo_right}`` feature
keys declared by
:func:`gear_sonic.data.features_x2_vla.get_features_x2_vla` when
``include_head_cameras=True``, so the recorder can drop the merged
``images`` dict straight into the LeRobot frame.

The rear camera is intentionally skipped by default (operator-facing,
not useful for VLA). Pass ``--include-rear`` to also publish it as
``head_rear``.

Frames are decoded → resized → re-encoded as MJPEG once per stream
(quality 85 by default) before being packed into the ZMQ payload.
Re-encode + resize takes ~1.5 ms per IMX900 frame on the Orin NX, well
inside the 50 Hz dataset tick budget. The whole bridge runs at the
slowest publisher's frame rate (~15 Hz for the HAL today).

Typical usage (from the laptop, via the helper script)::

    ./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve

…or directly on PC2::

    source /opt/ros/humble/setup.bash
    export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
    python3 /tmp/x2_pc2_camera_zmq_publisher.py --port 5555 --width 640 --height 480
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading

# Set while at least one ZMQ subscriber is connected (XPUB join/leave events). Workers skip all
# decode/resize/encode while it is clear, so the boot service costs ~nothing when nobody watches
# (measured 2026-09-08: 26 % of a core at 10 Hz when always-on).
ACTIVE = threading.Event()

# ---------------------------------------------------------------------------
# UDP fan-out (2026-09-08): TCP/ZMQ queues frames when the link is slower than the camera and a
# long-lived subscriber ends up seconds behind (measured 2-3 s on wifi). UDP never queues: each new
# frame goes out as a burst of MTU-sized datagrams the moment the camera callback has encoded it
# (no publish tick), to every peer that said hello in the last PEER_TTL seconds. A peer that
# misses a chunk drops that frame and shows the next one. Header (network order):
#   magic 'X2CF' | frame_id u32 | n_chunks u16 | chunk_idx u16 | stamp f64 | mount_key 16s (padded)
# ---------------------------------------------------------------------------
import socket
import struct

UDP_MAGIC = b"X2CF"
UDP_HDR = struct.Struct("!4sIHHd16s")
UDP_PAYLOAD = 1400 - UDP_HDR.size          # stays under the 1500-byte wifi MTU with IP/UDP headers
PEER_TTL = 3.0


class UdpFanout:
    def __init__(self, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512 * 1024)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self._peers: dict[tuple[str, int], float] = {}
        self._lock = threading.Lock()
        self._fid = 0
        self.sent_frames = 0
        self.dropped_send = 0

    def poll_hellos(self) -> int:
        """Register/refresh peers from their hello datagrams; expire silent ones. Returns live count."""
        now = time.monotonic()
        while True:
            try:
                _data, addr = self.sock.recvfrom(64)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            with self._lock:
                if addr not in self._peers:
                    print(f"[bridge] udp peer {addr[0]}:{addr[1]} joined", flush=True)
                self._peers[addr] = now
        with self._lock:
            gone = [a for a, t in self._peers.items() if now - t > PEER_TTL]
            for a in gone:
                del self._peers[a]
                print(f"[bridge] udp peer {a[0]}:{a[1]} left (silent {PEER_TTL:.0f} s)", flush=True)
            return len(self._peers)

    def emit(self, key: str, jpg: bytes, ts: float) -> None:
        with self._lock:
            peers = list(self._peers)
            self._fid = (self._fid + 1) & 0xFFFFFFFF
            fid = self._fid
        if not peers:
            return
        n = max(1, (len(jpg) + UDP_PAYLOAD - 1) // UDP_PAYLOAD)
        kb = key.encode()[:16].ljust(16, b"\0")
        for i in range(n):
            pkt = UDP_HDR.pack(UDP_MAGIC, fid, n, i, ts, kb) + jpg[i * UDP_PAYLOAD:(i + 1) * UDP_PAYLOAD]
            for addr in peers:
                try:
                    self.sock.sendto(pkt, addr)
                except (BlockingIOError, InterruptedError, OSError):
                    self.dropped_send += 1
        self.sent_frames += 1

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass
import time
from dataclasses import dataclass
from typing import Any

import cv2  # noqa: F401 — imported early; some ROS builds segfault otherwise
import msgpack
import numpy as np
import zmq

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamSpec:
    """A single ROS → ZMQ camera bridge stream."""

    mount_key: str
    """Key in the published ``ImageMessageSchema.images`` dict (e.g.
    ``"ego_view"``). Must match what the laptop-side recorder expects
    in :func:`gear_sonic.data.features_x2_vla.get_features_x2_vla`."""

    ros_topic: str
    """ROS 2 CompressedImage topic to subscribe to."""


DEFAULT_STREAMS: tuple[StreamSpec, ...] = (
    StreamSpec(
        mount_key="head_front",
        ros_topic="/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed",
    ),
    StreamSpec(
        mount_key="stereo_left",
        ros_topic="/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed",
    ),
    StreamSpec(
        mount_key="stereo_right",
        ros_topic="/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed",
    ),
)


REAR_STREAM = StreamSpec(
    mount_key="head_rear",
    ros_topic="/aima/hal/sensor/rgb_head_rear/rgb_image/compressed",
)

# The TRUE Orbbec Gemini 335 colour stream (1280x720; depth/pointcloud/IMU are
# siblings under rgbd_head_front/). NOT the default "head_front" above, which is
# the 5 MP centre unit (x2-groot-vla docs/CAMERAS.md). Opt in via --streams orbbec.
# Its raw stream is INVERTED (physical roll-180 mount) — consumers rotate 180.
ORBBEC_STREAM = StreamSpec(
    mount_key="orbbec",
    ros_topic="/aima/hal/sensor/rgbd_head_front/rgb_image/compressed",
)


# ---------------------------------------------------------------------------
# Per-stream worker
# ---------------------------------------------------------------------------


class _StreamWorker:
    """One ROS subscriber per stream, with its own latest-frame slot.

    Why one subscriber per stream (and not one node with many subs):
    we found empirically that on PC2's FastDDS profile a single
    multi-threaded executor with 4 subscriptions only ever gets
    scheduled for the first matched topic; per-process subscribers
    work robustly. See ``/tmp/x2_record_one_cam.py`` for the original
    test result.
    """

    def __init__(
        self,
        spec: StreamSpec,
        *,
        width: int,
        height: int,
        jpeg_quality: int,
        on_frame=None,
    ) -> None:
        self.spec = spec
        self._on_frame = on_frame
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_ts: float = 0.0
        self._n_received = 0
        self._n_dropped_decode = 0

        # Each worker gets its OWN rclpy Node so we can spin them in
        # separate threads / executors. rclpy supports multiple nodes
        # in one context as long as the names differ.
        self._node = Node(f"x2_pc2_camera_bridge_{spec.mount_key}")
        self._node.create_subscription(
            CompressedImage,
            spec.ros_topic,
            self._on_msg,
            self._build_qos(),
        )

    @staticmethod
    def _build_qos() -> QoSProfile:
        # Publishers use RELIABLE + TRANSIENT_LOCAL. Match RELIABLE on
        # our side so the QoS handshake actually completes (a
        # BEST_EFFORT subscriber against a RELIABLE publisher is
        # technically incompatible and FastDDS silently drops).
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

    def _on_msg(self, msg: CompressedImage) -> None:
        if not ACTIVE.is_set():
            return                      # idle: no subscriber -> no decode/resize/encode
        try:
            raw = bytes(msg.data)
            # Resize + re-encode at the bridge so the wire-side stays
            # constant regardless of native sensor resolution
            # (Orbbec is 2688×1944, IMX900 is 2064×1552).
            arr = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self._n_dropped_decode += 1
                return
            h, w = bgr.shape[:2]
            if (w, h) != (self.width, self.height):
                bgr = cv2.resize(
                    bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
            ok, jpg = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                self._n_dropped_decode += 1
                return
            jpg_bytes = jpg.tobytes()
            # Prefer the publisher's timestamp; fall back to wallclock
            # when the publisher left it empty.
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if ts == 0:
                ts = time.time()
            with self._lock:
                self._latest_jpeg = jpg_bytes
                self._latest_ts = ts
                self._n_received += 1
            if self._on_frame is not None:
                self._on_frame(self.spec.mount_key, jpg_bytes, ts)   # UDP: out the door now, no tick
        except Exception as exc:  # pragma: no cover — defensive
            print(
                f"[bridge:{self.spec.mount_key}] on_msg error: {exc}",
                flush=True,
            )
            self._n_dropped_decode += 1

    def snapshot(self) -> tuple[bytes | None, float]:
        with self._lock:
            return self._latest_jpeg, self._latest_ts

    def stats(self) -> tuple[int, int]:
        with self._lock:
            return self._n_received, self._n_dropped_decode

    def destroy(self) -> None:
        try:
            self._node.destroy_node()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Publisher main loop
# ---------------------------------------------------------------------------


def _spin_worker(worker: _StreamWorker, stop_event: threading.Event) -> None:
    """Spin a single rclpy node on its OWN executor until stop_event is set.

    Each thread needs its own executor; sharing rclpy's default global
    executor across threads triggers "generator already executing" errors
    because the underlying ``_wait_for_ready_callbacks`` generator is not
    re-entrant.
    """
    executor = SingleThreadedExecutor()
    executor.add_node(worker._node)
    try:
        while not stop_event.is_set():
            try:
                executor.spin_once(timeout_sec=0.05)
            except Exception as exc:  # pragma: no cover — defensive
                print(
                    f"[bridge:{worker.spec.mount_key}] spin error: {exc}",
                    flush=True,
                )
                time.sleep(0.1)
    finally:
        try:
            executor.remove_node(worker._node)
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass


def run(
    streams: list[StreamSpec],
    *,
    port: int,
    width: int,
    height: int,
    jpeg_quality: int,
    publish_rate_hz: float,
    bind_host: str,
    udp_port: int = 0,
) -> int:
    rclpy.init()
    udp = UdpFanout(udp_port) if udp_port > 0 else None
    stop_event = threading.Event()

    def _sig(sig: int, frame: Any) -> None:  # noqa: ARG001
        print(f"[bridge] signal {sig} → shutting down", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    workers = [
        _StreamWorker(
            spec, width=width, height=height, jpeg_quality=jpeg_quality,
            on_frame=(udp.emit if udp is not None else None),
        )
        for spec in streams
    ]
    spin_threads = [
        threading.Thread(target=_spin_worker, args=(w, stop_event), daemon=True)
        for w in workers
    ]
    for t in spin_threads:
        t.start()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.XPUB)            # XPUB = PUB + subscribe/unsubscribe events
    sock.setsockopt(zmq.XPUB_VERBOSE, 1)
    sock.setsockopt(zmq.SNDHWM, 20)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://{bind_host}:{port}")
    print(
        f"[bridge] zmq PUB bound tcp://{bind_host}:{port}, "
        f"streams={[s.mount_key for s in streams]}, "
        f"out={width}x{height}@q{jpeg_quality}, "
        f"publish_rate={publish_rate_hz:.1f}Hz"
        + (f", udp fan-out on :{udp_port} (hello to register, {PEER_TTL:.0f} s ttl)" if udp is not None else ""),
        flush=True,
    )

    period = 1.0 / max(publish_rate_hz, 1e-3)
    next_tick = time.monotonic()
    last_log = time.monotonic()
    last_log_counts = [w.stats()[0] for w in workers]
    sent = 0
    dropped_zmq = 0
    subs = 0

    try:
        while not stop_event.is_set():
            next_tick += period

            # subscriber joins/leaves (first byte 1 = subscribe, 0 = unsubscribe/disconnect)
            while True:
                try:
                    ev = sock.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if ev:
                    subs = subs + 1 if ev[0] == 1 else max(0, subs - 1)
            peers = udp.poll_hellos() if udp is not None else 0
            listeners = subs + peers
            if listeners > 0 and not ACTIVE.is_set():
                ACTIVE.set(); print(f"[bridge] listener connected (zmq {subs}, udp {peers}) -> streaming", flush=True)
            elif listeners == 0 and ACTIVE.is_set():
                ACTIVE.clear(); print("[bridge] no listeners -> idle (no decoding)", flush=True)

            images: dict[str, bytes] = {}
            timestamps: dict[str, float] = {}
            any_image = False
            for w in workers:
                jpg, ts = w.snapshot()
                if jpg is not None:
                    images[w.spec.mount_key] = jpg
                    timestamps[w.spec.mount_key] = ts
                    any_image = True

            if any_image and subs > 0:
                payload = {"timestamps": timestamps, "images": images}
                packed = msgpack.packb(payload, use_bin_type=True)
                try:
                    sock.send(packed, flags=zmq.NOBLOCK)
                    sent += 1
                except zmq.Again:
                    dropped_zmq += 1

            now = time.monotonic()
            if now - last_log >= 5.0:
                cur_counts = [w.stats()[0] for w in workers]
                rates = [
                    (c - p) / (now - last_log)
                    for c, p in zip(cur_counts, last_log_counts, strict=True)
                ]
                rate_str = ", ".join(
                    f"{w.spec.mount_key}={r:.1f}Hz"
                    for w, r in zip(workers, rates, strict=True)
                )
                print(
                    f"[bridge] subs={subs} udp_peers={peers} sent={sent} dropped_zmq={dropped_zmq} "
                    + (f"udp_frames={udp.sent_frames} udp_dropped={udp.dropped_send} " if udp is not None else "") + "| "
                    f"in_rates: {rate_str}",
                    flush=True,
                )
                last_log = now
                last_log_counts = cur_counts

            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # We're behind; reset baseline so we don't burst.
                next_tick = time.monotonic()
    finally:
        for w in workers:
            w.destroy()
        sock.close()
        ctx.term()
        if udp is not None:
            udp.close()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--udp-port", type=int, default=5556,
        help="UDP fan-out port (0 = off). Peers send any datagram here to receive frames; the"
             " camera callback sends each new frame as chunked datagrams immediately (no queue).",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ PUB port (default 5555 to match composed_camera convention).",
    )
    ap.add_argument(
        "--bind-host",
        type=str,
        default="*",
        help="Interface to bind ZMQ PUB on. Default '*' = all.",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=640,
        help="Output frame width after resize (matches the default render_width "
        "in features_x2_vla.py).",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output frame height after resize (matches the default "
        "render_height in features_x2_vla.py).",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG re-encode quality 1-100. 85 ≈ visually lossless at 640×480.",
    )
    ap.add_argument(
        "--publish-rate",
        type=float,
        default=50.0,
        help="ZMQ publish rate (Hz). Defaults to 50 to match the recorder tick. "
        "If a publisher streams slower than this rate, the bridge republishes "
        "the latest frame at the requested rate (recorder reads latest-only).",
    )
    ap.add_argument(
        "--include-rear",
        action="store_true",
        help="Also publish the rear head camera as the 'head_rear' stream "
        "(off by default; the recorder schema only declares the three "
        "front-facing slots).",
    )
    ap.add_argument(
        "--streams",
        type=str,
        default="",
        help="Comma list of mount keys to publish (head_front, stereo_left, "
        "stereo_right, head_rear); default = all. A headset ego-view over wifi "
        "wants head_front only: three 640x480 JPEGs per message at 50 Hz "
        "saturated the link (2026-09-07: ~4 fps arrived at the sender).",
    )
    ap.add_argument(
        "--dds-profile",
        type=str,
        default="/agibot/software/entry/cfg/ros_dds_configuration.xml",
        help="FastRTPS profile XML (auto-exported via FASTRTPS_DEFAULT_PROFILES_FILE "
        "before rclpy.init when not already set). Pass an empty string to skip.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dds_profile and "FASTRTPS_DEFAULT_PROFILES_FILE" not in os.environ:
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = args.dds_profile
        print(
            f"[bridge] FASTRTPS_DEFAULT_PROFILES_FILE = {args.dds_profile}",
            flush=True,
        )

    streams = list(DEFAULT_STREAMS)
    if args.include_rear:
        streams.append(REAR_STREAM)
    if args.streams:
        wanted = {k.strip() for k in args.streams.split(",") if k.strip()}
        candidates = list(DEFAULT_STREAMS) + [REAR_STREAM, ORBBEC_STREAM]
        streams = [s for s in candidates if s.mount_key in wanted]
        if not streams:
            raise SystemExit(f"--streams {args.streams!r} matched no stream")
    return run(
        streams,
        port=args.port,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
        publish_rate_hz=args.publish_rate,
        udp_port=args.udp_port,
        bind_host=args.bind_host,
    )


if __name__ == "__main__":
    sys.exit(main())
