import os
import queue
import sys
import threading
import time

import av
import numpy as np


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
    ):
        self.output_path = output_path
        self._first_frame = True

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        # Keep a handle so stop() can join the worker. Without this,
        # closing a writer leaves the worker blocked on queue.get()
        # forever; callers that churn through many short episodes
        # (e.g. the dataset segmenter) leak ~50-100 MB per writer
        # in orphaned thread state + libav codec contexts.
        self._worker_thread = threading.Thread(
            target=self._writer_worker, daemon=True
        )
        self._stopped = False
        self._worker_thread.start()

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height
        ), (
            f"Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. "
            f"Expected dimensions: {self.stream.width}x{self.stream.height}"
        )

    def add_frame(self, frame: np.ndarray) -> None:
        self._assert_dimensions(frame)
        self.queue.put(frame)

    def _writer_worker(self) -> None:
        while True:
            frame = self.queue.get()
            # A None sentinel from stop() means "drain done, exit".
            # Previously this was `continue` which silently leaked
            # the worker thread when the writer was retired.
            if frame is None:
                break
            self._assert_dimensions(frame)
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

            if self._first_frame:
                stderr_fd = sys.stderr.fileno()
                old_stderr = os.dup(stderr_fd)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stderr_fd)
                try:
                    packets = self.stream.encode(frame)
                    for packet in packets:
                        self.container.mux(packet)
                finally:
                    os.dup2(old_stderr, stderr_fd)
                    os.close(old_stderr)
                    os.close(devnull)
                    self._first_frame = False
            else:
                packets = self.stream.encode(frame)
                for packet in packets:
                    self.container.mux(packet)

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """Blocking call. Waits for queue to drain, flushes, and closes the container."""
        if self._stopped:
            return self.output_path
        # Signal the worker to exit after draining and join it. This
        # guarantees the worker is no longer touching self.stream or
        # self.container before we flush + close them (the old code
        # polled queue.empty() then immediately flushed, which had a
        # latent race against an in-flight encode call).
        self.queue.put(None)
        self._worker_thread.join()

        self._flush_stream()
        self.container.close()
        self._stopped = True
        return self.output_path

    def cancel(self) -> None:
        """Immediately stops writing and deletes the output file."""
        if self._stopped:
            return
        # Drain the worker before tearing down so we don't crash it
        # mid-encode by closing the container under it.
        try:
            self.queue.put(None)
            self._worker_thread.join(timeout=5.0)
        except Exception:
            pass
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        self.container.close()
        self._stopped = True

    def __del__(self) -> None:
        if getattr(self, "_stopped", False):
            return
        # Last-ditch cleanup if the caller forgot to stop() / cancel().
        try:
            self.queue.put_nowait(None)
            self._worker_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.container.close()
        except Exception:
            pass
