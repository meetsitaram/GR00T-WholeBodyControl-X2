"""PKL-driven ``planner_cmd`` publisher for the X2 kplanner stack.

Drop-in replacement for ``quest3_manager_x2.py``'s ZMQ publish role:
this script reads a MotionBricks PKL motion clip, computes the per-frame
4-D velocity intent ``(yaw_rate, vel_x, vel_z, hip_h)`` in body frame,
and publishes it on the ``planner_cmd`` topic at 50 Hz. The kplanner
daemon subscribes to the same wire as in production, so the only thing
that changes between Quest3 teleop and PKL replay is the *source* of the
velocity intent.

Sister scripts
--------------

This is the LIVE peer of the offline numeric-regression script
``motionbricks/scripts/replay_pkl_through_kplanner.py``. The naming is
deliberately distinct:

* ``replay_pkl_through_kplanner.py`` - offline. Calls
  ``NeuralPlannerCore.replan_with_velocity`` directly in-process and
  compares predicted vs ground-truth frames. Answers "did the model
  memorize the data?"
* ``x2_pkl_command_source.py`` (this file) - live. PUBs over ZMQ to
  the running ``x2_kplanner`` daemon while ``deploy_x2.sh sim`` is
  also running. Answers "does the full kplanner -> deploy chain walk?"

The two share the velocity-extraction helper
``_instant_intent_from_clip`` (kept identical in both scripts) so a
fix to the channel convention in one file lands in both.

Wire format
-----------

JSON payload on ``planner_cmd`` topic, identical to what
``quest3_manager_x2.py`` would emit, except we set ``target_velocity``
to the 4-element list extracted from the PKL. The kplanner's
``_zmq_command_thread`` parses this into ``LocomotionCommand.
direct_velocity`` which short-circuits ``intent_to_velocity`` and
returns the recorded velocity verbatim (no continuous shaping, no
runtime scales). See ``gear_sonic/scripts/x2_kplanner.py::
intent_to_velocity`` for the dispatch order.

Run from the repo root::

  gear_sonic/scripts/x2_pkl_command_source.py \\
      --pkl gear_sonic/data/motions/x2_demo_bank.pkl \\
      --clip-id walking_quip_360_R_002__A428 \\
      --bind tcp://*:5563 \\
      --rate-hz 50 \\
      --loop

The recommended way to invoke this is via ``run_x2_pkl_planner_stack.sh``
which spawns the deploy + kplanner first, then launches this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Clip -> qpos / rolling-window intent helpers (the same velocity-extraction
# logic the offline replay harness uses; any channel-convention fix must land
# in both consumers).
# ---------------------------------------------------------------------------


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        axis=-1,
    )


def _wxyz_to_yaw_rad(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> yaw (rad) around world +Z."""
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _build_clip_qpos(payload: dict) -> tuple[np.ndarray, float]:
    """Build (T, 38) MuJoCo qpos array for a motion-lib PKL clip.

    PKL layout:
        root_trans_offset : (T, 3) world m
        root_rot          : (T, 4) xyzw
        dof               : (T, 31) rad
        fps               : int
    """
    trans = np.asarray(payload["root_trans_offset"], dtype=np.float32)
    rot_xyzw = np.asarray(payload["root_rot"], dtype=np.float32)
    dof = np.asarray(payload["dof"], dtype=np.float32)
    if rot_xyzw.shape[-1] != 4:
        raise ValueError(f"root_rot has wrong last dim: {rot_xyzw.shape}")
    rot_wxyz = _quat_xyzw_to_wxyz(rot_xyzw)
    qpos = np.concatenate([trans, rot_wxyz, dof], axis=-1)
    return qpos.astype(np.float32), float(payload.get("fps", 30))


def _instant_intent_from_clip(
    qpos: np.ndarray,
    fps: float,
    frame_idx: int,
    window: int = 8,
) -> tuple[float, float, float, float]:
    """Compute a rolling-window intent at ``frame_idx``.

    Uses frames ``[frame_idx - window/2, frame_idx + window/2]`` so the
    intent reflects the velocity AROUND the current playback position,
    in body frame.
    """
    half = window // 2
    lo = max(0, frame_idx - half)
    hi = min(qpos.shape[0] - 1, frame_idx + half)
    if hi <= lo:
        return 0.0, 0.0, 0.0, float(qpos[frame_idx, 2])
    quat_wxyz = qpos[lo:hi + 1, 3:7]
    trans = qpos[lo:hi + 1, :3]
    yaw = np.unwrap(_wxyz_to_yaw_rad(quat_wxyz))
    duration_s = (hi - lo) / fps if fps > 0 else 1.0 / 30.0
    dyaw = float(yaw[-1] - yaw[0])
    dx_world = float(trans[-1, 0] - trans[0, 0])
    dy_world = float(trans[-1, 1] - trans[0, 1])
    yaw0 = float(yaw[0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    dx_body = c * dx_world - s * dy_world
    dy_body = s * dx_world + c * dy_world
    hip_h = float(trans[:, 2].mean())
    yaw_rate = dyaw / duration_s
    # Channel convention (matches NeuralPlannerCore._predict_with_velocity):
    #   vel_x = motion-rep X = MuJoCo body-Y = LATERAL  -> dy_body
    #   vel_z = motion-rep Z = MuJoCo body-X = FORWARD  -> dx_body
    vel_x_lateral = dy_body / duration_s
    vel_z_forward = dx_body / duration_s
    return yaw_rate, vel_x_lateral, vel_z_forward, hip_h


log = logging.getLogger("x2_pkl_command_source")


# ---------------------------------------------------------------------------
# Clip loader
# ---------------------------------------------------------------------------


def _pick_default_clip(raw: dict) -> str:
    """Heuristic pick: first key containing 'forward' (and not uppercase
    ``_M`` operator suffix). Falls back to the first key. Mirrors
    ``replay_pkl_through_kplanner.py``'s pick.
    """
    for k in raw.keys():
        if "forward" in str(k).lower() and "_M" not in str(k):
            return str(k)
    return str(next(iter(raw.keys())))


def _load_clip(
    pkl_path: Path,
    clip_id: Optional[str],
) -> tuple[np.ndarray, float, str]:
    """Load one clip from ``pkl_path``, returning (qpos[T, 38], fps, key)."""
    log.info("loading PKL: %s", pkl_path)
    raw = joblib.load(pkl_path)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{pkl_path}: expected dict of clips, got {type(raw).__name__}"
        )
    if clip_id is None:
        clip_id = _pick_default_clip(raw)
        log.info("auto-picked clip_id=%r (pass --clip-id to override)", clip_id)
    if clip_id not in raw:
        raise KeyError(
            f"clip {clip_id!r} not in {pkl_path}; "
            f"available: {list(raw.keys())[:10]}"
            + (" ..." if len(raw) > 10 else "")
        )
    qpos, fps = _build_clip_qpos(raw[clip_id])
    log.info(
        "clip loaded: key=%r frames=%d fps=%.2f duration=%.2fs",
        clip_id, qpos.shape[0], fps, qpos.shape[0] / max(fps, 1.0),
    )
    return qpos, fps, clip_id


# ---------------------------------------------------------------------------
# ZMQ publisher
# ---------------------------------------------------------------------------


def _build_payload(
    intent: tuple[float, float, float, float],
    frame_idx: int,
    clip_id: str,
) -> bytes:
    """Build the JSON ``planner_cmd`` payload the kplanner's
    ``_zmq_command_thread`` consumes.

    Required fields: ``intent``, ``magnitude`` (kplanner expects them
    even when ``target_velocity`` is also set; the dispatcher short-
    circuits on ``target_velocity`` and ignores the bucketed pair).
    Set them to ``"locomotion" / "raw_velocity"`` so a debug subscriber
    can distinguish PKL-replay payloads from Quest3 ones at a glance.
    """
    payload = {
        "intent": "locomotion",
        "magnitude": "raw_velocity",
        "target_velocity": [
            float(intent[0]),
            float(intent[1]),
            float(intent[2]),
            float(intent[3]),
        ],
        # Advisory fields (the kplanner ignores them; they're for
        # debug_planner_cmd / log readers).
        "frame_index": int(frame_idx),
        "clip_id": str(clip_id),
    }
    return json.dumps(payload).encode("utf-8")


def _spin(
    qpos: np.ndarray,
    fps: float,
    clip_id: str,
    bind: str,
    topic: str,
    rate_hz: float,
    start_frame: int,
    num_frames: Optional[int],
    loop: bool,
    velocity_window: int,
    print_every: int,
    stop_event: threading.Event,
    constant_intent: Optional[tuple[float, float, float, float]] = None,
    use_mean_intent: bool = False,
) -> int:
    """Main publish loop. Returns process exit code."""
    import zmq

    if rate_hz <= 0:
        log.error("--rate-hz must be > 0; got %f", rate_hz)
        return 1
    if start_frame < 0 or start_frame >= qpos.shape[0]:
        log.error(
            "--start-frame %d out of range [0, %d)",
            start_frame, qpos.shape[0],
        )
        return 1
    if num_frames is not None and num_frames <= 0:
        log.error("--num-frames must be > 0 (or omitted); got %d", num_frames)
        return 1
    end_frame = (
        min(qpos.shape[0], start_frame + num_frames)
        if num_frames is not None
        else qpos.shape[0]
    )
    n_emit = end_frame - start_frame

    # Resolve the per-tick intent strategy. The default is "per-frame
    # rolling-window intent" (production path used by the live wire),
    # but we expose two diagnostic overrides that pin the intent stream:
    #   * --constant-intent <yaw,vx,vz,hip> : every tick gets the same
    #     4-tuple. Used to isolate planner+deploy tracking from the
    #     source's per-frame oscillations.
    #   * --use-mean-intent : every tick gets the MEAN of the per-frame
    #     intents over the playback window. Same intent, lower noise.
    intent_override: Optional[tuple[float, float, float, float]] = None
    if constant_intent is not None:
        intent_override = (
            float(constant_intent[0]),
            float(constant_intent[1]),
            float(constant_intent[2]),
            float(constant_intent[3]),
        )
        log.info(
            "constant-intent override: yaw_rate=%+.3f vel_x=%+.3f "
            "vel_z=%+.3f hip_h=%.3f (per-frame stream IGNORED)",
            *intent_override,
        )
    elif use_mean_intent:
        # Walk the playback window once to compute the mean intent.
        vals = np.zeros((end_frame - start_frame, 4), dtype=np.float64)
        for i, src_idx in enumerate(range(start_frame, end_frame)):
            vals[i] = _instant_intent_from_clip(
                qpos, fps, src_idx, window=velocity_window,
            )
        m = vals.mean(axis=0)
        intent_override = (float(m[0]), float(m[1]), float(m[2]), float(m[3]))
        log.info(
            "mean-intent override: yaw_rate=%+.3f vel_x=%+.3f "
            "vel_z=%+.3f hip_h=%.3f (computed from window [%d, %d))",
            intent_override[0], intent_override[1], intent_override[2],
            intent_override[3], start_frame, end_frame,
        )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(bind)
    log.info(
        "ZMQ PUB bound on %s topic=%r rate=%.1f Hz frames=[%d, %d) "
        "(%d frames%s)",
        bind, topic, rate_hz, start_frame, end_frame, n_emit,
        " looping" if loop else "",
    )

    # Slow-joiner safeguard: SUB sockets need a moment after PUB bind to
    # complete the subscribe handshake; without this delay the first
    # ~50 ms of payloads are silently dropped. Matches the manager
    # pattern in quest3_manager_x2.py.
    settle_s = 0.5
    log.info(
        "settling %.2fs before first publish so the kplanner SUB can "
        "complete its handshake...",
        settle_s,
    )
    if stop_event.wait(timeout=settle_s):
        sock.close(linger=0)
        return 0

    period_s = 1.0 / rate_hz
    next_tick = time.monotonic()
    total_emitted = 0
    tx_loop = 0

    try:
        while not stop_event.is_set():
            for src_idx in range(start_frame, end_frame):
                if stop_event.is_set():
                    break
                if intent_override is not None:
                    intent = intent_override
                else:
                    intent = _instant_intent_from_clip(
                        qpos, fps, src_idx, window=velocity_window,
                    )
                payload = _build_payload(
                    intent, frame_idx=total_emitted, clip_id=clip_id,
                )
                # Multipart: [topic_bytes, payload_bytes] matches the
                # manager's send_multipart in _publish_planner_cmd.
                try:
                    sock.send_multipart(
                        [topic.encode("ascii"), payload],
                        flags=zmq.NOBLOCK,
                    )
                except zmq.Again:
                    log.warning("PUB queue full; dropping frame %d", src_idx)
                if (
                    print_every > 0
                    and total_emitted % print_every == 0
                ):
                    log.info(
                        "frame=%5d loop=%d src=%5d yaw_rate=%+.3f "
                        "vel_x=%+.3f vel_z=%+.3f hip_h=%.3f",
                        total_emitted, tx_loop, src_idx,
                        intent[0], intent[1], intent[2], intent[3],
                    )
                total_emitted += 1
                next_tick += period_s
                slack = next_tick - time.monotonic()
                if slack > 0:
                    if stop_event.wait(timeout=slack):
                        break
                else:
                    if -slack > 5 * period_s:
                        log.warning(
                            "publish loop fell behind by %.0f ms; resyncing",
                            -slack * 1000.0,
                        )
                        next_tick = time.monotonic()
            tx_loop += 1
            if not loop:
                break
            log.info(
                "loop complete (loop=%d total=%d); restarting",
                tx_loop, total_emitted,
            )
            # Continue without resetting next_tick; the cadence stays
            # locked to the same monotonic schedule.
    finally:
        log.info(
            "publish loop exiting; total_emitted=%d loops=%d",
            total_emitted, tx_loop,
        )
        # Drain any pending sends before close so the kplanner SUB
        # doesn't see a half-formed final payload.
        sock.close(linger=200)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="x2_pkl_command_source",
        description=(
            "Publish per-frame velocity intent from a recorded PKL motion "
            "clip onto the ``planner_cmd`` ZMQ wire that the x2_kplanner "
            "daemon subscribes to. Peer of quest3_manager_x2.py for "
            "isolation testing (replaces Quest3 + manager with a "
            "deterministic, replayable source)."
        ),
    )
    p.add_argument(
        "--pkl", type=Path, required=True,
        help="MotionBricks PKL containing dict[clip_id -> {root_trans_offset, "
             "root_rot, dof, fps, ...}].",
    )
    p.add_argument(
        "--clip-id", type=str, default=None,
        help="Clip key inside the PKL. Defaults to the first 'forward'-ish key.",
    )
    p.add_argument(
        "--start-frame", type=int, default=0,
        help="First frame index to publish (default 0).",
    )
    p.add_argument(
        "--num-frames", type=int, default=None,
        help="Number of frames to publish before stopping or looping. "
             "Default: publish all frames of the clip.",
    )
    p.add_argument(
        "--loop", action="store_true",
        help="Loop the clip continuously instead of stopping after one pass.",
    )
    p.add_argument(
        "--bind", type=str, default="tcp://*:5563",
        help="ZMQ bind URI (default tcp://*:5563, matches the kplanner stack's "
             "PLANNER_CMD_PORT).",
    )
    p.add_argument(
        "--topic", type=str, default="planner_cmd",
        help="ZMQ topic prefix (default 'planner_cmd', matches the kplanner's "
             "--zmq-cmd-topic default).",
    )
    p.add_argument(
        "--rate-hz", type=float, default=50.0,
        help="Publish rate (default 50 Hz, matches OUTPUT_FPS).",
    )
    p.add_argument(
        "--velocity-window", type=int, default=8,
        help="Rolling-window size (frames) used to compute the instantaneous "
             "velocity at each playback position. Larger = smoother but "
             "lags the source motion more. Default 8 (~0.27s at 30fps).",
    )
    p.add_argument(
        "--print-every", type=int, default=25,
        help="Print one diagnostic line every N emitted frames (default 25; "
             "set to 0 to silence).",
    )
    p.add_argument(
        "--constant-intent", type=str, default=None,
        help="DIAGNOSTIC: ignore the per-frame intent stream and publish a "
             "single fixed 4-tuple every tick. Format: 'yaw_rate,vel_x,"
             "vel_z,hip_h' (rad/s, m/s, m/s, m). Use to isolate kplanner+"
             "deploy tracking from the source's per-frame oscillations. "
             "Example: --constant-intent 0,0,0.5,0.7 (pure 0.5 m/s forward).",
    )
    p.add_argument(
        "--use-mean-intent", action="store_true",
        help="DIAGNOSTIC: pin every published tick to the MEAN of the "
             "per-frame intents over the playback window. Like the "
             "production path but with zero per-frame oscillation. "
             "Mutually exclusive with --constant-intent.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    if not args.pkl.is_file():
        log.error("--pkl not found: %s", args.pkl)
        return 1

    qpos, fps, clip_id = _load_clip(args.pkl, args.clip_id)
    if qpos.shape[0] < 2:
        log.error("clip %r has only %d frame(s); need >=2", clip_id, qpos.shape[0])
        return 1

    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("signal %d -> stopping", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    constant_intent: Optional[tuple[float, float, float, float]] = None
    if args.constant_intent is not None:
        parts = [p.strip() for p in args.constant_intent.split(",")]
        if len(parts) != 4:
            log.error(
                "--constant-intent must be 4 comma-separated floats "
                "'yaw_rate,vel_x,vel_z,hip_h'; got %r",
                args.constant_intent,
            )
            return 1
        try:
            constant_intent = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        except ValueError:
            log.error("--constant-intent values must be floats; got %r", parts)
            return 1
    if constant_intent is not None and args.use_mean_intent:
        log.error("--constant-intent and --use-mean-intent are mutually exclusive")
        return 1

    return _spin(
        qpos=qpos,
        fps=fps,
        clip_id=clip_id,
        bind=args.bind,
        topic=args.topic,
        rate_hz=args.rate_hz,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        loop=args.loop,
        velocity_window=args.velocity_window,
        print_every=args.print_every,
        stop_event=stop_event,
        constant_intent=constant_intent,
        use_mean_intent=args.use_mean_intent,
    )


if __name__ == "__main__":
    raise SystemExit(main())
