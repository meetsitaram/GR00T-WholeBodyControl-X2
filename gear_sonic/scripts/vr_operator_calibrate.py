"""VR operator calibration script.

Standalone CLI that boots the Quest 3 server, walks the operator
through three calibration poses (arms-down, T-pose, arms-forward),
captures stable wrist samples for each, fits a per-arm affine mapping,
and writes the result as a YAML file.

Usage::

    python -m gear_sonic.scripts.vr_operator_calibrate \\
        --output data/operator_calibrations/<id>.yaml \\
        --operator-id <id>

The output YAML can then be passed to teleop scripts via
``--calibration <path>`` to enable the stateless head-relative wrist
mapping (see :class:`gear_sonic.utils.teleop.vr_arm_teleop_v2.VRArmTeleopCalibrated`).

Workflow per pose
-----------------

1. The script sends a ``calibration_show_pose`` message to the WebXR
   client. The browser displays a stick-figure SVG and speaks the
   instruction line via TTS.
2. The operator gets into the pose and presses **A** on either
   controller.
3. The script samples wrist positions for ``--sample-window-s``
   seconds at the WS rate (~50 Hz). Stability is gated against
   ``--spread-threshold-m``: the 80th-percentile distance of samples
   from the median wrist position must be <= the threshold. This is
   robust to the per-frame mm-level tracking jitter that Quest 3
   produces at arm extension (which would inflate any velocity-based
   metric).
4. On a clean capture, the script flashes a "captured" overlay
   (``calibration_pose_captured``) and advances to the next pose.

When all three poses are captured, the script fits the calibration
(per-axis least squares; rejected if max residual > ``--reject-m``)
and writes the YAML.

Calibration is fully offline -- there's no MuJoCo, no IK, no SONIC
deploy. Just the Quest 3 server, the WebXR client, and a few hundred
samples of wrist data per pose.
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.operator_calibration import (  # noqa: E402
    CALIBRATION_POSE_IDS,
    CALIBRATION_POSE_INSTRUCTIONS,
    DEFAULT_POSE_RESIDUAL_REJECT_M,
    CalibrationFitResult,
    OperatorCalibration,
    PoseMeasurement,
    head_yaw_from_quat,
    try_fit_calibration,
    wrist_quat_to_head_yaw_frame,
    wrist_to_head_yaw_frame,
)
from gear_sonic.utils.teleop.vr.quest3_reader import Quest3Reader  # noqa: E402
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import (  # noqa: E402
    _is_controller_dropout,
    _is_twin_dropout,
)


# How long to display each pose card before the operator can press A,
# even if they're already in position. Gives TTS time to finish speaking
# without triggering an immediate capture from a left-over button-held
# state.
POSE_HOLD_GUARD_S = 1.0


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(q, dtype=np.float64) / n


def _antipodal_align(quats: np.ndarray) -> np.ndarray:
    """Flip rows so all quaternions are on the same hemisphere as quats[0].

    Quaternions are double-cover: ``q`` and ``-q`` describe the same
    rotation. For taking a per-component mean / median we must first
    align signs, otherwise a single sign-flip will yank the mean
    toward zero.
    """
    if quats.size == 0:
        return quats
    out = quats.copy()
    ref = out[0]
    for i in range(1, out.shape[0]):
        if float(np.dot(ref, out[i])) < 0.0:
            out[i] = -out[i]
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Guided VR operator calibration for X2 teleop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="YAML path to write. Defaults to "
             "data/operator_calibrations/<operator-id>.yaml.",
    )
    p.add_argument(
        "--operator-id", type=str, default="default",
        help="Free-form label stamped into the YAML file.",
    )
    p.add_argument(
        "--notes", type=str, default="",
        help="Optional free-form notes saved into the YAML "
             "(e.g., 'Quest 3 controllers, second attempt').",
    )
    p.add_argument("--quest3-ws-port", type=int, default=8765)
    p.add_argument("--quest3-http-port", type=int, default=8443)
    p.add_argument("--quest3-no-ssl", action="store_true")
    p.add_argument(
        "--sample-window-s", type=float, default=1.0,
        help="How long to sample wrist positions per pose (after A press).",
    )
    p.add_argument(
        "--spread-threshold-m", type=float, default=0.06,
        help="Per-arm cluster spread ceiling: the 80th-percentile "
             "distance of samples from the median wrist position must "
             "be <= this. Robust to per-frame Quest 3 tracking jitter "
             "(which is ~1-3 mm at arm extension and would inflate any "
             "velocity-based metric). Default 0.06 m (6 cm).",
    )
    # Hidden alias kept so older shell aliases / scripts that pass
    # ``--vel-threshold-mps`` don't break -- the value is accepted but
    # ignored. The frame-differenced velocity metric is fundamentally
    # noise-amplifying (1 mm jitter / 20 ms = 5 cm/s "velocity") so we
    # no longer use it as a gate; only as diagnostic logging.
    p.add_argument(
        "--vel-threshold-mps", type=float, default=None, help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--reject-m", type=float, default=None,
        help="UNIFORM per-arm fit-residual reject threshold (meters), "
             "applied to EVERY pose if set. Leave unset to use the "
             "per-pose defaults from "
             "operator_calibration.DEFAULT_POSE_RESIDUAL_REJECT_M, "
             "which are looser for ``namaste`` (palm-grip offset from "
             "controllers) than for the other poses.",
    )
    p.add_argument(
        "--namaste-reject-m", type=float, default=None,
        help="Override the namaste-pose residual threshold only "
             "(meters). Useful when the operator can hold the other "
             "three poses precisely but the controller-in-hand offset "
             "for namaste varies. Default: 0.18 m (18 cm). Ignored if "
             "``--reject-m`` is also set.",
    )
    p.add_argument(
        "--arms-down-reject-m", type=float, default=None,
        help="Override the arms-down-pose residual threshold (m). "
             "Default: 0.10 m. Ignored if ``--reject-m`` is set.",
    )
    p.add_argument(
        "--t-pose-reject-m", type=float, default=None,
        help="Override the T-pose residual threshold (m). "
             "Default: 0.10 m. Ignored if ``--reject-m`` is set.",
    )
    p.add_argument(
        "--arms-forward-reject-m", type=float, default=None,
        help="Override the arms-forward-pose residual threshold (m). "
             "Default: 0.10 m. Ignored if ``--reject-m`` is set.",
    )
    p.add_argument(
        "--max-retries-per-pose", type=int, default=3,
        help="How many times to let the operator retry a pose if the "
             "capture is rejected for being too jittery.",
    )
    p.add_argument(
        "--max-fit-recaptures", type=int, default=4,
        help="If the geometric fit is rejected (e.g. T-pose right arm "
             "angled forward instead of straight sideways), how many "
             "times to ask the operator to recapture the worst pose "
             "before giving up. The recapture targets only the "
             "specific pose that contributed the largest residual.",
    )
    return p.parse_args(argv)


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    return REPO_ROOT / "data" / "operator_calibrations" / f"{args.operator_id}.yaml"


def _resolve_residual_reject(args: argparse.Namespace) -> dict[str, float]:
    """Build the per-pose residual rejection dict from CLI flags.

    Precedence (highest first):
      1. ``--reject-m`` (uniform value applied to every pose).
      2. Per-pose ``--<pose>-reject-m`` overrides.
      3. :data:`DEFAULT_POSE_RESIDUAL_REJECT_M` baseline.

    Why not just pass ``args.reject_m or None`` straight through:
      * Banner-printing wants the resolved per-pose dict so the
        operator can SEE which thresholds will be enforced (catches
        accidental ``--reject-m 0.05`` typos that would have made
        the v2 calibration unfittable).
      * ``run_inline_calibration`` (called by the recorder /
        kinematic-teleop scripts) accepts a single ``reject_m``
        scalar OR dict, but the inline path benefits from getting
        the dict so its log output mirrors the standalone script.
    """
    if args.reject_m is not None:
        return {p: float(args.reject_m) for p in CALIBRATION_POSE_IDS}
    out = dict(DEFAULT_POSE_RESIDUAL_REJECT_M)
    overrides = {
        "namaste": args.namaste_reject_m,
        "arms_down": args.arms_down_reject_m,
        "t_pose": args.t_pose_reject_m,
        "arms_forward": args.arms_forward_reject_m,
    }
    for pose_id, val in overrides.items():
        if val is not None:
            out[pose_id] = float(val)
    return out


def _print_banner(
    args: argparse.Namespace,
    out_path: Path,
    reject_by_pose: dict[str, float],
) -> None:
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "<workstation-ip>"
    print("─" * 64, flush=True)
    print("  VR operator calibration  (X2 head-relative wrist mapping)", flush=True)
    print("─" * 64, flush=True)
    print(f"  operator_id    : {args.operator_id}", flush=True)
    print(f"  output         : {out_path}", flush=True)
    print(f"  Quest 3 URL    : https://{ip}:{args.quest3_http_port}", flush=True)
    print(f"  poses          : {' -> '.join(CALIBRATION_POSE_IDS)}", flush=True)
    print(f"  sample window  : {args.sample_window_s:.1f} s", flush=True)
    print(
        f"  spread thr.    : {args.spread_threshold_m*100:.1f} cm "
        f"(80th-pct distance from median)",
        flush=True,
    )
    print(f"  reject residual (per-pose):", flush=True)
    for pose_id in CALIBRATION_POSE_IDS:
        thr = reject_by_pose[pose_id]
        is_default = abs(thr - DEFAULT_POSE_RESIDUAL_REJECT_M[pose_id]) < 1e-9
        suffix = "" if is_default else "  (override)"
        print(f"    {pose_id:14s} {thr*100:5.1f} cm{suffix}", flush=True)
    print("─" * 64, flush=True)


def _wait_for_first_packet(quest: Quest3Reader, *, timeout_s: float = 120.0) -> None:
    """Block until the WebXR client has sent at least one tracking message."""
    print(
        "[calibrate] Waiting for Quest 3 WebSocket + first tracking packet …",
        flush=True,
    )
    t0 = time.monotonic()
    while True:
        if quest.is_connected and quest.get_3pt_pose() is not None:
            print("[calibrate] Quest 3 ready.", flush=True)
            return
        if time.monotonic() - t0 > timeout_s:
            raise SystemExit(
                "Timed out waiting for Quest 3. Open the WebXR page on the "
                "headset, press 'Connect WS', and try again."
            )
        time.sleep(0.1)


def _wait_for_button_edge(
    quest: Quest3Reader,
    *,
    button_idx: int,
    description: str,
    timeout_s: float | None = None,
    stop_flag: dict | None = None,
) -> None:
    """Block until the named button transitions from released -> pressed.

    Logs once per second so the operator knows the script is alive.
    """
    print(
        f"[calibrate] Press {description} on either Quest 3 controller "
        f"when ready.",
        flush=True,
    )
    prev = quest.get_buttons()[button_idx]
    last_log = 0.0
    t0 = time.monotonic()
    while True:
        if stop_flag and stop_flag.get("flag"):
            raise KeyboardInterrupt
        cur = quest.get_buttons()[button_idx]
        if cur and not prev:
            return
        prev = cur
        now = time.monotonic()
        if now - last_log >= 5.0:
            last_log = now
            print(
                f"[calibrate]   (still waiting for {description}; "
                f"buttons={quest.get_buttons()})",
                flush=True,
            )
        if timeout_s is not None and (now - t0) > timeout_s:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(0.02)


def _capture_pose(
    quest: Quest3Reader,
    *,
    pose_id: str,
    instructions: str,
    progress_label: str,
    sample_window_s: float,
    spread_threshold_m: float,
    max_retries: int,
    stop_flag: dict,
    audio_key: str | None = None,
) -> PoseMeasurement:
    """Drive the operator through one pose; return the captured measurement.

    ``audio_key`` overrides the default ``show_<pose_id>`` MP3 lookup
    on the WebXR client. Useful for recapture prompts, which the
    client should announce as ``recapture_<pose_id>`` rather than
    ``show_<pose_id>`` to make the audio cue distinct.
    """
    for attempt in range(1, max_retries + 1):
        suffix = "" if attempt == 1 else f"  (retry {attempt - 1}/{max_retries - 1})"
        msg = {
            "_type": "calibration_show_pose",
            "pose": pose_id,
            "instructions": instructions,
            "progress": progress_label + suffix,
        }
        if audio_key is not None:
            msg["audio_key"] = audio_key
        quest.send_message(msg)
        # Hold-guard: discourage immediate triggers from leftover button
        # state, give TTS time to finish.
        t_show = time.monotonic()
        while time.monotonic() - t_show < POSE_HOLD_GUARD_S:
            if stop_flag.get("flag"):
                raise KeyboardInterrupt
            time.sleep(0.02)

        try:
            _wait_for_button_edge(
                quest,
                button_idx=0,
                description=f"A (then hold {pose_id} pose for {sample_window_s:.1f}s)",
                stop_flag=stop_flag,
            )
        except KeyboardInterrupt:
            raise

        # Sample a window of wrist positions + quats in head-yaw frame.
        # Dropout frames (controller(s) lost tracking) are filtered out
        # since they contaminate both the mean position AND the RMS
        # velocity check (an origin-snapped wrist looks like a
        # 1.6-metre-per-tick velocity spike).
        samples_l: list[np.ndarray] = []
        samples_r: list[np.ndarray] = []
        quats_l: list[np.ndarray] = []
        quats_r: list[np.ndarray] = []
        dropouts = 0
        t_start = time.monotonic()
        while time.monotonic() - t_start < sample_window_s:
            if stop_flag.get("flag"):
                raise KeyboardInterrupt
            pose = quest.get_3pt_pose()
            if pose is None:
                time.sleep(0.01)
                continue
            l_drop = _is_controller_dropout(pose[0, :3], pose[0, 3:])
            r_drop = _is_controller_dropout(pose[1, :3], pose[1, 3:])
            twin_drop = _is_twin_dropout(pose[0, :3], pose[1, :3])
            if l_drop or r_drop or twin_drop:
                dropouts += 1
                time.sleep(0.01)
                continue
            head_pos = pose[2, :3]
            head_quat = pose[2, 3:]
            l = wrist_to_head_yaw_frame(pose[0, :3], head_pos, head_quat)
            r = wrist_to_head_yaw_frame(pose[1, :3], head_pos, head_quat)
            ql = wrist_quat_to_head_yaw_frame(pose[0, 3:], head_quat)
            qr = wrist_quat_to_head_yaw_frame(pose[1, 3:], head_quat)
            samples_l.append(l)
            samples_r.append(r)
            quats_l.append(ql)
            quats_r.append(qr)
            time.sleep(0.01)

        if len(samples_l) < 5:
            print(
                f"[calibrate] WARN: only {len(samples_l)} samples for {pose_id} "
                f"(dropouts skipped: {dropouts}); retrying. Make sure both "
                f"controllers are tracked.",
                flush=True,
            )
            continue

        L = np.asarray(samples_l)
        R = np.asarray(samples_r)

        # ── Stability check: cluster spread ──────────────────────────
        #
        # We only use the size of the cloud of samples. Frame-to-frame
        # velocity is NOT a useful metric here: Quest 3 inside-out
        # tracking has 1-3 mm of phantom jitter at arm extension (the
        # T-pose, where the controller is near the periphery of the
        # camera FOV). A 1.5 mm jitter on a 50 Hz feed reports as
        # ~7.5 cm/s "velocity" even when the operator is perfectly
        # still. The cluster spread (80th-percentile distance from
        # the median) is what physically matches "holding still": a
        # held wrist produces a 1-3 cm cloud over a 1-second window,
        # while a moving wrist produces a much larger one.
        #
        # The mean reported back as the calibration measurement uses
        # only inliers (samples within the spread radius of the
        # median), so a stray jitter spike doesn't bias it.
        med_l = np.median(L, axis=0)
        med_r = np.median(R, axis=0)
        dist_l = np.linalg.norm(L - med_l, axis=1)
        dist_r = np.linalg.norm(R - med_r, axis=1)
        spread_l = float(np.percentile(dist_l, 80))
        spread_r = float(np.percentile(dist_r, 80))

        thr_l = max(spread_l, 1e-3)
        thr_r = max(spread_r, 1e-3)
        inliers_l = dist_l <= thr_l
        inliers_r = dist_r <= thr_r
        mean_l = L[inliers_l].mean(axis=0) if inliers_l.any() else med_l
        mean_r = R[inliers_r].mean(axis=0) if inliers_r.any() else med_r

        # Diagnostic-only velocity (NOT used as a gate -- see comment
        # above; this just tells us how noisy the tracking was).
        elapsed = max(time.monotonic() - t_start, 1e-3)
        dt = elapsed / max(len(samples_l) - 1, 1)
        vel_l = np.linalg.norm(np.diff(L, axis=0), axis=1) / max(dt, 1e-3)
        vel_r = np.linalg.norm(np.diff(R, axis=0), axis=1) / max(dt, 1e-3)
        rms_l = float(np.sqrt(np.mean(vel_l**2))) if vel_l.size else 0.0
        rms_r = float(np.sqrt(np.mean(vel_r**2))) if vel_r.size else 0.0

        ok = (
            spread_l <= spread_threshold_m
            and spread_r <= spread_threshold_m
        )

        print(
            f"[calibrate] {pose_id}: n={len(samples_l)} "
            f"(dropouts skipped: {dropouts}), "
            f"L_spread={spread_l*100:.1f}cm  R_spread={spread_r*100:.1f}cm  "
            f"(threshold {spread_threshold_m*100:.0f}cm) -> {'OK' if ok else 'REJECT'}, "
            f"L_jitter={rms_l*1000:.1f}mm/frame  R_jitter={rms_r*1000:.1f}mm/frame "
            f"[diagnostic only], "
            f"L_wrist={mean_l}, R_wrist={mean_r}",
            flush=True,
        )

        if not ok:
            quest.send_message(
                {
                    "_type": "calibration_pose_captured",
                    "pose": pose_id,
                    "captured": False,
                    "audio_key": "moved_too_much",
                    "message": (
                        f"Wrist moved too much (L spread {spread_l*100:.0f} cm, "
                        f"R spread {spread_r*100:.0f} cm). Hold steadier."
                    ),
                    "progress": progress_label + suffix,
                }
            )
            time.sleep(1.5)
            continue

        quest.send_message(
            {
                "_type": "calibration_pose_captured",
                "pose": pose_id,
                "captured": True,
                "audio_key": "captured",
                "message": "Captured.",
                "progress": progress_label + suffix,
            }
        )
        time.sleep(0.5)
        # Mean wrist orientation in head-yaw frame. We take the median
        # in xyz (component-wise) after antipodal alignment to the first
        # sample, which is robust to small per-frame jitter without
        # introducing a SLERP-style averaging dependency. Restrict to
        # the same inlier mask used for the position mean so a stray
        # tracking spike doesn't contaminate the alignment quat.
        quats_l_arr = np.asarray(quats_l, dtype=np.float64)
        quats_r_arr = np.asarray(quats_r, dtype=np.float64)
        ql_in = quats_l_arr[inliers_l] if inliers_l.any() else quats_l_arr
        qr_in = quats_r_arr[inliers_r] if inliers_r.any() else quats_r_arr
        ql_arr = _antipodal_align(ql_in)
        qr_arr = _antipodal_align(qr_in)
        med_ql = _normalize_quat(np.median(ql_arr, axis=0))
        med_qr = _normalize_quat(np.median(qr_arr, axis=0))
        return PoseMeasurement(
            pose_id=pose_id,
            left_wrist_mean=mean_l,
            right_wrist_mean=mean_r,
            sample_count=len(samples_l),
            left_wrist_vel_rms_mps=rms_l,
            right_wrist_vel_rms_mps=rms_r,
            left_wrist_quat_head_yaw=med_ql,
            right_wrist_quat_head_yaw=med_qr,
        )

    raise SystemExit(
        f"Pose '{pose_id}' could not be captured after "
        f"{max_retries} attempts (operator was moving too much). "
        f"Re-run the calibration script when ready."
    )


def _capture_all_poses(
    quest: Quest3Reader,
    *,
    sample_window_s: float,
    spread_threshold_m: float,
    max_retries_per_pose: int,
    stop_flag: dict,
    log_prefix: str = "[calibrate]",
) -> dict[str, PoseMeasurement]:
    """Drive the operator through the 3 canonical poses in order."""
    measurements: dict[str, PoseMeasurement] = {}
    for i, pose_id in enumerate(CALIBRATION_POSE_IDS):
        progress_label = f"{i + 1}/{len(CALIBRATION_POSE_IDS)}"
        instructions = CALIBRATION_POSE_INSTRUCTIONS[pose_id]
        print(
            f"\n{log_prefix} === Pose {progress_label}: {pose_id} ===",
            flush=True,
        )
        measurements[pose_id] = _capture_pose(
            quest,
            pose_id=pose_id,
            instructions=instructions,
            progress_label=progress_label,
            sample_window_s=sample_window_s,
            spread_threshold_m=spread_threshold_m,
            max_retries=max_retries_per_pose,
            stop_flag=stop_flag,
        )
    return measurements


# Per-pose, per-side coaching strings shown when a fit residual is too
# high. We can't know the *exact* axis the operator was off on (the
# least-squares fit smears error across axes), but we know which pose
# was the largest contributor and which arm. The hints below are the
# common-cause checklist for each (pose, side) combination.
_RECAPTURE_HINTS: dict[str, dict[str, str]] = {
    "arms_down": {
        "left": "Hold the left arm fully relaxed straight down at your side; don't bend the elbow.",
        "right": "Hold the right arm fully relaxed straight down at your side; don't bend the elbow.",
    },
    "t_pose": {
        "left": "Stretch the left arm STRAIGHT SIDEWAYS at shoulder height; do NOT angle it forward.",
        "right": "Stretch the right arm STRAIGHT SIDEWAYS at shoulder height; do NOT angle it forward.",
    },
    "arms_forward": {
        "left": "Hold the left arm STRAIGHT FORWARD parallel to the right; arms shoulder-width apart.",
        "right": "Hold the right arm STRAIGHT FORWARD parallel to the left; arms shoulder-width apart.",
    },
    "namaste": {
        "left": "Bring both palms together at chest height with forearms vertical; controllers gently touching.",
        "right": "Bring both palms together at chest height with forearms vertical; controllers gently touching.",
    },
}


# Friendly pose names used in the "move out of X, into Y" preamble.
# We avoid the snake_case pose IDs in operator-facing text -- they
# read like internal symbols.
_POSE_FRIENDLY_NAME: dict[str, str] = {
    "arms_down": "arms-down",
    "t_pose": "T-pose",
    "arms_forward": "arms-forward",
    "namaste": "namaste",
}


def _coach_recapture_message(pose_id: str, side: str, residual_m: float) -> str:
    """Operator-facing coaching text for a single recapture.

    Why this is more verbose than just the hint
    -------------------------------------------
    The recapture loop runs AFTER all 4 poses have been captured,
    so the operator was just in (e.g.) namaste when the fit failed
    and we ask them to redo arms_down. Without the explicit "move
    out of ..., into ..." preamble, the operator hears the
    arms-down instructions immediately after finishing namaste and
    reasonably wonders why the script is talking about arms-down
    while they're still pressed in namaste.

    The "fit error was N cm" part also deserves a one-line reason:
    operators hear "12 cm error" and assume they did something
    badly wrong, when in fact the per-axis affine model has a hard
    floor around 10-15 cm for typical human/X2 anatomy -- it just
    needs a better-conditioned data point for that arm.
    """
    hint = _RECAPTURE_HINTS.get(pose_id, {}).get(side, "Hold the pose carefully.")
    friendly = _POSE_FRIENDLY_NAME.get(pose_id, pose_id.replace("_", " "))
    return (
        f"Step out of your last pose and back into the {friendly} pose: "
        f"the calibration fit had a {residual_m * 100:.0f} cm error on "
        f"the {side} arm, so we need a fresh capture for that arm only. "
        f"{hint}"
    )


def _fit_with_recapture(
    quest: Quest3Reader,
    measurements: dict[str, PoseMeasurement],
    *,
    operator_id: str,
    notes: str,
    sample_window_s: float,
    spread_threshold_m: float,
    reject_m: float | dict[str, float] | None,
    max_retries_per_pose: int,
    max_fit_recaptures: int,
    stop_flag: dict,
    log_prefix: str = "[calibrate]",
) -> tuple[OperatorCalibration | None, dict[str, PoseMeasurement], CalibrationFitResult]:
    """Fit calibration; on rejection, re-capture the worst pose and retry.

    ``reject_m`` accepts the same forms as
    :func:`try_fit_calibration` (``None``, ``float``, or per-pose
    ``dict``). The recapture loop treats whichever pose
    contributed the worst residual as the recapture target.

    Returns ``(calibration_or_None, updated_measurements, last_result)``.
    ``calibration_or_None`` is ``None`` if the recapture budget is
    exhausted with no acceptable fit.
    """
    result = try_fit_calibration(
        measurements,
        operator_id=operator_id,
        residual_reject_m=reject_m,
        notes=notes,
    )

    if result.accepted:
        return result.calibration, measurements, result

    for attempt in range(1, max_fit_recaptures + 1):
        if stop_flag.get("flag"):
            break

        # Pick the pose to recapture by walking up from "the pose that
        # actually exceeded its gate" (`rejected_pose`) -- NOT by the
        # absolute-largest residual (`worst_pose_overall`). With per-
        # pose thresholds those two can disagree: e.g. arms_down can
        # trip rejection at 11.5 cm (10 cm gate) while namaste sits
        # benignly at 14.9 cm (18 cm gate). Using "overall worst" in
        # that case would (a) print a self-contradictory log line
        # (residual N cm > namaste threshold 18 cm where N < 18) and
        # (b) coach the operator to recapture a pose that's already
        # within tolerance.
        target_pose = result.rejected_pose or result.worst_pose_overall()[0]
        target_side = result.rejected_side or "left"
        target_resid = (
            result.rejected_residual_m
            if result.rejected_residual_m is not None
            else result.per_pose_residual_m.get(target_pose, {}).get(target_side, 0.0)
        )
        target_threshold = result.residual_reject_m.get(target_pose, 0.0)

        msg = _coach_recapture_message(target_pose, target_side, target_resid)
        # Also surface the absolute-worst-overall residual when it's
        # different from the rejection target, so the operator can
        # see (e.g.) that namaste is at 14.9 cm even though the
        # rejected pose was arms_down at 11.5 cm. Helps diagnose
        # systematic capture issues that aren't yet bad enough to
        # gate.
        worst_pose, worst_side, worst_resid = result.worst_pose_overall()
        worst_overall_note = ""
        if worst_pose != target_pose or worst_side != target_side:
            worst_overall_note = (
                f" (worst-overall: {worst_pose} {worst_side} arm, "
                f"{worst_resid * 100:.1f} cm)"
            )

        print(
            f"{log_prefix} FIT REJECTED (attempt {attempt}/{max_fit_recaptures}): "
            f"{target_pose} {target_side} arm residual "
            f"{target_resid * 100:.1f} cm > threshold "
            f"{target_threshold * 100:.1f} cm{worst_overall_note}. "
            f"-> {msg}",
            flush=True,
        )

        # Per-pose residual breakdown (helps the operator + the log
        # reader understand why the fit was rejected).
        for pose_id in CALIBRATION_POSE_IDS:
            l = result.per_pose_residual_m.get(pose_id, {}).get("left", 0.0)
            r = result.per_pose_residual_m.get(pose_id, {}).get("right", 0.0)
            print(
                f"{log_prefix}   per-pose residual {pose_id:14s}  "
                f"L={l*100:5.1f} cm  R={r*100:5.1f} cm",
                flush=True,
            )

        # Send the coaching message to the headset (audio prompt +
        # overlay) so the operator hears AND sees it without taking
        # the headset off. The MP3 keyed off ``recapture_<pose>`` is
        # used so the audio cue is distinct from the initial pose.
        recapture_audio_key = f"recapture_{target_pose}"
        quest.send_message(
            {
                "_type": "calibration_show_pose",
                "pose": target_pose,
                "audio_key": recapture_audio_key,
                "instructions": msg,
                "progress": f"recapture {attempt}/{max_fit_recaptures}",
            }
        )
        # Give the audio prompt a moment to finish before we start
        # the hold-guard countdown for the recapture.
        time.sleep(2.5)

        progress_label = f"recapture {attempt}/{max_fit_recaptures}"
        try:
            measurements[target_pose] = _capture_pose(
                quest,
                pose_id=target_pose,
                instructions=msg,
                progress_label=progress_label,
                sample_window_s=sample_window_s,
                spread_threshold_m=spread_threshold_m,
                max_retries=max_retries_per_pose,
                stop_flag=stop_flag,
                audio_key=recapture_audio_key,
            )
        except SystemExit as exc:
            # The capture itself can still bail out if the operator
            # really can't hold steady. Surface that nicely instead of
            # propagating the SystemExit out through main().
            print(f"{log_prefix} recapture aborted: {exc}", flush=True)
            return None, measurements, result

        # Re-fit and loop.
        result = try_fit_calibration(
            measurements,
            operator_id=operator_id,
            residual_reject_m=reject_m,
            notes=notes,
        )
        if result.accepted:
            return result.calibration, measurements, result

    return None, measurements, result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Auto-install missing optional packages used by the calibration
    # audio prompts. No-op when already installed; falls back
    # gracefully (with a warning + speechSynthesis fallback) if pip
    # install fails. Runs BEFORE Quest3Reader.start() so the audio
    # cache generator can use gTTS on first launch. Calibration is
    # decoupled from the heavy recorder deps -- gtts only.
    from gear_sonic.utils.install import (
        CALIBRATION_DEPS,
        ensure_runtime_deps,
    )

    ensure_runtime_deps(
        CALIBRATION_DEPS,
        purpose="VR operator calibration (Quest 3 audio prompts)",
    )

    if args.vel_threshold_mps is not None:
        print(
            "[calibrate] WARNING: --vel-threshold-mps is deprecated and "
            "ignored. Stability is now gated on the cluster-spread "
            "metric (--spread-threshold-m, default 6 cm). Frame-to-frame "
            "velocity is too noisy at Quest 3's mm-level precision to "
            "use as a gate.",
            flush=True,
        )
    out_path = _resolve_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reject_by_pose = _resolve_residual_reject(args)
    _print_banner(args, out_path, reject_by_pose)

    stop_flag = {"flag": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"\n[calibrate] caught signal {signum}, stopping …", flush=True)
        stop_flag["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    quest = Quest3Reader(
        ws_port=args.quest3_ws_port,
        http_port=args.quest3_http_port,
        use_ssl=(not args.quest3_no_ssl),
        quiet_periodic=True,
    )
    quest.start()

    try:
        _wait_for_first_packet(quest)

        measurements = _capture_all_poses(
            quest,
            sample_window_s=args.sample_window_s,
            spread_threshold_m=args.spread_threshold_m,
            max_retries_per_pose=args.max_retries_per_pose,
            stop_flag=stop_flag,
        )

        if stop_flag["flag"]:
            return 130

        cal, measurements, last_result = _fit_with_recapture(
            quest,
            measurements,
            operator_id=args.operator_id,
            notes=args.notes,
            sample_window_s=args.sample_window_s,
            spread_threshold_m=args.spread_threshold_m,
            reject_m=reject_by_pose,
            max_retries_per_pose=args.max_retries_per_pose,
            max_fit_recaptures=args.max_fit_recaptures,
            stop_flag=stop_flag,
        )

        if cal is None:
            worst_pose, worst_side, worst_resid = last_result.worst_pose_overall()
            msg = (
                f"Calibration rejected after {args.max_fit_recaptures} "
                f"recapture attempts. Worst pose: {worst_pose} "
                f"({worst_side} arm, {worst_resid * 100:.1f} cm). "
                f"Either the controllers are losing tracking on that arm "
                f"or the pose is geometrically inconsistent with the others."
            )
            quest.send_message(
                {
                    "_type": "calibration_done",
                    "audio_key": "calibration_failed",
                    "message": msg,
                }
            )
            time.sleep(2.0)
            print(f"[calibrate] FAIL: {msg}", flush=True)
            return 2

        print("─" * 64, flush=True)
        print("[calibrate] FIT SUMMARY", flush=True)
        for side, f in cal.fit.items():
            print(
                f"  {side}: scale={f.scale}, translation={f.translation}, "
                f"residual={f.residual_m*100:.1f} cm",
                flush=True,
            )
        print("─" * 64, flush=True)

        cal.save_yaml(out_path)
        print(f"[calibrate] WROTE {out_path}", flush=True)

        quest.send_message(
            {
                "_type": "calibration_done",
                "audio_key": "calibration_saved",
                "message": (
                    f"Calibration saved. "
                    f"Left residual {cal.fit['left'].residual_m*100:.1f} cm, "
                    f"right {cal.fit['right'].residual_m*100:.1f} cm."
                ),
            }
        )
        time.sleep(2.5)
        return 0
    except KeyboardInterrupt:
        print("[calibrate] interrupted; no YAML written.", flush=True)
        return 130
    finally:
        try:
            quest.send_message({"_type": "calibration_hide"})
        except Exception:
            pass
        try:
            quest.stop()
        except Exception:
            pass


def run_inline_calibration(
    quest: Quest3Reader,
    *,
    output_path: Path,
    operator_id: str = "default",
    notes: str = "",
    sample_window_s: float = 1.0,
    spread_threshold_m: float = 0.06,
    reject_m: float | dict[str, float] | None = None,
    max_retries_per_pose: int = 3,
    max_fit_recaptures: int = 4,
):
    """Run calibration using an already-running ``Quest3Reader``.

    Used by the teleop scripts when ``--recalibrate`` is set, so the
    operator doesn't have to bounce out to a separate process. On a
    high-residual fit, the operator is asked (via TTS + overlay) to
    recapture only the worst-contributing pose -- up to
    ``max_fit_recaptures`` times. Returns the saved
    :class:`OperatorCalibration` on success; raises on
    interrupt or exhausted recapture budget.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stop_flag = {"flag": False}

    measurements = _capture_all_poses(
        quest,
        sample_window_s=sample_window_s,
        spread_threshold_m=spread_threshold_m,
        max_retries_per_pose=max_retries_per_pose,
        stop_flag=stop_flag,
        log_prefix="[calibrate-inline]",
    )

    cal, _, last_result = _fit_with_recapture(
        quest,
        measurements,
        operator_id=operator_id,
        notes=notes,
        sample_window_s=sample_window_s,
        spread_threshold_m=spread_threshold_m,
        reject_m=reject_m,
        max_retries_per_pose=max_retries_per_pose,
        max_fit_recaptures=max_fit_recaptures,
        stop_flag=stop_flag,
        log_prefix="[calibrate-inline]",
    )
    if cal is None:
        worst_pose, worst_side, worst_resid = last_result.worst_pose_overall()
        msg = (
            f"calibration rejected after {max_fit_recaptures} recapture "
            f"attempts. worst pose: {worst_pose} ({worst_side}, "
            f"{worst_resid * 100:.1f} cm)."
        )
        quest.send_message(
            {
                "_type": "calibration_done",
                "audio_key": "calibration_failed",
                "message": msg,
            }
        )
        time.sleep(2.0)
        quest.send_message({"_type": "calibration_hide"})
        raise ValueError(msg)

    cal.save_yaml(output_path)
    print(f"[calibrate-inline] wrote {output_path}", flush=True)
    quest.send_message(
        {
            "_type": "calibration_done",
            "audio_key": "calibration_saved",
            "message": (
                f"Calibration saved. "
                f"Left residual {cal.fit['left'].residual_m*100:.1f} cm, "
                f"right {cal.fit['right'].residual_m*100:.1f} cm."
            ),
        }
    )
    time.sleep(2.5)
    quest.send_message({"_type": "calibration_hide"})
    return cal


if __name__ == "__main__":
    raise SystemExit(main())
