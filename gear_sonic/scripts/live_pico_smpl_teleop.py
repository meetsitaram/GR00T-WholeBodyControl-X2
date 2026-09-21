#!/usr/bin/env python
"""LIVE whole-body Pico teleop -> SONIC smpl encoder -> X2 in MuJoCo.

The WHOLE_BODY mode from the plan doc, standalone-driver form: the operator's
full skeleton (Pico headset + waist + 2 ankle trackers, 5-sensor mode) drives
the policy DIRECTLY through the trained smpl encoder — no explicit retarget,
no stick commands; legs, torso and arms all follow the human.

*** SIM-ONLY. NEVER POINT THIS AT THE REAL ROBOT. ***
This is a research MuJoCo loop with NONE of the deploy safety stack: no
stable-stand fallback profile, no damp/e-stop integration, no target
clamps/LPF, no watchdog, no MC handoff. Robot-side whole-body teleop
requires SMPL ingestion inside the C++ deploy code (which also owns
stable-stand on disengage) — a separate work item, not this script.

Pipeline per 20 ms control tick:
  xrt body stream (poll thread, ~0.9 ms/frame conversion via the reference
  pico-manager pose_aa chain) -> 50 Hz ring of (smpl_joints_local,
  global_orient_quat) -> DELAY-BUFFER window: the encoder was trained with a
  0.2 s FUTURE lookahead, so live frame0 ("current") is the sample from
  0.18 s ago and frame9 is now — a fixed 0.18 s teleop latency by design.
  Root-ori term is re-anchored against the robot's live base quat each tick.
  Wrist term (robot-specific joint dofs the encoder expects): live proxy =
  the robot's own measured wrists tiled over the window (--wrist-mode
  measured|zero).

Safety: starts PAUSED holding the default stand (SPACE engages/disengages,
R resets); a stale body stream (>0.5 s) auto-freezes to held targets.

Headset-free regression: --tape-replay PATH routes the same live code path
through a recorded tape (TapeXrt), so the driver is testable end-to-end
without hardware; --headless --seconds N --qpos-dump OUT for automated gates.

Usage (live):
  .venv/bin/python -m gear_sonic.scripts.live_pico_smpl_teleop \
      --checkpoint ${CKPT_ROOT}/<run>/model_step_XXXX.pt   # see MODELS.md
Point --checkpoint at any 3-encoder .pt with a TRAINED smpl encoder (e.g.
the new-release SONIC ckpt); the frozen-G1 frozen-core merges have none.

NOTE on env: needs BOTH the sim stack (torch, mujoco — .venv) AND
xrobotoolkit_sdk for live mode. Live: run under .venv_teleop if it has
torch, else .venv works for --tape-replay; for live with .venv, the xrt
import must resolve (PYTHONPATH to the pybind build or install into .venv).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gear_sonic/scripts"))

import torch  # noqa: E402
import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from eval_x2_mujoco import (  # noqa: E402
    ACTION_SCALE, CONTROL_DT, DECIMATION, DEFAULT_DOF, IL_TO_MJ_DOF,
    JOINT_TO_ACTUATOR, KD, KP, MJCF_PATH, MJ_TO_IL_DOF, NUM_DOFS, SIM_DT,
    ProprioceptionBuffer, load_actor_from_checkpoint, quat_rotate_inverse,
)
from gear_sonic.scripts.pico_manager_thread_server import (  # noqa: E402
    compute_from_body_poses,
)
from gear_sonic.scripts.pico_tape_to_smpl_obs import SMPL_PARENTS  # noqa: E402

SMPL_WINDOW = 10
SMPL_DT = 0.02
DELAY_S = (SMPL_WINDOW - 1) * SMPL_DT  # 0.18 s design latency
X2_WRIST_IL = slice(25, 31)
ACTION_CLIP = 20.0


class LiveSmplSource:
    """Poll thread: xrt (or tape replay) -> 50 Hz ring of smpl frames."""

    def __init__(self, tape_replay: str | None = None, loop: bool = True):
        if tape_replay:
            from gear_sonic.utils.teleop.pico_tape import TapeXrt
            self.xrt = TapeXrt(tape_replay, loop=loop)
        else:
            import xrobotoolkit_sdk as xrt
            self.xrt = xrt
        self.xrt.init()
        self._grips = (0.0, 0.0)
        # Session capture (live mode): raw body frames in pico_tape format
        # so the offline pipeline (tape->SMPL->retarget->fine-tune corpus)
        # consumes live sessions directly. Tape replay never re-records.
        self.record = tape_replay is None
        self.rec_gate = False      # main sets True only while ENGAGED
        self._rec_stamps: list[int] = []
        self._rec_body: list[np.ndarray] = []
        self._rec_grips: list[tuple[float, float]] = []
        # FULL-SESSION recorder (operator 2026-09-05: "the replay has to
        # capture all the activity, not just something between engage/
        # disengage"): every device sample with EVERY control — body,
        # headset/controller poses, sticks, buttons, triggers, grips — in
        # the pico_tape field layout, so TapeXrt replays the whole session
        # (mode chord, stick walks, B toggles, grips) through the live
        # decoder. X long-press starts it, Y long-press stops it (sender).
        self.full_on = False
        self._full: dict[str, list] = {k: [] for k in
                                       ("stamp_ns", "body", "body_ok", "headset", "lctrl",
                                        "rctrl", "axes", "buttons", "analog")}
        self._ring: deque = deque(maxlen=128)  # (t_mono, joints(24,3), quat_wxyz)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_stamp = None
        # Body-stream freshness (2026-09-01): the Pico SDK can freeze the
        # body solve (cached pose re-served, or stamps stop) while grips
        # and controllers stay live. Nothing downstream could tell -- a
        # full sim session tracked one byte-identical, 26deg-pitched frame
        # for 25 minutes and fell at every engage. Track (a) when we last
        # APPENDED a frame and (b) whether recent appends are IDENTICAL,
        # so callers can refuse to treat a frozen mannequin as an operator.
        self._last_append_t = 0.0
        self._prev_body = None
        self._same_streak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        grip_err = False
        while not self._stop.is_set():
            try:
                self._grips = (float(self.xrt.get_left_grip()),
                               float(self.xrt.get_right_grip()))
            except Exception as e:
                if not grip_err:
                    print(f"[grips] poll failed ({e!r}) — grip gesture "
                          "unavailable, use SPACE in the viewer", flush=True)
                    grip_err = True
            stamp = self.xrt.get_time_stamp_ns()
            if self.full_on and stamp and stamp != self._full_last_stamp:
                self._full_last_stamp = stamp
                try:
                    self._full_append(int(stamp))
                except Exception as e:
                    print(f"[session] full recorder poll failed ({e!r}) — stopping it", flush=True)
                    self.full_on = False
            if stamp and stamp != self._last_stamp and self.xrt.is_body_data_available():
                self._last_stamp = stamp
                body = np.asarray(self.xrt.get_body_joints_pose(), np.float64)
                if body.shape == (24, 7):
                    if self._prev_body is not None and np.array_equal(body, self._prev_body):
                        self._same_streak += 1
                    else:
                        self._same_streak = 0
                    self._prev_body = body.copy()
                    self._last_append_t = time.monotonic()
                    if self.record and self.rec_gate:
                        self._rec_stamps.append(int(stamp))
                        self._rec_body.append(body.astype(np.float32))
                        self._rec_grips.append(self._grips)
                    r = compute_from_body_poses(SMPL_PARENTS, "cpu", body)
                    with self._lock:
                        self._ring.append((
                            time.monotonic(),
                            r["smpl_joints_local"][0].numpy().astype(np.float32),
                            r["global_orient_quat"][0].numpy().astype(np.float32),
                            # local wrist rotvecs [L, R] in the forearm frame
                            # (SMPL joints 20/21) -> X2 wrist pitch/roll
                            np.asarray(r["pose_aa"][[20, 21]], np.float32),
                            # local neck/head rotvecs (SMPL joints 12/15)
                            # -> X2 head yaw (pico_intent_sender head_targets,
                            # 2026-09-08)
                            np.asarray(r["pose_aa"][[12, 15]], np.float32),
                        ))
            time.sleep(0.004)

    # -- full-session recorder ------------------------------------------------
    _full_last_stamp = None

    def _full_append(self, stamp: int) -> None:
        x = self.xrt; f = self._full
        body_ok = bool(x.is_body_data_available())
        body = (np.asarray(x.get_body_joints_pose(), np.float32) if body_ok
                else np.zeros((24, 7), np.float32))
        lx, ly = x.get_left_axis(); rx, ry = x.get_right_axis()
        f["stamp_ns"].append(stamp); f["body"].append(body); f["body_ok"].append(body_ok)
        f["headset"].append(np.asarray(x.get_headset_pose(), np.float32))
        f["lctrl"].append(np.asarray(x.get_left_controller_pose(), np.float32))
        f["rctrl"].append(np.asarray(x.get_right_controller_pose(), np.float32))
        f["axes"].append(np.asarray([lx, ly, rx, ry], np.float32))
        f["buttons"].append(np.asarray(
            [x.get_A_button(), x.get_B_button(), x.get_X_button(), x.get_Y_button(),
             x.get_left_axis_click(), x.get_right_axis_click()], np.uint8))
        f["analog"].append(np.asarray(
            [x.get_left_trigger(), x.get_right_trigger(), x.get_left_grip(), x.get_right_grip()],
            np.float32))

    def start_full(self) -> None:
        for v in self._full.values():
            v.clear()
        self._full_last_stamp = None
        self.full_on = True

    def full_frames(self) -> int:
        return len(self._full["stamp_ns"])

    def stop_full(self, path, **meta) -> int:
        """Write the full-session tape (pico_tape layout + metadata). Returns count."""
        self.full_on = False
        n = len(self._full["stamp_ns"])
        if n == 0:
            return 0
        arrays = {k: np.asarray(v) for k, v in self._full.items()}
        for v in self._full.values():
            v.clear()
        # Write in the BACKGROUND: a compressed save of a long record stalls the
        # caller (the sender's 50 Hz loop) for ~1 s -> intent + hand streams pause
        # -> the kplanner's arm-ingest link times out and the hands drop to the
        # wire default (2026-09-05: "why is Y opening the fingers?").
        def _write() -> None:
            try:
                np.savez_compressed(path, **arrays, full_session=True, **meta)
                print(f"[session] record written: {path} ({n} samples)", flush=True)
            except Exception as e:
                print(f"[session] record write FAILED: {path} ({e!r})", flush=True)
        threading.Thread(target=_write, name="session-record-write", daemon=False).start()
        return n

    def body_fresh(self, now: float, max_age_s: float = 1.2,
                   max_same: int = 50) -> bool:
        """True iff the body stream is ALIVE: a frame appended within
        max_age_s AND the last max_same appends were not all identical.
        max_age_s 1.2 (2026-09-01, was 0.5): real Wi-Fi drops body frames
        in sub-second bursts; riding them out means the robot holds the
        last human pose briefly -- benign -- instead of a mid-session
        forced disengage. True freezes are still caught by the streak.
        (max_same=50 ~ 1 s of a genuinely frozen solve; a real human is
        never bit-identical between solves.) Tape replay counts as fresh
        by construction -- tapes advance their own frames."""
        if self._last_append_t <= 0.0:
            return False
        return (now - self._last_append_t) < max_age_s \
            and self._same_streak < max_same

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def save_session(self, path, clear: bool = False) -> int:
        """Write captured frames as a pico_tape-compatible npz. Returns count."""
        n = len(self._rec_stamps)
        if not n:
            return 0
        stamps, bodies, grips = (self._rec_stamps, self._rec_body,
                                 self._rec_grips)
        if clear:
            self._rec_stamps, self._rec_body, self._rec_grips = [], [], []
        analog = np.zeros((n, 4), np.float32)
        analog[:, 2:] = np.asarray(grips, np.float32)
        np.savez_compressed(
            path,
            stamp_ns=np.asarray(stamps, np.int64),
            body=np.stack(bodies),
            body_ok=np.ones(n, bool),
            headset=np.zeros((n, 7), np.float32),
            lctrl=np.zeros((n, 7), np.float32),
            rctrl=np.zeros((n, 7), np.float32),
            axes=np.zeros((n, 4), np.float32),
            buttons=np.zeros((n, 6), bool),
            analog=analog,
        )
        return n

    def grips(self) -> tuple[float, float]:
        return self._grips

    def age_s(self) -> float:
        with self._lock:
            if not self._ring:
                return float("inf")
            return time.monotonic() - self._ring[-1][0]

    def latest_wrist_aa(self) -> "np.ndarray | None":
        """Latest local wrist rotations, (2, 3) rotvec [left, right] in the
        forearm (elbow-joint) frame, from the Pico body tracking. None
        until the first frame (or for sources that carry no rotations)."""
        with self._lock:
            if not self._ring or len(self._ring[-1]) < 4:
                return None
            return self._ring[-1][3].copy()

    def latest_head_aa(self) -> "np.ndarray | None":
        """Latest local neck + head rotations, (2, 3) rotvec [neck(12),
        head(15)] in SMPL parent frames, from the Pico body tracking.
        None until the first frame (or for sources without rotations)."""
        with self._lock:
            if not self._ring or len(self._ring[-1]) < 5:
                return None
            return self._ring[-1][4].copy()

    def window(self, now: float) -> tuple[np.ndarray, np.ndarray] | None:
        """(joints (10,24,3), quats (10,4)) sampled at now-DELAY..now."""
        with self._lock:
            ring = list(self._ring)
        if len(ring) < 4:
            return None
        ts = np.array([r[0] for r in ring])
        want = now - DELAY_S + np.arange(SMPL_WINDOW) * SMPL_DT
        idx = np.searchsorted(ts, want, side="right") - 1
        idx = idx.clip(0, len(ring) - 1)
        joints = np.stack([ring[i][1] for i in idx])
        quats = np.stack([ring[i][2] for i in idx])
        return joints, quats


class SmplFileSource:
    """Direct SMPL ingestion: drive the smpl encoder from an SMPL file,
    bypassing the Pico body-stream conversion entirely.

    Accepts BOTH sidecar conventions:
      * smpl_filtered-format .pkl (source-corpus / pico fine-tune sidecars):
        ``pose_aa (T,72)`` raw SMPL (Y-up, base rot included) + ``fps``.
        Converted here through the SAME canonical chain the live path uses
        (``process_smpl_joints``): root aa -> quat -> y-up->z-up ->
        FK joints -> base-rot removal -> root-local joints.
      * pico_tape_to_smpl_obs .npz: ``smpl_joints_local`` +
        ``global_orient_quat_wxyz`` (already canonical) + ``fps``.

    Dataset ABSOLUTE heading does not need to match anything: the driver
    re-anchors human->robot yaw at engage, exactly as for a live operator.
    (Within-clip heading DYNAMICS are preserved — that is what the policy
    tracks.) Serves wall-clock-paced windows with the live ring's
    delay-buffer semantics; holds the final frame after the clip ends.
    """

    def __init__(self, path: str):
        self.record = False
        self.rec_gate = False
        if path.endswith(".npz"):
            z = np.load(path)
            joints = np.asarray(z["smpl_joints_local"], np.float32)
            quats = np.asarray(z["global_orient_quat_wxyz"], np.float32)
            fps = float(z["fps"])
        else:
            import joblib
            from gear_sonic.isaac_utils.rotations import (
                remove_smpl_base_rot, smpl_root_ytoz_up)
            from gear_sonic.trl.utils.torch_transform import (
                angle_axis_to_quaternion, compute_human_joints, quat_apply,
                quat_inv, quaternion_to_angle_axis)
            d = joblib.load(path)
            pa = torch.from_numpy(np.asarray(d["pose_aa"], np.float32))
            fps = float(d.get("fps", 50.0))
            if float(pa[:, 69:].abs().max()) == 0.0:
                raise ValueError(
                    f"{path}: pose_aa[:, 69:] is all-zero — this is a LEGACY "
                    "pico sidecar with [body(69)|pad(3)] layout (root "
                    "orientation MISSING; pre-2026-08-26 "
                    "pico_tape_to_smpl_obs bug). Regenerate the sidecar, or "
                    "replay the session npz instead.")
            root_q_z = smpl_root_ytoz_up(
                angle_axis_to_quaternion(pa[:, :3]))          # z-up, base in
            joints_w = compute_human_joints(
                pa[:, 3:66], quaternion_to_angle_axis(root_q_z))  # (T,24,3)
            root_q = remove_smpl_base_rot(root_q_z, w_last=False)
            inv = quat_inv(root_q).unsqueeze(1).repeat(1, joints_w.shape[1], 1)
            joints = quat_apply(inv, joints_w).numpy().astype(np.float32)
            quats = root_q.numpy().astype(np.float32)
        if abs(fps * SMPL_DT - 1.0) > 1e-6:   # resample to the 50 Hz ring rate
            n = int(round(len(joints) / (fps * SMPL_DT)))
            idx = np.minimum((np.arange(n) * fps * SMPL_DT).astype(int),
                             len(joints) - 1)
            joints, quats = joints[idx], quats[idx]
        self._joints, self._quats = joints, quats
        self._t0 = None
        self._end_announced = False
        print(f"[smpl-replay] {path}: {len(joints)} frames "
              f"({len(joints) * SMPL_DT:.1f}s @ 50 Hz, src {fps:.0f} Hz)",
              flush=True)

    def grips(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def age_s(self) -> float:
        return 0.0

    def stop(self) -> None:
        pass

    def save_session(self, path, clear: bool = False) -> int:
        return 0

    def window(self, now: float) -> tuple[np.ndarray, np.ndarray] | None:
        if self._t0 is None:
            self._t0 = now
        i = int((now - self._t0) / SMPL_DT)
        if i >= len(self._joints) and not self._end_announced:
            print(f"[smpl-replay] clip ended at t={len(self._joints) * SMPL_DT:.1f}s "
                  "— holding final frame.", flush=True)
            self._end_announced = True
        idx = np.clip(np.arange(i - SMPL_WINDOW + 1, i + 1),
                      0, len(self._joints) - 1)
        return self._joints[idx], self._quats[idx]


_CUE_DIR = None


def _audio_cue(kind: str) -> None:
    """Room-audio engage/disengage cues (the Pico is open-ear; the SDK
    exposes no controller-haptics API). Voice clips ("Teleop engaged" /
    "Teleop disengaged", gTTS — regen recipe in
    gen_pc3_audio_prompts.py) when present in gear_sonic/data/audio/;
    synthesized beeps otherwise (engage = two short, disengage = one
    long low)."""
    global _CUE_DIR
    import subprocess
    import tempfile
    import wave
    try:
        audio_dir = Path(__file__).resolve().parents[1] / "data" / "audio"
        # Named cues (gen_pico_audio_cues.py: Quest-stack voices + gTTS):
        # cue_mode_off / mode_locomotion / mode_whole_body / record_started /
        # record_stopped / replay_started / replay_done / estop_activating /
        # estop_damping ...; "engage"/"disengage" keep the original clips.
        voice = audio_dir / f"cue_{kind}.wav"
        if not voice.is_file() and kind in ("engage", "disengage", "release"):
            voice = audio_dir / f"teleop_{'engaged' if kind == 'engage' else 'disengaged'}.wav"
        if voice.is_file():
            subprocess.Popen(["aplay", "-q", str(voice)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        if _CUE_DIR is None:
            _CUE_DIR = tempfile.mkdtemp(prefix="pico_cues_")
            sr = 22050
            for name, segs in (
                    ("engage", [(880.0, 0.15), (0.0, 0.2), (880.0, 0.15)]),
                    ("disengage", [(440.0, 1.0)])):
                samples = []
                for freq, dur in segs:
                    t = np.arange(int(sr * dur)) / sr
                    x = (0.6 * np.sin(2 * np.pi * freq * t)
                         if freq else np.zeros(len(t)))
                    samples.append(x)
                x = np.concatenate(samples)
                x = (x * 32767).astype(np.int16)
                with wave.open(f"{_CUE_DIR}/{name}.wav", "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sr)
                    w.writeframes(x.tobytes())
        subprocess.Popen(
            ["aplay", "-q", f"{_CUE_DIR}/{kind}.wav"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # cues are best-effort; never break the control loop


def _heading(rot: Rot) -> float:
    v = rot.apply([1.0, 0.0, 0.0])
    return float(np.arctan2(v[1], v[0]))


def build_smpl_obs(joints, quats, base_quat_wxyz, wrist_il6,
                   yaw_align: Rot | None = None,
                   ori_mode: str = "full") -> np.ndarray:
    cur = Rot.from_quat([base_quat_wxyz[1], base_quat_wxyz[2],
                         base_quat_wxyz[3], base_quat_wxyz[0]])
    if ori_mode == "heading":
        # v1.1 cores: robot-YAW-only anchoring (see eval_x2_mujoco note)
        cur = Rot.from_euler("z", _heading(cur))
    frames = []
    for f in range(SMPL_WINDOW):
        oq = quats[f]
        o = Rot.from_quat([oq[1], oq[2], oq[3], oq[0]])
        if yaw_align is not None:
            o = yaw_align * o
        rel = cur.inv() * o
        frames.append(np.concatenate([
            joints[f].reshape(72),
            rel.as_matrix()[:, :2].reshape(6).astype(np.float32),
            wrist_il6.astype(np.float32),
        ]))
    return np.concatenate(frames).astype(np.float32)


class OnnxSmplActor:
    """onnxruntime drop-in for the torch smpl actor.

    Same call contract as ``UniversalTokenActor``: ``actor(prop, obs)`` ->
    action ``(B, 31)``.  The fused export's input is
    ``[tokenizer_obs(840) | proprioception(990)]`` in the exporter's order —
    ``build_smpl_obs`` already emits the per-frame-interleaved 840 layout
    (10 x [joints 72 | root6d 6 | wrists 6]), so assembly is a plain concat.
    Exported/validated by ``reexport_x2_g1_onnx.py --encoder-name smpl``
    against a ``dump_isaaclab_step0.py ++encoder_name=smpl`` dump.

    No ``ori_mode`` attribute on purpose: the plain 3-encoder exports (14k
    lineage) use full-orientation anchoring, which is what the caller's
    ``getattr(actor, "ori_mode", "full")`` defaults to. v1.1 heading cores
    go through the torch composite loaders, not this class.
    """

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        i_meta = self.sess.get_inputs()[0]
        o_meta = self.sess.get_outputs()[0]
        self.in_name, self.out_name = i_meta.name, o_meta.name
        self.in_dim = int(i_meta.shape[-1])

    def __call__(self, prop: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([obs, prop], dim=-1).cpu().numpy().astype(np.float32)
        if fused.shape[-1] != self.in_dim:
            raise ValueError(
                f"ONNX expects {self.in_dim}-D input, built {fused.shape[-1]}-D "
                f"(obs={obs.shape[-1]}, prop={prop.shape[-1]}) — encoder "
                "mismatch? This path needs the fused SMPL-encoder export.")
        out = self.sess.run([self.out_name], {self.in_name: fused})[0]
        return torch.from_numpy(out)


_TILT: dict = {}


def _tilt_probe_apply(args, model, data, t: float) -> None:
    """xfrc_applied: ramp/hold/release/rest torque on --tilt-body about its pitch (body Y) or roll
    (body X) axis, alternating sign; optional pelvis spring-damper hold at the pose seen on the
    first call (kp 1e4 / kd 1e3, ang 1e3 / 10 -- the X2 bridge ElasticBand constants)."""
    st = _TILT
    if not st:
        st["torso"] = model.body(args.tilt_body).id if args.tilt_torque > 0 else -1
        st["pelvis"] = model.body("pelvis").id if args.hold_pelvis else -1
        st["period"] = args.tilt_ramp + args.tilt_hold + args.tilt_release + args.tilt_rest
        st["last_idx"] = -1; st["mag"] = 0.0
        if st["pelvis"] >= 0:
            st["p0"] = np.array(data.xpos[st["pelvis"]]).copy(); st["q0"] = np.array(data.xquat[st["pelvis"]]).copy()
        print(f"[tilt] {args.tilt_torque:.0f} N*m {args.tilt_axis} on {args.tilt_body}, start {args.tilt_start}s ramp "
              f"{args.tilt_ramp}s hold {args.tilt_hold}s release {args.tilt_release}s rest {args.tilt_rest}s; hold_pelvis={args.hold_pelvis}", flush=True)
    mag = 0.0
    if st["torso"] >= 0:
        tt = t - args.tilt_start
        if tt >= 0:
            idx = int(tt // st["period"]); ph = tt - idx * st["period"]; a = args
            if ph < a.tilt_ramp: mag = a.tilt_torque * ph / max(a.tilt_ramp, 1e-6)
            elif ph < a.tilt_ramp + a.tilt_hold: mag = a.tilt_torque
            elif ph < a.tilt_ramp + a.tilt_hold + a.tilt_release:
                mag = a.tilt_torque * (1.0 - (ph - a.tilt_ramp - a.tilt_hold) / max(a.tilt_release, 1e-6))
            sign = -1.0 if idx % 2 else 1.0
            if idx != st["last_idx"]:
                st["last_idx"] = idx; print(f"[tilt] push #{idx} sign {sign:+.0f} at t={t:.1f}s", flush=True)
            mag *= sign
        R = np.asarray(data.xmat[st["torso"]]).reshape(3, 3)
        ax = np.array([0.0, 1.0, 0.0]) if args.tilt_axis == "pitch" else np.array([1.0, 0.0, 0.0])
        v = np.zeros(6); v[3:6] = R @ ax * mag; data.xfrc_applied[st["torso"]] = v
    st["mag"] = mag
    if st["pelvis"] >= 0:
        b = st["pelvis"]; p = data.xpos[b]; vlin = data.cvel[b][3:6]; wang = data.cvel[b][0:3]; qw = data.xquat[b]; q0 = st["q0"]
        r_err = (Rot.from_quat([q0[1], q0[2], q0[3], q0[0]]) * Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).inv()).as_rotvec()
        v = np.zeros(6); v[0:3] = 10000.0 * (st["p0"] - p) - 1000.0 * vlin; v[3:6] = 1000.0 * r_err - 10.0 * wang
        data.xfrc_applied[b] = v


def _tilt_probe_log(args, model, data, t: float, hold_target) -> None:
    st = _TILT
    if "csv" not in st:
        st["wp"] = int(model.joint("waist_pitch_joint").qposadr[0]); st["wr"] = int(model.joint("waist_roll_joint").qposadr[0])
        st["csv"] = open(args.waist_log, "w")
        st["csv"].write("t,q_waist_pitch_joint,target_waist_pitch_joint,q_waist_roll_joint,target_waist_roll_joint,tilt_nm\n")
    wp, wr = st["wp"], st["wr"]
    st["csv"].write(f"{t:.3f},{data.qpos[wp]:.5f},{hold_target[wp - 7]:.5f},{data.qpos[wr]:.5f},{hold_target[wr - 7]:.5f},{st.get('mag', 0.0):.2f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None,
                    help=".pt with a TRAINED smpl encoder (3-encoder layout), "
                         "or frozen-core-smpl:/smpl-g1: composite string")
    ap.add_argument("--onnx", default=None,
                    help="fused SMPL-encoder ONNX (reexport_x2_g1_onnx.py "
                         "--encoder-name smpl). Alone: ONNX drives the robot. "
                         "With --checkpoint: torch drives, ONNX shadows on "
                         "identical obs, per-step |action diff| -> "
                         "--compare-csv (in-loop parity check)")
    ap.add_argument("--compare-csv", default="/tmp/pico_onnx_compare.csv",
                    help="per-step action-diff CSV in shadow-compare mode")
    ap.add_argument("--robot", choices=("x2", "g1"), default="x2",
                    help="target embodiment; g1 = native G1 (no codec), "
                         "pair with a smpl-g1: checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tape-replay", default=None,
                    help="pico tape npz: drive the live path headset-free")
    ap.add_argument("--smpl-replay", default=None,
                    help="drive the smpl encoder DIRECTLY from an SMPL file "
                         "(smpl_filtered-format .pkl — source-corpus or pico "
                         "sidecar — or a pico_tape_to_smpl_obs .npz), "
                         "bypassing the Pico conversion. Dataset heading is "
                         "re-anchored at engage like a live operator's.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="headless run cap (0 = until Ctrl-C)")
    ap.add_argument("--qpos-dump", default=None)
    # HAND-TILT probe (2026-09-07), twin of the X2 bridge's --tilt-*: torque about the torso pitch/roll
    # axis (ramp/hold/release/rest, alternating), optional stiff pelvis hold (the operator's hand),
    # and a waist target-vs-measured CSV so the robot's manual-tilt table can be reproduced in sim
    # for ANY embodiment this harness runs (g1 v1.1 checkpoint, x2 checkpoints).
    ap.add_argument("--tilt-torque", type=float, default=0.0)
    ap.add_argument("--tilt-axis", choices=("pitch", "roll"), default="pitch")
    ap.add_argument("--tilt-body", default="torso_link")
    ap.add_argument("--tilt-start", type=float, default=10.0)
    ap.add_argument("--tilt-ramp", type=float, default=3.0)
    ap.add_argument("--tilt-hold", type=float, default=1.5)
    ap.add_argument("--tilt-release", type=float, default=0.1)
    ap.add_argument("--tilt-rest", type=float, default=4.0)
    ap.add_argument("--hold-pelvis", action="store_true")
    ap.add_argument("--waist-log", default=None, help="CSV: t, waist pitch/roll measured + target (rad), tilt N*m")
    ap.add_argument("--wrist-mode", choices=("measured", "zero"),
                    default="measured")
    ap.add_argument("--wrist-hold", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="hold wrist ACTIONS at neutral (default ON: the "
                         "smpl path carries no human wrist command, so "
                         "policy wrist output is prior noise — twisted "
                         "hands. --no-wrist-hold restores raw policy "
                         "wrists for testing)")
    ap.add_argument("--auto-engage", action="store_true",
                    help="skip the SPACE gate (implied by --headless)")
    ap.add_argument("--no-record", action="store_true",
                    help="live mode: do NOT save the session tape")
    ap.add_argument("--session-dir",
                    default=str(Path(os.environ.get("PICO_TAPES_DIR",
                                                    str(REPO_ROOT / "logs" / "pico_tapes"))) / "sessions"),
                    help="where live session tapes are saved "
                         "(default: $PICO_TAPES_DIR/sessions, PICO_TAPES_DIR=<repo>/logs/pico_tapes; "
                         "same root pico_intent_sender.py uses)")
    args = ap.parse_args()

    if not (args.checkpoint or args.onnx):
        ap.error("need --checkpoint and/or --onnx")

    actor_onnx = None
    if args.onnx:
        print(f"Loading fused smpl ONNX from {args.onnx} ...", flush=True)
        actor_onnx = OnnxSmplActor(args.onnx)
    cmp_log = None
    if args.checkpoint:
        print(f"Loading smpl-encoder actor from {args.checkpoint} ...",
              flush=True)
        actor = load_actor_from_checkpoint(args.checkpoint, args.device,
                                           encoder="smpl")
        if actor_onnx is not None:
            from onnx_policy_shim import CompareLogger
            cmp_log = CompareLogger(args.compare_csv, action_dim=31)
            print("shadow-compare: torch drives, ONNX shadows on identical "
                  f"obs; per-step diff -> {args.compare_csv}", flush=True)
    else:
        actor = actor_onnx
        actor_onnx = None   # single path; no shadow
    cmp_max = [0.0]

    # Per-robot plant BEFORE the stream thread: a plant/import failure must
    # abort instantly and cleanly, not after the operator connects.
    # All robot-dependent constants live in these locals so tick()/reset
    # close over the right embodiment.
    if args.robot == "g1":
        # v1.1-family checkpoints (smpl-g1:) trained on g1_model_12_dex:
        # per-joint action scale 0.25*effort/kp and the 12_dex actuator
        # plant. eval_g1_mujoco's hardcoded constants are the v1-release
        # plant (scale 1.0) and DO NOT pair with v1.1 — applying them
        # amplifies actions ~10-20x (instant yaw-over fall, 2026-08-21).
        import eval_g1_mujoco as _g1m
        from frozen_core_sonic_codec import harvest_g1_params
        plant = _g1m.g1_plant()          # model + name-mapped actuators
        model = plant["mj_model"]
        num_dofs, decimation = _g1m.NUM_DOFS, plant["decimation"]
        joint_to_act = plant["JOINT_TO_ACT"]
        gp = harvest_g1_params()
        slot = {n: s for s, n in enumerate(plant["mjcf_joints"])}
        il_to_mj = np.array([slot[n + "_joint"] for n in gp["il_names"]])
        kp = np.zeros(num_dofs); kd = np.zeros(num_dofs)
        default_dof = np.zeros(num_dofs); effort_lim = np.zeros(num_dofs)
        action_scale = np.zeros(num_dofs)   # per-joint, MJ slots
        for i in range(num_dofs):
            s = il_to_mj[i]
            kp[s], kd[s] = gp["kp_il"][i], gp["kd_il"][i]
            default_dof[s] = gp["default_il"][i]
            effort_lim[s] = gp["effort_il"][i]
            action_scale[s] = gp["scale_il"][i]
            model.dof_armature[6 + s] = gp["armature_il"][i]
        wrist_il = slice(23, 29)         # G1 IL wrist dofs
        init_z = 0.755
        prop_buf = _g1m.ProprioceptionBuffer()
    else:
        model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        model.opt.timestep = SIM_DT
        num_dofs, decimation, kp, kd = NUM_DOFS, DECIMATION, KP, KD
        default_dof, effort_lim = DEFAULT_DOF, None
        joint_to_act = JOINT_TO_ACTUATOR
        il_to_mj = IL_TO_MJ_DOF
        action_scale, wrist_il, init_z = ACTION_SCALE, X2_WRIST_IL, 0.68
        prop_buf = ProprioceptionBuffer()
    mj_to_il = np.argsort(il_to_mj)

    if args.smpl_replay and args.tape_replay:
        ap.error("--smpl-replay and --tape-replay are mutually exclusive")
    src = (SmplFileSource(args.smpl_replay) if args.smpl_replay
           else LiveSmplSource(args.tape_replay))
    print("Waiting for body stream ...", flush=True)
    t0 = time.monotonic()
    while src.age_s() > 0.5:
        if time.monotonic() - t0 > 120:
            print("no body stream after 120 s — is the app in the VR scene "
                  "with send-data ON and trackers calibrated?", file=sys.stderr)
            src.stop()
            return 1
        time.sleep(0.2)
    print("Body stream LIVE.", flush=True)

    # ---- startup stream validation (live mode only) --------------------
    # Frames flowing is NOT enough: with tracker calibration missing the
    # app streams a bit-identical seated DEFAULT RIG (post-mortem
    # 2026-08-22 — the robot dutifully sat down). Real tracking always
    # carries sensor noise, so bit-identical frames = invalid stream.
    if not (args.tape_replay or args.smpl_replay):
        while True:
            time.sleep(1.5)
            w = src.window(time.monotonic())
            if (w is not None
                    and float(np.abs(np.diff(w[0], axis=0)).max()) >= 1e-7):
                print("stream VALID: live skeleton detected.", flush=True)
                break
            print("stream INVALID: static default rig — tracker calibration "
                  "is missing. On the headset: tracker panel -> Calibrate "
                  "(all 3 trackers) -> re-enter VR scene. Re-checking ...",
                  flush=True)
            time.sleep(3.0)
            if time.monotonic() - t0 > 600:
                print("no valid skeleton after 10 min — giving up.",
                      file=sys.stderr)
                src.stop()
                return 1
    if not (args.headless or args.auto_engage):
        print("policy IDLE — the checkpoint does not drive the robot until "
              "you engage (squeeze both grips 0.5 s, or SPACE in the viewer).",
              flush=True)

    data = mujoco.MjData(model)
    data.qpos[2] = init_z
    data.qpos[3] = 1.0
    data.qpos[7:7 + num_dofs] = default_dof
    mujoco.mj_forward(model, data)
    # One-time settle so the feet plant on the ground plane, then freeze
    # (the pre-engage state is kinematic — no physics until first engage).
    for _ in range(int(0.2 / model.opt.timestep)):
        tq = kp * (default_dof - data.qpos[7:7 + num_dofs]) \
            - kd * data.qvel[6:6 + num_dofs]
        if effort_lim is not None:
            tq = np.clip(tq, -effort_lim, effort_lim)
        for j in range(num_dofs):
            data.ctrl[joint_to_act[j]] = tq[j]
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    last_action_mj = np.zeros(num_dofs, np.float32)
    engaged = args.headless or args.auto_engage
    hold_target = default_dof.copy()
    qpos_rows, qpos_t = [], []
    step = 0
    yaw_align = None      # recomputed at each engage: human heading -> robot heading
    grip_t0 = None
    last_toggle = -10.0
    last_win = None       # newest fresh window (policy input freeze source)
    frozen_win = None     # set on disengage: static reference to balance at
    policy_started = [False]  # False = pre-first-engage kinematic hold
    seg = {"n": 0, "t0": time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())}
    if engaged and src.record:   # --auto-engage live session records from t=0
        src.rec_gate = True
    grip_armed = [True]       # gesture re-arms only after grip release
    static_warned = [False]

    def _standing_check(j) -> list:
        """Engage-gate: the operator must be in a stable stand. Returns a
        list of violations (empty = OK). Catches bad/drifted calibration
        that the static-rig check cannot (skeleton moves but is
        geometrically wrong), and standardizes the teleop start pose."""
        reasons = []
        for side, (h, k, a) in (("L", (1, 4, 7)), ("R", (2, 5, 8))):
            u, v = j[h] - j[k], j[a] - j[k]
            c = u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
            flex = np.degrees(np.pi - np.arccos(np.clip(c, -1, 1)))
            if flex > 30:
                reasons.append(f"{side} knee bent {flex:.0f} deg (max 30)")
        ank = -min(j[7][2], j[8][2])          # pelvis height above ankles
        if not (0.55 < ank < 1.25):
            reasons.append(f"pelvis {ank:.2f}m above ankles "
                           "(want 0.55-1.25 — recalibrate if you ARE standing)")
        up = j[12] - j[0]                      # pelvis -> neck
        tilt = np.degrees(np.arccos(np.clip(
            up[2] / max(np.linalg.norm(up), 1e-9), -1, 1)))
        if tilt > 25:
            reasons.append(f"torso tilt {tilt:.0f} deg (max 25)")
        return reasons

    def _stream_is_static() -> bool:
        # The app serves a bit-identical default rig when tracker
        # calibration is missing (session post-mortem 2026-08-22: robot
        # "sat down and ignored the operator" = perfect imitation of the
        # seated mannequin). Real tracking always carries sensor noise.
        if last_win is None:
            return True
        return float(np.abs(np.diff(last_win[0], axis=0)).max()) < 1e-7

    def toggle_engage() -> None:
        nonlocal engaged, yaw_align, frozen_win
        if not engaged and _stream_is_static():
            print("REFUSING to engage: body stream is a STATIC default rig "
                  "— tracker calibration is missing. On the headset: "
                  "tracker panel -> Calibrate (all 3 trackers), re-enter "
                  "the VR scene, then try again.", flush=True)
            return
        if not engaged and last_win is not None:
            bad = _standing_check(last_win[0][-1])
            if bad:
                print("REFUSING to engage — start pose must be a stable "
                      "stand: " + "; ".join(bad), flush=True)
                return
        engaged = not engaged
        yaw_align = None
        # DISENGAGED != PD-freeze: SONIC robots cannot open-loop hold (the
        # policy must keep balancing — same law as deploy's idle stream).
        # Freeze the INPUT instead: keep running the actor on the last
        # window, so the robot balances in place and ignores the operator.
        frozen_win = None if engaged else last_win
        # Corpus capture: one ENGAGED stretch = one clip file, flushed to
        # disk at each disengage (crash-safe; repeated engage/disengage is
        # the intended way to record many small clips in one session).
        if src.record:
            if engaged:
                seg["t0"] = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
                src.rec_gate = True
            else:
                src.rec_gate = False
                seg["n"] += 1
                sdir = Path(args.session_dir)
                sdir.mkdir(parents=True, exist_ok=True)
                p = sdir / (f"session_{seg['t0']}_{args.robot}"
                            f"_seg{seg['n']:02d}.npz")
                cnt = src.save_session(p, clear=True)
                if cnt:
                    print(f"[clip] saved {cnt} frames -> {p}", flush=True)
        _audio_cue("engage" if engaged else "disengage")
        print("ENGAGED (heading re-anchored)" if engaged
              else "DISENGAGED (balancing in place — walk freely)", flush=True)

    def reset_robot() -> None:
        nonlocal yaw_align, hold_target, last_action_mj
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qpos[2] = init_z
        data.qpos[3] = 1.0
        data.qpos[7:7 + num_dofs] = default_dof
        mujoco.mj_forward(model, data)
        hold_target = default_dof.copy()
        last_action_mj = np.zeros(num_dofs, np.float32)
        yaw_align = None  # robot yaw changed; re-anchor on next window

    def tick() -> None:
        nonlocal last_action_mj, hold_target, step, yaw_align, \
            grip_t0, last_toggle, last_win, frozen_win
        base_quat = data.qpos[3:7].copy()
        qpos_j = data.qpos[7:7 + num_dofs].copy()
        qvel_j = data.qvel[6:6 + num_dofs].copy()
        gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
        dof_pos_il = qpos_j[il_to_mj]
        dof_vel_il = qvel_j[il_to_mj]
        prop_buf.append(gravity, data.qvel[3:6].copy(),
                        dof_pos_il - default_dof[il_to_mj], dof_vel_il,
                        last_action_mj[il_to_mj])

        # dual-grip gesture: both grips >0.8 held 0.5 s toggles engage.
        # ONE toggle per squeeze: after firing, the grips must be RELEASED
        # before the gesture re-arms (a long squeeze must not oscillate).
        # TAPE REPLAY: recorded grips are historical DATA, not operator
        # commands — replaying them re-fires engagement toggles phase-
        # inverted vs the live session (found 2026-08-22: replay froze at
        # the operator's recorded engage squeeze).
        lg, rg = (0.0, 0.0) if args.tape_replay else src.grips()
        now_m = time.monotonic()
        if lg > 0.8 and rg > 0.8:
            if grip_armed[0]:
                if grip_t0 is None:
                    grip_t0 = now_m
                elif now_m - grip_t0 > 0.5:
                    toggle_engage()
                    last_toggle = now_m
                    grip_armed[0] = False
                    grip_t0 = None
        else:
            grip_t0 = None
            grip_armed[0] = True

        age = src.age_s()
        stale = age > 0.5
        # Long dropout while engaged: the operator may have moved/turned in
        # the gap, so the heading anchor is stale — drop to DISENGAGED and
        # require a fresh grip gesture (which re-anchors) rather than
        # snapping to the old anchor when frames resume.
        if engaged and age > 3.0:
            print(f"stream lost {age:.1f}s — auto-DISENGAGE "
                  "(re-squeeze both grips when ready)", flush=True)
            toggle_engage()
        win = None if stale else src.window(time.monotonic())
        if win is not None:
            last_win = win
        # Policy input selection: live window when engaged; the frozen
        # capture when disengaged; last fresh window during a short stall.
        # The actor RUNS in all three. Before the FIRST engage the robot is
        # held KINEMATICALLY (physics paused) — humanoids cannot open-loop
        # PD-stand (the G1's crouch default falls in ~2 s), and the policy
        # must not run pre-engage; this is the sim stand-in for deploy's
        # stable-stand profile.
        if engaged:
            use_win = win if win is not None else last_win
        else:
            use_win = frozen_win
        if use_win is not None:
            policy_started[0] = True
            if yaw_align is None:
                oqn = use_win[1][-1]
                human = Rot.from_quat([oqn[1], oqn[2], oqn[3], oqn[0]])
                robot = Rot.from_quat([base_quat[1], base_quat[2],
                                       base_quat[3], base_quat[0]])
                yaw_align = Rot.from_euler(
                    "z", _heading(robot) - _heading(human))
                print(f"heading anchor: human {np.degrees(_heading(human)):+.0f} deg"
                      f" -> robot {np.degrees(_heading(robot)):+.0f} deg", flush=True)
            wrist6 = (dof_pos_il[wrist_il]
                      if args.wrist_mode == "measured" else np.zeros(6))
            obs = build_smpl_obs(use_win[0], use_win[1], base_quat, wrist6,
                                 yaw_align,
                                 ori_mode=getattr(actor, "ori_mode", "full"))
            prop_t = torch.from_numpy(prop_buf.get_flat()).unsqueeze(0)
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                a_t = actor(prop_t, obs_t)
                if cmp_log is not None:
                    a_onnx = actor_onnx(prop_t, obs_t)
                    cmp_log.log(a_t, a_onnx)
                    d = float((a_t - a_onnx).abs().max())
                    if d > cmp_max[0]:
                        cmp_max[0] = d
            a = a_t.squeeze(0).numpy()
            a = np.clip(a, -ACTION_CLIP, ACTION_CLIP)
            if args.wrist_hold or os.environ.get("DEBUG_FREEZE_WRIST_ACT"):
                a[wrist_il] = 0.0   # hands stay at natural neutral pose
            action_mj = a[mj_to_il]
            last_action_mj = action_mj.astype(np.float32)
            hold_target = default_dof + action_mj * action_scale
        if not policy_started[0]:
            # pre-first-engage: freeze in place, no physics stepping
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        else:
            # engaged: policy target; disengaged-after-engage: policy on
            # frozen input (balance in place)
            for _ in range(decimation):
                torque = (kp * (hold_target - data.qpos[7:7 + num_dofs])
                          - kd * data.qvel[6:6 + num_dofs])
                if effort_lim is not None:
                    torque = np.clip(torque, -effort_lim, effort_lim)
                for j in range(num_dofs):
                    data.ctrl[joint_to_act[j]] = torque[j]
                if args.tilt_torque > 0 or args.hold_pelvis:
                    _tilt_probe_apply(args, model, data, step * CONTROL_DT)
                mujoco.mj_step(model, data)
            if args.waist_log:
                _tilt_probe_log(args, model, data, step * CONTROL_DT, hold_target)
        step += 1
        if args.qpos_dump:
            qpos_t.append(step * CONTROL_DT)
            qpos_rows.append(data.qpos[:7 + num_dofs].copy())
        if step % 250 == 0:
            static = not engaged and last_win is not None and _stream_is_static()
            if static and not static_warned[0]:
                print("[warn] body stream is STATIC (default rig) — "
                      "recalibrate trackers on the headset before engaging.",
                      flush=True)
                static_warned[0] = True
            elif not static:
                static_warned[0] = False
            print(f"t={step * CONTROL_DT:6.1f}s pelvis_z={data.qpos[2]:.3f} "
                  f"stream_age={src.age_s()*1000:4.0f}ms "
                  f"grips={lg:.2f}/{rg:.2f} "
                  f"{'ENGAGED' if engaged else 'holding'}"
                  f"{' STATIC-RIG' if static else ''}"
                  f"{' STALE-FREEZE' if stale else ''}", flush=True)
        if lg > 0.8 and rg > 0.8 and grip_t0 is not None and step % 10 == 0:
            print(f"  grip gesture: {time.monotonic() - grip_t0:.1f}s held",
                  flush=True)

    try:
        if args.headless:
            # The source (headset or tape replay) is wall-clock; the sim
            # MUST be paced to it or window() serves stale ring frames.
            next_t = time.monotonic()
            while args.seconds <= 0 or step * CONTROL_DT < args.seconds:
                tick()
                next_t += CONTROL_DT
                lag = next_t - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
                elif lag < -1.0:
                    next_t = time.monotonic()  # fell behind; don't spiral
                if data.qpos[2] < 0.35:
                    print(f"FALL at t={step * CONTROL_DT:.1f}s "
                          f"(pelvis_z={data.qpos[2]:.3f})", flush=True)
                    break
        else:
            import mujoco.viewer as _mj_viewer

            def key_cb(keycode):
                if keycode == ord(' '):
                    toggle_engage()
                elif keycode == ord('R'):
                    reset_robot()

            print("Viewer: SPACE = engage/disengage whole-body follow, "
                  "R = reset robot.", flush=True)
            with _mj_viewer.launch_passive(
                    model, data, key_callback=key_cb) as viewer:
                next_t = time.monotonic()
                while viewer.is_running():
                    tick()
                    if data.qpos[2] < 0.35:
                        print(f"FALL at t={step * CONTROL_DT:.1f}s — "
                              "auto-reset (viewer mode)", flush=True)
                        reset_robot()
                    viewer.sync()
                    next_t += CONTROL_DT
                    lag = next_t - time.monotonic()
                    if lag > 0:
                        time.sleep(lag)
                    else:
                        next_t = time.monotonic()
    finally:
        src.stop()
        if cmp_log is not None:
            cmp_log.close()
            print(f"[compare] max|torch-onnx| over run = {cmp_max[0]:.2e} rad "
                  f"(pass bar 1e-3) -> {args.compare_csv}", flush=True)
        if src.record and not args.no_record and src.rec_gate:
            # session ended while still engaged: flush the final clip
            seg["n"] += 1
            sdir = Path(args.session_dir)
            sdir.mkdir(parents=True, exist_ok=True)
            p = sdir / f"session_{seg['t0']}_{args.robot}_seg{seg['n']:02d}.npz"
            n = src.save_session(p, clear=True)
            if n:
                print(f"[clip] saved {n} frames -> {p}", flush=True)
        if args.qpos_dump and qpos_rows:
            np.savez_compressed(args.qpos_dump, t=np.asarray(qpos_t),
                                qpos=np.stack(qpos_rows), fps=1.0 / CONTROL_DT)
            print(f"[qpos-dump] {len(qpos_rows)} steps -> {args.qpos_dump}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
