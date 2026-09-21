#!/usr/bin/env python3
"""LAPTOP half of robot whole-body teleop: Pico -> tokens -> PC2 pose topic.

The "second command", run on the laptop while the robot ritual runs
`deploy_x2.sh ... --whole-body-teleop --input-type=zmq` with the token
graph active. Workflow mirrors the Quest-VR overlay: launch alongside the
session, squeeze both grips 0.5 s to ENGAGE (whole-body follow), squeeze
again to DISENGAGE. v1 disengage = the token stream FREEZES at the last
value, so the robot balances in place (the sim path's proven semantic);
pad fallback on disengage is Phase 2 (see pico_wbc_robot_spec.md).

Per 50 Hz tick while engaged:
  Pico body stream -> smpl obs (840, wrist slots ZEROED — the laptop has
  no robot proprio; WRIST_OBS_ZERO-equivalent, the composite's documented
  diagnostic mode) -> release smpl encoder + FSQ (via FrozenCoreSmplActor's
  own modules == the parity reference `laptop_token`) -> motion_token[64]
  -> pack_pose_message -> PUB tcp://*:5556 topic "pose".

The same message carries joint_pos_mj = default stand pose with WRIST
OVERRIDES from the Pico controller orientations (pitch/roll deltas from
the engage-time baseline, clamped to X2 wrist ranges) — consumed by the
deploy's existing --wrist-bypass=ik, so "follow my wrist" rides the
Quest path's proven mechanism. An `engaged` field is included for the
Phase-2 hybrid; the v1 deploy ignores it.

Safety inherited robot-side: pose-ref starvation watchdog (stream stop
-> SAFE_IDLE + chord resume), soft-start ramp, dev clamps, tilt-trip.

    .venv/bin/python gear_sonic/scripts/pico_token_sender.py \
        --checkpoint ${CKPT_ROOT}/<run>/<merged>.pt      # see MODELS.md
    # that's the whole robot command: binds *:5556, the deploy connects here
    # headset-free rehearsal:
    ... --tape-replay logs/pico_tapes/sessions/<tape>.npz --auto-engage
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[1]))

from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from frozen_core_sonic_codec import FrozenCoreSmplActor, NUM_X2  # noqa: E402
from live_pico_smpl_teleop import (  # noqa: E402
    LiveSmplSource, build_smpl_obs, _audio_cue, _heading, SMPL_WINDOW)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402
import zmq  # noqa: E402

import torch  # noqa: E402

CONTROL_DT = 0.02          # 50 Hz, the deploy's expectation
TOKEN_DIM = 64

# X2 MJ-order wrist DOF indices (matches wrist_bypass.hpp's table) and the
# joint ranges used to clamp controller-derived targets.
WRIST_MJ = {"left_pitch": 20, "left_roll": 21, "right_pitch": 27, "right_roll": 28}
WRIST_RANGE = {"pitch": (-0.56, 0.56), "roll": (-1.57, 0.72)}   # rad, X2 URDF


def _quat_to_pitch_roll(q_xyzw: np.ndarray) -> tuple[float, float]:
    e = Rot.from_quat(q_xyzw).as_euler("ZYX")   # yaw, pitch, roll
    return float(e[1]), float(e[2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True,
                    help="merged armA ckpt (encoder/FSQ source, must be the "
                         "SAME merge the robot's token ONNX was exported from)")
    ap.add_argument("--pc2-host", default=None,
                    help="OPTIONAL alternate topology: connect-PUB to "
                         "tcp://HOST:PORT instead of binding. The normal "
                         "robot session needs nothing — the deploy connects "
                         "TO this laptop (ritual X2_WB_LAPTOP_HOST).")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--tape-replay", default=None)
    ap.add_argument("--auto-engage", action="store_true")
    ap.add_argument("--no-wrist-follow", action="store_true",
                    help="stream neutral wrists instead of controller-derived")
    ap.add_argument("--debug-host", default="127.0.0.1",
                    help="deploy x2_debug telemetry host (sim: this machine; "
                         "robot session: PC2, i.e. $PC2_HOST)")
    ap.add_argument("--debug-port", type=int, default=5659,
                    help="deploy x2_debug port (sim stack 5659; robot 5557)")
    ap.add_argument("--rate-report-s", type=float, default=10.0)
    args = ap.parse_args()


    print(f"loading composite (encoder/FSQ/codec) from {args.checkpoint} ...",
          flush=True)
    actor = FrozenCoreSmplActor(merged_ckpt=str(Path(args.checkpoint).expanduser()),
                             device="cpu")
    d_x2 = actor.codec.d_x2.astype(np.float32)          # IL-order defaults
    # IL->MJ map for the reference pose the deploy consumes (MJ order).
    from eval_x2_mujoco import IL_TO_MJ_DOF  # noqa: E402
    from live_pico_smpl_teleop import X2_WRIST_IL  # noqa: E402
    from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
        unpack_message)
    default_mj = np.zeros(NUM_X2, np.float32)
    default_mj[IL_TO_MJ_DOF] = d_x2

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    if args.pc2_host:
        sock.connect(f"tcp://{args.pc2_host}:{args.port}")
        print(f"PUB connect tcp://{args.pc2_host}:{args.port} topic 'pose'")
    else:
        sock.bind(f"tcp://*:{args.port}")
        print(f"PUB bind tcp://*:{args.port} topic 'pose' — the deploy "
              f"(or sim rehearsal) connects here")

    # SAFE_IDLE resume chord (Quest-parity): A+B held >= 1 s publishes on
    # the pose_resume topic (PUB bind :5566, deploy connects). Without this,
    # a mid-session stream trip is unrecoverable from the pico side.
    resume_sock = ctx.socket(zmq.PUB)
    try:
        resume_sock.bind("tcp://*:5566")
        print("resume chord ready: hold A+B >= 1 s to exit SAFE_IDLE "
              "(PUB *:5566 'pose_resume')")
    except zmq.ZMQError as e:
        print(f"WARNING: resume PUB bind failed ({e}) — another publisher "
              "on :5566? chord disabled here.")
        resume_sock = None

    # --- robot-state feedback (x2_debug echo) ----------------------------
    # The SMPL obs needs the robot's LIVE base orientation and measured
    # wrists (the in-process path reads them from MuJoCo directly; the
    # identity-quat shortcut fell the sim robot 2 s after engage —
    # rehearsal 2026-08-29). The deploy already publishes both on its
    # x2_debug telemetry; this thread mirrors them at stream rate.
    state = {"quat_wxyz": np.array([1.0, 0, 0, 0], np.float32),
             "body_q_mj": None, "t": 0.0}
    def _debug_rx():
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://{args.debug_host}:{args.debug_port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "x2_debug")
        sub.setsockopt(zmq.RCVTIMEO, 500)
        while True:
            try:
                raw = sub.recv()
            except zmq.Again:
                continue
            try:
                msg = unpack_message(raw, expected_topic="x2_debug")
                f = msg.fields
                q = np.asarray(f.get("base_quat"), np.float32).ravel()
                if q.shape[0] == 4:
                    state["quat_wxyz"] = q
                bq = f.get("body_q")
                if bq is not None:
                    state["body_q_mj"] = np.asarray(bq, np.float32).ravel()
                state["t"] = time.monotonic()
            except Exception as e:
                if time.monotonic() - state.get("t_decode_warn", 0) > 5.0:
                    print(f"[sender] x2_debug decode error: {e!r}", flush=True)
                    state["t_decode_warn"] = time.monotonic()
                continue
    import threading
    threading.Thread(target=_debug_rx, daemon=True).start()
    print(f"x2_debug feedback: tcp://{args.debug_host}:{args.debug_port}",
          flush=True)
    mj_to_il = np.argsort(IL_TO_MJ_DOF)

    src = LiveSmplSource(args.tape_replay)
    print("Waiting for body stream ...", flush=True)
    while src.window(time.monotonic()) is None:
        time.sleep(0.2)
    print("stream live.", flush=True)

    engaged = bool(args.auto_engage)
    grip_t0 = None
    grip_armed = True
    yaw_align = None
    frozen = None            # (token, wrists_mj) held while disengaged
    wrist_base = None        # controller pitch/roll at engage
    frame_idx = 0
    n_sent, t_rep = 0, time.monotonic()
    chord_t0, chord_last = None, 0.0
    t_fbwarn = 0.0
    id_quat = np.array([0.0, 0.0, 0.0, 1.0], np.float32)   # xyzw

    def controller_pr():
        l = np.asarray(src.xrt.get_left_controller_pose(), np.float32)
        r = np.asarray(src.xrt.get_right_controller_pose(), np.float32)
        return (_quat_to_pitch_roll(l[3:7]), _quat_to_pitch_roll(r[3:7]))

    print("DISENGAGED — squeeze both grips 0.5 s to engage whole-body "
          "(same toggle as sim)." if not engaged else "ENGAGED (auto).",
          flush=True)
    next_t = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(next_t - now, 0.005))
                continue
            next_t += CONTROL_DT

            # --- grip toggle (mirror of live_pico_smpl_teleop) -------------
            lg, rg = (0.0, 0.0) if args.tape_replay else src.grips()
            if lg > 0.8 and rg > 0.8:
                if grip_armed:
                    if grip_t0 is None:
                        grip_t0 = now
                    elif now - grip_t0 > 0.5:
                        engaged = not engaged
                        grip_armed = False
                        grip_t0 = None
                        yaw_align = None
                        wrist_base = None
                        frozen = None if engaged else frozen
                        _audio_cue("engage" if engaged else "disengage")
                        print("ENGAGED" if engaged else
                              "DISENGAGED (token frozen — robot balances; "
                              "pad fallback is Phase 2)", flush=True)
            else:
                grip_t0 = None
                grip_armed = True

            # --- SAFE_IDLE resume chord: A+B held >= 1 s ------------------
            if resume_sock is not None and not args.tape_replay:
                try:
                    ab = bool(src.xrt.get_A_button()) and bool(src.xrt.get_B_button())
                except Exception:
                    ab = False
                if ab:
                    if chord_t0 is None:
                        chord_t0 = now
                    elif now - chord_t0 >= 1.0 and now - chord_last >= 0.5:
                        resume_sock.send_multipart([b"pose_resume", b"1"])
                        chord_last = now
                        print("[resume] chord published (A+B held)", flush=True)
                else:
                    chord_t0 = None

            win = src.window(now)
            if win is None:
                continue   # stream gap: send nothing; robot watchdog governs

            joints, quats = win
            # SAFETY GATE: engaged tokens are only valid with LIVE robot
            # state (the identity-quat shortcut felled the sim robot twice).
            # Stale feedback -> the message's engaged flag drops, the deploy
            # falls back to the pad; tokens resume when feedback returns.
            fb_age = now - state["t"]
            fb_live = fb_age < 0.5 and state["t"] > 0.0
            if engaged and not fb_live and now - t_fbwarn > 5.0:
                print(f"[sender] waiting for robot-state feedback "
                      f"(x2_debug) — engaged withheld (age {fb_age:.1f}s)",
                      flush=True)
                t_fbwarn = now
            if (engaged and fb_live) or frozen is None:
                bq_wxyz = (state["quat_wxyz"] if fb_live
                           else np.array([1.0, 0, 0, 0], np.float32))
                if yaw_align is None and fb_live:
                    # engage anchor: human heading -> ROBOT heading (live
                    # path semantics, via the x2_debug echo)
                    robot = Rot.from_quat([bq_wxyz[1], bq_wxyz[2],
                                           bq_wxyz[3], bq_wxyz[0]])
                    human = Rot.from_quat(quats[-1])
                    yaw_align = Rot.from_euler(
                        "z", _heading(robot) - _heading(human))
                    print(f"heading anchor: human -> robot "
                          f"(fb_age {fb_age:.2f}s)", flush=True)
                if state["body_q_mj"] is not None and fb_live:
                    wrist6 = state["body_q_mj"][:NUM_X2][mj_to_il][X2_WRIST_IL]
                else:
                    wrist6 = np.zeros(6, np.float32)
                obs = build_smpl_obs(joints, quats, bq_wxyz, wrist6,
                                     yaw_align=yaw_align)
                with torch.no_grad():
                    tok = actor.fsq(actor.smpl_encoder(
                        torch.from_numpy(obs)[None])).numpy().ravel()

                wr = default_mj.copy()
                if not args.no_wrist_follow and not args.tape_replay:
                    (lp, lr), (rp, rr) = controller_pr()
                    if wrist_base is None:
                        wrist_base = (lp, lr, rp, rr)
                    dlp, dlr = lp - wrist_base[0], lr - wrist_base[1]
                    drp, drr = rp - wrist_base[2], rr - wrist_base[3]
                    wr[WRIST_MJ["left_pitch"]] = np.clip(
                        default_mj[WRIST_MJ["left_pitch"]] + dlp, *WRIST_RANGE["pitch"])
                    wr[WRIST_MJ["left_roll"]] = np.clip(
                        default_mj[WRIST_MJ["left_roll"]] + dlr, *WRIST_RANGE["roll"])
                    wr[WRIST_MJ["right_pitch"]] = np.clip(
                        default_mj[WRIST_MJ["right_pitch"]] + drp, *WRIST_RANGE["pitch"])
                    wr[WRIST_MJ["right_roll"]] = np.clip(
                        default_mj[WRIST_MJ["right_roll"]] - drr, *WRIST_RANGE["roll"])
                frozen = (tok.astype(np.float32), wr)

            tok, wr = frozen
            msg = pack_pose_message({
                "joint_pos_mj": wr,
                "root_quat_xyzw": id_quat,
                "motion_token": tok,
                "engaged": np.array(
                    [1.0 if (engaged and fb_live) else 0.0], np.float32),
                "frame_index": np.array([frame_idx], np.int64),
            }, topic="pose", version=4)
            sock.send(msg)
            frame_idx += 1
            n_sent += 1
            if now - t_rep >= args.rate_report_s:
                print(f"[sender] {n_sent / (now - t_rep):.1f} Hz  "
                      f"{'ENGAGED' if engaged else 'disengaged'}  "
                      f"frames={frame_idx}", flush=True)
                n_sent, t_rep = 0, now
    except KeyboardInterrupt:
        print("\nsender stopped.", flush=True)
    finally:
        src.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
