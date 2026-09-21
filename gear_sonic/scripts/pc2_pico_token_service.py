#!/usr/bin/env python3
"""PC2-side SMPL token service — the robot-local half of whole-body teleop.

ARCHITECTURE (operator pivot, 2026-08-29 — restores the Quest split after
the laptop-side-encoding rehearsal fell twice on cross-network state):
the LAPTOP streams only state-independent human INTENT (per-frame SMPL
joints, engage flag, controller wrist targets); THIS service, running next
to the deploy like pc2_kplanner_onnx.py does, assembles the 840-dim SMPL
observation using LOCAL robot state (the deploy's x2_debug echo on
localhost), runs the exported smpl-tokenizer ONNX (onnxruntime, CPU-cheap
MLP), and publishes tokens + engaged + wrist reference into the deploy's
token source on localhost:5574. Wifi loss stops the intent stream ->
tokens stop -> the hybrid deploy falls back to the pad, by construction.

Wire (laptop -> here, topic ``pico_intent``, this side SUB-binds :5573):
    smpl_joints    f32[72]   root-relative joints, newest frame
    human_quat     f32[4]    global orient (xyzw)
    engaged        f32[1]
    wrist_pr       f32[4]    [l_pitch l_roll r_pitch r_roll] targets (rad)
    frame_index    i64[1]

    python3 gear_sonic/scripts/pc2_pico_token_service.py \
        --tokenizer policies/x2_smpl_tokenizer_v11release.onnx \
        --debug-port 5557          # deploy's x2_debug (sim stack: 5659)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from os import environ as _environ
# TOKEN_SVC_OPERATOR_ROOT_LEVEL: '' (off, default) | 'zero' (keep heading, zero pitch/roll of the human root) | 'clamp:<deg>' | 'offset:<deg>' (subtract a fixed pitch)
_ROOT_LEVEL = _environ.get("TOKEN_SVC_OPERATOR_ROOT_LEVEL", "").strip()
# TOKEN_SVC_OPERATOR_LEAN_CAP (deg, 0 = off) / TOKEN_SVC_OPERATOR_REACH_CAP (m, 0 = off): hard-limit the OPERATOR's
# forward torso lean (neck vs pelvis, rotated back about the pelvis) and forward hand reach (wrist ahead of the shoulder,
# arm chain compressed along x) in the SMPL joints before tokenization. 2026-09-15: the waist snap-back is a hip
# impulse triggered by the waist being pushed onto its forward stop by the arms-out load; keeping the reference's
# lean/reach below what the 24 N*m waist can hold removes the trigger, at the cost of deep bends and long reaches.
_LEAN_CAP = float(_environ.get("TOKEN_SVC_OPERATOR_LEAN_CAP", "0") or 0.0)
_REACH_CAP = float(_environ.get("TOKEN_SVC_OPERATOR_REACH_CAP", "0") or 0.0)
_CAP_STATS = {"frames": 0, "lean": 0, "reach": 0}


def _cap_operator_pose(joints):
    """joints (W,24,3) root-relative SMPL joints, heading-normalized: x forward, y left, z up. Returns a capped copy."""
    J = np.array(joints, np.float32, copy=True)
    UPPER = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    for f in range(J.shape[0]):
        _CAP_STATS["frames"] += 1
        if _LEAN_CAP > 0.0:
            v = J[f, 12] - J[f, 0]; lean = float(np.degrees(np.arctan2(v[0], max(v[2], 1e-6))))
            if lean > _LEAN_CAP:
                _CAP_STATS["lean"] += 1
                rel = J[f, UPPER] - J[f, 0]
                best = None
                for sgn in (1.0, -1.0):
                    r = Rot.from_euler("y", sgn * np.radians(lean - _LEAN_CAP)).apply(rel)
                    vv = r[3]; l2 = float(np.degrees(np.arctan2(vv[0], max(vv[2], 1e-6))))   # index 3 of UPPER = neck (12)
                    if best is None or abs(l2 - _LEAN_CAP) < best[0]: best = (abs(l2 - _LEAN_CAP), r)
                J[f, UPPER] = J[f, 0] + best[1]
        if _REACH_CAP > 0.0:
            for sh, chain in ((16, (18, 20, 22)), (17, (19, 21, 23))):
                reach = float(J[f, chain[1], 0] - J[f, sh, 0])
                if reach > _REACH_CAP:
                    _CAP_STATS["reach"] += 1; k = _REACH_CAP / reach
                    for j in chain: J[f, j, 0] = J[f, sh, 0] + (J[f, j, 0] - J[f, sh, 0]) * k
    return J


REPO = Path(__file__).resolve().parents[2]
# PC2 install prefix (robot-side); intent-span dumps land under <prefix>/log.
_PC2_PREFIX = os.environ.get("PC2_PREFIX", os.path.expanduser("~/gear-sonic"))
sys.path.insert(0, str(REPO))

import zmq  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from gear_sonic.utils.teleop.smpl_obs import (  # noqa: E402
    DELAY_S, SMPL_DT, SMPL_WINDOW, _heading, build_smpl_obs)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message  # noqa: E402

CONTROL_DT = 0.02
NUM_X2 = 31
# MJ-order wrist rows (wrist_bypass.hpp table) for the reference overrides.
WRIST_MJ = {"lp": 20, "lr": 21, "rp": 27, "rr": 28}
WRIST_RANGE = {"p": (-0.56, 0.56), "r": (-1.57, 0.72)}
# IL wrist slice + IL->MJ map are only needed for measured-wrist obs; the
# deploy's debug body_q is MJ-order.
X2_WRIST_IL = slice(25, 31)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", required=True,
                    help="x2_smpl_tokenizer_*.onnx (smpl_obs 840 -> token 64)")
    ap.add_argument("--intent-port", type=int, default=5573)
    ap.add_argument("--token-port", type=int, default=5574)
    ap.add_argument("--debug-host", default="127.0.0.1")
    ap.add_argument("--debug-port", type=int, default=5557,
                    help="deploy x2_debug (robot ritual 5557; sim stack 5659)")
    ap.add_argument("--planner-cmd-host", default="127.0.0.1")
    ap.add_argument("--planner-cmd-port", type=int, default=0,
                    help="kplanner planner_cmd SUB port (robot 5563; sim "
                         "stack 5663). When >0 this service PUBs an "
                         "intent 'wb_state' {engaged 0/1} on every engage "
                         "transition + a 0.5 s keepalive while engaged, so "
                         "the kplanner can shadow/hold/return instead of "
                         "handing the deploy a stand-up. 0 = off.")
    ap.add_argument("--il-to-mj", default=str(REPO / "gear_sonic_deploy/configs/x2_il_to_mj.json"),
                    help="optional IL->MJ dof map json; identity-fallback if absent")
    ap.add_argument("--operator-root-level", default=None,
                    help="level the OPERATOR ROOT (SMPL pelvis of the Pico body stream, not the robot base) before tokenization: 'off' | 'zero' (keep heading, "
                         "zero pitch/roll) | 'clamp:<deg>' | 'offset:<deg>' (subtract a fixed forward "
                         "pitch). Default: $TOKEN_SVC_OPERATOR_ROOT_LEVEL or off. The Pico root carries a steady "
                         "forward pitch that the SMPL path otherwise commands as a lean (2026-09-15).")
    ap.add_argument("--operator-lean-cap", type=float, default=None,
                    help="hard-limit the operator's forward torso lean (deg, neck vs pelvis) in the SMPL joints before tokenization; 0 = off. Default: $TOKEN_SVC_OPERATOR_LEAN_CAP.")
    ap.add_argument("--operator-reach-cap", type=float, default=None,
                    help="hard-limit the operator's forward hand reach (m, wrist ahead of the shoulder); 0 = off. Default: $TOKEN_SVC_OPERATOR_REACH_CAP.")
    ap.add_argument("--ori-mode", choices=("full", "heading"), default=None,
                    help="obs root-ori anchoring; default: tokenizer ONNX "
                         "'ori_mode' metadata (REFUSES to start if neither)")
    args = ap.parse_args()
    global _ROOT_LEVEL
    if args.operator_root_level is not None:
        _ROOT_LEVEL = '' if args.operator_root_level.strip().lower() == 'off' else args.operator_root_level.strip()
    print(f"[token-svc] operator-root leveling: {_ROOT_LEVEL or 'off'}", flush=True)
    global _LEAN_CAP, _REACH_CAP
    if args.operator_lean_cap is not None: _LEAN_CAP = float(args.operator_lean_cap)
    if args.operator_reach_cap is not None: _REACH_CAP = float(args.operator_reach_cap)
    print(f"[token-svc] operator pose caps: lean {_LEAN_CAP:.1f} deg, reach {_REACH_CAP:.2f} m (0 = off)", flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.tokenizer,
                                providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    ori_mode = args.ori_mode or meta.get("ori_mode")
    if ori_mode not in ("full", "heading"):
        print(f"[token-svc] FATAL: tokenizer carries no 'ori_mode' metadata "
              f"and no --ori-mode given (got {ori_mode!r}). The obs "
              f"convention MUST match the encoder's training (v1.1 release "
              f"cores = 'heading'; plain 14k-lineage = 'full') — refusing "
              f"to guess.", flush=True)
        return 2
    print(f"[token-svc] tokenizer {args.tokenizer} (ori_mode={ori_mode})",
          flush=True)

    ctx = zmq.Context.instance()
    intent = ctx.socket(zmq.SUB)
    intent.bind(f"tcp://*:{args.intent_port}")
    intent.setsockopt_string(zmq.SUBSCRIBE, "pico_intent")
    intent.setsockopt(zmq.RCVTIMEO, 0)   # non-blocking drain: at 50 Hz
    # arrivals a 20 ms timeout NEVER expires -> the drain starved the
    # publish path to 0.1 Hz (rehearsal fall #3, 2026-08-29)
    tokens = ctx.socket(zmq.PUB)
    tokens.bind(f"tcp://*:{args.token_port}")
    debug = ctx.socket(zmq.SUB)
    debug.connect(f"tcp://{args.debug_host}:{args.debug_port}")
    debug.setsockopt_string(zmq.SUBSCRIBE, "x2_debug")
    debug.setsockopt(zmq.RCVTIMEO, 0)
    print(f"[token-svc] intent SUB *:{args.intent_port}  token PUB "
          f"*:{args.token_port}  debug {args.debug_host}:{args.debug_port}",
          flush=True)
    import json as _json
    wb_cmd = None
    if int(args.planner_cmd_port) > 0:
        wb_cmd = ctx.socket(zmq.PUB)
        wb_cmd.setsockopt(zmq.LINGER, 0)
        wb_cmd.connect(f"tcp://{args.planner_cmd_host}:{args.planner_cmd_port}")
        print(f"[token-svc] wb_state PUB -> planner_cmd "
              f"{args.planner_cmd_host}:{args.planner_cmd_port}", flush=True)
    wb_keepalive_t = 0.0
    wb_wrist_t = 0.0

    def _wb_state(flag: bool) -> None:
        if wb_cmd is None:
            return
        wb_cmd.send_multipart([
            b"planner_cmd",
            _json.dumps({"intent": "wb_state", "engaged": int(bool(flag)),
                         "source": "wb"}).encode("utf-8")])

    ring: deque = deque(maxlen=128)   # (t, joints(24,3), quat_xyzw)
    state = {"quat_wxyz": np.array([1.0, 0, 0, 0], np.float32),
             "body_q_mj": None, "t": 0.0}
    engaged_in, wrist_pr = 0.0, np.zeros(4, np.float32)
    yaw_align = None
    engaged_prev = False
    intent_alive_prev = False
    span_buf = {"t": [], "sj": [], "hq": []}
    engage_warmup_until = 0.0
    default_mj = None            # learned from first body_q (stand ~ default)
    frozen = None
    frame_idx = 0
    n_tok, n_int, n_dbg = 0, 0, 0
    t_rep = time.monotonic()
    t_diag = 0.0

    def window(now):
        if len(ring) < 4:
            return None
        ts = np.array([r[0] for r in ring])
        want = now - DELAY_S + np.arange(SMPL_WINDOW) * SMPL_DT
        idx = np.searchsorted(ts, want, side="right") - 1
        idx = idx.clip(0, len(ring) - 1)
        return (np.stack([ring[i][1] for i in idx]),
                np.stack([ring[i][2] for i in idx]))

    next_t = time.monotonic()
    while True:
        # drain intent + debug
        while True:
            try:
                raw = intent.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                f = unpack_message(raw, expected_topic="pico_intent").fields
                j = np.asarray(f["smpl_joints"], np.float32).reshape(24, 3)
                q = np.asarray(f["human_quat"], np.float32).ravel()[:4]
                ring.append((time.monotonic(), j, q))
                n_int += 1
                engaged_in = float(np.asarray(f.get("engaged", [0.0])).ravel()[0])
                if "wrist_pr" in f:
                    wrist_pr = np.asarray(f["wrist_pr"], np.float32).ravel()[:4]
            except Exception as e:
                print(f"[token-svc] intent decode error: {e!r}", flush=True)
        while True:
            try:
                raw = debug.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                f = unpack_message(raw, expected_topic="x2_debug").fields
                state["quat_wxyz"] = np.asarray(f["base_quat"], np.float32).ravel()[:4]
                bq = f.get("body_q")
                if bq is not None:
                    state["body_q_mj"] = np.asarray(bq, np.float32).ravel()
                state["t"] = time.monotonic()
                n_dbg += 1
            except Exception:
                pass

        now = time.monotonic()
        if now < next_t:
            time.sleep(min(next_t - now, 0.004))
            continue
        next_t += CONTROL_DT

        intent_fresh = len(ring) > 0 and (now - ring[-1][0]) < 0.5
        # FROZEN-INTENT GUARD (2026-09-01, defense in depth): a live human
        # solve is never byte-identical twice ~0.5 s apart -- but a frozen
        # Pico body stream is (25 min of one bit-identical, 26deg-pitched
        # frame; the robot tracked that mannequin and fell at every
        # engage). The sender now guards this too, but THIS service is the
        # last gate before the robot: refuse engage on frozen frames from
        # ANY upstream bug, present or future.
        intent_alive = (
            len(ring) >= 25
            and not (np.array_equal(ring[-1][1], ring[-25][1])
                     and np.array_equal(ring[-1][2], ring[-25][2])))
        if intent_alive != intent_alive_prev:
            _st = ("ALIVE" if intent_alive else
                   "*** FROZEN — engage blocked (identical frames; "
                   "body tracking dead?) ***")
            print(f"[token-svc] intent stream {_st}", flush=True)
            intent_alive_prev = intent_alive
        state_fresh = (now - state["t"]) < 0.5 and state["t"] > 0.0
        engaged = (engaged_in > 0.5 and intent_fresh and state_fresh
                   and intent_alive)

        # DIAG 2026-09-01 (engage forward-lean): print the human root
        # pitch/roll the encoder actually sees. ori_mode=heading anchors
        # YAW only — human pitch/roll passes into the reference verbatim,
        # so a pitched Pico root = commanded forward lean while the
        # operator stands straight.
        if len(ring) > 0 and now - t_diag >= 1.0:
            t_diag = now
            hq = ring[-1][2]
            hr = Rot.from_quat([hq[1], hq[2], hq[3], hq[0]])
            he = hr.as_euler("ZYX", degrees=True)  # yaw, pitch, roll
            bq = state["quat_wxyz"]
            br = Rot.from_quat([bq[1], bq[2], bq[3], bq[0]])
            be = br.as_euler("ZYX", degrees=True)
            print(f"[token-svc] DIAG human ypr=({he[0]:+.0f},{he[1]:+.0f},"
                  f"{he[2]:+.0f})deg  robot ypr=({be[0]:+.0f},{be[1]:+.0f},"
                  f"{be[2]:+.0f})deg  {'ENGAGED' if engaged else 'disengaged'}",
                  flush=True)

        # Heartbeat EVERY 5 s regardless of engage state — the operator's
        # flow confirmation (mirrors the sender's "[intent] 50.0 Hz" line).
        if now - t_rep >= 5.0:
            dt = now - t_rep
            print(f"[token-svc] in: intent {n_int / dt:.1f} Hz  "
                  f"robot-state {n_dbg / dt:.1f} Hz  |  out: tokens "
                  f"{n_tok / dt:.1f} Hz  |  "
                  f"{'ENGAGED' if engaged else 'disengaged'}"
                  f"{'' if intent_fresh else '  [NO INTENT STREAM]'}"
                  f"{'' if intent_alive else '  [FROZEN INTENT]'}"
                  f"{'' if state_fresh else '  [NO ROBOT STATE]'}",
                  flush=True)
            n_tok, n_int, n_dbg, t_rep = 0, 0, 0, now

        if engaged and now - wb_keepalive_t >= 0.5:
            _wb_state(True)
            wb_keepalive_t = now
        if engaged != engaged_prev:
            _wb_state(engaged)
            wb_keepalive_t = now
            print(f"[token-svc] {'ENGAGED' if engaged else 'released'} "
                  f"(intent_fresh={intent_fresh} state_fresh={state_fresh})",
                  flush=True)
            engaged_prev = engaged
            if engaged:
                yaw_align = None
                # ENGAGE WARM-UP (2026-09-01): the smpl encoder's rolling
                # window is stale at engage, so the first ~1 s of tokens
                # decode a crouch-forward transient (sim: knee target ->
                # 0.64, waist_pitch -> 0.45 while the operator stands
                # straight; the sim twin falls inside it). Swallow that
                # first second: deploy sees a stale token stream and keeps
                # pad authority, then engages on a SETTLED reference. The
                # deploy-side half of this fix is SoftStartRamp blending
                # from the measured pose (see 20260901_engage_crouch_dip.md).
                engage_warmup_until = now + 1.0
        if not engaged:
            frozen = None
            # SPAN CAPTURE (2026-09-01, operator ask): dump the smpl frames
            # this service actually fed the robot, on THIS machine's clock
            # (same monotonic domain as the deploy CSVs) -- corpus sidecars
            # then need NO cross-machine alignment. One npz per engaged span.
            if span_buf["t"]:
                import os
                os.makedirs(os.path.join(_PC2_PREFIX, "log/intent_spans"),
                            exist_ok=True)
                fn = os.path.expanduser(
                    os.path.join(_PC2_PREFIX, f"log/intent_spans/span_{int(span_buf['t'][0])}.npz"))
                np.savez_compressed(
                    fn,
                    t=np.asarray(span_buf["t"], np.float64),
                    smpl_joints=np.asarray(span_buf["sj"], np.float32),
                    human_quat=np.asarray(span_buf["hq"], np.float32),
                    clock="pc2_monotonic (same as deploy csv t)")
                print(f"[token-svc] span capture: {len(span_buf['t'])} frames "
                      f"-> {fn}", flush=True)
                span_buf = {"t": [], "sj": [], "hq": []}
            continue     # nothing published -> deploy token stream stale -> pad
        if now < engage_warmup_until:
            continue     # engage warm-up: window still filling; hold pad authority

        win = window(now)
        if win is None:
            continue
        joints, quats = win
        span_buf["t"].append(now)
        span_buf["sj"].append(joints[-1])
        span_buf["hq"].append(quats[-1])
        bq_wxyz = state["quat_wxyz"]
        if yaw_align is None:
            robot = Rot.from_quat([bq_wxyz[1], bq_wxyz[2], bq_wxyz[3], bq_wxyz[0]])
            # human quats are canonical WXYZ end-to-end (sidecar field is
            # global_orient_quat_wxyz; build_smpl_obs reorders internally).
            # Feeding them to scipy un-reordered gave a garbage heading
            # anchor -> every root6d frame rotated -> dive (fall #5).
            hq = quats[-1]
            human = Rot.from_quat([hq[1], hq[2], hq[3], hq[0]])
            yaw_align = Rot.from_euler("z", _heading(robot) - _heading(human))
            print(f"[token-svc] heading anchor human "
                  f"{np.degrees(_heading(human)):+.0f} -> robot "
                  f"{np.degrees(_heading(robot)):+.0f} deg", flush=True)
        body_q = state["body_q_mj"]
        if default_mj is None and body_q is not None:
            default_mj = body_q[:NUM_X2].copy()   # first sighting ~ stand pose
        # measured wrists for the obs (MJ->IL: wrist rows are the same six
        # dofs; use MJ values ordered [lp lr rp rr ...] mapped into the IL
        # slice via the known MJ rows)
        wrist6 = np.zeros(6, np.float32)
        if body_q is not None:
            wrist6[0] = body_q[19] if body_q.shape[0] > 19 else 0.0  # l yaw
            wrist6[1] = body_q[WRIST_MJ["lp"]]
            wrist6[2] = body_q[WRIST_MJ["lr"]]
            wrist6[3] = body_q[26] if body_q.shape[0] > 26 else 0.0  # r yaw
            wrist6[4] = body_q[WRIST_MJ["rp"]]
            wrist6[5] = body_q[WRIST_MJ["rr"]]
        if _ROOT_LEVEL:
            # ROOT LEVELING (sim test 2026-09-15): the Pico root carries a steady
            # forward pitch that the SMPL path passes into the reference as a
            # commanded lean (yaw is the only anchored axis). Keep the heading,
            # zero (or clamp) pitch/roll of every window frame; the torso bend
            # still reaches the tokenizer through the joint positions.
            _lim = None; _off = None
            if _ROOT_LEVEL.startswith("clamp:"):
                _lim = np.radians(float(_ROOT_LEVEL.split(":", 1)[1]))
            elif _ROOT_LEVEL.startswith("offset:"):
                _off = np.radians(float(_ROOT_LEVEL.split(":", 1)[1]))   # subtract a fixed forward pitch (deg), roll untouched
            _q = np.asarray(quats, np.float32).copy()
            for _i in range(_q.shape[0]):
                _e = Rot.from_quat([_q[_i][1], _q[_i][2], _q[_i][3], _q[_i][0]]).as_euler("ZYX")
                if _off is not None:
                    _e[1] = _e[1] - _off
                elif _lim is None:
                    _e[1] = 0.0; _e[2] = 0.0
                else:
                    _e[1] = float(np.clip(_e[1], -_lim, _lim)); _e[2] = float(np.clip(_e[2], -_lim, _lim))
                _x = Rot.from_euler("ZYX", _e).as_quat()   # xyzw
                _q[_i] = np.array([_x[3], _x[0], _x[1], _x[2]], np.float32)
            quats = _q
        if _LEAN_CAP > 0.0 or _REACH_CAP > 0.0:
            joints = _cap_operator_pose(joints)
            if _CAP_STATS["frames"] % 5000 < 10:
                print(f"[token-svc] pose caps: lean capped {_CAP_STATS['lean']} / reach capped {_CAP_STATS['reach']} of {_CAP_STATS['frames']} frames", flush=True)
        obs = build_smpl_obs(joints, quats, bq_wxyz, wrist6,
                             yaw_align=yaw_align, ori_mode=ori_mode)
        tok = sess.run(["motion_token"],
                       {"smpl_obs": obs[None].astype(np.float32)})[0].ravel()

        wr = (default_mj.copy() if default_mj is not None
              else np.zeros(NUM_X2, np.float32))
        wr[WRIST_MJ["lp"]] = np.clip(wrist_pr[0], *WRIST_RANGE["p"])
        wr[WRIST_MJ["lr"]] = np.clip(wrist_pr[1], *WRIST_RANGE["r"])
        wr[WRIST_MJ["rp"]] = np.clip(wrist_pr[2], *WRIST_RANGE["p"])
        wr[WRIST_MJ["rr"]] = np.clip(wrist_pr[3], *WRIST_RANGE["r"])
        # The deploy's --wrist-bypass ik samples the POSE-REF (kplanner)
        # wrists, not this token message (x2_deploy_onnx_ref.cpp ~3198), so
        # the operator's wrists never reached the robot in whole-body mode
        # ("dummy wrist pose", 2026-09-04). Forward them to the kplanner's
        # wrist overlay over planner_cmd (25 Hz while engaged); the overlay
        # latches the last targets on release (arm_release clears).
        if wb_cmd is not None and now - wb_wrist_t >= 0.04:
            wb_wrist_t = now
            wb_cmd.send_multipart([
                b"planner_cmd",
                _json.dumps({"intent": "wrist_targets", "source": "wb",
                             "lp": float(wr[WRIST_MJ["lp"]]), "lr": float(wr[WRIST_MJ["lr"]]),
                             "rp": float(wr[WRIST_MJ["rp"]]), "rr": float(wr[WRIST_MJ["rr"]])}
                            ).encode("utf-8")])

        msg = pack_pose_message({
            "joint_pos_mj": wr.astype(np.float32),
            "root_quat_xyzw": np.array([bq_wxyz[1], bq_wxyz[2], bq_wxyz[3],
                                        bq_wxyz[0]], np.float32),
            "motion_token": tok.astype(np.float32),
            "engaged": np.array([1.0], np.float32),
            "frame_index": np.array([frame_idx], np.int64),
        }, topic="pose", version=4)
        tokens.send(msg)
        frame_idx += 1
        n_tok += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
