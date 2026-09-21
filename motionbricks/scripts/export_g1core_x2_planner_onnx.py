#!/usr/bin/env python
"""Export the FROZEN G1 stock planner wrapped in analytic Phi heads as an
X2-contract ONNX graph (frozen-core S0).

Graph interface is byte-compatible with x2_planner_template.onnx /
x2_planner_velocity.onnx (export_x2_planner_onnx.py), so the pc2 daemon and
walk_seam_sweep run it unmodified:

  inputs:  context_mujoco_qpos [1, 4, 38] f32 (world frame, wxyz quat)
           velocity_intent     [1, 4]     f32 [yaw_rate, vel_x, vel_z, hip_h]
           mode [1] i64, random_seed [1] i64      (template graph only)
  outputs: mujoco_qpos [1, 64, 38] f32, num_pred_frames [1] i64

Internally: PhiEncode (X2 qpos38 -> G1 qpos36) -> frozen G1 2M-step core
(template or velocity replan) -> PhiDecode (G1 qpos36 -> X2 qpos38).

Phi is the ANALYTIC alignment (FrozenCore kinematic-alignment stage):
  * joint map: same-named joints 1:1 + the vendor wrist-naming swap
    (x2_wrist_yaw <- -g1_wrist_roll; x2_wrist_roll <- g1_wrist_yaw),
    per-joint affine x2 = a*g1 + b. Arms: stage-4 jointmap calibration
    (g1_to_x2_ultra_calibration.json, verified R^2 0.91-1.00 on 60 paired
    executed-feasible clips / 37k frames). Legs/waist: empirical affine fit
    on the same pairs (R^2 0.1-0.9 — X2 legs are IK-retargeted, a per-joint
    affine is the best analytic can do; the S1 learned heads absorb the
    residual). Encoder uses the exact inverses.
  * head (X2 dof 29/30): dropped on encode, decoded as 0 (planner never
    owns the head; the daemon head-look overlay does).
  * root: xy DELTAS scaled about the last context frame (stride scale
    0.8235, retarget position_scale; empirical 0.821), z affine
    x2 = 0.8109*g1 + 0.0509 m (fit), quat passthrough (heading identical
    in the paired corpus; lean diff <= 13 deg ignored at S0).
  * hip_h intent channel mapped through the inverse z affine.
  * mode: X2 {0 idle, 1 slow_walk, 2 walk, 3 run} -> G1 clip {0, 1, 2, 2}
    (G1-clip has no run; precedent kplanner_gen_from_log_g1.G1_MODE_MAP).

Usage:
  python motionbricks/scripts/export_g1core_x2_planner_onnx.py \
      --out-dir $X2_EVAL_ROOT/kplanner_g1core --mode template --smoke
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch as t
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MB_ROOT = _REPO_ROOT / "motionbricks"
for p in (str(_MB_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

log = logging.getLogger("export_g1core_x2")

X2_QPOS = 38
G1_QPOS = 36
NUM_CTX = 4
PADDED_FRAMES = 64
DEG = np.pi / 180.0

# ---------------------------------------------------------------------------
# Analytic Phi tables.
#
# X2 dof order = deploy mujoco_joint_names (policy_parameters.hpp), 31 dofs.
# G1 dof order = soma CSV header (record_motion_to_pkl.py), 29 dofs.
# Per X2 dof i: (g1_index, a, b_rad) with x2 = a * g1 + b. Head rows use
# g1_index = -1 (decoded as constant 0, dropped on encode).
#
# Arms: g1_to_x2_ultra_calibration.json shoulder_elbow_fit (degrees -> rad
# here) + wrist_remap. Legs/waist: empirical affine fit, 60 paired clips
# (scratchpad fit_phi.py, 2026-08-12). Wrist a from the empirical fit
# (calibration remap is sign-only; fit gave 0.94-0.997 ~ keep exact fit).
# ---------------------------------------------------------------------------

# (x2_name, g1_idx, a, b_deg)
_PHI_TABLE = [
    ("left_hip_pitch",      0, 0.950,   1.64),
    ("left_hip_roll",       1, 0.917,  -4.92),
    ("left_hip_yaw",        2, 0.899,   5.39),
    ("left_knee",           3, 0.928,  -5.87),
    ("left_ankle_pitch",    4, 0.807,   0.94),
    ("left_ankle_roll",     5, 0.697,   3.72),
    ("right_hip_pitch",     6, 0.886,  -2.98),
    ("right_hip_roll",      7, 1.102,   2.96),
    ("right_hip_yaw",       8, 0.932,  -1.21),
    ("right_knee",          9, 0.941,  -5.61),
    ("right_ankle_pitch",  10, 0.762,  -0.07),
    ("right_ankle_roll",   11, 0.836,  -4.02),
    ("waist_yaw",          12, 0.994,  -4.56),
    # G1 CSV waist order is yaw, ROLL, pitch; X2 is yaw, PITCH, roll.
    ("waist_pitch",        14, 1.287,   1.39),
    ("waist_roll",         13, 1.119,   0.64),
    # arms: calibration shoulder_elbow_fit (exact stage-4 jointmap)
    ("left_shoulder_pitch",  15, 0.94005,  -0.4308),
    ("left_shoulder_roll",   16, 0.76055, -18.1149),
    ("left_shoulder_yaw",    17, 0.85126,  -4.3932),
    ("left_elbow",           18, 1.11940, -79.1041),
    # G1 arm order: ..., elbow(18), wrist_roll(19), wrist_pitch(20), wrist_yaw(21)
    ("left_wrist_yaw",       19, -1.000,   0.0),   # <- g1 left_wrist_roll, sign flip
    ("left_wrist_pitch",     20, 0.943,    0.0),
    ("left_wrist_roll",      21, 0.973,    0.0),   # <- g1 left_wrist_yaw
    ("right_shoulder_pitch", 22, 0.69864,   0.2067),
    ("right_shoulder_roll",  23, 0.99352,  21.1384),
    ("right_shoulder_yaw",   24, 0.93100,  -1.8982),
    ("right_elbow",          25, 1.13695, -85.3979),
    ("right_wrist_yaw",      26, -1.000,   0.0),   # <- g1 right_wrist_roll, sign flip
    ("right_wrist_pitch",    27, 0.929,    0.0),
    ("right_wrist_roll",     28, 0.960,    0.0),   # <- g1 right_wrist_yaw
    ("head_yaw",             -1, 0.0,      0.0),
    ("head_pitch",           -1, 0.0,      0.0),
]
assert len(_PHI_TABLE) == 31

# root z / stride scaling (fit + retarget config)
_XY_SCALE = 0.8235          # x2 stride = 0.8235 * g1 stride
# z_x2 = a * z_g1 + b [m], b from the paired-corpus fit. The FK'd stance
# feet sit ~1.7 cm ABOVE the old planner's convention (9.4 vs 7.7 cm
# ankle_roll z p50) — and that float is LOAD-BEARING: grounding b to
# match the old convention (0.0339) verified at 7.7 cm but degraded
# closed-loop recovery steps ~3x (grounded seeds {11,12,15} vs floating
# {1,2,3,4}, walk 0.35, SONIC 14000, 2026-08-12 A/B). SONIC tracks the
# floating reference cleaner — do NOT "fix" this constant against
# offline contact stats; the seam sweep is the arbiter.
_Z_A, _Z_B = 0.8109, 0.0509

# G1 clip-library identities, measured from the clips' OWN kinematics
# (2026-08-12; the inherited "0=idle 2=walk" comment was WRONG — 0..2 are
# all idle variants): 3=deep squat, 4=walk+turn, 6/7=walks(.35/.80),
# 8=crouch run(z.49), 9=mild crouch walk, 10=rich slow walk(.36, 90f),
# 11=fast crouched, 13/14=step-turn pair (-/+95 dps).
# WALK templates: back to the IDLE-variant templates (0/1) — the routed
# clip-10 walk gated identically ({1,3,3} vs {1,2,3,4}) but the operator
# prefers the idle-template gait STYLE ("walk looked great" / "perfectly
# matches G1"); style is a free choice at equal gate numbers. Turns and
# crouch keep their dedicated clips (stepping + height matter there).
# walk stays on the operator-approved idle-variant templates; RUN goes to
# the real fast-walk clip 7 — idle-template runs hang the arms (operator,
# first VR session: "unnatural without arms moving"; arm style follows the
# POSE TEMPLATE, the root model only drives locomotion).
_G1_IDLE, _G1_WALK_SLOW, _G1_WALK_FAST = 0, 1, 2
_G1_RUN = 7
_G1_CROUCH = 8
# Injected step-turn modes (build_g1_turn_clip_library.py, 2026-08-12):
# G1-clip-turns.ckpt modes 15/16 = the idle-turn-270 take (+ its mirror), the
# heuristic-era X2-proven take family, G1-NATIVE side, -/+27 dps steady.
_G1_TURN_R, _G1_TURN_L = 15, 16
# Operator-captured gait-initiation templates (2026-08-13,
# build_g1_walkstart_clip_library.py from the tau=1.2 recording session):
# G1-clip-turns-walkstart.ckpt modes 17/18 = walk_start_left/right.
# Routed when the CONTEXT is stationary but the intent wants forward walk
# (stateless: the 4-frame ctx itself shows the standstill) — fixes the
# core-native first-stride surge without touching ramps or weights.
_G1_WALK_START = 17
_START_DISP_MAX = 0.01   # m: ctx xy travel below this = standing
_START_VEL_MIN = 0.10    # m/s commanded forward to trigger
# Static fallback LUT for X2 mode when the intent router does not
# override: idle->idle, slow_walk->slow walk, walk->fast walk, run->fast.
_MODE_LUT = [_G1_IDLE, _G1_WALK_SLOW, _G1_WALK_FAST, _G1_RUN]
# Intent-aware routing thresholds (applied in-graph):
_TURN_YAW_MIN = 0.25    # rad/s: yaw-dominant if |yaw| above AND speed below
_TURN_VEL_MAX = 0.15    # m/s
_CROUCH_HIP_MAX = 0.60  # m (X2 frame): hip_h below -> crouch template


def _phi_tensors(device="cpu"):
    """Constant tensors for encode (X2->G1) and decode (G1->X2)."""
    g1_idx = t.tensor([g for _, g, _, _ in _PHI_TABLE], dtype=t.long, device=device)
    a = t.tensor([a for _, _, a, _ in _PHI_TABLE], dtype=t.float32, device=device)
    b = t.tensor([b * DEG for _, _, _, b in _PHI_TABLE], dtype=t.float32, device=device)
    matched = g1_idx >= 0  # [31] bool; head rows False
    # encode scatter: for each matched x2 dof i -> g1 slot g1_idx[i],
    # g1 = (x2 - b) / a.  Every G1 dof has exactly one X2 source (verified
    # by the assert below), so a scatter fully populates the 29-vector.
    src_of_g1 = t.full((29,), -1, dtype=t.long, device=device)
    for i, (_, g, _, _) in enumerate(_PHI_TABLE):
        if g >= 0:
            assert src_of_g1[g] == -1, f"duplicate source for g1 dof {g}"
            src_of_g1[g] = i
    assert (src_of_g1 >= 0).all(), "unmapped G1 dof"
    return g1_idx, a, b, matched, src_of_g1


class PhiEncode(nn.Module):
    """X2 context qpos [B, T, 38] -> G1 context qpos [B, T, 36]."""

    def __init__(self):
        super().__init__()
        g1_idx, a, b, matched, src_of_g1 = _phi_tensors()
        self.register_buffer("src_of_g1", src_of_g1)          # [29] x2 dof idx
        self.register_buffer("a_g1", a.index_select(0, src_of_g1))   # [29]
        self.register_buffer("b_g1", b.index_select(0, src_of_g1))   # [29]

    def forward(self, x2_qpos: t.Tensor) -> t.Tensor:
        root_xyz = x2_qpos[..., 0:3]
        quat = x2_qpos[..., 3:7]
        dof = x2_qpos[..., 7:38]
        # anchor xy on the LAST context frame; scale deltas up to G1 stride
        anchor = root_xyz[:, -1:, 0:2]
        xy = anchor + (root_xyz[..., 0:2] - anchor) / _XY_SCALE
        z = (root_xyz[..., 2:3] - _Z_B) / _Z_A
        g1_dof = (dof.index_select(-1, self.src_of_g1) - self.b_g1) / self.a_g1
        return t.cat([xy, z, quat, g1_dof], dim=-1)


class PhiDecode(nn.Module):
    """G1 frames qpos [B, T, 36] -> X2 frames qpos [B, T, 38].

    anchor_xy: [B, 1, 2] — the same anchor PhiEncode used (last X2 context
    frame), so context and prediction stay in one consistent world frame.
    """

    def __init__(self):
        super().__init__()
        g1_idx, a, b, matched, _ = _phi_tensors()
        self.register_buffer("g1_idx", g1_idx.clamp(min=0))   # [31]
        self.register_buffer("a_x2", a)                       # [31]
        self.register_buffer("b_x2", b)                       # [31]
        self.register_buffer("matched", matched.float())      # [31]

    # Reference-side waist guardrail: G1's running-turn waist maps through
    # our WEAKEST affines with >1 gains (pitch x1.287, roll x1.119), so the
    # decoded reference can reach ~+43 deg roll / +39 deg pitch at
    # ~180 deg/s, which collapses SONIC in sim. Clamp the waist channels to
    # the deploy-safety band (0.45 rad ~ 26 deg, the 24-vs-32 Nm saturation
    # limit). S1's learned decoder owns the real fix.
    _WAIST_CLAMP_RAD = 0.45

    def forward(self, g1_qpos: t.Tensor, anchor_xy: t.Tensor) -> t.Tensor:
        xy = anchor_xy + (g1_qpos[..., 0:2] - anchor_xy) * _XY_SCALE
        z = g1_qpos[..., 2:3] * _Z_A + _Z_B
        quat = g1_qpos[..., 3:7]
        g1_dof = g1_qpos[..., 7:36]
        x2_dof = (g1_dof.index_select(-1, self.g1_idx) * self.a_x2 + self.b_x2)
        x2_dof = x2_dof * self.matched  # head rows -> 0
        waist = x2_dof[..., 12:15].clamp(min=-self._WAIST_CLAMP_RAD,
                                         max=self._WAIST_CLAMP_RAD)
        x2_dof = t.cat([x2_dof[..., :12], waist, x2_dof[..., 15:]], dim=-1)
        return t.cat([xy, z, quat, x2_dof], dim=-1)


class IntentMap(nn.Module):
    """velocity_intent X2->G1: hip_h through the inverse z affine; linear
    velocities scaled up by 1/xy_scale so the DECODED X2 speed matches the
    command (G1 realizes v/0.8235, decode scales strides back by 0.8235).
    yaw_rate is rotation -- scale-invariant, passthrough."""

    def forward(self, intent: t.Tensor) -> t.Tensor:
        yaw = intent[..., 0:1]
        vel = intent[..., 1:3] / _XY_SCALE
        hip_g1 = (intent[..., 3:4] - _Z_B) / _Z_A
        return t.cat([yaw, vel, hip_g1], dim=-1)


# ---------------------------------------------------------------------------
# Fused wrappers (reuse the X2 export scaffolding against the G1 core)
# ---------------------------------------------------------------------------


def _import_x2_export_bits():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_x2_planner_onnx", _MB_ROOT / "scripts/export_x2_planner_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _HeadedPhi(nn.Module):
    """Applies an S1 residual head over non-overlapping 4-frame windows.

    Mirrors train_frozen_core_heads._apply_enc/_apply_dec EXACTLY: the head
    READS the source-side window (f_enc: X2 ctx; f_dec: G1 plan) and its
    residual is ADDED to the analytic-Phi TARGET (root xyz + 29 non-head
    dof; quat passthrough; X2 head joints stay analytic zero). T must be
    a multiple of 4 (context 4 = 1 window; plan 64 = 16 — the VQVAE
    token grid). ONNX-safe: cat, no in-place slice writes."""

    def __init__(self, head: nn.Module, dst_has_head_joints: bool,
                 windowed: bool):
        super().__init__()
        self.head = head
        self.dst_tail = dst_has_head_joints
        self.windowed = windowed  # s1a MLP heads; s1c convs run full-T

    def forward(self, src_qpos: t.Tensor, dst_qpos: t.Tensor) -> t.Tensor:
        b, T, ds = src_qpos.shape
        if self.windowed:
            r = self.head(src_qpos.reshape(b * (T // 4), 4, ds))
            r = r.reshape(b, T, r.shape[-1])                 # [B,T,3+29]
        else:
            r = self.head(src_qpos)                          # stationary conv
        parts = [dst_qpos[..., 0:3] + r[..., 0:3],
                 dst_qpos[..., 3:7],
                 dst_qpos[..., 7:36] + r[..., 3:]]
        if self.dst_tail:
            parts.append(dst_qpos[..., 36:38])
        return t.cat(parts, dim=-1)


class FusedTemplateWrapper(nn.Module):
    """Template wrapper with an INTENT-AWARE mode router (all tensor ops):
    yaw-dominant + near-zero linear velocity -> step-turn template by yaw
    sign; hip_h below the crouch threshold -> crouch template; otherwise
    the static LUT (idle / slow walk / fast walk).

    heads (S1): optional (f_enc, f_dec) residual modules — applied after
    PhiEncode / after PhiDecode respectively, 4-frame-window granular."""

    def __init__(self, core, xexp, heads=None):
        super().__init__()
        self.enc, self.dec, self.imap = PhiEncode(), PhiDecode(), IntentMap()
        windowed = heads is not None and heads[2] not in ("s1c", "s1d", "s1e")
        self.f_enc = (_HeadedPhi(heads[0], False, windowed)
                      if heads and heads[0] is not None else None)
        self.f_dec = _HeadedPhi(heads[1], True, windowed) if heads else None
        self.inner = xexp.TemplateReplanWrapper(core)
        self.register_buffer("mode_lut", t.tensor(_MODE_LUT, dtype=t.long))

    def _route_mode(self, mode, velocity_intent, context_mujoco_qpos):
        # Turn routing v2 (2026-08-12): INJECTED library modes 15/16 — the
        # X2-proven heuristic-era turn take, G1-native side. Supersedes the
        # rejected stock clips 13/14/9 and the sliding fallback.
        base = self.mode_lut.index_select(0, mode.reshape([1]).clamp(min=0, max=3))
        yaw = velocity_intent[0, 0]
        speed = velocity_intent[0, 1:3].abs().amax()
        hip = velocity_intent[0, 3]
        is_idle = (mode.reshape([1]) == 0)
        turnish = (yaw.abs() > _TURN_YAW_MIN) & (speed < _TURN_VEL_MAX) & (~is_idle[0])
        turn_mode = t.where(yaw < 0,
                            t.tensor(_G1_TURN_R, device=yaw.device),
                            t.tensor(_G1_TURN_L, device=yaw.device)).reshape([1])
        crouch = (hip < _CROUCH_HIP_MAX) & (~is_idle[0])
        routed = t.where(turnish, turn_mode, base)
        routed = t.where(crouch & ~turnish,
                         t.tensor([_G1_CROUCH], device=routed.device), routed)
        # speed-aware WALK template (operator, 2026-08-13 run/walk test:
        # "gait still looks like faster slow-walk steps instead of longer
        # regular strides"): the idle-variant templates 0-2 carry the
        # operator-preferred SLOW gait style, but stride style follows
        # the POSE TEMPLATE — pushed past ~0.55 m/s they hurry small
        # steps. Route to the real fast-walk clip (7, 0.80 m/s native)
        # when the commanded speed is above the style crossover.
        fastish = (speed > 0.55) & (base[0] >= 1) & (base[0] <= 2) \
            & (~turnish) & (~crouch)
        routed = t.where(fastish,
                         t.tensor([_G1_RUN], device=routed.device), routed)
        # gait initiation: context shows STANDSTILL + intent wants forward
        # -> operator-captured walk_start template (highest precedence
        # after turn/crouch exclusions; fires for exactly the first
        # plan(s) — once the robot moves, ctx displacement kills it).
        ctx_xy = context_mujoco_qpos[0, :, 0:2]
        disp = (ctx_xy[-1] - ctx_xy[0]).norm()
        startish = ((disp < _START_DISP_MAX) & (speed > _START_VEL_MIN)
                    & (~is_idle[0]) & (~turnish) & (~crouch))
        routed = t.where(startish,
                         t.tensor([_G1_WALK_START], device=routed.device), routed)
        return routed

    def forward(self, context_mujoco_qpos, velocity_intent, mode, random_seed):
        anchor = context_mujoco_qpos[:, -1:, 0:2]
        g1_ctx = self.enc(context_mujoco_qpos)
        if self.f_enc is not None:
            g1_ctx = self.f_enc(context_mujoco_qpos, g1_ctx)
        g1_mode = self._route_mode(mode, velocity_intent, context_mujoco_qpos)
        g1_qpos, npf = self.inner(g1_ctx, self.imap(velocity_intent), g1_mode, random_seed)
        x2_qpos = self.dec(g1_qpos, anchor)
        if self.f_dec is not None:
            x2_qpos = self.f_dec(g1_qpos, x2_qpos)
            # safety guardrail stays OUTERMOST: the residual must not
            # re-open the waist band (saturation limit 0.45 rad, see
            # PhiDecode._WAIST_CLAMP_RAD).
            waist = x2_qpos[..., 19:22].clamp(min=-PhiDecode._WAIST_CLAMP_RAD,
                                              max=PhiDecode._WAIST_CLAMP_RAD)
            x2_qpos = t.cat([x2_qpos[..., :19], waist, x2_qpos[..., 22:]], dim=-1)
        return x2_qpos, npf


class FusedVelocityWrapper(nn.Module):
    def __init__(self, core, xexp):
        super().__init__()
        self.enc, self.dec, self.imap = PhiEncode(), PhiDecode(), IntentMap()
        self.inner = xexp.VelocityReplanWrapper(core)

    def forward(self, context_mujoco_qpos, velocity_intent):
        anchor = context_mujoco_qpos[:, -1:, 0:2]
        g1_ctx = self.enc(context_mujoco_qpos)
        g1_qpos, npf = self.inner(g1_ctx, self.imap(velocity_intent))
        return self.dec(g1_qpos, anchor), npf


# ---------------------------------------------------------------------------


def _load_g1_core(device="cpu"):
    from motionbricks.motion_backbone.inference.load_g1_planner import (
        G1PlannerPaths, load_g1_planner,
    )
    paths = G1PlannerPaths.default()
    paths.validate()
    lib = _MB_ROOT / "out/G1-clip-turns-walkstart.ckpt"
    if not lib.exists():
        lib = _MB_ROOT / "out/G1-clip-turns.ckpt"
    core = load_g1_planner(paths, device=device, clip_library_ckpt=lib)
    log.info("clip library: %s", lib.name)
    torch.backends.mha.set_fastpath_enabled(False)
    n = 0
    for m in core.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
            m.use_nested_tensor = False
            n += 1
    log.info("G1 core loaded; nested-tensor disabled on %d encoders", n)
    return core.eval()


def _x2_stand_context() -> np.ndarray:
    qpos = np.zeros(X2_QPOS, dtype=np.float32)
    qpos[2] = 0.78   # matches the daemon's hardcoded warmup stand
    qpos[3] = 1.0
    return np.tile(qpos[None, None, :], (1, NUM_CTX, 1)).astype(np.float32)


def _smoke(onnx_path: Path, template: bool) -> bool:
    """Finite + directional sanity on the exported graph."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ctx = _x2_stand_context()
    ok = True
    cases = [
        ("idle",    [0.0, 0.0, 0.0, 0.78], 0),
        ("fwd_0.4", [0.0, 0.0, 0.4, 0.78], 1),
        ("turn",    [0.4, 0.0, 0.4, 0.78], 2),
    ]
    for name, intent, mode in cases:
        feeds = {"context_mujoco_qpos": ctx,
                 "velocity_intent": np.array([intent], dtype=np.float32)}
        if template:
            feeds["mode"] = np.array([mode], dtype=np.int64)
            feeds["random_seed"] = np.array([424242], dtype=np.int64)
        qpos, npf = sess.run(None, feeds)
        npf = int(npf.reshape(-1)[0])
        valid = qpos[0, :npf]
        finite = np.isfinite(valid).all()
        head_zero = np.abs(valid[:, 36:38]).max() < 1e-6
        z = valid[:, 2]
        fwd = valid[-1, 0] - ctx[0, -1, 0]
        dur = npf / 30.0
        line = (f"[smoke {name}] npf={npf} finite={finite} head0={head_zero} "
                f"z[{z.min():.3f},{z.max():.3f}] fwd_x={fwd:+.3f}m in {dur:.2f}s")
        bad = (not finite) or (not head_zero) or not (0.45 < z.mean() < 0.95)
        if name.startswith("fwd") and fwd < 0.05:
            bad = True
        ok &= not bad
        log.info("%s %s", line, "FAIL" if bad else "ok")
    return ok


def _load_heads(heads_path: Path, mask_z: bool = False,
                resid_scale: float = 1.0):
    """S1: load trained residual heads (train_frozen_core_heads.py output)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_frozen_core_heads", _MB_ROOT / "scripts/train_frozen_core_heads.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sd = torch.load(heads_path, map_location="cpu", weights_only=True)
    stage, hidden = sd.get("stage", "s1a"), sd.get("hidden", 512)
    if stage in ("s1d", "s1e"):
        mask = torch.zeros(3 + 29)
        mask[2] = 1.0
        mask[3:3 + 15] = 1.0
        f_enc = None
        f_dec = mod.ConvResidualHead(29, 29, hidden=hidden // 2,
                                     k=(1 if stage == "s1d" else 5),
                                     use_vel=False, out_mask=mask)
    elif stage == "s1c":
        f_enc = mod.ConvResidualHead(31, 29, hidden=hidden // 2)
        f_dec = mod.ConvResidualHead(29, 29, hidden=hidden // 2)
    else:
        f_enc = mod.ResidualHead(31, 29, stage, hidden)
        f_dec = mod.ResidualHead(29, 29, stage, hidden)
    if f_enc is not None:
        f_enc.load_state_dict(sd["f_enc"])
        f_enc = f_enc.eval()
    f_dec.load_state_dict(sd["f_dec"])
    if mask_z and hasattr(f_dec, "out_mask"):
        f_dec.out_mask[2] = 0.0
        log.info("root-z residual channel MASKED (S0 float kept)")
    if resid_scale != 1.0 and hasattr(f_dec, "out_mask"):
        f_dec.out_mask.mul_(resid_scale)
        log.info("residual scaled x%.2f", resid_scale)
    log.info("S1 heads loaded: %s (stage=%s hidden=%d enc=%s)",
             heads_path, stage, hidden, f_enc is not None)
    return f_enc, f_dec.eval(), stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(os.environ.get("X2_EVAL_ROOT", Path.home() / ".cache/sonic/x2_eval")) / "kplanner_g1core"))
    ap.add_argument("--mode", choices=["velocity", "template", "both"], default="both")
    ap.add_argument("--heads", type=Path, default=None,
                    help="S1 heads_final.pt — bakes f_enc/f_dec into the graph "
                         "(template mode only)")
    ap.add_argument("--mask-z", action="store_true",
                    help="zero the head's root-z residual channel (keep S0's "
                         "floating z convention — it is LOAD-BEARING; the "
                         "retarget GT z grounds feet and costs ~3x recovery)")
    ap.add_argument("--resid-scale", type=float, default=1.0,
                    help="scale ALL remaining residual channels (closed-loop "
                         "strength sweep: correction vs recovery-step cost)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.manual_seed(0)

    xexp = _import_x2_export_bits()
    core = _load_g1_core("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = torch.from_numpy(_x2_stand_context())
    intent = torch.tensor([[0.0, 0.0, 0.35, 0.78]], dtype=torch.float32)
    rc = 0

    heads = (_load_heads(args.heads, mask_z=args.mask_z,
                         resid_scale=args.resid_scale)
             if args.heads else None)

    if args.mode in ("template", "both"):
        w = FusedTemplateWrapper(core, xexp, heads=heads).eval()
        path = out_dir / "x2_planner_template.onnx"
        xexp._export_graph(
            w, (ctx, intent, torch.tensor([1], dtype=torch.int64),
                torch.tensor([1234], dtype=torch.int64)),
            path,
            input_names=["context_mujoco_qpos", "velocity_intent", "mode", "random_seed"],
            output_names=["mujoco_qpos", "num_pred_frames"])
        if args.smoke and not _smoke(path, template=True):
            rc = 1

    if args.mode in ("velocity", "both"):
        w = FusedVelocityWrapper(core, xexp).eval()
        path = out_dir / "x2_planner_velocity.onnx"
        xexp._export_graph(
            w, (ctx, intent), path,
            input_names=["context_mujoco_qpos", "velocity_intent"],
            output_names=["mujoco_qpos", "num_pred_frames"])
        if args.smoke and not _smoke(path, template=False):
            rc = 1

    (out_dir / "provenance.json").write_text(json.dumps({
        "source": "export_g1core_x2_planner_onnx.py (frozen-core S0)",
        "core": "G1 stock 2M-step (motionbricks/out/*/version_1, frozen)",
        "phi": "analytic: calibration arms + empirical legs/waist affine, "
               "wrist naming swap, xy stride 0.8235, z=0.8109*g1+0.0509",
        "mode_lut": _MODE_LUT,
    }, indent=1))
    log.info("done rc=%d -> %s", rc, out_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
