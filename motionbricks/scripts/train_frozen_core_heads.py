#!/usr/bin/env python
"""Train the S1 residual heads (LP-1 linear probe / S1-a MLP residuals).

frozen-core S1 heads plan. Supervised, chunk-granular:
heads see 4-FRAME WINDOWS (MotionBricks token granularity — operator
constraint, never frame-by-frame), sampled uniformly-at-random from
paired clips built by build_frozen_core_pair_cache.py.

  f_dec:  G1 window  -> delta on Phi_dec output   (target: X2 retarget)
  f_enc:  X2 window  -> delta on Phi_enc output   (target: G1 native)

Residual channels: root dxy (2, yaw-canonical, rotated back to world),
root z (1), non-head dof (29 both robots). Quat passthrough (heading
identical in the paired corpus); X2 head joints stay analytic zero.
Final layers zero-init: at step 0 the system is bit-exact S0.

Corpus gating (tunable at load):
  arm RMSE > --gate-arm-rmse-deg (default 8.0) -> drop. Arms map through
    the calibration-EXACT affines, so genuine pairs agree within a few
    degrees no matter how little the arms move; heavy disagreement =
    branch-flip / corrupted retarget. (v1 used manifest arm_r2>=0.8 and
    gated out 26k/33k clips — R^2 is variance-normalized and near-static
    arms score noisy-low. RMSE fixes that.)
  jerk_p95 > --gate-jerk (default 2500) -> drop (spiky retargets; corpus
    p50=800, p95=2217 — the first run's 500 sat BELOW the median and cut
    23k clips, i.e. it accidentally selected a low-jerk subcorpus)
  knee branch-flip (corr < -0.2 on a moving knee) -> drop; detector
    independently re-finds the known ~808-clip flip list (821 hits).
Band weighting (north star = slow walk < 0.6 m/s):
  mean_speed in [0.05, 0.6) -> weight 2.0; >= 0.6 -> 0.5; idle -> 1.0

LP-1 gate: val report prints per-group MAE vs the Phi-only baseline —
LP-1 must improve legs/waist (it reproduces the affine fit with cross-
frame terms); S1-a must beat LP-1.

Run:
  PYTHONPATH="$PWD:$PWD/motionbricks" .venv/bin/python \
      motionbricks/scripts/train_frozen_core_heads.py --stage lp1
  ... --stage s1a --init-from <lp1_run>/heads_final.pt
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch as t
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "motionbricks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.export_g1core_x2_planner_onnx import (  # noqa: E402
    PhiDecode, PhiEncode, _PHI_TABLE)

WIN = 4  # frames per window == MotionBricks frames-per-token

# ---- channel groups (X2 dof index space, 31) -------------------------------
_X2_LEGS = list(range(0, 12))
_X2_WAIST = list(range(12, 15))
_X2_ARMS = list(range(15, 29))
_X2_HEAD = [29, 30]
_X2_NONHEAD = _X2_LEGS + _X2_WAIST + _X2_ARMS          # 29
# G1 dof index space (29): same layout, no head
_G1_LEGS = list(range(0, 12))
_G1_WAIST = list(range(12, 15))
_G1_ARMS = list(range(15, 29))


def _quat_to_yaw(q: t.Tensor) -> t.Tensor:
    """wxyz quat [..., 4] -> heading yaw about world z."""
    w, x, y, z = q.unbind(-1)
    return t.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_to_rot6d(q: t.Tensor) -> t.Tensor:
    w, x, y, z = q.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - w * z)
    r10 = 2 * (x * y + w * z);     r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - w * y);     r21 = 2 * (y * z + w * x)
    return t.stack([r00, r10, r20, r01, r11, r21], dim=-1)


def window_features(qpos: t.Tensor, n_dof: int) -> t.Tensor:
    """[B, 4, 7+n_dof] qpos window -> flat canonical features [B, F].

    Canonical frame: anchor = LAST frame's root xy + yaw. Features per
    frame: dxy(2), z(1), rot6d(6, yaw-removed), dof(n_dof)."""
    xy, z, quat, dof = qpos[..., 0:2], qpos[..., 2:3], qpos[..., 3:7], qpos[..., 7:]
    yaw = _quat_to_yaw(quat[:, -1])                       # [B]
    c, s = t.cos(-yaw), t.sin(-yaw)
    dxy = xy - xy[:, -1:, :]
    dxy = t.stack([c[:, None] * dxy[..., 0] - s[:, None] * dxy[..., 1],
                   s[:, None] * dxy[..., 0] + c[:, None] * dxy[..., 1]], dim=-1)
    # remove heading from quat: q_local = qz(-yaw) * q
    hw, hz = t.cos(-yaw / 2)[:, None], t.sin(-yaw / 2)[:, None]
    w, x, y, zz = quat.unbind(-1)
    ql = t.stack([hw * w - hz * zz, hw * x - hz * y,
                  hw * y + hz * x, hw * zz + hz * w], dim=-1)
    feats = t.cat([dxy, z, _quat_to_rot6d(ql), dof], dim=-1)
    return feats.reshape(feats.shape[0], -1), yaw


class ResidualHead(nn.Module):
    """Windowed heads (lp1/s1a) — KEPT for checkpoint compat; see gate
    post-mortem 2026-08-12: slot-dependent residual -> 7.5 Hz reference
    jitter, acc_mid 3x, recovery steps 21-32 vs S0 1-3. Superseded by
    ConvResidualHead (s1c)."""

    def __init__(self, n_dof_in: int, n_dof_out: int, stage: str, hidden: int = 512):
        super().__init__()
        d_in = WIN * (2 + 1 + 6 + n_dof_in)
        d_out = WIN * (2 + 1 + n_dof_out)
        self.n_dof_out = n_dof_out
        if stage == "lp1":
            self.net = nn.Linear(d_in, d_out)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, d_out))
        last = self.net if isinstance(self.net, nn.Linear) else self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, qpos_win: t.Tensor) -> t.Tensor:
        feats, yaw = window_features(qpos_win, qpos_win.shape[-1] - 7)
        out = self.net(feats).reshape(-1, WIN, 3 + self.n_dof_out)
        # rotate dxy residual back to world
        c, s = t.cos(yaw)[:, None], t.sin(yaw)[:, None]
        dx = c * out[..., 0] - s * out[..., 1]
        dy = s * out[..., 0] + c * out[..., 1]
        return t.cat([dx[..., None], dy[..., None], out[..., 2:]], dim=-1)


class ConvResidualHead(nn.Module):
    """Stage s1c: TIME-STATIONARY residual — temporal 1D convs, the same
    function at every frame, so a continuous input can NEVER pick up
    window-slot discontinuities (the s1a gate-failure mechanism). Works
    on any T (context 4, plan 64, training segments 16).

    Per-frame input features (all stationary): root z (1), yaw-removed
    root rot6d (6), dof (n_dof_in), plus finite-diff velocities of root
    xy-in-heading-frame (2), z (1), dof (n_dof_in). No window anchors.
    Output per-frame: [dxy(2, heading frame -> world), dz(1), ddof].
    Losses stay CHUNKED at 4 frames (MotionBricks granularity — the
    operator constraint binds the LOSS, not the function class)."""

    def __init__(self, n_dof_in: int, n_dof_out: int, stage: str = "s1c",
                 hidden: int = 256, k: int = 5, use_vel: bool = True,
                 out_mask: t.Tensor | None = None):
        super().__init__()
        self.n_dof_out = n_dof_out
        self.use_vel = use_vel
        d_in = 1 + 6 + n_dof_in + (2 + 1 + n_dof_in if use_vel else 0)
        d_out = 2 + 1 + n_dof_out
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_in, hidden, k, padding=pad, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, k, padding=pad, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(hidden, d_out, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # out_mask: fixed 0/1 per output channel (e.g. legs+waist+z only)
        self.register_buffer(
            "out_mask", out_mask if out_mask is not None else t.ones(d_out))

    def forward(self, qpos: t.Tensor) -> t.Tensor:
        """qpos [B, T, 7+n_dof] -> residual [B, T, 3+n_dof_out]."""
        xy, z, quat, dof = qpos[..., 0:2], qpos[..., 2:3], qpos[..., 3:7], qpos[..., 7:]
        yaw = _quat_to_yaw(quat)                                 # [B, T]
        c, s = t.cos(-yaw), t.sin(-yaw)
        hw, hz = t.cos(-yaw / 2), t.sin(-yaw / 2)
        w, x, y, zz = quat.unbind(-1)
        ql = t.stack([hw * w - hz * zz, hw * x - hz * y,
                      hw * y + hz * x, hw * zz + hz * w], dim=-1)
        feats = [z, _quat_to_rot6d(ql), dof]
        if self.use_vel:
            # velocities (frame diffs, first frame repeats) — deploy
            # hazard: amplifies core-output frame noise (v3 post-mortem)
            dxy_w = t.cat([xy[:, :1] * 0, xy[:, 1:] - xy[:, :-1]], dim=1)
            dxy = t.stack([c * dxy_w[..., 0] - s * dxy_w[..., 1],
                           s * dxy_w[..., 0] + c * dxy_w[..., 1]], dim=-1)
            dz = t.cat([z[:, :1] * 0, z[:, 1:] - z[:, :-1]], dim=1)
            ddof = t.cat([dof[:, :1] * 0, dof[:, 1:] - dof[:, :-1]], dim=1)
            feats += [dxy, dz, ddof]
        feats = t.cat(feats, dim=-1)
        out = self.net(feats.transpose(1, 2)).transpose(1, 2)    # [B,T,3+n]
        out = out * self.out_mask
        cy, sy = t.cos(yaw), t.sin(yaw)
        dx = cy * out[..., 0] - sy * out[..., 1]
        dy = sy * out[..., 0] + cy * out[..., 1]
        return t.cat([dx[..., None], dy[..., None], out[..., 2:]], dim=-1)


_ARM_AFF = [(i, g, a, b * np.pi / 180) for i, (n, g, a, b) in enumerate(_PHI_TABLE)
            if g >= 0 and ("shoulder" in n or "elbow" in n)]


def _arm_rmse_deg(g1_qpos: t.Tensor, x2_qpos: t.Tensor) -> float:
    """Pair agreement through the calibration-exact arm affines, degrees."""
    errs = [x2_qpos[:, 7 + xi] - (a * g1_qpos[:, 7 + gi] + b)
            for xi, gi, a, b in _ARM_AFF]
    return float(t.stack(errs).pow(2).mean().sqrt()) * 180 / np.pi


_KNEE_ROWS = [(i, g) for i, (n, g, _, _) in enumerate(_PHI_TABLE) if "knee" in n]


def _knee_flip(g1_qpos: t.Tensor, x2_qpos: t.Tensor) -> bool:
    """IK branch-flip signature: strongly negative knee correlation."""
    for xi, gi in _KNEE_ROWS:
        gk, xk = g1_qpos[:, 7 + gi], x2_qpos[:, 7 + xi]
        if gk.std() > 0.05 and xk.std() > 0.05:
            gc, xc = gk - gk.mean(), xk - xk.mean()
            c = (gc * xc).mean() / (gc.std() * xc.std() + 1e-9)
            if c < -0.2:
                return True
    return False


class PairWindowData:
    """Preloads the pair cache to RAM; samples gated, band-weighted windows."""

    def __init__(self, cache: Path, gate_arm_rmse_deg: float, gate_jerk: float,
                 device: str):
        man = json.loads((cache / "manifest.json").read_text())
        self.clips, self.weights, self.val = [], [], []
        n_gated = 0
        for key, s in man["clips"].items():
            if s["jerk_p95"] > gate_jerk:
                n_gated += 1
                continue
            d = torch.load(cache / "clips" / f"{key.replace('/', '__')}.pt",
                           map_location="cpu", weights_only=True)
            if _arm_rmse_deg(d["g1_qpos"], d["x2_qpos"]) > gate_arm_rmse_deg \
                    or _knee_flip(d["g1_qpos"], d["x2_qpos"]):
                n_gated += 1
                continue
            pair = (d["g1_qpos"], d["x2_qpos"])
            if s["split"] == "val":
                self.val.append(pair)
            else:
                self.clips.append(pair)
                sp = s["mean_speed"]
                self.weights.append(2.0 if 0.05 <= sp < 0.6 else
                                    (0.5 if sp >= 0.6 else 1.0))
        self.weights = torch.tensor(self.weights, dtype=torch.float64)
        self.weights /= self.weights.sum()
        self.lengths = torch.tensor([len(g) for g, _ in self.clips])
        self.device = device
        print(f"cache: {len(self.clips)} train / {len(self.val)} val clips, "
              f"{n_gated} gated out")

    def sample(self, batch: int, gen: torch.Generator, seg: int = WIN):
        w = self.weights
        if seg > int(self.lengths.min()):
            w = w * (self.lengths >= seg)          # too-short clips excluded
            w = w / w.sum()
        idx = torch.multinomial(w, batch, replacement=True, generator=gen)
        g1s, x2s = [], []
        for i in idx.tolist():
            g1, x2 = self.clips[i]
            s = int(torch.randint(0, len(g1) - seg + 1, (1,), generator=gen))
            g1s.append(g1[s:s + seg]); x2s.append(x2[s:s + seg])
        return (torch.stack(g1s).to(self.device),
                torch.stack(x2s).to(self.device))

    def val_windows(self, stride: int = 32, cap: int = 20000, seg: int = WIN):
        g1s, x2s = [], []
        for g1, x2 in self.val:
            for s in range(0, len(g1) - seg + 1, stride):
                g1s.append(g1[s:s + seg]); x2s.append(x2[s:s + seg])
        g1s, x2s = torch.stack(g1s[:cap]), torch.stack(x2s[:cap])
        return g1s.to(self.device), x2s.to(self.device)


def _apply_dec(phi_dec: PhiDecode, head: ResidualHead | None,
               g1_win: t.Tensor) -> t.Tensor:
    anchor = g1_win[:, -1:, 0:2]
    x2 = phi_dec(g1_win, anchor)
    if head is not None:
        r = head(g1_win)
        x2 = x2.clone()
        x2[..., 0:3] = x2[..., 0:3] + r[..., 0:3]
        x2[..., 7:36] = x2[..., 7:36] + r[..., 3:]   # non-head dof
    return x2


def _apply_enc(phi_enc: PhiEncode, head: ResidualHead | None,
               x2_win: t.Tensor) -> t.Tensor:
    g1 = phi_enc(x2_win)
    if head is not None:
        r = head(x2_win)
        g1 = g1.clone()
        g1[..., 0:3] = g1[..., 0:3] + r[..., 0:3]
        g1[..., 7:36] = g1[..., 7:36] + r[..., 3:]
    return g1


def _group_loss(pred_dof, gt_dof, legs, waist, arms, beta: float,
                hinge_deg: float):
    """Wide-knee smooth-L1 + heavy-deviation hinge (operator directive:
    small deviations are style — dead-zone via smooth-L1 beta; heavy ones
    are errors — extra L1 beyond hinge_deg)."""
    err = pred_dof - gt_dof
    sl1 = nn.functional.smooth_l1_loss(pred_dof, gt_dof, beta=beta,
                                       reduction="none")
    hinge = (err.abs() - hinge_deg * np.pi / 180).clamp(min=0)
    per = sl1 + 2.0 * hinge
    return (3.0 * per[..., legs].mean() + 2.0 * per[..., waist].mean()
            + 0.5 * per[..., arms].mean())


def losses(phi_enc, phi_dec, f_enc, f_dec, g1_win, x2_win, args):
    x2_pred = _apply_dec(phi_dec, f_dec, g1_win)
    g1_pred = _apply_enc(phi_enc, f_enc, x2_win)
    l_smooth = x2_pred.new_zeros(())
    if args.smooth_w > 0 and x2_pred.shape[1] > 1:
        # residual temporal smoothness (belt-and-braces on top of the
        # stationary conv): the correction field must vary slowly
        r_dec = x2_pred - _apply_dec(phi_dec, None, g1_win)
        l_smooth = args.smooth_w * (r_dec[:, 1:] - r_dec[:, :-1]).abs().mean()
        if f_enc is not None:
            r_enc = g1_pred - _apply_enc(phi_enc, None, x2_win)
            l_smooth = l_smooth + args.smooth_w * (
                (r_enc[:, 1:] - r_enc[:, :-1]).abs().mean())
    l_dec = _group_loss(x2_pred[..., 7:36], x2_win[..., 7:36],
                        _X2_LEGS, _X2_WAIST, _X2_ARMS,
                        args.beta, args.hinge_deg)
    l_dec = l_dec + 5.0 * (x2_pred[..., 2] - x2_win[..., 2]).abs().mean() \
                  + 2.0 * (x2_pred[..., 0:2] - x2_win[..., 0:2]).abs().mean()
    if f_enc is None:
        return l_dec, x2_pred.new_zeros(()), l_smooth
    l_enc = _group_loss(g1_pred[..., 7:36], g1_win[..., 7:36],
                        _G1_LEGS, _G1_WAIST, _G1_ARMS,
                        args.beta, args.hinge_deg)
    l_enc = l_enc + 5.0 * (g1_pred[..., 2] - g1_win[..., 2]).abs().mean() \
                  + 2.0 * (g1_pred[..., 0:2] - g1_win[..., 0:2]).abs().mean()
    return l_dec, l_enc, l_smooth


@torch.no_grad()
def val_report(phi_enc, phi_dec, f_enc, f_dec, vg1, vx2, tag):
    def mae(pred, gt, cols):
        return float((pred[..., cols] - gt[..., cols]).abs().mean()) * 180 / np.pi

    rows = {}
    for name, head in (("phi_only", None), (tag, f_dec)):
        x2p = _apply_dec(phi_dec, head, vg1)
        rows[name] = {
            "legs_deg": mae(x2p[..., 7:36], vx2[..., 7:36], _X2_LEGS),
            "waist_deg": mae(x2p[..., 7:36], vx2[..., 7:36], _X2_WAIST),
            "arms_deg": mae(x2p[..., 7:36], vx2[..., 7:36], _X2_ARMS),
            "z_mm": float((x2p[..., 2] - vx2[..., 2]).abs().mean()) * 1000,
        }
    g1p = _apply_enc(phi_enc, f_enc, vx2)
    g1b = _apply_enc(phi_enc, None, vx2)
    rows[tag]["enc_legs_deg"] = mae(g1p[..., 7:36], vg1[..., 7:36], _G1_LEGS)
    rows["phi_only"]["enc_legs_deg"] = mae(g1b[..., 7:36], vg1[..., 7:36], _G1_LEGS)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    eval_root = Path(os.environ.get("X2_EVAL_ROOT", Path.home() / ".cache/sonic/x2_eval"))
    ap.add_argument("--cache", type=Path, default=eval_root / "frozen_core_s1/pair_cache")
    ap.add_argument("--out", type=Path, default=eval_root / "frozen_core_s1/runs")
    ap.add_argument("--stage", choices=["lp1", "s1a", "s1c", "s1d", "s1e"],
                    required=True)
    ap.add_argument("--seg", type=int, default=16,
                    help="s1c training segment length (frames)")
    ap.add_argument("--smooth-w", type=float, default=1.0,
                    help="residual temporal-smoothness weight (s1c)")
    ap.add_argument("--steps", type=int, default=0, help="0 = stage default")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.05, help="smooth-L1 knee, rad")
    ap.add_argument("--hinge-deg", type=float, default=15.0)
    ap.add_argument("--gate-arm-rmse-deg", type=float, default=8.0)
    ap.add_argument("--gate-jerk", type=float, default=2500.0)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    steps = args.steps or (5000 if args.stage == "lp1" else 60000)
    if args.stage in ("lp1", "s1a"):
        args.smooth_w = 0.0
    seg = args.seg if args.stage in ("s1c", "s1d", "s1e") else WIN
    run = args.out / f"{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "steps": steps}, indent=1))

    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(project="frozen_core_s1", name=run.name,
                        config={**{k: str(v) for k, v in vars(args).items()},
                                "steps": steps})
        print(f"wandb: {wb.url}", flush=True)

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    dev = args.device

    data = PairWindowData(args.cache, args.gate_arm_rmse_deg, args.gate_jerk, dev)
    vseg = 64 if args.stage == "s1e" else seg   # s1e: val at DEPLOY plan length
    vg1, vx2 = data.val_windows(seg=vseg)
    print(f"val windows: {len(vg1)} (seg={vseg})")

    phi_enc, phi_dec = PhiEncode().to(dev), PhiDecode().to(dev)
    if args.stage in ("s1d", "s1e"):
        # v4 s1d: DEC-ONLY (encoder analytic = S0-proven; no core-input
        # distribution shift), per-frame (k=1: no edge/slot effects),
        # position-only (no velocity noise amplification), legs+waist+z.
        # v5 s1e (operator direction 2026-08-13): same guardrails but
        # SLIDING MULTI-FRAME input (k=5 convs) + per-step random
        # segment length 24..64 frames (kplanner-trainer style) so the
        # deployed 64-frame plan length is in-distribution.
        mask = torch.zeros(3 + 29)
        mask[2] = 1.0            # root z
        mask[3:3 + 15] = 1.0     # legs (12) + waist (3); arms/xy stay 0
        f_enc = None
        f_dec = ConvResidualHead(29, 29, hidden=args.hidden // 2,
                                 k=(1 if args.stage == "s1d" else 5),
                                 use_vel=False, out_mask=mask).to(dev)
    elif args.stage == "s1c":
        f_enc = ConvResidualHead(31, 29, hidden=args.hidden // 2).to(dev)
        f_dec = ConvResidualHead(29, 29, hidden=args.hidden // 2).to(dev)
    else:
        f_enc = ResidualHead(31, 29, args.stage, args.hidden).to(dev)
        f_dec = ResidualHead(29, 29, args.stage, args.hidden).to(dev)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=dev, weights_only=True)
        if args.stage == "s1a" and "lp1" in str(args.init_from):
            print("note: s1a from lp1 — architectures differ, skipping load")
        else:
            if f_enc is not None and sd.get("f_enc") is not None:
                f_enc.load_state_dict(sd["f_enc"])
            f_dec.load_state_dict(sd["f_dec"])
    params = list(f_dec.parameters()) + (
        list(f_enc.parameters()) if f_enc is not None else [])
    n_params = sum(p.numel() for p in params)
    print(f"stage {args.stage}: {n_params/1e3:.0f}k head params"
          f"{' (dec-only)' if f_enc is None else ''}, "
          f"{steps} steps, batch {args.batch}")

    base = val_report(phi_enc, phi_dec, f_enc, f_dec, vg1, vx2, args.stage)
    print("baseline (phi_only):", json.dumps(base["phi_only"]))

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    log = open(run / "train.jsonl", "w")
    t0 = time.time()
    for step in range(1, steps + 1):
        if args.stage == "s1e":
            # kplanner-trainer style: fresh random token count 6..16
            # per step -> segment 24..64 frames (deploy plan = 64)
            seg = 4 * int(torch.randint(6, 17, (1,), generator=gen))
        g1_win, x2_win = data.sample(args.batch, gen, seg=seg)
        l_dec, l_enc, l_smooth = losses(phi_enc, phi_dec, f_enc, f_dec,
                                        g1_win, x2_win, args)
        loss = l_dec + l_enc + l_smooth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            rec = {"step": step, "loss": round(float(loss), 5),
                   "l_dec": round(float(l_dec), 5), "l_enc": round(float(l_enc), 5),
                   "l_smooth": round(float(l_smooth), 5),
                   "sps": round(step / (time.time() - t0), 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if wb is not None:
                wb.log(rec, step=step)
            if step % 2000 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
        if step % 10000 == 0:
            torch.save({"f_enc": f_enc.state_dict() if f_enc else None,
                        "f_dec": f_dec.state_dict()},
                       run / f"heads_{step:06d}.pt")

    rep = val_report(phi_enc, phi_dec, f_enc, f_dec, vg1, vx2, args.stage)
    print("\nVAL (per-joint-group MAE, val clips held out BY CLIP):")
    for name, row in rep.items():
        print(f"  {name:>9}: " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()))
    torch.save({"f_enc": f_enc.state_dict() if f_enc else None,
                "f_dec": f_dec.state_dict(),
                "stage": args.stage, "hidden": args.hidden},
               run / "heads_final.pt")
    (run / "val_report.json").write_text(json.dumps(rep, indent=1))
    if wb is not None:
        for name, row in rep.items():
            wb.log({f"val/{name}/{k}": v for k, v in row.items()})
        wb.finish()
    print(f"\nrun dir: {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
