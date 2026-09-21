#!/usr/bin/env python
"""Export ONE deploy graph with BOTH reference heads of a native 3-encoder
X2 checkpoint ("one SONIC on the robot, both pipelines hot").

    <name>_dual.onnx   obs[B, 2511] -> action[B, 31]
    obs = [ g1 pose-ref features 680 | smpl obs 840 | X2 proprio 990 | head 1 ]
    head = 0.0 -> pose-ref encoder (kplanner path), 1.0 -> smpl encoder
    (Pico path). Both encoders run every tick (they are the small MLPs);
    the FSQ + decoder run once on the selected token. Fractional head
    values are NOT a crossfade (FSQ is a hard quantiser) -- the deploy
    switches the head in one tick and smooths on its own last action.

Gates (both must pass or the export is refused):
  A  head=0: action == <name>_g1.onnx(obs[680+990])          (pose graph)
  B  head=1: action == <name>_g1_token.onnx([tokenizer(smpl)|pad|prop])
Usage:
  .venv/bin/python gear_sonic/scripts/native_dual_head_onnx_export.py \
      --checkpoint <ckpt.pt> --name x2_sonic_39000 --out-dir <dir with the split graphs>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gear_sonic/scripts"))
from eval_x2_mujoco import (  # noqa: E402
    ENCODER_INPUT_DIMS, UniversalTokenActor, fsq_quantize, load_actor_from_checkpoint)

G1_DIM = ENCODER_INPUT_DIMS["g1"]        # 680
SMPL_DIM = ENCODER_INPUT_DIMS["smpl"]    # 840
PROP_DIM = 990
OBS_DIM = G1_DIM + SMPL_DIM + PROP_DIM + 1   # 2511
TOKEN_DIM = 64
TOK_SLOT = 680


class NativeDualHeadGraph(torch.nn.Module):
    def __init__(self, g1_actor: UniversalTokenActor, smpl_actor: UniversalTokenActor):
        super().__init__()
        self.enc_g1 = g1_actor.encoder
        self.enc_smpl = smpl_actor.encoder
        self.decoder = smpl_actor.decoder      # identical weights in both loads (asserted below)
        self.levels = UniversalTokenActor.FSQ_LEVELS
        self.ntok = UniversalTokenActor.MAX_NUM_TOKENS
        self.tdim = UniversalTokenActor.TOKEN_DIM

    def _token(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.view(*latent.shape[:-1], self.ntok, self.tdim)
        q = fsq_quantize(latent, levels=self.levels)
        return q.view(*q.shape[:-2], -1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        g1 = obs[..., :G1_DIM]
        smpl = obs[..., G1_DIM:G1_DIM + SMPL_DIM]
        prop = obs[..., G1_DIM + SMPL_DIM:G1_DIM + SMPL_DIM + PROP_DIM]
        head = obs[..., -1:]                       # 0 -> g1, 1 -> smpl
        tok_g1 = self._token(self.enc_g1(g1))
        tok_smpl = self._token(self.enc_smpl(smpl))
        sel = (head > 0.5).to(tok_g1.dtype)
        token = sel * tok_smpl + (1.0 - sel) * tok_g1
        return self.decoder(torch.cat([token, prop], dim=-1))


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _utc_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parent, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--n-gate", type=int, default=64)
    args = ap.parse_args()
    out = args.out_dir / f"{args.name}_dual.onnx"
    g1_graph = args.out_dir / f"{args.name}_g1.onnx"
    tok_graph = args.out_dir / f"{args.name}_g1_token.onnx"
    tokenizer = args.out_dir / f"{args.name}_smpl_tokenizer.onnx"
    for p in (g1_graph, tok_graph, tokenizer):
        if not p.is_file():
            print(f"[dual] missing gate graph {p}"); return 2

    a_g1 = load_actor_from_checkpoint(args.checkpoint, "cpu", encoder="g1").eval()
    a_smpl = load_actor_from_checkpoint(args.checkpoint, "cpu", encoder="smpl").eval()
    for (k1, v1), (k2, v2) in zip(a_g1.decoder.state_dict().items(), a_smpl.decoder.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2), f"decoder weights differ between loads at {k1}"
    graph = NativeDualHeadGraph(a_g1, a_smpl).eval()

    ex = torch.zeros(1, OBS_DIM)
    with torch.no_grad():
        torch.onnx.export(graph, ex, str(out), opset_version=args.opset,
                          input_names=["obs"], output_names=["action"],
                          dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})
    import onnx
    m = onnx.load(str(out))
    ck = Path(args.checkpoint)
    ck_md5 = _md5(ck)
    meta = {"graph_kind": "native_dual_head", "ori_mode": "full",
            "obs_layout": f"g1_feat[0:{G1_DIM}] | smpl_obs[{G1_DIM}:{G1_DIM+SMPL_DIM}] | proprio[{G1_DIM+SMPL_DIM}:{G1_DIM+SMPL_DIM+PROP_DIM}] | head[{OBS_DIM-1}] (0=pose-ref/kplanner, 1=smpl/pico)",
            "lineage": "native-3enc", "codec_fingerprint": f"native-3enc:{ck_md5}",
            "source_checkpoint": ck.name, "source_md5": ck_md5,
            "export_utc": _utc_now(), "git_sha": _git_sha(), "exporter": Path(__file__).name,
            "pairs_with": f"{args.name}_g1.onnx, {args.name}_g1_token.onnx, {args.name}_smpl_tokenizer.onnx (same checkpoint)"}
    for k, v in meta.items():
        e = m.metadata_props.add(); e.key, e.value = k, v
    onnx.save(m, str(out))

    import onnxruntime as ort
    prov = ["CPUExecutionProvider"]
    s_dual = ort.InferenceSession(str(out), providers=prov)
    s_g1 = ort.InferenceSession(str(g1_graph), providers=prov)
    s_tok = ort.InferenceSession(str(tokenizer), providers=prov)
    s_tg = ort.InferenceSession(str(tok_graph), providers=prov)
    rng = np.random.default_rng(0)
    worst_a = worst_b = 0.0
    for _ in range(args.n_gate):
        g1 = rng.standard_normal((1, G1_DIM)).astype(np.float32) * 0.5
        smpl = rng.standard_normal((1, SMPL_DIM)).astype(np.float32) * 0.5
        prop = rng.standard_normal((1, PROP_DIM)).astype(np.float32) * 0.5
        # gate A: pose head
        obs = np.concatenate([g1, smpl, prop, np.zeros((1, 1), np.float32)], -1)
        a_dual = s_dual.run(["action"], {"obs": obs})[0]
        a_ref = s_g1.run([s_g1.get_outputs()[0].name], {s_g1.get_inputs()[0].name: np.concatenate([g1, prop], -1)})[0]
        worst_a = max(worst_a, float(np.abs(a_dual - a_ref).max()))
        # gate B: smpl head
        obs[0, -1] = 1.0
        a_dual = s_dual.run(["action"], {"obs": obs})[0]
        tok = s_tok.run(["motion_token"], {"smpl_obs": smpl})[0]
        pad = np.zeros((1, TOK_SLOT - TOKEN_DIM), np.float32)
        a_ref = s_tg.run(["action"], {"obs": np.concatenate([tok, pad, prop], -1)})[0]
        worst_b = max(worst_b, float(np.abs(a_dual - a_ref).max()))
    ok = worst_a < 1e-4 and worst_b < 1e-4
    print(f"[dual] {out.name}: obs {OBS_DIM} -> 31; gate A (pose head vs _g1.onnx) max|diff|={worst_a:.2e}; "
          f"gate B (smpl head vs tokenizer+_g1_token) max|diff|={worst_b:.2e} -> {'PASS' if ok else 'FAIL'}")
    print(f"[dual] md5 {_md5(out)}  metadata {meta['obs_layout']}")
    if not ok:
        out.unlink(missing_ok=True); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
