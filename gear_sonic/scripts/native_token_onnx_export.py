#!/usr/bin/env python3
"""Split a NATIVE 3-encoder X2 checkpoint into the two deploy graphs the
Quest-split whole-body teleop needs (2026-09-02, stage-2 of the incumbent
vendor-plant adaptation):

    <name>_smpl_tokenizer.onnx   smpl_obs[B,840] -> motion_token[B,64]
                                 (smpl encoder MLP + FSQ; runs in
                                 pc2_pico_token_service.py on PC2 / sim)
    <name>_g1_token.onnx         obs[B,1670] -> action[B,31]
                                 obs = [token 64 | pad 616 | X2 proprio 990]
                                 (g1_dyn decoder; runs in the deploy binary
                                 with --whole-body-teleop)

Until now every *_token.onnx / *_smpl_tokenizer.onnx on disk came from the
FROZEN-G1 frozen-core lineage (`frozen_core_token_onnx_export.py`: release G1 smpl
encoder + phi codec). A native model's decoder was trained on ITS OWN
encoder's tokens, so feeding it v11release tokens tests the wrong policy.
This exporter cuts both halves from the same native checkpoint, and the
composition is gated against the IsaacLab-validated fused smpl graph
(`reexport_x2_g1_onnx --encoder-name smpl`, obs 1830) on a shared random
stream, plus the step-0 dump when given.

Byte-compatible with the existing contracts (checked against
x2_smpl_tokenizer_v11release.onnx / x2_sonic_armA_14900_token.onnx): same
input/output names, same dims, opset 17. The tokenizer carries
`ori_mode=full` metadata — the plain 3-encoder lineage convention the token
service refuses to guess (v1.1 release cores are `heading`).

    .venv/bin/python gear_sonic/scripts/native_token_onnx_export.py \\
        --checkpoint $X2_EVAL_ROOT/<run>/model_step_036000.pt \\
        --fused-smpl $X2_EVAL_ROOT/<run>/exported/x2_sonic_36000_smpl.onnx \\
        --dump /tmp/x2_step0_smpl.pt \\
        --name x2_sonic_36000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from eval_x2_mujoco import (  # noqa: E402
    ENCODER_INPUT_DIMS, UniversalTokenActor, fsq_quantize,
    load_actor_from_checkpoint)

SMPL_OBS_DIM = ENCODER_INPUT_DIMS["smpl"]          # 840
TOKEN_DIM = UniversalTokenActor.MAX_NUM_TOKENS * UniversalTokenActor.TOKEN_DIM  # 64
X2_TOK_SLOT = 680                                   # deploy command slot
X2_PROP_DIM = 990
X2_OBS_DIM = X2_TOK_SLOT + X2_PROP_DIM              # 1670 — deploy contract
ORI_MODE = "full"                                   # plain 3-encoder lineage


class NativeSmplTokenizer(torch.nn.Module):
    """smpl_obs[B,840] -> motion_token[B,64] (encoder MLP + FSQ)."""

    def __init__(self, actor: UniversalTokenActor):
        super().__init__()
        self.encoder = actor.encoder

    def forward(self, smpl_obs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(smpl_obs)
        latent = latent.view(-1, UniversalTokenActor.MAX_NUM_TOKENS,
                             UniversalTokenActor.TOKEN_DIM)
        q = fsq_quantize(latent, levels=UniversalTokenActor.FSQ_LEVELS)
        return q.reshape(-1, TOKEN_DIM)


class NativeTokenGraph(torch.nn.Module):
    """obs[B,1670] = [token 64 | pad 616 | proprio 990] -> action[B,31]."""

    def __init__(self, actor: UniversalTokenActor):
        super().__init__()
        self.decoder = actor.decoder

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        token = obs[:, :TOKEN_DIM]
        prop = obs[:, X2_TOK_SLOT:]
        return self.decoder(torch.cat([token, prop], dim=-1))


def _export(module: torch.nn.Module, example: torch.Tensor, path: str,
            in_name: str, out_name: str, opset: int, metadata: dict) -> None:
    import onnx
    with torch.no_grad():
        torch.onnx.export(
            module, example, path,
            input_names=[in_name], output_names=[out_name],
            dynamic_axes={in_name: {0: "batch"}, out_name: {0: "batch"}},
            opset_version=opset,
            do_constant_folding=False,   # keep FSQ rounding live, as reexport does
            verbose=False)
    m = onnx.load(path)
    for k, v in metadata.items():
        p = m.metadata_props.add()
        p.key, p.value = k, str(v)
    onnx.save(m, path)


def _sess(path: str):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="native 3-encoder .pt")
    ap.add_argument("--fused-smpl", required=True,
                    help="IsaacLab-validated fused smpl graph (obs 1830) = parity reference")
    ap.add_argument("--dump", default=None,
                    help="dump_isaaclab_step0 capture with ++encoder_name=smpl (extra GT gate)")
    ap.add_argument("--name", required=True, help="output stem, e.g. x2_sonic_36000")
    ap.add_argument("--out-dir", default=None, help="default: <checkpoint dir>/exported")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-action-diff", type=float, default=1e-4)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stamp-pose-graph", default=None,
                    help="also stamp the checkpoint's fused pose graph (<name>_g1.onnx) with the "
                         "same codec_fingerprint so start_x2_deploy_ritual.sh's era preflight "
                         "reports OK for a matched pair (metadata only; graph untouched)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (Path(args.checkpoint).parent / "exported"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / f"{args.name}_smpl_tokenizer.onnx"
    tg_path = out_dir / f"{args.name}_g1_token.onnx"
    tok_tmp, tg_tmp = str(tok_path) + ".tmp", str(tg_path) + ".tmp"

    actor = load_actor_from_checkpoint(args.checkpoint, "cpu", encoder="smpl")
    tokenizer = NativeSmplTokenizer(actor).eval()
    token_graph = NativeTokenGraph(actor).eval()
    ckpt_base = os.path.basename(args.checkpoint)
    print(f"[native-split] checkpoint {args.checkpoint}", flush=True)

    rng = np.random.default_rng(args.seed)
    ex_smpl = torch.from_numpy(rng.normal(0, 0.3, (1, SMPL_OBS_DIM)).astype(np.float32))
    ex_obs = torch.from_numpy(rng.normal(0, 0.3, (1, X2_OBS_DIM)).astype(np.float32))
    # graph_kind values are CONTRACT markers the consumers check literally:
    # the deploy binary (x2_deploy_onnx_ref.cpp ~L1612) refuses any token
    # graph without graph_kind=any2any_token_deploy ("token-input graph,
    # obs=[token 64|pad 616|proprio 990]") -- same contract here, so the same
    # marker; lineage is carried separately. The token service reads ori_mode.
    # codec_fingerprint: the ritual's era preflight (start_x2_deploy_ritual.sh
    # 2026-09-01) compares this stamp on the pose graph and the tokenizer and
    # REFUSES a mismatch. A native model has no phi tables; its "codec" is the
    # checkpoint itself, so the fingerprint is the checkpoint's md5.
    import hashlib
    with open(args.checkpoint, "rb") as fh:
        ckpt_md5 = hashlib.md5(fh.read()).hexdigest()
    fingerprint = f"native-3enc:{ckpt_md5}"
    import datetime as _dt, subprocess as _sp
    try:
        _git = _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                cwd=Path(__file__).resolve().parent, text=True).strip()
    except Exception:  # noqa: BLE001
        _git = "unknown"
    common = {"lineage": "native-3enc", "checkpoint": ckpt_base,
              "codec_fingerprint": fingerprint,
              "source_md5": fingerprint.split(":", 1)[-1],
              "export_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "git_sha": _git,
              "exporter": "native_token_onnx_export.py"}
    if args.stamp_pose_graph:
        import onnx
        pg = onnx.load(args.stamp_pose_graph, load_external_data=False)
        keep = [p for p in pg.metadata_props if p.key not in common]
        del pg.metadata_props[:]
        pg.metadata_props.extend(keep)
        for k, v in common.items():
            p = pg.metadata_props.add(); p.key, p.value = k, str(v)
        onnx.save(pg, args.stamp_pose_graph)
        print(f"[native-split] stamped pose graph {args.stamp_pose_graph} codec_fingerprint={fingerprint[:24]}…", flush=True)
    _export(tokenizer, ex_smpl, tok_tmp, "smpl_obs", "motion_token", args.opset,
            {**common, "graph_kind": "smpl_tokenizer", "ori_mode": ORI_MODE,
             "source": f"native smpl encoder + FSQ of {ckpt_base}",
             "pairs_with": tg_path.name})
    _export(token_graph, ex_obs, tg_tmp, "obs", "action", args.opset,
            {**common, "graph_kind": "any2any_token_deploy",
             "token_slot": "obs[0:64] (post-FSQ native smpl-encoder token; obs[64:680] padding, ignored)",
             "laptop_side": f"pc2_pico_token_service.py --tokenizer {tok_path.name} (ori_mode={ORI_MODE})",
             "pairs_with": tok_path.name})
    print(f"[native-split] wrote {tok_tmp}\n[native-split] wrote {tg_tmp}", flush=True)

    s_tok, s_tg, s_fused = _sess(tok_tmp), _sess(tg_tmp), _sess(args.fused_smpl)
    fin = s_fused.get_inputs()[0]
    assert list(fin.shape)[1] == SMPL_OBS_DIM + X2_PROP_DIM, \
        f"--fused-smpl must be the 1830-obs smpl graph, got {fin.shape}"
    pad = np.zeros((1, X2_TOK_SLOT - TOKEN_DIM), np.float32)

    def split_action(smpl_obs: np.ndarray, prop: np.ndarray) -> np.ndarray:
        tok = s_tok.run(["motion_token"], {"smpl_obs": smpl_obs})[0]
        obs = np.concatenate([tok, pad, prop], axis=-1).astype(np.float32)
        return s_tg.run(["action"], {"obs": obs})[0]

    worst = 0.0
    # --- gate A: step-0 dump ground truth (IsaacLab-captured) ---------------
    if args.dump:
        d = torch.load(args.dump, map_location="cpu", weights_only=False)
        if d.get("encoder_name") != "smpl":
            print(f"[native-split] FAILED: dump encoder is {d.get('encoder_name')!r}, need 'smpl'")
            return 2
        smpl_gt = d["encoder_input_for_mlp_view"].numpy().astype(np.float32)   # (1,840)
        prop_gt = d["proprioception_input"].squeeze(1).numpy().astype(np.float32)
        tok_gt = d["fsq_token_smpl"].reshape(1, TOKEN_DIM).numpy().astype(np.float32)
        act_gt = d["decoder_action_mean"].squeeze(1).numpy().astype(np.float32)
        tok = s_tok.run(["motion_token"], {"smpl_obs": smpl_gt})[0]
        d_tok = float(np.abs(tok - tok_gt).max())
        act = s_tg.run(["action"], {"obs": np.concatenate([tok_gt, pad, prop_gt], -1)})[0]
        d_act = float(np.abs(act - act_gt).max())
        d_e2e = float(np.abs(split_action(smpl_gt, prop_gt) - act_gt).max())
        print(f"[native-split] gate A (dump GT): tokenizer max|tok-gt|={d_tok:.2e}  "
              f"token-graph max|act-gt|={d_act:.2e} rad  end-to-end={d_e2e:.2e} rad", flush=True)
        worst = max(worst, d_act, d_e2e)
        if d_tok > 1e-4:
            print("[native-split] FAILED: tokenizer does not reproduce the dump's FSQ token")
            worst = max(worst, 1.0)

    # --- gate B: shared random stream vs the fused smpl graph --------------
    diffs = np.zeros(args.steps)
    for t in range(args.steps):
        smpl_obs = rng.normal(0, 0.3, (1, SMPL_OBS_DIM)).astype(np.float32)
        prop = rng.normal(0, 0.3, (1, X2_PROP_DIM)).astype(np.float32)
        a_ref = s_fused.run([s_fused.get_outputs()[0].name],
                            {fin.name: np.concatenate([smpl_obs, prop], -1)})[0]
        diffs[t] = float(np.abs(split_action(smpl_obs, prop) - a_ref).max())
    print(f"[native-split] gate B (vs fused smpl ONNX, {args.steps} random obs): "
          f"max={diffs.max():.2e} rad  mean={diffs.mean():.2e}  "
          f"steps>{args.max_action_diff:g}: {int((diffs > args.max_action_diff).sum())}", flush=True)
    worst = max(worst, float(diffs.max()))

    ok = worst <= args.max_action_diff
    if ok or args.force:
        os.replace(tok_tmp, tok_path)
        os.replace(tg_tmp, tg_path)
        print(f"[native-split] {'OK' if ok else '--force'} -- promoted:\n"
              f"    {tok_path}\n    {tg_path}", flush=True)
        return 0
    print(f"[native-split] FAILED: worst {worst:.2e} > {args.max_action_diff:g} rad; "
          f"left .tmp files, nothing promoted", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
