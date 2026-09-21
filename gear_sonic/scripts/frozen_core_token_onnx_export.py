#!/usr/bin/env python3
"""Export the TOKEN-INPUT deploy graph for whole-body Pico teleop on the robot.

Robot-side half of the token-streaming architecture (first whole-body
bring-up, 2026-08-29): the LAPTOP runs the Pico manager + the release G1
smpl encoder + FSQ (exactly what `FrozenCoreSmplActor` does in sim up to its
`token =` line) and streams the 64-D token into the deploy command channel;
the ROBOT runs this graph. It keeps the deploy contract byte-compatible
(`onnx_actor.cpp`: one input ``[*, 1670]``, one output ``[*, 31]``) so the
C++ side is unchanged — the token rides in the first 64 dims of the 680-dim
command slot, the remaining 616 are padding the graph ignores:

    obs[1670] = [ token 64 | pad 616 | X2 proprio 990 ]
      -> phi-encode proprio (G1 action history reconstructed by inverting
         the X2 action map — same validated machinery as the fused graph)
      -> g1_dyn decoder (merged, X2-adapted) -> clamp(+-20)
      -> phi-decode action (head rows exactly 0)
    -> action[31]

This is a strict TAIL of `frozen_core_t2_onnx_export.FusedFrozenCoreX2Wrapper`
(tokenizer transform + G1 encoder + FSQ removed; token injected). The
parity gate drives `FrozenCoreSmplActor` (the sim-proven composite) and this
graph with ONE shared closed-loop stream: random smpl obs -> the actor's
own wrist-rewrite + smpl encoder + FSQ produce the token (via
`laptop_token`, which doubles as the reference implementation for the
PC2 sender) — after the 10-frame history refills the two must agree.

    .venv/bin/python gear_sonic/scripts/frozen_core_token_onnx_export.py \
        --checkpoint $X2_EVAL_ROOT/<run>/<run>_merged.pt \
        --output $X2_EVAL_ROOT/<run>/exported/x2_sonic_<run>_token.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from frozen_core_sonic_codec import (  # noqa: E402
    FrozenCoreSmplActor, G1_ACTION_CLIP, NFRAMES, NUM_G1, NUM_X2,
    X2_PROP_DIM, X2_TOK_DIM)
from frozen_core_t2_onnx_export import (  # noqa: E402
    FusedFrozenCoreX2Wrapper, _mlp_from_state_dict, _phi_sidecar_check,
    tolerant_torch_load)
from frozen_core_sonic_codec import SonicPhiCodec, harvest_g1_params, harvest_x2_params  # noqa: E402

TOKEN_DIM = 64                      # FSQ token width (enc_dims[-1])
X2_OBS_DIM = X2_TOK_DIM + X2_PROP_DIM   # 1670 — deploy contract, unchanged


class TokenFusedX2Wrapper(FusedFrozenCoreX2Wrapper):
    """Fused graph tail: token + X2 proprio -> X2 action.

    forward() lines are the parent's minus the tokenizer/encoder/FSQ block;
    the phi buffers all come from the parent __init__ (encoder is loaded but
    unused — kept so the parent ctor and sidecar check stay identical).

    G1-HISTORY INVERSION (differs from the parent, deliberately): under the
    smpl-path wrist range-ratio remap, enc is NOT the inverse of dec (the
    remap installs enc_a=1/a, enc_b=-b/a, which composes to a FORWARD map
    a*q+b through the (q - enc_b)/enc_a encode formula — parity showed the
    parent's enc-based inversion shrinking wrist history by a^2 and
    poisoning every decoder output, settled 3.6-5.4 rad). The composite
    actor keeps TRUE G1 history; the only exact stateless reconstruction is
    inverting the DEC map row-by-row: q_g1[j] = (q_x2[i] - dec_b[i]) /
    dec_a[i] for the unique x2 row i with dec_src[i] == j."""

    def _build_dec_inverse(self, codec) -> None:
        import numpy as _np
        dec_src = _np.asarray(codec.phi["dec_src"])
        inv_src = _np.zeros(NUM_G1, _np.int64)
        inv_a = _np.zeros(NUM_G1, _np.float32)
        inv_b = _np.zeros(NUM_G1, _np.float32)
        for j in range(NUM_G1):
            rows = _np.nonzero(dec_src == j)[0]
            assert len(rows) == 1, f"g1 dof {j} maps to x2 rows {rows}"
            i = int(rows[0])
            inv_src[j] = i
            inv_a[j] = codec.phi["dec_a"][i]
            inv_b[j] = codec.phi["dec_b"][i]
        self.register_buffer("inv_src", torch.as_tensor(inv_src))
        self.register_buffer("inv_a", torch.as_tensor(inv_a))
        self.register_buffer("inv_b", torch.as_tensor(inv_b))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # noqa: D102
        B = obs.shape[0]
        token = obs[:, :TOKEN_DIM]                 # laptop-computed, post-FSQ
        prop_x2 = obs[:, X2_TOK_DIM:]              # pad 64:680 ignored

        n = NFRAMES * NUM_X2
        angvel = prop_x2[:, :30]
        jp_rel = prop_x2[:, 30:30 + n].view(B, NFRAMES, NUM_X2)
        jv_hist = prop_x2[:, 30 + n:30 + 2 * n].view(B, NFRAMES, NUM_X2)
        act_x2_hist = prop_x2[:, 30 + 2 * n:30 + 3 * n].view(B, NFRAMES, NUM_X2)
        grav = prop_x2[:, 30 + 3 * n:]

        jp_g1_rel = self._q_x2_to_g1(jp_rel + self.d_x2) - self.d_g1
        jv_g1_hist = self._v_x2_to_g1(jv_hist)
        # dec-inverse history reconstruction (see class docstring)
        q_tgt_x2 = self.d_x2 + act_x2_hist * self.s_x2
        q_g1_hist = (q_tgt_x2.index_select(-1, self.inv_src) - self.inv_b) / self.inv_a
        act_g1_hist = (q_g1_hist - self.d_g1) / self.s_g1

        prop_g1 = torch.cat(
            [angvel,
             jp_g1_rel.reshape(B, NFRAMES * NUM_G1),
             jv_g1_hist.reshape(B, NFRAMES * NUM_G1),
             act_g1_hist.reshape(B, NFRAMES * NUM_G1),
             grav], dim=-1)

        act_g1 = self.decoder(torch.cat([token, prop_g1], dim=-1))
        act_g1 = act_g1.clamp(-G1_ACTION_CLIP, G1_ACTION_CLIP)

        q_tgt_g1 = self.d_g1 + act_g1 * self.s_g1
        mapped = (self.dec_a * q_tgt_g1.index_select(-1, self.dec_src_clamped)
                  + self.dec_b)
        return self.dec_matched * (mapped - self.d_x2) / self.s_x2


def laptop_token(actor: FrozenCoreSmplActor, prop_x2: np.ndarray,
                 smpl_obs: np.ndarray) -> np.ndarray:
    """The LAPTOP side of the split, byte-identical to the composite actor's
    pre-decoder block (wrist rewrite -> smpl encoder -> FSQ). This function
    is the reference the PC2 token sender must reproduce."""
    obs = np.asarray(smpl_obs, np.float32).reshape(-1).copy()
    jp_rel_new = prop_x2[30 + 9 * NUM_X2: 30 + 10 * NUM_X2]
    q_g1 = actor.codec.q_x2_to_g1(jp_rel_new + actor.codec.d_x2)
    wrist_g1 = (q_g1[23:29]).astype(np.float32)
    for f in range(10):
        obs[f * 84 + 78: f * 84 + 84] = wrist_g1
    with torch.no_grad():
        tok = actor.fsq(actor.smpl_encoder(
            torch.from_numpy(obs)[None]))
    return tok.numpy().ravel().astype(np.float32)


def _sequential_parity(actor: FrozenCoreSmplActor, fused: TokenFusedX2Wrapper,
                       *, steps: int, seed: int):
    rng = np.random.default_rng(seed)
    actor._g1_action_hist = np.zeros((NFRAMES, NUM_G1), np.float32)
    act_hist_x2 = np.zeros((NFRAMES, NUM_X2), np.float32)
    diffs = np.zeros(steps)
    stream: list[np.ndarray] = []
    n = NFRAMES * NUM_X2
    for t in range(steps):
        angvel = rng.normal(0, 0.3, 30).astype(np.float32)
        jp_rel = rng.normal(0, 0.3, n).astype(np.float32)
        jv = rng.normal(0, 0.3, n).astype(np.float32)
        grav = rng.normal(0, 0.3, 30).astype(np.float32)
        smpl_obs = rng.normal(0, 0.3, 840).astype(np.float32)
        prop = np.concatenate([angvel, jp_rel, jv, act_hist_x2.reshape(-1), grav])
        assert prop.shape[0] == X2_PROP_DIM

        a_ref = actor(prop, smpl_obs).numpy().ravel()
        token = laptop_token(actor, prop, smpl_obs)
        pad = np.zeros(X2_TOK_DIM - TOKEN_DIM, np.float32)
        obs = np.concatenate([token, pad, prop])[None]
        stream.append(obs)
        with torch.no_grad():
            a_fused = fused(torch.from_numpy(obs)).numpy().ravel()
        diffs[t] = float(np.abs(a_ref - a_fused).max())

        act_hist_x2 = np.roll(act_hist_x2, -1, axis=0)
        act_hist_x2[-1] = a_ref
    return diffs, stream


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True,
                    help="MERGED frozen-core ckpt (frozen_core_t2_export.py output)")
    ap.add_argument("--release", default=None,
                    help="release G1-core ckpt for the smpl encoder "
                         "(default: the HF v1.1 last.pt the sim composite uses)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--phi-sidecar", default="")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-action-diff", type=float, default=1e-4)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--remap-waist", action="store_true",
                    help="bake the waist range-map (S1 lineage 2026-08-31+); "
                         "FORBIDDEN for earlier checkpoints")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).expanduser()
    out = Path(args.output).expanduser()
    sidecar = Path(args.phi_sidecar).expanduser() if args.phi_sidecar else \
        Path(str(ckpt) + ".phi.json")

    sd = tolerant_torch_load(str(ckpt))["policy_state_dict"]
    encoder, enc_dims = _mlp_from_state_dict(sd, "actor_module.encoders.g1.module.")
    decoder, dec_dims = _mlp_from_state_dict(sd, "actor_module.decoders.g1_dyn.module.")
    if enc_dims[-1] != TOKEN_DIM or dec_dims[-1] != NUM_G1:
        raise SystemExit(f"unexpected dims: enc {enc_dims}, dec {dec_dims}")

    codec = SonicPhiCodec(harvest_g1_params(), harvest_x2_params())
    if sidecar.exists():
        _phi_sidecar_check(codec, sidecar)
    else:
        print("[tok-onnx] WARNING: no phi sidecar — baked tables unchecked.")
    # The smpl COMPOSITE path applies the wrist range-ratio remap to its
    # codec (FrozenCoreSmplActor ctor). The deploy graph must bake the SAME
    # tables or wrist rows diverge by radians (first export attempt: settled
    # parity 3.6 rad, all wrist rows). Sidecar check above runs on the CLEAN
    # tables (merge-time state); the remap is applied after, mirroring the
    # actor's own construction order.
    from frozen_core_sonic_codec import remap_wrist_ranges
    remap_wrist_ranges(codec)
    if args.remap_waist:
        from frozen_core_sonic_codec import remap_waist_ranges
        remap_waist_ranges(codec)
        print("[tok-onnx] waist range-map APPLIED to baked tables")

    fused = TokenFusedX2Wrapper(encoder, decoder, codec)
    fused._build_dec_inverse(codec)
    fused = fused.eval()
    actor = FrozenCoreSmplActor(merged_ckpt=str(ckpt), release_ckpt=args.release,
                             device="cpu", remap_waist=args.remap_waist)

    diffs, stream = _sequential_parity(actor, fused, steps=args.steps,
                                       seed=args.seed)
    settled = diffs[NFRAMES:]
    print(f"[tok-onnx] parity vs FrozenCoreSmplActor: transient max|d| = "
          f"{diffs[:NFRAMES].max():.3e}; settled max|d| = {settled.max():.3e}")
    if settled.max() > args.max_action_diff and not args.force:
        raise SystemExit(f"[tok-onnx] FAILED parity gate: {settled.max():.3e}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    example = torch.from_numpy(stream[0].astype(np.float32))
    with torch.no_grad():
        torch.onnx.export(
            fused, example, tmp,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=args.opset, do_constant_folding=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    batch = np.concatenate(stream, axis=0).astype(np.float32)
    with torch.no_grad():
        ref = fused(torch.from_numpy(batch)).numpy()
    got = sess.run(["action"], {"obs": batch})[0]
    ort_max = float(np.abs(got - ref).max())
    head_max = float(np.abs(got[:, codec.phi["dec_src"] < 0]).max())
    print(f"[tok-onnx] ONNX vs torch fused: max|d| = {ort_max:.3e}; "
          f"head rows max|action| = {head_max:.3e}")
    if (ort_max > 1e-5 or head_max > 0.0) and not args.force:
        raise SystemExit("[tok-onnx] FAILED ONNX parity / head-zero gate.")

    import onnx
    m = onnx.load(tmp)
    for k, v in {
        "graph_kind": "any2any_token_deploy",
        "token_slot": f"obs[0:{TOKEN_DIM}] (post-FSQ smpl-encoder token; "
                      f"obs[{TOKEN_DIM}:{X2_TOK_DIM}] padding, ignored)",
        "laptop_side": "wrist-rewrite + release smpl encoder + FSQ "
                       "(frozen_core_token_onnx_export.laptop_token)",
        # v1.1 release cores anchor the smpl-obs root6d to robot YAW only
        # (smpl_root_ori_heading_multi_future). The obs builder MUST match
        # or the token commands a tilted pose (rehearsal fall #5,
        # 2026-08-29). pc2_pico_token_service reads this key and refuses
        # to start without it.
        "ori_mode": "heading",
        "checkpoint": str(ckpt),
    }.items():
        e = m.metadata_props.add(); e.key, e.value = k, str(v)
    onnx.save(m, tmp)
    Path(tmp).rename(out)
    print(f"[tok-onnx] PROMOTED {out}")


if __name__ == "__main__":
    main()
