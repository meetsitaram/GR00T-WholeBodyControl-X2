#!/usr/bin/env python3
"""Export an frozen-core T2 merged SONIC checkpoint to the fused X2 deploy ONNX.

The merged checkpoint (``frozen_core_t2_export.py`` output) is a G1-core policy
(encoder 640 -> FSQ -> decoder 994 -> 29). The X2 deploy contract
(``onnx_actor.cpp``: exactly one input ``[*, 1670]``, one output ``[*, 31]``)
is what ``sim_onnx_planner.sh MODEL=<onnx>`` and the robot run. This exporter
bakes the whole frozen-core sandwich into one stateless graph:

    obs[1670] = [X2 tokenizer 680 | X2 proprio 990]
      -> phi-encode tokenizer (gather + affine, scrambled-layout aware)
      -> phi-encode proprio   (angvel/grav passthrough, jpos/jvel affine)
      -> G1 encoder -> FSQ -> G1 decoder -> clamp(+-20)
      -> phi-decode action (absolute-target space; head rows exactly 0)
    -> action[31]

The one impedance mismatch vs the torch wrapper
(``frozen_core_sonic_codec.FrozenCoreG1SonicActor``) is its RECURRENT G1 action
history: a deploy graph must be stateless. Resolution: the X2 proprio
already carries the last 10 X2 actions (offset 650, 10x31); the action map
is a bijection on mapped rows, so the graph reconstructs the G1 history by
inverting it per frame:

    q_tgt_x2 = d_x2 + act_x2 * s_x2
    q_tgt_g1 = (q_tgt_x2[enc_src] - enc_b) / enc_a
    act_g1   = (q_tgt_g1 - d_g1) / s_g1

Two documented, bounded divergences from the torch wrapper (validated by the
sequential parity gate below, not hand-waved): (1) reset transient — the
wrapper starts its history at G1 zeros, the graph's inverse of X2 zeros lands
on the default-pose residual; converges once the 10-frame window refills.
(2) the deploy feeds back the post-clip X2 action; identical unless the +-20
clip binds (it does not in normal operation).

Validation gates (all must pass or the .onnx is not promoted):
  1. phi tables byte-checked against the checkpoint's ``.phi.json`` sidecar
     (catches FROZEN_CORE_WRIST_B_REFIT / _EXTRA env drift at export time).
  2. Sequential parity: N closed-history steps driven through BOTH the torch
     wrapper and the fused torch module; gate max|delta| after the history
     window refills (steps >= 10).
  3. ONNXRuntime vs fused torch module on the same stream + head-rows-zero.

Usage (IsaacLab conda env):
  $ISAACLAB_PYTHON \\
      -m gear_sonic.scripts.frozen_core_t2_onnx_export \\
      --checkpoint $X2_EVAL_ROOT/<run>/<run>_merged.pt \\
      --output $X2_EVAL_ROOT/<run>/exported/x2_sonic_<run>_g1.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(REPO_ROOT / "motionbricks" / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from frozen_core_sonic_codec import (  # noqa: E402
    FrozenCoreG1SonicActor,
    G1_ACTION_CLIP,
    NFRAMES,
    NUM_G1,
    NUM_X2,
    SonicPhiCodec,
    X2_PROP_DIM,
    X2_TOK_DIM,
    harvest_g1_params,
    harvest_x2_params,
)
from gear_sonic.trl.modules.onnx_helpers import (  # noqa: E402
    FsqQuantizer,
    _mlp_from_state_dict,
    tolerant_torch_load,
)

X2_OBS_DIM = X2_TOK_DIM + X2_PROP_DIM  # 1670, deploy fused-graph width


class FusedFrozenCoreX2Wrapper(nn.Module):
    """Stateless fused graph: X2 obs (B, 1670) -> X2 action (B, 31)."""

    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential,
                 codec: SonicPhiCodec):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fsq = FsqQuantizer(levels=32)

        p = codec.phi
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self.register_buffer("enc_src", torch.as_tensor(p["enc_src"], dtype=torch.long))
        self.register_buffer("enc_a", f32(p["enc_a"]))
        self.register_buffer("enc_b", f32(p["enc_b"]))
        # dec_src has -1 on head rows: clamp for the gather, mask the result.
        dec_src = np.asarray(p["dec_src"])
        self.register_buffer("dec_src_clamped",
                             torch.as_tensor(np.maximum(dec_src, 0), dtype=torch.long))
        self.register_buffer("dec_matched", f32(dec_src >= 0))
        self.register_buffer("dec_a", f32(p["dec_a"]))
        self.register_buffer("dec_b", f32(p["dec_b"]))
        self.register_buffer("d_g1", f32(codec.d_g1))
        self.register_buffer("s_g1", f32(codec.s_g1))
        self.register_buffer("d_x2", f32(codec.d_x2))
        self.register_buffer("s_x2", f32(codec.s_x2))
        # History inversion mirrors TRAINING (_act_x2_to_g1): DEC tables,
        # scatter into dec_gsrc. Pre-remap enc==dec so this equals the old
        # enc-based path bit-for-bit; post-remap they differ (the remap's
        # enc rows are NOT dec's inverse — a convention bug the remap-trained
        # policies absorbed) and training's dec-based inversion is the truth.
        # Scatter as a constant matmul so ONNX export stays gather/matmul-only.
        dec_rows = codec.dec_rows
        self.register_buffer("hist_rows", torch.as_tensor(dec_rows, dtype=torch.long))
        self.register_buffer("hist_dec_a", f32(np.asarray(p["dec_a"])[dec_rows]))
        self.register_buffer("hist_dec_b", f32(np.asarray(p["dec_b"])[dec_rows]))
        self.register_buffer("hist_d_g1", f32(np.asarray(codec.d_g1)[codec.dec_gsrc]))
        self.register_buffer("hist_s_g1", f32(np.asarray(codec.s_g1)[codec.dec_gsrc]))
        scat = np.zeros((len(dec_rows), NUM_G1), np.float32)
        scat[np.arange(len(dec_rows)), codec.dec_gsrc] = 1.0
        self.register_buffer("hist_scatter", torch.as_tensor(scat))

    # -- phi helpers (torch mirrors of SonicPhiCodec, batched) --------------
    def _q_x2_to_g1(self, q_x2: torch.Tensor) -> torch.Tensor:
        return (q_x2.index_select(-1, self.enc_src) - self.enc_b) / self.enc_a

    def _v_x2_to_g1(self, v_x2: torch.Tensor) -> torch.Tensor:
        return v_x2.index_select(-1, self.enc_src) / self.enc_a

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        tok_x2 = obs[:, :X2_TOK_DIM]
        prop_x2 = obs[:, X2_TOK_DIM:]

        # ---- tokenizer: X2 (B,680) -> G1 (B,640), scrambled layout -------
        blk = tok_x2.view(B, NFRAMES, 2 * NUM_X2 + 6)
        cmd = blk[:, :, :2 * NUM_X2].reshape(B, 2 * NFRAMES * NUM_X2)
        jp = cmd[:, :NFRAMES * NUM_X2].view(B, NFRAMES, NUM_X2)
        jv = cmd[:, NFRAMES * NUM_X2:].view(B, NFRAMES, NUM_X2)
        ori = blk[:, :, 2 * NUM_X2:]
        jp_g1 = self._q_x2_to_g1(jp)
        jv_g1 = self._v_x2_to_g1(jv)
        cmd_g1 = torch.cat(
            [jp_g1.reshape(B, NFRAMES * NUM_G1), jv_g1.reshape(B, NFRAMES * NUM_G1)],
            dim=-1,
        ).view(B, NFRAMES, 2 * NUM_G1)
        tok_g1 = torch.cat([cmd_g1, ori], dim=-1).reshape(B, NFRAMES * (2 * NUM_G1 + 6))

        # ---- proprio: X2 (B,990) -> G1 (B,930) ----------------------------
        n = NFRAMES * NUM_X2
        angvel = prop_x2[:, :30]
        jp_rel = prop_x2[:, 30:30 + n].view(B, NFRAMES, NUM_X2)
        jv_hist = prop_x2[:, 30 + n:30 + 2 * n].view(B, NFRAMES, NUM_X2)
        act_x2_hist = prop_x2[:, 30 + 2 * n:30 + 3 * n].view(B, NFRAMES, NUM_X2)
        grav = prop_x2[:, 30 + 3 * n:]

        jp_g1_rel = self._q_x2_to_g1(jp_rel + self.d_x2) - self.d_g1
        jv_g1_hist = self._v_x2_to_g1(jv_hist)
        # G1 action history reconstructed by inverting the action map —
        # DEC-table inversion, exactly as training's _act_x2_to_g1 (see
        # __init__ note; enc-based inversion diverges on remap-trained models).
        q_tgt_x2 = self.d_x2 + act_x2_hist * self.s_x2
        q_g1_from = (
            q_tgt_x2.index_select(-1, self.hist_rows) - self.hist_dec_b
        ) / self.hist_dec_a
        act_g1_hist = ((q_g1_from - self.hist_d_g1) / self.hist_s_g1) @ self.hist_scatter

        prop_g1 = torch.cat(
            [
                angvel,
                jp_g1_rel.reshape(B, NFRAMES * NUM_G1),
                jv_g1_hist.reshape(B, NFRAMES * NUM_G1),
                act_g1_hist.reshape(B, NFRAMES * NUM_G1),
                grav,
            ],
            dim=-1,
        )

        # ---- G1 core -------------------------------------------------------
        latent = self.encoder(tok_g1)
        token = self.fsq(latent)
        act_g1 = self.decoder(torch.cat([token, prop_g1], dim=-1))
        act_g1 = act_g1.clamp(-G1_ACTION_CLIP, G1_ACTION_CLIP)

        # ---- action: G1 (B,29) -> X2 (B,31); head rows exactly 0 ----------
        q_tgt_g1 = self.d_g1 + act_g1 * self.s_g1
        mapped = (
            self.dec_a * q_tgt_g1.index_select(-1, self.dec_src_clamped)
            + self.dec_b
        )
        return self.dec_matched * (mapped - self.d_x2) / self.s_x2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _phi_sidecar_check(codec: SonicPhiCodec, sidecar_path: Path) -> None:
    """The graph's baked tables must equal the checkpoint's shipped sidecar.

    Catches FROZEN_CORE_WRIST_B_REFIT / FROZEN_CORE_WRIST_B_EXTRA drift between the
    session that produced the merged .pt and this export."""
    side = json.loads(sidecar_path.read_text())
    p = codec.phi
    x2n = [n for n in codec.x2p["il_names"]]
    g1n = [n for n in codec.g1p["il_names"]]
    for i, row in enumerate(side["decode_x2_from_g1"]):
        assert row["x2"] == x2n[i], (row["x2"], x2n[i])
        src = p["dec_src"][i]
        want_g1 = g1n[int(src)] if src >= 0 else None
        assert row["g1"] == want_g1, (i, row["g1"], want_g1)
        if not (abs(row["a"] - p["dec_a"][i]) < 1e-12
                and abs(row["b"] - p["dec_b"][i]) < 1e-12):
            raise SystemExit(
                f"phi sidecar mismatch on decode row {i} ({x2n[i]}): sidecar "
                f"(a={row['a']}, b={row['b']}) vs codec (a={p['dec_a'][i]}, "
                f"b={p['dec_b'][i]}). Set FROZEN_CORE_WRIST_B_REFIT/_EXTRA to the "
                f"values used when the merged .pt was produced."
            )
    for j, row in enumerate(side["encode_g1_from_x2"]):
        assert row["g1"] == g1n[j]
        assert row["x2"] == x2n[int(p["enc_src"][j])]
        if not (abs(row["a"] - p["enc_a"][j]) < 1e-12
                and abs(row["b"] - p["enc_b"][j]) < 1e-12):
            raise SystemExit(f"phi sidecar mismatch on encode row {j} ({g1n[j]})")
    for side_key, arr in (("g1", codec.d_g1), ("x2", codec.d_x2)):
        np.testing.assert_allclose(side["defaults"][side_key], arr, atol=1e-12)
    for side_key, arr in (("g1", codec.s_g1), ("x2", codec.s_x2)):
        np.testing.assert_allclose(side["action_scales"][side_key], arr, atol=1e-12)
    print(f"[t2-onnx] phi sidecar check OK ({sidecar_path.name})")


def _sequential_parity(
    actor: FrozenCoreG1SonicActor,
    fused: FusedFrozenCoreX2Wrapper,
    *,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Drive both models with one shared closed-history stream.

    The X2 last-action block is fed back from the TORCH ACTOR's outputs on
    both sides (that's what the deploy/benchmark loop does), so the fused
    graph's inverse-mapped history must converge to the wrapper's internal
    G1 history once the 10-frame window has refilled.

    Returns (per_step_maxdiff, obs_stream) so the ONNX check can replay the
    identical stream.
    """
    rng = np.random.default_rng(seed)
    actor.reset()
    act_hist_x2 = np.zeros((NFRAMES, NUM_X2), np.float32)
    diffs = np.zeros(steps)
    stream: list[np.ndarray] = []
    for t in range(steps):
        angvel = rng.normal(0, 0.3, 30).astype(np.float32)
        jp_rel = rng.normal(0, 0.3, NFRAMES * NUM_X2).astype(np.float32)
        jv = rng.normal(0, 0.3, NFRAMES * NUM_X2).astype(np.float32)
        grav = rng.normal(0, 0.3, 30).astype(np.float32)
        tok = rng.normal(0, 0.3, X2_TOK_DIM).astype(np.float32)
        prop = np.concatenate([angvel, jp_rel, jv, act_hist_x2.reshape(-1), grav])
        assert prop.shape[0] == X2_PROP_DIM

        a_ref = actor(prop, tok).numpy().ravel()
        obs = np.concatenate([tok, prop])[None]
        stream.append(obs)
        with torch.no_grad():
            a_fused = fused(torch.from_numpy(obs)).numpy().ravel()
        diffs[t] = float(np.abs(a_ref - a_fused).max())

        act_hist_x2 = np.roll(act_hist_x2, -1, axis=0)
        act_hist_x2[-1] = a_ref  # closed loop on the reference actor
    return diffs, stream


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--phi-sidecar", default="",
                    help="default: <checkpoint>.phi.json")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-action-diff", type=float, default=1e-4,
                    help="parity gate (rad, action units) on steps >= 10")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--remap-waist", action="store_true",
                    help="bake the waist range-map (S1 fresh-vendor-waistmap "
                         "lineage, 2026-08-31 onward); FORBIDDEN for earlier "
                         "checkpoints — must match training exactly")
    ap.add_argument("--remap-wrist", action="store_true",
                    help="bake the wrist range-map into the graph; REQUIRED "
                         "for checkpoints trained with backbone."
                         "remap_wrist_ranges=true (S0 fresh-vendor lineage), "
                         "FORBIDDEN for earlier checkpoints — must match "
                         "training exactly (silent 3.07x wrist-pitch error "
                         "otherwise)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).expanduser()
    out = Path(args.output).expanduser()
    sidecar = Path(args.phi_sidecar).expanduser() if args.phi_sidecar else \
        Path(str(ckpt) + ".phi.json")

    sd = tolerant_torch_load(str(ckpt))["policy_state_dict"]
    encoder, enc_dims = _mlp_from_state_dict(sd, "actor_module.encoders.g1.module.")
    decoder, dec_dims = _mlp_from_state_dict(sd, "actor_module.decoders.g1_dyn.module.")
    print(f"[t2-onnx] encoder dims {enc_dims}; decoder dims {dec_dims}")
    if enc_dims[0] != 640 or dec_dims[-1] != NUM_G1:
        raise SystemExit(
            f"not a G1-core frozen-core checkpoint (enc in {enc_dims[0]}, dec out "
            f"{dec_dims[-1]}); for native X2 checkpoints use reexport_x2_g1_onnx."
        )

    codec = SonicPhiCodec(harvest_g1_params(), harvest_x2_params())
    # Sidecar check runs on the PRISTINE tables (its job is catching
    # FROZEN_CORE_WRIST_B_REFIT/_EXTRA env drift vs the merge session); the
    # remap — a deterministic transform of those same tables — applies after.
    if sidecar.exists():
        _phi_sidecar_check(codec, sidecar)
    if args.remap_wrist:
        from frozen_core_sonic_codec import remap_wrist_ranges
        remap_wrist_ranges(codec)
        print("[t2-onnx] wrist range-map APPLIED to baked tables "
              "(train==deploy for remap-trained checkpoints)")
    if args.remap_waist:
        from frozen_core_sonic_codec import remap_waist_ranges
        remap_waist_ranges(codec)
        print("[t2-onnx] waist range-map APPLIED to baked tables")
    else:
        print(f"[t2-onnx] WARNING: no phi sidecar at {sidecar} — baked tables "
              f"are unchecked against the merge session's env settings.")

    fused = FusedFrozenCoreX2Wrapper(encoder, decoder, codec).eval()
    actor = FrozenCoreG1SonicActor(ckpt_path=str(ckpt), device="cpu",
                                remap_wrist=args.remap_wrist,
                                remap_waist=args.remap_waist)

    diffs, stream = _sequential_parity(
        actor, fused, steps=args.steps, seed=args.seed)
    settled = diffs[NFRAMES:]
    print(f"[t2-onnx] parity vs torch wrapper: transient max|d| "
          f"(steps<10) = {diffs[:NFRAMES].max():.3e}; settled max|d| "
          f"(steps>={NFRAMES}) = {settled.max():.3e}")
    if settled.max() > args.max_action_diff and not args.force:
        raise SystemExit(
            f"[t2-onnx] FAILED parity gate: {settled.max():.3e} > "
            f"{args.max_action_diff:.1e} after history refill.")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    example = torch.from_numpy(stream[0].astype(np.float32))
    with torch.no_grad():
        torch.onnx.export(
            fused, example, tmp,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=False,  # folding kills FSQ round()
        )

    import onnxruntime as ort

    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    batch = np.concatenate(stream, axis=0).astype(np.float32)
    with torch.no_grad():
        ref = fused(torch.from_numpy(batch)).numpy()
    got = sess.run(["action"], {"obs": batch})[0]
    ort_max = float(np.abs(got - ref).max())
    head_max = float(np.abs(got[:, codec.phi["dec_src"] < 0]).max())
    print(f"[t2-onnx] ONNX vs torch fused: max|d| = {ort_max:.3e}; "
          f"head rows max|action| = {head_max:.3e}")
    if (ort_max > 1e-5 or head_max > 0.0) and not args.force:
        raise SystemExit("[t2-onnx] FAILED ONNX parity / head-zero gate.")

    # ---- STAMP THE OBS CONTRACT INTO THE GRAPH ---------------------------
    # WHY: the anchor-orientation term has TWO variants of identical shape --
    # motion_anchor_ori_b_mf_nonflat (full-orientation normalization, what the
    # C++ deploy tokenizer builds) and motion_anchor_ori_heading_mf_nonflat
    # (yaw-only, what every v1.1-lineage model is trained on). Feeding the
    # wrong one is SILENT: 60 values, right shape, wrong meaning. The two agree
    # while the robot is upright and diverge with its pitch/roll, so it passes
    # every gentle bring-up rung and fails once the robot tilts.
    #
    # Stamping it here makes the model SELF-DECLARE what it needs, so a deploy
    # can refuse instead of guessing. Read it back with onnx_obs_contract.py.
    # Resolve the contract from the config that shipped with the checkpoint.
    # Searched next to the merged ckpt, then one level up (the eval harness uses
    # the same convention). UNKNOWN is stamped rather than guessed -- a consumer
    # must be able to tell "not declared" from "declared as b".
    _ori = "UNKNOWN"
    for _c in (ckpt.parent / "config.yaml", ckpt.parent.parent / "config.yaml"):
        if _c.exists():
            import re as _re
            _txt = _c.read_text()
            _m = _re.search(r"ori_obs_name:\s*(\S+)", _txt)
            if _m:
                _ori = _m.group(1)
                break
            # Resolved training configs usually leave ori_obs_name at the
            # backbone kwarg default, which never serializes. Fall back to
            # the default IF the config's obs lists corroborate it.
            if "motion_anchor_ori_b_mf_nonflat" in _txt:
                _ori = "motion_anchor_ori_b_mf_nonflat"
                print(f"[t2-onnx] ori_obs_name from backbone default, "
                      f"corroborated by {_c.name} obs terms")
                break
    if _ori == "UNKNOWN":
        print("[t2-onnx] WARNING: no config.yaml beside the checkpoint -- the "
              "obs contract will be stamped UNKNOWN. A deploy cannot verify it.")
    try:
        import onnx
        m = onnx.load(str(tmp))
        meta = {
            "ori_obs_name": _ori,
            "obs_dim": str(X2_OBS_DIM),
            "action_dim": str(NUM_X2),
            "phi_sha256": sidecar_sha if "sidecar_sha" in dir() else "",
            "source_checkpoint": os.path.basename(str(ckpt)),
            "exporter": "frozen_core_t2_onnx_export",
            "remap_wrist": str(bool(args.remap_wrist)),
            "remap_waist": str(bool(args.remap_waist)),
        }
        for k, v in meta.items():
            e = m.metadata_props.add()
            e.key, e.value = k, v
        onnx.save(m, str(tmp))
        print(f"[t2-onnx] stamped obs contract: ori_obs_name={meta['ori_obs_name']}")
    except Exception as e:  # never block promotion on a metadata failure
        print(f"[t2-onnx] WARNING: could not stamp metadata ({type(e).__name__}: {e}); "
              "the .onnx will not self-declare its obs contract")

    os.replace(tmp, out)
    shutil.copy2(sidecar, Path(str(out) + ".phi.json")) if sidecar.exists() else None
    prov = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "phi_sidecar": sidecar.name if sidecar.exists() else None,
        "parity_settled_max": float(settled.max()),
        "parity_transient_max": float(diffs[:NFRAMES].max()),
        "onnx_vs_torch_max": ort_max,
        "enc_dims": enc_dims, "dec_dims": dec_dims,
        "obs_dim": X2_OBS_DIM, "action_dim": NUM_X2,
        "note": "stateless fused frozen-core graph; G1 action history "
                "reconstructed in-graph from the X2 last-action block",
    }
    Path(str(out) + ".provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"[t2-onnx] PROMOTED {out}")


if __name__ == "__main__":
    main()
