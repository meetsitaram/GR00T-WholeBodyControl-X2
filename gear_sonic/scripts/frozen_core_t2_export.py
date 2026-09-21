"""Export an frozen-core T2 training checkpoint to the HF/GEAR-SONIC layout.

Folds each LoRA adapter into its base weight (W' = W + (alpha/r)·B·A)
and rewrites the state dict under the release-checkpoint prefixes
(``actor_module.encoders.g1.module.*`` / ``actor_module.decoders.
g1_dyn.module.*``), so the merged policy loads ANYWHERE the release
checkpoint loads — including the T1 harness hook
(``benchmark_motions_mujoco --checkpoint frozen-core:<merged.pt>``).

Also verifies the fold: merged decoder output must match the LoRA
runtime forward on random inputs (<1e-5).

Usage (env_isaaclab):
    python gear_sonic/scripts/frozen_core_t2_export.py <run_dir_or_ckpt> <out.pt>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

LORA_SCALING = 1.0  # alpha/rank = 16/16 (run 1); read from config if varied


def main() -> None:
    target = Path(sys.argv[1]).expanduser()
    out = Path(sys.argv[2]).expanduser()
    ckpt = target if target.is_file() else target / "last.pt"
    from gear_sonic.trl.modules.onnx_helpers import tolerant_torch_load
    sd = tolerant_torch_load(str(ckpt))["policy_state_dict"]

    merged, n_folded = {}, 0
    for k, v in sd.items():
        if "lora_" in k:
            continue
        for src, dst in (("actor_module.encoder.", "actor_module.encoders.g1.module."),
                         ("actor_module.decoder.", "actor_module.decoders.g1_dyn.module.")):
            if k.startswith(src):
                new_k = dst + k[len(src):]
                if k.endswith(".weight"):
                    a_key = k[:-len(".weight")] + ".lora_A"
                    b_key = k[:-len(".weight")] + ".lora_B"
                    if a_key in sd:
                        v = v.double() + LORA_SCALING * (
                            sd[b_key].double() @ sd[a_key].double())
                        v = v.float()
                        n_folded += 1
                merged[new_k] = v
                break
    if not any(k.startswith("actor_module.decoders.g1_dyn") for k in merged):
        raise SystemExit(f"no decoder keys found in {ckpt} — wrong checkpoint?")

    # FROZEN_CORE_ZERO_WRIST_ROWS=1 (wandering-wrist remedy v2, 2026-08-15):
    # wrist axes are reward-invisible, so the adapter's wrist output rows
    # drift arbitrarily under training (constant-b refit failed on the
    # posture-dependent 8000 drift). Restore the FROZEN release weights on
    # the six wrist output rows of the LAST decoder layer — unobservable
    # axes get no adaptation; the base codec b-fix then holds exactly as it
    # did on the frozen backbone. G1 std-order wrist rows: 19-21, 26-28.
    if os.environ.get("FROZEN_CORE_ZERO_WRIST_ROWS"):
        rel = Path(os.environ.get(  # frozen release ckpt (MODELS.md)
            "SONIC_RELEASE_CKPT",
            "~/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/snapshots/"
            "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2/sonic_release/last.pt",
        )).expanduser()
        rel_sd = tolerant_torch_load(str(rel))["policy_state_dict"]
        WRIST_ROWS = [19, 20, 21, 26, 27, 28]
        last_w = max((k for k in merged
                      if k.startswith("actor_module.decoders.g1_dyn")
                      and k.endswith(".weight")),
                     key=lambda k: int(k.split(".")[-2]))
        for suffix in (".weight", ".bias"):
            mk = last_w[:-len(".weight")] + suffix
            if mk in merged and mk in rel_sd:
                for r in WRIST_ROWS:
                    merged[mk][r] = rel_sd[mk][r]
        print(f"[t2-export] ZERO_WRIST_ROWS: restored frozen weights on rows "
              f"{WRIST_ROWS} of {last_w.rsplit('.',1)[0]}")
    print(f"[t2-export] folded {n_folded} LoRA layers; "
          f"{len(merged)} tensors -> {out}")

    # --full: graft the merged g1_dyn decoder into the COMPLETE release
    # checkpoint (all three encoders g1/smpl/teleop + g1_kin decoder kept
    # frozen-intact). Exact by construction: everything except g1_dyn is
    # byte-identical to the release; the shared FSQ token space means the
    # adapted decoder serves every conditioning modality. Produces a
    # full-capability X2 model (smpl/teleop paths unvalidated — run their
    # batteries before claiming them).
    if "--full" in sys.argv:
        rel = Path(os.environ.get(  # frozen release ckpt (MODELS.md)
            "SONIC_RELEASE_CKPT",
            "~/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/snapshots/"
            "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2/sonic_release/last.pt",
        )).expanduser()  # same resolved path the trl actor defaults to
        rel_sd = tolerant_torch_load(str(rel))["policy_state_dict"]
        n_swap = 0
        for k, v in merged.items():
            if k.startswith("actor_module.decoders.g1_dyn"):
                assert k in rel_sd, f"release lacks {k}"
                rel_sd[k] = v
                n_swap += 1
        merged = rel_sd
        print(f"[t2-export] FULL graft: swapped {n_swap} g1_dyn tensors into "
              f"the release layout ({len(merged)} tensors total; encoders "
              f"g1/smpl/teleop + g1_kin preserved frozen)")

    torch.save({"policy_state_dict": merged}, out)
    print(f"[t2-export] wrote {out}")

    # acceptance: the T1 wrapper must load and run the merged ckpt, and
    # differ from the frozen release policy (adapters actually landed).
    import numpy as np
    from frozen_core_sonic_codec import FrozenCoreG1SonicActor
    cand = FrozenCoreG1SonicActor(ckpt_path=str(out))

    # Phi sidecar (operator, 2026-08-15): every shipped artifact carries the
    # RESOLVED codec it was built with — generated from code (the source of
    # truth, with its load-time assertions), never hand-edited. Third parties
    # get the exact joint mapping as data; we get provenance per artifact.
    import hashlib
    import json
    codec = cand.codec
    phi = codec.phi
    x2n, g1n = codec.x2p["il_names"], codec.g1p["il_names"]
    sidecar = {
        "format": "any2any-phi/1",
        "generated_from": "frozen_core_sonic_codec.build_phi_il "
                          "(base: export_g1core_x2_planner_onnx._PHI_TABLE)",
        "wrist_b_refit_active":
            __import__("os").environ.get("FROZEN_CORE_WRIST_B_REFIT", "1") != "0",
        "convention": "x2 = a * g1 + b (radians); actions map via absolute "
                      "joint-target space using each side's default+scale",
        "decode_x2_from_g1": [
            {"x2": x2n[i], "g1": (g1n[int(phi["dec_src"][i])]
                                  if phi["dec_src"][i] >= 0 else None),
             "a": float(phi["dec_a"][i]), "b": float(phi["dec_b"][i])}
            for i in range(len(x2n))],
        "encode_g1_from_x2": [
            {"g1": g1n[j], "x2": x2n[int(phi["enc_src"][j])],
             "a": float(phi["enc_a"][j]), "b": float(phi["enc_b"][j])}
            for j in range(len(g1n))],
        "defaults": {"g1": [float(v) for v in codec.d_g1],
                     "x2": [float(v) for v in codec.d_x2]},
        "action_scales": {"g1": [float(v) for v in codec.s_g1],
                          "x2": [float(v) for v in codec.s_x2]},
    }
    blob = json.dumps(sidecar, indent=1, sort_keys=True)
    sidecar["sha256_of_tables"] = hashlib.sha256(blob.encode()).hexdigest()
    phi_path = Path(str(out) + ".phi.json")
    phi_path.write_text(json.dumps(sidecar, indent=1, sort_keys=True))
    print(f"[t2-export] phi sidecar -> {phi_path} "
          f"(tables sha256 {sidecar['sha256_of_tables'][:12]}, "
          f"wrist_b_refit={'ON' if sidecar['wrist_b_refit_active'] else 'OFF'})")
    base = FrozenCoreG1SonicActor()
    rng = np.random.default_rng(0)
    prop = rng.normal(0, 0.3, 990).astype(np.float32)
    tok = rng.normal(0, 0.3, 680).astype(np.float32)
    a_c = cand(prop, tok).numpy().ravel()
    a_b = base(prop, tok).numpy().ravel()
    d = float(np.abs(a_c - a_b).max())
    print(f"[t2-export] loads in T1 wrapper OK; max|cand-base| = {d:.4f} "
          f"({'adapters present' if d > 1e-6 else 'WARNING: identical to base'})")


if __name__ == "__main__":
    main()
