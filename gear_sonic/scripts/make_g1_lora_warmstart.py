"""Build the G1-LoRA warm-start checkpoint from the frozen release.

Takes g1_release_last.pt and adds zero-effect LoRA keys (lora_A kaiming /
lora_B zeros, matching LoRALinear's own init) for every Linear in
decoders.g1_dyn, so the trainer's strict load into
G1LoraUniversalTokenModule succeeds and step 0 is bit-identical to the
release policy. Optimizer/scheduler stripped (the LoRA resume-bug recipe).

Usage: python make_g1_lora_warmstart.py IN_release.pt OUT_warmstart.pt [rank]
"""
import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gear_sonic.trl.modules.onnx_helpers import tolerant_torch_load  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rank = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    ck = tolerant_torch_load(src)
    sd = ck["policy_state_dict"]
    added = 0
    for k in [k for k in sd if k.endswith(".weight")
              and ".decoders.g1_dyn." in k]:
        base = k[: -len(".weight")]
        w = sd[k]
        if w.ndim != 2:
            continue
        out_f, in_f = w.shape
        a = torch.empty(rank, in_f, dtype=w.dtype)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        sd[base + ".lora_A"] = a
        sd[base + ".lora_B"] = torch.zeros(out_f, rank, dtype=w.dtype)
        added += 1
    # Minimal clean dict: the release file carries unpicklable stubs from
    # tolerant_torch_load, and ppo_trainer.load_checkpoint(resume=False)
    # reads ONLY policy_state_dict (strict=False) + value_state_dict.
    # Critic warm start kept when present — same embodiment, same obs.
    out = {"policy_state_dict": sd}
    if isinstance(ck.get("value_state_dict"), dict):
        out["value_state_dict"] = ck["value_state_dict"]
    torch.save(out, dst)
    print(f"[g1-lora-warmstart] added lora keys to {added} linears -> {dst}")
    assert added > 0


if __name__ == "__main__":
    main()
