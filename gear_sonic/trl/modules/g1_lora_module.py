"""G1-native LoRA fine-tune module: the standard UniversalTokenModule with the
g1_dyn decoder LoRA-injected and EVERYTHING ELSE frozen.

The same-embodiment leg of the frozen-platform recipe (frozen-core ledger,
"STRETCH GOAL"): no Phi codec — the robot IS G1 — so this is literally the
release architecture with 245.7k trainable parameters riding on it.

Freeze-and-inject runs at __init__ tail, i.e. BEFORE the trainer builds its
optimizer (which filters on requires_grad), and warm-start checkpoints must
already CONTAIN the lora_A/B keys (build with make_g1_lora_warmstart.py) so
the trainer's strict load succeeds — the same convention the frozen-core phase-2
warm starts used.
"""
from __future__ import annotations

from loguru import logger

from gear_sonic.trl.modules.lora import inject_lora, mark_only_lora_trainable
from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule


class G1LoraUniversalTokenModule(UniversalTokenModule):
    def __init__(self, *args, lora: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        lora = lora or {"rank": 16, "alpha": 16.0}
        replaced = inject_lora(self.decoders["g1_dyn"], lora.get("targets", ["*"]),
                               rank=int(lora.get("rank", 16)),
                               alpha=float(lora.get("alpha", 16.0)))
        trainable, total = mark_only_lora_trainable(self)
        logger.info(
            f"[g1-lora] injected LoRA into {len(replaced)} linear(s) of g1_dyn; "
            f"trainable {trainable/1e3:.1f}k / {total/1e6:.1f}M total "
            f"(encoders/FSQ/other decoders FROZEN)")
        assert trainable > 0, "no LoRA parameters trainable — injection failed"
