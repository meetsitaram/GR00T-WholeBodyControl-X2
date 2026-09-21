"""LoRA injection for the frozen-core T2 dynamics-adaptation stage.

Implements the FrozenCore (arXiv 2605.23733) recipe: freeze the pretrained
backbone; for each adapted linear projection W, learn a low-rank
correction W' = W + (alpha/rank) * B @ A with A ~ N(0, sigma), B = 0
(so the wrapped policy is EXACTLY the source policy at init). Per the
paper, adapters go into the proprioceptive input projection and the
actor/critic linear layers; the reference-motion encoder + FSQ stay
frozen (which also preserves the shared token space with the planner
stack).

Design constraints honored here:
  * ``LoRALinear`` keeps the wrapped layer's ``weight``/``bias`` param
    names (it IS an nn.Linear), so source checkpoints load unchanged
    (missing ``lora_A/lora_B`` keys are expected -> load with
    strict=False and assert the only missing keys are lora_*).
  * ``inject_lora`` replaces layers in-place by dotted-name pattern, so
    it composes with any module tree ``custom_instantiate`` builds.
  * ``mark_only_lora_trainable`` freezes everything else; optimizer
    construction can then simply take ``p for p in model.parameters()
    if p.requires_grad``.

Self-test:  python -m gear_sonic.trl.modules.lora   (or run the file)
"""

from __future__ import annotations

import fnmatch
import math

import torch
from torch import nn


class LoRALinear(nn.Linear):
    """nn.Linear with a parallel low-rank residual (frozen W, trainable AB)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rank: int = 16, alpha: float = 16.0, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError(f"rank {rank} invalid for {in_features}x{out_features}")
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    @classmethod
    def from_linear(cls, lin: nn.Linear, rank: int, alpha: float) -> "LoRALinear":
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                rank=rank, alpha=alpha, device=lin.weight.device,
                dtype=lin.weight.dtype)
        with torch.no_grad():
            m.weight.copy_(lin.weight)
            if lin.bias is not None:
                m.bias.copy_(lin.bias)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.weight, self.bias)
        return out + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        """W + scaling * B@A — for export (fold the adapter into W)."""
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)


def inject_lora(root: nn.Module, patterns: list[str], rank: int = 16,
                alpha: float = 16.0, verbose: bool = True) -> list[str]:
    """Replace every nn.Linear whose dotted name matches any fnmatch
    pattern (e.g. ``"actor_module.decoders.g1_dyn.*"``) with LoRALinear.

    Returns the list of replaced layer names. Raises if nothing matched
    (a silent no-op here would train zero parameters and "converge").
    """
    replaced = []
    for name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if (isinstance(child, nn.Linear)
                    and not isinstance(child, LoRALinear)
                    and any(fnmatch.fnmatch(full, p) for p in patterns)):
                setattr(module, child_name, LoRALinear.from_linear(child, rank, alpha))
                replaced.append(full)
    if not replaced:
        raise ValueError(f"inject_lora: no nn.Linear matched {patterns}")
    if verbose:
        n = sum(p.numel() for m in root.modules() if isinstance(m, LoRALinear)
                for p in (m.lora_A, m.lora_B))
        print(f"[lora] injected rank-{rank} adapters into {len(replaced)} layers "
              f"({n/1e3:.1f}k trainable params): {replaced[:4]}{'...' if len(replaced) > 4 else ''}")
    return replaced


def mark_only_lora_trainable(root: nn.Module) -> tuple[int, int]:
    """Freeze all params except lora_A/lora_B. Returns (trainable, total)."""
    for p in root.parameters():
        p.requires_grad_(False)
    for m in root.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    trainable = sum(p.numel() for p in root.parameters() if p.requires_grad)
    total = sum(p.numel() for p in root.parameters())
    return trainable, total


def lora_state_dict(root: nn.Module) -> dict:
    """Only the adapter params — the shippable per-embodiment artifact."""
    return {k: v for k, v in root.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def _self_test() -> int:
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, 29))
    x = torch.randn(32, 64)
    y0 = net(x)
    sd0 = {k: v.clone() for k, v in net.state_dict().items()}

    replaced = inject_lora(net, ["0", "2"], rank=8, alpha=8.0)
    assert replaced == ["0", "2"], replaced

    # identity at init (B=0)
    y1 = net(x)
    assert torch.allclose(y0, y1, atol=0), "LoRA must be exact identity at init"

    # source state dict still loads (strict=False, only lora_* missing)
    missing, unexpected = net.load_state_dict(sd0, strict=False)
    assert not unexpected and all("lora_" in k for k in missing), (missing, unexpected)

    # only adapters train
    tr, tot = mark_only_lora_trainable(net)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-2)
    w_before = net[0].weight.clone()
    for _ in range(5):
        opt.zero_grad()
        net(x).pow(2).mean().backward()
        opt.step()
    assert torch.equal(net[0].weight, w_before), "frozen W changed"
    assert not torch.allclose(net(x), y0), "adapters had no effect after training"

    # merged export == runtime forward
    merged = nn.functional.linear(x, net[0].merged_weight(), net[0].bias)
    assert torch.allclose(merged, net[0](x), atol=1e-6)

    ad = lora_state_dict(net)
    assert len(ad) == 4 and all("lora_" in k for k in ad)
    print(f"[lora] SELF-TEST PASS (trainable {tr/1e3:.1f}k / total {tot/1e3:.1f}k, "
          f"adapter file {len(ad)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
