"""Checkpoint / MLP / FSQ helpers shared by the ONNX export and frozen-core paths.

Small, dependency-free helpers for rebuilding SONIC MLP stacks from a
``policy_state_dict`` and reproducing the FSQ quantizer used by the deploy
ONNX graphs. Imported by :mod:`gear_sonic.trl.modules.frozen_core_g1_actor`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

FSQ_LEVELS = 32


def tolerant_torch_load(path: str):
    """torch.load that stubs out unimportable classes (HF-trl etc.).

    Same trick as ``eval_x2_mujoco.load_actor_from_checkpoint``: we only need
    ``policy_state_dict`` (plain tensors); optimizer/args/env objects become
    inert stubs.
    """
    import pickle as _pickle
    import types as _types

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    class _StubUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    _tolerant = _types.ModuleType("_tolerant_pickle")
    for _a in dir(_pickle):
        setattr(_tolerant, _a, getattr(_pickle, _a))
    _tolerant.Unpickler = _StubUnpickler
    return torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=_tolerant)


def _mlp_from_state_dict(sd: dict, prefix: str) -> tuple[nn.Sequential, list[int]]:
    """Rebuild a SimpleMLP (Linear/SiLU stack) from ``{prefix}{i}.weight`` keys.

    Layer dims come from the tensor shapes, so nothing is hard-coded.
    """
    idxs = sorted(
        int(k[len(prefix):].split(".")[0])
        for k in sd if k.startswith(prefix) and k.endswith(".weight")
    )
    if not idxs:
        raise KeyError(f"no keys with prefix '{prefix}' in state dict")
    layers: list[nn.Module] = []
    dims = [sd[f"{prefix}{idxs[0]}.weight"].shape[1]]
    for pos, i in enumerate(idxs):
        w = sd[f"{prefix}{i}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(sd[f"{prefix}{i}.bias"])
        layers.append(lin)
        dims.append(w.shape[0])
        if pos < len(idxs) - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers), dims


class FsqQuantizer(nn.Module):
    """vector_quantize_pytorch.FSQ with uniform ``levels`` per dim (no params).

    Matches ``eval_x2_mujoco.fsq_quantize`` exactly:
        half_l  = (L-1) * (1+eps) / 2
        offset  = 0.5 if L even else 0
        shift   = atanh(offset / half_l)
        bounded = tanh(z + shift) * half_l - offset
        out     = round(bounded) / (L // 2)
    """

    def __init__(self, levels: int = FSQ_LEVELS):
        super().__init__()
        L = float(levels)
        eps = 1e-3
        self.half_l = (L - 1.0) * (1.0 + eps) / 2.0
        self.offset = 0.5 if int(levels) % 2 == 0 else 0.0
        self.shift = math.atanh(self.offset / self.half_l) if self.offset else 0.0
        self.div = int(levels) // 2

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(z + self.shift) * self.half_l - self.offset
        return torch.round(bounded) / self.div
