"""ONNX-policy shim that drops into the IsaacLab eval pipeline.

Used by ``eval_x2_isaacsim_onnx.py`` to swap an in-sim ``Actor.forward``
PyTorch call for an ``onnxruntime`` invocation of a deploy-export ONNX
(e.g. ``x2_sonic_16k.onnx``).

Why this exists
---------------
The deployed C++ runner sees actions diverge wildly when fed observations
built from MuJoCo or sensor data. We need to know whether the bug is in:

1. The ONNX export itself (i.e. the ONNX produces unstable actions even
   for the exact obs IsaacLab gives the .pt during training), OR
2. The C++ / MuJoCo observation pipeline that we're feeding the ONNX.

Running the ONNX inside the same IsaacLab env that produced the great
.pt training results is the cleanest A/B test:

* If the ONNX walks/tracks like the .pt, the export is fine and the bug
  is in our deploy obs pipeline.
* If the ONNX falls over in IsaacSim too, the export is broken and we
  need to re-export (or split into encoder/FSQ/decoder and run FSQ
  explicitly in C++).

Layout
------
The fused g1 ONNX expects a single 1670-D vector =
``[encoder_input_for_mlp_view(680) | actor_obs(990)]``.

We reuse the loaded ``actor_module``'s helpers
(``parse_tokenizer_obs``, ``encoder_input_features['g1']``,
``proprioception_features``) so the obs slicing/order can never drift
from training. See ``gear_sonic/scripts/dump_isaaclab_step0.py`` for the
canonical assembly we mirror here.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class OnnxPolicyShim:
    """Minimal callable that mimics ``actor_module(input_data)`` via ONNX.

    Parameters
    ----------
    actor_module:
        The PyTorch ``UniversalTokenModule`` instance from the loaded .pt.
        Used for its ``parse_tokenizer_obs`` helper and feature dicts.
    onnx_path:
        Path to the fused g1 ONNX (e.g. ``x2_sonic_16k.onnx``).
    encoder_name:
        Which encoder bank to assemble. Defaults to ``"g1"`` to match
        the deploy ONNX which is the g1 fused export.
    providers:
        ONNX runtime providers. Default CPU; change to ``["CUDAExecutionProvider"]``
        if you want GPU inference (rarely worth it for 1670-D actions).
    """

    def __init__(
        self,
        actor_module,
        onnx_path: str,
        encoder_name: str = "g1",
        providers: Optional[list[str]] = None,
    ):
        import onnxruntime as ort

        self.actor_module = actor_module
        self.onnx_path = str(onnx_path)
        self.encoder_name = encoder_name

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        in_meta = self.session.get_inputs()[0]
        out_meta = self.session.get_outputs()[0]
        self.input_name = in_meta.name
        self.output_name = out_meta.name
        self.expected_in_dim = int(in_meta.shape[-1])
        self.expected_out_dim = int(out_meta.shape[-1])

        self.input_features = list(actor_module.encoder_input_features[encoder_name])
        self.proprioception_features = list(actor_module.proprioception_features)

        # Cached info for assertions.
        self._first_call = True

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def __call__(self, input_data, **_unused_kwargs):
        """Mimic ``actor_module(input_data)``.

        Parameters
        ----------
        input_data:
            Dict-like with at least the proprioception key(s) and a
            ``"tokenizer"`` tensor. Shapes are ``(B, S, dim)`` where
            ``S`` is the policy's outer rollout length (typically 1).

        Returns
        -------
        action_mean:
            Tensor of shape ``(B, S, action_dim)`` matching what
            ``actor_module(input_data)`` would have returned.
        """
        # Build the 1670-D observation in the same order training does.
        tokenizer_obs = self.actor_module.parse_tokenizer_obs(input_data)
        proprio = torch.cat(
            [input_data[k] for k in self.proprioception_features], dim=-1
        )  # (B, S, prop_dim)

        obs_list = [tokenizer_obs[k] for k in self.input_features]
        encoder_input_full = torch.cat(obs_list, dim=-1)  # (B, S, T_in, F_per_t)

        # Flatten batch+seq for ONNX (which has a fixed [1, 1670] input).
        bs = encoder_input_full.shape[0] * encoder_input_full.shape[1]
        enc_flat = encoder_input_full.reshape(bs, -1)  # (B*S, 680)
        prop_flat = proprio.reshape(bs, -1)  # (B*S, prop_dim)
        obs_full = torch.cat([enc_flat, prop_flat], dim=-1)  # (B*S, 1670)

        if self._first_call:
            assert obs_full.shape[-1] == self.expected_in_dim, (
                f"ONNX expects {self.expected_in_dim}-D input but built "
                f"{obs_full.shape[-1]}-D (enc={enc_flat.shape[-1]}, prop={prop_flat.shape[-1]}). "
                f"Encoder features used: {self.input_features}; proprio features: "
                f"{self.proprioception_features}."
            )
            print(
                f"[OnnxPolicyShim] First forward OK. obs_full.shape={tuple(obs_full.shape)}, "
                f"ONNX expects [batch, {self.expected_in_dim}] -> [batch, {self.expected_out_dim}].",
                flush=True,
            )
            self._first_call = False

        obs_np = obs_full.detach().cpu().numpy().astype(np.float32)

        # ONNX has fixed batch=1; loop if needed (rare in eval, B*S usually = 1).
        outs = []
        for i in range(bs):
            out = self.session.run(
                [self.output_name], {self.input_name: obs_np[i : i + 1]}
            )[0]
            outs.append(out)
        action_np = np.concatenate(outs, axis=0)  # (B*S, action_dim)
        action_mean = torch.from_numpy(action_np).to(
            input_data[self.proprioception_features[0]].device
        )
        action_mean = action_mean.view(
            encoder_input_full.shape[0], encoder_input_full.shape[1], -1
        )
        return action_mean

    # ``Actor.forward`` checks ``has_aux_loss`` on the actor_module before
    # calling it. We don't return aux losses from ONNX.
    has_aux_loss = False


# ---------------------------------------------------------------------------
# Compare-mode helper
# ---------------------------------------------------------------------------
class CompareLogger:
    """Per-step CSV logger of ``|action_mean_pt - action_mean_onnx|``.

    Used when ``--compare`` is passed: at every step we run BOTH the
    original .pt actor_module and the ONNX shim on the same obs_dict,
    then write the per-joint action diff to disk.
    """

    def __init__(self, csv_path: str, action_dim: int):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._fp)
        header = ["step", "wall_t", "max_abs", "mean_abs", "l2_pt", "l2_onnx"]
        header += [f"diff_j{i:02d}" for i in range(action_dim)]
        self._writer.writerow(header)
        self._step = 0
        self._t0 = time.monotonic()
        print(f"[CompareLogger] Writing per-step diff to {self.csv_path}", flush=True)

    def log(self, action_pt: torch.Tensor, action_onnx: torch.Tensor):
        a_pt = action_pt.detach().reshape(-1).cpu().numpy().astype(np.float64)
        a_on = action_onnx.detach().reshape(-1).cpu().numpy().astype(np.float64)
        diff = a_pt - a_on
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        row = [
            self._step,
            f"{time.monotonic() - self._t0:.4f}",
            f"{max_abs:.6f}",
            f"{mean_abs:.6f}",
            f"{float(np.linalg.norm(a_pt)):.4f}",
            f"{float(np.linalg.norm(a_on)):.4f}",
        ] + [f"{d:+.6f}" for d in diff]
        self._writer.writerow(row)
        if self._step % 50 == 0:
            print(
                f"[Compare step {self._step}] max|pt-onnx|={max_abs:.4f}  "
                f"mean|pt-onnx|={mean_abs:.4f}  L2(pt)={np.linalg.norm(a_pt):.3f}  "
                f"L2(onnx)={np.linalg.norm(a_on):.3f}",
                flush=True,
            )
        self._step += 1

    def close(self):
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass
