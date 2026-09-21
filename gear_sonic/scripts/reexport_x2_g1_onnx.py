#!/usr/bin/env python3
"""Re-export the fused X2 G1 deploy ONNX from a PyTorch checkpoint.

This script exists because the original ONNX shipped at
``gear_sonic_deploy/models/x2_sonic_16k.onnx`` was found to diverge from the
original PyTorch checkpoint by up to 9.8 rad per joint when fed identical
observations (see ``onnx_vs_pt.csv`` produced by ``eval_x2_isaacsim_onnx.py``
in compare mode).

The export wrapper here mirrors **exactly** the forward path captured by
``dump_isaaclab_step0.py``:

    [encoder_input_for_mlp_view (680) | proprioception_input (990)]
        |
        v   reshape -> (B, 10, 68); each row = cat(command(62), ori(6))
    g1 encoder  -> (B, max_num_tokens=2, token_dim=32)
        |
        v   FSQ quantizer
    quantized   -> (B, 2, 32)
        |
        v   flatten + cat with proprioception(990)
    g1_dyn decoder -> (B, 31) action mean

This is the same code path used during training & evaluation, so
``onnx(input) == decoder(... encoder(input) ...)`` to within ONNX-runtime
numerical precision.

The script:

  1. Hijacks ``Actor.forward`` so the very first call after model load:
       a. Captures the live ``UniversalTokenModule`` ("actor_module").
       b. Wraps it in :class:`FusedG1Wrapper`.
       c. Exports to ONNX (opset 17, no FSQ-killing constant folding).
       d. Validates the freshly exported ONNX against the saved IsaacLab
          step-0 dump, refusing to overwrite the deploy artefact unless the
          max-abs delta is below ``--max-action-diff`` (default 1e-3 rad).
       e. ``os._exit(0)`` so we don't run the full eval loop.

  2. Hands off to ``gear_sonic.eval_agent_trl`` via ``runpy.run_module`` so
     Hydra resolves config_path / cwd correctly (this is the same trick used
     by ``eval_x2_isaacsim_onnx.py`` and ``dump_isaaclab_step0.py``).

Typical usage (from the repo root, inside the IsaacLab conda env):

  conda activate env_isaaclab
  python -m gear_sonic.scripts.reexport_x2_g1_onnx \\
      --run-dir $CKPT_ROOT/<run-dir> \\
      --output gear_sonic_deploy/models/x2_sonic_16k.onnx \\
      --dump /tmp/x2_step0_isaaclab.pt

The wrapped checkpoint defaults to ``last.pt`` inside ``--run-dir``; pass
``--checkpoint`` to override.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Repo root on sys.path so ``gear_sonic`` imports work when run as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# CLI -- pre-parsed before Hydra so we can strip our flags before runpy
# ---------------------------------------------------------------------------
def _split_argv(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Pull our wrapper-specific flags out of ``sys.argv``.

    Anything we don't consume here is forwarded verbatim to
    ``gear_sonic.eval_agent_trl`` so Hydra can interpret it.
    """
    parser = argparse.ArgumentParser(
        description="Re-export X2 G1 deploy ONNX (fused encoder+FSQ+decoder)",
        add_help=False,
    )
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Training run directory (auto-injects checkpoint=)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit .pt checkpoint path (overrides --run-dir/last.pt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Where to write the new ONNX file")
    parser.add_argument("--dump", type=str, default="/tmp/x2_step0_isaaclab.pt",
                        help="IsaacLab step-0 dump used for validation")
    parser.add_argument("--encoder-name", type=str, default="g1")
    parser.add_argument("--decoder-name", type=str, default="g1_dyn")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-action-diff", type=float, default=1e-3,
                        help="Refuse to write the ONNX if max|onnx-pt| > this (radians)")
    parser.add_argument("--force", action="store_true",
                        help="Write the ONNX even if validation fails")
    parser.add_argument("--help", "-h", action="help")

    known = []
    rest = []
    consumed_with_value = {
        "--run-dir", "--checkpoint", "--output", "--dump",
        "--encoder-name", "--decoder-name", "--opset", "--max-action-diff",
    }
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok in consumed_with_value:
            known.extend([tok, argv[i + 1]])
            i += 2
            continue
        if any(tok.startswith(k + "=") for k in consumed_with_value):
            known.append(tok)
            i += 1
            continue
        if tok in {"--force", "--help", "-h"}:
            known.append(tok)
            i += 1
            continue
        rest.append(tok)
        i += 1

    args = parser.parse_args(known)
    return args, rest


def _ensure_checkpoint_override(rest_argv: list[str], args: argparse.Namespace) -> list[str]:
    """Make sure Hydra sees a ``checkpoint=...`` override if the user gave us
    only a ``--run-dir``.  Mirrors the behaviour of ``eval_x2_isaacsim_onnx``.
    """
    if any(a.startswith("checkpoint=") or a.startswith("+checkpoint=") for a in rest_argv):
        return rest_argv

    ckpt: str | None = None
    if args.checkpoint is not None:
        ckpt = args.checkpoint
    elif args.run_dir is not None:
        candidate = Path(args.run_dir) / "last.pt"
        if not candidate.exists():
            for alt in sorted(Path(args.run_dir).glob("model_step_*.pt")):
                candidate = alt
            if not candidate.exists():
                raise FileNotFoundError(
                    f"No checkpoint found in {args.run_dir} (looked for last.pt, model_step_*.pt)"
                )
        ckpt = str(candidate)

    if ckpt is None:
        return rest_argv

    print(f"[reexport_x2_g1_onnx] Injecting checkpoint={ckpt}", flush=True)
    return rest_argv + [f"checkpoint={ckpt}"]


# ---------------------------------------------------------------------------
# The fused export wrapper
# ---------------------------------------------------------------------------
class FusedG1Wrapper(nn.Module):
    """Fused encoder + FSQ + decoder module that matches IsaacLab forward.

    The wrapped layout is the **same** layout used by
    ``dump_isaaclab_step0.py`` and consumed by the C++ deploy binary:

        input shape: (B, 1670)
            [: 680 ]  encoder_input_for_mlp_view
                       per-frame interleaved over 10 future frames:
                       [cmd_f0(62) | ori_f0(6) | cmd_f1(62) | ori_f1(6) | ...]
            [680:1670] proprioception_input (= actor_obs)

        output shape: (B, 31) -- joint position targets in IsaacLab order.
    """

    def __init__(self, actor_module, encoder_name: str, decoder_name: str):
        super().__init__()
        self.module = actor_module
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        if encoder_name not in actor_module.encoders:
            raise KeyError(f"encoder '{encoder_name}' not in {list(actor_module.encoders)}")
        if decoder_name not in actor_module.decoders:
            raise KeyError(f"decoder '{decoder_name}' not in {list(actor_module.decoders)}")

        self.input_features: list[str] = list(
            actor_module.encoder_input_features[encoder_name]
        )
        self.proprioception_features: list[str] = list(actor_module.proprioception_features)

        self.t_in: int | None = None
        f_per_frame: list[int] = []
        for k in self.input_features:
            dims = actor_module.tokenizer_obs_dims[k]
            if len(dims) != 2:
                raise ValueError(
                    f"encoder '{encoder_name}' input '{k}' has dims {dims}, expected 2-D"
                )
            t_in_k = int(dims[0])
            if self.t_in is None:
                self.t_in = t_in_k
            elif self.t_in != t_in_k:
                raise ValueError(
                    f"inconsistent t_in for encoder '{encoder_name}': "
                    f"{self.input_features} -> {[actor_module.tokenizer_obs_dims[k] for k in self.input_features]}"
                )
            f_per_frame.append(int(dims[1]))

        self.f_per_frame: list[int] = f_per_frame
        self.f_total: int = int(sum(f_per_frame))
        self.enc_dim: int = int(self.t_in * self.f_total)
        self.token_dim: int = int(actor_module.token_dim)
        self.max_num_tokens: int = int(actor_module.max_num_tokens)
        self.token_total_dim: int = self.max_num_tokens * self.token_dim

        prop_dim = actor_module.obs_dim_dict.get("actor_obs")
        if prop_dim is None:
            prop_dim = 990
        self.prop_dim: int = int(prop_dim)
        self.input_dim: int = self.enc_dim + self.prop_dim

        # Cache encoder & decoder for cleaner forward graph
        self.encoder = actor_module.encoders[encoder_name]
        self.quantizer = actor_module.quantizer
        self.decoder = actor_module.decoders[decoder_name]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Slice the flat 1670-D input into encoder / proprioception parts.
        enc_flat = obs[:, : self.enc_dim]
        prop_flat = obs[:, self.enc_dim :]

        # Reshape encoder slice to (B, 10, 68) -- per-frame interleaved
        # [cmd_f0(62) | ori_f0(6) | ... | cmd_f9(62) | ori_f9(6)].
        encoder_input_flat = enc_flat.view(-1, self.t_in, self.f_total)

        # Encode -> (B, max_num_tokens, token_dim)
        latent = self.encoder(encoder_input_flat)

        if self.quantizer is not None:
            quantized, _ = self.quantizer(latent)
        else:
            quantized = latent

        # Decoder expects (B, S=1, token_total_dim + proprioception_dim).
        token_flattened = quantized.reshape(-1, 1, self.token_total_dim)
        prop_input = prop_flat.view(-1, 1, self.prop_dim)
        decoder_input = torch.cat([token_flattened, prop_input], dim=-1)

        decoder_out = self.decoder(decoder_input)
        # decoder_out: (B, 1, 31) -> (B, 31)
        return decoder_out.squeeze(1)


# ---------------------------------------------------------------------------
# Validation against the IsaacLab step-0 dump
# ---------------------------------------------------------------------------
def _validate_onnx(onnx_path: str, dump_path: str, max_diff: float,
                   encoder_name: str) -> tuple[float, float]:
    """Run the freshly-exported ONNX on the dump's GT input and compare.

    Returns ``(max_abs_diff, mean_abs_diff)`` between the ONNX action mean
    and the dump's ``decoder_action_mean``.  Raises ``RuntimeError`` if the
    max diff exceeds ``max_diff``.
    """
    import numpy as np
    import onnxruntime as ort

    if not os.path.exists(dump_path):
        raise FileNotFoundError(
            f"validation dump not found at {dump_path}; "
            "run gear_sonic/scripts/dump_isaaclab_step0.py first"
        )

    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    dump_encoder = dump.get("encoder_name", "g1")
    if dump_encoder != encoder_name:
        raise RuntimeError(
            f"dump {dump_path} was captured with encoder '{dump_encoder}' but "
            f"this export uses '--encoder-name {encoder_name}'. "
            f"Re-run dump_isaaclab_step0.py with ++encoder_name={encoder_name}."
        )
    enc_view = dump["encoder_input_for_mlp_view"]   # (B, enc_dim)
    prop = dump["proprioception_input"]              # (B, 1, 990)
    gt_action = dump["decoder_action_mean"].squeeze(1).numpy()  # (B, 31)

    if prop.dim() == 3:
        prop_2d = prop.squeeze(1)
    else:
        prop_2d = prop
    fused = torch.cat([enc_view, prop_2d], dim=-1).numpy().astype(np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    onnx_action = sess.run([out_name], {in_name: fused})[0]
    if onnx_action.ndim == 3:
        onnx_action = onnx_action.squeeze(1)

    diff = np.abs(onnx_action - gt_action)
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    print(f"[reexport_x2_g1_onnx] Validation vs IsaacLab GT:", flush=True)
    print(f"    max|onnx - pt| = {max_d:.6f} rad", flush=True)
    print(f"    mean|onnx - pt| = {mean_d:.6f} rad", flush=True)
    print(f"    onnx[0,:6]      = {onnx_action[0, :6]}", flush=True)
    print(f"    pt  [0,:6]      = {gt_action[0, :6]}", flush=True)

    if max_d > max_diff:
        raise RuntimeError(
            f"Validation failed: max|onnx-pt|={max_d:.6f} > {max_diff} rad. "
            "Pass --force to write anyway."
        )
    return max_d, mean_d


# ---------------------------------------------------------------------------
# Hijack Actor.forward so the first call performs the export
# ---------------------------------------------------------------------------
def _patch_actor_for_export(args: argparse.Namespace) -> None:
    from gear_sonic.trl.modules import actor_critic_modules as ac

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"

    original_forward = ac.Actor.forward
    state = {"done": False}

    def patched_forward(self, obs_dict, is_training=False, **kwargs):
        if state["done"]:
            return original_forward(self, obs_dict, is_training=is_training, **kwargs)

        state["done"] = True
        print(f"[reexport_x2_g1_onnx] First Actor.forward -- exporting ONNX...",
              flush=True)

        actor_module = getattr(self, "actor_module", None)
        if actor_module is None:
            raise RuntimeError(
                "Actor instance has no .actor_module attribute -- this script "
                "currently only supports the SONIC universal-token actor."
            )

        # Build wrapper on CPU for export portability
        import copy
        am_cpu = copy.deepcopy(actor_module).to("cpu").eval()
        wrapper = FusedG1Wrapper(am_cpu, args.encoder_name, args.decoder_name).eval()

        print(f"[reexport_x2_g1_onnx] FusedG1Wrapper:"
              f" enc_dim={wrapper.enc_dim} prop_dim={wrapper.prop_dim}"
              f" max_num_tokens={wrapper.max_num_tokens}"
              f" token_dim={wrapper.token_dim}", flush=True)

        # Use the dump's actual input as the trace example so any layout
        # disagreement explodes loudly at trace time, not silently later.
        if os.path.exists(args.dump):
            dump = torch.load(args.dump, map_location="cpu", weights_only=False)
            example = torch.cat(
                [dump["encoder_input_for_mlp_view"],
                 dump["proprioception_input"].squeeze(1)],
                dim=-1,
            ).float()
        else:
            print(f"[reexport_x2_g1_onnx] WARNING: dump {args.dump} missing; "
                  f"falling back to randn example input.", flush=True)
            example = torch.randn(1, wrapper.input_dim, dtype=torch.float32)

        with torch.no_grad():
            ref_action = wrapper(example).cpu().numpy()
        print(f"[reexport_x2_g1_onnx] PyTorch wrapper action[0,:6] = {ref_action[0, :6]}",
              flush=True)

        # Sanity check: the freshly-loaded checkpoint should reproduce the
        # dump's decoder_action_mean almost bit-perfectly when fed the dump's
        # input.  If not, we're exporting from a DIFFERENT checkpoint than the
        # one used to make the dump -- in which case ONNX-vs-dump comparison
        # later would be meaningless.
        if os.path.exists(args.dump):
            import numpy as np
            dump_check = torch.load(args.dump, map_location="cpu", weights_only=False)
            gt = dump_check["decoder_action_mean"].squeeze(1).cpu().numpy()
            pt_diff = float(np.max(np.abs(ref_action - gt)))
            print(f"[reexport_x2_g1_onnx] PyTorch wrapper vs dump GT: "
                  f"max|pt_wrapper - dump.decoder_action_mean| = {pt_diff:.6f}",
                  flush=True)
            if pt_diff > 1e-3:
                msg = (
                    f"PyTorch wrapper diverges from dump GT by {pt_diff:.4f} rad. "
                    f"The checkpoint loaded here is NOT the same one used to "
                    f"produce {args.dump}. Re-run dump_isaaclab_step0 with the "
                    f"same checkpoint, or pass --checkpoint to match."
                )
                if args.force:
                    print(f"[reexport_x2_g1_onnx] --force: {msg}", flush=True)
                else:
                    print(f"[reexport_x2_g1_onnx] FAILED: {msg}", flush=True)
                    os._exit(2)

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example,
                tmp_path,
                input_names=["obs"],
                output_names=["action"],
                dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                opset_version=args.opset,
                do_constant_folding=False,  # don't fold-away FSQ rounding
                verbose=False,
            )

        print(f"[reexport_x2_g1_onnx] Wrote temporary ONNX to {tmp_path}", flush=True)

        try:
            max_d, mean_d = _validate_onnx(tmp_path, args.dump, args.max_action_diff,
                                           args.encoder_name)
            os.replace(tmp_path, output_path)
            print(f"[reexport_x2_g1_onnx] OK -- promoted to {output_path}", flush=True)
            print(f"    max|onnx-pt|={max_d:.6f}  mean|onnx-pt|={mean_d:.6f}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            if args.force:
                os.replace(tmp_path, output_path)
                print(f"[reexport_x2_g1_onnx] --force: wrote {output_path} despite: {exc}",
                      flush=True)
            else:
                if os.path.exists(tmp_path):
                    print(f"[reexport_x2_g1_onnx] Leaving failed export at {tmp_path} "
                          f"for inspection.", flush=True)
                print(f"[reexport_x2_g1_onnx] FAILED: {exc}", flush=True)
                os._exit(1)

        os._exit(0)

    ac.Actor.forward = patched_forward
    print("[reexport_x2_g1_onnx] Installed Actor.forward export hook.", flush=True)


# ---------------------------------------------------------------------------
# Main entry: parse args, install hook, hand off to eval_agent_trl
# ---------------------------------------------------------------------------
def main() -> None:
    args, rest_argv = _split_argv(sys.argv)

    if args.run_dir is not None and not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"--run-dir {args.run_dir} does not exist")

    rest_argv = _ensure_checkpoint_override(rest_argv, args)

    # Force eval to short-circuit ASAP -- we only need a single Actor.forward
    # call so the export hook fires.  These overrides match what the dump
    # script uses.
    forced = []
    if not any(a.startswith("++run_eval_loop=") for a in rest_argv):
        forced.append("++run_eval_loop=true")
    if not any(a.startswith("++max_render_steps=") for a in rest_argv):
        forced.append("++max_render_steps=1")
    if not any(a.startswith("+headless=") or a.startswith("headless=") for a in rest_argv):
        forced.append("+headless=True")
    rest_argv = rest_argv + forced

    _patch_actor_for_export(args)

    sys.argv = [sys.argv[0]] + rest_argv
    print(f"[reexport_x2_g1_onnx] Forwarding to gear_sonic.eval_agent_trl with: "
          f"{' '.join(rest_argv)}", flush=True)

    import runpy
    runpy.run_module("gear_sonic.eval_agent_trl", run_name="__main__")


if __name__ == "__main__":
    main()
