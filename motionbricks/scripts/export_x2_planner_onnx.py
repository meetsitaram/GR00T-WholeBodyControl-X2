#!/usr/bin/env python
"""Export the X2 kinematic planner (MotionBricks NeuralPlannerCore) to ONNX.

One exported graph = one full replan: exactly 3 one-shot model forwards
(root -> pose -> vqvae-decode) plus the torch pre/post math that
``NeuralPlannerCore.replan_with_velocity`` / ``replan_with_pose_template``
performs around them (canonicalize, qpos->motion-transforms FK, constraint
window construction, uncanonicalize, FILTER_QPOS context blend, wxyz
reorder). The graph does NOT contain the Python-side ring buffer, cursor,
truncation to ``num_pred_frames``, ZMQ, or intent->velocity mapping -- see
the README written next to the exported .onnx for the runtime glue contract.

Two graphs are exported (mirroring the shipped G1 fused-planner ONNX,
``docs/source/references/planner_onnx.md``, but adapted to the X2
NeuralPlannerCore's velocity-intent design):

  * ``x2_planner_template.onnx``  (PRIMARY -- pose-template mode; the
    4-frame clip-library segment selection is baked into the graph with
    ``mode`` + ``random_seed`` int64 tensor inputs, exactly like the G1
    export did)
  * ``x2_planner_velocity.onnx``  (secondary -- velocity-only mode)

Interface (both graphs):

  inputs:
    context_mujoco_qpos [1, 4, 38] float32   world frame, wxyz root quat
    velocity_intent     [1, 4]     float32   [yaw_rate, vel_x, vel_z, hip_h]
                                             (motion-rep frame channels, the
                                             SAME 4-vector NeuralPlannerCore
                                             takes -- see neural_planner.py)
  template-only extra inputs:
    mode                [1] int64            clip-library mode index
    random_seed         [1] int64            segment-start selection seed
                                             (start = seed % max(1, n-4))
  outputs:
    mujoco_qpos         [1, 64, 38] float32  padded; world frame, wxyz
    num_pred_frames     [1] int64            valid-frame count (= 4*tokens)

Export choices (per the shipped precedents):
  * torch.onnx.export TRACE path, opset 17, do_constant_folding=False
    (protects quantizer/codebook paths -- same reasoning as
    gear_sonic/scripts/reexport_x2_g1_onnx.py).
  * FP32 only. The masked num-token logits use -inf; FP16 is unsafe.
  * batch=1, all shapes static.

Trace-compat deviations from the eager code path (each is behaviour-
preserving; parity is verified numerically by --verify):
  1. ``_X2ClipLibrary.sample_target_segment`` uses Python ``int()`` /
     ``.item()`` for the seed-modulo segment start. For the traced graph it
     is monkeypatched (export-time only) with a pure-tensor equivalent
     (same clamp + ``seed % max(1, n-4)`` arithmetic, via index_select).
  2. ``num_pred_frames`` is returned as an int64 tensor instead of a
     Python int (the eager API truncates the buffer with ``int(...)`` --
     that truncation is Python glue on the Jetson side).

Usage:
  python motionbricks/scripts/export_x2_planner_onnx.py \
      --vqvae-ckpt ... --pose-ckpt ... --root-ckpt ... \
      --clip-ckpt motionbricks/out/X2-clip.ckpt \
      --out-dir $SONIC_HOME/x2/kplanner_onnx \
      --mode both --verify
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch as t
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MB_ROOT = _REPO_ROOT / "motionbricks"
for p in (str(_MB_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

log = logging.getLogger("export_x2_planner_onnx")

QPOS_DIM = 38
NUM_CTX_FRAMES = 4
PADDED_FRAMES = 64  # max_tokens(16) * frames_per_token(4)


# ---------------------------------------------------------------------------
# Tensor-only clip-segment sampler (trace-compatible replacement for
# _X2ClipLibrary.sample_target_segment; identical arithmetic, no .item()).
# ---------------------------------------------------------------------------


def _sample_target_segment_traceable(lib, mode_idx: t.Tensor, random_seed: t.Tensor) -> dict:
    """Pure-tensor version of ``_X2ClipLibrary.sample_target_segment``.

    Args:
        lib: the ``_X2ClipLibrary`` instance (buffers registered on it).
        mode_idx: int64 tensor, any shape with 1 element.
        random_seed: int64 tensor, any shape with 1 element.

    Matches the eager logic exactly:
        m = clamp(mode_idx, 0, num_modes-1)
        n = num_frames_per_clip[m]
        max_start = max(1, n - 4)
        start = seed % max_start
        slice = [start, start+4)
    """
    nftok = lib.NUM_FRAMES_PER_TOKEN
    device = lib.num_frames_per_clip.device
    # Accept Python ints too: the eager NeuralPlannerCore API passes ints
    # (as_tensor is a no-op passthrough for traced tensor inputs).
    mode_idx = t.as_tensor(mode_idx, device=device)
    random_seed = t.as_tensor(random_seed, device=device)
    mode = mode_idx.reshape([1]).long().clamp(min=0, max=lib.num_modes - 1)
    n = lib.num_frames_per_clip.index_select(0, mode).long()  # [1]
    max_start = (n - nftok).clamp(min=1)  # [1]
    seed = random_seed.reshape([1]).long()
    # Python's % on non-negative operands == torch remainder; seeds from the
    # runtime are non-negative by contract (documented in the README).
    start = t.remainder(seed, max_start)  # [1]
    frame_idx = (start + t.arange(nftok, device=start.device).long())  # [4]

    def _pick(buf: t.Tensor) -> t.Tensor:
        per_mode = buf.index_select(0, mode)[0]  # [max_frames, ...]
        return per_mode.index_select(0, frame_idx)  # [4, ...]

    return {
        "root_positions": _pick(lib.global_root_positions),
        "joint_positions": _pick(lib.global_joint_positions),
        "joint_rotations": _pick(lib.global_joint_rotations),
        "headings": _pick(lib.global_headings),
    }


# ---------------------------------------------------------------------------
# Export wrapper modules. forward() == one full replan, pure tensors in/out.
# They call the SAME NeuralPlannerCore internals (_canonicalize, converter FK,
# _predict_with_velocity / _predict_with_pose_template) -- no reimplementation.
# ---------------------------------------------------------------------------


class _ReplanWrapperBase(nn.Module):
    def __init__(self, core) -> None:
        super().__init__()
        self.core = core
        assert core.diagnostic_mask_hook is None, "diagnostic_mask_hook must be None"

    def _build_input_state(self, context_mujoco_qpos: t.Tensor, velocity_intent: t.Tensor) -> dict:
        core = self.core
        ctx = context_mujoco_qpos.clone()
        input_state: dict = {
            "context_mujoco_qpos": ctx,
            "raw_context_mujoco_qpos": ctx.clone(),
            "velocity_intent": velocity_intent.float(),
        }
        if core.FORCE_CANONICALIZATION:
            core._canonicalize_mujoco_qpos(input_state)
        (
            input_state["context_global_joint_positions"],
            input_state["context_global_joint_rotations"],
        ) = core._converter.convert_mujoco_qpos_to_motion_transforms(
            input_state["context_mujoco_qpos"]
        )
        return input_state


class VelocityReplanWrapper(_ReplanWrapperBase):
    """forward = NeuralPlannerCore.replan_with_velocity minus ring buffer."""

    def forward(self, context_mujoco_qpos: t.Tensor, velocity_intent: t.Tensor):
        input_state = self._build_input_state(context_mujoco_qpos, velocity_intent)
        _, mujoco_qpos, num_pred_frames = self.core._predict_with_velocity(
            input_state, num_tokens=None
        )
        return mujoco_qpos, num_pred_frames.reshape([1]).long()


class TemplateReplanWrapper(_ReplanWrapperBase):
    """forward = NeuralPlannerCore.replan_with_pose_template minus ring buffer.

    The clip library's segment sampler is monkeypatched (in __init__, on the
    library instance only) with the pure-tensor equivalent above so mode /
    random_seed stay graph inputs.
    """

    def __init__(self, core) -> None:
        super().__init__(core)
        assert core._clip_library is not None, "template export needs a clip library"
        lib = core._clip_library
        # Export-time, instance-only monkeypatch (deviation #1 in module docstring).
        lib.sample_target_segment = (
            lambda mode_idx, random_seed: _sample_target_segment_traceable(
                lib, mode_idx, random_seed
            )
        )

    def forward(
        self,
        context_mujoco_qpos: t.Tensor,
        velocity_intent: t.Tensor,
        mode: t.Tensor,
        random_seed: t.Tensor,
    ):
        input_state = self._build_input_state(context_mujoco_qpos, velocity_intent)
        _, mujoco_qpos, num_pred_frames = self.core._predict_with_pose_template(
            input_state, mode_idx=mode, num_tokens=None, random_seed=random_seed
        )
        return mujoco_qpos, num_pred_frames.reshape([1]).long()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_planner(args, device: str = "cpu"):
    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths,
        load_x2_planner,
    )

    # Version dir (hparams/stats/skeleton) MUST come from the checkpoint's own
    # run dir, NOT the default. A checkpoint trained on one corpus carries its
    # own normalization stats; pairing it with a different run's stats is a
    # silent mismatch that --verify cannot catch (parity uses the same config on
    # both torch and onnx sides). Standard Lightning layout:
    #   <version_dir>/checkpoints/<ckpt>  ->  version_dir = ckpt.parents[1]
    default_paths = X2PlannerPaths.default()

    def _vdir(ckpt: str) -> Path:
        vd = Path(ckpt).resolve().parents[1]
        return vd if (vd / "hparams.yaml").exists() else default_paths.vqvae_version_dir

    paths = X2PlannerPaths(
        vqvae_ckpt=Path(args.vqvae_ckpt),
        pose_ckpt=Path(args.pose_ckpt),
        root_ckpt=Path(args.root_ckpt),
        vqvae_version_dir=_vdir(args.vqvae_ckpt),
        pose_version_dir=_vdir(args.pose_ckpt),
        root_version_dir=_vdir(args.root_ckpt),
    )
    log.info("version dirs: vqvae=%s pose=%s root=%s",
             paths.vqvae_version_dir, paths.pose_version_dir, paths.root_version_dir)
    t0 = time.time()
    core = load_x2_planner(paths, device=device, clip_library_ckpt=Path(args.clip_ckpt))
    log.info("planner loaded in %.1fs (device=%s)", time.time() - t0, device)

    # Disable ALL transformer fast paths so the trace never records the fused
    # aten::_transformer_encoder_layer_fwd / nested-tensor kernels (both are
    # un-exportable to ONNX). The global MHA switch covers the per-layer fused
    # path; the per-encoder flags cover the nested-tensor conversion.
    torch.backends.mha.set_fastpath_enabled(False)
    n_disabled = 0
    for m in core.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
            # torch caches the decision in use_nested_tensor at __init__;
            # flipping enable_nested_tensor alone does NOT disable the path.
            m.use_nested_tensor = False
            n_disabled += 1
    log.info("disabled nested-tensor fast path on %d TransformerEncoder(s)", n_disabled)
    return core.eval()


# ---------------------------------------------------------------------------
# Example / test inputs
# ---------------------------------------------------------------------------


def _default_stand_qpos() -> np.ndarray:
    """Zero-joint stand qpos[38] (same as x2_kplanner._build_hardcoded_warmup_qpos)."""
    qpos = np.zeros(QPOS_DIM, dtype=np.float32)
    qpos[2] = 0.78
    qpos[3] = 1.0
    return qpos


def _qpos_from_anchor_pkl(path: Path) -> Optional[np.ndarray]:
    """Frame 0 of a deploy-PKL schema anchor -> qpos[38] wxyz.

    Mirrors gear_sonic/scripts/x2_kplanner._qpos_from_deploy_pkl_frame.
    """
    try:
        import joblib

        obj = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("anchor pkl %s unreadable: %r", path, exc)
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    inner = next(iter(obj.values()))
    if not isinstance(inner, dict) or not {"root_trans_offset", "root_rot", "dof"}.issubset(inner):
        return None
    trans = np.asarray(inner["root_trans_offset"], dtype=np.float32).reshape(-1, 3)[0]
    rot_xyzw = np.asarray(inner["root_rot"], dtype=np.float32).reshape(-1, 4)[0]
    dof = np.asarray(inner["dof"], dtype=np.float32).reshape(-1, 31)[0]
    qpos = np.zeros(QPOS_DIM, dtype=np.float32)
    qpos[0:3] = trans
    qpos[3:7] = [rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]]
    qpos[7:38] = dof
    return qpos


def _context_from_qpos(qpos: np.ndarray) -> np.ndarray:
    """Tile a single stand frame into a [1, 4, 38] context."""
    return np.tile(qpos[None, None, :], (1, NUM_CTX_FRAMES, 1)).astype(np.float32)


def _build_parity_cases(anchor_pkl: Optional[Path]):
    """>=3 cases: idle stand, forward slow-walk, turning walk; anchor-based context."""
    stand = _default_stand_qpos()
    cases = [
        {
            "name": "synthetic_stand_idle",
            "context": _context_from_qpos(stand),
            "velocity_intent": np.array([[0.0, 0.0, 0.0, 0.78]], dtype=np.float32),
            "mode": 0,
            "random_seed": 1234,
        },
        {
            "name": "synthetic_stand_forward_slowwalk",
            "context": _context_from_qpos(stand),
            "velocity_intent": np.array([[0.0, 0.0, 0.35, 0.78]], dtype=np.float32),
            "mode": 1,
            "random_seed": 777,
        },
        {
            "name": "synthetic_stand_turning_walk",
            "context": _context_from_qpos(stand),
            "velocity_intent": np.array([[0.4, 0.1, 0.5, 0.78]], dtype=np.float32),
            "mode": 2,
            "random_seed": 424242,
        },
    ]
    if anchor_pkl is not None and anchor_pkl.is_file():
        anchor_qpos = _qpos_from_anchor_pkl(anchor_pkl)
        if anchor_qpos is not None:
            cases.append(
                {
                    "name": f"real_anchor_{anchor_pkl.stem}",
                    "context": _context_from_qpos(anchor_qpos),
                    "velocity_intent": np.array([[0.0, 0.0, 0.4, float(anchor_qpos[2])]], dtype=np.float32),
                    "mode": 1,
                    "random_seed": 20260716,
                }
            )
        else:
            log.warning("anchor pkl %s did not match deploy-PKL schema; skipped", anchor_pkl)
    else:
        log.warning("anchor pkl %s missing; parity uses synthetic cases only", anchor_pkl)
    return cases


# ---------------------------------------------------------------------------
# Torch reference (genuine NeuralPlannerCore API, not the wrapper)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _torch_reference_replan(core, case: dict, template: bool):
    """Run a replan through the real NeuralPlannerCore lifecycle.

    reset() with the exact 4 context frames leaves the cursor such that
    get_context_mujoco_qpos() returns those same 4 frames (verified below),
    so the predict path sees inputs identical to the ONNX graph's.
    """
    ctx = torch.from_numpy(case["context"][0])  # [4, 38]
    core.reset(ctx)
    got_ctx = core.get_context_mujoco_qpos().cpu().numpy()
    assert np.allclose(got_ctx, case["context"]), "ring-buffer context mismatch"
    intent = torch.from_numpy(case["velocity_intent"][0])
    if template:
        _, qpos, npf = core.replan_with_pose_template(
            intent, mode_idx=int(case["mode"]), random_seed=int(case["random_seed"])
        )
    else:
        _, qpos, npf = core.replan_with_velocity(intent)
    return qpos.cpu().numpy(), int(npf)


# ---------------------------------------------------------------------------
# Export + verify
# ---------------------------------------------------------------------------


def _register_lift_fresh_symbolic() -> None:
    """Map aten::lift_fresh -> identity for the ONNX exporter.

    ``torch.tensor([...])`` inside traced code (constant construction in
    neural_planner.py / quaternions.py) emits ``aten::lift_fresh``, which
    has no opset-17 symbolic in torch 2.7. lift_fresh is semantically a
    fresh-copy no-op, so identity is exact.
    """
    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic("aten::lift_fresh", lambda g, self_: self_, 17)


def _export_graph(wrapper: nn.Module, example_inputs: tuple, out_path: Path, input_names, output_names):
    _register_lift_fresh_symbolic()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".onnx.tmp")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(tmp_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=False,  # protect quantizer / codebook paths
            dynamo=False,
            verbose=False,
        )
    tmp_path.replace(out_path)
    log.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def _verify_graph(onnx_path: Path, core, cases, template: bool):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    results = []
    all_pass = True
    for case in cases:
        feeds = {
            "context_mujoco_qpos": case["context"],
            "velocity_intent": case["velocity_intent"],
        }
        if template:
            feeds["mode"] = np.array([case["mode"]], dtype=np.int64)
            feeds["random_seed"] = np.array([case["random_seed"]], dtype=np.int64)
        onnx_qpos, onnx_npf = sess.run(None, feeds)
        onnx_npf = int(onnx_npf.reshape(-1)[0])

        ref_qpos, ref_npf = _torch_reference_replan(core, case, template)

        npf_ok = onnx_npf == ref_npf
        n = min(onnx_npf, ref_npf)
        diff_valid = float(np.max(np.abs(onnx_qpos[:, :n] - ref_qpos[:, :n]))) if n else float("nan")
        diff_padded = float(np.max(np.abs(onnx_qpos - ref_qpos)))
        passed = npf_ok and diff_valid < 1e-4
        all_pass &= passed
        results.append(
            {
                "case": case["name"],
                "num_pred_frames_torch": ref_npf,
                "num_pred_frames_onnx": onnx_npf,
                "max_abs_qpos_diff_valid_frames": diff_valid,
                "max_abs_qpos_diff_full_padded": diff_padded,
                "pass": bool(passed),
            }
        )
        log.info(
            "[%s] %s: npf torch=%d onnx=%d | max|dqpos| valid=%.3e padded=%.3e -> %s",
            "template" if template else "velocity",
            case["name"], ref_npf, onnx_npf, diff_valid, diff_padded,
            "PASS" if passed else "FAIL",
        )
    return all_pass, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--pose-ckpt", required=True)
    ap.add_argument("--root-ckpt", required=True)
    ap.add_argument(
        "--clip-ckpt",
        default=str(_MB_ROOT / "out/X2-clip.ckpt"),
        help="baked pose-template clip library (X2-clip.ckpt)",
    )
    ap.add_argument("--out-dir", default=str(Path(os.environ.get("SONIC_HOME", Path.home() / ".cache/sonic")) / "x2/kplanner_onnx"))
    ap.add_argument("--mode", choices=["velocity", "template", "both"], default="both")
    ap.add_argument("--verify", action="store_true", help="run torch-vs-onnx parity harness")
    ap.add_argument(
        "--anchor-pkl",
        default=str(_REPO_ROOT / "gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl"),
        help="deploy-PKL stand anchor for the real-context parity case",
    )
    ap.add_argument("--device", default="cpu", help="cpu recommended (GPU may be busy)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.manual_seed(0)

    core = _load_planner(args, device=args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stand_ctx = torch.from_numpy(_context_from_qpos(_default_stand_qpos()))
    example_intent = torch.tensor([[0.0, 0.0, 0.35, 0.78]], dtype=torch.float32)

    cases = _build_parity_cases(Path(args.anchor_pkl)) if args.verify else []
    verify_report: dict = {}
    overall_pass = True

    if args.mode in ("velocity", "both"):
        log.info("=== exporting velocity graph ===")
        # .eval() matters: onnx.export saves/restores the WRAPPER's training
        # flag around the trace; a fresh nn.Module defaults to training=True
        # and the restore would flip the whole core into train mode.
        wrapper = VelocityReplanWrapper(core).eval()
        path = out_dir / "x2_planner_velocity.onnx"
        _export_graph(
            wrapper,
            (stand_ctx, example_intent),
            path,
            input_names=["context_mujoco_qpos", "velocity_intent"],
            output_names=["mujoco_qpos", "num_pred_frames"],
        )
        if args.verify:
            ok, res = _verify_graph(path, core, cases, template=False)
            verify_report["velocity"] = res
            overall_pass &= ok

    if args.mode in ("template", "both"):
        log.info("=== exporting template graph (primary) ===")
        wrapper = TemplateReplanWrapper(core).eval()  # monkeypatches lib sampler (tensor version)
        path = out_dir / "x2_planner_template.onnx"
        example_mode = torch.tensor([1], dtype=torch.int64)
        example_seed = torch.tensor([1234], dtype=torch.int64)
        _export_graph(
            wrapper,
            (stand_ctx, example_intent, example_mode, example_seed),
            path,
            input_names=["context_mujoco_qpos", "velocity_intent", "mode", "random_seed"],
            output_names=["mujoco_qpos", "num_pred_frames"],
        )
        if args.verify:
            # NOTE: after TemplateReplanWrapper.__init__ the library's eager
            # sampler is replaced by the tensor version; the torch reference
            # below therefore exercises the tensor sampler too. That is
            # intentional for segment-selection identity -- but ALSO cross-check
            # eager-vs-tensor sampler equivalence explicitly first:
            _check_sampler_equivalence(core, cases)
            ok, res = _verify_graph(path, core, cases, template=True)
            verify_report["template"] = res
            overall_pass &= ok

    if args.verify:
        report_path = out_dir / "parity_report.json"
        report_path.write_text(json.dumps(verify_report, indent=2))
        log.info("parity report -> %s", report_path)
        if not overall_pass:
            log.error("PARITY FAILED -- see report")
            return 1
        log.info("ALL PARITY CASES PASSED")
    return 0


def _check_sampler_equivalence(core, cases) -> None:
    """Assert tensor sampler == original eager sampler on every parity case.

    The original (un-patched) implementation is taken from the class, so this
    works even after the instance-level monkeypatch.
    """
    from motionbricks.motion_backbone.inference.neural_planner import _X2ClipLibrary

    lib = core._clip_library
    for case in cases:
        eager = _X2ClipLibrary.sample_target_segment(
            lib, int(case["mode"]), int(case["random_seed"])
        )
        tensor = _sample_target_segment_traceable(
            lib,
            torch.tensor([case["mode"]], dtype=torch.int64),
            torch.tensor([case["random_seed"]], dtype=torch.int64),
        )
        for k in eager:
            d = float((eager[k] - tensor[k]).abs().max())
            assert d == 0.0, f"sampler mismatch on {case['name']}/{k}: {d}"
    log.info("tensor segment sampler == eager sampler on all %d cases", len(cases))


if __name__ == "__main__":
    sys.exit(main())
