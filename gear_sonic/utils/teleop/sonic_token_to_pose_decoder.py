"""SONIC ``motion_token`` -> ``joint_pos_mj`` decoder for the live VLA bridge.

Why this module exists
----------------------
The live VLA bridge (:mod:`gear_sonic.scripts.vla.live_vla_publish_motion_token`)
publishes ``motion_token`` chunks over the ``body_pose`` ZMQ topic, but the
C++ deploy (``agi_x2_deploy_onnx_ref``) explicitly does NOT consume that
field -- the ``zmq_pose_input_source.hpp`` header documents it as a
"v1 hook for VLA-direct token streaming; currently logged but otherwise
unused" because the deploy runs a fully-fused encoder+FSQ+decoder ONNX
that re-tokenizes whichever ``joint_pos_mj_future`` window is on the
wire (i.e., the trained ``default_angles`` stand pose by default). Net
effect: the VLA's intent never reaches the body. Fingers move (left/
right hand joints are an explicit AimDK passthrough), the SONIC tracker
holds idle_stand on every other DOF.

This module closes the loop on the **bridge side** (no C++ rebuild
required): take the VLA's predicted ``motion_token`` chunk, run it
through the SONIC ``g1_dyn`` decoder to recover the IsaacLab-order
delta-action it implies, fold that into the C++ deploy's joint-target
formula

    target_mj[mj] = default_angles[mj] + action_il[il] * action_scale[mj]

(see ``policy_parameters.hpp:175-208`` and
``x2_deploy_onnx_ref.cpp:1394-1399``), and hand the resulting
``(31,)`` MuJoCo-order pose back to the publisher so it can be put on
the wire as ``joint_pos_mj`` / ``joint_pos_mj_future[k]``.

The deploy's encoder will then re-tokenize that wire trajectory (lossy
round-trip; ``validate_encode_decode_loop.py`` measures ~0.92 mean
cosine similarity between the encode->decode round-trip and the
original motion_token), and its decoder will produce an action_il close
to what the VLA wanted -- i.e., the body finally moves under VLA
authority instead of statically tracking idle_stand.

Joint-order constants (``IL_TO_MJ_DOF``, ``X2_ACTION_SCALE_MJ``,
``X2_DEFAULT_ANGLES_MJ``) are duplicated here to avoid a circular
import via ``gear_sonic.scripts.eval_x2_mujoco`` (which depends on
``mujoco`` at import time). Both versions are unit-tested for parity
in ``tests/test_sonic_token_to_pose_decoder.py``.

Caveats
-------
* The decoder is severely OOD when proprio is zero (per the
  ``validate_encode_decode_loop.py:46-56`` docstring: "decoder check
  uses a generous --decoder-rmse-threshold (default 0.30 rad)"). The
  bridge prefers the assembled live proprio when available
  (:func:`assemble_proprio_from_x2_debug` -- partial coverage, history
  fields are zero-filled) but falls back to all-zeros with a one-shot
  warning. Even at 0.30 rad RMSE the body will move *visibly*; that's
  the goal of v0 ("get something dynamic on the wire so we can see the
  VLA latching onto the prompt"). Better proprio is a v1 task.
* The decoder is ~5 M params -- single-instance per-tick CPU inference
  is sub-millisecond on the Blackwell test box, well under the 20 ms
  50 Hz publisher budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Joint-order + scaling constants (mirrored from
# ``gear_sonic_deploy/.../policy_parameters.hpp`` and
# ``gear_sonic/scripts/eval_x2_mujoco.py``)
# ---------------------------------------------------------------------------

NUM_BODY_DOFS = 31

# IL_TO_MJ_DOF[il_idx] = mj_idx -- use to map IL-ordered tensors to MJ.
IL_TO_MJ_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29, 15, 22, 4, 10,
     30, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
# MJ_TO_IL_DOF[mj_idx] = il_idx -- use to map MJ-ordered observations into IL.
MJ_TO_IL_DOF = np.array(
    [0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 12,
     17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30, 11, 16],
    dtype=np.int64,
)

# Mirror of ``policy_parameters.hpp::default_angles``: training-time
# stand pose in MuJoCo joint order (radians).
X2_DEFAULT_ANGLES_MJ = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
      0.0, 0.0, 0.0,
      0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
      0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
      0.0, 0.0],
    dtype=np.float64,
)
assert X2_DEFAULT_ANGLES_MJ.shape == (NUM_BODY_DOFS,)

# Mirror of ``policy_parameters.hpp::x2_action_scale``: per-joint
# action scaling in MuJoCo joint order. Computed as
# ``0.25 * EFFORT_LIMIT[joint_family] / kp[joint]`` per the C++ comment.
X2_ACTION_SCALE_MJ = np.array(
    [0.3027293235, 0.3027293235, 0.3027293235, 0.3027293235, 0.631551332,
     0.4210342213,
     0.3027293235, 0.3027293235, 0.3027293235, 0.3027293235, 0.631551332,
     0.4210342213,
     0.7466542707, 0.8420684427, 0.8420684427,
     0.631551332, 0.631551332, 0.4210342213, 0.4210342213, 0.4210342213,
     0.07152083551, 0.07152083551,
     0.631551332, 0.631551332, 0.4210342213, 0.4210342213, 0.4210342213,
     0.07152083551, 0.07152083551,
     0.03874045257, 0.008940104439],
    dtype=np.float64,
)
assert X2_ACTION_SCALE_MJ.shape == (NUM_BODY_DOFS,)

# Proprio dimension consumed by the SONIC ``g1_dyn`` decoder (see
# ``UniversalTokenActor.decoder = SimpleMLP([1054, ..., 31])`` in
# ``eval_x2_mujoco.py:329`` -- 1054 = 64 token + 990 proprio).
PROPRIO_DIM = 990
SONIC_TOKEN_DIM = 64


def proprio_zero(dtype=np.float32) -> np.ndarray:
    """Return a 990-D all-zero proprio placeholder.

    The decoder is OOD with this input but still produces non-trivial
    actions (~0.30 rad RMSE versus the matching commanded pose, per
    ``validate_encode_decode_loop.py``). Used as the v0 fallback when
    the bridge can't yet assemble a live 990-D vector.
    """
    return np.zeros(PROPRIO_DIM, dtype=dtype)


def action_il_to_target_pose_mj(
    action_il: np.ndarray,
    *,
    base_pose_mj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply ``target = base + action_il[il] * action_scale[mj]`` per joint.

    Mirrors ``x2_deploy_onnx_ref.cpp`` line 1398. ``base_pose_mj``
    defaults to ``X2_DEFAULT_ANGLES_MJ`` (the training-time stand pose
    that the C++ deploy uses as its reference origin).
    """
    if base_pose_mj is None:
        base = X2_DEFAULT_ANGLES_MJ
    else:
        base = np.asarray(base_pose_mj, dtype=np.float64).reshape(-1)
        if base.shape != (NUM_BODY_DOFS,):
            raise ValueError(
                f"base_pose_mj must be (31,); got {base.shape}"
            )
    a_il = np.asarray(action_il, dtype=np.float64).reshape(-1)
    if a_il.shape != (NUM_BODY_DOFS,):
        raise ValueError(
            f"action_il must be (31,); got {a_il.shape}"
        )
    target_mj = np.empty(NUM_BODY_DOFS, dtype=np.float64)
    for mj in range(NUM_BODY_DOFS):
        il = int(MJ_TO_IL_DOF[mj])
        target_mj[mj] = base[mj] + a_il[il] * X2_ACTION_SCALE_MJ[mj]
    return target_mj


class SonicTokenToPoseDecoder:
    """Decode VLA ``motion_token`` (64-D) -> body ``joint_pos_mj`` (31-D).

    Loads the SONIC ``g1_dyn`` decoder weights from the same .pt
    checkpoint the recorder already uses for label generation
    (``--sonic-checkpoint``), keeps a single ``UniversalTokenActor``
    instance on the chosen device, and exposes a thread-safe per-step
    decode method.

    Parameters
    ----------
    checkpoint_path:
        Path to a SONIC .pt checkpoint (e.g. ``model_step_025000.pt``).
        Must contain ``actor_module.decoders.g1_dyn.module.*`` keys.
    device:
        Torch device. ``"cpu"`` is fine -- the decoder is ~5 M params
        and per-tick inference takes <1 ms on a modern CPU. Pass
        ``"cuda"`` only if your local CUDA build supports the GPU
        (Blackwell sm_120 currently breaks the .venv torch 2.6 build,
        see ``run_x2_quest3_planner_stack.sh`` SONIC_TOKENIZER_DEVICE
        comment).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        # Defer torch import so module-level constants stay importable
        # in environments without torch (e.g. the C++ deploy parity
        # tests in CI).
        import torch  # noqa: F401

        from gear_sonic.scripts.sonic_motion_token_labeler import (
            SonicMotionTokenLabeler,
        )

        self._device = device
        self._labeler = SonicMotionTokenLabeler(
            checkpoint_path, device=device
        )
        # Sanity-check the decoder loaded; if the .pt is encoder-only
        # (some pre-fuse exports), error out loudly here rather than
        # silently emitting random actions at runtime.
        decoder = self._labeler._actor.decoder
        first_layer = next(iter(decoder.module.modules()), None)
        if first_layer is None:
            raise RuntimeError(
                f"SONIC checkpoint at {checkpoint_path} has no decoder "
                "module. Is this an encoder-only export?"
            )

    @property
    def device(self) -> str:
        return self._device

    def decode_step(
        self,
        token: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        """Decode one ``(64,)`` motion_token -> ``(31,)`` action_il.

        Parameters
        ----------
        token:
            ``(64,)`` SONIC motion token.
        proprio:
            ``(990,)`` proprioception vector. Pass :func:`proprio_zero`
            if you don't have history yet (decoder is OOD but still
            produces non-trivial output).

        Returns
        -------
        action_il:
            ``(31,)`` IsaacLab-order delta-action vector. Apply
            :func:`action_il_to_target_pose_mj` to convert to a
            wire-ready MuJoCo-order pose.
        """
        import torch

        tok = np.asarray(token, dtype=np.float32).reshape(-1)
        if tok.shape != (SONIC_TOKEN_DIM,):
            raise ValueError(
                f"token must be (64,); got {tok.shape}"
            )
        prop = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if prop.shape != (PROPRIO_DIM,):
            raise ValueError(
                f"proprio must be ({PROPRIO_DIM},); got {prop.shape}"
            )

        with torch.no_grad():
            tok_t = torch.from_numpy(tok).to(self._device).unsqueeze(0)
            prop_t = torch.from_numpy(prop).to(self._device).unsqueeze(0)
            decoder_in = torch.cat([tok_t, prop_t], dim=-1)
            action_il = self._labeler._actor.decoder(decoder_in)
            return action_il.squeeze(0).cpu().numpy().astype(np.float64)

    def decode_chunk(
        self,
        token_chunk: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        """Batched decode for a horizon of tokens, sharing one proprio.

        Parameters
        ----------
        token_chunk:
            ``(T, 64)`` array of motion tokens (typically ``T=40``).
        proprio:
            ``(990,)`` proprio shared across all T steps. The deploy's
            decoder normally sees a *current* proprio for *each* step
            during PPO rollouts; we don't have those future proprios
            at chunk-build time so we feed the same current proprio.
            This makes the decoded trajectory the model's
            "would-do-next assuming I stay where I am right now" --
            exactly the open-loop intent we want on the wire.

        Returns
        -------
        action_il_chunk:
            ``(T, 31)`` IsaacLab-order delta-action chunk.
        """
        import torch

        tok = np.asarray(token_chunk, dtype=np.float32)
        if tok.ndim != 2 or tok.shape[1] != SONIC_TOKEN_DIM:
            raise ValueError(
                f"token_chunk must be (T, 64); got {tok.shape}"
            )
        prop = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if prop.shape != (PROPRIO_DIM,):
            raise ValueError(
                f"proprio must be ({PROPRIO_DIM},); got {prop.shape}"
            )

        T = tok.shape[0]
        with torch.no_grad():
            tok_t = torch.from_numpy(tok).to(self._device)
            prop_t = (
                torch.from_numpy(prop)
                .to(self._device)
                .unsqueeze(0)
                .expand(T, -1)
            )
            decoder_in = torch.cat([tok_t, prop_t], dim=-1)
            action_il = self._labeler._actor.decoder(decoder_in)
            return action_il.cpu().numpy().astype(np.float64)


def decode_token_chunk_to_pose_chunk(
    decoder: SonicTokenToPoseDecoder,
    token_chunk: np.ndarray,
    proprio: np.ndarray,
    *,
    base_pose_mj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience: ``(T, 64)`` tokens -> ``(T, 31)`` MuJoCo-order poses.

    Wraps :meth:`SonicTokenToPoseDecoder.decode_chunk` plus
    :func:`action_il_to_target_pose_mj` for each step.
    """
    actions_il = decoder.decode_chunk(token_chunk, proprio)
    pose_chunk = np.empty(actions_il.shape, dtype=np.float64)
    for k in range(actions_il.shape[0]):
        pose_chunk[k] = action_il_to_target_pose_mj(
            actions_il[k], base_pose_mj=base_pose_mj
        )
    return pose_chunk


__all__ = [
    "IL_TO_MJ_DOF",
    "MJ_TO_IL_DOF",
    "NUM_BODY_DOFS",
    "PROPRIO_DIM",
    "SONIC_TOKEN_DIM",
    "SonicTokenToPoseDecoder",
    "X2_ACTION_SCALE_MJ",
    "X2_DEFAULT_ANGLES_MJ",
    "action_il_to_target_pose_mj",
    "decode_token_chunk_to_pose_chunk",
    "proprio_zero",
]
