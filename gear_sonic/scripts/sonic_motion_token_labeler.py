"""SONIC motion-token labeler for X2 LeRobot datasets (M6 / Stage C1).

Replaces the M3 placeholder ``action.motion_token = zeros(64)`` field in the
synthetic smoketest dataset with the SONIC tracking encoder's actual FSQ-
quantized output for each frame of the recorded body trajectory. Without this
replacement the VLA learns to predict zeros (the "stand still" embedding),
which means even a perfect fine-tune on the gradient/mujoco-backed dataset
produces a robot that just stands -- the prompt has nothing to ground against
in the action labels.

What this module does
---------------------

For each frame ``f`` in a recorded body trajectory ``body_q ∈ R^(T, 31)``:

1. Build the 680-D ``tokenizer_obs`` vector that IsaacLab feeds to the SONIC
   tracking encoder during training. The layout (10 future frames of
   joint_pos + joint_vel + 6D root rotation diff) comes from
   :func:`gear_sonic.scripts.eval_x2_mujoco.build_tokenizer_obs`, which is the
   single source of truth shared with the deploy harness's
   :class:`UniversalTokenActor`.
2. Run that obs through the SONIC ``g1`` encoder (a SimpleMLP loaded from a
   .pt checkpoint -- e.g. ``model_step_025000.pt`` from the h200 run) to get
   a 64-D continuous latent.
3. Reshape to ``(2, 32)`` and apply FSQ (Finite Scalar Quantization) with 32
   levels per dim, matching :func:`fsq_quantize` in
   :mod:`gear_sonic.scripts.eval_x2_mujoco` and the production training
   recipe (``num_fsq_levels=32, fsq_level_list=32``).
4. Flatten back to a 64-D float64 vector. That's the per-frame
   ``action.motion_token`` label.

The returned tokens are guaranteed to lie on the FSQ lattice
``{-1, -1+2/L, ..., 1-2/L, 1}`` for ``L=32``, which is the discrete codebook
the deploy harness's ONNX decoder expects on the wire.

Checkpoint loading
------------------

The SONIC .pt checkpoints contain pickled :class:`trl.experimental` objects
from HuggingFace's TRL library that are not in our runtime environment. To
avoid forcing every dataset build to install ``trl``, this module ships a
``_tolerant_torch_load`` helper that swaps unknown classes for an inert
stub during unpickling. Only the ``policy_state_dict`` tensors are then
extracted and remapped onto :class:`UniversalTokenActor` -- we never run
the unrelated trainer state.

Usage
-----

::

    from gear_sonic.scripts.sonic_motion_token_labeler import (
        SonicMotionTokenLabeler,
    )

    labeler = SonicMotionTokenLabeler(
        checkpoint_path="/path/to/model_step_025000.pt",
        device="cpu",
        motion_fps=50.0,
    )
    tokens = labeler.label_trajectory(body_trajectory)  # (T, 64) float64
    assert tokens.shape == (body_trajectory.shape[0], 64)

The same labeler instance is reused across episodes -- the encoder MLP is
loaded once (~50 MB on disk, ~5 MB in memory), and per-frame inference is
dominated by the 680->2048 first layer (~1.4 M FLOPs / frame, well under
1 ms on CPU).

Acceptance gate (tests/test_x2_motion_token_labels.py)
------------------------------------------------------

* tokens.shape == (T, 64)
* tokens.dtype == float64
* tokens are NOT all zero (deterministic vs. M3 placeholder)
* tokens lie on the FSQ lattice (each value rounds to k * 2/32 for some
  integer k, exactly).
* tokens vary across the trajectory (max - min > 0 per dim, summed over
  dims, yields nontrivial variation when the body trajectory has motion).
* labeler is deterministic (two calls with identical input produce
  bit-identical output on CPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
import types
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_x2_mujoco import (  # noqa: E402  (sys.path setup must come first)
    UniversalTokenActor,
    build_tokenizer_obs,
    fsq_quantize,
)


# ---------------------------------------------------------------------------
# Tolerant pickle loader -- handles checkpoints that reference modules we
# don't have installed (e.g. trl.experimental.ppo.ppo_trainer).
# ---------------------------------------------------------------------------


class _StubClass:
    """Inert placeholder for unpickled classes we don't care about."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401  (placeholder)
        pass

    def __setstate__(self, state) -> None:
        pass

    def __reduce__(self):
        return (_StubClass, ())


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler that returns ``_StubClass`` for any module-not-found case.

    The SONIC .pt checkpoints embed pickled
    ``trl.experimental.ppo.ppo_trainer.OnlineTrainerState`` objects from
    HuggingFace TRL. We only care about the tensor state dicts, so anything
    else can be safely stubbed out at load time.
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            return _StubClass


def _tolerant_pickle_module() -> types.ModuleType:
    """Return a module-like object torch.load accepts as ``pickle_module``."""
    mod = types.ModuleType("gear_sonic_tolerant_pickle")
    mod.Unpickler = _TolerantUnpickler  # type: ignore[attr-defined]
    mod.load = pickle.load  # type: ignore[attr-defined]
    mod.loads = pickle.loads  # type: ignore[attr-defined]
    return mod


def _tolerant_torch_load(path: str | Path, map_location: str = "cpu"):
    return torch.load(
        str(path),
        map_location=map_location,
        weights_only=False,
        pickle_module=_tolerant_pickle_module(),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


SONIC_MOTION_TOKEN_DIM: int = (
    UniversalTokenActor.MAX_NUM_TOKENS * UniversalTokenActor.TOKEN_DIM
)
"""Flattened SONIC motion-token dimensionality (2 tokens x 32 dims = 64)."""

TOKENIZER_OBS_DIM: int = 680
"""IsaacLab tokenizer-obs dimensionality (10 future frames * 68 features)."""

DEFAULT_MOTION_FPS: float = 50.0
"""Default frame rate for synthetic / replay X2 trajectories at 50 Hz."""

IDENTITY_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
"""Identity base quaternion (w, x, y, z) for the synthetic dataset where the
robot's pelvis stays in the world frame for every frame."""

IDENTITY_QUAT_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
"""Identity base quaternion (x, y, z, w) for ``motion_data["root_rot"]``."""


# ---------------------------------------------------------------------------
# Public labeler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelerStats:
    """Aggregate statistics returned by :meth:`SonicMotionTokenLabeler.label_trajectory`.

    Useful for asserting the labeler did real work (vs. the M3 zero
    placeholder) at dataset-build time.
    """

    num_frames: int
    """Frames labeled."""

    mean_l2: float
    """Mean L2 norm of the per-frame token vector."""

    max_abs: float
    """Maximum absolute token value across the batch."""

    unique_levels: int
    """Number of distinct values observed (FSQ should produce ~32 max)."""


class SonicMotionTokenLabeler:
    """Encode recorded body trajectories into 64-D SONIC motion tokens.

    Loads the SONIC ``g1`` encoder + FSQ once, then offers per-trajectory
    labeling. The encoder is a SimpleMLP (~5 M params, ~50 MB on disk in
    bf16); CPU inference is sub-millisecond per frame, so labeling 30 x
    200-frame episodes takes <1 s total.

    Args:
        checkpoint_path: Path to a SONIC .pt checkpoint (e.g.
            ``model_step_025000.pt`` from the h200 sphere-feet run). The
            checkpoint must contain a ``policy_state_dict`` with keys under
            ``actor_module.encoders.g1.module.*``.
        device: Where to run the encoder. ``"cpu"`` is fine for offline
            labeling; pass ``"cuda"`` if you happen to have a GPU and want
            to label a very large dataset.
        motion_fps: Frame rate of the body trajectories you'll pass to
            :meth:`label_trajectory`. Defaults to 50 Hz (matches the X2
            tokenizer rate and the ``record_synthetic_smoketest_dataset``
            default).

    Notes:
        The encoder is loaded with a tolerant unpickler so checkpoints
        containing pickled ``trl.experimental`` trainer state (which we
        don't have installed) load without erroring out -- only the
        encoder tensors are extracted and used.
    """

    _ENCODER_KEY_PREFIX = "actor_module.encoders.g1.module."

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        motion_fps: float = DEFAULT_MOTION_FPS,
    ) -> None:
        self._device = device
        self._motion_fps = float(motion_fps)
        self._actor = self._load_actor(checkpoint_path, device=device)
        self._actor.eval()

    # ----- properties --------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def motion_fps(self) -> float:
        return self._motion_fps

    # ----- public API --------------------------------------------------------

    def label_trajectory(
        self,
        body_trajectory: np.ndarray,
        *,
        root_rot_xyzw: Optional[np.ndarray] = None,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, LabelerStats]:
        """Encode a recorded body trajectory into per-frame motion tokens.

        Args:
            body_trajectory: ``(T, 31)`` float array in MuJoCo joint order.
                Must match the layout :func:`compose_body_trajectory`
                produces (legs/waist/head -> stand pose, arms -> recorded
                arm trajectory).
            root_rot_xyzw: Optional ``(T, 4)`` per-frame base quaternion in
                xyzw (scipy) format. Defaults to identity for every frame
                because the synthetic X2 dataset pins the pelvis in the
                world frame.
            return_stats: If True, also return a :class:`LabelerStats`
                summary alongside the tokens.

        Returns:
            ``(T, 64)`` float64 array of FSQ-quantized motion tokens. When
            ``return_stats=True``, returns ``(tokens, stats)``.
        """
        body = np.asarray(body_trajectory, dtype=np.float64)
        if body.ndim != 2 or body.shape[1] != 31:
            raise ValueError(
                f"body_trajectory must be (T, 31) in MuJoCo joint order; "
                f"got shape {body.shape}."
            )
        T = body.shape[0]

        if root_rot_xyzw is None:
            root_rot = np.tile(
                np.asarray(IDENTITY_QUAT_XYZW, dtype=np.float64), (T, 1)
            )
        else:
            root_rot = np.asarray(root_rot_xyzw, dtype=np.float64)
            if root_rot.shape != (T, 4):
                raise ValueError(
                    f"root_rot_xyzw must be (T, 4) xyzw; got shape "
                    f"{root_rot.shape}."
                )

        motion_data = {
            "x2_dataset_episode": {
                "dof": body,
                "root_rot": root_rot,
                "fps": self._motion_fps,
            }
        }

        # Pre-allocate the tokenizer-obs batch and fill it from the
        # canonical build_tokenizer_obs helper (single source of truth
        # shared with eval_x2_mujoco / the deploy harness).
        obs_batch = np.zeros((T, TOKENIZER_OBS_DIM), dtype=np.float32)
        for f in range(T):
            cur_quat_xyzw = root_rot[f]
            base_quat_wxyz = np.array(
                [cur_quat_xyzw[3], cur_quat_xyzw[0], cur_quat_xyzw[1], cur_quat_xyzw[2]],
                dtype=np.float64,
            )
            obs_batch[f] = build_tokenizer_obs(
                motion_data,
                current_time=f / self._motion_fps,
                base_quat_wxyz=base_quat_wxyz,
                motion_fps=self._motion_fps,
            )

        # Encode -> reshape -> FSQ -> flatten, all in one batched torch op.
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_batch).to(self._device)
            latent = self._actor.encoder(obs_t)
            latent = latent.view(
                T,
                UniversalTokenActor.MAX_NUM_TOKENS,
                UniversalTokenActor.TOKEN_DIM,
            )
            quantized = fsq_quantize(
                latent, levels=UniversalTokenActor.FSQ_LEVELS
            )
            tokens_t = quantized.view(T, -1)
            tokens = tokens_t.cpu().numpy().astype(np.float64, copy=False)

        if return_stats:
            stats = LabelerStats(
                num_frames=T,
                mean_l2=float(np.linalg.norm(tokens, axis=-1).mean()),
                max_abs=float(np.abs(tokens).max()),
                unique_levels=int(np.unique(np.round(tokens, 6)).size),
            )
            return tokens, stats
        return tokens

    # ----- internal helpers --------------------------------------------------

    def _load_actor(
        self, checkpoint_path: str | Path, *, device: str
    ) -> UniversalTokenActor:
        ckpt = _tolerant_torch_load(checkpoint_path, map_location=device)
        sd = ckpt.get("policy_state_dict") or ckpt.get("actor_model_state_dict")
        if sd is None:
            raise KeyError(
                "Cannot find 'policy_state_dict' or 'actor_model_state_dict' "
                f"in checkpoint at {checkpoint_path}."
            )
        actor = UniversalTokenActor()
        new_sd: dict[str, torch.Tensor] = {}
        encoder_keys_seen = 0
        for k, v in sd.items():
            if k == "std":
                new_sd["std"] = v
            elif k.startswith(self._ENCODER_KEY_PREFIX):
                new_sd[
                    k.replace(self._ENCODER_KEY_PREFIX, "encoder.module.")
                ] = v
                encoder_keys_seen += 1
            elif k.startswith("actor_module.decoders.g1_dyn.module."):
                new_sd[
                    k.replace(
                        "actor_module.decoders.g1_dyn.module.",
                        "decoder.module.",
                    )
                ] = v
        if encoder_keys_seen == 0:
            raise KeyError(
                "Checkpoint does not contain any keys under "
                f"'{self._ENCODER_KEY_PREFIX}*'. Is this a SONIC g1 "
                f"checkpoint? Path: {checkpoint_path}"
            )
        actor.load_state_dict(new_sd)
        return actor.to(device)


__all__ = [
    "DEFAULT_MOTION_FPS",
    "IDENTITY_QUAT_WXYZ",
    "IDENTITY_QUAT_XYZW",
    "LabelerStats",
    "SONIC_MOTION_TOKEN_DIM",
    "SonicMotionTokenLabeler",
    "TOKENIZER_OBS_DIM",
]
