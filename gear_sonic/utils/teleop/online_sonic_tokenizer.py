"""Online wrapper around :class:`SonicMotionTokenLabeler` for closed-loop teleop.

Two encode paths
----------------

The wrapper exposes two production encode paths plus one deprecated
freeze-pose path:

1. :meth:`encode_with_snapshot` (recommended) -- consumes the planner
   snapshot dict directly, builds the *real* 680-D 10-frame future
   observation via :class:`X2EncoderObsBuilder`, and runs the SONIC
   ``g1`` encoder on that obs. This is the byte-for-byte semantic
   match for what the deploy actor's internal encoder consumes from
   the same wire snapshot. Use this everywhere in the recorder's
   subscribe-mode path.

2. :meth:`encode_from_obs` -- power-user entry point for tests and
   validation tooling that already have a 680-D ``encoder_input``
   on hand (e.g. dumped from the C++ deploy via ``--obs-dump``).
   Skips the gather step and runs the encoder verbatim.

3. :meth:`encode` (DEPRECATED freeze-pose) -- the legacy v0 path that
   tiles the current ``body_q`` 11 times to fake a future window. Kept
   for the recorder's direct-mode loop (Quest-driven, no planner)
   because no real future is available there. Fires a one-shot warning
   on first use so dataset operators know the resulting tokens are
   semantically *frozen* (won't anticipate motion -- the VLA will only
   ever learn "stay where you are" from these labels). Will be removed
   once the direct-mode loop also drives a planner snapshot.

Loading conventions
-------------------

The recommended factory is :meth:`from_checkpoint_with_config`, which
pairs the ``.pt`` checkpoint (encoder weights) with the YAML config
(observation layout). The plain :meth:`from_checkpoint` is preserved
for the deprecated direct-mode loop and for tests that don't care
about the gather config.

Usage::

    tokenizer = OnlineSonicTokenizer.from_checkpoint_with_config(
        checkpoint_path="/path/to/model_step_025000.pt",
        config_path=Path("gear_sonic/data/encoder/x2_observation_config.yaml"),
    )
    tok = tokenizer.encode_with_snapshot(snap)   # (64,) float64

The first call performs the (possibly slow) checkpoint load. Each
subsequent ``encode_with_snapshot`` is a single 680-D batched encoder
pass -- sub-millisecond on CPU, sub-100 microseconds on GPU.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping, Optional

import numpy as np
import torch

# Importing the labeler implicitly pulls in eval_x2_mujoco via its
# own sys.path manipulation. We do the same here defensively so the
# recorder script can use this module without adding extra paths.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gear_sonic.scripts.sonic_motion_token_labeler import (  # noqa: E402
    DEFAULT_MOTION_FPS,
    IDENTITY_QUAT_XYZW,
    SONIC_MOTION_TOKEN_DIM,
    TOKENIZER_OBS_DIM,
    SonicMotionTokenLabeler,
)

from gear_sonic.utils.teleop.x2_encoder_obs_builder import (  # noqa: E402
    X2_ENCODER_OBS_DIM,
    X2EncoderObsBuilder,
)


# Length of the freeze-pose virtual clip used by the deprecated
# :meth:`encode` path. ``build_tokenizer_obs`` only reads frames
# 0..NUM_FUTURE_FRAMES-1 (= 10 future frames at 0.1 s spacing), so 11
# frames is enough headroom without being wasteful. The velocity
# computation reads ``f - 1`` and we want it to land at index 0 with
# zero velocity rather than borrowing a non-existent previous frame.
_VIRTUAL_CLIP_LEN: int = 11


_FREEZE_POSE_DEPRECATION_MSG: str = (
    "[OnlineSonicTokenizer] DEPRECATED encode() called -- this is the "
    "legacy v0 freeze-pose path: the current body_q is tiled across "
    "every future frame, so the resulting motion_token represents "
    "'stay still at this pose' rather than the operator's anticipated "
    "trajectory. The VLA will learn from these labels to predict only "
    "static poses. Switch to encode_with_snapshot(snap) (recorder "
    "subscribe mode, planner-driven) to feed the encoder the real "
    "10-frame future window. This warning fires once per process."
)


class OnlineSonicTokenizer:
    """Per-tick observation -> 64-D motion-token encoder.

    Wraps :class:`SonicMotionTokenLabeler` so the recorder can encode
    one tick's worth of observation into the FSQ-quantized motion
    token the VLA learns to predict.
    """

    def __init__(
        self,
        labeler: SonicMotionTokenLabeler,
        *,
        obs_builder: Optional[X2EncoderObsBuilder] = None,
        clip_len: int = _VIRTUAL_CLIP_LEN,
    ) -> None:
        self._labeler = labeler
        self._obs_builder = obs_builder
        self._clip_len = int(clip_len)
        if self._clip_len < 2:
            raise ValueError("clip_len must be >= 2")
        # One-shot deprecation flag for ``encode()``.
        self._freeze_pose_warned: bool = False

    # ----- factories --------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        motion_fps: float = DEFAULT_MOTION_FPS,
        clip_len: int = _VIRTUAL_CLIP_LEN,
    ) -> "OnlineSonicTokenizer":
        """Construct without an observation builder.

        The instance can still call :meth:`encode_from_obs` (it doesn't
        need the YAML), but :meth:`encode_with_snapshot` will raise
        because there's no gather config. Used by the deprecated
        direct-mode recorder loop and by tests that operate on
        pre-built obs.
        """
        labeler = SonicMotionTokenLabeler(
            checkpoint_path,
            device=device,
            motion_fps=motion_fps,
        )
        return cls(labeler, obs_builder=None, clip_len=clip_len)

    @classmethod
    def from_checkpoint_with_config(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path,
        *,
        device: str = "cpu",
        clip_len: int = _VIRTUAL_CLIP_LEN,
    ) -> "OnlineSonicTokenizer":
        """Recommended factory for the recorder.

        Loads the SONIC encoder weights from ``checkpoint_path`` (the
        ``.pt`` siblings of the deploy ONNX) and the gather config
        from ``config_path`` (typically
        ``gear_sonic/data/encoder/x2_observation_config.yaml``).
        ``motion_fps`` is taken from the YAML so the recorder, the
        gather, and the labeler all agree on the future-window stride.
        """
        builder = X2EncoderObsBuilder.from_yaml(Path(config_path))
        labeler = SonicMotionTokenLabeler(
            checkpoint_path,
            device=device,
            motion_fps=builder.motion_fps,
        )
        return cls(labeler, obs_builder=builder, clip_len=clip_len)

    # ----- properties -------------------------------------------------------

    @property
    def labeler(self) -> SonicMotionTokenLabeler:
        return self._labeler

    @property
    def obs_builder(self) -> Optional[X2EncoderObsBuilder]:
        return self._obs_builder

    @property
    def device(self) -> str:
        return self._labeler.device

    @property
    def motion_fps(self) -> float:
        return self._labeler.motion_fps

    # ----- encode paths -----------------------------------------------------

    def encode_from_obs(self, obs_680d: np.ndarray) -> np.ndarray:
        """Encode a pre-built 680-D ``encoder_input`` into a 64-D token.

        Skips the gather step. Used by tests, validation tooling, and
        by :meth:`encode_with_snapshot` after it computes the obs.
        Mirrors :meth:`SonicMotionTokenLabeler.label_trajectory`'s
        ``encode -> reshape -> FSQ -> flatten`` pipeline for a single
        observation row.

        Args:
            obs_680d: ``(680,)`` or ``(1, 680)`` float array. Layout
                must match
                :func:`~gear_sonic.scripts.eval_x2_mujoco.build_tokenizer_obs`.

        Returns:
            ``(SONIC_MOTION_TOKEN_DIM,)`` float64 token vector that
            lies on the FSQ lattice.
        """
        from eval_x2_mujoco import (  # noqa: E402  (defer for sys.path)
            UniversalTokenActor,
            fsq_quantize,
        )

        obs = np.asarray(obs_680d, dtype=np.float32).reshape(-1)
        if obs.shape[0] != X2_ENCODER_OBS_DIM:
            raise ValueError(
                f"obs must be ({X2_ENCODER_OBS_DIM},) (TOKENIZER_OBS_DIM="
                f"{TOKENIZER_OBS_DIM}); got {obs.shape}"
            )

        with torch.no_grad():
            obs_t = torch.from_numpy(obs[None, :]).to(self._labeler.device)
            latent = self._labeler._actor.encoder(obs_t)
            latent = latent.view(
                1,
                UniversalTokenActor.MAX_NUM_TOKENS,
                UniversalTokenActor.TOKEN_DIM,
            )
            quantized = fsq_quantize(
                latent, levels=UniversalTokenActor.FSQ_LEVELS
            )
            token_t = quantized.view(1, -1)
            token = token_t.cpu().numpy()[0].astype(np.float64, copy=False)
        return token

    def encode_with_snapshot(
        self,
        snap: Mapping[str, Optional[np.ndarray]],
        *,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Encode a planner snapshot into a 64-D motion token.

        Builds the 680-D encoder input via the YAML-driven
        :class:`X2EncoderObsBuilder` (real planner future, not freeze
        pose) and runs the SONIC encoder + FSQ on it. This is the
        byte-for-byte semantic match for what the deploy actor's
        internal encoder consumes from the same wire snapshot.

        Args:
            snap: Planner snapshot (see
                :meth:`_SubscribeModeState.snapshot`). Must contain
                ``body_pose_q_mj``, ``root_quat_xyzw``,
                ``joint_pos_mj_future``, ``root_quat_xyzw_future``.
            mode: Optional encoder-mode name. Defaults to the YAML's
                first declared mode (today: ``retargeted_body_q``).
                Reserved for future multi-modal X2 releases.

        Raises:
            RuntimeError: if the tokenizer was constructed without an
                observation builder (use
                :meth:`from_checkpoint_with_config` instead).
        """
        if self._obs_builder is None:
            raise RuntimeError(
                "encode_with_snapshot requires an X2EncoderObsBuilder; "
                "construct via OnlineSonicTokenizer."
                "from_checkpoint_with_config(checkpoint_path, config_path)."
            )
        if mode is None and self._obs_builder.encoder_modes:
            mode = self._obs_builder.encoder_modes[0].name
        obs = self._obs_builder.build_obs(snap, mode=mode)
        return self.encode_from_obs(obs)

    def encode(
        self,
        body_q: np.ndarray,
        *,
        root_rot_xyzw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """DEPRECATED: encode the current ``body_q`` as a freeze-pose token.

        Tiles ``body_q`` across an 11-frame virtual clip and runs the
        labeler at frame 0 -- so ``build_tokenizer_obs`` sees "stay at
        this pose" for every future frame. Resulting tokens encode
        static intent and *will not* drive the SONIC decoder to
        anticipate operator motion.

        Used today by the recorder's direct-mode loop (Quest-driven,
        no planner snapshot to source a real future window from).
        Prefer :meth:`encode_with_snapshot` whenever a planner
        snapshot is available.

        Args:
            body_q: ``(31,)`` body joint vector in MuJoCo joint order.
            root_rot_xyzw: optional ``(4,)`` xyzw quaternion for the
                root. Defaults to identity.

        Returns:
            ``(SONIC_MOTION_TOKEN_DIM,)`` float64 token vector.
        """
        if not self._freeze_pose_warned:
            print(_FREEZE_POSE_DEPRECATION_MSG, flush=True)
            self._freeze_pose_warned = True

        body_q = np.asarray(body_q, dtype=np.float64).reshape(-1)
        if body_q.shape[0] != 31:
            raise ValueError(
                f"body_q must be (31,) MuJoCo joint order; got {body_q.shape}"
            )

        if root_rot_xyzw is None:
            root_xyzw = np.asarray(IDENTITY_QUAT_XYZW, dtype=np.float64)
        else:
            root_xyzw = np.asarray(root_rot_xyzw, dtype=np.float64).reshape(-1)
            if root_xyzw.shape[0] != 4:
                raise ValueError(
                    f"root_rot_xyzw must be (4,) xyzw; got {root_xyzw.shape}"
                )

        body_clip = np.tile(body_q, (self._clip_len, 1))
        root_clip = np.tile(root_xyzw, (self._clip_len, 1))

        tokens = self._labeler.label_trajectory(
            body_clip, root_rot_xyzw=root_clip
        )
        return tokens[0]

    def encode_with_horizon(
        self,
        body_q: np.ndarray,
        *,
        horizon: int,
        root_rot_xyzw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """DEPRECATED convenience: tile :meth:`encode` ``horizon`` times.

        Kept for the deploy's optional pose-chunk protocol consumers
        that expected a horizon of identical motion tokens. New
        callers should encode per-tick with
        :meth:`encode_with_snapshot` instead.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        token = self.encode(body_q, root_rot_xyzw=root_rot_xyzw)
        return np.tile(token, (horizon, 1)).astype(np.float64, copy=False)


__all__ = [
    "OnlineSonicTokenizer",
    "SONIC_MOTION_TOKEN_DIM",
    "X2_ENCODER_OBS_DIM",
]
