"""X2 encoder-observation builder (G1-style YAML + registry).

What this module does
---------------------

Mirrors G1's deploy-side ``ObservationRegistry`` (see
``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/observation.hpp``)
in Python so the recorder can build the *exact* 680-D ``encoder_input``
the X2 SONIC encoder was trained on, *driven by a YAML config* instead
of a hard-coded layout. The YAML lives at
``gear_sonic/data/encoder/x2_observation_config.yaml`` and is consumed
unchanged by both the recorder (this file) and the validation tooling
(``compare_recorder_vs_deploy_obs.py``).

Why a registry
--------------

Today X2 ships exactly one observation
(``x2_command_multi_future_nonflat``, the 680-D 10-frame future window
of retargeted body_q + 6D rotation diff). G1 ships fourteen, of which
its three ``encoder_modes`` (``g1`` / ``teleop`` / ``smpl``) select
disjoint subsets. Structuring X2 the same way -- a registry mapping
observation name to a ``gather`` function plus a YAML that selects
which observations the encoder consumes -- gives us:

* a clean extension point for future X2 modalities (e.g. SMPL human
  pose, VR-only sparse points) without touching the recorder code
  path,
* parity with the G1 release wording so the artifact docs read the
  same across both robots,
* a single chokepoint the validator (Layer 3 in the plan) can target
  to assert byte-parity between recorder Python and deploy C++.

Reference layout (685 -> 680 contract)
--------------------------------------

The single registered observation,
``x2_command_multi_future_nonflat``, produces what
``gear_sonic.scripts.eval_x2_mujoco.build_tokenizer_obs`` produces:

* ``command_flat``: ``cat([all_jpos_flat, all_jvel_flat])`` -> 620
  (10 frames x (31 jpos + 31 jvel))
* reshape to ``(10, 62)`` -> ``command_nonflat``
* concat along last axis with ``ori_nonflat`` (10 frames x 6) ->
  ``(10, 68)``
* flatten -> 680.

We deliberately implement the gather *from the planner snapshot*
(``snap`` in the recorder's subscribe-mode loop) rather
than re-deriving from a fake motion clip the way
:class:`~gear_sonic.scripts.sonic_motion_token_labeler.SonicMotionTokenLabeler`
does for offline labeling. The planner already publishes the *real*
9-frame future (``joint_pos_mj_future``, ``root_quat_xyzw_future``)
plus the current frame, so the gather just stacks the current frame
in front of the planner's future window and runs the same arithmetic
:func:`build_tokenizer_obs` runs.

Bringing the gather into Python guarantees that:

* the recorder produces the *same* 680-D obs the deploy's C++
  ``ZmqPoseInputSource`` would build from the same wire snapshot
  (Layer 3 byte-parity test),
* the FSQ token written to ``action.motion_token`` is the encoder's
  output on that exact obs (Layer 2 parity test against
  ``label_trajectory``).

YAML schema
-----------

::

    encoder:
      dimension: 64           # FSQ output (2 x 32)
      motion_fps: 50.0
      dt_future_ref: 0.1
      num_future_frames: 10
      encoder_observations:
        - name: x2_command_multi_future_nonflat
          enabled: true
      encoder_modes:
        - name: retargeted_body_q
          mode_id: 0
          required_observations:
            - x2_command_multi_future_nonflat

The builder validates that:

* every ``encoder_observation.name`` is in
  :data:`X2_OBSERVATION_REGISTRY` (unknown names raise loudly so a
  typo can't silently fall through),
* every ``encoder_mode.required_observations`` is a subset of the
  enabled ``encoder_observations`` (otherwise the runtime call would
  see an unset slot),
* the total dimension matches what the encoder ``.pt`` expects
  (``dimension`` is informational; the *input* dim is what we assert).

Public API
----------

::

    builder = X2EncoderObsBuilder.from_yaml(
        Path("gear_sonic/data/encoder/x2_observation_config.yaml")
    )
    obs_680 = builder.build_obs(snap, mode="retargeted_body_q")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import yaml

# ``build_tokenizer_obs`` lives in the eval script and pulls in the IL
# joint-order remap. Mirror the labeler's defensive sys.path hack so
# the recorder can import this module without the caller massaging
# sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "gear_sonic" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Constants (single source of truth -- mirrors eval_x2_mujoco)
# ---------------------------------------------------------------------------


X2_NUM_BODY_DOFS: int = 31
"""X2 MuJoCo joint count (legs + waist + arms + head)."""

X2_NUM_FUTURE_FRAMES: int = 10
"""Future-window length the SONIC encoder was trained on."""

X2_DT_FUTURE_REF: float = 0.1
"""Future-frame spacing in seconds (10 frames -> 0.9 s lookahead)."""

X2_FEATURES_PER_FRAME: int = 68
"""31 jpos + 31 jvel + 6 ori = 68."""

X2_ENCODER_OBS_DIM: int = X2_NUM_FUTURE_FRAMES * X2_FEATURES_PER_FRAME  # 680
"""Total encoder-input dim (the SONIC g1 encoder's first layer in_features)."""


# ---------------------------------------------------------------------------
# Gather functions
# ---------------------------------------------------------------------------


def gather_x2_command_multi_future_nonflat(
    snap: Mapping[str, Optional[np.ndarray]],
    *,
    motion_fps: float,
    num_future_frames: int = X2_NUM_FUTURE_FRAMES,
) -> np.ndarray:
    """Build the 680-D ``encoder_input`` from a planner snapshot.

    Mirrors :func:`gear_sonic.scripts.eval_x2_mujoco.build_tokenizer_obs`
    bit-for-bit, but reads its inputs from the planner snapshot
    (``snap`` in
    the recorder's subscribe-mode loop)
    instead of an in-memory motion-lib clip.

    The planner publishes:

    * ``snap["body_pose_q_mj"]`` -- ``(31,)`` current commanded body_q
      (operator intent, post-arm-overlay).
    * ``snap["root_quat_xyzw"]`` -- ``(4,)`` current root quaternion
      in xyzw (scipy) order.
    * ``snap["joint_pos_mj_future"]`` -- ``(F, 31)`` planner's
      future-window joint targets, F = ``num_future_frames - 1`` = 9
      today (the planner stride matches ``DT_FUTURE_REF``).
    * ``snap["root_quat_xyzw_future"]`` -- ``(F, 4)`` matching root
      quaternions for the future frames.

    We stack ``[current, *future]`` to get a virtual 10-frame motion
    clip, then run the exact arithmetic
    :func:`~gear_sonic.scripts.eval_x2_mujoco.build_tokenizer_obs`
    runs at ``current_time=0``. The result is byte-identical to what
    the deploy's C++ ``ZmqPoseInputSource`` would emit from the same
    wire snapshot (asserted by Layer 3 of the plan's validation
    pyramid).

    Args:
        snap: Planner snapshot dict (see
            the recorder's subscribe-mode state
            ``.snapshot()``).
        motion_fps: Frame rate the planner runs at (50 Hz).
        num_future_frames: Future-window length. Defaults to 10 (the
            value the SONIC encoder was trained on); keep this fixed
            unless you're re-training the encoder.

    Returns:
        ``(680,)`` float32 ``encoder_input`` in the layout
        ``cat([command_nonflat (10, 62), ori_nonflat (10, 6)],
        axis=-1).reshape(-1)``.

    Raises:
        ValueError: if any snapshot field is missing or has the wrong
            shape (the recorder is expected to gate on this *before*
            calling the builder; this defensive check is for tests
            and Layer 3 tooling).
    """
    body_pose_cur = snap.get("body_pose_q_mj")
    root_quat_cur = snap.get("root_quat_xyzw")
    body_pose_fut = snap.get("joint_pos_mj_future")
    root_quat_fut = snap.get("root_quat_xyzw_future")

    if body_pose_cur is None:
        raise ValueError("snap['body_pose_q_mj'] is None")
    if root_quat_cur is None:
        raise ValueError("snap['root_quat_xyzw'] is None")
    if body_pose_fut is None:
        raise ValueError("snap['joint_pos_mj_future'] is None")
    if root_quat_fut is None:
        raise ValueError("snap['root_quat_xyzw_future'] is None")

    body_pose_cur = np.asarray(body_pose_cur, dtype=np.float64).reshape(-1)
    root_quat_cur = np.asarray(root_quat_cur, dtype=np.float64).reshape(-1)
    body_pose_fut = np.asarray(body_pose_fut, dtype=np.float64)
    root_quat_fut = np.asarray(root_quat_fut, dtype=np.float64)

    if body_pose_cur.shape[0] != X2_NUM_BODY_DOFS:
        raise ValueError(
            f"snap['body_pose_q_mj'] must be ({X2_NUM_BODY_DOFS},); "
            f"got {body_pose_cur.shape}"
        )
    if root_quat_cur.shape[0] != 4:
        raise ValueError(
            f"snap['root_quat_xyzw'] must be (4,); got {root_quat_cur.shape}"
        )
    if (
        body_pose_fut.ndim != 2
        or body_pose_fut.shape[1] != X2_NUM_BODY_DOFS
    ):
        raise ValueError(
            f"snap['joint_pos_mj_future'] must be (F, {X2_NUM_BODY_DOFS}); "
            f"got {body_pose_fut.shape}"
        )
    if root_quat_fut.shape != (body_pose_fut.shape[0], 4):
        raise ValueError(
            f"snap['root_quat_xyzw_future'] must be ({body_pose_fut.shape[0]},"
            f" 4); got {root_quat_fut.shape}"
        )

    # Stack current + future into a virtual 10-frame motion clip.
    # ``build_tokenizer_obs`` reads frames at ``current_time + f *
    # DT_FUTURE_REF``; with ``current_time=0`` and motion_fps that
    # matches DT_FUTURE_REF^-1's quantization (planner runs at 50 Hz =
    # 5 ticks per future step) it lands on the natural indices.
    body_clip = np.concatenate(
        [body_pose_cur[None, :], body_pose_fut], axis=0
    )
    root_clip = np.concatenate(
        [root_quat_cur[None, :], root_quat_fut], axis=0
    )

    # Pad if planner provides fewer than num_future_frames
    # (defensive -- the planner ships exactly 9 future frames today
    # and num_future_frames is 10, so the stack above already lands
    # at exactly 10. If a future planner ships fewer, repeat the
    # last frame -- mirrors build_tokenizer_obs's
    # ``min(int(future_time/dt), total_frames-1)`` index clamp).
    while body_clip.shape[0] < num_future_frames:
        body_clip = np.concatenate([body_clip, body_clip[-1:]], axis=0)
        root_clip = np.concatenate([root_clip, root_clip[-1:]], axis=0)

    # Defer to the canonical builder. We rebuild the motion_data dict
    # the labeler / eval script expect, then call
    # ``build_tokenizer_obs`` with the current frame's quat in WXYZ.
    from eval_x2_mujoco import build_tokenizer_obs  # noqa: E402

    cur_quat_xyzw = root_clip[0]
    base_quat_wxyz = np.array(
        [cur_quat_xyzw[3], cur_quat_xyzw[0], cur_quat_xyzw[1], cur_quat_xyzw[2]],
        dtype=np.float64,
    )
    motion_data = {
        "x2_recorder_planner_snapshot": {
            "dof": body_clip,
            "root_rot": root_clip,
            "fps": float(motion_fps),
        }
    }
    obs = build_tokenizer_obs(
        motion_data,
        current_time=0.0,
        base_quat_wxyz=base_quat_wxyz,
        motion_fps=float(motion_fps),
    )
    if obs.shape != (X2_ENCODER_OBS_DIM,):
        raise RuntimeError(
            f"build_tokenizer_obs returned shape {obs.shape}; "
            f"expected ({X2_ENCODER_OBS_DIM},). Did "
            f"NUM_FUTURE_FRAMES drift in eval_x2_mujoco?"
        )
    return obs


GatherFn = Callable[..., np.ndarray]
"""Signature: ``(snap, *, motion_fps, num_future_frames) -> np.ndarray``."""


X2_OBSERVATION_REGISTRY: Dict[str, GatherFn] = {
    "x2_command_multi_future_nonflat": gather_x2_command_multi_future_nonflat,
}
"""Name -> gather function. Add new modalities here (e.g. SMPL human
pose) and reference them from the YAML config; no recorder change
required."""


# ---------------------------------------------------------------------------
# YAML loader + builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderModeSpec:
    """One encoder-mode entry from the YAML."""

    name: str
    mode_id: int
    required_observations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class X2EncoderConfig:
    """Parsed X2 encoder YAML."""

    dimension: int
    motion_fps: float
    dt_future_ref: float
    num_future_frames: int
    encoder_observations: List[str]
    encoder_modes: List[EncoderModeSpec]

    @classmethod
    def from_yaml(cls, path: Path) -> "X2EncoderConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"X2 encoder config YAML not found at {path}. "
                "The default ships at "
                "gear_sonic/data/encoder/x2_observation_config.yaml; "
                "either restore it from git or pass --encoder-config "
                "to record_x2_dataset.py to point at your own."
            )
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict) or "encoder" not in raw:
            raise ValueError(
                f"{path}: top-level YAML must contain an 'encoder:' key "
                "(see gear_sonic/data/encoder/x2_observation_config.yaml "
                "for the canonical layout)."
            )
        enc = raw["encoder"]

        obs_entries = enc.get("encoder_observations") or []
        observations: List[str] = []
        for entry in obs_entries:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(
                    f"{path}: encoder_observations entries must be "
                    f"mappings with a 'name' key; got {entry!r}"
                )
            if entry.get("enabled", True):
                observations.append(str(entry["name"]))

        mode_entries = enc.get("encoder_modes") or []
        modes: List[EncoderModeSpec] = []
        for entry in mode_entries:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(
                    f"{path}: encoder_modes entries must be mappings "
                    f"with a 'name' key; got {entry!r}"
                )
            modes.append(
                EncoderModeSpec(
                    name=str(entry["name"]),
                    mode_id=int(entry.get("mode_id", 0)),
                    required_observations=[
                        str(n) for n in entry.get("required_observations", [])
                    ],
                )
            )

        return cls(
            dimension=int(enc.get("dimension", 64)),
            motion_fps=float(enc.get("motion_fps", 50.0)),
            dt_future_ref=float(enc.get("dt_future_ref", X2_DT_FUTURE_REF)),
            num_future_frames=int(
                enc.get("num_future_frames", X2_NUM_FUTURE_FRAMES)
            ),
            encoder_observations=observations,
            encoder_modes=modes,
        )


class X2EncoderObsBuilder:
    """Build the SONIC encoder's input from a planner snapshot.

    Loads an X2 encoder YAML (see
    ``gear_sonic/data/encoder/x2_observation_config.yaml``), validates
    every named observation against :data:`X2_OBSERVATION_REGISTRY`,
    then dispatches per-tick to the gather functions.

    For X2's single-modality release (``retargeted_body_q``) the
    builder is a thin one-shot dispatch -- it concatenates exactly
    one gather output. The class is structured this way so a future
    multi-modal X2 release (e.g. SMPL human pose + retargeted body_q)
    can drop in additional registry entries and YAML observations
    without changing the recorder.
    """

    def __init__(self, config: X2EncoderConfig) -> None:
        self._config = config
        # Eagerly validate every observation name -- a typo in the YAML
        # should fail at startup, never silently fall through to a
        # zero-filled token.
        for name in config.encoder_observations:
            if name not in X2_OBSERVATION_REGISTRY:
                raise KeyError(
                    f"encoder observation {name!r} is not in "
                    f"X2_OBSERVATION_REGISTRY. Known observations: "
                    f"{sorted(X2_OBSERVATION_REGISTRY)}. Either fix "
                    "the YAML or add a gather function in "
                    "gear_sonic/utils/teleop/x2_encoder_obs_builder.py."
                )
        for mode in config.encoder_modes:
            for req in mode.required_observations:
                if req not in config.encoder_observations:
                    raise ValueError(
                        f"encoder_mode {mode.name!r} requires observation "
                        f"{req!r} but it is not in encoder_observations. "
                        "Add it to the YAML's encoder_observations list "
                        "(with enabled: true) before this mode can be used."
                    )

    # ----- factory ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> "X2EncoderObsBuilder":
        return cls(X2EncoderConfig.from_yaml(path))

    # ----- properties -------------------------------------------------------

    @property
    def config(self) -> X2EncoderConfig:
        return self._config

    @property
    def encoder_observations(self) -> List[str]:
        return list(self._config.encoder_observations)

    @property
    def encoder_modes(self) -> List[EncoderModeSpec]:
        return list(self._config.encoder_modes)

    @property
    def motion_fps(self) -> float:
        return self._config.motion_fps

    @property
    def num_future_frames(self) -> int:
        return self._config.num_future_frames

    # ----- runtime ----------------------------------------------------------

    def build_obs(
        self,
        snap: Mapping[str, Optional[np.ndarray]],
        *,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Build the encoder input for one tick.

        Args:
            snap: Planner snapshot (see
                :meth:`_SubscribeModeState.snapshot`).
            mode: Optional encoder-mode name. When given, only the
                observations listed in that mode's
                ``required_observations`` are gathered (others are
                zero-filled to preserve layout). When ``None`` (the
                default), every enabled observation is gathered. For
                X2's single-modality release the result is identical
                either way.

        Returns:
            ``(dim,)`` float32 array, where ``dim`` is the sum of the
            per-observation gather outputs (today: 680).
        """
        if mode is not None:
            allowed: Optional[set[str]] = None
            for spec in self._config.encoder_modes:
                if spec.name == mode:
                    allowed = set(spec.required_observations)
                    break
            if allowed is None:
                raise KeyError(
                    f"encoder mode {mode!r} not declared in YAML. Known: "
                    f"{[m.name for m in self._config.encoder_modes]}"
                )
        else:
            allowed = None

        parts: List[np.ndarray] = []
        for name in self._config.encoder_observations:
            gather = X2_OBSERVATION_REGISTRY[name]
            if allowed is not None and name not in allowed:
                # Zero-fill non-required observations so the layout is
                # preserved even when a mode skips them. Mirrors G1's
                # encoder_modes behaviour (see
                # ``ObservationRegistry::Get`` in the deploy).
                # Today X2 has only one observation, so this branch is
                # exercised only by tests.
                stub = gather(
                    self._noop_snapshot(),
                    motion_fps=self._config.motion_fps,
                    num_future_frames=self._config.num_future_frames,
                )
                parts.append(np.zeros_like(stub))
            else:
                parts.append(
                    gather(
                        snap,
                        motion_fps=self._config.motion_fps,
                        num_future_frames=self._config.num_future_frames,
                    )
                )

        return np.concatenate(parts).astype(np.float32, copy=False)

    # ----- internal ---------------------------------------------------------

    def _noop_snapshot(self) -> Dict[str, np.ndarray]:
        """Snapshot of zeros / identity used to size disabled obs slots."""
        F = max(self._config.num_future_frames - 1, 1)
        return {
            "body_pose_q_mj": np.zeros(X2_NUM_BODY_DOFS, dtype=np.float64),
            "root_quat_xyzw": np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float64
            ),
            "joint_pos_mj_future": np.zeros(
                (F, X2_NUM_BODY_DOFS), dtype=np.float64
            ),
            "root_quat_xyzw_future": np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                (F, 1),
            ),
        }


__all__ = [
    "EncoderModeSpec",
    "GatherFn",
    "X2EncoderConfig",
    "X2EncoderObsBuilder",
    "X2_DT_FUTURE_REF",
    "X2_ENCODER_OBS_DIM",
    "X2_FEATURES_PER_FRAME",
    "X2_NUM_BODY_DOFS",
    "X2_NUM_FUTURE_FRAMES",
    "X2_OBSERVATION_REGISTRY",
    "gather_x2_command_multi_future_nonflat",
]
