"""On-demand motion-clip playback for the X2 recorder.

A *motion clip* is a single PKL motion clip that the operator can
trigger mid-session via:

* :file:`gear_sonic/scripts/play_gesture.py` (in-place gestures --
  arm wave, sit-down, bow, ...). The PKL's authored heading is
  yaw-rebased onto the robot's current yaw so the clip starts from
  whatever direction the robot is facing.
* :file:`gear_sonic/scripts/play_locomotion.py` (walks / turns).
  The PKL's authored yaw evolution is preserved verbatim so a clip
  authored as "walk 1 m forward then pivot left" actually steers
  the body the way the mocap intended.

The :class:`X2DatasetRecorder` listens on the ``motion_clip_cmd``
ZMQ topic; on ``play``, it constructs a :class:`MotionClipSession`
from a catalog entry (or an ad-hoc :class:`MotionClipEntry`), swaps
its publish path to emit clip frames verbatim (bypassing the
kplanner body_pose stream and the manager arm/hand merge), and
snaps back to the kplanner once the session completes or a ``stop``
arrives.

The session is kind-tagged (``"gesture"`` / ``"locomotion"``) so the
recorder logs which side fired the play and so the yaw rebase is
gated cleanly (gesture: rebase frame 0 to robot yaw; locomotion:
keep the authored yaw evolution).

Wire contract
-------------

Topic ``motion_clip_cmd`` (default port
:data:`MOTION_CLIP_CMD_DEFAULT_PORT`), JSON UTF-8 payload::

    {"action": "play", "name": "<catalog-name>", "kind": "gesture"}
    {"action": "play", "pkl": "<path>", "kind": "locomotion"}
    {"action": "stop"}

``kind`` defaults to ``"gesture"`` if absent so existing call sites
that pre-date the locomotion split keep working. ``stop`` payload
may carry any extra fields; only ``action`` is required.

Catalog format
--------------

A YAML file (default
:file:`gear_sonic/data/motions/gestures/gestures_v1.yaml`) listing
named gestures. Locomotion clips do not use the catalog (operators
pass ``--pkl`` directly because the clip library is too large to
curate by hand). Catalog format::

    name: x2_ultra_gestures_v1
    gestures:
      - name: squat_A359
        source: gear_sonic/data/motions/x2_demo_bank.pkl
        motion_key: null      # first key in the PKL
        start_frame: 0        # optional, default 0
        n_frames: null        # optional, default null (to end of clip)

Root rebase
-----------

The recorder ``pose`` wire only carries ``joint_pos_mj`` (31,) and
``root_quat_xyzw`` (4,) -- no root translation. ``MotionClipSession``
yaw-rebases the PKL's root quaternion so frame 0 matches the
operator-supplied ``robot_root_yaw``. The recorder chooses that
value at play-time off the same x2_debug ``base_quat`` it uses for
the idle-stand pose (with the kplanner body_pose snap as a stale-
x2_debug fallback) so the takeover from idle to clip is
C0-continuous in yaw on tick zero -- whether or not a kplanner is
running upstream. Roll and pitch pass through verbatim, so a
sit-down motion genuinely lowers the pelvis. Joint angles are
frame-invariant and need no rebasing.

The rebase is one rigid ``Rz(dyaw)`` applied uniformly to every
frame, so:

* ``kind="gesture"``: frame 0's world-frame yaw lands on the robot's
  current heading. Authored yaw evolution across the clip is near-
  zero for in-place motions (wave / sit / bow) so the body
  basically stays put yaw-wise.
* ``kind="locomotion"``: frame 0's world-frame yaw also lands on
  the robot's current heading -- this kills the teleport-rotation
  jerk that would otherwise fire at takeover for a clip authored
  with frame-0 yaw != robot yaw. The single rigid offset preserves
  the *relative* yaw evolution exactly, so a walk-and-turn clip
  still turns by the authored amount in the robot's body frame.

The ``kind`` discriminator is kept for :attr:`hold_after` semantics
(gesture-only), recorder takeover log clarity, and future
divergences (e.g. translation handling for locomotion). Both kinds
share the rebase math today.

Port registry coordination
--------------------------

:data:`MOTION_CLIP_CMD_DEFAULT_PORT` lives here as a local literal.
Once the ZMQ port registry lands (separate plan), this module will
import ``MOTION_CLIP_CMD`` from
``gear_sonic.utils.teleop.zmq.port_registry`` and re-export the
same constant; both call sites continue to work because the
existing constant name is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np
import yaml

from gear_sonic.utils.planner.blending import (
    resample_motion_30_to_50hz,
    rotate_quats_yaw_only,
    yaw_of_quat_xyzw,
)


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""Resolves to the GR00T-WholeBodyControl checkout root, used to make
catalog ``source`` paths relative to the repo regardless of cwd."""

MOTION_CLIP_CMD_DEFAULT_PORT: int = 5568
"""Default ZMQ port for the ``motion_clip_cmd`` topic. Unused by any
other process in the production range (5550-5580) per the port
audit. TODO: migrate to port_registry.MOTION_CLIP_CMD.port once
that plan lands."""

MOTION_CLIP_CMD_DEFAULT_TOPIC: str = "motion_clip_cmd"
"""Default ZMQ topic name for motion-clip commands."""

GESTURE_DEFAULT_CATALOG_PATH: Path = (
    REPO_ROOT / "gear_sonic" / "data" / "motions" / "gestures" / "gestures_v1.yaml"
)
"""Default location of the gesture catalog YAML. Locomotion clips
don't have a catalog (operators pass ``--pkl`` directly)."""

X2_NUM_BODY_DOFS: int = 31
"""Mirror of :data:`gear_sonic.utils.teleop.x2_encoder_obs_builder.X2_NUM_BODY_DOFS`
duplicated here to avoid a circular import (the recorder imports this
module; the encoder_obs_builder is heavier and not always available)."""

MotionClipKind = Literal["gesture", "locomotion"]
"""Discriminator on the wire + on :class:`MotionClipEntry` / play
request. Gates the yaw rebase in :class:`MotionClipSession`."""

_VALID_KINDS: frozenset[str] = frozenset(("gesture", "locomotion"))


# ── Catalog entries ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MotionClipEntry:
    """One row in the motion-clip catalog (or an ad-hoc clip
    materialized from a ``--pkl`` play).

    ``source`` is resolved relative to :data:`REPO_ROOT` if it doesn't
    point at an absolute path. ``motion_key=None`` means "first key in
    the PKL", which is the common case for single-clip PKLs.
    ``start_frame`` / ``n_frames`` slice the PKL clip; ``n_frames=None``
    plays to the end.

    ``hold_after`` controls what happens after the recorder consumes
    the final clip frame:

    * ``False`` (default): snap-handback. The recorder clears the
      active clip and resumes forwarding kplanner ``body_pose``
      on the next tick (or idle stand if no upstream).
    * ``True``: the recorder enters a *held* state, republishing the
      clip's last frame each tick at the publish rate until an
      explicit ``stop`` arrives or a new ``play`` supersedes it.
      Designed for pose-and-park sequences (e.g. sit-from-stand
      that should keep the robot seated indefinitely while the
      operator goes do something else). Only meaningful for
      ``kind="gesture"``; locomotion clips end at an idle stand
      via the recorder's existing fallback, not a frozen pose.

    ``kind`` discriminates gesture (yaw-rebased to robot heading)
    from locomotion (authored yaw preserved). Defaults to
    ``"gesture"`` because the catalog ships gesture clips only;
    locomotion clips are constructed ad-hoc from ``--pkl`` plays.
    """

    name: str
    source: Path
    motion_key: Optional[str] = None
    start_frame: int = 0
    n_frames: Optional[int] = None
    hold_after: bool = False
    kind: MotionClipKind = "gesture"

    def resolved_source(self) -> Path:
        """Absolute path to the PKL, anchored at the repo root."""
        p = Path(self.source)
        if p.is_absolute():
            return p
        return REPO_ROOT / p


def load_catalog(path: Path) -> dict[str, MotionClipEntry]:
    """Parse a gesture catalog YAML into ``{name: MotionClipEntry}``.

    All entries loaded from the YAML are ``kind="gesture"`` because
    the catalog is gesture-only by design (locomotion uses ``--pkl``
    directly). Raises :class:`FileNotFoundError` if the file is
    missing, :class:`ValueError` if the schema is malformed
    (missing required fields, duplicate names, or empty gestures
    list).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"gesture catalog not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    gestures = raw.get("gestures")
    if not isinstance(gestures, list) or not gestures:
        raise ValueError(f"{path}: 'gestures' must be a non-empty list")

    out: dict[str, MotionClipEntry] = {}
    for i, item in enumerate(gestures):
        if not isinstance(item, dict):
            raise ValueError(
                f"{path}: gestures[{i}] must be a mapping, got {type(item).__name__}"
            )
        try:
            name = str(item["name"])
            source = Path(item["source"])
        except KeyError as exc:
            raise ValueError(
                f"{path}: gestures[{i}] missing required key {exc!s}"
            ) from None
        if name in out:
            raise ValueError(f"{path}: duplicate gesture name {name!r}")
        motion_key = item.get("motion_key")
        if motion_key is not None:
            motion_key = str(motion_key)
        start_frame = int(item.get("start_frame", 0))
        n_frames_raw = item.get("n_frames")
        n_frames = None if n_frames_raw is None else int(n_frames_raw)
        hold_after = bool(item.get("hold_after", False))
        out[name] = MotionClipEntry(
            name=name,
            source=source,
            motion_key=motion_key,
            start_frame=start_frame,
            n_frames=n_frames,
            hold_after=hold_after,
            kind="gesture",
        )
    return out


# ── PKL loading helpers ──────────────────────────────────────────────


def _pick_motion(pkl_data: dict, motion_key: Optional[str]) -> tuple[str, dict]:
    """Pick a motion entry from a multi-entry PKL.

    Mirrors :func:`gear_sonic.scripts.play_x2_motion_mujoco.load_motion`'s
    behaviour: explicit key uses exact match then unique substring
    match; ``None`` returns the first entry.
    """
    keys = list(pkl_data.keys())
    if not keys:
        raise RuntimeError("PKL has no motion entries")
    if motion_key is None:
        return keys[0], pkl_data[keys[0]]
    if motion_key in pkl_data:
        return motion_key, pkl_data[motion_key]
    matches = [k for k in keys if motion_key in k]
    if len(matches) == 1:
        return matches[0], pkl_data[matches[0]]
    preview = ", ".join(keys[:6]) + (" ..." if len(keys) > 6 else "")
    raise KeyError(
        f"motion key {motion_key!r} not found (or ambiguous); "
        f"have {len(keys)} entries, first few: {preview}"
    )


def _load_pkl_arrays(
    entry: MotionClipEntry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Load (dof, root_rot_xyzw, root_trans_offset, fps, resolved_key)
    from the PKL referenced by ``entry``, sliced by start_frame/n_frames.

    Validates shapes against :data:`X2_NUM_BODY_DOFS` and requires at
    least 2 frames after slicing (the resampler's lower bound).
    """
    src = entry.resolved_source()
    if not src.is_file():
        raise FileNotFoundError(f"motion clip PKL not found: {src}")
    data = joblib.load(src)
    if not isinstance(data, dict):
        raise ValueError(f"{src}: expected dict-of-motions, got {type(data).__name__}")
    key, m = _pick_motion(data, entry.motion_key)

    dof = np.asarray(m["dof"], dtype=np.float32)
    rot = np.asarray(m["root_rot"], dtype=np.float32)  # xyzw (scipy convention)
    trans = np.asarray(m["root_trans_offset"], dtype=np.float64)
    fps = float(m["fps"])

    if dof.ndim != 2 or dof.shape[1] != X2_NUM_BODY_DOFS:
        raise ValueError(
            f"{src}[{key!r}]: dof must be (T, {X2_NUM_BODY_DOFS}); "
            f"got {dof.shape}"
        )
    if rot.shape != (dof.shape[0], 4):
        raise ValueError(
            f"{src}[{key!r}]: root_rot must be (T, 4) xyzw matching dof; "
            f"got {rot.shape}"
        )
    if trans.shape != (dof.shape[0], 3):
        raise ValueError(
            f"{src}[{key!r}]: root_trans_offset must be (T, 3); "
            f"got {trans.shape}"
        )

    start = max(0, int(entry.start_frame))
    if start >= dof.shape[0]:
        raise ValueError(
            f"{src}[{key!r}]: start_frame={start} >= clip length "
            f"{dof.shape[0]}"
        )
    end = (
        dof.shape[0]
        if entry.n_frames is None
        else min(dof.shape[0], start + int(entry.n_frames))
    )
    if end - start < 2:
        raise ValueError(
            f"{src}[{key!r}]: sliced length must be >= 2 frames; "
            f"got start={start} end={end}"
        )

    return dof[start:end], rot[start:end], trans[start:end], fps, key


def estimate_duration_s(
    entry: MotionClipEntry,
    target_rate_hz: float,
) -> float:
    """Return the clip's resampled duration in seconds.

    Used by :file:`gear_sonic/scripts/play_gesture.py` and
    :file:`gear_sonic/scripts/play_locomotion.py` to size the
    SIGINT-blockable sleep so the script exits naturally when the
    recorder finishes the clip. Loads the PKL fully (joblib has no
    cheap header-only path), which is fast for single-entry clips
    (<5 ms typical) but linear in clip length for monolithic PKLs.
    """
    dof, _rot, _trans, src_fps, _key = _load_pkl_arrays(entry)
    duration_s = dof.shape[0] / float(src_fps)
    # Match the resampler's floor-based output length so we don't
    # over- or under-estimate when the rate ratio isn't integer.
    n_resampled = int(np.floor(duration_s * float(target_rate_hz)))
    if n_resampled < 2:
        n_resampled = dof.shape[0]
    return float(n_resampled) / float(target_rate_hz)


# ── Session ──────────────────────────────────────────────────────────


@dataclass
class MotionClipSession:
    """Per-play state for one motion-clip invocation.

    Construction loads + resamples + yaw-rebases the clip;
    :meth:`next_frame` walks one tick at a time;
    :meth:`future_window` returns the strictly-future window the C++
    deploy's tokenizer wants (matches the kplanner's convention --
    see :func:`gear_sonic.utils.planner.state_machine.build_pose_payload`).
    The session is consumed sequentially; ``next_frame`` is the only
    mutator and ``is_done`` flips True after the final frame is
    returned.

    Yaw rebase per :attr:`MotionClipEntry.kind`:

    * ``"gesture"``: frame 0's yaw is aligned with
      ``robot_root_yaw_rad``; roll / pitch pass through. Authored
      yaw evolution is near-zero for in-place clips so the body
      stays put yaw-wise.
    * ``"locomotion"``: frame 0's yaw is also aligned with
      ``robot_root_yaw_rad`` (single rigid ``Rz(dyaw)`` applied to
      every frame), so the clip starts from the robot's current
      heading instead of teleport-rotating to the mocap's authored
      world frame. Relative yaw evolution across frames is
      preserved bit-for-bit -- a walk-and-turn clip still turns
      by the authored amount in the robot's body frame.

    The XY translation in the PKL is loaded by ``_load_pkl_arrays``
    but never published (the recorder ``pose`` wire is
    position-free, and so is the C++ ref-motion path). The
    session keeps it only because the resampler needs an aligned
    trans array.
    """

    entry: MotionClipEntry
    target_rate_hz: float
    robot_root_yaw_rad: float
    future_dt_s: float = 0.1
    motion_key_resolved: str = field(init=False)
    body_q_mj: np.ndarray = field(init=False)
    root_quat_xyzw: np.ndarray = field(init=False)
    n_frames: int = field(init=False)
    _idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        dof_src, rot_src, trans_src, src_fps, key = _load_pkl_arrays(self.entry)
        self.motion_key_resolved = key

        # Resample to target rate. The helper handles src_fps==target
        # by returning short identity output; we always go through it
        # to keep the path uniform.
        if abs(src_fps - self.target_rate_hz) < 1e-3:
            dof_out = dof_src.copy()
            rot_out = rot_src.copy()
        else:
            dof_out, rot_out, _trans_out = resample_motion_30_to_50hz(
                dof_src.astype(np.float32),
                rot_src.astype(np.float32),
                trans_src.astype(np.float64),
                src_fps=float(src_fps),
                target_fps=float(self.target_rate_hz),
            )

        if self.entry.kind == "gesture":
            # Yaw-only rebase: align PKL frame-0 yaw with the operator-
            # supplied robot yaw. Z and roll/pitch pass through
            # unchanged so a sit-down genuinely lowers the pelvis.
            yaw_pkl_0 = yaw_of_quat_xyzw(rot_out[0].astype(np.float64))
            dyaw = float(self.robot_root_yaw_rad) - yaw_pkl_0
            rot_final = rotate_quats_yaw_only(
                rot_out.astype(np.float64), dyaw
            ).astype(np.float32)
        else:
            # Locomotion: same single rigid Rz(dyaw) as gesture, but
            # scoped to its own branch so future divergences (e.g.
            # translation handling, future-window strategy) attach
            # here without touching the gesture path. Frame 0's yaw
            # lands on the robot's current heading -- this kills the
            # teleport-rotation jerk that would otherwise hit the
            # body when the recorder switches from idle-stand (which
            # is rebased to the live x2_debug base_quat) to a clip
            # authored with frame-0 yaw != robot yaw. Relative yaw
            # evolution across frames is preserved by the single-
            # offset construction: rotate_quats_yaw_only is just a
            # left-multiply by Rz(dyaw), so per-frame Rz(yaw_k) -
            # Rz(yaw_0) deltas (i.e. the actual turn pattern) are
            # untouched.
            yaw_pkl_0 = yaw_of_quat_xyzw(rot_out[0].astype(np.float64))
            dyaw = float(self.robot_root_yaw_rad) - yaw_pkl_0
            rot_final = rotate_quats_yaw_only(
                rot_out.astype(np.float64), dyaw
            ).astype(np.float32)

        self.body_q_mj = dof_out.astype(np.float32)
        self.root_quat_xyzw = rot_final
        self.n_frames = int(self.body_q_mj.shape[0])

    @property
    def duration_s(self) -> float:
        """Total clip duration at the target publish rate."""
        return float(self.n_frames) / float(self.target_rate_hz)

    @property
    def current_index(self) -> int:
        """Frame index of the *next* :meth:`next_frame` call."""
        return int(self._idx)

    @property
    def kind(self) -> MotionClipKind:
        """Pass-through to :attr:`entry.kind` so the recorder can
        log + branch on it without poking ``session.entry`` directly."""
        return self.entry.kind

    def is_done(self) -> bool:
        """True once :meth:`next_frame` has consumed the last frame."""
        return self._idx >= self.n_frames

    def next_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(body_q_mj (31,), root_quat_xyzw (4,))`` and advance.

        Raises ``StopIteration`` if the session is already done.
        """
        if self.is_done():
            raise StopIteration(
                f"motion clip {self.entry.name!r} exhausted at frame "
                f"{self._idx}/{self.n_frames}"
            )
        body = self.body_q_mj[self._idx]
        rot = self.root_quat_xyzw[self._idx]
        self._idx += 1
        return body, rot

    def future_window(
        self, n_strict_future: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return strictly-future ``(joint_pos_mj_future, root_quat_xyzw_future)``.

        Length ``n_strict_future`` with ``future_dt_s`` spacing in
        ticks (clamped to >=1 tick). Pads with the final clip frame
        when we walk past the end. Matches the contract the recorder's
        ``_publish_pose`` expects: shape ``(n, 31)`` for jpos and
        ``(n, 4)`` for rot.
        """
        if n_strict_future <= 0:
            return (
                np.zeros((0, X2_NUM_BODY_DOFS), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
            )
        step = max(1, int(round(self.future_dt_s * self.target_rate_hz)))
        # We've already advanced past the current frame's index by the
        # time the publish loop calls future_window, so future[k] is
        # the frame at (current_index - 1) + (k + 1) * step. Clamp to
        # the last available frame to pad past clip end.
        anchor = max(0, self._idx - 1)
        last = self.n_frames - 1
        jpos_out = np.empty((n_strict_future, X2_NUM_BODY_DOFS), dtype=np.float32)
        rot_out = np.empty((n_strict_future, 4), dtype=np.float32)
        for k in range(n_strict_future):
            fi = min(anchor + (k + 1) * step, last)
            jpos_out[k] = self.body_q_mj[fi]
            rot_out[k] = self.root_quat_xyzw[fi]
        return jpos_out, rot_out


# ── Command parsing ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MotionClipPlayRequest:
    """Decoded ``play`` command. Exactly one of ``name`` or
    ``pkl_path`` is populated; the recorder resolves ``name`` against
    its loaded catalog and ``pkl_path`` into an ad-hoc
    :class:`MotionClipEntry`.

    ``kind`` discriminates gesture (yaw-rebased to robot heading)
    from locomotion (authored yaw preserved). Defaults to
    ``"gesture"`` if the wire payload omits the field so existing
    catalog-name plays from ``play_gesture`` keep working without a
    coordinated rollout. ``play_locomotion`` always stamps
    ``"locomotion"``.

    ``hold_after`` is the optional wire-level override of the catalog
    entry's :attr:`MotionClipEntry.hold_after` flag:

    * ``None``: defer to the catalog (or default ``False`` for
      ad-hoc ``--pkl`` plays). Set by ``play_gesture`` when neither
      ``--hold`` nor ``--no-hold`` is on the CLI -- the recorder
      resolves the effective behaviour locally so the wire payload
      stays minimal.
    * ``True`` / ``False``: explicit override, ignores the catalog.
      Only meaningful for ``kind="gesture"``; ``play_locomotion``
      leaves this as ``None`` (locomotion clips fall back to idle
      stand, not a held pose).
    """

    name: Optional[str] = None
    pkl_path: Optional[Path] = None
    motion_key: Optional[str] = None
    start_frame: int = 0
    n_frames: Optional[int] = None
    hold_after: Optional[bool] = None
    kind: MotionClipKind = "gesture"


@dataclass(frozen=True)
class MotionClipStopRequest:
    """Decoded ``stop`` command. No fields; the recorder just drops
    the active session."""


def parse_motion_clip_command(
    payload: dict[str, Any],
) -> MotionClipPlayRequest | MotionClipStopRequest:
    """Decode a ``motion_clip_cmd`` JSON payload.

    Raises :class:`ValueError` on malformed input. The recorder logs +
    swallows that error so a stray command never tears down the loop.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"motion_clip_cmd payload must be a JSON object, got "
            f"{type(payload).__name__}"
        )
    action = payload.get("action")
    if action == "stop":
        return MotionClipStopRequest()
    if action != "play":
        raise ValueError(f"motion_clip_cmd: unknown action {action!r}")
    name = payload.get("name")
    pkl = payload.get("pkl")
    if (name is None) == (pkl is None):
        raise ValueError(
            "motion_clip_cmd play: exactly one of 'name' or 'pkl' must be set"
        )
    hold_after_raw = payload.get("hold_after")
    hold_after: Optional[bool]
    if hold_after_raw is None:
        hold_after = None
    elif isinstance(hold_after_raw, bool):
        hold_after = hold_after_raw
    else:
        raise ValueError(
            f"motion_clip_cmd play: 'hold_after' must be bool or omitted, "
            f"got {type(hold_after_raw).__name__}"
        )
    kind_raw = payload.get("kind", "gesture")
    if kind_raw not in _VALID_KINDS:
        raise ValueError(
            f"motion_clip_cmd play: 'kind' must be one of "
            f"{sorted(_VALID_KINDS)!r}, got {kind_raw!r}"
        )
    return MotionClipPlayRequest(
        name=None if name is None else str(name),
        pkl_path=None if pkl is None else Path(pkl),
        motion_key=(
            None if payload.get("motion_key") is None
            else str(payload["motion_key"])
        ),
        start_frame=int(payload.get("start_frame", 0)),
        n_frames=(
            None if payload.get("n_frames") is None
            else int(payload["n_frames"])
        ),
        hold_after=hold_after,
        kind=kind_raw,  # type: ignore[arg-type]
    )


__all__ = [
    "GESTURE_DEFAULT_CATALOG_PATH",
    "MOTION_CLIP_CMD_DEFAULT_PORT",
    "MOTION_CLIP_CMD_DEFAULT_TOPIC",
    "MotionClipEntry",
    "MotionClipKind",
    "MotionClipPlayRequest",
    "MotionClipSession",
    "MotionClipStopRequest",
    "X2_NUM_BODY_DOFS",
    "estimate_duration_s",
    "load_catalog",
    "parse_motion_clip_command",
]
