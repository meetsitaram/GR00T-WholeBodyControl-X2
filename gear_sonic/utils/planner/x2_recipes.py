"""Recipe DSL for building **AgiBot X2 Ultra** heuristic-planner primitives.

This module is X2-specific (31-DOF MuJoCo joint order, X2 stand pose,
X2 leg geometry for ``synthesize_crouch_ramp``). Other embodiments need
their own recipe module mirroring this layout. Renamed from ``recipes.py``
to ``x2_recipes.py`` so the embodiment scope is visible at the import site.

A primitive is no longer just a window pointer into the source mocap library;
it is the output of a small, deterministic ops pipeline declared in
``x2_planner_primitives_recipes.yaml``. The runtime PKL the deploy loads is
the cached build artifact.

Why this exists (vs. the curator's "best of K" pick):

  - Some primitives have no good source clip (the static_upper_body family —
    feet-planted leans / twists). Synthesizing them from idle is more
    reliable than auditioning bad mocap candidates.
  - Bilateral symmetry should be free: review ``turn_left_45deg`` and the
    right-side counterpart auto-derives via L<->R mirror.
  - Magnitude variants (1ft / 0.5ft / 0.25ft step) shouldn't each need a
    separate mocap pick; a single base step + ``scale_magnitude`` covers them.
  - Stripping arms / head is per-bin policy. Recipes encode it explicitly.

This module is pure logic: numpy + dataclasses. The script
``gear_sonic/scripts/build_x2_planner_primitives.py`` handles source-PKL
I/O and writes the runtime ``x2_planner_primitives.pkl``.

Op reference
------------

Each op takes the running (dof, root_rot_xyzw, root_trans, fps) buffer plus
the source-clip lookup table, and returns a new buffer. Ops are applied in
order. The first op of a recipe must be a *producer* (clip_window or
synthesize_*); later ops are *transforms*.

  clip_window {motion_key, start_frame, n_frames}
      Take frames [s:s+n] of a source clip. Producer.

  synthesize_waist_ramp {axis, peak_deg, ramp_in_frames, hold_frames,
                         ramp_out_frames, fps?,
                         hip_pitch_share?, hip_yaw_share?,
                         ankle_pitch_share?, ankle_roll_share?}
      Build from DEFAULT_STAND_POSE: ramp the waist axis (pitch/yaw/roll)
      from 0 -> peak over ramp_in, hold for hold_frames, ramp back to 0
      over ramp_out. Root quat = identity, root XY = 0, Z = DEFAULT_PELVIS_Z.
      Optional counter-balance shares co-actuate the hips / ankles in
      proportion to the waist track so the synthesized pose stays closer
      to the natural human "lean" / "twist" pattern (defaults 0.0 keep
      the pure-waist behavior). |peak_deg| is HARD-CAPPED per axis
      (pitch 20, roll 10, yaw 40) -- a buggy recipe asking for an
      unstable angle fails at build time. Producer.

  synthesize_crouch_ramp {peak_drop_m, ramp_in_frames, hold_frames,
                          ramp_out_frames, fps?}
      Build from DEFAULT_STAND_POSE: ramp pelvis Z down by peak_drop_m
      and bend hips/knees/ankles (knee = 2*hip, ankle = -hip) so the
      resulting squat is geometrically self-consistent. Always feet
      planted (knee + ankle counter-rotate to keep foot flat). Producer.

  synthesize_side_step_ramp {peak_lateral_m, ramp_in_frames, hold_frames,
                             ramp_out_frames, knee_lift_rad?, fps?}
      Build from DEFAULT_STAND_POSE: a 4-phase shuffle that translates
      the pelvis laterally by peak_lateral_m using ONLY hip-roll
      abduction/adduction + mild knee/ankle lift -- NO hip_pitch (so
      no leg-behind scissor) and NO waist_yaw (so no body twist).
      Sign: +peak_lateral_m moves robot to its LEFT (+Y body); negative
      moves to its RIGHT. Producer.

  freeze {groups: [arms, legs, head, waist, waist_pitch, waist_yaw,
                   waist_roll, all_but_legs, ...]}
      Replace the listed joint group(s) with DEFAULT_STAND_POSE values for
      every frame. Idempotent; safe to apply twice.

  mirror_lr {also_negate_root_yaw?: bool, also_negate_root_y?: bool}
      Sagittal-plane reflection. Swaps L<->R joint indices, negates the
      anti-symmetric joints (hip/shoulder roll-yaw, wrist yaw-roll, waist
      yaw-roll, head yaw), mirrors the root quaternion (-qx, qy, -qz, qw)
      and the root_trans Y component. Both negate flags default to True.

  scale_magnitude {factor, scale_xy?: bool, scale_yaw?: bool,
                   scale_z?: bool}
      Re-scale joint deltas relative to DEFAULT_STAND_POSE by factor.
      Default: also scales XY translation and root yaw, leaves Z alone. A
      0.5 factor on a 1-foot step gives a clean 0.5-foot variant.

  recenter_root {xy?: bool, yaw?: bool}
      Subtract the net XY drift / net yaw drift across the whole window so
      the primitive starts and ends at the same world XY / yaw. Used for
      walks where the curator's slice has small unwanted drift.

  pad_idle {leading_frames?: int, trailing_frames?: int}
      Prepend / append static frames with DEFAULT_STAND_POSE + identity
      root rot at the existing root XY. Buys the runtime blender extra
      headroom into / out of static bins.

Recipe schema (YAML)
--------------------

    primitives:
      - bin_name: idle_stand
        family: idle
        ops:
          - clip_window: {motion_key: ..., start_frame: ..., n_frames: ...}

      - bin_name: lean_fwd_medium
        family: static_upper_body
        ops:
          - synthesize_waist_ramp:
              axis: pitch
              peak_deg: 20.0
              ramp_in_frames: 30
              hold_frames: 20
              ramp_out_frames: 30
          - freeze: {groups: [arms, head]}

      - bin_name: turn_right_45deg
        family: locomotion
        derive_from: turn_left_45deg     # run that recipe first
        ops:
          - mirror_lr: {}

      - bin_name: fwd_step_half_ft
        family: locomotion
        derive_from: fwd_step_1ft
        ops:
          - scale_magnitude: {factor: 0.5}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from .constants import (
    DEFAULT_PELVIS_Z_M,
    DEFAULT_STAND_POSE_NP,
    HEAD_INDICES,
    LEFT_ANKLE_PITCH_IDX,
    LEFT_ANKLE_ROLL_IDX,
    LEFT_ARM_INDICES,
    LEFT_HIP_PITCH_IDX,
    LEFT_HIP_ROLL_IDX,
    LEFT_HIP_YAW_IDX,
    LEFT_KNEE_IDX,
    LEFT_LEG_INDICES,
    LEG_INDICES,
    NUM_BODY_DOFS,
    RIGHT_ANKLE_PITCH_IDX,
    RIGHT_ANKLE_ROLL_IDX,
    RIGHT_ARM_INDICES,
    RIGHT_HIP_PITCH_IDX,
    RIGHT_HIP_ROLL_IDX,
    RIGHT_HIP_YAW_IDX,
    RIGHT_KNEE_IDX,
    RIGHT_LEG_INDICES,
    WAIST_INDICES,
    WAIST_PITCH_IDX,
    WAIST_ROLL_IDX,
    WAIST_YAW_IDX,
)


# Indices that must change sign when the *swapped* (post-L<->R) joint slot is
# written, because the joint axis points oppositely on the two sides
# (hip_roll, hip_yaw, ankle_roll, shoulder_roll, shoulder_yaw, wrist_yaw,
# wrist_roll). Verified by mirroring DEFAULT_STAND_POSE: shoulder_roll has
# values +0.2 (left) and -0.2 (right); after swap those land in the wrong
# slots and need negation to restore the bilateral identity.
_POST_SWAP_NEGATE_INDICES: tuple[int, ...] = (
    1, 7,        # hip_roll
    2, 8,        # hip_yaw
    5, 11,       # ankle_roll
    16, 23,      # shoulder_roll
    17, 24,      # shoulder_yaw
    19, 26,      # wrist_yaw
    21, 28,      # wrist_roll
)

# Body-axial joints that simply negate (no swap, single index).
_BODY_AXIAL_NEGATE_INDICES: tuple[int, ...] = (
    WAIST_YAW_IDX,
    WAIST_ROLL_IDX,
    29,          # head_yaw
)

# L<->R joint-index swap pairs (legs + arms). Pitch joints stay in their slot
# but get the swapped sibling's value. Order: (left_idx, right_idx).
_LR_SWAP_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),         # legs
    (15, 22), (16, 23), (17, 24), (18, 25),                   # arms (sh+elbow)
    (19, 26), (20, 27), (21, 28),                             # arms (wrists)
)

# Named joint groups for the ``freeze`` op.
_GROUP_INDICES: dict[str, tuple[int, ...]] = {
    "left_leg": tuple(LEFT_LEG_INDICES),
    "right_leg": tuple(RIGHT_LEG_INDICES),
    "legs": tuple(LEG_INDICES),
    "left_arm": tuple(LEFT_ARM_INDICES),
    "right_arm": tuple(RIGHT_ARM_INDICES),
    "arms": tuple(LEFT_ARM_INDICES) + tuple(RIGHT_ARM_INDICES),
    "head": tuple(HEAD_INDICES),
    "waist": tuple(WAIST_INDICES),
    "waist_yaw": (WAIST_YAW_IDX,),
    "waist_pitch": (WAIST_PITCH_IDX,),
    "waist_roll": (WAIST_ROLL_IDX,),
    "all_but_legs": tuple(
        i for i in range(NUM_BODY_DOFS) if i not in LEG_INDICES
    ),
    "all_but_waist": tuple(
        i for i in range(NUM_BODY_DOFS) if i not in WAIST_INDICES
    ),
    "all_but_legs_and_waist": tuple(
        i for i in range(NUM_BODY_DOFS)
        if i not in LEG_INDICES and i not in WAIST_INDICES
    ),
}

# Default fps for synthesized primitives (matches state-machine OUTPUT_FPS).
_SYNTH_FPS_DEFAULT: float = 50.0


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceClip:
    """One clip from the source motion library, normalized to numpy."""

    motion_key: str
    dof: np.ndarray            # (T, 31) float32
    root_rot_xyzw: np.ndarray  # (T, 4) float32, scipy xyzw
    root_trans: np.ndarray     # (T, 3) float32
    fps: float


@dataclass
class Buffer:
    """Mutable working buffer threaded through the ops pipeline."""

    dof: np.ndarray            # (T, 31) float64
    root_rot_xyzw: np.ndarray  # (T, 4)  float64
    root_trans: np.ndarray     # (T, 3)  float64
    fps: float
    # Provenance: which source clips contributed (for the report / browser).
    sources: list[str] = field(default_factory=list)

    def n_frames(self) -> int:
        return int(self.dof.shape[0])

    def copy(self) -> "Buffer":
        return Buffer(
            dof=self.dof.copy(),
            root_rot_xyzw=self.root_rot_xyzw.copy(),
            root_trans=self.root_trans.copy(),
            fps=self.fps,
            sources=list(self.sources),
        )


@dataclass(frozen=True)
class Recipe:
    """One row of ``x2_planner_primitives_recipes.yaml``."""

    bin_name: str
    family: str
    ops: tuple[dict[str, Any], ...]
    derive_from: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Op implementations
# ---------------------------------------------------------------------------


def _stand_pose_64() -> np.ndarray:
    return DEFAULT_STAND_POSE_NP.astype(np.float64).copy()


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def op_clip_window(
    args: dict[str, Any],
    _buf: Buffer | None,
    source_clips: dict[str, SourceClip],
) -> Buffer:
    motion_key = str(args["motion_key"])
    start = int(args["start_frame"])
    n = int(args["n_frames"])
    if motion_key not in source_clips:
        raise KeyError(
            f"clip_window: motion_key {motion_key!r} not in source library"
        )
    clip = source_clips[motion_key]
    if start < 0 or start + n > clip.dof.shape[0]:
        raise ValueError(
            f"clip_window: window [{start}:{start + n}] out of bounds for "
            f"clip {motion_key!r} ({clip.dof.shape[0]} frames)"
        )
    return Buffer(
        dof=clip.dof[start : start + n].astype(np.float64),
        root_rot_xyzw=clip.root_rot_xyzw[start : start + n].astype(np.float64),
        root_trans=clip.root_trans[start : start + n].astype(np.float64),
        fps=float(clip.fps),
        sources=[f"{motion_key}[{start}:{start + n}]"],
    )


# Hard safety caps on the synthesized waist ramp peak. The op REFUSES to
# build a primitive that exceeds these -- a buggy recipe should fail at
# build time, not mid-deploy. Values are based on the SONIC tracking
# envelope around DEFAULT_STAND_POSE:
#   pitch (lean_fwd / lean_back): foot half-length is ~12 cm; 20 deg total
#     torso pitch puts the wrist + payload CG at the front of the support
#     polygon. Anything larger needs a stepping recovery, which static
#     bins can't provide.
#   roll  (lean_left / lean_right): foot half-WIDTH is only ~5 cm, so the
#     CG margin is tighter. 10 deg waist roll is right at the edge of the
#     standing support polygon.
#   yaw   (torso_left / torso_right): hip yaw is unloaded; 40 deg covers
#     the useful reach envelope without the SONIC policy losing tracking.
#
# Bumping these requires a deploy-side validation pass (record SONIC
# pelvis / foot trajectories at the new cap and confirm no fall).
_WAIST_RAMP_CAP_DEG: dict[str, float] = {
    "pitch": 20.0,
    "roll": 10.0,
    "yaw": 40.0,
}

# Maximum |share| for any of the counter-balance share parameters. Above
# this the leg / ankle deltas exceed the natural human "deadlift hinge"
# proportion (hip flex up to ~1.5x waist pitch) and start to look OOD to
# the SONIC policy.
_MAX_SHARE_MAGNITUDE: float = 1.5


def _validate_share_magnitudes(
    *,
    hip_pitch_share: float,
    hip_yaw_share: float,
    ankle_pitch_share: float,
    ankle_roll_share: float,
    src_label: str,
) -> None:
    """Common share-magnitude check used by build-time op + runtime helper."""
    for name, val in (
        ("hip_pitch_share", hip_pitch_share),
        ("hip_yaw_share", hip_yaw_share),
        ("ankle_pitch_share", ankle_pitch_share),
        ("ankle_roll_share", ankle_roll_share),
    ):
        if abs(val) > _MAX_SHARE_MAGNITUDE:
            raise ValueError(
                f"{src_label}: |{name}|={abs(val):.2f} exceeds "
                f"{_MAX_SHARE_MAGNITUDE} (would over-actuate the leg / "
                f"ankle joints relative to the waist track)."
            )


def make_waist_pose_frame(
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
    *,
    hip_pitch_share: float = 0.0,
    hip_yaw_share: float = 0.0,
    ankle_pitch_share: float = 0.0,
    ankle_roll_share: float = 0.0,
    clamp: bool = True,
) -> np.ndarray:
    """Build one 31-DOF frame layered on DEFAULT_STAND_POSE.

    Composes pitch + roll + yaw simultaneously (the build-time
    ``op_synthesize_waist_ramp`` drives one axis at a time; this helper
    stacks all three so the runtime planner can drive a continuous
    "lean + twist + sway" target from VR sticks).

    Args:
        pitch_deg / roll_deg / yaw_deg: signed angles in degrees.
        hip_pitch_share, hip_yaw_share, ankle_pitch_share,
        ankle_roll_share: counter-balance shares -- same convention as
            ``op_synthesize_waist_ramp``. Each defaults to 0.0 (pure
            waist motion). |share| is hard-capped at ``_MAX_SHARE_MAGNITUDE``.
        clamp: if True (default) angles outside ``_WAIST_RAMP_CAP_DEG``
            are clamped to the cap before evaluation. The build-time op
            instead RAISES at build time if peak_deg exceeds the cap;
            the runtime helper clamps because the operator's stick
            should not be able to crash the planner with a bad value.

    Returns:
        ``(NUM_BODY_DOFS,)`` float64 array, copy of DEFAULT_STAND_POSE
        with waist + counter-balance offsets applied.
    """
    _validate_share_magnitudes(
        hip_pitch_share=hip_pitch_share,
        hip_yaw_share=hip_yaw_share,
        ankle_pitch_share=ankle_pitch_share,
        ankle_roll_share=ankle_roll_share,
        src_label="make_waist_pose_frame",
    )

    if clamp:
        cap_p = _WAIST_RAMP_CAP_DEG["pitch"]
        cap_r = _WAIST_RAMP_CAP_DEG["roll"]
        cap_y = _WAIST_RAMP_CAP_DEG["yaw"]
        pitch_deg = max(-cap_p, min(cap_p, float(pitch_deg)))
        roll_deg = max(-cap_r, min(cap_r, float(roll_deg)))
        yaw_deg = max(-cap_y, min(cap_y, float(yaw_deg)))
    else:
        pitch_deg = float(pitch_deg)
        roll_deg = float(roll_deg)
        yaw_deg = float(yaw_deg)

    pitch_rad = float(np.deg2rad(pitch_deg))
    roll_rad = float(np.deg2rad(roll_deg))
    yaw_rad = float(np.deg2rad(yaw_deg))

    template = DEFAULT_STAND_POSE_NP.astype(np.float64)
    out = template.copy()

    # Primary waist DOFs.
    out[WAIST_PITCH_IDX] = template[WAIST_PITCH_IDX] + pitch_rad
    out[WAIST_ROLL_IDX] = template[WAIST_ROLL_IDX] + roll_rad
    out[WAIST_YAW_IDX] = template[WAIST_YAW_IDX] + yaw_rad

    # Counter-balance shares (same per-axis convention as the build-time
    # op; see op_synthesize_waist_ramp docstring for the sign rationale).
    if hip_pitch_share != 0.0 and pitch_rad != 0.0:
        out[LEFT_HIP_PITCH_IDX] = (
            template[LEFT_HIP_PITCH_IDX] - pitch_rad * hip_pitch_share
        )
        out[RIGHT_HIP_PITCH_IDX] = (
            template[RIGHT_HIP_PITCH_IDX] - pitch_rad * hip_pitch_share
        )
    if ankle_pitch_share != 0.0 and pitch_rad != 0.0:
        out[LEFT_ANKLE_PITCH_IDX] = (
            template[LEFT_ANKLE_PITCH_IDX] - pitch_rad * ankle_pitch_share
        )
        out[RIGHT_ANKLE_PITCH_IDX] = (
            template[RIGHT_ANKLE_PITCH_IDX] - pitch_rad * ankle_pitch_share
        )
    if hip_yaw_share != 0.0 and yaw_rad != 0.0:
        out[LEFT_HIP_YAW_IDX] = (
            template[LEFT_HIP_YAW_IDX] + yaw_rad * hip_yaw_share
        )
        out[RIGHT_HIP_YAW_IDX] = (
            template[RIGHT_HIP_YAW_IDX] - yaw_rad * hip_yaw_share
        )
    if ankle_roll_share != 0.0 and roll_rad != 0.0:
        out[LEFT_ANKLE_ROLL_IDX] = (
            template[LEFT_ANKLE_ROLL_IDX] + roll_rad * ankle_roll_share
        )
        out[RIGHT_ANKLE_ROLL_IDX] = (
            template[RIGHT_ANKLE_ROLL_IDX] - roll_rad * ankle_roll_share
        )

    return out


def op_synthesize_waist_ramp(
    args: dict[str, Any],
    _buf: Buffer | None,
    _src: dict[str, SourceClip],
) -> Buffer:
    """Synthesize a feet-planted lean / twist (parameterized by signed waist angle).

    Layered on top of DEFAULT_STAND_POSE: ramps a single waist axis
    (``pitch``/``yaw``/``roll``) from 0 to ``peak_deg`` over ``ramp_in_frames``,
    holds for ``hold_frames``, then ramps back to 0 over ``ramp_out_frames``.

    Optional counter-balance shares co-actuate the hips / ankles in
    proportion to the waist track to keep the synthesized pose closer to
    the natural human "lean" / "twist" the SONIC policy was trained
    against. Each share defaults to 0.0 (pure waist motion -- backward
    compatible with v1 torso_* recipes).

    Args:
        axis: ``pitch`` (forward/back lean), ``yaw`` (left/right twist),
            or ``roll`` (left/right lateral lean).
        peak_deg: signed peak angle in degrees. ABSOLUTE value is hard-
            capped per axis (``_WAIST_RAMP_CAP_DEG``); a buggy recipe
            asking for 30 deg roll will fail at build time.
        ramp_in_frames / hold_frames / ramp_out_frames: trapezoid envelope
            shape. Total >= 2 frames.
        fps: optional override (default 50, matches OUTPUT_FPS).
        hip_pitch_share: applied when ``axis=pitch``. Both hips flex by
            ``share * waist_track`` (more negative hip_pitch). 0.0 = pure
            waist motion; ~0.30 produces a natural pelvis-and-torso lean
            (matches the v2 body-check mocap reference).
        hip_yaw_share: applied when ``axis=yaw``. LEFT hip_yaw rotates
            ``+share * waist_track`` and RIGHT hip_yaw rotates the
            opposite (anti-symmetric joint axis), which under L<->R
            mirror produces the correct sign for ``torso_right_*`` --
            see ``_POST_SWAP_NEGATE_INDICES``. 0.0 = pure waist twist;
            ~0.30 produces a natural pelvis-shares-the-twist look.
        ankle_pitch_share: applied when ``axis=pitch``. Both ankles
            counter-rotate by ``-share * waist_track`` so the foot stays
            flatter as the body pitches forward. 0.0 = no counter.
        ankle_roll_share: applied when ``axis=roll``. LEFT and RIGHT
            ankle_roll co-rotate (anti-symmetric joint axis, same world
            direction) by ``share * waist_track`` so the feet stay
            flatter as the body tips sideways. 0.0 = no counter.

    Producer op (no input buffer required).
    """
    axis = str(args["axis"]).lower()
    if axis not in {"pitch", "yaw", "roll"}:
        raise ValueError(f"synthesize_waist_ramp: axis must be pitch/yaw/roll, got {axis!r}")
    peak_deg = float(args["peak_deg"])
    cap_deg = _WAIST_RAMP_CAP_DEG[axis]
    if abs(peak_deg) > cap_deg + 1e-6:
        raise ValueError(
            f"synthesize_waist_ramp: |peak_deg|={abs(peak_deg):.2f} exceeds "
            f"axis={axis!r} cap of {cap_deg:.1f} deg. Either lower peak_deg "
            f"or, if you have deploy evidence the new value is trackable, "
            f"raise _WAIST_RAMP_CAP_DEG in x2_recipes.py."
        )
    peak_rad = float(np.deg2rad(peak_deg))
    ramp_in = int(args.get("ramp_in_frames", 30))
    hold = int(args.get("hold_frames", 20))
    ramp_out = int(args.get("ramp_out_frames", 30))
    fps = float(args.get("fps", _SYNTH_FPS_DEFAULT))
    total = ramp_in + hold + ramp_out
    if total < 2:
        raise ValueError(
            "synthesize_waist_ramp: ramp_in + hold + ramp_out must be >= 2"
        )

    hip_pitch_share = float(args.get("hip_pitch_share", 0.0))
    hip_yaw_share = float(args.get("hip_yaw_share", 0.0))
    ankle_pitch_share = float(args.get("ankle_pitch_share", 0.0))
    ankle_roll_share = float(args.get("ankle_roll_share", 0.0))
    # Safety: counter-shares can produce huge joint deltas for large
    # waist peaks. 1.5 covers the natural "deadlift hinge" (hip flex up
    # to 1.5x waist pitch) without going OOD.
    _validate_share_magnitudes(
        hip_pitch_share=hip_pitch_share,
        hip_yaw_share=hip_yaw_share,
        ankle_pitch_share=ankle_pitch_share,
        ankle_roll_share=ankle_roll_share,
        src_label="synthesize_waist_ramp",
    )

    axis_idx = {
        "pitch": WAIST_PITCH_IDX,
        "yaw": WAIST_YAW_IDX,
        "roll": WAIST_ROLL_IDX,
    }[axis]

    waist_track = np.zeros(total, dtype=np.float64)
    if ramp_in > 0:
        # Inclusive ramp 0 -> peak across ramp_in frames.
        waist_track[:ramp_in] = np.linspace(0.0, peak_rad, ramp_in, endpoint=True)
    if hold > 0:
        waist_track[ramp_in : ramp_in + hold] = peak_rad
    if ramp_out > 0:
        waist_track[ramp_in + hold :] = np.linspace(
            peak_rad, 0.0, ramp_out, endpoint=True
        )

    dof = np.broadcast_to(_stand_pose_64(), (total, NUM_BODY_DOFS)).copy()
    template = DEFAULT_STAND_POSE_NP.astype(np.float64)
    dof[:, axis_idx] = template[axis_idx] + waist_track

    # Counter-balance application (axis-specific). All shares default to
    # 0.0, so existing torso_* recipes (which never set them) build the
    # exact same pure-waist pose as before.
    #
    # Same sign conventions as the runtime ``make_waist_pose_frame`` --
    # the regression test ``test_make_waist_pose_frame_matches_op_*``
    # asserts the two paths agree up to float tolerance.
    if axis == "pitch":
        if hip_pitch_share != 0.0:
            # Hip pitch baseline is NEGATIVE (-0.312 rad = forward flex).
            # Forward waist pitch (peak_deg > 0) -> add MORE flex
            # (subtract from the negative baseline). Backward lean
            # (peak_deg < 0) -> subtract less (extend), same formula.
            dof[:, LEFT_HIP_PITCH_IDX] = (
                template[LEFT_HIP_PITCH_IDX] - waist_track * hip_pitch_share
            )
            dof[:, RIGHT_HIP_PITCH_IDX] = (
                template[RIGHT_HIP_PITCH_IDX] - waist_track * hip_pitch_share
            )
        if ankle_pitch_share != 0.0:
            # Ankle pitch baseline is NEGATIVE (-0.363 rad). Counter-
            # rotate so the foot stays flatter as the body tips.
            dof[:, LEFT_ANKLE_PITCH_IDX] = (
                template[LEFT_ANKLE_PITCH_IDX] - waist_track * ankle_pitch_share
            )
            dof[:, RIGHT_ANKLE_PITCH_IDX] = (
                template[RIGHT_ANKLE_PITCH_IDX] - waist_track * ankle_pitch_share
            )
    elif axis == "yaw":
        if hip_yaw_share != 0.0:
            # See make_waist_pose_frame docstring for the sign rationale
            # on anti-symmetric joints.
            dof[:, LEFT_HIP_YAW_IDX] = (
                template[LEFT_HIP_YAW_IDX] + waist_track * hip_yaw_share
            )
            dof[:, RIGHT_HIP_YAW_IDX] = (
                template[RIGHT_HIP_YAW_IDX] - waist_track * hip_yaw_share
            )
    elif axis == "roll":
        if ankle_roll_share != 0.0:
            dof[:, LEFT_ANKLE_ROLL_IDX] = (
                template[LEFT_ANKLE_ROLL_IDX] + waist_track * ankle_roll_share
            )
            dof[:, RIGHT_ANKLE_ROLL_IDX] = (
                template[RIGHT_ANKLE_ROLL_IDX] - waist_track * ankle_roll_share
            )

    rot = np.broadcast_to(_identity_quat(), (total, 4)).copy()
    trans = np.zeros((total, 3), dtype=np.float64)
    trans[:, 2] = DEFAULT_PELVIS_Z_M

    src_tag = f"synth:waist_{axis}_ramp(peak={peak_rad:.4f}rad"
    if hip_pitch_share != 0.0:
        src_tag += f",hip_pitch_share={hip_pitch_share:.2f}"
    if hip_yaw_share != 0.0:
        src_tag += f",hip_yaw_share={hip_yaw_share:.2f}"
    if ankle_pitch_share != 0.0:
        src_tag += f",ankle_pitch_share={ankle_pitch_share:.2f}"
    if ankle_roll_share != 0.0:
        src_tag += f",ankle_roll_share={ankle_roll_share:.2f}"
    src_tag += ")"

    return Buffer(
        dof=dof,
        root_rot_xyzw=rot,
        root_trans=trans,
        fps=fps,
        sources=[src_tag],
    )


def op_synthesize_crouch_ramp(
    args: dict[str, Any],
    _buf: Buffer | None,
    _src: dict[str, SourceClip],
) -> Buffer:
    """Synthesize a feet-planted crouch (parameterized by peak pelvis-Z drop).

    Layered on top of DEFAULT_STAND_POSE: ramps the pelvis Z down by
    ``peak_drop_m`` and bends knees / hips / ankles in a self-consistent
    triangle (knee_delta = 2*hip_delta = 2*ankle_delta) so that

        pelvis_z drop ≈ 2 * thigh_length * (1 - cos(hip_delta))
                      = 2 * 0.4   * (1 - cos(knee_delta / 2))

    holds (assuming thigh_length = shin_length = 0.40 m on the X2). The
    ankle_pitch counter-rotates so the foot stays flat. Result is a
    physically coherent squat the SONIC policy can track at any depth,
    avoiding the "shallow-crouch hallucination" that scaling a real squat
    clip with ``scale_magnitude`` produces.

    Args:
        peak_drop_m: peak pelvis-Z drop below baseline (meters).
        ramp_in_frames: trapezoid up-ramp length.
        hold_frames: dwell at apex.
        ramp_out_frames: down-ramp back to stand.
        fps: optional override (default 50, matches OUTPUT_FPS).

    Producer op (no input buffer required).
    """
    peak_drop_m = float(args["peak_drop_m"])
    if peak_drop_m <= 0 or peak_drop_m > 0.18:
        raise ValueError(
            f"synthesize_crouch_ramp: peak_drop_m must be in (0, 0.18], "
            f"got {peak_drop_m}"
        )
    ramp_in = int(args.get("ramp_in_frames", 25))
    hold = int(args.get("hold_frames", 10))
    ramp_out = int(args.get("ramp_out_frames", 25))
    fps = float(args.get("fps", _SYNTH_FPS_DEFAULT))
    total = ramp_in + hold + ramp_out
    if total < 4:
        raise ValueError(
            "synthesize_crouch_ramp: ramp_in + hold + ramp_out must be >= 4"
        )

    # Reference geometry (X2 Ultra leg ~ 0.40 m thigh + 0.40 m shin).
    THIGH_M = 0.40
    # Inverse of pelvis_z drop = 2 * THIGH_M * (1 - cos(hip)). Solve for
    # hip when knee = 2 * hip.
    cos_hip = 1.0 - peak_drop_m / (2.0 * THIGH_M)
    cos_hip = max(-1.0, min(1.0, cos_hip))
    hip_peak = float(np.arccos(cos_hip))
    knee_peak = 2.0 * hip_peak
    ankle_peak = -hip_peak  # counter-rotate to keep foot flat

    # Ramp envelope (trapezoid 0 -> 1 -> 0).
    env = np.zeros(total, dtype=np.float64)
    if ramp_in > 0:
        env[:ramp_in] = np.linspace(0.0, 1.0, ramp_in, endpoint=True)
    if hold > 0:
        env[ramp_in : ramp_in + hold] = 1.0
    if ramp_out > 0:
        env[ramp_in + hold :] = np.linspace(1.0, 0.0, ramp_out, endpoint=True)

    dof = np.broadcast_to(_stand_pose_64(), (total, NUM_BODY_DOFS)).copy()
    template = DEFAULT_STAND_POSE_NP.astype(np.float64)
    # Hip pitch on the X2 stand pose is NEGATIVE (-0.312 rad). To bend
    # into a squat we add a NEGATIVE delta (more flex). Knee is positive
    # baseline (+0.669) and gets MORE positive (more bend). Ankle pitch
    # is negative (-0.363) and gets MORE negative (toes pull up).
    dof[:, LEFT_HIP_PITCH_IDX] = template[LEFT_HIP_PITCH_IDX] - hip_peak * env
    dof[:, RIGHT_HIP_PITCH_IDX] = template[RIGHT_HIP_PITCH_IDX] - hip_peak * env
    dof[:, LEFT_KNEE_IDX] = template[LEFT_KNEE_IDX] + knee_peak * env
    dof[:, RIGHT_KNEE_IDX] = template[RIGHT_KNEE_IDX] + knee_peak * env
    dof[:, LEFT_ANKLE_PITCH_IDX] = template[LEFT_ANKLE_PITCH_IDX] + ankle_peak * env
    dof[:, RIGHT_ANKLE_PITCH_IDX] = template[RIGHT_ANKLE_PITCH_IDX] + ankle_peak * env

    rot = np.broadcast_to(_identity_quat(), (total, 4)).copy()
    trans = np.zeros((total, 3), dtype=np.float64)
    trans[:, 2] = DEFAULT_PELVIS_Z_M - peak_drop_m * env

    return Buffer(
        dof=dof,
        root_rot_xyzw=rot,
        root_trans=trans,
        fps=fps,
        sources=[
            f"synth:crouch_ramp(peak_drop={peak_drop_m:.3f}m,"
            f"hip={hip_peak:.3f},knee={knee_peak:.3f})"
        ],
    )


# Joint indices for hip-roll synthesis. Constants module exposes
# LEFT/RIGHT_HIP_PITCH/KNEE/ANKLE_PITCH but not the roll axes (none of
# the prior recipes touched roll). Indices come from MUJOCO_JOINT_NAMES
# in constants.py: LEFT_HIP_ROLL=1, RIGHT_HIP_ROLL=7. Confirmed by
# _POST_SWAP_NEGATE_INDICES = (1, 7, ...) in this file -- those are
# the hip_roll slots that need sign-flip on L<->R mirror.
_LEFT_HIP_ROLL_IDX: int = 1
_RIGHT_HIP_ROLL_IDX: int = 7


def op_synthesize_side_step_ramp(
    args: dict[str, Any],
    _buf: Buffer | None,
    _src: dict[str, SourceClip],
) -> Buffer:
    """Synthesize a discrete side-step shuffle (parameterized by signed lateral distance).

    Layered on top of DEFAULT_STAND_POSE: a 2-phase shuffle that translates
    the pelvis laterally by ``peak_lateral_m`` using ONLY hip-roll abduction/
    adduction + mild knee/ankle lift -- NO hip_pitch (so no leg-behind
    scissor) and NO waist_yaw (so no body twist). The result is a clean
    feet-near-ground shuffle: swing foot lifts and slides outward, trailing
    foot follows, both feet end flat at the new pelvis-shifted stance.

    Phase structure (active region = ``hold_frames`` between ``ramp_in/out``
    flat tails):

      First half of hold: SWING foot (the foot in the direction of motion)
        lifts (knee flex + ankle dorsi-flex) and abducts outward via hip
        roll. Pelvis translates the first half of ``peak_lateral_m``.
      Second half of hold: TRAIL foot lifts and adducts toward the swing
        foot (closes the stance). Pelvis translates the remaining half.

    Direction convention: +peak_lateral_m moves robot to its LEFT (+Y in
    body frame). -peak_lateral_m moves to its RIGHT.

    Why this exists vs. mocap side-walk clips: every side-walk clip in
    the X2 source library is a continuous gait where the trailing leg
    pre-loads behind the body for the next stride (negative hip_pitch),
    plus the waist twists sideways during weight transfer. Both look
    wrong as a single discrete step. Synthesising lets us isolate the
    pure lateral-abduction component the policy needs to track.

    Args:
        peak_lateral_m: signed lateral translation in meters.
            Positive = robot's LEFT, negative = robot's RIGHT. Magnitude
            must be in (0.01, 0.20] m.
        ramp_in_frames: pre-step idle (DEFAULT_STAND_POSE held).
        hold_frames: active step duration (split half-half across the
            two foot-lift phases). Must be >= 4.
        ramp_out_frames: post-step idle at the new stance.
        knee_lift_rad: peak knee flex during foot lift (default 0.12 rad
            ≈ 7 deg, with matching ankle dorsi-flex of half that).
        fps: optional override (default 50, matches OUTPUT_FPS).

    Producer op (no input buffer required).
    """
    peak_lateral_m = float(args["peak_lateral_m"])
    if abs(peak_lateral_m) < 0.01 or abs(peak_lateral_m) > 0.20:
        raise ValueError(
            f"synthesize_side_step_ramp: |peak_lateral_m| must be in "
            f"(0.01, 0.20] m, got {peak_lateral_m}"
        )
    ramp_in = int(args.get("ramp_in_frames", 10))
    hold = int(args.get("hold_frames", 60))
    ramp_out = int(args.get("ramp_out_frames", 15))
    knee_lift_rad = float(args.get("knee_lift_rad", 0.30))
    fps = float(args.get("fps", _SYNTH_FPS_DEFAULT))
    if hold < 4:
        raise ValueError(
            f"synthesize_side_step_ramp: hold_frames must be >= 4, "
            f"got {hold}"
        )
    total = ramp_in + hold + ramp_out
    if total < 8:
        raise ValueError(
            "synthesize_side_step_ramp: total frames must be >= 8"
        )

    # Direction: +peak_lateral_m moves LEFT. swing leg = leg on the side
    # we're moving toward (it lifts first and lands at the new outer
    # position); trail leg follows.
    swing_is_left = peak_lateral_m > 0
    if swing_is_left:
        swing_knee_idx = LEFT_KNEE_IDX
        swing_ankle_idx = LEFT_ANKLE_PITCH_IDX
        swing_hip_roll_idx = _LEFT_HIP_ROLL_IDX
        trail_knee_idx = RIGHT_KNEE_IDX
        trail_ankle_idx = RIGHT_ANKLE_PITCH_IDX
        trail_hip_roll_idx = _RIGHT_HIP_ROLL_IDX
    else:
        swing_knee_idx = RIGHT_KNEE_IDX
        swing_ankle_idx = RIGHT_ANKLE_PITCH_IDX
        swing_hip_roll_idx = _RIGHT_HIP_ROLL_IDX
        trail_knee_idx = LEFT_KNEE_IDX
        trail_ankle_idx = LEFT_ANKLE_PITCH_IDX
        trail_hip_roll_idx = _LEFT_HIP_ROLL_IDX

    # Hip roll axis convention:
    #   LEFT_HIP_ROLL  > 0  -> LEFT  foot offsets to robot's +Y (LEFT)
    #   RIGHT_HIP_ROLL > 0  -> RIGHT foot offsets to robot's +Y (LEFT, ADDUCT)
    # (Both axes go the same way; mirror_lr handles the sign flip via the
    # POST_SWAP_NEGATE table for hip_roll indices 1 and 7.)
    # For a LEFT step (peak_lateral_m > 0), both abduct values are +.
    # For a RIGHT step, both are negative.
    abduct_sign = 1.0 if swing_is_left else -1.0

    # Hip abduction magnitude. Geometry: with the swing foot in the air
    # and the trail foot planted under a stationary pelvis, abducting
    # the swing hip by theta moves the swing foot laterally by
    # leg_length * sin(theta). Use the small-angle inverse to land the
    # swing foot exactly at +peak_lateral_m relative to the pelvis. The
    # trail leg then contributes the second half of the translation by
    # un-abducting the swing hip (foot stays planted -> pelvis catches up).
    LEG_LENGTH_M = 0.80
    abduct_mag = float(np.arcsin(min(0.99, abs(peak_lateral_m) / LEG_LENGTH_M)))
    ankle_lift_rad = knee_lift_rad / 2.0

    # Phase split. Phase 1 = swing-foot lift+plant; phase 2 = trail-foot
    # lift+plant + pelvis catches up.
    h_first = max(4, hold // 2)
    h_second = max(4, hold - h_first)

    # ---- Foot lift envelopes (half-sine: lift, plant) ----
    # Each foot lifts during ITS phase only; flat at 0 elsewhere.
    swing_lift_env = np.zeros(total, dtype=np.float64)
    trail_lift_env = np.zeros(total, dtype=np.float64)
    swing_lift_env[ramp_in : ramp_in + h_first] = np.sin(
        np.linspace(0.0, np.pi, h_first)
    )
    trail_lift_env[ramp_in + h_first : ramp_in + h_first + h_second] = np.sin(
        np.linspace(0.0, np.pi, h_second)
    )

    # ---- Swing-leg hip-roll: monotonic up-then-down ramp ----
    # Phase 1 (swing foot in air -> plants at +abduct_mag offset):
    #   smoothstep 0 -> 1 (foot ends at outer position)
    # Phase 2 (swing foot planted, pelvis catches up):
    #   smoothstep 1 -> 0 (foot stays put, pelvis translates)
    # This is the KEY difference from a half-sine envelope: the swing
    # foot does NOT return to its starting position mid-stride. It plants
    # at +offset; then in phase 2 the hip closes back to neutral while
    # the foot is on the ground, which (via ground reaction) translates
    # the pelvis instead of moving the foot.
    def _smoothstep(t: np.ndarray) -> np.ndarray:
        return t * t * (3.0 - 2.0 * t)

    hip_roll_swing_profile = np.zeros(total, dtype=np.float64)
    if h_first >= 2:
        t_p1 = np.linspace(0.0, 1.0, h_first)
        hip_roll_swing_profile[ramp_in : ramp_in + h_first] = _smoothstep(t_p1)
    if h_second >= 2:
        t_p2 = np.linspace(0.0, 1.0, h_second)
        hip_roll_swing_profile[
            ramp_in + h_first : ramp_in + h_first + h_second
        ] = 1.0 - _smoothstep(t_p2)
    # ramp_out tail stays at 0 (back to neutral stance).

    # Trail-leg hip roll: stays at 0 throughout. The trail foot starts
    # under the pelvis at frame 0, lifts during phase 2, and lands at
    # the NEW pelvis Y (which has translated by peak_lateral_m). Since
    # the trail leg is in the air during the pelvis translation and ends
    # under the new pelvis position, no hip-roll change is needed -- the
    # trail foot just rides with the pelvis.

    # ---- Pelvis Y track ----
    # Phase 1: pelvis stays at Y=0 (supported by trail foot).
    # Phase 2: pelvis translates 0 -> +peak_lateral_m via smoothstep
    #          (driven by ground-reaction as swing-leg hip un-abducts).
    # ramp_out: pelvis holds at +peak_lateral_m.
    trans_y = np.zeros(total, dtype=np.float64)
    if h_second >= 2:
        t_p2 = np.linspace(0.0, 1.0, h_second)
        trans_y[
            ramp_in + h_first : ramp_in + h_first + h_second
        ] = peak_lateral_m * _smoothstep(t_p2)
    trans_y[ramp_in + h_first + h_second :] = peak_lateral_m

    # ---- Assemble dof buffer ----
    dof = np.broadcast_to(_stand_pose_64(), (total, NUM_BODY_DOFS)).copy()
    template = DEFAULT_STAND_POSE_NP.astype(np.float64)
    # Knee flex during each foot's lift (positive delta = more bend).
    dof[:, swing_knee_idx] = template[swing_knee_idx] + knee_lift_rad * swing_lift_env
    dof[:, trail_knee_idx] = template[trail_knee_idx] + knee_lift_rad * trail_lift_env
    # Ankle dorsi-flex during each foot's lift (negative delta = toes up).
    dof[:, swing_ankle_idx] = template[swing_ankle_idx] - ankle_lift_rad * swing_lift_env
    dof[:, trail_ankle_idx] = template[trail_ankle_idx] - ankle_lift_rad * trail_lift_env
    # Swing-leg hip roll = monotonic ramp described above.
    dof[:, swing_hip_roll_idx] = abduct_sign * abduct_mag * hip_roll_swing_profile
    # Trail-leg hip roll stays at 0 (default already in stand pose; nothing
    # to write -- left here as a comment so the asymmetry is explicit).
    _ = trail_hip_roll_idx  # silence unused; trail hip roll intentionally not modulated

    # Identity quat (no body twist), Z held at default pelvis height.
    rot = np.broadcast_to(_identity_quat(), (total, 4)).copy()
    trans = np.zeros((total, 3), dtype=np.float64)
    trans[:, 1] = trans_y
    trans[:, 2] = DEFAULT_PELVIS_Z_M

    return Buffer(
        dof=dof,
        root_rot_xyzw=rot,
        root_trans=trans,
        fps=fps,
        sources=[
            f"synth:side_step_ramp(lat={peak_lateral_m:+.3f}m,"
            f"abduct={abduct_mag:.3f}rad,knee_lift={knee_lift_rad:.3f}rad)"
        ],
    )


def op_freeze(
    args: dict[str, Any],
    buf: Buffer | None,
    _src: dict[str, SourceClip],
) -> Buffer:
    if buf is None:
        raise ValueError("freeze: must follow a producer op")
    groups = args.get("groups") or []
    if not groups:
        raise ValueError("freeze: 'groups' must be a non-empty list")
    indices: list[int] = []
    for g in groups:
        g = str(g).lower()
        if g not in _GROUP_INDICES:
            raise ValueError(
                f"freeze: unknown group {g!r}. "
                f"Allowed: {sorted(_GROUP_INDICES)}"
            )
        indices.extend(_GROUP_INDICES[g])
    indices = sorted(set(indices))
    new = buf.copy()
    template = DEFAULT_STAND_POSE_NP.astype(np.float64)
    for i in indices:
        new.dof[:, i] = template[i]
    new.sources.append(f"op:freeze({','.join(sorted(set(groups)))})")
    return new


def _mirror_quats(rot_xyzw: np.ndarray) -> np.ndarray:
    """Mirror across the sagittal (XZ) plane: (qx, qy, qz, qw) -> (-qx, qy, -qz, qw)."""
    out = rot_xyzw.copy()
    out[..., 0] = -out[..., 0]
    out[..., 2] = -out[..., 2]
    return out


def op_mirror_lr(
    args: dict[str, Any],
    buf: Buffer | None,
    _src: dict[str, SourceClip],
) -> Buffer:
    if buf is None:
        raise ValueError("mirror_lr: must follow a producer op")
    also_negate_root_yaw = bool(args.get("also_negate_root_yaw", True))
    also_negate_root_y = bool(args.get("also_negate_root_y", True))

    new_dof = buf.dof.copy()
    # Step 1: swap L<->R joint columns.
    for left, right in _LR_SWAP_PAIRS:
        new_dof[:, left], new_dof[:, right] = (
            buf.dof[:, right].copy(),
            buf.dof[:, left].copy(),
        )
    # Step 2: post-swap negation for anti-symmetric joints.
    for i in _POST_SWAP_NEGATE_INDICES:
        new_dof[:, i] = -new_dof[:, i]
    # Step 3: body-axial single negation (waist_yaw/roll, head_yaw).
    for i in _BODY_AXIAL_NEGATE_INDICES:
        new_dof[:, i] = -new_dof[:, i]

    new_rot = _mirror_quats(buf.root_rot_xyzw) if also_negate_root_yaw else buf.root_rot_xyzw.copy()
    new_trans = buf.root_trans.copy()
    if also_negate_root_y:
        new_trans[:, 1] = -new_trans[:, 1]

    new = Buffer(
        dof=new_dof,
        root_rot_xyzw=new_rot,
        root_trans=new_trans,
        fps=buf.fps,
        sources=list(buf.sources) + ["op:mirror_lr"],
    )
    return new


def op_scale_magnitude(
    args: dict[str, Any],
    buf: Buffer | None,
    _src: dict[str, SourceClip],
) -> Buffer:
    if buf is None:
        raise ValueError("scale_magnitude: must follow a producer op")
    factor = float(args["factor"])
    scale_xy = bool(args.get("scale_xy", True))
    scale_yaw = bool(args.get("scale_yaw", True))
    scale_z = bool(args.get("scale_z", False))

    template = DEFAULT_STAND_POSE_NP.astype(np.float64)
    new_dof = template[None, :] + factor * (buf.dof - template[None, :])

    new_trans = buf.root_trans.copy()
    if scale_xy:
        anchor = buf.root_trans[0]
        new_trans[:, 0] = anchor[0] + factor * (buf.root_trans[:, 0] - anchor[0])
        new_trans[:, 1] = anchor[1] + factor * (buf.root_trans[:, 1] - anchor[1])
    if scale_z:
        anchor_z = buf.root_trans[0, 2]
        new_trans[:, 2] = anchor_z + factor * (buf.root_trans[:, 2] - anchor_z)

    if scale_yaw:
        # Decompose to euler ZYX, scale yaw delta from frame 0, recompose.
        rots = Rot.from_quat(buf.root_rot_xyzw)
        eulers = rots.as_euler("zyx")
        yaw0 = eulers[0, 0]
        yaw_delta = eulers[:, 0] - yaw0
        eulers[:, 0] = yaw0 + factor * yaw_delta
        new_rot = Rot.from_euler("zyx", eulers).as_quat()
    else:
        new_rot = buf.root_rot_xyzw.copy()

    return Buffer(
        dof=new_dof,
        root_rot_xyzw=new_rot,
        root_trans=new_trans,
        fps=buf.fps,
        sources=list(buf.sources) + [f"op:scale_magnitude({factor:g})"],
    )


def op_recenter_root(
    args: dict[str, Any],
    buf: Buffer | None,
    _src: dict[str, SourceClip],
) -> Buffer:
    if buf is None:
        raise ValueError("recenter_root: must follow a producer op")
    do_xy = bool(args.get("xy", False))
    do_yaw = bool(args.get("yaw", False))
    if not do_xy and not do_yaw:
        raise ValueError("recenter_root: at least one of xy / yaw must be true")

    new_trans = buf.root_trans.copy()
    new_rot = buf.root_rot_xyzw.copy()

    if do_xy:
        # Linearly subtract net XY drift across the window so end_xy == start_xy.
        net_xy = buf.root_trans[-1, :2] - buf.root_trans[0, :2]
        n = buf.n_frames()
        t = np.linspace(0.0, 1.0, n)
        for k in range(2):
            new_trans[:, k] = buf.root_trans[:, k] - t * net_xy[k]

    if do_yaw:
        rots = Rot.from_quat(buf.root_rot_xyzw)
        eulers = rots.as_euler("zyx")
        yaw0 = eulers[0, 0]
        net_yaw = eulers[-1, 0] - yaw0
        n = buf.n_frames()
        t = np.linspace(0.0, 1.0, n)
        eulers[:, 0] = eulers[:, 0] - t * net_yaw
        new_rot = Rot.from_euler("zyx", eulers).as_quat()

    return Buffer(
        dof=buf.dof.copy(),
        root_rot_xyzw=new_rot,
        root_trans=new_trans,
        fps=buf.fps,
        sources=list(buf.sources)
        + [f"op:recenter_root(xy={do_xy},yaw={do_yaw})"],
    )


def op_pad_idle(
    args: dict[str, Any],
    buf: Buffer | None,
    _src: dict[str, SourceClip],
) -> Buffer:
    if buf is None:
        raise ValueError("pad_idle: must follow a producer op")
    leading = int(args.get("leading_frames", 0))
    trailing = int(args.get("trailing_frames", 0))
    if leading < 0 or trailing < 0:
        raise ValueError("pad_idle: frame counts must be >= 0")
    if leading == 0 and trailing == 0:
        return buf.copy()

    template_dof = DEFAULT_STAND_POSE_NP.astype(np.float64)
    template_quat = _identity_quat()

    # Use the buffer's own start/end XY as the anchor so we don't teleport.
    head_xy = buf.root_trans[0, :2]
    tail_xy = buf.root_trans[-1, :2]
    head_z = buf.root_trans[0, 2]
    tail_z = buf.root_trans[-1, 2]

    head_dof = np.broadcast_to(template_dof, (leading, NUM_BODY_DOFS)).copy()
    head_rot = np.broadcast_to(template_quat, (leading, 4)).copy()
    head_trans = np.zeros((leading, 3), dtype=np.float64)
    head_trans[:, 0] = head_xy[0]
    head_trans[:, 1] = head_xy[1]
    head_trans[:, 2] = head_z

    tail_dof = np.broadcast_to(template_dof, (trailing, NUM_BODY_DOFS)).copy()
    tail_rot = np.broadcast_to(template_quat, (trailing, 4)).copy()
    tail_trans = np.zeros((trailing, 3), dtype=np.float64)
    tail_trans[:, 0] = tail_xy[0]
    tail_trans[:, 1] = tail_xy[1]
    tail_trans[:, 2] = tail_z

    new_dof = np.concatenate([head_dof, buf.dof, tail_dof], axis=0)
    new_rot = np.concatenate([head_rot, buf.root_rot_xyzw, tail_rot], axis=0)
    new_trans = np.concatenate([head_trans, buf.root_trans, tail_trans], axis=0)
    return Buffer(
        dof=new_dof,
        root_rot_xyzw=new_rot,
        root_trans=new_trans,
        fps=buf.fps,
        sources=list(buf.sources)
        + [f"op:pad_idle(lead={leading},trail={trailing})"],
    )


_OP_REGISTRY: dict[str, Callable[[dict[str, Any], Buffer | None, dict[str, SourceClip]], Buffer]] = {
    "clip_window": op_clip_window,
    "synthesize_waist_ramp": op_synthesize_waist_ramp,
    "synthesize_crouch_ramp": op_synthesize_crouch_ramp,
    "synthesize_side_step_ramp": op_synthesize_side_step_ramp,
    "freeze": op_freeze,
    "mirror_lr": op_mirror_lr,
    "scale_magnitude": op_scale_magnitude,
    "recenter_root": op_recenter_root,
    "pad_idle": op_pad_idle,
}

PRODUCER_OPS: frozenset[str] = frozenset(
    {
        "clip_window",
        "synthesize_waist_ramp",
        "synthesize_crouch_ramp",
        "synthesize_side_step_ramp",
    }
)


# ---------------------------------------------------------------------------
# YAML loading / recipe execution
# ---------------------------------------------------------------------------


def load_recipes(path: Path) -> dict[str, Recipe]:
    """Parse the recipes YAML into ``{bin_name: Recipe}``.

    Schema is described in the module docstring. Validates op names exist
    and that each non-derived recipe begins with a producer op.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "primitives" not in raw:
        raise ValueError(f"{path}: top-level must have a 'primitives' list")

    recipes: dict[str, Recipe] = {}
    for entry in raw["primitives"]:
        try:
            bin_name = str(entry["bin_name"])
            family = str(entry["family"])
        except KeyError as exc:
            raise ValueError(
                f"{path}: primitive entry missing field {exc}: {entry!r}"
            ) from exc

        ops_raw = entry.get("ops") or []
        if not isinstance(ops_raw, list):
            raise ValueError(f"{path}: primitive {bin_name!r}: 'ops' must be a list")
        ops_normalized: list[dict[str, Any]] = []
        for op in ops_raw:
            if not isinstance(op, dict) or len(op) != 1:
                raise ValueError(
                    f"{path}: primitive {bin_name!r}: each op must be a "
                    f"single-key dict, got {op!r}"
                )
            op_name = next(iter(op))
            if op_name not in _OP_REGISTRY:
                raise ValueError(
                    f"{path}: primitive {bin_name!r}: unknown op {op_name!r}. "
                    f"Allowed: {sorted(_OP_REGISTRY)}"
                )
            args = op[op_name] or {}
            if not isinstance(args, dict):
                raise ValueError(
                    f"{path}: primitive {bin_name!r}: op {op_name!r} args "
                    f"must be a dict, got {args!r}"
                )
            ops_normalized.append({op_name: dict(args)})

        derive_from = entry.get("derive_from")
        if derive_from is not None:
            derive_from = str(derive_from)

        # If not derived, the first op must be a producer.
        if derive_from is None:
            if not ops_normalized:
                raise ValueError(
                    f"{path}: primitive {bin_name!r}: must have at least one "
                    "op (or set derive_from)"
                )
            first_op = next(iter(ops_normalized[0]))
            if first_op not in PRODUCER_OPS:
                raise ValueError(
                    f"{path}: primitive {bin_name!r}: first op must be a "
                    f"producer ({sorted(PRODUCER_OPS)}), got {first_op!r}"
                )

        recipes[bin_name] = Recipe(
            bin_name=bin_name,
            family=family,
            ops=tuple(ops_normalized),
            derive_from=derive_from,
            notes=str(entry.get("notes", "")),
        )

    if len(recipes) != len(raw["primitives"]):
        raise ValueError(f"{path}: duplicate bin_name in 'primitives'")
    return recipes


def run_recipe(
    recipe: Recipe,
    recipes: dict[str, Recipe],
    source_clips: dict[str, SourceClip],
    _stack: tuple[str, ...] = (),
) -> Buffer:
    """Execute a recipe and return the resulting Buffer.

    Resolves ``derive_from`` recursively (with a cycle check), then applies
    the recipe's own ops. Cheap enough to call per-bin at build time.
    """
    if recipe.bin_name in _stack:
        raise ValueError(
            f"derive_from cycle: {' -> '.join(_stack + (recipe.bin_name,))}"
        )

    if recipe.derive_from is not None:
        if recipe.derive_from not in recipes:
            raise ValueError(
                f"recipe {recipe.bin_name!r}: derive_from "
                f"{recipe.derive_from!r} is not a known recipe"
            )
        buf = run_recipe(
            recipes[recipe.derive_from],
            recipes,
            source_clips,
            _stack + (recipe.bin_name,),
        )
        buf.sources.append(f"derive_from:{recipe.derive_from}")
    else:
        buf = None

    for op in recipe.ops:
        op_name = next(iter(op))
        op_fn = _OP_REGISTRY[op_name]
        args = op[op_name]
        buf = op_fn(args, buf, source_clips)
    if buf is None:
        # Pure derive_from with no ops -> still need a buffer; should be
        # unreachable because load_recipes requires either ops or derive_from
        # via the validation above.
        raise ValueError(
            f"recipe {recipe.bin_name!r}: produced no buffer"
        )
    return buf


__all__ = [
    "Buffer",
    "PRODUCER_OPS",
    "Recipe",
    "SourceClip",
    "load_recipes",
    "make_waist_pose_frame",
    "run_recipe",
    # Op functions exposed for unit tests:
    "op_clip_window",
    "op_freeze",
    "op_mirror_lr",
    "op_pad_idle",
    "op_recenter_root",
    "op_scale_magnitude",
    "op_synthesize_crouch_ramp",
    "op_synthesize_side_step_ramp",
    "op_synthesize_waist_ramp",
]
