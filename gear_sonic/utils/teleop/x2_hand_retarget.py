"""Quest 3 controller / hand-tracking -> X2 OmniHand 10-DOF retargeting.

Two retargeting paths are exposed:

1. Controllers (legacy): a single scalar grasp ratio per side derived
   from the trigger / grip analog axis. All 10 motors lerp uniformly
   between OPEN and CLOSED anchors. Use
   :func:`controller_grasp_ratio` + :func:`grasp_command_from_ratio`.
2. Hand-tracking: per-finger curls in ``[0, 1]`` from the WebXR
   ``XRHand`` 25-joint API, mapped onto each finger's motors
   independently. Use :func:`per_finger_grasp_command_from_curls`.

Both paths share the OPEN / CLOSED anchors below.

The OPEN / CLOSED anchors below are vendored verbatim from
``agitbot-x2-record-and-replay``'s ``constants.py`` (branch
``quest3-bare-hand-control``). Joint order is the firmware-canonical
``HAND_FINGER_NAMES_PER_SIDE``::

    1 thumb_roll   2 thumb_abad   3 thumb_mcp
    4 index_abad   5 index_pip
    6 middle_pip
    7 ring_abad    8 ring_pip
    9 pinky_abad  10 pinky_pip

Asymmetric per side because the abad ranges are mirrored on left vs
right.
"""

from __future__ import annotations

import math

import numpy as np


NUM_HAND_DOF_PER_SIDE: int = 10


HAND_FINGER_NAMES_PER_SIDE: tuple[str, ...] = (
    "thumb_roll",
    "thumb_abad",
    "thumb_mcp",
    "index_abad",
    "index_pip",
    "middle_pip",
    "ring_abad",
    "ring_pip",
    "pinky_abad",
    "pinky_pip",
)


def _deg_to_rad(d: float) -> float:
    return d * math.pi / 180.0


def _deg_list_to_rad(lst: list[float]) -> list[float]:
    return [_deg_to_rad(d) for d in lst]


# ── OPEN / CLOSED anchor poses (degrees, then radians) ────────────────────


HAND_GRASP_OPEN_LEFT_DEG: list[float] = [
    # thumb_roll / thumb_abad OPEN anchors biased into a natural
    # rest pose. History: original upstream values (0°, 10°) put the
    # thumb perpendicular to the palm at rest -- "thumb sticking out
    # to the side", uncanny on the kitchen-task viewer. Bumped to
    # (-12°, 35°) on 2026-05-13 to sit ~halfway across the palm.
    # Bumped again to (-12°, 55°) on 2026-06-03 because operators
    # reported the thumb still wasn't reaching "parallel to the
    # middle finger" mid-curl in controller mode -- pushing
    # thumb_abad from 35 % to 55 % of hardware travel positions the
    # thumb pad near mid-palm at full open so the open->closed lerp
    # sweeps it through proper opposition for any trigger ratio.
    # CLOSED anchors are UNCHANGED so we do NOT push any motor
    # closer to its hardware hardstop -- the only effect is the
    # OPEN -> CLOSED span shrinks for thumb_abad (now 25° travel
    # instead of 45°) and thumb_roll (unchanged at 28°). Iterate
    # visually in the MuJoCo viewer if these need further tuning.
    # See the mirrored values in HAND_GRASP_OPEN_RIGHT_DEG.
    -12.0,  # thumb_roll  (range -50..+10, was 0; pad rolled inward)
    55.0,   # thumb_abad  (range   0..+100, was 35; thumb ~mid-palm, parallel to middle finger)
    -5.0,   # thumb_mcp   (range -49..   0)
    0.0,    # index_abad  (range   0..+12)
    5.0,    # index_pip   (range   0..+90)
    5.0,    # middle_pip
    0.0,    # ring_abad   (range -10..   0)
    5.0,    # ring_pip
    0.0,    # pinky_abad  (range -10..   0)
    5.0,    # pinky_pip
]

HAND_GRASP_CLOSED_LEFT_DEG: list[float] = [
    # thumb_roll / thumb_abad CLOSED anchors expanded to ~80 % of
    # hardware travel (was 50 %). The upstream agitbot
    # quest3-bare-hand-control constants stopped the thumb at
    # half-travel which left the thumb visually "out to the side"
    # even at oppose = 1.0. Pushing the anchors closer to the
    # hardware extremes lets the robot's thumb actually swing
    # across the palm so a thumb-to-fingertip gesture from the
    # operator translates to a thumb-to-fingertip pose on the
    # robot. See data/lerobot/x2_quest3_kinematic_v4 analysis (May
    # 10) for the visual gap that motivated this.
    -40.0,  # thumb_roll  (range -50..+10, was -30; now uses 80 % of 60° travel)
    80.0,   # thumb_abad  (range   0..+100, was 60; now 80 % of 100° travel)
    -40.0,  # thumb_mcp
    6.0,    # index_abad
    # Non-thumb pip CLOSED anchors pushed from 80° to 88° (~98 % of
    # the 0..90° hardware range). Companion to the per-finger
    # ``finger_tip_oppose`` plumbing: even when the receiving
    # finger's PIP motor lerps all the way to its CLOSED anchor on a
    # deliberate thumb-finger touch, the OmniHand fingertip arc is
    # still narrower than the operator's because the human hand has
    # 3 cascaded knuckles per finger (MCP+PIP+DIP, total bend ~240°)
    # vs OmniHand's 1 active PIP + 1 mimic-coupled DIP (~189° at
    # 90° pip). Pushing the anchor to hardware max claws back the
    # last ~9 mm of fingertip travel toward the palm. See
    # docs/source/tutorials/x2_dataset_record_and_replay.md
    # "Open: non-thumb fingertip-to-thumb touch".
    88.0,   # index_pip   (range 0..+90, was 80; now 98 % of 90° travel)
    88.0,   # middle_pip
    -5.0,   # ring_abad
    88.0,   # ring_pip
    -5.0,   # pinky_abad
    88.0,   # pinky_pip
]

HAND_GRASP_OPEN_RIGHT_DEG: list[float] = [
    # See note on HAND_GRASP_OPEN_LEFT_DEG -- mirrored here for the
    # right hand so the resting thumb pose is symmetric.
    12.0,   # thumb_roll  (range -10..+50, was 0; pad rolled inward)
    -55.0,  # thumb_abad  (range -100..0, was -35; thumb ~mid-palm, parallel to middle finger)
    5.0,    # thumb_mcp   (range   0..+49)
    0.0,    # index_abad  (range -12..   0)
    5.0,    # index_pip
    5.0,    # middle_pip
    0.0,    # ring_abad   (range   0..+10)
    5.0,    # ring_pip
    0.0,    # pinky_abad  (range   0..+10)
    5.0,    # pinky_pip
]

HAND_GRASP_CLOSED_RIGHT_DEG: list[float] = [
    # See note on HAND_GRASP_CLOSED_LEFT_DEG -- mirrored here for the
    # right hand so the geometry stays symmetric.
    40.0,   # thumb_roll  (range -10..+50, was 30; now 80 % of 60° travel)
    -80.0,  # thumb_abad  (range -100..0, was -60; now 80 % of 100° travel)
    40.0,   # thumb_mcp
    -6.0,   # index_abad
    88.0,   # index_pip   (range 0..+90, was 80; now 98 % of 90° travel)
    88.0,   # middle_pip
    5.0,    # ring_abad
    88.0,   # ring_pip
    5.0,    # pinky_abad
    88.0,   # pinky_pip
]

HAND_GRASP_OPEN_RAD_LEFT: tuple[float, ...] = tuple(_deg_list_to_rad(HAND_GRASP_OPEN_LEFT_DEG))
HAND_GRASP_CLOSED_RAD_LEFT: tuple[float, ...] = tuple(_deg_list_to_rad(HAND_GRASP_CLOSED_LEFT_DEG))
HAND_GRASP_OPEN_RAD_RIGHT: tuple[float, ...] = tuple(_deg_list_to_rad(HAND_GRASP_OPEN_RIGHT_DEG))
HAND_GRASP_CLOSED_RAD_RIGHT: tuple[float, ...] = tuple(_deg_list_to_rad(HAND_GRASP_CLOSED_RIGHT_DEG))


# (lower, upper) per motor in radians; matches HAND_JOINT_RANGE_*_DEG.
HAND_JOINT_LIMITS_LEFT_RAD: tuple[tuple[float, float], ...] = tuple(
    (_deg_to_rad(lo), _deg_to_rad(hi))
    for lo, hi in [
        (-50.0, 10.0), (0.0, 100.0), (-49.0, 0.0),
        (0.0, 12.0), (0.0, 90.0),
        (0.0, 90.0),
        (-10.0, 0.0), (0.0, 90.0),
        (-10.0, 0.0), (0.0, 90.0),
    ]
)

HAND_JOINT_LIMITS_RIGHT_RAD: tuple[tuple[float, float], ...] = tuple(
    (_deg_to_rad(lo), _deg_to_rad(hi))
    for lo, hi in [
        (-10.0, 50.0), (-100.0, 0.0), (0.0, 49.0),
        (-12.0, 0.0), (0.0, 90.0),
        (0.0, 90.0),
        (0.0, 10.0), (0.0, 90.0),
        (0.0, 10.0), (0.0, 90.0),
    ]
)


# ── Public API ────────────────────────────────────────────────────────────


def grasp_command_from_ratio(side: str, ratio: float) -> np.ndarray:
    """Linearly interpolate between OPEN and CLOSED for a given side.

    Args:
        side: ``"left"`` or ``"right"``.
        ratio: 0.0 (open) .. 1.0 (closed). Clamped to [0, 1].

    Returns:
        ``np.ndarray`` of shape ``(NUM_HAND_DOF_PER_SIDE,)`` -- target
        motor positions in radians, in the canonical motor-axis order
        from :data:`HAND_FINGER_NAMES_PER_SIDE`. Output is clamped to
        the per-motor hardware limits as a final safety belt.
    """
    if side == "left":
        open_q = HAND_GRASP_OPEN_RAD_LEFT
        closed_q = HAND_GRASP_CLOSED_RAD_LEFT
        limits = HAND_JOINT_LIMITS_LEFT_RAD
    elif side == "right":
        open_q = HAND_GRASP_OPEN_RAD_RIGHT
        closed_q = HAND_GRASP_CLOSED_RAD_RIGHT
        limits = HAND_JOINT_LIMITS_RIGHT_RAD
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    r = float(np.clip(ratio, 0.0, 1.0))
    open_arr = np.asarray(open_q, dtype=np.float64)
    closed_arr = np.asarray(closed_q, dtype=np.float64)
    cmd = (1.0 - r) * open_arr + r * closed_arr
    lo = np.array([a for a, _ in limits], dtype=np.float64)
    hi = np.array([b for _, b in limits], dtype=np.float64)
    return np.clip(cmd, lo, hi)


# Mapping from each OmniHand motor (10) to the source finger curl
# index (5: thumb=0, index=1, middle=2, ring=3, pinky=4). Order
# matches HAND_FINGER_NAMES_PER_SIDE.
_MOTOR_TO_FINGER_INDEX: tuple[int, ...] = (
    0,  # thumb_roll
    0,  # thumb_abad
    0,  # thumb_mcp
    1,  # index_abad
    1,  # index_pip
    2,  # middle_pip
    3,  # ring_abad
    3,  # ring_pip
    4,  # pinky_abad
    4,  # pinky_pip
)

# Indices of the thumb motors that are driven by the COMBINED
# ``max(oppose, thumb_flex)`` signal rather than ``thumb_flex``
# alone. Anatomically, all three of the X2 thumb motors (CMC roll,
# CMC abduction, and MCP flexion) need to actuate during BOTH a
# thumb-finger touch (high opposition, often low MCP flex) AND a
# closed fist with the thumb wrapped over the curled fingers (high
# flex, moderate opposition). Quest 3's per-joint thumb signals are
# very biased toward "thumb_flex when fist, oppose when touching",
# so taking the per-frame max of the two channels yields a single
# clean drive that matches both gestures. See
# ``data/lerobot/x2_quest3_kinematic_v4/debug/teleop_episode_000000.npz``
# for the screenshot/analysis (May 10) where the operator's thumb
# touched each fingertip but the robot's ``thumb_mcp`` only reached
# ~22 % closure because raw ``thumb_flex`` peaked at ~0.37 -- folding
# in ``oppose`` (which peaked at ~0.95 on those frames) is what
# brings the robot thumb to the fingertip.
_THUMB_COMBINED_DRIVE_MOTORS: tuple[int, ...] = (
    0,  # thumb_roll
    1,  # thumb_abad
    2,  # thumb_mcp  (added so opposition drives the knuckle bend too)
)
# Backwards-compatible alias: legacy callers / tests imported the
# previous name. Keep it pointing at the new set so behaviour is the
# same (now includes thumb_mcp).
_THUMB_OPPOSITION_MOTORS: tuple[int, ...] = _THUMB_COMBINED_DRIVE_MOTORS

FINGER_CURL_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


# ── Quest 3 hand-prior coupling compensation ─────────────────────────────
#
# Quest 3's XRHand hand-pose estimator has a strong "fingers move
# together" prior that under-reports per-finger curls when the
# operator curls a single finger in isolation. Cross-correlation
# analysis on data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_000000.npz
# (666 hand-mode frames):
#
#   * Pairwise correlations of +0.99-+1.00 between
#     index/middle/ring/pinky curls -- the four fingers effectively
#     move as a single unit in Quest 3's reported joints.
#   * "Isolated curl" probe: when 4 of 5 fingers are <= 0.30, the
#     5th finger NEVER reaches >= 0.30 either; max observed was
#     ~0.29 for index/middle/pinky and ~0.30 for ring (4 frames).
#   * Full fist (all fingers >= 0.8): per-finger curls cap at
#     0.91-0.95, never 1.0.
#
# **Default live behaviour (teleop / dataset recording)** is *linear*
# full-range mapping: each finger's normalised Quest curl in ``[0,
# 1]`` drives the linear interpolation ratio between the OmniHand
# OPEN and CLOSED anchors for that finger's motors -- i.e. the full
# hardware span of each robot finger joint tracks the full reported
# span of the headset finger curl channel. No deadzone and no early
# saturation unless you opt in (see below).
#
# Quest 3's XRHand estimator still has a strong "fingers move
# together" prior: isolated fingers often report well below 1.0 even
# when the operator intends a full close. **Optional** compensation
# applies a piecewise gain on the raw curl signal (enable with
# ``apply_curl_compensation=True``).
#
# Default compensation is **smooth proportional**, NOT bimodal
# (2026-05-13 retune). Concretely the defaults below are
# ``deadzone=0.05, full_threshold=0.95, gamma=1`` for every finger
# and the thumb-opposition signal:
#
#   * Below DEADZONE (0.05): output 0   -- only filters rest-noise
#     jitter on the raw curl channel.
#   * Above FULL_THRESHOLD (0.95): output 1 -- small saturation
#     cushion so the operator doesn't have to hit the literal Quest
#     max to drive the robot to CLOSED.
#   * Linear in between (``gamma = 1``) so half-way curl produces
#     half-way closure.
#
# Why it changed (history matters for anyone reading the tuner
# script's older output):
#
# The previous defaults (``thumb dz=0.25 full=0.27 gamma=5``;
# ``finger dz=0.35 full=0.40 gamma=5``; oppose ``dz=0.25 full=0.40
# gamma=3``) were tuned by maximising bimodality of the post-
# compensation motor-command distribution on the v3 debug episodes
# (see gear_sonic/scripts/tune_finger_curl_compensation.py). That
# job was "decide whether the operator intends to curl this finger
# or not" -- i.e. a binary classifier. With those defaults a
# half-pulled curl produced either 0 % or 100 % closure with a tiny
# ramp window in between; operators describe this as "fingers are
# bang-bang -- they only either open or close, no smooth curling"
# (cf. the dexterous-grasp regression on 2026-05-13). For the
# teleop / dataset-recording use case where the operator already
# has per-finger ``hand_range`` calibration (``floor`` / ``ceiling``
# affine remap, see :func:`normalize_finger_curls` and
# :class:`HandRangeCalibration` in operator_calibration.py), the
# bimodal stretch then composes ON TOP of the proportional
# normalisation and silently destroys it.
#
# Smooth proportional gives the operator continuous control of the
# closure depth: a 50 % normalised curl produces ~50 % robot-finger
# closure regardless of which finger. Bimodal detection is still
# accessible by passing explicit per-finger params -- e.g.
# ``stretch_finger_curls(c, deadzone=0.35, full_threshold=0.40,
# gamma=5)`` recovers the old behaviour exactly. The tuner script
# also still emits its bimodal-tuned numbers for callers who want
# them.
#
# Empirical raw-curl distribution per finger (kept here for the
# tuner script and for anyone re-tuning later):
#
#   finger      p10    p25    p50    p75    p90
#   thumb     0.26   0.31   0.43   0.72   0.89
#   index     0.11   0.16   0.26   0.70   0.84
#   middle    0.11   0.15   0.28   0.76   0.87
#   ring      0.09   0.15   0.28   0.75   0.85
#   pinky     0.12   0.15   0.25   0.73   0.84
#
# The thumb's rest distribution sits ~0.10-0.15 higher than the
# four fingers because (a) Quest3's ``MAX_CURL_ANGLE.thumb = 90 deg``
# vs ``MAX_CURL_ANGLE.fingers = 180 deg`` amplifies small thumb
# bends, and (b) anatomically the thumb at "open hand" rest already
# has moderate MCP flexion. The per-operator
# :class:`HandRangeCalibration` (loaded from the operator
# calibration YAML) absorbs both of those biases via per-finger
# (floor, ceiling) and is the recommended fix for the resting
# offset -- the smooth defaults below are the proportional shaping
# that runs on TOP of (or instead of) that affine remap.
DEFAULT_CURL_DEADZONE: float = 0.05
DEFAULT_CURL_FULL_THRESHOLD: float = 0.95
DEFAULT_CURL_GAMMA: float = 1.0

# Per-finger parameter arrays (5 elements: thumb, index, middle, ring, pinky).
# Same smooth proportional values across all five fingers; per-finger
# arrays remain in the API surface so callers (or a future tuner) can
# differentiate between the thumb and the other four fingers if needed.
DEFAULT_CURL_DEADZONE_PER_FINGER: tuple[float, ...] = (0.05, 0.05, 0.05, 0.05, 0.05)
DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER: tuple[float, ...] = (0.95, 0.95, 0.95, 0.95, 0.95)
DEFAULT_CURL_GAMMA_PER_FINGER: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)

# Live defaults: linear map headset curl → motor lerp ratio (full robot range).
DEFAULT_APPLY_CURL_COMPENSATION: bool = False
DEFAULT_APPLY_OPPOSE_COMPENSATION: bool = False


# ── Thumb-opposition signal stretch ─────────────────────────────────────
#
# The JS-side ``computeThumbOpposition`` (in
# gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html) returns a
# scalar in [0, 1] from a thumb-tip-to-nearest-fingertip distance
# normalised by palm width. The mapping is linear:
#
#   oppose = (0.45 - d_norm) / (0.45 - 0.06)
#
# clamped to [0, 1], where d_norm < 0.06 = "touch" (saturate) and
# d_norm > 0.45 = "far apart" (zero).
#
# At the operator's natural "relaxed open hand" pose the thumb tip
# sits ~3-4 cm from the index fingertip and the palm width is
# ~8-9 cm, so d_norm ~ 0.35-0.45 and the JS-side signal drifts
# around 0.05-0.25 even though the operator's intent is "no
# opposition". The recommended fix for that resting bias is the
# per-operator :class:`HandRangeCalibration.oppose_floor` /
# ``oppose_ceiling`` affine remap (see
# :func:`normalize_thumb_oppose`), which absorbs the operator's
# actual rest distribution and saturation point.
#
# This module's ``stretch_thumb_oppose`` curve is the OPTIONAL
# shaping that composes on top of (or instead of) that
# normalisation -- analogous to ``stretch_finger_curls`` for the
# per-finger channel. Defaults are the same smooth proportional
# values: ``deadzone=0.05, full_threshold=0.95, gamma=1`` (linear
# in the active zone, with a small rest-noise gate at the bottom
# and a small saturation cushion at the top). The previous bimodal
# defaults (``dz=0.25, full=0.40, gamma=3``) collapsed any partial
# opposition into either 0 or 1, which made operators describe the
# behaviour as "thumb is bang-bang" once the per-operator
# ``oppose_*`` calibration was already absorbing the rest bias.
# Pass explicit ``deadzone`` / ``full_threshold`` / ``gamma`` if
# you want the bimodal behaviour back.
DEFAULT_OPPOSE_DEADZONE: float = 0.05
DEFAULT_OPPOSE_FULL_THRESHOLD: float = 0.95
DEFAULT_OPPOSE_GAMMA: float = 1.0


def _broadcast_5(name: str, val: float | tuple[float, ...] | list[float] | np.ndarray) -> np.ndarray:
    """Broadcast a scalar or 5-element sequence to a numpy ``(5,)`` array."""
    arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
    if arr.shape == (1,):
        arr = np.repeat(arr, 5)
    if arr.shape != (5,):
        raise ValueError(f"{name} must be a scalar or (5,) sequence; got shape {arr.shape}")
    return arr


def stretch_finger_curls(
    finger_curls: np.ndarray,
    *,
    deadzone: float | tuple[float, ...] | np.ndarray = DEFAULT_CURL_DEADZONE_PER_FINGER,
    full_threshold: float | tuple[float, ...] | np.ndarray = DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER,
    gamma: float | tuple[float, ...] | np.ndarray = DEFAULT_CURL_GAMMA_PER_FINGER,
) -> np.ndarray:
    """Compensate for Quest 3 hand-prior coupling.

    Maps per-finger curls in ``[deadzone, full_threshold]`` onto
    ``[0, 1]`` via the curve ``out = t**gamma`` where
    ``t = (raw - deadzone) / (full_threshold - deadzone)`` clamped to
    ``[0, 1]``. Values below ``deadzone`` go to 0; values at or above
    ``full_threshold`` go to 1.

    Each of ``deadzone``, ``full_threshold`` and ``gamma`` may be:

    * a scalar -- the same value is applied to all 5 fingers, OR
    * a 5-element sequence in the same order as ``finger_curls``
      (``[thumb, index, middle, ring, pinky]``) for per-finger
      parameters.

    The defaults are per-finger arrays tuned against pooled
    teleop data (see :data:`DEFAULT_CURL_*_PER_FINGER`). Passing a
    single scalar is supported for legacy / unit-test code paths
    that don't differentiate between fingers.

    With ``gamma = 1`` this is a pure linear stretch. With
    ``gamma > 1`` the response is squeezed near the deadzone (small
    raw inputs barely register) and ramps up to saturation -- this
    is the default and matches operator intuition that "slight
    movement should not close the hand".

    Args:
        finger_curls: ``(5,)`` raw XRHand curls in ``[0, 1]``,
            ordered ``[thumb, index, middle, ring, pinky]``.
        deadzone: scalar or ``(5,)`` -- curl below this maps to 0.
        full_threshold: scalar or ``(5,)`` -- curl at/above this maps to 1.
        gamma: scalar or ``(5,)`` -- power-curve exponent (>0).

    Returns:
        ``(5,)`` stretched curls in ``[0, 1]`` matching the input order.
    """
    raw = np.asarray(finger_curls, dtype=np.float64)
    if raw.shape != (5,):
        raise ValueError(f"finger_curls must be (5,); got {raw.shape}")
    dz = _broadcast_5("deadzone", deadzone)
    full = _broadcast_5("full_threshold", full_threshold)
    g = _broadcast_5("gamma", gamma)
    if not ((dz >= 0.0).all() and (dz < full).all() and (full <= 1.0).all()):
        raise ValueError(
            f"need 0 <= deadzone < full_threshold <= 1 element-wise; got "
            f"deadzone={dz}, full_threshold={full}"
        )
    if not (g > 0.0).all():
        raise ValueError(f"gamma must be > 0 element-wise; got {g}")
    span = full - dz
    t = np.clip((raw - dz) / span, 0.0, 1.0)
    return t ** g


def stretch_thumb_oppose(
    oppose: float,
    *,
    deadzone: float = DEFAULT_OPPOSE_DEADZONE,
    full_threshold: float = DEFAULT_OPPOSE_FULL_THRESHOLD,
    gamma: float = DEFAULT_OPPOSE_GAMMA,
) -> float:
    """Apply the same piecewise power-curve stretch to the
    thumb-opposition scalar as :func:`stretch_finger_curls`
    applies to the finger curls.

    The JS-side opposition signal (``computeThumbOpposition`` in
    ``index.html``) is a normalised thumb-tip-to-nearest-fingertip
    proximity score in ``[0, 1]``. At rest hand pose it drifts to
    0.05-0.25 even though the operator isn't intentionally opposing
    -- the stretch suppresses this bleed and saturates clear
    thumb-finger touches.

    Args:
        oppose: scalar in ``[0, 1]`` (will be clamped).
        deadzone / full_threshold / gamma: see :func:`stretch_finger_curls`.

    Returns:
        Stretched scalar in ``[0, 1]``.
    """
    if not (0.0 <= deadzone < full_threshold <= 1.0):
        raise ValueError(
            f"need 0 <= deadzone < full_threshold <= 1; got "
            f"deadzone={deadzone}, full_threshold={full_threshold}"
        )
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0; got {gamma}")
    o = float(np.clip(oppose, 0.0, 1.0))
    span = full_threshold - deadzone
    t = max(0.0, min(1.0, (o - deadzone) / span))
    return t ** gamma


# ── Per-operator hand-range normalization ──────────────────────────────
#
# Quest 3's reported per-finger curls are NOT pinned to ``[0, 1]``
# at the ergonomic extremes. On a single 56.7 s recorded session
# (``data/lerobot/x2_quest3_kinematic_v4/debug/teleop_episode_000000.npz``):
#
#   side    finger     p05    p95   max-observed   gap-at-floor   gap-at-ceiling
#   LEFT    thumb     0.20   0.99    1.00           20 % closure   1 %  open
#   LEFT    index     0.09   0.87    0.91            9 % closure  13 %  open
#   LEFT    middle    0.09   0.88    0.94            9 % closure  12 %  open
#   LEFT    ring      0.08   0.87    0.93            8 % closure  13 %  open
#   LEFT    pinky     0.10   0.85    0.93           10 % closure  15 %  open
#   RIGHT   thumb     0.20   0.99    1.00           20 % closure   1 %  open
#   RIGHT   index     0.06   0.88    0.90            6 % closure  12 %  open
#   ...
#
# Linear-mapping the raw curl directly to the OPEN→CLOSED motor
# lerp leaves the operator's "fully open hand" with up to 20 %
# of the thumb pre-closed and the operator's "full fist" with
# up to 15 % of each finger never reaching the CLOSED anchor.
# This is the systematic bias the user described as
# "robot's hand never fully opens AND never fully closes".
#
# The fix is per-operator, per-finger, per-side range
# normalisation: estimate ``(floor, ceiling)`` from a recorded
# session (or from a brief "hold open / make a fist" calibration
# pose pair) and remap the raw curl into ``[0, 1]`` before the
# motor lerp:
#
#     ratio = clip((raw - floor) / (ceiling - floor), 0, 1)
#     cmd   = lerp(open_anchor, closed_anchor, ratio)
#
# This is a strict superset of the linear baseline (set
# ``floor = 0`` and ``ceiling = 1`` to recover it) and stacks
# cleanly with the optional ``stretch_finger_curls`` shaping by
# applying normalisation FIRST then the stretch on the normalised
# value. The stretch is now off by default; the normaliser is the
# recommended way to fix the open/close gap.
#
# Defaults (population-level priors derived from the v4 recording):
#
#   * thumb floor 0.20, ceiling 0.95 (the thumb sits high at rest;
#     ceiling pulled in to 0.95 to avoid the ~5 % saturation gap
#     past p95).
#   * other fingers floor 0.10, ceiling 0.85 (rest noise floor +
#     "fingers move together" ceiling).
#
# These match the v4 distribution closely but are NOT a substitute
# for per-operator calibration -- they're a sensible fall-back when
# no calibration is loaded. The operator-calibration YAML is the
# preferred source.
DEFAULT_CURL_FLOOR_PER_FINGER: tuple[float, ...] = (0.20, 0.10, 0.10, 0.10, 0.10)
DEFAULT_CURL_CEILING_PER_FINGER: tuple[float, ...] = (0.95, 0.85, 0.85, 0.85, 0.85)

# Live default: do not apply the population-prior normalization
# automatically. Callers must opt in (typically by passing a
# per-operator calibration loaded from YAML). Keeps every code path
# bit-identical to the linear-mapping baseline unless explicitly
# enabled.
DEFAULT_APPLY_CURL_NORMALIZATION: bool = False


def normalize_finger_curls(
    finger_curls: np.ndarray,
    *,
    floor: float | tuple[float, ...] | np.ndarray = DEFAULT_CURL_FLOOR_PER_FINGER,
    ceiling: float | tuple[float, ...] | np.ndarray = DEFAULT_CURL_CEILING_PER_FINGER,
) -> np.ndarray:
    """Remap raw Quest 3 per-finger curls onto ``[0, 1]`` using a
    per-finger ``(floor, ceiling)`` range derived from the operator's
    hand pose extremes.

    For each finger the output is ::

        out = clip((raw - floor) / (ceiling - floor), 0, 1)

    Output ``0`` corresponds to the operator's natural "fully open
    hand" (raw at or below ``floor``) and output ``1`` corresponds to
    the operator's "fullest fist" (raw at or above ``ceiling``).
    Linearity is preserved between the two -- this is a pure affine
    rescale per finger, NOT a deadzone or non-linear stretch.

    Args:
        finger_curls: ``(5,)`` raw curls in ``[0, 1]`` ordered
            ``[thumb, index, middle, ring, pinky]``.
        floor: scalar or ``(5,)`` -- raw curl at "fully open hand".
        ceiling: scalar or ``(5,)`` -- raw curl at "fullest fist".
            Must be strictly greater than ``floor`` element-wise.

    Returns:
        ``(5,)`` normalised curl ratios in ``[0, 1]``.
    """
    raw = np.asarray(finger_curls, dtype=np.float64)
    if raw.shape != (5,):
        raise ValueError(f"finger_curls must be (5,); got {raw.shape}")
    fl = _broadcast_5("floor", floor)
    ce = _broadcast_5("ceiling", ceiling)
    if not (ce > fl).all():
        raise ValueError(
            f"need ceiling > floor element-wise; got floor={fl}, ceiling={ce}"
        )
    span = ce - fl
    return np.clip((raw - fl) / span, 0.0, 1.0)


def normalize_thumb_oppose(
    oppose: float,
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> float:
    """Affine rescale of the JS-side thumb-opposition scalar onto
    ``[0, 1]`` using a ``(floor, ceiling)`` range. Mirrors
    :func:`normalize_finger_curls` for the scalar opposition signal."""
    if not (0.0 <= floor < ceiling <= 1.0):
        raise ValueError(
            f"need 0 <= floor < ceiling <= 1; got floor={floor}, ceiling={ceiling}"
        )
    o = float(np.clip(oppose, 0.0, 1.0))
    return float(np.clip((o - floor) / (ceiling - floor), 0.0, 1.0))


def per_finger_grasp_command_from_curls(
    side: str,
    finger_curls: np.ndarray,
    *,
    apply_curl_compensation: bool = DEFAULT_APPLY_CURL_COMPENSATION,
    curl_floor: float | tuple[float, ...] | np.ndarray | None = None,
    curl_ceiling: float | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    """Map XRHand 5-finger curls onto the 10-DOF OmniHand command.

    Each motor lerps independently along its OPEN -> CLOSED anchor
    interval, driven by the curl of the finger that motor belongs to.
    For example, ``thumb_roll / thumb_abad / thumb_mcp`` all lerp by the
    thumb curl; ``middle_pip`` lerps by the middle curl; etc.

    Args:
        side: ``"left"`` or ``"right"``.
        finger_curls: ``(5,)`` array in ``[0, 1]``, ordered
            ``[thumb, index, middle, ring, pinky]``.
        apply_curl_compensation: if True, apply :func:`stretch_finger_curls`
            to boost isolated-finger closure on Quest 3. Default False:
            linear map raw curl ``[0, 1]`` to motor lerp ratio (full
            OPEN→CLOSED span).
        curl_floor: scalar or ``(5,)`` -- per-finger raw curl at
            "fully open hand". When both this and ``curl_ceiling`` are
            provided, applies :func:`normalize_finger_curls` BEFORE the
            optional ``stretch_finger_curls``. Pass both as ``None``
            (default) to skip normalisation entirely.
        curl_ceiling: scalar or ``(5,)`` -- per-finger raw curl at
            "fullest fist". Must be strictly greater than ``curl_floor``
            element-wise. See ``curl_floor`` above.

    Returns:
        ``(NUM_HAND_DOF_PER_SIDE,)`` motor command in radians, clamped
        to the per-motor hardware limits.
    """
    if side == "left":
        open_q = HAND_GRASP_OPEN_RAD_LEFT
        closed_q = HAND_GRASP_CLOSED_RAD_LEFT
        limits = HAND_JOINT_LIMITS_LEFT_RAD
    elif side == "right":
        open_q = HAND_GRASP_OPEN_RAD_RIGHT
        closed_q = HAND_GRASP_CLOSED_RAD_RIGHT
        limits = HAND_JOINT_LIMITS_RIGHT_RAD
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    curls = np.asarray(finger_curls, dtype=np.float64)
    if curls.shape != (5,):
        raise ValueError(f"finger_curls must be (5,); got {curls.shape}")
    curls = np.clip(curls, 0.0, 1.0)
    if curl_floor is not None or curl_ceiling is not None:
        if curl_floor is None or curl_ceiling is None:
            raise ValueError(
                "curl_floor and curl_ceiling must be provided together "
                "(pass both or neither)."
            )
        curls = normalize_finger_curls(
            curls, floor=curl_floor, ceiling=curl_ceiling
        )
    if apply_curl_compensation:
        curls = stretch_finger_curls(curls)

    open_arr = np.asarray(open_q, dtype=np.float64)
    closed_arr = np.asarray(closed_q, dtype=np.float64)

    cmd = np.empty(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
    for motor_idx in range(NUM_HAND_DOF_PER_SIDE):
        c = curls[_MOTOR_TO_FINGER_INDEX[motor_idx]]
        cmd[motor_idx] = (1.0 - c) * open_arr[motor_idx] + c * closed_arr[motor_idx]

    lo = np.array([a for a, _ in limits], dtype=np.float64)
    hi = np.array([b for _, b in limits], dtype=np.float64)
    return np.clip(cmd, lo, hi)


def per_finger_grasp_command_from_curls_and_oppose(
    side: str,
    finger_curls: np.ndarray,
    thumb_oppose: float | None,
    *,
    finger_tip_oppose: np.ndarray | tuple[float, ...] | None = None,
    apply_curl_compensation: bool = DEFAULT_APPLY_CURL_COMPENSATION,
    apply_oppose_compensation: bool = DEFAULT_APPLY_OPPOSE_COMPENSATION,
    curl_floor: float | tuple[float, ...] | np.ndarray | None = None,
    curl_ceiling: float | tuple[float, ...] | np.ndarray | None = None,
    oppose_floor: float | None = None,
    oppose_ceiling: float | None = None,
) -> np.ndarray:
    """Like :func:`per_finger_grasp_command_from_curls` but with an
    independent thumb-opposition signal driving the thumb motors
    AND an optional per-finger thumb-tip-to-fingertip proximity
    signal driving the four non-thumb fingers.

    The five finger-curl values come from XRHand chain bends and
    capture per-finger flexion well, but two anatomical motions are
    poorly captured by the chain alone:

    * The thumb's CMC opposition (swinging the thumb across the
      palm) is not in the XRHand chain, so the thumb-curl signal
      barely moves on a thumb-finger touch even when the operator's
      thumb is fully opposed. This is fixed by the WebXR client's
      ``computeThumbOpposition`` scalar (see ``index.html``), which
      is fed in as ``thumb_oppose`` and drives all three thumb
      motors via ``max(thumb_oppose, thumb_flex_curl)``.

    * Symmetrically, the OmniHand's non-thumb fingers each have one
      active flexion DOF (``*_pip``) cascading two mimic-coupled
      segments, but the operator's finger has THREE knuckles
      (MCP+PIP+DIP) summing to a much larger total bend. Even after
      per-operator curl normalisation the receiving finger only
      travels ~36 % to CLOSED on a deliberate thumb-finger touch
      because Quest 3's "fingers move together" prior caps isolated
      single-finger curls at ~0.30-0.40 raw. The
      ``finger_tip_oppose`` 4-vector (one entry per non-thumb
      finger, from the WebXR client's ``computeFingerTipOppose``)
      saturates at literal contact (d_norm < 0.06 ≈ 0.5 cm) and
      drives each ``*_pip`` (and the matching ``*_abad`` for index
      / ring / pinky) via ``max(curls[i], finger_tip_oppose[i])``.
      During non-touch frames the proximity signal is near zero
      and ``max`` reduces to the curl, preserving smooth
      proportional control.

    Args:
        side: ``"left"`` or ``"right"``.
        finger_curls: ``(5,)`` array in ``[0, 1]``,
            ``[thumb_flex, index, middle, ring, pinky]``.
        thumb_oppose: scalar in ``[0, 1]`` from XRHand. ``None``
            falls back to thumb-curl-driven opposition (legacy
            behaviour identical to
            :func:`per_finger_grasp_command_from_curls`).
        finger_tip_oppose: ``(4,)`` array in ``[0, 1]`` ordered
            ``[index, middle, ring, pinky]`` -- per-finger
            thumb-tip proximity (saturates at touch, ~0 when far).
            ``None`` (the back-compat default) skips the
            tip-proximity drive entirely. Individual NaN entries
            are tolerated and treated as "fall back to the curl
            signal for that finger only" -- this lets the WebXR
            client emit partial data when only some fingertips
            tracked this frame.

    Returns:
        ``(NUM_HAND_DOF_PER_SIDE,)`` motor command in radians,
        clamped to the per-motor hardware limits.
    """
    if thumb_oppose is None:
        return per_finger_grasp_command_from_curls(
            side, finger_curls,
            apply_curl_compensation=apply_curl_compensation,
            curl_floor=curl_floor,
            curl_ceiling=curl_ceiling,
        )

    if side == "left":
        open_q = HAND_GRASP_OPEN_RAD_LEFT
        closed_q = HAND_GRASP_CLOSED_RAD_LEFT
        limits = HAND_JOINT_LIMITS_LEFT_RAD
    elif side == "right":
        open_q = HAND_GRASP_OPEN_RAD_RIGHT
        closed_q = HAND_GRASP_CLOSED_RAD_RIGHT
        limits = HAND_JOINT_LIMITS_RIGHT_RAD
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    curls = np.asarray(finger_curls, dtype=np.float64)
    if curls.shape != (5,):
        raise ValueError(f"finger_curls must be (5,); got {curls.shape}")
    curls = np.clip(curls, 0.0, 1.0)
    if curl_floor is not None or curl_ceiling is not None:
        if curl_floor is None or curl_ceiling is None:
            raise ValueError(
                "curl_floor and curl_ceiling must be provided together "
                "(pass both or neither)."
            )
        curls = normalize_finger_curls(
            curls, floor=curl_floor, ceiling=curl_ceiling
        )
    if apply_curl_compensation:
        curls = stretch_finger_curls(curls)

    if oppose_floor is not None or oppose_ceiling is not None:
        if oppose_floor is None or oppose_ceiling is None:
            raise ValueError(
                "oppose_floor and oppose_ceiling must be provided together "
                "(pass both or neither)."
            )
        oppose = normalize_thumb_oppose(
            thumb_oppose, floor=oppose_floor, ceiling=oppose_ceiling
        )
    elif apply_oppose_compensation:
        oppose = stretch_thumb_oppose(thumb_oppose)
    else:
        oppose = float(np.clip(thumb_oppose, 0.0, 1.0))

    open_arr = np.asarray(open_q, dtype=np.float64)
    closed_arr = np.asarray(closed_q, dtype=np.float64)

    # Drive thumb opposition motors from max(oppose, thumb_flex_curl).
    # Anatomically, the thumb's CMC opposition motors swing through
    # their full travel during BOTH a thumb-finger touch (high
    # ``oppose``, possibly low flex) AND a closed fist with the
    # thumb wrapped over the curled fingers (high flex, but the
    # thumb-tip lands near palm centre giving only moderate
    # ``oppose`` ~0.5-0.6). Without this fold-in, a fist gesture in
    # hand-tracking mode drives ``thumb_mcp`` to CLOSED but leaves
    # ``thumb_roll`` / ``thumb_abad`` stuck at half-travel, which
    # the operator perceives as "thumb abad barely moves in hand
    # mode" while the same fist gesture in controller mode (where
    # one uniform trigger value drives all three motors) closes the
    # whole thumb cleanly. See
    # ``data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_000000.npz``
    # for the empirical evidence: hand-mode thumb_abad max=0.52
    # vs controller-mode thumb_abad max=1.00 with the same closed
    # fists. Taking the max of the two signals matches anatomy and
    # gives parity with controller mode without breaking the
    # thumb-finger-touch case (where ``oppose`` >= ``thumb_flex``
    # already, so ``max`` returns ``oppose`` unchanged).
    thumb_flex = curls[0]
    oppose_motor_signal = max(oppose, thumb_flex)

    # Per-finger combined drive for the four non-thumb fingers:
    #   tip_drive[i] = curls[i+1]                          if no tip_oppose
    #   tip_drive[i] = max(curls[i+1], tip_oppose[i])      if tip_oppose finite
    #   tip_drive[i] = curls[i+1]                          if tip_oppose[i] is NaN
    #
    # The 4-vector is ordered [index, middle, ring, pinky]; entry i
    # drives both the matching ``*_abad`` (index/ring/pinky) and the
    # matching ``*_pip``. middle has no abad motor. Driving abad on
    # the same combined signal as pip means a deliberate
    # thumb-to-finger-tip touch closes the receiving finger fully
    # (CLOSED anchor includes a small abad component for "fist
    # convergence" -- 6° radial for index, -5° ulnar for ring /
    # pinky). When the operator just curls the four fingers
    # without touching the thumb, ``tip_oppose`` is near zero and
    # the ``max`` reduces to the curl signal so smooth proportional
    # control is preserved.
    tip_drive = curls[1:5].copy()
    if finger_tip_oppose is not None:
        tip_arr = np.asarray(finger_tip_oppose, dtype=np.float64)
        if tip_arr.shape != (4,):
            raise ValueError(
                f"finger_tip_oppose must be (4,); got {tip_arr.shape}"
            )
        finite = np.isfinite(tip_arr)
        if finite.any():
            clipped = np.clip(tip_arr[finite], 0.0, 1.0)
            tip_drive[finite] = np.maximum(tip_drive[finite], clipped)

    cmd = np.empty(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
    for motor_idx in range(NUM_HAND_DOF_PER_SIDE):
        if motor_idx in _THUMB_COMBINED_DRIVE_MOTORS:
            c = oppose_motor_signal
        else:
            finger_idx = _MOTOR_TO_FINGER_INDEX[motor_idx]
            # finger_idx is 1..4 here (thumb motors took the branch
            # above). Map to tip_drive index 0..3.
            c = tip_drive[finger_idx - 1]
        cmd[motor_idx] = (1.0 - c) * open_arr[motor_idx] + c * closed_arr[motor_idx]

    lo = np.array([a for a, _ in limits], dtype=np.float64)
    hi = np.array([b for _, b in limits], dtype=np.float64)
    return np.clip(cmd, lo, hi)


def controller_grasp_ratio(
    left_trigger: float,
    right_trigger: float,
    left_grip: float,
    right_grip: float,
    *,
    mode: str = "trigger",
) -> tuple[float, float]:
    """Pick (left_ratio, right_ratio) in [0, 1] from controller analog inputs.

    Args:
        left_trigger / right_trigger / left_grip / right_grip: 0..1
            analog values from the Quest 3 controllers (already
            clamped on the WebXR side).
        mode: ``"trigger"`` (index trigger; default), ``"grip"``
            (middle-finger grip), or ``"max"`` (whichever is greater
            this frame).
    """
    def _pick(t: float, g: float) -> float:
        if mode == "trigger":
            v = t
        elif mode == "grip":
            v = g
        elif mode == "max":
            v = max(t, g)
        else:
            raise ValueError(f"mode must be trigger|grip|max, got {mode!r}")
        return float(np.clip(v, 0.0, 1.0))

    return _pick(left_trigger, left_grip), _pick(right_trigger, right_grip)


__all__ = [
    "DEFAULT_APPLY_CURL_COMPENSATION",
    "DEFAULT_APPLY_CURL_NORMALIZATION",
    "DEFAULT_APPLY_OPPOSE_COMPENSATION",
    "DEFAULT_CURL_CEILING_PER_FINGER",
    "DEFAULT_CURL_DEADZONE",
    "DEFAULT_CURL_DEADZONE_PER_FINGER",
    "DEFAULT_CURL_FLOOR_PER_FINGER",
    "DEFAULT_CURL_FULL_THRESHOLD",
    "DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER",
    "DEFAULT_CURL_GAMMA",
    "DEFAULT_CURL_GAMMA_PER_FINGER",
    "DEFAULT_OPPOSE_DEADZONE",
    "DEFAULT_OPPOSE_FULL_THRESHOLD",
    "DEFAULT_OPPOSE_GAMMA",
    "FINGER_CURL_NAMES",
    "HAND_FINGER_NAMES_PER_SIDE",
    "HAND_GRASP_CLOSED_RAD_LEFT",
    "HAND_GRASP_CLOSED_RAD_RIGHT",
    "HAND_GRASP_OPEN_RAD_LEFT",
    "HAND_GRASP_OPEN_RAD_RIGHT",
    "HAND_JOINT_LIMITS_LEFT_RAD",
    "HAND_JOINT_LIMITS_RIGHT_RAD",
    "NUM_HAND_DOF_PER_SIDE",
    "controller_grasp_ratio",
    "grasp_command_from_ratio",
    "normalize_finger_curls",
    "normalize_thumb_oppose",
    "per_finger_grasp_command_from_curls",
    "per_finger_grasp_command_from_curls_and_oppose",
    "stretch_finger_curls",
    "stretch_thumb_oppose",
]
