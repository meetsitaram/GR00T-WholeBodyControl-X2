"""Per-side stateful smoother for Quest 3 hand inputs.

Pipeline applied independently per side::

    raw input  ->  EMA(alpha)  ->  deadband-hold  ->  filtered output

Each per-side stream consists of 10 scalar channels:

    * 5 finger curls         [thumb, index, middle, ring, pinky]
    * 1 thumb-oppose
    * 4 finger-tip-oppose    [index, middle, ring, pinky]

Why this filter (and not e.g. a one-euro filter)
================================================

Empirical analysis on the v5/ep1 recording
(``data/lerobot/x2_quest3_kinematic_v5/debug/teleop_episode_000001.npz``,
35.9 s, 50 Hz):

* **Held-pose tremor.** Per-finger curl std on a 6.8 s static-pose
  segment was 0.003-0.012 (= 0.4-0.8° peak motor jitter at the X2
  OmniHand anchors). The noise spectrum sits in the 1-2 Hz band,
  which is **also where intentional motion lives**, so a 1st-order
  low-pass cutoff < 1 Hz costs ~200 ms of motion lag without
  measurably reducing the tremor.
* **Spike nature.** Of frames with raw ``|d/dt| > 0.05`` on the
  curl signal, **0 % returned to baseline on the next frame** -
  every spike was real intentional motion. A median or one-euro
  filter would mostly add lag without removing real noise.
* **Touch onsets are fast.** ``finger_tip_oppose`` has 3-frame
  ramp times on a touch event. Smoothing too aggressively would
  delay robot finger closure on a deliberate thumb-fingertip
  gesture, undoing the v0.5 fix.

The right shape, given those numbers, is:

1. **A light EMA (alpha = 0.5)** to take the edge off single-frame
   noise spikes (~20 % p99 reduction on motion-edge deltas) and
   add only ~20 ms of motion lag.
2. **A deadband-hold on top of the EMA**: when the rolling
   per-channel std stays below ``hold_std`` for ``hold_window``
   frames, latch the output to the rolling median of that window
   so static-pose tremor goes to ZERO. Release the latch the
   moment the operator genuinely starts moving (rolling std rises
   above ``release_std``, OR the new input differs from the
   latched value by more than ``release_disp``).

Calibration of the deadband thresholds against v5/ep1 ``LEFT``
hand-curls:

* ``hold_std`` = 0.005    sits between rest noise (0.003-0.012)
                          and motion onset (~0.05).
* ``release_std`` = 0.012 ~ p99 of the held-pose noise; first
                          actual motion in the recording crossed
                          0.025-0.04 within 1-2 frames.
* ``release_disp`` = 0.020 ~ 2 % range of the [0,1] signal; the
                          smallest "intentional" motion observed
                          on the v5 data crossed this within 2-3
                          frames.
* ``hold_window`` = 8 frames (160 ms) at 50 Hz: long enough to
                          band-pass through the held-pose ~1-2 Hz
                          noise, short enough that motion onset
                          is responsive.

Tuned to be a strict superset of "do nothing" (set
``ema_alpha = 1.0`` and ``hold_std = 0`` to recover identity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── Channel layout (per side) ──────────────────────────────────────────
# This filter is agnostic to channel meaning -- it operates on any 10
# stream-of-scalars. The constants below are documentation for the
# expected packing into the 10-vector that ``FingerSignalFilter`` consumes.
NUM_CURL_CHANNELS: int = 5             # [thumb, index, middle, ring, pinky]
NUM_TIP_OPPOSE_CHANNELS: int = 4       # [index, middle, ring, pinky]
NUM_THUMB_OPPOSE_CHANNELS: int = 1
NUM_TOTAL_CHANNELS: int = (
    NUM_CURL_CHANNELS + NUM_THUMB_OPPOSE_CHANNELS + NUM_TIP_OPPOSE_CHANNELS
)
assert NUM_TOTAL_CHANNELS == 10, "filter assumes 10 per-side channels"

# Slices into the 10-vec exposed by ``pack_signal`` /
# ``unpack_signal`` so callers can stay layout-agnostic.
CURL_SLICE: slice = slice(0, 5)
THUMB_OPPOSE_SLICE: slice = slice(5, 6)
TIP_OPPOSE_SLICE: slice = slice(6, 10)


# ── Defaults (calibrated against v5/ep1, see module docstring) ─────────
DEFAULT_EMA_ALPHA: float = 0.5
DEFAULT_HOLD_WINDOW: int = 8           # 160 ms at 50 Hz
DEFAULT_HOLD_STD: float = 0.005
DEFAULT_RELEASE_STD: float = 0.012
DEFAULT_RELEASE_DISP: float = 0.020


@dataclass(frozen=True)
class FingerFilterParams:
    """Tuneable filter parameters; defaults are the v5-calibrated values."""
    ema_alpha: float = DEFAULT_EMA_ALPHA
    hold_window: int = DEFAULT_HOLD_WINDOW
    hold_std: float = DEFAULT_HOLD_STD
    release_std: float = DEFAULT_RELEASE_STD
    release_disp: float = DEFAULT_RELEASE_DISP

    def validate(self) -> None:
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1]; got {self.ema_alpha}. "
                "Use 1.0 to disable EMA."
            )
        if self.hold_window < 2:
            raise ValueError(
                f"hold_window must be >= 2 frames; got {self.hold_window}. "
                "Use 0 to disable the deadband (set hold_std = 0 instead)."
            )
        if self.hold_std < 0.0 or self.release_std < 0.0 or self.release_disp < 0.0:
            raise ValueError(
                f"hold_std / release_std / release_disp must be >= 0; "
                f"got hold_std={self.hold_std}, release_std={self.release_std}, "
                f"release_disp={self.release_disp}"
            )
        if self.release_std < self.hold_std:
            raise ValueError(
                f"release_std ({self.release_std}) must be >= hold_std "
                f"({self.hold_std}) for the hysteresis to make sense."
            )


def pack_signal(
    curls: np.ndarray | None,
    thumb_oppose: float | None,
    finger_tip_oppose: np.ndarray | None,
) -> np.ndarray:
    """Pack the three Quest-3 streams into a single ``(10,)`` vector.

    ``None`` inputs produce NaN entries on the corresponding slice; the
    filter treats NaN as "input unavailable for this frame".
    """
    out = np.full(NUM_TOTAL_CHANNELS, np.nan, dtype=np.float64)
    if curls is not None:
        c = np.asarray(curls, dtype=np.float64)
        if c.shape != (NUM_CURL_CHANNELS,):
            raise ValueError(
                f"curls must be ({NUM_CURL_CHANNELS},); got {c.shape}"
            )
        out[CURL_SLICE] = c
    if thumb_oppose is not None:
        out[THUMB_OPPOSE_SLICE] = float(thumb_oppose)
    if finger_tip_oppose is not None:
        t = np.asarray(finger_tip_oppose, dtype=np.float64)
        if t.shape != (NUM_TIP_OPPOSE_CHANNELS,):
            raise ValueError(
                f"finger_tip_oppose must be ({NUM_TIP_OPPOSE_CHANNELS},); "
                f"got {t.shape}"
            )
        out[TIP_OPPOSE_SLICE] = t
    return out


def unpack_signal(
    vec: np.ndarray,
) -> tuple[np.ndarray | None, float | None, np.ndarray | None]:
    """Inverse of :func:`pack_signal`. NaN slices return ``None``."""
    if vec.shape != (NUM_TOTAL_CHANNELS,):
        raise ValueError(f"vec must be (10,); got {vec.shape}")
    curls = vec[CURL_SLICE]
    if np.isnan(curls).any():
        curls_out: np.ndarray | None = None
    else:
        curls_out = curls.copy()
    thumb_val = float(vec[THUMB_OPPOSE_SLICE][0])
    thumb_out: float | None = None if np.isnan(thumb_val) else thumb_val
    tip = vec[TIP_OPPOSE_SLICE]
    # tip_oppose may have per-finger NaN even when other entries are finite
    # (one fingertip dropped out this frame). Return the array as-is so
    # downstream callers can fall back to the curl signal per finger.
    if np.isnan(tip).all():
        tip_out: np.ndarray | None = None
    else:
        tip_out = tip.copy()
    return curls_out, thumb_out, tip_out


class _PerChannelEMA:
    """1st-order low-pass with NaN-tolerant carry-forward.

    On a NaN input the filter holds its previous state for that
    channel (so a brief tracking dropout doesn't reset the smoothed
    value). Channels that are NaN at first call stay NaN until the
    first finite sample arrives.
    """

    def __init__(self, alpha: float, n_channels: int) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self._alpha = float(alpha)
        self._state = np.full(n_channels, np.nan, dtype=np.float64)

    def update(self, x: np.ndarray) -> np.ndarray:
        """Returns the filtered ``x``; carries NaN inputs through state."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != self._state.shape:
            raise ValueError(
                f"input shape {x_arr.shape} != filter shape "
                f"{self._state.shape}"
            )
        new_state = self._state.copy()
        for c in range(x_arr.shape[0]):
            x_c = x_arr[c]
            if np.isnan(x_c):
                # hold last good value (already in new_state[c])
                continue
            if np.isnan(new_state[c]):
                # cold start on this channel
                new_state[c] = x_c
            else:
                new_state[c] = self._alpha * x_c + (1.0 - self._alpha) * new_state[c]
        self._state = new_state
        return new_state.copy()

    def reset(self) -> None:
        self._state = np.full_like(self._state, np.nan)


class _PerChannelDeadbandHold:
    """Latch each channel to the rolling median while motion stays below threshold.

    State per channel:

    * ``held[c] = False`` (motion mode): output = input.
    * ``held[c] = True``  (held mode):   output = rolling median of
      the last ``hold_window`` finite samples.

    Mode transitions:

    * **enter held**: rolling std over the last ``hold_window`` frames
      drops below ``hold_std``.
    * **exit held**: rolling std rises above ``release_std`` OR the
      new sample differs from the rolling median by more than
      ``release_disp``.

    Why rolling median (not a frozen "entry value"):

    A frozen-at-entry value cannot track slow drift -- if the
    operator slightly relaxes their hand over several seconds, the
    raw signal smoothly drifts but a frozen latch would either
    (a) release on displacement and re-latch at a discretely
    different median (stair-step output worse than raw) or
    (b) release on displacement then stay free (motion mode jitter).

    Outputting the rolling median fixes both: it follows slow drift
    naturally (the median of a sliding window slides too) while
    still rejecting single-frame impulse noise (one outlier in 8
    samples can't move the median).

    The hysteresis (``release_std`` >= ``hold_std``) prevents
    chattering when the noise level sits near the threshold. The
    displacement check (``release_disp``) catches sub-window
    impulse motions that haven't yet shown up in the std.

    NaN inputs:

    * In motion mode: output is NaN.
    * In held mode:   output is the most recent rolling median (the
      held value survives a brief NaN dropout, which is the
      usually-desirable "hold last good pose during a 1-3 frame
      XRHand re-acquire").
    """

    def __init__(
        self,
        n_channels: int,
        *,
        window: int,
        hold_std: float,
        release_std: float,
        release_disp: float,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2; got {window}")
        self._n = int(n_channels)
        self._window = int(window)
        self._hold_std = float(hold_std)
        self._release_std = float(release_std)
        self._release_disp = float(release_disp)

        # circular buffer of the last `window` inputs (one row per frame)
        self._buffer = np.full((self._window, self._n), np.nan, dtype=np.float64)
        self._head = 0          # next write index
        self._size = 0          # frames currently in buffer (capped at window)

        self._held = np.zeros(self._n, dtype=bool)
        # ``_held_value`` mirrors the most recent rolling median while
        # latched, NaN otherwise. Used for NaN-input bridging.
        self._held_value = np.full(self._n, np.nan, dtype=np.float64)

    def update(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != (self._n,):
            raise ValueError(f"input shape {x_arr.shape} != ({self._n},)")

        # Push x into the circular buffer (NaN ok; ignored in std/median).
        self._buffer[self._head] = x_arr
        self._head = (self._head + 1) % self._window
        self._size = min(self._size + 1, self._window)

        # Warm-up: until the buffer is full, pass-through.
        if self._size < self._window:
            return x_arr.copy()

        # Per-channel rolling stats (NaN-aware).
        std_per_channel = np.full(self._n, np.nan, dtype=np.float64)
        median_per_channel = np.full(self._n, np.nan, dtype=np.float64)
        valid = ~np.isnan(self._buffer)
        # Need at least half the window of valid samples to trust the stats.
        valid_count = valid.sum(axis=0)
        usable = valid_count >= max(2, self._window // 2)
        for c in range(self._n):
            if not usable[c]:
                continue
            vals = self._buffer[valid[:, c], c]
            std_per_channel[c] = float(vals.std())
            median_per_channel[c] = float(np.median(vals))

        out = np.empty(self._n, dtype=np.float64)
        for c in range(self._n):
            x_c = x_arr[c]

            # NaN input handling: hold value if in held mode, else NaN.
            if np.isnan(x_c):
                out[c] = self._held_value[c] if self._held[c] else np.nan
                continue

            std_c = std_per_channel[c]
            med_c = median_per_channel[c]

            if self._held[c]:
                # Currently latched -- check exit conditions against the
                # rolling median (which tracks slow drift).
                stats_unavailable = np.isnan(std_c) or np.isnan(med_c)
                if stats_unavailable:
                    # Window has too few finite samples to trust stats:
                    # release the latch defensively.
                    self._held[c] = False
                    self._held_value[c] = np.nan
                    out[c] = x_c
                    continue
                disp = abs(x_c - med_c)
                exit_motion = (
                    std_c > self._release_std or disp > self._release_disp
                )
                if exit_motion:
                    self._held[c] = False
                    self._held_value[c] = np.nan
                    out[c] = x_c
                else:
                    # Track slow drift via the rolling median.
                    self._held_value[c] = med_c
                    out[c] = med_c
            else:
                # Currently free -- check entry conditions.
                if (
                    not np.isnan(std_c)
                    and not np.isnan(med_c)
                    and std_c < self._hold_std
                ):
                    self._held[c] = True
                    self._held_value[c] = med_c
                    out[c] = med_c
                else:
                    out[c] = x_c
        return out

    @property
    def held_mask(self) -> np.ndarray:
        return self._held.copy()

    def reset(self) -> None:
        self._buffer[:] = np.nan
        self._head = 0
        self._size = 0
        self._held[:] = False
        self._held_value[:] = np.nan


class FingerSignalFilter:
    """Per-side stateful smoother for Quest 3 hand inputs.

    Build one instance per ``side`` ("left" / "right") and call
    :meth:`update` once per teleop tick. The filter is internally
    stateful (EMA + rolling buffer for the deadband), so do NOT
    share across sides or threads.

    Example::

        filt_left  = FingerSignalFilter()
        filt_right = FingerSignalFilter()

        # in the per-frame loop:
        l_curls, r_curls, _, _ = quest.get_hand_curls()
        l_oppose, r_oppose     = quest.get_thumb_opposition()
        l_tip, r_tip           = quest.get_finger_tip_oppose()

        l_curls_f, l_oppose_f, l_tip_f = filt_left.update(
            l_curls, l_oppose, l_tip,
        )
        r_curls_f, r_oppose_f, r_tip_f = filt_right.update(
            r_curls, r_oppose, r_tip,
        )

        # ... feed _f outputs into the retargeter.
    """

    def __init__(self, params: FingerFilterParams | None = None) -> None:
        self._params = params or FingerFilterParams()
        self._params.validate()
        self._ema = _PerChannelEMA(
            alpha=self._params.ema_alpha,
            n_channels=NUM_TOTAL_CHANNELS,
        )
        self._deadband = _PerChannelDeadbandHold(
            n_channels=NUM_TOTAL_CHANNELS,
            window=self._params.hold_window,
            hold_std=self._params.hold_std,
            release_std=self._params.release_std,
            release_disp=self._params.release_disp,
        )

    def update(
        self,
        curls: np.ndarray | None,
        thumb_oppose: float | None,
        finger_tip_oppose: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float | None, np.ndarray | None]:
        """Apply the filter to a single frame's worth of inputs.

        Returns ``(curls_filtered, thumb_oppose_filtered,
        finger_tip_oppose_filtered)``. ``None`` is preserved for any
        input that was ``None`` AND whose deadband state is "free" --
        a held-mode channel survives a brief NaN dropout and returns
        the latched value instead of ``None``.
        """
        x = pack_signal(curls, thumb_oppose, finger_tip_oppose)
        y_ema = self._ema.update(x)
        y_db = self._deadband.update(y_ema)
        return unpack_signal(y_db)

    def reset(self) -> None:
        """Clear all internal state. Useful between recording episodes."""
        self._ema.reset()
        self._deadband.reset()

    @property
    def params(self) -> FingerFilterParams:
        return self._params


def filter_npz_offline(
    curls: np.ndarray,
    thumb_oppose: np.ndarray,
    finger_tip_oppose: np.ndarray,
    *,
    params: FingerFilterParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the per-side filter to recorded NPZ arrays in one pass.

    Used by ``replay_recorded_dataset.py`` to retroactively filter
    recordings that were captured before the live filter landed.
    Stateless (creates a fresh :class:`FingerSignalFilter`); call
    once per side.

    Args:
        curls: ``(N, 5)`` numpy array, may contain NaN rows.
        thumb_oppose: ``(N,)``  numpy array.
        finger_tip_oppose: ``(N, 4)``  numpy array.
        params: filter params (defaults to :class:`FingerFilterParams`).

    Returns:
        ``(curls_filt, thumb_oppose_filt, finger_tip_oppose_filt)`` --
        same shapes as inputs. NaN rows in the input remain NaN in
        the output unless the deadband had a latched value at that
        frame (in which case the latched value is filled in).
    """
    n = curls.shape[0]
    if thumb_oppose.shape != (n,):
        raise ValueError(
            f"thumb_oppose must be (N,); got {thumb_oppose.shape}"
        )
    if finger_tip_oppose.shape != (n, NUM_TIP_OPPOSE_CHANNELS):
        raise ValueError(
            f"finger_tip_oppose must be (N, 4); got {finger_tip_oppose.shape}"
        )

    filt = FingerSignalFilter(params)
    curls_out = np.full_like(curls, np.nan)
    thumb_out = np.full_like(thumb_oppose, np.nan)
    tip_out = np.full_like(finger_tip_oppose, np.nan)

    for t in range(n):
        c_in = None if np.isnan(curls[t]).any() else curls[t]
        o_in = None if np.isnan(thumb_oppose[t]) else float(thumb_oppose[t])
        # finger_tip_oppose can have per-finger NaN; only convert to
        # None when the entire row is NaN. Otherwise pass through and
        # let pack_signal copy the NaNs into the right slots.
        if np.isnan(finger_tip_oppose[t]).all():
            t_in: np.ndarray | None = None
        else:
            t_in = finger_tip_oppose[t]
        c_f, o_f, t_f = filt.update(c_in, o_in, t_in)
        if c_f is not None:
            curls_out[t] = c_f
        if o_f is not None:
            thumb_out[t] = o_f
        if t_f is not None:
            tip_out[t] = t_f
    return curls_out, thumb_out, tip_out


__all__ = [
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_HOLD_STD",
    "DEFAULT_HOLD_WINDOW",
    "DEFAULT_RELEASE_DISP",
    "DEFAULT_RELEASE_STD",
    "FingerFilterParams",
    "FingerSignalFilter",
    "NUM_CURL_CHANNELS",
    "NUM_THUMB_OPPOSE_CHANNELS",
    "NUM_TIP_OPPOSE_CHANNELS",
    "NUM_TOTAL_CHANNELS",
    "filter_npz_offline",
    "pack_signal",
    "unpack_signal",
]
