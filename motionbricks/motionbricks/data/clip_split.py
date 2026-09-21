"""Long-clip windowing for X2 planner training data.

NOTE ON ``subsample`` (2026-08-07): the ``subsample`` argument threaded through
here is an X2-port artifact with no upstream equivalent — MotionBricks states
its datasets are 30 FPS and never resamples. It existed only because our direct
retarget once produced 120 FPS clips. That corpus has since been resampled once,
offline, to a true 30 FPS (``x2_direct_motionlib_30fps``), so clips are now
self-describing (``fps: 30``) and ``subsample=1`` is correct by construction.
Treat any call site passing subsample > 1 as suspect, and prefer resampling the
corpus over striding at load time: a wrong stride is silent — the motion simply
trains N times too fast and nothing downstream notices.

The SONIC recipe splits long motions into training-window-sized clips.
Our pipeline previously DISCARDED any clip whose effective length
(post-subsample) exceeded ``max_frames`` (default 200) — silently dropping
long teleop walks and turn sessions, exactly the material the planner
needs for in-place turns and turning walks (found 2026-08-02).

``window_bounds`` computes consecutive raw-frame windows whose effective
length is ``max_frames``, keeping a tail only if it still clears
``min_frames``. ``slice_clip`` slices every leading-T array field of a
motion-lib clip dict (dof, root_trans_offset, root_rot, pose_aa,
smpl_joints, ...) and passes scalars/metadata through untouched.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def window_bounds(
    n_raw: int, subsample: int, min_frames: int, max_frames: int
) -> List[Tuple[str, int, int]]:
    """Return [(suffix, start_raw, end_raw)] windows for one clip.

    * in-band clips  -> [("", 0, n_raw)] (identity; caller keeps the key)
    * short clips    -> [] (below min_frames — unchanged behavior)
    * long clips     -> consecutive ``__winN`` windows of max_frames
                        effective; tail kept iff >= min_frames.
    """
    sub = max(1, int(subsample))
    n_eff = (n_raw + sub - 1) // sub
    if n_eff < min_frames:
        return []
    if n_eff <= max_frames:
        return [("", 0, n_raw)]
    win_raw = max_frames * sub
    out: List[Tuple[str, int, int]] = []
    idx = 0
    for s in range(0, n_raw, win_raw):
        e = min(s + win_raw, n_raw)
        seg_eff = ((e - s) + sub - 1) // sub
        if seg_eff < min_frames:
            break  # short tail: drop (previous windows already cover bulk)
        out.append((f"__win{idx}", s, e))
        idx += 1
    return out


def slice_clip(clip: dict, s: int, e: int) -> dict:
    """Slice every leading-T ndarray field of a motion-lib clip dict."""
    n_raw = int(clip["dof"].shape[0])
    out = {}
    for k, v in clip.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_raw:
            out[k] = v[s:e]
        else:
            out[k] = v
    return out
