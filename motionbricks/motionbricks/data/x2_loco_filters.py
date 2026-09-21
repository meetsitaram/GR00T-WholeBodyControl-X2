"""Filter X2 motion-lib keys to walk-and-turn locomotion only.

Scope: manipulation tasks that need the robot to walk around a room and
reach — not stylistic gaits, running, crawling, or object-interaction clips.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# Keys must match at least one include pattern (case-insensitive).
#
# 2026-06-25: Replaced ``\b`` with letter-only boundaries
# ``(?<![A-Za-z])`` / ``(?![A-Za-z])``. Python's ``\b`` treats underscore
# as a word character, so ``\bwalk\b`` could not match
# ``Loop_Forward_Walk_001`` (the source-corpus naming convention).
# Letter-boundaries match across underscores / digits while still
# preventing partial-word matches (e.g. ``walker``, ``boardwalk``).
DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = (
    r"(?i)(?<![A-Za-z])(walk|stride|pace|gait)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(forward|backward|back_step|fwd_step|fwd|back)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(sideway|sideways|lateral|straf|side_step|side)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(turn|pivot|rotate)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(idle|stand|rest|stationary|relaxed)(?![A-Za-z])",
    r"(?i)^loco(?![A-Za-z])",  # e.g. locowalk__, loco__
)

# Reject even if an include matched.
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"(?i)(?<![A-Za-z])(run|sprint|jog|dash)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(crawl|kneel|crouch|squat|sit|lying|prone)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(jump|hop|leap|vault|fall)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(box|punch|kick|dance|zombie|injured|stealth|happy|scared|gun)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(pick|place|grasp|carry|lift|throw|push|pull|open|close)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(climb|stair|ladder|handstand|cartwheel)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(manip|object|prop|tool|sword|bench|chair)(?![A-Za-z])",
    r"(?i)(?<![A-Za-z])(stoop|bend|reach|wave|gesture|point)(?![A-Za-z])",
    r"(?i)standing(?![A-Za-z])",  # standing-manipulation subset
)


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def motion_key_passes_loco_filter(
    key: str,
    *,
    include_patterns: Sequence[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> bool:
    """Return True if ``key`` is kept for X2 walk/turn MotionBricks training."""
    inc = compile_patterns(include_patterns)
    exc = compile_patterns(exclude_patterns)
    if not any(p.search(key) for p in inc):
        return False
    if any(p.search(key) for p in exc):
        return False
    return True


def filter_motion_keys(
    keys: Iterable[str],
    *,
    include_patterns: Sequence[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[str]:
    return [
        k
        for k in keys
        if motion_key_passes_loco_filter(
            k,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    ]
