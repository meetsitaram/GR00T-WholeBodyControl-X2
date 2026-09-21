"""Install-time utilities (auto pip-install for optional runtime deps)."""

from .runtime_deps import (
    CALIBRATION_DEPS,
    RECORDER_DEPS,
    TELEOP_RECORD_DEPS,
    ensure_runtime_deps,
)

__all__ = [
    "CALIBRATION_DEPS",
    "RECORDER_DEPS",
    "TELEOP_RECORD_DEPS",
    "ensure_runtime_deps",
]
