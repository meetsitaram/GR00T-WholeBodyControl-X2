"""Joystick deadzone, deflection mapping, and yaw accumulation for VR teleop.

These utilities live here (rather than alongside the G1-specific
``LocomotionMode`` enum in ``gear_sonic/utils/teleop/common.py``) so the
new X2 manager (``quest3_manager_x2.py``) can share them with the
existing G1 manager (`the upstream G1 Quest manager`) without
pulling in any G1-only enums or wire formats.

For backward compatibility, ``YawAccumulator`` and ``JOYSTICK_DEADZONE``
continue to be re-exported from
:mod:`gear_sonic.utils.teleop.common`, so existing call sites keep
working with no edits required.

Design references
-----------------

The deadzone math (radial deadzone with rescale to ``[0, 1]`` above
threshold) is taken verbatim from
``Quest3PlannerStreamer.run_once()`` in the original G1 manager (lines
~159-178 of `the upstream G1 Quest manager`); preserving the
behavior is required for the manager refactor to be a pure
extract-method change.
"""

from __future__ import annotations

import numpy as np


# Radial deadzone applied to all joystick magnitudes. Tuned in the G1
# manager script; do NOT change without coordinating with both managers
# AND the Quest 3 jitter analysis in
# ``gear_sonic/utils/teleop/finger_signal_filter.py``.
JOYSTICK_DEADZONE: float = 0.15


def apply_radial_deadzone(
    raw_mag: float,
    deadzone: float = JOYSTICK_DEADZONE,
) -> float:
    """Apply a radial deadzone with linear rescale above the threshold.

    Args:
        raw_mag: Raw joystick magnitude in ``[0, 1]`` (typically
            ``np.hypot(lx, ly)``).
        deadzone: Magnitude below which the output is clamped to 0.

    Returns:
        Normalized magnitude in ``[0, 1]``: 0 inside the deadzone,
        ``(raw_mag - deadzone) / (1 - deadzone)`` clamped to ``[0, 1]``
        outside it.
    """
    raw_mag = float(np.clip(raw_mag, 0.0, 1.0))
    if abs(raw_mag) < deadzone:
        return 0.0
    mag = (raw_mag - deadzone) / (1.0 - deadzone)
    if mag > 1.0:
        mag = 1.0
    return mag


class YawAccumulator:
    """Accumulates yaw heading angle based on right-stick X input.

    Mirrors the implementation that previously lived in
    ``gear_sonic.utils.teleop.common`` (still re-exported from there for
    back-compat). The integration uses the same ``-rx`` sign convention
    as the original so call sites that pass ``rx`` directly from the
    Quest 3 reader continue to produce the same heading.
    """

    def __init__(
        self,
        yaw_gain: float = 1.5,
        deadzone: float = JOYSTICK_DEADZONE,
    ) -> None:
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self) -> None:
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [
                np.cos(self.yaw_angle_rad),
                np.sin(self.yaw_angle_rad),
                0.0,
            ]
        return self.heading


__all__ = [
    "JOYSTICK_DEADZONE",
    "apply_radial_deadzone",
    "YawAccumulator",
]
