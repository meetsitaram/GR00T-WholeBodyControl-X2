"""Factory function to instantiate a configured X2 Ultra RobotModel from URDF."""

from pathlib import Path
from typing import Literal

from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
    HandVariant,
    X2UltraSupplementalInfo,
)


def instantiate_x2_ultra_robot_model(
    hand_variant: Literal["omnihand_10", "g1_compat_7"] = "omnihand_10",
):
    """
    Instantiate an AgiBot X2 Ultra robot model with the given hand variant.

    Args:
        hand_variant: ``"omnihand_10"`` for the full 10-DOF AgiBot OmniHand
            (v0 default) or ``"g1_compat_7"`` for a G1-compatible 7-DOF
            subset used in cross-embodiment evaluation.

    Returns:
        RobotModel: Configured X2 Ultra robot model whose Pinocchio chain
        contains the 31 body DOFs only. Hand DOFs live in the supplemental
        info as opaque names + limits and are exercised via the AimDK HAL
        rather than the URDF.

    Notes:
        Asset path: ``gear_sonic/data/assets/robot_description/urdf/x2_ultra``.
        URDF: ``x2_ultra.urdf`` (31 revolute joints).
    """
    asset_root = (
        Path(__file__).resolve().parent.parent
        / "../assets/robot_description/urdf/x2_ultra"
    ).resolve()
    urdf_path = asset_root / "x2_ultra.urdf"

    if hand_variant not in ("omnihand_10", "g1_compat_7"):
        raise ValueError(
            f"Invalid hand_variant: {hand_variant!r}. "
            "Must be 'omnihand_10' or 'g1_compat_7'."
        )

    hand_variant_enum = {
        "omnihand_10": HandVariant.OMNIHAND_10,
        "g1_compat_7": HandVariant.G1_COMPAT_7,
    }[hand_variant]

    supplemental_info = X2UltraSupplementalInfo(hand_variant=hand_variant_enum)

    return RobotModel(
        str(urdf_path),
        str(asset_root),
        supplemental_info=supplemental_info,
    )
