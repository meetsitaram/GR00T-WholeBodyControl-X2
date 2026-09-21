"""Embodiment registry for the kinematic-replay CLI.

Importing this package auto-registers every concrete embodiment
defined in this directory (X2 today, G1 stub today, more in the
future). External consumers should import from this package, not from
the per-robot modules:

.. code-block:: python

    from gear_sonic.utils.embodiment import EmbodimentConfig, get_embodiment

    cfg = get_embodiment("x2")
    model, layout, body_qposadr = cfg.build_kinematic_model(with_omnihand=True)
"""

from __future__ import annotations

from gear_sonic.utils.embodiment.config import EmbodimentConfig
from gear_sonic.utils.embodiment.registry import (
    get_embodiment,
    register_embodiment,
    registered_embodiments,
)

# Side-effect imports: registering the concrete embodiments.
from gear_sonic.utils.embodiment import x2 as _x2  # noqa: F401
from gear_sonic.utils.embodiment import g1 as _g1  # noqa: F401


__all__ = [
    "EmbodimentConfig",
    "get_embodiment",
    "register_embodiment",
    "registered_embodiments",
]
