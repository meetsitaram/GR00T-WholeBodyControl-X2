"""Registry of :class:`EmbodimentConfig` instances keyed by short robot name.

Concrete embodiments live in sibling modules (``x2.py``, ``g1.py``)
that auto-register their configs on import. Consumers (kinematic replay /
teleop CLIs) call :func:`get_embodiment` with the value of ``--robot`` to
look up the right factory bundle.

To add a new robot:

1. Write ``gear_sonic/utils/embodiment/<name>.py`` that builds an
   :class:`EmbodimentConfig` and calls :func:`register_embodiment` at
   module scope.
2. Import the new module from
   :file:`gear_sonic/utils/embodiment/__init__.py` so registration runs
   on first import of the package.
"""

from __future__ import annotations

from typing import Dict

from gear_sonic.utils.embodiment.config import EmbodimentConfig


__all__ = [
    "get_embodiment",
    "register_embodiment",
    "registered_embodiments",
]


_REGISTRY: Dict[str, EmbodimentConfig] = {}


def register_embodiment(cfg: EmbodimentConfig) -> None:
    """Insert (or overwrite) an embodiment config in the registry.

    Idempotent: re-registering the same name overwrites the previous
    entry so tests can stub robots cheaply. Production code only calls
    this once per robot, at module-import time.
    """
    if not isinstance(cfg, EmbodimentConfig):
        raise TypeError(
            f"register_embodiment expects an EmbodimentConfig; got {type(cfg).__name__}"
        )
    _REGISTRY[cfg.name] = cfg


def get_embodiment(name: str) -> EmbodimentConfig:
    """Return the registered :class:`EmbodimentConfig` for ``name``.

    Raises:
        KeyError: if ``name`` is unknown to the registry.

    Notes:
        Stub embodiments (e.g. ``g1`` until a real config is written)
        are registered but their factories raise
        :class:`NotImplementedError` when actually invoked. That keeps
        ``get_embodiment("g1")`` cheap (no model load) while still
        producing a clear error at the point a caller tries to build a
        MuJoCo model.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown embodiment {name!r}. Registered: {known}. "
            f"Add a new gear_sonic/utils/embodiment/<name>.py module."
        ) from exc


def registered_embodiments() -> tuple[str, ...]:
    """Return the sorted tuple of currently-registered embodiment names."""
    return tuple(sorted(_REGISTRY.keys()))
