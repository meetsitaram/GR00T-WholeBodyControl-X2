"""Auto-install missing optional runtime dependencies on first launch.

Why this exists
---------------

The X2 teleop / dataset-recorder workflow has a couple of optional
dependencies that aren't strictly required for every code path
(``gtts`` for the Quest 3 calibration audio prompts is the main
one). Listing them in ``gear_sonic[data_collection]`` is the right
home for documentation, but we don't want a fresh checkout to fail
silently the first time an operator runs the calibration -- they'd
get the (poorly-supported) ``speechSynthesis`` fallback and conclude
"it doesn't work" before noticing the warning.

So the entry-point scripts call :func:`ensure_runtime_deps` *before*
they touch the calibration / recorder code. The first run pip-installs
the missing wheels into the active interpreter; subsequent runs are
no-ops because the modules are importable.

Design choices
--------------

* **No silent installs by default.** We print exactly what we're about
  to install and into which Python interpreter. Operators have been
  burned by tools that quietly mutate their venv; the verbose log
  here is a feature.
* **Graceful failure.** If pip fails (no network, locked env, etc.)
  the function returns ``False`` instead of raising. The caller then
  decides whether to abort or continue with degraded behavior.
* **Opt-out via env var.** Set ``GEAR_SONIC_NO_AUTO_INSTALL=1`` to
  disable the auto-installer entirely (useful in CI / Docker /
  reproducible-build settings where the env should be locked).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys


def _is_importable(module_name: str) -> bool:
    """Return True if ``module_name`` can be imported in the current
    interpreter (without actually importing it -- avoids triggering
    expensive side-effect-laden imports).
    """
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return False
    return spec is not None


def _resolve_install_cmd(specs: list[str], *, quiet: bool) -> tuple[list[str], str]:
    """Pick the best installer available for the current interpreter.

    Returns ``(cmd_list, label)``. Strategy:

    1. **Stdlib pip** if importable in this interpreter
       (``python -m pip install``) -- the canonical path.
    2. **uv pip** with explicit ``--python <sys.executable>`` if the
       ``uv`` binary is on PATH -- handles ``uv venv`` environments
       that ship without stdlib pip.
    3. Raise ``RuntimeError`` so the caller can fall back to the
       manual-install message.
    """
    if _is_importable("pip"):
        cmd = [sys.executable, "-m", "pip", "install"]
        if quiet:
            cmd.append("-q")
        cmd.extend(specs)
        return cmd, "pip"

    uv_bin = shutil.which("uv")
    if uv_bin is not None:
        cmd = [uv_bin, "pip", "install", "--python", sys.executable]
        if quiet:
            cmd.append("-q")
        cmd.extend(specs)
        return cmd, "uv pip"

    raise RuntimeError(
        "no installer available: stdlib pip is not importable in "
        f"{sys.executable} and no `uv` binary on PATH"
    )


def ensure_runtime_deps(
    packages: dict[str, str],
    *,
    purpose: str = "the teleop / recorder workflow",
    interactive: bool = False,
    quiet: bool = False,
) -> bool:
    """Pip-install every package in ``packages`` whose import name is
    not already importable.

    Args:
        packages: ``{import_name: pip_spec}``. ``import_name`` is the
            module the script will eventually ``import``;
            ``pip_spec`` is whatever you'd write after ``pip install``
            (e.g. ``"gtts>=2.5"``, ``"my-pkg @ git+https://…"``).
        purpose: human-readable phrase explaining WHY these are needed
            (used in the log message).
        interactive: if True, prompt the user before installing.
            Default False -- the caller scripts run unattended.
        quiet: pass ``-q`` to pip.

    Returns:
        True if all packages are importable when the function exits
        (either they already were, or the pip install succeeded).
        False on installation failure -- caller decides whether to
        abort.
    """
    if os.environ.get("GEAR_SONIC_NO_AUTO_INSTALL", "").strip() == "1":
        # Honour the opt-out: just report what's missing and bail.
        missing = [pip for mod, pip in packages.items() if not _is_importable(mod)]
        if missing:
            print(
                f"[runtime-deps] GEAR_SONIC_NO_AUTO_INSTALL=1 set; "
                f"{len(missing)} package(s) for {purpose} are missing: "
                f"{missing}. Install manually or unset the env var.",
                flush=True,
            )
            return False
        return True

    missing: dict[str, str] = {
        mod: pip for mod, pip in packages.items() if not _is_importable(mod)
    }
    if not missing:
        return True

    print(
        f"[runtime-deps] {len(missing)} optional package(s) for "
        f"{purpose} are missing: {sorted(missing.values())}. "
        f"Auto-installing into {sys.executable}.",
        flush=True,
    )

    if interactive and sys.stdin.isatty():
        try:
            answer = input("[runtime-deps] proceed? [Y/n] ").strip().lower()
        except EOFError:
            answer = "y"
        if answer not in ("", "y", "yes"):
            print(
                "[runtime-deps] aborted by user; the script may run with "
                "degraded functionality (e.g. silent calibration audio).",
                flush=True,
            )
            return False

    specs = list(missing.values())
    try:
        cmd, label = _resolve_install_cmd(specs, quiet=quiet)
    except RuntimeError as exc:
        # Neither pip nor uv is available; no automatic remediation.
        print(
            f"[runtime-deps] {exc}. "
            f"Install the missing packages manually:\n"
            f"    {sys.executable} -m pip install {' '.join(specs)}\n"
            f"  (or)  uv pip install --python {sys.executable} {' '.join(specs)}",
            flush=True,
        )
        return False

    print(
        f"[runtime-deps] using {label}: "
        f"{' '.join(cmd)}",
        flush=True,
    )
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(
            f"[runtime-deps] {label} install failed (exit {exc.returncode}). "
            f"Continuing with degraded functionality. Re-run manually:\n"
            f"    {' '.join(cmd)}",
            flush=True,
        )
        return False
    except FileNotFoundError:
        print(
            f"[runtime-deps] could not invoke {label} ({cmd[0]}). "
            f"Install the missing packages manually:\n"
            f"    {sys.executable} -m pip install {' '.join(specs)}\n"
            f"  (or)  uv pip install --python {sys.executable} {' '.join(specs)}",
            flush=True,
        )
        return False

    # Verify the installs took effect (catches the "wheel installed but
    # different python on PATH" trap).
    still_missing = [mod for mod in missing if not _is_importable(mod)]
    if still_missing:
        print(
            f"[runtime-deps] WARNING: still cannot import {still_missing} "
            f"after install. The wheels may have landed in a different "
            f"interpreter than {sys.executable}.",
            flush=True,
        )
        return False

    print(
        f"[runtime-deps] installed {len(missing)} package(s) successfully.",
        flush=True,
    )
    return True


# Canonical dep maps for the X2 VR teleop / dataset-recorder flow.
# Import-name -> pip-spec. Keep aligned with
# gear_sonic[data_collection] extras and requirements-teleop-record.txt.
#
# Two groups so we don't force a 200 MB ``datasets`` / ``av`` /
# ``lerobot`` install on operators who only run the calibration step
# (which happens BEFORE recording, sometimes on a separate machine).

# Calibration-only deps. Small, fast, low-risk.
CALIBRATION_DEPS: dict[str, str] = {
    "gtts": "gtts>=2.5",
}

# Full recorder deps. Includes the LeRobot dataset writer chain.
# These are heavy (gigabytes when ``av`` and torch are not already
# installed), so the recorder script auto-installs them only when the
# operator actually starts recording.
RECORDER_DEPS: dict[str, str] = {
    **CALIBRATION_DEPS,
    "datasets": "datasets==3.6.0",
    "av": "av>=14.2",
    "lerobot": (
        "lerobot @ git+https://github.com/huggingface/lerobot.git"
        "@a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
    ),
}

# Deprecated alias kept for one release so external callers don't
# break. Use ``CALIBRATION_DEPS`` or ``RECORDER_DEPS`` directly.
TELEOP_RECORD_DEPS = CALIBRATION_DEPS


__all__ = [
    "CALIBRATION_DEPS",
    "RECORDER_DEPS",
    "TELEOP_RECORD_DEPS",
    "ensure_runtime_deps",
]
