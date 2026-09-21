"""Pre-recorded audio prompts for the Quest 3 calibration overlay.

Why this module exists
----------------------

The Quest 3 Browser has a flaky ``speechSynthesis`` implementation: TTS
needs to be primed inside a user gesture (``Start VR`` click), the
priming sometimes lapses when the immersive-ar session takes over the
audio focus, and on some headsets it never emits any sound at all even
though the engine reports success. We can't ship a calibration UX that
"hopefully" speaks.

Workaround: ship pre-rendered MP3 files for each calibration prompt and
play them from the WebXR client via a regular ``<audio>`` element.
``<audio>`` playback is robust in immersive-ar because it doesn't
depend on the speech engine -- it routes through the same audio path
as YouTube, Spotify, etc., which the headset always plumbs through to
the speakers.

The MP3 files are generated on-demand using gTTS (the only Python TTS
package that doesn't depend on system services) and cached in
``gear_sonic/utils/teleop/vr/quest3_webxr_app/audio/``. They are
generated lazily by :func:`ensure_prompt_audio_files` -- the
:class:`Quest3Reader` calls it on startup so first-run users get them
without an extra step.

Voice / language choices are intentional defaults; expose them on the
generator if you ever need a different voice for accessibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


# Map of audio "key" -> spoken text. Keys are stable identifiers the
# WebXR client uses to decide which file to play. Adding a new key
# requires regenerating the MP3 cache (delete the audio/ folder).
PROMPT_TEXTS: dict[str, str] = {
    # Pose-show prompts.
    "show_arms_down": (
        "Pose 1 of 4. Stand relaxed with both arms hanging fully straight down "
        "at your sides. Do not bend your elbows. Palms face inward toward your "
        "thighs. Hold the controllers the same way you will during teleop. "
        "Press A on either controller when ready."
    ),
    "show_t_pose": (
        "Pose 2 of 4. Raise both arms straight out sideways, parallel to the "
        "floor. Palms face down toward the floor. Press A when steady."
    ),
    "show_arms_forward": (
        "Pose 3 of 4. Hold both arms straight out forward at shoulder height. "
        "Keep your hands close together, about as wide as your shoulders. "
        "Palms face each other. Press A when steady."
    ),
    "show_namaste": (
        "Pose 4 of 4. Bring both palms together at your chest in a namaste "
        "pose. Forearms vertical, palms touching. Press A when steady."
    ),
    # Capture-status prompts.
    "captured": "Captured.",
    "moved_too_much": "Wrist moved too much. Hold steadier and press A again.",
    # Recapture prompts (same family as show, different lead-in).
    "recapture_arms_down": (
        "Recapture arms down pose. Hold both arms fully straight down at your "
        "sides with palms facing inward toward your thighs, and press A again."
    ),
    "recapture_t_pose": (
        "Recapture t-pose. Stretch both arms straight sideways with palms "
        "facing down, and press A again."
    ),
    "recapture_arms_forward": (
        "Recapture arms-forward pose. Hold both arms straight forward at "
        "shoulder height with palms facing each other, and press A again."
    ),
    "recapture_namaste": (
        "Recapture namaste pose. Bring both palms together at your chest and "
        "press A again."
    ),
    # Done.
    "calibration_saved": "Calibration saved successfully.",
    "calibration_failed": "Calibration could not be saved. Please try again.",
    # Audio sanity check.
    "audio_test": (
        "Audio test successful. Calibration prompts will speak through this "
        "audio device."
    ),
    # Manager UX prompts: mode transitions and episode lifecycle. These
    # fire on every chord / single-button press the operator makes, so
    # the prompts are intentionally short (sub-second) -- a long prompt
    # would still be playing when the operator's next press lands and
    # we'd queue up a backlog of overlapping voice cues.
    "mode_off":              "Off.",
    "mode_locomotion":       "Locomotion.",
    # Operator e-stop gesture (A+X + rapid triggers, 2026-08-03).
    # Spoken the moment the SOFT phase trips so the operator gets
    # immediate confirmation the gesture registered (the robot
    # freezing to idle stand can be mistaken for a hang), and again
    # on escalation to pure damping.
    "estop_activating":      "Emergency stop activating.",
    "estop_damping":         "Emergency stop. Pure damping engaged.",
    "mode_arm_manipulation": "Arm manipulation.",
    # Played in addition to ``mode_arm_manipulation`` when the operator
    # crosses LOCOMOTION -> ARM_MANIPULATION while holding a non-neutral
    # waist pose (>= 1 deg on any axis). Tells the operator the planner
    # has frozen their lean / twist so they can drive arms without
    # losing the upper-body pose. See
    # ``Quest3ManagerX2._on_mode_transition`` for the latch logic.
    "mode_torso_locked":     "Torso locked.",
    # R-thumbstick-click toggles the waist freeze independent of mode.
    # ``torso_frozen`` plays on freeze ON (operator wants the body to
    # stay at the current lean / twist while the right stick moves
    # freely); ``torso_released`` plays on freeze OFF (right stick
    # resumes driving the waist live).
    "torso_frozen":          "Torso frozen.",
    "torso_released":        "Torso released.",
    "record_start":          "Recording.",
    "record_save":           "Saved.",
}


# Manager-driven prompt keys (mode + recording lifecycle). Exported as a
# tuple so callers can iterate without hard-coding the names; tests use
# this to verify the WebXR client + audio cache stay in sync with the
# server-side enums.
MANAGER_PROMPT_KEYS: tuple[str, ...] = (
    "estop_activating",
    "estop_damping",
    "mode_off",
    "mode_locomotion",
    "mode_arm_manipulation",
    "mode_torso_locked",
    "torso_frozen",
    "torso_released",
    "record_start",
    "record_save",
)


def audio_dir() -> Path:
    """Where the cached MP3 files live (served by the WebXR HTTP server)."""
    return Path(__file__).resolve().parent / "quest3_webxr_app" / "audio"


def filename_for(key: str) -> str:
    return f"{key}.mp3"


def ensure_prompt_audio_files(*, force_regenerate: bool = False) -> dict[str, Path]:
    """Generate and cache MP3 prompts.

    Idempotent: existing files are kept unless ``force_regenerate`` is
    set. Returns a ``key -> path`` map for callers that want to log /
    serve the files explicitly.

    Falls back gracefully: if gTTS or the network are unavailable, the
    function logs a warning and returns the keys that *do* exist on
    disk. The WebXR client falls back to ``speechSynthesis`` for any
    missing key.
    """
    out_dir = audio_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    needs_generation: list[str] = []
    for key in PROMPT_TEXTS:
        path = out_dir / filename_for(key)
        if path.is_file() and not force_regenerate:
            paths[key] = path
        else:
            needs_generation.append(key)

    if not needs_generation:
        return paths

    try:
        from gtts import gTTS  # type: ignore
    except ImportError:
        log.warning(
            "gTTS not installed; prompt audio cache will be incomplete. "
            "Run `pip install gtts` and re-run the calibration to enable "
            "headset audio prompts. The WebXR client will fall back to "
            "speechSynthesis (less reliable on Quest 3)."
        )
        return paths

    log.info(
        "[quest3-audio] generating %d prompt MP3 files in %s …",
        len(needs_generation), out_dir,
    )
    for key in needs_generation:
        text = PROMPT_TEXTS[key]
        path = out_dir / filename_for(key)
        try:
            tts = gTTS(text, lang="en", slow=False)
            tts.save(str(path))
            paths[key] = path
        except Exception as exc:  # network failure, gTTS API change, …
            log.warning(
                "[quest3-audio] failed to generate %s: %s", path.name, exc
            )

    return paths


__all__ = [
    "MANAGER_PROMPT_KEYS",
    "PROMPT_TEXTS",
    "audio_dir",
    "filename_for",
    "ensure_prompt_audio_files",
]
