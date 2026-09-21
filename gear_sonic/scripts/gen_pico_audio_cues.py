#!/usr/bin/env python3
"""Generate the Pico teleop room-audio cues (gear_sonic/data/audio/cue_<name>.wav).

Operator 2026-09-05: "i also need audio for all those — you can get them from the quest vr
stack (locomotion / arm-manipulation / record started / recording stopped / etc). we need to
create a new one for whole-body-control mode." The Quest WebXR app's mp3s
(gear_sonic/utils/teleop/vr/quest3_webxr_app/audio/) are converted 1:1 (same voice) for the
cues that exist there; the Pico-only cues are synthesized with gTTS (same recipe as the
teleop_engaged/disengaged.wav clips). Idempotent: existing files are kept unless --force.

    .venv/bin/python gear_sonic/scripts/gen_pico_audio_cues.py [--force]
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "gear_sonic/data/audio"
QUEST = REPO / "gear_sonic/utils/teleop/vr/quest3_webxr_app/audio"
# cue name -> Quest mp3 stem (reuse the Quest voice)
FROM_QUEST = {"mode_off": "mode_off", "mode_locomotion": "mode_locomotion",
              "mode_arm_manipulation": "mode_arm_manipulation", "estop_activating": "estop_activating",
              "estop_damping": "estop_damping", "record_save": "record_save", "captured": "captured"}
# Pico-only cues -> spoken text (gTTS)
NEW = {"mode_whole_body": "Whole body control", "record_started": "Recording started",
       "record_stopped": "Recording stopped, saved", "replay_started": "Replay started",
       "replay_done": "Replay finished", "deadman_on": "Sticks live", "micro_step": "Micro step",
       "b_needs_trigger": "Hold the trigger with B", "not_recording": "Not recording",
       "already_recording": "Already recording"}


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ar", "22050", "-ac", "1", str(dst)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, stem in FROM_QUEST.items():
        dst = OUT / f"cue_{name}.wav"
        if dst.exists() and not a.force: print(f"keep  {dst.name}"); continue
        to_wav(QUEST / f"{stem}.mp3", dst); print(f"quest {dst.name}")
    from gtts import gTTS
    with tempfile.TemporaryDirectory() as td:
        for name, text in NEW.items():
            dst = OUT / f"cue_{name}.wav"
            if dst.exists() and not a.force: print(f"keep  {dst.name}"); continue
            mp3 = Path(td) / f"{name}.mp3"; gTTS(text=text, lang="en", slow=False).save(str(mp3))
            to_wav(mp3, dst); print(f"gtts  {dst.name}  \"{text}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
