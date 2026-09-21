#!/usr/bin/env bash
# Convert a clip to a README gif the way the upstream media/ gifs are cut:
# two-pass palette, 400 px wide (800 for a full-width slot), 12 fps.
#   tools/video_to_gif.sh <in.mp4|.mov> <out.gif> [width=400] [fps=12] [start_s] [dur_s]
set -euo pipefail
IN="$1"; OUT="$2"; W="${3:-400}"; FPS="${4:-12}"; SS="${5:-}"; T="${6:-}"
CUT=(); [[ -n "$SS" ]] && CUT+=(-ss "$SS"); [[ -n "$T" ]] && CUT+=(-t "$T")
FILT="fps=${FPS},scale=${W}:-1:flags=lanczos"
ffmpeg -v error -y "${CUT[@]}" -i "$IN" -vf "${FILT},palettegen=stats_mode=diff" /tmp/_pal_$$.png
ffmpeg -v error -y "${CUT[@]}" -i "$IN" -i /tmp/_pal_$$.png -lavfi "${FILT} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" "$OUT"
rm -f /tmp/_pal_$$.png
echo "$OUT: $(du -h "$OUT" | cut -f1) $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "$OUT")"
