#!/bin/bash
# Per move: send the reference frame and MOVE.txt in CREATURE's prompts to video models, one clip per seed.
# tools/film.sh CREATURE MOVE [models=h3max] [seeds=1,2]     -> videos/MOVE/s<seed>/<model>.mp4 in CREATURE's outputs
# Needs FAL_KEY. MiniMax H3 Max is about $0.26 a 5 s clip (October 2026); models: h3, h3max, h3maxturbo, seedance, flux3max.
# wan22 and wan22-14b (Wan 2.2 TI2V-5B and I2V-A14B, open weights) run on this machine's GPU instead: no key, no
# cost, one seed at a time.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
m=${2:?move}; models=${3:-h3max}; seeds=${4:-1,2}
[ -f "$P/$m.txt" ] || { echo "write $P/$m.txt first (tools/make_prompt.py)" >&2; exit 1; }
for s in ${seeds//,/ }; do
  if [[ ",$models," == *,wan22* ]]; then  # one local GPU: one seed after another
    $PY -m mocap.generate "$R/ref/ref.png" "$P/$m.txt" "$R/videos/$m/s$s" "$models" 5 "$s"
  else
    $PY -m mocap.generate "$R/ref/ref.png" "$P/$m.txt" "$R/videos/$m/s$s" "$models" 5 "$s" &
  fi
done
wait
ls "$R"/videos/$m/s*/*.mp4
