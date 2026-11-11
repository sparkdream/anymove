#!/bin/bash
# Per clip: read every joint in both views, fit the rig, and write the animation as a GLB with renders and a
# side-by-side comparison of the AI clip and the rig.
# tools/capture.sh CREATURE VIDEO.mp4 NAME [views=persp,side]     -> clips/NAME/ in CREATURE's outputs
#   det.npz (joints read), overlay.mp4, motion_raw.json (the fit), motion.json (the fit cleaned up by mocap.clean),
#   render/animation.glb, showcase.mp4
# REF=frame.png: the clip's first frame was another reference (e.g. a posed one from render_ref.py --pose)
# SOLVE="--acc 0.002": extra mocap.solve options (a lighter smoothness for fast moves)
# CLEAN="--no-settle": extra mocap.clean options (--off keeps the raw fit)
# SPEED=2: play the motion that much faster (a move filmed in slow motion reads better; the poses stay, the frame
#   rate goes up). The showcase still compares frame by frame, at the clip's speed.
# BODY3D=1: depth from the front view alone (mocap.body3d, SAM 3D Body) instead of the video's side half;
#   BODY3D="--keep-side" keeps the side reading where it agrees. det_reader.npz keeps the reader's own reading.
#   BODY3D_PYTHON: the Python that has sam_3d_body (default ~/venvs/sam3d-xpu/bin/python)
#   Unless SOLVE or CLEAN are given, the arms are smoothed lightly and the body firmly (a slash crosses the body in
#   a frame or two), and clean's occlusion hold and smoothing are off: both act where the reader's confidence dips,
#   which a fast arm causes, so they froze or blunted the slash (SAM still sees the arm, and the solver smooths).
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
video=$(from_caller "${2:?video}"); name=${3:?name}; views=${4:-persp,side}
ref=$(from_caller "${REF:-$R/ref/ref.png}")
rig=$A/rigged; out=$R/clips/$name
# the 3/4 showcase camera: creature.json "camera" (frame it to the creature's biggest move; see anymove-capture)
read -r cam target lens < <($PY -c "import json; c = json.load(open('$A/creature.json')).get('camera', {}); print(c.get('cam', '-3.2,-3.0,1.6'), c.get('target', '0,0,0.5'), c.get('lens', 50))")
mkdir -p "$out"
$PY -m mocap.detect "$video" "$R/reader.pt" "$ref" "$out/det.npz" "$out/overlay.mp4" | tail -1
if [ -n "${BODY3D:-}" ]; then
  mv "$out/det.npz" "$out/det_reader.npz"
  b3d=${BODY3D#1}
  ${BODY3D_PYTHON:-$HOME/venvs/sam3d-xpu/bin/python} -m mocap.body3d "$video" "$out/det_reader.npz" "$out/det.npz" ${b3d:-}
  SOLVE=${SOLVE:-"--acc 0.02 --acc-arms 0.0001 --vel-arms 0.2"}
  CLEAN=${CLEAN:-"--no-occlusion --no-smooth"}
fi
$PY -m mocap.solve "$out/det.npz" "$rig/rig_info.json" "$out/motion_raw.json" ${SOLVE:-} | grep -v "iter"
$PY -m mocap.clean "$out/det.npz" "$rig/rig_info.json" "$out/motion_raw.json" "$out/motion.json" ${CLEAN:-}
if [ "${SPEED:-1}" != 1 ]; then
  $PY -c "import json, sys; p = sys.argv[1]; m = json.load(open(p)); m['speed'] = float(sys.argv[2]); m['fps'] *= m['speed']
json.dump(m, open(p, 'w'))" "$out/motion.json" "$SPEED"
  echo "played ${SPEED}x faster"
fi
$BLENDER -b -P blender/apply_motion.py -- "$rig/rigged.blend" "$rig/rig_info.json" "$out/motion.json" "$out/render" \
  --views "$views" --samples 24 --cam "$cam" --target "$target" --lens "$lens" --glb \
  $([ -f "$rig/springs.json" ] && echo --springs "$rig/springs.json") > "$out/blender.log" 2>&1
$PY -m mocap.showcase "$video" "$out/det.npz" "$out/render" "$out/showcase.mp4" "$C $name"
if command -v ffmpeg > /dev/null; then
  ffmpeg -loglevel error -y -i "$out/showcase.mp4" -c:v libx264 -crf 20 -pix_fmt yuv420p "$out/showcase_h264.mp4"
fi
echo "done: $out (animation: $out/render/animation.glb)"
