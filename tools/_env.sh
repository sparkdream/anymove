# Shared by the tools/*.sh scripts: run from the repo root, find Python and Blender.
CALLER_DIR=$PWD
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
BLENDER=${BLENDER:-blender}
# A path given relative to where the script was called from, made absolute.
from_caller() { (cd "$CALLER_DIR" && realpath -m "$1"); }
# need_creature CREATURE: an example's name (assets/NAME, outputs in runs/NAME, prompts in prompts/NAME) or the path to
# a creature folder anywhere, i.e. anything with a / in it (outputs in FOLDER/runs, prompts in FOLDER/prompts).
# Sets A (assets), R (outputs), P (prompts) and C (creature.json "name", else the folder's name).
need_creature() {
  local arg=${1:-}
  if [[ $arg == */* ]]; then
    A=$(from_caller "$arg"); R=$A/runs; P=$A/prompts
  else
    A=assets/$arg; R=runs/$arg; P=prompts/$arg
  fi
  [ -n "$arg" ] || { echo "give a creature: an example's name or a folder with a creature.json" >&2; exit 1; }
  [ -f "$A/creature.json" ] || { echo "no $A/creature.json" >&2; exit 1; }
  C=$($PY -c "import json, sys; print(json.load(open(sys.argv[1])).get('name') or sys.argv[2])" "$A/creature.json" "$(basename "$A")")
  export MOCAP_CREATURE=$A
}
