#!/usr/bin/env bash
# Build the X2 motion banks (pad/dance bank = the shipped X2 MC stock gestures; planner
# primitives synthesized). Add --with-upstream-examples to also retarget the upstream
# G1 reference clips. See the header of tools/build_demo_bank_from_upstream.py for what
# is produced and how the retarget works.
#
#   tools/build_demo_bank_from_upstream.sh [extra args for the .py]
#
# Interpreter: $PYTHON, else the repo .venv, else python3 (needs numpy, scipy,
# joblib, pyyaml; gear_sonic importable from the repo root).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$REPO/.venv/bin/python" ]]; then PY="$REPO/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
cd "$REPO"
exec "$PY" tools/build_demo_bank_from_upstream.py --python "$PY" "$@"
