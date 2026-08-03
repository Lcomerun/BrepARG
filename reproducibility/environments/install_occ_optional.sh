#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${V13_ENV_NAME:-v13-repro-cu128}"
REPORT_DIR="${V13_ENV_REPORT_DIR:-$SCRIPT_DIR/../repro_outputs/environment}"
OCCWL_COMMIT="418deb247e6d0bb2beb59dd3c610edeb94f4ee77"

find_conda_frontend() {
  for candidate in micromamba mamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

CONDA_FRONTEND="$(find_conda_frontend || true)"
if [[ -z "$CONDA_FRONTEND" ]]; then
  echo "No conda-compatible frontend is available for pythonocc-core." >&2
  exit 2
fi

mkdir -p "$REPORT_DIR"
if [[ "$CONDA_FRONTEND" == "micromamba" ]]; then
  micromamba install -y -n "$ENV_NAME" -c conda-forge pythonocc-core=7.9.3
  RUN=(micromamba run -n "$ENV_NAME")
else
  "$CONDA_FRONTEND" install -y -n "$ENV_NAME" -c conda-forge pythonocc-core=7.9.3
  RUN=(conda run -n "$ENV_NAME" --no-capture-output)
fi

"${RUN[@]}" python -m pip install --no-deps \
  "git+https://github.com/AutodeskAILab/occwl.git@$OCCWL_COMMIT"
"${RUN[@]}" python "$SCRIPT_DIR/probe_occ.py" --output "$REPORT_DIR/occ_probe.json"
"${RUN[@]}" python -m pip freeze --all > "$REPORT_DIR/pip-freeze.with-occ.resolved.txt"

echo "Optional CAD layer ready in environment: $ENV_NAME"
