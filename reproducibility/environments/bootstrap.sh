#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${V13_ENV_NAME:-v13-repro-cu128}"
REPORT_DIR="${V13_ENV_REPORT_DIR:-$PACKAGE_ROOT/repro_outputs/environment}"

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
  echo "No conda, mamba, or micromamba executable is available." >&2
  exit 2
fi

mkdir -p "$REPORT_DIR"

if [[ "$CONDA_FRONTEND" == "micromamba" ]]; then
  if micromamba env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    micromamba env update -y -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.linux-gpu.yml" --prune
  else
    micromamba env create -y -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.linux-gpu.yml"
  fi
  RUN=(micromamba run -n "$ENV_NAME")
else
  if "$CONDA_FRONTEND" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    "$CONDA_FRONTEND" env update -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.linux-gpu.yml" --prune
  else
    "$CONDA_FRONTEND" env create -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.linux-gpu.yml"
  fi
  RUN=(conda run -n "$ENV_NAME" --no-capture-output)
fi

"${RUN[@]}" python -m pip install --requirement "$SCRIPT_DIR/requirements.linux-gpu.lock.txt"
"${RUN[@]}" python -m pip check
"${RUN[@]}" python -m pip freeze --all > "$REPORT_DIR/pip-freeze.resolved.txt"
ENV_PYTHON="$("${RUN[@]}" python -c 'import sys; print(sys.executable)')"
printf 'V13_REPRO_PYTHON=%s\n' "$ENV_PYTHON" > "$PACKAGE_ROOT/configs/python.env"
"${RUN[@]}" python "$SCRIPT_DIR/probe_environment.py" \
  --source-root "$PACKAGE_ROOT/source/current" \
  --output "$REPORT_DIR/environment_probe.json" \
  --require-cuda

if [[ "${V13_INSTALL_OCC:-0}" == "1" ]]; then
  V13_ENV_NAME="$ENV_NAME" V13_ENV_REPORT_DIR="$REPORT_DIR" \
    bash "$SCRIPT_DIR/install_occ_optional.sh"
fi

echo "Environment ready: $ENV_NAME"
echo "Launcher Python: $ENV_PYTHON"
echo "Resolved environment evidence: $REPORT_DIR"
