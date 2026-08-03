#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${V13_REPRO_PYTHON:-}

if [[ -z "$PYTHON_BIN" && -f "$PACKAGE_ROOT/configs/python.env" ]]; then
  python_line=$(grep -E '^V13_REPRO_PYTHON=' "$PACKAGE_ROOT/configs/python.env" | tail -n 1 || true)
  PYTHON_BIN=${python_line#V13_REPRO_PYTHON=}
fi
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Configured Python is not executable: $PYTHON_BIN" >&2
  echo "Run 'bash reproduce.sh bootstrap' or set V13_REPRO_PYTHON." >&2
  exit 2
fi

exec "$PYTHON_BIN" "$PACKAGE_ROOT/launchers/repro_cli.py" \
  --package-root "$PACKAGE_ROOT" "$@"
