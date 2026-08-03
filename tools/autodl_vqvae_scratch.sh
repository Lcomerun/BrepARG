#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/workplace}"
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda_envs/breparg}"
PYTHON="${PYTHON:-$ENV_PREFIX/bin/python}"
PARSED_SHARD_ROOT="${PARSED_SHARD_ROOT:-$REPO_ROOT/V13_abc_parsed_shards}"
PATCH_SHARD_ROOT="${PATCH_SHARD_ROOT:-/root/autodl-tmp/ABC/processed/vqvae_patch_shards}"
OUTBASE="${OUTBASE:-/root/autodl-tmp/ABC/processed/train_outputs}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/local_reports}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-newscheme_vqvae_5090_scratch_smoke_20260710}"
FULL_RUN_NAME="${FULL_RUN_NAME:-newscheme_vqvae_5090_scratch_20260710}"

SMOKE_SAMPLES="${SMOKE_SAMPLES:-2048}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
FULL_SAMPLES="${FULL_SAMPLES:-300000}"
FULL_EPOCHS="${FULL_EPOCHS:-160}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

usage() {
  cat <<'USAGE'
Usage: bash tools/autodl_vqvae_scratch.sh ACTION [--no-tail]

Actions:
  preflight      Check repo files, paths, disk, Python path, and patch shard summary.
  verify-preflight Check the preflight log for path and patch-shard readiness.
  diagnose       Check the current Python/CUDA environment without installing packages.
  verify-diagnose Check the diagnose log for CUDA, imports, and no fatal errors.
  repair         Install/repair pinned VQ-VAE deps and PyTorch cu128 for RTX 5090.
  build-patches  Build VQ patch shards from parsed shards.
  smoke          Start a 1-epoch scratch VQ-VAE smoke run from existing patch shards.
  verify-smoke   Check the smoke log for patch sampling, CUDA, and epoch/loss output.
  gate           Verify preflight, diagnose, and smoke logs before full training.
  full           Start the full scratch VQ-VAE run from existing patch shards.
  status         Show preflight/diagnose/repair/patch/smoke/full logs and nvidia-smi.

Defaults can be overridden with environment variables:
  REPO_ROOT, ENV_PREFIX, PYTHON, PARSED_SHARD_ROOT, PATCH_SHARD_ROOT, OUTBASE,
  REPORT_DIR, SMOKE_RUN_NAME, FULL_RUN_NAME, SMOKE_SAMPLES, FULL_SAMPLES,
  FULL_EPOCHS, BATCH_SIZE, LEARNING_RATE, CUDA_VISIBLE_DEVICES
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

run_logged() {
  local log="$1"
  shift
  mkdir -p "$REPORT_DIR"
  echo "Logging to: $log"
  set +e
  "$@" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  return "$rc"
}

preflight_check() {
  local failures=0

  note_fail() {
    echo "FAIL: $*"
    failures=$((failures + 1))
  }

  note_ok() {
    echo "OK: $*"
  }

  echo "== AutoDL VQ-VAE Scratch Preflight =="
  echo "repo_root=$REPO_ROOT"
  echo "env_prefix=$ENV_PREFIX"
  echo "python=$PYTHON"
  echo "parsed_shard_root=$PARSED_SHARD_ROOT"
  echo "patch_shard_root=$PATCH_SHARD_ROOT"
  echo "outbase=$OUTBASE"
  echo "report_dir=$REPORT_DIR"
  echo

  [[ -d "$REPO_ROOT" ]] && note_ok "repo root exists" || note_fail "repo root missing: $REPO_ROOT"
  for path in \
    "$REPO_ROOT/tools/server_bootstrap.sh" \
    "$REPO_ROOT/tools/run_vqvae_from_patch_shards.sh" \
    "$REPO_ROOT/tools/autodl_vqvae_scratch.sh" \
    "$REPO_ROOT/breparg_improvements/train.py" \
    "$REPO_ROOT/breparg_improvements/vqvae_sampling.py" \
    "$REPO_ROOT/breparg_improvements/sharded_data.py"; do
    [[ -s "$path" ]] && note_ok "required file exists: $path" || note_fail "required file missing: $path"
  done

  [[ -x "$PYTHON" ]] && note_ok "python executable exists" || note_fail "python executable missing: $PYTHON"
  [[ -d "$PATCH_SHARD_ROOT" ]] && note_ok "patch shard root exists" || note_fail "patch shard root missing: $PATCH_SHARD_ROOT"
  [[ -s "$PATCH_SHARD_ROOT/_summary.json" ]] && note_ok "patch shard summary exists" || note_fail "patch shard summary missing: $PATCH_SHARD_ROOT/_summary.json"

  if [[ -d "$PARSED_SHARD_ROOT" ]]; then
    local parsed_count
    parsed_count="$(find "$PARSED_SHARD_ROOT" -maxdepth 1 -name 'parsed_abc_*.pkl.zst' 2>/dev/null | wc -l | tr -d ' ')"
    echo "parsed shard files: $parsed_count"
  else
    echo "parsed shard root not found; this is acceptable only when training from already-built patch shards."
  fi

  if [[ -x "$PYTHON" && -s "$PATCH_SHARD_ROOT/_summary.json" ]]; then
    "$PYTHON" - "$PATCH_SHARD_ROOT" <<'PY' || failures=$((failures + 1))
import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "_summary.json").read_text(encoding="utf-8"))
shards = sorted(glob.glob(str(root / "vq_patch_shard_*.pkl.zst")))
print("patch summary status:", summary.get("status"))
print("patch summary patch_shards:", summary.get("patch_shards"))
print("actual patch shard files:", len(shards))
print("patches:", summary.get("patches"))
print("surfaces:", summary.get("surfaces"))
print("edges:", summary.get("edges"))
if summary.get("status") != "BUILT":
    raise SystemExit("patch summary status is not BUILT")
if int(summary.get("patch_shards", -1)) != len(shards):
    raise SystemExit("patch shard file count does not match summary")
if int(summary.get("patches", 0)) <= 0:
    raise SystemExit("patch summary has zero patches")
PY
  fi

  echo
  if command -v df >/dev/null 2>&1; then
    echo "== Disk =="
    df -h "$REPO_ROOT" "$(dirname "$PATCH_SHARD_ROOT")" "$OUTBASE" 2>/dev/null || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "== nvidia-smi =="
    nvidia-smi || true
  fi

  if [[ "$failures" -gt 0 ]]; then
    echo
    echo "Preflight failed with $failures issue(s)."
    return 2
  fi
  echo
  echo "Preflight OK."
}

tail_follow_pid() {
  local pid="$1"
  local log="$2"
  sleep 1
  if tail --help 2>/dev/null | grep -q -- '--pid'; then
    tail --pid="$pid" -f "$log"
  else
    tail -f "$log"
  fi
}

bootstrap_env() {
  local mode="$1"
  cd "$REPO_ROOT"
  if [[ "$mode" == "diagnose" ]]; then
    [[ -x "$PYTHON" ]] || die "Python not found: $PYTHON. Run 'repair' first or set PYTHON."
    run_logged "$REPORT_DIR/autodl_diagnose.log" \
      env REPO_ROOT="$REPO_ROOT" \
        PYTHON="$PYTHON" \
        V13_SKIP_INSTALL=1 \
        V13_REQUIRE_CUDA=1 \
        bash tools/server_bootstrap.sh
  elif [[ "$mode" == "repair" ]]; then
    if [[ -x "$PYTHON" ]]; then
      run_logged "$REPORT_DIR/autodl_repair.log" \
        env REPO_ROOT="$REPO_ROOT" \
          PYTHON="$PYTHON" \
          V13_FORCE_CU128=1 \
          V13_REQUIRE_CUDA=1 \
          bash tools/server_bootstrap.sh
    else
      run_logged "$REPORT_DIR/autodl_repair.log" \
        env REPO_ROOT="$REPO_ROOT" \
          ENV_PREFIX="$ENV_PREFIX" \
          V13_FORCE_CU128=1 \
          V13_REQUIRE_CUDA=1 \
          bash tools/server_bootstrap.sh
    fi
  else
    die "Unknown bootstrap mode: $mode"
  fi
}

run_patch_build() {
  [[ -x "$PYTHON" ]] || die "Python not found: $PYTHON. Run 'repair' first or set PYTHON."
  [[ -d "$PARSED_SHARD_ROOT" ]] || die "Parsed shard root not found: $PARSED_SHARD_ROOT"
  mkdir -p "$REPORT_DIR"
  local log="$REPORT_DIR/build_vq_patch_shards_autodl.log"
  local pidfile="$REPORT_DIR/build_vq_patch_shards_autodl.pid"

  cd "$REPO_ROOT"
  nohup bash tools/run_vqvae_from_patch_shards.sh \
    --repo-root "$REPO_ROOT" \
    --python "$PYTHON" \
    --parsed-shard-root "$PARSED_SHARD_ROOT" \
    --patch-shard-root "$PATCH_SHARD_ROOT" \
    --outbase "$OUTBASE" \
    --build-patch-shards \
    > "$log" 2>&1 &
  local pid="$!"
  echo "$pid" > "$pidfile"
  echo "Started patch shard build: pid=$pid log=$log"
}

start_training_run() {
  local kind="$1"
  local run_name samples epochs target_epoch log pidfile
  if [[ "$kind" == "smoke" ]]; then
    run_name="$SMOKE_RUN_NAME"
    samples="$SMOKE_SAMPLES"
    epochs="$SMOKE_EPOCHS"
    target_epoch="$SMOKE_EPOCHS"
    log="$REPORT_DIR/vqvae_scratch_smoke_20260710.log"
    pidfile="$REPORT_DIR/vqvae_scratch_smoke_20260710.pid"
  elif [[ "$kind" == "full" ]]; then
    run_name="$FULL_RUN_NAME"
    samples="$FULL_SAMPLES"
    epochs="$FULL_EPOCHS"
    target_epoch="$FULL_EPOCHS"
    log="$REPORT_DIR/vqvae_scratch_5090_20260710.log"
    pidfile="$REPORT_DIR/vqvae_scratch_5090_20260710.pid"
  else
    die "Unknown training kind: $kind"
  fi

  [[ -x "$PYTHON" ]] || die "Python not found: $PYTHON. Run 'repair' first or set PYTHON."
  [[ -s "$PATCH_SHARD_ROOT/_summary.json" ]] || die "Patch shard summary missing: $PATCH_SHARD_ROOT/_summary.json"
  mkdir -p "$REPORT_DIR"

  cd "$REPO_ROOT"
  nohup bash tools/run_vqvae_from_patch_shards.sh \
    --repo-root "$REPO_ROOT" \
    --python "$PYTHON" \
    --patch-shard-root "$PATCH_SHARD_ROOT" \
    --outbase "$OUTBASE" \
    --run-train \
    --scratch \
    --run-name "$run_name" \
    --samples "$samples" \
    --epochs "$epochs" \
    --target-epoch "$target_epoch" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_VALUE" \
    > "$log" 2>&1 &
  local pid="$!"
  echo "$pid" > "$pidfile"
  echo "Started $kind VQ-VAE run: pid=$pid run_name=$run_name log=$log"
  if [[ "${TAIL_LOG:-1}" == "1" ]]; then
    tail_follow_pid "$pid" "$log"
  fi
}

show_one_status() {
  local label="$1"
  local pidfile="$2"
  local log="$3"
  echo "== $label =="
  if [[ -s "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile")"
    ps -p "$pid" -o pid,stat,etime,cmd || true
  else
    echo "No pid file: $pidfile"
  fi
  if [[ -s "$log" ]]; then
    echo "-- tail $log --"
    tail -n 80 "$log"
  else
    echo "No log yet: $log"
  fi
}

show_log_status() {
  local label="$1"
  local log="$2"
  echo "== $label =="
  if [[ -s "$log" ]]; then
    echo "-- tail $log --"
    tail -n 60 "$log"
  else
    echo "No log yet: $log"
  fi
}

show_status() {
  show_log_status "preflight" "$REPORT_DIR/autodl_preflight.log"
  show_log_status "diagnose" "$REPORT_DIR/autodl_diagnose.log"
  show_log_status "repair" "$REPORT_DIR/autodl_repair.log"
  show_one_status "patch-build" "$REPORT_DIR/build_vq_patch_shards_autodl.pid" "$REPORT_DIR/build_vq_patch_shards_autodl.log"
  show_one_status "smoke" "$REPORT_DIR/vqvae_scratch_smoke_20260710.pid" "$REPORT_DIR/vqvae_scratch_smoke_20260710.log"
  show_one_status "full" "$REPORT_DIR/vqvae_scratch_5090_20260710.pid" "$REPORT_DIR/vqvae_scratch_5090_20260710.log"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "== nvidia-smi =="
    nvidia-smi || true
  fi
  return 0
}

verify_smoke_log() {
  local log="$REPORT_DIR/vqvae_scratch_smoke_20260710.log"
  local failures=0

  [[ -s "$log" ]] || die "Smoke log missing or empty: $log"
  echo "Checking smoke log: $log"

  require_log_pattern() {
    local pattern="$1"
    local label="$2"
    if grep -Eq "$pattern" "$log"; then
      echo "OK: $label"
    else
      echo "FAIL: $label"
      failures=$((failures + 1))
    fi
  }

  reject_log_pattern() {
    local pattern="$1"
    local label="$2"
    if grep -Eq "$pattern" "$log"; then
      echo "FAIL: $label"
      failures=$((failures + 1))
    else
      echo "OK: $label"
    fi
  }

  require_log_pattern 'Skipping parsed shard verification because --build-patch-shards was not requested|Verifying parsed shards' 'launcher reached shard verification decision'
  require_log_pattern 'patch shard summary status: BUILT' 'patch shard summary is BUILT'
  require_log_pattern 'actual patch shard files: [1-9][0-9]*' 'patch shard files are present'
  require_log_pattern 'runtime python:' 'runtime Python printed'
  require_log_pattern 'diffusers\.VQModel:' 'diffusers VQModel import succeeded'
  require_log_pattern 'cuda matmul ok:' 'CUDA kernel smoke test passed'
  require_log_pattern 'VQ patch-shard sampling selected=' 'trainer read VQ patch shards'
  require_log_pattern 'vqvae ep[[:space:]]+[0-9]+[[:space:]]+train=.*val=.*' 'VQ-VAE epoch/loss line printed'
  reject_log_pattern 'Traceback|ModuleNotFoundError|CUDA kernel smoke test failed|no kernel image|out of memory|RuntimeError: CUDA error|ERROR:' 'no known fatal error pattern'

  if [[ "$failures" -gt 0 ]]; then
    echo "Smoke verification failed with $failures issue(s)."
    return 2
  fi
  echo "Smoke verification OK."
}

verify_preflight_log() {
  local log="$REPORT_DIR/autodl_preflight.log"
  local failures=0

  [[ -s "$log" ]] || die "Preflight log missing or empty: $log"
  echo "Checking preflight log: $log"

  require_log_pattern() {
    local pattern="$1"
    local label="$2"
    if grep -Eq "$pattern" "$log"; then
      echo "OK: $label"
    else
      echo "FAIL: $label"
      failures=$((failures + 1))
    fi
  }

  reject_log_pattern() {
    local pattern="$1"
    local label="$2"
    if grep -Eq "$pattern" "$log"; then
      echo "FAIL: $label"
      failures=$((failures + 1))
    else
      echo "OK: $label"
    fi
  }

  require_log_pattern 'OK: repo root exists' 'repo root exists'
  require_log_pattern 'OK: required file exists: .*tools/server_bootstrap\.sh' 'server bootstrap script exists'
  require_log_pattern 'OK: required file exists: .*tools/run_vqvae_from_patch_shards\.sh' 'VQ launcher script exists'
  require_log_pattern 'OK: required file exists: .*tools/autodl_vqvae_scratch\.sh' 'AutoDL helper exists'
  require_log_pattern 'OK: required file exists: .*breparg_improvements/train\.py' 'train.py exists'
  require_log_pattern 'OK: python executable exists' 'Python executable exists'
  require_log_pattern 'OK: patch shard root exists' 'patch shard root exists'
  require_log_pattern 'OK: patch shard summary exists' 'patch shard summary exists'
  require_log_pattern 'patch summary status: BUILT' 'patch summary is BUILT'
  require_log_pattern 'actual patch shard files: [1-9][0-9]*' 'patch shard files are present'
  require_log_pattern 'patches: [1-9][0-9]*' 'patch count is positive'
  require_log_pattern 'Preflight OK\.' 'preflight completed'
  reject_log_pattern '^FAIL:|Preflight failed|Traceback|ModuleNotFoundError' 'no known preflight fatal error pattern'

  if [[ "$failures" -gt 0 ]]; then
    echo "Preflight verification failed with $failures issue(s)."
    return 2
  fi
  echo "Preflight verification OK."
}

verify_diagnose_log() {
  local log="$REPORT_DIR/autodl_diagnose.log"
  local failures=0

  [[ -s "$log" ]] || die "Diagnose log missing or empty: $log"
  echo "Checking diagnose log: $log"

  require_log_pattern() {
    local pattern="$1"
    local label="$2"
    if grep -Eq "$pattern" "$log"; then
      echo "OK: $label"
    else
      echo "FAIL: $label"
      failures=$((failures + 1))
    fi
  }

  reject_log_pattern() {
    local pattern="$1"
    local label="$2"
    if grep -Eq "$pattern" "$log"; then
      echo "FAIL: $label"
      failures=$((failures + 1))
    else
      echo "OK: $label"
    fi
  }

  require_log_pattern 'V13_SKIP_INSTALL=1|Using explicit Python:' 'diagnose ran through bootstrap helper'
  require_log_pattern 'python:' 'Python executable printed'
  require_log_pattern 'torch:.*ok|torch:[[:space:]]+[0-9]' 'torch import/version printed'
  require_log_pattern 'numpy: ok' 'numpy import succeeded'
  require_log_pattern 'transformers: ok' 'transformers import succeeded'
  require_log_pattern 'diffusers: ok' 'diffusers import succeeded'
  require_log_pattern 'diffusers\.VQModel:' 'diffusers VQModel import succeeded'
  require_log_pattern 'cuda available: True' 'CUDA is visible to torch'
  require_log_pattern 'device:' 'CUDA device printed'
  require_log_pattern 'capability:' 'CUDA capability printed'
  require_log_pattern 'cuda matmul ok:' 'CUDA kernel smoke test passed'
  require_log_pattern 'Environment bootstrap complete\.' 'bootstrap completed'
  reject_log_pattern 'Missing modules:|missing$|ModuleNotFoundError|CUDA kernel smoke test failed|CUDA is required but|no kernel image|Traceback' 'no known diagnose fatal error pattern'

  if [[ "$failures" -gt 0 ]]; then
    echo "Diagnose verification failed with $failures issue(s)."
    return 2
  fi
  echo "Diagnose verification OK."
}

run_gate() {
  local failures=0
  echo "== AutoDL VQ-VAE Gate =="
  verify_preflight_log || failures=$((failures + 1))
  echo
  verify_diagnose_log || failures=$((failures + 1))
  echo
  verify_smoke_log || failures=$((failures + 1))

  if [[ "$failures" -gt 0 ]]; then
    echo
    echo "Gate failed with $failures failed verifier(s). Do not start full training yet."
    return 2
  fi
  echo
  echo "Gate OK: ready to start full scratch VQ-VAE training."
}

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-tail) TAIL_LOG=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

case "$ACTION" in
  preflight) run_logged "$REPORT_DIR/autodl_preflight.log" preflight_check ;;
  verify-preflight) verify_preflight_log ;;
  diagnose) bootstrap_env diagnose ;;
  verify-diagnose) verify_diagnose_log ;;
  repair) bootstrap_env repair ;;
  build-patches) run_patch_build ;;
  smoke) start_training_run smoke ;;
  verify-smoke) verify_smoke_log ;;
  gate) run_gate ;;
  full) start_training_run full ;;
  status) show_status ;;
  help|--help|-h) usage ;;
  *) usage >&2; die "Unknown action: $ACTION" ;;
esac
