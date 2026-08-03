#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/V13}"
PYTHON="${PYTHON:-python}"
PARSED_SHARD_ROOT="/workspace/ABC/processed/abc_parsed_shards"
PATCH_SHARD_ROOT="/workspace/ABC/processed/vqvae_patch_shards"
OUTBASE="/workspace/ABC/processed/train_outputs"
RUN_NAME="newscheme_vqvae_sharded_recovery"
RESUME_FROM="/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt"
HISTORY_IN="/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/vqvae_history.json"
SCRATCH="0"
SAMPLES="300000"
EPOCHS="120"
BATCH_SIZE="128"
LEARNING_RATE="1e-5"
TARGET_EPOCH="220"
COMPLEX_FRACTION="0.40"
CURVED_FRACTION="0.35"
COMPLEX_MIN_FACES="12"
COMPLEX_MIN_EDGES="20"
MAX_SOURCE_FACES="50"
MAX_SOURCE_EDGES="150"
COMPLEX_LOSS_WEIGHT="1.15"
CURVED_LOSS_WEIGHT="1.5"
CURVED_LOSS_THRESHOLD="0.02"
PATCHES_PER_SHARD="100000"
BUILD_PATCH_SHARDS="0"
RUN_TRAIN="0"
REBUILD_PATCH_SHARDS="0"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'USAGE'
Usage: bash tools/run_vqvae_from_patch_shards.sh [options]

Required action:
  --build-patch-shards      Build VQ patch shards from parsed shards.
  --run-train               Start VQ-VAE training from VQ patch shards.

Common options:
  --repo-root PATH          Repository root. Default: /workspace/V13.
  --python PATH             Python executable. Default: python.
  --parsed-shard-root PATH  Parsed shard directory.
  --patch-shard-root PATH   VQ patch shard directory.
  --outbase PATH            Training output root.
  --run-name NAME           VQ-VAE output directory name.
  --resume-from PATH        Starting VQ-VAE checkpoint.
  --history-in PATH         Previous VQ-VAE history JSON.
  --scratch                 Train from scratch; ignore resume/history inputs.
  --samples N               VQ patch sample budget.
  --epochs N                Epochs for this run.
  --batch-size N            VQ-VAE batch size.
  --lr VALUE                VQ-VAE learning rate.
  --target-epoch N          Absolute target epoch for history accounting.
  --rebuild-patch-shards    Delete existing patch shards before rebuilding.
  --cuda-visible-devices V  CUDA_VISIBLE_DEVICES value.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --parsed-shard-root) PARSED_SHARD_ROOT="$2"; shift 2 ;;
    --patch-shard-root) PATCH_SHARD_ROOT="$2"; shift 2 ;;
    --outbase) OUTBASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --history-in) HISTORY_IN="$2"; shift 2 ;;
    --scratch) SCRATCH="1"; shift ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LEARNING_RATE="$2"; shift 2 ;;
    --target-epoch) TARGET_EPOCH="$2"; shift 2 ;;
    --complex-fraction) COMPLEX_FRACTION="$2"; shift 2 ;;
    --curved-fraction) CURVED_FRACTION="$2"; shift 2 ;;
    --complex-min-faces) COMPLEX_MIN_FACES="$2"; shift 2 ;;
    --complex-min-edges) COMPLEX_MIN_EDGES="$2"; shift 2 ;;
    --max-source-faces) MAX_SOURCE_FACES="$2"; shift 2 ;;
    --max-source-edges) MAX_SOURCE_EDGES="$2"; shift 2 ;;
    --complex-loss-weight) COMPLEX_LOSS_WEIGHT="$2"; shift 2 ;;
    --curved-loss-weight) CURVED_LOSS_WEIGHT="$2"; shift 2 ;;
    --curved-loss-threshold) CURVED_LOSS_THRESHOLD="$2"; shift 2 ;;
    --patches-per-shard) PATCHES_PER_SHARD="$2"; shift 2 ;;
    --build-patch-shards) BUILD_PATCH_SHARDS="1"; shift ;;
    --run-train) RUN_TRAIN="1"; shift ;;
    --rebuild-patch-shards) REBUILD_PATCH_SHARDS="1"; shift ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$BUILD_PATCH_SHARDS" != "1" && "$RUN_TRAIN" != "1" ]]; then
  echo "Choose --build-patch-shards, --run-train, or both." >&2
  usage >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p local_reports "$OUTBASE"

OUT_DIR="${OUTBASE%/}/$RUN_NAME"
mkdir -p "$OUT_DIR"
LEDGER="$OUT_DIR/server_run_ledger.txt"

{
  echo "run_slug=$RUN_NAME"
  echo "stage=vqvae_from_patch_shards"
  echo "repo_root=$REPO_ROOT"
  echo "parsed_shard_root=$PARSED_SHARD_ROOT"
  echo "patch_shard_root=$PATCH_SHARD_ROOT"
  echo "outbase=$OUTBASE"
  echo "scratch=$SCRATCH"
  echo "resume_from=$RESUME_FROM"
  echo "history_in=$HISTORY_IN"
  echo "samples=$SAMPLES"
  echo "epochs=$EPOCHS"
  echo "batch_size=$BATCH_SIZE"
  echo "learning_rate=$LEARNING_RATE"
  echo "started_at=$(date -Is)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi || echo "nvidia-smi failed; continuing because torch will run the authoritative CUDA check."
  fi
  echo "--- python ---"
  "$PYTHON" --version
} > "$LEDGER"

if [[ "$BUILD_PATCH_SHARDS" == "1" ]]; then
  echo "Verifying parsed shards..."
  "$PYTHON" tools/verify_parsed_shards.py "$PARSED_SHARD_ROOT"/parsed_abc_*.pkl.zst \
    --output local_reports/v13_parsed_shards_verify_server.json
else
  echo "Skipping parsed shard verification because --build-patch-shards was not requested."
fi

if [[ "$BUILD_PATCH_SHARDS" == "1" ]]; then
  if [[ "$REBUILD_PATCH_SHARDS" == "1" && -d "$PATCH_SHARD_ROOT" ]]; then
    rm -f "$PATCH_SHARD_ROOT"/vq_patch_shard_*.pkl "$PATCH_SHARD_ROOT"/vq_patch_shard_*.pkl.gz "$PATCH_SHARD_ROOT"/vq_patch_shard_*.pkl.zst
    rm -f "$PATCH_SHARD_ROOT"/_manifest.jsonl "$PATCH_SHARD_ROOT"/_summary.json
  fi
  mkdir -p "$PATCH_SHARD_ROOT"
  "$PYTHON" tools/build_vqvae_patch_shards.py \
    --parsed-shard-root "$PARSED_SHARD_ROOT" \
    --patch-shard-root "$PATCH_SHARD_ROOT" \
    --manifest "$PATCH_SHARD_ROOT/_manifest.jsonl" \
    --compression zstd \
    --compression-level 6 \
    --patches-per-shard "$PATCHES_PER_SHARD" \
    --complex-min-faces "$COMPLEX_MIN_FACES" \
    --complex-min-edges "$COMPLEX_MIN_EDGES" \
    --max-source-faces "$MAX_SOURCE_FACES" \
    --max-source-edges "$MAX_SOURCE_EDGES"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  [[ -s "$PATCH_SHARD_ROOT/_summary.json" ]] || { echo "Missing patch shard summary: $PATCH_SHARD_ROOT/_summary.json" >&2; exit 2; }
  if [[ "$SCRATCH" != "1" ]]; then
    [[ -f "$RESUME_FROM" ]] || { echo "Missing resume checkpoint: $RESUME_FROM" >&2; exit 2; }
  fi

  export PYTHONUTF8="1"
  export PYTHONIOENCODING="utf-8"
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  "$PYTHON" - "$PATCH_SHARD_ROOT" <<'PY'
import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = root / "_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
shards = sorted(glob.glob(str(root / "vq_patch_shard_*.pkl.zst")))
print("patch shard summary status:", summary.get("status"))
print("patch shard summary patch_shards:", summary.get("patch_shards"))
print("actual patch shard files:", len(shards))
print("patches:", summary.get("patches"))
print("surfaces:", summary.get("surfaces"))
print("edges:", summary.get("edges"))
if summary.get("status") != "BUILT":
    raise SystemExit(f"Patch shard summary is not BUILT: {summary.get('status')}")
if int(summary.get("patch_shards", -1)) != len(shards):
    raise SystemExit("Patch shard file count does not match _summary.json")
if int(summary.get("patches", 0)) <= 0:
    raise SystemExit("Patch shard summary has zero patches")
PY

  "$PYTHON" - <<'PY'
import sys

from diffusers import VQModel
import torch

print("runtime python:", sys.executable)
print("diffusers.VQModel:", VQModel)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for VQ-VAE training but is not visible to torch.")
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("cuda matmul ok:", float(y.mean().detach().cpu()))
PY

  export NS_VQ_PATCH_SHARD_ROOT="$PATCH_SHARD_ROOT"
  export NS_OUTBASE="$OUTBASE"
  export NS_OUT="$RUN_NAME"
  export NS_N="999999"
  export NS_VQ_SAMPLES="$SAMPLES"
  export NS_VQ_EPOCHS="$EPOCHS"
  export NS_VQ_TARGET_EPOCH="$TARGET_EPOCH"
  export NS_VQ_BS="$BATCH_SIZE"
  export NS_VQ_LR="$LEARNING_RATE"
  export NS_VQ_MIN_EPOCHS="30"
  export NS_VQ_PATIENCE="14"
  export NS_VQ_MIN_DELTA="1e-6"
  export NS_VQ_MAX_NONFINITE_VAL_EPOCHS="2"
  export NS_DISABLE_AMP_VQVAE="1"
  export NS_VQ_COMPLEX_FRACTION="$COMPLEX_FRACTION"
  export NS_VQ_COMPLEX_MIN_FACES="$COMPLEX_MIN_FACES"
  export NS_VQ_COMPLEX_MIN_EDGES="$COMPLEX_MIN_EDGES"
  export NS_VQ_CURVED_FRACTION="$CURVED_FRACTION"
  export NS_VQ_MAX_SOURCE_FACES="$MAX_SOURCE_FACES"
  export NS_VQ_MAX_SOURCE_EDGES="$MAX_SOURCE_EDGES"
  export NS_VQ_COMPLEX_LOSS_WEIGHT="$COMPLEX_LOSS_WEIGHT"
  export NS_VQ_CURVED_LOSS_WEIGHT="$CURVED_LOSS_WEIGHT"
  export NS_VQ_CURVED_LOSS_THRESHOLD="$CURVED_LOSS_THRESHOLD"
  if [[ "$SCRATCH" == "1" ]]; then
    unset NS_VQ_RESUME_FROM
    unset NS_VQ_HISTORY_IN
  else
    export NS_VQ_RESUME_FROM="$RESUME_FROM"
    if [[ -f "$HISTORY_IN" ]]; then
      export NS_VQ_HISTORY_IN="$HISTORY_IN"
    fi
  fi

  "$PYTHON" breparg_improvements/train.py --stage vqvae
fi

{
  echo "finished_at=$(date -Is)"
  echo "patch_summary=$PATCH_SHARD_ROOT/_summary.json"
  echo "best_checkpoint=$OUT_DIR/fsq_vqvae_best.pt"
  echo "history=$OUT_DIR/vqvae_history.json"
} >> "$LEDGER"

echo "Done."
echo "Ledger: $LEDGER"
