#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/V13}"
PYTHON="${PYTHON:-python}"
POOL="${POOL:-/workspace/ABC/processed/abc_parsed_full}"
OUTBASE="${OUTBASE:-/workspace/ABC/processed/train_outputs}"
RUN_NAME="newscheme_full_vqvae_complex_recovery"
SAMPLES="450000"
EPOCHS="80"
BATCH_SIZE="128"
LEARNING_RATE="1e-4"
MIN_EPOCHS="30"
PATIENCE="14"
MIN_DELTA="1e-6"
COMPLEX_FRACTION="0.50"
COMPLEX_MIN_FACES="12"
COMPLEX_MIN_EDGES="20"
CURVED_FRACTION="0.35"
MAX_SOURCE_FACES="50"
MAX_SOURCE_EDGES="150"
COMPLEX_LOSS_WEIGHT="1.25"
CURVED_LOSS_WEIGHT="2.0"
CURVED_LOSS_THRESHOLD="0.02"
RESUME_FROM="/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt"
HISTORY_IN="/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/vqvae_history.json"
TARGET_EPOCH="180"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
RUN_BENCHMARK="0"
SEQUENCE="/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl"
BENCHMARK_OUTPUT_ROOT="/workspace/V13/local_runs/reconstruction_eval"
BENCHMARK_PREFIX=""
BENCHMARK_DEVICE="cpu"
BENCHMARK_MAX_SAMPLES="10"
BENCHMARK_MAX_SEQ_LEN="1024"
BENCHMARK_COLS="5"

usage() {
  cat <<'USAGE'
Usage: bash tools/run_vqvae_complex_recovery.sh [options]

Options:
  --repo-root PATH          Repository root on the rented server.
  --python PATH            Python executable, for example /opt/conda/bin/python.
  --pool PATH              Parsed ABC pickle pool.
  --outbase PATH           Training output root.
  --run-name NAME          Fresh VQ-VAE output directory name.
  --samples N              VQ-VAE geometry patch budget.
  --epochs N               Epochs requested for this continuation.
  --batch-size N           VQ-VAE batch size.
  --lr VALUE               VQ-VAE learning rate.
  --min-epochs N           Early-stop minimum absolute epochs for this run.
  --patience N             Early-stop patience.
  --min-delta VALUE        Early-stop minimum improvement.
  --complex-fraction VALUE Fraction of patch budget reserved for complex sources.
  --complex-min-faces N    Face-count threshold for complex sources.
  --complex-min-edges N    Edge-count threshold for complex sources.
  --curved-fraction VALUE  Fraction reserved for high-curvature patches.
  --max-source-faces N     Drop VQ patches from sources above this face count; 0 disables.
  --max-source-edges N     Drop VQ patches from sources above this edge count; 0 disables.
  --complex-loss-weight V  Training MSE multiplier for complex-source patches.
  --curved-loss-weight V   Training MSE multiplier for curved patches.
  --curved-loss-threshold V
                           Curvature threshold used by loss weighting.
  --resume-from PATH       Starting VQ-VAE checkpoint.
  --history-in PATH        Previous VQ-VAE history JSON.
  --target-epoch N         Absolute target epoch for history accounting.
  --cuda-visible-devices V CUDA_VISIBLE_DEVICES value.
  --run-benchmark          After training, run the four-slice VQ-VAE promotion benchmark.
                           Optional curved diagnostics can be run separately with
                           --orders shortest,random,longest,most_faces,most_curved
                           after sequences preserve source_path metadata.
  --sequence PATH          Sequence package for post-training benchmark.
  --benchmark-output-root PATH
                           Reconstruction-evaluation output root.
  --benchmark-prefix NAME  Prefix for post-training benchmark runs.
  --benchmark-device NAME  Benchmark device: cpu, cuda, or auto.
  --benchmark-max-samples N
                           Samples per benchmark slice.
  --benchmark-max-seq-len N
                           Maximum source sequence length for benchmark slices.
  --benchmark-cols N       Contact-sheet columns for rendered benchmark outputs.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --pool) POOL="$2"; shift 2 ;;
    --outbase) OUTBASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LEARNING_RATE="$2"; shift 2 ;;
    --min-epochs) MIN_EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --min-delta) MIN_DELTA="$2"; shift 2 ;;
    --complex-fraction) COMPLEX_FRACTION="$2"; shift 2 ;;
    --complex-min-faces) COMPLEX_MIN_FACES="$2"; shift 2 ;;
    --complex-min-edges) COMPLEX_MIN_EDGES="$2"; shift 2 ;;
    --curved-fraction) CURVED_FRACTION="$2"; shift 2 ;;
    --max-source-faces) MAX_SOURCE_FACES="$2"; shift 2 ;;
    --max-source-edges) MAX_SOURCE_EDGES="$2"; shift 2 ;;
    --complex-loss-weight) COMPLEX_LOSS_WEIGHT="$2"; shift 2 ;;
    --curved-loss-weight) CURVED_LOSS_WEIGHT="$2"; shift 2 ;;
    --curved-loss-threshold) CURVED_LOSS_THRESHOLD="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --history-in) HISTORY_IN="$2"; shift 2 ;;
    --target-epoch) TARGET_EPOCH="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --run-benchmark) RUN_BENCHMARK="1"; shift ;;
    --sequence) SEQUENCE="$2"; shift 2 ;;
    --benchmark-output-root) BENCHMARK_OUTPUT_ROOT="$2"; shift 2 ;;
    --benchmark-prefix) BENCHMARK_PREFIX="$2"; shift 2 ;;
    --benchmark-device) BENCHMARK_DEVICE="$2"; shift 2 ;;
    --benchmark-max-samples) BENCHMARK_MAX_SAMPLES="$2"; shift 2 ;;
    --benchmark-max-seq-len) BENCHMARK_MAX_SEQ_LEN="$2"; shift 2 ;;
    --benchmark-cols) BENCHMARK_COLS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"

OUT_DIR="${OUTBASE%/}/$RUN_NAME"
mkdir -p "$OUT_DIR"
LEDGER="$OUT_DIR/server_run_ledger.txt"

{
  echo "run_slug=$RUN_NAME"
  echo "stage=vqvae_complex_recovery"
  echo "repo_root=$REPO_ROOT"
  echo "pool=$POOL"
  echo "outbase=$OUTBASE"
  echo "resume_from=$RESUME_FROM"
  echo "history_in=$HISTORY_IN"
  echo "samples=$SAMPLES"
  echo "epochs=$EPOCHS"
  echo "batch_size=$BATCH_SIZE"
  echo "learning_rate=$LEARNING_RATE"
  echo "complex_fraction=$COMPLEX_FRACTION"
  echo "curved_fraction=$CURVED_FRACTION"
  echo "max_source_faces=$MAX_SOURCE_FACES"
  echo "max_source_edges=$MAX_SOURCE_EDGES"
  echo "complex_loss_weight=$COMPLEX_LOSS_WEIGHT"
  echo "curved_loss_weight=$CURVED_LOSS_WEIGHT"
  echo "curved_loss_threshold=$CURVED_LOSS_THRESHOLD"
  echo "target_epoch=$TARGET_EPOCH"
  echo "run_benchmark=$RUN_BENCHMARK"
  echo "benchmark_sequence=$SEQUENCE"
  echo "benchmark_output_root=$BENCHMARK_OUTPUT_ROOT"
  echo "started_at=$(date -Is)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi
  fi
  echo "--- python ---"
  "$PYTHON" --version
} > "$LEDGER"

export PYTHONUTF8="1"
export PYTHONIOENCODING="utf-8"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export NS_POOL="$POOL"
export NS_OUTBASE="$OUTBASE"
export NS_OUT="$RUN_NAME"
export NS_N="999999"

export NS_VQ_SAMPLES="$SAMPLES"
export NS_VQ_EPOCHS="$EPOCHS"
export NS_VQ_TARGET_EPOCH="$TARGET_EPOCH"
export NS_VQ_BS="$BATCH_SIZE"
export NS_VQ_LR="$LEARNING_RATE"
export NS_VQ_MIN_EPOCHS="$MIN_EPOCHS"
export NS_VQ_PATIENCE="$PATIENCE"
export NS_VQ_MIN_DELTA="$MIN_DELTA"
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

if [[ -n "$RESUME_FROM" ]]; then
  export NS_VQ_RESUME_FROM="$RESUME_FROM"
fi
if [[ -n "$HISTORY_IN" ]]; then
  export NS_VQ_HISTORY_IN="$HISTORY_IN"
fi

"$PYTHON" breparg_improvements/train.py --stage split
"$PYTHON" breparg_improvements/train.py --stage vqvae

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  if [[ -z "$BENCHMARK_PREFIX" ]]; then
    BENCHMARK_PREFIX="${RUN_NAME}_posttrain_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$BENCHMARK_OUTPUT_ROOT"
  "$PYTHON" tools/run_vqvae_slice_benchmark.py \
    --python "$PYTHON" \
    --sequence "$SEQUENCE" \
    --vqvae-checkpoint "$OUT_DIR/fsq_vqvae_best.pt" \
    --output-root "$BENCHMARK_OUTPUT_ROOT" \
    --run-prefix "$BENCHMARK_PREFIX" \
    --device "$BENCHMARK_DEVICE" \
    --max-samples "$BENCHMARK_MAX_SAMPLES" \
    --max-seq-len "$BENCHMARK_MAX_SEQ_LEN" \
    --cols "$BENCHMARK_COLS"

  BENCHMARK_SUMMARY="${BENCHMARK_OUTPUT_ROOT%/}/${BENCHMARK_PREFIX}_benchmark_summary.json"
  HANDOFF_MANIFEST="$OUT_DIR/copy_back_manifest.json"
  "$PYTHON" tools/write_vqvae_server_handoff.py \
    --run-dir "$OUT_DIR" \
    --benchmark-summary "$BENCHMARK_SUMMARY" \
    --repo-root "$REPO_ROOT" \
    --output "$HANDOFF_MANIFEST"
fi

{
  echo "finished_at=$(date -Is)"
  echo "best_checkpoint=$OUT_DIR/fsq_vqvae_best.pt"
  echo "final_checkpoint=$OUT_DIR/fsq_vqvae_final.pt"
  echo "history=$OUT_DIR/vqvae_history.json"
  if [[ "$RUN_BENCHMARK" == "1" ]]; then
    echo "benchmark_prefix=$BENCHMARK_PREFIX"
    echo "benchmark_summary=$BENCHMARK_SUMMARY"
    echo "copy_back_manifest=$HANDOFF_MANIFEST"
  fi
} >> "$LEDGER"

echo "Complex VQ-VAE recovery finished."
echo "Ledger: $LEDGER"
echo "Best checkpoint: $OUT_DIR/fsq_vqvae_best.pt"
if [[ "$RUN_BENCHMARK" == "1" ]]; then
  echo "Benchmark summary: $BENCHMARK_SUMMARY"
  echo "Copy-back manifest: $HANDOFF_MANIFEST"
fi
