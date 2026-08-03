#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/V13}"
PYTHON="${PYTHON:-python}"
OUTBASE="${OUTBASE:-$REPO_ROOT/local_runs/ar_training/train_outputs}"
RUN_NAME="newscheme_full_v13_ar1536"
SEQUENCE_SOURCE=""
SPLIT_SOURCE=""
RESUME_FROM=""
LEARNING_RATE="5e-4"
TARGET_EPOCHS="120"
BATCH_SIZE="8"
DMODEL="256"
LAYERS="8"
MAX_SEQ_LEN="1024"
SAVE_EVERY="20"
LOG_EVERY_BATCHES="2000"
NO_AUTO_RESUME="0"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'USAGE'
Usage: bash tools/run_ar_v13_long_context.sh [options]

Options:
  --repo-root PATH          Repository root on the rented server.
  --python PATH            Python executable, for example /opt/conda/bin/python.
  --outbase PATH           AR output root.
  --run-name NAME          Fresh AR branch name.
  --sequence-source PATH   Existing sequences_fsq_rcm.pkl to copy into the branch.
  --split-source PATH      Existing split.pkl to copy into the branch.
  --resume-from PATH       Optional AR checkpoint to seed the branch.
  --lr VALUE               AR learning rate.
  --target-epochs N        Absolute AR target epoch count.
  --batch-size N           AR batch size.
  --dmodel N               Transformer width.
  --layers N               Transformer layer count.
  --max-seq-len N          AR maximum sequence length, for example 1536 or 2048.
  --save-every N           Periodic checkpoint interval.
  --log-every-batches N    Training log interval.
  --no-auto-resume         Do not resume from this branch's ar_latest.pt.
  --cuda-visible-devices V CUDA_VISIBLE_DEVICES value.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --outbase) OUTBASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --sequence-source) SEQUENCE_SOURCE="$2"; shift 2 ;;
    --split-source) SPLIT_SOURCE="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --lr) LEARNING_RATE="$2"; shift 2 ;;
    --target-epochs) TARGET_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --dmodel) DMODEL="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --log-every-batches) LOG_EVERY_BATCHES="$2"; shift 2 ;;
    --no-auto-resume) NO_AUTO_RESUME="1"; shift ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"

OUT_DIR="${OUTBASE%/}/$RUN_NAME"
mkdir -p "$OUT_DIR"
LEDGER="$OUT_DIR/server_run_ledger.txt"
SEQUENCE_PATH="$OUT_DIR/sequences_fsq_rcm.pkl"
SPLIT_PATH="$OUT_DIR/split.pkl"
LATEST_CHECKPOINT="$OUT_DIR/ar_latest.pt"
BEST_CHECKPOINT="$OUT_DIR/ar_best.pt"

if [[ -n "$SEQUENCE_SOURCE" ]]; then
  [[ -f "$SEQUENCE_SOURCE" ]] || { echo "Missing sequence source: $SEQUENCE_SOURCE" >&2; exit 2; }
  cp -f "$SEQUENCE_SOURCE" "$SEQUENCE_PATH"
fi
if [[ -n "$SPLIT_SOURCE" ]]; then
  [[ -f "$SPLIT_SOURCE" ]] || { echo "Missing split source: $SPLIT_SOURCE" >&2; exit 2; }
  cp -f "$SPLIT_SOURCE" "$SPLIT_PATH"
fi

[[ -f "$SEQUENCE_PATH" ]] || { echo "Missing AR sequence input: $SEQUENCE_PATH" >&2; exit 2; }
[[ -f "$SPLIT_PATH" ]] || { echo "Missing AR split file: $SPLIT_PATH" >&2; exit 2; }

RESOLVED_RESUME="$RESUME_FROM"
if [[ -z "$RESOLVED_RESUME" && "$NO_AUTO_RESUME" != "1" && -f "$LATEST_CHECKPOINT" ]]; then
  RESOLVED_RESUME="$LATEST_CHECKPOINT"
fi
if [[ -n "$RESOLVED_RESUME" ]]; then
  [[ -f "$RESOLVED_RESUME" ]] || { echo "Missing resume checkpoint: $RESOLVED_RESUME" >&2; exit 2; }
  if [[ "$RESOLVED_RESUME" != "$LATEST_CHECKPOINT" ]]; then
    cp -f "$RESOLVED_RESUME" "$LATEST_CHECKPOINT"
    cp -f "$RESOLVED_RESUME" "$BEST_CHECKPOINT"
    RESOLVED_RESUME="$LATEST_CHECKPOINT"
  fi
fi

{
  echo "run_slug=$RUN_NAME"
  echo "stage=ar_long_context"
  echo "repo_root=$REPO_ROOT"
  echo "outbase=$OUTBASE"
  echo "sequence_path=$SEQUENCE_PATH"
  echo "split_path=$SPLIT_PATH"
  echo "resume_from=$RESOLVED_RESUME"
  echo "learning_rate=$LEARNING_RATE"
  echo "target_epochs=$TARGET_EPOCHS"
  echo "batch_size=$BATCH_SIZE"
  echo "dmodel=$DMODEL"
  echo "layers=$LAYERS"
  echo "max_seq_len=$MAX_SEQ_LEN"
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

export NS_OUTBASE="$OUTBASE"
export NS_OUT="$RUN_NAME"
export NS_N="999999"
export NS_AR_EPOCHS="$TARGET_EPOCHS"
export NS_AR_BS="$BATCH_SIZE"
export NS_AR_DMODEL="$DMODEL"
export NS_AR_LAYERS="$LAYERS"
export NS_AR_LR="$LEARNING_RATE"
export NS_AR_SAVE_EVERY="$SAVE_EVERY"
export NS_AR_LOG_EVERY_BATCHES="$LOG_EVERY_BATCHES"
export NS_AR_MAX_SEQ_LEN="$MAX_SEQ_LEN"

if [[ -n "$RESOLVED_RESUME" ]]; then
  export NS_AR_RESUME_FROM="$RESOLVED_RESUME"
fi

"$PYTHON" breparg_improvements/train.py --stage ar

{
  echo "finished_at=$(date -Is)"
  echo "best_checkpoint=$BEST_CHECKPOINT"
  echo "latest_checkpoint=$LATEST_CHECKPOINT"
  echo "history=$OUT_DIR/ar_history.jsonl"
} >> "$LEDGER"

echo "AR long-context training finished."
echo "Ledger: $LEDGER"
echo "Best checkpoint: $BEST_CHECKPOINT"
