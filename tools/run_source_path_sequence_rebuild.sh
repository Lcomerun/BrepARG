#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/V13}"
PYTHON="${PYTHON:-python}"
OUTBASE="${OUTBASE:-$REPO_ROOT/local_runs/ar_training/train_outputs}"
RUN_NAME="newscheme_full_v13_sourcepath_sequence"
SPLIT="$REPO_ROOT/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl"
VQVAE_CHECKPOINT="/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_complex_recovery/fsq_vqvae_best.pt"
WORKERS="2"
CHUNKS=""
RESUME="0"
MERGE_ONLY="0"
SEED_BASE="100000"
AUDIT_SAMPLE_LIMIT="5000"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'USAGE'
Usage: bash tools/run_source_path_sequence_rebuild.sh [options]

Options:
  --repo-root PATH          Repository root on the rented server.
  --python PATH            Python executable, for example /opt/conda/bin/python.
  --outbase PATH           Output root for the rebuilt sequence branch.
  --run-name NAME          Fresh sequence branch name.
  --split PATH             Split pickle with train/val/test parsed-geometry paths.
  --vqvae-checkpoint PATH  VQ-VAE checkpoint used to tokenize local geometry.
  --workers N              Parallel shard workers.
  --chunks LIST            Optional chunk subset, for example 0-3 or abc_0004,abc_0005.
  --resume                 Reuse existing shard files.
  --merge-only             Skip shard generation and merge existing shards.
  --seed-base N            Base random seed for per-chunk workers.
  --audit-sample-limit N   Number of groups per split to scan for source_path readiness.
  --cuda-visible-devices V CUDA_VISIBLE_DEVICES value.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --outbase) OUTBASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --vqvae-checkpoint) VQVAE_CHECKPOINT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    --merge-only) MERGE_ONLY="1"; shift ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --audit-sample-limit) AUDIT_SAMPLE_LIMIT="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"

[[ -f "$SPLIT" ]] || { echo "Missing split file: $SPLIT" >&2; exit 2; }
[[ -f "$VQVAE_CHECKPOINT" ]] || { echo "Missing VQ-VAE checkpoint: $VQVAE_CHECKPOINT" >&2; exit 2; }

OUT_DIR="${OUTBASE%/}/$RUN_NAME"
SHARD_DIR="$OUT_DIR/sequence_shards"
SEQUENCE_PATH="$OUT_DIR/sequences_fsq_rcm.pkl"
SUMMARY_PATH="$OUT_DIR/sequence_sharded_summary.json"
MANIFEST_PATH="$OUT_DIR/sequence_sharded_manifest.jsonl"
REPORT_PATH="$REPO_ROOT/breparg_improvements/repro_outputs/$RUN_NAME/train_report.json"
AUDIT_PATH="$OUT_DIR/source_path_audit.json"
LEDGER="$OUT_DIR/server_run_ledger.txt"

mkdir -p "$OUT_DIR" "$SHARD_DIR" "$(dirname "$REPORT_PATH")"

{
  echo "run_slug=$RUN_NAME"
  echo "stage=source_path_sequence_rebuild"
  echo "repo_root=$REPO_ROOT"
  echo "outbase=$OUTBASE"
  echo "split=$SPLIT"
  echo "vqvae_checkpoint=$VQVAE_CHECKPOINT"
  echo "workers=$WORKERS"
  echo "chunks=$CHUNKS"
  echo "resume=$RESUME"
  echo "merge_only=$MERGE_ONLY"
  echo "sequence_path=$SEQUENCE_PATH"
  echo "audit_path=$AUDIT_PATH"
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

SHARDED_ARGS=(
  --split "$SPLIT"
  --checkpoint "$VQVAE_CHECKPOINT"
  --shard-dir "$SHARD_DIR"
  --merge-output "$SEQUENCE_PATH"
  --summary "$SUMMARY_PATH"
  --manifest "$MANIFEST_PATH"
  --report "$REPORT_PATH"
  --workers "$WORKERS"
  --seed-base "$SEED_BASE"
)

if [[ -n "$CHUNKS" ]]; then
  SHARDED_ARGS+=(--chunks "$CHUNKS")
fi
if [[ "$RESUME" == "1" ]]; then
  SHARDED_ARGS+=(--resume)
fi
if [[ "$MERGE_ONLY" == "1" ]]; then
  SHARDED_ARGS+=(--merge-only)
fi

"$PYTHON" tools/run_sharded_sequence.py "${SHARDED_ARGS[@]}"

"$PYTHON" tools/audit_sequence_source_paths.py "$SEQUENCE_PATH" \
  --sample-limit "$AUDIT_SAMPLE_LIMIT" \
  --output "$AUDIT_PATH"

{
  echo "finished_at=$(date -Is)"
  echo "sequence_summary=$SUMMARY_PATH"
  echo "sequence_manifest=$MANIFEST_PATH"
  echo "sequence_path=$SEQUENCE_PATH"
  echo "source_path_audit=$AUDIT_PATH"
} >> "$LEDGER"

echo "Source-path-aware sequence rebuild finished."
echo "Ledger: $LEDGER"
echo "Sequence package: $SEQUENCE_PATH"
echo "Source-path audit: $AUDIT_PATH"
