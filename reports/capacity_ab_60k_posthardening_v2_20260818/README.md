# Capacity A/B formal 60k VQ-8192/RVQ stability result

This is a lightweight, Git-safe archive of the completed formal training run.
The four tasks used 60,000 train patches, 12,000 validation patches, bf16,
batch size 128, and 100 epochs. The launcher validator reports the run as
formal-result eligible with identical train/validation inventories across all
four tasks.

## Result

| Arm | Seed | Best val MSE | Best curved parent MSE | Final perplexity | Final coverage | Non-finite events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rvq_2x4096_64d_random | 3 | 0.00083757 | 0.0019984547 | 2455.26 | 82.52% | 0 |
| rvq_2x4096_64d_random | 4 | 0.00089388 | 0.0020876739 | 2615.18 | 87.44% | 0 |
| vq_8192_64d_random | 3 | 0.00092354 | 0.0023043219 | 2858.46 | 88.17% | 0 |
| vq_8192_64d_random | 4 | 0.00109029 | 0.0027970005 | 2613.23 | 82.58% | 0 |

- All four tasks completed epochs 0 through 99 with zero skipped batches and
  zero non-finite loss, gradient, state-audit, validation-batch, or validation-
  sample events.
- This archive is reconstruction and codebook-health evidence only. The fixed 100-CAD unchanged-chain measurement decides the winner and must include RVQ's downstream sequence cost.

## Evidence

- `training_summary.json` and `training_summary.csv`: four-task metrics,
  inventory binding, finite-state totals, and cross-seed aggregates.
- `epoch_metrics.csv`: compact metrics for all 400 epochs.
- `tasks/`: metric-complete history, task manifest, train report, and sweep
  JSON files with machine-local absolute paths replaced by a stable marker.
- `logs/`: complete stdout/stderr normalized to portable UTF-8/LF text plus
  `log_summary.json`; original and archived hashes are both retained.
- `tensorboard/`: all `4` small TensorBoard
  event files. A task may have more than one file after an automatic resume;
  every segment is preserved and hash-bound.
- `checkpoint_manifest.json`: size and SHA-256 for all 12 local checkpoints.
  It does not contain checkpoint bytes.
- `source_archive_manifest.json`: separate source/archive size and SHA-256
  bindings plus the named path-redaction or identity transformation.
- `artifact_manifest.json`: size and SHA-256 for every archived file.

No checkpoint, pickle, NumPy array, raw protocol data, CAD, or STEP file is
present in this directory.
