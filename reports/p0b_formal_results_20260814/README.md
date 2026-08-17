# P0-B formal 60k VQ/bypass stability result

This is a lightweight, Git-safe archive of the completed formal training run.
The four tasks used 60,000 train patches, 12,000 validation patches, bf16,
batch size 128, and 100 epochs. The launcher validator reports the run as
formal-result eligible with identical train/validation inventories across all
four tasks.

## Result

| Arm | Seed | Best val MSE | Best curved parent MSE | Final perplexity | Final coverage | Non-finite events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| continuous_bypass_64d | 3 | 0.00050906 | 0.0011781044 | n/a | n/a | 0 |
| continuous_bypass_64d | 4 | 0.00050326 | 0.0010659431 | n/a | n/a | 0 |
| vq_4096_64d_random | 3 | 0.00115389 | 0.0026052967 | 1632.88 | 97.36% | 0 |
| vq_4096_64d_random | 4 | 0.00115876 | 0.0028966157 | 1776.56 | 98.44% | 0 |

- All four tasks completed epochs 0 through 99 with zero skipped batches and
  zero non-finite loss, gradient, state-audit, validation-batch, or validation-
  sample events.
- Learned VQ final codebook coverage is
  `97.90%`
  on average and final perplexity is
  `1704.72`.
- Mean learned-VQ curved parent MSE is
  `2.452x` the continuous-bypass mean.
  This metric is representation evidence, not an assembly-validity result.
- The next decision requires the fixed 100-CAD, same-cohort assembly comparison
  for learned VQ seed 3 and bypass seed 3.

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
