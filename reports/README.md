# V13 Experiment Summaries

This directory is the lightweight entry point for understanding V13 experiment status from another machine. It intentionally contains no datasets, checkpoints, token sequences, generated CAD, complete logs, or machine-specific run directories.

Start with these repository documents:

- `reproducibility/reports/current_conclusions.md` contains the current scientific conclusions, root-cause priorities, excluded hypotheses, and next experiment order.
- `reproducibility/project_history/03_experiment_ledger/experiment_ledger.md` is the readable experiment ledger.
- `reproducibility/project_history/03_experiment_ledger/experiment_ledger.json` is the structured experiment ledger.
- `docs/full_experiment_postmortem_20260731.md` contains the full end-to-end audit and evidence interpretation.
- `docs/audits/` contains lightweight split-integrity, sequence-length, and distribution audits.

`experiment_index.json` provides machine-readable navigation to those summaries. An entry with `artifacts_external: true` means that its datasets, model weights, generated STEP files, and raw logs remain outside Git by design. Artifact identity and provenance should be represented by catalog entries and hashes rather than by committing binary payloads.

When adding a new experiment, commit its aggregate metrics and conclusion only when they are small, stable, and interpretable without the raw run directory. Put raw outputs under an ignored run directory and add or update an index entry here.

## TensorBoard logs

`reports/tensorboard/` contains the currently available lightweight TensorBoard event files copied from ignored local runs. The directory preserves each run's relative path below `local_runs/`, so logs with identical filenames from different experiments do not overwrite one another. These files total less than 0.1 MiB and are stored directly in Git; Git LFS is not required.

From the repository root, inspect them on another machine with:

    tensorboard --logdir reports/tensorboard

These are selected historical logs, not a guarantee that every experiment produced or retained TensorBoard events. The raw run directories, checkpoints, and training datasets remain excluded.

`reports/training_results/` contains richer packages for selected later runs. Each package combines the original event file with a CSV scalar export and concise JSON/Markdown summaries, allowing results to be inspected even when TensorBoard is unavailable.
