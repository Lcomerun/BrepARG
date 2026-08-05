# V13 Experiment Summaries

This directory is the lightweight entry point for understanding V13 experiment status from another machine. It intentionally contains no datasets, checkpoints, token sequences, generated CAD, complete logs, or machine-specific run directories.

Start with these repository documents:

- `reports/protocol_v4/README.md` records the completed 100-epoch, three-seed FSQ cohort, its `NO_PROMOTED_ARM` decision, and the boundary that still blocks full-data and AR work.
- `reports/protocol_v4/fsq_abc_100epoch_three_seed_20260805.json` contains the E030 per-run checkpoint metrics, cross-seed aggregates, hashes, and artifact inventory; complete histories are beside it under `histories/`.
- `reports/protocol_v3/README.md` summarizes the clean-commit, two-seed, three-arm Protocol V3 FSQ engineering cohort and its `NO_PROMOTED_ARM` decision.
- `reports/protocol_v3/fsq_abc_15epoch_two_seed_20260804.json` contains the complete lightweight E029 histories, sampling audits, promotion bindings, and TensorBoard hashes.
- `reproducibility/reports/current_conclusions.md` contains the current scientific conclusions, root-cause priorities, excluded hypotheses, and next experiment order.
- `reproducibility/project_history/03_experiment_ledger/experiment_ledger.md` is the readable experiment ledger.
- `reproducibility/project_history/03_experiment_ledger/experiment_ledger.json` is the structured experiment ledger.
- `docs/full_experiment_postmortem_20260731.md` contains the full end-to-end audit and evidence interpretation.
- `docs/audits/` contains lightweight split-integrity, sequence-length, and distribution audits.

`experiment_index.json` provides machine-readable navigation to those summaries. An entry with `artifacts_external: true` means that its datasets, model weights, generated STEP files, and raw logs remain outside Git by design. Artifact identity and provenance should be represented by catalog entries and hashes rather than by committing binary payloads.

When adding a new experiment, commit its aggregate metrics and conclusion only when they are small, stable, and interpretable without the raw run directory. Put raw outputs under an ignored run directory and add or update an index entry here.

## TensorBoard logs

`reports/tensorboard/` contains the currently available lightweight TensorBoard event files copied from ignored local runs. The directory preserves each run's relative path below `local_runs/`, so logs with identical filenames from different experiments do not overwrite one another. The Protocol V4 package adds nine 100-epoch event files totaling about 0.82 MiB; these remain small enough for Git, so Git LFS is not required.

From the repository root, inspect them on another machine with:

    tensorboard --logdir reports/tensorboard

These are selected historical logs, not a guarantee that every experiment produced or retained TensorBoard events. The raw run directories, checkpoints, and training datasets remain excluded.

## Recovery training results

`reports/training_results/rootcause_recovery_20260717/` mirrors lightweight training evidence from the external recovery workspace. It includes all available TensorBoard events plus result-oriented JSON, JSONL, Markdown, CSV, TXT, and LOG files no larger than 5 MiB. Relative paths are preserved so that short VQ-VAE/AR training, long VQ400/AR300 training, generation audits, and recovery records remain attributable to their original runs.

The mirror deliberately excludes checkpoints, pickle datasets and token sequences, model binaries, `.err` streams, generated STEP/STL files, images, and other binary run products. The external recovery directory remains the authoritative source for those large artifacts.

`reports/training_results/` contains richer packages for selected later runs. Each package combines the original event file with a CSV scalar export and concise JSON/Markdown summaries, allowing results to be inspected even when TensorBoard is unavailable.
