# Protocol V6: five-seed 100-epoch cohort

This is a lightweight snapshot generated at `2026-08-13T10:55:00+08:00` from the local
Protocol V6 run. The formal matrix contains four representation arms at seeds
0 through 4, with 300,000 train patches, 12,000 validation patches, batch size
128, learning rate 3e-4, and 100 requested epochs per arm.

## Snapshot status

- Effective launcher status: `INTERRUPTED`.
- Last recorded launcher status: `RUNNING`; launcher PID
  `22496` alive: `False`.
- Fully completed seeds: `[0, 1, 2]`.
- Active seed: `3`.
- Numerically healthy completed arm/seed histories: `4`.
- Histories with at least one incomplete/non-finite train or validation epoch: `10`.
- Fully finite histories interrupted before their target epoch: `1`.
- Surface reconstruction: `pending`.
- Sequence regeneration and AR: blocked.

`training_health_summary.csv` and `.json` distinguish a fixed 100-epoch loop
from numerical health. `NUMERICALLY_UNSTABLE` means at least one epoch did not
have all expected train and validation batches finite; such a result must not
be promoted even when the launcher accepted checkpoint/cap integrity.

## Tracked evidence

- `cohort_state.json`: launcher state and checkpoint hashes for completed seeds.
- `seedN/`: available per-arm histories and completed sweep manifests.
- `logs/`: stdout/stderr snapshots.
- `tensorboard/`: small TensorBoard event snapshots.
- `checkpoint_manifest.json`: local checkpoint sizes and SHA-256 hashes without
  checkpoint bytes.
- `interruption_evidence.json`: Windows restart and stale-launcher evidence.
- `continuation_assessment.md`: recovery value, constraints, and recommended scope.
- `artifact_manifest.json`: byte size and SHA-256 for every archived artifact.

Model checkpoints (`*.pt`), reconstructed arrays (`*.npz`), raw protocol data,
and PID files are excluded. Surface reconstruction JSON/JSONL/CSV evidence will
be archived after the training matrix finishes and the automatic evaluator runs.
