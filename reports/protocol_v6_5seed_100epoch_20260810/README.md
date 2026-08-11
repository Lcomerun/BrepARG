# Protocol V6: five-seed 100-epoch cohort

This report is an in-progress snapshot of the formal Protocol V6 representation
cohort launched on 2026-08-10. The local run is outside Git at:

`D:\luolin\V13\local_runs\protocol_v6_5seed_100epoch_20260810`

The matrix is four representation arms (`fsq_8192_4d`, `fsq_4096_6d`,
`vq_4096_64d_random`, and `continuous_bypass_64d`) at seeds 0 through 4.
Each arm is configured for 300,000 train patches, 12,000 validation patches,
batch size 128, learning rate 3e-4, and exactly 100 epochs. Surface
reconstruction is scheduled only after all 20 arm/seed sweeps pass the launcher
integrity checks.

## Snapshot status

- Seed 0: completed all four arms at 100 epochs; launcher validation passed.
- Seed 1: completed all four arms at 100 epochs; launcher validation passed.
- Seed 2: running; the `fsq_8192_4d` arm had reached epoch 40 when this snapshot
  was collected.
- Seeds 3 and 4: not started yet.
- Surface reconstruction: not started.
- AR and sequence regeneration: still blocked.

The launcher completion gate currently checks epoch count, final epoch index,
sampling caps, parent coverage, and checkpoint existence. It does not promote a
model based on those checks alone. The histories must also be inspected for
finite losses and useful code usage before any downstream decision.

## Health finding

Seed 0 shows numerical instability in all four arms after an initially finite
phase. The learned VQ arm reached `best_val_recon=0.00154` at epoch 7 and then
reported non-finite validation samples through epoch 99; the two FSQ arms and
continuous bypass also became non-finite later in training. In seed 1, the
learned VQ and continuous-bypass arms stayed finite for all 100 epochs, while
both FSQ arms became non-finite after their early finite phase. These runs are
retained as evidence, but an arm with any non-finite validation epoch is not a
healthy representation result and must not be promoted to sequence/AR.

The promotion gate was false for every completed arm because the configured
curved-parent-MSE and perplexity criteria were not satisfied. See the per-seed
`vqvae_hp_sweep.json` and `*_history.json` files for the complete bucket metrics
and non-finite sample counts.

## Tracked artifacts

- `cohort_state.json`: launcher state and checkpoint SHA-256 records for
  completed seeds.
- `seed0/` and `seed1/`: sweep manifests and per-arm training histories.
- `logs/`: stdout/stderr snapshots for completed seeds and the active seed.
- `training_health_summary.csv` and `.json`: compact cross-seed status table.

Model checkpoints (`*.pt`), TensorBoard event files, raw protocol data, and
surface reconstruction arrays are deliberately not tracked in Git.
