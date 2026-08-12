# Run a five-seed 100-epoch representation cohort and surface reconstruction

This ExecPlan is a living document and follows `PLANS.md` in the repository root. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated while work proceeds.

## Purpose / Big Picture

The previous learned-VQ experiment used only two seeds and stopped at 40 and 67 epochs, so it cannot answer whether apparent differences persist after every representation has received the same optimization budget. This plan replaces that local run with a clean Protocol V6 cohort. Four representation arms are trained on the same 300,000 training patches and 12,000 validation patches for exactly 100 epochs at seeds 0 through 4. Only after all checkpoints pass integrity gates are the same parent-isolated validation CADs reconstructed at the surface-patch level. The observable result is a five-seed comparison of curved, planar, and overall surface reconstruction error with local reconstructed arrays for inspection and lightweight JSON/CSV summaries suitable for Git.

The four arms are `fsq_8192_4d`, `fsq_4096_6d`, `vq_4096_64d_random`, and `continuous_bypass_64d`. Autoregressive training, sequence regeneration, and CAD generation remain blocked.

## Progress

- [x] (2026-08-10 09:00 +08:00) Verified the superseded two-seed learned-VQ run had stopped at 40 and 67 epochs and occupied about 461.6 MB.
- [x] (2026-08-10 09:05 +08:00) Deleted `D:\luolin\V13\local_runs\vq4096_300k_2seed_20260809` after separately resolving and verifying that it was a single child of `local_runs` and that no training process remained.
- [x] (2026-08-10 09:45 +08:00) Implemented and tested a five-seed, four-arm launcher that requires exactly 100 completed epochs per arm, best and final checkpoints, exact caps, and fail-closed state.
- [x] (2026-08-10 09:55 +08:00) Implemented and tested fixed-cohort surface reconstruction with local NPZ artifacts and lightweight per-CAD, per-checkpoint, and cross-seed summaries.
- [x] (2026-08-10 00:29 +08:00) Committed and pushed the launcher, evaluator, focused tests, and plan as `d379d4b` on `experiment/protocol-v5-scaling-ladder`.
- [x] (2026-08-10 00:36 +08:00) Started the cohort and verified the exact `train=300000 val=12000` inventory gate, an active TensorBoard event file, no stderr traceback, and sustained ~97% GPU utilization on seed 0. The first epoch is still running; the launcher is left unattended as configured.
- [x] (2026-08-12 21:37 +08:00) Audited the live cohort: seeds 0, 1, and 2 completed all four fixed 100-epoch loops and passed checkpoint/cap integrity; seed 3 is active on `fsq_4096_6d`; GPU utilization is ~97% and all stderr logs are empty. Archived every currently available lightweight history, sweep, text log, TensorBoard event, state file, and artifact hash while excluding checkpoints and reconstruction arrays.
- [ ] After all five seeds finish, run automatic surface reconstruction, archive lightweight evidence, compare arms, and decide the next representation gate. AR remains blocked until this item completes.

## Surprises & Discoveries

- Observation: The superseded run was nominally configured for 100 epochs but did not train to 100 because plateau early stopping was enabled.
  Evidence: seed 0 reported 40 epochs and seed 1 reported 67 epochs in their sweep manifests.

- Observation: Existing sweep training already supported a rolling final checkpoint internally, but the stage did not expose it.
  Evidence: `_train_vqvae` accepted `save_final_path`; V6 connects it through `NS_VQ_SAVE_FINAL=1` and records `checkpoint_final` plus final epoch in the sweep manifest.

- Observation: Completing the fixed 100-epoch loop is not equivalent to numerical health. Of the 12 completed seed0-2 histories, only seed1/2 learned VQ and continuous bypass stayed fully finite; every FSQ history and both seed0 64D histories developed incomplete/non-finite epochs. Seed3 FSQ-8192 also became non-finite after epoch 14.
  Evidence: `reports/protocol_v6_5seed_100epoch_20260810/training_health_summary.json` derives finite epoch counts directly from every history row's expected and finite train/validation batch counts.

- Observation: The launcher validation is an artifact-integrity gate, not a numerical-health gate.
  Evidence: seeds 0-2 have `validation.valid=true` in `cohort_state.json` despite non-finite histories. No such arm is eligible for representation promotion or AR.

## Decision Log

- Decision: Interpret “each” as the four representation arms used by the current capacity matrix: two FSQ arms, learned VQ, and continuous bypass.
  Rationale: These arms isolate FSQ dimension/codebook restrictions from learned-codebook and continuous decoder limits on the same protocol.
  Date/Author: 2026-08-10 / Codex

- Decision: Use seeds 0, 1, 2, 3, and 4 and require `epochs_ran == 100` for every arm.
  Rationale: Five deterministic seeds provide variance estimates and satisfy the user’s request. Setting both minimum epochs and patience to 100 prevents ordinary plateau early stopping before the fixed budget is consumed.
  Date/Author: 2026-08-10 / Codex

- Decision: Reconstruct surfaces only after all training arms and seeds pass checkpoint gates.
  Rationale: This prevents partial results from being mistaken for a complete comparison and ensures the reconstruction cohort is identical across all checkpoints.
  Date/Author: 2026-08-10 / Codex

- Decision: Store reconstructed arrays locally and only version lightweight manifests, metrics, plots, and hashes.
  Rationale: Git remains usable across machines without uploading model weights or hundreds of megabytes of arrays.
  Date/Author: 2026-08-10 / Codex

## Outcomes & Retrospective

The previous two-seed local output has been removed. The new cohort and reconstruction have not yet completed. This section must be updated after the launcher and evaluator finish.

## Context and Orientation

The Git-managed runtime repository is `D:\luolin\V13\local_runs\protocol_v5_runtime_20260806`. The verified Protocol V5 data split reused by V6 is `D:\luolin\V13\local_runs\protocol_v5_scaling_run_20260806\protocol`. It contains a parent-isolated split and materialized CAD pickle files. The Python environment is `C:\Users\YU\.conda\envs\brepgen_env\python.exe` and the local upstream dependency is `D:\luolin\V13\BrepARG`.

Training is implemented by `breparg_improvements/train.py --stage vqsweep`. It reads configuration from `NS_*` environment variables. `NS_VQ_SWEEP_ARMS` selects arms, `NS_VQ_SWEEP_TRAIN_CAP` and `NS_VQ_VAL_SAMPLES` select patch counts, and `NS_VQ_SWEEP_EPOCHS`, `NS_VQ_MIN_EPOCHS`, and `NS_VQ_PATIENCE` control stopping. The sweep writes one best checkpoint and one history per arm plus `vqvae_hp_sweep.json`.

A normalized-coordinate-space surface patch is a 32 by 32 grid of XYZ points. Surface reconstruction passes the patch through encoder, quantizer or bypass, and decoder. Curved versus planar classification uses the rotation-invariant plane-residual helper already used by Protocol V5. Reconstruction outputs are grouped by CAD and parent before averaging so CADs with many faces do not dominate the reported metric.

## Plan of Work

Add `tools/run_protocol_v6_5seed_cohort.py`. It will launch one seed at a time and one four-arm sweep per seed. It must strip unrelated `NS_*` variables, inject process-local Git safe-directory configuration, write an atomic state file before and after each seed, and accept restart. A seed is complete only when the sweep contains exactly the four requested arms, each reports exactly 100 epochs, train cap is met, parent coverage is at least 90%, validation contains 12,000 patches, and all expected best checkpoints exist.

Add `tools/evaluate_surface_reconstruction_cohort.py`. It will deterministically select a fixed set of validation CADs from the verified protocol, load each of the 20 checkpoints strictly, reconstruct surface patches in bounded GPU batches, and write one JSONL row per CAD/checkpoint. For each local reconstruction it writes a compressed NPZ under the experiment workspace containing the reconstructed float32 surfaces and identifiers. It aggregates patch MSE into per-CAD curved, planar, and overall means, then averages CADs equally for checkpoint and cross-seed summaries. Nonfinite values and missing checkpoints are failures, not exclusions.

The launcher invokes the evaluator only after all training seeds are complete. It records the evaluator command and status in the same cohort state. The evaluator writes checkpoint hashes so local arrays and lightweight summaries remain bound to exact models.

## Concrete Steps

From `D:\luolin\V13\local_runs\protocol_v5_runtime_20260806`, run focused tests with:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m pytest tests\test_run_protocol_v6_5seed_cohort.py tests\test_evaluate_surface_reconstruction_cohort.py -q

After committing and pushing, launch in a hidden background process with output root `D:\luolin\V13\local_runs\protocol_v6_5seed_100epoch_20260810`. The launcher uses seeds `0,1,2,3,4`, train cap `300000`, val cap `12000`, batch size `128`, learning rate `3e-4`, and exactly 100 epochs.

Training health requires the log to report `train=300000 val=12000`, TensorBoard events to grow, no stderr traceback, and sustained CUDA utilization once batches begin. The process may then continue unattended as previously requested.

## Validation and Acceptance

Launcher unit tests must verify the exact arm set, five seeds, fixed 100-epoch environment, process-local Git safety, and strict sweep completion semantics. Evaluator tests must verify deterministic selection, per-CAD equal weighting, curved/planar aggregation, nonfinite rejection, and checkpoint discovery.

The training cohort is accepted only when 20 arm-seed rows exist, every row reports 100 epochs, all requested caps and parent coverage gates pass, and 20 best checkpoints are hash-bound. Surface reconstruction is accepted only when every checkpoint has the same CAD identities, all attempts remain in the denominator, local NPZ files exist for successful attempts, and cross-seed summaries contain means and standard deviations for each arm.

## Idempotence and Recovery

The launcher skips only seeds whose complete sweep passes all exact gates. A failed or interrupted seed is rerun in its seed directory only after the incomplete outputs are explicitly moved aside or removed by the launcher’s fail-closed recovery logic. The evaluator keys rows by checkpoint hash, arm, seed, and CAD identity so exact completed rows can be resumed. No existing Protocol V5 dataset, checkpoint, report, or Git history is deleted by V6.

## Artifacts and Notes

The output workspace is intentionally outside Git. Checkpoints, TensorBoard events, and NPZ reconstruction arrays remain there. After completion, JSONL metrics, summary JSON/CSV, state, text logs, plots, and SHA-256 manifests are copied into an allow-listed `reports/protocol_v6_*` directory for Git.

## Interfaces and Dependencies

`tools.run_protocol_v6_5seed_cohort.training_environment(...)` returns the exact process environment for one seed. `validate_sweep(...)` returns a structured pass/fail report rather than a bare boolean. `tools.evaluate_surface_reconstruction_cohort.evaluate_checkpoint(...)` returns per-CAD rows and writes local NPZ files. `aggregate_checkpoint_rows(...)` computes CAD-equal curved, planar, and overall MSE. Dependencies are NumPy, PyTorch, Pillow only if a plot is added, the existing `breparg_improvements` modules, and the local BrepARG quantizer implementation.

Revision note 2026-08-10: Created this plan after the user replaced the two-seed early-stopped run with a four-arm, five-seed, fixed-100-epoch cohort followed by surface reconstruction.
