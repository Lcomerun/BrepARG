# Implement the learned-VQ capacity A/B experiment

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current while work proceeds. This document follows `PLANS.md` in the repository root. It begins after `plans/p0_assembly_stability_boundary_consistency_execplan.md`, whose paired 60k assembly measurement found `Delta_q=13` percentage points and therefore selected capacity A/B before boundary consistency.

## Purpose / Big Picture

The paired assembly result showed that learned VQ loses 13 strict-valid percentage points relative to the continuous bypass on the same 60,000-patch cohort. This experiment separates two capacity hypotheses without changing the encoder, decoder, protocol, precision, optimizer, or training budget. One arm doubles the flat learned codebook from 4,096 to 8,192 entries. The other arm uses residual vector quantization (RVQ): a first 4,096-entry codebook quantizes the latent, and a second independent 4,096-entry codebook quantizes the remaining residual. After this implementation, the user can launch or resume an immutable four-task matrix and inspect separate stage-1 and stage-2 utilization so an apparently healthy first codebook cannot hide a collapsed second codebook.

This plan implements and tests the experiment but does not start formal GPU training. It does not modify upstream `BrepARG/`, regenerate sequences, start AR, or implement boundary consistency.

## Progress

- [x] (2026-08-17 01:25 +08:00) Read the completed P0 assembly/stability plan, confirmed `Delta_q=13 pp`, and mapped the existing learned-VQ, validation histogram, checkpoint, FeaturePool, and formal launcher contracts.
- [x] (2026-08-17 10:25 +08:00) Added the 8,192-entry learned-VQ and two-stage 4,096-entry RVQ model arms with a tuple-compatible `VQModel.quantize` information contract. RVQ quantizes `latent - stage1_quantized.detach()` and exposes one aggregate straight-through path.
- [x] (2026-08-17 10:38 +08:00) Added device-side stage histogram tracking, JSON-serializable train/validation stage usage, TensorBoard scalars, and a fail-closed stage-2 health gate. Invalid, missing, non-finite, one-code, or perplexity-at-most-one stage-2 evidence cannot select a checkpoint.
- [x] (2026-08-17 10:49 +08:00) Added `tools/run_capacity_ab_60k.py` with independent schema `capacity-ab-60k-v1`, immutable formal matrix, Protocol V5/source/inventory signatures, automatic resume, and rolling FeaturePool validation.
- [x] (2026-08-17 10:55 +08:00) Added quantizer and launcher tests covering construction, forward/backward, detached residual semantics, exact stage statistics, collapse rejection, immutable bf16 task matrix, dry-run non-mutation, RVQ FeaturePool evidence, and cross-task inventory equality.
- [x] (2026-08-17 11:18 +08:00) Ran the focused and regression suites (`155 passed`), compiled all changed Python modules, passed `git diff --check` for tracked capacity files, and executed a real Protocol V5 dry-run that returned four bf16 tasks without creating its output root.
- [x] (2026-08-17 11:36 +08:00) Closed the final artifact-validation gap: history now names `ReduceLROnPlateau`, the launcher proves its requested upstream root is exactly the one `train.py` will discover, and FeaturePool recovery remains exact for one-stage VQ and two-stage RVQ. The expanded suite passes `157 passed in 9.29s`; compilation and `git diff --check` pass; a second real Protocol V5 dry-run planned the same four tasks and left `E:/V13_experiments/capacity_ab_60k_20260817` absent. Formal training remains unstarted and the worktree remains uncommitted.
- [x] (2026-08-17 11:36 +08:00) Diagnosed the first real-CUDA smoke as an evidence-schema failure rather than a numerical failure. The VQ-8192 train and validation passes were finite, but the task signature omitted the scheduler metric and the checkpoint context omitted `parent_overlap_counts`; the fail-closed launcher correctly stopped before RVQ. The producers now emit both fields, focused and compatibility tests pass (`157 passed`), and dedicated regressions freeze the corrected contract.

## Surprises & Discoveries

- Observation: The upstream quantizer already exposes the flattened selected code indices as the third item of its `(quantized, loss, info)` contract, and full-state recovery discovers FeaturePool objects by walking named modules.
  Evidence: `D:/luolin/V13/BrepARG/quantise.py::VectorQuantiser.forward` returns `(perplexity, min_encodings, encoding_indices)`, while `breparg_improvements/training_stability.py::capture_feature_pools` records every named module with a `pool` attribute. Two independently named RVQ stages therefore fit the existing recovery format without changing upstream code.

- Observation: The current validation accumulator accepts any number of code indices per image batch because it checks reconstruction-sample count and code-index range independently.
  Evidence: `VQValidationAccumulator.update` advances bucket offsets from per-sample losses but builds the usage histogram from all flattened indices. RVQ can preserve the existing aggregate `code_usage` field by concatenating both stage index streams, while adding independent stage histograms beside it.

- Observation: A residual tensor must retain the encoder gradient through the original latent even though the stage-1 quantized value is detached from stage 2's residual path.
  Evidence: The RVQ forward uses `residual = latent - stage1_quantized.detach()`. The residual has a `SubBackward` node, while stage-1 codebook values cannot receive gradients through that subtraction. The dedicated test asserts this distinction.

- Observation: The existing P0-B writer lock uses `.p0b_writer.lock`; capacity output roots must exclude that same file when checking for unexpected pre-existing artifacts.
  Evidence: `tools/run_capacity_ab_60k.py` reuses the P0-B lock helper and its lock-file name, while every capacity run is isolated under a new output root.

- Observation: RVQ stage 2 must receive an fp32 residual even though the encoder and decoder run under bf16 autocast.
  Evidence: `ResidualLearnedVectorQuantiser.forward` casts the latent to fp32 before stage 1, subtracts the detached fp32 stage-1 reconstruction, executes stage 2 through the AMP-safe fp32 adapter, and casts only the final aggregate straight-through value back to the incoming dtype. A bf16 behavior test observes fp32 at the stage-2 boundary and finite gradients.

- Observation: Restoring only model/optimizer state is not an exact continuation when a learned quantizer's historical FeaturePool is missing.
  Evidence: `training_stability.restore_feature_pools` now requires the checkpoint module set to equal the live model's pool module set. The formal validator additionally requires one fp32 pool for VQ and two independent fp32 pools for RVQ.

- Observation: The prior VQ-4096 60k job took about 110 minutes on the local RTX 3060 and retained approximately 1.07 GiB across best, final, and rolling checkpoints per arm/seed task.
  Evidence: `p0b_stability_vq_bypass_60k_20260814` logs span 12:25 through 14:16, and its learned-VQ checkpoints are approximately 0.214, 0.214, and 0.643 GiB. The two capacity arms perform roughly twice the codebook lookup work, so the conservative scheduling estimate is 2-3 hours per task, 8-12 hours sequential, and about 4.5 GiB retained across four tasks.

- Observation: The signed scheduler contract included its implementation kind, but the human-readable history configuration initially omitted that field.
  Evidence: The formal validator requires `kind=ReduceLROnPlateau`; adding the same field to `_train_vqvae` history prevents a healthy real run from being rejected after all 100 epochs.

- Observation: Signing an arbitrary `--breparg-root` is insufficient if `train.py` will discover a different nearer `BrepARG/` directory at runtime.
  Evidence: The capacity launcher now mirrors `train.py`'s six-level discovery walk and rejects any requested root that is not exactly the discovered root; a focused test places a conflicting nearer directory and observes the fail-closed error.

- Observation: A numerically healthy task can still be scientifically unusable when producer and validator disagree about evidence fields.
  Evidence: the first CUDA smoke completed VQ-8192 with three finite train batches, two finite validation batches, and best validation reconstruction near `0.26301`, but validation rejected it because history lacked scheduler `kind` relative to the signed schema and best/final checkpoint contexts lacked parent-overlap evidence. Producer-side schema fixes and regression tests were chosen instead of relaxing validation.

## Decision Log

- Decision: Quantize the second stage from `latent - stage1_quantized.detach()`, sum both stage losses, and expose the final quantized value through one aggregate straight-through estimator.
  Rationale: Detaching the first quantized value keeps stage 2 focused on the actual residual and prevents its loss from updating stage 1 through the residual path. Reapplying one aggregate straight-through estimator gives the encoder one identity-gradient path rather than accidentally doubling decoder-to-encoder gradients by summing two independently straight-through outputs.
  Date/Author: 2026-08-17 / Codex.

- Decision: Keep `info[2]` as a concatenated flat index tensor and add named `stage_indices` and `stage_perplexities` fields.
  Rationale: Existing callers and aggregate validation continue to work unchanged, while new code can separately audit each RVQ stage. A tuple-compatible named record preserves the `VQModel` quantizer interface.
  Date/Author: 2026-08-17 / Codex.

- Decision: Define hard stage-2 collapse as missing statistics, invalid index bounds, a non-finite metric, fewer than two unique codes, or perplexity not greater than 1.0 on a non-empty validation pass.
  Rationale: This is a fail-closed structural health gate, not a post-hoc quality threshold. It detects total second-stage collapse without inventing an aggressive utilization target before the A/B data exists. Every epoch records the values, and the formal validator rejects any unhealthy epoch.
  Date/Author: 2026-08-17 / Codex.

- Decision: Add `tools/run_capacity_ab_60k.py` instead of changing the completed P0-B launcher contract.
  Rationale: `tools/run_p0b_stability_retest.py` is evidence for an already completed experiment and deliberately accepts only its original VQ/bypass arms. A dedicated schema prevents an interrupted capacity run from being confused with or resumed into the completed stability cohort.
  Date/Author: 2026-08-17 / Codex.

- Decision: Track stage usage only for quantizers that expose `stage_codebook_sizes`; single-level learned VQ keeps the legacy aggregate usage path.
  Rationale: The stage-2 collapse gate is specific to RVQ. Adding a synthetic stage to ordinary VQ would change its checkpoint selection semantics and make historical single-level runs incomparable.
  Date/Author: 2026-08-17 / Codex.

- Decision: Expose `run`, bounded `probe`, read-only `status`, and fail-closed `validate` launcher actions, but execute only `run --dry-run` in this implementation turn.
  Rationale: The checked-in capacity/repair plan requires explicit probe and validation entry points. Keeping probe bounded by the smoke caps makes later CUDA preflight reproducible without weakening the immutable formal command.
  Date/Author: 2026-08-17 / Codex.

- Decision: Treat both the scheduler kind and the actually discovered upstream source root as formal evidence, not optional descriptive metadata.
  Rationale: Either mismatch would make a run non-reproducible: the validator could reject a correct schedule after completion, or the process could execute unsigned upstream code. Both checks are cheap to enforce before allocating GPU time.
  Date/Author: 2026-08-17 / Codex.

- Decision: Add `metric=curved_parent_mse` to the signed scheduler object and copy `parent_overlap_counts` into every checkpoint context.
  Rationale: These facts already govern checkpoint selection and data-isolation validity. They must be produced and signed consistently at every evidence layer; accepting their absence in the validator would weaken the experiment after observing a failure.
  Date/Author: 2026-08-17 / Codex.

## Outcomes & Retrospective

Implementation, fail-closed validation, and dry-run preflight are complete. The formal four-task run remains intentionally unstarted. The capacity code now includes VQ-8192 and fp32-residual RVQ, per-stage history/TensorBoard evidence, exact FeaturePool resume, an immutable four-task launcher, and paired assembly coordinator support. The final regression result is 157 passing tests, and the real Protocol V5 dry-run proves the four formal task signatures without creating the E: output root. No assembly-repair source was rewritten by this plan.

## Context and Orientation

`breparg_improvements/train.py` builds the local VQ-VAE variants and contains the shared `_train_vqvae` loop. It wraps the upstream `BrepARG/quantise.py::VectorQuantiser` so codebook and FeaturePool calculations remain fp32 under mixed precision. `breparg_improvements/vqvae_metrics.py` computes full-validation reconstruction buckets and code histograms. `breparg_improvements/training_stability.py` captures model, optimizer, scheduler, scaler, random-number generator, and every named FeaturePool into an atomic rolling checkpoint.

The completed P0-B launcher in `tools/run_p0b_stability_retest.py` is the reference for immutable formal inputs, output-root writer exclusion, task signatures, exact Protocol V5 hashes, exact patch-inventory digests, and automatic continuation. The new launcher must preserve those behaviors but use schema `capacity-ab-60k-v1` and arms `vq_8192_64d_random` and `rvq_2x4096_64d_random` only.

RVQ means residual vector quantization. Stage 1 chooses a 64-dimensional code vector for each latent token. Stage 2 receives the difference between the original token and the detached stage-1 code and chooses another 64-dimensional code. The decoder receives the sum. Each stage owns a separate embedding, usage EMA, and historical FeaturePool.

## Plan of Work

Extend `breparg_improvements/train.py` with a tuple-compatible residual quantizer information type, a two-stage quantizer module, a builder, and two quantizer comparison configurations. Keep learned codebook operations inside the existing fp32 wrapper. Add helpers that extract stage indices and accumulate per-stage histograms on the active device, transferring only the final epoch histograms to the CPU. Add `train_stage_code_usage`, `val_stage_code_usage`, and `stage_usage_health` to each history row and validation checkpoint. Preserve the existing aggregate `val_code_usage` field by combining both RVQ index streams. Write stage scalars to TensorBoard.

Make checkpoint selection aware of stage health. An RVQ checkpoint cannot be selected when stage statistics are missing or when stage 2 has fewer than two unique codes or perplexity at most 1.0. Keep the old selection behavior for FSQ, single learned VQ, and bypass.

Add `tools/run_capacity_ab_60k.py`. It must sign Protocol V5 identity, source hashes, the two quantizer metadata objects, stage-collapse thresholds, precision, scheduler, sampling, seed, cap, and inventory. Formal mode accepts exactly two arms, seeds 3 and 4, 60,000 train patches, 12,000 validation patches, batch 128, 100 epochs, bf16, learning rate 3e-4, gradient clip 1.0, and the existing ReduceLROnPlateau settings. It launches one task at a time into separate directories, uses the P0-B output writer lock, and resumes only from a matching rolling checkpoint. Its validator requires exactly four tasks, contiguous epochs 0 through 99, zero non-finite/skipped events, finite lifecycle audits, correct quantizer metadata, correct stage usage on every epoch, complete checkpoint state, and identical train/validation inventory digests across all tasks.

Add focused tests in `tests/test_capacity_ab_quantizers.py` and `tests/test_run_capacity_ab_60k.py`. Use tiny latents and fake lightweight task artifacts; do not allocate the full dataset or start CUDA training.

## Concrete Steps

Run commands from `D:/luolin/V13/v6git` with `C:/Users/YU/.conda/envs/brepgen_env/python.exe`.

First run the focused quantizer and launcher tests:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe -m pytest tests/test_capacity_ab_quantizers.py tests/test_run_capacity_ab_60k.py -q

Then run the existing representation, stability, sampling, and launcher regressions:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe -m pytest tests/test_vqvae_protocol_training.py tests/test_p0b_train_integration.py tests/test_training_stability.py tests/test_run_p0b_stability_retest.py -q

After implementation, a non-mutating formal dry run has this shape:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe tools/run_capacity_ab_60k.py run --repo-root D:/luolin/V13/v6git --protocol-dir D:/luolin/V13/local_runs/protocol_v5_scaling_run_20260806/protocol --breparg-root D:/luolin/V13/BrepARG --output-root D:/luolin/V13/local_runs/capacity_ab_vq8192_rvq60k_20260817 --python C:/Users/YU/.conda/envs/brepgen_env/python.exe --dry-run

The dry run must print four planned tasks and must not create the output directory. The same command without `--dry-run` is the later formal launch command, but this implementation task must not execute it.

For the later formal run, use E: for the heavy output rather than the D: dry-run path:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe tools/run_capacity_ab_60k.py run --repo-root D:/luolin/V13/v6git --protocol-dir D:/luolin/V13/local_runs/protocol_v5_scaling_run_20260806/protocol --breparg-root D:/luolin/V13/BrepARG --output-root E:/V13_experiments/capacity_ab_60k_20260817 --python C:/Users/YU/.conda/envs/brepgen_env/python.exe

Do not run that command until the capacity implementation and its concurrently developed assembly coordinator are committed, because formal `train.py` requires a clean committed worktree and signs the commit plus source hashes. On the local RTX 3060, budget approximately 8-12 hours for the four tasks in serial and approximately 4.5 GiB for their retained best/final/rolling checkpoints.

## Validation and Acceptance

The 8,192 VQ arm must construct an 8,192 by 64 embedding and complete a finite forward/backward pass. The RVQ arm must own two distinct 4,096 by 64 embeddings and FeaturePools, quantize stage 2 from the detached stage-1 residual, return the original latent shape, produce a finite summed quantizer loss, and backpropagate finite gradients.

Given controlled stage index streams, the training and validation summaries must report exact independent token counts, unique counts, coverage, and entropy perplexity for stage 1 and stage 2. A controlled stage-2 all-zero stream must produce an unhealthy collapse result; missing stage 2 must also fail. Single-level VQ behavior and existing aggregate metrics must remain unchanged.

The launcher dry run must expose exactly four tasks and the required environment. Formal construction with any changed arm, seed, cap, epoch, batch, precision, or learning rate must raise before creating training artifacts. A synthetic complete state must validate only when all four histories and checkpoint bindings share the same exact inventory, each RVQ epoch has a healthy stage 2, and rolling state contains both FeaturePools. A changed inventory, collapsed stage 2, incomplete history, or incompatible resume signature must fail with a specific reason.

## Idempotence and Recovery

Formal output uses a new directory and schema. Repeating the identical run command skips valid completed tasks and resumes the active task from its atomic rolling checkpoint. Any changed signed input fails closed. The launcher never overwrites the completed P0-B output, never deletes a checkpoint, and never terminates another process. This implementation turn creates no formal output directory.

## Artifacts and Notes

Git may later track source, tests, the ExecPlan, concise logs, histories, TensorBoard events, and hash manifests. It must not track `.pt`, `.pth`, `.ckpt`, `.pkl`, `.npz`, STEP, raw CAD, or upstream `BrepARG/` bytes.

## Interfaces and Dependencies

In `breparg_improvements/train.py`, `ResidualLearnedVectorQuantiser.forward(latent, *args, **kwargs)` returns `(quantized, loss, info)`. `info` remains indexable at positions 0 through 2 and additionally exposes `stage_indices` and `stage_perplexities`. `quantizer_stage_indices(info)` returns a tuple of flattened tensors. `QuantizerStageUsageTracker(stage_codebook_sizes, device)` updates from those tensors and returns JSON-serializable usage summaries.

In `tools/run_capacity_ab_60k.py`, `CapacityRunConfig` validates formal arguments, `build_state(config)` produces the signed four-task plan, `validate_task(task, formal)` validates one arm/seed artifact set, `validation_summary(state)` enforces matrix and inventory equality, and `run_cohort(config, dry_run=False)` runs or resumes under one output-root writer lock.

Revision note 2026-08-17 01:25 +08:00: Created after the paired fixed-cohort measurement reported `Delta_q=13 pp`, activating the predeclared capacity A/B branch. It freezes the VQ-8192 versus RVQ-2x4096 design, stage-2 collapse gate, formal matrix, recovery contract, and no-launch boundary for this implementation task.

Revision note 2026-08-17 11:36 +08:00: Recorded the final scheduler-history and upstream-discovery hardening, the 157-test regression result, and the repeated non-mutating Protocol V5 dry-run so a later formal launcher has no known evidence-contract gap.

Revision note 2026-08-17 11:36 +08:00: Recorded the first CUDA smoke's producer/validator schema mismatch, the producer-side correction, and the regression evidence. The failed smoke root is diagnostic-only and will not be reused as formal evidence.
