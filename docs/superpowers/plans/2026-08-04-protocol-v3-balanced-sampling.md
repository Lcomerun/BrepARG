# Protocol V3 Balanced Sampling and Rotation-Invariant Validation Plan

This ExecPlan is a living document and follows the repository rules in `AGENTS.md` and `PLANS.md`. It must remain self-contained and must be updated when implementation or evidence changes.

**Goal:** replace prefix-like patch sampling with parent/source-balanced sampling, preserve original-parent provenance in clustered validation, bind every protocol consumer to one verified split, enforce a VQ-to-AR promotion gate, match shared initialization across FSQ arms, and publish a clean-commit controlled experiment.

**Architecture:** `breparg_improvements/vqvae_sampling.py` scans every requested source, preserves provenance, deduplicates before applying the requested cap, and selects records by deterministic parent/source round-robin with parent coverage taking precedence over optional curved quotas. `breparg_improvements/vqvae_metrics.py` classifies surfaces using PCA plane-fit residual and expands each unique validation patch over all original parent identities for clustered summaries. `breparg_improvements/train.py` verifies the split through one fail-closed loader at every consumer, records a reproducible run manifest, constructs all FSQ arms with matched shared initialization, and blocks sequence/AR work unless the VQ promotion metrics pass or a diagnostic override is explicit. Tests establish every protocol contract before implementation; the real-data rerun starts from an immutable commit and writes only lightweight reports/TensorBoard outputs.

## Purpose / Big Picture

The previous 10-epoch comparison was a valid pipeline smoke but sampled only 27/7 parent CADs because a small patch cap stopped after a random prefix of whole files. After this plan, a cap of 951/370 or larger will be drawn from the full requested split after exact deduplication, with a report proving how many parents contributed. A tilted plane will remain in the planar bucket, while a genuinely non-planar surface will enter the curved bucket. Validation will expose both patch counts and effective parent-cluster counts, so a later 100-epoch promotion decision cannot mistake correlated patches for independent CAD observations.

## Progress

- [x] (2026-08-04) Reviewed the Protocol V2 implementation, evidence, and independent critique.
- [x] (2026-08-04) Created isolated worktree `.worktrees/protocol-v3-balanced-sampling` from `ed2ac2b`.
- [x] (2026-08-04) Added tests for full scanning, deduplication before caps, deterministic parent/source round-robin selection, direct-parent coverage, cross-split deletion reports and exact-cap refill.
- [x] (2026-08-04) Added tests for rotation-invariant PCA buckets, parent-cluster validation summaries, nonfinite counts and TensorBoard/history propagation.
- [x] (2026-08-04) Implemented the sampling, validation and split-hash changes; the final tilted-plane regression test failed with the old score `0.601815` and passed after the sampling proxy reused the PCA residual.
- [x] (2026-08-04) Ran final compile checks and tests. Protocol V3 focused verification reports 122 passing tests; the complete suite reports 364 passed and the same 16 excluded/baseline failures described below.
- [x] (2026-08-04) Ran a preliminary real-data three-arm cohort for 15 epochs and two seeds. Independent review invalidated its parent-cluster values and strict ablation interpretation, so its E029 JSON and six TensorBoard events are quarantined from the final commit.
- [x] (2026-08-04) Added failing regression tests and repaired provenance-aware clustering, verified split loading, VQ promotion, coverage-first curved sampling, shared initialization, matched training RNG resets and run-manifest capture. The focused suite now reports 137 passing tests before final full verification.
- [x] (2026-08-04) Verified the review repairs: `compileall` succeeds, the focused suite reports 139 passed, the full suite reports 381 passed and the unchanged 16 excluded-tree/legacy-fixture/Python-version baseline failures. The real Protocol V2 split passes the strict loader with 795/100/99 records.
- [x] (2026-08-04) Bound promotion to the actual best checkpoint, its best-epoch validation metrics, FSQ levels, split/protocol hashes and clean Git commit; made artifact binding non-overridable in sequence/AR stages and embedded the binding in generated sequence packages.
- [x] (2026-08-04) Re-verified the completed source: 159 focused tests pass, `compileall` and diff checks succeed, the real 795/100/99 split is SHA-bound with zero overlap, and the full suite reports 401 passed with the same 16 documented baseline failures.
- [x] (2026-08-04) Stopped the first clean seed-0 rerun before completion after final review proved that `curved_fraction=0` still selected each parent's highest-curvature representative; quarantined that partial local run from all evidence.
- [x] (2026-08-04) Added red-green repairs for archive-qualified materialization, unbiased zero-curvature parent representatives, duplicate-safe explicit curved quota replacement, longitudinal curved/usage checkpoint selection, and promotion-aware sweep winner reporting.
- [x] (2026-08-04) Re-ran the four-module focused suite with 170 passes and audited a real 1,200/463 collection: 795/100 sources scanned, zero failures/overlaps, 95.98%/100% final parent coverage, and a natural 63.8% curved share among selected surfaces rather than forced 100% curved representatives.
- [x] (2026-08-04) Re-ran the complete suite before the immutable experiment commit: 412 tests pass and the remaining 16 failures are exactly the documented 11 excluded-`BrepARG/`, one legacy sequence-fixture, and four Python 3.10 `Path.write_text(newline=...)` baseline failures.
- [ ] Commit source/tests/plan as the immutable experiment revision.
- [ ] Re-run the two-seed, three-arm 15-epoch cohort from that clean commit and rebuild E029 plus six lightweight TensorBoard events.
- [ ] Commit the corrected evidence, push the branch, verify the remote SHA and audit the uploaded tree for excluded data, weights, `BrepARG/`, `papers/` and files over 10 MiB.

## Surprises & Discoveries

- Observation: the earlier cap selected complete files until roughly 1.2× the cap, yielding 30/7 source keys and 27/7 parents.
  Evidence: 951/370 final patches came from 27/7 parents even though the Protocol split contains 323/100 parents.
- Observation: the existing surface proxy uses axis-aligned spans and can classify a tilted plane as curved.
  Evidence: the previous metric definition is `min(axis_span)/max(axis_span)` in `vqvae_metrics.py`.
- Observation: the repository's prior worktree contains ACL-protected pytest temporary directories.
  Evidence: the new worktree starts clean from the pushed commit and avoids those paths.
- Observation: NumPy SVD and `eigvalsh` crash this Windows environment with process code `0xc06d007f` even for the required 3 by 3 covariance calculation.
  Evidence: the stable implementation uses a pure-Python Jacobi iteration and the focused test suite completes normally.
- Observation: exact train/validation duplicates are more numerous when validation is fixed in full before train selection than in the early small audit.
  Evidence: the formal cohort removed 277 candidate train records, comprising 153 surfaces and 124 edges, before rebalancing to exactly 12,000 train patches.
- Observation: one reconstructed materialized pool is incomplete, but the original Protocol V2 split paths required for the formal rerun remain intact.
  Evidence: a fresh audit found all 994/994 source paths in the 795/100/99 train/validation/test split, with status `VERIFIED`, zero parent overlaps and split SHA-256 `df72b5757c3aabc89c707fd351c086ca8914cd96a49868decb8d15c104b17357`.
- Observation: the earlier single-seed 4096/6D lead did not survive the larger two-seed cohort.
  Evidence: final entropy perplexity ranges are 840.20-867.53 for 8192/4D, 58.14-152.04 for 4096/6D and 61.97-610.50 for 8192/6D. Only 8192/4D clears the 800 reference in both seeds.
- Observation: the preliminary cohort did not share all non-quantizer initialization across arms.
  Evidence: constructing the legacy `VQModel` with 4096 versus 8192 embeddings consumes different RNG amounts before `post_quant_conv` and decoder initialization; a seed-17 comparison found 98 differing common non-quantizer tensors.
- Observation: deduplicated validation records can represent exact patches from more than one original parent CAD.
  Evidence: the preliminary validation inventory reduced 9,006 occurrences to 4,637 unique patches and included 513 multi-record exact groups, while the accumulator used only the representative `parent_id`.
- Observation: the split SHA and the no-AR decision were not yet executable pipeline gates.
  Evidence: direct `vqsweep`/`vqvae` stages loaded `split.pkl` without calling `stage_split`, and `--stage all` could proceed after finite global MSE without checking perplexity or curved parent-cluster MSE.
- Observation: the first promotion implementation evaluated the last epoch while loading the best checkpoint, and helper-level binding tests did not prove that stage entry points used the binding.
  Evidence: `_train_vqvae` saved before constructing full validation metrics, while `stage_vqvae` and `stage_vqsweep` read `last_val_metrics`; new regressions distinguish best epoch zero from a worse final epoch and intercept every downstream stage before input loading.
- Observation: archive-qualified source identities were collapsed again during materialization, and zero requested curved quota still biased every parent representative toward maximum curvature.
  Evidence: two archives containing the same member produced two split rows pointing to one overwritten path, while four parents with one flat and one curved record each produced four curved selections at `curved_fraction=0`.
- Observation: replacing flat parent representatives to satisfy an explicit curved quota can leave the replacement in a previously built curved candidate list.
  Evidence: with two parents, twelve records, target eight, quota 0.75 and seed one, one curved `record_id` appeared twice before the append loop rechecked selected identities.

## Decision Log

- Decision: create a new branch from `ed2ac2b` instead of modifying the previous experiment branch.
  Rationale: keep the accepted Protocol V2 evidence immutable and isolate the larger sampling change.
  Date/Author: 2026-08-04 / Codex.
- Decision: scan all requested sources before applying the cap in Protocol V2 mode.
  Rationale: `require_all_paths=True` already validates every source; full scanning removes the prefix-file bias without changing the fail-closed contract.
  Date/Author: 2026-08-04 / Codex.
- Decision: deduplicate exact patch content before cap selection.
  Rationale: requested and effective sample counts become interpretable, and repeated primitives do not consume the cap multiple times.
  Date/Author: 2026-08-04 / Codex.
- Decision: use PCA plane-fit residual, normalized by total surface spread, as the default rotation-invariant curved proxy.
  Rationale: a rigid rotation preserves singular values, while an axis-aligned bounding box does not.
  Date/Author: 2026-08-04 / Codex.
- Decision: make parent coverage a warning by default and a hard gate only when an explicit minimum is configured.
  Rationale: tiny unit-test fixtures and exploratory runs may legitimately contain fewer parents; the real 100-epoch run will set the gate to 0.90.
  Date/Author: 2026-08-04 / Codex.
- Decision: fix validation first, collect a train candidate reserve equal to the validation size, remove validation exact hashes from train and then perform a final balanced selection to the requested train cap.
  Rationale: validation remains authoritative while exact filtering cannot silently shrink a nominal 12,000-patch train cohort.
  Date/Author: 2026-08-04 / Codex.
- Decision: implement the 3 by 3 symmetric eigensolver in pure Python instead of calling NumPy LAPACK.
  Rationale: the normal linear-algebra calls terminate the Windows process in this environment; the small fixed-size Jacobi iteration is deterministic and covered by rotation and curvature tests.
  Date/Author: 2026-08-04 / Codex.
- Decision: run 15 epochs rather than call the experiment the planned 100-epoch capacity test.
  Rationale: fifteen epochs over 12,000/4,637 patches and two seeds is enough to expose large seed instability while remaining explicitly an engineering cohort.
  Date/Author: 2026-08-04 / Codex.
- Decision: withdraw the preliminary 8192/4D candidate ranking until a matched-initialization rerun.
  Rationale: the arm construction changed decoder initialization as well as FSQ dimension/codebook size, so the ranking is not a clean architecture comparison.
  Date/Author: 2026-08-04 / Codex.
- Decision: disclose that the six runs were launched from an uncommitted worktree based on `ed2ac2b63f70983cac6d84d14a9712ac5c8b7fae`.
  Rationale: source now lives on this branch, but there was no immutable run commit at launch; future formal runs must commit before execution.
  Date/Author: 2026-08-04 / Codex.
- Decision: count a unique validation patch once for every original parent listed in `provenance_parent_ids` when computing parent-cluster metrics, while retaining one reconstruction for patch-level metrics.
  Rationale: this preserves efficient exact deduplication without assigning a shared primitive arbitrarily to one CAD.
  Date/Author: 2026-08-04 / Codex.
- Decision: fail closed on an unbound legacy protocol summary in all new formal runs.
  Rationale: accepting a missing `split_pickle_sha256` makes an immutable cohort unverifiable; the Protocol V2 artifact used for rerun must have an explicit matching SHA.
  Date/Author: 2026-08-04 / Codex.
- Decision: prioritize one selected record per available parent before satisfying an optional curved quota.
  Rationale: curved oversampling is best effort and must never consume the slots needed for the configured parent-coverage guarantee.
  Date/Author: 2026-08-04 / Codex.
- Decision: treat representation thresholds as overridable only for diagnostic progression, but never allow the override to bypass checkpoint SHA, FSQ levels, split/protocol hash or clean Git revision binding.
  Rationale: an override may deliberately inspect a weak representation, but it must not silently combine unrelated artifacts or source revisions.
  Date/Author: 2026-08-04 / Codex.
- Decision: save each sweep arm's ignored best checkpoint and derive its promotion observation from the metrics stored inside that checkpoint.
  Rationale: ranking or promotion based on final-epoch metrics can disagree with the model weights actually selected by best validation loss.
  Date/Author: 2026-08-04 / Codex.
- Decision: preserve archive identity in materialized paths as `split/archive_stem/member` and fail closed if those qualified targets still collide.
  Rationale: the manifest source key already distinguishes archives; the filesystem artifact must preserve the same identity before the protocol can safely expand across chunks.
  Date/Author: 2026-08-04 / Codex.
- Decision: use seeded round-robin representatives for parent coverage and invoke curvature ranking only for an explicit positive curved quota.
  Rationale: a zero quota means natural deterministic sampling, not implicit per-parent curved oversampling.
  Date/Author: 2026-08-04 / Codex.
- Decision: distinguish global-MSE optimization from representation-checkpoint selection, and publish no sweep winner when no checkpoint passes the absolute representation gate.
  Rationale: the saved checkpoint must improve curved parent-cluster MSE while keeping perplexity and coverage at least 90% of their prior finite means; a failed arm may remain in diagnostic MSE ranking but cannot be called promoted.
  Date/Author: 2026-08-04 / Codex.

## Outcomes & Retrospective

Protocol V3 removes the prefix-file failure that made a small patch cap look random while covering only 27/7 parent CADs. The production collector scans every requested source, exact-deduplicates before capping and rotates over parents and their source files. The formal cohort reaches 322/323 train parents and 100/100 validation parents after exact filtering, with no source, parent or exact-content overlap. Validation now reports both patch-weighted and parent-cluster reconstruction, rotation-invariant surface buckets, aggregate full-validation code usage and explicit nonfinite counts.

Post-review hardening additionally expands every unique validation patch over all provenance parents for CAD-cluster metrics, verifies the SHA-bound split in every VQ/sequence consumer, makes representation promotion executable before sequence/AR work, protects parent coverage from curved quotas, and matches both shared network initialization and subsequent training RNG streams across FSQ arms.

The final source hardening stores complete best-epoch validation metrics and immutable run context in every best checkpoint. Promotion is rebuilt from that payload, records the checkpoint SHA, and is verified against the current split, FSQ levels and clean Git commit before sequence or AR inputs are opened. Sequence packages carry the same binding, so stale packages fail closed even when the explicit weak-representation override is enabled.

The final review found that the first clean rerun would still be scientifically confounded, so it was terminated and excluded before publication. Materialization now retains archive identity, zero curved quota follows seeded natural round-robin representatives, and representation checkpoints use a longitudinal curved-MSE plus stable-usage selector. Sweep JSON separately exposes global-MSE ranking and promotion-eligible candidates; if all arms fail the absolute gate, `winner` is null and status is `NO_PROMOTED_ARM`.

The preliminary 15-epoch experiment is retained locally only as debugging evidence. It showed that the larger cohort and logging path run, but it cannot select an architecture: shared decoder initialization differed across arms, and its parent-cluster values assigned each deduplicated patch to only one representative CAD. The corrected cohort must be rerun from a clean commit before any arm ranking is published. Sequence regeneration and AR training remain blocked by executable promotion criteria, not merely by report prose.

The final lightweight deliverables will be a rebuilt `reports/protocol_v3/fsq_abc_15epoch_two_seed_20260804.json`, `reports/protocol_v3/README.md` and six corrected TensorBoard event files under `reports/tensorboard/protocol_v3_fsq_abc_15epoch_two_seed_20260804/`. The currently generated versions must not be staged. Raw CADs, checkpoints, diagnostic runs and local sweep directories remain ignored. Formal experiments begin only after the exact source revision is committed.

## Context and Orientation

The repository is a Python VQ-VAE/FSQ training project. Protocol V2 data are represented by a `split.pkl` containing source paths grouped as `train`, `val`, and `test`. `vqvae_sampling.py` loads parsed CAD pickle files into patch records with `source_path`, `source_key`, `parent_id`, `kind`, and `array`. `train.py` calls `collect_protocol_vq_data`, converts records to tensors, and supplies validation bucket labels to `_train_vqvae`. `vqvae_metrics.py` aggregates code usage and reconstruction metrics. The `BrepARG/` and `papers/` directories are intentionally out of scope and must remain unmodified and untracked.

## Plan of Work

First add tests that construct several synthetic parent/source files with duplicate patches and assert that collection loads every source, deduplicates before applying the cap, and returns records from the configured minimum fraction of parents. The test will also assert summary fields such as `scan_complete`, `unique_records_before_cap`, `parent_cads_contributing`, and `parent_coverage`.

Next add metric tests with a plane rotated in 3-D and a bent surface. The rotated plane must be `surface_planar_like`; the bent surface must be `surface_curved_proxy`. Add a clustered summary helper that averages patch MSE within each parent first, then averages parent means, while retaining patch counts, parent counts, and nonfinite sample counts.

Implement sampling by adding a deterministic helper that groups records by canonical parent/source identity, shuffles groups with a local seed, and emits one record per group in round-robin passes. In Protocol V2 collection, load all requested paths, apply source caps, deduplicate exact records, then use the helper to select the requested number. Keep the existing complex/curved weighting as an optional selection phase over the balanced candidate pool. Add `min_parent_coverage` and fail with a clear `RuntimeError` when configured and unmet.

Implement PCA classification in `vqvae_metrics.py`: center the point cloud, construct its 3 by 3 covariance matrix and compute the symmetric eigenvalues with a pure-Python Jacobi iteration. Use the square root of the smallest-to-largest variance ratio as the plane residual. A small finite threshold remains configurable; an empty or degenerate patch is planar-like. Add `summarize_parent_cluster_mse` and extend `VQValidationAccumulator` to accept parent labels and report `nonfinite_samples`, `nonfinite_parents`, and clustered MSE.

Thread parent labels from `collect_protocol_vq_data` through `_train_vqvae` and TensorBoard/history metadata. Keep global MSE and existing bucket MSE for continuity, but make the parent-cluster values the reported promotion metric for the larger experiment. Update stage defaults so `stage_vqsweep` uses the same full-scan and parent-coverage settings as `stage_vqvae` rather than a hidden prefix cap.

Finally run the focused tests, compileall, and a real-data experiment using the existing Protocol V2 `split.pkl`. The completed engineering cohort uses three arms, `[8,8,8,16]`, `[4,4,4,4,4,4]`, and `[4,4,4,4,4,8]`, with seeds 0 and 1, 15 epochs, batch 128, learning rate `3e-4`, 12,000 train patches, 4,637 validation patches, no oversampling and all loss weights equal to one. It is labeled as a 15-epoch engineering comparison and not as the originally contemplated 100-epoch capacity result.

Before that rerun, add regression tests and implementations for five review gates. A shared validation patch must contribute its loss to every provenance parent. Every stage that consumes the split must call `load_verified_protocol_split`, which reads bytes once and verifies summary status, the exact three zero overlap keys, SHA binding and non-empty train/validation lists. `evaluate_vq_promotion` must require finite perplexity at least 800, finite curved parent-cluster MSE at most `5e-5`, no nonfinite validation samples and acceptable parent coverage; sequence and AR fail closed unless that result is eligible or `NS_ALLOW_UNPROMOTED_VQ_DOWNSTREAM=1` is explicit. Balanced selection covers as many distinct parents as the target permits before curved quota fill. FSQ construction uses a fixed dummy legacy embedding count so equal seeds produce identical common non-quantizer parameters across all arms. The sweep report captures Git commit/dirty state, argv, relevant `NS_` environment, GPU, PyTorch/CUDA/cuDNN, TF32, AMP, seed, split/protocol hashes, caps and model/optimizer settings.

## Validation and Acceptance

Run from `D:\\luolin\\V13\\.worktrees\\protocol-v3-balanced-sampling`:

    C:\\Users\\YU\\.conda\\envs\\brepgen_env\\python.exe -m pytest -p no:cacheprovider --basetemp=local_runs/protocol_v3_pytest tests/test_cad_protocol.py tests/test_vqvae_protocol_sampling.py tests/test_vqvae_protocol_training.py tests/test_vqvae_metrics.py -q

The new tests must pass. Then run:

    C:\\Users\\YU\\.conda\\envs\\brepgen_env\\python.exe -m compileall -q breparg_improvements tools tests
    C:\\Users\\YU\\.conda\\envs\\brepgen_env\\python.exe -m pytest -p no:cacheprovider --basetemp=local_runs/protocol_v3_pytest_all tests -q

Acceptance for the real-data collection is: `scan_complete=true`, `failed_paths=0`, exact split overlap remains zero, `parent_coverage` is at least 0.90 for both train and validation, and the report states the effective parent counts. Acceptance for the larger experiment is finite history, no validation epoch with silent nonfinite samples, and a three-arm table that includes patch-level and parent-cluster metrics. It is not acceptable to promote AR from this experiment alone.

The review-repaired source passes 170 tests with the exact four-file focused command above. The fresh complete-suite run reports 412 passed and the same 16 documented baseline failures: 11 require the intentionally excluded `BrepARG/` tree, one uses a legacy sequence fixture without `ordering`, and four call the Python 3.11 `Path.write_text(newline=...)` API under Python 3.10. No Protocol V3 test fails. The strict real-data loader accepts 795/100/99 records with protocol SHA `43d0c5b36375cc78f3386a78a020a9baacc5a314372380f29e2eedb446345e6f`, split SHA `df72b5757c3aabc89c707fd351c086ca8914cd96a49868decb8d15c104b17357`, and all three actual parent-overlap counts equal to zero. The corrected six histories must contain 15 finite validation epochs, provenance-aware contribution counts, explicit longitudinal checkpoint-selection records, checkpoint-bound promotion decisions and matched-initialization metadata. Six TensorBoard events must match the rebuilt histories tag-for-tag and step-for-step. The inventory must again report exact caps, at least 90% parent coverage and zero source, parent and exact-hash overlap.

## Idempotence and Recovery

Sampling is deterministic for a fixed seed and sorted source inventory. Re-running collection overwrites only ignored local run outputs. If a larger run fails because of GPU memory or missing CUDA, keep the code and tests, record the failure, and rerun with fewer epochs or a smaller micro-batch without changing the scientific controls. Never delete the prior Protocol V2 reports or modify `BrepARG/` or `papers/`.

## Artifacts and Notes

Tracked artifacts will be limited to source code, tests, the plan, and a concise report. Raw datasets, checkpoints, full histories, and generated CAD remain ignored under `local_runs/` or other existing ignore rules. The report must include SHA-256 hashes for any lightweight JSON/TensorBoard evidence that is committed.

The corrected evidence JSON will embed each compact 15-row history so another machine can inspect the complete curves without a checkpoint or local run directory. Its curated sampling section removes the internal helper's `min_parent_coverage=0.0` field and instead states `gate_scope: collect_protocol_vq_data candidate and post-filter final inventory`. It omits long parent UUID lists and machine-specific output paths. Event sizes and hashes are recorded only after the clean-commit rerun.

## Interfaces and Dependencies

The implementation must preserve these public functions:

    collect_vqvae_sample_records(paths, cap, ...)
    deduplicate_patch_records(records)
    collect_protocol_vq_data(split, train_cap=..., val_cap=...)
    patch_bucket(record, curved_threshold=...)
    summarize_bucket_mse(buckets, per_sample_mse)

New helpers should be pure and testable:

    balanced_round_robin_records(records, target, seed=0)
    surface_plane_residual(surface)
    summarize_parent_cluster_mse(parent_groups, per_sample_mse)
    load_verified_protocol_split()
    evaluate_vq_promotion(metrics, min_perplexity=800, max_curved_parent_mse=5e-5)

All new behavior must use only Python, NumPy, PyTorch, and the existing project dependencies; no new package is required.

Revision note 2026-08-04: updated the living plan after implementation and the two-seed 15-epoch cohort. The revision records the pure-Python PCA fallback, exact-cap refill, formal coverage numbers, changed FSQ candidate decision, evidence limitations and remaining final verification/push work.

Revision note 2026-08-04: recorded final focused, compile and full-suite results. It also documents the corrected non-planar curvature fixture and classifies all 16 remaining full-suite failures as pre-existing excluded-tree, legacy-fixture or Python-version failures.

Revision note 2026-08-04: independent final review found that the preliminary evidence used representative-parent clustering, direct stages could bypass split SHA verification, AR promotion was prose-only, curved quota could reduce coverage, and FSQ arms consumed different RNG amounts before decoder initialization. This revision quarantines the old E029/events, withdraws the architecture ranking, defines the five fail-closed repairs plus manifest capture, and requires a clean-commit rerun before publication.

Revision note 2026-08-04: completed the review-driven red-green repairs, added a second RNG reset after model construction, recorded the 994/994 source-path audit and exact split SHA, and removed the preliminary E029 ranking from current conclusions pending the clean-commit rerun.

Revision note 2026-08-04: recorded fresh post-review verification: 139 focused tests pass, `compileall` succeeds, the full suite has 381 passes and only the same 16 documented baseline failures, and the strict loader accepts the real 795/100/99 Protocol V2 split.

Revision note 2026-08-04: completed the final checkpoint/promotion contract after an additional review. Best checkpoints now carry best-epoch metrics and clean run context, all downstream stages verify artifact binding independently of the representation override, sequence packages preserve that binding, and fresh verification reports 159 focused plus 401 full-suite passes with no new failure category.

Revision note 2026-08-04: a final clean-commit review found archive materialization collisions, implicit curved oversampling at zero quota, and unconditional sweep winner reporting. The first seed-zero rerun was terminated and excluded. This revision records the red-green fixes, adds longitudinal curved/usage checkpoint selection, replaces the ambiguous focused count with its exact four-file command, and requires a new clean commit before any experiment resumes.

Revision note 2026-08-04: recorded the final pre-commit complete-suite result of 412 passes and confirmed that all 16 remaining failures stay within the three previously documented baseline categories, with no Protocol V3 regression.
