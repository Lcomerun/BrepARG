# Diagnose assembly, stabilize representation training, and gate boundary consistency

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current while work proceeds. This document follows `PLANS.md` in the repository root. `AGENTS.md` names `.agent/PLANS.md`, but that path is absent; the checked-in root `PLANS.md` is the available authority.

## Purpose / Big Picture

Recent experiments show two upstream failure sources before autoregressive generation can be evaluated. First, the same 100 validation CADs are not reliably assembled even from their original normalized patches: the original control was 84/100 valid in the calibration runner. Second, the fixed 100-epoch representation cohort repeatedly became non-finite and was then interrupted by a Windows update. This plan turns both observations into reproducible, bounded experiments.

After P0-A, a reader can inspect every original-control failure and see the exact OpenCascade stage, exception, strict-validity component, joint-optimization sensitivity, and sewing-tolerance sensitivity. At least 80 percent of the 16 original-control invalid CADs must receive a concrete primary attribution and the report must contain an actionable repair list. After P0-B, learned VQ and continuous bypass at seeds 3 and 4 train on the same 60,000 patches with no non-finite train or validation batch, complete checkpoints can resume after interruption, and the fixed seed-3 best checkpoints from both arms receive the same frozen 100-CAD reconstruction and assembly measurement used for the published 84/70/49 comparison. Measuring both 60k arms removes the data-scale confound in the historical 300k bypass number.

Only after both P0 gates pass may the project add a shared-boundary consistency loss. That later experiment samples adjacent face-edge pairs, maps normalized-coordinate-space patches into world-coordinate space, computes a differentiable nearest-neighbor boundary penalty with `torch.cdist`, and compares weights 0, 0.1, and 1.0. Its acceptance metric is strict 100-CAD assembly validity, not reconstruction MSE. Sequence regeneration and autoregressive training remain blocked throughout this plan.

## Progress

- [x] (2026-08-13 15:00 +08:00) Confirmed no training process is alive and the RTX 3060 is idle, so P0-A CPU work and P0-B implementation cannot collide with the interrupted V6 run.
- [x] (2026-08-13 15:10 +08:00) Froze the P0-A cohort to the 16 `original` rows marked invalid in `reports/assembly_calibration_100cad_20260809/calibration_manifest.jsonl`; six failed before STEP and ten saved STEP but failed the original strict check.
- [x] (2026-08-13 15:20 +08:00) Located stage-level evidence in the six pre-STEP failures: three curve B-spline fits, two wire builds, and one shell-to-solid conversion.
- [x] (2026-08-13 15:30 +08:00) Identified P0-B defects: the formal loop unscales before clipping but does not validate or record the returned gradient norm; its non-finite stop is delayed by `min_epochs=100`; VQ checkpoints omit optimizer/scaler/scheduler/RNG state; and skipped non-finite batches can leave a nominally completed but unusable history.
- [x] (2026-08-13 22:10 +08:00) Implemented and tested the P0-A stage-aware runner, compact snapshot generator, classification gate, and repair checklist; the focused P0-A and snapshot suite passes 29 tests.
- [x] (2026-08-13 22:10 +08:00) Completed all 96 P0-A attempts for the frozen 16 cases. All 16 received concrete attribution: ten wire self-intersections, three curve-fit failures, two wire-build failures, and one non-unit solid. The 100-percent attribution rate passes the 80-percent gate.
- [x] (2026-08-14 00:20 +08:00) Implemented explicit fp32/fp16/bf16 handling, effective gradient clipping, plateau LR decay, strict non-finite fuse, atomic full-state checkpoints, exact automatic resume, the immutable four-task launcher, and the gated learned-VQ 100-CAD measurement coordinator. The combined P0-B/VQ suite passes 105 tests; assembly/P0-A regression passes 30 tests; sampling and V5/V6 launcher regression passes 78 tests.
- [x] (2026-08-13 23:04 +08:00) Changed Windows Update active hours from `09:00-03:00` to `12:00-06:00`, verified the write by re-reading the registry, archived the reversible original values, and confirmed that Windows Update and CBS do not report a pending reboot. A generic pending-file-rename signal remains documented, so automatic resume is still mandatory.
- [x] (2026-08-14 11:20 +08:00) Replaced per-batch full model/optimizer/FeaturePool scans with lifecycle audits at startup or post-resume and once per epoch immediately before checkpoint writes. Loss and reduced gradient norm remain immediate batch-level fuses. Finite full-state audits now aggregate to one host-read scalar per device, record their cadence and tensor coverage in history/checkpoints, and pass 95 focused P0-B and VQ regressions.
- [x] (2026-08-14 11:50 +08:00) Separated the bounded precision-probe launcher contract from formal Protocol V5 sampling by disabling parent balancing, pre-cap deduplication, exact-cap, and the 90-percent parent-coverage gate. A first live replacement probe then exposed one remaining hard-coded full-source-scan flag inside `train.py`.
- [x] (2026-08-14 11:50 +08:00) Hardened the formal launcher with an output-root operating-system writer lock, immutable Protocol V5 hashes, exact measured 60,000/12,000 caps, content-and-order inventory digests, four-task inventory equality, full checkpoint deserialization, and runtime resume compatibility. Read-only `status` and `validate` no longer mutate state.
- [x] (2026-08-14 11:50 +08:00) Re-ran the implementation regressions after wiring inventory and lifecycle evidence into the 100-CAD coordinator: 131 P0-B/inventory/measurement tests and 91 assembly/V5/V6/sampling tests passed, for 222 passing tests.
- [x] (2026-08-14 12:00 +08:00) Removed the hard-coded `require_all_paths=True` from the probe path. A full scan is now derived from the four formal cohort requirements, so formal behavior remains fail-closed while a bounded probe stops after reaching its cap. A six-source regression proves cap 1 loads one source and reports `scan_complete=false`; the two invalid probe directories remain local evidence and are never resumed. The final suites pass 132 plus 91 tests, for 223 passing tests; Python compilation and `git diff --check` pass.
- [x] (2026-08-14 12:06 +08:00) Ran the bounded fp32 `v3` probe. Learned VQ reached GPU training immediately and completed one epoch in about five seconds with 15/15 finite train batches, 12/12 finite validation batches, finite gradients, and finite lifecycle audits. The launcher correctly stopped before bypass because its smoke validator still assumed requested caps after final deduplication produced 116 train and 90 validation patches.
- [x] (2026-08-14 12:08 +08:00) Changed only smoke validation to accept a positive, digest-bound realized inventory no larger than the requested cap. Formal validation still requires exactly 60,000/12,000 in every artifact. Added regressions for reduced smoke inventories and formal 59,999 rejection; the full regression count is now 225.
- [x] (2026-08-14 12:09 +08:00) Completed fresh fp32 and bf16 CUDA probes for learned VQ and continuous bypass. Both precisions used identical 116-train/90-validation inventory digests, completed all train and validation batches with zero non-finite events, and passed history plus best/final/rolling checkpoint validation. Selected bf16 under the predeclared rule.
- [x] (2026-08-14 12:14 +08:00) Completed a real bf16 interruption-and-resume smoke. The first process was stopped after atomic rolling epoch 0; a concurrent second writer was rejected; the identical restart recorded stale/unreleased-lock recovery, restored full state after epoch 0, and completed epoch 1 with final history exactly `[0, 1]` and no non-finite event.
- [x] (2026-08-14 19:37 +08:00) Completed learned VQ and continuous bypass at seeds 3 and 4 on the same exact-hash-bound 60,000/12,000 patch protocol for 100 epochs. All four tasks reached epoch 99 with zero non-finite events, zero skipped batches, valid resumable checkpoints, and identical train/validation inventories.
- [x] (2026-08-17 01:05 +08:00) Archived and pushed the four formal P0-B histories, stdout/stderr summaries, TensorBoard events, compact statistics, checkpoint size/SHA-256 bindings, and the expanded Git-safe P0-A per-case/per-attempt evidence package. No checkpoint, STEP, pickle, or raw CAD bytes were included.
- [x] (2026-08-17 01:01 +08:00) Reconstructed and assembled the frozen 100 CADs with both the learned-VQ seed-3 best checkpoint (epoch 99) and continuous-bypass seed-3 best checkpoint (epoch 98). Each arm retained all 100 attempts and used the same ordered cohort, assembly chain, and OCC audit.
- [x] (2026-08-17 01:05 +08:00) Published the strict comparison `GT 84% | bypass@300k 70% | FSQ@300k 49% | bypass@60k 70% | VQ@60k 57%`. The measured gates are `Delta_q=13 pp` and `Delta_r=14 pp`; the predeclared decision is `CAPACITY_AB_FIRST`.
- [x] (2026-08-17 01:06 +08:00) Evaluated both P0 gates. P0-A attribution is 16/16 (100%), P0-B has zero non-finite events across all 400 epochs, and paired VQ/bypass assembly evidence covers 100 attempts per arm. The measured `Delta_q=13 pp` takes precedence, so boundary consistency is intentionally not started; capacity A/B is the next experiment.
- [x] (2026-08-14 12:18 +08:00) Generated `reports/p0b_runtime_evidence_20260814` with compact metrics, inventory digests, stdout/stderr, small TensorBoard events, resume/writer-lock evidence, and SHA-256 manifests for 15 local checkpoints without checkpoint bytes. Added a fail-closed reusable snapshot tool and three tests. Final pre-formal regression is 137 plus 91 tests, for 228 passing tests.
- [x] (2026-08-17 01:05 +08:00) Archived and pushed the formal histories, logs, TensorBoard events, paired 100-CAD reports, P0-A per-case/per-attempt evidence, and SHA-256 manifests. No checkpoint, STEP, raw CAD, pickle, reconstructed array, or upstream `BrepARG/` bytes are tracked.

## Surprises & Discoveries

- Observation: The original-control failures are already split into two fundamentally different populations.
  Evidence: Six of 100 attempts never wrote STEP; ten wrote STEP but the calibration-time strict check returned false. A final `brep_valid` boolean cannot diagnose both populations.

- Observation: Validity changes when the saved STEP is independently re-read.
  Evidence: The calibration runner reported original strict validity 84/100. The later dual-validity audit reported 86/100 strict, 85/100 OpenCascade-native, and 81/100 satisfying both. P0-A must report construction, export, re-import, native, strict, and both-valid stages separately.

- Observation: Joint optimization is not a likely global repair, but it still needs per-case attribution.
  Evidence: The prior 0-versus-200 iteration paired cohort tied at 81/100 both-valid. The new diagnostic limits this ablation to the 16 frozen failures and records which individual cases change.

- Observation: The current strict checker repairs wires before analyzing them and uses a fixed precision of 0.01.
  Evidence: `D:/luolin/V13/BrepARG/utils.py::check_brep_validity` calls `ShapeFix_Wire` and `ShapeAnalysis_Wire` at 0.01, while construction sews at `1e-3`. P0-A treats the three-level scan as a construction sewing-tolerance scan and keeps the strict checker fixed so the dependent variable does not move with the treatment.

- Observation: The formal VQ-VAE loop has the correct AMP unscale order, but the result is neither checked nor logged.
  Evidence: `_train_vqvae` executes scale, backward, unscale, clip, and step, but discards the pre-clip norm returned by `clip_grad_norm_` and does not request an error on non-finite gradients.

- Observation: The existing non-finite stop is ineffective for the fixed-100-epoch V6 protocol.
  Evidence: `update_vqvae_stop_state` gates the non-finite stop behind `reached_min_epochs`; V6 set `min_epochs=100`, so histories with nearly every batch skipped still reached epoch 99.

- Observation: A learned-VQ forward can mutate its embedding/usage buffers before the scalar loss is checked.
  Evidence: seed 0's final learned-VQ embedding contains 253,440 non-finite values. Skipping the optimizer step after a non-finite loss does not undo quantizer-side in-place state updates.

- Observation: The interrupted seed-3 learned-VQ checkpoint cannot support exact continuation.
  Evidence: It stores `model_state_dict`, validation values, and metadata but no optimizer, gradient scaler, LR scheduler, Python/NumPy/Torch RNG state, or prior history.

- Observation: P0-A exceeded its attribution gate but did not identify a global tolerance or joint-optimization repair.
  Evidence: The full 96-attempt matrix attributed 16/16 cases. Four cases changed full outcome signature when joint optimization was disabled and one changed signature across sewing tolerances, but only one case reached both-valid under any treatment, in its three `joint=0` variants. Signature sensitivity is diagnostic evidence, not repair success.

- Observation: The boundary loss cannot be added to the current deduplicated patch loader because that loader intentionally discards topology and placement.
  Evidence: `breparg_improvements/vqvae_sampling.py::patch_records_from_parsed` retains patch kind, array, source, and parent, then `records_to_chw_array` keeps only tensors. Shared-edge incidence, entity indices, and world-coordinate bounding boxes are absent, and content deduplication can merge geometrically identical patches from different CAD placements.

- Observation: Raw edge-to-surface Chamfer has a non-zero ground-truth floor at the stored 32-by-32 sampling resolution.
  Evidence: On the frozen 100 CADs, 6,024 shared edges had mean ground-truth nearest-surface squared distance about `3.503e-4`, with a long tail up to about `0.221`. Optimizing raw distance would move otherwise correct geometry toward the coarse UV grid.

- Observation: An exact resume needs a stable total-epoch interval in both execution and evidence, not merely a restored optimizer.
  Evidence: The first integrated implementation correctly resumed the loop at the next epoch but wrote `target_epoch = resumed_start + requested_epochs` to `history.json`, overstating the formal target after interruption. The history contract now records the original requested start and fixed target, and a regression test proves a checkpoint at epoch 0 resumes through epochs 1 and 2 while retaining `start_epoch=0` and `target_epoch=3`.

- Observation: Learned-VQ validation does not mutate the codebook or FeaturePool when the model is in evaluation mode.
  Evidence: The upstream quantizer wraps its EMA, random-anchor FeaturePool query, codebook restart, and contrastive term in `if self.training`; `model.eval()` recursively disables that block while preserving index and reconstruction measurement.

- Observation: The first strict P0-B implementation performed two complete state scans around every training batch, and each tensor check called `.item()` separately.
  Evidence: With 469 formal train batches, this implied about 938 full model/optimizer/FeaturePool traversals per epoch before validation. The `lifecycle_v1` regression executes eight train batches across two epochs but observes exactly three complete audits: startup plus one epoch-end pre-save audit per epoch. On CUDA, AdamW may keep scalar `step` state on CPU while parameters live on GPU, so the finite path performs one aggregated scalar read for each of those two device groups rather than one read per tensor.

- Observation: Two layers independently forced a full source scan before the first GPU batch.
  Evidence: The original fp32 probe inherited parent balancing, deduplication before cap, exact-cap enforcement, and 90-percent split-parent coverage. After those launcher flags were disabled, the first replacement probe still remained CPU-bound because `_collect_protocol_inventory` passed `require_all_paths=True` unconditionally. The obsolete local directories `p0b_precision_probe_fp32_20260814` and `p0b_precision_probe_fp32_v2_20260814` contain no valid precision result and must not be reused.

- Observation: Equal patch counts and equal sampling summaries do not cryptographically prove that independently launched tasks trained on identical tensors in identical order.
  Evidence: Protocol V5 has enough data for the requested caps, but prior summaries retained only counts and coverage. The new `vq-exact-hash-inventory-v1` stores a SHA-256 digest over the ordered exact patch hashes and another over the sorted hashes, without storing patch bytes.

- Observation: Exact optimizer restoration is not meaningful if the numerical runtime changes underneath it.
  Evidence: The rolling signature now binds stable Python, NumPy, PyTorch, Diffusers, CUDA, cuDNN, compute-capability, dtype, TF32, and deterministic-mode fields. Volatile timestamps, paths, process IDs, hostnames, and GPU display names remain audit-only metadata so they do not create false resume failures.

- Observation: A bounded probe can legitimately materialize fewer unique tensors than its requested cap even after raw selection reaches the cap.
  Evidence: The fp32 `v3` probe selected 128 validation records, then final exact deduplication reduced the tensor inventory to 90; train exact-overlap filtering left 116. The inventory digests correctly bound those realized tensors, but the initial smoke validator compared them to 128 and rejected an otherwise finite run. Formal mode is unaffected because deduplication occurs before selection and exact-cap enforcement is mandatory.

- Observation: The same-scale bypass control is materially stronger than learned VQ on assembly validity, even though both use the same 60k cohort and the same unmodified assembly chain.
  Evidence: The paired 100-CAD audit produced bypass@60k = 70 strict / 73 native / 64 both-valid and VQ@60k = 57 strict / 55 native / 49 both-valid, with 95 STEP-readable attempts for each arm. The strict gap is `Delta_q=13 pp`, exceeding the registered five-point noise band.

- Observation: Reconstruction loss remains a separate assembly bottleneck after removing the quantizer.
  Evidence: The 60k bypass strict rate is 70/100 against the historical GT reference of 84/100, giving `Delta_r=14 pp`. Both gates numerically fire, but the pre-registered precedence rule selects capacity A/B before boundary consistency so the two effects are not changed simultaneously.

## Decision Log

- Decision: Freeze P0-A to exactly the 16 original-control invalid CAD identities from the completed 100-CAD calibration.
  Rationale: Re-selection would confound diagnosis with cohort variation and break the existing 84/70/49 comparison.
  Date/Author: 2026-08-13 / Codex.

- Decision: Scan only the BRep sewing tolerance at `1e-4`, `1e-3`, and `1e-2`; keep surface fitting, curve fallback tolerances, and strict-validation precision unchanged but log every curve fallback attempt.
  Rationale: A tolerance experiment must vary one construction parameter. Changing fitting, sewing, and validity tolerances together would make a recovered CAD impossible to attribute.
  Date/Author: 2026-08-13 / Codex.

- Decision: Define a clear P0-A attribution as a named pre-STEP failure stage, a named failed strict component after STEP, a reproducible joint-optimization sensitivity, or a reproducible sewing-tolerance sensitivity. `unknown` and generic `assembly_error` do not count.
  Rationale: This definition is machine-checkable and maps directly to a repair action.
  Date/Author: 2026-08-13 / Codex.

- Decision: Report both OpenCascade-native validity and project strict validity, and use their conjunction as the strongest assembly outcome while preserving the historical strict-only number for comparison.
  Rationale: The two definitions disagree on the frozen cohort. Discarding either would hide a real measurement ambiguity.
  Date/Author: 2026-08-13 / Codex.

- Decision: Use explicit precision names `fp32`, `fp16`, and `bf16`; enable `GradScaler` only for fp16, unscale before clipping, and require the returned pre-clip norm to be finite.
  Rationale: A boolean AMP flag cannot distinguish fp16 overflow from bf16 or full-fp32 behavior. The existing unscale ordering is correct, but it does not prove the gradients are finite or that clipping was active.
  Date/Author: 2026-08-13 / Codex.

- Decision: Make any non-finite loss, gradient norm, validation sample, or parameter update a formal P0-B failure rather than silently skipping it.
  Rationale: The only P0-B acceptance condition is zero non-finite events. Continuing after one event can create a nominal 100-epoch result that should never be promoted.
  Date/Author: 2026-08-13 / Codex.

- Decision: Save an atomic rolling checkpoint after every completed epoch with model, optimizer, scaler, scheduler, stop state, history, and all RNG states. Resume only when the immutable experiment signature matches.
  Rationale: This is the minimum state required to survive another Windows restart without silently changing the experiment.
  Date/Author: 2026-08-13 / Codex.

- Decision: Keep strict non-finite mode and automatic full-state resume disabled by default in historical entry points, and enable both explicitly in the P0-B launcher.
  Rationale: Existing experiment scripts keep their prior behavior, while every formal P0-B task binds the stricter behavior into its signed environment and rejects configuration drift.
  Date/Author: 2026-08-14 / Codex.

- Decision: Validate all four P0-B histories, then run the fixed seed-3 best checkpoint from both learned VQ and continuous bypass through the same frozen 100-CAD assembly measurement.
  Rationale: The historical bypass 70-percent result came from 300k-patch training. Comparing it directly with VQ@60k confounds quantization with data scale. The paired 60k measurement isolates the assembly-level quantization tax while preserving historical GT, bypass@300k, and FSQ@300k as labeled reference columns.
  Date/Author: 2026-08-17 / Codex.

- Decision: Apply the post-measurement gates in percentage points: start boundary consistency when `Delta_r > 8`; run a VQ-8192 versus residual-VQ capacity comparison first when `Delta_q > 5`; when both quantities are within the five-point sampling-noise band, release the representation layer and prioritize P0-A assembly repairs until GT strict validity is at least 95 percent.
  Rationale: These rules separate reconstruction-induced assembly loss from quantization-induced loss and prevent a new boundary loss from masking an unresolved quantizer-capacity problem.
  Date/Author: 2026-08-17 / Codex.

- Decision: Use learned VQ and continuous bypass only for the 60k P0-B stability retest at seeds 3 and 4, with a fixed 100-epoch budget and the same validation cohort.
  Rationale: Both FSQ arms already failed in all eight completed attempts. P0-B tests the viable quantized arm and its continuous lower bound without spending time reproducing a known FSQ failure.
  Date/Author: 2026-08-13 / Codex.

- Decision: Keep boundary-consistency implementation gated until both P0 reports are complete.
  Rationale: Otherwise a new loss would be evaluated through an unclassified assembly floor and an unstable optimizer, making its result uninterpretable.
  Date/Author: 2026-08-13 / Codex.

- Decision: Use full-state audit cadence `lifecycle_v1`: audit once at startup or immediately after checkpoint restoration, then once after validation and scheduler update at every epoch before any checkpoint write; do not scan the complete state inside each training batch.
  Rationale: Scalar loss finiteness and the reduced pre-clip gradient norm still fail immediately per batch. Complete model, optimizer, and FeaturePool traversal is needed for lifecycle integrity but its former per-batch placement forced hundreds of GPU synchronizations per epoch and depressed utilization. The epoch audit still catches quantizer-side buffer corruption before any checkpoint can be promoted or saved.
  Date/Author: 2026-08-14 / Codex.

- Decision: Use two explicit sampling contracts. Formal P0-B keeps parent balancing, deduplication before cap, exact 60,000/12,000 caps, and at least 90-percent parent coverage; precision probes disable those cohort-wide gates and remain bounded health checks.
  Rationale: A precision probe answers only whether a small forward/backward/checkpoint path is finite. Making it scan the full formal cohort delays GPU work without strengthening that answer, while weakening the formal gates would invalidate the actual comparison.
  Date/Author: 2026-08-14 / Codex.

- Decision: Derive full-source scanning from the formal cohort requirements rather than hard-code it in the shared inventory helper.
  Rationale: Parent balancing, pre-cap deduplication, exact-cap validation, or a positive parent-coverage gate each requires the full split and therefore implies fail-closed source loading. When all four are absent, as in a numerical probe, continuing to load thousands of unused CADs adds no evidence. The derivation preserves historical Protocol V2 defaults because balancing and deduplication are enabled there.
  Date/Author: 2026-08-14 / Codex.

- Decision: Bind every formal task to the frozen Protocol V5 hashes and to identical ordered and sorted train/validation inventory digests.
  Rationale: The four tasks are a controlled arm-and-seed comparison. Exact counts alone allow silent input drift between processes; ordered digests also prevent a changed tensor order from masquerading as an exact resume.
  Date/Author: 2026-08-14 / Codex.

- Decision: Permit only one writer per P0-B output root and make inspection commands read-only.
  Rationale: Concurrent launchers could race on state, logs, and rolling checkpoints. An operating-system lock fails the second writer immediately and is released by the kernel after a crash, while stale metadata remains useful recovery evidence.
  Date/Author: 2026-08-14 / Codex.

- Decision: In smoke mode, validate the positive realized inventory and its cross-artifact digests; do not require requested cap equality. In formal mode, continue requiring requested and realized counts to be exactly 60,000/12,000.
  Rationale: A precision probe tests numerical execution, not cohort completeness, and post-selection deduplication may reduce its tiny sample. The formal arm comparison depends on exact cohort equality and therefore keeps the strict count gate.
  Date/Author: 2026-08-14 / Codex.

- Decision: Treat P0-A as diagnostically complete but do not globally disable joint optimization or relax sewing tolerance.
  Rationale: Attribution reached 100 percent, yet only one of 16 cases recovered under any treatment. Four joint-sensitive and one tolerance-sensitive signatures mostly represent changes between invalid failure modes, not validity recovery.
  Date/Author: 2026-08-13 / Codex.

- Decision: If the P0 gates open, construct a separate topology-preserving boundary-triplet loader from per-CAD parsed pickles instead of extending the deduplicated patch cache.
  Rationale: Each training item must bind one shared edge, its two incident faces, their NCS patches, WCS bounding boxes, and entity indices. The current cache has deliberately erased those one-to-one relationships.
  Date/Author: 2026-08-13 / Codex.

- Decision: The first boundary penalty will compare predicted edge-to-surface distance against a stopped-gradient ground-truth discretization baseline, and weight zero will short-circuit before the relation loader or auxiliary quantizer forward.
  Rationale: The coarse ground-truth surface grid already has non-zero nearest-neighbor distance. A baseline-excess hinge avoids rewarding geometric distortion, while an actual short circuit preserves bitwise control behavior and prevents learned-VQ codebook side effects.
  Date/Author: 2026-08-13 / Codex.

## Outcomes & Retrospective

P0-A is complete. The frozen 16-case, 96-attempt diagnosis reached 100-percent concrete attribution, exceeding the 80-percent gate. Its dominant cause was wire self-intersection in ten cases, followed by three curve fits, two wire builds, and one non-unit solid. Only one case recovered under any joint/tolerance treatment, so the outcome supports targeted assembly repair rather than a global tolerance change. The normalized Git snapshot is in `reports/p0a_assembly_chain_diagnosis_20260813`; generated STEP bytes remain local and 66 saved STEP files are bound by size and SHA-256.

P0-B training and the paired assembly gate are complete. The four bf16 60k/12k tasks reached 100 epochs with zero non-finite events and exact common inventories. Learned VQ retained 3,988 and 4,032 of 4,096 validation codes at seeds 3 and 4, while its lowest parent-cluster curved MSE remained about 2.45 times the continuous-bypass mean. On the fixed 100-CAD cohort, bypass@60k reached 70/100 strict and VQ@60k 57/100 strict; `Delta_q=13 pp` exceeds the five-point capacity trigger and `Delta_r=14 pp` exceeds the boundary-loss trigger. Capacity A/B therefore has precedence; boundary consistency remains blocked until that capacity comparison is completed.

## Context and Orientation

The Git-managed working tree is `D:/luolin/V13/v6git` on local branch `protocol-v5-runtime`, pushed to remote branch `experiment/protocol-v5-scaling-ladder`. Heavy experiment output belongs under `D:/luolin/V13/local_runs` and must not be committed. The verified Protocol V5 split is `D:/luolin/V13/local_runs/protocol_v5_scaling_run_20260806/protocol`. Its internal protocol SHA-256 is `6b588ee0a9dc337a683d9cc94cde7d79a80963720d22098d99e7f6eaa8101cf3`, its `split.pkl` SHA-256 is `6ff0a0c3ee6a04ee056fa1ab982eb436a9f59d3d21f21f17babf34e6dc701d29`, and the `protocol_summary.json` file SHA-256 is `cf0dfaf71b544851e0531bc33332753d4f06202b27499a5a487946dd9e9eed59`. Formal launch refuses any mismatch, any non-`VERIFIED` status, or any nonzero parent overlap. Parsed validation CADs referenced by that split are materialized under the same Protocol V5 run.

`tools/run_assembly_calibration_oracle.py` performs normalized-patch reconstruction, `cpu_joint_optimize`, upstream BrepARG construction, STEP writing, and strict validation. `tools/audit_assembly_step_validity.py` independently re-reads STEP and reports both `BRepCheck_Analyzer` native validity and upstream project-strict validity. `tools/diagnose_step_validity_components.py` counts wire order errors, wire self-intersections, bad shell edges, free edges, shell count, and solid count. P0-A combines these responsibilities in one stage-aware attempt record rather than changing upstream `D:/luolin/V13/BrepARG/utils.py`.

The phrase “joint optimization” means translating each reconstructed surface in world coordinates to reduce nearest-neighbor distance between its sampled points and all incident edge points. The original calibration used 200 optimization iterations. The zero-iteration arm keeps the same initial normalized-to-world placement but does not optimize surface offsets.

The phrase “sewing tolerance” means the positional tolerance passed to OpenCascade `BRepBuilderAPI_Sewing` when faces are combined into a shell. It is not the validation tolerance. P0-A scans `1e-4`, the historical `1e-3`, and `1e-2` while holding all other fitting and validity settings fixed.

`breparg_improvements/train.py --stage vqsweep` trains representation arms. The learned-VQ arm imports the local upstream `VectorQuantiser` without tracking the upstream source. Continuous bypass uses the same encoder and decoder but does not discretize the latent. P0-B extends this training loop and adds `tools/run_p0b_stability_retest.py`, which launches one arm and seed at a time into a new output root. “Automatic resume” means a restarted launcher validates and reloads the matching rolling checkpoint; it never overwrites the interrupted Protocol V6 directory.

## Plan of Work

First add `tools/diagnose_assembly_chain.py` and `tests/test_diagnose_assembly_chain.py`. The runner reads the frozen calibration manifest, selects only original rows with `brep_valid != true`, verifies there are exactly 16 unique CADs, and binds the source manifest hash. For every CAD it runs six variants: joint iterations 200 and 0 crossed with sewing tolerance `1e-4`, `1e-3`, and `1e-2`. It implements the same surface fit, curve fit, topology ordering, wire/face trim, sewing, solid conversion, STEP write, and re-read operations as the original pipeline but wraps every stage and entity index. It writes one JSONL row immediately after each variant so one OCC exception cannot abort the cohort.

The P0-A summary groups variants per CAD, names the baseline failure at joint 200 and tolerance `1e-3`, records whether joint removal or tolerance changes any validity stage, and assigns a primary cause. Saved STEP rows include native and strict validity plus wire, shell, free-edge, and solid component counts. The summary reports attributed cases divided by 16 and fails its acceptance gate below 0.8. A Markdown repair checklist maps observed cause families to bounded code changes; it is evidence, not an automatic broad repair.

Next extend `breparg_improvements/training_stability.py` with explicit precision parsing, serializable stop/training state helpers, RNG capture/restore, finite-gradient inspection, device-aggregated full-state auditing, and experiment-signature validation. Extend `_train_vqvae` in `breparg_improvements/train.py` to use an autocast context selected by precision, unscale fp16 gradients before clipping, record pre-clip norm and effective LR, use `ReduceLROnPlateau` on the configured validation metric, trip an immediate strict fuse on any non-finite loss or gradient norm, and run a complete `lifecycle_v1` state audit at startup or post-resume and after each epoch before atomically saving rolling state. Existing best checkpoints remain loadable for evaluation; rolling P0-B checkpoints add fields rather than changing `model_state_dict`.

Add `tools/run_p0b_stability_retest.py` and focused tests. The launcher accepts only arms `vq_4096_64d_random` and `continuous_bypass_64d`, seeds 3 and 4, 60,000 train patches, 12,000 validation patches, batch 128, 100 epochs, and the frozen Protocol V5 split. Formal mode verifies the fixed protocol hashes, exact caps, zero parent overlap, and a `vq-exact-hash-inventory-v1` binding for both splits. All four tasks must share the same ordered and sorted inventory digests. A single output-root writer lock protects state, logs, and checkpoints; `status` and `validate` stay read-only. On restart the launcher preserves logs and resumes only after matching the model, optimizer, scheduler, scaler, random-number generators, FeaturePool, selector, inventory, source hashes, and stable numerical runtime. It refuses promotion unless every expected train and validation batch is finite, every gradient norm and lifecycle state audit is finite, all checkpoint schemas deserialize correctly, and exactly 100 epochs exist.

Before the formal retest, run bounded fp32 and bf16 forward/backward probes on both P0-B arms in new output roots. These probes deliberately disable full-cohort parent balancing, deduplication before cap, exact-cap enforcement, and parent-coverage gating, so 128 requested patches lead promptly to GPU work. Select bf16 only if CUDA reports support and both arms match fp32 finiteness; otherwise select fp32. The probe is not used for quality comparison. Then run a real CUDA smoke that saves a rolling checkpoint, restarts with the identical signature at the next epoch, leaves no duplicate history epoch, and proves that a concurrent second writer is rejected. Query Windows active hours and retain the already recorded 18-hour interval covering the expected run. Do not disable the Windows Update service.

After all four P0-B trainings pass, bind exactly the learned-VQ seed-3 best checkpoint and continuous-bypass seed-3 best checkpoint by size and SHA-256. Use the deterministic selection seed `20260809` and exactly the same 100 CAD identities as the existing assembly calibration. For each arm, first run surface reconstruction, then run the unchanged historical assembly pipeline at 200 joint iterations and sewing tolerance `1e-3`, retaining failures in the denominator. Report STEP readability, strict validity, native validity, and both-valid separately. Publish the strict five-column comparison `GT 84% | bypass@300k 70% | FSQ@300k 49% | bypass@60k | VQ@60k`, explicitly labeling the first three values as historical strict-only runner results. Compute `Delta_q = bypass@60k strict - VQ@60k strict` and `Delta_r = 84 - bypass@60k strict`, both in percentage points. Apply the fixed gates from the Decision Log without reinterpretation after seeing the results.

Finally evaluate the hard gate. If P0-A attribution coverage is at least 0.8, P0-B has zero non-finite events, and learned-VQ assembly evidence covers all 100 CADs, build a separate relation dataset from parsed per-CAD records. Each valid record binds one manifold edge to its two incident faces, their normalized patches, WCS bounding boxes, and entity indices; it verifies `edgeFace_adj` and `faceEdge_adj` bidirectionally and never uses cross-CAD content deduplication. Map decoded NCS points to WCS with `center=(bbox_min+bbox_max)/2`, `scale=max(bbox_max-bbox_min)`, and `x_wcs=x_ncs*scale/2+center`. The initial loss uses edge-to-surface `torch.cdist` only and penalizes the positive excess above the stopped-gradient ground-truth nearest-neighbor squared-distance baseline. The three weights 0, 0.1, and 1.0 share the same data, seeds, training budget, and assembly evaluator. Weight zero returns before creating or iterating the relation loader. If any P0 condition fails, do not implement or launch this loss; update this plan with the blocking evidence and repair the failed P0 first.

## Concrete Steps

Run all Git-managed commands from `D:/luolin/V13/v6git` with `C:/Users/YU/.conda/envs/brepgen_env/python.exe`.

The P0-A focused tests and real run have this shape:

    python -m pytest tests/test_diagnose_assembly_chain.py tests/test_assembly_calibration_oracle.py tests/test_diagnose_step_validity_components.py -q

    python tools/diagnose_assembly_chain.py --calibration-manifest reports/assembly_calibration_100cad_20260809/calibration_manifest.jsonl --breparg-root D:/luolin/V13/BrepARG --output-dir D:/luolin/V13/local_runs/p0a_assembly_chain_diagnosis_20260813 --joint-iterations 200,0 --sewing-tolerances 1e-4,1e-3,1e-2 --expected-invalid-cads 16

Expected terminal evidence includes `cases=16`, `attempts=96`, `attribution_rate>=0.8`, and a zero process exit status only when the gate passes.

The P0-B tests and replacement precision probes use:

    python -m pytest tests/test_run_p0b_vq_assembly_measurement.py tests/test_run_p0b_stability_retest.py tests/test_p0b_train_integration.py tests/test_training_stability.py tests/test_vqvae_protocol_training.py -q

    python tools/run_p0b_stability_retest.py probe --repo-root D:/luolin/V13/v6git --protocol-dir D:/luolin/V13/local_runs/protocol_v5_scaling_run_20260806/protocol --breparg-root D:/luolin/V13/BrepARG --output-root D:/luolin/V13/local_runs/p0b_precision_probe_fp32_v3_20260814 --python C:/Users/YU/.conda/envs/brepgen_env/python.exe --precision fp32

    python tools/run_p0b_stability_retest.py probe --repo-root D:/luolin/V13/v6git --protocol-dir D:/luolin/V13/local_runs/protocol_v5_scaling_run_20260806/protocol --breparg-root D:/luolin/V13/BrepARG --output-root D:/luolin/V13/local_runs/p0b_precision_probe_bf16_v3_20260814 --python C:/Users/YU/.conda/envs/brepgen_env/python.exe --precision bf16

The obsolete `D:/luolin/V13/local_runs/p0b_precision_probe_fp32_20260814` and `D:/luolin/V13/local_runs/p0b_precision_probe_fp32_v2_20260814` directories are retained as failure evidence but never resumed. The valid `v3` fp32 and bf16 roots and the bf16 resume-smoke root are summarized in `reports/p0b_runtime_evidence_20260814`. The formal launcher command uses `D:/luolin/V13/local_runs/p0b_stability_vq_bypass_60k_20260814` with `--precision bf16`. It must encode arms, seeds, caps, epochs, batch size, learning rate, gradient clipping, scheduler settings, sampling contract, inventory, runtime compatibility, and experiment signature in its state file. A restart repeats the same command and resumes only the active arm from its full-state rolling checkpoint.

After training, the paired measurement coordinator invokes the frozen 100-CAD surface and assembly evaluators. Expected summary evidence is four 100-epoch rows with `nonfinite_events=0`, two fixed seed-3 best-checkpoint hashes, identical ordered 100-CAD identity digests for both arms, 200 reconstruction rows, 200 assembly attempts, per-arm STEP-readable/strict/native/both-valid counts, the five-column strict comparison, and the two gate deltas.

## Validation and Acceptance

P0-A unit tests must prove frozen-case selection, exact six-variant construction, stage-local exception capture, strict-component decomposition, sensitivity classification, and the 80-percent attribution gate. A real one-CAD smoke must produce six rows without modifying the source pickle or old calibration. Full acceptance requires 96 rows for 16 CADs, at least 13 clearly attributed CADs, and a repair checklist whose item counts reconcile with case classifications.

P0-B unit tests must prove fp32/fp16/bf16 context selection, fp16 unscale-before-clip ordering, bf16 scaler disablement, finite-gradient fuse behavior, scheduler state persistence, atomic full-state checkpoints, exact resume epoch, RNG restoration, signature mismatch rejection, zero-nonfinite validation, writer exclusion, immutable Protocol V5 hashes, exact caps, bounded probe sampling, inventory consistency, runtime compatibility, evidence snapshot validation, and `lifecycle_v1` audit cadence. The cadence regression must show that complete state audit calls scale with epochs rather than batches, while a deliberately corrupted FeaturePool still fails before checkpoint creation. The final pre-formal baseline is 228 passing tests: 137 P0-B/inventory/measurement/snapshot tests plus 91 assembly/V5/V6/sampling tests. The completed CUDA smoke executed forward, backward, clipping, optimizer step, checkpoint save, process interruption, and next-epoch resume.

Formal P0-B acceptance is deliberately narrow: all four arm/seed histories reach epoch 99 with every expected train and validation batch finite and no finite-gradient or finite-parameter violation. Quality metrics are recorded but do not alter this stability gate. Paired assembly acceptance requires learned VQ and bypass to use identical ordered 100-CAD identities and 100-attempt denominators, with all reconstruction or assembly failures retained rather than excluded. The report must expose STEP-readable, strict, native, and both-valid counts independently and must reproduce `Delta_q` and `Delta_r` directly from the attempts-based strict counts.

The boundary-consistency stage is accepted only by a strict-valid assembly-rate improvement on the frozen 100-CAD cohort relative to weight 0. Reconstruction MSE may be reported but cannot promote the loss. If weight 0.1 or 1.0 lowers MSE without improving assembly validity, the innovation hypothesis is not supported.

## Idempotence and Recovery

P0-A never overwrites the old calibration. Each attempt key includes source-manifest SHA-256, CAD identity, joint iteration count, sewing tolerance, and runner version. Re-running skips an exact completed row and regenerates summaries from all valid rows. STEP paths live below the new P0-A output root.

P0-B never writes into `protocol_v6_5seed_100epoch_20260810`. Each arm/seed has a separate directory and immutable experiment signature. The output root has a kernel-backed single-writer lock; a second writer fails immediately, while `status` and `validate` do not acquire the lock or change files. Rolling checkpoint writes use a temporary sibling followed by `os.replace`. A stale or partial temporary file is ignored. A signature, inventory, protocol, source, or runtime mismatch fails closed and requires a new output directory. Completed healthy arms are skipped; failed arms retain their logs and state. The launcher never terminates another process.

Changing Windows active hours is bounded and reversible. The runner records the original values before changing them and writes a restoration command. It does not stop services, delete scheduled tasks, or disable updates. If elevation is unavailable, training can proceed only after the state records that active hours were not changed and automatic resume has passed its smoke test.

## Artifacts and Notes

Git tracks this plan, source, tests, concise JSON/JSONL/CSV/Markdown summaries, text logs, small TensorBoard events, and SHA-256 manifests. Git excludes all `.pt`, `.pth`, `.ckpt`, `.npz`, raw pickle, materialized CAD, generated STEP, PID, and upstream `BrepARG/` bytes. STEP files remain local; their relative paths, sizes, and SHA-256 values may be archived.

The CUDA preflight snapshot is `reports/p0b_runtime_evidence_20260814`. Its `runtime_evidence.json` records both precision probes and the resume smoke; `checkpoint_manifest.json` binds 15 local best/final/rolling files by size and SHA-256 without archiving their bytes; `logs/` and `tensorboard/` contain only lightweight probe evidence. `tools/snapshot_p0b_runtime_evidence.py` refuses incomplete launcher validation, non-finite probe rows, mismatched fp32/bf16 inventories, an invalid `[0, 1]` resume sequence, missing writer rejection, or absent stale-lock recovery.

The historical comparison numbers 84% original, 70% continuous bypass, and 49% FSQ are strict-valid rates from the original calibration runner. P0-A additionally reports independent re-read native, strict, and both-valid values; it must not silently replace the historical denominator or label.

## Interfaces and Dependencies

`tools.diagnose_assembly_chain.select_frozen_failures(manifest_path, expected_count)` returns the 16 unique original-control invalid source records. `run_attempt(parsed, case, joint_iterations, sewing_tolerance, output_dir, breparg_root)` returns one fully serializable row and catches stage-local exceptions. `classify_case(rows)` returns a primary cause, secondary evidence, sensitivity flags, and `attributed: bool`. `summarize_cases(cases, attempts)` returns the acceptance gate and repair counts.

`breparg_improvements.training_stability.PrecisionPolicy` and `resolve_precision()` validate `fp32`, `fp16`, or `bf16` and provide autocast dtype and scaler policy. `capture_rng_state()` and `restore_rng_state(state)` cover Python, NumPy, Torch CPU, and all CUDA RNGs. `atomic_torch_save(payload, path)` writes a recoverable rolling checkpoint. `build_experiment_signature()`, `load_training_checkpoint()`, and `restore_training_checkpoint()` reject incomplete or mismatched resume state.

`breparg_improvements.train._train_vqvae` gains explicit precision, gradient-clip, scheduler, strict-fuse, rolling-checkpoint, resume, and experiment-signature parameters while preserving existing callers through defaults. Its history rows record LR, finite batch counts, non-finite event counts, pre-clip gradient norm, whether clipping was applied, resume provenance, the `lifecycle_v1` cadence, per-epoch full-state audit coverage, and the explicit fact that no complete state audit runs inside a train batch. `breparg_improvements.training_stability.audit_finite_training_state(model, optimizer)` returns a JSON-serializable coverage summary and reads only one aggregate finite scalar per device on the normal path; if a device fails, it falls back to a detailed tensor scan to name the corrupt model, optimizer, or FeaturePool value.

`breparg_improvements.vqvae_sampling.summarize_exact_hash_inventory(records)` returns schema `vq-exact-hash-inventory-v1`, count, ordered SHA-256, and sorted SHA-256. The digests cover canonical exact patch hashes and bind both input membership and training order without putting patch bytes into Git.

`tools.run_p0b_stability_retest` exposes `probe`, `run`, `status`, and `validate` commands. The run state names the four arm/seed tasks, their exact signatures, rolling checkpoint paths, log paths, inventory bindings, and effective status. Probe mode is a bounded numerical-health path; formal run mode is hard-bound to Protocol V5 and exact 60,000/12,000 inventories. The measurement coordinator validates all four tasks and inventory equality, binds the fixed learned-VQ seed-3 best and continuous-bypass seed-3 best checkpoints, invokes the existing reconstruction and assembly entry points for each arm with one frozen cohort, and refuses any report whose arm identity sequence or denominator differs.

If the gate opens, the boundary dataset yields topology-preserving shared-edge triplets rather than extending the current patch cache. The loss helper accepts predicted and ground-truth surfaces and edge, their WCS bounding boxes, and a weight; it returns total loss plus separately logged reconstruction, VQ, raw boundary distance, ground-truth distance floor, and weighted excess. Weight zero must be bit-for-bit equivalent to the current loss path, including learned-VQ buffers and optimizer state.

Revision note 2026-08-13: Created after the user prioritized assembly-chain attribution, stable resumable VQ/bypass retesting, and a hard gate before boundary consistency. It records the frozen cohorts, exact acceptance gates, the validity-definition mismatch, and the non-destructive Git/runtime artifact policy.

Revision note 2026-08-13 22:20 +08:00: Updated after completing P0-A. It records the 16/16 attribution result, distinguishes sensitivity from recovery, marks P0-A complete, and incorporates the topology-preserving relation-loader and ground-truth-baseline boundary-loss decisions discovered during the gated design audit.

Revision note 2026-08-14 00:20 +08:00: Updated after completing P0-B implementation and compatibility regression. It records the immutable launcher and measurement coordinator, reversible Windows active-hours change, exact resume evidence contract, explicit compatibility defaults, current test evidence, and the remaining CUDA/formal-runtime gate.

Revision note 2026-08-14 11:20 +08:00: Updated after profiling the strict finite-state path before CUDA probes. It records the `lifecycle_v1` audit decision, removal of complete state scans from the per-batch hot path, device-aggregated normal-path checks, retained immediate loss/gradient fuses, checkpoint and history evidence, and the 95-test focused regression result.

Revision note 2026-08-14 11:50 +08:00: Updated after closing the P0-B launcher and measurement evidence contracts. It records the invalid first-probe root cause, separate probe and formal sampling contracts, writer exclusion, fixed Protocol V5 hashes, exact caps, ordered and sorted patch-inventory digests, stable runtime resume binding, four-task inventory equality, the new output roots, and the final 222-test implementation baseline.

Revision note 2026-08-14 12:00 +08:00: Updated after the first live replacement probe found that `_collect_protocol_inventory` still hard-coded a full source scan. It records the second invalid probe, the derived full-scan rule that preserves every formal gate, the cap-bounded regression, and the fresh `v3` probe roots.

Revision note 2026-08-14 12:08 +08:00: Updated after the fp32 `v3` probe reached GPU training and exposed a smoke-only requested-versus-realized inventory validation error. It records the finite learned-VQ evidence, the 116/90 realized counts, the smoke validator correction, the unchanged formal exact-cap gate, and the 225-test baseline.

Revision note 2026-08-14 12:18 +08:00: Updated after both precision probes and the real interruption smoke passed. It records bf16 selection, zero-nonfinite evidence for both arms and precisions, exact `[0, 1]` restoration, second-writer rejection, stale-lock recovery, the Git-safe runtime snapshot, and the remaining formal four-task gate.

Revision note 2026-08-17 00:15 +08:00: Updated after all four formal P0-B tasks completed. It replaces the single-VQ measurement with a paired fixed-seed-3 VQ/bypass 100-CAD measurement, removes the historical 300k data-scale confound, records the five-column comparison and Delta gates, and adds the formal P0-A/P0-B Git-safe archive milestone.

Revision note 2026-08-17 01:05 +08:00: Completed and archived the paired 60k assembly measurement. The VQ arm produced 95 STEP-readable, 55 native-valid, 57 strict-valid, and 49 both-valid attempts; bypass produced 95, 73, 70, and 64 respectively. The measured `Delta_q=13 pp` selects capacity A/B before boundary consistency.

Revision note 2026-08-17 01:06 +08:00: Closed the archive and gate milestones after independent JSON/CSV recomputation and 44 focused regression tests. The full task is complete through the paired diagnostic measurement; the next experiment is deliberately outside this plan: VQ-8192 versus RVQ capacity A/B.
