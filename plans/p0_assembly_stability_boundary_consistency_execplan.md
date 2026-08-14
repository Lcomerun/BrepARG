# Diagnose assembly, stabilize representation training, and gate boundary consistency

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current while work proceeds. This document follows `PLANS.md` in the repository root. `AGENTS.md` names `.agent/PLANS.md`, but that path is absent; the checked-in root `PLANS.md` is the available authority.

## Purpose / Big Picture

Recent experiments show two upstream failure sources before autoregressive generation can be evaluated. First, the same 100 validation CADs are not reliably assembled even from their original normalized patches: the original control was 84/100 valid in the calibration runner. Second, the fixed 100-epoch representation cohort repeatedly became non-finite and was then interrupted by a Windows update. This plan turns both observations into reproducible, bounded experiments.

After P0-A, a reader can inspect every original-control failure and see the exact OpenCascade stage, exception, strict-validity component, joint-optimization sensitivity, and sewing-tolerance sensitivity. At least 80 percent of the 16 original-control invalid CADs must receive a concrete primary attribution and the report must contain an actionable repair list. After P0-B, learned VQ and continuous bypass at seeds 3 and 4 train on the same 60,000 patches with no non-finite train or validation batch, complete checkpoints can resume after interruption, and one selected healthy learned-VQ checkpoint receives the same frozen 100-CAD reconstruction and assembly measurement used for the published 84/70/49 comparison.

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
- [ ] Run finite forward/backward precision probes and select the stable CUDA precision for the formal P0-B retest.
- [ ] Run learned VQ and continuous bypass at seeds 3 and 4 on the same 60,000/12,000 patch protocol for 100 epochs, requiring zero non-finite batches and resumable checkpoints.
- [ ] Reconstruct and assemble the frozen 100 CADs with the best healthy P0-B learned-VQ checkpoint and report attempts-based strict validity beside original 84%, bypass 70%, and FSQ 49%.
- [ ] Evaluate both P0 gates. Implement the boundary-consistency experiment only if P0-A attribution is at least 80 percent, P0-B has zero non-finite batches, and the VQ assembly measurement is complete.
- [ ] Archive lightweight code, tests, manifests, logs, histories, summaries, and hashes to GitHub while excluding checkpoints, raw data, reconstructed arrays, and upstream `BrepARG/` source.

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

- Decision: Validate all four P0-B histories before selecting the learned-VQ seed with the lowest finite curved parent-cluster MSE, then run only that learned-VQ checkpoint through the frozen 100-CAD assembly measurement.
  Rationale: Stability requires both VQ and bypass controls, but the missing measurement requested by the user is the learned-VQ strict/native/both-valid result. Re-running GT, bypass, and FSQ would waste compute and risk changing the historical 84/70/49 denominator.
  Date/Author: 2026-08-14 / Codex.

- Decision: Use learned VQ and continuous bypass only for the 60k P0-B stability retest at seeds 3 and 4, with a fixed 100-epoch budget and the same validation cohort.
  Rationale: Both FSQ arms already failed in all eight completed attempts. P0-B tests the viable quantized arm and its continuous lower bound without spending time reproducing a known FSQ failure.
  Date/Author: 2026-08-13 / Codex.

- Decision: Keep boundary-consistency implementation gated until both P0 reports are complete.
  Rationale: Otherwise a new loss would be evaluated through an unclassified assembly floor and an unstable optimizer, making its result uninterpretable.
  Date/Author: 2026-08-13 / Codex.

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

P0-B remains active, but its implementation phase is complete. The formal training loop, immutable four-task launcher, full-state resume contract, fail-closed validator, Windows training-window audit, and learned-VQ 100-CAD measurement coordinator are implemented and covered by focused and compatibility regressions. The remaining P0-B work is runtime evidence: CUDA precision probes, four 100-epoch 60k/12k runs, and the selected healthy learned-VQ fixed-cohort assembly measurement. Boundary consistency therefore remains blocked until those runtime gates pass.

## Context and Orientation

The Git-managed working tree is `D:/luolin/V13/v6git` on branch `experiment/protocol-v5-scaling-ladder`. Heavy experiment output belongs under `D:/luolin/V13/local_runs` and must not be committed. The verified Protocol V5 split is `D:/luolin/V13/local_runs/protocol_v5_scaling_run_20260806/protocol`; its protocol SHA-256 is `6b588ee0a9dc337a683d9cc94cde7d79a80963720d22098d99e7f6eaa8101cf3`. Parsed validation CADs referenced by that split are materialized under the same Protocol V5 run.

`tools/run_assembly_calibration_oracle.py` performs normalized-patch reconstruction, `cpu_joint_optimize`, upstream BrepARG construction, STEP writing, and strict validation. `tools/audit_assembly_step_validity.py` independently re-reads STEP and reports both `BRepCheck_Analyzer` native validity and upstream project-strict validity. `tools/diagnose_step_validity_components.py` counts wire order errors, wire self-intersections, bad shell edges, free edges, shell count, and solid count. P0-A combines these responsibilities in one stage-aware attempt record rather than changing upstream `D:/luolin/V13/BrepARG/utils.py`.

The phrase “joint optimization” means translating each reconstructed surface in world coordinates to reduce nearest-neighbor distance between its sampled points and all incident edge points. The original calibration used 200 optimization iterations. The zero-iteration arm keeps the same initial normalized-to-world placement but does not optimize surface offsets.

The phrase “sewing tolerance” means the positional tolerance passed to OpenCascade `BRepBuilderAPI_Sewing` when faces are combined into a shell. It is not the validation tolerance. P0-A scans `1e-4`, the historical `1e-3`, and `1e-2` while holding all other fitting and validity settings fixed.

`breparg_improvements/train.py --stage vqsweep` trains representation arms. The learned-VQ arm imports the local upstream `VectorQuantiser` without tracking the upstream source. Continuous bypass uses the same encoder and decoder but does not discretize the latent. P0-B extends this training loop and adds `tools/run_p0b_stability_retest.py`, which launches one arm and seed at a time into a new output root. “Automatic resume” means a restarted launcher validates and reloads the matching rolling checkpoint; it never overwrites the interrupted Protocol V6 directory.

## Plan of Work

First add `tools/diagnose_assembly_chain.py` and `tests/test_diagnose_assembly_chain.py`. The runner reads the frozen calibration manifest, selects only original rows with `brep_valid != true`, verifies there are exactly 16 unique CADs, and binds the source manifest hash. For every CAD it runs six variants: joint iterations 200 and 0 crossed with sewing tolerance `1e-4`, `1e-3`, and `1e-2`. It implements the same surface fit, curve fit, topology ordering, wire/face trim, sewing, solid conversion, STEP write, and re-read operations as the original pipeline but wraps every stage and entity index. It writes one JSONL row immediately after each variant so one OCC exception cannot abort the cohort.

The P0-A summary groups variants per CAD, names the baseline failure at joint 200 and tolerance `1e-3`, records whether joint removal or tolerance changes any validity stage, and assigns a primary cause. Saved STEP rows include native and strict validity plus wire, shell, free-edge, and solid component counts. The summary reports attributed cases divided by 16 and fails its acceptance gate below 0.8. A Markdown repair checklist maps observed cause families to bounded code changes; it is evidence, not an automatic broad repair.

Next extend `breparg_improvements/training_stability.py` with explicit precision parsing, serializable stop/training state helpers, RNG capture/restore, finite-gradient inspection, and experiment-signature validation. Extend `_train_vqvae` in `breparg_improvements/train.py` to use an autocast context selected by precision, unscale fp16 gradients before clipping, record pre-clip norm and effective LR, use `ReduceLROnPlateau` on the configured validation metric, trip a strict fuse on any non-finite event, and atomically save a full rolling state after each epoch. Existing best checkpoints remain loadable for evaluation; rolling P0-B checkpoints add fields rather than changing `model_state_dict`.

Add `tools/run_p0b_stability_retest.py` and focused tests. The launcher accepts only arms `vq_4096_64d_random` and `continuous_bypass_64d`, seeds 3 and 4, 60,000 train patches, 12,000 validation patches, batch 128, 100 epochs, and the frozen Protocol V5 split. It launches each arm separately so an interruption does not force unrelated arms to rerun. On restart it preserves logs, validates a full-state checkpoint signature, and resumes. It refuses promotion unless every expected train and validation batch is finite, every gradient norm is finite, and exactly 100 epochs exist.

Before the formal retest, run bounded fp32, fp16, and bf16 forward/backward probes on both P0-B arms. Select bf16 only if CUDA reports support and both arms match fp32 finiteness; otherwise select fp32. The probe is not used for quality comparison. Query Windows active hours and, if the current process has permission, set a documented 18-hour interval covering the expected run. Do not disable the Windows Update service. Record the before/after registry values and pending-restart check; failure to change policy is reported rather than silently ignored.

After all four P0-B trainings pass, choose the learned-VQ best checkpoint with the lowest finite curved parent MSE. Use the same deterministic selection seed `20260809` and the same 100 CAD identities as the existing assembly calibration. First run surface reconstruction. Then run the historical assembly pipeline at 200 joint iterations and sewing tolerance `1e-3`, retaining all attempts. Report learned-VQ strict, native, and both-valid rates beside original 84%, bypass 70%, and FSQ 49%, with an explicit note that 84/70/49 are historical strict-only runner values.

Finally evaluate the hard gate. If P0-A attribution coverage is at least 0.8, P0-B has zero non-finite events, and learned-VQ assembly evidence covers all 100 CADs, build a separate relation dataset from parsed per-CAD records. Each valid record binds one manifold edge to its two incident faces, their normalized patches, WCS bounding boxes, and entity indices; it verifies `edgeFace_adj` and `faceEdge_adj` bidirectionally and never uses cross-CAD content deduplication. Map decoded NCS points to WCS with `center=(bbox_min+bbox_max)/2`, `scale=max(bbox_max-bbox_min)`, and `x_wcs=x_ncs*scale/2+center`. The initial loss uses edge-to-surface `torch.cdist` only and penalizes the positive excess above the stopped-gradient ground-truth nearest-neighbor squared-distance baseline. The three weights 0, 0.1, and 1.0 share the same data, seeds, training budget, and assembly evaluator. Weight zero returns before creating or iterating the relation loader. If any P0 condition fails, do not implement or launch this loss; update this plan with the blocking evidence and repair the failed P0 first.

## Concrete Steps

Run all Git-managed commands from `D:/luolin/V13/v6git` with `C:/Users/YU/.conda/envs/brepgen_env/python.exe`.

The P0-A focused tests and real run have this shape:

    python -m pytest tests/test_diagnose_assembly_chain.py tests/test_assembly_calibration_oracle.py tests/test_diagnose_step_validity_components.py -q

    python tools/diagnose_assembly_chain.py --calibration-manifest reports/assembly_calibration_100cad_20260809/calibration_manifest.jsonl --breparg-root D:/luolin/V13/BrepARG --output-dir D:/luolin/V13/local_runs/p0a_assembly_chain_diagnosis_20260813 --joint-iterations 200,0 --sewing-tolerances 1e-4,1e-3,1e-2 --expected-invalid-cads 16

Expected terminal evidence includes `cases=16`, `attempts=96`, `attribution_rate>=0.8`, and a zero process exit status only when the gate passes.

The P0-B tests and precision probe will use:

    python -m pytest tests/test_vqvae_protocol_training.py tests/test_training_stability.py tests/test_run_p0b_stability_retest.py -q

    python tools/run_p0b_stability_retest.py probe --repo-root D:/luolin/V13/v6git --protocol-dir D:/luolin/V13/local_runs/protocol_v5_scaling_run_20260806/protocol --breparg-root D:/luolin/V13/BrepARG --output-root D:/luolin/V13/local_runs/p0b_stability_vq_bypass_60k_20260813

The formal launcher command will be recorded after the probe chooses precision. It must encode arms, seeds, caps, epochs, batch size, learning rate, gradient clipping, scheduler settings, and experiment signature in its state file. A restart repeats the same command and resumes only the active arm from its full-state rolling checkpoint.

After training, the launcher invokes the frozen 100-CAD surface and assembly evaluators. Expected summary evidence is four 100-epoch rows with `nonfinite_events=0`, one selected learned-VQ checkpoint hash, 100 reconstruction rows, 100 assembly attempts, and strict/native/both-valid counts.

## Validation and Acceptance

P0-A unit tests must prove frozen-case selection, exact six-variant construction, stage-local exception capture, strict-component decomposition, sensitivity classification, and the 80-percent attribution gate. A real one-CAD smoke must produce six rows without modifying the source pickle or old calibration. Full acceptance requires 96 rows for 16 CADs, at least 13 clearly attributed CADs, and a repair checklist whose item counts reconcile with case classifications.

P0-B unit tests must prove fp32/fp16/bf16 context selection, fp16 unscale-before-clip ordering, bf16 scaler disablement, finite-gradient fuse behavior, scheduler state persistence, atomic full-state checkpoints, exact resume epoch, RNG restoration, signature mismatch rejection, and zero-nonfinite validation. A CUDA smoke must execute forward, backward, clipping, optimizer step, checkpoint save, and resume for both VQ and bypass.

Formal P0-B acceptance is deliberately narrow: all four arm/seed histories reach epoch 99 with every expected train and validation batch finite and no finite-gradient or finite-parameter violation. Quality metrics are recorded but do not alter this stability gate. Learned-VQ assembly acceptance requires the identical 100 CAD identities and attempts denominator, with all reconstruction or assembly failures retained rather than excluded.

The boundary-consistency stage is accepted only by a strict-valid assembly-rate improvement on the frozen 100-CAD cohort relative to weight 0. Reconstruction MSE may be reported but cannot promote the loss. If weight 0.1 or 1.0 lowers MSE without improving assembly validity, the innovation hypothesis is not supported.

## Idempotence and Recovery

P0-A never overwrites the old calibration. Each attempt key includes source-manifest SHA-256, CAD identity, joint iteration count, sewing tolerance, and runner version. Re-running skips an exact completed row and regenerates summaries from all valid rows. STEP paths live below the new P0-A output root.

P0-B never writes into `protocol_v6_5seed_100epoch_20260810`. Each arm/seed has a separate directory and immutable experiment signature. Rolling checkpoint writes use a temporary sibling followed by `os.replace`. A stale or partial temporary file is ignored. A signature mismatch fails closed and requires a new output directory. Completed healthy arms are skipped; failed arms retain their logs and state.

Changing Windows active hours is bounded and reversible. The runner records the original values before changing them and writes a restoration command. It does not stop services, delete scheduled tasks, or disable updates. If elevation is unavailable, training can proceed only after the state records that active hours were not changed and automatic resume has passed its smoke test.

## Artifacts and Notes

Git tracks this plan, source, tests, concise JSON/JSONL/CSV/Markdown summaries, text logs, small TensorBoard events, and SHA-256 manifests. Git excludes all `.pt`, `.pth`, `.ckpt`, `.npz`, raw pickle, materialized CAD, generated STEP, PID, and upstream `BrepARG/` bytes. STEP files remain local; their relative paths, sizes, and SHA-256 values may be archived.

The historical comparison numbers 84% original, 70% continuous bypass, and 49% FSQ are strict-valid rates from the original calibration runner. P0-A additionally reports independent re-read native, strict, and both-valid values; it must not silently replace the historical denominator or label.

## Interfaces and Dependencies

`tools.diagnose_assembly_chain.select_frozen_failures(manifest_path, expected_count)` returns the 16 unique original-control invalid source records. `run_attempt(parsed, case, joint_iterations, sewing_tolerance, output_dir, breparg_root)` returns one fully serializable row and catches stage-local exceptions. `classify_case(rows)` returns a primary cause, secondary evidence, sensitivity flags, and `attributed: bool`. `summarize_cases(cases, attempts)` returns the acceptance gate and repair counts.

`breparg_improvements.training_stability.PrecisionPolicy` and `resolve_precision()` validate `fp32`, `fp16`, or `bf16` and provide autocast dtype and scaler policy. `capture_rng_state()` and `restore_rng_state(state)` cover Python, NumPy, Torch CPU, and all CUDA RNGs. `atomic_torch_save(payload, path)` writes a recoverable rolling checkpoint. `build_experiment_signature()`, `load_training_checkpoint()`, and `restore_training_checkpoint()` reject incomplete or mismatched resume state.

`breparg_improvements.train._train_vqvae` gains explicit precision, gradient-clip, scheduler, strict-fuse, rolling-checkpoint, resume, and experiment-signature parameters while preserving existing callers through defaults. Its history rows record LR, finite batch counts, non-finite event counts, pre-clip gradient norm, whether clipping was applied, and resume provenance.

`tools.run_p0b_stability_retest` exposes `probe` and `run` commands. The run state names the four arm/seed tasks, their exact signatures, rolling checkpoint paths, log paths, and effective status. It selects one healthy learned-VQ best checkpoint and invokes the existing reconstruction and assembly entry points with the frozen cohort.

If the gate opens, the boundary dataset yields topology-preserving shared-edge triplets rather than extending the current patch cache. The loss helper accepts predicted and ground-truth surfaces and edge, their WCS bounding boxes, and a weight; it returns total loss plus separately logged reconstruction, VQ, raw boundary distance, ground-truth distance floor, and weighted excess. Weight zero must be bit-for-bit equivalent to the current loss path, including learned-VQ buffers and optimizer state.

Revision note 2026-08-13: Created after the user prioritized assembly-chain attribution, stable resumable VQ/bypass retesting, and a hard gate before boundary consistency. It records the frozen cohorts, exact acceptance gates, the validity-definition mismatch, and the non-destructive Git/runtime artifact policy.

Revision note 2026-08-13 22:20 +08:00: Updated after completing P0-A. It records the 16/16 attribution result, distinguishes sensitivity from recovery, marks P0-A complete, and incorporates the topology-preserving relation-loader and ground-truth-baseline boundary-loss decisions discovered during the gated design audit.

Revision note 2026-08-14 00:20 +08:00: Updated after completing P0-B implementation and compatibility regression. It records the immutable launcher and measurement coordinator, reversible Windows active-hours change, exact resume evidence contract, explicit compatibility defaults, current test evidence, and the remaining CUDA/formal-runtime gate.
