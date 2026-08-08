# Calibrate representation error against assembled CAD validity

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current while the work proceeds. This document follows `PLANS.md` in the repository root. `AGENTS.md` refers to `.agent/PLANS.md`, but that path is absent in this checkout; the checked-in root `PLANS.md` is the available authority.

## Purpose / Big Picture

Protocol V5 showed that a continuous 64-dimensional latent reconstructs curved patches much better than either finite scalar quantization configuration, but even the continuous result remains above the inherited `5e-5` curved-MSE target. Patch MSE alone does not say whether reconstructed patches can be assembled into usable CAD. This plan creates that missing calibration: reconstruct 100-200 parent-isolated validation CAD records with the existing continuous-bypass and FSQ-8192/4D checkpoints, assemble the surfaces and edges with the same joint optimization and OpenCascade pipeline used downstream, save STEP, check strict BRep validity, and measure how validity changes with per-CAD curved reconstruction error.

The calibration result determines the next work. If validity is acceptable at current error, the historical `5e-5` gate is relaxed and the learned VQ-4096/64D 300k experiment starts. If validity is low and strongly related to reconstruction error, the observed calibration curve supplies a task-derived representation gate and authorizes stability or decoder work. If validity is low and weakly related to reconstruction error, assembly is the primary suspect and must be repaired before decoder changes. No sequence regeneration or autoregressive training is allowed until the representation gate passes.

## Progress

- [x] (2026-08-09 10:00 +08:00) Archived and pushed the lightweight Protocol V5 evidence, including both 300k sweeps, both bypass sweeps and histories, scaling analysis, ladder state, oracle logs, and checkpoint SHA-256 bindings, as commit `e6c0061`.
- [x] (2026-08-09 10:20 +08:00) Confirmed that the old token reconstruction evaluator cannot be used as the calibration entry point because Protocol V5 has patch checkpoints but no matching sequence package.
- [x] (2026-08-09 10:35 +08:00) Proved the assembly path on one validation CAD using unmodified parsed NCS patches and original topology: 16 faces and 36 edges produced a 127,729-byte STEP that passed strict BRep validity.
- [x] (2026-08-09 11:15 +08:00) Added tests for deterministic validation-CAD selection, parent isolation, edge decode semantics, CAD-level bucket MSE, failure retention, duplicate-safe bins, three-way decisions, and a Matplotlib-free PNG renderer; 12 focused tests pass.
- [x] (2026-08-09 11:20 +08:00) Implemented the fail-closed calibration runner with strict checkpoint loading, direct patch reconstruction, CPU joint optimization, STEP/OCC validation, attempt-preserving JSONL, restart keys, and Pillow summary rendering.
- [x] (2026-08-09 11:25 +08:00) Ran a one-CAD three-arm smoke cohort. Original and continuous bypass were strict BRep-valid; bypass curved MSE was `7.5644e-5`. FSQ-8192/4D curved MSE was `0.0020393` and its STEP was BRep-invalid.
- [ ] Run 100-200 validation CAD for continuous bypass and FSQ-8192/4D, write the calibration curve and decision, and push code plus lightweight evidence.
- [ ] Follow exactly one decision branch: accept current gate and start learned VQ 300k; define and investigate a representation gate; or repair assembly first.
- [ ] Apply the explicit wall trigger after calibration plus conditional stability/decoder work: if curved error remains more than five times the task-derived gate, stop for a separate representation-upgrade cost/benefit review.

## Surprises & Discoveries

- Observation: Protocol V5's split is suitable for CAD-level calibration without a new data split.
  Evidence: `protocol_summary.json` records 1,500 validation records from 1,500 validation parents, zero parent overlap, and protocol status `VERIFIED`.

- Observation: The existing reconstruction evaluator already contains a CPU-safe implementation of BrepARG joint surface/edge optimization and strict STEP validation, but its public entry point consumes AR token sequences.
  Evidence: `tools/evaluate_reconstruction_v13.py` defines `cpu_safe_joint_optimize` and patches `BrepARG.utils.joint_optimize`, while `evaluate()` selects records from a sequence pickle and calls `reconstruct_cad_from_sequence`.

- Observation: Protocol V5 checkpoints are full patch autoencoders, so calibration can reconstruct every known face and edge without tokenization or AR.
  Evidence: `breparg_improvements/train.py` constructs FSQ and continuous variants around the same `VQModel`; parsed CAD pickle records contain `surf_ncs`, `edge_ncs`, `faceEdge_adj`, and world-coordinate geometry needed to recover topology and placement.

- Observation: The first matched-CAD smoke result is directionally consistent with a representation bottleneck, but is not sufficient for a decision.
  Evidence: The original and bypass arms assembled as strict-valid, while FSQ failed strict validity on the same topology; bypass curved MSE was about 27 times lower than FSQ. The formal decision remains gated on at least 100 CAD.

## Decision Log

- Decision: Use the Protocol V5 validation split and deterministic parent/CAD selection, not training records or an old sequence split.
  Rationale: The calibration must remain independent of model fitting and inherit the audited parent isolation.
  Date/Author: 2026-08-09 / Codex

- Decision: Run an original-patch assembly control before model checkpoints.
  Rationale: This separates defects in topology/assembly from errors introduced by patch reconstruction. A model cannot be blamed when the same CAD fails with its original parsed patches.
  Date/Author: 2026-08-09 / Codex

- Decision: Reconstruct patches directly through the VQ-VAE forward path and preserve original CAD topology and bounding information.
  Rationale: Creating a sequence would add quantization/token grammar and ordering variables that are outside the representation-to-assembly question.
  Date/Author: 2026-08-09 / Codex

- Decision: Treat attempts as the validity denominator and report original-control, STEP-saved, BRep-valid, and failure stages separately.
  Rationale: Success-only denominators hide assembly failures and would repeat the evaluation ambiguity already identified in earlier experiments.
  Date/Author: 2026-08-09 / Codex

- Decision: Use deterministic bootstrap confidence intervals and both rank and point-biserial association between error and validity; never decide from a single correlation coefficient.
  Rationale: Validity is binary, MSE is skewed, and 100-200 CAD provide limited statistical resolution. Reporting multiple robust summaries prevents a fragile threshold claim.
  Date/Author: 2026-08-09 / Codex

- Decision: Do not upload checkpoint binaries or retained STEP files; upload code, summaries, manifests, logs converted to text, plots under an allow-listed reports directory, and checkpoint hashes.
  Rationale: This preserves reproducibility across machines without turning Git into artifact storage.
  Date/Author: 2026-08-09 / Codex

## Outcomes & Retrospective

Protocol V5 evidence is now version-controlled and bound to local checkpoints by SHA-256. The direct reconstruction and assembly calibration path has passed both an original-data control and a matched three-arm smoke test. The smoke outcome is informative but not a scientific decision because it contains one CAD. This section must be updated after the 100-200 CAD cohort, including validity by arm, the error-validity relationship, the selected branch, and any limitations caused by nonfinite reconstructions or OCC failures.

## Context and Orientation

The version-controlled runtime repository is `D:\luolin\V13\local_runs\protocol_v5_runtime_20260806`. The completed experiment workspace is `D:\luolin\V13\local_runs\protocol_v5_scaling_run_20260806`. The verified split is `protocol\split.pkl`; materialized validation CAD pickles are under `materialized`, and `protocol_manifest.jsonl` binds archive members to split assignments.

The two model inputs are `rungs\continuous_bypass_300k\seed1\continuous_bypass_64d_best.pt` and `rungs\300k\seed0\fsq_8192_4d_best.pt` by default. Seed selection must be recorded and may be expanded to both seeds if the first 100-CAD calibration is noisy. The checkpoints are excluded from Git and identified by `reports\protocol_v5_scaling_20260807\checkpoint_sha256.tsv`.

A normalized-coordinate-space patch, abbreviated NCS patch, is a 32 by 32 surface point grid or a 32-point edge curve centered and scaled independently. Joint optimization places these decoded patches back into world coordinates using the parsed surface bounding boxes, reconstructed edge endpoints, and known face-edge incidence. OpenCascade then fits curves and surfaces, trims faces, constructs the BRep, writes STEP, and checks whether the resulting topology and geometry are valid.

`D:\luolin\V13\BrepARG\utils.py` is a local upstream source dependency and is not copied into this Git branch. The calibration runner must accept `--breparg-root` and fail clearly if the dependency or OpenCascade is unavailable.

## Plan of Work

First inspect one materialized validation pickle and prove original-patch assembly. Implement topology extraction as a small pure function where possible: use `faceEdge_adj`; derive edge-face incidence; obtain or deterministically merge edge endpoint vertices using the same tolerance and logic as the established reconstruction path; preserve surface bounding boxes from parsed world-coordinate surfaces or explicit bbox fields. Any malformed or missing field is a per-CAD failure with a stage label, never a silent exclusion.

Add `tools/run_assembly_calibration_oracle.py`. It will read the verified split, deterministically select CADs, load each checkpoint once, reconstruct surfaces and tiled edge patches in bounded batches, compute per-CAD surface-curved, planar, edge, and aggregate MSE, and call a separately testable assembly adapter. It writes one JSONL manifest row per arm and CAD immediately so interruption is recoverable. Retained STEP files remain in `local_runs`; the manifest records their hashes and sizes.

Add `tools/summarize_assembly_calibration.py`. It will validate the manifest, aggregate all attempts, compute validity rates and confidence intervals, form log-spaced or quantile MSE bins, measure association between log curved MSE and BRep validity, and issue one of `CURRENT_ERROR_ACCEPTABLE`, `REPRESENTATION_ERROR_CORRELATED`, or `ASSEMBLY_DOMINATED`. The decision must be conservative when the original-patch control itself is poor or sample coverage is insufficient. Render the calibration PNG with Pillow to avoid the Matplotlib crash already observed on this Windows host.

After tests pass, run an original-control smoke cohort of at least three validation CADs. Then run the same CADs through bypass and FSQ. Inspect saved STEP files and OCC validation counts. Scale to 100 CAD first; increase to 200 only if confidence intervals or association estimates remain inconclusive and runtime/storage are acceptable.

Push code before the long cohort. After the cohort, copy only lightweight manifests, summary, plot, text logs, and hashes into `reports\assembly_calibration_<date>` and push a separate results commit. The chosen branch becomes the next milestone: learned VQ 300k for acceptable current error, assembly repair for assembly-dominated failure, or stability/decoder investigation for error-correlated failure.

## Concrete Steps

All Git-managed commands run from `D:\luolin\V13\local_runs\protocol_v5_runtime_20260806`. Python commands use `C:\Users\YU\.conda\envs\brepgen_env\python.exe` for PyTorch and `C:\Users\YU\.conda\envs\edit2patch_freecad\python.exe` only if OpenCascade is unavailable in `brepgen_env`. The runner must detect and report the actual environment rather than silently split model and assembly work across incompatible processes.

The smoke command will have this shape after implementation:

    python tools\run_assembly_calibration_oracle.py --protocol-dir D:\luolin\V13\local_runs\protocol_v5_scaling_run_20260806\protocol --materialized-root D:\luolin\V13\local_runs\protocol_v5_scaling_run_20260806\materialized --breparg-root D:\luolin\V13\BrepARG --checkpoint continuous_bypass_64d=<path> --checkpoint fsq_8192_4d=<path> --include-original-control --max-cads 3 --output-dir D:\luolin\V13\local_runs\assembly_calibration_smoke

The full cohort changes `--max-cads` to 100 and records its selection seed. If the summary says evidence is insufficient, rerun only the missing CADs up to 200 using the same manifest and deterministic order.

## Validation and Acceptance

Pure unit tests must verify deterministic CAD selection, rejection of a non-verified or overlapping split, correct patch-shape conversion for edges, per-CAD MSE calculation, attempts-based validity aggregation, confidence intervals, error bins, and all three decision outcomes. A fake assembly adapter must prove that individual failures are recorded and do not abort the cohort.

Integration acceptance requires the original-patch control to write at least one STEP from a real validation pickle and independently run strict OCC validation. Before scaling beyond smoke, both model arms must load their bound checkpoints, reconstruct finite arrays of the expected shapes, and attempt the same selected CAD identities.

The full calibration is accepted when at least 100 selected validation CADs have manifest rows for original, bypass, and FSQ; the summary reports attempts, STEP-saved rate, strict BRep-valid rate, failure stages, curved-MSE bins, confidence intervals, and one fail-closed decision. The original-control validity rate is always shown alongside checkpoint arms.

## Idempotence and Recovery

The runner writes JSONL rows after each arm/CAD and uses a stable key composed of protocol hash, source identity, arm, checkpoint hash, and runner version. On restart it verifies existing rows and skips exact completed keys. STEP filenames include stable CAD and arm identifiers and may be safely overwritten only when their manifest binding matches.

If a checkpoint produces nonfinite patches, the CAD row records `nonfinite_reconstruction` and counts against attempts. If joint optimization or OCC fails, the row records the exact stage and exception type. A single CAD must never abort the cohort. If the original control is broadly invalid, stop model interpretation and select the assembly-dominated branch.

## Artifacts and Notes

Protocol V5's continuous bypass reached curved parent MSE around `4e-4` to `7e-4`, while quantized arms remained around `2e-3` or worse. Those values are patch-level validation metrics, not CAD assembly thresholds. This plan deliberately avoids treating `5e-5` as ground truth until STEP/BRep evidence calibrates it.

The representation-upgrade trigger is explicit: after the calibration and any authorized continuous stability/decoder work, if curved MSE remains more than five times the task-derived gate, stop and open a separate review. Do not add wider token grids, larger latent maps, or other sequence-length-changing architecture changes opportunistically. In particular, a 4 by 4 latent grid is last because it doubles downstream AR token length.

## Interfaces and Dependencies

`tools.run_assembly_calibration_oracle.select_validation_cads(...)` returns deterministic, parent-unique materialized CAD records bound to the verified protocol. `reconstruct_patch_batch(model, patches, kind, device, batch_size)` returns finite NCS arrays with exactly the input semantic shape. `extract_assembly_inputs(parsed)` returns face-edge and edge-vertex topology plus world-coordinate placement or raises a labeled validation error. `evaluate_cad_arm(...)` returns one serializable manifest row and catches stage-local exceptions.

`tools.summarize_assembly_calibration.summarize_calibration(rows, ...)` returns arm summaries, confidence intervals, binned curves, association estimates, and a three-way decision. `render_calibration_png(summary, output_path)` uses Pillow and produces a version-controlled plot without Matplotlib.

The runtime dependencies are NumPy, PyTorch, Pillow, the local BrepARG source tree, and pythonocc-core/OpenCascade. No new model library is introduced.

Revision note 2026-08-09: Created the assembly-calibration execution plan after Protocol V5 evidence was pushed. It records the direct patch reconstruction design, original-data control, three-way branch logic, Git evidence policy, and five-times wall trigger.

Revision note 2026-08-09 11:25 +08:00: Recorded the successful original-control and three-arm smoke integrations, focused test coverage, and the explicit restriction that the one-CAD result cannot select a scientific branch.
