# Complete Experiment Postmortem and Workspace Governance

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current while the investigation proceeds. It follows `PLANS.md` at the repository root.

## Purpose / Big Picture

The project has repeatedly produced technically writable but visually simple or invalid CAD geometry across both the V13 method and a same-data BrepARG baseline. This plan turns the accumulated experiments into one evidence-backed diagnosis. At completion, a researcher can read one report to understand the experiment timeline, code and data pipelines, environment, model and metric implementation, multi-run results, confirmed root causes, remaining unknowns, and the smallest next experiments that distinguish competing explanations. The workspace will also have a proposed maintainable directory layout and an explicit, non-destructive cleanup inventory; no artifact is deleted without user confirmation.

## Progress

- [x] (2026-07-31 16:30 Asia/Shanghai) Established the investigation scope and the required report sections zero through eight.
- [x] (2026-07-31 16:35 Asia/Shanghai) Confirmed that the repository worktree contains extensive modified and untracked research work after commit `16cf19b`; historical Git commits alone do not describe the July root-cause and BrepARG baseline work.
- [x] (2026-07-31 17:10 Asia/Shanghai) Inventoried repository code, configuration, data references, environments, logs, checkpoints, generated artifacts, drive health, disk usage, duplicate hashes, and NTFS hardlinks.
- [x] (2026-07-31 17:20 Asia/Shanghai) Reconstructed the experiment timeline and normalized V13/BrepARG/FSQ/DFS-RCM result matrix, keeping incompatible generation protocols separate.
- [x] (2026-07-31 17:30 Asia/Shanghai) Audited data provenance, split overlap, sequence distributions, context coverage, preprocessing, loading, augmentation, and source-path integrity.
- [x] (2026-07-31 17:40 Asia/Shanghai) Audited model, loss, optimizer, training loop, checkpoint restoration, train/eval behavior, generation, and metric implementations.
- [x] (2026-07-31 17:50 Asia/Shanghai) Classified conclusions as confirmed, strongly inferred, or evidence insufficient, and recorded rejected and still-open hypotheses.
- [x] (2026-07-31 18:00 Asia/Shanghai) Wrote `docs/full_experiment_postmortem_20260731.md` with all required sections zero through eight and evidence paths.
- [x] (2026-07-31 18:03 Asia/Shanghai) Added a proposed target directory tree, reversible migration sequence, and exact-path deletion candidates separated from unmanifested cleanup categories.
- [x] (2026-07-31 18:09 Asia/Shanghai) Verified report structure and UTF-8, 114 concrete evidence paths, key JSON/checkpoint transcriptions, focused tests, current process/GPU/drive status, Git whitespace, and absence of deleted or renamed paths.

## Surprises & Discoveries

- Observation: Git history is not a complete experiment ledger for the current state.
  Evidence: `git status --short` shows many modified and untracked implementation, test, tool, plan, and documentation files, while `git log -1` is commit `16cf19b` from the earlier AR low-learning-rate branch.
- Observation: The long same-data original BrepARG run improved validation cross entropy and the tail of generated complexity, but did not change the median generated topology.
  Evidence: The prior 80-epoch baseline had `5/92` complex outputs and `0/92` strict acceptances; the VQ-VAE-400/AR-300 run had `13/100` complex and `6/100` strict acceptances, while both had median `6` faces and `12` edges.
- Observation: The current V13 split is not parent-CAD independent.
  Evidence: `docs/audits/v13_sequence_split_integrity_20260731.json` reports that `56.75%` of validation records and `57.17%` of test records share a parent CAD with another split; the same-data BrepARG split also leaks at the parent level.
- Observation: Current-package distributions cannot automatically be assigned to every historical training run.
  Evidence: The current ubuntu package is split `382,903/21,214/21,003`, while the local 1024-context AR preflight recorded `382,720/21,124/21,276`; both total 425,120 but no record-level identity manifest exists.
- Observation: Complex geometry fails before free-running AR.
  Evidence: On 50 complex-curved shapes and 3,399 patches, surface Chamfer p95 is `0.41238`; true-token reconstruction saves only `27/50` STEP and passes the current custom BRep check for `9/50`.
- Observation: The original BrepARG AR checkpoint namespace is duplicated but the inspected long-run checkpoint did not lose learned tensors.
  Evidence: The epoch-127 checkpoint has 202 keys, 101 raw plus 101 equal `model.*` duplicates; permissive load reports 101 unexpected keys but no missing keys. This is a future reproducibility hazard, not a supported explanation for the observed quality failure.
- Observation: Disk layout statistics based on filenames substantially overstate reclaimable space.
  Evidence: Six 1.349 GiB V13 sequence paths are hardlinks to one NTFS file; removing five aliases releases approximately zero physical GiB. E: simultaneously reports `Full Repair Needed` and cannot be the sole authority.

## Decision Log

- Decision: Separate facts, strong inferences, and unknowns in the final report.
  Rationale: Several prior conclusions combine direct metric evidence with causal interpretation; preserving confidence levels prevents a plausible hypothesis from being reported as proven.
  Date/Author: 2026-07-31 / Codex.
- Decision: Treat workspace cleanup as a proposal until the user approves exact paths.
  Rationale: The repository is a dirty research worktree and several apparently redundant files may be the only record of an experiment or recovery event.
  Date/Author: 2026-07-31 / Codex.
- Decision: Evaluate BrepARG and V13 using matched data and common quality gates, while separately labeling official-weight incompatibility.
  Rationale: The available official BrepARG checkpoint uses a different vocabulary shape and cannot serve as a direct same-protocol numerical baseline.
  Date/Author: 2026-07-31 / Codex.
- Decision: Treat the component diagnosis as an ordered oracle ladder: ground-truth assembly, continuous decoder, FSQ, teacher-forced argmax AR, then free-running AR.
  Rationale: Existing true-token reconstruction still combines FSQ decode and OCC assembly, so calling FSQ the sole cause would exceed the evidence.
  Date/Author: 2026-07-31 / Codex.
- Decision: Keep survivor-conditioned quality, attempt-conditioned validity, and rejection-gate acceptance as separate metrics.
  Rationale: Combining them previously made retained STEP quality look like model-level generation success.
  Date/Author: 2026-07-31 / Codex.

## Outcomes & Retrospective

The investigation produced `docs/full_experiment_postmortem_20260731.md`, a single zero-through-eight report that separates experimental operability from scientific success, distinguishes confirmed facts from inference, normalizes incompatible metric denominators, and proposes an ordered repair program. The central technical result is not that VQ, AR, and ordering failed equally: complex surface/edge reconstruction and BRep assembly have the strongest direct failure evidence; AR adds a second failure mode; DFS-versus-RCM is a smaller, incompletely reproduced signal. The central scientific-validity result is that parent-CAD leakage and incompatible validity denominators currently prevent a fair paper-level claim.

The governance audit found that large apparent duplicates are often NTFS hardlinks, while the largest defensible derivative candidate is `data_staged`. Exact-path candidates totaling about 7.743 GiB are documented but remain untouched pending user approval. Vague cache categories were deliberately kept outside the approval-ready list until an exact manifest exists. E: remains unhealthy and is not an archive authority.

Fresh acceptance evidence: five focused split/coverage tests passed; `git diff --check` exited 0 with only existing LF/CRLF warnings; all 114 concrete report references resolved; the returned V13 AR/VQ checkpoints were finite; no Git deletion or rename appeared; and no V13/BrepARG training or generation process was running.

## Context and Orientation

`BrepARG` contains the upstream-style baseline implementation: surface/edge VQ-VAE training, sequence construction, autoregressive training, and BRep generation. `breparg_improvements` contains the V13 method and training pipeline, including finite scalar quantization (FSQ), RCM/GNN ordering, and the V13 autoregressive model. `tools` contains experiment launchers, validators, diagnostics, migration helpers, and report builders. `tests` contains focused regression tests for those helpers. `plans` contains chronological execution plans, but several overlap and must be indexed rather than treated as independent current truth.

Large data and run outputs live both under the workspace and under `D:\V13_rootcause_recovery_20260717`. The latter contains same-data BrepARG fallback and long-baseline artifacts. The user previously attempted migration to `E:`, but the drive reported `Full Repair Needed`; important artifacts must not be removed from D based only on an E-drive copy.

The final report will distinguish reconstruction from generation. Reconstruction feeds known or true tokens through a decoder and tests representational and geometry-building capability. Free-running generation samples tokens from an autoregressive model and additionally tests sequence modeling, exposure bias, grammar, and reconstruction. A writable STEP file is not by itself a high-quality CAD result; the common audit also tests OpenCascade readability, BRep validity, closed-solid structure, topology complexity, and a strict quality gate.

## Plan of Work

First, collect immutable facts from Git, repository files, environment metadata, drive state, experiment manifests, logs, and checkpoint summaries. Record paths and timestamps without moving data. Second, normalize historical experiments into a version matrix that names the data split, representation, ordering, VQ-VAE checkpoint, AR checkpoint, training epochs, generation settings, and quality outcomes. Third, trace the data and model code paths and compare training with inference, including checkpoint restore semantics and metric definitions. Fourth, rank root causes only after those component-level checks, and define one-variable-at-a-time experiments with explicit stop/go thresholds. Fifth, write the final report and a proposed target workspace layout. Cleanup candidates will be grouped as definitely rebuildable, likely archival, and uncertain; all deletion actions remain pending user approval.

## Concrete Steps

Run all evidence gathering from `D:\luolin\V13`. Use PowerShell and `rg` for text and file discovery. Use the configured Python environments only for read-only checkpoint or package inspection. Do not run long training jobs during the audit.

The primary report will be written to:

    D:\luolin\V13\docs\full_experiment_postmortem_20260731.md

The report must link to the source plans, code, logs, JSON summaries, manifests, and environment files used for each conclusion. A separate machine-readable inventory may be added under `docs\audits` if needed to keep the report readable.

## Validation and Acceptance

The work is accepted when the report contains every requested section zero through eight, identifies current evidence and missing evidence, provides source paths for each conclusion, includes an explicit rejected-hypothesis list, and gives tiered repairs with measurable acceptance thresholds. The workspace section must include the current top-level tree, a proposed target tree, exact move/archive candidates, and exact deletion candidates marked as requiring approval. Verification must confirm that no path in a deletion list was actually deleted, that cited JSON metrics match their source files, and that local Markdown links resolve where the target exists.

## Idempotence and Recovery

All audit commands are read-only. Report and plan edits are additive. Re-running inventory commands may update timestamps or free-space values but must not alter artifacts. Any future cleanup phase must first copy or hash-verify retained artifacts, write a manifest, and receive user approval before deletion. E: must not be treated as the only authoritative copy until its filesystem health is repaired and rechecked.

## Artifacts and Notes

Primary existing evidence includes:

    plans\ar_training_v13_execplan.md
    plans\complex_curved_fsq_ar_diagnostics_execplan.md
    plans\complex_curved_rootcause_current_status_20260718_execplan.md
    plans\breparg_long_baseline_and_v13_diagnosis_20260720_execplan.md
    D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\breparg_same_data_resume_best_quality_summary_20260726.json

## Interfaces and Dependencies

The audit uses Git, PowerShell, `rg`, JSON parsing through PowerShell or Python, PyTorch checkpoint inspection, and OpenCascade-derived quality summaries already produced by repository tools. No new model API is required. If an audit helper is added, it must emit JSON with source paths, timestamps, and confidence labels so the report can be regenerated without relying on conversation memory.

Revision note 2026-07-31: Created this plan to consolidate the full experiment postmortem and workspace governance work requested by the user. It explicitly separates evidence collection, causal diagnosis, and deletion approval to prevent further loss of research provenance.

Revision note 2026-07-31 18:09 +08:00: Completed the report, integrated the normalized experiment matrix and checkpoint-loading audit, replaced ambiguous evidence references with concrete paths, narrowed cleanup to exact approval candidates, and recorded fresh verification evidence.
