# V13 Project Organization, 100 STEP Generation, and AAAI Paper ExecPlan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document follows `PLANS.md` in the repository root.

## Purpose / Big Picture

The user wants the V13 project to move from model training evidence into publication-ready evidence. That means organizing the project files without losing training or reconstruction artifacts, generating a larger random set of 100 STEP files from the best model, summarizing the generation quality and diversity, preparing for an AAAI conference submission, and drafting the paper. A future reader should be able to open this plan, find the best model, find the generated STEP files and statistics, see what evidence supports the paper claims, and continue the manuscript work without relying on chat history.

## Progress

- [x] (2026-07-05 00:44 +08:00) Read `AGENTS.md` and `PLANS.md`; confirmed complex work should use a living ExecPlan.
- [x] (2026-07-05 00:45 +08:00) Inspected the V13 repository root, top-level directories, tracked helper files, AR training output directories, and recent reconstruction runs.
- [x] (2026-07-05 00:48 +08:00) Created this ExecPlan for the combined project organization, 100-sample generation, AAAI preparation, and paper draft goal.
- [x] (2026-07-05 00:55 +08:00) Generated 100 random samples from the epoch-120 best AR checkpoint with `temperature=0.9`, `top_p=0.92`, `max_new_tokens=320`, constrained decoding, fresh seed `3911446532`, and retained STEP files under `local_runs\reconstruction_eval\eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_005342`.
- [x] (2026-07-05 00:58 +08:00) Computed the 100-sample statistics and wrote `local_reports\generated100_lr5e6_epoch120_best_20260705.md`: 87/100 grammar-valid, 87/100 reconstructed and saved as STEP, 78/100 strict BREP-valid, 87 unique STEP SHA256 hashes, 13 `truncated (no END)` grammar failures, and 9 saved STEP files that failed strict BREP validation.
- [x] (2026-07-05 01:02 +08:00) Built a non-destructive project inventory and organization proposal in `local_reports\project_inventory_and_organization_20260705.md`; no files were moved or deleted.
- [x] (2026-07-05 01:08 +08:00) Checked official AAAI-27 pages and wrote `local_reports\aaai27_preparation_brief_20260705.md`, including deadlines, page limits, review process, anonymity expectations, supplementary material constraints, and paper-positioning risks.
- [x] (2026-07-05 01:08 +08:00) Built an initial related-work and positioning map covering DeepCAD, SolidGen, BrepGen, BrepARG, AutoBrep, BrepGPT, and RECAD in `local_reports\aaai27_preparation_brief_20260705.md`.
- [x] (2026-07-05 01:13 +08:00) Created the paper workspace `papers\aaai_v13` with `README.md`, `evidence_map.md`, and `draft_aaai27_v13_cad_generation.md`.
- [x] (2026-07-05 01:13 +08:00) Drafted a first full AAAI-style Markdown manuscript using the verified training and 100-sample generation artifacts, while explicitly marking missing baseline, figure, metric, and formatting work as limitations.
- [x] (2026-07-05 01:17 +08:00) Added `papers\aaai_v13\selected_step_examples.md`, listing representative BREP-valid STEP files by topology plus saved-but-invalid and grammar-failed examples for figure and failure-analysis selection.
- [x] (2026-07-05 01:25 +08:00) Verified pythonocc offscreen rendering works in `brepgen_env`, then added `papers\aaai_v13\render_selected_steps.py` and rendered 12 selected STEP examples plus `figures\step_renders\selected_step_contact_sheet.png`.
- [x] (2026-07-05 01:29 +08:00) Added publication-preparation materials: `papers\aaai_v13\tables\experiment_summary.md`, `papers\aaai_v13\reproducibility_checklist_draft.md`, and `papers\aaai_v13\supplementary_material_plan.md`.
- [x] (2026-07-05 01:30 +08:00) Updated the manuscript draft, paper README, and evidence map to reference the rendered figure candidates, experiment tables, reproducibility draft, and supplementary material plan.
- [x] (2026-07-05 01:33 +08:00) Added root `PROJECT_INDEX.md` as a lightweight organization entry point for source, canonical checkpoints, generated STEP evidence, paper materials, and directories that should not be moved casually.
- [x] (2026-07-05 01:37 +08:00) Created `papers\aaai_v13\supplement_staging` and `aaai_v13_supplement_staging_20260705.zip` with 9 valid STEP examples, 3 BREP-invalid STEP examples, rendered figures, the 100-sample reconstruction report/manifest, AR history, and a README; verified the zip contains 18 small entries and no checkpoints or datasets.
- [x] (2026-07-05 16:14 +08:00) Converted the paper draft into an AAAI-27 LaTeX workspace under `papers\aaai_v13\latex`, copied the local AuthorKit27 style files, wrote `main.tex` and `references.bib`, compiled `main.pdf`, and rendered all 4 pages to PNG for visual verification.
- [x] (2026-07-05 16:14 +08:00) Ran an additional diagnostic generation from the epoch-120 best checkpoint with `temperature=0.95`, `top_p=0.95`, and `max_new_tokens=512`; it retained 19 unique STEP files from 20 attempts, with 16 strict BREP-valid files and only 1 `truncated (no END)` grammar failure.
- [x] (2026-07-05 20:52 +08:00) Repositioned the LaTeX paper as a diagnostic/reproducibility draft, compiled `papers\aaai_v13\latex\main.pdf` as a 7-page AAAI-style main paper, compiled `papers\aaai_v13\latex\supplement.pdf` as a supplement, and rendered both PDFs to PNG pages under `papers\aaai_v13\latex\rendered` for visual inspection. After the 21:55 recovery-controls update, the supplement renders as 13 pages.
- [x] (2026-07-05 20:52 +08:00) Added the VQ-VAE-only validation-longest diagnostic and the rendered generated-20 diagnostic to the paper evidence, explicitly labeling both as failure/diagnostic evidence rather than positive qualitative results.
- [x] (2026-07-05 21:20 +08:00) Added the complexity-sliced VQ-VAE-only benchmark to the supplement and local reports, showing degradation from shortest validation rows to longest and most-face validation rows.
- [x] (2026-07-05 22:35 +08:00) Added the benchmark-summary promotion gate to the supplement and paper README; the current VQ100 summary is retained as `local_runs\reconstruction_eval\vqvae_epoch100_complexity_benchmark_20260705_benchmark_summary.json` and correctly holds the checkpoint.
- [x] (2026-07-05 22:51 +08:00) Added server handoff manifest coverage to the supplement and paper README so future improved figures must come back with checkpoint, history, ledger, benchmark reports, manifests, retained STEP files, and rendered contact sheets.
- [x] (2026-07-05 23:01 +08:00) Added generated-run paper-gate summaries for G20 and G100; both current generated runs are now machine-labeled `hold_for_failure_analysis`, matching the decision not to use them as positive qualitative figures.
- [x] (2026-07-05 23:17 +08:00) Rendered the full G100 contact sheet, reran the generated-run gate, and updated the supplement/README language so G100 is held for topology collapse and missing complex strict-valid outputs, not for a missing render.
- [x] (2026-07-05 23:39 +08:00) Added optional `most_curved` VQ-VAE diagnostic tooling and updated paper notes to say it requires source-path-aware sequence packages before it can become evidence.
- [x] (2026-07-06 00:00 +08:00) Recorded the new sequence-source provenance hook: future sequence rebuilds through `BrepARG/2sequence.py` now attach `source_path`, and the helper test suite reports 66 passed. This makes future `most_curved` diagnostics auditable after a sequence rebuild, but it does not change the current old `SEQ` package.
- [x] (2026-07-06 00:23 +08:00) Expanded the main LaTeX paper with a quality-recovery protocol table that makes the VQ-VAE-first restart, source-path-aware sequence rebuild, long-context AR branches, and generated-run paper gate explicit; recompiled `main.pdf` and verified it remains 7 pages.
- [x] (2026-07-06 00:36 +08:00) Added a source-path audit tool and report for the current AR120 sequence package; the report confirms that current sequence metadata is insufficient for a real `most_curved` paper claim.
- [x] (2026-07-06 00:52 +08:00) Added a Linux source-path-aware sequence rebuild launcher for the next server run, so a future paper update can trace curved-surface evidence to a rebuilt, audited sequence package.

## Surprises & Discoveries

- Observation: The best available AR checkpoint is the `lr=5e-6` epoch-120 checkpoint, not an older epoch-95 or epoch-76 checkpoint.
  Evidence: The previous training plan `plans\ar_training_v13_execplan.md` records epoch 120 with `val_CE=0.29493329663972306`, and the inspected output directories show `newscheme_full_v13_ar_lr5e6` as the latest AR training branch.
- Observation: The current repository already separates source-like files from generated heavy artifacts in practice, but it has no single publication inventory.
  Evidence: Source and helper files are under `BrepARG`, `breparg_improvements`, `tools`, `tests`, `plans`, and `local_reports`; heavy outputs are under `ABC`, `local_runs`, `processed_local`, and `breparg_improvements\repro_outputs`.
- Observation: Only small generated reconstruction batches are present so far for the epoch-120 best checkpoint.
  Evidence: The newest reconstruction directories are three 6-sample runs under `local_runs\reconstruction_eval`, with names beginning `eval_generated6_lr5e6_epoch120_best_...`.
- Observation: The 100-sample generated run confirms non-duplicate retained STEP files, but also exposes a diversity concentration.
  Evidence: `local_reports\generated100_lr5e6_epoch120_best_20260705.md` records 87 retained STEP files with 87 unique SHA256 hashes. The top two saved topologies, 6 faces / 12 edges and 4 faces / 6 edges, account for 62 of 87 saved STEP files.
- Observation: The main failure mode at the selected 100-sample setting is incomplete generated sequences rather than reconstruction crashes.
  Evidence: All 13 generation errors in the 100-sample run have grammar reason `truncated (no END)`. The 87 grammar-valid rows all reconstructed and saved STEP files, though 9 of those saved files failed strict BREP validation.
- Observation: Project storage is dominated by offline data archives and AR training checkpoints, not source files.
  Evidence: `local_reports\project_inventory_and_organization_20260705.md` records `ABC` at about 167.39 GiB and `local_runs` at about 8.95 GiB, while source, tools, tests, plans, and reports are each only KiB-to-MiB scale.
- Observation: The current paper draft can support a preliminary technical-report claim, but the AAAI main-track novelty and comparison bar remains unmet.
  Evidence: `local_reports\aaai27_preparation_brief_20260705.md` identifies missing baseline comparisons, larger validity/diversity metrics, qualitative figures, ablations, and reproducibility checklist answers; the draft records these as limitations rather than claiming state-of-the-art performance.
- Observation: STEP rendering is available in the current `brepgen_env`, so the paper can include qualitative figures without installing a new renderer.
  Evidence: `papers\aaai_v13\render_selected_steps.py` rendered 12 STEP examples and wrote `figures\step_renders\render_manifest.json`; pixel checks showed each generated PNG is non-flat, and visual inspection of `selected_step_contact_sheet.png` shows nonblank CAD renderings.
- Observation: Visual inspection changed the paper positioning from positive generation claims to diagnostic evidence.
  Evidence: The rendered generated-20 contact sheet contains mostly bars, plates, rings, cylinders, and malformed surfaces. The VQ-VAE-only validation-longest contact sheet shows failures on real long validation sequences before AR sampling is involved.
- Observation: The supplement now has a stronger layer-isolation table than the earlier validation-longest-only control.
  Evidence: `local_reports\v13_vqvae_complexity_benchmark_20260705.md` records shortest 8/10 strict-valid, random 6/10, longest 3/10, and most-faces 5/10 under VQ-VAE-only reconstruction.

- Observation: The paper now has a machine-readable criterion for when a future VQ-VAE checkpoint can replace diagnostic figures.
  Evidence: `tools\run_vqvae_slice_benchmark.py` writes `<run-prefix>_benchmark_summary.json`, and the current baseline summary reports `hold_vqvae_checkpoint`.

- Observation: Future positive figure replacement now has a server-copy integrity check.
  Evidence: `tools\write_vqvae_server_handoff.py` writes `copy_back_manifest.json`, and the supplement instructs that the manifest should report `"complete": true` before a rented machine is deleted.

- Observation: Current generated outputs fail the final paper-candidate gate even when STEP hash uniqueness and rendering coverage are high.
  Evidence: `G100/generated_quality_summary.json` reports `hold_for_failure_analysis`: top-two topology fraction is 0.713, strict-valid complex outputs are 0, and the full generated contact sheet is present. `G20/generated_quality_summary.json` also reports `hold_for_failure_analysis`: attempts and strict-valid counts are below threshold, top-two fraction is 0.737, and strict-valid complex outputs are only 2.

- Observation: Curved-surface evidence now has an implementation path but not yet a current result.
  Evidence: `BrepARG/2sequence.py` now preserves `source_path` for newly generated sequence groups, and tests confirm the hook plus manifest curvature propagation. The existing AR120 `sequences_fsq_rcm.pkl` was created before this hook and still lacks source paths, so current PDFs should describe `most_curved` as a future diagnostic until a source-path-aware package is rebuilt and evaluated.

- Observation: The main paper can be made more actionable without exceeding the 7-page target.
  Evidence: The new quality-recovery protocol table was added to `papers/aaai_v13/latex/main.tex`, the PDF still reports 7 pages, and the rendered page check shows the table fits cleanly on page 4.

- Observation: The paper can now cite an explicit audit for why curved-surface evidence is deferred.
  Evidence: `local_reports/v13_ar120_sequence_source_path_audit_20260706.json` records `groups_with_source_path=0` after scanning 5000 train, 5000 validation, and 5000 test groups from the current AR120 package. It records `validation_most_curved_ready=false`.

- Observation: The next curved-surface paper update now has a reproducible sequence-rebuild entry point.
  Evidence: `tools/run_source_path_sequence_rebuild.sh` wraps `tools/run_sharded_sequence.py` and `tools/audit_sequence_source_paths.py`, producing a rebuilt sequence package and `source_path_audit.json` in the same run directory.

## Decision Log

- Decision: Treat `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_best.pt` as the current best model for the 100-sample generation run.
  Rationale: It is the latest verified best checkpoint, at epoch 120, with the lowest recorded validation cross-entropy in the current training sequence.
  Date/Author: 2026-07-05 / Codex.
- Decision: Use the already verified sampling setting `temperature=0.9`, `top_p=0.92`, `max_new_tokens=320`, constrained decoding, and `--seed -1` for the first 100-sample generation run.
  Rationale: The 6-sample pilot at this setting saved 6/6 strict BREP-valid STEP files and had no duplicate STEP hashes. It is the best balance so far between validity and diversity.
  Date/Author: 2026-07-05 / Codex.
- Decision: Make project organization non-destructive at first by creating inventories, manifests, and proposed grouping rules before moving, deleting, or compressing any current files.
  Rationale: The workspace contains long-running experiment outputs and dirty worktree changes. A mistaken cleanup would be expensive; a manifest-first pass creates safety and supports reproducible paper writing.
  Date/Author: 2026-07-05 / Codex.
- Decision: Write the first manuscript as Markdown before moving to the official AAAI LaTeX template.
  Rationale: The project still needs claim/evidence review, baseline decisions, and figure selection. Markdown lets the user revise argument structure quickly; conversion to AAAI LaTeX should happen after the technical content stabilizes.
  Date/Author: 2026-07-05 / Codex.
- Decision: Render selected STEP examples into the paper workspace but keep original STEP files in their reconstruction run directory.
  Rationale: PNG figures are lightweight and easy to inspect in the manuscript workspace, while the canonical STEP outputs remain traceable under `local_runs\reconstruction_eval` with their manifest and report.
  Date/Author: 2026-07-05 / Codex.
- Decision: Organize the project by adding indexes and manifests before physically moving files.
  Rationale: Most storage is in data archives and experiment outputs that are already referenced by existing reports. A root index improves navigation while preserving reproducibility and avoiding broken paths.
  Date/Author: 2026-07-05 / Codex.
- Decision: Keep the current main paper and supplement honest by presenting weak generated images as diagnostic evidence, not as publication-quality positive examples.
  Rationale: The current images are too simple or broken to support a strong generative CAD claim. The defensible contribution today is the reproducible pipeline, retained artifacts, failure taxonomy, and quality-recovery plan.
  Date/Author: 2026-07-05 / Codex.

## Outcomes & Retrospective

This plan is newly started. The previous AR training goal is complete and provides the best checkpoint. The next concrete milestone is to generate 100 random STEP files from the epoch-120 best model and summarize validity, failure causes, topology distribution, sequence lengths, retained STEP count, and hash diversity.

The first milestone is now complete. The 100-sample generation run produced a retained artifact directory and a human-readable statistics report. The next milestone is file organization: create an inventory and safe cleanup proposal before touching any generated or source files.

The file organization milestone has a first non-destructive pass. The project inventory report recommends keeping canonical heavy artifacts in place, using `local_reports` for lightweight paper evidence, and delaying destructive cleanup until the manuscript evidence map is complete.

The AAAI preparation and first-draft milestone now has initial artifacts: an official-requirements brief, a related-work positioning map, an evidence map, and a full Markdown manuscript draft. The draft is useful for revision, but it still needs baseline experiments, qualitative figure selection, and conversion to the official AAAI template before submission.

The paper workspace now includes first-pass rendered figures, experiment summary tables, a reproducibility checklist draft, and a supplementary material plan. The remaining submission-quality gaps are baseline/ablation evidence, final figure selection, and conversion to the official AAAI LaTeX format.

The conversion milestone is now complete for a first LaTeX draft. `papers\aaai_v13\latex\main.pdf` compiles under the AAAI-27 anonymous-submission style and visually renders as a 4-page draft. The main remaining paper gaps are still scientific rather than formatting: direct baselines, ablations, larger generation metrics, and stronger qualitative selection.

The diagnostic-paper milestone is now current. `papers\aaai_v13\latex\main.pdf` is a 7-page AAAI-style main paper, and `papers\aaai_v13\latex\supplement.pdf` is a 13-page supplement after adding the implemented recovery controls. Both compile locally and were rendered to PNG for visual checking. The paper is now more honest and usable, but it remains a diagnostic/reproducibility draft until a retrained VQ-VAE and longer-sequence AR run produce stronger positive CAD examples.

The supplement now includes a VQ-VAE complexity-sliced benchmark, so the current paper evidence better supports the claim that complex reconstruction fails before AR sampling. The next scientific gap is no longer diagnosis alone; it is producing a new VQ-VAE checkpoint that improves the long and most-face validation slices.

The paper workspace now also records a promotion gate for future checkpoint updates. The current VQ100 summary intentionally says `hold_vqvae_checkpoint`; future positive figures should not replace the diagnostic contact sheets until a new candidate summary says `promote_for_ar_rebuild` and the rendered outputs are visually credible.

The rented-server handoff language is now stronger. The supplement no longer only says to copy back files; it points to an executable manifest generator that verifies the checkpoint, training history, ledger, benchmark reports, manifests, and contact sheets are all present.

The final generated-output promotion language is also stricter. A future generated contact sheet should not replace the current diagnostic figures unless `generated_quality_summary.json` says `promote_as_paper_candidates` and visual inspection confirms nontrivial CAD quality.

The paper notes now distinguish current complex-slice evidence from future curved-surface evidence. The code can plan an optional `most_curved` VQ-VAE diagnostic, but the current AR120 sequence package does not contain source paths, so the current manuscript should not claim a completed curved-slice result until a rebuilt source-path-aware sequence package is evaluated.

## Context and Orientation

The repository root is `D:\luolin\V13`. The repository contains the original `BrepARG` code, local improvements under `breparg_improvements`, helper tools under `tools`, tests under `tests`, plans under `plans`, and generated outputs under `local_runs`, `ABC`, `processed_local`, and `breparg_improvements\repro_outputs`.

The best AR checkpoint is:

    D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_best.pt

The matching AR sequence package is:

    D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\sequences_fsq_rcm.pkl

The VQ-VAE checkpoint used to decode generated tokens into geometry is:

    D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\fsq_vqvae_best.pt

The reconstruction evaluator is:

    D:\luolin\V13\tools\evaluate_reconstruction_v13.py

In this project, STEP files are CAD exchange files written by the reconstruction pipeline. BREP validity means the OpenCascade-based validation path accepted the reconstructed boundary representation as valid enough for downstream visualization.

## Plan of Work

First, generate a 100-sample reconstruction run from the epoch-120 best checkpoint. The run should use fresh runtime randomness by omitting `--seed`, which defaults to `-1`, and should retain STEP files under `local_runs\reconstruction_eval`. After the run, compute a statistics summary from `reconstruction_report.json`, `reconstruction_manifest.jsonl`, and the retained STEP file hashes.

Second, create a project inventory that classifies files into source code, plans, reports, training inputs, model checkpoints, reconstruction outputs, raw or processed datasets, caches, and unknowns. The first pass must not move or delete files. Any later cleanup should preserve paths that current reports and plans depend on, or write a relocation manifest.

Third, gather current AAAI submission requirements from official sources and record the relevant constraints for manuscript length, formatting, anonymity, supplementary material, reproducibility checklist or artifact expectations, and deadline timing. Because conference requirements change, this must be verified from official AAAI pages rather than memory.

Fourth, prepare paper materials: problem statement, method summary, experiment table, qualitative generation examples, limitations, related-work notes, and a claim-evidence map. The paper draft should not overclaim; any missing experiment required for an AAAI-quality submission should be explicitly marked.

Fifth, write the initial paper draft in a dedicated paper directory. The draft should be coherent enough to revise, with sections for abstract, introduction, related work, method, experiments, results, limitations, and conclusion.

## Concrete Steps

Run all commands from `D:\luolin\V13` in PowerShell.

The 100-sample generation command is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\evaluate_reconstruction_v13.py `
      --sequence 'D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\sequences_fsq_rcm.pkl' `
      --ar-checkpoint 'D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_best.pt' `
      --vqvae-checkpoint 'D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\fsq_vqvae_best.pt' `
      --source generated `
      --max-samples 100 `
      --device cpu `
      --constrained-decoding `
      --max-new-tokens 320 `
      --temperature 0.9 `
      --top-p 0.92 `
      --write-step `
      --validate-step `
      --run-name 'eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_<timestamp>'

Expected result is a JSON report with `status` equal to `VERIFIED` if at least one STEP is saved, and a `steps` directory containing retained `.step` files. The stricter goal is to record how many of the 100 attempts are grammar-valid, reconstructed, saved as STEP, and BREP-valid.

## Validation and Acceptance

This goal is accepted only when all four user requirements are satisfied. File organization acceptance requires a current inventory and any executed cleanup to be reversible or documented. The 100-sample generation acceptance requires a retained output directory with `reconstruction_report.json`, `reconstruction_manifest.jsonl`, retained STEP files, and a statistics summary. AAAI preparation acceptance requires current official submission constraints, a related-work/positioning brief, and an experiment evidence map. Paper draft acceptance requires a saved manuscript draft that cites or references the project evidence without claiming unverified results.

## Idempotence and Recovery

The generation run is additive because it writes a new timestamped directory under `local_runs\reconstruction_eval`. If it fails partway, keep the partial directory for diagnosis and rerun with a new name. Project organization starts with inventory only; no destructive cleanup should occur without a manifest and recovery plan. Paper drafts and briefs should be saved as new files rather than overwriting experiment reports.

## Artifacts and Notes

Current best model evidence:

    best AR checkpoint: D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_best.pt
    best epoch: 120
    best validation CE: 0.29493329663972306
    preferred pilot reconstruction: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_002743
    100-sample reconstruction: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_005342
    100-sample statistics report: D:\luolin\V13\local_reports\generated100_lr5e6_epoch120_best_20260705.md
    project organization report: D:\luolin\V13\local_reports\project_inventory_and_organization_20260705.md
    AAAI preparation brief: D:\luolin\V13\local_reports\aaai27_preparation_brief_20260705.md
    paper workspace: D:\luolin\V13\papers\aaai_v13
    project index: D:\luolin\V13\PROJECT_INDEX.md
    first manuscript draft: D:\luolin\V13\papers\aaai_v13\draft_aaai27_v13_cad_generation.md
    paper evidence map: D:\luolin\V13\papers\aaai_v13\evidence_map.md
    selected STEP examples: D:\luolin\V13\papers\aaai_v13\selected_step_examples.md
    rendered STEP contact sheet: D:\luolin\V13\papers\aaai_v13\figures\step_renders\selected_step_contact_sheet.png
    render manifest: D:\luolin\V13\papers\aaai_v13\figures\step_renders\render_manifest.json
    experiment tables: D:\luolin\V13\papers\aaai_v13\tables\experiment_summary.md
    reproducibility checklist draft: D:\luolin\V13\papers\aaai_v13\reproducibility_checklist_draft.md
    supplementary material plan: D:\luolin\V13\papers\aaai_v13\supplementary_material_plan.md
    supplement staging directory: D:\luolin\V13\papers\aaai_v13\supplement_staging
    supplement zip draft: D:\luolin\V13\papers\aaai_v13\aaai_v13_supplement_staging_20260705.zip
    AAAI LaTeX draft: D:\luolin\V13\papers\aaai_v13\latex\main.tex
    AAAI PDF draft: D:\luolin\V13\papers\aaai_v13\latex\main.pdf
    AAAI supplement LaTeX: D:\luolin\V13\papers\aaai_v13\latex\supplement.tex
    AAAI supplement PDF: D:\luolin\V13\papers\aaai_v13\latex\supplement.pdf
    rendered PDF page checks: D:\luolin\V13\papers\aaai_v13\latex\rendered
    latest diagnostic generation: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated20_lr5e6_epoch120_best_temp095_topp95_max512_random_cpu_20260705_diag
    VQ-VAE validation-longest diagnostic: D:\luolin\V13\local_runs\reconstruction_eval\eval_validation_longest10_vqvae_epoch100_cpu_20260705_diag
    VQ-VAE complexity-sliced benchmark report: D:\luolin\V13\local_reports\v13_vqvae_complexity_benchmark_20260705.md
    VQ-VAE baseline promotion summary: D:\luolin\V13\local_runs\reconstruction_eval\vqvae_epoch100_complexity_benchmark_20260705_benchmark_summary.json
    VQ-VAE server handoff helper: D:\luolin\V13\tools\write_vqvae_server_handoff.py
    G100 generated quality summary: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_005342\generated_quality_summary.json
    G20 generated quality summary: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated20_lr5e6_epoch120_best_temp095_topp95_max512_random_cpu_20260705_diag\generated_quality_summary.json

Revision note 2026-07-05 00:48 +08:00: Created this plan after the user expanded the goal from training/reconstruction monitoring to project organization, 100-sample generation, AAAI preparation, and paper drafting.

Revision note 2026-07-05 00:58 +08:00: Recorded the completed 100-sample random generation run and statistics summary. The run retained 87 unique STEP files, with 78 strict BREP-valid results and a clear limitation that common simple topologies dominate the sample.

Revision note 2026-07-05 01:02 +08:00: Recorded the non-destructive project inventory and organization proposal. The recommendation is to preserve canonical heavy artifacts in place, create lightweight paper evidence maps, and avoid destructive cleanup until manuscript evidence needs are clear.

Revision note 2026-07-05 01:13 +08:00: Recorded the AAAI-27 preparation brief, paper workspace, evidence map, and first Markdown manuscript draft. The draft is intentionally conservative: it claims only verified training and 100-sample generation evidence, and lists missing baseline comparisons, figure selection, and AAAI LaTeX conversion as remaining work.

Revision note 2026-07-05 01:17 +08:00: Added a selected STEP examples list for future figure work, covering BREP-valid topology representatives, saved STEP files that failed strict validation, and grammar-failed truncation examples.

Revision note 2026-07-05 01:30 +08:00: Added reproducible STEP rendering and publication-preparation materials. The paper workspace now has rendered qualitative figure candidates, experiment tables, a reproducibility checklist draft, and a supplementary material plan, all linked from the manuscript draft and evidence map.

Revision note 2026-07-05 01:33 +08:00: Added `PROJECT_INDEX.md` as a non-destructive organization layer for the V13 workspace. The index links canonical model artifacts, generated STEP evidence, paper materials, and cleanup cautions.

Revision note 2026-07-05 01:37 +08:00: Created a lightweight supplement staging package and zip draft with selected STEP examples, reports, AR history, and rendered figures. The zip was inspected and contains only small review-facing artifacts.

Revision note 2026-07-05 16:14 +08:00: Added the first official-style AAAI LaTeX draft in `papers\aaai_v13\latex`. `latexmk` is unavailable because MiKTeX lacks Perl, so the verified compile path is `pdflatex main.tex`, `bibtex main`, `pdflatex main.tex`, `pdflatex main.tex`. The resulting `main.pdf` has 4 Letter pages, no undefined references/citations, no overfull boxes, and rendered PNG page checks show readable text, tables, figure, and references. Also added a 20-sample diagnostic generation with longer `max_new_tokens=512`; it reduced END truncation compared with the 100-sample 320-token run but still showed strict BREP failures and topology concentration.

Revision note 2026-07-05 20:52 +08:00: Reworked the manuscript into a diagnostic/reproducibility paper after visual inspection showed the current generated examples are too weak for positive claims. The latest `main.pdf` has 7 pages, and `supplement.pdf` now has 13 pages after the recovery-controls update. Both were compiled with MiKTeX and rendered to PNG pages for inspection. The supplement includes artifact aliases, reproduction recipes, generated/VQ-VAE diagnostics, sequence-length evidence, a server experiment schedule, implemented recovery controls, and copy-back verification guidance.

Revision note 2026-07-05 21:20 +08:00: Added complexity-sliced VQ-VAE evidence to the supplement and reports. The new table compares shortest, random, longest, and most-face validation reconstructions under VQ-VAE-only decoding, making the current paper's root-cause argument more concrete.

Revision note 2026-07-05 22:35 +08:00: Added the benchmark-summary promotion gate to the supplement and paper workspace notes. This keeps future paper updates tied to a reproducible JSON decision and rendered contact sheets rather than subjective selection after server training.

Revision note 2026-07-05 22:51 +08:00: Added server handoff manifest language to the supplement and paper workspace notes. Future positive figures should only be promoted from returned server runs whose manifest is complete and whose benchmark summary passes the VQ-VAE gate.

Revision note 2026-07-05 23:01 +08:00: Added generated-run paper-gate summaries for G20 and G100. This makes the paper-positioning decision executable: current generated figures stay diagnostic until a future generated run passes diversity, complexity, validity, uniqueness, and render checks.

Revision note 2026-07-05 23:17 +08:00: Rendered the full G100 contact sheet and refreshed the generated-run gate. The supplement and README now state that G100 is held because of topology collapse and absent complex strict-valid outputs, while the render artifact is complete.

Revision note 2026-07-05 23:39 +08:00: Added optional curved-surface diagnostic tooling to the paper plan. This is a reproducibility improvement for the next server run, not a new current result, because existing sequences do not include the metadata needed to rank validation rows by source-geometry curvature.

Revision note 2026-07-06 00:00 +08:00: Documented the source-path preservation hook for future sequence rebuilds. The paper plan now distinguishes three states: current generated outputs remain diagnostic failures, current AR120 sequences still lack source-path metadata, and future rebuilt sequences can support audited `most_curved` VQ-VAE evidence.

Revision note 2026-07-06 00:23 +08:00: Expanded the main paper's quality-improvement argument by adding a compact recovery protocol table. This does not claim improved generation quality; it makes the restart order and promotion gates explicit for the next training phase.

Revision note 2026-07-06 00:36 +08:00: Added the sequence source-path audit result to the paper plan. This strengthens the manuscript's distinction between current complexity evidence and future curved-surface evidence.

Revision note 2026-07-06 00:52 +08:00: Added the source-path-aware sequence rebuild launcher to the paper plan. Future positive paper updates should cite the launcher output and audit report before claiming curved-surface diagnostics.
