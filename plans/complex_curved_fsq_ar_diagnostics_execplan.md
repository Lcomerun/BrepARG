# Complex Curved FSQ and AR Diagnostics

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document follows `PLANS.md` from the repository root. It is self-contained so a future agent can continue the experiment without relying on prior conversation.

## Purpose / Big Picture

The current generated CAD samples are often valid but visually too simple. Generation-time constraints and quality gates can filter outputs, but they do not explain whether the weakness comes from FSQ geometry quantization, AR token modeling, ordering, or long-sequence coverage. After this plan, the user can run a repeatable diagnostic suite on a complex curved subset and inspect separate evidence for FSQ-only reconstruction error, real-token reconstruction success, AR teacher-forcing loss, and failure rates by face count and sequence length.

The first demonstrable behavior is a new local working folder under `local_runs/complex_curved_diagnostics_20260715` containing JSON, Markdown, JSONL manifests, and optional STEP/STL outputs for a small smoke subset. Later milestones extend this to full 50-sample and larger evaluations, FSQ capacity retraining, DFS ordering control, and an official or same-protocol BrepARG baseline.

## Progress

- [x] (2026-07-15 00:20 +08:00) Read `PLANS.md`, inspected the current `ubuntu` artifacts, and confirmed that `sequences_fsq_rcm.pkl` carries server-style `parsed-shard://...!/abc_XXXX/file.pkl` source metadata while the local machine has the matching zip archives in `ABC/processed/abc_parsed_full_archives`.
- [x] (2026-07-15 00:32 +08:00) Verified current reusable tools: `tools/evaluate_reconstruction_v13.py` can reconstruct true token sequences and load FSQ checkpoints; `tools/diagnose_vqvae_buckets.py` computes patch MSE but cannot directly read server parsed-shard URIs from the sequence package.
- [x] (2026-07-15 00:38 +08:00) Searched current public sources for BrepARG baseline weights. The official GitHub repository `123qiang06/BrepARG` links pretrained weights hosted on Hugging Face under `qingtiannihao/BrepARG`, including ABC AR and ABC SE-VQ-VAE checkpoints.
- [x] (2026-07-15 00:45 +08:00) Added `tools/complex_curved_diagnostics.py` and `tests/test_complex_curved_diagnostics.py`. The script resolves local zip archives from sequence metadata, selects complex curved records, computes FSQ MSE/Chamfer, computes AR teacher-forcing CE, and optionally reconstructs true token sequences.
- [x] (2026-07-15 00:47 +08:00) Ran syntax and focused unit tests. `python -m py_compile tools\complex_curved_diagnostics.py` passed, and `python -m unittest tests.test_complex_curved_diagnostics` ran 4 tests successfully.
- [x] (2026-07-15 00:50 +08:00) Ran a 3-sample smoke diagnostic under `local_runs/complex_curved_diagnostics_20260715\smoke_p95`. It selected 3 records, evaluated 107 patches, and wrote 2 STEP files from true token reconstruction.
- [x] (2026-07-15 00:56 +08:00) Ran a 50-sample metrics-only diagnostic under `local_runs/complex_curved_diagnostics_20260715\full_50_metrics`. It selected 50 records from 5,000 validation candidates and evaluated 3,399 patches without writing STEP files.
- [x] (2026-07-15 01:02 +08:00) Ran a 50-sample true-token reconstruction diagnostic under `local_runs/complex_curved_diagnostics_20260715\full_50_reconstruct`. It selected the same style of complex curved subset, saved 27 STEP files, and recorded reconstruction failures in JSONL without crashing.
- [x] (2026-07-15 00:41 +08:00) Started a fresh BrepARG original-generation-logic comparison run under `local_runs\breparg_logic_compare_20260715\breparg_original_logic_100_20260715`. The process id is `37044`. It uses `BrepARG.generate_brep.generate_sequence` with the current V13 `ubuntu` AR and FSQ checkpoints, so this isolates generation logic while keeping model weights constant.
- [x] (2026-07-15 00:56 +08:00) Completed the fresh BrepARG original-generation-logic comparison and post-run STEP quality check. The run saved 100 PNG previews from 124 attempts in 8.146 minutes; quality check found 100/100 STEP-readable, 94/100 strict BRep valid, 100/100 closed shell, and 11/100 complex by the 12-face-or-20-edge rule.
- [x] (2026-07-15 00:58 +08:00) Compared the fresh BrepARG-logic run against the previous quality-gated run in `local_runs\breparg_logic_compare_20260715\comparison_summary.md`.
- [x] (2026-07-15 03:13 +08:00) Ran a new user-requested BrepARG original-generation-logic rerun under `local_runs\breparg_logic_compare_20260715\breparg_original_logic_100_20260715_rerun`. It produced 100 PNG previews and 103 STEP files from 124 attempts in 7.438 minutes. Strict quality audit found 103/103 STEP-readable, 94 BRep-valid, 103 closed shell, 11 complex by the 12-face-or-20-edge rule, and 9 complex+BRep-valid+closed.
- [x] (2026-07-15 01:05 +08:00) Added `--skip-ar` and `--skip-reconstruction` to `tools/complex_curved_diagnostics.py`, so new FSQ capacity checkpoints can be evaluated on the same complex curved subset without trying incompatible old AR tokens or FSQ/OCC reconstruction.
- [x] (2026-07-15 01:08 +08:00) Added `--ordering rcm|dfs` to `tools/run_sharded_sequence.py`, preserved RCM as the default, recorded `ordering` in sequence metadata, and made sequence shard merging reject mixed DFS/RCM packages.
- [x] (2026-07-15 01:28 +08:00) Added `tools/prepare_complex_curved_control_workspace.py` and `tests/test_prepare_complex_curved_control_workspace.py`. The script creates a portable workspace under `local_runs/complex_curved_control_suite_20260715` with README, JSON config, and PowerShell entrypoints for the current FSQ/AR diagnostic, FSQ-capacity candidate diagnostic, DFS/RCM sequence rebuild, official BrepARG baseline, and report summarization.
- [x] (2026-07-15 01:31 +08:00) Generated `local_runs/complex_curved_control_suite_20260715` and ran a 1-sample smoke diagnostic in `experiments/00_smoke_one`. The smoke report is `VERIFIED` and proves the new workspace can read the current sequence package, selected FSQ checkpoint, selected AR checkpoint, and local parsed zip archives.
- [x] (2026-07-15 01:47 +08:00) Ran the portable 50-sample current-method diagnostic via `scripts/00_current_fsq_ar_teacher_reconstruction.ps1`. The report under `experiments/00_current_fsq_ar_teacher_reconstruction` is `VERIFIED`, selected 50 shapes, evaluated 3,399 FSQ patches, wrote 27 STEP files from true tokens, and found 9 strict BRep-valid reconstructions.
- [x] (2026-07-15 01:49 +08:00) Generated `local_runs/complex_curved_control_suite_20260715/complex_curved_diagnostic_summary.md` with `scripts/04_summarize_reports.ps1`.
- [x] (2026-07-15 01:57 +08:00) Added `scripts/01a_train_fsq_capacity_candidate.ps1` to the portable workspace generator. It launches the first FSQ-capacity variable test by setting `NS_LEVELS=16,16,8,8`, keeping the rest of the VQ-VAE architecture fixed, and writing the candidate under the suite's `experiments/01a_train_fsq_capacity_candidate` folder.
- [x] (2026-07-15 02:08 +08:00) Added `tools/analyze_reconstruction_fsq_correlation.py` and `tests/test_analyze_reconstruction_fsq_correlation.py`, then generated `experiments/00_current_fsq_ar_teacher_reconstruction/reconstruction_fsq_correlation.json` and `.md`. `scripts/04_summarize_reports.ps1` now regenerates this correlation report automatically when the current diagnostic manifest and FSQ patch metrics exist.
- [x] (2026-07-15 02:18 +08:00) Added `tools/audit_breparg_baseline_outputs.py` and `tests/test_audit_breparg_baseline_outputs.py`. The tool normalizes upstream BrepARG output directories into JSON, Markdown, and JSONL reports using the same face/edge complexity vocabulary as V13 generation reports.
- [x] (2026-07-15 02:20 +08:00) Updated `scripts/03_breparg_official_baseline.ps1` in the portable workspace generator so an official-weight BrepARG smoke run is immediately audited into `breparg_baseline_quality_summary.json`, `.md`, and `breparg_baseline_quality_manifest.jsonl`.
- [x] (2026-07-15 02:27 +08:00) Ran the new baseline audit on the existing same-weight BrepARG original-logic comparison folder. The normalized report under `local_runs\breparg_logic_compare_20260715\breparg_original_logic_100_20260715` found 103 STEP files, 100 quality-checked retained rows, 94 BRep-valid rows, 103 closed solid STEP files, 11 complex-by-entity rows, and 9 complex+BRep-valid+closed rows.
- [x] (2026-07-15 02:44 +08:00) Extended `tools/prepare_complex_curved_control_workspace.py` with `scripts/02b_train_dfs_rcm_ar.ps1` and `scripts/02c_eval_dfs_rcm_ar_complex_curved.ps1`. The ordering experiment now has entrypoints to train matched DFS/RCM AR branches with identical hyperparameters and evaluate their teacher-forcing CE on the same complex-curved subset.
- [x] (2026-07-15 02:45 +08:00) Regenerated `local_runs\complex_curved_control_suite_20260715`; it now contains eight scripts. PowerShell parser checks pass for all eight scripts, and focused unittest plus py_compile validation pass for the modified generator.
- [x] (2026-07-15 02:57 +08:00) Added `scripts/03b_breparg_same_data_training_fallback.ps1` to the portable workspace. This fallback documents and launches original BrepARG SE VQ-VAE training, original DFS sequence building, original BrepARG AR training, generation, and shared baseline auditing on the same-data split when official weights cannot be used.
- [x] (2026-07-15 02:58 +08:00) Regenerated `local_runs\complex_curved_control_suite_20260715`; it now contains nine scripts. PowerShell parser checks pass for all nine scripts, and the focused workspace-generator test verifies the fallback references `train_vqvae.py`, `2sequence.py`, `train_ar.py`, `generate_brep.py`, and `tools\audit_breparg_baseline_outputs.py`.
- [x] (2026-07-15 03:06 +08:00) Added `tools/audit_complex_curved_control_suite.py` and `tests/test_audit_complex_curved_control_suite.py`, then wired `scripts/05_audit_suite_status.ps1` into the portable workspace. This creates `suite_status.json` and `.md` with completed/missing experiment artifacts and next recommended commands.
- [x] (2026-07-15 03:08 +08:00) Ran `scripts/05_audit_suite_status.ps1` against the current workspace. The status report shows `completed: 1`, `missing: 7`; the completed experiment is the current-method complex-curved diagnostic, and the first next action is `01a_train_fsq_capacity_candidate.ps1`.
- [x] (2026-07-15 02:45 +08:00) Prepared DFS-ordering control launch commands so sequence rebuild, matched AR training, and complex-curved teacher-forcing evaluation can be run independently in the portable workspace.
- [x] (2026-07-15 03:20 +08:00) Created a new temporary local root-cause workspace requested by the user under `local_runs\complex_curved_rootcause_suite_20260715` with portable scripts and experiment folders for FSQ-only metrics, teacher-forcing reconstruction, FSQ capacity, DFS/RCM, and BrepARG baselines.
- [x] (2026-07-15 03:25 +08:00) Ran a pure FSQ-only complex-curved diagnostic in the new workspace at `experiments\00_fsq_only_patch_metrics` with `--skip-ar --skip-reconstruction`. It selected 50 shapes, evaluated 3,399 patches, and reproduced the FSQ heavy-tail evidence: all-patch Chamfer p95 `0.1501216799`, surface Chamfer p95 `0.4123849943`, edge Chamfer p95 `0.1016861200`.
- [x] (2026-07-15 03:49 +08:00) Ran the teacher-forcing and true-token reconstruction diagnostic in the new workspace at `experiments\01_teacher_forcing_true_token_reconstruction`. It selected the same 50 shapes, measured AR token-weighted CE `0.7467434714`, saved 27/50 STEP files from true tokens, and found 9/50 strict BRep-valid reconstructions. `reconstruction_fsq_correlation.json` shows BRep-valid shape Chamfer-p95 median `0.0564963926` versus reconstruct-failed median `0.1200256310`.
- [x] (2026-07-15 03:55 +08:00) Added `local_runs\complex_curved_rootcause_suite_20260715\rootcause_readout_20260715.md` so the new workspace is self-describing before it is moved to the external SSD.
- [x] (2026-07-15 04:35 +08:00) Verified the FSQ-capacity data-preparation path on local archives. `abc_0000_parsed.zip` was converted to `parsed_shards_smoke\parsed_abc_0000.pkl.gz`, then to `vq_patch_shards_smoke` with 4 patch shards and 303,431 patches. Added smoke/full patch-shard build scripts and separated the smoke trainer from the real capacity launcher.
- [x] (2026-07-15 04:45 +08:00) Ran a one-epoch higher-level FSQ smoke training (`NS_LEVELS=16,16,8,8`, 32,768 samples) and verified that its checkpoint loads in the FSQ-only complex-curved diagnostic. The 5-shape smoke eval under `experiments\01_fsq_capacity_candidate_smoke_eval` is `VERIFIED`; it is a wiring check only, not the capacity result.
- [x] (2026-07-15 05:05 +08:00) Added and tested `tools\build_vqvae_patch_shards_from_archives.py`, a disk-safe direct archive-to-patch builder. On `abc_0000`, it matched the older two-step smoke counts exactly: 5,943 sources, 1,724 skipped by cap, 303,431 patches, 87,694 surfaces, 215,737 edges, and 4 patch shards. Updated rootcause full/smoke patch-shard scripts to use the direct builder.
- [x] (2026-07-15 05:05 +08:00) Ran a 10-archive medium patch-shard build for chunks `0-9`. It produced 3,377,590 patches in 34 shards, used 3.49 GB, had zero source failures, and took about 7.2 minutes. This estimates full `0-99` patch-shard build at roughly 35 GB and 70-80 minutes.
- [x] (2026-07-15 06:40 +08:00) Completed the full direct archive-to-patch build for chunks `0-99`. It produced 34,393,215 patches in 344 shards, used 35.34 GB, saw 681,406 source records, failed 458 sources, skipped 147,621 by the 50-face/150-edge cap, and sampled shard reads verified the `v13.vq_patch_shard.v1` format.
- [x] (2026-07-15 07:15 +08:00) Started the formal FSQ capacity training on the local RTX 3060 12GB, observed that full patch-shard sampling for 450k samples remained CPU-bound for about 27 minutes with no GPU training yet, then stopped it intentionally. Added a Linux server launcher so the completed full patch shards can be trained on the intended faster environment.
- [x] (2026-07-15 07:53 +08:00) Downloaded official BrepARG ABC weights from `qingtiannihao/BrepARG` and tested the local official-baseline path. The files downloaded, but `abc_ar.pt` has embedding shape `7222 x 256` while the repository ABC config builds vocab size `10294`; an inferred `se_codebook_size=5120` smoke produced no STEP outputs within the local 20-minute smoke window. Recorded this under `experiments/03_breparg_official_baseline/official_baseline_incompatibility_report.*`, hardened the baseline launcher against dependency/PATH/exit-code pitfalls, and updated the suite audit so official baseline is `partial` with `INCOMPATIBLE` evidence while the next baseline action is the same-data fallback.
- [x] (2026-07-15 08:25 +08:00) Ran the user-requested BrepARG original-generation-logic comparison in a fresh folder `local_runs\breparg_logic_compare_20260715\breparg_original_logic_100_user_compare_20260715`. The run used `BrepARG.generate_brep.generate_sequence` with current V13 `ubuntu` weights, retained 100 PNG outputs from 124 attempts, saved 103 STEP files, and the shared audit found 94 BRep-valid, 103 closed-solid, 11 complex-by-entity, and 9 complex+BRep-valid+closed outputs. This exactly reproduced the earlier rerun and supports the conclusion that original BrepARG sampling improves legality but does not solve simple-shape collapse under the current FSQ/AR stack.
- [x] (2026-07-15 08:35 +08:00) Added the same-data BrepARG input preparation path. `tools\prepare_breparg_same_data_inputs.py` materializes original BrepARG-readable `same_data_split.pkl`, `deduplicated_surface_source.pkl`, and `deduplicated_edge_source.pkl` from the current V13 sequence provenance and local parsed zip archives. The rootcause workspace now writes `scripts\03a_prepare_breparg_same_data_inputs.ps1`, and `scripts\03b_breparg_same_data_training_fallback.ps1` defaults to those prepared `data\` files instead of placeholder paths.
- [x] (2026-07-15 08:36 +08:00) Ran a tiny real-data smoke for same-data BrepARG inputs under `local_runs\complex_curved_rootcause_suite_20260715\experiments\03b_breparg_same_data_training_fallback\data_smoke`. It wrote train/val/test counts `10/3/3`, produced 160 surface patches and 334 edge patches, opened materialized parsed pickle files successfully, and loaded the deduplicated surface/edge source arrays. This proves the current sequence provenance can resolve zip archive members into original BrepARG training inputs.
- [x] (2026-07-15 08:37 +08:00) Updated `tools\audit_complex_curved_control_suite.py` so suite status now tracks `breparg_same_data_inputs` separately from `breparg_same_data_fallback`. The current rootcause suite status is `total=10`, `completed=2`, `partial=1`, `missing=7`; the official BrepARG baseline is partial due to `INCOMPATIBLE`, and the next baseline actions include `03a_prepare_breparg_same_data_inputs.ps1` followed by `03b_breparg_same_data_training_fallback.ps1`.
- [x] (2026-07-15 08:43 +08:00) Adjusted the same-data BrepARG input scripts to default to a disk-safer medium pool (`10000/1000/1000` train/val/test, capped at 300k surface patches and 500k edge patches) and added `03a_prepare_breparg_same_data_inputs_full.ps1` for the larger `50000/5000/5000` pool intended for the external SSD.
- [x] (2026-07-15 08:44 +08:00) Prepared the medium same-data BrepARG input pool under `local_runs\complex_curved_rootcause_suite_20260715\experiments\03b_breparg_same_data_training_fallback\data`. The summary is `VERIFIED`, with train/val/test written `10000/1000/1000`, surface patches `176605`, edge patches `429555`, zero skipped records, and about 8.0 GB of files. The generated split pickle and SE source pickles were opened successfully, and suite status now reports `completed=3`, `partial=1`, `missing=6`.
- [x] (2026-07-15 09:05 +08:00) Hardened the same-data BrepARG fallback launcher before any long training run. `03b_breparg_same_data_training_fallback.ps1` now has a native-command fail-fast wrapper, checks required Python modules, installs `tensorboard` if missing, and passes explicit `--tb_log_dir` values to both BrepARG VQ-VAE and AR training. Fresh verification confirmed `BrepARG\train_vqvae.py --help` and `BrepARG\train_ar.py --help` reach argparse, all generated PowerShell scripts parse, focused tests pass, and the suite audit remains `completed=3`, `partial=1`, `missing=6`.
- [x] (2026-07-15 09:25 +08:00) Added generation-time quality gating to `tools\generate_breparg_logic_step_png.py` and fixed `BrepARG\generate_brep.py` so batch mode actually forwards `--max_attempts`. The gated BrepARG-logic runner can now validate each candidate STEP in a subprocess, require BRep validity, closed watertight solid, and PNG preview, then copy only accepted samples into `accepted\steps`, `accepted\stl`, and `accepted\png` while preserving the full attempt manifest. Focused py_compile and unit tests passed.
- [x] (2026-07-15 10:44 +08:00) Verified the same-data BrepARG fallback smoke pipeline end-to-end after fixing bounded generation. The smoke script trained a 1-epoch original BrepARG SE VQ-VAE on `data_smoke`, built `sequence_smoke\breparg_same_data_smoke_sequences.pkl`, trained a 1-epoch original BrepARG AR model, ran generation with `--max_attempts 20`, stopped cleanly with `total_attempts: 20` and `saved_count: 0`, then wrote `breparg_same_data_smoke_quality_summary.json` and `same_data_breparg_fallback_smoke_manifest.json`. This proves the fallback baseline wiring is bounded and auditable; the zero STEP result is expected for the intentionally tiny 1-epoch smoke model.
- [x] (2026-07-15 10:56 +08:00) Added an external-SSD migration path for the root-cause suite. `tools\prepare_rootcause_ssd_migration.py` can dry-run or execute a non-destructive copy of the suite, existing experiment artifacts, and optional reference models / parsed archives. The regenerated workspace now includes `scripts\06_prepare_external_ssd_migration.ps1`. A dry-run to `E:\V13_rootcause_20260715` with `--copy-reference-models` found the source suite ready to execute, estimated `experiments` at about 56.0 GB plus reference inputs, and wrote `ssd_migration_plan.json` plus `ssd_migration_commands.md`. Focused tests, py_compile, and PowerShell parser checks passed.
- [x] (2026-07-15 11:04 +08:00) Removed a blocker from the DFS-vs-RCM ordering experiment by adding `tools\prepare_v13_same_data_split.py` and wiring `scripts\02a_prepare_v13_same_data_split.ps1` into the root-cause workspace. The tool materializes parsed `.pkl` files from current V13 sequence `source_relpath` / parsed-shard URI metadata and writes a lightweight `split.pkl` for `tools\run_sharded_sequence.py`, without duplicating the large BrepARG SE VQ-VAE surface/edge source arrays. A real-data smoke under `experiments\02_dfs_rcm_ordering\same_data_split_smoke` materialized train/val/test `5/3/3` records with no skips. Focused tests, py_compile, and PowerShell parser checks passed.
- [x] (2026-07-15 13:28 +08:00) Ran a validity-gated BrepARG original-generation-logic comparison inside the root-cause workspace at `experiments\04_breparg_logic_generation_baseline\breparg_logic_validity_gate_100_20260715`. It used `BrepARG.generate_brep.generate_sequence` with the current V13 `ubuntu` sequence/VQ-VAE/AR artifacts, retained 100 accepted STEP/STL/PNG outputs from 133 attempts, and wrote `breparg_logic_report.json`, `accepted_manifest.jsonl`, and `accepted\breparg_logic_contact_sheet.png`.
- [x] (2026-07-15 13:47 +08:00) Strengthened the read-only BrepARG same-data fallback preflight so it records the planned command surfaces for VQ-VAE training, sequence building, AR training, generation, and audit, then checks each required flag against the corresponding CLI `--help` output. The refreshed `breparg_same_data_preflight.json` remains `READY`, with all required CLI args present and `training_started=false`.
- [x] (2026-07-15 14:05 +08:00) Added an optional FSQ VQ-VAE sample cache for capacity training. `breparg_improvements\train.py` now honors `NS_VQ_SAMPLE_CACHE`: if the cache exists it loads cached samples/weights/summary, otherwise it writes a compressed cache after the expensive patch-shard sampling step. The generated capacity training and preflight scripts use `experiments\01a_train_fsq_capacity_candidate\vq_samples_450000_seed0.npz`, and the refreshed preflight reports that the cache is enabled but not yet created.
- [x] (2026-07-15 14:05 +08:00) Added `tools\build_vqvae_sample_cache.py` and `scripts\01a_build_fsq_capacity_sample_cache.ps1` so the expensive 450k FSQ sample cache can be built as a separate CPU/I/O step before GPU VQ-VAE training. The FSQ capacity preflight now points `next_command` to the sample-cache script while the cache is missing, then to training once the cache exists.
- [x] (2026-07-15 14:55 +08:00) Built the reusable FSQ capacity sample cache at `local_runs\complex_curved_rootcause_suite_20260715\experiments\01a_train_fsq_capacity_candidate\vq_samples_450000_seed0.npz`. It contains 450,000 `(3,32,32)` float32 patches, matching weights, sample range `[-1.0, 1.0]`, weight range `[1.0, 2.5]`, and summary values `complex_records_selected=421606`, `loaded_shards=6`, `failed_shards=0`, size `175,329,729` bytes. The refreshed FSQ preflight is `READY` with `sample_cache_exists=true`, and suite next action is now `01a_train_fsq_capacity_candidate.ps1`.
- [x] (2026-07-15 16:52 +08:00) Corrected the FSQ capacity state after checking the current worktree: the formal capacity run is not running and only reached epoch 5/target 180 with no `train_report.json`. Added `tools/check_fsq_capacity_completion.py` Windows PowerShell PID probing, generated `scripts\01a_resume_fsq_capacity_candidate.ps1`, refreshed suite status so the next action is resume rather than fresh train, and updated `requirement_audit_20260715.md`, `fsq_capacity_completion_handoff_20260715.md`, and `rootcause_readout_20260715.md`.
- [x] (2026-07-15 16:52 +08:00) Ran a clearly labeled partial-only FSQ diagnostic at `experiments\01_fsq_capacity_candidate_partial_epoch5` and recorded `partial_epoch5_fsq_capacity_readout_20260715.md`. The partial epoch-5 checkpoint is worse than the current FSQ baseline on the same 50 complex-curved shapes (`Chamfer p95 0.23284` versus `0.15012`, surface p95 `0.54888` versus `0.41238`), which is expected for an interrupted early checkpoint and is not a scientific capacity conclusion. The official capacity result still requires resumed training to completion.
- [ ] Run the FSQ capacity retrain and evaluate the resulting checkpoint on the complex-curved subset.
- [ ] Run the DFS/RCM sequence rebuild, matched AR training, and complex-curved teacher-forcing comparison after the required split/checkpoint paths are staged.
- [x] Download or stage the official BrepARG ABC weights into a baseline workspace and record whether they can be used under the shared protocol.
- [ ] Train a medium-data same-data BrepARG baseline using the prepared `03b_breparg_same_data_training_fallback\data` split and report the decision, because the official ABC weights are incompatible with the current local protocol. If the external SSD is available, optionally prepare the larger `data_full` pool with `scripts\03a_prepare_breparg_same_data_inputs_full.ps1` before a larger baseline run.

## Surprises & Discoveries

- Observation: The current sequence package has enough metadata to map server parsed-shard URIs back to local archive members without extracting all archives. A sample validation row contains `source_relpath` such as `abc_0000/00000140_..._step_003.pkl`.
  Evidence: Inspecting `ABC/processed/train_outputs/ubuntu/sequences_fsq_rcm.pkl` showed `source_path`, `source_relpath`, and `source_shard` fields for `train`, `val`, and `test` records.
- Observation: Existing VQ diagnostics assume `split.pkl` paths point to extracted parsed `.pkl` files, but the authoritative local data is currently stored as zip archives.
  Evidence: `tools/diagnose_vqvae_buckets.py` loads paths through `vqvae_sampling.load_patch_records(Path(source_path))`, while `ABC/processed/abc_parsed_full` is empty and `ABC/processed/abc_parsed_full_archives` contains `abc_0000_parsed.zip` through `abc_0099_parsed.zip`.
- Observation: Official BrepARG weights appear available online, so the baseline should first try official weights before training a local medium-data baseline.
  Evidence: The Hugging Face model card lists `checkpoint/weights/abc_ar.pt` and `checkpoint/weights/abc_vqvae.pt` for ABC, and the official GitHub README links the pretrained weights.
- Observation: On the 50-sample p95-ranked complex curved subset, FSQ reconstruction has a low median but a heavy tail, especially on surfaces.
  Evidence: `full_50_reconstruct\complex_curved_diagnostics_report.json` reports FSQ MSE median `3.479047154542059e-05`, p95 `0.005424340721219778`, max `0.6015003323554993`; surface MSE mean is `0.008634337413986514` versus edge MSE mean `0.002106617011620905`.
- Observation: The AR model also struggles on true complex curved tokens, so generation weakness is not only a free-running sampling issue.
  Evidence: The same 50-sample diagnostic reports AR token-weighted teacher-forcing CE `0.7467434714048675`, with sample CE p95 `1.8077699542045589` and max `2.2479090690612793`, much higher than the approximate best validation CE around `0.299` seen in the previous AR training run.
- Observation: True-token reconstruction through the FSQ/OCC path fails often on complex curved shapes even without AR sampling.
  Evidence: `full_50_reconstruct` attempted 50 grammar-valid real sequences, saved 27 STEP files, but only 9 passed BRep validity. The manifest has 23 `reconstruct_failed` rows, all with `reconstruct_cad_from_sequence returned None`.
- Observation: The current sequence representation is capped at 50 faces and 150 edges by the BrepARG sequence processor.
  Evidence: `BrepARG/2sequence.py` defines `--max_face` default `50` and `--max_edge` default `150`, and `BrepARG/config.json` sets `abc.face_index_size` to `50`. Therefore a 50-face case is a boundary case, while a >50-face source is outside the current AR token design.
- Observation: The local Windows process id `41868` mentioned by the user was not active in the current process table when checked.
  Evidence: `Get-Process -Id 41868` and a WMI command-line scan found no matching active V13 generation process; the new BrepARG-logic comparison process is active as pid `37044`.
- Observation: The original BrepARG generation function with the current V13 weights produces valid closed solids efficiently, but the retained distribution is still simple.
  Evidence: `local_runs\breparg_logic_compare_20260715\breparg_original_logic_100_20260715\quality_check\step_quality_summary.json` reports 100 checked outputs, 94 BRep valid, 100 closed shell, but only 11 complex by 12-face-or-20-edge. Median faces/edges are 6/12 and max faces/edges among the first 100 PNG-retained samples are 16/34.
- Observation: The first standalone FSQ sample-cache build spent a long time in CPU without writing output because the sampler rebuilt `set(exclude_ids)` once per candidate during uniform selection.
  Evidence: The initial cache process accumulated more than 1,000 seconds of CPU with no `.npz` output; code inspection found `record.get("record_id") not in set(exclude_ids or [])` inside the list comprehension. Moving the set construction outside the loop and rerunning built the 450k cache in `0.609` minutes.
- Observation: FSQ capacity checkpoints with changed levels cannot be fairly evaluated with old AR tokens or old token reconstruction.
  Evidence: `breparg_improvements/train.py` reads `NS_LEVELS` into `FSQ_LEVELS`, sets `SE_CODEBOOK` to the product of those levels, and `build_fsq_vqvae()` creates a model whose token vocabulary depends on that product. Therefore a levels change such as `16,16,8,8` or `16,16,16,16` should first be tested with FSQ-only patch MSE/Chamfer; AR and STEP reconstruction require rebuilding sequences and training/evaluating a matching AR model.
- Observation: DFS and RCM sequence packages are now distinguishable and cannot be accidentally merged.
  Evidence: `tools/run_sharded_sequence.py --help` lists `--ordering {rcm,dfs}`, `metadata_from_preprocessor(..., ordering=...)` writes `DFS` or `RCM`, and `breparg_improvements/sequence_sharding.py` includes `ordering` in `METADATA_KEYS`.
- Observation: The official BrepARG Hugging Face model card is currently available and lists ABC AR and ABC VQ-VAE checkpoints.
  Evidence: A 2026-07-15 web search returned `https://huggingface.co/qingtiannihao/BrepARG`, whose model card says the files include `checkpoint/weights/abc_ar.pt` and `checkpoint/weights/abc_vqvae.pt`, and that the ABC VQ-VAE codebook size is 8192.
- Observation: The official BrepARG ABC weights are downloadable but not usable as a completed fair baseline under the current local protocol.
  Evidence: `local_runs\complex_curved_rootcause_suite_20260715\experiments\03_breparg_official_baseline\official_baseline_incompatibility_report.json` records that `abc_ar.pt` has `transformer.wte.weight` shape `[7222, 256]`, while local `BrepARG/config.json` for ABC builds `50 + 8192 + 2048 + 4 = 10294` tokens. Loading with the local config fails on `transformer.wte.weight` and `lm_head.weight`; a smoke attempt with inferred `se_codebook_size=5120` produced no STEP files within the 20-minute local smoke window.
- Observation: The new portable workspace entrypoints are syntactically valid and the diagnostic path works on a tiny subset.
  Evidence: `python -m unittest tests.test_complex_curved_diagnostics tests.test_prepare_complex_curved_control_workspace` passed 6 tests; `python -m py_compile tools\prepare_complex_curved_control_workspace.py tools\complex_curved_diagnostics.py tools\summarize_complex_curved_diagnostics.py tools\run_sharded_sequence.py breparg_improvements\sequence_sharding.py` passed; all generated `scripts\*.ps1` parsed with the PowerShell parser; and `experiments\00_smoke_one\complex_curved_diagnostics_report.json` reports `status: VERIFIED`, `selected_count: 1`, `fsq_patch_metrics`, and `ar_teacher_forcing`.
- Observation: The portable 50-sample current-method run reproduces the earlier bottleneck evidence inside the new workspace.
  Evidence: `experiments\00_current_fsq_ar_teacher_reconstruction\complex_curved_diagnostics_report.json` reports `status: VERIFIED`, `selected_count: 50`, `elapsed_min: 5.303`, `fsq_patch_metrics.patch_count: 3399`, FSQ MSE median `3.47904715454206e-05`, MSE p95 `0.005424340721219778`, Chamfer median `0.015260953456163406`, Chamfer p95 `0.15012167990207662`, AR token-weighted CE `0.7467434714048675`, true-token reconstruction `step_saved: 27/50`, and strict `brep_valid: 9/50`.
- Observation: Reconstruction failures are not confined to a single bucket.
  Evidence: `teacher_reconstruction_report.json` under the portable current-method run reports BRep valid counts of `1/13` for `faces_00_11`, `5/18` for `faces_12_19`, `2/9` for `faces_20_29`, and `1/10` for `faces_30_50`; by length it reports `4/20`, `3/18`, `2/8`, and `0/4` valid for the four length buckets.
- Observation: Shape-level FSQ error is higher for reconstruction-failed shapes than for BRep-valid shapes, so FSQ patch quality is likely contributing to OCC failure.
  Evidence: `reconstruction_fsq_correlation.json` under the portable current-method run reports `brep_valid` count `9` with shape Chamfer-p95 median `0.0564963925629854`, while `reconstruct_failed` count `23` has shape Chamfer-p95 median `0.1200256310403347`. The top nine shape-level Chamfer-p95 cases are all `reconstruct_failed`, with the worst at Chamfer-p95 `1.08278` and Chamfer max `1.90369`.
- Observation: Existing quality manifests may contain both stale reconstruction-time fields such as `brep_valid: false` and later quality-check fields such as `quality_brep_valid: true`.
  Evidence: The first row of `local_runs\breparg_logic_compare_20260715\breparg_original_logic_100_20260715\quality_check\step_quality_manifest.jsonl` contains both `brep_valid: false` and `quality_brep_valid: true`. `tools/audit_breparg_baseline_outputs.py` therefore prefers `quality_*` fields when present.
- Observation: The same-weight original BrepARG generation logic produces many legal closed outputs, but truly complex legal outputs are still rare under the shared protocol.
  Evidence: `breparg_baseline_quality_summary.json` for `breparg_original_logic_100_20260715` reports `brep_valid: 94`, `files_solid_closed_no_open_shell: 103`, `complex_by_step_entities_12faces_or_20edges: 11`, and `complex_and_brep_valid_closed: 9`. Only `strict_quality_accepted: 2` remain after additionally rejecting primitive-like topology.
- Observation: A second same-weight BrepARG generation-logic rerun reproduced the same legality/complexity split.
  Evidence: `local_runs\breparg_logic_compare_20260715\breparg_original_logic_100_20260715_rerun\breparg_baseline_quality_summary.json` reports `step_files: 103`, `png_files: 101`, `quality_manifest_rows: 103`, `brep_valid: 94`, `files_solid_closed_no_open_shell: 103`, `complex_by_step_entities_12faces_or_20edges: 11`, and `complex_and_brep_valid_closed: 9`.
- Observation: The new root-cause workspace now contains separated read-only diagnostics, not only the combined current-method report.
  Evidence: `local_runs\complex_curved_rootcause_suite_20260715\experiments\00_fsq_only_patch_metrics\complex_curved_diagnostics_report.json` has AR and reconstruction explicitly skipped while reporting FSQ patch Chamfer, and `experiments\01_teacher_forcing_true_token_reconstruction\complex_curved_diagnostics_report.json` contains AR teacher-forcing CE plus true-token reconstruction outcomes on the same 50-shape subset.
- Observation: The AR training entrypoint reads `NS_OUTBASE/NS_OUT/sequences_fsq_rcm.pkl`, so matched DFS and RCM AR training can be isolated without changing `train.py`.
  Evidence: `breparg_improvements/train.py` sets `SEQ_PKL = os.path.join(OUT, 'sequences_fsq_rcm.pkl')` and `stage_ar()` calls `_load_ar_seqs(AR_MAX_SEQ_LEN)`. `scripts/02b_train_dfs_rcm_ar.ps1` copies each ordering package into its own AR run directory under that expected filename before calling `train.py --stage ar`.
- Observation: Same-data BrepARG fallback requires more than the split file.
  Evidence: Official `BrepARG/train_vqvae.py` reads `--data_list`, `--surface_list`, and `--edge_list`; `BrepARG/2sequence.py` then needs the trained `--vqvae_se_weight`; `BrepARG/train_ar.py` trains from the resulting `--sequence_file`. Therefore `03b_breparg_same_data_training_fallback.ps1` fail-fast checks `$SPLIT`, `$DEDUP_SURFACES`, and `$DEDUP_EDGES`.
- Observation: A validity-gated BrepARG original-logic run can retain 100 watertight outputs, but the topology distribution remains centered on simple shapes.
  Evidence: `experiments\04_breparg_logic_generation_baseline\breparg_logic_validity_gate_100_20260715\breparg_logic_report.json` reports `quality_accepted: 100` from `attempted: 133`, with face min/median/max `2/6/44`, edge min/median/max `2/12/96`, sequence length min/median/max `49/213/1639`, and `complex_grammar: 28`. The run saw rejected complex candidates around `15-20` faces and `39-53` edges failing during FSQ/OCC reconstruction, which supports the existing FSQ/OCC bottleneck hypothesis.

## Decision Log

- Decision: Build a new script `tools/complex_curved_diagnostics.py` rather than expanding generation scripts.
  Rationale: The user wants root-cause isolation, not another generation variant. A dedicated diagnostic script can keep FSQ-only patch metrics, teacher-forcing CE, true-token reconstruction, and bucket summaries aligned on the same selected subset.
  Date/Author: 2026-07-15 / Codex.
- Decision: Use the local zip archives as the parsed geometry source for this diagnostic instead of extracting all parsed files.
  Rationale: The archives are already present and verified. Reading only the selected members is idempotent and avoids large disk churn while the user prepares the external SSD.
  Date/Author: 2026-07-15 / Codex.
- Decision: Treat "teacher-forcing reconstruction" as two separate measurements: true-token reconstruction through the FSQ decoder/OCC path and AR teacher-forcing cross-entropy on the same true token sequences.
  Rationale: The phrase can mean either "use ground-truth tokens instead of sampled AR tokens" or "measure AR next-token loss under teacher forcing." Running both disambiguates geometry-chain failure from AR-modeling difficulty.
  Date/Author: 2026-07-15 / Codex.
- Decision: Leave FSQ level/latent-dimension retraining and DFS ordering as later controlled milestones, not part of the first smoke script.
  Rationale: Those experiments require new training or sequence rebuilding. The first milestone must be read-only and quick enough to prove the diagnosis pipeline works.
  Date/Author: 2026-07-15 / Codex.
- Decision: Run the immediate BrepARG comparison with the original `BrepARG.generate_brep.generate_sequence` sampling function but the current V13 `ubuntu` AR and FSQ checkpoints.
  Rationale: This isolates generation-time logic from model-weight differences. Official BrepARG pretrained weights are still useful, but they should be evaluated as a separate baseline because changing both weights and sampling logic would confound the comparison.
  Date/Author: 2026-07-15 / Codex.
- Decision: Add skip flags to the complex-curved diagnostic instead of creating a separate FSQ-only script.
  Rationale: The selection, source resolution, MSE/Chamfer code, and bucket summaries should stay identical across baseline and capacity experiments. Skipping AR/reconstruction keeps one source of truth while supporting checkpoints whose token vocabulary is incompatible with the current AR package.
  Date/Author: 2026-07-15 / Codex.
- Decision: Preserve RCM as the default sequence ordering and make DFS an explicit opt-in.
  Rationale: Existing runs and scripts expect RCM. The controlled experiment must change one variable at a time, so DFS packages should be named and recorded distinctly rather than replacing default behavior.
  Date/Author: 2026-07-15 / Codex.
- Decision: Create a portable control workspace now, before the external SSD is connected.
  Rationale: The user asked to start in a new folder and later move comparison experiments and new models to the SSD. A workspace generator makes the current folder reproducible, gives explicit script entrypoints, and avoids depending on chat history when the folder is copied.
  Date/Author: 2026-07-15 / Codex.
- Decision: Make the first FSQ capacity training experiment change only `NS_LEVELS` to `16,16,8,8`.
  Rationale: `breparg_improvements/train.py` already exposes FSQ levels through `NS_LEVELS`, so this is the lowest-risk capacity intervention. Latent-channel or embedding-dimension changes require code changes in `build_fsq_vqvae()` and should be a separate later experiment if the level-only candidate improves the complex-curved FSQ-only metrics.
  Date/Author: 2026-07-15 / Codex.
- Decision: Add reconstruction-vs-FSQ correlation as a first-class report rather than relying on manual inspection.
  Rationale: The key question is whether FSQ quality is plausibly driving downstream STEP/OCC failure. Aggregating patch Chamfer/MSE to shape-level and comparing BRep-valid versus failed groups gives a direct bridge between FSQ-only diagnostics and true-token reconstruction outcomes.
  Date/Author: 2026-07-15 / Codex.
- Decision: Add a BrepARG baseline-output adapter instead of forcing upstream `generate_brep.py` outputs into V13's reconstruction manifest layout.
  Rationale: Official BrepARG and same-data BrepARG baselines may write flat output directories with no V13 manifest. A small adapter can audit either flat upstream outputs or V13-style `quality_check/step_quality_manifest.jsonl` outputs without changing the generator being evaluated.
  Date/Author: 2026-07-15 / Codex.
- Decision: Keep `strict_quality_accepted` separate from `complex_and_brep_valid_closed`.
  Rationale: `strict_quality_accepted` applies the display-oriented primitive-like rejection gate, while `complex_and_brep_valid_closed` measures the scientific baseline property the user cares about: complex STEP entities that are legal and closed. Merging them would hide why a run failed the gate.
  Date/Author: 2026-07-15 / Codex.
- Decision: Train DFS and RCM AR branches in separate run directories by copying their sequence package to the filename expected by `train.py`.
  Rationale: This avoids widening the training script API and keeps the comparison one-variable-at-a-time. Both branches use the same FSQ checkpoint, AR hyperparameters, max sequence length, batch size, and training launcher; only sequence ordering changes.
  Date/Author: 2026-07-15 / Codex.
- Decision: Run the root-cause workspace BrepARG-logic comparison with an all-valid gate, not the primitive-rejection gate.
  Rationale: The user asked what happens when low-quality, non-watertight, or invalid outputs are rejected during generation. Natural CAD data can include simple parts with few faces and edges, so rejecting primitive-like topology would conflate validity filtering with demo-oriented complexity selection. The run therefore used `gate_min_faces=0`, `gate_min_edges=0`, `reject_primitive_like=false`, and retained only STEP-readable, BRep-valid, closed, previewable outputs inside the representation cap.
  Date/Author: 2026-07-15 / Codex.
- Decision: Evaluate DFS-vs-RCM ordering with AR teacher-forcing CE first and skip STEP reconstruction in that comparison.
  Rationale: The ordering question is whether AR modeling of complex curved token sequences improves or degrades. Running OCC reconstruction at the same time would mix in the known FSQ/OCC failure mode and blur the AR-ordering signal.
  Date/Author: 2026-07-15 / Codex.
- Decision: Keep official BrepARG weights as the preferred baseline and make same-data BrepARG training an explicit fallback script, not an automatic continuation of the official script.
  Rationale: If the official checkpoint can be loaded and evaluated, it is the cleanest paper baseline. If it cannot, same-data self-training needs additional deduplicated surface/edge sources and significant GPU time, so it should be launched deliberately with those paths recorded.
  Date/Author: 2026-07-15 / Codex.
- Decision: Treat the official BrepARG ABC checkpoint path as attempted but incompatible for the current suite, and move the fair baseline path to same-data BrepARG training unless a matching official config/protocol is obtained.
  Rationale: The official files download successfully, but the AR checkpoint vocabulary does not match the repository ABC config. A guessed config can match the AR embedding size but did not produce STEP outputs in the local smoke window, so using it would not be a defensible baseline.
  Date/Author: 2026-07-15 / Codex.
- Decision: Put STEP/OCC/PNG validation at generation time as a final promotion gate, but do not treat this as a model-side fix.
  Rationale: The BrepARG original-logic comparison shows that sampling can produce many legal closed solids, yet retained shapes still collapse toward simple topologies under the current V13 FSQ/AR stack. Quality gates should prevent bad outputs from entering final folders, while FSQ capacity and same-data baseline experiments remain the route to improving geometric richness.
  Date/Author: 2026-07-15 / Codex.

## Outcomes & Retrospective

Milestone 1 is complete. The current Windows environment can resolve server-style sequence source metadata back to local zip archives, compute FSQ-only MSE and Chamfer distance on complex curved patches, compute AR teacher-forcing CE on the same true token records, and reconstruct true token sequences to STEP while recording OCC failures. The 50-sample diagnostic shows that the problem is not purely generation-time filtering: FSQ surface reconstruction has a heavy error tail, AR true-token CE is elevated on the complex curved subset, and true-token BRep reconstruction has a low strict-valid rate.

Milestone 2 is partially complete. A same-weight BrepARG original-generation-logic run has been produced in a fresh folder and quality-checked. It shows that switching from constrained decoding to the original unconstrained BrepARG sampling function increases speed and strict validity but does not by itself solve simple-shape collapse. The official pretrained BrepARG checkpoint baseline remains outstanding.

Milestone 3 infrastructure is partly complete. The FSQ-only diagnostic can now be run with `--skip-ar --skip-reconstruction`, and sequence rebuilding can now choose `--ordering dfs` for a DFS-vs-RCM ordering experiment. The actual capacity retrain, DFS sequence rebuild, DFS AR training, and official BrepARG checkpoint download/load validation remain outstanding.

Milestone 4 workspace staging is complete. `tools/prepare_complex_curved_control_workspace.py` creates `local_runs/complex_curved_control_suite_20260715` with five scripts that can be copied to the external SSD later. A 1-sample smoke under this workspace proves that the local diagnostic path is wired correctly. The 50-sample current-method diagnostic already exists in `local_runs/complex_curved_diagnostics_20260715/full_50_reconstruct`; the new workspace `00_current...` script is the portable rerun entrypoint, not a replacement for that evidence.

Milestone 5 current-method portable evidence is complete. The 50-sample current-method diagnostic has now been rerun inside `local_runs/complex_curved_control_suite_20260715/experiments/00_current_fsq_ar_teacher_reconstruction`, and the suite summary has been generated. The next meaningful work is no longer another generation-time tweak; it is to run `01a_train_fsq_capacity_candidate.ps1` on a GPU machine with patch shards or parsed pool, evaluate the resulting checkpoint with `01_fsq_capacity_candidate.ps1`, then decide whether FSQ levels alone improve the p95/max Chamfer tail.

Milestone 6 correlation evidence is complete. The current-method run now has `reconstruction_fsq_correlation.json` and `.md`, showing that failed reconstructions have materially worse shape-level FSQ Chamfer p95 than BRep-valid reconstructions. This strengthens the priority of the FSQ capacity experiment before investing in new generation-time filters.

Milestone 7 baseline audit infrastructure is complete, and the official pretrained BrepARG checkpoint path has now been attempted. `tools/audit_breparg_baseline_outputs.py` can audit upstream BrepARG-style flat folders or V13-style quality manifests and writes JSON, Markdown, and JSONL outputs. The existing same-weight original BrepARG logic comparison has been normalized with this tool: it confirms 94/100 quality-checked rows are BRep-valid and closed, but only 9 rows are both complex, BRep-valid, and closed. The official ABC files downloaded, but the AR checkpoint is incompatible with the current local ABC vocabulary and the inferred-config smoke produced no STEP outputs, so the suite records the official baseline as `partial` / `INCOMPATIBLE` rather than complete.

Milestone 8 ordering-control launch infrastructure is complete. The portable workspace now contains `02_dfs_rcm_ordering_rebuild.ps1` for rebuilding packages, `02b_train_dfs_rcm_ar.ps1` for matched DFS/RCM AR training, and `02c_eval_dfs_rcm_ar_complex_curved.ps1` for the complex-curved teacher-forcing comparison. The actual DFS and RCM AR training/evaluation runs remain outstanding; this milestone prepared the repeatable launch path and validation, not the final ordering result.

Milestone 9 BrepARG same-data fallback launch infrastructure is complete and is now the required fair-baseline path unless a matching official config/protocol is later found. The portable workspace contains `03b_breparg_same_data_training_fallback.ps1`, which requires same-data split and deduplicated surface/edge source paths, trains original BrepARG VQ-VAE, builds original DFS sequences, trains original BrepARG AR, generates 100 samples, and audits them with `tools/audit_breparg_baseline_outputs.py`. The fallback self-training run remains outstanding.

Milestone 10 suite-status audit infrastructure is complete. `tools/audit_complex_curved_control_suite.py` and `scripts/05_audit_suite_status.ps1` provide a cheap, repeatable way to inspect the portable workspace after moving disks or switching machines. The current rootcause status report is `total: 10`, `completed: 3`, `partial: 1`, `missing: 6`: current-method diagnostics, full FSQ capacity patch shards, and medium same-data BrepARG input preparation are complete; official BrepARG is partial/incompatible; FSQ capacity training/evaluation, DFS/RCM ordering comparison, and same-data BrepARG fallback training remain missing.

## Context and Orientation

The repository root is `D:\luolin\V13`. The current method artifacts for this diagnostic are in `ABC/processed/train_outputs/ubuntu`: `sequences_fsq_rcm.pkl` is the sequence package, `fsq_vqvae_best.pt` is the selected FSQ VQ-VAE checkpoint, and `ar_best.pt` is the selected AR checkpoint. A "sequence package" is a Python pickle dictionary with `train`, `val`, and `test` lists. Each row contains `original.input_ids`, which are the integer tokens consumed by the AR model and by `BrepARG.utils.reconstruct_cad_from_sequence`.

The local parsed geometry archives live in `ABC/processed/abc_parsed_full_archives`. Each archive is named `abc_XXXX_parsed.zip` and contains members such as `abc_0000/00000003_..._step_002.pkl`. These parsed pickle files include arrays `surf_ncs` and `edge_ncs`. `surf_ncs` is a stack of surface patches shaped like `(faces, 32, 32, 3)`. `edge_ncs` is a stack of edge curve patches shaped like `(edges, 32, 3)`.

FSQ means finite scalar quantization. In this repo the current VQ-VAE is built by `breparg_improvements/train.py::build_fsq_vqvae`, and the checkpoint stores `fsq_levels`. A "patch MSE" is the mean squared error between a raw surface or edge patch and the FSQ decoded patch. A "Chamfer distance" is a nearest-neighbor point distance between the original patch points and reconstructed patch points; it is useful when point correspondence is imperfect, while MSE is useful because these patches have ordered samples.

"Teacher-forcing CE" means evaluating the AR model on real token sequences while feeding each real prefix and asking the model to predict the next real token. It does not sample. High CE on complex curved samples means the AR model struggles to model those true sequences even before free-running generation errors accumulate.

## Plan of Work

The first milestone adds `tools/complex_curved_diagnostics.py`. It reads a sequence package, selects records from a chosen split whose grammar is valid, whose face or edge count meets the complex threshold, whose sequence length fits the requested context limit, and whose parsed patch curvature exceeds the curved threshold. It resolves each selected record's `source_relpath` to the matching local zip archive, reads the parsed pickle from the archive, and computes shape-level curvature and patch counts.

The script writes a subset manifest as JSONL. It then loads the FSQ VQ-VAE checkpoint and computes per-patch MSE plus symmetric Chamfer distance for the selected records. It also loads the AR checkpoint and computes token-weighted teacher-forcing CE per selected sequence. If requested with `--write-step`, it reconstructs the same true token sequences through `tools/evaluate_reconstruction_v13.py::reconstruct_one`, saving STEP/STL files and optional validity flags.

The second milestone runs the script with `--max-samples 3` to verify the path and artifacts. The expected output folder is `local_runs/complex_curved_diagnostics_20260715/smoke`. It must contain a summary JSON and Markdown report, a `selected_subset.jsonl`, an FSQ metric report, and an AR teacher-forcing report. If `--write-step` is used, it must also contain a reconstruction manifest and at least one STEP file unless all selected examples fail reconstruction.

The third milestone runs `--max-samples 50` and stores it under `local_runs/complex_curved_diagnostics_20260715/full_50`. This produces the evidence needed for the user to decide whether FSQ capacity is the first bottleneck. If FSQ MSE/Chamfer is already poor on complex curved real patches, capacity retraining is prioritized. If FSQ metrics are acceptable but AR CE or true-token reconstruction fails in length/face buckets, AR training or sequence construction is prioritized.

The fourth milestone prepares two controlled experiment branches. For FSQ capacity, run a new VQ-VAE training branch where only FSQ levels or latent dimension changes, then rerun the same diagnostic subset. For DFS ordering, rebuild a sequence package using the original BrepARG DFS-style ordering, keep the same FSQ checkpoint, train or evaluate a same-size AR branch, and rerun teacher-forcing/bucket diagnostics.

The fifth milestone stages the BrepARG baseline. First download official ABC checkpoints from `qingtiannihao/BrepARG` on Hugging Face. Then run original BrepARG generation and reconstruction under the same validity and quality reporting used for this method. If the official checkpoint vocabulary, sequence format, or evaluation protocol cannot be aligned with the local ABC subset, create a medium-data same-split BrepARG training run and record that official weights were not used for final comparison.

## Concrete Steps

From `D:\luolin\V13`, create the diagnostic output root:

    New-Item -ItemType Directory -Force local_runs\complex_curved_diagnostics_20260715\smoke

Run the smoke diagnostic after the script exists:

    $PY = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
    & $PY tools\complex_curved_diagnostics.py `
      --sequence ABC\processed\train_outputs\ubuntu\sequences_fsq_rcm.pkl `
      --vqvae-checkpoint ABC\processed\train_outputs\ubuntu\fsq_vqvae_best.pt `
      --ar-checkpoint ABC\processed\train_outputs\ubuntu\ar_best.pt `
      --archive-root ABC\processed\abc_parsed_full_archives `
      --output-dir local_runs\complex_curved_diagnostics_20260715\smoke `
      --split val `
      --max-samples 3 `
      --max-scan 200 `
      --max-seq-len 2048 `
      --device auto `
      --write-step `
      --validate-step

If the smoke passes, run the 50-sample diagnostic:

    & $PY tools\complex_curved_diagnostics.py `
      --sequence ABC\processed\train_outputs\ubuntu\sequences_fsq_rcm.pkl `
      --vqvae-checkpoint ABC\processed\train_outputs\ubuntu\fsq_vqvae_best.pt `
      --ar-checkpoint ABC\processed\train_outputs\ubuntu\ar_best.pt `
      --archive-root ABC\processed\abc_parsed_full_archives `
      --output-dir local_runs\complex_curved_diagnostics_20260715\full_50 `
      --split val `
      --max-samples 50 `
      --max-scan 5000 `
      --max-seq-len 2048 `
      --device auto `
      --write-step `
      --validate-step

To evaluate only FSQ reconstruction for a capacity checkpoint whose codebook or token vocabulary differs from the current AR package, reuse the current sequence package only for selecting the complex curved records and skip AR plus token reconstruction:

    & $PY tools\complex_curved_diagnostics.py `
      --sequence ABC\processed\train_outputs\ubuntu\sequences_fsq_rcm.pkl `
      --vqvae-checkpoint PATH\TO\capacity_candidate\fsq_vqvae_best.pt `
      --ar-checkpoint ABC\processed\train_outputs\ubuntu\ar_best.pt `
      --archive-root ABC\processed\abc_parsed_full_archives `
      --output-dir local_runs\complex_curved_diagnostics_20260715\fsq_capacity_candidate `
      --split val `
      --max-samples 50 `
      --max-scan 5000 `
      --max-seq-len 2048 `
      --device auto `
      --skip-ar `
      --skip-reconstruction

To rebuild a DFS sequence package for ordering control after a promoted VQ-VAE checkpoint is available, run `tools/run_sharded_sequence.py` with `--ordering dfs`. The existing RCM path remains the default. Use separate output directories so the two packages cannot be confused:

    & $PY tools\run_sharded_sequence.py `
      --split PATH\TO\split.pkl `
      --checkpoint PATH\TO\fsq_vqvae_best.pt `
      --shard-dir local_runs\complex_curved_diagnostics_20260715\sequence_dfs_shards `
      --merge-output local_runs\complex_curved_diagnostics_20260715\sequences_fsq_dfs.pkl `
      --summary local_runs\complex_curved_diagnostics_20260715\sequences_fsq_dfs_summary.json `
      --manifest local_runs\complex_curved_diagnostics_20260715\sequences_fsq_dfs_manifest.jsonl `
      --workers 8 `
      --chunks 0-99 `
      --resume `
      --ordering dfs

To rebuild the matching RCM package for the one-variable-at-a-time control, use the same command but set `--ordering rcm` and write to `sequence_rcm_shards` plus `sequences_fsq_rcm.pkl`.

To create the portable local control workspace, run:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe tools\prepare_complex_curved_control_workspace.py

This creates `local_runs\complex_curved_control_suite_20260715` with:

    README.md
    experiment_config.json
    scripts\00_current_fsq_ar_teacher_reconstruction.ps1
    scripts\01a_train_fsq_capacity_candidate.ps1
    scripts\01_fsq_capacity_candidate.ps1
    scripts\02_dfs_rcm_ordering_rebuild.ps1
    scripts\02b_train_dfs_rcm_ar.ps1
    scripts\02c_eval_dfs_rcm_ar_complex_curved.ps1
    scripts\03_breparg_official_baseline.ps1
    scripts\03b_breparg_same_data_training_fallback.ps1
    scripts\04_summarize_reports.ps1
    scripts\05_audit_suite_status.ps1

Run the current-method portable diagnostic with:

    powershell -ExecutionPolicy Bypass -File local_runs\complex_curved_control_suite_20260715\scripts\00_current_fsq_ar_teacher_reconstruction.ps1

This is the 50-sample run for FSQ-only metrics, AR teacher-forcing CE, and true-token STEP reconstruction. It may take several minutes because it performs FSQ decoding and OCC reconstruction.

To launch the first FSQ capacity candidate after setting the training data path inside the script, run:

    powershell -ExecutionPolicy Bypass -File local_runs\complex_curved_control_suite_20260715\scripts\01a_train_fsq_capacity_candidate.ps1

Edit `$PATCH_SHARD_ROOT` to a folder containing `vq_patch_shard_*.pkl.zst`, or clear it and set `$PARSED_POOL` to an extracted parsed-pickle pool. The script writes a candidate checkpoint under `local_runs\complex_curved_control_suite_20260715\experiments\01a_train_fsq_capacity_candidate\fsq_levels_16_16_8_8_complex_curved_20260715`. After training, set `$CAPACITY_VQVAE` in `scripts\01_fsq_capacity_candidate.ps1` to that `fsq_vqvae_best.pt` and run the FSQ-only diagnostic.

To regenerate the suite summary and the reconstruction-vs-FSQ correlation report after running `00_current...`, run:

    powershell -ExecutionPolicy Bypass -File local_runs\complex_curved_control_suite_20260715\scripts\04_summarize_reports.ps1

This writes `complex_curved_diagnostic_summary.md` at the suite root and `reconstruction_fsq_correlation.json/.md` under `experiments\00_current_fsq_ar_teacher_reconstruction`.

To audit the whole control suite at any time, run:

    powershell -ExecutionPolicy Bypass -File local_runs\complex_curved_control_suite_20260715\scripts\05_audit_suite_status.ps1

This writes:

    local_runs\complex_curved_control_suite_20260715\suite_status.json
    local_runs\complex_curved_control_suite_20260715\suite_status.md

On 2026-07-15, the status report showed only `current_method` complete and recommended this order: train the FSQ capacity candidate, rebuild DFS/RCM sequence packages, then run the official BrepARG baseline.

To run the ordering control after `02_dfs_rcm_ordering_rebuild.ps1` has produced both sequence packages, run:

    powershell -ExecutionPolicy Bypass -File local_runs\complex_curved_control_suite_20260715\scripts\02b_train_dfs_rcm_ar.ps1
    powershell -ExecutionPolicy Bypass -File local_runs\complex_curved_control_suite_20260715\scripts\02c_eval_dfs_rcm_ar_complex_curved.ps1

The training script writes matched branches under:

    local_runs\complex_curved_control_suite_20260715\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_dfs_matched_20260715
    local_runs\complex_curved_control_suite_20260715\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_rcm_matched_20260715

The evaluation script writes complex-curved teacher-forcing reports under:

    local_runs\complex_curved_control_suite_20260715\experiments\02_dfs_rcm_ordering\ar_complex_curved_eval\dfs_teacher_forcing
    local_runs\complex_curved_control_suite_20260715\experiments\02_dfs_rcm_ordering\ar_complex_curved_eval\rcm_teacher_forcing

If official BrepARG weights cannot be loaded or cannot be evaluated under this protocol, edit the three required same-data paths in `scripts\03b_breparg_same_data_training_fallback.ps1`:

    $SPLIT = "PATH\TO\same_data_split.pkl"
    $DEDUP_SURFACES = "PATH\TO\deduplicated_surface_source.pkl"
    $DEDUP_EDGES = "PATH\TO\deduplicated_edge_source.pkl"

Then run:

    powershell -ExecutionPolicy Bypass -File local_runs\complex_curved_control_suite_20260715\scripts\03b_breparg_same_data_training_fallback.ps1

This writes the fallback under:

    local_runs\complex_curved_control_suite_20260715\experiments\03b_breparg_same_data_training_fallback

Download official BrepARG weights only after confirming enough disk or after the external SSD is available:

    $PY = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
    & $PY -m pip install -U huggingface_hub
    huggingface-cli download qingtiannihao/BrepARG `
      checkpoint/weights/abc_ar.pt `
      checkpoint/weights/abc_vqvae.pt `
      --local-dir local_runs\complex_curved_diagnostics_20260715\breparg_official `
      --local-dir-use-symlinks False

After downloading, first run a load-only smoke before generating large batches. The official checkpoint filenames may use the upstream BrepARG state-dict format rather than the V13 FSQ-aware format, so the smoke must prove that `BrepARG/generate_brep.py` can load both checkpoints with `--dataset_type abc`. If it fails, record the mismatch and switch to same-data BrepARG self-training for the baseline.

To audit any upstream BrepARG output directory after generation, run:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe tools\audit_breparg_baseline_outputs.py `
      --run-dir PATH\TO\BREPARG_OUTPUT_DIR `
      --output PATH\TO\breparg_baseline_quality_summary.json `
      --markdown-output PATH\TO\breparg_baseline_quality_summary.md `
      --manifest-output PATH\TO\breparg_baseline_quality_manifest.jsonl `
      --min-faces 12 `
      --min-edges 20 `
      --max-faces 45 `
      --max-edges 120

## Validation and Acceptance

Run syntax checks:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m py_compile tools\complex_curved_diagnostics.py

Run the focused unit tests for source resolution and bucket summaries after they exist:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_complex_curved_diagnostics

Run the focused unit test for the portable workspace generator:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_prepare_complex_curved_control_workspace

Run the focused unit test for the BrepARG baseline-output adapter:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_audit_breparg_baseline_outputs

Run the focused tests for summarization and reconstruction/FSQ correlation:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_summarize_complex_curved_diagnostics tests.test_analyze_reconstruction_fsq_correlation

Run the focused tests for ordering metadata and mixed-ordering shard rejection:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_local_pipeline_helpers.LocalPipelineHelperTests.test_run_sharded_sequence_metadata_records_ordering tests.test_local_pipeline_helpers.LocalPipelineHelperTests.test_sequence_sharding_rejects_mixed_ordering_shards

A quick combined validation for the newly staged workspace is:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_audit_breparg_baseline_outputs tests.test_prepare_complex_curved_control_workspace tests.test_complex_curved_diagnostics tests.test_summarize_complex_curved_diagnostics tests.test_analyze_reconstruction_fsq_correlation tests.test_local_pipeline_helpers.LocalPipelineHelperTests.test_run_sharded_sequence_metadata_records_ordering tests.test_local_pipeline_helpers.LocalPipelineHelperTests.test_sequence_sharding_rejects_mixed_ordering_shards
    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m py_compile tools\audit_breparg_baseline_outputs.py tools\prepare_complex_curved_control_workspace.py tools\complex_curved_diagnostics.py tools\summarize_complex_curved_diagnostics.py tools\analyze_reconstruction_fsq_correlation.py tools\run_sharded_sequence.py breparg_improvements\sequence_sharding.py

The smoke run is accepted when `local_runs\complex_curved_diagnostics_20260715\smoke\complex_curved_diagnostics_report.json` exists, reports `status` as `VERIFIED`, and has `selected_count` equal to 3. The report must contain `fsq_patch_metrics`, `ar_teacher_forcing`, `teacher_reconstruction`, and `bucket_summary` sections. If `--write-step` is enabled, `teacher_reconstruction.attempted` must equal 3 and any reconstruction failures must be listed by reason in the JSONL manifest rather than crashing the run.

The portable workspace smoke is accepted when `local_runs\complex_curved_control_suite_20260715\experiments\00_smoke_one\complex_curved_diagnostics_report.json` exists, reports `status` as `VERIFIED`, and includes both `fsq_patch_metrics` and `ar_teacher_forcing`.

The portable 50-sample current diagnostic is accepted when `local_runs\complex_curved_control_suite_20260715\experiments\00_current_fsq_ar_teacher_reconstruction\complex_curved_diagnostics_report.json` exists, reports `status` as `VERIFIED`, has `selected_count` equal to 50, has `fsq_patch_metrics.patch_count` greater than 0, and has `teacher_reconstruction.attempted` equal to 50. On 2026-07-15 it produced 27 STEP files and 9 BRep-valid reconstructions, which is evidence of a real reconstruction bottleneck rather than a run failure.

The reconstruction-vs-FSQ correlation report is accepted when `reconstruction_fsq_correlation.json` exists under the current-method experiment, has `shape_count` equal to 50, and contains `groups.brep_valid.chamfer_p95.median` and `groups.reconstruct_failed.chamfer_p95.median`. On 2026-07-15 those medians were `0.0564963925629854` and `0.1200256310403347`, respectively.

The 50-sample run is accepted when the full report exists, has `selected_count` equal to 50, and includes bucketed evidence for face count and sequence length. It is not required that all 50 reconstruct to valid STEP; the purpose is to measure failure rate, not hide failures.

The BrepARG baseline-output audit is accepted when `breparg_baseline_quality_summary.json`, `breparg_baseline_quality_summary.md`, and `breparg_baseline_quality_manifest.jsonl` exist for the run directory. The JSON summary must report `step_files`, `quality_manifest_rows`, `brep_valid`, `complex_by_step_entities_12faces_or_20edges`, and `complex_and_brep_valid_closed`. On the same-weight original-logic comparison, those key values are `103`, `100`, `94`, `11`, and `9`.

The ordering-control launch infrastructure is accepted when the portable workspace contains `02b_train_dfs_rcm_ar.ps1` and `02c_eval_dfs_rcm_ar_complex_curved.ps1`, all workspace scripts parse with PowerShell, and `tests.test_prepare_complex_curved_control_workspace` verifies both scripts reference the DFS/RCM sequence packages, `tools/preflight_ar_training.py`, `breparg_improvements\train.py --stage ar`, and `tools/complex_curved_diagnostics.py`. The ordering experiment itself is accepted only after both AR checkpoints exist and both complex-curved diagnostic reports exist with comparable `ar_teacher_forcing` sections.

The same-data BrepARG fallback launch infrastructure is accepted when the portable workspace contains `03b_breparg_same_data_training_fallback.ps1`, all workspace scripts parse with PowerShell, and `tests.test_prepare_complex_curved_control_workspace` verifies it references `BrepARG\train_vqvae.py`, `BrepARG\2sequence.py`, `BrepARG\train_ar.py`, `BrepARG\generate_brep.py`, `tools\audit_breparg_baseline_outputs.py`, `$SPLIT`, `$DEDUP_SURFACES`, and `$DEDUP_EDGES`. The fallback baseline itself is accepted only after its generated audit summary exists and records the same-data paths plus trained VQ-VAE/AR artifact paths.

The suite-status audit is accepted when `suite_status.json` and `suite_status.md` exist at the portable workspace root, and `tests.test_audit_complex_curved_control_suite` verifies that completed, partial, and missing experiments are classified correctly. On the current workspace the summary is `total: 8`, `completed: 1`, `partial: 0`, `missing: 7`, with FSQ capacity training as the first next action.

## Idempotence and Recovery

The diagnostic script must create its output directory if missing and overwrite the final JSON/Markdown reports for the same output directory. JSONL manifests are rewritten from scratch per run to avoid mixing old and new rows. It only reads the sequence pickle, checkpoints, and zip archives; it does not modify model artifacts or parsed archives.

If a run is interrupted, rerun the same command. For long 50-sample runs, use a new output directory suffix such as `full_50_retry1` if preserving partial output matters. Do not delete current generated or training outputs unless the user explicitly asks.

The portable workspace generator is idempotent: rerunning `tools\prepare_complex_curved_control_workspace.py` rewrites the README, config, and script entrypoints while leaving experiment outputs in place. It does not download official BrepARG weights or start training by itself.

## Artifacts and Notes

The official BrepARG web sources inspected on 2026-07-15 are:

- `https://github.com/123qiang06/BrepARG`, whose README describes it as the official PyTorch implementation and links pretrained weights.
- `https://huggingface.co/qingtiannihao/BrepARG`, whose model card lists `checkpoint/weights/abc_ar.pt` and `checkpoint/weights/abc_vqvae.pt`.

## Interfaces and Dependencies

`tools/complex_curved_diagnostics.py` must provide these importable helpers so tests and future scripts can reuse them:

    source_relpath_from_group(group: dict) -> str | None
    archive_path_for_relpath(relpath: str, archive_root: Path) -> Path
    load_parsed_from_archive(relpath: str, archive_root: Path) -> dict
    sequence_length_bucket(length: int) -> str
    face_count_bucket(faces: int) -> str
    skipped_stage_report(stage: str, reason: str) -> dict

`tools/run_sharded_sequence.py` must provide these importable helpers so tests and future scripts can reuse them:

    normalize_ordering(value: str) -> str
    load_sequence_module(ordering: str = "rcm")
    metadata_from_preprocessor(pre, ordering: str = "rcm") -> dict

`tools/prepare_complex_curved_control_workspace.py` must provide this importable helper so tests and future agents can recreate the portable workspace:

    prepare_workspace(output_dir: Path, python_exe: str, sequence_path: Path, vqvae_checkpoint: Path, ar_checkpoint: Path, archive_root: Path) -> dict

`tools/analyze_reconstruction_fsq_correlation.py` must provide this importable helper so tests and future agents can connect FSQ patch error to reconstruction outcomes:

    analyze_correlation(manifest_path: Path, patch_metrics_path: Path, top_k: int = 10) -> dict

`tools/audit_breparg_baseline_outputs.py` must provide these importable helpers so tests and future scripts can normalize official or same-data BrepARG generation folders:

    audit_breparg_baseline_outputs(run_dir: Path, *, min_faces: int = 12, min_edges: int = 20, max_faces: int = 45, max_edges: int = 120, require_quality: bool = True) -> dict
    write_outputs(audit: dict, output: Path | None, markdown_output: Path | None, manifest_output: Path | None) -> None

`tools/audit_complex_curved_control_suite.py` must provide these importable helpers so tests and future scripts can check progress after moving the workspace:

    audit_suite(root: Path) -> dict
    write_outputs(audit: dict, output: Path | None, markdown_output: Path | None) -> None

The script depends on `numpy`, `torch`, the local `BrepARG` modules, `breparg_improvements.vqvae_sampling`, and `tools.evaluate_reconstruction_v13`. It intentionally uses Python's standard `zipfile` for archive reads so it does not require `zstandard` in the Windows training environment.

Revision note 2026-07-15: Initial plan created to support the user's requested FSQ curved reconstruction, AR teacher-forcing, four controlled experiments, and BrepARG baseline path. It records the decision to start with a read-only zip-backed diagnostic before launching new training.

Revision note 2026-07-15: Added the fresh BrepARG original-generation-logic comparison run, recorded the 50-face/150-edge representation cap, and clarified that official BrepARG weights remain a separate baseline from the same-weight generation-logic comparison.

Revision note 2026-07-15: Added FSQ-only skip flags, DFS/RCM sequence ordering controls, mixed-ordering merge protection, and concrete commands for capacity and ordering control experiments.

Revision note 2026-07-15: Added the portable `complex_curved_control_suite_20260715` workspace generator, recorded smoke/test evidence, and made the official BrepARG baseline script point to the current Hugging Face ABC checkpoint files.

Revision note 2026-07-15: Reran the 50-sample current-method diagnostic inside the portable workspace, recorded the exact metrics, generated the suite summary, and added a level-only FSQ capacity training launcher.

Revision note 2026-07-15: Added reconstruction-vs-FSQ correlation analysis and recorded evidence that failed true-token reconstructions have higher shape-level FSQ Chamfer p95 than BRep-valid reconstructions.

Revision note 2026-07-15: Added the BrepARG baseline-output audit adapter, wired it into the official baseline workspace script, regenerated the portable workspace, and normalized the same-weight original BrepARG logic comparison. The normalized audit reports 94 BRep-valid rows but only 9 complex+BRep-valid+closed rows, confirming that legality and complexity remain separate failure axes.

Revision note 2026-07-15: Added matched DFS/RCM AR training and complex-curved teacher-forcing evaluation entrypoints to the portable workspace. The scripts train each ordering in a separate run directory while copying the selected sequence package to `sequences_fsq_rcm.pkl` for compatibility with `train.py`, then evaluate AR teacher-forcing CE with reconstruction skipped to isolate the ordering variable.

Revision note 2026-07-15: Added the same-data BrepARG fallback training script for the case where official ABC weights cannot be used. The fallback explicitly requires the same split and deduplicated surface/edge source data, trains original BrepARG VQ-VAE and AR, generates a smoke set, and audits it with the shared baseline-output adapter.

Revision note 2026-07-15: Added the suite-status audit tool and `05_audit_suite_status.ps1`. The audit writes JSON/Markdown summaries and next actions; the current workspace reports one complete tracked experiment and seven missing tracked experiments, with FSQ capacity training first.

Revision note 2026-07-15: Added the user-requested same-weight BrepARG generation-logic rerun under `breparg_original_logic_100_20260715_rerun`, recorded its matching 94/103 BRep-valid and 9 complex+BRep-valid+closed counts, and made `tools/audit_breparg_baseline_outputs.py` read JSONL manifests with `utf-8-sig` so PowerShell-written quality manifests with a BOM can be audited.

Revision note 2026-07-15: Ran the official BrepARG ABC baseline smoke path. Hardened `03_breparg_official_baseline.ps1` to pin `huggingface_hub>=0.20.2,<0.26`, use the Python `hf_hub_download` API instead of PATH-dependent `huggingface-cli`, and fail fast for native command nonzero exits. Recorded official checkpoint incompatibility under `experiments/03_breparg_official_baseline/official_baseline_incompatibility_report.*`, updated `tools/audit_complex_curved_control_suite.py` to classify zero-output baselines correctly and read BOM JSON, and made the suite next action point to same-data BrepARG fallback after official incompatibility is recorded.

Revision note 2026-07-15: Created `local_runs\complex_curved_rootcause_suite_20260715` as the new temporary workspace requested by the user, reran separated FSQ-only and teacher-forcing/true-token reconstruction diagnostics there, generated the reconstruction-vs-FSQ correlation report, and added `rootcause_readout_20260715.md` with the key metrics and next actions for later transfer to the external SSD.

Revision note 2026-07-15: Added the fresh user-requested BrepARG original-logic generation comparison under `local_runs\breparg_logic_compare_20260715\breparg_original_logic_100_user_compare_20260715` and the readout `breparg_original_logic_user_compare_readout_20260715.md`. The new run reproduces the prior BrepARG-logic metrics exactly: 124 attempts, 103 STEP files, 100 retained PNG outputs, 94 BRep-valid STEP files, 11 complex-by-entity outputs, and 9 complex+BRep-valid+closed outputs.

Revision note 2026-07-15: Added the same-data BrepARG input preparation stage and wired it into both the workspace generator and suite audit. `03a_prepare_breparg_same_data_inputs.ps1` now creates `same_data_split.pkl`, `deduplicated_surface_source.pkl`, `deduplicated_edge_source.pkl`, and `same_data_input_summary.json` under the fallback `data` folder; `03b_breparg_same_data_training_fallback.ps1` uses those files by default. A tiny real-data smoke under `data_smoke` verified 10 train, 3 val, and 3 test materialized records with readable parsed pickle files and nonempty SE source arrays. The full `data` preparation remains outstanding.

Revision note 2026-07-15: Changed the default same-data BrepARG input script to prepare a medium local baseline pool rather than the larger external-SSD pool. The medium `data` pool now exists and is verified with `10000/1000/1000` train/val/test records, 176,605 surface patches, 429,555 edge patches, and zero skipped records. The larger `03a_prepare_breparg_same_data_inputs_full.ps1` writes to `data_full` and should be used when the external SSD or another large target disk is available.

Revision note 2026-07-15: Hardened `03b_breparg_same_data_training_fallback.ps1` after CLI compatibility review. The current Windows environment was missing `tensorboard`, so `BrepARG/train_vqvae.py --help` and `BrepARG/train_ar.py --help` failed during import before argparse. The fallback launcher now preflights dependencies, installs `tensorboard` if absent, fails fast on native command errors, and passes explicit TensorBoard log directories to avoid `os.makedirs("")` failures in `BrepARG/trainer.py`.

Revision note 2026-07-15: Added `--allow-primitive-like` to `tools\generate_quality_gated_step_png.py` so generation-time quality gating can be run in two explicit modes: an `all_valid` mode that keeps simple but watertight valid CAD parts, and a `complex_valid` mode that requires face/edge complexity for demo selection. This records the decision that minimum face/edge thresholds are a selection policy, not a universal definition of correctness for the data distribution.

Revision note 2026-07-15: Added `02_smoke_dfs_rcm_ordering_rebuild.ps1` to the root-cause workspace generator and suite audit. The smoke prepares a 5/3/3 same-data split, rebuilds DFS and RCM sequences with `workers=1` and `chunks=0-0`, and records verified summaries before the full DFS/RCM ordering experiment is launched. On the current root-cause workspace it completed with `split_total_written=11`, `dfs_sequences=11`, `rcm_sequences=11`, and zero out-of-vocabulary tokens for both orderings. The suite audit now tracks `dfs_rcm_sequence_rebuild_smoke` separately from the full sequence rebuild, AR training, and teacher-forcing comparison.

Revision note 2026-07-15: Added and ran `02_medium_dfs_rcm_ordering_rebuild.ps1` because the local D: drive had only about 40 GB free and the full 50k/5k/5k materialized split would be too close to the remaining disk budget. The medium script reuses the already materialized BrepARG same-data input pool instead of duplicating parsed records, writes outputs under `experiments\02_dfs_rcm_ordering\sequence_rebuild_medium`, and now falls back into the matched AR train/eval scripts when full sequence packages are absent. On the current root-cause workspace the medium DFS and RCM packages each contain `12000` sequences (`10000/1000/1000` train/val/test), `6` shards, `out_of_vocab=0`, and `se_tokens_per_element=4`; the output folder is about `0.163 GB`. The suite audit tracks this as `dfs_rcm_sequence_rebuild_medium`, while the full sequence rebuild remains missing until the SSD/server run.

Revision note 2026-07-15: Added `tools\subset_ar_sequence_package.py` and `02b_smoke_dfs_rcm_ar_medium_safe.ps1` to prove the medium DFS/RCM AR training path end-to-end before launching longer local training. The smoke subsets each medium sequence package to `64/16/16` train/val/test at `max_seq_len=2048`, runs AR preflight, trains `train.py --stage ar` for `1` epoch with `bs=2`, and copies `train_report.json` back into each run directory for SSD migration. On the current root-cause workspace both DFS and RCM smoke branches saved `ar_best.pt` and `ar_latest.pt`; DFS smoke recorded `best_val_ce=8.6746`, RCM smoke recorded `best_val_ce=8.6689`, and suite audit now marks `dfs_rcm_ar_training_medium_smoke` complete. This remains a wiring smoke only; the next local ordering diagnostic is `02b_train_dfs_rcm_ar_medium_safe.ps1`, while the full ordering experiment still requires the full sequence rebuild and matched AR training on SSD/server.

Revision note 2026-07-15: Completed the medium matched DFS/RCM AR diagnostic and refreshed the root-cause suite audit. The current suite status is `completed=7`, `partial=1`, and `missing=6`. Medium DFS training used `9875/988` usable train/val sequences for 5 epochs at `max_seq_len=2048`, `bs=4`, and `lr=5e-5`, saving `ar_best.pt` with `best_val_ce=2.4564`; medium RCM used the same settings and saved `ar_best.pt` with `best_val_ce=2.5889`. This suggests DFS is slightly better on the local medium diagnostic, but the full ordering experiment remains open until the full sequence rebuild and matched AR training are run on SSD/server.

Revision note 2026-07-15: Ran `02c_eval_dfs_rcm_ar_complex_curved.ps1` using the medium sequence and medium AR fallback artifacts. Both DFS and RCM evaluated the same 50 complex curved validation records with reconstruction skipped. DFS recorded AR token-weighted teacher CE `3.6398` with median `3.6056` and p95 `5.0378`; RCM recorded token-weighted CE `3.8508` with median `3.8159` and p95 `5.2234`. FSQ patch metrics were identical for both (`Chamfer p95=0.1023834702`) because this isolates only ordering/AR. The suite audit now reports `completed=8`, `partial=1`, and `missing=5`.

Revision note 2026-07-15: Added and ran a read-only BrepARG same-data fallback preflight. `tools\preflight_breparg_same_data_fallback.py` verifies the medium same-data input pickles, required Python modules, BrepARG CLI `--help` compatibility, the official incompatibility evidence, and output directories without starting training. The current root-cause workspace now contains `scripts\03b_preflight_breparg_same_data_fallback.ps1`, and the preflight report at `experiments\03b_breparg_same_data_training_fallback\breparg_same_data_preflight.json` is `READY` with no blocking reasons. It confirms train/val/test `10000/1000/1000`, surface patches `176605`, edge patches `429555`, all required modules available, all BrepARG CLI help checks passed, official status `INCOMPATIBLE` with `abc_ar.pt` embedding shape `[7222, 256]`, and `training_started=false`. Suite audit now tracks this as `breparg_same_data_preflight`, bringing the status to `completed=9`, `partial=1`, and `missing=5`.

Revision note 2026-07-15: Added and ran a read-only FSQ capacity candidate preflight. `tools\preflight_fsq_capacity_candidate.py` verifies the full VQ patch shards, requested FSQ levels, required modules, and `breparg_improvements\train.py --help` without starting training. The workspace now contains `scripts\01a_preflight_fsq_capacity_candidate.ps1`, and `scripts\01a_train_fsq_capacity_candidate.ps1` defaults `$PATCH_SHARD_ROOT` to the existing `experiments\01a_train_fsq_capacity_candidate\vq_patch_shards_full` instead of the old placeholder path. The preflight report at `experiments\01a_train_fsq_capacity_candidate\fsq_capacity_preflight.json` is `READY` with no blocking reasons. It confirms `344` shard files, `34,393,215` patches, estimated shard size `35.34 GB`, levels `[16,16,8,8]`, codebook size `16384`, requested samples `450000`, and `training_started=false`. Suite audit now tracks this as `fsq_capacity_preflight`, bringing the status to `completed=10`, `partial=1`, and `missing=5`.

Revision note 2026-07-15: Refreshed the external-SSD migration dry run after adding the FSQ and BrepARG preflights. `tools\prepare_rootcause_ssd_migration.py` now renders continuation commands for `01a_preflight_fsq_capacity_candidate.ps1`, `01a_train_fsq_capacity_candidate.ps1`, `03b_preflight_breparg_same_data_fallback.ps1`, and `03b_breparg_same_data_training_fallback.ps1`. The current dry-run plan at `local_runs\complex_curved_rootcause_suite_20260715\ssd_migration_plan.json` is `ready_to_execute=true` with `copy_reference_models=true` and `copy_archives=false`; it will copy the experiment artifacts (`57.688 GB`) plus reference sequence/VQ-VAE/AR files (`1.507 GB`, `0.229 GB`, `0.114 GB`). The command document now explicitly notes that parsed archives are not copied by default and that `--copy-archives` is needed if the SSD copy should rerun zip-backed diagnostics or prepare larger same-data splits without reading the original workspace.

Revision note 2026-07-15: Removed the manual placeholder from the FSQ capacity evaluation entrypoint. `tools\prepare_complex_curved_control_workspace.py` now generates `scripts\01_fsq_capacity_candidate.ps1` so it first uses `$env:V13_CAPACITY_VQVAE` when present, otherwise it automatically points to `experiments\01a_train_fsq_capacity_candidate\fsq_levels_16_16_8_8_complex_curved_20260715\fsq_vqvae_best.pt`. If the checkpoint is missing, the script tells the operator to run `scripts\01a_train_fsq_capacity_candidate.ps1` or set the override variable. The root-cause workspace scripts were regenerated, focused tests passed, `py_compile` passed, and all PowerShell scripts parsed successfully. The suite status remains `completed=10`, `partial=1`, and `missing=5` because capacity training and eval still have not been run.

Revision note 2026-07-15: Strengthened `tools\preflight_breparg_same_data_fallback.py` so the same-data BrepARG fallback preflight now writes `planned_commands` and `cli_required_args` sections. These sections verify the exact argument names expected by `03b_breparg_same_data_training_fallback.ps1` against `BrepARG\train_vqvae.py --help`, `BrepARG\2sequence.py --help`, `BrepARG\train_ar.py --help`, `BrepARG\generate_brep.py --help`, and `tools\audit_breparg_baseline_outputs.py --help` without starting training. Fresh verification ran the preflight script successfully, refreshed `suite_status.md` to `completed=10`, `partial=1`, `missing=5`, and passed focused unittest, py_compile, and PowerShell parser checks.

Revision note 2026-07-15: Added `breparg_improvements\vqvae_sample_cache.py` and wired `NS_VQ_SAMPLE_CACHE` into `breparg_improvements\train.py` so FSQ capacity training can reuse the expensive 450k patch-shard sample set across restarts. The capacity preflight now records `sample_cache.path`, `sample_cache.enabled`, existence, and size; `01a_train_fsq_capacity_candidate.ps1` and `01a_train_fsq_capacity_candidate_server_linux.sh` both point to `vq_samples_450000_seed0.npz`. Verification passed focused cache/preflight/workspace/audit tests, py_compile, PowerShell parser checks, refreshed FSQ preflight (`READY`), and refreshed suite audit (`completed=10`, `partial=1`, `missing=5`).

Revision note 2026-07-15: Added a standalone FSQ capacity sample-cache builder. `tools\build_vqvae_sample_cache.py` samples from VQ patch shards with the same complex/curved selection and loss-weight settings used by capacity training, writes `vq_samples_450000_seed0.npz`, and emits `vq_samples_450000_seed0_summary.json`. The workspace generator now creates `scripts\01a_build_fsq_capacity_sample_cache.ps1`, and `tools\preflight_fsq_capacity_candidate.py` sets `next_command` to that script while the cache is absent. Fresh verification passed focused tests, py_compile, PowerShell parser checks, refreshed FSQ preflight (`READY`, next command is cache build), and refreshed suite audit.

Revision note 2026-07-15: Refreshed the active root-cause status while the FSQ capacity run is still alive. Process check shows wrapper PID `35488`, Python PID `50344`, and watcher PID `21168`; completion checker reports `training_process_alive` and `train_report_missing`, with history at epoch `15`, last val `0.00016431`, and best val `0.00015291` at epoch `13`. Added `current_status_answer_20260715_1823.md` to clarify in Chinese that the active FSQ capacity training is experiment 2 from the controlled table, not the BrepARG fallback and not proof that FSQ is the sole cause. Also corrected the BrepARG original-logic comparison to count complexity from the final accepted set: 9/100 accepted samples meet the faces>=12 or edges>=20 threshold.
