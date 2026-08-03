# Complex-Curved Root-Cause Current Status and Next Work

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document follows `PLANS.md` from the repository root. It is intentionally self-contained: a new engineer or agent should be able to read only this file plus the current working tree and understand what has already been proven, what remains open, and which command should be run next.

## Purpose / Big Picture

The project is trying to diagnose why generated CAD/B-rep results on complex curved shapes are often too simple or invalid. Generation-time filtering can remove bad outputs, but it cannot tell whether the core failure comes from the FSQ VQ-VAE geometry representation, AR token modeling, ordering, topology reconstruction, or the BrepARG baseline protocol. After this plan is followed, the user will have a clean evidence chain: current FSQ-only reconstruction, true-token teacher-forcing reconstruction, higher-capacity FSQ control, DFS-vs-RCM ordering control, failure buckets, and a BrepARG same-data baseline run under the same protocol.

The current user-visible outcome is a status-checked experiment suite under `local_runs/complex_curved_rootcause_suite_20260715`, a recovered D-drive DFS/RCM ordering evaluation under `D:\V13_rootcause_recovery_20260717`, and a clear queue for the remaining BrepARG same-data fallback baseline. The final observable success condition is not simply "training finished"; it is that the suite contains comparable reports for V13 and BrepARG and that each report can be audited for FSQ Chamfer, AR teacher-forcing cross entropy, generated STEP validity, face/edge complexity, and failure buckets.

## Progress

- [x] (2026-07-15 03:24 +08:00) Ran the current FSQ-only complex-curved diagnostic on 50 validation shapes. It evaluated 3,399 patches and found FSQ Chamfer p95 `0.1501216799`; surface Chamfer p95 was much worse at `0.4123849943`, showing a heavy tail on complex curved surfaces.
- [x] (2026-07-15 03:49 +08:00) Ran true-token teacher-forcing and reconstruction for the current method. AR token-weighted CE was `0.7467434714`; true token sequences saved STEP for `27/50`, but only `9/50` were strict BRep-valid. This proves failures occur before free-running generation.
- [x] (2026-07-16 06:04 +08:00) Completed the higher-capacity FSQ candidate training with levels `(16,16,8,8)` and codebook size `16384`. It early-stopped at epoch `100` with best validation reconstruction `5.283e-05` at epoch `82`.
- [x] (2026-07-17 04:51 +08:00) Completed the higher-capacity FSQ candidate evaluation. Overall FSQ Chamfer p95 changed from `0.1501216799` to `0.1564181745`, a `+4.19%` regression; surface Chamfer p95 improved, but not enough to isolate capacity as the main root cause.
- [x] (2026-07-17 10:33 +08:00) Completed BrepARG original generation logic using current compatible V13 weights. It saved 100 STEP+PNG rows from 111 attempts, but only about `11%` were complex by the `faces >= 12 or edges >= 20` rule and about `79%` were very simple. This shows that switching to original BrepARG sampling logic alone does not solve simple-topology collapse.
- [x] (2026-07-17 10:18 +08:00) Confirmed official BrepARG ABC weights were tried first and are incompatible with the current local protocol. The official ABC AR embedding shape is `[7222, 256]`, while the current local vocabulary protocol expects `10294` tokens.
- [x] (2026-07-17 10:40 +08:00) Moved active recovery runs to `D:\V13_rootcause_recovery_20260717` because `E:` is exFAT and Windows reports `HealthStatus=Warning`, `OperationalStatus=Full Repair Needed`.
- [x] (2026-07-18 12:53 +08:00) Completed recovered matched DFS/RCM AR training on `D:`. DFS recovered from the E-drive best checkpoint and finished with best val CE `0.5013600627` at epoch `58`; RCM trained from scratch and finished with best val CE `0.5260249617` at epoch `60`.
- [x] (2026-07-18 15:53 +08:00) Fixed `tools/watch_recovered_training_then_eval_on_d.ps1` so checkpoint finite checks pipe Python code through stdin rather than `python -c`, avoiding Windows quoting breakage.
- [x] (2026-07-18 15:56 +08:00) Fixed `tools/eval_recovered_dfs_rcm_ar_on_d.ps1` to evaluate recovered AR checkpoints with `ArMaxSeqLen=1536`, matching the checkpoint position embeddings.
- [x] (2026-07-18 16:00 +08:00) Reran fresh D-drive DFS/RCM teacher-forcing evaluation successfully. DFS complex-curved AR token-weighted CE is `1.2586939125`; RCM is `1.3130185280`. Both use identical FSQ patch metrics, so DFS is better on this matched ordering control.
- [x] (2026-07-18 16:13 +08:00) The original BrepARG same-data fallback watcher started the fallback after fresh DFS/RCM evaluation became ready, but the PowerShell wrapper treated Python/tqdm stderr progress output as a terminating `NativeCommandError`. No Python model traceback was produced.
- [x] (2026-07-18 16:19 +08:00) Reproduced the wrapper failure with a minimal Python process that writes only to stderr. The root cause is the combination of Windows PowerShell `$ErrorActionPreference = "Stop"` and `2>&1 | Tee-Object`, not the staged data or CUDA model initialization.
- [x] (2026-07-18 16:25 +08:00) Patched `tools/run_breparg_same_data_fallback_on_d.ps1` with `Invoke-NativeLogged`, which uses `Start-Process` and separate live stdout/stderr log files for VQ-VAE, sequence building, AR training, and generation.
- [x] (2026-07-18 16:26 +08:00) Found and patched a second resume bug in `Get-LatestWeight`: with `-LiteralPath`, PowerShell `-Include` did not reliably filter files, so a stale 12-byte `train_vqvae.log` was selected as a checkpoint after the first failed attempt.
- [x] (2026-07-18 16:28 +08:00) Restarted the D-drive BrepARG same-data fallback. Active parent PID is `23432`; active VQ-VAE Python PID is `14760`. The process has entered `BrepARG\train_vqvae.py` after staging completed.
- [x] (2026-07-18 16:30 +08:00) Verified the corrected run is actively training: VQ-VAE epoch 1/160 reached about 10% (`481/4735` batches), with finite loss values and a sampled GPU utilization of `96%`. No stderr traceback is present.
- [x] (2026-07-19 15:13 +08:00) Checked the D-drive fallback after the overnight run. No training process was active. The VQ-VAE run reached epoch 73, then failed during `torch.save` because D: was full; this was a disk-space failure, not a model numerical failure.
- [x] (2026-07-19 15:14 +08:00) Verified `abc_se_vqvae_best.pt` is loadable and finite. It records completed epoch `70` with best validation reconstruction loss `0.0002403665`. The partially written `abc_se_vqvae_epoch_73.pt` is corrupt and was not used.
- [x] (2026-07-19 15:15 +08:00) Deleted 35 non-best BrepARG VQ-VAE epoch checkpoints under `vqvae_3060_safe_len1536_bs4_20260717_d\same_data_abc`, freeing about `22.49 GiB`. Kept only `abc_se_vqvae_best.pt`.
- [x] (2026-07-19 15:18 +08:00) Restarted the fallback after disk cleanup. It reused the existing staged data and `abc_se_vqvae_best.pt`, then entered `BrepARG\2sequence.py`. Active parent PID is `28416`; active sequence Python PID is `50468`.
- [x] (2026-07-19 15:32 +08:00) Confirmed sequence building completed and saved `breparg_same_data_sequences.pkl`. The same fallback parent PID `28416` is now running `BrepARG\train_ar.py` as Python PID `48836`.
- [x] (2026-07-19 15:39 +08:00) Confirmed BrepARG same-data AR epoch 1 completed with validation CE `4.960481` and perplexity `142.66`. It saved `epoch_1.pt`, `abc_ar_vqvae_best_model.pt`, and `abc_ar_vqvae_best_model_hf`; each `.pt` checkpoint is about `107 MiB`, so AR checkpoint growth is much smaller than the prior VQ-VAE checkpoint growth.
- [x] (2026-07-20 14:27 +08:00) Completed an auditable BrepARG same-data fallback generation/evaluation pass from the trained checkpoints. The AR run finished all 80 epochs with best validation CE `0.871925` at epoch `77`. Generation with checkpoint-aware `max_seq_len=1536`, CPU joint optimization, and Windows serial STEP/STL writing produced `92` STEP and `92` STL files before one OCC/joint-optimization candidate stopped making progress; the process was stopped deliberately after the artifact count stayed at `92/100` for more than 8 minutes.
- [x] (2026-07-20 14:27 +08:00) Rendered/validated the `92` BrepARG same-data outputs. `91/92` STEP files were readable, `75/92` were BRep-valid, `91/92` had closed-solid entity structure, and `91/92` PNG previews were saved. The normalized baseline audit reports only `5/92` outputs meet the complex threshold (`advanced_faces >= 12 OR edge_curves >= 20`), only `3/92` are both complex and BRep-valid closed, and `0/92` pass the strict quality gate because all outputs are primitive-like or otherwise rejected.
- [x] (2026-07-20 14:32 +08:00) Copied the small D-drive BrepARG same-data fallback reports back into the local suite and refreshed `local_runs\complex_curved_rootcause_suite_20260715\suite_status.json/md`. The local suite now marks `breparg_same_data_fallback` as complete with `brep_valid=75` and `complex_valid_closed=3`.
- [x] (2026-07-20 14:32 +08:00) Fixed the suite audit next-action logic so an already complete same-data fallback is not suggested again when official BrepARG weights are incompatible. The current local next actions are now only: resume FSQ capacity candidate bookkeeping, rebuild full DFS/RCM sequence packages, and train matched full DFS/RCM AR branches.
- [ ] The D-drive fresh DFS/RCM eval has not yet been copied back into the E-drive suite. Do not write high-frequency or large outputs to `E:` until it is repaired; treat E as read-mostly.
- [ ] The final root-cause conclusion is not yet closed. Current evidence supports a multi-factor diagnosis rather than "FSQ only".

## Surprises & Discoveries

- Observation: The FSQ-capacity watcher marked the local run incomplete even though training printed `DONE {'vqvae': 'PASS'}`.
  Evidence: `local_runs\complex_curved_rootcause_suite_20260715\experiments\01a_train_fsq_capacity_candidate\logs\fsq_capacity_resume_20260715_165543.out.log` shows early stop and `DONE`, but `fsq_capacity_completion_check.json` reports `train_report_missing`. The E-drive suite later contains a completed `train_report.json`, so the scientific FSQ-capacity evaluation exists there.
- Observation: Increasing FSQ codebook capacity from the current `(8,8,8,16)` / `8192` setup to `(16,16,8,8)` / `16384` did not improve the overall complex-curved Chamfer p95.
  Evidence: `fsq_capacity_comparison.json` reports baseline p95 `0.15012167990207662`, candidate p95 `0.15641817450523376`, and capacity signal `inconclusive`.
- Observation: True-token reconstruction fails before free-running generation.
  Evidence: The current teacher-forcing diagnostic selected 50 grammar-valid complex-curved records, saved only 27 STEP files from true tokens, and only 9 were strict BRep-valid.
- Observation: Original BrepARG sampling logic is not enough to recover complexity under the current compatible V13 weights.
  Evidence: The BrepARG logic run saved 100 outputs, but median faces were `6`, median edges were `12`, and only about `11%` were complex.
- Observation: The D-drive eval watcher failure was a script invocation bug, not a bad checkpoint.
  Evidence: After changing the finite check from `python -c $code` to stdin, both recovered checkpoints loaded on CPU and all 101 floating tensors were finite.
- Observation: The recovered DFS/RCM eval must use `max_seq_len=1536`.
  Evidence: The recovered checkpoints were trained with `NS_AR_MAX_SEQ_LEN=1536`; evaluating with `2048` can select sequences longer than the model position embedding table.
- Observation: The first D-drive BrepARG fallback failure was a PowerShell stream-handling failure.
  Evidence: A minimal Python command that exited with code `0` but wrote one line to stderr was converted into a `NativeCommandError` under the same `$ErrorActionPreference = "Stop"` and `2>&1 | Tee-Object` pattern.
- Observation: A stale log file was incorrectly accepted as a VQ-VAE checkpoint after the wrapper failure.
  Evidence: `Get-LatestWeight` returned `train_vqvae.log`; `2sequence.py` then reported `invalid load key, '\xff'` and failed because the loaded model was `None`.
- Observation: BrepARG same-data VQ-VAE training on D: produced many redundant epoch checkpoints and filled the disk before epoch 160.
  Evidence: D: free space reached about `3 MiB`; `torch.save` failed at epoch 73 with `PytorchStreamWriter failed writing file`. The best checkpoint from epoch 70 loaded successfully and was finite.
- Observation: BrepARG same-data AR checkpoints are comparatively small enough to keep the current run alive.
  Evidence: After epoch 1, `epoch_1.pt` and `abc_ar_vqvae_best_model.pt` are each about `107 MiB`, and the HuggingFace best directory is about `35.67 MiB`; D: still has about `22.11 GiB` free.
- Observation: The trained BrepARG same-data baseline can now write STEP/STL files on Windows after the generation fixes, but the free-running samples still collapse toward simple topology.
  Evidence: The fixed generation pass saved `92` STEP/STL files; the quality audit found median `6` advanced faces and median `12` edge curves, only `5/92` complex-by-entity outputs, and `0/92` strict quality acceptances.
- Observation: One generated BrepARG candidate can keep CPU busy in OCC/joint optimization without producing new files or logs.
  Evidence: The generation process stayed alive with increasing CPU time, but STEP/STL count remained `92` and log tail remained unchanged for more than 8 minutes, so it was stopped and the `92` saved outputs were audited.

## Decision Log

- Decision: Keep treating generation-time quality gates as final selection and reporting tools, not the primary model fix.
  Rationale: True-token reconstruction fails before sampling, so filtering bad free-running outputs cannot fix the representation and topology bottlenecks.
  Date/Author: 2026-07-18 / Codex
- Decision: Do not claim FSQ is the sole root cause.
  Rationale: FSQ has a heavy-tail surface problem, but increasing one capacity setting did not improve overall p95, and matched ordering/AR still changes complex-curved CE.
  Date/Author: 2026-07-18 / Codex
- Decision: Use the same-data BrepARG fallback as the fair baseline after official weights are found incompatible.
  Rationale: Official weights were tried first, but the AR embedding shape `[7222, 256]` does not match the current local token protocol. A fair comparison now requires training BrepARG on the same data split and evaluating with the same face/edge/quality protocol.
  Date/Author: 2026-07-18 / Codex
- Decision: Keep active training and high-frequency checkpoint writes on `D:` until `E:` is repaired.
  Rationale: Windows reports `E:` as `Full Repair Needed`; `D:` is NTFS and healthy. The D-drive recovery root already contains the fresh DFS/RCM checkpoints and eval.
  Date/Author: 2026-07-18 / Codex
- Decision: Evaluate recovered DFS/RCM AR with `ArMaxSeqLen=1536`.
  Rationale: The checkpoints were trained with max sequence length 1536. Evaluating with 2048 can select incompatible long sequences and fail for the wrong reason.
  Date/Author: 2026-07-18 / Codex
- Decision: Use `Start-Process` with separate stdout and stderr files for long-running BrepARG subprocesses.
  Rationale: Windows PowerShell 5.1 can turn native stderr into an error record when stderr is merged into the pipeline under `ErrorActionPreference=Stop`; tqdm writes progress to stderr, so `2>&1 | Tee-Object` is unsafe here.
  Date/Author: 2026-07-18 / Codex
- Decision: Accept only files with `.pt` or `.pth` extensions and size greater than 1 MiB as fallback checkpoints.
  Rationale: The prior wrapper failure left a tiny `.log` file, and the old `-Include` query with `-LiteralPath` did not provide a reliable extension filter.
  Date/Author: 2026-07-18 / Codex
- Decision: Continue the BrepARG baseline from the finite epoch-70 best VQ-VAE checkpoint instead of restarting VQ-VAE training.
  Rationale: The failed epoch-73 save was caused by disk exhaustion; epoch 70 was already a strong best checkpoint, and continuing to sequence/AR is the minimum-change path to complete the same-data baseline.
  Date/Author: 2026-07-19 / Codex
- Decision: Keep only `abc_se_vqvae_best.pt` from the BrepARG VQ-VAE stage during the D-drive recovery run.
  Rationale: Each epoch checkpoint is about `0.645 GiB`; retaining dozens of them leaves too little space for sequence and AR artifacts. The baseline only needs the best VQ-VAE checkpoint.
  Date/Author: 2026-07-19 / Codex
- Decision: Treat the `92` saved BrepARG same-data outputs as the current auditable baseline rather than restarting generation for exactly 100 samples.
  Rationale: The remaining gap to 100 was caused by a long-running OCC/joint-optimization candidate after a large enough sample was already saved. The audit signal is decisive: complexity is only `5/92` and strict acceptance is `0/92`, so spending more time to force eight additional samples is unlikely to change the conclusion.
  Date/Author: 2026-07-20 / Codex

## Outcomes & Retrospective

The user asked whether generation-time filtering is enough and whether the FSQ curved-surface experiment had been done. The answer is now evidence-based: filtering is not enough; FSQ-only and true-token teacher-forcing diagnostics have been run; FSQ has heavy-tail reconstruction failures; a single higher-capacity FSQ change did not solve overall p95; DFS ordering is better than RCM in the matched recovered eval; original BrepARG logic also collapses toward simple outputs; official BrepARG weights were found but incompatible; and the same-data BrepARG fallback has now been trained and audited. The same-data BrepARG baseline can write files, but it still collapses to mostly primitive-like/simple topology under the current protocol.

The recovery work also found and fixed two orchestration bugs: a Windows quoting issue in the finite-check watcher and a max-sequence-length mismatch in the recovered eval script. These were script-level issues, not model failures.

## Context and Orientation

This repository contains the V13 BrepARG recovery code. The upstream BrepARG checkout lives in `BrepARG/`. V13 model and training improvements live in `breparg_improvements/`. Operational scripts live in `tools/`. Long-running experiment outputs are intentionally outside the lightweight source package.

Key local roots:

- `local_runs\complex_curved_rootcause_suite_20260715`: local root-cause control suite and scripts.
- `E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715`: external SSD copy of the suite. It has the freshest completed suite audit from 2026-07-17, but the drive is unhealthy.
- `D:\V13_rootcause_recovery_20260717`: safe recovery root for active training and fresh eval after the E-drive warning.

Important terms:

FSQ VQ-VAE means the model that turns surface and edge geometry patches into discrete tokens and reconstructs patches from those tokens. If this stage blurs complex curved patches, the AR model cannot generate good geometry later.

AR means the autoregressive token model. Teacher forcing means evaluating the AR model on the real next token at each step instead of sampling from its own previous predictions. If teacher-forcing CE is high on complex curved shapes, the AR model is weak even before free-running exposure bias.

RCM and DFS are sequence ordering strategies. RCM is the current graph-aware ordering branch; DFS is a depth-first traversal control. Comparing them with identical VQ-VAE, data, and AR hyperparameters isolates the ordering variable.

BRep-valid means OpenCascade can read the generated STEP/B-rep as a valid boundary representation. A closed shell means the generated faces form a closed solid-like shell. The complexity rule used in these diagnostics is `faces >= 12 OR edges >= 20`.

Current protocol caps are `max faces = 50` and `max edges = 150`. Inputs above those caps are outside the current model protocol. Cases near 50 faces are allowed but should be reported separately because they are boundary/high-difficulty cases.

## Plan of Work

First, preserve the current evidence and avoid restarting completed jobs. The FSQ capacity candidate is already trained and evaluated on the E-drive suite; do not restart it unless a new one-variable FSQ experiment is explicitly chosen.

Second, keep the completed D-drive BrepARG same-data fallback as the fair baseline. The watcher conditions were satisfied on 2026-07-18, but the first launch exposed two Windows PowerShell orchestration bugs. The corrected `tools\run_breparg_same_data_fallback_on_d.ps1` trained BrepARG SE VQ-VAE through epoch 73 before D: filled during checkpoint saving. The finite best VQ-VAE checkpoint is epoch 70, `abc_se_vqvae_best.pt`, with validation reconstruction loss `0.0002403665`. After deleting redundant epoch checkpoints, the corrected fallback resumed, completed original BrepARG sequence building, and trained BrepARG AR through 80 epochs. Fresh generation and quality validation now exist under `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\generated_3060_safe_len1536_bs4_20260717_d`.

Third, use the refreshed local suite status as the work queue. The comparison should not mix current-method results, BrepARG original-logic-with-V13-weights results, and BrepARG same-data trained baseline results without labeling them clearly. As of 2026-07-20 14:32 +08:00, the same-data BrepARG fallback is no longer a pending action; the remaining tracked gaps are FSQ capacity bookkeeping and full DFS/RCM ordering controls.

Fourth, choose the next model intervention based on evidence. Since a simple capacity increase did not fix the overall FSQ p95, likely next model-side experiments are surface-heavy FSQ loss/sampling, topology reconstruction/OCC assembly diagnostics, and AR curriculum or ordering changes. Run only one variable at a time on the complex-curved subset.

## Concrete Steps

To check current D-drive recovery status from the repository root:

    cd D:\luolin\V13
    Get-Content D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md
    Get-Content D:\V13_rootcause_recovery_20260717\logs\watch_then_start_breparg_same_data_on_d.log -Tail 40
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_breparg_same_data_fallback_on_d|train_vqvae.py|train_ar.py|2sequence.py|generate_brep.py|watch_then_start_breparg_same_data_on_d' } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine

To inspect the completed corrected BrepARG fallback:

    Get-Content D:\V13_rootcause_recovery_20260717\logs\run_breparg_same_data_fallback_on_d.resume_after_disk_cleanup_20260719_151543.out.log -Tail 80
    Get-Content D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\sequence_3060_safe_len1536_bs4_20260717_d\build_sequence.log -Tail 80
    Get-Content D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\sequence_3060_safe_len1536_bs4_20260717_d\build_sequence.log.err -Tail 80
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_breparg_same_data_fallback_on_d|train_vqvae.py|train_ar.py|2sequence.py|generate_brep.py' } | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine

If the watcher has not started BrepARG fallback and the user wants to start it manually after verifying disk and GPU:

    cd D:\luolin\V13
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_breparg_same_data_fallback_on_d.ps1

To avoid duplicate fallback starts, first verify no fallback process is already running and no manifest exists:

    Test-Path D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\same_data_breparg_fallback_manifest.json
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_breparg_same_data_fallback_on_d|train_vqvae.py|train_ar.py|2sequence.py|generate_brep.py' }

After fallback generation, refresh the D-drive fallback audit with the fixed generation artifacts if needed:

    cd D:\luolin\V13
    C:\Users\YU\.conda\envs\brepgen_env\python.exe tools\validate_breparg_generated_directory.py --run-dir D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\generated_3060_safe_len1536_bs4_20260717_d --manifest-output D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\generated_3060_safe_len1536_bs4_20260717_d\quality_check\step_quality_manifest.jsonl --summary-output D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\generated_3060_safe_len1536_bs4_20260717_d\quality_check\step_quality_summary.json --timeout-sec 45
    C:\Users\YU\.conda\envs\brepgen_env\python.exe tools\audit_breparg_baseline_outputs.py --run-dir D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\generated_3060_safe_len1536_bs4_20260717_d --output D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_summary_3060_safe_len1536_bs4_20260717_d.json --markdown-output D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_summary_3060_safe_len1536_bs4_20260717_d.md --manifest-output D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_manifest_3060_safe_len1536_bs4_20260717_d.jsonl --min-faces 12 --min-edges 20 --max-faces 45 --max-edges 120

Then copy only summary reports back into the repository notes or E-drive suite after `E:` is repaired. Do not mirror active training outputs to `E:` while it reports `Full Repair Needed`.

To rerun the recovered DFS/RCM teacher-forcing eval if needed:

    cd D:\luolin\V13
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\eval_recovered_dfs_rcm_ar_on_d.ps1 -ArMaxSeqLen 1536

Expected successful output includes:

    "status": "VERIFIED", "selected": 50, "output": "D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_teacher_forcing"
    "status": "VERIFIED", "selected": 50, "output": "D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\rcm_teacher_forcing"
    D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md

## Validation and Acceptance

The current validation commands that already passed on 2026-07-18 are:

    powershell -NoProfile -ExecutionPolicy Bypass -File tools\eval_recovered_dfs_rcm_ar_on_d.ps1 -ArMaxSeqLen 1536

Observed DFS/RCM acceptance:

- `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_teacher_forcing\complex_curved_diagnostics_report.json` exists and has `status=VERIFIED`.
- `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\rcm_teacher_forcing\complex_curved_diagnostics_report.json` exists and has `status=VERIFIED`.
- `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\checkpoint_finite_check.json` shows both DFS and RCM checkpoints are loadable and finite.
- The summary reports DFS token-weighted CE `1.2586939125` and RCM token-weighted CE `1.3130185280`.

The BrepARG fallback acceptance is now:

- `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\same_data_breparg_fallback_manifest.json` exists.
- `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_summary.json` exists.
- The quality summary reports STEP readability, BRep validity, closed-shell count, complexity count, and complex-valid-closed count.
- Fixed-generation artifacts include `92` STEP files, `92` STL files, `91` PNG previews, and a `92` row quality manifest.
- No fallback checkpoint contains non-finite tensors.

## Idempotence and Recovery

The recovered DFS/RCM eval is idempotent: rerunning `tools\eval_recovered_dfs_rcm_ar_on_d.ps1 -ArMaxSeqLen 1536` overwrites report JSON/Markdown files under `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval` without changing model checkpoints.

The BrepARG fallback script is partially resumable. It checks for an existing VQ-VAE weight, existing sequence package, and existing AR weight before training each stage. It removes and recreates only the generation output folder before generating the final 100 samples. Do not launch two fallback instances at once.

Do not delete or overwrite:

- `D:\V13_rootcause_recovery_20260717\ar_train_outputs\ar_dfs_matched_20260715\ar_best.pt`
- `D:\V13_rootcause_recovery_20260717\ar_train_outputs\ar_rcm_matched_20260715\ar_best.pt`
- `E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715`
- `local_runs\complex_curved_rootcause_suite_20260715`

If `E:` remains `Full Repair Needed`, continue writing active outputs to `D:` and copy back only after the drive is repaired or after the user explicitly chooses a safe archive/copy plan.

## Artifacts and Notes

Key evidence artifacts:

- Current FSQ-only report: `local_runs\complex_curved_rootcause_suite_20260715\experiments\00_fsq_only_patch_metrics\complex_curved_diagnostics_report.json`
- Current teacher-forcing report: `local_runs\complex_curved_rootcause_suite_20260715\experiments\01_teacher_forcing_true_token_reconstruction\complex_curved_diagnostics_report.json`
- FSQ capacity comparison: `E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\fsq_capacity_comparison.json`
- E-drive suite status: `E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\suite_status.json`
- D-drive recovered DFS/RCM finite check: `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\checkpoint_finite_check.json`
- D-drive recovered DFS/RCM teacher-forcing summary: `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md`
- BrepARG fallback watcher log: `D:\V13_rootcause_recovery_20260717\logs\watch_then_start_breparg_same_data_on_d.log`

Most important current readings:

- FSQ-only baseline Chamfer p95: `0.1501216799`
- Higher-capacity FSQ Chamfer p95: `0.1564181745`
- Current true-token reconstruction strict BRep valid: `9/50`
- Current AR teacher-forcing CE: `0.7467434714`
- Recovered DFS teacher-forcing CE: `1.2586939125`
- Recovered RCM teacher-forcing CE: `1.3130185280`
- BrepARG original logic complex fraction with compatible current weights: about `11%`

## Interfaces and Dependencies

Scripts touched or used in this status update:

- `tools\watch_recovered_training_then_eval_on_d.ps1` checks recovered DFS/RCM checkpoint presence and finite tensors, then launches D-drive eval.
- `tools\eval_recovered_dfs_rcm_ar_on_d.ps1` runs complex-curved DFS/RCM teacher-forcing diagnostics from recovered checkpoints. It now accepts `-ArMaxSeqLen`, defaulting to `1536`.
- `tools\complex_curved_diagnostics.py` selects complex-curved records, computes FSQ patch MSE/Chamfer, computes AR teacher-forcing CE, and optionally reconstructs true token sequences.
- `tools\watch_then_start_breparg_same_data_on_d.ps1` waits for fresh recovered eval and GPU idle, then starts BrepARG same-data fallback.
- `tools\run_breparg_same_data_fallback_on_d.ps1` stages same-data inputs on D, trains BrepARG SE VQ-VAE, builds original BrepARG sequences, trains BrepARG AR, generates STEP/PNG outputs, and audits them. It now skips staging if the staged data has already been verified, logs long-running subprocess stdout/stderr separately, and treats only `.pt`/`.pth` files larger than 1 MiB as checkpoints.

Required Python environment for these scripts:

- `C:\Users\YU\.conda\envs\brepgen_env\python.exe`
- Modules used by diagnostics and fallback include `torch`, `numpy`, `diffusers`, `transformers`, `OCC`, `occwl`, `tensorboard`, `shutup`, and `tqdm`.

Revision note 2026-07-18 16:00 +08: Created this current-status ExecPlan after checking the latest local, E-drive, and D-drive artifacts. The note records that FSQ-only, teacher-forcing, FSQ-capacity, original BrepARG logic, and recovered DFS/RCM ordering diagnostics have produced evidence; BrepARG same-data fallback remains the main missing baseline.

Revision note 2026-07-18 16:28 +08: Updated the plan after reproducing and fixing two BrepARG fallback orchestration bugs: PowerShell native stderr handling and unreliable checkpoint discovery after a failed stage. The corrected fallback is active on D: with parent PID `23432` and VQ-VAE PID `14760`; completion and quality audit remain pending.

Revision note 2026-07-19 15:18 +08: Updated the plan after the D-drive VQ-VAE run failed from disk exhaustion at epoch 73. Verified the epoch-70 best checkpoint, deleted redundant VQ-VAE epoch checkpoints to free space, and resumed the fallback into `2sequence.py` with parent PID `28416` and sequence PID `50468`.

Revision note 2026-07-19 15:32 +08: Updated the plan after checking the live fallback. Sequence building has completed; the fallback is training BrepARG AR with PID `48836`, `max_seq_len=1536`, `batch_size=4`, `learning_rate=5e-5`, 9,129 training groups, and 912 validation groups after length filtering. D: has about `22.35 GiB` free, so continue monitoring rather than restarting or changing parameters mid-run.

Revision note 2026-07-19 15:39 +08: Added the first BrepARG same-data AR validation result. Epoch 1 completed with validation CE `4.960481`; the first `.pt` checkpoint size is about `107 MiB`, so current disk risk is acceptable compared with the earlier VQ-VAE checkpoint burst.

Revision note 2026-07-20 14:28 +08: Updated after fixing and rerunning BrepARG same-data generation on Windows. Added checkpoint `max_seq_len` recovery, CPU joint optimization fallback, serial STEP/STL writing, and a batch STEP quality/PNG validation helper. The auditable fallback result is `92` STEP, `92` STL, `91` PNG, `75` BRep-valid files, but only `5/92` complex-by-entity outputs and `0/92` strict quality acceptances.

Revision note 2026-07-20 14:32 +08: Refreshed the local suite audit after copying D-drive BrepARG fallback reports into `local_runs`. Fixed `tools\audit_complex_curved_control_suite.py` so complete same-data fallback results suppress the stale "Run BrepARG same-data fallback" next action.
