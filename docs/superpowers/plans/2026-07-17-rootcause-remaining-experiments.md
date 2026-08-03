# Root-Cause Remaining Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining complex-curved root-cause experiments without mixing training jobs, storage risk, or incompatible evaluation protocols.

**Architecture:** Keep active training and high-frequency outputs on `D:\V13_rootcause_recovery_20260717` while `E:` reports `Full Repair Needed`. Let the existing watchers serialize DFS/RCM training, DFS/RCM teacher-forcing evaluation, and same-data BrepARG fallback; only intervene if a watcher exits with a concrete failure.

**Tech Stack:** PowerShell orchestration, `C:\Users\YU\.conda\envs\brepgen_env\python.exe`, PyTorch/Transformers, local `breparg_improvements`, original `BrepARG`, OpenCascade/occwl for STEP audits.

---

### Task 1: Let Matched DFS/RCM AR Recovery Finish

**Files:**
- Read: `D:\V13_rootcause_recovery_20260717\ar_train_outputs\ar_dfs_matched_20260715\ar_train.log`
- Read: `D:\V13_rootcause_recovery_20260717\logs\resume_matched_dfs_rcm_ar_on_d.log`
- Read: `D:\V13_rootcause_recovery_20260717\logs\watch_recovered_training_then_eval_on_d.log`
- Output expected: `D:\V13_rootcause_recovery_20260717\ar_train_outputs\ar_rcm_matched_20260715\ar_best.pt`

- [ ] **Step 1: Confirm current DFS run is still active**

Run:

```powershell
Get-Process -Id 34092,10600 -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,CPU,WorkingSet,StartTime
Get-Content D:\V13_rootcause_recovery_20260717\ar_train_outputs\ar_dfs_matched_20260715\ar_train.log -Tail 40
nvidia-smi
```

Expected: wrapper PID `34092` and Python PID `10600` exist while DFS is training; GPU is busy; no traceback or OOM appears.

- [ ] **Step 2: Do not start another GPU job while DFS/RCM is active**

Run:

```powershell
Get-Content D:\V13_rootcause_recovery_20260717\logs\watch_recovered_training_then_eval_on_d.log -Tail 30
```

Expected: `training_alive=True` until wrapper exits. Do not manually start BrepARG fallback during this period.

- [ ] **Step 3: After DFS completes, confirm RCM starts**

Run:

```powershell
Test-Path D:\V13_rootcause_recovery_20260717\ar_train_outputs\ar_rcm_matched_20260715\ar_train.log
Get-Content D:\V13_rootcause_recovery_20260717\logs\resume_matched_dfs_rcm_ar_on_d.log -Tail 80
```

Expected: log contains `finished dfs` and `starting rcm`, and RCM `ar_train.log` exists.

### Task 2: Verify Fresh DFS/RCM Teacher-Forcing Evaluation

**Files:**
- Script: `tools\eval_recovered_dfs_rcm_ar_on_d.ps1`
- Watcher: `tools\watch_recovered_training_then_eval_on_d.ps1`
- Output: `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md`
- Output: `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\checkpoint_finite_check.json`

- [ ] **Step 1: Wait for watcher to detect training completion**

Run:

```powershell
Get-Content D:\V13_rootcause_recovery_20260717\logs\watch_recovered_training_then_eval_on_d.log -Tail 80
```

Expected: after wrapper exits, log contains `training finished and checkpoints are finite; starting D-drive evaluation`.

- [ ] **Step 2: Confirm finite checkpoint report**

Run:

```powershell
Get-Content D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\checkpoint_finite_check.json
```

Expected: both `dfs` and `rcm` rows have `finite_model: true`, non-null epoch, and finite `best_val_ce`.

- [ ] **Step 3: Confirm DFS/RCM eval freshness**

Run:

```powershell
$root = "D:\V13_rootcause_recovery_20260717"
Get-Item "$root\ar_complex_curved_eval\dfs_teacher_forcing\complex_curved_diagnostics_report.json",
         "$root\ar_complex_curved_eval\rcm_teacher_forcing\complex_curved_diagnostics_report.json",
         "$root\ar_train_outputs\ar_dfs_matched_20260715\ar_best.pt",
         "$root\ar_train_outputs\ar_rcm_matched_20260715\ar_best.pt" |
  Select-Object FullName,LastWriteTime
Get-Content "$root\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md"
```

Expected: both report timestamps are later than their corresponding checkpoints; summary compares DFS and RCM on the same complex-curved protocol.

### Task 3: Let Same-Data BrepARG Fallback Run On D

**Files:**
- Watcher: `tools\watch_then_start_breparg_same_data_on_d.ps1`
- Runner: `tools\run_breparg_same_data_fallback_on_d.ps1`
- Output root: `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback`
- Output: `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\same_data_breparg_fallback_manifest.json`

- [ ] **Step 1: Confirm watcher conditions**

Run:

```powershell
Get-Content D:\V13_rootcause_recovery_20260717\logs\watch_then_start_breparg_same_data_on_d.log -Tail 80
```

Expected: before launch, entries show `summary_ready=True`, `fresh_dfs_eval=True`, `fresh_rcm_eval=True`, `blocking_processes=0`, and GPU memory below threshold.

- [ ] **Step 2: Confirm staged same-data input**

Run after fallback starts:

```powershell
Get-Content D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\data_staged\staging_manifest.json
```

Expected: counts are `train=10000`, `val=1000`, `test=1000`; split path points to `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\data_staged\same_data_split.pkl`.

- [ ] **Step 3: Monitor BrepARG VQ-VAE, sequence, AR, generation**

Run:

```powershell
Get-Content D:\V13_rootcause_recovery_20260717\logs\run_breparg_same_data_fallback_on_d.out.log -Tail 120
Get-ChildItem D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback -Recurse -File |
  Where-Object { $_.Name -match 'abc_se_vqvae_best|abc_ar_vqvae_best|quality_summary|manifest' } |
  Select-Object FullName,Length,LastWriteTime
```

Expected: VQ-VAE best, AR best, generated output directory, quality summary, and manifest are produced without reading staged training data from E during training.

### Task 4: Final Cross-Method Comparison

**Files:**
- Current method summary: `E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\complex_curved_diagnostic_summary.md`
- FSQ capacity comparison: `E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\fsq_capacity_comparison.md`
- DFS/RCM summary: `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md`
- BrepARG fallback summary: `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_summary.md`
- Final note: `D:\V13_rootcause_recovery_20260717\rootcause_final_comparison_20260717.md`

- [ ] **Step 1: Gather all summaries**

Run:

```powershell
Get-Content E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\complex_curved_diagnostic_summary.md
Get-Content E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\fsq_capacity_comparison.md
Get-Content D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md
Get-Content D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_summary.md
```

Expected: all four files exist and describe comparable protocols.

- [ ] **Step 2: Write final comparison**

Create `D:\V13_rootcause_recovery_20260717\rootcause_final_comparison_20260717.md` with these sections:

```markdown
# Root-Cause Final Comparison 2026-07-17

## Evidence Table

| Axis | Result | Interpretation |
| --- | --- | --- |
| FSQ-only reconstruction | ... | ... |
| Higher FSQ capacity | ... | ... |
| Teacher-forcing current AR | ... | ... |
| DFS vs RCM ordering | ... | ... |
| BrepARG original logic | ... | ... |
| BrepARG same-data fallback | ... | ... |

## Conclusion

Do not claim FSQ is the sole root cause unless the final evidence proves it. Distinguish FSQ/OCC reconstruction heavy-tail failures, AR/order sensitivity, sequence-length/topology effects, and BrepARG baseline behavior.

## Next Training Recommendation

State the next model change only after the above evidence table is complete.
```

Expected: conclusion explicitly separates FSQ, AR, ordering, sequence length/topology, and baseline effects.

### Task 5: Copy-Back After E Drive Repair

**Files:**
- Existing plan: `D:\V13_rootcause_recovery_20260717\safe_copy_plan_20260717.md`
- Output archive: `D:\V13_copy_stage_20260717\V13_rootcause_recovery_20260717.tar.zst`

- [ ] **Step 1: Check E drive health**

Run:

```powershell
Get-Volume -DriveLetter E | Select-Object DriveLetter,FileSystem,HealthStatus,OperationalStatus
```

Expected: do not write back to E unless `OperationalStatus` no longer says `Full Repair Needed`.

- [ ] **Step 2: Pack D recovery output**

Run only after active jobs finish:

```powershell
New-Item -ItemType Directory -Force 'D:\V13_copy_stage_20260717' | Out-Null
tar -acf 'D:\V13_copy_stage_20260717\V13_rootcause_recovery_20260717.tar.zst' -C 'D:\' 'V13_rootcause_recovery_20260717'
Get-Item 'D:\V13_copy_stage_20260717\V13_rootcause_recovery_20260717.tar.zst' |
  Select-Object FullName,Length,LastWriteTime
```

Expected: one archive exists on D for safe copy-back or external transfer.
