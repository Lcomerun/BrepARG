# Safe Copy Plan After E Drive Recovery

- Created: 2026-07-17T10:46:58
- Suite root: 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715'
- Recovery root: 'D:\V13_rootcause_recovery_20260717'
- Stage root: 'D:\V13_copy_stage_20260717'
- E: FileSystem='exFAT', HealthStatus='Warning', OperationalStatus='Full Repair Needed'

## Current Recommendation

1. Keep active training outputs on 'D:' until 'E:' is repaired.
2. For 'E:' -> 'D:' recovery copies, prefer creating archives on 'D:' so the unstable drive is read-only.
3. For 'D:' -> 'E:' copy-back, wait until 'chkdsk E: /f' or equivalent repair completes.
4. 'robocopy /MT' helps many-file directory copies. For one huge archive, '/J' matters more than '/MT'.
5. Do not use '/MIR' or '/PURGE' for this recovery work; both can delete destination files.

## Size Inventory

| path | exists | files | dirs | GB |
| --- | ---: | ---: | ---: | ---: |
| 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715' | true | 47286 | 166 | 69.979 |
| 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\experiments' | true | 47236 | 163 | 69.979 |
| 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\scripts' | true | 27 | 0 | 0 |
| 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\experiments\02_dfs_rcm_ordering' | true | 32512 | 48 | 16.961 |
| 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\experiments\03b_breparg_same_data_training_fallback' | true | 12065 | 43 | 10.283 |
| 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\experiments\04_breparg_logic_generation_baseline' | true | 2140 | 38 | 0.228 |
| 'D:\V13_rootcause_recovery_20260717' | true | 14 | 3 | 0.409 |
| 'D:\luolin\V13\ABC\processed\abc_parsed_full_archives' | true | 101 | 0 | 162.399 |
| 'D:\luolin\V13\ABC\processed\train_outputs\ubuntu' | true | 3 | 0 | 1.723 |

## Option A: Read E Once, Pack Suite To D

Use this if the remaining source is many small files under the suite on 'E:' and you want a more stable transfer artifact on 'D:'.

~~~powershell
New-Item -ItemType Directory -Force 'D:\V13_copy_stage_20260717' | Out-Null
tar -acf 'D:\V13_copy_stage_20260717\complex_curved_rootcause_suite_20260715.tar.zst' -C 'E:\V13_rootcause_20260715' 'complex_curved_rootcause_suite_20260715'
Get-Item 'D:\V13_copy_stage_20260717\complex_curved_rootcause_suite_20260715.tar.zst' | Select-Object FullName,Length,LastWriteTime
~~~

## Option B: Multi-thread Directory Copy E To D

Use this if you need the directory tree directly instead of a tar/zstd archive. This is restartable and multi-threaded, but slower on many tiny files than a clean archive workflow.

~~~powershell
New-Item -ItemType Directory -Force 'D:\V13_copy_stage_20260717' | Out-Null
robocopy 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715' 'D:\V13_copy_stage_20260717\complex_curved_rootcause_suite_20260715' /E /MT:16 /J /R:2 /W:2 /FFT /NP /TEE /LOG+:'D:\V13_copy_stage_20260717\robocopy_suite_to_stage.log'
if ($LASTEXITCODE -le 7) { "ROBOCOPY_OK_OR_WARN" } else { throw "robocopy failed with exit code $LASTEXITCODE" }
~~~

## Option C: Pack Active D Recovery Output

Use this after the D-drive DFS/RCM recovery training and evaluation finish. It creates one copy-back artifact without touching 'E:' during training.

~~~powershell
New-Item -ItemType Directory -Force 'D:\V13_copy_stage_20260717' | Out-Null
tar -acf 'D:\V13_copy_stage_20260717\V13_rootcause_recovery_20260717.tar.zst' -C 'D:\' 'V13_rootcause_recovery_20260717'
Get-Item 'D:\V13_copy_stage_20260717\V13_rootcause_recovery_20260717.tar.zst' | Select-Object FullName,Length,LastWriteTime
~~~

## Option D: Copy D Recovery Output Back To Repaired E

Run this only after 'E:' no longer reports 'Full Repair Needed'.

~~~powershell
Get-Volume -DriveLetter E | Select-Object DriveLetter,FileSystem,HealthStatus,OperationalStatus
robocopy 'D:\V13_rootcause_recovery_20260717' 'E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\..\D_recovery_20260717' /E /MT:16 /J /R:2 /W:2 /FFT /NP /TEE /LOG+:'D:\V13_copy_stage_20260717\robocopy_recovery_to_repaired_e.log'
if ($LASTEXITCODE -le 7) { "ROBOCOPY_OK_OR_WARN" } else { throw "robocopy failed with exit code $LASTEXITCODE" }
~~~

## Training Throughput Note

The current AR recovery training reads the sequence package from 'D:' and writes checkpoints to 'D:'. It should not be blocked by slow 'E:' copies unless a separate copy job saturates the same CPU, GPU, or system disk. Keep copy jobs off while the GPU job is active if you see training step time increase.
