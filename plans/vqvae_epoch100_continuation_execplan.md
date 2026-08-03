# Continue Stable VQ-VAE Toward Epoch 100

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` in the repository root. The purpose is to continue the verified stable VQ-VAE training run toward absolute epoch 100 while preserving the safeguards that prevented the previous `val=inf` failure. The observable result is a new output directory with a continuation history, a best checkpoint, a final checkpoint, and a report showing whether validation loss keeps improving without overfitting.

## Purpose / Big Picture

The previous stable VQ-VAE run under `E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable` completed 40 requested epochs with finite losses. Its best validation reconstruction loss was about `0.00056`, and its final validation loss was about `0.00055`. The user wants to continue training toward epoch 100 and evaluate whether validation loss can approach `10e-5`, which means `0.00010` if interpreted literally as ten times ten to the minus five.

This plan creates a continuation path rather than restarting blindly. A continuation path means the training script can load an existing VQ-VAE checkpoint before training, keep absolute epoch numbers in the history, save the best checkpoint when validation improves, and also save a final checkpoint so the exact last state is not lost again.

## Progress

- [x] (2026-06-27 10:22 +08:00) Verified the stable VQ-VAE baseline: `epochs_ran=40`, `best_val_recon=0.00056`, `best_epoch=35`, `val_final=0.00055`, all batches finite, and checkpoint loadable.
- [x] (2026-06-27 10:35 +08:00) Read `AGENTS.md` and `PLANS.md`; confirmed this training behavior change requires a maintained ExecPlan.
- [x] (2026-06-27 10:40 +08:00) Inspected current stable output directory and found only `fsq_vqvae_best.pt`, not an epoch-40 final checkpoint.
- [x] (2026-06-27 10:45 +08:00) Wrote failing helper tests for VQ-VAE history summary and absolute continuation epoch calculation. The RED result is two `ImportError` failures for missing `summarize_vqvae_history` and `continuation_epoch_count`.
- [x] (2026-06-27 10:48 +08:00) Implemented pure helper functions for summarizing a previous VQ-VAE history and computing continuation epoch count; the two focused helper tests passed.
- [x] (2026-06-27 10:53 +08:00) Added VQ-VAE resume, lower learning-rate continuation, final checkpoint saving, absolute epoch numbering, and report fields to `breparg_improvements\train.py`.
- [x] (2026-06-27 10:53 +08:00) Added `tools\run_vqvae_continue_to_epoch100.ps1`, which writes to `newscheme_full_vqvae_epoch100` and resumes from the verified stable best checkpoint.
- [x] (2026-06-27 10:54 +08:00) Ran validation: `tests/test_local_pipeline_helpers.py` passed with `16 passed`, Python compilation passed, and the PowerShell parser accepted the continuation script.
- [x] (2026-06-27 10:55 +08:00) Launched the continuation script in the background. Parent PowerShell PID was `18628`, and child Python VQ-VAE PID was `19468`.
- [x] (2026-06-27 11:01 +08:00) Confirmed the continuation wrote its first VQ-VAE record at absolute epoch 40 with finite losses: `train_loss=0.00029`, `val_loss=0.00043`, `best_val_recon=0.00043`, `best_epoch=40`, `finite_train_batches=2227/2227`, `finite_val_batches=118/118`, and `skipped_train_batches=0`.
- [x] (2026-06-27 11:35 +08:00) Confirmed the continuation reached epoch 46 with finite losses and continued validation improvement: `train_loss=0.00023`, `val_loss=0.00040`, `best_val_recon=0.00040`, `best_epoch=46`, and no skipped batches.
- [x] (2026-06-27 11:54 +08:00) Confirmed the continuation reached epoch 49. The best validation loss remains `0.00040` from epoch 46, while epochs 47 through 49 reported `0.00042`, `0.00041`, and `0.00041`. The run remains finite and active.
- [x] (2026-06-27 12:15 +08:00) Confirmed the continuation reached epoch 53. The run remains finite and active; epoch 52 recorded a small best refresh at rounded `0.00040`, and epoch 53 reported `train_loss=0.00021`, `val_loss=0.00042`, and no skipped batches.
- [x] (2026-06-27 12:47 +08:00) Confirmed the continuation reached epoch 59 and refreshed best validation loss to rounded `0.00039`, with `train_loss=0.00019`, `finite_train_batches=2227/2227`, `finite_val_batches=118/118`, and no skipped batches.
- [x] (2026-06-27 13:18 +08:00) Confirmed the continuation reached epoch 64. The run remains finite and active, but best validation remains rounded `0.00039` from epoch 59; the latest epoch reported `train_loss=0.00019`, `val_loss=0.00041`, and `epochs_without_improvement=5`.
- [x] (2026-06-27 13:59 +08:00) Confirmed the continuation reached epoch 72. The run refreshed best validation to rounded `0.00038` at epoch 70, with latest `train_loss=0.00017`, `val_loss=0.00038`, and no skipped batches.
- [x] (2026-06-27 14:40 +08:00) Confirmed the continuation reached epoch 79. The run refreshed best validation to rounded `0.00037` at epoch 73, then remained stable through epoch 79 with no skipped batches.
- [x] (2026-06-27 15:11 +08:00) The continuation stopped early at epoch 85 with `early_stop_reason=patience=12`, `status=VERIFIED`, `best_val_recon=0.00037`, `best_epoch=73`, `val_final=0.00038`, and all batches finite.
- [x] (2026-06-27 15:15 +08:00) Final validation checks passed: no `Infinity` or `NaN` in stable history/report, `fsq_vqvae_best.pt` and `fsq_vqvae_final.pt` both load with PyTorch, local helper tests passed with `16 passed`, and Python compilation passed.

## Surprises & Discoveries

- Observation: The stable baseline output has a best checkpoint but no final epoch-40 checkpoint.
  Evidence: `Get-ChildItem E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable` lists `fsq_vqvae_best.pt`, `vqvae_history.json`, `vqvae_hp_sweep.json`, and `split.pkl`, but no final VQ-VAE checkpoint.
- Observation: Because only the best checkpoint exists, continuation cannot strictly resume from the exact epoch-40 weights.
  Evidence: `vqvae_history.json` says `best_epoch=35`; `fsq_vqvae_best.pt` was last modified before the final history write. The practical continuation source is the best checkpoint, not the final epoch-40 state.
- Observation: The current validation curve is already near a plateau.
  Evidence: epoch 30 had validation about `0.00058`, epoch 35 had about `0.00056`, and epoch 39 had about `0.00055`. The improvement from epoch 30 to 39 is small compared with the improvement from epoch 0 to 10.
- Observation: Restarting from the stable best checkpoint with lower learning rate produced a strong first continuation improvement.
  Evidence: The first continuation record at epoch 40 reported `val_loss=0.00043`, which is lower than the previous stable best `0.00056`. All train and validation batches were finite.
- Observation: The continuation still improves after entering the early-stop-eligible range.
  Evidence: The history reached epoch 46 with `val_loss=0.00040` and `best_epoch=46`, after earlier values of `0.00043`, `0.00044`, `0.00042`, `0.00043`, `0.00042`, and `0.00043`.
- Observation: After epoch 46, validation is showing a possible new plateau rather than a monotonic descent.
  Evidence: epochs 47, 48, and 49 reported validation losses `0.00042`, `0.00041`, and `0.00041`, while training loss continued down to `0.00022`. This is not a collapse, but it means `0.00010` remains uncertain and may require more than simply more epochs.
- Observation: The plateau signal remains visible through epoch 53, with occasional very small rounded improvements.
  Evidence: epochs 50 through 53 reported validation losses `0.00042`, `0.00041`, `0.00040`, and `0.00042`; training loss reached `0.00021`. The process remains numerically stable, but the validation loss is not descending quickly toward `0.00010`.
- Observation: The continuation can still refresh best validation after the plateau, but the slope is shallow.
  Evidence: epoch 57 reached `0.00039`, epoch 59 also reached `0.00039`, and training loss reached `0.00019`. This is a real improvement over `0.00056`, but not yet a trajectory that obviously reaches `0.00010` by epoch 100.
- Observation: Through epoch 64, validation has not clearly moved below the rounded `0.00039` level.
  Evidence: epochs 60 through 64 reported validation losses `0.00040`, `0.00040`, `0.00039`, `0.00039`, and `0.00041`, while `epochs_without_improvement` reached 5. The run is still healthy, but additional epochs alone are showing diminishing returns.
- Observation: The run can still escape short plateaus, but the improvement remains slow.
  Evidence: after plateauing around rounded `0.00039`, epochs 68 and 70 refreshed best validation to rounded `0.00038`. This supports continuing the current run, but does not yet support expecting `0.00010` by epoch 100.
- Observation: By epoch 79, the continuation has improved the stable baseline substantially but still has a large gap to `0.00010`.
  Evidence: best validation went from stable baseline `0.00056` to rounded `0.00037`, a meaningful improvement. However, epoch 73 to 79 stayed around `0.00037` to `0.00039`, so the remaining reduction to `0.00010` would require a much steeper improvement than currently observed.
- Observation: The early stop did the intended job.
  Evidence: epoch 73 had the best rounded validation loss `0.00037`. Epochs 74 through 85 did not improve by at least `min_delta=1e-6`, so `patience=12` stopped the run at epoch 85 instead of wasting the remaining epochs through target epoch 100.

## Decision Log

- Decision: Continue from the stable best checkpoint, but record the continuation as starting at absolute epoch 40.
  Rationale: The exact epoch-40 final weights were not saved. The best checkpoint is the most validation-safe source and is the right basis for a no-overfitting continuation. Absolute epoch numbering keeps the training record readable: new records should be epoch 40 through 99 when targeting epoch 100.
  Date/Author: 2026-06-27 / Codex.
- Decision: Use a new output directory named `newscheme_full_vqvae_epoch100` for the continuation.
  Rationale: The verified stable baseline should remain untouched. The continuation can copy or regenerate the split and write its own `fsq_vqvae_best.pt`, `fsq_vqvae_final.pt`, `vqvae_history.json`, and report.
  Date/Author: 2026-06-27 / Codex.
- Decision: Lower the continuation learning rate from `3e-4` to `1e-4` and use `min_delta=1e-6` with a longer patience window.
  Rationale: The baseline is already stable and near a validation plateau. A lower learning rate reduces the chance of overshooting or numerical instability, while a smaller minimum improvement threshold prevents meaningful small improvements from being ignored.
  Date/Author: 2026-06-27 / Codex.

## Outcomes & Retrospective

The plan is currently in implementation. The most important early conclusion is that the target of `0.00010` validation MSE is ambitious under the current architecture and data pipeline because the loss curve has flattened around `0.00055`. Continuing to epoch 100 is still a useful experiment: if validation loss keeps falling, it justifies longer or staged training; if it stays around `0.00055` while train loss keeps falling, the limiting factor is likely model capacity, sampling, or objective design rather than epoch count alone.

The implementation and training experiment are complete. The continuation improved the stable baseline from `best_val_recon=0.00056` to `best_val_recon=0.00037`, then stopped early at epoch 85 after 12 epochs without sufficient improvement. This is a successful continuation run because it improved validation, remained numerically stable, saved both best and final checkpoints, and avoided wasting the full target-to-100 budget once validation plateaued.

The target `0.00010` was not reached in this run. Based on the observed curve, that target is not a reasonable expectation for the current architecture and training recipe by simply adding epochs. It remains a useful aspirational target for a new plan that changes model capacity, data sampling, loss weighting, or learning-rate schedule.

## Context and Orientation

The repository root is `D:\luolin\V13`. The main training script is `breparg_improvements\train.py`. It reads environment variables at process start, builds an FSQ VQ-VAE model using Diffusers `VQModel`, samples surface and edge patches from parsed ABC pickle files, trains on reconstruction loss, and writes reports under `breparg_improvements\repro_outputs\<run-name>`.

The existing stable run used `NS_OUT=newscheme_full_vqvae_stable`, `NS_VQ_SAMPLES=300000`, `NS_VQ_EPOCHS=40`, `NS_VQ_BS=128`, `NS_DISABLE_AMP_VQVAE=1`, and learning rate `3e-4`. It wrote heavy artifacts under `E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable`.

The continuation should use the same parsed data pool and sample count, but load the existing stable best checkpoint before training. It should write to `E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100` and mirror its report under `D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_vqvae_epoch100`.

## Plan of Work

First, extend `breparg_improvements\training_stability.py` with pure helper functions. `summarize_vqvae_history(path)` should read a VQ-VAE history JSON file and return the last epoch, next epoch, best epoch, best validation reconstruction, final validation loss, and history count. `continuation_epoch_count(start_epoch, target_epoch)` should return the number of additional epochs and raise `ValueError` if the target is not greater than the start.

Second, update the active `_train_vqvae` function in `breparg_improvements\train.py`. The function should accept `start_epoch`, `initial_best_val`, `initial_best_epoch`, `history_prefix`, and `save_final_path`. It should write absolute epoch numbers to records, initialize the stopping state from a previous best value when supplied, and save `fsq_vqvae_final.pt` at the end of each epoch. Saving a final checkpoint matters because the previous stable run did not preserve the exact epoch-40 weights.

Third, update `stage_vqvae` in `breparg_improvements\train.py` to read optional environment variables. `NS_VQ_RESUME_FROM` points to a checkpoint to load. `NS_VQ_HISTORY_IN` points to the previous history JSON. `NS_VQ_TARGET_EPOCH` means absolute target epoch; when set with a previous history, the number of epochs to run is `target - next_epoch`. `NS_VQ_LR` controls the VQ-VAE learning rate. These variables should be reflected in `train_report.json`.

Fourth, add `tools\run_vqvae_continue_to_epoch100.ps1`. The script should set the parsed pool, output base, output name, sample count, batch size, lower learning rate, target epoch, resume checkpoint, and history input. It should run `split` and then `vqvae`. It should not run sequence or AR.

Fifth, run focused tests and syntax checks. The new helper tests must fail before implementation and pass afterward. Python compilation should pass for the touched files. The PowerShell script should parse without errors.

Finally, launch the continuation as a background PowerShell process and monitor `vqvae_history.json`, `train_report.json`, GPU utilization, and the newly written checkpoints.

## Concrete Steps

Run commands from `D:\luolin\V13`.

The RED test command was:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py::LocalPipelineHelperTests::test_vqvae_history_summary_supports_continuation_from_previous_best tests/test_local_pipeline_helpers.py::LocalPipelineHelperTests::test_vqvae_continuation_epochs_rejects_non_increasing_target -q

The expected RED result before implementation is two failures caused by missing imports.

After implementation, run:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile breparg_improvements\training_stability.py breparg_improvements\train.py
    [System.Management.Automation.Language.Parser]::ParseFile('D:\luolin\V13\tools\run_vqvae_continue_to_epoch100.ps1', [ref]$null, [ref]$null)

To start the continuation:

    Start-Process powershell.exe -ArgumentList '-ExecutionPolicy','Bypass','-File','D:\luolin\V13\tools\run_vqvae_continue_to_epoch100.ps1' -WindowStyle Hidden

To monitor:

    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_vqvae_continue_to_epoch100|train.py' } | Select-Object ProcessId,Name,CreationDate,CommandLine
    Get-Content -LiteralPath 'E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\vqvae_history.json' -Raw
    Get-Content -LiteralPath 'D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_vqvae_epoch100\train_report.json' -Raw
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits

## Validation and Acceptance

The code change is accepted when helper tests pass, Python compilation passes, the continuation script parses, and a small checkpoint-load check can load the continuation best checkpoint. The training behavior is accepted when the continuation history starts at epoch 40, writes finite train and validation losses, preserves previous history context or resume metadata, saves both `fsq_vqvae_best.pt` and `fsq_vqvae_final.pt`, and records whether it reached epoch 100 or stopped early.

The training target is accepted as an experiment, not a guaranteed outcome. If validation loss approaches `0.00010` without train-validation divergence, the target is feasible for this model and data. If validation loss remains around `0.00055` while training loss continues down, longer training alone is not enough and the next plan should consider model capacity, sample diversity, loss weighting, or a learning-rate schedule.

## Idempotence and Recovery

The continuation writes to `newscheme_full_vqvae_epoch100`, leaving `newscheme_full_vqvae_stable` untouched. If the process is interrupted, the final checkpoint can be used as a future resume source because this plan adds `fsq_vqvae_final.pt`. Re-running the script will recreate the split and begin again from the stable best checkpoint unless the script is edited to point at the continuation final checkpoint.

## Artifacts and Notes

Baseline stable evidence:

    report_status=VERIFIED
    epochs_ran=40
    best_val_recon=0.00056
    best_epoch=35
    val_final=0.00055
    history_count=40

Initial RED test evidence:

    FAILED test_vqvae_history_summary_supports_continuation_from_previous_best
    ImportError: cannot import name 'summarize_vqvae_history'
    FAILED test_vqvae_continuation_epochs_rejects_non_increasing_target
    ImportError: cannot import name 'continuation_epoch_count'

Implementation validation evidence:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q
    ................                                                         [100%]
    16 passed in 1.09s

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile breparg_improvements\training_stability.py breparg_improvements\train.py tools\prepare_ssd_pipeline.py tools\run_ssd_archive_pipeline.py
    no output; exit code 0

    PowerShell parser for tools\run_vqvae_continue_to_epoch100.ps1
    parser_ok

First continuation epoch evidence:

    count=1
    first_epoch=40
    last_epoch=40
    last_train=0.00029
    last_val=0.00043
    best=0.00043
    best_epoch=40
    finite_train=2227/2227
    finite_val=118/118
    skipped=0

Epoch 46 continuation evidence:

    count=7
    first_epoch=40
    last_epoch=46
    last_train=0.00023
    last_val=0.00040
    best=0.00040
    best_epoch=46
    finite_train=2227/2227
    finite_val=118/118
    skipped=0

Epoch 49 continuation evidence:

    count=10
    first_epoch=40
    last_epoch=49
    last_train=0.00022
    last_val=0.00041
    best=0.00040
    best_epoch=46
    finite_train=2227/2227
    finite_val=118/118
    skipped=0

Epoch 53 continuation evidence:

    count=14
    first_epoch=40
    last_epoch=53
    last_train=0.00021
    last_val=0.00042
    best=0.00040
    best_epoch=52
    finite_train=2227/2227
    finite_val=118/118
    skipped=0

Epoch 59 continuation evidence:

    count=20
    first_epoch=40
    last_epoch=59
    last_train=0.00019
    last_val=0.00039
    best=0.00039
    best_epoch=59
    finite_train=2227/2227
    finite_val=118/118
    skipped=0

Epoch 64 continuation evidence:

    count=25
    first_epoch=40
    last_epoch=64
    last_train=0.00019
    last_val=0.00041
    best=0.00039
    best_epoch=59
    finite_train=2227/2227
    finite_val=118/118
    skipped=0
    epochs_without_improvement=5

Epoch 72 continuation evidence:

    count=33
    first_epoch=40
    last_epoch=72
    last_train=0.00017
    last_val=0.00038
    best=0.00038
    best_epoch=70
    finite_train=2227/2227
    finite_val=118/118
    skipped=0

Epoch 79 continuation evidence:

    count=40
    first_epoch=40
    last_epoch=79
    last_train=0.00017
    last_val=0.00038
    best=0.00037
    best_epoch=73
    finite_train=2227/2227
    finite_val=118/118
    skipped=0

Final continuation evidence:

    status=VERIFIED
    target_epoch=100
    start_epoch=40
    end_epoch=85
    epochs_ran=46
    train_init=0.00029
    train_final=0.00015
    val_init=0.00043
    val_final=0.00038
    best_val_recon=0.00037
    best_epoch=73
    baseline_best_val_recon=0.00056
    baseline_best_epoch=35
    stopped_early=True
    early_stop_reason=patience=12
    amp=False
    lr=0.0001

Final validation evidence:

    Select-String for Infinity, -Infinity, and NaN in history/report
    no matches

    torch.load(fsq_vqvae_best.pt, map_location='cpu')
    keys=['fsq_levels', 'model_state_dict']
    state_entries=305

    torch.load(fsq_vqvae_final.pt, map_location='cpu')
    keys=['fsq_levels', 'model_state_dict']
    state_entries=305

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q
    ................                                                         [100%]
    16 passed in 0.98s

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile breparg_improvements\training_stability.py breparg_improvements\train.py tools\prepare_ssd_pipeline.py tools\run_ssd_archive_pipeline.py
    no output; exit code 0

Revision note: Created this ExecPlan after verifying that the stable baseline has no final epoch-40 checkpoint, which changes the continuation source from exact epoch-40 weights to the safer epoch-35 best checkpoint.

Revision note: Updated this ExecPlan after implementing and launching the epoch-100 continuation. The first record proves the continuation starts at absolute epoch 40 and is writing finite metrics.

Revision note: Updated this ExecPlan after epoch 46 showed continued finite validation improvement to `0.00040`. The target `0.00010` remains ambitious, but the continuation is not yet overfitting.

Revision note: Updated this ExecPlan after epoch 49 showed an active finite run with a possible validation plateau after epoch 46.

Revision note: Updated this ExecPlan after epoch 53 showed continued stability, a tiny best refresh at epoch 52, and no clear fast path toward `0.00010` yet.

Revision note: Updated this ExecPlan after epoch 59 refreshed best validation to rounded `0.00039`. The continuation remains worthwhile but the improvement slope is shallow.

Revision note: Updated this ExecPlan after epoch 64 showed a clear plateau around rounded `0.00039` to `0.00041`, with patience protecting against a wasteful long tail.

Revision note: Updated this ExecPlan after epoch 72 showed another small improvement to rounded `0.00038`, supporting continued monitoring while keeping the `0.00010` target as ambitious.

Revision note: Updated this ExecPlan after epoch 79 showed best validation at rounded `0.00037` and stable but slow progress.

Revision note: Updated this ExecPlan after the continuation completed by early stopping at epoch 85, with final validation evidence and acceptance results.

## Interfaces and Dependencies

In `breparg_improvements\training_stability.py`, define:

    summarize_vqvae_history(path)
    continuation_epoch_count(start_epoch, target_epoch)

In `breparg_improvements\train.py`, support these optional environment variables:

    NS_VQ_RESUME_FROM
    NS_VQ_HISTORY_IN
    NS_VQ_TARGET_EPOCH
    NS_VQ_LR

The continuation script should run with `C:\Users\YU\.conda\envs\brepgen_env\python.exe`, the same environment used for prior verified training.
