# Stable VQ-VAE Retraining

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` in the repository root. The user has already processed all ABC chunks into parsed pickle files under `E:\ABC\processed\abc_parsed_full`. The current full training run completed VQ-VAE training but showed `val=inf` from about epoch 20 onward. The desired observable result is a stable VQ-VAE retraining path that stops instead of wasting epochs after non-finite validation loss, records enough evidence to diagnose the run, and provides a safe command for rerunning only the VQ-VAE preparation and training under a new output name.

## Purpose / Big Picture

After this change, the user can rerun the VQ-VAE part of the new BrepARG scheme without repeating the earlier failure mode where many epochs report `train=0.00000 val=inf`. The training code will treat non-finite losses as an explicit condition to record and stop on, rather than silently running to the requested epoch count. The user can see the change working by running focused tests and by starting a stable retrain that writes `vqvae_history.json`, a guarded `train_report.json`, and `fsq_vqvae_best.pt` under a new run directory.

## Progress

- [x] (2026-06-27 00:27 +08:00) Read `AGENTS.md` and `PLANS.md`; confirmed complex training behavior changes require this ExecPlan.
- [x] (2026-06-27 00:29 +08:00) Confirmed the active process is still the original `train.py --stage all` child of the archive pipeline, so file edits will not mutate that already-loaded Python process.
- [x] (2026-06-27 00:31 +08:00) Gathered current evidence: full local VQ-VAE run reached `best_val_recon=0.00082`, then logged `val=inf` through epoch 119 while still reporting `status=VERIFIED`.
- [x] (2026-06-27 00:36 +08:00) Added lightweight stability helper tests and observed the expected RED failure: three tests failed with `ModuleNotFoundError: No module named 'training_stability'`.
- [x] (2026-06-27 00:39 +08:00) Implemented `breparg_improvements\training_stability.py`; reran local helper tests and observed `12 passed`.
- [x] (2026-06-27 00:47 +08:00) Added stable VQ-VAE stopping, safer metric reporting, and `vqvae_history.json` output wiring in `breparg_improvements\train.py`.
- [x] (2026-06-27 00:50 +08:00) Added stable retrain configuration plumbing in `local_training_config.json`, `tools\prepare_ssd_pipeline.py`, and `tools\run_ssd_archive_pipeline.py`.
- [x] (2026-06-27 00:52 +08:00) Added Windows stable retrain command script `tools\run_stable_vqvae_retrain.ps1`.
- [x] (2026-06-27 00:54 +08:00) Ran focused tests and syntax checks: local helper tests passed, Python compilation passed, and PowerShell parser reported no script syntax errors.
- [x] (2026-06-27 01:02 +08:00) Added and fixed a regression test for parsing string `"false"` as disabled VQ-VAE AMP rather than accidentally treating it as truthy.
- [x] (2026-06-27 03:46 +08:00) Stopped the old `newscheme_full_local` training process and launched `tools\run_stable_vqvae_retrain.ps1` in the background.
- [x] (2026-06-27 03:56 +08:00) Stable retrain completed `vqsweep` and started `vqvae`; `lr3e-4_L8.8.8.16` won with `best_val_recon=0.04658`.
- [x] (2026-06-27 04:03 +08:00) Stable `vqvae` wrote epoch 0 to `vqvae_history.json`: `train_loss=0.04778`, `val_loss=0.01194`, `finite_train_batches=2227/2227`, `finite_val_batches=118/118`, `skipped_train_batches=0`.
- [x] (2026-06-27 04:29 +08:00) Stable `vqvae` reached epoch 4 with finite losses and no skipped batches; `best_val_recon=0.00238`, `consecutive_nonfinite_val_epochs=0`.
- [x] (2026-06-27 04:59 +08:00) Stable `vqvae` reached epoch 10 with finite losses and no skipped batches; `best_val_recon=0.00108`, `consecutive_nonfinite_val_epochs=0`.
- [x] (2026-06-27 06:00 +08:00) Stable `vqvae` crossed the previous failure window at epoch 20 with finite losses: `train_loss=0.00061`, `val_loss=0.00076`, `best_val_recon=0.00073`, `consecutive_nonfinite_val_epochs=0`, `skipped_train_batches=0`.
- [x] (2026-06-27 10:22 +08:00) Stable `vqvae` completed naturally at 40 epochs. Final report shows `status=VERIFIED`, `epochs_ran=40`, `train_final=0.00038`, `val_final=0.00055`, `best_val_recon=0.00056`, `best_epoch=35`, and no early-stop reason.
- [x] (2026-06-27 10:22 +08:00) Verified `vqvae_history.json` has 40 epoch records, no `Infinity`/`NaN`, no skipped train batches, all train batches finite at `2227/2227`, and all validation batches finite at `118/118`.
- [x] (2026-06-27 10:22 +08:00) Verified `fsq_vqvae_best.pt` exists, is loadable with PyTorch on CPU, and contains `model_state_dict` plus `fsq_levels`.

## Surprises & Discoveries

- Observation: The current full local VQ-VAE run reached a usable best checkpoint before becoming numerically unstable.
  Evidence: `D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_local\train_report.json` reports `best_val_recon=0.00082`, while `E:\ABC\processed\logs\archive_pipeline_20260626_111814.log` shows `vqvae ep 20 train=0.00000 val=inf` through epoch 119.
- Observation: Older smaller runs did not show the same late `val=inf` pattern.
  Evidence: `breparg_improvements\repro_outputs\newscheme_big\train_report.json` reports `val_final=0.0005` after 120 epochs, and `newscheme_home\train_report.json` reports `val_final=0.00062` after 120 epochs.
- Observation: The existing training loop reports `train=0.00000` when no finite train batch contributes to a round.
  Evidence: `breparg_improvements\train.py` computes `tr = tot / max(1, nb)`, so when `nb` is zero the displayed value is zero rather than a warning value.
- Observation: This checkout is not a git repository.
  Evidence: `git status --short` returned `fatal: not a git repository (or any of the parent directories): .git`.
- Observation: The existing VQ-VAE function area contains historical encoding-corrupted comments, including comments and code visually appearing on the same line in PowerShell output.
  Evidence: line-number output around `breparg_improvements\train.py` showed the old `_train_vqvae` block with malformed comments. The implemented safe path redefines `_train_vqvae` after the legacy block so Python uses the new definition without requiring a risky large rewrite of the corrupted text.
- Observation: Python treats non-empty strings such as `"false"` as truthy when passed to `bool(...)`.
  Evidence: The new test `test_training_env_parses_disable_amp_string_false` first failed because `tools\prepare_ssd_pipeline.py` rendered `"NS_DISABLE_AMP_VQVAE": "1"` for `"false"`.
- Observation: The stable FP32 VQ-VAE retrain crossed the old failure window without reproducing non-finite validation.
  Evidence: `vqvae_history.json` shows epoch 20 with `train_loss=0.00061`, `val_loss=0.00076`, `best_val=0.00073`, `finite_train_batches=2227/2227`, `finite_val_batches=118/118`, and `skipped_train_batches=0`. The old `newscheme_full_local` report wrote `val_final=Infinity`.
- Observation: The final stable run did not waste epochs after collapse; there was no collapse to stop.
  Evidence: The final stable report shows `epochs_ran=40`, `stopped_early=False`, `early_stop_reason=""`, `val_final=0.00055`, and `best_val_recon=0.00056`. Searches for `Infinity`, `-Infinity`, and `NaN` in both `vqvae_history.json` and `train_report.json` returned no matches.

## Decision Log

- Decision: Preserve the current running training process and make code/config changes for the next retrain.
  Rationale: The active Python process already imported the old code and is currently in sequence generation. Stopping it would discard progress, while changing files now prepares a safer next run without side effects.
  Date/Author: 2026-06-27 / Codex.
- Decision: Add explicit early stopping for consecutive non-finite validation epochs.
  Rationale: A best checkpoint can still be useful, but continuing dozens of epochs after `val=inf` wastes GPU time and hides the failure mode. A stop reason in the report is easier to trust than a long run with invalid final metrics.
  Date/Author: 2026-06-27 / Codex.
- Decision: Record per-epoch VQ-VAE history to `vqvae_history.json`.
  Rationale: The existing report only stores initial, final, and best values. A separate history file gives enough evidence to confirm that loss stayed finite, early stopping fired when expected, and the selected checkpoint corresponds to the best validation loss.
  Date/Author: 2026-06-27 / Codex.
- Decision: Make VQ-VAE AMP optional through an environment variable, while leaving AR AMP unchanged.
  Rationale: The likely failure mode is mixed precision overflow inside the VQ-VAE path. Disabling AMP only for VQ-VAE gives a stable retrain option without slowing AR by default.
  Date/Author: 2026-06-27 / Codex.
- Decision: Do not launch the stable retrain immediately.
  Rationale: The original `train.py --stage all` process is still active. Starting another GPU-heavy VQ-VAE retrain would compete with the active process and make both runs slower or less stable. The retrain script is ready to run when the current process is intentionally stopped or finishes.
  Date/Author: 2026-06-27 / Codex.
- Decision: Stop the old full `newscheme_full_local` training and start the stable VQ-VAE-only retrain.
  Rationale: The user's active objective is to set and verify stable VQ-VAE retraining. The old process was still consuming GPU on downstream sequence generation from the prior VQ-VAE run. Parsed data and existing artifacts are preserved, so stopping the old training frees GPU for the requested stable retrain without data loss.
  Date/Author: 2026-06-27 / Codex.
- Decision: Treat `newscheme_full_vqvae_stable` as the verified VQ-VAE checkpoint source for any next downstream experiment, rather than the old `newscheme_full_local` VQ-VAE report with `val_final=Infinity`.
  Rationale: The stable run has a finite full history, a finite final validation loss, a lower best validation loss than the old run, and a loadable checkpoint.
  Date/Author: 2026-06-27 / Codex.

## Outcomes & Retrospective

The code, command setup, and empirical stable VQ-VAE retrain are complete. The stable retrain ran under `newscheme_full_vqvae_stable` and finished all 40 requested epochs without the previous `val=inf` failure signature. The final report records `status=VERIFIED`, `train_final=0.00038`, `val_final=0.00055`, `best_val_recon=0.00056`, and `best_epoch=35`. The history file records finite train and validation batches for every epoch, and the best checkpoint is loadable.

Compared with the old `newscheme_full_local` VQ-VAE evidence, the stable run is stronger: the old run had a useful best checkpoint at `best_val_recon=0.00082` but ended with `val_final=Infinity`; the stable run improved the best validation reconstruction to `0.00056` and ended with finite `val_final=0.00055`. The remaining next step, if full generation training is desired, is to start downstream `sequence` and `ar` from this stable checkpoint/output context rather than from the old unstable report.

## Context and Orientation

The repository root is `D:\luolin\V13`. It is not a git repository in this checkout, so validation must rely on file contents, tests, and process outputs rather than commits.

The main training script is `breparg_improvements\train.py`. It defines stages named `split`, `vqsweep`, `vqvae`, `sequence`, `ar`, and `ar_sweep`. Running `python breparg_improvements\train.py --stage all` executes `split`, `vqsweep`, `vqvae`, `sequence`, and `ar` in that order. The VQ-VAE stage trains a vector-quantized autoencoder. In plain terms, it learns to turn surface and edge geometry patches into compact discrete codes and back into reconstructed geometry. The later `sequence` stage uses the best VQ-VAE checkpoint to encode parsed CAD data into token sequences.

The current full data run was launched by `tools\run_ssd_archive_pipeline.py`, which sets environment variables such as `NS_POOL`, `NS_OUTBASE`, `NS_OUT`, `NS_VQ_SAMPLES`, `NS_VQ_EPOCHS`, and `NS_VQ_BS` before starting `train.py`. The run output directory is `E:\ABC\processed\train_outputs\newscheme_full_local`; lightweight reports are mirrored under `breparg_improvements\repro_outputs\newscheme_full_local`.

The failure mode to prevent is not that the best checkpoint is missing. The best checkpoint exists and reports `best_val_recon=0.00082`. The problem is that the loop continued after validation became non-finite, and the final report wrote `val_final=Infinity`, which is poor evidence for a stable training run.

## Plan of Work

First, create a lightweight helper module named `breparg_improvements\training_stability.py`. This module will not import `torch` or `diffusers`; it will only contain small pure-Python functions and data classes that decide whether a validation value is an improvement, whether a stop condition has fired, and how to represent non-finite numeric values in JSON-friendly form.

Second, add tests in `tests\test_local_pipeline_helpers.py` that import `training_stability.py`. The tests will prove that two consecutive non-finite validation epochs after a minimum epoch count produce an early-stop reason, that finite improvements reset the non-finite counter, and that a missing finite train batch is represented as `inf` or `None` rather than fake zero.

Third, update `breparg_improvements\train.py`. The `_train_vqvae` function will track total train and validation batches, finite batches, skipped non-finite train batches, best epoch, consecutive non-finite validation epochs, and the stop reason. It will write `vqvae_history.json` only for the full VQ-VAE stage. It will compute `train_loss` as non-finite when no finite train batch exists, instead of `0.00000`. It will stop when either patience is exhausted or consecutive non-finite validation epochs exceed the configured maximum. The function will still save only finite validation improvements to `fsq_vqvae_best.pt`.

Fourth, update configuration plumbing. `local_training_config.json` and `tools\prepare_ssd_pipeline.py` will expose stable retrain knobs: `NS_VQ_MIN_EPOCHS`, `NS_VQ_PATIENCE`, `NS_VQ_MIN_DELTA`, `NS_VQ_MAX_NONFINITE_VAL_EPOCHS`, and `NS_DISABLE_AMP_VQVAE`. The archive pipeline's existing full training launch can remain backward compatible, but future prepared commands should include the new safeguards.

Fifth, add a Windows-friendly command script `tools\run_stable_vqvae_retrain.ps1`. It will set `NS_OUT=newscheme_full_vqvae_stable`, point `NS_POOL` at `E:\ABC\processed\abc_parsed_full`, write heavy outputs under `E:\ABC\processed\train_outputs`, disable AMP for VQ-VAE by default, cap VQ-VAE epochs at 40, and run `split`, `vqsweep`, then `vqvae`. It will not run `sequence` or `ar`, so it is safe for checking a stable VQ-VAE before committing to downstream work.

## Concrete Steps

Run all commands from `D:\luolin\V13` unless a command says otherwise.

Before implementation, observe the current failure evidence:

    Get-Content -LiteralPath 'D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_local\train_report.json' -Raw
    Select-String -LiteralPath 'E:\ABC\processed\logs\archive_pipeline_20260626_111814.log' -Pattern 'vqvae ep|saved best|===== STAGE sequence' | Select-Object -Last 20

Expected evidence includes `best_val_recon` near `0.00082`, `val_final` as `Infinity`, and log lines where epoch 20 through epoch 119 show `val=inf`.

After implementation, run focused tests:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q

Expected output is all local helper tests passing. The new stability tests should fail before the implementation and pass after it.

Run syntax checks:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile breparg_improvements\training_stability.py breparg_improvements\train.py tools\prepare_ssd_pipeline.py

Expected output is no output and exit code zero.

Start a stable VQ-VAE-only retrain when the user is ready:

    powershell -ExecutionPolicy Bypass -File tools\run_stable_vqvae_retrain.ps1

Expected artifacts are:

    E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\split.pkl
    E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\vqvae_hp_sweep.json
    E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\fsq_vqvae_best.pt
    E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\vqvae_history.json
    D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_vqvae_stable\train_report.json

## Validation and Acceptance

The code change is accepted when the focused helper tests pass, syntax checks pass, and the stable retrain command is present with safe environment defaults. The training behavior is accepted when a stable retrain writes `vqvae_history.json` with finite validation entries until either normal completion or a clear early-stop reason, and when `train_report.json` records `best_val_recon` as a finite value, `status` as `VERIFIED`, and a non-empty `early_stop_reason` if the run stops before the requested epoch count.

A stable retrain should not produce the old failure signature where many later epochs continue after `val=inf` with `train=0.00000`. If non-finite validation appears, the new behavior should stop within the configured number of consecutive non-finite validation epochs and preserve the last finite best checkpoint.

## Idempotence and Recovery

The code changes are additive and safe to rerun. The stable retrain script uses a new output name, `newscheme_full_vqvae_stable`, so it does not overwrite the active `newscheme_full_local` run. If a stable retrain is interrupted, rerunning the script will recreate the split and retrain VQ-VAE from scratch under the same stable output directory. The current active `train.py --stage all` process is not modified by editing files because Python already loaded its module code when that process started.

## Artifacts and Notes

Important existing evidence:

    Full local VQ-VAE:
    best_val_recon = 0.00082
    val_final = Infinity
    status = VERIFIED

    Current log excerpt:
    [23:52:45]   vqvae ep  10 train=0.00096 val=0.00087
    [00:10:19]   vqvae ep  20 train=0.00000 val=inf
    ...
    [02:53:51]   vqvae ep 119 train=0.00000 val=inf
    [02:53:52]   saved best -> E:\ABC\processed\train_outputs\newscheme_full_local\fsq_vqvae_best.pt  (val_init 0.0176 -> best 0.00082)

Expected new knobs:

    NS_VQ_MIN_EPOCHS=12
    NS_VQ_PATIENCE=8
    NS_VQ_MIN_DELTA=1e-5
    NS_VQ_MAX_NONFINITE_VAL_EPOCHS=2
    NS_DISABLE_AMP_VQVAE=1

Validation transcript:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q
    .............                                                            [100%]
    13 passed in 1.02s

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile breparg_improvements\training_stability.py breparg_improvements\train.py tools\prepare_ssd_pipeline.py tools\run_ssd_archive_pipeline.py
    no output; exit code 0

    [System.Management.Automation.Language.Parser]::ParseFile('D:\luolin\V13\tools\run_stable_vqvae_retrain.ps1', ...)
    no parser errors; exit code 0

Final stable retrain validation transcript:

    Get-CimInstance Win32_Process | Where-Object { stable retrain process or train.py stage process }
    no stable retrain or train.py stage process remained after completion

    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    NVIDIA GeForce RTX 3060, 0, 1681, 12288, 53, 13.69

    E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\vqvae_history.json
    history_count=40
    final_epoch=39
    final_train=0.00038
    final_val=0.00055
    best_val_recon=0.00056
    best_epoch=35
    nonfinite_or_skipped_epochs=0
    total_skipped_train_batches=0
    all_train_batches_finite=True
    all_val_batches_finite=True

    D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_vqvae_stable\train_report.json
    status=VERIFIED
    samples=300000
    epochs_requested=40
    epochs_ran=40
    train_init=0.04778
    train_final=0.00038
    val_init=0.01194
    val_final=0.00055
    best_val_recon=0.00056
    best_epoch=35
    stopped_early=False
    early_stop_reason=""

    Select-String for Infinity, -Infinity, and NaN in stable history and report
    no matches

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q
    .............                                                            [100%]
    13 passed in 3.39s

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile breparg_improvements\training_stability.py breparg_improvements\train.py tools\prepare_ssd_pipeline.py tools\run_ssd_archive_pipeline.py
    no output; exit code 0

    torch.load('E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\fsq_vqvae_best.pt', map_location='cpu')
    checkpoint_bytes=228776003
    checkpoint_type=dict
    top_level_keys=['fsq_levels', 'model_state_dict']
    state_entries=305

## Interfaces and Dependencies

In `breparg_improvements\training_stability.py`, define pure helpers that do not depend on GPU libraries:

    VQVAEStopConfig
    VQVAEStopState
    parse_env_bool(value, default=False)
    safe_json_number(value)
    finite_average(total, count)
    update_vqvae_stop_state(epoch, val_loss, state, config)

In `breparg_improvements\train.py`, import those helpers and use them inside `_train_vqvae`. The function should continue to return training history and best validation loss for existing callers, but it may also return metadata needed by `stage_vqvae`. If the return signature changes, update both `stage_vqsweep` and `stage_vqvae` in the same edit.

Revision note: Initial ExecPlan created after observing the full local VQ-VAE run's late non-finite validation behavior and before implementing stable retraining safeguards.

Revision note: Implemented the stable retraining safeguards and updated this plan with validation evidence. The stable retrain itself was not launched because the original training process remains active.
