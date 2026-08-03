# V13 AR Training and Parsed Archive ExecPlan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document follows `PLANS.md` in the repository root.

## Purpose / Big Picture

The user wants to move from completed VQ-VAE and sequence generation into a robust autoregressive, or AR, training run. AR training learns a language model over CAD token sequences. It does not read the parsed ABC `.pkl` geometry files directly; it reads a single sequence pickle containing token IDs, attention masks, vocabulary metadata, and train/validation/test splits. After this work, the training inputs and outputs needed for AR will live under `D:\luolin\V13`, the AR trainer will support common long-run controls such as resume, best/latest checkpoints, periodic checkpoints every 20 epochs, and durable logs, and the parsed `.pkl` chunks on `E:` will be archived by chunk for storage hygiene without destroying the only source of recovery.

## Progress

- [x] (2026-06-28 04:05 +08:00) Verified current sequence outputs: canonical and sharded-merged sequence files both contain 425120 valid sequences with `out_of_vocab=0`.
- [x] (2026-06-28 04:07 +08:00) Verified D drive has about 149 GB free and can hold the AR input, checkpoints, logs, and reports.
- [x] (2026-06-28 04:08 +08:00) Read `breparg_improvements\train.py` and confirmed AR currently reads `sequences_fsq_rcm.pkl`, saves only `ar_best.pt`, and lacks resume/latest/periodic checkpoint controls.
- [x] (2026-06-28 13:31 +08:00) Added tests for AR checkpoint naming, sequence validation, V13 launcher settings, and length-bucketed AR batches.
- [x] (2026-06-28 13:31 +08:00) Implemented `breparg_improvements\ar_training_utils.py` and wired AR resume/latest/best/periodic checkpoint/history support into `breparg_improvements\train.py`.
- [x] (2026-06-28 13:36 +08:00) Created V13 local AR run directory and copied `sequences_fsq_rcm.pkl` plus `split.pkl` into it.
- [x] (2026-06-28 13:48 +08:00) Ran AR preflight on the V13-local sequence file; report status is `VERIFIED` with finite smoke loss.
- [x] (2026-06-28 13:31 +08:00) Added `tools\run_ar_v13_epoch100.ps1`; launcher writes AR inputs, outputs, logs, checkpoints, and history under `D:\luolin\V13\local_runs\ar_training`.
- [x] (2026-06-28 13:37 +08:00) Started parsed chunk archive creation with manifest records and without deleting parsed directories.
- [x] (2026-06-28 13:57 +08:00) Started full AR training process with hardware monitoring.
- [x] (2026-06-28 14:54 +08:00) Verified first completed AR epoch wrote `ar_latest.pt`, `ar_best.pt`, and one `ar_history.jsonl` row.
- [x] (2026-06-28 15:18 +08:00) Added a read-only status checker, `tools\check_ar_archive_status.py`, for AR/parsed-archive/GPU/disk monitoring.
- [x] (2026-06-28 15:24 +08:00) Enhanced the status checker to parse current AR epoch, batch progress, running CE, and recent-speed ETA from training logs.
- [x] (2026-06-28 16:08 +08:00) Re-verified epoch 2 latest/best checkpoints are CPU-loadable while AR continued through epoch 3.
- [x] (2026-06-28 16:52 +08:00) Verified epoch 3 latest/best checkpoints are CPU-loadable and AR entered epoch 4.
- [x] (2026-06-28 17:46 +08:00) Verified epoch 4 completed with a new best validation loss and AR entered epoch 5.
- [x] (2026-06-28 17:55 +08:00) Fixed the read-only status checker so JSON output remains printable in GBK PowerShell consoles.
- [x] (2026-06-28 18:00 +08:00) Hardened the status checker so one missing drive does not hide the status of other drives.
- [x] (2026-06-28 18:44 +08:00) Verified epoch 5 completed with a new best validation loss and AR entered epoch 6.
- [x] (2026-06-28 19:41 +08:00) Verified epoch 6 completed with a new best validation loss and AR entered epoch 7.
- [x] (2026-06-28 20:38 +08:00) Verified epoch 7 completed with a new best validation loss and AR entered epoch 8.
- [x] (2026-06-28 21:35 +08:00) Verified epoch 8 completed with a new best validation loss and AR entered epoch 9.
- [x] (2026-06-28 21:51 +08:00) Restored `E:` visibility and resumed parsed archive creation through `abc_0040`.
- [x] (2026-06-28 22:32 +08:00) Verified epoch 9 completed with a new best validation loss and AR entered epoch 10.
- [x] (2026-06-28 22:45 +08:00) Verified parsed archive creation continued through `abc_0054`.
- [x] (2026-06-28 23:29 +08:00) Verified epoch 10 completed with a new best validation loss and AR entered epoch 11.
- [x] (2026-06-28 23:45 +08:00) Verified parsed archive creation continued through `abc_0068`.
- [x] (2026-06-29 00:26 +08:00) Verified epoch 11 completed with a new best validation loss and AR entered epoch 12.
- [x] (2026-06-29 00:42 +08:00) Verified parsed archive creation continued through `abc_0082` and removed an old verified-safe `abc_0041` temp file.
- [x] (2026-06-29 01:23 +08:00) Verified epoch 12 completed with a new best validation loss and AR entered epoch 13.
- [x] (2026-06-29 01:48 +08:00) Completed parsed chunk archive creation through `abc_0099`.
- [x] (2026-06-29 09:00 +08:00) Verified epoch 20 completed and wrote loadable periodic checkpoint `ar_epoch_0020.pt`.
- [x] (2026-06-29 18:33 +08:00) Verified AR completed epoch 30 with a new best validation CE, GPU utilization about 97-98%, and latest/best/epoch-20 checkpoints are CPU-loadable.
- [x] (2026-06-29 18:40 +08:00) Audited the original four-part objective: parsed archives, V13-local AR inputs, preflight/init tests, checkpoint/resume/logging, and GPU saturation are verified; full AR training remains in progress at epoch 31.
- [x] (2026-06-29 18:50 +08:00) Added and launched a read-only epoch-40 gate monitor that will keep polling until `ar_epoch_0040.pt` exists and can be CPU-loaded.
- [x] (2026-06-29 19:20 +08:00) Added a read-only AR history analyzer and used it on the live history; epoch 30 remains the best epoch and the recommendation is to continue unchanged.
- [x] (2026-06-29 19:31 +08:00) Verified epoch 31 completed with a new best validation CE and both `ar_best.pt` and `ar_latest.pt` CPU-load with model, optimizer, and scaler state.
- [x] (2026-06-29 20:12 +08:00) Recovered D drive free space by deleting a then-duplicate local parsed-archive mirror under `D:\luolin\V13\ABC\processed\abc_parsed_full_archives`; later the user intentionally moved the offline archive set back under `D:\luolin\V13\ABC`.
- [x] (2026-06-29 22:42 +08:00) Verified the E drive is no longer needed by the active AR run; updated the read-only status checker to use `D:\luolin\V13\ABC` as the archive/log root.
- [x] (2026-06-30 04:16 +08:00) Verified epoch 40 completed, wrote loadable `ar_epoch_0040.pt`, and continued training unchanged because the curve shows only a mild plateau, not a stop/restart signal.
- [x] (2026-06-30 04:26 +08:00) Started a read-only epoch-60 gate monitor writing to `D:\luolin\V13\local_runs\ar_training\logs\ar_epoch60_gate_20260630_042607.jsonl`; current status is waiting for epoch 60 while live training continues in epoch 41.
- [x] (2026-06-30 05:05 +08:00) Verified epoch 41 completed, wrote a loadable `ar_latest.pt`, and triggered the AR analyzer's plateau/overfit caution because validation had not improved for 5 epochs after the epoch-36 best.
- [x] (2026-06-30 05:10 +08:00) Fixed AR resume so loading an optimizer checkpoint does not silently keep the old learning rate when `NS_AR_LR` is changed.
- [x] (2026-06-30 05:14 +08:00) Restarted AR from epoch 41 `ar_latest.pt` with `lr=1e-4` and `TargetEpochs=100`; the new launcher process is `37868`, training Python process is `3884`, and GPU utilization returned to about 95-99%.
- [x] (2026-06-30 06:18 +08:00) Verified the post-lower-LR epoch 42 result: `val_CE` improved from the previous best `0.3302254088390201` to `0.30511944202840924`; both `ar_latest.pt` and `ar_best.pt` CPU-load with `learning_rate=0.0001` and optimizer param groups at `0.0001`.
- [x] (2026-06-30 07:18 +08:00) Verified epoch 43 continued the lower-LR improvement: `val_CE=0.30437195893850855`, both latest and best checkpoints CPU-load at epoch 43 with optimizer learning rate `0.0001`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-06-30 08:18 +08:00) Verified epoch 44 again improved validation to `val_CE=0.3035273262473785`; latest and best checkpoints CPU-load at epoch 44 with optimizer learning rate `0.0001`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-06-30 09:18 +08:00) Verified epoch 45 again improved validation to `val_CE=0.3026073543786854`; latest and best checkpoints CPU-load at epoch 45 with optimizer learning rate `0.0001`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-06-30 10:12 +08:00) Verified epoch 46 completed with `val_CE=0.303381397759082`; it did not beat the epoch-45 best, but `ar_latest.pt` CPU-loads at epoch 46 with optimizer learning rate `0.0001`, `ar_best.pt` correctly remains epoch 45, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-06-30 12:12 +08:00) Verified epoch 48 produced a new best `val_CE=0.30222854646215414`; both `ar_latest.pt` and `ar_best.pt` CPU-load at epoch 48 with optimizer learning rate `0.0001`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-06-30 13:09 +08:00) Verified epoch 49 produced another new best `val_CE=0.30193200306664975`; both `ar_latest.pt` and `ar_best.pt` CPU-load at epoch 49 with optimizer learning rate `0.0001`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-06-30 14:09 +08:00) Verified epoch 50 produced another new best `val_CE=0.3014168582354995`; both `ar_latest.pt` and `ar_best.pt` CPU-load at epoch 50 with optimizer learning rate `0.0001`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-06-30 16:03 +08:00) Added `tools\evaluate_reconstruction_v13.py`, a V13-local FSQ-aware reconstruction evaluator, and verified its dry-run path on one validation sequence.
- [x] (2026-06-30 16:16 +08:00) Verified FSQ VQ-VAE reconstruction from existing validation token sequences: 5 shortest validation samples wrote 5 retained STEP files under `local_runs\reconstruction_eval\eval_validation_short5_arbest_cpu\steps`, with 4 of 5 passing strict BREP validity.
- [x] (2026-06-30 16:23 +08:00) Verified the current `ar_best.pt` generation-to-STEP path: constrained AR generation produced one grammar-valid sequence with 4 faces and 6 edges, wrote `generated_000000_len0119.step`, and strict BREP validity passed.
- [x] (2026-06-30 17:04 +08:00) Verified epoch 53 produced a new best `val_CE=0.30052309570894425`; both latest and best checkpoints CPU-load at epoch 53 with optimizer learning rate `0.0001`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-06-30 20:05 +08:00) Verified epoch 56 produced a new best `val_CE=0.29962874219155966`; both latest and best checkpoints CPU-load at epoch 56 with optimizer learning rate `0.0001`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-07-01 00:07 +08:00) Verified epoch 60 wrote loadable periodic checkpoint `ar_epoch_0060.pt`; latest remains epoch 60, best remains epoch 56 with `val_CE=0.29962874219155966`, and the analyzer recommends continuing unchanged.
- [x] (2026-07-01 06:35 +08:00) Stopped the exhausted `lr=1e-4` continuation after epoch 66 because the analyzer had flagged plateau/overfit caution for epochs 61 through 66.
- [x] (2026-07-01 06:46 +08:00) Added a parameterized AR launcher and status checker support for separate continuation branches; helper tests passed with `44 passed`.
- [x] (2026-07-01 06:46 +08:00) Started a new V13-local `newscheme_full_v13_ar_lr5e5` branch from the epoch-56 `ar_best.pt` with `lr=5e-5`; the new branch is running in epoch 57 with GPU utilization about `95%`.
- [x] (2026-07-01 07:36 +08:00) Verified the first completed `lr=5e-5` branch epoch: epoch 57 wrote `val_CE=0.29771688658363205`, refreshed both latest and best checkpoints, and the analyzer recommends `continue_unchanged`.
- [x] (2026-07-01 08:35 +08:00) Verified epoch 58 on the `lr=5e-5` branch: `val_CE=0.2980387463729057`, latest checkpoint is epoch 58, best remains epoch 57, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-07-01 12:37 +08:00) Verified epoch 62 on the `lr=5e-5` branch: epoch 57 remains best, epochs 58 through 62 did not improve, and the analyzer now recommends `consider_stop_or_lower_lr`.
- [x] (2026-07-01 12:50 +08:00) Stopped the exhausted `lr=5e-5` branch and started isolated `newscheme_full_v13_ar_lr2e5` from the epoch-57 `ar_best.pt` with `lr=2e-5`; GPU utilization returned to about `89-92%`.
- [x] (2026-07-01 14:00 +08:00) Verified the first completed `lr=2e-5` branch epoch: epoch 58 improved validation to `0.2973588657237942`, refreshed best/latest checkpoints with optimizer learning rate `2e-5`, and the analyzer recommends `continue_unchanged`.
- [x] (2026-07-01 14:00 +08:00) Re-ran reconstruction evaluation for the new epoch-58 best checkpoint; 5 STEP files were retained, 5/5 grammar-valid and reconstructed, 4/5 strict BREP-valid, and 0 errors.
- [x] (2026-07-01 14:54 +08:00) Verified epoch 59 on the `lr=2e-5` branch: validation improved again to `0.29660862653964476`, latest/best checkpoints are CPU-loadable at epoch 59 with optimizer learning rate `2e-5`, and the analyzer recommends `continue_unchanged`.
- [x] (2026-07-01 14:57 +08:00) Clarified reconstruction semantics and ran true AR-generated reconstruction from the epoch-59 best checkpoint: seed 0 and seed 1 each produced 5 retained STEP files, with 10/10 grammar-valid, 10/10 reconstructed, 10/10 strict BREP-valid, and 0 errors.
- [x] (2026-07-01 15:55 +08:00) Verified epoch 60 gate on the `lr=2e-5` branch: `ar_epoch_0060.pt` exists, CPU-loads with model/optimizer/scaler state, latest is epoch 60, best remains epoch 59, and the analyzer recommends `continue_unchanged`.
- [x] (2026-07-01 16:49 +08:00) Verified epoch 61 on the `lr=2e-5` branch: latest checkpoint is epoch 61, best remains epoch 59 by a small margin, `epochs_since_best=2`, and the analyzer recommends `continue_unchanged`.
- [x] (2026-07-02 15:19 +08:00) Verified the `lr=2e-5` branch progressed to latest completed epoch 84, best stayed at epoch 76 with `val_CE=0.29538217296178204`, and the analyzer flagged plateau/overfit with `epochs_since_best=8`.
- [x] (2026-07-02 15:24 +08:00) Stopped the plateaued `lr=2e-5` branch and launched isolated `newscheme_full_v13_ar_lr1e5` from the epoch-76 best checkpoint with `lr=1e-5`; GPU utilization returned to about `92-93%`.
- [x] (2026-07-02 15:24 +08:00) Re-ran true AR-generated reconstruction for the epoch-76 best checkpoint using the new `lr=1e-5` run directory; seed 0 and seed 1 each produced 5 retained STEP files, with 10/10 grammar-valid, 10/10 reconstructed, 10/10 strict BREP-valid, and 0 errors.
- [x] (2026-07-02 16:40 +08:00) Verified the first completed `lr=1e-5` epoch: epoch 77 wrote `val_CE=0.29588743756313873`, did not beat epoch 76, latest checkpoint has optimizer learning rate `1e-5`, and the analyzer recommends `continue_unchanged`.
- [x] (2026-07-02 18:04 +08:00) Verified epoch 78 on the `lr=1e-5` branch: `val_CE=0.29567606687322817`, best remains epoch 76, latest checkpoint CPU-loads with optimizer learning rate `1e-5`, and the analyzer still recommends `continue_unchanged`.
- [x] (2026-07-03 09:55 +08:00) Verified the active `lr=1e-5` branch reached epoch 95 with a new best `val_CE=0.2951271435705727`; `ar_best.pt` CPU-loads at epoch 95 with `learning_rate=1e-5`, the analyzer recommends `continue_unchanged`, and the epoch-120 monitor is still polling.
- [x] (2026-07-03 09:55 +08:00) Re-ran true AR-generated reconstruction from the latest epoch-95 best checkpoint with a random runtime seed. Report `local_runs\reconstruction_eval\eval_generated5_latest_arbest_lr1e5_epoch95_seed_random_cpu_20260703_0955\reconstruction_report.json` is `VERIFIED`; 5 STEP files were retained, 5/5 grammar-valid and reconstructed, 4/5 strict BREP-valid, and 0 errors.
- [x] (2026-07-03 09:55 +08:00) Updated `tools\evaluate_reconstruction_v13.py` so future generated reports record sampling controls and can use `--seed -1` for a fresh runtime seed while preserving reproducible fixed-seed runs.
- [x] (2026-07-03 19:10 +08:00) Verified the `lr=1e-5` branch completed epoch 100 and wrote loadable `ar_latest.pt`, `ar_best.pt`, and periodic `ar_epoch_0100.pt`; best remained epoch 95, and the apparent `STAGE ar FAILED` was traced to an old stage success predicate rather than a crash.
- [x] (2026-07-03 19:15 +08:00) Re-ran generated reconstruction from epoch-95 `ar_best.pt` with three different sampling settings and fresh seeds: temp/top-p `0.8/0.90`, `1.0/0.95`, and `1.2/0.98`. The runs retained 11 total STEP files, with no duplicate STEP hashes across the retained files.
- [x] (2026-07-03 19:17 +08:00) Changed generated reconstruction to default to a fresh runtime seed (`--seed -1`) and fixed AR stage status so finite resumed branches with saved best checkpoints are not marked failed merely because final train CE plateaus. Helper tests now pass with `48 passed`.
- [x] (2026-07-03 19:22 +08:00) Started isolated `newscheme_full_v13_ar_lr5e6` from the epoch-95 `lr=1e-5` best checkpoint with `lr=5e-6` and target epoch 120; the branch resumed at epoch 95 and reached epoch 96 batch `2000/36289` with GPU utilization about `97%`.
- [x] (2026-07-03 19:34 +08:00) Re-ran generated reconstruction from the active `lr=5e-6` branch's `ar_best.pt` with two fresh-seed settings. The best current visualization candidate is `temperature=0.9`, `top_p=0.92`, `max_new_tokens=320`, which retained 6 STEP files with 6/6 grammar-valid, reconstructed, strict BREP-valid, and no errors.
- [x] (2026-07-03 19:40 +08:00) Fixed `tools\monitor_ar_epoch_gate.py` so resumed branches report the checkpoint epoch before the first new history row exists; restarted the active `lr=5e-6` epoch-120 monitor at `ar_lr5e6_epoch120_monitor_20260703_1942.jsonl` and stopped the stale `lr=1e-5` monitor.
- [x] (2026-07-03 19:46 +08:00) Rechecked the active `lr=5e-6` branch while epoch 96 was running: latest log reached batch `16000/36289`, history was still empty because validation had not run yet, and both `ar_best.pt` and `ar_latest.pt` still reflected the copied epoch-95 checkpoint.
- [x] (2026-07-03 20:24 +08:00) Verified the first completed `lr=5e-6` epoch: epoch 96 wrote `train_CE=0.2989420899666821`, `val_CE=0.2955302662144129`, did not beat epoch 95, and `ar_latest.pt` now records `learning_rate=5e-6` while `ar_best.pt` remains the epoch-95 best.
- [x] (2026-07-03 20:27 +08:00) Rechecked the active `lr=5e-6` branch after epoch 96: training had entered epoch 97 and reached batch `6000/36289`, monitor reported latest completed epoch 96, and GPU utilization remained about `95%`.
- [x] (2026-07-03 20:33 +08:00) Rechecked epoch 97: training reached batch `10000/36289`, GPU utilization was about `97%`, `ar_history.jsonl` still contained only the epoch-96 row, and `ar_best.pt` still pointed to the epoch-95 best checkpoint.
- [x] (2026-07-03 20:36 +08:00) Rechecked epoch 97 again: training reached batch `12000/36289`, GPU utilization was about `99%`, `ar_history.jsonl` still contained only the epoch-96 row, and `ar_best.pt` still pointed to the epoch-95 best checkpoint.
- [x] (2026-07-03 20:40 +08:00) Rechecked epoch 97 again: training reached batch `14000/36289`, GPU utilization was about `100%`, and the active monitor still reported latest completed epoch 96 while waiting for epoch 120.
- [x] (2026-07-03 20:46 +08:00) Rechecked epoch 97 again: training reached batch `18000/36289`, GPU utilization was about `90%`, `ar_history.jsonl` still contained only the epoch-96 row, and `ar_best.pt` still pointed to the epoch-95 best checkpoint.
- [x] (2026-07-03 20:52 +08:00) Rechecked epoch 97 again: training reached batch `22000/36289`, GPU utilization was about `96%`, monitor still reported latest completed epoch 96, and no checkpoint timestamp changed.
- [x] (2026-07-03 20:58 +08:00) Rechecked epoch 97 again: training reached batch `26000/36289`, GPU utilization was about `97%`, `ar_history.jsonl` still contained only the epoch-96 row, and no checkpoint timestamp changed.
- [x] (2026-07-03 21:08 +08:00) Rechecked epoch 97 again: training reached batch `32000/36289`, GPU utilization was about `100%`, `ar_history.jsonl` still contained only the epoch-96 row, and no checkpoint timestamp changed.
- [x] (2026-07-03 21:20 +08:00) Verified epoch 97 completed: `train_CE=0.30000715950230006`, `val_CE=0.295356298067103`, best remained epoch 95 at `0.2951271435705727`, `ar_latest.pt` CPU-loads at epoch 97 with model/optimizer/scaler state, the monitor reports latest completed epoch 97, and the training process has entered epoch 98.
- [x] (2026-07-03 21:29 +08:00) Rechecked epoch 98: training reached batch `8000/36289`, GPU utilization was about `96%`, `ar_history.jsonl` still contained through epoch 97 only, and no checkpoint timestamp changed.
- [x] (2026-07-03 21:42 +08:00) Rechecked epoch 98: training reached batch `16000/36289`, GPU utilization was about `96%`, training and monitor processes were still active, `ar_history.jsonl` still contained through epoch 97 only, and no checkpoint timestamp changed.
- [x] (2026-07-03 21:57 +08:00) Rechecked epoch 98: training reached batch `26000/36289`, GPU utilization was about `92%`, monitor still reported latest completed epoch 97, and no checkpoint timestamp changed.
- [x] (2026-07-03 22:18 +08:00) Verified epoch 98 completed: `train_CE=0.29992070919525726`, `val_CE=0.29545644794096076`, best remained epoch 95 at `0.2951271435705727`, `ar_latest.pt` CPU-loads at epoch 98 with model/optimizer/scaler state, the monitor reports latest completed epoch 98, and the training process has entered epoch 99.
- [x] (2026-07-03 22:27 +08:00) Rechecked epoch 99: training reached batch `8000/36289`, GPU utilization was about `100%`, `ar_history.jsonl` still contained through epoch 98 only, and no checkpoint timestamp changed.
- [x] (2026-07-03 22:40 +08:00) Rechecked epoch 99: training reached batch `16000/36289`, GPU utilization was about `97%`, monitor still reported latest completed epoch 98, and no checkpoint timestamp changed.
- [x] (2026-07-03 22:56 +08:00) Rechecked epoch 99: training reached batch `26000/36289`, GPU utilization was about `99%`, monitor still reported latest completed epoch 98, and no checkpoint timestamp changed.
- [x] (2026-07-03 23:16 +08:00) Verified epoch 99 completed: `train_CE=0.29984116893179646`, `val_CE=0.2953488017165304`, best remained epoch 95 at `0.2951271435705727`, `ar_latest.pt` CPU-loads at epoch 99 with model/optimizer/scaler state, the monitor reports latest completed epoch 99, and the training process has entered epoch 100.
- [x] (2026-07-03 23:35 +08:00) Rechecked epoch 100: training reached batch `14000/36289`, GPU utilization was about `99%`, monitor still reported latest completed epoch 99, `ar_epoch_0100.pt` did not exist yet, and no checkpoint timestamp changed.
- [x] (2026-07-03 23:57 +08:00) Rechecked epoch 100: training reached batch `28000/36289`, GPU utilization was about `94%`, monitor still reported latest completed epoch 99, `ar_epoch_0100.pt` did not exist yet, and no checkpoint timestamp changed.
- [x] (2026-07-04 00:17 +08:00) Verified epoch 100 completed: `train_CE=0.2996873783031376`, `val_CE=0.2954869701947432`, best remained epoch 95 at `0.2951271435705727`, `ar_latest.pt` and `ar_checkpoints\ar_epoch_0100.pt` both CPU-load at epoch 100 with model/optimizer/scaler state, the monitor reports the epoch-100 checkpoint ready, and the training process has entered epoch 101.
- [x] (2026-07-04 00:37 +08:00) Rechecked epoch 101: training reached batch `16000/36289`, GPU utilization was about `96%`, monitor still reported latest completed epoch 100, `ar_history.jsonl` still contained through epoch 100 only, and no checkpoint timestamp changed.
- [x] (2026-07-04 00:59 +08:00) Rechecked epoch 101: training reached batch `30000/36289`, GPU utilization was about `95%`, monitor still reported latest completed epoch 100, `ar_history.jsonl` still contained through epoch 100 only, and no checkpoint timestamp changed.
- [x] (2026-07-05 00:25 +08:00) Verified the `lr=5e-6` branch completed the target epoch 120 and stopped. `ar_history.jsonl` has 25 rows for epochs 96 through 120, epoch 120 is the best with `train_CE=0.3013073519163702`, `val_CE=0.29493329663972306`, and `best_val_CE=0.29493329663972306`, `ar_latest.pt`, `ar_best.pt`, and `ar_checkpoints\ar_epoch_0120.pt` all CPU-load with model/optimizer/scaler state and optimizer learning rate `5e-6`, the epoch-120 monitor reports `ready=true`, no matching training Python process remains, and GPU utilization was idle at about `0%`.
- [x] (2026-07-05 00:28 +08:00) Re-ran generated reconstruction from the refreshed epoch-120 `ar_best.pt` with `temperature=0.9`, `top_p=0.92`, `max_new_tokens=320`, and default fresh seed. Report `local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_002743\reconstruction_report.json` is `VERIFIED`, effective seed is `3899885580`, and 6 retained STEP files are grammar-valid, reconstructed, and strict BREP-valid.
- [x] (2026-07-05 00:29 +08:00) Re-ran generated reconstruction from epoch-120 `ar_best.pt` with a more conservative `temperature=0.75`, `top_p=0.88`, `max_new_tokens=320`, and default fresh seed. Report `local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp075_topp88_max320_random_cpu_20260705_002856\reconstruction_report.json` is `VERIFIED`, effective seed is `4275042948`, and 6 retained STEP files are grammar-valid, reconstructed, and strict BREP-valid.
- [x] (2026-07-05 00:30 +08:00) Re-ran generated reconstruction from epoch-120 `ar_best.pt` with a more diverse `temperature=1.0`, `top_p=0.95`, `max_new_tokens=340`, and default fresh seed. Report `local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp10_topp95_max340_random_cpu_20260705_002948\reconstruction_report.json` is `VERIFIED`, effective seed is `4105039696`, and it saved 5 BREP-valid STEP files out of 6 attempts; the sixth sample was grammar-failed because it truncated without an `END` token.
- [x] (2026-07-05 00:31 +08:00) Verified generated STEP diversity by SHA256 hash. Each new epoch-120 run has no duplicate STEP hashes within the run, and the 23 STEP files across the three new runs plus the older 2026-07-03 `temperature=0.9`, `top_p=0.92` run have 23 unique hashes.

## Surprises & Discoveries

- Observation: The final `lr=5e-6` continuation did improve after the epoch-100 plateau review and reached a new best exactly at the target epoch.
  Evidence: The epoch-120 row in `ar_history.jsonl` has `improved=true`, `val_ce=0.29493329663972306`, and `best_val_ce=0.29493329663972306`. CPU loading `ar_latest.pt`, `ar_best.pt`, and `ar_checkpoints\ar_epoch_0120.pt` showed all three at `epoch=120` with model, optimizer, and scaler state.
- Observation: Fresh-seed AR generation is no longer repeating retained STEP files across runs; repeated-looking outputs are more likely a model-quality and topology-distribution issue than a seed plumbing issue.
  Evidence: `tools\evaluate_reconstruction_v13.py` now defaults `--seed` to `-1`, resolves that to a time-derived `effective_seed`, records `requested_seed` and `effective_seed` in each report, and the latest reports used distinct effective seeds `3899885580`, `4275042948`, and `4105039696`. SHA256 hashing found 23 unique STEP hashes across 23 retained STEP files when comparing the three epoch-120 runs with the earlier 2026-07-03 run.
- Observation: Higher sampling diversity can reduce validity even with topology-constrained decoding.
  Evidence: At epoch 120, `temperature=0.9`, `top_p=0.92` and `temperature=0.75`, `top_p=0.88` each saved 6/6 strict BREP-valid STEP files, while `temperature=1.0`, `top_p=0.95`, `max_new_tokens=340` saved 5/6 and produced one grammar failure with reason `truncated (no END)`.
- Observation: The parsed `.pkl` files are not needed by the AR training stage once `sequences_fsq_rcm.pkl` exists, but they remain useful for regenerating sequence data or rerunning VQ-VAE/sequence experiments.
  Evidence: `breparg_improvements\train.py` `_load_ar_seqs()` loads only `SEQ_PKL`, and `stage_ar()` calls `_load_ar_seqs()` before training.
- Observation: The existing AR training function does not yet meet the requested operational requirements.
  Evidence: `_train_ar()` saves only `ar_best.pt` when validation improves, does not save `ar_latest.pt`, does not save epoch-20 checkpoint files, does not load a resume checkpoint, and writes only console logs.
- Observation: Local V13 storage is sufficient for AR inputs and outputs.
  Evidence: `Get-PSDrive D` showed about 148.9 GB free. The sequence file is about 1.4 GB and expected AR checkpoints are hundreds of MB each depending on model size.
- Observation: The V13-local AR preflight confirmed the sequence package is usable for AR but the 1024-token AR limit filters out long sequences.
  Evidence: `ar_preflight_report.json` status is `VERIFIED`, with `raw_train=382720`, `usable_train=290306`, `raw_val=21124`, `usable_val=16038`, `out_of_vocab=0`, `over_max_len=102574`, and `smoke_loss` about 9.2869.
- Observation: The first AR training epoch is progressing normally and the GPU is already near the requested utilization target.
  Evidence: `ar_v13_epoch100_20260628_135704.log` showed epoch 1 reaching batch `10000/36289` at 14:12:58 with running train CE `2.0048`; `nvidia-smi` showed GPU utilization about 98% and 8645-8651 MiB VRAM used.
- Observation: Parsed chunk archiving is progressing independently while AR trains.
  Evidence: `archive_parsed_chunks_20260628_133736.log` showed archives completed through `abc_0007`, with `abc_0008_parsed.zip.tmp` present.
- Observation: The first full AR epoch completed and produced resumable checkpoints.
  Evidence: `ar_v13_epoch100_20260628_135704.log` showed epoch 1 `train_CE=1.2521`, `val_CE=0.8038`, `best=0.8038`. `ar_history.jsonl` contains epoch 1 with `train_batches=36289`, `val_batches=2005`, and `elapsed_min=57.275`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=1`, `vocab_size=10294`, `d_model=256`, `layers=8`, `batch_size=8`, `learning_rate=0.0005`, plus model, optimizer, and scaler state dictionaries.
- Observation: AR training continued after first-epoch checkpointing.
  Evidence: The same log showed epoch 2 reaching batch `2000/36289` at 14:57:48 with running train CE `0.7548`.
- Observation: AR training remains healthy in epoch 2 and still saturates the GPU.
  Evidence: `tools\check_ar_archive_status.py --tail 8` showed epoch 2 reaching batch `14000/36289` at 15:16:30 with running train CE `0.7087`; `nvidia-smi` showed GPU utilization about 95-99% and about 8.6 GiB of 12 GiB VRAM used.
- Observation: Parsed archive creation continued past the first AR checkpoint.
  Evidence: The status checker showed archives completed through `abc_0023`, `manifest_lines=24`, and `abc_0024_parsed.zip.tmp` being written.
- Observation: AR training continued deeper into epoch 2 with stable speed and loss decline.
  Evidence: `tools\check_ar_archive_status.py --tail 8` showed epoch 2 batch `18000/36289`, running train CE `0.6972`, recent speed about `641` batches/minute, and estimated epoch remaining time about `28.53` minutes.
- Observation: Parsed archive creation continued to advance.
  Evidence: The status checker showed archives completed through `abc_0025`, `manifest_lines=26`, and `abc_0026_parsed.zip.tmp` being written.
- Observation: AR training remained healthy with the same hyperparameters later in epoch 2.
  Evidence: A fresh status sample showed epoch 2 batch `22000/36289`, running train CE `0.6856`, recent speed about `651` batches/minute, estimated epoch remaining time about `21.93` minutes, and `nvidia-smi` GPU utilization at about 99%.
- Observation: Parsed archive creation continued beyond chunk 25.
  Evidence: The archive log and status checker showed archives completed through `abc_0026`, `manifest_lines=27`, and `abc_0027_parsed.zip.tmp` being written.
- Observation: AR epoch 2 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 2 with `train_ce=0.6574728841948527`, `val_ce=0.5912807370399001`, `best_val_ce=0.5912807370399001`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=2`, `d_model=256`, `layers=8`, `batch_size=8`, `learning_rate=0.0005`, plus model, optimizer, and scaler state dictionaries.
- Observation: AR training continued into epoch 3 after writing epoch 2 checkpoints.
  Evidence: `tools\check_ar_archive_status.py --tail 18` showed epoch 3 batch `4000/36289`, running train CE `0.5859`, and GPU utilization about 99%.
- Observation: Parsed archive creation advanced while AR trained.
  Evidence: The archive log and status checker showed archives completed through `abc_0033`, `manifest_lines=34`, and `abc_0034_parsed.zip.tmp` being written.
- Observation: AR training continued through early epoch 3 with stable utilization.
  Evidence: A fresh status sample showed epoch 3 batch `6000/36289`, running train CE `0.5852`, GPU utilization about 99%, and 8623-8637 MiB VRAM in use.
- Observation: Parsed archive creation continued past one third of the dataset.
  Evidence: The archive log and status checker showed archives completed through `abc_0034`, `manifest_lines=35`, and `abc_0035_parsed.zip.tmp` being written.
- Observation: AR training continued deeper into epoch 3 without changing hyperparameters.
  Evidence: A fresh status sample showed epoch 3 batch `10000/36289`, running train CE `0.5769`, recent speed about `658` batches/minute, estimated epoch remaining time about `39.96` minutes, and `nvidia-smi` GPU utilization at about 98%.
- Observation: The epoch 2 resumability checkpoints are valid after continued training.
  Evidence: CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=2`, `train_ce=0.6574728841948527`, `val_ce=0.5912807370399001`, `best_val_ce=0.5912807370399001`, `d_model=256`, `layers=8`, `batch_size=8`, `learning_rate=0.0005`, plus model, optimizer, and scaler state dictionaries.
- Observation: Periodic AR checkpoint files are not expected yet.
  Evidence: `ar_checkpoints` contained no files while the latest completed epoch was 2; `NS_AR_SAVE_EVERY=20`, so the first periodic file should be `ar_epoch_0020.pt` after epoch 20 completes.
- Observation: Parsed archive creation continued beyond chunk 36.
  Evidence: The archive log and status checker showed archives completed through `abc_0036`, `manifest_lines=37`, and `abc_0037_parsed.zip.tmp` being written.
- Observation: AR epoch 3 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 3 with `train_ce=0.5525593687485449`, `val_ce=0.5024932347739723`, `best_val_ce=0.5024932347739723`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=3` plus model, optimizer, and scaler state dictionaries.
- Observation: The original `BrepARG\generate_brep.py` and `BrepARG\utils.py::load_se_vqvae_model` are not safe to use directly with this run's `fsq_vqvae_best.pt`.
  Evidence: Those paths instantiate the original `VectorQuantiser`, while the trained checkpoint uses `breparg_improvements\train.py::build_fsq_vqvae` and `FSQQuantiser`. Direct reconstruction initially failed because `BrepARG\utils.py::decode_tokens_to_ncs` expected a 64-channel decoder embedding but FSQ exposes 4 scalar-code channels.
- Observation: Current Windows reconstruction needs a local CPU-safe optimization path.
  Evidence: The upstream `BrepARG\utils.py::joint_optimize` hard-codes `.cuda()` and then calls the installed `chamferdist` extension, which failed with `RuntimeError: Not compiled with GPU support.` The evaluator now patches only the current process to use a `torch.cdist` CPU implementation, leaving training code untouched.
- Observation: The latest AR best checkpoint can already produce at least one reconstructable sample when sampled with topology-constrained decoding.
  Evidence: `tools\evaluate_reconstruction_v13.py --source generated --max-samples 1 --device cpu --constrained-decoding --max-new-tokens 160 --write-step --validate-step --run-name smoke_generated_step_arbest_constrained --seed 0` wrote `D:\luolin\V13\local_runs\reconstruction_eval\smoke_generated_step_arbest_constrained\steps\generated_000000_len0119.step`; the report status is `VERIFIED`, with `grammar_valid=1`, `step_saved=1`, and `brep_valid=1`.
- Observation: Parsed archive recovery correctly did not skip the missing `abc_0040` archive, but the `E:` drive disappeared before recovery could complete.
  Evidence: A resumed archive process skipped valid archives `abc_0000` through `abc_0039` and began writing `abc_0040_parsed.zip.tmp`. Later `Get-PSDrive E` failed with `Cannot find drive. A drive with the name 'E' does not exist`; `Get-Volume`, `Get-CimInstance Win32_LogicalDisk`, and `Get-Partition` showed only `C:` and `D:` available.
- Observation: AR training is isolated on `D:` and continued after `E:` disappeared.
  Evidence: A status sample after the `E:` loss showed AR entering epoch 5 at batch `2000/36289`, with GPU utilization around 99%.
- Observation: AR epoch 4 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 4 with `train_ce=0.5031435569475884`, `val_ce=0.4710762618430089`, `best_val_ce=0.4710762618430089`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=4` plus model, optimizer, and scaler state dictionaries.
- Observation: The status checker needed ASCII-safe JSON for Windows console compatibility.
  Evidence: A monitor run failed with `UnicodeEncodeError: 'gbk' codec can't encode character '\ue160'`. A regression test now covers this case and the status checker renders JSON with escaped non-ASCII characters.
- Observation: The status checker now reports partial drive availability correctly.
  Evidence: With `E:` missing, a fresh status sample still reported `D` free space and included a structured `E` error. `tests/test_local_pipeline_helpers.py` passed with `30 passed`.
- Observation: AR epoch 5 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 5 with `train_ce=0.4738814607785131`, `val_ce=0.44519996965615233`, `best_val_ce=0.44519996965615233`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=5` plus model, optimizer, and scaler state dictionaries.
- Observation: AR epoch 6 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 6 with `train_ce=0.4539241893644274`, `val_ce=0.42378115375291675`, `best_val_ce=0.42378115375291675`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=6` plus model, optimizer, and scaler state dictionaries.
- Observation: AR epoch 7 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 7 with `train_ce=0.4390543853502878`, `val_ce=0.41660729310384714`, `best_val_ce=0.41660729310384714`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=7` plus model, optimizer, and scaler state dictionaries.
- Observation: AR epoch 8 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 8 with `train_ce=0.42693425646102734`, `val_ce=0.4014375453299269`, `best_val_ce=0.4014375453299269`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=8` plus model, optimizer, and scaler state dictionaries.
- Observation: Parsed archive recovery resumed successfully after `E:` became visible again.
  Evidence: A fresh status sample showed `E:` with about 259.8 GB free, the archive process running again, `abc_0040_parsed.zip` recreated with `archive_bytes=1836381272`, and `abc_0041_parsed.zip.tmp` being written.
- Observation: AR epoch 9 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 9 with `train_ce=0.4172350860857129`, `val_ce=0.39118778486167105`, `best_val_ce=0.39118778486167105`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=9` plus model, optimizer, and scaler state dictionaries.
- Observation: Parsed archive creation continued after recovery.
  Evidence: The archive manifest shows archived rows from `abc_0040` through `abc_0054`; a status sample showed `zip_count=54` and `abc_0054_parsed.zip` completed.
- Observation: AR epoch 10 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 10 with `train_ce=0.40916101405348143`, `val_ce=0.3850843170392989`, `best_val_ce=0.3850843170392989`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=10` plus model, optimizer, and scaler state dictionaries.
- Observation: Parsed archive creation advanced past two thirds of the chunks.
  Evidence: The archive manifest shows archived rows through `abc_0068`; a status sample showed `zip_count=68` and `abc_0068_parsed.zip` completed.
- Observation: AR epoch 11 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 11 with `train_ce=0.40228315561835504`, `val_ce=0.3825993174812443`, `best_val_ce=0.3825993174812443`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=11` plus model, optimizer, and scaler state dictionaries.
- Observation: Parsed archive creation advanced into the final fifth of chunks, and a stale temp file from the earlier `E:` disconnect was safely removed.
  Evidence: The archive log shows archived rows through `abc_0082` and `abc_0083_parsed.zip.tmp` being written. `abc_0041_parsed.zip` passed `zipfile.testzip()` with no bad member, after which the stale `abc_0041_parsed.zip.tmp` was removed.
- Observation: AR epoch 12 completed, improved validation loss, and wrote resumable latest/best checkpoints.
  Evidence: `ar_history.jsonl` contains epoch 12 with `train_ce=0.39649410939763935`, `val_ce=0.37262188478598274`, `best_val_ce=0.37262188478598274`, `train_batches=36289`, and `val_batches=2005`. CPU loading both `ar_best.pt` and `ar_latest.pt` showed `epoch=12` plus model, optimizer, and scaler state dictionaries.
- Observation: Parsed archive creation completed all chunks.
  Evidence: The archive log shows `archived abc_0099`; `E:\ABC\processed\abc_parsed_full_archives` contains 100 `abc_*_parsed.zip` files, zero `.tmp` files, and `_manifest.jsonl` has 181 lines including resumed skip records and archive records.
- Observation: The first periodic AR checkpoint was created and is usable for resume/recovery.
  Evidence: `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_checkpoints\ar_epoch_0020.pt` exists with 110716810 bytes. CPU loading it showed `epoch=20`, `train_ce=0.36784262608729257`, `val_ce=0.35454165665827636`, `best_val_ce=0.35289947435241237`, `d_model=256`, `layers=8`, `batch_size=8`, `learning_rate=0.0005`, plus model, optimizer, and scaler state dictionaries.
- Observation: The AR validation curve has entered a plateau by epoch 20.
  Evidence: Best validation CE improved to `0.35289947435241237` at epoch 19, while epoch 20 had `val_ce=0.35454165665827636` and `improved=false`. The latest checkpoint is epoch 20, while `ar_best.pt` remains epoch 19.
- Observation: Training after epoch 20 is not wasted so far; the validation curve resumed improving after the temporary plateau.
  Evidence: `ar_history.jsonl` shows new best validation CE at epoch 22 (`0.34779535751520396`), epoch 24 (`0.34419052741604106`), epoch 25 (`0.34388686744983954`), epoch 26 (`0.34342681005355574`), epoch 27 (`0.3416744249382816`), epoch 28 (`0.34150161020587805`), epoch 29 (`0.33840847481645997`), and epoch 30 (`0.33647562982769974`). CPU loading `ar_best.pt` and `ar_latest.pt` showed both at epoch 29 with model, optimizer, and scaler state dictionaries before epoch 30 completed; the status checker then reported updated latest/best checkpoint mtimes for epoch 30.
- Observation: AR training remains hardware-saturated enough that increasing batch size or changing model shape mid-run is not justified right now.
  Evidence: Fresh status samples around epoch 30 completion showed recent speed about `636-667` batches/minute, GPU utilization about `97-98%`, memory use about `8537 MiB` of `12288 MiB`, temperature about `73 C`, and power about `147-149 W`.
- Observation: Parsed archive creation is complete and no archive temp files remain.
  Evidence: `E:\ABC\processed\abc_parsed_full_archives` contains 100 `abc_*_parsed.zip` files, zero `.tmp` files, and `_manifest.jsonl` has 181 lines including resumed skip and archive records.
- Observation: The original four-part objective is not fully complete yet because the full AR training run is still active.
  Evidence: A fresh status sample at `2026-06-29 18:40 +08:00` showed AR process `25996` running in epoch 31 at batch `4000/36289`, `ar_history.jsonl` containing completed epochs 1-30, GPU utilization at `98-100%`, and only the first periodic checkpoint `ar_epoch_0020.pt` present so far. The V13 AR preflight report remained `VERIFIED`, `ar_best.pt` and `ar_latest.pt` CPU-loaded at epoch 30 with model, optimizer, and scaler states, and `tests/test_local_pipeline_helpers.py` passed with `30 passed`.
- Observation: The epoch-40 decision point now has an explicit read-only gate.
  Evidence: `tools\monitor_ar_epoch_gate.py --once --target-epoch 40` returned `ready=false`, `latest_epoch=30`, and `reason=waiting_for_epoch_40`. The long-running monitor process `2644` is polling every 300 seconds and writing status rows to `D:\luolin\V13\local_runs\ar_training\logs\ar_epoch40_gate_20260629_184918.jsonl`; its stderr log is empty. The helper tests now include this behavior and passed with `32 passed`.
- Observation: A read-only AR history analysis at epoch 30 does not show overfitting or a plateau that justifies interrupting the current run.
  Evidence: `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5` reported `status=VERIFIED`, `latest_epoch=30`, `best_epoch=30`, `best_val_ce=0.33647562982769974`, `epochs_since_best=0`, `overfit_signal=false`, and `recommendation=continue_unchanged`. The same run estimated about `66.79` hours from completed epoch 30 to epoch 100 at the recent mean of `57.25` minutes per epoch.
- Observation: Epoch 31 continued the validation improvement, so training past epoch 30 is still productive.
  Evidence: `ar_history.jsonl` contains epoch 31 with `train_ce=0.34892181385547794`, `val_ce=0.33623531360337117`, `best_val_ce=0.33623531360337117`, and `improved=true`. CPU loading `ar_best.pt` and `ar_latest.pt` showed both at `epoch=31` with `vocab_size=10294`, `d_model=256`, `layers=8`, `batch_size=8`, `learning_rate=0.0005`, plus model, optimizer, and scaler state dictionaries. A fresh GPU sample showed about `96%` utilization and `8604 / 12288 MiB` VRAM used.
- Observation: D drive nearly filled during epoch 32 because a duplicate parsed-archive mirror had been created under the V13 working tree.
  Evidence: `Get-PSDrive D` reported only `423555072` bytes free while `D:\luolin\V13\ABC\processed\abc_parsed_full_archives` contained 81 `abc_*_parsed.zip` files using about `131.612` GiB. A comparison script showed every D-side zip was present on `E:\ABC\processed\abc_parsed_full_archives` with the same byte size, no D-only files, and no size mismatches; E-side archive count remained 100. After deleting only the D-side duplicate directory, D free space rose to `141741121536` bytes, and AR training continued in epoch 32.
- Observation: The data storage layout changed after the E drive was cleared: `D:\luolin\V13\ABC` is now the offline ABC data and experiment-output mirror.
  Evidence: `D:\luolin\V13\ABC\processed\abc_parsed_full_archives` contains 100 `abc_*_parsed.zip` files totaling `174374864736` bytes, zero `.tmp` files, and `_manifest.jsonl` has 181 lines. `D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100` contains `sequences_fsq_rcm.pkl`, `sequences_fsq_rcm_sharded_merged.pkl`, `split.pkl`, `fsq_vqvae_best.pt`, `fsq_vqvae_final.pt`, `vqvae_history.json`, and `sequence_shards`. `Get-PSDrive E` reported only `137981952` bytes used, consistent with the user's note that E has been cleared for ejection.
- Observation: The active AR training process is separated from E and can continue if E is ejected.
  Evidence: `tools\run_ar_v13_epoch100.ps1` sets `NS_OUTBASE=D:\luolin\V13\local_runs\ar_training\train_outputs`, and the active command line is `breparg_improvements\train.py --stage ar`. The live AR input `sequences_fsq_rcm.pkl`, `ar_latest.pt`, `ar_best.pt`, `ar_history.jsonl`, and `ar_v13_epoch100_20260628_135704.log` are all under `D:\luolin\V13\local_runs\ar_training`. A fresh status sample from `tools\check_ar_archive_status.py` using D defaults showed AR in epoch 35, archive `zip_count=100`, and GPU utilization about 95%.
- Observation: The epoch-40 gate passed and the run should continue unchanged to the next checkpoint review.
  Evidence: `tools\monitor_ar_epoch_gate.py --once --target-epoch 40` returned `ready=true` with checkpoint `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_checkpoints\ar_epoch_0040.pt`, 110716810 bytes, `epoch=40`, `train_ce=0.3403130233437792`, `val_ce=0.33065376463543594`, `best_val_ce=0.3302254088390201`, and model, optimizer, and scaler states present. CPU loading `ar_best.pt`, `ar_latest.pt`, `ar_epoch_0020.pt`, and `ar_epoch_0040.pt` succeeded. `ar_best.pt` remains epoch 36 with `val_ce=0.3302254088390201`; latest and epoch-40 are epoch 40. `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5` reported `overfit_signal=false` and `recommendation=continue_unchanged`. A fresh GPU sample showed about `99%` utilization and D had about `78.8` GB free.
- Observation: The run has not reached the epoch-60 decision gate yet, but live execution is healthy.
  Evidence: A fresh `tools\check_ar_archive_status.py --tail 80` sample showed the AR Python process `25996`, launcher process `40824`, epoch 41 batch `12000/36289`, running train CE `0.3372`, recent speed about `634.92` batches/minute, and estimated epoch remaining time about `38.26` minutes. `nvidia-smi` reported RTX 3060 utilization `99%`, `8541 / 12288` MiB VRAM, `75 C`, and about `151 W`. `tools\monitor_ar_epoch_gate.py --once --target-epoch 60` returned `ready=false`, `latest_epoch=40`, and `reason=waiting_for_epoch_60`. A background read-only epoch-60 monitor is running as Python PID `4828` and wrote the same waiting status to `D:\luolin\V13\local_runs\ar_training\logs\ar_epoch60_gate_20260630_042607.jsonl`.
- Observation: Epoch 41 made the plateau signal strong enough to change course before epoch 60.
  Evidence: `ar_latest.pt` CPU-loaded successfully with `epoch=41`, `train_ce=0.33961104362725286`, `val_ce=0.33155879084011564`, `best_val_ce=0.3302254088390201`, and model, optimizer, and scaler states present. `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5` then reported `plateau_signal=true`, `overfit_signal=true`, `epochs_since_best=5`, `best_epoch=36`, and `recommendation=consider_stop_or_lower_lr`.
- Observation: Resuming from an AR optimizer checkpoint previously kept the optimizer's old learning rate even when the new run requested a lower `NS_AR_LR`.
  Evidence: The new test `test_ar_resume_applies_requested_learning_rate_to_optimizer` failed before the fix with `AssertionError: 0.001 != 0.0002`, showing the resumed checkpoint metadata and optimizer parameter group could disagree. After setting each optimizer param group to the requested `lr` after `opt.load_state_dict`, the full helper suite passed with `37 passed`.
- Observation: The lower-learning-rate continuation is active and using the intended checkpoint and learning rate.
  Evidence: The new log `D:\luolin\V13\local_runs\ar_training\logs\ar_v13_epoch100_20260630_051440.log` shows `train=290306 val=16038 vocab=10294 d_model=256 layers=8 bs=8 lr=0.0001` and `resumed from ...\ar_latest.pt at epoch=41 best=0.3302 lr=0.0001`. A status sample showed launcher PID `37868`, training Python PID `3884`, epoch 42 batch `2000/36289`, running train CE `0.3232`, and GPU utilization around `95-99%`.
- Observation: The lower-learning-rate continuation produced a large validation improvement on the first completed epoch after the restart.
  Evidence: Epoch 42 wrote `train_ce=0.3191331968506213`, `val_ce=0.30511944202840924`, `best_val_ce=0.30511944202840924`, and `improved=true` to `ar_history.jsonl`. CPU loading `ar_latest.pt` and `ar_best.pt` showed both checkpoints are epoch 42, include model, optimizer, and scaler states, and have both checkpoint metadata `learning_rate=0.0001` and optimizer param group learning rates `[0.0001]`. `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5` now reports `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`.
- Observation: The lower-learning-rate continuation kept improving at epoch 43.
  Evidence: Epoch 43 wrote `train_ce=0.31519020255924524`, `val_ce=0.30437195893850855`, `best_val_ce=0.30437195893850855`, and `improved=true` to `ar_history.jsonl`. CPU loading both `ar_latest.pt` and `ar_best.pt` showed `epoch=43`, model, optimizer, and scaler states present, checkpoint `learning_rate=0.0001`, and optimizer learning rates `[0.0001]`. The epoch-60 monitor remains `ready=false` with `latest_epoch=43`, so the run should continue unchanged to the next periodic checkpoint.
- Observation: The lower-learning-rate continuation also improved at epoch 44.
  Evidence: Epoch 44 wrote `train_ce=0.3138610266946477`, `val_ce=0.3035273262473785`, `best_val_ce=0.3035273262473785`, and `improved=true` to `ar_history.jsonl`. CPU loading both `ar_latest.pt` and `ar_best.pt` showed `epoch=44`, model, optimizer, and scaler states present, checkpoint `learning_rate=0.0001`, and optimizer learning rates `[0.0001]`. `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5` reports `overfit_signal=false`, `plateau_signal=false`, and `recommendation=continue_unchanged`.
- Observation: The lower-learning-rate continuation still had useful improvement left at epoch 53.
  Evidence: Epoch 53 wrote `train_ce=0.3103011045542324`, `val_ce=0.30052309570894425`, `best_val_ce=0.30052309570894425`, and `improved=true` to `ar_history.jsonl`. CPU loading `ar_latest.pt` and `ar_best.pt` showed both checkpoints are epoch 53, include model, optimizer, and scaler states, and have checkpoint `learning_rate=0.0001` with optimizer learning rates `[0.0001]`. `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5 --output D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_analysis_latest.json` reported `best_epoch=53`, `best_val_ce=0.30052309570894425`, `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`. A fresh status sample showed the live run continuing with GPU utilization about `96-99%`.
- Observation: The lower-learning-rate continuation crossed below `0.30` validation CE at epoch 56.
  Evidence: Epoch 56 wrote `train_ce=0.3096163382570437`, `val_ce=0.29962874219155966`, `best_val_ce=0.29962874219155966`, and `improved=true` to `ar_history.jsonl`. CPU loading `ar_latest.pt` and `ar_best.pt` showed both checkpoints are epoch 56, include model, optimizer, and scaler states, and have checkpoint `learning_rate=0.0001` with optimizer learning rates `[0.0001]`. `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5 --output D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_analysis_latest.json` reported `best_epoch=56`, `best_val_ce=0.29962874219155966`, `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`. Ten consecutive GPU samples were mostly `92-100%`, with one sample at `88%`, so utilization remains near the requested target.
- Observation: The epoch-60 gate passed, and there is not enough evidence to stop or change hyperparameters.
  Evidence: `tools\monitor_ar_epoch_gate.py --once --target-epoch 60` returned `ready=true` with checkpoint `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_checkpoints\ar_epoch_0060.pt`, 110717898 bytes, `epoch=60`, `train_ce=0.3088528621460758`, `val_ce=0.2997024929898784`, `best_val_ce=0.29962874219155966`, and model, optimizer, and scaler states present. CPU loading `ar_latest.pt`, `ar_best.pt`, `ar_epoch_0020.pt`, `ar_epoch_0040.pt`, and `ar_epoch_0060.pt` succeeded. `ar_latest.pt` is epoch 60 with optimizer learning rate `[0.0001]`; `ar_best.pt` remains epoch 56 with optimizer learning rate `[0.0001]`. `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5 --output D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_analysis_latest.json` reported `best_epoch=56`, `epochs_since_best=4`, `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`. The live process continued into epoch 61 with GPU utilization about `93%` and D drive free space about `78.7` GB.
- Observation: The `lr=1e-4` continuation became exhausted after epoch 60 and should not continue burning GPU unchanged.
  Evidence: Epochs 61 through 66 did not improve over the epoch-56 best. `tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5` reported `recommendation=consider_stop_or_lower_lr` at epoch 61 and continued doing so through epoch 66; at epoch 66 it reported `latest_val_ce=0.3000866492220811`, `best_epoch=56`, `best_val_ce=0.29962874219155966`, `epochs_since_best=10`, `plateau_signal=true`, and `overfit_signal=true`. The train CE was still drifting down to `0.30771254730125774`, so continuing the same schedule was likely optimizing train loss without validation benefit.
- Observation: The new `lr=5e-5` branch is isolated from the old history and is using the intended checkpoint, input files, and GPU.
  Evidence: `tools\run_ar_v13_epoch100.ps1` now accepts `-RunName`, `-OutBase`, `-SequenceSource`, and `-SplitSource`; it seeds a new run by copying the requested resume checkpoint to the new branch's `ar_latest.pt` and `ar_best.pt`. The active command line is `run_ar_v13_epoch100.ps1 -RunName newscheme_full_v13_ar_lr5e5 ... -ResumeFrom ...\newscheme_full_v13_ar\ar_best.pt -LearningRate 5e-5 -TargetEpochs 100 -NoAutoResume`, with launcher PID `30552` and training Python PID `23900`. The new log `D:\luolin\V13\local_runs\ar_training\logs\ar_newscheme_full_v13_ar_lr5e5_20260701_063612.log` shows `train=290306 val=16038 vocab=10294 d_model=256 layers=8 bs=8 lr=5e-05` and `resumed from ...\newscheme_full_v13_ar_lr5e5\ar_latest.pt at epoch=56 best=0.2996 lr=5e-05`. The new branch has its own `sequences_fsq_rcm.pkl`, `split.pkl`, `ar_latest.pt`, and `ar_best.pt`, and status checks show epoch 57 batch progress with GPU utilization around `95%`.
- Observation: The first completed `lr=5e-5` epoch improved validation and refreshed the branch best.
  Evidence: `newscheme_full_v13_ar_lr5e5\ar_history.jsonl` now contains one row: epoch 57 with `train_ce=0.3055522505285005`, `val_ce=0.29771688658363205`, `best_val_ce=0.29771688658363205`, and `improved=true`. CPU loading both `ar_latest.pt` and `ar_best.pt` showed epoch 57, model/optimizer/scaler states present, checkpoint `learning_rate=5e-05`, and optimizer param group learning rates `[5e-05]`. `tools\analyze_ar_training.py --history D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e5\ar_history.jsonl --target-epoch 100 --recent-window 5 --plateau-patience 5` reported `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`. `tools\check_ar_archive_status.py --ar-out D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e5 --ar-log-dir D:\luolin\V13\local_runs\ar_training\logs --tail 12` now reads the new branch log correctly and shows epoch 57 batch progress with GPU utilization around `95%`.
- Observation: The `lr=5e-5` branch remained healthy through epoch 58 but did not improve over the new best.
  Evidence: `newscheme_full_v13_ar_lr5e5\ar_history.jsonl` now contains epoch 58 with `train_ce=0.3049508988477031`, `val_ce=0.2980387463729057`, `best_val_ce=0.29771688658363205`, and `improved=false`. CPU loading `ar_latest.pt` showed epoch 58 while `ar_best.pt` remains epoch 57, and both carry checkpoint `learning_rate=5e-05` with optimizer param group learning rates `[5e-05]`. `tools\analyze_ar_training.py --history D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e5\ar_history.jsonl --target-epoch 100 --recent-window 5 --plateau-patience 5` still reported `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`. The branch log and status checker show epoch 58 batch progress and GPU utilization around `93-99%`.
- Observation: The `lr=5e-5` branch plateaued after its epoch-57 best, so continuing the same schedule is no longer the best use of GPU.
  Evidence: `newscheme_full_v13_ar_lr5e5\ar_history.jsonl` contains epoch 62 with `train_ce=0.30427138190425473`, `val_ce=0.29808368393013307`, `best_val_ce=0.29771688658363205`, and `improved=false`. CPU loading `ar_best.pt`, `ar_latest.pt`, and `ar_checkpoints\ar_epoch_0060.pt` succeeded; all include model, optimizer, and scaler states with optimizer learning rates `[5e-05]`. `tools\analyze_ar_training.py --history D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e5\ar_history.jsonl --target-epoch 100 --recent-window 5 --plateau-patience 5` reported `epochs_since_best=5`, `plateau_signal=true`, `overfit_signal=true`, and `recommendation=consider_stop_or_lower_lr`.
- Observation: The isolated `lr=2e-5` branch started from the epoch-57 `lr=5e-5` best checkpoint and entered training normally.
  Evidence: The old `lr=5e-5` Python process and launcher were stopped, and `nvidia-smi` dropped to about `1%` utilization before restart. The new launcher command is `run_ar_v13_epoch100.ps1 -RunName newscheme_full_v13_ar_lr2e5 ... -ResumeFrom ...\newscheme_full_v13_ar_lr5e5\ar_best.pt -LearningRate 2e-5 -TargetEpochs 100 -NoAutoResume`, with launcher PID `40904` and Python PID `26968`. The new log `D:\luolin\V13\local_runs\ar_training\logs\ar_newscheme_full_v13_ar_lr2e5_20260701_124445.log` shows `train=290306 val=16038 vocab=10294 d_model=256 layers=8 bs=8 lr=2e-05` and `resumed from ...\newscheme_full_v13_ar_lr2e5\ar_latest.pt at epoch=57 best=0.2977 lr=2e-05`; it reached epoch 58 batch `2000/36289` with GPU utilization around `89-92%`.
- Observation: The first completed `lr=2e-5` epoch improved validation and should continue unchanged.
  Evidence: `newscheme_full_v13_ar_lr2e5\ar_history.jsonl` contains epoch 58 with `train_ce=0.2999067451803234`, `val_ce=0.2973588657237942`, `best_val_ce=0.2973588657237942`, and `improved=true`. CPU loading `ar_best.pt` and `ar_latest.pt` showed both at epoch 58 with model, optimizer, and scaler states present, checkpoint `learning_rate=2e-05`, and optimizer param group learning rates `[2e-05]`. The analyzer command with `--baseline-best-epoch 57 --baseline-best-val-ce 0.29771688658363205` returned `best_epoch=58`, `epochs_since_best=0`, `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`.
- Observation: The epoch-58 `lr=2e-5` best checkpoint preserves reconstruction quality on the retained validation smoke set.
  Evidence: `tools\evaluate_reconstruction_v13.py --device cpu --write-step --validate-step` wrote report `D:\luolin\V13\local_runs\reconstruction_eval\eval_validation_short5_arbest_lr2e5_epoch58_cpu\reconstruction_report.json` with `status=VERIFIED`, `attempted=5`, `grammar_valid=5`, `reconstruct_success=5`, `step_saved=5`, `brep_valid=4`, and `errors=0`. The retained STEP files are under `D:\luolin\V13\local_runs\reconstruction_eval\eval_validation_short5_arbest_lr2e5_epoch58_cpu\steps`.
- Observation: Validation-set reconstruction and generated reconstruction answer different questions, and the validation smoke set is deterministic by design.
  Evidence: `tools\evaluate_reconstruction_v13.py --source validation --order shortest --max-samples 5` always selects the same five shortest validation sequences and decodes those fixed tokens through the VQ-VAE path, so repeated runs produce the same STEP filenames and geometry. That mode checks VQ-VAE decode/reconstruction on known tokens; it is not evidence that the AR model is generating identical CADs. The true AR path is `--source generated`, which samples token sequences from `ar_best.pt`.
- Observation: The epoch-59 `lr=2e-5` AR best checkpoint generates BREP-valid STEP files in the current constrained decoding smoke tests.
  Evidence: Epoch 59 wrote `val_CE=0.29660862653964476` and refreshed `ar_best.pt`. CPU loading `ar_best.pt` and `ar_latest.pt` showed epoch 59, model/optimizer/scaler states present, checkpoint `learning_rate=2e-05`, and optimizer learning rates `[2e-05]`. Two generated runs were executed with `--source generated --constrained-decoding --write-step --validate-step`: `eval_generated5_arbest_lr2e5_epoch59_seed0_cpu` and `eval_generated5_arbest_lr2e5_epoch59_seed1_cpu`. Each report is `VERIFIED` with `attempted=5`, `grammar_valid=5`, `reconstruct_success=5`, `step_saved=5`, `brep_valid=5`, and `errors=0`; the retained STEP files live under each run's `steps` directory.
- Observation: The epoch-60 periodic checkpoint gate passed for the active `lr=2e-5` branch.
  Evidence: `tools\monitor_ar_epoch_gate.py --out-dir D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr2e5 --target-epoch 60` returned `ready=true` with `ar_epoch_0060.pt`, epoch 60, `val_ce=0.2967708992671937`, `best_val_ce=0.29660862653964476`, and model, optimizer, and scaler states present. CPU loading `ar_best.pt`, `ar_latest.pt`, and `ar_checkpoints\ar_epoch_0060.pt` succeeded; `ar_best.pt` remains epoch 59 while latest and the periodic checkpoint are epoch 60, all with optimizer learning rates `[2e-05]`. The analyzer reported `epochs_since_best=1`, `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`.
- Observation: Epoch 61 did not beat the epoch-59 best but remains within the normal post-best window.
  Evidence: `ar_history.jsonl` contains epoch 61 with `train_ce=0.30264771853747185`, `val_ce=0.2966308543054466`, `best_val_ce=0.29660862653964476`, and `improved=false`. CPU loading `ar_best.pt` and `ar_latest.pt` succeeded; best remains epoch 59, latest is epoch 61, both include model/optimizer/scaler states and optimizer learning rates `[2e-05]`. The analyzer reported `epochs_since_best=2`, `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`.
- Observation: The `lr=2e-5` branch became exhausted after epoch 84 and should not continue unchanged.
  Evidence: On 2026-07-02 15:19 +08:00, `ar_history.jsonl` contained 27 completed rows through epoch 84. The best checkpoint was epoch 76 with `val_CE=0.29538217296178204`; latest was epoch 84 with `val_CE=0.2958134048272324`; `ar_epoch_0080.pt` existed and CPU-loaded with model, optimizer, and scaler state. The analyzer reported `epochs_since_best=8`, `plateau_signal=true`, `overfit_signal=true`, and `recommendation=consider_stop_or_lower_lr`.
- Observation: The isolated `lr=1e-5` branch is now the active AR training branch.
  Evidence: The `lr=2e-5` Python process and launcher were stopped and GPU usage dropped to 0%. `tools\run_ar_v13_epoch100.ps1` was launched as `newscheme_full_v13_ar_lr1e5` from `newscheme_full_v13_ar_lr2e5\ar_best.pt` with `LearningRate 1e-5` and `TargetEpochs 100`. The log `D:\luolin\V13\local_runs\ar_training\logs\ar_newscheme_full_v13_ar_lr1e5_20260702_152112.log` shows `train=290306 val=16038 vocab=10294 d_model=256 layers=8 bs=8 lr=1e-05` and `resumed from ...\newscheme_full_v13_ar_lr1e5\ar_latest.pt at epoch=76 best=0.2954 lr=1e-05`. GPU utilization returned to about `92-93%`.
- Observation: The `lr=1e-5` branch completed epoch 78 and is still within the continue-unchanged window while epoch 79 trains.
  Evidence: `newscheme_full_v13_ar_lr1e5\ar_history.jsonl` contains epoch 78 with `train_ce=0.3008974045566388`, `val_ce=0.29567606687322817`, `best_val_ce=0.29538217296178204`, and `improved=false`. CPU loading `ar_latest.pt` showed epoch 78 with model, optimizer, and scaler states present, checkpoint `learning_rate=1e-05`, and optimizer learning rates `[1e-05]`. The analyzer command with `--baseline-best-epoch 76 --baseline-best-val-ce 0.29538217296178204` reported `epochs_since_best=2`, `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`. `check_ar_archive_status.py` shows the live `breparg_improvements\\train.py --stage ar` process at epoch 79 batch 28000/36289 with GPU utilization around `96%`.
- Observation: The latest best checkpoint, epoch 76, generates BREP-valid STEP files in the retained smoke tests.
  Evidence: `tools\evaluate_reconstruction_v13.py --source generated --constrained-decoding --write-step --validate-step` was run against `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr1e5\ar_best.pt` for seed 0 and seed 1. Reports `eval_generated5_arbest_lr1e5_seed0_epoch76_cpu\reconstruction_report.json` and `eval_generated5_arbest_lr1e5_seed1_epoch76_cpu\reconstruction_report.json` are both `VERIFIED`; together they show `attempted=10`, `grammar_valid=10`, `reconstruct_success=10`, `step_saved=10`, `brep_valid=10`, and `errors=0`. STEP files are retained under each report directory's `steps` folder.
- Observation: The first completed `lr=1e-5` epoch did not improve the best but confirms the resumed optimizer is using the requested lower learning rate.
  Evidence: `newscheme_full_v13_ar_lr1e5\ar_history.jsonl` contains epoch 77 with `train_ce=0.2995949923169265`, `val_ce=0.29588743756313873`, `best_val_ce=0.29538217296178204`, and `improved=false`. CPU loading `ar_latest.pt` showed epoch 77 with model, optimizer, and scaler states present, checkpoint `learning_rate=1e-05`, and optimizer learning rates `[1e-05]`. `ar_best.pt` remains the inherited epoch-76 checkpoint and therefore still carries its original checkpoint metadata `learning_rate=2e-05`, which is expected until a new `lr=1e-5` best is written. The analyzer reported `epochs_since_best=1`, `plateau_signal=false`, `overfit_signal=false`, and `recommendation=continue_unchanged`.
- Observation: Repeated generated samples can look identical when the same fixed seed is reused, and constrained decoding still tends to favor a small set of topology templates.
  Evidence: `tools\evaluate_reconstruction_v13.py` previously defaulted to `--seed 0`, and the evaluator seeded Python, NumPy, and PyTorch at process start. Earlier seed-0 generated runs produced the same structural pattern even across nearby checkpoints, while seed-1 and a later random seed produced different face/edge patterns. The tool now records `sampling.requested_seed`, `sampling.effective_seed`, `temperature`, `top_p`, `max_new_tokens`, and `max_samples` in `reconstruction_report.json`; using `--seed -1` generated epoch-95 report `eval_generated5_latest_arbest_lr1e5_epoch95_seed_random_cpu_20260703_0955` with effective seed `3161541872`.
- Observation: The latest epoch-95 generated reconstruction is useful but not yet strong evidence of high visual quality.
  Evidence: The epoch-95 generated report saved 5 STEP files and all 5 reconstructed, but only 4/5 passed strict BREP validity. The generated samples were all length 213 and used repeated low-complexity topology counts, mostly 4 or 6 faces with 6 or 12 edges, so the model is valid enough for smoke reconstruction but still appears template-biased.
- Observation: Epoch 100 ended the `lr=1e-5` launcher with `STAGE ar FAILED`, but the AR training loop itself had completed and saved checkpoints.
  Evidence: The log `ar_newscheme_full_v13_ar_lr1e5_20260702_152112.log` shows epoch 100 completing with `train_CE=0.3018`, `val_CE=0.2955`, and `best=0.2951`, then saving `ar_best.pt` and `ar_latest.pt` before `STAGE ar FAILED`. CPU loading showed `ar_best.pt` at epoch 95 and `ar_latest.pt` at epoch 100, and `ar_checkpoints\ar_epoch_0100.pt` exists. The failure came from `stage_ar()` returning false when `ce_final >= ce_init`; this is a poor predicate for resumed low-LR branches because a valid branch can plateau while still producing a finite best checkpoint.
- Observation: Different sampling settings increase diversity but trade off validity.
  Evidence: Three epoch-95 generated runs on 2026-07-03 used fresh effective seeds `3750870228`, `2795605260`, and `4056854188`. The retained STEP hashes were all unique. The conservative `temperature=0.8, top_p=0.90` run saved 4/5 STEP files and all 4 were BREP-valid; the middle `1.0/0.95` run saved 4/5 and all 4 were BREP-valid; the more exploratory `1.2/0.98` run saved 3/5 and all 3 were BREP-valid. Failures were either `truncated (no END)` or `reconstruct_failed`, so more aggressive sampling is visibly more diverse but less reliable.
- Observation: Continuing unchanged from the `lr=1e-5` branch is no longer the best use of GPU after epoch 100.
  Evidence: `tools\analyze_ar_training.py --history ...\newscheme_full_v13_ar_lr1e5\ar_history.jsonl --target-epoch 120` reported `latest_epoch=100`, `best_epoch=95`, `epochs_since_best=5`, `plateau_signal=true`, `overfit_signal=true`, and `recommendation=consider_stop_or_lower_lr`. The next branch therefore uses `lr=5e-6` from the epoch-95 best checkpoint rather than continuing the stale epoch-100 latest checkpoint unchanged.
- Observation: The most reliable current generated-STEP visualization setting is `temperature=0.9`, `top_p=0.92`, `max_new_tokens=320`.
  Evidence: Running `tools\evaluate_reconstruction_v13.py` against `newscheme_full_v13_ar_lr5e6\ar_best.pt` produced report `eval_generated6_lr5e6_activebest_temp09_topp92_max320_random_cpu_20260703_193036\reconstruction_report.json` with effective seed `3410736040`, `attempted=6`, `grammar_valid=6`, `reconstruct_success=6`, `step_saved=6`, `brep_valid=6`, and `errors=0`. The retained STEP files have unique hashes and live under that run's `steps` directory. A lower `temperature=0.75`, `top_p=0.88`, `max_new_tokens=300` run also saved 5 BREP-valid STEP files, but one sample was truncated without END.
- Observation: The epoch monitor needs checkpoint fallback for resumed branches before the first new validation row.
  Evidence: The `lr=5e-6` branch resumed from epoch 95, but the first monitor process showed `latest_epoch=0` because the new branch had no `ar_history.jsonl` rows until epoch 96 completes. `monitor_ar_epoch_gate.py --once` now reads `ar_latest.pt` when history is empty and reports `latest_epoch=95` with the checkpoint summary. The restarted active monitor writes to `D:\luolin\V13\local_runs\ar_training\logs\ar_lr5e6_epoch120_monitor_20260703_1942.jsonl`.
- Observation: The copied `lr=5e-6` branch checkpoint metadata still shows the source checkpoint learning rate until the first new epoch is saved.
  Evidence: CPU loading `newscheme_full_v13_ar_lr5e6\ar_best.pt` and `ar_latest.pt` during epoch 96 showed both at epoch 95 with `learning_rate=1e-05`, because the launcher copied the epoch-95 `lr=1e-5` best checkpoint into the new run. The active training log is the authoritative source for the current optimizer setting before the next checkpoint save; it shows `lr=5e-06` and `resumed ... at epoch=95 best=0.2951 lr=5e-06`.
- Observation: The first `lr=5e-6` continuation epoch did not improve the best but produced a valid latest checkpoint with the corrected learning-rate metadata.
  Evidence: `newscheme_full_v13_ar_lr5e6\ar_history.jsonl` contains epoch 96 with `train_ce=0.2989420899666821`, `val_ce=0.2955302662144129`, `best_val_ce=0.2951271435705727`, and `improved=false`. CPU loading `ar_latest.pt` showed epoch 96 and `learning_rate=5e-06`; CPU loading `ar_best.pt` showed it correctly remains epoch 95. Analyzer output with baseline epoch 95 reports `epochs_since_best=1`, no plateau/overfit signal, and `recommendation=continue_unchanged`.
- Observation: Periodic AR checkpoints live under the `ar_checkpoints` subdirectory, not beside `ar_latest.pt` and `ar_best.pt`.
  Evidence: After epoch 100 completed, `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_checkpoints\ar_epoch_0100.pt` existed and CPU-loaded with epoch 100 model, optimizer, and scaler state. Querying the run root for `ar_epoch_0100.pt` returned no file because the expected location is the checkpoint subdirectory.

## Decision Log

- Decision: Treat parsed files as no longer required for direct AR training but do not delete them during this implementation.
  Rationale: The user asked to compress by chunk if no longer needed. Compression is safe and reversible; deletion is destructive and can wait until archive verification and an explicit cleanup decision.
  Date/Author: 2026-06-28 / Codex.
- Decision: Use the canonical sequence filename `sequences_fsq_rcm.pkl` inside the V13 AR run directory.
  Rationale: Existing AR code expects this filename. Copying the selected sequence artifact to that name avoids refactoring the loader while satisfying the requirement that AR input and output live in V13.
  Date/Author: 2026-06-28 / Codex.
- Decision: Start with AR hyperparameters `d_model=256`, `layers=8`, `batch_size=8`, `epochs=120`, `learning_rate=5e-4`, AMP enabled, gradient clipping enabled, and checkpoint interval 20.
  Rationale: These match the existing project defaults and local config, are likely to fit the RTX 3060 12 GB, and can be resumed. If GPU utilization is low and memory headroom is large after the smoke test or first training window, increase batch size to 12 or 16 by restarting from the latest checkpoint.
  Date/Author: 2026-06-28 / Codex.
- Decision: Keep `batch_size=8` for the first production AR run rather than increasing immediately.
  Rationale: Runtime monitoring showed 97-99% GPU utilization and about 8.6 GB of 12 GB VRAM in use. The hardware is already saturated enough, and increasing batch size before the first checkpoint would raise OOM risk without clear upside.
  Date/Author: 2026-06-28 / Codex.
- Decision: Use length-bucketed batches for AR training.
  Rationale: Sorting validation by length and shuffling train examples in sorted buckets reduces padding waste while preserving training randomness across buckets.
  Date/Author: 2026-06-28 / Codex.
- Decision: Do not restart AR training to change hyperparameters while GPU utilization remains above 90% and first-epoch validation is improving.
  Rationale: Restarting would discard time or rely on resume only to test a riskier batch size. Current evidence shows high utilization, stable memory, decreasing train CE, and a valid first-epoch checkpoint.
  Date/Author: 2026-06-28 / Codex.
- Decision: Continue AR training past epoch 20 and reassess at epoch 40 instead of stopping immediately at the first periodic checkpoint.
  Rationale: Epoch 20 did not improve over epoch 19, but epochs 18 and 19 had just produced new best validation losses. A single non-improving epoch is not enough evidence of overfitting; the epoch-20 latest and periodic checkpoints are valid recovery points if the run later needs to be stopped or rewound.
  Date/Author: 2026-06-29 / Codex.
- Decision: Keep the current AR run unchanged through epoch 40 rather than restarting with a different batch size or learning rate.
  Rationale: Epochs 22, 24, 25, 26, 27, 28, 29, and 30 produced new best validation losses after the epoch-20 pause, while GPU utilization remained near saturation. Restarting or changing hyperparameters before the planned epoch-40 review would add risk without clear evidence of overfitting.
  Date/Author: 2026-06-29 / Codex.
- Decision: Do not mark the overall goal complete until the AR run reaches a defensible training endpoint or the user explicitly accepts the in-progress training state.
  Rationale: The operational pipeline work is verified, but the user asked for full AR training. Current evidence proves the run is healthy and efficient, not that the training objective has finished.
  Date/Author: 2026-06-29 / Codex.
- Decision: Add a reusable read-only AR history analyzer instead of relying on ad hoc manual inspection of `ar_history.jsonl`.
  Rationale: The user wants to know whether training is effective and whether to continue. A small tested analyzer gives a consistent rule for recent-best, plateau, overfit signal, train/validation gap, and ETA without touching the live training process.
  Date/Author: 2026-06-29 / Codex.
- Decision: Delete the duplicate D-side parsed archive mirror but keep the E-side official archive and source parsed directories untouched.
  Rationale: D drive free space dropped below 0.5 GiB while AR training still needed to write checkpoints. The D-side archive mirror was a verified same-size subset of the E-side official archive and had no unique files. Keeping it would risk a checkpoint write failure; deleting it restored about 132 GiB of space without removing the official archive.
  Date/Author: 2026-06-29 / Codex.
- Decision: Treat `D:\luolin\V13\ABC` as the current offline data mirror and stop using E-drive defaults for active status checks.
  Rationale: The user moved parsed archives and earlier experiment outputs into the project-local `ABC` folder and cleared E for ejection. The active AR run already uses D-only local inputs and outputs, so monitoring should follow the new D-side data root and tolerate E being absent.
  Date/Author: 2026-06-29 / Codex.
- Decision: Continue the current AR run unchanged after epoch 40 and use epoch 60 as the next formal review point.
  Rationale: Epoch 40 did not beat the epoch-36 best, but validation CE at epochs 39 and 40 stayed close to the best, the analyzer did not flag overfitting, and GPU utilization remains high. Restarting or lowering learning rate now would interrupt a stable run without enough evidence that the current schedule is exhausted.
  Date/Author: 2026-06-30 / Codex.
- Decision: Supersede the epoch-60 wait with an immediate lower-learning-rate continuation from epoch 41.
  Rationale: Epoch 41 completed with a valid latest checkpoint but became the fifth epoch after the epoch-36 best without a validation improvement. The analyzer flagged both plateau and overfit caution, while the latest checkpoint gave a safe recovery point. Continuing at `5e-4` would spend GPU time on a schedule that appears exhausted; lowering to `1e-4` is a conservative fine-tuning step that preserves model, optimizer, scaler, best checkpoint, history, and periodic checkpoint behavior.
  Date/Author: 2026-06-30 / Codex.
- Decision: Make AR resume explicitly apply the requested learning rate after loading optimizer state.
  Rationale: PyTorch optimizer checkpoints include parameter group learning rates. Without overriding them after `opt.load_state_dict`, a resumed run launched with `NS_AR_LR=1e-4` can still step with the old `5e-4`. The new test captures this failure mode, and the fix makes resumed fine-tuning behavior match the launcher and checkpoint metadata.
  Date/Author: 2026-06-30 / Codex.
- Decision: Keep the current `lr=1e-4` AR continuation unchanged through epoch 60.
  Rationale: The first completed lower-learning-rate epoch, epoch 42, improved validation CE from the old best `0.3302254088390201` to `0.30511944202840924` and restored the analyzer recommendation to `continue_unchanged`. This is strong evidence that lowering the learning rate was the right intervention. Further changes before the epoch-60 checkpoint would add noise and make the curve harder to interpret.
  Date/Author: 2026-06-30 / Codex.
- Decision: Continue the current `lr=1e-4` AR run unchanged after epoch 60 toward epoch 100.
  Rationale: Epoch 60 completed with a valid periodic checkpoint, validation CE `0.2997024929898784` remained very close to the epoch-56 best `0.29962874219155966`, and the analyzer reported no plateau or overfit signal. The next change should be evidence-driven; lowering the learning rate or stopping now would be premature because the recent 10-epoch validation delta is still negative and the model crossed below `0.30` only four epochs ago.
  Date/Author: 2026-07-01 / Codex.
- Decision: Supersede the post-epoch-60 unchanged continuation with a separate `lr=5e-5` branch from the epoch-56 best checkpoint.
  Rationale: By epoch 66, validation had failed to improve for 10 epochs while train CE continued to drift downward, and the analyzer repeatedly recommended `consider_stop_or_lower_lr`. Restarting from `ar_best.pt` in a separate run directory preserves the old branch for audit, avoids mixing histories, and tests whether a smaller step size can improve validation without spending more time on the exhausted `1e-4` schedule.
  Date/Author: 2026-07-01 / Codex.
- Decision: Keep the `lr=5e-5` branch unchanged after its first completed epoch.
  Rationale: Epoch 57 immediately beat the previous best validation CE, the branch history remains isolated, and the analyzer reports no plateau or overfit signal. The safe move is to keep the new schedule steady long enough to see whether the improvement persists before making any second adjustment.
  Date/Author: 2026-07-01 / Codex.
- Decision: Keep the `lr=5e-5` branch unchanged after epoch 58 as well.
  Rationale: Epoch 58 did not beat the epoch-57 best, but it stayed comfortably below the old `lr=1e-4` branch's plateau point and the analyzer still reports no plateau or overfit signal. One non-improving epoch after a new best is normal; changing again now would erase the evidence needed to judge whether the branch is stabilizing.
  Date/Author: 2026-07-01 / Codex.
- Decision: Stop the `lr=5e-5` branch after epoch 62 and test a new isolated `lr=2e-5` continuation from the branch's epoch-57 best checkpoint.
  Rationale: Epochs 58 through 62 failed to beat the epoch-57 validation best while train CE continued to drift downward, and the analyzer now reports both plateau and overfit signals. A separate lower-learning-rate branch preserves all previous evidence while testing whether a smaller step size can improve validation without continuing the plateaued schedule.
  Date/Author: 2026-07-01 / Codex.
- Decision: Add a V13-local FSQ-aware evaluator instead of invoking `BrepARG\generate_brep.py` directly.
  Rationale: The original generation script loads the original vector-quantized VQ-VAE, not the FSQ VQ-VAE used to build `sequences_fsq_rcm.pkl`. A focused evaluator can load `fsq_vqvae_best.pt` through `build_fsq_vqvae`, materialize a legacy-compatible 64-channel decoder embedding for reconstruction, use constrained decoding for AR samples, write manifest/report files, and retain STEP outputs under ignored `local_runs`.
  Date/Author: 2026-06-30 / Codex.
- Decision: Patch `joint_optimize` only inside the reconstruction evaluator process with a CPU `torch.cdist` implementation.
  Rationale: The active AR training already saturates the GPU and should not be disturbed. The installed `chamferdist` extension is not CUDA-enabled on this machine, while CPU reconstruction is fast enough for smoke and small-batch evaluation. Keeping the patch local avoids changing training behavior or the upstream BrepARG dependency.
  Date/Author: 2026-06-30 / Codex.
- Decision: Continue the active `lr=5e-6` branch unchanged after the epoch-100 checkpoint and review again around epoch 105 unless a new best checkpoint appears earlier.
  Rationale: Epoch 100 wrote a valid periodic checkpoint and the active process continued into epoch 101 with high GPU utilization. The analyzer reports `epochs_since_best=5`, `plateau_signal=true`, `overfit_signal=false`, and `recommendation=review_at_checkpoint`, not `consider_stop_or_lower_lr`. Because the validation loss remains very close to the epoch-95 best and there is no overfit signal, stopping or branching immediately would be premature; a short continuation gives enough evidence to distinguish noise from a true exhausted low-learning-rate schedule.
  Date/Author: 2026-07-04 / Codex.
- Decision: Treat epoch 120 of `newscheme_full_v13_ar_lr5e6` as the current defensible AR training endpoint.
  Rationale: The branch reached its explicit target epoch, wrote a loadable periodic checkpoint `ar_epoch_0120.pt`, refreshed both latest and best checkpoints at epoch 120, and improved validation CE to `0.29493329663972306`. The analyzer reports `remaining_epochs=0`, `epochs_since_best=0`, no plateau or overfit signal, and no training process remains, so continuing would require a new objective rather than monitoring the original run.
  Date/Author: 2026-07-05 / Codex.
- Decision: Keep the `temperature=0.9`, `top_p=0.92`, `max_new_tokens=320` epoch-120 reconstruction run as the current default visualization candidate, while retaining the conservative `0.75/0.88` run as a stable comparison.
  Rationale: Both settings saved 6/6 strict BREP-valid STEP files with fresh seeds and no duplicate hashes. The `0.9/0.92` setting gives a little more sampling diversity while preserving full validity; the `1.0/0.95` setting is useful for exploring diversity but already showed a grammar truncation at 1 of 6 samples.
  Date/Author: 2026-07-05 / Codex.

## Outcomes & Retrospective

This plan has reached the requested AR training and reconstruction endpoint for the current objective. The AR workflow has been implemented and verified by helper tests, syntax checks, PowerShell parser checks, AR preflight, resumable checkpoint loading, monitor gate checks, and generated reconstruction runs. Parsed archive creation completed all 100 chunks, and the offline archive plus prior VQ-VAE/sequence outputs now live under `D:\luolin\V13\ABC` because E was cleared for ejection. The final branch is `newscheme_full_v13_ar_lr5e6`, started from the epoch-95 `lr=1e-5` best checkpoint with `lr=5e-6`; it completed target epoch 120, stopped cleanly, and refreshed `ar_best.pt` with `val_CE=0.29493329663972306`. Reconstruction semantics are separated: deterministic validation reconstruction checks fixed VQ-VAE decode quality, while generated reconstruction samples from AR. True generated reconstruction from the latest epoch-120 `ar_best.pt` has been verified with fresh runtime seeds and retained STEP files. The strongest current visualization run is `eval_generated6_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_002743`, which saved 6/6 strict BREP-valid STEP files; all 23 STEP files compared across the latest three epoch-120 runs and the earlier 2026-07-03 active-best run had unique SHA256 hashes, so current generated outputs are not byte-identical repeats.

## Context and Orientation

The repository root is `D:\luolin\V13`. The completed VQ-VAE and sequence artifacts now live under `D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100`. The canonical sequence file is `D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm.pkl`. A sharded merge also exists at `D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm_sharded_merged.pkl`. The final AR branch has its own local copy at `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\sequences_fsq_rcm.pkl`, so it does not depend on E. Its final checkpoint set is `ar_latest.pt`, `ar_best.pt`, and `ar_checkpoints\ar_epoch_0120.pt`, all at epoch 120.

The AR trainer is in `breparg_improvements\train.py`. At import time it sets `OUT` from `NS_OUTBASE` and `NS_OUT`, then sets `SEQ_PKL` to `OUT\sequences_fsq_rcm.pkl` and `AR_PT` to `OUT\ar_best.pt`. The AR model class is `BrepARG\model.py::ARModel`, which wraps a Hugging Face GPT-2 language model. The AR data loader extracts `g['original']['input_ids']` for train and validation sequences and filters sequences longer than 1024 tokens.

The parsed ABC files have been archived by chunk under `D:\luolin\V13\ABC\processed\abc_parsed_full_archives\abc_XXXX_parsed.zip`. These archives are useful for regenerating sequence data or rerunning VQ-VAE/sequence experiments after extraction, but AR training does not read them.

## Plan of Work

First, add tests that describe the requested training controls without requiring a long GPU run. These tests should check pure helper behavior such as periodic checkpoint naming, resume checkpoint metadata, and launcher environment variables. They should also test that the AR preflight script accepts a small fake sequence pickle and rejects bad token ranges.

Second, add a focused helper module, `breparg_improvements\ar_training_utils.py`, for checkpoint path naming, saving/loading checkpoint dictionaries, history JSONL append behavior, and sequence summary validation. Keeping these helpers separate makes them testable without starting a GPU training run.

Third, update `breparg_improvements\train.py` so `_train_ar()` can resume from `NS_AR_RESUME_FROM`, save `ar_best.pt`, save `ar_latest.pt` each epoch, save `ar_epoch_0020.pt` and so on every `NS_AR_SAVE_EVERY` epochs, append `ar_history.jsonl`, and write a richer `stages.ar` report. The saved checkpoint dictionary must include `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `epoch`, `best_val_ce`, `vocab_size`, `d_model`, `layers`, `batch_size`, `learning_rate`, and `config`.

Fourth, create `tools\preflight_ar_training.py`. This script should load a sequence pickle, summarize train/val/test counts, confirm token IDs are in range, instantiate `ARModel`, run a forward loss on a small batch, run a backward pass, and write a JSON report. It must support `--max-samples` so it can run quickly on the full sequence file.

Fifth, create `tools\run_ar_v13_epoch100.ps1`. This launcher should set `NS_OUTBASE=D:\luolin\V13\local_runs\ar_training\train_outputs`, `NS_OUT=newscheme_full_v13_ar`, `NS_AR_EPOCHS=120`, `NS_AR_BS=8`, `NS_AR_DMODEL=256`, `NS_AR_LAYERS=8`, `NS_AR_SAVE_EVERY=20`, and CUDA environment variables. It should tee console logs to `D:\luolin\V13\local_runs\ar_training\logs`.

Sixth, copy the canonical sequence pickle into `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\sequences_fsq_rcm.pkl`. This copy is the AR training input. The source copy currently lives in `D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100`.

Seventh, create `tools\archive_parsed_chunks.py` and a Windows launcher if needed. It should compress each `abc_XXXX` parsed directory into `D:\luolin\V13\ABC\processed\abc_parsed_full_archives\abc_XXXX_parsed.zip` or `.7z` depending on available tools, write `_manifest.jsonl`, and verify that each archive exists and has nonzero size. It should not delete the original parsed directory unless a future explicit flag such as `--delete-after-verify` is supplied.

Eighth, run preflight and then start AR training in a hidden PowerShell process. Monitor GPU utilization, memory, log growth, `ar_latest.pt`, `ar_best.pt`, and epoch checkpoint files.

## Concrete Steps

Run commands from `D:\luolin\V13` with `C:\Users\YU\.conda\envs\brepgen_env\python.exe`.

The first verification command after adding tests is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q

Expected result after implementation is `passed` with no failures.

The syntax check is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile breparg_improvements\ar_training_utils.py tools\preflight_ar_training.py tools\archive_parsed_chunks.py breparg_improvements\train.py

The AR preflight command is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\preflight_ar_training.py --sequence D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\sequences_fsq_rcm.pkl --output D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_preflight_report.json --max-samples 128 --batch-size 4 --d-model 256 --layers 8

Expected result is a JSON report with `status` equal to `VERIFIED`, nonzero train and validation counts, `out_of_vocab=0`, and a finite `smoke_loss`.

The AR training launch command is:

    Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\luolin\V13\tools\run_ar_v13_epoch100.ps1' -WindowStyle Hidden

The epoch-40 gate check is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\monitor_ar_epoch_gate.py --once --target-epoch 40

It returns a nonzero exit code while waiting and exit code `0` once the target periodic checkpoint exists and includes model and optimizer state. For unattended monitoring, run it without `--once` and pass `--status-log` to append JSONL status rows.

The next formal gate after epoch 40 is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\monitor_ar_epoch_gate.py --once --target-epoch 60

The read-only AR history analysis command is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\analyze_ar_training.py --target-epoch 100 --recent-window 10 --plateau-patience 5 --output D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_analysis_latest.json

It reads `ar_history.jsonl`, prints a JSON report, and optionally writes the same report under the ignored `local_runs` output tree. It must not start, stop, or modify the active training process.

The FSQ-aware reconstruction smoke command for existing validation sequences is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\evaluate_reconstruction_v13.py --source validation --max-samples 5 --order shortest --device cpu --write-step --validate-step --run-name eval_validation_short5_arbest_cpu --seed 0

Expected result is a JSON report with `status` equal to `VERIFIED`, `summary.step_saved` greater than zero, and retained STEP files under `D:\luolin\V13\local_runs\reconstruction_eval\eval_validation_short5_arbest_cpu\steps`.

The FSQ-aware reconstruction smoke command for the current AR best checkpoint is:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\evaluate_reconstruction_v13.py --source generated --max-samples 1 --device cpu --constrained-decoding --max-new-tokens 160 --write-step --validate-step --run-name smoke_generated_step_arbest_constrained --seed 0

Expected result is a JSON report with `status` equal to `VERIFIED`, `summary.grammar_valid=1`, `summary.step_saved=1`, and a retained STEP file under `D:\luolin\V13\local_runs\reconstruction_eval\smoke_generated_step_arbest_constrained\steps`.

## Validation and Acceptance

The work is accepted when the V13 local AR run directory contains the sequence input, preflight report, AR logs, and checkpoint files; the AR preflight report is `VERIFIED`; tests and syntax checks pass; parsed chunk archives are written with manifest records; and AR training is completed at a defensible endpoint with GPU utilization and process status monitored. The training loop must write `ar_latest.pt`, `ar_best.pt`, and `ar_epoch_0020.pt` style periodic checkpoints every 20 epochs, and it must be possible to resume by setting `NS_AR_RESUME_FROM` to `ar_latest.pt`. Reconstruction acceptance requires `tools\evaluate_reconstruction_v13.py` to load the FSQ VQ-VAE checkpoint rather than the original VQ loader, write manifest/report files, retain STEP files under `local_runs\reconstruction_eval`, and demonstrate generated `ar_best.pt` samples that are grammar-valid, reconstructed, saved as STEP, and BREP-valid.

As of 2026-07-05, the current objective meets acceptance. The final branch `newscheme_full_v13_ar_lr5e6` reached epoch 120 and wrote loadable latest, best, and periodic checkpoints. The epoch-120 analyzer status is `VERIFIED`, the monitor reports `ready=true`, no matching training process remains, and the GPU is idle. Generated reconstruction from the refreshed epoch-120 best checkpoint was rerun with three fresh seeds; the two stable settings saved 6/6 BREP-valid STEP files each, and all retained STEP files in the latest comparison had unique hashes.

## Idempotence and Recovery

Copying the sequence file is idempotent if the destination size matches the source size. The archive script should skip existing valid archives when `--resume` is used. AR training can be restarted from `ar_latest.pt`; periodic checkpoints provide additional recovery points. The project-local offline data mirror under `D:\luolin\V13\ABC` should be preserved unless the user explicitly approves cleanup.

## Artifacts and Notes

Latest AR endpoint evidence:

    final run directory: D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6
    final log: D:\luolin\V13\local_runs\ar_training\logs\ar_newscheme_full_v13_ar_lr5e6_20260703_191835.log
    history: D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_history.jsonl
    best checkpoint: D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_best.pt
    latest checkpoint: D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_latest.pt
    periodic checkpoint: D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\ar_checkpoints\ar_epoch_0120.pt
    final epoch: 120
    final best validation CE: 0.29493329663972306
    monitor status: ready=true, latest_epoch=120

Latest generated reconstruction evidence:

    preferred report: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_002743\reconstruction_report.json
    preferred STEP directory: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_002743\steps
    preferred sampling: requested_seed=-1, effective_seed=3899885580, temperature=0.9, top_p=0.92, max_new_tokens=320
    preferred summary: attempted=6, grammar_valid=6, reconstruct_success=6, step_saved=6, brep_valid=6, errors=0
    conservative report: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp075_topp88_max320_random_cpu_20260705_002856\reconstruction_report.json
    conservative STEP directory: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp075_topp88_max320_random_cpu_20260705_002856\steps
    conservative sampling: requested_seed=-1, effective_seed=4275042948, temperature=0.75, top_p=0.88, max_new_tokens=320
    conservative summary: attempted=6, grammar_valid=6, reconstruct_success=6, step_saved=6, brep_valid=6, errors=0
    diverse report: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp10_topp95_max340_random_cpu_20260705_002948\reconstruction_report.json
    diverse STEP directory: D:\luolin\V13\local_runs\reconstruction_eval\eval_generated6_lr5e6_epoch120_best_temp10_topp95_max340_random_cpu_20260705_002948\steps
    diverse sampling: requested_seed=-1, effective_seed=4105039696, temperature=1.0, top_p=0.95, max_new_tokens=340
    diverse summary: attempted=6, grammar_valid=5, reconstruct_success=5, step_saved=5, brep_valid=5, errors=1
    STEP hash check: 23 retained STEP files across the three epoch-120 runs and the earlier 2026-07-03 0.9/0.92 run produced 23 unique SHA256 hashes.

Current validated sequence evidence:

    canonical sequence: D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm.pkl
    canonical length: 1414689276
    sharded merged sequence: D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm_sharded_merged.pkl
    sequence count: 425120
    train count: 382720
    validation count: 21124
    test count: 21276
    vocab_size: 10294
    max_token: 10292
    out_of_vocab: 0
    se_tokens_per_element: 4

Revision note: Created this plan after confirming the completed sequence stage and before changing AR training code.

## Interfaces and Dependencies

In `breparg_improvements\ar_training_utils.py`, define pure helpers for AR training metadata:

    ar_checkpoint_paths(out_dir)
    periodic_checkpoint_path(out_dir, epoch)
    summarize_ar_sequences(package, max_seq_len=1024)
    validate_ar_sequence_package(package)
    append_jsonl(path, row)
    save_ar_checkpoint(path, payload)
    load_ar_checkpoint(path, map_location)

In `tools\preflight_ar_training.py`, expose a CLI with `--sequence`, `--output`, `--max-samples`, `--batch-size`, `--d-model`, and `--layers`.

In `tools\archive_parsed_chunks.py`, expose a CLI with `--parsed-root`, `--archive-root`, `--manifest`, `--chunks`, `--resume`, `--compression`, and an explicit opt-in `--delete-after-verify` for future cleanup.

In `tools\analyze_ar_training.py`, expose a read-only CLI with `--history`, `--target-epoch`, `--recent-window`, `--plateau-patience`, `--min-delta`, and optional `--output`. It should return a JSON report containing latest epoch, best epoch, train and validation CE, train-validation gap, recent epoch time, ETA to the requested target, plateau signal, overfit signal, recommendation, and reason.

In `tools\evaluate_reconstruction_v13.py`, expose a CLI with `--sequence`, `--vqvae-checkpoint`, `--ar-checkpoint`, `--output-root`, `--run-name`, `--source`, `--max-samples`, `--order`, `--device`, `--write-step`, `--write-stl`, `--validate-step`, `--constrained-decoding`, `--max-new-tokens`, `--temperature`, and `--top-p`. It must load FSQ checkpoints through `breparg_improvements.train.build_fsq_vqvae`, materialize a 64-channel decoder embedding for the legacy BrepARG reconstruction helper, validate token grammar before OCC reconstruction, optionally generate AR samples with `TopologyConstrainedLogitsProcessor`, patch `joint_optimize` locally to a CPU `torch.cdist` implementation, and write `reconstruction_manifest.jsonl`, `reconstruction_report.json`, and retained `.step` files under `local_runs\reconstruction_eval`.

Revision note 2026-06-29 19:20 +08:00: Added the AR history analyzer to make epoch-40 and later training decisions repeatable. The live epoch-30 analysis recommends continuing unchanged, so the current GPU-saturated run remains untouched.

Revision note 2026-06-29 22:42 +08:00: Updated the plan and status checker after the user moved the offline ABC archives and prior experiment outputs from E into `D:\luolin\V13\ABC` and cleared E for ejection. Active AR training remains D-local under `local_runs`.

Revision note 2026-06-30 04:16 +08:00: Epoch 40 gate passed. The run wrote a valid periodic checkpoint and continued into epoch 41. The validation curve has a mild plateau after epoch 36 but has not triggered the overfit signal, so the run continues unchanged toward epoch 60.

Revision note 2026-06-30 04:26 +08:00: Continued the active goal audit. The full objective is not complete because AR training has not reached a defensible endpoint. Live training remains healthy in epoch 41, and a read-only epoch-60 gate monitor now records progress under the V13 local logs directory.

Revision note 2026-06-30 05:20 +08:00: Updated the plan after epoch 41 triggered the analyzer's plateau/overfit caution. AR resume now forces optimizer param groups to the requested learning rate, the launcher accepts `-LearningRate` and `-TargetEpochs`, tests pass with `37 passed`, and the active run is continuing from epoch 41 at `lr=1e-4` toward epoch 100.

Revision note 2026-06-30 06:20 +08:00: Recorded the first completed lower-learning-rate epoch. Epoch 42 improved validation CE to `0.30511944202840924`, refreshed both latest and best checkpoints, and restored the analyzer recommendation to `continue_unchanged`; the current plan is to keep `lr=1e-4` through the epoch-60 checkpoint review.

Revision note 2026-06-30 07:20 +08:00: Recorded epoch 43 evidence. The lower-learning-rate continuation again improved validation CE, this time to `0.30437195893850855`, and both resumable checkpoints are loadable at epoch 43 with optimizer learning rate `0.0001`; continue unchanged toward the epoch-60 periodic checkpoint.

Revision note 2026-06-30 08:20 +08:00: Recorded epoch 44 evidence. Validation CE improved again to `0.3035273262473785`, latest and best checkpoints are loadable at epoch 44, and the analyzer still recommends continuing unchanged at `lr=1e-4` toward epoch 60.

Revision note 2026-06-30 09:20 +08:00: Recorded epoch 45 evidence. Validation CE improved again to `0.3026073543786854`, latest and best checkpoints are loadable at epoch 45 with optimizer learning rate `0.0001`, and the analyzer still recommends continuing unchanged toward epoch 60.

Revision note 2026-06-30 10:12 +08:00: Recorded epoch 46 evidence. Validation CE rebounded slightly to `0.303381397759082`, so `ar_best.pt` correctly remains epoch 45 while `ar_latest.pt` advances to epoch 46. The analyzer reports no plateau or overfit signal and still recommends continuing unchanged toward epoch 60.

Revision note 2026-06-30 12:12 +08:00: Recorded epoch 48 evidence. Validation CE improved again to `0.30222854646215414`; both latest and best checkpoints are loadable at epoch 48 with optimizer learning rate `0.0001`, and the analyzer still recommends continuing unchanged toward epoch 60.

Revision note 2026-06-30 13:09 +08:00: Recorded epoch 49 evidence. Validation CE improved again to `0.30193200306664975`; both latest and best checkpoints are loadable at epoch 49 with optimizer learning rate `0.0001`, and the analyzer still recommends continuing unchanged toward epoch 60.

Revision note 2026-06-30 14:09 +08:00: Recorded epoch 50 evidence. Validation CE improved again to `0.3014168582354995`; both latest and best checkpoints are loadable at epoch 50 with optimizer learning rate `0.0001`, and the analyzer still recommends continuing unchanged toward epoch 60.

Revision note 2026-06-30 16:25 +08:00: Added the FSQ-aware reconstruction evaluator and recorded smoke evidence. Existing validation sequences can be reconstructed into retained STEP files, and the current `ar_best.pt` can generate a constrained, grammar-valid sequence that reconstructs to a BREP-valid STEP file. AR training remains active and unchanged while waiting for the epoch-60 gate.

Revision note 2026-06-30 17:05 +08:00: Recorded epoch 53 evidence. Validation CE improved again to `0.30052309570894425`; both latest and best checkpoints are loadable at epoch 53 with optimizer learning rate `0.0001`, the analyzer recommends `continue_unchanged`, and live GPU utilization remains near the requested target while waiting for the epoch-60 periodic checkpoint.

Revision note 2026-06-30 20:05 +08:00: Recorded epoch 56 evidence. Validation CE improved again to `0.29962874219155966`, crossing below `0.30`; both latest and best checkpoints are loadable at epoch 56 with optimizer learning rate `0.0001`, and the analyzer recommends continuing unchanged toward the epoch-60 checkpoint.

Revision note 2026-07-01 00:08 +08:00: Epoch 60 gate passed. The run wrote a loadable `ar_epoch_0060.pt`, all latest/best/periodic checkpoints CPU-load, the analyzer reports no plateau or overfit signal, and the decision is to continue the current `lr=1e-4` run unchanged toward epoch 100.

Revision note 2026-07-01 06:50 +08:00: The `lr=1e-4` continuation was stopped after epoch 66 because epochs 61 through 66 repeatedly triggered the analyzer's plateau/overfit caution. Added parameterized launcher support for isolated AR continuation branches, updated the status checker to find `ar_*.log` files, verified helper tests with `44 passed`, and launched `newscheme_full_v13_ar_lr5e5` from the epoch-56 best checkpoint at `lr=5e-5`.

Revision note 2026-07-01 07:36 +08:00: The new `lr=5e-5` branch completed epoch 57, improved validation CE to `0.29771688658363205`, refreshed both latest and best checkpoints, and the analyzer recommends continuing unchanged for now.

Revision note 2026-07-01 08:35 +08:00: The new `lr=5e-5` branch completed epoch 58. Validation CE rebounded slightly to `0.2980387463729057`, but the branch best remains epoch 57 and the analyzer still recommends continuing unchanged.

Revision note 2026-07-01 12:37 +08:00: The `lr=5e-5` branch completed epoch 62 without improving over epoch 57. The analyzer now reports `epochs_since_best=5`, plateau and overfit signals, and `recommendation=consider_stop_or_lower_lr`; the plan is to stop this branch and launch an isolated `lr=2e-5` continuation from the epoch-57 best checkpoint.

Revision note 2026-07-01 12:50 +08:00: Stopped the plateaued `lr=5e-5` branch and launched `newscheme_full_v13_ar_lr2e5` from the epoch-57 best checkpoint at `lr=2e-5`. The new branch resumed from epoch 57, reached epoch 58 batch `2000/36289`, and GPU utilization returned to about `89-92%`.

Revision note 2026-07-01 14:00 +08:00: The `lr=2e-5` branch completed epoch 58 with a new best validation CE of `0.2973588657237942`. CPU loading latest/best checkpoints verified model, optimizer, and scaler states with optimizer learning rate `[2e-05]`. The analyzer recommends continuing unchanged. Reconstruction was rerun against this new best checkpoint on CPU; the report is `VERIFIED` with 5 STEP files saved, 5/5 grammar-valid and reconstructed, 4/5 strict BREP-valid, and 0 errors.

Revision note 2026-07-01 14:57 +08:00: The `lr=2e-5` branch completed epoch 59 with a new best validation CE of `0.29660862653964476`. The previous validation reconstruction runs were deterministic because they decoded fixed validation token sequences; they were not AR generation tests. True generated reconstruction from epoch-59 `ar_best.pt` was run with constrained decoding for seeds 0 and 1. Both generated reports are `VERIFIED`; together they saved 10 STEP files, all 10 are grammar-valid, reconstructed, and strict BREP-valid.

Revision note 2026-07-01 15:55 +08:00: The `lr=2e-5` branch completed epoch 60, wrote loadable periodic checkpoint `ar_epoch_0060.pt`, and continued into epoch 61. Epoch 60 did not improve over the epoch-59 best, but the analyzer reports only `epochs_since_best=1` and recommends continuing unchanged.

Revision note 2026-07-01 16:49 +08:00: The `lr=2e-5` branch completed epoch 61 with `val_CE=0.2966308543054466`, narrowly above the epoch-59 best. Latest and best checkpoints CPU-load, the analyzer reports `epochs_since_best=2`, and the run should continue unchanged.

Revision note 2026-07-02 15:24 +08:00: The `lr=2e-5` branch reached latest completed epoch 84 but had not improved since epoch 76. The analyzer reported plateau/overfit and recommended stop or lower learning rate. Stopped that branch and launched `newscheme_full_v13_ar_lr1e5` from the epoch-76 best checkpoint with `lr=1e-5`. Generated reconstruction was rerun from the latest best checkpoint with seeds 0 and 1; both reports are `VERIFIED`, saving 10 STEP files total, all grammar-valid, reconstructed, and strict BREP-valid.

Revision note 2026-07-02 16:40 +08:00: The `lr=1e-5` branch completed epoch 77 with `val_CE=0.29588743756313873`, not beating the inherited epoch-76 best. The latest checkpoint CPU-loads with optimizer learning rate `[1e-05]`, and the analyzer recommends continuing unchanged.

Revision note 2026-07-02 18:04 +08:00: The `lr=1e-5` branch completed epoch 78 with `val_CE=0.29567606687322817`, still short of the epoch-76 best. The analyzer continues to recommend unchanged training, and the live process is already into epoch 79 batch 28000 with GPU utilization around `96%`.

Revision note 2026-07-03 09:55 +08:00: The active `lr=1e-5` branch reached epoch 95 and refreshed `ar_best.pt` with `val_CE=0.2951271435705727`. The analyzer reports `epochs_since_best=0`, no plateau or overfit signal, and `recommendation=continue_unchanged`; the epoch-120 monitor remains active. Generated reconstruction was rerun from the epoch-95 best checkpoint with `--seed -1`, producing report `eval_generated5_latest_arbest_lr1e5_epoch95_seed_random_cpu_20260703_0955`, retaining 5 STEP files, and passing strict BREP validity for 4/5. The evaluator now records sampling settings and the effective runtime seed in reports so fixed-seed reproducibility is distinguishable from fresh random sampling.

Revision note 2026-07-03 19:22 +08:00: The `lr=1e-5` branch completed epoch 100, wrote `ar_latest.pt`, `ar_best.pt`, and `ar_epoch_0100.pt`, then was marked failed by the old resumed-branch train-CE predicate. The predicate has been fixed and helper tests pass with `48 passed`. Analyzer evidence says the branch plateaued after epoch 95, so a new isolated `newscheme_full_v13_ar_lr5e6` branch was started from epoch-95 `ar_best.pt` with `lr=5e-6` and target epoch 120; it resumed at epoch 95 and reached epoch 96 batch `2000/36289` with GPU utilization about `97%`. Three fresh-seed generated reconstruction runs from epoch-95 best retained 11 unique STEP files across temp/top-p settings `0.8/0.90`, `1.0/0.95`, and `1.2/0.98`; higher sampling diversity reduced the valid reconstruction rate.

Revision note 2026-07-03 19:34 +08:00: While the `lr=5e-6` branch was training epoch 96, generated reconstruction was rerun from that active branch's `ar_best.pt`. The best current visualization setting is `temperature=0.9`, `top_p=0.92`, `max_new_tokens=320`, with report `eval_generated6_lr5e6_activebest_temp09_topp92_max320_random_cpu_20260703_193036`, effective seed `3410736040`, and 6 retained STEP files that are all grammar-valid, reconstructed, and strict BREP-valid. The active branch reached epoch 96 batch `8000/36289`; no epoch 96 validation row existed yet.

Revision note 2026-07-03 19:40 +08:00: The active `lr=5e-6` branch reached epoch 96 batch `12000/36289` with GPU utilization around `96%`. The first monitor process was restarted after fixing resumed-branch baseline reporting; the new status log is `ar_lr5e6_epoch120_monitor_20260703_1942.jsonl`, and its first row correctly reports `latest_epoch=95` from `ar_latest.pt` before any new history rows exist. The stale `lr=1e-5` monitor was stopped to avoid confusing old-branch status with the active branch.

Revision note 2026-07-03 20:24 +08:00: Epoch 96 completed on the active `lr=5e-6` branch with `val_CE=0.2955302662144129`, so it did not improve over the epoch-95 best `0.2951271435705727`. `ar_latest.pt` now records epoch 96 and `learning_rate=5e-6`; `ar_best.pt` correctly remains epoch 95. The analyzer reports `recommendation=continue_unchanged`, and the training process has already entered epoch 97. No new reconstruction was run because the best checkpoint did not change; the current best visualization run remains `eval_generated6_lr5e6_activebest_temp09_topp92_max320_random_cpu_20260703_193036`.

Revision note 2026-07-03 20:27 +08:00: The active branch is still healthy in epoch 97, with the log reaching batch `6000/36289` and GPU utilization around `95%`. The active monitor `ar_lr5e6_epoch120_monitor_20260703_1942.jsonl` now reports `latest_epoch=96` and `history_rows=1`. Since `ar_best.pt` did not change after epoch 96, no additional generated reconstruction was run.

Revision note 2026-07-03 20:33 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 97 reached batch `10000/36289` with GPU utilization around `97%`; no epoch-97 validation row exists yet, `ar_latest.pt` remains epoch 96, and `ar_best.pt` remains the epoch-95 checkpoint. Reconstruction is intentionally not rerun until a later epoch refreshes `ar_best.pt` or the sampling setup itself needs another comparison.

Revision note 2026-07-03 20:36 +08:00: The active `lr=5e-6` branch continued normally through epoch 97 batch `12000/36289`, with GPU utilization around `99%` and memory around `8.6 GB`. No epoch-97 validation row exists yet, so there is still no newer best checkpoint to evaluate.

Revision note 2026-07-03 20:40 +08:00: The active `lr=5e-6` branch continued normally through epoch 97 batch `14000/36289`, with GPU utilization at `100%` and memory around `8.6 GB`. The monitor still reports latest completed epoch 96 because validation for epoch 97 has not run yet.

Revision note 2026-07-03 20:46 +08:00: The active `lr=5e-6` branch continued normally through epoch 97 batch `18000/36289`, with GPU utilization around `90%` and memory around `8.6 GB`. There is still no epoch-97 history row and no newer best checkpoint to evaluate.

Revision note 2026-07-03 20:52 +08:00: The active `lr=5e-6` branch continued normally through epoch 97 batch `22000/36289`, with running train CE down to about `0.2998`, GPU utilization around `96%`, and monitor status still at latest completed epoch 96. The epoch has not reached validation/checkpoint save yet.

Revision note 2026-07-03 20:58 +08:00: The active `lr=5e-6` branch continued normally through epoch 97 batch `26000/36289`, with running train CE around `0.2999`, GPU utilization around `97%`, and no new history row or checkpoint timestamp change yet.

Revision note 2026-07-03 21:08 +08:00: The active `lr=5e-6` branch continued normally through epoch 97 batch `32000/36289`, with running train CE around `0.3002`, GPU utilization around `100%`, and no new history row or checkpoint timestamp change yet.

Revision note 2026-07-03 21:20 +08:00: Epoch 97 completed on the active `lr=5e-6` branch with `val_CE=0.295356298067103`, which improved over epoch 96 but did not beat the epoch-95 best `0.2951271435705727`. `ar_latest.pt` now CPU-loads at epoch 97 with model, optimizer, and scaler state, while `ar_best.pt` remains the epoch-95 checkpoint. The analyzer reports `epochs_since_best=2`, no plateau or overfit signal, and `recommendation=continue_unchanged`; the training process has already entered epoch 98. No reconstruction was run because the best checkpoint did not change.

Revision note 2026-07-03 21:29 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 98 reached batch `8000/36289` with running train CE around `0.3030`, GPU utilization around `96%`, and no new history row or checkpoint timestamp change yet. The analyzer still recommends `continue_unchanged` after epoch 97.

Revision note 2026-07-03 21:42 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 98 reached batch `16000/36289` with running train CE around `0.3013`, GPU utilization around `96%`, and both the training Python process and epoch-120 monitor still active. There is still no epoch-98 history row or checkpoint timestamp change yet.

Revision note 2026-07-03 21:57 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 98 reached batch `26000/36289` with running train CE around `0.3003`, GPU utilization around `92%`, and monitor status still at latest completed epoch 97. There is still no epoch-98 history row or checkpoint timestamp change yet.

Revision note 2026-07-03 22:18 +08:00: Epoch 98 completed on the active `lr=5e-6` branch with `val_CE=0.29545644794096076`, which did not beat the epoch-95 best `0.2951271435705727`. `ar_latest.pt` now CPU-loads at epoch 98 with model, optimizer, and scaler state, while `ar_best.pt` remains the epoch-95 checkpoint. The analyzer reports `epochs_since_best=3`, no plateau or overfit signal, and `recommendation=continue_unchanged`; the training process has already entered epoch 99. No reconstruction was run because the best checkpoint did not change.

Revision note 2026-07-03 22:27 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 99 reached batch `8000/36289` with running train CE around `0.2994`, GPU utilization around `100%`, and no new history row or checkpoint timestamp change yet.

Revision note 2026-07-03 22:40 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 99 reached batch `16000/36289` with running train CE around `0.3003`, GPU utilization around `97%`, and monitor status still at latest completed epoch 98. There is still no epoch-99 history row or checkpoint timestamp change yet.

Revision note 2026-07-03 22:56 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 99 reached batch `26000/36289` with running train CE around `0.3000`, GPU utilization around `99%`, and monitor status still at latest completed epoch 98. There is still no epoch-99 history row or checkpoint timestamp change yet.

Revision note 2026-07-03 23:16 +08:00: Epoch 99 completed on the active `lr=5e-6` branch with `val_CE=0.2953488017165304`, which did not beat the epoch-95 best `0.2951271435705727`. `ar_latest.pt` now CPU-loads at epoch 99 with model, optimizer, and scaler state, while `ar_best.pt` remains the epoch-95 checkpoint. The analyzer reports `epochs_since_best=4`, no plateau or overfit signal, and `recommendation=continue_unchanged`; the training process has already entered epoch 100. `ar_epoch_0100.pt` is not expected until epoch 100 completes, and it does not exist yet. No reconstruction was run because the best checkpoint did not change.

Revision note 2026-07-03 23:35 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 100 reached batch `14000/36289` with running train CE around `0.3003`, GPU utilization around `99%`, and monitor status still at latest completed epoch 99. `ar_epoch_0100.pt` has not been written yet because epoch 100 has not completed.

Revision note 2026-07-03 23:57 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 100 reached batch `28000/36289` with running train CE around `0.2996`, GPU utilization around `94%`, and monitor status still at latest completed epoch 99. `ar_epoch_0100.pt` has not been written yet because epoch 100 has not completed.

Revision note 2026-07-04 00:17 +08:00: Epoch 100 completed on the active `lr=5e-6` branch with `val_CE=0.2954869701947432`, which did not beat the epoch-95 best `0.2951271435705727`. `ar_latest.pt` and `ar_checkpoints\ar_epoch_0100.pt` both CPU-load at epoch 100 with model, optimizer, and scaler state, while `ar_best.pt` remains the epoch-95 checkpoint. The analyzer reports `epochs_since_best=5`, `plateau_signal=true`, `overfit_signal=false`, and `recommendation=review_at_checkpoint`; the decision is to continue unchanged and review again around epoch 105 unless a new best appears earlier. No reconstruction was run because the best checkpoint did not change.

Revision note 2026-07-04 00:37 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 101 reached batch `16000/36289` with running train CE around `0.3018`, GPU utilization around `96%`, and monitor status still at latest completed epoch 100. There is still no epoch-101 history row or checkpoint timestamp change yet.

Revision note 2026-07-04 00:59 +08:00: Continued monitoring the active `lr=5e-6` branch. Epoch 101 reached batch `30000/36289` with running train CE around `0.2998`, GPU utilization around `95%`, and monitor status still at latest completed epoch 100. There is still no epoch-101 history row or checkpoint timestamp change yet.

Revision note 2026-07-05 00:31 +08:00: The `lr=5e-6` branch reached target epoch 120 and stopped cleanly. Epoch 120 refreshed `ar_best.pt` with `val_CE=0.29493329663972306`; latest, best, and `ar_epoch_0120.pt` all CPU-load with model, optimizer, and scaler state. Generated reconstruction was rerun from the refreshed epoch-120 best checkpoint with three fresh-seed sampling settings. The preferred `temperature=0.9`, `top_p=0.92`, `max_new_tokens=320` run saved 6/6 BREP-valid STEP files under `eval_generated6_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_002743`; a conservative `0.75/0.88` run also saved 6/6, while a more diverse `1.0/0.95` run saved 5/6 and had one `truncated (no END)` grammar failure. STEP hash comparison found 23 unique hashes across 23 retained STEP files, confirming current generated outputs are not byte-identical repeats.
