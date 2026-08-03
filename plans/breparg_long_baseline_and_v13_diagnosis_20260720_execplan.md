# BrepARG Long Baseline and V13 Diagnosis

This ExecPlan is a living document. It follows `.agent/PLANS.md` and must keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as work proceeds.

## Purpose / Big Picture

The user needs a stronger same-data BrepARG baseline and a careful diagnosis of why the current V13 method produces many simple or invalid solids. After this plan is executed, the workspace will contain a long BrepARG baseline trained with the same 10k/1k/1k split, sparse checkpoint saving that does not fill D:, generated STEP/PNG outputs, and a written comparison against the V13 FSQ plus RCM/AR method. The result can be inspected by reading the quality summaries under `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720` and the curated backup under `E:\V13_rootcause_recovery_20260717_curated`.

## Progress

- [x] (2026-07-20 19:00 Asia/Shanghai) Verified the prior same-data BrepARG fallback: VQ-VAE configured for 160 epochs but kept best epoch 70 after a D: checkpoint save failure around epoch 73; AR completed 80 epochs with best epoch 77 and best validation CE about 0.871925.
- [x] (2026-07-20 19:05 Asia/Shanghai) Safely copied key prior results to `E:\V13_rootcause_recovery_20260717_curated`, including generated outputs, quality summaries, VQ-VAE best, AR best, epoch 80, sequence pickle, and AR comparison outputs.
- [x] (2026-07-20 19:10 Asia/Shanghai) Deleted verified redundant D: AR epoch checkpoints from the prior same-data run, freeing about 7.845 GiB.
- [x] (2026-07-20 19:30 Asia/Shanghai) Deleted verified copied `D:\V13_rootcause_recovery_20260717\ar_train_outputs`, freeing about 3.015 GiB.
- [x] (2026-07-20 19:35 Asia/Shanghai) Updated BrepARG training code so best checkpoints update stable best aliases without forcing extra `epoch_*.pt` files.
- [x] (2026-07-20 19:35 Asia/Shanghai) Added `--target_val_loss` to BrepARG VQ-VAE and AR training arguments.
- [x] (2026-07-20 19:40 Asia/Shanghai) Added `tools/run_breparg_same_data_long_vq400_ar300.ps1`, a resumable long-baseline pipeline.
- [x] (2026-07-20 19:40 Asia/Shanghai) Started the long BrepARG pipeline. Pipeline PID is 34900 and the active VQ-VAE child process PID is 27412. The main logs are under `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\pipeline_logs`.
- [x] (2026-07-23 12:59 Asia/Shanghai) Completed the long VQ-VAE stage through epoch 400 without a model or CUDA error. Best validation loss was about `0.000211` at epoch 269, compared with about `0.000240` for the previous same-data VQ-VAE best; epoch 400 validation was about `0.000262`, so downstream work correctly uses the best alias rather than the final epoch.
- [x] (2026-07-23 13:16 Asia/Shanghai) Rebuilt `breparg_same_data_sequences.pkl` successfully from the long-run VQ-VAE best checkpoint and started a fresh same-data AR run.
- [x] (2026-07-26 14:35 Asia/Shanghai) Completed BrepARG AR through epoch 300. The original process reached a new best validation CE `0.819257915` at epoch 127 before the computer rebooted during epoch 128. Training then resumed from the SHA-256-verified epoch-127 best checkpoint under `local_runs\breparg_long_ar_resume_best_20260724\ar_epoch127_best_to300`, restoring model, optimizer, AMP scaler, iteration state, best loss, and learning rate. The resumed run completed all 300 epochs without OOM, NaN, nonfinite values, traceback, or runtime error. Its best validation CE was `0.765318`, and the epoch-300 validation CE was `0.767252`. The finite best checkpoint is `same_data_abc\abc_ar_vqvae_best_model.pt`; the final periodic checkpoint is `same_data_abc\epoch_300.pt`.
- [x] (2026-07-28 14:00 Asia/Shanghai) Generated and audited 100 long-run original BrepARG baseline outputs under `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\generated_3060_long_breparg_resume_best_20260726`. Timeout-bounded recovery batches reached `100/100` STEP/STL pairs, and validation rendered `100/100` PNG previews with no validation timeout. All `100` STEP files were readable and had closed-solid structure; `86/100` passed strict BRep validity. Only `13/100` met the `faces >= 12 OR edges >= 20` complexity rule, and only `6/100` passed the full strict quality gate. Median topology remained `6` faces and `12` edges; `43/100` outputs had exactly 6 faces. Visual inspection confirmed that low-face outputs are distorted primitive-like shells and even high-face outputs are often irregular assemblies rather than convincing complex CAD. The long run improves the prior 80-epoch baseline (`5/92` complex, `0/92` strict accepted) but does not solve simple-topology collapse.
- [ ] Compare the long BrepARG baseline against V13 FSQ/RCM/AR and update this plan with final evidence.

## Surprises & Discoveries

- Observation: E: has large capacity but is exFAT and produced an I/O device error during one checkpoint copy.
  Evidence: Copying `abc_se_vqvae_best.pt` with `Copy-Item` failed once; retrying with robocopy succeeded and size matched. Therefore D: originals should not be deleted unless the E: copy has been verified.
- Observation: The previous same-data BrepARG AR run filled D: mainly through dense epoch checkpointing.
  Evidence: `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\ar_3060_safe_len1536_bs4_20260717_d\same_data_abc` contained 75 redundant `epoch_*.pt` files at about 107 MiB each, which were safely deleted after the best model and epoch 80 were copied.
- Observation: BrepARG same-data fallback produces readable files but collapses toward simple topology.
  Evidence: The prior quality audit recorded 92 STEP files, 91 readable, 75 BRep-valid, 5 complex by entity thresholds, 3 complex plus valid closed, 0 strict accepted, median 6 faces and 12 edges.
- Observation: Official BrepARG ABC pretrained weights are not directly comparable to the current local protocol.
  Evidence: The official AR embedding shape was `[7222, 256]`, while the current local vocabulary protocol is 10294 tokens.
- Observation: The long VQ-VAE stage completed normally, but its periodic checkpoints now dominate the active D-drive run size.
  Evidence: The long-run root is about `12.661 GiB`; the VQ-VAE subtree is about `12.038 GiB` and contains the best alias plus 17 periodic checkpoints of about `0.645 GiB` each. D: had only about `7.84 GiB` free on 2026-07-23, while E: had about `3.64 TiB` free. Periodic VQ-VAE checkpoints should be copied and verified on E: or removed after explicit approval, while the best alias must remain available to the active pipeline.

## Decision Log

- Decision: Train a same-data BrepARG fallback baseline rather than using official pretrained weights directly.
  Rationale: The official checkpoint vocabulary does not match the current local sequence protocol, so using it would be an unfair and technically incompatible comparison.
  Date/Author: 2026-07-20 / Codex.
- Decision: Continue BrepARG VQ-VAE from the previous best checkpoint but write outputs into a new long-run directory.
  Rationale: This preserves the prior baseline as a fixed result while testing whether longer VQ-VAE training helps.
  Date/Author: 2026-07-20 / Codex.
- Decision: Rebuild sequences after VQ-VAE long training and train AR from scratch for the long baseline.
  Rationale: A changed VQ-VAE can change token assignments; reusing the old AR optimizer state would mix protocols and weaken the comparison.
  Date/Author: 2026-07-20 / Codex.
- Decision: Save only best aliases, final checkpoint, and every 20 epochs.
  Rationale: This meets the user's disk-space constraint and prevents D: from filling during long training.
  Date/Author: 2026-07-20 / Codex.
- Decision: Treat generation-time filtering as a final quality gate, not as the main fix.
  Rationale: Prior teacher-forcing and true-token reconstruction diagnostics showed failures before free-running generation, so filtering bad outputs cannot repair the learned model or reconstruction path.
  Date/Author: 2026-07-20 / Codex.

## Outcomes & Retrospective

The long original BrepARG baseline is complete through VQ-VAE, sequence construction, AR epoch 300, generation, STEP/PNG validation, and quality auditing. Resuming from the finite epoch-127 AR best improved validation CE from `0.819257915` to `0.765318`, but the generated geometry remains strongly biased toward simple or irregular topology. The final 100-output run produced `100/100` readable closed STEP files and `100/100` PNG previews, with `86/100` BRep-valid, `13/100` complex by the shared face/edge threshold, and `6/100` strict-quality accepted. This is better than the prior 80-epoch same-data baseline, which had `5/92` complex and `0/92` strict accepted, but the unchanged median of `6` faces and `12` edges shows that more training alone does not solve topology collapse. The remaining work is a matched final comparison against V13 FSQ/RCM/AR and a repair decision based on the already identified VQ-VAE, AR, and ordering bottlenecks.

## Context and Orientation

`BrepARG\train_vqvae.py` trains the original BrepARG surface/edge VQ-VAE. A VQ-VAE is an autoencoder that compresses geometry patches into discrete tokens. `BrepARG\2sequence.py` converts parsed CAD records into BrepARG token sequences using a trained VQ-VAE. `BrepARG\train_ar.py` trains the original BrepARG autoregressive transformer on those token sequences. An autoregressive model predicts the next token from previous tokens. `BrepARG\generate_brep.py` samples token sequences and reconstructs STEP/STL geometry.

The same-data BrepARG input files are already staged under `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\data_staged`. The initial VQ-VAE checkpoint for continuation is `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\vqvae_3060_safe_len1536_bs4_20260717_d\same_data_abc\abc_se_vqvae_best.pt`. The new long-run root is `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720`.

The current V13 method uses FSQ VQ-VAE plus RCM/GNN ordering and AR training. Current evidence indicates several bottlenecks: FSQ heavy-tail reconstruction on complex curved surfaces, AR weakness on complex token sequences even under teacher forcing, and ordering sensitivity where DFS was better than RCM in a matched recovered evaluation.

## Plan of Work

The first milestone is a disk-safe BrepARG long baseline. The code in `BrepARG\utils.py`, `BrepARG\train_vqvae.py`, and `BrepARG\trainer.py` exposes a target validation loss and sparse checkpoint behavior. The script `tools\run_breparg_same_data_long_vq400_ar300.ps1` runs VQ-VAE continuation to 400 epochs or validation reconstruction loss at or below `1e-6`, rebuilds sequences, trains AR to 300 epochs, generates 100 outputs, and audits them.

The second milestone is a comparison report. It will compare the long BrepARG baseline against the existing V13 results using the same quality metrics: STEP readability, BRep validity, closed solid structure, entity complexity, strict quality accepted count, median face count, median edge count, and complex-curved subset behavior.

The third milestone is a repair plan for V13. Candidate repairs must be isolated one variable at a time: improve FSQ/patch reconstruction, test DFS order against RCM on the same data, add length/complexity bucketed training diagnostics, and use generation filtering only as a final quality gate.

## Concrete Steps

From `D:\luolin\V13`, run:

    powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_breparg_same_data_long_vq400_ar300.ps1

The long script writes logs under:

    D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\vqvae_3060_long_vq400_ar300_save20_20260720\train_vqvae_long.log
    D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\sequence_3060_long_vq400_ar300_save20_20260720\build_sequence.log
    D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\ar_3060_long_vq400_ar300_save20_20260720\train_ar_long.log
    D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\generated_3060_long_vq400_ar300_save20_20260720\generate.log

To monitor D: space:

    wmic logicaldisk where "DeviceID='D:' or DeviceID='E:'" get DeviceID,FreeSpace,Size /format:list

To inspect checkpoints after VQ-VAE training:

    dir D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\vqvae_3060_long_vq400_ar300_save20_20260720\same_data_abc

Expected VQ-VAE checkpoint pattern: `abc_se_vqvae_best.pt`, periodic `abc_se_vqvae_epoch_80.pt`, `abc_se_vqvae_epoch_100.pt`, and so on, plus final if the final epoch is not already periodic. There should not be an epoch file for every best update.

Expected AR checkpoint pattern: `abc_ar_vqvae_best_model.pt`, `abc_ar_vqvae_best_model_hf`, periodic `epoch_20.pt`, `epoch_40.pt`, and so on, plus final if the final epoch is not already periodic. There should not be an epoch file for every best update.

## Validation and Acceptance

Run:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_breparg_training_args
    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m py_compile BrepARG\train_vqvae.py BrepARG\train_ar.py BrepARG\trainer.py BrepARG\utils.py

Expected result: all unit tests pass and `py_compile` exits with code 0.

The long baseline is accepted when `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\breparg_same_data_long_quality_summary.json` exists and contains the audit metrics for generated outputs. The result should then be compared to the prior same-data BrepARG summary and the V13 summaries.

## Idempotence and Recovery

The long script is designed to be rerun. It resumes VQ-VAE from the latest periodic checkpoint in the long-run VQ-VAE directory, or from the long-run best alias, or from the prior baseline best if no long-run checkpoint exists. It rebuilds the sequence pickle if the sequence file is older than the selected VQ-VAE best checkpoint. It resumes AR from the latest long-run AR periodic checkpoint if one exists; otherwise it starts AR from scratch after sequence construction.

Do not delete D: training inputs under `data_staged` while the long run is active. Do not delete D: best checkpoints until the E: copy is verified by file size, because E: has shown intermittent I/O errors.

## Artifacts and Notes

Changed files:

    BrepARG\utils.py
    BrepARG\train_vqvae.py
    BrepARG\trainer.py
    tests\test_breparg_training_args.py
    tools\run_breparg_same_data_long_vq400_ar300.ps1

Verification already observed:

    Ran 13 tests in 2.780s
    OK

## Interfaces and Dependencies

The long baseline uses the existing Python environment:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe

The GPU is the local NVIDIA RTX 3060. Generation uses `BREPARG_SERIAL_WRITE=1` for Windows-safe STEP/STL writing and `BREPARG_JOINT_OPTIMIZE_DEVICE=cpu` to avoid CUDA-only chamfer extension assumptions during reconstruction.

Revision note 2026-07-20: Created this plan after preparing disk-safe checkpointing and the long BrepARG baseline script. The plan records why AR is retrained from scratch after VQ-VAE continuation and why E: copies must be verified before D: deletion.

Revision note 2026-07-23: Recorded completed VQ-VAE and sequence stages, live AR epoch/best metrics, numerical health checks, comparison with the prior AR baseline, and the renewed D-drive capacity risk.

Revision note 2026-07-24: Recorded the reboot during epoch 128, the finite epoch-127 best checkpoint (`0.819257915`), successful best-checkpoint resume under a separate workspace run directory, new PID/log locations, and the post-cleanup D-drive capacity.
Revision note 2026-07-26: Recorded successful completion of resumed AR training through epoch 300, best validation CE `0.765318`, epoch-300 validation CE `0.767252`, the finite best and periodic checkpoints, and the fact that the current low GPU utilization is caused by training having ended rather than an active CPU data-loading bottleneck.
Revision note 2026-07-26 15:17: Started a generation-only original BrepARG baseline from the resumed AR best checkpoint. The wrapper separates native stdout/stderr so benign Windows CUDA allocator warnings cannot abort the run; generated STEP/STL artifacts are retained in a new directory and validation will render PNG previews afterward.
Revision note 2026-07-26 17:04: Diagnosed the original generator as stuck in CPU BRep reconstruction after `8/100`, stopped the non-progressing process, audited the partial outputs, and replaced the single 100-sample invocation with a tested resumable batch controller. Each four-sample batch has an external 180-second process-tree timeout and a new seed, so one pathological OCC/joint-optimization candidate cannot block the experiment indefinitely. This changes orchestration reliability only; original BrepARG model weights and sampling parameters remain unchanged.
Revision note 2026-07-28: Recorded completion of the 100-output long BrepARG generation and final audit. All STEP and PNG artifacts were retained; `86/100` were BRep-valid, `13/100` met the complexity threshold, and `6/100` passed the strict quality gate. Visual inspection confirmed that increased entity counts do not consistently correspond to convincing complex CAD geometry.
