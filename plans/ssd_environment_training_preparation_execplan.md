# Prepare SSD-Based Local Processing and Training

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` in the repository root. The user is copying the ABC dataset to a solid-state drive and wants the environment, data-processing entry points, and training launch preparation ready before the copy finishes. The observable result is that the `brepgen_env` conda environment imports the required packages, the project tests run, and reusable scripts can validate an SSD path, parse ABC chunks on Windows, and print the exact commands for training.

## Purpose / Big Picture

After this preparation, the user can point one config file or command at the SSD location and immediately run preflight checks, parse raw ABC STEP chunks into parsed pkl files, and launch the existing FSQ/RCM training pipeline. The scripts are designed for the local Windows machine and RTX 3060 GPU rather than the original Linux server paths.

## Progress

- [x] (2026-06-25 01:00 +08:00) Inspected the existing conda environments and confirmed `brepgen_env` is the intended Python 3.9 environment.
- [x] (2026-06-25 01:01 +08:00) Verified PyTorch 2.2.2+cu118 can see the NVIDIA GeForce RTX 3060 with 12 GB VRAM.
- [x] (2026-06-25 01:02 +08:00) Installed missing `shutup==0.3.0` into `brepgen_env`.
- [x] (2026-06-25 01:03 +08:00) Discovered `breparg_improvements/process_abc.py` uses Linux-only `signal.SIGALRM`, so Windows needs a local parser wrapper.
- [x] (2026-06-25 01:06 +08:00) Added failing tests for chunk selection, training environment rendering, and per-chunk output paths.
- [x] (2026-06-25 01:10 +08:00) Implemented `tools/prepare_ssd_pipeline.py`, `tools/process_abc_windows.py`, and `local_training_config.json`.
- [x] (2026-06-25 01:13 +08:00) Ran preflight against the current local `D:\luolin\V13\ABC` path and wrote readiness/command reports.
- [x] (2026-06-25 01:15 +08:00) Investigated project test crash; found Windows fatal exception from NumPy BLAS matmul inside `gnn_ordering.node_features`.
- [x] (2026-06-25 01:18 +08:00) Replaced the tiny dense `A @ A` two-hop calculation with adjacency-set traversal and added a regression test.
- [x] (2026-06-25 01:19 +08:00) Verified helper tests pass and `breparg_improvements/test_all.py` reports 58 passed, 0 failed.
- [x] (2026-06-25 01:20 +08:00) Ran Windows parser smoke: 5 STEP files, 2 parsed successfully, 3 filtered, 0 errors.
- [x] (2026-06-25 01:24 +08:00) Wrote `local_reports/ssd_environment_training_preparation.md` with exact SSD follow-up commands.

## Surprises & Discoveries

- Observation: `brepgen_env` is already GPU-capable.
  Evidence: `torch.cuda.is_available()` returned `True`, CUDA version was `11.8`, and `torch.cuda.get_device_name(0)` returned `NVIDIA GeForce RTX 3060`.
- Observation: The original full-corpus parser is not Windows-safe as written.
  Evidence: `breparg_improvements/process_abc.py` uses `signal.SIGALRM`; this signal is not available in standard Windows Python.
- Observation: Existing `breparg_improvements/train.py` is already configurable enough for SSD paths.
  Evidence: it reads `NS_POOL`, `NS_OUTBASE`, `NS_OUT`, `NS_N`, `NS_VQ_SAMPLES`, `NS_VQ_EPOCHS`, `NS_VQ_BS`, `NS_AR_EPOCHS`, `NS_AR_BS`, and related settings from environment variables.
- Observation: `conda run` on this Windows console can fail when forwarding non-GBK output.
  Evidence: running `conda run -n brepgen_env python breparg_improvements/test_all.py` raised `UnicodeEncodeError: 'gbk' codec can't encode character`.
- Observation: The environment's NumPy/PyTorch combination had a native crash on tiny NumPy matrix multiplication in the GNN test path.
  Evidence: a minimal script crashed with Windows fatal exception `0xc06d007f` at `two_hop = ((A @ A) > 0) ...`; replacing it with adjacency-set traversal made tests pass.

## Decision Log

- Decision: Reuse `brepgen_env` instead of creating a new conda environment.
  Rationale: It already contains the heavy CAD and GPU dependencies, including `occwl`, `OCC`, PyTorch with CUDA, `diffusers`, and `transformers`.
  Date/Author: 2026-06-25 / Codex.
- Decision: Add Windows-specific preparation scripts rather than modifying the original server-oriented parser in place.
  Rationale: This keeps the upstream script intact and gives the local machine a safe timeout strategy.
  Date/Author: 2026-06-25 / Codex.
- Decision: Select chunks by the numeric ID in `abc_XXXX_step_v00`, not by sorted position.
  Rationale: The local SSD may contain only some chunks during copying. If only chunk4 exists, `--chunks 4` must mean chunk ID 4, not the fifth directory in a partial list.
  Date/Author: 2026-06-25 / Codex.
- Decision: Use direct `C:\Users\YU\.conda\envs\brepgen_env\python.exe` commands with `PYTHONUTF8=1`.
  Rationale: Direct environment Python avoids the observed `conda run` stdout encoding failure and gives more reliable logs for long jobs.
  Date/Author: 2026-06-25 / Codex.

## Outcomes & Retrospective

The environment and local preparation scripts are now verified. The remaining work is only to point `local_training_config.json` at the SSD paths after copying finishes and rerun preflight.

## Context and Orientation

ABC chunks are directories named like `abc_0004_step_v00`. Each chunk contains model subdirectories, and each model directory contains a STEP CAD file. A STEP file is the raw CAD exchange format. A parsed pkl is a Python pickle file containing one CAD model's parsed surfaces, edges, bounding boxes, and adjacency data. The existing training code reads parsed pkl files from a flat directory or one chunk-subdirectory level.

The existing improved training script is `breparg_improvements/train.py`. It does not need hard-coded Linux paths if the caller sets environment variables. `NS_POOL` points to parsed pkls, `NS_OUTBASE` points to the output root, and `NS_OUT` names the training run. On this local RTX 3060, default batch sizes should be more conservative than the multi-GPU server defaults.

## Plan of Work

Create `tools/prepare_ssd_pipeline.py` to validate paths, discover chunk directories, check imports and GPU state, render PowerShell commands, and write reports. Create `tools/process_abc_windows.py` to parse chunks on Windows with per-file subprocess timeouts and per-chunk output directories. Create `local_training_config.json` with editable defaults. Run a unit test for helper behavior, run the project unit test, run a tiny parser smoke test on current chunk4, and write a readiness report.

## Concrete Steps

Run from `D:\luolin\V13`.

First verify helper tests fail before implementation:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q

Then implement the scripts and rerun:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q

Run project method tests:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' breparg_improvements/test_all.py

Run preflight against the current local data path until the SSD path is known:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools/prepare_ssd_pipeline.py --raw-root D:\luolin\V13\ABC --write-report

Run a Windows parser smoke test on current chunk4:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools/process_abc_windows.py --raw-root D:\luolin\V13\ABC --chunks 4 --out D:\luolin\V13\processed_local\windows_parser_smoke --workers 2 --timeout 30 --limit 5

## Validation and Acceptance

The preparation is accepted when imports pass, the helper tests pass, `test_all.py` reports no failures, preflight writes a report with GPU and chunk information, and the Windows parser smoke command creates at least one compatible parsed pkl or records only expected statuses such as `multi`, `filtered`, or `timeout`.

## Idempotence and Recovery

All scripts are additive and safe to rerun. Parser output is resumable because existing pkl files are skipped. Temporary files use a `.tmp<PID>` suffix and are atomically renamed after successful writes. If an SSD path changes, rerun preflight with the new path or update `local_training_config.json`.

## Artifacts and Notes

Expected artifacts are:

    D:\luolin\V13\local_training_config.json
    D:\luolin\V13\tools\prepare_ssd_pipeline.py
    D:\luolin\V13\tools\process_abc_windows.py
    D:\luolin\V13\tests\test_local_pipeline_helpers.py
    D:\luolin\V13\local_reports\ssd_pipeline_readiness.json
    D:\luolin\V13\local_reports\ssd_pipeline_commands.md

## Interfaces and Dependencies

Use `conda run -n brepgen_env python` for all commands. `prepare_ssd_pipeline.py` uses only the Python standard library plus installed `torch` for GPU checks when available. `process_abc_windows.py` imports `BrepARG/process_data/process_brep.py` and `occwl.io.load_step`, then writes parsed pkl dictionaries compatible with the existing BrepARG loaders.

Revision note: Initial ExecPlan created after inspecting environment and before implementing SSD preparation scripts.

Revision note: Updated after implementing scripts, fixing the local GNN/NumPy crash, and verifying tests plus parser smoke.
