# Clean V13 Workspace and Build Server-Ready Package

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows the repository-level execution-plan rules in `PLANS.md`. The user wants the V13 workspace to become clean again after many validation sessions, while still preserving the information and files needed to restart work on a rented server.

## Purpose / Big Picture

After this work, a human or agent opening `D:\luolin\V13` should not need to sort through dozens of stale reports to understand what matters. The repository should have a small set of canonical documents, a clear server-start entry point, and a packaged transfer bundle or manifest that can be copied to a Linux server and verified before training starts. The cleanup must not erase useful source code, checkpoints, paper materials, or the current server recovery path.

The visible result is a tidy root with concise documentation, refreshed manifests, safe deletion of generated clutter, and a server package that can be checked locally with the repository's verification tools.

## Progress

- [x] (2026-07-08 11:36 +08:00) Read `AGENTS.md`, `PLANS.md`, root project index, core README/HANDOFF, and server handoff tooling enough to identify the cleanup shape.
- [x] (2026-07-08 11:39 +08:00) Confirmed this work must operate in the current workspace rather than a new git worktree, because the task is to clean the exact dirty workspace and many cleanup targets are untracked generated files.
- [x] (2026-07-08 12:03 +08:00) Read and classified documentation/report files under `docs/`, `plans/`, `local_reports/`, `breparg_improvements/docs/`, and paper docs. The useful server restart path is now parsed shards to VQ patch shards to VQ-VAE, while older G20/G100 and VQ diagnostics remain paper/failure-analysis context.
- [x] (2026-07-08 12:06 +08:00) Created concise canonical entry docs: `README.md`, refreshed `PROJECT_INDEX.md`, `docs/SERVER_START_HERE.md`, and `docs/CLEANUP_MANIFEST_20260708.md`.
- [x] (2026-07-08 12:06 +08:00) Added the server environment file `environment.server.yml`, bootstrap script `tools/server_bootstrap.sh`, direct server runner `tools/run_vqvae_from_patch_shards.sh`, and local package builder `tools/build_server_package.ps1`.
- [x] (2026-07-08 12:08 +08:00) Removed safe generated clutter: Python caches, `tmp/`, selected local report run logs, LaTeX auxiliary files, and historical PDF render checks under `papers/aaai_v13/latex/rendered/`, while retaining canonical `main_page-*` and `supplement_page-*` renders.
- [x] (2026-07-08 12:16 +08:00) Built the lightweight server package at `dist/v13_server_ready_20260708.zip`; it is about 396 KB and contains 102 entries including the server docs, environment file, scripts, source, tools, tests, and package manifest.
- [x] (2026-07-08 12:12 +08:00) Verified local code path: `py_compile` passed for sharding/training files, `C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_local_pipeline_helpers` ran 118 tests OK, `PYTHONUTF8=1 ... breparg_improvements/test_all.py` reported 58 passed and 0 failed, and Git Bash `bash -n` passed for the server shell scripts.
- [x] (2026-07-08 12:13 +08:00) Removed test-regenerated Python caches after validation.

## Surprises & Discoveries

- Observation: `AGENTS.md` references `.agent/PLANS.md`, but that path does not exist in this checkout.
  Evidence: `Test-Path '.agent/PLANS.md'` returned false, while root `PLANS.md` exists and contains the ExecPlan rules.
- Observation: The project is already in a normal checkout, not an isolated git worktree.
  Evidence: `git rev-parse --git-dir` and `git rev-parse --git-common-dir` both returned `.git`.
- Observation: There is no obvious Python dependency lock file or environment file at the repository root.
  Evidence: searching for `requirements*.txt`, `environment*.yml`, `pyproject.toml`, `Dockerfile`, and related files returned no matches.
- Observation: The default `python` on this Windows machine can compile files but does not have `numpy` or `torch`, so full helper tests fail there for environment reasons.
  Evidence: `python -m unittest tests.test_local_pipeline_helpers` failed with `ModuleNotFoundError: No module named 'numpy'` and `ModuleNotFoundError: No module named 'torch'`; the same command passed in `C:\Users\YU\.conda\envs\brepgen_env\python.exe`.
- Observation: Bash is not on PATH, but Git Bash exists locally and can syntax-check the Linux scripts.
  Evidence: plain `bash -n` failed with "The term 'bash' is not recognized"; `D:\Program Files\Git\bin\bash.exe -n ...` returned 0.

## Decision Log

- Decision: Work in the current checkout instead of creating a new worktree.
  Rationale: The requested cleanup targets the current dirty workspace, including untracked reports and generated outputs. A fresh worktree would hide most of the mess and would not satisfy the user's goal.
  Date/Author: 2026-07-08 / Codex
- Decision: Treat root `PLANS.md` as the active ExecPlan specification.
  Rationale: The repository contains root `PLANS.md` with the required rules, while `.agent/PLANS.md` is missing.
  Date/Author: 2026-07-08 / Codex
- Decision: Preserve source code and currently modified tracked files unless they are directly required for packaging.
  Rationale: `git status --short` shows existing modifications that may be user or prior-session work. Cleanup can focus on generated reports, documentation consolidation, caches, and packaging without reverting code.
  Date/Author: 2026-07-08 / Codex
- Decision: Add a new sharded-server runner instead of replacing older 2026-07-06 transfer/recovery tooling.
  Rationale: Existing tests and older recovery docs depend on the historical transfer-manifest path. The current server restart needs the newer parsed-shard and VQ patch-shard path, so `tools/run_vqvae_from_patch_shards.sh` provides a direct current entry while leaving older guardrail tools intact.
  Date/Author: 2026-07-08 / Codex
- Decision: Ignore generated local state in `.gitignore` rather than forcing generated reports into source control.
  Rationale: `local_reports/`, `tmp/`, `dist/`, `breparg_improvements/repro_outputs/`, and historical paper renders are regenerated evidence or packaging byproducts. Ignoring them keeps the project visually clean while preserving canonical docs and scripts.
  Date/Author: 2026-07-08 / Codex

## Outcomes & Retrospective

Final outcome 2026-07-08: The workspace now has a concise root entry (`README.md`), a compact project map (`PROJECT_INDEX.md`), a server restart guide (`docs/SERVER_START_HERE.md`), and a cleanup manifest (`docs/CLEANUP_MANIFEST_20260708.md`). The server path is explicit: upload parsed shards, build VQ patch shards, then run VQ-VAE from `NS_VQ_PATCH_SHARD_ROOT` using `tools/run_vqvae_from_patch_shards.sh`.

The lightweight server package was created at `dist/v13_server_ready_20260708.zip` with 102 entries and size about 396 KB. It excludes heavy data, checkpoints, `local_runs/`, generated local reports, caches, and temporary render checks. Large artifacts still need explicit upload according to `docs/SERVER_START_HERE.md`.

Validation passed in the real project Python environment: 118 helper tests OK and 58 BrepARG improvement tests OK. Local default Python is not sufficient because it lacks NumPy and Torch. Bash syntax checks passed with Git Bash, but full server CUDA/OCC training readiness must still be verified on the Linux GPU server after upload.

## Context and Orientation

The repository root is `D:\luolin\V13`. It contains an upstream `BrepARG/` checkout, project code in `breparg_improvements/`, helper scripts in `tools/`, tests in `tests/`, paper work in `papers/aaai_v13/`, local run artifacts in `local_runs/`, raw or processed data in `ABC/` and `processed_local/`, and many generated reports in `local_reports/`.

The root `AGENTS.md` says complex work should use an ExecPlan. The root `PLANS.md` defines that an ExecPlan must be self-contained and updated as work proceeds. This file is the living plan for cleanup and packaging.

The most important server path already exists in code. `tools/prepare_server_handoff.py` checks local artifacts and writes a preflight manifest. `tools/build_server_transfer_manifest.py` lists files that need to be uploaded to `/workspace/V13` and `/workspace/ABC/processed/train_outputs`. `tools/verify_server_transfer.py` verifies uploaded files. `tools/run_server_quality_recovery.py` runs server-side gates and can start training only with `--start`.

Important source files are in `breparg_improvements/`, especially `train.py`, `fsq_quantise.py`, `gnn_ordering.py`, `constrained_decoding.py`, `vqvae_sampling.py`, `sharded_data.py`, `sequence_sharding.py`, `ar_training_utils.py`, and `training_stability.py`. Important launchers are in `tools/`, especially `run_vqvae_complex_recovery.sh`, `run_source_path_sequence_rebuild.sh`, and `run_ar_v13_long_context.sh`.

Important large artifacts are intentionally not ordinary documentation. Existing docs refer to `local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt`, `local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl`, `local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl`, and `ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt`. These should not be deleted by this cleanup.

## Plan of Work

First, classify documentation and reports. The classification should cover `docs/`, `plans/`, `local_reports/`, `breparg_improvements/README.md`, `breparg_improvements/HANDOFF.md`, `breparg_improvements/docs/`, and key paper docs under `papers/aaai_v13/`. Each file should land in one of four groups: keep as canonical, merge into a canonical document, delete as stale/generated, or leave untouched because it belongs to paper source or code history.

Second, write concise canonical documents. The cleaned workspace should have a root `README.md` or updated `PROJECT_INDEX.md` that answers what the project is, what files matter, and how to run basic tests. It should also have a server start guide that explains environment setup, upload, verification, and guarded training start. If an environment file is missing, add a minimal environment specification from the imports used by the code and the known conda environment notes in existing docs.

Third, remove safe generated clutter. Safe candidates include `__pycache__/`, `.pytest_cache/`, zero-byte logs, temporary PDF render checks under `tmp/`, redundant generated render outputs that can be rebuilt from paper source, and stale local reports after their useful facts are merged. Deletion must be done with path checks that keep all removed paths inside `D:\luolin\V13`.

Fourth, refresh packaging artifacts. Use `tools/prepare_server_handoff.py` and `tools/build_server_transfer_manifest.py` to create fresh local reports and an upload helper. If the repo lacks a single command for a server package, add a small script under `tools/` that creates a lightweight source package excluding large generated outputs and data, while relying on the manifest for large checkpoint/data transfer.

Fifth, verify. Run syntax checks for shell launchers if Bash is available, run the focused Python unit tests that do not require unavailable heavy data, and run the transfer manifest verifier in local dry-run mode with path maps. Record any missing external dependencies or GPU-only checks as residual risk rather than claiming they passed locally.

## Concrete Steps

Run these commands from `D:\luolin\V13` unless a step says otherwise.

Inspect the dirty tree:

    git status --short

Read and classify documentation:

    Get-ChildItem -Recurse -File docs,plans,local_reports,breparg_improvements\docs,papers\aaai_v13 -ErrorAction SilentlyContinue

Create or update canonical docs using normal file edits.

Remove generated caches only after classification:

    Get-ChildItem -Recurse -Directory -Force -Filter __pycache__
    Get-ChildItem -Recurse -Directory -Force -Filter .pytest_cache

Build the lightweight server package:

    powershell -ExecutionPolicy Bypass -File tools/build_server_package.ps1

On the server, after uploading parsed shards and baseline checkpoints, use:

    cd /workspace/V13
    bash tools/server_bootstrap.sh
    bash tools/run_vqvae_from_patch_shards.sh --build-patch-shards --run-train

Run focused tests:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m unittest tests.test_local_pipeline_helpers
    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:\Users\YU\.conda\envs\brepgen_env\python.exe breparg_improvements/test_all.py

If the full `test_all.py` command fails because local Python lacks heavy ML dependencies, record the failure and run the lighter available tests instead.

## Validation and Acceptance

The cleanup is accepted when the repository has a concise canonical entry point, a concise server-start path, and a cleanup manifest listing what was removed or merged. `git status --short` should show intentional documentation/package changes and should no longer be dominated by disposable cache and log files. This was achieved by adding canonical docs, updating `.gitignore`, deleting `tmp/`, deleting historical PDF render checks, and removing caches after validation.

The packaging is accepted when `tools/build_server_package.ps1` writes `dist/v13_server_ready_20260708.zip`, the zip contains the server guide, environment file, bootstrap and runner scripts, source, tools, tests, and package manifest, and the server guide states how to upload excluded large artifacts. This was achieved with a zip of about 396 KB containing 102 entries.

The runtime path is accepted when at least one focused local test command passes. GPU, CUDA, OCC, and full training checks may only be marked as verified if the current machine actually has those capabilities. Local validation passed for helper tests, improvement tests, Python compilation, and Bash syntax; server CUDA/OCC readiness remains a server-side verification item.

## Idempotence and Recovery

All cleanup operations should be repeatable. Generated caches can be recreated by Python. Fresh local reports should use `20260708` names so older reports are not overwritten until their facts have been merged. Deletions must be limited to paths discovered inside `D:\luolin\V13`.

If a deletion removes something useful, recovery is available from git for tracked files. For untracked generated reports, the cleanup manifest should preserve enough context to understand what was removed and why. Large checkpoints and data directories must not be deleted by this plan.

## Artifacts and Notes

The cleanup manifest should be written to `docs/cleanup_manifest_20260708.md` or another concise canonical location. Fresh server handoff reports should use `local_reports/*20260708*` while the final user-facing start guide should live in `docs/` or at the root.

## Interfaces and Dependencies

The project is Python-based. Known runtime dependencies from code imports are `torch`, `numpy`, `transformers`, `diffusers`, `einops`, and OpenCascade Python bindings exposed as `OCC.Core.*`. Existing docs mention a conda environment named `brepgen` on Linux and a separate OCC-capable environment. The packaging result should make this explicit in an environment file or server guide.

Revision note 2026-07-08 / Codex: Initial ExecPlan created after reading repository rules and core server tooling so cleanup can proceed from a self-contained plan.

Revision note 2026-07-08 / Codex: Updated after implementation to record canonical docs, safe deletion scope, server package artifact, validation commands, and residual server-side risks.
