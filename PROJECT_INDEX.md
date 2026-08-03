# V13 Project Index

Updated: 2026-07-08 +08:00

This is the compact map for the cleaned V13 workspace. It points to the files needed to understand, package, and restart the project on a server.

## Start Here

Read these first:

    README.md
    docs/SERVER_START_HERE.md
    docs/v13_sharded_dataset_operator_guide.md
    docs/CLEANUP_MANIFEST_20260708.md

The active cleanup and packaging plan is:

    plans/v13_workspace_cleanup_and_server_packaging_execplan.md

## Code

V13 improvements:

    breparg_improvements/

Server and data tools:

    tools/

Tests:

    tests/

Upstream BrepARG source used by the project:

    BrepARG/

## Server Package

Environment file:

    environment.server.yml

Local package builder:

    tools/build_server_package.ps1

Server bootstrap:

    tools/server_bootstrap.sh

AutoDL VQ-VAE scratch helper:

    tools/autodl_vqvae_scratch.sh

Direct VQ-VAE launch from patch shards:

    tools/run_vqvae_from_patch_shards.sh

Build the package locally with:

    powershell -ExecutionPolicy Bypass -File tools/build_server_package.ps1

The package is written to `dist/` and excludes heavy data, checkpoints, generated reports, caches, and temporary render checks.

## Data And Artifacts

Authoritative local parsed-shard root:

    C:\V13_abc_parsed_shards

Server parsed-shard target:

    /workspace/ABC/processed/abc_parsed_shards

Server patch-shard target:

    /workspace/ABC/processed/vqvae_patch_shards

Baseline VQ-VAE checkpoint:

    ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt

Current AR branch:

    local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/

Do not include `ABC/` or `local_runs/` in the lightweight package. Upload them by explicit data/checkpoint transfer steps when the server needs them.

## Current Server Order

1. Upload lightweight project package.
2. Upload parsed shards to `/workspace/ABC/processed/abc_parsed_shards`.
3. Upload baseline VQ-VAE checkpoint.
4. Run `bash tools/server_bootstrap.sh` from the repo root for VQ-VAE mode. On RTX 5090 / Blackwell, diagnose first with `REPO_ROOT=$(pwd) V13_SKIP_INSTALL=1 V13_REQUIRE_CUDA=1 bash tools/server_bootstrap.sh`, then repair only if needed with `REPO_ROOT=$(pwd) V13_FORCE_CU128=1 V13_REQUIRE_CUDA=1 bash tools/server_bootstrap.sh`.
5. On AutoDL, run `bash tools/autodl_vqvae_scratch.sh preflight`, `verify-preflight`, `diagnose`, `verify-diagnose`, `smoke`, `verify-smoke`, and `gate` before `bash tools/autodl_vqvae_scratch.sh full --no-tail`. On generic servers, run `bash tools/run_vqvae_from_patch_shards.sh --build-patch-shards --run-train` for resume training, or add `--scratch` for a from-zero VQ-VAE run.
6. Promote the VQ-VAE checkpoint only after the reconstruction benchmark passes.
7. Rebuild source-path-aware sequences.
8. Train AR1536/AR2048 only after sequence verification.
9. Generate and promote paper figures only after automated gates and human visual review.

## Paper Workspace

Paper source and diagnostic materials live under:

    papers/aaai_v13/

The current manuscript is diagnostic/reproducibility evidence. It should not claim positive paper-quality generated CAD until the VQ-VAE recovery, sequence rebuild, AR long-context run, generated-quality gate, and human visual review all pass.

## Generated Local State

Generated reports and temporary outputs are ignored by git:

    local_reports/
    tmp/
    dist/
    breparg_improvements/repro_outputs/
    papers/aaai_v13/latex/rendered/

These directories may exist locally, but they are not the source of truth for a fresh server start. The canonical path is documented in `docs/SERVER_START_HERE.md`.
