# V13 BrepARG Recovery Workspace

This repository is the cleaned working entry point for the V13 CAD/B-rep generation recovery project. It contains the BrepARG upstream checkout, V13 method improvements, data-sharding tools, server launch scripts, tests, and paper materials. Heavy datasets, checkpoints, and generated runs are kept outside the source package and are transferred by manifest or explicit upload commands.

## What Matters Now

Core code:

    breparg_improvements/
    tools/
    tests/
    BrepARG/

Current operator docs:

    docs/SERVER_START_HERE.md
    docs/v13_sharded_dataset_operator_guide.md
    docs/CLEANUP_MANIFEST_20260708.md
    PROJECT_INDEX.md

Execution plans:

    plans/v13_workspace_cleanup_and_server_packaging_execplan.md
    plans/v13_sharded_dataset_execplan.md
    plans/v13_generation_quality_recovery_execplan.md

Server environment:

    environment.server.yml
    tools/server_bootstrap.sh
    tools/autodl_vqvae_scratch.sh
    tools/run_vqvae_from_patch_shards.sh
    tools/build_server_package.ps1

## Current Data State

The local parsed ABC pool was compressed into 100 verified parsed shards at:

    C:\V13_abc_parsed_shards

The final local shard summary is:

    local_reports/v13_parsed_shards_manifest_croot_0000_0099_20260708.json

The server should receive those shards at:

    /workspace/ABC/processed/abc_parsed_shards

Then the server builds VQ patch shards at:

    /workspace/ABC/processed/vqvae_patch_shards

VQ-VAE training should read the patch shards through `NS_VQ_PATCH_SHARD_ROOT`, not by scanning hundreds of thousands of extracted parsed `.pkl` files.

## Quick Local Checks

From the repository root:

    python -m unittest tests.test_local_pipeline_helpers

If the Python environment has Torch, NumPy, Transformers, Diffusers, and OpenCascade bindings, also run:

    python breparg_improvements/test_all.py

The full training path requires a GPU server and the data/checkpoint artifacts described in `docs/SERVER_START_HERE.md`.

## Package For Server

Build the lightweight code/docs package from Windows:

    powershell -ExecutionPolicy Bypass -File tools/build_server_package.ps1

The script writes a zip and a `.sha256` sidecar under `dist/`. That zip intentionally excludes `ABC/`, `local_runs/`, `tmp/`, `local_reports/`, caches, and generated render histories. Upload large checkpoints and parsed shards separately, as described in `docs/SERVER_START_HERE.md`.

## Server Start

On the server, unzip the package to `/workspace/V13`, upload parsed shards to `/workspace/ABC/processed/abc_parsed_shards`, then run:

    cd /workspace/V13
    bash tools/server_bootstrap.sh
    bash tools/run_vqvae_from_patch_shards.sh --build-patch-shards --run-train

The bootstrap defaults to VQ-VAE mode, where OCC is not required. On RTX 5090 / Blackwell hosts, first run it from the repo root with `REPO_ROOT=$(pwd) V13_SKIP_INSTALL=1 V13_REQUIRE_CUDA=1` for a no-install CUDA smoke test. If that fails because of missing modules or an incompatible CUDA kernel, rerun with `REPO_ROOT=$(pwd) V13_FORCE_CU128=1 V13_REQUIRE_CUDA=1` so the official PyTorch CUDA 12.8 wheel and pinned VQ-VAE dependencies are installed.

The VQ launcher verifies parsed shards, builds patch shards if requested, writes a ledger, checks the runtime Python/Torch/Diffusers stack, and starts `breparg_improvements/train.py --stage vqvae` with the sharded training inputs. Add `--scratch` when training a new VQ-VAE from zero instead of resuming from the baseline checkpoint.

On the current AutoDL layout, the shortest scratch path is:

    cd /root/autodl-tmp/workplace
    bash tools/autodl_vqvae_scratch.sh preflight
    bash tools/autodl_vqvae_scratch.sh verify-preflight
    bash tools/autodl_vqvae_scratch.sh diagnose
    bash tools/autodl_vqvae_scratch.sh verify-diagnose
    bash tools/autodl_vqvae_scratch.sh smoke
    bash tools/autodl_vqvae_scratch.sh verify-smoke
    bash tools/autodl_vqvae_scratch.sh gate
    bash tools/autodl_vqvae_scratch.sh full --no-tail
    bash tools/autodl_vqvae_scratch.sh status

Run `bash tools/autodl_vqvae_scratch.sh repair` only if `diagnose` reports missing modules or a CUDA kernel incompatibility.

## Do Not Delete Casually

Do not delete these without a verified replacement:

    C:\V13_abc_parsed_shards
    ABC/processed/abc_parsed_full_archives
    ABC/processed/train_outputs
    local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6
    papers/aaai_v13

Generated caches and temporary render checks are safe to regenerate and are ignored by git.
