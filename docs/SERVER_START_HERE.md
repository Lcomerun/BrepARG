# V13 Server Start Here

This is the shortest server restart guide for the cleaned V13 workspace. It assumes a Linux GPU server, a repository root at `/workspace/V13`, and data under `/workspace/ABC/processed`.

On AutoDL, the equivalent paths used in the July 2026 restart are:

    repo root: /root/autodl-tmp/workplace
    parsed shards: /root/autodl-tmp/workplace/V13_abc_parsed_shards
    patch shards: /root/autodl-tmp/ABC/processed/vqvae_patch_shards
    train outputs: /root/autodl-tmp/ABC/processed/train_outputs
    conda env: /root/autodl-tmp/conda_envs/breparg

## 1. Upload The Lightweight Project Package

On Windows, build the source package:

    cd D:\luolin\V13
    powershell -ExecutionPolicy Bypass -File tools/build_server_package.ps1

Upload the generated zip from `dist/` to the server, then unpack it:

    mkdir -p /workspace/V13
    sha256sum -c v13_server_ready_20260710.zip.sha256
    unzip v13_server_ready_20260710.zip -d /workspace/V13
    cd /workspace/V13

On the current AutoDL layout, if an older package is already unpacked, overwrite only the lightweight code and docs:

    cd /root/autodl-tmp/workplace
    sha256sum -c v13_server_ready_20260710.zip.sha256
    unzip -o v13_server_ready_20260710.zip -d /root/autodl-tmp/workplace

This does not delete the uploaded parsed shards or the built VQ patch shards.

The package intentionally excludes large local data and checkpoints. Those are transferred separately.

## 2. Upload Data And Checkpoints

Upload the completed parsed shards:

    C:\V13_abc_parsed_shards

to:

    /workspace/ABC/processed/abc_parsed_shards

For 100 chunks, the server directory should contain about 101 files: 100 `parsed_abc_*.pkl.zst` files and `_manifest.jsonl`.

Upload the baseline VQ-VAE checkpoint directory if it is not already on the server:

    D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100

to:

    /workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100

Upload the current AR branch only when you plan to run sequence rebuild, AR training, or generated reconstruction:

    D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6

to:

    /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6

## 3. Create Or Check The Environment

For VQ-VAE training, bootstrap in VQ-only mode. OCC is intentionally optional here.

    cd /workspace/V13
    bash tools/server_bootstrap.sh

The bootstrap script uses `environment.server.yml` when possible, then checks the required Python modules. The important VQ-VAE imports are:

    torch
    numpy
    transformers
    diffusers
    einops
    zstandard

For AutoDL RTX 5090 / Blackwell, first run a no-install diagnostic with a real CUDA kernel smoke test:

    cd /root/autodl-tmp/workplace
    REPO_ROOT=/root/autodl-tmp/workplace \
      PYTHON=/root/autodl-tmp/conda_envs/breparg/bin/python \
      V13_SKIP_INSTALL=1 \
      V13_REQUIRE_CUDA=1 \
      bash tools/server_bootstrap.sh

If that reports missing Python modules or a CUDA kernel incompatibility, repair with the official PyTorch CUDA 12.8 wheels:

    cd /root/autodl-tmp/workplace
    REPO_ROOT=/root/autodl-tmp/workplace \
      PYTHON=/root/autodl-tmp/conda_envs/breparg/bin/python \
      V13_FORCE_CU128=1 \
      V13_REQUIRE_CUDA=1 \
      bash tools/server_bootstrap.sh

The same AutoDL diagnose/repair steps are wrapped by:

    cd /root/autodl-tmp/workplace
    bash tools/autodl_vqvae_scratch.sh preflight
    bash tools/autodl_vqvae_scratch.sh verify-preflight
    bash tools/autodl_vqvae_scratch.sh diagnose
    bash tools/autodl_vqvae_scratch.sh verify-diagnose
    bash tools/autodl_vqvae_scratch.sh repair

Run `preflight` first to check paths, disk, and patch-shard summary. Use `repair` only after `diagnose` or `verify-diagnose` fails. After `repair`, rerun `diagnose` and `verify-diagnose` before starting smoke training.

If the environment does not exist yet, let the script create it on the data disk:

    cd /root/autodl-tmp/workplace
    REPO_ROOT=/root/autodl-tmp/workplace \
      ENV_PREFIX=/root/autodl-tmp/conda_envs/breparg \
      V13_FORCE_CU128=1 \
      V13_REQUIRE_CUDA=1 \
      bash tools/server_bootstrap.sh

Expected output includes:

    diffusers.VQModel: <class ...>
    torch cuda: 12.8
    device: NVIDIA GeForce RTX 5090
    capability: (12, 0)
    cuda matmul ok: ...

If `OCC.Core.TopoDS` is missing, do not block VQ-VAE training. Install an OpenCascade Python binding only before STEP reconstruction or server readiness gates, and then run with `V13_BOOTSTRAP_MODE=full`.

## 4. Verify Parsed Shards

From `/workspace/V13`:

    python tools/verify_parsed_shards.py /workspace/ABC/processed/abc_parsed_shards/parsed_abc_*.pkl.zst \
      --output /workspace/V13/local_reports/v13_parsed_shards_verify_server.json

Expected JSON field:

    "status": "VERIFIED"

If this fails, do not start training. Re-upload the missing or corrupt shard and rerun the verifier.

## 5. Build Patch Shards And Start VQ-VAE

The resume-from-baseline server command is:

    cd /workspace/V13
    bash tools/run_vqvae_from_patch_shards.sh --build-patch-shards --run-train

Defaults:

    parsed shard root: /workspace/ABC/processed/abc_parsed_shards
    patch shard root: /workspace/ABC/processed/vqvae_patch_shards
    train output root: /workspace/ABC/processed/train_outputs
    run name: newscheme_vqvae_sharded_recovery
    resume checkpoint: /workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt

The training log should include:

    VQ patch-shard sampling selected=...

That line proves the trainer is reading patch shards rather than scanning extracted parsed `.pkl` files.

For scratch VQ-VAE training on AutoDL after patch shards already exist, first run a 1-epoch smoke test:

    cd /root/autodl-tmp/workplace
    bash tools/autodl_vqvae_scratch.sh smoke
    bash tools/autodl_vqvae_scratch.sh verify-smoke
    bash tools/autodl_vqvae_scratch.sh gate

The helper writes the pid and log under `local_reports/`. The expanded command is:

    cd /root/autodl-tmp/workplace
    PY=/root/autodl-tmp/conda_envs/breparg/bin/python
    bash tools/run_vqvae_from_patch_shards.sh \
      --repo-root /root/autodl-tmp/workplace \
      --python "$PY" \
      --patch-shard-root /root/autodl-tmp/ABC/processed/vqvae_patch_shards \
      --outbase /root/autodl-tmp/ABC/processed/train_outputs \
      --run-train \
      --scratch \
      --run-name newscheme_vqvae_5090_scratch_smoke \
      --samples 2048 \
      --epochs 1 \
      --target-epoch 1 \
      --lr 1e-4

Then start the full scratch run:

    cd /root/autodl-tmp/workplace
    bash tools/autodl_vqvae_scratch.sh verify-smoke
    bash tools/autodl_vqvae_scratch.sh gate
    bash tools/autodl_vqvae_scratch.sh full --no-tail
    bash tools/autodl_vqvae_scratch.sh status

The expanded full-run command is:

    cd /root/autodl-tmp/workplace
    PY=/root/autodl-tmp/conda_envs/breparg/bin/python
    nohup bash tools/run_vqvae_from_patch_shards.sh \
      --repo-root /root/autodl-tmp/workplace \
      --python "$PY" \
      --patch-shard-root /root/autodl-tmp/ABC/processed/vqvae_patch_shards \
      --outbase /root/autodl-tmp/ABC/processed/train_outputs \
      --run-train \
      --scratch \
      --run-name newscheme_vqvae_5090_scratch_20260710 \
      --samples 300000 \
      --epochs 160 \
      --target-epoch 160 \
      --batch-size 128 \
      --lr 1e-4 \
      > /root/autodl-tmp/workplace/local_reports/vqvae_scratch_5090_20260710.log 2>&1 &

## 6. Useful Variants

Build patch shards only:

    bash tools/run_vqvae_from_patch_shards.sh --build-patch-shards

Train from existing patch shards only:

    bash tools/run_vqvae_from_patch_shards.sh --run-train

Use a different run name:

    bash tools/run_vqvae_from_patch_shards.sh --build-patch-shards --run-train --run-name newscheme_vqvae_server_001

Use a different GPU:

    CUDA_VISIBLE_DEVICES=1 bash tools/run_vqvae_from_patch_shards.sh --run-train

Increase the patch budget:

    bash tools/run_vqvae_from_patch_shards.sh --run-train --samples 450000 --epochs 120 --batch-size 128

## 7. After VQ-VAE Training

Do not train AR until the VQ-VAE checkpoint is promoted by the reconstruction benchmark. The next guarded steps are:

    bash tools/run_vqvae_complex_recovery.sh --run-benchmark
    python tools/monitor_vqvae_recovery_gate.py ...
    python tools/verify_vqvae_copyback.py ...
    bash tools/run_source_path_sequence_rebuild.sh ...
    bash tools/run_ar_v13_long_context.sh --max-seq-len 1536 ...

Use `docs/v13_sharded_dataset_operator_guide.md` for the longer data-shard workflow and `plans/v13_generation_quality_recovery_execplan.md` for the broader recovery sequence.

## 8. Stop Conditions

Stop and inspect before renting more GPU time if any of these occur:

    parsed shard verification is not VERIFIED
    patch shard summary has zero patches
    Torch cannot see CUDA when training is about to start
    the CUDA kernel smoke test fails on RTX 5090 / Blackwell
    the largest GPU has less than 32 GB VRAM for VQ-VAE scratch training
    the largest GPU has less than 40-48 GB VRAM for later AR long-context runs
    the training log does not mention VQ patch-shard sampling
    the VQ-VAE benchmark decision is hold_vqvae_checkpoint

The current project policy is VQ-VAE recovery first, source-path sequence rebuild second, AR1536/AR2048 only after sequence verification, and positive paper figures only after generated-quality gates plus human visual review.
