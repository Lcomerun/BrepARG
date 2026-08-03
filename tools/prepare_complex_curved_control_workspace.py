"""Prepare a reproducible workspace for complex-curved FSQ/AR experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_runs" / "complex_curved_control_suite_20260715"
DEFAULT_PYTHON = r"C:\Users\YU\.conda\envs\brepgen_env\python.exe"
DEFAULT_SEQUENCE = REPO_ROOT / "ABC" / "processed" / "train_outputs" / "ubuntu" / "sequences_fsq_rcm.pkl"
DEFAULT_VQVAE = REPO_ROOT / "ABC" / "processed" / "train_outputs" / "ubuntu" / "fsq_vqvae_best.pt"
DEFAULT_AR = REPO_ROOT / "ABC" / "processed" / "train_outputs" / "ubuntu" / "ar_best.pt"
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "ABC" / "processed" / "abc_parsed_full_archives"


def ps_path(path: str | Path) -> str:
    return str(path).replace("/", "\\")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def workspace_config(
    output_dir: Path,
    python_exe: str,
    sequence_path: Path,
    vqvae_checkpoint: Path,
    ar_checkpoint: Path,
    archive_root: Path,
) -> dict[str, Any]:
    return {
        "created_by": "tools/prepare_complex_curved_control_workspace.py",
        "workspace_version": 1,
        "output_dir": str(output_dir),
        "python_exe": str(python_exe),
        "sequence_path": str(sequence_path),
        "vqvae_checkpoint": str(vqvae_checkpoint),
        "ar_checkpoint": str(ar_checkpoint),
        "archive_root": str(archive_root),
        "official_breparg_hf_repo": "qingtiannihao/BrepARG",
        "official_breparg_files": [
            "checkpoint/weights/abc_ar.pt",
            "checkpoint/weights/abc_vqvae.pt",
        ],
        "complex_subset": {
            "split": "val",
            "max_samples": 50,
            "max_scan": 5000,
            "max_seq_len": 2048,
            "complex_min_faces": 12,
            "complex_min_edges": 20,
            "curved_threshold": 0.02,
            "max_source_faces": 50,
            "max_source_edges": 150,
        },
        "experiments": {
            "00_current_fsq_ar_teacher_reconstruction": "Current FSQ-only metrics, AR teacher-forcing CE, and true-token STEP reconstruction on the complex-curved subset.",
            "01a_train_fsq_capacity_candidate": "Train a higher-capacity FSQ VQ-VAE candidate by changing only FSQ levels first.",
            "01_fsq_capacity_candidate": "Evaluate a new FSQ checkpoint with identical subset selection while skipping incompatible AR/reconstruction stages.",
            "02_dfs_rcm_ordering_rebuild": "Rebuild DFS and matching RCM sequence packages so ordering is the only changed variable.",
            "02a_prepare_v13_same_data_split": "Materialize parsed pkl paths from current sequence provenance for DFS/RCM rebuilding.",
            "02_smoke_dfs_rcm_ordering_rebuild": "Small DFS/RCM rebuild smoke using the same provenance path before launching the full ordering control.",
            "02_medium_dfs_rcm_ordering_rebuild": "Disk-safe medium DFS/RCM rebuild using the already materialized same-data BrepARG input pool.",
            "02b_smoke_dfs_rcm_ar_medium_safe": "Tiny 1-epoch smoke proving train.py can train/save matched DFS/RCM AR branches from medium sequences.",
            "02b_train_dfs_rcm_ar_medium_safe": "Local-safe short matched AR training on medium DFS/RCM sequence packages.",
            "02b_train_dfs_rcm_ar": "Train matched DFS and RCM AR branches with identical hyperparameters.",
            "02c_eval_dfs_rcm_ar_complex_curved": "Evaluate matched DFS and RCM AR checkpoints on the complex-curved subset with teacher-forcing CE.",
            "03_breparg_official_baseline": "Try official BrepARG ABC weights first; if incompatible, use this folder to document same-data BrepARG retraining.",
            "03a_prepare_breparg_same_data_inputs": "Materialize the current V13 sequence provenance into original BrepARG split and SE VQ-VAE source files.",
            "03b_breparg_same_data_training_fallback": "Fallback BrepARG self-training path using the same split and shared audit protocol when official weights are incompatible.",
            "06_prepare_external_ssd_migration": "Dry-run or execute migration of this root-cause suite to an external SSD without deleting local files.",
        },
    }


def readme_text(config: dict[str, Any]) -> str:
    return f"""
# Complex Curved Control Suite

This folder is a portable control workspace for diagnosing why generated CAD
results collapse toward simple shapes. It keeps the experiments separate from
the existing training outputs so the whole directory can later be copied to an
external SSD.

## What This Suite Measures

The core subset is validation records that are both complex and curved:
`faces >= {config["complex_subset"]["complex_min_faces"]}` or
`edges >= {config["complex_subset"]["complex_min_edges"]}`, with parsed patch
curvature above the configured threshold. The same subset definition is reused
so one experiment changes one variable at a time.

The first script runs three read-only diagnostics on the current method:
FSQ patch MSE/Chamfer, AR teacher-forcing cross entropy on true tokens, and
true-token reconstruction through FSQ/OCC to STEP. This directly answers
whether the issue is already present before free-running AR generation.

The second script is for a new FSQ capacity checkpoint. It intentionally uses
`--skip-ar --skip-reconstruction` because changing FSQ levels changes the token
vocabulary; old AR checkpoints and old token reconstruction are no longer a
fair match.

The `01a` script is the long-running training launcher for such a candidate. It
changes FSQ levels first, because latent-channel changes require code changes
and should be a separate experiment.

The third script prepares DFS versus RCM sequence rebuilding. Train/evaluate a
matching AR for each package before comparing generation quality.

The fourth script tries the official BrepARG ABC weights from Hugging Face
(`qingtiannihao/BrepARG`) before any local retraining. If official weights do
not align with the local protocol, keep the failure log and train a medium
same-data BrepARG baseline in this workspace.

## Run Order

1. Run `scripts\\00_current_fsq_ar_teacher_reconstruction.ps1`.
2. Run `scripts\\01a_preflight_fsq_capacity_candidate.ps1` to verify the full
   patch shards and training entrypoint without starting training.
3. Edit and run `scripts\\01a_train_fsq_capacity_candidate.ps1` when the GPU
   machine has the patch shards or parsed pool available. If that run is
   interrupted after writing `fsq_vqvae_best.pt` and `vqvae_history.json`, use
   `scripts\\01a_resume_fsq_capacity_candidate.ps1` instead of restarting from
   scratch.
4. Train or copy in a completed capacity FSQ checkpoint, then edit and run
   `scripts\\01_fsq_capacity_candidate.ps1`.
5. Run `scripts\\02_smoke_dfs_rcm_ordering_rebuild.ps1` for a 5/3/3
   same-data split and DFS/RCM rebuild smoke.
6. On the temporary local disk, run `scripts\\02_medium_dfs_rcm_ordering_rebuild.ps1`
   after the medium BrepARG same-data inputs exist. This reuses that parsed pool
   without duplicating a full split.
7. On the external SSD or server, run `scripts\\02a_prepare_v13_same_data_split.ps1`, then run
   `scripts\\02_dfs_rcm_ordering_rebuild.ps1`.
8. Run `scripts\\02b_smoke_dfs_rcm_ar_medium_safe.ps1` to prove the medium
   DFS/RCM AR training and checkpoint path before a longer local run.
9. For a local medium ordering diagnostic, run
   `scripts\\02b_train_dfs_rcm_ar_medium_safe.ps1`.
10. For the full ordering control after full sequences exist, run
   `scripts\\02b_train_dfs_rcm_ar.ps1`.
11. Run `scripts\\02c_eval_dfs_rcm_ar_complex_curved.ps1` to compare teacher-forcing CE on the complex-curved subset.
12. Run `scripts\\03_breparg_official_baseline.ps1` when network/disk are ready.
13. If official weights cannot be used, run the medium same-data input prep
   `scripts\\03a_prepare_breparg_same_data_inputs.ps1`.
14. If the external SSD is available and you want a larger baseline input pool,
   run `scripts\\03a_prepare_breparg_same_data_inputs_full.ps1` and then copy
   or point the fallback script at that data root.
15. Run `scripts\\03b_preflight_breparg_same_data_fallback.ps1` to verify
   fallback dependencies, inputs, and CLI compatibility without starting
   training.
16. Then edit and run
   `scripts\\03b_breparg_same_data_training_fallback.ps1`.
17. Run `scripts\\04_summarize_reports.ps1` after the comparable reports exist.
18. Run `scripts\\05_audit_suite_status.ps1` anytime to see completed/missing artifacts and next actions.
19. When the external SSD is connected, run
    `scripts\\06_prepare_external_ssd_migration.ps1 -DestRoot E:\\V13_rootcause_20260715`
    first as a dry-run, then add `-Execute` after reviewing the manifest.

## Current Inputs

- Sequence: `{config["sequence_path"]}`
- FSQ VQ-VAE: `{config["vqvae_checkpoint"]}`
- AR checkpoint: `{config["ar_checkpoint"]}`
- Parsed archives: `{config["archive_root"]}`
"""


def script_00(config: dict[str, Any]) -> str:
    out = Path(config["output_dir"]) / "experiments" / "00_current_fsq_ar_teacher_reconstruction"
    subset = config["complex_subset"]
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$OUT = "{ps_path(out)}"
New-Item -ItemType Directory -Force $OUT | Out-Null

& $PY tools\\complex_curved_diagnostics.py `
  --sequence "{ps_path(config["sequence_path"])}" `
  --vqvae-checkpoint "{ps_path(config["vqvae_checkpoint"])}" `
  --ar-checkpoint "{ps_path(config["ar_checkpoint"])}" `
  --archive-root "{ps_path(config["archive_root"])}" `
  --output-dir $OUT `
  --split {subset["split"]} `
  --max-samples {subset["max_samples"]} `
  --max-scan {subset["max_scan"]} `
  --max-seq-len {subset["max_seq_len"]} `
  --complex-min-faces {subset["complex_min_faces"]} `
  --complex-min-edges {subset["complex_min_edges"]} `
  --curved-threshold {subset["curved_threshold"]} `
  --max-source-faces {subset["max_source_faces"]} `
  --max-source-edges {subset["max_source_edges"]} `
  --device auto `
  --write-step `
  --validate-step
"""


def script_01(config: dict[str, Any]) -> str:
    out = Path(config["output_dir"]) / "experiments" / "01_fsq_capacity_candidate"
    expected_checkpoint = (
        Path(config["output_dir"])
        / "experiments"
        / "01a_train_fsq_capacity_candidate"
        / "fsq_levels_16_16_8_8_complex_curved_20260715"
        / "fsq_vqvae_best.pt"
    )
    subset = config["complex_subset"]
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$OUT = "{ps_path(out)}"
$EXPECTED_CAPACITY_VQVAE = "{ps_path(expected_checkpoint)}"

# Optional override for copied/server-trained capacity checkpoints.
$CAPACITY_VQVAE = if ($env:V13_CAPACITY_VQVAE) {{ $env:V13_CAPACITY_VQVAE }} else {{ $EXPECTED_CAPACITY_VQVAE }}
if (!(Test-Path $CAPACITY_VQVAE)) {{
  throw "Missing FSQ capacity checkpoint: $CAPACITY_VQVAE. Run scripts\\01a_train_fsq_capacity_candidate.ps1 first, or set env:V13_CAPACITY_VQVAE to a copied fsq_vqvae_best.pt."
}}

New-Item -ItemType Directory -Force $OUT | Out-Null

& $PY tools\\complex_curved_diagnostics.py `
  --sequence "{ps_path(config["sequence_path"])}" `
  --vqvae-checkpoint $CAPACITY_VQVAE `
  --ar-checkpoint "{ps_path(config["ar_checkpoint"])}" `
  --archive-root "{ps_path(config["archive_root"])}" `
  --output-dir $OUT `
  --split {subset["split"]} `
  --max-samples {subset["max_samples"]} `
  --max-scan {subset["max_scan"]} `
  --max-seq-len {subset["max_seq_len"]} `
  --complex-min-faces {subset["complex_min_faces"]} `
  --complex-min-edges {subset["complex_min_edges"]} `
  --curved-threshold {subset["curved_threshold"]} `
  --max-source-faces {subset["max_source_faces"]} `
  --max-source-edges {subset["max_source_edges"]} `
  --device auto `
  --skip-ar `
  --skip-reconstruction
"""


def script_01a(config: dict[str, Any]) -> str:
    workspace_root = Path(config["output_dir"])
    root = workspace_root / "experiments" / "01a_train_fsq_capacity_candidate"
    patch_shards = root / "vq_patch_shards_full"
    sample_cache = root / "vq_samples_450000_seed0.npz"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$OUTBASE = "{ps_path(root)}"
$RUN = "fsq_levels_16_16_8_8_complex_curved_20260715"

# Prefer patch shards if available. If you only have an extracted parsed pool,
# clear $PATCH_SHARD_ROOT and set $PARSED_POOL instead.
$PATCH_SHARD_ROOT = "{ps_path(patch_shards)}"
$SAMPLE_CACHE = "{ps_path(sample_cache)}"
$PARSED_POOL = "PATH\\TO\\abc_parsed_full"

New-Item -ItemType Directory -Force $OUTBASE | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$env:NS_OUTBASE = $OUTBASE
$env:NS_OUT = $RUN
$env:NS_N = "999999"

# Capacity variable under test: change FSQ levels first, keep architecture
# otherwise fixed. 16*16*8*8 = 16384 codebook entries versus current 8192.
$env:NS_LEVELS = "16,16,8,8"

$env:NS_VQ_SAMPLES = "450000"
$env:NS_VQ_SAMPLE_CACHE = $SAMPLE_CACHE
$env:NS_VQ_EPOCHS = "180"
$env:NS_VQ_TARGET_EPOCH = "180"
$env:NS_VQ_BS = "128"
$env:NS_VQ_LR = "1e-4"
$env:NS_VQ_MIN_EPOCHS = "40"
$env:NS_VQ_PATIENCE = "18"
$env:NS_VQ_MIN_DELTA = "1e-6"
$env:NS_VQ_MAX_NONFINITE_VAL_EPOCHS = "2"
$env:NS_DISABLE_AMP_VQVAE = "1"

$env:NS_VQ_COMPLEX_FRACTION = "0.50"
$env:NS_VQ_COMPLEX_MIN_FACES = "12"
$env:NS_VQ_COMPLEX_MIN_EDGES = "20"
$env:NS_VQ_CURVED_FRACTION = "0.35"
$env:NS_VQ_MAX_SOURCE_FACES = "50"
$env:NS_VQ_MAX_SOURCE_EDGES = "150"
$env:NS_VQ_COMPLEX_LOSS_WEIGHT = "1.25"
$env:NS_VQ_CURVED_LOSS_WEIGHT = "2.0"
$env:NS_VQ_CURVED_LOSS_THRESHOLD = "0.02"

if (Test-Path $PATCH_SHARD_ROOT) {{
  $env:NS_VQ_PATCH_SHARD_ROOT = $PATCH_SHARD_ROOT
}} elseif (Test-Path $PARSED_POOL) {{
  $env:NS_POOL = $PARSED_POOL
}} else {{
  throw "Set `$PATCH_SHARD_ROOT or `$PARSED_POOL to existing training data before running."
}}

& $PY breparg_improvements\\train.py --stage vqvae

Write-Host "Capacity checkpoint:"
Write-Host "  $OUTBASE\\$RUN\\fsq_vqvae_best.pt"
Write-Host "Next: edit scripts\\01_fsq_capacity_candidate.ps1 and set `$CAPACITY_VQVAE to this checkpoint."
"""


def script_01a_resume(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "01a_train_fsq_capacity_candidate"
    run = "fsq_levels_16_16_8_8_complex_curved_20260715"
    run_dir = root / run
    patch_shards = root / "vq_patch_shards_full"
    sample_cache = root / "vq_samples_450000_seed0.npz"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$OUTBASE = "{ps_path(root)}"
$RUN = "{run}"
$RUN_DIR = "{ps_path(run_dir)}"
$PATCH_SHARD_ROOT = "{ps_path(patch_shards)}"
$SAMPLE_CACHE = "{ps_path(sample_cache)}"
$RESUME_FROM = "$RUN_DIR\\fsq_vqvae_best.pt"
$HISTORY_IN = "$RUN_DIR\\vqvae_history.json"
$CHECK_REPORT = "$RUN_DIR\\fsq_capacity_completion_check.json"

if (!(Test-Path $RESUME_FROM)) {{
  throw "Missing resume checkpoint: $RESUME_FROM. Run scripts\\01a_train_fsq_capacity_candidate.ps1 first if you want a fresh start."
}}
if (!(Test-Path $HISTORY_IN)) {{
  throw "Missing resume history: $HISTORY_IN. Resume needs vqvae_history.json to continue epochs safely."
}}
if (!(Test-Path $PATCH_SHARD_ROOT)) {{
  throw "Missing patch shard root: $PATCH_SHARD_ROOT"
}}
if (!(Test-Path $SAMPLE_CACHE)) {{
  throw "Missing sample cache: $SAMPLE_CACHE. Run scripts\\01a_build_fsq_capacity_sample_cache.ps1 first."
}}

$previousPreference = $ErrorActionPreference
$nativePreferenceVar = Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue
if ($nativePreferenceVar) {{
  $previousNativePreference = $PSNativeCommandUseErrorActionPreference
  $PSNativeCommandUseErrorActionPreference = $false
}}
$ErrorActionPreference = "Continue"
try {{
  & $PY tools\\check_fsq_capacity_completion.py `
    --run-dir $RUN_DIR `
    --output $CHECK_REPORT
  $completionExit = $LASTEXITCODE
}} finally {{
  $ErrorActionPreference = $previousPreference
  if ($nativePreferenceVar) {{
    $PSNativeCommandUseErrorActionPreference = $previousNativePreference
  }}
}}
if ($completionExit -eq 0) {{
  Write-Host "FSQ capacity run is already complete according to train_report.json:"
  Write-Host "  $CHECK_REPORT"
  exit 0
}}
if ($completionExit -ne 2) {{
  throw "Completion check failed with exit code $completionExit"
}}

New-Item -ItemType Directory -Force $OUTBASE | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$env:NS_OUTBASE = $OUTBASE
$env:NS_OUT = $RUN
$env:NS_N = "999999"

$env:NS_LEVELS = "16,16,8,8"
$env:NS_VQ_SAMPLES = "450000"
$env:NS_VQ_SAMPLE_CACHE = $SAMPLE_CACHE
$env:NS_VQ_EPOCHS = "180"
$env:NS_VQ_TARGET_EPOCH = "180"
$env:NS_VQ_BS = "128"
$env:NS_VQ_LR = "1e-4"
$env:NS_VQ_MIN_EPOCHS = "40"
$env:NS_VQ_PATIENCE = "18"
$env:NS_VQ_MIN_DELTA = "1e-6"
$env:NS_VQ_MAX_NONFINITE_VAL_EPOCHS = "2"
$env:NS_DISABLE_AMP_VQVAE = "1"

$env:NS_VQ_COMPLEX_FRACTION = "0.50"
$env:NS_VQ_COMPLEX_MIN_FACES = "12"
$env:NS_VQ_COMPLEX_MIN_EDGES = "20"
$env:NS_VQ_CURVED_FRACTION = "0.35"
$env:NS_VQ_MAX_SOURCE_FACES = "50"
$env:NS_VQ_MAX_SOURCE_EDGES = "150"
$env:NS_VQ_COMPLEX_LOSS_WEIGHT = "1.25"
$env:NS_VQ_CURVED_LOSS_WEIGHT = "2.0"
$env:NS_VQ_CURVED_LOSS_THRESHOLD = "0.02"

$env:NS_VQ_PATCH_SHARD_ROOT = $PATCH_SHARD_ROOT
$env:NS_VQ_RESUME_FROM = $RESUME_FROM
$env:NS_VQ_HISTORY_IN = $HISTORY_IN

& $PY breparg_improvements\\train.py --stage vqvae

Write-Host "Resumed capacity checkpoint:"
Write-Host "  $RUN_DIR\\fsq_vqvae_best.pt"
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\01_fsq_capacity_candidate.ps1"
"""


def script_01a_watch_then_eval(config: dict[str, Any]) -> str:
    workspace_root = Path(config["output_dir"])
    root = Path(config["output_dir"]) / "experiments" / "01a_train_fsq_capacity_candidate"
    run = "fsq_levels_16_16_8_8_complex_curved_20260715"
    run_dir = root / run
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$RUN_DIR = "{ps_path(run_dir)}"
$LOG_DIR = "{ps_path(root / "logs")}"
$PID_FILE = "$LOG_DIR\\fsq_capacity_resume.pid"
$CHECK_REPORT = "$RUN_DIR\\fsq_capacity_completion_check.json"
$WATCH_LOG = "$LOG_DIR\\fsq_capacity_watch_then_eval.log"

New-Item -ItemType Directory -Force $LOG_DIR | Out-Null

function Write-WatchLog {{
  param([string]$Message)
  $line = "[{{0}}] {{1}}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  Write-Host $line
  Add-Content -Encoding UTF8 -Path $WATCH_LOG -Value $line
}}

if (!(Test-Path $PID_FILE)) {{
  throw "Missing resume PID file: $PID_FILE. Start scripts\\01a_resume_fsq_capacity_candidate.ps1 first."
}}

$ResumePid = [int](Get-Content -Raw $PID_FILE)
Write-WatchLog "watching FSQ capacity resume PID=$ResumePid"

while (Get-Process -Id $ResumePid -ErrorAction SilentlyContinue) {{
  $hist = "$RUN_DIR\\vqvae_history.json"
  if (Test-Path $hist) {{
    $j = Get-Content -Raw $hist | ConvertFrom-Json
    $last = $j.history[-1]
    Write-WatchLog ("still training: history_count={{0}} epoch={{1}} val={{2}} best={{3}}" -f $j.history.Count, $last.epoch, $last.val_loss, $j.best_val_recon)
  }} else {{
    Write-WatchLog "still training: history file not found yet"
  }}
  Start-Sleep -Seconds 300
}}

Write-WatchLog "resume PID exited; checking completion"

$previousPreference = $ErrorActionPreference
$nativePreferenceVar = Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue
if ($nativePreferenceVar) {{
  $previousNativePreference = $PSNativeCommandUseErrorActionPreference
  $PSNativeCommandUseErrorActionPreference = $false
}}
$ErrorActionPreference = "Continue"
try {{
  & $PY tools\\check_fsq_capacity_completion.py `
    --run-dir $RUN_DIR `
    --output $CHECK_REPORT
  $completionExit = $LASTEXITCODE
}} finally {{
  $ErrorActionPreference = $previousPreference
  if ($nativePreferenceVar) {{
    $PSNativeCommandUseErrorActionPreference = $previousNativePreference
  }}
}}

if ($completionExit -ne 0) {{
  Write-WatchLog "capacity training did not complete cleanly; not running eval. See $CHECK_REPORT"
  throw "FSQ capacity training incomplete; completion checker exit code $completionExit"
}}

Write-WatchLog "capacity training complete; running official FSQ-only eval"

powershell -ExecutionPolicy Bypass -File "{ps_path(workspace_root / "scripts" / "01_fsq_capacity_candidate.ps1")}"
powershell -ExecutionPolicy Bypass -File "{ps_path(workspace_root / "scripts" / "04_summarize_reports.ps1")}"
powershell -ExecutionPolicy Bypass -File "{ps_path(workspace_root / "scripts" / "05_audit_suite_status.ps1")}"

Write-WatchLog "capacity eval pipeline finished"
"""


def script_01a_preflight(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "01a_train_fsq_capacity_candidate"
    patch_shards = root / "vq_patch_shards_full"
    sample_cache = root / "vq_samples_450000_seed0.npz"
    run = "fsq_levels_16_16_8_8_complex_curved_20260715"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$OUTBASE = "{ps_path(root)}"
$PATCH_SHARD_ROOT = "{ps_path(patch_shards)}"
$SAMPLE_CACHE = "{ps_path(sample_cache)}"
$RUN = "{run}"
$OUT = "$OUTBASE\\fsq_capacity_preflight.json"

& $PY tools\\preflight_fsq_capacity_candidate.py `
  --patch-shard-root $PATCH_SHARD_ROOT `
  --outbase $OUTBASE `
  --run-name $RUN `
  --python-exe $PY `
  --train-script breparg_improvements\\train.py `
  --samples 450000 `
  --levels 16,16,8,8 `
  --sample-cache $SAMPLE_CACHE `
  --output $OUT

Write-Host "FSQ capacity preflight:"
Write-Host "  $OUT"
"""


def script_01a_sample_cache(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "01a_train_fsq_capacity_candidate"
    patch_shards = root / "vq_patch_shards_full"
    sample_cache = root / "vq_samples_450000_seed0.npz"
    summary = root / "vq_samples_450000_seed0_summary.json"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$PATCH_SHARD_ROOT = "{ps_path(patch_shards)}"
$SAMPLE_CACHE = "{ps_path(sample_cache)}"
$SUMMARY = "{ps_path(summary)}"

if (!(Test-Path $PATCH_SHARD_ROOT)) {{
  throw "Missing patch shard root: $PATCH_SHARD_ROOT. Run scripts\\01a_build_fsq_capacity_patch_shards_full.ps1 first."
}}

if (Test-Path $SAMPLE_CACHE) {{
  Write-Host "Sample cache already exists:"
  Write-Host "  $SAMPLE_CACHE"
  Write-Host "Use tools\\build_vqvae_sample_cache.py --force manually if you intentionally want to rebuild it."
  exit 0
}}

& $PY tools\\build_vqvae_sample_cache.py `
  --patch-shard-root $PATCH_SHARD_ROOT `
  --output $SAMPLE_CACHE `
  --summary-output $SUMMARY `
  --samples 450000 `
  --seed 0 `
  --complex-fraction 0.50 `
  --complex-min-faces 12 `
  --complex-min-edges 20 `
  --curved-fraction 0.35 `
  --max-source-faces 50 `
  --max-source-edges 150 `
  --complex-loss-weight 1.25 `
  --curved-loss-weight 2.0 `
  --curved-loss-threshold 0.02

Write-Host "FSQ capacity sample cache:"
Write-Host "  $SAMPLE_CACHE"
Write-Host "Summary:"
Write-Host "  $SUMMARY"
"""


def script_02(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "02_dfs_rcm_ordering"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$SPLIT = "$ROOT\\same_data_split\\split.pkl"
$FSQ = "{ps_path(config["vqvae_checkpoint"])}"
if (!(Test-Path $SPLIT)) {{
  throw "Missing `$SPLIT. Run scripts\\02a_prepare_v13_same_data_split.ps1 first."
}}
if (!(Test-Path $FSQ)) {{
  throw "Set `$FSQ to a real FSQ checkpoint."
}}

New-Item -ItemType Directory -Force $ROOT | Out-Null

& $PY tools\\run_sharded_sequence.py `
  --split $SPLIT `
  --checkpoint $FSQ `
  --shard-dir "$ROOT\\sequence_dfs_shards" `
  --merge-output "$ROOT\\sequences_fsq_dfs.pkl" `
  --summary "$ROOT\\sequences_fsq_dfs_summary.json" `
  --manifest "$ROOT\\sequences_fsq_dfs_manifest.jsonl" `
  --workers 8 `
  --chunks 0-99 `
  --resume `
  --ordering dfs

& $PY tools\\run_sharded_sequence.py `
  --split $SPLIT `
  --checkpoint $FSQ `
  --shard-dir "$ROOT\\sequence_rcm_shards" `
  --merge-output "$ROOT\\sequences_fsq_rcm.pkl" `
  --summary "$ROOT\\sequences_fsq_rcm_summary.json" `
  --manifest "$ROOT\\sequences_fsq_rcm_manifest.jsonl" `
  --workers 8 `
  --chunks 0-99 `
  --resume `
  --ordering rcm
"""


def script_02a(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "02_dfs_rcm_ordering"
    data = root / "same_data_split"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$DATA = "{ps_path(data)}"
New-Item -ItemType Directory -Force $DATA | Out-Null

# This materializes parsed .pkl paths from the current V13 sequence package so
# DFS and RCM can be rebuilt from exactly the same source records.
& $PY tools\\prepare_v13_same_data_split.py `
  --sequence "{ps_path(config["sequence_path"])}" `
  --archive-root "{ps_path(config["archive_root"])}" `
  --output-dir $DATA `
  --train-limit 50000 `
  --val-limit 5000 `
  --test-limit 5000 `
  --max-faces 50 `
  --max-edges 150

if (!(Test-Path "$DATA\\split.pkl")) {{
  throw "split.pkl was not created under $DATA"
}}
if (!(Test-Path "$DATA\\v13_same_data_split_summary.json")) {{
  throw "v13_same_data_split_summary.json was not created under $DATA"
}}

Write-Host "Prepared V13 same-data split:"
Write-Host "  $DATA\\split.pkl"
Write-Host "  $DATA\\v13_same_data_split_summary.json"
"""


def script_02_smoke(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "02_dfs_rcm_ordering"
    split_data = root / "same_data_split_smoke"
    smoke = root / "sequence_rebuild_smoke"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$SPLIT_DATA = "{ps_path(split_data)}"
$SMOKE = "{ps_path(smoke)}"
$FSQ = "{ps_path(config["vqvae_checkpoint"])}"

New-Item -ItemType Directory -Force $SPLIT_DATA | Out-Null
New-Item -ItemType Directory -Force $SMOKE | Out-Null

& $PY tools\\prepare_v13_same_data_split.py `
  --sequence "{ps_path(config["sequence_path"])}" `
  --archive-root "{ps_path(config["archive_root"])}" `
  --output-dir $SPLIT_DATA `
  --train-limit 5 `
  --val-limit 3 `
  --test-limit 3 `
  --max-faces 50 `
  --max-edges 150

if (!(Test-Path "$SPLIT_DATA\\split.pkl")) {{
  throw "split.pkl was not created under $SPLIT_DATA"
}}

& $PY tools\\run_sharded_sequence.py `
  --split "$SPLIT_DATA\\split.pkl" `
  --checkpoint $FSQ `
  --shard-dir "$SMOKE\\sequence_dfs_shards" `
  --merge-output "$SMOKE\\sequences_fsq_dfs.pkl" `
  --summary "$SMOKE\\sequences_fsq_dfs_summary.json" `
  --manifest "$SMOKE\\sequences_fsq_dfs_manifest.jsonl" `
  --workers 1 `
  --chunks 0-0 `
  --resume `
  --ordering dfs

& $PY tools\\run_sharded_sequence.py `
  --split "$SPLIT_DATA\\split.pkl" `
  --checkpoint $FSQ `
  --shard-dir "$SMOKE\\sequence_rcm_shards" `
  --merge-output "$SMOKE\\sequences_fsq_rcm.pkl" `
  --summary "$SMOKE\\sequences_fsq_rcm_summary.json" `
  --manifest "$SMOKE\\sequences_fsq_rcm_manifest.jsonl" `
  --workers 1 `
  --chunks 0-0 `
  --resume `
  --ordering rcm

Write-Host "DFS/RCM ordering smoke outputs:"
Write-Host "  $SPLIT_DATA\\v13_same_data_split_summary.json"
Write-Host "  $SMOKE\\sequences_fsq_dfs_summary.json"
Write-Host "  $SMOKE\\sequences_fsq_rcm_summary.json"
"""


def script_02_medium(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "02_dfs_rcm_ordering"
    fallback_data = Path(config["output_dir"]) / "experiments" / "03b_breparg_same_data_training_fallback" / "data"
    medium = root / "sequence_rebuild_medium"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$MEDIUM = "{ps_path(medium)}"
$SPLIT = "{ps_path(fallback_data / "same_data_split.pkl")}"
$FSQ = "{ps_path(config["vqvae_checkpoint"])}"
$WORKERS = if ($env:V13_ORDERING_WORKERS) {{ $env:V13_ORDERING_WORKERS }} else {{ "1" }}

if (!(Test-Path $SPLIT)) {{
  throw "Missing medium same-data split. Run scripts\\03a_prepare_breparg_same_data_inputs.ps1 first."
}}
if (!(Test-Path $FSQ)) {{
  throw "Set `$FSQ to a real FSQ checkpoint."
}}

New-Item -ItemType Directory -Force $MEDIUM | Out-Null

# Disk-safe local ordering control: reuses the already materialized medium
# same-data parsed pool instead of duplicating a full 50k/5k/5k split on D:.
& $PY tools\\run_sharded_sequence.py `
  --split $SPLIT `
  --checkpoint $FSQ `
  --shard-dir "$MEDIUM\\sequence_dfs_shards" `
  --merge-output "$MEDIUM\\sequences_fsq_dfs.pkl" `
  --summary "$MEDIUM\\sequences_fsq_dfs_summary.json" `
  --manifest "$MEDIUM\\sequences_fsq_dfs_manifest.jsonl" `
  --workers $WORKERS `
  --chunks 0-99 `
  --resume `
  --ordering dfs

& $PY tools\\run_sharded_sequence.py `
  --split $SPLIT `
  --checkpoint $FSQ `
  --shard-dir "$MEDIUM\\sequence_rcm_shards" `
  --merge-output "$MEDIUM\\sequences_fsq_rcm.pkl" `
  --summary "$MEDIUM\\sequences_fsq_rcm_summary.json" `
  --manifest "$MEDIUM\\sequences_fsq_rcm_manifest.jsonl" `
  --workers $WORKERS `
  --chunks 0-99 `
  --resume `
  --ordering rcm

Write-Host "Medium DFS/RCM ordering outputs:"
Write-Host "  $MEDIUM\\sequences_fsq_dfs.pkl"
Write-Host "  $MEDIUM\\sequences_fsq_rcm.pkl"
"""


def script_02b(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "02_dfs_rcm_ordering"
    outbase = root / "ar_train_outputs"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$OUTBASE = "{ps_path(outbase)}"
$DFS_SEQUENCE = "$ROOT\\sequences_fsq_dfs.pkl"
$RCM_SEQUENCE = "$ROOT\\sequences_fsq_rcm.pkl"
$ORDERING_SOURCE = "full"

if (!(Test-Path $DFS_SEQUENCE) -or !(Test-Path $RCM_SEQUENCE)) {{
  $MEDIUM_DFS = "$ROOT\\sequence_rebuild_medium\\sequences_fsq_dfs.pkl"
  $MEDIUM_RCM = "$ROOT\\sequence_rebuild_medium\\sequences_fsq_rcm.pkl"
  if ((Test-Path $MEDIUM_DFS) -and (Test-Path $MEDIUM_RCM)) {{
    $DFS_SEQUENCE = $MEDIUM_DFS
    $RCM_SEQUENCE = $MEDIUM_RCM
    $ORDERING_SOURCE = "medium"
  }}
}}

if (!(Test-Path $DFS_SEQUENCE)) {{
  throw "Missing DFS sequence package. Run scripts\\02_medium_dfs_rcm_ordering_rebuild.ps1 locally or scripts\\02_dfs_rcm_ordering_rebuild.ps1 on SSD/server first."
}}
if (!(Test-Path $RCM_SEQUENCE)) {{
  throw "Missing RCM sequence package. Run scripts\\02_medium_dfs_rcm_ordering_rebuild.ps1 locally or scripts\\02_dfs_rcm_ordering_rebuild.ps1 on SSD/server first."
}}

New-Item -ItemType Directory -Force $OUTBASE | Out-Null
Write-Host "Ordering sequence source: $ORDERING_SOURCE"
Write-Host "  DFS: $DFS_SEQUENCE"
Write-Host "  RCM: $RCM_SEQUENCE"

# Keep these identical for DFS and RCM. Adjust only if both branches are changed
# together; otherwise the ordering variable is no longer isolated.
$AR_EPOCHS = "60"
$AR_BS = "24"
$AR_DMODEL = "256"
$AR_LAYERS = "8"
$AR_LR = "5e-5"
$AR_MAX_SEQ_LEN = "2048"
$AR_SAVE_EVERY = "5"
$AR_LOG_EVERY_BATCHES = "500"

function Train-OrderingAr {{
  param(
    [string]$Name,
    [string]$SequencePath
  )

  $RUN = "ar_$Name`_matched_20260715"
  $RUN_DIR = "$OUTBASE\\$RUN"
  New-Item -ItemType Directory -Force $RUN_DIR | Out-Null

  # train.py expects this filename inside NS_OUTBASE/NS_OUT.
  Copy-Item -Force $SequencePath "$RUN_DIR\\sequences_fsq_rcm.pkl"

  & $PY tools\\preflight_ar_training.py `
    --sequence "$RUN_DIR\\sequences_fsq_rcm.pkl" `
    --output "$RUN_DIR\\ar_preflight.json" `
    --max-seq-len $AR_MAX_SEQ_LEN `
    --batch-size 4 `
    --d-model $AR_DMODEL `
    --layers $AR_LAYERS `
    --max-samples 128

  $env:PYTHONUTF8 = "1"
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONUNBUFFERED = "1"
  $env:CUDA_VISIBLE_DEVICES = "0"
  $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
  $env:NS_OUTBASE = $OUTBASE
  $env:NS_OUT = $RUN
  $env:NS_N = "999999"
  $env:NS_AR_EPOCHS = $AR_EPOCHS
  $env:NS_AR_BS = $AR_BS
  $env:NS_AR_DMODEL = $AR_DMODEL
  $env:NS_AR_LAYERS = $AR_LAYERS
  $env:NS_AR_LR = $AR_LR
  $env:NS_AR_MAX_SEQ_LEN = $AR_MAX_SEQ_LEN
  $env:NS_AR_SAVE_EVERY = $AR_SAVE_EVERY
  $env:NS_AR_LOG_EVERY_BATCHES = $AR_LOG_EVERY_BATCHES
  Remove-Item Env:\\NS_AR_RESUME_FROM -ErrorAction SilentlyContinue

  & $PY breparg_improvements\\train.py --stage ar 2>&1 | Tee-Object -FilePath "$RUN_DIR\\ar_train.log"
  $REPORT_SRC = "breparg_improvements\\repro_outputs\\$RUN\\train_report.json"
  if (Test-Path $REPORT_SRC) {{
    Copy-Item -Force $REPORT_SRC "$RUN_DIR\\train_report.json"
  }}
}}

Train-OrderingAr -Name "dfs" -SequencePath $DFS_SEQUENCE
Train-OrderingAr -Name "rcm" -SequencePath $RCM_SEQUENCE

Write-Host "Matched AR branches:"
Write-Host "  DFS: $OUTBASE\\ar_dfs_matched_20260715"
Write-Host "  RCM: $OUTBASE\\ar_rcm_matched_20260715"
"""


def script_02b_medium_safe(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "02_dfs_rcm_ordering"
    outbase = root / "ar_train_outputs"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$OUTBASE = "{ps_path(outbase)}"
$DFS_SEQUENCE = "$ROOT\\sequence_rebuild_medium\\sequences_fsq_dfs.pkl"
$RCM_SEQUENCE = "$ROOT\\sequence_rebuild_medium\\sequences_fsq_rcm.pkl"

if (!(Test-Path $DFS_SEQUENCE)) {{
  throw "Missing medium DFS sequence package. Run scripts\\02_medium_dfs_rcm_ordering_rebuild.ps1 first."
}}
if (!(Test-Path $RCM_SEQUENCE)) {{
  throw "Missing medium RCM sequence package. Run scripts\\02_medium_dfs_rcm_ordering_rebuild.ps1 first."
}}

New-Item -ItemType Directory -Force $OUTBASE | Out-Null

# Local-safe short matched AR run. It is useful for ordering diagnostics on the
# temporary local disk, but the full paper/control result still needs the full
# sequence rebuild and a longer training budget on SSD/server.
$AR_EPOCHS = if ($env:V13_MEDIUM_AR_EPOCHS) {{ $env:V13_MEDIUM_AR_EPOCHS }} else {{ "5" }}
$AR_BS = if ($env:V13_MEDIUM_AR_BS) {{ $env:V13_MEDIUM_AR_BS }} else {{ "4" }}
$AR_DMODEL = if ($env:V13_MEDIUM_AR_DMODEL) {{ $env:V13_MEDIUM_AR_DMODEL }} else {{ "256" }}
$AR_LAYERS = if ($env:V13_MEDIUM_AR_LAYERS) {{ $env:V13_MEDIUM_AR_LAYERS }} else {{ "8" }}
$AR_LR = if ($env:V13_MEDIUM_AR_LR) {{ $env:V13_MEDIUM_AR_LR }} else {{ "5e-5" }}
$AR_MAX_SEQ_LEN = if ($env:V13_MEDIUM_AR_MAX_SEQ_LEN) {{ $env:V13_MEDIUM_AR_MAX_SEQ_LEN }} else {{ "2048" }}
$AR_SAVE_EVERY = if ($env:V13_MEDIUM_AR_SAVE_EVERY) {{ $env:V13_MEDIUM_AR_SAVE_EVERY }} else {{ "5" }}
$AR_LOG_EVERY_BATCHES = if ($env:V13_MEDIUM_AR_LOG_EVERY_BATCHES) {{ $env:V13_MEDIUM_AR_LOG_EVERY_BATCHES }} else {{ "500" }}

function Train-MediumOrderingAr {{
  param(
    [string]$Name,
    [string]$SequencePath
  )

  $RUN = "ar_$Name`_medium_safe_20260715"
  $RUN_DIR = "$OUTBASE\\$RUN"
  New-Item -ItemType Directory -Force $RUN_DIR | Out-Null

  Copy-Item -Force $SequencePath "$RUN_DIR\\sequences_fsq_rcm.pkl"

  & $PY tools\\summarize_ar_length_coverage.py "$RUN_DIR\\sequences_fsq_rcm.pkl" `
    --limits 1024,1536,2048 `
    --output "$RUN_DIR\\ar_length_coverage.json" `
    --markdown-output "$RUN_DIR\\ar_length_coverage.md"

  & $PY tools\\preflight_ar_training.py `
    --sequence "$RUN_DIR\\sequences_fsq_rcm.pkl" `
    --output "$RUN_DIR\\ar_preflight.json" `
    --max-seq-len $AR_MAX_SEQ_LEN `
    --batch-size 4 `
    --d-model $AR_DMODEL `
    --layers $AR_LAYERS `
    --max-samples 128

  $env:PYTHONUTF8 = "1"
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONUNBUFFERED = "1"
  $env:CUDA_VISIBLE_DEVICES = "0"
  $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
  $env:NS_OUTBASE = $OUTBASE
  $env:NS_OUT = $RUN
  $env:NS_N = "999999"
  $env:NS_AR_EPOCHS = $AR_EPOCHS
  $env:NS_AR_BS = $AR_BS
  $env:NS_AR_DMODEL = $AR_DMODEL
  $env:NS_AR_LAYERS = $AR_LAYERS
  $env:NS_AR_LR = $AR_LR
  $env:NS_AR_MAX_SEQ_LEN = $AR_MAX_SEQ_LEN
  $env:NS_AR_SAVE_EVERY = $AR_SAVE_EVERY
  $env:NS_AR_LOG_EVERY_BATCHES = $AR_LOG_EVERY_BATCHES
  Remove-Item Env:\\NS_AR_RESUME_FROM -ErrorAction SilentlyContinue

  & $PY breparg_improvements\\train.py --stage ar 2>&1 | Tee-Object -FilePath "$RUN_DIR\\ar_train.log"
  $REPORT_SRC = "breparg_improvements\\repro_outputs\\$RUN\\train_report.json"
  if (Test-Path $REPORT_SRC) {{
    Copy-Item -Force $REPORT_SRC "$RUN_DIR\\train_report.json"
  }}
}}

Train-MediumOrderingAr -Name "dfs" -SequencePath $DFS_SEQUENCE
Train-MediumOrderingAr -Name "rcm" -SequencePath $RCM_SEQUENCE

Write-Host "Medium matched AR branches:"
Write-Host "  DFS: $OUTBASE\\ar_dfs_medium_safe_20260715"
Write-Host "  RCM: $OUTBASE\\ar_rcm_medium_safe_20260715"
"""


def script_02b_medium_smoke(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "02_dfs_rcm_ordering"
    outbase = root / "ar_train_outputs"
    smoke_root = root / "ar_train_smoke_medium"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$OUTBASE = "{ps_path(outbase)}"
$SMOKE_ROOT = "{ps_path(smoke_root)}"
$DFS_SEQUENCE = "$ROOT\\sequence_rebuild_medium\\sequences_fsq_dfs.pkl"
$RCM_SEQUENCE = "$ROOT\\sequence_rebuild_medium\\sequences_fsq_rcm.pkl"

if (!(Test-Path $DFS_SEQUENCE)) {{
  throw "Missing medium DFS sequence package. Run scripts\\02_medium_dfs_rcm_ordering_rebuild.ps1 first."
}}
if (!(Test-Path $RCM_SEQUENCE)) {{
  throw "Missing medium RCM sequence package. Run scripts\\02_medium_dfs_rcm_ordering_rebuild.ps1 first."
}}

New-Item -ItemType Directory -Force $OUTBASE | Out-Null
New-Item -ItemType Directory -Force $SMOKE_ROOT | Out-Null

$AR_MAX_SEQ_LEN = "2048"
$AR_DMODEL = "256"
$AR_LAYERS = "8"

function Smoke-MediumOrderingAr {{
  param(
    [string]$Name,
    [string]$SequencePath
  )

  $SUBSET_DIR = "$SMOKE_ROOT\\$Name`_subset"
  $RUN = "ar_$Name`_medium_smoke_20260715"
  $RUN_DIR = "$OUTBASE\\$RUN"
  New-Item -ItemType Directory -Force $SUBSET_DIR | Out-Null
  New-Item -ItemType Directory -Force $RUN_DIR | Out-Null

  & $PY tools\\subset_ar_sequence_package.py `
    --sequence $SequencePath `
    --output "$SUBSET_DIR\\sequences_fsq_rcm.pkl" `
    --summary "$SUBSET_DIR\\subset_summary.json" `
    --train-limit 64 `
    --val-limit 16 `
    --test-limit 16 `
    --max-seq-len $AR_MAX_SEQ_LEN

  Copy-Item -Force "$SUBSET_DIR\\sequences_fsq_rcm.pkl" "$RUN_DIR\\sequences_fsq_rcm.pkl"

  & $PY tools\\preflight_ar_training.py `
    --sequence "$RUN_DIR\\sequences_fsq_rcm.pkl" `
    --output "$RUN_DIR\\ar_preflight.json" `
    --max-seq-len $AR_MAX_SEQ_LEN `
    --batch-size 2 `
    --d-model $AR_DMODEL `
    --layers $AR_LAYERS `
    --max-samples 32

  $env:PYTHONUTF8 = "1"
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONUNBUFFERED = "1"
  $env:CUDA_VISIBLE_DEVICES = "0"
  $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
  $env:NS_OUTBASE = $OUTBASE
  $env:NS_OUT = $RUN
  $env:NS_N = "999999"
  $env:NS_AR_EPOCHS = "1"
  $env:NS_AR_BS = "2"
  $env:NS_AR_DMODEL = $AR_DMODEL
  $env:NS_AR_LAYERS = $AR_LAYERS
  $env:NS_AR_LR = "5e-5"
  $env:NS_AR_MAX_SEQ_LEN = $AR_MAX_SEQ_LEN
  $env:NS_AR_SAVE_EVERY = "1"
  $env:NS_AR_LOG_EVERY_BATCHES = "10"
  Remove-Item Env:\\NS_AR_RESUME_FROM -ErrorAction SilentlyContinue

  & $PY breparg_improvements\\train.py --stage ar 2>&1 | Tee-Object -FilePath "$RUN_DIR\\ar_train.log"
  $REPORT_SRC = "breparg_improvements\\repro_outputs\\$RUN\\train_report.json"
  if (Test-Path $REPORT_SRC) {{
    Copy-Item -Force $REPORT_SRC "$RUN_DIR\\train_report.json"
  }}
}}

Smoke-MediumOrderingAr -Name "dfs" -SequencePath $DFS_SEQUENCE
Smoke-MediumOrderingAr -Name "rcm" -SequencePath $RCM_SEQUENCE

Write-Host "Medium AR smoke branches:"
Write-Host "  DFS: $OUTBASE\\ar_dfs_medium_smoke_20260715"
Write-Host "  RCM: $OUTBASE\\ar_rcm_medium_smoke_20260715"
"""


def script_02c(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "02_dfs_rcm_ordering"
    outbase = root / "ar_train_outputs"
    eval_root = root / "ar_complex_curved_eval"
    subset = config["complex_subset"]
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$OUTBASE = "{ps_path(outbase)}"
$EVAL_ROOT = "{ps_path(eval_root)}"
$VQVAE = "{ps_path(config["vqvae_checkpoint"])}"
$ARCHIVES = "{ps_path(config["archive_root"])}"

$DFS_SEQUENCE = "$ROOT\\sequences_fsq_dfs.pkl"
$RCM_SEQUENCE = "$ROOT\\sequences_fsq_rcm.pkl"
$DFS_AR = "$OUTBASE\\ar_dfs_matched_20260715\\ar_best.pt"
$RCM_AR = "$OUTBASE\\ar_rcm_matched_20260715\\ar_best.pt"
$ORDERING_SOURCE = "full"

if (!(Test-Path $DFS_SEQUENCE) -or !(Test-Path $RCM_SEQUENCE)) {{
  $MEDIUM_DFS = "$ROOT\\sequence_rebuild_medium\\sequences_fsq_dfs.pkl"
  $MEDIUM_RCM = "$ROOT\\sequence_rebuild_medium\\sequences_fsq_rcm.pkl"
  if ((Test-Path $MEDIUM_DFS) -and (Test-Path $MEDIUM_RCM)) {{
    $DFS_SEQUENCE = $MEDIUM_DFS
    $RCM_SEQUENCE = $MEDIUM_RCM
    $ORDERING_SOURCE = "medium"
  }}
}}

if (!(Test-Path $DFS_AR) -or !(Test-Path $RCM_AR)) {{
  $MEDIUM_DFS_AR = "$OUTBASE\\ar_dfs_medium_safe_20260715\\ar_best.pt"
  $MEDIUM_RCM_AR = "$OUTBASE\\ar_rcm_medium_safe_20260715\\ar_best.pt"
  if ((Test-Path $MEDIUM_DFS_AR) -and (Test-Path $MEDIUM_RCM_AR)) {{
    $DFS_AR = $MEDIUM_DFS_AR
    $RCM_AR = $MEDIUM_RCM_AR
    $ORDERING_SOURCE = "$ORDERING_SOURCE+medium_ar"
  }}
}}

if (!(Test-Path $DFS_SEQUENCE)) {{ throw "Missing DFS sequence package." }}
if (!(Test-Path $RCM_SEQUENCE)) {{ throw "Missing RCM sequence package." }}
if (!(Test-Path $DFS_AR)) {{ throw "Missing DFS AR checkpoint. Run scripts\\02b_train_dfs_rcm_ar.ps1 first." }}
if (!(Test-Path $RCM_AR)) {{ throw "Missing RCM AR checkpoint. Run scripts\\02b_train_dfs_rcm_ar.ps1 first." }}

New-Item -ItemType Directory -Force $EVAL_ROOT | Out-Null
Write-Host "Ordering sequence source: $ORDERING_SOURCE"

& $PY tools\\complex_curved_diagnostics.py `
  --sequence $DFS_SEQUENCE `
  --vqvae-checkpoint $VQVAE `
  --ar-checkpoint $DFS_AR `
  --archive-root $ARCHIVES `
  --output-dir "$EVAL_ROOT\\dfs_teacher_forcing" `
  --split {subset["split"]} `
  --max-samples {subset["max_samples"]} `
  --max-scan {subset["max_scan"]} `
  --max-seq-len {subset["max_seq_len"]} `
  --complex-min-faces {subset["complex_min_faces"]} `
  --complex-min-edges {subset["complex_min_edges"]} `
  --curved-threshold {subset["curved_threshold"]} `
  --max-source-faces {subset["max_source_faces"]} `
  --max-source-edges {subset["max_source_edges"]} `
  --device auto `
  --skip-reconstruction

& $PY tools\\complex_curved_diagnostics.py `
  --sequence $RCM_SEQUENCE `
  --vqvae-checkpoint $VQVAE `
  --ar-checkpoint $RCM_AR `
  --archive-root $ARCHIVES `
  --output-dir "$EVAL_ROOT\\rcm_teacher_forcing" `
  --split {subset["split"]} `
  --max-samples {subset["max_samples"]} `
  --max-scan {subset["max_scan"]} `
  --max-seq-len {subset["max_seq_len"]} `
  --complex-min-faces {subset["complex_min_faces"]} `
  --complex-min-edges {subset["complex_min_edges"]} `
  --curved-threshold {subset["curved_threshold"]} `
  --max-source-faces {subset["max_source_faces"]} `
  --max-source-edges {subset["max_source_edges"]} `
  --device auto `
  --skip-reconstruction

& $PY tools\\summarize_complex_curved_diagnostics.py `
  --report "dfs=$EVAL_ROOT\\dfs_teacher_forcing\\complex_curved_diagnostics_report.json" `
  --report "rcm=$EVAL_ROOT\\rcm_teacher_forcing\\complex_curved_diagnostics_report.json" `
  --output "$EVAL_ROOT\\dfs_vs_rcm_teacher_forcing_summary.md"
"""


def script_03(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "03_breparg_official_baseline"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$WEIGHTS = "$ROOT\\weights"
$GEN = "$ROOT\\official_abc_generate_smoke"
New-Item -ItemType Directory -Force $WEIGHTS, $GEN | Out-Null

function Invoke-Native {{
  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {{
    & $args[0] @($args[1..($args.Count - 1)])
    $exitCode = $LASTEXITCODE
  }} finally {{
    $ErrorActionPreference = $previousPreference
  }}
  if ($exitCode -ne 0) {{
    throw "Command failed with exit code $exitCode`: $($args -join ' ')"
  }}
}}

Invoke-Native $PY -m pip install "huggingface_hub>=0.20.2,<0.26"

$env:BREPARG_WEIGHTS_DIR = $WEIGHTS
@'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "qingtiannihao/BrepARG"
local_dir = Path(os.environ["BREPARG_WEIGHTS_DIR"])
files = [
    "checkpoint/weights/abc_ar.pt",
    "checkpoint/weights/abc_vqvae.pt",
]
for filename in files:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"downloaded {{filename}} -> {{path}}", flush=True)
'@ | & $PY -
if ($LASTEXITCODE -ne 0) {{
  throw "Command failed with exit code $LASTEXITCODE`: $PY -"
}}

# Load and generate a small official-weight smoke first. If this fails, keep
# the log in this folder and switch to same-data BrepARG training for baseline.
Invoke-Native $PY BrepARG\\generate_brep.py `
  --dataset_type abc `
  --config BrepARG\\config.json `
  --ar_model "$WEIGHTS\\checkpoint\\weights\\abc_ar.pt" `
  --se_vqvae "$WEIGHTS\\checkpoint\\weights\\abc_vqvae.pt" `
  --num_samples 10 `
  --mode batch `
  --max_length 2048 `
  --temperature 1.0 `
  --top_p 0.9 `
  --output_dir $GEN `
  --filename_prefix breparg_official_abc `
  --device cuda `
  --gpu 0

# Normalize upstream BrepARG outputs into the same complexity/quality protocol
# used by the V13 generation reports. If the official generator did not create
# a quality manifest, this still records STEP entity complexity and marks strict
# BRep validity as unknown.
Invoke-Native $PY tools\\audit_breparg_baseline_outputs.py `
  --run-dir $GEN `
  --output "$ROOT\\breparg_baseline_quality_summary.json" `
  --markdown-output "$ROOT\\breparg_baseline_quality_summary.md" `
  --manifest-output "$ROOT\\breparg_baseline_quality_manifest.jsonl" `
  --min-faces {config["complex_subset"]["complex_min_faces"]} `
  --min-edges {config["complex_subset"]["complex_min_edges"]} `
  --max-faces 45 `
  --max-edges 120
"""


def script_03a(config: dict[str, Any], *, full: bool = False) -> str:
    data = Path(config["output_dir"]) / "experiments" / "03b_breparg_same_data_training_fallback" / "data"
    train_limit = 50000 if full else 10000
    val_limit = 5000 if full else 1000
    test_limit = 5000 if full else 1000
    surface_limit = 1000000 if full else 300000
    edge_limit = 1500000 if full else 500000
    label = "full" if full else "medium"
    sizing_note = (
        "This full pool is intended for the external SSD or a machine with enough free disk."
        if full
        else "The default medium pool is intentionally smaller than the full protocol so it can run on the temporary local drive before the external SSD is available."
    )
    if full:
        data = data.parent / "data_full"
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$DATA = "{ps_path(data)}"
New-Item -ItemType Directory -Force $DATA | Out-Null

# This prepares the {label} same-data BrepARG baseline input pool.
# {sizing_note}
& $PY tools\\prepare_breparg_same_data_inputs.py `
  --sequence "{ps_path(config["sequence_path"])}" `
  --archive-root "{ps_path(config["archive_root"])}" `
  --output-dir $DATA `
  --train-limit {train_limit} `
  --val-limit {val_limit} `
  --test-limit {test_limit} `
  --max-faces 50 `
  --max-edges 150 `
  --surface-patch-limit {surface_limit} `
  --edge-patch-limit {edge_limit}

if (!(Test-Path "$DATA\\same_data_input_summary.json")) {{
  throw "same_data_input_summary.json was not created under $DATA"
}}
if (!(Test-Path "$DATA\\same_data_split.pkl")) {{
  throw "same_data_split.pkl was not created under $DATA"
}}
if (!(Test-Path "$DATA\\deduplicated_surface_source.pkl")) {{
  throw "deduplicated_surface_source.pkl was not created under $DATA"
}}
if (!(Test-Path "$DATA\\deduplicated_edge_source.pkl")) {{
  throw "deduplicated_edge_source.pkl was not created under $DATA"
}}

Write-Host "Prepared BrepARG same-data inputs:"
Write-Host "  $DATA\\same_data_input_summary.json"
Write-Host "  $DATA\\same_data_split.pkl"
Write-Host "  $DATA\\deduplicated_surface_source.pkl"
Write-Host "  $DATA\\deduplicated_edge_source.pkl"
"""


def script_03b(config: dict[str, Any], *, smoke: bool = False) -> str:
    root = Path(config["output_dir"]) / "experiments" / "03b_breparg_same_data_training_fallback"
    data_folder = "data_smoke" if smoke else "data"
    run_suffix = "_smoke" if smoke else ""
    env_name = "same_data_abc_smoke" if smoke else "same_data_abc"
    vqvae_epochs = "1" if smoke else "300"
    vqvae_bs = "16" if smoke else "256"
    vqvae_save_every = "1" if smoke else "20"
    ar_epochs = "1" if smoke else "120"
    ar_bs = "2" if smoke else "24"
    ar_save_every = "1" if smoke else "10"
    generate_samples = "2" if smoke else "100"
    generate_max_attempts = "20" if smoke else "5000"
    manifest_name = (
        "same_data_breparg_fallback_smoke_manifest.json"
        if smoke
        else "same_data_breparg_fallback_manifest.json"
    )
    quality_prefix = "breparg_same_data_smoke" if smoke else "breparg_same_data"
    training_note = (
        "SMOKE ONLY: verifies the same-data BrepARG fallback pipeline wiring with tiny data and 1 epoch."
        if smoke
        else "same-data BrepARG fallback baseline; use only if official weights are incompatible"
    )
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$VQVAE_RUN = "$ROOT\\vqvae{run_suffix}"
$SEQ_RUN = "$ROOT\\sequence{run_suffix}"
$AR_RUN = "$ROOT\\ar{run_suffix}"
$GEN = "$ROOT\\generated{run_suffix}"
$VQVAE_TB = "$VQVAE_RUN\\tensorboard"
$AR_TB = "$AR_RUN\\tensorboard"
New-Item -ItemType Directory -Force $VQVAE_RUN, $SEQ_RUN, $AR_RUN, $GEN, $VQVAE_TB, $AR_TB | Out-Null

function Invoke-Native {{
  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {{
    & $args[0] @($args[1..($args.Count - 1)])
    $exitCode = $LASTEXITCODE
  }} finally {{
    $ErrorActionPreference = $previousPreference
  }}
  if ($exitCode -ne 0) {{
    throw "Command failed with exit code $exitCode`: $($args -join ' ')"
  }}
}}

function Test-PythonModule {{
  param([string]$Name)
  $env:PY_MODULE_TO_CHECK = $Name
  & $PY -c "import importlib.util, os, sys; sys.exit(0 if importlib.util.find_spec(os.environ['PY_MODULE_TO_CHECK']) else 1)"
  $ok = ($LASTEXITCODE -eq 0)
  Remove-Item Env:\\PY_MODULE_TO_CHECK -ErrorAction SilentlyContinue
  return $ok
}}

if (!(Test-PythonModule "tensorboard")) {{
  Write-Host "Python module tensorboard is missing; installing it into the selected environment."
  Invoke-Native $PY -m pip install tensorboard
}}
foreach ($MODULE in @("torch", "tensorboard", "diffusers", "transformers", "OCC", "occwl", "shutup", "tqdm")) {{
  if (!(Test-PythonModule $MODULE)) {{
    throw "Missing required Python module '$MODULE' in $PY"
  }}
}}

# Official weights should be tried first. Use this fallback only when the
# official ABC checkpoint format cannot be loaded or cannot be evaluated under
# the same protocol.
$OFFICIAL_SUMMARY = "{ps_path(Path(config["output_dir"]) / "experiments" / "03_breparg_official_baseline" / "breparg_baseline_quality_summary.json")}"
if (Test-Path $OFFICIAL_SUMMARY) {{
  Write-Host "Official baseline summary exists: $OFFICIAL_SUMMARY"
  Write-Host "Review it before launching same-data BrepARG fallback training."
}}

# Required same-data inputs. Run
# scripts\\03a_prepare_breparg_same_data_inputs.ps1 first to create these
# from the current sequence package and parsed zip archives.
$DATA = "$ROOT\\{data_folder}"
$SPLIT = "$DATA\\same_data_split.pkl"
$DEDUP_SURFACES = "$DATA\\deduplicated_surface_source.pkl"
$DEDUP_EDGES = "$DATA\\deduplicated_edge_source.pkl"
$INPUT_SUMMARY = "$DATA\\same_data_input_summary.json"
if (!(Test-Path $INPUT_SUMMARY)) {{ throw "Missing same-data input summary under $DATA." }}
if (!(Test-Path $SPLIT)) {{ throw "Missing same_data_split.pkl under $DATA." }}
if (!(Test-Path $DEDUP_SURFACES)) {{ throw "Missing deduplicated_surface_source.pkl under $DATA." }}
if (!(Test-Path $DEDUP_EDGES)) {{ throw "Missing deduplicated_edge_source.pkl under $DATA." }}

# Keep this medium-size unless you intentionally want a full-paper baseline.
$VQVAE_EPOCHS = "{vqvae_epochs}"
$VQVAE_BS = "{vqvae_bs}"
$VQVAE_SAVE_EVERY = "{vqvae_save_every}"
$AR_EPOCHS = "{ar_epochs}"
$AR_BS = "{ar_bs}"
$AR_LR = "5e-5"
$AR_MAX_SEQ_LEN = "2048"
$AR_SAVE_EVERY = "{ar_save_every}"
$GENERATE_SAMPLES = "{generate_samples}"
$GENERATE_MAX_ATTEMPTS = "{generate_max_attempts}"
$GPU = "0"

# 1. Train original BrepARG SE VQ-VAE on the same data split and deduplicated
# surface/edge source pools.
Invoke-Native $PY BrepARG\\train_vqvae.py `
  --data_list $SPLIT `
  --surface_list $DEDUP_SURFACES `
  --edge_list $DEDUP_EDGES `
  --dataset_type abc `
  --batch_size $VQVAE_BS `
  --train_epoch $VQVAE_EPOCHS `
  --test_epoch 1 `
  --save_epoch $VQVAE_SAVE_EVERY `
  --max_face 50 `
  --max_edge 150 `
  --dir_name $VQVAE_RUN `
  --env {env_name} `
  --loss_dir $VQVAE_RUN `
  --tb_log_dir $VQVAE_TB `
  --no_aug `
  --gpu $GPU 2>&1 | Tee-Object -FilePath "$VQVAE_RUN\\train_vqvae.log"

$VQVAE_WEIGHT = Get-ChildItem $VQVAE_RUN -Recurse -File -Include *.pt,*.pth |
  Sort-Object LastWriteTime |
  Select-Object -Last 1 -ExpandProperty FullName
if (!$VQVAE_WEIGHT) {{ throw "Could not find trained BrepARG VQ-VAE weight under $VQVAE_RUN" }}

# 2. Build original BrepARG DFS-style sequence package with that checkpoint.
$SEQUENCE = "$SEQ_RUN\\breparg_same_data{run_suffix}_sequences.pkl"
Invoke-Native $PY BrepARG\\2sequence.py `
  --data_list $SPLIT `
  --output_file $SEQUENCE `
  --vqvae_se_weight $VQVAE_WEIGHT `
  --dataset_type abc `
  --max_face 50 `
  --max_edge 150 `
  --scale 1.0 `
  --aug true `
  --gpu $GPU 2>&1 | Tee-Object -FilePath "$SEQ_RUN\\build_sequence.log"

if (!(Test-Path $SEQUENCE)) {{ throw "BrepARG sequence build did not produce $SEQUENCE" }}

# 3. Train original BrepARG AR model on the same-data sequence package.
Invoke-Native $PY BrepARG\\train_ar.py `
  --sequence_file $SEQUENCE `
  --dataset_type abc `
  --batch_size $AR_BS `
  --train_epoch $AR_EPOCHS `
  --test_epoch 1 `
  --save_epoch $AR_SAVE_EVERY `
  --max_face 50 `
  --max_edge 150 `
  --max_seq_len $AR_MAX_SEQ_LEN `
  --learning_rate $AR_LR `
  --d_model 256 `
  --nhead 8 `
  --num_layers 8 `
  --dim_feedforward 1024 `
  --dir_name $AR_RUN `
  --env {env_name} `
  --loss_dir $AR_RUN `
  --tb_log_dir $AR_TB 2>&1 | Tee-Object -FilePath "$AR_RUN\\train_ar.log"

$AR_WEIGHT = Get-ChildItem $AR_RUN -Recurse -File -Include *.pt,*.pth |
  Sort-Object LastWriteTime |
  Select-Object -Last 1 -ExpandProperty FullName
if (!$AR_WEIGHT) {{ throw "Could not find trained BrepARG AR weight under $AR_RUN" }}

# 4. Generate a smoke set and audit it with the same protocol used elsewhere.
if (Test-Path $GEN) {{
  Remove-Item -LiteralPath $GEN -Recurse -Force
}}
New-Item -ItemType Directory -Force $GEN | Out-Null
Invoke-Native $PY BrepARG\\generate_brep.py `
  --dataset_type abc `
  --config BrepARG\\config.json `
  --ar_model $AR_WEIGHT `
  --se_vqvae $VQVAE_WEIGHT `
  --num_samples $GENERATE_SAMPLES `
  --max_attempts $GENERATE_MAX_ATTEMPTS `
  --mode batch `
  --max_length 2048 `
  --temperature 1.0 `
  --top_p 0.9 `
  --output_dir $GEN `
  --filename_prefix {quality_prefix} `
  --device cuda `
  --gpu 0 2>&1 | Tee-Object -FilePath "$GEN\\generate.log"

Invoke-Native $PY tools\\audit_breparg_baseline_outputs.py `
  --run-dir $GEN `
  --output "$ROOT\\{quality_prefix}_quality_summary.json" `
  --markdown-output "$ROOT\\{quality_prefix}_quality_summary.md" `
  --manifest-output "$ROOT\\{quality_prefix}_quality_manifest.jsonl" `
  --min-faces {config["complex_subset"]["complex_min_faces"]} `
  --min-edges {config["complex_subset"]["complex_min_edges"]} `
  --max-faces 45 `
  --max-edges 120

@{{
  split = $SPLIT
  dedup_surfaces = $DEDUP_SURFACES
  dedup_edges = $DEDUP_EDGES
  vqvae_weight = $VQVAE_WEIGHT
  sequence = $SEQUENCE
  ar_weight = $AR_WEIGHT
  generated = $GEN
  note = "{training_note}"
}} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 "$ROOT\\{manifest_name}"
"""


def script_03b_preflight(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"]) / "experiments" / "03b_breparg_same_data_training_fallback"
    official_incompat = (
        Path(config["output_dir"])
        / "experiments"
        / "03_breparg_official_baseline"
        / "official_baseline_incompatibility_report.json"
    )
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$DATA = "$ROOT\\data"
$OUT = "$ROOT\\breparg_same_data_preflight.json"

& $PY tools\\preflight_breparg_same_data_fallback.py `
  --root $ROOT `
  --data-dir $DATA `
  --python-exe $PY `
  --official-incompat-report "{ps_path(official_incompat)}" `
  --output $OUT

Write-Host "BrepARG same-data fallback preflight:"
Write-Host "  $OUT"
"""


def script_04(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"])
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$REPORTS = @()
$CURRENT = "$ROOT\\experiments\\00_current_fsq_ar_teacher_reconstruction\\complex_curved_diagnostics_report.json"
$FSQ_ONLY = "$ROOT\\experiments\\00_fsq_only_patch_metrics\\complex_curved_diagnostics_report.json"
$TEACHER = "$ROOT\\experiments\\01_teacher_forcing_true_token_reconstruction\\complex_curved_diagnostics_report.json"
$CAPACITY = "$ROOT\\experiments\\01_fsq_capacity_candidate\\complex_curved_diagnostics_report.json"
if (Test-Path $CURRENT) {{ $REPORTS += "current=$CURRENT" }}
if (Test-Path $FSQ_ONLY) {{ $REPORTS += "fsq_only=$FSQ_ONLY" }}
if (Test-Path $TEACHER) {{ $REPORTS += "teacher_forcing=$TEACHER" }}
if (Test-Path $CAPACITY) {{ $REPORTS += "capacity_candidate=$CAPACITY" }}
if ($REPORTS.Count -eq 0) {{
  throw "No complex curved diagnostic reports found yet."
}}

$ARGS = @(
  "tools\\summarize_complex_curved_diagnostics.py",
  "--output",
  "$ROOT\\complex_curved_diagnostic_summary.md"
)
foreach ($REPORT in $REPORTS) {{
  $ARGS += "--report"
  $ARGS += $REPORT
}}

& $PY @ARGS

if ((Test-Path $FSQ_ONLY) -and (Test-Path $CAPACITY)) {{
  & $PY tools\\compare_fsq_capacity_diagnostics.py `
    --baseline $FSQ_ONLY `
    --candidate $CAPACITY `
    --output "$ROOT\\fsq_capacity_comparison.json" `
    --markdown-output "$ROOT\\fsq_capacity_comparison.md"
}}

$CURRENT_DIR = "$ROOT\\experiments\\00_current_fsq_ar_teacher_reconstruction"
$TEACHER_DIR = "$ROOT\\experiments\\01_teacher_forcing_true_token_reconstruction"
if (!(Test-Path "$CURRENT_DIR\\teacher_reconstruction_manifest.jsonl") -and (Test-Path "$TEACHER_DIR\\teacher_reconstruction_manifest.jsonl")) {{
  $CURRENT_DIR = $TEACHER_DIR
}}
$MANIFEST = "$CURRENT_DIR\\teacher_reconstruction_manifest.jsonl"
$PATCH_METRICS = "$CURRENT_DIR\\fsq_patch_metrics.jsonl"
if ((Test-Path $MANIFEST) -and (Test-Path $PATCH_METRICS)) {{
  & $PY tools\\analyze_reconstruction_fsq_correlation.py `
    --manifest $MANIFEST `
    --patch-metrics $PATCH_METRICS `
    --output "$CURRENT_DIR\\reconstruction_fsq_correlation.json" `
    --markdown-output "$CURRENT_DIR\\reconstruction_fsq_correlation.md" `
    --top-k 10
}}
"""


def script_05(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"])
    return f"""
$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"

& $PY tools\\audit_complex_curved_control_suite.py `
  --root $ROOT `
  --output "$ROOT\\suite_status.json" `
  --markdown-output "$ROOT\\suite_status.md"

Write-Host "Suite status:"
Write-Host "  $ROOT\\suite_status.md"
"""


def script_06(config: dict[str, Any]) -> str:
    root = Path(config["output_dir"])
    return f"""
param(
  [string]$DestRoot = "E:\\V13_rootcause_20260715",
  [switch]$Execute,
  [switch]$CopyReferenceModels,
  [switch]$CopyArchives
)

$ErrorActionPreference = "Stop"
cd "{ps_path(REPO_ROOT)}"

$PY = "{config["python_exe"]}"
$ROOT = "{ps_path(root)}"
$ARGS = @(
  "tools\\prepare_rootcause_ssd_migration.py",
  "--suite-root", $ROOT,
  "--dest-root", $DestRoot,
  "--manifest", "$ROOT\\ssd_migration_plan.json",
  "--commands-output", "$ROOT\\ssd_migration_commands.md"
)
if ($Execute) {{ $ARGS += "--execute" }}
if ($CopyReferenceModels) {{ $ARGS += "--copy-reference-models" }}
if ($CopyArchives) {{ $ARGS += "--copy-archives" }}

& $PY @ARGS

Write-Host "Migration manifest:"
Write-Host "  $ROOT\\ssd_migration_plan.json"
Write-Host "Commands:"
Write-Host "  $ROOT\\ssd_migration_commands.md"
"""


def prepare_workspace(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    python_exe: str = DEFAULT_PYTHON,
    sequence_path: Path = DEFAULT_SEQUENCE,
    vqvae_checkpoint: Path = DEFAULT_VQVAE,
    ar_checkpoint: Path = DEFAULT_AR,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    config = workspace_config(
        output_dir=output_dir,
        python_exe=python_exe,
        sequence_path=Path(sequence_path),
        vqvae_checkpoint=Path(vqvae_checkpoint),
        ar_checkpoint=Path(ar_checkpoint),
        archive_root=Path(archive_root),
    )
    for rel in (
        "experiments/00_current_fsq_ar_teacher_reconstruction",
        "experiments/01a_train_fsq_capacity_candidate",
        "experiments/01_fsq_capacity_candidate",
        "experiments/02_dfs_rcm_ordering",
        "experiments/02_dfs_rcm_ordering/same_data_split",
        "experiments/02_dfs_rcm_ordering/same_data_split_smoke",
        "experiments/02_dfs_rcm_ordering/sequence_rebuild_smoke",
        "experiments/02_dfs_rcm_ordering/sequence_rebuild_medium",
        "experiments/02_dfs_rcm_ordering/ar_train_outputs",
        "experiments/02_dfs_rcm_ordering/ar_train_smoke_medium",
        "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_dfs_medium_smoke_20260715",
        "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_rcm_medium_smoke_20260715",
        "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_dfs_medium_safe_20260715",
        "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_rcm_medium_safe_20260715",
        "experiments/02_dfs_rcm_ordering/ar_complex_curved_eval",
        "experiments/03_breparg_official_baseline",
        "experiments/03b_breparg_same_data_training_fallback",
        "experiments/03b_breparg_same_data_training_fallback/data",
        "experiments/03b_breparg_same_data_training_fallback/data_smoke",
        "scripts",
    ):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_config.json", config)
    write_text(output_dir / "README.md", readme_text(config))
    write_text(output_dir / "scripts" / "00_current_fsq_ar_teacher_reconstruction.ps1", script_00(config))
    write_text(output_dir / "scripts" / "01a_preflight_fsq_capacity_candidate.ps1", script_01a_preflight(config))
    write_text(output_dir / "scripts" / "01a_build_fsq_capacity_sample_cache.ps1", script_01a_sample_cache(config))
    write_text(output_dir / "scripts" / "01a_train_fsq_capacity_candidate.ps1", script_01a(config))
    write_text(output_dir / "scripts" / "01a_resume_fsq_capacity_candidate.ps1", script_01a_resume(config))
    write_text(output_dir / "scripts" / "01a_watch_fsq_capacity_then_eval.ps1", script_01a_watch_then_eval(config))
    write_text(output_dir / "scripts" / "01_fsq_capacity_candidate.ps1", script_01(config))
    write_text(output_dir / "scripts" / "02a_prepare_v13_same_data_split.ps1", script_02a(config))
    write_text(output_dir / "scripts" / "02_smoke_dfs_rcm_ordering_rebuild.ps1", script_02_smoke(config))
    write_text(output_dir / "scripts" / "02_medium_dfs_rcm_ordering_rebuild.ps1", script_02_medium(config))
    write_text(output_dir / "scripts" / "02_dfs_rcm_ordering_rebuild.ps1", script_02(config))
    write_text(output_dir / "scripts" / "02b_smoke_dfs_rcm_ar_medium_safe.ps1", script_02b_medium_smoke(config))
    write_text(output_dir / "scripts" / "02b_train_dfs_rcm_ar_medium_safe.ps1", script_02b_medium_safe(config))
    write_text(output_dir / "scripts" / "02b_train_dfs_rcm_ar.ps1", script_02b(config))
    write_text(output_dir / "scripts" / "02c_eval_dfs_rcm_ar_complex_curved.ps1", script_02c(config))
    write_text(output_dir / "scripts" / "03_breparg_official_baseline.ps1", script_03(config))
    write_text(output_dir / "scripts" / "03a_prepare_breparg_same_data_inputs.ps1", script_03a(config))
    write_text(output_dir / "scripts" / "03a_prepare_breparg_same_data_inputs_full.ps1", script_03a(config, full=True))
    write_text(
        output_dir / "scripts" / "03b_smoke_breparg_same_data_training_fallback.ps1",
        script_03b(config, smoke=True),
    )
    write_text(
        output_dir / "scripts" / "03b_preflight_breparg_same_data_fallback.ps1",
        script_03b_preflight(config),
    )
    write_text(output_dir / "scripts" / "03b_breparg_same_data_training_fallback.ps1", script_03b(config))
    write_text(output_dir / "scripts" / "04_summarize_reports.ps1", script_04(config))
    write_text(output_dir / "scripts" / "05_audit_suite_status.ps1", script_05(config))
    write_text(output_dir / "scripts" / "06_prepare_external_ssd_migration.ps1", script_06(config))
    return {
        "output_dir": str(output_dir),
        "scripts": sorted(str(path) for path in (output_dir / "scripts").glob("*.ps1")),
        "config": str(output_dir / "experiment_config.json"),
        "readme": str(output_dir / "README.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python-exe", default=DEFAULT_PYTHON)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--vqvae-checkpoint", type=Path, default=DEFAULT_VQVAE)
    parser.add_argument("--ar-checkpoint", type=Path, default=DEFAULT_AR)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_workspace(
        output_dir=args.output_dir,
        python_exe=args.python_exe,
        sequence_path=args.sequence,
        vqvae_checkpoint=args.vqvae_checkpoint,
        ar_checkpoint=args.ar_checkpoint,
        archive_root=args.archive_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
