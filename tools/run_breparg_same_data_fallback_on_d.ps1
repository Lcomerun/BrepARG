$ErrorActionPreference = "Stop"
cd "D:\luolin\V13"

$PY = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
$SourceRoot = "E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\experiments\03b_breparg_same_data_training_fallback"
$Root = "D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback"
$StagedData = "$Root\data_staged"
$RunTag = "3060_safe_len1536_bs4_20260717_d"
$VqvaeRun = "$Root\vqvae_$RunTag"
$SeqRun = "$Root\sequence_$RunTag"
$ArRun = "$Root\ar_$RunTag"
$Gen = "$Root\generated_$RunTag"
$VqvaeTb = "$VqvaeRun\tensorboard"
$ArTb = "$ArRun\tensorboard"
$Manifest = "$Root\same_data_breparg_fallback_$RunTag`_manifest.json"
New-Item -ItemType Directory -Force $Root, $StagedData, $VqvaeRun, $SeqRun, $ArRun, $Gen, $VqvaeTb, $ArTb | Out-Null

function Invoke-Native {
  $previousPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $args[0] @($args[1..($args.Count - 1)])
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousPreference
  }
  if ($exitCode -ne 0) {
    throw "Command failed with exit code $exitCode`: $($args -join ' ')"
  }
}

function Invoke-NativeLogged {
  param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath,

    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Command
  )

  if ($Command.Count -lt 1) {
    throw "Invoke-NativeLogged requires an executable path"
  }

  $FilePath = $Command[0]
  $CommandArgs = @()
  if ($Command.Count -gt 1) {
    $CommandArgs = @($Command[1..($Command.Count - 1)])
  }

  function ConvertTo-WindowsArgumentString {
    param([string[]]$Arguments)

    $quoted = foreach ($arg in $Arguments) {
      if ($null -eq $arg) {
        '""'
      } elseif ($arg -notmatch '[\s"]') {
        $arg
      } else {
        $escaped = $arg -replace '\\(?=($|"))', '\\' -replace '"', '\"'
        '"' + $escaped + '"'
      }
    }
    return ($quoted -join ' ')
  }

  $ArgumentString = ConvertTo-WindowsArgumentString $CommandArgs

  $LogDir = Split-Path -Parent $LogPath
  New-Item -ItemType Directory -Force $LogDir | Out-Null
  $ErrPath = "$LogPath.err"
  Remove-Item -Force -ErrorAction SilentlyContinue $LogPath, $ErrPath

  Write-Host "Running: $FilePath $($CommandArgs -join ' ')"
  Write-Host "  stdout: $LogPath"
  Write-Host "  stderr: $ErrPath"

  $proc = Start-Process `
    -FilePath $FilePath `
    -ArgumentList $ArgumentString `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError $ErrPath `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

  if ($proc.ExitCode -ne 0) {
    Write-Host "Command failed with exit code $($proc.ExitCode). Recent stdout:"
    Get-Content -LiteralPath $LogPath -Tail 80 -ErrorAction SilentlyContinue
    Write-Host "Recent stderr:"
    Get-Content -LiteralPath $ErrPath -Tail 80 -ErrorAction SilentlyContinue
    throw "Command failed with exit code $($proc.ExitCode): $FilePath $($CommandArgs -join ' ')"
  }
}

function Test-PythonModule {
  param([string]$Name)
  $env:PY_MODULE_TO_CHECK = $Name
  & $PY -c "import importlib.util, os, sys; sys.exit(0 if importlib.util.find_spec(os.environ['PY_MODULE_TO_CHECK']) else 1)"
  $ok = ($LASTEXITCODE -eq 0)
  Remove-Item Env:\PY_MODULE_TO_CHECK -ErrorAction SilentlyContinue
  return $ok
}

function Get-LatestWeight {
  param([string]$Directory, [string]$BestPattern)
  if (Test-Path -LiteralPath $Directory) {
    $best = Get-ChildItem -LiteralPath $Directory -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object {
        $_.Name -eq $BestPattern -and $_.Extension -in @(".pt", ".pth")
      } |
      Sort-Object LastWriteTime |
      Select-Object -Last 1 -ExpandProperty FullName
    if ($best) { return $best }
    $latest = Get-ChildItem -LiteralPath $Directory -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object {
        $_.Extension -in @(".pt", ".pth") -and $_.Length -gt 1MB
      } |
      Sort-Object LastWriteTime |
      Select-Object -Last 1 -ExpandProperty FullName
    if ($latest) { return $latest }
  }
  return $null
}

function Invoke-RobocopyOk {
  param(
    [string]$Source,
    [string]$Destination,
    [string[]]$ExtraArgs = @()
  )
  New-Item -ItemType Directory -Force $Destination | Out-Null
  $args = @($Source, $Destination) + $ExtraArgs
  & robocopy @args
  $exitCode = $LASTEXITCODE
  if ($exitCode -gt 7) {
    throw "robocopy failed with exit code $exitCode`: robocopy $($args -join ' ')"
  }
}

function Test-StagedSameDataInputs {
  param(
    [string]$SourceData,
    [string]$DestinationData
  )

  $required = @(
    "$DestinationData\same_data_input_summary.source.json",
    "$DestinationData\deduplicated_surface_source.pkl",
    "$DestinationData\deduplicated_edge_source.pkl",
    "$DestinationData\same_data_split.pkl",
    "$DestinationData\staging_manifest.json"
  )
  foreach ($path in $required) {
    if (!(Test-Path -LiteralPath $path)) {
      return $false
    }
  }

  $sourceParsed = "$SourceData\parsed_pool"
  $destParsed = "$DestinationData\parsed_pool"
  if (!(Test-Path -LiteralPath $sourceParsed) -or !(Test-Path -LiteralPath $destParsed)) {
    return $false
  }

  $sourceCount = (Get-ChildItem -LiteralPath $sourceParsed -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
  $destCount = (Get-ChildItem -LiteralPath $destParsed -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
  if ($sourceCount -le 0 -or $sourceCount -ne $destCount) {
    return $false
  }

  return $true
}

function Stage-SameDataInputs {
  param(
    [string]$SourceData,
    [string]$DestinationData
  )

  if (!(Test-Path -LiteralPath $SourceData)) {
    throw "Missing same-data source directory: $SourceData"
  }

  New-Item -ItemType Directory -Force $DestinationData | Out-Null

  if (Test-StagedSameDataInputs -SourceData $SourceData -DestinationData $DestinationData) {
    Write-Host "Using existing staged same-data BrepARG inputs: $DestinationData"
    return
  }

  Write-Host "Staging same-data BrepARG inputs to D: $DestinationData"
  Copy-Item -Force -LiteralPath "$SourceData\same_data_input_summary.json" -Destination "$DestinationData\same_data_input_summary.source.json"
  Copy-Item -Force -LiteralPath "$SourceData\deduplicated_surface_source.pkl" -Destination "$DestinationData\deduplicated_surface_source.pkl"
  Copy-Item -Force -LiteralPath "$SourceData\deduplicated_edge_source.pkl" -Destination "$DestinationData\deduplicated_edge_source.pkl"

  Invoke-RobocopyOk `
    -Source "$SourceData\parsed_pool" `
    -Destination "$DestinationData\parsed_pool" `
    -ExtraArgs @("/E", "/MT:16", "/J", "/R:2", "/W:2", "/FFT", "/NP")

  $env:V13_BREPARG_SOURCE_DATA = $SourceData
  $env:V13_BREPARG_STAGED_DATA = $DestinationData
  @'
import json
import os
import pickle
from pathlib import Path

source = Path(os.environ["V13_BREPARG_SOURCE_DATA"]).resolve()
dest = Path(os.environ["V13_BREPARG_STAGED_DATA"]).resolve()
split_path = source / "same_data_split.pkl"
with split_path.open("rb") as handle:
    split = pickle.load(handle)

rewritten = {}
counts = {}
missing = []
for key, paths in split.items():
    out = []
    for raw in paths:
        path = Path(raw)
        try:
            rel = path.resolve().relative_to(source)
        except Exception:
            text = str(raw).replace("\\", "/")
            marker = "/parsed_pool/"
            if marker not in text:
                missing.append(str(raw))
                continue
            rel = Path("parsed_pool") / Path(text.split(marker, 1)[1])
        target = dest / rel
        out.append(str(target))
        if not target.exists():
            missing.append(str(target))
    rewritten[key] = out
    counts[key] = len(out)

if missing:
    preview = "\n".join(missing[:10])
    raise SystemExit(f"staged split has {len(missing)} missing paths; first entries:\n{preview}")

with (dest / "same_data_split.pkl").open("wb") as handle:
    pickle.dump(rewritten, handle, protocol=pickle.HIGHEST_PROTOCOL)

manifest = {
    "source_data": str(source),
    "staged_data": str(dest),
    "split": str(dest / "same_data_split.pkl"),
    "counts": counts,
    "deduplicated_surface_source": str(dest / "deduplicated_surface_source.pkl"),
    "deduplicated_edge_source": str(dest / "deduplicated_edge_source.pkl"),
}
(dest / "staging_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=True))
'@ | & $PY -
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to rewrite staged same-data split under $DestinationData"
  }
  Remove-Item Env:\V13_BREPARG_SOURCE_DATA -ErrorAction SilentlyContinue
  Remove-Item Env:\V13_BREPARG_STAGED_DATA -ErrorAction SilentlyContinue
}

foreach ($Module in @("torch", "tensorboard", "diffusers", "transformers", "OCC", "occwl", "shutup", "tqdm")) {
  if (!(Test-PythonModule $Module)) {
    throw "Missing required Python module '$Module' in $PY"
  }
}

$OfficialIncompat = "E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715\experiments\03_breparg_official_baseline\official_baseline_incompatibility_report.json"
if (!(Test-Path -LiteralPath $OfficialIncompat)) {
  throw "Official BrepARG incompatibility report is missing. Try official weights before same-data fallback."
}

$SourceData = "$SourceRoot\data"
$SourceSplit = "$SourceData\same_data_split.pkl"
$SourceDedupSurfaces = "$SourceData\deduplicated_surface_source.pkl"
$SourceDedupEdges = "$SourceData\deduplicated_edge_source.pkl"
$SourceInputSummary = "$SourceData\same_data_input_summary.json"
foreach ($Path in @($SourceInputSummary, $SourceSplit, $SourceDedupSurfaces, $SourceDedupEdges)) {
  if (!(Test-Path -LiteralPath $Path)) {
    throw "Missing same-data fallback input: $Path"
  }
}

Stage-SameDataInputs -SourceData $SourceData -DestinationData $StagedData

$Data = $StagedData
$Split = "$Data\same_data_split.pkl"
$DedupSurfaces = "$Data\deduplicated_surface_source.pkl"
$DedupEdges = "$Data\deduplicated_edge_source.pkl"
$InputSummary = "$Data\same_data_input_summary.source.json"
foreach ($Path in @($InputSummary, $Split, $DedupSurfaces, $DedupEdges)) {
  if (!(Test-Path -LiteralPath $Path)) {
    throw "Missing staged same-data fallback input: $Path"
  }
}

$VqvaeEpochs = "160"
$VqvaeBs = "128"
$VqvaeSaveEvery = "10"
$ArEpochs = "80"
$ArBs = "4"
$ArLr = "5e-5"
$ArMaxSeqLen = "1536"
$ArSaveEvery = "5"
$GenerateSamples = "100"
$GenerateMaxAttempts = "5000"
$Gpu = "0"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$VqvaeWeight = Get-LatestWeight -Directory $VqvaeRun -BestPattern "abc_se_vqvae_best.pt"
if (!$VqvaeWeight) {
  Write-Host "Training BrepARG SE VQ-VAE from scratch on D: $VqvaeRun"
  Invoke-NativeLogged -LogPath "$VqvaeRun\train_vqvae.log" $PY BrepARG\train_vqvae.py `
    --data_list $Split `
    --surface_list $DedupSurfaces `
    --edge_list $DedupEdges `
    --dataset_type abc `
    --batch_size $VqvaeBs `
    --train_epoch $VqvaeEpochs `
    --test_epoch 1 `
    --save_epoch $VqvaeSaveEvery `
    --max_face 50 `
    --max_edge 150 `
    --dir_name $VqvaeRun `
    --env same_data_abc `
    --loss_dir $VqvaeRun `
    --tb_log_dir $VqvaeTb `
    --no_aug `
    --gpu $Gpu
  $VqvaeWeight = Get-LatestWeight -Directory $VqvaeRun -BestPattern "abc_se_vqvae_best.pt"
} else {
  Write-Host "Using existing BrepARG VQ-VAE weight: $VqvaeWeight"
}
if (!$VqvaeWeight) { throw "Could not find trained BrepARG VQ-VAE weight under $VqvaeRun" }

$Sequence = "$SeqRun\breparg_same_data_sequences.pkl"
if (!(Test-Path -LiteralPath $Sequence)) {
  Write-Host "Building BrepARG same-data sequences: $Sequence"
  Invoke-NativeLogged -LogPath "$SeqRun\build_sequence.log" $PY BrepARG\2sequence.py `
    --data_list $Split `
    --output_file $Sequence `
    --vqvae_se_weight $VqvaeWeight `
    --dataset_type abc `
    --max_face 50 `
    --max_edge 150 `
    --scale 1.0 `
    --aug true `
    --gpu $Gpu
} else {
  Write-Host "Using existing BrepARG sequence package: $Sequence"
}
if (!(Test-Path -LiteralPath $Sequence)) { throw "BrepARG sequence build did not produce $Sequence" }

$ArWeight = Get-LatestWeight -Directory $ArRun -BestPattern "abc_ar_vqvae_best_model.pt"
$ArArgs = @(
  "BrepARG\train_ar.py",
  "--sequence_file", $Sequence,
  "--dataset_type", "abc",
  "--batch_size", $ArBs,
  "--train_epoch", $ArEpochs,
  "--test_epoch", "1",
  "--save_epoch", $ArSaveEvery,
  "--max_face", "50",
  "--max_edge", "150",
  "--max_seq_len", $ArMaxSeqLen,
  "--learning_rate", $ArLr,
  "--d_model", "256",
  "--nhead", "8",
  "--num_layers", "8",
  "--dim_feedforward", "1024",
  "--dir_name", $ArRun,
  "--env", "same_data_abc",
  "--loss_dir", $ArRun,
  "--tb_log_dir", $ArTb
)
if ($ArWeight) {
  Write-Host "Resuming BrepARG AR from existing weight: $ArWeight"
  $ArArgs += @("--weight", $ArWeight)
} else {
  Write-Host "Training BrepARG AR from scratch on D: $ArRun"
}
Invoke-NativeLogged -LogPath "$ArRun\train_ar.log" $PY @ArArgs

$ArWeight = Get-LatestWeight -Directory $ArRun -BestPattern "abc_ar_vqvae_best_model.pt"
if (!$ArWeight) { throw "Could not find trained BrepARG AR weight under $ArRun" }

if (Test-Path -LiteralPath $Gen) {
  Remove-Item -LiteralPath $Gen -Recurse -Force
}
New-Item -ItemType Directory -Force $Gen | Out-Null
Invoke-NativeLogged -LogPath "$Gen\generate.log" $PY BrepARG\generate_brep.py `
  --dataset_type abc `
  --config BrepARG\config.json `
  --ar_model $ArWeight `
  --se_vqvae $VqvaeWeight `
  --num_samples $GenerateSamples `
  --max_attempts $GenerateMaxAttempts `
  --mode batch `
  --max_length $ArMaxSeqLen `
  --temperature 1.0 `
  --top_p 0.9 `
  --output_dir $Gen `
  --filename_prefix "breparg_same_data_$RunTag" `
  --device cuda `
  --gpu 0

Invoke-Native $PY tools\audit_breparg_baseline_outputs.py `
  --run-dir $Gen `
  --output "$Root\breparg_same_data_quality_summary_$RunTag.json" `
  --markdown-output "$Root\breparg_same_data_quality_summary_$RunTag.md" `
  --manifest-output "$Root\breparg_same_data_quality_manifest_$RunTag.jsonl" `
  --min-faces 12 `
  --min-edges 20 `
  --max-faces 45 `
  --max-edges 120

Copy-Item -Force "$Root\breparg_same_data_quality_summary_$RunTag.json" "$Root\breparg_same_data_quality_summary.json"
Copy-Item -Force "$Root\breparg_same_data_quality_summary_$RunTag.md" "$Root\breparg_same_data_quality_summary.md"
Copy-Item -Force "$Root\breparg_same_data_quality_manifest_$RunTag.jsonl" "$Root\breparg_same_data_quality_manifest.jsonl"

@{
  run_tag = $RunTag
  source_root = $SourceRoot
  source_data = $SourceData
  staged_data = $StagedData
  output_root = $Root
  split = $Split
  dedup_surfaces = $DedupSurfaces
  dedup_edges = $DedupEdges
  vqvae_weight = $VqvaeWeight
  sequence = $Sequence
  ar_weight = $ArWeight
  generated = $Gen
  ar_max_seq_len = $ArMaxSeqLen
  ar_batch_size = $ArBs
  ar_epochs = $ArEpochs
  vqvae_batch_size = $VqvaeBs
  vqvae_epochs = $VqvaeEpochs
  note = "D-drive local RTX 3060-safe same-data BrepARG fallback baseline; use because official ABC weights are incompatible with the current protocol."
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $Manifest

Copy-Item -Force $Manifest "$Root\same_data_breparg_fallback_manifest.json"

Write-Host "BrepARG same-data fallback complete:"
Write-Host "  manifest: $Manifest"
Write-Host "  generated: $Gen"
