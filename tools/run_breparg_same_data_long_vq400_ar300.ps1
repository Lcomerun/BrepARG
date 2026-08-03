$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

cd "D:\luolin\V13"

$PY = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
$BaseRoot = "D:\V13_rootcause_recovery_20260717"
$InputRoot = "$BaseRoot\breparg_same_data_fallback\data_staged"
$OldVqvaeBest = "$BaseRoot\breparg_same_data_fallback\vqvae_3060_safe_len1536_bs4_20260717_d\same_data_abc\abc_se_vqvae_best.pt"
$Root = "$BaseRoot\breparg_same_data_fallback_long_20260720"
$RunTag = "3060_long_vq400_ar300_save20_20260720"
$VqvaeRun = "$Root\vqvae_$RunTag"
$SeqRun = "$Root\sequence_$RunTag"
$ArRun = "$Root\ar_$RunTag"
$GenRun = "$Root\generated_$RunTag"
$VqvaeTb = "$VqvaeRun\tensorboard"
$ArTb = "$ArRun\tensorboard"
$Manifest = "$Root\breparg_same_data_long_manifest.json"

$VqvaeEpochs = "400"
$VqvaeBs = "128"
$VqvaeSaveEvery = "20"
$VqvaeTargetValLoss = "1e-6"
$ArEpochs = "300"
$ArBs = "4"
$ArLr = "5e-5"
$ArMaxSeqLen = "1536"
$ArSaveEvery = "20"
$GenerateSamples = "100"
$GenerateMaxAttempts = "8000"
$Gpu = "0"

New-Item -ItemType Directory -Force `
  $Root, $VqvaeRun, $SeqRun, $ArRun, $GenRun, $VqvaeTb, $ArTb | Out-Null

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

function Get-LatestEpochWeight {
  param([string]$Directory)

  if (!(Test-Path -LiteralPath $Directory)) {
    return $null
  }

  $latest = Get-ChildItem -LiteralPath $Directory -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Extension -in @(".pt", ".pth") -and
      $_.Length -gt 1MB -and
      $_.Name -match "(_epoch_|^epoch_).*[.]pt$"
    } |
    Sort-Object LastWriteTime |
    Select-Object -Last 1 -ExpandProperty FullName

  return $latest
}

function Get-BestWeight {
  param([string]$Directory, [string]$BestName)

  if (!(Test-Path -LiteralPath $Directory)) {
    return $null
  }

  $best = Get-ChildItem -LiteralPath $Directory -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq $BestName -and $_.Length -gt 1MB } |
    Sort-Object LastWriteTime |
    Select-Object -Last 1 -ExpandProperty FullName

  return $best
}

function Get-ResumeWeight {
  param([string]$Directory, [string]$BestName, [string]$Fallback)

  $latest = Get-LatestEpochWeight -Directory $Directory
  if ($latest) { return $latest }

  $best = Get-BestWeight -Directory $Directory -BestName $BestName
  if ($best) { return $best }

  if ($Fallback -and (Test-Path -LiteralPath $Fallback)) { return $Fallback }
  return $null
}

$Split = "$InputRoot\same_data_split.pkl"
$DedupSurfaces = "$InputRoot\deduplicated_surface_source.pkl"
$DedupEdges = "$InputRoot\deduplicated_edge_source.pkl"
foreach ($Path in @($Split, $DedupSurfaces, $DedupEdges, $OldVqvaeBest)) {
  if (!(Test-Path -LiteralPath $Path)) {
    throw "Missing required input: $Path"
  }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:BREPARG_SERIAL_WRITE = "1"
$env:BREPARG_JOINT_OPTIMIZE_DEVICE = "cpu"

$VqvaeSaveDir = "$VqvaeRun\same_data_abc"
New-Item -ItemType Directory -Force $VqvaeSaveDir | Out-Null
if (!(Test-Path -LiteralPath "$VqvaeSaveDir\abc_se_vqvae_best.pt")) {
  Copy-Item -LiteralPath $OldVqvaeBest -Destination "$VqvaeSaveDir\abc_se_vqvae_best.pt" -Force
}

$VqvaeResume = Get-ResumeWeight `
  -Directory $VqvaeRun `
  -BestName "abc_se_vqvae_best.pt" `
  -Fallback $OldVqvaeBest
if (!$VqvaeResume) { throw "Could not find VQ-VAE resume weight" }

Write-Host "=== BrepARG VQ-VAE long training ==="
Write-Host "Resume VQ-VAE from: $VqvaeResume"
Invoke-NativeLogged -LogPath "$VqvaeRun\train_vqvae_long.log" $PY BrepARG\train_vqvae.py `
  --data_list $Split `
  --surface_list $DedupSurfaces `
  --edge_list $DedupEdges `
  --dataset_type abc `
  --batch_size $VqvaeBs `
  --train_epoch $VqvaeEpochs `
  --test_epoch 1 `
  --save_epoch $VqvaeSaveEvery `
  --target_val_loss $VqvaeTargetValLoss `
  --max_face 50 `
  --max_edge 150 `
  --dir_name $VqvaeRun `
  --env same_data_abc `
  --loss_dir $VqvaeRun `
  --tb_log_dir $VqvaeTb `
  --no_aug `
  --weight $VqvaeResume `
  --gpu $Gpu

$VqvaeBest = Get-BestWeight -Directory $VqvaeRun -BestName "abc_se_vqvae_best.pt"
if (!$VqvaeBest) { throw "Could not find long-training VQ-VAE best weight under $VqvaeRun" }

$Sequence = "$SeqRun\breparg_same_data_sequences.pkl"
$NeedSequence = $true
if (Test-Path -LiteralPath $Sequence) {
  $NeedSequence = ((Get-Item -LiteralPath $Sequence).LastWriteTime -lt (Get-Item -LiteralPath $VqvaeBest).LastWriteTime)
}

if ($NeedSequence) {
  Write-Host "=== Build BrepARG sequences from long VQ-VAE ==="
  Remove-Item -LiteralPath $Sequence -Force -ErrorAction SilentlyContinue
  Invoke-NativeLogged -LogPath "$SeqRun\build_sequence.log" $PY BrepARG\2sequence.py `
    --data_list $Split `
    --output_file $Sequence `
    --vqvae_se_weight $VqvaeBest `
    --dataset_type abc `
    --max_face 50 `
    --max_edge 150 `
    --scale 1.0 `
    --aug true `
    --gpu $Gpu
} else {
  Write-Host "Using existing sequence package: $Sequence"
}
if (!(Test-Path -LiteralPath $Sequence)) { throw "Sequence build did not produce $Sequence" }

$ArResume = Get-LatestEpochWeight -Directory $ArRun
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
if ($ArResume) {
  Write-Host "=== Resume BrepARG AR long training ==="
  Write-Host "Resume AR from: $ArResume"
  $ArArgs += @("--weight", $ArResume)
} else {
  Write-Host "=== Train BrepARG AR long baseline from scratch ==="
}
Invoke-NativeLogged -LogPath "$ArRun\train_ar_long.log" $PY @ArArgs

$ArBest = Get-BestWeight -Directory $ArRun -BestName "abc_ar_vqvae_best_model.pt"
if (!$ArBest) { throw "Could not find long-training AR best weight under $ArRun" }

if (Test-Path -LiteralPath $GenRun) {
  Remove-Item -LiteralPath $GenRun -Recurse -Force
}
New-Item -ItemType Directory -Force $GenRun | Out-Null
Write-Host "=== Generate and audit BrepARG long baseline ==="
Invoke-NativeLogged -LogPath "$GenRun\generate.log" $PY BrepARG\generate_brep.py `
  --dataset_type abc `
  --config BrepARG\config.json `
  --ar_model $ArBest `
  --se_vqvae $VqvaeBest `
  --num_samples $GenerateSamples `
  --max_attempts $GenerateMaxAttempts `
  --mode batch `
  --max_length $ArMaxSeqLen `
  --temperature 1.0 `
  --top_p 0.9 `
  --output_dir $GenRun `
  --filename_prefix "breparg_same_data_long_$RunTag" `
  --device cuda `
  --gpu $Gpu

Invoke-NativeLogged -LogPath "$GenRun\validate_generated.log" $PY tools\validate_breparg_generated_directory.py `
  --run-dir $GenRun `
  --manifest-output "$GenRun\quality_check\step_quality_manifest.jsonl" `
  --summary-output "$GenRun\quality_check\step_quality_summary.json" `
  --timeout-sec 120

Invoke-NativeLogged -LogPath "$GenRun\audit.log" $PY tools\audit_breparg_baseline_outputs.py `
  --run-dir $GenRun `
  --output "$Root\breparg_same_data_long_quality_summary.json" `
  --markdown-output "$Root\breparg_same_data_long_quality_summary.md" `
  --manifest-output "$Root\breparg_same_data_long_quality_manifest.jsonl" `
  --min-faces 12 `
  --min-edges 20 `
  --max-faces 45 `
  --max-edges 120

$payload = [ordered]@{
  run_tag = $RunTag
  root = $Root
  input_root = $InputRoot
  vqvae_initial_weight = $OldVqvaeBest
  vqvae_best = $VqvaeBest
  sequence = $Sequence
  ar_best = $ArBest
  generated = $GenRun
  vqvae_epochs = $VqvaeEpochs
  vqvae_save_every = $VqvaeSaveEvery
  vqvae_target_val_loss = $VqvaeTargetValLoss
  ar_epochs = $ArEpochs
  ar_save_every = $ArSaveEvery
  ar_max_seq_len = $ArMaxSeqLen
  ar_batch_size = $ArBs
  ar_learning_rate = $ArLr
  generated_samples = $GenerateSamples
  completed_at = (Get-Date).ToString("o")
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $Manifest
Write-Host "PIPELINE DONE"
Write-Host "Manifest: $Manifest"
