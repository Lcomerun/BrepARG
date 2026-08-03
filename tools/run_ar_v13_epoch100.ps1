param(
    [string]$RunName = "newscheme_full_v13_ar",
    [string]$OutBase = "D:\luolin\V13\local_runs\ar_training\train_outputs",
    [string]$SequenceSource = "",
    [string]$SplitSource = "",
    [string]$ResumeFrom = "",
    [string]$LearningRate = "5e-4",
    [int]$TargetEpochs = 120,
    [int]$MaxSeqLen = 1024,
    [switch]$NoAutoResume
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\luolin\V13"
$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
$OutDir = Join-Path $OutBase $RunName
$LogDir = "D:\luolin\V13\local_runs\ar_training\logs"
$LogPath = Join-Path $LogDir ("ar_{0}_{1}.log" -f $RunName, (Get-Date -Format "yyyyMMdd_HHmmss"))

$CurrentPid = $PID
$RunningAr = Get-CimInstance Win32_Process |
    Where-Object {
        $_.ProcessId -ne $CurrentPid -and
        $_.CommandLine -match "breparg_improvements\\train.py.*--stage ar"
    }
if ($RunningAr) {
    $Existing = ($RunningAr | Select-Object -ExpandProperty ProcessId) -join ", "
    throw "Another AR training process is already running: PID $Existing"
}

Set-Location -LiteralPath $RepoRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$env:NS_OUTBASE = $OutBase
$env:NS_OUT = $RunName
$env:NS_N = "999999"
$env:NS_AR_EPOCHS = [string]$TargetEpochs
$env:NS_AR_BS = "8"
$env:NS_AR_DMODEL = "256"
$env:NS_AR_LAYERS = "8"
$env:NS_AR_LR = $LearningRate
$env:NS_AR_SAVE_EVERY = "20"
$env:NS_AR_LOG_EVERY_BATCHES = "2000"
$env:NS_AR_MAX_SEQ_LEN = [string]$MaxSeqLen

$SequencePath = Join-Path $OutDir "sequences_fsq_rcm.pkl"
$SplitPath = Join-Path $OutDir "split.pkl"
$LatestCheckpoint = Join-Path $OutDir "ar_latest.pt"
$BestCheckpoint = Join-Path $OutDir "ar_best.pt"

if (-not [string]::IsNullOrWhiteSpace($SequenceSource)) {
    if (-not (Test-Path -LiteralPath $SequenceSource)) {
        throw "Sequence source does not exist: $SequenceSource"
    }
    if (-not (Test-Path -LiteralPath $SequencePath) -or ((Get-Item -LiteralPath $SequencePath).Length -ne (Get-Item -LiteralPath $SequenceSource).Length)) {
        Copy-Item -LiteralPath $SequenceSource -Destination $SequencePath -Force
        Write-Host "Copied sequence input: $SequenceSource -> $SequencePath"
    }
}

if (-not [string]::IsNullOrWhiteSpace($SplitSource)) {
    if (-not (Test-Path -LiteralPath $SplitSource)) {
        throw "Split source does not exist: $SplitSource"
    }
    if (-not (Test-Path -LiteralPath $SplitPath) -or ((Get-Item -LiteralPath $SplitPath).Length -ne (Get-Item -LiteralPath $SplitSource).Length)) {
        Copy-Item -LiteralPath $SplitSource -Destination $SplitPath -Force
        Write-Host "Copied split input: $SplitSource -> $SplitPath"
    }
}

$ResolvedResumeFrom = $ResumeFrom
if ([string]::IsNullOrWhiteSpace($ResolvedResumeFrom) -and -not $NoAutoResume) {
    if (Test-Path -LiteralPath $LatestCheckpoint) {
        $ResolvedResumeFrom = $LatestCheckpoint
    }
}
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    if (-not (Test-Path -LiteralPath $ResumeFrom)) {
        throw "Resume checkpoint does not exist: $ResumeFrom"
    }
    Copy-Item -LiteralPath $ResumeFrom -Destination $LatestCheckpoint -Force
    Copy-Item -LiteralPath $ResumeFrom -Destination $BestCheckpoint -Force
    $ResolvedResumeFrom = $LatestCheckpoint
    Write-Host "Seeded new run from checkpoint: $ResumeFrom"
}
if (-not [string]::IsNullOrWhiteSpace($ResolvedResumeFrom)) {
    if (-not (Test-Path -LiteralPath $ResolvedResumeFrom)) {
        throw "Resume checkpoint does not exist: $ResolvedResumeFrom"
    }
    $env:NS_AR_RESUME_FROM = $ResolvedResumeFrom
    Write-Host "AR resume checkpoint: $ResolvedResumeFrom"
}

if (-not (Test-Path -LiteralPath $SequencePath)) {
    throw "Missing AR input sequence: $SequencePath"
}
if (-not (Test-Path -LiteralPath $SplitPath)) {
    throw "Missing AR split file: $SplitPath"
}

& $Python "breparg_improvements\train.py" --stage ar 2>&1 | Tee-Object -FilePath $LogPath

if ($LASTEXITCODE -ne 0) {
    throw "AR training failed with exit code $LASTEXITCODE. See $LogPath"
}

Write-Host "AR training finished. Check:"
Write-Host "  $OutDir"
Write-Host "  $BestCheckpoint"
Write-Host "  $LatestCheckpoint"
Write-Host "  $(Join-Path $OutDir 'ar_history.jsonl')"
Write-Host "  $LogPath"
