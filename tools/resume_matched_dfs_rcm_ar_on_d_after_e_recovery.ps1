param(
  [string]$SuiteRoot = "E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715",
  [string]$SafeRoot = "D:\V13_rootcause_recovery_20260717",
  [string]$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
)

$ErrorActionPreference = "Stop"

cd "D:\luolin\V13"

$SourceRoot = "$SuiteRoot\experiments\02_dfs_rcm_ordering"
$SourceOut = "$SourceRoot\ar_train_outputs"
$SafeOutBase = "$SafeRoot\ar_train_outputs"
$LogDir = "$SafeRoot\logs"
$DFSSequence = "$SourceRoot\sequences_fsq_dfs.pkl"
$RCMSequence = "$SourceRoot\sequences_fsq_rcm.pkl"
$DFSResume = "$SourceOut\ar_dfs_matched_20260715\ar_best.pt"

foreach ($Path in @($Python, $DFSSequence, $RCMSequence, $DFSResume)) {
  if (!(Test-Path -LiteralPath $Path)) {
    throw "Missing required path: $Path"
  }
}

New-Item -ItemType Directory -Force $SafeOutBase, $LogDir | Out-Null

$AR_EPOCHS = "60"
$AR_BS = "4"
$AR_DMODEL = "256"
$AR_LAYERS = "8"
$AR_LR = "5e-5"
$AR_MAX_SEQ_LEN = "1536"
$AR_SAVE_EVERY = "5"
$AR_LOG_EVERY_BATCHES = "500"

function Write-RecoveryLog {
  param([string]$Message)
  $line = "$(Get-Date -Format s) $Message"
  $line | Tee-Object -FilePath "$LogDir\resume_matched_dfs_rcm_ar_on_d.log" -Append
}

function Train-OrderingAr {
  param(
    [string]$Name,
    [string]$SequencePath,
    [string]$ResumeFrom = ""
  )

  $Run = "ar_$Name`_matched_20260715"
  $RunDir = "$SafeOutBase\$Run"
  New-Item -ItemType Directory -Force $RunDir | Out-Null

  Copy-Item -Force -LiteralPath $SequencePath -Destination "$RunDir\sequences_fsq_rcm.pkl"

  & $Python tools\preflight_ar_training.py `
    --sequence "$RunDir\sequences_fsq_rcm.pkl" `
    --output "$RunDir\ar_preflight.json" `
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
  $env:NS_OUTBASE = $SafeOutBase
  $env:NS_OUT = $Run
  $env:NS_N = "999999"
  $env:NS_AR_EPOCHS = $AR_EPOCHS
  $env:NS_AR_BS = $AR_BS
  $env:NS_AR_DMODEL = $AR_DMODEL
  $env:NS_AR_LAYERS = $AR_LAYERS
  $env:NS_AR_LR = $AR_LR
  $env:NS_AR_MAX_SEQ_LEN = $AR_MAX_SEQ_LEN
  $env:NS_AR_SAVE_EVERY = $AR_SAVE_EVERY
  $env:NS_AR_LOG_EVERY_BATCHES = $AR_LOG_EVERY_BATCHES
  if ($ResumeFrom) {
    $env:NS_AR_RESUME_FROM = $ResumeFrom
  } else {
    Remove-Item Env:\NS_AR_RESUME_FROM -ErrorAction SilentlyContinue
  }

  Write-RecoveryLog "starting $Name run=$Run out=$RunDir resume_from=$ResumeFrom"
  & $Python breparg_improvements\train.py --stage ar 2>&1 |
    Tee-Object -FilePath "$RunDir\ar_train.log"
  Write-RecoveryLog "finished $Name run=$Run"

  $ReportSrc = "breparg_improvements\repro_outputs\$Run\train_report.json"
  if (Test-Path -LiteralPath $ReportSrc) {
    Copy-Item -Force -LiteralPath $ReportSrc -Destination "$RunDir\train_report.json"
  }
}

Write-RecoveryLog "safe recovery root=$SafeRoot"
Write-RecoveryLog "source suite root=$SuiteRoot"
Write-RecoveryLog "DFS resume checkpoint=$DFSResume"

Train-OrderingAr -Name "dfs" -SequencePath $DFSSequence -ResumeFrom $DFSResume
Train-OrderingAr -Name "rcm" -SequencePath $RCMSequence

Write-RecoveryLog "all matched DFS/RCM AR recovery training finished"
