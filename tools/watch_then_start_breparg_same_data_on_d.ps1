param(
  [string]$SafeRoot = "D:\V13_rootcause_recovery_20260717",
  [int]$PollSeconds = 300,
  [int]$MaxGpuMemoryUsedMiB = 2000
)

$ErrorActionPreference = "Stop"

cd "D:\luolin\V13"

$LogDir = "$SafeRoot\logs"
$BrepRoot = "$SafeRoot\breparg_same_data_fallback"
$Log = "$LogDir\watch_then_start_breparg_same_data_on_d.log"
$PidFile = "$LogDir\watch_then_start_breparg_same_data_on_d.pid"
$EvalSummary = "$SafeRoot\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md"
$DfsReport = "$SafeRoot\ar_complex_curved_eval\dfs_teacher_forcing\complex_curved_diagnostics_report.json"
$RcmReport = "$SafeRoot\ar_complex_curved_eval\rcm_teacher_forcing\complex_curved_diagnostics_report.json"
$DfsCkpt = "$SafeRoot\ar_train_outputs\ar_dfs_matched_20260715\ar_best.pt"
$RcmCkpt = "$SafeRoot\ar_train_outputs\ar_rcm_matched_20260715\ar_best.pt"
$BrepScript = "D:\luolin\V13\tools\run_breparg_same_data_fallback_on_d.ps1"
$DoneManifest = "$BrepRoot\same_data_breparg_fallback_manifest.json"

New-Item -ItemType Directory -Force $LogDir, $BrepRoot | Out-Null
$PID | Set-Content -Encoding ASCII $PidFile

function Write-Log {
  param([string]$Message)
  $line = "$(Get-Date -Format s) $Message"
  $line | Tee-Object -FilePath $Log -Append
}

function Get-BlockingProcess {
  Get-CimInstance Win32_Process |
    Where-Object {
      ($_.CommandLine -match 'resume_matched_dfs_rcm_ar_on_d_after_e_recovery\.ps1') -or
      ($_.CommandLine -match 'watch_recovered_training_then_eval_on_d\.ps1') -or
      ($_.CommandLine -match 'eval_recovered_dfs_rcm_ar_on_d\.ps1') -or
      ($_.CommandLine -match 'breparg_improvements\\train.py --stage ar') -or
      ($_.CommandLine -match 'run_breparg_same_data_fallback_on_d\.ps1')
    }
}

function Get-GpuMemoryUsedMiB {
  if (!(Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    return 0
  }
  $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
  if ($LASTEXITCODE -ne 0 -or !$raw) {
    return 999999
  }
  $first = @($raw)[0]
  return [int]($first.ToString().Trim())
}

function Test-FreshReport {
  param([string]$Report, [string]$Checkpoint)
  if (!(Test-Path -LiteralPath $Report) -or !(Test-Path -LiteralPath $Checkpoint)) {
    return $false
  }
  $reportTime = (Get-Item -LiteralPath $Report).LastWriteTimeUtc
  $checkpointTime = (Get-Item -LiteralPath $Checkpoint).LastWriteTimeUtc
  return $reportTime -gt $checkpointTime
}

Write-Log "watcher started; safe_root=$SafeRoot poll_seconds=$PollSeconds max_gpu_memory_used_mib=$MaxGpuMemoryUsedMiB"
Write-Log "will start D-drive BrepARG fallback only after recovered DFS/RCM eval reports exist and GPU is idle"

while ($true) {
  if (Test-Path -LiteralPath $DoneManifest) {
    Write-Log "BrepARG same-data D-drive manifest already exists; nothing to start: $DoneManifest"
    exit 0
  }

  $blocking = @(Get-BlockingProcess)
  $dfsReady = Test-FreshReport -Report $DfsReport -Checkpoint $DfsCkpt
  $rcmReady = Test-FreshReport -Report $RcmReport -Checkpoint $RcmCkpt
  $summaryReady = Test-Path -LiteralPath $EvalSummary
  $gpuMem = Get-GpuMemoryUsedMiB
  Write-Log "blocking_processes=$($blocking.Count) summary_ready=$summaryReady fresh_dfs_eval=$dfsReady fresh_rcm_eval=$rcmReady gpu_mem_mib=$gpuMem"

  if ($blocking.Count -eq 0 -and $summaryReady -and $dfsReady -and $rcmReady -and $gpuMem -le $MaxGpuMemoryUsedMiB) {
    break
  }

  Start-Sleep -Seconds $PollSeconds
}

Write-Log "conditions met; starting D-drive BrepARG same-data fallback"
powershell -NoProfile -ExecutionPolicy Bypass -File $BrepScript 2>&1 |
  Tee-Object -FilePath "$LogDir\run_breparg_same_data_fallback_on_d.out.log"

Write-Log "D-drive BrepARG same-data fallback process finished"
