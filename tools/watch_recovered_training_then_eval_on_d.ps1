param(
  [int]$TrainingPid = 34092,
  [string]$SafeRoot = "D:\V13_rootcause_recovery_20260717",
  [string]$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe",
  [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"

cd "D:\luolin\V13"

$LogDir = "$SafeRoot\logs"
$LogPath = "$LogDir\watch_recovered_training_then_eval_on_d.log"
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Write-WatchLog {
  param([string]$Message)
  $line = "$(Get-Date -Format s) $Message"
  $line | Tee-Object -FilePath $LogPath -Append
}

function Test-FiniteCheckpoint {
  param([string]$Path)
  if (!(Test-Path -LiteralPath $Path)) {
    return $false
  }
  $env:V13_CHECKPOINT_TO_TEST = $Path
  $code = @'
import os
from pathlib import Path

import torch

path = Path(os.environ["V13_CHECKPOINT_TO_TEST"])
ck = torch.load(path, map_location="cpu")
state = ck.get("model_state_dict") or ck.get("state_dict") or {}
for name, value in state.items():
    if torch.is_tensor(value) and torch.is_floating_point(value):
        if not torch.isfinite(value).all():
            raise SystemExit(f"nonfinite tensor {name}")
print({"epoch": ck.get("epoch"), "best_val_ce": ck.get("best_val_ce")})
'@
  $code | & $Python - | Out-Null
  return ($LASTEXITCODE -eq 0)
}

$DfsBest = "$SafeRoot\ar_train_outputs\ar_dfs_matched_20260715\ar_best.pt"
$RcmBest = "$SafeRoot\ar_train_outputs\ar_rcm_matched_20260715\ar_best.pt"
$EvalSummary = "$SafeRoot\ar_complex_curved_eval\dfs_vs_rcm_teacher_forcing_summary.md"

Write-WatchLog "watching training pid=$TrainingPid safe_root=$SafeRoot"

while ($true) {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$TrainingPid" -ErrorAction SilentlyContinue
  $dfsReady = Test-Path -LiteralPath $DfsBest
  $rcmReady = Test-Path -LiteralPath $RcmBest
  Write-WatchLog "training_alive=$([bool]$proc) dfs_best=$dfsReady rcm_best=$rcmReady"
  if (!$proc) {
    break
  }
  Start-Sleep -Seconds $PollSeconds
}

if (!(Test-Path -LiteralPath $DfsBest)) {
  throw "DFS best checkpoint missing after training process exited: $DfsBest"
}
if (!(Test-Path -LiteralPath $RcmBest)) {
  throw "RCM best checkpoint missing after training process exited: $RcmBest"
}
if (!(Test-FiniteCheckpoint -Path $DfsBest)) {
  throw "DFS best checkpoint failed finite/load check: $DfsBest"
}
if (!(Test-FiniteCheckpoint -Path $RcmBest)) {
  throw "RCM best checkpoint failed finite/load check: $RcmBest"
}

Write-WatchLog "training finished and checkpoints are finite; starting D-drive evaluation"
powershell -NoProfile -ExecutionPolicy Bypass -File tools\eval_recovered_dfs_rcm_ar_on_d.ps1 2>&1 |
  Tee-Object -FilePath "$LogDir\eval_recovered_dfs_rcm_ar_on_d.out.log"

if (!(Test-Path -LiteralPath $EvalSummary)) {
  throw "Evaluation summary missing after eval script: $EvalSummary"
}

Write-WatchLog "evaluation finished summary=$EvalSummary"
