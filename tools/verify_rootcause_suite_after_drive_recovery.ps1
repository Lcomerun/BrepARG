param(
  [string]$Root = "E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715",
  [string]$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
)

$ErrorActionPreference = "Continue"

function Section($Name) {
  Write-Host ""
  Write-Host "===== $Name ====="
}

Section "drive"
Get-PSDrive -PSProvider FileSystem | Format-Table Name,Root,Used,Free -AutoSize
if (!(Test-Path -LiteralPath $Root)) {
  Write-Host "ROOT_MISSING: $Root"
  exit 2
}
Write-Host "ROOT_OK: $Root"

Section "recent storage events"
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddHours(-6)} -ErrorAction SilentlyContinue |
  Where-Object {
    $_.ProviderName -match 'disk|ntfs|exfat|stor|partmgr|volmgr|Kernel-PnP|UASPStor|USBSTOR' -or
    $_.Message -match 'disk|volume|E:|removed|surprise|reset|I/O|device|Delayed Write'
  } |
  Select-Object TimeCreated,ProviderName,Id,LevelDisplayName,Message |
  Select-Object -First 60 |
  Format-List

Section "expected files"
$Paths = @(
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\matched_ar_train_len1536_bs4.out.log",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_dfs_matched_20260715\ar_history.jsonl",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_dfs_matched_20260715\ar_best.pt",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_dfs_matched_20260715\ar_latest.pt",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_rcm_matched_20260715\ar_history.jsonl",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_rcm_matched_20260715\ar_best.pt",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_rcm_matched_20260715\ar_latest.pt",
  "$Root\scripts\05_audit_suite_status.ps1",
  "$Root\scripts\02b_train_dfs_rcm_ar.ps1",
  "$Root\scripts\02d_watch_matched_ar_then_eval.ps1",
  "$Root\scripts\03c_watch_then_start_breparg_3060_safe.ps1"
)
foreach ($Path in $Paths) {
  if (Test-Path -LiteralPath $Path) {
    Get-Item -LiteralPath $Path | Select-Object FullName,Length,LastWriteTime | Format-List
  } else {
    Write-Host "MISSING: $Path"
  }
}

Section "log tails"
$Logs = @(
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\matched_ar_train_len1536_bs4.out.log",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_dfs_matched_20260715\ar_train.log",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_rcm_matched_20260715\ar_train.log",
  "$Root\experiments\02_dfs_rcm_ordering\ar_train_outputs\watch_matched_ar_then_eval.log",
  "$Root\experiments\03b_breparg_same_data_training_fallback\watch_then_start_breparg_3060_safe.log"
)
foreach ($Log in $Logs) {
  Write-Host "--- $Log ---"
  if (Test-Path -LiteralPath $Log) {
    Get-Content -LiteralPath $Log -Tail 40
  } else {
    Write-Host "MISSING"
  }
}

Section "checkpoint torch/finite check"
if (!(Test-Path -LiteralPath $Python)) {
  Write-Host "PYTHON_MISSING: $Python"
} else {
  $env:V13_RECOVERY_ROOT = $Root
  @'
import json
import os
from pathlib import Path

import torch

root = Path(os.environ["V13_RECOVERY_ROOT"])
checkpoints = [
    root / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_dfs_matched_20260715/ar_best.pt",
    root / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_dfs_matched_20260715/ar_latest.pt",
    root / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_rcm_matched_20260715/ar_best.pt",
    root / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_rcm_matched_20260715/ar_latest.pt",
]
rows = []
for path in checkpoints:
    row = {"path": str(path), "exists": path.exists()}
    if path.exists():
        try:
            ck = torch.load(path, map_location="cpu")
            row.update({
                "load_ok": True,
                "epoch": ck.get("epoch"),
                "train_ce": ck.get("train_ce"),
                "val_ce": ck.get("val_ce"),
                "best_val_ce": ck.get("best_val_ce"),
            })
            state = ck.get("model_state_dict") or ck.get("state_dict") or {}
            bad = []
            total_float = 0
            for name, value in state.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    total_float += 1
                    if not torch.isfinite(value).all():
                        bad.append(name)
                        break
            row["finite_model"] = not bad
            row["float_tensors"] = total_float
            row["first_bad_tensor"] = bad[0] if bad else None
        except Exception as exc:
            row.update({"load_ok": False, "error": f"{type(exc).__name__}: {exc}"})
    rows.append(row)
print(json.dumps(rows, indent=2, ensure_ascii=True))
'@ | & $Python -
}

Section "suite audit"
$Audit = "$Root\scripts\05_audit_suite_status.ps1"
if (Test-Path -LiteralPath $Audit) {
  powershell -NoProfile -ExecutionPolicy Bypass -File $Audit
  Get-Content -LiteralPath "$Root\suite_status.md" -TotalCount 140
} else {
  Write-Host "AUDIT_SCRIPT_MISSING: $Audit"
}

Section "processes"
$Needles = @("02b_train_dfs_rcm_ar", "02d_watch_matched", "03c_watch", "breparg_improvements\train.py --stage ar")
foreach ($Needle in $Needles) {
  Write-Host "--- $Needle ---"
  Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*$Needle*" } |
    Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine |
    Format-List
}
