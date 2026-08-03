param(
  [string]$SuiteRoot = "E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715",
  [string]$SafeRoot = "D:\V13_rootcause_recovery_20260717",
  [string]$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe",
  [string]$VqvaeCheckpoint = "D:\luolin\V13\ABC\processed\train_outputs\ubuntu\fsq_vqvae_best.pt",
  [string]$ArchiveRoot = "D:\luolin\V13\ABC\processed\abc_parsed_full_archives",
  [int]$ArMaxSeqLen = 1536
)

$ErrorActionPreference = "Stop"

cd "D:\luolin\V13"

$SafeOutBase = "$SafeRoot\ar_train_outputs"
$EvalRoot = "$SafeRoot\ar_complex_curved_eval"
$LogDir = "$SafeRoot\logs"

$DfsRun = "$SafeOutBase\ar_dfs_matched_20260715"
$RcmRun = "$SafeOutBase\ar_rcm_matched_20260715"
$DfsSequence = "$DfsRun\sequences_fsq_rcm.pkl"
$RcmSequence = "$RcmRun\sequences_fsq_rcm.pkl"
$DfsAr = "$DfsRun\ar_best.pt"
$RcmAr = "$RcmRun\ar_best.pt"

foreach ($Path in @($Python, $VqvaeCheckpoint, $ArchiveRoot, $DfsSequence, $RcmSequence, $DfsAr, $RcmAr)) {
  if (!(Test-Path -LiteralPath $Path)) {
    throw "Missing required path: $Path"
  }
}

New-Item -ItemType Directory -Force $EvalRoot, $LogDir | Out-Null

function Write-EvalLog {
  param([string]$Message)
  $line = "$(Get-Date -Format s) $Message"
  $line | Tee-Object -FilePath "$LogDir\eval_recovered_dfs_rcm_ar_on_d.log" -Append
}

Write-EvalLog "safe root=$SafeRoot"
Write-EvalLog "suite root=$SuiteRoot"
Write-EvalLog "vqvae=$VqvaeCheckpoint"
Write-EvalLog "archive root=$ArchiveRoot"
Write-EvalLog "dfs ar=$DfsAr"
Write-EvalLog "rcm ar=$RcmAr"
Write-EvalLog "ar max seq len=$ArMaxSeqLen"

$env:V13_DFS_AR = $DfsAr
$env:V13_RCM_AR = $RcmAr
$env:V13_FINITE_CHECK_OUT = "$EvalRoot\checkpoint_finite_check.json"
@'
import json
import os
from pathlib import Path

import torch

rows = []
for label, env_name in [("dfs", "V13_DFS_AR"), ("rcm", "V13_RCM_AR")]:
    path = Path(os.environ[env_name])
    row = {"label": label, "path": str(path), "exists": path.exists()}
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
    float_tensors = 0
    for name, value in state.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            float_tensors += 1
            if not torch.isfinite(value).all():
                bad.append(name)
                break
    row["finite_model"] = not bad
    row["float_tensors"] = float_tensors
    row["first_bad_tensor"] = bad[0] if bad else None
    rows.append(row)

text = json.dumps(rows, ensure_ascii=True, indent=2)
Path(os.environ["V13_FINITE_CHECK_OUT"]).write_text(text + "\n", encoding="utf-8")
print(text)
if any(not row.get("finite_model") for row in rows):
    raise SystemExit("non-finite AR checkpoint detected")
'@ | & $Python -

Write-EvalLog "starting dfs teacher-forcing diagnostic"
& $Python tools\complex_curved_diagnostics.py `
  --sequence $DfsSequence `
  --vqvae-checkpoint $VqvaeCheckpoint `
  --ar-checkpoint $DfsAr `
  --archive-root $ArchiveRoot `
  --output-dir "$EvalRoot\dfs_teacher_forcing" `
  --split val `
  --max-samples 50 `
  --max-scan 5000 `
  --max-seq-len $ArMaxSeqLen `
  --complex-min-faces 12 `
  --complex-min-edges 20 `
  --curved-threshold 0.02 `
  --max-source-faces 50 `
  --max-source-edges 150 `
  --device auto `
  --skip-reconstruction

Write-EvalLog "starting rcm teacher-forcing diagnostic"
& $Python tools\complex_curved_diagnostics.py `
  --sequence $RcmSequence `
  --vqvae-checkpoint $VqvaeCheckpoint `
  --ar-checkpoint $RcmAr `
  --archive-root $ArchiveRoot `
  --output-dir "$EvalRoot\rcm_teacher_forcing" `
  --split val `
  --max-samples 50 `
  --max-scan 5000 `
  --max-seq-len $ArMaxSeqLen `
  --complex-min-faces 12 `
  --complex-min-edges 20 `
  --curved-threshold 0.02 `
  --max-source-faces 50 `
  --max-source-edges 150 `
  --device auto `
  --skip-reconstruction

& $Python tools\summarize_complex_curved_diagnostics.py `
  --report "dfs=$EvalRoot\dfs_teacher_forcing\complex_curved_diagnostics_report.json" `
  --report "rcm=$EvalRoot\rcm_teacher_forcing\complex_curved_diagnostics_report.json" `
  --output "$EvalRoot\dfs_vs_rcm_teacher_forcing_summary.md"

Write-EvalLog "finished recovered DFS/RCM complex-curved evaluation"
Write-Host "$EvalRoot\dfs_vs_rcm_teacher_forcing_summary.md"
