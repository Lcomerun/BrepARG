$ErrorActionPreference = "Stop"

$RepoRoot = "D:\luolin\V13"
$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"

Set-Location -LiteralPath $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$env:NS_POOL = "E:\ABC\processed\abc_parsed_full"
$env:NS_OUTBASE = "E:\ABC\processed\train_outputs"
$env:NS_OUT = "newscheme_full_vqvae_epoch100"
$env:NS_N = "999999"

& $Python "breparg_improvements\train.py" --stage sequence

Write-Host "Sequence stage finished. Check:"
Write-Host "  E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm.pkl"
Write-Host "  D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_vqvae_epoch100\train_report.json"
