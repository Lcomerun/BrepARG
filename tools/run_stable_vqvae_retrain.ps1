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
$env:NS_OUT = "newscheme_full_vqvae_stable"
$env:NS_N = "999999"

$env:NS_VQ_SAMPLES = "300000"
$env:NS_VQ_EPOCHS = "40"
$env:NS_VQ_BS = "128"
$env:NS_VQ_MIN_EPOCHS = "12"
$env:NS_VQ_PATIENCE = "8"
$env:NS_VQ_MIN_DELTA = "1e-5"
$env:NS_VQ_MAX_NONFINITE_VAL_EPOCHS = "2"
$env:NS_DISABLE_AMP_VQVAE = "1"

& $Python "breparg_improvements\train.py" --stage split
& $Python "breparg_improvements\train.py" --stage vqsweep
& $Python "breparg_improvements\train.py" --stage vqvae

Write-Host "Stable VQ-VAE retrain finished. Check:"
Write-Host "  E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\vqvae_history.json"
Write-Host "  D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_vqvae_stable\train_report.json"
