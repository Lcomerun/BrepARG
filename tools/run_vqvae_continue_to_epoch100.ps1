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

$env:NS_VQ_SAMPLES = "300000"
$env:NS_VQ_EPOCHS = "60"
$env:NS_VQ_TARGET_EPOCH = "100"
$env:NS_VQ_BS = "128"
$env:NS_VQ_LR = "1e-4"
$env:NS_VQ_MIN_EPOCHS = "45"
$env:NS_VQ_PATIENCE = "12"
$env:NS_VQ_MIN_DELTA = "1e-6"
$env:NS_VQ_MAX_NONFINITE_VAL_EPOCHS = "2"
$env:NS_DISABLE_AMP_VQVAE = "1"
$env:NS_VQ_RESUME_FROM = "E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\fsq_vqvae_best.pt"
$env:NS_VQ_HISTORY_IN = "E:\ABC\processed\train_outputs\newscheme_full_vqvae_stable\vqvae_history.json"

& $Python "breparg_improvements\train.py" --stage split
& $Python "breparg_improvements\train.py" --stage vqvae

Write-Host "Epoch-100 VQ-VAE continuation finished. Check:"
Write-Host "  E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\vqvae_history.json"
Write-Host "  E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\fsq_vqvae_best.pt"
Write-Host "  E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\fsq_vqvae_final.pt"
Write-Host "  D:\luolin\V13\breparg_improvements\repro_outputs\newscheme_full_vqvae_epoch100\train_report.json"
