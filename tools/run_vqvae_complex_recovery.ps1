param(
    [string]$RepoRoot = "D:\luolin\V13",
    [string]$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe",
    [string]$Pool = "E:\ABC\processed\abc_parsed_full",
    [string]$OutBase = "E:\ABC\processed\train_outputs",
    [string]$RunName = "newscheme_full_vqvae_complex_recovery",
    [int]$Samples = 450000,
    [int]$Epochs = 80,
    [int]$BatchSize = 128,
    [string]$LearningRate = "1e-4",
    [int]$MinEpochs = 30,
    [int]$Patience = 14,
    [string]$MinDelta = "1e-6",
    [double]$ComplexFraction = 0.50,
    [int]$ComplexMinFaces = 12,
    [int]$ComplexMinEdges = 20,
    [double]$CurvedFraction = 0.35,
    [int]$MaxSourceFaces = 50,
    [int]$MaxSourceEdges = 150,
    [double]$ComplexLossWeight = 1.25,
    [double]$CurvedLossWeight = 2.0,
    [double]$CurvedLossThreshold = 0.02,
    [string]$ResumeFrom = "E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\fsq_vqvae_best.pt",
    [string]$HistoryIn = "E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\vqvae_history.json",
    [string]$TargetEpoch = "180"
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$env:NS_POOL = $Pool
$env:NS_OUTBASE = $OutBase
$env:NS_OUT = $RunName
$env:NS_N = "999999"

$env:NS_VQ_SAMPLES = [string]$Samples
$env:NS_VQ_EPOCHS = [string]$Epochs
$env:NS_VQ_TARGET_EPOCH = $TargetEpoch
$env:NS_VQ_BS = [string]$BatchSize
$env:NS_VQ_LR = $LearningRate
$env:NS_VQ_MIN_EPOCHS = [string]$MinEpochs
$env:NS_VQ_PATIENCE = [string]$Patience
$env:NS_VQ_MIN_DELTA = $MinDelta
$env:NS_VQ_MAX_NONFINITE_VAL_EPOCHS = "2"
$env:NS_DISABLE_AMP_VQVAE = "1"

$env:NS_VQ_COMPLEX_FRACTION = [string]$ComplexFraction
$env:NS_VQ_COMPLEX_MIN_FACES = [string]$ComplexMinFaces
$env:NS_VQ_COMPLEX_MIN_EDGES = [string]$ComplexMinEdges
$env:NS_VQ_CURVED_FRACTION = [string]$CurvedFraction
$env:NS_VQ_MAX_SOURCE_FACES = [string]$MaxSourceFaces
$env:NS_VQ_MAX_SOURCE_EDGES = [string]$MaxSourceEdges
$env:NS_VQ_COMPLEX_LOSS_WEIGHT = [string]$ComplexLossWeight
$env:NS_VQ_CURVED_LOSS_WEIGHT = [string]$CurvedLossWeight
$env:NS_VQ_CURVED_LOSS_THRESHOLD = [string]$CurvedLossThreshold

if ($ResumeFrom -ne "") {
    $env:NS_VQ_RESUME_FROM = $ResumeFrom
}
if ($HistoryIn -ne "") {
    $env:NS_VQ_HISTORY_IN = $HistoryIn
}

& $Python "breparg_improvements\train.py" --stage split
& $Python "breparg_improvements\train.py" --stage vqvae

Write-Host "Complex VQ-VAE recovery finished. Check:"
Write-Host "  $OutBase\$RunName\vqvae_history.json"
Write-Host "  $OutBase\$RunName\fsq_vqvae_best.pt"
Write-Host "  $OutBase\$RunName\fsq_vqvae_final.pt"
Write-Host "  $RepoRoot\breparg_improvements\repro_outputs\$RunName\train_report.json"
