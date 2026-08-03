$ErrorActionPreference = "Stop"

$RepoRoot = "D:\luolin\V13"
$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
$RunName = "newscheme_full_vqvae_epoch100"
$OutDir = "E:\ABC\processed\train_outputs\$RunName"
$ShardDir = Join-Path $OutDir "sequence_shards"
$LogDir = "E:\ABC\processed\logs"
$LogPath = Join-Path $LogDir ("sharded_sequence_epoch100_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Set-Location -LiteralPath $RepoRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$env:NS_POOL = "E:\ABC\processed\abc_parsed_full"
$env:NS_OUTBASE = "E:\ABC\processed\train_outputs"
$env:NS_OUT = $RunName
$env:NS_N = "999999"

if (-not $env:NS_SEQ_WORKERS) {
    $env:NS_SEQ_WORKERS = "5"
}

& $Python "tools\run_sharded_sequence.py" `
    --split (Join-Path $OutDir "split.pkl") `
    --checkpoint (Join-Path $OutDir "fsq_vqvae_best.pt") `
    --shard-dir $ShardDir `
    --merge-output (Join-Path $OutDir "sequences_fsq_rcm_sharded_merged.pkl") `
    --summary (Join-Path $ShardDir "_summary.json") `
    --manifest (Join-Path $ShardDir "_manifest.jsonl") `
    --report "D:\luolin\V13\breparg_improvements\repro_outputs\$RunName\train_report.json" `
    --workers ([int]$env:NS_SEQ_WORKERS) `
    --resume `
    2>&1 | Tee-Object -FilePath $LogPath

if ($LASTEXITCODE -ne 0) {
    throw "Sharded sequence failed with exit code $LASTEXITCODE. See $LogPath"
}

Write-Host "Sharded sequence finished. Check:"
Write-Host "  $ShardDir"
Write-Host "  $(Join-Path $ShardDir '_manifest.jsonl')"
Write-Host "  $(Join-Path $ShardDir '_summary.json')"
Write-Host "  $(Join-Path $OutDir 'sequences_fsq_rcm_sharded_merged.pkl')"
Write-Host "  $LogPath"
