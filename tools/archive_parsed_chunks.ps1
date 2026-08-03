$ErrorActionPreference = "Stop"

$RepoRoot = "D:\luolin\V13"
$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
$ParsedRoot = "E:\ABC\processed\abc_parsed_full"
$ArchiveRoot = "E:\ABC\processed\abc_parsed_full_archives"
$LogDir = "E:\ABC\processed\logs"
$Manifest = Join-Path $ArchiveRoot "_manifest.jsonl"
$LogPath = Join-Path $LogDir ("archive_parsed_chunks_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Set-Location -LiteralPath $RepoRoot
New-Item -ItemType Directory -Force -Path $ArchiveRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Python "tools\archive_parsed_chunks.py" `
    --parsed-root $ParsedRoot `
    --archive-root $ArchiveRoot `
    --manifest $Manifest `
    --chunks "all" `
    --resume `
    --compression "deflate" `
    2>&1 | Tee-Object -FilePath $LogPath

if ($LASTEXITCODE -ne 0) {
    throw "Parsed archive failed with exit code $LASTEXITCODE. See $LogPath"
}

Write-Host "Parsed archive finished. Check:"
Write-Host "  $ArchiveRoot"
Write-Host "  $Manifest"
Write-Host "  $LogPath"
