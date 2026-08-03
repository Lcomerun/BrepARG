param(
    [string]$VqRoot = "D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\vqvae_3060_long_vq400_ar300_save20_20260720\same_data_abc",
    [string]$StagingRoot = "C:\V13_cleanup_staging_20260723\breparg_vq_periodic",
    [switch]$PreviewOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedEpochs = @(80..380 | Where-Object { $_ % 20 -eq 0 })
$sourceFiles = @(Get-ChildItem -LiteralPath $VqRoot -File -Filter "abc_se_vqvae_epoch_*.pt" |
    Where-Object {
        if ($_.BaseName -match "epoch_(\d+)$") {
            return $expectedEpochs -contains [int]$Matches[1]
        }
        return $false
    } |
    Sort-Object -Property Name)

if ($sourceFiles.Count -ne $expectedEpochs.Count) {
    throw "Expected $($expectedEpochs.Count) VQ periodic checkpoints; found $($sourceFiles.Count)"
}
if (-not (Test-Path -LiteralPath (Join-Path $VqRoot "abc_se_vqvae_best.pt"))) {
    throw "VQ best checkpoint is missing"
}
if (-not (Test-Path -LiteralPath (Join-Path $VqRoot "abc_se_vqvae_epoch_400.pt"))) {
    throw "VQ epoch-400 checkpoint is missing"
}

$sourceBytes = [int64](($sourceFiles | Measure-Object -Property Length -Sum).Sum)
$cDrive = [System.IO.DriveInfo]::new("C")
if (-not $cDrive.IsReady -or $cDrive.DriveFormat -ne "NTFS") {
    throw "C: must be a ready NTFS volume"
}
$requiredFree = $sourceBytes + 5GB
if ($cDrive.AvailableFreeSpace -lt $requiredFree) {
    throw ("C: has {0:N3} GiB free but this staging operation requires at least {1:N3} GiB" -f ($cDrive.AvailableFreeSpace / 1GB), ($requiredFree / 1GB))
}

if ($PreviewOnly) {
    Write-Host "C: staging preview verified"
    Write-Host "  source files : $($sourceFiles.Count)"
    Write-Host ("  source size  : {0:N3} GiB" -f ($sourceBytes / 1GB))
    Write-Host ("  C free       : {0:N3} GiB" -f ($cDrive.AvailableFreeSpace / 1GB))
    Write-Host "  destination  : $StagingRoot"
    exit 0
}

New-Item -ItemType Directory -Path $StagingRoot -Force | Out-Null
$manifestRows = @()

foreach ($source in $sourceFiles) {
    $destination = Join-Path $StagingRoot $source.Name
    $partial = $destination + ".partial"
    $sourceHash = (Get-FileHash -LiteralPath $source.FullName -Algorithm SHA256).Hash

    if (Test-Path -LiteralPath $destination) {
        $destinationItem = Get-Item -LiteralPath $destination
        if ($destinationItem.Length -ne $source.Length) {
            throw "Existing C: staging file has the wrong size: $destination"
        }
        $destinationHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        if ($destinationHash -ne $sourceHash) {
            throw "Existing C: staging file has the wrong hash: $destination"
        }
        Write-Host "Verified existing: $($source.Name)"
    }
    else {
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force -ErrorAction Stop
        }
        Write-Host "Copying: $($source.Name)"
        Copy-Item -LiteralPath $source.FullName -Destination $partial -Force
        $partialItem = Get-Item -LiteralPath $partial
        if ($partialItem.Length -ne $source.Length) {
            throw "Partial C: copy has the wrong size: $partial"
        }
        $partialHash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
        if ($partialHash -ne $sourceHash) {
            throw "Partial C: copy has the wrong hash: $partial"
        }
        Move-Item -LiteralPath $partial -Destination $destination
        $destinationHash = $partialHash
        Write-Host "Verified copy: $($source.Name)"
    }

    $manifestRows += [ordered]@{
        name = $source.Name
        bytes = [int64]$source.Length
        sha256 = $sourceHash
        source = $source.FullName
        staging = $destination
    }
}

$manifestPath = Join-Path $StagingRoot "vq_periodic_staging_manifest_20260723.json"
$manifest = [ordered]@{
    created_at = (Get-Date).ToString("o")
    source_root = $VqRoot
    staging_root = $StagingRoot
    file_count = $manifestRows.Count
    total_bytes = $sourceBytes
    kept_on_d = @("abc_se_vqvae_best.pt", "abc_se_vqvae_epoch_400.pt")
    files = $manifestRows
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$stagedFiles = @(Get-ChildItem -LiteralPath $StagingRoot -File -Filter "abc_se_vqvae_epoch_*.pt")
if ($stagedFiles.Count -ne $sourceFiles.Count) {
    throw "Final C: staging count mismatch: expected $($sourceFiles.Count), found $($stagedFiles.Count)"
}

Write-Host "C: staging completed"
Write-Host "  files       : $($manifestRows.Count)"
Write-Host ("  size        : {0:N3} GiB" -f ($sourceBytes / 1GB))
Write-Host "  manifest    : $manifestPath"
Write-Host ("  C free now  : {0:N3} GiB" -f ([System.IO.DriveInfo]::new("C").AvailableFreeSpace / 1GB))
