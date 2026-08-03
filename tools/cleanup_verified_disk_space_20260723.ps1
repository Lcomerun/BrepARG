param(
    [switch]$Execute,
    [string]$VqArchiveRoot = "C:\V13_cleanup_staging_20260723\breparg_vq_periodic"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path -LiteralPath "D:\luolin\V13").Path.TrimEnd("\")
$expectedSequenceHash = "25F9BD5A5E9E06502168C5F92011B3C2ED89D5ED5330C94974D04BB01DDD17A1"
$canonicalSequence = Join-Path $workspace "ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm.pkl"
$sequenceLinks = @(
    (Join-Path $workspace "local_runs\ar_training\train_outputs\newscheme_full_v13_ar\sequences_fsq_rcm.pkl"),
    (Join-Path $workspace "local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr1e5\sequences_fsq_rcm.pkl"),
    (Join-Path $workspace "local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr2e5\sequences_fsq_rcm.pkl"),
    (Join-Path $workspace "local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e5\sequences_fsq_rcm.pkl"),
    (Join-Path $workspace "local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\sequences_fsq_rcm.pkl")
)

$vqRoot = "D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\vqvae_3060_long_vq400_ar300_save20_20260720\same_data_abc"
$rootCauseSuite = Join-Path $workspace "local_runs\complex_curved_rootcause_suite_20260715"
$fsqCapacityRoot = Join-Path $rootCauseSuite "experiments\01a_train_fsq_capacity_candidate"
$rebuildableDirectories = @(
    (Join-Path $fsqCapacityRoot "vq_patch_shards_full"),
    (Join-Path $fsqCapacityRoot "vq_patch_shards_medium_0000_0009"),
    (Join-Path $fsqCapacityRoot "vq_patch_shards_smoke"),
    (Join-Path $fsqCapacityRoot "vq_patch_shards_direct_smoke"),
    (Join-Path $fsqCapacityRoot "parsed_shards_smoke"),
    (Join-Path $fsqCapacityRoot "tmp_extracted_parsed_smoke")
)
$cacheDirectories = @(
    (Join-Path $workspace "tests\__pycache__"),
    (Join-Path $workspace "tools\__pycache__"),
    (Join-Path $workspace "BrepARG\__pycache__"),
    (Join-Path $workspace "breparg_improvements\__pycache__"),
    (Join-Path $workspace ".pytest_cache")
)

function Get-TreeBytes {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)

    if (-not (Test-Path -LiteralPath $LiteralPath)) {
        return [int64]0
    }
    $item = Get-Item -LiteralPath $LiteralPath
    if (-not $item.PSIsContainer) {
        return [int64]$item.Length
    }
    $files = @(Get-ChildItem -LiteralPath $LiteralPath -File -Recurse -Force -ErrorAction Stop)
    if ($files.Count -eq 0) {
        return [int64]0
    }
    $measurement = $files | Measure-Object -Property Length -Sum
    return [int64]$measurement.Sum
}

function Assert-ChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$LiteralPath,
        [Parameter(Mandatory = $true)][string]$AllowedRoot
    )

    $fullPath = [System.IO.Path]::GetFullPath($LiteralPath).TrimEnd("\")
    $fullRoot = [System.IO.Path]::GetFullPath($AllowedRoot).TrimEnd("\")
    if ($fullPath -eq $fullRoot -or -not $fullPath.StartsWith($fullRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside approved root: $fullPath"
    }
}

$driveBefore = [System.IO.DriveInfo]::new("D")
$freeBefore = [int64]$driveBefore.AvailableFreeSpace
$plannedBytes = [int64]0

if (-not (Test-Path -LiteralPath $canonicalSequence)) {
    throw "Canonical sequence is missing: $canonicalSequence"
}
$canonicalHash = (Get-FileHash -LiteralPath $canonicalSequence -Algorithm SHA256).Hash
if ($canonicalHash -ne $expectedSequenceHash) {
    throw "Canonical sequence hash changed: $canonicalHash"
}

$sequenceBackups = @()
foreach ($linkPath in $sequenceLinks) {
    Assert-ChildPath -LiteralPath $linkPath -AllowedRoot $workspace
    $backupPath = $linkPath + ".pre_hardlink_20260723"
    if (-not (Test-Path -LiteralPath $linkPath)) {
        throw "Expected sequence link is missing: $linkPath"
    }
    if (-not (Test-Path -LiteralPath $backupPath)) {
        throw "Expected rollback copy is missing: $backupPath"
    }
    $link = Get-Item -LiteralPath $linkPath
    if ($link.LinkType -ne "HardLink") {
        throw "Sequence path is not a hard link: $linkPath"
    }
    if ((Get-FileHash -LiteralPath $linkPath -Algorithm SHA256).Hash -ne $expectedSequenceHash) {
        throw "Sequence hard-link hash mismatch: $linkPath"
    }
    if ((Get-FileHash -LiteralPath $backupPath -Algorithm SHA256).Hash -ne $expectedSequenceHash) {
        throw "Sequence rollback-copy hash mismatch: $backupPath"
    }
    $sequenceBackups += $backupPath
    $plannedBytes += (Get-Item -LiteralPath $backupPath).Length
}

if (-not (Test-Path -LiteralPath (Join-Path $vqRoot "abc_se_vqvae_best.pt"))) {
    throw "VQ best checkpoint is missing"
}
if (-not (Test-Path -LiteralPath (Join-Path $vqRoot "abc_se_vqvae_epoch_400.pt"))) {
    throw "VQ epoch-400 checkpoint is missing"
}
$periodicVqCheckpoints = @(Get-ChildItem -LiteralPath $vqRoot -File -Filter "abc_se_vqvae_epoch_*.pt" |
    Where-Object {
        if ($_.BaseName -match "epoch_(\d+)$") {
            $epoch = [int]$Matches[1]
            return $epoch -ge 80 -and $epoch -le 380
        }
        return $false
    } |
    Sort-Object -Property Name)
if ($periodicVqCheckpoints.Count -ne 16) {
    throw "Expected 16 VQ periodic checkpoints from epoch 80 through 380; found $($periodicVqCheckpoints.Count)"
}
$vqArchiveManifestPath = Join-Path $VqArchiveRoot "vq_periodic_staging_manifest_20260723.json"
if (-not (Test-Path -LiteralPath $vqArchiveManifestPath)) {
    throw "Verified C: VQ staging manifest is missing. Run tools\stage_vq_periodic_to_c_20260723.ps1 first."
}
$vqArchiveManifest = Get-Content -LiteralPath $vqArchiveManifestPath -Raw | ConvertFrom-Json
if ([int]$vqArchiveManifest.file_count -ne 16 -or @($vqArchiveManifest.files).Count -ne 16) {
    throw "C: VQ staging manifest does not contain exactly 16 files"
}
foreach ($checkpoint in $periodicVqCheckpoints) {
    Assert-ChildPath -LiteralPath $checkpoint.FullName -AllowedRoot $vqRoot
    $archivedCheckpoint = Join-Path $VqArchiveRoot $checkpoint.Name
    if (-not (Test-Path -LiteralPath $archivedCheckpoint)) {
        throw "C: VQ staging file is missing: $archivedCheckpoint"
    }
    $archivedItem = Get-Item -LiteralPath $archivedCheckpoint
    if ($archivedItem.Length -ne $checkpoint.Length) {
        throw "C: VQ staging size mismatch: $archivedCheckpoint"
    }
    $sourceHash = (Get-FileHash -LiteralPath $checkpoint.FullName -Algorithm SHA256).Hash
    $archiveHash = (Get-FileHash -LiteralPath $archivedCheckpoint -Algorithm SHA256).Hash
    if ($sourceHash -ne $archiveHash) {
        throw "C: VQ staging hash mismatch: $archivedCheckpoint"
    }
    $plannedBytes += $checkpoint.Length
}

$archiveRoot = Join-Path $workspace "ABC\processed\abc_parsed_full_archives"
$parsedArchives = @(Get-ChildItem -LiteralPath $archiveRoot -File -Filter "abc_*_parsed.zip")
if ($parsedArchives.Count -ne 100) {
    throw "Expected 100 parsed ABC ZIP archives before deleting rebuildable intermediates; found $($parsedArchives.Count)"
}
foreach ($directory in $rebuildableDirectories) {
    Assert-ChildPath -LiteralPath $directory -AllowedRoot $fsqCapacityRoot
    $plannedBytes += Get-TreeBytes -LiteralPath $directory
}
foreach ($directory in $cacheDirectories) {
    Assert-ChildPath -LiteralPath $directory -AllowedRoot $workspace
    $plannedBytes += Get-TreeBytes -LiteralPath $directory
}

Write-Host "Verified cleanup plan"
Write-Host "  sequence rollback copies : $($sequenceBackups.Count)"
Write-Host "  VQ periodic checkpoints  : $($periodicVqCheckpoints.Count)"
Write-Host "  verified VQ backup root  : $VqArchiveRoot"
Write-Host "  rebuildable directories  : $($rebuildableDirectories.Count)"
Write-Host "  cache directories        : $($cacheDirectories.Count)"
Write-Host ("  planned recovery         : {0:N3} GiB" -f ($plannedBytes / 1GB))
Write-Host ("  D free before            : {0:N3} GiB" -f ($freeBefore / 1GB))

if (-not $Execute) {
    Write-Host "Preview only. Re-run with -Execute after reviewing the verified plan."
    exit 0
}

foreach ($backupPath in $sequenceBackups) {
    Remove-Item -LiteralPath $backupPath -Force -ErrorAction Stop
}
foreach ($checkpoint in $periodicVqCheckpoints) {
    Remove-Item -LiteralPath $checkpoint.FullName -Force -ErrorAction Stop
}
foreach ($directory in $rebuildableDirectories) {
    if (Test-Path -LiteralPath $directory) {
        Remove-Item -LiteralPath $directory -Recurse -Force -ErrorAction Stop
    }
}
foreach ($directory in $cacheDirectories) {
    if (Test-Path -LiteralPath $directory) {
        Remove-Item -LiteralPath $directory -Recurse -Force -ErrorAction Stop
    }
}

foreach ($linkPath in $sequenceLinks) {
    if (-not (Test-Path -LiteralPath $linkPath)) {
        throw "Sequence hard link disappeared after cleanup: $linkPath"
    }
    if ((Get-FileHash -LiteralPath $linkPath -Algorithm SHA256).Hash -ne $expectedSequenceHash) {
        throw "Sequence hard link changed after cleanup: $linkPath"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $vqRoot "abc_se_vqvae_best.pt"))) {
    throw "VQ best checkpoint disappeared after cleanup"
}
if (-not (Test-Path -LiteralPath (Join-Path $vqRoot "abc_se_vqvae_epoch_400.pt"))) {
    throw "VQ epoch-400 checkpoint disappeared after cleanup"
}

$driveAfter = [System.IO.DriveInfo]::new("D")
$freeAfter = [int64]$driveAfter.AvailableFreeSpace
Write-Host "Cleanup completed"
Write-Host ("  D free after             : {0:N3} GiB" -f ($freeAfter / 1GB))
Write-Host ("  measured recovery        : {0:N3} GiB" -f (($freeAfter - $freeBefore) / 1GB))
