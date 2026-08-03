param(
    [string]$OutputDir = 'dist',
    [string]$PackageName = 'v13_server_ready_20260710.zip'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutputRoot = Join-Path $RepoRoot $OutputDir
$PackagePath = Join-Path $OutputRoot $PackageName
$StageRoot = Join-Path $OutputRoot 'v13_server_ready_stage'

if (Test-Path $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $StageRoot | Out-Null
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$includeFiles = @(
    'AGENTS.md',
    'PLANS.md',
    'README.md',
    'PROJECT_INDEX.md',
    'environment.server.yml',
    '.gitignore'
)

$includeGlobs = @(
    'docs/*.md',
    'plans/v13_workspace_cleanup_and_server_packaging_execplan.md',
    'plans/v13_sharded_dataset_execplan.md',
    'plans/v13_generation_quality_recovery_execplan.md',
    'breparg_improvements/*.py',
    'breparg_improvements/README.md',
    'breparg_improvements/HANDOFF.md',
    'breparg_improvements/docs/*.md',
    'tools/*.py',
    'tools/*.sh',
    'tools/*.ps1',
    'tests/*.py',
    'BrepARG/*.py',
    'BrepARG/*.json',
    'BrepARG/requirements.txt',
    'papers/aaai_v13/render_step_directory.py',
    'papers/aaai_v13/render_selected_steps.py',
    'papers/aaai_v13/README.md',
    'papers/aaai_v13/evidence_map.md'
)

function Copy-RepoFile {
    param([string]$RelativePath)
    $Source = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path $Source -PathType Leaf)) {
        return
    }
    $Target = Join-Path $StageRoot $RelativePath
    $TargetDir = Split-Path -Parent $Target
    if (-not (Test-Path $TargetDir)) {
        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $Target -Force
}

foreach ($file in $includeFiles) {
    Copy-RepoFile $file
}

foreach ($glob in $includeGlobs) {
    Get-ChildItem -Path (Join-Path $RepoRoot $glob) -File -ErrorAction SilentlyContinue | ForEach-Object {
        $relative = $_.FullName.Substring($RepoRoot.Length + 1)
        Copy-RepoFile $relative
    }
}

$manifest = [ordered]@{
    created = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    repo_root = "$RepoRoot"
    package = "$PackagePath"
    excluded_heavy_paths = @(
        'ABC/',
        'local_runs/',
        'processed_local/',
        'tmp/',
        'local_reports/',
        'breparg_improvements/repro_outputs/',
        'papers/aaai_v13/latex/rendered/'
    )
    server_start = 'docs/SERVER_START_HERE.md'
    environment = 'environment.server.yml'
    parsed_shards_upload_source = 'C:\V13_abc_parsed_shards'
    parsed_shards_server_target = '/workspace/ABC/processed/abc_parsed_shards'
    autodl_repo_root = '/root/autodl-tmp/workplace'
    autodl_parsed_shards = '/root/autodl-tmp/workplace/V13_abc_parsed_shards'
    autodl_patch_shards = '/root/autodl-tmp/ABC/processed/vqvae_patch_shards'
    autodl_train_outputs = '/root/autodl-tmp/ABC/processed/train_outputs'
    autodl_helper = 'tools/autodl_vqvae_scratch.sh'
    rtx5090_diagnose = 'REPO_ROOT=$(pwd) V13_SKIP_INSTALL=1 V13_REQUIRE_CUDA=1 bash tools/server_bootstrap.sh'
    rtx5090_repair = 'REPO_ROOT=$(pwd) V13_FORCE_CU128=1 V13_REQUIRE_CUDA=1 bash tools/server_bootstrap.sh'
}

($manifest | ConvertTo-Json -Depth 4) | Set-Content -Path (Join-Path $StageRoot 'SERVER_PACKAGE_MANIFEST.json') -Encoding UTF8

if (Test-Path $PackagePath) {
    Remove-Item -LiteralPath $PackagePath -Force
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Zip = [System.IO.Compression.ZipFile]::Open($PackagePath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    Get-ChildItem -Path $StageRoot -File -Recurse | Sort-Object FullName | ForEach-Object {
        $Relative = $_.FullName.Substring($StageRoot.Length + 1)
        $EntryName = $Relative.Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Zip,
            $_.FullName,
            $EntryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $Zip.Dispose()
}

$RequiredEntries = @(
    'README.md',
    'PROJECT_INDEX.md',
    'environment.server.yml',
    'SERVER_PACKAGE_MANIFEST.json',
    'docs/SERVER_START_HERE.md',
    'tools/server_bootstrap.sh',
    'tools/autodl_vqvae_scratch.sh',
    'tools/run_vqvae_from_patch_shards.sh',
    'tools/build_server_package.ps1',
    'breparg_improvements/train.py',
    'breparg_improvements/sharded_data.py',
    'breparg_improvements/vqvae_sampling.py',
    'tools/build_vqvae_patch_shards.py',
    'tools/verify_parsed_shards.py'
)

$ForbiddenPrefixes = @(
    'ABC/',
    'local_runs/',
    'processed_local/',
    'tmp/',
    'local_reports/',
    'dist/'
)

$ZipRead = [System.IO.Compression.ZipFile]::OpenRead($PackagePath)
try {
    $EntryNames = @($ZipRead.Entries | ForEach-Object { $_.FullName })
    $BackslashEntries = @($EntryNames | Where-Object { $_.Contains('\') })
    if ($BackslashEntries.Count -gt 0) {
        throw "Zip contains Windows-style paths: $($BackslashEntries[0])"
    }

    $NonAsciiEntries = @($EntryNames | Where-Object { $_ -match '[^\x00-\x7F]' })
    if ($NonAsciiEntries.Count -gt 0) {
        throw "Zip contains non-ASCII entry names: $($NonAsciiEntries[0])"
    }

    foreach ($entry in $RequiredEntries) {
        if ($EntryNames -notcontains $entry) {
            throw "Required package entry missing: $entry"
        }
    }

    foreach ($prefix in $ForbiddenPrefixes) {
        $ForbiddenEntry = $EntryNames | Where-Object { $_.StartsWith($prefix) } | Select-Object -First 1
        if ($ForbiddenEntry) {
            throw "Forbidden generated/heavy path packaged: $ForbiddenEntry"
        }
    }

    $SchemeDocEntry = $EntryNames | Where-Object { $_ -like '*方案*' } | Select-Object -First 1
    if ($SchemeDocEntry) {
        throw "Non-server Chinese design note should not be packaged: $SchemeDocEntry"
    }
}
finally {
    $ZipRead.Dispose()
}

$PackageHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PackagePath).Hash.ToLowerInvariant()
$HashPath = "$PackagePath.sha256"
Set-Content -Path $HashPath -Encoding ASCII -Value "$PackageHash  $PackageName"
Remove-Item -LiteralPath $StageRoot -Recurse -Force

Write-Host "Package written: $PackagePath"
Write-Host "SHA256 written: $HashPath"
