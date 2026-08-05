param(
    [Parameter(Mandatory = $true)]
    [string]$ArchiveRoot,

    [Parameter(Mandatory = $true)]
    [string]$WorkspaceRoot,

    [string]$Python = "C:\Users\YU\.conda\envs\brepgen_env\python.exe",
    [string]$LoadFailureAllowlist = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archivePath = (Resolve-Path $ArchiveRoot).Path
$pythonPath = (Resolve-Path $Python).Path
$workspacePath = [System.IO.Path]::GetFullPath($WorkspaceRoot)

if (Test-Path -LiteralPath $workspacePath) {
    $existing = Get-ChildItem -LiteralPath $workspacePath -Force
    if ($existing.Count -gt 0) {
        throw "WorkspaceRoot must be new or empty: $workspacePath"
    }
} else {
    New-Item -ItemType Directory -Path $workspacePath | Out-Null
}

$arguments = @(
    (Join-Path $repoRoot "tools\run_protocol_v5_scaling_ladder.py"),
    "--repo-root", $repoRoot,
    "--archive-root", $archivePath,
    "--workspace-root", $workspacePath,
    "--python", $pythonPath
)
if ($LoadFailureAllowlist) {
    $allowlistPath = (Resolve-Path $LoadFailureAllowlist).Path
    $arguments += @("--load-failure-allowlist", $allowlistPath)
}

$launcherStdout = Join-Path $workspacePath "launcher.stdout.log"
$launcherStderr = Join-Path $workspacePath "launcher.stderr.log"
$process = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $launcherStdout `
    -RedirectStandardError $launcherStderr `
    -WindowStyle Hidden `
    -PassThru

$pidPath = Join-Path $workspacePath "launcher.pid"
[System.IO.File]::WriteAllText($pidPath, [string]$process.Id + [Environment]::NewLine)

Write-Output "Protocol V5 ladder launched."
Write-Output "PID: $($process.Id)"
Write-Output "State: $(Join-Path $workspacePath 'ladder_state.json')"
Write-Output "Launcher stdout: $launcherStdout"
Write-Output "Launcher stderr: $launcherStderr"
Write-Output "GPU is expected only when ladder_state.json has gpu_expected=true."
