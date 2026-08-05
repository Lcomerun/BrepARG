[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProtocolDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [string]$PythonExecutable = "C:\Users\YU\.conda\envs\brepgen_env\python.exe",

    [string]$Seeds = "0,1,2",

    [int]$Epochs = 100
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ProtocolDir = [System.IO.Path]::GetFullPath($ProtocolDir)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$PythonExecutable = [System.IO.Path]::GetFullPath($PythonExecutable)
$Orchestrator = Join-Path $RepoRoot "tools\run_protocol_v4_100epoch_cohort.py"

foreach ($RequiredPath in @($ProtocolDir, $PythonExecutable, $Orchestrator)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path does not exist: $RequiredPath"
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$LauncherStdout = Join-Path $OutputRoot "launcher.stdout.log"
$LauncherStderr = Join-Path $OutputRoot "launcher.stderr.log"
$PidPath = Join-Path $OutputRoot "cohort_launcher.pid"

$Arguments = @(
    $Orchestrator,
    "--repo-root", $RepoRoot,
    "--protocol-dir", $ProtocolDir,
    "--output-root", $OutputRoot,
    "--python", $PythonExecutable,
    "--seeds", $Seeds,
    "--epochs", [string]$Epochs,
    "--train-cap", "12000",
    "--val-cap", "4637",
    "--batch-size", "128",
    "--learning-rate", "3e-4",
    "--min-parent-coverage", "0.9"
)

$Process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $LauncherStdout `
    -RedirectStandardError $LauncherStderr `
    -WindowStyle Hidden `
    -PassThru

[System.IO.File]::WriteAllText($PidPath, ([string]$Process.Id + [Environment]::NewLine))
Start-Sleep -Seconds 2
$Process.Refresh()
if ($Process.HasExited) {
    $ErrorTail = ""
    if (Test-Path -LiteralPath $LauncherStderr) {
        $ErrorTail = (Get-Content -LiteralPath $LauncherStderr -Tail 20) -join [Environment]::NewLine
    }
    throw "Protocol V4 cohort launcher exited immediately with code $($Process.ExitCode). $ErrorTail"
}

[ordered]@{
    status = "STARTED"
    launcher_pid = $Process.Id
    output_root = $OutputRoot
    state = (Join-Path $OutputRoot "cohort_state.json")
    seed0_stdout = (Join-Path $OutputRoot "seed0\stdout.log")
    launcher_stdout = $LauncherStdout
    launcher_stderr = $LauncherStderr
} | ConvertTo-Json

