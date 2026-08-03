param(
  [string]$SuiteRoot = "E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715",
  [string]$RecoveryRoot = "D:\V13_rootcause_recovery_20260717",
  [string]$StageRoot = "D:\V13_copy_stage_20260717",
  [string]$Output = "D:\V13_rootcause_recovery_20260717\safe_copy_plan_20260717.md",
  [int]$Threads = 16
)

$ErrorActionPreference = "Stop"

function Get-TreeStats {
  param([string]$Path)
  if (!(Test-Path -LiteralPath $Path)) {
    return [pscustomobject]@{
      Path = $Path
      Exists = $false
      Files = 0
      Dirs = 0
      Bytes = 0L
      GB = 0.0
    }
  }
  $item = Get-Item -LiteralPath $Path
  if (!$item.PSIsContainer) {
    $bytes = [int64]$item.Length
    return [pscustomobject]@{
      Path = $Path
      Exists = $true
      Files = 1
      Dirs = 0
      Bytes = $bytes
      GB = [math]::Round($bytes / 1GB, 3)
    }
  }
  $files = 0
  $dirs = 0
  [int64]$bytes = 0
  Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.PSIsContainer) {
      $dirs += 1
    } else {
      $files += 1
      $bytes += [int64]$_.Length
    }
  }
  [pscustomobject]@{
    Path = $Path
    Exists = $true
    Files = $files
    Dirs = $dirs
    Bytes = $bytes
    GB = [math]::Round($bytes / 1GB, 3)
  }
}

function Escape-Md {
  param([string]$Text)
  return ($Text -replace '\|', '\|')
}

$SuiteParent = Split-Path -Parent $SuiteRoot
$SuiteName = Split-Path -Leaf $SuiteRoot
$StageRoot = [System.IO.Path]::GetFullPath($StageRoot)
$Output = [System.IO.Path]::GetFullPath($Output)

$paths = @(
  $SuiteRoot,
  (Join-Path $SuiteRoot "experiments"),
  (Join-Path $SuiteRoot "scripts"),
  (Join-Path $SuiteRoot "experiments\02_dfs_rcm_ordering"),
  (Join-Path $SuiteRoot "experiments\03b_breparg_same_data_training_fallback"),
  (Join-Path $SuiteRoot "experiments\04_breparg_logic_generation_baseline"),
  $RecoveryRoot,
  "D:\luolin\V13\ABC\processed\abc_parsed_full_archives",
  "D:\luolin\V13\ABC\processed\train_outputs\ubuntu"
)

$stats = $paths | ForEach-Object { Get-TreeStats -Path $_ }

$volumeLine = ""
try {
  $vol = Get-Volume -DriveLetter E -ErrorAction Stop
  $volumeLine = "- E: FileSystem='$($vol.FileSystem)', HealthStatus='$($vol.HealthStatus)', OperationalStatus='$($vol.OperationalStatus)'"
} catch {
  $volumeLine = "- E: not currently visible to Get-Volume"
}

$archivePath = Join-Path $StageRoot "$SuiteName.tar.zst"
$recoveryArchivePath = Join-Path $StageRoot "V13_rootcause_recovery_20260717.tar.zst"
$robocopyLog = Join-Path $StageRoot "robocopy_suite_to_stage.log"
$recoveryCopyLog = Join-Path $StageRoot "robocopy_recovery_to_repaired_e.log"

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Safe Copy Plan After E Drive Recovery")
$lines.Add("")
$lines.Add("- Created: $(Get-Date -Format s)")
$lines.Add("- Suite root: '$SuiteRoot'")
$lines.Add("- Recovery root: '$RecoveryRoot'")
$lines.Add("- Stage root: '$StageRoot'")
$lines.Add($volumeLine)
$lines.Add("")
$lines.Add("## Current Recommendation")
$lines.Add("")
$lines.Add("1. Keep active training outputs on 'D:' until 'E:' is repaired.")
$lines.Add("2. For 'E:' -> 'D:' recovery copies, prefer creating archives on 'D:' so the unstable drive is read-only.")
$lines.Add("3. For 'D:' -> 'E:' copy-back, wait until 'chkdsk E: /f' or equivalent repair completes.")
$lines.Add("4. 'robocopy /MT' helps many-file directory copies. For one huge archive, '/J' matters more than '/MT'.")
$lines.Add("5. Do not use '/MIR' or '/PURGE' for this recovery work; both can delete destination files.")
$lines.Add("")
$lines.Add("## Size Inventory")
$lines.Add("")
$lines.Add("| path | exists | files | dirs | GB |")
$lines.Add("| --- | ---: | ---: | ---: | ---: |")
foreach ($row in $stats) {
  $lines.Add("| '$(Escape-Md $row.Path)' | $($row.Exists.ToString().ToLowerInvariant()) | $($row.Files) | $($row.Dirs) | $($row.GB) |")
}
$lines.Add("")
$lines.Add("## Option A: Read E Once, Pack Suite To D")
$lines.Add("")
$lines.Add("Use this if the remaining source is many small files under the suite on 'E:' and you want a more stable transfer artifact on 'D:'.")
$lines.Add("")
$lines.Add("~~~powershell")
$lines.Add("New-Item -ItemType Directory -Force '$StageRoot' | Out-Null")
$lines.Add("tar -acf '$archivePath' -C '$SuiteParent' '$SuiteName'")
$lines.Add("Get-Item '$archivePath' | Select-Object FullName,Length,LastWriteTime")
$lines.Add("~~~")
$lines.Add("")
$lines.Add("## Option B: Multi-thread Directory Copy E To D")
$lines.Add("")
$lines.Add("Use this if you need the directory tree directly instead of a tar/zstd archive. This is restartable and multi-threaded, but slower on many tiny files than a clean archive workflow.")
$lines.Add("")
$lines.Add("~~~powershell")
$lines.Add("New-Item -ItemType Directory -Force '$StageRoot' | Out-Null")
$lines.Add("robocopy '$SuiteRoot' '$StageRoot\$SuiteName' /E /MT:$Threads /J /R:2 /W:2 /FFT /NP /TEE /LOG+:'$robocopyLog'")
$lines.Add('if ($LASTEXITCODE -le 7) { "ROBOCOPY_OK_OR_WARN" } else { throw "robocopy failed with exit code $LASTEXITCODE" }')
$lines.Add("~~~")
$lines.Add("")
$lines.Add("## Option C: Pack Active D Recovery Output")
$lines.Add("")
$lines.Add("Use this after the D-drive DFS/RCM recovery training and evaluation finish. It creates one copy-back artifact without touching 'E:' during training.")
$lines.Add("")
$lines.Add("~~~powershell")
$lines.Add("New-Item -ItemType Directory -Force '$StageRoot' | Out-Null")
$lines.Add("tar -acf '$recoveryArchivePath' -C '$(Split-Path -Parent $RecoveryRoot)' '$(Split-Path -Leaf $RecoveryRoot)'")
$lines.Add("Get-Item '$recoveryArchivePath' | Select-Object FullName,Length,LastWriteTime")
$lines.Add("~~~")
$lines.Add("")
$lines.Add("## Option D: Copy D Recovery Output Back To Repaired E")
$lines.Add("")
$lines.Add("Run this only after 'E:' no longer reports 'Full Repair Needed'.")
$lines.Add("")
$lines.Add("~~~powershell")
$lines.Add("Get-Volume -DriveLetter E | Select-Object DriveLetter,FileSystem,HealthStatus,OperationalStatus")
$lines.Add("robocopy '$RecoveryRoot' '$SuiteRoot\..\D_recovery_20260717' /E /MT:$Threads /J /R:2 /W:2 /FFT /NP /TEE /LOG+:'$recoveryCopyLog'")
$lines.Add('if ($LASTEXITCODE -le 7) { "ROBOCOPY_OK_OR_WARN" } else { throw "robocopy failed with exit code $LASTEXITCODE" }')
$lines.Add("~~~")
$lines.Add("")
$lines.Add("## Training Throughput Note")
$lines.Add("")
$lines.Add("The current AR recovery training reads the sequence package from 'D:' and writes checkpoints to 'D:'. It should not be blocked by slow 'E:' copies unless a separate copy job saturates the same CPU, GPU, or system disk. Keep copy jobs off while the GPU job is active if you see training step time increase.")

$OutputDir = Split-Path -Parent $Output
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$lines -join "`n" | Set-Content -LiteralPath $Output -Encoding UTF8
Write-Host $Output
