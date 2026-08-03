param(
  [string]$Root = "E:\V13_rootcause_20260715\complex_curved_rootcause_suite_20260715"
)

$ErrorActionPreference = "Continue"

Write-Host "===== filesystem drives ====="
Get-PSDrive -PSProvider FileSystem | Format-Table Name,Root,Used,Free -AutoSize

Write-Host ""
Write-Host "===== disks ====="
Get-Disk | Select-Object Number,FriendlyName,SerialNumber,OperationalStatus,HealthStatus,PartitionStyle,Size |
  Format-Table -AutoSize

Write-Host ""
Write-Host "===== volumes ====="
Get-Volume | Select-Object DriveLetter,FileSystemLabel,FileSystem,DriveType,HealthStatus,OperationalStatus,SizeRemaining,Size |
  Format-Table -AutoSize

Write-Host ""
if (Test-Path -LiteralPath $Root) {
  Write-Host "RECOVERED: $Root"
  exit 0
}

if (Test-Path -LiteralPath "E:\") {
  Write-Host "E_DRIVE_PRESENT_BUT_ROOT_MISSING: $Root"
  Get-ChildItem -LiteralPath "E:\" -Force | Select-Object Name,LastWriteTime,Length | Format-Table -AutoSize
  exit 1
}

Write-Host "E_DRIVE_MISSING"
exit 2
