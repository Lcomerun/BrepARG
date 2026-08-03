$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location "D:\luolin\V13"

$PY = "C:\Users\YU\.conda\envs\brepgen_env\python.exe"
$BaseRoot = "D:\V13_rootcause_recovery_20260717"
$LongRoot = "$BaseRoot\breparg_same_data_fallback_long_20260720"
$GenRun = "$LongRoot\generated_3060_long_breparg_resume_best_20260726"
$VqvaeBest = "$LongRoot\vqvae_3060_long_vq400_ar300_save20_20260720\same_data_abc\abc_se_vqvae_best.pt"
$ArBest = "D:\luolin\V13\local_runs\breparg_long_ar_resume_best_20260724\ar_epoch127_best_to300\same_data_abc\abc_ar_vqvae_best_model.pt"
$LogDir = "$GenRun\logs"
$QualityDir = "$GenRun\quality_check"
$Manifest = "$GenRun\generation_manifest.json"

New-Item -ItemType Directory -Force $GenRun, $LogDir, $QualityDir | Out-Null

foreach ($Path in @($PY, $VqvaeBest, $ArBest, "BrepARG\config.json")) {
  if (!(Test-Path -LiteralPath $Path)) {
    throw "Missing required generation input: $Path"
  }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:BREPARG_SERIAL_WRITE = "1"
$env:BREPARG_JOINT_OPTIMIZE_DEVICE = "cpu"

function Invoke-NativeLogged {
  param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath,

    [Parameter(Mandatory = $true)]
    [string[]]$Command
  )

  $FilePath = $Command[0]
  $CommandArgs = @()
  if ($Command.Count -gt 1) {
    $CommandArgs = @($Command[1..($Command.Count - 1)])
  }

  $ErrPath = "$LogPath.err"
  $proc = Start-Process `
    -FilePath $FilePath `
    -ArgumentList $CommandArgs `
    -WorkingDirectory "D:\luolin\V13" `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError $ErrPath `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

  if ($proc.ExitCode -ne 0) {
    Write-Host "Command failed with exit code $($proc.ExitCode): $FilePath"
    Get-Content -LiteralPath $LogPath -Tail 80 -ErrorAction SilentlyContinue
    Get-Content -LiteralPath $ErrPath -Tail 80 -ErrorAction SilentlyContinue
    throw "Command failed with exit code $($proc.ExitCode): $FilePath"
  }
}

Write-Host "=== BrepARG generation-only baseline ==="
Write-Host "VQ-VAE: $VqvaeBest"
Write-Host "AR best: $ArBest"
Write-Host "Output: $GenRun"
Write-Host "Started: $((Get-Date).ToString('o'))"

Invoke-NativeLogged -LogPath "$GenRun\generate.log" -Command @(
  $PY,
  "tools\run_breparg_generation_batches.py",
  "--python", $PY,
  "--config", "BrepARG\config.json",
  "--ar-model", $ArBest,
  "--se-vqvae", $VqvaeBest,
  "--output-dir", $GenRun,
  "--target-count", "100",
  "--batch-size", "4",
  "--max-batches", "100",
  "--batch-timeout-sec", "180",
  "--max-attempts-per-batch", "80",
  "--start-seed", "43",
  "--max-length", "1536",
  "--temperature", "1.0",
  "--top-p", "0.9",
  "--filename-prefix", "breparg_same_data_resume_best_20260726",
  "--gpu", "0",
  "--write-timeout", "120",
  "--state", "$GenRun\batch_generation_state.json"
)

Write-Host "=== Validate STEP and render PNG ==="
Invoke-NativeLogged -LogPath "$GenRun\validate_generated.log" -Command @(
  $PY,
  "tools\validate_breparg_generated_directory.py",
  "--run-dir", $GenRun,
  "--manifest-output", "$QualityDir\step_quality_manifest.jsonl",
  "--summary-output", "$QualityDir\step_quality_summary.json",
  "--timeout-sec", "120"
)

Write-Host "=== Audit baseline quality ==="
Invoke-NativeLogged -LogPath "$GenRun\audit.log" -Command @(
  $PY,
  "tools\audit_breparg_baseline_outputs.py",
  "--run-dir", $GenRun,
  "--output", "$LongRoot\breparg_same_data_resume_best_quality_summary_20260726.json",
  "--markdown-output", "$LongRoot\breparg_same_data_resume_best_quality_summary_20260726.md",
  "--manifest-output", "$LongRoot\breparg_same_data_resume_best_quality_manifest_20260726.jsonl",
  "--min-faces", "12",
  "--min-edges", "20",
  "--max-faces", "45",
  "--max-edges", "120"
)

$payload = [ordered]@{
  run_type = "original_breparg_generation_only"
  generated = $GenRun
  vqvae_best = $VqvaeBest
  ar_best = $ArBest
  num_samples_target = 100
  max_attempts = 8000
  max_seq_len = 1536
  temperature = 1.0
  top_p = 0.9
  step_and_png_preserved = $true
  completed_at = (Get-Date).ToString("o")
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $Manifest

Write-Host "GENERATION PIPELINE DONE"
Write-Host "Manifest: $Manifest"
