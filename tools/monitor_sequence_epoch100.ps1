$ErrorActionPreference = "Stop"

$LogDir = "E:\ABC\processed\logs"
$RunName = "newscheme_full_vqvae_epoch100"
$OutDir = "E:\ABC\processed\train_outputs\$RunName"
$Report = "D:\luolin\V13\breparg_improvements\repro_outputs\$RunName\train_report.json"
$LogPath = Join-Path $LogDir ("sequence_epoch100_monitor_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-MonitorLog {
    param([string]$Message)
    Add-Content -LiteralPath $LogPath -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"), $Message)
}

Write-MonitorLog "monitor_start run=$RunName out=$OutDir report=$Report"

while ($true) {
    $seqProc = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "train.py.*--stage sequence" } |
        Select-Object -First 1

    if (-not $seqProc) {
        Write-MonitorLog "sequence_process_missing; monitor_stop"
        break
    }

    $proc = Get-Process -Id $seqProc.ProcessId -ErrorAction SilentlyContinue
    if ($proc) {
        Write-MonitorLog ("process pid={0} cpu_seconds={1:n1} working_set_gb={2:n2} private_gb={3:n2} start={4}" -f `
            $proc.Id, $proc.CPU, ($proc.WorkingSet64 / 1GB), ($proc.PrivateMemorySize64 / 1GB), $proc.StartTime)
    }

    try {
        $gpu = & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
        Write-MonitorLog "gpu $gpu"
    } catch {
        Write-MonitorLog ("gpu_query_failed {0}" -f $_.Exception.Message)
    }

    if (Test-Path -LiteralPath $OutDir) {
        $files = Get-ChildItem -LiteralPath $OutDir -Force |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 8 |
            ForEach-Object { "{0}:{1}:{2}" -f $_.Name, $_.Length, $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss") }
        Write-MonitorLog ("files {0}" -f ($files -join " | "))
    }

    if (Test-Path -LiteralPath $Report) {
        try {
            $json = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
            $stageNames = $json.stages.PSObject.Properties.Name -join ","
            $sequence = $json.stages.sequence
            if ($sequence) {
                Write-MonitorLog ("report stages={0} sequence_status={1} sequences={2} out_of_vocab={3}" -f `
                    $stageNames, $sequence.status, $sequence.sequences, $sequence.out_of_vocab)
            } else {
                Write-MonitorLog ("report stages={0} sequence_status=pending" -f $stageNames)
            }
        } catch {
            Write-MonitorLog ("report_parse_failed {0}" -f $_.Exception.Message)
        }
    }

    Start-Sleep -Seconds 300
}
