# Weekly refresh: recent catalog and core feeds, rebuilt features, then both strategy tapes.
# Registered as the Windows task "eqpred weekly refresh" (Sundays 03:00). Safe to run by hand.
# It never writes issued cards, frozen files, or ledger rows.

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$logs = Join-Path $repo "data\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs ("weekly_refresh_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmm"))
$env:PYTHONUNBUFFERED = "1"
$eqpred = Join-Path $repo ".venv\Scripts\eqpred.exe"
$now = (Get-Date).ToUniversalTime()
$start = $now.AddDays(-45).ToString("yyyy-MM-dd")
$end = $now.AddDays(1).ToString("yyyy-MM-dd")

function Invoke-Step([string]$Name, [string[]]$Arguments) {
    "=== $Name started $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
    & $eqpred @Arguments 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    "=== $Name finished with exit code $code at $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
    return $code
}

Invoke-Step "ingest core feeds" @("ingest", "--wave", "1", "--start", $start, "--end", $end) | Out-Null
Invoke-Step "ingest GeoNet catalog" @("ingest", "--wave", "2", "--only", "geonet.fdsn", "--start", $start, "--end", $end) | Out-Null
if ((Invoke-Step "features" @("features")) -ne 0) { exit 1 }
$v1 = Invoke-Step "strategy tape, answer key version 1" @("tape", "--label", "1")
$v2 = Invoke-Step "strategy tape, answer key version 2" @("tape", "--label", "2")
if ($v1 -ne 0 -or $v2 -ne 0) { exit 1 }
exit 0
