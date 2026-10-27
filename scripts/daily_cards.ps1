# Daily cards: refresh the last 10 days, rebuild features, then write today's calls.
# Registered as the Windows task "eqpred daily cards" (07:00 every day).
# A card written more than a day late does not count as a forecast, so this run
# is the morning job. It does not replace frozen recipe files or old ledger rows.
# Sundays the weekly tape rebuild starts at 03:00 and is finished before this.

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$logs = Join-Path $repo "data\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs ("daily_cards_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmm"))
$env:PYTHONUNBUFFERED = "1"
$eqpred = Join-Path $repo ".venv\Scripts\eqpred.exe"
$now = (Get-Date).ToUniversalTime()
$start = $now.AddDays(-10).ToString("yyyy-MM-dd")
$end = $now.AddDays(1).ToString("yyyy-MM-dd")
$today = $now.ToString("yyyy-MM-dd")

function Invoke-Step([string]$Name, [string[]]$Arguments) {
    "=== $Name started $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
    & $eqpred @Arguments 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    "=== $Name finished with exit code $code at $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
    return $code
}

if ((Invoke-Step "watchdog" @("watchdog")) -ne 0) {
    "required feeds are stale, so no new card was written" | Out-File -FilePath $log -Append -Encoding utf8
    exit 2
}
Invoke-Step "ingest core feeds" @("ingest", "--wave", "1", "--start", $start, "--end", $end) | Out-Null
Invoke-Step "ingest GeoNet catalog" @("ingest", "--wave", "2", "--only", "geonet.fdsn", "--start", $start, "--end", $end) | Out-Null
if ((Invoke-Step "features" @("features")) -ne 0) { exit 1 }
if ((Invoke-Step "mainshock" @("mainshock")) -ne 0) { exit 1 }
Invoke-Step "predict today" @("predict", "--start", $today, "--end", $end) | Out-Null
exit 0
