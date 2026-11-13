"""Exercise the real daily orchestration with a local fake CLI, never live data."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def daily_runner(tmp_path):
    shell = shutil.which("powershell.exe")
    if not shell:
        pytest.skip("Windows PowerShell required")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    fake = tmp_path / "fake.ps1"
    fake.write_text('''$step = $args[0]
$step | Out-File -Append -Encoding utf8 (Join-Path $PSScriptRoot "calls.txt")
if ($step -eq "predict") { $global:LASTEXITCODE = 7 }
elseif ($step -eq "watchdog" -and -not (Test-Path (Join-Path $PSScriptRoot "refreshed"))) { $global:LASTEXITCODE = 2 }
else { $global:LASTEXITCODE = 0 }
if ($step -eq "ingest") { New-Item -ItemType File -Force (Join-Path $PSScriptRoot "refreshed") | Out-Null }
''', encoding="utf-8")
    source = (Path(__file__).parents[1] / "scripts/daily_cards.ps1").read_text()
    source = source.replace('$eqpred = Join-Path $repo ".venv\\Scripts\\eqpred.exe"', '$eqpred = Join-Path $repo "fake.ps1"')
    target = scripts / "daily_cards.ps1"
    target.write_text(source, encoding="utf-8")

    def run(fresh=False, refresh=True, predict_exit=7):
        fake_source = fake.read_text(encoding="utf-8")
        fake_source = fake_source.replace('$global:LASTEXITCODE = 7', f'$global:LASTEXITCODE = {predict_exit}')
        if not refresh:
            fake_source = fake_source.replace('if ($step -eq "ingest")', 'if ($false)')
        fake.write_text(fake_source, encoding="utf-8")
        if fresh:
            (tmp_path / "refreshed").touch()
        proc = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(target)], capture_output=True, text=True)
        calls = (tmp_path / "calls.txt").read_text(encoding="utf-8-sig").splitlines()
        return proc, calls
    return run


def test_daily_propagates_predict_failure(daily_runner):
    proc, calls = daily_runner(fresh=True)
    assert calls[-1] == "predict"
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_daily_refreshes_before_staleness_gate(daily_runner):
    proc, calls = daily_runner()
    assert calls[:3] == ["ingest", "ingest", "watchdog"]
    assert "predict" in calls


def test_daily_still_blocks_issuance_if_refresh_does_not_recover(daily_runner):
    proc, calls = daily_runner(refresh=False)
    assert proc.returncode == 2
    assert calls == ["ingest", "ingest", "watchdog"]


def test_daily_returns_success_only_after_successful_predict(daily_runner):
    proc, calls = daily_runner(predict_exit=0)
    assert calls == ["ingest", "ingest", "watchdog", "features", "mainshock", "predict"]
    assert proc.returncode == 0
