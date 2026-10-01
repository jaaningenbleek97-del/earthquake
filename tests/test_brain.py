from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.backtest import _STORE
from eqpred.eval.brain import _alarm, load_journal, run_brain


def _planted(tmp_path, n=2200):
    start = datetime(2016, 1, 1, tzinfo=timezone.utc)
    days = [start + timedelta(days=i) for i in range(n)]
    issue = pd.to_datetime(days, utc=True)
    rng = np.random.default_rng(3)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(size=n)
    quake_rows = []
    catalog_leak = np.zeros(n)
    # Isolated events: raise both knobs across the 30-day look-ahead, not one morning.
    for i in range(80, n - 40, 90):
        a[max(0, i - 20) : i + 1] = 2.8
        b[max(0, i - 20) : i + 1] = 2.8
        catalog_leak[i] = 3.0
        when = days[i] + timedelta(days=3)
        quake_rows.append(
            {
                "observed_at": when,
                "available_at": when + timedelta(hours=6),
                "magnitude": 7.2,
                "lat": float((i * 7) % 170 - 85),
                "lon": float((i * 13) % 350 - 175),
                "event_id": f"ev{i}",
            }
        )
    features = pd.DataFrame(
        {
            "issue_time": issue,
            "knob_a": a,
            "knob_b": b,
            "noise_knob": noise,
            "n_m6_7d": catalog_leak,
        }
    )
    quakes = pd.DataFrame(quake_rows)
    _STORE[str(tmp_path)] = {"features": features, "quakes": quakes}
    return features, quakes


def test_brain_finds_combination_and_learns(tmp_path):
    features, quakes = _planted(tmp_path)
    first = run_brain(root=tmp_path, features=features, quakes=quakes)
    assert first["writes_issued_p"] is False
    assert first["looks_at_all_sources"] is True
    assert first["n_knobs"] == 3
    assert first["journal_runs"] == 1
    assert first["locked_climate_brier"] is not None
    assert first["recent"]["looks_at_all_sources"] is True
    assert first["recent"]["n_features"] > 0
    assert first["quiet_start"]["lookback_days"] == 10
    assert first["rare"]["targets"][0]["magnitude_min"] == 8.0
    assert not (tmp_path / "data" / "models").exists() or not any((tmp_path / "data" / "models").glob("*.json"))
    names = {row["driver"] for row in first["connections"]}
    assert "n_m6_7d" not in names
    assert {"knob_a", "knob_b"} & names
    global_row = next(r for r in first["regions"] if r["region"] == "global")
    assert global_row["lift"] is not None
    assert global_row["lift"] > 0
    noise_imp = [row["importance"] for row in first["connections"] if row["driver"] == "noise_knob"]
    signal_imp = [
        row["importance"]
        for row in first["connections"]
        if row["driver"] in {"knob_a", "knob_b"}
    ]
    if noise_imp and signal_imp:
        assert max(signal_imp) > max(noise_imp)

    second = run_brain(root=tmp_path, features=features, quakes=quakes)
    journal = load_journal(tmp_path)
    assert len(journal) == 2
    assert second["journal_runs"] == 2
    assert "memory.json" in {p.name for p in (tmp_path / "data" / "eval" / "brain").iterdir()}
    assert (tmp_path / "data" / "eval" / "brain.json").exists()


def test_strict_alarm_does_not_flag_a_flat_score():
    days = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    score = np.full(200, 0.12)
    hits = np.zeros(200, dtype=bool)
    hits[10] = True
    train = np.ones(200, dtype=bool)
    train[160:] = False
    hold = ~train
    loose = _alarm(score, hits, days, train, hold, strict=False)
    tight = _alarm(score, hits, days, train, hold, strict=True)
    assert loose["time_in_alarm"] > 0.8
    assert tight["n_trades"] == 0
    assert tight["time_in_alarm"] == 0.0
