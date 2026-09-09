from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.backtest import _STORE
from eqpred.eval.brain import load_journal, run_brain


def _planted(tmp_path, n=1400):
    start = datetime(2016, 1, 1, tzinfo=timezone.utc)
    days = [start + timedelta(days=i) for i in range(n)]
    issue = pd.to_datetime(days, utc=True)
    rng = np.random.default_rng(3)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(size=n)
    both = (a > 0.55) & (b > 0.55)
    quake_rows = []
    catalog_leak = np.zeros(n)
    for i, on in enumerate(both):
        if not on or i >= n - 10:
            continue
        when = days[i] + timedelta(days=3)
        catalog_leak[i] = 3.0
        quake_rows.append(
            {
                "observed_at": when,
                "available_at": when + timedelta(hours=6),
                "magnitude": 7.2,
                "lat": 0.0,
                "lon": 0.0,
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
    assert first["n_knobs"] == 3
    assert first["journal_runs"] == 1
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
