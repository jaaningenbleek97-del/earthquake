from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.backtest import _STORE, paper_trade


def test_paper_trade_scores_closed_weeks_now(tmp_path):
    start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    days = [start + timedelta(days=i) for i in range(900)]
    issue = pd.to_datetime(days, utc=True)
    signal = np.zeros(len(days))
    quake_rows = []
    for i in range(20, len(days) - 10, 25):
        signal[i] = 1.0
        when = days[i] + timedelta(days=3)
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
    rng = np.random.default_rng(1)
    features = pd.DataFrame(
        {
            "issue_time": issue,
            "proton_spike": signal,
            "noise_knob": rng.normal(size=len(days)),
        }
    )
    _STORE[str(tmp_path)] = {"features": features, "quakes": pd.DataFrame(quake_rows)}
    good = paper_trade("proton_spike", region_id="global", share=0.2, root=tmp_path)
    assert good["n_days"] > 100
    assert good["hit_rate"] is not None
    assert good["hit_rate"] > good["base_rate"]
    assert good["note"].startswith("Uses weeks that already closed")
    noise = paper_trade("noise_knob", region_id="global", share=0.2, root=tmp_path)
    assert abs(noise["lift"] or 0) < good["lift"]
