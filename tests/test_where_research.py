from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.where_research import (
    choose_locked_patch,
    interface_fresh_hits,
    known_event_rate,
)


def _days(n=6):
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return pd.DatetimeIndex([start + timedelta(days=i) for i in range(n)])


def test_rate_ignores_events_not_yet_reported():
    days = _days(5)
    observed = np.array(["2020-01-01"], dtype="datetime64[ns]")
    available = np.array(["2020-01-04"], dtype="datetime64[ns]")
    rate = known_event_rate(observed, available, days, lookback_days=10, horizon_days=30)
    assert rate[1] == 0.0
    assert rate[3] > 0.0


def test_locked_patch_prefers_the_quiet_still_patch():
    slip = np.array([5.0, 0.1, 4.0])
    quakes = np.array([8.0, 0.0, 3.0])
    assert choose_locked_patch(slip, quakes) == 1


def test_two_rates_use_only_windows_that_have_closed():
    from eqpred.eval.research_cards import expanding_two_rate

    hits = np.array([1, 0, 1, 0, 1], dtype=float)
    gate = np.array([True, False, True, False, True])
    on_p, off_p = expanding_two_rate(hits, gate, horizon_days=2)
    assert np.isnan(on_p[1])
    assert on_p[2] == 1.0
    assert off_p[3] == 0.0
    assert on_p[4] == 1.0


def test_interface_hits_drop_deep_slab(monkeypatch):
    from eqpred.eval import where_research as wr

    rows = []
    for depth, mag in ((20.0, 5.4), (90.0, 5.6)):
        rows.append(
            {
                "lat": -39.0,
                "lon": 177.0,
                "magnitude": mag,
                "depth_km": depth,
                "observed_at": "2020-01-02T00:00:00Z",
                "available_at": "2020-01-02T00:00:00Z",
            }
        )
    catalog = pd.DataFrame(rows)
    catalog["observed_at"] = pd.to_datetime(catalog["observed_at"], utc=True)
    catalog["available_at"] = pd.to_datetime(catalog["available_at"], utc=True)
    days = pd.DatetimeIndex([datetime(2020, 1, 1, tzinfo=timezone.utc)])
    hits = interface_fresh_hits(catalog, days, mag=5.0, horizon_days=5)
    assert bool(hits[0])
    deep = catalog.loc[catalog["depth_km"] > 60]
    monkeypatch.setattr(wr, "_box_events", lambda *_a, **_k: deep)
    assert not bool(interface_fresh_hits(catalog, days, mag=5.0, horizon_days=5)[0])
