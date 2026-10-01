from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.features.bvalue import bvalue_frame


def test_bvalue_uses_available_at_and_rises_after_many_small_events():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    days = pd.date_range(start, periods=400, freq="D", tz="UTC")
    rows = []
    for i in range(80):
        when = start + timedelta(days=20 + i)
        rows.append(
            {
                "observed_at": when,
                "available_at": when + timedelta(hours=6),
                "magnitude": 4.6 + (i % 5) * 0.2,
                "lon": 176.0,
                "lat": -40.0,
            }
        )
    # A late-reported cluster must not count on the morning it happened.
    late = start + timedelta(days=10)
    rows.append(
        {
            "observed_at": late,
            "available_at": late + timedelta(days=40),
            "magnitude": 7.2,
            "lon": 176.0,
            "lat": -40.0,
        }
    )
    frame = bvalue_frame(pd.DataFrame(rows), days, m0=4.5, min_n=20)
    early = frame.loc[frame["issue_time"] == days[15], "n_fresh_m45_90d"].iloc[0]
    later = frame.loc[frame["issue_time"] == days[120], "n_fresh_m45_90d"].iloc[0]
    assert int(early) == 0
    assert int(later) >= 20
    assert pd.notna(frame.loc[frame["issue_time"] == days[120], "b_fresh_90d"].iloc[0])
    assert pd.notna(frame.loc[frame["issue_time"] == days[120], "b_fresh_hikurangi_90d"].iloc[0])


def test_bvalue_empty_catalog():
    days = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    frame = bvalue_frame(pd.DataFrame(), days)
    assert list(frame["b_fresh_90d"].isna()) == [True] * 5
