import numpy as np
import pandas as pd

from datetime import datetime, timedelta, timezone

from eqpred.eval.foreshock import (
    _sequence_pool,
    catalog_columns,
    catalog_foreshock_frame,
    run_foreshock,
)
from eqpred.eval.hunt import driver_family
from eqpred.regions import REGIONS


def test_foreshock_columns_are_catalog_only():
    days = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    quakes = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "observed_at": pd.to_datetime(
                ["2020-01-10T00:00:00Z", "2020-01-11T00:00:00Z", "2020-01-20T00:00:00Z"],
                utc=True,
            ),
            "available_at": pd.to_datetime(
                ["2020-01-10T06:00:00Z", "2020-01-11T06:00:00Z", "2020-01-20T06:00:00Z"],
                utc=True,
            ),
            "magnitude": [4.2, 4.8, 5.1],
            "lon": [176.0, 176.1, 176.2],
            "lat": [-39.0, -39.1, -39.2],
        }
    )
    feat = catalog_foreshock_frame(quakes, pd.Series(days), versions=None)
    cols = catalog_columns(feat)
    assert cols
    assert all(driver_family(c) == "catalog" for c in cols)
    assert "n_m4_7d" in feat.columns
    assert "n_m4_7d_hikurangi" in feat.columns
    assert "seq_n_hikurangi" in feat.columns
    assert feat["n_m4_7d"].iloc[12] >= 1


def test_sequence_recipe_pool_is_local_and_global_is_coarse():
    names = [
        "seq_n_max",
        "seq_active_any__lag1",
        "seq_n_hikurangi",
        "seq_b_pos_delta_hikurangi__lag1",
        "seq_n_alpine",
        "n_m4_7d",
    ]
    usable = list(range(len(names)))
    assert _sequence_pool(names, usable, "global") == [0, 1]
    assert _sequence_pool(names, usable, "hikurangi") == [2, 3]
    assert _sequence_pool(names, usable, "alpine") == [4]


def test_foreshock_does_not_write_issued_p(tmp_path):
    rng = np.random.default_rng(3)
    days = pd.date_range("2014-01-01", periods=2200, freq="D", tz="UTC")
    n4 = rng.integers(0, 8, size=len(days)).astype(float)
    features = pd.DataFrame({"issue_time": days, "n_m4_7d": n4})
    origins = days[::18][:80]
    quakes = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(len(origins))],
            "observed_at": origins,
            "available_at": origins,
            "magnitude": [7.1] * len(origins),
            "lon": [0.0] * len(origins),
            "lat": [0.0] * len(origins),
        }
    )
    report = run_foreshock(
        train_end="2018-12-31",
        holdout_start="2019-01-01",
        root=tmp_path,
        features=features,
        quakes=quakes,
    )
    assert report["writes_issued_p"] is False
    assert report["track"] == "catalog_precursor"
    assert (tmp_path / "data" / "eval" / "foreshock.json").exists()
    for row in report.get("week") or []:
        assert "foreshock_raw_p" in row
        assert row.get("track") == "catalog_precursor"
        assert "sequence" in row
    for row in report.get("regions") or []:
        assert row.get("issued_admission", {}).get("applied") is False
        assert "sequence_conditional" in row
        assert "csep" in row
        assert "fresh_rate_train" in row["sequence_conditional"]
    assert "n_beats_etas" in report


def test_foreshock_box_counts_stay_inside_bbox():
    days = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    quakes = pd.DataFrame(
        {
            "event_id": ["in", "out"],
            "observed_at": pd.to_datetime(["2020-01-05T00:00:00Z", "2020-01-05T00:00:00Z"], utc=True),
            "available_at": pd.to_datetime(["2020-01-05T06:00:00Z", "2020-01-05T06:00:00Z"], utc=True),
            "magnitude": [4.6, 4.6],
            "lon": [176.0, 0.0],
            "lat": [-39.0, 0.0],
        }
    )
    feat = catalog_foreshock_frame(quakes, pd.Series(days), versions=None)
    assert int(feat["n_m4_7d_hikurangi"].iloc[8]) == 1
    assert int(feat["n_m4_7d_alpine"].iloc[8]) == 0
    assert REGIONS["hikurangi"]["kind"] == "bbox"


def _plant(day0, offsets, mags, lon=176.2, lat=-39.15):
    rows = []
    for off, mag in zip(offsets, mags, strict=True):
        when = day0 + timedelta(days=float(off))
        rows.append(
            {
                "event_id": f"{when.isoformat()}-{mag}",
                "observed_at": when,
                "available_at": when,
                "magnitude": mag,
                "lon": lon + 0.01 * off,
                "lat": lat + 0.005 * off,
                "depth_km": 12.0,
            }
        )
    return rows


def test_planted_accelerating_cluster_lifts_and_flat_swarm_does_not(tmp_path):
    start = datetime(2014, 1, 1, tzinfo=timezone.utc)
    days = pd.date_range(start, periods=2800, freq="D", tz="UTC")
    rows = []
    accel_offs = [0.0, 2.0, 3.4, 4.4, 5.1, 5.6, 5.9, 6.1]
    accel_mags = [3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3]
    flat_offs = np.linspace(0.0, 6.0, 8)
    flat_mags = [3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3]
    origins = [d for d in days[40::80] if d + pd.Timedelta(days=50) <= days[-1]]
    for i, origin in enumerate(origins):
        if i % 2 == 0:
            rows.extend(_plant(origin.to_pydatetime(), accel_offs, accel_mags))
            main = origin + pd.Timedelta(days=9)
            rows.append(
                {
                    "event_id": f"main-{main.date()}",
                    "observed_at": main.to_pydatetime(),
                    "available_at": main.to_pydatetime(),
                    "magnitude": 5.6,
                    "lon": 176.2,
                    "lat": -39.15,
                    "depth_km": 14.0,
                }
            )
        else:
            rows.extend(_plant(origin.to_pydatetime(), flat_offs, flat_mags, lon=176.4, lat=-39.4))
    quakes = pd.DataFrame(rows)
    features = pd.DataFrame({"issue_time": days})
    report = run_foreshock(
        train_end="2018-12-31",
        holdout_start="2019-01-01",
        root=tmp_path,
        features=features,
        quakes=quakes,
    )
    assert report["writes_issued_p"] is False
    hik = next(r for r in report["regions"] if r["region"] == "hikurangi")
    feat = catalog_foreshock_frame(quakes, features["issue_time"], versions=None, root=tmp_path)
    accel_morning = days[40] + pd.Timedelta(days=8)
    swarm_morning = days[120] + pd.Timedelta(days=8)
    accel = feat.loc[feat["issue_time"] == accel_morning].iloc[0]
    swarm = feat.loc[feat["issue_time"] == swarm_morning].iloc[0]
    assert int(accel["seq_n_hikurangi"]) >= 4
    assert int(swarm["seq_n_hikurangi"]) >= 4
    assert abs(int(accel["seq_n_hikurangi"]) - int(swarm["seq_n_hikurangi"])) <= 2
    assert float(accel["seq_pprime_hikurangi"]) > float(swarm["seq_pprime_hikurangi"])
    assert (hik.get("sequence_conditional") or {}).get("n_mornings", 0) > 0
    assert hik.get("issued_admission", {}).get("applied") is False
