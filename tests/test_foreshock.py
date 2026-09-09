import numpy as np
import pandas as pd

from eqpred.eval.foreshock import catalog_columns, catalog_foreshock_frame, run_foreshock
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
    assert feat["n_m4_7d"].iloc[12] >= 1


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
