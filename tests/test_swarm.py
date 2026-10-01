from datetime import datetime, timezone

import pandas as pd

from eqpred.eval.hunt import driver_family
from eqpred.ingest.swarm import daily_swarm_features, models_for_window, pull_swarm


def test_daily_swarm_features_global_and_boxes():
    samples = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                [
                    "2016-01-01T00:00:00Z",
                    "2016-01-01T06:00:00Z",
                    "2016-01-01T12:00:00Z",
                    "2016-01-01T18:00:00Z",
                ]
            ),
            "Latitude": [-40.0, -44.0, -30.0, 10.0],
            "Longitude": [177.0, 170.0, -178.0, 20.0],
            "F_res_MCO_SHA_2D": [12.0, 8.0, 20.0, 100.0],
        }
    )
    frame = daily_swarm_features(samples)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["source"] == "esa.swarm"
    assert abs(float(row["swarm_f_res_nt"]) - 35.0) < 1e-9
    assert float(row["swarm_hikurangi_f_res_nt"]) == 12.0
    assert float(row["swarm_alpine_f_res_nt"]) == 8.0
    assert float(row["swarm_kermadec_f_res_nt"]) == 20.0
    assert row["available_at"] > row["observed_at"]
    assert driver_family("swarm_f_res_nt") == "wave2"
    assert driver_family("mig_swarm_along") == "catalog"


def test_models_for_window_drops_dead_2d():
    assert models_for_window(datetime(2017, 12, 1, tzinfo=timezone.utc)) == (
        "MCO_SHA_2D",
        "CHAOS-Core",
    )
    assert models_for_window(datetime(2018, 1, 1, tzinfo=timezone.utc)) == ("CHAOS-Core",)


def test_chaos_residual_column_normalizes():
    samples = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2020-06-01T00:00:00Z", "2020-06-01T12:00:00Z"]),
            "Latitude": [-40.0, -30.0],
            "Longitude": [177.0, -178.0],
            "F_res_CHAOS-Core": [10.0, 20.0],
        }
    )
    frame = daily_swarm_features(samples)
    assert len(frame) == 1
    assert abs(float(frame.iloc[0]["swarm_f_res_nt"]) - 15.0) < 1e-9


def test_pull_swarm_skips_without_token(monkeypatch):
    monkeypatch.delenv("VIRES_TOKEN", raising=False)
    monkeypatch.delenv("ESA_VIRES_TOKEN", raising=False)
    monkeypatch.delenv("VIRES_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr("eqpred.ingest.swarm.vires_token", lambda: None)
    frame = pull_swarm(
        datetime(2016, 1, 1, tzinfo=timezone.utc),
        datetime(2016, 2, 1, tzinfo=timezone.utc),
        "run-test",
    )
    assert frame.empty
