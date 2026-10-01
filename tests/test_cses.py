from datetime import datetime, timezone

import pandas as pd

from eqpred.eval.hunt import driver_family
from eqpred.ingest.cses import daily_cses_features, pull_cses


def test_daily_cses_features_hepd_flux():
    samples = pd.DataFrame(
        {
            "observed_at": ["2019-01-01T00:00:00Z", "2019-01-01T12:00:00Z"],
            "cses_hepd_e": [10.0, 14.0],
            "cses_hepd_p": [2.0, 4.0],
        }
    )
    frame = daily_cses_features(samples)
    assert len(frame) == 1
    assert float(frame.iloc[0]["cses_hepd_e"]) == 12.0
    assert float(frame.iloc[0]["cses_hepd_p"]) == 3.0
    assert frame.iloc[0]["available_at"] > frame.iloc[0]["observed_at"]
    assert driver_family("cses_hepd_e") == "wave2"


def test_pull_cses_skips_without_credentials(monkeypatch):
    monkeypatch.setattr("eqpred.ingest.cses.limadou_credentials", lambda: None)
    frame = pull_cses(
        datetime(2019, 1, 1, tzinfo=timezone.utc),
        datetime(2019, 2, 1, tzinfo=timezone.utc),
        "run-test",
    )
    assert frame.empty
