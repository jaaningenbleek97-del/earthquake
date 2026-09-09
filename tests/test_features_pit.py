from datetime import datetime, timezone

import pandas as pd

from eqpred.features.build import _asof_available
from eqpred.stamp import parse_utc
from eqpred.ingest.omni import LATENCY
from eqpred.records import iso


def test_asof_hides_late_omni():
    observed = datetime(2020, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "proton_density_cm3": 12.0,
            }
        ]
    )
    days = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    aligned = _asof_available(frame, days, ["proton_density_cm3"])
    assert pd.isna(aligned.loc[aligned["issue_time"] == days[0], "proton_density_cm3"].iloc[0])
    assert aligned.loc[aligned["issue_time"] == days[3], "proton_density_cm3"].iloc[0] == 12.0


def test_parse_utc_mixed_fractional_seconds():
    parsed = parse_utc(["2018-06-26T11:03:05Z", "2018-06-26T11:03:05.123000Z"])
    assert len(parsed) == 2
    assert parsed.dt.tz is not None
