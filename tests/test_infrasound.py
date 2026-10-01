from datetime import datetime, timezone

import numpy as np
import pandas as pd

from eqpred.ingest.infrasound import release_time, same_day_public


def test_chatham_years_through_2020_were_not_public_until_november_2021():
    assert release_time(2019) == datetime(2021, 11, 8, tzinfo=timezone.utc)
    assert release_time(2024) > release_time(2019)


def test_same_day_loudness_stays_hidden_until_the_file_is_public():
    frame = pd.DataFrame(
        {
            "observed_at": ["2019-06-01T00:00:00Z", "2022-06-02T00:00:00Z"],
            "available_at": ["2021-11-08T00:00:00Z", "2022-06-02T00:00:00Z"],
            "infra_is36_n": [4.0, 9.0],
        }
    )
    issue = pd.to_datetime(["2019-06-01", "2022-06-01", "2022-06-02"], utc=True)
    out = same_day_public(frame, issue, ["infra_is36_n"])
    assert np.isnan(out["infra_is36_n"].iloc[0])
    assert np.isnan(out["infra_is36_n"].iloc[1])
    assert out["infra_is36_n"].iloc[2] == 9.0
