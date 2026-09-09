from datetime import datetime, timezone

import numpy as np
import pandas as pd

from eqpred.eval.eepas import eepas_daily_rate


def test_eepas_rate_is_history_only_and_positive_after_event():
    t0 = datetime(2016, 1, 10, tzinfo=timezone.utc)
    events = pd.DataFrame(
        {
            "observed_at": pd.to_datetime([t0], utc=True),
            "magnitude": [6.8],
        }
    )
    days = pd.date_range("2016-01-01", periods=40, freq="D", tz="UTC")
    rate = eepas_daily_rate(events, days, m0=4.5)
    assert rate[0] == 0.0
    assert rate[9] == 0.0
    assert rate[10] > 0.0
    assert rate[20] > 0.0
    assert rate[20] < rate[10]
    assert np.all(np.isfinite(rate))
