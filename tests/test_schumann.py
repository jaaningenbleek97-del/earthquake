import numpy as np
import pandas as pd

from eqpred.ingest.schumann import FS_HZ, schumann_mode1_ratio


def test_yesterdays_reading_does_not_stick_after_the_series_ends():
    from eqpred.features.build import yesterday_frame

    frame = pd.DataFrame(
        {
            "observed_at": ["2019-01-01T00:00:00Z", "2019-01-02T00:00:00Z"],
            "pore_apg_hpa": [1.0, 3.0],
        }
    )
    issue = pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"], utc=True)
    out = yesterday_frame(frame, issue, ["pore_apg_hpa"])
    assert np.isnan(out["pore_apg_hpa"].iloc[0])
    assert out["pore_apg_hpa"].iloc[1] == 1.0
    assert out["pore_apg_hpa"].iloc[2] == 3.0
    assert np.isnan(out["pore_apg_hpa"].iloc[3])


def test_a_7hz_tone_raises_the_mode1_ratio():
    t = np.arange(int(FS_HZ * 120)) / FS_HZ
    tone = np.sin(2 * np.pi * 7.8 * t)
    quiet = np.sin(2 * np.pi * 6.4 * t)
    assert schumann_mode1_ratio(tone) > schumann_mode1_ratio(quiet)
