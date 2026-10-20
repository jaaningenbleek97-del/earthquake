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


def test_yesterdays_reading_waits_for_available_at():
    from eqpred.features.build import yesterday_frame

    frame = pd.DataFrame(
        {
            "observed_at": ["2019-01-01T00:00:00Z", "2019-01-02T00:00:00Z"],
            "available_at": ["2019-01-15T00:00:00Z", "2019-01-02T00:00:00Z"],
            "pore_apg_hpa": [1.0, 3.0],
        }
    )
    issue = pd.to_datetime(["2019-01-02", "2019-01-03"], utc=True)
    out = yesterday_frame(frame, issue, ["pore_apg_hpa"])
    # Jan 1 reading was not public on Jan 2 morning.
    assert np.isnan(out["pore_apg_hpa"].iloc[0])
    # Jan 2 reading was public on Jan 3 morning.
    assert out["pore_apg_hpa"].iloc[1] == 3.0


def test_merged_station_keeps_its_own_release_stamp():
    from eqpred.features.build import yesterday_frame
    from eqpred.stamp import merge_daily

    first = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(["2019-01-01"], utc=True),
            "available_at": ["2019-01-02T00:00:00Z"],
            "infra_a": [1.0],
        }
    )
    second = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(["2019-01-01", "2019-01-02"], utc=True),
            "available_at": ["2019-03-01T00:00:00Z", "2019-01-03T00:00:00Z"],
            "infra_b": [5.0, 7.0],
        }
    )
    merged = merge_daily(first, second, ["infra_b"])
    stamps = dict(zip(merged["observed_at"].dt.date.astype(str), merged["available_at"]))
    # A day both stations share is public only when the later file is.
    assert stamps["2019-01-01"] == pd.Timestamp("2019-03-01", tz="UTC")
    # A day only the second station has keeps that station's stamp.
    assert stamps["2019-01-02"] == pd.Timestamp("2019-01-03", tz="UTC")
    issue = pd.to_datetime(["2019-01-02", "2019-01-03"], utc=True)
    out = yesterday_frame(merged, issue, ["infra_a", "infra_b"])
    assert np.isnan(out["infra_a"].iloc[0])
    assert out["infra_b"].iloc[1] == 7.0


def test_a_7hz_tone_raises_the_mode1_ratio():
    t = np.arange(int(FS_HZ * 120)) / FS_HZ
    tone = np.sin(2 * np.pi * 7.8 * t)
    quiet = np.sin(2 * np.pi * 6.4 * t)
    assert schumann_mode1_ratio(tone) > schumann_mode1_ratio(quiet)
