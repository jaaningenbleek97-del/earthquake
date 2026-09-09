from datetime import datetime, timezone

import numpy as np
import pandas as pd

from eqpred.cusp_geom import cusp_lon_deg, cusp_weight
from eqpred.features.cusp import cusp_frame
from eqpred.ingest.storms import parse_omni_hourly_dst, storm_onsets
from eqpred.models.cusp import box_cusp_gain
from eqpred.regions import REGIONS, region_centroid_lon


def test_nz_is_under_cusp_near_19utc_not_07utc():
    hik = region_centroid_lon(REGIONS["hikurangi"])
    noon_over_nz = float(cusp_lon_deg(19.5))
    noon_over_atlantic = float(cusp_lon_deg(7.0))
    assert 160.0 < noon_over_nz < 180.0
    assert float(cusp_weight(hik, noon_over_nz)) > 0.85
    assert float(cusp_weight(hik, noon_over_atlantic)) < 0.15


def test_storm_onset_fires_once_after_quiet():
    hours = pd.date_range("2016-03-01", periods=40, freq="h", tz="UTC")
    dst = np.full(40, -10.0)
    dst[20:28] = -80.0
    dst[30:36] = -70.0
    hourly = pd.DataFrame({"observed_at": hours, "dst_nt": dst})
    onsets = storm_onsets(hourly)
    assert len(onsets) == 1
    assert onsets.iloc[0]["observed_at"] == hours[20]


def test_cusp_frame_is_asof_and_expires():
    onset = datetime(2016, 3, 2, 19, tzinfo=timezone.utc)
    available = onset + pd.Timedelta(days=1)
    onsets = pd.DataFrame(
        {
            "observed_at": [onset],
            "available_at": [available],
            "cusp_lon_deg": [float(cusp_lon_deg(19.0))],
        }
    )
    live = cusp_frame(pd.DatetimeIndex([available + pd.Timedelta(hours=12)]), onsets)
    assert float(live["cusp_align_hikurangi"].iloc[0]) > 0.8
    stale = cusp_frame(pd.DatetimeIndex([available + pd.Timedelta(days=3)]), onsets)
    assert float(stale["cusp_align_hikurangi"].iloc[0]) == 0.0
    early = cusp_frame(pd.DatetimeIndex([onset + pd.Timedelta(hours=6)]), onsets)
    assert float(early["cusp_align_hikurangi"].iloc[0]) == 0.0


def test_box_cusp_gain_held_out_is_one():
    row = pd.Series({"cusp_align_hikurangi": 0.9})
    assert box_cusp_gain({"regions": {}}, "hikurangi", row) == 1.0


def test_parse_omni_hourly_dst_line():
    text = "2016   1  0 2488 51 52  55  36  15.0  15.0 -70.0 343.2   4.9  -1.5 -14.1   2.2 -14.0   0.1   1.3   0.9   0.7   0.5   325.0   8.1  412.0    5.5     2.8   0.6   2.8   3.91  00022  999.9    4.19  999.9  99999  999.9  9.9  9.9   30    -11     6   13  13  999  999  999   3  23  999.9    72.0\n"
    # If field count is short, parser skips. Build a 41+ field line.
    parts = ["2016", "1", "6"] + ["0"] * 37 + ["-80"]
    text = " ".join(parts) + "\n"
    frame = parse_omni_hourly_dst(text)
    assert len(frame) == 1
    assert frame.iloc[0]["dst_nt"] == -80.0
    assert frame.iloc[0]["observed_at"].hour == 6
