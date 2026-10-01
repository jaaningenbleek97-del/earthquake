from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.hunt import driver_family
from eqpred.features.sequence import (
    aki_utsu_local,
    b_positive,
    inverse_omori_pprime,
    sequence_frame,
    traffic_light,
)
from eqpred.features.migration import CLUSTER_DAYS, CLUSTER_KM


def test_cluster_constants_stay_locked():
    assert CLUSTER_KM == 80.0
    assert CLUSTER_DAYS == 7.0


def test_sequence_columns_stay_catalog():
    assert driver_family("seq_n_hikurangi") == "catalog"
    assert driver_family("seq_b_pos_delta_alpine") == "catalog"
    assert driver_family("seq_pprime_kermadec") == "catalog"
    assert driver_family("seq_active_any") == "catalog"


def test_accelerating_cluster_has_higher_pprime_than_flat_swarm():
    accel = np.array([0.0, 48.0, 80.0, 100.0, 112.0, 120.0, 124.0, 126.0])
    flat = np.linspace(0.0, 126.0, 8)
    assert inverse_omori_pprime(accel) > inverse_omori_pprime(flat) + 0.15
    assert inverse_omori_pprime(flat) < 0.4


def test_b_positive_survives_detection_hole_after_m6():
    complete = np.array([3.1, 3.3, 3.2, 3.5, 3.4, 3.7, 3.6, 3.9] * 8)
    after_full = np.concatenate([[6.2], complete[:40]])
    after_hole = np.concatenate([[6.2], np.array([4.5, 4.7, 4.6, 4.9] * 8)])
    raw_complete = aki_utsu_local(complete, min_n=20, mc=3.1)
    raw_hole = aki_utsu_local(after_hole, min_n=20, mc=3.1)
    b_full = b_positive(np.concatenate([complete, after_full]), min_n=8)
    b_hole = b_positive(np.concatenate([complete, after_hole]), min_n=8)
    assert np.isfinite(b_hole) and np.isfinite(b_full)
    assert abs(b_hole - b_full) / b_full < 0.10
    assert raw_hole < raw_complete - 0.05
    assert (
        traffic_light(
            region_id="hikurangi",
            max_m=6.2,
            n=len(after_hole),
            n_bg=len(complete),
            b_pos=b_hole,
            b_pos_bg=b_positive(complete, min_n=8),
            mc_shift=True,
        )
        == "yellow"
    )


def test_traffic_light_none_outside_nz_or_without_m6():
    assert traffic_light(region_id="kermadec", max_m=6.4, n=40, n_bg=40, b_pos=0.8, b_pos_bg=1.0, mc_shift=False) == "none"
    assert traffic_light(region_id="hikurangi", max_m=5.4, n=40, n_bg=40, b_pos=0.8, b_pos_bg=1.0, mc_shift=False) == "none"
    assert (
        traffic_light(region_id="hikurangi", max_m=6.4, n=40, n_bg=40, b_pos=0.85, b_pos_bg=1.0, mc_shift=False)
        == "red"
    )


def _event(day, lon, lat, mag, depth=12.0):
    observed = datetime(2017, 6, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "observed_at": observed,
        "available_at": observed,
        "lon": lon,
        "lat": lat,
        "depth_km": depth,
        "magnitude": mag,
        "event_id": f"e{day}-{mag}",
    }


def test_sequence_frame_is_as_of_and_quantifies_an_active_cluster():
    rows = [
        _event(0, 176.1, -39.1, 3.6),
        _event(1, 176.15, -39.08, 3.7),
        _event(2, 176.2, -39.05, 3.8),
        _event(3, 176.25, -39.02, 4.0, depth=10.0),
        _event(4, 176.3, -39.0, 4.2, depth=8.0),
    ]
    catalog = pd.DataFrame(rows)
    days = pd.date_range("2017-06-01", periods=8, freq="D", tz="UTC")
    frame = sequence_frame(days, catalogs={"hikurangi": catalog, "alpine": catalog.iloc[0:0], "kermadec": catalog.iloc[0:0]})
    early = frame.loc[frame["issue_time"] == days[1]].iloc[0]
    late = frame.loc[frame["issue_time"] == days[6]].iloc[0]
    hidden = sequence_frame(
        pd.Series([days[6]]),
        catalogs={
            "hikurangi": catalog.assign(available_at=catalog["available_at"] + pd.Timedelta(days=10)),
            "alpine": catalog.iloc[0:0],
            "kermadec": catalog.iloc[0:0],
        },
    ).iloc[0]
    assert int(early["seq_active_hikurangi"]) == 0
    assert int(late["seq_n_hikurangi"]) >= 4
    assert late["seq_shape_hikurangi"] in {"swarm", "omori", "mixed"}
    assert int(late["seq_active_any"]) == 1
    assert int(hidden["seq_active_hikurangi"]) == 0
