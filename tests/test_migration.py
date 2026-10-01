from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.decluster import haversine_km
from eqpred.features.migration import (
    CLUSTER_KM,
    MAX_DEPTH_KM,
    active_cluster,
    along_updip_km,
    cluster_asof,
    fault_plane_distance_km,
    migration_frame,
    slope_per_day,
    swarm_score,
)
from eqpred.models.migration import box_migration_gain
from eqpred.regions import region_centroid_lon, REGIONS


def _event(day: int, lon: float, lat: float, depth: float, mag: float, delay_hours: float = 0.0) -> dict:
    observed = datetime(2017, 6, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "observed_at": observed,
        "available_at": observed + timedelta(hours=delay_hours),
        "lon": lon,
        "lat": lat,
        "depth_km": depth,
        "magnitude": mag,
        "event_id": f"e{day}",
    }


def test_along_strike_walk_is_positive():
    # Alpine strike 55°: walk northeast along the fault.
    region = REGIONS["alpine"]
    lon0 = region_centroid_lon(region)
    lat0 = 0.5 * (region["min_lat"] + region["max_lat"])
    strike = 55.0
    step = 15.0
    east = np.array([0.0, np.sin(np.radians(strike)) * step])
    north = np.array([0.0, np.cos(np.radians(strike)) * step])
    lon = lon0 + east / (111.32 * np.cos(np.radians(lat0)))
    lat = lat0 + north / 111.32
    along, _updip = along_updip_km(lon, lat, np.array([10.0, 10.0]), lon0, lat0, strike, 50.0)
    assert along[1] > along[0]
    hours = np.array([0.0, 24.0])
    assert slope_per_day(hours, along) > 10


def test_shallowing_is_positive_updip():
    lon0 = region_centroid_lon(REGIONS["hikurangi"])
    lat0 = -39.7
    along, updip = along_updip_km(
        np.array([lon0, lon0]),
        np.array([lat0, lat0]),
        np.array([40.0, 15.0]),
        lon0,
        lat0,
        45.0,
        12.0,
    )
    assert updip[1] > updip[0]
    assert slope_per_day(np.array([0.0, 48.0]), updip) > 0


def test_fault_plane_distance_is_zero_on_locked_plane():
    lon0 = region_centroid_lon(REGIONS["hikurangi"])
    lat0 = 0.5 * (REGIONS["hikurangi"]["min_lat"] + REGIONS["hikurangi"]["max_lat"])
    dip = 12.0
    down_h = 40.0
    az = np.radians(45.0 + 90.0)
    east = down_h * np.sin(az)
    north = down_h * np.cos(az)
    lon = lon0 + east / (111.32 * np.cos(np.radians(lat0)))
    lat = lat0 + north / 111.32
    depth = down_h * np.tan(np.radians(dip))
    distance = fault_plane_distance_km(
        np.array([lon]),
        np.array([lat]),
        np.array([depth]),
        lon0,
        lat0,
        45.0,
        dip,
    )
    assert float(distance[0]) < 1e-6


def test_future_event_hidden_until_available():
    lon0 = 178.0
    lat0 = -39.5
    rows = [
        _event(0, lon0, lat0, 20, 4.2),
        _event(1, lon0 + 0.1, lat0 + 0.05, 18, 4.1),
        _event(2, lon0 + 0.2, lat0 + 0.1, 16, 4.0),
        _event(3, lon0 + 0.3, lat0 + 0.15, 14, 3.9, delay_hours=72),
    ]
    catalog = pd.DataFrame(rows)
    issue = datetime(2017, 6, 5, tzinfo=timezone.utc)
    asof = cluster_asof(catalog, "hikurangi", issue)
    assert asof["n"] == 3
    later = cluster_asof(catalog, "hikurangi", datetime(2017, 6, 8, tzinfo=timezone.utc))
    assert later["n"] == 4


def test_deep_slab_dropped_from_cluster():
    lon0 = 179.0
    lat0 = -31.0
    rows = [
        _event(0, lon0, lat0, 20, 5.0),
        _event(1, lon0 + 0.1, lat0 + 0.05, 18, 4.8),
        _event(2, lon0 + 0.2, lat0 + 0.1, 16, 4.7),
        _event(3, lon0 + 0.05, lat0, MAX_DEPTH_KM + 80, 6.0),
    ]
    cluster = active_cluster(pd.DataFrame(rows))
    # The deep event is still in the raw frame here; the catalog prep drops it.
    from eqpred.features.migration import _prepare_catalog

    prepared = _prepare_catalog(pd.DataFrame(rows), 4.5)
    assert (prepared["depth_km"] <= MAX_DEPTH_KM).all()
    assert len(prepared) == 3
    assert len(cluster) >= 1


def test_haversine_wraps_dateline():
    dist = float(haversine_km(179.8, -32.0, np.array([-179.8]), np.array([-32.0]))[0])
    assert dist < 50


def test_kermadec_dateline_cluster():
    rows = [
        _event(0, 179.5, -32.0, 20, 5.0),
        _event(1, 179.8, -31.7, 19, 4.9),
        _event(2, -179.8, -31.4, 18, 4.8),
        _event(3, -179.4, -31.1, 17, 4.7),
    ]
    catalog = pd.DataFrame(rows)
    asof = cluster_asof(catalog, "kermadec", datetime(2017, 6, 8, tzinfo=timezone.utc))
    assert asof["n"] >= 4
    assert asof["along_km_d"] is not None


def test_swarm_score_high_without_mainshock():
    assert swarm_score(np.array([4.1, 4.0, 4.2, 3.9, 4.1])) > 0.9
    assert swarm_score(np.array([6.5, 4.0, 4.1, 3.9, 4.2])) < 0.2


def test_cluster_does_not_jump_space():
    near = [
        _event(0, 178.0, -39.5, 20, 4.2),
        _event(1, 178.1, -39.4, 19, 4.1),
        _event(2, 178.2, -39.3, 18, 4.0),
        _event(3, 178.3, -39.2, 17, 3.9),
    ]
    far_old = _event(-1, 175.0, -41.0, 15, 5.5)
    from_near = active_cluster(pd.DataFrame([far_old, *near]))
    assert len(from_near) == 4
    assert float(from_near["lon"].min()) > 177
    far_latest = _event(4, 175.0, -41.0, 15, 5.5)
    from_far = active_cluster(pd.DataFrame(near + [far_latest]))
    assert len(from_far) == 1
    assert CLUSTER_KM < 200


def test_migration_frame_columns_and_nan_when_quiet():
    days = pd.date_range("2017-06-01", periods=5, freq="D", tz="UTC")
    catalogs = {rid: pd.DataFrame() for rid in ("hikurangi", "alpine", "kermadec")}
    frame = migration_frame(pd.Series(days), catalogs=catalogs)
    assert "mig_along_hikurangi" in frame.columns
    assert "mig_updip_kermadec" in frame.columns
    assert frame["mig_along_hikurangi"].isna().all()
    assert (frame["mig_n_alpine"] == 0).all()


def test_box_gain_one_when_held_out():
    row = pd.Series({"mig_along_hikurangi": 3.0})
    assert box_migration_gain({"regions": {}}, "hikurangi", row) == 1.0
    assert box_migration_gain(
        {"regions": {"hikurangi": {"used_on_issued": False, "column": "mig_along_hikurangi"}}},
        "hikurangi",
        row,
    ) == 1.0
