import pandas as pd

from eqpred.eval.earth_state import analog_neighbors, attach_anomaly_features
from eqpred.eval.hunt import driver_family
from eqpred.faults import GRAPH_EDGES, SEGMENTS, segment_region
from eqpred.features.graph import attach_graph_features, segment_energy
from eqpred.regions import in_bbox


def _days(n=200, start="2016-01-01"):
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def test_segment_boxes_cover_parent_without_crossing_boxes():
    hik_south = pd.DataFrame({"lon": [176.0], "lat": [-41.5]})
    ker_north = pd.DataFrame({"lon": [-178.0], "lat": [-28.0]})
    alpine_north = pd.DataFrame({"lon": [173.0], "lat": [-43.0]})
    assert bool(in_bbox(hik_south, segment_region("hikurangi_south")).iloc[0])
    assert not bool(in_bbox(hik_south, segment_region("kermadec_south")).iloc[0])
    assert bool(in_bbox(ker_north, segment_region("kermadec_north")).iloc[0])
    assert bool(in_bbox(alpine_north, segment_region("alpine_north")).iloc[0])
    assert {e[0] for e in GRAPH_EDGES} <= set(SEGMENTS)
    assert ("alpine_north", "hikurangi_south", 0.4, "transfer") in GRAPH_EDGES


def test_neighbor_energy_flows_across_an_edge():
    days = _days(20)
    quakes = pd.DataFrame(
        {
            "observed_at": [days[10]] * 2,
            "available_at": [days[10]] * 2,
            "lon": [176.2, 176.2],
            "lat": [-41.8, -41.8],
            "magnitude": [6.2, 5.4],
        }
    )
    local = segment_energy(quakes, pd.Series(days), "hikurangi_south")
    quiet = segment_energy(quakes, pd.Series(days), "alpine_north")
    feat = pd.DataFrame({"issue_time": days})
    graph = attach_graph_features(feat, quakes, train_end="2016-01-15")
    assert float(local[12]) > 0
    assert float(quiet[12]) == 0.0
    assert float(graph.loc[12, "graph_inflow_alpine_north"]) > 0
    assert driver_family("graph_inflow_hikurangi") == "graph"
    assert driver_family("graph_n_hikurangi_south") == "catalog"


def test_anomaly_model_fits_without_quake_labels():
    days = _days(160)
    rng = list(range(160))
    feat = pd.DataFrame(
        {
            "issue_time": days,
            "proton_density_cm3": [8.0 + 0.2 * (i % 5) for i in rng],
            "solar_wind_speed_kms": [400.0 + (i % 7) for i in rng],
            "imf_bz_nt": [-1.0 + 0.1 * (i % 3) for i in rng],
            "kp": [2.0 + 0.1 * (i % 4) for i in rng],
            "dst_nt": [-10.0 - (i % 6) for i in rng],
            "f107": [80.0 + (i % 5) for i in rng],
        }
    )
    feat.loc[feat.index[-1], "proton_density_cm3"] = 80.0
    feat.loc[feat.index[-1], "kp"] = 8.0
    feat.loc[feat.index[-1], "dst_nt"] = -180.0
    out = attach_anomaly_features(feat, train_end="2016-05-01")
    assert "earth_state_z" in out.columns
    assert float(out["earth_state_z"].iloc[-1]) > float(out["earth_state_z"].iloc[:120].median())
    assert driver_family("earth_state_z") == "earth"


def test_analog_neighbors_prefer_the_matching_morning():
    x = __import__("numpy").array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [5.0, 5.0],
            [0.05, 0.02],
        ],
        dtype=float,
    )
    library = __import__("numpy").array([0, 1, 2])
    neigh, _dist = analog_neighbors(x, 3, library, k=2)
    assert int(neigh[0]) in {0, 1}
    assert 2 not in set(int(i) for i in neigh)
    days = pd.date_range("2016-01-01", periods=4, freq="D", tz="UTC")
    spread, _ = analog_neighbors(x, 3, library, k=2, days=days)
    assert list(spread) == [0]
    assert driver_family("analog_p_global") == "analog"
