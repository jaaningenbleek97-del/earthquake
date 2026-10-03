import numpy as np
import pandas as pd

from eqpred.eval.explain import furthest_above_rate, neighborhood_from_ref, story_plain
from eqpred.eval.lineup import (
    OTHER_CAP,
    WATCH_CAP,
    _measurement_group,
    _motion_base,
    _raw_scale,
    _reading_groups,
    _share_ceiling,
    vote_cap,
    watch_agreement,
)
from eqpred.eval.mainshock import _BlankAwareScaler, _SitOutScaler


def test_extra_slip_removes_the_steady_plate_speed():
    from eqpred.features.gnss_slip import extra_slip, known_plate_rate

    position = np.arange(10, dtype=float) * 10.0
    rate = np.full(10, 365.25)
    extra = extra_slip(position, rate, 7)
    # 70 mm moved, plate speed accounts for 7 mm.
    assert abs(float(extra[7]) - 63.0) < 1e-6
    assert np.isnan(extra[3])
    published = np.full(12, np.nan)
    published[10:] = 1.0
    steady = known_plate_rate(np.arange(12, dtype=float), published, lookback=4)
    assert abs(float(steady[6]) - 365.25) < 1e-6
    assert float(steady[10]) == 1.0


def test_geonet_tail_stays_off_nevada_columns(tmp_path):
    from eqpred.features.gnss_slip import _geonet_tail

    days = pd.date_range("2020-01-01", periods=5, tz="UTC")
    columns = _geonet_tail(days, tmp_path)
    assert set(columns) == {
        "geonet_nz_east_mm",
        "geonet_nz_north_mm",
        "geonet_nz_east_7d",
        "geonet_nz_north_7d",
        "geonet_nzmean_east_7d",
        "geonet_nzmean_north_7d",
    }
    assert all(not name.startswith("gnss_") for name in columns)
    groups = dict(_reading_groups(["hydro_gwet__lag0", "hydro_gwet__lag7", "solar_kp"]))
    assert groups["hydro_gwet"] == [0, 1]
    assert groups["solar_kp"] == [2]
    z = np.linspace(-2.0, 2.0, 200)
    roll = np.full(200, 0.1)
    hits = np.clip(roll + 0.05 * z, 0.0, 1.0)
    assert abs(_raw_scale(z, hits, roll, np.ones(200, dtype=bool)) - 0.05) < 0.02


def test_a_blank_column_has_no_flag_and_a_real_zero_still_differs():
    train = np.array([[1.0, 0.0], [3.0, 1.0], [5.0, 0.0], [np.nan, 2.0]])
    sit = _SitOutScaler().fit(train)
    blank_aware = _BlankAwareScaler().fit(train)
    row = np.array([[np.nan, 0.0]])
    assert sit.transform(row).shape[1] == 2
    assert blank_aware.transform(row).shape[1] == 4
    assert sit.transform(row)[0, 0] == 0.0
    zero = sit.transform(np.array([[0.0, np.nan]]))
    assert zero[0, 0] != 0.0
    assert zero[0, 1] == 0.0


def test_neighborhood_counts_hits_without_changing_the_rate():
    scores = np.linspace(0.0, 1.0, 100)
    hits = np.zeros(100)
    hits[:40] = 1.0
    report = neighborhood_from_ref(0.2, scores, hits, min_n=80)
    assert report["n"] == 80
    assert report["hits"] == 40
    assert "80 closest check-year mornings" in report["plain"]
    assert report["plain"]


def test_agreement_needs_two_thirds_and_at_least_three_readings():
    quiet = watch_agreement([("solar", 0.02), ("spf", None)])
    assert quiet["agreed"] is False
    assert "Only 1" in quiet["plain"]
    split = watch_agreement([("solar", 0.02), ("spf", -0.02), ("geometry", 0.02), ("cosmic", -0.02)])
    assert split["agreed"] is False
    down = watch_agreement([("solar", -0.02), ("spf", -0.02), ("geometry", 0.001), ("dust", 0.02)])
    assert down["agreed"] is False
    agreed = watch_agreement([("solar", -0.02), ("spf", -0.02), ("geometry", 0.001)])
    assert agreed["agreed"] is True
    assert agreed["direction"] == "down"
    assert agreed["plain"].startswith("2 of 3")


def test_the_story_follows_the_lift_not_the_larger_chance():
    rows = [
        {"region": "global", "label": "Earth-wide", "p": 0.8, "climatology_p": 0.75, "chance_plain": "80 in 100", "climatology_chance_plain": "75 in 100"},
        {"region": "hikurangi", "label": "Hikurangi", "p": 0.4, "climatology_p": 0.1, "chance_plain": "40 in 100", "climatology_chance_plain": "10 in 100"},
    ]
    lift = furthest_above_rate(rows)
    assert lift["region"] == "hikurangi"
    waiting = story_plain(lift=lift, odd={"region": "alpine", "label": "Alpine Fault"}, agreement={"plain": "No agreement.", "agreed": False}, shocks=[])
    assert "Hikurangi" in waiting
    assert "Alpine Fault" in waiting
    assert "waits for a first shock" in waiting
    happened = story_plain(
        lift=lift,
        odd={"region": "alpine", "label": "Alpine Fault"},
        agreement={"plain": "No agreement.", "agreed": False, "direction": None},
        shocks=[{"region": "hikurangi", "label": "Hikurangi"}],
    )
    assert "furthest above its usual rate" in happened
    assert "had not agreed" in happened


def test_copies_of_one_measurement_share_a_ceiling():
    assert _measurement_group("gnss_auck_east_mm") == _measurement_group("gnss_wgtn_north_mm") == "GPS positions"
    assert _measurement_group("goes_proton_10mev") == _measurement_group("goes_proton_10mev_max7")
    assert _measurement_group("neutron_domc_count") == _measurement_group("neutron_domc_count_min7")
    assert _measurement_group("gnss_auck_east_extra_7d") == "extra GPS motion"
    assert _motion_base("gnss_strain_hikurangi_wide_shear_30d")
    assert vote_cap("solar") == WATCH_CAP
    assert vote_cap("wave2") == OTHER_CAP
    shared, total = _share_ceiling(
        [np.array([0.02, 0.004]), np.array([0.02, 0.004]), np.array([0.02, np.nan])],
        0.01,
    )
    assert abs(float(total[0]) - 0.01) < 1e-9
    assert abs(float(shared[0][0]) - 0.01 / 3) < 1e-9
    assert abs(float(total[1]) - 0.008) < 1e-9
    assert shared[2][1] == 0.0
