import numpy as np
import pandas as pd

from eqpred.eval.decluster import LABEL_V2, fresh_hits, use_label_version

BOX = {"kind": "bbox", "name": "box", "min_lon": 170.0, "max_lon": 172.0, "min_lat": -42.0, "max_lat": -40.0}


def _quakes(rows):
    frame = pd.DataFrame(rows, columns=["observed_at", "magnitude", "lon", "lat"])
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["available_at"] = frame["observed_at"]
    return frame


def _hit_on(quakes, day, version):
    days = pd.DatetimeIndex([pd.Timestamp(day, tz="UTC")])
    return bool(fresh_hits(quakes, BOX, 6.0, days, 30, label_version=version)[0])


def test_a_late_aftershock_of_a_big_quake_is_fresh_in_v1_but_not_v2():
    quakes = _quakes([("2020-01-01", 7.6, 171.0, -41.0), ("2020-04-01", 6.2, 171.3, -41.2)])
    assert _hit_on(quakes, "2020-03-20", 1)
    assert not _hit_on(quakes, "2020-03-20", 2)


def test_a_bigger_quake_just_outside_the_box_counts_in_v2():
    quakes = _quakes([("2020-01-01", 6.5, 172.5, -41.0), ("2020-01-04", 6.1, 171.9, -41.0)])
    assert _hit_on(quakes, "2020-01-02", 1)
    assert not _hit_on(quakes, "2020-01-02", 2)


def test_a_far_quake_and_a_bigger_later_shock_stay_fresh_in_v2():
    far = _quakes([("2020-01-01", 7.6, 120.0, 10.0), ("2020-02-01", 6.2, 171.0, -41.0)])
    assert _hit_on(far, "2020-01-20", 2)
    growing = _quakes([("2020-01-01", 6.1, 171.0, -41.0), ("2020-01-03", 6.8, 171.1, -41.0)])
    days = pd.DatetimeIndex([pd.Timestamp("2020-01-02", tz="UTC")])
    assert bool(fresh_hits(growing, BOX, 6.0, days, 30, label_version=2)[0])


def test_the_context_switches_the_default_and_restores_it():
    quakes = _quakes([("2020-01-01", 7.6, 171.0, -41.0), ("2020-04-01", 6.2, 171.3, -41.2)])
    days = pd.DatetimeIndex([pd.Timestamp("2020-03-20", tz="UTC")])
    with use_label_version(LABEL_V2):
        inside = fresh_hits(quakes, BOX, 6.0, days, 30)
    after = fresh_hits(quakes, BOX, 6.0, days, 30)
    assert not bool(inside[0])
    assert bool(after[0])
    assert isinstance(after, np.ndarray)
