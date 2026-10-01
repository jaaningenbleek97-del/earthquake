import numpy as np
import pandas as pd

from eqpred.eval.waiting import catalog_wait, is_aftershock_swarm, odd_box, remaining_wait, wait_plain
from eqpred.regions import REGIONS


def _events():
    rows = []
    for origin, available, mag in (
        ("2020-01-01", "2020-01-01", 7.2),
        ("2020-01-21", "2020-01-21", 7.1),
        ("2020-03-01", "2020-03-05", 7.4),
    ):
        rows.append(
            {
                "observed_at": pd.Timestamp(origin, tz="UTC"),
                "available_at": pd.Timestamp(available, tz="UTC"),
                "magnitude": mag,
                "lon": 175.0,
                "lat": -40.0,
            }
        )
    return pd.DataFrame(rows)


def test_remaining_wait_uses_only_reported_gaps():
    origins = np.array(["2020-01-01", "2020-01-11", "2020-02-01"], dtype="datetime64[ns]")
    available = np.array(["2020-01-01", "2020-01-11", "2020-02-05"], dtype="datetime64[ns]")
    issue = np.datetime64("2020-01-15")
    wait, since, n_gaps = remaining_wait(origins, available, issue)
    assert since == 4.0
    assert n_gaps == 1
    assert wait == 6.0


def test_unreported_shock_does_not_shorten_the_wait():
    frame = _events()
    early = catalog_wait(frame, REGIONS["global"], 7.0, pd.Timestamp("2020-01-25", tz="UTC"))
    late = catalog_wait(frame, REGIONS["global"], 7.0, pd.Timestamp("2020-03-06", tz="UTC"))
    assert early["wait_days"] == 16.0
    assert "Not part of the 30-day chance." in early["wait_plain"]
    assert early["writes_issued_p"] is False
    assert late["wait_since_days"] == 5.0
    assert late["wait_days"] is not None


def test_wait_plain_says_when_the_record_has_no_longer_gap():
    text = wait_plain(None, 40)
    assert text.startswith("Longer than the gaps")
    assert "40 days since" in text


def _hikurangi(when, mag):
    return {
        "observed_at": pd.Timestamp(when, tz="UTC"),
        "available_at": pd.Timestamp(when, tz="UTC"),
        "magnitude": mag,
        "lon": 176.0,
        "lat": -40.0,
    }


def test_aftershock_swarm_blocks_a_reported_shock():
    rows = [_hikurangi("2020-06-01", 5.6)]
    for day in range(1, 5):
        rows.append(_hikurangi(f"2020-06-0{day + 1}", 4.0))
    frame = pd.DataFrame(rows)
    assert is_aftershock_swarm(frame, REGIONS["hikurangi"], 5.0, pd.Timestamp("2020-06-06", tz="UTC"))
    assert not is_aftershock_swarm(
        pd.DataFrame([_hikurangi("2020-06-01", 5.6)]),
        REGIONS["hikurangi"],
        5.0,
        pd.Timestamp("2020-06-06", tz="UTC"),
    )


def test_odd_box_names_the_place_furthest_from_its_usual_gap():
    rows = [
        _hikurangi("2020-01-01", 5.2),
        _hikurangi("2020-01-20", 5.1),
        _hikurangi("2020-02-10", 5.4),
    ]
    found = odd_box(pd.DataFrame(rows), pd.Timestamp("2020-08-01", tz="UTC"))
    assert found["region"] == "hikurangi"
    assert "odd one out" in found["plain"]
    assert found["writes_issued_p"] is False
    assert "30-day chance" in found["plain"]
