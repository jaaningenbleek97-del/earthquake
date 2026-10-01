from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.hunt import driver_family
from eqpred.features.repeaters import days_since_hikurangi_repeater


def _row(when, lat, lon, mag, depth=20.0, delay_hours=48.0):
    observed = datetime(when.year, when.month, when.day, tzinfo=timezone.utc)
    return {
        "observed_at": observed,
        "available_at": observed + timedelta(hours=delay_hours),
        "lat": lat,
        "lon": lon,
        "magnitude": mag,
        "depth_km": depth,
    }


def test_nearby_small_pair_resets_the_clock_only_after_both_are_reported():
    first = datetime(2020, 1, 1, tzinfo=timezone.utc)
    second = datetime(2020, 6, 1, tzinfo=timezone.utc)
    events = pd.DataFrame(
        [
            _row(first, -40.0, 177.0, 3.4),
            _row(second, -40.02, 177.02, 3.6),
        ]
    )
    mornings = pd.to_datetime(
        ["2020-01-03", "2020-06-02", "2020-06-04"],
        utc=True,
    )
    days = days_since_hikurangi_repeater(events, mornings)
    assert np.isnan(days[0])
    assert np.isnan(days[1])
    assert days[2] == 1.0


def test_distant_or_large_events_are_not_a_repeat():
    first = datetime(2020, 1, 1, tzinfo=timezone.utc)
    events = pd.DataFrame(
        [
            _row(first, -40.0, 177.0, 3.4),
            _row(datetime(2020, 2, 1, tzinfo=timezone.utc), -41.5, 178.5, 3.4),
            _row(datetime(2020, 3, 1, tzinfo=timezone.utc), -40.01, 177.01, 5.4),
        ]
    )
    mornings = pd.to_datetime(["2020-04-01"], utc=True)
    days = days_since_hikurangi_repeater(events, mornings)
    assert np.isnan(days[0])


def test_repeater_clock_can_reach_the_mainshock_stack():
    assert driver_family("days_since_hikurangi_repeater") == "slowslip"


def test_new_series_can_reach_the_mainshock_stack():
    from eqpred.eval.mainshock import SHARED_FAMILIES

    assert driver_family("pore_u1518_kpa") == "pressure"
    assert driver_family("infra_is36_rms") == "infrasound"
    assert driver_family("schumann_mode1") == "schumann"
    assert driver_family("days_since_hikurangi_repeater") == "slowslip"
    assert {"infrasound", "pressure", "schumann"} <= set(SHARED_FAMILIES)


def test_a_later_report_cannot_move_an_earlier_morning():
    events = pd.DataFrame(
        [
            _row(datetime(2020, 1, 1, tzinfo=timezone.utc), -40.0, 177.0, 3.2),
            _row(datetime(2020, 3, 1, tzinfo=timezone.utc), -40.01, 177.01, 3.3),
        ]
    )
    mornings = pd.to_datetime(["2020-02-01", "2020-03-02", "2020-03-04"], utc=True)
    days = days_since_hikurangi_repeater(events, mornings)
    assert np.isnan(days[0])
    assert np.isnan(days[1])
    assert days[2] == 1.0


def test_a_ten_day_pair_is_an_aftershock_not_a_repeat():
    events = pd.DataFrame(
        [
            _row(datetime(2020, 1, 1, tzinfo=timezone.utc), -40.0, 177.0, 3.2),
            _row(datetime(2020, 1, 11, tzinfo=timezone.utc), -40.01, 177.01, 3.3),
        ]
    )
    mornings = pd.to_datetime(["2020-02-01"], utc=True)
    days = days_since_hikurangi_repeater(events, mornings)
    assert np.isnan(days[0])
