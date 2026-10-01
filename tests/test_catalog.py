from datetime import datetime, timezone

import pandas as pd

from eqpred.catalog import (
    apply_available_at,
    catalog_asof,
    completeness_mag,
    event_available_as,
    reporting_delay_hours,
)
from eqpred.features.nowcast import catalog_nowcast_frame
from eqpred.models.etas import EtasParams, daily_background_and_cluster


def test_big_events_report_faster_than_small():
    assert reporting_delay_hours(7.6) < reporting_delay_hours(7.0)
    assert reporting_delay_hours(7.0) < reporting_delay_hours(6.0)
    assert reporting_delay_hours(6.0) < reporting_delay_hours(5.0)
    assert reporting_delay_hours(4.6) == 48.0


def test_catalog_asof_hides_unreported_aftershock():
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime([t0, t0 + pd.Timedelta(hours=3)], utc=True),
            "magnitude": [7.2, 5.1],
            "lon": [140.0, 140.1],
            "lat": [38.0, 38.1],
        }
    )
    quakes = apply_available_at(quakes)
    morning = datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
    visible = catalog_asof(quakes, morning)
    assert list(visible["magnitude"]) == [7.2]
    later = datetime(2020, 1, 3, tzinfo=timezone.utc)
    assert len(catalog_asof(quakes, later)) == 2


def test_etas_waits_for_available_at():
    t0 = datetime(2010, 1, 1, tzinfo=timezone.utc)
    events = pd.DataFrame(
        {
            "observed_at": pd.to_datetime([t0], utc=True),
            "available_at": pd.to_datetime(["2010-01-06T12:00:00Z"], utc=True),
            "magnitude": [7.2],
        }
    )
    days = pd.date_range("2010-01-01", periods=20, freq="D", tz="UTC")
    params = EtasParams(mu=0.05, k=0.2, alpha=1.0, c=0.05, p=1.1)
    rate = daily_background_and_cluster(events, days, params)
    assert rate[0] == params.mu
    assert rate[5] == params.mu
    assert rate[6] > params.mu


def _revision_frames():
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    quakes = pd.DataFrame(
        {
            "event_id": ["us0rev"],
            "observed_at": pd.to_datetime([t0], utc=True),
            "magnitude": [7.4],
            "lon": [142.0],
            "lat": [38.0],
        }
    )
    versions = pd.DataFrame(
        {
            "event_id": ["us0rev", "us0rev"],
            "version_at": pd.to_datetime(
                ["2020-01-01T02:00:00Z", "2020-01-03T00:00:00Z"], utc=True
            ),
            "observed_at": pd.to_datetime([t0, t0], utc=True),
            "magnitude": [6.8, 7.4],
            "lon": [142.0, 142.1],
            "lat": [38.0, 38.1],
            "status": ["automatic", "reviewed"],
        }
    )
    return quakes, versions


def test_catalog_asof_uses_revision_not_final_mag():
    quakes, versions = _revision_frames()
    early = catalog_asof(quakes, datetime(2020, 1, 2, tzinfo=timezone.utc), versions=versions)
    assert list(early["magnitude"]) == [6.8]
    late = catalog_asof(quakes, datetime(2020, 1, 4, tzinfo=timezone.utc), versions=versions)
    assert list(late["magnitude"]) == [7.4]


def test_catalog_asof_hides_deleted_event():
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    quakes = pd.DataFrame(
        {
            "event_id": ["us0del"],
            "observed_at": pd.to_datetime([t0], utc=True),
            "magnitude": [6.5],
        }
    )
    versions = pd.DataFrame(
        {
            "event_id": ["us0del", "us0del"],
            "version_at": pd.to_datetime(
                ["2020-01-01T03:00:00Z", "2020-01-02T00:00:00Z"], utc=True
            ),
            "observed_at": pd.to_datetime([t0, t0], utc=True),
            "magnitude": [6.5, 6.5],
            "status": ["automatic", "deleted"],
        }
    )
    seen = catalog_asof(quakes, datetime(2020, 1, 1, 12, tzinfo=timezone.utc), versions=versions)
    gone = catalog_asof(quakes, datetime(2020, 1, 3, tzinfo=timezone.utc), versions=versions)
    assert list(seen["event_id"]) == ["us0del"]
    assert gone.empty


def test_catalog_asof_mc_hides_later_added_incomplete():
    t0 = datetime(1995, 6, 1, tzinfo=timezone.utc)
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime([t0], utc=True),
            "magnitude": [4.6],
            "lon": [140.0],
            "lat": [38.0],
        }
    )
    quakes = apply_available_at(quakes)
    assert completeness_mag(t0) == 5.0
    visible = catalog_asof(quakes, datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert visible.empty


def test_first_report_beats_delay_from_final_mag():
    quakes, versions = _revision_frames()
    # Final mag 7.4 would be available in 6 hours. First origin is at +2h,
    # but the 7.4 product is not on the wire until Jan 3.
    stamped = apply_available_at(quakes, versions=versions)
    assert stamped.iloc[0]["available_at"] == pd.Timestamp("2020-01-01T02:00:00Z")
    first7 = event_available_as(quakes, 7.0, versions)
    assert first7.iloc[0] == pd.Timestamp("2020-01-03T00:00:00Z")


def test_etas_uses_asof_mag_not_final():
    quakes, versions = _revision_frames()
    quakes = apply_available_at(quakes, versions=versions)
    days = pd.date_range("2020-01-01", periods=8, freq="D", tz="UTC")
    params = EtasParams(mu=0.05, k=0.2, alpha=1.0, c=0.05, p=1.1)
    leaky = daily_background_and_cluster(quakes, days, params)
    honest = daily_background_and_cluster(quakes, days, params, versions=versions)
    small = quakes.copy()
    small["magnitude"] = 6.8
    only68 = daily_background_and_cluster(small, days, params)
    # Day 1 (2020-01-02) sees only M6.8. Final-mag λ would already use 7.4.
    assert honest[1] > params.mu
    assert abs(honest[1] - only68[1]) < 1e-9
    assert honest[1] < leaky[1]
    # After the 7.4 product, piecewise matches the final-mag path.
    assert (abs(honest[2:] - leaky[2:]) < 1e-9).all()


def test_nowcast_energy_uses_asof_mag():
    quakes, versions = _revision_frames()
    quakes = apply_available_at(quakes, versions=versions)
    issue = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    leaky = catalog_nowcast_frame(quakes, issue, m0=4.5)
    honest = catalog_nowcast_frame(quakes, issue, m0=4.5, versions=versions)
    early = issue[1]
    assert float(honest.loc[honest["issue_time"] == early, "log_energy_7d"].iloc[0]) < float(
        leaky.loc[leaky["issue_time"] == early, "log_energy_7d"].iloc[0]
    )


def test_bulk_recatalog_stamp_does_not_hide_old_quake():
    """A 1990 M6.5 re-reviewed in bulk in 2014 must not look unknown until 2014."""
    t0 = datetime(1990, 1, 4, tzinfo=timezone.utc)
    bulk = datetime(2014, 11, 7, tzinfo=timezone.utc)
    quakes = pd.DataFrame(
        {
            "event_id": ["usp00043wc"],
            "observed_at": pd.to_datetime([t0], utc=True),
            "magnitude": [6.5],
            "lon": [140.0],
            "lat": [38.0],
        }
    )
    versions = pd.DataFrame(
        {
            "event_id": ["usp00043wc"],
            "version_at": pd.to_datetime([bulk], utc=True),
            "observed_at": pd.to_datetime([t0], utc=True),
            "magnitude": [6.5],
            "status": ["reviewed"],
        }
    )
    cutoff = pd.Timestamp("1990-02-01T00:00:00Z")

    assert apply_available_at(quakes, versions=versions).iloc[0]["available_at"] < cutoff
    assert event_available_as(quakes, 6.0, versions).iloc[0] < cutoff

    seen = catalog_asof(
        quakes,
        datetime(1990, 6, 1, tzinfo=timezone.utc),
        versions=versions,
        apply_mc=False,
    )
    assert list(seen["event_id"]) == ["usp00043wc"]
