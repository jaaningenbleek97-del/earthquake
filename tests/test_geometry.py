from datetime import datetime, timezone

from eqpred.ingest.geometry import _lunar_state, _tidal_proxy, pull_geometry


def test_new_and_full_moon_raise_tide():
    assert _tidal_proxy(60.0, 0.0) > _tidal_proxy(60.0, 90.0)
    assert _tidal_proxy(60.0, 180.0) > _tidal_proxy(60.0, 90.0)
    phase, dist, _ = _lunar_state(datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert 0 <= phase < 360
    assert dist > 50


def test_geometry_daily_rows():
    frame = pull_geometry(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 10, tzinfo=timezone.utc),
        ingest_run_id="t",
    )
    assert len(frame) == 9
    assert frame["tidal_strain"].notna().all()
