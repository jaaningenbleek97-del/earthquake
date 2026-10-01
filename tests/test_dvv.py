from datetime import datetime, timezone

import numpy as np

from eqpred.ingest.dvv import read_mseed_traces, stretch_dvv, traces_to_dvv, write_mseed_int32
from eqpred.ingest.run import WAVE4_SOURCES
from eqpred.ingest.watchdog import SPECS


def test_stretch_recovers_known_dvv():
    fs = 50.0
    t = np.arange(0, 25.0, 1.0 / fs)
    ref = np.sin(2.0 * np.pi * 1.1 * t) + 0.3 * np.sin(2.0 * np.pi * 1.8 * t)
    delta = 0.008
    current = np.interp(t / (1.0 + delta), t, ref, left=0.0, right=0.0)
    dvv, cc = stretch_dvv(current, ref, fs)
    assert abs(dvv - (-delta)) < 0.003
    assert cc > 0.85


def test_mseed_int32_roundtrip():
    start = datetime(2024, 1, 1, 11, tzinfo=timezone.utc)
    samples = np.arange(80, dtype=np.int32)
    blob = write_mseed_int32(start, 50.0, samples)
    traces = read_mseed_traces(blob)
    assert traces
    when, fs, data = traces[0]
    assert abs(fs - 50.0) < 1e-6
    assert when.year == 2024
    assert np.allclose(data[: len(samples)], samples.astype(float))


def test_traces_to_dvv_needs_a_few_nights():
    fs = 40.0
    t = np.arange(0, 20.0, 1.0 / fs)
    windows = []
    for i in range(6):
        when = datetime(2024, 2, 1 + i, 11, tzinfo=timezone.utc)
        windows.append((when, fs, np.sin(2.0 * np.pi * 1.0 * t)))
    daily = traces_to_dvv(windows)
    assert not daily.empty
    assert "dvv" in daily.columns


def test_wave4_and_watchdog_include_new_sources():
    assert "geonet.dvv" in WAVE4_SOURCES
    assert "ham.wspr" in WAVE4_SOURCES
    assert "wiki.pageviews" in WAVE4_SOURCES
    names = {spec.source for spec in SPECS}
    assert {"geonet.dvv", "ham.wspr", "wiki.pageviews"} <= names
