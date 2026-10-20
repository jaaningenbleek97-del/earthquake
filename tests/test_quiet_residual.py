import numpy as np
import pandas as pd

from eqpred.eval.hunt import driver_family
from eqpred.eval.mainshock import is_issued_driver
from eqpred.features.quiet_residual import (
    EAST_TRANSIENT_MM,
    expanding_residual,
    quiet_mornings,
    quiet_residual_columns,
)
from eqpred.ingest.aardvark import parse_aardvark
from eqpred.ingest.unwin import cusp_onset_rows


def test_storm_mornings_are_not_quiet():
    assert quiet_mornings(np.array([0.0, -30.0, 5.0]), np.array([1.0, 1.0, 4.0])).tolist() == [True, False, False]


def test_residual_uses_only_earlier_quiet_mornings():
    n = 50
    f107 = np.linspace(80, 120, n)
    y = 2.0 * (f107 / 100.0)
    drivers = np.column_stack([np.ones(n), f107 / 100.0])
    quiet = np.ones(n, dtype=bool)
    quiet[45] = False
    out = expanding_residual(y, drivers, quiet)
    assert np.isnan(out[45])
    assert np.isnan(out[10])
    assert abs(out[49]) < 1e-6


def test_laic_is_the_hikurangi_residual_only_when_slip_and_quiet():
    n = 60
    frame = pd.DataFrame(
        {
            "dst_nt": np.zeros(n),
            "kp": np.ones(n),
            "f107": np.linspace(90, 110, n),
            "ae_nt": np.linspace(40, 80, n),
            "solar_wind_speed_kms": np.linspace(350, 450, n),
            "tec_hikurangi_tecu": np.linspace(10, 12, n),
            "sse_hikurangi": np.zeros(n),
            "gnss_wgtn_east_extra_30d": np.zeros(n),
        }
    )
    frame.loc[50, "dst_nt"] = -40
    frame.loc[55, "sse_hikurangi"] = 1
    frame.loc[56, "gnss_wgtn_east_extra_30d"] = EAST_TRANSIENT_MM
    out = quiet_residual_columns(frame)
    assert "laic_hikurangi" in out.columns
    assert np.isnan(out.loc[50, "laic_hikurangi"])
    assert np.isfinite(out.loc[55, "laic_hikurangi"])
    assert np.isfinite(out.loc[56, "laic_hikurangi"])
    assert "qres_tec_alpine" not in out.columns or "laic_alpine" not in out.columns
    assert is_issued_driver("qres_tec_hikurangi") is False
    assert is_issued_driver("laic_hikurangi") is False
    assert is_issued_driver("aardvark_nwc_amp") is False
    assert driver_family("laic_hikurangi") == "quiet"


def test_aardvark_needs_amplitude_and_phase():
    text = "time,amplitude,phase\n2020-01-01T00:00:00Z,12.5,40\n2020-01-01T06:00:00Z,13.5,42\n"
    frame = parse_aardvark(text)
    assert len(frame) == 1
    assert abs(frame.iloc[0]["aardvark_nwc_amp"] - 13.0) < 1e-9
    assert parse_aardvark("time,amplitude\n2020-01-01,1\n").empty


def test_cusp_index_keeps_new_zealand_longitude_on_storm_onset():
    gates = pd.DataFrame(
        {
            "observed_at": ["2020-01-02", "2020-01-02", "2020-01-03"],
            "glat": [-70.0, -40.0, -70.0],
            "glon": [170.0, 170.0, 170.0],
            "power": [8.0, 99.0, 5.0],
        }
    )
    kept = cusp_onset_rows(gates, quiet_yesterday=np.array([True, False]), storm_today=np.array([True, True]))
    assert len(kept) == 1
    assert abs(float(kept.iloc[0]["sdarn_cusp_power"]) - 8.0) < 1e-9
