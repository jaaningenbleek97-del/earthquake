import numpy as np
import pandas as pd

from eqpred.eval.circus import calendar_foils, circus_checks, spearman_foil
from eqpred.eval.hunt import driver_family


def test_circus_catches_weekday_foil():
    n = 800
    days = pd.date_range("2016-01-01", periods=n, freq="D", tz="UTC")
    weekday = days.dayofweek.to_numpy(dtype=float)
    foils = calendar_foils(days)
    hold = np.ones(n, dtype=bool)
    hold[:200] = False
    checks = circus_checks(weekday, foils, hold)
    by_name = {row["foil"]: row for row in checks}
    assert by_name["cal_weekday"]["fail"] is True
    noise = np.sin(2.0 * np.pi * np.arange(n) / 37.0)
    quiet = spearman_foil(noise, foils["cal_weekday"].to_numpy())
    assert quiet["fail"] is False


def test_foil_family_is_not_a_precursor():
    assert driver_family("wiki_eq_views") == "foil"
    assert driver_family("vlf_amp") == "foil"
    assert driver_family("tir_ts") == "foil"
    assert driver_family("cal_weekday") == "foil"
    assert driver_family("cal_payday") == "foil"
    assert driver_family("cal_doy_sin") == "geometry"
    assert driver_family("dvv_hikurangi") == "wave2"
    assert driver_family("wspr_snr_med") == "wave2"
