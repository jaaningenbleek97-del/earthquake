import numpy as np
import pandas as pd

from eqpred.models.modulator import DRIVER_COLS, fit_modulator


def test_modulator_raises_rate_when_window_predicts_counts():
    n = 400
    days = pd.date_range("2010-01-01", periods=n, freq="D", tz="UTC")
    window = (np.arange(n) % 20 < 8).astype(float)
    features = pd.DataFrame({"issue_time": days})
    for col in DRIVER_COLS:
        features[col] = 0.0
    features["spf_significant_window"] = window
    features["proton_density_lag1d"] = 4.0
    etas = np.full(n, 0.05)
    counts = (window * 2 + np.random.default_rng(0).poisson(0.02, n)).astype(float)
    mask = np.ones(n, dtype=bool)
    mod = fit_modulator(features, etas, counts, mask)
    gain = mod.gain(features)
    assert gain[window == 1].mean() > gain[window == 0].mean()
