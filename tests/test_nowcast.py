import numpy as np
import pandas as pd

from eqpred.features.nowcast import catalog_nowcast_frame
from eqpred.models.nowcast import fit_nowcast_gain, identity_nowcast


def test_identity_nowcast_gain_is_one():
    feat = pd.DataFrame({"b_90d": [1.0, 1.2]})
    gain = identity_nowcast().gain(feat)
    assert np.allclose(gain, 1.0)


def test_nowcast_energy_rises_after_visible_m7():
    issue = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(["2020-01-10T00:00:00Z"], utc=True),
            "available_at": pd.to_datetime(["2020-01-10T06:00:00Z"], utc=True),
            "magnitude": [7.2],
        }
    )
    frame = catalog_nowcast_frame(quakes, issue, m0=4.5)
    before = float(frame.loc[frame["issue_time"] == issue[8], "log_energy_7d"].iloc[0])
    after = float(frame.loc[frame["issue_time"] == issue[11], "log_energy_7d"].iloc[0])
    assert after > before


def test_nowcast_uses_available_at_not_origin():
    issue = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(["2020-01-02T00:00:00Z"], utc=True),
            "available_at": pd.to_datetime(["2020-01-08T00:00:00Z"], utc=True),
            "magnitude": [7.0],
        }
    )
    frame = catalog_nowcast_frame(quakes, issue, m0=4.5)
    early = float(frame.loc[frame["issue_time"] == issue[3], "log_energy_7d"].iloc[0])
    late = float(frame.loc[frame["issue_time"] == issue[8], "log_energy_7d"].iloc[0])
    assert early == 0.0
    assert late > 0.0


def test_nowcast_gain_issued_only_if_train_brier_falls():
    rng = np.random.default_rng(2)
    n = 400
    feat = pd.DataFrame(
        {
            "b_90d": rng.normal(1.2, 0.05, n),
            "log_energy_7d": rng.normal(2.0, 0.4, n),
            "log_energy_30d": rng.normal(3.0, 0.4, n),
            "energy_accel": rng.normal(1.0, 0.2, n),
        }
    )
    lam = rng.uniform(2.0, 6.0, n)
    hits = (rng.random(n) < 0.2).astype(float)
    mask = np.ones(n, dtype=bool)
    _model, meta = fit_nowcast_gain(
        feat, lam, hits, mask, scale=1.0, mag=7.0, m0=4.5, b=1.2, horizon=7
    )
    assert "used_on_issued" in meta
    assert "train_brier_scale_only" in meta
