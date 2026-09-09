import numpy as np
import pandas as pd

from eqpred.eval.nulls import _score, fusion_nulls, shift_drivers
from eqpred.models.etas import EtasParams
from eqpred.models.modulator import DRIVER_COLS, Modulator


def test_shift_drivers_moves_a_pulse():
    days = pd.date_range("2016-01-01", periods=20, freq="D", tz="UTC")
    feat = pd.DataFrame({"issue_time": days, "spf_significant_window": 0.0})
    feat.loc[5, "spf_significant_window"] = 1.0
    out = shift_drivers(feat, 3)
    assert out["spf_significant_window"].iloc[8] == 1.0
    assert out["spf_significant_window"].iloc[5] == 0.0


def test_aligned_gain_beats_shifted_gain_on_brier():
    lam = np.full(80, 2.0)
    hits = np.zeros(80)
    hits[10:16] = 1
    hold = np.ones(80, dtype=bool)
    aligned = np.ones(80)
    aligned[10:16] = 2.5
    shifted = np.ones(80)
    shifted[40:46] = 2.5
    kw = dict(mag=7.0, m0=4.5, b=1.2, horizon=7)
    good = _score(lam, aligned, hits, hold, **kw)
    bad = _score(lam, shifted, hits, hold, **kw)
    assert good["brier"] < bad["brier"]


def test_fusion_nulls_flat_g_is_shift_invariant():
    days = pd.date_range("2016-01-01", periods=1400, freq="D", tz="UTC")
    feat = pd.DataFrame({"issue_time": days})
    for col in DRIVER_COLS:
        feat[col] = 0.0
    feat["spf_significant_window"] = (days.dayofyear < 15).astype(float)
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(["2019-06-01T00:00:00Z"], utc=True),
            "magnitude": [7.1],
            "lon": [0.0],
            "lat": [0.0],
        }
    )
    etas = EtasParams(mu=0.4, k=0.01, alpha=1.0, c=0.05, p=1.1, m0=4.5)
    mod = Modulator(
        columns=list(DRIVER_COLS),
        beta=np.zeros(len(DRIVER_COLS) + 1),
        means=np.zeros(len(DRIVER_COLS)),
        scales=np.ones(len(DRIVER_COLS)),
    )
    rows = {r["name"]: r for r in fusion_nulls(feat, quakes, etas=etas, mod=mod, b=1.2, m0=4.5)}
    assert set(rows) == {
        "fusion_issued_m7",
        "fusion_nowcast_m7",
        "fusion_modulator_m7",
        "fusion_timeshift_180d",
        "fusion_climatology_m7",
        "fusion_etas_world",
    }
    assert rows["fusion_issued_m7"]["n"] > 100
    assert abs(rows["fusion_issued_m7"]["brier"] - rows["fusion_timeshift_180d"]["brier"]) < 1e-9
    assert rows["fusion_climatology_m7"]["mean_p"] is not None
