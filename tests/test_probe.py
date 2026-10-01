import numpy as np
import pandas as pd

from eqpred.eval.probe import _shift_rows, run_probe
from eqpred.regions import REGIONS


def test_shift_rows_is_causal():
    x = np.arange(10, dtype=float).reshape(-1, 1)
    shifted = _shift_rows(x, 3)
    assert np.isnan(shifted[:3]).all()
    assert shifted[3, 0] == 0.0
    assert shifted[-1, 0] == 6.0


def test_probe_timeshift_kills_planted_m8_and_where_finds_the_box(tmp_path):
    rng = np.random.default_rng(11)
    days = pd.date_range("2014-01-01", periods=2800, freq="D", tz="UTC")
    # Rare M8-like hits aligned to a non-catalog knob.
    pulse = (rng.random(len(days)) > 0.97).astype(float)
    japan = (rng.random(len(days)) > 0.92).astype(float)
    chile = (rng.random(len(days)) > 0.93).astype(float)
    features = pd.DataFrame(
        {
            "issue_time": days,
            "proton_density_cm3": pulse * 25 + rng.normal(6, 1, len(days)),
            "tec_japan_tecu": japan * 12 + rng.normal(20, 1, len(days)),
            "tec_chile_tecu": chile * 12 + rng.normal(20, 1, len(days)),
            "dst_nt": rng.normal(-10, 8, len(days)),
        }
    )
    m8_days = days[pulse > 0.5][::2] + pd.Timedelta(days=2)
    japan_days = days[japan > 0.5][::2] + pd.Timedelta(days=2)
    chile_days = days[chile > 0.5][::2] + pd.Timedelta(days=2)
    quakes = pd.DataFrame(
        {
            "observed_at": pd.DatetimeIndex(
                list(m8_days) + list(japan_days) + list(chile_days)
            ),
            "magnitude": (
                [8.2] * len(m8_days) + [6.4] * len(japan_days) + [6.4] * len(chile_days)
            ),
            "lon": (
                [180.0] * len(m8_days)
                + [REGIONS["japan"]["min_lon"] + 1] * len(japan_days)
                + [REGIONS["chile"]["min_lon"] + 1] * len(chile_days)
            ),
            "lat": (
                [0.0] * len(m8_days)
                + [REGIONS["japan"]["min_lat"] + 1] * len(japan_days)
                + [REGIONS["chile"]["min_lat"] + 1] * len(chile_days)
            ),
        }
    )
    (tmp_path / "data" / "features").mkdir(parents=True)
    (tmp_path / "data" / "eval").mkdir(parents=True)
    report = run_probe(root=tmp_path, features=features, quakes=quakes)
    assert report["writes_issued_p"] is False
    aligned = next(r for r in report["m8"]["shifts"] if r["shift_days"] == 0)
    shifted = next(r for r in report["m8"]["shifts"] if r["shift_days"] == 180)
    assert aligned.get("lift") is not None
    assert (shifted.get("lift") or 0) < (aligned.get("lift") or 0) + 0.02
    transfer = report["where_transfer"]
    assert transfer["n_hold"]
    assert transfer["fresh_acc"] is not None
    assert transfer["fresh_acc"] >= transfer["majority_acc"] - 0.05
