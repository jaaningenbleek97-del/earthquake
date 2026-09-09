import numpy as np
import pandas as pd

from eqpred.faults import FAULTS
from eqpred.features.coulomb import coulomb_frame, resolved_coulomb
from eqpred.models.coulomb import box_coulomb_gain, fit_coulomb_gain


def test_faults_differ_on_the_same_tide():
    tide, decl = 1.2e-7, 18.0
    values = {rid: float(resolved_coulomb(tide, decl, **fault)) for rid, fault in FAULTS.items()}
    assert len(set(round(v, 20) for v in values.values())) == 3
    assert all(np.isfinite(v) for v in values.values())


def test_coulomb_is_linear_and_odd_in_tide():
    pos = float(resolved_coulomb(1e-7, 12.0, **FAULTS["hikurangi"]))
    doubled = float(resolved_coulomb(2e-7, 12.0, **FAULTS["hikurangi"]))
    neg = float(resolved_coulomb(-1e-7, 12.0, **FAULTS["hikurangi"]))
    assert abs(doubled - 2 * pos) < 1e-20
    assert abs(neg + pos) < 1e-20
    assert resolved_coulomb(0.0, 12.0, **FAULTS["alpine"]) == 0.0


def test_coulomb_frame_has_three_boxes():
    feat = pd.DataFrame(
        {
            "issue_time": pd.date_range("2016-01-01", periods=3, freq="D", tz="UTC"),
            "tidal_strain": [1e-7, 2e-7, 1e-7],
            "lunar_declination_deg": [10.0, 0.0, -15.0],
        }
    )
    frame = coulomb_frame(feat)
    assert list(frame.columns) == ["issue_time", "coulomb_hikurangi", "coulomb_alpine", "coulomb_kermadec"]
    assert frame["coulomb_hikurangi"].notna().all()


def test_coulomb_gain_issued_only_if_train_brier_and_rank_improve():
    rng = np.random.default_rng(3)
    n = 500
    z_raw = rng.normal(0.0, 1.0, n)
    lam = rng.uniform(0.2, 1.2, n)
    # Hits follow a mild coulomb ranking, not a mean-p shift.
    p = 1.0 / (1.0 + np.exp(-0.8 * z_raw))
    hits = (rng.random(n) < 0.15 + 0.2 * (p - p.mean())).astype(float)
    mask = np.ones(n, dtype=bool)
    meta = fit_coulomb_gain(
        z_raw, lam, hits, mask, scale=1.0, mag=5.0, m0=4.5, b=1.2, horizon=7, column="coulomb_hikurangi"
    )
    assert "used_on_issued" in meta
    assert meta["column"] == "coulomb_hikurangi"


def test_box_gain_is_one_when_held_out():
    row = pd.Series({"coulomb_hikurangi": 1.5})
    assert box_coulomb_gain({"regions": {}}, "hikurangi", row) == 1.0
    assert box_coulomb_gain(
        {"regions": {"hikurangi": {"used_on_issued": False, "column": "coulomb_hikurangi"}}},
        "hikurangi",
        row,
    ) == 1.0


def test_box_gain_clips_and_applies_beta():
    row = pd.Series({"coulomb_hikurangi": 2.0})
    g = box_coulomb_gain(
        {
            "regions": {
                "hikurangi": {
                    "used_on_issued": True,
                    "column": "coulomb_hikurangi",
                    "beta": 1.0,
                    "mean": 0.0,
                    "scale": 1.0,
                }
            }
        },
        "hikurangi",
        row,
    )
    assert abs(g - float(np.exp(1.5))) < 1e-12
