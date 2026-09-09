import numpy as np

from eqpred.models.calibrate import fit_rate_scale, issued_scale
from eqpred.models.gr import rates_to_p


def test_scale_recovers_known_multiplier():
    rng = np.random.default_rng(4)
    lam = rng.uniform(2.0, 8.0, size=400)
    true_s = 3.0
    p = rates_to_p(true_s * lam, 7.0, 4.5, 1.2, 7)
    hits = (rng.random(400) < p).astype(float)
    hat = fit_rate_scale(lam, hits, np.ones(400, dtype=bool), mag=7.0, m0=4.5, b=1.2, horizon=7)
    assert 1.5 < hat < 6.0


def test_low_rate_vs_frequent_hits_raises_scale():
    lam = np.full(200, 0.5)
    hits = np.ones(200)
    hat = fit_rate_scale(lam, hits, np.ones(200, dtype=bool), mag=7.0, m0=4.5, b=1.2, horizon=7)
    assert hat > 2.0


def test_issued_scale_reads_region_then_global():
    cal = {"scale": 4.3, "scales": {"global": 4.3, "hikurangi": 1.8}}
    assert issued_scale(cal, "global") == 4.3
    assert issued_scale(cal, "hikurangi") == 1.8
    assert issued_scale(cal, "kermadec") == 1.0
    assert issued_scale({"scale": 2.0}, "global") == 2.0


def test_issued_scale_reads_per_horizon_block():
    cal = {
        "scale": 4.3,
        "scales": {"global": 4.3, "hikurangi": 1.8},
        "by_horizon": {
            "1": {"scale": 2.0, "scales": {"global": 2.0, "hikurangi": 0.9}},
            "90": {"scale": 8.0, "scales": {"global": 8.0}},
        },
    }
    assert issued_scale(cal, "global", 1) == 2.0
    assert issued_scale(cal, "hikurangi", 1) == 0.9
    assert issued_scale(cal, "global", 90) == 8.0
    assert issued_scale(cal, "global", 7) == 4.3
