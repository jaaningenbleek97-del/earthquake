import numpy as np
import pandas as pd

from eqpred.eval.score import summarize_ledger
from eqpred.models.gr import fit_b_value, poisson_at_least_one, rate_to_p, rates_to_p, scale_rate
from eqpred.models.predict import BOX_MAG, GLOBAL_MAG, KERMADEC_MAG, NZ_MAG, QUIET_MAG


def test_scale_rate_gr_factor():
    assert scale_rate(1.0, 6.0, 4.5, 1.0) == 10 ** (-1.5)
    assert scale_rate(10.0, 4.5, 4.5, 1.0) == 10.0
    assert scale_rate(10.0, 4.0, 4.5, 1.0) == 10.0


def test_issued_global_floor_is_m7():
    assert GLOBAL_MAG == 7.0
    assert QUIET_MAG == 7.0
    assert NZ_MAG == 5.0
    assert KERMADEC_MAG == 6.0
    assert BOX_MAG["kermadec"] == 6.0
    assert BOX_MAG["hikurangi"] == 5.0


def test_m6_p_is_far_below_m45_saturation():
    lam = 9.0
    p_m45 = rate_to_p(lam, 4.5, 4.5, 1.0, 7)
    p_m6 = rate_to_p(lam, 6.0, 4.5, 1.0, 7)
    p_m7 = rate_to_p(lam, 7.0, 4.5, 1.0, 7)
    assert p_m45 > 0.99
    assert 0.05 < p_m6 < 0.9
    assert 0.01 < p_m7 < p_m6


def test_rates_to_p_matches_scalar():
    vec = rates_to_p(np.array([1.0, 9.0]), 7.0, 4.5, 1.0, 7)
    assert abs(float(vec[0]) - rate_to_p(1.0, 7.0, 4.5, 1.0, 7)) < 1e-12
    assert vec[1] > vec[0]


def test_poisson_bounds():
    assert poisson_at_least_one(0.0, 7) == 0.0
    assert poisson_at_least_one(100.0, 7) > 0.999


def test_aki_utsu_recovers_synthetic_b():
    rng = np.random.default_rng(1)
    b_true = 1.05
    m0 = 4.5
    u = rng.random(8000)
    mag = np.round(m0 - np.log10(np.clip(u, 1e-12, 1.0)) / b_true, 1)
    mag = mag[mag >= m0]
    hat = fit_b_value(mag, m0)
    assert abs(hat - b_true) < 0.12


def test_summarize_splits_fusion_by_region():
    ledger = pd.DataFrame(
        {
            "model_id": ["fusion.etas_x_mod.v1"] * 4,
            "region": ["global", "global", "new_zealand", "new_zealand"],
            "magnitude_min": [6.0, 6.0, 5.0, 5.0],
            "kind": ["probability"] * 4,
            "issued_at": ["2019-01-01T00:00:00Z"] * 4,
            "value": [0.8, 0.7, 0.1, 0.2],
            "competitor_etas_p": [0.75, 0.65, 0.08, 0.12],
            "hit": [1, 1, 0, 1],
        }
    )
    summary = summarize_ledger(ledger)
    assert summary["n_cards"] == 4
    by_region = {row["region"]: row for row in summary["heads"]}
    assert by_region["global"]["hit_rate"] == 1.0
    assert by_region["new_zealand"]["n"] == 2
    assert by_region["global"]["mean_p"] == 0.75
    assert by_region["new_zealand"]["magnitude_min"] == 5.0
    assert by_region["global"]["brier"] is not None
