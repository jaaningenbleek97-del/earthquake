import numpy as np
import pytest

from eqpred.eval.csep import l_test, molchan, n_test, s_test, score_forecast, t_test


def test_n_test_passes_when_count_matches():
    rng = np.random.default_rng(3)
    p = np.full(200, 0.2)
    hits = (rng.random(200) < 0.2).astype(float)
    row = n_test(p, hits)
    assert row["pass"] is True
    assert row["n_forecast"] == pytest.approx(40.0)
    assert abs(row["n_forecast"] - row["n_observed"]) < 20


def test_n_test_fails_when_forecast_is_wildly_high():
    p = np.full(80, 0.9)
    hits = np.zeros(80)
    row = n_test(p, hits)
    assert row["pass"] is False


def test_molchan_perfect_ranking():
    p = np.linspace(0.01, 0.99, 100)
    hits = (p >= 0.8).astype(float)
    row = molchan(p, hits)
    assert row["area_skill_score"] > 0.5


def test_molchan_random_near_zero():
    rng = np.random.default_rng(1)
    p = rng.random(400)
    hits = (rng.random(400) < 0.15).astype(float)
    row = molchan(p, hits)
    assert abs(row["area_skill_score"]) < 0.25


def test_s_test_passes_when_boxes_match():
    expected = {"global": 40.0, "hikurangi": 20.0, "alpine": 5.0}
    observed = {"global": 38.0, "hikurangi": 21.0, "alpine": 6.0}
    row = s_test(expected, observed, n_sim=400, seed=2)
    assert row["pass"] is True


def test_score_forecast_bundle():
    rng = np.random.default_rng(8)
    p = np.clip(rng.normal(0.15, 0.05, 120), 0.01, 0.6)
    hits = (rng.random(120) < p).astype(float)
    pack = score_forecast(p, hits)
    assert pack["n_windows"] == 120
    assert "n" in pack["tests"]
    assert "molchan" in pack["tests"]
    _ = l_test(p, hits)
    _ = t_test(p, hits)
