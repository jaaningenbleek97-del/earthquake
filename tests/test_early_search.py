import numpy as np

from eqpred.eval.early_search import pick_best_alarm


def _masks(n=400):
    early = np.zeros(n, dtype=bool)
    develop = np.zeros(n, dtype=bool)
    early[:120] = True
    develop[120:240] = True
    return early, develop


def test_the_later_years_do_not_choose_the_variation():
    n = 400
    signal = np.linspace(0, 1, n)
    hits = np.zeros(n)
    hits[:120][signal[:120] > 0.7] = 1
    hits[120:240][signal[120:240] > 0.7] = 1
    climate = np.full(n, 0.2)
    early, develop = _masks(n)
    first, tried = pick_best_alarm({"model": signal}, hits, climate, early, develop)
    hits[240:] = 1
    second, _ = pick_best_alarm({"model": signal}, hits, climate, early, develop)
    assert tried > 0
    assert first is not None and second is not None
    assert first["signal"] == second["signal"]
    assert first["side"] == second["side"]
    assert first["share"] == second["share"]
    assert first["lag_days"] == second["lag_days"]
    assert first["threshold"] == second["threshold"]


def test_a_flat_early_record_is_not_adopted():
    n = 400
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    hits = np.zeros(n)
    climate = np.full(n, 0.25)
    early, develop = _masks(n)
    choice, tried = pick_best_alarm({"noise": signal}, hits, climate, early, develop)
    assert tried == 0 or choice is None or choice["develop"]["verdict"] != "up" or (choice["develop"]["information_gain"] or 0) <= 0
