import numpy as np

from eqpred.eval.strategy_desk import (
    PRESETS,
    STRATEGIES,
    _matched,
    apply_alarm_rarity,
    apply_rarity,
    apply_shrink,
    apply_stay,
    resolve_strategy,
    shift_score,
)


def test_shrink_zero_is_the_usual_rate():
    climate = np.array([0.1, 0.2, 0.4])
    model = np.array([0.9, 0.9, 0.9])
    out = apply_shrink(model, climate, 0.0)
    assert np.allclose(out, climate)


def test_rarity_shrinks_a_rare_card_more():
    climate = np.full(20, 0.1)
    model = np.full(20, 0.5)
    hits = np.zeros(20)
    hits[0] = 1
    fit = np.ones(20, dtype=bool)
    rare = apply_rarity(model, hits, climate, fit, n0=50)
    hits[:] = 1
    common = apply_rarity(model, hits, climate, fit, n0=50)
    assert rare.mean() < common.mean()


def test_stay_home_keeps_a_losing_model_on_the_usual_rate():
    climate = np.full(30, 0.2)
    model = np.full(30, 0.9)
    hits = np.zeros(30)
    val = np.ones(30, dtype=bool)
    out = apply_stay(model, hits, climate, val)
    assert np.allclose(out, climate)


def test_quiet_morning_stays_on_the_usual_rate():
    climate = np.linspace(0.1, 0.3, 40)
    model = np.linspace(0.0, 1.0, 40)
    hits = (model > 0.8).astype(float)
    fit = np.ones(40, dtype=bool)
    val = fit
    out = apply_alarm_rarity(model, hits, climate, fit, val, share=0.2, n0=50)
    quiet = model < np.quantile(model, 0.8)
    assert np.allclose(out[quiet], climate[quiet])


def test_delay_reads_an_earlier_morning():
    out = shift_score(np.arange(5, dtype=float), 2)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == 0 and out[4] == 2


def test_named_backtest_rules_keep_their_knobs():
    names = {name for name, _label in STRATEGIES}
    assert set(PRESETS) <= names
    kind, share, n0, lag = resolve_strategy("alarm_14_40", 0.2, 50, 0)
    assert (kind, share, lag) == ("alarm", 0.4, 14)
    kind, share, n0, lag = resolve_strategy("rarity_3_200", 0.2, 50, 0)
    assert (kind, n0, lag) == ("rarity", 200, 3)


def test_gain_interval_stays_above_zero_when_the_call_matches_the_quake():
    from eqpred.eval.strategy_desk import gain_band

    y = np.array([1, 0, 1, 0, 1, 0] * 8, dtype=float)
    climate = np.full(len(y), 0.2)
    good = np.where(y > 0.5, 0.8, 0.2)
    band = gain_band(good, y, climate, np.ones(len(y), dtype=bool))
    assert band is not None
    assert band["low"] > 0
    assert band["tests_agree"]


def test_matched_morning_does_not_copy_itself():
    matrix = np.eye(8)
    hits = np.zeros(8)
    hits[0] = 1
    fit = np.ones(8, dtype=bool)
    leaked = _matched(matrix, hits, fit, np.zeros(8, dtype=bool), k=5, purge=0)
    held = _matched(matrix, hits, fit, np.zeros(8, dtype=bool), k=5, purge=2)
    assert leaked[0] > held[0]
    assert held[0] < 1e-3
