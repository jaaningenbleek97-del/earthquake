import numpy as np

from eqpred.eval.new_strategies import (
    apply_after_quiet,
    apply_believer,
    apply_compound,
    apply_crest,
    apply_crossing,
    apply_equity,
    apply_fade,
    apply_markov,
    apply_month_rate,
    apply_quiet_catalog,
    apply_skeptic,
    apply_veto,
)


def _series():
    model = np.linspace(0.0, 1.0, 80)
    hits = np.zeros(80)
    climate = np.full(80, 0.2)
    fit = np.zeros(80, dtype=bool)
    fit[:60] = True
    val = fit.copy()
    return model, hits, climate, fit, val


def test_crest_needs_a_rise():
    model, hits, climate, fit, val = _series()
    model[:] = 0.9
    model[40] = 0.1
    out = apply_crest(model, hits, climate, fit, val)
    assert np.allclose(out[40], climate[40])


def test_crossing_is_only_the_first_loud_morning():
    model = np.zeros(120)
    hits = np.zeros(120)
    climate = np.full(120, 0.2)
    fit = np.ones(120, dtype=bool)
    for start in range(0, 120, 8):
        model[start : start + 4] = 1.0
        hits[start] = 1
    out = apply_crossing(model, hits, climate, fit, fit)
    assert out[0] > climate[0]
    assert np.allclose(out[1], climate[1])


def test_after_quiet_stays_home_when_the_last_window_hit():
    model, hits, climate, fit, val = _series()
    model[:] = 1.0
    hits[:] = 1
    out = apply_after_quiet(model, hits, climate, fit, val, horizon=5)
    assert np.allclose(out[20], climate[20])


def test_month_rate_uses_only_the_fit():
    hits = np.zeros(40)
    hits[:10] = 1
    climate = np.full(40, 0.25)
    fit = np.zeros(40, dtype=bool)
    fit[:20] = True
    month = np.array([1] * 20 + [1] * 20)
    out = apply_month_rate(hits, climate, fit, month)
    assert np.allclose(out, 0.5)


def test_equity_waits_for_a_closed_record():
    model = np.full(30, 0.8)
    hits = np.zeros(30)
    climate = np.full(30, 0.2)
    out = apply_equity(model, hits, climate, horizon=7, lookback=10)
    assert np.allclose(out[0], climate[0])


def test_fade_does_nothing_when_hot_mornings_hit_more():
    model, hits, climate, fit, val = _series()
    hits[model > 0.8] = 1
    out = apply_fade(model, hits, climate, fit, val)
    assert np.allclose(out, climate)


def test_skeptic_never_raises_the_usual_rate():
    model = np.array([0.1, 0.8, 0.2])
    climate = np.array([0.2, 0.2, 0.2])
    out = apply_skeptic(model, climate)
    assert out[0] < climate[0]
    assert np.isclose(out[1], climate[1])


def test_believer_never_lowers_the_usual_rate():
    model = np.array([0.1, 0.8])
    climate = np.array([0.2, 0.2])
    out = apply_believer(model, climate)
    assert np.isclose(out[0], climate[0])
    assert out[1] > climate[1]


def test_one_day_chances_stack_into_the_longer_window():
    short = np.full(5, 0.01)
    climate = np.full(5, 0.2)
    out = apply_compound(short, climate, 30, True)
    assert np.allclose(out, 1.0 - (0.99 ** 30))
    skipped = apply_compound(short, climate, 30, False)
    assert np.allclose(skipped, climate)


def test_previous_window_is_unknown_on_the_first_morning():
    hits = np.zeros(20)
    hits[0] = 1
    climate = np.full(20, 0.25)
    fit = np.ones(20, dtype=bool)
    out = apply_markov(hits, climate, fit, horizon=5)
    assert np.allclose(out[:5], climate[:5])


def test_a_down_group_cancels_the_model():
    model = np.array([0.8, 0.8])
    climate = np.array([0.2, 0.2])
    out = apply_veto(model, {"slab quakes": np.array([0.0, -0.05])}, climate)
    assert out[0] > climate[0]
    assert np.isclose(out[1], climate[1])


def test_recent_levels_do_not_use_the_open_window():
    from eqpred.eval.new_strategies import apply_online_decile

    model = np.linspace(0.1, 0.9, 40)
    hits = np.zeros(40)
    hits[:20] = 1
    climate = np.full(40, 0.4)
    out = apply_online_decile(model, hits, climate, horizon=5, lookback=15, bins=2)
    assert np.allclose(out[:20], climate[:20])


def test_quiet_catalog_stays_on_the_usual_rate_without_that_group():
    _model, hits, climate, fit, val = _series()
    out = apply_quiet_catalog({}, hits, climate, fit, val)
    assert np.allclose(out, climate)
