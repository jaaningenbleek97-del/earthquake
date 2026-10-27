import numpy as np
import pandas as pd

from eqpred.eval.idea_bench import (
    _hedge_series,
    b_positive,
    fit_hidden_state,
    kappa1,
    mapped_chances,
    natural_time_entropy,
    pick_variation,
    signals_amr_curvature,
    signals_bath_gap,
    signals_burstiness,
    signals_depth_shift,
    signals_etas_residual,
    signals_fractal_entropy,
    signals_inverse_omori,
    signals_m8_tip,
    signals_moment_balance,
    signals_network_graph,
    signals_nowcast_eps,
    signals_pattern_informatics,
    signals_reversal_entropy,
    signals_rtl,
    signals_spatial_migration,
    signals_stochastic_background,
    tide_potential,
)
from eqpred.models.etas import EtasParams


def test_b_positive_recovers_a_b_value_of_one():
    rng = np.random.default_rng(1)
    mags = 3.0 + np.round(rng.exponential(1.0 / np.log(10.0), 20000), 1)
    b = b_positive(mags, delta=0.2)
    assert b is not None
    assert 0.85 < b < 1.15


def test_equal_quakes_have_the_uniform_natural_time_variance():
    size = 400
    expected = (size**2 - 1) / (12 * size**2)
    assert abs(kappa1(np.full(size, 4.0)) - expected) < 1e-9


def test_the_tide_rises_and_falls_about_twice_a_day():
    hours = np.arange(0, 24 * 10) / 24.0
    unix_days = (pd.Timestamp("2020-03-01", tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1) + hours
    w = tide_potential(unix_days, np.full(hours.size, -41.3), np.full(hours.size, 174.8))
    turns = np.sum(np.diff(np.sign(np.diff(w))) != 0)
    assert 30 <= turns <= 46


def test_a_busy_stretch_is_found_by_the_hidden_state():
    rng = np.random.default_rng(2)
    base = np.full(3000, 0.5)
    counts = rng.poisson(base).astype(float)
    counts[1200:1300] = rng.poisson(5.0, 100)
    model = fit_hidden_state(counts, base, states=2, rounds=15)
    assert model["factors"][1] > 3 * model["factors"][0]


def test_later_years_do_not_choose_the_variation():
    n = 900
    signal = np.linspace(0.0, 1.0, n)
    hits = np.zeros(n)
    hits[:300][signal[:300] > 0.25] = 1
    hits[300:600][signal[300:600] > 0.55] = 1
    usual = np.full(n, 0.3)
    early = np.zeros(n, dtype=bool)
    develop = np.zeros(n, dtype=bool)
    early[:300] = True
    develop[300:600] = True
    first, tried = pick_variation({"rising": signal}, hits, usual, early, develop)
    hits[600:] = 1 - hits[600:]
    second, _ = pick_variation({"rising": signal}, hits, usual, early, develop)
    assert tried > 0
    assert first["mapping"] == second["mapping"]
    assert np.allclose(first["chance"][:600], second["chance"][:600])


def test_a_chance_only_moves_the_usual_rate():
    n = 400
    rng = np.random.default_rng(3)
    signal = rng.normal(size=n)
    hits = (rng.random(n) < 0.2).astype(float)
    usual = np.linspace(0.1, 0.4, n)
    early = np.zeros(n, dtype=bool)
    early[:200] = True
    for _name, chance in mapped_chances(signal, hits, usual, early):
        assert np.all(np.isfinite(chance))
        assert np.corrcoef(chance, usual)[0, 1] > 0.5


def test_nowcast_burstiness_rtl_m8_depth_and_migration_build_causal_signals():
    rng = np.random.default_rng(11)
    n = 2500
    t = np.sort(rng.uniform(0.0, n - 5.0, 500))
    known = np.clip(np.maximum.accumulate(np.floor(t).astype(np.int64) + 1), 0, n)
    mag = 3.0 + rng.exponential(0.8, 500)
    mag[50::60] = 6.2
    lat = -40.0 + rng.normal(0.0, 1.0, 500)
    lon = 176.0 + rng.normal(0.0, 1.0, 500)
    depth = rng.uniform(5.0, 80.0, 500)
    ev = {"t": t, "known": known, "mag": mag, "lat": lat, "lon": lon, "depth": depth}

    assert signals_nowcast_eps(ev, n, 3.0, 5.0, 1500)
    assert signals_burstiness(ev, n)
    assert signals_rtl(ev, n, 3.0)
    assert signals_m8_tip(ev, n, 3.0)
    assert signals_depth_shift(ev, n)
    assert signals_spatial_migration(ev, n)


def test_round_three_catalog_ideas_build_causal_signals():
    rng = np.random.default_rng(19)
    n = 2500
    days = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    t = np.sort(rng.uniform(0.0, n - 40.0, 600))
    known = np.clip(np.maximum.accumulate(np.floor(t).astype(np.int64) + 1), 0, n)
    mag = 3.0 + rng.exponential(0.75, 600)
    mag[40::55] = 6.1
    lat = -40.0 + rng.normal(0.0, 0.25, 600)
    lon = 176.0 + rng.normal(0.0, 0.25, 600)
    depth = rng.uniform(5.0, 80.0, 600)
    ev = {"t": t, "known": known, "mag": mag, "lat": lat, "lon": lon, "depth": depth}
    frame = pd.DataFrame(
        {
            "observed_at": days[0] + pd.to_timedelta(t, unit="D"),
            "available_at": days[0] + pd.to_timedelta(np.floor(t) + 1.0, unit="D"),
            "magnitude": mag,
            "lat": lat,
            "lon": lon,
            "depth_km": depth,
        }
    )

    assert signals_pattern_informatics(frame, "hikurangi", days[0], n, 3.0)
    assert signals_fractal_entropy(frame, "hikurangi", days[0], n, 3.0)
    assert signals_network_graph(frame, "hikurangi", days[0], n, 3.0)
    assert signals_amr_curvature(ev, n, 3.0)
    assert signals_etas_residual(ev, n, 3.0, 1500, EtasParams())
    assert signals_bath_gap(ev, n, 3.0, 5.0)
    assert signals_stochastic_background(ev, n, 3.0, 1500, EtasParams())
    assert signals_moment_balance(ev, n, 3.0)
    assert signals_inverse_omori(ev, n)
    assert signals_reversal_entropy(ev, n)


def test_natural_time_entropy_is_reversal_symmetric_for_equal_quakes():
    mags = np.full(80, 4.0)
    forward = natural_time_entropy(mags)
    reversed_entropy = natural_time_entropy(mags[::-1])
    assert abs(forward - reversed_entropy) < 1e-12


def test_hedge_series_uses_only_closed_windows():
    m = 80
    x = np.column_stack([np.full(m, -1.0), np.full(m, 1.0)])
    usual = np.full(m, 0.2)
    y = np.ones(m)
    closed = np.ones(m, dtype=bool)
    closed[40:] = False
    first = _hedge_series(x, y, usual, closed, eta=2.0, decay=1.0)
    y2 = y.copy()
    y2[40:] = 0.0
    second = _hedge_series(x, y2, usual, closed, eta=2.0, decay=1.0)
    assert np.allclose(first, second)
    assert first[39] > first[0]

