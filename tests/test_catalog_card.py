import numpy as np
import pandas as pd

from eqpred.eval.catalog_card import covering_magnitude, fresh_daily_rate, grid_cells, window_expected
from eqpred.eval.saved_strategies import shuffle_check
from eqpred.regions import REGIONS


def test_a_bigger_recent_quake_removes_smaller_first_shocks_only():
    # Day 0 is an M7.6. Its window is much longer than 10 days.
    cover = covering_magnitude(np.array([0]), np.array([7.6]), 40)
    none = np.full(40, -np.inf)
    rate = np.ones(40)
    during = fresh_daily_rate(rate, cover, target=6.0, b=1.0)
    quiet = fresh_daily_rate(rate, none, target=6.0, b=1.0)
    bigger = fresh_daily_rate(rate, cover, target=8.0, b=1.0)
    bigger_quiet = fresh_daily_rate(rate, none, target=8.0, b=1.0)
    assert cover[5] >= 7.6
    assert during[5] < quiet[5] * 0.1
    assert bigger[5] == bigger_quiet[5]


def test_the_window_sum_looks_forward():
    daily = np.array([1.0, 10.0, 0.0, 0.0])
    out = window_expected(daily, 2)
    assert out[0] == 11.0
    assert np.isnan(out[-1])


def test_each_issued_box_splits_into_patches():
    assert len(grid_cells(REGIONS["hikurangi"])) == 4
    assert len(grid_cells(REGIONS["kermadec"])) == 8


def test_a_matched_card_beats_most_time_shifts():
    hits = np.zeros(400)
    hits[::10] = 1.0
    pred = np.where(hits > 0, 0.8, 0.05)
    climate = np.full(400, 0.1)
    mask = np.zeros(400, dtype=bool)
    mask[30:370] = True
    result = shuffle_check(pred, hits, climate, mask, horizon=30, draws=40)
    assert result is not None
    assert result["beats"] >= 30
