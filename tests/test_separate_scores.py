import numpy as np

from eqpred.eval.separate_scores import split_windows


def test_a_copy_of_the_usual_rate_is_a_tie():
    usual = np.full(40, 0.2)
    happened = np.zeros(40)
    happened[::5] = 1
    scored = split_windows(usual, happened, usual)
    assert scored["verdict"] == "tie"
    assert scored["gap"] is None
    assert scored["windows"] == 40


def test_marked_up_windows_are_counted_on_their_own():
    usual = np.full(40, 0.2)
    happened = np.zeros(40)
    happened[:10] = 1
    chance = np.full(40, 0.2)
    chance[:10] = 0.8
    chance[10:20] = 0.05
    scored = split_windows(chance, happened, usual)
    assert scored["verdict"] == "up"
    assert scored["n_up"] == 10
    assert scored["up_hits"] == 10
    assert scored["n_down"] == 10
    assert scored["down_hits"] == 0
    assert scored["gap"] > 0


def test_one_sided_calls_are_not_a_win():
    usual = np.full(30, 0.3)
    happened = np.zeros(30)
    chance = np.full(30, 0.3)
    chance[:3] = 0.9
    scored = split_windows(chance, happened, usual)
    assert scored["verdict"] == "tie"
    assert scored["n_up"] == 3
