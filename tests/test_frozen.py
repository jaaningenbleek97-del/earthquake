import numpy as np

from eqpred.eval.frozen import MIN_FAMILY_IG, family_earns_card, promotion_status
from eqpred.eval.lineup import DAVIDSON_BURNS_FAMILIES, _scale, earned_ceiling, stack_votes


def test_three_day_votes_use_the_solar_watch_not_the_tectonic_families():
    assert "solar" in DAVIDSON_BURNS_FAMILIES
    assert "spf" in DAVIDSON_BURNS_FAMILIES
    assert "schumann" in DAVIDSON_BURNS_FAMILIES
    assert "geometry" in DAVIDSON_BURNS_FAMILIES
    assert "slowslip" not in DAVIDSON_BURNS_FAMILIES
    assert "infrasound" not in DAVIDSON_BURNS_FAMILIES


def test_blank_vote_adds_nothing_and_votes_are_capped():
    quiet = stack_votes([("solar", None), ("dust", 0.01)])
    assert quiet["total"] == 0.01
    assert quiet["n_blank"] == 1
    assert earned_ceiling("solar", 0.5, -0.2) == 0.03
    assert earned_ceiling("slowslip", 0.5, -0.2) == 0.01
    assert earned_ceiling("solar", 0.05, 0.02) == 0.05
    assert earned_ceiling("solar", 0.2, 0.02) == 0.06
    assert earned_ceiling("infrasound", 0.04, -0.01) == 0.01
    assert earned_ceiling("infrasound", 0.04, 0.01) == 0.04


def test_a_noisy_weight_shrinks_and_a_clear_one_keeps_its_size():
    n = 400
    z = np.linspace(-2.0, 2.0, n)
    roll = np.full(n, 0.1)
    clear_hits = np.clip(roll + 0.05 * z, 0.0, 1.0)
    clear = _scale(z, clear_hits, roll, np.ones(n, dtype=bool))
    rng = np.random.default_rng(0)
    noisy_hits = np.clip(roll + rng.normal(0.0, 0.4, n), 0.0, 1.0)
    noisy = _scale(z, noisy_hits, roll, np.ones(n, dtype=bool))
    assert abs(clear - 0.05) < 0.02
    assert abs(noisy) < abs(clear)
    assert family_earns_card(None) is False
    assert family_earns_card(-0.01) is False
    assert family_earns_card(0.0005) is False
    assert family_earns_card(MIN_FAMILY_IG) is True


def test_draft_waits_for_closed_mornings_after_the_lock():
    check = {"global": 0.02, "hikurangi": 0.02, "alpine": 0.02, "kermadec": 0.02}
    post = dict(check)
    waiting = promotion_status(check, post, n_post_lock=3)
    assert waiting["ready"] is False
    assert "closed mornings" in waiting["reason"]


def test_draft_replaces_the_frozen_card_only_after_both_tests():
    check = {"global": 0.02, "hikurangi": 0.02}
    post = {"global": 0.02, "hikurangi": -0.01}
    blocked = promotion_status(check, post, n_post_lock=40)
    assert blocked["ready"] is False
    ready = promotion_status(check, {"global": 0.03, "hikurangi": 0.02}, n_post_lock=40)
    assert ready["ready"] is True
