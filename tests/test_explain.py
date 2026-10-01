import numpy as np

from eqpred.eval.explain import furthest_above_rate, neighborhood_from_ref, story_plain
from eqpred.eval.lineup import watch_agreement
from eqpred.eval.mainshock import _BlankAwareScaler, _SitOutScaler


def test_a_blank_column_has_no_flag_and_a_real_zero_still_differs():
    train = np.array([[1.0, 0.0], [3.0, 1.0], [5.0, 0.0], [np.nan, 2.0]])
    sit = _SitOutScaler().fit(train)
    blank_aware = _BlankAwareScaler().fit(train)
    row = np.array([[np.nan, 0.0]])
    assert sit.transform(row).shape[1] == 2
    assert blank_aware.transform(row).shape[1] == 4
    assert sit.transform(row)[0, 0] == 0.0
    zero = sit.transform(np.array([[0.0, np.nan]]))
    assert zero[0, 0] != 0.0
    assert zero[0, 1] == 0.0


def test_neighborhood_counts_hits_without_changing_the_rate():
    scores = np.linspace(0.0, 1.0, 100)
    hits = np.zeros(100)
    hits[:40] = 1.0
    report = neighborhood_from_ref(0.2, scores, hits, min_n=80)
    assert report["n"] == 80
    assert report["hits"] == 40
    assert "80 closest check-year mornings" in report["plain"]
    assert report["plain"]


def test_agreement_needs_two_thirds_and_at_least_three_readings():
    quiet = watch_agreement([("solar", 0.02), ("spf", None)])
    assert quiet["agreed"] is False
    assert "Only 1" in quiet["plain"]
    split = watch_agreement([("solar", 0.02), ("spf", -0.02), ("geometry", 0.02), ("cosmic", -0.02)])
    assert split["agreed"] is False
    down = watch_agreement([("solar", -0.02), ("spf", -0.02), ("geometry", 0.001), ("dust", 0.02)])
    assert down["agreed"] is False
    agreed = watch_agreement([("solar", -0.02), ("spf", -0.02), ("geometry", 0.001)])
    assert agreed["agreed"] is True
    assert agreed["direction"] == "down"
    assert agreed["plain"].startswith("2 of 3")


def test_the_story_follows_the_lift_not_the_larger_chance():
    rows = [
        {"region": "global", "label": "Earth-wide", "p": 0.8, "climatology_p": 0.75, "chance_plain": "80 in 100", "climatology_chance_plain": "75 in 100"},
        {"region": "hikurangi", "label": "Hikurangi", "p": 0.4, "climatology_p": 0.1, "chance_plain": "40 in 100", "climatology_chance_plain": "10 in 100"},
    ]
    lift = furthest_above_rate(rows)
    assert lift["region"] == "hikurangi"
    waiting = story_plain(lift=lift, odd={"region": "alpine", "label": "Alpine Fault"}, agreement={"plain": "No agreement.", "agreed": False}, shocks=[])
    assert "Hikurangi" in waiting
    assert "Alpine Fault" in waiting
    assert "waits for a first shock" in waiting
    happened = story_plain(
        lift=lift,
        odd={"region": "alpine", "label": "Alpine Fault"},
        agreement={"plain": "No agreement.", "agreed": False, "direction": None},
        shocks=[{"region": "hikurangi", "label": "Hikurangi"}],
    )
    assert "furthest above its usual rate" in happened
    assert "had not agreed" in happened
