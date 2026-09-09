import pandas as pd

from eqpred.eval.nearmiss import (
    KM_GRACE,
    MAG_GRACE,
    TIME_GRACE_HOURS,
    combine_credits,
    dist_km_to_region,
    grade_card,
)
from eqpred.eval.score import summarize_ledger
from eqpred.regions import REGIONS


HIK = REGIONS["hikurangi"]
KERM = REGIONS["kermadec"]


def _quakes(**kwargs) -> pd.DataFrame:
    frame = pd.DataFrame(kwargs)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    return frame


def _card(*, mag=5.0, region=None, start="2019-01-01T00:00:00Z", end="2019-01-08T00:00:00Z") -> dict:
    return {
        "valid_from": start,
        "valid_to": end,
        "magnitude_min": mag,
        "region": region or HIK,
    }


def test_exact_hit_is_one_and_binary_hit():
    quakes = _quakes(
        observed_at=["2019-01-03T12:00:00Z"],
        magnitude=[5.4],
        lon=[176.0],
        lat=[-39.0],
        place=["inside"],
        event_id=["a"],
    )
    graded = grade_card(_card(), quakes)
    assert graded["hit"] == 1
    assert graded["n_events"] == 1
    assert graded["near_miss"] == 1.0
    assert graded["miss_km"] == 0.0
    assert graded["miss_hours"] == 0.0
    assert graded["miss_mag"] == 0.0


def test_magnitude_short_inside_box_is_partial():
    quakes = _quakes(
        observed_at=["2019-01-03T12:00:00Z"],
        magnitude=[5.9],
        lon=[176.0],
        lat=[-39.0],
        place=["short"],
        event_id=["b"],
    )
    graded = grade_card(_card(mag=6.0), quakes)
    assert graded["hit"] == 0
    assert graded["n_events"] == 0
    assert 0.4 < graded["near_miss"] < 1.0
    assert abs(graded["miss_mag"] - 0.1) < 1e-9
    assert graded["miss_km"] == 0.0
    assert graded["miss_hours"] == 0.0


def test_eight_hours_late_is_partial():
    quakes = _quakes(
        observed_at=["2019-01-08T08:00:00Z"],
        magnitude=[5.4],
        lon=[176.0],
        lat=[-39.0],
        place=["late"],
        event_id=["c"],
    )
    graded = grade_card(_card(), quakes)
    assert graded["hit"] == 0
    assert graded["miss_hours"] == 8.0
    assert graded["miss_km"] == 0.0
    assert 0.4 < graded["near_miss"] < 1.0


def test_just_outside_box_is_partial():
    quakes = _quakes(
        observed_at=["2019-01-03T12:00:00Z"],
        magnitude=[5.4],
        lon=[174.0],
        lat=[-39.5],
        place=["west"],
        event_id=["d"],
    )
    km = float(dist_km_to_region([-39.5], [174.0], HIK)[0])
    assert 0 < km < KM_GRACE
    graded = grade_card(_card(), quakes)
    assert graded["hit"] == 0
    assert graded["miss_km"] > 0
    assert graded["near_miss"] > 0


def test_far_event_is_zero():
    quakes = _quakes(
        observed_at=["2019-01-20T00:00:00Z"],
        magnitude=[4.0],
        lon=[10.0],
        lat=[45.0],
        place=["far"],
        event_id=["e"],
    )
    graded = grade_card(_card(), quakes)
    assert graded["hit"] == 0
    assert graded["near_miss"] == 0.0
    assert graded["miss_km"] is None


def test_kermadec_dateline_distance_is_small_just_west():
    km = float(dist_km_to_region([-30.0], [176.4], KERM)[0])
    assert 0 < km < KM_GRACE


def test_combine_credits_geometric_mean():
    assert combine_credits(1, 1, 1) == 1.0
    assert combine_credits(0, 1, 1) == 0.0
    assert abs(combine_credits(0.8, 0.8, 0.8) - 0.8) < 1e-12


def test_summarize_brier_stays_on_binary_hit():
    ledger = pd.DataFrame(
        {
            "model_id": ["fusion.etas_x_mod.v1"] * 3,
            "region": ["hikurangi", "hikurangi", "hikurangi"],
            "magnitude_min": [5.0, 5.0, 5.0],
            "kind": ["probability", "probability", "probability"],
            "issued_at": ["2019-01-01T00:00:00Z", "2019-01-02T00:00:00Z", "2019-01-03T00:00:00Z"],
            "value": [0.2, 0.5, 0.8],
            "competitor_etas_p": [0.1, 0.15, 0.2],
            "hit": [0, 0, 0],
            "near_miss": [0.6, 0.75, 0.9],
        }
    )
    summary = summarize_ledger(ledger)
    head = summary["heads"][0]
    assert head["hit_rate"] == 0.0
    assert abs(head["brier"] - ((0.2**2 + 0.5**2 + 0.8**2) / 3)) < 1e-12
    assert abs(head["mean_near_miss"] - 0.75) < 1e-12
    assert head["p_near_corr"] is not None
    assert head["p_hit_corr"] is None


def test_summarize_rank_corr_is_spearman():
    from scipy.stats import pearsonr, spearmanr

    value = [0.01, 0.02, 0.03, 0.04, 0.05, 0.90]
    hit = [0, 0, 0, 1, 1, 1]
    ledger = pd.DataFrame(
        {
            "model_id": ["fusion.etas_x_mod.v1"] * 6,
            "region": ["hikurangi"] * 6,
            "magnitude_min": [5.0] * 6,
            "kind": ["probability"] * 6,
            "issued_at": [f"2019-01-0{i}T00:00:00Z" for i in range(1, 7)],
            "value": value,
            "hit": hit,
        }
    )
    summary = summarize_ledger(ledger)
    expected = float(spearmanr(value, hit).correlation)
    pearson = float(pearsonr(value, hit).correlation)
    assert abs(summary["heads"][0]["p_hit_corr"] - expected) < 1e-12
    assert abs(expected - pearson) > 0.05


def test_open_windows_are_excluded_from_scoreboard():
    ledger = pd.DataFrame(
        {
            "model_id": ["fusion.etas_x_mod.v1"] * 3,
            "region": ["global"] * 3,
            "magnitude_min": [7.0] * 3,
            "kind": ["probability"] * 3,
            "issued_at": ["2019-01-01T00:00:00Z", "2019-01-02T00:00:00Z", "2019-01-03T00:00:00Z"],
            "value": [0.2, 0.3, 0.9],
            "hit": [0, 0, 1],
            "window_closed": [True, True, False],
        }
    )
    summary = summarize_ledger(ledger)
    assert summary["n_cards"] == 3
    assert summary["n_scored"] == 2
    assert summary["n_open_windows"] == 1
    head = summary["heads"][0]
    assert head["n"] == 2
    assert head["hit_rate"] == 0.0


def test_grace_constants_are_locked():
    assert KM_GRACE == 150.0
    assert TIME_GRACE_HOURS == 48.0
    assert MAG_GRACE == 0.5
