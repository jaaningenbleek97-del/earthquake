import numpy as np
import pandas as pd

from eqpred.eval.das_foil import run_das_foil
from eqpred.eval.mainshock import _mix, fresh_beats_climatology, week_state_of
from eqpred.eval.skill import skill_beats_climate
from eqpred.eval.widenet import run_widenet
from eqpred.features.sequence import traffic_sentence


def test_quiet_check_years_do_not_set_the_level():
    from eqpred.eval.mainshock import _lift_onto_rate

    similar = np.array([0.57])
    hits = np.array([0.0, 1.0, 0.0, 1.0])
    ref = np.array([True, True, False, False])
    roll = np.array([0.67])
    out = _lift_onto_rate(similar, hits, ref, roll, 0.67)
    assert abs(float(out[0]) - (0.67 + 0.57 - 0.5)) < 1e-9


def test_mix_picks_smallest_a_on_skill_not_brier():
    n = 80
    y = np.zeros(n)
    y[:16] = 1.0
    raw = np.linspace(0.05, 0.95, n)
    raw[:16] = 0.9
    climate = np.full(n, 0.2)
    mask = np.ones(n, dtype=bool)
    a, _brier = _mix(raw, y, 0.2, mask, roll=climate)
    assert a in {0.15, 0.3, 0.5, 0.75, 1.0}
    flat = np.full(n, 0.2)
    a0, _ = _mix(flat, y, 0.2, mask, roll=climate)
    assert a0 == 0.0


def test_copy_columns_drop_and_a_different_signal_stays():
    from eqpred.eval.mainshock import _drop_copy_columns

    n = 120
    alpine = np.linspace(0.0, 5.0, n)
    hikurangi = np.sin(np.linspace(0.0, 8.0, n))
    strain = np.random.default_rng(0).normal(size=n)
    stack = np.column_stack([alpine, alpine + 0.001, hikurangi, strain])
    names = ["grace_alpine", "grace_alpine_min7", "grace_hikurangi", "strain"]
    kept, kept_names = _drop_copy_columns(stack, names, np.ones(n, dtype=bool), min_corr=0.97)
    assert kept_names[0] == "grace_alpine"
    assert "grace_alpine_min7" not in kept_names
    assert "grace_hikurangi" in kept_names
    assert "strain" in kept_names
    assert kept.shape[1] == 3
    from eqpred.eval.mainshock import _score_to_rate

    score = np.linspace(0.0, 1.0, 500)
    hits = np.zeros(500)
    hits[-10:] = 1.0
    train = np.ones(500, dtype=bool)
    chance = _score_to_rate(score, hits, train, min_n=80)
    assert abs(float(chance[-1]) - float(chance[-2])) < 1e-9
    assert float(chance[-1]) < 0.05
    assert float(chance[0]) < 0.05


def test_missing_months_do_not_inflate_a_high_reading():
    from eqpred.eval.mainshock import _ObservedScaler

    column = np.full((1000, 1), np.nan)
    column[:100, 0] = np.linspace(-2.0, 6.6, 100)
    column[-1, 0] = 6.6
    scaled = _ObservedScaler().fit(column).transform(column[-1:])
    assert abs(float(scaled[0, 0])) < 4.0


def test_mix_report_does_not_pin_the_published_label():
    row = {
        "recipe": "fresh_shared|head|mix0.00",
        "lift": -0.01,
        "information_gain": -0.01,
        "n_trades": 20,
    }
    assert week_state_of(row, {"p": 0.4, "climatology_p": 0.2}) == "moved"


def test_worse_probability_score_is_not_called_a_beat():
    from eqpred.eval.mainshock import fresh_beats_climatology, week_state_of

    row = {
        "recipe": "fresh_shared|model|mix0.00",
        "lift": 0.04,
        "time_in_alarm": 0.2,
        "n_trades": 25,
        "information_gain": -0.14,
    }
    assert fresh_beats_climatology(row) is False
    assert week_state_of(row, {"p": 0.10, "climatology_p": 0.31}) == "moved"
    from eqpred.eval.mainshock import fresh_beats_climatology, week_state_of

    row = {
        "recipe": "fresh_shared|mix0.15",
        "lift": 0.41,
        "time_in_alarm": 0.02,
        "n_trades": 2,
        "information_gain": -0.004,
        "brier": 0.077,
        "climatology_brier": 0.076,
    }
    assert fresh_beats_climatology(row) is False
    assert week_state_of(row, {"p": 0.05971, "climatology_p": 0.05973}) == "rate"


def test_brier_win_without_skill_does_not_leave_climate():
    row = {
        "recipe": "fresh_thin|mix0.00",
        "brier": 0.10,
        "climatology_brier": 0.12,
        "lift": -0.02,
        "time_in_alarm": 0.2,
        "information_gain": -0.01,
    }
    assert fresh_beats_climatology(row) is False
    assert week_state_of(row, {"p": 0.2, "climatology_p": 0.2}) == "rate"


def test_three_week_states():
    beat = {"recipe": "fresh_shared|mix0.15", "lift": 0.05, "time_in_alarm": 0.2}
    moved = {"recipe": "fresh_shared|mix0.15", "lift": -0.01, "information_gain": -0.01}
    assert week_state_of(beat, {"p": 0.3, "climatology_p": 0.2}) == "beat"
    assert week_state_of(moved, {"p": 0.3, "climatology_p": 0.2}) == "moved"
    assert week_state_of(moved, {"p": 0.2, "climatology_p": 0.2}) == "rate"


def test_skill_ticket_accepts_information_gain():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    climate = np.full(10, 0.5)
    loud = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    ticket = skill_beats_climate(loud, y, climate, train_p=loud)
    assert ticket["beats"] is True
    assert ticket["ig_beats"] is True


def test_traffic_sentence_does_not_imply_a_new_p():
    red = traffic_sentence("hikurangi", "red", 6.4)
    assert red is not None and "does not change the published 30-day chance" in red
    assert traffic_sentence("hikurangi", "none", 6.4) is None
    assert traffic_sentence("global", "red", 7.1) is None
    yellow = traffic_sentence("alpine", "yellow", 6.1)
    assert yellow is not None and "yellow" in yellow


def test_widenet_and_das_do_not_write_issued_p(tmp_path):
    n = 40
    days = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    feat = pd.DataFrame({"issue_time": days, "proton_density_cm3": np.linspace(0, 1, n)})
    quakes = pd.DataFrame(
        {
            "observed_at": [days[10], days[20]],
            "magnitude": [5.4, 5.2],
            "lat": [36.0, -33.0],
            "lon": [141.0, -71.0],
            "event_id": ["j1", "c1"],
        }
    )
    wide = run_widenet(
        train_end="2018-01-20",
        holdout_start="2018-01-21",
        root=tmp_path,
        quakes=quakes,
        features=feat,
    )
    assert wide["writes_issued_p"] is False
    das = run_das_foil(tmp_path)
    assert das["writes_issued_p"] is False
    assert das["trains_leftover"] is False
    assert das["n_files"] == 0
