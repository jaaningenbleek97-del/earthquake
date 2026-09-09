from datetime import datetime, timezone

from eqpred.models.anti import (
    ANTI_DAYS_PER_MONTH,
    ANTI_MODEL,
    ANTI_QUANTILE,
    below_cutoff,
    invert_grade,
    is_anti_model,
)
import pandas as pd

from eqpred.eval.score import summarize_ledger
from eqpred.models.budget import DEFAULT_CAPS


def test_invert_grade_empty_window_is_quiet_success():
    graded = invert_grade({"hit": 0, "n_events": 0, "near_miss": 0.0, "miss_km": None})
    assert graded["hit"] == 1
    assert graded["near_miss"] == 1.0
    assert graded["n_events"] == 0


def test_invert_grade_event_in_box_breaks_quiet():
    graded = invert_grade({"hit": 1, "n_events": 1, "near_miss": 1.0, "miss_km": 0.0})
    assert graded["hit"] == 0
    assert graded["near_miss"] == 0.0


def test_invert_grade_close_call_stays_a_hit():
    graded = invert_grade({"hit": 0, "n_events": 0, "near_miss": 0.4, "miss_km": 20.0})
    assert graded["hit"] == 1
    assert abs(graded["near_miss"] - 0.6) < 1e-9


def test_below_cutoff_uses_locked_p_max():
    cfg = {"regions": {"alpine": {"p_max": 0.01}}}
    assert below_cutoff(cfg, "alpine", 0.009)
    assert below_cutoff(cfg, "alpine", 0.01)
    assert below_cutoff(cfg, "alpine", 0.011) is False
    assert below_cutoff({"regions": {}}, "alpine", 0.001) is False


def test_anti_budget_cap_is_locked():
    assert DEFAULT_CAPS[ANTI_MODEL] == ANTI_DAYS_PER_MONTH
    assert ANTI_QUANTILE == 0.15
    assert is_anti_model(ANTI_MODEL)
    assert is_anti_model("fusion.etas_x_mod.v1") is False


def test_summarize_anti_vs_always_quiet():
    ledger = pd.DataFrame(
        [
            {
                "model_id": "fusion.etas_x_mod.v1",
                "region": "alpine_fault",
                "magnitude_min": 5.0,
                "kind": "probability",
                "value": 0.02,
                "hit": 0,
                "competitor_etas_p": 0.03,
                "issued_at": "2019-01-01T00:00:00Z",
                "near_miss": 0.0,
            },
            {
                "model_id": "fusion.etas_x_mod.v1",
                "region": "alpine_fault",
                "magnitude_min": 5.0,
                "kind": "probability",
                "value": 0.03,
                "hit": 1,
                "competitor_etas_p": 0.03,
                "issued_at": "2019-01-02T00:00:00Z",
                "near_miss": 1.0,
            },
            {
                "model_id": "anti.quiet.v1",
                "region": "alpine_fault",
                "magnitude_min": 5.0,
                "kind": "alarm",
                "value": 1.0,
                "hit": 1,
                "competitor_etas_p": 0.02,
                "issued_at": "2019-01-01T00:00:00Z",
                "near_miss": 1.0,
            },
        ]
    )
    summary = summarize_ledger(ledger)
    anti = next(h for h in summary["heads"] if h["model_id"] == "anti.quiet.v1")
    assert anti["quiet_base_rate"] == 0.5
    assert abs(anti["quiet_lift"] - 0.5) < 1e-9


def test_anti_budget_is_per_region():
    from eqpred.models.budget import AlertBudget

    budget = AlertBudget({ANTI_MODEL: 1})
    day = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert budget.spend(ANTI_MODEL, "alpine", day)
    assert budget.spend(ANTI_MODEL, "hikurangi", day)
    assert budget.spend(ANTI_MODEL, "alpine", day) is False
