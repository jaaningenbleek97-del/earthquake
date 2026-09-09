from datetime import datetime, timezone

import pandas as pd

from eqpred.models.budget import AlertBudget, DEFAULT_CAPS, month_key


def _day(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def test_month_key_utc():
    assert month_key(_day(2024, 8, 31)) == "2024-08"
    assert month_key("2024-09-01T00:00:00Z") == "2024-09"


def test_first_come_then_deny():
    budget = AlertBudget({"alarm.spf.v2": 2})
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 1, 1))
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 1, 2))
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 1, 3)) is False
    assert budget.used("alarm.spf.v2", "global", _day(2020, 1, 15)) == 2


def test_month_rollover_resets():
    budget = AlertBudget({"alarm.spf.v2": 1})
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 1, 31))
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 2, 1))
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 2, 2)) is False


def test_models_and_regions_do_not_share():
    budget = AlertBudget({"alarm.spf.v2": 1, "alarm.quiet.v1": 1})
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 3, 1))
    assert budget.spend("alarm.quiet.v1", "global", _day(2020, 3, 1))
    assert budget.spend("alarm.spf.v2", "hikurangi", _day(2020, 3, 1))
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 3, 2)) is False


def test_seed_from_cards_replays_spend_except_replaced_days():
    budget = AlertBudget({"alarm.spf.v2": 2, "anti.quiet.v1": 4})
    cards = [
        {
            "kind": "alarm",
            "model_id": "alarm.spf.v2",
            "issued_at": "2020-01-01T00:00:00Z",
            "region": {"name": "global"},
        },
        {
            "kind": "alarm",
            "model_id": "alarm.spf.v2",
            "issued_at": "2020-01-02T00:00:00Z",
            "region": {"name": "global"},
        },
        {
            "kind": "probability",
            "model_id": "fusion.etas_x_mod.v1",
            "issued_at": "2020-01-02T00:00:00Z",
            "region": {"name": "global"},
        },
    ]
    skip = {pd.Timestamp("2020-01-02", tz="UTC")}
    assert budget.seed_from_cards(cards, exclude_days=skip) == 1
    assert budget.used("alarm.spf.v2", "global", _day(2020, 1, 15)) == 1
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 1, 3))
    assert budget.spend("alarm.spf.v2", "global", _day(2020, 1, 4)) is False


def test_uncapped_model_always_allowed():
    budget = AlertBudget()
    assert budget.spend("alarm.unknown.v1", "global", _day(2020, 4, 1))
    assert budget.cap("alarm.unknown.v1") is None


def test_default_caps_are_locked_constants():
    assert DEFAULT_CAPS["alarm.spf.v2"] == 6
    assert DEFAULT_CAPS["alarm.quiet.v1"] == 2
    assert DEFAULT_CAPS["alarm.quiet_spf.v1"] == 2
    assert DEFAULT_CAPS["anti.quiet.v1"] == 4
