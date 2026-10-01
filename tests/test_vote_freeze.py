from datetime import datetime, timezone

import numpy as np
import pandas as pd

from eqpred.eval.vote_freeze import (
    append_entries,
    hold_vote_cards,
    read_ledger,
    read_votes,
    score_frozen_ledger,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observed_at": pd.to_datetime([], utc=True),
            "magnitude": pd.Series(dtype=float),
            "latitude": pd.Series(dtype=float),
            "longitude": pd.Series(dtype=float),
            "lat": pd.Series(dtype=float),
            "lon": pd.Series(dtype=float),
        }
    )


def _row(p: float = 0.2) -> dict:
    return {
        "region": "global",
        "label": "Earth-wide",
        "magnitude_min": 7.0,
        "horizon_days": 3,
        "p": p,
        "chance_plain": "20 in 100",
        "climatology_p": 0.1,
        "climatology_chance_plain": "10 in 100",
        "issue_day": "2020-01-05",
        "votes_up": [],
        "votes_down": [],
        "information_gain": None,
    }


def _fit(scale: float, p: float) -> dict:
    signal = {
        "family": "solar",
        "label": "solar",
        "columns": ["solar_x"],
        "mu": [0.0],
        "sd": [1.0],
        "scale": scale,
        "cap": 0.03,
    }
    return {
        "horizon_days": 3,
        "boxes": {
            "global": {
                "signals": [signal],
                "information_gain": None,
                "label": "Earth-wide",
                "magnitude_min": 7.0,
            }
        },
        "rows": [_row(p)],
    }


def test_a_later_vote_model_does_not_replace_the_frozen_card(tmp_path):
    days = pd.date_range("2020-01-01", periods=5, tz="UTC")
    design = (np.ones((5, 1)), ["solar_x"])
    published, draft = hold_vote_cards(
        _fit(0.02, 0.2),
        design=design,
        catalog=_catalog(),
        days=days,
        root=tmp_path,
    )
    assert published[0]["p"] == 0.2
    assert draft["same_as_frozen"] is True
    published_again, working = hold_vote_cards(
        _fit(0.05, 0.8),
        design=design,
        catalog=_catalog(),
        days=days,
        root=tmp_path,
    )
    frozen_global = next(row for row in published_again if row["region"] == "global")
    assert frozen_global["p"] == 0.2
    assert working["same_as_frozen"] is False
    assert working["week"][0]["p"] == 0.8
    assert working["writes_issued_p"] is False
    assert read_votes(tmp_path)["boxes"]["global"]["signals"][0]["scale"] == 0.02


def test_ledger_keeps_the_first_writing_of_a_morning(tmp_path):
    early = datetime(2026, 9, 26, tzinfo=timezone.utc)
    late = datetime(2026, 10, 1, tzinfo=timezone.utc)
    row = _row(0.2)
    row["horizon_days"] = 30
    row["issue_day"] = "2026-09-21"
    append_entries([row], root=tmp_path, written_at=early)
    changed = dict(row, p=0.9)
    append_entries([changed], root=tmp_path, written_at=late)
    entries = read_ledger(tmp_path)["entries"]
    assert len(entries) == 1
    assert entries[0]["p"] == 0.2
    assert entries[0]["prospective"] is True
    closed = dict(row, horizon_days=3)
    append_entries([closed], root=tmp_path, written_at=late)
    three = [entry for entry in read_ledger(tmp_path)["entries"] if entry["horizon_days"] == 3]
    assert three[0]["prospective"] is False


def test_live_score_ignores_a_card_written_after_its_window(tmp_path):
    append_entries(
        [
            {
                "region": "global",
                "label": "Earth-wide",
                "magnitude_min": 7.0,
                "horizon_days": 3,
                "p": 0.2,
                "climatology_p": 0.1,
                "issue_day": "2026-09-21",
            }
        ],
        root=tmp_path,
        written_at=datetime(2026, 9, 26, tzinfo=timezone.utc),
    )
    quakes = pd.DataFrame(
        {
            "observed_at": [pd.Timestamp("2026-11-01", tz="UTC")],
            "magnitude": [4.0],
            "latitude": [0.0],
            "longitude": [0.0],
            "lat": [0.0],
            "lon": [0.0],
        }
    )
    score = score_frozen_ledger(quakes, tmp_path)
    assert score["3"]["n_prospective"] == 0
    assert "next new morning" in score["3"]["plain"]
    assert score["30"]["n_prospective"] == 0
