from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.aftershock import load_aftershock, run_aftershock
from eqpred.eval.horizon import HORIZON_DAYS


def _frames(n=1400):
    start = datetime(2016, 1, 1, tzinfo=timezone.utc)
    days = pd.date_range(start, periods=n, freq="D", tz="UTC")
    features = pd.DataFrame({"issue_time": days, "proton_density_cm3": np.full(n, 6.0)})
    rows = []
    for i in range(30, n - 20, 40):
        when = days[i]
        rows.append(
            {
                "observed_at": when,
                "available_at": when + timedelta(hours=6),
                "magnitude": 7.2,
                "lat": 0.0,
                "lon": 180.0,
                "event_id": f"ev{i}",
            }
        )
    return features, pd.DataFrame(rows)


def test_aftershock_scores_itself_and_never_issues_p(tmp_path):
    features, quakes = _frames()
    (tmp_path / "data" / "features").mkdir(parents=True)
    features.to_parquet(tmp_path / "data" / "features" / "global_daily.parquet")

    report = run_aftershock(root=tmp_path, features=features, quakes=quakes)

    assert report["writes_issued_p"] is False
    assert report["issue_day"]
    assert (tmp_path / "data" / "eval" / "aftershock.json").exists()
    assert not (tmp_path / "data" / "models").exists() or not any(
        (tmp_path / "data" / "models").glob("*.json")
    )

    week = {row["region"]: row for row in report["week"]}
    assert "global" in week
    assert week["global"]["horizon_days"] == HORIZON_DAYS
    # Aftershocks count here, so the sentence must not claim a fresh-shock call.
    assert "not the issued call" in week["global"]["sentence"]

    scored = {row["region"]: row for row in report["scored"]}
    assert scored["global"]["n_score"] > 0
    assert scored["global"]["etas_brier"] is not None

    again = load_aftershock(tmp_path)
    assert again["issue_day"] == report["issue_day"]


def test_aftershock_target_counts_aftershocks(tmp_path):
    """The scored target is any qualifying quake, not the declustered first shock."""
    features, quakes = _frames()
    swarm = quakes.copy()
    swarm["observed_at"] = swarm["observed_at"] + pd.Timedelta(days=1)
    swarm["available_at"] = swarm["available_at"] + pd.Timedelta(days=1)
    swarm["magnitude"] = 7.0
    swarm["event_id"] = swarm["event_id"] + "b"
    both = pd.concat([quakes, swarm], ignore_index=True)

    (tmp_path / "data" / "features").mkdir(parents=True)
    features.to_parquet(tmp_path / "data" / "features" / "global_daily.parquet")
    report = run_aftershock(root=tmp_path, features=features, quakes=both)

    scored = {row["region"]: row for row in report["scored"]}
    assert scored["global"]["hit_rate_holdout"] > 0
