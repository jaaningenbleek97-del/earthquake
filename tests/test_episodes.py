import pandas as pd

from eqpred.eval.episodes import _cluster_dates, quiet_episodes


def test_cluster_dates_splits_on_gap():
    stamps = pd.to_datetime(
        ["2024-08-14T00:00:00Z", "2024-08-15T00:00:00Z", "2024-10-11T00:00:00Z"],
        utc=True,
    )
    groups = _cluster_dates(list(stamps))
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 1


def test_quiet_episode_marks_aftershock_echo():
    ledger = pd.DataFrame(
        {
            "prediction_id": ["a"],
            "model_id": ["alarm.quiet.v1"],
            "issued_at": ["2020-01-10T00:00:00Z"],
            "hit": [1],
            "competitor_etas_p": [0.06],
            "drivers": [
                [
                    {"driver": "etas_quiet", "lag_hours": 0, "contribution": 1.0},
                    {"driver": "proton_spike", "lag_hours": 0, "contribution": 1.0},
                ]
            ],
        }
    )
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                ["2020-01-08T00:00:00Z", "2020-01-12T00:00:00Z"], utc=True
            ),
            "magnitude": [7.4, 7.1],
            "lon": [140.0, 140.2],
            "lat": [38.0, 38.1],
            "place": ["main", "after"],
        }
    )
    episodes = quiet_episodes(ledger, quakes)
    assert len(episodes) == 1
    assert episodes[0]["verdict"] in {"hit_echo", "hit_after_prior_m7"}
    assert episodes[0]["window_m7"][0]["aftershock_like"] is True
