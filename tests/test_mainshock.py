import numpy as np
import pandas as pd
import pytest

from eqpred.eval.mainshock import (
    fresh_hits,
    is_last_week_column,
    leftover_week_copy,
    leftover_week_is_held,
    run_mainshock,
)
from eqpred.eval.strategies import _window_hits
from eqpred.regions import REGIONS


def test_fresh_hits_drop_aftershocks():
    days = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                ["2020-01-10T00:00:00Z", "2020-01-10T06:00:00Z", "2020-01-11T00:00:00Z"],
                utc=True,
            ),
            "magnitude": [7.4, 6.2, 5.8],
            "lon": [174.0, 174.1, 174.2],
            "lat": [-41.0, -41.1, -41.05],
        }
    )
    raw = _window_hits(quakes, REGIONS["global"], 5.0, pd.DatetimeIndex(days), 7)
    fresh = fresh_hits(quakes, REGIONS["global"], 5.0, pd.DatetimeIndex(days), 7)
    assert int(raw.sum()) > int(fresh.sum())
    assert int(fresh.sum()) >= 1


def test_mainshock_fits_leftover_not_etas(tmp_path):
    rng = np.random.default_rng(7)
    days = pd.date_range("2016-01-01", periods=2000, freq="D", tz="UTC")
    signal = (rng.random(len(days)) > 0.82).astype(float)
    features = pd.DataFrame(
        {
            "issue_time": days,
            "proton_density_cm3": signal * 20 + rng.normal(6, 1, len(days)),
            "dst_nt": rng.normal(-10, 8, len(days)),
            "tidal_strain": rng.normal(1.6, 0.2, len(days)),
            "n_m6_7d": rng.integers(0, 8, len(days)),
        }
    )
    quake_days = days[signal > 0.5][::3]
    quakes = pd.DataFrame(
        {
            "observed_at": quake_days + pd.Timedelta(days=2),
            "magnitude": np.full(len(quake_days), 7.2),
            "lon": np.full(len(quake_days), 180.0),
            "lat": np.full(len(quake_days), 0.0),
        }
    )
    (tmp_path / "data" / "features").mkdir(parents=True)
    (tmp_path / "data" / "eval").mkdir(parents=True)
    features.to_parquet(tmp_path / "data" / "features" / "global_daily.parquet")
    report = run_mainshock(root=tmp_path, features=features, quakes=quakes)
    assert report["writes_issued_p"] is False
    assert report["n_features"] > 0
    assert report["last_week"]["writes_issued_p"] is False
    assert report["last_week"]["week"]
    assert any("last 7 days" in (w.get("sentence") or "") for w in report["last_week"]["week"])
    global_row = next(r for r in report["regions"] if r["region"] == "global")
    assert global_row["this_week"]["p"] > 0
    assert "n_m6_7d" not in {c["driver"] for c in global_row["connections"]}
    assert all(row["region"] != "japan" for row in report["week"])


def test_mainshock_skips_empty_train_columns(tmp_path):
    rng = np.random.default_rng(3)
    days = pd.date_range("2016-01-01", periods=2000, freq="D", tz="UTC")
    signal = (rng.random(len(days)) > 0.8).astype(float)
    late = np.full(len(days), np.nan)
    late[1600:] = rng.normal(0, 1, late[1600:].size)
    features = pd.DataFrame(
        {
            "issue_time": days,
            "proton_density_cm3": signal * 18 + rng.normal(6, 1, len(days)),
            "dst_nt": rng.normal(-10, 8, len(days)),
            "tidal_strain": rng.normal(1.6, 0.2, len(days)),
            "late_gnss": late,
            "constant_one": np.ones(len(days)),
        }
    )
    quake_days = days[signal > 0.5][::3]
    quakes = pd.DataFrame(
        {
            "observed_at": quake_days + pd.Timedelta(days=2),
            "magnitude": np.full(len(quake_days), 7.2),
            "lon": np.full(len(quake_days), 180.0),
            "lat": np.full(len(quake_days), 0.0),
        }
    )
    (tmp_path / "data" / "features").mkdir(parents=True)
    (tmp_path / "data" / "eval").mkdir(parents=True)
    report = run_mainshock(root=tmp_path, features=features, quakes=quakes)
    assert report["n_regions"] >= 1
    global_row = next(r for r in report["regions"] if r["region"] == "global")
    assert global_row["this_week"]["p"] > 0
    assert "late_gnss" not in {c["driver"] for c in global_row["connections"]}


def test_hunt_recipe_does_not_count_as_a_beat():
    from eqpred.eval.mainshock import leftover_beats_climatology, leftover_this_week_held

    leaked = {
        "recipe": "leftover_hunt_logit|mix0.75",
        "brier": 0.018,
        "climatology_brier": 0.019,
        "lift": -0.02,
    }
    assert leftover_beats_climatology(leaked) is False
    assert leftover_this_week_held(leaked) is True


def test_tiny_brier_win_stays_held():
    from eqpred.eval.mainshock import leftover_beats_climatology, leftover_this_week_held

    sliver = {
        "recipe": "leftover_thin|mix1.00",
        "brier": 0.01235,
        "climatology_brier": 0.01238,
        "lift": 0.02,
    }
    assert leftover_beats_climatology(sliver) is True
    assert leftover_this_week_held(sliver) is True


def test_material_win_can_unhold():
    from eqpred.eval.mainshock import leftover_beats_climatology, leftover_this_week_held

    win = {
        "recipe": "leftover_joint|mix0.50",
        "brier": 0.10,
        "climatology_brier": 0.12,
        "lift": 0.05,
    }
    assert leftover_beats_climatology(win) is True
    assert leftover_this_week_held(win) is False


def test_leftover_week_copy_says_historical_rate_when_held():
    week = {
        "week": [
            {"region": "global", "held_to_climatology": True},
            {"region": "hikurangi", "held_to_climatology": True},
            {"region": "alpine", "held_to_climatology": True},
            {"region": "kermadec", "held_to_climatology": True},
        ]
    }
    held = leftover_week_copy(week)
    assert leftover_week_is_held(week) is True
    assert held["held"] is True
    assert "leftover attempt" in held["title"]
    assert "historical rate" in held["lede"]
    assert "attempt" in held["lede"]
    live = leftover_week_copy(
        {
            "week": [
                {"region": "global", "held_to_climatology": False},
                {"region": "hikurangi", "held_to_climatology": False},
                {"region": "alpine", "held_to_climatology": False},
                {"region": "kermadec", "held_to_climatology": False},
            ]
        }
    )
    assert live["held"] is False
    assert "leftover call" in live["title"]


def test_last_week_column_filter_drops_long_memory():
    assert is_last_week_column("proton_density_cm3") is True
    assert is_last_week_column("proton_density_cm3_max7") is True
    assert is_last_week_column("proton_density_cm3_d7") is True
    assert is_last_week_column("proton_density_cm3_z90") is False
    assert is_last_week_column("proton_density_cm3_lag27") is False
    assert is_last_week_column("cal_doy_sin") is False
    assert is_last_week_column("x_imf_bz_nt_tec_tecu") is False
    last = leftover_week_copy(
        {
            "week": [
                {"region": "global", "held_to_climatology": True},
                {"region": "hikurangi", "held_to_climatology": True},
                {"region": "alpine", "held_to_climatology": True},
                {"region": "kermadec", "held_to_climatology": True},
            ]
        },
        mode="last_week",
    )
    assert last["mode"] == "last_week"
    assert "last-week leftover attempt" in last["title"]
    assert "last 7 days" in last["lede"]


def test_seal_holds_leaked_alpine_and_m8_sliver():
    from eqpred.eval.mainshock import _seal_report

    report = {
        "n_features": 3912,
        "regions": [
            {
                "region": "alpine",
                "label": "Alpine Fault",
                "magnitude_min": 5.0,
                "recipe": "leftover_hunt_logit|mix0.75",
                "brier": 0.01866,
                "climatology_brier": 0.01880,
                "lift": -0.0186,
                "n_hits": 0,
                "n_trades": 12,
                "fresh_rate_train": 0.0423,
                "this_week": {
                    "issue_day": "2026-09-05",
                    "p": 0.01,
                    "leftover_raw_p": 0.01,
                    "held_to_climatology": False,
                    "chance_plain": "1 in 100",
                    "climatology_p": 0.0423,
                    "climatology_chance_plain": "4 in 100",
                },
            }
        ],
        "week": [{"region": "alpine", "horizon_days": 7, "why": ["gps"]}],
        "experiments": [
            {
                "id": "fresh_7d_m8",
                "horizon_days": 7,
                "regions": [
                    {
                        "region": "global",
                        "label": "Earth-wide",
                        "magnitude_min": 8.0,
                        "recipe": "leftover_thin|mix1.00",
                        "brier": 0.01235,
                        "climatology_brier": 0.01238,
                        "lift": 0.021,
                        "n_hits": 13,
                        "n_trades": 385,
                        "fresh_rate_train": 0.0181,
                        "this_week": {
                            "issue_day": "2026-09-05",
                            "p": 0.002,
                            "leftover_raw_p": 0.002,
                            "held_to_climatology": False,
                            "chance_plain": "less than 1 in 100",
                            "climatology_p": 0.0181,
                            "climatology_chance_plain": "2 in 100",
                        },
                    }
                ],
                "week": [{"region": "global", "horizon_days": 7}],
            }
        ],
    }
    sealed = _seal_report(report)
    alpine = sealed["regions"][0]
    assert alpine["beats_climatology"] is False
    assert alpine["recipe_leaked"] is True
    assert alpine["this_week"]["held_to_climatology"] is True
    assert alpine["this_week"]["p"] == pytest.approx(0.0423)
    assert alpine["this_week"]["leftover_raw_p"] == pytest.approx(0.01)
    assert alpine["this_week"]["leftover_raw_chance_plain"]
    assert "attempt" in sealed["week"][0]["sentence"]
    assert sealed["week"][0]["why"] == ["gps"]
    assert sealed["n_beats_climatology"] == 0
    m8 = sealed["experiments"][0]["regions"][0]
    assert m8["beats_climatology"] is True
    assert m8["this_week"]["held_to_climatology"] is True
    assert m8["this_week"]["p"] == pytest.approx(0.0181)
    assert "does not count" in sealed["note"]
    assert "M8" in sealed["note"]
