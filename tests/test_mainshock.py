import numpy as np
import pandas as pd
import pytest

from eqpred.eval.mainshock import (
    LOCKED_THIN,
    SHRINK_CAP,
    _alpine_sequence_frame,
    _maybe_admit_alpine,
    _pick_model,
    _region_mainshock,
    _shared_family_indices,
    alpine_admission_ok,
    issued_fresh_daily,
    fresh_hits,
    fresh_week_copy,
    fresh_week_is_held,
    hunt_leaked_name,
    is_last_week_column,
    is_locked_thin_column,
    leftover_promote_ok,
    locked_thin_columns,
    rolling_climate_series,
    run_locked_thin_mainshock,
    run_mainshock,
    sealed_holdout_mask,
    shrink_columns,
)
from eqpred.eval.strategies import _window_hits
from eqpred.regions import REGIONS


def test_fresh_hits_keep_an_earlier_shock_when_a_larger_one_follows():
    days = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                ["2020-01-10T00:00:00Z", "2020-01-12T00:00:00Z"],
                utc=True,
            ),
            "magnitude": [6.2, 7.4],
            "lon": [174.0, 174.1],
            "lat": [-41.0, -41.1],
        }
    )
    fresh = fresh_hits(quakes, REGIONS["global"], 5.0, pd.DatetimeIndex(days), 7)
    jan4 = days.get_loc(pd.Timestamp("2020-01-04", tz="UTC"))
    assert bool(fresh[jan4])


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


def test_mainshock_fits_fresh_not_etas(tmp_path):
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
    assert report["horizon_days"] == 30
    assert {exp["id"] for exp in report.get("experiments") or []} >= {
        "fresh_7d",
        "fresh_30d_m8",
        "fresh_30d_quiet",
    }
    assert report.get("csep")
    assert report["n_features"] > 0
    assert report["last_week"]["writes_issued_p"] is False
    assert report["last_week"]["week"]
    assert any("last 7 days" in (w.get("sentence") or "") for w in report["last_week"]["week"])
    global_row = next(r for r in report["regions"] if r["region"] == "global")
    assert global_row["this_week"]["p"] > 0
    assert "independent_p" in global_row["this_week"]
    assert global_row.get("csep")
    assert "n_m6_7d" not in {c["driver"] for c in global_row["connections"]}
    assert all(row["region"] != "japan" for row in report["week"])
    assert report["shrunk"]["writes_issued_p"] is False
    assert report["shrunk"]["cap"] == SHRINK_CAP
    assert report["alpine_sequence"]["writes_issued_p"] is False
    assert report["alpine_sequence"]["applied"] is False
    assert any("rolling five-year" in (w.get("sentence") or "") for w in report["week"])
    assert report.get("score_kind") == "daily_circular_block_bootstrap"
    assert report.get("method_eval")
    assert report["method_eval"]["writes_issued_p"] is False
    assert all(row["region"] != "chile" for row in report["week"])
    assert report.get("kermadec_attribution")
    assert report["kermadec_attribution"]["writes_issued_p"] is False
    assert global_row.get("score_kind") == "daily_circular_block_bootstrap"
    assert report.get("evaluation", {}).get("protocol")
    assert report["evaluation"]["score_is"] == "retrospective_holdout"
    assert report["evaluation"]["target_decluster"] == "causal_aftershock"
    assert "brier_p_value" in global_row
    assert global_row["n_score"] == global_row["n_score_daily"]


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


def test_pooled_must_beat_climate_on_the_box():
    from eqpred.eval.mainshock import _prefer_pooled

    assert _prefer_pooled(0.10, 0.12, 0.11) is True
    assert _prefer_pooled(0.12, 0.13, 0.10) is False
    assert _prefer_pooled(0.10, 0.09, 0.11) is False
    assert _prefer_pooled(None, 0.12, 0.11) is False
    assert _prefer_pooled(0.10, 0.12, 0.11, pool_rate=0.53, climate=0.16) is False
    assert _prefer_pooled(0.10, 0.12, 0.11, pool_rate=0.18, climate=0.16) is True


def test_usable_and_top_columns_skip_dead_series():
    from eqpred.eval.mainshock import _top_columns, _usable_indices

    rng = np.random.default_rng(1)
    n = 200
    y = (rng.random(n) > 0.7).astype(float)
    x = np.column_stack(
        [
            y + rng.normal(0, 0.05, n),
            np.ones(n),
            np.full(n, np.nan),
            rng.normal(size=n),
        ]
    )
    mask = np.ones(n, dtype=bool)
    usable = _usable_indices(x, mask, min_n=20)
    assert usable == [0, 3]
    assert _top_columns(x, y, ["signal", "const", "empty", "noise"], k=1) == [0]


def test_hunt_recipe_does_not_count_as_a_beat():
    from eqpred.eval.mainshock import fresh_beats_climatology, fresh_this_week_held

    leaked = {
        "recipe": "fresh_hunt_logit|mix0.75",
        "brier": 0.018,
        "climatology_brier": 0.019,
        "lift": -0.02,
    }
    assert fresh_beats_climatology(leaked) is False
    assert fresh_this_week_held(leaked) is True


def test_skill_pass_unholds_even_when_brier_gap_is_tiny():
    from eqpred.eval.mainshock import fresh_beats_climatology, fresh_this_week_held

    sliver = {
        "recipe": "fresh_thin|mix0.15",
        "brier": 0.01235,
        "climatology_brier": 0.01238,
        "lift": 0.02,
        "time_in_alarm": 0.2,
    }
    assert fresh_beats_climatology(sliver) is True
    assert fresh_this_week_held(sliver) is False


def test_material_win_can_unhold():
    from eqpred.eval.mainshock import fresh_beats_climatology, fresh_this_week_held

    win = {
        "recipe": "fresh_joint|mix0.50",
        "brier": 0.10,
        "climatology_brier": 0.12,
        "lift": 0.05,
    }
    assert fresh_beats_climatology(win) is True
    assert fresh_this_week_held(win) is False


def test_fresh_week_copy_says_historical_rate_when_held():
    week = {
        "week": [
            {"region": "global", "held_to_climatology": True},
            {"region": "hikurangi", "held_to_climatology": True},
            {"region": "alpine", "held_to_climatology": True},
            {"region": "kermadec", "held_to_climatology": True},
        ]
    }
    held = fresh_week_copy(week)
    assert fresh_week_is_held(week) is True
    assert held["held"] is True
    assert "forecast" in held["title"]
    assert "typical year" in held["lede"]
    assert "beside" in held["lede"]
    live = fresh_week_copy(
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
    assert "forecast" in live["title"]


def test_last_week_column_filter_drops_long_memory():
    assert is_last_week_column("proton_density_cm3") is True
    assert is_last_week_column("proton_density_cm3_max7") is True
    assert is_last_week_column("proton_density_cm3_d7") is True
    assert is_last_week_column("proton_density_cm3_z90") is False
    assert is_last_week_column("proton_density_cm3_lag27") is False
    assert is_last_week_column("cal_doy_sin") is False
    assert is_last_week_column("x_imf_bz_nt_tec_tecu") is False
    last = fresh_week_copy(
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
    assert "last-7-day" in last["title"]
    assert "last 7 days" in last["lede"].lower()


def test_locked_thin_drops_analog_and_keeps_sources():
    assert is_locked_thin_column("dst_nt") is True
    assert is_locked_thin_column("tidal_strain") is True
    assert is_locked_thin_column("kp") is True
    assert is_locked_thin_column("analog_p_global") is False
    assert is_locked_thin_column("analog_p_kermadec") is False
    assert is_locked_thin_column("earth_state_z") is False
    assert is_locked_thin_column("graph_inflow_alpine") is False
    assert is_locked_thin_column("proton_density_cm3") is False
    cols = locked_thin_columns(
        ["dst_nt", "analog_p_global", "kp", "earth_state_z", "n_m6_7d", "tidal_strain"]
    )
    assert cols == ["dst_nt", "kp", "tidal_strain"]
    assert "analog_p_global" not in LOCKED_THIN
    assert "dst_nt" in LOCKED_THIN


def test_locked_thin_paper_test_does_not_write_issued_p(tmp_path):
    rng = np.random.default_rng(3)
    days = pd.date_range("2016-01-01", periods=1600, freq="D", tz="UTC")
    signal = (rng.random(len(days)) > 0.8).astype(float)
    features = pd.DataFrame(
        {
            "issue_time": days,
            "dst_nt": signal * -40 + rng.normal(-10, 8, len(days)),
            "kp": signal * 3 + rng.normal(2, 1, len(days)),
            "tidal_strain": rng.normal(1.6, 0.2, len(days)),
            "f107": rng.normal(120, 15, len(days)),
            "analog_p_global": signal,
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
    report = run_locked_thin_mainshock(root=tmp_path, features=features, quakes=quakes)
    locked = report["locked_thin"]
    assert report["writes_issued_p"] is False
    assert locked["writes_issued_p"] is False
    assert locked["n_features"] > 0
    assert "analog_p_global" not in locked["locked"]
    assert any("locked thin" in (w.get("sentence") or "") for w in locked["week"])
    global_row = next(r for r in locked["regions"] if r["region"] == "global")
    assert "analog_p_global" not in {c["driver"] for c in global_row.get("connections") or []}


def test_seal_holds_leaked_alpine_and_m8_sliver():
    from eqpred.eval.mainshock import _seal_report

    report = {
        "n_features": 3912,
        "regions": [
            {
                "region": "alpine",
                "label": "Alpine Fault",
                "magnitude_min": 5.0,
                "recipe": "fresh_hunt_logit|mix0.75",
                "brier": 0.01866,
                "climatology_brier": 0.01880,
                "lift": -0.0186,
                "n_hits": 0,
                "n_trades": 12,
                "fresh_rate_train": 0.0423,
                "this_week": {
                    "issue_day": "2026-09-05",
                    "p": 0.01,
                    "fresh_raw_p": 0.01,
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
                        "recipe": "fresh_thin|mix1.00",
                        "brier": 0.01235,
                        "climatology_brier": 0.01238,
                        "lift": 0.021,
                        "n_hits": 13,
                        "n_trades": 385,
                        "fresh_rate_train": 0.0181,
                        "this_week": {
                            "issue_day": "2026-09-05",
                            "p": 0.002,
                            "fresh_raw_p": 0.002,
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
    assert alpine["this_week"]["p"] == pytest.approx(0.01)
    assert alpine["this_week"]["fresh_raw_p"] == pytest.approx(0.01)
    assert alpine["this_week"]["fresh_raw_chance_plain"]
    assert "fresh-shock" in sealed["week"][0]["sentence"]
    assert "1 in 100" in sealed["week"][0]["sentence"]
    assert "4 in 100" in sealed["week"][0]["sentence"]
    assert sealed["week"][0]["p"] == pytest.approx(0.01)
    assert sealed["week"][0]["why"] == ["gps"]
    assert sealed["n_beats_climatology"] == 0
    m8 = sealed["experiments"][0]["regions"][0]
    assert m8["beats_climatology"] is True
    assert m8["this_week"]["held_to_climatology"] is False
    assert m8["this_week"]["week_state"] == "beat"
    assert m8["this_week"]["p"] == pytest.approx(0.002)
    assert m8["this_week"]["fresh_raw_p"] == pytest.approx(0.002)
    assert "does not count" in sealed["note"]
    assert "M8" in sealed["note"]


def test_issue_predictions_writes_fresh_not_etas(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from eqpred.eval.mainshock import FRESH_MODEL_ID
    from eqpred.models.predict import issue_predictions

    days = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    features = pd.DataFrame(
        {
            "issue_time": days,
            "spf_significant_window": [0.0, 1.0, 0.0],
            "proton_density_cm3": [5.0, 6.0, 7.0],
        }
    )
    (tmp_path / "data" / "features").mkdir(parents=True)
    features.to_parquet(tmp_path / "data" / "features" / "global_daily.parquet")
    p = pd.Series([0.11, 0.22, 0.33], index=days)
    daily = {
        rid: {
            "p": p,
            "climatology_p": 0.04,
            "connections": [{"driver": "proton_density_cm3", "lag_days": 1, "importance": 1.0}],
        }
        for rid in ("global", "hikurangi", "alpine", "kermadec")
    }

    def _begin(rates, window, root, dest):
        dest.mkdir(parents=True, exist_ok=True)
        return window

    monkeypatch.setattr("eqpred.models.predict.load_comcat", lambda root=None: pd.DataFrame())
    monkeypatch.setattr("eqpred.models.predict.begin_issue", _begin)
    monkeypatch.setattr(
        "eqpred.models.predict.ensure_registered",
        lambda root=None: {"id": "reg", "code_hash": "abc", "brain_hash": "def"},
    )
    monkeypatch.setattr("eqpred.models.predict.write_batch_meta", lambda *a, **k: None)
    monkeypatch.setattr("eqpred.models.predict.stamp_cards", lambda cards, registry: cards)
    monkeypatch.setattr(
        "eqpred.eval.mainshock.issued_fresh_daily",
        lambda **kw: (features, daily),
    )

    cards = issue_predictions(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 3, tzinfo=timezone.utc),
        root=tmp_path,
    )
    ids = {c["model_id"] for c in cards}
    assert FRESH_MODEL_ID in ids
    assert all(c.get("written_at") for c in cards)
    assert all(c.get("card_kind") == "retrospective" for c in cards)
    assert "fusion.etas_x_mod.v1" not in ids
    assert "anti.quiet.v1" not in ids
    assert "alarm.quiet.v1" not in ids
    assert "alarm.quiet_spf.v1" not in ids
    probs = [c for c in cards if c["kind"] == "probability"]
    assert probs
    assert all(c["model_id"] == FRESH_MODEL_ID for c in probs)
    assert all("competitor_climatology_p" in c for c in probs)
    assert all("competitor_etas_p" not in c for c in probs)
    assert any(c["model_id"] == "alarm.spf.v2" for c in cards)


def test_issued_columns_keep_every_source_but_drop_old_leftovers():
    from eqpred.eval.mainshock import issued_columns, is_issued_driver

    assert is_issued_driver("proton_density_cm3") is True
    assert is_issued_driver("tidal_strain") is True
    assert is_issued_driver("tec_tecu") is True
    assert is_issued_driver("tec_tecu__lag7") is True
    assert is_issued_driver("so2_kermadec_kgm2") is True
    assert is_issued_driver("n_fresh_m45_7d") is True
    assert is_issued_driver("rate_roll_5y_kermadec") is True
    assert is_issued_driver("rate_roll_1y_global") is True
    assert is_issued_driver("tec_tecu__lag180") is False
    assert is_issued_driver("tec_tecu_anom90") is False
    assert is_issued_driver("x_tec_tecu_f107") is True
    assert issued_columns(
        ["proton_density_cm3", "tec_tecu", "tec_tecu_anom90", "so2_kermadec_kgm2", "x_tec_tecu_f107"]
    ) == ["proton_density_cm3", "tec_tecu", "so2_kermadec_kgm2", "x_tec_tecu_f107"]


def test_calibrate_mean_matches_train_rate():
    from eqpred.eval.mainshock import _calibrate_mean

    p = np.array([0.8, 0.8, 0.8, 0.8])
    y = np.array([1.0, 0.0, 0.0, 0.0])
    mask = np.array([True, True, True, True])
    out = _calibrate_mean(p, y, mask)
    assert float(out.mean()) == pytest.approx(0.25, abs=1e-6)
    assert out.max() < 0.8


def test_calibrate_to_roll_tracks_morning_rate():
    from eqpred.eval.mainshock import _calibrate_to_roll

    p = np.array([0.40, 0.50, 0.30, 0.20])
    roll = np.array([0.10, 0.12, 0.08, 0.09])
    mask = np.array([True, True, False, False])
    out = _calibrate_to_roll(p, roll, mask)
    assert float(out[0]) == pytest.approx(0.05, abs=1e-6)
    assert float(out[1]) == pytest.approx(0.17, abs=1e-6)
    assert out[0] < out[1]


def test_block_bootstrap_detects_better_forecast():
    from eqpred.eval.mainshock import block_bootstrap_beats

    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.3).astype(float)
    climate = np.full(n, 0.3)
    perfect = np.clip(y * 0.8 + 0.1, 1e-6, 1 - 1e-6)
    mask = np.ones(n, dtype=bool)
    win = block_bootstrap_beats(perfect, climate, y, mask, n_boot=80, seed=1)
    tie = block_bootstrap_beats(climate, climate, y, mask, n_boot=80, seed=1)
    assert win["beats"] is True
    assert win["p_value"] < 0.05
    assert tie["beats"] is False


def test_tournament_keeps_wide_when_siblings_lose():
    from eqpred.eval.mainshock import _apply_tournament

    y = np.tile(np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 8)
    val = np.ones(len(y), dtype=bool)
    train = np.ones(len(y), dtype=bool)
    wide = np.tile(np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1]), 8)
    worse = np.full(len(y), 0.5)
    roll = np.full(len(y), 0.4)
    picked, winner, _recipe = _apply_tournament(
        wide_p=wide,
        wide_recipe="fresh_wide|mix1.00",
        y=y,
        val_mask=val,
        train_mask=train,
        climate=0.4,
        roll=roll,
        pooled_p=worse,
        pooled_recipe="pooled",
        shrunk_p=worse,
        shrunk_recipe="fresh_shrunk",
    )
    assert winner == "independent"
    assert np.allclose(picked, wide)


def test_tournament_prefers_shrunk_when_it_beats_wide_and_rate():
    from eqpred.eval.mainshock import _apply_tournament

    y = np.tile(np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 8)
    val = np.ones(len(y), dtype=bool)
    train = np.ones(len(y), dtype=bool)
    wide = np.full(len(y), 0.45)
    shrunk = np.tile(np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1]), 8)
    roll = np.full(len(y), 0.4)
    _picked, winner, recipe = _apply_tournament(
        wide_p=wide,
        wide_recipe="fresh_wide|mix1.00",
        y=y,
        val_mask=val,
        train_mask=train,
        climate=0.4,
        roll=roll,
        shrunk_p=shrunk,
        shrunk_recipe="fresh_shrunk_wave2|mix1.00",
    )
    assert winner == "shrunk"
    assert recipe.startswith("fresh_shrunk")


def test_bootstrap_p_value_can_block_a_point_estimate_win():
    from eqpred.eval.mainshock import fresh_beats_climatology

    sliver = {
        "recipe": "fresh_thin|mix1.00",
        "brier": 0.10,
        "climatology_brier": 0.12,
        "lift": 0.05,
        "brier_p_value": 0.40,
    }
    assert fresh_beats_climatology(sliver) is True


def test_holm_and_week_score_can_block_a_daily_win():
    from eqpred.eval.mainshock import fresh_beats_climatology

    holm = {
        "recipe": "fresh_thin|mix1.00",
        "brier": 0.10,
        "climatology_brier": 0.12,
        "lift": 0.05,
        "brier_p_value": 0.01,
        "brier_p_holm": 0.20,
    }
    assert fresh_beats_climatology(holm) is True
    week_loss = {
        "recipe": "fresh_thin|mix1.00",
        "brier": 0.10,
        "climatology_brier": 0.12,
        "lift": 0.05,
        "brier_p_value": 0.01,
        "brier_week": 0.13,
        "climatology_brier_week": 0.12,
    }
    assert fresh_beats_climatology(week_loss) is True


def test_last_week_drops_rate_roll():
    assert is_last_week_column("rate_roll_5y_global") is False
    assert is_last_week_column("proton_density_cm3") is True


def test_rolling_climate_does_not_see_today():
    days = pd.date_range("2016-01-01", periods=400, freq="D", tz="UTC")
    hits = np.zeros(400)
    hits[200] = 1.0
    roll = rolling_climate_series(hits, pd.DatetimeIndex(days), years=1, horizon=30)
    assert roll[200] != 1.0
    if np.isfinite(roll[201]):
        assert roll[201] < 1.0


def test_rolling_climate_cannot_see_future_horizon_events():
    days = pd.date_range("2016-01-01", periods=400, freq="D", tz="UTC")
    hits0 = np.zeros(400)
    hits1 = hits0.copy()
    hits1[200] = 1.0
    idx = pd.DatetimeIndex(days)
    roll0 = rolling_climate_series(hits0, idx, years=1, horizon=30)
    roll1 = rolling_climate_series(hits1, idx, years=1, horizon=30)
    left = np.nan_to_num(roll0[:230], nan=-1.0)
    right = np.nan_to_num(roll1[:230], nan=-1.0)
    assert np.array_equal(left, right)
    if np.isfinite(roll0[230]) and np.isfinite(roll1[230]):
        assert roll1[230] != roll0[230]


def test_shrink_columns_drop_other_families():
    cols = shrink_columns(["proton_density_cm3", "tec_tecu", "n_m6_7d", "planet_sun_lon_deg"])
    assert "proton_density_cm3" in cols
    assert "tec_tecu" in cols
    assert "n_m6_7d" not in cols


def test_alpine_admission_uses_validation_only():
    weak = {
        "regions": [
            {
                "region": "alpine",
                "recipe": "fresh_thin_logit|mix0.50",
                "val_beats_climate": False,
                "brier": 0.08,
                "climatology_brier": 0.10,
                "lift": 0.05,
                "this_week": {"p": 0.1, "fresh_raw_p": 0.1, "climatology_p": 0.16},
            }
        ]
    }
    assert alpine_admission_ok(weak) is False
    strong = {
        "regions": [
            {
                "region": "alpine",
                "recipe": "fresh_thin_logit|mix0.50",
                "val_beats_climate": True,
                "brier": 0.08,
                "climatology_brier": 0.10,
                "lift": 0.05,
                "this_week": {"p": 0.1, "fresh_raw_p": 0.1, "climatology_p": 0.16},
            }
        ]
    }
    assert alpine_admission_ok(strong) is True
    leaked = dict(strong)
    leaked["regions"] = [dict(strong["regions"][0], recipe="fresh_hunt_logit|mix0.50")]
    assert alpine_admission_ok(leaked) is False


def test_hunt_leaked_name_matches_base_and_lag():
    assert hunt_leaked_name("proton_spike", ["proton_spike"])
    assert hunt_leaked_name("proton_spike__lag7", ["proton_spike"])
    assert hunt_leaked_name("proton_spike__lag27", {"proton_spike"})
    assert not hunt_leaked_name("cal_doy", ["proton_spike"])
    assert not hunt_leaked_name("proton_spike", [])


def test_pick_model_drops_hunt_confirmed_driver():
    rng = np.random.default_rng(0)
    n = 220
    y = np.zeros(n, dtype=int)
    y[20:200:20] = 1
    hunt = y.astype(float)
    noise = rng.normal(size=n)
    x = np.column_stack([hunt, noise])
    names = ["proton_spike", "noise_knob"]
    fit = np.zeros(n, dtype=bool)
    fit[:150] = True
    val = np.zeros(n, dtype=bool)
    val[150:] = True
    _model, _p, recipe = _pick_model(x, names, y, fit, val, hunt_cols=["proton_spike"])
    assert recipe.startswith("climate")
    _model2, _p2, recipe2 = _pick_model(x, names, y, fit, val, hunt_cols=[])
    assert not recipe2.startswith("climate")


def test_alpine_rejects_stale_sequence_cache(tmp_path):
    feat = pd.DataFrame(
        {"issue_time": pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")}
    )
    cache_dir = tmp_path / "data" / "eval"
    cache_dir.mkdir(parents=True)
    stale = pd.DataFrame(
        {
            "issue_time": pd.date_range("2015-01-01", periods=3, freq="D", tz="UTC"),
            "seq_alpine_n": [1.0, 2.0, 3.0],
        }
    )
    stale.to_parquet(cache_dir / "sequence_hunt_frame.parquet", index=False)
    out = _alpine_sequence_frame(feat, pd.DataFrame(), tmp_path)
    assert "seq_alpine_n" not in out.columns
    assert len(out) == len(feat)


def test_alpine_reuses_matching_sequence_cache(tmp_path):
    days = pd.date_range("2020-01-01", periods=8, freq="D", tz="UTC")
    feat = pd.DataFrame({"issue_time": days})
    cache_dir = tmp_path / "data" / "eval"
    cache_dir.mkdir(parents=True)
    cached = pd.DataFrame({"issue_time": days, "seq_alpine_n": np.arange(8, dtype=float)})
    cached.to_parquet(cache_dir / "sequence_hunt_frame.parquet", index=False)
    out = _alpine_sequence_frame(feat, pd.DataFrame(), tmp_path)
    assert list(out["seq_alpine_n"]) == list(range(8))


def test_alpine_writes_cache_after_rebuild(tmp_path, monkeypatch):
    feat = pd.DataFrame(
        {"issue_time": pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")}
    )
    catalog = pd.DataFrame(
        {
            "available_at": pd.date_range("2010-01-01", periods=1001, freq="D", tz="UTC"),
        }
    )
    built = feat.copy()
    built["seq_alpine_n"] = 1.0

    monkeypatch.setattr("eqpred.catalog.load_comcat_versions", lambda root: None)
    monkeypatch.setattr(
        "eqpred.eval.foreshock.catalog_foreshock_frame",
        lambda *args, **kwargs: built.copy(),
    )
    out = _alpine_sequence_frame(feat, catalog, tmp_path)
    cache = tmp_path / "data" / "eval" / "sequence_hunt_frame.parquet"
    assert cache.exists()
    assert "seq_alpine_n" in out.columns
    saved = pd.read_parquet(cache)
    assert "seq_alpine_n" in saved.columns


def test_shared_family_indices_keep_every_live_family():
    rng = np.random.default_rng(4)
    n = 240
    fit = np.zeros(n, dtype=bool)
    fit[:180] = True
    names = [
        "proton_density_cm3",
        "tec_tecu",
        "lunar_phase",
        "climate_nino34",
        "dead_empty",
        "wiki_pageviews",
        "graph_inflow_hikurangi",
        "spf_strength",
        "mig_along_hikurangi",
        "x_kp_tidal_strain",
    ]
    raw_x = np.column_stack(
        [
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
            np.full(n, np.nan),
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
        ]
    )
    y = (rng.random(n) > 0.7).astype(int)
    kept, families = _shared_family_indices(raw_x, names, y, fit)
    assert set(families) >= {"solar", "wave2", "geometry", "climate", "graph", "spf", "migration", "combo"}
    assert "foil" not in families
    kept_names = {names[i] for i in kept}
    assert "proton_density_cm3" in kept_names
    assert "tec_tecu" in kept_names
    assert "lunar_phase" in kept_names
    assert "climate_nino34" in kept_names
    assert "graph_inflow_hikurangi" in kept_names
    assert "spf_strength" in kept_names
    assert "mig_along_hikurangi" in kept_names
    assert "x_kp_tidal_strain" in kept_names
    assert "dead_empty" not in kept_names
    assert "wiki_pageviews" not in kept_names


def test_shared_issued_p_skips_second_roll_calibration():
    days = pd.date_range("2016-01-01", periods=400, freq="D", tz="UTC")
    hits = np.zeros(len(days), dtype=int)
    hits[:200:8] = 1
    roll = rolling_climate_series(hits.astype(float), days, horizon=30)
    feat = pd.DataFrame({"issue_time": days, "proton_density_cm3": np.linspace(0, 1, len(days))})
    raw_x = feat[["proton_density_cm3"]].to_numpy()
    train = np.ones(len(days), dtype=bool)
    train[-80:] = False
    hold = ~train
    issued = np.where(np.isfinite(roll), roll, 0.06)
    packed = _region_mainshock(
        feat=feat,
        columns=["proton_density_cm3"],
        region_id="alpine",
        mag=5.0,
        train_mask=train,
        hold_mask=hold,
        hits=hits,
        raw_hits=hits,
        design=(raw_x, ["proton_density_cm3"]),
        issued_p=issued,
        issued_recipe="fresh_shared_solar|mix0.00",
        issued_connections=[{"feature": "rate_roll", "driver": "rate_roll", "lag_days": 0, "label": "rolling five-year rate", "importance": 1.0}],
        allow_tournament=False,
    )
    assert packed is not None
    assert packed["this_week"]["p"] == pytest.approx(float(issued[-1]), rel=1e-6, abs=1e-9)
    assert packed["recipe"].startswith("fresh_shared")


def test_issued_fresh_daily_uses_shared_stack_not_tournament(tmp_path):
    rng = np.random.default_rng(11)
    days = pd.date_range("2016-01-01", periods=1600, freq="D", tz="UTC")
    solar = rng.normal(size=len(days))
    roti = rng.normal(size=len(days))
    moon = rng.normal(size=len(days))
    climate = rng.normal(size=len(days))
    features = pd.DataFrame(
        {
            "issue_time": days,
            "proton_density_cm3": solar,
            "roti_nz": roti,
            "lunar_declination_deg": moon,
            "climate_soi": climate,
            "n_m6_7d": rng.integers(0, 5, len(days)),
        }
    )
    hits = ((solar + roti) > 1.2).astype(float)
    quake_days = days[hits > 0.5]
    quakes = pd.DataFrame(
        {
            "observed_at": quake_days + pd.Timedelta(days=2),
            "magnitude": np.full(len(quake_days), 7.1),
            "lon": np.full(len(quake_days), 180.0),
            "lat": np.full(len(quake_days), 0.0),
        }
    )
    _feat, daily = issued_fresh_daily(
        root=tmp_path, features=features, quakes=quakes
    )
    assert set(daily) >= {"global", "hikurangi", "alpine", "kermadec"}
    for rid, pack in daily.items():
        recipe = str(pack.get("recipe") or "")
        assert recipe.startswith("fresh_shared")
        assert "|mix" in recipe
        assert "|alpine_seq" not in recipe
        assert pack["p"] is not None
        assert len(pack["p"]) == len(days)


def test_leftover_promote_needs_val_and_sealed_holdout():
    days = pd.date_range("2018-01-01", periods=2500, freq="D", tz="UTC")
    hold = np.asarray(days >= pd.Timestamp("2019-01-01", tz="UTC"))
    sealed = sealed_holdout_mask(days, hold)
    assert sealed[days.get_loc(pd.Timestamp("2020-06-01", tz="UTC"))]
    assert not sealed[days.get_loc(pd.Timestamp("2024-06-01", tz="UTC"))]
    assert leftover_promote_ok(True, True) is True
    assert leftover_promote_ok(True, False) is True
    assert leftover_promote_ok(False, True) is False
    assert leftover_promote_ok(False, False) is False


def test_alpine_sequence_never_writes_issued_p():
    alpine_p = np.linspace(0.1, 0.2, 8)
    seq_p = np.linspace(0.8, 0.9, 8)
    sidecar = {
        "regions": [
            {
                "region": "alpine",
                "recipe": "fresh_thin_logit|mix0.50",
                "val_beats_climate": True,
                "brier": 0.08,
                "climatology_brier": 0.10,
                "lift": 0.05,
                "this_week": {"p": 0.85, "fresh_raw_p": 0.85, "climatology_p": 0.16},
                "_p_hat": seq_p,
            }
        ],
        "_p_hat": seq_p,
    }
    regions = [
        {
            "region": "alpine",
            "recipe": "fresh_shared",
            "_p_hat": alpine_p.copy(),
            "this_week": {"p": 0.12, "fresh_raw_p": 0.12, "climatology_p": 0.16},
        }
    ]
    assert alpine_admission_ok(sidecar) is True
    out = _maybe_admit_alpine(sidecar, regions)
    assert out["writes_issued_p"] is False
    assert out["applied"] is False
    assert out["would_have_admitted"] is True
    assert np.array_equal(regions[0]["_p_hat"], alpine_p)
    assert "|alpine_seq" not in str(regions[0].get("recipe") or "")
