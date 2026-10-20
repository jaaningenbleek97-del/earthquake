import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.honest import purged_train_hold
from eqpred.eval.leftover import (
    _bound_residual,
    attach_leftover_sidecar,
    fit_leftover_region,
    leftover_columns,
    load_leftover,
    run_leftover,
    schedule_columns,
)
from eqpred.paths import data_dir, prediction_dir


def test_runaway_residual_is_clipped_before_it_can_saturate():
    hat = _bound_residual(np.array([-14.7, 0.2, 25.0]))
    assert hat.tolist() == [-1.0, 0.2, 1.0]
    feat = pd.DataFrame(
        {
            "issue_time": pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC"),
            "cal_doy_sin": [0.1, 0.2, 0.3, 0.4],
            "tide_nz_mm": [1.0, 1.0, 1.0, 1.0],
            "proton_density_cm3": [4.0, 5.0, 6.0, 7.0],
            "wiki_eq_views": [10.0, 11.0, 12.0, 13.0],
            "vlf_amp": [1.0, 1.1, 1.2, 1.3],
            "tir_ts": [10.0, 11.0, 10.5, 9.0],
            "hydro_gwet": [0.2, 0.3, 0.4, 0.5],
            "cal_weekday": [0.0, 1.0, 2.0, 3.0],
            "n_m6_7d": [0.0, 1.0, 0.0, 0.0],
        }
    )
    assert "proton_density_cm3" in leftover_columns(feat)
    assert "cal_doy_sin" not in leftover_columns(feat)
    assert "wiki_eq_views" not in leftover_columns(feat)
    assert "vlf_amp" not in leftover_columns(feat)
    assert "tir_ts" not in leftover_columns(feat)
    assert "hydro_gwet" not in leftover_columns(feat)
    assert "hydro_gwet" in schedule_columns(feat)
    assert "cal_doy_sin" in schedule_columns(feat)
    assert "tide_nz_mm" in schedule_columns(feat)


def test_leftover_beats_schedule_on_planted_residual():
    n = 2400
    days = pd.date_range("2012-01-01", periods=n, freq="D", tz="UTC")
    doy = days.dayofyear.to_numpy()
    season = np.sin(2.0 * np.pi * doy / 365.25)
    leftover_sig = np.sin(2.0 * np.pi * np.arange(n) / 47.0)
    rng = np.random.default_rng(4)
    p = np.clip(0.06 + 0.10 * (season > 0) + 0.42 * (leftover_sig > 0.15), 0.01, 0.9)
    hits = (rng.random(n) < p).astype(float)
    feat = pd.DataFrame(
        {
            "issue_time": days,
            "cal_doy_sin": np.sin(2.0 * np.pi * doy / 365.25),
            "cal_doy_cos": np.cos(2.0 * np.pi * doy / 365.25),
            "proton_density_cm3": leftover_sig,
            "tec_tecu": rng.normal(size=n),
        }
    )
    closed = np.ones(n, dtype=bool)
    closed[-40:] = False
    train, hold = purged_train_hold(days, closed, "2016-12-31", "2017-01-01")
    packed = fit_leftover_region(
        feat,
        hits,
        region_id="global",
        mag=7.0,
        train_mask=train,
        hold_mask=hold,
    )
    assert packed["writes_issued_p"] is False
    assert packed["beats_schedule"] is True
    assert packed["this_week"]["writes_issued_p"] is False
    assert float(np.max(np.abs(packed["leftover_hat"]))) <= 1.0


def test_hikurangi_leftover_uses_sse_two_rate_clock():
    n = 400
    days = pd.date_range("2014-01-01", periods=n, freq="D", tz="UTC")
    sse = np.zeros(n)
    sse[80:200] = 1.0
    rng = np.random.default_rng(2)
    hits = np.zeros(n)
    on = sse > 0.5
    hits[on] = (rng.random(int(on.sum())) < 0.35).astype(float)
    hits[~on] = (rng.random(int((~on).sum())) < 0.04).astype(float)
    feat = pd.DataFrame(
        {
            "issue_time": days,
            "sse_on": sse,
            "proton_density_cm3": rng.normal(size=n),
            "cal_doy_sin": np.sin(2.0 * np.pi * days.dayofyear / 365.25),
            "cal_doy_cos": np.cos(2.0 * np.pi * days.dayofyear / 365.25),
        }
    )
    train = np.ones(n, dtype=bool)
    train[320:] = False
    packed = fit_leftover_region(
        feat,
        hits,
        region_id="hikurangi",
        mag=5.0,
        train_mask=train,
        hold_mask=~train,
    )
    assert packed["writes_issued_p"] is False
    assert packed["regime_kind"] == "sse_two_rate"
    assert packed["sse_rate_on"] > packed["sse_rate_off"]


def test_leftover_does_not_write_issued_cards_or_hollow_mainshock(tmp_path):
    n = 400
    start = datetime(2016, 1, 1, tzinfo=timezone.utc)
    days = [start + timedelta(days=i) for i in range(n)]
    issue = pd.to_datetime(days, utc=True)
    rng = np.random.default_rng(1)
    quakes = pd.DataFrame(
        [
            {
                "observed_at": days[i] + timedelta(days=2),
                "available_at": days[i] + timedelta(days=2, hours=6),
                "magnitude": 7.1,
                "lat": 0.0,
                "lon": 0.0,
                "event_id": f"ev{i}",
            }
            for i in range(60, n - 40, 80)
        ]
    )
    feat = pd.DataFrame(
        {
            "issue_time": issue,
            "cal_doy_sin": np.sin(2.0 * np.pi * issue.dayofyear / 365.25),
            "cal_doy_cos": np.cos(2.0 * np.pi * issue.dayofyear / 365.25),
            "proton_density_cm3": rng.normal(size=n),
        }
    )
    dest = data_dir(tmp_path) / "eval"
    dest.mkdir(parents=True)
    (dest / "mainshock.json").write_text(
        json.dumps(
            {
                "writes_issued_p": True,
                "week": [{"region": "global", "p": 0.11, "keep": True}],
                "regions": [{"region": "global", "p": 0.11}],
            }
        ),
        encoding="utf-8",
    )
    report = run_leftover(
        train_end="2016-12-31",
        holdout_start="2017-01-01",
        root=tmp_path,
        features=feat,
        quakes=quakes,
        targets=(("global", 7.0),),
        attach=True,
    )
    assert report["writes_issued_p"] is False
    assert report["attached_to_mainshock"] is True
    assert (tmp_path / "data" / "eval" / "leftover.json").exists()
    pack = json.loads((dest / "mainshock.json").read_text(encoding="utf-8"))
    assert pack["writes_issued_p"] is True
    assert pack["week"][0]["keep"] is True
    assert pack["leftover"]["writes_issued_p"] is False
    assert not list(prediction_dir(tmp_path).glob("batch-*.json"))
    assert load_leftover(tmp_path)["writes_issued_p"] is False


def test_leftover_skips_missing_mainshock(tmp_path):
    report = run_leftover(root=tmp_path, attach=True)
    assert report["writes_issued_p"] is False
    assert report["attached_to_mainshock"] is False
    assert not (data_dir(tmp_path) / "eval" / "mainshock.json").exists()
    assert attach_leftover_sidecar({"week": []}, tmp_path) is False
