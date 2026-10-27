import json

import numpy as np
import pandas as pd

from eqpred.eval.attempts import forecast_frame, list_attempts, save_forecasts
from eqpred.eval.strategy_desk import score_strategy


def _days(n=90):
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


def test_an_open_window_is_kept_without_a_result(tmp_path):
    days = _days(40)
    p = np.full(40, 0.2)
    climate = np.full(40, 0.1)
    hits = np.zeros(40)
    mask = np.zeros(40, dtype=bool)
    mask[0] = True
    mask[-1] = True
    frame = forecast_frame(
        days=days,
        region="kermadec",
        label="Kermadec",
        horizon=30,
        magnitude=6,
        p=p,
        climatology=climate,
        happened=hits,
        mask=mask,
        catalog_end=days[-1],
    )
    assert frame.loc[0, "slice"] == "later"
    assert frame.loc[0, "happened"] == 0
    assert bool(frame.loc[1, "closed"]) is False
    assert np.isnan(frame.loc[1, "happened"])
    saved = save_forecasts(frame, source="strategy", name="alarm", root=tmp_path, answer_key="fresh.v1")
    again = save_forecasts(frame, source="strategy", name="alarm", root=tmp_path, answer_key="fresh.v1")
    assert saved["attempt_id"] != again["attempt_id"]
    assert saved["writes_issued_p"] is False
    summary = list_attempts(tmp_path)
    assert summary["n_attempts"] == 2
    assert summary["n_forecasts"] == 4
    stored = pd.read_parquet(tmp_path / "data" / "eval" / "attempts" / saved["file"])
    assert list(stored["issue_day"]) == list(frame["issue_day"])
    assert not (tmp_path / "data" / "eval" / "frozen" / "ledger.json").exists()
    raw = (tmp_path / "data" / "eval" / "attempts" / "index.jsonl").read_text(encoding="utf-8")
    assert json.loads(raw.splitlines()[0])["kind"] == "rehearsal"


def test_a_strategy_replay_saves_its_forecasts(tmp_path):
    n = 120
    folder = tmp_path / "data" / "eval" / "tiny_tape"
    folder.mkdir(parents=True)
    meta = {
        "horizons": {
            "30": {
                "regions": ["global"],
                "groups": {"global": ["solar"]},
                "mags": {"global": 7.0},
            }
        },
        "days": n,
        "start": "2000-01-01",
        "train_end": "2000-03-01",
        "holdout_start": "2000-04-01",
        "label_version": 1,
        "fit_hash": "abc",
        "built_at": "2020-01-01T00:00:00Z",
    }
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    model = np.full(n, 0.3)
    zeros = np.zeros(n)
    fit = np.zeros(n, dtype=bool)
    fit[:60] = True
    val = np.zeros(n, dtype=bool)
    val[60:80] = True
    exam = np.zeros(n, dtype=bool)
    exam[90] = True
    np.savez(
        folder / "tape.npz",
        matrix=np.zeros((n, 2), dtype=np.float32),
        **{
            "30_fit": fit,
            "30_val": val,
            "30_exam": exam,
            "30_global_model": model,
            "30_global_live": model,
            "30_global_base": model,
            "30_global_y": zeros,
            "30_global_c": np.full(n, 0.1),
            "30_global_g0": model,
        },
    )
    report = score_strategy(horizon=30, region="global", strategy="current", root=tmp_path, tape_name="tiny_tape")
    assert report["ready"] is True
    assert report["saved"]["n_forecasts"] >= 2
    assert report["saved"]["writes_issued_p"] is False
    assert "rehearsal forecasts" in report["note"]
    summary = list_attempts(tmp_path)
    assert summary["n_attempts"] == 1
    assert summary["recent"][0]["answer_key"] == "fresh.v1"
