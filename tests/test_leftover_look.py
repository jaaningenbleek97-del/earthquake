from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.leftover_look import run_leftover_look
from eqpred.paths import data_dir, prediction_dir


def test_leftover_look_names_the_box_that_followed_neighbors(tmp_path):
    n = 280
    start = datetime(2016, 1, 1, tzinfo=timezone.utc)
    days = pd.date_range(start, periods=n, freq="D", tz="UTC")
    analog = list(range(18, 240, 18))
    leftover_sig = np.zeros(n)
    leftover_sig[analog] = 100.0
    leftover_sig[-1] = 100.0
    feat = pd.DataFrame(
        {
            "issue_time": days,
            "proton_density_cm3": leftover_sig,
            "cal_doy_sin": np.sin(2.0 * np.pi * days.dayofyear / 365.25),
        }
    )
    quakes = []
    for i in analog:
        when = days[i] + timedelta(days=3)
        quakes.append(
            {
                "observed_at": when,
                "available_at": when + timedelta(hours=6),
                "magnitude": 5.2,
                "lat": -39.5,
                "lon": 177.2,
                "event_id": f"hik{i}",
            }
        )
    quakes.append(
        {
            "observed_at": days[-1] + timedelta(days=40),
            "available_at": days[-1] + timedelta(days=40, hours=6),
            "magnitude": 4.6,
            "lat": 0.0,
            "lon": 0.0,
            "event_id": "tail",
        }
    )
    report = run_leftover_look(
        train_end="2016-12-31",
        holdout_start="2017-01-01",
        root=tmp_path,
        features=feat,
        quakes=pd.DataFrame(quakes),
        k=12,
    )
    assert report["writes_issued_p"] is False
    assert report["n_analogs"] == 12
    assert report["which_box"] == "hikurangi"
    assert report["which_box_rate"] > 0.5
    pack = (data_dir(tmp_path) / "eval" / "leftover_look.json").read_text(encoding="utf-8")
    assert "hikurangi" in pack
    assert not list(prediction_dir(tmp_path).glob("batch-*.json"))
