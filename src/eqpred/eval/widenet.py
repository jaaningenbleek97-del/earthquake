"""S0-style wide-net foil. Japan or Chile, M5+, 7 days. Never issued p."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat
from eqpred.eval.brain import _brier
from eqpred.eval.honest import purged_train_hold, rolling_climate_series
from eqpred.eval.skill import information_gain, rank_alarm
from eqpred.eval.strategies import _window_hits
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc

WRITES_ISSUED_P = False
HORIZON_DAYS = 7
MAG = 5.0
ALARM_BOXES = ("japan", "chile")
POLYNESIA = {
    "kind": "bbox",
    "name": "polynesia",
    "label": "Polynesia",
    "issued": False,
    "min_lon": -180.0,
    "max_lon": -140.0,
    "min_lat": -30.0,
    "max_lat": 0.0,
}


def load_widenet(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "widenet.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_widenet(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    quakes: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Alarm when Japan or Chile had an M5+ in the previous 7 days.

    The next-7-day union is the scored event. Polynesia is confirm-only.
    """
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    if features is None:
        path = feature_dir(root) / "global_daily.parquet"
        feat = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    else:
        feat = features
    catalog = quakes.copy() if quakes is not None else load_comcat(root)
    empty = {
        "writes_issued_p": False,
        "id": "foil.widenet.v1",
        "horizon_days": HORIZON_DAYS,
        "magnitude_min": MAG,
        "note": "Wide-net foil needs a daily stack and a catalog. It does not write issued p.",
        "scanned_at": iso(datetime.now(timezone.utc)),
    }
    if feat.empty or "issue_time" not in feat.columns or catalog.empty:
        (dest / "widenet.json").write_text(json.dumps(empty, indent=2), encoding="utf-8")
        return empty
    feat = feat.copy()
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    _train, hold = purged_train_hold(days, closed, train_end, holdout_start, purge=HORIZON_DAYS)
    parts = [
        _window_hits(catalog, REGIONS[rid], MAG, days, HORIZON_DAYS).astype(bool)
        for rid in ALARM_BOXES
    ]
    union = np.logical_or.reduce(parts)
    prior = np.zeros(len(days), dtype=bool)
    if len(days) > HORIZON_DAYS:
        prior[HORIZON_DAYS:] = union[:-HORIZON_DAYS]
    roll = rolling_climate_series(union.astype(float), days, horizon=HORIZON_DAYS)
    climate = np.where(np.isfinite(roll), roll, float(np.nanmean(union.astype(float))))
    y = union.astype(float)
    alarm = rank_alarm(prior[hold].astype(float), y[hold], train_p=prior[_train].astype(float))
    # The alarm's probability is 1 on prior-hit mornings and the climate elsewhere,
    # so the ledger can show Brier and IG without touching issued p.
    p = np.where(prior, np.maximum(climate, 0.5), climate)
    ig = information_gain(p[hold], y[hold], climate[hold])
    poly = _window_hits(catalog, POLYNESIA, MAG, days, HORIZON_DAYS).astype(bool)
    alarmed = hold & prior
    n_alarm = int(alarmed.sum())
    confirm = float(poly[alarmed].mean()) if n_alarm else None
    latest = int(np.argmax(days.to_numpy()))
    report = {
        "writes_issued_p": False,
        "id": "foil.widenet.v1",
        "horizon_days": HORIZON_DAYS,
        "magnitude_min": MAG,
        "boxes": list(ALARM_BOXES),
        "confirm_box": "polynesia",
        "n_holdout": int(hold.sum()),
        "hit_rate": float(y[hold].mean()) if hold.any() else None,
        "brier": _brier(p[hold], y[hold]),
        "climatology_brier": _brier(climate[hold], y[hold]),
        "information_gain": ig,
        "lift": alarm.get("lift"),
        "time_in_alarm": alarm.get("time_in_alarm"),
        "polynesia_confirm_rate": confirm,
        "issue_day": days[latest].date().isoformat(),
        "alarm_on": bool(prior[latest]),
        "scanned_at": iso(datetime.now(timezone.utc)),
    }
    report["note"] = (
        f"Wide-net foil: Japan or Chile M{MAG:g}+ inside {HORIZON_DAYS} days, "
        f"alarmed from the previous window. "
        f"Holdout hit rate {report['hit_rate']}. "
        f"Polynesia confirm-only {confirm}. Not the published number."
    )
    (dest / "widenet.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
