from __future__ import annotations

"""Train on one arc, score the other. Same Sun, different catalog.

Does not write issued p.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.eval.brain import (
    HORIZON_DAYS,
    MIN_SCORE,
    MIN_TRAIN,
    _brier,
    _design,
    _driver_columns,
    _fit_tree,
    _impute,
    _mean,
)
from eqpred.eval.hunt import TARGETS, _etas_p
from eqpred.eval.strategies import _window_hits as _hits
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc, utc_ts

PAIRS = (("japan", "chile"), ("chile", "japan"))


def load_transfer(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "transfer.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_transfer(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = feat.copy()
    feat["issue_time"] = parse_utc(feat["issue_time"])
    if quakes is None:
        catalog = load_comcat(root)
        versions = load_comcat_versions(root)
    else:
        catalog = quakes
        versions = None
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    if "available_at" in catalog.columns:
        catalog["available_at"] = parse_utc(catalog["available_at"])

    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_cut = utc_ts(train_end)
    hold_cut = utc_ts(holdout_start)
    train_mask = closed & (days <= train_cut)
    hold_mask = closed & (days >= hold_cut)

    columns = _driver_columns(feat, unconventional_only=True)
    raw_x, _names = _design(feat, columns)
    mag_by = {rid: mag for rid, mag in TARGETS}
    rows: list[dict[str, Any]] = []
    for train_id, score_id in PAIRS:
        if train_id not in REGIONS or score_id not in REGIONS:
            continue
        train_mag = mag_by.get(train_id, 6.0)
        score_mag = mag_by.get(score_id, 6.0)
        train_hits = _hits(catalog, REGIONS[train_id], train_mag, days, HORIZON_DAYS)
        score_hits = _hits(catalog, REGIONS[score_id], score_mag, days, HORIZON_DAYS)
        train_etas = _etas_p(catalog, days, train_id, train_mag, root, versions=versions)
        score_etas = _etas_p(catalog, days, score_id, score_mag, root, versions=versions)
        if int(train_mask.sum()) < MIN_TRAIN or int(hold_mask.sum()) < MIN_SCORE:
            continue
        if raw_x.shape[1] == 0:
            continue
        x_train, x_all = _impute(raw_x[train_mask], raw_x)
        residual = train_hits.astype(float) - train_etas
        model = _fit_tree(x_train, residual[train_mask])
        if model is None:
            continue
        pred = model.predict(x_all)
        p_hat = np.clip(score_etas + pred, 1e-6, 1 - 1e-6)
        hold_y = score_hits[hold_mask].astype(float)
        rows.append(
            {
                "train_region": train_id,
                "score_region": score_id,
                "train_label": REGIONS[train_id].get("label") or train_id,
                "score_label": REGIONS[score_id].get("label") or score_id,
                "n_train": int(train_mask.sum()),
                "n_score": int(hold_mask.sum()),
                "brier": _brier(p_hat[hold_mask], hold_y),
                "etas_brier": _brier(score_etas[hold_mask], hold_y),
                "hit_rate": float(hold_y.mean()) if hold_y.size else None,
            }
        )

    beats = [r for r in rows if r.get("brier") is not None and r.get("etas_brier") is not None and r["brier"] < r["etas_brier"]]
    report = {
        "writes_issued_p": False,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "n_pairs": len(rows),
        "n_beats_etas": len(beats),
        "pairs": rows,
        "mean_brier": _mean([r.get("brier") for r in rows]),
        "mean_etas_brier": _mean([r.get("etas_brier") for r in rows]),
        "scanned_at": iso(datetime.now(timezone.utc)),
        "note": "",
    }
    if not rows:
        report["note"] = "No transfer pair had enough closed weeks. Issued p is unchanged."
    elif beats:
        names = ", ".join(f"{r['train_label']}→{r['score_label']}" for r in beats)
        report["note"] = (
            f"Transfer beat ETAS error on {len(beats)} pair(s): {names}. "
            "Same unconventional stack, other catalog. Issued p is unchanged."
        )
    else:
        report["note"] = (
            "Train one arc and score the other did not beat ETAS error. "
            "The signal is not a shared Sun signal we can move. Issued p is unchanged."
        )
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "transfer.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
