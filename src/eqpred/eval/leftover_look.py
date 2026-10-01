from __future__ import annotations

"""Last 40 leftover-lookalike mornings, and which issued box was next.

Research extra. Uses leftover residual shape, not the published model.
Never writes issued p.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat
from eqpred.eval.brain import _design, _impute
from eqpred.eval.decluster import fresh_hits
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.honest import purged_train_hold
from eqpred.eval.leftover import (
    ensure_schedule,
    leftover_columns,
    load_leftover,
)
from eqpred.eval.mainshock import ISSUED_TARGETS, _chance
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from sklearn.preprocessing import StandardScaler

from eqpred.stamp import parse_utc

WRITES_ISSUED_P = False
LOOK_ID = "fresh.leftover_look.v1"
K_ANALOGS = 40
BOX_TARGETS = tuple((rid, mag) for rid, mag in ISSUED_TARGETS if rid != "global")


def load_leftover_look(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "leftover_look.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _neighbors(x_all: np.ndarray, query: int, library: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if library.size == 0 or x_all.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    delta = x_all[library] - x_all[query]
    dist = np.sqrt(np.sum(delta * delta, axis=1))
    take = min(int(k), int(library.size))
    order = np.argpartition(dist, take - 1)[:take]
    order = order[np.argsort(dist[order])]
    return library[order], dist[order]


def run_leftover_look(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
    k: int = K_ANALOGS,
) -> dict[str, Any]:
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    empty = {
        "writes_issued_p": False,
        "id": LOOK_ID,
        "mode": "leftover_look",
        "analogs": [],
        "which_box": None,
        "note": "Leftover analog extra needs a daily feature stack. It never writes the published number.",
        "scanned_at": iso(datetime.now(timezone.utc)),
    }
    if features is None:
        path = feature_dir(root) / "global_daily.parquet"
        if not path.exists():
            (dest / "leftover_look.json").write_text(json.dumps(empty, indent=2, default=str), encoding="utf-8")
            return empty
        feat = pd.read_parquet(path)
    else:
        feat = features.copy()
    if feat.empty or "issue_time" not in feat.columns:
        (dest / "leftover_look.json").write_text(json.dumps(empty, indent=2, default=str), encoding="utf-8")
        return empty
    feat = ensure_schedule(feat)
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog = quakes.copy() if quakes is not None else load_comcat(root)
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    cols = leftover_columns(feat)
    raw_x, _names = _design(feat, cols, lags=(0, 7, 14))
    _train, x_all = _impute(raw_x[train_mask] if raw_x.size else raw_x, raw_x)
    query = int(len(days) - 1)
    library = np.flatnonzero(closed & (np.arange(len(days)) != query))
    if x_all.size and library.size:
        scaler = StandardScaler(with_mean=True)
        scaler.fit(x_all[library])
        x_all = scaler.transform(x_all)
    neigh, dist = _neighbors(x_all, query, library, k)
    hits = {
        rid: fresh_hits(catalog, REGIONS[rid], mag, days, HORIZON_DAYS).astype(int)
        for rid, mag in ISSUED_TARGETS
    }
    analogs: list[dict[str, Any]] = []
    box_hits = {rid: 0 for rid, _mag in BOX_TARGETS}
    for idx, distance in zip(neigh, dist, strict=True):
        followed = {rid: bool(hits[rid][int(idx)]) for rid, _mag in ISSUED_TARGETS}
        for rid, _mag in BOX_TARGETS:
            if followed[rid]:
                box_hits[rid] += 1
        analogs.append(
            {
                "issue_day": days[int(idx)].date().isoformat(),
                "distance": float(distance),
                "followed": followed,
            }
        )
    n = max(len(analogs), 1)
    rates = {rid: box_hits[rid] / n for rid, _mag in BOX_TARGETS}
    which = max(rates, key=rates.get) if analogs else None
    which_rate = rates.get(which) if which else None
    leftover = load_leftover(root) or {}
    sentence = (
        f"The last {len(analogs)} closed mornings that looked like this leftover "
        + (
            f"next hit {REGIONS[which]['label']} most often ({_chance(which_rate)} of those analogs)."
            if which
            else "did not name a box."
        )
        + " This is not the published number."
    )
    report = {
        "writes_issued_p": False,
        "id": LOOK_ID,
        "mode": "leftover_look",
        "horizon_days": HORIZON_DAYS,
        "k": k,
        "n_analogs": len(analogs),
        "n_leftover_columns": len(cols),
        "issue_day": days[query].date().isoformat(),
        "analogs": analogs,
        "which_box": which,
        "which_box_label": None if which is None else REGIONS[which]["label"],
        "which_box_rate": which_rate,
        "box_rates": rates,
        "leftover_issue_day": leftover.get("issue_day"),
        "sentence": sentence,
        "note": (
            f"Leftover analog extra: {len(analogs)} closed mornings nearest in leftover-source "
            f"space. Which-box-next is {REGIONS[which]['label'] if which else 'none'} "
            f"({_chance(which_rate) if which_rate is not None else '—'} of analogs). "
            "It never writes the published number."
        ),
        "scanned_at": iso(datetime.now(timezone.utc)),
    }
    (dest / "leftover_look.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    main = dest / "mainshock.json"
    if main.exists():
        pack = json.loads(main.read_text(encoding="utf-8"))
        pack["leftover_look"] = {
            "id": LOOK_ID,
            "mode": "leftover_look",
            "writes_issued_p": False,
            "n_analogs": report["n_analogs"],
            "which_box": which,
            "which_box_label": report["which_box_label"],
            "which_box_rate": which_rate,
            "analogs": analogs[:12],
            "sentence": sentence,
            "note": report["note"],
        }
        main.write_text(json.dumps(pack, indent=2, default=str), encoding="utf-8")
        report["attached_to_mainshock"] = True
    else:
        report["attached_to_mainshock"] = False
    return report
