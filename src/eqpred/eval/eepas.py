"""90-day EEPAS foil. Medium-term catalog physics, not issued p.

Rhoades–Evison style: every earlier event contributes a decaying rate of
later larger shocks. Time kernel is months–years, not Omori hours.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.eval.brain import _brier
from eqpred.eval.hunt import TARGETS, _etas_p
from eqpred.eval.strategies import _window_hits
from eqpred.models.gr import load_gr, rates_to_p
from eqpred.models.predict import load_models
from eqpred.paths import data_dir, feature_dir
from eqpred.regions import REGIONS, in_bbox
from eqpred.records import iso
from eqpred.stamp import parse_utc, utc_ts

HORIZON_DAYS = 90
# Medium-term kernel. c in days, p > 1.
EEPAS_C = 30.0
EEPAS_P = 1.15
EEPAS_A = -1.8
EEPAS_B = 0.65


def eepas_daily_rate(
    events: pd.DataFrame,
    days: pd.DatetimeIndex,
    m0: float,
) -> np.ndarray:
    """Expected daily count at completeness from past events (history only)."""
    n = len(days)
    rate = np.zeros(n, dtype=float)
    if events is None or events.empty:
        return rate
    work = events.loc[pd.to_numeric(events["magnitude"], errors="coerce") >= m0].copy()
    if work.empty:
        return rate
    work["observed_at"] = parse_utc(work["observed_at"])
    origin = days[0]
    t_ev = ((work["observed_at"] - origin) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    mag = pd.to_numeric(work["magnitude"], errors="coerce").to_numpy(dtype=float)
    t_day = np.arange(n, dtype=float)
    one_m_p = 1.0 - EEPAS_P
    prod = 10.0 ** (EEPAS_A + EEPAS_B * (mag - m0))
    for ti, pr in zip(t_ev, prod):
        if not np.isfinite(pr) or not np.isfinite(ti):
            continue
        start = max(int(np.floor(ti)) + 1, 0)
        if start >= n:
            continue
        t0 = t_day[start:] - ti
        t1 = t0 + 1.0
        integ = pr * ((t1 + EEPAS_C) ** one_m_p - (t0 + EEPAS_C) ** one_m_p) / one_m_p
        rate[start:] += np.maximum(integ, 0.0)
    return rate


def load_eepas(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "eepas.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_eepas(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
) -> dict[str, Any]:
    feat = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog = load_comcat(root)
    versions = load_comcat_versions(root)
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    etas, _mod = load_models(root)
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    hold = (days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end) & (days >= utc_ts(holdout_start))
    issued = {"global", "hikurangi", "alpine", "kermadec"}
    latest = int(np.argmax(days.to_numpy())) if len(days) else None
    rows = []
    week: list[dict[str, Any]] = []
    for rid, mag in TARGETS:
        region = REGIONS[rid]
        if region.get("kind") == "bbox":
            hist = catalog.loc[in_bbox(catalog, region)]
        else:
            hist = catalog
        rate = eepas_daily_rate(hist, days, etas.m0)
        hits = _window_hits(catalog, region, mag, days, HORIZON_DAYS)
        gr = load_gr(root)
        b = float(gr.get("b") or 1.0)
        m0 = float(gr.get("m0") or etas.m0)
        p_eepas = rates_to_p(rate, mag, m0, b, HORIZON_DAYS)
        p_etas = _etas_p(catalog, days, rid, mag, root, versions=versions, horizon_days=HORIZON_DAYS)
        y = hits[hold].astype(float)
        row = {
            "region": rid,
            "magnitude_min": mag,
            "n_holdout": int(hold.sum()),
            "hit_rate": float(y.mean()) if y.size else None,
            "eepas_brier": _brier(p_eepas[hold], y),
            "etas_brier": _brier(p_etas[hold], y),
            "mean_eepas_p": float(np.nanmean(p_eepas[hold])) if y.size else None,
            "mean_etas_p": float(np.nanmean(p_etas[hold])) if y.size else None,
        }
        if row["eepas_brier"] is not None and row["etas_brier"] is not None:
            row["beats_etas"] = bool(row["eepas_brier"] < row["etas_brier"])
        else:
            row["beats_etas"] = False
        rows.append(row)
        if rid in issued and latest is not None and np.isfinite(p_eepas[latest]):
            p_now = float(np.clip(p_eepas[latest], 1e-6, 1.0 - 1e-6))
            week.append(
                {
                    "region": rid,
                    "label": region.get("label") or rid,
                    "magnitude_min": mag,
                    "horizon_days": HORIZON_DAYS,
                    "issue_day": days[latest].date().isoformat(),
                    "p": p_now,
                    "chance_plain": f"{int(round(p_now * 100))} in 100",
                    "writes_issued_p": False,
                    "sentence": (
                        "Medium-term catalog (EEPAS) · 90 days · "
                        "not the published 30-day number."
                    ),
                }
            )
    beats = sum(1 for r in rows if r.get("beats_etas"))
    report = {
        "writes_issued_p": False,
        "horizon_days": HORIZON_DAYS,
        "holdout_start": holdout_start,
        "n_holdout": int(hold.sum()),
        "n_beats_etas": beats,
        "regions": rows,
        "week": week,
        "card_label": "medium-term catalog (EEPAS) · 90 days · not the published 30-day number.",
        "scanned_at": iso(datetime.now(timezone.utc)),
        "note": (
            f"EEPAS 90d foil beat ETAS on {beats}/{len(rows)} targets. "
            "Not written to issued p."
        ),
    }
    dest = data_dir(root) / "eval" / "eepas.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["path"] = str(dest)
    return report
