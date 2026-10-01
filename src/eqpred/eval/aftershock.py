"""Aftershock (ETAS) track: what the clustering model says, scored on its own question.

This is a second opinion, not the product. The issued card is the fresh-shock
model in `eval/mainshock.py`. ETAS answers a different question — "will any
qualifying quake happen", aftershocks included — so it is scored here against
raw hits, not against the declustered first-shock target. Nothing here writes
issued p.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.eval.hunt import BOX_MAG, GLOBAL_MAG, _etas_p
from eqpred.eval.honest import week_mask
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.mainshock import ISSUED_TARGETS, _chance, _place, load_mainshock
from eqpred.eval.strategies import _window_hits
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc, utc_ts

MIN_SCORE = 40


def load_aftershock(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "aftershock.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _brier(prob: np.ndarray, hits: np.ndarray) -> float | None:
    ok = np.isfinite(prob) & np.isfinite(hits)
    if int(ok.sum()) < MIN_SCORE:
        return None
    return float(np.mean((prob[ok] - hits[ok]) ** 2))


def _params(root: Path | None) -> dict[str, Any]:
    path = Path(data_dir(root)) / "models" / "etas.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a missing fit is not fatal for the report
        return {}


def run_aftershock(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score the aftershock model on closed weeks and read off this week's chance."""
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
    train_mask = closed & (days <= utc_ts(train_end))
    hold_mask = closed & (days >= utc_ts(holdout_start))
    score_hold = week_mask(hold_mask)
    latest = int(np.argmax(days.to_numpy()))

    pack = load_mainshock(root) or {}
    issued = {row.get("region"): row for row in (pack.get("week") or [])}

    week: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for region_id, mag in ISSUED_TARGETS:
        region = REGIONS[region_id]
        etas = _etas_p(catalog, days, region_id, mag, root, versions=versions, horizon_days=HORIZON_DAYS)
        hits = _window_hits(catalog, region, mag, days, HORIZON_DAYS).astype(float)

        match = issued.get(region_id) or {}
        this_p = float(etas[latest]) if np.isfinite(etas[latest]) else None
        week.append(
            {
                "region": region_id,
                "label": _place(region_id),
                "magnitude_min": float(mag),
                "horizon_days": HORIZON_DAYS,
                "p": this_p,
                "chance_plain": _chance(this_p),
                "fresh_p": match.get("p"),
                "fresh_chance_plain": match.get("chance_plain"),
                "fresh_raw_p": match.get("fresh_raw_p"),
                "fresh_raw_chance_plain": match.get("fresh_raw_chance_plain"),
                "climatology_p": match.get("climatology_p"),
                "climatology_chance_plain": match.get("climatology_chance_plain"),
                "sentence": (
                    f"The aftershock model puts the chance of any magnitude {mag:g}+ quake "
                    f"{'anywhere on Earth' if region_id == 'global' else 'in ' + _place(region_id)} "
                    f"in the next {HORIZON_DAYS} days at {_chance(this_p)}. "
                    "Aftershocks count here. This is not the issued call."
                ),
            }
        )
        if this_p is not None:
            try:
                from eqpred.models.garnish import garnish_p, lam0_from_etas_p
                from eqpred.models.gr import load_gr

                gr = load_gr(root)
                lam0 = lam0_from_etas_p(
                    this_p,
                    mag,
                    float(gr.get("m0") or 4.5),
                    float(gr.get("b") or 1.0),
                    HORIZON_DAYS,
                )
                foil = garnish_p(
                    lam0,
                    feat.iloc[latest],
                    region_id,
                    mag=mag,
                    horizon_days=HORIZON_DAYS,
                    root=root,
                )
                week[-1]["foil_p"] = foil.get("garnish_p")
                week[-1]["foil_chance_plain"] = _chance(foil.get("garnish_p"))
                week[-1]["foil_gain"] = foil.get("garnish_gain")
            except Exception:  # noqa: BLE001 — foil is garnish, not the product
                pass

        base = float(hits[train_mask].mean()) if bool(train_mask.any()) else float("nan")
        etas_brier = _brier(etas[score_hold], hits[score_hold])
        climate_brier = _brier(np.full(int(score_hold.sum()), base), hits[score_hold])
        scored.append(
            {
                "region": region_id,
                "label": _place(region_id),
                "magnitude_min": float(mag),
                "n_train": int(train_mask.sum()),
                "n_score": int(score_hold.sum()),
                "base_rate_train": None if not np.isfinite(base) else base,
                "hit_rate_holdout": float(hits[score_hold].mean()) if bool(score_hold.any()) else None,
                "etas_brier": etas_brier,
                "climatology_brier": climate_brier,
                "beats_climatology": bool(
                    etas_brier is not None and climate_brier is not None and etas_brier < climate_brier
                ),
            }
        )

    report = {
        "writes_issued_p": False,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "horizon_days": HORIZON_DAYS,
        "issue_day": days[latest].date().isoformat(),
        "catalog_end": catalog_end.date().isoformat() if hasattr(catalog_end, "date") else str(catalog_end),
        "params": _params(root),
        "week": week,
        "scored": scored,
        "n_beats_climatology": sum(1 for row in scored if row["beats_climatology"]),
        "scanned_at": iso(datetime.now(timezone.utc)),
        "note": _english(scored),
    }
    dest = data_dir(root) / "eval" / "aftershock.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _english(scored: list[dict[str, Any]]) -> str:
    bits = [
        "The aftershock model asks a different question than the issued card: "
        "will any qualifying quake happen, aftershocks included. It is scored here "
        "against that question, so this is not a head-to-head with the fresh-shock model."
    ]
    usable = [r for r in scored if r.get("etas_brier") is not None and r.get("climatology_brier") is not None]
    if not usable:
        bits.append("Not enough closed weeks to score it yet.")
        return " ".join(bits)
    wins = [r for r in usable if r["beats_climatology"]]
    if wins:
        names = ", ".join(r["label"] for r in wins)
        bits.append(f"It beats a constant guess on {len(wins)} of {len(usable)} boxes: {names}.")
    else:
        bits.append("It does not beat a constant guess on any box yet.")
    best = min(usable, key=lambda r: r["etas_brier"] - r["climatology_brier"])
    bits.append(
        f"Best box is {best['label']} (error {best['etas_brier']:.4f} "
        f"vs constant {best['climatology_brier']:.4f})."
    )
    bits.append("Nothing here changes the issued card.")
    return " ".join(bits)
