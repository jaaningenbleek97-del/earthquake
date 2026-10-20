from __future__ import annotations

"""Climate + schedule foil + leftover of every other source.

This is a research extra. It never writes issued p, never replaces a model,
and never drops an ingested source from the official stack. Climate and the
schedule (season, tides, load, catalog rate) are peeled off first; then a
ridge looks at the remaining unconventional columns.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eqpred.catalog import load_comcat
from eqpred.eval.brain import MIN_TRAIN, _brier, _design, _driver_columns, _impute
from eqpred.eval.decluster import fresh_hits
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.honest import purged_train_hold, rolling_climate_series
from eqpred.eval.hunt import driver_family
from eqpred.eval.mainshock import ISSUED_TARGETS, _chance
from eqpred.eval.skill import skill_beats_climate
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc

WRITES_ISSUED_P = False
LEFTOVER_ID = "fresh.leftover.v1"
LEFTOVER_LAGS = (0, 7, 14, 30)
SCHEDULE_LAGS = (0,)
RIDGE_SCHEDULE = 4.0
RIDGE_LEFTOVER = 12.0

SCHEDULE_PREFIXES = (
    "sse_",
    "tide_",
    "era5_",
    "hydro_",
    "smap_",
    "rate_roll_",
    "cal_doy",
    "n_fresh_",
    "days_since_fresh_",
)
FOIL_EXACT = frozenset({"cal_weekday", "cal_payday", "cal_weekend"})
FOIL_PREFIXES = ("wiki_", "vlf_", "tir_")


def is_schedule_column(name: str) -> bool:
    raw = str(name).split("__lag", 1)[0].lower()
    return raw.startswith(SCHEDULE_PREFIXES)


def is_foil_column(name: str) -> bool:
    raw = str(name).split("__lag", 1)[0].lower()
    return raw in FOIL_EXACT or raw.startswith(FOIL_PREFIXES) or driver_family(raw) == "foil"


def leftover_columns(features: pd.DataFrame) -> list[str]:
    """Every unconventional source except schedule and circus foils."""
    out: list[str] = []
    for col in _driver_columns(features, unconventional_only=True):
        if is_schedule_column(col) or is_foil_column(col):
            continue
        out.append(col)
    return out


def schedule_columns(features: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in features.columns:
        if col in {"issue_time"} or not pd.api.types.is_numeric_dtype(features[col]):
            continue
        if is_schedule_column(str(col)):
            out.append(str(col))
    return out


def ensure_schedule(feat: pd.DataFrame) -> pd.DataFrame:
    work = feat.copy()
    days = pd.DatetimeIndex(parse_utc(work["issue_time"]))
    if "cal_doy_sin" not in work.columns:
        doy = days.dayofyear.to_numpy()
        work["cal_doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        work["cal_doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    return work


def _sse_two_rate(
    work: pd.DataFrame,
    hits: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Train-only SSE-on vs SSE-off first-shock rates. Observed slip only."""
    if "sse_on" in work.columns:
        raw = pd.to_numeric(work["sse_on"], errors="coerce").to_numpy(dtype=float)
    elif "sse_hikurangi" in work.columns:
        raw = pd.to_numeric(work["sse_hikurangi"], errors="coerce").to_numpy(dtype=float)
    else:
        return None
    flag = np.isfinite(raw) & (raw > 0.5)
    on = np.asarray(train_mask, dtype=bool) & flag
    off = np.asarray(train_mask, dtype=bool) & ~flag
    if int(on.sum()) < 20 or int(off.sum()) < 20:
        return None
    rate_on = float(np.clip(hits[on].mean(), 1e-6, 1.0 - 1e-6))
    rate_off = float(np.clip(hits[off].mean(), 1e-6, 1.0 - 1e-6))
    regime = np.where(flag, rate_on, rate_off).astype(float)
    return regime, rate_on, rate_off


def _ready_climate(hits: np.ndarray, days: pd.DatetimeIndex, train_mask: np.ndarray) -> np.ndarray:
    climate = rolling_climate_series(np.asarray(hits, dtype=float), days)
    fill = float(np.nanmean(np.asarray(hits, dtype=float)[np.asarray(train_mask, dtype=bool)]))
    if not np.isfinite(fill):
        fill = 0.0
    return np.where(np.isfinite(climate), climate, fill)


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Ridge | None:
    if x_train.size == 0 or x_train.shape[0] < MIN_TRAIN or x_train.shape[1] < 1:
        return None
    if float(np.nanstd(y_train)) == 0:
        return None
    model = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    return model


def _predict(model, x_all: np.ndarray, n: int) -> np.ndarray:
    if model is None or x_all.size == 0:
        return np.zeros(n, dtype=float)
    return np.asarray(model.predict(x_all), dtype=float)


def _logistic_prob(x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray, n: int) -> np.ndarray:
    """The model's own probability from the columns that actually vary with the outcome.

    Using every column lets the fit sit on 0 or 1. That ceiling is not a prediction.
    """
    y = np.asarray(y_train, dtype=float)
    fallback = float(np.nanmean(y)) if y.size else 0.5
    if not np.isfinite(fallback):
        fallback = 0.5
    out = np.full(n, np.clip(fallback, 1e-6, 1.0 - 1e-6))
    if x_train.size == 0 or x_train.shape[0] < MIN_TRAIN or x_train.shape[1] < 1:
        return out
    labels = y[np.isfinite(y)]
    if labels.size < MIN_TRAIN or np.unique(labels.astype(int)).size < 2:
        return out
    y_center = y - float(np.nanmean(y))
    centered = x_train - np.nanmean(x_train, axis=0)
    denom = np.linalg.norm(centered, axis=0) * float(np.linalg.norm(y_center)) + 1e-12
    strength = np.abs(centered.T @ y_center) / denom
    keep = np.argsort(strength)[-min(40, x_train.shape[1]) :]
    keep = keep[strength[keep] > 0]
    if keep.size == 0:
        return out
    model = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(C=0.2, max_iter=500, solver="lbfgs"),
    )
    model.fit(x_train[:, keep], y.astype(int))
    proba = model.predict_proba(x_all[:, keep])
    classes = list(model.named_steps["logisticregression"].classes_)
    if 1 not in classes:
        return out
    return np.clip(proba[:, classes.index(1)], 1e-6, 1.0 - 1e-6)


def _bound_residual(hat: np.ndarray) -> np.ndarray:
    """A hit-minus-probability residual cannot leave [-1, 1].

    The ridge will extrapolate far outside that on a wide column set.
    Those values are what saturated the issued encoder.
    """
    return np.clip(np.asarray(hat, dtype=float), -1.0, 1.0)


def fit_leftover_region(
    feat: pd.DataFrame,
    hits: np.ndarray,
    *,
    region_id: str,
    mag: float,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    leftover_cols: list[str] | None = None,
    schedule_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Peel climate, then schedule, then leftover. Does not write issued p."""
    work = ensure_schedule(feat)
    days = pd.DatetimeIndex(parse_utc(work["issue_time"]))
    hits = np.asarray(hits, dtype=float)
    sched_cols = schedule_cols if schedule_cols is not None else schedule_columns(work)
    left_cols = leftover_cols if leftover_cols is not None else leftover_columns(work)
    climate = _ready_climate(hits, days, train_mask)
    regime = climate
    regime_kind = "climate"
    sse_rate_on = None
    sse_rate_off = None
    if region_id == "hikurangi":
        clock = _sse_two_rate(work, hits, train_mask)
        if clock is not None:
            regime, sse_rate_on, sse_rate_off = clock
            regime_kind = "sse_two_rate"
            sched_cols = [
                col
                for col in sched_cols
                if not str(col).split("__lag", 1)[0].lower().startswith("sse_")
            ]
    sched_x, _sched_names = _design(work, sched_cols, lags=SCHEDULE_LAGS)
    left_x, left_names = _design(work, left_cols, lags=LEFTOVER_LAGS)
    sched_train, sched_all = _impute(sched_x[train_mask] if sched_x.size else sched_x, sched_x)
    left_train, left_all = _impute(left_x[train_mask] if left_x.size else left_x, left_x)

    schedule_p = _logistic_prob(sched_train, hits[train_mask], sched_all, len(hits))
    y_leftover = hits - schedule_p
    left_model = _fit_ridge(left_train, y_leftover[train_mask], RIDGE_LEFTOVER)
    leftover_hat = _bound_residual(_predict(left_model, left_all, len(hits)))
    leftover_p = _logistic_prob(left_train, hits[train_mask], left_all, len(hits))

    climate_brier = _brier(climate[hold_mask], hits[hold_mask])
    regime_brier = _brier(regime[hold_mask], hits[hold_mask])
    schedule_brier = _brier(schedule_p[hold_mask], hits[hold_mask])
    leftover_brier = _brier(leftover_p[hold_mask], hits[hold_mask])
    vs_regime = skill_beats_climate(
        leftover_p[hold_mask], hits[hold_mask], regime[hold_mask], train_p=leftover_p[train_mask]
    )
    vs_schedule = skill_beats_climate(
        leftover_p[hold_mask],
        hits[hold_mask],
        schedule_p[hold_mask],
        train_p=leftover_p[train_mask],
    )
    beats_climate = bool(vs_regime["beats"])
    beats_schedule = bool(vs_schedule["beats"])
    last = int(np.flatnonzero(np.isfinite(leftover_p))[-1]) if np.isfinite(leftover_p).any() else len(hits) - 1
    issue_day = days[last].date().isoformat()
    this_p = float(leftover_p[last])
    this_climate = float(climate[last])
    this_schedule = float(schedule_p[last])
    this_delta = float(leftover_hat[last])
    place = "anywhere on Earth" if region_id == "global" else f"in {REGIONS[region_id]['label']}"
    sentence = (
        f"The leftover extra (climate + schedule + every other source) puts the chance of a "
        f"fresh magnitude {mag:g}+ quake {place} in the next {HORIZON_DAYS} days at {_chance(this_p)}. "
        f"The rolling five-year rate is {_chance(this_climate)}; after the schedule foil it is "
        f"{_chance(this_schedule)}. This is not the published number."
    )
    return {
        "writes_issued_p": False,
        "region": region_id,
        "label": REGIONS[region_id]["label"] if region_id in REGIONS else region_id,
        "magnitude_min": mag,
        "n_leftover_columns": len(left_names),
        "n_schedule_columns": len(sched_cols),
        "climate_brier": climate_brier,
        "regime_brier": regime_brier,
        "regime_kind": regime_kind,
        "sse_rate_on": sse_rate_on,
        "sse_rate_off": sse_rate_off,
        "information_gain": vs_regime.get("information_gain"),
        "schedule_brier": schedule_brier,
        "leftover_brier": leftover_brier,
        "beats_climate": beats_climate,
        "beats_schedule": beats_schedule,
        "leftover_hat": leftover_hat,
        "leftover_p": leftover_p,
        "schedule_p": schedule_p,
        "climate": climate,
        "regime": regime,
        "this_week": {
            "region": region_id,
            "label": REGIONS[region_id]["label"] if region_id in REGIONS else region_id,
            "magnitude_min": mag,
            "p": this_p,
            "leftover_p": this_p,
            "schedule_p": this_schedule,
            "climatology_p": this_climate,
            "leftover_delta": this_delta,
            "chance_plain": _chance(this_p),
            "leftover_chance_plain": _chance(this_p),
            "schedule_chance_plain": _chance(this_schedule),
            "climatology_chance_plain": _chance(this_climate),
            "issue_day": issue_day,
            "sentence": sentence,
            "writes_issued_p": False,
        },
    }


def load_leftover(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "leftover.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def attach_leftover_sidecar(report: dict[str, Any], root: Path | None = None) -> bool:
    """Merge leftover onto an existing mainshock.json without hollowing issued cards."""
    path = data_dir(root) / "eval" / "mainshock.json"
    if not path.exists():
        return False
    pack = json.loads(path.read_text(encoding="utf-8"))
    pack["leftover"] = _sidecar_view(report)
    path.write_text(json.dumps(pack, indent=2, default=str), encoding="utf-8")
    return True


def _sidecar_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": LEFTOVER_ID,
        "mode": "leftover",
        "writes_issued_p": False,
        "horizon_days": report.get("horizon_days", HORIZON_DAYS),
        "note": report.get("note"),
        "n_leftover_columns": report.get("n_leftover_columns"),
        "n_schedule_columns": report.get("n_schedule_columns"),
        "n_beats_climate": report.get("n_beats_climate"),
        "n_beats_schedule": report.get("n_beats_schedule"),
        "week": report.get("week") or [],
        "regions": [
            {k: r.get(k) for k in (
                "region",
                "label",
                "magnitude_min",
                "n_leftover_columns",
                "n_schedule_columns",
                "climate_brier",
                "regime_brier",
                "regime_kind",
                "sse_rate_on",
                "sse_rate_off",
                "information_gain",
                "schedule_brier",
                "leftover_brier",
                "beats_climate",
                "beats_schedule",
                "this_week",
                "writes_issued_p",
            )}
            for r in report.get("regions") or []
        ],
        "scanned_at": report.get("scanned_at"),
        "issue_day": report.get("issue_day"),
    }


def _english(report: dict[str, Any]) -> str:
    n_box = len(report.get("regions") or [])
    n_sched = int(report.get("n_beats_schedule") or 0)
    n_clim = int(report.get("n_beats_climate") or 0)
    n_feat = int(report.get("n_leftover_columns") or 0)
    return (
        f"Leftover extra: climate and the schedule foil first, then {n_feat} other-source "
        f"columns. It beat the schedule foil in {n_sched} of {n_box} boxes and the rolling "
        f"five-year rate in {n_clim} of {n_box}. It never writes the published number."
    )


def run_leftover(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
    targets: tuple[tuple[str, float], ...] = ISSUED_TARGETS,
    attach: bool = True,
) -> dict[str, Any]:
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    empty = {
        "writes_issued_p": False,
        "id": LEFTOVER_ID,
        "mode": "leftover",
        "horizon_days": HORIZON_DAYS,
        "regions": [],
        "week": [],
        "n_beats_climate": 0,
        "n_beats_schedule": 0,
        "n_leftover_columns": 0,
        "n_schedule_columns": 0,
        "note": "Leftover extra needs a daily feature stack. It never writes the published number.",
        "scanned_at": iso(datetime.now(timezone.utc)),
        "attached_to_mainshock": False,
    }
    if features is None:
        path = feature_dir(root) / "global_daily.parquet"
        if not path.exists():
            (dest / "leftover.json").write_text(json.dumps(empty, indent=2, default=str), encoding="utf-8")
            return empty
        feat = pd.read_parquet(path)
    else:
        feat = features.copy()
    if feat.empty or "issue_time" not in feat.columns:
        (dest / "leftover.json").write_text(json.dumps(empty, indent=2, default=str), encoding="utf-8")
        return empty
    feat = ensure_schedule(feat)
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog = quakes.copy() if quakes is not None else load_comcat(root)
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    if "available_at" in catalog.columns:
        catalog["available_at"] = parse_utc(catalog["available_at"])
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    left_cols = leftover_columns(feat)
    sched_cols = schedule_columns(feat)
    regions: list[dict[str, Any]] = []
    week: list[dict[str, Any]] = []
    hats: dict[str, np.ndarray] = {}
    for region_id, mag in targets:
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, HORIZON_DAYS).astype(float)
        packed = fit_leftover_region(
            feat,
            hits,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask,
            hold_mask=hold_mask,
            leftover_cols=left_cols,
            schedule_cols=sched_cols,
        )
        hats[region_id] = packed.pop("leftover_hat")
        packed.pop("leftover_p", None)
        packed.pop("schedule_p", None)
        packed.pop("climate", None)
        packed.pop("regime", None)
        regions.append(packed)
        week.append(packed["this_week"])
    issue_day = week[-1]["issue_day"] if week else None
    report = {
        "writes_issued_p": False,
        "id": LEFTOVER_ID,
        "mode": "leftover",
        "horizon_days": HORIZON_DAYS,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "n_leftover_columns": len(left_cols),
        "n_schedule_columns": len(sched_cols),
        "n_beats_climate": sum(1 for r in regions if r.get("beats_climate")),
        "n_beats_schedule": sum(1 for r in regions if r.get("beats_schedule")),
        "regions": regions,
        "week": week,
        "issue_day": issue_day,
        "scanned_at": iso(datetime.now(timezone.utc)),
        "looks_at_all_sources": True,
        "held_to_climatology": True,
    }
    report["note"] = _english(report)
    (dest / "leftover.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["attached_to_mainshock"] = attach_leftover_sidecar(report, root) if attach else False
    report["_hats"] = hats
    return report
