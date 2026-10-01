from __future__ import annotations

"""Stress-test the M8 hint, and ask where — not whether.

The last model pass could not beat a constant guess on issued boxes.
Earth-wide M8 ranked a bit better. This probe checks whether that ranking
dies when non-catalog series are shifted, and whether the model can name the
box that gets the next fresh shock. Nothing here writes issued p.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eqpred.catalog import load_comcat
from eqpred.eval.brain import (
    MIN_SCORE,
    MIN_TRAIN,
    _alarm,
    _brier,
    _design,
    _driver_columns,
)
from eqpred.eval.mainshock import (
    HORIZON_DAYS,
    MAINSHOCK_LAGS,
    _fit_kind,
    _mix,
    _place,
    _pool_top,
    _proba,
    _usable_indices,
    fresh_hits,
)
from eqpred.models.predict import BOX_MAG
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc, utc_ts

SHIFT_DAYS = (0, 90, 180, 365)
ISSUED_WHERE = (
    ("hikurangi", float(BOX_MAG["hikurangi"])),
    ("alpine", float(BOX_MAG["alpine"])),
    ("kermadec", float(BOX_MAG["kermadec"])),
)
TRANSFER_WHERE = (
    ("japan", float(BOX_MAG["japan"])),
    ("chile", float(BOX_MAG["chile"])),
)


def load_probe(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "probe.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _shift_rows(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return x
    out = np.full(x.shape, np.nan, dtype=float)
    if n < len(x):
        out[n:] = x[:-n]
    return out


def _masks(
    days: pd.DatetimeIndex,
    catalog_end: pd.Timestamp,
    train_end: str,
    holdout_start: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closed = days + pd.Timedelta(days=horizon) <= catalog_end
    train_mask = closed & (days <= utc_ts(train_end))
    hold_mask = closed & (days >= utc_ts(holdout_start))
    fit_mask = train_mask.copy()
    val_mask = np.zeros(len(train_mask), dtype=bool)
    idx = np.flatnonzero(train_mask)
    if idx.size > MIN_TRAIN + 200:
        val_mask[idx[-365:]] = True
        fit_mask[idx[-365:]] = False
    return fit_mask, val_mask, hold_mask


def _thin_p(
    raw_x: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, str, list[int]]:
    climate = float(y[fit_mask].mean()) if fit_mask.any() else float(y.mean())
    climate_full = float(y[fit_mask | val_mask].mean()) if (fit_mask | val_mask).any() else climate
    usable = _usable_indices(raw_x, fit_mask)
    cols = _pool_top(raw_x, y, fit_mask, usable, 48)
    if len(cols) < 2 or int(fit_mask.sum()) < MIN_TRAIN or int(np.unique(y[fit_mask]).size) < 2:
        return np.full(len(raw_x), climate_full), "climate|mix0.00", []
    kw = {"max_depth": 3, "min_samples_leaf": 30}
    model = _fit_kind("tree", raw_x[np.ix_(fit_mask, cols)], y[fit_mask], kw)
    raw_p = _proba(model, raw_x[:, cols])
    score_mask = val_mask if val_mask.any() else fit_mask
    mix_a, _ = _mix(raw_p, y, climate, score_mask)
    train_both = fit_mask | val_mask
    model = _fit_kind("tree", raw_x[np.ix_(train_both, cols)], y[train_both], kw)
    raw_p = _proba(model, raw_x[:, cols])
    p = np.clip((1.0 - mix_a) * climate_full + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
    return p, f"fresh_thin|mix{mix_a:.2f}", cols


def _score_binary(
    p: np.ndarray,
    y: np.ndarray,
    days: pd.DatetimeIndex,
    fit_mask: np.ndarray,
    hold_mask: np.ndarray,
    recipe: str,
) -> dict[str, Any]:
    climate = float(y[fit_mask].mean()) if fit_mask.any() else float(y.mean())
    alarm = _alarm(p, y.astype(bool), days, fit_mask, hold_mask)
    return {
        "recipe": recipe,
        "brier": _brier(p[hold_mask], y[hold_mask].astype(float)),
        "climatology_brier": _brier(np.full(int(hold_mask.sum()), climate), y[hold_mask].astype(float)),
        "climatology_p": climate,
        **alarm,
    }


def probe_m8_shifts(
    raw_x: np.ndarray,
    y: np.ndarray,
    days: pd.DatetimeIndex,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    hold_mask: np.ndarray,
) -> dict[str, Any]:
    train_for_alarm = fit_mask | val_mask
    rows: list[dict[str, Any]] = []
    aligned_lift = None
    for shift in SHIFT_DAYS:
        x = _shift_rows(raw_x, shift)
        p, recipe, _cols = _thin_p(x, y, fit_mask, val_mask)
        scored = _score_binary(p, y, days, train_for_alarm, hold_mask, recipe)
        scored["shift_days"] = shift
        if shift == 0:
            aligned_lift = scored.get("lift")
        rows.append(scored)
    years: list[dict[str, Any]] = []
    p0, _recipe, _ = _thin_p(raw_x, y, fit_mask, val_mask)
    hold_years = sorted({int(y) for y in days[hold_mask].year})
    for year in hold_years:
        year_mask = hold_mask & (days.year == year)
        if int(year_mask.sum()) < MIN_SCORE:
            continue
        years.append(
            {
                "year": year,
                **_score_binary(p0, y, days, train_for_alarm, year_mask, "fresh_thin"),
            }
        )
    shifted = [r for r in rows if r["shift_days"] > 0]
    aligned = next((r for r in rows if r["shift_days"] == 0), None)
    selective = [
        r
        for r in shifted
        if r.get("lift") is not None and (r.get("time_in_alarm") or 1.0) < 0.4
    ]
    best_shift = max(selective, key=lambda r: float(r.get("lift") or -1.0), default=None)
    leak = False
    if aligned and best_shift and aligned.get("lift") is not None:
        leak = float(best_shift["lift"]) >= float(aligned["lift"]) - 0.005
    return {
        "id": "m8_timeshift",
        "aligned_lift": aligned_lift,
        "looks_like_leakage": leak,
        "shifts": rows,
        "years": years,
    }


def _where_hits(
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    boxes: tuple[tuple[str, float], ...],
    horizon: int,
) -> tuple[list[str], np.ndarray]:
    names = [rid for rid, _mag in boxes]
    mat = np.column_stack(
        [fresh_hits(catalog, REGIONS[rid], mag, days, horizon) for rid, mag in boxes]
    )
    return names, mat


def _where_cols(
    raw_x: np.ndarray,
    names: list[str],
    y_box: np.ndarray,
    fit_mask: np.ndarray,
    box_ids: list[str],
) -> list[int]:
    usable = _usable_indices(raw_x, fit_mask)
    picked: list[int] = []
    for b, rid in enumerate(box_ids):
        local = [
            i
            for i in usable
            if rid in names[i].split("__lag")[0]
            or names[i].split("__lag")[0].startswith(("planet_", "lunar_", "tidal_", "climate_", "tec_tecu", "kp", "dst_", "imf_", "spf_"))
        ]
        picked.extend(_pool_top(raw_x, y_box[:, b], fit_mask, local or usable, 20))
    # unique, stable
    seen: set[int] = set()
    out: list[int] = []
    for i in picked:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:80]


def _where_pack(
    raw_x: np.ndarray,
    feat_names: list[str],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    boxes: tuple[tuple[str, float], ...],
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    hold_mask: np.ndarray,
    label: str,
) -> dict[str, Any]:
    box_ids, hits = _where_hits(catalog, days, boxes, HORIZON_DAYS)
    n_hit = hits.sum(axis=1)
    single = n_hit == 1
    y = hits.argmax(axis=1)
    train_single = (fit_mask | val_mask) & single
    hold_single = hold_mask & single
    n_train = int(train_single.sum())
    n_hold = int(hold_single.sum())
    counts_train = {box_ids[i]: int((y[train_single] == i).sum()) for i in range(len(box_ids))}
    majority = max(counts_train, key=counts_train.get) if n_train else box_ids[0]
    majority_idx = box_ids.index(majority)
    if n_train < MIN_TRAIN or n_hold < MIN_SCORE or len(set(y[train_single].tolist())) < 2:
        return {
            "id": label,
            "boxes": box_ids,
            "n_train": n_train,
            "n_hold": n_hold,
            "majority": majority,
            "fresh_acc": None,
            "majority_acc": None,
            "beats_majority": False,
            "note": "Not enough single-box fresh hits to ask where.",
        }
    cols = _where_cols(raw_x, feat_names, hits, train_single, box_ids)
    if len(cols) < 2:
        return {
            "id": label,
            "boxes": box_ids,
            "n_train": n_train,
            "n_hold": n_hold,
            "majority": majority,
            "fresh_acc": None,
            "majority_acc": float((y[hold_single] == majority_idx).mean()),
            "beats_majority": False,
            "note": "No usable non-catalog columns for the where question.",
        }
    model = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=0.25, max_iter=500, solver="lbfgs")),
        ]
    )
    model.fit(raw_x[np.ix_(train_single, cols)], y[train_single])
    pred = model.predict(raw_x[np.ix_(hold_single, cols)])
    fresh_acc = float((pred == y[hold_single]).mean())
    majority_acc = float((y[hold_single] == majority_idx).mean())
    per_box = []
    for i, rid in enumerate(box_ids):
        true_i = y[hold_single] == i
        n = int(true_i.sum())
        hit = float((pred[true_i] == i).mean()) if n else None
        per_box.append(
            {
                "region": rid,
                "label": _place(rid),
                "n_hold": n,
                "recall": hit,
            }
        )
    return {
        "id": label,
        "boxes": box_ids,
        "n_train": n_train,
        "n_hold": n_hold,
        "majority": majority,
        "majority_label": _place(majority),
        "fresh_acc": fresh_acc,
        "majority_acc": majority_acc,
        "beats_majority": fresh_acc > majority_acc + 1e-12,
        "per_box": per_box,
        "issued": label == "where_issued",
        "note": (
            f"On later mornings with a fresh hit in exactly one box, the model named the box "
            f"{fresh_acc * 100:.1f}% of the time. Always guessing {_place(majority)} gets "
            f"{majority_acc * 100:.1f}%."
        ),
    }


def _english(report: dict[str, Any]) -> str:
    bits = [
        "The probe asked two different questions: is the Earth-wide M8 ranking real, "
        "and can the model name the box that gets the next fresh shock."
    ]
    m8 = report.get("m8") or {}
    aligned = next((r for r in m8.get("shifts") or [] if r.get("shift_days") == 0), None)
    if aligned and aligned.get("lift") is not None:
        bits.append(
            f"Aligned M8 lift was {aligned['lift'] * 100:+.1f} pt on later years."
        )
    if m8.get("looks_like_leakage"):
        shifted = [
            r
            for r in m8.get("shifts") or []
            if r.get("shift_days") and (r.get("time_in_alarm") or 1) < 0.4
        ]
        best = max(shifted, key=lambda r: r.get("lift") or -1.0, default=None)
        if best:
            bits.append(
                f"A {best['shift_days']}-day shift still ranked "
                f"{(best.get('lift') or 0) * 100:+.1f} pt — a yearly cycle, not a forecast."
            )
        else:
            bits.append("A time-shift kept similar ranking — season or leakage, not a forecast.")
    years = [y for y in m8.get("years") or [] if y.get("lift") is not None]
    loud = [y for y in years if float(y.get("lift") or 0) > 0.01]
    if loud:
        bits.append(
            "The aligned M8 lift is concentrated in "
            + ", ".join(f"{y['year']} ({y['lift'] * 100:+.1f} pt)" for y in loud)
            + "."
        )
    for key in ("where_issued", "where_transfer"):
        pack = report.get(key) or {}
        if pack.get("fresh_acc") is None:
            continue
        beat = "beat" if pack.get("beats_majority") else "did not beat"
        bits.append(
            f"{'Issued NZ boxes' if key == 'where_issued' else 'Japan vs Chile (practice)'} "
            f"{beat} always-guess-{pack.get('majority_label') or pack.get('majority')} "
            f"({(pack.get('fresh_acc') or 0) * 100:.1f}% vs {(pack.get('majority_acc') or 0) * 100:.1f}%)."
        )
    bits.append("Nothing here writes issued p or this week's model number.")
    return " ".join(bits)


def run_probe(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = feat.copy()
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog = quakes if quakes is not None else load_comcat(root)
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    columns = _driver_columns(feat, unconventional_only=True)
    raw_x, names = _design(feat, columns, MAINSHOCK_LAGS)
    fit_mask, val_mask, hold_mask = _masks(days, catalog_end, train_end, holdout_start, HORIZON_DAYS)
    y_m8 = fresh_hits(catalog, REGIONS["global"], 8.0, days, HORIZON_DAYS).astype(int)
    report = {
        "scanned_at": iso(datetime.now(timezone.utc)),
        "train_end": train_end,
        "holdout_start": holdout_start,
        "writes_issued_p": False,
        "n_features": int(raw_x.shape[1]),
        "m8": probe_m8_shifts(raw_x, y_m8, days, fit_mask, val_mask, hold_mask),
        "where_issued": _where_pack(
            raw_x, names, catalog, days, ISSUED_WHERE, fit_mask, val_mask, hold_mask, "where_issued"
        ),
        "where_transfer": _where_pack(
            raw_x, names, catalog, days, TRANSFER_WHERE, fit_mask, val_mask, hold_mask, "where_transfer"
        ),
    }
    report["note"] = _english(report)
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "probe.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
