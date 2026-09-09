from __future__ import annotations

"""Catalog precursor: foreshock → fresh mainshock. Never leftover. Never issued p.

Looks at small and moderate quakes known that morning, in a box that has
not just had a large shock. The target is a declustered first large shock,
not the swarm. Train on old years. Score later closed weeks.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import apply_completeness, event_available_as, load_comcat, load_comcat_versions
from eqpred.eval.brain import MIN_SCORE, MIN_TRAIN, _alarm, _brier, _design
from eqpred.eval.honest import fit_val_split, week_mask
from eqpred.eval.hunt import driver_family
from eqpred.eval.mainshock import (
    HORIZON_DAYS,
    ISSUED_TARGETS,
    THIS_WEEK_BRIER_FLOOR,
    _chance,
    _fit_kind,
    _mix,
    _place,
    _pool_top,
    _proba,
    _top_columns,
    _usable_indices,
    fresh_hits,
    leftover_beats_climatology,
    leftover_this_week_held,
    prior_quiet,
)
from eqpred.eval.strategies import _window_hits
from eqpred.ingest.run import try_latest_raw
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS, in_bbox
from eqpred.stamp import parse_utc, utc_ts

FORESHOCK_LAGS = (0, 1, 2, 3, 7)
QUIET_LOOKBACK = 10
RECIPE_PREFIX = "catalog_foreshock"

DRIVER_PLAIN = {
    "n_m4_1d": "M4+ quakes in the last day",
    "n_m4_3d": "M4+ quakes in the last 3 days",
    "n_m4_7d": "M4+ quakes in the last week",
    "n_m45_1d": "M4.5+ quakes in the last day",
    "n_m45_3d": "M4.5+ quakes in the last 3 days",
    "n_m45_7d": "M4.5+ quakes in the last week",
    "n_m5_3d": "M5+ quakes in the last 3 days",
    "n_m5_7d": "M5+ quakes in the last week",
    "n_m6_7d": "M6+ quakes in the last week",
    "days_since_m7": "days since a magnitude 7",
    "n_m4_7d_after_quiet": "M4+ this week after a quiet spell",
    "n_geonet_m3_7d": "GeoNet M3+ in the last week",
}


def _driver_plain(name: str) -> str:
    base, _, lag = name.partition("__lag")
    label = DRIVER_PLAIN.get(base, base.replace("_", " "))
    if lag:
        return f"{label}, {lag} days earlier"
    return label


def _count_available(
    quakes: pd.DataFrame,
    issue: pd.Series,
    minmag: float,
    days: int,
    versions: pd.DataFrame | None = None,
    region: dict | None = None,
) -> np.ndarray:
    work = apply_completeness(quakes)
    if region and region.get("kind") == "bbox":
        work = work.loc[in_bbox(work, region)]
    if work.empty:
        return np.zeros(len(issue), dtype=int)
    if versions is not None and not versions.empty:
        stamps = event_available_as(work, minmag, versions)
    else:
        stamps = work.loc[work["magnitude"] >= minmag, "available_at"]
    times = np.sort(parse_utc(stamps.dropna()).to_numpy(dtype="datetime64[ns]"))
    if times.size == 0:
        return np.zeros(len(issue), dtype=int)
    right = np.searchsorted(times, issue.to_numpy(dtype="datetime64[ns]"), side="right")
    left = np.searchsorted(
        times,
        (issue - pd.Timedelta(days=days)).to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    return (right - left).astype(int)


def _days_since(
    quakes: pd.DataFrame,
    issue: pd.Series,
    minmag: float,
    versions: pd.DataFrame | None = None,
    region: dict | None = None,
) -> np.ndarray:
    work = apply_completeness(quakes)
    if region and region.get("kind") == "bbox":
        work = work.loc[in_bbox(work, region)]
    if work.empty:
        return np.full(len(issue), np.nan)
    if versions is not None and not versions.empty:
        stamps = event_available_as(work, minmag, versions)
    else:
        stamps = work.loc[work["magnitude"] >= minmag, "available_at"]
    times = np.sort(parse_utc(stamps.dropna()).to_numpy(dtype="datetime64[ns]"))
    if times.size == 0:
        return np.full(len(issue), np.nan)
    idx = np.searchsorted(times, issue.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    out = np.full(len(issue), np.nan)
    ok = idx >= 0
    delta = issue.to_numpy(dtype="datetime64[ns]")[ok] - times[idx[ok]]
    out[ok] = delta / np.timedelta64(1, "D")
    return out


def catalog_foreshock_frame(
    quakes: pd.DataFrame,
    issue: pd.Series,
    versions: pd.DataFrame | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """As-of catalog counts. Names stay in the catalog family (n_m*, days_since_*)."""
    feat = pd.DataFrame({"issue_time": parse_utc(issue)})
    work = quakes.copy()
    work["available_at"] = parse_utc(work["available_at"])
    specs = (
        (4.0, 1, "n_m4_1d"),
        (4.0, 3, "n_m4_3d"),
        (4.0, 7, "n_m4_7d"),
        (4.5, 1, "n_m45_1d"),
        (4.5, 3, "n_m45_3d"),
        (4.5, 7, "n_m45_7d"),
        (5.0, 3, "n_m5_3d"),
        (5.0, 7, "n_m5_7d"),
        (6.0, 7, "n_m6_7d"),
    )
    for mag, window, name in specs:
        feat[name] = _count_available(work, feat["issue_time"], mag, window, versions)
    feat["days_since_m7"] = _days_since(work, feat["issue_time"], 7.0, versions)
    quiet = pd.to_numeric(feat["days_since_m7"], errors="coerce").fillna(0) >= QUIET_LOOKBACK
    feat["n_m4_7d_after_quiet"] = np.where(quiet, feat["n_m4_7d"], 0)
    for region_id, floor in ISSUED_TARGETS:
        if region_id == "global":
            continue
        region = REGIONS[region_id]
        slug = region_id
        feat[f"n_m4_3d_{slug}"] = _count_available(work, feat["issue_time"], 4.0, 3, versions, region)
        feat[f"n_m4_7d_{slug}"] = _count_available(work, feat["issue_time"], 4.0, 7, versions, region)
        feat[f"n_m5_7d_{slug}"] = _count_available(work, feat["issue_time"], 5.0, 7, versions, region)
        feat[f"days_since_m{int(floor)}_{slug}"] = _days_since(
            work, feat["issue_time"], float(floor), versions, region
        )
        box_quiet = (
            pd.to_numeric(feat[f"days_since_m{int(floor)}_{slug}"], errors="coerce").fillna(0)
            >= QUIET_LOOKBACK
        )
        feat[f"n_m4_7d_{slug}_after_quiet"] = np.where(box_quiet, feat[f"n_m4_7d_{slug}"], 0)
        DRIVER_PLAIN.setdefault(f"n_m4_3d_{slug}", f"M4+ in { _place(slug) } last 3 days")
        DRIVER_PLAIN.setdefault(f"n_m4_7d_{slug}", f"M4+ in { _place(slug) } last week")
        DRIVER_PLAIN.setdefault(f"n_m5_7d_{slug}", f"M5+ in { _place(slug) } last week")
        DRIVER_PLAIN.setdefault(
            f"days_since_m{int(floor)}_{slug}",
            f"days since M{int(floor)}+ in { _place(slug) }",
        )
        DRIVER_PLAIN.setdefault(
            f"n_m4_7d_{slug}_after_quiet",
            f"M4+ in { _place(slug) } this week after a quiet spell",
        )
    geonet = try_latest_raw("geonet.fdsn", root)
    if not geonet.empty and "available_at" in geonet.columns:
        overlay = geonet.copy()
        overlay["available_at"] = parse_utc(overlay["available_at"])
        feat["n_geonet_m3_7d"] = _count_available(overlay, feat["issue_time"], 3.0, 7)
    return feat


def catalog_columns(frame: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in frame.columns:
        if col in {"issue_time"}:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if driver_family(col) != "catalog":
            continue
        out.append(str(col))
    return out


def read_foreshock(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "foreshock.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_foreshock(root: Path | None = None) -> dict[str, Any] | None:
    pack = read_foreshock(root)
    if not pack:
        return None
    return _seal_report(pack)


def foreshock_week_copy(pack: dict[str, Any] | None) -> dict[str, Any]:
    held = True
    if pack:
        week = [w for w in (pack.get("week") or []) if w.get("region") in {"global", "hikurangi", "alpine", "kermadec"}]
        held = not week or all(bool(w.get("held_to_climatology")) for w in week)
    if held:
        return {
            "held": True,
            "title": "This week’s catalog precursor attempt",
            "lede": (
                "This track looks at small and moderate quakes known that morning, "
                "in a box that has not just had a large shock, and calls a fresh mainshock. "
                "It has not beaten a constant guess on later years, so the historical rate "
                "is still the number we trust. Official ledger cards stay calibrated ETAS. "
                "This is a catalog precursor, not leftover."
            ),
            "method": (
                "The mapping from recent small quakes to next week’s first large shock is "
                "learned on old years and scored on later closed weeks. Cards show the call "
                "even when it loses. It is not a public warning."
            ),
        }
    return {
        "held": False,
        "title": "This week’s catalog precursor call",
        "lede": (
            "This track looks at small and moderate quakes known that morning, "
            "in a box that has not just had a large shock, and calls a fresh mainshock. "
            "Official ledger cards stay calibrated ETAS until we promote on purpose."
        ),
        "method": (
            "The mapping from recent small quakes to next week’s first large shock is "
            "learned on old years and scored on later closed weeks. It is not a public warning."
        ),
    }


def _pick_foreshock(
    raw_x: np.ndarray,
    names: list[str],
    y: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    region_id: str,
) -> tuple[Any, np.ndarray, str]:
    climate = float(y[fit_mask].mean()) if fit_mask.any() else float(y.mean())
    climate_full = float(y[fit_mask | val_mask].mean()) if (fit_mask | val_mask).any() else climate
    usable = _usable_indices(raw_x, fit_mask)
    score_mask = val_mask if val_mask.any() else fit_mask
    climate_brier = float(np.mean((climate - y[score_mask]) ** 2))
    best: tuple[float, float, str, list[int], str, dict] = (
        0.0,
        climate_brier,
        "climate",
        [],
        "climate",
        {},
    )
    local = [i for i in usable if region_id != "global" and region_id in names[i].split("__lag")[0]]
    candidates: list[tuple[str, list[int], str, dict]] = [
        (f"{RECIPE_PREFIX}_thin", _pool_top(raw_x, y, fit_mask, usable, 24), "logit", {"C": 0.25}),
        (f"{RECIPE_PREFIX}_wide", _pool_top(raw_x, y, fit_mask, usable, 48), "logit", {"C": 0.15}),
        (f"{RECIPE_PREFIX}_tree", _pool_top(raw_x, y, fit_mask, usable, 24), "tree", {"max_depth": 2, "min_samples_leaf": 40}),
        (f"{RECIPE_PREFIX}_local", _pool_top(raw_x, y, fit_mask, local or usable, 16), "logit", {"C": 0.2}),
    ]
    y_fit = y[fit_mask]
    if int(fit_mask.sum()) < MIN_TRAIN or int(np.unique(y_fit).size) < 2:
        return None, np.full(len(raw_x), climate_full), "climate|mix0.00"
    for label, cols, kind, kw in candidates:
        cols = [c for c in cols if c in set(usable)]
        if len(cols) < 2:
            continue
        try:
            model = _fit_kind(kind, raw_x[np.ix_(fit_mask, cols)], y_fit, kw)
            p_hat = _proba(model, raw_x[:, cols])
        except ValueError:
            continue
        mix_a, brier = _mix(p_hat, y, climate, score_mask)
        score = brier - climate_brier
        if score >= -1e-12:
            continue
        confirm = np.zeros(len(fit_mask), dtype=bool)
        idx = np.flatnonzero(fit_mask)
        if idx.size > 365:
            confirm[idx[-365:]] = True
        if confirm.any() and int(confirm.sum()) >= MIN_SCORE:
            mixed_c = (1.0 - mix_a) * climate + mix_a * p_hat[confirm]
            confirm_brier = float(np.mean((mixed_c - y[confirm]) ** 2))
            confirm_climate = float(np.mean((climate - y[confirm]) ** 2))
            if confirm_brier >= confirm_climate - 1e-12:
                continue
        if score < best[0] - 1e-12:
            best = (score, brier, f"{label}|mix{mix_a:.2f}", cols, kind, kw)
    _score, _brier_val, recipe, cols, kind, kw = best
    mix_a = 0.0
    if recipe.startswith(RECIPE_PREFIX) and "|mix" in recipe:
        mix_a = float(recipe.rsplit("|mix", 1)[1])
    train_both = fit_mask | val_mask
    if cols and kind != "climate":
        try:
            model = _fit_kind(kind, raw_x[np.ix_(train_both, cols)], y[train_both], kw)
            raw_p = _proba(model, raw_x[:, cols])
        except ValueError:
            model = None
            raw_p = np.full(len(raw_x), climate_full)
            mix_a = 0.0
            recipe = "climate|mix0.00"
    else:
        model = None
        raw_p = np.full(len(raw_x), climate_full)
    p_all = np.clip((1.0 - mix_a) * climate_full + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
    return model, p_all, recipe


def _region_foreshock(
    *,
    feat: pd.DataFrame,
    region_id: str,
    mag: float,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    hits: np.ndarray,
    raw_hits: np.ndarray,
    design: tuple[np.ndarray, list[str]],
) -> dict[str, Any] | None:
    days = pd.DatetimeIndex(feat["issue_time"])
    if int(train_mask.sum()) < MIN_TRAIN or int(hold_mask.sum()) < MIN_SCORE:
        return None
    raw_x, names = design
    if raw_x.shape[1] == 0:
        return None
    y = hits.astype(int)
    fit_mask, val_mask = fit_val_split(days, train_mask)
    val_score = week_mask(val_mask) if val_mask.any() else val_mask
    score_hold = week_mask(hold_mask)
    try:
        _model, p_hat, recipe = _pick_foreshock(raw_x, names, y, fit_mask, val_score, region_id)
    except ValueError:
        return None
    climate = float(y[train_mask].mean())
    alarm = _alarm(p_hat, hits, days, train_mask, score_hold)
    thin = _top_columns(raw_x[train_mask], y[train_mask], names, 8)
    ranked = [
        {
            "feature": names[i],
            "driver": names[i].split("__lag")[0],
            "lag_days": 0 if "__lag" not in names[i] else int(names[i].rsplit("__lag", 1)[1]),
            "label": _driver_plain(names[i]),
            "importance": 1.0,
        }
        for i in thin
    ]
    why = [row["label"] for row in ranked][:6]
    latest = int(np.argmax(days.to_numpy()))
    foreshock_p = float(p_hat[latest])
    hold_brier = _brier(p_hat[score_hold], hits[score_hold].astype(float))
    climate_brier = _brier(np.full(int(score_hold.sum()), climate), hits[score_hold].astype(float))
    held = bool(hold_brier is None or climate_brier is None or hold_brier >= climate_brier)
    this_p = float(climate) if held else foreshock_p
    return {
        "region": region_id,
        "label": _place(region_id),
        "magnitude_min": mag,
        "n_train": int(train_mask.sum()),
        "n_score": int(score_hold.sum()),
        "n_score_daily": int(hold_mask.sum()),
        "fresh_rate_train": climate,
        "fresh_rate_holdout": float(hits[hold_mask].mean()) if hold_mask.any() else None,
        "raw_rate_holdout": float(raw_hits[hold_mask].mean()) if hold_mask.any() else None,
        "brier": hold_brier,
        "climatology_brier": climate_brier,
        **alarm,
        "recipe": recipe,
        "connections": ranked[:10],
        "why": why,
        "this_week": {
            "issue_day": days[latest].date().isoformat(),
            "p": this_p,
            "foreshock_raw_p": foreshock_p,
            "foreshock_raw_chance_plain": _chance(foreshock_p),
            "held_to_climatology": held,
            "chance_plain": _chance(this_p),
            "climatology_p": climate,
            "climatology_chance_plain": _chance(climate),
        },
    }


def _week_sentence(packed: dict[str, Any], tw: dict[str, Any], horizon: int) -> str:
    region_id = str(packed.get("region") or "")
    mag = packed.get("magnitude_min")
    place = "anywhere on Earth" if region_id == "global" else f"in {packed.get('label')}"
    raw_chance = tw.get("foreshock_raw_chance_plain") or _chance(tw.get("foreshock_raw_p"))
    if tw.get("held_to_climatology"):
        return (
            f"The catalog precursor’s attempt for a fresh magnitude {mag:g}+ quake {place} "
            f"in the next {horizon} days is {raw_chance}. "
            f"It has not beaten a constant guess on later years, so the historical rate "
            f"stays {tw['climatology_chance_plain']}."
        )
    return (
        f"The catalog precursor puts the chance of a fresh magnitude {mag:g}+ quake "
        f"{place} in the next {horizon} days at {tw['chance_plain']}. "
        f"A constant guess from old years is {tw['climatology_chance_plain']}."
    )


def _seal_region(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["recipe_leaked"] = False
    out["beats_climatology"] = leftover_beats_climatology(out)
    held = leftover_this_week_held(out)
    tw = dict(out.get("this_week") or {})
    climate = tw.get("climatology_p")
    if climate is None:
        climate = out.get("fresh_rate_train")
    climate = float(climate or 0.0)
    raw = tw.get("foreshock_raw_p")
    tw["climatology_p"] = climate
    tw["climatology_chance_plain"] = _chance(climate)
    tw["held_to_climatology"] = held
    tw["p"] = climate if held else float(raw if raw is not None else climate)
    tw["chance_plain"] = _chance(tw["p"])
    if raw is None:
        tw["foreshock_raw_p"] = tw["p"]
    tw["foreshock_raw_chance_plain"] = _chance(tw.get("foreshock_raw_p"))
    out["this_week"] = tw
    return out


def _seal_week(week_row: dict[str, Any], packed: dict[str, Any]) -> dict[str, Any]:
    tw = packed["this_week"]
    out = dict(week_row)
    out["p"] = tw["p"]
    out["chance_plain"] = tw["chance_plain"]
    out["held_to_climatology"] = tw["held_to_climatology"]
    out["foreshock_raw_p"] = tw.get("foreshock_raw_p")
    out["foreshock_raw_chance_plain"] = tw.get("foreshock_raw_chance_plain") or _chance(
        tw.get("foreshock_raw_p")
    )
    out["climatology_p"] = tw["climatology_p"]
    out["climatology_chance_plain"] = tw["climatology_chance_plain"]
    out["why"] = list(out.get("why") or packed.get("why") or [])
    horizon = int(out.get("horizon_days") or packed.get("horizon_days") or HORIZON_DAYS)
    out["sentence"] = _week_sentence(packed, tw, horizon)
    return out


def _seal_report(report: dict[str, Any]) -> dict[str, Any]:
    regions = [_seal_region(r) for r in report.get("regions") or []]
    report["regions"] = regions
    report["n_beats_climatology"] = sum(1 for r in regions if r.get("beats_climatology"))
    by_region = {r["region"]: r for r in regions}
    report["week"] = [
        _seal_week(w, by_region[w["region"]]) if w.get("region") in by_region else w
        for w in report.get("week") or []
    ]
    report["note"] = _english(report)
    report["writes_issued_p"] = False
    report["track"] = "catalog_precursor"
    return report


def _english(report: dict[str, Any]) -> str:
    bits = [
        f"The catalog precursor looked at {report.get('n_features') or 0} as-of quake-count series "
        "(small and moderate shocks known that morning — not leftover sources) "
        "and tried to predict a fresh large shock in the next week."
    ]
    regions = report.get("regions") or []
    honest = [r for r in regions if r.get("beats_climatology")]
    if not honest:
        bits.append(
            "On the issued boxes, no catalog-precursor mix beat a constant guess on later years."
        )
    else:
        best = min(honest, key=lambda r: r["brier"] - r["climatology_brier"])
        bits.append(
            f"Best later-year score: {best['label']} M≥{best['magnitude_min']:g} "
            f"beat a constant guess (error {best['brier']:.3f} vs {best['climatology_brier']:.3f})."
        )
    bits.append("Official ledger cards stay calibrated ETAS. This is not leftover.")
    return " ".join(bits)


def run_foreshock(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    feat_days = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat_days = feat_days.copy()
    feat_days["issue_time"] = parse_utc(feat_days["issue_time"])
    catalog = quakes if quakes is not None else load_comcat(root)
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    if "available_at" not in catalog.columns:
        catalog["available_at"] = catalog["observed_at"]
    else:
        catalog["available_at"] = parse_utc(catalog["available_at"])
    versions = load_comcat_versions(root)
    days = pd.DatetimeIndex(feat_days["issue_time"])
    catalog_end = catalog["observed_at"].max()
    print(f"catalog precursor features ({len(days)} mornings)", flush=True)
    feat = catalog_foreshock_frame(catalog, feat_days["issue_time"], versions, root)
    columns = catalog_columns(feat)
    leaked = [c for c in columns if driver_family(c) != "catalog"]
    if leaked:
        raise RuntimeError(f"foreshock columns must stay catalog: {leaked[:8]}")
    design = _design(feat, columns, FORESHOCK_LAGS)
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask = closed & (days <= utc_ts(train_end))
    hold_mask = closed & (days >= utc_ts(holdout_start))
    packed_rows: list[dict[str, Any]] = []
    week_rows: list[dict[str, Any]] = []
    for region_id, mag in ISSUED_TARGETS:
        print(f"catalog precursor {region_id} M{mag:g} ({design[0].shape[1]} series)", flush=True)
        region = REGIONS[region_id]
        hits = fresh_hits(catalog, region, mag, days, HORIZON_DAYS)
        raw = _window_hits(catalog, region, mag, days, HORIZON_DAYS)
        quiet = prior_quiet(catalog, region, mag, days, lookback_days=QUIET_LOOKBACK)
        packed = _region_foreshock(
            feat=feat,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask & quiet,
            hold_mask=hold_mask & quiet,
            hits=hits,
            raw_hits=raw,
            design=design,
        )
        if not packed:
            continue
        packed["horizon_days"] = HORIZON_DAYS
        packed["quiet_start"] = True
        packed["track"] = "catalog_precursor"
        packed_rows.append(packed)
        tw = packed["this_week"]
        week_rows.append(
            {
                "region": region_id,
                "label": packed["label"],
                "magnitude_min": mag,
                "horizon_days": HORIZON_DAYS,
                "quiet_start": True,
                "track": "catalog_precursor",
                "p": tw["p"],
                "foreshock_raw_p": tw.get("foreshock_raw_p"),
                "foreshock_raw_chance_plain": tw.get("foreshock_raw_chance_plain")
                or _chance(tw.get("foreshock_raw_p")),
                "held_to_climatology": tw.get("held_to_climatology"),
                "chance_plain": tw["chance_plain"],
                "climatology_p": tw["climatology_p"],
                "climatology_chance_plain": tw["climatology_chance_plain"],
                "why": packed.get("why") or [],
                "issue_day": tw["issue_day"],
                "sentence": _week_sentence(packed, tw, HORIZON_DAYS),
            }
        )
    report = {
        "scanned_at": iso(datetime.now(timezone.utc)),
        "train_end": train_end,
        "holdout_start": holdout_start,
        "horizon_days": HORIZON_DAYS,
        "target": "fresh_mainshock",
        "track": "catalog_precursor",
        "writes_issued_p": False,
        "n_features": int(design[0].shape[1]),
        "n_regions": len(packed_rows),
        "n_beats_climatology": 0,
        "regions": packed_rows,
        "week": week_rows,
        "catalog_end": catalog_end.date().isoformat() if hasattr(catalog_end, "date") else str(catalog_end),
    }
    report = _seal_report(report)
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "foreshock.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
