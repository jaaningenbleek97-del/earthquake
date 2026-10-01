"""Look at every source at once, then learn from closed weeks.

Hunt tests one knob. This tree sees every source together — including
combinations — then residualizes against the rolling 5-year rate. A last-
two-week tree, an M8 track, and a quiet-start track sit beside the wide
all-lag tree. A journal remembers what failed so the next run can improve.
Nothing here writes issued p.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.eval.backtest import _STORE, _key, paper_trade, paper_trade_combo
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.honest import (
    HONEST_LAGS,
    purged_train_hold,
    quiet_start_mask,
    rolling_climate_series,
)
from eqpred.eval.hunt import SKIP_COLS, SKIP_DRIVER_EXACT, TARGETS, _etas_p, driver_family
from eqpred.eval.decluster import fresh_hits as _hits
from eqpred.models.quiet import QUIET_QUANTILE
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc, utc_ts

SPECIALIST_FAMILIES = (
    "solar",
    "wave2",
    "geometry",
    "mass",
    "volcano",
    "cosmic",
    "earth",
    "dust",
    "climate",
    "fresh_catalog",
    "slowslip",
    "analog",
    "graph",
    "combo",
    "spf",
    "radio",
)

BRAIN_LAGS = (0, 7, 14, 27, 30, 54, 81, 90, 180)
BRAIN_LAGS_RECENT = HONEST_LAGS
RARE_TARGETS = (("global", 8.0),)
SHARE = 0.2
MIN_TRAIN = 80
MIN_SCORE = 40
TOP_N = 8
PAIR_N = 4
DEAD_AFTER = 2
WF_CUTS = ("2018-12-31", "2020-12-31", "2022-12-31", "2024-12-31")
RECENT = 12

_TREE = dict(
    n_estimators=60,
    max_depth=3,
    learning_rate=0.06,
    subsample=0.75,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=7,
)


def _eval_dir(root: Path | None) -> Path:
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _brain_dir(root: Path | None) -> Path:
    dest = _eval_dir(root) / "brain"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def load_brain(root: Path | None = None) -> dict[str, Any] | None:
    path = _eval_dir(root) / "brain.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_journal(root: Path | None = None) -> list[dict[str, Any]]:
    path = _brain_dir(root) / "journal.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_memory(root: Path | None = None) -> dict[str, Any]:
    path = _brain_dir(root) / "memory.json"
    if not path.exists():
        return {"failures": {}, "runs": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _driver_columns(features: pd.DataFrame, *, unconventional_only: bool) -> list[str]:
    out: list[str] = []
    for col in features.columns:
        if col in SKIP_COLS or col in SKIP_DRIVER_EXACT:
            continue
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue
        if driver_family(col) == "foil":
            continue
        if unconventional_only and driver_family(col) == "catalog":
            continue
        out.append(str(col))
    return out


def _split_name(name: str) -> tuple[str, int]:
    if "__lag" in name:
        base, lag = name.rsplit("__lag", 1)
        return base, int(lag)
    return name, 0


def _design(feat: pd.DataFrame, columns: list[str], lags: tuple[int, ...] = BRAIN_LAGS) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []
    for col in columns:
        raw = feat[col].to_numpy(dtype=float)
        for lag in lags:
            series = raw if lag <= 0 else np.roll(raw, lag).astype(float)
            if lag > 0:
                series = series.copy()
                series[:lag] = np.nan
            names.append(col if lag == 0 else f"{col}__lag{lag}")
            parts.append(series)
    kept_parts = []
    kept_names = []
    for name, part in zip(names, parts, strict=True):
        if np.isfinite(part).any():
            kept_parts.append(part)
            kept_names.append(name)
    if not kept_parts:
        return np.zeros((len(feat), 0)), []
    return np.column_stack(kept_parts), kept_names


def _impute(train_x: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_x.size == 0:
        return train_x, all_x
    finite_any = np.isfinite(train_x).any(axis=0)
    med = np.zeros(train_x.shape[1], dtype=float)
    if finite_any.any():
        med[finite_any] = np.nanmedian(train_x[:, finite_any], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(train_x), train_x, med), np.where(np.isfinite(all_x), all_x, med)


def _brier(prob: np.ndarray, hits: np.ndarray) -> float | None:
    ok = np.isfinite(prob) & np.isfinite(hits)
    if int(ok.sum()) < MIN_SCORE:
        return None
    return float(np.mean((prob[ok] - hits[ok]) ** 2))


def _alarm(
    score: np.ndarray,
    hits: np.ndarray,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    train = score[train_mask]
    finite = train[np.isfinite(train)]
    if finite.size < MIN_TRAIN:
        return {"n_trades": 0, "hit_rate": None, "base_rate": None, "lift": None, "time_in_alarm": 0.0, "threshold": None, "recent": []}
    threshold = float(np.nanquantile(finite, 1.0 - SHARE))
    entered = np.isfinite(score) & (score > threshold if strict else score >= threshold)
    hold_hits = hits[hold_mask]
    hold_in = entered[hold_mask]
    n_days = int(hold_mask.sum())
    n_trades = int(hold_in.sum())
    n_wins = int((hold_hits & hold_in).sum())
    base = float(hold_hits.mean()) if hold_hits.size else 0.0
    win = float(n_wins / n_trades) if n_trades else None
    hold_days = days[hold_mask]
    hold_score = score[hold_mask]
    recent = []
    for idx in np.flatnonzero(hold_in)[-RECENT:]:
        recent.append(
            {
                "date": hold_days[idx].date().isoformat(),
                "value": None if not np.isfinite(hold_score[idx]) else float(hold_score[idx]),
                "hit": bool(hold_hits[idx]),
            }
        )
    return {
        "n_days": n_days,
        "n_trades": n_trades,
        "n_hits": n_wins,
        "hit_rate": win,
        "base_rate": base,
        "lift": None if win is None else win - base,
        "time_in_alarm": n_trades / n_days if n_days else 0.0,
        "threshold": threshold,
        "recent": recent,
    }


def _ready_climate(hits: np.ndarray, days: pd.DatetimeIndex, train_mask: np.ndarray) -> np.ndarray:
    climate = rolling_climate_series(np.asarray(hits, dtype=float), days)
    fill = float(np.nanmean(np.asarray(hits, dtype=float)[np.asarray(train_mask, dtype=bool)]))
    if not np.isfinite(fill):
        fill = 0.0
    return np.where(np.isfinite(climate), climate, fill)


def _fit_tree(x_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingRegressor | None:
    if x_train.shape[0] < MIN_TRAIN or x_train.shape[1] < 1:
        return None
    if float(np.nanstd(y_train)) == 0:
        return None
    model = GradientBoostingRegressor(**_TREE)
    model.fit(x_train, y_train)
    return model


def _region_fit(
    *,
    feat: pd.DataFrame,
    columns: list[str],
    region_id: str,
    mag: float,
    train_mask: np.ndarray,
    score_mask: np.ndarray,
    hits: np.ndarray,
    etas: np.ndarray,
    raw_x_names: tuple[np.ndarray, list[str]] | None = None,
) -> dict[str, Any] | None:
    days = pd.DatetimeIndex(feat["issue_time"])
    if int(train_mask.sum()) < MIN_TRAIN or int(score_mask.sum()) < MIN_SCORE:
        return None
    raw_x, names = raw_x_names if raw_x_names is not None else _design(feat, columns)
    if raw_x.shape[1] == 0:
        return None
    x_train, x_all = _impute(raw_x[train_mask], raw_x)
    climate = _ready_climate(hits, days, train_mask)
    residual = hits.astype(float) - climate
    model = _fit_tree(x_train, residual[train_mask])
    if model is None:
        return None
    pred = model.predict(x_all)
    p_hat = np.clip(climate + pred, 1e-6, 1 - 1e-6)
    alarm = _alarm(pred, hits, days, train_mask, score_mask, strict=True)
    ranked = sorted(
        (
            {
                "feature": names[i],
                "driver": _split_name(names[i])[0],
                "lag_days": _split_name(names[i])[1],
                "family": driver_family(_split_name(names[i])[0]),
                "importance": float(imp),
            }
            for i, imp in enumerate(model.feature_importances_)
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )
    hold_hits = hits[score_mask].astype(float)
    climate_brier = _brier(climate[score_mask], hold_hits)
    model_brier = _brier(p_hat[score_mask], hold_hits)
    return {
        "region": region_id,
        "region_label": REGIONS[region_id].get("label") or region_id,
        "magnitude_min": mag,
        "n_train": int(train_mask.sum()),
        "n_score": int(score_mask.sum()),
        "brier": model_brier,
        "etas_brier": _brier(etas[score_mask], hold_hits),
        "climatology_brier": climate_brier,
        "climate_brier": climate_brier,
        "beats_climate": bool(
            model_brier is not None and climate_brier is not None and model_brier < climate_brier
        ),
        **alarm,
        "connections": [row for row in ranked if row["importance"] > 0][:12],
    }


def _mem_key(region: str, driver: str, lag: int) -> str:
    return f"{region}::{driver}::{lag}"


def _combo_key(region: str, a: tuple[str, int], b: tuple[str, int]) -> str:
    legs = tuple(sorted((a, b)))
    return f"{region}::{legs[0][0]}@{legs[0][1]}+{legs[1][0]}@{legs[1][1]}"


def _is_dead(memory: dict[str, Any], key: str) -> bool:
    return int((memory.get("failures") or {}).get(key) or 0) >= DEAD_AFTER


def _mark(memory: dict[str, Any], key: str, lift: float | None) -> None:
    fails = memory.setdefault("failures", {})
    if lift is None or lift <= 0:
        fails[key] = int(fails.get(key) or 0) + 1
    else:
        fails[key] = 0


def _experiments(
    connections: list[dict[str, Any]],
    memory: dict[str, Any],
    *,
    train_end: str,
    holdout_start: str,
    root: Path | None,
) -> list[dict[str, Any]]:
    tried: list[dict[str, Any]] = []
    seen: set[str] = set()
    singles: list[dict[str, Any]] = []
    for row in connections:
        key = _mem_key(row["region"], row["driver"], row["lag_days"])
        if key in seen or _is_dead(memory, key):
            continue
        seen.add(key)
        singles.append(row)
        if len(singles) >= TOP_N:
            break
    for row in singles:
        key = _mem_key(row["region"], row["driver"], row["lag_days"])
        try:
            trade = paper_trade(
                row["driver"],
                lag_days=int(row["lag_days"]),
                region_id=row["region"],
                share=SHARE,
                train_end=train_end,
                holdout_start=holdout_start,
                root=root,
            )
        except ValueError:
            continue
        trade["kind"] = "single"
        trade["memory_key"] = key
        _mark(memory, key, trade.get("lift"))
        tried.append(trade)
    pairs = 0
    for i, a in enumerate(singles):
        for b in singles[i + 1 :]:
            if a["region"] != b["region"]:
                continue
            if (a["driver"], a["lag_days"]) == (b["driver"], b["lag_days"]):
                continue
            key = _combo_key(a["region"], (a["driver"], a["lag_days"]), (b["driver"], b["lag_days"]))
            if _is_dead(memory, key):
                continue
            try:
                trade = paper_trade_combo(
                    [(a["driver"], a["lag_days"]), (b["driver"], b["lag_days"])],
                    region_id=a["region"],
                    share=SHARE,
                    train_end=train_end,
                    holdout_start=holdout_start,
                    root=root,
                )
            except ValueError:
                continue
            trade["kind"] = "combo"
            trade["memory_key"] = key
            _mark(memory, key, trade.get("lift"))
            tried.append(trade)
            pairs += 1
            if pairs >= PAIR_N:
                return tried
    return tried


def _english(report: dict[str, Any]) -> str:
    n_feat = report.get("n_features") or 0
    regions = report.get("regions") or []
    usable = [r for r in regions if r.get("lift") is not None]
    best = max(usable, key=lambda r: r["lift"]) if usable else None
    catalog = report.get("catalog_twin") or []
    catalog_best = max((r for r in catalog if r.get("lift") is not None), key=lambda r: r["lift"], default=None)
    wf = report.get("walk_forward") or {}
    improved = report.get("improved")
    top = (report.get("connections") or [])[:3]
    names = ", ".join(f"{c['driver']} ({c['family']})" for c in top) if top else "nothing loud"
    bits = [
        f"The learner looked at {n_feat} series at once "
        f"(every non-catalog source, lags {'/'.join(str(l) for l in BRAIN_LAGS)} days), "
        "then fit the leftover after the rolling 5-year rate, not one knob at a time."
    ]
    if best is None:
        bits.append("It did not get a usable score on later closed weeks.")
    else:
        lift = best["lift"]
        vs = "better than" if lift > 0 else "worse than" if lift < 0 else "even with"
        bits.append(
            f"Best locked alarm: {best['region_label']} - {vs} always-on by {lift * 100:+.1f} pt "
            f"(on {best.get('n_trades') or 0} of {best.get('n_days') or 0} later mornings)."
        )
    brier = report.get("locked_brier")
    etas_brier = report.get("locked_etas_brier")
    climate_brier = report.get("locked_climate_brier")
    if brier is not None and climate_brier is not None:
        vs_c = "better than" if brier < climate_brier else "worse than" if brier > climate_brier else "even with"
        bits.append(f"Chance error is {vs_c} the rolling 5-year rate.")
    if brier is not None and etas_brier is not None and brier > etas_brier:
        bits.append("Its raw chance number is still worse than ETAS. A small alarm lift is not a better probability.")
    if catalog_best and best and catalog_best.get("lift") is not None and best.get("lift") is not None:
        gap = catalog_best["lift"] - best["lift"]
        if gap > 0.005:
            bits.append(
                "A twin that was allowed recent-quake counts did better. That extra is aftershock echo, not a new driver."
            )
        else:
            bits.append("Letting it see recent-quake counts did not help. The signal is not just aftershocks.")
    if wf.get("mean_lift") is not None:
        bits.append(
            f"Walk-forward (train, then score the next years, repeat as weeks closed) mean lift {wf['mean_lift'] * 100:+.1f} pt across {wf.get('n_folds') or 0} folds."
        )
    if improved is True:
        bits.append("This run beat the last journal entry on model error. It is learning.")
    elif improved is False:
        bits.append("This run did not beat the last journal entry. Memory still drops knobs that failed twice.")
    else:
        bits.append("First journal entry. Later runs will say whether it improved.")
    quiet = report.get("quiet") or {}
    if quiet.get("locked_lift") is not None:
        bits.append(
            f"Quiet-only tree (ETAS in the boring tail): lift {quiet['locked_lift'] * 100:+.1f} pt."
        )
    quiet_start = report.get("quiet_start") or {}
    if quiet_start.get("locked_lift") is not None:
        bits.append(
            f"Quiet-start tree (no large shock in the last 10 days): lift {quiet_start['locked_lift'] * 100:+.1f} pt."
        )
    recent = report.get("recent") or {}
    if recent.get("n_features"):
        bits.append(
            f"Last-two-week tree still sees every source ({recent['n_features']} series)."
        )
    rare = report.get("rare") or {}
    if rare.get("regions"):
        bits.append("An M8 track is scored separately. It does not write issued p.")
    mixer = report.get("mixer") or {}
    if mixer.get("brier") is not None and mixer.get("etas_brier") is not None:
        vs = "better than" if mixer["brier"] < mixer["etas_brier"] else "worse than"
        bits.append(f"Specialist mixer error is {vs} ETAS.")
    bits.append(f"What it leaned on: {names}. Official issued p is unchanged.")
    return " ".join(bits)


def _mean(values: list[float | None]) -> float | None:
    good = [v for v in values if v is not None and np.isfinite(v)]
    if not good:
        return None
    return float(np.mean(good))


def _quiet_mask(etas: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, float | None]:
    sample = np.asarray(etas, dtype=float)[np.asarray(train_mask, dtype=bool)]
    sample = sample[np.isfinite(sample)]
    if sample.size < MIN_TRAIN:
        return np.zeros(len(etas), dtype=bool), None
    cut = float(np.quantile(sample, QUIET_QUANTILE))
    return np.isfinite(etas) & (etas <= cut), cut


def run_brain(
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
    _STORE[_key(root)] = {"features": feat, "quakes": catalog}

    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)

    unconv = _driver_columns(feat, unconventional_only=True)
    all_cols = _driver_columns(feat, unconventional_only=False)
    memory = load_memory(root)
    journal = load_journal(root)
    target_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for region_id, mag in TARGETS:
        target_cache[(region_id, mag)] = (
            _hits(catalog, REGIONS[region_id], mag, days, HORIZON_DAYS),
            _etas_p(catalog, days, region_id, mag, root, versions=versions),
        )
    unconv_x = _design(feat, unconv)
    all_x = _design(feat, all_cols)

    regions: list[dict[str, Any]] = []
    catalog_twin: list[dict[str, Any]] = []
    connections: list[dict[str, Any]] = []
    for region_id, mag in TARGETS:
        hits, etas = target_cache[(region_id, mag)]
        locked = _region_fit(
            feat=feat,
            columns=unconv,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask,
            score_mask=hold_mask,
            hits=hits,
            etas=etas,
            raw_x_names=unconv_x,
        )
        if locked:
            regions.append(locked)
            for row in locked.get("connections") or []:
                connections.append({**row, "region": region_id, "region_label": locked["region_label"]})
        twin = _region_fit(
            feat=feat,
            columns=all_cols,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask,
            score_mask=hold_mask,
            hits=hits,
            etas=etas,
            raw_x_names=all_x,
        )
        if twin:
            catalog_twin.append(
                {
                    k: twin[k]
                    for k in ("region", "region_label", "lift", "brier", "etas_brier", "climate_brier", "beats_climate")
                    if k in twin
                }
            )

    quiet_regions: list[dict[str, Any]] = []
    quiet_cut = None
    for region_id, mag in TARGETS:
        hits, etas = target_cache[(region_id, mag)]
        qmask, qcut = _quiet_mask(etas, train_mask)
        if qcut is not None:
            quiet_cut = qcut
        locked = _region_fit(
            feat=feat,
            columns=unconv,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask & qmask,
            score_mask=hold_mask & qmask,
            hits=hits,
            etas=etas,
            raw_x_names=unconv_x,
        )
        if locked:
            quiet_regions.append({k: v for k, v in locked.items() if k != "connections"})

    recent_x = _design(feat, unconv, lags=BRAIN_LAGS_RECENT)
    recent_regions: list[dict[str, Any]] = []
    for region_id, mag in TARGETS:
        hits, etas = target_cache[(region_id, mag)]
        row = _region_fit(
            feat=feat,
            columns=unconv,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask,
            score_mask=hold_mask,
            hits=hits,
            etas=etas,
            raw_x_names=recent_x,
        )
        if row:
            recent_regions.append({k: v for k, v in row.items() if k != "connections"})

    quiet_start_regions: list[dict[str, Any]] = []
    for region_id, mag in TARGETS:
        hits, etas = target_cache[(region_id, mag)]
        qstart = quiet_start_mask(catalog, REGIONS[region_id], mag, days)
        row = _region_fit(
            feat=feat,
            columns=unconv,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask & qstart,
            score_mask=hold_mask & qstart,
            hits=hits,
            etas=etas,
            raw_x_names=unconv_x,
        )
        if row:
            quiet_start_regions.append({k: v for k, v in row.items() if k != "connections"})

    rare_regions: list[dict[str, Any]] = []
    for region_id, mag in RARE_TARGETS:
        rare_hits = _hits(catalog, REGIONS[region_id], mag, days, HORIZON_DAYS)
        rare_etas = _etas_p(catalog, days, region_id, mag, root, versions=versions)
        row = _region_fit(
            feat=feat,
            columns=unconv,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask,
            score_mask=hold_mask,
            hits=rare_hits,
            etas=rare_etas,
            raw_x_names=unconv_x,
        )
        if row:
            rare_regions.append({k: v for k, v in row.items() if k != "connections"})

    specialists: list[dict[str, Any]] = []
    mixer_pred = None
    mixer_n = 0
    global_pair = next((pair for pair in TARGETS if pair[0] == "global"), None)
    if global_pair:
        g_hits, g_etas = target_cache[global_pair]
        mix_parts: list[np.ndarray] = []
        for family in SPECIALIST_FAMILIES:
            fam_cols = [c for c in unconv if driver_family(c) == family]
            if len(fam_cols) < 2:
                continue
            fam_x = _design(feat, fam_cols)
            row = _region_fit(
                feat=feat,
                columns=fam_cols,
                region_id="global",
                mag=global_pair[1],
                train_mask=train_mask,
                score_mask=hold_mask,
                hits=g_hits,
                etas=g_etas,
                raw_x_names=fam_x,
            )
            if not row:
                continue
            specialists.append(
                {
                    "family": family,
                    "n_features": fam_x[0].shape[1],
                    "lift": row.get("lift"),
                    "brier": row.get("brier"),
                    "etas_brier": row.get("etas_brier"),
                }
            )
            raw_x, _names = fam_x
            x_train, x_all = _impute(raw_x[train_mask], raw_x)
            g_climate = _ready_climate(g_hits, days, train_mask)
            model = _fit_tree(x_train, (g_hits.astype(float) - g_climate)[train_mask])
            if model is None:
                continue
            mix_parts.append(np.clip(g_climate + model.predict(x_all), 1e-6, 1 - 1e-6))
        if mix_parts:
            mixer_pred = np.mean(np.column_stack(mix_parts), axis=1)
            mixer_n = len(mix_parts)

    connections = sorted(connections, key=lambda r: r.get("importance") or 0, reverse=True)
    folds: list[dict[str, Any]] = []
    cuts = [utc_ts(c) for c in WF_CUTS]
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else None
        wf_train = closed & (days <= cut)
        wf_score = closed & (days > cut)
        if nxt is not None:
            wf_score = wf_score & (days <= nxt)
        for region_id, mag in TARGETS:
            hits, etas = target_cache[(region_id, mag)]
            row = _region_fit(
                feat=feat,
                columns=unconv,
                region_id=region_id,
                mag=mag,
                train_mask=wf_train,
                score_mask=wf_score,
                hits=hits,
                etas=etas,
                raw_x_names=unconv_x,
            )
            if not row:
                continue
            folds.append(
                {
                    "train_end": cut.date().isoformat(),
                    "score_end": None if nxt is None else nxt.date().isoformat(),
                    "region": region_id,
                    "lift": row.get("lift"),
                    "brier": row.get("brier"),
                    "etas_brier": row.get("etas_brier"),
                    "n_train": row.get("n_train"),
                    "n_score": row.get("n_score"),
                }
            )

    experiments = _experiments(
        connections,
        memory,
        train_end=train_end,
        holdout_start=holdout_start,
        root=root,
    )
    locked_brier = _mean([r.get("brier") for r in regions])
    locked_etas = _mean([r.get("etas_brier") for r in regions])
    locked_climate = _mean([r.get("climate_brier") for r in regions])
    locked_lift = _mean([r.get("lift") for r in regions])
    prev = journal[-1] if journal else None
    improved = None
    if prev and locked_brier is not None and prev.get("locked_brier") is not None:
        improved = locked_brier < float(prev["locked_brier"]) - 1e-4

    mixer = {"n_specialists": mixer_n, "brier": None, "etas_brier": None, "climate_brier": None}
    if mixer_pred is not None and global_pair is not None:
        g_hits, g_etas = target_cache[global_pair]
        hold_y = g_hits[hold_mask].astype(float)
        g_climate = _ready_climate(g_hits, days, train_mask)
        mixer["brier"] = _brier(mixer_pred[hold_mask], hold_y)
        mixer["etas_brier"] = _brier(g_etas[hold_mask], hold_y)
        mixer["climate_brier"] = _brier(g_climate[hold_mask], hold_y)

    report = {
        "writes_issued_p": False,
        "looks_at_all_sources": True,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "n_features": int(unconv_x[0].shape[1]),
        "n_knobs": len(unconv),
        "n_days": int(closed.sum()),
        "catalog_end": catalog_end.date().isoformat() if hasattr(catalog_end, "date") else str(catalog_end),
        "locked_brier": locked_brier,
        "locked_etas_brier": locked_etas,
        "locked_climate_brier": locked_climate,
        "locked_beats_climate": bool(
            locked_brier is not None and locked_climate is not None and locked_brier < locked_climate
        ),
        "locked_lift": locked_lift,
        "improved": improved,
        "recent": {
            "lags_days": list(BRAIN_LAGS_RECENT),
            "n_features": int(recent_x[0].shape[1]),
            "looks_at_all_sources": True,
            "locked_lift": _mean([r.get("lift") for r in recent_regions]),
            "locked_brier": _mean([r.get("brier") for r in recent_regions]),
            "locked_climate_brier": _mean([r.get("climate_brier") for r in recent_regions]),
            "regions": recent_regions,
        },
        "quiet": {
            "quantile": QUIET_QUANTILE,
            "etas_p_cut": quiet_cut,
            "locked_lift": _mean([r.get("lift") for r in quiet_regions]),
            "locked_brier": _mean([r.get("brier") for r in quiet_regions]),
            "locked_etas_brier": _mean([r.get("etas_brier") for r in quiet_regions]),
            "locked_climate_brier": _mean([r.get("climate_brier") for r in quiet_regions]),
            "regions": quiet_regions,
        },
        "quiet_start": {
            "lookback_days": 10,
            "locked_lift": _mean([r.get("lift") for r in quiet_start_regions]),
            "locked_brier": _mean([r.get("brier") for r in quiet_start_regions]),
            "locked_climate_brier": _mean([r.get("climate_brier") for r in quiet_start_regions]),
            "regions": quiet_start_regions,
        },
        "rare": {
            "targets": [{"region": rid, "magnitude_min": mag} for rid, mag in RARE_TARGETS],
            "locked_lift": _mean([r.get("lift") for r in rare_regions]),
            "locked_brier": _mean([r.get("brier") for r in rare_regions]),
            "locked_climate_brier": _mean([r.get("climate_brier") for r in rare_regions]),
            "regions": rare_regions,
        },
        "specialists": specialists,
        "mixer": mixer,
        "regions": [
            {k: v for k, v in r.items() if k != "connections"} | {"top": (r.get("connections") or [])[:4]}
            for r in regions
        ],
        "catalog_twin": catalog_twin,
        "walk_forward": {
            "cuts": list(WF_CUTS),
            "n_folds": len(folds),
            "mean_lift": _mean([f.get("lift") for f in folds]),
            "mean_brier": _mean([f.get("brier") for f in folds]),
            "mean_etas_brier": _mean([f.get("etas_brier") for f in folds]),
            "folds": folds,
        },
        "connections": connections[:24],
        "experiments": [
            {
                "kind": t.get("kind"),
                "driver": t.get("driver"),
                "lag_days": t.get("lag_days"),
                "region": t.get("region"),
                "region_label": t.get("region_label"),
                "lift": t.get("lift"),
                "hit_rate": t.get("hit_rate"),
                "base_rate": t.get("base_rate"),
                "time_in_alarm": t.get("time_in_alarm"),
                "n_trades": t.get("n_trades"),
                "legs": t.get("legs"),
            }
            for t in experiments
        ],
        "journal_runs": len(journal) + 1,
        "dead_keys": [
            key for key, n in (memory.get("failures") or {}).items() if int(n) >= DEAD_AFTER
        ],
        "scanned_at": iso(datetime.now(timezone.utc)),
        "note": "",
    }
    report["note"] = _english(report)

    dest = _eval_dir(root) / "brain.json"
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    memory["runs"] = int(memory.get("runs") or 0) + 1
    memory["updated_at"] = report["scanned_at"]
    (_brain_dir(root) / "memory.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
    entry = {
        "ran_at": report["scanned_at"],
        "locked_brier": locked_brier,
        "locked_etas_brier": locked_etas,
        "locked_lift": locked_lift,
        "walk_mean_lift": report["walk_forward"]["mean_lift"],
        "n_features": report["n_features"],
        "improved": improved,
    }
    with (_brain_dir(root) / "journal.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    report["journal"] = (journal + [entry])[-8:]
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report

