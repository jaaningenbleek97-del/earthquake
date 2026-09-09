"""Replay a knob on weeks that already happened.

Cards still wait for a live window to close. This bench does not. It locks a
threshold on the train era, then scores every holdout morning whose week is
already in the catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat
from eqpred.eval.hunt import LAGS_DAYS, TARGETS, driver_family, load_hunt
from eqpred.eval.strategies import _window_hits
from eqpred.models.predict import BOX_MAG, GLOBAL_MAG
from eqpred.paths import data_dir, feature_dir
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc, utc_ts

DEFAULT_SHARE = 0.2
RECENT = 12

_STORE: dict[str, dict[str, Any]] = {}


def _key(root: Path | None) -> str:
    return str(root) if root is not None else "__default__"


def _store(root: Path | None = None) -> dict[str, Any]:
    key = _key(root)
    cached = _STORE.get(key)
    if cached is not None:
        return cached
    features = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    features["issue_time"] = parse_utc(features["issue_time"])
    quakes = load_comcat(root)
    quakes = quakes.copy()
    quakes["observed_at"] = parse_utc(quakes["observed_at"])
    payload = {"features": features, "quakes": quakes}
    _STORE[key] = payload
    return payload


def list_drivers(root: Path | None = None) -> list[str]:
    features = _store(root)["features"]
    return [
        str(col)
        for col in features.columns
        if col != "issue_time" and pd.api.types.is_numeric_dtype(features[col])
    ]


def _lagged(values: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return values
    out = np.roll(values, lag).astype(float)
    out[:lag] = np.nan
    return out


def _default_mag(region_id: str) -> float:
    if region_id == "global":
        return GLOBAL_MAG
    return float(BOX_MAG.get(region_id, 5.0))


def paper_trade(
    driver: str,
    *,
    lag_days: int = 0,
    region_id: str = "global",
    share: float = DEFAULT_SHARE,
    side: str = "auto",
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    horizon_days: int = 7,
    magnitude_min: float | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Alarm on the train-era tail of one knob. Score closed holdout weeks."""
    if region_id not in REGIONS:
        raise ValueError(f"unknown region {region_id}")
    store = _store(root)
    feat = store["features"]
    if driver not in feat.columns:
        raise ValueError(f"unknown driver {driver}")
    mag = float(magnitude_min if magnitude_min is not None else _default_mag(region_id))
    region = REGIONS[region_id]
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = store["quakes"]["observed_at"].max()
    closed = days + pd.Timedelta(days=horizon_days) <= catalog_end
    train_cut = utc_ts(train_end)
    hold_cut = utc_ts(holdout_start)
    train_mask = closed & (days <= train_cut)
    hold_mask = closed & (days >= hold_cut)

    raw = feat[driver].to_numpy(dtype=float)
    series = _lagged(raw, int(lag_days))
    hits = _window_hits(store["quakes"], region, mag, days, horizon_days)

    train_x = series[train_mask]
    finite = train_x[np.isfinite(train_x)]
    if finite.size < 40:
        raise ValueError(f"not enough train days for {driver}")

    unique = np.unique(finite)
    flag_like = unique.size <= 3 and float(unique.max() - unique.min()) <= 1.0
    chosen_side = side
    if chosen_side == "auto":
        if flag_like:
            chosen_side = "high"
        else:
            train_hits = hits[train_mask]
            ok = np.isfinite(train_x)
            if int(ok.sum()) >= 40 and float(np.nanstd(train_x[ok])) > 0:
                rho = float(pd.Series(train_x[ok]).corr(pd.Series(train_hits[ok].astype(float)), method="spearman"))
                chosen_side = "low" if np.isfinite(rho) and rho < 0 else "high"
            else:
                chosen_side = "high"

    share = float(np.clip(share, 0.02, 0.8))
    if flag_like:
        threshold = 0.5
        entered = series > threshold if chosen_side == "high" else series < threshold
    elif chosen_side == "low":
        threshold = float(np.nanquantile(finite, share))
        entered = series <= threshold
    else:
        threshold = float(np.nanquantile(finite, 1.0 - share))
        entered = series >= threshold

    hold_hits = hits[hold_mask]
    hold_in = entered[hold_mask] & np.isfinite(series[hold_mask])
    n_days = int(hold_mask.sum())
    n_trades = int(hold_in.sum())
    n_wins = int((hold_hits & hold_in).sum())
    base = float(hold_hits.mean()) if hold_hits.size else 0.0
    win = float(n_wins / n_trades) if n_trades else None
    hold_days = days[hold_mask]
    hold_vals = series[hold_mask]
    recent = []
    picked = np.flatnonzero(hold_in)
    for idx in picked[-RECENT:]:
        recent.append(
            {
                "date": hold_days[idx].date().isoformat(),
                "value": None if not np.isfinite(hold_vals[idx]) else float(hold_vals[idx]),
                "entered": True,
                "hit": bool(hold_hits[idx]),
            }
        )

    return _trade_payload(
        driver=driver,
        lag_days=int(lag_days),
        region_id=region_id,
        region=region,
        mag=mag,
        horizon_days=horizon_days,
        share=share,
        side=chosen_side,
        threshold=threshold,
        train_end=train_end,
        holdout_start=holdout_start,
        n_days=n_days,
        n_trades=n_trades,
        n_hits=n_wins,
        hit_rate=win,
        base_rate=base,
        recent=recent,
        extra={"legs": [{"driver": driver, "lag_days": int(lag_days)}]},
    )


def paper_trade_combo(
    legs: list[tuple[str, int]],
    *,
    region_id: str = "global",
    share: float = DEFAULT_SHARE,
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    horizon_days: int = 7,
    magnitude_min: float | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """AND of several knob tails. Same closed-week bench as paper_trade."""
    if not legs:
        raise ValueError("combo needs at least one leg")
    if region_id not in REGIONS:
        raise ValueError(f"unknown region {region_id}")
    entered = None
    labels = []
    for driver, lag in legs:
        one = paper_trade(
            driver,
            lag_days=int(lag),
            region_id=region_id,
            share=share,
            train_end=train_end,
            holdout_start=holdout_start,
            horizon_days=horizon_days,
            magnitude_min=magnitude_min,
            root=root,
        )
        labels.append(f"{driver}@{lag}d")
        store = _store(root)
        feat = store["features"]
        series = _lagged(feat[driver].to_numpy(dtype=float), int(lag))
        days = pd.DatetimeIndex(feat["issue_time"])
        catalog_end = store["quakes"]["observed_at"].max()
        closed = days + pd.Timedelta(days=horizon_days) <= catalog_end
        train_cut = utc_ts(train_end)
        train_mask = closed & (days <= train_cut)
        train_x = series[train_mask]
        finite = train_x[np.isfinite(train_x)]
        side = one["side"]
        if side == "low":
            flag = series <= float(one["threshold"])
        else:
            flag = series >= float(one["threshold"])
        flag = flag & np.isfinite(series)
        entered = flag if entered is None else (entered & flag)
        _ = finite
    store = _store(root)
    feat = store["features"]
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = store["quakes"]["observed_at"].max()
    mag = float(magnitude_min if magnitude_min is not None else _default_mag(region_id))
    region = REGIONS[region_id]
    hits = _window_hits(store["quakes"], region, mag, days, horizon_days)
    closed = days + pd.Timedelta(days=horizon_days) <= catalog_end
    hold_cut = utc_ts(holdout_start)
    hold_mask = closed & (days >= hold_cut)
    hold_hits = hits[hold_mask]
    hold_in = entered[hold_mask]
    n_days = int(hold_mask.sum())
    n_trades = int(hold_in.sum())
    n_wins = int((hold_hits & hold_in).sum())
    base = float(hold_hits.mean()) if hold_hits.size else 0.0
    win = float(n_wins / n_trades) if n_trades else None
    hold_days = days[hold_mask]
    recent = []
    picked = np.flatnonzero(hold_in)
    for idx in picked[-RECENT:]:
        recent.append(
            {
                "date": hold_days[idx].date().isoformat(),
                "value": None,
                "entered": True,
                "hit": bool(hold_hits[idx]),
            }
        )
    name = " + ".join(labels)
    return _trade_payload(
        driver=name,
        lag_days=0,
        region_id=region_id,
        region=region,
        mag=mag,
        horizon_days=horizon_days,
        share=share,
        side="and",
        threshold=None,
        train_end=train_end,
        holdout_start=holdout_start,
        n_days=n_days,
        n_trades=n_trades,
        n_hits=n_wins,
        hit_rate=win,
        base_rate=base,
        recent=recent,
        extra={
            "legs": [{"driver": d, "lag_days": int(lag)} for d, lag in legs],
            "combo": True,
        },
    )


def _trade_payload(
    *,
    driver: str,
    lag_days: int,
    region_id: str,
    region: dict[str, Any],
    mag: float,
    horizon_days: int,
    share: float,
    side: str,
    threshold: float | None,
    train_end: str,
    holdout_start: str,
    n_days: int,
    n_trades: int,
    n_hits: int,
    hit_rate: float | None,
    base_rate: float,
    recent: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "driver": driver,
        "family": driver_family(driver.split(" + ", 1)[0]) if " + " not in driver else "combo",
        "aftershock_shaped": driver_family(driver.split(" + ", 1)[0]) == "catalog" and " + " not in driver,
        "lag_days": int(lag_days),
        "region": region_id,
        "region_label": region.get("label") or region.get("name") or region_id,
        "magnitude_min": mag,
        "horizon_days": int(horizon_days),
        "share": share,
        "side": side,
        "threshold": threshold,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "n_days": n_days,
        "n_trades": n_trades,
        "n_hits": n_hits,
        "time_in_alarm": n_trades / n_days if n_days else 0.0,
        "hit_rate": hit_rate,
        "base_rate": base_rate,
        "lift": None if hit_rate is None else hit_rate - base_rate,
        "recent": recent,
        "note": "Uses weeks that already closed. No waiting.",
    }
    if extra:
        payload.update(extra)
    return payload


def pack_from_hunt(
    share: float = DEFAULT_SHARE,
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    limit: int = 24,
) -> dict[str, Any]:
    hunt = load_hunt(root)
    if not hunt:
        raise FileNotFoundError("no hunt.json; run eqpred hunt first")
    rows = [
        row
        for row in hunt.get("confirmed") or []
        if not row.get("aftershock_shaped")
    ]
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row["driver"]), str(row["region"]))
        prev = best.get(key)
        if prev is None or abs(row.get("holdout_rho") or 0) > abs(prev.get("holdout_rho") or 0):
            best[key] = row
    ranked = sorted(best.values(), key=lambda r: abs(r.get("holdout_rho") or 0), reverse=True)[:limit]
    trades = []
    for row in ranked:
        try:
            trades.append(
                paper_trade(
                    row["driver"],
                    lag_days=int(row.get("lag_days") or 0),
                    region_id=str(row["region"]),
                    share=share,
                    train_end=train_end,
                    holdout_start=holdout_start,
                    magnitude_min=row.get("magnitude_min"),
                    root=root,
                )
            )
        except ValueError:
            continue
    report = {
        "share": share,
        "n_trades": len(trades),
        "trades": trades,
        "note": "Paper trades from hunt survivors. Threshold locked on train. Holdout weeks already happened.",
    }
    dest = data_dir(root) / "eval" / "backtest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def load_backtest(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "backtest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def available_lags() -> list[int]:
    return list(LAGS_DAYS)


def available_targets() -> list[dict[str, Any]]:
    return [
        {"region": rid, "magnitude_min": mag, "label": REGIONS[rid].get("label") or rid}
        for rid, mag in TARGETS
    ]
