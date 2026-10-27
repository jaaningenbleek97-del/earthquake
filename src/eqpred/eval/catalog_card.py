"""Catalog-only benchmark. ETAS plus an EEPAS addition, scored on answer key version 2.

A quake covered by a bigger recent shock does not count. Quakes larger than that
shock still count. Slow slip on Hikurangi only scales the rate, using a factor
chosen on years before 2009. Nothing here writes the live forecast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.eval.decluster import fresh_hits, uhrhammer_days
from eqpred.eval.honest import rolling_climate_series, week_mask
from eqpred.eval.strategy_desk import gain_band
from eqpred.models.etas import EtasParams, daily_background_and_cluster, params_from_dict
from eqpred.models.gr import fit_b_value
from eqpred.paths import data_dir, feature_dir, model_dir
from eqpred.regions import REGIONS, in_bbox, lon_spans
from eqpred.safe_io import write_json
from eqpred.stamp import parse_utc

M0 = 4.5
HEADLINE = (
    ("global", 7.0, 30),
    ("hikurangi", 5.0, 30),
    ("alpine", 5.0, 30),
    ("kermadec", 6.0, 30),
    ("global", 7.0, 7),
    ("hikurangi", 5.0, 7),
    ("alpine", 5.0, 7),
    ("kermadec", 6.0, 7),
)


def catalog_path(root: Path | None = None) -> Path:
    return data_dir(root) / "eval" / "catalog_card.json"


def load_catalog_card(root: Path | None = None) -> dict[str, Any] | None:
    path = catalog_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _params(root: Path | None) -> EtasParams:
    path = model_dir(root) / "etas.json"
    if not path.exists():
        return EtasParams(m0=M0)
    return params_from_dict(json.loads(path.read_text(encoding="utf-8"))).clip()


def _as_reported(events: pd.DataFrame) -> pd.DataFrame:
    """The rate may use a quake only once it had been reported."""
    work = events.copy()
    work["observed_at"] = parse_utc(work["observed_at"])
    if "available_at" in work.columns:
        work["observed_at"] = parse_utc(work["available_at"])
    return work.dropna(subset=["observed_at", "magnitude"]).sort_values("observed_at")


def _in_region(events: pd.DataFrame, region: dict) -> pd.DataFrame:
    if region.get("kind") != "bbox":
        return events
    if "lat" not in events.columns and "latitude" in events.columns:
        events = events.copy()
        events["lat"] = events["latitude"]
        events["lon"] = events["longitude"]
    return events.loc[in_bbox(events, region)]


def covering_magnitude(event_day: np.ndarray, event_mag: np.ndarray, n_days: int) -> np.ndarray:
    """Largest earlier quake whose aftershock window still covers each day. -inf if none."""
    cover = np.full(n_days, -np.inf)
    order = np.argsort(event_day, kind="mergesort")
    for idx in order:
        day = int(event_day[idx])
        mag = float(event_mag[idx])
        if not np.isfinite(mag) or day >= n_days:
            continue
        span = int(np.ceil(max(10.0, float(uhrhammer_days(mag)))))
        start = max(day + 1, 0)
        stop = min(n_days, day + span + 1)
        if start < stop:
            cover[start:stop] = np.maximum(cover[start:stop], mag)
    return cover


def fresh_daily_rate(
    rate_m0: np.ndarray,
    cover: np.ndarray,
    target: float,
    b: float,
    m0: float = M0,
) -> np.ndarray:
    """Expected first-shock count that day. A covering quake removes shocks up to its own size."""
    covered = np.isfinite(cover) & (cover >= target)
    floor = np.where(covered, cover + 0.1, target)
    gap = np.maximum(floor - m0, 0.0)
    return np.asarray(rate_m0, dtype=float) * (10.0 ** (-float(b) * gap))


def _bounded_eepas(events: pd.DataFrame, days: pd.DatetimeIndex, m0: float, max_age: int = 1825) -> np.ndarray:
    """EEPAS-shaped addition, kept to five years so a worldwide catalog stays runnable."""
    from eqpred.eval.eepas import EEPAS_A, EEPAS_B, EEPAS_C, EEPAS_P

    n = len(days)
    rate = np.zeros(n, dtype=float)
    if events.empty:
        return rate
    origin = days[0]
    t_ev = ((events["observed_at"] - origin) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    mag = events["magnitude"].to_numpy(dtype=float)
    one_m_p = 1.0 - EEPAS_P
    prod = 10.0 ** (EEPAS_A + EEPAS_B * (mag - m0))
    for ti, pr in zip(t_ev, prod):
        if not np.isfinite(pr) or not np.isfinite(ti):
            continue
        start = max(int(np.floor(ti)) + 1, 0)
        stop = min(n, start + max_age)
        if start >= stop:
            continue
        t0 = np.arange(start, stop, dtype=float) - ti
        integ = pr * ((t0 + 1.0 + EEPAS_C) ** one_m_p - (t0 + EEPAS_C) ** one_m_p) / one_m_p
        rate[start:stop] += np.maximum(integ, 0.0)
    return rate


def _match_training_rate(expected: np.ndarray, hits: np.ndarray, train: np.ndarray) -> float:
    """One level, chosen before 2009, so the average chance matches the average outcome."""
    use = train & np.isfinite(expected) & np.isfinite(hits)
    if int(use.sum()) < 30:
        return 1.0
    target = float(np.clip(hits[use].mean(), 1e-4, 1.0 - 1e-4))
    base = np.clip(expected[use], 0.0, None)

    def mean_p(level: float) -> float:
        return float(np.mean(1.0 - np.exp(-base * level)))

    low, high = 1e-4, 1e3
    if mean_p(high) < target:
        return high
    for _ in range(24):
        mid = (low + high) / 2.0
        if mean_p(mid) < target:
            low = mid
        else:
            high = mid
    return float(high)


def window_expected(daily: np.ndarray, horizon: int) -> np.ndarray:
    """Sum of the next `horizon` days, not including a day before this morning."""
    out = np.full(len(daily), np.nan)
    if horizon <= 0 or len(daily) <= horizon:
        return out
    rolled = np.convolve(np.nan_to_num(daily, nan=0.0), np.ones(horizon), mode="full")
    stop = len(daily) - horizon
    out[:stop] = rolled[horizon - 1 : horizon - 1 + stop]
    return out


def _slowslip_factor(hits: np.ndarray, sse: np.ndarray | None, train: np.ndarray) -> float:
    if sse is None:
        return 1.0
    on = train & np.isfinite(sse) & (sse > 0.5) & np.isfinite(hits)
    off = train & np.isfinite(sse) & (sse <= 0.5) & np.isfinite(hits)
    if int(on.sum()) < 8 or int(off.sum()) < 8 or float(hits[off].mean()) <= 0:
        return 1.0
    return float(np.clip(hits[on].mean() / hits[off].mean(), 0.5, 2.0))


def _sse_series(root: Path | None, days: pd.DatetimeIndex) -> np.ndarray | None:
    path = feature_dir(root) / "global_daily.parquet"
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    if "sse_hikurangi" not in names or "issue_time" not in names:
        return None
    frame = pd.read_parquet(path, columns=["issue_time", "sse_hikurangi"])
    frame["issue_time"] = parse_utc(frame["issue_time"]).dt.normalize()
    left = pd.DataFrame({"issue_time": days.normalize()})
    merged = left.merge(frame, on="issue_time", how="left")
    return pd.to_numeric(merged["sse_hikurangi"], errors="coerce").to_numpy(dtype=float)


def _score(pred: np.ndarray, hits: np.ndarray, days: pd.DatetimeIndex, horizon: int, train_end: str) -> dict[str, Any]:
    from eqpred.eval.skill import information_gain

    closed = np.asarray(days + pd.Timedelta(days=horizon) <= days[-1])
    roll = rolling_climate_series(hits.astype(float), days, horizon=horizon)
    climate = np.where(np.isfinite(roll), roll, np.nan)
    later = closed & (days >= "2009-01-01")
    mask = week_mask(later, stride=horizon, fallback=False)
    ok = mask & np.isfinite(pred) & np.isfinite(hits) & np.isfinite(climate)
    gain = information_gain(pred[ok], hits[ok].astype(float), climate[ok]) if int(ok.sum()) else None
    band = gain_band(pred, hits.astype(float), climate, ok)
    return {
        "mornings": int(ok.sum()),
        "information_gain": None if gain is None or not np.isfinite(gain) else round(float(gain), 4),
        "gain_low": None if band is None else round(float(band["low"]), 4),
        "gain_high": None if band is None else round(float(band["high"]), 4),
        "said": None if not ok.any() else round(float(pred[ok].mean()), 4),
        "happened": None if not ok.any() else round(float(hits[ok].mean()), 4),
        "usual": None if not ok.any() else round(float(climate[ok].mean()), 4),
        "train_end": train_end,
    }


def score_region(
    events: pd.DataFrame,
    days: pd.DatetimeIndex,
    region: dict,
    target: float,
    horizon: int,
    params: EtasParams,
    sse: np.ndarray | None,
    train_end: str,
) -> dict[str, Any]:
    local = _in_region(events, region)
    source_floor = max(M0, float(target) - 2.0)
    local = local.loc[local["magnitude"] >= source_floor]
    hits = fresh_hits(events, region, target, days, horizon, label_version=2).astype(float)
    train = np.asarray(days <= train_end)
    if local.empty:
        pred = np.full(len(days), 1e-6)
        summary = {"target": target, "horizon_days": horizon, "slowslip_factor": 1.0, **_score(pred, hits, days, horizon, train_end)}
        return _with_forecasts(summary, days, region, pred, hits, horizon, target)
    origin = days[0]
    day_index = ((local["observed_at"] - origin) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    mag = local["magnitude"].to_numpy(dtype=float)
    fitted = EtasParams(params.mu, params.k, params.alpha, params.c, params.p, source_floor).clip()
    rate = daily_background_and_cluster(local, days, fitted, max_age_days=1825)
    combined = rate + 0.25 * _bounded_eepas(local, days, source_floor)
    cover = covering_magnitude(np.floor(day_index).astype(int), mag, len(days))
    cut = int(np.searchsorted(days, pd.Timestamp(train_end, tz="UTC"), side="right"))
    b = fit_b_value(mag[day_index < cut], M0)
    fresh = fresh_daily_rate(combined, cover, target, b, m0=source_floor)
    factor = 1.0
    if region.get("name") == "hikurangi":
        factor = _slowslip_factor(hits, sse, train)
        if sse is not None:
            on = np.isfinite(sse) & (sse > 0.5)
            fresh = np.where(on, fresh * factor, fresh)
    expected = window_expected(fresh, horizon)
    train_closed = train & np.isfinite(expected) & (days + pd.Timedelta(days=horizon) <= days[-1])
    level = _match_training_rate(expected, hits, train_closed)
    pred = 1.0 - np.exp(-np.clip(expected, 0.0, None) * level)
    pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
    summary = {
        "target": target,
        "horizon_days": horizon,
        "b_value": round(float(b), 3),
        "slowslip_factor": round(float(factor), 3),
        **_score(pred, hits, days, horizon, train_end),
    }
    return _with_forecasts(summary, days, region, pred, hits, horizon, target)


def _with_forecasts(
    summary: dict[str, Any],
    days: pd.DatetimeIndex,
    region: dict,
    pred: np.ndarray,
    hits: np.ndarray,
    horizon: int,
    magnitude: float,
) -> dict[str, Any]:
    from eqpred.eval.attempts import forecast_frame

    roll = rolling_climate_series(hits.astype(float), days, horizon=horizon)
    closed = np.asarray(days + pd.Timedelta(days=horizon) <= days[-1])
    later = closed & (days >= "2009-01-01")
    summary["_forecasts"] = forecast_frame(
        days=days,
        region=str(region.get("name") or ""),
        label=str(region.get("label") or region.get("name") or ""),
        horizon=horizon,
        magnitude=float(magnitude),
        p=pred,
        climatology=np.where(np.isfinite(roll), roll, np.nan),
        happened=hits,
        mask=week_mask(later, stride=horizon, fallback=False),
        catalog_end=days[-1],
    )
    return summary


def grid_cells(region: dict) -> list[dict]:
    """Four patches inside a box. A date-line box is split on each side."""
    if region.get("kind") != "bbox":
        return [region]
    lat_edges = np.linspace(float(region["min_lat"]), float(region["max_lat"]), 3)
    cells = []
    spans = lon_spans(float(region["min_lon"]), float(region["max_lon"]))
    for lo, hi in spans:
        lon_edges = np.linspace(lo, hi, 3)
        for i in range(2):
            for j in range(2):
                cells.append(
                    {
                        "kind": "bbox",
                        "name": f"{region['name']}_{i}{j}",
                        "label": f"{region['label']} patch {i + 1},{j + 1}",
                        "min_lat": float(lat_edges[i]),
                        "max_lat": float(lat_edges[i + 1]),
                        "min_lon": float(lon_edges[j]),
                        "max_lon": float(lon_edges[j + 1]),
                    }
                )
    return cells


def run_catalog_card(root: Path | None = None, train_end: str = "2008-12-31") -> dict[str, Any]:
    from eqpred.catalog import load_comcat

    catalog = _as_reported(load_comcat(root))
    if "lat" not in catalog.columns and "latitude" in catalog.columns:
        catalog["lat"] = catalog["latitude"]
        catalog["lon"] = catalog["longitude"]
    days = pd.date_range(catalog["observed_at"].min().normalize(), catalog["observed_at"].max().normalize(), freq="D", tz="UTC")
    params = _params(root)
    sse = _sse_series(root, days)
    rows = []
    frames = []
    for region_id, target, horizon in HEADLINE:
        print(f"catalog card {region_id} M{target:g} {horizon}d", flush=True)
        row = score_region(catalog, days, REGIONS[region_id], target, horizon, params, sse, train_end)
        frames.append(row.pop("_forecasts"))
        row["region"] = region_id
        row["label"] = REGIONS[region_id]["label"]
        row["kind"] = "box"
        rows.append(row)
    grid = []
    for region_id in ("hikurangi", "alpine", "kermadec"):
        for cell in grid_cells(REGIONS[region_id]):
            print(f"catalog grid {cell['label']} M4.5 30d", flush=True)
            row = score_region(catalog, days, cell, 4.5, 30, params, sse, train_end)
            frames.append(row.pop("_forecasts"))
            row["region"] = region_id
            row["label"] = cell["label"]
            row["kind"] = "grid"
            grid.append(row)
    from eqpred.eval.attempts import save_forecasts

    kept = [frame for frame in frames if frame is not None and len(frame)]
    saved = (
        save_forecasts(
            pd.concat(kept, ignore_index=True),
            source="catalog",
            name="etas_eepas",
            label="ETAS plus EEPAS",
            root=root,
            label_version=2,
            answer_key="fresh.v2",
            knobs={"train_end": train_end},
            note=(
                "Catalog benchmark on answer key version 2. "
                "Each row is one dated chance. This does not change the live exam."
            ),
        )
        if kept
        else {"n_forecasts": 0, "attempt_id": None}
    )
    report = {
        "writes_issued_p": False,
        "label_version": 2,
        "train_end": train_end,
        "note": (
            "Catalog only. ETAS plus a quarter of an EEPAS-shaped rate, using quakes within two magnitudes of the target. "
            "A bigger recent quake removes first shocks up to its own size. "
            "Hikurangi slow slip is allowed to scale the rate only if it has readings before 2009. "
            "This does not replace the live card."
        ),
        "rows": rows,
        "grid": grid,
        "saved": {"attempt_id": saved.get("attempt_id"), "n_forecasts": saved.get("n_forecasts") or 0},
    }
    write_json(catalog_path(root), report)
    return report
