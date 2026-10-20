"""Three mainshock questions that do not touch the frozen cards.

1. A 30-day chance from declustered first shocks already known that morning.
2. Whether the next first shock falls in the quiet, still-locked patch.
3. One Hikurangi retest on interface shocks only.

Nothing here writes issued probability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.eval.decluster import decluster_causal, fresh_hits
from eqpred.eval.honest import fit_val_split, purged_train_hold, rolling_climate_series
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.skill import information_gain
from eqpred.faults import SEGMENTS, segment_region
from eqpred.features.repeaters import DEPTH_MAX_KM, DEPTH_MIN_KM
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS, in_bbox
from eqpred.stamp import parse_utc

LOOKBACK_DAYS = 365 * 5
GAIN_MIN = 0.01
QUIET_DAYS = 90
PATCH_PARENTS = ("hikurangi", "alpine")
SLIP_COLUMNS = ("east_extra_30d", "north_extra_30d")


def known_counts(
    observed: np.ndarray,
    available: np.ndarray,
    days: pd.DatetimeIndex,
    *,
    lookback_days: int = LOOKBACK_DAYS,
) -> np.ndarray:
    """How many events were already reported, inside the lookback, on each morning."""
    obs = np.asarray(observed, dtype="datetime64[ns]")
    avail = np.asarray(available, dtype="datetime64[ns]")
    if obs.size == 0:
        return np.zeros(len(days), dtype=float)
    order = np.argsort(obs, kind="mergesort")
    obs = obs[order]
    avail = avail[order]
    issue = pd.DatetimeIndex(days).tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    look = np.timedelta64(int(lookback_days), "D")
    counts = np.zeros(len(issue), dtype=float)
    for i, day in enumerate(issue):
        right = int(np.searchsorted(obs, day, side="right"))
        left = int(np.searchsorted(obs, day - look, side="right"))
        if right > left:
            counts[i] = float(np.count_nonzero(avail[left:right] <= day))
    return counts


def known_event_rate(
    observed: np.ndarray,
    available: np.ndarray,
    days: pd.DatetimeIndex,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    horizon_days: int = HORIZON_DAYS,
) -> np.ndarray:
    """30-day chance from events already reported, counted over the past five years."""
    counts = known_counts(observed, available, days, lookback_days=lookback_days)
    lam = counts / float(lookback_days)
    return 1.0 - np.exp(-lam * float(horizon_days))


def background_rates(
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    *,
    region_id: str,
    mag: float,
) -> np.ndarray:
    """Declustered first shocks in one box. A later quake cannot change an earlier label."""
    work = _box_events(catalog, REGIONS[region_id], mag)
    if work.empty:
        return np.full(len(days), 1e-6)
    mains = decluster_causal(work)
    return known_event_rate(
        parse_utc(mains["observed_at"]).to_numpy(dtype="datetime64[ns]"),
        parse_utc(mains["available_at"]).to_numpy(dtype="datetime64[ns]"),
        days,
    )


def interface_fresh_hits(
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    *,
    mag: float,
    horizon_days: int = HORIZON_DAYS,
) -> np.ndarray:
    """First shocks on the Hikurangi plate interface. Crust and deep slab stay out."""
    from eqpred.eval.strategies import _window_hits

    work = _box_events(catalog, REGIONS["hikurangi"], mag)
    if work.empty or "depth_km" not in work.columns:
        return np.zeros(len(days), dtype=bool)
    depth = pd.to_numeric(work["depth_km"], errors="coerce")
    work = work.loc[depth.ge(DEPTH_MIN_KM) & depth.le(DEPTH_MAX_KM)]
    if work.empty:
        return np.zeros(len(days), dtype=bool)
    mains = decluster_causal(work)
    return _window_hits(mains, REGIONS["hikurangi"], mag, days, horizon_days)


def choose_locked_patch(
    slip: np.ndarray,
    quakes: np.ndarray,
) -> int:
    """Index of the patch that is quiet and has barely moved. Lower is more locked."""
    slip = np.asarray(slip, dtype=float)
    quakes = np.asarray(quakes, dtype=float)
    slip_rank = _rank_low(slip)
    quake_rank = _rank_low(quakes)
    score = slip_rank + quake_rank
    score[~np.isfinite(slip) & ~np.isfinite(quakes)] = np.inf
    if not np.isfinite(score).any():
        return 0
    return int(np.argmin(score))


def run_where_research(
    root: Path | None = None,
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
) -> dict[str, Any]:
    """Score the three questions and write data/eval/where_research.json."""
    from eqpred.catalog import load_comcat
    from eqpred.eval.mainshock import ISSUED_TARGETS

    feat = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat["issue_time"] = parse_utc(feat["issue_time"])
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog = load_comcat(root)
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    catalog["available_at"] = parse_utc(catalog["available_at"])
    catalog_end = catalog["observed_at"].max()
    closed = np.asarray(days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end)
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start, purge=HORIZON_DAYS)
    _fit, val = fit_val_split(days, train_mask, purge=HORIZON_DAYS)
    stations = _station_coords()

    backgrounds = []
    for region_id, mag in ISSUED_TARGETS:
        if region_id == "global":
            continue
        rate = background_rates(catalog, days, region_id=region_id, mag=mag)
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, HORIZON_DAYS).astype(float)
        roll = rolling_climate_series(hits, days, horizon=HORIZON_DAYS)
        backgrounds.append(
            _scored(region_id, mag, rate, hits, roll, val, hold_mask)
            | {"question": "declustered background rate"}
        )

    patches = []
    for parent in PATCH_PARENTS:
        mag = dict(ISSUED_TARGETS)[parent]
        patches.append(
            _patch_score(
                parent,
                mag,
                feat,
                catalog,
                days,
                stations,
                val,
                hold_mask,
            )
        )

    interface = _interface_retest(feat, catalog, days, stations, val, hold_mask)
    earns = [row for row in [*backgrounds, *patches, *interface] if row.get("earns_frozen_card")]
    report = {
        "scored_at": iso(datetime.now(timezone.utc)),
        "writes_issued_p": False,
        "earns_frozen_card": bool(earns),
        "gain_min": GAIN_MIN,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "background": backgrounds,
        "locked_patch": patches,
        "interface": interface,
        "note": _note(backgrounds, patches, interface),
    }
    dest = data_dir(root) / "eval" / "where_research.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _scored(
    region_id: str,
    mag: float,
    p: np.ndarray,
    hits: np.ndarray,
    roll: np.ndarray,
    val: np.ndarray,
    hold: np.ndarray,
) -> dict[str, Any]:
    check = information_gain(p[val], hits[val], roll[val])
    replay = information_gain(p[hold], hits[hold], roll[hold])
    earns = check is not None and check >= GAIN_MIN
    return {
        "region": region_id,
        "magnitude_min": mag,
        "check_information_gain": check,
        "replay_information_gain": replay,
        "earns_frozen_card": earns,
        "writes_issued_p": False,
        "n_check": int(val.sum()),
        "n_replay": int(hold.sum()),
    }


def _patch_score(
    parent: str,
    mag: float,
    feat: pd.DataFrame,
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    stations: dict[str, tuple[float, float]],
    val: np.ndarray,
    hold: np.ndarray,
) -> dict[str, Any]:
    names = [name for name, rec in SEGMENTS.items() if rec["parent"] == parent]
    regions = [segment_region(name) for name in names]
    slip = np.column_stack([_segment_slip(feat, region, stations) for region in regions])
    quakes = np.column_stack(
        [_segment_quakes(catalog, region, days, mag) for region in regions]
    )
    chosen = np.array([choose_locked_patch(slip[i], quakes[i]) for i in range(len(days))])
    hits = np.column_stack(
        [
            fresh_hits(catalog, region, mag, days, HORIZON_DAYS).astype(float)
            for region in regions
        ]
    )
    y = hits[np.arange(len(days)), chosen]
    parent_hits = fresh_hits(catalog, REGIONS[parent], mag, days, HORIZON_DAYS).astype(float)
    in_box = parent_hits > 0
    correct = (y > 0) & in_box
    n_shocks = int(in_box[hold].sum())
    located = int(correct[hold].sum())
    chance = 1.0 / len(names)
    roll = rolling_climate_series(y, days, horizon=HORIZON_DAYS)
    box_rate = background_rates(catalog, days, region_id=parent, mag=mag)
    # The box rate is placed on the one quiet locked patch. The usual rate is an equal share.
    p = np.clip(box_rate, 1e-6, 1.0 - 1e-6)
    equal = np.clip(box_rate / len(names), 1e-6, 1.0 - 1e-6)
    scored = _scored(parent, mag, p, y, equal, val, hold)
    scored.update(
        {
            "question": "quiet locked patch",
            "n_patches": len(names),
            "patches": names,
            "replay_shocks_in_box": n_shocks,
            "replay_shocks_in_chosen_patch": located,
            "replay_located_share": None if n_shocks == 0 else located / n_shocks,
            "equal_share": chance,
            "segment_climate_information_gain": information_gain(p[hold], y[hold], roll[hold]),
        }
    )
    return scored


def _interface_retest(
    feat: pd.DataFrame,
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    stations: dict[str, tuple[float, float]],
    val: np.ndarray,
    hold: np.ndarray,
) -> list[dict[str, Any]]:
    from eqpred.eval.lineup import _scale

    mag = 5.0
    hits = interface_fresh_hits(catalog, days, mag=mag).astype(float)
    roll = rolling_climate_series(hits, days, horizon=HORIZON_DAYS)
    closed = np.asarray(days + pd.Timedelta(days=HORIZON_DAYS) <= catalog["observed_at"].max())
    train_mask, _hold = purged_train_hold(days, closed, "2018-12-31", "2019-01-01", purge=HORIZON_DAYS)
    fit, _val = fit_val_split(days, train_mask, purge=HORIZON_DAYS)
    columns = {"qres_tec_hikurangi": "Hikurangi TEC residual"}
    med_east = _box_slip_median(feat, "hikurangi", "east_extra_30d", stations)
    med_north = _box_slip_median(feat, "hikurangi", "north_extra_30d", stations)
    series = {
        "qres_tec_hikurangi": _numeric(feat, "qres_tec_hikurangi"),
        "hikurangi_east_extra_30d_median": med_east,
        "hikurangi_north_extra_30d_median": med_north,
    }
    labels = {
        "qres_tec_hikurangi": "Hikurangi TEC residual",
        "hikurangi_east_extra_30d_median": "Hikurangi median extra east slip, 30 days",
        "hikurangi_north_extra_30d_median": "Hikurangi median extra north slip, 30 days",
    }
    _ = columns
    rows = []
    for name, values in series.items():
        if values is None or int(np.isfinite(values).sum()) < 80:
            rows.append(
                {
                    "region": "hikurangi",
                    "column": name,
                    "label": labels[name],
                    "question": "interface-only first shock",
                    "check_information_gain": None,
                    "replay_information_gain": None,
                    "earns_frozen_card": False,
                    "writes_issued_p": False,
                    "note": "Too few readings to score.",
                }
            )
            continue
        fit_vals = values[fit & np.isfinite(values)]
        mu = float(np.mean(fit_vals)) if fit_vals.size else 0.0
        sd = float(np.std(fit_vals)) if fit_vals.size else 0.0
        z = np.full(len(values), np.nan)
        if sd > 1e-12:
            ok = np.isfinite(values)
            z[ok] = (values[ok] - mu) / sd
        scale = _scale(z, hits, roll, fit)
        p = np.clip(np.where(np.isfinite(roll), roll, 0.0) + scale * np.where(np.isfinite(z), z, 0.0), 1e-6, 1 - 1e-6)
        row = _scored("hikurangi", mag, p, hits, roll, val, hold)
        row.update(
            {
                "column": name,
                "label": labels[name],
                "question": "interface-only first shock",
                "learned_points_per_sigma": scale * 100.0,
            }
        )
        rows.append(row)
    return rows


def _segment_slip(
    feat: pd.DataFrame,
    region: dict,
    stations: dict[str, tuple[float, float]],
) -> np.ndarray:
    columns = []
    for slug, (lon, lat) in stations.items():
        if not _in_segment(lat, lon, region):
            continue
        for kind in SLIP_COLUMNS:
            name = f"gnss_{slug}_{kind}"
            if name in feat.columns:
                columns.append(name)
    if not columns:
        return np.full(len(feat), np.nan)
    stack = np.column_stack([_numeric(feat, name) for name in columns])
    return np.nanmedian(np.abs(stack), axis=1)


def _box_slip_median(
    feat: pd.DataFrame,
    region_id: str,
    kind: str,
    stations: dict[str, tuple[float, float]],
) -> np.ndarray | None:
    region = REGIONS[region_id]
    columns = []
    for slug, (lon, lat) in stations.items():
        if not _in_segment(lat, lon, region):
            continue
        name = f"gnss_{slug}_{kind}"
        if name in feat.columns:
            columns.append(name)
    if not columns:
        return None
    stack = np.column_stack([_numeric(feat, name) for name in columns])
    return np.nanmedian(stack, axis=1)


def _segment_quakes(
    catalog: pd.DataFrame,
    region: dict,
    days: pd.DatetimeIndex,
    mag: float,
) -> np.ndarray:
    work = _box_events(catalog, region, 3.0)
    if work.empty:
        return np.zeros(len(days))
    work = work.loc[work["magnitude"] < mag]
    if work.empty:
        return np.zeros(len(days))
    mains = decluster_causal(work, space_km=50.0, time_days=10.0)
    return known_counts(
        parse_utc(mains["observed_at"]).to_numpy(dtype="datetime64[ns]"),
        parse_utc(mains["available_at"]).to_numpy(dtype="datetime64[ns]"),
        days,
        lookback_days=QUIET_DAYS,
    )


def _box_events(catalog: pd.DataFrame, region: dict, mag: float) -> pd.DataFrame:
    work = catalog.loc[pd.to_numeric(catalog["magnitude"], errors="coerce") >= mag].copy()
    if region.get("kind") == "bbox":
        work = work.loc[in_bbox(work, region)]
    return work


def _in_segment(lat: float, lon: float, region: dict) -> bool:
    if lon > 180:
        lon -= 360.0
    return (
        region["min_lat"] <= lat <= region["max_lat"]
        and region["min_lon"] <= lon <= region["max_lon"]
    )


def _station_coords() -> dict[str, tuple[float, float]]:
    try:
        from eqpred.ingest.gnss import ngl_station_coords
    except Exception:
        return {}
    try:
        found = ngl_station_coords()
    except Exception:
        return {}
    return {site.lower(): (lon, lat) for site, (lon, lat) in found.items()}


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in frame.columns:
        return None
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _rank_low(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    ok = np.isfinite(values)
    if int(ok.sum()) == 0:
        return out
    order = np.argsort(values[ok])
    ranks = np.empty(int(ok.sum()), dtype=float)
    ranks[order] = np.arange(1, int(ok.sum()) + 1)
    out[ok] = ranks
    return out


def _note(background: list[dict], patches: list[dict], interface: list[dict]) -> str:
    def bit(row: dict) -> str:
        gain = row.get("check_information_gain")
        shown = "none" if gain is None else f"{gain:.4f}"
        return f"{row.get('region')} {row.get('question') or row.get('column')} check gain {shown}"

    parts = [bit(row) for row in [*background, *patches, *interface]]
    return (
        "Research only. A check-year information gain of 0.01 would make a row eligible later. "
        "None of these rows replace the frozen 30-day cards. " + "; ".join(parts)
    )
