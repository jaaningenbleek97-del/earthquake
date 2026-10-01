"""Point-in-time sequence objects grown from the locked clusterer.

A morning only sees events already reported. Constants stay locked.
These columns stay in the catalog family and never write issued p.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.features.migration import (
    BOX_IDS,
    COMCAT_MAG,
    GEONET_MAG,
    LOOKBACK_DAYS,
    MIN_EVENTS,
    _prepare_catalog,
    active_cluster,
    load_migration_catalogs,
    measure_cluster,
    shape_label,
    slope_per_day,
    swarm_score,
    visible_window,
    wrap180_arr,
)
from eqpred.regions import REGIONS, in_bbox
from eqpred.stamp import parse_utc, utc_ts

BG_DAYS = 365
MIN_B_POS = 8
MIN_AKI = 30
MIN_TRAFFIC_N = 30
MC_SHIFT = 0.3
TRAFFIC_DROP = 0.10
M6 = 6.0
NZ_TRAFFIC = ("hikurangi", "alpine")
RJ_P = 0.05

SEQ_METRICS = (
    "n",
    "active",
    "hours",
    "rate_vs_bg",
    "days_since_prev",
    "max_m",
    "max_med",
    "mag_gap",
    "swarm",
    "b_pos",
    "b_aki",
    "b_pos_delta",
    "pprime",
    "n_1d",
    "n_7d",
    "along",
    "updip",
    "area",
    "area_growth",
    "depth_slope",
    "fault_km",
)


def b_positive(magnitudes, dm_min: float = 0.0, min_n: int = MIN_B_POS) -> float:
    """van der Elst B-positive: only positive consecutive magnitude differences."""
    mag = np.asarray(magnitudes, dtype=float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 2:
        return float("nan")
    diffs = np.diff(mag)
    pos = diffs[diffs > dm_min]
    if pos.size < min_n:
        return float("nan")
    mean = float(pos.mean() - dm_min)
    if mean <= 1e-9:
        return float("nan")
    return float(np.log10(np.e) / mean)


def aki_utsu_local(magnitudes, min_n: int = MIN_AKI, mc: float | None = None) -> float:
    """Aki–Utsu on the cluster's own Mc. Diagnostic only — not the traffic light."""
    mag = np.asarray(magnitudes, dtype=float)
    mag = mag[np.isfinite(mag)]
    if mag.size < min_n:
        return float("nan")
    if mc is None:
        mc = float(np.floor(np.min(mag) * 10.0) / 10.0)
    mag = mag[mag >= mc]
    if mag.size < min_n:
        return float("nan")
    denom = float(mag.mean() - (mc - 0.05))
    if denom <= 1e-9:
        return float("nan")
    return float(np.clip(np.log10(np.e) / denom, 0.4, 2.5))


def inverse_omori_pprime(hours) -> float:
    """p′ from the cluster clock. tc is last event plus the median gap (locked)."""
    t = np.sort(np.asarray(hours, dtype=float))
    t = t[np.isfinite(t)]
    if t.size < 5:
        return float("nan")
    dt = np.diff(t)
    if not np.any(dt > 0):
        return float("nan")
    pad = max(float(np.median(dt[dt > 0])), 1.0)
    rem = (t[-1] + pad) - t[:-1]
    rate = 1.0 / np.maximum(dt, 1e-3)
    ok = (rem > 0) & np.isfinite(rate) & (rate > 0)
    if int(ok.sum()) < 4:
        return float("nan")
    x = np.log(rem[ok])
    y = np.log(rate[ok])
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.dot(x, x))
    if den < 1e-8:
        return float("nan")
    return float(np.clip(-float(np.dot(x, y) / den), 0.0, 4.0))


def hull_area_km2(lon, lat) -> float:
    pts_lon = np.asarray(lon, dtype=float)
    pts_lat = np.asarray(lat, dtype=float)
    ok = np.isfinite(pts_lon) & np.isfinite(pts_lat)
    pts_lon, pts_lat = pts_lon[ok], pts_lat[ok]
    if pts_lon.size < 3:
        return float("nan")
    lat0 = float(np.mean(pts_lat))
    x = wrap180_arr(pts_lon - float(np.mean(pts_lon))) * 111.32 * np.cos(np.radians(lat0))
    y = (pts_lat - float(np.mean(pts_lat))) * 111.32
    pts = np.column_stack([x, y])
    uniq = np.unique(np.round(pts, 3), axis=0)
    if uniq.shape[0] < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        return float("nan")
    try:
        return float(ConvexHull(uniq).volume)
    except QhullError:
        return float("nan")


def traffic_light(
    *,
    region_id: str,
    max_m: float,
    n: int,
    n_bg: int,
    b_pos: float,
    b_pos_bg: float,
    mc_shift: bool,
) -> str:
    """Gulia-style garnish on NZ boxes after a local M≥6. B-positive only."""
    if region_id not in NZ_TRAFFIC:
        return "none"
    if not np.isfinite(max_m) or float(max_m) < M6:
        return "none"
    if (
        int(n) < MIN_TRAFFIC_N
        or int(n_bg) < MIN_TRAFFIC_N
        or not np.isfinite(b_pos)
        or not np.isfinite(b_pos_bg)
        or float(b_pos_bg) <= 0
        or mc_shift
    ):
        return "yellow"
    rel = (float(b_pos) - float(b_pos_bg)) / float(b_pos_bg)
    if rel <= -TRAFFIC_DROP:
        return "red"
    if rel >= 0.0:
        return "green"
    return "yellow"


def traffic_sentence(region_id: str, light: str, max_m: float | None) -> str | None:
    """Gulia line for This week. Does not change issued p."""
    if region_id not in NZ_TRAFFIC:
        return None
    lamp = str(light or "none")
    if lamp in {"none", "nan", ""}:
        return None
    mag = ""
    if max_m is not None and np.isfinite(max_m):
        mag = f"M{float(max_m):.1f}"
    event = f"the local {mag}" if mag else "the local M≥6"
    tail = "This does not change the published 30-day chance."
    if lamp == "red":
        return (
            f"After {event}, b+ is down versus the quiet year — "
            f"this may still be a foreshock. {tail}"
        )
    if lamp == "green":
        return f"After {event}, b+ is not down versus the quiet year. {tail}"
    if lamp == "yellow":
        return f"After {event}, the sequence light is yellow. {tail}"
    return None


def catalogs_from_quakes(quakes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a provided catalog the same way migration does when GeoNet is absent."""
    work = quakes.copy()
    if "depth_km" not in work.columns:
        work["depth_km"] = 0.0
    out: dict[str, pd.DataFrame] = {}
    for rid in BOX_IDS:
        minmag = GEONET_MAG if rid in ("hikurangi", "alpine") else COMCAT_MAG
        prep = _prepare_catalog(work, minmag)
        if prep.empty:
            out[rid] = prep
            continue
        out[rid] = prep.loc[in_bbox(prep, REGIONS[rid])].copy()
    return out


def load_sequence_catalogs(
    quakes: pd.DataFrame | None = None,
    root: Path | None = None,
    catalogs: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    if catalogs is not None:
        return catalogs
    if root is not None:
        loaded = load_migration_catalogs(root)
        if any(not df.empty for df in loaded.values()):
            return loaded
    if quakes is None:
        return {rid: pd.DataFrame() for rid in BOX_IDS}
    return catalogs_from_quakes(quakes)


def _finite(value) -> float:
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _mc_percentile(magnitudes) -> float:
    mag = np.asarray(magnitudes, dtype=float)
    mag = mag[np.isfinite(mag)]
    if mag.size == 0:
        return float("nan")
    return float(np.percentile(mag, 5))


def describe_sequence(
    cluster: pd.DataFrame,
    region_id: str,
    issue,
    background: pd.DataFrame | None = None,
) -> dict:
    """Numeric as-of row plus display fields. Empty cluster → inactive zeros."""
    stamp = utc_ts(issue)
    empty = {
        "n": 0,
        "active": 0,
        "hours": 0.0,
        "rate_vs_bg": 0.0,
        "max_m": float("nan"),
        "max_med": float("nan"),
        "mag_gap": float("nan"),
        "swarm": float("nan"),
        "shape": "none",
        "b_pos": float("nan"),
        "b_aki": float("nan"),
        "b_pos_delta": float("nan"),
        "pprime": float("nan"),
        "n_1d": 0,
        "n_7d": 0,
        "along": float("nan"),
        "updip": float("nan"),
        "area": float("nan"),
        "area_growth": float("nan"),
        "depth_slope": float("nan"),
        "fault_km": float("nan"),
        "traffic_light": "none",
        "n_bg": 0,
    }
    if cluster is None or len(cluster) < MIN_EVENTS:
        return empty
    measured = measure_cluster(cluster, region_id)
    stamps = parse_utc(cluster["observed_at"])
    hours = (stamps - stamps.min()) / pd.Timedelta(hours=1)
    hours_v = hours.to_numpy(dtype=float)
    mags = pd.to_numeric(cluster["magnitude"], errors="coerce").to_numpy(dtype=float)
    duration_h = float(np.nanmax(hours_v)) if hours_v.size else 0.0
    duration_d = max(duration_h / 24.0, 1.0 / 24.0)
    last_day = stamps.max()
    n_1d = int((stamps >= last_day - pd.Timedelta(days=1)).sum())
    n_7d = int((stamps >= last_day - pd.Timedelta(days=7)).sum())
    finite_m = mags[np.isfinite(mags)]
    if finite_m.size >= 2:
        ranked = np.sort(finite_m)
        mag_gap = float(ranked[-1] - ranked[-2])
        max_med = float(ranked[-1] - np.median(ranked))
        max_m = float(ranked[-1])
    elif finite_m.size == 1:
        mag_gap = float("nan")
        max_med = 0.0
        max_m = float(finite_m[0])
    else:
        mag_gap = float("nan")
        max_med = float("nan")
        max_m = float("nan")
    area = hull_area_km2(cluster["lon"], cluster["lat"])
    mid = duration_h / 2.0
    early = cluster.loc[hours_v <= mid] if duration_h > 0 else cluster.iloc[0:0]
    area_early = hull_area_km2(early["lon"], early["lat"]) if len(early) >= 3 else float("nan")
    area_growth = (
        float(area / area_early)
        if np.isfinite(area) and np.isfinite(area_early) and area_early > 1e-6
        else float("nan")
    )
    bg_mags = np.asarray([], dtype=float)
    if background is not None and not background.empty:
        bg_mags = pd.to_numeric(background["magnitude"], errors="coerce").to_numpy(dtype=float)
    b_now = b_positive(mags)
    b_bg = b_positive(bg_mags)
    mc_shift = bool(
        np.isfinite(_mc_percentile(mags))
        and np.isfinite(_mc_percentile(bg_mags))
        and _mc_percentile(mags) > _mc_percentile(bg_mags) + MC_SHIFT
    )
    bg_n = int(np.isfinite(bg_mags).sum())
    bg_rate = bg_n / float(BG_DAYS)
    cluster_rate = len(cluster) / duration_d
    rate_vs_bg = float(cluster_rate / bg_rate) if bg_rate > 1e-9 else float("nan")
    light = traffic_light(
        region_id=region_id,
        max_m=max_m,
        n=int(len(cluster)),
        n_bg=bg_n,
        b_pos=b_now,
        b_pos_bg=b_bg,
        mc_shift=mc_shift,
    )
    return {
        "n": int(len(cluster)),
        "active": 1,
        "hours": duration_h,
        "rate_vs_bg": rate_vs_bg,
        "max_m": max_m,
        "max_med": max_med,
        "mag_gap": mag_gap,
        "swarm": _finite(measured["swarm"] if np.isfinite(measured["swarm"]) else swarm_score(mags)),
        "shape": measured.get("shape") or shape_label(swarm_score(mags)),
        "b_pos": b_now,
        "b_aki": aki_utsu_local(mags),
        "b_pos_delta": (b_now - b_bg) if np.isfinite(b_now) and np.isfinite(b_bg) else float("nan"),
        "pprime": inverse_omori_pprime(hours_v),
        "n_1d": n_1d,
        "n_7d": n_7d,
        "along": _finite(measured["along_km_d"]),
        "updip": _finite(measured["updip_km_d"]),
        "area": area,
        "area_growth": area_growth,
        "depth_slope": slope_per_day(hours_v, pd.to_numeric(cluster["depth_km"], errors="coerce").to_numpy(dtype=float)),
        "fault_km": _finite(measured.get("fault_km")),
        "traffic_light": light,
        "n_bg": bg_n,
        "issue": stamp.isoformat().replace("+00:00", "Z"),
    }


def _background(catalog: pd.DataFrame, cluster: pd.DataFrame, issue) -> pd.DataFrame:
    if catalog is None or catalog.empty:
        return catalog if catalog is not None else pd.DataFrame()
    stamp = utc_ts(issue)
    start = stamp - pd.Timedelta(days=BG_DAYS)
    cluster_t0 = parse_utc(cluster["observed_at"]).min() if cluster is not None and len(cluster) else stamp
    work = catalog.loc[
        (catalog["available_at"] <= stamp)
        & (catalog["observed_at"] >= start)
        & (catalog["observed_at"] < cluster_t0)
    ]
    return work


def sequence_asof(catalog: pd.DataFrame, region_id: str, issue) -> dict:
    stamp = utc_ts(issue)
    window = visible_window(catalog, stamp)
    cluster = active_cluster(window) if not window.empty else window
    return describe_sequence(cluster, region_id, stamp, _background(catalog, cluster, stamp))


def sequence_snapshot(feat: pd.DataFrame, region_id: str, index: int) -> dict:
    """Read the daily sequence columns for one morning — no event lists."""
    row = feat.iloc[int(index)]
    if region_id == "global":
        n = int(pd.to_numeric(row.get("seq_n_max"), errors="coerce") or 0)
        active = int(pd.to_numeric(row.get("seq_active_any"), errors="coerce") or 0) >= 1
        return {
            "n": n,
            "active": active,
            "shape": "mixed" if active else "none",
            "b_pos_delta": None,
            "traffic_light": "none",
            "pprime": None,
            "max_m": None,
            "status": "Active sequence" if active else "No active sequence",
        }
    n = int(pd.to_numeric(row.get(f"seq_n_{region_id}"), errors="coerce") or 0)
    active = int(pd.to_numeric(row.get(f"seq_active_{region_id}"), errors="coerce") or 0) >= 1
    b_delta = pd.to_numeric(row.get(f"seq_b_pos_delta_{region_id}"), errors="coerce")
    pprime = pd.to_numeric(row.get(f"seq_pprime_{region_id}"), errors="coerce")
    max_m = pd.to_numeric(row.get(f"seq_max_m_{region_id}"), errors="coerce")
    shape = str(row.get(f"seq_shape_{region_id}") or ("none" if not active else "mixed"))
    light = str(row.get(f"seq_tl_{region_id}") or "none")
    return {
        "n": n,
        "active": active,
        "shape": shape if active else "none",
        "b_pos_delta": None if not np.isfinite(b_delta) else float(b_delta),
        "traffic_light": light if active else "none",
        "pprime": None if not np.isfinite(pprime) else float(pprime),
        "max_m": None if not np.isfinite(max_m) else float(max_m),
        "status": "Active sequence" if active else "No active sequence",
    }


def sequence_frame(
    issue_times: pd.Series,
    catalogs: dict[str, pd.DataFrame] | None = None,
    root: Path | None = None,
    quakes: pd.DataFrame | None = None,
    sse=None,
) -> pd.DataFrame:
    """Daily as-of sequence vector per issued box. No event lists in the frame."""
    issue = pd.DatetimeIndex(parse_utc(issue_times))
    cats = load_sequence_catalogs(quakes=quakes, root=root, catalogs=catalogs)
    out = pd.DataFrame({"issue_time": issue})
    if sse is not None and len(np.asarray(sse)) == len(issue):
        out["seq_sse_hikurangi"] = pd.to_numeric(pd.Series(np.asarray(sse)), errors="coerce").to_numpy(dtype=float)
    else:
        out["seq_sse_hikurangi"] = np.full(len(issue), np.nan)
    n_by_box: dict[str, np.ndarray] = {}
    active_by_box: dict[str, np.ndarray] = {}
    for rid in BOX_IDS:
        catalog = cats.get(rid, pd.DataFrame())
        cols = {name: np.full(len(issue), np.nan) for name in SEQ_METRICS}
        cols["n"] = np.zeros(len(issue), dtype=float)
        cols["active"] = np.zeros(len(issue), dtype=float)
        cols["hours"] = np.zeros(len(issue), dtype=float)
        cols["n_1d"] = np.zeros(len(issue), dtype=float)
        cols["n_7d"] = np.zeros(len(issue), dtype=float)
        shapes = np.full(len(issue), "none", dtype=object)
        lights = np.full(len(issue), "none", dtype=object)
        last_end = None
        was_active = False
        if not catalog.empty:
            catalog = catalog.sort_values("available_at")
            avail = parse_utc(catalog["available_at"]).to_numpy(dtype="datetime64[ns]")
            obs = parse_utc(catalog["observed_at"]).to_numpy(dtype="datetime64[ns]")
            for i, stamp in enumerate(issue):
                t = np.datetime64(pd.Timestamp(stamp).to_datetime64())
                lo = t - np.timedelta64(LOOKBACK_DAYS, "D")
                sl = slice(int(np.searchsorted(avail, lo, side="left")), int(np.searchsorted(avail, t, side="right")))
                part = catalog.iloc[sl]
                part = part.loc[obs[sl] >= lo]
                cluster = active_cluster(part) if len(part) else part
                active = len(cluster) >= MIN_EVENTS
                if not active:
                    if was_active:
                        last_end = pd.Timestamp(stamp)
                    cols["days_since_prev"][i] = (
                        float((pd.Timestamp(stamp) - last_end) / pd.Timedelta(days=1))
                        if last_end is not None
                        else float("nan")
                    )
                    was_active = False
                    continue
                if not was_active:
                    cols["days_since_prev"][i] = (
                        float((pd.Timestamp(stamp) - last_end) / pd.Timedelta(days=1))
                        if last_end is not None
                        else float("nan")
                    )
                else:
                    cols["days_since_prev"][i] = 0.0
                was_active = True
                described = describe_sequence(cluster, rid, stamp, _background(catalog, cluster, stamp))
                for name in SEQ_METRICS:
                    if name == "days_since_prev":
                        continue
                    cols[name][i] = described.get(name, cols[name][i])
                shapes[i] = described.get("shape") or "none"
                lights[i] = described.get("traffic_light") or "none"
        for name in SEQ_METRICS:
            out[f"seq_{name}_{rid}"] = cols[name]
        out[f"seq_shape_{rid}"] = shapes
        out[f"seq_tl_{rid}"] = lights
        n_by_box[rid] = cols["n"]
        active_by_box[rid] = cols["active"]
    stacked_n = np.vstack([n_by_box[rid] for rid in BOX_IDS])
    stacked_a = np.vstack([active_by_box[rid] for rid in BOX_IDS])
    out["seq_n_max"] = stacked_n.max(axis=0)
    out["seq_n_any"] = stacked_n.sum(axis=0)
    out["seq_active_any"] = stacked_a.max(axis=0)
    out["seq_n_boxes"] = stacked_a.sum(axis=0)
    return out
