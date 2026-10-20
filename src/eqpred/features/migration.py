from __future__ import annotations

"""Point-in-time earthquake migration on issued boxes.

Clusters events a morning could already have seen, then fits two velocities
on the locked receiver plane: along-strike and up-dip. Deep slab events
(>70 km) are dropped so a Kermadec intermediate quake cannot fake an
upward crustal chain.

Constants are locked. Do not retune them on holdout hits.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.catalog import apply_available_at, load_comcat_versions
from eqpred.faults import FAULTS
from eqpred.ingest.run import try_latest_raw
from eqpred.regions import REGIONS, in_bbox, region_centroid_lon
from eqpred.stamp import parse_utc, utc_ts

LOOKBACK_DAYS = 14
CLUSTER_KM = 80.0
CLUSTER_DAYS = 7.0
MIN_EVENTS = 4
MAX_DEPTH_KM = 70.0
GEONET_MAG = 3.5
COMCAT_MAG = 4.5
BOX_IDS = ("hikurangi", "alpine", "kermadec")


def along_updip_km(
    lon: np.ndarray,
    lat: np.ndarray,
    depth_km: np.ndarray,
    lon0: float,
    lat0: float,
    strike: float,
    dip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project hypocenters onto the locked strike / up-dip axes (km)."""
    lat0r = np.radians(lat0)
    east = wrap180_arr(np.asarray(lon, dtype=float) - lon0) * 111.32 * np.cos(lat0r)
    north = (np.asarray(lat, dtype=float) - lat0) * 111.32
    depth = np.asarray(depth_km, dtype=float)
    strike_r = np.radians(strike)
    dip_r = np.radians(dip)
    along = east * np.sin(strike_r) + north * np.cos(strike_r)
    az = np.radians(strike + 90.0)
    down_h = east * np.sin(az) + north * np.cos(az)
    updip = -(down_h * np.sin(dip_r) + depth * np.cos(dip_r))
    return along, updip


def fault_plane_distance_km(
    lon: np.ndarray,
    lat: np.ndarray,
    depth_km: np.ndarray,
    lon0: float,
    lat0: float,
    strike: float,
    dip: float,
) -> np.ndarray:
    """Perpendicular distance to the locked receiver plane in kilometres."""
    lat0r = np.radians(lat0)
    east = wrap180_arr(np.asarray(lon, dtype=float) - lon0) * 111.32 * np.cos(lat0r)
    north = (np.asarray(lat, dtype=float) - lat0) * 111.32
    depth = np.asarray(depth_km, dtype=float)
    az = np.radians(strike + 90.0)
    down_h = east * np.sin(az) + north * np.cos(az)
    dip_r = np.radians(dip)
    # The plane descends along strike+90: depth = down_h * tan(dip).
    normal = depth * np.cos(dip_r) - down_h * np.sin(dip_r)
    return np.abs(normal)


def wrap180_arr(lon: np.ndarray) -> np.ndarray:
    return np.mod(lon + 180.0, 360.0) - 180.0


def _haversine_km(lon0: float, lat0: float, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    r = 6371.0
    lon0r, lat0r = np.radians(lon0), np.radians(lat0)
    lonr, latr = np.radians(lon), np.radians(lat)
    dlon = np.mod(lonr - lon0r + np.pi, 2 * np.pi) - np.pi
    dlat = latr - lat0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def slope_per_day(hours: np.ndarray, values: np.ndarray) -> float:
    t = np.asarray(hours, dtype=float) / 24.0
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x)
    if int(ok.sum()) < 2:
        return float("nan")
    t = t[ok] - t[ok].mean()
    x = x[ok] - x[ok].mean()
    den = float(np.dot(t, t))
    if den < 1e-8:
        return float("nan")
    return float(np.dot(t, x) / den)


def swarm_score(magnitudes: np.ndarray) -> float:
    mag = np.asarray(magnitudes, dtype=float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 2:
        return float("nan")
    spread = float(mag.max() - np.median(mag))
    if spread <= 0.8:
        return 1.0
    if spread >= 1.5:
        return 0.0
    return float((1.5 - spread) / 0.7)


def shape_label(score: float) -> str:
    if not np.isfinite(score):
        return "none"
    if score >= 0.66:
        return "swarm"
    if score <= 0.33:
        return "omori"
    return "mixed"


def active_cluster(
    events: pd.DataFrame,
    *,
    space_km: float = CLUSTER_KM,
    time_days: float = CLUSTER_DAYS,
) -> pd.DataFrame:
    """Single-linkage cluster grown from the latest visible event."""
    if events.empty:
        return events
    work = events.sort_values("observed_at").reset_index(drop=True)
    n = len(work)
    t = parse_utc(work["observed_at"]).to_numpy(dtype="datetime64[ns]")
    lon = work["lon"].to_numpy(dtype=float)
    lat = work["lat"].to_numpy(dtype=float)
    in_cluster = np.zeros(n, dtype=bool)
    in_cluster[-1] = True
    changed = True
    while changed:
        changed = False
        members = np.where(in_cluster)[0]
        for i in range(n):
            if in_cluster[i]:
                continue
            dt = np.abs(t[i] - t[members]) / np.timedelta64(1, "D")
            near_t = dt <= time_days
            if not np.any(near_t):
                continue
            dist = _haversine_km(float(lon[i]), float(lat[i]), lon[members], lat[members])
            if np.any(near_t & (dist <= space_km)):
                in_cluster[i] = True
                changed = True
    return work.loc[in_cluster].reset_index(drop=True)


def _prepare_catalog(
    frame: pd.DataFrame,
    minmag: float,
    versions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = apply_available_at(frame, versions=versions)
    work["observed_at"] = parse_utc(work["observed_at"])
    work["available_at"] = parse_utc(work["available_at"])
    work["magnitude"] = pd.to_numeric(work["magnitude"], errors="coerce")
    work["depth_km"] = pd.to_numeric(work["depth_km"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work = work.loc[
        (work["magnitude"] >= minmag)
        & work["lon"].notna()
        & work["lat"].notna()
        & (work["depth_km"].fillna(0) <= MAX_DEPTH_KM)
    ]
    return work


def load_migration_catalogs(root: Path | None = None) -> dict[str, pd.DataFrame]:
    """GeoNet for NZ boxes when present; ComCat for Kermadec / fallback."""
    versions = load_comcat_versions(root)
    comcat = _prepare_catalog(try_latest_raw("usgs.comcat", root), COMCAT_MAG, versions)
    geonet = _prepare_catalog(try_latest_raw("geonet.fdsn", root), GEONET_MAG)
    out: dict[str, pd.DataFrame] = {}
    for rid in BOX_IDS:
        region = REGIONS[rid]
        local_geo = geonet.loc[in_bbox(geonet, region)] if not geonet.empty else geonet
        local_com = comcat.loc[in_bbox(comcat, region)] if not comcat.empty else comcat
        if rid in ("hikurangi", "alpine") and not local_geo.empty:
            out[rid] = local_geo.copy()
        else:
            out[rid] = local_com.copy()
    return out


def measure_cluster(cluster: pd.DataFrame, region_id: str) -> dict:
    fault = FAULTS[region_id]
    region = REGIONS[region_id]
    lon0 = region_centroid_lon(region)
    lat0 = 0.5 * (float(region["min_lat"]) + float(region["max_lat"]))
    stamps = parse_utc(cluster["observed_at"])
    t0 = stamps.min()
    hours = (stamps - t0) / pd.Timedelta(hours=1)
    along, updip = along_updip_km(
        cluster["lon"].to_numpy(dtype=float),
        cluster["lat"].to_numpy(dtype=float),
        cluster["depth_km"].to_numpy(dtype=float),
        lon0,
        lat0,
        float(fault["strike"]),
        float(fault["dip"]),
    )
    lon = cluster["lon"].to_numpy(dtype=float)
    lat = cluster["lat"].to_numpy(dtype=float)
    depth = cluster["depth_km"].to_numpy(dtype=float)
    centroid_fault_km = fault_plane_distance_km(
        np.array([float(np.nanmean(lon))]),
        np.array([float(np.nanmean(lat))]),
        np.array([float(np.nanmean(depth))]),
        lon0,
        lat0,
        float(fault["strike"]),
        float(fault["dip"]),
    )[0]
    swarm = swarm_score(cluster["magnitude"].to_numpy(dtype=float))
    return {
        "n": int(len(cluster)),
        "along_km_d": slope_per_day(hours.to_numpy(dtype=float), along),
        "updip_km_d": slope_per_day(hours.to_numpy(dtype=float), updip),
        "fault_km": float(centroid_fault_km),
        "swarm": swarm,
        "shape": shape_label(swarm),
        "hours": hours.to_numpy(dtype=float),
        "along_km": along,
        "updip_km": updip,
        "depth_km": cluster["depth_km"].to_numpy(dtype=float),
        "magnitude": cluster["magnitude"].to_numpy(dtype=float),
        "lon": cluster["lon"].to_numpy(dtype=float),
        "lat": cluster["lat"].to_numpy(dtype=float),
        "observed_at": [pd.Timestamp(s).isoformat().replace("+00:00", "Z") for s in stamps],
    }


def visible_window(catalog: pd.DataFrame, issue: pd.Timestamp) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    lo = issue - pd.Timedelta(days=LOOKBACK_DAYS)
    return catalog.loc[(catalog["available_at"] <= issue) & (catalog["observed_at"] >= lo)]


def cluster_asof(catalog: pd.DataFrame, region_id: str, issue) -> dict:
    """Active cluster and velocities as of `issue`. Empty dict if too few events."""
    stamp = utc_ts(issue)
    window = visible_window(catalog, stamp)
    cluster = active_cluster(window)
    if len(cluster) < MIN_EVENTS:
        return {
            "region_id": region_id,
            "n": int(len(cluster)),
            "along_km_d": None,
            "updip_km_d": None,
            "swarm": None,
            "shape": "none",
            "events": [],
        }
    measured = measure_cluster(cluster, region_id)
    events = []
    for i in range(measured["n"]):
        events.append(
            {
                "observed_at": measured["observed_at"][i],
                "magnitude": float(measured["magnitude"][i]),
                "lon": float(measured["lon"][i]),
                "lat": float(measured["lat"][i]),
                "depth_km": float(measured["depth_km"][i]),
                "along_km": float(measured["along_km"][i]),
                "updip_km": float(measured["updip_km"][i]),
                "hours": float(measured["hours"][i]),
            }
        )
    return {
        "region_id": region_id,
        "n": measured["n"],
        "along_km_d": _finite_or_none(measured["along_km_d"]),
        "updip_km_d": _finite_or_none(measured["updip_km_d"]),
        "swarm": _finite_or_none(measured["swarm"]),
        "shape": measured["shape"],
        "events": events,
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def migration_frame(
    issue_times: pd.Series,
    catalogs: dict[str, pd.DataFrame] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Daily along-strike / up-dip velocities per issued box."""
    issue = pd.DatetimeIndex(parse_utc(issue_times))
    cats = catalogs if catalogs is not None else load_migration_catalogs(root)
    out = pd.DataFrame({"issue_time": issue})
    for rid in BOX_IDS:
        catalog = cats.get(rid, pd.DataFrame())
        along = np.full(len(issue), np.nan)
        updip = np.full(len(issue), np.nan)
        swarm = np.full(len(issue), np.nan)
        count = np.zeros(len(issue), dtype=int)
        if not catalog.empty:
            catalog = catalog.sort_values("available_at")
            avail = parse_utc(catalog["available_at"]).to_numpy(dtype="datetime64[ns]")
            obs = parse_utc(catalog["observed_at"]).to_numpy(dtype="datetime64[ns]")
            for i, stamp in enumerate(issue):
                t = np.datetime64(pd.Timestamp(stamp).to_datetime64())
                lo = t - np.timedelta64(LOOKBACK_DAYS, "D")
                sl = slice(
                    int(np.searchsorted(avail, lo, side="left")),
                    int(np.searchsorted(avail, t, side="right")),
                )
                part = catalog.iloc[sl]
                part = part.loc[obs[sl] >= lo]
                if len(part) < MIN_EVENTS:
                    continue
                cluster = active_cluster(part)
                count[i] = int(len(cluster))
                if len(cluster) < MIN_EVENTS:
                    continue
                measured = measure_cluster(cluster, rid)
                along[i] = measured["along_km_d"]
                updip[i] = measured["updip_km_d"]
                swarm[i] = measured["swarm"]
        out[f"mig_along_{rid}"] = along
        out[f"mig_updip_{rid}"] = updip
        out[f"mig_swarm_{rid}"] = swarm
        out[f"mig_n_{rid}"] = count
    return out
