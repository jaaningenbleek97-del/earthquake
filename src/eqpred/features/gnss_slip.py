"""Extra GPS slip: motion beyond the steady plate speed.

Nevada positions are millimetres. MIDAS speeds are millimetres per year.
GeoNet displacements stay in their own columns and are not pasted onto Nevada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.features.gnss_strain import _network_strain
from eqpred.ingest.run import try_latest_raw
from eqpred.regions import REGIONS, point_in_region
from eqpred.stamp import parse_utc

WINDOWS = (7, 30)
YEAR_DAYS = 365.25


def known_plate_rate(position: np.ndarray, published: np.ndarray, lookback: int = 1095) -> np.ndarray:
    """Millimetres per year known that morning.

    The published MIDAS speed is used once that file is in hand. Before then the
    speed is this station's own change over the previous three years.
    """
    pos = np.asarray(position, dtype=float)
    pub = np.asarray(published, dtype=float)
    earlier = np.roll(pos, lookback)
    trailing = (pos - earlier) * (YEAR_DAYS / float(lookback))
    trailing[:lookback] = np.nan
    trailing[~np.isfinite(pos) | ~np.isfinite(earlier)] = np.nan
    return np.where(np.isfinite(pub), pub, trailing)


def extra_slip(position: np.ndarray, rate_mm_yr: np.ndarray, window: int) -> np.ndarray:
    """Millimetres moved over `window` days, after subtracting the steady plate speed."""
    pos = np.asarray(position, dtype=float)
    rate = np.asarray(rate_mm_yr, dtype=float)
    delta = pos - np.roll(pos, window)
    delta[:window] = np.nan
    expected = rate * (float(window) / YEAR_DAYS)
    out = delta - expected
    out[~np.isfinite(rate)] = np.nan
    return out


def gnss_slip_columns(feat: pd.DataFrame, root=None) -> pd.DataFrame:
    """Attach extra slip, wider fault stretch, and the GeoNet tail."""
    if feat.empty or "issue_time" not in feat.columns:
        return feat
    days = pd.DatetimeIndex(parse_utc(feat["issue_time"]))
    extra = gnss_slip_frame(days, root)
    drop = [col for col in extra.columns if col != "issue_time" and col in feat.columns]
    work = feat.drop(columns=drop) if drop else feat
    return work.merge(extra, on="issue_time", how="left")


def gnss_slip_frame(days: pd.DatetimeIndex, root=None) -> pd.DataFrame:
    """Columns to add to the daily table. Raw Nevada positions are not copied in."""
    from eqpred.features.build import _asof_available
    from eqpred.ingest.gnss import ngl_station_coords

    days = pd.DatetimeIndex(days)
    n = len(days)
    columns: dict[str, np.ndarray] = {}
    ngl = try_latest_raw("ngl.gnss", root)
    midas = try_latest_raw("ngl.midas", root)
    rates = _rates(midas, days, root)
    coords = ngl_station_coords()
    positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not ngl.empty and "site" in ngl.columns:
        for n_site, (site, part) in enumerate(ngl.groupby(ngl["site"].astype(str).str.upper()), start=1):
            slug = str(site).lower()
            aligned = _asof_available(part, days, ["east_mm", "north_mm"])
            east = pd.to_numeric(aligned["east_mm"], errors="coerce").to_numpy(dtype=float)
            north = pd.to_numeric(aligned["north_mm"], errors="coerce").to_numpy(dtype=float)
            positions[slug] = (east, north)
            rate_e, rate_n = rates.get(str(site).upper(), (np.full(n, np.nan), np.full(n, np.nan)))
            rate_e = known_plate_rate(east, rate_e)
            rate_n = known_plate_rate(north, rate_n)
            for window in WINDOWS:
                columns[f"gnss_{slug}_east_extra_{window}d"] = extra_slip(east, rate_e, window)
                columns[f"gnss_{slug}_north_extra_{window}d"] = extra_slip(north, rate_n, window)
            if n_site % 40 == 0:
                print(f"  extra slip {n_site} stations", flush=True)
    for name, series in _wide_strain(days, positions, coords).items():
        columns[name] = series
    for name, series in _geonet_tail(days, root).items():
        columns[name] = series
    out = pd.DataFrame({"issue_time": days, **columns})
    keep = ["issue_time"]
    for name in out.columns:
        if name == "issue_time":
            continue
        values = pd.to_numeric(out[name], errors="coerce")
        finite = int(np.isfinite(values.to_numpy(dtype=float)).sum())
        if finite >= 20 or (name.startswith("geonet_") and finite > 0):
            keep.append(name)
    return out[keep]


def _rates(midas: pd.DataFrame, days: pd.DatetimeIndex, root) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from eqpred.features.build import _asof_available

    _ = root
    found: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if midas is None or midas.empty or "site" not in midas.columns:
        return found
    for site, part in midas.groupby(midas["site"].astype(str).str.upper()):
        aligned = _asof_available(part, days, ["east_mm_yr", "north_mm_yr"])
        found[str(site)] = (
            pd.to_numeric(aligned["east_mm_yr"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(aligned["north_mm_yr"], errors="coerce").to_numpy(dtype=float),
        )
    return found


def _wide_strain(
    days: pd.DatetimeIndex,
    positions: dict[str, tuple[np.ndarray, np.ndarray]],
    coords: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Stretch on Hikurangi and the Alpine Fault using every station in the box."""
    extra: dict[str, np.ndarray] = {}
    n = len(days)
    for region_id in ("hikurangi", "alpine"):
        sites: dict[str, tuple[float, float]] = {}
        data: dict[str, np.ndarray] = {}
        for site, (lon, lat) in coords.items():
            slug = site.lower()
            if slug not in positions:
                continue
            if not point_in_region(lat, lon, REGIONS[region_id]):
                continue
            sites[slug] = (lon, lat)
            east, north = positions[slug]
            data[f"gnss_{slug}_east_mm"] = east
            data[f"gnss_{slug}_north_mm"] = north
        frame = pd.DataFrame(data, index=range(n))
        frame.insert(0, "issue_time", days)
        print(f"  wide strain {region_id}: {len(sites)} stations", flush=True)
        dil, shear, n_used = _network_strain(frame, sites, 30) if len(sites) >= 2 else (
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.zeros(n),
        )
        extra[f"gnss_strain_{region_id}_wide_dilatation_30d"] = dil
        extra[f"gnss_strain_{region_id}_wide_shear_30d"] = shear
        extra[f"gnss_strain_{region_id}_wide_nsites_30d"] = n_used
    return extra


def _geonet_tail(days: pd.DatetimeIndex, root) -> dict[str, np.ndarray]:
    """GeoNet's own displacements. Not subtracted from the Nevada plate speed."""
    n = len(days)
    empty = {
        "geonet_nz_east_mm": np.full(n, np.nan),
        "geonet_nz_north_mm": np.full(n, np.nan),
        "geonet_nz_east_7d": np.full(n, np.nan),
        "geonet_nz_north_7d": np.full(n, np.nan),
        "geonet_nzmean_east_7d": np.full(n, np.nan),
        "geonet_nzmean_north_7d": np.full(n, np.nan),
    }
    geonet = try_latest_raw("geonet.gnss", root)
    if geonet.empty or "site" not in geonet.columns:
        return empty
    sites = geonet["site"].astype(str).str.lower()
    out = dict(empty)
    mean = _geonet_daily(geonet.loc[sites != "nzmean"], days)
    if mean is not None:
        east, north = mean
        out["geonet_nz_east_mm"] = east
        out["geonet_nz_north_mm"] = north
        out["geonet_nz_east_7d"] = pd.Series(east).diff(7).to_numpy(dtype=float)
        out["geonet_nz_north_7d"] = pd.Series(north).diff(7).to_numpy(dtype=float)
    older = _geonet_daily(geonet.loc[sites == "nzmean"], days)
    if older is not None:
        east, north = older
        out["geonet_nzmean_east_7d"] = pd.Series(east).diff(7).to_numpy(dtype=float)
        out["geonet_nzmean_north_7d"] = pd.Series(north).diff(7).to_numpy(dtype=float)
    return out


def _geonet_daily(work: pd.DataFrame, days: pd.DatetimeIndex):
    from eqpred.features.build import _asof_available

    if work.empty or "east_mm" not in work.columns or "north_mm" not in work.columns:
        return None
    part = work.copy()
    part["observed_at"] = parse_utc(part["observed_at"])
    part["available_at"] = parse_utc(part["available_at"])
    daily = (
        part.assign(_day=part["observed_at"].dt.floor("D"))
        .groupby("_day", as_index=False)
        .agg(
            east_mm=("east_mm", "mean"),
            north_mm=("north_mm", "mean"),
            available_at=("available_at", "min"),
        )
        .rename(columns={"_day": "observed_at"})
    )
    aligned = _asof_available(daily, days, ["east_mm", "north_mm"])
    east = pd.to_numeric(aligned["east_mm"], errors="coerce").to_numpy(dtype=float)
    north = pd.to_numeric(aligned["north_mm"], errors="coerce").to_numpy(dtype=float)
    return east, north
