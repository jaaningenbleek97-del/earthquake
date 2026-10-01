"""Days since the last repeating small quake on the Hikurangi interface.

One morning number. It is a clock, not another count of M5s. A repeat is
known only once both members have been reported. Issued p does not read it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.eval.decluster import haversine_km
from eqpred.regions import REGIONS, in_bbox
from eqpred.stamp import parse_utc

# Same-patch tolerance when we have locations and not waveforms.
RADIUS_KM = 10.0
MAG_MIN = 3.0
MAG_MAX = 5.0
MAG_TOL = 0.5
# Shorter gaps are aftershocks of the same burst, not a repeating patch.
GAP_MIN_DAYS = 30.0
# Plate-interface depths in this box. Crustal and deep-slab events stay out.
DEPTH_MIN_KM = 8.0
DEPTH_MAX_KM = 60.0
CELL_DEG = 0.12


def days_since_hikurangi_repeater(events: pd.DataFrame, issue: pd.Series) -> np.ndarray:
    """Days since the last small interface repeat that was already reported."""
    stamps = parse_utc(issue)
    out = np.full(len(stamps), np.nan)
    times = _repeater_known_at(events)
    if times.size == 0:
        return out
    ask = stamps.to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(times, ask, side="right") - 1
    ok = idx >= 0
    out[ok] = (ask[ok] - times[idx[ok]]) / np.timedelta64(1, "D")
    return out


def _repeater_known_at(events: pd.DataFrame) -> np.ndarray:
    """When each repeat became knowable, sorted. Empty if there are none."""
    work = _interface_events(events)
    if work.empty:
        return np.array([], dtype="datetime64[ns]")
    work = work.sort_values("observed_at")
    lat = work["lat"].to_numpy(dtype=float)
    lon = work["lon"].to_numpy(dtype=float)
    mag = work["magnitude"].to_numpy(dtype=float)
    obs = work["observed_at"].to_numpy(dtype="datetime64[ns]")
    avail = parse_utc(work["available_at"]).to_numpy(dtype="datetime64[ns]")
    cells: dict[tuple[int, int], list[int]] = {}
    known: list[np.datetime64] = []
    for i in range(len(work)):
        parents = _matching_parents(i, lat, lon, mag, obs, cells)
        if parents.size:
            parent_avail = avail[parents].min()
            known.append(max(avail[i], parent_avail))
        key = _cell(lat[i], lon[i])
        cells.setdefault(key, []).append(i)
    if not known:
        return np.array([], dtype="datetime64[ns]")
    return np.sort(np.asarray(known, dtype="datetime64[ns]"))


def _interface_events(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    need = {"lat", "lon", "magnitude", "observed_at", "available_at", "depth_km"}
    if not need.issubset(events.columns):
        return pd.DataFrame()
    work = events.loc[in_bbox(events, REGIONS["hikurangi"])].copy()
    if work.empty:
        return work
    work["magnitude"] = pd.to_numeric(work["magnitude"], errors="coerce")
    work["depth_km"] = pd.to_numeric(work["depth_km"], errors="coerce")
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work["observed_at"] = parse_utc(work["observed_at"])
    work["available_at"] = parse_utc(work["available_at"])
    keep = (
        work["magnitude"].ge(MAG_MIN)
        & work["magnitude"].lt(MAG_MAX)
        & work["depth_km"].ge(DEPTH_MIN_KM)
        & work["depth_km"].le(DEPTH_MAX_KM)
        & work["observed_at"].notna()
        & work["available_at"].notna()
        & work["lat"].notna()
        & work["lon"].notna()
    )
    return work.loc[keep]


def _cell(lat: float, lon: float) -> tuple[int, int]:
    return (int(np.floor(lat / CELL_DEG)), int(np.floor(lon / CELL_DEG)))


def _matching_parents(
    i: int,
    lat: np.ndarray,
    lon: np.ndarray,
    mag: np.ndarray,
    obs: np.ndarray,
    cells: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    here = _cell(lat[i], lon[i])
    chosen: list[int] = []
    for dlat in (-1, 0, 1):
        for dlon in (-1, 0, 1):
            for j in cells.get((here[0] + dlat, here[1] + dlon), ()):
                if abs(mag[j] - mag[i]) > MAG_TOL:
                    continue
                gap = (obs[i] - obs[j]) / np.timedelta64(1, "D")
                if gap < GAP_MIN_DAYS:
                    continue
                dist = float(haversine_km(lon[i], lat[i], np.array([lon[j]]), np.array([lat[j]]))[0])
                if dist <= RADIUS_KM:
                    chosen.append(j)
    return np.asarray(chosen, dtype=int)
