from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.stamp import parse_utc


def decluster(events: pd.DataFrame, space_km: float = 200.0, time_days: float = 10.0) -> pd.DataFrame:
    """Keep largest event in a space-time neighborhood. Simple, replayable."""
    ev = events.sort_values(["magnitude", "observed_at"], ascending=[False, True]).copy()
    ev["observed_at"] = parse_utc(ev["observed_at"])
    keep = np.ones(len(ev), dtype=bool)
    t = ev["observed_at"].to_numpy(dtype="datetime64[ns]")
    lon = ev["lon"].to_numpy(dtype=float)
    lat = ev["lat"].to_numpy(dtype=float)
    for i in range(len(ev)):
        if not keep[i]:
            continue
        dt = np.abs(t - t[i]) / np.timedelta64(1, "D")
        near_t = dt <= time_days
        dist = haversine_km(lon[i], lat[i], lon, lat)
        near = near_t & (dist <= space_km)
        near[i] = False
        keep[near] = False
    return ev.loc[keep].sort_values("observed_at")


def decluster_causal(
    events: pd.DataFrame,
    space_km: float = 200.0,
    time_days: float = 10.0,
) -> pd.DataFrame:
    """Drop aftershocks using only earlier events, so a feature never peeks ahead.

    `decluster` keeps the biggest event in a window, which needs the future.
    Here an event is an aftershock only if an earlier event nearby was at least
    as large. The parent must also have been reported no later than the child,
    so any morning that can see the child could have made the same call.
    """
    ev = events.sort_values("observed_at").copy()
    ev["observed_at"] = parse_utc(ev["observed_at"])
    if ev.empty:
        return ev
    t = ev["observed_at"].to_numpy(dtype="datetime64[ns]")
    lon = ev["lon"].to_numpy(dtype=float)
    lat = ev["lat"].to_numpy(dtype=float)
    mag = ev["magnitude"].to_numpy(dtype=float)
    if "available_at" in ev.columns:
        avail = parse_utc(ev["available_at"]).to_numpy(dtype="datetime64[ns]")
    else:
        avail = t
    window = np.timedelta64(int(round(time_days * 86400)), "s").astype("timedelta64[ns]")
    start = np.searchsorted(t, t - window, side="left")
    keep = np.ones(len(ev), dtype=bool)
    for i in range(len(ev)):
        lo = start[i]
        if lo >= i:
            continue
        bigger = (mag[lo:i] >= mag[i]) & (avail[lo:i] <= avail[i])
        if not bigger.any():
            continue
        idx = np.flatnonzero(bigger) + lo
        dist = haversine_km(lon[i], lat[i], lon[idx], lat[idx])
        if (dist <= space_km).any():
            keep[i] = False
    return ev.loc[keep]


def haversine_km(lon0: float, lat0: float, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    r = 6371.0
    lon0r, lat0r = np.radians(lon0), np.radians(lat0)
    lonr, latr = np.radians(lon), np.radians(lat)
    dlon = np.mod(lonr - lon0r + np.pi, 2 * np.pi) - np.pi
    dlat = latr - lat0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def fresh_hits(
    quakes: pd.DataFrame,
    region: dict,
    mag: float,
    days: pd.DatetimeIndex,
    horizon_days: int = 7,
) -> np.ndarray:
    """Next-horizon hit using only the first large shock in a causal cluster.

    Bidirectional `decluster` keeps the largest event in a ±10-day window, which
    needs the future: a later mainshock can un-label an earlier foreshock.
    Labels use `decluster_causal` so a morning never needs tomorrow's catalog
    to decide whether today's qualifying shock was "fresh."
    """
    from eqpred.eval.strategies import _window_hits

    work = quakes.loc[quakes["magnitude"] >= mag].copy()
    if region.get("kind") == "bbox":
        from eqpred.regions import in_bbox

        work = work.loc[in_bbox(work, region)]
    if work.empty:
        return np.zeros(len(days), dtype=bool)
    mains = decluster_causal(work, space_km=200.0, time_days=10.0)
    return _window_hits(mains, region, mag, days, horizon_days)
