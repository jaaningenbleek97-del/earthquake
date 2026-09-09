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


def haversine_km(lon0: float, lat0: float, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    r = 6371.0
    lon0r, lat0r = np.radians(lon0), np.radians(lat0)
    lonr, latr = np.radians(lon), np.radians(lat)
    dlon = np.mod(lonr - lon0r + np.pi, 2 * np.pi) - np.pi
    dlat = latr - lat0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.clip(np.sqrt(a), 0, 1))
