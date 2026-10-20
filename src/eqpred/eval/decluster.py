from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import numpy as np
import pandas as pd

from eqpred.stamp import parse_utc

LABEL_V1 = 1
LABEL_V2 = 2
LABEL_NAMES = {
    LABEL_V1: "Version 1: a bigger quake in the same box within 200 km and 10 days",
    LABEL_V2: "Version 2: also a bigger quake anywhere within a window that grows with its size",
}
_LABEL_VERSION: ContextVar[int] = ContextVar("first_shock_label", default=LABEL_V1)


def current_label_version() -> int:
    return _LABEL_VERSION.get()


@contextmanager
def use_label_version(version: int) -> Iterator[None]:
    """Every fresh_hits call inside this block uses the given answer key."""
    token = _LABEL_VERSION.set(int(version))
    try:
        yield
    finally:
        _LABEL_VERSION.reset(token)


def uhrhammer_km(magnitude: np.ndarray | float) -> np.ndarray:
    """Aftershock distance window for a parent of this magnitude (Uhrhammer 1986)."""
    return np.exp(-1.024 + 0.804 * np.asarray(magnitude, dtype=float))


def uhrhammer_days(magnitude: np.ndarray | float) -> np.ndarray:
    """Aftershock time window for a parent of this magnitude (Uhrhammer 1986)."""
    return np.exp(-2.87 + 1.235 * np.asarray(magnitude, dtype=float))


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


def decluster_causal_v2(
    events: pd.DataFrame,
    parents: pd.DataFrame,
    space_km: float = 200.0,
    time_days: float = 10.0,
) -> pd.DataFrame:
    """Version 2 of the first-shock rule.

    An event is an aftershock if an earlier event at least as large, reported no
    later, sits within 200 km and 10 days, or within the Uhrhammer (1986) distance
    and time for that earlier event's magnitude. The earlier event may be outside
    the box; `parents` is the whole catalog above the target magnitude.
    """
    ev = events.sort_values("observed_at").copy()
    if ev.empty:
        return ev
    ev["observed_at"] = parse_utc(ev["observed_at"])
    par = parents.copy()
    par["observed_at"] = parse_utc(par["observed_at"])
    par = par.sort_values("observed_at")
    t_ev = ev["observed_at"].to_numpy(dtype="datetime64[ns]")
    t_par = par["observed_at"].to_numpy(dtype="datetime64[ns]")
    a_ev = parse_utc(ev["available_at"]).to_numpy(dtype="datetime64[ns]") if "available_at" in ev.columns else t_ev
    a_par = parse_utc(par["available_at"]).to_numpy(dtype="datetime64[ns]") if "available_at" in par.columns else t_par
    m_ev = ev["magnitude"].to_numpy(dtype=float)
    m_par = par["magnitude"].to_numpy(dtype=float)
    lon_ev, lat_ev = ev["lon"].to_numpy(dtype=float), ev["lat"].to_numpy(dtype=float)
    lon_par, lat_par = par["lon"].to_numpy(dtype=float), par["lat"].to_numpy(dtype=float)
    reach_days = float(max(time_days, float(np.max(uhrhammer_days(m_par))) if m_par.size else time_days))
    reach = np.timedelta64(int(np.ceil(reach_days * 86400)), "s").astype("timedelta64[ns]")
    keep = np.ones(len(ev), dtype=bool)
    for i in range(len(ev)):
        lo = np.searchsorted(t_par, t_ev[i] - reach, side="left")
        hi = np.searchsorted(t_par, t_ev[i], side="left")
        if hi <= lo:
            continue
        sl = slice(lo, hi)
        bigger = (m_par[sl] >= m_ev[i]) & (a_par[sl] <= a_ev[i])
        if not bigger.any():
            continue
        idx = np.flatnonzero(bigger) + lo
        dt = (t_ev[i] - t_par[idx]) / np.timedelta64(1, "D")
        dist = haversine_km(lon_ev[i], lat_ev[i], lon_par[idx], lat_par[idx])
        near = (dist <= space_km) & (dt <= time_days)
        scaled = (dist <= uhrhammer_km(m_par[idx])) & (dt <= uhrhammer_days(m_par[idx]))
        if (near | scaled).any():
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
    label_version: int | None = None,
) -> np.ndarray:
    """Next-horizon hit using only the first large shock in a causal cluster.

    Bidirectional `decluster` keeps the largest event in a ±10-day window, which
    needs the future: a later mainshock can un-label an earlier foreshock.
    Labels use `decluster_causal` so a morning never needs tomorrow's catalog
    to decide whether today's qualifying shock was "fresh."

    `label_version` picks the answer key; without it the surrounding
    `use_label_version` block decides, and the default is version 1.
    """
    from eqpred.eval.strategies import _window_hits

    version = int(label_version) if label_version is not None else current_label_version()
    pool = quakes.loc[quakes["magnitude"] >= mag].copy()
    work = pool
    if region.get("kind") == "bbox":
        from eqpred.regions import in_bbox

        work = pool.loc[in_bbox(pool, region)]
    if work.empty:
        return np.zeros(len(days), dtype=bool)
    if version >= LABEL_V2:
        mains = decluster_causal_v2(work, pool, space_km=200.0, time_days=10.0)
    else:
        mains = decluster_causal(work, space_km=200.0, time_days=10.0)
    return _window_hits(mains, region, mag, days, horizon_days)
