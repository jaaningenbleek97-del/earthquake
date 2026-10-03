"""Horizontal strain rate from GNSS site positions already on the daily table.

Least-squares dilatation and shear on each named network, plus the 30-day
baseline extension between a couple of short pairs. Positions are millimetres
east/north; strain is dimensionless (nanostrain = 1e-9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# site slug (as gnss_{slug}_east_mm) -> (lon, lat)
NETWORKS: dict[str, dict[str, tuple[float, float]]] = {
    "nz": {
        "auck": (174.8344, -36.6028),
        "wgtn": (174.8058, -41.3235),
        "lytt": (172.7228, -43.6056),
        "dund": (170.5972, -45.8836),
    },
    "hikurangi": {
        "auck": (174.8344, -36.6028),
        "wgtn": (174.8058, -41.3235),
        "raul": (-177.9289, -29.2456),
    },
    "alpine": {
        "wgtn": (174.8058, -41.3235),
        "lytt": (172.7228, -43.6056),
        "dund": (170.5972, -45.8836),
    },
    "japan": {
        "tskb": (140.0876, 36.1056),
        "usud": (138.3620, 36.1331),
        "mizu": (141.1328, 39.1352),
    },
    "chile": {
        "antc": (-71.5328, -37.3386),
        "sant": (-70.6689, -33.1500),
        "iqqe": (-70.1319, -20.2739),
    },
}

PAIRS: dict[str, tuple[str, str]] = {
    "hikurangi": ("auck", "wgtn"),
    "alpine": ("lytt", "dund"),
    "kermadec": ("wgtn", "raul"),
}

WINDOWS = (30, 90)
R_EARTH_M = 6_371_000.0


def gnss_strain_columns(feat: pd.DataFrame) -> pd.DataFrame:
    """Add strain-rate columns in place-friendly fashion; returns a new frame."""
    work = feat
    extra: dict[str, np.ndarray] = {}
    n = len(work)
    for net, sites in NETWORKS.items():
        for days in WINDOWS:
            dil, shear, n_used = _network_strain(work, sites, days)
            extra[f"gnss_strain_{net}_dilatation_{days}d"] = dil
            extra[f"gnss_strain_{net}_shear_{days}d"] = shear
            extra[f"gnss_strain_{net}_nsites_{days}d"] = n_used
    for name, (a, b) in PAIRS.items():
        extra[f"gnss_strain_{name}_extension_30d"] = _pair_extension(work, a, b, 30)
    if extra:
        work = pd.concat([work, pd.DataFrame(extra, index=work.index)], axis=1)
    else:
        _ = n
    return work


def _east_north(feat: pd.DataFrame, slug: str) -> tuple[np.ndarray, np.ndarray] | None:
    e_col, n_col = f"gnss_{slug}_east_mm", f"gnss_{slug}_north_mm"
    if e_col not in feat.columns or n_col not in feat.columns:
        return None
    east = pd.to_numeric(feat[e_col], errors="coerce").to_numpy(dtype=float)
    north = pd.to_numeric(feat[n_col], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(east).any() and np.isfinite(north).any()):
        return None
    return east, north


def _network_strain(
    feat: pd.DataFrame,
    sites: dict[str, tuple[float, float]],
    days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(feat)
    dil = np.full(n, np.nan)
    shear = np.full(n, np.nan)
    n_used = np.zeros(n, dtype=float)
    series: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for slug, (lon, lat) in sites.items():
        en = _east_north(feat, slug)
        if en is None:
            continue
        series.append((lon, lat, en[0], en[1]))
    if len(series) < 2:
        return dil, shear, n_used

    origin_lon = float(np.mean([s[0] for s in series]))
    origin_lat = float(np.mean([s[1] for s in series]))
    xy = np.array([_en_m(lon, lat, origin_lon, origin_lat) for lon, lat, _, _ in series], dtype=float)
    de = np.column_stack([s[2] for s in series])  # mm
    dn = np.column_stack([s[3] for s in series])
    dde = de - np.roll(de, days, axis=0)
    ddn = dn - np.roll(dn, days, axis=0)
    dde[:days] = np.nan
    ddn[:days] = np.nan
    # mm over `days` → strain increment (dimensionless), then per-year rate in nanostrain/yr
    year_scale = 365.25 / float(days)
    for i in range(days, n):
        ok = np.isfinite(dde[i]) & np.isfinite(ddn[i])
        k = int(ok.sum())
        if k < 2:
            continue
        x = xy[ok, 0]
        y = xy[ok, 1]
        ue = dde[i, ok] * 1e-3
        un = ddn[i, ok] * 1e-3
        g = np.zeros((2 * k, 3), dtype=float)
        rhs = np.empty(2 * k, dtype=float)
        g[0::2, 0] = x
        g[0::2, 1] = y
        rhs[0::2] = ue
        g[1::2, 1] = x
        g[1::2, 2] = y
        rhs[1::2] = un
        try:
            coef, *_ = np.linalg.lstsq(g, rhs, rcond=None)
        except np.linalg.LinAlgError:
            continue
        exx, exy, eyy = float(coef[0]), float(coef[1]), float(coef[2])
        dil[i] = (exx + eyy) * year_scale * 1e9
        shear[i] = float(np.sqrt(((exx - eyy) / 2.0) ** 2 + exy**2)) * year_scale * 1e9
        n_used[i] = float(ok.sum())
    return dil, shear, n_used


def _pair_extension(feat: pd.DataFrame, a: str, b: str, days: int) -> np.ndarray:
    ea = _east_north(feat, a)
    eb = _east_north(feat, b)
    n = len(feat)
    out = np.full(n, np.nan)
    if ea is None or eb is None:
        return out
    pos = None
    for net in NETWORKS.values():
        if a in net and b in net:
            pos = (net[a], net[b])
            break
    if pos is None:
        return out
    (lon0, lat0), (lon1, lat1) = pos
    dx, dy = _en_m(lon1, lat1, lon0, lat0)
    baseline = float(np.hypot(dx, dy))
    if baseline < 1_000.0:
        return out
    ux, uy = dx / baseline, dy / baseline
    dae = ea[0] - np.roll(ea[0], days)
    dan = ea[1] - np.roll(ea[1], days)
    dbe = eb[0] - np.roll(eb[0], days)
    dbn = eb[1] - np.roll(eb[1], days)
    # relative displacement of B minus A, projected on baseline, metres
    rel_e = (dbe - dae) * 1e-3
    rel_n = (dbn - dan) * 1e-3
    ext = (rel_e * ux + rel_n * uy) / baseline
    ext[:days] = np.nan
    year_scale = 365.25 / float(days)
    return ext * year_scale * 1e9


def _en_m(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    dlon = np.radians(((lon - lon0 + 180.0) % 360.0) - 180.0)
    dlat = np.radians(lat - lat0)
    east = R_EARTH_M * np.cos(np.radians(0.5 * (lat + lat0))) * dlon
    north = R_EARTH_M * dlat
    return float(east), float(north)
