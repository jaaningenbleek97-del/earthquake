from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.records import iso

LICENSE = "computed-ephemeris-approx"
# Lightweight lunar/solar geometry (Meeus-style). Good enough for phase,
# distance, and a scalar tidal proxy. Not a substitute for skyfield later.


def _julian(dt: datetime) -> float:
    y, m, d = dt.year, dt.month, dt.day + dt.hour / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def _lunar_state(dt: datetime) -> tuple[float, float, float]:
    """Return (phase_deg, distance_re, declination_deg)."""
    jd = _julian(dt)
    t = (jd - 2451545.0) / 36525.0
    # Mean elongation, anomaly, etc. (degrees)
    d = (297.8501921 + 445267.1114034 * t) % 360
    m = (134.9633964 + 477198.8676313 * t) % 360
    f = (93.2720950 + 483202.0175233 * t) % 360
    sun_m = (357.5291092 + 35999.0502909 * t) % 360
    # Distance in Earth radii (approx)
    d_rad = math.radians(d)
    m_rad = math.radians(m)
    dist_km = 385000.56 - 20905.355 * math.cos(m_rad) - 3699.111 * math.cos(
        2 * d_rad - m_rad
    ) - 2955.968 * math.cos(2 * d_rad)
    dist_re = dist_km / 6378.14
    # Phase angle ~ elongation
    phase = (d + 180) % 360
    # Crude ecliptic latitude -> declination proxy
    lat = 5.128 * math.sin(math.radians(f))
    decl = lat  # good enough as a seasonal-ish lunar latitude proxy
    _ = sun_m
    return phase, dist_re, decl


def _tidal_proxy(dist_re: float, phase_deg: float) -> float:
    lunar = (60.3 / dist_re) ** 3
    # Spring/neap: new and full moons amplify.
    spring = 0.5 + 0.5 * math.cos(math.radians(2 * phase_deg))
    solar = 0.46
    return lunar * (1.0 + 0.4 * spring) + solar


def pull_geometry(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    day = start.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    stop = end.astimezone(timezone.utc)
    while day < stop:
        phase, dist, decl = _lunar_state(day)
        to_new = min(phase, 360.0 - phase)
        to_full = abs(phase - 180.0)
        near_node = abs(decl) < 1.8
        rows.append(
            {
                "source": "computed.geometry",
                "observed_at": iso(day),
                "available_at": iso(day),
                "license": LICENSE,
                "ingest_run_id": ingest_run_id,
                "lunar_phase_deg": phase,
                "lunar_distance_re": dist,
                "lunar_declination_deg": decl,
                "tidal_strain": _tidal_proxy(dist, phase),
                "near_new_or_full": float(min(to_new, to_full) < 20),
                "solar_eclipse_window": float(to_new < 18.0 and near_node),
                "lunar_eclipse_geom": float(to_full < 18.0 and near_node),
            }
        )
        day += timedelta(days=1)
    return pd.DataFrame(rows)
