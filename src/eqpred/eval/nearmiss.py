from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.regions import in_bbox
from eqpred.stamp import parse_utc, utc_ts

# Locked grace envelope. Not tuned on holdout.
# ~one 1° cell, half a magnitude, two days late/early.
KM_GRACE = 150.0
TIME_GRACE_HOURS = 48.0
MAG_GRACE = 0.5
R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1 = np.radians(np.asarray(lat1, dtype=float))
    l1 = np.radians(np.asarray(lon1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    l2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = p2 - p1
    dlon = l2 - l1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _clamp_lon(lon: np.ndarray, min_lon: float, max_lon: float) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    width = (max_lon - min_lon) % 360.0
    origin = min_lon
    if width == 0.0 and min_lon == max_lon:
        return np.mod(np.full(lon.shape, min_lon) + 180.0, 360.0) - 180.0
    if min_lon <= max_lon:
        width = max_lon - min_lon
    offset = np.mod(lon - origin, 360.0)
    inside = offset <= width + 1e-12
    to_hi = offset - width
    to_lo = 360.0 - offset
    clamped = np.where(inside, offset, np.where(to_hi <= to_lo, width, 0.0))
    wrapped = np.mod(origin + clamped + 180.0, 360.0) - 180.0
    return wrapped


def dist_km_to_region(lat, lon, region: dict) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if region.get("kind") != "bbox":
        return np.zeros(lat.shape, dtype=float)
    min_lat = float(region["min_lat"])
    max_lat = float(region["max_lat"])
    inside = in_bbox(pd.DataFrame({"lat": lat, "lon": lon}), region).to_numpy()
    clat = np.clip(lat, min_lat, max_lat)
    clon = _clamp_lon(lon, float(region["min_lon"]), float(region["max_lon"]))
    dist = haversine_km(lat, lon, clat, clon)
    return np.where(inside, 0.0, dist)


def _credit(distance: float, grace: float) -> float:
    if distance <= 0:
        return 1.0
    if grace <= 0:
        return 0.0
    return float(max(0.0, 1.0 - float(distance) / float(grace)))


def _hours_off(observed, start, end) -> float:
    t = pd.Timestamp(observed)
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    if lo <= t < hi:
        return 0.0
    if t >= hi:
        return float((t - hi) / pd.Timedelta(hours=1))
    return float((lo - t) / pd.Timedelta(hours=1))


def combine_credits(space: float, time: float, mag: float) -> float:
    if space <= 0 or time <= 0 or mag <= 0:
        return 0.0
    return float((space * time * mag) ** (1.0 / 3.0))


def grade_event(region: dict, mag_min: float, start, end, quake: pd.Series) -> dict:
    mag = float(quake["magnitude"])
    hours = _hours_off(quake["observed_at"], start, end)
    km = float(dist_km_to_region([quake["lat"]], [quake["lon"]], region)[0])
    mag_short = max(0.0, float(mag_min) - mag)
    space = _credit(km, KM_GRACE)
    time = _credit(hours, TIME_GRACE_HOURS)
    mag_c = _credit(mag_short, MAG_GRACE)
    hit = bool(mag >= mag_min and hours <= 0 and km <= 0)
    score = 1.0 if hit else combine_credits(space, time, mag_c)
    return {
        "hit": hit,
        "near_miss": score,
        "miss_km": km,
        "miss_hours": hours,
        "miss_mag": mag_short,
        "near_magnitude": mag,
        "near_place": quake.get("place"),
        "near_event_id": quake.get("event_id"),
    }


def grade_card(card: dict, quakes: pd.DataFrame) -> dict:
    start = utc_ts(card["valid_from"])
    end = utc_ts(card["valid_to"])
    mag = float(card["magnitude_min"])
    region = card["region"]
    if quakes.empty:
        return {
            "hit": 0,
            "n_events": 0,
            "near_miss": 0.0,
            "miss_km": None,
            "miss_hours": None,
            "miss_mag": None,
            "near_magnitude": None,
            "near_place": None,
            "near_event_id": None,
        }
    pad = pd.Timedelta(hours=TIME_GRACE_HOURS)
    window = quakes.loc[
        (quakes["magnitude"] >= mag - MAG_GRACE)
        & (quakes["observed_at"] >= start - pad)
        & (quakes["observed_at"] < end + pad)
    ]
    if window.empty:
        return {
            "hit": 0,
            "n_events": 0,
            "near_miss": 0.0,
            "miss_km": None,
            "miss_hours": None,
            "miss_mag": None,
            "near_magnitude": None,
            "near_place": None,
            "near_event_id": None,
        }
    km = dist_km_to_region(window["lat"].to_numpy(), window["lon"].to_numpy(), region)
    near = window.loc[km <= KM_GRACE].copy()
    if near.empty:
        return {
            "hit": 0,
            "n_events": 0,
            "near_miss": 0.0,
            "miss_km": None,
            "miss_hours": None,
            "miss_mag": None,
            "near_magnitude": None,
            "near_place": None,
            "near_event_id": None,
        }
    graded = [grade_event(region, mag, start, end, row) for _, row in near.iterrows()]
    hits = [g for g in graded if g["hit"]]
    n_hit = len(hits)
    if hits:
        best = max(hits, key=lambda g: (g["near_magnitude"], -g["miss_km"]))
        best = {**best, "near_miss": 1.0, "miss_km": 0.0, "miss_hours": 0.0, "miss_mag": 0.0}
    else:
        best = max(graded, key=lambda g: (g["near_miss"], g["near_magnitude"], -g["miss_km"]))
    return {
        "hit": int(n_hit > 0),
        "n_events": int(n_hit),
        "near_miss": float(best["near_miss"]),
        "miss_km": None if best["miss_km"] is None else float(best["miss_km"]),
        "miss_hours": None if best["miss_hours"] is None else float(best["miss_hours"]),
        "miss_mag": None if best["miss_mag"] is None else float(best["miss_mag"]),
        "near_magnitude": best.get("near_magnitude"),
        "near_place": best.get("near_place"),
        "near_event_id": best.get("near_event_id"),
    }


def attach_observed(quakes: pd.DataFrame) -> pd.DataFrame:
    out = quakes.copy()
    out["observed_at"] = parse_utc(out["observed_at"])
    return out
