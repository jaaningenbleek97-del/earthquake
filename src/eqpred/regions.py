from __future__ import annotations

import pandas as pd

# Issued boxes. The old one-rectangle "new_zealand" card is gone — a quake
# in Fiordland is not a hit on Hikurangi.
REGIONS = {
    "global": {"kind": "named", "name": "global", "label": "global"},
    "hikurangi": {
        "kind": "bbox",
        "name": "hikurangi",
        "label": "Hikurangi",
        "min_lon": 174.6,
        "max_lon": 179.9,
        "min_lat": -42.4,
        "max_lat": -37.0,
    },
    "alpine": {
        "kind": "bbox",
        "name": "alpine_fault",
        "label": "Alpine Fault",
        "min_lon": 166.2,
        "max_lon": 174.5,
        "min_lat": -46.8,
        "max_lat": -41.3,
    },
    "kermadec": {
        "kind": "bbox",
        "name": "kermadec",
        "label": "Kermadec",
        # Crosses the date line (Raoul / trench sit near 178°W).
        "min_lon": 177.0,
        "max_lon": -175.0,
        "min_lat": -37.0,
        "max_lat": -25.5,
    },
    # Transfer boxes. Same 7-day question, not issued live cards.
    "japan": {
        "kind": "bbox",
        "name": "japan_trench",
        "label": "Japan",
        "issued": False,
        "min_lon": 128.0,
        "max_lon": 146.0,
        "min_lat": 30.0,
        "max_lat": 46.0,
    },
    "chile": {
        "kind": "bbox",
        "name": "chile_trench",
        "label": "Chile",
        "issued": False,
        "min_lon": -76.0,
        "max_lon": -66.0,
        "min_lat": -45.0,
        "max_lat": -17.0,
    },
}


def wrap180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def region_centroid_lat(region: dict) -> float:
    return 0.5 * (float(region["min_lat"]) + float(region["max_lat"]))


def region_centroid_lon(region: dict) -> float:
    lo = float(region["min_lon"])
    hi = float(region["max_lon"])
    if lo <= hi:
        return 0.5 * (lo + hi)
    span = (180.0 - lo) + (hi + 180.0)
    return wrap180(lo + span / 2.0)


def lon_spans(min_lon: float, max_lon: float) -> list[tuple[float, float]]:
    if min_lon <= max_lon:
        return [(min_lon, max_lon)]
    return [(min_lon, 180.0), (-180.0, max_lon)]


def in_bbox(frame: pd.DataFrame, region: dict) -> pd.Series:
    if region.get("kind") != "bbox":
        return pd.Series(True, index=frame.index)
    lat_ok = (frame["lat"] >= region["min_lat"]) & (frame["lat"] <= region["max_lat"])
    lon_ok = pd.Series(False, index=frame.index)
    for lo, hi in lon_spans(region["min_lon"], region["max_lon"]):
        lon_ok = lon_ok | ((frame["lon"] >= lo) & (frame["lon"] <= hi))
    return lat_ok & lon_ok


def is_issued_box(region: dict) -> bool:
    return region.get("kind") == "bbox" and region.get("issued", True)


def issued_boxes() -> list[dict]:
    return [r for r in REGIONS.values() if is_issued_box(r)]


def point_in_region(lat: float, lon: float, region: dict) -> bool:
    if region.get("kind") != "bbox":
        return True
    if lat < float(region["min_lat"]) or lat > float(region["max_lat"]):
        return False
    for lo, hi in lon_spans(float(region["min_lon"]), float(region["max_lon"])):
        if lo <= lon <= hi:
            return True
    return False


def in_any_box(frame: pd.DataFrame, regions: list[dict] | None = None) -> pd.Series:
    boxes = regions if regions is not None else issued_boxes()
    if not boxes:
        return pd.Series(False, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    for region in boxes:
        mask = mask | in_bbox(frame, region)
    return mask


def leaflet_rectangles(region: dict) -> list[dict]:
    """One or two lat-lon rectangles for a (possibly wrapping) bbox."""
    if region.get("kind") != "bbox":
        return []
    out = []
    for lo, hi in lon_spans(region["min_lon"], region["max_lon"]):
        out.append(
            {
                "min_lat": region["min_lat"],
                "max_lat": region["max_lat"],
                "min_lon": lo,
                "max_lon": hi,
                "name": region.get("name"),
                "label": region.get("label") or region.get("name"),
            }
        )
    return out
