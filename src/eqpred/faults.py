from __future__ import annotations

# Locked receiver planes. Not an NSHM. Strike/dip/rake in degrees, Aki convention.
# Alpine: NE-striking dextral. Hikurangi / Kermadec: W-dipping megathrust.
FAULTS = {
    "hikurangi": {"strike": 45.0, "dip": 12.0, "rake": 90.0, "mu": 0.4},
    "alpine": {"strike": 55.0, "dip": 50.0, "rake": 180.0, "mu": 0.4},
    "kermadec": {"strike": 20.0, "dip": 20.0, "rake": 90.0, "mu": 0.4},
}

# Along-strike patches inside the issued boxes. Scoring still uses the parent
# box; these nodes carry neighbor stress so the model is not one rectangle.
SEGMENTS = {
    "hikurangi_south": {
        "parent": "hikurangi",
        "label": "Hikurangi south",
        "min_lon": 174.6,
        "max_lon": 179.9,
        "min_lat": -42.4,
        "max_lat": -40.6,
    },
    "hikurangi_central": {
        "parent": "hikurangi",
        "label": "Hikurangi central",
        "min_lon": 174.6,
        "max_lon": 179.9,
        "min_lat": -40.6,
        "max_lat": -38.8,
    },
    "hikurangi_north": {
        "parent": "hikurangi",
        "label": "Hikurangi north",
        "min_lon": 174.6,
        "max_lon": 179.9,
        "min_lat": -38.8,
        "max_lat": -37.0,
    },
    "alpine_south": {
        "parent": "alpine",
        "label": "Alpine south",
        "min_lon": 166.2,
        "max_lon": 169.0,
        "min_lat": -46.8,
        "max_lat": -41.3,
    },
    "alpine_central": {
        "parent": "alpine",
        "label": "Alpine central",
        "min_lon": 169.0,
        "max_lon": 171.8,
        "min_lat": -46.8,
        "max_lat": -41.3,
    },
    "alpine_north": {
        "parent": "alpine",
        "label": "Alpine north",
        "min_lon": 171.8,
        "max_lon": 174.5,
        "min_lat": -46.8,
        "max_lat": -41.3,
    },
    "kermadec_south": {
        "parent": "kermadec",
        "label": "Kermadec south",
        "min_lon": 177.0,
        "max_lon": -175.0,
        "min_lat": -37.0,
        "max_lat": -31.2,
    },
    "kermadec_north": {
        "parent": "kermadec",
        "label": "Kermadec north",
        "min_lon": 177.0,
        "max_lon": -175.0,
        "min_lat": -31.2,
        "max_lat": -25.5,
    },
}

# Directed coupling. Along-strike = 1, trench continuity = 0.6, transfer = 0.4.
GRAPH_EDGES = (
    ("hikurangi_south", "hikurangi_central", 1.0, "along"),
    ("hikurangi_central", "hikurangi_south", 1.0, "along"),
    ("hikurangi_central", "hikurangi_north", 1.0, "along"),
    ("hikurangi_north", "hikurangi_central", 1.0, "along"),
    ("alpine_south", "alpine_central", 1.0, "along"),
    ("alpine_central", "alpine_south", 1.0, "along"),
    ("alpine_central", "alpine_north", 1.0, "along"),
    ("alpine_north", "alpine_central", 1.0, "along"),
    ("kermadec_south", "kermadec_north", 1.0, "along"),
    ("kermadec_north", "kermadec_south", 1.0, "along"),
    ("hikurangi_north", "kermadec_south", 0.6, "trench"),
    ("kermadec_south", "hikurangi_north", 0.6, "trench"),
    ("alpine_north", "hikurangi_south", 0.4, "transfer"),
    ("hikurangi_south", "alpine_north", 0.4, "transfer"),
)


def segment_region(segment_id: str) -> dict:
    """Bbox dict that `in_bbox` understands."""
    rec = SEGMENTS[segment_id]
    return {
        "kind": "bbox",
        "name": segment_id,
        "label": rec["label"],
        "min_lon": rec["min_lon"],
        "max_lon": rec["max_lon"],
        "min_lat": rec["min_lat"],
        "max_lat": rec["max_lat"],
    }
