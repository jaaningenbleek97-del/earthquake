from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from eqpred.catalog import reporting_delay
from eqpred.ingest.http import fetch_json
from eqpred.records import observation
from eqpred.regions import REGIONS, lon_spans
from eqpred.stamp import utc_ts

USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
LICENSE = "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"
# Same first-report clock as ComCat. Hits still score against the score catalog.
MINMAG = 4.5
CHUNK_DAYS = 90

ARCS = {
    "fdsn.japan": "japan",
    "fdsn.chile": "chile",
}


def parse_fdsn_geojson(payload: object, source: str, ingest_run_id: str) -> pd.DataFrame:
    """Flatten one USGS FDSN GeoJSON FeatureCollection."""
    if not isinstance(payload, dict):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for feat in payload.get("features") or []:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None, None]
        t_ms = props.get("time")
        mag = props.get("mag")
        if t_ms is None or mag is None:
            continue
        observed = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(t_ms) / 1000.0)
        lon, lat = coords[0], coords[1]
        depth = coords[2] if len(coords) > 2 else None
        rec = observation(
            source=source,
            observed_at=observed,
            available_at=observed + reporting_delay(float(mag)),
            geom={
                "kind": "point",
                "lon": float(lon) if lon is not None else 0.0,
                "lat": float(lat) if lat is not None else 0.0,
            },
            payload={
                "event_id": feat.get("id"),
                "magnitude": float(mag),
                "magnitude_type": props.get("magType"),
                "depth_km": float(depth) if depth is not None else None,
                "place": props.get("place"),
                "status": props.get("status"),
            },
            license=LICENSE,
            ingest_run_id=ingest_run_id,
        )
        geom = rec["geom"]
        payload_row = rec["payload"]
        rows.append(
            {
                "source": rec["source"],
                "observed_at": rec["observed_at"],
                "available_at": rec["available_at"],
                "license": rec["license"],
                "ingest_run_id": rec["ingest_run_id"],
                "event_id": payload_row.get("event_id"),
                "magnitude": payload_row.get("magnitude"),
                "magnitude_type": payload_row.get("magnitude_type"),
                "depth_km": payload_row.get("depth_km"),
                "lon": geom.get("lon"),
                "lat": geom.get("lat"),
                "place": payload_row.get("place"),
            }
        )
    return pd.DataFrame(rows)


def _bbox_params(region: dict) -> list[dict[str, str]]:
    """One or two USGS bbox queries when a box wraps the date line."""
    out = []
    for lo, hi in lon_spans(float(region["min_lon"]), float(region["max_lon"])):
        out.append(
            {
                "minlatitude": f"{region['min_lat']}",
                "maxlatitude": f"{region['max_lat']}",
                "minlongitude": f"{lo}",
                "maxlongitude": f"{hi}",
            }
        )
    return out


def _chunk(start: datetime, end: datetime, region: dict, minmag: float) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for box in _bbox_params(region):
        params = {
            "format": "geojson",
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": f"{minmag}",
            "orderby": "time",
            "limit": "20000",
            **box,
        }
        payload = fetch_json(f"{USGS}?{urlencode(params)}")
        if isinstance(payload, dict):
            features.extend(payload.get("features") or [])
    return features


def pull_fdsn_arc(
    start: datetime,
    end: datetime,
    ingest_run_id: str,
    source: str,
    minmag: float = MINMAG,
) -> pd.DataFrame:
    """USGS FDSN slice for a transfer arc. Not the score catalog."""
    if source not in ARCS:
        raise ValueError(f"unknown FDSN source {source}")
    region = REGIONS[ARCS[source]]
    parts: list[pd.DataFrame] = []
    cursor = utc_ts(start).to_pydatetime()
    stop = utc_ts(end).to_pydatetime()
    while cursor < stop:
        nxt = min(cursor + timedelta(days=CHUNK_DAYS), stop)
        print(f"  {source} {cursor.date()} -> {nxt.date()}", flush=True)
        feats = _chunk(cursor, nxt, region, minmag)
        if len(feats) >= 20000:
            print(f"  warning: {source} hit 20k cap for {cursor.date()}-{nxt.date()}", flush=True)
        frame = parse_fdsn_geojson({"features": feats}, source, ingest_run_id)
        if not frame.empty:
            parts.append(frame)
        cursor = nxt
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates("event_id").sort_values("observed_at").reset_index(drop=True)
