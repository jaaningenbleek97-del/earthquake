from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from eqpred.catalog import reporting_delay
from eqpred.ingest.http import fetch_json
from eqpred.records import observation

COMCAT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
LICENSE = "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"


def _events_chunk(start: datetime, end: datetime, minmag: float) -> list[dict[str, Any]]:
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": f"{minmag}",
        "orderby": "time",
        "limit": "20000",
    }
    payload = fetch_json(f"{COMCAT}?{urlencode(params)}")
    assert isinstance(payload, dict)
    return list(payload.get("features") or [])


def pull_comcat(
    start: datetime,
    end: datetime,
    minmag: float,
    ingest_run_id: str,
) -> pd.DataFrame:
    """Fetch ComCat in half-year chunks so we stay under the 20k cap."""
    rows: list[dict[str, Any]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + timedelta(days=180), end)
        print(f"  ComCat {cursor.date()} -> {nxt.date()}")
        chunk = _events_chunk(cursor, nxt, minmag)
        if len(chunk) >= 20000:
            print(f"  warning: hit 20k cap for {cursor.date()}-{nxt.date()}; raise minmag or shrink chunks")
        for feat in chunk:
            props = feat.get("properties") or {}
            coords = (feat.get("geometry") or {}).get("coordinates") or [None, None, None]
            t_ms = props.get("time")
            if t_ms is None:
                continue
            observed = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            mag = props.get("mag")
            if mag is None:
                continue
            lon, lat, depth = coords[0], coords[1], coords[2] if len(coords) > 2 else None
            rec = observation(
                source="usgs.comcat",
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
            rows.append(_flatten(rec))
        cursor = nxt
    return pd.DataFrame(rows)


def _flatten(rec: dict[str, Any]) -> dict[str, Any]:
    payload = rec["payload"]
    geom = rec["geom"]
    return {
        "source": rec["source"],
        "observed_at": rec["observed_at"],
        "available_at": rec["available_at"],
        "license": rec["license"],
        "ingest_run_id": rec["ingest_run_id"],
        "event_id": payload.get("event_id"),
        "magnitude": payload.get("magnitude"),
        "magnitude_type": payload.get("magnitude_type"),
        "depth_km": payload.get("depth_km"),
        "lon": geom.get("lon"),
        "lat": geom.get("lat"),
        "place": payload.get("place"),
    }
