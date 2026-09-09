from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from eqpred.catalog import reporting_delay
from eqpred.ingest.http import fetch_text
from eqpred.records import observation
from eqpred.stamp import utc_ts


def _parse_geonet_text(text: str) -> list[dict[str, Any]]:
    """GeoNet FDSN format=text. Header line starts with #."""
    rows: list[dict[str, Any]] = []
    header: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            header = [p.strip() for p in line[1:].split("|")]
            if len(header) < 6:
                header = [p.strip() for p in line[1:].split("\t")]
            continue
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 6:
            continue
        named = {header[i].lower(): parts[i] for i in range(min(len(header), len(parts)))} if header else {}
        try:
            observed = utc_ts(named.get("time") or parts[1]).to_pydatetime()
            lat = float(named.get("latitude") or parts[2])
            lon = float(named.get("longitude") or parts[3])
            depth = float(named.get("depth/km") or named.get("depth") or parts[4])
            mag = float(named.get("magnitude") or parts[6] if len(parts) > 6 else parts[5])
        except (TypeError, ValueError, IndexError):
            continue
        mag_type = named.get("magnitudetype") or named.get("magnitude type") or (parts[5] if len(parts) > 6 else None)
        rows.append(
            {
                "event_id": named.get("eventid") or parts[0],
                "observed_at": observed,
                "lat": lat,
                "lon": lon,
                "depth_km": depth,
                "magnitude": mag,
                "magnitude_type": mag_type,
                "place": named.get("eventtype") or "",
            }
        )
    return rows

GEONET = "https://service.geonet.org.nz/fdsnws/event/1/query"
LICENSE = "https://www.geonet.org.nz/data/types/eq_catalogue"
SOURCE = "geonet.fdsn"


def pull_geonet(
    start: datetime,
    end: datetime,
    ingest_run_id: str,
    minmag: float = 3.0,
) -> pd.DataFrame:
    """NZ FDSN catalog overlay. Hits still score against ComCat."""
    rows: list[dict[str, Any]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + timedelta(days=30), end)
        params = {
            "format": "text",
            "starttime": cursor.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": nxt.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": f"{minmag}",
            "orderby": "time",
        }
        print(f"  GeoNet {cursor.date()} -> {nxt.date()}", flush=True)
        text = fetch_text(f"{GEONET}?{urlencode(params)}")
        for feat in _parse_geonet_text(text):
            observed = feat["observed_at"]
            mag = feat["magnitude"]
            rec = observation(
                source=SOURCE,
                observed_at=observed,
                available_at=observed + reporting_delay(float(mag)),
                geom={"kind": "point", "lon": feat["lon"], "lat": feat["lat"]},
                payload={
                    "event_id": feat.get("event_id"),
                    "magnitude": float(mag),
                    "magnitude_type": feat.get("magnitude_type"),
                    "depth_km": feat.get("depth_km"),
                    "place": feat.get("place"),
                },
                license=LICENSE,
                ingest_run_id=ingest_run_id,
            )
            payload_flat = rec["payload"]
            geom = rec["geom"]
            rows.append(
                {
                    "source": rec["source"],
                    "observed_at": rec["observed_at"],
                    "available_at": rec["available_at"],
                    "license": rec["license"],
                    "ingest_run_id": rec["ingest_run_id"],
                    "event_id": payload_flat.get("event_id"),
                    "magnitude": payload_flat.get("magnitude"),
                    "magnitude_type": payload_flat.get("magnitude_type"),
                    "depth_km": payload_flat.get("depth_km"),
                    "lon": geom.get("lon"),
                    "lat": geom.get("lat"),
                    "place": payload_flat.get("place"),
                }
            )
        cursor = nxt
    return pd.DataFrame(rows)
