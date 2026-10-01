from __future__ import annotations

"""Smithsonian / NCEI Holocene eruptions as a daily unrest flag.

Burns watches volcanic load as a sibling of the seismic cycle. This is an
event list with a short hold window, not a live VAAC scrape.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from eqpred.ingest.http import fetch_json
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "si.gvp"
LICENSE = "https://volcano.si.edu/database/"
NCEI_URL = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/volcanicevents"
GVP_WFS = (
    "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows"
    "?service=WFS&version=1.0.0&request=GetFeature"
    "&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Eruptions"
    "&outputFormat=application/json"
)
LATENCY = timedelta(days=2)
HOLD_DAYS = 14
# Kermadec / NZ / SW Pacific box plus a global VEI>=4 flag.
NZ_LAT = (-48.0, -25.0)
NZ_LON = (165.0, 185.0)


def _as_int(value: object) -> int | None:
    try:
        if value in (None, "", "None"):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        if value in (None, "", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_day(rec: dict[str, Any]) -> datetime | None:
    year = _as_int(
        rec.get("year")
        or rec.get("StartDateYear")
        or rec.get("StartYear")
        or rec.get("startYear")
    )
    if year is None or year < 1800:
        return None
    month = _as_int(
        rec.get("month")
        or rec.get("StartDateMonth")
        or rec.get("StartMonth")
        or rec.get("startMonth")
    ) or 1
    day = _as_int(
        rec.get("day")
        or rec.get("StartDateDay")
        or rec.get("StartDay")
        or rec.get("startDay")
    ) or 1
    month = min(max(month, 1), 12)
    day = min(max(day, 1), 28 if month == 2 else 30 if month in {4, 6, 9, 11} else 31)
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_volcano_events(rows: list[Any]) -> pd.DataFrame:
    """One row per dated eruption onset."""
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        props = rec.get("properties") if "properties" in rec else rec
        if not isinstance(props, dict):
            continue
        when = _event_day(props)
        if when is None:
            continue
        vei = _as_float(props.get("vei") or props.get("VEI") or props.get("ExplosivityIndexMax"))
        lat = _as_float(props.get("latitude") or props.get("Latitude") or props.get("latitudeNorth"))
        lon = _as_float(props.get("longitude") or props.get("Longitude") or props.get("longitudeEast"))
        geom = rec.get("geometry") if "geometry" in rec else None
        if (lat is None or lon is None) and isinstance(geom, dict):
            coords = geom.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lon = _as_float(coords[0]) if lon is None else lon
                lat = _as_float(coords[1]) if lat is None else lat
        if lon is not None and lon > 180:
            lon -= 360.0
        near_nz = False
        if lat is not None and lon is not None:
            lon_w = lon if lon >= 0 else lon + 360.0
            near_nz = NZ_LAT[0] <= lat <= NZ_LAT[1] and NZ_LON[0] <= lon_w <= NZ_LON[1]
        out.append(
            {
                "observed_at": iso(when),
                "volcano_vei": vei,
                "volcano_lat": lat,
                "volcano_lon": lon,
                "volcano_nz": 1.0 if near_nz else 0.0,
            }
        )
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def _daily_flags(events: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    onset = pd.to_datetime(events["observed_at"], utc=True).dt.floor("D")
    days = pd.date_range(utc_ts(start), utc_ts(end), freq="D", inclusive="left")
    if days.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for day in days:
        window = (onset >= day - pd.Timedelta(days=HOLD_DAYS - 1)) & (onset <= day)
        match = events.loc[window]
        vei = pd.to_numeric(match["volcano_vei"], errors="coerce") if not match.empty else pd.Series(dtype=float)
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(day.to_pydatetime()),
                "available_at": iso((day + pd.Timedelta(days=LATENCY.days)).to_pydatetime()),
                "license": LICENSE,
                "volcano_eruption": 1.0 if not match.empty else 0.0,
                "volcano_vei_max": None if match.empty or not vei.notna().any() else float(vei.max()),
                "volcano_vei4": 1.0 if (vei >= 4).any() else 0.0,
                "volcano_nz": 1.0 if (not match.empty and float(match["volcano_nz"].max()) > 0) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _extract_rows(payload: object) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("items", "features", "data", "volcanicEvents"):
        block = payload.get(key)
        if isinstance(block, list):
            return block
    return []


def pull_volcano(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    payload: object | None = None
    try:
        payload = fetch_json(
            f"{NCEI_URL}?minYear={utc_ts(start).year}&maxYear={utc_ts(end).year}",
            timeout=90,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  si.gvp NCEI skipped: {exc}")
    rows = _extract_rows(payload) if payload is not None else []
    if not rows:
        try:
            payload = fetch_json(GVP_WFS, timeout=120)
            rows = _extract_rows(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"  si.gvp WFS skipped: {exc}")
            return pd.DataFrame()
    events = parse_volcano_events(rows)
    if events.empty:
        return pd.DataFrame()
    out = _daily_flags(events, start, end)
    if out.empty:
        return out
    out["ingest_run_id"] = ingest_run_id
    return out
