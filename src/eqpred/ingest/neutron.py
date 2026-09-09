from __future__ import annotations

"""Daily neutron-monitor counts from the Oulu Cosmic Ray Station.

Replayable CGI. Corrected counts are already adjusted for pressure and
efficiency. A new leftover family — not another OMNI transform.
"""

from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

QUERY_URL = "https://cosmicrays.oulu.fi/webform/onlinequery.cgi"
LICENSE = "https://cosmicrays.oulu.fi/readme.html"
LATENCY = timedelta(days=3)
SOURCE = "oulu.neutron"

# Oulu since 1964. DOMC (Concordia, Antarctica) is the southern complement.
STATIONS = (
    ("OULU", "neutron_oulu"),
    ("DOMC", "neutron_domc"),
    ("MOSC", "neutron_mosc"),
    ("NEWK", "neutron_newk"),
    ("SOPO", "neutron_sopo"),
    ("THUL", "neutron_thul"),
)
CHUNK_YEARS = 5


def _query_url(station: str, start: datetime, end: datetime) -> str:
    params = {
        "station": station,
        "startday": f"{start.day:02d}",
        "startmonth": f"{start.month:02d}",
        "startyear": str(start.year),
        "starttime": "00:00",
        "endday": f"{end.day:02d}",
        "endmonth": f"{end.month:02d}",
        "endyear": str(end.year),
        "endtime": "23:30",
        "resolution": "1440",
        "outputmode": "csv",
        "senddata": "on",
    }
    return f"{QUERY_URL}?{urlencode(params)}"


def _as_count(value: str) -> float | None:
    token = value.strip()
    if not token or token in {"nan", "NaN", "NA", "-"}:
        return None
    try:
        num = float(token)
    except ValueError:
        return None
    if not (0.0 < num < 1_000_000.0):
        return None
    return num


def parse_oulu_csv(text: str, prefix: str) -> pd.DataFrame:
    """One row per UTC day: corrected count, uncorrected count, pressure."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.lower().startswith("timestamp"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            observed = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            continue
        uncorr = _as_count(parts[2]) if len(parts) > 2 else None
        corr = _as_count(parts[3])
        pressure = None
        if len(parts) > 4:
            try:
                pressure = float(parts[4])
                if not (300.0 < pressure < 1100.0):
                    pressure = None
            except ValueError:
                pressure = None
        if corr is None:
            continue
        row: dict[str, object] = {
            "source": SOURCE,
            "observed_at": iso(observed.replace(hour=0, minute=0, second=0, microsecond=0)),
            "available_at": iso(observed.replace(hour=0, minute=0, second=0, microsecond=0) + LATENCY),
            "license": LICENSE,
            f"{prefix}_count": corr,
        }
        if uncorr is not None:
            row[f"{prefix}_uncorrected"] = uncorr
        if pressure is not None:
            row[f"{prefix}_pressure_mbar"] = pressure
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def _year_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        nxt = datetime(cursor.year + CHUNK_YEARS, 1, 1, tzinfo=timezone.utc)
        stop = min(nxt, end)
        chunks.append((cursor, stop))
        cursor = stop
    return chunks


def pull_neutron(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_utc = utc_ts(start)
    end_utc = utc_ts(end)
    for station, prefix in STATIONS:
        station_frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in _year_chunks(start_utc, end_utc):
            inclusive_end = chunk_end - timedelta(seconds=1)
            if inclusive_end < chunk_start:
                continue
            url = _query_url(station, chunk_start, inclusive_end)
            try:
                text = fetch_text(url, timeout=180)
            except Exception as exc:  # noqa: BLE001 — leftover overlay
                print(f"  {SOURCE} skip {station} {chunk_start.date()}: {exc}")
                continue
            frame = parse_oulu_csv(text, prefix)
            if frame.empty:
                print(f"  {SOURCE} empty {station} {chunk_start.date()}")
                continue
            station_frames.append(frame)
        if not station_frames:
            continue
        frames.append(pd.concat(station_frames, ignore_index=True).drop_duplicates("observed_at"))
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        cols = [c for c in extra.columns if str(c).startswith("neutron_")]
        out = out.merge(extra[["observed_at", *cols]], on="observed_at", how="outer")
    out["source"] = SOURCE
    out["license"] = LICENSE
    out["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(out["observed_at"], utc=True)
    out["available_at"] = (stamp + LATENCY).map(iso)
    mask = (stamp >= start_utc) & (stamp < end_utc)
    return out.loc[mask].sort_values("observed_at").reset_index(drop=True)
