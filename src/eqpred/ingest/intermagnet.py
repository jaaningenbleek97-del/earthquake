from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from math import hypot
from urllib.parse import urlencode

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

# Public BGS GIN — no account for ordinary (embargoed) downloads.
# HAPI sibling: https://imag-data.bgs.ac.uk/GIN_V1/hapi
GIN = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
HAPI = "https://imag-data.bgs.ac.uk/GIN_V1/hapi/data"
SOURCE = "intermagnet.gin"
LICENSE = "https://intermagnet.org/data_conditions.html"
LATENCY = timedelta(days=1)
FILL_ABS = 99990.0
# NZ first, then Pacific / Japan / Hawaii / Chile.
# AMS returns HTTP 400 from 2021-03 on. IPM returns HTTP 500 for most months.
# Do not sit through 120s × 4 retries for known-dead codes.
STATIONS = ("EYR", "API", "MCQ", "CNB", "CTA", "KAK", "HON", "PPT")
SKIP_STATIONS = ("AMS", "IPM")
CHUNK_DAYS = 31
FETCH_TIMEOUT = 120
MAX_TRIES = 4
# Hourly is enough for daily mean / range / max |dH|. Minute is 60× the bytes.
SAMPLE = "Hour"


def _is_fill(value: float) -> bool:
    return abs(value) >= FILL_ABS


def _orientation(text: str) -> str:
    for line in text[:2500].splitlines():
        if "Reported" not in line:
            continue
        for token in line.replace("|", " ").split():
            tok = token.upper()
            if tok in {"HDZF", "XYZF", "HDZS", "XYZS", "DIFF", "DIFS", "HDZ", "XYZ"}:
                return tok
    return "HDZF"


def _h_from_components(c1: float, c2: float, orientation: str) -> float | None:
    if orientation.startswith("XYZ"):
        if _is_fill(c1) or _is_fill(c2):
            return None
        return hypot(c1, c2)
    if _is_fill(c1):
        return None
    return c1


def _parse_stamp(date_part: str, time_part: str | None = None) -> datetime:
    if time_part is None:
        raw = date_part.replace("Z", "+00:00")
        stamp = datetime.fromisoformat(raw)
    else:
        stamp = datetime.fromisoformat(f"{date_part}T{time_part.split('.')[0]}")
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def parse_iaga2002(text: str) -> pd.DataFrame:
    """Parse IAGA-2002 minute records to UTC stamps and horizontal intensity."""
    orientation = _orientation(text)
    rows: list[tuple[datetime, float]] = []
    for line in text.splitlines():
        if not line or not line[0].isdigit() or len(line) < 24:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            stamp = _parse_stamp(parts[0], parts[1])
            c1 = float(parts[3])
            c2 = float(parts[4])
        except ValueError:
            continue
        h = _h_from_components(c1, c2, orientation)
        if h is None:
            continue
        rows.append((stamp, h))
    if not rows:
        return pd.DataFrame(columns=["observed_at", "h_nt"])
    return pd.DataFrame(rows, columns=["observed_at", "h_nt"])


def parse_hapi_csv(text: str) -> pd.DataFrame:
    """Parse INTERMAGNET HAPI CSV (Time,H,D,Z,F or Time,X,Y,Z,F)."""
    rows: list[tuple[datetime, float]] = []
    orientation = "HDZF"
    if "/xyz" in text[:800].lower() or ",X," in text[:200] or ",x," in text[:200]:
        orientation = "XYZF"
    for line in text.splitlines():
        if not line or line[0] in "#;":
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0][:1].isdigit():
            continue
        try:
            stamp = _parse_stamp(parts[0])
            c1 = float(parts[1])
            c2 = float(parts[2])
        except ValueError:
            continue
        h = _h_from_components(c1, c2, orientation)
        if h is None:
            continue
        rows.append((stamp, h))
    if not rows:
        return pd.DataFrame(columns=["observed_at", "h_nt"])
    return pd.DataFrame(rows, columns=["observed_at", "h_nt"])


def daily_magnet_features(minutes: pd.DataFrame, station: str) -> pd.DataFrame:
    """Daily mean H, daily range, and max |dH| between samples.

    Samples may be hourly (preferred) or minute. Range and dB/dt are
    coarser on the hour; both still flag storm days.
    """
    if minutes.empty or "h_nt" not in minutes.columns:
        return pd.DataFrame()
    work = minutes.dropna(subset=["h_nt"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["observed_at"] = pd.to_datetime(work["observed_at"], utc=True)
    work = work.sort_values("observed_at")
    work["day"] = work["observed_at"].dt.floor("D")
    work["dh"] = work.groupby("day", sort=False)["h_nt"].diff().abs()
    grouped = work.groupby("day", sort=True)
    out = pd.DataFrame(
        {
            "observed_at": grouped["h_nt"].mean().index,
            "mag_h_nt": grouped["h_nt"].mean().to_numpy(),
            "mag_h_range_nt": (grouped["h_nt"].max() - grouped["h_nt"].min()).to_numpy(),
            "mag_dbdt_nt": grouped["dh"].max().to_numpy(),
        }
    )
    out["source"] = SOURCE
    out["mag_station"] = station
    out["license"] = LICENSE
    out["available_at"] = [iso(ts.to_pydatetime() + LATENCY) for ts in out["observed_at"]]
    out["observed_at"] = [iso(ts.to_pydatetime()) for ts in out["observed_at"]]
    return out.reset_index(drop=True)


def _gin_data_url(station: str, start: datetime, duration: int) -> str:
    params = {
        "Request": "GetData",
        "Format": "iaga2002",
        "observatoryIagaCode": station,
        "samplesPerDay": SAMPLE,
        "dataStartDate": start.strftime("%Y-%m-%d"),
        "dataDuration": str(max(1, duration)),
        "publicationState": "best-avail",
        "orientation": "HDZF",
    }
    return f"{GIN}?{urlencode(params)}"


def _hapi_url(station: str, start: datetime, end: datetime) -> str:
    params = {
        "id": f"{station}/best-avail/PT1H/hdzf",
        "time.min": start.strftime("%Y-%m-%dT00:00:00Z"),
        "time.max": end.strftime("%Y-%m-%dT00:00:00Z"),
        "format": "csv",
    }
    return f"{HAPI}?{urlencode(params)}"


def _missing_http(exc: Exception) -> bool:
    msg = str(exc)
    return "HTTP 404" in msg or "HTTP 400" in msg or "HTTP 500" in msg


def _fetch_text_retry(url: str) -> str:
    last: Exception | None = None
    for attempt in range(MAX_TRIES):
        try:
            return fetch_text(url, timeout=FETCH_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if _missing_http(exc) or attempt == MAX_TRIES - 1:
                raise
            wait = 2**attempt
            print(f"  intermagnet.gin retry {attempt + 1} in {wait}s: {exc}", flush=True)
            time.sleep(wait)
    raise last or RuntimeError(f"GET {url} failed")


def _minutes_from_payload(text: str) -> pd.DataFrame:
    if "IAGA-2002" in text[:400] or "DATE       TIME" in text[:800]:
        return parse_iaga2002(text)
    minutes = parse_hapi_csv(text)
    if minutes.empty:
        minutes = parse_iaga2002(text)
    return minutes


def _fetch_station(station: str, start: datetime, end: datetime) -> pd.DataFrame:
    duration = max(1, (end - start).days)
    last_exc: Exception | None = None
    try:
        text = _fetch_text_retry(_hapi_url(station, start, end))
        frame = daily_magnet_features(_minutes_from_payload(text), station)
        if not frame.empty:
            return frame
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
        if _missing_http(exc):
            raise
        print(f"  intermagnet.gin {station} {start.date()} HAPI failed ({exc}); GIN {SAMPLE}", flush=True)
    try:
        text = _fetch_text_retry(_gin_data_url(station, start, duration))
    except Exception as exc:  # noqa: BLE001
        raise last_exc or exc
    return daily_magnet_features(_minutes_from_payload(text), station)


def _pull_chunk(start: datetime, end: datetime) -> pd.DataFrame:
    if (end - start).days <= 0:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for station in STATIONS:
        try:
            frame = _fetch_station(station, start, end)
        except Exception as exc:  # noqa: BLE001 — skip dead / missing station-months
            print(f"  intermagnet.gin skip {station} {start.date()}: {exc}", flush=True)
            continue
        if frame.empty:
            print(f"  intermagnet.gin skip {station} {start.date()}: no valid minutes", flush=True)
            continue
        print(f"  intermagnet.gin {station} {start.date()} {len(frame)} days", flush=True)
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def pull_intermagnet(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily NZ/nearby ground magnetic variation from the public INTERMAGNET GIN."""
    frames: list[pd.DataFrame] = []
    cursor = utc_ts(start).to_pydatetime()
    stop = utc_ts(end).to_pydatetime()
    while cursor < stop:
        nxt = min(cursor + timedelta(days=CHUNK_DAYS), stop)
        print(f"  intermagnet.gin {cursor.date()} -> {nxt.date()}", flush=True)
        part = _pull_chunk(cursor, nxt)
        if not part.empty:
            frames.append(part)
        time.sleep(0.4)
        cursor = nxt
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    frame = frame.loc[mask].drop_duplicates(["observed_at", "mag_station"]).sort_values(
        ["observed_at", "mag_station"]
    )
    return frame.reset_index(drop=True)
