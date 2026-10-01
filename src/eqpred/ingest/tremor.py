from __future__ import annotations

"""Hikurangi / Alpine tremor and LFE catalogs from the Slow Earthquake Database.

Published event lists, not a live GeoNet feed. Complements GNSS SSE as a
stress-state sensor. Replay with a 30-day hold so we never use a catalog
row before it could have been public.
"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.records import iso
from eqpred.regions import REGIONS, point_in_region
from eqpred.stamp import utc_ts

SOURCE = "eqdb.tremor"
LICENSE = "https://eqdb.sakura.ne.jp/"
EQDB = "https://eqdb.sakura.ne.jp"
LATENCY = timedelta(days=30)
BOXES = ("hikurangi", "alpine", "kermadec")
NZ_CATALOGS = (
    "Baratin2018-LFE",
    "Romanet2019a-Tremor",
    "Romanet2019b-Tremor",
    "Romanet2019c-Tremor",
    "Romanet2019d-Tremor",
    "Wech2012-Tremor",
    "WTD(NewZealand)-Tremor",
)
CHUNK_DAYS = 90
# sqlIntrTrem silently truncates around 10k rows.
ROW_CAP = 10000


def _as_float(value: object) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_time(rec: dict[str, object]) -> datetime | None:
    date = str(rec.get("date") or "").strip()
    time = str(rec.get("time") or "00:00:00").strip() or "00:00:00"
    if not date:
        return None
    text = f"{date}T{time}Z" if "T" not in date else date
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = pd.to_datetime(f"{date} {time}", utc=True).to_pydatetime()
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _box_for(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    for name in BOXES:
        if point_in_region(lat, lon, REGIONS[name]):
            return name
    return None


def parse_eqdb_events(payload: object) -> pd.DataFrame:
    """sqlIntrTrem JSON rows into dated LFE / tremor events."""
    if isinstance(payload, str):
        text = payload.strip()
        if not text or text[0] not in "[{":
            return pd.DataFrame()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return pd.DataFrame()
    if not isinstance(payload, list):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for rec in payload:
        if not isinstance(rec, dict):
            continue
        when = _event_time(rec)
        if when is None:
            continue
        catalog = str(rec.get("catalog") or "")
        kind = "lfe" if "lfe" in catalog.lower() else "tremor"
        lat = _as_float(rec.get("lat"))
        lon = _as_float(rec.get("lon"))
        rows.append(
            {
                "observed_at": iso(when),
                "lat": lat,
                "lon": lon,
                "dep_km": _as_float(rec.get("dep")),
                "duration_s": _as_float(rec.get("duration")),
                "catalog": catalog,
                "kind": kind,
                "box": _box_for(lat, lon),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def events_to_daily(events: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """One morning: counts and on/off flags per issued box."""
    days = pd.date_range(utc_ts(start), utc_ts(end), freq="D", inclusive="left")
    if days.empty:
        return pd.DataFrame()
    empty = {
        "tremor_n": 0.0,
        "lfe_n": 0.0,
        "tremor_hikurangi": 0.0,
        "tremor_alpine": 0.0,
        "tremor_kermadec": 0.0,
        "lfe_hikurangi": 0.0,
        "lfe_alpine": 0.0,
        "tremor_on": 0.0,
    }
    if events.empty:
        rows = []
        for day in days:
            rec = {
                "source": SOURCE,
                "observed_at": iso(day.to_pydatetime()),
                "available_at": iso((day + pd.Timedelta(days=LATENCY.days)).to_pydatetime()),
                "license": LICENSE,
            }
            rec.update(empty)
            rows.append(rec)
        return pd.DataFrame(rows)
    work = events.copy()
    work["obs_day"] = pd.to_datetime(work["observed_at"], utc=True).dt.floor("D")
    grouped = {day: part for day, part in work.groupby("obs_day")}
    rows = []
    for day in days:
        part = grouped.get(day, pd.DataFrame())
        tremor = part.loc[part["kind"] == "tremor"] if not part.empty else part
        lfe = part.loc[part["kind"] == "lfe"] if not part.empty else part
        rec = {
            "source": SOURCE,
            "observed_at": iso(day.to_pydatetime()),
            "available_at": iso((day + pd.Timedelta(days=LATENCY.days)).to_pydatetime()),
            "license": LICENSE,
            "tremor_n": float(len(tremor)),
            "lfe_n": float(len(lfe)),
            "tremor_hikurangi": float((tremor["box"] == "hikurangi").sum()) if not tremor.empty else 0.0,
            "tremor_alpine": float((tremor["box"] == "alpine").sum()) if not tremor.empty else 0.0,
            "tremor_kermadec": float((tremor["box"] == "kermadec").sum()) if not tremor.empty else 0.0,
            "lfe_hikurangi": float((lfe["box"] == "hikurangi").sum()) if not lfe.empty else 0.0,
            "lfe_alpine": float((lfe["box"] == "alpine").sum()) if not lfe.empty else 0.0,
        }
        rec["tremor_on"] = 1.0 if (rec["tremor_n"] + rec["lfe_n"]) > 0 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def _chunk_starts(start: datetime, end: datetime) -> list[datetime]:
    begin = utc_ts(start).replace(hour=0, minute=0, second=0, microsecond=0)
    stop = utc_ts(end)
    out: list[datetime] = []
    cursor = begin
    while cursor < stop:
        out.append(cursor)
        cursor = cursor + timedelta(days=CHUNK_DAYS)
    return out


def _fetch_chunk(name: str, chunk_start: datetime, span: int) -> pd.DataFrame:
    raw = fetch_bytes(
        f"{EQDB}/php/sqlIntrTrem.php",
        timeout=90,
        data={
            "sdate": chunk_start.strftime("%Y-%m-%d"),
            "span": str(span),
            "tbldb": "data",
            "catalogs": name,
            "output": "",
        },
    )
    return parse_eqdb_events(raw.decode("utf-8", errors="replace"))


def pull_eqdb_catalog(name: str, start: datetime, end: datetime) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    queue = [
        (chunk_start, min(CHUNK_DAYS, max(1, int((utc_ts(end) - chunk_start).days))))
        for chunk_start in _chunk_starts(start, end)
    ]
    while queue:
        chunk_start, span = queue.pop(0)
        try:
            parsed = _fetch_chunk(name, chunk_start, span)
        except Exception as exc:  # noqa: BLE001
            print(f"    {name} {chunk_start.date()} skipped: {exc}", flush=True)
            continue
        if parsed.empty:
            continue
        print(f"    {name} {chunk_start.date()} +{span}d {len(parsed)} events", flush=True)
        if len(parsed) >= ROW_CAP and span > 7:
            half = max(1, span // 2)
            mid = chunk_start + timedelta(days=half)
            queue.append((chunk_start, half))
            queue.append((mid, span - half))
            print(f"    {name} {chunk_start.date()} hit {ROW_CAP}, split {span}d", flush=True)
            continue
        frames.append(parsed)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["observed_at", "lat", "lon", "catalog"], keep="last")


def pull_tremor(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """NZ tremor / LFE catalogs, daily box counts."""
    frames: list[pd.DataFrame] = []
    for name in NZ_CATALOGS:
        print(f"  eqdb.tremor {name}", flush=True)
        part = pull_eqdb_catalog(name, start, end)
        if part.empty:
            print(f"    {name} empty", flush=True)
            continue
        print(f"    {name} {len(part)} events", flush=True)
        frames.append(part)
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    daily = events_to_daily(events, start, end)
    if daily.empty:
        return daily
    daily["ingest_run_id"] = ingest_run_id
    return daily
