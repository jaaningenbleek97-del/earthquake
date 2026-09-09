from __future__ import annotations

"""NASA DONKI event lists: flares, high-speed streams, SEP, shocks.

Replayable CCMC JSON. This is the labeled catalog Davidson/Burns point at
on a flare / coronal-hole / proton-storm day — not a spectrogram.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from eqpred.ingest.http import fetch_json
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "nasa.donki"
LICENSE = "https://ccmc.gsfc.nasa.gov/donki/"
LATENCY = timedelta(days=1)
BASE = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get"
HOLD_DAYS = 2


def _parse_stamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    if len(text) == 10:
        text = text + "T00:00:00+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _flare_score(class_type: object) -> float | None:
    raw = str(class_type or "").strip().upper()
    if len(raw) < 2:
        return None
    letter = raw[0]
    try:
        mag = float(raw[1:])
    except ValueError:
        mag = 0.0
    base = {"B": 0.0, "C": 1.0, "M": 2.0, "X": 3.0}.get(letter)
    if base is None:
        return None
    return base + min(mag, 9.9) / 10.0


def parse_donki_flares(rows: list[Any]) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        when = _parse_stamp(rec.get("peakTime") or rec.get("beginTime"))
        if when is None:
            continue
        score = _flare_score(rec.get("classType"))
        klass = str(rec.get("classType") or "").upper()
        out.append(
            {
                "observed_at": iso(when.replace(hour=0, minute=0, second=0, microsecond=0)),
                "flare_count": 1.0,
                "flare_m": 1.0 if klass.startswith("M") else 0.0,
                "flare_x": 1.0 if klass.startswith("X") else 0.0,
                "flare_score": score if score is not None else 0.0,
            }
        )
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out)
    return frame.groupby("observed_at", as_index=False).agg(
        flare_count=("flare_count", "sum"),
        flare_m_count=("flare_m", "sum"),
        flare_x_count=("flare_x", "sum"),
        flare_score=("flare_score", "max"),
    )


def parse_donki_flags(rows: list[Any], column: str, time_keys: tuple[str, ...]) -> pd.DataFrame:
    days: list[str] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        when = None
        for key in time_keys:
            when = _parse_stamp(rec.get(key))
            if when is not None:
                break
        if when is None:
            continue
        days.append(iso(when.replace(hour=0, minute=0, second=0, microsecond=0)))
    if not days:
        return pd.DataFrame()
    frame = pd.DataFrame({"observed_at": days, column: 1.0})
    return frame.groupby("observed_at", as_index=False)[column].max()


def _safe_list(kind: str, start: datetime, end: datetime) -> list[Any]:
    url = (
        f"{BASE}/{kind}?startDate={start.date().isoformat()}&endDate={end.date().isoformat()}"
    )
    try:
        payload = fetch_json(url, timeout=120)
    except Exception as exc:  # noqa: BLE001 — keep the other event types
        print(f"  nasa.donki skip {kind}: {exc}")
        return []
    return payload if isinstance(payload, list) else []


def _year_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    year = max(start.year, 2010)
    stop = end.year
    while year <= stop:
        lo = datetime(year, 1, 1, tzinfo=timezone.utc)
        hi = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        if lo < start:
            lo = start
        if hi > end:
            hi = end
        if lo < hi:
            chunks.append((lo, hi))
        year += 1
    return chunks


def _merge_days(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        if extra.empty:
            continue
        cols = [c for c in extra.columns if c != "observed_at"]
        out = out.merge(extra[["observed_at", *cols]], on="observed_at", how="outer")
    return out


def _hold_flags(daily: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Every morning in range. Event flags stay on for HOLD_DAYS."""
    days = pd.date_range(utc_ts(start), utc_ts(end), freq="D", inclusive="left")
    if days.empty:
        return pd.DataFrame()
    if daily.empty:
        empty = pd.DataFrame({"observed_at": [iso(d.to_pydatetime()) for d in days]})
        for col in ("flare_count", "flare_m_count", "flare_x_count", "flare_score", "hss_flag", "sep_flag", "ips_flag"):
            empty[col] = 0.0
        return empty
    work = daily.copy()
    work["_day"] = pd.to_datetime(work["observed_at"], utc=True).dt.floor("D")
    rows: list[dict[str, object]] = []
    flag_cols = [c for c in work.columns if c not in {"observed_at", "_day"}]
    for day in days:
        window = (work["_day"] >= day - pd.Timedelta(days=HOLD_DAYS - 1)) & (work["_day"] <= day)
        hit = work.loc[window]
        rec: dict[str, object] = {
            "observed_at": iso(day.to_pydatetime()),
            "available_at": iso((day + pd.Timedelta(days=1)).to_pydatetime()),
            "source": SOURCE,
            "license": LICENSE,
        }
        for col in flag_cols:
            rec[col] = float(pd.to_numeric(hit[col], errors="coerce").max()) if not hit.empty else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def pull_donki(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for lo, hi in _year_chunks(start, end):
        print(f"  nasa.donki {lo.date()} -> {hi.date()}", flush=True)
        flares = parse_donki_flares(_safe_list("FLR", lo, hi))
        hss = parse_donki_flags(_safe_list("HSS", lo, hi), "hss_flag", ("eventTime", "startTime", "peakTime"))
        sep = parse_donki_flags(_safe_list("SEP", lo, hi), "sep_flag", ("eventTime", "startTime", "peakTime"))
        ips = parse_donki_flags(_safe_list("IPS", lo, hi), "ips_flag", ("eventTime", "startTime", "peakTime"))
        year = _merge_days([f for f in (flares, hss, sep, ips) if not f.empty])
        if not year.empty:
            parts.append(year)
    daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not daily.empty:
        daily = daily.groupby("observed_at", as_index=False).max()
    out = _hold_flags(daily, start, end)
    if out.empty:
        return out
    out["ingest_run_id"] = ingest_run_id
    return out.sort_values("observed_at").reset_index(drop=True)
