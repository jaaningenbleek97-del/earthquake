from __future__ import annotations

"""GOES integral electrons — non-catalog radiation-belt / magnetosphere flux.

Replayable daily rows. Live SWPC JSON covers the tail. NCEI Daily Particle
Data files sometimes carry electron fluence columns after the proton bands.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd

from eqpred.ingest.http import fetch_json, fetch_text
from eqpred.ingest.protons import (
    LATENCY_DPD,
    LATENCY_LIVE,
    LATENCY_SEISS,
    fill_missing_available_at,
    stamp_available_at,
)
from eqpred.records import iso

SOURCE = "noaa.goes_electrons"
LICENSE = "https://www.weather.gov/disclaimer"
LATENCY = timedelta(hours=12)

ELECTRON_URLS = (
    "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json",
    "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-3-day.json",
    "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json",
)
FLUENCE_URLS = (
    "https://services.swpc.noaa.gov/json/goes/primary/integral-electron-fluence-7-day.json",
)
DPD_BASE = (
    "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/"
    "annual_reports/daily_solar_indices_summaries/daily_particle_data/"
)
DPD_FILES = (
    "2016_DPD.txt",
    "2017_DPD.txt",
    "2018_DPD.txt",
    "2019Q1_DPD.txt",
    "2019Q2_DPD.txt",
    "2019Q3_DPD.txt",
    "2019Q4_DPD.txt",
    "2020Q1_DPD.txt",
)

ENERGY_COLS = {
    ">=0.8 mev": "electron_08mev",
    ">=2 mev": "electron_2mev",
    ">=4 mev": "electron_4mev",
}
DPD_MISSING = {-1.0e5, -1.0e05, -999.99, -1.0}
SECONDS_PER_DAY = 86400.0


def _parse_stamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _energy_col(label: object) -> str | None:
    if label is None:
        return None
    compact = str(label).strip().lower().replace(" ", "")
    if not compact.startswith(">="):
        compact = ">=" + compact
    lookup = {k.replace(" ", ""): v for k, v in ENERGY_COLS.items()}
    return lookup.get(compact)


def _day_stamp(observed: datetime) -> str:
    return iso(observed.replace(hour=0, minute=0, second=0, microsecond=0))


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number in DPD_MISSING or number < 0:
        return None
    return number


def parse_goes_electrons(rows: list[Any]) -> pd.DataFrame:
    """Daily max integral electron flux (pfu) per energy band."""
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        observed = _parse_stamp(rec.get("time_tag") or rec.get("time"))
        if observed is None:
            continue
        col = _energy_col(rec.get("energy") or rec.get("energy_channel"))
        flux = rec.get("flux")
        try:
            flux_f = float(flux) if flux is not None else None
        except (TypeError, ValueError):
            flux_f = None
        if col is None or flux_f is None:
            continue
        out.append({"observed_at": _day_stamp(observed), col: flux_f})
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out)
    cols = [c for c in ENERGY_COLS.values() if c in frame.columns]
    return frame.groupby("observed_at", as_index=False)[cols].max()


def parse_goes_electron_fluence(rows: list[Any]) -> pd.DataFrame:
    """Daily fluence → mean flux (pfu)."""
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        observed = _parse_stamp(rec.get("time_tag") or rec.get("time"))
        if observed is None:
            continue
        col = _energy_col(rec.get("energy") or rec.get("energy_channel"))
        fluence = _finite(rec.get("fluence"))
        if col is None or fluence is None:
            continue
        out.append({"observed_at": _day_stamp(observed), col: fluence / SECONDS_PER_DAY})
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out)
    cols = [c for c in ENERGY_COLS.values() if c in frame.columns]
    return frame.groupby("observed_at", as_index=False)[cols].max()


def parse_dpd_electrons(text: str) -> pd.DataFrame:
    """NCEI DPD: if extra columns after the proton trio look like electron fluence, keep them."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":") or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 7:
            continue
        try:
            observed = datetime(int(parts[0]), int(parts[1]), int(parts[2]), tzinfo=timezone.utc)
        except ValueError:
            continue
        rec: dict[str, object] = {"observed_at": _day_stamp(observed)}
        extras = parts[6:]
        for raw, col in zip(extras, ("electron_08mev", "electron_2mev", "electron_4mev")):
            value = _finite(raw)
            if value is None:
                continue
            rec[col] = value / SECONDS_PER_DAY if value > 1000.0 else value
        if len(rec) > 1:
            rows.append(rec)
    return pd.DataFrame(rows)


def combine_electron_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    by_day: dict[str, dict[str, object]] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for rec in frame.to_dict(orient="records"):
            stamp = str(rec.get("observed_at") or "")
            if not stamp:
                continue
            day = stamp[:10]
            slot = by_day.setdefault(day, {"observed_at": f"{day}T00:00:00Z"})
            for key, value in rec.items():
                if key == "observed_at" or value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass
                slot[key] = value
    if not by_day:
        return pd.DataFrame()
    return pd.DataFrame(list(by_day.values())).sort_values("observed_at").reset_index(drop=True)


def _safe_json(url: str) -> list[Any]:
    try:
        payload = fetch_json(url, timeout=60)
    except Exception:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _safe_text(url: str) -> str:
    try:
        return fetch_text(url, timeout=90)
    except Exception:
        return ""


def _in_window(frame: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    if frame.empty or "observed_at" not in frame.columns:
        return frame
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    keep = pd.Series(True, index=frame.index)
    if start is not None:
        keep &= stamp >= pd.Timestamp(start)
    if end is not None:
        keep &= stamp < pd.Timestamp(end)
    return frame.loc[keep].reset_index(drop=True)


def pull_electrons(
    ingest_run_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """GOES electrons: DPD extras + GOES-R SEISS MPS-HI + SWPC JSON tail."""
    print("  NCEI daily particle data (electron columns)", flush=True)
    dpd_parts: list[pd.DataFrame] = []
    for name in DPD_FILES:
        year = int(name[:4])
        if start is not None and year < start.year:
            continue
        if end is not None and year > end.year:
            continue
        text = _safe_text(DPD_BASE + name)
        parsed = parse_dpd_electrons(text)
        if parsed.empty:
            print(f"    skip electrons {name}", flush=True)
            continue
        print(f"    {name} {len(parsed)} electron days", flush=True)
        dpd_parts.append(parsed)
    dpd = pd.concat(dpd_parts, ignore_index=True) if dpd_parts else pd.DataFrame()

    print("  GOES-R SEISS MPS-HI (NCEI L2 avg5m)", flush=True)
    from eqpred.ingest.seiss import pull_seiss

    seiss = pull_seiss("mpsh", ingest_run_id, start, end)

    rows: list[Any] = []
    seen: set[str] = set()
    for url in ELECTRON_URLS:
        for rec in _safe_json(url):
            if not isinstance(rec, dict):
                continue
            key = f"{rec.get('time_tag')}|{rec.get('energy')}|{rec.get('satellite')}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec)
    live_flux = parse_goes_electrons(rows)
    fluence_rows: list[Any] = []
    for url in FLUENCE_URLS:
        fluence_rows.extend(_safe_json(url))
    live_fluence = parse_goes_electron_fluence(fluence_rows)

    dpd = stamp_available_at(dpd, LATENCY_DPD)
    seiss = stamp_available_at(seiss, LATENCY_SEISS)
    live_fluence = stamp_available_at(live_fluence, LATENCY_LIVE)
    live_flux = stamp_available_at(live_flux, LATENCY_LIVE)
    daily = combine_electron_frames((dpd, seiss, live_fluence, live_flux))
    daily = _in_window(daily, start, end)
    if daily.empty:
        return pd.DataFrame()
    daily["source"] = SOURCE
    daily["ingest_run_id"] = ingest_run_id
    daily["license"] = LICENSE
    daily = fill_missing_available_at(daily, LATENCY_LIVE)
    return daily.sort_values("observed_at").reset_index(drop=True)
