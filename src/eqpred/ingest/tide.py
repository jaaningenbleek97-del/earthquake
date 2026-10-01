"""UHSLC fast-delivery daily sea level. Loading proxy for the issued boxes.

Fast delivery is typically a few weeks behind the gauge. We add a 21-day
available_at lag so a morning cannot see a row that has not been posted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso

SOURCE = "uhslc.tide"
LICENSE = "https://uhslc.soest.hawaii.edu/"
LATENCY = timedelta(days=21)
# Fast-delivery daily CSV: year, month, day, sea level (mm). 9999 / -32767 = missing.
CSV_URL = "https://uhslc.soest.hawaii.edu/data/csv/fast/daily/d{uid:03d}d.csv"
ERDDAP = (
    "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_daily_fast.csv"
    "?time,sea_level,uhslc_id,station_name"
    "&uhslc_id={uid}"
)

# Slug -> UHSLC fast-delivery id (global_daily_fast). Auckland and Lyttelton
# are not on the fast-delivery feed; Jackson Bay / Bluff stand in for the
# South Island, Chatham for the east, Kushiro for Japan.
STATIONS: dict[str, int] = {
    "wellington": 71,
    "bluff": 72,
    "chatham": 79,
    "jackson": 403,
    "valparaiso": 81,
    "kushiro": 350,
}


def _parse_daily_csv(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0].isalpha():
            continue
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        if len(parts) < 4:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            level = float(parts[3])
        except ValueError:
            continue
        if level >= 9000 or level <= -30000:
            continue
        try:
            observed = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append({"observed_at": observed, "tide_mm": level})
    return pd.DataFrame(rows)


def _parse_erddap(text: str) -> pd.DataFrame:
    rows = []
    for i, line in enumerate(text.splitlines()):
        if i == 0 or not line or line.startswith("time,"):
            continue
        # ERDDAP unit row looks like "UTC,mm,,"
        if i == 1 and ("UTC" in line or line.startswith(",")):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            observed = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            level = float(parts[1])
        except (TypeError, ValueError):
            continue
        if not np_finite(level) or level >= 9000 or level <= -30000:
            continue
        rows.append({"observed_at": observed, "tide_mm": level})
    return pd.DataFrame(rows)


def np_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def pull_tide(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily sea level at a handful of gauges. Unused on issued p until a head uses it."""
    frames = []
    for slug, uid in STATIONS.items():
        print(f"  UHSLC {slug} #{uid}", flush=True)
        frame = pd.DataFrame()
        try:
            frame = _parse_erddap(fetch_text(ERDDAP.format(uid=uid), timeout=180))
        except Exception as exc:  # noqa: BLE001 — fall back to the static CSV
            print(f"    erddap skip {slug}: {exc}", flush=True)
        if frame.empty:
            try:
                frame = _parse_daily_csv(fetch_text(CSV_URL.format(uid=uid), timeout=180))
            except Exception as exc:  # noqa: BLE001
                print(f"    csv skip {slug}: {exc}", flush=True)
        if frame.empty:
            continue
        frame = frame.loc[(frame["observed_at"] >= start) & (frame["observed_at"] < end)]
        if frame.empty:
            continue
        frame = (
            frame.assign(day=frame["observed_at"].dt.floor("D"))
            .groupby("day", as_index=False)
            .agg(tide_mm=("tide_mm", "mean"))
            .rename(columns={"day": "observed_at"})
        )
        frame["site"] = slug
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    out = []
    for rec in raw.to_dict(orient="records"):
        observed = rec["observed_at"].to_pydatetime()
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        out.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "ingest_run_id": ingest_run_id,
                "site": rec["site"],
                f"tide_{rec['site']}_mm": float(rec["tide_mm"]),
            }
        )
    wide = pd.DataFrame(out)
    if wide.empty:
        return wide
    # One row per day: mean NZ gauges plus the named columns.
    value_cols = [c for c in wide.columns if c.startswith("tide_") and c.endswith("_mm")]
    daily = (
        wide.assign(_day=pd.to_datetime(wide["observed_at"], utc=True).dt.floor("D"))
        .groupby("_day", as_index=False)
        .agg(
            **{c: (c, "mean") for c in value_cols},
            available_at=("available_at", "min"),
            ingest_run_id=("ingest_run_id", "first"),
        )
        .rename(columns={"_day": "observed_at"})
    )
    nz = [c for c in ("tide_wellington_mm", "tide_bluff_mm", "tide_jackson_mm", "tide_chatham_mm") if c in daily.columns]
    if nz:
        daily["tide_nz_mm"] = daily[nz].mean(axis=1)
    daily["source"] = SOURCE
    daily["license"] = LICENSE
    daily["observed_at"] = pd.to_datetime(daily["observed_at"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
    return daily
