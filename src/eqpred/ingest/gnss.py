from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso

# Nevada Geodetic Lab daily tenv3. NZ + transfer arcs. Skip 404s.
NGL_URLS = (
    "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{site}.tenv3",
    "https://geodesy.unr.edu/gps_timeseries/tenv3/IGS20/{site}.tenv3",
    "http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{site}.tenv3",
)
NZ_SITES = ("AUCK", "WGTN", "LYTT", "DUND", "CHTI", "RAUL")
JAPAN_SITES = ("TSKB", "USUD", "MIZU")
CHILE_SITES = ("ANTC", "SANT", "IQQE")
SITES = NZ_SITES + JAPAN_SITES + CHILE_SITES
SLOW_SLIP_SITES = ("WGTN", "RAUL")
LICENSE = "http://geodesy.unr.edu/"
SOURCE = "ngl.gnss"
# Solutions are reprocessed; do not treat same-day files as live.
LATENCY = timedelta(days=14)


def _from_decimal_year(value: float) -> datetime:
    year = int(value)
    origin = datetime(year, 1, 1, tzinfo=timezone.utc)
    return origin + timedelta(days=(value - year) * 365.25)


def _parse_tenv3(text: str, site: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 13 or parts[0].lower() == "site":
            continue
        try:
            observed = _from_decimal_year(float(parts[2]))
            east = float(parts[8])
            north = float(parts[10])
            vert = float(parts[12])
        except (IndexError, ValueError):
            continue
        rows.append({"site": site, "observed_at": observed, "east_m": east, "north_m": north, "vert_m": vert})
    return pd.DataFrame(rows)


def pull_gnss(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily GNSS positions by site. Unused on issued p until a head beats ETAS."""
    frames = []
    for site in SITES:
        print(f"  NGL {site}", flush=True)
        text = None
        last_exc: Exception | None = None
        for template in NGL_URLS:
            url = template.format(site=site)
            try:
                text = fetch_text(url, timeout=240)
                break
            except Exception as exc:  # noqa: BLE001 — try the next scheme
                last_exc = exc
                continue
        if text is None:
            print(f"    skip {site}: {last_exc}", flush=True)
            continue
        frame = _parse_tenv3(text, site)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[(raw["observed_at"] >= start) & (raw["observed_at"] < end)]
    if raw.empty:
        return pd.DataFrame()
    daily = (
        raw.assign(day=raw["observed_at"].dt.floor("D"))
        .groupby(["site", "day"], as_index=False)
        .agg(east_m=("east_m", "mean"), north_m=("north_m", "mean"), vert_m=("vert_m", "mean"))
    )
    out = []
    for rec in daily.to_dict(orient="records"):
        observed = rec["day"].to_pydatetime()
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
                "east_mm": float(rec["east_m"]) * 1000.0,
                "north_mm": float(rec["north_m"]) * 1000.0,
                "vert_mm": float(rec["vert_m"]) * 1000.0,
            }
        )
    return pd.DataFrame(out)
