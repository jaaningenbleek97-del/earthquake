from __future__ import annotations

"""Thermal IR / skin-temperature trap. Hunt confirm = leak.

NASA POWER daily skin temperature over NZ boxes. This is the wget-able
thermal series. If hunt 'confirms' it, the leftover stack is leaking a
known non-precursor, not discovering one.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_json
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "nasa.tir"
LICENSE = "https://power.larc.nasa.gov/"
LATENCY = timedelta(days=3)
SITES = (
    ("hikurangi", -39.5, 178.0),
    ("alpine", -43.5, 170.5),
    ("kermadec", -30.0, 178.5),
    ("southland", -46.4, 168.3),
)


def parse_power_daily(payload: object, site: str) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    props = ((payload.get("properties") or {}).get("parameter")) or {}
    series = props.get("TS") or props.get("T2M") or {}
    if not isinstance(series, dict):
        return pd.DataFrame()
    rows = []
    for key, val in series.items():
        try:
            day = datetime.strptime(str(key), "%Y%m%d").replace(tzinfo=timezone.utc)
            number = float(val)
        except (TypeError, ValueError):
            continue
        if number > 1000:  # POWER fill
            continue
        rows.append({"observed_at": day, "tir_ts": number, "tir_site": site})
    return pd.DataFrame(rows)


def pull_tir(start: datetime, end: datetime, run_id: str, *, fetch=fetch_json) -> pd.DataFrame:
    _ = run_id
    start = utc_ts(start)
    end = utc_ts(end)
    frames: list[pd.DataFrame] = []
    for site, lat, lon in SITES:
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=TS,T2M&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}&format=JSON"
        )
        try:
            payload = fetch(url, timeout=90)
        except Exception as exc:
            print(f"  nasa.tir {site} skipped: {exc}", flush=True)
            continue
        parsed = parse_power_daily(payload, site)
        if not parsed.empty:
            frames.append(parsed)
            print(f"  nasa.tir {site} {len(parsed)} days", flush=True)
    if not frames:
        return pd.DataFrame()
    work = pd.concat(frames, ignore_index=True)
    daily = work.groupby("observed_at", as_index=False).agg(tir_ts=("tir_ts", "median"))
    hik = work.loc[work["tir_site"] == "hikurangi"].groupby("observed_at", as_index=False)["tir_ts"].median()
    hik = hik.rename(columns={"tir_ts": "tir_hikurangi"})
    daily = daily.merge(hik, on="observed_at", how="left")
    rows = []
    for rec in daily.itertuples(index=False):
        day = rec.observed_at.to_pydatetime()
        if day.tzinfo is None:
            day = day.replace(tzinfo=timezone.utc)
        day = day.replace(hour=0, minute=0, second=0, microsecond=0)
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(day),
                "available_at": iso(day + LATENCY),
                "tir_ts": float(rec.tir_ts),
                "tir_hikurangi": None if pd.isna(getattr(rec, "tir_hikurangi", None)) else float(rec.tir_hikurangi),
            }
        )
    return pd.DataFrame(rows)
