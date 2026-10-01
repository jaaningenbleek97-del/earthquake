from __future__ import annotations

"""NZ lake and river stage so wet weeks cannot look like precursors.

Hilltop Stage/Flow from the same councils as the wells overlay, plus ECan,
Otago, Southland, and Horizons. This is surface water, not groundwater.
"""

import re
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode

import pandas as pd

from eqpred.ingest.http import fetch_json, fetch_text
from eqpred.ingest.wells import _hilltop_get, parse_hilltop_xml
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "hydro.surface"
LICENSE = "https://www.lawa.org.nz/"
LATENCY = timedelta(days=2)
MEASUREMENTS = ("Stage", "Flow", "Lake Level", "Water Level")
HILLTOP_SERVERS: tuple[tuple[str, str], ...] = (
    ("https://hilltop.gw.govt.nz/Data.hts", "hikurangi"),
    ("https://hydro.marlborough.govt.nz/data.hts", "alpine"),
    ("https://envdata.tasman.govt.nz/data.hts", "alpine"),
    ("https://data.ecan.govt.nz/data.hts", "alpine"),
    ("https://hilltop.orc.govt.nz/data.hts", "alpine"),
    ("https://envdata.es.govt.nz/data.hts", "alpine"),
    ("https://hilltop.horizons.govt.nz/data.hts", "hikurangi"),
)
SITE_RE = re.compile(r"\b(lake|river|stream|at |bridge|weir|harbour|lagoon)\b", re.I)
SKIP_RE = re.compile(r"\b(well|bore|piezo|groundwater|rainfall|rain |temp|wind)\b", re.I)
MAX_SITES = 3


def daily_hydro(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    work = samples.copy()
    work["obs_day"] = pd.to_datetime(work["observed_at"], utc=True).dt.floor("D")
    grouped = (
        work.groupby(["obs_day", "hydro_station"], as_index=False)
        .agg(hydro_level=("hydro_level", "median"), hydro_region=("hydro_region", "first"))
        .sort_values(["hydro_station", "obs_day"])
    )
    grouped["hydro_d1"] = grouped.groupby("hydro_station")["hydro_level"].diff()
    rows = []
    for rec in grouped.itertuples(index=False):
        day = rec.obs_day.to_pydatetime()
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(day),
                "available_at": iso(day + LATENCY),
                "hydro_station": rec.hydro_station,
                "hydro_level": float(rec.hydro_level),
                "hydro_d1": None if pd.isna(rec.hydro_d1) else float(rec.hydro_d1),
                "hydro_region": rec.hydro_region,
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    wide = (
        daily.assign(_day=pd.to_datetime(daily["observed_at"], utc=True).dt.floor("D"))
        .groupby("_day", as_index=False)
        .agg(
            hydro_nz=("hydro_level", "median"),
            hydro_nz_d1=("hydro_d1", "median"),
            available_at=("available_at", "min"),
        )
        .rename(columns={"_day": "observed_at"})
    )
    wide["observed_at"] = pd.to_datetime(wide["observed_at"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
    hik = daily.loc[daily["hydro_region"] == "hikurangi"]
    alp = daily.loc[daily["hydro_region"] == "alpine"]
    if not hik.empty:
        h = hik.groupby("observed_at", as_index=False)["hydro_level"].median().rename(columns={"hydro_level": "hydro_hikurangi"})
        wide = wide.merge(h, on="observed_at", how="left")
    if not alp.empty:
        a = alp.groupby("observed_at", as_index=False)["hydro_level"].median().rename(columns={"hydro_level": "hydro_alpine"})
        wide = wide.merge(a, on="observed_at", how="left")
    return wide


def _discover(base: str) -> list[str]:
    found: list[str] = []
    for measurement in MEASUREMENTS:
        query = urlencode({"Service": "Hilltop", "Request": "SiteList", "Measurement": measurement})
        try:
            text = fetch_text(f"{base}?{query}", timeout=40)
        except Exception:
            continue
        names = re.findall(r'Site Name="([^"]+)"', text)
        keep = [name for name in names if SITE_RE.search(name) and not SKIP_RE.search(name)]
        for name in keep:
            if name not in found:
                found.append(name)
        if found:
            break
    return found[:MAX_SITES]


def _probe_measurement(base: str, site: str, start: datetime, end: datetime) -> str:
    last_year = utc_ts(end).year
    first_year = max(utc_ts(start).year, 2005)
    years = []
    for year in (last_year - 1, last_year, 2016, first_year):
        if first_year <= year <= last_year and year not in years:
            years.append(year)
    for year in years:
        chunk_start = datetime(year, 1, 1, tzinfo=start.tzinfo)
        chunk_end = datetime(year + 1, 1, 1, tzinfo=start.tzinfo)
        if chunk_start < start:
            chunk_start = start
        if chunk_end > end:
            chunk_end = end
        for measurement in MEASUREMENTS:
            try:
                text = _hilltop_get(base, site, measurement, chunk_start, chunk_end)
                parsed = parse_hilltop_xml(text, site)
            except Exception:
                parsed = pd.DataFrame()
            if not parsed.empty:
                print(f"    {measurement} from {year}", flush=True)
                return measurement
    return MEASUREMENTS[0]


def pull_hydro(start: datetime, end: datetime, run_id: str) -> pd.DataFrame:
    _ = run_id
    start = utc_ts(start)
    end = utc_ts(end)
    wet = _pull_power_wet(start, end)
    samples: list[pd.DataFrame] = []
    for base, region in HILLTOP_SERVERS:
        sites = _discover(base)
        print(f"  hydro.surface {base} {len(sites)} sites", flush=True)
        for site in sites:
            print(f"  hydro.surface {site}", flush=True)
            measurement = _probe_measurement(base, site, start, end)
            year = max(start.year, 2005)
            while year <= end.year:
                chunk_start = datetime(year, 1, 1, tzinfo=start.tzinfo)
                chunk_end = datetime(year + 1, 1, 1, tzinfo=start.tzinfo)
                if chunk_start < start:
                    chunk_start = start
                if chunk_end > end:
                    chunk_end = end
                try:
                    text = _hilltop_get(base, site, measurement, chunk_start, chunk_end)
                    parsed = parse_hilltop_xml(text, site)
                except Exception:
                    parsed = pd.DataFrame()
                if not parsed.empty:
                    parsed = parsed.rename(columns={"well_station": "hydro_station", "well_level_m": "hydro_level"})
                    parsed["hydro_region"] = region
                    samples.append(parsed)
                    print(f"    {year} {len(parsed)}", flush=True)
                year += 1
    if not samples:
        surface = pd.DataFrame()
    else:
        surface = daily_hydro(pd.concat(samples, ignore_index=True))
    if surface.empty:
        return wet
    if wet.empty:
        return surface
    wet_cols = [c for c in ("hydro_gwet", "hydro_precip") if c in wet.columns]
    return surface.merge(wet[["observed_at", *wet_cols]], on="observed_at", how="outer")


POWER_SITES = (
    ("hikurangi", -39.5, 178.0),
    ("alpine", -43.5, 170.5),
    ("southland", -46.4, 168.3),
)


def _pull_power_wet(start: datetime, end: datetime) -> pd.DataFrame:
    """NASA POWER soil wetness / rain so wet weeks are visible without SMAP."""
    frames: list[pd.DataFrame] = []
    for site, lat, lon in POWER_SITES:
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=GWETTOP,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}&format=JSON"
        )
        try:
            payload = fetch_json(url, timeout=90)
        except Exception as exc:
            print(f"  hydro.surface POWER {site} skipped: {exc}", flush=True)
            continue
        props = ((payload.get("properties") or {}).get("parameter")) or {}
        wet = props.get("GWETTOP") or {}
        rain = props.get("PRECTOTCORR") or {}
        rows = []
        for key, val in wet.items() if isinstance(wet, dict) else []:
            try:
                day = datetime.strptime(str(key), "%Y%m%d").replace(tzinfo=start.tzinfo)
                number = float(val)
            except (TypeError, ValueError):
                continue
            if number > 100:
                continue
            precip = None
            if isinstance(rain, dict) and key in rain:
                try:
                    precip = float(rain[key])
                except (TypeError, ValueError):
                    precip = None
            rows.append({"observed_at": day, "hydro_gwet": number, "hydro_precip": precip, "site": site})
        if rows:
            frames.append(pd.DataFrame(rows))
            print(f"  hydro.surface POWER {site} {len(rows)} days", flush=True)
    if not frames:
        return pd.DataFrame()
    work = pd.concat(frames, ignore_index=True)
    daily = work.groupby("observed_at", as_index=False).agg(
        hydro_gwet=("hydro_gwet", "median"),
        hydro_precip=("hydro_precip", "median"),
    )
    out = []
    for rec in daily.itertuples(index=False):
        day = rec.observed_at.to_pydatetime()
        out.append(
            {
                "observed_at": iso(day.replace(hour=0, minute=0, second=0, microsecond=0)),
                "available_at": iso(day.replace(hour=0, minute=0, second=0, microsecond=0) + LATENCY),
                "hydro_gwet": float(rec.hydro_gwet),
                "hydro_precip": None if pd.isna(rec.hydro_precip) else float(rec.hydro_precip),
            }
        )
    return pd.DataFrame(out)
