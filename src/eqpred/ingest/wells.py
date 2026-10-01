from __future__ import annotations

"""NZ regional-council groundwater (Hilltop) plus a few USGS NWIS wells.

Wells respond to strain and shaking. Most published spikes are coseismic;
this is still a replayable hydrograph overlay the lab did not have. Levels
are stored so that a rise means more water (USGS depth-to-water is flipped).
"""

import json
import re
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
from xml.etree import ElementTree as ET

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "hydro.wells"
LICENSE = "https://www.lawa.org.nz/"
USGS_LICENSE = "https://waterdata.usgs.gov/nwis"
LATENCY = timedelta(days=2)
MEASUREMENTS = (
    "Stage",
    "Water Level",
    "Groundwater Level",
    "RWL",
    "Depth to Water",
    "Well Level",
)
HILLTOP_SERVERS: tuple[tuple[str, str], ...] = (
    ("https://hydro.marlborough.govt.nz/data.hts", "alpine"),
    ("https://hilltop.gw.govt.nz/Data.hts", "hikurangi"),
    ("https://envdata.tasman.govt.nz/data.hts", "alpine"),
)
WELL_NAME_RE = re.compile(r"\b(well|bore|borefield|piezo|groundwater|gw zone)\b", re.I)
MAX_SITES_PER_SERVER = 4
# Long public daily groundwater. Parameter 72019 is feet below land surface.
USGS_SITES: tuple[tuple[str, str], ...] = (
    ("323527117050001", "usgs_ca"),
    ("323527117050002", "usgs_ca"),
    ("323527117050003", "usgs_ca"),
    ("324448117035801", "usgs_ca"),
    ("325411117060801", "usgs_ca"),
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_hilltop_xml(text: str, station: str) -> pd.DataFrame:
    """Hilltop GetData XML → UTC samples. Higher value = more water."""
    if not text or "<" not in text:
        return pd.DataFrame()
    if re.search(r"<Error\b", text, flags=re.I):
        return pd.DataFrame()
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return pd.DataFrame()
    rows: list[tuple[datetime, float]] = []
    for node in root.iter():
        name = _local_name(node.tag)
        if name == "e":
            stamp = None
            value = None
            for child in list(node):
                child_name = _local_name(child.tag)
                if child_name in {"t", "datetime"} and child.text:
                    stamp = child.text.strip()
                if child_name in {"i1", "i2", "value", "v"} and child.text:
                    value = child.text.strip()
            if stamp and value:
                rows.append(_pair(stamp, value))
        elif name in {"v", "value"}:
            stamp = node.attrib.get("DateTime") or node.attrib.get("t")
            if stamp and node.text:
                rows.append(_pair(stamp, node.text.strip()))
    pairs = [(when, val) for when, val in rows if when is not None and val is not None]
    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "observed_at": [iso(when) for when, _ in pairs],
            "well_level_m": [val for _, val in pairs],
            "well_station": station,
        }
    )


def _pair(stamp: str, value: str) -> tuple[datetime | None, float | None]:
    try:
        when = pd.to_datetime(stamp, utc=True).to_pydatetime()
    except (TypeError, ValueError):
        return None, None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, None
    if number != number:  # NaN
        return None, None
    return when, number


def parse_nwis_dv(payload: object, default_station: str = "") -> pd.DataFrame:
    """USGS daily values JSON. 72019 depth-to-water is flipped to a water level."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return pd.DataFrame()
    if not isinstance(payload, dict):
        return pd.DataFrame()
    series = ((payload.get("value") or {}).get("timeSeries")) or []
    rows: list[dict[str, object]] = []
    for block in series:
        if not isinstance(block, dict):
            continue
        source_info = block.get("sourceInfo") or {}
        site = str((source_info.get("siteCode") or [{}])[0].get("value") or default_station)
        var = ((block.get("variable") or {}).get("variableCode") or [{}])[0]
        code = str(var.get("value") or "")
        unit = str(((block.get("variable") or {}).get("unit") or {}).get("unitCode") or "")
        values = ((block.get("values") or [{}])[0].get("value")) or []
        for rec in values:
            if not isinstance(rec, dict):
                continue
            stamp = rec.get("dateTime")
            raw = rec.get("value")
            if stamp is None or raw in (None, "", "-999999"):
                continue
            try:
                when = pd.to_datetime(stamp, utc=True).to_pydatetime()
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if "ft" in unit.lower():
                number *= 0.3048
            if code == "72019":
                number = -number
            rows.append(
                {
                    "observed_at": iso(when),
                    "well_level_m": number,
                    "well_station": site or default_station,
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def daily_well_features(samples: pd.DataFrame) -> pd.DataFrame:
    """One morning per station: mean level and day-to-day change."""
    if samples.empty or "well_level_m" not in samples.columns:
        return pd.DataFrame()
    work = samples.dropna(subset=["well_level_m"]).copy()
    if work.empty:
        return pd.DataFrame()
    # Hilltop Stage is often millimetres. Keep metres so d1 is comparable.
    for station, part in work.groupby("well_station"):
        if part["well_level_m"].abs().median() > 50:
            work.loc[work["well_station"] == station, "well_level_m"] = part["well_level_m"] / 1000.0
    work["obs_day"] = pd.to_datetime(work["observed_at"], utc=True).dt.floor("D")
    agg = {
        "well_level_m": ("well_level_m", "mean"),
        "well_range_m": ("well_level_m", lambda s: float(s.max() - s.min()) if len(s) else None),
    }
    if "well_region" in work.columns:
        agg["well_region"] = ("well_region", "first")
    grouped = (
        work.groupby(["obs_day", "well_station"], as_index=False)
        .agg(**agg)
        .sort_values(["well_station", "obs_day"])
    )
    grouped["well_d1_m"] = grouped.groupby("well_station")["well_level_m"].diff(1)
    rows: list[dict[str, object]] = []
    for rec in grouped.itertuples(index=False):
        day = rec.obs_day.to_pydatetime()
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(day),
                "available_at": iso(day + LATENCY),
                "license": LICENSE,
                "well_station": rec.well_station,
                "well_level_m": float(rec.well_level_m),
                "well_range_m": None if rec.well_range_m is None else float(rec.well_range_m),
                "well_d1_m": None if pd.isna(rec.well_d1_m) else float(rec.well_d1_m),
                "well_region": getattr(rec, "well_region", None),
            }
        )
    return pd.DataFrame(rows)


def _hilltop_get(base: str, site: str, measurement: str, start: datetime, end: datetime) -> str:
    query = urlencode(
        {
            "Service": "Hilltop",
            "Request": "GetData",
            "Site": site,
            "Measurement": measurement,
            "From": utc_ts(start).strftime("%d/%m/%Y"),
            "To": utc_ts(end).strftime("%d/%m/%Y"),
        },
        quote_via=quote,
    )
    return fetch_text(f"{base}?{query}", timeout=40)


def _hilltop_probe_measurement(base: str, site: str, start: datetime, end: datetime) -> str:
    """Find the measurement that actually has rows. Probe a recent year first."""
    last_year = utc_ts(end).year
    first_year = max(utc_ts(start).year, 2000)
    probe_years = []
    for year in (last_year - 1, last_year, 2016, first_year):
        if first_year <= year <= last_year and year not in probe_years:
            probe_years.append(year)
    for year in probe_years:
        chunk_start = datetime(year, 1, 1, tzinfo=utc_ts(start).tzinfo)
        chunk_end = datetime(year + 1, 1, 1, tzinfo=utc_ts(start).tzinfo)
        if chunk_start < utc_ts(start):
            chunk_start = utc_ts(start)
        if chunk_end > utc_ts(end):
            chunk_end = utc_ts(end)
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


def _hilltop_site_frame(base: str, site: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Year-sized Hilltop pulls. A 1990–now Stage dump can be tens of MB."""
    frames: list[pd.DataFrame] = []
    year = max(utc_ts(start).year, 2000)
    last_year = utc_ts(end).year
    measurement = _hilltop_probe_measurement(base, site, start, end)
    while year <= last_year:
        chunk_start = datetime(year, 1, 1, tzinfo=utc_ts(start).tzinfo)
        chunk_end = datetime(year + 1, 1, 1, tzinfo=utc_ts(start).tzinfo)
        if chunk_start < utc_ts(start):
            chunk_start = utc_ts(start)
        if chunk_end > utc_ts(end):
            chunk_end = utc_ts(end)
        try:
            text = _hilltop_get(base, site, measurement, chunk_start, chunk_end)
            parsed = parse_hilltop_xml(text, site)
        except Exception:
            parsed = pd.DataFrame()
        if not parsed.empty:
            print(f"    {year} {len(parsed)}", flush=True)
            frames.append(parsed)
        year += 1
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _discover_hilltop_wells(base: str) -> list[str]:
    found: list[str] = []
    for measurement in MEASUREMENTS:
        query = urlencode({"Service": "Hilltop", "Request": "SiteList", "Measurement": measurement})
        try:
            text = fetch_text(f"{base}?{query}", timeout=40)
        except Exception:
            continue
        names = re.findall(r'Site Name="([^"]+)"', text)
        wells = [name for name in names if WELL_NAME_RE.search(name)]
        for name in wells:
            if name not in found:
                found.append(name)
        if found:
            break
    found.sort(key=lambda name: (0 if re.search(r"monitor|observ", name, re.I) else 1, name))
    return found[:MAX_SITES_PER_SERVER]


def _pull_hilltop(start: datetime, end: datetime) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    regions: dict[str, str] = {}
    for base, region in HILLTOP_SERVERS:
        sites = _discover_hilltop_wells(base)
        print(f"  hydro.wells {base} {len(sites)} well sites", flush=True)
        for site in sites:
            regions[site] = region
            print(f"  hydro.wells {site}", flush=True)
            parsed = _hilltop_site_frame(base, site, start, end)
            if parsed.empty:
                print("    empty", flush=True)
                continue
            print(f"    {len(parsed)} samples", flush=True)
            frames.append(parsed)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["well_region"] = out["well_station"].map(regions)
    return out


def _pull_usgs(start: datetime, end: datetime) -> pd.DataFrame:
    sites = ",".join(site for site, _region in USGS_SITES)
    url = (
        "https://waterservices.usgs.gov/nwis/dv/?format=json"
        f"&sites={sites}&startDT={utc_ts(start).date()}&endDT={utc_ts(end).date()}"
        "&parameterCd=72019,62610"
    )
    print("  hydro.wells USGS NWIS", flush=True)
    try:
        text = fetch_text(url, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"    USGS skip: {exc}", flush=True)
        return pd.DataFrame()
    frame = parse_nwis_dv(text)
    if frame.empty:
        return frame
    lookup = {site: region for site, region in USGS_SITES}
    frame["well_region"] = frame["well_station"].map(lookup).fillna("usgs")
    frame["license"] = USGS_LICENSE
    print(f"    USGS {len(frame)} samples", flush=True)
    return frame


def pull_wells(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    parts = [part for part in (_pull_hilltop(start, end), _pull_usgs(start, end)) if not part.empty]
    if not parts:
        return pd.DataFrame()
    samples = pd.concat(parts, ignore_index=True)
    if samples.empty:
        return pd.DataFrame()
    daily = daily_well_features(samples)
    if daily.empty:
        return daily
    region_lookup = {site: region for site, region in USGS_SITES}
    if "well_region" not in daily.columns:
        daily["well_region"] = daily["well_station"].map(region_lookup)
    else:
        daily["well_region"] = daily["well_region"].fillna(daily["well_station"].map(region_lookup))
    daily["ingest_run_id"] = ingest_run_id
    return daily
