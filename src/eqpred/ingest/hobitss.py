"""Hikurangi offshore campaign files. Both were published after the measurements.

Shear-wave splits: Nakai et al. files, public 12 September 2019.
Offshore quake list: Yarce et al. catalog, public 7 December 2018.
A morning before the publication date cannot see either file.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timezone

import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.records import iso

SOURCE = "hobitss.deep"
LICENSE = "https://zenodo.org/"
SWS_PUBLIC = datetime(2019, 9, 12, tzinfo=timezone.utc)
OBS_PUBLIC = datetime(2018, 12, 7, tzinfo=timezone.utc)
SWS_FILES = (
    ("https://zenodo.org/api/records/3383417/files/OBS_NEWbayes_50depth.summ/content", "obs"),
    ("https://zenodo.org/api/records/3383417/files/GEONET_NEWbayes_50depth.summ/content", "land"),
)
OBS_EVENTS = "https://zenodo.org/api/records/7181589/files/Yarce2019_catalog_events_v3.dat/content"

_FULL_TIME = re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.(\d{1,2})\.(\d{1,2})\.(\d{1,2})")
_SHORT_TIME = re.compile(r"(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})")


def _split_time(text: str) -> datetime | None:
    found = _FULL_TIME.match(text.strip())
    if found:
        year, month, day, hour, minute, second = (int(part) for part in found.groups())
    else:
        found = _SHORT_TIME.match(text.strip())
        if found is None:
            return None
        year, month, day, hour, minute, second = (int(part) for part in found.groups())
        year += 2000
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_sws_summary(text: str, network: str) -> pd.DataFrame:
    """One splitting summary. Grades C and worse are dropped."""
    rows: list[dict[str, object]] = []
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return pd.DataFrame()
    rename = {}
    for name in reader.fieldnames:
        plain = name
        while plain and plain[0].isdigit():
            plain = plain[1:]
        rename[name] = plain
    for raw in reader:
        rec = {rename[key]: value for key, value in raw.items()}
        grade = str(rec.get("gradeABCNR") or "")
        if not grade.startswith(("A", "B")):
            continue
        when = _split_time(str(rec.get("event") or ""))
        if when is None:
            continue
        try:
            delay = float(rec.get("tlag") or "")
        except ValueError:
            continue
        rows.append(
            {
                "source": SOURCE,
                "network": network,
                "observed_at": iso(when),
                "available_at": iso(SWS_PUBLIC),
                "license": LICENSE,
                "station": str(rec.get("stat") or "").strip(),
                "tlag_s": delay,
                "grade": grade,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def parse_obs_catalog(text: str) -> pd.DataFrame:
    """Yarce event list: evid, lat, lon, depth, then a date and time."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 11 or parts[0] == "evid":
            continue
        try:
            lat = float(parts[1])
            lon = float(parts[2])
            depth = float(parts[3])
            magnitude = float(parts[5])
            when = datetime.strptime(f"{parts[-2]} {parts[-1]}", "%m/%d/%Y %H:%M:%S.%f")
        except ValueError:
            try:
                when = datetime.strptime(f"{parts[-2]} {parts[-1]}", "%m/%d/%Y %H:%M:%S")
            except ValueError:
                continue
        when = when.replace(tzinfo=timezone.utc)
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(when),
                "available_at": iso(OBS_PUBLIC),
                "license": LICENSE,
                "lat": lat,
                "lon": lon,
                "depth_km": depth,
                "magnitude": magnitude,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def pull_hobitss(ingest_run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits: list[pd.DataFrame] = []
    for url, network in SWS_FILES:
        print(f"  hobitss.deep splits {network}", flush=True)
        text = fetch_bytes(url, timeout=120).decode("utf-8", errors="replace")
        part = parse_sws_summary(text, network)
        if not part.empty:
            splits.append(part)
    sws = pd.concat(splits, ignore_index=True) if splits else pd.DataFrame()
    print("  hobitss.deep offshore catalog", flush=True)
    catalog = parse_obs_catalog(fetch_bytes(OBS_EVENTS, timeout=120).decode("utf-8", errors="replace"))
    if not sws.empty:
        sws["ingest_run_id"] = ingest_run_id
    if not catalog.empty:
        catalog["ingest_run_id"] = ingest_run_id
    return sws, catalog
