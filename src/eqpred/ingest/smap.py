from __future__ import annotations

"""SMAP L3 soil moisture, NZ subset only.

Global daily SPL3SMP granules are ~30 MB/day (~120 GB for the mission).
This puller asks CMR for SPL3SMP v9 and downloads only if Earthdata
credentials exist. It never wget's the global archive.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import earthdata_credentials, fetch_bytes, fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "nasa.smap"
LICENSE = "https://nsidc.org/data/spl3smp"
LATENCY = timedelta(days=3)
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
# NZ bounding box used for granule search. Harmony subset would be better;
# without a token we stop after listing.
NZ_BBOX = "165,-48,-175,-34"


def _cmr_granules(start: datetime, end: datetime, *, fetch_page=fetch_text) -> list[dict]:
    params = (
        f"{CMR}?short_name=SPL3SMP&version=009"
        f"&temporal={start.strftime('%Y-%m-%dT00:00:00Z')},{end.strftime('%Y-%m-%dT00:00:00Z')}"
        f"&bounding_box={NZ_BBOX}&page_size=20&sort_key=-start_date"
    )
    try:
        payload = json.loads(fetch_page(params, timeout=40))
    except Exception as exc:
        print(f"  nasa.smap CMR skipped: {exc}", flush=True)
        return []
    entries = ((payload.get("feed") or {}).get("entry")) or []
    return [e for e in entries if isinstance(e, dict)]


def parse_smap_nz_mean(data: bytes) -> float | None:
    """Best-effort NZ mean from a downloaded granule. HDF5 if h5py exists."""
    if not data or data[:8] != b"\x89HDF\r\n\x1a\n":
        return None
    try:
        import h5py
        import numpy as np
    except ImportError:
        return None
    import io

    try:
        with h5py.File(io.BytesIO(data), "r") as handle:
            for path in (
                "Soil_Moisture_Retrieval_Data_AM/soil_moisture",
                "Soil_Moisture_Retrieval_Data/soil_moisture",
            ):
                if path not in handle:
                    continue
                arr = np.asarray(handle[path][()], dtype=float)
                lat_path = path.rsplit("/", 1)[0] + "/latitude"
                lon_path = path.rsplit("/", 1)[0] + "/longitude"
                if lat_path in handle and lon_path in handle:
                    lat = np.asarray(handle[lat_path][()], dtype=float)
                    lon = np.asarray(handle[lon_path][()], dtype=float)
                    lon = np.where(lon < 0, lon + 360.0, lon)
                    mask = (lat >= -48) & (lat <= -34) & (lon >= 165) & (lon <= 185)
                    arr = arr[mask]
                vals = arr[np.isfinite(arr) & (arr >= 0) & (arr <= 1)]
                if vals.size:
                    return float(vals.mean())
    except Exception:
        return None
    return None


def pull_smap(start: datetime, end: datetime, run_id: str, *, fetch_page=fetch_text, fetch=fetch_bytes) -> pd.DataFrame:
    _ = run_id
    start = utc_ts(start)
    end = utc_ts(end)
    creds = earthdata_credentials()
    if not creds:
        print("  nasa.smap blocked: Earthdata login missing; will not fetch global 30 MB/day granules", flush=True)
        return pd.DataFrame()
    windows: list[tuple[datetime, datetime]] = []
    month = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while month < end:
        nxt = datetime(month.year + (month.month // 12), (month.month % 12) + 1, 1, tzinfo=timezone.utc)
        windows.append((max(month, start), min(nxt, end)))
        month = nxt
    recent = end - timedelta(days=14)
    if recent > start:
        windows.append((recent, end))
    rows = []
    print(f"  nasa.smap {len(windows)} month/recent windows creds=yes", flush=True)
    for win_start, win_end in windows:
        granules = _cmr_granules(win_start, win_end, fetch_page=fetch_page)
        take = granules[:1]
        for entry in take:
            links = [ln.get("href") for ln in (entry.get("links") or []) if isinstance(ln, dict)]
            href = next((h for h in links if h and h.endswith(".h5")), None)
            if not href:
                continue
            stamp = pd.to_datetime(entry.get("time_start"), utc=True, errors="coerce")
            if pd.isna(stamp):
                continue
            try:
                data = fetch(href, timeout=180, earthdata=True)
            except Exception as exc:
                print(f"    skip {href.split('/')[-1]}: {exc}", flush=True)
                continue
            mean = parse_smap_nz_mean(data)
            if mean is None:
                continue
            day = stamp.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
            rows.append(
                {
                    "source": SOURCE,
                    "observed_at": iso(day),
                    "available_at": iso(day + LATENCY),
                    "smap_sm": mean,
                    "smap_nz": mean,
                }
            )
            print(f"    {day.date()} smap_sm={mean:.3f}", flush=True)
    return pd.DataFrame(rows)
