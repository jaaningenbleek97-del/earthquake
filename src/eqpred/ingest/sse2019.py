"""Northern Hikurangi earthquakes around the 2019 slow slip.

Zenodo record 15713529. The catalog file is public from 31 July 2025.
The waveform archive beside it is left alone.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.records import iso

SOURCE = "hikurangi.sse2019"
LICENSE = "https://zenodo.org/records/15713529"
URL = "https://zenodo.org/api/records/15713529/files/Earthquake_catalog.csv/content"
WAVEFORMS = "https://zenodo.org/api/records/15713529/files/Event_waveform.tgz/content"
PUBLIC = datetime(2025, 7, 31, tzinfo=timezone.utc)


def parse_sse2019_catalog(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text))
    if frame.empty or "time" not in frame.columns:
        return pd.DataFrame()
    observed = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "source": SOURCE,
            "observed_at": observed,
            "lat": pd.to_numeric(frame.get("lat"), errors="coerce"),
            "lon": pd.to_numeric(frame.get("lon"), errors="coerce"),
            "depth_km": pd.to_numeric(frame.get("dep"), errors="coerce"),
            "magnitude": pd.to_numeric(frame.get("mag"), errors="coerce"),
            "license": LICENSE,
        }
    ).dropna(subset=["observed_at"])
    out["available_at"] = iso(PUBLIC)
    out["observed_at"] = out["observed_at"].map(lambda stamp: iso(stamp.to_pydatetime()))
    return out.reset_index(drop=True)


def pull_sse2019(ingest_run_id: str) -> pd.DataFrame:
    print("  hikurangi.sse2019 catalog", flush=True)
    text = fetch_bytes(URL, timeout=60).decode("utf-8", errors="replace")
    frame = parse_sse2019_catalog(text)
    if frame.empty:
        return frame
    frame["ingest_run_id"] = ingest_run_id
    return frame


def pull_sse2019_waveforms(dest) -> object:
    """The event waveforms beside the catalog. About 224 MB, not a daily number."""
    from pathlib import Path

    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"  hikurangi.sse2019 waveforms already at {target}", flush=True)
        return target
    print("  hikurangi.sse2019 waveforms", flush=True)
    target.write_bytes(fetch_bytes(WAVEFORMS, timeout=600))
    return target
