"""GeoNet moment-tensor list. Separate from the earthquake catalog.

The public file has no solution time, only the origin. A solution is held
7 days, the same wait used for a quick Global CMT.
"""

from __future__ import annotations

import io
from datetime import timedelta

import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.records import iso

SOURCE = "geonet.cmt"
LICENSE = "https://github.com/GeoNet/data/tree/main/moment-tensor"
URL = "https://raw.githubusercontent.com/GeoNet/data/main/moment-tensor/GeoNet_CMT_solutions.csv"
HOLD = timedelta(days=7)


def parse_geonet_cmt(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text))
    if frame.empty or "Date" not in frame.columns:
        return pd.DataFrame()
    stamps = pd.to_datetime(frame["Date"].astype(str), format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "source": SOURCE,
            "event_id": frame.get("PublicID", pd.Series(dtype=object)).astype(str),
            "observed_at": stamps,
            "lat": pd.to_numeric(frame.get("Latitude"), errors="coerce"),
            "lon": pd.to_numeric(frame.get("Longitude"), errors="coerce"),
            "mw": pd.to_numeric(frame.get("Mw"), errors="coerce"),
            "rake": pd.to_numeric(frame.get("rake1"), errors="coerce"),
            "license": LICENSE,
        }
    ).dropna(subset=["observed_at", "lat", "lon", "mw"])
    out["available_at"] = out["observed_at"] + HOLD
    out["observed_at"] = out["observed_at"].map(lambda stamp: iso(stamp.to_pydatetime()))
    out["available_at"] = out["available_at"].map(lambda stamp: iso(stamp.to_pydatetime()))
    return out.sort_values("observed_at").reset_index(drop=True)


def pull_geonet_cmt(ingest_run_id: str) -> pd.DataFrame:
    print("  geonet.cmt", flush=True)
    text = fetch_bytes(URL, timeout=60).decode("utf-8", errors="replace")
    frame = parse_geonet_cmt(text)
    if frame.empty:
        return frame
    frame["ingest_run_id"] = ingest_run_id
    return frame
