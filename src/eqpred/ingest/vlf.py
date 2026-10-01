from __future__ import annotations

"""VLF amplitude trap. Hunt confirm = leak.

A careful Tohoku VLF study found no pre-quake shift. NWC–Dunedin AARDDVARK
is not in the public Zenodo dump (DHO/NAA/NRK only). Whatever public path
answers is ingested as a foil, not a precursor.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd

from eqpred.ingest.http import fetch_bytes, fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "vlf.public"
LICENSE = "https://zenodo.org/records/18740376"
LATENCY = timedelta(days=2)
ZENODO = "https://zenodo.org/api/records/18740376"
SUPERSID = (
    "http://sidstation.loudet.org/data/csv/",
    "https://sidstation.loudet.org/data/csv/",
)
NWC_RE = re.compile(r'href="([^"]*NWC[^"]+\.csv)"', re.I)


def parse_vlf_csv(text: str, path: str = "") -> pd.DataFrame:
    if not text or "," not in text and "\t" not in text:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(StringIO(text), comment="#")
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    time_col = next((c for c in raw.columns if re.search(r"time|date", str(c), re.I)), raw.columns[0])
    num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    if not num_cols:
        return pd.DataFrame()
    stamp = pd.to_datetime(raw[time_col], utc=True, errors="coerce")
    amp = pd.to_numeric(raw[num_cols[0]], errors="coerce")
    work = pd.DataFrame({"observed_at": stamp, "vlf_amp": amp}).dropna()
    if work.empty:
        return pd.DataFrame()
    work["observed_at"] = work["observed_at"].dt.floor("D")
    daily = work.groupby("observed_at", as_index=False)["vlf_amp"].median()
    daily["vlf_path"] = path
    daily["vlf_nwc"] = 1.0 if "nwc" in path.lower() else 0.0
    return daily


def pull_vlf(start: datetime, end: datetime, run_id: str, *, fetch_page=fetch_text, fetch=fetch_bytes) -> pd.DataFrame:
    _ = (run_id, fetch_page, fetch, utc_ts(start), utc_ts(end))
    print(
        "  vlf.public skipped: NWC–Dunedin AARDDVARK is account-gated; "
        "Zenodo dump is northern-path foil only. No wget.",
        flush=True,
    )
    return pd.DataFrame()
