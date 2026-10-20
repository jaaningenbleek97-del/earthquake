"""Japan trench tremor from the Slow Earthquake Database NIED list.

Practice box only. Same 30-day publication hold as the New Zealand tremor list.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from eqpred.ingest.tremor import LATENCY, LICENSE, pull_eqdb_catalog
from eqpred.records import iso
from eqpred.regions import REGIONS, point_in_region

SOURCE = "eqdb.nied"
CATALOG = "NIED-Tremor"


def in_japan(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    region = REGIONS["japan"]
    mask = []
    for lat, lon in zip(events["lat"], events["lon"], strict=True):
        if pd.isna(lat) or pd.isna(lon):
            mask.append(False)
            continue
        mask.append(point_in_region(float(lat), float(lon), region))
    return events.loc[mask].copy()


def pull_nied(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    print(f"  eqdb.nied {CATALOG}", flush=True)
    events = in_japan(pull_eqdb_catalog(CATALOG, start, end))
    if events.empty:
        return pd.DataFrame()
    observed = pd.to_datetime(events["observed_at"], utc=True)
    events = events.copy()
    events["available_at"] = [(stamp + LATENCY) for stamp in observed]
    events["source"] = SOURCE
    events["license"] = LICENSE
    events["ingest_run_id"] = ingest_run_id
    events["observed_at"] = [iso(stamp.to_pydatetime()) for stamp in observed]
    events["available_at"] = [iso(stamp.to_pydatetime()) for stamp in events["available_at"]]
    return events.reset_index(drop=True)
