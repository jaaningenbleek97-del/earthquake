from __future__ import annotations

"""Madden–Julian Oscillation RMM from the Bureau of Meteorology.

Daily tropical-atmosphere leftover. Burns talks about global circulation
coupling; this is the public daily index.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

URL = "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
LICENSE = "http://www.bom.gov.au/climate/mjo/"
LATENCY = timedelta(days=3)
SOURCE = "bom.mjo"


def parse_rmm(text: str) -> pd.DataFrame:
    """Year month day ... RMM1 RMM2 phase amplitude."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 7:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            if year < 1970 or year > 2100:
                continue
            observed = datetime(year, month, day, tzinfo=timezone.utc)
            # BOM layout: year month day RMM1 RMM2 phase amplitude ...
            rmm1 = float(parts[3])
            rmm2 = float(parts[4])
            phase = float(parts[5])
            amp = float(parts[6])
        except (ValueError, OverflowError):
            continue
        if abs(rmm1) > 20 or abs(rmm2) > 20 or amp > 20 or not (0 <= phase <= 8):
            continue
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "climate_mjo_rmm1": rmm1,
                "climate_mjo_rmm2": rmm2,
                "climate_mjo_phase": phase,
                "climate_mjo_amp": amp,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def pull_mjo(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frame = parse_rmm(fetch_text(URL, timeout=90))
    if frame.empty:
        return frame
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return frame.loc[mask].reset_index(drop=True)
