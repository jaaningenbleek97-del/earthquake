from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SILSO_URL = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
LICENSE = "https://www.sidc.be/SILSO/infosndtot"
LATENCY = timedelta(days=2)


def parse_silso(text: str) -> pd.DataFrame:
    raw = pd.read_csv(
        StringIO(text),
        sep=";",
        header=None,
        names=["year", "month", "day", "frac_year", "sunspot_number", "std", "n_obs", "definitive"],
    )
    rows: list[dict[str, object]] = []
    for rec in raw.itertuples(index=False):
        spots = float(rec.sunspot_number)
        if spots < 0:
            continue
        observed = datetime(int(rec.year), int(rec.month), int(rec.day), tzinfo=timezone.utc)
        rows.append(
            {
                "source": "sidc.silso",
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "sunspot_number": spots,
            }
        )
    return pd.DataFrame(rows)


def pull_silso(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frame = parse_silso(fetch_text(SILSO_URL, timeout=60))
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return frame.loc[mask].reset_index(drop=True)
