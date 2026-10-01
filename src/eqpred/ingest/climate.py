from __future__ import annotations

"""Monthly climate indices from NOAA PSL. Replayable ASCII."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

LICENSE = "https://psl.noaa.gov/data/climateindices/"
# Index for month M is treated as known a week into the next month.
LATENCY = timedelta(days=38)

PSL_SERIES = (
    ("climate_soi", "https://psl.noaa.gov/data/correlation/soi.data"),
    ("climate_nao", "https://psl.noaa.gov/data/correlation/nao.data"),
    ("climate_ao", "https://psl.noaa.gov/data/correlation/ao.data"),
    ("climate_pna", "https://psl.noaa.gov/data/correlation/pna.data"),
    ("climate_aao", "https://psl.noaa.gov/data/correlation/aao.data"),
    ("climate_qbo", "https://psl.noaa.gov/data/correlation/qbo.data"),
    ("climate_pdo", "https://psl.noaa.gov/data/correlation/pdo.data"),
    ("climate_amo", "https://psl.noaa.gov/data/correlation/amon.us.data"),
    ("climate_nina34", "https://psl.noaa.gov/data/correlation/nina34.data"),
    ("climate_wp", "https://psl.noaa.gov/data/correlation/wp.data"),
    ("climate_ea", "https://psl.noaa.gov/data/correlation/ea.data"),
    ("climate_olr", "https://psl.noaa.gov/data/correlation/olr.data"),
    ("climate_nina3", "https://psl.noaa.gov/data/correlation/nina3.data"),
    ("climate_nina4", "https://psl.noaa.gov/data/correlation/nina4.data"),
    ("climate_tna", "https://psl.noaa.gov/data/correlation/tna.data"),
    ("climate_tsa", "https://psl.noaa.gov/data/correlation/tsa.data"),
)


def parse_psl_monthly(text: str, column: str) -> pd.DataFrame:
    """PSL year-range header, then `year m1..m12` with a large missing sentinel."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if year < 1800 or year > 2100:
            continue
        values = []
        ok = True
        for token in parts[1:13]:
            try:
                values.append(float(token))
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        finite = [v for v in values if abs(v) < 90]
        if not finite:
            continue
        fill = 99.0
        for month, value in enumerate(values, start=1):
            if abs(value) >= fill:
                continue
            observed = datetime(year, month, 1, tzinfo=timezone.utc)
            rows.append(
                {
                    "source": "noaa.psl",
                    "observed_at": iso(observed),
                    "available_at": iso(observed + LATENCY),
                    "license": LICENSE,
                    column: value,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def pull_climate(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column, url in PSL_SERIES:
        try:
            text = fetch_text(url, timeout=90)
        except Exception as exc:  # noqa: BLE001 — keep the other indices
            print(f"  noaa.psl skip {column}: {exc}")
            continue
        frame = parse_psl_monthly(text, column)
        if frame.empty:
            print(f"  noaa.psl empty {column}")
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        cols = [c for c in extra.columns if str(c).startswith("climate_")]
        out = out.merge(extra[["observed_at", *cols]], on="observed_at", how="outer")
    out["source"] = "noaa.psl"
    out["license"] = LICENSE
    out["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(out["observed_at"], utc=True)
    out["available_at"] = (stamp + LATENCY).map(iso)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return out.loc[mask].sort_values("observed_at").reset_index(drop=True)
