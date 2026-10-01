from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SILSO_URL = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
SILSO_HEM_URL = "https://www.sidc.be/SILSO/DATA/SN_d_hem_V2.0.csv"
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


def parse_silso_hem(text: str) -> pd.DataFrame:
    """Daily hemispheric sunspot numbers. Missing days are -1.

    SILSO hem CSV is semicolon-separated: year, month, day, fraction,
    total, north, south, then error / count columns.
    """
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        parts = [p.strip() for p in raw.replace(",", ";").split(";")]
        if len(parts) < 7:
            continue
        try:
            year = int(float(parts[0]))
            month = int(float(parts[1]))
            day = int(float(parts[2]))
            north = float(parts[5])
            south = float(parts[6])
        except (TypeError, ValueError):
            continue
        if month < 1 or month > 12 or day < 1 or day > 31:
            continue
        if north < 0 and south < 0:
            continue
        try:
            observed = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append(
            {
                "source": "sidc.silso",
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "sunspot_north": None if north < 0 else north,
                "sunspot_south": None if south < 0 else south,
            }
        )
    return pd.DataFrame(rows)


def pull_silso(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frame = parse_silso(fetch_text(SILSO_URL, timeout=60))
    try:
        hem = parse_silso_hem(fetch_text(SILSO_HEM_URL, timeout=60))
    except Exception as exc:  # noqa: BLE001 — total count still lands
        print(f"  sidc.silso hemispheric skipped: {exc}")
        hem = pd.DataFrame()
    if not hem.empty:
        frame = frame.merge(hem[["observed_at", "sunspot_north", "sunspot_south"]], on="observed_at", how="left")
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return frame.loc[mask].reset_index(drop=True)
