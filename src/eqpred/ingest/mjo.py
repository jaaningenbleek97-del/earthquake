from __future__ import annotations

"""Madden–Julian Oscillation RMM from the Bureau of Meteorology.

Daily tropical-atmosphere series. Burns talks about global circulation
coupling; this is the public daily index.
"""

import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

URL = "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
OMI_URL = "https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt"
LICENSE = "http://www.bom.gov.au/climate/mjo/"
OMI_LICENSE = "https://psl.noaa.gov/mjo/mjoindex/"
LATENCY = timedelta(days=3)
SOURCE = "bom.mjo"


def rmm_phase(rmm1: float, rmm2: float) -> float:
    """Wheeler–Hendon 8-phase number; phase 5 is centred on +RMM1."""
    deg = math.degrees(math.atan2(rmm2, rmm1))
    idx = int(math.floor((deg + 22.5) / 45.0))
    mapping = {0: 5, 1: 6, 2: 7, 3: 8, 4: 1, -1: 4, -2: 3, -3: 2, -4: 1}
    return float(mapping.get(idx, ((idx % 8) + 8) % 8 + 1))


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


def parse_omi(text: str) -> pd.DataFrame:
    """PSL OMI daily PC1 PC2 amplitude. Fills the BOM RMM gap after 2024-02-24."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            if year < 1979 or year > 2100:
                continue
            observed = datetime(year, month, day, tzinfo=timezone.utc)
            pc1 = float(parts[3])
            pc2 = float(parts[4])
            amp = float(parts[5]) if len(parts) >= 6 else math.hypot(pc1, pc2)
        except (ValueError, OverflowError):
            continue
        if abs(pc1) > 20 or abs(pc2) > 20 or amp > 20:
            continue
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": OMI_LICENSE,
                "climate_mjo_rmm1": pc1,
                "climate_mjo_rmm2": pc2,
                "climate_mjo_phase": rmm_phase(pc1, pc2),
                "climate_mjo_amp": amp,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def pull_mjo(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frame = parse_rmm(fetch_text(URL, timeout=90))
    try:
        omi = parse_omi(fetch_text(OMI_URL, timeout=90))
    except Exception as exc:  # noqa: BLE001 — BOM alone is still usable
        print(f"  bom.mjo OMI skip: {exc}")
        omi = pd.DataFrame()
    if frame.empty:
        frame = omi
    elif not omi.empty:
        extra = omi.loc[~omi["observed_at"].isin(set(frame["observed_at"]))]
        if not extra.empty:
            frame = pd.concat([frame, extra], ignore_index=True)
    if frame.empty:
        return frame
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return frame.loc[mask].sort_values("observed_at").reset_index(drop=True)
