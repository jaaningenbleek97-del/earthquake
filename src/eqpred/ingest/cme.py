from __future__ import annotations

"""SOHO/LASCO CME onsets from the CDAW catalog.

Halo / fast CMEs are the leftover solar-ejecta list Davidson watches.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

URL = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt"
LICENSE = "https://cdaw.gsfc.nasa.gov/CME_list/"
LATENCY = timedelta(days=3)
SOURCE = "cdaw.cme"
HOLD_DAYS = 2


def parse_cdaw_cme(text: str) -> pd.DataFrame:
    """`YYYY/MM/DD  HH:MM   CPA  width  ...  speed` — one row per CME day max."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 16 or not line[:4].isdigit() or line[4] != "/":
            continue
        try:
            onset = datetime.strptime(line[:10], "%Y/%m/%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        parts = line.split()
        halo = 1.0 if "Halo" in line or "halo" in line else 0.0
        speed = None
        for token in parts[4:]:
            try:
                num = float(token)
            except ValueError:
                continue
            if 50.0 <= num <= 5000.0:
                speed = num
                break
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(onset),
                "available_at": iso(onset + LATENCY),
                "license": LICENSE,
                "cme_count": 1.0,
                "cme_halo": halo,
                "cme_speed_kms": speed,
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    frame["_day"] = stamp.dt.floor("D")
    daily = (
        frame.groupby("_day", as_index=False)
        .agg(
            cme_count=("cme_count", "sum"),
            cme_halo=("cme_halo", "max"),
            cme_speed_kms=("cme_speed_kms", "max"),
        )
        .rename(columns={"_day": "observed_at"})
    )
    daily["source"] = SOURCE
    daily["license"] = LICENSE
    daily["observed_at"] = daily["observed_at"].map(lambda ts: iso(ts.to_pydatetime()))
    daily["available_at"] = (
        pd.to_datetime(daily["observed_at"], utc=True) + LATENCY
    ).map(lambda ts: iso(ts.to_pydatetime()))
    return daily.sort_values("observed_at").reset_index(drop=True)


def _daily_flags(events: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    onset = pd.to_datetime(events["observed_at"], utc=True).dt.floor("D")
    days = pd.date_range(utc_ts(start), utc_ts(end), freq="D", inclusive="left")
    if days.empty:
        return pd.DataFrame()
    flags = []
    for day in days:
        window = (onset >= day - pd.Timedelta(days=HOLD_DAYS - 1)) & (onset <= day)
        match = events.loc[window]
        if match.empty:
            flags.append(
                {
                    "source": SOURCE,
                    "observed_at": iso(day.to_pydatetime()),
                    "available_at": iso((day + pd.Timedelta(days=1)).to_pydatetime()),
                    "license": LICENSE,
                    "cme_count": 0.0,
                    "cme_halo": 0.0,
                    "cme_speed_kms": None,
                }
            )
            continue
        flags.append(
            {
                "source": SOURCE,
                "observed_at": iso(day.to_pydatetime()),
                "available_at": iso((day + pd.Timedelta(days=1)).to_pydatetime()),
                "license": LICENSE,
                "cme_count": float(match["cme_count"].sum()),
                "cme_halo": float(match["cme_halo"].max()),
                "cme_speed_kms": float(match["cme_speed_kms"].max())
                if match["cme_speed_kms"].notna().any()
                else None,
            }
        )
    return pd.DataFrame(flags).sort_values("observed_at").reset_index(drop=True)


def pull_cme(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    events = parse_cdaw_cme(fetch_text(URL, timeout=180))
    if events.empty:
        return pd.DataFrame()
    out = _daily_flags(events, start, end)
    if out.empty:
        return out
    out["ingest_run_id"] = ingest_run_id
    return out
