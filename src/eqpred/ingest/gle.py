from __future__ import annotations

"""Ground-level enhancement onsets from the International GLE Database.

Replayable JSON from gle.oulu.fi. This is an event list — not another
daily neutron-count transform.
"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

GRID_URL = "https://gle.oulu.fi/grid.php"
LICENSE = "https://gle.oulu.fi/"
LATENCY = timedelta(days=1)
SOURCE = "oulu.gle"
# Keep the flag on for a short window so a morning hunt can see yesterday's GLE.
HOLD_DAYS = 2


def parse_gle_grid(text: str) -> pd.DataFrame:
    """`{"gles":[{"gle":"72","date":"2017-09-10",...}, ...]}` → one row per onset day."""
    pack = json.loads(text)
    rows: list[dict[str, object]] = []
    for item in pack.get("gles") or []:
        raw = str(item.get("date") or "").strip()
        if len(raw) < 10:
            continue
        try:
            onset = datetime.fromisoformat(raw[:10]).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        gle_id = item.get("gle")
        try:
            gle_num = int(gle_id)
        except (TypeError, ValueError):
            gle_num = None
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(onset),
                "available_at": iso(onset + LATENCY),
                "license": LICENSE,
                "cosmic_gle": 1.0,
                "cosmic_gle_id": gle_num,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def _daily_flags(events: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Every morning in range. 1 if a GLE onset was in the last HOLD_DAYS (known next day)."""
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
        hit = None if match.empty else match.iloc[-1]
        flags.append(
            {
                "source": SOURCE,
                "observed_at": iso(day.to_pydatetime()),
                "available_at": iso((day + pd.Timedelta(days=1)).to_pydatetime()),
                "license": LICENSE,
                "cosmic_gle": 1.0 if hit is not None else 0.0,
                "cosmic_gle_id": None if hit is None else hit.get("cosmic_gle_id"),
            }
        )
    return pd.DataFrame(flags).sort_values("observed_at").reset_index(drop=True)


def pull_gle(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    text = fetch_text(GRID_URL, timeout=90)
    events = parse_gle_grid(text)
    if events.empty:
        return pd.DataFrame()
    out = _daily_flags(events, start, end)
    if out.empty:
        return out
    out["ingest_run_id"] = ingest_run_id
    return out
