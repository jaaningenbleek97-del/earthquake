from __future__ import annotations

"""CNEOS fireball onsets — the public space-dust / meteoroid list."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_json
from eqpred.records import iso
from eqpred.stamp import utc_ts

URL = "https://ssd-api.jpl.nasa.gov/fireball.api"
LICENSE = "https://cneos.jpl.nasa.gov/fireballs/"
LATENCY = timedelta(days=1)
SOURCE = "cneos.fireball"
HOLD_DAYS = 2


def parse_fireball_api(pack: object) -> pd.DataFrame:
    if not isinstance(pack, dict):
        return pd.DataFrame()
    fields = [str(f) for f in (pack.get("fields") or [])]
    data = pack.get("data") or []
    if not fields or not data:
        return pd.DataFrame()
    idx = {name: i for i, name in enumerate(fields)}
    date_i = idx.get("date")
    energy_i = idx.get("energy")
    if date_i is None:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) <= date_i:
            continue
        raw = str(item[date_i] or "").strip()
        if len(raw) < 10:
            continue
        try:
            onset = datetime.fromisoformat(raw[:10]).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        energy = None
        if energy_i is not None and energy_i < len(item) and item[energy_i] not in (None, ""):
            try:
                energy = float(item[energy_i])
            except (TypeError, ValueError):
                energy = None
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(onset),
                "available_at": iso(onset + LATENCY),
                "license": LICENSE,
                "dust_fireball": 1.0,
                "dust_fireball_kt": energy,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


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
        hit = None if match.empty else match.iloc[-1]
        flags.append(
            {
                "source": SOURCE,
                "observed_at": iso(day.to_pydatetime()),
                "available_at": iso((day + pd.Timedelta(days=1)).to_pydatetime()),
                "license": LICENSE,
                "dust_fireball": 1.0 if hit is not None else 0.0,
                "dust_fireball_kt": None if hit is None else hit.get("dust_fireball_kt"),
            }
        )
    return pd.DataFrame(flags).sort_values("observed_at").reset_index(drop=True)


def pull_fireball(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    events = parse_fireball_api(fetch_json(URL, timeout=90))
    if events.empty:
        return pd.DataFrame()
    out = _daily_flags(events, start, end)
    if out.empty:
        return out
    out["ingest_run_id"] = ingest_run_id
    return out
