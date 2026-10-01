from __future__ import annotations

import io
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "earthscope.strain"
LICENSE = "https://www.unavco.org/community/policies_forms/attribution/attribution.html"
L2 = "https://web-services.unavco.org/strain/data/L2"
# L2 is 5-minute processed. A few days late is typical.
LATENCY = timedelta(days=7)
CHUNK_DAYS = 31
# Long-running PBO / NOTA borehole strainmeters (Cascadia + California).
# These are overseas overlays, not NZ instruments.
STATIONS = ("B004", "B012", "B073", "B078")
FILL_ABS = 900000.0
MICRO_TO_NANO = 1000.0


def _l2_url(station: str, start: datetime, end: datetime) -> str:
    params = {
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
        "format": "csv",
    }
    return f"{L2}/{station}/beta?{urlencode(params)}"


def parse_strain_geocsv(text: str, station: str) -> pd.DataFrame:
    """Parse UNAVCO Strain L2 GeoCSV into 5-minute Eee nanostrain."""
    if not text or not text.strip():
        return pd.DataFrame()
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()
    frame = pd.read_csv(io.StringIO("\n".join(lines)))
    if frame.empty:
        return pd.DataFrame()
    cols = {str(c).strip(): c for c in frame.columns}
    time_col = cols.get("DateTime")
    areal = cols.get("Eee+Enn Corrected")
    diff = cols.get("Eee-Enn Corrected")
    if time_col is None or areal is None or diff is None:
        return pd.DataFrame()
    areal_v = pd.to_numeric(frame[areal], errors="coerce")
    diff_v = pd.to_numeric(frame[diff], errors="coerce")
    bad = areal_v.abs().ge(FILL_ABS) | diff_v.abs().ge(FILL_ABS)
    eee = 0.5 * (areal_v + diff_v)
    out = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(frame[time_col], utc=True, errors="coerce"),
            "strain_ee_nstrain": eee * MICRO_TO_NANO,
        }
    )
    out.loc[bad, "strain_ee_nstrain"] = pd.NA
    out = out.dropna(subset=["observed_at", "strain_ee_nstrain"]).reset_index(drop=True)
    if out.empty:
        return out
    out["strain_station"] = station
    return out


def daily_strain_features(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty or "strain_ee_nstrain" not in samples.columns:
        return pd.DataFrame()
    work = samples.dropna(subset=["strain_ee_nstrain"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["observed_at"] = pd.to_datetime(work["observed_at"], utc=True)
    work = work.sort_values("observed_at")
    work["day"] = work["observed_at"].dt.floor("D")
    work["de"] = work.groupby("day", sort=False)["strain_ee_nstrain"].diff().abs()
    grouped = work.groupby("day", sort=True)
    out = pd.DataFrame(
        {
            "observed_at": grouped["strain_ee_nstrain"].mean().index,
            "strain_ee_nstrain": grouped["strain_ee_nstrain"].mean().to_numpy(),
            "strain_ee_range_nstrain": (
                grouped["strain_ee_nstrain"].max() - grouped["strain_ee_nstrain"].min()
            ).to_numpy(),
            "strain_ee_dbdt_nstrain": grouped["de"].max().to_numpy(),
        }
    )
    station = ""
    if "strain_station" in work.columns and work["strain_station"].notna().any():
        station = str(work["strain_station"].iloc[0])
        out["strain_station"] = station
    out["source"] = SOURCE
    out["license"] = LICENSE
    out["available_at"] = [iso(ts.to_pydatetime() + LATENCY) for ts in out["observed_at"]]
    out["observed_at"] = [iso(ts.to_pydatetime()) for ts in out["observed_at"]]
    return out.reset_index(drop=True)


def _fetch_station(station: str, start: datetime, end: datetime) -> pd.DataFrame:
    text = fetch_text(_l2_url(station, start, end), timeout=180)
    return daily_strain_features(parse_strain_geocsv(text, station))


def pull_strain(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily PBO borehole Eee from the public UNAVCO Strain L2 web service. No login."""
    _ = ingest_run_id
    cursor = utc_ts(start).to_pydatetime()
    stop = utc_ts(end).to_pydatetime()
    frames: list[pd.DataFrame] = []
    while cursor < stop:
        nxt = min(cursor + timedelta(days=CHUNK_DAYS), stop)
        print(f"  earthscope.strain {cursor.date()} -> {nxt.date()}", flush=True)
        for station in STATIONS:
            try:
                part = _fetch_station(station, cursor, nxt)
            except Exception as exc:  # noqa: BLE001 — optional overlay
                print(f"  earthscope.strain skip {station} {cursor.date()}: {exc}", flush=True)
                continue
            if part.empty:
                print(f"  earthscope.strain skip {station} {cursor.date()}: no rows", flush=True)
                continue
            print(f"  earthscope.strain {station} {cursor.date()} {len(part)} days", flush=True)
            frames.append(part)
            time.sleep(0.2)
        cursor = nxt
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
