from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.cusp_geom import cusp_lon_deg
from eqpred.ingest.http import fetch_text
from eqpred.ingest.omni import FILLS, _as_float
from eqpred.records import iso

# Hourly OMNI2. Daily averages cannot time a cusp — the dayside sweeps the globe in 24h.
OMNI2_YEAR_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
LICENSE = "https://omniweb.gsfc.nasa.gov/html/ow_data.html"
SOURCE = "nasa.omni_storm"
# Hourly Dst is faster than daily OMNI averages. One day keeps onset off the card until it would have been posted.
LATENCY = timedelta(days=1)
DST_STORM = -50.0
DST_QUIET = -30.0
QUIET_HOURS = 12


def parse_omni_hourly_dst(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 41:
            continue
        try:
            year = int(parts[0])
            doy = int(parts[1])
            hour = int(float(parts[2]))
        except ValueError:
            continue
        dst = _as_float(parts[40], FILLS["dst_nt"])
        observed = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, hours=hour)
        rows.append({"observed_at": observed, "dst_nt": dst})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("observed_at").reset_index(drop=True)


def storm_onsets(hourly: pd.DataFrame) -> pd.DataFrame:
    """First Dst ≤ −50 after ≥12 quiet hours (Dst > −30). One onset per storm."""
    if hourly.empty:
        return pd.DataFrame()
    work = hourly.dropna(subset=["dst_nt"]).sort_values("observed_at")
    quiet_run = 0
    armed = False
    rows = []
    for rec in work.itertuples(index=False):
        dst = float(rec.dst_nt)
        if dst > DST_QUIET:
            quiet_run += 1
            if quiet_run >= QUIET_HOURS:
                armed = True
        else:
            quiet_run = 0
        if armed and dst <= DST_STORM:
            observed = rec.observed_at
            if getattr(observed, "tzinfo", None) is None:
                observed = observed.replace(tzinfo=timezone.utc)
            ut = observed.hour + observed.minute / 60.0
            rows.append(
                {
                    "observed_at": observed,
                    "dst_nt": dst,
                    "ut_hour": ut,
                    "cusp_lon_deg": float(cusp_lon_deg(ut)),
                }
            )
            armed = False
    return pd.DataFrame(rows)


def pull_storm_onsets(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Hourly Dst storm onsets + locked 12-MLT cusp longitude. Unused on global p."""
    frames = []
    years = range(start.year - 1, end.year + 1)
    for year in years:
        url = OMNI2_YEAR_URL.format(year=year)
        print(f"  OMNI2 hourly {year}", flush=True)
        try:
            text = fetch_text(url, timeout=180)
        except Exception as exc:  # noqa: BLE001 — one dead year must not kill the overlay
            print(f"    skip {year}: {exc}", flush=True)
            continue
        hourly = parse_omni_hourly_dst(text)
        if hourly.empty:
            continue
        frames.append(hourly)
    if not frames:
        return pd.DataFrame()
    hourly = pd.concat(frames, ignore_index=True)
    hourly = hourly.loc[(hourly["observed_at"] >= start - timedelta(days=2)) & (hourly["observed_at"] < end)]
    onsets = storm_onsets(hourly)
    if onsets.empty:
        return pd.DataFrame()
    out = []
    for rec in onsets.to_dict(orient="records"):
        observed = rec["observed_at"]
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        if observed < start or observed >= end:
            continue
        out.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "ingest_run_id": ingest_run_id,
                "dst_nt": float(rec["dst_nt"]),
                "ut_hour": float(rec["ut_hour"]),
                "cusp_lon_deg": float(rec["cusp_lon_deg"]),
            }
        )
    return pd.DataFrame(out)
