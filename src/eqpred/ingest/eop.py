from __future__ import annotations

"""Earth orientation: polar motion and length of day.

IERS EOP C04. Replayable ASCII. Davidson / Burns talk about Earth wobble
and spin-rate changes as a leftover coupling path.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

URLS = (
    "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.62-now",
    "https://datacenter.iers.org/data/latestVersion/EOP_14_C04_IAU2000A_one_file_1962-now.txt",
)
LICENSE = "https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html"
LATENCY = timedelta(days=5)
SOURCE = "iers.eop"


def parse_eop_c04(text: str) -> pd.DataFrame:
    """Year month day MJD x(") y(") UT1-UTC(s) LOD(s) ..."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.lower().startswith("year"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            year = int(float(parts[0]))
            month = int(float(parts[1]))
            day = int(float(parts[2]))
            if year < 1900 or year > 2100:
                continue
            observed = datetime(year, month, day, tzinfo=timezone.utc)
            pmx = float(parts[4])
            pmy = float(parts[5])
            lod = float(parts[7])
        except (ValueError, OverflowError):
            continue
        if abs(pmx) > 2 or abs(pmy) > 2 or abs(lod) > 0.01:
            continue
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "eop_pmx_arcsec": pmx,
                "eop_pmy_arcsec": pmy,
                "eop_lod_s": lod,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def pull_eop(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    last_error: Exception | None = None
    frame = pd.DataFrame()
    for url in URLS:
        try:
            frame = parse_eop_c04(fetch_text(url, timeout=120))
        except Exception as exc:  # noqa: BLE001 — try the next archive
            last_error = exc
            continue
        if not frame.empty:
            break
    if frame.empty:
        raise RuntimeError(f"IERS EOP download failed: {last_error}")
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return frame.loc[mask].reset_index(drop=True)
