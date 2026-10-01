from __future__ import annotations

"""Earth orientation: polar motion and length of day.

IERS EOP C04. Replayable ASCII. Davidson / Burns talk about Earth wobble
and spin-rate changes as a coupling path.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

# 14 C04 is the bulletin series (ends weeks behind). finals2000A is the
# rapid Bulletin A file and is current. 20 C04 lags further. Paris 62-now 404s.
C04_URLS = (
    "https://datacenter.iers.org/data/latestVersion/EOP_14_C04_IAU2000A_one_file_1962-now.txt",
    "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.62-now",
)
FINALS_URL = "https://datacenter.iers.org/data/9/finals2000A.all"
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


def parse_eop_finals(text: str) -> pd.DataFrame:
    """IERS finals2000A.all — fixed-width Bulletin A (I = observed, P = predicted)."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        if len(raw) < 26:
            continue
        try:
            year = int(raw[0:2])
            month = int(raw[2:4])
            day = int(raw[4:6])
            mjd = float(raw[7:15])
        except ValueError:
            continue
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        year += 2000 if mjd >= 51544 else 1900
        body = raw[15:]
        flag_at = next((i for i, ch in enumerate(body) if ch in "IP"), -1)
        if flag_at < 0:
            continue
        nums = body[flag_at + 1 :].replace("P", " ").replace("I", " ").split()
        if len(nums) < 3:
            continue
        try:
            pmx = float(nums[0])
            pmy = float(nums[2])
        except ValueError:
            continue
        if abs(pmx) > 2 or abs(pmy) > 2:
            continue
        lod = float("nan")
        if len(nums) >= 7:
            try:
                cand = float(nums[6])
            except ValueError:
                cand = float("nan")
            if abs(cand) >= 0.01 and abs(cand) < 10:
                cand = cand / 1000.0
            if abs(cand) < 0.01:
                lod = cand
        try:
            observed = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
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
    frames: list[pd.DataFrame] = []
    for url in C04_URLS:
        try:
            frame = parse_eop_c04(fetch_text(url, timeout=120))
        except Exception as exc:  # noqa: BLE001 — try the next archive
            last_error = exc
            continue
        if not frame.empty:
            frames.append(frame)
            break
    try:
        finals = parse_eop_finals(fetch_text(FINALS_URL, timeout=120))
        if not finals.empty:
            frames.append(finals)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
    if not frames:
        raise RuntimeError(f"IERS EOP download failed: {last_error}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates("observed_at", keep="first")
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return frame.loc[mask].sort_values("observed_at").reset_index(drop=True)
