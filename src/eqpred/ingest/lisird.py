from __future__ import annotations

"""Solar irradiance series from LASP LISIRD.

TSI, Mg II, and Lyman-alpha — solar / EUV conditions Davidson and Burns
watch, not another OMNI Kp transform.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

LICENSE = "https://lasp.colorado.edu/lisird/"
BREMEN_LICENSE = "https://www.iup.uni-bremen.de/gome/gomemgii.html"
LATENCY = timedelta(days=4)
SOURCE = "lasp.lisird"

# NRL2 TSI ends 2023-12-31. TSIS is the live daily TSI. Bremen Mg II is live;
# the LISIRD Mg II composite stopped in 2013. Lyman-alpha uses days since 1947.
TSI_URLS = (
    "https://lasp.colorado.edu/lisird/latis/dap/nrl2_tsi_P1D.csv",
    "https://lasp.colorado.edu/lisird/latis/dap/tsis_tsi_24hr.csv",
)
MGII_URL = "https://www.iup.uni-bremen.de/gome/solar/MgII_composite.dat"
LYA_URL = "https://lasp.colorado.edu/lisird/latis/dap/composite_lyman_alpha.csv"
JD_UNIX = 2440587.5


def _lisird_stamp(token: str) -> datetime:
    raw = token.replace("Z", "+00:00")
    if "T" in raw or (len(raw) >= 10 and raw[4] == "-"):
        stamp = datetime.fromisoformat(raw[:19] if "T" in raw else raw[:10])
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    val = float(token)
    if val >= 1_800_000:
        observed = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=val - JD_UNIX)
    elif val >= 80_000:
        observed = datetime(1610, 1, 1, tzinfo=timezone.utc) + timedelta(days=val)
    elif val >= 10_000:
        observed = datetime(1947, 1, 1, tzinfo=timezone.utc) + timedelta(days=val)
    else:
        raise ValueError(f"unrecognised LISIRD time token {token}")
    return observed.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_lisird_csv(text: str, column: str) -> pd.DataFrame:
    """LISIRD latis CSV: time plus one or more value columns."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        if parts[0].lower().startswith(("time", "date")) or not parts[0][:1].isdigit():
            continue
        try:
            observed = _lisird_stamp(parts[0])
            value = float(parts[1])
        except (ValueError, OverflowError):
            continue
        if not (0.0 < value < 2000.0):
            continue
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                column: value,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def parse_bremen_mgii(text: str) -> pd.DataFrame:
    """Bremen GOME/S5P Mg II composite: year.frac month day index ..."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 4:
            continue
        try:
            year = int(float(parts[0]))
            month = int(parts[1])
            day = int(parts[2])
            value = float(parts[3])
            observed = datetime(year, month, day, tzinfo=timezone.utc)
        except (ValueError, OverflowError):
            continue
        if not (0.05 < value < 0.5):
            continue
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": BREMEN_LICENSE,
                "solar_mgii": value,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def _fetch_column(column: str, urls: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for url in urls:
        try:
            text = fetch_text(url, timeout=120)
        except Exception as exc:  # noqa: BLE001 — keep the other series
            print(f"  lasp.lisird skip {column} {url}: {exc}")
            continue
        frame = parse_lisird_csv(text, column)
        if frame.empty:
            print(f"  lasp.lisird empty {column} {url}")
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("observed_at", keep="first").sort_values("observed_at").reset_index(drop=True)


def pull_lisird(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    tsi = _fetch_column("solar_tsi_wm2", TSI_URLS)
    if not tsi.empty:
        frames.append(tsi)
    try:
        mgii = parse_bremen_mgii(fetch_text(MGII_URL, timeout=120))
    except Exception as exc:  # noqa: BLE001
        print(f"  lasp.lisird skip solar_mgii: {exc}")
        mgii = pd.DataFrame()
    if not mgii.empty:
        frames.append(mgii)
    lya = _fetch_column("solar_lya", (LYA_URL,))
    if not lya.empty:
        frames.append(lya)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        cols = [c for c in extra.columns if str(c).startswith("solar_")]
        out = out.merge(extra[["observed_at", *cols]], on="observed_at", how="outer")
    out["source"] = SOURCE
    out["license"] = LICENSE
    out["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(out["observed_at"], utc=True)
    out["available_at"] = (stamp + LATENCY).map(iso)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return out.loc[mask].sort_values("observed_at").reset_index(drop=True)
