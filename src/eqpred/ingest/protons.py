from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd

from eqpred.ingest.http import fetch_json, fetch_text
from eqpred.records import iso

SOURCE = "noaa.goes_protons"
LICENSE = "https://www.weather.gov/disclaimer"
LATENCY = timedelta(hours=12)

# Operational JSON is short-horizon. Long replay is NCEI Daily Particle Data
# (fluence, 1994–2020 Q1) plus the NOAA SPE list for later storm days.
PROTON_URLS = (
    "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json",
    "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json",
    "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json",
)
FLUENCE_URLS = (
    "https://services.swpc.noaa.gov/json/goes/primary/integral-proton-fluence-7-day.json",
)
DPD_BASE = (
    "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/"
    "annual_reports/daily_solar_indices_summaries/daily_particle_data/"
)
DPD_FILES = (
    "2016_DPD.txt",
    "2017_DPD.txt",
    "2018_DPD.txt",
    "2019Q1_DPD.txt",
    "2019Q2_DPD.txt",
    "2019Q3_DPD.txt",
    "2019Q4_DPD.txt",
    "2020Q1_DPD.txt",
)
SPE_URLS = (
    "https://services.swpc.noaa.gov/text/indices/SPE.txt",
    "https://www.swpc.noaa.gov/ftpdir/indices/SPE.txt",
    "https://sohoftp.nascom.nasa.gov/sdb/goes/other/SPE.txt",
    "https://umbra.nascom.nasa.gov/SEP/seps.html",
)

SECONDS_PER_DAY = 86400.0
DPD_MISSING = {-1.0e5, -1.0e05, -999.99, -1.0}

# Map SWPC energy labels onto stable columns.
ENERGY_COLS = {
    ">=1 mev": "goes_proton_1mev",
    ">=5 mev": "goes_proton_5mev",
    ">=10 mev": "goes_proton_10mev",
    ">=30 mev": "goes_proton_30mev",
    ">=50 mev": "goes_proton_50mev",
    ">=60 mev": "goes_proton_60mev",
    ">=100 mev": "goes_proton_100mev",
    ">=500 mev": "goes_proton_500mev",
}
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_stamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _energy_col(label: object) -> str | None:
    if label is None:
        return None
    compact = str(label).strip().lower().replace(" ", "")
    if not compact.startswith(">="):
        compact = ">=" + compact
    lookup = {k.replace(" ", ""): v for k, v in ENERGY_COLS.items()}
    return lookup.get(compact)


def _day_stamp(observed: datetime) -> str:
    return iso(observed.replace(hour=0, minute=0, second=0, microsecond=0))


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number in DPD_MISSING or number < 0:
        return None
    return number


def parse_goes_protons(rows: list[Any]) -> pd.DataFrame:
    """Daily max integral proton flux (pfu) per energy band."""
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        observed = _parse_stamp(rec.get("time_tag") or rec.get("time"))
        if observed is None:
            continue
        col = _energy_col(rec.get("energy") or rec.get("energy_channel"))
        flux = rec.get("flux")
        try:
            flux_f = float(flux) if flux is not None else None
        except (TypeError, ValueError):
            flux_f = None
        if col is None or flux_f is None:
            continue
        out.append({"observed_at": _day_stamp(observed), col: flux_f})
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out)
    cols = [c for c in ENERGY_COLS.values() if c in frame.columns]
    return frame.groupby("observed_at", as_index=False)[cols].max()


def parse_goes_fluence(rows: list[Any]) -> pd.DataFrame:
    """Daily fluence (protons/cm2-sr-day) → mean flux (pfu)."""
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        observed = _parse_stamp(rec.get("time_tag") or rec.get("time"))
        if observed is None:
            continue
        col = _energy_col(rec.get("energy") or rec.get("energy_channel"))
        fluence = _finite(rec.get("fluence"))
        if col is None or fluence is None:
            continue
        out.append({"observed_at": _day_stamp(observed), col: fluence / SECONDS_PER_DAY})
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out)
    cols = [c for c in ENERGY_COLS.values() if c in frame.columns]
    return frame.groupby("observed_at", as_index=False)[cols].max()


def parse_dpd_text(text: str) -> pd.DataFrame:
    """NCEI/SWPC Daily Particle Data. Fluence columns → mean pfu."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":") or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 6:
            continue
        try:
            observed = datetime(int(parts[0]), int(parts[1]), int(parts[2]), tzinfo=timezone.utc)
        except ValueError:
            continue
        rec: dict[str, object] = {"observed_at": _day_stamp(observed)}
        for raw, col in (
            (parts[3], "goes_proton_1mev"),
            (parts[4], "goes_proton_10mev"),
            (parts[5], "goes_proton_100mev"),
        ):
            value = _finite(raw)
            if value is None:
                continue
            rec[col] = value / SECONDS_PER_DAY
        if len(rec) > 1:
            rows.append(rec)
    return pd.DataFrame(rows)


def _parse_spe_day(token: str, year: int) -> datetime | None:
    text = token.strip().replace("|", " ")
    parts = text.replace("/", " ").split()
    if len(parts) < 2:
        return None
    month = _MONTHS.get(parts[0][:3].lower())
    try:
        day = int(parts[1])
    except ValueError:
        return None
    if month is None:
        return None
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_spe_text(text: str) -> pd.DataFrame:
    """NOAA SPE list: peak >10 MeV flux on the event start day."""
    year: int | None = None
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip().replace("|", " ")
        if not line or line.startswith(":") or line.startswith("#") or line.startswith("<"):
            continue
        tokens = [t for t in line.split() if t and t != "-"]
        if len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) == 4:
            year = int(tokens[0])
            continue
        if year is None or len(tokens) < 3:
            continue
        start = _parse_spe_day(f"{tokens[0]} {tokens[1]}" if "/" not in tokens[0] else tokens[0], year)
        if start is None and len(tokens) >= 2:
            start = _parse_spe_day(tokens[0], year)
        flux = None
        for token in tokens:
            compact = token.replace(",", "")
            try:
                number = float(compact)
            except ValueError:
                continue
            if number >= 10:
                flux = number
                break
        if start is None or flux is None:
            continue
        rows.append({"observed_at": _day_stamp(start), "goes_proton_10mev": flux, "goes_spe": 1.0})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("observed_at", as_index=False).max()


def combine_proton_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Later frames overwrite only the columns they actually carry."""
    by_day: dict[str, dict[str, object]] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for rec in frame.to_dict(orient="records"):
            stamp = str(rec.get("observed_at") or "")
            if not stamp:
                continue
            day = stamp[:10]
            slot = by_day.setdefault(day, {"observed_at": f"{day}T00:00:00Z"})
            for key, value in rec.items():
                if key == "observed_at" or value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass
                slot[key] = value
    if not by_day:
        return pd.DataFrame()
    return pd.DataFrame(list(by_day.values())).sort_values("observed_at").reset_index(drop=True)


def _safe_json(url: str) -> list[Any]:
    try:
        payload = fetch_json(url, timeout=60)
    except Exception:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _safe_text(url: str) -> str:
    try:
        return fetch_text(url, timeout=90)
    except Exception:
        return ""


def _in_window(frame: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    if frame.empty or "observed_at" not in frame.columns:
        return frame
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    keep = pd.Series(True, index=frame.index)
    if start is not None:
        keep &= stamp >= pd.Timestamp(start)
    if end is not None:
        keep &= stamp < pd.Timestamp(end)
    return frame.loc[keep].reset_index(drop=True)


def pull_protons(
    ingest_run_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """GOES integral protons: NCEI DPD archive + SPE storms + SWPC JSON tail."""
    print("  NCEI daily particle data", flush=True)
    dpd_parts: list[pd.DataFrame] = []
    for name in DPD_FILES:
        year = int(name[:4])
        if start is not None and year < start.year:
            continue
        if end is not None and year > end.year:
            continue
        text = _safe_text(DPD_BASE + name)
        parsed = parse_dpd_text(text)
        if parsed.empty:
            print(f"    skip {name}", flush=True)
            continue
        print(f"    {name} {len(parsed)} days", flush=True)
        dpd_parts.append(parsed)
    dpd = pd.concat(dpd_parts, ignore_index=True) if dpd_parts else pd.DataFrame()

    spe = pd.DataFrame()
    for url in SPE_URLS:
        spe = parse_spe_text(_safe_text(url))
        if not spe.empty:
            print(f"  SPE list {len(spe)} events from {url}", flush=True)
            break
    if spe.empty:
        print("  SPE list skipped", flush=True)

    rows: list[Any] = []
    seen: set[str] = set()
    for url in PROTON_URLS:
        for rec in _safe_json(url):
            if not isinstance(rec, dict):
                continue
            key = f"{rec.get('time_tag')}|{rec.get('energy')}|{rec.get('satellite')}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec)
    live_flux = parse_goes_protons(rows)
    live_fluence = pd.DataFrame()
    fluence_rows: list[Any] = []
    for url in FLUENCE_URLS:
        fluence_rows.extend(_safe_json(url))
    live_fluence = parse_goes_fluence(fluence_rows)

    # Archive first; live JSON overwrites the last few days. SPE fills storm
    # days after DPD ended (2020 Q1) when no fluence file exists.
    daily = combine_proton_frames((dpd, spe, live_fluence, live_flux))
    daily = _in_window(daily, start, end)
    if daily.empty:
        return pd.DataFrame()
    daily["source"] = SOURCE
    daily["ingest_run_id"] = ingest_run_id
    daily["license"] = LICENSE
    stamp = pd.to_datetime(daily["observed_at"], utc=True)
    daily["available_at"] = [iso(t.to_pydatetime() + LATENCY) for t in stamp]
    if "goes_spe" not in daily.columns:
        daily["goes_spe"] = 0.0
    else:
        daily["goes_spe"] = pd.to_numeric(daily["goes_spe"], errors="coerce").fillna(0.0)
    return daily.sort_values("observed_at").reset_index(drop=True)
