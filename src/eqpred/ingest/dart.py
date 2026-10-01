from __future__ import annotations

"""Ocean-bottom pressure / water column from NOAA DART and GeoNet Tilde.

Replayable daily height. Used as a Hikurangi / Kermadec loading overlay,
not a tsunami alarm.
"""

import gzip
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_bytes, fetch_json, fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "noaa.dart"
LICENSE = "https://www.ndbc.noaa.gov/"
GEONET_LICENSE = "https://www.geonet.org.nz/data/types/dart"
LATENCY = timedelta(hours=12)
NDBC_HIST = "https://www.ndbc.noaa.gov/data/historical/dart/{station}t{year}.txt.gz"
NDBC_HIST_ALT = "https://www.ndbc.noaa.gov/data/historical/dart/{station}d{year}.txt.gz"
NDBC_REALTIME = "https://www.ndbc.noaa.gov/data/realtime2/{station}.dart"
TILDE_TYPES = "https://tilde.geonet.org.nz/v4/data-types/dart"
TILDE_DATA = (
    "https://tilde.geonet.org.nz/v4/data/dart/"
    "{site}/water-height/nil/{aspect}/{start}/{end}"
)
# South Pacific / Kermadec / Tonga buoys with long public histories.
# Chile / Japan / SW Pacific buoys that still have yearly `tYYYY` archives.
NDBC_STATIONS = ("32401", "32402", "21413", "55012", "55023")
# Hikurangi / Kermadec-facing GeoNet buoys. The array starts ~2019.
GEONET_SITES = ("NZC", "NZG", "NZJ", "NZK")
GEONET_START = datetime(2019, 1, 1, tzinfo=timezone.utc)


def parse_tilde_height(payload: object) -> list[tuple[datetime, float]]:
    """Tilde DART water-height. Keep native metres; do not reuse the GNSS mm scale."""
    rows: list[tuple[datetime, float]] = []
    blocks = payload if isinstance(payload, list) else [payload]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for rec in block.get("data") or []:
            ts = rec.get("ts")
            val = rec.get("val")
            if ts is None or val is None:
                continue
            try:
                observed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                rows.append((observed, float(val)))
            except (TypeError, ValueError):
                continue
    return rows


def parse_ndbc_dart(text: str, station: str) -> pd.DataFrame:
    """NDBC historical or realtime DART: year month day hour minute second T height_m."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("YY"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        try:
            year = int(parts[0])
            if year < 100:
                year += 2000
            observed = datetime(
                year,
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                int(float(parts[5])),
                tzinfo=timezone.utc,
            )
            height = float(parts[7])
        except (TypeError, ValueError, IndexError):
            continue
        if not (1000.0 < height < 8000.0):
            continue
        rows.append(
            {
                "observed_at": iso(observed),
                "dart_station": station,
                "dart_height_m": height,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def daily_dart_features(frame: pd.DataFrame) -> pd.DataFrame:
    """One morning per station: mean height, range, and day-to-day drop."""
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["obs_day"] = pd.to_datetime(work["observed_at"], utc=True).dt.floor("D")
    grouped = (
        work.groupby(["obs_day", "dart_station"], as_index=False)
        .agg(
            dart_height_m=("dart_height_m", "mean"),
            dart_range_m=("dart_height_m", lambda s: float(s.max() - s.min()) if len(s) else None),
        )
        .sort_values(["dart_station", "obs_day"])
    )
    grouped["dart_d1_m"] = grouped.groupby("dart_station")["dart_height_m"].diff(1)
    rows: list[dict[str, object]] = []
    for rec in grouped.itertuples(index=False):
        day = rec.obs_day.to_pydatetime()
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(day),
                "available_at": iso(day + LATENCY),
                "license": LICENSE,
                "dart_station": rec.dart_station,
                "dart_height_m": float(rec.dart_height_m),
                "dart_range_m": None if rec.dart_range_m is None else float(rec.dart_range_m),
                "dart_d1_m": None if pd.isna(rec.dart_d1_m) else float(rec.dart_d1_m),
            }
        )
    return pd.DataFrame(rows)


def _fetch_ndbc_year(station: str, year: int) -> str:
    last_exc: Exception | None = None
    for template in (NDBC_HIST, NDBC_HIST_ALT):
        url = template.format(station=station, year=year)
        try:
            raw = fetch_bytes(url, timeout=40)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        if raw[:2] == b"\x1f\x8b":
            return gzip.decompress(raw).decode("utf-8", errors="replace")
        return raw.decode("utf-8", errors="replace")
    raise last_exc or RuntimeError(f"no DART file for {station} {year}")


def _pull_ndbc(start: datetime, end: datetime) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    years = range(utc_ts(start).year, utc_ts(end).year + 1)
    for station in NDBC_STATIONS:
        print(f"  noaa.dart {station} {years.start}-{years.stop - 1}", flush=True)
        for year in years:
            try:
                text = _fetch_ndbc_year(station, year)
            except Exception as exc:  # noqa: BLE001 — one missing year is normal
                print(f"  noaa.dart skip {station} {year}: {exc}", flush=True)
                continue
            part = parse_ndbc_dart(text, station)
            if not part.empty:
                frames.append(part)
        try:
            live = parse_ndbc_dart(fetch_text(NDBC_REALTIME.format(station=station), timeout=40), station)
        except Exception as exc:  # noqa: BLE001
            print(f"  noaa.dart live skip {station}: {exc}")
            live = pd.DataFrame()
        if not live.empty:
            frames.append(live)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pull_geonet(start: datetime, end: datetime) -> pd.DataFrame:
    """Daily-ish GeoNet DART water height if Tilde answers."""
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    begin = max(start, GEONET_START)
    stop = min(end, now)
    if begin >= stop:
        return pd.DataFrame()
    sites = list(GEONET_SITES)
    try:
        types = fetch_json(TILDE_TYPES, timeout=40)
        names = []
        blocks = types if isinstance(types, list) else [types]
        for block in blocks:
            if not isinstance(block, dict):
                continue
            for rec in block.get("station") or block.get("stations") or []:
                code = str(rec.get("id") or rec.get("station") or rec or "").upper()
                if code.startswith("NZ"):
                    names.append(code)
        if names:
            sites = sorted(set(names))
    except Exception as exc:  # noqa: BLE001 — fall back to the known NZ list
        print(f"  geonet.dart types skipped: {exc}")
    rows: list[dict[str, object]] = []
    for site in sites:
        parsed: list[tuple[datetime, float]] = []
        cursor = begin
        while cursor < stop:
            nxt = min(cursor + timedelta(days=30), stop)
            url = TILDE_DATA.format(
                site=site,
                aspect="15m",
                start=cursor.strftime("%Y-%m-%d"),
                end=nxt.strftime("%Y-%m-%d"),
            )
            try:
                payload = fetch_json(url, timeout=90)
                parsed.extend(parse_tilde_height(payload))
            except Exception as exc:  # noqa: BLE001
                if "HTTP 400" not in str(exc) and "HTTP 404" not in str(exc):
                    print(f"  geonet.dart skip {site}: {exc}")
            cursor = nxt
        for observed, value in parsed:
            if not (1000.0 < value < 8000.0) and not (1.0 < value < 20.0):
                # Tilde may return metres of water column or a residual.
                if value <= 0:
                    continue
            rows.append(
                {
                    "observed_at": iso(observed),
                    "dart_station": site,
                    "dart_height_m": float(value),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def pull_dart(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frames = [_pull_ndbc(start, end)]
    # GeoNet Tilde DART is short and chatty; try it only for the recent tail.
    recent = max(start, datetime.now(timezone.utc) - timedelta(days=400))
    if recent < end:
        frames.append(_pull_geonet(recent, end))
    raw = [f for f in frames if not f.empty]
    if not raw:
        return pd.DataFrame()
    daily = daily_dart_features(pd.concat(raw, ignore_index=True))
    if daily.empty:
        return daily
    stamp = pd.to_datetime(daily["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    out = daily.loc[mask].copy()
    out["ingest_run_id"] = ingest_run_id
    return out.reset_index(drop=True)
