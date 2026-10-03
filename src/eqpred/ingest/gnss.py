from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso

# Nevada Geodetic Lab daily tenv3. NZ + transfer arcs. Skip 404s.
NGL_URLS = (
    "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{site}.tenv3",
    "https://geodesy.unr.edu/gps_timeseries/tenv3/IGS20/{site}.tenv3",
    "http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{site}.tenv3",
)
NZ_SITES = ("AUCK", "WGTN", "LYTT", "DUND", "CHTI", "RAUL")
JAPAN_SITES = ("TSKB", "USUD", "MIZU")
CHILE_SITES = ("ANTC", "SANT", "IQQE")
SITES = NZ_SITES + JAPAN_SITES + CHILE_SITES
SLOW_SLIP_SITES = ("WGTN", "RAUL")
HOLDINGS_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
MIDAS_URL = "https://geodesy.unr.edu/velocities/midas.IGS14.txt"
SOURCE_VELOCITY = "ngl.midas"
LICENSE = "http://geodesy.unr.edu/"
SOURCE = "ngl.gnss"
# Solutions are reprocessed; do not treat same-day files as live.
LATENCY = timedelta(days=14)


def _from_decimal_year(value: float) -> datetime:
    year = int(value)
    origin = datetime(year, 1, 1, tzinfo=timezone.utc)
    return origin + timedelta(days=(value - year) * 365.25)


def _parse_tenv3(text: str, site: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 13 or parts[0].lower() == "site":
            continue
        try:
            observed = _from_decimal_year(float(parts[2]))
            east = float(parts[8])
            north = float(parts[10])
            vert = float(parts[12])
        except (IndexError, ValueError):
            continue
        rows.append({"site": site, "observed_at": observed, "east_m": east, "north_m": north, "vert_m": vert})
    return pd.DataFrame(rows)


def _in_nz(lat: float, lon: float) -> bool:
    """New Zealand, Chatham Islands, and the Kermadec ridge. Not the whole Pacific."""
    if lon > 180:
        lon -= 360.0
    return (-52.0 <= lat <= -29.0) and ((165.0 <= lon <= 180.0) or (-180.0 <= lon <= -175.0))


def ngl_station_coords() -> dict[str, tuple[float, float]]:
    """site -> (lon, lat). Longitude is wrapped to -180..180."""
    try:
        text = fetch_text(HOLDINGS_URL, timeout=90)
    except Exception as exc:  # noqa: BLE001
        print(f"  NGL station list skipped: {exc}", flush=True)
        return {}
    found: dict[str, tuple[float, float]] = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 3 or len(parts[0]) != 4:
            continue
        try:
            lat = float(parts[1])
            lon = float(parts[2])
        except ValueError:
            continue
        if lon > 180.0:
            lon -= 360.0
        found[parts[0].upper()] = (lon, lat)
    return found


def nz_ngl_sites() -> tuple[str, ...]:
    """Every 4-character Nevada station in the NZ region. Falls back to the short list."""
    coords = ngl_station_coords()
    found = [site for site, (lon, lat) in coords.items() if _in_nz(lat, lon)]
    if not found:
        return NZ_SITES
    return tuple(dict.fromkeys([*found, *NZ_SITES]))


def _pull_site(site: str) -> pd.DataFrame:
    text = None
    last_exc: Exception | None = None
    for template in NGL_URLS:
        url = template.format(site=site)
        try:
            text = fetch_text(url, timeout=180)
            break
        except Exception as exc:  # noqa: BLE001 — try the next scheme
            last_exc = exc
            continue
    if text is None:
        print(f"    skip {site}: {last_exc}", flush=True)
        return pd.DataFrame()
    frame = _parse_tenv3(text, site)
    print(f"    {site} {len(frame)} days", flush=True)
    return frame


def pull_gnss(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily GNSS positions. Every NZ station, plus the short Japan and Chile lists."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sites = tuple(dict.fromkeys([*nz_ngl_sites(), *JAPAN_SITES, *CHILE_SITES]))
    print(f"  NGL {len(sites)} stations", flush=True)
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_pull_site, site): site for site in sites}
        for fut in as_completed(futures):
            frame = fut.result()
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[(raw["observed_at"] >= start) & (raw["observed_at"] < end)]
    if raw.empty:
        return pd.DataFrame()
    daily = (
        raw.assign(day=raw["observed_at"].dt.floor("D"))
        .groupby(["site", "day"], as_index=False)
        .agg(east_m=("east_m", "mean"), north_m=("north_m", "mean"), vert_m=("vert_m", "mean"))
    )
    out = []
    for rec in daily.to_dict(orient="records"):
        observed = rec["day"].to_pydatetime()
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        out.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "ingest_run_id": ingest_run_id,
                "site": rec["site"],
                "east_mm": float(rec["east_m"]) * 1000.0,
                "north_mm": float(rec["north_m"]) * 1000.0,
                "vert_mm": float(rec["vert_m"]) * 1000.0,
            }
        )
    return pd.DataFrame(out)


def parse_midas(text: str) -> pd.DataFrame:
    """Nevada MIDAS velocities. East, north, and up are millimetres per year."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 11 or len(parts[0]) != 4 or parts[0].lower() == "site":
            continue
        try:
            end = _from_decimal_year(float(parts[3]))
            rows.append(
                {
                    "site": parts[0],
                    "observed_at": end,
                    "span_years": float(parts[4]),
                    "east_mm_yr": float(parts[8]) * 1000.0,
                    "north_mm_yr": float(parts[9]) * 1000.0,
                    "up_mm_yr": float(parts[10]) * 1000.0,
                }
            )
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def pull_midas(ingest_run_id: str, sites: tuple[str, ...] | None = None) -> pd.DataFrame:
    """One speed per station. Kept for the NZ set plus the short Japan and Chile lists."""
    text = fetch_text(MIDAS_URL, timeout=90)
    frame = parse_midas(text)
    if frame.empty:
        return frame
    keep = {code.upper() for code in (sites or (*nz_ngl_sites(), *JAPAN_SITES, *CHILE_SITES))}
    frame = frame.loc[frame["site"].str.upper().isin(keep)].copy()
    if frame.empty:
        return frame
    frame["source"] = SOURCE_VELOCITY
    frame["observed_at"] = frame["observed_at"].map(lambda stamp: iso(stamp))
    frame["available_at"] = frame["observed_at"].map(
        lambda stamp: iso(datetime.fromisoformat(stamp.replace("Z", "+00:00")) + LATENCY)
    )
    frame["license"] = LICENSE
    frame["ingest_run_id"] = ingest_run_id
    print(f"  {SOURCE_VELOCITY} {len(frame)} stations", flush=True)
    return frame.reset_index(drop=True)
