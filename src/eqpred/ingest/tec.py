from __future__ import annotations

import gzip
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import earthdata_credentials, fetch_bytes
from eqpred.records import iso
from eqpred.regions import REGIONS, point_in_region


def _ionex_lon(lon: float) -> float:
    """IONEX grids are sometimes 0–360. Keep 180 as 180, not 0."""
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon

# Official IGS archive needs Earthdata + a one-time CDDIS authorize click.
# Public JPL GIM (same IONEX grid) is what actually answers from this lab.
JPL_FINAL = "https://sideshow.jpl.nasa.gov/pub/iono_daily/IONEX_final"
CDDIS_URLS = (
    "https://cddis.nasa.gov/archive/gnss/products/ionex/{year}/{doy:03d}/igsg{doy:03d}0.{yy:02d}i.Z",
    "https://cddis.nasa.gov/archive/gnss/products/ionex/{year}/{doy:03d}/igsg{doy:03d}0.{yy:02d}i.gz",
    "https://cddis.nasa.gov/archive/gnss/products/ionex/{year}/{doy:03d}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX.gz",
)
IONEX_URLS = (
    "https://igs.ign.fr/pub/igs/products/ionex/{year}/{doy:03d}/igsg{doy:03d}0.{yy:02d}i.Z",
    "https://igs.bkg.bund.de/root_ftp/IGS/products/ionex/{year}/{doy:03d}/igsg{doy:03d}0.{yy:02d}i.Z",
    "https://igs.gnsswhu.cn/pub/gps/products/ionex/{year}/{doy:03d}/igsg{doy:03d}0.{yy:02d}i.Z",
)
LICENSE = "https://sideshow.jpl.nasa.gov/pub/iono_daily/README.txt"
SOURCE = "igs.tec"
LATENCY = timedelta(days=2)
PUBLIC_TIMEOUT = 45
FETCH_TIMEOUT = 8
CDDIS_TIMEOUT = 25
ABORT_AFTER_MISSES = 5
_CDDIS_DISABLED = False


def _map_date(year: int, doy: int) -> datetime:
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def ionex_candidate_urls(year: int, doy: int, yy: int) -> list[str]:
    """JPL public first, UPC Chapman next, dead IGS mirrors last."""
    yymmdd = _map_date(year, doy).strftime("%y%m%d")
    jpl_short = f"{JPL_FINAL}/y{year}/JPLG{doy:03d}0.{yy:02d}I.gz"
    jpl_long = (
        f"{JPL_FINAL}/y{year}/JPL0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX.gz"
    )
    chapman = (
        f"https://chapman.upc.es/tomion/rapid/{year}/{doy:03d}_{yymmdd}.15min/"
        f"uqrg{doy:03d}0.{yy:02d}i.Z"
    )
    public = [jpl_long, jpl_short, chapman] if year >= 2023 else [jpl_short, jpl_long, chapman]
    igs = [template.format(year=year, doy=doy, yy=yy) for template in IONEX_URLS]
    return public + igs


def _cddis_unusable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "login html",
            "authorize cddis",
            "connection was reset",
            "recv failure",
            "curl: (56)",
        )
    )


TEC_BOXES = ("hikurangi", "alpine", "kermadec", "japan", "chile")


def _in_nz(lat: float, lon: float) -> bool:
    if lat < -48 or lat > -25:
        return False
    return lon >= 165 or lon <= -175


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def parse_ionex(text: str) -> dict[str, float | None]:
    """Global, NZ, and per-box mean TECU from one IONEX map."""
    exponent = 0.0
    values: list[float] = []
    nz_values: list[float] = []
    box_values: dict[str, list[float]] = {rid: [] for rid in TEC_BOXES}
    lat = None
    lon = None
    dlon = 5.0
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if "END OF TEC MAP" in line or "START OF TEC MAP" in line:
            lat = None
            lon = None
            continue
        if "EXPONENT" in line and "LAT" not in line:
            try:
                exponent = float(line.split()[0])
            except (IndexError, ValueError):
                exponent = 0.0
            continue
        if "LAT/LON1/LON2/DLON/H" in line:
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
            try:
                lat = float(nums[0])
                lon = float(nums[1]) if len(nums) > 1 else -180.0
                dlon = float(nums[3]) if len(nums) > 3 else 5.0
            except (IndexError, ValueError):
                lat = None
                lon = None
            continue
        if lat is None or lon is None:
            continue
        parts = line.split()
        nums = []
        for part in parts:
            try:
                nums.append(float(part) * (10.0**exponent))
            except ValueError:
                nums = []
                break
        if not nums:
            continue
        for tec in nums:
            values.append(tec)
            grid_lon = _ionex_lon(lon)
            if _in_nz(lat, grid_lon):
                nz_values.append(tec)
            for rid in TEC_BOXES:
                if point_in_region(lat, grid_lon, REGIONS[rid]):
                    box_values[rid].append(tec)
            lon += dlon
    if not values:
        return {"tec_tecu": None, "tec_nz_tecu": None}
    out: dict[str, float | None] = {
        "tec_tecu": _mean(values),
        "tec_nz_tecu": _mean(nz_values),
    }
    for rid in TEC_BOXES:
        out[f"tec_{rid}_tecu"] = _mean(box_values[rid])
    return out


def _decompress(data: bytes) -> str:
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data).decode("ascii", errors="replace")
    if data[:2] == b"\x1f\x9d":
        import unlzw3

        return unlzw3.unlzw(data).decode("ascii", errors="replace")
    try:
        import zlib

        return zlib.decompress(data, -15).decode("ascii", errors="replace")
    except zlib.error:
        return data.decode("ascii", errors="replace")


def _fetch_ionex(year: int, doy: int, yy: int, preferred: int = 0) -> tuple[str, int]:
    global _CDDIS_DISABLED
    last_err: Exception | None = None
    urls = ionex_candidate_urls(year, doy, yy)
    if preferred:
        urls = urls[preferred:] + urls[:preferred]
    for idx, url in enumerate(urls):
        timeout = PUBLIC_TIMEOUT if ("sideshow.jpl.nasa.gov" in url or "chapman.upc.es" in url) else FETCH_TIMEOUT
        try:
            raw = fetch_bytes(url, timeout=timeout)
            return _decompress(raw), idx
        except (RuntimeError, OSError, ValueError) as exc:
            last_err = exc
            continue
    if not _CDDIS_DISABLED and earthdata_credentials() is not None:
        for template in CDDIS_URLS:
            url = template.format(year=year, doy=doy, yy=yy)
            try:
                raw = fetch_bytes(url, timeout=CDDIS_TIMEOUT, earthdata=True)
                return _decompress(raw), -1
            except (RuntimeError, OSError, ValueError) as exc:
                last_err = exc
                if _cddis_unusable(exc):
                    _CDDIS_DISABLED = True
                    break
    raise RuntimeError(str(last_err) if last_err else "no IONEX mirror")


def _tec_one(cursor: datetime, ingest_run_id: str) -> dict | None:
    doy = int(cursor.strftime("%j"))
    yy = cursor.year % 100
    print(f"  TEC {cursor.date()}", flush=True)
    text, _idx = _fetch_ionex(cursor.year, doy, yy)
    means = parse_ionex(text)
    if means.get("tec_tecu") is None:
        return None
    observed = datetime(cursor.year, cursor.month, cursor.day, tzinfo=timezone.utc)
    return {
        "source": SOURCE,
        "observed_at": iso(observed),
        "available_at": iso(observed + LATENCY),
        "license": LICENSE,
        "ingest_run_id": ingest_run_id,
        **means,
    }


def pull_tec(
    start: datetime,
    end: datetime,
    ingest_run_id: str,
    step_days: int = 7,
    workers: int = 4,
) -> pd.DataFrame:
    """Weekly GIM TEC (JPL public, IGS fallback). Unused on issued p."""
    days: list[datetime] = []
    cursor = start
    while cursor < end:
        days.append(cursor)
        cursor += timedelta(days=step_days)
    rows: list[dict] = []
    misses = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_tec_one, day, ingest_run_id): day for day in days}
        for fut in as_completed(futures):
            day = futures[fut]
            try:
                row = fut.result()
            except (RuntimeError, OSError, ValueError) as exc:
                print(f"    skip {day.date()}: {exc}", flush=True)
                misses += 1
                if misses >= ABORT_AFTER_MISSES and not rows:
                    print("  TEC: no mirror answered the first weeks — stop", flush=True)
                    break
                continue
            if row is None:
                continue
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("observed_at").reset_index(drop=True)
