from __future__ import annotations

"""COSMIC-2 GNSS-RO scintillation (S4) from UCAR CDAAC.

Daily max / median S4 from the public rapid `scnLv2` tars (from 2022-03-25).
NRT level2 has atmospheric profiles only, not S4. Each day is cached so a
later pull does not re-download the 3–15 MB tars.

A day is one tar of hundreds of radio-occultation profiles (typically
~200–700). We keep every morning and, by default, every 4th profile in
that tar so the sample is spread through the day. Set
EQPRED_COSMIC_PROFILE_STRIDE=1 for every occultation.
"""

import io
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.paths import data_dir
from eqpred.records import iso
from eqpred.regions import REGIONS, point_in_region
from eqpred.stamp import utc_ts

SOURCE = "ucar.cosmic2"
LICENSE = "https://cdaac-www.cosmic.ucar.edu/"
LATENCY = timedelta(days=3)
RAPID = "https://data.cosmic.ucar.edu/gnss-ro/cosmic2/rapid/level2/{year}/{doy:03d}/scnLv2_rapid_{year}_{doy:03d}.tar.gz"
# Rapid scnLv2 starts 2022-03-25 (doy 84). NRT level2 only has atmPrf, not S4.
MISSION_START = datetime(2022, 3, 25)
# Threads + netCDF4 crashed on Windows (ACCESS_VIOLATION). Separate
# processes keep one Dataset per interpreter. Cap at 6 so UCAR is not flooded.
# HDF5 file locking deadlocks when several processes open netCDF in memory.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
MAX_WORKERS = 6
DEFAULT_DAY_STRIDE = 1
DEFAULT_PROFILE_STRIDE = 4
NZ_LAT = (-48.0, -25.0)
NZ_LON = (165.0, 185.0)
BOXES = ("hikurangi", "alpine", "kermadec")


def _env_stride(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, int(default))


def worker_count() -> int:
    """Default 1: UCAR + Windows netCDF stalled when six days ran at once.

    Set EQPRED_COSMIC_WORKERS=4 to retry a process pool.
    """
    raw = os.environ.get("EQPRED_COSMIC_WORKERS")
    if raw:
        try:
            return max(1, min(MAX_WORKERS, int(raw)))
        except ValueError:
            pass
    return 1


def day_stride() -> int:
    """Keep one morning in N. Default 1: sample every day. Cached days stay."""
    return _env_stride("EQPRED_COSMIC_DAY_STRIDE", DEFAULT_DAY_STRIDE)


def profile_stride() -> int:
    """Keep one occultation in N inside a day's tar."""
    return _env_stride("EQPRED_COSMIC_PROFILE_STRIDE", DEFAULT_PROFILE_STRIDE)


def on_day_grid(day: datetime, stride: int, origin: datetime | None = None) -> bool:
    start = (origin or MISSION_START).date()
    when = day.date() if hasattr(day, "date") else day
    return ((when - start).days % max(int(stride), 1)) == 0


def thin_pending(pending: list[datetime], stride: int | None = None) -> list[datetime]:
    step = day_stride() if stride is None else max(int(stride), 1)
    if step <= 1:
        return list(pending)
    return [day for day in pending if on_day_grid(day, step)]


def _cache_dir() -> Path:
    path = data_dir() / "cache" / "cosmic2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(day: datetime) -> Path:
    return _cache_dir() / f"{day:%Y_%j}.parquet"


def _in_nz(lat: float, lon: float) -> bool:
    lon_w = lon if lon >= 0 else lon + 360.0
    return NZ_LAT[0] <= lat <= NZ_LAT[1] and NZ_LON[0] <= lon_w <= NZ_LON[1]


def _box_for(lat: float, lon: float) -> str | None:
    for name in BOXES:
        if point_in_region(lat, lon, REGIONS[name]):
            return name
    return None


def parse_scn_profile(blob: bytes) -> dict[str, float] | None:
    """One scnLv2 netCDF: max L1 S4 and the profile's mean lat/lon."""
    if not blob or blob[:3] != b"CDF":
        return None
    try:
        import netCDF4
    except ImportError:
        return None
    try:
        ds = netCDF4.Dataset("mem", memory=blob)
    except Exception:
        return None
    try:
        if "s4_L1" not in ds.variables:
            return None
        s4 = np.asarray(ds.variables["s4_L1"][...], dtype=float)
        lat = np.asarray(ds.variables["lat"][...], dtype=float) if "lat" in ds.variables else np.array([])
        lon = np.asarray(ds.variables["lon"][...], dtype=float) if "lon" in ds.variables else np.array([])
    finally:
        ds.close()
    s4 = s4[np.isfinite(s4)]
    if s4.size == 0:
        return None
    lat = lat[np.isfinite(lat)]
    lon = lon[np.isfinite(lon)]
    return {
        "s4": float(np.nanmax(s4)),
        "lat": float(np.nanmean(lat)) if lat.size else float("nan"),
        "lon": float(np.nanmean(lon)) if lon.size else float("nan"),
    }


def parse_scn_tar(blob: bytes, stride: int | None = None) -> pd.DataFrame:
    """Profiles in one daily scnLv2 tar.gz. Default: every 4th occultation."""
    if not blob:
        return pd.DataFrame()
    step = max(int(stride if stride is not None else profile_stride()), 1)
    rows: list[dict[str, float]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            seen = 0
            for member in tar.getmembers():
                if not member.isfile() or member.size < 64:
                    continue
                keep = seen % step == 0
                seen += 1
                if not keep:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                parsed = parse_scn_profile(handle.read())
                if parsed is not None:
                    rows.append(parsed)
    except (tarfile.TarError, OSError, ValueError):
        return pd.DataFrame()
    return pd.DataFrame(rows)


def profiles_to_daily(profiles: pd.DataFrame, day: datetime) -> pd.DataFrame:
    """One morning: global and NZ S4 from that day's profiles."""
    if profiles.empty or "s4" not in profiles.columns:
        return pd.DataFrame()
    work = profiles.dropna(subset=["s4"]).copy()
    if work.empty:
        return pd.DataFrame()
    nz_mask = []
    box_hits = {name: [] for name in BOXES}
    for rec in work.itertuples(index=False):
        lat = float(rec.lat) if rec.lat == rec.lat else None
        lon = float(rec.lon) if rec.lon == rec.lon else None
        nz_mask.append(bool(lat is not None and lon is not None and _in_nz(lat, lon)))
        if lat is not None and lon is not None:
            box = _box_for(lat, lon)
            if box:
                box_hits[box].append(float(rec.s4))
    work["_nz"] = nz_mask
    nz = work.loc[work["_nz"], "s4"]
    when = utc_ts(day).replace(hour=0, minute=0, second=0, microsecond=0)
    row = {
        "source": SOURCE,
        "observed_at": iso(when),
        "available_at": iso(when + LATENCY),
        "license": LICENSE,
        "scint_s4": float(work["s4"].median()),
        "scint_s4_max": float(work["s4"].max()),
        "scint_s4_nz": float(nz.median()) if len(nz) else None,
        "scint_n": float(len(work)),
        "scint_n_nz": float(len(nz)),
        "scint_hikurangi": float(np.median(box_hits["hikurangi"])) if box_hits["hikurangi"] else None,
        "scint_alpine": float(np.median(box_hits["alpine"])) if box_hits["alpine"] else None,
        "scint_kermadec": float(np.median(box_hits["kermadec"])) if box_hits["kermadec"] else None,
    }
    return pd.DataFrame([row])


def _cached_day(day: datetime) -> pd.DataFrame:
    path = _cache_path(day)
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _store_day(day: datetime, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame.to_parquet(_cache_path(day), index=False)


def _download_tar(day: datetime) -> bytes:
    year = day.year
    doy = int(day.strftime("%j"))
    url = RAPID.format(year=year, doy=doy)
    last: Exception | None = None
    for attempt in range(3):
        try:
            return fetch_bytes(url, timeout=45)
        except Exception as exc:  # noqa: BLE001 — truncated tars are common
            last = exc
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    assert last is not None
    raise last


def _aware(when: datetime) -> datetime:
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when


def _pull_one_day(when: datetime) -> pd.DataFrame:
    when = _aware(when)
    cached = _cached_day(when)
    if not cached.empty:
        return cached
    print(f"  ucar.cosmic2 {when.date()}", flush=True)
    blob = _download_tar(when)
    daily = profiles_to_daily(parse_scn_tar(blob), when)
    if daily.empty:
        return pd.DataFrame()
    _store_day(when, daily)
    return daily


def _worker_day(iso_day: str) -> str:
    """Process-pool entry. Writes the daily cache. Returns ok / empty / skip."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    when = datetime.fromisoformat(iso_day).replace(tzinfo=timezone.utc)
    try:
        daily = _pull_one_day(when)
    except Exception as exc:  # noqa: BLE001
        return f"skip: {exc}"
    if daily.empty:
        return "empty"
    return "ok"


def _frontier_first(pending: list[datetime], cached: list[datetime]) -> list[datetime]:
    """Resume the first long gap from mission start. Short holes (bad tars) go last."""
    if not pending:
        return []
    pending = sorted(pending)
    have = {day.date() for day in cached}
    start = MISSION_START.date()
    end = max(have | {day.date() for day in pending}) if (have or pending) else start
    cursor = start
    resume = pending[0].date()
    while cursor <= end:
        if cursor in have:
            cursor += timedelta(days=1)
            continue
        hole_end = cursor
        while hole_end <= end and hole_end not in have:
            hole_end += timedelta(days=1)
        if (hole_end - cursor).days > 5:
            resume = cursor
            break
        cursor = hole_end
    frontier = [day for day in pending if day.date() >= resume]
    holes = [day for day in pending if day.date() < resume]
    return frontier + holes


def _prefetch_pull(pending: list[datetime], frames: list[pd.DataFrame]) -> None:
    """Parse one day at a time; download the next tar on a background thread."""
    from concurrent.futures import ThreadPoolExecutor

    n = len(pending)
    with ThreadPoolExecutor(max_workers=1) as pool:
        next_blob = pool.submit(_download_tar, pending[0]) if pending else None
        for i, when in enumerate(pending):
            try:
                blob = next_blob.result() if next_blob is not None else _download_tar(when)
            except Exception as exc:  # noqa: BLE001
                print(f"    {when.date()} skip: {exc}", flush=True)
                next_blob = pool.submit(_download_tar, pending[i + 1]) if i + 1 < n else None
                continue
            if i + 1 < n:
                next_blob = pool.submit(_download_tar, pending[i + 1])
            else:
                next_blob = None
            print(f"  ucar.cosmic2 {when.date()} ({i + 1}/{n})", flush=True)
            daily = profiles_to_daily(parse_scn_tar(blob), when)
            if daily.empty:
                print(f"    {when.date()} empty", flush=True)
                continue
            _store_day(when, daily)
            frames.append(daily)


def pull_cosmic2(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily COSMIC-2 S4. Cached days are skipped; missing days prefetch the next tar."""
    begin = max(utc_ts(start), MISSION_START.replace(tzinfo=utc_ts(start).tzinfo))
    days = pd.date_range(begin, utc_ts(end), freq="D", inclusive="left")
    frames: list[pd.DataFrame] = []
    pending: list[datetime] = []
    cached_days: list[datetime] = []
    for day in days:
        when = _aware(day.to_pydatetime())
        cached = _cached_day(when)
        if not cached.empty:
            frames.append(cached)
            cached_days.append(when)
            continue
        pending.append(when)
    step = day_stride()
    pending = thin_pending(pending, step)
    pending = _frontier_first(pending, cached_days)
    workers = worker_count() if len(pending) > 1 else 1
    print(
        f"  ucar.cosmic2 {len(frames)} cached, {len(pending)} to fetch "
        f"× {workers} (every {step} days, every {profile_stride()} profiles)",
        flush=True,
    )
    if pending and workers == 1:
        _prefetch_pull(pending, frames)
    elif pending:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_worker_day, when.date().isoformat()): when for when in pending}
            for fut in as_completed(futs):
                when = futs[fut]
                done += 1
                try:
                    status = fut.result()
                except Exception as exc:  # noqa: BLE001
                    status = f"skip: {exc}"
                if status == "ok":
                    daily = _cached_day(when)
                    if not daily.empty:
                        frames.append(daily)
                elif status == "empty":
                    print(f"    {when.date()} empty", flush=True)
                else:
                    print(f"    {when.date()} {status}", flush=True)
                if done % 20 == 0:
                    print(f"  ucar.cosmic2 fetch {done}/{len(pending)}", flush=True)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ingest_run_id"] = ingest_run_id
    return out.drop_duplicates(subset=["observed_at"], keep="last").sort_values("observed_at")
