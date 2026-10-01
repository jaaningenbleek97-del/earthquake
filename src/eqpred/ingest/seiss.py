from __future__ import annotations

"""GOES-R SEISS L2 daily particles from NCEI.

Fills the 2020–2026 hole left when SWPC Daily Particle Data stopped
(2020 Q1) and live JSON only keeps seven days. SGPS → integral protons.
MPS-HI → radiation-belt electrons.

NCEI SGPS 5-minute averages begin 2020-11-01. MPS-HI covers 2017 onward.
"""

import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from eqpred.ingest.http import fetch_bytes, fetch_text
from eqpred.paths import data_dir
from eqpred.records import iso

LICENSE = "https://www.ncei.noaa.gov/products/goes-r-space-environment-in-situ"
BASE = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
    "goes/goes{sat}/l2/data/{product}/{year}/{month:02d}/"
)
FILE_RE = re.compile(
    r"(sci_(?:sgps|mpsh)-l2-avg5m_g\d+_d(\d{8})_v[\d-]+\.nc)",
    re.IGNORECASE,
)
FILL = -1.0e30
SGPS_START = datetime(2020, 11, 1, tzinfo=timezone.utc)
MPSH_START = datetime(2017, 1, 8, tzinfo=timezone.utc)
G16_EAST_END = date(2025, 4, 7)
# G16 SGPS L2 is empty on these mornings; G18 published the same days.
G16_SGPS_HOLE = (date(2024, 4, 16), date(2024, 7, 22))
# Documented SGPS differential bounds (MeV) when the file omits them.
SGPS_BOUNDS_MEV = (
    (1.0, 1.9),
    (1.9, 2.3),
    (2.3, 3.4),
    (3.4, 6.5),
    (6.5, 12.0),
    (12.0, 25.0),
    (25.0, 40.0),
    (40.0, 80.0),
    (83.0, 99.0),
    (99.0, 118.0),
    (118.0, 150.0),
    (150.0, 275.0),
    (275.0, 500.0),
)
PROTON_THRESHOLDS = (
    (1.0, "goes_proton_1mev"),
    (5.0, "goes_proton_5mev"),
    (10.0, "goes_proton_10mev"),
    (30.0, "goes_proton_30mev"),
    (50.0, "goes_proton_50mev"),
    (60.0, "goes_proton_60mev"),
    (100.0, "goes_proton_100mev"),
    (500.0, "goes_proton_500mev"),
)
SGPS_PRODUCTS = ("sgps-l2-avg5m", "sgps-l2-avg5m_science")
MPSH_PRODUCTS = ("mpsh-l2-avg5m", "mpsh-l2-avg5m_science")


def _cache_dir() -> Path:
    path = data_dir() / "cache" / "seiss"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _day_stamp(day: datetime) -> str:
    return iso(day.replace(hour=0, minute=0, second=0, microsecond=0))


def _clean(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(out) & (out > FILL) & (out >= 0.0), out, np.nan)


def _first_var(ds: object, names: Iterable[str]) -> np.ndarray | None:
    for name in names:
        if name in ds.variables:
            return np.asarray(ds.variables[name][...])
    return None


def _as_mev(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo_c = _clean(lo)
    hi_c = _clean(hi)
    sample = lo_c[np.isfinite(lo_c)]
    if sample.size and float(np.nanmedian(sample)) > 50.0:
        return lo_c / 1000.0, hi_c / 1000.0
    return lo_c, hi_c


def _flatten_energy(arr: np.ndarray) -> np.ndarray:
    work = np.asarray(arr, dtype=float)
    if work.ndim > 1:
        work = np.nanmean(work, axis=tuple(range(work.ndim - 1)))
    return work.reshape(-1)


def _channel_bounds(lo: np.ndarray | None, hi: np.ndarray | None, nchan: int) -> tuple[np.ndarray, np.ndarray]:
    if lo is None or hi is None or lo.size == 0 or hi.size == 0:
        default = np.asarray(SGPS_BOUNDS_MEV[:nchan], dtype=float)
        if default.shape[0] < nchan:
            extra = np.array([[500.0 + i, 500.0 + i + 1] for i in range(nchan - default.shape[0])], dtype=float)
            default = np.vstack([default, extra])
        return default[:, 0], default[:, 1]
    lo_m, hi_m = _as_mev(_flatten_energy(lo), _flatten_energy(hi))
    lo_m = np.asarray(lo_m, dtype=float).reshape(-1)[:nchan]
    hi_m = np.asarray(hi_m, dtype=float).reshape(-1)[:nchan]
    if lo_m.size < nchan:
        default = np.asarray(SGPS_BOUNDS_MEV[:nchan], dtype=float)
        lo_m = default[:, 0]
        hi_m = default[:, 1]
    return lo_m, hi_m


def bounds_from_effective_kev(eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neighbor midpoints around MPS-HI effective energies (keV) → MeV edges."""
    mid = _clean(_flatten_energy(eff))
    mid = mid[np.isfinite(mid)]
    if mid.size < 2:
        return np.array([]), np.array([])
    edges = np.zeros(mid.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (mid[:-1] + mid[1:])
    edges[0] = max(mid[0] - (edges[1] - mid[0]), 0.0)
    edges[-1] = mid[-1] + (mid[-1] - edges[-2])
    return edges[:-1] / 1000.0, edges[1:] / 1000.0


def integral_above(
    diff: np.ndarray,
    lo_mev: np.ndarray,
    hi_mev: np.ndarray,
    threshold_mev: float,
    p11: np.ndarray | None = None,
) -> np.ndarray:
    """Differential flux (cm^-2 sr^-1 keV^-1 s^-1) → integral pfu above threshold."""
    work = _clean(diff)
    if work.ndim == 2:
        work = work[:, None, :]
    nchan = work.shape[-1]
    pfu = np.zeros(work.shape[:2], dtype=float)
    for i in range(nchan):
        lo = float(lo_mev[i]) if i < len(lo_mev) and np.isfinite(lo_mev[i]) else 0.0
        hi = float(hi_mev[i]) if i < len(hi_mev) and np.isfinite(hi_mev[i]) else lo
        if hi <= threshold_mev:
            continue
        overlap = hi - max(lo, threshold_mev)
        if overlap <= 0:
            continue
        pfu += work[..., i] * (overlap * 1000.0)
    if p11 is not None and threshold_mev <= 500.0:
        extra = _clean(p11)
        if extra.ndim == 1:
            extra = extra[:, None]
        pfu = pfu + extra
    return pfu


def _daily_max(pfu: np.ndarray) -> float | None:
    finite = pfu[np.isfinite(pfu)]
    if finite.size == 0:
        return None
    return float(np.nanmax(finite))


def parse_sgps_nc(blob: bytes, day: datetime) -> pd.DataFrame:
    """One SGPS 5-minute file → one morning of integral proton flux."""
    if not blob:
        return pd.DataFrame()
    try:
        import netCDF4
    except ImportError:
        return pd.DataFrame()
    try:
        ds = netCDF4.Dataset("mem", memory=blob)
    except Exception:
        return pd.DataFrame()
    try:
        diff = _first_var(ds, ("AvgDiffProtonFlux", "DiffProtonFlux"))
        if diff is None:
            return pd.DataFrame()
        p11 = _first_var(ds, ("AvgIntProtonFlux", "IntProtonFlux"))
        lo = _first_var(ds, ("DiffProtonLowerEnergy", "proton_diff_lower_energy"))
        hi = _first_var(ds, ("DiffProtonUpperEnergy", "proton_diff_upper_energy"))
        nchan = int(np.asarray(diff).shape[-1])
        lo_m, hi_m = _channel_bounds(lo, hi, nchan)
        rec: dict[str, object] = {"observed_at": _day_stamp(day)}
        for thresh, col in PROTON_THRESHOLDS:
            value = _daily_max(integral_above(np.asarray(diff), lo_m, hi_m, thresh, p11))
            if value is not None:
                rec[col] = value
        if "goes_proton_10mev" in rec and float(rec["goes_proton_10mev"]) >= 10.0:
            rec["goes_spe"] = 1.0
        else:
            rec["goes_spe"] = 0.0
        return pd.DataFrame([rec]) if len(rec) > 2 else pd.DataFrame()
    finally:
        ds.close()


def parse_mpsh_nc(blob: bytes, day: datetime) -> pd.DataFrame:
    """One MPS-HI 5-minute file → one morning of electron flux."""
    if not blob:
        return pd.DataFrame()
    try:
        import netCDF4
    except ImportError:
        return pd.DataFrame()
    try:
        ds = netCDF4.Dataset("mem", memory=blob)
    except Exception:
        return pd.DataFrame()
    try:
        rec: dict[str, object] = {"observed_at": _day_stamp(day)}
        integral = _first_var(ds, ("AvgIntElectronFlux", "IntElectronFlux"))
        if integral is not None:
            value = _daily_max(_clean(np.asarray(integral)))
            if value is not None:
                rec["electron_2mev"] = value
        diff = _first_var(ds, ("AvgDiffElectronFlux", "DiffElectronFlux"))
        lo = _first_var(ds, ("DiffElectronLowerEnergy", "electron_diff_lower_energy"))
        hi = _first_var(ds, ("DiffElectronUpperEnergy", "electron_diff_upper_energy"))
        eff = _first_var(ds, ("DiffElectronEffectiveEnergy",))
        if diff is not None:
            work = np.asarray(diff)
            nchan = int(work.shape[-1])
            if lo is not None and hi is not None:
                lo_m, hi_m = _channel_bounds(lo, hi, nchan)
            elif eff is not None:
                lo_m, hi_m = bounds_from_effective_kev(eff)
            else:
                lo_m, hi_m = np.array([]), np.array([])
            if lo_m.size:
                e08 = _daily_max(integral_above(work, lo_m, hi_m, 0.8))
                e4 = _daily_max(integral_above(work, lo_m, hi_m, 4.0))
                if e08 is not None:
                    rec["electron_08mev"] = e08
                if e4 is not None:
                    rec["electron_4mev"] = e4
        return pd.DataFrame([rec]) if len(rec) > 1 else pd.DataFrame()
    finally:
        ds.close()


def sat_order(day: datetime, kind: str) -> tuple[int, ...]:
    when = day.date() if hasattr(day, "date") else day
    if when >= G16_EAST_END:
        return (19, 18, 16, 17)
    if kind == "sgps" and G16_SGPS_HOLE[0] <= when <= G16_SGPS_HOLE[1]:
        return (18, 16, 17, 19)
    if kind == "sgps" and when < SGPS_START.date():
        return (17, 16, 18, 19)
    return (16, 17, 18, 19)


def _version_key(name: str) -> tuple[int, ...]:
    match = re.search(r"_v([\d-]+)\.nc$", name, re.IGNORECASE)
    if not match:
        return (0,)
    parts = [int(p) for p in match.group(1).split("-") if p.isdigit()]
    return tuple(parts) if parts else (0,)


def _list_month(sat: int, product: str, year: int, month: int) -> list[str]:
    cache = _cache_dir() / f"list_g{sat}_{product}_{year}{month:02d}.txt"
    if cache.is_file():
        return [line.strip() for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    url = BASE.format(sat=sat, product=product, year=year, month=month)
    try:
        html = fetch_text(url, timeout=60)
    except Exception:
        return []
    names = sorted({m.group(1) for m in FILE_RE.finditer(html)}, key=_version_key)
    if names:
        cache.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def _file_for_day(sat: int, product: str, day: datetime) -> str | None:
    names = _list_month(sat, product, day.year, day.month)
    stamp = day.strftime("%Y%m%d")
    matches = [name for name in names if f"_d{stamp}_" in name]
    if not matches:
        return None
    return max(matches, key=_version_key)


def _cache_path(kind: str, day: datetime) -> Path:
    return _cache_dir() / f"{kind}_{day:%Y%m%d}.parquet"


def _cached_day(kind: str, day: datetime) -> pd.DataFrame:
    path = _cache_path(kind, day)
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _store_day(kind: str, day: datetime, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    frame.to_parquet(_cache_path(kind, day), index=False)


def _download(sat: int, product: str, name: str, day: datetime) -> bytes:
    url = BASE.format(sat=sat, product=product, year=day.year, month=day.month) + name
    return fetch_bytes(url, timeout=90)


def _pull_one_day(kind: str, day: datetime) -> pd.DataFrame:
    cached = _cached_day(kind, day)
    if not cached.empty:
        return cached
    products = SGPS_PRODUCTS if kind == "sgps" else MPSH_PRODUCTS
    parse = parse_sgps_nc if kind == "sgps" else parse_mpsh_nc
    for sat in sat_order(day, kind):
        for product in products:
            name = _file_for_day(sat, product, day)
            if not name:
                continue
            try:
                blob = _download(sat, product, name, day)
            except Exception:
                continue
            daily = parse(blob, day)
            if daily.empty:
                continue
            _store_day(kind, day, daily)
            return daily
    return pd.DataFrame()


def pull_seiss(
    kind: str,
    ingest_run_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Daily SGPS protons or MPS-HI electrons. Cached per day."""
    mission = SGPS_START if kind == "sgps" else MPSH_START
    begin = mission if start is None else max(start, mission)
    if begin.tzinfo is None:
        begin = begin.replace(tzinfo=timezone.utc)
    stop = end or datetime.now(timezone.utc)
    if stop.tzinfo is None:
        stop = stop.replace(tzinfo=timezone.utc)
    if begin >= stop:
        return pd.DataFrame()
    days = pd.date_range(begin, stop, freq="D", inclusive="left")
    frames: list[pd.DataFrame] = []
    got = 0
    missing = 0
    for day in days:
        when = day.to_pydatetime()
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        try:
            daily = _pull_one_day(kind, when)
        except Exception as exc:  # noqa: BLE001
            print(f"    seiss {kind} {when.date()} skip: {exc}", flush=True)
            missing += 1
            continue
        if daily.empty:
            missing += 1
            if when.day == 1 or missing <= 3:
                print(f"    seiss {kind} {when.date()} empty", flush=True)
            continue
        frames.append(daily)
        got += 1
        if when.day == 1 or got % 50 == 0:
            print(f"  seiss {kind} {when.date()} ({got} days)", flush=True)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ingest_run_id"] = ingest_run_id
    print(f"  seiss {kind} kept {got} days, empty {missing}", flush=True)
    return out.drop_duplicates(subset=["observed_at"], keep="last").sort_values("observed_at")
