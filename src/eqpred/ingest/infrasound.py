"""Chatham Island infrasound, station IS36, from the public BGR product.

The file is a yearly archive (Hupe et al. 2022), not a next-morning feed.
Years through 2020 were issued on 8 November 2021. Later years are in the
copy downloaded on 24 September 2026; an earlier update date was not stated.
A morning only sees the loudness of that same calendar day, and only once
the file that contains it was already public.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.paths import raw_dir
from eqpred.records import iso
from eqpred.stamp import merge_daily, parse_utc

SOURCE = "bgr.infrasound"
LICENSE = "https://doi.org/10.25928/bgrseis_mblf-ifsd"
FIRST_RELEASE = datetime(2021, 11, 8, tzinfo=timezone.utc)
LATER_RELEASE = datetime(2026, 9, 24, tzinfo=timezone.utc)
STATION = "IS36"


def release_time(year: int) -> datetime:
    if int(year) <= 2020:
        return FIRST_RELEASE
    return LATER_RELEASE


def same_day_public(frame: pd.DataFrame, issue: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Value of that calendar day, hidden until the file was public."""
    empty = pd.DataFrame({"issue_time": issue, **{c: np.nan for c in columns}})
    if frame is None or frame.empty:
        return empty
    work = frame.copy()
    work["day"] = parse_utc(work["observed_at"]).dt.floor("D")
    work["available_at"] = parse_utc(work["available_at"])
    work = work.drop_duplicates("day", keep="last")
    left = pd.DataFrame({"issue_time": parse_utc(issue)})
    left["day"] = left["issue_time"].dt.floor("D")
    merged = left.merge(work, on="day", how="left")
    hidden = merged["available_at"].isna() | (merged["available_at"] > merged["issue_time"])
    out = pd.DataFrame({"issue_time": left["issue_time"]})
    for col in columns:
        values = pd.to_numeric(merged[col], errors="coerce") if col in merged.columns else np.nan
        if not isinstance(values, pd.Series):
            values = pd.Series(np.nan, index=merged.index)
        values = values.mask(hidden)
        out[col] = values.to_numpy()
    return out


def _chars_to_time(raw: np.ndarray) -> pd.Series:
    """NetCDF char time, shape (15, n), text yyyymmddTHHMMSS."""
    letters = np.asarray(raw).T
    text = [bytes(row).decode("ascii", "replace").strip("\x00").strip() for row in letters]
    return pd.to_datetime(pd.Series(text), format="%Y%m%dT%H%M%S", utc=True, errors="coerce")


def parse_station_file(path: Path, station: str = "is36") -> pd.DataFrame:
    """One yearly station file → daily detection count and mean RMS amplitude."""
    station = station.lower()
    n_col = f"infra_{station}_n"
    rms_col = f"infra_{station}_rms_pa"
    import netCDF4 as nc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = nc.Dataset(path)
        try:
            clock = _chars_to_time(ds.variables["time"][:])
            count = np.asarray(ds.variables["num"][1, :], dtype=float)
            detect_time = _chars_to_time(ds.variables["time_p"][:])
            rms = np.asarray(ds.variables["a_rms"][0, :], dtype=float)
        finally:
            ds.close()
    counted = pd.DataFrame({"observed_at": clock, n_col: count}).dropna(subset=["observed_at"])
    counted["observed_at"] = counted["observed_at"].dt.floor("D")
    daily_n = counted.groupby("observed_at", as_index=False)[n_col].sum()
    heard = pd.DataFrame({"observed_at": detect_time, rms_col: rms}).dropna()
    heard["observed_at"] = heard["observed_at"].dt.floor("D")
    daily_rms = heard.groupby("observed_at", as_index=False)[rms_col].mean()
    daily = daily_n.merge(daily_rms, on="observed_at", how="left")
    if daily.empty:
        return daily
    daily["available_at"] = [iso(release_time(int(t.year))) for t in daily["observed_at"]]
    daily["source"] = SOURCE
    daily["license"] = LICENSE
    daily["station"] = station
    return daily


def infrasound_daily(root: Path | None = None) -> pd.DataFrame:
    """Every downloaded station, one row per day."""
    base = raw_dir(root) / SOURCE
    folders = [path for path in sorted(base.glob("is*")) if path.is_dir()] if base.exists() else []
    frames = []
    for folder in folders:
        files = sorted(folder.glob("IS*.nc"))
        if not files:
            continue
        cache = base / f"{folder.name}_daily.parquet"
        if cache.exists() and cache.stat().st_mtime >= max(path.stat().st_mtime for path in files):
            part = pd.read_parquet(cache)
        else:
            parts = [parse_station_file(path, folder.name) for path in files]
            parts = [frame for frame in parts if frame is not None and not frame.empty]
            if not parts:
                continue
            part = (
                pd.concat(parts, ignore_index=True)
                .sort_values("observed_at")
                .drop_duplicates("observed_at", keep="last")
                .reset_index(drop=True)
            )
            part.to_parquet(cache, index=False)
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for part in frames[1:]:
        cols = [c for c in part.columns if str(c).startswith("infra_")]
        out = merge_daily(out, part, cols)
    return out.sort_values("observed_at").reset_index(drop=True)
