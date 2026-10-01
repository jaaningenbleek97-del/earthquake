"""Hikurangi borehole pressure from the public Carlson et al. 2026 files.

Sites U1518 and U1519, one-minute formation pressure reduced to a daily mean.
The zip on Zenodo is the release. A morning before 30 March 2026 cannot see it.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eqpred.paths import raw_dir
from eqpred.records import iso

SOURCE = "iodp.pore"
LICENSE = "https://doi.org/10.5281/zenodo.19340667"
# Zenodo record 19340667 publication date. Not the sample time.
PUBLIC_AT = datetime(2026, 3, 30, tzinfo=timezone.utc)


def parse_pressure_csv(text: str, column: str) -> pd.DataFrame:
    """One site CSV → daily mean formation pressure, released on PUBLIC_AT."""
    if not text or "formation_pressure" not in text:
        return pd.DataFrame()
    raw = pd.read_csv(io.StringIO(text))
    if raw.empty or "date" not in raw.columns or "formation_pressure" not in raw.columns:
        return pd.DataFrame()
    stamp = pd.to_datetime(raw["date"], format="%d-%b-%Y %H:%M:%S", utc=True, errors="coerce")
    work = pd.DataFrame(
        {
            "observed_at": stamp,
            column: pd.to_numeric(raw["formation_pressure"], errors="coerce"),
        }
    ).dropna()
    if work.empty:
        return pd.DataFrame()
    work["observed_at"] = work["observed_at"].dt.floor("D")
    daily = work.groupby("observed_at", as_index=False)[column].mean()
    daily["available_at"] = iso(PUBLIC_AT)
    daily["source"] = SOURCE
    daily["license"] = LICENSE
    return daily


def pore_daily(root: Path | None = None) -> pd.DataFrame:
    """Both sites, if the CSVs are on disk. Empty when they are not."""
    folder = raw_dir(root) / SOURCE
    frames = []
    for site, name in (("u1518", "U1518_raw.csv"), ("u1519", "U1519_raw.csv")):
        path = folder / name
        if not path.exists():
            continue
        part = parse_pressure_csv(path.read_text(encoding="utf-8", errors="replace"), f"pore_{site}_kpa")
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for part in frames[1:]:
        cols = [c for c in part.columns if c.startswith("pore_")]
        out = out.merge(part[["observed_at", *cols]], on="observed_at", how="outer")
    return out.sort_values("observed_at").reset_index(drop=True)
