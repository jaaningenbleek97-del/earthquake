from __future__ import annotations

from datetime import datetime

import pandas as pd


def utc_ts(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def parse_utc(values) -> pd.Series:
    """Parse ISO-8601 stamps that may or may not include fractional seconds."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)
    return pd.to_datetime(series, utc=True, format="mixed")


def merge_daily(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Outer-join two daily frames on observed_at.

    A merged row is only known once every source on it is public, so it keeps
    the later available_at. A source missing that day does not hide the row.
    """
    keep = ["observed_at", *[c for c in columns if c in right.columns]]
    if "available_at" in right.columns:
        keep.append("available_at")
    out = left.merge(right[keep], on="observed_at", how="outer", suffixes=("", "_right"))
    if "available_at_right" in out.columns:
        mine = parse_utc(out["available_at"]) if "available_at" in left.columns else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        theirs = parse_utc(out.pop("available_at_right"))
        out["available_at"] = mine.where(theirs.isna() | (mine >= theirs), theirs)
    return out
