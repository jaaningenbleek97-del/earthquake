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
