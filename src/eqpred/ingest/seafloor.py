"""Public Hikurangi seafloor pressure, reduced to one daily number.

The 2019 absolute-pressure gauges and the 2019–2021 flow-meter loggers.
Both archives were published in 2022, after the measurements.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from eqpred.paths import raw_dir
from eqpred.records import iso

SOURCE = "hikurangi.press"
APG_PUBLIC = datetime(2022, 1, 11, tzinfo=timezone.utc)
RBR_PUBLIC = datetime(2022, 7, 31, tzinfo=timezone.utc)
MATLAB_UNIX = 719529


def _daily_median(wide: pd.DataFrame, column: str, public_at: datetime) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()
    daily = wide.median(axis=1, skipna=True).dropna()
    frame = daily.rename(column).reset_index()
    frame.columns = ["observed_at", column]
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["available_at"] = iso(public_at)
    frame["source"] = SOURCE
    return frame


def apg_daily(root: Path | None = None) -> pd.DataFrame:
    path = raw_dir(root) / SOURCE / "APG_hikurangi_2019.json"
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = []
    for site, item in payload.items():
        stamp = pd.to_datetime(item.get("date") or [], utc=True, errors="coerce")
        values = pd.to_numeric(pd.Series(item.get("pressure") or []), errors="coerce")
        if len(stamp) != len(values):
            continue
        part = pd.DataFrame({"observed_at": stamp, site: values}).dropna()
        if part.empty:
            continue
        part["observed_at"] = part["observed_at"].dt.floor("D")
        columns.append(part.groupby("observed_at")[site].mean())
    if not columns:
        return pd.DataFrame()
    return _daily_median(pd.concat(columns, axis=1), "pore_apg_hpa", APG_PUBLIC)


def rbr_daily(root: Path | None = None) -> pd.DataFrame:
    path = raw_dir(root) / SOURCE / "Data-20220730T223122Z-001.zip"
    if not path.exists():
        return pd.DataFrame()
    columns = []
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if "Detided" in name and name.endswith(".mat")]
        for name in names:
            mat = scipy.io.loadmat(io.BytesIO(archive.read(name)))
            if "time" not in mat or "Praw_detided" not in mat:
                continue
            stamp = pd.to_datetime(
                np.ravel(mat["time"]).astype(float) - MATLAB_UNIX,
                unit="D",
                origin="unix",
                utc=True,
            )
            values = pd.to_numeric(pd.Series(np.ravel(mat["Praw_detided"])), errors="coerce")
            part = pd.DataFrame({"observed_at": stamp, "value": values}).dropna()
            if part.empty:
                continue
            part["observed_at"] = part["observed_at"].dt.floor("D")
            columns.append(part.groupby("observed_at")["value"].mean().rename(Path(name).stem))
    if not columns:
        return pd.DataFrame()
    return _daily_median(pd.concat(columns, axis=1), "pore_rbr_dbar", RBR_PUBLIC)
