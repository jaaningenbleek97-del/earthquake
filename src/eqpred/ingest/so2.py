from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, is_zipfile

import pandas as pd

from eqpred.records import iso

SOURCE = "cams.so2"
LICENSE = "https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4"
LATENCY = timedelta(days=5)
# CAMS EAC4 monthly / daily total column SO2 over Kermadec (no dateline wrap).
DATASET = "cams-global-reanalysis-eac4"
# North / West / South / East. Raoul / trench west of the date line.
AREA = [-25.5, 177.0, -37.0, 180.0]
VARIABLES = ("total_column_sulphur_dioxide",)


def _cds_client():
    """CAMS lives on the Atmosphere Data Store, not the Climate Data Store."""
    import os

    import cdsapi
    import certifi
    import truststore

    truststore.inject_into_ssl()
    bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    return cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api")


# EAC4 is a delayed reanalysis. Asking through "today" returns 400.
ADS_LAG = timedelta(days=80)


def eac4_end(end: datetime, now: datetime | None = None) -> datetime:
    """Last day ADS will actually serve for EAC4."""
    current = now or datetime.now(timezone.utc)
    cap = current.replace(hour=0, minute=0, second=0, microsecond=0) - ADS_LAG
    return min(end, cap)


def _year_request(year: int, start: datetime, end: datetime) -> dict | None:
    """ADS EAC4 wants a date range, not year/month/day keys."""
    lo = max(start, datetime(year, 1, 1, tzinfo=timezone.utc))
    hi_exclusive = min(end, datetime(year + 1, 1, 1, tzinfo=timezone.utc))
    if hi_exclusive <= lo:
        return None
    hi = hi_exclusive - timedelta(seconds=1)
    return {
        "variable": list(VARIABLES),
        "date": [f"{lo:%Y-%m-%d}/{hi:%Y-%m-%d}"],
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
    }


def _netcdf_path(path: Path) -> Path:
    if not is_zipfile(path):
        return path
    with ZipFile(path) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".nc")]
        if not names:
            raise ValueError(f"no netcdf in {path}")
        dest = path.with_suffix(".nc")
        dest.write_bytes(zf.read(names[0]))
    return dest


def parse_so2_netcdf(path: Path) -> pd.DataFrame:
    """Spatial-mean daily total-column SO2 (kg m-2) over the Kermadec box."""
    import netCDF4
    import numpy as np

    with netCDF4.Dataset(path) as ds:
        time_name = "valid_time" if "valid_time" in ds.variables else "time"
        times = netCDF4.num2date(
            ds[time_name][:],
            ds[time_name].units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        name = next(
            (n for n in ("tcso2", "total_column_sulphur_dioxide", "so2") if n in ds.variables),
            None,
        )
        if name is None:
            return pd.DataFrame()
        arr = np.asarray(ds[name][:], dtype=float)
        if hasattr(ds[name], "missing_value"):
            arr = np.where(arr == ds[name].missing_value, np.nan, arr)
        while arr.ndim > 1:
            arr = np.nanmean(arr, axis=-1)
    rows = []
    for i, stamp in enumerate(times):
        observed = stamp if getattr(stamp, "tzinfo", None) else stamp.replace(tzinfo=timezone.utc)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        else:
            observed = observed.astimezone(timezone.utc)
        rows.append({"observed_at": observed, "so2_kermadec_kgm2": float(arr[i])})
    return pd.DataFrame(rows)


def _month_request(year: int, month: int, start: datetime, end: datetime) -> dict | None:
    lo = max(start, datetime(year, month, 1, tzinfo=timezone.utc))
    if month == 12:
        hi_exclusive = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        hi_exclusive = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    hi_exclusive = min(end, hi_exclusive)
    if hi_exclusive <= lo:
        return None
    hi = hi_exclusive - timedelta(seconds=1)
    return {
        "variable": list(VARIABLES),
        "date": [f"{lo:%Y-%m-%d}/{hi:%Y-%m-%d}"],
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
    }


def _retrieve_so2(client, dest: Path, request: dict, label: str) -> pd.DataFrame | None:
    print(f"  CAMS SO2 {label} {request['date'][0]}", flush=True)
    try:
        client.retrieve(DATASET, request, str(dest))
    except Exception as exc:  # noqa: BLE001
        print(f"  cams.so2 {label} skipped: {exc}", flush=True)
        return None
    try:
        return parse_so2_netcdf(_netcdf_path(dest))
    except Exception as exc:  # noqa: BLE001
        print(f"  cams.so2 parse {label} failed: {exc}", flush=True)
        return None


def pull_so2(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """CAMS SO2 over Kermadec. Skip if CDS is not configured."""
    end = eac4_end(end)
    if end <= start:
        print("  cams.so2 skipped (inside EAC4 lag)", flush=True)
        return pd.DataFrame()
    try:
        client = _cds_client()
    except Exception as exc:  # noqa: BLE001
        print(f"  cams.so2 skipped (no CDS): {exc}", flush=True)
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    with TemporaryDirectory(prefix="eqpred_so2_") as tmp:
        tmp_path = Path(tmp)
        for year in range(start.year, end.year + 1):
            if datetime(year, 1, 1, tzinfo=timezone.utc) >= end:
                break
            if datetime(year, 12, 31, tzinfo=timezone.utc) < start:
                continue
            request = _year_request(year, start, end)
            if request is None:
                continue
            dest = tmp_path / f"so2_{year}.nc"
            part = _retrieve_so2(client, dest, request, str(year))
            if part is not None and not part.empty:
                frames.append(part)
                continue
            year_published = False
            for month in range(1, 13):
                month_req = _month_request(year, month, start, end)
                if month_req is None:
                    continue
                month_dest = tmp_path / f"so2_{year}{month:02d}.nc"
                month_part = _retrieve_so2(client, month_dest, month_req, f"{year}-{month:02d}")
                if month_part is not None and not month_part.empty:
                    frames.append(month_part)
                    year_published = True
                elif not year_published and month_part is None:
                    print(f"  cams.so2 {year} not on ADS yet", flush=True)
                    break
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[(raw["observed_at"] >= start) & (raw["observed_at"] < end)]
    if raw.empty:
        return pd.DataFrame()
    out = []
    for rec in raw.to_dict(orient="records"):
        observed = rec["observed_at"]
        if hasattr(observed, "to_pydatetime"):
            observed = observed.to_pydatetime()
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        out.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "ingest_run_id": ingest_run_id,
                "so2_kermadec_kgm2": rec["so2_kermadec_kgm2"],
            }
        )
    return pd.DataFrame(out)
