from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from eqpred.records import iso

SOURCE = "copernicus.era5"
LICENSE = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"
# ERA5 preliminary is ~5 days late; final later. Gate as conservative.
LATENCY = timedelta(days=5)
DATASET = "reanalysis-era5-single-levels"
# NZ + Kermadec, no dateline wrap. CDS area is North / West / South / East.
AREA = [-25.0, 165.0, -48.0, 179.5]
VARIABLES = ("surface_pressure", "volumetric_soil_water_layer_1")


def _cds_client():
    import os

    import cdsapi
    import certifi
    import truststore

    truststore.inject_into_ssl()
    bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    return cdsapi.Client()


def parse_era5_netcdf(path: Path) -> pd.DataFrame:
    """Spatial-mean daily surface pressure (Pa) and soil water (m3/m3)."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        times = netCDF4.num2date(
            ds["valid_time"][:] if "valid_time" in ds.variables else ds["time"][:],
            ds["valid_time"].units if "valid_time" in ds.variables else ds["time"].units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        def _mean(name: str) -> np.ndarray | None:
            if name not in ds.variables:
                return None
            arr = np.asarray(ds[name][:], dtype=float)
            if hasattr(ds[name], "missing_value"):
                arr = np.where(arr == ds[name].missing_value, np.nan, arr)
            # time, lat, lon
            while arr.ndim > 1:
                arr = np.nanmean(arr, axis=-1)
            return arr

        sp = _mean("sp") if "sp" in ds.variables else _mean("surface_pressure")
        soil = _mean("swvl1") if "swvl1" in ds.variables else _mean("volumetric_soil_water_layer_1")
    rows = []
    for i, stamp in enumerate(times):
        observed = stamp if getattr(stamp, "tzinfo", None) else stamp.replace(tzinfo=timezone.utc)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        else:
            observed = observed.astimezone(timezone.utc)
        rows.append(
            {
                "observed_at": observed,
                "era5_sp_pa": None if sp is None else float(sp[i]),
                "era5_swvl1": None if soil is None else float(soil[i]),
            }
        )
    return pd.DataFrame(rows)


def _year_request(year: int, start: datetime, end: datetime) -> dict:
    months = []
    for month in range(1, 13):
        first = datetime(year, month, 1, tzinfo=timezone.utc)
        if first >= end or datetime(year + (month == 12), 1 if month == 12 else month + 1, 1, tzinfo=timezone.utc) <= start:
            continue
        months.append(f"{month:02d}")
    return {
        "product_type": ["reanalysis"],
        "variable": list(VARIABLES),
        "year": [str(year)],
        "month": months,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def pull_era5(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """NZ-box ERA5 daily loading. Unused on issued p until a head beats ETAS."""
    client = _cds_client()
    frames: list[pd.DataFrame] = []
    with TemporaryDirectory(prefix="eqpred_era5_") as tmp:
        tmp_path = Path(tmp)
        for year in range(start.year, end.year + 1):
            if datetime(year, 1, 1, tzinfo=timezone.utc) >= end:
                break
            if datetime(year, 12, 31, tzinfo=timezone.utc) < start:
                continue
            request = _year_request(year, start, end)
            if not request["month"]:
                continue
            dest = tmp_path / f"era5_{year}.nc"
            print(f"  ERA5 {year} months={','.join(request['month'])}", flush=True)
            client.retrieve(DATASET, request, str(dest))
            frames.append(parse_era5_netcdf(dest))
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[(raw["observed_at"] >= start) & (raw["observed_at"] < end)]
    if raw.empty:
        return pd.DataFrame()
    out = []
    for rec in raw.to_dict(orient="records"):
        observed = rec["observed_at"].to_pydatetime() if hasattr(rec["observed_at"], "to_pydatetime") else rec["observed_at"]
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        out.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "ingest_run_id": ingest_run_id,
                "era5_sp_pa": rec["era5_sp_pa"],
                "era5_swvl1": rec["era5_swvl1"],
            }
        )
    return pd.DataFrame(out)
