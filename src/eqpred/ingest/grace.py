from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from eqpred.ingest.http import earthdata_credentials, fetch_bytes
from eqpred.records import iso
from eqpred.regions import REGIONS, point_in_region

SOURCE = "csr.grace"
LICENSE = "https://www2.csr.utexas.edu/grace/"
# Monthly mascons post late. Conservative gate.
LATENCY = timedelta(days=60)

# Public CSR RL06 mascon (one file, all months). Earthdata PODAAC is the fallback.
CSR_URLS = (
    "https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
    "http://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
    "https://www2.csr.utexas.edu/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
)
PODAAC_URLS = (
    "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/"
    "TELLUS_GRAC_L3_JPL_RL06_LND_v04/GRCTellus.JPL.200204_202207.GLO.RL06M.MSCNv03CRI.nc",
)

BOXES = ("hikurangi", "alpine", "kermadec", "japan", "chile")


def parse_grace_netcdf(path: Path) -> pd.DataFrame:
    """Monthly equivalent-water-height means: global + issued/transfer boxes."""
    import netCDF4
    import numpy as np

    with netCDF4.Dataset(path) as ds:
        time_name = "time" if "time" in ds.variables else next(iter(ds.variables))
        times = netCDF4.num2date(
            ds[time_name][:],
            getattr(ds[time_name], "units", "days since 2002-01-01"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        lat_name = next((n for n in ("lat", "latitude") if n in ds.variables), None)
        lon_name = next((n for n in ("lon", "longitude") if n in ds.variables), None)
        field_name = next(
            (n for n in ("lwe_thickness", "water_equivalent", "mascon", "lwe") if n in ds.variables),
            None,
        )
        if lat_name is None or lon_name is None or field_name is None:
            return pd.DataFrame()
        lats = np.asarray(ds[lat_name][:], dtype=float)
        lons = np.asarray(ds[lon_name][:], dtype=float)
        field = np.asarray(ds[field_name][:], dtype=float)
        if hasattr(ds[field_name], "missing_value"):
            field = np.where(field == ds[field_name].missing_value, np.nan, field)
        if hasattr(ds[field_name], "_FillValue"):
            field = np.where(field == ds[field_name]._FillValue, np.nan, field)

    if lons.size and float(np.nanmax(lons)) > 180:
        lons = ((lons + 180.0) % 360.0) - 180.0
    if field.ndim == 2:
        field = field[None, ...]
    # time, lat, lon or time, lon, lat
    if field.ndim != 3:
        return pd.DataFrame()
    if field.shape[1] == lons.size and field.shape[2] == lats.size:
        field = np.swapaxes(field, 1, 2)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    masks = {
        rid: np.array(
            [
                point_in_region(float(la), float(lo), REGIONS[rid])
                for la, lo in zip(lat_grid.ravel(), lon_grid.ravel(), strict=True)
            ]
        ).reshape(lat_grid.shape)
        for rid in BOXES
        if rid in REGIONS
    }
    rows = []
    for i, stamp in enumerate(times):
        observed = stamp if getattr(stamp, "tzinfo", None) else stamp.replace(tzinfo=timezone.utc)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        else:
            observed = observed.astimezone(timezone.utc)
        slab = np.asarray(field[i], dtype=float)
        rec: dict[str, object] = {
            "observed_at": observed.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            "grace_lwe_cm": float(np.nanmean(slab)),
        }
        for rid, mask in masks.items():
            vals = slab[mask]
            rec[f"grace_{rid}_lwe_cm"] = float(np.nanmean(vals)) if vals.size and np.isfinite(vals).any() else None
        rows.append(rec)
    return pd.DataFrame(rows)


def _try_urls(urls: tuple[str, ...], dest: Path, *, earthdata: bool) -> Path | None:
    for url in urls:
        try:
            data = fetch_bytes(url, timeout=180, earthdata=earthdata)
        except Exception as exc:  # noqa: BLE001
            print(f"  {SOURCE} skip {url}: {exc}", flush=True)
            continue
        if len(data) < 1000:
            print(f"  {SOURCE} skip {url}: tiny payload", flush=True)
            continue
        dest.write_bytes(data)
        return dest
    return None


def pull_grace(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Monthly GRACE / GRACE-FO mascon means. Skip if no public file or Earthdata."""
    with TemporaryDirectory(prefix="eqpred_grace_") as tmp:
        dest = Path(tmp) / "grace.nc"
        got = _try_urls(CSR_URLS, dest, earthdata=False)
        if got is None and earthdata_credentials() is not None:
            print("  csr.grace trying Earthdata PODAAC", flush=True)
            got = _try_urls(PODAAC_URLS, dest, earthdata=True)
        if got is None:
            print("  csr.grace empty (no public mascon / no Earthdata)", flush=True)
            return pd.DataFrame()
        try:
            raw = parse_grace_netcdf(got)
        except Exception as exc:  # noqa: BLE001
            print(f"  csr.grace parse failed: {exc}", flush=True)
            return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
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
        row = {
            "source": SOURCE,
            "observed_at": iso(observed),
            "available_at": iso(observed + LATENCY),
            "license": LICENSE,
            "ingest_run_id": ingest_run_id,
        }
        for key, value in rec.items():
            if key != "observed_at":
                row[key] = value
        out.append(row)
    return pd.DataFrame(out)
