"""Montpellier shear-wave splitting database, New Zealand stations only.

Each measurement stays hidden until the year after its cited paper.
The world file is about 7 MB. Stations outside New Zealand and the
Kermadecs are dropped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.records import iso

SOURCE = "sws.db"
LICENSE = "http://splitting.gm.univ-montp2.fr/DB"
MEASUREMENTS = "https://splitting.gm.univ-montp2.fr/DB/public/splittingDB.txt"
REFERENCES = "https://splitting.gm.univ-montp2.fr/DB/public/splittingDB_ref.txt"
FILE_DATE = datetime(2026, 10, 5, tzinfo=timezone.utc)


def in_new_zealand(lat: float, lon: float) -> bool:
    if not (-48.0 <= lat <= -25.0):
        return False
    return (165.0 <= lon <= 180.0) or (-180.0 <= lon <= -175.0)


def parse_splitting_db(measurements: str, references: str) -> pd.DataFrame:
    refs = pd.read_csv(StringIO(references), sep="|", dtype=str, engine="python")
    years = {}
    if "id" in refs.columns and "Year" in refs.columns:
        for ref_id, year in zip(refs["id"], refs["Year"], strict=True):
            text = str(year).strip()
            if text.isdigit():
                years[str(ref_id).strip()] = int(text)
    rows = pd.read_csv(StringIO(measurements), sep="|", dtype=str, engine="python")
    if rows.empty or "Latitude" not in rows.columns:
        return pd.DataFrame()
    lat = pd.to_numeric(rows["Latitude"], errors="coerce")
    lon = pd.to_numeric(rows["Longitude"], errors="coerce")
    mask = []
    for a, b in zip(lat, lon, strict=True):
        if pd.isna(a) or pd.isna(b):
            mask.append(False)
        else:
            mask.append(in_new_zealand(float(a), float(b)))
    work = rows.loc[mask].copy()
    if work.empty:
        return pd.DataFrame()
    public = []
    for ref_id in work.get("refID", pd.Series(dtype=str)).astype(str):
        year = years.get(ref_id.strip())
        if year is None:
            public.append(FILE_DATE)
        else:
            public.append(datetime(year + 1, 1, 1, tzinfo=timezone.utc))
    out = pd.DataFrame(
        {
            "source": SOURCE,
            "station": work.get("Station", pd.Series(dtype=str)).astype(str).to_numpy(),
            "lat": pd.to_numeric(work["Latitude"], errors="coerce").to_numpy(),
            "lon": pd.to_numeric(work["Longitude"], errors="coerce").to_numpy(),
            "phi_deg": pd.to_numeric(work.get("phi"), errors="coerce").to_numpy(),
            "dt_s": pd.to_numeric(work.get("dt"), errors="coerce").to_numpy(),
            "ref_id": work.get("refID", pd.Series(dtype=str)).astype(str).to_numpy(),
            "license": LICENSE,
        }
    )
    out["available_at"] = [iso(stamp) for stamp in public]
    out["observed_at"] = out["available_at"]
    return out.reset_index(drop=True)


def pull_splitting_db(ingest_run_id: str) -> pd.DataFrame:
    print("  sws.db", flush=True)
    measurements = fetch_bytes(MEASUREMENTS, timeout=60).decode("utf-8", errors="replace")
    references = fetch_bytes(REFERENCES, timeout=60).decode("utf-8", errors="replace")
    frame = parse_splitting_db(measurements, references)
    if frame.empty:
        return frame
    frame["ingest_run_id"] = ingest_run_id
    return frame
