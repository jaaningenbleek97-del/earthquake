"""GeoNet 1 Hz barometer / infrasound, reduced to a daily microbarom loudness.

The public IMS product ends on 1 January 2025. Station ETVZ has a 1 Hz
pressure channel from 2014 through the present, which covers the check years,
the 2019 exam, and the current card. Yesterday's reading only.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.paths import raw_dir
from eqpred.records import iso

SOURCE = "geonet.infrasound"
LICENSE = "https://www.geonet.org.nz/data/types/eq_catalogue"
GEONET = "https://service.geonet.org.nz/fdsnws/dataselect/1/query"
STATION = "ETVZ"
# Location 30 is the long barometer. Location 31 is the infrasound sensor.
SEGMENTS = (
    ("30", datetime(2014, 3, 1, tzinfo=timezone.utc), datetime(2025, 11, 1, tzinfo=timezone.utc)),
    ("31", datetime(2025, 3, 1, tzinfo=timezone.utc), datetime(2026, 9, 22, tzinfo=timezone.utc)),
)
UA = {"User-Agent": "earthquake-lab research"}


def _band_rms(samples: np.ndarray, fs: float = 1.0) -> float:
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < int(fs * 3600):
        return float("nan")
    x = x - np.mean(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    band = (freqs >= 0.15) & (freqs < 0.35)
    power = float(np.sum(np.abs(spec[band]) ** 2) / (x.size ** 2))
    return float(np.sqrt(max(power, 0.0)))


def _month_bytes(location: str, start: datetime, end: datetime) -> bytes | None:
    params = (
        f"network=NZ&station={STATION}&location={location}&channel=LDF"
        f"&starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&endtime={end.strftime('%Y-%m-%dT%H:%M:%S')}"
    )
    req = urllib.request.Request(f"{GEONET}?{params}", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            if response.status == 204:
                return None
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {204, 404}:
            return None
        raise


def _daily_from_bytes(blob: bytes, location: str) -> pd.DataFrame:
    from obspy import read

    stream = read(io.BytesIO(blob))
    pieces = []
    for trace in stream:
        times = pd.to_datetime(trace.times("timestamp"), unit="s", utc=True).floor("D")
        pieces.append(pd.DataFrame({"day": times, "value": np.asarray(trace.data, dtype=np.float64)}))
    frame = pd.concat(pieces, ignore_index=True)
    rows = []
    for day, part in frame.groupby("day"):
        rows.append(
            {
                "observed_at": day,
                "infra_etvz_band": _band_rms(part["value"].to_numpy(), fs=1.0),
                "infra_etvz_location": location,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def pull_etvz(root: Path | None = None) -> pd.DataFrame:
    """Month by month. Waveforms are not kept."""
    frames = []
    for location, start, stop in SEGMENTS:
        cursor = start
        while cursor < stop:
            year = cursor.year + (1 if cursor.month == 12 else 0)
            month = 1 if cursor.month == 12 else cursor.month + 1
            nxt = min(datetime(year, month, 1, tzinfo=timezone.utc), stop)
            try:
                blob = _month_bytes(location, cursor, nxt)
            except urllib.error.HTTPError as exc:
                print(f"etvz {location} {cursor.date()} skipped {exc.code}", flush=True)
                blob = None
            if blob:
                part = _daily_from_bytes(blob, location)
                if not part.empty:
                    frames.append(part)
                print(f"etvz {location} {cursor.date()} days {len(part)} bytes {len(blob)}", flush=True)
            else:
                print(f"etvz {location} {cursor.date()} empty", flush=True)
            cursor = nxt
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["observed_at", "infra_etvz_location"])
    # Prefer the infrasound sensor (location 31) when both exist.
    out = out.drop_duplicates("observed_at", keep="last").reset_index(drop=True)
    out["available_at"] = iso(datetime(2026, 9, 25, tzinfo=timezone.utc))
    out["source"] = SOURCE
    out["license"] = LICENSE
    dest = raw_dir(root) / "bgr.infrasound"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest / "etvz_daily.parquet", index=False)
    return out


def etvz_daily(root: Path | None = None) -> pd.DataFrame:
    path = raw_dir(root) / "bgr.infrasound" / "etvz_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
