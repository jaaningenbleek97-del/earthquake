from __future__ import annotations

import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from eqpred.ingest.http import fetch_bytes, fetch_json, write_cache
from eqpred.paths import data_dir
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "radon.public"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
# IRON sampled ~every 2–4 h. Replay as if we had the live feed, not the 2025 dump date.
LATENCY = timedelta(days=2)
ARTICLE_API = "https://api.figshare.com/v2/articles/27292554"
# Piersanti et al. 2025, Scientific Data. Soil radon 2009–2021, 62 Italian stations.
PREFIX = "Radon_Time_Serie_of_"


def _cache_dir() -> Path:
    path = data_dir() / "cache" / "iron"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _col(frame: pd.DataFrame, *needles: str, exclude: tuple[str, ...] = ()) -> str | None:
    for col in frame.columns:
        key = str(col).strip().lower()
        if any(token in key for token in exclude):
            continue
        if all(token in key for token in needles):
            return col
    return None


def _read_table(text: str) -> pd.DataFrame:
    sample = text[:2000]
    sep = ";" if sample.count(";") >= sample.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def parse_radon_csv(text: str, station: str) -> pd.DataFrame:
    """Parse one IRON Radon_Time_Serie CSV into UTC samples in Bq/m^3."""
    if not text or not text.strip():
        return pd.DataFrame()
    frame = _read_table(text)
    if frame.empty:
        return pd.DataFrame()
    date_col = _col(frame, "date")
    corr_col = _col(frame, "corrected")
    raw_col = _col(frame, "rn", exclude=("corrected", "instrument", "source"))
    if date_col is None or (corr_col is None and raw_col is None):
        return pd.DataFrame()
    observed = pd.to_datetime(frame[date_col], utc=True, errors="coerce")
    values = pd.to_numeric(frame[corr_col], errors="coerce") if corr_col is not None else pd.Series(dtype=float)
    if raw_col is not None:
        raw = pd.to_numeric(frame[raw_col], errors="coerce")
        values = raw if corr_col is None else values.fillna(raw)
    out = pd.DataFrame({"observed_at": observed, "radon_bq_m3": values})
    out = out.dropna(subset=["observed_at", "radon_bq_m3"])
    out = out.drop_duplicates(subset=["observed_at"], keep="first")
    if out.empty:
        return out
    out["radon_station"] = station
    return out.reset_index(drop=True)


def daily_radon_features(samples: pd.DataFrame) -> pd.DataFrame:
    """Network daily summary: median of station means, range, max |dRn|."""
    if samples.empty or "radon_bq_m3" not in samples.columns:
        return pd.DataFrame()
    work = samples.dropna(subset=["radon_bq_m3"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["observed_at"] = pd.to_datetime(work["observed_at"], utc=True)
    work = work.sort_values("observed_at")
    work["day"] = work["observed_at"].dt.floor("D")
    if "radon_station" in work.columns:
        work["dr"] = work.groupby(["radon_station", "day"], sort=False)["radon_bq_m3"].diff().abs()
        station_means = work.groupby(["day", "radon_station"], sort=True)["radon_bq_m3"].mean()
        network = station_means.groupby("day").median()
    else:
        work["dr"] = work.groupby("day", sort=False)["radon_bq_m3"].diff().abs()
        network = work.groupby("day", sort=True)["radon_bq_m3"].mean()
    day_groups = work.groupby("day", sort=True)
    out = pd.DataFrame(
        {
            "observed_at": network.index,
            "radon_bq_m3": network.to_numpy(),
            "radon_range_bq_m3": (
                day_groups["radon_bq_m3"].max() - day_groups["radon_bq_m3"].min()
            ).to_numpy(),
            "radon_dbdt_bq_m3": day_groups["dr"].max().reindex(network.index).to_numpy(),
        }
    )
    out["source"] = SOURCE
    out["license"] = LICENSE
    out["available_at"] = [iso(ts.to_pydatetime() + LATENCY) for ts in out["observed_at"]]
    out["observed_at"] = [iso(ts.to_pydatetime()) for ts in out["observed_at"]]
    return out.reset_index(drop=True)


def _station_from_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith(PREFIX):
        return stem[len(PREFIX) :]
    return stem


def _radon_files() -> list[dict]:
    payload = fetch_json(ARTICLE_API)
    if not isinstance(payload, dict):
        raise RuntimeError("figshare IRON article payload was not an object")
    files = payload.get("files") or []
    out: list[dict] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name.startswith(PREFIX) and name.endswith(".csv"):
            out.append(item)
    if not out:
        raise RuntimeError("figshare IRON article listed no Radon_Time_Serie CSVs")
    return out


def _load_station_csv(item: dict) -> str:
    name = str(item.get("name") or "radon.csv")
    url = str(item.get("download_url") or "")
    cache = _cache_dir() / name
    if cache.is_file() and cache.stat().st_size > 80:
        return cache.read_text(encoding="utf-8", errors="replace")
    if not url:
        return ""
    data = fetch_bytes(url, timeout=300)
    if not data:
        return ""
    write_cache(cache, data)
    return data.decode("utf-8", errors="replace")


def pull_radon(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily IRON soil radon series from the public Figshare dump. No account."""
    _ = ingest_run_id
    start_utc = utc_ts(start).to_pydatetime()
    stop = utc_ts(end).to_pydatetime()
    if start_utc >= stop:
        return pd.DataFrame()
    print("  radon.public IRON Figshare 27292554", flush=True)
    samples: list[pd.DataFrame] = []
    for item in _radon_files():
        name = str(item.get("name") or "")
        station = _station_from_name(name)
        try:
            text = _load_station_csv(item)
            part = parse_radon_csv(text, station)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  radon.public skip {station}: {exc}", flush=True)
            continue
        if part.empty:
            print(f"  radon.public skip {station}: no rows", flush=True)
            continue
        print(f"  radon.public {station} {len(part)} samples", flush=True)
        samples.append(part)
        time.sleep(0.15)
    if not samples:
        return pd.DataFrame()
    daily = daily_radon_features(pd.concat(samples, ignore_index=True))
    if daily.empty:
        return daily
    observed = pd.to_datetime(daily["observed_at"], utc=True)
    keep = (observed >= start_utc) & (observed < stop)
    return daily.loc[keep].reset_index(drop=True)
