from __future__ import annotations

import configparser
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from eqpred.records import iso
from eqpred.regions import REGIONS, in_bbox
from eqpred.stamp import utc_ts

SOURCE = "esa.swarm"
LICENSE = "https://earth.esa.int/eogateway/missions/swarm"
VIRES_OWS = "https://vires.services/ows"
# OPER MAG L1b is usually a few days late. FAST exists; we stay conservative.
LATENCY = timedelta(days=4)
# First routine MAGA_LR OPER after commissioning.
MISSION_START = datetime(2014, 4, 8, tzinfo=timezone.utc)
CHUNK_DAYS = 31
SAMPLE_STEP = "PT5M"
COLLECTION = "SW_OPER_MAGA_LR_1B"
# Dedicated 2D core is only valid through 2018-01-01. CHAOS-Core covers the rest.
CORE_MODEL = "MCO_SHA_2D"
CORE_MODELS = ("MCO_SHA_2D", "CHAOS-Core")
MCO_2D_END = datetime(2018, 1, 1, tzinfo=timezone.utc)
TRANSIENT_ERRORS = (
    "10054",
    "10060",
    "timed out",
    "Timeout",
    "ConnectionReset",
    "RemoteDisconnected",
    "Connection aborted",
)
BOXES = ("hikurangi", "alpine", "kermadec")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_local_env() -> None:
    """Read VIRES_* from a local .env. Never copies a password into environ."""
    path = _repo_root() / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"VIRES_TOKEN", "ESA_VIRES_TOKEN", "VIRES_ACCESS_TOKEN"}:
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def vires_token() -> str | None:
    """Access token from env or ~/.viresclient.ini. Not an ESA password."""
    _load_local_env()
    for key in ("VIRES_TOKEN", "ESA_VIRES_TOKEN", "VIRES_ACCESS_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    home = Path.home()
    for path in (home / ".viresclient.ini", home / "_viresclient.ini"):
        if not path.is_file():
            continue
        cfg = configparser.ConfigParser()
        try:
            cfg.read(path, encoding="utf-8")
        except (OSError, configparser.Error):
            continue
        for section in (VIRES_OWS, "default"):
            if cfg.has_option(section, "token"):
                token = cfg.get(section, "token").strip()
                if token:
                    return token
    return None


def _residual_series(frame: pd.DataFrame) -> pd.Series | None:
    for name in frame.columns:
        if str(name).startswith("F_res"):
            return pd.to_numeric(frame[name], errors="coerce")
    if "f_res_nt" in frame.columns:
        return pd.to_numeric(frame["f_res_nt"], errors="coerce")
    if "F" in frame.columns:
        return pd.to_numeric(frame["F"], errors="coerce")
    return None


def normalize_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a VirES MAG frame (or a test table) into observed_at / lat / lon / f_res_nt."""
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    if work.index.name in {"Timestamp", "timestamp"} or (
        not {"observed_at", "Timestamp", "timestamp"}.intersection(work.columns)
        and pd.api.types.is_datetime64_any_dtype(work.index)
    ):
        work = work.reset_index()
    time_col = next(
        (c for c in ("observed_at", "Timestamp", "timestamp") if c in work.columns),
        None,
    )
    lat_col = next((c for c in ("lat", "Latitude", "latitude") if c in work.columns), None)
    lon_col = next((c for c in ("lon", "Longitude", "longitude") if c in work.columns), None)
    residual = _residual_series(work)
    if time_col is None or lat_col is None or lon_col is None or residual is None:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(work[time_col], utc=True),
            "lat": pd.to_numeric(work[lat_col], errors="coerce"),
            "lon": pd.to_numeric(work[lon_col], errors="coerce"),
            "f_res_nt": residual.to_numpy(),
        }
    )
    return out.dropna(subset=["observed_at", "lat", "lon", "f_res_nt"]).reset_index(drop=True)


def _box_mean(samples: pd.DataFrame, key: str) -> pd.Series:
    region = REGIONS[key]
    tagged = samples.assign(_in=in_bbox(samples, region))
    return tagged.loc[tagged["_in"]].groupby("day")["f_res_nt"].mean()


def daily_swarm_features(samples: pd.DataFrame) -> pd.DataFrame:
    """Daily mean / range / max |dF| of the core-field residual, plus issued-box means."""
    work = normalize_samples(samples)
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values("observed_at")
    work["day"] = work["observed_at"].dt.floor("D")
    work["df"] = work.groupby("day", sort=False)["f_res_nt"].diff().abs()
    grouped = work.groupby("day", sort=True)
    out = pd.DataFrame(
        {
            "observed_at": grouped["f_res_nt"].mean().index,
            "swarm_f_res_nt": grouped["f_res_nt"].mean().to_numpy(),
            "swarm_f_res_range_nt": (grouped["f_res_nt"].max() - grouped["f_res_nt"].min()).to_numpy(),
            "swarm_f_res_dbdt_nt": grouped["df"].max().to_numpy(),
        }
    )
    for key in BOXES:
        means = _box_mean(work, key)
        out[f"swarm_{key}_f_res_nt"] = out["observed_at"].map(means)
    out["source"] = SOURCE
    out["license"] = LICENSE
    out["available_at"] = [iso(ts.to_pydatetime() + LATENCY) for ts in out["observed_at"]]
    out["observed_at"] = [iso(ts.to_pydatetime()) for ts in out["observed_at"]]
    return out.reset_index(drop=True)


def models_for_window(start: datetime) -> tuple[str, ...]:
    """Skip 2D after its validity end so we do not download a month of NaNs."""
    if utc_ts(start).to_pydatetime() >= MCO_2D_END:
        return ("CHAOS-Core",)
    return CORE_MODELS


def _fetch_chunk(token: str, start: datetime, end: datetime, model: str) -> pd.DataFrame:
    from viresclient import SwarmRequest

    request = SwarmRequest(url=VIRES_OWS, token=token)
    request.set_collection(COLLECTION)
    request.set_products(
        measurements=["F"],
        models=[model],
        residuals=True,
        sampling_step=SAMPLE_STEP,
    )
    data = request.get_between(start_time=start, end_time=end, asynchronous=True)
    frame = data.as_dataframe()
    if frame is None or (hasattr(frame, "empty") and frame.empty):
        return pd.DataFrame()
    return daily_swarm_features(frame)


def _fetch_with_retry(token: str, start: datetime, end: datetime, model: str) -> pd.DataFrame:
    last: Exception | None = None
    for attempt in range(3):
        try:
            return _fetch_chunk(token, start, end, model)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            last = exc
            text = str(exc)
            if not any(needle in text for needle in TRANSIENT_ERRORS):
                raise
            print(f"  esa.swarm retry {start.date()} {model} ({attempt + 1}/3): {exc}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    if last is not None:
        raise last
    return pd.DataFrame()


def pull_swarm(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily Swarm-A MAG residual via VirES. Needs a token, not an ESA password."""
    _ = ingest_run_id
    token = vires_token()
    if not token:
        print(
            "  esa.swarm skipped: set VIRES_TOKEN from "
            "https://vires.services/accounts/tokens/ (access token, not your ESA password)",
            flush=True,
        )
        return pd.DataFrame()
    try:
        import viresclient  # noqa: F401
    except ImportError:
        print("  esa.swarm skipped: pip install viresclient", flush=True)
        return pd.DataFrame()

    cursor = max(utc_ts(start).to_pydatetime(), MISSION_START)
    stop = utc_ts(end).to_pydatetime()
    if cursor >= stop:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    while cursor < stop:
        nxt = min(cursor + timedelta(days=CHUNK_DAYS), stop)
        print(f"  esa.swarm {cursor.date()} -> {nxt.date()}", flush=True)
        part = pd.DataFrame()
        used = ""
        for model in models_for_window(cursor):
            try:
                part = _fetch_with_retry(token, cursor, nxt, model)
            except Exception as exc:  # noqa: BLE001 — optional overlay
                print(f"  esa.swarm skip {cursor.date()} {model}: {exc}", flush=True)
                continue
            if not part.empty:
                used = model
                break
            print(f"  esa.swarm {cursor.date()} {model} 0 days", flush=True)
        if not part.empty:
            frames.append(part)
            print(f"  esa.swarm {cursor.date()} {used} {len(part)} days", flush=True)
        elif used == "":
            print(f"  esa.swarm {cursor.date()} empty", flush=True)
        time.sleep(0.4)
        cursor = nxt
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
