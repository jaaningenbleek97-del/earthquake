from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from eqpred.ingest.http import fetch_json
from eqpred.records import iso

LICENSE = "https://www.weather.gov/disclaimer"
LATENCY = timedelta(hours=12)

# Operational JSON is short-horizon. OMNI/SILSO remain the long replay.
# These endpoints still matter for live cards and GOES X-ray peaks.
XRAY_URLS = (
    "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-30-day.json",
    "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json",
)
F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"


def _parse_stamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _class_to_flux(label: object) -> float | None:
    if label is None:
        return None
    text = str(label).strip().upper()
    if not text:
        return None
    letter = text[0]
    try:
        mult = float(text[1:] or "1")
    except ValueError:
        return None
    decade = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}.get(letter)
    if decade is None:
        return None
    return decade * mult


def parse_xray_flares(rows: list[Any]) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        observed = _parse_stamp(rec.get("max_time") or rec.get("time_tag") or rec.get("begin_time"))
        if observed is None:
            continue
        flux = rec.get("max_xrlong")
        try:
            flux_f = float(flux) if flux is not None else _class_to_flux(rec.get("max_class"))
        except (TypeError, ValueError):
            flux_f = _class_to_flux(rec.get("max_class"))
        out.append(
            {
                "source": "noaa.swpc",
                "observed_at": iso(observed.replace(hour=0, minute=0, second=0, microsecond=0)),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "xray_peak_wm2": flux_f,
                "xray_class": rec.get("max_class"),
            }
        )
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out)
    return (
        frame.sort_values("xray_peak_wm2", ascending=False)
        .drop_duplicates("observed_at")
        .sort_values("observed_at")
        .reset_index(drop=True)
    )


def parse_f107(rows: list[Any]) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        observed = _parse_stamp(rec.get("time_tag"))
        if observed is None:
            continue
        flux = rec.get("flux")
        try:
            flux_f = float(flux) if flux is not None else None
        except (TypeError, ValueError):
            flux_f = None
        out.append(
            {
                "observed_at": iso(observed.replace(hour=0, minute=0, second=0, microsecond=0)),
                "swpc_f107": flux_f,
            }
        )
    return pd.DataFrame(out).drop_duplicates("observed_at") if out else pd.DataFrame()


def parse_kp(rows: list[Any]) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        observed = _parse_stamp(rec.get("time_tag"))
        if observed is None:
            continue
        kp = rec.get("kp_index", rec.get("estimated_kp"))
        try:
            kp_f = float(kp) if kp is not None else None
        except (TypeError, ValueError):
            kp_f = None
        out.append(
            {
                "observed_at": iso(observed.replace(hour=0, minute=0, second=0, microsecond=0)),
                "swpc_kp": kp_f,
            }
        )
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out)
    return frame.groupby("observed_at", as_index=False)["swpc_kp"].max()


def _safe_json(url: str) -> list[Any]:
    try:
        payload = fetch_json(url, timeout=60)
    except Exception:
        return []
    if isinstance(payload, list):
        return payload
    return []


def pull_swpc(ingest_run_id: str) -> pd.DataFrame:
    xray_rows: list[Any] = []
    for url in XRAY_URLS:
        xray_rows = _safe_json(url)
        if xray_rows:
            break
    xray = parse_xray_flares(xray_rows)
    f107 = parse_f107(_safe_json(F107_URL))
    kp = parse_kp(_safe_json(KP_URL))

    if xray.empty and f107.empty and kp.empty:
        return pd.DataFrame()

    days = pd.concat(
        [part["observed_at"] for part in (xray, f107, kp) if not part.empty],
        ignore_index=True,
    ).drop_duplicates()
    frame = pd.DataFrame({"observed_at": days})
    if not xray.empty:
        frame = frame.merge(
            xray[["observed_at", "xray_peak_wm2", "xray_class", "available_at", "license"]],
            on="observed_at",
            how="left",
        )
    if not f107.empty:
        frame = frame.merge(f107, on="observed_at", how="left")
    if not kp.empty:
        frame = frame.merge(kp, on="observed_at", how="left")
    frame["source"] = "noaa.swpc"
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True) + LATENCY
    fallback = [iso(t.to_pydatetime()) for t in stamp]
    if "available_at" not in frame.columns:
        frame["available_at"] = fallback
    else:
        frame["available_at"] = frame["available_at"].fillna(pd.Series(fallback, index=frame.index))
    if "license" not in frame.columns:
        frame["license"] = LICENSE
    else:
        frame["license"] = frame["license"].fillna(LICENSE)
    return frame.sort_values("observed_at").reset_index(drop=True)
