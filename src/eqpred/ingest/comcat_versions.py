from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from eqpred.ingest.comcat import COMCAT, LICENSE
from eqpred.ingest.http import fetch_json
from eqpred.records import iso
from eqpred.stamp import parse_utc

SOURCE = "usgs.comcat.versions"
# Detail history is expensive. M>=6 worldwide; issued boxes may go to 5.5.
VERSION_MINMAG = 6.0
BOX_VERSION_MINMAG = 5.5
VERSION_COLUMNS = (
    "source",
    "event_id",
    "version_at",
    "observed_at",
    "magnitude",
    "lon",
    "lat",
    "depth_km",
    "place",
    "status",
    "license",
    "ingest_run_id",
)


def _ms_to_utc(ms: object) -> datetime | None:
    """UTC from Unix milliseconds. Works for dates before 1970 on Windows."""
    if ms is None or ms == "":
        return None
    try:
        from datetime import timedelta

        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(ms) / 1000.0)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _parse_eventtime(raw: object, fallback_ms: object = None) -> datetime | None:
    if raw:
        try:
            ts = pd.Timestamp(str(raw))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.to_pydatetime()
        except (TypeError, ValueError):
            pass
    return _ms_to_utc(fallback_ms)


def _norm_status(product_status: object, evaluation_status: object) -> str:
    product = str(product_status or "").strip().upper()
    if product in {"DELETE", "DELETED"}:
        return "deleted"
    ev = str(evaluation_status or "").strip().lower()
    if ev in {"automatic", "preliminary"}:
        return "automatic"
    if ev in {"reviewed", "confirmed", "final"}:
        return "reviewed"
    if ev == "deleted":
        return "deleted"
    return "reviewed" if ev == "" else ev


def _float(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return value


def parse_origin_versions(detail: dict[str, Any], event_id: str | None = None) -> list[dict[str, Any]]:
    """Flatten ComCat detail `products.origin` (+ deleted-event) into version rows."""
    props = detail.get("properties") or {}
    products = props.get("products") or {}
    eid = event_id or detail.get("id")
    if not eid:
        return []
    place = props.get("place")
    fallback_ms = props.get("time")
    rows: list[dict[str, Any]] = []

    for origin in products.get("origin") or []:
        if not isinstance(origin, dict):
            continue
        version_at = _ms_to_utc(origin.get("updateTime"))
        if version_at is None:
            continue
        op = origin.get("properties") or {}
        status = _norm_status(origin.get("status"), op.get("evaluation-status"))
        mag = _float(op.get("magnitude"))
        if mag is None and status != "deleted":
            continue
        observed = _parse_eventtime(op.get("eventtime"), fallback_ms)
        if observed is None:
            continue
        rows.append(
            {
                "event_id": str(eid),
                "version_at": iso(version_at),
                "observed_at": iso(observed),
                "magnitude": mag,
                "lon": _float(op.get("longitude")),
                "lat": _float(op.get("latitude")),
                "depth_km": _float(op.get("depth")),
                "place": op.get("place") or place,
                "status": status,
            }
        )

    for gone in products.get("deleted-event") or []:
        if not isinstance(gone, dict):
            continue
        version_at = _ms_to_utc(gone.get("updateTime"))
        if version_at is None:
            continue
        gp = gone.get("properties") or {}
        observed = _parse_eventtime(gp.get("eventtime"), fallback_ms)
        rows.append(
            {
                "event_id": str(eid),
                "version_at": iso(version_at),
                "observed_at": iso(observed) if observed is not None else iso(version_at),
                "magnitude": _float(gp.get("magnitude")),
                "lon": _float(gp.get("longitude")),
                "lat": _float(gp.get("latitude")),
                "depth_km": _float(gp.get("depth")),
                "place": gp.get("place") or place,
                "status": "deleted",
            }
        )

    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["_vat"] = parse_utc(frame["version_at"])
    frame = frame.drop_duplicates(subset=["event_id", "_vat"], keep="last").drop(columns=["_vat"])
    return frame.to_dict("records")


def detail_url(event_id: str) -> str:
    params = {"eventid": event_id, "format": "geojson", "includesuperseded": "true"}
    return f"{COMCAT}?{urlencode(params)}"


def pull_versions(
    event_ids: list[str],
    ingest_run_id: str,
    *,
    pause_s: float = 0.08,
) -> pd.DataFrame:
    """Fetch ComCat detail history for each id. Failures are skipped, not fatal."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw_id in enumerate(event_ids):
        eid = str(raw_id).strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        try:
            payload = fetch_json(detail_url(eid))
        except Exception as exc:  # noqa: BLE001 — one event must not kill the pull
            print(f"  versions skip {eid}: {exc}")
            continue
        if not isinstance(payload, dict):
            print(f"  versions skip {eid}: not an object")
            continue
        parsed = parse_origin_versions(payload, event_id=eid)
        for rec in parsed:
            rec["source"] = SOURCE
            rec["license"] = LICENSE
            rec["ingest_run_id"] = ingest_run_id
            rows.append(rec)
        if pause_s and i + 1 < len(event_ids):
            time.sleep(pause_s)
        if (i + 1) % 50 == 0:
            print(f"  versions {i + 1}/{len(event_ids)} ({len(rows)} rows)")
    if not rows:
        return pd.DataFrame(columns=list(VERSION_COLUMNS))
    return pd.DataFrame(rows)


def normalize_version_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """UTC timestamps so merge on (event_id, version_at) is stable across pulls."""
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame(columns=list(VERSION_COLUMNS))
    out = frame.copy()
    if "event_id" in out.columns:
        out["event_id"] = out["event_id"].astype(str)
    if "version_at" in out.columns:
        out["version_at"] = parse_utc(out["version_at"])
    if "observed_at" in out.columns:
        out["observed_at"] = parse_utc(out["observed_at"])
    if "magnitude" in out.columns:
        out["magnitude"] = pd.to_numeric(out["magnitude"], errors="coerce")
    return out


def ids_in_window(
    catalog: pd.DataFrame,
    start: datetime,
    end: datetime,
    minmag: float,
    regions: list | None = None,
) -> list[str]:
    if catalog is None or catalog.empty or "event_id" not in catalog.columns:
        return []
    work = catalog.copy()
    work["observed_at"] = parse_utc(work["observed_at"])
    work["magnitude"] = pd.to_numeric(work["magnitude"], errors="coerce")
    pick = work.loc[
        (work["observed_at"] >= pd.Timestamp(start))
        & (work["observed_at"] < pd.Timestamp(end))
        & (work["magnitude"] >= float(minmag))
    ]
    if regions:
        from eqpred.regions import in_any_box

        pick = pick.loc[in_any_box(pick, regions)]
    return pick["event_id"].dropna().astype(str).unique().tolist()
