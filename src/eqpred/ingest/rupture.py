"""GeoNet finite-fault models. A model is visible after its paper year.

Unpublished folders stay hidden until the day this file was fetched.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.records import iso

SOURCE = "geonet.rupture"
LICENSE = "https://github.com/GeoNet/data/tree/main/rupture-models"
API = "https://api.github.com/repos/GeoNet/data/contents/rupture-models"
FETCHED = datetime(2026, 10, 13, tzinfo=timezone.utc)

_DATE = re.compile(r"evdate\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
_ID = re.compile(r"public_ids\s*=\s*'([^']+)'")
_NUM = re.compile(r"(evlat|evlon|evdep|Mw)\s*=\s*([+-]?\d+\.?\d*)")
_YEAR = re.compile(r"(20\d{2}|19\d{2})")


def publication_day(folder: str) -> datetime:
    if "unpublished" in folder.lower():
        return FETCHED
    years = _YEAR.findall(folder)
    if not years:
        return FETCHED
    return datetime(int(years[-1]), 12, 31, tzinfo=timezone.utc)


def parse_model_meta(text: str, folder: str) -> dict[str, object] | None:
    found = _DATE.search(text)
    if found is None:
        return None
    day, month, year = (int(part) for part in found.groups())
    try:
        when = datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None
    numbers = {name: float(value) for name, value in _NUM.findall(text)}
    ident = _ID.search(text)
    return {
        "source": SOURCE,
        "event_id": ident.group(1) if ident else "",
        "observed_at": iso(when),
        "available_at": iso(publication_day(folder)),
        "license": LICENSE,
        "folder": folder,
        "lat": numbers.get("evlat"),
        "lon": numbers.get("evlon"),
        "depth_km": numbers.get("evdep"),
        "mw": numbers.get("Mw"),
    }


def _api_json(url: str) -> list[dict]:
    payload = json.loads(fetch_bytes(url, timeout=60).decode("utf-8"))
    if isinstance(payload, dict):
        raise RuntimeError(payload.get("message") or "github listing failed")
    return payload


def _walk(url: str) -> list[dict]:
    rows = []
    for item in _api_json(url):
        if item.get("type") == "dir":
            rows.extend(_walk(item["url"]))
        elif item.get("type") == "file" and item.get("download_url"):
            rows.append(item)
    return rows


def pull_rupture_models(root: Path, ingest_run_id: str) -> pd.DataFrame:
    """Save every model file and return one row per model.meta."""
    dest = root / "data" / "raw" / SOURCE / "files"
    metas: list[dict[str, object]] = []
    for item in _walk(API):
        relative = item["path"].split("rupture-models/", 1)[-1]
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size == 0:
            print(f"  geonet.rupture {relative}", flush=True)
            target.write_bytes(fetch_bytes(item["download_url"], timeout=60))
        if target.name == "model.meta":
            parsed = parse_model_meta(target.read_text(encoding="utf-8", errors="replace"), str(target.parent))
            if parsed is not None:
                parsed["ingest_run_id"] = ingest_run_id
                metas.append(parsed)
    if not metas:
        return pd.DataFrame()
    return pd.DataFrame(metas).sort_values("observed_at").reset_index(drop=True)


def models_from_tree(folder: Path, ingest_run_id: str) -> pd.DataFrame:
    """Read model.meta files already saved under folder."""
    metas: list[dict[str, object]] = []
    for path in folder.rglob("model.meta"):
        parsed = parse_model_meta(path.read_text(encoding="utf-8", errors="replace"), str(path.parent))
        if parsed is None:
            continue
        parsed["ingest_run_id"] = ingest_run_id
        metas.append(parsed)
    if not metas:
        return pd.DataFrame()
    return pd.DataFrame(metas).sort_values("observed_at").reset_index(drop=True)
