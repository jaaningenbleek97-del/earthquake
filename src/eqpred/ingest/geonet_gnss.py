from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_json
from eqpred.records import iso

# GeoNet Tilde daily GNSS. FITS (fits.geonet.org.nz) is gone — NXDOMAIN.
TILDE_URL = (
    "https://tilde.geonet.org.nz/v4/data/gnss/"
    "{site}/displacement/nil/1d/{aspect}/{start}/{end}"
)
NZ_SITES = ("AUCK", "WGTG", "WGTN", "LYTT", "DUND", "CHTI", "RAUL")
ASPECTS = {"east": "east_mm", "north": "north_mm", "up": "vert_mm"}
LICENSE = "https://www.geonet.org.nz/data/types/geodetic"
SOURCE = "geonet.gnss"
LATENCY = timedelta(days=2)


def parse_fits_csv(text: str) -> list[tuple[datetime, float]]:
    """Legacy FITS CSV. Host is dead; kept so old fixtures still parse."""
    rows: list[tuple[datetime, float]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            parts = line.split()
        if len(parts) < 4:
            continue
        if parts[0].lower().replace("-", "") in {"siteid", "site"}:
            continue
        try:
            stamp = parts[2].replace("Z", "+00:00")
            observed = datetime.fromisoformat(stamp)
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            value = float(parts[3])
        except (ValueError, IndexError):
            continue
        rows.append((observed, value))
    return rows


def parse_tilde_json(payload: object) -> list[tuple[datetime, float]]:
    """Tilde GNSS displacements. API values are metres; we store millimetres."""
    rows: list[tuple[datetime, float]] = []
    blocks = payload if isinstance(payload, list) else [payload]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for rec in block.get("data") or []:
            ts = rec.get("ts")
            val = rec.get("val")
            if ts is None or val is None:
                continue
            try:
                observed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                rows.append((observed, float(val) * 1000.0))
            except (TypeError, ValueError):
                continue
    return rows


def _windows(start: datetime, end: datetime, days: int = 30):
    """Tilde 400s on windows longer than about a month, and on future end dates."""
    cursor = start
    step = timedelta(days=days)
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


def _tilde_window(site: str, aspect: str, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    if end <= start:
        return []
    url = TILDE_URL.format(
        site=site,
        aspect=aspect,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    print(f"  Tilde {site} {aspect} {start.date()}->{end.date()}", flush=True)
    try:
        payload = fetch_json(url, timeout=90)
    except Exception as exc:  # noqa: BLE001 — one dead site must not kill the overlay
        span = (end - start).days
        if "HTTP 400" in str(exc) and span > 3:
            mid = start + timedelta(days=span // 2)
            return _tilde_window(site, aspect, start, mid) + _tilde_window(site, aspect, mid, end)
        print(f"    skip {site} {aspect}: {exc}", flush=True)
        return []
    return parse_tilde_json(payload)


def pull_geonet_gnss(
    start: datetime,
    end: datetime,
    ingest_run_id: str,
    sites: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Daily NZ GNSS from GeoNet Tilde, one row per station. A dead site is skipped."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if end > now:
        end = now
    chosen = sites or NZ_SITES
    print(f"  GeoNet Tilde {len(chosen)} stations {start.date()} -> {end.date()}", flush=True)

    def _one(site: str) -> list[dict]:
        rows: list[dict] = []
        parsed: dict[str, list[tuple[datetime, float]]] = {col: [] for col in ASPECTS.values()}
        for aspect, col in ASPECTS.items():
            got: list[tuple[datetime, float]] = []
            for win_start, win_end in _windows(start, end):
                got.extend(_tilde_window(site, aspect, win_start, win_end))
            parsed[col] = got
        if not any(parsed.values()):
            return rows
        by_day: dict[pd.Timestamp, dict[str, float]] = {}
        for col, samples in parsed.items():
            for observed, value in samples:
                day = pd.Timestamp(observed).tz_convert("UTC").floor("D")
                by_day.setdefault(day, {})[col] = value
        for day, cols in by_day.items():
            if "east_mm" not in cols:
                continue
            observed = day.to_pydatetime()
            rows.append(
                {
                    "source": SOURCE,
                    "observed_at": iso(observed),
                    "available_at": iso(observed + LATENCY),
                    "license": LICENSE,
                    "ingest_run_id": ingest_run_id,
                    "site": site,
                    "east_mm": cols.get("east_mm"),
                    "north_mm": cols.get("north_mm"),
                    "vert_mm": cols.get("vert_mm"),
                }
            )
        print(f"    {site} {len(rows)} days", flush=True)
        return rows

    found: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_one, site) for site in chosen]
        for fut in as_completed(futures):
            found.extend(fut.result())
    return pd.DataFrame(found)
