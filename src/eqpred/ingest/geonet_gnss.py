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


def _windows(start: datetime, end: datetime, years: int = 2):
    cursor = start
    while cursor < end:
        try:
            nxt = datetime(cursor.year + years, 1, 1, tzinfo=timezone.utc)
        except ValueError:
            nxt = end
        yield cursor, min(nxt, end)
        cursor = min(nxt, end)


def pull_geonet_gnss(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily NZ GNSS from GeoNet Tilde. Unused on issued p until a head beats ETAS."""
    by_day: dict[pd.Timestamp, dict[str, list[float]]] = {}
    for site in NZ_SITES:
        got_any = False
        for aspect, col in ASPECTS.items():
            parsed: list[tuple[datetime, float]] = []
            for win_start, win_end in _windows(start, end):
                url = TILDE_URL.format(
                    site=site,
                    aspect=aspect,
                    start=win_start.strftime("%Y-%m-%d"),
                    end=win_end.strftime("%Y-%m-%d"),
                )
                print(f"  Tilde {site} {aspect} {win_start.date()}->{win_end.date()}", flush=True)
                try:
                    payload = fetch_json(url, timeout=90)
                except Exception as exc:  # noqa: BLE001 — one dead site must not kill the overlay
                    print(f"    skip {site} {aspect}: {exc}", flush=True)
                    continue
                parsed.extend(parse_tilde_json(payload))
            if not parsed:
                continue
            got_any = True
            for observed, value in parsed:
                day = pd.Timestamp(observed).tz_convert("UTC").floor("D")
                slot = by_day.setdefault(day, {"east_mm": [], "north_mm": [], "vert_mm": []})
                slot[col].append(value)
        if not got_any and site == "WGTN":
            print("    WGTN empty (WGTG is the usual Wellington code)", flush=True)
    if not by_day:
        return pd.DataFrame()
    out = []
    for day, cols in sorted(by_day.items()):
        if not cols["east_mm"]:
            continue
        observed = day.to_pydatetime()
        out.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "ingest_run_id": ingest_run_id,
                "east_mm": float(sum(cols["east_mm"]) / len(cols["east_mm"])),
                "north_mm": float(sum(cols["north_mm"]) / len(cols["north_mm"])) if cols["north_mm"] else None,
                "vert_mm": float(sum(cols["vert_mm"]) / len(cols["vert_mm"])) if cols["vert_mm"] else None,
            }
        )
    return pd.DataFrame(out)
