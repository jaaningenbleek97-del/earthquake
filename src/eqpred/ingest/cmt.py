"""Global CMT solutions. A magnitude does not say thrust versus slab.

The monthly archive is public by the first day of the next month.
Quick solutions in the current file are held 7 days.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.records import iso

SOURCE = "gcmt.cmt"
LICENSE = "https://www.globalcmt.org/"
ARCHIVE_URL = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec25.ndk"
QUICK_URL = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_QUICK/qcmt.ndk"
QUICK_HOLD = timedelta(days=7)

_HYPO = re.compile(
    r"(\d{4})/(\d{2})/(\d{2})\s+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
    r"([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\d+\.\d+)"
)
_CENTROID_DEPTH = re.compile(
    r"CENTROID:\s+[+-]?\d+\.?\d*\s+[+-]?\d+\.?\d*\s+[+-]?\d+\.?\d*\s+[+-]?\d+\.?\d*\s+"
    r"[+-]?\d+\.?\d*\s+[+-]?\d+\.?\d*\s+(\d+\.?\d*)"
)


def _moment_magnitude(exponent: int, components: list[float]) -> float:
    moment = math.sqrt(0.5 * sum(value * value for value in components)) * (10.0 ** exponent)
    if moment <= 0:
        return float("nan")
    return (2.0 / 3.0) * math.log10(moment) - 10.73


def _next_month(when: datetime) -> datetime:
    year = when.year + (1 if when.month == 12 else 0)
    month = 1 if when.month == 12 else when.month + 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _rake(line: str) -> float:
    numbers: list[float] = []
    for token in line.split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    if len(numbers) < 6:
        return float("nan")
    return numbers[-4]


def parse_ndk(text: str) -> pd.DataFrame:
    """Five-line NDK records into one row per solution."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    rows: list[dict[str, object]] = []
    for offset in range(0, len(lines) - 4, 5):
        block = lines[offset : offset + 5]
        found = _HYPO.search(block[0])
        if found is None:
            continue
        year, month, day, clock, lat, lon, depth = found.groups()
        hour, minute, second = clock.split(":")
        sec = float(second)
        whole = int(sec)
        micro = int(round((sec - whole) * 1_000_000))
        if whole >= 60 or micro >= 1_000_000:
            whole = 59
            micro = 999999
        when = datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            whole,
            micro,
            tzinfo=timezone.utc,
        )
        parts = block[3].split()
        try:
            exponent = int(float(parts[0]))
            components = [float(parts[index]) for index in range(1, 13, 2)]
        except (ValueError, IndexError):
            continue
        centroid = _CENTROID_DEPTH.search(block[2])
        centroid_depth = float(centroid.group(1)) if centroid else float(depth)
        quick = block[0].lstrip().startswith("PDEQ")
        public = when + QUICK_HOLD if quick else _next_month(when)
        rows.append(
            {
                "source": SOURCE,
                "cmt_id": block[1].split()[0] if block[1].split() else "",
                "observed_at": iso(when),
                "available_at": iso(public),
                "license": LICENSE,
                "lat": float(lat),
                "lon": float(lon),
                "depth_km": centroid_depth,
                "mw": _moment_magnitude(exponent, components),
                "rake": _rake(block[4]),
                "quick": quick,
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["cmt_id"], keep="last")
    return frame.sort_values("observed_at").reset_index(drop=True)


def pull_cmt(ingest_run_id: str) -> pd.DataFrame:
    """Monthly archive through 2025, then the quick file for newer solutions."""
    frames = []
    for url in (ARCHIVE_URL, QUICK_URL):
        print(f"  gcmt.cmt {url.rsplit('/', 1)[-1]}", flush=True)
        text = fetch_bytes(url, timeout=180).decode("utf-8", errors="replace")
        part = parse_ndk(text)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    # The monthly row is parsed first. A later quick copy of the same id loses.
    frame = frame.drop_duplicates(subset=["cmt_id"], keep="first")
    frame["ingest_run_id"] = ingest_run_id
    return frame.sort_values("observed_at").reset_index(drop=True)
