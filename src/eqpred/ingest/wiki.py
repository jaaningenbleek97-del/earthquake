from __future__ import annotations

"""Wikipedia earthquake pageviews. Circus foil, not a precursor.

Wikimedia pageview REST. Two-day publisher lag. Never writes issued p.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_json
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "wiki.pageviews"
LICENSE = "https://wikimedia.org/api/rest_v1/"
LATENCY = timedelta(days=2)
ARTICLE = "Earthquake"
REST = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/daily/{start}/{end}"
)


def parse_pageviews(payload: object) -> pd.DataFrame:
    items = []
    if isinstance(payload, dict):
        items = payload.get("items") or []
    rows: list[dict[str, object]] = []
    for item in items:
        stamp = str(item.get("timestamp") or "")
        if len(stamp) < 8:
            continue
        day = datetime(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]), tzinfo=timezone.utc)
        views = item.get("views")
        try:
            n = float(views)
        except (TypeError, ValueError):
            continue
        observed = day
        rows.append(
            {
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "wiki_eq_views": n,
                "wiki_article": str(item.get("article") or ARTICLE),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates(subset=["observed_at"], keep="last")
    return frame.sort_values("observed_at").reset_index(drop=True)


def _chunk_bounds(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        nxt = min(datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc), end)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


def pull_wiki(
    start: datetime,
    end: datetime,
    run_id: str,
    *,
    fetch=fetch_json,
) -> pd.DataFrame:
    _ = run_id
    start = utc_ts(start)
    end = utc_ts(end)
    frames: list[pd.DataFrame] = []
    for a, b in _chunk_bounds(start, end):
        url = REST.format(
            article=ARTICLE,
            start=a.strftime("%Y%m%d00"),
            end=(b - timedelta(days=1)).strftime("%Y%m%d00"),
        )
        try:
            payload = fetch(url)
        except Exception:  # noqa: BLE001 — optional foil
            continue
        daily = parse_pageviews(payload)
        if not daily.empty:
            frames.append(daily)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["observed_at"], keep="last")
