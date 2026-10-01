from __future__ import annotations

"""WSPR path SNR over New Zealand.

Ham Weak Signal Propagation Reporter spots via wspr.live ClickHouse.
Nightly median SNR on ZL / NZ-grid paths. One-day latency.
"""

from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pandas as pd

from eqpred.ingest.http import fetch_json
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "ham.wspr"
LICENSE = "https://wspr.live/"
LATENCY = timedelta(days=1)
CLICKHOUSE = "https://db1.wspr.live/?query="


def _sql(start: datetime, end: datetime) -> str:
    a = start.strftime("%Y-%m-%d %H:%M:%S")
    b = end.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "SELECT toDate(time) AS d, quantile(0.5)(snr) AS wspr_snr_med, count() AS wspr_n "
        "FROM wspr.rx "
        f"WHERE time >= toDateTime('{a}') AND time < toDateTime('{b}') "
        "AND ("
        "startsWith(tx_sign, 'ZL') OR startsWith(rx_sign, 'ZL') "
        "OR startsWith(tx_loc, 'RE') OR startsWith(rx_loc, 'RE') "
        "OR startsWith(tx_loc, 'RF') OR startsWith(rx_loc, 'RF')"
        ") "
        "GROUP BY d ORDER BY d FORMAT JSON"
    )


def parse_wspr(payload: object) -> pd.DataFrame:
    rows_in: list[object] = []
    if isinstance(payload, dict):
        rows_in = payload.get("data") or payload.get("rows") or []
    elif isinstance(payload, list):
        rows_in = payload
    out: list[dict[str, object]] = []
    for item in rows_in:
        if not isinstance(item, dict):
            continue
        day = item.get("d") or item.get("observed_at") or item.get("time")
        if day is None:
            continue
        ts = pd.Timestamp(str(day)[:10], tz="UTC")
        try:
            snr = float(item.get("wspr_snr_med") if item.get("wspr_snr_med") is not None else item.get("med_snr"))
        except (TypeError, ValueError):
            continue
        try:
            n = float(item.get("wspr_n") if item.get("wspr_n") is not None else item.get("n") or 0)
        except (TypeError, ValueError):
            n = 0.0
        observed = ts.to_pydatetime()
        out.append(
            {
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "wspr_snr_med": snr,
                "wspr_n": n,
                "wspr_nz_snr": snr,
            }
        )
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out).drop_duplicates(subset=["observed_at"], keep="last")
    return frame.sort_values("observed_at").reset_index(drop=True)


def _chunk_bounds(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Year windows. wspr.live 408s an 8-year scan."""
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        nxt = min(datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc), end)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


def pull_wspr(
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
        url = CLICKHOUSE + quote(_sql(a, b), safe="")
        try:
            payload = fetch(url, timeout=180)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  ham.wspr {a.date()}->{b.date()} skipped: {exc}", flush=True)
            continue
        daily = parse_wspr(payload)
        if daily.empty:
            print(f"  ham.wspr {a.date()}->{b.date()} empty", flush=True)
            continue
        print(f"  ham.wspr {a.date()}->{b.date()} {len(daily)} days", flush=True)
        frames.append(daily)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["observed_at"], keep="last")
