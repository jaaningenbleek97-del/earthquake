from __future__ import annotations

"""GFZ Hpo (Hp30 / ap30) — higher-cadence Kp-like magnetosphere index."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "gfz.hpo"
LICENSE = "https://doi.org/10.5880/Hpo.0003"
LATENCY = timedelta(days=2)
URLS = (
    "https://kp.gfz.de/fileadmin/files_for_gfz_cms/Hp30_ap30_complete_series.txt",
    "https://kp.gfz.de/fileadmin/files_for_gfz_cms/Hp30_ap30_nowcast.txt",
    "https://kp.gfz.de/app/files/Hp30_ap30_nowcast.txt",
)


def parse_hpo_text(text: str) -> pd.DataFrame:
    """`YYYY MM DD hh.h hh._m days days_m Hp30 ap30 D` → one row per UT day."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            hp = float(parts[7])
            ap = float(parts[8])
            observed = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
        if hp < 0:
            hp = float("nan")
        if ap < 0:
            ap = float("nan")
        rows.append({"observed_at": iso(observed), "hp30": hp, "ap30": ap})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    daily = (
        frame.assign(_day=stamp.dt.floor("D"))
        .groupby("_day", as_index=False)
        .agg(hp_max=("hp30", "max"), hp_mean=("hp30", "mean"), ap30_max=("ap30", "max"))
        .rename(columns={"_day": "observed_at"})
    )
    daily["observed_at"] = daily["observed_at"].map(lambda t: iso(t.to_pydatetime()))
    return daily.sort_values("observed_at").reset_index(drop=True)


def pull_hpo(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for url in URLS:
        try:
            text = fetch_text(url, timeout=180)
        except Exception as exc:  # noqa: BLE001 — try the next GFZ file
            print(f"  gfz.hpo skip {url}: {exc}")
            continue
        parsed = parse_hpo_text(text)
        if parsed.empty:
            continue
        parts.append(parsed)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    stamp = pd.to_datetime(out["observed_at"], utc=True)
    out = (
        out.assign(_day=stamp.dt.floor("D"))
        .sort_values("_day")
        .drop_duplicates("_day", keep="last")
        .drop(columns=["_day"])
    )
    stamp = pd.to_datetime(out["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    out = out.loc[mask].copy()
    if out.empty:
        return pd.DataFrame()
    out["source"] = SOURCE
    out["license"] = LICENSE
    out["ingest_run_id"] = ingest_run_id
    out["available_at"] = (pd.to_datetime(out["observed_at"], utc=True) + LATENCY).map(
        lambda t: iso(t.to_pydatetime())
    )
    return out.sort_values("observed_at").reset_index(drop=True)
