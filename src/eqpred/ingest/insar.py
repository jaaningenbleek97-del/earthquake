"""COMET LiCSAR frame-average line-of-sight displacement.

Full rasters are huge. We only keep a frame if the public JASMIN store has a
small text time series (LiCSBAS ts / metadata epochs). Frames that only have
GeoTIFFs are skipped. 30-day available_at lag — LiCSAR is a delayed product.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso

SOURCE = "comet.insar"
LICENSE = "https://comet.nerc.ac.uk/comet-lics-portal/"
LATENCY = timedelta(days=30)
BASE = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products"

# track / frame / slug. Centroids sit in the issued boxes.
# Live JASMIN folders use track 8/52/81/175, not CEDA's stale 008/. Polys
# checked 2026-09-20: these four sit on NZ (BOP, Hikurangi, Kaikoura, Hawke's Bay).
FRAMES: tuple[tuple[str, str, str], ...] = (
    ("175", "175D_12845_111312", "hikurangi"),
    ("8", "008A_12731_060000", "bop"),
    ("52", "052A_13183_181613", "alpine"),
    ("81", "081A_12881_141414", "hawkes"),
)


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, val in attrs:
            if key == "href" and val:
                self.hrefs.append(val)


def _list_dir(url: str) -> list[str]:
    text = fetch_text(url if url.endswith("/") else url + "/", timeout=90)
    parser = _HrefParser()
    parser.feed(text)
    return parser.hrefs


def _parse_ts_csv(text: str) -> pd.DataFrame:
    """Accept date,value or year month day value text time series."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#%;":
            continue
        if line.lower().startswith(("date", "time", "yyyy", "year")):
            continue
        parts = [p.strip() for p in line.replace(",", " ").replace("\t", " ").split() if p.strip()]
        if len(parts) < 2:
            continue
        value = None
        observed = None
        try:
            if "-" in parts[0] and len(parts[0]) >= 8:
                observed = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
                value = float(parts[1])
            elif len(parts) >= 4 and parts[0].isdigit():
                observed = datetime(int(parts[0]), int(parts[1]), int(parts[2]), tzinfo=timezone.utc)
                value = float(parts[3])
        except (TypeError, ValueError):
            continue
        if observed is None or value is None:
            continue
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        rows.append({"observed_at": observed, "los_mm": value})
    return pd.DataFrame(rows)


def _pull_frame(track: str, frame: str, slug: str) -> pd.DataFrame:
    root = f"{BASE}/{track}/{frame}"
    candidates = (
        f"{root}/LiCSBAS/ts.txt",
        f"{root}/LiCSBAS/ts_cf.txt",
        f"{root}/LiCSBAS/network.txt",
        f"{root}/timeseries/ts.txt",
        f"{root}/metadata/ts.txt",
    )
    for url in candidates:
        try:
            text = fetch_text(url, timeout=90)
        except Exception:
            continue
        parsed = _parse_ts_csv(text)
        if not parsed.empty:
            parsed["site"] = slug
            print(f"    {slug} {len(parsed)} epochs from {url.rsplit('/', 1)[-1]}", flush=True)
            return parsed
    # Last try: directory listing of LiCSBAS for any small .txt / .csv
    for folder in ("LiCSBAS", "timeseries", "metadata"):
        try:
            hrefs = _list_dir(f"{root}/{folder}")
        except Exception:
            continue
        for href in hrefs:
            name = href.split("?")[0].rstrip("/").rsplit("/", 1)[-1].lower()
            if not name.endswith((".txt", ".csv", ".dat")):
                continue
            if any(skip in name for skip in ("readme", "param", "log", "baseline")):
                continue
            try:
                text = fetch_text(f"{root}/{folder}/{href.split('/')[-1]}", timeout=90)
            except Exception:
                continue
            parsed = _parse_ts_csv(text)
            if len(parsed) >= 8:
                parsed["site"] = slug
                print(f"    {slug} {len(parsed)} epochs from {folder}/{name}", flush=True)
                return parsed
    print(f"    skip {slug}: no small time series on the public store", flush=True)
    return pd.DataFrame()


def pull_insar(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Frame-average LOS if a lightweight text series exists. Otherwise empty."""
    frames = []
    for track, frame, slug in FRAMES:
        print(f"  LiCSAR {slug} {frame}", flush=True)
        try:
            part = _pull_frame(track, frame, slug)
        except Exception as exc:  # noqa: BLE001 — one dead frame must not kill the overlay
            print(f"    skip {slug}: {exc}", flush=True)
            part = pd.DataFrame()
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[(raw["observed_at"] >= start) & (raw["observed_at"] < end)]
    if raw.empty:
        return pd.DataFrame()
    by_day: dict[pd.Timestamp, dict[str, float]] = {}
    avail_by_day: dict[pd.Timestamp, datetime] = {}
    for rec in raw.to_dict(orient="records"):
        observed = rec["observed_at"]
        if getattr(observed, "to_pydatetime", None):
            observed = observed.to_pydatetime()
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        day = pd.Timestamp(observed).tz_convert("UTC").floor("D")
        slot = by_day.setdefault(day, {})
        slot[f"insar_{rec['site']}_los_mm"] = float(rec["los_mm"])
        avail_by_day[day] = min(avail_by_day.get(day, observed + LATENCY), observed + LATENCY)
    out: list[dict] = []
    for day, cols in sorted(by_day.items()):
        row = {
            "source": SOURCE,
            "observed_at": iso(day.to_pydatetime()),
            "available_at": iso(avail_by_day[day]),
            "license": LICENSE,
            "ingest_run_id": ingest_run_id,
            **cols,
        }
        out.append(row)
    daily = pd.DataFrame(out)
    value_cols = [c for c in daily.columns if c.startswith("insar_") and c.endswith("_los_mm")]
    if value_cols:
        daily["insar_los_mm"] = daily[value_cols].mean(axis=1)
    return daily
