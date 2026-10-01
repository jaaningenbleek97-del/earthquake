from __future__ import annotations

"""Hikurangi GNSS-A seafloor arrays on MGDS (2019–2024).

Campaign acoustic ranges, not land GNSS. Land GPS cannot see the shallow
offshore slip these arrays sit on.

Public download is not the DOI landing page. MGDS serves files through
`download_accept.php` after a research-use POST. Dataset UID 32490 is the
acoustic set (DOI 10.60521/332490). File UIDs are 3493225–3494630.
"""

import gzip
import io
import json
import re
import tarfile
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import pandas as pd

from eqpred.ingest.http import fetch_bytes, fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "mgds.gnssa"
LICENSE = "https://doi.org/10.60521/332490"
LATENCY = timedelta(days=30)
DATASET_UID = "32490"
FILE_UIDS_URL = "https://www.marine-geo.org/tools/search/file_uids.php?data_set_uid={uid}"
DOWNLOAD_ACCEPT = "https://api.marine-geo.org/services/download/download_accept.php"
BATCH = 40
RANGE_RE = re.compile(r"R(\d{5,})\[")
STAMP_RE = re.compile(r"(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})")
SITE_RE = re.compile(r"([A-Z]{2,6}\d{0,3})")


def parse_acoustics_csv(text: str, site: str = "") -> pd.DataFrame:
    """One acoustics CSV → daily median two-way time / range."""
    if not text or "," not in text:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    time_col = next((c for c in raw.columns if re.search(r"time|date", str(c), re.I)), None)
    val_col = next(
        (c for c in raw.columns if re.search(r"range|travel|twtt|delay|sec", str(c), re.I)),
        None,
    )
    if time_col is None:
        return pd.DataFrame()
    stamp = pd.to_datetime(raw[time_col], utc=True, errors="coerce")
    nums = pd.to_numeric(raw[time_col], errors="coerce")
    if nums.notna().any() and float(nums.dropna().median()) > 1e9:
        stamp = pd.to_datetime(nums, unit="s", utc=True)
    if val_col is None:
        nums = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        val_col = nums[0] if nums else None
    if val_col is None:
        return pd.DataFrame()
    work = pd.DataFrame({"observed_at": stamp, "gnssa_range": pd.to_numeric(raw[val_col], errors="coerce")})
    work = work.dropna()
    if work.empty:
        return pd.DataFrame()
    work["observed_at"] = work["observed_at"].dt.floor("D")
    daily = work.groupby("observed_at", as_index=False)["gnssa_range"].median()
    daily["gnssa_site"] = site or "hikurangi"
    return daily


def parse_bcsonardyne(text: str, site: str = "", name: str = "") -> pd.DataFrame:
    """Raw Sonardyne modem log → daily median R-count (two-way proxy)."""
    if not text:
        return pd.DataFrame()
    rows = []
    fallback = None
    stamp_from_name = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if stamp_from_name:
        fallback = datetime(
            int(stamp_from_name.group(1)),
            int(stamp_from_name.group(2)),
            int(stamp_from_name.group(3)),
            tzinfo=timezone.utc,
        )
    for line in text.splitlines():
        ranges = [int(m) for m in RANGE_RE.findall(line)]
        if not ranges:
            continue
        match = STAMP_RE.search(line)
        if match:
            try:
                observed = datetime.strptime(match.group(1), "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                observed = fallback
        else:
            observed = fallback
        if observed is None:
            continue
        rows.append({"observed_at": observed, "gnssa_range": float(sum(ranges) / len(ranges))})
    if not rows:
        return pd.DataFrame()
    work = pd.DataFrame(rows)
    work["observed_at"] = pd.to_datetime(work["observed_at"], utc=True).dt.floor("D")
    daily = work.groupby("observed_at", as_index=False)["gnssa_range"].median()
    daily["gnssa_site"] = site or "hikurangi"
    return daily


def _site_from_name(name: str) -> str:
    match = SITE_RE.search(name.replace("\\", "/").rsplit("/", 1)[-1])
    return match.group(1) if match else "hikurangi"


def _list_dataset_files(*, fetch_page=fetch_text) -> list[tuple[str, str]]:
    raw = fetch_page(FILE_UIDS_URL.format(uid=DATASET_UID), timeout=60)
    uids = json.loads(raw)
    if not isinstance(uids, list) or not uids:
        return []
    # Acoustics CSVs sit at the end of 32490 (UIDs 3494619–3494630). Pull those first.
    rows = [(str(uid), f"{uid}.bin") for uid in uids]
    prefer = [row for row in rows if row[0] >= "3494619"]
    extra = [row for row in rows if row[0] < "3494619"]
    return prefer + extra


def _download_uids(uids: list[str], *, fetch=fetch_bytes) -> bytes:
    body = urlencode(
        {
            "purpose": "Research",
            "client": "DataLink",
            "force_download": "1",
            "data_uids": ",".join(uids),
        }
    ).encode()
    return fetch(DOWNLOAD_ACCEPT, timeout=300, data=body)


def _frames_from_member(data: bytes, name: str) -> list[pd.DataFrame]:
    lower = name.lower()
    payload = data
    if lower.endswith(".gz") or data[:2] == b"\x1f\x8b":
        try:
            payload = gzip.decompress(data)
            lower = lower[: -3] if lower.endswith(".gz") else lower
            name = name[: -3] if name.lower().endswith(".gz") else name
        except OSError:
            return []
    text = payload.decode("utf-8", errors="replace")
    site = _site_from_name(name)
    if "acoust" in lower or lower.endswith(".csv"):
        parsed = parse_acoustics_csv(text, site)
    else:
        parsed = parse_bcsonardyne(text, site, name)
    return [parsed] if not parsed.empty else []


def _frames_from_archive(blob: bytes) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    if not blob:
        return out
    if blob[:2] == b"PK":
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for inner in zf.namelist():
                    if inner.endswith("/"):
                        continue
                    out.extend(_frames_from_member(zf.read(inner), inner))
        except zipfile.BadZipFile:
            return out
        return out
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            for inner in tf.getnames():
                member = tf.getmember(inner)
                if not member.isfile():
                    continue
                handle = tf.extractfile(member)
                if handle is None:
                    continue
                out.extend(_frames_from_member(handle.read(), inner))
    except tarfile.TarError:
        out.extend(_frames_from_member(blob, "download.bin"))
    return out


def pull_gnssa(start: datetime, end: datetime, run_id: str, *, fetch_page=fetch_text, fetch=fetch_bytes) -> pd.DataFrame:
    _ = run_id
    start = utc_ts(start)
    end = utc_ts(end)
    try:
        listed = _list_dataset_files(fetch_page=fetch_page)
    except Exception as exc:
        print(f"  mgds.gnssa file list skipped: {exc}", flush=True)
        listed = []
    if not listed:
        print("  mgds.gnssa no public file UIDs", flush=True)
        return pd.DataFrame()
    prefer = [(uid, name) for uid, name in listed if "acoust" in name.lower() or name.lower().endswith(".csv")]
    extra = [(uid, name) for uid, name in listed if (uid, name) not in prefer]
    # Acoustics CSVs first (12 files). Then campaign hourly logs in batches.
    queue = prefer + extra
    print(f"  mgds.gnssa {len(prefer)} acoustics + {len(extra)} campaign files", flush=True)
    frames: list[pd.DataFrame] = []
    for i in range(0, len(queue), BATCH):
        batch = queue[i : i + BATCH]
        uids = [uid for uid, _ in batch]
        try:
            blob = _download_uids(uids, fetch=fetch)
        except Exception as exc:
            print(f"    batch {i}-{i + len(batch)} skipped: {exc}", flush=True)
            continue
        parsed = _frames_from_archive(blob)
        frames.extend(parsed)
        days = sum(len(part) for part in parsed)
        print(f"    batch {i}-{i + len(batch)} {days} day-rows", flush=True)
        if prefer and i + BATCH >= len(prefer) and extra and i == 0:
            # keep going into hourly logs
            pass
    if not frames:
        return pd.DataFrame()
    work = pd.concat(frames, ignore_index=True)
    work = work.loc[(work["observed_at"] >= start) & (work["observed_at"] < end)]
    if work.empty:
        work = pd.concat(frames, ignore_index=True)
    daily = (
        work.groupby("observed_at", as_index=False)
        .agg(gnssa_range=("gnssa_range", "median"), gnssa_n=("gnssa_range", "count"))
    )
    daily["gnssa_hikurangi"] = daily["gnssa_range"]
    daily["gnssa_d1"] = daily["gnssa_range"].diff()
    rows = []
    for rec in daily.itertuples(index=False):
        day = rec.observed_at.to_pydatetime()
        if day.tzinfo is None:
            day = day.replace(tzinfo=timezone.utc)
        rows.append(
            {
                "observed_at": iso(day.replace(hour=0, minute=0, second=0, microsecond=0)),
                "available_at": iso(day.replace(hour=0, minute=0, second=0, microsecond=0) + LATENCY),
                "gnssa_range": float(rec.gnssa_range),
                "gnssa_hikurangi": float(rec.gnssa_hikurangi),
                "gnssa_d1": None if pd.isna(rec.gnssa_d1) else float(rec.gnssa_d1),
                "gnssa_n": int(rec.gnssa_n),
            }
        )
    return pd.DataFrame(rows)
