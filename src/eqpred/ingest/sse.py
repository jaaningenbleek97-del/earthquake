from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from eqpred.ingest.http import fetch_bytes, fetch_text
from eqpred.records import iso
from eqpred.regions import REGIONS, point_in_region

SOURCE = "slowslip.sse"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
EQDB_LICENSE = "https://eqdb.sakura.ne.jp/"
# Published catalogs are after-the-fact GNSS picks, not live detections.
LATENCY = timedelta(days=30)
ZENODO_ZIP = "https://zenodo.org/records/14848128/files/files_SSEcatalog.zip?download=1"
EQDB = "https://eqdb.sakura.ne.jp"
# SSE-only catalogs. Tremor/LFE dumps are huge and not this join.
EQDB_SSE_CATALOGS = (
    "Sekine2010-SSE",
    "Hirose2010-SSE",
    "Nishimura2013-SSE",
    "Ozawa2007-SSE",
    "Ozawa2013-SSE",
    "Wallace2012-SSE",
    "Wallace2016-SSE",
    "Wallace2018-SSE",
    "Wallace2020-SSE",
    "Klein2018-SSE",
    "Klein2022-SSE",
    "Socquet2017-SSE",
    "Rolandone2018-SSE",
    "Ruiz2014-SSE",
    "Radiguet2012-SSE",
)
BOXES = ("hikurangi", "alpine", "kermadec", "japan", "chile")
_DATE_KEYS = ("start", "tstart", "t_start", "date_start", "startdate", "onset")
_END_KEYS = ("end", "tend", "t_end", "date_end", "enddate", "finish")


def _utc_day(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "nat", "none"}:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = pd.to_datetime(text, utc=True).to_pydatetime()
            except (TypeError, ValueError):
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _matlab_datenum(value: float) -> datetime | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 200000 or number > 800000:
        return None
    origin = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return origin + timedelta(days=number - 719529)


def _as_day(value: object) -> datetime | None:
    parsed = _utc_day(value)
    if parsed is not None:
        return parsed
    if isinstance(value, (int, float)):
        return _matlab_datenum(float(value))
    return None


def _box_for(lat: float | None, lon: float | None, default: str | None = None) -> str | None:
    if lat is None or lon is None:
        return default
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return default
    for name in BOXES:
        if point_in_region(lat_f, lon_f, REGIONS[name]):
            return name
    return default


def parse_sloweq_csv(text: str) -> pd.DataFrame:
    """Slow Earthquake Database simple / duration CSV."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return pd.DataFrame()
    header = lines[0].lower()
    if "lat" in header and ("year" in header or "date" in header):
        frame = pd.read_csv(io.StringIO("\n".join(lines)))
    else:
        cols = [
            "year",
            "month",
            "day",
            "hour",
            "min",
            "sec",
            "lat",
            "lon",
            "dep",
            "mag",
            "catalog",
            "ref",
            "udate",
        ]
        if header.startswith("year") or "," in lines[0]:
            raw = pd.read_csv(io.StringIO("\n".join(lines)), header=None)
            if raw.shape[1] >= 10:
                names = cols[: raw.shape[1]]
                raw.columns = names + [f"extra_{i}" for i in range(len(names), raw.shape[1])]
                frame = raw
            else:
                frame = pd.read_csv(io.StringIO("\n".join(lines)))
        else:
            return pd.DataFrame()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    rows: list[dict[str, object]] = []
    for rec in frame.to_dict(orient="records"):
        if "date" in rec and rec.get("date"):
            observed = _utc_day(rec.get("date"))
        else:
            try:
                observed = datetime(
                    int(float(rec["year"])),
                    int(float(rec["month"])),
                    int(float(rec["day"])),
                    tzinfo=timezone.utc,
                )
            except (KeyError, TypeError, ValueError):
                continue
        if observed is None:
            continue
        lat = rec.get("lat")
        lon = rec.get("lon")
        try:
            lat_f = float(lat) if lat is not None and str(lat) != "" else None
            lon_f = float(lon) if lon is not None and str(lon) != "" else None
        except (TypeError, ValueError):
            lat_f = lon_f = None
        duration_s = rec.get("duration")
        try:
            duration_days = float(duration_s) / 86400.0 if duration_s not in (None, "") else 0.0
        except (TypeError, ValueError):
            duration_days = 0.0
        mag = rec.get("mag")
        try:
            mag_f = float(mag) if mag not in (None, "") else None
        except (TypeError, ValueError):
            mag_f = None
        end = observed + timedelta(days=max(0.0, duration_days))
        rows.append(
            {
                "start": observed,
                "end": end,
                "lat": lat_f,
                "lon": lon_f,
                "mag": mag_f,
                "catalog": rec.get("catalog"),
                "box": _box_for(lat_f, lon_f),
            }
        )
    return pd.DataFrame(rows)


def events_to_daily(events: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Expand SSE start/end windows into daily on/off flags per box."""
    if events is None or events.empty:
        return pd.DataFrame()
    days = pd.date_range(start.date(), (end - timedelta(seconds=1)).date(), freq="D", tz="UTC")
    if len(days) == 0:
        return pd.DataFrame()
    flags = {f"sse_{box}": [0.0] * len(days) for box in BOXES}
    counts = {f"sse_{box}_n": [0.0] * len(days) for box in BOXES}
    mags = {f"sse_{box}_mag": [float("nan")] * len(days) for box in BOXES}
    index = {ts.normalize(): i for i, ts in enumerate(days)}
    for rec in events.to_dict(orient="records"):
        box = rec.get("box") or _box_for(rec.get("lat"), rec.get("lon"))
        if box not in BOXES:
            continue
        t0 = _as_day(rec.get("start") or rec.get("observed_at"))
        t1 = _as_day(rec.get("end")) or t0
        if t0 is None:
            continue
        if t1 < t0:
            t0, t1 = t1, t0
        cursor = t0
        mag = rec.get("mag")
        try:
            mag_f = float(mag) if mag is not None and not pd.isna(mag) else None
        except (TypeError, ValueError):
            mag_f = None
        while cursor <= t1:
            stamp = pd.Timestamp(cursor).tz_convert("UTC").normalize()
            i = index.get(stamp)
            if i is not None:
                flags[f"sse_{box}"][i] = 1.0
                counts[f"sse_{box}_n"][i] += 1.0
                if mag_f is not None:
                    prev = mags[f"sse_{box}_mag"][i]
                    if prev != prev or mag_f > prev:
                        mags[f"sse_{box}_mag"][i] = mag_f
            cursor += timedelta(days=1)
    rows = []
    for i, day in enumerate(days):
        rec = {"observed_at": iso(day.to_pydatetime())}
        rec.update({key: values[i] for key, values in flags.items()})
        rec.update({key: values[i] for key, values in counts.items()})
        rec.update({key: values[i] for key, values in mags.items()})
        rows.append(rec)
    return pd.DataFrame(rows)


def _mat_attr(obj: object, name: str) -> object | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _looks_like_dates(values: list[object]) -> bool:
    hits = sum(1 for value in values[:8] if _as_day(value) is not None)
    return hits >= max(1, min(3, len(values[:8])))


def parse_hikurangi_mat(path: Path) -> pd.DataFrame:
    """Perez-Silva et al. 2025 MATLAB catalog. All events are Hikurangi."""
    from scipy.io import loadmat

    try:
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(f"MATLAB v7.3 file needs h5py: {path}") from exc
    starts: list[object] = []
    ends: list[object] = []

    def walk(obj: object, depth: int = 0) -> None:
        nonlocal starts, ends
        if depth > 6 or obj is None:
            return
        if isinstance(obj, dict):
            lower = {str(k).lower(): v for k, v in obj.items() if not str(k).startswith("__")}
            for key in _DATE_KEYS:
                if key in lower and not starts:
                    values = lower[key]
                    seq = list(values) if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)) else [values]
                    if _looks_like_dates(seq):
                        starts = seq
            for key in _END_KEYS:
                if key in lower and not ends:
                    values = lower[key]
                    seq = list(values) if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)) else [values]
                    if _looks_like_dates(seq):
                        ends = seq
            for value in lower.values():
                walk(value, depth + 1)
            return
        names = getattr(obj, "_fieldnames", None)
        if names:
            walk({name: _mat_attr(obj, name) for name in names}, depth + 1)
            return
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, np.ndarray, np.generic)):
            try:
                seq = list(obj)
            except TypeError:
                return
            if seq and all(getattr(item, "_fieldnames", None) for item in seq[:3] if item is not None):
                for item in seq:
                    walk(item, depth + 1)

    walk({k: v for k, v in mat.items() if not k.startswith("__")})
    if not starts:
        # Fall back: first two date-like vectors of equal length.
        vectors: list[list[object]] = []

        def collect(obj: object, depth: int = 0) -> None:
            if depth > 5 or obj is None:
                return
            if isinstance(obj, dict):
                for value in obj.values():
                    collect(value, depth + 1)
                return
            names = getattr(obj, "_fieldnames", None)
            if names:
                collect({name: _mat_attr(obj, name) for name in names}, depth + 1)
                return
            if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                try:
                    seq = list(obj)
                except TypeError:
                    return
                if seq and _looks_like_dates(seq):
                    vectors.append(seq)

        collect({k: v for k, v in mat.items() if not k.startswith("__")})
        if vectors:
            starts = vectors[0]
            ends = vectors[1] if len(vectors) > 1 and len(vectors[1]) == len(starts) else starts
    rows = []
    for i, start in enumerate(starts):
        t0 = _as_day(start)
        t1 = _as_day(ends[i]) if i < len(ends) else t0
        if t0 is None:
            continue
        rows.append({"start": t0, "end": t1 or t0, "box": "hikurangi", "catalog": "PerezSilva2025"})
    return pd.DataFrame(rows)


def parse_hikurangi_rda_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Perez-Silva simplified catalog: POSIXct start/end as unix seconds."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    cols = {str(c).lower(): c for c in frame.columns}
    start_col = next((cols[k] for k in cols if "start" in k and "date" in k or k in {"datestart", "date.start", "timestart"}), None)
    end_col = next((cols[k] for k in cols if "end" in k and "date" in k or k in {"dateend", "date.end", "timeend"}), None)
    if start_col is None:
        return pd.DataFrame()
    rows = []
    for rec in frame.to_dict(orient="records"):
        t0 = _posix_or_day(rec.get(start_col))
        t1 = _posix_or_day(rec.get(end_col)) if end_col else t0
        if t0 is None:
            continue
        rows.append(
            {
                "start": t0,
                "end": t1 or t0,
                "box": "hikurangi",
                "catalog": "PerezSilva2025",
                "nsta": rec.get("nsta") or rec.get("nSta"),
            }
        )
    return pd.DataFrame(rows)


def _posix_or_day(value: object) -> datetime | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if 1e8 < number < 2.2e9:
            return datetime.fromtimestamp(number, tz=timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
    return _as_day(value)


def parse_hikurangi_rda(path: Path) -> pd.DataFrame:
    import warnings

    import rdata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = rdata.read_rda(path)
    frame = parsed.get("SSEcatalog") if isinstance(parsed, dict) else parsed
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    return parse_hikurangi_rda_frame(frame)


def _pull_hikurangi(start: datetime, end: datetime) -> pd.DataFrame:
    print("  Zenodo Hikurangi SSE catalog", flush=True)
    blob = fetch_bytes(ZENODO_ZIP, timeout=180)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        zip_path = folder / "sse.zip"
        zip_path.write_bytes(blob)
        with ZipFile(zip_path) as zf:
            zf.extractall(folder)
        rda_files = [
            path
            for path in list(folder.rglob("*.Rda")) + list(folder.rglob("*.rda"))
            if not path.name.startswith("._")
        ]
        rda_files.sort(key=lambda p: (0 if "simplified_refmodel" in p.name else 1, p.name))
        events = pd.DataFrame()
        for path in rda_files:
            try:
                events = parse_hikurangi_rda(path)
            except Exception as exc:  # noqa: BLE001
                print(f"    rda skip {path.name}: {exc}", flush=True)
                continue
            if not events.empty:
                print(f"    {path.name} {len(events)} events", flush=True)
                break
        if events.empty:
            mat_files = [
                path
                for path in folder.rglob("SSEcatalog.mat")
                if not path.name.startswith("._")
            ]
            for path in mat_files:
                try:
                    events = parse_hikurangi_mat(path)
                except Exception as exc:  # noqa: BLE001
                    print(f"    mat skip {path.name}: {exc}", flush=True)
                    continue
                if not events.empty:
                    print(f"    {path.name} {len(events)} events", flush=True)
                    break
    if events.empty:
        print("    Hikurangi catalog empty", flush=True)
        return pd.DataFrame()
    return events


def _eqdb_catalogs_from_html(html: str) -> list[str]:
    found = set(re.findall(r"([A-Za-z][A-Za-z0-9]+-SSE)", html))
    return sorted(found) or list(EQDB_SSE_CATALOGS)


def _pull_eqdb(start: datetime, end: datetime) -> pd.DataFrame:
    print("  Slow Earthquake Database SSE catalogs", flush=True)
    html = ""
    try:
        html = fetch_text(EQDB, timeout=20)
    except Exception as exc:  # noqa: BLE001
        print(f"    eqdb homepage skipped: {exc}", flush=True)
    catalogs = [name for name in _eqdb_catalogs_from_html(html) if name.upper().endswith("-SSE")]
    if not catalogs:
        catalogs = list(EQDB_SSE_CATALOGS)
    frames: list[pd.DataFrame] = []
    span = f"{start.date()}/{end.date()}"
    for name in catalogs:
        params = {
            "catalog[]": name,
            "catalog": name,
            "format": "year,month,day,hour,min,sec,lat,lon,dep,mag,catalog,ref,udate",
            "start": str(start.date()),
            "end": str(end.date()),
            "duration": span,
        }
        urls = (
            f"{EQDB}/originals/{name}.csv",
            f"{EQDB}/php/download.php?{urlencode({'catalog': name, 'format': 'simple'})}",
        )
        text = ""
        for url in urls:
            try:
                text = fetch_text(url, timeout=12)
            except Exception:
                continue
            if text and "lat" in text.lower() and not text.lstrip().startswith("<!"):
                break
            text = ""
        if not text:
            try:
                raw = fetch_bytes(
                    f"{EQDB}/php/download.php",
                    timeout=12,
                    data=params,
                )
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = ""
        if not text or text.lstrip().startswith("<"):
            continue
        parsed = parse_sloweq_csv(text)
        if parsed.empty:
            continue
        print(f"    {name} {len(parsed)} events", flush=True)
        frames.append(parsed)
    if not frames:
        print("    eqdb SSE empty (form-only site)", flush=True)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pull_sse(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Hikurangi published SSE windows + Slow Earthquake Database SSE rows."""
    parts: list[pd.DataFrame] = []
    try:
        hik = _pull_hikurangi(start, end)
    except Exception as exc:  # noqa: BLE001
        print(f"  Hikurangi skipped: {exc}", flush=True)
        hik = pd.DataFrame()
    if not hik.empty:
        parts.append(hik)
    try:
        eqdb = _pull_eqdb(start, end)
    except Exception as exc:  # noqa: BLE001
        print(f"  eqdb skipped: {exc}", flush=True)
        eqdb = pd.DataFrame()
    if not eqdb.empty:
        parts.append(eqdb)
    if not parts:
        return pd.DataFrame()
    events = pd.concat(parts, ignore_index=True)
    daily = events_to_daily(events, start, end)
    if daily.empty:
        return pd.DataFrame()
    daily["source"] = SOURCE
    daily["ingest_run_id"] = ingest_run_id
    daily["license"] = LICENSE
    stamp = pd.to_datetime(daily["observed_at"], utc=True)
    daily["available_at"] = [iso(t.to_pydatetime() + LATENCY) for t in stamp]
    return daily
