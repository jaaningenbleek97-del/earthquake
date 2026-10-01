from __future__ import annotations

"""GeoNet GNSS ROTI and cycle slips from 30-second RINEX.

IGS official ROTI is northern-hemisphere polar only. Weekly IGS TEC is not
this product. These files sit on the public GeoNet S3 bucket.
"""

import gzip
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_bytes
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "geonet.roti"
LICENSE = "https://www.geonet.org.nz/data/types/geodetic"
LATENCY = timedelta(days=2)
STATIONS = ("GISB", "WGTN", "DUND", "AUCK")
C = 299792458.0
F1 = 1575.42e6
F2 = 1227.60e6
LAM1 = C / F1
LAM2 = C / F2
TEC_FACTOR = (F1**2 * F2**2) / (40.3e16 * (F1**2 - F2**2))
SLIP_CYCLES = 2.0
ROTI_WINDOW = 10  # 5 minutes of 30 s samples
EPOCH2_RE = re.compile(
    r"^ \d{2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+\d{1,2}\s+[-\d.]+\s+[0-1]\s+\d+"
)
SAT2_RE = re.compile(r"[GREJCS]\d{2}")
S3 = "https://geonet-open-data.s3-ap-southeast-2.amazonaws.com/gnss/rinex"


def _rinex3_url(station: str, day: datetime) -> str:
    doy = day.timetuple().tm_yday
    name = f"{station}00NZL_R_{day.year}{doy:03d}0000_01D_30S_MO.rnx.gz"
    return f"{S3}/{day.year}/{doy:03d}/{name}"


def _rinex2_url(station: str, day: datetime) -> str:
    doy = day.timetuple().tm_yday
    yy = day.year % 100
    name = f"{station.lower()}{doy:03d}0.{yy:02d}o.gz"
    return f"{S3}/{day.year}/{doy:03d}/{name}"


def _obs_index(types: list[str]) -> tuple[int | None, int | None]:
    l1 = next((i for i, name in enumerate(types) if name in {"L1C", "L1W", "L1"}), None)
    l2 = next((i for i, name in enumerate(types) if name in {"L2W", "L2X", "L2P", "L2"}), None)
    return l1, l2


def _parse_header(text: str) -> tuple[list[str], bool]:
    types: list[str] = []
    rinex3 = False
    for line in text.splitlines():
        label = line[60:80].strip() if len(line) >= 60 else ""
        if "RINEX VERSION / TYPE" in line:
            label = "RINEX VERSION / TYPE"
        elif "SYS / # / OBS TYPES" in line:
            label = "SYS / # / OBS TYPES"
        elif "# / TYPES OF OBSERV" in line:
            label = "# / TYPES OF OBSERV"
        elif "END OF HEADER" in line:
            label = "END OF HEADER"
        if label == "RINEX VERSION / TYPE":
            rinex3 = line.strip().startswith("3")
        if label == "SYS / # / OBS TYPES" and line.startswith("G"):
            body = line.split("SYS / # / OBS TYPES", 1)[0]
            parts = body.split()
            if len(parts) >= 2:
                types.extend(parts[2:])
        if label == "# / TYPES OF OBSERV":
            types.extend(line[:60].split()[1:])
        if label == "END OF HEADER":
            break
    return types, rinex3


def _read_fields(line: str, n_types: int) -> list[float | None]:
    body = line[3:] if line[:1] in {"G", "R", "E", "C", "J", "S"} and len(line) > 3 else line
    out: list[float | None] = []
    for i in range(n_types):
        chunk = body[i * 16 : (i * 16) + 14].strip()
        if not chunk:
            out.append(None)
            continue
        try:
            out.append(float(chunk))
        except ValueError:
            out.append(None)
    if n_types and all(v is None for v in out):
        nums = []
        for part in body.split():
            try:
                nums.append(float(part))
            except ValueError:
                continue
        if nums:
            out = nums[:n_types] + [None] * max(0, n_types - len(nums))
    return out


def _read_rinex2_record(lines: list[str], i: int, n_types: int) -> tuple[list[float | None], int]:
    """RINEX2: five 16-char obs per line, including blank slots; wrap for the rest."""
    fields: list[float | None] = []
    while len(fields) < n_types and i < len(lines):
        if EPOCH2_RE.match(lines[i]) or lines[i].startswith(">"):
            break
        line = lines[i]
        i += 1
        for k in range(5):
            if len(fields) >= n_types:
                break
            chunk = line[k * 16 : (k * 16) + 14].strip()
            if not chunk:
                fields.append(None)
                continue
            try:
                fields.append(float(chunk))
            except ValueError:
                fields.append(None)
    if len(fields) < n_types:
        fields.extend([None] * (n_types - len(fields)))
    return fields[:n_types], i


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, int(default))


def day_stride() -> int:
    """Mornings to keep. Prefer EQPRED_ROTI_DAY_STRIDE; old ROTI_STRIDE still works."""
    if os.environ.get("EQPRED_ROTI_DAY_STRIDE", "").strip():
        return _env_int("EQPRED_ROTI_DAY_STRIDE", 1)
    return _env_int("EQPRED_ROTI_STRIDE", 1)


def epoch_stride() -> int:
    """Keep one 30 s epoch in N, spread through the day. Default 4."""
    return _env_int("EQPRED_ROTI_EPOCH_STRIDE", 4)


def parse_rinex_roti(text: str, epoch_stride: int = 1) -> dict[str, float | None]:
    """One daily RINEX → median ROTI and cycle-slip rate.

    `epoch_stride` 4 keeps every 4th 30 s epoch (2 min samples). The file is
    still downloaded whole; this only thins the parse.
    """
    if not text or "END OF HEADER" not in text:
        return {"roti": None, "roti_slips": None, "roti_n": 0}
    types, rinex3 = _parse_header(text)
    l1_i, l2_i = _obs_index(types)
    if l1_i is None or l2_i is None:
        return {"roti": None, "roti_slips": None, "roti_n": 0}
    n_types = len(types)
    step = max(1, int(epoch_stride))
    max_dt_min = 3.0 * step
    prev: dict[str, tuple[float, float]] = {}
    tec_series: dict[str, list[tuple[float, float]]] = {}
    slips = 0
    n_obs = 0
    epoch: datetime | None = None
    epoch_i = 0
    lines = text.splitlines()
    i = 0
    while i < len(lines) and "END OF HEADER" not in lines[i]:
        i += 1
    i += 1
    while i < len(lines):
        line = lines[i]
        if rinex3 and line.startswith(">"):
            parts = line[1:].split()
            try:
                sec = float(parts[5]) if len(parts) > 5 else 0.0
                epoch = datetime(
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                    tzinfo=timezone.utc,
                ) + timedelta(seconds=sec)
                nsat = int(parts[7])
            except (IndexError, ValueError):
                i += 1
                continue
            i += 1
            keep = epoch_i % step == 0
            epoch_i += 1
            if not keep:
                i += nsat
                continue
            for _ in range(nsat):
                if i >= len(lines):
                    break
                sat_line = lines[i]
                i += 1
                if not sat_line.startswith("G"):
                    continue
                sat = sat_line[:3]
                fields = _read_fields(sat_line, n_types)
                _accumulate(sat, fields, l1_i, l2_i, epoch, prev, tec_series)
                n_obs += 1
                if sat in prev and fields[l1_i] is not None and fields[l2_i] is not None:
                    d_gf = abs((fields[l1_i] - fields[l2_i]) - (prev[sat][0] - prev[sat][1]))
                    if d_gf > SLIP_CYCLES:
                        slips += 1
                if fields[l1_i] is not None and fields[l2_i] is not None:
                    prev[sat] = (fields[l1_i], fields[l2_i])
            continue
        if (not rinex3) and EPOCH2_RE.match(line):
            try:
                yy = int(line[1:3])
                year = 2000 + yy if yy < 80 else 1900 + yy
                sec = float(line[15:26].strip() or 0.0)
                epoch = datetime(
                    year,
                    int(line[4:6]),
                    int(line[7:9]),
                    int(line[10:12]),
                    int(line[13:15]),
                    tzinfo=timezone.utc,
                ) + timedelta(seconds=sec)
                nsat = int(line[29:32])
            except ValueError:
                i += 1
                continue
            sat_ids = SAT2_RE.findall(line[32:])
            i += 1
            while len(sat_ids) < nsat and i < len(lines) and not EPOCH2_RE.match(lines[i]):
                extra = SAT2_RE.findall(lines[i])
                if not extra:
                    break
                sat_ids.extend(extra)
                i += 1
            sat_ids = sat_ids[:nsat]
            keep = epoch_i % step == 0
            epoch_i += 1
            if not keep:
                for _ in sat_ids:
                    if i >= len(lines):
                        break
                    _, i = _read_rinex2_record(lines, i, n_types)
                continue
            for sat in sat_ids:
                if i >= len(lines):
                    break
                fields, i = _read_rinex2_record(lines, i, n_types)
                if not sat.startswith("G"):
                    continue
                _accumulate(sat, fields, l1_i, l2_i, epoch, prev, tec_series)
                n_obs += 1
                if sat in prev and l1_i < len(fields) and l2_i < len(fields) and fields[l1_i] is not None and fields[l2_i] is not None:
                    d_gf = abs((fields[l1_i] - fields[l2_i]) - (prev[sat][0] - prev[sat][1]))
                    if d_gf > SLIP_CYCLES:
                        slips += 1
                if l1_i < len(fields) and l2_i < len(fields) and fields[l1_i] is not None and fields[l2_i] is not None:
                    prev[sat] = (fields[l1_i], fields[l2_i])
            continue
        i += 1
    roti_vals = [_window_roti(pairs, max_dt_min=max_dt_min) for pairs in tec_series.values()]
    roti_vals = [v for v in roti_vals if v is not None]
    return {
        "roti": float(sum(roti_vals) / len(roti_vals)) if roti_vals else None,
        "roti_slips": (slips / n_obs) if n_obs else None,
        "roti_n": n_obs,
    }


def _accumulate(
    sat: str,
    fields: list[float | None],
    l1_i: int,
    l2_i: int,
    epoch: datetime | None,
    prev: dict[str, tuple[float, float]],
    tec_series: dict[str, list[tuple[float, float]]],
) -> None:
    _ = prev
    if epoch is None or l1_i >= len(fields) or l2_i >= len(fields):
        return
    l1, l2 = fields[l1_i], fields[l2_i]
    if l1 is None or l2 is None:
        return
    tec = TEC_FACTOR * ((l2 * LAM2) - (l1 * LAM1))
    if not math.isfinite(tec):
        return
    tec_series.setdefault(sat, []).append((epoch.timestamp(), tec))


def _window_roti(pairs: list[tuple[float, float]], max_dt_min: float = 3.0) -> float | None:
    if len(pairs) < ROTI_WINDOW + 1:
        return None
    pairs = sorted(pairs)
    rots: list[float] = []
    gap = max(float(max_dt_min), 3.0)
    for i in range(1, len(pairs)):
        dt_min = (pairs[i][0] - pairs[i - 1][0]) / 60.0
        if dt_min <= 0 or dt_min > gap:
            continue
        rots.append((pairs[i][1] - pairs[i - 1][1]) / dt_min)
    if len(rots) < 4:
        return None
    windows = []
    for start in range(0, len(rots) - 4, 5):
        chunk = rots[start : start + 5]
        if len(chunk) < 4:
            continue
        mean = sum(chunk) / len(chunk)
        var = sum((x - mean) ** 2 for x in chunk) / len(chunk)
        windows.append(math.sqrt(var))
    if not windows:
        return None
    windows.sort()
    return windows[len(windows) // 2]


def _decode(data: bytes) -> str:
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data).decode("ascii", errors="replace")
    if data[:2] == b"\x1f\x9d":
        import unlzw3

        return unlzw3.unlzw(data).decode("ascii", errors="replace")
    return data.decode("ascii", errors="replace")


def _wanted_days(
    start: datetime,
    end: datetime,
    *,
    now: datetime | None = None,
    stride: int | None = None,
) -> list[datetime]:
    """Keep every Nth morning, plus the last 90 days relative to today.

    Year-chunk ingest used to treat *chunk* end as “today”, so each historical
    year densified its last 90 mornings. Dense tail is calendar-now only.
    """
    step = max(1, int(stride if stride is not None else day_stride()))
    clock = utc_ts(now) if now is not None else datetime.now(timezone.utc)
    recent = clock.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=90)
    days: list[datetime] = []
    day = start
    while day < end:
        if day >= recent or ((day - start).days % step == 0):
            days.append(day)
        day += timedelta(days=1)
    return days, step


def _station_day(station: str, day: datetime, fetch, epoch_step: int) -> dict[str, float | None]:
    parsed: dict[str, float | None] = {"roti": None, "roti_slips": None, "roti_n": 0}
    urls = (_rinex3_url(station, day), _rinex2_url(station, day))
    if day.year < 2020:
        urls = (_rinex2_url(station, day), _rinex3_url(station, day))
    for url in urls:
        try:
            data = fetch(url, timeout=40)
            parsed = parse_rinex_roti(_decode(data), epoch_stride=epoch_step)
            if parsed.get("roti_n"):
                break
        except Exception:
            continue
    return parsed


def _have_dates() -> set[str]:
    try:
        from eqpred.ingest.run import try_latest_raw

        frame = try_latest_raw(SOURCE)
    except Exception:
        return set()
    if frame is None or getattr(frame, "empty", True) or "observed_at" not in frame.columns:
        return set()
    obs = pd.to_datetime(frame["observed_at"], utc=True, format="ISO8601", errors="coerce")
    return {stamp.date().isoformat() for stamp in obs.dropna()}


def pull_roti(start: datetime, end: datetime, run_id: str, *, fetch=fetch_bytes) -> pd.DataFrame:
    _ = run_id
    start = utc_ts(start).replace(hour=0, minute=0, second=0, microsecond=0)
    end = utc_ts(end)
    rows: list[dict[str, object]] = []
    todo, day_step = _wanted_days(start, end)
    have = _have_dates()
    if have:
        todo = [day for day in todo if day.date().isoformat() not in have]
    epoch_step = epoch_stride()
    print(
        f"  geonet.roti {len(todo)} days x {len(STATIONS)} stations "
        f"(every morning, every {epoch_step}th 30s epoch, 4 stations parallel"
        f"{'' if day_step == 1 else f', day stride {day_step}'})",
        flush=True,
    )
    workers = min(len(STATIONS), 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, day in enumerate(todo):
            parsed_list = list(
                pool.map(lambda station, when=day: _station_day(station, when, fetch, epoch_step), STATIONS)
            )
            station_vals = [float(p["roti"]) for p in parsed_list if p.get("roti") is not None]
            slip_vals = [float(p["roti_slips"]) for p in parsed_list if p.get("roti_slips") is not None]
            n_total = sum(int(p.get("roti_n") or 0) for p in parsed_list)
            if station_vals or slip_vals:
                rows.append(
                    {
                        "source": SOURCE,
                        "observed_at": iso(day),
                        "available_at": iso(day + LATENCY),
                        "roti_nz": float(sum(station_vals) / len(station_vals)) if station_vals else None,
                        "roti_slips": float(sum(slip_vals) / len(slip_vals)) if slip_vals else None,
                        "roti_n": n_total,
                    }
                )
            if (i + 1) % 20 == 0:
                print(f"  geonet.roti {i + 1}/{len(todo)} {day.date()} {len(rows)} days", flush=True)
    return pd.DataFrame(rows)
