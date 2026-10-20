from __future__ import annotations

"""Daily Unwin radar numbers from FITACF files already on disk.

Two columns: the share of good gates that are ionospheric scatter, and the
mean absolute line-of-sight speed of those gates. A day with no file stays
blank. Later mornings do not keep the last reading.
"""

import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.paths import raw_dir

SOURCE = "superdarn.unwin"
COLUMNS = ("sdarn_iono_share", "sdarn_abs_v")
_TYPE_SIZE = {1: 1, 2: 2, 3: 4, 4: 4, 8: 8}
_WANTED = {b"qflg": 1, b"gflg": 1, b"v": 4}
_MAGIC = b"\x01\x00\x01\x00"


def _cstr(buf: bytes, pos: int) -> tuple[bytes, int]:
    end = buf.index(0, pos)
    return buf[pos:end], end + 1


def scan_fitacf(raw: bytes) -> dict[int, tuple[int, int, float, int]]:
    """Count quality gates, ionospheric gates, and |velocity| by UTC day."""
    out: dict[int, list] = {}
    offset = 0
    size_of = _TYPE_SIZE
    limit = len(raw)
    while offset + 16 <= limit:
        code, size, n_scalar, n_array = struct.unpack_from("<4i", raw, offset)
        if code != 65537 or size < 16:
            break
        claimed = offset + size
        if claimed == limit or (claimed + 4 <= limit and raw[claimed : claimed + 4] == _MAGIC):
            end = min(claimed, limit)
        else:
            nxt = raw.find(_MAGIC, offset + 4)
            if nxt <= offset:
                break
            end = nxt
        if end > limit:
            break
        pos = offset + 16
        year = month = day = 0
        qflg = gflg = vel = None
        try:
            for _ in range(n_scalar):
                if pos >= end:
                    raise ValueError("scalar past record")
                name, pos = _cstr(raw, pos)
                kind = raw[pos]
                pos += 1
                if kind == 9:
                    _, pos = _cstr(raw, pos)
                    continue
                width = size_of.get(kind)
                if width is None or pos + width > end:
                    raise ValueError("bad scalar")
                if name == b"time.yr":
                    year = struct.unpack_from("<h" if kind == 2 else "<i", raw, pos)[0]
                elif name == b"time.mo":
                    month = struct.unpack_from("<h" if kind == 2 else "<i", raw, pos)[0]
                elif name == b"time.dy":
                    day = struct.unpack_from("<h" if kind == 2 else "<i", raw, pos)[0]
                pos += width
            for _ in range(n_array):
                if pos >= end:
                    raise ValueError("array past record")
                name, pos = _cstr(raw, pos)
                kind = raw[pos]
                pos += 1
                dim = struct.unpack_from("<i", raw, pos)[0]
                pos += 4
                if dim < 0 or dim > 8 or pos + 4 * dim > end:
                    raise ValueError("bad array rank")
                dims = struct.unpack_from("<" + "i" * dim, raw, pos) if dim else ()
                pos += 4 * dim
                count = 1
                for n in dims:
                    count *= int(n)
                if count < 0 or count > 200_000:
                    raise ValueError("bad array length")
                width = 1 if kind == 9 else size_of.get(kind, 0)
                if pos + width * count > end:
                    raise ValueError("array runs past the record")
                if name in _WANTED and count and kind == _WANTED[name]:
                    if kind == 1:
                        blob = raw[pos : pos + count]
                    else:
                        blob = struct.unpack_from("<" + "f" * count, raw, pos)
                    if name == b"qflg":
                        qflg = blob
                    elif name == b"gflg":
                        gflg = blob
                    else:
                        vel = blob
                pos += width * count
            if pos != end:
                qflg = gflg = vel = None
        except (ValueError, struct.error, IndexError):
            qflg = gflg = vel = None
        offset = end
        if not year or qflg is None or gflg is None or len(qflg) != len(gflg):
            continue
        try:
            ordinal = datetime(int(year), int(month), int(day), tzinfo=timezone.utc).toordinal()
        except ValueError:
            continue
        bucket = out.get(ordinal)
        if bucket is None:
            bucket = [0, 0, 0.0, 0]
            out[ordinal] = bucket
        for i, flag in enumerate(qflg):
            if flag != 1:
                continue
            bucket[0] += 1
            if gflg[i] != 0:
                continue
            bucket[1] += 1
            if vel is not None and i < len(vel):
                speed = vel[i]
                if speed == speed:
                    bucket[2] += abs(float(speed))
                    bucket[3] += 1
    return {day: (vals[0], vals[1], vals[2], vals[3]) for day, vals in out.items()}


def _scan_path(path: str) -> dict[int, tuple[int, int, float, int]]:
    import bz2

    try:
        raw = bz2.BZ2File(path, "rb").read()
        return scan_fitacf(raw)
    except (OSError, ValueError, struct.error, IndexError, MemoryError):
        print(f"unwin skip {Path(path).name}", flush=True)
        return {}


def _fitacf_paths(folder: Path) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    if not folder.exists():
        return paths
    for path in sorted(folder.rglob("*.unw.fitacf.bz2")):
        if path.name in seen:
            continue
        seen.add(path.name)
        paths.append(path)
    return paths


def build_unwin_daily(folder: Path, *, workers: int = 4) -> pd.DataFrame:
    """Sum every Unwin file in folder into one row per UTC day."""
    paths = _fitacf_paths(folder)
    totals: dict[int, list] = defaultdict(lambda: [0, 0, 0.0, 0])
    if paths:
        jobs = [str(path) for path in paths]
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for part in pool.map(_scan_path, jobs, chunksize=8):
                for day, (qual, iono, vsum, nv) in part.items():
                    bucket = totals[day]
                    bucket[0] += qual
                    bucket[1] += iono
                    bucket[2] += vsum
                    bucket[3] += nv
    rows = []
    for ordinal in sorted(totals):
        qual, iono, vsum, nv = totals[ordinal]
        if qual <= 0:
            continue
        when = datetime.fromordinal(ordinal).replace(tzinfo=timezone.utc)
        rows.append(
            {
                "observed_at": when,
                "available_at": when + pd.Timedelta(days=1),
                "sdarn_iono_share": iono / qual,
                "sdarn_abs_v": (vsum / nv) if nv else np.nan,
                "sdarn_quality_gates": qual,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["observed_at", "available_at", *COLUMNS])
    return frame


def cusp_onset_rows(
    gates: pd.DataFrame,
    *,
    quiet_yesterday: np.ndarray | None = None,
    storm_today: np.ndarray | None = None,
) -> pd.DataFrame:
    """Unwin gates over New Zealand longitude, and only on a storm-onset morning.

    A daily mean of every gate is not this index. Onset means yesterday was
    quiet and today is not. Gates need a latitude and a longitude.
    """
    needed = {"observed_at", "glat", "glon", "power"}
    if gates.empty or not needed.issubset(gates.columns):
        return pd.DataFrame(columns=["observed_at", "sdarn_cusp_power"])
    work = gates.copy()
    work["glat"] = pd.to_numeric(work["glat"], errors="coerce")
    work["glon"] = pd.to_numeric(work["glon"], errors="coerce")
    work["power"] = pd.to_numeric(work["power"], errors="coerce")
    lon = work["glon"].where(work["glon"] <= 180, work["glon"] - 360)
    cusp = work["glat"].between(-80.0, -65.0) & lon.between(165.0, 180.0) & work["power"].notna()
    work = work.loc[cusp]
    if work.empty:
        return pd.DataFrame(columns=["observed_at", "sdarn_cusp_power"])
    day = pd.to_datetime(work["observed_at"], utc=True).dt.floor("D")
    day.name = "observed_at"
    daily = work.groupby(day, as_index=False)["power"].mean()
    daily = daily.rename(columns={"power": "sdarn_cusp_power"})
    if quiet_yesterday is None or storm_today is None:
        return daily
    onset = np.asarray(quiet_yesterday, dtype=bool) & np.asarray(storm_today, dtype=bool)
    if len(onset) != len(daily):
        return daily.iloc[0:0]
    return daily.loc[onset].reset_index(drop=True)


def unwin_daily_path(root: Path | None = None) -> Path:
    return raw_dir(root) / SOURCE / "daily.parquet"


def unwin_daily(root: Path | None = None) -> pd.DataFrame:
    path = unwin_daily_path(root)
    if not path.exists():
        return pd.DataFrame(columns=["observed_at", "available_at", *COLUMNS])
    frame = pd.read_parquet(path)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    if "available_at" in frame.columns:
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    return frame


def write_unwin_daily(folder: Path, root: Path | None = None, *, workers: int = 4) -> pd.DataFrame:
    frame = build_unwin_daily(folder, workers=workers)
    dest = unwin_daily_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    print(
        f"unwin daily {len(frame)} days"
        + (f" {frame['observed_at'].min().date()} to {frame['observed_at'].max().date()}" if len(frame) else ""),
        flush=True,
    )
    return frame
