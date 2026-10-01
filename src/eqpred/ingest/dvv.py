from __future__ import annotations

"""GeoNet daily dv/v from single-station ambient-noise stretch.

Night-hour HHZ windows, autocorrelation, then stretching versus a stacked
reference. Optional obspy decode; a compact miniSEED reader covers INT32
and STEIM2. Two-day waveform latency.
"""

import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from eqpred.ingest.http import fetch_bytes
from eqpred.paths import data_dir
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "geonet.dvv"
LICENSE = "https://www.geonet.org.nz/data/tools/FDSN"
LATENCY = timedelta(days=2)
DATASELECT = "https://service.geonet.org.nz/fdsnws/dataselect/1/query"
NIGHT_HOUR_UTC = 11
WINDOW_HOURS = 1
BAND = (0.5, 2.0)
TMIN = 2.0
TMAX = 20.0
EPS_MAX = 0.02
N_EPS = 81
REF_DAYS = 14

STATIONS = (
    ("NZ", "BFZ", "10", "HHZ", "hikurangi"),
    ("NZ", "PUZ", "10", "HHZ", "hikurangi"),
    ("NZ", "WVZ", "10", "HHZ", "alpine"),
    ("NZ", "JCZ", "10", "HHZ", "alpine"),
)


def stretch_dvv(
    current: np.ndarray,
    reference: np.ndarray,
    fs: float,
    *,
    eps_max: float = EPS_MAX,
    n_eps: int = N_EPS,
    tmin: float = TMIN,
    tmax: float = TMAX,
) -> tuple[float, float]:
    """Return (dv/v, peak CC). dv/v = -ε at the best stretch of current."""
    cur = np.asarray(current, dtype=float)
    ref = np.asarray(reference, dtype=float)
    n = min(len(cur), len(ref))
    if n < 8 or not np.isfinite(fs) or fs <= 0:
        return 0.0, 0.0
    cur = cur[:n]
    ref = ref[:n]
    t = np.arange(n, dtype=float) / float(fs)
    mask = (t >= tmin) & (t <= min(tmax, t[-1]))
    if int(mask.sum()) < 8:
        mask = np.ones(n, dtype=bool)
    t_win = t[mask]
    ref_win = ref[mask]
    ref_win = ref_win - ref_win.mean()
    ref_norm = float(np.sqrt(np.dot(ref_win, ref_win)))
    if ref_norm == 0:
        return 0.0, 0.0
    best_eps = 0.0
    best_cc = -2.0
    t_cur = np.arange(len(cur), dtype=float) / float(fs)
    for eps in np.linspace(-eps_max, eps_max, n_eps):
        query = t_win * (1.0 + float(eps))
        stretched = np.interp(query, t_cur, cur, left=0.0, right=0.0)
        stretched = stretched - stretched.mean()
        denom = float(np.sqrt(np.dot(stretched, stretched))) * ref_norm
        if denom == 0:
            continue
        cc = float(np.dot(stretched, ref_win) / denom)
        if cc > best_cc:
            best_cc = cc
            best_eps = float(eps)
    return -best_eps, best_cc


def bandpass(x: np.ndarray, fs: float, lo: float = BAND[0], hi: float = BAND[1]) -> np.ndarray:
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.99)
    if hi_n <= lo_n:
        return np.asarray(x, dtype=float)
    b, a = butter(2, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=float))


def autocorr(x: np.ndarray, fs: float, tmax: float = TMAX + 5.0) -> np.ndarray:
    series = np.asarray(x, dtype=float)
    series = series - series.mean()
    series = np.sign(series)
    n = min(len(series), max(int(tmax * fs) + 1, 8))
    full = np.correlate(series, series, mode="full")
    mid = len(series) - 1
    return full[mid : mid + n].astype(float)


def traces_to_dvv(windows: list[tuple[datetime, float, np.ndarray]]) -> pd.DataFrame:
    """Stack a reference from the first nights, then stretch each night."""
    if len(windows) < 3:
        return pd.DataFrame()
    autocorrs: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for when, fs, trace in windows:
        if trace is None or len(trace) < 16 or not np.isfinite(fs):
            continue
        ac = autocorr(bandpass(trace, fs), fs)
        autocorrs.append(ac)
        rows.append({"when": when, "fs": fs, "ac": ac})
    if len(autocorrs) < 3:
        return pd.DataFrame()
    ref_n = min(REF_DAYS, max(3, len(autocorrs) // 3))
    width = min(len(a) for a in autocorrs)
    stack = np.median(np.stack([a[:width] for a in autocorrs[:ref_n]]), axis=0)
    out: list[dict[str, object]] = []
    fs0 = float(rows[0]["fs"])
    for rec in rows:
        dvv, cc = stretch_dvv(np.asarray(rec["ac"])[:width], stack, fs0)
        observed = rec["when"]
        if not isinstance(observed, datetime):
            observed = utc_ts(observed)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        out.append(
            {
                "observed_at": iso(observed.replace(hour=0, minute=0, second=0, microsecond=0)),
                "available_at": iso(observed.replace(hour=0, minute=0, second=0, microsecond=0) + LATENCY),
                "dvv": float(dvv),
                "dvv_cc": float(cc),
            }
        )
    return pd.DataFrame(out)


def _mseed_time(raw: bytes) -> datetime | None:
    if len(raw) < 10:
        return None
    year, doy, hour, minute, sec = struct.unpack(">HHBBB", raw[:7])
    fract = struct.unpack(">H", raw[8:10])[0]
    try:
        start = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        return start.replace(hour=hour, minute=minute, second=sec, microsecond=int(fract * 100))
    except ValueError:
        return None


def _sample_rate(factor: int, mult: int) -> float:
    if factor == 0 or mult == 0:
        return 0.0
    if factor > 0 and mult > 0:
        return float(factor * mult)
    if factor > 0:
        return float(factor) / float(-mult)
    if mult > 0:
        return float(mult) / float(-factor)
    return 1.0 / float(factor * mult)


def _decode_steim2(payload: bytes, nsamp: int) -> np.ndarray:
    samples = np.zeros(max(nsamp, 0), dtype=np.int64)
    if nsamp <= 0 or len(payload) < 64:
        return samples
    n = 0
    x0 = xn = 0
    first = True
    for frame_off in range(0, len(payload) - 63, 64):
        words = struct.unpack(">" + "i" * 16, payload[frame_off : frame_off + 64])
        nibbles = words[0] & 0xFFFFFFFF
        if first:
            x0 = words[1]
            xn = words[2]
            first = False
        for w in range(1, 16):
            code = (nibbles >> (30 - 2 * w)) & 0x3
            word = words[w]
            diffs: list[int] = []
            if code == 1:
                diffs = [((word >> shift) & 0xFF) - 256 if ((word >> shift) & 0x80) else ((word >> shift) & 0xFF) for shift in (24, 16, 8, 0)]
            elif code == 2:
                diffs = [((word >> shift) & 0xFFFF) - 65536 if ((word >> shift) & 0x8000) else ((word >> shift) & 0xFFFF) for shift in (16, 0)]
            elif code == 3:
                dnib = (word >> 30) & 0x3
                if dnib == 1:
                    val = word & 0x3FFFFFFF
                    if val & 0x20000000:
                        val -= 0x40000000
                    diffs = [val]
                elif dnib == 2:
                    diffs = []
                    for shift in (15, 0):
                        val = (word >> shift) & 0x7FFF
                        if val & 0x4000:
                            val -= 0x8000
                        diffs.append(val)
                elif dnib == 3:
                    diffs = []
                    for shift in (20, 10, 0):
                        val = (word >> shift) & 0x3FF
                        if val & 0x200:
                            val -= 0x400
                        diffs.append(val)
            for diff in diffs:
                if n == 0:
                    samples[0] = x0
                    n = 1
                    continue
                if n >= nsamp:
                    return samples[:nsamp]
                samples[n] = int(samples[n - 1]) + int(diff)
                n += 1
    if n and xn:
        samples[min(n, nsamp) - 1] = xn
    return samples[:nsamp]


def _decode_int32(payload: bytes, nsamp: int) -> np.ndarray:
    need = nsamp * 4
    raw = payload[:need]
    if len(raw) < 4:
        return np.zeros(0, dtype=np.int32)
    n = len(raw) // 4
    return np.frombuffer(raw, dtype=">i4", count=n).astype(np.int32)


def read_mseed_traces(blob: bytes) -> list[tuple[datetime, float, np.ndarray]]:
    traces: list[tuple[datetime, float, np.ndarray]] = []
    off = 0
    while off + 48 <= len(blob):
        rec_len = 512
        rec = blob[off : off + rec_len]
        if len(rec) < 48:
            break
        start = _mseed_time(rec[20:30])
        nsamp = struct.unpack(">H", rec[30:32])[0]
        factor, mult = struct.unpack(">hh", rec[32:36])
        data_off = struct.unpack(">H", rec[44:46])[0]
        encoding = 11
        if data_off >= 56 and len(rec) >= data_off:
            blk = rec[48:56]
            if len(blk) >= 8 and blk[0:2] == b"\x03\xe8":
                encoding = blk[4]
                rec_exp = blk[6]
                rec_len = 2 ** rec_exp if 7 <= rec_exp <= 12 else 512
                rec = blob[off : off + rec_len]
        payload = rec[data_off:] if data_off < len(rec) else b""
        fs = _sample_rate(factor, mult)
        if encoding == 3:
            samples = _decode_int32(payload, nsamp)
        elif encoding in {10, 11}:
            samples = _decode_steim2(payload, nsamp)
        else:
            samples = np.zeros(0, dtype=np.int32)
        if start is not None and len(samples) and fs > 0:
            traces.append((start, fs, samples.astype(float)))
        off += rec_len if rec_len > 0 else 512
    return traces


def write_mseed_int32(start: datetime, fs: float, samples: np.ndarray) -> bytes:
    """Test helper: one 512-byte INT32 record (truncated if needed)."""
    data = np.asarray(samples, dtype=">i4")
    nsamp = min(len(data), 112)
    data = data[:nsamp]
    rec = bytearray(512)
    rec[0:6] = b"000001"
    rec[6:7] = b"D"
    rec[8:13] = b"BFZ  "
    rec[13:15] = b"10"
    rec[15:18] = b"HHZ"
    rec[18:20] = b"NZ"
    rec[20:22] = struct.pack(">H", start.year)
    rec[22:24] = struct.pack(">H", start.timetuple().tm_yday)
    rec[24] = start.hour
    rec[25] = start.minute
    rec[26] = start.second
    rec[30:32] = struct.pack(">H", nsamp)
    rec[32:34] = struct.pack(">h", int(fs) if fs == int(fs) else 1)
    rec[34:36] = struct.pack(">h", 1 if fs == int(fs) else int(round(fs)))
    rec[39] = 1
    rec[44:46] = struct.pack(">H", 56)
    rec[48:50] = struct.pack(">H", 1000)
    rec[50:52] = struct.pack(">H", 8)
    rec[52] = 3
    rec[53] = 1
    rec[54] = 9
    rec[56 : 56 + nsamp * 4] = data.tobytes()
    return bytes(rec)


def _cache_dir(root: Path | None) -> Path:
    dest = data_dir(root) / "cache" / "dvv"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _fetch_window(net: str, sta: str, loc: str, cha: str, day: datetime, fetch=fetch_bytes) -> bytes:
    start = day.replace(hour=NIGHT_HOUR_UTC, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=WINDOW_HOURS)
    loc_q = loc if loc.strip() else "--"
    url = (
        f"{DATASELECT}?network={net}&station={sta}&location={loc_q}&channel={cha}"
        f"&starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&endtime={end.strftime('%Y-%m-%dT%H:%M:%S')}&nodata=404"
    )
    return fetch(url)


def _decode_bytes(blob: bytes) -> list[tuple[datetime, float, np.ndarray]]:
    try:
        import obspy  # type: ignore

        stream = obspy.read(blob)
        out: list[tuple[datetime, float, np.ndarray]] = []
        for tr in stream:
            start = tr.stats.starttime.datetime.replace(tzinfo=timezone.utc)
            out.append((start, float(tr.stats.sampling_rate), np.asarray(tr.data, dtype=float)))
        if out:
            return out
    except Exception:
        pass
    return read_mseed_traces(blob)


def pull_dvv(
    start: datetime,
    end: datetime,
    run_id: str,
    *,
    root: Path | None = None,
    fetch=fetch_bytes,
) -> pd.DataFrame:
    _ = run_id
    start = utc_ts(start)
    end = utc_ts(end)
    days = pd.date_range(start.date(), (end - timedelta(days=1)).date(), freq="D", tz="UTC")
    cache = _cache_dir(root)
    by_box: dict[str, list[pd.DataFrame]] = {"hikurangi": [], "alpine": []}
    for net, sta, loc, cha, box in STATIONS:
        windows: list[tuple[datetime, float, np.ndarray]] = []
        for day in days:
            key = cache / f"{net}.{sta}.{cha}.{day.strftime('%Y%m%d')}.npy"
            if key.exists():
                packed = np.load(key, allow_pickle=True).item()
                windows.append((utc_ts(packed["when"]), float(packed["fs"]), np.asarray(packed["trace"], dtype=float)))
                continue
            try:
                blob = _fetch_window(net, sta, loc, cha, day.to_pydatetime(), fetch=fetch)
            except Exception:
                continue
            traces = _decode_bytes(blob)
            if not traces:
                continue
            when, fs, trace = traces[0]
            np.save(key, {"when": iso(when), "fs": fs, "trace": trace}, allow_pickle=True)
            windows.append((when, fs, trace))
            if len(windows) % 30 == 0:
                print(f"  geonet.dvv {sta} {day.date()} {len(windows)} traces", flush=True)
        daily = traces_to_dvv(windows)
        if daily.empty:
            continue
        daily[f"dvv_{box}"] = daily["dvv"]
        daily[f"dvv_{box}_cc"] = daily["dvv_cc"]
        by_box[box].append(daily)
    frames: list[pd.DataFrame] = []
    for box, parts in by_box.items():
        if not parts:
            continue
        stacked = pd.concat(parts, ignore_index=True)
        mean = (
            stacked.groupby("observed_at", as_index=False)
            .agg(dvv=( "dvv", "median"), dvv_cc=("dvv_cc", "median"), available_at=("available_at", "min"))
        )
        mean[f"dvv_{box}"] = mean["dvv"]
        frames.append(mean[["observed_at", "available_at", f"dvv_{box}"]])
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=["observed_at", "available_at"], how="outer")
    value_cols = [c for c in out.columns if c.startswith("dvv_")]
    if value_cols:
        out["dvv_nz"] = out[value_cols].median(axis=1)
    return out.sort_values("observed_at").reset_index(drop=True)
