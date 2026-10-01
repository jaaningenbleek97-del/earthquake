"""One midday hour per day from the Sierra Nevada ELF archive.

The public years are about 26 GB of raw waveforms each. This keeps a single
north-south hour per day, turns it into a 7.8 Hz band ratio, and does not
store the hour. The archive was published on 12 March 2022, so a morning in
2013–2017 could not see that day's number.
"""

from __future__ import annotations

import re
import struct
import threading
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.paths import raw_dir
from eqpred.records import iso

SOURCE = "schumann.elf"
LICENSE = "https://doi.org/10.5281/zenodo.6348690"
PUBLIC_AT = datetime(2022, 3, 12, tzinfo=timezone.utc)
FS_HZ = 1_000_000 / 3906.0
BYTE_CAP = 3_200_000_000
HOUR_RE = re.compile(r"smplGRTU1_sensor_0_(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$")
ZIPS = (
    "https://zenodo.org/api/records/6348930/files/2013_2017.zip/content",
    "https://zenodo.org/api/records/6348691/files/2014.zip/content",
    "https://zenodo.org/api/records/6348773/files/2015.zip/content",
    "https://zenodo.org/api/records/6348838/files/2016.zip/content",
)
UA = {"User-Agent": "earthquake-lab research"}


def schumann_mode1_ratio(samples: np.ndarray, fs: float = FS_HZ) -> float:
    """Power near 7.8 Hz divided by the power just below it."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size < int(fs * 60):
        return float("nan")
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    low = (freqs >= 6.0) & (freqs < 7.0)
    band = (freqs >= 7.2) & (freqs < 8.5)
    base = float(np.mean(power[low])) if low.any() else 0.0
    if base <= 0.0 or not band.any():
        return float("nan")
    return float(np.mean(power[band]) / base)


def _get(url: str, start: int, end: int) -> bytes:
    last = "no attempt"
    for _ in range(4):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as response:
                data = response.read()
            if len(data) == (end - start + 1):
                return data
            last = f"short {len(data)}"
        except Exception as exc:  # noqa: BLE001 — retry a flaky range read
            last = str(exc)
    raise RuntimeError(last)


def _zip_directory(url: str) -> bytes:
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=-65536"})
    with urllib.request.urlopen(req, timeout=120) as response:
        tail = response.read()
        span = response.headers.get("Content-Range") or ""
    tail_start = int(span.split(" ")[1].split("-")[0]) if "bytes" in span else 0
    marker = tail.rfind(b"PK\x06\x06")
    if marker >= 0:
        fields = struct.unpack_from("<QHHIIQQQQ", tail, marker + 4)
        cd_size = int(fields[7])
        cd_off = int(fields[8])
    else:
        marker = tail.rfind(b"PK\x05\x06")
        if marker < 0:
            raise RuntimeError(f"no zip directory in {url}")
        cd_size, cd_off = struct.unpack_from("<II", tail, marker + 12)
    return _get(url, cd_off, cd_off + cd_size - 1)


def _entries(cd: bytes):
    pos = 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        method = struct.unpack_from("<H", cd, pos + 10)[0]
        comp = struct.unpack_from("<I", cd, pos + 20)[0]
        uncomp = struct.unpack_from("<I", cd, pos + 24)[0]
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", cd, pos + 28)
        local = struct.unpack_from("<I", cd, pos + 42)[0]
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        ep = 0
        while ep + 4 <= len(extra):
            hid, hsz = struct.unpack_from("<HH", extra, ep)
            body = extra[ep + 4 : ep + 4 + hsz]
            if hid == 1:
                bp = 0
                if uncomp == 0xFFFFFFFF:
                    uncomp = struct.unpack_from("<Q", body, bp)[0]
                    bp += 8
                if comp == 0xFFFFFFFF:
                    comp = struct.unpack_from("<Q", body, bp)[0]
                    bp += 8
                if local == 0xFFFFFFFF and bp + 8 <= len(body):
                    local = struct.unpack_from("<Q", body, bp)[0]
            ep += 4 + hsz
        yield name, method, comp, local
        pos += 46 + name_len + extra_len + comment_len


def _chosen_hours(url: str) -> list[tuple[str, int, int, int]]:
    """One sensor-0 hour closest to 12:00 UTC for each day in this zip."""
    best: dict[str, tuple[int, str, int, int, int]] = {}
    for name, method, comp, local in _entries(_zip_directory(url)):
        match = HOUR_RE.search(name.split("/")[-1])
        if not match:
            continue
        yy, mo, dy, hh, mm = (int(part) for part in match.groups())
        if not (1 <= mo <= 12 and 1 <= dy <= 31):
            continue
        day = f"20{yy:02d}-{mo:02d}-{dy:02d}"
        distance = abs((hh * 60 + mm) - 12 * 60)
        current = best.get(day)
        if current is None or distance < current[0]:
            best[day] = (distance, name, method, comp, local)
    return [(name, method, comp, local) for _distance, name, method, comp, local in best.values()]


def _read_hour(url: str, method: int, comp: int, local: int) -> np.ndarray:
    head = _get(url, local, local + 29)
    name_len, extra_len = struct.unpack_from("<HH", head, 26)
    data_off = local + 30 + name_len + extra_len
    blob = _get(url, data_off, data_off + comp - 1)
    if method == 8:
        blob = zlib.decompress(blob, -15)
    elif method != 0:
        raise RuntimeError(f"zip method {method}")
    if len(blob) % 2:
        blob = blob[: len(blob) - 1]
    return np.frombuffer(blob, dtype="<i2")


def pull_midday_sample(root: Path | None = None, workers: int = 3) -> pd.DataFrame:
    """Download within BYTE_CAP and save the daily ratio table."""
    chosen = []
    for url in ZIPS:
        for name, method, comp, local in _chosen_hours(url):
            chosen.append((url, name, method, comp, local))
    chosen.sort(key=lambda item: item[1])
    print(f"schumann hours {len(chosen)} compressed {sum(item[3] for item in chosen)/1e9:.2f} GB", flush=True)
    spent = 0
    lock = threading.Lock()
    rows = []

    def one(item):
        nonlocal spent
        url, name, method, comp, local = item
        with lock:
            if spent + comp > BYTE_CAP:
                return None
            spent += comp
        match = HOUR_RE.search(name.split("/")[-1])
        yy, mo, dy, hh, mm = (int(part) for part in match.groups())
        try:
            samples = _read_hour(url, method, comp, local)
            ratio = schumann_mode1_ratio(samples)
        except Exception as exc:  # noqa: BLE001 — skip one bad hour and keep the day list
            print(f"schumann skip {name.split('/')[-1]} {exc}", flush=True)
            return None
        return {
            "observed_at": datetime(2000 + yy, mo, dy, tzinfo=timezone.utc),
            "schumann_mode1": ratio,
            "schumann_hour_utc": hh + mm / 60.0,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in chosen]
        done = 0
        for future in as_completed(futures):
            done += 1
            row = future.result()
            if row is not None:
                rows.append(row)
            if done % 25 == 0 or done == len(futures):
                print(f"schumann {done}/{len(futures)} kept {len(rows)}", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["available_at"] = iso(PUBLIC_AT)
    frame["source"] = SOURCE
    frame["license"] = LICENSE
    frame = frame.sort_values("observed_at").drop_duplicates("observed_at", keep="last")
    dest = raw_dir(root) / SOURCE
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest / "daily.parquet", index=False)
    return frame.reset_index(drop=True)


def schumann_daily(root: Path | None = None) -> pd.DataFrame:
    path = raw_dir(root) / SOURCE / "daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
