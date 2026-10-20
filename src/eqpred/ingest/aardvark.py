"""NWC–Dunedin AARDDVARK, if a file has been placed on disk.

Rodger and Clilverd were already asked. This does not write to them and it
does not download. The public Zenodo series stays the northern-path foil.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from eqpred.paths import raw_dir

SOURCE = "vlf.aardvark"


def parse_aardvark(text: str) -> pd.DataFrame:
    """Amplitude and phase. A table without both is not this path."""
    if not text or "amp" not in text.lower():
        return pd.DataFrame()
    raw = pd.read_csv(StringIO(text), comment="#")
    if raw.empty:
        return pd.DataFrame()
    columns = {str(col).strip().lower(): col for col in raw.columns}
    time_name = next((columns[key] for key in columns if "time" in key or "date" in key), None)
    amp_name = next((columns[key] for key in columns if "amp" in key), None)
    phase_name = next((columns[key] for key in columns if "phase" in key or key == "pha"), None)
    if time_name is None or amp_name is None or phase_name is None:
        return pd.DataFrame()
    stamp = pd.to_datetime(raw[time_name], utc=True, errors="coerce")
    work = pd.DataFrame(
        {
            "observed_at": stamp,
            "aardvark_nwc_amp": pd.to_numeric(raw[amp_name], errors="coerce"),
            "aardvark_nwc_phase": pd.to_numeric(raw[phase_name], errors="coerce"),
        }
    ).dropna()
    if work.empty:
        return pd.DataFrame()
    work["observed_at"] = work["observed_at"].dt.floor("D")
    daily = work.groupby("observed_at", as_index=False)[["aardvark_nwc_amp", "aardvark_nwc_phase"]].median()
    return daily


def read_local_aardvark(root: Path | None = None) -> pd.DataFrame:
    """Files already in data/raw/vlf.aardvark. An empty folder stays empty."""
    folder = raw_dir(root) / SOURCE
    if not folder.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(folder.glob("*.csv")):
        frame = parse_aardvark(path.read_text(encoding="utf-8", errors="replace"))
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("observed_at")
