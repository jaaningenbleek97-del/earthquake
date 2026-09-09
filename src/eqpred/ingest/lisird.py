from __future__ import annotations

"""Solar irradiance leftovers from LASP LISIRD.

TSI, Mg II, and Lyman-alpha — solar / EUV conditions Davidson and Burns
watch, not another OMNI Kp transform.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

LICENSE = "https://lasp.colorado.edu/lisird/"
LATENCY = timedelta(days=4)
SOURCE = "lasp.lisird"

SERIES = (
    ("solar_tsi_wm2", "https://lasp.colorado.edu/lisird/latis/dap/nrl2_tsi_P1D.csv"),
    ("solar_mgii", "https://lasp.colorado.edu/lisird/latis/dap/composite_mg_index.csv"),
    ("solar_lya", "https://lasp.colorado.edu/lisird/latis/dap/composite_lyman_alpha.csv"),
)


def parse_lisird_csv(text: str, column: str) -> pd.DataFrame:
    """LISIRD latis CSV: time plus one or more value columns."""
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        if parts[0].lower().startswith(("time", "date")) or not parts[0][:1].isdigit():
            continue
        try:
            token = parts[0].replace("Z", "+00:00")
            if "T" in token or (len(token) >= 10 and token[4] == "-"):
                stamp = datetime.fromisoformat(token[:19] if "T" in token else token[:10])
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                observed = stamp.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                observed = datetime(1610, 1, 1, tzinfo=timezone.utc) + timedelta(days=float(token))
                observed = observed.replace(hour=0, minute=0, second=0, microsecond=0)
            value = float(parts[1])
        except (ValueError, OverflowError):
            continue
        if not (0.0 < value < 2000.0):
            continue
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                column: value,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("observed_at").sort_values("observed_at").reset_index(drop=True)


def pull_lisird(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column, url in SERIES:
        try:
            text = fetch_text(url, timeout=120)
        except Exception as exc:  # noqa: BLE001 — keep the other series
            print(f"  lasp.lisird skip {column}: {exc}")
            continue
        frame = parse_lisird_csv(text, column)
        if frame.empty:
            print(f"  lasp.lisird empty {column}")
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        cols = [c for c in extra.columns if str(c).startswith("solar_")]
        out = out.merge(extra[["observed_at", *cols]], on="observed_at", how="outer")
    out["source"] = SOURCE
    out["license"] = LICENSE
    out["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(out["observed_at"], utc=True)
    out["available_at"] = (stamp + LATENCY).map(iso)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return out.loc[mask].sort_values("observed_at").reset_index(drop=True)
