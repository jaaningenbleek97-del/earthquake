"""Dated forecasts from every replay, backtest, and model try.

Each attempt is one run. The file under it is one row per morning, place, and
horizon. These rows are rehearsal data. They are not the live exam and they
do not change a frozen card.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from eqpred.paths import data_dir
from eqpred.safe_io import file_lock

FORECAST_COLUMNS = (
    "attempt_id",
    "issue_day",
    "region",
    "label",
    "horizon_days",
    "magnitude_min",
    "p",
    "climatology_p",
    "happened",
    "closed",
    "slice",
)


def attempts_dir(root: Path | None = None) -> Path:
    path = data_dir(root) / "eval" / "attempts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path(root: Path | None) -> Path:
    return attempts_dir(root) / "index.jsonl"


def _slice_name(day: pd.Timestamp, horizon: int, catalog_end: pd.Timestamp) -> str:
    if day + pd.Timedelta(days=int(horizon)) > catalog_end:
        return "open"
    if day >= pd.Timestamp("2009-01-01", tz="UTC"):
        return "later"
    if day >= pd.Timestamp("1990-01-01", tz="UTC") and day + pd.Timedelta(days=int(horizon)) <= pd.Timestamp(
        "2009-01-01", tz="UTC"
    ):
        return "develop"
    return "early"


def forecast_frame(
    *,
    days: pd.DatetimeIndex,
    region: str,
    label: str,
    horizon: int,
    magnitude: float,
    p: np.ndarray,
    climatology: np.ndarray,
    happened: np.ndarray | None,
    mask: np.ndarray,
    catalog_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per selected morning. An open window has no happened value."""
    days = pd.DatetimeIndex(days)
    if len(days) == 0:
        return pd.DataFrame(columns=list(FORECAST_COLUMNS))
    end = pd.Timestamp(catalog_end) if catalog_end is not None else days[-1]
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    keep = np.asarray(mask, dtype=bool) & np.isfinite(p) & np.isfinite(climatology)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return pd.DataFrame(columns=list(FORECAST_COLUMNS))
    issue = days[idx]
    closed = np.asarray(issue + pd.Timedelta(days=int(horizon)) <= end)
    outcome = np.full(idx.size, np.nan)
    if happened is not None:
        y = np.asarray(happened, dtype=float)[idx]
        outcome[closed] = (y[closed] > 0).astype(float)
    return pd.DataFrame(
        {
            "issue_day": [stamp.date().isoformat() for stamp in issue],
            "region": region,
            "label": label,
            "horizon_days": int(horizon),
            "magnitude_min": float(magnitude),
            "p": np.asarray(p, dtype=float)[idx],
            "climatology_p": np.asarray(climatology, dtype=float)[idx],
            "happened": outcome,
            "closed": closed,
            "slice": [_slice_name(stamp, int(horizon), end) for stamp in issue],
        }
    )


def save_forecasts(
    frame: pd.DataFrame,
    *,
    source: str,
    name: str,
    root: Path | None = None,
    label: str | None = None,
    label_version: int | None = None,
    tape: str | None = None,
    knobs: dict[str, Any] | None = None,
    answer_key: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Append one attempt. A second run of the same rule is a new attempt."""
    if frame is None or len(frame) == 0:
        return {"n_forecasts": 0, "attempt_id": None, "writes_issued_p": False}
    tried = datetime.now(timezone.utc)
    attempt_id = tried.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    stored = frame.copy()
    stored.insert(0, "attempt_id", attempt_id)
    folder = attempts_dir(root)
    dest = folder / f"{attempt_id}.parquet"
    stored.to_parquet(dest, index=False)
    horizons = sorted({int(v) for v in stored["horizon_days"].unique()})
    regions = sorted({str(v) for v in stored["region"].unique()})
    entry = {
        "attempt_id": attempt_id,
        "tried_at": tried.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "name": name,
        "label": label or name,
        "n_forecasts": int(len(stored)),
        "regions": regions,
        "horizons": horizons,
        "label_version": label_version,
        "tape": tape,
        "answer_key": answer_key,
        "knobs": knobs or {},
        "file": dest.name,
        "writes_issued_p": False,
        "kind": "rehearsal",
        "note": note
        or "Rehearsal forecasts kept for later scoring. They do not change the live exam.",
    }
    index = _index_path(root)
    with file_lock(index):
        with index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    return entry


def list_attempts(root: Path | None = None, limit: int = 15) -> dict[str, Any]:
    """Newest attempts, plus the running total."""
    index = _index_path(root)
    rows: list[dict[str, Any]] = []
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n_forecasts = sum(int(row.get("n_forecasts") or 0) for row in rows)
    recent = list(reversed(rows[-max(int(limit), 0) :]))
    return {
        "n_attempts": len(rows),
        "n_forecasts": n_forecasts,
        "recent": recent,
        "writes_issued_p": False,
        "note": (
            f"{len(rows)} attempts, {n_forecasts} dated forecasts. "
            "Each run is kept. These are rehearsal rows for later scoring. They do not change the live exam."
        ),
    }
