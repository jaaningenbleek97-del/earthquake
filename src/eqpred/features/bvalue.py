"""Point-in-time Gutenberg–Richter b on declustered events.

Global and per-box windows. A falling b (more large events relative to small
ones) is the evolution the model is allowed to see. Raw-catalog b_90d stays
on the nowcast track and is blocked from issued p; these b_fresh_* columns
are fresh_catalog and may enter the issued model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.models.gr import fit_b_value
from eqpred.regions import REGIONS, in_bbox
from eqpred.stamp import parse_utc

MIN_N = 30
WINDOWS = (90, 365)
BOXES = ("hikurangi", "alpine", "kermadec", "japan", "chile")


def bvalue_frame(
    quakes: pd.DataFrame,
    issue_times: pd.Series,
    *,
    m0: float = 4.5,
    boxes: tuple[str, ...] = BOXES,
    min_n: int = MIN_N,
) -> pd.DataFrame:
    """Daily b on events already reported by each issue morning."""
    issue = pd.DatetimeIndex(parse_utc(issue_times))
    cols = {
        "issue_time": issue,
        "b_fresh_90d": np.nan,
        "b_fresh_365d": np.nan,
        "b_fresh_delta": np.nan,
        "n_fresh_m45_90d": 0,
        "n_fresh_m45_365d": 0,
    }
    for rid in boxes:
        cols[f"b_fresh_{rid}_90d"] = np.nan
        cols[f"b_fresh_{rid}_365d"] = np.nan
        cols[f"b_fresh_{rid}_delta"] = np.nan
    empty = pd.DataFrame(cols)
    if quakes is None or quakes.empty or "magnitude" not in quakes.columns:
        return empty

    work = quakes.loc[pd.to_numeric(quakes["magnitude"], errors="coerce") >= m0].copy()
    if work.empty:
        return empty
    if "available_at" not in work.columns:
        work["available_at"] = work["observed_at"]
    work["observed_at"] = parse_utc(work["observed_at"])
    work["available_at"] = parse_utc(work["available_at"])
    work = work.dropna(subset=["observed_at", "available_at", "magnitude"])
    if work.empty:
        return empty
    work = work.sort_values("observed_at")
    obs = work["observed_at"].to_numpy(dtype="datetime64[ns]")
    avail = work["available_at"].to_numpy(dtype="datetime64[ns]")
    mag = pd.to_numeric(work["magnitude"], errors="coerce").to_numpy(dtype=float)
    t = issue.to_numpy(dtype="datetime64[ns]")

    b90, n90 = _window_b(obs, avail, mag, t, 90, m0, min_n)
    b365, n365 = _window_b(obs, avail, mag, t, 365, m0, min_n)
    out = empty.copy()
    out["b_fresh_90d"] = b90
    out["b_fresh_365d"] = b365
    out["b_fresh_delta"] = b90 - b365
    out["n_fresh_m45_90d"] = n90
    out["n_fresh_m45_365d"] = n365

    for rid in boxes:
        region = REGIONS.get(rid)
        if not region or region.get("kind") != "bbox":
            continue
        if "lat" not in work.columns or "lon" not in work.columns:
            continue
        mask = in_bbox(work, region).to_numpy()
        if not mask.any():
            continue
        box_b90, _ = _window_b(obs[mask], avail[mask], mag[mask], t, 90, m0, min_n)
        box_b365, _ = _window_b(obs[mask], avail[mask], mag[mask], t, 365, m0, min_n)
        out[f"b_fresh_{rid}_90d"] = box_b90
        out[f"b_fresh_{rid}_365d"] = box_b365
        out[f"b_fresh_{rid}_delta"] = box_b90 - box_b365
    return out


def _window_b(
    obs: np.ndarray,
    avail: np.ndarray,
    mag: np.ndarray,
    issue: np.ndarray,
    window_days: int,
    m0: float,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(issue)
    b = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    if obs.size == 0:
        return b, counts
    w = np.timedelta64(int(window_days), "D")
    left = np.searchsorted(obs, issue - w, side="left")
    right = np.searchsorted(obs, issue, side="right")
    for i in range(n):
        lo, hi = int(left[i]), int(right[i])
        if hi <= lo:
            continue
        ok = avail[lo:hi] <= issue[i]
        mags = mag[lo:hi][ok]
        counts[i] = int(mags.size)
        if mags.size >= min_n:
            b[i] = fit_b_value(mags, m0)
    return b, counts
