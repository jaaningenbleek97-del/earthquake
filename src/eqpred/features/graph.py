from __future__ import annotations

"""Fault-network stress. Segments are nodes; neighbors push energy across edges."""

import numpy as np
import pandas as pd

from eqpred.faults import GRAPH_EDGES, SEGMENTS, segment_region
from eqpred.regions import in_bbox
from eqpred.stamp import parse_utc

MINMAG = 4.5
WINDOW_DAYS = 7


def _moment(mag: pd.Series) -> np.ndarray:
    return np.power(10.0, 1.5 * (pd.to_numeric(mag, errors="coerce").to_numpy(dtype=float) - 4.5))


def _window_sum(stamps: np.ndarray, values: np.ndarray, issue: np.ndarray, days: int) -> np.ndarray:
    if stamps.size == 0:
        return np.zeros(len(issue), dtype=float)
    order = np.argsort(stamps)
    times = stamps[order]
    energy = np.nan_to_num(values[order], nan=0.0)
    cum = np.concatenate([[0.0], np.cumsum(energy)])
    right = np.searchsorted(times, issue, side="right")
    left = np.searchsorted(times, issue - np.timedelta64(int(days), "D"), side="right")
    return cum[right] - cum[left]


def segment_energy(
    quakes: pd.DataFrame,
    issue: pd.Series,
    segment_id: str,
    minmag: float = MINMAG,
    days: int = WINDOW_DAYS,
) -> np.ndarray:
    """Point-in-time seismic energy inside one segment over the last `days`."""
    if quakes.empty or "lat" not in quakes.columns:
        return np.zeros(len(issue), dtype=float)
    work = quakes.copy()
    work["available_at"] = parse_utc(work["available_at"] if "available_at" in work.columns else work["observed_at"])
    work["magnitude"] = pd.to_numeric(work["magnitude"], errors="coerce")
    work = work.loc[work["magnitude"] >= minmag]
    work = work.loc[in_bbox(work, segment_region(segment_id))]
    if work.empty:
        return np.zeros(len(issue), dtype=float)
    stamps = work["available_at"].to_numpy(dtype="datetime64[ns]")
    values = _moment(work["magnitude"])
    return _window_sum(stamps, values, parse_utc(issue).to_numpy(dtype="datetime64[ns]"), days)


def graph_columns(quakes: pd.DataFrame, issue: pd.Series) -> pd.DataFrame:
    """Daily local energy, neighbor inflow, and parent-box stress."""
    issue = pd.Series(issue)
    energy: dict[str, np.ndarray] = {}
    for sid in SEGMENTS:
        energy[sid] = segment_energy(quakes, issue, sid)
    inflow = {sid: np.zeros(len(issue), dtype=float) for sid in SEGMENTS}
    for src, dst, weight, _kind in GRAPH_EDGES:
        inflow[dst] = inflow[dst] + weight * energy[src]
    out = pd.DataFrame({"issue_time": issue.to_numpy()})
    parent_inflow: dict[str, np.ndarray] = {}
    parent_energy: dict[str, np.ndarray] = {}
    for sid, rec in SEGMENTS.items():
        out[f"graph_n_{sid}"] = energy[sid]
        out[f"graph_inflow_{sid}"] = inflow[sid]
        out[f"graph_stress_{sid}"] = energy[sid] + inflow[sid]
        parent = rec["parent"]
        parent_inflow[parent] = parent_inflow.get(parent, np.zeros(len(issue))) + inflow[sid]
        parent_energy[parent] = parent_energy.get(parent, np.zeros(len(issue))) + energy[sid]
    for parent, series in parent_inflow.items():
        out[f"graph_inflow_{parent}"] = series
        out[f"graph_stress_{parent}"] = parent_energy[parent] + series
    stacked = np.vstack([inflow[sid] for sid in SEGMENTS])
    out["graph_max_inflow"] = stacked.max(axis=0)
    return out


def attach_graph_features(
    feat: pd.DataFrame,
    quakes: pd.DataFrame,
    train_end: str | None = None,
) -> pd.DataFrame:
    if feat.empty or "issue_time" not in feat.columns:
        return feat
    extra = graph_columns(quakes, feat["issue_time"])
    extra["issue_time"] = parse_utc(extra["issue_time"])
    work = feat.copy()
    work["issue_time"] = parse_utc(work["issue_time"])
    cols = [c for c in extra.columns if c != "issue_time"]
    work = work.drop(columns=[c for c in cols if c in work.columns], errors="ignore")
    work = work.merge(extra, on="issue_time", how="left")
    work["graph_transfer_on"] = graph_transfer_flag(work, train_end=train_end)
    return work


def graph_transfer_flag(feat: pd.DataFrame, train_end: str | None = None) -> pd.Series:
    """On when any parent-box inflow clears the training-era 95th percentile."""
    cols = [c for c in ("graph_inflow_hikurangi", "graph_inflow_alpine", "graph_inflow_kermadec") if c in feat.columns]
    if not cols:
        return pd.Series(0.0, index=feat.index)
    stacked = feat[cols].apply(pd.to_numeric, errors="coerce")
    peak = stacked.max(axis=1)
    if train_end:
        from eqpred.stamp import utc_ts

        train = parse_utc(feat["issue_time"]) <= utc_ts(train_end)
        sample = peak.loc[train]
    else:
        sample = peak
    cut = float(sample.quantile(0.95)) if sample.notna().any() else np.inf
    return (peak >= cut).astype(float)
