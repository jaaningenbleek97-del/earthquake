from __future__ import annotations

"""Shared split and scoring rules so hunt and leftover AI cannot cheat.

Discover on early train years. Confirm on late train years. Later years
(2019+) are a replay only — they do not pick knobs or recipes.
Overlapping 7-day windows are collapsed to one morning per week when
there are enough weeks.
"""

import numpy as np
import pandas as pd

from eqpred.stamp import utc_ts

HORIZON_DAYS = 7
MIN_WEEKLY = 80
# Hunt / leftover AI scan uncorrected neutrons and station pressure.
# Only a numeric GLE event id is skipped (not leftover physics).
JUNK_SUFFIXES = ("_uncorrected", "_pressure_mbar")
JUNK_EXACT = frozenset({"cosmic_gle_id"})


def junk_knob(column: str) -> bool:
    """True only for cosmic_gle_id. Uncorrected counts and pressure stay in the scan."""
    return str(column).lower() in JUNK_EXACT


def week_mask(mask: np.ndarray, stride: int = HORIZON_DAYS, min_keep: int = MIN_WEEKLY) -> np.ndarray:
    """One morning per horizon. Fall back to daily if there are too few weeks."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return mask
    weekly = np.zeros(len(mask), dtype=bool)
    weekly[idx[:: max(int(stride), 1)]] = True
    if int(weekly.sum()) >= min_keep:
        return weekly
    return mask


def era_masks(
    days: pd.DatetimeIndex,
    closed: np.ndarray,
    train_end: str,
    holdout_start: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Discover / confirm / replay. Confirm never uses the later-year exam."""
    train_cut = utc_ts(train_end)
    hold_cut = utc_ts(holdout_start)
    closed = np.asarray(closed, dtype=bool)
    train_mask = closed & (days <= train_cut)
    replay_mask = closed & (days >= hold_cut)
    train_idx = np.flatnonzero(train_mask)
    if train_idx.size == 0:
        empty = np.zeros(len(days), dtype=bool)
        return empty, empty, replay_mask, train_end
    train_days = days[train_mask]
    span = train_days[-1] - train_days[0]
    if span >= pd.Timedelta(days=365 * 4):
        confirm_cut = pd.Timestamp(train_cut) - pd.DateOffset(years=3)
    else:
        split = train_idx[int(train_idx.size * 0.70)]
        confirm_cut = days[split]
    discover_mask = train_mask & (days < confirm_cut)
    confirm_mask = train_mask & (days >= confirm_cut)
    if not discover_mask.any() or not confirm_mask.any():
        split = train_idx[max(int(train_idx.size * 0.70), 1)]
        confirm_cut = days[split]
        discover_mask = train_mask & (days < confirm_cut)
        confirm_mask = train_mask & (days >= confirm_cut)
    return discover_mask, confirm_mask, replay_mask, pd.Timestamp(confirm_cut).date().isoformat()


def fit_val_split(days: pd.DatetimeIndex, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit on early train, validate on late train — not a single last year."""
    train_mask = np.asarray(train_mask, dtype=bool)
    idx = np.flatnonzero(train_mask)
    fit_mask = train_mask.copy()
    val_mask = np.zeros(len(train_mask), dtype=bool)
    if idx.size < 280:
        return fit_mask, val_mask
    train_days = days[train_mask]
    span = train_days[-1] - train_days[0]
    if span >= pd.Timedelta(days=365 * 4):
        val_cut = train_days[-1] - pd.DateOffset(years=2)
        fit_mask = train_mask & (days < val_cut)
        val_mask = train_mask & (days >= val_cut)
    else:
        split = idx[int(idx.size * 0.70)]
        fit_mask[idx[idx >= split]] = False
        val_mask[idx[idx >= split]] = True
    if int(fit_mask.sum()) < 80 or not val_mask.any():
        fit_mask = train_mask.copy()
        val_mask = np.zeros(len(train_mask), dtype=bool)
        if idx.size > 280:
            val_mask[idx[-365:]] = True
            fit_mask[idx[-365:]] = False
    return fit_mask, val_mask
