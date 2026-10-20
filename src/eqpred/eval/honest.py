from __future__ import annotations

"""Shared split and scoring rules so hunt and the fresh-shock model cannot cheat.

Discover on early train years. Confirm on late train years. Later years
(2019+) are a replay only — they do not pick knobs or recipes.
Overlapping forecast windows are collapsed to one morning per horizon when
there are enough closed windows.

Train / validation / holdout mornings whose forecast windows cross a
boundary are dropped (a horizon-length embargo).
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd

from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.stamp import utc_ts

MIN_WEEKLY = 80
# Hunt and the fresh-shock model scan uncorrected neutrons and station pressure.
# Only a numeric GLE event id is skipped (not physics).
JUNK_SUFFIXES = ("_uncorrected", "_pressure_mbar")
JUNK_EXACT = frozenset({"cosmic_gle_id"})

EVAL_PROTOCOL = "mainshock.honest.v2"
PROSPECTIVE_EPOCH_START = "2026-09-19"
CARD_KIND_PROSPECTIVE = "prospective"
CARD_KIND_RETROSPECTIVE = "retrospective"
# Last two weeks. Hunt still scans every source; these lags are the
# climate-residual / family track, not a source filter.
HONEST_LAGS = (0, 1, 2, 3, 5, 7, 10, 14)
LONG_FAMILY_LAGS = (27, 54, 90, 180)
CLIMATE_YEARS = 5.0
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_ALPHA = 0.05
MIN_SCORE = 40
DERIVED_MODEL_PREFIXES = ("analog_", "earth_state_", "graph_")


def junk_knob(column: str) -> bool:
    """True only for cosmic_gle_id. Uncorrected counts and pressure stay in the scan."""
    return str(column).lower() in JUNK_EXACT


def week_mask(
    mask: np.ndarray,
    stride: int = HORIZON_DAYS,
    min_keep: int = MIN_WEEKLY,
    fallback: bool = True,
) -> np.ndarray:
    """One morning per horizon.

    When `fallback` is True and there are too few strided mornings, return the
    original daily mask. CSEP and official non-overlapping scores pass
    `fallback=False` so a short sample cannot silently become overlapping days.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return mask
    weekly = np.zeros(len(mask), dtype=bool)
    weekly[idx[:: max(int(stride), 1)]] = True
    if int(weekly.sum()) >= min_keep or not fallback:
        return weekly
    return mask


def purged_train_hold(
    days: pd.DatetimeIndex,
    closed: np.ndarray,
    train_end: str,
    holdout_start: str,
    purge: int = HORIZON_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Train and holdout masks with a horizon embargo so labels cannot overlap.

    A morning `t` labels earthquakes in `[t, t+purge)`. Training keeps only
    mornings whose window ends at or before the holdout start.
    """
    train_cut = utc_ts(train_end)
    hold_cut = utc_ts(holdout_start)
    closed = np.asarray(closed, dtype=bool)
    gap = pd.Timedelta(days=max(int(purge), 0))
    train_mask = closed & (days <= train_cut) & (days + gap <= hold_cut)
    hold_mask = closed & (days >= hold_cut)
    return train_mask, hold_mask


def era_masks(
    days: pd.DatetimeIndex,
    closed: np.ndarray,
    train_end: str,
    holdout_start: str,
    purge: int = HORIZON_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Discover / confirm / replay. Confirm never uses the later-year exam.

    Each boundary drops mornings whose forecast window crosses the next era.
    """
    train_cut = utc_ts(train_end)
    hold_cut = utc_ts(holdout_start)
    closed = np.asarray(closed, dtype=bool)
    gap = pd.Timedelta(days=max(int(purge), 0))
    train_mask, replay_mask = purged_train_hold(days, closed, train_end, holdout_start, purge=purge)
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
    discover_mask = train_mask & (days + gap <= confirm_cut)
    confirm_mask = train_mask & (days >= confirm_cut)
    if not discover_mask.any() or not confirm_mask.any():
        split = train_idx[max(int(train_idx.size * 0.70), 1)]
        confirm_cut = days[split]
        discover_mask = train_mask & (days + gap <= confirm_cut)
        confirm_mask = train_mask & (days >= confirm_cut)
    _ = hold_cut
    return discover_mask, confirm_mask, replay_mask, pd.Timestamp(confirm_cut).date().isoformat()


def fit_val_split(
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    purge: int = HORIZON_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on early train, validate on late train — not a single last year.

    A horizon-length gap sits between fit and validation so a training label
    cannot include a validation earthquake.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    idx = np.flatnonzero(train_mask)
    fit_mask = train_mask.copy()
    val_mask = np.zeros(len(train_mask), dtype=bool)
    gap = pd.Timedelta(days=max(int(purge), 0))
    if idx.size < 280:
        return fit_mask, val_mask
    train_days = days[train_mask]
    span = train_days[-1] - train_days[0]
    if span >= pd.Timedelta(days=365 * 4):
        val_cut = train_days[-1] - pd.DateOffset(years=2)
        fit_mask = train_mask & (days + gap <= val_cut)
        val_mask = train_mask & (days >= val_cut)
    else:
        split = idx[int(idx.size * 0.70)]
        val_cut = days[split]
        fit_mask = train_mask & (days + gap <= val_cut)
        val_mask = train_mask & (days >= val_cut)
    if int(fit_mask.sum()) < 80 or not val_mask.any():
        fit_mask = train_mask.copy()
        val_mask = np.zeros(len(train_mask), dtype=bool)
        if idx.size > 280:
            val_start = days[idx[-365]]
            fit_try = train_mask & (days + gap <= val_start)
            val_try = train_mask & (days >= val_start)
            # No validation beats a validation year whose labels overlap the fit.
            if int(fit_try.sum()) >= 80 and val_try.any():
                fit_mask, val_mask = fit_try, val_try
    return fit_mask, val_mask


def holm_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Holm–Bonferroni adjusted p-values. None stays None. Monotone non-decreasing."""
    indexed = [
        (i, float(p))
        for i, p in enumerate(pvalues)
        if p is not None and math.isfinite(float(p))
    ]
    out: list[float | None] = [None] * len(pvalues)
    if not indexed:
        return out
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 0.0
    for rank, (i, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running = max(running, adj)
        out[i] = running
    return out


def classify_card_kind(
    issued_at: datetime | str | pd.Timestamp,
    written_at: datetime | str | pd.Timestamp | None,
    grace_days: int = 1,
) -> str:
    """Prospective only if the card was physically written at or before issue morning (+ grace)."""
    if written_at is None:
        return CARD_KIND_RETROSPECTIVE
    issued = utc_ts(issued_at).normalize()
    written = utc_ts(written_at).normalize()
    if written <= issued + pd.Timedelta(days=int(grace_days)):
        return CARD_KIND_PROSPECTIVE
    return CARD_KIND_RETROSPECTIVE


def counts_as_prospective(entry: dict) -> bool:
    """A saved card is prospective only if it was flagged so and written by its issue morning (+ grace).

    Read-time check, so rows already on file are never rewritten.
    """
    issue = entry.get("issue_day") or entry.get("issued_at")
    written = entry.get("written_at")
    if not entry.get("prospective") or not issue or not written:
        return False
    return classify_card_kind(issue, written) == CARD_KIND_PROSPECTIVE


def derived_model_column(column: str) -> bool:
    """Analog / Earth-state / graph columns are model outputs, not sensors."""
    name = str(column).split("__lag", 1)[0].split("family:", 1)[-1]
    return name.startswith(DERIVED_MODEL_PREFIXES)


def rolling_climate_series(
    hits: np.ndarray,
    days: pd.DatetimeIndex,
    years: float = CLIMATE_YEARS,
    horizon: int = HORIZON_DAYS,
    min_periods: int = MIN_SCORE,
) -> np.ndarray:
    """Point-in-time hit rate of already-closed forecast windows."""
    s = pd.Series(np.asarray(hits, dtype=float), index=pd.DatetimeIndex(days))
    window = f"{int(years * 365)}D"
    past = s.shift(max(int(horizon), 1))
    roll = past.rolling(window, min_periods=min_periods).mean()
    expand = past.expanding(min_periods=min_periods).mean()
    return roll.fillna(expand).to_numpy(dtype=float)


def quiet_start_mask(
    quakes: pd.DataFrame,
    region: dict,
    mag: float,
    days: pd.DatetimeIndex,
    lookback_days: int = 10,
) -> np.ndarray:
    """Mornings with no large shock in this box for the last lookback days."""
    work = quakes.loc[quakes["magnitude"] >= mag].copy()
    if region.get("kind") == "bbox":
        from eqpred.regions import in_bbox

        work = work.loc[in_bbox(work, region)]
    if work.empty:
        return np.ones(len(days), dtype=bool)
    time_col = "available_at" if "available_at" in work.columns else "observed_at"
    times = np.sort(pd.to_datetime(work[time_col], utc=True).to_numpy(dtype="datetime64[ns]"))
    issue = pd.DatetimeIndex(days).to_numpy(dtype="datetime64[ns]")
    right = np.searchsorted(times, issue, side="right")
    left = np.searchsorted(times, issue - np.timedelta64(int(lookback_days), "D"), side="right")
    return (right - left) == 0


def family_mean_z(
    feat: pd.DataFrame,
    columns: list[str],
    family_fn,
) -> dict[str, np.ndarray]:
    """One z-scored mean per source family. Every listed column is used."""
    groups: dict[str, list[str]] = {}
    for col in columns:
        groups.setdefault(family_fn(col), []).append(col)
    out: dict[str, np.ndarray] = {}
    for family, cols in groups.items():
        mat = feat[cols].to_numpy(dtype=float)
        keep = np.isfinite(mat).any(axis=0)
        if not keep.any():
            continue
        mat = mat[:, keep]
        mu = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
        sd = np.where((~np.isfinite(sd)) | (np.abs(sd) < 1e-12), np.nan, sd)
        z = (mat - mu) / sd
        if not np.isfinite(z).any():
            continue
        with np.errstate(all="ignore"):
            stacked = np.nanmean(z, axis=1)
        if not np.isfinite(stacked).any():
            continue
        out[family] = stacked
    return out


def signed_mix(
    driver: np.ndarray,
    climate: np.ndarray,
    hits: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """OLS add-on: p = clip(climate + scale * z_train(driver))."""
    driver = np.asarray(driver, dtype=float)
    climate = np.asarray(climate, dtype=float)
    hits = np.asarray(hits, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)
    ok = train_mask & np.isfinite(driver) & np.isfinite(climate) & np.isfinite(hits)
    if int(ok.sum()) < MIN_WEEKLY:
        return None, None
    mu = float(np.mean(driver[ok]))
    sd = float(np.std(driver[ok]))
    if (not math.isfinite(sd)) or sd < 1e-12:
        return None, None
    z = (driver - mu) / sd
    resid = hits[ok] - climate[ok]
    zz = z[ok]
    var = float(np.mean(zz * zz))
    if var < 1e-12:
        return None, None
    scale = float(np.mean(zz * resid) / var)
    if not math.isfinite(scale):
        return None, None
    p = np.clip(climate + scale * z, 1e-6, 1.0 - 1e-6)
    return p, scale


def brier_score(prob: np.ndarray, hits: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    prob = np.asarray(prob, dtype=float)
    hits = np.asarray(hits, dtype=float)
    ok = np.isfinite(prob) & np.isfinite(hits)
    if mask is not None:
        ok = ok & np.asarray(mask, dtype=bool)
    if int(ok.sum()) < MIN_SCORE:
        return None
    return float(np.mean((prob[ok] - hits[ok]) ** 2))


def brier_block_bootstrap(
    p: np.ndarray,
    climate: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    block: int = HORIZON_DAYS,
    n_boot: int = BOOTSTRAP_DRAWS,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 7,
) -> dict:
    """Does p beat climate on the masked days? Moving-block bootstrap of the Brier gap."""
    idx = np.flatnonzero(
        np.asarray(mask, dtype=bool)
        & np.isfinite(p)
        & np.isfinite(climate)
        & np.isfinite(y)
    )
    empty = {"beats": False, "p_value": None, "gap": None, "n_days": int(idx.size)}
    if idx.size < MIN_SCORE:
        return empty
    pp = np.asarray(p, dtype=float)[idx]
    cc = np.asarray(climate, dtype=float)[idx]
    yy = np.asarray(y, dtype=float)[idx]
    gap = float(np.mean((cc - yy) ** 2) - np.mean((pp - yy) ** 2))
    n = int(pp.size)
    width = max(int(block), 1)
    n_blocks = int(np.ceil(n / width))
    rng = np.random.default_rng(int(seed))
    worse = 0
    for _ in range(int(n_boot)):
        chosen = rng.integers(0, n, size=n_blocks)
        chunks_p = []
        chunks_c = []
        chunks_y = []
        for start in chosen:
            take = (np.arange(width) + int(start)) % n
            chunks_p.append(pp[take])
            chunks_c.append(cc[take])
            chunks_y.append(yy[take])
        boot_p = np.concatenate(chunks_p)[:n]
        boot_c = np.concatenate(chunks_c)[:n]
        boot_y = np.concatenate(chunks_y)[:n]
        boot_gap = float(np.mean((boot_c - boot_y) ** 2) - np.mean((boot_p - boot_y) ** 2))
        if boot_gap <= 0:
            worse += 1
    p_value = (worse + 1) / (int(n_boot) + 1)
    return {
        "beats": bool(gap > 0 and p_value < float(alpha)),
        "p_value": float(p_value),
        "gap": gap,
        "n_days": n,
    }


def lag_shift(raw: np.ndarray, lag: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if int(lag) <= 0:
        return raw
    out = np.roll(raw, int(lag)).astype(float)
    out[: int(lag)] = np.nan
    return out
