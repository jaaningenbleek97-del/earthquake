from __future__ import annotations

"""Frozen Significant Window recipes.

v1 was a loose encoding of Davidson, U-Yen, Holloman 2015 and stayed on
~80% of days — a weather channel, not an alarm.

v2 keeps the same ingredients (strong extrema, average-field reversals,
post-M8 delay) but only the loud peaks. Window lengths are constants.
Do not retune them after seeing a new M8. The 20% train-era alert budget
is a separate cap, locked on data available at train_end.
"""

import numpy as np
import pandas as pd

from eqpred.stamp import parse_utc, utc_ts

# v1 (kept for the battery comparison only).
V1 = {
    "peak_half": 12,
    "peak_extra": 8,
    "peak_quantile": 0.0,
    "peak_min_abs": 0.0,
    "reversal_half": 18,
    "reversal_gap": 80,
    "reversal_intensity": 8.0,
    "peaks_on": ("north", "south", "avg"),
}

# v2 — locked before the 2019+ holdout is scored.
V2 = {
    "peak_half": 8,
    "peak_extra": 6,
    "peak_quantile": 0.80,
    "peak_min_abs": 40.0,
    "reversal_half": 12,
    "reversal_gap": 80,
    "reversal_intensity": 25.0,
    "peaks_on": ("north", "south"),
}

DELAY_CLOSE_DAYS = 30
DELAY_CLOSE_SHIFT = 20
DELAY_FAR_DAYS = 100
DELAY_FAR_SHIFT = 10
ALARM_BUDGET = 0.20


def _daily_spf(spf: pd.DataFrame) -> pd.DataFrame:
    frame = spf.copy()
    frame["observed_at"] = parse_utc(frame["observed_at"])
    frame = frame.sort_values("observed_at").drop_duplicates("observed_at")
    daily_idx = pd.date_range(
        frame["observed_at"].min().floor("D"),
        frame["observed_at"].max().ceil("D"),
        freq="D",
        tz="UTC",
    )
    indexed = frame.set_index("observed_at")
    out = pd.DataFrame(index=daily_idx)
    for col in ("spf_north_ut", "spf_south_ut", "spf_average_ut"):
        out[col] = indexed[col].reindex(daily_idx).interpolate(limit=15)
    return out.reset_index().rename(columns={"index": "observed_at"})


def _local_extrema(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peaks: list[int] = []
    troughs: list[int] = []
    for i in range(2, len(values) - 2):
        window = values[i - 2 : i + 3]
        if np.any(np.isnan(window)):
            continue
        if values[i] == np.nanmax(window) and values[i] > values[i - 1]:
            peaks.append(i)
        if values[i] == np.nanmin(window) and values[i] < values[i - 1]:
            troughs.append(i)
    return np.array(peaks, dtype=int), np.array(troughs, dtype=int)


def _zero_crossings(values: np.ndarray) -> list[int]:
    hits: list[int] = []
    last = -10**9
    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if np.isnan(a) or np.isnan(b):
            continue
        if a == 0 or a * b < 0:
            if i - last >= 5:
                hits.append(i)
                last = i
    return hits


def field_strength(spf: pd.DataFrame) -> pd.Series:
    daily = _daily_spf(spf)
    stacked = np.abs(
        np.vstack(
            [
                daily["spf_north_ut"].to_numpy(),
                daily["spf_south_ut"].to_numpy(),
                daily["spf_average_ut"].to_numpy(),
            ]
        )
    )
    strength = np.full(stacked.shape[1], np.nan)
    ok = np.isfinite(stacked).any(axis=0)
    if ok.any():
        with np.errstate(all="ignore"):
            strength[ok] = np.nanmax(stacked[:, ok], axis=0)
    return pd.Series(strength, index=daily["observed_at"], name="spf_strength")


def significant_windows(
    spf: pd.DataFrame,
    quakes: pd.DataFrame | None = None,
    version: str = "v2",
    train_end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Boolean daily series: True if the frozen Significant Window is on.

    `train_end` locks the "how loud is loud" thresholds to the training era, so
    a 1995 window is not judged against solar peaks from later holdout years.
    """
    cfg = V2 if version == "v2" else V1
    daily = _daily_spf(spf)
    n = len(daily)
    on = np.zeros(n, dtype=bool)
    name = f"spf_significant_window_{version}"
    if n < 10:
        return pd.Series(on, index=daily["observed_at"], name=name)

    series_map = {
        "north": daily["spf_north_ut"].to_numpy(),
        "south": daily["spf_south_ut"].to_numpy(),
        "avg": daily["spf_average_ut"].to_numpy(),
    }
    stacked = np.abs(np.vstack(list(series_map.values())))
    if not np.isfinite(stacked).any():
        return pd.Series(on, index=daily["observed_at"], name=name)
    # Thresholds come from the training era only; the windows themselves still
    # span the whole series.
    era = np.ones(n, dtype=bool)
    if train_end is not None:
        era = pd.DatetimeIndex(daily["observed_at"]) <= utc_ts(train_end)
        if not era.any():
            era = np.ones(n, dtype=bool)
    strength = np.full(n, np.nan)
    ok = np.isfinite(stacked).any(axis=0)
    if ok.any():
        with np.errstate(all="ignore"):
            strength[ok] = np.nanmax(stacked[:, ok], axis=0)
            train_strength = strength[ok & era]
            if not np.isfinite(train_strength).any():
                train_strength = strength[ok]
            p95 = float(np.nanpercentile(train_strength, 95) or 1.0)
    else:
        return pd.Series(on, index=daily["observed_at"], name=name)
    if not np.isfinite(p95) or p95 <= 0:
        p95 = 1.0

    for key in cfg["peaks_on"]:
        series = series_map[key]
        finite = np.abs(series[np.isfinite(series) & era])
        if not finite.size:
            finite = np.abs(series[np.isfinite(series)])
        qcut = float(np.quantile(finite, cfg["peak_quantile"])) if finite.size else 0.0
        min_abs = max(float(cfg["peak_min_abs"]), qcut)
        peaks, troughs = _local_extrema(series)
        parts = [p for p in (peaks, troughs) if len(p)]
        extrema = np.concatenate(parts) if parts else np.array([], dtype=int)
        for idx in extrema:
            if abs(series[idx]) < min_abs:
                continue
            extra = int(cfg["peak_extra"] * min(1.0, abs(series[idx]) / p95))
            half = int(cfg["peak_half"]) + extra
            lo, hi = max(0, idx - half), min(n, idx + half + 1)
            on[lo:hi] = True

    last_avg_rev = -10**9
    for key, is_avg in (("north", False), ("south", False), ("avg", True)):
        series = series_map[key]
        for idx in _zero_crossings(series):
            if is_avg and idx - last_avg_rev < int(cfg["reversal_gap"]):
                continue
            if not is_avg:
                look = series[max(0, idx - 40) : idx + 1]
                if look.size and np.nanmax(np.abs(look)) < float(cfg["reversal_intensity"]):
                    continue
            if is_avg:
                last_avg_rev = idx
            half = int(cfg["reversal_half"])
            lo, hi = max(0, idx - half), min(n, idx + half + 1)
            on[lo:hi] = True

    if quakes is not None and not quakes.empty:
        q = quakes.copy()
        q["observed_at"] = parse_utc(q["observed_at"])
        big = q.loc[q["magnitude"] >= 8.0, "observed_at"]
        dates = daily["observed_at"]
        delayed = on.copy()
        for t0 in big:
            for i, day in enumerate(dates):
                if not on[i]:
                    continue
                delta = (day - t0).days
                if 0 <= delta <= DELAY_CLOSE_DAYS:
                    shift = DELAY_CLOSE_SHIFT
                elif DELAY_CLOSE_DAYS < delta <= DELAY_FAR_DAYS:
                    shift = DELAY_FAR_SHIFT
                else:
                    continue
                delayed[i] = False
                j = min(n - 1, i + shift)
                delayed[j] = True
        on = delayed

    return pd.Series(on, index=daily["observed_at"], name=name)


def apply_train_budget(
    windows: pd.Series,
    strength: pd.Series,
    train_end: str,
    max_frac: float = ALARM_BUDGET,
) -> tuple[pd.Series, float]:
    """Keep the strongest train-era window days until `max_frac` of train days.

    The cutoff is locked on the training era and applied forward. It is not
    fit to holdout M8s.
    """
    cut = utc_ts(train_end)
    aligned = strength.reindex(windows.index)
    train = windows.loc[windows.index < cut]
    if train.empty or float(train.mean()) <= max_frac:
        return windows.astype(bool), 0.0
    train_s = aligned.loc[aligned.index < cut]
    on_s = train_s.where(train.reindex(train_s.index).fillna(False))
    n_keep = max(1, int(np.floor(max_frac * len(train))))
    ranked = on_s.dropna().sort_values(ascending=False)
    if ranked.empty:
        return windows.astype(bool), 0.0
    cutoff = float(ranked.iloc[min(n_keep, len(ranked)) - 1])
    out = windows.astype(bool) & (aligned >= cutoff)
    return out, cutoff
