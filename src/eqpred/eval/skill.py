"""Shared ticket for leaving climatology.

Brier stays a report. A stack may leave the rolling rate when information
gain is positive or a loud-morning alarm has lift, without a huge duty cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ALARM_SHARE = 0.2
MIN_N = 8
MIN_ALARM = 8


def _ok(p: np.ndarray, y: np.ndarray, climate: np.ndarray) -> np.ndarray:
    return np.isfinite(p) & np.isfinite(y) & np.isfinite(climate)


def information_gain(p: np.ndarray, y: np.ndarray, climate: np.ndarray) -> float | None:
    """Mean log score of p minus the mean log score of the climate series."""
    mask = _ok(np.asarray(p, dtype=float), np.asarray(y, dtype=float), np.asarray(climate, dtype=float))
    if int(mask.sum()) < MIN_N:
        return None
    yy = np.asarray(y, dtype=float)[mask]
    model = np.clip(np.asarray(p, dtype=float)[mask], 1e-6, 1.0 - 1e-6)
    base = np.clip(np.asarray(climate, dtype=float)[mask], 1e-6, 1.0 - 1e-6)
    model_ll = float(np.mean(yy * np.log(model) + (1.0 - yy) * np.log(1.0 - model)))
    base_ll = float(np.mean(yy * np.log(base) + (1.0 - yy) * np.log(1.0 - base)))
    return model_ll - base_ll


def rank_alarm(
    p: np.ndarray,
    y: np.ndarray,
    train_p: np.ndarray | None = None,
    share: float = ALARM_SHARE,
) -> dict[str, Any]:
    """Loud mornings, threshold from the training (or raw) scores."""
    score = np.asarray(p, dtype=float)
    hits = np.asarray(y, dtype=float)
    src = np.asarray(train_p if train_p is not None else p, dtype=float)
    finite = src[np.isfinite(src)]
    empty = {"lift": None, "hit_rate": None, "base_rate": None, "time_in_alarm": 0.0, "beats": False}
    if finite.size < MIN_N:
        return empty
    threshold = float(np.nanquantile(finite, 1.0 - share))
    entered = np.isfinite(score) & np.isfinite(hits) & (score >= threshold)
    n_days = int(np.isfinite(hits).sum())
    n_trades = int(entered.sum())
    if n_days < MIN_N or n_trades < MIN_ALARM:
        return empty
    base = float(np.nanmean(hits[np.isfinite(hits)]))
    hit_rate = float(np.mean(hits[entered]))
    lift = hit_rate - base
    time_in_alarm = n_trades / n_days
    beats = bool(lift > 0.0 and 0.0 < time_in_alarm < 0.95)
    return {
        "lift": lift,
        "hit_rate": hit_rate,
        "base_rate": base,
        "time_in_alarm": time_in_alarm,
        "beats": beats,
        "threshold": threshold,
    }


def skill_beats_climate(
    p: np.ndarray,
    y: np.ndarray,
    climate: np.ndarray,
    train_p: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pass on information gain or rank/alarm. Brier is reported, not a veto."""
    pp = np.asarray(p, dtype=float)
    yy = np.asarray(y, dtype=float)
    cc = np.asarray(climate, dtype=float)
    ig = information_gain(pp, yy, cc)
    alarm = rank_alarm(pp, yy, train_p=train_p)
    mask = _ok(pp, yy, cc)
    if int(mask.sum()) >= MIN_N:
        brier = float(np.mean((pp[mask] - yy[mask]) ** 2))
        climate_brier = float(np.mean((cc[mask] - yy[mask]) ** 2))
    else:
        brier = None
        climate_brier = None
    ig_beats = ig is not None and ig > 1e-12
    return {
        "beats": bool(ig_beats or alarm["beats"]),
        "information_gain": ig,
        "ig_beats": ig_beats,
        "alarm_beats": bool(alarm["beats"]),
        "lift": alarm["lift"],
        "time_in_alarm": alarm["time_in_alarm"],
        "brier": brier,
        "climatology_brier": climate_brier,
        "brier_beats": (
            brier is not None and climate_brier is not None and brier < climate_brier - 1e-12
        ),
    }
