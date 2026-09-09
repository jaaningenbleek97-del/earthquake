from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from eqpred.models.gr import rates_to_p

CLIP = 1.5


def residual_gain(z: np.ndarray, beta: float) -> np.ndarray:
    return np.exp(np.clip(beta * np.asarray(z, dtype=float), -CLIP, CLIP))


def _rank_corr(p: np.ndarray, y: np.ndarray) -> float:
    if p.size < 8 or np.std(p) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    corr = spearmanr(p, y).correlation
    return float(corr) if np.isfinite(corr) else 0.0


def fit_residual_gain(
    series: np.ndarray,
    lam_m0: np.ndarray,
    hits: np.ndarray,
    train_mask: np.ndarray,
    *,
    scale: float,
    mag: float,
    m0: float,
    b: float,
    horizon: float,
    column: str,
) -> dict:
    """One-coefficient residual gain. No intercept — the box scale already owns mean p."""
    mask = np.asarray(train_mask, dtype=bool)
    y = np.asarray(hits, dtype=float)
    lam = np.asarray(lam_m0, dtype=float)
    raw = np.asarray(series, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    empty = {
        "used_on_issued": False,
        "column": column,
        "beta": 0.0,
        "mean": 0.0,
        "scale": 1.0,
        "train_brier_scale_only": None,
        "train_brier_with_gain": None,
        "train_rank_scale_only": None,
        "train_rank_with_gain": None,
    }
    if not mask.any():
        return empty
    train = raw[mask]
    mean = float(train.mean())
    std = float(train.std())
    if std < 1e-12:
        return empty
    z = (raw - mean) / std
    lam_t = lam[mask]
    y_t = y[mask]
    z_t = z[mask]

    def objective(beta: float) -> float:
        g = residual_gain(z_t, beta)
        p = rates_to_p(scale * lam_t * g, mag, m0, b, horizon)
        return float(((p - y_t) ** 2).mean() + 0.05 * beta * beta)

    res = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
    beta = float(res.x) if res.success else 0.0
    p0 = rates_to_p(scale * lam, mag, m0, b, horizon)
    p1 = rates_to_p(scale * lam * residual_gain(z, beta), mag, m0, b, horizon)
    brier0 = float(((p0[mask] - y[mask]) ** 2).mean())
    brier1 = float(((p1[mask] - y[mask]) ** 2).mean())
    corr0 = _rank_corr(p0[mask], y[mask])
    corr1 = _rank_corr(p1[mask], y[mask])
    brier_ok = brier1 < brier0 - 1e-4
    rank_ok = corr1 > corr0 + 0.01
    used = bool((brier_ok or rank_ok) and brier1 <= brier0 + 1e-9 and corr1 >= corr0 - 1e-6)
    return {
        "used_on_issued": used,
        "column": column,
        "beta": beta,
        "mean": mean,
        "scale": std,
        "train_brier_scale_only": brier0,
        "train_brier_with_gain": brier1,
        "train_rank_scale_only": corr0,
        "train_rank_with_gain": corr1,
    }


def apply_trained_gain(meta: dict, raw_value: object) -> float:
    """Apply a fitted residual even if it is held off issued p."""
    if not meta:
        return 1.0
    raw = pd.to_numeric(raw_value, errors="coerce")
    raw = 0.0 if pd.isna(raw) else float(raw)
    std = float(meta.get("scale") or 1.0) or 1.0
    z = (raw - float(meta.get("mean") or 0.0)) / std
    return float(residual_gain(np.array([z]), float(meta.get("beta") or 0.0))[0])


def apply_residual_gain(meta: dict, raw_value: object) -> float:
    if not meta.get("used_on_issued"):
        return 1.0
    return apply_trained_gain(meta, raw_value)
