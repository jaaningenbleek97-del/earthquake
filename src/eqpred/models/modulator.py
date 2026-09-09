from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DRIVER_COLS = [
    "spf_significant_window",
    "proton_spike",
    "proton_density_lag1d",
    "kp_high",
    "dst_storm",
    "tidal_strain",
    "near_new_or_full",
    "solar_cycle_phase",
]


@dataclass
class Modulator:
    columns: list[str]
    beta: np.ndarray  # intercept + weights
    means: np.ndarray
    scales: np.ndarray

    def gain(self, features: pd.DataFrame) -> np.ndarray:
        z = self._z(features)
        raw = self.beta[0] + z @ self.beta[1:]
        return np.exp(np.clip(raw, -2.0, 2.0))

    def _z(self, features: pd.DataFrame) -> np.ndarray:
        n = len(features)
        if not self.columns:
            return np.zeros((n, 0))
        mat = np.zeros((n, len(self.columns)))
        for i, col in enumerate(self.columns):
            if col in features.columns:
                mat[:, i] = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=float)
        mat = np.where(np.isfinite(mat), mat, 0.0)
        return (mat - self.means) / np.where(self.scales == 0, 1.0, self.scales)


def fit_modulator(
    features: pd.DataFrame,
    etas_daily: np.ndarray,
    daily_counts: np.ndarray,
    train_mask: np.ndarray,
    columns: list[str] | None = None,
) -> Modulator:
    cols = columns or [c for c in DRIVER_COLS if c in features.columns]
    train = features.loc[train_mask, cols].to_numpy(dtype=float)
    train = np.where(np.isfinite(train), train, 0.0)
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)

    y = daily_counts[train_mask].astype(float)
    lam0 = np.clip(etas_daily[train_mask], 1e-8, None)
    z = (train - means) / scales

    def nll(beta: np.ndarray) -> float:
        g = np.exp(np.clip(beta[0] + z @ beta[1:], -2.0, 2.0))
        lam = np.clip(lam0 * g, 1e-12, None)
        return float(-(y * np.log(lam) - lam).sum() + 0.5 * np.dot(beta, beta))

    seed = np.zeros(len(cols) + 1)
    res = minimize(nll, seed, method="L-BFGS-B")
    beta = res.x if res.success else seed
    return Modulator(columns=cols, beta=beta, means=means, scales=scales)


def modulator_to_dict(model: Modulator) -> dict:
    return {
        "columns": model.columns,
        "beta": model.beta.tolist(),
        "means": model.means.tolist(),
        "scales": model.scales.tolist(),
    }


def modulator_from_dict(data: dict) -> Modulator:
    return Modulator(
        columns=list(data["columns"]),
        beta=np.array(data["beta"], dtype=float),
        means=np.array(data["means"], dtype=float),
        scales=np.array(data["scales"], dtype=float),
    )
