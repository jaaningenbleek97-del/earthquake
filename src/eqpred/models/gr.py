from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eqpred.paths import model_dir

# Fallback if the train catalog is too thin to estimate b.
LOCKED_B = 1.0
BIN_WIDTH = 0.1


def fit_b_value(magnitudes, m0: float, bin_width: float = BIN_WIDTH) -> float:
    """Aki–Utsu MLE with the usual 0.1-mag bin correction. Frozen after train."""
    mag = np.asarray(magnitudes, dtype=float)
    mag = mag[np.isfinite(mag) & (mag >= m0)]
    if mag.size < 50:
        return LOCKED_B
    denom = float(mag.mean() - (m0 - bin_width / 2.0))
    if denom <= 1e-9:
        return LOCKED_B
    b = float(np.log10(np.e) / denom)
    return float(np.clip(b, 0.6, 1.5))


def scale_rate(rate_m0: float | np.ndarray, magnitude_min: float, m0: float, b: float) -> float | np.ndarray:
    """Expected count at M≥magnitude_min from a rate fitted at completeness m0.

    Never scales *up*. Asking for a floor below completeness would invent events
    the catalog cannot support.
    """
    dm = max(float(magnitude_min) - float(m0), 0.0)
    factor = 10.0 ** (-float(b) * dm)
    return rate_m0 * factor


def poisson_at_least_one(daily_rate: float, horizon_days: float) -> float:
    expected = max(float(daily_rate), 0.0) * float(horizon_days)
    return float(1.0 - np.exp(-expected))


def rates_to_p(
    daily_rate_m0: float | np.ndarray,
    magnitude_min: float,
    m0: float,
    b: float,
    horizon_days: float,
) -> np.ndarray:
    lam = np.asarray(scale_rate(np.asarray(daily_rate_m0, dtype=float), magnitude_min, m0, b), dtype=float)
    return 1.0 - np.exp(-np.clip(lam, 0.0, None) * float(horizon_days))


def rate_to_p(daily_rate_m0: float, magnitude_min: float, m0: float, b: float, horizon_days: float) -> float:
    return float(np.asarray(rates_to_p(daily_rate_m0, magnitude_min, m0, b, horizon_days)))


def save_gr(b: float, m0: float, n: int, root: Path | None = None) -> dict:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "b": float(b),
        "m0": float(m0),
        "n": int(n),
        "method": "aki_utsu",
        "bin_width": BIN_WIDTH,
    }
    (folder / "gr.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_gr(root: Path | None = None) -> dict:
    path = model_dir(root) / "gr.json"
    if not path.exists():
        return {"b": LOCKED_B, "m0": 4.5, "n": 0, "method": "fallback"}
    return json.loads(path.read_text(encoding="utf-8"))
