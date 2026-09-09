from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from eqpred.features.nowcast import NOWCAST_COLS
from eqpred.models.gr import rates_to_p
from eqpred.models.modulator import Modulator, modulator_from_dict, modulator_to_dict
from eqpred.paths import model_dir


def identity_nowcast() -> Modulator:
    return Modulator(columns=[], beta=np.array([0.0]), means=np.array([]), scales=np.array([]))


def fit_nowcast_gain(
    features: pd.DataFrame,
    lam_m0: np.ndarray,
    hits: np.ndarray,
    train_mask: np.ndarray,
    *,
    scale: float,
    mag: float,
    m0: float,
    b: float,
    horizon: float,
) -> tuple[Modulator, dict]:
    """Residual catalog gain on ETAS. Fit on train Brier of issued 7-day p."""
    cols = [c for c in NOWCAST_COLS if c in features.columns]
    mask = np.asarray(train_mask, dtype=bool)
    y = np.asarray(hits, dtype=float)
    lam = np.asarray(lam_m0, dtype=float)
    if not cols or not mask.any():
        model = identity_nowcast()
    else:
        train = features.loc[mask, cols].to_numpy(dtype=float)
        train = np.where(np.isfinite(train), train, 0.0)
        means = train.mean(axis=0)
        scales = train.std(axis=0)
        scales = np.where(scales < 1e-6, 1.0, scales)
        z = (train - means) / scales
        lam_t = lam[mask]
        y_t = y[mask]

        def objective(beta: np.ndarray) -> float:
            g = np.exp(np.clip(beta[0] + z @ beta[1:], -2.0, 2.0))
            p = rates_to_p(scale * lam_t * g, mag, m0, b, horizon)
            return float(((p - y_t) ** 2).mean() + 0.05 * np.dot(beta, beta))

        seed = np.zeros(len(cols) + 1)
        res = minimize(objective, seed, method="L-BFGS-B")
        beta = res.x if res.success else seed
        model = Modulator(columns=cols, beta=beta, means=means, scales=scales)
    gain = np.asarray(model.gain(features), dtype=float)
    p0 = rates_to_p(scale * lam, mag, m0, b, horizon)
    p1 = rates_to_p(scale * lam * gain, mag, m0, b, horizon)
    brier0 = float(((p0[mask] - y[mask]) ** 2).mean()) if mask.any() else 1.0
    brier1 = float(((p1[mask] - y[mask]) ** 2).mean()) if mask.any() else 1.0
    used = bool(brier1 < brier0 - 1e-6)
    nowcast_scale = 1.0
    brier2 = brier1
    if used:
        from eqpred.models.calibrate import fit_rate_scale

        nowcast_scale = fit_rate_scale(
            scale * lam * gain, y, mask, mag=mag, m0=m0, b=b, horizon=horizon
        )
        p2 = rates_to_p(nowcast_scale * scale * lam * gain, mag, m0, b, horizon)
        brier2 = float(((p2[mask] - y[mask]) ** 2).mean())
    meta = {
        "used_on_issued": used,
        "nowcast_scale": float(nowcast_scale),
        "train_brier_scale_only": brier0,
        "train_brier_with_nowcast": brier1,
        "train_brier_nowcast_recal": brier2,
        "columns": model.columns,
    }
    return model, meta


def save_nowcast(model: Modulator, meta: dict, root: Path | None = None) -> dict:
    from eqpred.models.quiet import hold_issued_garnish

    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    payload = hold_issued_garnish({**modulator_to_dict(model), **meta})
    (folder / "nowcast.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def fit_issued_nowcast(
    train_end: str = "2018-12-31",
    horizon_days: int = 7,
    root: Path | None = None,
) -> tuple[Modulator, dict]:
    from eqpred.catalog import load_comcat, load_comcat_versions
    from eqpred.eval.strategies import _window_hits
    from eqpred.models.calibrate import issued_scale, load_calibration
    from eqpred.models.etas import daily_background_and_cluster
    from eqpred.models.gr import load_gr
    from eqpred.models.predict import GLOBAL_MAG, load_models
    from eqpred.models.quiet import issued_train_mask
    from eqpred.paths import feature_dir
    from eqpred.regions import REGIONS
    from eqpred.stamp import parse_utc

    features = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    features["issue_time"] = parse_utc(features["issue_time"])
    quakes = load_comcat(root)
    versions = load_comcat_versions(root)
    etas, _mod = load_models(root)
    gr = load_gr(root)
    b = float(gr.get("b") or 1.0)
    m0 = float(gr.get("m0") or etas.m0)
    days = pd.DatetimeIndex(features["issue_time"])
    hist = quakes.loc[quakes["magnitude"] >= etas.m0]
    lam = daily_background_and_cluster(hist, days, etas, versions=versions)
    hits = _window_hits(quakes, REGIONS["global"], GLOBAL_MAG, days, horizon_days).astype(float)
    mask = issued_train_mask(features["issue_time"], train_end)
    scale = issued_scale(load_calibration(root), "global")
    model, meta = fit_nowcast_gain(
        features,
        lam,
        hits,
        mask,
        scale=scale,
        mag=GLOBAL_MAG,
        m0=m0,
        b=b,
        horizon=horizon_days,
    )
    save_nowcast(model, meta, root=root)
    return model, meta


def load_nowcast(root: Path | None = None) -> tuple[Modulator, dict]:
    path = model_dir(root) / "nowcast.json"
    if not path.exists():
        return identity_nowcast(), {"used_on_issued": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = modulator_from_dict(payload) if payload.get("columns") else identity_nowcast()
    return model, payload
