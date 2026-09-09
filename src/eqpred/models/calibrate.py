from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from eqpred.models.gr import rates_to_p
from eqpred.paths import model_dir

# Issued p is s × ETAS λ. Solar g is a diagnostic, not this number.
SCALE_LO = 0.2
SCALE_HI = 25.0


def fit_rate_scale(
    lam_m0: np.ndarray,
    hits: np.ndarray,
    train_mask: np.ndarray,
    *,
    mag: float,
    m0: float,
    b: float,
    horizon: float,
) -> float:
    """Freeze the scale that minimizes train Brier of 7-day p vs hits. No solar g."""
    mask = np.asarray(train_mask, dtype=bool)
    y = np.asarray(hits, dtype=float)[mask]
    lam = np.asarray(lam_m0, dtype=float)[mask]
    if y.size == 0:
        return 1.0

    def brier(log_s: float) -> float:
        s = float(np.exp(log_s))
        p = rates_to_p(s * lam, mag, m0, b, horizon)
        return float(((p - y) ** 2).mean())

    res = minimize_scalar(brier, bounds=(np.log(SCALE_LO), np.log(SCALE_HI)), method="bounded")
    return float(np.clip(np.exp(res.x), SCALE_LO, SCALE_HI))


def issued_scale(cal: dict, region_id: str = "global", horizon_days: int = 7) -> float:
    by_h = (cal.get("by_horizon") or {}).get(str(int(horizon_days))) or {}
    scales = by_h.get("scales") or cal.get("scales") or {}
    if region_id in scales:
        return float(scales[region_id])
    if region_id == "global":
        return float(by_h.get("scale") or cal.get("scale") or 1.0)
    return 1.0


def fit_issued_scale(
    train_end: str = "2018-12-31",
    horizon_days: int = 7,
    root: Path | None = None,
) -> float:
    from eqpred.catalog import load_comcat, load_comcat_versions
    from eqpred.eval.strategies import _window_hits
    from eqpred.models.etas import daily_background_and_cluster
    from eqpred.models.gr import load_gr
    from eqpred.models.predict import BOX_MAG, GLOBAL_MAG, _region_etas, load_models
    from eqpred.models.quiet import ISSUED_SCALE_START, issued_train_mask
    from eqpred.paths import feature_dir
    from eqpred.regions import REGIONS, is_issued_box
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
    mask = issued_train_mask(features["issue_time"], train_end)
    scales: dict[str, float] = {}
    ns: dict[str, int] = {}

    hist = quakes.loc[quakes["magnitude"] >= etas.m0]
    lam = daily_background_and_cluster(hist, days, etas, versions=versions)
    hits = _window_hits(quakes, REGIONS["global"], GLOBAL_MAG, days, horizon_days)
    scales["global"] = fit_rate_scale(lam, hits, mask, mag=GLOBAL_MAG, m0=m0, b=b, horizon=horizon_days)
    ns["global"] = int(mask.sum())

    for rid, region in REGIONS.items():
        if not is_issued_box(region):
            continue
        mag = float(BOX_MAG.get(rid, 5.0))
        local, params = _region_etas(quakes, etas, region)
        lam_box = daily_background_and_cluster(local, days, params, versions=versions)
        box_hits = _window_hits(quakes, region, mag, days, horizon_days)
        scales[rid] = fit_rate_scale(
            lam_box, box_hits, mask, mag=mag, m0=m0, b=b, horizon=horizon_days
        )
        ns[rid] = int(mask.sum())

    save_calibration(
        scales,
        train_end=train_end,
        n=ns,
        horizon_days=horizon_days,
        scale_train_start=ISSUED_SCALE_START,
        root=root,
    )
    return scales["global"]


def save_calibration(
    scale: float | dict[str, float],
    *,
    train_end: str,
    n: int | dict[str, int],
    horizon_days: int = 7,
    scale_train_start: str | None = None,
    root: Path | None = None,
) -> dict:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "calibration.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    scales = {"global": float(scale)} if isinstance(scale, (int, float)) else {k: float(v) for k, v in scale.items()}
    block = {
        "scale": float(scales.get("global") or 1.0),
        "scales": scales,
        "n": n if isinstance(n, dict) else {"global": int(n)},
        "horizon_days": int(horizon_days),
    }
    by_horizon = dict(existing.get("by_horizon") or {})
    by_horizon[str(int(horizon_days))] = block
    payload = {
        "scale": float((by_horizon.get("7") or block).get("scale") or 1.0),
        "scales": (by_horizon.get("7") or block).get("scales") or scales,
        "train_end": train_end,
        "n": (by_horizon.get("7") or block).get("n") or n,
        "scale_train_start": scale_train_start or existing.get("scale_train_start"),
        "issued": "calibrated_etas",
        "modulator": "unused_on_issued_p",
        "by_horizon": by_horizon,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_calibration(root: Path | None = None) -> dict:
    path = model_dir(root) / "calibration.json"
    if not path.exists():
        return {"scale": 1.0, "scales": {"global": 1.0}, "issued": "raw_etas"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "scales" not in payload:
        payload["scales"] = {"global": float(payload.get("scale") or 1.0)}
    return payload
