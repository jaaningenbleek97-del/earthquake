from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from eqpred.faults import FAULTS
from eqpred.models.box_gain import apply_residual_gain, fit_residual_gain
from eqpred.paths import model_dir


def fit_coulomb_gain(
    coulomb: np.ndarray,
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
    meta = fit_residual_gain(
        coulomb,
        lam_m0,
        hits,
        train_mask,
        scale=scale,
        mag=mag,
        m0=m0,
        b=b,
        horizon=horizon,
        column=column,
    )
    meta["train_brier_with_coulomb"] = meta.get("train_brier_with_gain")
    meta["train_rank_with_coulomb"] = meta.get("train_rank_with_gain")
    return meta


def box_coulomb_gain(payload: dict, region_id: str, feat_row: pd.Series) -> float:
    meta = (payload.get("regions") or {}).get(region_id) or {}
    column = str(meta.get("column") or f"coulomb_{region_id}")
    return apply_residual_gain(meta, feat_row.get(column))


def save_coulomb(regions: dict[str, dict], train_end: str, root: Path | None = None) -> dict:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    from eqpred.models.quiet import hold_issued_garnish

    payload = hold_issued_garnish(
        {"train_end": train_end, "issued": "held_out_of_issued_p", "regions": regions}
    )
    (folder / "coulomb.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_coulomb(root: Path | None = None) -> dict:
    path = model_dir(root) / "coulomb.json"
    if not path.exists():
        return {"regions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def fit_issued_coulomb(
    train_end: str = "2018-12-31",
    horizon_days: int = 7,
    root: Path | None = None,
) -> dict:
    from eqpred.catalog import load_comcat, load_comcat_versions
    from eqpred.eval.strategies import _window_hits
    from eqpred.models.calibrate import issued_scale, load_calibration
    from eqpred.models.etas import daily_background_and_cluster
    from eqpred.models.gr import load_gr
    from eqpred.models.predict import BOX_MAG, _region_etas, load_models
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
    mask = issued_train_mask(features["issue_time"], train_end)
    cal = load_calibration(root)
    regions: dict[str, dict] = {}
    for rid in FAULTS:
        region = REGIONS.get(rid)
        if not region or region.get("kind") != "bbox":
            continue
        column = f"coulomb_{rid}"
        if column not in features.columns:
            regions[rid] = {
                "used_on_issued": False,
                "column": column,
                "beta": 0.0,
                "mean": 0.0,
                "scale": 1.0,
            }
            continue
        mag = float(BOX_MAG.get(rid, 5.0))
        local, params = _region_etas(quakes, etas, region)
        lam = daily_background_and_cluster(local, days, params, versions=versions)
        hits = _window_hits(quakes, region, mag, days, horizon_days)
        regions[rid] = fit_coulomb_gain(
            features[column].to_numpy(dtype=float),
            lam,
            hits,
            mask,
            scale=issued_scale(cal, rid),
            mag=mag,
            m0=m0,
            b=b,
            horizon=horizon_days,
            column=column,
        )
    return save_coulomb(regions, train_end=train_end, root=root)
