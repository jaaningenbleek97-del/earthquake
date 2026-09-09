from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eqpred.models.box_gain import apply_residual_gain, fit_residual_gain
from eqpred.paths import model_dir
from eqpred.regions import REGIONS, is_issued_box


def box_cusp_gain(payload: dict, region_id: str, feat_row: pd.Series) -> float:
    meta = (payload.get("regions") or {}).get(region_id) or {}
    column = str(meta.get("column") or f"cusp_align_{region_id}")
    return apply_residual_gain(meta, feat_row.get(column))


def save_cusp(regions: dict[str, dict], train_end: str, root: Path | None = None) -> dict:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    from eqpred.models.quiet import hold_issued_garnish

    payload = hold_issued_garnish(
        {"train_end": train_end, "issued": "held_out_of_issued_p", "regions": regions}
    )
    (folder / "cusp.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_cusp(root: Path | None = None) -> dict:
    path = model_dir(root) / "cusp.json"
    if not path.exists():
        return {"regions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def fit_issued_cusp(
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
    for rid, region in REGIONS.items():
        if not is_issued_box(region):
            continue
        column = f"cusp_align_{rid}"
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
        regions[rid] = fit_residual_gain(
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
    return save_cusp(regions, train_end=train_end, root=root)
