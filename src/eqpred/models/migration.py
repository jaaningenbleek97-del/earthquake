from __future__ import annotations

"""Residual gain from box migration velocities.

Along-strike vs up-dip: at most one column per box, vs the current issued
baseline (cal × Coulomb × cusp × Wave 2). Same stingy train gate as the
other box residuals. Not fit to holdout hits.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.models.box_gain import apply_residual_gain, fit_residual_gain
from eqpred.models.wave2 import existing_issued_gain, load_wave2, wave2_gain_series
from eqpred.paths import model_dir

BOX_IDS = ("hikurangi", "alpine", "kermadec")
CANDIDATES = ("mig_along", "mig_updip")


def box_migration_gain(payload: dict, region_id: str, feat_row: pd.Series) -> float:
    meta = (payload.get("regions") or {}).get(region_id) or {}
    column = str(meta.get("column") or "")
    return apply_residual_gain(meta, feat_row.get(column) if column else None)


def save_migration(regions: dict[str, dict], train_end: str, root: Path | None = None) -> dict:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    from eqpred.models.quiet import hold_issued_garnish

    payload = hold_issued_garnish(
        {
            "train_end": train_end,
            "issued": "held_out_of_issued_p",
            "baseline": "issued_etas_x_existing_gains_x_wave2",
            "regions": regions,
        }
    )
    (folder / "migration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_migration(root: Path | None = None) -> dict:
    path = model_dir(root) / "migration.json"
    if not path.exists():
        return {"regions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _better_train(left: dict, right: dict) -> bool:
    brier_l = left.get("train_brier_with_gain")
    brier_r = right.get("train_brier_with_gain")
    if brier_l is not None and brier_r is not None and brier_l != brier_r:
        return float(brier_l) < float(brier_r)
    rank_l = float(left.get("train_rank_with_gain") or 0.0)
    rank_r = float(right.get("train_rank_with_gain") or 0.0)
    return rank_l > rank_r


def fit_issued_migration(
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
    wave2 = load_wave2(root)
    regions: dict[str, dict] = {}
    for rid in BOX_IDS:
        region = REGIONS[rid]
        mag = float(BOX_MAG.get(rid, 5.0))
        local, params = _region_etas(quakes, etas, region)
        lam = daily_background_and_cluster(local, days, params, versions=versions)
        extra = existing_issued_gain(features, rid, root) * wave2_gain_series(wave2, rid, features)
        lam_eff = np.asarray(lam, dtype=float) * extra
        hits = _window_hits(quakes, region, mag, days, horizon_days)
        scale = issued_scale(cal, rid)
        candidates: list[dict] = []
        for prefix in CANDIDATES:
            column = f"{prefix}_{rid}"
            if column not in features.columns:
                candidates.append(
                    {
                        "used_on_issued": False,
                        "column": column,
                        "beta": 0.0,
                        "mean": 0.0,
                        "scale": 1.0,
                        "held_out_reason": "missing_column",
                    }
                )
                continue
            series = pd.to_numeric(features[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            meta = fit_residual_gain(
                series,
                lam_eff,
                hits,
                mask,
                scale=scale,
                mag=mag,
                m0=m0,
                b=b,
                horizon=horizon_days,
                column=column,
            )
            candidates.append(meta)
        winners = [row for row in candidates if row.get("used_on_issued")]
        if not winners:
            pick = {
                "used_on_issued": False,
                "column": f"mig_along_{rid}",
                "beta": 0.0,
                "mean": 0.0,
                "scale": 1.0,
                "candidates": candidates,
            }
        else:
            pick = dict(winners[0])
            for row in winners[1:]:
                if _better_train(row, pick):
                    pick = dict(row)
            pick["candidates"] = candidates
        regions[rid] = pick
    return save_migration(regions, train_end=train_end, root=root)
