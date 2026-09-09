from __future__ import annotations

"""Anti-prediction: this box stays quiet for the horizon.

Issued only when today's fusion p is in the train-era low tail, then
spent under the monthly alarm budget. Hit = no qualifying quake in the
window. Caps and the quantile are locked constants — not fit to holdout.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.paths import model_dir

ANTI_MODEL = "anti.quiet.v1"
ANTI_QUANTILE = 0.15
ANTI_DAYS_PER_MONTH = 4


def is_anti_model(model_id: object) -> bool:
    return str(model_id or "").startswith("anti.")


def invert_grade(graded: dict) -> dict:
    """Quiet succeeds when the usual event-in-box hit is 0."""
    event = int(graded.get("hit") or 0)
    near = float(graded.get("near_miss") or 0.0)
    return {
        **graded,
        "hit": 0 if event else 1,
        "near_miss": 0.0 if event else 1.0 - near,
    }


def below_cutoff(cfg: dict, region_id: str, issued_p: float) -> bool:
    meta = (cfg.get("regions") or {}).get(region_id) or {}
    cut = meta.get("p_max")
    if cut is None or not np.isfinite(issued_p):
        return False
    return float(issued_p) <= float(cut)


def save_anti(payload: dict, root: Path | None = None) -> dict:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "anti.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_anti(root: Path | None = None) -> dict:
    path = model_dir(root) / "anti.json"
    if not path.exists():
        return {"regions": {}, "quantile": ANTI_QUANTILE, "model_id": ANTI_MODEL}
    return json.loads(path.read_text(encoding="utf-8"))


def fit_issued_anti(
    train_end: str = "2018-12-31",
    horizon_days: int = 7,
    quantile: float = ANTI_QUANTILE,
    root: Path | None = None,
) -> dict:
    from eqpred.catalog import load_comcat, load_comcat_versions
    from eqpred.eval.strategies import _window_hits
    from eqpred.models.calibrate import issued_scale, load_calibration
    from eqpred.models.coulomb import load_coulomb
    from eqpred.models.cusp import load_cusp
    from eqpred.models.etas import daily_background_and_cluster
    from eqpred.models.gr import load_gr, rate_to_p
    from eqpred.models.migration import load_migration
    from eqpred.models.nowcast import load_nowcast
    from eqpred.models.predict import (
        BOX_MAG,
        GLOBAL_MAG,
        _region_etas,
        expected_counts,
        issued_fusion_p,
        load_models,
    )
    from eqpred.models.quiet import issued_train_mask
    from eqpred.models.wave2 import load_wave2
    from eqpred.paths import feature_dir
    from eqpred.regions import REGIONS, is_issued_box
    from eqpred.stamp import parse_utc

    features = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    features["issue_time"] = parse_utc(features["issue_time"])
    quakes = load_comcat(root)
    versions = load_comcat_versions(root)
    etas, mod = load_models(root)
    gr = load_gr(root)
    b = float(gr.get("b") or 1.0)
    m0 = float(gr.get("m0") or etas.m0)
    cal = load_calibration(root)
    nowcast, nowcast_meta = load_nowcast(root)
    coulomb = load_coulomb(root)
    cusp = load_cusp(root)
    wave2 = load_wave2(root)
    migration = load_migration(root)
    rates = expected_counts(features, quakes, etas, mod, versions=versions)
    rates["issue_time"] = parse_utc(rates["issue_time"])
    days = pd.DatetimeIndex(rates["issue_time"])
    train_mask = issued_train_mask(features["issue_time"], train_end)
    box_lams: dict[str, pd.Series] = {}
    for rid, region in REGIONS.items():
        if not is_issued_box(region):
            continue
        hist, params = _region_etas(quakes, etas, region)
        box_lams[rid] = pd.Series(
            daily_background_and_cluster(hist, days, params, versions=versions),
            index=rates.index,
        )

    targets = [("global", REGIONS["global"], GLOBAL_MAG)]
    targets.extend(
        (rid, region, float(BOX_MAG.get(rid, 5.0)))
        for rid, region in REGIONS.items()
        if is_issued_box(region)
    )
    nowcast_on = bool(nowcast_meta.get("used_on_issued"))
    nowcast_scale = float(nowcast_meta.get("nowcast_scale") or 1.0)
    feat_by_day = features.set_index("issue_time", drop=False)
    regions: dict[str, dict] = {}
    for rid, region, mag in targets:
        p_days = np.full(len(rates), np.nan)
        for i, (idx, row) in enumerate(rates.iterrows()):
            if not bool(train_mask[i]):
                continue
            feat_row = feat_by_day.loc[row["issue_time"]]
            if isinstance(feat_row, pd.DataFrame):
                feat_row = feat_row.iloc[0]
            lam0 = float(row["lambda_etas"]) if rid == "global" else float(box_lams[rid].loc[idx])
            g_now_global = float(nowcast.gain(feat_row.to_frame().T)[0]) if nowcast_on else 1.0
            p_days[i], _etas_p = issued_fusion_p(
                rid,
                feat_row,
                lam0,
                cal=cal,
                nowcast_on=nowcast_on,
                nowcast_scale=nowcast_scale,
                g_now_global=g_now_global,
                coulomb=coulomb,
                cusp=cusp,
                wave2=wave2,
                migration=migration,
                mag=mag,
                m0=m0,
                b=b,
                horizon_days=horizon_days,
            )
        sample = p_days[np.asarray(train_mask, dtype=bool)]
        sample = sample[np.isfinite(sample)]
        cut = float(np.quantile(sample, quantile)) if sample.size else 0.0
        hits = _window_hits(quakes, region, mag, days, horizon_days)
        train_hits = np.asarray(hits, dtype=float)[np.asarray(train_mask, dtype=bool)]
        quiet_rate = float(1.0 - train_hits.mean()) if train_hits.size else None
        regions[rid] = {
            "p_max": cut,
            "quantile": quantile,
            "train_n": int(sample.size),
            "train_p_mean": float(sample.mean()) if sample.size else None,
            "train_quiet_rate": quiet_rate,
        }
    payload = {
        "issued": "low_p_tail_walk_forward",
        "model_id": ANTI_MODEL,
        "quantile": quantile,
        "days_per_month": ANTI_DAYS_PER_MONTH,
        "note": (
            "Locked train-era issued-p quantile per box. "
            "Not fit to holdout quiet days. Monthly cap is a constant."
        ),
        "regions": regions,
    }
    return save_anti(payload, root=root)
