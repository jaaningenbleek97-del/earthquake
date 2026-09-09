from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.models.box_gain import apply_residual_gain, fit_residual_gain, residual_gain
from eqpred.paths import model_dir

# Global-mean TEC can modulate the worldwide card. NZ-local sensors stay on boxes.
GLOBAL_COLUMNS = ("tec_tecu",)
BOX_COLUMNS = (
    "tec_nz_tecu",
    "era5_sp_pa",
    "era5_swvl1",
    "mag_h_nt",
    "mag_h_range_nt",
    "mag_dbdt_nt",
    "gnss_east_mm",
    "gnss_east_7d",
    "n_geonet_m3_7d",
)
FAMILIES = {
    "tec": frozenset({"tec_tecu", "tec_nz_tecu"}),
    "era5": frozenset({"era5_sp_pa", "era5_swvl1"}),
    "mag": frozenset({"mag_h_nt", "mag_h_range_nt", "mag_dbdt_nt"}),
    "gnss": frozenset({"gnss_east_mm", "gnss_east_7d"}),
    "geonet": frozenset({"n_geonet_m3_7d"}),
}
MIN_TRAIN_DAYS = 100


def family_of(column: str) -> str:
    for name, cols in FAMILIES.items():
        if column in cols:
            return name
    return column


def _better_train(left: dict, right: dict) -> bool:
    brier_l = left.get("train_brier_with_gain")
    brier_r = right.get("train_brier_with_gain")
    if brier_l is not None and brier_r is not None and brier_l != brier_r:
        return float(brier_l) < float(brier_r)
    rank_l = float(left.get("train_rank_with_gain") or 0.0)
    rank_r = float(right.get("train_rank_with_gain") or 0.0)
    return rank_l > rank_r


def select_family_winners(trials: list[dict]) -> list[dict]:
    """Keep at most one issued column per physical family. Decision is train-only."""
    best: dict[str, dict] = {}
    for trial in trials:
        if not trial.get("used_on_issued"):
            continue
        fam = str(trial.get("family") or family_of(str(trial.get("column") or "")))
        prev = best.get(fam)
        if prev is None or _better_train(trial, prev):
            best[fam] = trial
    keep = {str(trial["column"]) for trial in best.values()}
    out: list[dict] = []
    for trial in trials:
        row = dict(trial)
        row["family"] = str(row.get("family") or family_of(str(row.get("column") or "")))
        if row.get("used_on_issued") and row.get("column") not in keep:
            row["used_on_issued"] = False
            row["held_out_reason"] = "same_family_worse_on_train"
        out.append(row)
    return out


def issued_column_metas(payload: dict, region_id: str) -> list[dict]:
    region = (payload.get("regions") or {}).get(region_id) or {}
    cols = region.get("columns") or []
    return [dict(meta) for meta in cols if meta.get("used_on_issued")]


def residual_gain_series(meta: dict, series: np.ndarray) -> np.ndarray:
    raw = np.asarray(series, dtype=float)
    if not meta.get("used_on_issued"):
        return np.ones(len(raw), dtype=float)
    mean = float(meta.get("mean") or 0.0)
    std = float(meta.get("scale") or 1.0) or 1.0
    filled = np.where(np.isfinite(raw), raw, mean)
    z = (filled - mean) / std
    return residual_gain(z, float(meta.get("beta") or 0.0))


def apply_wave2_column(meta: dict, raw_value: object) -> float:
    if not meta.get("used_on_issued"):
        return 1.0
    raw = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(raw):
        return 1.0
    return float(apply_residual_gain(meta, raw))


def wave2_gain(payload: dict, region_id: str, feat_row: pd.Series) -> float:
    gain = 1.0
    for meta in issued_column_metas(payload, region_id):
        column = str(meta.get("column") or "")
        gain *= apply_wave2_column(meta, feat_row.get(column) if column else None)
    return float(gain)


def wave2_gain_series(payload: dict, region_id: str, features: pd.DataFrame) -> np.ndarray:
    n = len(features)
    gain = np.ones(n, dtype=float)
    for meta in issued_column_metas(payload, region_id):
        column = str(meta.get("column") or "")
        if column and column in features.columns:
            series = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
        else:
            series = np.full(n, np.nan)
        gain = gain * residual_gain_series(meta, series)
    return gain


def save_wave2(regions: dict[str, dict], train_end: str, root: Path | None = None) -> dict:
    from eqpred.models.quiet import hold_issued_garnish

    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    payload = hold_issued_garnish(
        {
            "train_end": train_end,
            "issued": "held_out_of_issued_p",
            "baseline": "issued_etas_x_existing_gains",
            "regions": regions,
        }
    )
    (folder / "wave2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_wave2(root: Path | None = None) -> dict:
    path = model_dir(root) / "wave2.json"
    if not path.exists():
        return {"regions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _coulomb_or_cusp_series(payload: dict, region_id: str, features: pd.DataFrame) -> np.ndarray:
    meta = (payload.get("regions") or {}).get(region_id) or {}
    column = str(meta.get("column") or "")
    if column and column in features.columns:
        series = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
    else:
        series = np.full(len(features), np.nan)
    return residual_gain_series(meta, series)


def existing_issued_gain(features: pd.DataFrame, region_id: str, root: Path | None = None) -> np.ndarray:
    """Current issued multipliers except Wave 2. New columns must beat this baseline."""
    n = len(features)
    gain = np.ones(n, dtype=float)
    if region_id == "global":
        from eqpred.models.nowcast import load_nowcast

        nowcast, meta = load_nowcast(root)
        if meta.get("used_on_issued"):
            gain = gain * np.asarray(nowcast.gain(features), dtype=float) * float(meta.get("nowcast_scale") or 1.0)
        return gain
    from eqpred.models.coulomb import load_coulomb
    from eqpred.models.cusp import load_cusp

    gain = gain * _coulomb_or_cusp_series(load_coulomb(root), region_id, features)
    gain = gain * _coulomb_or_cusp_series(load_cusp(root), region_id, features)
    return gain


def _fit_column(
    series: np.ndarray,
    lam_eff: np.ndarray,
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
    finite = np.isfinite(series)
    mask = np.asarray(train_mask, dtype=bool) & finite
    if int(mask.sum()) < MIN_TRAIN_DAYS:
        return {
            "used_on_issued": False,
            "column": column,
            "family": family_of(column),
            "beta": 0.0,
            "mean": 0.0,
            "scale": 1.0,
            "held_out_reason": "too_few_train_days",
            "train_days": int(mask.sum()),
        }
    meta = fit_residual_gain(
        series,
        lam_eff,
        hits,
        mask,
        scale=scale,
        mag=mag,
        m0=m0,
        b=b,
        horizon=horizon,
        column=column,
    )
    meta["family"] = family_of(column)
    meta["train_days"] = int(mask.sum())
    if not meta.get("used_on_issued"):
        meta["held_out_reason"] = "train_gate"
    return meta


def _forward_select(
    series_by_col: dict[str, np.ndarray],
    lam_eff: np.ndarray,
    hits: np.ndarray,
    train_mask: np.ndarray,
    *,
    scale: float,
    mag: float,
    m0: float,
    b: float,
    horizon: float,
    candidates: list[str],
) -> tuple[list[dict], list[dict]]:
    """Add at most one family at a time. Each new column must beat the growing baseline."""
    remaining = [col for col in candidates if col in series_by_col]
    lam_work = np.asarray(lam_eff, dtype=float)
    issued: list[dict] = []
    log: list[dict] = []
    step = 0
    while remaining:
        step += 1
        round_fits: list[dict] = []
        for column in remaining:
            meta = _fit_column(
                series_by_col[column],
                lam_work,
                hits,
                train_mask,
                scale=scale,
                mag=mag,
                m0=m0,
                b=b,
                horizon=horizon,
                column=column,
            )
            meta["step"] = step
            round_fits.append(meta)
        winners = [row for row in round_fits if row.get("used_on_issued")]
        log.extend(round_fits)
        if not winners:
            break
        pick = winners[0]
        for row in winners[1:]:
            if _better_train(row, pick):
                pick = row
        pick = dict(pick)
        pick["held_out_reason"] = None
        issued.append(pick)
        lam_work = lam_work * residual_gain_series(pick, series_by_col[str(pick["column"])])
        fam = str(pick.get("family") or family_of(str(pick["column"])))
        remaining = [col for col in remaining if family_of(col) != fam]
    return issued, log


def fit_issued_wave2(
    train_end: str = "2018-12-31",
    horizon_days: int = 7,
    root: Path | None = None,
) -> dict:
    from eqpred.catalog import load_comcat, load_comcat_versions
    from eqpred.eval.strategies import _window_hits
    from eqpred.models.calibrate import issued_scale, load_calibration
    from eqpred.models.etas import daily_background_and_cluster
    from eqpred.models.gr import load_gr
    from eqpred.models.predict import BOX_MAG, GLOBAL_MAG, _region_etas, load_models
    from eqpred.models.quiet import issued_train_mask
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
    train_mask = issued_train_mask(features["issue_time"], train_end)
    cal = load_calibration(root)
    regions: dict[str, dict] = {}

    targets: list[tuple[str, dict, float, tuple[str, ...]]] = [
        ("global", REGIONS["global"], GLOBAL_MAG, GLOBAL_COLUMNS),
    ]
    for rid, region in REGIONS.items():
        if is_issued_box(region):
            targets.append((rid, region, float(BOX_MAG.get(rid, 5.0)), BOX_COLUMNS))

    for rid, region, mag, columns in targets:
        local, params = _region_etas(quakes, etas, region)
        lam = daily_background_and_cluster(local, days, params, versions=versions)
        extra = existing_issued_gain(features, rid, root)
        lam_eff = np.asarray(lam, dtype=float) * extra
        hits = _window_hits(quakes, region, mag, days, horizon_days)
        scale = issued_scale(cal, rid)
        series_by_col: dict[str, np.ndarray] = {}
        first_round: list[dict] = []
        for column in columns:
            if column not in features.columns:
                first_round.append(
                    {
                        "used_on_issued": False,
                        "column": column,
                        "family": family_of(column),
                        "beta": 0.0,
                        "mean": 0.0,
                        "scale": 1.0,
                        "held_out_reason": "missing_column",
                    }
                )
                continue
            series = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
            series_by_col[column] = series
            first_round.append(
                _fit_column(
                    series,
                    lam_eff,
                    hits,
                    train_mask,
                    scale=scale,
                    mag=mag,
                    m0=m0,
                    b=b,
                    horizon=horizon_days,
                    column=column,
                )
            )
        first_round = select_family_winners(first_round)
        issued_cols, sequential_trials = _forward_select(
            series_by_col,
            lam_eff,
            hits,
            train_mask,
            scale=scale,
            mag=mag,
            m0=m0,
            b=b,
            horizon=horizon_days,
            candidates=list(columns),
        )
        regions[rid] = {
            "used_on_issued": bool(issued_cols),
            "columns": issued_cols,
            "trials": first_round,
            "sequential_trials": sequential_trials,
        }
    return save_wave2(regions, train_end=train_end, root=root)
