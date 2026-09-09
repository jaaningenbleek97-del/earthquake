"""Paper-score the current issued stack on closed holdout mornings.

Rebuilds fusion p the same way predict does. Does not write cards.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.eval.brain import _brier
from eqpred.eval.hunt import HORIZON_DAYS
from eqpred.eval.score import murphy_decompose
from eqpred.eval.strategies import _window_hits
from eqpred.models.calibrate import issued_scale, load_calibration
from eqpred.models.coulomb import load_coulomb
from eqpred.models.cusp import load_cusp
from eqpred.models.etas import daily_background_and_cluster
from eqpred.models.gr import load_gr, rates_to_p
from eqpred.models.migration import load_migration
from eqpred.models.nowcast import load_nowcast
from eqpred.models.predict import BOX_MAG, GLOBAL_MAG, _region_etas, expected_counts, load_models
from eqpred.models.registry import load_current
from eqpred.models.wave2 import load_wave2, residual_gain_series, wave2_gain_series
from eqpred.paths import data_dir, feature_dir
from eqpred.regions import REGIONS, is_issued_box
from eqpred.records import iso
from eqpred.stamp import parse_utc, utc_ts

ISSUED_TARGETS = (
    ("global", GLOBAL_MAG),
    ("hikurangi", BOX_MAG["hikurangi"]),
    ("alpine", BOX_MAG["alpine"]),
    ("kermadec", BOX_MAG["kermadec"]),
)


def _meta_gain_series(payload: dict, region_id: str, features: pd.DataFrame, default_col: str) -> np.ndarray:
    meta = (payload.get("regions") or {}).get(region_id) or {}
    if not meta.get("used_on_issued"):
        return np.ones(len(features), dtype=float)
    col = str(meta.get("column") or default_col)
    if col and col in features.columns:
        series = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=float)
    else:
        series = np.full(len(features), np.nan)
    return residual_gain_series(meta, series)


def _wave2_global_tec_only(wave2: dict) -> dict:
    """Issued recipe before this refit: global TEC, no box wave2."""
    out = json.loads(json.dumps(wave2))
    for rid, meta in (out.get("regions") or {}).items():
        if rid == "global":
            continue
        meta["used_on_issued"] = False
        for col in meta.get("columns") or []:
            col["used_on_issued"] = False
    return out


def _score_vec(prob: np.ndarray, hits: np.ndarray, hold: np.ndarray) -> dict[str, Any]:
    y = hits[hold].astype(float)
    p = np.asarray(prob, dtype=float)[hold]
    return {
        "n": int(hold.sum()),
        "hit_rate": float(y.mean()) if y.size else None,
        "mean_p": float(np.nanmean(p)) if p.size else None,
        "brier": _brier(p, y),
        "murphy": murphy_decompose(p, y),
        "reliability": _reliability_bins(p, y),
    }


def _reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 4) -> list[dict[str, Any]]:
    ok = np.isfinite(p) & np.isfinite(y)
    if int(ok.sum()) < 8 or np.unique(p[ok]).size < 2:
        return []
    frame = pd.DataFrame({"p": p[ok], "y": y[ok]})
    frame["bin"] = pd.qcut(frame["p"], n_bins, duplicates="drop")
    rows = []
    for interval, grp in frame.groupby("bin", observed=True):
        if grp.empty:
            continue
        rows.append(
            {
                "lo": float(interval.left),
                "hi": float(interval.right),
                "n": int(len(grp)),
                "mean_p": float(grp["p"].mean()),
                "hit_rate": float(grp["y"].mean()),
            }
        )
    return rows


def run_issued_stacks(
    holdout_start: str = "2019-01-01",
    horizons: tuple[int, ...] = (1, 7, 90),
    root: Path | None = None,
) -> dict[str, Any]:
    reports = [run_issued_stack(holdout_start=holdout_start, horizon_days=h, root=root) for h in horizons]
    return {
        "writes_issued_p": False,
        "horizons": [r["horizon_days"] for r in reports],
        "reports": reports,
    }


def run_issued_stack(
    holdout_start: str = "2019-01-01",
    horizon_days: int = HORIZON_DAYS,
    root: Path | None = None,
) -> dict[str, Any]:
    feat = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog = load_comcat(root)
    versions = load_comcat_versions(root)
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    etas, mod = load_models(root)
    gr = load_gr(root)
    b = float(gr.get("b") or 1.0)
    m0 = float(gr.get("m0") or etas.m0)
    cal = load_calibration(root)
    nowcast, now_meta = load_nowcast(root)
    wave2 = load_wave2(root)
    wave2_prior = _wave2_global_tec_only(wave2)
    coulomb = load_coulomb(root)
    cusp = load_cusp(root)
    migration = load_migration(root)

    rates = expected_counts(feat, catalog, etas, mod, versions=versions)
    days = pd.DatetimeIndex(rates["issue_time"])
    catalog_end = catalog["observed_at"].max()
    horizon = int(horizon_days)
    hold = (days + pd.Timedelta(days=horizon) <= catalog_end) & (days >= utc_ts(holdout_start))
    nowcast_on = bool(now_meta.get("used_on_issued"))
    nowcast_scale = float(now_meta.get("nowcast_scale") or 1.0)
    g_now_global = (
        np.asarray(nowcast.gain(feat), dtype=float) if nowcast_on else np.ones(len(feat))
    )

    box_lams: dict[str, np.ndarray] = {}
    for rid, region in REGIONS.items():
        if not is_issued_box(region):
            continue
        hist, params = _region_etas(catalog, etas, region)
        box_lams[rid] = daily_background_and_cluster(hist, days, params, versions=versions)

    regions: list[dict[str, Any]] = []
    for rid, mag in ISSUED_TARGETS:
        hits = _window_hits(catalog, REGIONS[rid], mag, days, horizon)
        lam0 = (
            np.asarray(rates["lambda_etas"], dtype=float)
            if rid == "global"
            else box_lams[rid]
        )
        scale = issued_scale(cal, rid, horizon)
        g_w2 = wave2_gain_series(wave2, rid, feat)
        g_w2_prior = wave2_gain_series(wave2_prior, rid, feat)
        if rid == "global":
            g_box = np.ones(len(feat))
            g_now = g_now_global * nowcast_scale
        else:
            g_box = (
                _meta_gain_series(coulomb, rid, feat, f"coulomb_{rid}")
                * _meta_gain_series(cusp, rid, feat, f"cusp_align_{rid}")
                * _meta_gain_series(migration, rid, feat, f"mig_{rid}")
            )
            g_now = np.ones(len(feat))
        p_etas = rates_to_p(lam0, mag, m0, b, horizon)
        p_cal = rates_to_p(lam0 * scale, mag, m0, b, horizon)
        p_prior = rates_to_p(lam0 * scale * g_now * g_box * g_w2_prior, mag, m0, b, horizon)
        p_issued = rates_to_p(lam0 * scale * g_now * g_box * g_w2, mag, m0, b, horizon)
        wave2_cols = [
            c.get("column")
            for c in ((wave2.get("regions") or {}).get(rid) or {}).get("columns") or []
            if c.get("used_on_issued")
        ]
        etas_s = _score_vec(p_etas, hits, hold)
        cal_s = _score_vec(p_cal, hits, hold)
        prior_s = _score_vec(p_prior, hits, hold)
        issued_s = _score_vec(p_issued, hits, hold)
        regions.append(
            {
                "region": rid,
                "magnitude_min": mag,
                "wave2_on_issued": wave2_cols,
                "etas": etas_s,
                "calibrated": cal_s,
                "prior_issued": prior_s,
                "issued": issued_s,
                "issued_minus_etas": None
                if issued_s["brier"] is None or etas_s["brier"] is None
                else issued_s["brier"] - etas_s["brier"],
                "issued_minus_prior": None
                if issued_s["brier"] is None or prior_s["brier"] is None
                else issued_s["brier"] - prior_s["brier"],
            }
        )

    def _mean(key_path: tuple[str, ...]) -> float | None:
        vals = []
        for row in regions:
            cur: Any = row
            for k in key_path:
                cur = cur[k]
            if cur is not None:
                vals.append(float(cur))
        return None if not vals else float(np.mean(vals))

    report = {
        "writes_issued_p": False,
        "brain_id": (load_current(root) or {}).get("id"),
        "holdout_start": holdout_start,
        "horizon_days": horizon,
        "garnish_trained_on_days": HORIZON_DAYS,
        "n_holdout": int(hold.sum()),
        "catalog_end": str(catalog_end),
        "nowcast_on_issued": nowcast_on,
        "nowcast_scale": nowcast_scale,
        "mean_etas_brier": _mean(("etas", "brier")),
        "mean_calibrated_brier": _mean(("calibrated", "brier")),
        "mean_prior_issued_brier": _mean(("prior_issued", "brier")),
        "mean_issued_brier": _mean(("issued", "brier")),
        "regions": regions,
        "scanned_at": iso(datetime.now(timezone.utc)),
        "note": (
            f"Holdout Brier for the {horizon}-day issued stack vs raw ETAS and vs the "
            "pre-refit recipe (cal × nowcast × global TEC only). Garnish knobs were "
            f"trained on {HORIZON_DAYS}d. No cards written."
        ),
    }
    dest = data_dir(root) / "eval" / (
        "issued_stack_holdout.json" if horizon == HORIZON_DAYS else f"issued_stack_holdout_{horizon}d.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["path"] = str(dest)
    return report
