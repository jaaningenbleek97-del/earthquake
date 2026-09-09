from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.eval.strategies import _window_hits
from eqpred.models.etas import EtasParams, daily_background_and_cluster
from eqpred.models.gr import load_gr, rates_to_p, scale_rate
from eqpred.models.modulator import DRIVER_COLS, Modulator, fit_modulator
from eqpred.models.calibrate import load_calibration
from eqpred.models.nowcast import load_nowcast
from eqpred.models.predict import GLOBAL_MAG, load_models
from eqpred.models.quiet import ready_train_mask
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc, utc_ts


def shift_drivers(features: pd.DataFrame, days: int, columns: list[str] | None = None) -> pd.DataFrame:
    """At day T, use driver values from T - days. Catalog λ is not shifted."""
    cols = columns or [c for c in DRIVER_COLS if c in features.columns]
    out = features.copy()
    out[cols] = out[cols].shift(int(days))
    return out


def _score(
    lam_m0: np.ndarray,
    gain: np.ndarray,
    hits: np.ndarray,
    hold: np.ndarray,
    *,
    mag: float,
    m0: float,
    b: float,
    horizon: int,
) -> dict:
    p = rates_to_p(lam_m0 * gain, mag, m0, b, horizon)
    p_etas = rates_to_p(lam_m0, mag, m0, b, horizon)
    y = hits.astype(float)
    p_h, e_h, y_h = p[hold], p_etas[hold], y[hold]
    brier = float(((p_h - y_h) ** 2).mean())
    brier_etas = float(((e_h - y_h) ** 2).mean())
    return {
        "n": int(hold.sum()),
        "hit_rate": float(y_h.mean()) if hold.any() else None,
        "mean_p": float(p_h.mean()) if hold.any() else None,
        "mean_etas_p": float(e_h.mean()) if hold.any() else None,
        "brier": brier,
        "brier_etas": brier_etas,
        "beats_etas": bool(brier < brier_etas),
    }


def fusion_nulls(
    features: pd.DataFrame,
    quakes: pd.DataFrame,
    *,
    etas: EtasParams,
    mod: Modulator,
    b: float,
    m0: float,
    holdout_start: str = "2019-01-01",
    train_end: str = "2018-12-31",
    shift_days: int = 180,
    horizon_days: int = 7,
    mag: float = GLOBAL_MAG,
    rng_seed: int = 13,
    issued_scale: float = 1.0,
    nowcast_mod: Modulator | None = None,
    nowcast_meta: dict | None = None,
    wave2_gain: np.ndarray | None = None,
    versions: pd.DataFrame | None = None,
) -> list[dict]:
    """Did fusion beat ETAS because solar timing is real, or because mean p moved?"""
    feats = features.copy()
    feats["issue_time"] = parse_utc(feats["issue_time"])
    quakes = quakes.copy()
    quakes["observed_at"] = parse_utc(quakes["observed_at"])
    days = pd.DatetimeIndex(feats["issue_time"])
    hist = quakes.loc[quakes["magnitude"] >= etas.m0]
    lam = daily_background_and_cluster(hist, days, etas, versions=versions)
    hits = _window_hits(quakes, REGIONS["global"], mag, days, horizon_days)
    hold = (feats["issue_time"] >= utc_ts(holdout_start)).to_numpy()
    solar_gain = np.asarray(mod.gain(feats), dtype=float)
    if nowcast_mod is None:
        from eqpred.models.nowcast import identity_nowcast

        nowcast_mod = identity_nowcast()
        nowcast_meta = {"used_on_issued": False, "nowcast_scale": 1.0}
    nowcast_meta = nowcast_meta or {}
    nowcast_gain = np.asarray(nowcast_mod.gain(feats), dtype=float)
    ncal = float(nowcast_meta.get("nowcast_scale") or 1.0)
    issued_gain = np.full(len(lam), float(issued_scale), dtype=float)
    if nowcast_meta.get("used_on_issued"):
        issued_gain = issued_gain * nowcast_gain * ncal
    if wave2_gain is not None:
        issued_gain = issued_gain * np.asarray(wave2_gain, dtype=float)
    common = {
        "holdout_start": holdout_start,
        "magnitude_min": mag,
        "horizon_days": horizon_days,
    }

    issued = {
        "name": "fusion_issued_m7",
        **common,
        "scale": float(issued_scale),
        "nowcast_on": bool(nowcast_meta.get("used_on_issued")),
        **_score(
            lam,
            issued_gain,
            hits,
            hold,
            mag=mag,
            m0=m0,
            b=b,
            horizon=horizon_days,
        ),
    }
    nowcast_row = {
        "name": "fusion_nowcast_m7",
        **common,
        **_score(
            lam,
            np.full(len(lam), float(issued_scale), dtype=float) * nowcast_gain * ncal,
            hits,
            hold,
            mag=mag,
            m0=m0,
            b=b,
            horizon=horizon_days,
        ),
    }
    modulator = {
        "name": "fusion_modulator_m7",
        **common,
        **_score(lam, solar_gain, hits, hold, mag=mag, m0=m0, b=b, horizon=horizon_days),
    }

    shifted = shift_drivers(feats, shift_days)
    shift_gain = np.asarray(mod.gain(shifted), dtype=float)
    shifted_rep = {
        "name": f"fusion_timeshift_{shift_days}d",
        **common,
        "shift_days": shift_days,
        **_score(lam, shift_gain, hits, hold, mag=mag, m0=m0, b=b, horizon=horizon_days),
    }

    train_mask = ready_train_mask(feats["issue_time"], train_end)
    clim_p = float(hits[train_mask].mean()) if train_mask.any() else 0.0
    p_clim = np.full(len(feats), clim_p)
    p_etas = rates_to_p(lam, mag, m0, b, horizon_days)
    y = hits.astype(float)
    clim = {
        "name": "fusion_climatology_m7",
        **common,
        "n": int(hold.sum()),
        "hit_rate": float(y[hold].mean()) if hold.any() else None,
        "mean_p": clim_p,
        "mean_etas_p": float(p_etas[hold].mean()) if hold.any() else None,
        "brier": float(((p_clim[hold] - y[hold]) ** 2).mean()) if hold.any() else None,
        "brier_etas": float(((p_etas[hold] - y[hold]) ** 2).mean()) if hold.any() else None,
        "train_base_rate": clim_p,
    }
    clim["beats_etas"] = bool(clim["brier"] is not None and clim["brier"] < clim["brier_etas"])

    lam_m7 = np.asarray(scale_rate(lam, mag, m0, b), dtype=float)
    fake = np.random.default_rng(rng_seed).poisson(np.clip(lam_m7, 0.0, None))
    world_mod = fit_modulator(feats, lam_m7, fake.astype(float), train_mask)
    world = {
        "name": "fusion_etas_world",
        **common,
        "mean_abs_driver_beta": float(np.mean(np.abs(world_mod.beta[1:]))) if len(world_mod.beta) > 1 else 0.0,
        **_score(
            lam,
            np.asarray(world_mod.gain(feats), dtype=float),
            hits,
            hold,
            mag=mag,
            m0=m0,
            b=b,
            horizon=horizon_days,
        ),
    }

    return [issued, nowcast_row, modulator, shifted_rep, clim, world]


def run_fusion_nulls(
    features: pd.DataFrame,
    quakes: pd.DataFrame,
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
) -> list[dict]:
    etas, mod = load_models(root)
    gr = load_gr(root)
    scale = float(load_calibration(root).get("scale") or 1.0)
    nowcast_mod, nowcast_meta = load_nowcast(root)
    from eqpred.models.wave2 import load_wave2, wave2_gain_series

    wave2_gain = wave2_gain_series(load_wave2(root), "global", features)
    from eqpred.catalog import load_comcat_versions

    return fusion_nulls(
        features,
        quakes,
        etas=etas,
        mod=mod,
        b=float(gr.get("b") or 1.0),
        m0=float(gr.get("m0") or etas.m0),
        holdout_start=holdout_start,
        issued_scale=scale,
        nowcast_mod=nowcast_mod,
        nowcast_meta=nowcast_meta,
        wave2_gain=wave2_gain,
        versions=load_comcat_versions(root),
    )
