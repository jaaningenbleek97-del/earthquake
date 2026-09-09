"""Scan every feature knob for a connection to next-week quakes.

Discover on early train years. Confirm on late train years. Later years
are a replay only — they do not decide “confirmed.” Do not write issued p.
Catalog leftovers (counts, energy, ETAS) are kept so we can see aftershock theft.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.eval.honest import era_masks, week_mask
from eqpred.eval.strategies import _window_hits
from eqpred.models.etas import EtasParams, daily_background_and_cluster
from eqpred.models.gr import load_gr, rates_to_p
from eqpred.models.predict import BOX_MAG, GLOBAL_MAG, _region_etas, load_models
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc

LAGS_DAYS = (0, 1, 2, 3, 7, 14, 21, 27, 30, 45, 54, 60, 81, 90, 180)
FRESH_LAGS = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 18, 21, 24, 26, 27, 28, 30, 35, 40, 45,
    53, 54, 55, 60, 81, 90, 108, 120, 162, 180,
)
HORIZON_DAYS = 7
FDR_Q = 0.10
HOLDOUT_P = 0.05
MIN_PAIR = 80
LOUDEST = 40

TARGETS = (
    ("global", GLOBAL_MAG),
    ("hikurangi", BOX_MAG["hikurangi"]),
    ("alpine", BOX_MAG["alpine"]),
    ("kermadec", BOX_MAG["kermadec"]),
    ("japan", BOX_MAG["japan"]),
    ("chile", BOX_MAG["chile"]),
)
FRESH_TARGETS = (
    ("global", GLOBAL_MAG),
    ("global", 8.0),
    ("hikurangi", BOX_MAG["hikurangi"]),
    ("alpine", BOX_MAG["alpine"]),
    ("kermadec", BOX_MAG["kermadec"]),
)

SKIP_COLS = {"issue_time", "mag_station"}
# Numeric GLE catalog IDs are not leftover physics. Uncorrected neutrons and
# station pressure stay in the scan (junk_knob is not applied here).
SKIP_DRIVER_EXACT = frozenset({"cosmic_gle_id"})


def driver_family(column: str) -> str:
    name = column.lower()
    if name.startswith(("n_m", "n_geonet", "n_fdsn", "days_since", "b_90", "log_energy", "energy_", "mig_swarm", "mig_n_")):
        return "catalog"
    if name.startswith("mig_"):
        return "migration"
    if name.startswith("sse_"):
        return "slowslip"
    if name.startswith(("tec_", "gnss_", "era5_", "mag_", "ulf_", "strain_", "radon_")):
        return "wave2"
    if name.startswith("grace_"):
        return "mass"
    if name.startswith("so2_"):
        return "volcano"
    if name.startswith("coulomb_"):
        return "coulomb"
    if name.startswith("cusp_"):
        return "cusp"
    if name.startswith("spf_") or name == "spf_significant_window":
        return "spf"
    if name.startswith(("lunar_", "tidal_", "near_new", "planet_", "cal_")):
        return "geometry"
    if name.startswith(("eop_", "lod_", "earth_", "pole_")):
        return "earth"
    if name.startswith(("dust_", "fireball_", "meteor_")):
        return "dust"
    if name.startswith("x_"):
        return "combo"
    if name.startswith(("climate_", "mjo_", "olr_", "pdo_", "amo_")):
        return "climate"
    if name.startswith(("neutron", "cosmic")):
        return "cosmic"
    if name.startswith(
        (
            "proton_",
            "solar_",
            "imf_",
            "kp",
            "dst_",
            "ap",
            "f107",
            "sunspot",
            "xray_",
            "swpc_",
            "goes_",
            "ae",
            "al_",
            "au_",
            "pc_",
            "flow_",
            "plasma_",
            "electric_",
            "cme_",
            "flare_",
            "sep_",
            "electron_",
            "hp_",
            "flare_",
            "hss_",
            "sep_",
            "ips_",
            "tsi_",
            "mgii_",
            "lya_",
            "bz_south",
        )
    ):
        return "solar"
    return "other"


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = int(p.size)
    q = np.ones(n, dtype=float)
    if n == 0:
        return q
    order = np.argsort(p)
    ranked = np.clip(p[order], 0.0, 1.0)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        q[order[i]] = float(min(running, 1.0))
    return q


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < MIN_PAIR:
        return None, None, n
    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return None, None, n
    rho, pval = spearmanr(xx, yy)
    if not np.isfinite(rho) or not np.isfinite(pval):
        return None, None, n
    return float(rho), float(pval), n


_ETAS_P_CACHE: dict[tuple, np.ndarray] = {}


def _etas_p(
    quakes: pd.DataFrame,
    days: pd.DatetimeIndex,
    region_id: str,
    mag: float,
    root: Path | None,
    versions: pd.DataFrame | None = None,
    horizon_days: int = HORIZON_DAYS,
) -> np.ndarray:
    cache_key = (
        id(quakes),
        id(versions) if versions is not None else 0,
        region_id,
        float(mag),
        int(len(days)),
        str(days[0]) if len(days) else "",
        int(horizon_days),
    )
    cached = _ETAS_P_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        etas, _mod = load_models(root)
    except FileNotFoundError:
        etas = EtasParams()
    gr = load_gr(root)
    b = float(gr.get("b") or 1.0)
    m0 = float(gr.get("m0") or etas.m0)
    region = REGIONS[region_id]
    if region.get("kind") == "bbox":
        hist, params = _region_etas(quakes, etas, region)
    else:
        hist = quakes.loc[quakes["magnitude"] >= etas.m0].copy()
        hist["observed_at"] = parse_utc(hist["observed_at"])
        params = etas
    if "available_at" in hist.columns:
        hist["available_at"] = parse_utc(hist["available_at"])
    lam = daily_background_and_cluster(hist, days, params, versions=versions)
    out = rates_to_p(lam, mag, m0, b, horizon_days)
    _ETAS_P_CACHE[cache_key] = out
    return out


def _driver_columns(features: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in features.columns:
        if col in SKIP_COLS or col in SKIP_DRIVER_EXACT:
            continue
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue
        out.append(str(col))
    return out


def _median_lift(driver: np.ndarray, hit: np.ndarray) -> float | None:
    """High-vs-low split at the train median. Works for any leftover knob."""
    ok = np.isfinite(driver) & np.isfinite(hit)
    if int(ok.sum()) < MIN_PAIR:
        return None
    cut = float(np.nanmedian(driver[ok]))
    high = ok & (driver >= cut)
    low = ok & (driver < cut)
    if int(high.sum()) < 20 or int(low.sum()) < 20:
        return None
    base = float(hit[ok].mean())
    if base <= 0:
        return None
    return float(hit[high].mean() / base)


def _lift(flag: np.ndarray, hit: np.ndarray) -> float | None:
    on = flag > 0.5
    if int(on.sum()) < 10:
        return None
    base = float(hit.mean()) if hit.size else 0.0
    if base <= 0:
        return None
    return float(hit[on].mean() / base)


def run_hunt(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = feat.copy()
    feat["issue_time"] = parse_utc(feat["issue_time"])
    if quakes is None:
        catalog = load_comcat(root)
        versions = load_comcat_versions(root)
    else:
        catalog = quakes
        versions = None
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    if "available_at" in catalog.columns:
        catalog["available_at"] = parse_utc(catalog["available_at"])

    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    discover_raw, confirm_raw, replay_raw, confirm_start = era_masks(
        days, closed, train_end, holdout_start
    )
    discover_mask = week_mask(discover_raw)
    confirm_mask = week_mask(confirm_raw)
    replay_mask = week_mask(replay_raw)

    columns = _driver_columns(feat)
    if fresh:
        columns = [c for c in columns if driver_family(c) != "catalog"]
    targets = FRESH_TARGETS if fresh else TARGETS
    lags = FRESH_LAGS if fresh else LAGS_DAYS
    trials: list[dict[str, Any]] = []
    for region_id, mag in targets:
        region = REGIONS[region_id]
        if fresh:
            from eqpred.eval.mainshock import fresh_hits

            hits = fresh_hits(catalog, region, mag, days, HORIZON_DAYS).astype(float)
            residual = hits
        else:
            hits = _window_hits(catalog, region, mag, days, HORIZON_DAYS).astype(float)
            etas_p = _etas_p(catalog, days, region_id, mag, root, versions=versions)
            residual = hits - etas_p
        for col in columns:
            raw = feat[col].to_numpy(dtype=float)
            for lag in lags:
                driver = np.roll(raw, lag) if lag else raw
                if lag:
                    driver = driver.copy()
                    driver[:lag] = np.nan
                train_rho, train_p, train_n = _spearman(driver[discover_mask], residual[discover_mask])
                confirm_rho, confirm_p, confirm_n = _spearman(driver[confirm_mask], residual[confirm_mask])
                replay_rho, replay_p, replay_n = _spearman(driver[replay_mask], residual[replay_mask])
                raw_rho, raw_p, _ = _spearman(driver[discover_mask], hits[discover_mask])
                finite = raw[np.isfinite(raw)]
                flag_like = (
                    finite.size > 0
                    and float(np.nanmax(finite) - np.nanmin(finite)) <= 1.0
                    and len(np.unique(finite)) <= 3
                )
                lift = _lift(driver[discover_mask], hits[discover_mask]) if flag_like else _median_lift(driver[discover_mask], hits[discover_mask])
                trials.append(
                    {
                        "driver": col,
                        "family": driver_family(col),
                        "lag_days": int(lag),
                        "region": region_id,
                        "magnitude_min": float(mag),
                        "horizon_days": HORIZON_DAYS,
                        "train_n": train_n,
                        "train_rho": train_rho,
                        "train_p": train_p,
                        "train_raw_rho": raw_rho,
                        "train_raw_p": raw_p,
                        "train_lift": lift,
                        "confirm_n": confirm_n,
                        "confirm_rho": confirm_rho,
                        "confirm_p": confirm_p,
                        "replay_n": replay_n,
                        "replay_rho": replay_rho,
                        "replay_p": replay_p,
                        "holdout_n": replay_n,
                        "holdout_rho": replay_rho,
                        "holdout_p": replay_p,
                    }
                )

    pvals = np.array([t["train_p"] if t["train_p"] is not None else 1.0 for t in trials], dtype=float)
    qvals = benjamini_hochberg(pvals)
    for trial, q in zip(trials, qvals, strict=True):
        trial["train_q"] = float(q) if trial["train_p"] is not None else None
        discovered = bool(trial["train_q"] is not None and trial["train_q"] < FDR_Q)
        same_confirm = (
            trial["train_rho"] is not None
            and trial["confirm_rho"] is not None
            and trial["train_rho"] * trial["confirm_rho"] > 0
        )
        confirmed = bool(
            discovered
            and same_confirm
            and trial["confirm_p"] is not None
            and trial["confirm_p"] < HOLDOUT_P
        )
        replayed = bool(
            confirmed
            and trial["replay_rho"] is not None
            and trial["train_rho"] is not None
            and trial["train_rho"] * trial["replay_rho"] > 0
            and trial["replay_p"] is not None
            and trial["replay_p"] < HOLDOUT_P
        )
        trial["discovered"] = discovered
        trial["confirmed"] = confirmed
        trial["replayed"] = replayed
        trial["aftershock_shaped"] = trial["family"] == "catalog"

    confirmed = [t for t in trials if t["confirmed"]]
    discovered = [t for t in trials if t["discovered"]]
    replayed = [t for t in trials if t.get("replayed")]
    unconventional_hits = [t for t in confirmed if t["family"] != "catalog"]
    loudest = sorted(
        [t for t in trials if t["train_rho"] is not None],
        key=lambda t: abs(t["train_rho"]),
        reverse=True,
    )[:LOUDEST]

    report = {
        "train_end": train_end,
        "holdout_start": holdout_start,
        "confirm_start": confirm_start,
        "horizon_days": HORIZON_DAYS,
        "lags_days": list(lags),
        "fdr_q": FDR_Q,
        "holdout_p": HOLDOUT_P,
        "n_knobs": len(columns),
        "n_tests": len(trials),
        "n_discovered": len(discovered),
        "n_confirmed": len(confirmed),
        "n_replayed": len(replayed),
        "n_confirmed_unconventional": len(unconventional_hits),
        "family_counts": {
            family: int(sum(1 for col in columns if driver_family(col) == family))
            for family in sorted({driver_family(col) for col in columns})
        },
        "confirmed": sorted(confirmed, key=lambda t: abs(t["confirm_rho"] or 0), reverse=True),
        "replayed": sorted(replayed, key=lambda t: abs(t["replay_rho"] or 0), reverse=True),
        "loudest_train": loudest,
        "scanned_at": iso(datetime.now(timezone.utc)),
        "fresh": fresh,
        "note": (
            "Fresh hunt: leftover knobs vs a declustered big-quake hit. "
            if fresh
            else "Discovered = early-train residual Spearman q < 0.10 after FDR. "
            "Catalog family is aftershock leftovers. "
        )
        + (
            "Confirmed = same sign on late train years with p < 0.05. "
            "Replayed = that link still pointed the same way after the train cut. "
            "Later years do not pick the knob. Nothing here writes issued p."
        ),
    }
    dest = data_dir(root) / "eval" / ("hunt_fresh.json" if fresh else "hunt.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def load_hunt(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "hunt.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_hunt_fresh(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "hunt_fresh.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
