"""Scan feature knobs for a connection to a new big quake next week.

Default hunt asks about a first shock after aftershock swarms are stripped.
It looks at every non-catalog source and delays from the same morning
through six months. A climate-residual sidecar keeps those same sources,
adds family stacks and quiet-start / M8 questions, and scores against the
rolling 5-year rate. A second hunt scans sequence objects and catalog
foreshock counts only, at delays through two weeks. Discover on early
train years. Confirm on late train years. Later years are a replay only —
they do not decide “confirmed.” Do not write issued p.
Raw catalog counts stay out of the fresh hunt so we can see aftershock theft
on the old ETAS-residual hunt.
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
from eqpred.eval.honest import (
    HONEST_LAGS,
    LONG_FAMILY_LAGS,
    brier_block_bootstrap,
    brier_score,
    derived_model_column,
    era_masks,
    family_mean_z,
    lag_shift,
    quiet_start_mask,
    rolling_climate_series,
    signed_mix,
    week_mask,
)
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.decluster import fresh_hits
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
# Foreshock sequences are a near-term question. Six-month-old swarm
# leftovers are not a foreshock of this morning's box.
SEQUENCE_LAGS = (0, 1, 2, 3, 5, 7, 10, 14)
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
# Numeric GLE catalog IDs are not physics. Uncorrected neutrons and
# station pressure stay in the scan (junk_knob is not applied here).
SKIP_DRIVER_EXACT = frozenset({"cosmic_gle_id"})


def driver_family(column: str) -> str:
    name = column.lower()
    if "hikurangi_repeater" in name:
        return "slowslip"
    if name.startswith(("n_fresh_", "days_since_fresh_", "b_fresh_", "rate_roll_")):
        return "fresh_catalog"
    if name.startswith(("n_m", "n_geonet", "n_fdsn", "days_since", "b_90", "log_energy", "energy_", "mig_swarm", "mig_n_", "seq_")):
        return "catalog"
    if name.startswith("mig_"):
        return "migration"
    if name.startswith("infra_"):
        return "infrasound"
    if name.startswith("pore_"):
        return "pressure"
    if name.startswith("schumann_"):
        return "schumann"
    if name.startswith(("wiki_", "vlf_", "tir_")) or name in {"cal_weekday", "cal_payday", "cal_weekend"}:
        return "foil"
    if name.startswith(("sse_", "tremor_", "lfe_")):
        return "slowslip"
    if name.startswith(("sdarn_",)):
        return "radio"
    if name.startswith(("tec_", "gnss_", "gnssa_", "geonet_", "era5_", "mag_", "ulf_", "strain_", "radon_", "swarm_", "cses_", "tide_", "insar_", "dart_", "well_", "hydro_", "smap_", "roti_", "scint_", "dvv_", "wspr_")):
        return "wave2"
    if name.startswith("grace_"):
        return "mass"
    if name.startswith(("so2_", "volcano_")):
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
    if name.startswith("graph_n_"):
        return "catalog"
    if name.startswith("graph_"):
        return "graph"
    if name.startswith("analog_"):
        return "analog"
    if name.startswith(("dust_", "fireball_", "meteor_")):
        return "dust"
    if name.startswith("x_"):
        return "combo"
    if name.startswith(("climate_", "mjo_", "olr_", "pdo_", "amo_", "polar_")):
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
            "coronal_hole",
            "spe_on",
            "ssc_",
            "sunspot",
            "ap_storm",
            "haarp_",
        )
    ):
        return "solar"
    return "other"


def hunt_family(column: str) -> str:
    """UI family. seq_* stay catalog for issued-feature filters."""
    name = str(column).split("__lag", 1)[0]
    if name.startswith("seq_"):
        return "sequence"
    return driver_family(name)


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
        if driver_family(str(col)) == "foil":
            continue
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue
        out.append(str(col))
    return out


def _median_lift(driver: np.ndarray, hit: np.ndarray) -> float | None:
    """High-vs-low split at the train median. Works for any non-catalog knob."""
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


def _mark_trials(trials: list[dict[str, Any]]) -> None:
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
        trial["derived_model"] = bool(trial.get("derived_model") or derived_model_column(str(trial.get("driver") or "")))


def _spearman_row(
    *,
    driver: np.ndarray,
    residual: np.ndarray,
    hits: np.ndarray,
    raw: np.ndarray,
    discover_mask: np.ndarray,
    confirm_mask: np.ndarray,
    replay_mask: np.ndarray,
    col: str,
    family: str,
    lag: int,
    region_id: str,
    mag: float,
    question: str,
) -> dict[str, Any]:
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
    lift = (
        _lift(driver[discover_mask], hits[discover_mask])
        if flag_like
        else _median_lift(driver[discover_mask], hits[discover_mask])
    )
    return {
        "driver": col,
        "family": family,
        "lag_days": int(lag),
        "region": region_id,
        "magnitude_min": float(mag),
        "horizon_days": HORIZON_DAYS,
        "question": question,
        "derived_model": derived_model_column(col),
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
        "climate_brier": None,
        "mix_brier": None,
        "beats_climate": False,
        "boot_p": None,
        "mix_scale": None,
    }


def _score_climate(trial: dict[str, Any], driver: np.ndarray, climate: np.ndarray, hits: np.ndarray, confirm_mask: np.ndarray) -> None:
    p_hat, scale = signed_mix(driver, climate, hits, confirm_mask)
    trial["mix_scale"] = scale
    if p_hat is None:
        return
    trial["climate_brier"] = brier_score(climate, hits, confirm_mask)
    trial["mix_brier"] = brier_score(p_hat, hits, confirm_mask)
    boot = brier_block_bootstrap(p_hat, climate, hits, confirm_mask)
    trial["beats_climate"] = bool(boot["beats"])
    trial["boot_p"] = boot["p_value"]
    trial["boot_gap"] = boot["gap"]


def _honest_sidecar(
    *,
    feat: pd.DataFrame,
    columns: list[str],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    discover_mask: np.ndarray,
    confirm_mask: np.ndarray,
    replay_mask: np.ndarray,
) -> dict[str, Any]:
    """All sources again, climate-residual, family stacks, quiet-start, leftover delays."""
    families = family_mean_z(feat, columns, hunt_family)
    knob_map = {col: feat[col].to_numpy(dtype=float) for col in columns}
    family_map = {f"family:{name}": series for name, series in families.items()}
    climate_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

    def hits_climate(region_id: str, mag: float) -> tuple[np.ndarray, np.ndarray]:
        key = (region_id, float(mag))
        cached = climate_cache.get(key)
        if cached is not None:
            return cached
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, HORIZON_DAYS).astype(float)
        climate = rolling_climate_series(hits, days)
        climate_cache[key] = (hits, climate)
        return hits, climate

    def scan(
        series_map: dict[str, np.ndarray],
        lags: tuple[int, ...],
        targets: tuple[tuple[str, float], ...],
        question: str,
        extra_mask: np.ndarray | None = None,
        family_from_name: bool = False,
        mark: bool = True,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        disc = discover_mask if extra_mask is None else discover_mask & extra_mask
        conf = confirm_mask if extra_mask is None else confirm_mask & extra_mask
        replay = replay_mask if extra_mask is None else replay_mask & extra_mask
        for region_id, mag in targets:
            hits, climate = hits_climate(region_id, mag)
            residual = hits - climate
            for name, raw in series_map.items():
                family = name.split("family:", 1)[1] if family_from_name else hunt_family(name)
                for lag in lags:
                    driver = lag_shift(raw, lag)
                    rows.append(
                        _spearman_row(
                            driver=driver,
                            residual=residual,
                            hits=hits,
                            raw=raw,
                            discover_mask=disc,
                            confirm_mask=conf,
                            replay_mask=replay,
                            col=name,
                            family=family,
                            lag=lag,
                            region_id=region_id,
                            mag=mag,
                            question=question,
                        )
                    )
        if mark:
            _finish_honest(rows, series_map, extra_mask)
        return rows

    def _finish_honest(
        rows: list[dict[str, Any]],
        series_map: dict[str, np.ndarray],
        extra_mask: np.ndarray | None,
    ) -> None:
        _mark_trials(rows)
        for row in rows:
            if not row["confirmed"]:
                continue
            if extra_mask is not None:
                conf = confirm_mask & extra_mask
            elif str(row.get("question") or "").startswith("quiet"):
                conf = confirm_mask & quiet_start_mask(
                    catalog, REGIONS[row["region"]], row["magnitude_min"], days
                )
            else:
                conf = confirm_mask
            raw = series_map[row["driver"]]
            hits, climate = hits_climate(row["region"], row["magnitude_min"])
            _score_climate(row, lag_shift(raw, row["lag_days"]), climate, hits, conf)

    knob_rows = scan(knob_map, HONEST_LAGS, FRESH_TARGETS, "fresh")
    family_rows = scan(family_map, HONEST_LAGS, FRESH_TARGETS, "family", family_from_name=True)
    leftover_rows = scan(family_map, LONG_FAMILY_LAGS, FRESH_TARGETS, "family_long", family_from_name=True)
    quiet_knob: list[dict[str, Any]] = []
    quiet_fam: list[dict[str, Any]] = []
    for region_id, mag in FRESH_TARGETS:
        if float(mag) >= 8.0:
            continue
        quiet = quiet_start_mask(catalog, REGIONS[region_id], mag, days)
        quiet_knob.extend(
            scan(knob_map, HONEST_LAGS, ((region_id, mag),), "quiet_start", extra_mask=quiet, mark=False)
        )
        quiet_fam.extend(
            scan(
                family_map,
                HONEST_LAGS,
                ((region_id, mag),),
                "quiet_family",
                extra_mask=quiet,
                family_from_name=True,
                mark=False,
            )
        )
    _finish_honest(quiet_knob, knob_map, None)
    _finish_honest(quiet_fam, family_map, None)
    quiet_rows = quiet_knob + quiet_fam

    all_rows = knob_rows + family_rows + leftover_rows + quiet_rows
    confirmed = [t for t in all_rows if t["confirmed"]]
    beats = [t for t in confirmed if t.get("beats_climate")]
    rare = [t for t in confirmed if float(t["magnitude_min"]) >= 8.0]
    quiet_confirmed = [t for t in confirmed if str(t.get("question") or "").startswith("quiet")]
    family_confirmed = [t for t in confirmed if str(t.get("question") or "").startswith("family")]
    loudest = sorted(
        [t for t in knob_rows if t["train_rho"] is not None],
        key=lambda t: abs(t["train_rho"]),
        reverse=True,
    )[:LOUDEST]
    return {
        "lags_days": list(HONEST_LAGS),
        "long_family_lags": list(LONG_FAMILY_LAGS),
        "looks_at_all_sources": True,
        "n_knobs": len(columns),
        "n_families": len(families),
        "n_tests": len(all_rows),
        "n_discovered": int(sum(1 for t in all_rows if t["discovered"])),
        "n_confirmed": len(confirmed),
        "n_confirmed_unconventional": int(sum(1 for t in confirmed if t["family"] not in {"catalog", "sequence"})),
        "n_beats_climate": len(beats),
        "n_confirmed_m8": len(rare),
        "n_confirmed_quiet": len(quiet_confirmed),
        "n_confirmed_family": len(family_confirmed),
        "n_replayed": int(sum(1 for t in all_rows if t.get("replayed"))),
        "family_counts": {name: int(sum(1 for col in columns if hunt_family(col) == name)) for name in sorted(families)},
        "confirmed": sorted(confirmed, key=lambda t: abs(t["confirm_rho"] or 0), reverse=True),
        "beats_climate": sorted(beats, key=lambda t: abs(t.get("boot_gap") or 0), reverse=True),
        "families": sorted(family_confirmed, key=lambda t: abs(t["confirm_rho"] or 0), reverse=True),
        "quiet": sorted(quiet_confirmed, key=lambda t: abs(t["confirm_rho"] or 0), reverse=True),
        "m8": sorted(rare, key=lambda t: abs(t["confirm_rho"] or 0), reverse=True),
        "loudest_train": loudest,
        "writes_issued_p": False,
        "note": (
            "Same sources as the wide hunt. Last-two-week delays plus family stacks "
            "and leftover 27/54/90/180-day family delays. Target is the residual "
            "after the rolling 5-year rate. Quiet-start and M8 are extra questions. "
            "Confirmed still means same sign on late train. Beats-climate is a "
            "block-bootstrap on that late-train window. Nothing here writes issued p."
        ),
    }


def run_hunt(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
    fresh: bool = True,
    sequence: bool = False,
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
    else:
        catalog["available_at"] = catalog["observed_at"]

    if sequence and not any(str(col).startswith("seq_") for col in feat.columns):
        feat = _sequence_hunt_frame(feat, catalog, versions, root)

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
    if sequence:
        columns = [c for c in columns if driver_family(c) == "catalog"]
        targets = FRESH_TARGETS
        lags = SEQUENCE_LAGS
        use_fresh = True
    elif fresh:
        columns = [c for c in columns if driver_family(c) != "catalog"]
        targets = FRESH_TARGETS
        lags = FRESH_LAGS
        use_fresh = True
    else:
        targets = TARGETS
        lags = LAGS_DAYS
        use_fresh = False
    trials: list[dict[str, Any]] = []
    for region_id, mag in targets:
        region = REGIONS[region_id]
        if use_fresh:
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
                        "family": hunt_family(col),
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

    _mark_trials(trials)

    confirmed = [t for t in trials if t["confirmed"]]
    discovered = [t for t in trials if t["discovered"]]
    replayed = [t for t in trials if t.get("replayed")]
    unconventional_hits = [t for t in confirmed if t["family"] not in {"catalog", "sequence"}]
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
        "n_confirmed_sequence": int(sum(1 for t in confirmed if t["family"] == "sequence")),
        "family_counts": {
            family: int(sum(1 for col in columns if hunt_family(col) == family))
            for family in sorted({hunt_family(col) for col in columns})
        },
        "confirmed": sorted(confirmed, key=lambda t: abs(t["confirm_rho"] or 0), reverse=True),
        "replayed": sorted(replayed, key=lambda t: abs(t["replay_rho"] or 0), reverse=True),
        "loudest_train": loudest,
        "scanned_at": iso(datetime.now(timezone.utc)),
        "fresh": use_fresh,
        "sequence": sequence,
        "writes_issued_p": False,
        "looks_at_all_sources": not sequence,
        "note": (
            "Sequence hunt: sequence objects and catalog foreshock counts only, "
            "delays from the same morning through two weeks, vs a first large quake. "
            if sequence
            else (
                "Fresh hunt: every source, delays from the same morning through 180 days, "
                "vs a first large quake after aftershock swarms are stripped. "
                if use_fresh
                else "Old hunt: non-catalog knobs vs what the aftershock model leaves behind. "
                "Raw catalog counts are aftershock leftovers. "
            )
        )
        + (
            "Confirmed = same sign on late train years with p < 0.05. "
            "Replayed = that link still pointed the same way after the train cut. "
            "Later years do not pick the knob. Nothing here writes issued p."
        ),
    }
    if use_fresh and not sequence:
        report["honest"] = _honest_sidecar(
            feat=feat,
            columns=columns,
            catalog=catalog,
            days=days,
            discover_mask=discover_mask,
            confirm_mask=confirm_mask,
            replay_mask=replay_mask,
        )
        report["note"] += (
            " The climate-residual sidecar keeps every source, adds family stacks "
            "and quiet-start / M8, and asks whether a knob beats the rolling 5-year rate."
        )
    dest_name = "hunt_sequence.json" if sequence else ("hunt_fresh.json" if use_fresh else "hunt.json")
    dest = data_dir(root) / "eval" / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def sequence_cache_matches(cached: pd.DataFrame, feat: pd.DataFrame) -> bool:
    """True only when the cached sequence frame covers the current feature calendar."""
    if cached is None or cached.empty or feat is None or feat.empty:
        return False
    if "issue_time" not in cached.columns or "issue_time" not in feat.columns:
        return False
    issue = parse_utc(feat["issue_time"])
    cached_t = parse_utc(cached["issue_time"])
    if len(cached_t) != len(issue) or len(issue) == 0:
        return False
    return (
        cached_t.iloc[0] == issue.iloc[0]
        and cached_t.iloc[-1] == issue.iloc[-1]
        and any(str(col).startswith("seq_") for col in cached.columns)
    )


def _sequence_hunt_frame(
    feat: pd.DataFrame,
    catalog: pd.DataFrame,
    versions: pd.DataFrame | None,
    root: Path | None,
) -> pd.DataFrame:
    """Build or reuse as-of sequence + catalog foreshock columns."""
    from eqpred.eval.foreshock import catalog_foreshock_frame

    cache = data_dir(root) / "eval" / "sequence_hunt_frame.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        cached["issue_time"] = parse_utc(cached["issue_time"])
        if sequence_cache_matches(cached, feat):
            return cached
    built = catalog_foreshock_frame(catalog, feat["issue_time"], versions, root)
    built["issue_time"] = parse_utc(built["issue_time"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    built.to_parquet(cache, index=False)
    return built


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


def load_hunt_sequence(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "hunt_sequence.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
