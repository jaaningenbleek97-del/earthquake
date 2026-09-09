from __future__ import annotations

"""Predict a fresh big quake from leftover sources. No aftershock rate.

Train on old years. Score later closed weeks. The target is a declustered
hit — the first large shock, not the swarm that follows. Catalog counts
stay out of the feature list so the tree cannot steal aftershocks.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eqpred.catalog import load_comcat
from eqpred.eval.brain import (
    MIN_SCORE,
    MIN_TRAIN,
    SHARE,
    _alarm,
    _brier,
    _design,
    _driver_columns,
    _impute,
)
from eqpred.features.leftover import leftover_derived
from eqpred.eval.decluster import decluster
from eqpred.eval.honest import fit_val_split, week_mask
from eqpred.eval.hunt import TARGETS, driver_family, load_hunt_fresh
from eqpred.eval.strategies import _window_hits
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS, is_issued_box
from eqpred.stamp import parse_utc, utc_ts

HORIZON_DAYS = 7
MAINSHOCK_LAGS = (0, 1, 2, 3, 5, 7, 10, 14, 21, 27, 30, 45, 54, 60, 81, 90, 180)
LAST_WEEK_LAGS = (0, 1, 2, 3, 5, 7)
DELAY_LAGS = frozenset({14, 21, 27, 30, 45, 54, 60, 81, 90, 180})
LONG_MEMORY_MARKS = (
    "_lag14",
    "_lag27",
    "_lag54",
    "_d14",
    "_d27",
    "_max14",
    "_max27",
    "_rec27",
    "_z30",
    "_z90",
    "_anom90",
    "_14d",
    "_trend_90d",
    "cal_",
    "solar_bartels",
    "pole_dipole",
    "dust_meteor_shower",
    "f107_trend",
)
ISSUED_TARGETS = tuple(
    (rid, mag) for rid, mag in TARGETS if rid == "global" or is_issued_box(REGIONS[rid])
)

DRIVER_PLAIN = {
    "spf_significant_window": "solar proton-flare window",
    "proton_density_cm3": "solar proton density",
    "proton_density_lag1d": "yesterday's proton density",
    "proton_spike": "proton spike",
    "solar_wind_speed_kms": "solar wind speed",
    "imf_bz_nt": "IMF Bz",
    "kp": "magnetic Kp",
    "kp_high": "magnetic storm (Kp)",
    "dst_nt": "Dst",
    "dst_storm": "magnetic storm (Dst)",
    "f107": "solar radio flux",
    "sunspot_number": "sunspots",
    "tidal_strain": "tidal strain",
    "near_new_or_full": "new or full moon",
    "lunar_distance_re": "moon distance",
    "lunar_declination_deg": "moon declination",
    "tec_tecu": "upper-atmosphere electrons",
    "tec_nz_tecu": "NZ ionosphere",
    "tec_hikurangi_tecu": "Hikurangi ionosphere",
    "tec_alpine_tecu": "Alpine ionosphere",
    "tec_kermadec_tecu": "Kermadec ionosphere",
    "tec_japan_tecu": "Japan ionosphere",
    "tec_chile_tecu": "Chile ionosphere",
    "f107_trend_90d": "90-day solar radio trend",
    "lunar_phase_deg": "moon phase",
    "gnss_usud_north_mm": "Japan GPS (north)",
    "grace_japan_lwe_cm": "Japan gravity water",
    "coulomb_hikurangi": "tidal stress on Hikurangi",
    "coulomb_alpine": "tidal stress on the Alpine Fault",
    "coulomb_kermadec": "tidal stress on Kermadec",
    "gnss_east_mm": "GPS east motion",
    "gnss_east_1d": "GPS east motion today vs yesterday",
    "gnss_east_3d": "GPS east motion over 3 days",
    "gnss_east_7d": "GPS east motion this week",
    "ulf_pc5_nt": "ULF / Pc5 magnetic range",
    "ulf_dbdt_nt": "ULF magnetic dH/dt",
    "strain_ee_nstrain": "borehole strain",
    "radon_bq_m3": "radon",
    "era5_sp_pa": "air pressure",
    "era5_swvl1": "soil water",
    "mag_h_nt": "local magnetic field",
    "mag_h_range_nt": "magnetic field range",
    "mig_along_hikurangi": "quakes walking along Hikurangi",
    "mig_along_alpine": "quakes walking along the Alpine Fault",
    "mig_along_kermadec": "quakes walking along Kermadec",
    "mig_updip_hikurangi": "quakes walking up Hikurangi",
    "mig_updip_alpine": "quakes walking up the Alpine Fault",
    "mig_updip_kermadec": "quakes walking up Kermadec",
    "solar_cycle_phase": "solar cycle phase",
    "lunar_distance_re_d7": "week-to-week moon distance change",
    "lunar_distance_re_max7": "nearest moon this week",
    "lunar_distance_re_z30": "moon distance anomaly",
    "tec_tecu_anom90": "ionosphere vs last 90 days",
    "tec_hikurangi_tecu_z30": "Hikurangi ionosphere anomaly",
    "tec_alpine_tecu_z30": "Alpine ionosphere anomaly",
    "tec_kermadec_tecu_z30": "Kermadec ionosphere anomaly",
    "planet_hard_aspects": "hard planetary aspects",
    "planet_sun_jupiter_sep": "Sun–Jupiter separation",
    "bz_south_7d": "southward IMF this week",
    "goes_spe_7d": "solar proton event this week",
    "goes_proton_1mev": "GOES low-energy protons",
    "dst_storm_3d": "magnetic storm in the last 3 days",
    "kp_high_3d": "high Kp in the last 3 days",
    "grace_chile_lwe_cm": "Chile gravity water",
    "grace_kermadec_lwe_cm": "Kermadec gravity water",
    "grace_hikurangi_lwe_cm": "Hikurangi gravity water",
    "cal_doy_sin": "day of year (sine)",
    "cal_doy_cos": "day of year (cosine)",
    "ae_nt": "auroral electrojet",
    "al_nt": "AL index",
    "imf_bx_nt": "IMF Bx",
    "imf_by_nt": "IMF By",
    "imf_b_nt": "IMF |B|",
    "imf_clock_deg": "IMF clock angle",
    "flow_pressure_npa": "solar-wind pressure",
    "electric_field_mvm": "solar-wind electric field",
    "plasma_beta": "solar-wind plasma beta",
    "pc_index": "polar cap index",
    "climate_soi": "Southern Oscillation",
    "climate_nao": "North Atlantic Oscillation",
    "climate_ao": "Arctic Oscillation",
    "climate_qbo": "quasi-biennial oscillation",
    "climate_ao_min7": "Arctic Oscillation (week low)",
    "climate_ao_max7": "Arctic Oscillation (week high)",
    "climate_nao_min7": "North Atlantic Oscillation (week low)",
    "planet_mercury_sun_sep": "Mercury–Sun separation",
    "x_kp_tidal_strain": "Kp × tidal strain",
    "x_dst_nt_lunar_distance_re": "Dst × moon distance",
    "x_imf_bz_nt_tidal_strain": "IMF Bz × tidal strain",
    "x_tec_tecu_f107": "TEC × solar radio flux",
    "neutron_oulu_count": "Oulu neutron monitor",
    "neutron_domc_count": "Dome C neutron monitor",
    "neutron_oulu_count_d7": "week-to-week Oulu neutron change",
    "neutron_oulu_count_z30": "Oulu neutron anomaly",
    "neutron_forbush": "cosmic-ray Forbush decrease",
    "neutron_forbush_pct": "3% neutron-count drop",
    "cosmic_gle": "ground-level solar cosmic-ray event",
    "cosmic_gle_7d": "GLE in the last week",
    "x_neutron_oulu_count_kp": "Oulu neutrons × Kp",
    "eop_lod_s": "length of day",
    "eop_pmx_arcsec": "polar motion X",
    "eop_pmy_arcsec": "polar motion Y",
    "solar_tsi_wm2": "total solar irradiance",
    "solar_mgii": "Mg II solar index",
    "solar_lya": "Lyman-alpha",
    "solar_newell": "Newell coupling",
    "solar_epsilon": "Akasofu epsilon",
    "cme_count": "coronal mass ejections",
    "cme_halo": "halo CME",
    "cme_speed_kms": "CME speed",
    "dust_fireball": "fireball / meteoroid",
    "dust_meteor_shower": "meteor shower peak",
    "climate_mjo_amp": "Madden–Julian amplitude",
    "climate_pdo": "Pacific Decadal Oscillation",
    "climate_aao": "Antarctic Oscillation",
    "climate_olr": "outgoing longwave radiation",
    "electron_2mev": "GOES >2 MeV electrons",
    "lunar_eclipse_window": "eclipse geometry window",
    "hp_max": "Hp30 magnetosphere index",
    "flare_score": "DONKI flare class",
    "flare_m_count": "M-class flares",
    "hss_flag": "coronal-hole high-speed stream",
    "sep_flag": "solar energetic particles",
    "pole_dipole_lat": "geomagnetic dipole pole latitude",
}

PREREGISTERED = {
    "lunar_declination_deg",
    "lunar_distance_re_d7",
    "tec_tecu_anom90",
    "dst_nt",
    "tidal_strain",
    "spf_strength",
    "kp",
    "f107",
    "planet_sun_jupiter_sep",
    "bz_south_7d",
    "imf_bz_nt",
    "cal_doy_sin",
    "cal_doy_cos",
    "ae_nt",
    "climate_soi",
    "imf_clock_deg",
    "flow_pressure_npa",
    "x_kp_tidal_strain",
    "x_imf_bz_nt_tidal_strain",
    "neutron_oulu_count",
    "neutron_oulu_count_d7",
    "neutron_forbush",
    "neutron_forbush_pct",
    "cosmic_gle",
    "eop_lod_s",
    "solar_tsi_wm2",
    "cme_halo",
    "dust_fireball",
    "climate_mjo_amp",
}

MIX_GRID = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)
HUNT_RECIPE_PREFIXES = ("leftover_hunt",)
THIS_WEEK_BRIER_FLOOR = 0.001


def _recipe_name(recipe: str) -> str:
    return (recipe or "climate").split("|", 1)[0]


def recipe_is_leaked(recipe: str) -> bool:
    """Hunt-confirmed knobs were picked on the same later years we score."""
    name = _recipe_name(recipe)
    return any(name == prefix or name.startswith(prefix + "_") for prefix in HUNT_RECIPE_PREFIXES)


def leftover_beats_climatology(row: dict[str, Any]) -> bool:
    if recipe_is_leaked(str(row.get("recipe") or "")):
        return False
    brier = row.get("brier")
    climate = row.get("climatology_brier")
    if brier is None or climate is None:
        return False
    lift = row.get("lift")
    if lift is not None and float(lift) <= 0:
        return False
    return float(brier) < float(climate)


def leftover_this_week_held(row: dict[str, Any]) -> bool:
    """Do not put a leftover number on the globe unless the win is honest and material."""
    if not leftover_beats_climatology(row):
        return True
    delta = float(row["climatology_brier"]) - float(row["brier"])
    return delta < THIS_WEEK_BRIER_FLOOR


ISSUED_WEEK_REGIONS = ("global", "hikurangi", "alpine", "kermadec")


def leftover_week_is_held(pack: dict[str, Any] | None) -> bool:
    """True unless every issued this-week leftover number is an honest leftover call."""
    if not pack:
        return True
    week = [w for w in (pack.get("week") or []) if w.get("region") in ISSUED_WEEK_REGIONS]
    if not week:
        return True
    return all(bool(w.get("held_to_climatology")) for w in week)


def leftover_week_copy(pack: dict[str, Any] | None, *, mode: str = "all") -> dict[str, Any]:
    """Globe / lab wording. Cards always show the leftover attempt; held means it lost."""
    held = leftover_week_is_held(pack)
    last_week = mode == "last_week"
    if last_week and held:
        return {
            "held": True,
            "mode": "last_week",
            "title": "This week’s last-week leftover attempt",
            "lede": (
                "This leftover mode only looks at the last 7 days of leftover sources "
                "and calls the next 7 days. It has not beaten a constant guess on later years, "
                "so the historical rate is still the number we trust. Official ledger cards stay "
                "calibrated ETAS. Aftershock rates are too late. Japan and Chile are practice only."
            ),
            "method": (
                "The mapping from last week’s leftover pattern to next week is learned on old years "
                "and scored on later closed weeks. Cards show the call even when it loses. "
                "It is not a public warning."
            ),
        }
    if last_week:
        return {
            "held": False,
            "mode": "last_week",
            "title": "This week’s last-week leftover call",
            "lede": (
                "This leftover mode only looks at the last 7 days of leftover sources "
                "and calls the next 7 days — the first shock, not the aftershocks. "
                "Aftershock rates are too late. Japan and Chile are practice only."
            ),
            "method": (
                "The mapping from last week’s leftover pattern to next week is learned on old years "
                "and scored on later closed weeks. It is not a public warning."
            ),
        }
    if held:
        return {
            "held": True,
            "mode": "all",
            "title": "This week’s leftover attempt",
            "lede": (
                "Leftover cards show the leftover model’s attempt at a fresh large shock. "
                "It has not beaten a constant guess on later years, so the historical rate "
                "is still the number we trust. Official ledger cards stay calibrated ETAS. "
                "Aftershock rates are too late. Japan and Chile are practice only."
            ),
            "method": (
                "The leftover model trains on old years and is scored on later closed weeks. "
                "Cards show its call even when it loses. It is not a public warning."
            ),
        }
    return {
        "held": False,
        "mode": "all",
        "title": "This week’s leftover call",
        "lede": (
            "Our call is a leftover model trying to catch a fresh large earthquake — "
            "the first shock, not the aftershocks. Aftershock rates are too late. "
            "Japan and Chile are practice only."
        ),
        "method": (
            "The model trains on old years and is scored on later closed weeks. "
            "It is not a public warning."
        ),
    }


def is_last_week_column(name: str) -> bool:
    """Keep leftover series that only look at the last 7 days, not delayed memory."""
    base = str(name or "").split("__lag", 1)[0]
    if base.startswith("cal_") or base.startswith("x_"):
        return False
    return not any(mark in base for mark in LONG_MEMORY_MARKS)


def last_week_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if is_last_week_column(col)]


def read_mainshock(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "mainshock.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_mainshock(root: Path | None = None) -> dict[str, Any] | None:
    pack = read_mainshock(root)
    return _seal_report(pack) if pack else None


def _chance(value: float | None) -> str | None:
    if value is None or not math.isfinite(float(value)):
        return None
    n = int(round(max(0.0, min(1.0, float(value))) * 100))
    if n <= 0:
        return "less than 1 in 100"
    return f"{n} in 100"


def _place(region_id: str) -> str:
    if region_id == "global":
        return "Earth-wide"
    return str((REGIONS.get(region_id) or {}).get("label") or region_id)


def _driver_plain(name: str) -> str:
    base, _, lag = name.partition("__lag")
    label = DRIVER_PLAIN.get(base, base.replace("_", " "))
    if lag:
        return f"{label}, {lag} days earlier"
    return label


def prior_quiet(
    quakes: pd.DataFrame,
    region: dict,
    mag: float,
    days: pd.DatetimeIndex,
    lookback_days: int = 10,
) -> np.ndarray:
    """Mornings with no large shock in this box for the last lookback days."""
    work = quakes.loc[quakes["magnitude"] >= mag].copy()
    if region.get("kind") == "bbox":
        from eqpred.regions import in_bbox

        work = work.loc[in_bbox(work, region)]
    if work.empty:
        return np.ones(len(days), dtype=bool)
    times = np.sort(parse_utc(work["observed_at"]).to_numpy(dtype="datetime64[ns]"))
    issue = days.to_numpy(dtype="datetime64[ns]")
    right = np.searchsorted(times, issue, side="right")
    left = np.searchsorted(times, issue - np.timedelta64(lookback_days, "D"), side="right")
    return (right - left) == 0


def fresh_hits(
    quakes: pd.DataFrame,
    region: dict,
    mag: float,
    days: pd.DatetimeIndex,
    horizon_days: int = HORIZON_DAYS,
) -> np.ndarray:
    """Next-week hit using only declustered large shocks — not the swarm."""
    work = quakes.loc[quakes["magnitude"] >= mag].copy()
    if region.get("kind") == "bbox":
        from eqpred.regions import in_bbox

        work = work.loc[in_bbox(work, region)]
    if work.empty:
        return np.zeros(len(days), dtype=bool)
    mains = decluster(work, space_km=200.0, time_days=10.0)
    return _window_hits(mains, region, mag, days, horizon_days)


def _hgb(**kwargs) -> HistGradientBoostingClassifier:
    spec = dict(
        max_depth=3,
        learning_rate=0.05,
        max_iter=80,
        min_samples_leaf=40,
        l2_regularization=1.0,
        max_bins=64,
        max_features=0.4,
        random_state=7,
    )
    spec.update(kwargs)
    return HistGradientBoostingClassifier(**spec)


def _logit(**kwargs) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=float(kwargs.get("C", 0.25)),
                    max_iter=400,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _elastic(**kwargs) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=float(kwargs.get("l1_ratio", 0.5)),
                    C=float(kwargs.get("C", 0.12)),
                    max_iter=int(kwargs.get("max_iter", 180)),
                ),
            ),
        ]
    )


def _proba(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    cls = list(model.classes_)
    if 1 not in cls:
        return np.zeros(len(x))
    return np.clip(proba[:, cls.index(1)], 1e-6, 1.0 - 1e-6)


def _hunt_drivers(region_id: str, root: Path | None) -> list[str]:
    pack = load_hunt_fresh(root)
    if not pack:
        return []
    return sorted(
        {
            str(row["driver"])
            for row in pack.get("confirmed") or []
            if row.get("region") == region_id and row.get("family") != "catalog"
        }
    )


def _usable_indices(x: np.ndarray, mask: np.ndarray, min_n: int = 80) -> list[int]:
    """Keep columns that actually vary in the fit window. Late GNSS/TEC is empty here."""
    keep: list[int] = []
    for j in range(x.shape[1]):
        finite = x[mask, j]
        finite = finite[np.isfinite(finite)]
        if finite.size < min_n:
            continue
        if np.unique(np.round(finite, 6)).size < 2:
            continue
        keep.append(j)
    return keep


def _top_columns(x: np.ndarray, y: np.ndarray, names: list[str], k: int = 48) -> list[int]:
    if x.shape[1] <= k:
        return list(range(x.shape[1]))
    filled, _ = _impute(x, x)
    scores = []
    for j in range(filled.shape[1]):
        col = filled[:, j]
        if np.std(col) < 1e-12:
            scores.append(0.0)
            continue
        rho = np.corrcoef(col, y.astype(float))[0, 1]
        scores.append(0.0 if not np.isfinite(rho) else abs(float(rho)))
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:k]]


def _pool_top(raw_x: np.ndarray, y: np.ndarray, mask: np.ndarray, pool: list[int], k: int) -> list[int]:
    if not pool:
        return []
    local = _top_columns(raw_x[np.ix_(mask, pool)], y[mask], [str(i) for i in pool], k)
    return [pool[i] for i in local]


def _local_pool(names: list[str], region_id: str, usable: list[int]) -> list[int]:
    if region_id == "global":
        keys = (
            "planet_",
            "lunar_",
            "tidal_",
            "near_new",
            "proton_",
            "solar_",
            "imf_",
            "kp",
            "dst_",
            "f107",
            "sunspot",
            "xray_",
            "goes_",
            "spf_",
            "tec_tecu",
            "bz_south",
            "climate_",
            "x_",
            "ae",
            "flow_",
            "neutron_",
            "cosmic_",
            "eop_",
            "dust_",
            "cme_",
            "electron_",
        )
    else:
        keys = (
            region_id,
            "planet_",
            "lunar_",
            "tidal_",
            "near_new",
            "proton_",
            "solar_",
            "imf_",
            "kp",
            "dst_",
            "f107",
            "spf_",
            "neutron_",
            "cosmic_",
            "eop_",
            "dust_",
            "cme_",
            "electron_",
            "climate_",
        )
    out: list[int] = []
    for i in usable:
        base = names[i].split("__lag")[0]
        if any(key in base for key in keys):
            if region_id == "global" and base.startswith("tec_") and base not in {"tec_tecu", "tec_tecu_anom90"} and not base.startswith("tec_tecu_"):
                continue
            out.append(i)
    return out


def _family_pool(names: list[str], family: str, usable: list[int]) -> list[int]:
    return [i for i in usable if driver_family(names[i].split("__lag")[0]) == family]


def _mix(p: np.ndarray, y: np.ndarray, climate: float, mask: np.ndarray) -> tuple[float, float]:
    """Pick a shrink-to-climate weight on the validation days. a=0 is the constant guess."""
    if not mask.any():
        return 1.0, float(np.mean((p - y) ** 2))
    target = y[mask].astype(float)
    best_a = 0.0
    best = float(np.mean((climate - target) ** 2))
    for a in MIX_GRID:
        mixed = (1.0 - a) * climate + a * p[mask]
        brier = float(np.mean((mixed - target) ** 2))
        if brier < best - 1e-12:
            best = brier
            best_a = float(a)
    return best_a, best


def _fit_kind(kind: str, x: np.ndarray, y: np.ndarray, kw: dict) -> Any:
    if kind == "logit":
        model = _logit(**kw)
        model.fit(x, y)
        return model
    if kind == "elastic":
        model = _elastic(**kw)
        model.fit(x, y)
        return model
    model = _hgb(**kw)
    model.fit(x, y)
    return model


def _pick_model(
    raw_x: np.ndarray,
    names: list[str],
    y: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    hunt_cols: list[str],
    region_id: str = "global",
) -> tuple[Any, np.ndarray, str]:
    climate = float(y[fit_mask].mean()) if fit_mask.any() else float(y.mean())
    climate_full = float(y[fit_mask | val_mask].mean()) if (fit_mask | val_mask).any() else climate
    usable = _usable_indices(raw_x, fit_mask)
    score_mask = val_mask if val_mask.any() else fit_mask
    climate_brier = float(np.mean((climate - y[score_mask]) ** 2))
    best: tuple[float, float, str, list[int], str, dict] | None = (
        0.0,
        climate_brier,
        "climate",
        [],
        "climate",
        {},
    )
    candidates: list[tuple[str, list[int], str, dict]] = [
        ("leftover_thin", _pool_top(raw_x, y, fit_mask, usable, 48), "tree", {"max_depth": 3, "min_samples_leaf": 30}),
        ("leftover_thin_logit", _pool_top(raw_x, y, fit_mask, usable, 48), "logit", {"C": 0.25}),
        ("leftover_wide", _pool_top(raw_x, y, fit_mask, usable, 96), "tree", {"max_depth": 2, "min_samples_leaf": 80, "l2_regularization": 2.0}),
        ("leftover_wide_logit", _pool_top(raw_x, y, fit_mask, usable, 96), "logit", {"C": 0.15}),
        ("leftover_local", _pool_top(raw_x, y, fit_mask, _local_pool(names, region_id, usable), 64), "logit", {"C": 0.2}),
        ("leftover_season", [i for i in usable if names[i].split("__lag")[0].startswith("cal_")], "logit", {"C": 1.0}),
        ("leftover_prereg", [i for i in usable if names[i].split("__lag")[0] in PREREGISTERED], "logit", {"C": 0.2}),
        ("leftover_joint", _pool_top(raw_x, y, fit_mask, usable, 96), "logit", {"C": 0.12}),
        ("leftover_max", _pool_top(raw_x, y, fit_mask, usable, 160), "tree", {"max_depth": 2, "min_samples_leaf": 80, "l2_regularization": 2.0}),
        ("leftover_combo", _pool_top(raw_x, y, fit_mask, _family_pool(names, "combo", usable), 24), "logit", {"C": 0.25}),
        (
            "leftover_delay",
            _pool_top(
                raw_x,
                y,
                fit_mask,
                [i for i in usable if int((names[i].rsplit("__lag", 1)[1] if "__lag" in names[i] else 0)) in DELAY_LAGS],
                64,
            ),
            "logit",
            {"C": 0.2},
        ),
    ]
    # Hunt-confirmed knobs were picked on the same later years. Fitting them here is leakage.
    _ = hunt_cols
    for fam in ("solar", "geometry", "wave2", "mass", "climate", "combo", "cusp", "cosmic", "earth", "dust", "volcano", "slowslip"):
        pool = _family_pool(names, fam, usable)
        if len(pool) >= 4:
            candidates.append((f"leftover_{fam}", _pool_top(raw_x, y, fit_mask, pool, 32), "logit", {"C": 0.3}))
    y_fit = y[fit_mask]
    if int(fit_mask.sum()) < MIN_TRAIN or int(np.unique(y_fit).size) < 2:
        p_all = np.full(len(raw_x), climate_full)
        return None, p_all, "climate|mix0.00"
    for label, cols, kind, kw in candidates:
        cols = [c for c in cols if c in set(usable)]
        if len(cols) < 2:
            continue
        try:
            model = _fit_kind(kind, raw_x[np.ix_(fit_mask, cols)], y_fit, kw)
            p_hat = _proba(model, raw_x[:, cols])
        except ValueError:
            continue
        mix_a, brier = _mix(p_hat, y, climate, score_mask)
        score = brier - climate_brier
        if score >= -1e-12:
            continue
        confirm = np.zeros(len(fit_mask), dtype=bool)
        idx = np.flatnonzero(fit_mask)
        if idx.size > 365:
            confirm[idx[-365:]] = True
        if confirm.any() and int(confirm.sum()) >= MIN_SCORE:
            mixed_c = (1.0 - mix_a) * climate + mix_a * p_hat[confirm]
            confirm_brier = float(np.mean((mixed_c - y[confirm]) ** 2))
            confirm_climate = float(np.mean((climate - y[confirm]) ** 2))
            if confirm_brier >= confirm_climate - 1e-12:
                continue
        if score < best[0] - 1e-12:
            best = (score, brier, f"{label}|mix{mix_a:.2f}", cols, kind, kw)
    _score, _brier_val, recipe, cols, kind, kw = best
    mix_a = 0.0
    if recipe.startswith("leftover_") and "|mix" in recipe:
        mix_a = float(recipe.rsplit("|mix", 1)[1])
    train_both = fit_mask | val_mask
    if cols and kind != "climate":
        try:
            model = _fit_kind(kind, raw_x[np.ix_(train_both, cols)], y[train_both], kw)
            raw_p = _proba(model, raw_x[:, cols])
        except ValueError:
            model = None
            raw_p = np.full(len(raw_x), climate_full)
            mix_a = 0.0
            recipe = "climate|mix0.00"
    else:
        model = None
        raw_p = np.full(len(raw_x), climate_full)
    p_all = np.clip((1.0 - mix_a) * climate_full + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
    return model, p_all, recipe


def _region_mainshock(
    *,
    feat: pd.DataFrame,
    columns: list[str],
    region_id: str,
    mag: float,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    hits: np.ndarray,
    raw_hits: np.ndarray,
    design: tuple[np.ndarray, list[str]],
    root: Path | None = None,
) -> dict[str, Any] | None:
    days = pd.DatetimeIndex(feat["issue_time"])
    if int(train_mask.sum()) < MIN_TRAIN or int(hold_mask.sum()) < MIN_SCORE:
        return None
    raw_x, names = design
    if raw_x.shape[1] == 0:
        return None
    y = hits.astype(int)
    fit_mask, val_mask = fit_val_split(days, train_mask)
    val_score = week_mask(val_mask) if val_mask.any() else val_mask
    score_hold = week_mask(hold_mask)
    try:
        _model, p_hat, recipe = _pick_model(
            raw_x,
            names,
            y,
            fit_mask,
            val_score,
            hunt_cols=_hunt_drivers(region_id, root),
            region_id=region_id,
        )
    except ValueError:
        return None
    climate = float(y[train_mask].mean())
    alarm = _alarm(p_hat, hits, days, train_mask, score_hold)
    thin = _top_columns(raw_x[train_mask], y[train_mask], names, 8)
    ranked = [
        {
            "feature": names[i],
            "driver": names[i].split("__lag")[0],
            "lag_days": 0 if "__lag" not in names[i] else int(names[i].rsplit("__lag", 1)[1]),
            "label": _driver_plain(names[i]),
            "importance": 1.0,
        }
        for i in thin
    ]
    why = [row["label"] for row in ranked][:6]
    latest = int(np.argmax(days.to_numpy()))
    leftover_p = float(p_hat[latest])
    hold_brier = _brier(p_hat[score_hold], hits[score_hold].astype(float))
    climate_brier = _brier(np.full(int(score_hold.sum()), climate), hits[score_hold].astype(float))
    held = bool(
        hold_brier is None
        or climate_brier is None
        or hold_brier >= climate_brier
    )
    this_p = float(climate) if held else leftover_p
    return {
        "region": region_id,
        "label": _place(region_id),
        "magnitude_min": mag,
        "n_train": int(train_mask.sum()),
        "n_score": int(score_hold.sum()),
        "n_score_daily": int(hold_mask.sum()),
        "fresh_rate_train": climate,
        "fresh_rate_holdout": float(hits[hold_mask].mean()) if hold_mask.any() else None,
        "raw_rate_holdout": float(raw_hits[hold_mask].mean()) if hold_mask.any() else None,
        "brier": hold_brier,
        "climatology_brier": climate_brier,
        **alarm,
        "recipe": recipe,
        "connections": ranked[:10],
        "why": why,
        "this_week": {
            "issue_day": days[latest].date().isoformat(),
            "p": this_p,
            "leftover_raw_p": leftover_p,
            "leftover_raw_chance_plain": _chance(leftover_p),
            "held_to_climatology": held,
            "chance_plain": _chance(this_p),
            "climatology_p": climate,
            "climatology_chance_plain": _chance(climate),
        },
    }


def _week_sentence(packed: dict[str, Any], tw: dict[str, Any], horizon: int) -> str:
    region_id = str(packed.get("region") or "")
    mag = packed.get("magnitude_min")
    place = "anywhere on Earth" if region_id == "global" else f"in {packed.get('label')}"
    raw_chance = tw.get("leftover_raw_chance_plain") or _chance(tw.get("leftover_raw_p"))
    last_week = packed.get("mode") == "last_week"
    look = (
        "Looking only at the last 7 days of leftover sources, the leftover model's attempt"
        if last_week
        else "The leftover model's attempt"
    )
    put = (
        "Looking only at the last 7 days of leftover sources, the leftover model puts"
        if last_week
        else "The leftover model puts"
    )
    if tw.get("held_to_climatology"):
        return (
            f"{look} for a fresh magnitude {mag:g}+ quake {place} "
            f"in the next {horizon} days is {raw_chance}. "
            f"It has not beaten a constant guess on later years, so the historical rate "
            f"stays {tw['climatology_chance_plain']}."
        )
    return (
        f"{put} the chance of a fresh magnitude {mag:g}+ quake "
        f"{place} in the next {horizon} days at {tw['chance_plain']}. "
        f"A constant guess from old years is {tw['climatology_chance_plain']}."
    )


def _seal_region(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["recipe_leaked"] = recipe_is_leaked(str(out.get("recipe") or ""))
    out["beats_climatology"] = leftover_beats_climatology(out)
    held = leftover_this_week_held(out)
    tw = dict(out.get("this_week") or {})
    climate = tw.get("climatology_p")
    if climate is None:
        climate = out.get("fresh_rate_train")
    climate = float(climate or 0.0)
    leftover_raw = tw.get("leftover_raw_p")
    tw["climatology_p"] = climate
    tw["climatology_chance_plain"] = _chance(climate)
    tw["held_to_climatology"] = held
    tw["p"] = climate if held else float(leftover_raw if leftover_raw is not None else climate)
    tw["chance_plain"] = _chance(tw["p"])
    if leftover_raw is None:
        tw["leftover_raw_p"] = tw["p"]
    tw["leftover_raw_chance_plain"] = _chance(tw.get("leftover_raw_p"))
    out["this_week"] = tw
    return out


def _seal_week(week_row: dict[str, Any], packed: dict[str, Any]) -> dict[str, Any]:
    tw = packed["this_week"]
    out = dict(week_row)
    out["p"] = tw["p"]
    out["chance_plain"] = tw["chance_plain"]
    out["held_to_climatology"] = tw["held_to_climatology"]
    out["leftover_raw_p"] = tw.get("leftover_raw_p")
    out["leftover_raw_chance_plain"] = tw.get("leftover_raw_chance_plain") or _chance(tw.get("leftover_raw_p"))
    out["climatology_p"] = tw["climatology_p"]
    out["climatology_chance_plain"] = tw["climatology_chance_plain"]
    out["why"] = list(out.get("why") or packed.get("why") or [])
    horizon = int(out.get("horizon_days") or packed.get("horizon_days") or HORIZON_DAYS)
    out["sentence"] = _week_sentence(packed, tw, horizon)
    return out


def _seal_report(report: dict[str, Any]) -> dict[str, Any]:
    regions = [_seal_region(r) for r in report.get("regions") or []]
    report["regions"] = regions
    report["n_beats_climatology"] = sum(1 for r in regions if r.get("beats_climatology"))
    by_region = {r["region"]: r for r in regions}
    report["week"] = [
        _seal_week(w, by_region[w["region"]]) if w.get("region") in by_region else w
        for w in report.get("week") or []
    ]
    for exp in report.get("experiments") or []:
        exp_regions = [_seal_region(r) for r in exp.get("regions") or []]
        exp["regions"] = exp_regions
        exp["n_beats_climatology"] = sum(1 for r in exp_regions if r.get("beats_climatology"))
        by_r = {r["region"]: r for r in exp_regions}
        exp["week"] = [
            _seal_week(w, by_r[w["region"]]) if w.get("region") in by_r else w
            for w in exp.get("week") or []
        ]
    if report.get("last_week"):
        lw = dict(report["last_week"])
        lw_regions = [_seal_region(r) for r in lw.get("regions") or []]
        lw["regions"] = lw_regions
        lw["n_beats_climatology"] = sum(1 for r in lw_regions if r.get("beats_climatology"))
        by_lw = {r["region"]: r for r in lw_regions}
        lw["week"] = [
            _seal_week(w, by_lw[w["region"]]) if w.get("region") in by_lw else w
            for w in lw.get("week") or []
        ]
        lw["note"] = _english_last_week(lw)
        report["last_week"] = lw
    report["note"] = _english(report)
    return report


def seal_mainshock_file(root: Path | None = None) -> dict[str, Any] | None:
    pack = read_mainshock(root)
    if not pack:
        return None
    sealed = _seal_report(pack)
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "mainshock.json").write_text(json.dumps(sealed, indent=2, default=str), encoding="utf-8")
    return sealed


def _english(report: dict[str, Any]) -> str:
    bits = [
        f"The leftover learner looked at {report.get('n_features') or 0} series "
        "(space weather, tides, ionosphere, GPS, and the rest — no recent-quake counts) "
        "and tried to predict a fresh large shock in the next week, not an aftershock."
    ]
    regions = report.get("regions") or []
    honest = [r for r in regions if r.get("beats_climatology")]
    if not honest:
        bits.append(
            "On the issued boxes, no leftover mix beat a constant guess on later years "
            "once hunt-selected features are treated as leakage and the loud days have to catch more quakes."
        )
    else:
        best = min(honest, key=lambda r: r["brier"] - r["climatology_brier"])
        bits.append(
            f"Best later-year score: {best['label']} M≥{best['magnitude_min']:g} "
            f"beat a constant guess (error {best['brier']:.3f} vs {best['climatology_brier']:.3f})."
        )
        if best.get("lift") is not None:
            bits.append(
                f"When it raised an alert (top {int(SHARE * 100)}% of old years), "
                f"the hit rate was {best['lift'] * 100:+.1f} pt vs always-on."
            )
    alpine = next((r for r in regions if r.get("region") == "alpine"), None)
    if alpine and alpine.get("recipe_leaked"):
        n_hits = alpine.get("n_hits")
        n_trades = alpine.get("n_trades")
        bits.append(
            f"Alpine’s tiny error dip used hunt-confirmed knobs picked on the same later years"
            + (
                f", and the loud days hit {n_hits} of {n_trades}"
                if n_hits is not None and n_trades is not None
                else ""
            )
            + " — that does not count."
        )
    kerm = next((r for r in regions if r.get("region") == "kermadec"), None)
    if kerm and (kerm.get("lift") or 0) > 0 and not kerm.get("beats_climatology"):
        bits.append(
            f"Kermadec loud days caught more M6s ({kerm['lift'] * 100:+.1f} pt) but the chance "
            "error was worse than a constant — it leaned on season, not a leftover driver."
        )
    for exp in report.get("experiments") or []:
        n = int(exp.get("n_beats_climatology") or 0)
        rows = exp.get("regions") or []
        if exp.get("id") == "fresh_7d_m8":
            row = rows[0] if rows else None
            if row and row.get("beats_climatology"):
                bits.append(
                    f"Earth-wide M8: leftover ranked later weeks a bit better "
                    f"({row.get('n_hits')} hits in {row.get('n_trades')} loud days, "
                    f"{(row.get('lift') or 0) * 100:+.1f} pt) but the error score is almost tied "
                    f"({row['brier']:.4f} vs {row['climatology_brier']:.4f}), "
                    "so this week's M8 number stays at the historical rate."
                )
            else:
                bits.append("Earth-wide M8 leftover hunt did not beat the constant guess.")
        elif exp.get("horizon_days") == 30:
            bits.append(
                "A next 30 days leftover hunt did not beat the constant guess."
                if not n
                else f"A next 30 days leftover hunt beat the constant guess on {n} issued box(es)."
            )
        elif exp.get("id") == "fresh_7d_lastweek":
            bits.append(
                "A last-week-only leftover mode (only the past 7 days of leftover sources, calling next week) "
                + (
                    "did not beat a constant guess."
                    if not n
                    else f"beat a constant guess on {n} issued box(es)."
                )
            )
        else:
            bits.append(
                "A quiet-start next week leftover hunt did not beat the constant guess."
                if not n
                else f"A quiet-start next week leftover hunt beat the constant guess on {n} issued box(es)."
            )
    lw = report.get("last_week")
    if lw and not any(exp.get("id") == "fresh_7d_lastweek" for exp in report.get("experiments") or []):
        n = int(lw.get("n_beats_climatology") or 0)
        bits.append(
            "A last-week-only leftover mode (only the past 7 days of leftover sources, calling next week) "
            + (
                "did not beat a constant guess."
                if not n
                else f"beat a constant guess on {n} issued box(es)."
            )
        )
    return " ".join(bits)


def _english_last_week(pack: dict[str, Any]) -> str:
    n_feat = pack.get("n_features") or 0
    n = int(pack.get("n_beats_climatology") or 0)
    if n:
        return (
            f"Last-week leftover looked at {n_feat} last-7-day leftover series and beat a constant "
            f"guess on {n} issued box(es). Official ledger cards stay calibrated ETAS."
        )
    return (
        f"Last-week leftover looked at {n_feat} last-7-day leftover series and called next week. "
        "It did not beat a constant guess on later years. Official ledger cards stay calibrated ETAS."
    )


def run_mainshock(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = leftover_derived(feat.copy())
    feat["issue_time"] = parse_utc(feat["issue_time"])
    if "cal_doy_sin" not in feat.columns:
        doy = pd.DatetimeIndex(feat["issue_time"]).dayofyear.to_numpy()
        feat["cal_doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        feat["cal_doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    catalog = quakes if quakes is not None else load_comcat(root)
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])

    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    columns = _driver_columns(feat, unconventional_only=True)
    design = _design(feat, columns, MAINSHOCK_LAGS)

    def _collect(
        horizon: int,
        quiet: bool,
        targets: tuple[tuple[str, float], ...] = ISSUED_TARGETS,
        *,
        used_design: tuple[np.ndarray, list[str]] | None = None,
        used_columns: list[str] | None = None,
        mode: str = "all",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        closed = days + pd.Timedelta(days=horizon) <= catalog_end
        train_mask = closed & (days <= utc_ts(train_end))
        hold_mask = closed & (days >= utc_ts(holdout_start))
        packed_rows: list[dict[str, Any]] = []
        week_rows: list[dict[str, Any]] = []
        design_x = used_design if used_design is not None else design
        cols = used_columns if used_columns is not None else columns
        for region_id, mag in targets:
            region = REGIONS[region_id]
            hits = fresh_hits(catalog, region, mag, days, horizon)
            raw = _window_hits(catalog, region, mag, days, horizon)
            tmask = train_mask
            hmask = hold_mask
            if quiet:
                quiet_days = prior_quiet(catalog, region, mag, days, lookback_days=10)
                tmask = tmask & quiet_days
                hmask = hmask & quiet_days
            packed = _region_mainshock(
                feat=feat,
                columns=cols,
                region_id=region_id,
                mag=mag,
                train_mask=tmask,
                hold_mask=hmask,
                hits=hits,
                raw_hits=raw,
                design=design_x,
                root=root,
            )
            if not packed:
                continue
            packed["horizon_days"] = horizon
            packed["quiet_start"] = quiet
            packed["mode"] = mode
            packed_rows.append(packed)
            tw = packed["this_week"]
            week_rows.append(
                {
                    "region": region_id,
                    "label": packed["label"],
                    "magnitude_min": mag,
                    "horizon_days": horizon,
                    "quiet_start": quiet,
                    "mode": mode,
                    "p": tw["p"],
                    "leftover_raw_p": tw.get("leftover_raw_p"),
                    "leftover_raw_chance_plain": tw.get("leftover_raw_chance_plain") or _chance(tw.get("leftover_raw_p")),
                    "held_to_climatology": tw.get("held_to_climatology"),
                    "chance_plain": tw["chance_plain"],
                    "climatology_p": tw["climatology_p"],
                    "climatology_chance_plain": tw["climatology_chance_plain"],
                    "why": packed.get("why") or [],
                    "issue_day": tw["issue_day"],
                    "sentence": _week_sentence(packed, tw, horizon),
                }
            )
        return packed_rows, week_rows

    regions, week = _collect(HORIZON_DAYS, quiet=False)
    experiments: list[dict[str, Any]] = []
    for exp_id, horizon, quiet, targets in (
        ("fresh_30d", 30, False, ISSUED_TARGETS),
        ("fresh_7d_quiet", 7, True, ISSUED_TARGETS),
        ("fresh_7d_m8", 7, False, (("global", 8.0),)),
    ):
        exp_regions, exp_week = _collect(horizon, quiet, targets)
        experiments.append(
            {
                "id": exp_id,
                "horizon_days": horizon,
                "quiet_start": quiet,
                "n_beats_climatology": 0,
                "regions": exp_regions,
                "week": exp_week,
            }
        )
    week_cols = last_week_columns(columns)
    week_design = _design(feat, week_cols, LAST_WEEK_LAGS)
    last_regions, last_week_rows = _collect(
        HORIZON_DAYS,
        quiet=False,
        used_design=week_design,
        used_columns=week_cols,
        mode="last_week",
    )
    last_week = {
        "id": "fresh_7d_lastweek",
        "mode": "last_week",
        "horizon_days": HORIZON_DAYS,
        "lookback_days": 7,
        "writes_issued_p": False,
        "n_features": int(week_design[0].shape[1]),
        "n_beats_climatology": 0,
        "regions": last_regions,
        "week": last_week_rows,
    }

    report = {
        "scanned_at": iso(datetime.now(timezone.utc)),
        "train_end": train_end,
        "holdout_start": holdout_start,
        "horizon_days": HORIZON_DAYS,
        "target": "fresh_mainshock",
        "writes_issued_p": False,
        "n_features": int(design[0].shape[1]),
        "n_regions": len(regions),
        "n_beats_climatology": 0,
        "regions": regions,
        "week": week,
        "last_week": last_week,
        "experiments": experiments,
        "catalog_end": catalog_end.date().isoformat() if hasattr(catalog_end, "date") else str(catalog_end),
    }
    report = _seal_report(report)
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "mainshock.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def run_last_week_mainshock(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score leftover using only the last 7 days of leftover sources. Does not rewrite issued p."""
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = leftover_derived(feat.copy())
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog = quakes if quakes is not None else load_comcat(root)
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    columns = last_week_columns(_driver_columns(feat, unconventional_only=True))
    design = _design(feat, columns, LAST_WEEK_LAGS)
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask = closed & (days <= utc_ts(train_end))
    hold_mask = closed & (days >= utc_ts(holdout_start))
    packed_rows: list[dict[str, Any]] = []
    week_rows: list[dict[str, Any]] = []
    for region_id, mag in ISSUED_TARGETS:
        print(f"last-week leftover {region_id} M{mag:g} ({design[0].shape[1]} series)", flush=True)
        region = REGIONS[region_id]
        hits = fresh_hits(catalog, region, mag, days, HORIZON_DAYS)
        raw = _window_hits(catalog, region, mag, days, HORIZON_DAYS)
        packed = _region_mainshock(
            feat=feat,
            columns=columns,
            region_id=region_id,
            mag=mag,
            train_mask=train_mask,
            hold_mask=hold_mask,
            hits=hits,
            raw_hits=raw,
            design=design,
            root=root,
        )
        if not packed:
            continue
        packed["horizon_days"] = HORIZON_DAYS
        packed["quiet_start"] = False
        packed["mode"] = "last_week"
        packed_rows.append(packed)
        tw = packed["this_week"]
        week_rows.append(
            {
                "region": region_id,
                "label": packed["label"],
                "magnitude_min": mag,
                "horizon_days": HORIZON_DAYS,
                "quiet_start": False,
                "mode": "last_week",
                "p": tw["p"],
                "leftover_raw_p": tw.get("leftover_raw_p"),
                "leftover_raw_chance_plain": tw.get("leftover_raw_chance_plain") or _chance(tw.get("leftover_raw_p")),
                "held_to_climatology": tw.get("held_to_climatology"),
                "chance_plain": tw["chance_plain"],
                "climatology_p": tw["climatology_p"],
                "climatology_chance_plain": tw["climatology_chance_plain"],
                "why": packed.get("why") or [],
                "issue_day": tw["issue_day"],
                "sentence": _week_sentence(packed, tw, HORIZON_DAYS),
            }
        )
    last_week = {
        "id": "fresh_7d_lastweek",
        "mode": "last_week",
        "horizon_days": HORIZON_DAYS,
        "lookback_days": 7,
        "writes_issued_p": False,
        "scanned_at": iso(datetime.now(timezone.utc)),
        "n_features": int(design[0].shape[1]),
        "n_beats_climatology": 0,
        "regions": packed_rows,
        "week": week_rows,
        "catalog_end": catalog_end.date().isoformat() if hasattr(catalog_end, "date") else str(catalog_end),
    }
    existing = read_mainshock(root) or {
        "train_end": train_end,
        "holdout_start": holdout_start,
        "horizon_days": HORIZON_DAYS,
        "target": "fresh_mainshock",
        "writes_issued_p": False,
        "regions": [],
        "week": [],
        "experiments": [],
    }
    existing["last_week"] = last_week
    existing["writes_issued_p"] = False
    existing["catalog_end"] = existing.get("catalog_end") or last_week["catalog_end"]
    sealed = _seal_report(existing)
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "mainshock.json").write_text(json.dumps(sealed, indent=2, default=str), encoding="utf-8")
    return sealed
