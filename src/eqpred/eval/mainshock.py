from __future__ import annotations

"""Predict a fresh big quake from non-catalog sources plus cleaned quake counts.

Train on old years. Score later closed weeks. The target is a declustered
hit — the first large shock, not the swarm that follows. Raw aftershock
counts stay out. Independent-event counts (n_fresh_*) are allowed.
Issued cards use this model's p. Climatology sits beside it as a foil.
"""

import hashlib
import gc
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
from eqpred.eval.compat import saved_fresh
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.skill import MIN_ALARM, information_gain, skill_beats_climate
from eqpred.features.fresh import fresh_derived
from eqpred.eval.decluster import fresh_hits
from eqpred.eval.honest import (
    EVAL_PROTOCOL,
    PROSPECTIVE_EPOCH_START,
    fit_val_split,
    holm_adjust,
    purged_train_hold,
    week_mask,
)
from eqpred.eval.hunt import TARGETS, driver_family, load_hunt_fresh, sequence_cache_matches
from eqpred.eval.strategies import _window_hits
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS, is_issued_box
from eqpred.safe_io import write_json
from eqpred.stamp import parse_utc
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
PRACTICE_TARGETS = tuple((rid, mag) for rid, mag in TARGETS if rid in {"japan", "chile"})
FRESH_MODEL_ID = "fresh.mainshock.v1"

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
    "swarm_f_res_nt": "Swarm magnetic residual",
    "swarm_f_res_range_nt": "Swarm magnetic residual range",
    "swarm_f_res_dbdt_nt": "Swarm magnetic residual dF/dt",
    "swarm_hikurangi_f_res_nt": "Swarm residual over Hikurangi",
    "swarm_alpine_f_res_nt": "Swarm residual over the Alpine Fault",
    "swarm_kermadec_f_res_nt": "Swarm residual over Kermadec",
    "cses_hepd_e": "CSES HEPD electrons",
    "cses_hepd_p": "CSES HEPD protons",
    "strain_ee_nstrain": "borehole strain",
    "strain_ee_range_nstrain": "borehole strain daily range",
    "strain_ee_dbdt_nstrain": "borehole strain dE/dt",
    "radon_bq_m3": "radon",
    "radon_range_bq_m3": "radon daily range",
    "radon_dbdt_bq_m3": "radon dRn/dt",
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
    "n_fresh_m45_7d": "independent M4.5+ quakes this week",
    "n_fresh_m45_30d": "independent M4.5+ quakes this month",
    "n_fresh_m6_7d": "independent M6+ quakes this week",
    "n_fresh_m6_30d": "independent M6+ quakes this month",
    "n_fresh_m7_30d": "independent M7+ quakes this month",
    "days_since_fresh_m7": "days since an independent M7",
    "b_fresh_90d": "fresh-shock b-value (90 days)",
    "b_fresh_365d": "fresh-shock b-value (year)",
    "b_fresh_delta": "fresh-shock b-value change",
    "b_fresh_hikurangi_90d": "Hikurangi fresh-shock b-value",
    "b_fresh_alpine_90d": "Alpine Fault fresh-shock b-value",
    "b_fresh_kermadec_90d": "Kermadec fresh-shock b-value",
    "gnss_strain_nz_dilatation_30d": "NZ GPS dilatation (30 days)",
    "gnss_strain_nz_shear_30d": "NZ GPS shear (30 days)",
    "gnss_strain_hikurangi_dilatation_30d": "Hikurangi GPS dilatation",
    "gnss_strain_alpine_dilatation_30d": "Alpine Fault GPS dilatation",
    "gnss_strain_hikurangi_extension_30d": "Hikurangi GPS baseline extension",
    "gnss_strain_alpine_extension_30d": "Alpine Fault GPS baseline extension",
    "gnss_strain_japan_dilatation_30d": "Japan GPS dilatation",
    "gnss_strain_chile_dilatation_30d": "Chile GPS dilatation",
    "tide_nz_mm": "NZ sea level",
    "tide_nz_30d": "NZ sea-level change (30 days)",
    "tide_wellington_mm": "Wellington sea level",
    "tide_bluff_mm": "Bluff sea level",
    "tide_jackson_mm": "Jackson Bay sea level",
    "tide_chatham_mm": "Chatham Islands sea level",
    "tide_valparaiso_mm": "Valparaíso sea level",
    "tide_kushiro_mm": "Kushiro sea level",
    "insar_los_mm": "InSAR line-of-sight displacement",
    "insar_los_30d": "InSAR line-of-sight change (30 days)",
    "insar_hikurangi_los_mm": "Hikurangi InSAR displacement",
    "insar_alpine_los_mm": "Alpine Fault InSAR displacement",
    "n_fresh_m45_90d": "independent M4.5+ quakes (90 days)",
    "n_fresh_m45_365d": "independent M4.5+ quakes (year)",
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
    "hss_on": "coronal-hole high-speed stream",
    "sep_flag": "solar energetic particles",
    "spe_on": "solar energetic protons",
    "pole_dipole_lat": "geomagnetic dipole pole latitude",
    "solar_eclipse_window": "solar eclipse geometry",
    "lunar_eclipse_geom": "lunar eclipse geometry",
    "solar_max_near": "near solar maximum",
    "solar_min_near": "near solar minimum",
    "solar_descending": "descending solar cycle",
    "solar_days_since_max": "days since solar maximum",
    "coronal_hole_stream": "fast thin solar wind",
    "imf_sector_boundary": "IMF sector boundary",
    "polar_vortex_weak": "weak polar vortex",
    "ap_storm": "magnetic storm (Ap)",
    "flare_x_on": "X-class solar flare",
    "planet_hard_on": "hard planetary aspects",
    "ssc_dst": "sudden Dst commencement",
    "sunspot_north": "northern-hemisphere sunspots",
    "sunspot_south": "southern-hemisphere sunspots",
    "sunspot_asymmetry": "sunspot north–south asymmetry",
    "sunspot_hemisphere_skew": "lopsided sunspot hemispheres",
    "dart_height_m": "ocean-bottom water column",
    "dart_range_m": "ocean-bottom pressure range",
    "dart_d1_m": "ocean-bottom day-to-day change",
    "dart_nz_m": "NZ DART water column",
    "dart_drop": "ocean-bottom pressure drop",
    "volcano_eruption": "volcanic eruption",
    "volcano_vei_max": "eruption VEI",
    "volcano_vei4": "VEI 4+ eruption",
    "volcano_nz": "SW Pacific eruption",
    "volcano_unrest": "volcanic unrest",
    "ulf_high": "ULF / magnetometer pulse",
    "tec_storm": "ionosphere TEC anomaly",
    "sse_on": "Hikurangi slow slip",
    "tremor_on": "Hikurangi / Alpine tremor",
    "tremor_n": "daily tremor count",
    "lfe_n": "daily LFE count",
    "well_d1_m": "groundwater day-to-day change",
    "well_drop": "groundwater drop",
    "scint_s4": "COSMIC-2 scintillation S4",
    "scint_high": "ionosphere scintillation",
    "haarp_on": "HAARP campaign",
    "radon_high": "radon spike",
    "grace_wet": "GRACE water-mass load",
    "so2_high": "volcanic SO2",
    "strain_pulse": "borehole strain pulse",
    "era5_wet": "soil / hydrology load",
    "olr_anomaly": "outgoing longwave anomaly",
    "gnss_dilatation": "GPS strain dilatation",
    "earth_state_z": "unusual Earth-state score",
    "earth_state_unusual": "unusual Earth-state flag",
    "earth_state_mahal": "Earth-state Mahalanobis distance",
    "earth_state_recon": "Earth-state reconstruction error",
    "graph_inflow_hikurangi": "Hikurangi neighbor stress",
    "graph_inflow_alpine": "Alpine Fault neighbor stress",
    "graph_inflow_kermadec": "Kermadec neighbor stress",
    "graph_max_inflow": "strongest neighbor stress",
    "graph_transfer_on": "fault-network stress transfer",
    "analog_p_global": "historical analog chance (Earth-wide)",
    "analog_p_hikurangi": "historical analog chance (Hikurangi)",
    "analog_p_alpine": "historical analog chance (Alpine Fault)",
    "analog_p_kermadec": "historical analog chance (Kermadec)",
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
    "solar_cycle_phase",
    "solar_descending",
    "solar_eclipse_window",
    "hss_on",
    "flare_x_on",
    "spe_on",
    "coronal_hole_stream",
    "ap_storm",
    "ulf_high",
    "tec_storm",
    "sse_on",
    "tremor_on",
    "well_d1_m",
    "scint_s4",
    "haarp_on",
    "dart_d1_m",
    "volcano_unrest",
    "polar_vortex_weak",
    "sunspot_asymmetry",
    "radon_high",
    "grace_wet",
    "so2_high",
    "strain_pulse",
    "era5_wet",
    "olr_anomaly",
    "gnss_dilatation",
    "earth_state_z",
    "graph_inflow_hikurangi",
    "graph_inflow_alpine",
    "graph_inflow_kermadec",
    "graph_transfer_on",
    "analog_p_global",
}

# Frozen source list for the paper-only thin test. Analog / Earth-state /
# graph columns are model outputs, not sensors — they stay off this lock.
LOCKED_THIN = frozenset(
    name
    for name in PREREGISTERED
    if not name.startswith(("analog_p_", "earth_state_", "graph_"))
)
LOCKED_THIN_LAGS = LAST_WEEK_LAGS

MIX_GRID = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)
# "leftover_hunt" is the pre-rename name, still present in saved reports.
HUNT_RECIPE_PREFIXES = ("fresh_hunt", "leftover_hunt")
THIS_WEEK_BRIER_FLOOR = 0.001
POOLED_MAX_RATE_GAP = 0.15
# Every source is allowed. Lags stop at two weeks so leftover TEC/SO2 at day 180 stay out.
ISSUED_LAGS = (0, 1, 2, 3, 5, 7, 10, 14)
# Cards beside the 30-day published chance. They do not replace it.
SHORT_CARD_HORIZONS = (7, 3)
RECENT_CLIMATE_YEARS = 5
RECENT_CLIMATE_DAYS = int(RECENT_CLIMATE_YEARS * 365)
RATE_SPINE_YEARS = (1.0, 2.0, 5.0)
BOOTSTRAP_BLOCK = HORIZON_DAYS
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_ALPHA = 0.05
SCORE_KIND = "daily_circular_block_bootstrap"
SHRINK_FAMILIES = (
    "solar",
    "wave2",
    "geometry",
    "climate",
    "cosmic",
    "earth",
    "mass",
    "fresh_catalog",
)
SHRINK_PER_FAMILY = 8
SHRINK_CAP = 32
SHARED_FAMILIES = (
    *SHRINK_FAMILIES,
    "slowslip",
    "radio",
    "volcano",
    "dust",
    "combo",
    "analog",
    "cusp",
    "coulomb",
    "graph",
    "spf",
    "migration",
    "other",
    "infrasound",
    "pressure",
    "schumann",
    "interface",
    "slab",
    "mechanism",
    "mantle",
    "offshore",
    "rupture",
)
ALL_EXPERIMENT_IDS = (
    "fresh_7d",
    "fresh_7d_quiet",
    "fresh_7d_m8",
    "fresh_30d_m8",
    "fresh_30d_quiet",
)


def _recipe_name(recipe: str) -> str:
    return (recipe or "climate").split("|", 1)[0]


def recipe_is_leaked(recipe: str) -> bool:
    """Hunt-confirmed knobs were picked on the same later years we score."""
    name = _recipe_name(recipe)
    return any(name == prefix or name.startswith(prefix + "_") for prefix in HUNT_RECIPE_PREFIXES)


def fresh_beats_climatology(row: dict[str, Any]) -> bool:
    """Leak still fails. Otherwise information gain or a loud-morning alarm is enough.

    Brier, the week-Brier check, and the bootstrap p-value stay on the report.
    They do not veto a stack that ranks the loud mornings correctly.
    """
    if recipe_is_leaked(str(row.get("recipe") or "")):
        return False
    ig = row.get("information_gain")
    if ig is not None:
        return float(ig) > 1e-12
    lift = row.get("lift")
    share = row.get("time_in_alarm")
    trades = row.get("n_trades")
    enough_alarms = trades is None or int(trades) >= MIN_ALARM
    alarm_ok = (
        enough_alarms
        and lift is not None
        and float(lift) > 0.0
        and (share is None or 0.0 < float(share) < 0.95)
    )
    return bool(alarm_ok)


def week_state_of(row: dict[str, Any], tw: dict[str, Any] | None = None) -> str:
    """rate / moved / beat. Climate stays beside every state."""
    if fresh_beats_climatology(row):
        return "beat"
    card = tw or {}
    p = card.get("p", card.get("fresh_raw_p"))
    climate = card.get("climatology_p")
    close = (
        p is not None
        and climate is not None
        and abs(float(p) - float(climate)) < 1e-4
    )
    if close or p is None:
        return "rate"
    return "moved"


def fresh_this_week_held(row: dict[str, Any]) -> bool:
    """True unless the skill ticket passed. Held boxes still show the model."""
    return not fresh_beats_climatology(row)


ISSUED_WEEK_REGIONS = ("global", "hikurangi", "alpine", "kermadec")


def fresh_week_is_held(pack: dict[str, Any] | None) -> bool:
    """True unless every issued this-week number is an honest fresh-shock call."""
    if not pack:
        return True
    week = [w for w in (pack.get("week") or []) if w.get("region") in ISSUED_WEEK_REGIONS]
    if not week:
        return True
    return all(bool(w.get("held_to_climatology")) for w in week)


def fresh_week_copy(pack: dict[str, Any] | None, *, mode: str = "all") -> dict[str, Any]:
    """Globe wording. The published chance is always the fresh-shock model."""
    held = fresh_week_is_held(pack)
    last_week = mode == "last_week"
    climate_note = (
        " It has not beaten a typical year on later mornings yet, so that typical rate sits beside each number."
        if held
        else ""
    )
    if last_week:
        return {
            "held": held,
            "mode": "last_week",
            "title": "This week’s last-7-day forecast",
        "lede": (
            "Last 7 days of sensors only, calling a first large shock in the next "
            f"{HORIZON_DAYS} days.{climate_note}"
        ),
        "method": (
            f"The mapping from last week’s pattern to the next {HORIZON_DAYS} days is learned "
            "on old years and scored on later closed mornings. It is not a public warning."
        ),
        }
    return {
        "held": held,
        "mode": "all",
        "title": "This week’s forecast",
        "lede": (
            "30-day chance of a first large quake, using sensors known that morning."
            f"{climate_note}"
        ),
        "method": (
            f"The model trains on old years and is scored on later closed daily "
            f"{HORIZON_DAYS}-day windows, with a block bootstrap on the error gap. "
            "It is not a public warning."
        ),
    }


def is_last_week_column(name: str) -> bool:
    """Keep fresh-shock series that only look at the last 7 days, not delayed memory."""
    base = str(name or "").split("__lag", 1)[0]
    if base.startswith("cal_") or base.startswith("x_") or base.startswith("rate_roll_"):
        return False
    return not any(mark in base for mark in LONG_MEMORY_MARKS)


def last_week_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if is_last_week_column(col)]


def is_locked_thin_column(name: str) -> bool:
    """Keep only the frozen pre-registered source knobs."""
    return str(name or "").split("__lag", 1)[0] in LOCKED_THIN


def locked_thin_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if is_locked_thin_column(col)]


def is_issued_driver(name: str) -> bool:
    """Any source, but only recent values. No 27-day+ rolls, 90-day z-scores, or lag-180 leftovers."""
    raw = str(name or "")
    base = raw.split("__lag", 1)[0]
    if "__lag" in raw:
        try:
            lag = int(raw.rsplit("__lag", 1)[1])
        except ValueError:
            lag = 0
        if lag > max(ISSUED_LAGS):
            return False
    if base.startswith(("rate_roll_", "x_")):
        return True
    # Unwin ends in 2019 and most of that year has no fitted gates.
    # It can be scored. It does not write today's issued chance.
    if base.startswith("sdarn_"):
        return False
    # Quiet residuals and the one LAIC conjunction are a hunt. Slow slip stays
    # out of issued probability, so the conjunction that uses it stays out too.
    if base.startswith(("qres_", "laic_", "aardvark_")):
        return False
    return is_last_week_column(raw)


def issued_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if is_issued_driver(col)]


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
    if base.startswith("rate_roll_"):
        rest = base[len("rate_roll_") :]
        window, _, rid = rest.partition("_")
        label = f"{_place(rid)} {window} rolling fresh-shock rate"
    else:
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
    # A morning cannot stop being "quiet" until the qualifying event has
    # actually reached the catalog used by the model.
    time_col = "available_at" if "available_at" in work.columns else "observed_at"
    times = np.sort(parse_utc(work[time_col]).to_numpy(dtype="datetime64[ns]"))
    issue = days.to_numpy(dtype="datetime64[ns]")
    right = np.searchsorted(times, issue, side="right")
    left = np.searchsorted(times, issue - np.timedelta64(lookback_days, "D"), side="right")
    return (right - left) == 0


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


class _BlankAwareScaler(BaseEstimator, TransformerMixin):
    """Scale observed days. A blank morning is a missing flag, not the average day."""

    def fit(self, X, y=None):
        _ = y
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        count = np.sum(np.isfinite(X), axis=0)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            # A column with no observed day is expected; it gets mean 0 and scale 1 below.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(X, axis=0)
            scale = np.sqrt(np.nanvar(X, axis=0))
        self.mean_ = np.where(np.isfinite(mean), mean, 0.0)
        self.scale_ = np.where((count >= 2) & np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        missing = ~np.isfinite(X)
        with np.errstate(all="ignore"):
            scaled = (X - self.mean_) / self.scale_
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        return np.column_stack([scaled, missing.astype(float)])


class _SitOutScaler(BaseEstimator, TransformerMixin):
    """Scale real readings. A blank column adds nothing, and there is no missing-reading flag.

    A stored zero is still a reading. Only a non-finite value sits out.
    """

    def fit(self, X, y=None):
        _ = y
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        count = np.sum(np.isfinite(X), axis=0)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            # A column with no observed day is expected; it gets mean 0 and scale 1 below.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(X, axis=0)
            scale = np.sqrt(np.nanvar(X, axis=0))
        self.mean_ = np.where(np.isfinite(mean), mean, 0.0)
        self.scale_ = np.where((count >= 2) & np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        with np.errstate(all="ignore"):
            scaled = (X - self.mean_) / self.scale_
        return np.where(np.isfinite(scaled), scaled, 0.0)


class _ObservedScaler(BaseEstimator, TransformerMixin):
    """Scale each column on the days that actually have a value.

    Missing months filled with the median before scaling make a normal
    high reading look like 15 standard deviations. The issued model uses
    `_BlankAwareScaler` so a blank morning is not stored as that average.
    """

    def fit(self, X, y=None):
        _ = y
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        count = np.sum(np.isfinite(X), axis=0)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            # A column with no observed day is expected; it gets mean 0 and scale 1 below.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(X, axis=0)
            scale = np.sqrt(np.nanvar(X, axis=0))
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where((count >= 2) & np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        self.mean_ = mean
        self.scale_ = scale
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        with np.errstate(all="ignore"):
            Z = (X - self.mean_) / self.scale_
        return np.where(np.isfinite(Z), Z, 0.0)


def _logit(**kwargs) -> Pipeline:
    sit_out = bool(kwargs.pop("sit_out", False))
    return Pipeline(
        [
            ("sc", _SitOutScaler() if sit_out else _BlankAwareScaler()),
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
            ("sc", _BlankAwareScaler()),
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


def hunt_leaked_name(name: str, hunt_cols: list[str] | set[str] | None) -> bool:
    """True when this series is a hunt-confirmed knob (base name or lagged copy)."""
    if not hunt_cols:
        return False
    leaked = set(hunt_cols)
    raw = str(name)
    base = raw.split("__lag")[0]
    return raw in leaked or base in leaked


def _without_hunt(names: list[str], indices: list[int], hunt_cols: list[str] | set[str] | None) -> list[int]:
    return [i for i in indices if not hunt_leaked_name(names[i], hunt_cols)]


def _usable_indices(x: np.ndarray, mask: np.ndarray, min_n: int = 80) -> list[int]:
    """Keep columns that actually vary in the fit window. Late GNSS/TEC is empty here."""
    if x.size == 0 or not mask.any():
        return []
    sl = x[mask]
    n = np.isfinite(sl).sum(axis=0)
    std = np.zeros(sl.shape[1], dtype=float)
    ok = n >= 2
    if ok.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            std[ok] = np.nanstd(sl[:, ok], axis=0)
    keep = (n >= min_n) & np.isfinite(std) & (std > 1e-12)
    return [int(j) for j in np.flatnonzero(keep)]


def _top_columns(x: np.ndarray, y: np.ndarray, names: list[str], k: int = 48) -> list[int]:
    if x.shape[1] <= k:
        return list(range(x.shape[1]))
    filled, _ = _impute(x, x)
    yf = y.astype(float)
    yf = yf - float(yf.mean())
    ys = float(yf.std())
    if ys < 1e-12:
        return list(range(k))
    xs = filled.std(axis=0)
    xc = filled - filled.mean(axis=0)
    n = float(filled.shape[0])
    scores = np.abs(xc.T @ yf) / (xs * ys * n + 1e-15)
    scores = np.where(xs < 1e-12, 0.0, scores)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    order = np.argsort(-scores)
    return [int(i) for i in order[:k]]


def _prefer_candidate(
    cand_val: float | None,
    current_val: float | None,
    climate_val: float | None,
    cand_rate: float | None = None,
    climate: float | None = None,
    max_rate_gap: float = POOLED_MAX_RATE_GAP,
) -> bool:
    """Take a sibling only when it beats the current call and the rate spine, and stays near this box's rate."""
    if cand_val is None or current_val is None or climate_val is None:
        return False
    if not (cand_val < current_val - 1e-12 and cand_val < climate_val - 1e-12):
        return False
    if cand_rate is not None and climate is not None:
        if abs(float(cand_rate) - float(climate)) > max_rate_gap:
            return False
    return True


def _prefer_pooled(
    pool_val: float | None,
    ind_val: float | None,
    climate_val: float | None,
    pool_rate: float | None = None,
    climate: float | None = None,
    max_rate_gap: float = POOLED_MAX_RATE_GAP,
) -> bool:
    """Use the stacked model only when it beats specialist and climate, and stays near this box's rate."""
    return _prefer_candidate(
        pool_val,
        ind_val,
        climate_val,
        cand_rate=pool_rate,
        climate=climate,
        max_rate_gap=max_rate_gap,
    )


def _calibrate_mean(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift p so the mean on `mask` matches the hit rate. Ranking stays; loudness is fixed."""
    if p.size == 0 or not mask.any():
        return p
    target = float(np.mean(y[mask]))
    current = float(np.mean(p[mask]))
    if not math.isfinite(target) or not math.isfinite(current):
        return p
    return np.clip(p + (target - current), 1e-6, 1.0 - 1e-6)


def _calibrate_to_roll(
    p: np.ndarray,
    roll: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Each morning: keep p's ranking, set the mean to that morning's rolling rate."""
    p = np.asarray(p, dtype=float)
    roll = np.asarray(roll, dtype=float)
    if p.size == 0:
        return p
    use = np.isfinite(p)
    if mask is not None:
        use = use & np.asarray(mask, dtype=bool)
    if not use.any():
        return p
    current = float(np.mean(p[use]))
    if not math.isfinite(current):
        return p
    residual = p - current
    out = np.where(np.isfinite(roll), roll + residual, p)
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def attach_rate_spine(
    feat: pd.DataFrame,
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    horizon: int,
    targets: tuple[tuple[str, float], ...] | None = None,
) -> pd.DataFrame:
    """Point-in-time 1y/2y/5y declustered box rates. Causal: only mornings before today."""
    out = feat.copy()
    for region_id, mag in targets or TARGETS:
        region = REGIONS.get(region_id)
        if region is None:
            continue
        hits = fresh_hits(catalog, region, mag, days, horizon).astype(float)
        for years in RATE_SPINE_YEARS:
            out[f"rate_roll_{int(years)}y_{region_id}"] = rolling_climate_series(
                hits, days, years=years, horizon=horizon
            )
    return out


def block_bootstrap_beats(
    p: np.ndarray,
    climate: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    block: int = BOOTSTRAP_BLOCK,
    n_boot: int = BOOTSTRAP_DRAWS,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 7,
) -> dict[str, Any]:
    """Does p beat climate on overlapping days? Moving-block bootstrap of the Brier gap."""
    idx = np.flatnonzero(
        np.asarray(mask, dtype=bool)
        & np.isfinite(p)
        & np.isfinite(climate)
        & np.isfinite(y)
    )
    empty = {"beats": False, "p_value": None, "gap": None, "n_days": int(idx.size)}
    if idx.size < MIN_SCORE:
        return empty
    pp = np.asarray(p, dtype=float)[idx]
    cc = np.asarray(climate, dtype=float)[idx]
    yy = np.asarray(y, dtype=float)[idx]
    gap = float(np.mean((cc - yy) ** 2) - np.mean((pp - yy) ** 2))
    n = int(pp.size)
    width = max(int(block), 1)
    n_blocks = int(np.ceil(n / width))
    rng = np.random.default_rng(int(seed))
    worse = 0
    for _ in range(int(n_boot)):
        chosen = rng.integers(0, n, size=n_blocks)
        chunks_p: list[np.ndarray] = []
        chunks_c: list[np.ndarray] = []
        chunks_y: list[np.ndarray] = []
        for start in chosen:
            take = (np.arange(width) + int(start)) % n
            chunks_p.append(pp[take])
            chunks_c.append(cc[take])
            chunks_y.append(yy[take])
        boot_p = np.concatenate(chunks_p)[:n]
        boot_c = np.concatenate(chunks_c)[:n]
        boot_y = np.concatenate(chunks_y)[:n]
        boot_gap = float(np.mean((boot_c - boot_y) ** 2) - np.mean((boot_p - boot_y) ** 2))
        if boot_gap <= 0:
            worse += 1
    p_value = (worse + 1) / (int(n_boot) + 1)
    return {
        "beats": bool(gap > 0 and p_value < float(alpha)),
        "p_value": float(p_value),
        "gap": gap,
        "n_days": n,
    }


def _apply_tournament(
    *,
    wide_p: np.ndarray,
    wide_recipe: str,
    y: np.ndarray,
    val_mask: np.ndarray,
    train_mask: np.ndarray,
    climate: float,
    roll: np.ndarray,
    pooled_p: np.ndarray | None = None,
    pooled_recipe: str | None = None,
    shrunk_p: np.ndarray | None = None,
    shrunk_recipe: str | None = None,
) -> tuple[np.ndarray, str, str]:
    """Pick wide, pooled, or shrunk. Rate spine is the foil they must beat, not an issued winner."""
    foil = np.where(np.isfinite(roll), roll, climate)
    target = y[val_mask].astype(float) if val_mask.any() else None

    def _val(series: np.ndarray) -> float | None:
        return None if target is None else _brier(series[val_mask], target)

    climate_val = _val(foil)
    wide_val = _val(wide_p)
    winner = "independent"
    best_p = wide_p
    best_recipe = wide_recipe
    best_val = wide_val
    cands: list[tuple[str, np.ndarray, str]] = []
    if pooled_p is not None and len(pooled_p) == len(wide_p):
        cands.append(("pooled", np.asarray(pooled_p, dtype=float), pooled_recipe or "pooled"))
    if shrunk_p is not None and len(shrunk_p) == len(wide_p):
        cands.append(("shrunk", np.asarray(shrunk_p, dtype=float), shrunk_recipe or "fresh_shrunk"))
    for name, series, rec in cands:
        cand_val = _val(series)
        cand_rate = float(np.nanmean(series[train_mask])) if train_mask.any() else None
        if not _prefer_candidate(cand_val, wide_val, climate_val, cand_rate=cand_rate, climate=climate):
            continue
        if best_val is None or (cand_val is not None and cand_val < best_val - 1e-12):
            winner = name
            best_p = series
            best_val = cand_val
            best_recipe = f"{rec}|{wide_recipe}" if name == "pooled" else rec
    return best_p, winner, best_recipe


def recent_mask(days: pd.DatetimeIndex, mask: np.ndarray, years: float = RECENT_CLIMATE_YEARS) -> np.ndarray:
    """Last `years` of a boolean mask. Falls back to the full mask if too short."""
    if mask is None or not np.any(mask):
        return mask
    end = days[mask].max()
    start = end - pd.DateOffset(years=years)
    recent = np.asarray(mask) & (days >= start)
    if int(recent.sum()) < MIN_SCORE:
        return np.asarray(mask)
    return recent


def recent_climate(
    y: np.ndarray,
    days: pd.DatetimeIndex,
    mask: np.ndarray,
    years: float = RECENT_CLIMATE_YEARS,
) -> float:
    use = recent_mask(days, mask, years)
    if use is None or not np.any(use):
        return float(np.mean(y)) if len(y) else 0.0
    return float(np.mean(y[use]))


def rolling_climate_series(
    hits: np.ndarray,
    days: pd.DatetimeIndex,
    years: float = RECENT_CLIMATE_YEARS,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Point-in-time hit rate of already-closed forecast windows.

    `hits[t]` labels earthquakes in `[t, t+horizon)`. Those labels are delayed
    by the full horizon so a quake cannot move this morning's rate until its
    window has closed.
    """
    s = pd.Series(np.asarray(hits, dtype=float), index=pd.DatetimeIndex(days))
    window = f"{int(years * 365)}D"
    past = s.shift(max(int(horizon), 1))
    roll = past.rolling(window, min_periods=MIN_SCORE).mean()
    expand = past.expanding(min_periods=MIN_SCORE).mean()
    return roll.fillna(expand).to_numpy(dtype=float)


def _recent_slice(mask: np.ndarray, n: int = RECENT_CLIMATE_DAYS) -> np.ndarray:
    out = np.zeros(len(mask), dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    out[idx[-n:] if idx.size > n else idx] = True
    if int(out.sum()) < MIN_SCORE:
        return np.asarray(mask)
    return out


def shrink_columns(columns: list[str]) -> list[str]:
    keep = set(SHRINK_FAMILIES)
    return [col for col in columns if driver_family(str(col).split("__lag", 1)[0]) in keep]


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
            "b_fresh",
            "rate_roll",
            "gnss_strain",
            "tide_",
            "insar_",
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
            "b_fresh",
            "rate_roll",
            "gnss_strain",
            "tide_",
            "insar_",
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


def _mix(
    p: np.ndarray,
    y: np.ndarray,
    climate: float,
    mask: np.ndarray,
    roll: np.ndarray | None = None,
) -> tuple[float, float]:
    """Smallest shrink weight whose mix beats climate on IG or rank/alarm.

    a=0 stays when nothing on MIX_GRID wins. Brier of the chosen mix is
    returned as a report, not as the reason to travel.
    """
    if not mask.any():
        return 0.0, float("nan")
    target = y[mask].astype(float)
    raw = np.asarray(p, dtype=float)[mask]
    if roll is not None:
        foil = np.where(np.isfinite(roll), roll, climate)
        foil_m = np.asarray(foil[mask], dtype=float)
    else:
        foil_m = np.full(target.shape, float(climate))
    base = float(np.mean((foil_m - target) ** 2))
    for a in MIX_GRID:
        if a <= 0:
            continue
        mixed = (1.0 - a) * foil_m + a * raw
        ticket = skill_beats_climate(mixed, target, foil_m, train_p=raw)
        if ticket["beats"]:
            return float(a), float(np.mean((mixed - target) ** 2))
    return 0.0, base


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
    recent_n: int = RECENT_CLIMATE_DAYS,
    days: pd.DatetimeIndex | None = None,
    horizon: int = HORIZON_DAYS,
) -> tuple[Any, np.ndarray, str]:
    cal_fit = _recent_slice(fit_mask, recent_n) if fit_mask.any() else fit_mask
    climate = float(y[cal_fit].mean()) if np.any(cal_fit) else (float(y[fit_mask].mean()) if fit_mask.any() else float(y.mean()))
    train_both = fit_mask | val_mask
    cal_both = _recent_slice(train_both, recent_n) if train_both.any() else train_both
    climate_full = float(y[cal_both].mean()) if np.any(cal_both) else climate
    roll = (
        rolling_climate_series(y.astype(float), days, horizon=horizon)
        if days is not None and len(days) == len(y)
        else None
    )
    print(
        f"fresh-shock pick {region_id} {raw_x.shape[0]} mornings × {raw_x.shape[1]} series",
        flush=True,
    )
    usable = _without_hunt(names, _usable_indices(raw_x, fit_mask), hunt_cols)
    print(f"fresh-shock pick {region_id} usable {len(usable)}", flush=True)
    ranked = _pool_top(raw_x, y, fit_mask, usable, 160)
    score_mask = val_mask if val_mask.any() else fit_mask
    if roll is not None:
        foil_score = np.where(np.isfinite(roll[score_mask]), roll[score_mask], climate)
        climate_brier = float(np.mean((foil_score - y[score_mask]) ** 2))
    else:
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
        ("fresh_thin", ranked[:48], "tree", {"max_depth": 3, "min_samples_leaf": 30}),
        ("fresh_thin_logit", ranked[:48], "logit", {"C": 0.25}),
        ("fresh_wide", ranked[:96], "tree", {"max_depth": 2, "min_samples_leaf": 80, "l2_regularization": 2.0}),
        ("fresh_wide_logit", ranked[:96], "logit", {"C": 0.15}),
        ("fresh_local", _pool_top(raw_x, y, fit_mask, _local_pool(names, region_id, usable), 64), "logit", {"C": 0.2}),
        ("fresh_season", [i for i in usable if names[i].split("__lag")[0].startswith("cal_")], "logit", {"C": 1.0}),
        ("fresh_prereg", [i for i in usable if names[i].split("__lag")[0] in PREREGISTERED], "logit", {"C": 0.2}),
        ("fresh_joint", ranked[:96], "logit", {"C": 0.12}),
        ("fresh_max", ranked[:160], "tree", {"max_depth": 2, "min_samples_leaf": 80, "l2_regularization": 2.0}),
        ("fresh_combo", _pool_top(raw_x, y, fit_mask, _family_pool(names, "combo", usable), 24), "logit", {"C": 0.25}),
        (
            "fresh_delay",
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
    # Hunt-confirmed knobs were picked on the same later years. They stay out of usable.
    for fam in ("solar", "geometry", "wave2", "mass", "climate", "combo", "cusp", "cosmic", "earth", "dust", "volcano", "slowslip"):
        pool = _family_pool(names, fam, usable)
        if len(pool) >= 4:
            candidates.append((f"fresh_{fam}", _pool_top(raw_x, y, fit_mask, pool, 32), "logit", {"C": 0.3}))
    y_fit = y[fit_mask]
    force = [i for i, name in enumerate(names) if str(name).startswith("pool_")]
    if int(fit_mask.sum()) < MIN_TRAIN or int(np.unique(y_fit).size) < 2:
        if roll is not None:
            p_all = np.where(np.isfinite(roll), roll, climate_full)
        else:
            p_all = np.full(len(raw_x), climate_full)
        return None, p_all, "climate|mix0.00"
    for label, cols, kind, kw in candidates:
        cols = [c for c in cols if c in set(usable)]
        cols = list(dict.fromkeys([*cols, *[i for i in force if i in usable]]))
        if len(cols) < 2:
            continue
        try:
            model = _fit_kind(kind, raw_x[np.ix_(fit_mask, cols)], y_fit, kw)
            p_hat = _proba(model, raw_x[:, cols])
        except ValueError:
            continue
        mix_a, brier = _mix(p_hat, y, climate, score_mask, roll=roll)
        score = brier - climate_brier
        if score >= -1e-12:
            continue
        confirm = np.zeros(len(fit_mask), dtype=bool)
        idx = np.flatnonzero(fit_mask)
        if idx.size > 365:
            confirm[idx[-365:]] = True
        if confirm.any() and int(confirm.sum()) >= MIN_SCORE:
            if roll is not None:
                foil_c = np.where(np.isfinite(roll[confirm]), roll[confirm], climate)
                mixed_c = (1.0 - mix_a) * foil_c + mix_a * p_hat[confirm]
                confirm_climate = float(np.mean((foil_c - y[confirm]) ** 2))
            else:
                mixed_c = (1.0 - mix_a) * climate + mix_a * p_hat[confirm]
                confirm_climate = float(np.mean((climate - y[confirm]) ** 2))
            confirm_brier = float(np.mean((mixed_c - y[confirm]) ** 2))
            if confirm_brier >= confirm_climate - 1e-12:
                continue
        if score < best[0] - 1e-12:
            best = (score, brier, f"{label}|mix{mix_a:.2f}", cols, kind, kw)
            print(f"fresh-shock pick {region_id} best {best[2]}", flush=True)
    _score, _brier_val, recipe, cols, kind, kw = best
    mix_a = 0.0
    if recipe.startswith("fresh_") and "|mix" in recipe:
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
    if roll is not None:
        foil_all = np.where(np.isfinite(roll), roll, climate_full)
        p_all = np.clip((1.0 - mix_a) * foil_all + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
    else:
        p_all = np.clip((1.0 - mix_a) * climate_full + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
        p_all = _calibrate_mean(p_all, y, cal_both if np.any(cal_both) else (train_both if train_both.any() else fit_mask))
    print(f"fresh-shock pick {region_id} done ({recipe})", flush=True)
    return model, p_all, recipe


def _pick_shrunk(
    raw_x: np.ndarray,
    names: list[str],
    y: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    region_id: str = "global",
    days: pd.DatetimeIndex | None = None,
    horizon: int = HORIZON_DAYS,
    hunt_cols: list[str] | None = None,
) -> tuple[Any, np.ndarray, str]:
    """Family bake-off, then one logit on at most SHRINK_CAP columns. Sidecar unless a tournament admits it."""
    cal_fit = _recent_slice(fit_mask) if fit_mask.any() else fit_mask
    climate = float(y[cal_fit].mean()) if np.any(cal_fit) else (float(y.mean()) if len(y) else 0.0)
    train_both = fit_mask | val_mask
    cal_both = _recent_slice(train_both) if train_both.any() else train_both
    climate_full = float(y[cal_both].mean()) if np.any(cal_both) else climate
    roll = (
        rolling_climate_series(y.astype(float), days, horizon=horizon)
        if days is not None and len(days) == len(y)
        else None
    )
    print(f"fresh-shock shrink {region_id} {raw_x.shape[0]} mornings × {raw_x.shape[1]} series", flush=True)
    usable = _without_hunt(names, _usable_indices(raw_x, fit_mask), hunt_cols)
    score_mask = val_mask if val_mask.any() else fit_mask
    if roll is not None and score_mask.any():
        foil_score = np.where(np.isfinite(roll[score_mask]), roll[score_mask], climate)
        climate_brier = float(np.mean((foil_score - y[score_mask]) ** 2))
    else:
        climate_brier = float(np.mean((climate - y[score_mask]) ** 2)) if score_mask.any() else 1.0
    kept: list[int] = []
    beaten: list[str] = []
    y_fit = y[fit_mask]
    if int(fit_mask.sum()) < MIN_TRAIN or int(np.unique(y_fit).size) < 2:
        if roll is not None:
            p_all = np.where(np.isfinite(roll), roll, climate_full)
        else:
            p_all = np.full(len(raw_x), climate_full)
        return None, p_all, "climate|mix0.00"
    for fam in SHRINK_FAMILIES:
        pool = _family_pool(names, fam, usable)
        cols = _pool_top(raw_x, y, fit_mask, pool, SHRINK_PER_FAMILY)
        cols = [c for c in cols if c in set(usable)]
        if len(cols) < 2:
            continue
        try:
            model = _fit_kind("logit", raw_x[np.ix_(fit_mask, cols)], y_fit, {"C": 0.3})
            p_hat = _proba(model, raw_x[:, cols])
        except ValueError:
            continue
        mix_a, brier = _mix(p_hat, y, climate, score_mask, roll=roll)
        if brier < climate_brier - 1e-12:
            beaten.append(fam)
            kept.extend(cols)
    kept = list(dict.fromkeys(kept))[:SHRINK_CAP]
    if len(kept) < 2:
        if roll is not None:
            p_all = np.where(np.isfinite(roll), roll, climate_full)
        else:
            p_all = np.full(len(raw_x), climate_full)
        return None, p_all, "climate|mix0.00"
    try:
        model = _fit_kind("logit", raw_x[np.ix_(train_both, kept)], y[train_both], {"C": 0.2})
        raw_p = _proba(model, raw_x[:, kept])
    except ValueError:
        if roll is not None:
            p_all = np.where(np.isfinite(roll), roll, climate_full)
        else:
            p_all = np.full(len(raw_x), climate_full)
        return None, p_all, "climate|mix0.00"
    mix_a, _brier = _mix(raw_p, y, climate, score_mask, roll=roll)
    if roll is not None:
        foil_all = np.where(np.isfinite(roll), roll, climate_full)
        p_all = np.clip((1.0 - mix_a) * foil_all + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
    else:
        p_all = np.clip((1.0 - mix_a) * climate_full + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
        p_all = _calibrate_mean(p_all, y, cal_both if np.any(cal_both) else train_both)
    fams = "+".join(beaten) if beaten else "none"
    recipe = f"fresh_shrunk_{fams}|mix{mix_a:.2f}"
    print(f"fresh-shock shrink {region_id} done ({recipe}, {len(kept)} cols)", flush=True)
    return model, p_all, recipe


def _pick_locked(
    raw_x: np.ndarray,
    names: list[str],
    y: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    region_id: str = "global",
    days: pd.DatetimeIndex | None = None,
    horizon: int = HORIZON_DAYS,
    hunt_cols: list[str] | None = None,
) -> tuple[Any, np.ndarray, str]:
    """One logit on every usable locked column. No ranking, no tournament."""
    cal_fit = _recent_slice(fit_mask) if fit_mask.any() else fit_mask
    climate = float(y[cal_fit].mean()) if np.any(cal_fit) else (float(y.mean()) if len(y) else 0.0)
    train_both = fit_mask | val_mask
    cal_both = _recent_slice(train_both) if train_both.any() else train_both
    climate_full = float(y[cal_both].mean()) if np.any(cal_both) else climate
    roll = (
        rolling_climate_series(y.astype(float), days, horizon=horizon)
        if days is not None and len(days) == len(y)
        else None
    )
    print(f"fresh-shock locked {region_id} {raw_x.shape[0]} mornings × {raw_x.shape[1]} series", flush=True)
    usable = _without_hunt(names, _usable_indices(raw_x, fit_mask), hunt_cols)
    cols = [i for i in usable if is_locked_thin_column(names[i])]
    score_mask = val_mask if val_mask.any() else fit_mask
    y_fit = y[fit_mask]
    if len(cols) < 2 or int(fit_mask.sum()) < MIN_TRAIN or int(np.unique(y_fit).size) < 2:
        if roll is not None:
            p_all = np.where(np.isfinite(roll), roll, climate_full)
        else:
            p_all = np.full(len(raw_x), climate_full)
        print(f"fresh-shock locked {region_id} done (climate|mix0.00)", flush=True)
        return None, p_all, "climate|mix0.00"
    try:
        model = _fit_kind("logit", raw_x[np.ix_(fit_mask, cols)], y_fit, {"C": 0.25})
        raw_p = _proba(model, raw_x[:, cols])
    except ValueError:
        if roll is not None:
            p_all = np.where(np.isfinite(roll), roll, climate_full)
        else:
            p_all = np.full(len(raw_x), climate_full)
        return None, p_all, "climate|mix0.00"
    mix_a, _brier = _mix(raw_p, y, climate, score_mask, roll=roll)
    try:
        model = _fit_kind("logit", raw_x[np.ix_(train_both, cols)], y[train_both], {"C": 0.25})
        raw_p = _proba(model, raw_x[:, cols])
    except ValueError:
        model = None
        raw_p = np.full(len(raw_x), climate_full)
        mix_a = 0.0
    if roll is not None:
        foil_all = np.where(np.isfinite(roll), roll, climate_full)
        p_all = np.clip((1.0 - mix_a) * foil_all + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
    else:
        p_all = np.clip((1.0 - mix_a) * climate_full + mix_a * raw_p, 1e-6, 1.0 - 1e-6)
        p_all = _calibrate_mean(p_all, y, cal_both if np.any(cal_both) else train_both)
    recipe = f"fresh_locked|mix{mix_a:.2f}"
    print(f"fresh-shock locked {region_id} done ({recipe}, {len(cols)} cols)", flush=True)
    return model, p_all, recipe


def _fit_pooled(
    *,
    feat: pd.DataFrame,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    horizon: int,
    root: Path | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    """One model on stacked issued boxes, with a region flag and magnitude."""
    raw_x, names = design
    n = len(days)
    print(f"fresh-shock pooled stack ({n} mornings × {raw_x.shape[1]} series)", flush=True)
    usable = _usable_indices(raw_x, train_mask)
    # _pick_model never keeps more than 160 series. Stacking the full usable
    # matrix four times is what killed the last 30-day retrain.
    if len(usable) > 192:
        global_y = fresh_hits(catalog, REGIONS["global"], 7.0, days, horizon).astype(int)
        usable = _pool_top(raw_x, global_y, train_mask, usable, 192)
    thin = raw_x[:, usable] if usable else raw_x
    thin_names = [names[i] for i in usable] if usable else list(names)
    print(f"fresh-shock pooled usable {thin.shape[1]} series", flush=True)
    extra_names = [f"pool_is_{rid}" for rid, _mag in ISSUED_TARGETS] + ["pool_mag"]
    stacked_x = []
    stacked_y = []
    for i, (region_id, mag) in enumerate(ISSUED_TARGETS):
        print(f"fresh-shock pooled hits {region_id}", flush=True)
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(int)
        onehot = np.zeros((n, len(extra_names)))
        onehot[:, i] = 1.0
        onehot[:, -1] = float(mag)
        stacked_x.append(np.column_stack([thin, onehot]))
        stacked_y.append(hits)
    x_all = np.vstack(stacked_x)
    y_all = np.concatenate(stacked_y)
    print(f"fresh-shock pooled fit {x_all.shape[0]} rows × {x_all.shape[1]} series", flush=True)
    fit0, val0 = fit_val_split(days, train_mask, purge=horizon)
    fit_s = np.tile(fit0, len(ISSUED_TARGETS))
    val_s = np.tile(val0, len(ISSUED_TARGETS))
    _model, p_all, recipe = _pick_model(
        x_all,
        thin_names + extra_names,
        y_all,
        fit_s,
        val_s,
        hunt_cols=[],
        region_id="global",
        recent_n=RECENT_CLIMATE_DAYS * len(ISSUED_TARGETS),
    )
    out: dict[str, np.ndarray] = {}
    for i, (region_id, _mag) in enumerate(ISSUED_TARGETS):
        out[region_id] = np.asarray(p_all[i * n : (i + 1) * n], dtype=float)
    _ = hold_mask
    print(f"fresh-shock pooled done ({recipe})", flush=True)
    return out, f"pooled|{recipe}"


SEALED_PROMOTE_START = "2019-01-01"
SEALED_PROMOTE_END = "2024-01-01"


def sealed_holdout_mask(
    days: pd.DatetimeIndex,
    hold_mask: np.ndarray,
    start: str = SEALED_PROMOTE_START,
    end: str = SEALED_PROMOTE_END,
) -> np.ndarray:
    """2019–2023 slice of later years. Promote cannot peek at 2024–now."""
    idx = pd.DatetimeIndex(days)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC")
    return np.asarray(hold_mask, dtype=bool) & (idx >= lo) & (idx < hi)


def leftover_promote_ok(val_beats: bool, sealed_beats: bool | None = None) -> bool:
    """Val skill only. Sealed years are printed, never used to pick."""
    _ = sealed_beats
    return bool(val_beats)


def _shared_family_indices(
    raw_x: np.ndarray,
    names: list[str],
    y_rank: np.ndarray,
    fit_mask: np.ndarray,
    hunt_cols: list[str] | None = None,
) -> tuple[list[int], list[str]]:
    """Top columns per family that actually has data. Empty families stay out; live ones stay in."""
    usable = _without_hunt(names, _usable_indices(raw_x, fit_mask), hunt_cols)
    kept: list[int] = []
    families: list[str] = []
    sl = raw_x[fit_mask] if fit_mask.any() else raw_x
    for fam in SHARED_FAMILIES:
        pool = _family_pool(names, fam, usable)
        live: list[int] = []
        for idx in pool:
            col = sl[:, idx]
            finite = col[np.isfinite(col)]
            if finite.size >= 20 and float(np.std(finite)) > 1e-12:
                live.append(idx)
        if not live:
            continue
        top = _pool_top(raw_x, y_rank, fit_mask, live, SHRINK_PER_FAMILY)
        if not top:
            continue
        kept.extend(top)
        families.append(fam)
    return list(dict.fromkeys(kept)), families


def _promoted_leftover_score(
    feat: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    catalog: pd.DataFrame,
    horizon: int,
) -> list[tuple[str, np.ndarray]]:
    """Hikurangi M5 and Kermadec M6 leftover. Val skill vs the regime clock.

    Earth-wide M7 is printed and not stacked. Sealed years are printed and
    not used to pick. Returns named residual columns, possibly empty.
    """
    try:
        from eqpred.eval.leftover import fit_leftover_region, leftover_columns
    except Exception:
        return []
    cols = leftover_columns(feat)
    if len(cols) < 8:
        return []
    fit_mask, val_mask = fit_val_split(days, train_mask, purge=horizon)
    if int(fit_mask.sum()) < MIN_TRAIN:
        return []
    sealed = sealed_holdout_mask(days, hold_mask)
    promoted: list[tuple[str, np.ndarray]] = []
    questions = (("hikurangi", 5.0), ("alpine", 5.0), ("kermadec", 6.0), ("global", 7.0))
    for region_id, mag in questions:
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(float)
        if int(np.unique(hits[fit_mask]).size) < 2:
            print(f"fresh-shock leftover {region_id} M{mag:g} skipped (one class)", flush=True)
            continue
        packed = fit_leftover_region(
            feat,
            hits,
            region_id=region_id,
            mag=mag,
            train_mask=fit_mask,
            hold_mask=val_mask,
            leftover_cols=cols,
        )
        gc.collect()
        leftover_p = packed.get("leftover_p")
        leftover_hat = packed.get("leftover_hat")
        regime = packed.get("regime")
        if leftover_p is None or leftover_hat is None or regime is None:
            continue
        leftover_p = np.asarray(leftover_p, dtype=float)
        leftover_hat = np.asarray(leftover_hat, dtype=float)
        regime = np.asarray(regime, dtype=float)

        def _beats(mask: np.ndarray, _p=leftover_p, _r=regime) -> bool:
            if int(mask.sum()) < MIN_SCORE:
                return False
            ticket = skill_beats_climate(
                _p[mask], hits[mask], _r[mask], train_p=_p[fit_mask]
            )
            return bool(ticket["beats"])

        val_ok = _beats(val_mask)
        sealed_ok = _beats(sealed)
        print(
            f"fresh-shock leftover {region_id} M{mag:g} "
            f"val_skill={val_ok} sealed_skill={sealed_ok} in issued stack",
            flush=True,
        )
        promoted.append((f"leftover_p_{region_id}", np.clip(leftover_p, 1e-6, 1.0 - 1e-6)))
    return promoted


def _append_column(stack: np.ndarray, names: list[str], name: str, values) -> tuple[np.ndarray, list[str]]:
    col = np.asarray(values, dtype=float).reshape(-1)
    if stack.ndim != 2 or col.size != stack.shape[0] or name in names:
        return stack, names
    col = np.where(np.isfinite(col), col, np.nan).reshape(-1, 1)
    stack = np.column_stack([stack, col]) if stack.shape[1] else col
    names.append(name)
    return stack, names


def _useful_mainshock_signals(
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    root: Path | None,
) -> list[tuple[str, np.ndarray]]:
    """Point-in-time signals from the other tracks. One column each. DAS clips are not a daily series."""
    out: list[tuple[str, np.ndarray]] = []
    n = len(days)
    try:
        from eqpred.eval.hunt import _etas_p

        for region_id, mag in ISSUED_TARGETS:
            out.append((f"etas_p_{region_id}", np.asarray(_etas_p(catalog, days, region_id, mag, root), dtype=float)))
    except Exception as exc:
        print(f"fresh-shock etas signal skipped: {exc}", flush=True)
    try:
        from eqpred.eval.eepas import eepas_daily_rate
        from eqpred.models.gr import load_gr, rates_to_p
        from eqpred.models.predict import load_models

        etas, _mod = load_models(root)
        gr = load_gr(root)
        rate = eepas_daily_rate(catalog, days, float(etas.m0))
        eepas_p = rates_to_p(rate, 7.0, float(gr.get("m0") or etas.m0), float(gr.get("b") or 1.0), HORIZON_DAYS)
        out.append(("eepas_p_global", np.asarray(eepas_p, dtype=float)))
    except Exception as exc:
        print(f"fresh-shock eepas signal skipped: {exc}", flush=True)
    try:
        from eqpred.eval.strategies import _window_hits

        japan = _window_hits(catalog, REGIONS["japan"], 5.0, days, 7).astype(float)
        chile = _window_hits(catalog, REGIONS["chile"], 5.0, days, 7).astype(float)
        prior = np.zeros(n, dtype=float)
        if n > 7:
            prior[7:] = np.maximum(japan, chile)[:-7]
        out.append(("transfer_japan_chile_m5", prior))
    except Exception as exc:
        print(f"fresh-shock transfer signal skipped: {exc}", flush=True)
    try:
        from eqpred.features.sequence import sequence_frame

        frame = sequence_frame(pd.Series(days), quakes=catalog, root=root)
        lights = {"none": 0.0, "green": 1.0, "yellow": 2.0, "red": 3.0}
        for col in frame.columns:
            if col == "issue_time" or not str(col).startswith("seq_"):
                continue
            series = frame[col]
            if series.dtype == object:
                values = series.astype(str).map(lambda v: lights.get(v, np.nan)).to_numpy(dtype=float)
            else:
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            if values.size == n:
                out.append((str(col), values))
    except Exception as exc:
        print(f"fresh-shock sequence signal skipped: {exc}", flush=True)
    return out


def _drop_copy_columns(
    stack: np.ndarray,
    names: list[str],
    train_mask: np.ndarray,
    min_corr: float = 0.97,
) -> tuple[np.ndarray, list[str]]:
    """Drop a column that is a copy of one already kept.

    Rolling min, rolling max, and extra lags of a monthly reading often
    repeat the same number. The first copy stays. A different place stays.
    """
    if stack.ndim != 2 or stack.shape[1] == 0:
        return stack, names
    train = np.asarray(train_mask, dtype=bool)
    if int(train.sum()) < 30:
        return stack, names
    block = stack[train]
    keep: list[int] = []
    for i in range(stack.shape[1]):
        col = block[:, i]
        copy = False
        for j in keep:
            other = block[:, j]
            ok = np.isfinite(col) & np.isfinite(other)
            if int(ok.sum()) < 30:
                continue
            a = col[ok]
            b = other[ok]
            if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
                copy = bool(np.allclose(a, b))
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
                copy = bool(np.isfinite(corr) and abs(corr) >= min_corr)
            if copy:
                break
        if not copy:
            keep.append(i)
    return stack[:, keep], [names[i] for i in keep]


def _score_to_rate(
    score: np.ndarray,
    hits: np.ndarray,
    train_mask: np.ndarray,
    min_n: int = 80,
    max_percentile: float = 90.0,
    days: pd.DatetimeIndex | None = None,
    horizon: int | None = None,
) -> np.ndarray:
    """Chance equals how often this box shook on past mornings with a similar score.

    The lookup mornings should be ones the encoder was not fit on. A score
    above the given percentile is read as that percentile, so the extreme
    tip of memorized hits cannot be quoted.

    With `days` and `horizon`, a lookup morning that is itself being predicted
    leaves out every lookup morning within `horizon` days of it. Neighbouring
    mornings share most of their window, so otherwise it would see its own outcome.
    """
    score = np.asarray(score, dtype=float).reshape(-1)
    hits = np.asarray(hits, dtype=float).reshape(-1)
    train = np.asarray(train_mask, dtype=bool) & np.isfinite(score) & np.isfinite(hits)
    fallback = float(hits[train].mean()) if int(train.sum()) else 0.0
    out = np.full(score.shape[0], np.clip(fallback, 1e-6, 1.0 - 1e-6))
    n_train = int(train.sum())
    if n_train < 2:
        return out
    s = score[train]
    y = hits[train]
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    csum = np.cumsum(y_sorted)
    k = min(int(min_n), n_train)
    s_sorted = s[order]
    cap = float(np.nanpercentile(s, max_percentile))
    query = np.where(np.isfinite(score), score, np.nanmedian(s))
    query = np.minimum(query, cap)
    center = np.searchsorted(s_sorted, query, side="left")
    left = np.clip(center - (k // 2), 0, n_train - k)
    right = left + k
    prev = np.zeros(score.shape[0])
    has_prev = left > 0
    prev[has_prev] = csum[left[has_prev] - 1]
    total = csum[right - 1] - prev
    out = np.clip(total / k, 1e-6, 1.0 - 1e-6)
    if days is None or not horizon:
        return out
    stamps = pd.DatetimeIndex(days)
    if stamps.tz is not None:
        stamps = stamps.tz_convert("UTC").tz_localize(None)
    day_num = stamps.values.astype("datetime64[D]").astype(np.int64)
    ref_idx = np.flatnonzero(train)
    ref_day = day_num[ref_idx]
    sorted_pos = np.empty(n_train, dtype=np.int64)
    sorted_pos[order] = np.arange(n_train)
    gap = int(horizon)
    for j, i in enumerate(ref_idx):
        lo = np.searchsorted(ref_day, ref_day[j] - gap, side="left")
        hi = np.searchsorted(ref_day, ref_day[j] + gap, side="right")
        skip = np.sort(sorted_pos[lo:hi])
        kept = n_train - skip.size
        if kept < 1:
            continue
        kk = min(int(min_n), kept)
        reduced_center = int(center[i]) - int(np.searchsorted(skip, center[i], side="left"))
        first = int(np.clip(reduced_center - kk // 2, 0, kept - kk))
        a = _kept_position(first, skip)
        b = _kept_position(first + kk - 1, skip)
        inside = skip[(skip >= a) & (skip <= b)]
        window_sum = csum[b] - (csum[a - 1] if a > 0 else 0.0) - float(y_sorted[inside].sum())
        out[i] = np.clip(window_sum / kk, 1e-6, 1.0 - 1e-6)
    return out


def _kept_position(rank: int, skip: np.ndarray) -> int:
    """Sorted position of the rank-th lookup morning once the skipped positions are removed."""
    pos = int(rank)
    for s in skip:
        if s <= pos:
            pos += 1
        else:
            break
    return pos


def _lift_onto_rate(
    similar: np.ndarray,
    hits: np.ndarray,
    ref: np.ndarray,
    roll: np.ndarray,
    climate: float,
) -> np.ndarray:
    """Add the check-year lift to today's five-year rate.

    The quiet or busy level of the check years is not copied. Only how much
    similar mornings differed from those years' own average is kept.
    """
    ref = np.asarray(ref, dtype=bool)
    base = float(np.mean(hits[ref])) if int(ref.sum()) else float(climate)
    foil = np.where(np.isfinite(roll), roll, float(climate))
    return np.clip(foil + (np.asarray(similar, dtype=float) - base), 1e-6, 1.0 - 1e-6)


def _encoder_connections(encoder, names: list[str]) -> list[dict[str, Any]]:
    """Global shared-encoder weights, not local probability contributions.

    Input order is not importance: a first-ten slice advertised early CME lags
    with dummy weights on every box. This diagnostic never changes issued p.
    """
    weights = np.asarray(encoder.named_steps["lr"].coef_, dtype=float).reshape(-1)
    # BlankAwareScaler appends one missing-reading flag per original column.
    if weights.size == 2 * len(names):
        names = list(names) + ["missing__" + name for name in names]
    if weights.size != len(names):
        raise ValueError("encoder coefficient/feature count mismatch")
    order = sorted(
        (i for i, w in enumerate(weights) if np.isfinite(w) and w != 0.0),
        key=lambda i: (-abs(weights[i]), names[i]),
    )[:10]
    return [
        {
            "feature": names[i],
            "driver": names[i].split("__lag")[0],
            "lag_days": 0 if "__lag" not in names[i] else int(names[i].rsplit("__lag", 1)[1]),
            "label": _driver_plain(names[i]),
            "importance": float(weights[i]),
            "attribution_kind": "shared_encoder_coefficient",
        }
        for i in order
    ]


def _fit_shared_stack(
    *,
    feat: pd.DataFrame,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    horizon: int,
    root: Path | None = None,
    sit_out: bool = False,
    targets: tuple[tuple[str, float], ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str], list[str], dict[str, list[dict[str, Any]]]]:
    """One encoder on every non-empty family, then each box is lifted onto its rolling rate.

    The mix weight is reported; it does not multiply the published chance.
    """
    boxes = tuple(targets) if targets else ISSUED_TARGETS
    raw_x, names = design
    n = len(days)
    y_rank = fresh_hits(catalog, REGIONS["global"], 7.0, days, horizon).astype(int)
    hunt_cols = _hunt_drivers("global", root)
    fit0, val0 = fit_val_split(days, train_mask, purge=horizon)
    kept, families = _shared_family_indices(raw_x, names, y_rank, fit0, hunt_cols)
    extras = _promoted_leftover_score(
        feat, days, train_mask, hold_mask, catalog, horizon
    )
    stack = raw_x[:, kept] if kept else np.zeros((n, 0))
    stack_names = [names[i] for i in kept]
    before = len(stack_names)
    stack, stack_names = _drop_copy_columns(stack, stack_names, fit0)
    added = 0
    for name, col in extras + _useful_mainshock_signals(catalog, days, root):
        before_n = len(stack_names)
        stack, stack_names = _append_column(stack, stack_names, name, col)
        added += len(stack_names) - before_n
    print(
        f"fresh-shock shared stack {n} mornings × {stack.shape[1]} series "
        f"({'+'.join(families) or 'none'}), dropped {before - (len(stack_names) - added)} copies, "
        f"added {added} useful signals",
        flush=True,
    )
    rate_conn = [
        {
            "feature": "rate_roll",
            "driver": "rate_roll",
            "lag_days": 0,
            "label": "rolling five-year rate",
            "importance": 1.0,
        }
    ]

    def _climate_fallback() -> tuple[
        dict[str, np.ndarray], dict[str, str], list[str], dict[str, list[dict[str, Any]]]
    ]:
        out_fb: dict[str, np.ndarray] = {}
        recs = {rid: "fresh_shared|climate" for rid, _mag in boxes}
        conns = {rid: rate_conn for rid, _mag in boxes}
        for region_id, mag in boxes:
            hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(float)
            out_fb[region_id] = rolling_climate_series(hits, days, horizon=horizon)
        return out_fb, recs, families, conns

    if stack.shape[1] < 2 or int(fit0.sum()) < MIN_TRAIN:
        return _climate_fallback()
    stacked_y = np.concatenate(
        [
            fresh_hits(catalog, REGIONS[rid], mag, days, horizon).astype(int)[fit0]
            for rid, mag in boxes
        ]
    )
    if int(np.unique(stacked_y).size) < 2:
        return _climate_fallback()
    try:
        encoder = _fit_kind(
            "logit",
            np.vstack([stack[fit0]] * len(boxes)),
            stacked_y,
            {"C": 0.2, "sit_out": sit_out},
        )
        z = _proba(encoder, stack)
    except ValueError:
        return _climate_fallback()
    stack_conn = _encoder_connections(encoder, stack_names)
    fams = "+".join(families) if families else "none"
    out: dict[str, np.ndarray] = {}
    recipes: dict[str, str] = {}
    conns: dict[str, list[dict[str, Any]]] = {}
    for region_id, mag in boxes:
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(int)
        climate = recent_climate(hits.astype(float), days, train_mask)
        roll = rolling_climate_series(hits.astype(float), days, horizon=horizon)
        score = z[:, 0] if getattr(z, "ndim", 1) == 2 else np.asarray(z, dtype=float).reshape(-1)
        ref = val0 if int(val0.sum()) >= 80 else fit0
        if int(ref.sum()) >= 2:
            similar = _score_to_rate(
                score, hits.astype(float), ref, max_percentile=100, days=days, horizon=horizon
            )
            raw_p = _lift_onto_rate(similar, hits.astype(float), ref, roll, climate)
        else:
            raw_p = score
        mix_a, _brier = _mix(raw_p, hits, climate, val0 if val0.any() else fit0, roll=roll)
        issued = np.clip(np.asarray(raw_p, dtype=float), 1e-6, 1.0 - 1e-6)
        out[region_id] = issued
        recipes[region_id] = f"fresh_shared_{fams}|lift|mix{mix_a:.2f}"
        conns[region_id] = stack_conn
        print(
            f"fresh-shock shared {region_id} check-year lift on today's rate, "
            f"mix report {mix_a:.2f} ({len(stack_names)} cols)",
            flush=True,
        )
    print(f"fresh-shock shared stack done ({'+'.join(recipes.values())})", flush=True)
    return out, recipes, families, conns


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
    pooled_p: np.ndarray | None = None,
    pooled_recipe: str | None = None,
    picker: str = "wide",
    allow_tournament: bool = False,
    horizon: int = HORIZON_DAYS,
    issued_p: np.ndarray | None = None,
    issued_recipe: str | None = None,
    issued_connections: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    days = pd.DatetimeIndex(feat["issue_time"])
    if int(train_mask.sum()) < MIN_TRAIN or int(hold_mask.sum()) < MIN_SCORE:
        return None
    raw_x, names = design
    if raw_x.shape[1] == 0:
        return None
    y = hits.astype(int)
    fit_mask, val_mask = fit_val_split(days, train_mask, purge=horizon)
    val_score = val_mask if val_mask.any() else fit_mask
    score_hold = hold_mask
    alarm_hold = week_mask(hold_mask, fallback=False)
    hunt_cols = _hunt_drivers(region_id, root)
    climate_all = float(y[train_mask].mean())
    climate = recent_climate(y, days, train_mask)
    roll = rolling_climate_series(y.astype(float), days, horizon=horizon)
    cal = recent_mask(days, train_mask)
    if issued_p is not None and len(np.asarray(issued_p)) == len(y):
        # Shared stack already mixed onto the rolling rate. Calibrating again
        # is 2*roll - mean(roll) and can clip a quiet box to 1e-6.
        p_hat = np.clip(np.asarray(issued_p, dtype=float), 1e-6, 1.0 - 1e-6)
        recipe = issued_recipe or "fresh_shared"
        independent_p = p_hat
        independent_recipe = recipe
        issued_from = "shared"
    else:
        try:
            if picker == "shrunk":
                _model, p_hat, recipe = _pick_shrunk(
                    raw_x,
                    names,
                    y,
                    fit_mask,
                    val_score,
                    region_id=region_id,
                    days=days,
                    horizon=horizon,
                    hunt_cols=hunt_cols,
                )
            elif picker == "locked":
                _model, p_hat, recipe = _pick_locked(
                    raw_x,
                    names,
                    y,
                    fit_mask,
                    val_score,
                    region_id=region_id,
                    days=days,
                    horizon=horizon,
                    hunt_cols=hunt_cols,
                )
            else:
                _model, p_hat, recipe = _pick_model(
                    raw_x,
                    names,
                    y,
                    fit_mask,
                    val_score,
                    hunt_cols=hunt_cols,
                    region_id=region_id,
                    days=days,
                    horizon=horizon,
                )
        except ValueError:
            return None
        p_hat = _calibrate_to_roll(np.asarray(p_hat, dtype=float), roll, cal)
        independent_p = p_hat
        independent_recipe = recipe
        issued_from = "independent"
    p_pooled = None
    if pooled_p is not None and len(pooled_p) == len(p_hat):
        p_pooled = _calibrate_to_roll(np.asarray(pooled_p, dtype=float), roll, cal)
    p_shrunk = None
    shrink_recipe = None
    if allow_tournament and picker == "wide" and issued_from != "shared":
        try:
            _sm, raw_s, shrink_recipe = _pick_shrunk(
                raw_x,
                names,
                y,
                fit_mask,
                val_score,
                region_id=region_id,
                days=days,
                horizon=horizon,
                hunt_cols=hunt_cols,
            )
            p_shrunk = _calibrate_to_roll(np.asarray(raw_s, dtype=float), roll, cal)
        except ValueError:
            p_shrunk = None
        p_hat, issued_from, recipe = _apply_tournament(
            wide_p=independent_p,
            wide_recipe=recipe,
            y=y,
            val_mask=val_score,
            train_mask=train_mask,
            climate=climate,
            roll=roll,
            pooled_p=p_pooled,
            pooled_recipe=pooled_recipe,
            shrunk_p=p_shrunk,
            shrunk_recipe=shrink_recipe,
        )
    elif issued_from != "shared" and picker != "shrunk" and p_pooled is not None:
        foil = np.where(np.isfinite(roll), roll, climate)
        target = hits[val_score].astype(float) if val_score.any() else None
        ind_val = _brier(independent_p[val_score], target) if target is not None else None
        pool_val = _brier(p_pooled[val_score], target) if target is not None else None
        climate_val = _brier(foil[val_score], target) if target is not None else None
        pool_rate = float(np.nanmean(p_pooled[train_mask])) if train_mask.any() else None
        if _prefer_pooled(pool_val, ind_val, climate_val, pool_rate=pool_rate, climate=climate):
            p_hat = p_pooled
            recipe = f"{pooled_recipe or 'pooled'}|{recipe}"
            issued_from = "pooled"
    alarm = _alarm(p_hat, hits, days, train_mask, alarm_hold)
    if issued_connections:
        ranked = list(issued_connections)[:10]
    else:
        why_src = _usable_indices(raw_x, train_mask)
        thin = _pool_top(raw_x, y, train_mask, why_src, 8)
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
    fresh_p = float(p_hat[latest])
    hold_y = hits[score_hold].astype(float)
    hold_brier = _brier(p_hat[score_hold], hold_y)
    hold_climate = roll[score_hold]
    if hold_climate.size == 0 or not np.isfinite(hold_climate).any():
        hold_climate = np.full(int(score_hold.sum()), climate)
    else:
        hold_climate = np.where(np.isfinite(hold_climate), hold_climate, climate)
    climate_brier = _brier(hold_climate, hold_y)
    info_gain = information_gain(p_hat[score_hold], hold_y, hold_climate)
    foil_all = np.where(np.isfinite(roll), roll, climate)
    boot = block_bootstrap_beats(p_hat, foil_all, hits.astype(float), hold_mask)
    week_hold = week_mask(hold_mask, fallback=False)
    if int(week_hold.sum()) >= 8:
        week_y = hits[week_hold].astype(float)
        week_brier = _brier(p_hat[week_hold], week_y)
        week_climate_brier = _brier(foil_all[week_hold], week_y)
    else:
        week_brier = None
        week_climate_brier = None
    val_beats = False
    if val_score.any() and int(val_score.sum()) >= MIN_SCORE:
        v_brier = _brier(p_hat[val_score], hits[val_score].astype(float))
        v_base = np.where(np.isfinite(roll[val_score]), roll[val_score], climate)
        v_climate = _brier(v_base, hits[val_score].astype(float))
        val_beats = bool(
            v_brier is not None and v_climate is not None and v_brier < v_climate - 1e-12
        )
    latest_climate = float(roll[latest]) if np.isfinite(roll[latest]) else climate
    held = bool(
        hold_brier is None
        or climate_brier is None
        or hold_brier >= climate_brier
        or (boot.get("p_value") is not None and float(boot["p_value"]) >= BOOTSTRAP_ALPHA)
    )
    from eqpred.eval.csep import attach_region_csep

    return {
        "region": region_id,
        "label": _place(region_id),
        "magnitude_min": mag,
        "n_train": int(train_mask.sum()),
        "n_score": int(score_hold.sum()),
        "n_score_daily": int(hold_mask.sum()),
        "n_score_week": int(alarm_hold.sum()),
        "score_kind": SCORE_KIND,
        "brier_p_value": boot.get("p_value"),
        "brier_gap": boot.get("gap"),
        "brier_week": week_brier,
        "climatology_brier_week": week_climate_brier,
        "fresh_rate_train": climate_all,
        "fresh_rate_recent": climate,
        "fresh_rate_holdout": float(hits[hold_mask].mean()) if hold_mask.any() else None,
        "raw_rate_holdout": float(raw_hits[hold_mask].mean()) if hold_mask.any() else None,
        "brier": hold_brier,
        "climatology_brier": climate_brier,
        "information_gain": info_gain,
        **alarm,
        "recipe": recipe,
        "issued_from": issued_from,
        "independent_recipe": independent_recipe,
        "connections": ranked[:10],
        "why": why,
        "confirm_beats_climate": val_beats,
        "val_beats_climate": val_beats,
        "climate_kind": "rolling_5y",
        "csep": attach_region_csep(p_hat, hits.astype(float), hold_mask),
        "_p_hat": p_hat,
        "_hits": hits.astype(float),
        "this_week": {
            "issue_day": days[latest].date().isoformat(),
            "p": fresh_p,
            "fresh_raw_p": fresh_p,
            "fresh_raw_chance_plain": _chance(fresh_p),
            "held_to_climatology": held,
            "chance_plain": _chance(fresh_p),
            "climatology_p": latest_climate,
            "climatology_chance_plain": _chance(latest_climate),
            "climate_kind": "rolling_5y",
            "issued_from": issued_from,
            "tournament_winner": issued_from,
            "independent_p": float(independent_p[latest]),
            "independent_chance_plain": _chance(float(independent_p[latest])),
            "pooled_p": None if p_pooled is None else float(p_pooled[latest]),
            "pooled_chance_plain": None if p_pooled is None else _chance(float(p_pooled[latest])),
            "shrunk_p": None if p_shrunk is None else float(p_shrunk[latest]),
            "shrunk_chance_plain": None if p_shrunk is None else _chance(float(p_shrunk[latest])),
        },
    }


def _short_horizon_cards(
    *,
    feat: pd.DataFrame,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_end: str,
    holdout_start: str,
    catalog_end: pd.Timestamp,
    root: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """7-day and 3-day chances beside the 30-day card. The 3-day card uses the same model, then readings add on top."""
    rows: list[dict[str, Any]] = []
    draft: dict[str, Any] = {
        "writes_issued_p": False,
        "note": (
            "The 3-day card uses the same model as the 7-day and 30-day cards. "
            "Every reading can then add or subtract on top. It does not replace the 30-day card."
        ),
        "week": [],
    }
    latest = len(days) - 1
    if latest < 0:
        return rows, draft
    for horizon in SHORT_CARD_HORIZONS:
        closed = np.asarray(days + pd.Timedelta(days=horizon) <= catalog_end)
        train_mask, hold_mask = purged_train_hold(
            days, closed, train_end, holdout_start, purge=horizon
        )
        print(f"fresh-shock short card next {horizon} days", flush=True)
        from eqpred.eval.lineup import three_day_targets

        boxes = three_day_targets() if horizon == 3 else ISSUED_TARGETS
        shared_map, _recipes, _families, _conns = _fit_shared_stack(
            feat=feat,
            design=design,
            catalog=catalog,
            days=days,
            train_mask=train_mask,
            hold_mask=hold_mask,
            horizon=horizon,
            root=root,
            targets=boxes,
        )
        if horizon == 3:
            from eqpred.eval.lineup import open_three_day_rows

            voted = open_three_day_rows(
                design=design,
                catalog=catalog,
                days=days,
                train_mask=train_mask,
                hold_mask=hold_mask,
                horizon=3,
                shared_base=shared_map,
            )
            rows.extend(voted)
            draft = {
                "writes_issued_p": False,
                "horizon_days": 3,
                "same_as_frozen": False,
                "note": (
                    "The 3-day card uses the same model as the 7-day and 30-day cards. "
                    "Every reading can then add or subtract on top. It does not replace the 30-day card."
                ),
                "week": voted,
            }
            continue
        for region_id, mag in ISSUED_TARGETS:
            hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(float)
            roll = rolling_climate_series(hits, days, horizon=horizon)
            series = np.asarray(shared_map.get(region_id), dtype=float)
            if series.size != len(days):
                continue
            p = float(series[latest])
            climate = float(roll[latest]) if np.isfinite(roll[latest]) else float("nan")
            ticket = skill_beats_climate(series[hold_mask], hits[hold_mask], roll[hold_mask])
            ig = ticket.get("information_gain")
            ig_out = None if ig is None or not math.isfinite(float(ig)) else float(ig)
            climate_out = None if not math.isfinite(climate) else climate
            rows.append(
                {
                    "region": region_id,
                    "label": _place(region_id),
                    "magnitude_min": mag,
                    "horizon_days": horizon,
                    "p": None if not math.isfinite(p) else p,
                    "chance_plain": _chance(p),
                    "climatology_p": climate_out,
                    "climatology_chance_plain": _chance(climate_out),
                    "issue_day": days[latest].date().isoformat(),
                    "information_gain": ig_out,
                    "beats_climatology": bool(ticket.get("ig_beats")),
                    "writes_issued_p": False,
                    "beside_30_day": True,
                }
            )
    return rows, draft


def _week_sentence(packed: dict[str, Any], tw: dict[str, Any], horizon: int) -> str:
    region_id = str(packed.get("region") or "")
    mag = packed.get("magnitude_min")
    place = "anywhere on Earth" if region_id == "global" else f"in {packed.get('label')}"
    raw_chance = saved_fresh(tw, "fresh_raw_chance_plain") or _chance(saved_fresh(tw, "fresh_raw_p"))
    last_week = packed.get("mode") == "last_week"
    mode = packed.get("mode")
    look = (
        "Looking only at the last 7 days of non-catalog sources, the fresh-shock model puts"
        if last_week
        else (
            "The locked thin fresh-shock model puts"
            if mode == "locked_thin"
            else (
                "The shrunk fresh-shock model puts"
                if mode == "shrunk"
                else (
                    "The Alpine sequence model puts"
                    if mode == "alpine_sequence"
                    else "The fresh-shock model puts"
                )
            )
        )
    )
    extra = ""
    if tw.get("issued_from") == "gated":
        families = [str(name) for name in (tw.get("families") or [])]
        if families:
            extra += (
                " Only families that beat the five-year rate on the check years were allowed to move it: "
                + ", ".join(families)
                + "."
            )
        else:
            extra += " No family beat the five-year rate on the check years, so this card stays on that rate."
    elif tw.get("issued_from") == "shared":
        extra += " One shared encoder read every live family, then a small head for this box issued the chance."
    elif tw.get("issued_from") == "pooled":
        extra += " A pooled model that shares strength across the four boxes made this call."
    elif tw.get("issued_from") == "shrunk":
        extra += " A shrunk sibling won the per-box tournament for this call."
    elif tw.get("pooled_p") is not None and not last_week:
        extra += " This box kept its own specialist; a pooled four-box model was also fit."
    issued_chance = tw.get("chance_plain") or raw_chance or _chance(tw.get("p"))
    state = tw.get("week_state") or ("rate" if tw.get("held_to_climatology") else "beat")
    if state == "beat":
        extra += " It moved off the rolling five-year rate and beat it on later mornings."
    elif state == "moved":
        extra += " It moved off the rolling five-year rate, but has not beaten that rate yet."
    else:
        extra += " It did not leave the rolling five-year rate."
    climate_plain = tw.get("climatology_chance_plain") or _chance(tw.get("climatology_p"))
    traffic = tw.get("traffic_sentence")
    traffic_bit = f" {traffic}" if traffic else ""
    return (
        f"{look} the chance of a fresh magnitude {mag:g}+ quake "
        f"{place} in the next {horizon} days at {issued_chance}."
        f"{extra} The rolling five-year rate known that morning is {climate_plain}.{traffic_bit}"
    )


def _seal_region(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["recipe_leaked"] = recipe_is_leaked(str(out.get("recipe") or ""))
    out["beats_climatology"] = fresh_beats_climatology(out)
    held = fresh_this_week_held(out)
    tw = dict(out.get("this_week") or {})
    climate = tw.get("climatology_p")
    if climate is None:
        climate = out.get("fresh_rate_train")
    climate = float(climate or 0.0)
    fresh_raw = saved_fresh(tw, "fresh_raw_p")
    tw["climatology_p"] = climate
    tw["climatology_chance_plain"] = _chance(climate)
    tw["held_to_climatology"] = held
    model_p = float(fresh_raw if fresh_raw is not None else climate)
    tw["fresh_raw_p"] = model_p
    tw["fresh_raw_chance_plain"] = _chance(model_p)
    tw["p"] = model_p
    tw["chance_plain"] = _chance(model_p)
    tw["week_state"] = week_state_of(out, tw)
    out["this_week"] = tw
    out.pop("_p_hat", None)
    out.pop("_hits", None)
    return out


def _seal_week(week_row: dict[str, Any], packed: dict[str, Any]) -> dict[str, Any]:
    tw = packed["this_week"]
    out = dict(week_row)
    out["p"] = tw["p"]
    out["chance_plain"] = tw["chance_plain"]
    out["held_to_climatology"] = tw["held_to_climatology"]
    out["week_state"] = tw.get("week_state")
    out["traffic_light"] = tw.get("traffic_light")
    out["traffic_sentence"] = tw.get("traffic_sentence")
    out["traffic_max_m"] = tw.get("traffic_max_m")
    out["fresh_raw_p"] = tw.get("fresh_raw_p")
    out["fresh_raw_chance_plain"] = tw.get("fresh_raw_chance_plain") or _chance(tw.get("fresh_raw_p"))
    out["climatology_p"] = tw["climatology_p"]
    out["climatology_chance_plain"] = tw["climatology_chance_plain"]
    out["issued_from"] = tw.get("issued_from") or packed.get("issued_from")
    out["tournament_winner"] = tw.get("tournament_winner") or out.get("issued_from")
    out["independent_p"] = tw.get("independent_p")
    out["independent_chance_plain"] = tw.get("independent_chance_plain")
    out["pooled_p"] = tw.get("pooled_p")
    out["pooled_chance_plain"] = tw.get("pooled_chance_plain")
    out["shrunk_p"] = tw.get("shrunk_p")
    out["shrunk_chance_plain"] = tw.get("shrunk_chance_plain")
    out["climate_kind"] = tw.get("climate_kind") or packed.get("climate_kind")
    out["why"] = list(out.get("why") or packed.get("why") or [])
    horizon = int(out.get("horizon_days") or packed.get("horizon_days") or HORIZON_DAYS)
    out["sentence"] = _week_sentence(packed, tw, horizon)
    return out


def _attach_holm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Holm-adjust bootstrap p-values inside one comparison family."""
    adj = holm_adjust([row.get("brier_p_value") for row in rows])
    n = sum(1 for row in rows if row.get("brier_p_value") is not None)
    out: list[dict[str, Any]] = []
    for row, value in zip(rows, adj):
        packed = dict(row)
        packed["brier_p_holm"] = value
        packed["n_multiplicity"] = n
        out.append(packed)
    return out


def _attach_traffic_lights(report: dict[str, Any], feat: pd.DataFrame | None) -> dict[str, Any]:
    """Copy the latest Gulia light onto Hikurangi and Alpine. Issued p is untouched."""
    from eqpred.features.sequence import traffic_sentence

    if feat is None or feat.empty:
        return report
    row = feat.iloc[-1]
    for region in report.get("regions") or []:
        rid = str(region.get("region") or "")
        if rid not in {"hikurangi", "alpine"}:
            continue
        raw_light = row.get(f"seq_tl_{rid}") if hasattr(row, "get") else None
        if raw_light is None or (isinstance(raw_light, float) and not np.isfinite(raw_light)):
            light = "none"
        else:
            light = str(raw_light)
        max_raw = row.get(f"seq_max_m_{rid}") if hasattr(row, "get") else None
        try:
            max_m = float(max_raw)
            if not np.isfinite(max_m):
                max_m = None
        except (TypeError, ValueError):
            max_m = None
        tw = dict(region.get("this_week") or {})
        tw["traffic_light"] = light
        tw["traffic_max_m"] = max_m
        tw["traffic_sentence"] = traffic_sentence(rid, light, max_m)
        region["this_week"] = tw
    return report


def _seal_report(report: dict[str, Any]) -> dict[str, Any]:
    regions = [_seal_region(r) for r in _attach_holm(report.get("regions") or [])]
    report["regions"] = regions
    report["n_beats_climatology"] = sum(1 for r in regions if r.get("beats_climatology"))
    by_region = {r["region"]: r for r in regions}
    report["week"] = [
        _seal_week(w, by_region[w["region"]]) if w.get("region") in by_region else w
        for w in report.get("week") or []
    ]
    for exp in report.get("experiments") or []:
        exp_regions = [_seal_region(r) for r in _attach_holm(exp.get("regions") or [])]
        exp["regions"] = exp_regions
        exp["n_beats_climatology"] = sum(1 for r in exp_regions if r.get("beats_climatology"))
        by_r = {r["region"]: r for r in exp_regions}
        exp["week"] = [
            _seal_week(w, by_r[w["region"]]) if w.get("region") in by_r else w
            for w in exp.get("week") or []
        ]
    if report.get("last_week"):
        lw = dict(report["last_week"])
        lw_regions = [_seal_region(r) for r in _attach_holm(lw.get("regions") or [])]
        lw["regions"] = lw_regions
        lw["n_beats_climatology"] = sum(1 for r in lw_regions if r.get("beats_climatology"))
        by_lw = {r["region"]: r for r in lw_regions}
        lw["week"] = [
            _seal_week(w, by_lw[w["region"]]) if w.get("region") in by_lw else w
            for w in lw.get("week") or []
        ]
        lw["note"] = _english_last_week(lw)
        report["last_week"] = lw
    if report.get("shrunk"):
        sh = dict(report["shrunk"])
        sh_regions = [_seal_region(r) for r in _attach_holm(sh.get("regions") or [])]
        sh["regions"] = sh_regions
        sh["n_beats_climatology"] = sum(1 for r in sh_regions if r.get("beats_climatology"))
        by_sh = {r["region"]: r for r in sh_regions}
        sh["week"] = [
            _seal_week(w, by_sh[w["region"]]) if w.get("region") in by_sh else w
            for w in sh.get("week") or []
        ]
        sh["note"] = _english_shrunk(sh)
        sh["writes_issued_p"] = False
        report["shrunk"] = sh
    if report.get("locked_thin"):
        lt = dict(report["locked_thin"])
        lt_regions = [_seal_region(r) for r in _attach_holm(lt.get("regions") or [])]
        lt["regions"] = lt_regions
        lt["n_beats_climatology"] = sum(1 for r in lt_regions if r.get("beats_climatology"))
        by_lt = {r["region"]: r for r in lt_regions}
        lt["week"] = [
            _seal_week(w, by_lt[w["region"]]) if w.get("region") in by_lt else w
            for w in lt.get("week") or []
        ]
        lt["note"] = _english_locked(lt)
        lt["writes_issued_p"] = False
        report["locked_thin"] = lt
    if report.get("alpine_sequence"):
        al = dict(report["alpine_sequence"])
        al_regions = [_seal_region(r) for r in _attach_holm(al.get("regions") or [])]
        al["regions"] = al_regions
        al["n_beats_climatology"] = sum(1 for r in al_regions if r.get("beats_climatology"))
        by_al = {r["region"]: r for r in al_regions}
        al["week"] = [
            _seal_week(w, by_al[w["region"]]) if w.get("region") in by_al else w
            for w in al.get("week") or []
        ]
        al["applied"] = False
        al["writes_issued_p"] = False
        al["note"] = _english_alpine(al)
        report["alpine_sequence"] = al
    if report.get("method_eval"):
        me = dict(report["method_eval"])
        me_regions = [_seal_region(r) for r in _attach_holm(me.get("regions") or [])]
        me["regions"] = me_regions
        me["n_beats_climatology"] = sum(1 for r in me_regions if r.get("beats_climatology"))
        by_me = {r["region"]: r for r in me_regions}
        me["week"] = [
            _seal_week(w, by_me[w["region"]]) if w.get("region") in by_me else w
            for w in me.get("week") or []
        ]
        me["writes_issued_p"] = False
        me["note"] = _english_method(me)
        report["method_eval"] = me
    if report.get("kermadec_attribution"):
        ka = dict(report["kermadec_attribution"])
        ka["writes_issued_p"] = False
        ka["note"] = ka.get("note") or _english_kermadec(ka)
        report["kermadec_attribution"] = ka
    report["note"] = _english(report)
    return report


def seal_mainshock_file(root: Path | None = None) -> dict[str, Any] | None:
    pack = read_mainshock(root)
    if not pack:
        return None
    sealed = _seal_report(pack)
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "mainshock.json", sealed, default=str)
    return sealed


def _english(report: dict[str, Any]) -> str:
    bits = [
        f"The fresh-shock model looked at {report.get('n_features') or 0} series "
        "(space weather, tides, ionosphere, GPS, strain, and the rest — no raw aftershock counts) "
        f"and issued a call for a fresh large shock in the next {HORIZON_DAYS} days, not an aftershock. "
        f"The number beside it is the last {RECENT_CLIMATE_YEARS} years known that morning. "
        "Issued p is shifted onto that rolling rate so the model adds ranking, not a 2018-era mean. "
        "The rolling rate waits until each forecast window has closed. "
        "Train, validation, and later years are separated by a 30-day embargo. "
        "Later years are a retrospective backtest: overlapping days with a circular block "
        "bootstrap, Holm-adjusted across boxes, and a non-overlapping week score that must not lose."
    ]
    regions = report.get("regions") or []
    honest = [r for r in regions if r.get("beats_climatology")]
    shared = [r for r in regions if r.get("issued_from") == "shared"]
    if shared:
        bits.append(
            "Issued p is one shared encoder over every live family, lifted onto each box's rolling five-year rate. "
            "Box specialists stay as sidecars. The leftover score is one more column in that encoder; "
            "its check-year and 2019–2023 scores are printed, not used to let it in."
        )
    tour_used = [r for r in regions if r.get("issued_from") in {"pooled", "shrunk"}]
    if tour_used and not shared:
        bits.append(
            "A per-box tournament (wide vs pooled vs shrunk, must beat the rolling rate and stay near the box rate) "
            "issued "
            + ", ".join(f"{r['label']} ({r.get('issued_from')})" for r in tour_used)
            + "; the other boxes keep the wide specialist."
        )
    if not honest:
        bits.append(
            "On the issued boxes, no mix beat the rolling five-year rate on later mornings "
            "once hunt-selected features are treated as leakage, the loud days have to catch more quakes, "
            "and the error gap needs a block-bootstrap win. "
            "Issued p is still the model call; the historical rate sits beside it."
        )
    else:
        best = min(honest, key=lambda r: r["brier"] - r["climatology_brier"])
        bits.append(
            f"Best later-year score: {best['label']} M≥{best['magnitude_min']:g} "
            f"beat the rolling rate (error {best['brier']:.3f} vs {best['climatology_brier']:.3f}"
            + (
                f", Holm p={best.get('brier_p_holm'):.3f}"
                if best.get("brier_p_holm") is not None
                else (
                    f", bootstrap p={best.get('brier_p_value'):.3f}"
                    if best.get("brier_p_value") is not None
                    else ""
                )
            )
            + ")."
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
            "Alpine’s tiny error dip used hunt-confirmed knobs picked on the same later years"
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
            "error was worse than a constant — it leaned on season, not a non-catalog driver."
        )
    for exp in report.get("experiments") or []:
        n = int(exp.get("n_beats_climatology") or 0)
        rows = exp.get("regions") or []
        if exp.get("id") == "fresh_7d_m8":
            row = rows[0] if rows else None
            if row and row.get("beats_climatology"):
                bits.append(
                    f"Earth-wide M8 (7 days): the model ranked later weeks a bit better "
                    f"({row.get('n_hits')} hits in {row.get('n_trades')} loud days, "
                    f"{(row.get('lift') or 0) * 100:+.1f} pt) but the error score is almost tied "
                    f"({row['brier']:.4f} vs {row['climatology_brier']:.4f}), "
                    "so the historical rate is shown beside this week's M8 call."
                )
            else:
                bits.append("Earth-wide M8 next-7-days hunt did not beat the constant guess.")
        elif exp.get("id") == "fresh_30d_m8":
            row = rows[0] if rows else None
            bits.append(
                "Earth-wide M8 next-30-days hunt did not beat the constant guess."
                if not (row and row.get("beats_climatology"))
                else (
                    f"Earth-wide M8 next 30 days beat a constant guess "
                    f"(error {row['brier']:.4f} vs {row['climatology_brier']:.4f})."
                )
            )
        elif exp.get("id") == "fresh_30d_quiet":
            bits.append(
                "A quiet-start next 30 days fresh-shock hunt did not beat the constant guess."
                if not n
                else f"A quiet-start next 30 days fresh-shock hunt beat the constant guess on {n} issued box(es)."
            )
        elif exp.get("id") == "fresh_7d" or (
            exp.get("horizon_days") == 7
            and exp.get("id")
            not in {
                "fresh_7d_quiet",
                "fresh_7d_m8",
                "fresh_7d_lastweek",
            }
        ):
            bits.append(
                "A next 7 days fresh-shock hunt did not beat the constant guess."
                if not n
                else f"A next 7 days fresh-shock hunt beat the constant guess on {n} issued box(es)."
            )
        elif exp.get("horizon_days") == 30 and exp.get("id") != "fresh_30d":
            bits.append(
                "A next 30 days fresh-shock hunt did not beat the constant guess."
                if not n
                else f"A next 30 days fresh-shock hunt beat the constant guess on {n} issued box(es)."
            )
        elif exp.get("id") == "fresh_30d":
            bits.append(
                "A next 30 days fresh-shock hunt did not beat the constant guess."
                if not n
                else f"A next 30 days fresh-shock hunt beat the constant guess on {n} issued box(es)."
            )
        elif exp.get("id") == "fresh_7d_lastweek":
            bits.append(
                f"A last-week-only mode (only the past 7 days of non-catalog sources, calling the next {HORIZON_DAYS} days) "
                + (
                    "did not beat a constant guess."
                    if not n
                    else f"beat a constant guess on {n} issued box(es)."
                )
            )
        else:
            bits.append(
                "A quiet-start next week fresh-shock hunt did not beat the constant guess."
                if not n
                else f"A quiet-start next week fresh-shock hunt beat the constant guess on {n} issued box(es)."
            )
    sh = report.get("shrunk")
    if sh:
        n = int(sh.get("n_beats_climatology") or 0)
        bits.append(
            f"A separate shrunk model (family bake-off, at most {sh.get('cap') or SHRINK_CAP} columns) "
            + (
                f"beat a constant guess on {n} issued box(es). It does not write issued p."
                if n
                else "did not beat a constant guess. It does not write issued p."
            )
        )
    lt = report.get("locked_thin")
    if lt:
        n = int(lt.get("n_beats_climatology") or 0)
        bits.append(
            f"A locked thin recipe ({lt.get('n_features') or 0} pre-registered source knobs, last week only) "
            + (
                f"beat a constant guess on {n} issued box(es). It does not write issued p."
                if n
                else "did not beat a constant guess. It does not write issued p."
            )
        )
    al = report.get("alpine_sequence")
    if al:
        bits.append(_english_alpine(al))
    lw = report.get("last_week")
    if lw and not any(exp.get("id") == "fresh_7d_lastweek" for exp in report.get("experiments") or []):
        n = int(lw.get("n_beats_climatology") or 0)
        bits.append(
            f"A last-week-only mode (only the past 7 days of non-catalog sources, calling the next {HORIZON_DAYS} days) "
            + (
                "did not beat a constant guess."
                if not n
                else f"beat a constant guess on {n} issued box(es)."
            )
        )
    me = report.get("method_eval")
    if me:
        bits.append(me.get("note") or _english_method(me))
    ka = report.get("kermadec_attribution")
    if ka:
        bits.append(ka.get("note") or _english_kermadec(ka))
    ev = report.get("evaluation") or {}
    if ev.get("note"):
        bits.append(str(ev["note"]))
    return " ".join(bits)


def _english_last_week(pack: dict[str, Any]) -> str:
    n_feat = pack.get("n_features") or 0
    n = int(pack.get("n_beats_climatology") or 0)
    if n:
        return (
            f"Last-week model looked at {n_feat} last-7-day fresh-shock series and beat a constant "
            f"guess on {n} issued box(es). Issued p is still the model call; climatology sits beside it."
        )
    return (
        f"Last-week model looked at {n_feat} last-7-day fresh-shock series and called the next {HORIZON_DAYS} days. "
        "It did not beat a constant guess on later years. Issued p is still the model call; climatology sits beside it."
    )


def _english_locked(pack: dict[str, Any]) -> str:
    n_feat = pack.get("n_features") or 0
    n = int(pack.get("n_beats_climatology") or 0)
    n_lock = pack.get("n_locked") or len(LOCKED_THIN)
    if n:
        return (
            f"Locked thin model used {n_feat} series from {n_lock} pre-registered source knobs "
            f"(no analog, no extra archives) and beat a constant guess on {n} issued box(es). "
            "It does not write issued p."
        )
    return (
        f"Locked thin model used {n_feat} series from {n_lock} pre-registered source knobs "
        f"(no analog, no extra archives) and called the next {HORIZON_DAYS} days. "
        "It did not beat a constant guess on later years. It does not write issued p."
    )


def _english_shrunk(pack: dict[str, Any]) -> str:
    n_feat = pack.get("n_features") or 0
    n = int(pack.get("n_beats_climatology") or 0)
    cap = pack.get("cap") or SHRINK_CAP
    return (
        f"Shrunk model looked at {n_feat} series (one family at a time, at most {cap} columns) "
        f"and beat a constant guess on {n} issued box(es). "
        "This is a sibling of the wide issued model. It does not write issued p."
    )


def _english_method(pack: dict[str, Any]) -> str:
    n = int(pack.get("n_beats_climatology") or 0)
    rows = pack.get("regions") or []
    labels = ", ".join(str(r.get("label") or r.get("region")) for r in rows) or "Japan and Chile"
    if n:
        return (
            f"Japan and Chile practice boxes ({labels}) scored the same daily-window method; "
            f"{n} beat the rolling rate. They are not issued cards."
        )
    return (
        f"Japan and Chile practice boxes ({labels}) scored the same daily-window method "
        "and did not beat the rolling rate. They are not issued cards."
    )


def _english_kermadec(pack: dict[str, Any]) -> str:
    verdict = str(pack.get("verdict") or "none")
    if verdict == "precursor":
        return (
            "Kermadec still has leftover skill after season and slow-slip are removed. "
            "That leftover does not write issued p unless the per-box tournament already picked this box."
        )
    if verdict == "sse":
        return (
            "Kermadec’s extra lift tracks slow-slip state more than a leftover precursor. "
            "That is legitimate inside a rate model; it is not claimed as a sensor precursor."
        )
    if verdict == "season":
        return (
            "Kermadec’s extra lift looks like season, not a leftover precursor. "
            "The rolling-rate spine already carries that."
        )
    return (
        "Kermadec has no extra skill beyond the rolling rate once season and slow-slip are checked. "
        "This look does not write issued p."
    )


def _english_alpine(pack: dict[str, Any]) -> str:
    n_feat = pack.get("n_features") or 0
    would = bool(pack.get("would_have_admitted"))
    extra = (
        " It would have passed the old validation gate, but sequence stays off issued p."
        if would
        else " Later years are not used to admit it."
    )
    return (
        f"Alpine sequence model looked at {n_feat} sequence/foreshock columns. "
        f"It stays a sidecar and never writes issued p.{extra}"
    )


def _alpine_sequence_frame(
    feat: pd.DataFrame,
    catalog: pd.DataFrame,
    root: Path | None,
) -> pd.DataFrame:
    if any(str(col).startswith("seq_") and "alpine" in str(col) for col in feat.columns):
        return feat
    cache = data_dir(root) / "eval" / "sequence_hunt_frame.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        cached["issue_time"] = parse_utc(cached["issue_time"])
        if sequence_cache_matches(cached, feat):
            return cached
    if catalog is None or "available_at" not in catalog.columns or len(catalog) < 1000:
        return feat
    from eqpred.catalog import load_comcat_versions
    from eqpred.eval.foreshock import catalog_foreshock_frame

    try:
        versions = load_comcat_versions(root)
    except Exception:
        versions = None
    built = catalog_foreshock_frame(catalog, feat["issue_time"], versions, root)
    built["issue_time"] = parse_utc(built["issue_time"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    built.to_parquet(cache, index=False)
    return built


def _alpine_seq_columns(frame: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in frame.columns:
        if col == "issue_time" or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        name = str(col)
        if name.startswith("seq_") and "alpine" in name:
            out.append(name)
        elif "alpine" in name and name.startswith(("n_m4_", "n_m5_", "n_geonet")):
            out.append(name)
    return out


def _empty_alpine() -> dict[str, Any]:
    return {
        "id": "alpine_sequence",
        "mode": "alpine_sequence",
        "horizon_days": HORIZON_DAYS,
        "writes_issued_p": False,
        "applied": False,
        "n_features": 0,
        "n_beats_climatology": 0,
        "regions": [],
        "week": [],
        "note": "No Alpine sequence columns on this feature frame.",
    }


def _alpine_sequence_sidecar(
    *,
    feat: pd.DataFrame,
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_end: str,
    holdout_start: str,
    catalog_end,
    root: Path | None,
) -> dict[str, Any]:
    empty = _empty_alpine()
    frame = _alpine_sequence_frame(feat, catalog, root)
    cols = _alpine_seq_columns(frame)
    if len(cols) < 2:
        return empty
    work = pd.DataFrame({"issue_time": parse_utc(feat["issue_time"])})
    other = frame[["issue_time", *cols]].copy()
    other["issue_time"] = parse_utc(other["issue_time"])
    work = work.merge(other.drop_duplicates("issue_time"), on="issue_time", how="left")
    design = _design(work, cols, (0, 1, 2, 3, 5, 7))
    if design[0].shape[1] < 2:
        return empty
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    alpine_mag = next(mag for rid, mag in ISSUED_TARGETS if rid == "alpine")
    hits = fresh_hits(catalog, REGIONS["alpine"], alpine_mag, days, HORIZON_DAYS)
    raw = _window_hits(catalog, REGIONS["alpine"], alpine_mag, days, HORIZON_DAYS)
    packed = _region_mainshock(
        feat=work,
        columns=cols,
        region_id="alpine",
        mag=alpine_mag,
        train_mask=train_mask,
        hold_mask=hold_mask,
        hits=hits,
        raw_hits=raw,
        design=design,
        root=root,
        picker="wide",
    )
    if not packed:
        return empty
    packed["horizon_days"] = HORIZON_DAYS
    packed["quiet_start"] = False
    packed["mode"] = "alpine_sequence"
    tw = packed["this_week"]
    return {
        "id": "alpine_sequence",
        "mode": "alpine_sequence",
        "horizon_days": HORIZON_DAYS,
        "writes_issued_p": False,
        "applied": False,
        "n_features": int(design[0].shape[1]),
        "n_beats_climatology": 0,
        "regions": [packed],
        "week": [
            {
                "region": "alpine",
                "label": packed["label"],
                "magnitude_min": alpine_mag,
                "horizon_days": HORIZON_DAYS,
                "quiet_start": False,
                "mode": "alpine_sequence",
                "p": tw["p"],
                "fresh_raw_p": tw.get("fresh_raw_p"),
                "fresh_raw_chance_plain": tw.get("fresh_raw_chance_plain") or _chance(tw.get("fresh_raw_p")),
                "held_to_climatology": tw.get("held_to_climatology"),
                "chance_plain": tw["chance_plain"],
                "climatology_p": tw["climatology_p"],
                "climatology_chance_plain": tw["climatology_chance_plain"],
                "why": packed.get("why") or [],
                "issue_day": tw["issue_day"],
                "sentence": _week_sentence(packed, tw, HORIZON_DAYS),
            }
        ],
        "_p_hat": packed.get("_p_hat"),
    }


def _logit_on_columns(
    feat: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    climate = float(y[train_mask].mean()) if train_mask.any() else 0.0
    fallback = np.full(len(y), climate)
    if len(columns) < 1 or int(train_mask.sum()) < MIN_TRAIN:
        return fallback
    raw_x, _names = _design(feat, columns, (0,))
    if raw_x.shape[1] < 1 or int(np.unique(y[train_mask]).size) < 2:
        return fallback
    try:
        model = _fit_kind("logit", raw_x[train_mask], y[train_mask], {"C": 0.4})
        return _proba(model, raw_x)
    except ValueError:
        return fallback


def _kermadec_attribution(
    *,
    feat: pd.DataFrame,
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_end: str,
    holdout_start: str,
    catalog_end,
    p_issued: np.ndarray | None = None,
) -> dict[str, Any]:
    """Season vs slow-slip vs leftover on Kermadec. Never writes issued p by itself."""
    mag = next(m for rid, m in ISSUED_TARGETS if rid == "kermadec")
    work = feat.copy()
    if "cal_doy_sin" not in work.columns:
        doy = days.dayofyear.to_numpy()
        work["cal_doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        work["cal_doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    hits = fresh_hits(catalog, REGIONS["kermadec"], mag, days, HORIZON_DAYS).astype(int)
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    roll = rolling_climate_series(hits.astype(float), days, horizon=HORIZON_DAYS)
    climate = recent_climate(hits, days, train_mask)
    cal = recent_mask(days, train_mask)
    foil = np.where(np.isfinite(roll), roll, climate)
    season_cols = [c for c in ("cal_doy_sin", "cal_doy_cos") if c in work.columns]
    sse_cols = [c for c in work.columns if str(c).startswith("sse_")]
    y = hits
    season_p = _calibrate_to_roll(_logit_on_columns(work, season_cols, y, train_mask), roll, cal)
    sse_p = _calibrate_to_roll(
        _logit_on_columns(work, list(dict.fromkeys([*season_cols, *sse_cols])), y, train_mask),
        roll,
        cal,
    )
    leftover_p = None if p_issued is None else np.asarray(p_issued, dtype=float)

    def _pack(name: str, series: np.ndarray, against: np.ndarray) -> dict[str, Any]:
        boot = block_bootstrap_beats(series, against, y.astype(float), hold_mask)
        brier = _brier(series[hold_mask], y[hold_mask].astype(float)) if hold_mask.any() else None
        foil_brier = _brier(against[hold_mask], y[hold_mask].astype(float)) if hold_mask.any() else None
        return {
            "name": name,
            "brier": brier,
            "foil_brier": foil_brier,
            "brier_p_value": boot.get("p_value"),
            "brier_gap": boot.get("gap"),
            "beats": bool(boot.get("beats")),
            "n_sse_columns": len(sse_cols),
        }

    season = _pack("season", season_p, foil)
    season_sse = _pack("season_sse", sse_p, season_p)
    leftover = (
        _pack("leftover", leftover_p, sse_p)
        if leftover_p is not None
        else {"name": "leftover", "beats": False, "brier": None, "foil_brier": None, "brier_p_value": None}
    )
    if leftover.get("beats"):
        verdict = "precursor"
    elif season_sse.get("beats") and len(sse_cols) >= 1:
        verdict = "sse"
    elif season.get("beats"):
        verdict = "season"
    else:
        verdict = "none"
    pack = {
        "id": "kermadec_attribution",
        "writes_issued_p": False,
        "region": "kermadec",
        "label": _place("kermadec"),
        "magnitude_min": mag,
        "verdict": verdict,
        "season": season,
        "season_sse": season_sse,
        "leftover": leftover,
        "n_sse_columns": len(sse_cols),
        "score_kind": SCORE_KIND,
    }
    pack["note"] = _english_kermadec(pack)
    return pack


def alpine_admission_ok(sidecar: dict[str, Any] | None) -> bool:
    """Admit on validation only. Later-year scores never pick the issued mix."""
    rows = (sidecar or {}).get("regions") or []
    if not rows:
        return False
    row = rows[0]
    return bool(row.get("val_beats_climate") and not recipe_is_leaked(str(row.get("recipe") or "")))


def _maybe_admit_alpine(sidecar: dict[str, Any], regions: list[dict[str, Any]]) -> dict[str, Any]:
    would = alpine_admission_ok(sidecar)
    sidecar["applied"] = False
    sidecar["writes_issued_p"] = False
    sidecar["would_have_admitted"] = would
    rows = sidecar.get("regions") or []
    if rows:
        sidecar["beats_climatology"] = bool(_seal_region(dict(rows[0])).get("beats_climatology"))
        sidecar["confirm_beats_climate"] = bool(rows[0].get("confirm_beats_climate"))
    _ = regions
    return sidecar


def issued_fresh_daily(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Daily issued p per issued box. Catalog features stay out."""
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = fresh_derived(feat.copy())
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
    feat = attach_rate_spine(feat, catalog, days, HORIZON_DAYS)
    columns = issued_columns(_driver_columns(feat, unconventional_only=True))
    design = _design(feat, columns, ISSUED_LAGS)
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    out: dict[str, dict[str, Any]] = {}
    print("fresh-shock shared stack (all issued boxes)", flush=True)
    shared_map, shared_recipes, _families, shared_conns = _fit_shared_stack(
        feat=feat,
        design=design,
        catalog=catalog,
        days=days,
        train_mask=train_mask,
        hold_mask=hold_mask,
        horizon=HORIZON_DAYS,
        root=root,
    )
    for region_id, mag in ISSUED_TARGETS:
        print(f"fresh-shock {region_id} M{mag:g} ({design[0].shape[1]} series)", flush=True)
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
            issued_p=shared_map.get(region_id),
            issued_recipe=shared_recipes.get(region_id),
            issued_connections=shared_conns.get(region_id),
            allow_tournament=False,
        )
        if not packed or packed.get("_p_hat") is None:
            continue
        sealed = _seal_region(dict(packed))
        out[region_id] = {
            "p": pd.Series(np.asarray(packed["_p_hat"], dtype=float), index=days),
            "climatology_p": float(sealed["this_week"]["climatology_p"]),
            "fresh_raw_p": sealed["this_week"].get("fresh_raw_p"),
            "connections": packed.get("connections") or [],
            "recipe": sealed.get("recipe"),
            "magnitude_min": mag,
            "held_to_climatology": sealed["this_week"].get("held_to_climatology"),
        }
    return feat, out


def _digest_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _eval_fingerprint(
    *,
    feat: pd.DataFrame,
    catalog: pd.DataFrame,
    root: Path | None,
    train_end: str,
    holdout_start: str,
) -> dict[str, Any]:
    from eqpred.models.registry import code_hash

    _ = root
    days = pd.DatetimeIndex(feat["issue_time"]) if "issue_time" in feat.columns else pd.DatetimeIndex([])
    observed = (
        parse_utc(catalog["observed_at"])
        if "observed_at" in catalog.columns and len(catalog)
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    inferred = "available_at" not in catalog.columns
    if not inferred and len(catalog):
        inferred = bool((parse_utc(catalog["available_at"]) == observed).all())
    return {
        "protocol": EVAL_PROTOCOL,
        "purge_days": HORIZON_DAYS,
        "target_decluster": "causal_aftershock",
        "score_kind": SCORE_KIND,
        "csep_windows": "nonoverlapping_bernoulli",
        "code_hash": code_hash(),
        "feature_digest": _digest_payload(
            {
                "n": int(len(feat)),
                "columns": sorted(str(c) for c in feat.columns),
                "start": None if len(days) == 0 else str(days.min()),
                "end": None if len(days) == 0 else str(days.max()),
            }
        ),
        "catalog_digest": _digest_payload(
            {
                "n": int(len(catalog)),
                "start": None if observed.empty else str(observed.min()),
                "end": None if observed.empty else str(observed.max()),
                "inferred_latency": inferred,
            }
        ),
        "inferred_latency": inferred,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "prospective_epoch_start": PROSPECTIVE_EPOCH_START,
        "score_is": "retrospective_holdout",
        "note": (
            "Holdout Brier is a backtest. Prospective skill starts at "
            f"{PROSPECTIVE_EPOCH_START} and only counts cards written at or before issue time."
        ),
    }


def prepare_issued(
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, tuple[np.ndarray, list[str]], pd.Timestamp]:
    """Feature table, catalog, and the issued design. Does not fit a model."""
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = fresh_derived(feat.copy())
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
    feat = attach_rate_spine(feat, catalog, days, HORIZON_DAYS)
    columns = issued_columns(_driver_columns(feat, unconventional_only=True))
    design = _design(feat, columns, ISSUED_LAGS)
    return feat, catalog, days, design, catalog_end


def refit_sitout_draft(
    root: Path | None = None,
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
) -> dict[str, Any]:
    """Refit the working 30-day model so a blank column sits out. The frozen card stays."""
    from eqpred.eval.honest import fit_val_split
    from eqpred.eval.skill import information_gain

    feat, catalog, days, design, catalog_end = prepare_issued(root)
    closed = np.asarray(days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end)
    train_mask, _hold_mask = purged_train_hold(
        days, closed, train_end, holdout_start, purge=HORIZON_DAYS
    )
    print("working draft, blank columns sit out", flush=True)
    shared, _recipes, families, _conns = _fit_shared_stack(
        feat=feat,
        design=design,
        catalog=catalog,
        days=days,
        train_mask=train_mask,
        hold_mask=_hold_mask,
        horizon=HORIZON_DAYS,
        root=root,
        sit_out=True,
    )
    fit0, val0 = fit_val_split(days, train_mask, purge=HORIZON_DAYS)
    latest = len(days) - 1
    week: list[dict[str, Any]] = []
    for region_id, mag in ISSUED_TARGETS:
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, HORIZON_DAYS).astype(float)
        roll = rolling_climate_series(hits, days, horizon=HORIZON_DAYS)
        series = np.asarray(shared.get(region_id), dtype=float)
        if series.size != len(days):
            continue
        p = float(series[latest])
        climate = float(roll[latest]) if np.isfinite(roll[latest]) else float("nan")
        check_ig = information_gain(series[val0], hits[val0], roll[val0]) if int(val0.sum()) else None
        week.append(
            {
                "region": region_id,
                "label": _place(region_id),
                "magnitude_min": mag,
                "horizon_days": HORIZON_DAYS,
                "p": p,
                "chance_plain": _chance(p),
                "climatology_p": None if not math.isfinite(climate) else climate,
                "climatology_chance_plain": _chance(climate),
                "issue_day": days[latest].date().isoformat(),
                "check_information_gain": None if check_ig is None or not math.isfinite(check_ig) else float(check_ig),
                "writes_issued_p": False,
                "sit_out": True,
            }
        )
        print(f"sit-out draft {region_id} {_chance(p)} typical {_chance(climate)}", flush=True)
    dest = data_dir(root) / "eval" / "mainshock.json"
    report = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    frozen_week = [dict(row) for row in report.get("week") or []]
    previous = report.get("draft") or {}
    report["draft"] = {
        "writes_issued_p": False,
        "locked_at": previous.get("locked_at"),
        "sit_out": True,
        "families": list(families),
        "note": (
            "Working model. A blank column sits out and a real zero still counts. "
            "It does not replace the frozen card."
        ),
        "week": week,
        "promote": previous.get("promote") or {"ready": False, "reason": "The frozen card is still the exam."},
    }
    report["week"] = frozen_week
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, report, default=str)
    return report["draft"]


def run_mainshock(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
    experiment_ids: tuple[str, ...] | None = None,
    reuse_sidecars: bool = False,
) -> dict[str, Any]:
    feat, catalog, days, design, catalog_end = prepare_issued(root, features, quakes)
    columns = issued_columns(_driver_columns(feat, unconventional_only=True))

    def _collect(
        horizon: int,
        quiet: bool,
        targets: tuple[tuple[str, float], ...] = ISSUED_TARGETS,
        *,
        used_design: tuple[np.ndarray, list[str]] | None = None,
        used_columns: list[str] | None = None,
        mode: str = "all",
        picker: str = "wide",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        closed = days + pd.Timedelta(days=horizon) <= catalog_end
        train_mask, hold_mask = purged_train_hold(
            days, closed, train_end, holdout_start, purge=horizon
        )
        packed_rows: list[dict[str, Any]] = []
        week_rows: list[dict[str, Any]] = []
        design_x = used_design if used_design is not None else design
        cols = used_columns if used_columns is not None else columns
        shared_map: dict[str, np.ndarray] = {}
        shared_recipes: dict[str, str] = {}
        shared_conns: dict[str, list[dict[str, Any]]] = {}
        if (
            not quiet
            and mode == "all"
            and picker == "wide"
            and horizon == HORIZON_DAYS
            and targets == ISSUED_TARGETS
        ):
            print("fresh-shock shared stack (all issued boxes)", flush=True)
            shared_map, shared_recipes, _families, shared_conns = _fit_shared_stack(
                feat=feat,
                design=design_x,
                catalog=catalog,
                days=days,
                train_mask=train_mask,
                hold_mask=hold_mask,
                horizon=horizon,
                root=root,
            )
        for region_id, mag in targets:
            print(f"fresh-shock {mode} {region_id} M{mag:g} h{horizon}", flush=True)
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
                issued_p=shared_map.get(region_id),
                issued_recipe=shared_recipes.get(region_id),
                issued_connections=shared_conns.get(region_id),
                picker=picker,
                allow_tournament=False,
                horizon=horizon,
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
                    "fresh_raw_p": tw.get("fresh_raw_p"),
                    "fresh_raw_chance_plain": tw.get("fresh_raw_chance_plain") or _chance(tw.get("fresh_raw_p")),
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
    wanted = set(experiment_ids) if experiment_ids is not None else set(ALL_EXPERIMENT_IDS)
    prior = read_mainshock(root) if reuse_sidecars else None
    experiments: list[dict[str, Any]] = []
    for exp_id, horizon, quiet, targets in (
        ("fresh_7d", 7, False, ISSUED_TARGETS),
        ("fresh_7d_quiet", 7, True, ISSUED_TARGETS),
        ("fresh_7d_m8", 7, False, (("global", 8.0),)),
        ("fresh_30d_m8", HORIZON_DAYS, False, (("global", 8.0),)),
        ("fresh_30d_quiet", HORIZON_DAYS, True, ISSUED_TARGETS),
    ):
        if exp_id not in wanted:
            old = next((e for e in (prior or {}).get("experiments") or [] if e.get("id") == exp_id), None)
            if old:
                experiments.append(old)
            continue
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
    if reuse_sidecars and prior and prior.get("last_week"):
        last_week = prior["last_week"]
        week_design = _design(feat, last_week_columns(columns), LAST_WEEK_LAGS)
    else:
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
    shrink_cols = shrink_columns(columns)
    shrink_design = _design(feat, shrink_cols, ISSUED_LAGS)
    shrink_regions, shrink_week = _collect(
        HORIZON_DAYS,
        quiet=False,
        used_design=shrink_design,
        used_columns=shrink_cols,
        mode="shrunk",
        picker="shrunk",
    )
    shrunk = {
        "id": "fresh_shrunk",
        "mode": "shrunk",
        "horizon_days": HORIZON_DAYS,
        "writes_issued_p": False,
        "n_features": int(shrink_design[0].shape[1]),
        "cap": SHRINK_CAP,
        "per_family": SHRINK_PER_FAMILY,
        "families": list(SHRINK_FAMILIES),
        "n_beats_climatology": 0,
        "regions": shrink_regions,
        "week": shrink_week,
    }
    alpine_sequence = _alpine_sequence_sidecar(
        feat=feat,
        catalog=catalog,
        days=days,
        train_end=train_end,
        holdout_start=holdout_start,
        catalog_end=catalog_end,
        root=root,
    )
    alpine_sequence = _maybe_admit_alpine(alpine_sequence, regions)
    practice_regions, practice_week = _collect(
        HORIZON_DAYS,
        quiet=False,
        targets=PRACTICE_TARGETS,
        mode="method_eval",
    )
    method_eval = {
        "id": "method_eval",
        "mode": "method_eval",
        "horizon_days": HORIZON_DAYS,
        "writes_issued_p": False,
        "n_beats_climatology": 0,
        "regions": practice_regions,
        "week": practice_week,
    }
    kerm_p = next((row.get("_p_hat") for row in regions if row.get("region") == "kermadec"), None)
    kermadec_attribution = _kermadec_attribution(
        feat=feat,
        catalog=catalog,
        days=days,
        train_end=train_end,
        holdout_start=holdout_start,
        catalog_end=catalog_end,
        p_issued=kerm_p,
    )

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
        "climate_kind": "rolling_5y",
        "score_kind": SCORE_KIND,
        "evaluation": _eval_fingerprint(
            feat=feat,
            catalog=catalog,
            root=root,
            train_end=train_end,
            holdout_start=holdout_start,
        ),
        "regions": regions,
        "week": week,
        "last_week": last_week,
        "shrunk": shrunk,
        "alpine_sequence": alpine_sequence,
        "method_eval": method_eval,
        "kermadec_attribution": kermadec_attribution,
        "experiments": experiments,
        "catalog_end": catalog_end.date().isoformat() if hasattr(catalog_end, "date") else str(catalog_end),
    }
    report = _attach_traffic_lights(report, feat)
    from eqpred.eval.frozen import hold_published_card

    report = hold_published_card(report, root)
    report["short_horizons"], report["vote_draft"] = _short_horizon_cards(
        feat=feat,
        design=design,
        catalog=catalog,
        days=days,
        train_end=train_end,
        holdout_start=holdout_start,
        catalog_end=catalog_end,
        root=root,
    )
    from eqpred.eval.vote_freeze import save_model_calls, seal_new_mornings

    save_model_calls(report, root)
    seal_new_mornings(design=design, catalog=catalog, days=days, root=root)
    report = _seal_report(report)
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    _write_csep_sidecar(report, dest)
    write_json(dest / "mainshock.json", report, default=str)
    return report


def _write_csep_sidecar(report: dict[str, Any], dest: Path) -> None:
    from eqpred.eval.csep import _english as csep_english
    from eqpred.eval.csep import s_test

    boxes = []
    expected: dict[str, float] = {}
    observed: dict[str, float] = {}
    for row in report.get("regions") or []:
        block = row.get("csep") or {}
        if not block:
            continue
        n_test = (block.get("tests") or {}).get("n") or {}
        rid = str(row.get("region") or "")
        expected[rid] = float(n_test.get("n_forecast") or 0.0)
        observed[rid] = float(n_test.get("n_observed") or 0.0)
        boxes.append({"region": rid, "label": row.get("label"), **block})
    spatial = s_test(expected, observed) if expected else {"name": "S", "error": "no boxes"}
    sidecar = {
        "writes_issued_p": False,
        "train_end": report.get("train_end"),
        "holdout_start": report.get("holdout_start"),
        "horizon_days": report.get("horizon_days"),
        "n_windows": max((int(b.get("n_windows") or 0) for b in boxes), default=0),
        "window_kind": "nonoverlapping_bernoulli",
        "boxes": boxes,
        "spatial": spatial,
        "n_pass": sum(int(b.get("n_pass") or 0) for b in boxes) + (1 if spatial.get("pass") else 0),
        "n_fail": sum(int(b.get("n_fail") or 0) for b in boxes) + (0 if spatial.get("pass") else 1),
        "scanned_at": report.get("scanned_at"),
        "note": csep_english(boxes, spatial),
    }
    (dest / "csep.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    report["csep"] = sidecar


def run_last_week_mainshock(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score the model using only the last 7 days of non-catalog sources. Does not rewrite issued p."""
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = fresh_derived(feat.copy())
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog = quakes if quakes is not None else load_comcat(root)
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    columns = last_week_columns(issued_columns(_driver_columns(feat, unconventional_only=True)))
    design = _design(feat, columns, LAST_WEEK_LAGS)
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    packed_rows: list[dict[str, Any]] = []
    week_rows: list[dict[str, Any]] = []
    for region_id, mag in ISSUED_TARGETS:
        print(f"last-week model {region_id} M{mag:g} ({design[0].shape[1]} series)", flush=True)
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
                "fresh_raw_p": tw.get("fresh_raw_p"),
                "fresh_raw_chance_plain": tw.get("fresh_raw_chance_plain") or _chance(tw.get("fresh_raw_p")),
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
    write_json(dest / "mainshock.json", sealed, default=str)
    return sealed


def run_locked_thin_mainshock(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Paper-score the frozen pre-registered source list. Does not rewrite issued p."""
    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = fresh_derived(feat.copy())
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
    columns = locked_thin_columns(_driver_columns(feat, unconventional_only=True))
    design = _design(feat, columns, LOCKED_THIN_LAGS)
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    packed_rows: list[dict[str, Any]] = []
    week_rows: list[dict[str, Any]] = []
    for region_id, mag in ISSUED_TARGETS:
        print(f"locked-thin model {region_id} M{mag:g} ({design[0].shape[1]} series)", flush=True)
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
            picker="locked",
        )
        if not packed:
            continue
        packed["horizon_days"] = HORIZON_DAYS
        packed["quiet_start"] = False
        packed["mode"] = "locked_thin"
        packed_rows.append(packed)
        tw = packed["this_week"]
        week_rows.append(
            {
                "region": region_id,
                "label": packed["label"],
                "magnitude_min": mag,
                "horizon_days": HORIZON_DAYS,
                "quiet_start": False,
                "mode": "locked_thin",
                "p": tw["p"],
                "fresh_raw_p": tw.get("fresh_raw_p"),
                "fresh_raw_chance_plain": tw.get("fresh_raw_chance_plain") or _chance(tw.get("fresh_raw_p")),
                "held_to_climatology": tw.get("held_to_climatology"),
                "chance_plain": tw["chance_plain"],
                "climatology_p": tw["climatology_p"],
                "climatology_chance_plain": tw["climatology_chance_plain"],
                "why": packed.get("why") or [],
                "issue_day": tw["issue_day"],
                "sentence": _week_sentence(packed, tw, HORIZON_DAYS),
            }
        )
    locked = {
        "id": "fresh_locked_thin",
        "mode": "locked_thin",
        "horizon_days": HORIZON_DAYS,
        "lookback_days": max(LOCKED_THIN_LAGS),
        "writes_issued_p": False,
        "scanned_at": iso(datetime.now(timezone.utc)),
        "n_features": int(design[0].shape[1]),
        "n_locked": len(LOCKED_THIN),
        "locked": sorted(LOCKED_THIN),
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
    existing["locked_thin"] = locked
    existing["writes_issued_p"] = False
    existing["catalog_end"] = existing.get("catalog_end") or locked["catalog_end"]
    sealed = _seal_report(existing)
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "mainshock.json", sealed, default=str)
    (dest / "locked_thin.json").write_text(
        json.dumps(sealed.get("locked_thin") or locked, indent=2, default=str),
        encoding="utf-8",
    )
    return sealed
