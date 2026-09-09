from __future__ import annotations

"""Structured card attribution.

Every issued card lists the terms that entered the call as
`{driver, lag_hours, contribution}`. Fusion contribution is log-rate
(`ln g`). Alarm contribution is 1 for a gate that fired, or tail-depth
for anti-prediction. Solar flags do not appear on fusion cards — they
do not multiply issued p.
"""

import math
from typing import Any, Iterable

# Locked lags: how the feature was built, not fit to holdout.
LAG_HOURS: dict[str, int] = {
    "spf_significant_window": 40 * 24,
    "proton_spike": 0,
    "proton_density_lag1d": 24,
    "kp_high": 0,
    "dst_storm": 0,
    "near_new_or_full": 0,
    "etas_calibrated": 0,
    "etas_quiet": 0,
    "catalog_nowcast": 7 * 24,
    "tec_tecu": 0,
    "tec_nz_tecu": 0,
    "era5_sp_pa": 0,
    "era5_swvl1": 0,
    "mag_h_nt": 0,
    "mag_h_range_nt": 0,
    "mag_dbdt_nt": 0,
    "gnss_east_mm": 0,
    "gnss_east_7d": 7 * 24,
    "n_geonet_m3_7d": 7 * 24,
    "anti_quiet": 0,
    "issued_p_low": 0,
}


def lag_hours(driver: str) -> int:
    name = str(driver or "")
    if name in LAG_HOURS:
        return int(LAG_HOURS[name])
    if name.startswith("mig_"):
        return 14 * 24
    if name.startswith("coulomb_") or name.startswith("cusp_"):
        return 0
    if name.endswith("_7d"):
        return 7 * 24
    return 0


def log_contrib(gain: float) -> float:
    g = float(gain)
    if not math.isfinite(g) or g <= 0.0:
        return 0.0
    return float(round(math.log(g), 6))


def attr(driver: str, hours: int | None = None, contribution: float = 1.0) -> dict:
    name = str(driver)
    return {
        "driver": name,
        "lag_hours": int(hours if hours is not None else lag_hours(name)),
        "contribution": float(contribution),
    }


def driver_name(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("driver") or "")
    return str(item or "")


def driver_names(drivers: Iterable[object] | None) -> list[str]:
    names: list[str] = []
    for item in drivers or []:
        name = driver_name(item)
        if name:
            names.append(name)
    return names


def ensure_attrs(drivers: list | None) -> list[dict]:
    """Coerce a card's drivers list to structured objects. Empty is invalid."""
    out: list[dict] = []
    for item in drivers or []:
        if isinstance(item, dict) and item.get("driver"):
            name = str(item["driver"])
            hours = item.get("lag_hours")
            contrib = item.get("contribution")
            out.append(
                attr(
                    name,
                    int(hours) if hours is not None else None,
                    1.0 if contrib is None else float(contrib),
                )
            )
        elif item:
            name = str(item)
            out.append(attr(name))
    if not out:
        raise ValueError("drivers must be a non-empty structured list")
    return out


def fusion_attrs(gains: dict[str, float]) -> list[dict]:
    """One row per issued multiplier. Contribution = ln(gain)."""
    rows = [attr(name, contribution=log_contrib(gain)) for name, gain in gains.items() if name]
    rows.sort(key=lambda row: (-abs(float(row["contribution"])), str(row["driver"])))
    return rows


def alarm_attrs(*names: str) -> list[dict]:
    return [attr(name, contribution=1.0) for name in names if name]


def anti_attrs(issued_p: float, p_max: float | None) -> list[dict]:
    depth = 0.0
    if p_max is not None and math.isfinite(float(p_max)) and float(p_max) > 0 and math.isfinite(issued_p):
        depth = float(min(1.0, max(0.0, (float(p_max) - float(issued_p)) / float(p_max))))
    return [
        attr("anti_quiet", contribution=1.0),
        attr("issued_p_low", contribution=round(depth, 6)),
    ]


def flatten_driver_field(raw: Any) -> list[str]:
    """Episode / UI helper: accept a list of strings or objects."""
    if isinstance(raw, list):
        return driver_names(raw)
    return []
