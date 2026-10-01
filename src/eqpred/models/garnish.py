from __future__ import annotations

"""Leftover stack: what the unconventional mix would say this morning.

Issued cards stay calibrated ETAS until this mix beats it on later years.
This module still applies the trained leftovers so the lab can show our call.
"""

import math
from pathlib import Path

import pandas as pd

from eqpred.models.box_gain import apply_trained_gain
from eqpred.models.calibrate import issued_scale, load_calibration
from eqpred.models.coulomb import load_coulomb
from eqpred.models.cusp import load_cusp
from eqpred.models.gr import load_gr, rate_to_p
from eqpred.models.migration import load_migration
from eqpred.models.nowcast import load_nowcast
from eqpred.models.predict import load_models
from eqpred.models.wave2 import load_wave2

WATCH_FLAGS = (
    ("spf_significant_window", "Solar proton-flare window"),
    ("proton_spike", "Proton spike"),
    ("kp_high", "Magnetic storm (Kp)"),
    ("dst_storm", "Magnetic storm (Dst)"),
    ("near_new_or_full", "New or full moon"),
    ("neutron_forbush", "Cosmic-ray Forbush decrease"),
    ("cosmic_gle", "Ground-level solar cosmic-ray event"),
    ("cme_halo", "Halo coronal mass ejection"),
    ("dust_fireball", "Fireball / meteoroid"),
    ("lunar_eclipse_window", "Eclipse geometry window"),
)

GAIN_LABELS = {
    "space_weather": "space weather and tides",
    "recent_quake_energy": "recent quake energy",
    "ionosphere": "upper-atmosphere electrons",
    "tidal_stress": "tidal stress on the fault",
    "storm_alignment": "storm-to-fault alignment",
    "local_quakes": "local small quakes",
    "fault_migration": "quakes walking along the fault",
    "mag": "the local magnetic field",
    "gnss": "GPS ground motion",
    "era5": "weather and ground loading",
    "tec": "upper-atmosphere electrons",
    "geonet": "local small quakes",
}


def _as_frame(feat_row: pd.Series | pd.DataFrame | dict) -> pd.DataFrame:
    if isinstance(feat_row, pd.DataFrame):
        return feat_row
    if isinstance(feat_row, dict):
        return pd.DataFrame([feat_row])
    return feat_row.to_frame().T


def _gate_on(meta: dict | None) -> bool:
    if not meta:
        return False
    return bool(meta.get("train_gate_pass") or meta.get("used_on_issued"))


def _wave2_metas(payload: dict, region_id: str) -> list[dict]:
    region = (payload.get("regions") or {}).get(region_id) or {}
    cols = region.get("columns") or []
    return [dict(meta) for meta in cols if _gate_on(meta)]


def leftover_gains(
    feat_row: pd.Series | pd.DataFrame | dict,
    region_id: str,
    root: Path | None = None,
) -> dict[str, float]:
    """Trained leftover multipliers. 1.0 means that term did not move the rate."""
    frame = _as_frame(feat_row)
    row = frame.iloc[0]
    _etas, mod = load_models(root)
    nowcast, now_meta = load_nowcast(root)
    wave2 = load_wave2(root)
    coulomb = load_coulomb(root)
    cusp = load_cusp(root)
    migration = load_migration(root)

    parts: dict[str, float] = {}
    solar = float(mod.gain(frame)[0])
    if abs(solar - 1.0) > 1e-4:
        parts["space_weather"] = solar
    if _gate_on(now_meta):
        now = float(nowcast.gain(frame)[0]) * float(now_meta.get("nowcast_scale") or 1.0)
        if abs(now - 1.0) > 1e-4:
            parts["recent_quake_energy"] = now

    rid = "global" if region_id in {"global", ""} else region_id
    if rid == "alpine_fault":
        rid = "alpine"
    iono = 1.0
    local = 1.0
    for meta in _wave2_metas(wave2, rid):
        column = str(meta.get("column") or "")
        family = str(meta.get("family") or "")
        gain = apply_trained_gain(meta, row.get(column) if column else None)
        if family == "tec" or column.startswith("tec_"):
            iono *= gain
        elif family == "geonet" or "geonet" in column:
            local *= gain
        elif abs(gain - 1.0) > 1e-4:
            parts[family or column] = gain
    if abs(iono - 1.0) > 1e-4:
        parts["ionosphere"] = iono
    if abs(local - 1.0) > 1e-4:
        parts["local_quakes"] = local

    if rid != "global":
        coul = (coulomb.get("regions") or {}).get(rid) or {}
        if _gate_on(coul):
            g = apply_trained_gain(coul, row.get(coul.get("column") or f"coulomb_{rid}"))
            if abs(g - 1.0) > 1e-4:
                parts["tidal_stress"] = g
        cu = (cusp.get("regions") or {}).get(rid) or {}
        if _gate_on(cu):
            g = apply_trained_gain(cu, row.get(cu.get("column") or f"cusp_align_{rid}"))
            if abs(g - 1.0) > 1e-4:
                parts["storm_alignment"] = g
        mig = (migration.get("regions") or {}).get(rid) or {}
        if _gate_on(mig):
            g = apply_trained_gain(mig, row.get(mig.get("column") or f"mig_updip_{rid}"))
            if abs(g - 1.0) > 1e-4:
                parts["fault_migration"] = g
    return parts


def lam0_from_etas_p(p, mag: float, m0: float, b: float, horizon_days: float) -> float:
    """Invert the Poisson card so we can replay leftover on the same morning."""
    clipped = min(max(float(p), 1e-12), 1.0 - 1e-12)
    daily_at_mag = -math.log(1.0 - clipped) / float(horizon_days)
    dm = max(float(mag) - float(m0), 0.0)
    factor = 10.0 ** (-float(b) * dm)
    return daily_at_mag / factor if factor else daily_at_mag


def leftover_p(
    lam0: float,
    feat_row: pd.Series | pd.DataFrame | dict,
    region_id: str,
    *,
    mag: float,
    horizon_days: int = 7,
    root: Path | None = None,
) -> dict:
    cal = load_calibration(root)
    gr = load_gr(root)
    m0 = float(gr.get("m0") or 4.5)
    b = float(gr.get("b") or 1.0)
    scale = issued_scale(cal, "global" if region_id in {"global", ""} else region_id, horizon_days)
    parts = leftover_gains(feat_row, region_id, root)
    leftover = 1.0
    for gain in parts.values():
        leftover *= float(gain)
    catalog = rate_to_p(lam0 * scale, mag, m0, b, horizon_days)
    etas = rate_to_p(lam0, mag, m0, b, horizon_days)
    value = rate_to_p(lam0 * scale * leftover, mag, m0, b, horizon_days)
    why = [GAIN_LABELS.get(name, name.replace("_", " ")) for name in parts]
    return {
        "leftover_p": value,
        "catalog_p": catalog,
        "etas_p": etas,
        "leftover_gain": leftover,
        "parts": [{"name": name, "label": GAIN_LABELS.get(name, name.replace("_", " ")), "gain": gain} for name, gain in parts.items()],
        "why": why,
    }


def watch_list(feat_row: dict | pd.Series | None) -> list[dict]:
    if feat_row is None:
        return []
    row = feat_row if isinstance(feat_row, dict) else feat_row.to_dict()
    out = []
    for key, label in WATCH_FLAGS:
        on = float(pd.to_numeric(row.get(key), errors="coerce") or 0) > 0.5
        out.append({"id": key, "label": label, "on": on})
    return out
