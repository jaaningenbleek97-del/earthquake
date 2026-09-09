from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.faults import FAULTS


def resolved_coulomb(
    tidal_strain: float | np.ndarray,
    lunar_decl_deg: float | np.ndarray,
    *,
    strike: float,
    dip: float,
    rake: float,
    mu: float = 0.4,
) -> np.ndarray:
    """Project the scalar tide onto a receiver fault.

    Extension azimuth is a locked function of lunar declination (high |decl|
    → more N–S). This is a geometric tide, not a 3D body-tide tensor.
    """
    tide = np.asarray(tidal_strain, dtype=float)
    decl = np.asarray(lunar_decl_deg, dtype=float)
    ext_az = 90.0 - decl
    alpha = np.radians(strike - ext_az)
    dip_r = np.radians(dip)
    rake_r = np.radians(rake)
    e_nn = tide * np.cos(alpha) ** 2
    e_ss = tide * np.sin(alpha) ** 2
    e_ns = tide * np.sin(alpha) * np.cos(alpha)
    dn = e_nn * np.sin(dip_r) ** 2
    dt = e_ns * np.cos(dip_r) * np.cos(rake_r) + e_ss * np.sin(rake_r) * np.sin(dip_r)
    return dt + mu * dn


def coulomb_frame(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"issue_time": features["issue_time"]})
    tide = features["tidal_strain"] if "tidal_strain" in features.columns else 0.0
    decl = features["lunar_declination_deg"] if "lunar_declination_deg" in features.columns else 0.0
    for rid, fault in FAULTS.items():
        out[f"coulomb_{rid}"] = resolved_coulomb(tide, decl, **fault)
    return out
