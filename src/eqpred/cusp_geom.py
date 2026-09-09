from __future__ import annotations

import numpy as np

# IGRF-13 2020 north geomagnetic pole longitude. Locked.
DIPOLE_LON = -72.68
CUSP_WIDTH_DEG = 30.0


def wrap180(lon: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(lon, dtype=float)
    return (arr + 180.0) % 360.0 - 180.0


def cusp_lon_deg(ut_hours: float | np.ndarray) -> np.ndarray:
    """Geographic longitude of 12 MLT (dayside cusp) at this UT.

    MLT ≈ UT + (λ − λ0) / 15, so 12 MLT is λ = λ0 + (12 − UT) × 15.
    Locked dipole, not AACGM.
    """
    hours = np.asarray(ut_hours, dtype=float)
    return wrap180(DIPOLE_LON + (12.0 - hours) * 15.0)


def angle_diff_deg(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return np.abs(wrap180(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def cusp_weight(box_lon: float, cusp_lon: float | np.ndarray, width: float = CUSP_WIDTH_DEG) -> np.ndarray:
    dist = angle_diff_deg(box_lon, cusp_lon)
    return np.exp(-0.5 * (dist / width) ** 2)
