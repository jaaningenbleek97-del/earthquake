"""Quiet-morning residuals and one pre-registered Hikurangi conjunction.

The residual is the local reading minus the space-weather part, fit only on
earlier quiet mornings. Storm mornings stay blank. These columns are a hunt.
They do not write issued probability, and neither does slow slip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUIET_DST_NT = 20.0
QUIET_KP = 3.0
# Millimetres of extra east motion in 30 days. Fixed before looking at hits.
EAST_TRANSIENT_MM = 10.0
MIN_QUIET = 40
DRIVER_SCALES = {
    "f107": 100.0,
    "kp": 3.0,
    "dst_nt": 20.0,
    "ae_nt": 100.0,
    "solar_wind_speed_kms": 400.0,
}
TARGETS = (
    ("tec_hikurangi_tecu", "qres_tec_hikurangi"),
    ("tec_alpine_tecu", "qres_tec_alpine"),
    ("tec_kermadec_tecu", "qres_tec_kermadec"),
    ("roti_nz", "qres_roti_nz"),
    ("swarm_hikurangi_f_res_nt", "qres_swarm_hikurangi"),
    ("swarm_alpine_f_res_nt", "qres_swarm_alpine"),
    ("swarm_kermadec_f_res_nt", "qres_swarm_kermadec"),
)


def dobrovolsky_km(magnitude: float) -> float:
    """Kilometres. The daily table has box means, not a station inside this disk."""
    return float(10.0 ** (0.43 * float(magnitude)))


def quiet_mornings(dst: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """|Dst|≤20 nT and Kp<3. A missing index is not quiet."""
    dst = np.asarray(dst, dtype=float)
    kp = np.asarray(kp, dtype=float)
    return np.isfinite(dst) & np.isfinite(kp) & (np.abs(dst) <= QUIET_DST_NT) & (kp < QUIET_KP)


def expanding_residual(observed: np.ndarray, drivers: np.ndarray, quiet: np.ndarray) -> np.ndarray:
    """Observed minus the space-weather fit from earlier quiet mornings only."""
    y = np.asarray(observed, dtype=float)
    x = np.asarray(drivers, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    gate = np.asarray(quiet, dtype=bool)
    n, k = x.shape
    out = np.full(n, np.nan)
    xtx = np.zeros((k, k))
    xty = np.zeros(k)
    seen = 0
    for i in range(n):
        row = x[i]
        if not gate[i] or not np.isfinite(y[i]) or not np.all(np.isfinite(row)):
            continue
        if seen >= MIN_QUIET:
            beta, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
            if np.all(np.isfinite(beta)):
                out[i] = float(y[i] - row @ beta)
        xtx += np.outer(row, row)
        xty += row * y[i]
        seen += 1
    return out


def _driver_matrix(frame: pd.DataFrame) -> np.ndarray | None:
    columns = [name for name in DRIVER_SCALES if name in frame.columns]
    if len(columns) < 3:
        return None
    parts = [np.ones(len(frame))]
    for name in columns:
        scale = DRIVER_SCALES[name]
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        parts.append(values / scale)
    return np.column_stack(parts)


def quiet_residual_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach quiet residuals and the one Hikurangi conjunction. Missing inputs stay absent."""
    if frame.empty or "dst_nt" not in frame.columns or "kp" not in frame.columns:
        return frame
    work = frame.copy()
    dst = pd.to_numeric(work["dst_nt"], errors="coerce").to_numpy(dtype=float)
    kp = pd.to_numeric(work["kp"], errors="coerce").to_numpy(dtype=float)
    quiet = quiet_mornings(dst, kp)
    drivers = _driver_matrix(work)
    if drivers is None:
        return work
    made: dict[str, np.ndarray] = {}
    for source, dest in TARGETS:
        if source not in work.columns:
            continue
        observed = pd.to_numeric(work[source], errors="coerce").to_numpy(dtype=float)
        made[dest] = expanding_residual(observed, drivers, quiet)
    if not made:
        return work
    extra = pd.DataFrame(made, index=work.index)
    laic = _laic_hikurangi(work, made.get("qres_tec_hikurangi"), quiet)
    if laic is not None:
        extra["laic_hikurangi"] = laic
    return pd.concat([work, extra], axis=1)


def _laic_hikurangi(
    frame: pd.DataFrame,
    tec_residual: np.ndarray | None,
    quiet: np.ndarray,
) -> np.ndarray | None:
    """Quiet gate, Hikurangi TEC residual, and slow slip or a 30-day east transient.

    Thrust only. The Alpine Fault is strike-slip, so it is not this column.
    """
    if tec_residual is None:
        return None
    slip = np.zeros(len(frame), dtype=bool)
    if "sse_hikurangi" in frame.columns:
        sse = pd.to_numeric(frame["sse_hikurangi"], errors="coerce").to_numpy(dtype=float)
        slip |= np.isfinite(sse) & (sse > 0)
    for name in ("gnss_wgtn_east_extra_30d", "gnss_raul_east_extra_30d", "gnss_wgtn_slowslip_30d"):
        if name not in frame.columns:
            continue
        east = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        slip |= np.isfinite(east) & (np.abs(east) >= EAST_TRANSIENT_MM)
    out = np.full(len(frame), np.nan)
    on = quiet & slip & np.isfinite(tec_residual)
    out[on] = tec_residual[on]
    return out
