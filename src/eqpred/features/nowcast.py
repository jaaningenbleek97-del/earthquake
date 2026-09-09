from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.catalog import apply_completeness, completeness_mag_series
from eqpred.models.gr import fit_b_value
from eqpred.stamp import parse_utc

# Catalog physics residual to ETAS: trailing b and energy, not raw counts.
NOWCAST_COLS = (
    "b_90d",
    "log_energy_7d",
    "log_energy_30d",
    "energy_accel",
)


def catalog_nowcast_frame(
    quakes: pd.DataFrame,
    issue_times: pd.Series,
    m0: float = 4.5,
    versions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Point-in-time trailing b-value and seismic energy. available_at only."""
    issue = pd.DatetimeIndex(parse_utc(issue_times))
    empty = pd.DataFrame(
        {
            "issue_time": issue,
            "b_90d": np.nan,
            "log_energy_7d": 0.0,
            "log_energy_30d": 0.0,
            "energy_accel": 1.0,
            "n_m45_90d": 0,
        }
    )
    if quakes is None or quakes.empty:
        return empty

    has_versions = (
        versions is not None
        and not versions.empty
        and "event_id" in quakes.columns
        and "event_id" in versions.columns
    )
    if has_versions:
        return _nowcast_asof(quakes, issue, m0, versions, empty)

    work = apply_completeness(quakes.loc[quakes["magnitude"] >= m0].copy())
    if work.empty:
        return empty
    return _nowcast_from_arrays(work, issue, m0, empty)


def _nowcast_from_arrays(
    work: pd.DataFrame,
    issue: pd.DatetimeIndex,
    m0: float,
    empty: pd.DataFrame,
) -> pd.DataFrame:
    avail = parse_utc(work["available_at"]).to_numpy(dtype="datetime64[ns]")
    obs = parse_utc(work["observed_at"]).to_numpy(dtype="datetime64[ns]")
    mag = pd.to_numeric(work["magnitude"], errors="coerce").to_numpy(dtype=float)
    energy = np.power(10.0, 1.5 * (mag - m0))
    t = issue.to_numpy(dtype="datetime64[ns]")
    w7 = np.timedelta64(7, "D")
    w30 = np.timedelta64(30, "D")
    w90 = np.timedelta64(90, "D")

    b90 = np.full(len(t), np.nan)
    e7 = np.zeros(len(t))
    e30 = np.zeros(len(t))
    n90 = np.zeros(len(t), dtype=int)
    for i, stamp in enumerate(t):
        vis = avail <= stamp
        in90 = vis & (obs >= stamp - w90) & (obs <= stamp)
        in30 = vis & (obs >= stamp - w30) & (obs <= stamp)
        in7 = vis & (obs >= stamp - w7) & (obs <= stamp)
        mags = mag[in90]
        n90[i] = int(mags.size)
        if mags.size >= 50:
            b90[i] = fit_b_value(mags, m0)
        e7[i] = float(energy[in7].sum())
        e30[i] = float(energy[in30].sum())

    accel = (e7 * (30.0 / 7.0)) / np.maximum(e30, 1e-9)
    out = empty.copy()
    out["b_90d"] = b90
    out["log_energy_7d"] = np.log1p(e7)
    out["log_energy_30d"] = np.log1p(e30)
    out["energy_accel"] = accel
    out["n_m45_90d"] = n90
    return out


def _nowcast_asof(
    quakes: pd.DataFrame,
    issue: pd.DatetimeIndex,
    m0: float,
    versions: pd.DataFrame,
    empty: pd.DataFrame,
) -> pd.DataFrame:
    """Energy and b use the mag on the wire that morning."""
    from eqpred.catalog import apply_available_at, completeness_mag_series

    vids = set(versions["event_id"].dropna().astype(str))
    eids = quakes["event_id"].astype(str)
    plain = quakes.loc[~eids.isin(vids)].copy()
    if not plain.empty:
        if "available_at" not in plain.columns:
            plain = apply_available_at(plain)
        plain = apply_completeness(plain.loc[plain["magnitude"] >= m0].copy())

    u_avail = u_obs = u_mag = u_energy = None
    if not plain.empty:
        u_avail = parse_utc(plain["available_at"]).to_numpy(dtype="datetime64[ns]")
        u_obs = parse_utc(plain["observed_at"]).to_numpy(dtype="datetime64[ns]")
        u_mag = pd.to_numeric(plain["magnitude"], errors="coerce").to_numpy(dtype=float)
        u_energy = np.power(10.0, 1.5 * (u_mag - m0))

    steps = _version_step_tables(quakes, versions)
    t = issue.to_numpy(dtype="datetime64[ns]")
    w7 = np.timedelta64(7, "D")
    w30 = np.timedelta64(30, "D")
    w90 = np.timedelta64(90, "D")
    b90 = np.full(len(t), np.nan)
    e7 = np.zeros(len(t))
    e30 = np.zeros(len(t))
    n90 = np.zeros(len(t), dtype=int)
    for i, stamp in enumerate(t):
        mags: list[np.ndarray] = []
        en7 = 0.0
        en30 = 0.0
        if u_mag is not None:
            vis = u_avail <= stamp
            in90 = vis & (u_obs >= stamp - w90) & (u_obs <= stamp)
            in30 = vis & (u_obs >= stamp - w30) & (u_obs <= stamp)
            in7 = vis & (u_obs >= stamp - w7) & (u_obs <= stamp)
            mags.append(u_mag[in90])
            en7 += float(u_energy[in7].sum())
            en30 += float(u_energy[in30].sum())
        if steps is not None:
            mag_t, ok = _asof_mags(steps, stamp, m0)
            v_obs = steps["obs"]
            in90 = ok & (v_obs >= stamp - w90) & (v_obs <= stamp)
            in30 = ok & (v_obs >= stamp - w30) & (v_obs <= stamp)
            in7 = ok & (v_obs >= stamp - w7) & (v_obs <= stamp)
            vm = mag_t[in90]
            mags.append(vm)
            en7 += float(np.sum(np.power(10.0, 1.5 * (mag_t[in7] - m0))))
            en30 += float(np.sum(np.power(10.0, 1.5 * (mag_t[in30] - m0))))
        if mags:
            combined = np.concatenate(mags) if len(mags) > 1 else mags[0]
        else:
            combined = np.array([], dtype=float)
        n90[i] = int(combined.size)
        if combined.size >= 50:
            b90[i] = fit_b_value(combined, m0)
        e7[i] = en7
        e30[i] = en30
    accel = (e7 * (30.0 / 7.0)) / np.maximum(e30, 1e-9)
    out = empty.copy()
    out["b_90d"] = b90
    out["log_energy_7d"] = np.log1p(e7)
    out["log_energy_30d"] = np.log1p(e30)
    out["energy_accel"] = accel
    out["n_m45_90d"] = n90
    return out


def _version_step_tables(quakes: pd.DataFrame, versions: pd.DataFrame) -> dict | None:
    v = versions.copy()
    v["event_id"] = v["event_id"].astype(str)
    v["version_at"] = parse_utc(v["version_at"])
    v["observed_at"] = parse_utc(v["observed_at"])
    v["magnitude"] = pd.to_numeric(v["magnitude"], errors="coerce")
    status = v["status"].astype(str).str.lower() if "status" in v.columns else "reviewed"
    v["deleted"] = status == "deleted"
    v = v.sort_values(["event_id", "version_at"])
    groups = list(v.groupby("event_id", sort=False))
    if not groups:
        return None
    origins = quakes.copy()
    origins["event_id"] = origins["event_id"].astype(str)
    origins["observed_at"] = parse_utc(origins["observed_at"])
    origin_by = origins.drop_duplicates("event_id").set_index("event_id")["observed_at"]
    n_ev = len(groups)
    max_v = max(len(g) for _, g in groups)
    times = np.full((n_ev, max_v), np.datetime64("NaT", "ns"))
    mags = np.full((n_ev, max_v), np.nan)
    deleted = np.zeros((n_ev, max_v), dtype=bool)
    obs = np.empty(n_ev, dtype="datetime64[ns]")
    for i, (eid, grp) in enumerate(groups):
        n = len(grp)
        times[i, :n] = grp["version_at"].to_numpy(dtype="datetime64[ns]")
        mags[i, :n] = grp["magnitude"].to_numpy(dtype=float)
        deleted[i, :n] = grp["deleted"].to_numpy()
        if eid in origin_by.index:
            obs[i] = origin_by.loc[eid].to_datetime64()
        else:
            obs[i] = grp["observed_at"].iloc[0].to_datetime64()
    mc = completeness_mag_series(pd.Series(pd.to_datetime(obs, utc=True))).to_numpy(dtype=float)
    return {"times": times, "mags": mags, "deleted": deleted, "obs": obs, "mc": mc}


def _asof_mags(steps: dict, stamp, m0: float) -> tuple[np.ndarray, np.ndarray]:
    times = steps["times"]
    idx = (times <= np.datetime64(stamp)).sum(axis=1) - 1
    has = idx >= 0
    n = len(idx)
    mag_t = np.full(n, np.nan)
    gone = np.zeros(n, dtype=bool)
    rows = np.arange(n)
    mag_t[has] = steps["mags"][rows[has], idx[has]]
    gone[has] = steps["deleted"][rows[has], idx[has]]
    ok = has & ~gone & np.isfinite(mag_t) & (mag_t >= m0) & (mag_t >= steps["mc"])
    return mag_t, ok
