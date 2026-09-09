from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eqpred.stamp import parse_utc, utc_ts


@dataclass
class EtasParams:
    mu: float = 0.2
    k: float = 0.04
    alpha: float = 1.2
    c: float = 0.02
    p: float = 1.15
    m0: float = 4.5

    def clip(self) -> "EtasParams":
        return EtasParams(
            mu=float(np.clip(self.mu, 1e-6, 80.0)),
            k=float(np.clip(self.k, 1e-6, 5.0)),
            alpha=float(np.clip(self.alpha, 0.2, 3.0)),
            c=float(np.clip(self.c, 1e-4, 2.0)),
            p=float(np.clip(self.p, 1.01, 2.2)),
            m0=self.m0,
        )


def _times_days(stamps: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return ((stamps - origin) / pd.Timedelta(days=1)).to_numpy(dtype=float)


def _accumulate_omori(
    rate: np.ndarray,
    ti: float,
    t_vis: float,
    t_stop: float,
    kt: float,
    n_days: int,
    max_age_days: int,
    c: float,
    one_m_p: float,
) -> None:
    start = max(int(np.floor(ti)) + 1, int(np.ceil(t_vis - 1e-12)), 0)
    if start >= n_days:
        return
    stop = min(n_days, start + max_age_days)
    if np.isfinite(t_stop):
        stop = min(stop, max(0, int(np.ceil(t_stop - 1e-12))))
    if start >= stop:
        return
    day_idx = np.arange(start, stop)
    t0 = day_idx.astype(float) - ti
    t1 = t0 + 1.0
    t0 = np.clip(t0, 0.0, None)
    integ = kt * ((t1 + c) ** one_m_p - (t0 + c) ** one_m_p) / one_m_p
    rate[day_idx] += np.maximum(integ, 0.0)


def _event_weights(work: pd.DataFrame) -> np.ndarray:
    if "_etas_w" not in work.columns:
        return np.ones(len(work), dtype=float)
    w = pd.to_numeric(work["_etas_w"], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(w), w, 1.0)


def _plain_kterm_loop(
    ev: pd.DataFrame,
    origin: pd.Timestamp,
    params: EtasParams,
    rate: np.ndarray,
    n_days: int,
    max_age_days: int,
    c: float,
    one_m_p: float,
) -> None:
    if ev.empty:
        return
    work = ev.sort_values("observed_at")
    t = _times_days(parse_utc(work["observed_at"]), origin)
    mag = pd.to_numeric(work["magnitude"], errors="coerce").to_numpy(dtype=float)
    kterm = params.k * np.exp(params.alpha * (mag - params.m0)) * _event_weights(work)
    if "available_at" in work.columns:
        t_avail = _times_days(parse_utc(work["available_at"]), origin)
    else:
        t_avail = t
    for ti, t_vis, kt in zip(t, t_avail, kterm):
        if not np.isfinite(kt):
            continue
        _accumulate_omori(rate, float(ti), float(t_vis), np.inf, float(kt), n_days, max_age_days, c, one_m_p)


def _piecewise_kterm_loop(
    ev: pd.DataFrame,
    versions: pd.DataFrame,
    origin: pd.Timestamp,
    params: EtasParams,
    rate: np.ndarray,
    n_days: int,
    max_age_days: int,
    c: float,
    one_m_p: float,
) -> None:
    """Omori kterm follows the mag on the wire that morning, not the final bulletin."""
    v = versions.copy()
    v["event_id"] = v["event_id"].astype(str)
    v["version_at"] = parse_utc(v["version_at"])
    v["magnitude"] = pd.to_numeric(v["magnitude"], errors="coerce")
    if "status" in v.columns:
        v["status"] = v["status"].astype(str).str.lower()
    else:
        v["status"] = "reviewed"
    v = v.sort_values(["event_id", "version_at"])

    origins = ev.copy()
    origins["event_id"] = origins["event_id"].astype(str)
    origins["observed_at"] = parse_utc(origins["observed_at"])
    origin_by_id = origins.drop_duplicates("event_id").set_index("event_id")["observed_at"]

    for eid, grp in v.groupby("event_id", sort=False):
        if eid not in origin_by_id.index:
            continue
        ti = float(_times_days(pd.Series([origin_by_id.loc[eid]]), origin)[0])
        stamps = grp["version_at"].to_numpy()
        mags = grp["magnitude"].to_numpy(dtype=float)
        statuses = grp["status"].to_numpy()
        t_vis_all = _times_days(pd.Series(pd.to_datetime(stamps, utc=True)), origin)
        for i in range(len(grp)):
            if statuses[i] == "deleted":
                break
            if not np.isfinite(mags[i]):
                continue
            t_vis = float(t_vis_all[i])
            t_stop = float(t_vis_all[i + 1]) if i + 1 < len(grp) else np.inf
            weight = 1.0
            if "_etas_w" in ev.columns:
                match = ev.loc[ev["event_id"].astype(str) == str(eid), "_etas_w"]
                if not match.empty:
                    raw = pd.to_numeric(match.iloc[0], errors="coerce")
                    if pd.notna(raw):
                        weight = float(raw)
            kt = params.k * np.exp(params.alpha * (float(mags[i]) - params.m0)) * weight
            _accumulate_omori(rate, ti, t_vis, t_stop, kt, n_days, max_age_days, c, one_m_p)


def daily_background_and_cluster(
    events: pd.DataFrame,
    days: pd.DatetimeIndex,
    params: EtasParams,
    max_age_days: int = 1800,
    versions: pd.DataFrame | None = None,
) -> np.ndarray:
    """Expected event count per calendar day from temporal ETAS (history only)."""
    from eqpred.catalog import apply_completeness

    params = params.clip()
    origin = days[0]
    n_days = len(days)
    rate = np.full(n_days, params.mu, dtype=float)
    if events is None or events.empty:
        return rate

    ev = apply_completeness(events)
    if ev.empty:
        return rate

    c = params.c
    one_m_p = 1.0 - params.p
    has_versions = (
        versions is not None
        and not versions.empty
        and "event_id" in ev.columns
        and "event_id" in versions.columns
    )
    if not has_versions:
        _plain_kterm_loop(ev, origin, params, rate, n_days, max_age_days, c, one_m_p)
        return rate

    versioned_ids = set(versions["event_id"].dropna().astype(str))
    eids = ev["event_id"].astype(str)
    _plain_kterm_loop(
        ev.loc[~eids.isin(versioned_ids)],
        origin,
        params,
        rate,
        n_days,
        max_age_days,
        c,
        one_m_p,
    )
    _piecewise_kterm_loop(
        ev.loc[eids.isin(versioned_ids)],
        versions,
        origin,
        params,
        rate,
        n_days,
        max_age_days,
        c,
        one_m_p,
    )
    return rate


def _etas_nll(
    t: np.ndarray,
    mag: np.ndarray,
    duration: float,
    mu: float,
    k: float,
    alpha: float,
    c: float,
    p: float,
    m0: float,
) -> float:
    """Vectorized temporal ETAS negative log-likelihood."""
    n = int(t.size)
    if n == 0 or duration <= 0:
        return 1e12
    dt = t[:, None] - t[None, :]
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    safe = lower & (dt > 0)
    kterm = k * np.exp(alpha * (mag - m0))
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = kterm[None, :] / np.power(np.maximum(dt, 0.0) + c, p)
    lam = np.clip(mu + np.where(safe, contrib, 0.0).sum(axis=1), 1e-12, None)
    one_m_p = 1.0 - p
    remain = np.clip(duration - t, 0.0, None)
    integ = mu * duration + np.sum(
        k * np.exp(alpha * (mag - m0)) * ((remain + c) ** one_m_p - c**one_m_p) / one_m_p
    )
    return float(-(np.log(lam).sum() - integ))


ETAS_MLE_MAX_EVENTS = 4000


def _even_subsample(ev: pd.DataFrame, max_events: int) -> tuple[pd.DataFrame, float]:
    """Keep the full window. Independent thinning: μ_obs = πμ, k_obs = πk."""
    n = len(ev)
    if n <= max_events:
        return ev, 1.0
    idx = np.unique(np.rint(np.linspace(0, n - 1, max_events)).astype(int))
    keep = ev.iloc[idx].copy()
    return keep, float(len(keep) / n)


def _quiet_day_mu(ev: pd.DataFrame, start: datetime, end: datetime) -> float:
    """Background rate from the quiet quarter of daily counts on the full window."""
    origin = utc_ts(start)
    stop = utc_ts(end)
    duration = float((stop - origin) / pd.Timedelta(days=1))
    if ev.empty or duration <= 0:
        return 0.2
    days = pd.date_range(origin.floor("D"), stop.floor("D"), freq="D", tz="UTC")
    if len(days) >= 2:
        days = days[:-1]
    counts = ev.groupby(ev["observed_at"].dt.floor("D")).size()
    daily = counts.reindex(days, fill_value=0).to_numpy(dtype=float)
    if daily.size == 0:
        return float(np.clip(len(ev) / duration, 1e-6, 80.0))
    return float(np.clip(np.quantile(daily, 0.25), 1e-6, 80.0))


def fit_etas(
    events: pd.DataFrame,
    start: datetime,
    end: datetime,
    m0: float = 4.5,
) -> EtasParams:
    """Lock μ from quiet days on [start, end). Fit Omori on an even thin; rescale k by 1/π."""
    from eqpred.catalog import apply_completeness

    ev = apply_completeness(events)
    ev = ev.loc[ev["magnitude"] >= m0].copy()
    ev["observed_at"] = parse_utc(ev["observed_at"])
    ev = ev[(ev["observed_at"] >= start) & (ev["observed_at"] < end)].sort_values("observed_at")
    n_full = len(ev)
    origin = utc_ts(start)
    duration = float((utc_ts(end) - origin) / pd.Timedelta(days=1))
    mu_full = _quiet_day_mu(ev, start, end)
    ev, pi = _even_subsample(ev, ETAS_MLE_MAX_EVENTS)
    if len(ev) < 50:
        return EtasParams(mu=mu_full, m0=m0).clip()

    t = _times_days(ev["observed_at"], origin)
    mag = pd.to_numeric(ev["magnitude"], errors="coerce").to_numpy(dtype=float)
    mu_obs = float(mu_full * pi)
    # x = log k, α, log c, p — μ stays locked on the thinned catalog
    seed = np.array([np.log(max(0.04 * pi, 1e-6)), 1.1, np.log(0.02), 1.12])
    bounds = [(-12.0, 1.6), (0.2, 3.0), (-9.0, 0.7), (1.01, 2.2)]

    def nll(x: np.ndarray) -> float:
        params = EtasParams(
            mu=mu_obs,
            k=float(np.exp(x[0])),
            alpha=float(x[1]),
            c=float(np.exp(x[2])),
            p=float(x[3]),
            m0=m0,
        ).clip()
        return _etas_nll(t, mag, duration, params.mu, params.k, params.alpha, params.c, params.p, m0)

    try:
        res = minimize(nll, seed, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
        x = res.x if res.success or np.isfinite(res.fun) else seed
    except Exception:
        x = seed
    scale = 1.0 / max(pi, 1e-12)
    print(
        f"etas MLE window {n_full} events, kept {len(ev)}, pi={pi:.4f}, "
        f"duration={duration:.1f}d, mu_locked={mu_full:.3f}"
    )
    return EtasParams(
        mu=mu_full,
        k=float(np.exp(x[0]) * scale),
        alpha=float(x[1]),
        c=float(np.exp(x[2])),
        p=float(x[3]),
        m0=m0,
    ).clip()


def params_to_dict(params: EtasParams) -> dict:
    return asdict(params)


def params_from_dict(data: dict) -> EtasParams:
    return EtasParams(**{k: data[k] for k in EtasParams.__dataclass_fields__})
