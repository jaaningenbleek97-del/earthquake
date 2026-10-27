from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.models.etas import daily_background_and_cluster
from eqpred.models.predict import load_models
from eqpred.models.quiet import UNCONVENTIONAL, is_quiet, load_quiet, quiet_threshold
from eqpred.paths import data_dir, feature_dir
from eqpred.regions import REGIONS, in_bbox
from eqpred.stamp import parse_utc, utc_ts


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    label: str
    region_id: str
    magnitude_min: float
    horizon_days: int
    flags: tuple[str, ...] = ()
    require_quiet: bool = False
    require_any_unconv: bool = False
    locked: bool = False
    quiet_quantile: float | None = None
    require_prior_quiet: bool = False
    prior_quiet_days: int = 10
    score_fresh: bool = False


# Locked = already a product head. The rest are paper trades: same walk-forward
# rules, not findings until they survive a later window we have not peeked at.
BOOK: tuple[Strategy, ...] = (
    Strategy(
        "spf_m8",
        "SPF window → global M8 / 20d",
        "global",
        8.0,
        20,
        flags=("spf_significant_window",),
        locked=True,
    ),
    Strategy(
        "quiet_unconv_m7",
        "Quiet ETAS + driver → global M7 / 7d",
        "global",
        7.0,
        7,
        require_quiet=True,
        require_any_unconv=True,
        locked=True,
    ),
    Strategy(
        "proton_hikurangi_m5",
        "Proton spike → Hikurangi M5 / 7d",
        "hikurangi",
        5.0,
        7,
        flags=("proton_spike",),
    ),
    Strategy(
        "dst_hikurangi_m5",
        "Dst storm → Hikurangi M5 / 7d",
        "hikurangi",
        5.0,
        7,
        flags=("dst_storm",),
    ),
    Strategy(
        "moon_alpine_m5",
        "New/full moon → Alpine M5 / 7d",
        "alpine",
        5.0,
        7,
        flags=("near_new_or_full",),
    ),
    Strategy(
        "spf_kermadec_m6",
        "SPF window → Kermadec M6 / 7d",
        "kermadec",
        6.0,
        7,
        flags=("spf_significant_window",),
    ),
    Strategy(
        "quiet_etas_fresh_m7",
        "Quiet ETAS only → global M7 / 7d (fresh)",
        "global",
        7.0,
        7,
        require_quiet=True,
        score_fresh=True,
    ),
    Strategy(
        "quiet_wide_fresh_m7",
        "Wider quiet ETAS (q=0.50) → global M7 / 7d (fresh)",
        "global",
        7.0,
        7,
        require_quiet=True,
        quiet_quantile=0.50,
        score_fresh=True,
    ),
    Strategy(
        "quiet_start_fresh_m7",
        "No recent M7 → global M7 / 7d (fresh)",
        "global",
        7.0,
        7,
        require_prior_quiet=True,
        score_fresh=True,
    ),
)


def _window_hits(
    quakes: pd.DataFrame,
    region: dict,
    mag: float,
    starts: pd.DatetimeIndex,
    horizon_days: int,
) -> np.ndarray:
    work = quakes.loc[quakes["magnitude"] >= mag]
    if region.get("kind") == "bbox":
        work = work.loc[in_bbox(work, region)]
    if work.empty:
        return np.zeros(len(starts), dtype=bool)
    times = np.sort(work["observed_at"].to_numpy(dtype="datetime64[ns]"))
    t0 = starts.to_numpy(dtype="datetime64[ns]")
    t1 = t0 + np.timedelta64(int(horizon_days), "D")
    left = np.searchsorted(times, t0, side="left")
    right = np.searchsorted(times, t1, side="left")
    return (right - left) > 0


def _entries(
    frame: pd.DataFrame,
    rule: Strategy,
    quiet_cut: float,
    prior_quiet_mask: np.ndarray | None = None,
) -> np.ndarray:
    on = np.ones(len(frame), dtype=bool)
    for flag in rule.flags:
        if flag not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        on = on & (frame[flag].to_numpy(dtype=float) > 0.5)
    if rule.require_any_unconv:
        any_un = np.zeros(len(frame), dtype=bool)
        for flag in UNCONVENTIONAL:
            if flag in frame.columns:
                any_un = any_un | (frame[flag].to_numpy(dtype=float) > 0.5)
        on = on & any_un
    if rule.require_quiet:
        lam = frame["lambda_etas"].to_numpy(dtype=float)
        on = on & np.array([is_quiet(float(x), quiet_cut) for x in lam])
    if rule.require_prior_quiet:
        if prior_quiet_mask is None:
            return np.zeros(len(frame), dtype=bool)
        on = on & prior_quiet_mask
    return on


def run_strategies(
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    book: tuple[Strategy, ...] | None = None,
) -> list[dict]:
    features = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    features["issue_time"] = parse_utc(features["issue_time"])
    quakes = load_comcat(root)
    versions = load_comcat_versions(root)
    etas, _mod = load_models(root)
    hist = quakes.loc[quakes["magnitude"] >= etas.m0].copy()
    days = pd.DatetimeIndex(features["issue_time"])
    features = features.copy()
    features["lambda_etas"] = daily_background_and_cluster(hist, days, etas, versions=versions)
    quiet_cut = float(load_quiet(root).get("lambda_etas_max") or 0.0)

    from eqpred.eval.mainshock import fresh_hits, prior_quiet

    hold = utc_ts(holdout_start)
    train_mask = (features["issue_time"] < hold).to_numpy()
    feat = features.loc[features["issue_time"] >= hold].copy()
    starts = pd.DatetimeIndex(feat["issue_time"])
    n_days = max(int(len(feat)), 1)
    rules = book or BOOK
    reports: list[dict] = []

    for rule in rules:
        region = REGIONS[rule.region_id]
        hits = _window_hits(quakes, region, rule.magnitude_min, starts, rule.horizon_days)
        fresh = fresh_hits(quakes, region, rule.magnitude_min, starts, rule.horizon_days)
        cut = quiet_cut
        if rule.quiet_quantile is not None:
            cut = quiet_threshold(
                features["lambda_etas"].to_numpy(dtype=float),
                train_mask,
                quantile=rule.quiet_quantile,
            )
        pq = None
        if rule.require_prior_quiet:
            pq = prior_quiet(
                quakes,
                region,
                rule.magnitude_min,
                starts,
                lookback_days=rule.prior_quiet_days,
            )
        entered = _entries(feat, rule, cut, prior_quiet_mask=pq)
        n_trades = int(entered.sum())
        n_wins = int((hits & entered).sum())
        n_fresh = int((fresh & entered).sum())
        base = float(hits.mean()) if hits.size else 0.0
        fresh_base = float(fresh.mean()) if fresh.size else 0.0
        win = float(n_wins / n_trades) if n_trades else None
        fresh_win = float(n_fresh / n_trades) if n_trades else None
        _save_rule_forecasts(
            rule=rule,
            region=region,
            quakes=quakes,
            features=features,
            days=days,
            train_mask=train_mask,
            cut=cut,
            hold=hold,
            root=root,
        )
        reports.append(
            {
                **asdict(rule),
                "region": region.get("name"),
                "n_days": n_days,
                "n_trades": n_trades,
                "n_hits": n_wins,
                "fresh_n_hits": n_fresh,
                "time_in_alarm": n_trades / n_days,
                "hit_rate": win,
                "fresh_hit_rate": fresh_win,
                "base_rate": base,
                "fresh_base_rate": fresh_base,
                "lift": None if win is None else win - base,
                "fresh_lift": None if fresh_win is None else fresh_win - fresh_base,
                "quiet_cut": cut,
                "holdout_start": holdout_start,
            }
        )

    dest = data_dir(root) / "eval" / "strategies.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def _save_rule_forecasts(
    *,
    rule: Strategy,
    region: dict,
    quakes: pd.DataFrame,
    features: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    cut: float,
    hold: pd.Timestamp,
    root: Path | None,
) -> None:
    """One attempt per rule. The chance uses only years before the holdout."""
    from eqpred.eval.attempts import forecast_frame, save_forecasts
    from eqpred.eval.honest import week_mask
    from eqpred.eval.mainshock import fresh_hits, prior_quiet

    full_pq = None
    if rule.require_prior_quiet:
        full_pq = prior_quiet(
            quakes,
            region,
            rule.magnitude_min,
            days,
            lookback_days=rule.prior_quiet_days,
        )
    entered = _entries(features, rule, cut, prior_quiet_mask=full_pq)
    fresh = fresh_hits(quakes, region, rule.magnitude_min, days, rule.horizon_days).astype(float)
    train = np.asarray(train_mask, dtype=bool)

    def _rate(mask: np.ndarray) -> float | None:
        chosen = np.asarray(mask, dtype=bool) & np.isfinite(fresh)
        if int(chosen.sum()) < 1:
            return None
        return float(fresh[chosen].mean())

    usual = _rate(train)
    if usual is None:
        return
    p_on = _rate(train & entered)
    p_off = _rate(train & ~entered)
    chance = np.where(entered, usual if p_on is None else p_on, usual if p_off is None else p_off)
    closed = days + pd.Timedelta(days=rule.horizon_days) <= days[-1]
    later = np.asarray((days >= hold) & closed)
    frame = forecast_frame(
        days=days,
        region=rule.region_id,
        label=rule.label,
        horizon=rule.horizon_days,
        magnitude=rule.magnitude_min,
        p=chance,
        climatology=np.full(len(days), usual),
        happened=fresh,
        mask=week_mask(later, stride=rule.horizon_days, fallback=False),
        catalog_end=days[-1],
    )
    save_forecasts(
        frame,
        source="rulebook",
        name=rule.strategy_id,
        label=rule.label,
        root=root,
        label_version=1,
        answer_key="fresh.v1",
        knobs={
            "horizon_days": rule.horizon_days,
            "magnitude_min": rule.magnitude_min,
            "require_quiet": rule.require_quiet,
        },
        note=(
            f"Rule-book replay of {rule.label}. "
            "The chance is the train-era first-shock rate. This does not change the live exam."
        ),
    )
