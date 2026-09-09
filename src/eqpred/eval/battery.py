from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.eval.decluster import decluster
from eqpred.eval.nulls import run_fusion_nulls
from eqpred.features.davidson import significant_windows
from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.ingest.run import latest_raw
from eqpred.models.etas import daily_background_and_cluster
from eqpred.models.predict import load_models
from eqpred.models.quiet import load_quiet
from eqpred.paths import feature_dir
from eqpred.stamp import parse_utc, utc_ts


def _load(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quakes = load_comcat(root)
    spf = latest_raw("wilcox.spf", root)
    feats = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feats["issue_time"] = parse_utc(feats["issue_time"])
    return quakes, spf, feats


def _shift_spf(spf: pd.DataFrame, days: int) -> pd.DataFrame:
    out = spf.copy()
    out["observed_at"] = parse_utc(out["observed_at"]) + pd.Timedelta(days=days)
    return out


def davidson_test(
    quakes: pd.DataFrame,
    spf: pd.DataFrame,
    holdout_start: str,
    version: str = "v2",
) -> dict:
    windows = significant_windows(spf, quakes, version=version)
    m8 = quakes.loc[quakes["magnitude"] >= 8.0, "observed_at"].dt.floor("D")
    hold = utc_ts(holdout_start)
    win = windows.loc[windows.index >= hold]
    hits = m8[m8 >= hold]
    if win.empty:
        return {"name": f"davidson_spf_m8_{version}", "error": "no window days"}
    day_frac = float(win.mean())
    aligned = windows.reindex(pd.DatetimeIndex(hits)).fillna(False)
    in_win = int(aligned.sum())
    n = int(len(hits))
    # Monte Carlo: drop M8 times onto the holdout calendar.
    rng = np.random.default_rng(7)
    if n and len(win):
        sim = []
        flags = win.to_numpy()
        for _ in range(2000):
            draw = rng.choice(flags, size=n, replace=True)
            sim.append(draw.sum())
        p_value = float(np.mean(np.array(sim) >= in_win))
    else:
        p_value = 1.0
    return {
        "name": f"davidson_spf_m8_{version}",
        "holdout_start": holdout_start,
        "n_m8": n,
        "m8_in_window": in_win,
        "hit_rate": None if n == 0 else in_win / n,
        "time_in_alarm": day_frac,
        "expected_if_independent": day_frac,
        "monte_carlo_p_ge_observed": p_value,
    }


def marchitelli_test(feats: pd.DataFrame, quakes: pd.DataFrame, holdout_start: str) -> dict:
    hold = utc_ts(holdout_start)
    m6 = quakes.loc[quakes["magnitude"] >= 6.0, "observed_at"].dt.floor("D")
    m6 = m6[m6 >= hold]
    series = feats.set_index("issue_time")["proton_density_lag1d"]
    before = series.reindex(m6).dropna()
    background = series.loc[series.index >= hold].dropna()
    if before.empty or background.empty:
        return {"name": "marchitelli_proton_lag1_m6", "error": "empty"}
    # Bootstrap difference of means.
    rng = np.random.default_rng(11)
    obs = float(before.mean() - background.mean())
    pooled = background.to_numpy()
    sim = []
    for _ in range(2000):
        draw = rng.choice(pooled, size=len(before), replace=True)
        sim.append(draw.mean() - pooled.mean())
    p_value = float(np.mean(np.array(sim) >= obs))
    return {
        "name": "marchitelli_proton_lag1_m6",
        "holdout_start": holdout_start,
        "n_m6": int(len(before)),
        "mean_proton_day_before_m6": float(before.mean()),
        "mean_proton_background": float(background.mean()),
        "difference": obs,
        "bootstrap_p_ge_observed": p_value,
    }


def love_thomas_test(feats: pd.DataFrame, quakes: pd.DataFrame, holdout_start: str) -> dict:
    hold = utc_ts(holdout_start)
    daily = feats.loc[feats["issue_time"] >= hold, ["issue_time", "kp", "dst_nt", "sunspot_number"]]
    q = quakes.loc[quakes["magnitude"] >= 5.0]
    counts = (
        q.assign(day=q["observed_at"].dt.floor("D"))
        .groupby("day")
        .size()
        .rename("n")
    )
    daily = daily.merge(counts, left_on="issue_time", right_index=True, how="left")
    daily["n"] = daily["n"].fillna(0)
    out = {"name": "love_thomas_indices_vs_m5", "holdout_start": holdout_start}
    for col in ("kp", "dst_nt", "sunspot_number"):
        pair = daily[[col, "n"]].dropna()
        if len(pair) < 20:
            out[f"corr_{col}"] = None
            continue
        out[f"corr_{col}"] = float(pair[col].corr(pair["n"]))
    return out


def issued_spf_test(feats: pd.DataFrame, quakes: pd.DataFrame, holdout_start: str) -> dict:
    """Score the window we actually issue (v2 + train budget + 40d delay)."""
    hold = utc_ts(holdout_start)
    flags = feats.set_index("issue_time")["spf_significant_window"] > 0.5
    win = flags.loc[flags.index >= hold]
    m8 = quakes.loc[quakes["magnitude"] >= 8.0, "observed_at"].dt.floor("D")
    hits = m8[m8 >= hold]
    if win.empty:
        return {"name": "davidson_spf_m8_v2_budgeted", "error": "no window days"}
    aligned = flags.reindex(pd.DatetimeIndex(hits)).fillna(False)
    n = int(len(hits))
    in_win = int(aligned.sum())
    return {
        "name": "davidson_spf_m8_v2_budgeted",
        "holdout_start": holdout_start,
        "n_m8": n,
        "m8_in_window": in_win,
        "hit_rate": None if n == 0 else in_win / n,
        "time_in_alarm": float(win.mean()),
    }


def quiet_spf_test(feats: pd.DataFrame, quakes: pd.DataFrame, holdout_start: str, root: Path | None) -> dict:
    hold = utc_ts(holdout_start)
    etas, _mod = load_models(root)
    days = pd.DatetimeIndex(feats["issue_time"])
    hist = quakes.loc[quakes["magnitude"] >= etas.m0]
    versions = load_comcat_versions(root)
    lam = daily_background_and_cluster(hist, days, etas, versions=versions)
    quiet_cut = float(load_quiet(root).get("lambda_etas_max") or 0.0)
    alarm = (lam <= quiet_cut) & (feats["spf_significant_window"].to_numpy() > 0.5)
    flagged = pd.Series(alarm, index=days)
    win = flagged.loc[flagged.index >= hold]
    m8 = quakes.loc[quakes["magnitude"] >= 8.0, "observed_at"].dt.floor("D")
    hits = m8[m8 >= hold]
    if win.empty:
        return {"name": "quiet_spf_m8", "error": "no days"}
    aligned = flagged.reindex(pd.DatetimeIndex(hits)).fillna(False)
    n = int(len(hits))
    in_win = int(aligned.sum())
    return {
        "name": "quiet_spf_m8",
        "holdout_start": holdout_start,
        "n_m8": n,
        "m8_in_window": in_win,
        "hit_rate": None if n == 0 else in_win / n,
        "time_in_alarm": float(win.mean()),
        "quiet_lambda_etas_max": quiet_cut,
    }


def run_battery(holdout_start: str = "2019-01-01", root: Path | None = None) -> list[dict]:
    quakes, spf, feats = _load(root)
    reports = [
        davidson_test(quakes, spf, holdout_start, version="v1"),
        davidson_test(quakes, spf, holdout_start, version="v2"),
        davidson_test(quakes, _shift_spf(spf, 180), holdout_start, version="v2")
        | {"name": "davidson_spf_m8_v2_timeshift_180d"},
        issued_spf_test(feats, quakes, holdout_start),
        quiet_spf_test(feats, quakes, holdout_start, root),
        marchitelli_test(feats, quakes, holdout_start),
        marchitelli_test(feats, decluster(quakes), holdout_start)
        | {"name": "marchitelli_proton_lag1_m6_declustered"},
        love_thomas_test(feats, quakes, holdout_start),
        *run_fusion_nulls(feats, quakes, holdout_start=holdout_start, root=root),
    ]
    dest = feature_dir(root).parent / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_json(dest / "battery.json", orient="records", indent=2)
    return reports
