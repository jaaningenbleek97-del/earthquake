"""Bz-south family sweep — research bench only. Never writes issued p.

Scores a battery of southward-IMF derived variants against first-shock
(fresh) hits on closed windows, using the same honest discover/confirm/replay
splits and Spearman-residual scoring as the hunt. This is exploration: no
frozen recipe, model, card, or prospective ledger is touched, and nothing
here is promoted to an issued head. Later (2019+) years are a replay only.

Variant families (all built from already-ingested OMNI columns on the
daily feature table: imf_bz_nt, imf_by_nt, imf_b_nt, solar_wind_speed_kms,
flow_pressure_npa, proton_density_cm3, kp, dst_nt, electric_field_mvm):

  1. south_min_w:     minimum Bz over a trailing window (strength of the dip)
  2. south_mean_w:    mean Bz over the window (bias toward south)
  3. south_sum_neg_w: -sum(Bz where Bz<0) — the southward integral
  4. south_days_w:    count of days Bz < threshold (duration)
  5. south_frac_w:    fraction of days Bz < threshold
  6. south_streak_w:  longest consecutive run Bz < threshold
  7. south_peak_w:    most negative single-day Bz
  8. south_onset_w:   days since last Bz <-threshold crossing
  9. south_by_w:      cross product |Bz-| * By / |B|  (magnitude of south+dawn/dusk)
 10. south_duration_since: days since Bz first turned south in the window
 11. south_gate*:     strength * kp, strength * wind (strong solar conditions
                      while Bz is south, as requested), south * dst

Every variant is scanned at several lags. Output is a JSON report plus a
CSV, both written under data/eval/research/.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eqpred.catalog import load_comcat, load_comcat_versions
from eqpred.eval.decluster import fresh_hits
from eqpred.eval.honest import era_masks, week_mask
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.hunt import _spearman
from eqpred.paths import data_dir, feature_dir
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc

TRAIN_END = "2018-12-31"
HOLDOUT_START = "2019-01-01"

# (region, magnitude floor) of the short-horizon research bench.
TARGETS = (
    ("global", 7.0),
    ("hikurangi", 5.0),
    ("alpine", 5.0),
    ("kermadec", 6.0),
)

# Bz-below thresholds (nT) for duration/streak/count variants.
THRESHOLDS = (-2.0, -4.0, -5.0, -8.0, -10.0)
# Trailing windows (days) for rolling aggregates.
WINDOWS = (3, 7, 14, 27)
# Lags (days) applied to the driver before scoring.
LAGS = (0, 1, 2, 3, 7, 14, 27)

HOLDOUT_P = 0.05
FDR_Q = 0.10
MIN_PAIR = 80


def _build_variants(feat: pd.DataFrame) -> pd.DataFrame:
    bz = pd.to_numeric(feat["imf_bz_nt"], errors="coerce")
    by = pd.to_numeric(feat["imf_by_nt"], errors="coerce") if "imf_by_nt" in feat.columns else bz * np.nan
    btot = pd.to_numeric(feat["imf_b_nt"], errors="coerce").abs() if "imf_b_nt" in feat.columns else np.nan
    wind = pd.to_numeric(feat["solar_wind_speed_kms"], errors="coerce")
    kp = pd.to_numeric(feat["kp"], errors="coerce")
    dst = pd.to_numeric(feat["dst_nt"], errors="coerce")

    neg = np.where(bz.to_numpy() < 0, bz.to_numpy(), 0.0)
    out = pd.DataFrame(index=feat.index)

    for w in WINDOWS:
        w_min = bz.rolling(w, min_periods=max(2, w // 2)).min()
        out[f"south_min_{w}d"] = w_min.to_numpy()
        out[f"south_mean_{w}d"] = bz.rolling(w, min_periods=max(2, w // 2)).mean().to_numpy()
        out[f"south_sum_neg_{w}d"] = pd.Series(neg, index=feat.index).rolling(w, min_periods=max(2, w // 2)).sum().to_numpy()

    for thr in THRESHOLDS:
        for w in WINDOWS:
            below = (bz < thr)
            out[f"south_days{int(abs(thr))}_{w}d"] = below.rolling(w, min_periods=max(2, w // 2)).sum().to_numpy()
            out[f"south_frac{int(abs(thr))}_{w}d"] = below.rolling(w, min_periods=max(2, w // 2)).mean().to_numpy()
        # streak and onset are threshold-specific but window-free
        below = (bz < thr).astype(int).to_numpy()
        # streak: longest consecutive run within the last 14d
        def _streak_run(v: np.ndarray, n: int) -> np.ndarray:
            res = np.full(len(v), np.nan)
            arr = np.asarray(v)
            for i in range(len(arr)):
                lo = max(0, i - n + 1)
                seg = arr[lo : i + 1]
                best = cur = 0
                for x in seg:
                    cur = cur + 1 if x else 0
                    best = max(best, cur)
                res[i] = best
            return res
        out[f"south_streak{int(abs(thr))}_14d"] = _streak_run(below, 14)
        # onset: days since last crossing below threshold (capped at 90)
        onset = np.full(len(bz), np.nan)
        last_cross = -np.inf
        dates_idx = np.arange(len(bz))
        arr_below = (bz < thr).to_numpy()
        for i in range(len(bz)):
            if arr_below[i]:
                last_cross = i
            if np.isfinite(last_cross):
                onset[i] = min(90, i - last_cross)
        out[f"south_onset{int(abs(thr))}"] = onset

    # south_peak over 14d: most negative single-day Bz
    out["south_peak_14d"] = bz.rolling(14, min_periods=3).min().to_numpy()

    # southward strength with dawn/dusk (By) sign, over 3d
    south_mag = np.clip(-bz.to_numpy(), 0.0, None)
    out["south_by_3d"] = pd.Series(south_mag * by.to_numpy(), index=feat.index).rolling(3, min_periods=1).mean().to_numpy()

    # southward-flux coupling proxies (strong solar conditions while south)
    south = np.clip(-bz.to_numpy(), 0.0, None)
    out["south_gate_kp_1d"] = south * np.clip(kp.to_numpy(), 0.0, None)
    out["south_gate_wind_1d"] = south * np.clip(wind.to_numpy(), 0.0, None)
    out["south_gate_dst_1d"] = south * np.clip(-dst.to_numpy(), 0.0, None)
    # southward coupling over 7d, gated by storm strength
    out["south_flux_x_wind_7d"] = pd.Series(south * wind.to_numpy(), index=feat.index).rolling(7, min_periods=3).sum().to_numpy()

    return out


def main() -> None:
    root = Path.cwd()
    feat = pd.read_parquet(feature_dir(root) / "global_daily.parquet").copy()
    feat["issue_time"] = parse_utc(feat["issue_time"])

    catalog = load_comcat(root)
    versions = load_comcat_versions(root)
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    catalog["available_at"] = (
        parse_utc(catalog["available_at"]) if "available_at" in catalog.columns else catalog["observed_at"]
    )

    variants = _build_variants(feat)

    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    discover_raw, confirm_raw, replay_raw, confirm_start = era_masks(
        days, closed, TRAIN_END, HOLDOUT_START
    )
    discover_mask = week_mask(discover_raw)
    confirm_mask = week_mask(confirm_raw)
    replay_mask = week_mask(replay_raw)

    trials: list[dict] = []

    for region_id, mag in TARGETS:
        region = REGIONS[region_id]
        hits = fresh_hits(catalog, region, mag, days, HORIZON_DAYS).astype(float)
        for vcol in variants.columns:
            raw = np.nan_to_num(variants[vcol].to_numpy(dtype=float), nan=np.nan)
            for lag in LAGS:
                if lag:
                    driver = np.roll(raw, lag).astype(float)
                    driver[:lag] = np.nan
                else:
                    driver = raw.astype(float)
                train_rho, train_p, train_n = _spearman(driver[discover_mask], hits[discover_mask])
                confirm_rho, confirm_p, confirm_n = _spearman(driver[confirm_mask], hits[confirm_mask])
                replay_rho, replay_p, replay_n = _spearman(driver[replay_mask], hits[replay_mask])
                trials.append({
                    "variant": vcol,
                    "region": region_id,
                    "magnitude_min": float(mag),
                    "lag_days": int(lag),
                    "train_n": train_n,
                    "train_rho": train_rho,
                    "train_p": train_p,
                    "confirm_n": confirm_n,
                    "confirm_rho": confirm_rho,
                    "confirm_p": confirm_p,
                    "replay_n": replay_n,
                    "replay_rho": replay_rho,
                    "replay_p": replay_p,
                })

    # Mark: discovered (BH q<0.10), confirmed (same sign + confirm p<0.05), replayed.
    pvals = np.array([t["train_p"] if t["train_p"] is not None else 1.0 for t in trials], dtype=float)
    n = int(pvals.size)
    order = np.argsort(pvals)
    ranked = np.clip(pvals[order], 0.0, 1.0)
    running = 1.0
    qvals = np.ones(n)
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        qvals[order[i]] = float(min(running, 1.0))
    for t, q in zip(trials, qvals, strict=True):
        t["train_q"] = float(q) if t["train_p"] is not None else None
        t["discovered"] = bool(t["train_q"] is not None and t["train_q"] < FDR_Q)
        t["confirmed"] = bool(
            t["discovered"]
            and t["train_rho"] is not None
            and t["confirm_rho"] is not None
            and t["train_rho"] * t["confirm_rho"] > 0
            and t["confirm_p"] is not None
            and t["confirm_p"] < HOLDOUT_P
        )
        t["replayed"] = bool(
            t["confirmed"]
            and t["replay_rho"] is not None
            and t["train_rho"] is not None
            and t["train_rho"] * t["replay_rho"] > 0
            and t["replay_p"] is not None
            and t["replay_p"] < HOLDOUT_P
        )

    confirmed = [t for t in trials if t["confirmed"]]
    replayed = [t for t in trials if t["replayed"]]
    # Best replay and best confirm, for the summary.
    summary = {
        "train_end": TRAIN_END,
        "holdout_start": HOLDOUT_START,
        "confirm_start": confirm_start,
        "horizon_days": HORIZON_DAYS,
        "n_variants": len(variants.columns),
        "n_trials": len(trials),
        "n_discovered": int(sum(1 for t in trials if t["discovered"])),
        "n_confirmed": len(confirmed),
        "n_replayed": len(replayed),
        "top_confirm": sorted(confirmed, key=lambda t: abs(t["confirm_rho"] or 0), reverse=True)[:30],
        "top_replay": sorted(replayed, key=lambda t: abs(t["replay_rho"] or 0), reverse=True)[:30],
        "note": (
            "Research sweep of Bz-south variants vs first-shock (fresh) hits. "
            "Discover=early train (q<0.10), confirm=same sign on late train p<0.05, "
            "replay=2019+ replay same sign p<0.05. Nothing writes issued p; "
            "replayed years are not a pristine holdout."
        ),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "writes_issued_p": False,
    }

    dest = data_dir(root) / "eval" / "research"
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "bz_south_sweep.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    df = pd.DataFrame(trials)
    csv_path = dest / "bz_south_sweep.csv"
    df.to_csv(csv_path, index=False)

    print(f"variants={len(variants.columns)} trials={len(trials)} "
          f"discovered={summary['n_discovered']} confirmed={summary['n_confirmed']} "
          f"replayed={summary['n_replayed']}")
    print(f"report: {report_path}")
    print(f"csv:    {csv_path}")
    if confirmed:
        print("\nConfirmed (same sign on late train, p<0.05):")
        for t in sorted(confirmed, key=lambda x: abs(x["confirm_rho"] or 0), reverse=True)[:30]:
            print(f"  {t['variant']:<24} {t['region']:<10} M{t['magnitude_min']:g} "
                  f"lag={t['lag_days']:>2}  train_rho={t['train_rho']:+.4f} "
                  f"confirm_rho={t['confirm_rho']:+.4f} replay_rho={t['replay_rho']}")
    else:
        print("\nNo variant passed the confirm gate.")


if __name__ == "__main__":
    main()
