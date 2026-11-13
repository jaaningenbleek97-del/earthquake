"""Bz-south storm-state sweep — research bench only. Never writes issued p.

Second pass over the southward-IMF hypothesis, driven by the physics rather
than raw Bz level. Southward Bz is the *switch* (Dungey reconnection); the
energy actually deposited into the magnetosphere shows up as the geomagnetic
response (Dst sudden commencement, substorm/AE onset). This sweep tests:

  1. RATE OF CHANGE    — dBz/dt and the size of the southward swing.
  2. STORM-STATE GATE  — Bz south AND a concurrent/lagged Dst drop or storm.
  3. SUDDEN COMMENCEMENT — the Dst day-over-day drop itself, as a driver.
  4. SUBSTORM ONSET    — AE index jump (auroral electrojet).
  5. COUPLING GATING   — Newell / epsilon coupling high WHILE Bz is south.
  6. LEAD-LAG CHAIN    — Bz south at t, storm at t+1..2, quake at t+lag.
                         (retrospective-only: uses the following day's Dst,
                          so it can never be a live/issuable driver.)

All scored against first-shock (fresh) hits with the same discover/confirm/
replay split and Spearman machinery as the hunt. Lead-lag variants are marked
`retrospective_only` and are never eligible to issue. Nothing writes issued p.
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

TARGETS = (
    ("global", 7.0),
    ("hikurangi", 5.0),
    ("alpine", 5.0),
    ("kermadec", 6.0),
)

LAGS = (0, 1, 2, 3, 7, 14, 27)
HOLDOUT_P = 0.05
FDR_Q = 0.10


def _build_variants(feat: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Return (variants, retrospective_only_cols)."""
    bz = pd.to_numeric(feat["imf_bz_nt"], errors="coerce")
    dst = pd.to_numeric(feat["dst_nt"], errors="coerce")
    ae = pd.to_numeric(feat["ae_nt"], errors="coerce")
    kp = pd.to_numeric(feat["kp"], errors="coerce")
    wind = pd.to_numeric(feat["solar_wind_speed_kms"], errors="coerce")
    newell = pd.to_numeric(feat["solar_newell"], errors="coerce")
    epsilon = pd.to_numeric(feat["solar_epsilon"], errors="coerce")

    out = pd.DataFrame(index=feat.index)
    retro: set[str] = set()

    south = np.clip(-bz.to_numpy(), 0.0, None)  # magnitude of southward Bz

    # 1. Rate of change / southward swing size
    out["bz_drop_1d"] = (bz.shift(1) - bz).clip(lower=0.0).to_numpy()       # + when flipping south
    out["bz_drop_3d"] = (bz.shift(3) - bz).clip(lower=0.0).to_numpy()
    out["bz_dbdt_1d"] = (bz - bz.shift(1)).to_numpy()                        # signed rate

    # 2. Storm-state gating: Bz south AND Dst drop (same day / next day)
    dst_drop_1d = (dst.shift(1) - dst).clip(lower=0.0).to_numpy()           # sudden commencement size
    dst_storm = (dst <= -50.0).astype(float).to_numpy()
    bz_south_flag = (bz < -5.0).astype(float).to_numpy()

    out["dst_drop_1d"] = dst_drop_1d
    out["dst_storm_flag"] = dst_storm
    out["bz_south_x_dstdrop"] = south * dst_drop_1d                        # concurrent
    out["bz_south_and_dstdrop"] = (bz_south_flag * (dst_drop_1d >= 20.0)).astype(float)
    out["bz_south_and_storm"] = (bz_south_flag * dst_storm).astype(float)

    # 3. Sudden commencement as a driver (geomagnetic response itself)
    out["ssc_1d"] = (dst_drop_1d >= 30.0).astype(float)
    out["ssc_2d"] = (dst.shift(2).to_numpy() - dst.to_numpy() >= 30.0).astype(float)

    # 4. Substorm onset: AE jump (auroral electrojet)
    out["ae_jump_1d"] = (ae - ae.shift(1)).clip(lower=0.0).to_numpy()
    out["ae_substorm"] = (ae >= 400.0).astype(float).to_numpy()
    out["ae_jump_x_bz_south"] = out["ae_jump_1d"].to_numpy() * south

    # 5. Coupling gating: Newell / epsilon high WHILE Bz south
    out["newell_x_south"] = newell.to_numpy() * south
    out["epsilon_x_south"] = epsilon.to_numpy() * south
    out["newell_x_south_7d"] = pd.Series(newell.to_numpy() * south, index=feat.index).rolling(7, min_periods=3).mean().to_numpy()

    # 6. Lead-lag chain: Bz south at t, Dst drops at t+1 / t+2  (retrospective-only)
    dst_drop_next = np.roll(dst_drop_1d, -1)
    dst_drop_next[0] = np.nan
    dst_drop_next2 = np.roll(dst_drop_1d, -2)
    dst_drop_next2[:2] = np.nan
    out["bz_south_leads_ssc1"] = south * (dst_drop_next >= 20.0).astype(float)
    out["bz_south_leads_ssc2"] = south * (dst_drop_next2 >= 20.0).astype(float)
    retro.update({"bz_south_leads_ssc1", "bz_south_leads_ssc2"})

    # Integrated southward flux over storm windows (storm energy proxy)
    for w in (1, 3, 7):
        out[f"south_flux_{w}d"] = pd.Series(south, index=feat.index).rolling(w, min_periods=1).sum().to_numpy()
        out[f"bz_south_kp_{w}d"] = pd.Series(south * np.clip(kp.to_numpy(), 0, None), index=feat.index).rolling(w, min_periods=1).mean().to_numpy()

    return out, retro


def _mark(trials: list[dict]) -> None:
    from eqpred.eval.hunt import benjamini_hochberg
    pvals = np.array([t["train_p"] if t["train_p"] is not None else 1.0 for t in trials], dtype=float)
    qvals = benjamini_hochberg(pvals)
    for t, q in zip(trials, qvals, strict=True):
        t["train_q"] = float(q) if t["train_p"] is not None else None
        t["discovered"] = bool(t["train_q"] is not None and t["train_q"] < FDR_Q)
        t["confirmed"] = bool(
            t["discovered"]
            and t["train_rho"] is not None and t["confirm_rho"] is not None
            and t["train_rho"] * t["confirm_rho"] > 0
            and t["confirm_p"] is not None and t["confirm_p"] < HOLDOUT_P
        )
        t["replayed"] = bool(
            t["confirmed"]
            and t["replay_rho"] is not None and t["train_rho"] is not None
            and t["train_rho"] * t["replay_rho"] > 0
            and t["replay_p"] is not None and t["replay_p"] < HOLDOUT_P
        )


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

    variants, retro_cols = _build_variants(feat)

    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    discover_raw, confirm_raw, replay_raw, confirm_start = era_masks(days, closed, TRAIN_END, HOLDOUT_START)
    discover_mask = week_mask(discover_raw)
    confirm_mask = week_mask(confirm_raw)
    replay_mask = week_mask(replay_raw)

    trials: list[dict] = []
    for region_id, mag in TARGETS:
        region = REGIONS[region_id]
        hits = fresh_hits(catalog, region, mag, days, HORIZON_DAYS).astype(float)
        for vcol in variants.columns:
            raw = variants[vcol].to_numpy(dtype=float)
            for lag in LAGS:
                if lag:
                    driver = np.roll(raw, lag).astype(float)
                    driver[:lag] = np.nan
                else:
                    driver = raw.astype(float)
                tr, tp, tn = _spearman(driver[discover_mask], hits[discover_mask])
                cr, cp, cn = _spearman(driver[confirm_mask], hits[confirm_mask])
                rr, rp, rn = _spearman(driver[replay_mask], hits[replay_mask])
                trials.append({
                    "variant": vcol,
                    "region": region_id,
                    "magnitude_min": float(mag),
                    "lag_days": int(lag),
                    "train_n": tn, "train_rho": tr, "train_p": tp,
                    "confirm_n": cn, "confirm_rho": cr, "confirm_p": cp,
                    "replay_n": rn, "replay_rho": rr, "replay_p": rp,
                    "retrospective_only": vcol in retro_cols,
                })

    _mark(trials)
    confirmed = [t for t in trials if t["confirmed"]]
    replayed = [t for t in trials if t["replayed"]]

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
        "retrospective_only_variants": sorted(retro_cols),
        "top_confirm": sorted(confirmed, key=lambda t: abs(t["confirm_rho"] or 0), reverse=True)[:30],
        "top_replay": sorted(replayed, key=lambda t: abs(t["replay_rho"] or 0), reverse=True)[:30],
        "note": (
            "Storm-state sweep of southward-Bz onset: rate of change, Dst sudden "
            "commencement, substorm/AE onset, coupling-function gating, and lead-lag "
            "chain vs first-shock hits. Lead-lag variants use the following day's Dst "
            "and are retrospective-only (never issuable). Nothing writes issued p."
        ),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "writes_issued_p": False,
    }

    dest = data_dir(root) / "eval" / "research"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "bz_south_storm_sweep.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(trials).to_csv(dest / "bz_south_storm_sweep.csv", index=False)

    print(f"variants={len(variants.columns)} trials={len(trials)} "
          f"discovered={summary['n_discovered']} confirmed={summary['n_confirmed']} "
          f"replayed={summary['n_replayed']}")
    if confirmed:
        print("\nConfirmed (same sign late train, p<0.05):")
        for t in sorted(confirmed, key=lambda x: abs(x["confirm_rho"] or 0), reverse=True)[:40]:
            tag = " [RETRO]" if t["retrospective_only"] else ""
            print(f"  {t['variant']:<24} {t['region']:<10} M{t['magnitude_min']:g} "
                  f"lag={t['lag_days']:>2} train={t['train_rho']:+.4f} "
                  f"confirm={t['confirm_rho']:+.4f} replay={t['replay_rho']}{tag}")
    else:
        print("\nNo variant passed the confirm gate.")


if __name__ == "__main__":
    main()
