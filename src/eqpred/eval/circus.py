from __future__ import annotations

"""Negative-control circus for the leftover extra.

Veto only. Weekday, payday, HAARP on/off, and Wikipedia earthquake pageviews
are foils. If leftover tracks a foil on later mornings, circus fails and the
leftover card is marked vetoed. Never writes issued p.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eqpred.catalog import load_comcat
from eqpred.eval.decluster import fresh_hits
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.honest import purged_train_hold
from eqpred.eval.leftover import (
    WRITES_ISSUED_P,
    ensure_schedule,
    fit_leftover_region,
    leftover_columns,
    load_leftover,
    schedule_columns,
)
from eqpred.eval.mainshock import ISSUED_TARGETS
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc

FOIL_RHO = 0.25
FOIL_P = 0.05
CIRCUS_ID = "fresh.circus.v1"


def calendar_foils(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Replayable weekday / payday / weekend flags. No download."""
    index = pd.DatetimeIndex(days)
    dom = index.day
    weekday = index.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "cal_weekday": weekday,
            "cal_weekend": (weekday >= 5).astype(float),
            "cal_payday": ((dom >= 14) & (dom <= 16)).astype(float),
        },
        index=index,
    )


def foil_frame(feat: pd.DataFrame) -> pd.DataFrame:
    work = feat.copy()
    days = pd.DatetimeIndex(parse_utc(work["issue_time"]))
    cal = calendar_foils(days)
    out = cal.reset_index(drop=True)
    if "haarp_on" in work.columns:
        out["haarp_on"] = pd.to_numeric(work["haarp_on"], errors="coerce").to_numpy()
    if "wiki_eq_views" in work.columns:
        out["wiki_eq_views"] = pd.to_numeric(work["wiki_eq_views"], errors="coerce").to_numpy()
    if "vlf_amp" in work.columns:
        out["vlf_amp"] = pd.to_numeric(work["vlf_amp"], errors="coerce").to_numpy()
    if "tir_ts" in work.columns:
        out["tir_ts"] = pd.to_numeric(work["tir_ts"], errors="coerce").to_numpy()
    return out


def spearman_foil(leftover_hat: np.ndarray, foil: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(leftover_hat) & np.isfinite(foil)
    if int(ok.sum()) < 40:
        return {"rho": None, "pvalue": None, "n": int(ok.sum()), "fail": False}
    rho, pvalue = spearmanr(leftover_hat[ok], foil[ok])
    rho_f = None if not np.isfinite(rho) else float(rho)
    p_f = None if not np.isfinite(pvalue) else float(pvalue)
    fail = rho_f is not None and p_f is not None and abs(rho_f) >= FOIL_RHO and p_f < FOIL_P
    return {"rho": rho_f, "pvalue": p_f, "n": int(ok.sum()), "fail": fail}


def circus_checks(leftover_hat: np.ndarray, foils: pd.DataFrame, hold_mask: np.ndarray) -> list[dict[str, Any]]:
    hold = np.asarray(hold_mask, dtype=bool)
    rows: list[dict[str, Any]] = []
    for name in foils.columns:
        check = spearman_foil(np.asarray(leftover_hat, dtype=float)[hold], foils[name].to_numpy(dtype=float)[hold])
        rows.append({"foil": name, **check})
    return rows


def load_circus(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "circus.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def attach_circus_sidecar(report: dict[str, Any], root: Path | None = None) -> bool:
    path = data_dir(root) / "eval" / "mainshock.json"
    if not path.exists():
        return False
    pack = json.loads(path.read_text(encoding="utf-8"))
    pack["circus"] = {
        "id": CIRCUS_ID,
        "mode": "circus",
        "writes_issued_p": False,
        "pass": report.get("pass"),
        "n_fail": report.get("n_fail"),
        "note": report.get("note"),
        "foils": report.get("foils"),
        "regions": report.get("regions"),
        "scanned_at": report.get("scanned_at"),
    }
    path.write_text(json.dumps(pack, indent=2, default=str), encoding="utf-8")
    return True


def run_circus(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
    leftover: dict[str, Any] | None = None,
    leftover_hats: dict[str, np.ndarray] | None = None,
    attach: bool = True,
) -> dict[str, Any]:
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    if features is None:
        path = feature_dir(root) / "global_daily.parquet"
        feat = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    else:
        feat = features.copy()
    empty = {
        "writes_issued_p": WRITES_ISSUED_P,
        "id": CIRCUS_ID,
        "mode": "circus",
        "pass": True,
        "n_fail": 0,
        "foils": [],
        "regions": [],
        "note": "Circus extra has nothing to check yet. It never writes the published number.",
        "scanned_at": iso(datetime.now(timezone.utc)),
        "attached_to_mainshock": False,
    }
    if feat.empty:
        (dest / "circus.json").write_text(json.dumps(empty, indent=2, default=str), encoding="utf-8")
        return empty
    feat = ensure_schedule(feat)
    feat["issue_time"] = parse_utc(feat["issue_time"])
    leftover = leftover if leftover is not None else load_leftover(root)
    hats = leftover_hats or {}
    days = pd.DatetimeIndex(feat["issue_time"])
    have_hats = all(rid in hats for rid, _mag in ISSUED_TARGETS)
    catalog = None
    if not have_hats:
        catalog = quakes.copy() if quakes is not None else load_comcat(root)
        catalog["observed_at"] = parse_utc(catalog["observed_at"])
        catalog_end = catalog["observed_at"].max()
    else:
        catalog_end = days.max() + pd.Timedelta(days=HORIZON_DAYS)
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    foils = foil_frame(feat)
    left_cols = leftover_columns(feat)
    sched_cols = schedule_columns(feat)
    region_rows: list[dict[str, Any]] = []
    all_fails: list[dict[str, Any]] = []
    for region_id, mag in ISSUED_TARGETS:
        hat = hats.get(region_id)
        if hat is None:
            hits = fresh_hits(catalog, REGIONS[region_id], mag, days, HORIZON_DAYS).astype(float)
            packed = fit_leftover_region(
                feat,
                hits,
                region_id=region_id,
                mag=mag,
                train_mask=train_mask,
                hold_mask=hold_mask,
                leftover_cols=left_cols,
                schedule_cols=sched_cols,
            )
            hat = packed["leftover_hat"]
        checks = circus_checks(np.asarray(hat, dtype=float), foils, hold_mask)
        failed = [c for c in checks if c.get("fail")]
        all_fails.extend({**c, "region": region_id} for c in failed)
        region_rows.append(
            {
                "region": region_id,
                "label": REGIONS[region_id]["label"],
                "magnitude_min": mag,
                "pass": not failed,
                "n_fail": len(failed),
                "checks": checks,
                "writes_issued_p": False,
            }
        )
    passed = not all_fails
    leftover_week = (leftover or {}).get("week") or []
    if leftover_week and not passed:
        leftover_week = [{**row, "circus_veto": True, "sentence": (row.get("sentence") or "") + " Circus vetoed this leftover card."} for row in leftover_week]
        leftover = dict(leftover or {})
        leftover.pop("_hats", None)
        leftover["week"] = leftover_week
        leftover["circus_veto"] = True
        leftover_path = dest / "leftover.json"
        if leftover_path.exists():
            leftover_path.write_text(json.dumps(leftover, indent=2, default=str), encoding="utf-8")
        main = dest / "mainshock.json"
        if main.exists():
            pack = json.loads(main.read_text(encoding="utf-8"))
            if pack.get("leftover"):
                pack["leftover"]["week"] = leftover_week
                pack["leftover"]["circus_veto"] = True
                main.write_text(json.dumps(pack, indent=2, default=str), encoding="utf-8")
    foil_names = list(foils.columns)
    note = (
        f"Circus extra checked leftover against {', '.join(foil_names)}. "
        + (
            f"Failed {len(all_fails)} foil test(s); leftover cards are vetoed this week."
            if all_fails
            else "No foil tracked leftover on later mornings."
        )
        + " It never writes the published number."
    )
    report = {
        "writes_issued_p": False,
        "id": CIRCUS_ID,
        "mode": "circus",
        "pass": passed,
        "n_fail": len(all_fails),
        "foils": foil_names,
        "fails": all_fails,
        "regions": region_rows,
        "note": note,
        "scanned_at": iso(datetime.now(timezone.utc)),
        "rho_fail": FOIL_RHO,
    }
    (dest / "circus.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["attached_to_mainshock"] = attach_circus_sidecar(report, root) if attach else False
    return report
