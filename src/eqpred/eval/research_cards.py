"""Research cards beside the frozen 30-day set. None of them write issued p."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.eval.decluster import fresh_hits
from eqpred.eval.honest import fit_val_split, purged_train_hold, rolling_climate_series
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.skill import information_gain
from eqpred.eval.where_research import (
    choose_locked_patch,
    interface_fresh_hits,
    known_event_rate,
)
from eqpred.faults import SEGMENTS, segment_region
from eqpred.paths import data_dir
from eqpred.records import iso
from eqpred.stamp import parse_utc

GAIN_MIN = 0.01


def expanding_two_rate(
    hits: np.ndarray,
    gate: np.ndarray,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Past hit rate while a gate was on, and while it was off. Today is not included."""
    y = np.asarray(hits, dtype=float)
    on = np.asarray(gate, dtype=bool)
    n = len(y)
    on_p = np.full(n, np.nan)
    off_p = np.full(n, np.nan)
    sum_on = sum_off = 0.0
    n_on = n_off = 0
    for t in range(n):
        if t >= horizon_days and np.isfinite(y[t - horizon_days]):
            if on[t - horizon_days]:
                sum_on += y[t - horizon_days]
                n_on += 1
            else:
                sum_off += y[t - horizon_days]
                n_off += 1
        if n_on:
            on_p[t] = sum_on / n_on
        if n_off:
            off_p[t] = sum_off / n_off
    return on_p, off_p


def run_research_cards(root: Path | None = None) -> dict[str, Any]:
    """Build the four research cards and write data/eval/research_cards.json."""
    from eqpred.catalog import load_comcat
    from eqpred.eval.eepas import load_eepas
    from eqpred.eval.mainshock import ISSUED_TARGETS
    from eqpred.eval.where_research import _patch_score, _station_coords
    from eqpred.paths import feature_dir

    feat = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat["issue_time"] = parse_utc(feat["issue_time"])
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog = load_comcat(root).copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    catalog["available_at"] = parse_utc(catalog["available_at"])
    catalog_end = catalog["observed_at"].max()
    closed = np.asarray(days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end)
    train_mask, hold_mask = purged_train_hold(days, closed, "2018-12-31", "2019-01-01", purge=HORIZON_DAYS)
    _fit, val = fit_val_split(days, train_mask, purge=HORIZON_DAYS)
    stations = _station_coords()
    mags = dict(ISSUED_TARGETS)

    where = []
    for parent in ("hikurangi", "alpine"):
        mag = mags[parent]
        for name, rec in SEGMENTS.items():
            if rec["parent"] != parent:
                continue
            region = segment_region(name)
            work = catalog.loc[catalog["magnitude"] >= mag]
            from eqpred.regions import in_bbox

            work = work.loc[in_bbox(work, region)]
            if work.empty:
                rate = np.full(len(days), 1e-6)
            else:
                from eqpred.eval.decluster import decluster_causal

                mains = decluster_causal(work)
                rate = known_event_rate(
                    parse_utc(mains["observed_at"]).to_numpy(dtype="datetime64[ns]"),
                    parse_utc(mains["available_at"]).to_numpy(dtype="datetime64[ns]"),
                    days,
                )
            hits = fresh_hits(catalog, region, mag, days, HORIZON_DAYS).astype(float)
            roll = rolling_climate_series(hits, days, horizon=HORIZON_DAYS)
            check = information_gain(rate[val], hits[val], roll[val])
            where.append(_chance_row(rec["label"], name, mag, rate, roll, days, check, "which third of the fault"))

    patch = _patch_score("alpine", mags["alpine"], feat, catalog, days, stations, val, hold_mask)
    location = {
        "kind": "alpine_location",
        "title": "Alpine Fault, which third",
        "writes_issued_p": False,
        "patch": _latest_patch_name(feat, catalog, days, stations),
        "later_share": patch.get("replay_located_share"),
        "equal_share": patch.get("equal_share"),
        "check_information_gain": patch.get("check_information_gain"),
        "sentence": (
            "Names the quiet, still-locked third of the Alpine Fault. "
            "Scored by how often the next first shock was actually there, against one in three."
        ),
    }

    eepas = load_eepas(root) or {}
    eepas_cards = []
    for row in eepas.get("week") or []:
        if row.get("region") not in {"hikurangi", "alpine", "kermadec", "global"}:
            continue
        eepas_cards.append(
            {
                "kind": "eepas",
                "title": f"{row.get('label')} · 90 days",
                "chance_plain": row.get("chance_plain"),
                "issue_day": row.get("issue_day"),
                "writes_issued_p": False,
                "sentence": "Medium-term catalog. Every earlier quake is a precursor scaled by its size.",
            }
        )

    sse = _sse_card(feat, catalog, days, val, hold_mask)
    report = {
        "scored_at": iso(datetime.now(timezone.utc)),
        "writes_issued_p": False,
        "where": where,
        "alpine_location": location,
        "eepas": eepas_cards,
        "eepas_note": eepas.get("note"),
        "slow_slip": sse,
        "note": _note(where, location, sse),
    }
    dest = data_dir(root) / "eval" / "research_cards.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _sse_card(feat, catalog, days, val, hold) -> dict[str, Any]:
    hits = interface_fresh_hits(catalog, days, mag=5.0).astype(float)
    if "sse_hikurangi" not in feat.columns:
        gate = np.zeros(len(days), dtype=bool)
    else:
        gate = pd.to_numeric(feat["sse_hikurangi"], errors="coerce").to_numpy(dtype=float) > 0
    on_p, off_p = expanding_two_rate(hits, gate)
    roll = rolling_climate_series(hits, days, horizon=HORIZON_DAYS)
    model = np.where(gate, on_p, off_p)
    check = information_gain(model[val], hits[val], roll[val])
    latest = len(days) - 1
    return {
        "kind": "slow_slip",
        "title": "Hikurangi interface, slow slip on or off",
        "writes_issued_p": False,
        "on_plain": _plain(on_p[latest]),
        "off_plain": _plain(off_p[latest]),
        "on_p": None if not np.isfinite(on_p[latest]) else float(on_p[latest]),
        "off_p": None if not np.isfinite(off_p[latest]) else float(off_p[latest]),
        "slip_on_today": bool(gate[latest]),
        "check_information_gain": check,
        "earns_frozen_card": check is not None and check >= GAIN_MIN,
        "issue_day": days[latest].date().isoformat(),
        "sentence": "Next 30 days on the plate interface. One rate while a slow slip is on, one while it is off.",
    }


def _latest_patch_name(feat, catalog, days, stations) -> str:
    from eqpred.eval.where_research import _segment_quakes, _segment_slip

    names = [name for name, rec in SEGMENTS.items() if rec["parent"] == "alpine"]
    regions = [segment_region(name) for name in names]
    slip = np.column_stack([_segment_slip(feat, region, stations) for region in regions])
    quakes = np.column_stack(
        [_segment_quakes(catalog, region, days, 5.0) for region in regions]
    )
    idx = choose_locked_patch(slip[-1], quakes[-1])
    return SEGMENTS[names[idx]]["label"]


def _chance_row(label, name, mag, rate, roll, days, check, question) -> dict[str, Any]:
    latest = len(days) - 1
    return {
        "kind": "where",
        "region": name,
        "title": label,
        "magnitude_min": mag,
        "question": question,
        "chance_plain": _plain(rate[latest]),
        "usual_plain": _plain(roll[latest]),
        "check_information_gain": check,
        "earns_frozen_card": check is not None and check >= GAIN_MIN,
        "writes_issued_p": False,
        "issue_day": days[latest].date().isoformat(),
    }


def _plain(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    shown = int(round(float(np.clip(value, 0.0, 1.0)) * 100))
    if shown <= 0 and value > 0:
        return "less than 1 in 100"
    return f"{shown} in 100"


def _note(where, location, sse) -> str:
    best = max(where, key=lambda row: row.get("check_information_gain") or -999)
    share = location.get("later_share")
    share_txt = "—" if share is None else f"{share * 100:.0f} in 100"
    return (
        "Research cards only. They do not replace the frozen 30-day numbers. "
        f"Best third on the check years is {best.get('title')} "
        f"with gain {best.get('check_information_gain')}. "
        f"Alpine location later share {share_txt} against 33 in 100. "
        f"Slow slip on {sse.get('on_plain')}, off {sse.get('off_plain')}."
    )
