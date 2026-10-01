from __future__ import annotations

"""Published card: a family must earn its place, then the recipe stays frozen.

The working draft can still use every family. It replaces the frozen card only
after it beats the five-year rate on the check years and on mornings that
closed after that draft was locked.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from eqpred.eval.skill import MIN_N, information_gain
from eqpred.paths import data_dir
from eqpred.records import iso

MIN_FAMILY_IG = 0.01


def family_earns_card(information_gain_value: float | None) -> bool:
    """A real check-year gain. A hair above zero, or a negative gain, stays off."""
    if information_gain_value is None:
        return False
    return float(information_gain_value) >= MIN_FAMILY_IG


def promotion_status(
    check_igs: dict[str, float | None],
    post_lock_igs: dict[str, float | None],
    n_post_lock: int,
) -> dict[str, Any]:
    """Both tests are required. Closed mornings after the lock are the second one."""
    if int(n_post_lock) < MIN_N:
        return {
            "ready": False,
            "reason": "Not enough closed mornings since the draft was locked.",
        }
    if not check_igs:
        return {"ready": False, "reason": "The draft has no check-year score."}
    for region_id, ig in check_igs.items():
        post = post_lock_igs.get(region_id)
        if not family_earns_card(ig) or not family_earns_card(post):
            return {
                "ready": False,
                "reason": (
                    f"{region_id} has not beaten the five-year rate on both the check years "
                    "and the mornings since the draft was locked."
                ),
            }
    return {
        "ready": True,
        "reason": "The draft beat the five-year rate on the check years and on mornings after it was locked.",
    }


def frozen_dir(root: Path | None = None) -> Path:
    path = data_dir(root) / "eval" / "frozen"
    path.mkdir(parents=True, exist_ok=True)
    return path


def frozen_path(root: Path | None = None) -> Path:
    return frozen_dir(root) / "recipe.json"


def read_frozen(root: Path | None = None) -> dict[str, Any] | None:
    path = frozen_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def hold_published_card(report: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """A later refit updates the draft. The frozen card stays the published one."""
    frozen = read_frozen(root)
    if not frozen:
        return report
    incoming = [dict(row) for row in report.get("week") or []]
    report["draft"] = {
        "writes_issued_p": False,
        "locked_at": frozen.get("draft_locked_at") or frozen.get("locked_at"),
        "note": "Working draft. Every family can move this number. It does not replace the frozen card.",
        "week": incoming,
        "promote": frozen.get("promote") or {"ready": False, "reason": "The frozen card is still the exam."},
    }
    report["week"] = [dict(row) for row in frozen.get("week") or []]
    report["frozen"] = {
        "locked_at": frozen.get("locked_at"),
        "family_ig_min": frozen.get("family_ig_min"),
        "writes_issued_p": True,
        "promote": frozen.get("promote"),
    }
    boxes = frozen.get("boxes") or {}
    for region in report.get("regions") or []:
        box = boxes.get(str(region.get("region") or ""))
        if not box:
            continue
        region["recipe"] = box.get("recipe")
        region["information_gain"] = box.get("information_gain")
        region["brier"] = box.get("brier")
        region["climatology_brier"] = box.get("climatology_brier")
        region["issued_from"] = "gated"
        region["why"] = list(box.get("families") or [])
        tw = dict(region.get("this_week") or {})
        tw.update(box.get("this_week") or {})
        region["this_week"] = tw
    return report


def _jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _score_family(
    raw_x: np.ndarray,
    columns: list[int],
    hits: np.ndarray,
    days: pd.DatetimeIndex,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    horizon: int,
) -> float | None:
    from eqpred.eval.mainshock import (
        _lift_onto_rate,
        _logit,
        _proba,
        _score_to_rate,
        rolling_climate_series,
    )

    if len(columns) < 1 or int(fit_mask.sum()) < 80 or int(val_mask.sum()) < MIN_N:
        return None
    y = np.asarray(hits, dtype=float)
    if int(np.unique(y[fit_mask].astype(int)).size) < 2:
        return None
    model = _logit(C=0.2)
    try:
        model.fit(raw_x[fit_mask][:, columns], y[fit_mask].astype(int))
        score = np.asarray(_proba(model, raw_x[:, columns]), dtype=float).reshape(-1)
    except ValueError:
        return None
    roll = rolling_climate_series(y, days, horizon=horizon)
    climate = float(np.mean(y[fit_mask]))
    similar = _score_to_rate(score, y, fit_mask, max_percentile=100)
    issued = _lift_onto_rate(similar, y, fit_mask, roll, climate)
    return information_gain(issued[val_mask], y[val_mask], roll[val_mask])


def _fit_box(
    raw_x: np.ndarray,
    names: list[str],
    columns: list[int],
    hits: np.ndarray,
    days: pd.DatetimeIndex,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    train_mask: np.ndarray,
    horizon: int,
) -> dict[str, Any] | None:
    from eqpred.eval.mainshock import (
        _drop_copy_columns,
        _lift_onto_rate,
        _logit,
        _proba,
        _score_to_rate,
        recent_climate,
        rolling_climate_series,
    )

    y = np.asarray(hits, dtype=float)
    roll = rolling_climate_series(y, days, horizon=horizon)
    if len(columns) < 1:
        return None
    block = raw_x[:, columns]
    col_names = [names[i] for i in columns]
    block, col_names = _drop_copy_columns(block, col_names, fit_mask)
    if block.shape[1] < 1 or int(np.unique(y[fit_mask].astype(int)).size) < 2:
        return None
    model = _logit(C=0.2)
    try:
        model.fit(block[fit_mask], y[fit_mask].astype(int))
        score = np.asarray(_proba(model, block), dtype=float).reshape(-1)
    except ValueError:
        return None
    ref = val_mask if int(val_mask.sum()) >= 80 else fit_mask
    climate = recent_climate(y, days, train_mask)
    similar = _score_to_rate(score, y, ref, max_percentile=100)
    issued = _lift_onto_rate(similar, y, ref, roll, climate)
    return {
        "p": np.clip(np.asarray(issued, dtype=float), 1e-6, 1.0 - 1e-6),
        "roll": roll,
        "pipeline": model,
        "columns": col_names,
        "ref_scores": np.asarray(score[ref], dtype=float),
        "ref_hits": np.asarray(y[ref], dtype=float),
        "ref_base": float(np.mean(y[ref])) if int(ref.sum()) else float(climate),
    }


def install_frozen_and_draft(
    *,
    feat: pd.DataFrame,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    horizon: int,
    root: Path | None = None,
) -> dict[str, Any]:
    """Fit the gated published card and the ungated draft. Archive the previous card."""
    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.honest import fit_val_split
    from eqpred.eval.mainshock import (
        ISSUED_TARGETS,
        SHARED_FAMILIES,
        SHRINK_PER_FAMILY,
        _chance,
        _family_pool,
        _fit_shared_stack,
        _hunt_drivers,
        _place,
        _pool_top,
        _usable_indices,
        _without_hunt,
        recent_climate,
        rolling_climate_series,
    )
    from eqpred.eval.skill import skill_beats_climate
    from eqpred.regions import REGIONS

    raw_x, names = design
    fit0, val0 = fit_val_split(days, train_mask, purge=horizon)
    hunt_cols = _hunt_drivers("global", root)
    usable = _without_hunt(names, _usable_indices(raw_x, fit0), hunt_cols)
    print("fresh-shock draft stack (all families)", flush=True)
    draft_map, draft_recipes, draft_families, _conns = _fit_shared_stack(
        feat=feat,
        design=design,
        catalog=catalog,
        days=days,
        train_mask=train_mask,
        hold_mask=hold_mask,
        horizon=horizon,
        root=root,
    )
    locked_at = iso(datetime.now(timezone.utc))
    boxes: dict[str, Any] = {}
    models: dict[str, Any] = {}
    week: list[dict[str, Any]] = []
    draft_week: list[dict[str, Any]] = []
    check_igs: dict[str, float | None] = {}
    latest = len(days) - 1
    for region_id, mag in ISSUED_TARGETS:
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(float)
        family_ig: dict[str, float | None] = {}
        passing: list[str] = []
        chosen: list[int] = []
        for family in SHARED_FAMILIES:
            pool = _family_pool(names, family, usable)
            sl = raw_x[fit0] if fit0.any() else raw_x
            live = []
            for idx in pool:
                col = sl[:, idx]
                finite = col[np.isfinite(col)]
                if finite.size >= 20 and float(np.std(finite)) > 1e-12:
                    live.append(idx)
            top = _pool_top(raw_x, hits, fit0, live, SHRINK_PER_FAMILY)
            if not top:
                continue
            ig = _score_family(raw_x, top, hits, days, fit0, val0, horizon)
            family_ig[family] = _jsonable(ig)
            if family_earns_card(ig):
                passing.append(family)
                chosen.extend(top)
            print(f"family gate {region_id} {family} ig={ig}", flush=True)
        fitted = _fit_box(raw_x, names, list(dict.fromkeys(chosen)), hits, days, fit0, val0, train_mask, horizon)
        roll = rolling_climate_series(hits, days, horizon=horizon)
        if fitted is None:
            series = np.clip(np.where(np.isfinite(roll), roll, recent_climate(hits, days, train_mask)), 1e-6, 1.0 - 1e-6)
            recipe = "fresh_gated|climate"
            bundle = None
        else:
            series = fitted["p"]
            recipe = "fresh_gated|" + "+".join(passing) + "|lift"
            bundle = {key: fitted[key] for key in ("pipeline", "columns", "ref_scores", "ref_hits", "ref_base")}
            models[region_id] = bundle
        p = float(series[latest])
        climate = float(roll[latest]) if np.isfinite(roll[latest]) else float(recent_climate(hits, days, train_mask))
        exam = skill_beats_climate(series[hold_mask], hits[hold_mask], roll[hold_mask])
        draft_series = np.asarray(draft_map.get(region_id), dtype=float)
        draft_ig = None
        if draft_series.size == len(days):
            draft_ig = information_gain(draft_series[val0], hits[val0], roll[val0])
        check_igs[region_id] = _jsonable(draft_ig)
        brier = exam.get("brier")
        climate_brier = exam.get("climatology_brier")
        this_week = {
            "issue_day": days[latest].date().isoformat(),
            "p": p,
            "fresh_raw_p": p,
            "fresh_raw_chance_plain": _chance(p),
            "chance_plain": _chance(p),
            "climatology_p": climate,
            "climatology_chance_plain": _chance(climate),
            "held_to_climatology": True,
            "issued_from": "gated",
            "tournament_winner": "gated",
            "families": passing,
            "climate_kind": "rolling_5y",
        }
        boxes[region_id] = {
            "families": passing,
            "check_ig": family_ig,
            "recipe": recipe,
            "information_gain": _jsonable(exam.get("information_gain")),
            "brier": _jsonable(brier),
            "climatology_brier": _jsonable(climate_brier),
            "this_week": this_week,
        }
        label = _place(region_id)
        week.append(
            {
                "region": region_id,
                "label": label,
                "magnitude_min": mag,
                "horizon_days": horizon,
                "p": p,
                "fresh_raw_p": p,
                "fresh_raw_chance_plain": _chance(p),
                "chance_plain": _chance(p),
                "climatology_p": climate,
                "climatology_chance_plain": _chance(climate),
                "issue_day": days[latest].date().isoformat(),
                "families": passing,
                "issued_from": "gated",
                "why": passing,
                "mode": "frozen",
            }
        )
        if draft_series.size == len(days):
            draft_p = float(draft_series[latest])
            draft_week.append(
                {
                    "region": region_id,
                    "label": label,
                    "magnitude_min": mag,
                    "horizon_days": horizon,
                    "p": draft_p,
                    "chance_plain": _chance(draft_p),
                    "climatology_p": climate,
                    "climatology_chance_plain": _chance(climate),
                    "issue_day": days[latest].date().isoformat(),
                    "check_information_gain": _jsonable(draft_ig),
                    "writes_issued_p": False,
                }
            )
        print(
            f"frozen {region_id} {_chance(p)} typical {_chance(climate)} families={passing or 'none'}",
            flush=True,
        )
    promote = promotion_status(check_igs, {rid: None for rid in check_igs}, n_post_lock=0)
    payload = {
        "locked_at": locked_at,
        "draft_locked_at": locked_at,
        "issue_day": days[latest].date().isoformat(),
        "horizon_days": horizon,
        "family_ig_min": MIN_FAMILY_IG,
        "writes_issued_p": True,
        "week": week,
        "boxes": boxes,
        "draft_families": list(draft_families),
        "draft_recipes": draft_recipes,
        "draft_check_ig": check_igs,
        "promote": promote,
    }
    dest = frozen_dir(root)
    for region_id, bundle in models.items():
        joblib.dump(bundle, dest / f"{region_id}.joblib")
    frozen_path(root).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "frozen": payload,
        "draft_week": draft_week,
        "promote": promote,
        "draft_families": list(draft_families),
    }


def apply_to_mainshock(
    root: Path | None = None,
    *,
    draft_week: list[dict[str, Any]] | None = None,
    promote: dict[str, Any] | None = None,
) -> None:
    """Archive the current card, then publish the frozen week and keep a draft beside it."""
    from eqpred.eval.mainshock import read_mainshock

    report = read_mainshock(root)
    frozen = read_frozen(root)
    if report is None or frozen is None:
        return
    archive = data_dir(root) / "eval" / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source = data_dir(root) / "eval" / "mainshock.json"
    shutil.copy2(source, archive / f"mainshock-before-family-gate-{stamp}.json")
    by_region = {row["region"]: row for row in frozen["week"]}
    for region in report.get("regions") or []:
        box = (frozen.get("boxes") or {}).get(region.get("region"))
        row = by_region.get(region.get("region"))
        if not box or not row:
            continue
        region["recipe"] = box["recipe"]
        region["information_gain"] = box["information_gain"]
        region["brier"] = box["brier"]
        region["climatology_brier"] = box["climatology_brier"]
        region["issued_from"] = "gated"
        region["why"] = list(box.get("families") or [])
        region["beats_climatology"] = family_earns_card(box.get("information_gain"))
        tw = dict(region.get("this_week") or {})
        tw.update(box["this_week"])
        region["this_week"] = tw
        row["sentence"] = (
            "No family beat the five-year rate on the check years, so this card stays on that rate."
            if not box.get("families")
            else "Families that beat the check years moved this card: " + ", ".join(box["families"]) + "."
        )
    report["week"] = frozen["week"]
    report["draft"] = {
        "writes_issued_p": False,
        "locked_at": frozen.get("draft_locked_at"),
        "note": "Working draft. Every family can move this number. It does not replace the frozen card.",
        "week": draft_week or [],
        "check_ig": frozen.get("draft_check_ig") or {},
        "families": frozen.get("draft_families") or [],
        "promote": promote or frozen.get("promote"),
    }
    report["frozen"] = {
        "locked_at": frozen["locked_at"],
        "family_ig_min": MIN_FAMILY_IG,
        "writes_issued_p": True,
        "promote": promote or frozen.get("promote"),
    }
    report["n_beats_climatology"] = sum(1 for region in report.get("regions") or [] if region.get("beats_climatology"))
    source.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
