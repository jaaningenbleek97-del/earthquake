from __future__ import annotations

"""Lines that explain the frozen cards. They do not change a chance."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.eval.frozen import frozen_dir, read_frozen
from eqpred.eval.lineup import DAVIDSON_BURNS_FAMILIES, watch_agreement

TIGHT_SPAN = 0.25
EXPLAIN_NAME = "explain.json"


def explain_path(root: Path | None = None) -> Path:
    return frozen_dir(root) / EXPLAIN_NAME


def read_explain(root: Path | None = None) -> dict[str, Any] | None:
    path = explain_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def neighborhood_from_ref(
    score: float,
    ref_scores: np.ndarray,
    ref_hits: np.ndarray,
    *,
    min_n: int = 80,
) -> dict[str, Any]:
    """The 80 closest check-year mornings behind a frozen chance. The chance stays put."""
    scores = np.asarray(ref_scores, dtype=float).reshape(-1)
    hits = np.asarray(ref_hits, dtype=float).reshape(-1)
    ok = np.isfinite(scores) & np.isfinite(hits)
    scores = scores[ok]
    hits = hits[ok]
    n_ref = int(scores.size)
    if n_ref < 2 or not np.isfinite(score):
        return {"n": 0, "hits": 0, "tight": False, "plain": None}
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    hits = hits[order]
    k = min(int(min_n), n_ref)
    query = min(float(score), float(scores[-1]))
    center = int(np.searchsorted(scores, query, side="left"))
    left = int(np.clip(center - (k // 2), 0, n_ref - k))
    window_scores = scores[left : left + k]
    window_hits = hits[left : left + k]
    n_hits = int(np.sum(window_hits >= 0.5))
    full = float(scores[-1] - scores[0])
    span = float(window_scores[-1] - window_scores[0])
    tight = full <= 0 or span / full < TIGHT_SPAN
    band = "Those scores sit in a tight band." if tight else "Those scores are spread across most of the range."
    if n_hits == 0:
        shook = f"None of the {k} closest check-year mornings shook."
    elif n_hits == 1:
        shook = f"1 of the {k} closest check-year mornings shook."
    else:
        shook = f"{n_hits} of the {k} closest check-year mornings shook."
    return {"n": k, "hits": n_hits, "tight": tight, "plain": f"{shook} {band}"}


def furthest_above_rate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The box whose chance sits furthest above its own usual rate."""
    best = None
    best_lift = None
    for row in rows:
        chance = row.get("p")
        climate = row.get("climatology_p")
        if chance is None or climate is None:
            continue
        if not np.isfinite(float(chance)) or not np.isfinite(float(climate)):
            continue
        lift = float(chance) - float(climate)
        if best_lift is None or lift > best_lift:
            best_lift = lift
            best = row
    if best is None or best_lift is None:
        return None
    return {
        "region": best.get("region"),
        "label": best.get("label"),
        "lift": best_lift,
        "chance_plain": best.get("chance_plain"),
        "climatology_chance_plain": best.get("climatology_chance_plain"),
    }


def story_plain(
    *,
    lift: dict[str, Any] | None,
    odd: dict[str, Any] | None,
    agreement: dict[str, Any] | None,
    shocks: list[dict[str, Any]],
) -> str:
    """Three names, then which one matched once a saved window contains a first shock."""
    parts: list[str] = []
    if lift and lift.get("label"):
        parts.append(
            f"{lift['label']} is furthest above its usual rate, "
            f"{lift.get('chance_plain') or '—'} against {lift.get('climatology_chance_plain') or '—'}."
        )
    if odd and odd.get("label"):
        parts.append(f"The odd gap is {odd['label']}.")
    if agreement and agreement.get("plain"):
        parts.append(str(agreement["plain"]))
    if not shocks:
        parts.append("Which story was right waits for a first shock in a window that was saved in time.")
        return " ".join(parts)
    labels = [str(row.get("label") or row.get("region")) for row in shocks]
    where = ", ".join(labels)
    parts.append(f"A first shock has been reported in {where}.")
    shock_ids = {str(row.get("region")) for row in shocks}
    matched: list[str] = []
    if lift and str(lift.get("region")) in shock_ids:
        matched.append(f"it was {lift.get('label')}, the box furthest above its usual rate")
    if odd and str(odd.get("region")) in shock_ids:
        matched.append(f"it was {odd.get('label')}, the odd-gap box")
    if matched:
        parts.append("That shock matched " + " and ".join(matched) + ".")
    else:
        parts.append("That shock was neither the high-lift box nor the odd-gap box.")
    if agreement and agreement.get("agreed") and agreement.get("direction") == "up":
        parts.append("The solar-watch families were pointing up.")
    elif agreement and agreement.get("agreed") and agreement.get("direction") == "down":
        parts.append("The solar-watch families were pointing down.")
    else:
        parts.append("The solar-watch families had not agreed.")
    return " ".join(parts)


def _encoder_score(bundle: dict[str, Any], raw_x: np.ndarray, names: list[str], row: int) -> float:
    from eqpred.eval.mainshock import _proba

    columns = list(bundle.get("columns") or [])
    index = {name: i for i, name in enumerate(names)}
    block = np.full((1, len(columns)), np.nan)
    for j, name in enumerate(columns):
        idx = index.get(str(name))
        if idx is not None:
            block[0, j] = raw_x[row, idx]
    return float(_proba(bundle["pipeline"], block)[0])


def _watch_votes_today(signals: list[dict[str, Any]], raw_x: np.ndarray, names: list[str]) -> list[tuple[str, float | None]]:
    from eqpred.eval.vote_freeze import _apply_signal

    latest = raw_x.shape[0] - 1
    votes: list[tuple[str, float | None]] = []
    for signal in signals:
        family = str(signal.get("family") or "")
        if family not in DAVIDSON_BURNS_FAMILIES:
            continue
        vote = _apply_signal(raw_x, names, signal)
        value = float(vote[latest]) if np.isfinite(vote[latest]) else None
        votes.append((family, value))
    return votes


def _agreement_exam(
    signals: list[dict[str, Any]],
    raw_x: np.ndarray,
    names: list[str],
    hits: np.ndarray,
    closed: np.ndarray,
) -> dict[str, Any]:
    from eqpred.eval.vote_freeze import _apply_signal

    series: dict[str, np.ndarray] = {}
    for signal in signals:
        family = str(signal.get("family") or "")
        if family not in DAVIDSON_BURNS_FAMILIES:
            continue
        series[family] = _apply_signal(raw_x, names, signal)
    flags = np.zeros(raw_x.shape[0], dtype=bool)
    for i in range(raw_x.shape[0]):
        votes = []
        for family, column in series.items():
            value = float(column[i]) if np.isfinite(column[i]) else None
            votes.append((family, value))
        flags[i] = bool(watch_agreement(votes)["agreed"])
    use = np.asarray(closed, dtype=bool)
    agree = use & flags
    other = use & ~flags
    n_agree = int(agree.sum())
    n_other = int(other.sum())
    rate_agree = float(np.mean(hits[agree])) if n_agree else None
    rate_other = float(np.mean(hits[other])) if n_other else None
    if n_agree < 8 or rate_agree is None or rate_other is None:
        plain = (
            f"Only {n_agree} later mornings were agreement days, so that comparison waits."
        )
    else:
        plain = (
            "On later closed mornings, Earth-wide agreement days were followed by a first shock "
            f"{int(round(100 * rate_agree))} times in 100, against {int(round(100 * rate_other))} in 100 on other days."
        )
    return {
        "n_agree": n_agree,
        "n_other": n_other,
        "rate_agree": rate_agree,
        "rate_other": rate_other,
        "plain": plain,
    }


def build_explain(root: Path | None = None) -> dict[str, Any]:
    """Neighborhood, agreement, and the three-story line for the frozen morning."""
    import joblib

    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.mainshock import ISSUED_TARGETS, prepare_issued
    from eqpred.eval.vote_freeze import read_votes
    from eqpred.eval.waiting import odd_box
    from eqpred.regions import REGIONS
    from eqpred.stamp import utc_ts

    feat, catalog, days, design, catalog_end = prepare_issued(root)
    raw_x, names = design
    frozen = read_frozen(root) or {}
    issue_day = str(frozen.get("issue_day") or days[-1].date().isoformat())
    issue_ts = utc_ts(issue_day)
    issue_stamp = pd.Timestamp(issue_day, tz="UTC")
    day_floor = pd.DatetimeIndex(days).tz_convert("UTC").floor("D")
    matches = np.flatnonzero(day_floor == issue_stamp)
    row = int(matches[-1]) if matches.size else len(days) - 1
    neighborhoods: dict[str, Any] = {}
    for region_id, _mag in ISSUED_TARGETS:
        path = frozen_dir(root) / f"{region_id}.joblib"
        if not path.exists():
            continue
        bundle = joblib.load(path)
        score = _encoder_score(bundle, raw_x, names, row)
        neighborhoods[region_id] = neighborhood_from_ref(
            score,
            np.asarray(bundle["ref_scores"]),
            np.asarray(bundle["ref_hits"]),
        )
    votes = read_votes(root) or {}
    boxes = votes.get("boxes") or {}
    agreements: dict[str, Any] = {}
    for region_id, _mag in ISSUED_TARGETS:
        signals = list((boxes.get(region_id) or {}).get("signals") or [])
        agreements[region_id] = watch_agreement(_watch_votes_today(signals, raw_x, names))
    global_signals = list((boxes.get("global") or {}).get("signals") or [])
    hits = fresh_hits(catalog, REGIONS["global"], 7.0, days, 3).astype(float)
    closed = np.asarray(days >= pd.Timestamp("2019-01-01", tz="UTC")) & np.asarray(
        days + pd.Timedelta(days=3) <= catalog_end
    )
    exam = _agreement_exam(global_signals, raw_x, names, hits, closed)
    week = [dict(item) for item in frozen.get("week") or []]
    lift = furthest_above_rate(week)
    odd = odd_box(catalog, issue_ts)
    shocks = _saved_window_shocks(catalog, issue_ts, root)
    story = {
        "lift_region": None if lift is None else lift.get("region"),
        "lift_label": None if lift is None else lift.get("label"),
        "odd_region": odd.get("region"),
        "odd_label": odd.get("label"),
        "agreement": agreements.get("global"),
        "shocks": shocks,
        "plain": story_plain(
            lift=lift,
            odd=odd,
            agreement=agreements.get("global"),
            shocks=shocks,
        ),
    }
    payload = {
        "issue_day": issue_day,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "neighborhoods": neighborhoods,
        "agreements": agreements,
        "agreement_exam": exam,
        "story": story,
        "writes_issued_p": False,
    }
    explain_path(root).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _saved_window_shocks(catalog: pd.DataFrame, issue: pd.Timestamp, root: Path | None) -> list[dict[str, Any]]:
    """First shocks inside a 30-day card that was saved before its window closed."""
    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.mainshock import ISSUED_TARGETS, _place
    from eqpred.eval.vote_freeze import read_ledger
    from eqpred.regions import REGIONS

    closes = issue + pd.Timedelta(days=30)

    def _saved_before_close(entry: dict[str, Any]) -> bool:
        written = entry.get("written_at")
        if not written:
            return False
        stamp = pd.Timestamp(written)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        return stamp <= closes

    saved = [
        entry
        for entry in (read_ledger(root).get("entries") or [])
        if int(entry.get("horizon_days") or 0) == 30
        and str(entry.get("issue_day") or "")[:10] == str(issue.date())
        and _saved_before_close(entry)
    ]
    if not saved:
        return []
    days = pd.DatetimeIndex([issue])
    found: list[dict[str, Any]] = []
    for region_id, mag in ISSUED_TARGETS:
        if not any(str(entry.get("region")) == region_id for entry in saved):
            continue
        hit = bool(fresh_hits(catalog, REGIONS[region_id], mag, days, 30, label_version=1)[0])
        if hit:
            found.append({"region": region_id, "label": _place(region_id)})
    return found
