from __future__ import annotations

"""Frozen 3-day votes and a ledger of the cards we actually score.

The working fit can change. The frozen weights and a card already written
for a morning do not.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.eval.frozen import frozen_dir, read_frozen
from eqpred.eval.skill import MIN_N, information_gain
from eqpred.records import iso


def votes_path(root: Path | None = None) -> Path:
    return frozen_dir(root) / "votes.json"


def ledger_path(root: Path | None = None) -> Path:
    return frozen_dir(root) / "ledger.json"


def read_votes(root: Path | None = None) -> dict[str, Any] | None:
    path = votes_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_ledger(root: Path | None = None) -> dict[str, Any]:
    path = ledger_path(root)
    if not path.exists():
        return {"entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def recipe_hash(boxes: dict[str, Any]) -> str:
    import hashlib

    payload = {
        region: [
            {
                "family": signal.get("family"),
                "columns": signal.get("columns"),
                "mu": [round(float(v), 8) for v in signal.get("mu") or []],
                "sd": [round(float(v), 8) for v in signal.get("sd") or []],
                "scale": round(float(signal.get("scale") or 0.0), 8),
                "cap": round(float(signal.get("cap") or 0.0), 8),
            }
            for signal in (box.get("signals") or [])
        ]
        for region, box in sorted(boxes.items())
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def install_votes(fit: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Lock the current 3-day weights. A file that already exists stays."""
    existing = read_votes(root)
    if existing:
        return existing
    boxes = fit.get("boxes") or {}
    payload = {
        "locked_at": iso(datetime.now(timezone.utc)),
        "horizon_days": int(fit.get("horizon_days") or 3),
        "recipe_hash": recipe_hash(boxes),
        "boxes": boxes,
        "note": "3-day vote weights locked for scoring. A later fit is the working model.",
    }
    votes_path(root).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _window_open(issue_day: str, horizon: int, written_at: datetime) -> bool:
    """True when the card was written before that morning's window closed."""
    issue = pd.Timestamp(issue_day, tz="UTC")
    closes = issue + pd.Timedelta(days=int(horizon))
    written = pd.Timestamp(written_at)
    if written.tzinfo is None:
        written = written.tz_localize("UTC")
    return written <= closes


def _entry_key(entry: dict[str, Any]) -> tuple[int, str, str]:
    return (int(entry.get("horizon_days") or 0), str(entry.get("region") or ""), str(entry.get("issue_day") or ""))


def append_entries(rows: list[dict[str, Any]], *, root: Path | None = None, written_at: datetime | None = None) -> list[dict[str, Any]]:
    """Add cards that are not already on file. An existing morning is left alone."""
    ledger = read_ledger(root)
    entries = list(ledger.get("entries") or [])
    seen = {_entry_key(entry) for entry in entries}
    stamp = written_at or datetime.now(timezone.utc)
    added: list[dict[str, Any]] = []
    for row in rows:
        issue_day = str(row.get("issue_day") or "")
        region = str(row.get("region") or "")
        horizon = int(row.get("horizon_days") or 0)
        if not issue_day or not region or horizon <= 0:
            continue
        key = (horizon, region, issue_day)
        if key in seen:
            continue
        entry = {
            "horizon_days": horizon,
            "region": region,
            "label": row.get("label"),
            "magnitude_min": row.get("magnitude_min"),
            "issue_day": issue_day,
            "p": _jsonable(row.get("p")),
            "climatology_p": _jsonable(row.get("climatology_p")),
            "chance_plain": row.get("chance_plain"),
            "climatology_chance_plain": row.get("climatology_chance_plain"),
            "votes_up": row.get("votes_up") or [],
            "votes_down": row.get("votes_down") or [],
            "lineup_plain": row.get("lineup_plain"),
            "swarm": bool(row.get("swarm")),
            "swarm_plain": row.get("swarm_plain"),
            "information_gain": _jsonable(row.get("information_gain")),
            "prospective": _window_open(issue_day, horizon, stamp),
            "written_at": iso(stamp),
            "locked_at": row.get("locked_at"),
        }
        entries.append(entry)
        seen.add(key)
        added.append(entry)
    ledger["entries"] = entries
    ledger_path(root).write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return added


def _mark_frozen(row: dict[str, Any], locked_at: str) -> dict[str, Any]:
    out = dict(row)
    out["frozen"] = True
    out["score_role"] = "frozen"
    out["locked_at"] = locked_at
    out["writes_issued_p"] = False
    return out


def _apply_signal(raw_x: np.ndarray, names: list[str], signal: dict[str, Any]) -> np.ndarray:
    from eqpred.eval.lineup import _z_from_stats

    index = {name: i for i, name in enumerate(names)}
    columns = list(signal.get("columns") or [])
    if not columns:
        return np.full(raw_x.shape[0], np.nan)
    block = np.full((raw_x.shape[0], len(columns)), np.nan)
    stats: list[tuple[float, float] | None] = []
    present = False
    for j, name in enumerate(columns):
        idx = index.get(name)
        mu = float(signal["mu"][j])
        sd = float(signal["sd"][j])
        stats.append((mu, sd if sd > 1e-12 else 1.0))
        if idx is None:
            continue
        block[:, j] = raw_x[:, idx]
        present = True
    if not present:
        return np.full(raw_x.shape[0], np.nan)
    z = _z_from_stats(block, stats)
    scale = float(signal.get("scale") or 0.0)
    cap = float(signal.get("cap") or 0.0)
    return np.clip(scale * z, -cap, cap)


def apply_vote_rows(
    frozen: dict[str, Any],
    *,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    hold_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Cards from the locked weights. This does not refit a scale."""
    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.lineup import NOTICE, WATCH_CAP, _points, stack_votes
    from eqpred.eval.mainshock import ISSUED_TARGETS, _chance, recent_climate, rolling_climate_series
    from eqpred.eval.skill import skill_beats_climate
    from eqpred.eval.waiting import swarm_mornings
    from eqpred.regions import REGIONS

    raw_x, names = design
    latest = len(days) - 1
    if latest < 0:
        return []
    horizon = int(frozen.get("horizon_days") or 3)
    rows: list[dict[str, Any]] = []
    boxes = frozen.get("boxes") or {}
    for region_id, mag in ISSUED_TARGETS:
        box = boxes.get(region_id) or {}
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(float)
        roll = rolling_climate_series(hits, days, horizon=horizon)
        climate_series = np.where(np.isfinite(roll), roll, recent_climate(hits, days, np.ones(len(days), dtype=bool)))
        added = np.zeros(len(days), dtype=float)
        today: list[tuple[str, float | None]] = []
        reports: list[dict[str, Any]] = []
        for signal in box.get("signals") or []:
            vote = _apply_signal(raw_x, names, signal)
            known = np.isfinite(vote)
            added = added + np.where(known, vote, 0.0)
            today_vote = float(vote[latest]) if known[latest] else None
            family = str(signal.get("family") or "")
            today.append((family, today_vote))
            if today_vote is not None and abs(today_vote) >= NOTICE:
                reports.append(
                    {
                        "family": signal.get("label") or family,
                        "points": today_vote,
                        "points_plain": _points(today_vote),
                    }
                )
        summary = stack_votes(today)
        swarm_flags = swarm_mornings(catalog, REGIONS[region_id], mag, days)
        p_series = np.clip(climate_series + added, 1e-6, 1.0 - 1e-6)
        p_series = np.where(swarm_flags, climate_series, p_series)
        p_series = np.clip(np.where(np.isfinite(p_series), p_series, climate_series), 1e-6, 1.0 - 1e-6)
        climate = float(climate_series[latest]) if np.isfinite(climate_series[latest]) else float("nan")
        p = float(p_series[latest])
        exam_ig = _jsonable((box.get("information_gain")))
        if hold_mask is not None and int(np.asarray(hold_mask).sum()) >= MIN_N:
            exam = skill_beats_climate(p_series[hold_mask], hits[hold_mask], roll[hold_mask])
            live_ig = exam.get("information_gain")
            if exam_ig is None and live_ig is not None and np.isfinite(float(live_ig)):
                exam_ig = float(live_ig)
        up = [item for item in reports if item["points"] > 0]
        down = [item for item in reports if item["points"] < 0]
        up.sort(key=lambda item: -item["points"])
        down.sort(key=lambda item: item["points"])
        label = box.get("label") or region_id
        rows.append(
            _mark_frozen(
                {
                    "region": region_id,
                    "label": label,
                    "magnitude_min": mag,
                    "horizon_days": horizon,
                    "p": p,
                    "chance_plain": _chance(p),
                    "climatology_p": None if not np.isfinite(climate) else climate,
                    "climatology_chance_plain": _chance(climate),
                    "issue_day": days[latest].date().isoformat(),
                    "information_gain": exam_ig,
                    "beats_climatology": exam_ig is not None and exam_ig >= 0.01,
                    "beside_30_day": True,
                    "issued_from": "votes",
                    "vote_cap": WATCH_CAP,
                    "lineup_plain": (
                        f"{len(summary['up'])} point up, {len(summary['down'])} point down, "
                        f"{summary['n_blank']} had no reading."
                    ),
                    "votes_up": up[:6],
                    "votes_down": down[:6],
                    "swarm": bool(swarm_flags[latest]),
                    "swarm_plain": (
                        "Aftershocks of a shock already reported. The votes are listed and do not move this chance."
                        if bool(swarm_flags[latest])
                        else None
                    ),
                },
                str(frozen.get("locked_at") or ""),
            )
        )
    return rows


def _ledger_rows(rows: list[dict[str, Any]], *, root: Path | None) -> list[dict[str, Any]]:
    """Prefer a card already written for that morning."""
    by_key = {_entry_key(entry): entry for entry in read_ledger(root).get("entries") or []}
    out: list[dict[str, Any]] = []
    for row in rows:
        saved = by_key.get(_entry_key(row))
        if not saved:
            out.append(row)
            continue
        kept = dict(row)
        for field in (
            "p",
            "climatology_p",
            "chance_plain",
            "climatology_chance_plain",
            "votes_up",
            "votes_down",
            "lineup_plain",
            "swarm",
            "swarm_plain",
        ):
            if saved.get(field) is not None:
                kept[field] = saved[field]
        kept["prospective"] = bool(saved.get("prospective"))
        out.append(kept)
    return out


def hold_vote_cards(
    fit: dict[str, Any],
    *,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    hold_mask: np.ndarray | None = None,
    root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Published 3-day cards stay on the frozen weights. The new fit is the working model."""
    frozen = install_votes(fit, root)
    fresh_rows = [_mark_frozen(row, str(frozen.get("locked_at") or "")) for row in fit.get("rows") or []]
    if read_votes(root) and recipe_hash(fit.get("boxes") or {}) == frozen.get("recipe_hash") and fresh_rows:
        published = fresh_rows
        same = True
    else:
        published = apply_vote_rows(
            frozen,
            design=design,
            catalog=catalog,
            days=days,
            hold_mask=hold_mask,
        ) or fresh_rows
        same = recipe_hash(fit.get("boxes") or {}) == frozen.get("recipe_hash")
    published = _ledger_rows(published, root=root)
    append_entries(published, root=root)
    published = _ledger_rows(published, root=root)
    draft_rows = []
    for row in fit.get("rows") or []:
        copy = dict(row)
        copy["frozen"] = False
        copy["score_role"] = "working"
        copy["writes_issued_p"] = False
        draft_rows.append(copy)
    draft = {
        "writes_issued_p": False,
        "locked_at": frozen.get("locked_at"),
        "horizon_days": 3,
        "same_as_frozen": same,
        "note": (
            "Working 3-day model. It matches the frozen votes until this model is changed."
            if same
            else "Working 3-day model. It can change. It does not replace the frozen card we score."
        ),
        "week": draft_rows,
    }
    return published, draft


def frozen_30_rows(root: Path | None = None) -> list[dict[str, Any]]:
    frozen = read_frozen(root)
    if not frozen:
        return []
    rows = []
    for row in frozen.get("week") or []:
        copy = dict(row)
        copy["horizon_days"] = int(copy.get("horizon_days") or frozen.get("horizon_days") or 30)
        copy["locked_at"] = frozen.get("locked_at")
        copy["frozen"] = True
        rows.append(copy)
    return rows


def seal_frozen_30(root: Path | None = None, *, written_at: datetime | None = None) -> list[dict[str, Any]]:
    """Record the frozen 30-day card once. A later refit does not replace it."""
    return append_entries(frozen_30_rows(root), root=root, written_at=written_at)


def apply_frozen_30_latest(
    *,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray | None = None,
    root: Path | None = None,
    replay: bool = False,
) -> list[dict[str, Any]]:
    """One new morning from the locked 30-day model. The saved card for an old morning stays."""
    import joblib

    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.mainshock import (
        ISSUED_TARGETS,
        _chance,
        _lift_onto_rate,
        _proba,
        _score_to_rate,
        recent_climate,
        rolling_climate_series,
    )
    from eqpred.regions import REGIONS

    frozen = read_frozen(root)
    if frozen is None or len(days) == 0:
        return []
    raw_x, names = design
    latest = len(days) - 1
    issue_day = days[latest].date().isoformat()
    if issue_day == str(frozen.get("issue_day") or "") and not replay:
        return frozen_30_rows(root)
    index = {name: i for i, name in enumerate(names)}
    horizon = int(frozen.get("horizon_days") or 30)
    rows: list[dict[str, Any]] = []
    mask = np.ones(len(days), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    for region_id, mag in ISSUED_TARGETS:
        path = frozen_dir(root) / f"{region_id}.joblib"
        if not path.exists():
            continue
        bundle = joblib.load(path)
        columns = list(bundle.get("columns") or [])
        block = np.full((raw_x.shape[0], len(columns)), np.nan)
        for j, name in enumerate(columns):
            idx = index.get(str(name))
            if idx is not None:
                block[:, j] = raw_x[:, idx]
        score = np.asarray(_proba(bundle["pipeline"], block), dtype=float).reshape(-1)
        ref_scores = np.asarray(bundle["ref_scores"], dtype=float).reshape(-1)
        ref_hits = np.asarray(bundle["ref_hits"], dtype=float).reshape(-1)
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(float)
        roll = rolling_climate_series(hits, days, horizon=horizon)
        combined_score = np.concatenate([ref_scores, score[-1:]])
        combined_hits = np.concatenate([ref_hits, hits[-1:]])
        ref = np.zeros(combined_score.shape[0], dtype=bool)
        ref[: ref_scores.shape[0]] = True
        similar = _score_to_rate(combined_score, combined_hits, ref, max_percentile=100)
        climate = recent_climate(hits, days, mask)
        foil = np.concatenate([np.full(ref_scores.shape[0], climate), roll[-1:]])
        issued = _lift_onto_rate(similar, combined_hits, ref, foil, float(bundle.get("ref_base") or climate))
        p = float(issued[-1])
        climate_p = float(roll[-1]) if np.isfinite(roll[-1]) else float(climate)
        box = (frozen.get("boxes") or {}).get(region_id) or {}
        rows.append(
            {
                "region": region_id,
                "label": next(
                    (
                        row.get("label")
                        for row in (frozen.get("week") or [])
                        if row.get("region") == region_id
                    ),
                    region_id,
                ),
                "magnitude_min": mag,
                "horizon_days": horizon,
                "p": p,
                "chance_plain": _chance(p),
                "climatology_p": climate_p,
                "climatology_chance_plain": _chance(climate_p),
                "issue_day": issue_day,
                "families": list(box.get("families") or []),
                "issued_from": "gated",
                "frozen": True,
                "locked_at": frozen.get("locked_at"),
                "writes_issued_p": True,
            }
        )
    return rows


def seal_new_mornings(
    *,
    design: tuple[np.ndarray, list[str]] | None = None,
    catalog: pd.DataFrame | None = None,
    days: pd.DatetimeIndex | None = None,
    train_mask: np.ndarray | None = None,
    vote_rows: list[dict[str, Any]] | None = None,
    root: Path | None = None,
) -> None:
    """Write any frozen card that is not on the ledger yet."""
    seal_frozen_30(root)
    if vote_rows:
        append_entries(vote_rows, root=root)
    if design is None or catalog is None or days is None or len(days) == 0:
        return
    frozen = read_frozen(root)
    latest = days[-1].date().isoformat()
    if frozen and latest != str(frozen.get("issue_day") or ""):
        append_entries(
            apply_frozen_30_latest(
                design=design,
                catalog=catalog,
                days=days,
                train_mask=train_mask,
                root=root,
            ),
            root=root,
        )
    votes = read_votes(root)
    if votes and latest != _latest_ledger_day(3, root):
        hold_mask = np.zeros(len(days), dtype=bool)
        append_entries(
            apply_vote_rows(votes, design=design, catalog=catalog, days=days, hold_mask=hold_mask),
            root=root,
        )


def _latest_ledger_day(horizon: int, root: Path | None) -> str | None:
    days = [
        str(entry.get("issue_day") or "")
        for entry in read_ledger(root).get("entries") or []
        if int(entry.get("horizon_days") or 0) == horizon and entry.get("issue_day")
    ]
    return max(days) if days else None


def _plain(horizon: int, n_prospective: int, n_closed: int, n_hits: int, information_gain_value: float | None) -> str:
    window = f"{horizon}-day"
    if n_prospective <= 0:
        if horizon == 3:
            return (
                "The 3-day votes are frozen. This morning's window had already closed before the lock, "
                "so the live score starts at the next new morning."
            )
        return f"The frozen {window} cards are not on the score ledger yet."
    if n_closed <= 0:
        return (
            f"The frozen {window} cards are the ones we score. "
            f"{n_prospective} window{'s' if n_prospective != 1 else ''} still open."
        )
    shocks = "1 first shock" if n_hits == 1 else f"{n_hits} first shocks"
    closed = f"{n_closed} closed window{'s' if n_closed != 1 else ''}"
    if information_gain_value is None:
        return (
            f"Frozen {window} score: {shocks} in {closed}. "
            "A skill number waits until 8 windows have closed."
        )
    return (
        f"Frozen {window} score: {shocks} in {closed}. "
        f"Information gain against the typical rate is {information_gain_value:+.3f}."
    )


def score_frozen_ledger(quakes: pd.DataFrame, root: Path | None = None) -> dict[str, Any]:
    """Score only cards written before their window closed."""
    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.mainshock import ISSUED_TARGETS
    from eqpred.regions import REGIONS
    from eqpred.stamp import parse_utc

    entries = list(read_ledger(root).get("entries") or [])
    catalog_end = None
    if quakes is not None and len(quakes) and "observed_at" in quakes.columns:
        observed = parse_utc(quakes["observed_at"])
        if len(observed):
            catalog_end = observed.max()
    mags = {region_id: mag for region_id, mag in ISSUED_TARGETS}
    out: dict[str, Any] = {}
    for horizon in (30, 3):
        group = [
            entry
            for entry in entries
            if int(entry.get("horizon_days") or 0) == horizon and entry.get("prospective")
        ]
        closed_p: list[float] = []
        closed_y: list[float] = []
        closed_c: list[float] = []
        n_hits = 0
        if catalog_end is not None and group:
            by_region: dict[str, list[dict[str, Any]]] = {}
            for entry in group:
                by_region.setdefault(str(entry.get("region") or ""), []).append(entry)
            for region_id, rows in by_region.items():
                mag = mags.get(region_id)
                if mag is None or region_id not in REGIONS:
                    continue
                issue_days = pd.DatetimeIndex([pd.Timestamp(row["issue_day"], tz="UTC") for row in rows])
                hits = fresh_hits(quakes, REGIONS[region_id], mag, issue_days, horizon)
                for entry, hit in zip(rows, hits, strict=True):
                    issue = pd.Timestamp(entry["issue_day"], tz="UTC")
                    if catalog_end < issue + pd.Timedelta(days=horizon):
                        continue
                    p = entry.get("p")
                    climate = entry.get("climatology_p")
                    if p is None or climate is None:
                        continue
                    closed_p.append(float(p))
                    closed_y.append(1.0 if bool(hit) else 0.0)
                    closed_c.append(float(climate))
                    n_hits += int(bool(hit))
        ig = None
        if len(closed_p) >= MIN_N:
            ig = information_gain(np.asarray(closed_p), np.asarray(closed_y), np.asarray(closed_c))
            ig = None if ig is None or not np.isfinite(ig) else float(ig)
        n_closed = len(closed_p)
        n_prospective = len(group)
        out[str(horizon)] = {
            "horizon_days": horizon,
            "n_prospective": n_prospective,
            "n_closed": n_closed,
            "n_open": n_prospective - n_closed,
            "n_hits": n_hits,
            "information_gain": ig,
            "plain": _plain(horizon, n_prospective, n_closed, n_hits, ig),
        }
    return out
