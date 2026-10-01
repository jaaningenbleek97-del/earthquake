from __future__ import annotations

"""3-day card: each signal adds a few probability points. The 30-day card is untouched."""

from typing import Any

import numpy as np
import pandas as pd

from eqpred.eval.skill import MIN_N, information_gain

VOTE_CAP = 0.02
NOTICE = 0.005
WATCH_CAP = 0.03
OTHER_CAP = 0.01
EARNED_CAP = 0.06
# What Davidson and Burns tell people to watch. The weight still comes from
# the early years. A signal that pointed the wrong way still votes that way.
DAVIDSON_BURNS_FAMILIES = (
    "solar",
    "spf",
    "schumann",
    "geometry",
    "cosmic",
    "cusp",
    "dust",
    "volcano",
)
AGREE_MIN = 3


def watch_agreement(votes: list[tuple[str, float | None]]) -> dict[str, Any]:
    """Solar-watch families agree when two-thirds of the readings point the same way.

    At least three families need a reading. A nudge smaller than the notice
    does not count as pointing. This mark does not change a chance.
    """
    readings = [
        (name, float(value))
        for name, value in votes
        if name in DAVIDSON_BURNS_FAMILIES and value is not None and np.isfinite(value)
    ]
    n = len(readings)
    up = [name for name, value in readings if value >= NOTICE]
    down = [name for name, value in readings if value <= -NOTICE]
    if n < AGREE_MIN:
        return {
            "agreed": False,
            "direction": None,
            "n_readings": n,
            "n_up": len(up),
            "n_down": len(down),
            "plain": f"No agreement. Only {n} solar-watch reading{'' if n == 1 else 's'}.",
        }
    direction = None
    if len(up) * 3 >= n * 2:
        direction = "up"
    elif len(down) * 3 >= n * 2:
        direction = "down"
    if direction == "up":
        plain = f"{len(up)} of {n} solar-watch readings point up."
    elif direction == "down":
        plain = f"{len(down)} of {n} solar-watch readings point down."
    else:
        plain = (
            f"No agreement. {len(up)} of {n} solar-watch readings point up and {len(down)} point down."
        )
    return {
        "agreed": direction is not None,
        "direction": direction,
        "n_readings": n,
        "n_up": len(up),
        "n_down": len(down),
        "plain": plain,
    }


def vote_cap(family: str) -> float:
    """Default ceiling. Solar-watch signals stop at 3 points. The rest stop at 1."""
    return WATCH_CAP if family in DAVIDSON_BURNS_FAMILIES else OTHER_CAP


def earned_ceiling(family: str, fit_scale: float, check_scale: float) -> float:
    """Raise the ceiling only when the early-year weight is already large and the check years agree."""
    base = vote_cap(family)
    if not np.isfinite(fit_scale) or not np.isfinite(check_scale):
        return base
    if fit_scale == 0.0 or check_scale == 0.0 or (fit_scale > 0) != (check_scale > 0):
        return base
    if abs(fit_scale) <= base:
        return base
    return float(min(EARNED_CAP, abs(fit_scale)))


def _direction(z: np.ndarray, hits: np.ndarray, roll: np.ndarray, mask: np.ndarray) -> float:
    """Sign of the link on one period. This does not replace the learned weight."""
    ok = np.asarray(mask, dtype=bool) & np.isfinite(z) & np.isfinite(hits) & np.isfinite(roll)
    if int(ok.sum()) < 80:
        return 0.0
    resid = hits[ok] - roll[ok]
    zz = z[ok] - float(np.mean(z[ok]))
    var = float(np.mean(zz * zz))
    if var < 1e-12:
        return 0.0
    direction = float(np.mean(zz * resid) / var)
    return direction if np.isfinite(direction) else 0.0


def stack_votes(nudges: list[tuple[str, float | None]]) -> dict[str, Any]:
    """Add each family's nudge. None means that family had no reading."""
    total = 0.0
    up: list[tuple[str, float]] = []
    down: list[tuple[str, float]] = []
    present = 0
    for name, nudge in nudges:
        if nudge is None or not np.isfinite(nudge):
            continue
        present += 1
        clipped = float(nudge)
        total += clipped
        if clipped >= NOTICE:
            up.append((name, clipped))
        elif clipped <= -NOTICE:
            down.append((name, clipped))
    up.sort(key=lambda item: -item[1])
    down.sort(key=lambda item: item[1])
    return {
        "total": total,
        "up": up,
        "down": down,
        "n_present": present,
        "n_blank": len(nudges) - present,
    }


def _points(vote: float) -> str:
    return f"{vote * 100:+.1f}"


def _column_stats(
    block: np.ndarray, fit_mask: np.ndarray
) -> list[tuple[float, float] | None]:
    """Fit-year mean and spread. A dead column stays unused."""
    fit = np.asarray(fit_mask, dtype=bool)
    stats: list[tuple[float, float] | None] = []
    for i in range(block.shape[1]):
        train = block[:, i][fit]
        finite = train[np.isfinite(train)]
        if finite.size < 20 or float(np.std(finite)) < 1e-12:
            stats.append(None)
            continue
        stats.append((float(np.mean(finite)), float(np.std(finite))))
    return stats


def _z_from_stats(block: np.ndarray, stats: list[tuple[float, float] | None]) -> np.ndarray:
    """Mean standard score from saved mean and spread. A blank morning stays blank."""
    parts: list[np.ndarray] = []
    present = np.zeros(block.shape[0], dtype=bool)
    for i, stat in enumerate(stats):
        if stat is None:
            continue
        mu, sd = stat
        column = block[:, i]
        z = np.full(column.shape, np.nan)
        ok = np.isfinite(column)
        z[ok] = (column[ok] - mu) / sd
        parts.append(z)
        present |= ok
    if not parts:
        return np.full(block.shape[0], np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(np.vstack(parts), axis=0)
    return np.where(present, mean, np.nan)


def _family_z(block: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Mean standard score. A morning with no finite column stays blank."""
    return _z_from_stats(block, _column_stats(block, fit_mask))


def _scale(z: np.ndarray, hits: np.ndarray, roll: np.ndarray, fit_mask: np.ndarray) -> float:
    """Points of 3-day chance per standard score, fit on early years only.

    A noisy weight is pulled toward zero. The 3-point and 1-point caps still
    apply afterwards, so one signal cannot take over the card.
    """
    ok = np.asarray(fit_mask, dtype=bool) & np.isfinite(z) & np.isfinite(hits) & np.isfinite(roll)
    n = int(ok.sum())
    if n < 80:
        return 0.0
    resid = hits[ok] - roll[ok]
    zz = z[ok]
    zz = zz - float(np.mean(zz))
    var = float(np.mean(zz * zz))
    if var < 1e-12:
        return 0.0
    scale = float(np.mean(zz * resid) / var)
    if not np.isfinite(scale):
        return 0.0
    fitted = scale * zz
    noise = float(np.mean((resid - fitted) ** 2))
    uncertainty = noise / (n * var)
    if not np.isfinite(uncertainty) or uncertainty < 0:
        return 0.0
    strength = scale * scale
    shrink = strength / (strength + uncertainty) if strength + uncertainty > 0 else 0.0
    return float(scale * shrink)


def three_day_vote_rows(
    *,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    horizon: int = 3,
) -> list[dict[str, Any]]:
    """One 3-day card per issued box. Chance is the typical rate plus the votes."""
    return three_day_vote_fit(
        design=design,
        catalog=catalog,
        days=days,
        train_mask=train_mask,
        hold_mask=hold_mask,
        horizon=horizon,
    )["rows"]


def three_day_vote_fit(
    *,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    horizon: int = 3,
) -> dict[str, Any]:
    """3-day cards plus the weights that made them. The weights can be frozen."""
    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.honest import fit_val_split
    from eqpred.eval.mainshock import (
        ISSUED_TARGETS,
        SHARED_FAMILIES,
        SHRINK_PER_FAMILY,
        _chance,
        _family_pool,
        _place,
        _pool_top,
        _usable_indices,
        recent_climate,
        rolling_climate_series,
    )
    from eqpred.eval.skill import skill_beats_climate
    from eqpred.regions import REGIONS

    raw_x, names = design
    fit0, val0 = fit_val_split(days, train_mask, purge=horizon)
    usable = _usable_indices(raw_x, fit0)
    latest = len(days) - 1
    rows: list[dict[str, Any]] = []
    boxes: dict[str, Any] = {}
    for region_id, mag in ISSUED_TARGETS:
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(float)
        roll = rolling_climate_series(hits, days, horizon=horizon)
        climate_series = np.where(np.isfinite(roll), roll, recent_climate(hits, days, train_mask))
        added = np.zeros(len(days), dtype=float)
        today: list[tuple[str, float | None]] = []
        reports: list[dict[str, Any]] = []
        signals: list[dict[str, Any]] = []
        for family in SHARED_FAMILIES:
            pool = _family_pool(names, family, usable)
            sl = raw_x[fit0] if fit0.any() else raw_x
            live = [
                idx
                for idx in pool
                if int(np.isfinite(sl[:, idx]).sum()) >= 20 and float(np.nanstd(sl[:, idx])) > 1e-12
            ]
            top = _pool_top(raw_x, hits, fit0, live, SHRINK_PER_FAMILY)
            if not top:
                continue
            candidates = [("family", top)]
            for idx in top:
                candidates.append((names[idx], [idx]))
            best_ig = None
            best = None
            for label, cols in candidates:
                z_try = _family_z(raw_x[:, cols], fit0)
                scale_try = _scale(z_try, hits, roll, fit0)
                if scale_try == 0.0:
                    continue
                check_try = _direction(z_try, hits, roll, val0)
                trial = np.clip(scale_try * z_try, -EARNED_CAP, EARNED_CAP)
                known_try = np.isfinite(z_try)
                check_p = np.clip(
                    climate_series + np.where(known_try, np.where(np.isfinite(trial), trial, 0.0), 0.0),
                    1e-6,
                    1.0 - 1e-6,
                )
                ig_try = (
                    information_gain(check_p[val0], hits[val0], roll[val0]) if int(val0.sum()) >= MIN_N else None
                )
                if label == "family":
                    best = (label, z_try, scale_try, check_try, ig_try, cols)
                    best_ig = ig_try
                    continue
                agrees = check_try != 0.0 and (scale_try > 0) == (check_try > 0)
                beats = (
                    agrees
                    and ig_try is not None
                    and ig_try > 0.0
                    and best_ig is not None
                    and ig_try > best_ig
                )
                if beats and (best is None or best[0] == "family" or ig_try > best[4]):
                    best = (label, z_try, scale_try, check_try, ig_try, cols)
            if best is None or best[0] == "family":
                chosen = top
                z = _family_z(raw_x[:, chosen], fit0)
                scale = _scale(z, hits, roll, fit0)
                check_scale = _direction(z, hits, roll, val0)
                shown = family
            else:
                shown, z, scale, check_scale, _ig_col, chosen = best
                shown = f"{family}: {shown.split('__lag')[0].replace('_', ' ')}"
            stats = _column_stats(raw_x[:, chosen], fit0)
            kept = [
                (names[idx], mu, sd)
                for idx, stat in zip(chosen, stats, strict=True)
                if stat is not None
                for mu, sd in [stat]
            ]
            if kept and scale != 0.0:
                signals.append(
                    {
                        "family": family,
                        "label": shown,
                        "columns": [name for name, _mu, _sd in kept],
                        "mu": [mu for _name, mu, _sd in kept],
                        "sd": [sd for _name, _mu, sd in kept],
                        "scale": float(scale),
                        "cap": float(earned_ceiling(family, scale, check_scale)),
                    }
                )
            cap = earned_ceiling(family, scale, check_scale)
            vote = np.clip(scale * z, -cap, cap)
            known = np.isfinite(z)
            added = added + np.where(known, np.where(np.isfinite(vote), vote, 0.0), 0.0)
            check_p = np.clip(climate_series + np.where(known, vote, 0.0), 1e-6, 1.0 - 1e-6)
            ig = information_gain(check_p[val0], hits[val0], roll[val0]) if int(val0.sum()) >= MIN_N else None
            today_vote = float(vote[latest]) if known[latest] and np.isfinite(vote[latest]) else None
            today.append((family, today_vote))
            if today_vote is not None and abs(today_vote) >= NOTICE:
                reports.append(
                    {
                        "family": shown,
                        "points": today_vote,
                        "points_plain": _points(today_vote),
                        "check_information_gain": None if ig is None or not np.isfinite(ig) else float(ig),
                    }
                )
        summary = stack_votes(today)
        from eqpred.eval.waiting import swarm_mornings

        swarm_flags = swarm_mornings(catalog, REGIONS[region_id], mag, days)
        p_series = np.clip(climate_series + added, 1e-6, 1.0 - 1e-6)
        p_series = np.where(swarm_flags, climate_series, p_series)
        p_series = np.clip(np.where(np.isfinite(p_series), p_series, climate_series), 1e-6, 1.0 - 1e-6)
        swarm_today = bool(swarm_flags[latest])
        climate = float(climate_series[latest]) if np.isfinite(climate_series[latest]) else float("nan")
        p = float(p_series[latest])
        exam = skill_beats_climate(p_series[hold_mask], hits[hold_mask], roll[hold_mask])
        ig = exam.get("information_gain")
        ig_out = None if ig is None or not np.isfinite(float(ig)) else float(ig)
        up = [item for item in reports if item["points"] > 0]
        down = [item for item in reports if item["points"] < 0]
        up.sort(key=lambda item: -item["points"])
        down.sort(key=lambda item: item["points"])
        rows.append(
            {
                "region": region_id,
                "label": _place(region_id),
                "magnitude_min": mag,
                "horizon_days": horizon,
                "p": p,
                "chance_plain": _chance(p),
                "climatology_p": None if not np.isfinite(climate) else climate,
                "climatology_chance_plain": _chance(climate),
                "issue_day": days[latest].date().isoformat(),
                "information_gain": ig_out,
                "beats_climatology": ig_out is not None and ig_out >= 0.01,
                "writes_issued_p": False,
                "beside_30_day": True,
                "issued_from": "votes",
                "vote_cap": WATCH_CAP,
                "lineup_plain": (
                    f"{len(summary['up'])} point up, {len(summary['down'])} point down, "
                    f"{summary['n_blank']} had no reading."
                ),
                "votes_up": up[:6],
                "votes_down": down[:6],
                "swarm": swarm_today,
                "swarm_plain": (
                    "Aftershocks of a shock already reported. The votes are listed and do not move this chance."
                    if swarm_today
                    else None
                ),
            }
        )
        boxes[region_id] = {
            "signals": signals,
            "information_gain": ig_out,
            "magnitude_min": mag,
            "label": _place(region_id),
        }
        print(
            f"3-day votes {region_id} {_chance(p)} typical {_chance(climate)} {rows[-1]['lineup_plain']}",
            flush=True,
        )
    return {"rows": rows, "boxes": boxes, "horizon_days": horizon}
