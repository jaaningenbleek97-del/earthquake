from __future__ import annotations

"""3-day card: each signal adds a few probability points. The 30-day card is untouched."""

from typing import Any

import warnings

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
# 3-day cards only. Earth-wide stays at magnitude 7. Hikurangi and Alpine
# step to 4.5, the bottom of the catalog. Kermadec steps to 5.
THREE_DAY_MAG = {
    "hikurangi": 4.5,
    "alpine": 4.5,
    "kermadec": 5.0,
}


def three_day_targets() -> tuple[tuple[str, float], ...]:
    """The 3-day question. Same places, with the lower magnitude floors."""
    from eqpred.eval.mainshock import ISSUED_TARGETS

    return tuple((rid, three_day_magnitude(rid, mag)) for rid, mag in ISSUED_TARGETS)


def three_day_magnitude(region_id: str, issued_mag: float) -> float:
    """Magnitude floor for the open 3-day card. The 30-day card keeps issued_mag."""
    return float(THREE_DAY_MAG.get(str(region_id), issued_mag))


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
    with warnings.catch_warnings():
        # A morning where every column is blank is expected; it stays blank.
        warnings.simplefilter("ignore", category=RuntimeWarning)
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


def _raw_scale(z: np.ndarray, hits: np.ndarray, roll: np.ndarray, fit_mask: np.ndarray) -> float:
    """Uncapped points per standard score, from the early years only. No shrink."""
    ok = np.asarray(fit_mask, dtype=bool) & np.isfinite(z) & np.isfinite(hits) & np.isfinite(roll)
    n = int(ok.sum())
    if n < 20:
        return 0.0
    resid = hits[ok] - roll[ok]
    zz = z[ok] - float(np.mean(z[ok]))
    var = float(np.mean(zz * zz))
    if var < 1e-12:
        return 0.0
    scale = float(np.mean(zz * resid) / var)
    return scale if np.isfinite(scale) else 0.0


def _motion_base(name: str) -> bool:
    base = str(name).split("__lag", 1)[0]
    return "_extra_" in base or "_wide_" in base or base.startswith("geonet_")


def _measurement_group(name: str) -> str:
    """Copies of one physical measurement. They share one ceiling."""
    from eqpred.eval.hunt import driver_family

    base = str(name).split("__lag", 1)[0]
    if _motion_base(base):
        return "extra GPS motion"
    low = base.lower()
    if low.startswith(("goes_proton", "proton_")):
        return "solar protons"
    if low.startswith(("cmt_", "geonet_cmt_")):
        return "focal mechanisms"
    if low.startswith(("sws_", "swsdb_")):
        return "mantle splitting"
    if low.startswith("n_iface_"):
        return "interface quakes"
    if low.startswith("n_slab_"):
        return "slab quakes"
    if low.startswith("nied_"):
        return "Japan tremor"
    if low.startswith("obs_"):
        return "offshore campaign"
    if low.startswith("rupture_"):
        return "rupture models"
    if low.startswith("neutron"):
        return "neutrons"
    if low.startswith("grace_"):
        return "GRACE water"
    if low.startswith(("so2_", "volcano_")):
        return "volcano gas"
    if low.startswith("planet_"):
        return "planets"
    if low.startswith(("lunar_", "tidal_", "near_new")):
        return "tides"
    family = driver_family(low)
    if family != "wave2":
        return family
    head = low.split("_", 1)[0]
    return {
        "tec": "ionosphere TEC",
        "gnss": "GPS positions",
        "gnssa": "seafloor GPS",
        "hydro": "soil water",
        "smap": "soil moisture",
        "mag": "magnetometer",
        "ulf": "magnetic pulsations",
        "strain": "strain",
        "era5": "weather",
        "well": "wells",
        "radon": "radon",
        "tide": "ocean tide",
        "cses": "satellite electric field",
        "roti": "ionosphere ROTI",
        "scint": "ionosphere scintillation",
        "dvv": "seismic velocity",
        "wspr": "radio spots",
        "dart": "ocean pressure",
        "swarm": "Swarm satellites",
        "insar": "InSAR",
    }.get(head, head)


# A solar-watch group may change the usual rate by at most half.
# The whole pile may rise by one and a half times the usual rate, and may not cut it by more than half.
SOLAR_RATE_SHARE = 0.5
PILE_UP = 1.5
PILE_DOWN = 0.5


def group_rate_cap(base_cap: float, climate: np.ndarray) -> np.ndarray:
    """Point stop for one measurement, as a fraction of this fault's usual rate."""
    rate = np.asarray(climate, dtype=float)
    rate = np.where(np.isfinite(rate), rate, 0.0)
    rate = np.clip(rate, 0.0, 1.0)
    share = SOLAR_RATE_SHARE * (float(base_cap) / WATCH_CAP)
    return share * rate


def pile_factor(added: np.ndarray, climate: np.ndarray) -> np.ndarray:
    """Keep the chance between half and two and a half times the usual rate."""
    rate = np.asarray(climate, dtype=float)
    rate = np.where(np.isfinite(rate), rate, 0.0)
    total = np.asarray(added, dtype=float)
    lo = -PILE_DOWN * rate
    hi = PILE_UP * rate
    factor = np.ones(total.shape, dtype=float)
    up = total > hi
    down = total < lo
    factor[up] = np.divide(hi[up], total[up], out=np.ones(int(up.sum())), where=np.abs(total[up]) > 1e-15)
    factor[down] = np.divide(lo[down], total[down], out=np.ones(int(down.sum())), where=np.abs(total[down]) > 1e-15)
    return factor


def _share_ceiling(votes: list[np.ndarray], cap: float | np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Shrink a measurement's readings so their sum stops at `cap`. Each keeps its share."""
    if not votes:
        return [], np.zeros(0)
    filled = [np.where(np.isfinite(vote), vote, 0.0) for vote in votes]
    total = np.sum(filled, axis=0)
    cap_arr = np.asarray(cap, dtype=float)
    if cap_arr.ndim == 0:
        cap_arr = np.full(total.shape, float(cap_arr))
    factor = np.ones(total.shape, dtype=float)
    over = np.abs(total) > cap_arr
    factor[over] = cap_arr[over] / np.abs(total[over])
    return [vote * factor for vote in filled], total * factor


def _reading_groups(names: list[str]) -> list[tuple[str, list[int]]]:
    """One group per measurement. Lag copies of the same column stay together."""
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(name.split("__lag", 1)[0], []).append(idx)
    return list(groups.items())


def open_three_day_rows(
    *,
    design: tuple[np.ndarray, list[str]],
    catalog: pd.DataFrame,
    days: pd.DatetimeIndex,
    train_mask: np.ndarray,
    hold_mask: np.ndarray,
    horizon: int = 3,
    shared_base: dict[str, np.ndarray] | None = None,
    series_out: dict[str, dict[str, Any]] | None = None,
    targets: tuple[tuple[str, float], ...] | None = None,
) -> list[dict[str, Any]]:
    """Shared-model chance, then every reading adds on top. Each vote has a ceiling."""
    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.honest import fit_val_split
    from eqpred.eval.hunt import driver_family
    from eqpred.eval.mainshock import (
        ISSUED_TARGETS,
        _chance,
        _place,
        recent_climate,
        rolling_climate_series,
    )
    from eqpred.eval.skill import skill_beats_climate
    from eqpred.regions import REGIONS

    raw_x, names = design
    fit0, _val0 = fit_val_split(days, train_mask, purge=horizon)
    latest = len(days) - 1
    groups = _reading_groups(names)
    rows: list[dict[str, Any]] = []
    pairs = (
        targets
        if targets is not None
        else tuple((rid, three_day_magnitude(rid, mag)) for rid, mag in ISSUED_TARGETS)
    )
    for region_id, mag in pairs:
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, horizon).astype(float)
        roll = rolling_climate_series(hits, days, horizon=horizon)
        climate_series = np.where(np.isfinite(roll), roll, recent_climate(hits, days, train_mask))
        added = np.zeros(len(days), dtype=float)
        reports: list[dict[str, Any]] = []
        grouped: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
        n_live = 0
        n_blank_today = 0
        for base, idxs in groups:
            block = raw_x[:, idxs]
            stats = _column_stats(block, fit0)
            if all(stat is None for stat in stats):
                continue
            z = _z_from_stats(block, stats)
            scale = _scale(z, hits, roll, fit0)
            if scale == 0.0:
                continue
            n_live += 1
            vote = scale * z
            if not np.isfinite(z[latest]):
                n_blank_today += 1
            grouped.setdefault(_measurement_group(base), []).append((base, vote, np.isfinite(z)))
        group_rows: list[dict[str, Any]] = []
        held: list[tuple[str, list, list, np.ndarray]] = []
        for label, members in grouped.items():
            cap = group_rate_cap(vote_cap(driver_family(members[0][0])), climate_series)
            shared, group_total = _share_ceiling([vote for _base, vote, _known in members], cap)
            held.append((label, members, shared, group_total))
            added = added + group_total
        shrink = pile_factor(added, climate_series)
        added = added * shrink
        for label, members, shared, group_total in held:
            today_group = float(group_total[latest] * shrink[latest])
            if abs(today_group) >= 0.0005:
                group_rows.append(
                    {
                        "family": label,
                        "points": today_group,
                        "points_plain": _points(today_group),
                    }
                )
            for (base, vote, known), scaled in zip(members, shared, strict=True):
                if not known[latest]:
                    continue
                today_vote = float(scaled[latest] * shrink[latest])
                if abs(today_vote) < NOTICE:
                    continue
                reports.append(
                    {
                        "family": base.replace("_", " "),
                        "points": today_vote,
                        "points_plain": _points(today_vote),
                    }
                )
        start = climate_series
        model = None if shared_base is None else shared_base.get(region_id)
        if model is not None and len(np.asarray(model)) == len(days):
            model = np.asarray(model, dtype=float)
            start = np.where(np.isfinite(model), model, climate_series)
        p_series = np.clip(start + added, 1e-6, 1.0 - 1e-6)
        p_series = np.clip(np.where(np.isfinite(p_series), p_series, start), 1e-6, 1.0 - 1e-6)
        if series_out is not None:
            series_out[region_id] = {
                "p": p_series,
                "base": np.asarray(start, dtype=float),
                "groups": {label: np.asarray(total, dtype=float) for label, _members, _shared, total in held},
            }
        climate = float(climate_series[latest]) if np.isfinite(climate_series[latest]) else float("nan")
        base_p = float(start[latest]) if np.isfinite(start[latest]) else float("nan")
        p = float(p_series[latest])
        exam = skill_beats_climate(p_series[hold_mask], hits[hold_mask], roll[hold_mask])
        ig = exam.get("information_gain")
        ig_out = None if ig is None or not np.isfinite(float(ig)) else float(ig)
        up = [item for item in reports if item["points"] > 0]
        down = [item for item in reports if item["points"] < 0]
        up.sort(key=lambda item: -item["points"])
        down.sort(key=lambda item: item["points"])
        group_rows.sort(key=lambda item: -abs(item["points"]))
        group_up = [item for item in group_rows if item["points"] > 0]
        group_down = [item for item in group_rows if item["points"] < 0]
        group_bits = [f"{item['family']} {item['points_plain']}" for item in group_rows[:8]]
        rows.append(
            {
                "region": region_id,
                "label": _place(region_id),
                "magnitude_min": mag,
                "horizon_days": horizon,
                "p": p,
                "chance_plain": _chance(p),
                "base_p": None if not np.isfinite(base_p) else base_p,
                "base_chance_plain": _chance(base_p),
                "climatology_p": None if not np.isfinite(climate) else climate,
                "climatology_chance_plain": _chance(climate),
                "issue_day": days[latest].date().isoformat(),
                "information_gain": ig_out,
                "beats_climatology": ig_out is not None and ig_out >= 0.01,
                "writes_issued_p": False,
                "beside_30_day": True,
                "issued_from": "open",
                "n_signals": n_live,
                "lineup_plain": (
                    f"{len(group_up)} measurements point up, {len(group_down)} point down, "
                    f"{n_blank_today} readings had no value today. {n_live} readings have a value."
                ),
                "votes_up": up[:8],
                "votes_down": down[:8],
                "groups_up": group_up[:8],
                "groups_down": group_down[:8],
                "swarm": False,
                "swarm_plain": None,
                "group_plain": f"Added together: {', '.join(group_bits)}." if group_bits else None,
            }
        )
        print(
            f"3-day open {region_id} M{mag:g} {_chance(p)} model {_chance(base_p)} typical {_chance(climate)} {rows[-1]['lineup_plain']}",
            flush=True,
        )
    return rows


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
