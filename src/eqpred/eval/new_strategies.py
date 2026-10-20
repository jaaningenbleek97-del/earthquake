"""Scoring rules that are not a rescaling of today's model score.

Every rate and threshold comes from the fit or the check years. A morning
only sees earlier scores, and a hit only after its window has closed.
"""

from __future__ import annotations

import numpy as np

from eqpred.eval.strategy_desk import _gain, apply_alarm, clip, shift_score

SHARE = 0.2
# Neighbor used as the alarm, and how many days earlier that neighbor is read.
CROSS_LEAD = {
    "hikurangi": ("kermadec", 7),
    "kermadec": ("global", 0),
    "alpine": ("hikurangi", 14),
}


def _threshold(score: np.ndarray, fit: np.ndarray, share: float) -> float | None:
    src = np.asarray(score, dtype=float)[np.asarray(fit, dtype=bool)]
    src = src[np.isfinite(src)]
    if src.size < 40:
        return None
    share = min(0.9, max(0.05, float(share)))
    return float(np.quantile(src, 1.0 - share))


def _mean(mask: np.ndarray, hits: np.ndarray) -> float | None:
    m = np.asarray(mask, dtype=bool) & np.isfinite(hits)
    if int(m.sum()) < 8:
        return None
    return float(np.mean(hits[m]))


def _trailing(values: np.ndarray, window: int, reduce) -> np.ndarray:
    """Reduce the previous `window` mornings, not including today."""
    prev = np.concatenate([np.full(window, np.nan), np.asarray(values, dtype=float)[:-1]])
    view = np.lib.stride_tricks.sliding_window_view(prev, window)
    ok = np.isfinite(view).any(axis=1)
    out = np.full(view.shape[0], np.nan)
    if ok.any():
        filled = np.where(np.isfinite(view[ok]), view[ok], np.nan)
        with np.errstate(all="ignore"):
            out[ok] = reduce(filled, axis=1)
    return out


def apply_crest(model, hits, climate, fit, val, share: float = SHARE) -> np.ndarray:
    """Alarm on a crest: higher than yesterday and higher than the last week."""
    out = np.asarray(climate, dtype=float).copy()
    thr = _threshold(model, fit, share)
    if thr is None:
        return clip(out)
    previous = shift_score(model, 1)
    week = _trailing(model, 7, np.nanmax)
    crest = np.isfinite(model) & np.isfinite(previous) & (model > previous) & (model >= week) & (model >= thr)
    loud = _mean(np.asarray(val, dtype=bool) & crest, hits)
    quiet = _mean(np.asarray(val, dtype=bool) & ~crest, hits)
    if loud is None or quiet is None or loud <= quiet:
        return clip(out)
    out[crest] = loud
    return clip(out)


def apply_two_sided(model, hits, climate, fit, val, share: float = SHARE) -> np.ndarray:
    """Top tail and bottom tail each get the hit rate they had on check years."""
    out = np.asarray(climate, dtype=float).copy()
    src = np.asarray(model, dtype=float)[np.asarray(fit, dtype=bool)]
    src = src[np.isfinite(src)]
    if src.size < 40:
        return clip(out)
    share = min(0.4, max(0.05, float(share)))
    hi = float(np.quantile(src, 1.0 - share))
    lo = float(np.quantile(src, share))
    finite = np.isfinite(model)
    for choose in (finite & (model >= hi), finite & (model <= lo)):
        rate = _mean(np.asarray(val, dtype=bool) & choose, hits)
        if rate is not None:
            out[choose] = rate
    return clip(out)


def _above(model, fit, share: float) -> np.ndarray | None:
    thr = _threshold(model, fit, share)
    if thr is None:
        return None
    return np.isfinite(model) & (model >= thr)


def apply_crossing(model, hits, climate, fit, val, share: float = SHARE) -> np.ndarray:
    """The first morning a score crosses the loud line, not every loud morning."""
    out = np.asarray(climate, dtype=float).copy()
    above = _above(model, fit, share)
    if above is None:
        return clip(out)
    yesterday = np.concatenate([[False], above[:-1]])
    cross = above & ~yesterday
    loud = _mean(np.asarray(val, dtype=bool) & cross, hits)
    base = _mean(np.asarray(val, dtype=bool), hits)
    if loud is None or base is None or loud <= base:
        return clip(out)
    out[cross] = loud
    return clip(out)


def apply_run(model, hits, climate, fit, val, share: float = SHARE, days: int = 3) -> np.ndarray:
    """Alarm only after the score has stayed loud for several mornings."""
    out = np.asarray(climate, dtype=float).copy()
    above = _above(model, fit, share)
    if above is None:
        return clip(out)
    run = above.copy()
    for lag in range(1, int(days)):
        run = run & np.concatenate([np.zeros(lag, dtype=bool), above[:-lag]])
    loud = _mean(np.asarray(val, dtype=bool) & run, hits)
    base = _mean(np.asarray(val, dtype=bool), hits)
    if loud is None or base is None or loud <= base:
        return clip(out)
    out[run] = loud
    return clip(out)


def apply_quiet_catalog(groups, hits, climate, fit, val, share: float = 0.15) -> np.ndarray:
    """A very quiet catalog, if check years say that quiet came before shocks."""
    out = np.asarray(climate, dtype=float).copy()
    group = groups.get("fresh_catalog")
    if group is None:
        return clip(out)
    src = np.asarray(group, dtype=float)[np.asarray(fit, dtype=bool)]
    src = src[np.isfinite(src)]
    if src.size < 40:
        return clip(out)
    line = float(np.quantile(src, share))
    quiet = np.isfinite(group) & (group <= line)
    loud = _mean(np.asarray(val, dtype=bool) & quiet, hits)
    base = _mean(np.asarray(val, dtype=bool), hits)
    if loud is None or base is None or loud <= base:
        return clip(out)
    out[quiet] = loud
    return clip(out)


def apply_cross_lead(region, peers, hits, climate, fit, val, share: float = SHARE) -> np.ndarray:
    """Read a physically related place, then alarm on that other score."""
    if region == "global":
        parts = [peers.get(name) for name in ("hikurangi", "alpine", "kermadec")]
        parts = [np.asarray(part, dtype=float) for part in parts if part is not None]
        if not parts:
            return clip(climate)
        foreign = np.nanmax(np.vstack(parts), axis=0)
    else:
        spec = CROSS_LEAD.get(region)
        if spec is None or peers.get(spec[0]) is None:
            return clip(climate)
        foreign = shift_score(peers[spec[0]], spec[1])
    return apply_alarm(foreign, hits, climate, fit, val, share)


def apply_jump(model, hits, climate, fit, val, share: float = SHARE, window: int = 14) -> np.ndarray:
    """Alarm when the score jumps above its own last two weeks."""
    out = np.asarray(climate, dtype=float).copy()
    base = _trailing(model, window, np.nanmedian)
    jump = np.asarray(model, dtype=float) - base
    thr = _threshold(jump, fit, share)
    if thr is None:
        return clip(out)
    choose = np.isfinite(jump) & (jump >= thr)
    loud = _mean(np.asarray(val, dtype=bool) & choose, hits)
    quiet = _mean(np.asarray(val, dtype=bool) & ~choose, hits)
    if loud is None or quiet is None or loud <= quiet:
        return clip(out)
    out[choose] = loud
    return clip(out)


def apply_month_rate(hits, climate, fit, month) -> np.ndarray:
    """Replace the model with the hit rate of that calendar month in the fit."""
    out = np.asarray(climate, dtype=float).copy()
    month = np.asarray(month)
    fit = np.asarray(fit, dtype=bool)
    for value in range(1, 13):
        rate = _mean(fit & (month == value), hits)
        if rate is not None:
            out[month == value] = rate
    return clip(out)


def apply_equity(model, hits, climate, horizon: int, lookback: int = 365) -> np.ndarray:
    """Publish the model only while its own closed windows are beating the usual rate."""
    p = clip(model)
    c = clip(climate)
    y = np.asarray(hits, dtype=float)
    ok = np.isfinite(p) & np.isfinite(c) & np.isfinite(y)
    adv = np.zeros(len(y))
    adv[ok] = (y[ok] * np.log(p[ok]) + (1.0 - y[ok]) * np.log(1.0 - p[ok])) - (
        y[ok] * np.log(c[ok]) + (1.0 - y[ok]) * np.log(1.0 - c[ok])
    )
    cs = np.cumsum(adv)
    n = len(y)
    end = np.arange(n) - int(horizon)
    left = end - int(lookback)
    total = np.full(n, np.nan)
    valid = (end >= 0) & (end < n) & (left >= 0)
    total[valid] = cs[end[valid]] - cs[left[valid]]
    out = np.asarray(climate, dtype=float).copy()
    use = valid & (total > 0) & np.isfinite(model)
    out[use] = model[use]
    return clip(out)


def apply_after_quiet(model, hits, climate, fit, val, horizon: int, share: float = SHARE) -> np.ndarray:
    """Alarm only when the previous window closed empty and the score is loud."""
    out = np.asarray(climate, dtype=float).copy()
    above = _above(model, fit, share)
    if above is None:
        return clip(out)
    closed = shift_score(hits, horizon)
    choose = above & np.isfinite(closed) & (closed == 0)
    loud = _mean(np.asarray(val, dtype=bool) & choose, hits)
    base = _mean(np.asarray(val, dtype=bool), hits)
    if loud is None or base is None or loud <= base:
        return clip(out)
    out[choose] = loud
    return clip(out)


def apply_tide_gate(model, groups, hits, climate, fit, val, share: float = SHARE) -> np.ndarray:
    """A loud score counts only when the tide group is also pointing up."""
    out = np.asarray(climate, dtype=float).copy()
    tides = groups.get("tides")
    above = _above(model, fit, share)
    if tides is None or above is None:
        return clip(out)
    choose = above & np.isfinite(tides) & (tides > 0)
    loud = _mean(np.asarray(val, dtype=bool) & choose, hits)
    base = _mean(np.asarray(val, dtype=bool), hits)
    if loud is None or base is None or loud <= base:
        return clip(out)
    out[choose] = loud
    return clip(out)


def apply_pair(groups, hits, climate, val) -> np.ndarray:
    """Move only when interface quakes and slab quakes point the same way."""
    out = np.asarray(climate, dtype=float).copy()
    left = groups.get("interface quakes")
    right = groups.get("slab quakes")
    if left is None or right is None:
        return clip(out)
    step = np.zeros(len(out))
    up = (left > 0.005) & (right > 0.005)
    down = (left < -0.005) & (right < -0.005)
    step[up] = left[up] + right[up]
    step[down] = left[down] + right[down]
    limit = 0.5 * np.asarray(climate, dtype=float)
    pred = clip(out + np.clip(step, -limit, limit))
    from eqpred.eval.strategy_desk import _gain

    if (_gain(pred, hits, climate, val) or -1) <= 0:
        return clip(out)
    return pred


def apply_cost(model, hits, climate, val, miss_weight: float = 3.0) -> np.ndarray:
    """Pick a small shift on check years, counting a missed quake triple."""
    y = np.asarray(hits, dtype=float)
    mask = np.asarray(val, dtype=bool) & np.isfinite(model) & np.isfinite(y) & np.isfinite(climate)
    if int(mask.sum()) < 8:
        return clip(model)
    yy = y[mask]
    weight = np.where(yy >= 0.5, float(miss_weight), 1.0)

    def score(shift: float) -> float:
        guess = clip(np.asarray(model, dtype=float)[mask] + shift)
        base = clip(np.asarray(climate, dtype=float)[mask])
        model_ll = np.sum(weight * (yy * np.log(guess) + (1.0 - yy) * np.log(1.0 - guess)))
        base_ll = np.sum(weight * (yy * np.log(base) + (1.0 - yy) * np.log(1.0 - base)))
        return float(model_ll - base_ll)

    best_shift, best = 0.0, score(0.0)
    for shift in (-0.05, -0.02, 0.02, 0.05, 0.1):
        value = score(shift)
        if value > best:
            best_shift, best = shift, value
    return clip(np.asarray(model, dtype=float) + best_shift)


def apply_fade(model, hits, climate, fit, val, share: float = SHARE) -> np.ndarray:
    """If check years say a hot score was followed by fewer quakes, publish that lower rate."""
    out = np.asarray(climate, dtype=float).copy()
    above = _above(model, fit, share)
    if above is None:
        return clip(out)
    hot = _mean(np.asarray(val, dtype=bool) & above, hits)
    cool = _mean(np.asarray(val, dtype=bool) & ~above, hits)
    if hot is None or cool is None or hot >= cool:
        return clip(out)
    out[above] = hot
    return clip(out)


def apply_hold(model, hits, climate, fit, val, horizon: int, share: float = SHARE) -> np.ndarray:
    """A crossing opens a window and that chance stays up until the window closes."""
    out = np.asarray(climate, dtype=float).copy()
    above = _above(model, fit, share)
    if above is None:
        return clip(out)
    yesterday = np.concatenate([[False], above[:-1]])
    cross = above & ~yesterday
    loud = _mean(np.asarray(val, dtype=bool) & cross, hits)
    base = _mean(np.asarray(val, dtype=bool), hits)
    if loud is None or base is None or loud <= base:
        return clip(out)
    until = -1
    for day, opened in enumerate(cross):
        if day <= until:
            out[day] = loud
        elif opened:
            out[day] = loud
            until = day + int(horizon) - 1
    return clip(out)


def _keep_if_check_years_agree(pred, hits, climate, val) -> np.ndarray:
    if (_gain(pred, hits, climate, val) or -1.0) <= 0:
        return clip(climate)
    return pred


def apply_skeptic(model, climate) -> np.ndarray:
    """Follow the model only when it is quieter than the usual rate."""
    out = np.asarray(climate, dtype=float).copy()
    below = np.isfinite(model) & (np.asarray(model, dtype=float) < out)
    out[below] = np.asarray(model, dtype=float)[below]
    return clip(out)


def apply_believer(model, climate) -> np.ndarray:
    """Follow the model only when it is louder than the usual rate."""
    out = np.asarray(climate, dtype=float).copy()
    above = np.isfinite(model) & (np.asarray(model, dtype=float) > out)
    out[above] = np.asarray(model, dtype=float)[above]
    return clip(out)


def apply_residual_alarm(model, hits, climate, fit, val) -> np.ndarray:
    """Alarm on the gap above today's usual rate, with the share chosen on check years."""
    gap = np.asarray(model, dtype=float) - np.asarray(climate, dtype=float)
    best, score = clip(climate), 0.0
    for share in (0.1, 0.2, 0.3):
        pred = apply_alarm(gap, hits, climate, fit, val, share)
        gain = _gain(pred, hits, climate, val) or -1.0
        if gain > score:
            best, score = pred, gain
    return best


def _trailing_rank(values: np.ndarray, window: int) -> np.ndarray:
    prev = np.concatenate([np.full(window, np.nan), np.asarray(values, dtype=float)[:-1]])
    view = np.lib.stride_tricks.sliding_window_view(prev, window)
    finite = np.isfinite(view)
    below = (view < np.asarray(values, dtype=float)[:, None]) & finite
    count = finite.sum(axis=1)
    rank = np.full(len(values), np.nan)
    ok = count >= max(10, window // 5)
    rank[ok] = below.sum(axis=1)[ok] / count[ok]
    return rank


def apply_trailing_rank(model, hits, climate, fit, val) -> np.ndarray:
    """Alarm when today is in the top fifth of its own recent scores. Window chosen on check years."""
    best, score = clip(climate), 0.0
    base = _mean(np.asarray(val, dtype=bool), hits)
    for window in (30, 90, 180):
        rank = _trailing_rank(model, window)
        choose = np.isfinite(rank) & (rank >= 0.8)
        loud = _mean(np.asarray(val, dtype=bool) & choose, hits)
        pred = np.asarray(climate, dtype=float).copy()
        if loud is not None and base is not None and loud > base:
            pred[choose] = loud
        gain = _gain(pred, hits, climate, val) or -1.0
        if gain > score:
            best, score = clip(pred), gain
    return best


def apply_group_spread(groups, hits, climate, fit, val) -> np.ndarray:
    """Treat disagreement among the groups as the signal. Direction chosen on check years."""
    if not groups:
        return clip(climate)
    stack = np.vstack([np.asarray(group, dtype=float) for group in groups.values()])
    with np.errstate(all="ignore"):
        spread = np.nanstd(stack, axis=0)
    best, score = clip(climate), 0.0
    for series in (spread, -spread):
        pred = apply_alarm(series, hits, climate, fit, val, SHARE)
        gain = _gain(pred, hits, climate, val) or -1.0
        if gain > score:
            best, score = pred, gain
    return best


def apply_pattern(groups, hits, climate, val) -> np.ndarray:
    """Eight leaves from the signs of interface, slab, and the fresh catalog."""
    names = ("interface quakes", "slab quakes", "fresh_catalog")
    cols = [groups.get(name) for name in names]
    if any(col is None for col in cols):
        return clip(climate)
    code = np.zeros(len(climate), dtype=int)
    for bit, col in enumerate(cols):
        code += (np.asarray(col, dtype=float) > 0).astype(int) << bit
    out = np.asarray(climate, dtype=float).copy()
    val = np.asarray(val, dtype=bool)
    for leaf in range(8):
        choose = code == leaf
        seen = val & choose & np.isfinite(hits)
        if int(seen.sum()) < 20:
            continue
        out[choose] = float(hits[seen].mean())
    return _keep_if_check_years_agree(clip(out), hits, climate, val)


def apply_one_group(groups, hits, climate, val) -> np.ndarray:
    """Keep only the single group that helped on the check years."""
    best, score = clip(climate), 0.0
    base = np.asarray(climate, dtype=float)
    for group in groups.values():
        pred = clip(base + np.asarray(group, dtype=float))
        gain = _gain(pred, hits, climate, val) or -1.0
        if gain > score:
            best, score = pred, gain
    return best


def apply_veto(model, groups, climate) -> np.ndarray:
    """A strongly downward group cancels the model and returns the usual rate."""
    out = np.asarray(model, dtype=float).copy()
    if not groups:
        return clip(out)
    stack = np.vstack([np.asarray(group, dtype=float) for group in groups.values()])
    with np.errstate(all="ignore"):
        worst = np.nanmin(stack, axis=0)
    cancel = np.isfinite(worst) & (worst < -0.01)
    out[cancel] = np.asarray(climate, dtype=float)[cancel]
    return clip(out)


def apply_compound(short, climate, horizon: int, same_mag: bool) -> np.ndarray:
    """Build the longer chance from the one-day chance when they ask the same question."""
    if not same_mag or short is None:
        return clip(climate)
    daily = clip(short)
    return clip(1.0 - np.power(1.0 - daily, max(int(horizon), 1)))


def apply_cross_horizon(model, short, hits, climate, fit, val, same_mag: bool) -> np.ndarray:
    """Use the long model only when the shorter model, asking the same question, is also loud."""
    if not same_mag or short is None:
        return clip(climate)
    thr = _threshold(short, fit, SHARE)
    if thr is None:
        return clip(climate)
    pred = np.asarray(climate, dtype=float).copy()
    loud = np.isfinite(short) & (short >= thr) & np.isfinite(model)
    pred[loud] = np.asarray(model, dtype=float)[loud]
    return _keep_if_check_years_agree(clip(pred), hits, climate, val)


def apply_markov(hits, climate, fit, horizon: int) -> np.ndarray:
    """Two rates: the previous window hit, or it did not. Rates come from the fit only."""
    prev = shift_score(hits, horizon)
    out = np.asarray(climate, dtype=float).copy()
    fit = np.asarray(fit, dtype=bool)
    for choose in (np.isfinite(prev) & (prev >= 0.5), np.isfinite(prev) & (prev < 0.5)):
        rate = _mean(fit & choose, hits)
        if rate is not None:
            out[choose] = rate
    return clip(out)


def apply_solar_phase(hits, climate, fit, year_index) -> np.ndarray:
    """Four seasons of the 11-year solar cycle, counted on the fit years."""
    phase = np.floor((np.asarray(year_index, dtype=float) % 11.0) / 2.75).astype(int)
    phase = np.clip(phase, 0, 3)
    out = np.asarray(climate, dtype=float).copy()
    fit = np.asarray(fit, dtype=bool)
    for leaf in range(4):
        rate = _mean(fit & (phase == leaf), hits)
        if rate is not None:
            out[phase == leaf] = rate
    return clip(out)


def apply_contrarian(model, hits, climate, horizon: int, lookback: int = 365) -> np.ndarray:
    """Publish the model only after a year in which it was losing to the usual rate."""
    p = clip(model)
    c = clip(climate)
    y = np.asarray(hits, dtype=float)
    ok = np.isfinite(p) & np.isfinite(c) & np.isfinite(y)
    adv = np.zeros(len(y))
    adv[ok] = (y[ok] * np.log(p[ok]) + (1.0 - y[ok]) * np.log(1.0 - p[ok])) - (
        y[ok] * np.log(c[ok]) + (1.0 - y[ok]) * np.log(1.0 - c[ok])
    )
    cs = np.cumsum(adv)
    end = np.arange(len(y)) - int(horizon)
    left = end - int(lookback)
    total = np.full(len(y), np.nan)
    valid = (end >= 0) & (end < len(y)) & (left >= 0)
    total[valid] = cs[end[valid]] - cs[left[valid]]
    out = np.asarray(climate, dtype=float).copy()
    use = valid & (total < 0) & np.isfinite(model)
    out[use] = np.asarray(model, dtype=float)[use]
    return clip(out)


def apply_median_groups(groups, hits, climate, val) -> np.ndarray:
    """Add the middle group vote, so one wild group cannot drag the card."""
    if not groups:
        return clip(climate)
    stack = np.vstack([np.asarray(group, dtype=float) for group in groups.values()])
    with np.errstate(all="ignore"):
        middle = np.nanmedian(stack, axis=0)
    limit = 0.5 * np.asarray(climate, dtype=float)
    pred = clip(np.asarray(climate, dtype=float) + np.clip(middle, -limit, limit))
    return _keep_if_check_years_agree(pred, hits, climate, val)


def _fit_rates_by_code(code: np.ndarray, hits, climate, fit, leaves: int) -> np.ndarray:
    out = np.asarray(climate, dtype=float).copy()
    fit = np.asarray(fit, dtype=bool)
    for leaf in range(leaves):
        rate = _mean(fit & (code == leaf), hits)
        if rate is not None:
            out[code == leaf] = rate
    return clip(out)


def apply_lunar(hits, climate, fit, lunar) -> np.ndarray:
    """Eight phases of the 29.5-day lunar cycle, counted on the fit years."""
    return _fit_rates_by_code(np.asarray(lunar, dtype=int), hits, climate, fit, 8)


def apply_weekday(hits, climate, fit, dow) -> np.ndarray:
    """The hit rate of each weekday, counted on the fit years."""
    return _fit_rates_by_code(np.asarray(dow, dtype=int), hits, climate, fit, 7)


def apply_online_decile(model, hits, climate, horizon: int, lookback: int = 1825, bins: int = 5) -> np.ndarray:
    """Recent closed windows decide what each score level means. Today cannot see its own quake."""
    model = np.asarray(model, dtype=float)
    hits = np.asarray(hits, dtype=float)
    out = np.asarray(climate, dtype=float).copy()
    horizon = max(int(horizon), 1)
    step = horizon
    for start in range(lookback + horizon, len(model), step):
        end = min(start + step, len(model))
        stop = start - horizon
        begin = max(0, stop - lookback)
        past_m = model[begin:stop]
        past_y = hits[begin:stop]
        ok = np.isfinite(past_m) & np.isfinite(past_y)
        if int(ok.sum()) < bins * 20:
            continue
        edges = np.quantile(past_m[ok], np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        for i in range(bins):
            sel = ok & (past_m >= edges[i]) & (past_m < edges[i + 1] if i < bins - 1 else past_m <= edges[i + 1])
            if i == bins - 1:
                sel = ok & (past_m >= edges[i])
            if int(sel.sum()) < 8:
                continue
            rate = float(past_y[sel].mean())
            today = model[start:end]
            choose = np.isfinite(today) & (today >= edges[i])
            if i < bins - 1:
                choose &= today < edges[i + 1]
            out[start + np.flatnonzero(choose)] = rate
    return clip(out)


def apply_online_trust(model, hits, climate, horizon: int, lookback: int = 1825) -> np.ndarray:
    """Every quarter, trust the model only as far as the last closed years earned."""
    model = np.asarray(model, dtype=float)
    hits = np.asarray(hits, dtype=float)
    base = np.asarray(climate, dtype=float)
    out = base.copy()
    step = 90
    for start in range(lookback + int(horizon), len(model), step):
        end = min(start + step, len(model))
        stop = start - int(horizon)
        window = np.zeros(len(model), dtype=bool)
        window[max(0, stop - lookback) : stop] = True
        best_b, best = 0.0, 0.0
        for trust in (0.0, 0.25, 0.5, 0.75, 1.0):
            pred = clip(base + trust * (model - base))
            gain = _gain(pred, hits, base, window) or -1.0
            if gain > best:
                best_b, best = trust, gain
        out[start:end] = base[start:end] + best_b * (model[start:end] - base[start:end])
    return clip(out)


def apply_peer_outcome(region, peer_hits, hits, climate, fit, horizon: int) -> np.ndarray:
    """A neighbor's first shock, already finished, sets today's rate."""
    links = {
        "hikurangi": ("kermadec",),
        "kermadec": ("hikurangi",),
        "alpine": ("hikurangi",),
        "global": ("hikurangi", "alpine", "kermadec"),
    }
    names = links.get(region, ())
    series = [peer_hits.get(name) for name in names if peer_hits.get(name) is not None]
    if not series:
        return clip(climate)
    stacked = np.vstack([shift_score(item, horizon) for item in series])
    recent = np.nanmax(stacked, axis=0)
    out = np.asarray(climate, dtype=float).copy()
    fit = np.asarray(fit, dtype=bool)
    for choose in (np.isfinite(recent) & (recent >= 0.5), np.isfinite(recent) & (recent < 0.5)):
        rate = _mean(fit & choose, hits)
        if rate is not None:
            out[choose] = rate
    return clip(out)


def apply_stack_hot(stack, hits, climate, fit, val) -> np.ndarray:
    """Move only when at least three window lengths are loud together."""
    if not stack:
        return clip(climate)
    hot = np.zeros(len(climate), dtype=int)
    for series in stack:
        series = np.asarray(series, dtype=float)
        src = series[np.asarray(fit, dtype=bool)]
        src = src[np.isfinite(src)]
        if src.size < 40:
            continue
        line = float(np.median(src))
        hot += (np.isfinite(series) & (series >= line)).astype(int)
    pred = np.asarray(climate, dtype=float).copy()
    choose = hot >= 3
    pred[choose] = np.nan
    # Use this horizon's own model where the stack agrees; caller passes it as stack[0] is not the model.
    return choose


def apply_stack_hot_rate(model, stack, hits, climate, fit, val) -> np.ndarray:
    choose = apply_stack_hot(stack, hits, climate, fit, val)
    if isinstance(choose, np.ndarray) and choose.dtype != bool:
        return choose
    out = np.asarray(climate, dtype=float).copy()
    loud = _mean(np.asarray(val, dtype=bool) & choose, hits)
    base = _mean(np.asarray(val, dtype=bool), hits)
    if loud is None or base is None or loud <= base:
        return clip(out)
    use = choose & np.isfinite(model)
    out[use] = np.asarray(model, dtype=float)[use]
    return _keep_if_check_years_agree(clip(out), hits, climate, val)


def apply_deadzone(model, hits, climate, val) -> np.ndarray:
    """Ignore a score that sits close to the usual rate. The gap is chosen on check years."""
    model = np.asarray(model, dtype=float)
    base = np.asarray(climate, dtype=float)
    best, score = clip(base), 0.0
    for width in (0.01, 0.02, 0.05):
        pred = base.copy()
        far = np.isfinite(model) & (np.abs(model - base) >= width)
        pred[far] = model[far]
        gain = _gain(clip(pred), hits, base, val) or -1.0
        if gain > score:
            best, score = clip(pred), gain
    return best


def apply_climate_rise(climate, hits, fit, val, lag: int = 90) -> np.ndarray:
    """Alarm when the usual rate itself has been climbing."""
    rise = np.asarray(climate, dtype=float) - shift_score(climate, lag)
    return apply_alarm(rise, hits, climate, fit, val, SHARE)


def apply_group_tail(groups, name: str, hits, climate, fit, val, low: bool) -> np.ndarray:
    """One named measurement, pre-specified, in its extreme tail."""
    group = groups.get(name)
    if group is None:
        return clip(climate)
    series = -np.asarray(group, dtype=float) if low else np.asarray(group, dtype=float)
    return apply_alarm(series, hits, climate, fit, val, SHARE)


def apply_accel(model, hits, climate, fit, val) -> np.ndarray:
    """Alarm when the score is speeding up, not merely high."""
    first = np.asarray(model, dtype=float) - shift_score(model, 1)
    second = first - shift_score(first, 1)
    return apply_alarm(second, hits, climate, fit, val, SHARE)


def apply_contradict(model, groups, hits, climate, val) -> np.ndarray:
    """The model is loud while the group votes are quiet, or the reverse."""
    if not groups:
        return clip(climate)
    total = np.nansum(np.vstack([np.asarray(group, dtype=float) for group in groups.values()]), axis=0)
    model = np.asarray(model, dtype=float)
    base = np.asarray(climate, dtype=float)
    out = base.copy()
    val = np.asarray(val, dtype=bool)
    for choose in ((model > base) & (total < -0.01), (model < base) & (total > 0.01)):
        rate = _mean(val & choose & np.isfinite(model), hits)
        if rate is not None:
            out[choose] = rate
    return _keep_if_check_years_agree(clip(out), hits, climate, val)


def apply_peer_sum(region, peers, hits, climate, fit, val) -> np.ndarray:
    """Add the other places' scores, then alarm on that sum."""
    others = [np.asarray(series, dtype=float) for name, series in peers.items() if name != region]
    if not others:
        return clip(climate)
    total = np.nansum(np.vstack(others), axis=0)
    return apply_alarm(total, hits, climate, fit, val, SHARE)


def apply_stable_alarm(score, hits, climate, fit, years, share: float = SHARE) -> np.ndarray:
    """Alarm only when the loud tail beat the quiet tail in three separate old stretches."""
    out = np.asarray(climate, dtype=float).copy()
    thr = _threshold(score, fit, share)
    if thr is None:
        return clip(out)
    loud = np.isfinite(score) & (np.asarray(score, dtype=float) >= thr)
    eras = (
        np.asarray(years) < 1979,
        (np.asarray(years) >= 1979) & (np.asarray(years) < 1994),
        (np.asarray(years) >= 1994) & (np.asarray(years) < 2009),
    )
    fit = np.asarray(fit, dtype=bool)
    for era in eras:
        loud_rate = _mean(fit & era & loud, hits)
        quiet_rate = _mean(fit & era & ~loud & np.isfinite(score), hits)
        if loud_rate is None or quiet_rate is None or loud_rate <= quiet_rate:
            return clip(out)
    pooled = _mean(fit & loud, hits)
    if pooled is None:
        return clip(out)
    out[loud] = pooled
    return clip(out)


def _column_rate(group, hits, climate, early, choose_high: bool) -> tuple[np.ndarray, np.ndarray]:
    """Threshold and chosen days from the years before 1990."""
    src = np.asarray(group, dtype=float)[early]
    src = src[np.isfinite(src)]
    if src.size < 40:
        return np.zeros(len(group), dtype=bool), np.asarray(climate, dtype=float)
    line = float(np.quantile(src, 0.8 if choose_high else 0.2))
    series = np.asarray(group, dtype=float)
    choose = np.isfinite(series) & (series >= line if choose_high else series <= line)
    return choose, series


def apply_best_conjunction(groups, hits, climate, years) -> np.ndarray:
    """Pick one pair of extremes using 1990–2008, with the rates counted before 1990."""
    specs = (
        ("slowslip", True, "interface quakes", False),
        ("neutrons", False, "solar", True),
        ("spf", True, "solar", True),
        ("fresh_catalog", False, "migration", True),
        ("slab quakes", True, "interface quakes", False),
        ("extra GPS motion", True, "slowslip", True),
        ("climate", True, "neutrons", False),
        ("tides", True, "slowslip", True),
    )
    years = np.asarray(years)
    early = years < 1990
    select = (years >= 1990) & (years < 2009)
    best, score = clip(climate), 0.0
    base = np.asarray(climate, dtype=float)
    for left, left_high, right, right_high in specs:
        if groups.get(left) is None or groups.get(right) is None:
            continue
        a, _ = _column_rate(groups[left], hits, climate, early, left_high)
        b, _ = _column_rate(groups[right], hits, climate, early, right_high)
        choose = a & b
        rate = _mean(early & choose, hits)
        usual = _mean(early, hits)
        if rate is None or usual is None or rate <= usual:
            continue
        pred = base.copy()
        pred[choose] = rate
        gain = _gain(pred, hits, base, select) or -1.0
        if gain > score:
            best, score = clip(pred), gain
    return best


def apply_four_linear(groups, hits, climate, years) -> np.ndarray:
    """Four measurements, coefficients from before 1990, shrink from 1990–2008."""
    names = ("slowslip", "neutrons", "interface quakes", "solar")
    if any(groups.get(name) is None for name in names):
        return clip(climate)
    years = np.asarray(years)
    early = years < 1990
    select = (years >= 1990) & (years < 2009)
    cols = []
    for name in names:
        series = np.asarray(groups[name], dtype=float)
        train = series[early]
        train = train[np.isfinite(train)]
        if train.size < 40 or float(np.std(train)) < 1e-8:
            return clip(climate)
        cols.append((series - float(np.mean(train))) / float(np.std(train)))
    design = np.vstack(cols).T
    y = np.asarray(hits, dtype=float) - np.asarray(climate, dtype=float)
    ok = early & np.isfinite(y) & np.isfinite(design).all(axis=1)
    if int(ok.sum()) < 80:
        return clip(climate)
    beta, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)
    raw = np.asarray(climate, dtype=float) + design @ beta
    best, score = clip(climate), 0.0
    for trust in (0.0, 0.25, 0.5, 0.75, 1.0):
        pred = clip(np.asarray(climate, dtype=float) + trust * (raw - np.asarray(climate, dtype=float)))
        gain = _gain(pred, hits, climate, select) or -1.0
        if gain > score:
            best, score = pred, gain
    return best


def apply_quiet_decade(model, hits, climate, fit, val) -> np.ndarray:
    """A loud score during a stretch when the usual rate itself is low."""
    out = np.asarray(climate, dtype=float).copy()
    src = np.asarray(climate, dtype=float)[np.asarray(fit, dtype=bool)]
    src = src[np.isfinite(src)]
    thr = _threshold(model, fit, SHARE)
    if src.size < 40 or thr is None:
        return clip(out)
    floor = float(np.quantile(src, 0.3))
    choose = np.isfinite(model) & (model >= thr) & np.isfinite(climate) & (np.asarray(climate, dtype=float) <= floor)
    loud = _mean(np.asarray(val, dtype=bool) & choose, hits)
    base = _mean(np.asarray(val, dtype=bool), hits)
    if loud is None or base is None or loud <= base:
        return clip(out)
    out[choose] = loud
    return clip(out)


NEW_STRATEGIES = (
    ("crest", "Crest only"),
    ("two_sided", "Both tails"),
    ("crossing", "First crossing"),
    ("run3", "Three loud days"),
    ("quiet_catalog", "Quiet catalog"),
    ("cross_lead", "Neighbor fault leads"),
    ("jump", "Sudden jump"),
    ("month_rate", "Calendar month"),
    ("equity", "Only while skill is up"),
    ("after_quiet", "After a quiet window"),
    ("tide_gate", "Tide must agree"),
    ("pair", "Interface and slab"),
    ("cost3", "Misses count triple"),
    ("fade", "Fade a hot score"),
    ("hold", "Hold the window open"),
)

ROUND2 = (
    ("skeptic", "Only trust a quieter call"),
    ("believer", "Only trust a louder call"),
    ("residual_alarm", "Gap above the usual rate"),
    ("trailing_rank", "Hot versus its own month"),
    ("group_spread", "Groups disagree"),
    ("pattern8", "Eight sign patterns"),
    ("one_group", "One best group"),
    ("veto", "A down group cancels"),
    ("compound", "Stack the one-day chance"),
    ("cross_horizon", "Short window must agree"),
    ("markov", "Previous window"),
    ("solar_phase", "Solar-cycle season"),
    ("contrarian", "Trust it after a bad year"),
    ("median_groups", "Middle group vote"),
    ("quiet_decade", "Loud call in a quiet stretch"),
)

ROUND3 = (
    ("online_decile", "Recent score levels"),
    ("online_trust", "Trust earned last quarter"),
    ("lunar", "Lunar phase"),
    ("weekday", "Weekday"),
    ("peer_outcome", "Neighbor already shook"),
    ("stack_hot", "Three windows agree"),
    ("deadzone", "Ignore a near miss"),
    ("climate_rise", "Usual rate is climbing"),
    ("neutron_low", "Neutrons are low"),
    ("slowslip_high", "Slow slip is high"),
    ("accel", "Score is speeding up"),
    ("contradict", "Model and groups disagree"),
    ("peer_sum", "Sum of the other places"),
    ("spf_high", "Polar field is high"),
    ("solar_high", "Sun is loud"),
)

ROUND4 = (
    ("stable_alarm", "Alarm that held in three eras"),
    ("stable_slowslip", "Slow slip that held in three eras"),
    ("stable_neutrons", "Neutrons that held in three eras"),
    ("stable_interface", "Interface quakes that held in three eras"),
    ("stable_spf", "Polar field that held in three eras"),
    ("four_linear", "Four old measurements"),
    ("best_conjunction", "Pair that held from 1990 to 2008"),
)


def apply_new(strategy: str, card: dict, model: np.ndarray) -> np.ndarray | None:
    hits, climate, fit, val = card["y"], card["c"], card["fit"], card["val"]
    groups = card["groups"]
    horizon = int(card.get("horizon") or card.get("purge") or 30)
    if strategy == "crest":
        return apply_crest(model, hits, climate, fit, val)
    if strategy == "two_sided":
        return apply_two_sided(model, hits, climate, fit, val)
    if strategy == "crossing":
        return apply_crossing(model, hits, climate, fit, val)
    if strategy == "run3":
        return apply_run(model, hits, climate, fit, val)
    if strategy == "quiet_catalog":
        return apply_quiet_catalog(groups, hits, climate, fit, val)
    if strategy == "cross_lead":
        return apply_cross_lead(card.get("region"), card.get("peers") or {}, hits, climate, fit, val)
    if strategy == "jump":
        return apply_jump(model, hits, climate, fit, val)
    if strategy == "month_rate":
        month = card.get("month")
        if month is None:
            return clip(climate)
        return apply_month_rate(hits, climate, fit, month)
    if strategy == "equity":
        return apply_equity(model, hits, climate, horizon)
    if strategy == "after_quiet":
        return apply_after_quiet(model, hits, climate, fit, val, horizon)
    if strategy == "tide_gate":
        return apply_tide_gate(model, groups, hits, climate, fit, val)
    if strategy == "pair":
        return apply_pair(groups, hits, climate, val)
    if strategy == "cost3":
        return apply_cost(model, hits, climate, val)
    if strategy == "fade":
        return apply_fade(model, hits, climate, fit, val)
    if strategy == "hold":
        return apply_hold(model, hits, climate, fit, val, horizon)
    if strategy == "skeptic":
        return apply_skeptic(model, climate)
    if strategy == "believer":
        return apply_believer(model, climate)
    if strategy == "residual_alarm":
        return apply_residual_alarm(model, hits, climate, fit, val)
    if strategy == "trailing_rank":
        return apply_trailing_rank(model, hits, climate, fit, val)
    if strategy == "group_spread":
        return apply_group_spread(groups, hits, climate, fit, val)
    if strategy == "pattern8":
        return apply_pattern(groups, hits, climate, val)
    if strategy == "one_group":
        return apply_one_group(groups, hits, climate, val)
    if strategy == "veto":
        return apply_veto(model, groups, climate)
    if strategy == "compound":
        return apply_compound(card.get("short"), climate, horizon, bool(card.get("same_mag")))
    if strategy == "cross_horizon":
        return apply_cross_horizon(model, card.get("short"), hits, climate, fit, val, bool(card.get("same_mag")))
    if strategy == "markov":
        return apply_markov(hits, climate, fit, horizon)
    if strategy == "solar_phase":
        year_index = card.get("year_index")
        if year_index is None:
            return clip(climate)
        return apply_solar_phase(hits, climate, fit, year_index)
    if strategy == "contrarian":
        return apply_contrarian(model, hits, climate, horizon)
    if strategy == "median_groups":
        return apply_median_groups(groups, hits, climate, val)
    if strategy == "quiet_decade":
        return apply_quiet_decade(model, hits, climate, fit, val)
    if strategy == "online_decile":
        return apply_online_decile(model, hits, climate, horizon)
    if strategy == "online_trust":
        return apply_online_trust(model, hits, climate, horizon)
    if strategy == "lunar":
        lunar = card.get("lunar")
        if lunar is None:
            return clip(climate)
        return apply_lunar(hits, climate, fit, lunar)
    if strategy == "weekday":
        dow = card.get("dow")
        if dow is None:
            return clip(climate)
        return apply_weekday(hits, climate, fit, dow)
    if strategy == "peer_outcome":
        return apply_peer_outcome(card.get("region"), card.get("peer_hits") or {}, hits, climate, fit, horizon)
    if strategy == "stack_hot":
        return apply_stack_hot_rate(model, card.get("stack") or [], hits, climate, fit, val)
    if strategy == "deadzone":
        return apply_deadzone(model, hits, climate, val)
    if strategy == "climate_rise":
        return apply_climate_rise(climate, hits, fit, val)
    if strategy == "neutron_low":
        return apply_group_tail(groups, "neutrons", hits, climate, fit, val, low=True)
    if strategy == "slowslip_high":
        return apply_group_tail(groups, "slowslip", hits, climate, fit, val, low=False)
    if strategy == "accel":
        return apply_accel(model, hits, climate, fit, val)
    if strategy == "contradict":
        return apply_contradict(model, groups, hits, climate, val)
    if strategy == "peer_sum":
        return apply_peer_sum(card.get("region"), card.get("peers") or {}, hits, climate, fit, val)
    if strategy == "spf_high":
        return apply_group_tail(groups, "spf", hits, climate, fit, val, low=False)
    if strategy == "solar_high":
        return apply_group_tail(groups, "solar", hits, climate, fit, val, low=False)
    years = card.get("year")
    if years is None:
        return clip(climate)
    if strategy == "stable_alarm":
        return apply_stable_alarm(model, hits, climate, fit, years)
    if strategy == "stable_slowslip":
        group = groups.get("slowslip")
        if group is None:
            return clip(climate)
        return apply_stable_alarm(group, hits, climate, fit, years)
    if strategy == "stable_neutrons":
        group = groups.get("neutrons")
        if group is None:
            return clip(climate)
        return apply_stable_alarm(-np.asarray(group, dtype=float), hits, climate, fit, years)
    if strategy == "stable_interface":
        group = groups.get("interface quakes")
        if group is None:
            return clip(climate)
        return apply_stable_alarm(group, hits, climate, fit, years)
    if strategy == "stable_spf":
        group = groups.get("spf")
        if group is None:
            return clip(climate)
        return apply_stable_alarm(group, hits, climate, fit, years)
    if strategy == "four_linear":
        return apply_four_linear(groups, hits, climate, years)
    if strategy == "best_conjunction":
        return apply_best_conjunction(groups, hits, climate, years)
    return None
