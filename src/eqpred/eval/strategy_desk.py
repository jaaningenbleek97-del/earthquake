"""Fast replays of scoring rules. The heavy model is saved once. Knobs stay cheap."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.paths import data_dir
from eqpred.safe_io import replace_file, write_json

REGIONS = ("global", "hikurangi", "alpine", "kermadec")
LABELS = {
    "global": "Earth-wide",
    "hikurangi": "Hikurangi",
    "alpine": "Alpine Fault",
    "kermadec": "Kermadec",
}
STRATEGIES = (
    ("best_conjunction", "Pair that held from 1990 to 2008"),
    ("stable_slowslip", "Slow slip that held in three eras"),
    ("median_groups", "Middle group vote"),
    ("stable_spf", "Polar field that held in three eras"),
    ("quiet_catalog", "Quiet catalog"),
    ("fade", "Fade a hot score"),
    ("quiet_decade", "Loud call in a quiet stretch"),
    ("climate_rise", "Usual rate is climbing"),
    ("stable_alarm", "Alarm that held in three eras"),
    ("stable_neutrons", "Neutrons that held in three eras"),
    ("stable_interface", "Interface quakes that held in three eras"),
    ("hold", "Hold the window open"),
    ("spf_high", "Polar field is high"),
    ("alarm_14_15", "14-day alarm, top 15%"),
    ("accel", "Score is speeding up"),
    ("pair", "Interface and slab"),
    ("neutron_low", "Neutrons are low"),
    ("alarm_14_50", "14-day alarm, top half"),
    ("crossing", "First crossing"),
    ("group_spread", "Groups disagree"),
    ("alarm_14_40", "14-day alarm, top 40%"),
    ("cross_lead", "Neighbor fault leads"),
    ("peer_sum", "Sum of the other places"),
    ("crest", "Crest only"),
    ("solar_high", "Sun is loud"),
    ("veto", "A down group cancels"),
    ("online_trust", "Trust earned last quarter"),
    ("alarm_14_30", "14-day alarm, top 30%"),
    ("jump", "Sudden jump"),
    ("slowslip_high", "Slow slip is high"),
    ("cross_horizon", "Short window must agree"),
    ("believer", "Only trust a louder call"),
    ("alarm_rarity", "Alarm plus rarity"),
    ("markov", "Previous window"),
    ("after_quiet", "After a quiet window"),
    ("residual_alarm", "Gap above the usual rate"),
    ("alarm", "Alarm, then size"),
    ("tide_gate", "Tide must agree"),
    ("peer_outcome", "Neighbor already shook"),
    ("four_linear", "Four old measurements"),
    ("weekday", "Weekday"),
    ("one_group", "One best group"),
    ("trailing_rank", "Hot versus its own month"),
    ("solar_phase", "Solar-cycle season"),
    ("lunar", "Lunar phase"),
    ("stack_hot", "Three windows agree"),
    ("rarity_3_200", "3-day rarity"),
    ("contradict", "Model and groups disagree"),
    ("equity", "Only while skill is up"),
    ("run3", "Three loud days"),
    ("month_rate", "Calendar month"),
    ("bin", "Bin map"),
    ("online_decile", "Recent score levels"),
    ("rarity", "Rarity sets the rules"),
    ("shrink", "Shrink the move"),
    ("contrarian", "Trust it after a bad year"),
    ("odds", "Odds, not points"),
    ("two_sided", "Both tails"),
    ("skeptic", "Only trust a quieter call"),
    ("deadzone", "Ignore a near miss"),
    ("stay", "Stay home"),
    ("current", "Current card"),
    ("leftover", "Only the leftover"),
    ("cost3", "Misses count triple"),
    ("pattern8", "Eight sign patterns"),
    ("agree", "Agree or stay"),
    ("earned", "Earned cap"),
    ("compound", "Stack the one-day chance"),
    ("matched", "Matched mornings"),
)
# Rules that cleared the 2009–2018 backtest. The name locks the knobs.
PRESETS = {
    "alarm_14_15": {"kind": "alarm", "lag_days": 14, "alarm_share": 0.15},
    "alarm_14_30": {"kind": "alarm", "lag_days": 14, "alarm_share": 0.3},
    "alarm_14_40": {"kind": "alarm", "lag_days": 14, "alarm_share": 0.4},
    "alarm_14_50": {"kind": "alarm", "lag_days": 14, "alarm_share": 0.5},
    "rarity_3_200": {"kind": "rarity", "lag_days": 3, "rarity_n0": 200},
}
SHRINKS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_TAPE = "strategy_tape_2008"
TAPE_FOR_LABEL = {1: DEFAULT_TAPE, 2: "strategy_tape_2008_v2"}
_TAPES: dict[str, dict[str, Any]] = {}


def tape_dir(root: Path | None = None, name: str = DEFAULT_TAPE) -> Path:
    return data_dir(root) / "eval" / name


class _Arrays(dict):
    """Tape arrays held in memory, so a rebuild can replace the file while the lab is open."""

    @property
    def files(self) -> list[str]:
        return list(self.keys())


def fit_fingerprint(packed: dict[str, np.ndarray], meta: dict[str, Any]) -> str | None:
    """Short hash of everything the tape says before its later years. Changes when the fit's inputs change."""
    import hashlib

    if not meta.get("start") or not meta.get("holdout_start"):
        return None
    days = pd.date_range(meta["start"], periods=int(meta["days"]), freq="D", tz="UTC")
    cut = int(np.searchsorted(days, pd.Timestamp(meta["holdout_start"], tz="UTC")))
    digest = hashlib.sha256()
    for key in sorted(packed.keys()):
        values = np.asarray(packed[key])
        if values.ndim == 0 or values.shape[0] != len(days):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(values[:cut]).tobytes())
    return digest.hexdigest()[:16]


def list_tapes(root: Path | None = None) -> list[dict[str, Any]]:
    """Every built strategy tape, with its answer key and how far it reaches."""
    out = []
    folder = data_dir(root) / "eval"
    for meta_path in sorted(folder.glob("strategy_tape*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        end = None
        if meta.get("start") and meta.get("days"):
            end = (pd.Timestamp(meta["start"]) + pd.Timedelta(days=int(meta["days"]) - 1)).date().isoformat()
        out.append(
            {
                "name": meta_path.parent.name,
                "label_version": int(meta.get("label_version") or 1),
                "train_end": meta.get("train_end"),
                "built_at": meta.get("built_at"),
                "end": end,
            }
        )
    return out


def shift_score(values: np.ndarray, days: int) -> np.ndarray:
    """Yesterday's score is allowed. A positive lag reads that many mornings back."""
    values = np.asarray(values, dtype=float)
    days = int(days or 0)
    if days <= 0:
        return values.copy()
    out = np.full(len(values), np.nan, dtype=float)
    out[days:] = values[:-days]
    return out


def resolve_strategy(
    strategy: str,
    alarm_share: float,
    rarity_n0: float,
    lag_days: int,
) -> tuple[str, float, float, int]:
    preset = PRESETS.get(strategy)
    if preset is None:
        return strategy, float(alarm_share), float(rarity_n0), int(lag_days or 0)
    return (
        str(preset["kind"]),
        float(preset.get("alarm_share", alarm_share)),
        float(preset.get("rarity_n0", rarity_n0)),
        int(preset.get("lag_days", lag_days)),
    )


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def _gain(p: np.ndarray, y: np.ndarray, c: np.ndarray, mask: np.ndarray) -> float | None:
    from eqpred.eval.skill import information_gain

    m = np.asarray(mask, dtype=bool) & np.isfinite(p) & np.isfinite(y) & np.isfinite(c)
    if int(m.sum()) < 8:
        return None
    return information_gain(p[m], y[m], c[m])


def _logit(p: np.ndarray) -> np.ndarray:
    x = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def apply_shrink(model: np.ndarray, climate: np.ndarray, factor: float) -> np.ndarray:
    return clip(climate + factor * (model - climate))


def apply_odds(model: np.ndarray, climate: np.ndarray, factor: float) -> np.ndarray:
    return clip(_expit(_logit(climate) + factor * (_logit(model) - _logit(climate))))


def apply_bin_map(model: np.ndarray, hits: np.ndarray, climate: np.ndarray, val: np.ndarray, bins: int) -> np.ndarray:
    out = np.asarray(climate, dtype=float).copy()
    pv = model[val]
    yv = hits[val]
    ok = np.isfinite(pv) & np.isfinite(yv)
    bins = max(2, int(bins))
    if int(ok.sum()) < bins * 8:
        return clip(out)
    edges = np.quantile(pv[ok], np.linspace(0, 1, bins + 1))
    for i in range(bins):
        hi = edges[i + 1]
        if i == bins - 1:
            sel = ok & (pv >= edges[i]) & (pv <= hi)
            choose = np.isfinite(model) & (model >= edges[i]) & (model <= hi)
        else:
            sel = ok & (pv >= edges[i]) & (pv < hi)
            choose = np.isfinite(model) & (model >= edges[i]) & (model < hi)
        if int(sel.sum()) < 20:
            continue
        out[choose] = float(yv[sel].mean())
    return clip(out)


def apply_alarm(
    model: np.ndarray,
    hits: np.ndarray,
    climate: np.ndarray,
    fit: np.ndarray,
    val: np.ndarray,
    share: float,
) -> np.ndarray:
    out = np.asarray(climate, dtype=float).copy()
    src = model[fit]
    src = src[np.isfinite(src)]
    share = min(0.9, max(0.05, float(share)))
    if src.size < 40:
        return clip(out)
    thr = float(np.quantile(src, 1.0 - share))
    loud = np.asarray(val, dtype=bool) & np.isfinite(model) & np.isfinite(hits) & (model >= thr)
    quiet = np.asarray(val, dtype=bool) & np.isfinite(hits)
    if int(loud.sum()) < 8 or int(quiet.sum()) < 8:
        return clip(out)
    rate = float(hits[loud].mean())
    if rate <= float(hits[quiet].mean()):
        return clip(out)
    out[np.isfinite(model) & (model >= thr)] = rate
    return clip(out)


def apply_stay(model: np.ndarray, hits: np.ndarray, climate: np.ndarray, val: np.ndarray) -> np.ndarray:
    score = _gain(model, hits, climate, val)
    if score is not None and score > 0:
        return clip(model)
    return clip(climate)


def apply_rarity(model: np.ndarray, hits: np.ndarray, climate: np.ndarray, fit: np.ndarray, n0: float) -> np.ndarray:
    n = float(np.nansum(hits[fit]))
    n0 = max(1.0, float(n0))
    factor = n / (n + n0)
    return clip(climate + factor * (model - climate))


def apply_alarm_rarity(
    model: np.ndarray,
    hits: np.ndarray,
    climate: np.ndarray,
    fit: np.ndarray,
    val: np.ndarray,
    share: float,
    n0: float,
) -> np.ndarray:
    out = np.asarray(climate, dtype=float).copy()
    src = model[fit]
    src = src[np.isfinite(src)]
    share = min(0.9, max(0.05, float(share)))
    if src.size < 40:
        return clip(out)
    thr = float(np.quantile(src, 1.0 - share))
    loud = np.asarray(val, dtype=bool) & np.isfinite(model) & np.isfinite(hits) & (model >= thr)
    quiet = np.asarray(val, dtype=bool) & np.isfinite(hits)
    if int(loud.sum()) < 8 or int(quiet.sum()) < 8:
        return clip(out)
    gap = float(hits[loud].mean()) - float(hits[quiet].mean())
    if gap <= 0:
        return clip(out)
    n = float(np.nansum(hits[fit]))
    factor = n / (n + max(1.0, float(n0)))
    choose = np.isfinite(model) & (model >= thr)
    out[choose] = climate[choose] + factor * gap
    return clip(out)


def apply_earned(base: np.ndarray, groups: dict[str, np.ndarray], hits: np.ndarray, climate: np.ndarray, val: np.ndarray) -> np.ndarray:
    current = clip(base)
    current_ig = _gain(current, hits, climate, val)
    if current_ig is None:
        current_ig = -1e9
    base_ig = _gain(base, hits, climate, val)
    if base_ig is None:
        base_ig = -1e9
    ranked = []
    for name, group in groups.items():
        trial_ig = _gain(clip(base + group), hits, climate, val)
        if trial_ig is None:
            continue
        ranked.append((trial_ig - base_ig, group))
    ranked.sort(key=lambda item: item[0], reverse=True)
    acc = np.zeros(len(base), dtype=float)
    for delta, group in ranked:
        if delta <= 0:
            continue
        trial_ig = _gain(clip(current + group), hits, climate, val)
        if trial_ig is None or trial_ig <= current_ig + 1e-6:
            continue
        current = clip(current + group)
        current_ig = trial_ig
        acc = acc + group
    return clip(base + acc)


def apply_leftover(
    base: np.ndarray,
    groups: dict[str, np.ndarray],
    hits: np.ndarray,
    climate: np.ndarray,
    fit: np.ndarray,
    val: np.ndarray,
) -> np.ndarray:
    if not groups:
        return clip(base)
    added = np.sum(list(groups.values()), axis=0)
    dev = np.asarray(base, dtype=float) - np.asarray(climate, dtype=float)
    m = np.asarray(fit, dtype=bool) & np.isfinite(added) & np.isfinite(dev)
    var = float(np.mean(dev[m] ** 2)) if int(m.sum()) else 0.0
    beta = float(np.mean(added[m] * dev[m]) / var) if var > 1e-12 else 0.0
    pred = clip(base + (added - beta * dev))
    if (_gain(pred, hits, climate, val) or -1e9) <= (_gain(base, hits, climate, val) or -1e9):
        return clip(base)
    return pred


def apply_agree(base: np.ndarray, groups: dict[str, np.ndarray], climate: np.ndarray, minimum: int) -> np.ndarray:
    if not groups:
        return clip(base)
    ups = []
    downs = []
    for group in groups.values():
        ups.append(np.where(group > 0.005, group, 0.0))
        downs.append(np.where(group < -0.005, group, 0.0))
    up_sum = np.sum(ups, axis=0)
    down_sum = np.sum(downs, axis=0)
    n_up = np.sum([u > 0 for u in ups], axis=0)
    n_down = np.sum([d < 0 for d in downs], axis=0)
    step = np.zeros(len(base), dtype=float)
    need = max(2, int(minimum))
    step[(n_up >= need) & (n_up >= 2 * n_down)] = up_sum[(n_up >= need) & (n_up >= 2 * n_down)]
    step[(n_down >= need) & (n_down >= 2 * n_up)] = down_sum[(n_down >= need) & (n_down >= 2 * n_up)]
    limit = 0.5 * np.asarray(climate, dtype=float)
    return clip(base + np.clip(step, -limit, limit))


def _best_factor(kind: str, bundle: list[dict[str, np.ndarray]]) -> float:
    best_a, best = 0.0, -1e9
    fn = apply_odds if kind == "odds" else apply_shrink
    for factor in SHRINKS:
        scores = []
        for item in bundle:
            score = _gain(fn(item["model"], item["c"], factor), item["y"], item["c"], item["val"])
            if score is not None:
                scores.append(score)
        if scores and float(np.mean(scores)) > best:
            best_a, best = factor, float(np.mean(scores))
    return best_a


def _matched(matrix: np.ndarray, hits: np.ndarray, fit: np.ndarray, val: np.ndarray, k: int, purge: int = 0) -> np.ndarray:
    lib = np.flatnonzero(fit)
    targets = np.arange(len(hits))
    k = max(5, min(int(k), max(len(lib) - 1, 5)))
    pred = np.empty(len(hits), dtype=float)
    lib_x = matrix[lib]
    lib_y = hits[lib]
    purge = int(purge or 0)
    for start in range(0, len(targets), 64):
        block_i = targets[start : start + 64]
        dist = ((matrix[block_i, None, :] - lib_x[None, :, :]) ** 2).sum(axis=2)
        if purge > 0:
            for row, day in enumerate(block_i):
                dist[row, np.abs(lib - day) <= purge] = np.inf
        k_use = min(k, dist.shape[1] - 1)
        take = np.argpartition(dist, k_use, axis=1)[:, :k_use]
        chosen = lib_y[take]
        chosen = np.where(np.isfinite(dist[np.arange(len(block_i))[:, None], take]), chosen, np.nan)
        pred[block_i] = np.nanmean(chosen, axis=1)
    return clip(pred)


def _best_k(matrix: np.ndarray, hits: np.ndarray, fit: np.ndarray, val: np.ndarray, climate: np.ndarray, purge: int = 0) -> int:
    best_k, best = 40, -1e9
    for k in (20, 40, 80):
        pred = _matched(matrix, hits, fit, val, k, purge)
        score = _gain(pred, hits, climate, val)
        if score is not None and score > best:
            best_k, best = k, score
    return best_k


def _t_critical(df: int) -> float:
    if df >= 120:
        return 1.96
    if df >= 40:
        return 2.02
    if df >= 20:
        return 2.09
    return 2.26


def gain_band(p: np.ndarray, y: np.ndarray, c: np.ndarray, mask: np.ndarray) -> dict[str, Any] | None:
    """Mean log-score gap versus the usual rate, with a 95% interval and a rank check.

    The interval is the paired t interval used for earthquake-forecast comparisons.
    The rank check is the sign of a Wilcoxon statistic. Both have to say the gap
    is above zero before tests_agree is true.
    """
    m = np.asarray(mask, dtype=bool) & np.isfinite(p) & np.isfinite(y) & np.isfinite(c)
    if int(m.sum()) < 8:
        return None
    yy = np.asarray(y, dtype=float)[m]
    pp = clip(np.asarray(p, dtype=float)[m])
    bb = clip(np.asarray(c, dtype=float)[m])
    diff = (yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)) - (
        yy * np.log(bb) + (1.0 - yy) * np.log(1.0 - bb)
    )
    n = int(diff.size)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    width = _t_critical(n - 1) * se
    nonzero = diff[np.abs(diff) > 1e-12]
    rank_positive = False
    if nonzero.size >= 8:
        from scipy.stats import rankdata

        magnitude = np.round(np.abs(nonzero), 12)
        ranks = rankdata(magnitude)
        w_plus = float(ranks[nonzero > 0].sum())
        nn = int(nonzero.size)
        mu = nn * (nn + 1) / 4.0
        _, tie_counts = np.unique(magnitude, return_counts=True)
        tie_term = float(np.sum(tie_counts**3 - tie_counts)) / 48.0
        variance = nn * (nn + 1) * (2 * nn + 1) / 24.0 - tie_term
        sigma = np.sqrt(variance) if variance > 0 else 0.0
        rank_positive = bool(sigma > 0 and (w_plus - mu) / sigma > 1.96)
    low = mean - width
    return {
        "mean": mean,
        "low": low,
        "high": mean + width,
        "tests_agree": bool(low > 0 and rank_positive),
    }


def _row(region: str, mag: float, pred: np.ndarray, hits: np.ndarray, climate: np.ndarray, mask: np.ndarray, check: float | None) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(hits) & np.isfinite(climate)
    above = m & (pred > climate)
    below = m & (pred < climate)
    exam = _gain(pred, hits, climate, mask)
    band = gain_band(pred, hits, climate, mask)
    return {
        "region": region,
        "label": LABELS.get(region, region),
        "magnitude_min": mag,
        "mornings": int(m.sum()),
        "check_information_gain": None if check is None else round(float(check), 4),
        "information_gain": None if exam is None else round(float(exam), 4),
        "gain_low": None if band is None else round(float(band["low"]), 4),
        "gain_high": None if band is None else round(float(band["high"]), 4),
        "tests_agree": False if band is None else bool(band["tests_agree"]),
        "card_mean": None if not m.any() else round(float(pred[m].mean()), 4),
        "usual_mean": None if not m.any() else round(float(climate[m].mean()), 4),
        "happened": None if not m.any() else round(float(hits[m].mean()), 4),
        "n_above": int(above.sum()),
        "n_below": int(below.sum()),
        "above": None if int(above.sum()) < 8 else round(float(hits[above].mean()), 4),
        "below": None if int(below.sum()) < 8 else round(float(hits[below].mean()), 4),
    }


def load_tape(root: Path | None = None, name: str = DEFAULT_TAPE) -> dict[str, Any] | None:
    """Load a tape into memory. A rebuilt tape is picked up on the next call.

    Only the default tape falls back to the older 2018 tape; any other name
    that has not been built returns None rather than a different answer key.
    """
    name = name or DEFAULT_TAPE
    requested = name
    folder = tape_dir(root, name)
    if name == DEFAULT_TAPE and not (folder / "tape.npz").exists():
        folder = tape_dir(root, "strategy_tape")
        name = "strategy_tape"
    arrays = folder / "tape.npz"
    meta_path = folder / "meta.json"
    if not arrays.exists() or not meta_path.exists():
        return None
    stamp = (arrays.stat().st_mtime_ns, meta_path.stat().st_mtime_ns)
    cache_key = f"{folder.resolve()}::{stamp}"
    cached = _TAPES.get(requested)
    if cached is not None and cached.get("cache_key") == cache_key:
        return cached
    with np.load(arrays) as npz:
        packed = _Arrays({key: npz[key] for key in npz.files if key != "__meta__"})
        embedded = str(npz["__meta__"]) if "__meta__" in npz.files else None
    meta = json.loads(embedded) if embedded else json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("label_version", 1)
    if not meta.get("fit_hash"):
        meta["fit_hash"] = fit_fingerprint(packed, meta)
    tape = {
        "npz": packed,
        "meta": meta,
        "name": name,
        "requested_name": requested,
        "cache_key": cache_key,
        "root": str(folder.resolve()),
    }
    _TAPES[requested] = tape
    return tape


def score_strategy(
    *,
    horizon: int = 30,
    region: str = "all",
    strategy: str = "alarm_rarity",
    shrink: float | None = None,
    odds: float | None = None,
    alarm_share: float = 0.2,
    rarity_n0: float = 50,
    bins: int = 5,
    agree_min: int = 3,
    matched_k: int | None = None,
    use_readings: bool = False,
    lag_days: int = 0,
    root: Path | None = None,
    tape_name: str | None = None,
) -> dict[str, Any]:
    tape_name = tape_name or DEFAULT_TAPE
    tape = load_tape(root, name=tape_name)
    if tape is None:
        version = next((v for v, n in TAPE_FOR_LABEL.items() if n == tape_name), None)
        hint = f" Build it with: eqpred tape --label {version}" if version else ""
        return {
            "ready": False,
            "note": f"The {tape_name} tape has not been built yet.{hint}",
        }
    meta = tape["meta"]
    packed = tape["npz"]
    horizon = int(horizon)
    if str(horizon) not in meta["horizons"]:
        return {"ready": False, "note": f"No saved tape for a {horizon}-day card."}
    kind, alarm_share, rarity_n0, lag_days = resolve_strategy(strategy, alarm_share, rarity_n0, lag_days)
    full_mask = _full_mask(meta, horizon)
    info = meta["horizons"][str(horizon)]
    places = list(info["regions"]) if region == "all" else [region]
    names = [name for name, _label in STRATEGIES]
    if strategy not in names:
        return {"ready": False, "note": "Unknown strategy."}
    matrix = packed["matrix"]
    fit = packed[f"{horizon}_fit"]
    val = packed[f"{horizon}_val"]
    exam = packed[f"{horizon}_exam"]
    bundle = []
    cards = []
    for rid in places:
        groups = {
            name: packed[f"{horizon}_{rid}_g{i}"]
            for i, name in enumerate(info["groups"][rid])
        }
        card = {
            "region": rid,
            "mag": info["mags"][rid],
            "model": packed[f"{horizon}_{rid}_model"],
            "live": packed[f"{horizon}_{rid}_live"],
            "base": packed[f"{horizon}_{rid}_base"],
            "y": packed[f"{horizon}_{rid}_y"],
            "c": packed[f"{horizon}_{rid}_c"],
            "groups": groups,
            "fit": fit,
            "val": val,
            "exam": exam,
            "purge": horizon,
        }
        cards.append(card)
        bundle.append({"model": card["model"], "c": card["c"], "y": card["y"], "val": val})
    chosen_shrink = float(shrink) if shrink is not None else _best_factor("shrink", bundle)
    chosen_odds = float(odds) if odds is not None else _best_factor("odds", bundle)
    peers = {card["region"]: card["model"] for card in cards}
    month = None
    year_index = None
    if meta.get("start"):
        days = pd.date_range(meta["start"], periods=int(meta["days"]), freq="D", tz="UTC")
        month = days.month.to_numpy()
        year_index = np.asarray((days - days[0]).days, dtype=float) / 365.25
        dow = days.dayofweek.to_numpy()
        year = days.year.to_numpy()
        new_moon = pd.Timestamp("2000-01-06", tz="UTC")
        lunar = np.floor(np.mod((days - new_moon).days, 29.530588853) / 29.530588853 * 8).astype(int)
    else:
        days = None
        dow = None
        lunar = None
        year = None
    shorter = {}
    for other_s, other in meta["horizons"].items():
        other_h = int(other_s)
        if other_h >= horizon:
            continue
        for rid in other["regions"]:
            shorter.setdefault(rid, [])
            shorter[rid].append((other_h, float(other["mags"][rid]), packed[f"{other_h}_{rid}_model"]))
    for card in cards:
        card["peers"] = peers
        card["month"] = month
        card["year_index"] = year_index
        card["dow"] = dow
        card["lunar"] = lunar
        card["year"] = year
        card["horizon"] = horizon
        card["peer_hits"] = {
            other: packed[f"{horizon}_{other}_y"] for other in info["regions"] if other != card["region"]
        }
        card["stack"] = [
            packed[f"{other_h}_{card['region']}_model"]
            for other_h in (1, 3, 7, 30)
            if f"{other_h}_{card['region']}_model" in packed.files
        ]
        card["short"] = None
        card["same_mag"] = False
        for _other_h, other_mag, other_model in shorter.get(card["region"], []):
            if abs(other_mag - float(card["mag"])) < 1e-6:
                card["short"] = other_model
                card["same_mag"] = True
                break
    rows = []
    forecast_parts = []
    for card in cards:
        model = card["live"] if use_readings else card["model"]
        pred, check = _predict(
            kind,
            card,
            model,
            matrix,
            chosen_shrink,
            chosen_odds,
            alarm_share,
            rarity_n0,
            bins,
            agree_min,
            matched_k,
            lag_days,
        )
        if days is not None:
            from eqpred.eval.attempts import forecast_frame

            keep = np.array(full_mask, copy=True)
            if horizon > 0 and np.isfinite(pred[-1]):
                keep[-1] = True
            forecast_parts.append(
                forecast_frame(
                    days=days,
                    region=card["region"],
                    label=LABELS.get(card["region"], card["region"]),
                    horizon=horizon,
                    magnitude=float(card["mag"]),
                    p=pred,
                    climatology=card["c"],
                    happened=card["y"],
                    mask=keep,
                    catalog_end=days[-1],
                )
            )
        row = _row(card["region"], card["mag"], pred, card["y"], card["c"], exam, check)
        develop = _row(card["region"], card["mag"], pred, card["y"], card["c"], _develop_mask(meta, horizon), None)
        row["develop_information_gain"] = develop["information_gain"]
        row["develop_gain_low"] = develop["gain_low"]
        row["develop_gain_high"] = develop["gain_high"]
        row["develop_mornings"] = develop["mornings"]
        whole = _row(card["region"], card["mag"], pred, card["y"], card["c"], full_mask, check)
        row["full_information_gain"] = whole["information_gain"]
        row["full_card_mean"] = whole["card_mean"]
        row["full_usual_mean"] = whole["usual_mean"]
        row["full_happened"] = whole["happened"]
        row["full_above"] = whole["above"]
        row["full_below"] = whole["below"]
        row["full_mornings"] = whole["mornings"]
        rows.append(row)
    label = dict(STRATEGIES)[strategy]
    version = int(meta.get("label_version") or 1)
    saved = _save_replay(
        forecast_parts,
        strategy=strategy,
        label=label,
        version=version,
        tape_name=tape["requested_name"],
        knobs={
            "shrink": chosen_shrink,
            "odds": chosen_odds,
            "alarm_share": alarm_share,
            "rarity_n0": rarity_n0,
            "bins": bins,
            "agree_min": agree_min,
            "matched_k": matched_k,
            "use_readings": use_readings,
            "lag_days": lag_days,
            "kind": kind,
            "horizon_days": horizon,
        },
        root=root,
    )
    saved_sentence = ""
    if saved.get("n_forecasts"):
        saved_sentence = (
            f" Saved {saved['n_forecasts']} rehearsal forecasts as {saved['attempt_id']}."
            " They do not change the live exam."
        )
    return {
        "ready": True,
        "strategy": strategy,
        "label": label,
        "saved": saved,
        "horizon_days": horizon,
        "choices": {
            "shrink": chosen_shrink,
            "odds": chosen_odds,
            "alarm_share": alarm_share,
            "rarity_n0": rarity_n0,
            "bins": bins,
            "agree_min": agree_min,
            "matched_k": matched_k,
            "use_readings": use_readings,
            "lag_days": lag_days,
            "kind": kind,
        },
        "rows": rows,
        "tape": {
            "name": tape["requested_name"],
            "label_version": int(meta.get("label_version") or 1),
            "built_at": meta.get("built_at"),
            "fit_hash": meta.get("fit_hash"),
        },
        "note": (
            f"Answer key version {version}. "
            f"Tape starts {meta.get('start', '1990')}. Check years sit inside the fit, through {meta.get('train_end', '2018')}. "
            "Later years are the held-out mornings. Full record is every closed window from the start of the tape. "
            "A named 14-day alarm or 3-day rarity keeps its own knobs."
            + saved_sentence
        ),
    }


def _save_replay(
    parts: list[pd.DataFrame],
    *,
    strategy: str,
    label: str,
    version: int,
    tape_name: str,
    knobs: dict[str, Any],
    root: Path | None,
) -> dict[str, Any]:
    from eqpred.eval.attempts import save_forecasts

    frame = pd.concat([part for part in parts if part is not None and len(part)], ignore_index=True) if parts else None
    if frame is None or len(frame) == 0:
        return {"n_forecasts": 0, "attempt_id": None, "writes_issued_p": False}
    return save_forecasts(
        frame,
        source="strategy",
        name=strategy,
        label=label,
        root=root,
        label_version=version,
        tape=tape_name,
        knobs=knobs,
        answer_key=f"fresh.v{version}",
        note=(
            "Strategy replay. Each row is one dated chance. "
            "Closed windows can be scored later. This does not change the live exam."
        ),
    )


def archive_desk_forecasts(
    root: Path | None = None,
    horizons: tuple[int, ...] = (3, 7, 30),
    tapes: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Save a forecast row for every rule already on the desk. Does not re-rank them."""
    names = list(tapes) if tapes else [TAPE_FOR_LABEL[1], TAPE_FOR_LABEL[2]]
    saved: list[dict[str, Any]] = []
    for tape_name in names:
        if load_tape(root, name=tape_name) is None:
            print(f"skip {tape_name}: tape is not built", flush=True)
            continue
        for horizon in horizons:
            for strategy, label in STRATEGIES:
                print(f"archive {tape_name} {horizon}d {strategy}", flush=True)
                try:
                    report = score_strategy(
                        horizon=int(horizon),
                        region="all",
                        strategy=strategy,
                        root=root,
                        tape_name=tape_name,
                    )
                except (ValueError, KeyError, OSError) as exc:
                    print(f"skip {strategy}: {exc}", flush=True)
                    continue
                item = report.get("saved") or {}
                saved.append(
                    {
                        "strategy": strategy,
                        "label": label,
                        "horizon_days": int(horizon),
                        "tape": tape_name,
                        "n_forecasts": item.get("n_forecasts") or 0,
                        "attempt_id": item.get("attempt_id"),
                        "ready": bool(report.get("ready")),
                    }
                )
    return saved


def _develop_mask(meta: dict[str, Any], horizon: int) -> np.ndarray:
    """1990–2008, one morning per window, embargoed at the 2009 boundary."""
    from eqpred.eval.honest import week_mask

    if not meta.get("start"):
        return np.zeros(int(meta.get("days") or 0), dtype=bool)
    days = pd.date_range(meta["start"], periods=int(meta["days"]), freq="D", tz="UTC")
    hold = pd.Timestamp("2009-01-01", tz="UTC")
    closed = (days >= "1990-01-01") & (days < hold) & (days + pd.Timedelta(days=int(horizon)) <= hold)
    return week_mask(np.asarray(closed), stride=int(horizon), fallback=False)


def _full_mask(meta: dict[str, Any], horizon: int) -> np.ndarray:
    from eqpred.eval.honest import week_mask

    if not meta.get("start"):
        return np.zeros(int(meta.get("days") or 0), dtype=bool)
    days = pd.date_range(meta["start"], periods=int(meta["days"]), freq="D", tz="UTC")
    catalog_end = days[-1]
    closed = np.asarray(days + pd.Timedelta(days=int(horizon)) <= catalog_end)
    return week_mask(closed, stride=int(horizon), fallback=False)


def _predict(strategy, card, model, matrix, shrink, odds, alarm_share, rarity_n0, bins, agree_min, matched_k, lag_days=0):
    y, c, fit, val = card["y"], card["c"], card["fit"], card["val"]
    base, groups = card["base"], card["groups"]
    lagged = shift_score(model, lag_days)
    if strategy == "current":
        pred = clip(lagged)
    elif strategy == "shrink":
        pred = apply_shrink(lagged, c, shrink)
    elif strategy == "odds":
        pred = apply_odds(lagged, c, odds)
    elif strategy == "bin":
        pred = apply_bin_map(lagged, y, c, val, bins)
    elif strategy == "earned":
        pred = apply_earned(base, groups, y, c, val)
    elif strategy == "leftover":
        pred = apply_leftover(base, groups, y, c, fit, val)
    elif strategy == "stay":
        pred = apply_stay(lagged, y, c, val)
    elif strategy == "alarm":
        pred = apply_alarm(lagged, y, c, fit, val, alarm_share)
    elif strategy == "matched":
        k = matched_k if matched_k is not None else _best_k(matrix, y, fit, val, c, int(card.get("purge") or 0))
        pred = _matched(matrix, y, fit, val, k, int(card.get("purge") or 0))
    elif strategy == "agree":
        pred = apply_agree(base, groups, c, agree_min)
    elif strategy == "rarity":
        pred = apply_rarity(lagged, y, c, fit, rarity_n0)
    elif strategy == "alarm_rarity":
        pred = apply_alarm_rarity(lagged, y, c, fit, val, alarm_share, rarity_n0)
    else:
        from eqpred.eval.new_strategies import apply_new

        fresh = apply_new(strategy, card, model)
        pred = fresh if fresh is not None else clip(lagged)
    return pred, _gain(pred, y, c, val)


def build_tape(
    root: Path | None = None,
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    name: str = "strategy_tape",
    label_version: int = 1,
) -> Path:
    """Fit the shared model once and save every series the desk replays.

    `label_version` picks the first-shock answer key for the model and the hits.
    The new file is written beside the old one and swapped in, so the lab can stay open.
    """
    from eqpred.eval.decluster import use_label_version

    with use_label_version(label_version):
        return _build_tape(root, train_end, holdout_start, name, int(label_version))


def _build_tape(root: Path | None, train_end: str, holdout_start: str, name: str, label_version: int) -> Path:
    from eqpred.eval.decluster import fresh_hits
    from eqpred.eval.honest import fit_val_split, purged_train_hold, week_mask
    from eqpred.eval.lineup import _measurement_group, open_three_day_rows, three_day_targets
    from eqpred.eval.mainshock import ISSUED_TARGETS, _fit_shared_stack, prepare_issued, rolling_climate_series
    from eqpred.regions import REGIONS as BOXES

    folder = data_dir(root) / "eval" / name
    folder.mkdir(parents=True, exist_ok=True)
    feat, catalog, days, design, catalog_end = prepare_issued(root)
    closed30 = np.asarray(days + pd.Timedelta(days=30) <= catalog_end)
    train30, _hold30 = purged_train_hold(days, closed30, train_end, holdout_start, purge=30)
    fit30, _val30 = fit_val_split(days, train30, purge=30)
    print("family matrix", flush=True)
    matrix = _family_matrix(design, fit30, _measurement_group)
    arrays: dict[str, np.ndarray] = {"matrix": matrix.astype(np.float32)}
    meta: dict[str, Any] = {
        "horizons": {},
        "days": len(days),
        "start": str(pd.Timestamp(days[0]).date()),
        "train_end": train_end,
        "holdout_start": holdout_start,
    }
    specs = [
        (1, (("global", 7.0), ("hikurangi", 4.5), ("alpine", 4.5), ("kermadec", 5.0))),
        (3, three_day_targets()),
        (7, ISSUED_TARGETS),
        (30, ISSUED_TARGETS),
    ]
    for horizon, targets in specs:
        print(f"tape {horizon}d", flush=True)
        closed = np.asarray(days + pd.Timedelta(days=horizon) <= catalog_end)
        train, hold = purged_train_hold(days, closed, train_end, holdout_start, purge=horizon)
        fit, val = fit_val_split(days, train, purge=horizon)
        exam = week_mask(hold, stride=horizon, fallback=False)
        shared, _r, _f, _c = _fit_shared_stack(
            feat=feat,
            design=design,
            catalog=catalog,
            days=days,
            train_mask=train,
            hold_mask=hold,
            horizon=horizon,
            targets=targets,
        )
        series: dict = {}
        open_three_day_rows(
            design=design,
            catalog=catalog,
            days=days,
            train_mask=train,
            hold_mask=hold,
            horizon=horizon,
            shared_base=shared,
            series_out=series,
            targets=targets,
        )
        arrays[f"{horizon}_fit"] = fit
        arrays[f"{horizon}_val"] = val
        arrays[f"{horizon}_exam"] = exam
        meta["horizons"][str(horizon)] = {"regions": [rid for rid, _mag in targets], "mags": {}, "groups": {}}
        for rid, mag in targets:
            hits = fresh_hits(catalog, BOXES[rid], mag, days, horizon).astype(np.float32)
            roll = rolling_climate_series(hits, days, horizon=horizon)
            climate = np.where(np.isfinite(roll), roll, float(np.nanmean(hits[train]))).astype(np.float32)
            block = series[rid]
            arrays[f"{horizon}_{rid}_model"] = np.asarray(shared[rid], dtype=np.float32)
            arrays[f"{horizon}_{rid}_live"] = np.asarray(block["p"], dtype=np.float32)
            arrays[f"{horizon}_{rid}_base"] = np.asarray(block["base"], dtype=np.float32)
            arrays[f"{horizon}_{rid}_y"] = hits
            arrays[f"{horizon}_{rid}_c"] = climate
            names = list(block["groups"])
            meta["horizons"][str(horizon)]["mags"][rid] = mag
            meta["horizons"][str(horizon)]["groups"][rid] = names
            for i, group_name in enumerate(names):
                arrays[f"{horizon}_{rid}_g{i}"] = np.asarray(block["groups"][group_name], dtype=np.float32)
        del shared, series
        gc.collect()
    meta["label_version"] = label_version
    meta["built_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    meta["fit_hash"] = fit_fingerprint(arrays, meta)
    path = folder / "tape.npz"
    tmp = folder / f".tape.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, __meta__=np.array(json.dumps(meta)), **arrays)
    replace_file(tmp, path)
    write_json(folder / "meta.json", meta, indent=None)
    _TAPES.clear()
    print(f"wrote {path}", flush=True)
    return path


def _family_matrix(design, fit: np.ndarray, measurement_group) -> np.ndarray:
    raw_x, names = design
    buckets: dict[str, list[np.ndarray]] = {}
    fit = np.asarray(fit, dtype=bool)
    for idx, name in enumerate(names):
        col = raw_x[:, idx]
        train = col[fit]
        finite = train[np.isfinite(train)]
        if finite.size < 20 or float(np.std(finite)) < 1e-12:
            continue
        mu = float(np.mean(finite))
        sd = float(np.std(finite))
        z = np.where(np.isfinite(col), (col - mu) / sd, np.nan)
        buckets.setdefault(measurement_group(name.split("__lag", 1)[0]), []).append(z)
    cols = []
    for series in buckets.values():
        with np.errstate(all="ignore"):
            cols.append(np.nanmean(np.vstack(series[:8]), axis=0))
    mat = np.vstack(cols).T if cols else np.zeros((raw_x.shape[0], 1))
    return np.where(np.isfinite(mat), mat, 0.0).astype(np.float32)
