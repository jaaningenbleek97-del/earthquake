"""New variations, chosen on old years and checked once on 2009 onward.

Thresholds and hit rates come from mornings before 1990. The 1990–2008
windows pick which variation is kept. 2009 onward is scored once and does
not choose anything. Nothing here changes the live exam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.eval.attempts import forecast_frame, save_forecasts
from eqpred.eval.honest import week_mask
from eqpred.eval.separate_scores import LABELS, split_windows
from eqpred.eval.strategy_desk import TAPE_FOR_LABEL, _develop_mask, load_tape
from eqpred.paths import data_dir
from eqpred.safe_io import write_json

SHARES = (0.1, 0.2, 0.3, 0.5)
LAGS = (0, 7, 14)
PAIR_SHARES = (0.2, 0.5)
PAIR_LAGS = (0, 14)
USEFUL_GAIN = 0.01
HORIZONS = (3, 7, 30)


def early_search_path(root: Path | None = None) -> Path:
    return data_dir(root) / "eval" / "early_search.json"


def _lag(values: np.ndarray, lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if lag <= 0:
        return values.copy()
    out = np.full(len(values), np.nan, dtype=float)
    out[lag:] = values[:-lag]
    return out


def _threshold(values: np.ndarray, mask: np.ndarray, share: float, side: str) -> float | None:
    finite = values[np.asarray(mask, dtype=bool) & np.isfinite(values)]
    if finite.size < 40:
        return None
    if side == "high":
        return float(np.quantile(finite, 1.0 - share))
    return float(np.quantile(finite, share))


def _alarm(values: np.ndarray, threshold: float, side: str) -> np.ndarray:
    finite = np.isfinite(values)
    if side == "high":
        return finite & (values >= threshold)
    return finite & (values <= threshold)


def _rate(on: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    chosen = np.asarray(mask, dtype=bool) & on & np.isfinite(y)
    if int(chosen.sum()) < 8:
        return None
    return float(y[chosen].mean())


def _choice_key(scored: dict[str, Any]) -> tuple:
    gain = scored.get("information_gain")
    gap = scored.get("gap")
    gain_n = -9.0 if gain is None else float(gain)
    gap_n = -9.0 if gap is None else float(gap)
    adopted = scored.get("verdict") == "up" and gain_n > 0
    return (adopted, gain_n, gap_n)


def _better(scored: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    return _choice_key(scored) > _choice_key(best["develop"])


def pick_best_alarm(
    signals: dict[str, np.ndarray],
    y: np.ndarray,
    climate: np.ndarray,
    early: np.ndarray,
    develop: np.ndarray,
) -> tuple[dict[str, Any] | None, int]:
    """Best single tail. The threshold and the rates use the early mask only."""
    best: dict[str, Any] | None = None
    tried = 0
    y = np.asarray(y, dtype=float)
    climate = np.asarray(climate, dtype=float)
    for name, raw in signals.items():
        for lag in LAGS:
            lagged = _lag(raw, lag)
            for share in SHARES:
                for side in ("high", "low"):
                    cut = _threshold(lagged, early, share, side)
                    if cut is None:
                        continue
                    on = _alarm(lagged, cut, side)
                    high = _rate(on, y, early)
                    low = _rate(~on & np.isfinite(lagged), y, early)
                    if high is None or low is None:
                        continue
                    tried += 1
                    chance = np.where(on, high, low)
                    scored = split_windows(chance[develop], y[develop], climate[develop])
                    if _better(scored, best):
                        best = {
                            "family": "single",
                            "signal": name,
                            "lag_days": lag,
                            "share": share,
                            "side": side,
                            "threshold": cut,
                            "p_on": high,
                            "p_off": low,
                            "develop": scored,
                            "chance": chance,
                        }
    return best, tried


def pick_best_pair(
    signals: dict[str, np.ndarray],
    y: np.ndarray,
    climate: np.ndarray,
    early: np.ndarray,
    develop: np.ndarray,
) -> tuple[dict[str, Any] | None, int]:
    """Both signals in a chosen tail. Rates still come from the early years."""
    names = list(signals)
    best: dict[str, Any] | None = None
    tried = 0
    y = np.asarray(y, dtype=float)
    climate = np.asarray(climate, dtype=float)
    lagged = {name: {lag: _lag(values, lag) for lag in PAIR_LAGS} for name, values in signals.items()}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for lag in PAIR_LAGS:
                a = lagged[left][lag]
                b = lagged[right][lag]
                for share in PAIR_SHARES:
                    for side_a in ("high", "low"):
                        for side_b in ("high", "low"):
                            cut_a = _threshold(a, early, share, side_a)
                            cut_b = _threshold(b, early, share, side_b)
                            if cut_a is None or cut_b is None:
                                continue
                            on = _alarm(a, cut_a, side_a) & _alarm(b, cut_b, side_b)
                            known = np.isfinite(a) & np.isfinite(b)
                            high = _rate(on, y, early)
                            low = _rate(known & ~on, y, early)
                            if high is None or low is None:
                                continue
                            tried += 1
                            chance = np.where(on, high, low)
                            scored = split_windows(chance[develop], y[develop], climate[develop])
                            if _better(scored, best):
                                best = {
                                    "family": "pair",
                                    "signal": f"{left} {side_a} and {right} {side_b}",
                                    "lag_days": lag,
                                    "share": share,
                                    "side": f"{side_a}+{side_b}",
                                    "threshold": None,
                                    "p_on": high,
                                    "p_off": low,
                                    "develop": scored,
                                    "chance": chance,
                                }
    return best, tried


def pick_best_bins(
    signals: dict[str, np.ndarray],
    y: np.ndarray,
    climate: np.ndarray,
    early: np.ndarray,
    develop: np.ndarray,
    bins: int = 4,
) -> tuple[dict[str, Any] | None, int]:
    """Four level bins. The bin edges and the rates use the early years only."""
    best: dict[str, Any] | None = None
    tried = 0
    y = np.asarray(y, dtype=float)
    climate = np.asarray(climate, dtype=float)
    for name, raw in signals.items():
        for lag in LAGS:
            lagged = _lag(raw, lag)
            finite = lagged[early & np.isfinite(lagged)]
            if finite.size < 40:
                continue
            edges = np.quantile(finite, np.linspace(0, 1, bins + 1))
            edges[0] = -np.inf
            edges[-1] = np.inf
            if np.unique(edges[1:-1]).size < bins - 1:
                continue
            band = np.digitize(lagged, edges[1:-1], right=False)
            rates = []
            ok = True
            for level in range(bins):
                rate = _rate((band == level) & np.isfinite(lagged), y, early)
                if rate is None:
                    ok = False
                    break
                rates.append(rate)
            if not ok:
                continue
            tried += 1
            chance = np.array(rates, dtype=float)[np.clip(band, 0, bins - 1)]
            chance = np.where(np.isfinite(lagged), chance, np.nan)
            scored = split_windows(chance[develop], y[develop], climate[develop])
            if _better(scored, best):
                best = {
                    "family": "bins",
                    "signal": name,
                    "lag_days": lag,
                    "share": None,
                    "side": f"{bins} bins",
                    "threshold": None,
                    "p_on": None,
                    "p_off": None,
                    "develop": scored,
                    "chance": chance,
                }
    return best, tried


def _masks(meta: dict[str, Any], horizon: int) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    days = pd.date_range(meta["start"], periods=int(meta["days"]), freq="D", tz="UTC")
    hold = pd.Timestamp("2009-01-01", tz="UTC")
    early_cut = pd.Timestamp("1990-01-01", tz="UTC")
    catalog_end = days[-1]
    closed = np.asarray(days + pd.Timedelta(days=int(horizon)) <= catalog_end)
    early = week_mask(
        closed & np.asarray(days + pd.Timedelta(days=int(horizon)) <= early_cut),
        stride=int(horizon),
        fallback=False,
    )
    later = week_mask(closed & np.asarray(days >= hold), stride=int(horizon), fallback=False)
    return days, early, _develop_mask(meta, horizon), later


def _signals(packed, horizon: int, region: str, names: list[str]) -> dict[str, np.ndarray]:
    out = {"model": np.asarray(packed[f"{horizon}_{region}_model"], dtype=float)}
    for i, name in enumerate(names):
        out[name] = np.asarray(packed[f"{horizon}_{region}_g{i}"], dtype=float)
    return out


def _public(choice: dict[str, Any] | None, later: dict[str, Any] | None) -> dict[str, Any]:
    if choice is None:
        return {"adopted": False, "choice": None, "develop": None, "later": None, "passed": False}
    develop = {k: v for k, v in choice["develop"].items()}
    adopted = bool(_choice_key(develop)[0])
    passed = False
    if adopted and later is not None:
        gain = later.get("information_gain") or 0
        passed = later.get("verdict") == "up" and gain >= USEFUL_GAIN
    return {
        "adopted": adopted,
        "passed": passed,
        "choice": {
            "family": choice["family"],
            "signal": choice["signal"],
            "lag_days": choice["lag_days"],
            "share": choice["share"],
            "side": choice["side"],
        },
        "develop": develop,
        "later": later,
    }


def run_early_search(root: Path | None = None, tapes: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Search variations for each place and window. 2009 does not pick the winner."""
    names = list(tapes) if tapes else [TAPE_FOR_LABEL[1], TAPE_FOR_LABEL[2]]
    cards: list[dict[str, Any]] = []
    tried = 0
    for tape_name in names:
        tape = load_tape(root, name=tape_name)
        if tape is None:
            print(f"skip {tape_name}: tape is not built", flush=True)
            continue
        meta = tape["meta"]
        packed = tape["npz"]
        version = int(meta.get("label_version") or 1)
        for horizon in HORIZONS:
            info = meta["horizons"][str(horizon)]
            days, early, develop, later_mask = _masks(meta, horizon)
            print(f"early search {tape_name} {horizon}d", flush=True)
            for region in info["regions"]:
                y = np.asarray(packed[f"{horizon}_{region}_y"], dtype=float)
                climate = np.asarray(packed[f"{horizon}_{region}_c"], dtype=float)
                signals = _signals(packed, horizon, region, list(info["groups"][region]))
                picked = []
                for picker in (pick_best_alarm, pick_best_pair, pick_best_bins):
                    choice, n_try = picker(signals, y, climate, early, develop)
                    tried += n_try
                    picked.append(choice)
                for choice in picked:
                    later = None
                    if choice is not None:
                        chance = choice.pop("chance")
                        later = split_windows(chance[later_mask], y[later_mask], climate[later_mask])
                        _save_choice(
                            choice,
                            chance,
                            y,
                            climate,
                            days,
                            horizon,
                            region,
                            float(info["mags"][region]),
                            version,
                            tape_name,
                            root,
                        )
                    row = _public(choice, later)
                    row.update(
                        {
                            "place": region,
                            "place_label": LABELS.get(region, region),
                            "horizon_days": horizon,
                            "magnitude_min": info["mags"][region],
                            "answer_key": f"fresh.v{version}",
                            "tape": tape_name,
                        }
                    )
                    cards.append(row)
                    print(
                        f"  {LABELS.get(region, region)} {row['choice']['family'] if row['choice'] else 'none'} "
                        f"adopted {row['adopted']} later {None if later is None else later.get('information_gain')}",
                        flush=True,
                    )
    passed = [card for card in cards if card["passed"]]
    pointed = [
        card
        for card in cards
        if card["adopted"] and card["later"] and card["later"].get("verdict") == "up" and (card["later"].get("information_gain") or 0) > 0
    ]
    report = {
        "ready": True,
        "writes_issued_p": False,
        "n_variations": tried,
        "n_cards": len(cards),
        "n_adopted": sum(1 for card in cards if card["adopted"]),
        "n_pointed_right": len(pointed),
        "n_passed": len(passed),
        "useful_gain": USEFUL_GAIN,
        "cards": cards,
        "note": (
            "Thresholds and rates come from before 1990. "
            "1990–2008 picks the variation. 2009 onward is checked once. "
            "A pass means the marked-up windows happened more often and the gain is at least 0.01. "
            "This does not change the live exam."
        ),
    }
    write_json(early_search_path(root), report)
    return report


def _save_choice(
    choice: dict[str, Any],
    chance: np.ndarray,
    y: np.ndarray,
    climate: np.ndarray,
    days: pd.DatetimeIndex,
    horizon: int,
    region: str,
    magnitude: float,
    version: int,
    tape_name: str,
    root: Path | None,
) -> None:
    from eqpred.eval.honest import week_mask

    catalog_end = days[-1]
    closed = np.asarray(days + pd.Timedelta(days=int(horizon)) <= catalog_end)
    mask = week_mask(closed, stride=int(horizon), fallback=False)
    frame = forecast_frame(
        days=days,
        region=region,
        label=LABELS.get(region, region),
        horizon=horizon,
        magnitude=magnitude,
        p=chance,
        climatology=climate,
        happened=y,
        mask=mask,
        catalog_end=catalog_end,
    )
    if len(frame) == 0:
        return
    save_forecasts(
        frame,
        source="early-search",
        name=f"early_{choice['family']}",
        label=str(choice["signal"]),
        root=root,
        label_version=version,
        tape=tape_name,
        answer_key=f"fresh.v{version}",
        knobs={
            "family": choice["family"],
            "signal": choice["signal"],
            "lag_days": choice["lag_days"],
            "share": choice["share"],
            "side": choice["side"],
            "horizon_days": horizon,
            "region": region,
        },
        note=(
            "Chosen on years before 2009. 2009 onward was not used to pick it. "
            "This does not change the live exam."
        ),
    )
