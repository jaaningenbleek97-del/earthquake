"""Score each place and each window length on its own.

A card is judged by the windows it marked up and the windows it marked down.
Copying the usual rate on every window is a tie, not a win. Nothing here
changes the live exam.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.eval.attempts import attempts_dir
from eqpred.eval.skill import information_gain
from eqpred.paths import data_dir
from eqpred.safe_io import write_json

MIN_SIDE = 8
LABELS = {
    "global": "Earth-wide",
    "hikurangi": "Hikurangi",
    "alpine": "Alpine Fault",
    "kermadec": "Kermadec",
}
PLACE_ORDER = ("global", "hikurangi", "alpine", "kermadec")


def separate_path(root: Path | None = None) -> Path:
    return data_dir(root) / "eval" / "separate_scores.json"


def split_windows(
    p: np.ndarray,
    happened: np.ndarray,
    climate: np.ndarray,
    *,
    min_side: int = MIN_SIDE,
) -> dict[str, Any]:
    """One card. Marked-up windows against marked-down windows."""
    chance = np.asarray(p, dtype=float)
    outcome = np.asarray(happened, dtype=float)
    usual = np.asarray(climate, dtype=float)
    ok = np.isfinite(chance) & np.isfinite(outcome) & np.isfinite(usual)
    chance, outcome, usual = chance[ok], outcome[ok], usual[ok]
    windows = int(chance.size)
    loud = chance > usual + 1e-4
    quiet = chance < usual - 1e-4
    n_up = int(loud.sum())
    n_down = int(quiet.sum())
    up_hits = int(outcome[loud].sum()) if n_up else 0
    down_hits = int(outcome[quiet].sum()) if n_down else 0
    up_rate = float(outcome[loud].mean()) if n_up else None
    down_rate = float(outcome[quiet].mean()) if n_down else None
    if n_up >= min_side and n_down >= min_side and up_rate is not None and down_rate is not None:
        gap = up_rate - down_rate
        if gap > 0:
            verdict = "up"
        elif gap < 0:
            verdict = "down"
        else:
            verdict = "flat"
    elif n_up < min_side and n_down < min_side:
        verdict = "tie"
        gap = None
    else:
        verdict = "thin"
        gap = None
    gain = information_gain(chance, outcome, usual) if windows else None
    return {
        "windows": windows,
        "happened": None if windows == 0 else round(float(outcome.mean()), 4),
        "n_up": n_up,
        "up_hits": up_hits,
        "up_rate": None if up_rate is None else round(up_rate, 4),
        "n_down": n_down,
        "down_hits": down_hits,
        "down_rate": None if down_rate is None else round(down_rate, 4),
        "n_tied": windows - n_up - n_down,
        "gap": None if gap is None else round(float(gap), 4),
        "information_gain": None if gain is None else round(float(gain), 4),
        "verdict": verdict,
    }


def _latest_strategy_files(root: Path | None) -> list[dict[str, Any]]:
    index = attempts_dir(root) / "index.jsonl"
    if not index.exists():
        return []
    chosen: dict[tuple, dict[str, Any]] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("source") != "strategy" or not row.get("file"):
            continue
        answer = str(row.get("answer_key") or "")
        if not answer.startswith("fresh."):
            continue
        for horizon in row.get("horizons") or []:
            chosen[(row.get("name"), row.get("tape"), int(horizon), answer)] = row
    return list(chosen.values())


def build_separate_scores(root: Path | None = None) -> dict[str, Any]:
    """Score every saved strategy card on its own place and window length."""
    cards: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for row in _latest_strategy_files(root):
        path = attempts_dir(root) / str(row["file"])
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["region", "label", "horizon_days", "p", "climatology_p", "happened", "closed", "slice"])
        later = frame[(frame["slice"] == "later") & frame["closed"]]
        if later.empty:
            continue
        answer = str(row.get("answer_key") or "")
        name = str(row.get("name") or "")
        for (region, horizon), block in later.groupby(["region", "horizon_days"], sort=False):
            key = (name, answer, str(region), int(horizon))
            if key in seen:
                continue
            seen.add(key)
            scored = split_windows(block["p"].to_numpy(), block["happened"].to_numpy(), block["climatology_p"].to_numpy())
            label = str(block["label"].iloc[0] or LABELS.get(str(region), region))
            cards.append(
                {
                    "strategy": name,
                    "rule": str(row.get("label") or name),
                    "place": str(region),
                    "place_label": LABELS.get(str(region), label),
                    "horizon_days": int(horizon),
                    "answer_key": answer,
                    "tape": row.get("tape"),
                    **scored,
                }
            )
    cards.sort(
        key=lambda card: (
            0 if card["verdict"] == "up" else 1,
            -(card["gap"] or -9),
            PLACE_ORDER.index(card["place"]) if card["place"] in PLACE_ORDER else 9,
            card["horizon_days"],
            card["rule"],
        )
    )
    moved = [card for card in cards if card["verdict"] in {"up", "down", "flat"}]
    report = {
        "ready": True,
        "writes_issued_p": False,
        "min_side": MIN_SIDE,
        "n_cards": len(cards),
        "n_left_the_usual_rate": len(moved),
        "n_marked_up_happened_more": sum(1 for card in cards if card["verdict"] == "up"),
        "cards": cards,
        "note": (
            "Each place and each window length is its own test, on mornings from 2009 onward. "
            "A card counts when the windows it marked up happened more often than the windows it marked down. "
            "A card that stays on the usual rate is a tie, not a win. This does not change the live exam."
        ),
    }
    write_json(separate_path(root), report)
    return report


def load_separate_scores(root: Path | None = None) -> dict[str, Any]:
    path = separate_path(root)
    if not path.exists():
        return {
            "ready": False,
            "writes_issued_p": False,
            "cards": [],
            "note": "The separate scores have not been built yet.",
        }
    return json.loads(path.read_text(encoding="utf-8"))
