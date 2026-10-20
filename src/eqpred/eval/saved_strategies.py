"""Frozen strategy cards. A saved card keeps its knobs. New experiments do not rewrite it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from eqpred.eval.honest import week_mask
from eqpred.eval.strategy_desk import (
    DEFAULT_TAPE,
    LABELS,
    STRATEGIES,
    _predict,
    _row,
    load_tape,
    resolve_strategy,
)
from eqpred.paths import data_dir
from eqpred.safe_io import file_lock, read_json, write_json

SEED_ID = "kermadec-30-pair"


def _store_path(root: Path | None = None) -> Path:
    folder = data_dir(root) / "eval"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "saved_strategy_cards.json"


def _seed() -> dict[str, Any]:
    return {
        "id": SEED_ID,
        "saved_at": "2026-10-20",
        "strategy": "best_conjunction",
        "horizon": 30,
        "region": "kermadec",
        "knobs": {
            "alarm_share": 0.2,
            "rarity_n0": 50.0,
            "lag_days": 0,
            "bins": 5,
            "agree_min": 3,
            "shrink": 1.0,
            "odds": 1.0,
            "matched_k": 40,
            "use_readings": False,
        },
        "trials_at_save": 69,
        "name": "Kermadec 30-day pair",
    }


def load_store(root: Path | None = None) -> dict[str, Any]:
    path = _store_path(root)
    if not path.exists():
        store = {"cards": [_seed()]}
        write_json(path, store)
        return store
    return read_json(path)


def _write(store: dict[str, Any], root: Path | None = None) -> None:
    write_json(_store_path(root), store, backup=True)


def _trial_count() -> int:
    path = Path(__file__).resolve().parent / "strategy_rank.json"
    if not path.exists():
        return len(STRATEGIES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(len(payload.get("ranked") or STRATEGIES))


def _key(card: dict[str, Any]) -> tuple:
    knobs = card.get("knobs") or {}
    return (
        card.get("strategy"),
        int(card.get("horizon") or 0),
        card.get("region"),
        json.dumps(knobs, sort_keys=True),
        card.get("tape") or DEFAULT_TAPE,
    )


def save_card(spec: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    card = {
        "id": uuid4().hex[:12],
        "saved_at": datetime.now(timezone.utc).date().isoformat(),
        "strategy": spec["strategy"],
        "horizon": int(spec["horizon"]),
        "region": spec["region"],
        "knobs": spec.get("knobs") or {},
        "tape": spec.get("tape") or DEFAULT_TAPE,
        "tape_fit": spec.get("tape_fit"),
        "trials_at_save": _trial_count(),
        "name": spec.get("name") or "",
    }
    with file_lock(_store_path(root)):
        store = load_store(root)
        for existing in store["cards"]:
            if _key(existing) == _key(card):
                return existing
        store["cards"].append(card)
        _write(store, root)
    return card


def drop_card(card_id: str, root: Path | None = None) -> None:
    with file_lock(_store_path(root)):
        store = load_store(root)
        store["cards"] = [card for card in store["cards"] if card.get("id") != card_id]
        _write(store, root)


def _cards(horizon: int, packed, meta) -> tuple[list[dict[str, Any]], np.ndarray, pd.DatetimeIndex]:
    info = meta["horizons"][str(horizon)]
    fit = packed[f"{horizon}_fit"]
    val = packed[f"{horizon}_val"]
    exam = packed[f"{horizon}_exam"]
    built = []
    for rid in info["regions"]:
        groups = {name: packed[f"{horizon}_{rid}_g{i}"] for i, name in enumerate(info["groups"][rid])}
        built.append(
            {
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
                "horizon": horizon,
            }
        )
    peers = {card["region"]: card["model"] for card in built}
    days = pd.date_range(meta["start"], periods=int(meta["days"]), freq="D", tz="UTC")
    month = days.month.to_numpy()
    year_index = np.asarray((days - days[0]).days, dtype=float) / 365.25
    dow = days.dayofweek.to_numpy()
    year = days.year.to_numpy()
    lunar = np.floor(np.mod((days - pd.Timestamp("2000-01-06", tz="UTC")).days, 29.530588853) / 29.530588853 * 8).astype(int)
    shorter: dict[str, list] = {}
    for other_s, other in meta["horizons"].items():
        other_h = int(other_s)
        if other_h >= horizon:
            continue
        for rid in other["regions"]:
            shorter.setdefault(rid, []).append((float(other["mags"][rid]), packed[f"{other_h}_{rid}_model"]))
    for card in built:
        rid = card["region"]
        card["peers"] = peers
        card["month"] = month
        card["year_index"] = year_index
        card["dow"] = dow
        card["lunar"] = lunar
        card["year"] = year
        card["peer_hits"] = {other: packed[f"{horizon}_{other}_y"] for other in info["regions"] if other != rid}
        card["stack"] = [
            packed[f"{other_h}_{rid}_model"] for other_h in (1, 3, 7, 30) if f"{other_h}_{rid}_model" in packed.files
        ]
        card["short"] = None
        card["same_mag"] = False
        for other_mag, other_model in shorter.get(rid, []):
            if abs(other_mag - float(card["mag"])) < 1e-6:
                card["short"] = other_model
                card["same_mag"] = True
                break
    return built, packed["matrix"], days


def _slice(days: pd.DatetimeIndex, horizon: int, start: str, end: str | None) -> np.ndarray:
    opened = days >= start
    if end is not None:
        opened = opened & (days < end)
    closed = opened & (days + pd.Timedelta(days=int(horizon)) <= days[-1])
    return week_mask(np.asarray(closed), stride=int(horizon), fallback=False)


def _predict_region(card_spec: dict[str, Any], built: list[dict[str, Any]], matrix, knobs: dict[str, Any]) -> np.ndarray:
    kind, share, n0, lag = resolve_strategy(
        card_spec["strategy"],
        float(knobs.get("alarm_share") or 0.2),
        float(knobs.get("rarity_n0") or 50),
        int(knobs.get("lag_days") or 0),
    )
    target = next(card for card in built if card["region"] == card_spec["region"])
    model = target["live"] if knobs.get("use_readings") else target["model"]
    pred, _check = _predict(
        kind,
        target,
        model,
        matrix,
        float(knobs.get("shrink") if knobs.get("shrink") is not None else 1.0),
        float(knobs.get("odds") if knobs.get("odds") is not None else 1.0),
        share,
        n0,
        int(knobs.get("bins") or 5),
        int(knobs.get("agree_min") or 3),
        int(knobs.get("matched_k") or 40),
        lag,
    )
    return pred


def _tape_end(meta: dict[str, Any]) -> str:
    return (pd.Timestamp(meta["start"]) + pd.Timedelta(days=int(meta["days"]) - 1)).date().isoformat()


def _backfill_tape_fit(root: Path | None) -> None:
    """Older cards did not store the fit they were saved on. Adopt it only if the tape is not newer than the card."""
    with file_lock(_store_path(root)):
        store = load_store(root)
        changed = False
        for card in store.get("cards") or []:
            if card.get("tape_fit"):
                continue
            tape = load_tape(root, name=card.get("tape") or DEFAULT_TAPE)
            if tape is None or not tape["meta"].get("fit_hash"):
                continue
            built = tape["meta"].get("built_at")
            if built and str(built)[:10] > str(card.get("saved_at") or ""):
                continue
            card["tape"] = card.get("tape") or DEFAULT_TAPE
            card["tape_fit"] = tape["meta"]["fit_hash"]
            changed = True
        if changed:
            _write(store, root)


def score_saved(root: Path | None = None) -> dict[str, Any]:
    _backfill_tape_fit(root)
    store = load_store(root)
    names = dict(STRATEGIES)
    scored = []
    bundles: dict[tuple[str, int], tuple] = {}
    tapes: dict[str, dict[str, Any] | None] = {}
    ends: dict[str, str] = {}
    for spec in store.get("cards") or []:
        horizon = int(spec["horizon"])
        tape_name = spec.get("tape") or DEFAULT_TAPE
        if tape_name not in tapes:
            tapes[tape_name] = load_tape(root, name=tape_name)
        tape = tapes[tape_name]
        heading = {
            "id": spec["id"],
            "saved_at": spec["saved_at"],
            "name": spec.get("name") or names.get(spec["strategy"], spec["strategy"]),
            "strategy": spec["strategy"],
            "rule_label": names.get(spec["strategy"], spec["strategy"]),
            "horizon_days": horizon,
            "region": spec["region"],
            "place": LABELS.get(spec["region"], spec["region"]),
            "trials_at_save": spec.get("trials_at_save"),
            "tape": tape_name,
        }
        built = None
        if tape is not None and str(horizon) in tape["meta"]["horizons"]:
            key = (tape_name, horizon)
            if key not in bundles:
                bundles[key] = _cards(horizon, tape["npz"], tape["meta"])
            built, matrix, days = bundles[key]
        target = next((card for card in built or [] if card["region"] == spec["region"]), None)
        if target is None:
            scored.append(
                {
                    **heading,
                    "magnitude_min": None,
                    "label_version": None,
                    "refit_since_save": False,
                    "develop": {},
                    "rehearsal": {},
                    "since_save": {"mornings": 0},
                    "status": "tape not built",
                }
            )
            continue
        meta = tape["meta"]
        ends[tape_name] = _tape_end(meta)
        pred = _predict_region(spec, built, matrix, spec.get("knobs") or {})
        # The save-day morning had already begun when the card was saved.
        first_new = (pd.Timestamp(spec["saved_at"]) + pd.Timedelta(days=1)).date().isoformat()
        develop = _slice(days, horizon, "1990-01-01", "2009-01-01")
        rehearsal = _slice(days, horizon, "2009-01-01", first_new)
        prospective = _slice(days, horizon, first_new, None)
        dev = _row(target["region"], target["mag"], pred, target["y"], target["c"], develop, None)
        later = _row(target["region"], target["mag"], pred, target["y"], target["c"], rehearsal, None)
        ahead = _row(target["region"], target["mag"], pred, target["y"], target["c"], prospective, None)
        if ahead["mornings"] >= 8 and ahead["tests_agree"]:
            status = "holding"
        elif ahead["mornings"] >= 8:
            status = "not holding"
        else:
            status = "waiting"
        saved_fit = spec.get("tape_fit")
        scored.append(
            {
                **heading,
                "magnitude_min": target["mag"],
                "label_version": int(meta.get("label_version") or 1),
                "refit_since_save": bool(saved_fit) and saved_fit != meta.get("fit_hash"),
                "develop": dev,
                "rehearsal": later,
                "since_save": ahead,
                "status": status,
            }
        )
    reach = "; ".join(f"{name} ends {end}" for name, end in sorted(ends.items())) or "no tape is built yet"
    return {
        "ready": True,
        "trials": _trial_count(),
        "tape_end": max(ends.values()) if ends else None,
        "cards": scored,
        "note": (
            "A saved card keeps the knobs it had on the day it was saved. "
            "1990–2008 is where a rule is chosen. 2009 through the save day is a rehearsal, already looked at. "
            "Only mornings after the save day count as new evidence, once their window has closed inside the tape "
            f"({reach}). The weekly refresh rebuilds the tapes; 'refit since save' means the model under a card was "
            "refit from changed inputs after it was saved."
        ),
    }


def freeze_from_request(spec: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Store the desk choices so a later fit of the shrink factor cannot move this card."""
    from eqpred.eval.strategy_desk import score_strategy

    regions = ["global", "hikurangi", "alpine", "kermadec"] if spec.get("region") == "all" else [spec["region"]]
    saved = []
    for region in regions:
        report = score_strategy(
            horizon=int(spec["horizon"]),
            region=region,
            strategy=spec["strategy"],
            shrink=spec.get("shrink"),
            odds=spec.get("odds"),
            alarm_share=float(spec.get("alarm_share") or 0.2),
            rarity_n0=float(spec.get("rarity_n0") or 50),
            bins=int(spec.get("bins") or 5),
            agree_min=int(spec.get("agree_min") or 3),
            matched_k=spec.get("matched_k"),
            use_readings=bool(spec.get("use_readings")),
            lag_days=int(spec.get("lag_days") or 0),
            root=root,
            tape_name=spec.get("tape") or DEFAULT_TAPE,
        )
        if not report.get("ready"):
            raise ValueError(report.get("note") or "Replay is not ready.")
        choices = report["choices"]
        saved.append(
            save_card(
                {
                    "strategy": spec["strategy"],
                    "horizon": int(spec["horizon"]),
                    "region": region,
                    "tape": report["tape"]["name"],
                    "tape_fit": report["tape"]["fit_hash"],
                    "name": report.get("label") or spec["strategy"],
                    "knobs": {
                        "alarm_share": choices.get("alarm_share"),
                        "rarity_n0": choices.get("rarity_n0"),
                        "lag_days": choices.get("lag_days"),
                        "bins": choices.get("bins"),
                        "agree_min": choices.get("agree_min"),
                        "shrink": choices.get("shrink"),
                        "odds": choices.get("odds"),
                        "matched_k": choices.get("matched_k") or 40,
                        "use_readings": choices.get("use_readings"),
                    },
                },
                root,
            )
        )
    return {"saved": saved}
