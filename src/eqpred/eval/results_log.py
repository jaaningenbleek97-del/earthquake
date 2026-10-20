"""One append-only record of lab results. The big JSON files stay where they are.

Each line of results_log.jsonl is one snapshot. results_log.txt is the same
history in plain language. Nothing here writes issued probability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eqpred.eval.honest import counts_as_prospective
from eqpred.paths import data_dir
from eqpred.records import iso


def results_log_path(root: Path | None = None) -> Path:
    return data_dir(root) / "eval" / "results_log.jsonl"


def results_text_path(root: Path | None = None) -> Path:
    return data_dir(root) / "eval" / "results_log.txt"


def record_results(root: Path | None = None) -> dict[str, Any]:
    """Append a snapshot of every result file and refresh the plain-language log."""
    entry = _snapshot(root)
    path = results_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
    _write_text(root)
    return entry


def _snapshot(root: Path | None) -> dict[str, Any]:
    folder = data_dir(root) / "eval"
    files = []
    if folder.exists():
        for path in sorted(folder.glob("*.json")):
            files.append(_file_row(path))
    return {
        "recorded_at": iso(datetime.now(timezone.utc)),
        "writes_issued_p": False,
        "git": _git_rev(root),
        "cards": _cards(folder / "mainshock.json"),
        "scores": _region_scores(folder / "mainshock.json"),
        "frozen": _frozen(folder / "frozen"),
        "ledger": _ledger(folder / "frozen" / "ledger.json"),
        "missing": _missing(root),
        "files": files,
        "note": (
            "Snapshot of the scores, hunt rows, frozen cards, and result files. "
            "This log does not change a card."
        ),
    }


def _file_row(path: Path) -> dict[str, Any]:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    row: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "written_at": iso(stamp),
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        row["readable"] = False
        return row
    if isinstance(payload, dict):
        row["headline"] = _headline(path.name, payload)
    elif isinstance(payload, list):
        row["headline"] = {"rows": len(payload)}
    return row


def _headline(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name.startswith("hunt"):
        return _hunt_headline(payload)
    if name == "mainshock.json":
        return {"see": "cards and scores on this snapshot"}
    if name == "promote.json":
        return {
            "pass": payload.get("pass"),
            "reason": payload.get("reason"),
            "writes_issued_p": payload.get("writes_issued_p"),
            "walk_forward_lift": payload.get("walk_forward_lift"),
            "locked_brier": payload.get("locked_brier"),
            "locked_etas_brier": payload.get("locked_etas_brier"),
            "note": payload.get("note"),
        }
    kept = {}
    for key in ("note", "writes_issued_p", "n_beats_climatology", "scanned_at", "train_end", "holdout_start"):
        if key in payload and not isinstance(payload[key], (dict, list)):
            kept[key] = payload[key]
    return kept


def _hunt_headline(payload: dict[str, Any]) -> dict[str, Any]:
    confirmed = [_trial(item) for item in payload.get("confirmed") or []]
    replayed = [_trial(item) for item in payload.get("replayed") or []]
    headline: dict[str, Any] = {
        "fresh": payload.get("fresh"),
        "sequence": payload.get("sequence"),
        "quiet_slice": payload.get("quiet_slice"),
        "train_end": payload.get("train_end"),
        "holdout_start": payload.get("holdout_start"),
        "n_knobs": payload.get("n_knobs"),
        "n_tests": payload.get("n_tests"),
        "n_discovered": payload.get("n_discovered"),
        "n_confirmed": payload.get("n_confirmed"),
        "n_replayed": payload.get("n_replayed"),
        "writes_issued_p": payload.get("writes_issued_p"),
        "note": payload.get("note"),
        "confirmed": confirmed,
        "replayed": replayed,
        "quiet_columns_confirmed": [
            item for item in confirmed if str(item.get("driver") or "").startswith(("qres_", "laic_", "aardvark_"))
        ],
    }
    honest = payload.get("honest")
    if isinstance(honest, dict):
        headline["honest"] = {
            "n_tests": honest.get("n_tests"),
            "n_confirmed": honest.get("n_confirmed"),
            "n_beats_climate": honest.get("n_beats_climate"),
            "n_confirmed_quiet": honest.get("n_confirmed_quiet"),
            "n_confirmed_m8": honest.get("n_confirmed_m8"),
            "n_replayed": honest.get("n_replayed"),
            "confirmed": [_trial(item) for item in honest.get("confirmed") or []],
            "beats_climate": [_trial(item) for item in honest.get("beats_climate") or []],
            "quiet": [_trial(item) for item in honest.get("quiet") or []],
            "m8": [_trial(item) for item in honest.get("m8") or []],
            "families": [_trial(item) for item in honest.get("families") or []],
        }
    return headline


def _trial(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "region": item.get("region"),
        "magnitude_min": item.get("magnitude_min"),
        "driver": item.get("driver"),
        "family": item.get("family"),
        "lag_days": item.get("lag_days"),
        "train_rho": item.get("train_rho"),
        "confirm_rho": item.get("confirm_rho"),
        "confirm_p": item.get("confirm_p"),
        "replay_rho": item.get("replay_rho"),
        "replayed": item.get("replayed"),
        "train_lift": item.get("train_lift"),
    }


def _cards(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _cards_from_pack(payload)


def _cards_from_pack(payload: dict[str, Any]) -> dict[str, Any]:
    def brief(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            out.append(
                {
                    "region": row.get("region"),
                    "label": row.get("label"),
                    "horizon_days": row.get("horizon_days"),
                    "magnitude_min": row.get("magnitude_min"),
                    "chance_plain": row.get("chance_plain"),
                    "climatology_chance_plain": row.get("climatology_chance_plain"),
                    "issue_day": row.get("issue_day"),
                    "p": row.get("p"),
                    "climatology_p": row.get("climatology_p"),
                    "information_gain": row.get("information_gain"),
                    "families": row.get("families") or row.get("why"),
                    "issued_from": row.get("issued_from"),
                    "lineup_plain": row.get("lineup_plain"),
                    "group_plain": row.get("group_plain"),
                    "groups_up": row.get("groups_up"),
                    "groups_down": row.get("groups_down"),
                    "sentence": row.get("sentence"),
                }
            )
        return out

    short = payload.get("short_horizons") or []
    return {
        "issue_day": next((row.get("issue_day") for row in payload.get("week") or [] if row.get("issue_day")), None),
        "day_30": brief(payload.get("week") or []),
        "day_7": brief([row for row in short if int(row.get("horizon_days") or 0) == 7]),
        "day_3": brief([row for row in short if int(row.get("horizon_days") or 0) == 3]),
    }


def _frozen(folder: Path) -> dict[str, Any]:
    recipe = folder / "recipe.json"
    if not recipe.exists():
        return {}
    try:
        payload = json.loads(recipe.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "locked_at": payload.get("locked_at"),
        "issue_day": payload.get("issue_day"),
        "horizon_days": payload.get("horizon_days"),
        "week": [
            {
                "region": row.get("region"),
                "chance_plain": row.get("chance_plain"),
                "climatology_chance_plain": row.get("climatology_chance_plain"),
                "families": row.get("families") or row.get("why"),
                "p": row.get("p"),
                "climatology_p": row.get("climatology_p"),
            }
            for row in payload.get("week") or []
        ],
    }


def _region_scores(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = []
    for region in payload.get("regions") or []:
        rows.append(
            {
                "region": region.get("region"),
                "label": region.get("label"),
                "information_gain": region.get("information_gain"),
                "brier": region.get("brier"),
                "climatology_brier": region.get("climatology_brier"),
                "beats_climatology": region.get("beats_climatology"),
                "lift": region.get("lift"),
                "n_hits": region.get("n_hits"),
                "n_score": region.get("n_score"),
                "issued_from": region.get("issued_from"),
            }
        )
    return rows


def _ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = []
    for entry in payload.get("entries") or []:
        rows.append(
            {
                "region": entry.get("region"),
                "horizon_days": entry.get("horizon_days"),
                "issue_day": entry.get("issue_day") or entry.get("issued_at"),
                "p": entry.get("p"),
                "chance_plain": entry.get("chance_plain"),
                "prospective": counts_as_prospective(entry),
                "hit": entry.get("hit"),
                "scored": entry.get("scored"),
            }
        )
    return rows


def _missing(root: Path | None) -> dict[str, Any]:
    from eqpred.paths import raw_dir

    raw = raw_dir(root)
    aardvark = list((raw / "vlf.aardvark").glob("*.csv")) if (raw / "vlf.aardvark").exists() else []
    unwin = list((raw / "superdarn.unwin").rglob("*.unw.fitacf.bz2")) if (raw / "superdarn.unwin").exists() else []
    return {
        "nwc_dunedin_aardvark_files": len(aardvark),
        "unwin_gate_files": len(unwin),
        "cses_on_disk": (raw / "esa.cses").exists(),
        "note": (
            "NWC-Dunedin amplitude and phase, Unwin gate files, and a fault mesh "
            "are the inputs still missing. Quiet residuals are already on the daily table."
        ),
    }


def _git_rev(root: Path | None) -> str | None:
    import subprocess

    try:
        found = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root or Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if found.returncode != 0:
        return None
    return found.stdout.strip() or None


def _write_text(root: Path | None) -> None:
    lines: list[str] = []
    path = results_log_path(root)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        entry = json.loads(raw)
        lines.extend(_entry_text(entry))
        lines.append("")
    results_text_path(root).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _entry_text(entry: dict[str, Any]) -> list[str]:
    lines = [
        f"Recorded {entry.get('recorded_at')}",
        f"Code {entry.get('git') or '—'}. This record does not change a card.",
        "",
        "Frozen 30-day cards",
    ]
    frozen = entry.get("frozen") or {}
    lines.append(f"Locked {frozen.get('locked_at') or '—'} for issue morning {frozen.get('issue_day') or '—'}.")
    for row in frozen.get("week") or []:
        families = ", ".join(row.get("families") or [])
        lines.append(
            f"  {row.get('region')}: {row.get('chance_plain')} , usual {row.get('climatology_chance_plain')}"
            + (f" . {families}" if families else "")
        )
    lines.extend(["", "Later-year scores"])
    for row in entry.get("scores") or []:
        lines.append(
            f"  {row.get('label') or row.get('region')}: information gain {row.get('information_gain')}, "
            f"error {row.get('brier')} , usual error {row.get('climatology_brier')}, "
            f"beats usual rate {row.get('beats_climatology')}"
        )
    cards = entry.get("cards") or {}
    lines.extend(["", "Saved 7-day cards"])
    for row in cards.get("day_7") or []:
        lines.append(
            f"  {row.get('label') or row.get('region')}: {row.get('chance_plain')} , usual {row.get('climatology_chance_plain')}"
        )
    lines.extend(["", "Saved 3-day cards"])
    for row in cards.get("day_3") or []:
        lines.append(
            f"  {row.get('label') or row.get('region')}: {row.get('chance_plain')} , usual {row.get('climatology_chance_plain')}"
        )
        if row.get("group_plain"):
            lines.append(f"    {row.get('group_plain')}")
        if row.get("information_gain") is not None:
            lines.append(f"    information gain {row.get('information_gain')}")
    if entry.get("ledger"):
        lines.extend(["", "Frozen ledger"])
        for row in entry["ledger"]:
            lines.append(
                f"  {row.get('region')} {row.get('horizon_days')} days {row.get('issue_day')}: "
                f"p {row.get('p')} prospective {row.get('prospective')} hit {row.get('hit')}"
            )
    missing = entry.get("missing") or {}
    if missing:
        lines.extend(
            [
                "",
                "Still missing",
                f"  NWC-Dunedin files {missing.get('nwc_dunedin_aardvark_files')}",
                f"  Unwin gate files {missing.get('unwin_gate_files')}",
                f"  CSES on disk {missing.get('cses_on_disk')}",
            ]
        )
    lines.extend(["", "Result files"])
    for row in entry.get("files") or []:
        head = row.get("headline") or {}
        if "n_confirmed" not in head:
            if row["file"] == "promote.json":
                lines.append(f"  promote.json: pass {head.get('pass')}. {head.get('reason') or ''}")
            elif head.get("note"):
                lines.append(f"  {row['file']}: {head.get('note')}")
            else:
                lines.append(f"  {row['file']} written {row.get('written_at')}")
            continue
        kind = "first-shock" if head.get("fresh") else "aftershock-residual"
        quiet = " quiet mornings" if head.get("quiet_slice") else ""
        lines.append(
            f"  {row['file']} ({kind}{quiet}): "
            f"confirmed {head.get('n_confirmed')}, replayed {head.get('n_replayed')}, "
            f"tests {head.get('n_tests')}"
        )
        confirmed = head.get("confirmed") or head.get("confirmed_top") or []
        for item in confirmed:
            lines.append(
                f"    {item.get('region')} M{item.get('magnitude_min')} {item.get('driver')} "
                f"lag {item.get('lag_days')} confirm {item.get('confirm_rho')} replay {item.get('replay_rho')}"
            )
    return lines
