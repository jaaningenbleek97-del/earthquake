"""Version the issued brain.

A later fit may write new coefficients. It must not erase the JSON that
already issued cards. Each version is model files + a hash of the code that
turns those files into a number + the feature-column recipe.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eqpred.paths import feature_dir, model_dir, prediction_batches, repo_root
from eqpred.records import iso

MODEL_FILES = (
    "etas.json",
    "modulator.json",
    "quiet.json",
    "gr.json",
    "calibration.json",
    "nowcast.json",
    "coulomb.json",
    "cusp.json",
    "wave2.json",
    "migration.json",
    "anti.json",
    "budget.json",
)

# Source that can change issued p without touching the JSON brain.
CODE_RELATIVE = (
    "features/build.py",
    "features/nowcast.py",
    "features/coulomb.py",
    "features/cusp.py",
    "features/davidson.py",
    "features/migration.py",
    "models/predict.py",
    "models/etas.py",
    "models/gr.py",
    "models/calibrate.py",
    "models/nowcast.py",
    "models/coulomb.py",
    "models/cusp.py",
    "models/wave2.py",
    "models/migration.py",
    "models/anti.py",
    "models/budget.py",
    "models/quiet.py",
    "models/modulator.py",
    "models/attribution.py",
    "catalog.py",
    "regions.py",
    "records.py",
)


def registry_dir(root: Path | None = None) -> Path:
    path = model_dir(root) / "registry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path(root: Path | None = None) -> Path:
    return model_dir(root) / "registry.json"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _digest(parts: list[bytes]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part)
    return hasher.hexdigest()


def read_brain(root: Path | None = None) -> dict[str, Any]:
    folder = model_dir(root)
    out: dict[str, Any] = {}
    for name in MODEL_FILES:
        path = folder / name
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def brain_hash(brain: dict[str, Any] | None = None, root: Path | None = None) -> str:
    payload = brain if brain is not None else read_brain(root)
    parts: list[bytes] = []
    for name in MODEL_FILES:
        if name in payload:
            parts.append(name.encode("utf-8"))
            parts.append(_canonical(payload[name]))
    return _digest(parts) if parts else ""


def code_hash() -> str:
    base = _package_root()
    parts: list[bytes] = []
    for rel in CODE_RELATIVE:
        path = base / rel
        if not path.exists():
            continue
        parts.append(rel.encode("utf-8"))
        parts.append(path.read_bytes().replace(b"\r\n", b"\n"))
    return _digest(parts)


def feature_recipe(root: Path | None = None) -> dict[str, Any]:
    path = feature_dir(root) / "global_daily.parquet"
    columns: list[str] = []
    if path.exists():
        import pandas as pd

        frame = pd.read_parquet(path)
        columns = [str(col) for col in frame.columns if col != "issue_time"]
    train_end = None
    cal = model_dir(root) / "calibration.json"
    if cal.exists():
        train_end = json.loads(cal.read_text(encoding="utf-8")).get("train_end")
    return {
        "builder": "eqpred.features.build.build_features",
        "columns": columns,
        "train_end": train_end or "2018-12-31",
    }


def recipe_hash(recipe: dict[str, Any]) -> str:
    return _digest([_canonical({"builder": recipe.get("builder"), "columns": recipe.get("columns"), "train_end": recipe.get("train_end")})])


def _identity(brain: dict[str, Any], recipe: dict[str, Any], code: str) -> str:
    return _digest(
        [
            brain_hash(brain).encode("utf-8"),
            code.encode("utf-8"),
            recipe_hash(recipe).encode("utf-8"),
        ]
    )


def load_index(root: Path | None = None) -> dict[str, Any]:
    path = _index_path(root)
    if not path.exists():
        return {"current": None, "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("current", None)
    payload.setdefault("entries", [])
    return payload


def _save_index(index: dict[str, Any], root: Path | None = None) -> None:
    _index_path(root).write_text(json.dumps(index, indent=2), encoding="utf-8")


def load_current(root: Path | None = None) -> dict[str, Any] | None:
    index = load_index(root)
    current = index.get("current")
    if not current:
        return None
    for entry in index.get("entries") or []:
        if entry.get("id") == current:
            return entry
    return None


def _write_snapshot(entry_id: str, brain: dict[str, Any], root: Path | None = None) -> Path:
    dest = registry_dir(root) / entry_id
    dest.mkdir(parents=True, exist_ok=True)
    for name, payload in brain.items():
        (dest / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def register_current(root: Path | None = None) -> dict[str, Any]:
    """Snapshot the active model dir if this identity is new. Never deletes an old id."""
    brain = read_brain(root)
    if not brain:
        raise FileNotFoundError("no model JSON to register; run eqpred fit first")
    recipe = feature_recipe(root)
    code = code_hash()
    ident = _identity(brain, recipe, code)
    index = load_index(root)
    for entry in index["entries"]:
        if entry.get("identity") == ident:
            index["current"] = entry["id"]
            _save_index(index, root)
            return entry
    created = datetime.now(timezone.utc)
    entry_id = created.strftime("%Y%m%dT%H%M%SZ") + "-" + ident[:8]
    entry = {
        "id": entry_id,
        "identity": ident,
        "created_at": iso(created),
        "code_hash": code,
        "brain_hash": brain_hash(brain),
        "feature_recipe": recipe,
        "files": [name for name in MODEL_FILES if name in brain],
        "model_ids": [
            "fusion.etas_x_mod.v1",
            "alarm.spf.v2",
            "alarm.quiet.v1",
            "alarm.quiet_spf.v1",
            "anti.quiet.v1",
        ],
    }
    _write_snapshot(entry_id, brain, root)
    (registry_dir(root) / entry_id / "manifest.json").write_text(
        json.dumps(entry, indent=2), encoding="utf-8"
    )
    index["entries"].append(entry)
    index["current"] = entry_id
    _save_index(index, root)
    return entry


def preserve_current(root: Path | None = None) -> dict[str, Any] | None:
    """Register the brain that is about to be overwritten. Returns None if empty."""
    if not read_brain(root):
        return None
    return register_current(root)


def ensure_registered(root: Path | None = None) -> dict[str, Any]:
    """Return the registry entry for the active brain, creating it if needed."""
    return register_current(root)


def write_batch_meta(batch_path: Path, entry: dict[str, Any], n_cards: int) -> Path:
    meta = {
        "registry_id": entry["id"],
        "code_hash": entry["code_hash"],
        "brain_hash": entry["brain_hash"],
        "feature_recipe": entry.get("feature_recipe"),
        "n_cards": int(n_cards),
        "pinned_at": iso(datetime.now(timezone.utc)),
    }
    dest = batch_path.with_name(batch_path.stem + ".meta.json")
    dest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def pin_unpinned_batches(root: Path | None = None, entry: dict[str, Any] | None = None) -> list[str]:
    """Attach a sidecar to existing batches that have no pin yet."""
    pin = entry or ensure_registered(root)
    written: list[str] = []
    for path in prediction_batches(root):
        meta = path.with_name(path.stem + ".meta.json")
        if meta.exists():
            continue
        cards = json.loads(path.read_text(encoding="utf-8"))
        write_batch_meta(path, pin, n_cards=len(cards) if isinstance(cards, list) else 0)
        written.append(str(meta))
    return written


def stamp_cards(cards: list[dict], entry: dict[str, Any]) -> list[dict]:
    for card in cards:
        card["registry_id"] = entry["id"]
        card["code_hash"] = entry["code_hash"]
    return cards


def registry_report(root: Path | None = None) -> dict[str, Any]:
    index = load_index(root)
    current = load_current(root)
    return {
        "current": current,
        "n_versions": len(index.get("entries") or []),
        "entries": [
            {
                "id": row.get("id"),
                "created_at": row.get("created_at"),
                "brain_hash": row.get("brain_hash"),
                "code_hash": row.get("code_hash"),
                "train_end": (row.get("feature_recipe") or {}).get("train_end"),
                "n_columns": len((row.get("feature_recipe") or {}).get("columns") or []),
            }
            for row in index.get("entries") or []
        ],
        "repo": str(repo_root()),
    }
