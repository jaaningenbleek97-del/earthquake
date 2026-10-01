"""SISSLE Haast clips as a campaign foil. Not a daily leftover series."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eqpred.paths import data_dir
from eqpred.records import iso

WRITES_ISSUED_P = False
SOURCE = "das.haast"


def clip_dir(root: Path | None = None) -> Path:
    return data_dir(root) / "raw" / "das.haast" / "sissle-clips"


def load_das_foil(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "das_foil.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _clip_meta(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size}
    try:
        import h5py
    except ImportError:
        meta["hdf5"] = "h5py not installed"
        return meta
    try:
        with h5py.File(path, "r") as handle:
            meta["keys"] = [str(name) for name in list(handle.keys())[:12]]
            node = handle["DAS"] if "DAS" in handle else None
            if node is not None and hasattr(node, "shape"):
                meta["shape"] = [int(n) for n in node.shape]
                meta["dtype"] = str(node.dtype)
    except Exception as exc:
        meta["hdf5"] = str(exc)
    return meta


def run_das_foil(root: Path | None = None) -> dict[str, Any]:
    """Record the local clips. Four minutes cannot train leftover or issued p."""
    folder = clip_dir(root)
    files = sorted(folder.rglob("*.h5")) if folder.exists() else []
    events = sorted({path.parent.name for path in files})
    clips = [_clip_meta(path) for path in files]
    report = {
        "writes_issued_p": False,
        "id": "foil.das.sissle.v1",
        "source": SOURCE,
        "n_files": len(files),
        "events": events,
        "clips": clips,
        "trains_leftover": False,
        "scanned_at": iso(datetime.now(timezone.utc)),
        "note": (
            f"SISSLE Haast foil: {len(files)} local HDF5 minute(s) across {len(events)} event(s). "
            "Campaign sample only. Not trained into leftover and not in the published 30-day chance."
            if files
            else "No local SISSLE clips. Haast DAS stays a partnership archive, not issued p."
        ),
    }
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "das_foil.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
