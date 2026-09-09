"""Discretionary human cards. Same ledger as machines. Never used in fit."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from eqpred.models.attribution import fusion_attrs
from eqpred.models.predict import BOX_MAG, GLOBAL_MAG
from eqpred.paths import prediction_dir
from eqpred.records import prediction
from eqpred.regions import REGIONS
from eqpred.stamp import utc_ts

HUMAN_MODEL = "human"


def write_human_card(
    *,
    region_id: str,
    magnitude_min: float | None = None,
    horizon_days: int = 7,
    kind: str = "probability",
    value: float,
    note: str = "",
    issued_at: str | None = None,
    root: Path | None = None,
) -> dict:
    if region_id not in REGIONS:
        raise ValueError(f"unknown region {region_id}")
    region = REGIONS[region_id]
    if region.get("kind") == "bbox" and region.get("issued", True) is False:
        raise ValueError(f"{region_id} is transfer-only; human cards stay on issued boxes")
    mag = float(magnitude_min if magnitude_min is not None else (GLOBAL_MAG if region_id == "global" else BOX_MAG.get(region_id, 5.0)))
    issued = utc_ts(issued_at) if issued_at else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    issued = issued.to_pydatetime() if hasattr(issued, "to_pydatetime") else issued
    valid_to = issued + timedelta(days=int(horizon_days))
    card = prediction(
        prediction_id=f"{issued.date()}-{region_id}-human-{uuid4().hex[:8]}",
        issued_at=issued,
        valid_from=issued,
        valid_to=valid_to,
        horizon_days=int(horizon_days),
        region=region,
        magnitude_min=mag,
        kind=kind,
        value=float(value),
        model_id=HUMAN_MODEL,
        drivers=fusion_attrs({"human": 1.0}),
        rationale=note or "human discretionary card",
    )
    dest = prediction_dir(root)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"batch-human-{issued.date().isoformat()}.json"
    existing: list = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    existing.append(card)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    card["_path"] = str(path)
    return card
