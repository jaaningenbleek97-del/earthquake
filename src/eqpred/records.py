from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from eqpred.models.attribution import ensure_attrs


def utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso(dt: datetime) -> str:
    return utc(dt).isoformat().replace("+00:00", "Z")


def observation(
    *,
    source: str,
    observed_at: datetime,
    available_at: datetime,
    payload: dict[str, Any],
    license: str,
    ingest_run_id: str,
    geom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "observed_at": iso(observed_at),
        "available_at": iso(available_at),
        "geom": geom or {"kind": "global"},
        "payload": payload,
        "license": license,
        "ingest_run_id": ingest_run_id,
    }


def prediction(
    *,
    prediction_id: str,
    issued_at: datetime,
    valid_from: datetime,
    valid_to: datetime,
    horizon_days: float,
    region: dict[str, Any],
    magnitude_min: float,
    kind: str,
    value: float,
    model_id: str,
    drivers: list,
    rationale: str | None = None,
    competitor_etas_p: float | None = None,
    competitor_climatology_p: float | None = None,
    registry_id: str | None = None,
    code_hash: str | None = None,
    written_at: datetime | None = None,
    card_kind: str | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "prediction_id": prediction_id,
        "issued_at": iso(issued_at),
        "valid_from": iso(valid_from),
        "valid_to": iso(valid_to),
        "horizon_days": horizon_days,
        "region": region,
        "magnitude_min": magnitude_min,
        "kind": kind,
        "value": float(value),
        "model_id": model_id,
        "drivers": ensure_attrs(drivers),
    }
    if rationale:
        rec["rationale"] = rationale
    if competitor_etas_p is not None:
        rec["competitor_etas_p"] = float(competitor_etas_p)
    if competitor_climatology_p is not None:
        rec["competitor_climatology_p"] = float(competitor_climatology_p)
    if registry_id:
        rec["registry_id"] = registry_id
    if code_hash:
        rec["code_hash"] = code_hash
    if written_at is not None:
        rec["written_at"] = iso(written_at)
    if card_kind:
        rec["card_kind"] = card_kind
    return rec
