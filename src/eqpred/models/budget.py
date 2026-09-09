from __future__ import annotations

"""Monthly alarm budget.

A head that is on 40% of days is a weather channel. SPF already has a
lifetime 20% train-era strength cutoff; this is the second gate: at most
N issued alarm-days per (model, region, calendar month), spent in date
order. Caps are locked constants. Do not fit them to holdout hits.

Quiet specialists get a tighter cap because they are only allowed to
speak when ETAS is boring. Probability cards are not budgeted.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from eqpred.paths import model_dir
from eqpred.stamp import utc_ts

# ~20% of a 30-day month — same philosophy as davidson.ALARM_BUDGET.
SPF_DAYS_PER_MONTH = 6
QUIET_DAYS_PER_MONTH = 2
ANTI_DAYS_PER_MONTH = 4

DEFAULT_CAPS = {
    "alarm.spf.v2": SPF_DAYS_PER_MONTH,
    "alarm.quiet.v1": QUIET_DAYS_PER_MONTH,
    "alarm.quiet_spf.v1": QUIET_DAYS_PER_MONTH,
    "anti.quiet.v1": ANTI_DAYS_PER_MONTH,
}


def month_key(when: datetime | str | pd.Timestamp) -> str:
    ts = utc_ts(when)
    return f"{ts.year:04d}-{ts.month:02d}"


class AlertBudget:
    """Walk-forward monthly quota. First-come is what a live issuer can do."""

    def __init__(self, days_per_month: dict[str, int] | None = None):
        self.caps = {**DEFAULT_CAPS, **(days_per_month or {})}
        self.spent: dict[tuple[str, str, str], int] = {}

    @classmethod
    def from_config(cls, cfg: dict | None) -> AlertBudget:
        payload = cfg or {}
        raw = payload.get("days_per_month") or {}
        caps = {str(k): int(v) for k, v in raw.items()}
        return cls(caps)

    def cap(self, model_id: str) -> int | None:
        value = self.caps.get(model_id)
        return None if value is None else int(value)

    def used(self, model_id: str, region_id: str, when: datetime | str | pd.Timestamp) -> int:
        return int(self.spent.get((model_id, region_id, month_key(when)), 0))

    def remember(self, model_id: str, region_id: str, when: datetime | str | pd.Timestamp) -> None:
        """Count a past issued alarm. Used when appending a later window."""
        if self.cap(model_id) is None:
            return
        key = (model_id, region_id, month_key(when))
        self.spent[key] = self.spent.get(key, 0) + 1

    def seed_from_cards(
        self,
        cards: list[dict],
        *,
        exclude_days: set[pd.Timestamp] | None = None,
    ) -> int:
        """Replay already-issued alarm spend so an append cannot double-spend a month."""
        from eqpred.regions import REGIONS

        names = {str(meta.get("name")): rid for rid, meta in REGIONS.items()}
        skip = exclude_days or set()
        n = 0
        for card in cards:
            if card.get("kind") != "alarm":
                continue
            issued = utc_ts(card["issued_at"])
            if issued.tz_convert("UTC").normalize() in skip:
                continue
            name = (card.get("region") or {}).get("name")
            region_id = names.get(name, name)
            if not region_id:
                continue
            self.remember(str(card.get("model_id") or ""), str(region_id), issued)
            n += 1
        return n

    def spend(self, model_id: str, region_id: str, when: datetime | str | pd.Timestamp) -> bool:
        """Return True and consume one slot, or False if that month is spent."""
        limit = self.cap(model_id)
        if limit is None:
            return True
        key = (model_id, region_id, month_key(when))
        if self.spent.get(key, 0) >= limit:
            return False
        self.spent[key] = self.spent.get(key, 0) + 1
        return True


def save_budget(root: Path | None = None) -> dict:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "issued": "monthly_cap_walk_forward",
        "days_per_month": dict(DEFAULT_CAPS),
        "note": (
            "Locked constants. SPF cap matches the 20% train-era day budget "
            "(~6 days in a 30-day month). Quiet specialists get 2. "
            "Anti-prediction (box stays quiet) gets 4. "
            "Not fit to holdout hits. Spent in date order."
        ),
    }
    (folder / "budget.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_budget(root: Path | None = None) -> dict:
    path = model_dir(root) / "budget.json"
    if not path.exists():
        return {
            "issued": "monthly_cap_walk_forward",
            "days_per_month": dict(DEFAULT_CAPS),
        }
    return json.loads(path.read_text(encoding="utf-8"))
