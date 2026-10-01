"""Read keys that were renamed after the leftover → fresh split."""

from typing import Any


def saved_fresh(row: dict[str, Any] | None, key: str):
    """Read a sealed-report value. Older files still use leftover_*."""
    if not row:
        return None
    if key in row:
        return row.get(key)
    return row.get(key.replace("fresh_", "leftover_", 1))
