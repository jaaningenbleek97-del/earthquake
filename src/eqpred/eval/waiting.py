from __future__ import annotations

"""How long until the next fresh shock, from gaps already on record.

This sits beside the 30-day chance. It does not change that chance.
"""

import math

import numpy as np
import pandas as pd

from eqpred.eval.decluster import decluster_causal
from eqpred.stamp import parse_utc

NS_PER_DAY = np.timedelta64(1, "D")


def _fresh_times(quakes: pd.DataFrame, region: dict, mag: float) -> tuple[np.ndarray, np.ndarray]:
    work = quakes.loc[quakes["magnitude"] >= mag].copy()
    if region.get("kind") == "bbox":
        from eqpred.regions import in_bbox

        work = work.loc[in_bbox(work, region)]
    if work.empty:
        empty = np.array([], dtype="datetime64[ns]")
        return empty, empty
    mains = decluster_causal(work)
    origins = parse_utc(mains["observed_at"]).to_numpy(dtype="datetime64[ns]")
    if "available_at" in mains.columns:
        available = parse_utc(mains["available_at"]).to_numpy(dtype="datetime64[ns]")
    else:
        available = origins
    order = np.argsort(origins, kind="mergesort")
    return origins[order], available[order]


def remaining_wait(
    origins: np.ndarray,
    available: np.ndarray,
    issue: np.datetime64,
) -> tuple[float | None, float | None, int]:
    """Expected days still to wait, days since the last known shock, and how many gaps were used.

    A shock counts only after it has been reported. A gap counts only after the later shock is known.
    """
    known = available <= issue
    if not np.any(known):
        return None, None, 0
    origins = origins[known]
    elapsed_origin = origins[origins <= issue]
    if elapsed_origin.size == 0:
        return None, None, 0
    since = float((issue - elapsed_origin[-1]) / NS_PER_DAY)
    if origins.size < 2:
        return None, since, 0
    gaps = (np.diff(origins) / NS_PER_DAY).astype(float)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    longer = gaps[gaps > since]
    if longer.size == 0:
        return None, since, int(gaps.size)
    return float(np.mean(longer - since)), since, int(longer.size)


def wait_plain(wait_days: float | None, since_days: float | None) -> str | None:
    since = ""
    if since_days is not None and math.isfinite(since_days):
        since = f" {int(round(since_days))} days since the last one."
    if wait_days is None or not math.isfinite(wait_days):
        if not since:
            return None
        return f"Longer than the gaps already on record.{since} Not part of the 30-day chance."
    n = max(0, int(round(wait_days)))
    if n < 1:
        lead = "The next fresh shock is inside a day, from the gaps already on record."
    else:
        lead = f"About {n} days until the next fresh shock, from the gaps already on record."
    return f"{lead}{since} Not part of the 30-day chance."


def catalog_wait(quakes: pd.DataFrame, region: dict, mag: float, issue: pd.Timestamp) -> dict:
    origins, available = _fresh_times(quakes, region, mag)
    when = np.datetime64(parse_utc(pd.Series([issue])).iloc[0].to_datetime64(), "ns")
    wait_days, since_days, n_gaps = remaining_wait(origins, available, when)
    return {
        "wait_days": wait_days,
        "wait_since_days": since_days,
        "wait_n_gaps": n_gaps,
        "wait_plain": wait_plain(wait_days, since_days),
        "writes_issued_p": False,
    }


SWARM_DAYS = 10
SWARM_MIN_EVENTS = 4
SWARM_SHARE = 0.5
SWARM_FLOOR = 3.0


def _issue_ns(issue: pd.Timestamp) -> np.datetime64:
    return np.datetime64(parse_utc(pd.Series([issue])).iloc[0].to_datetime64(), "ns")


def _box_events(quakes: pd.DataFrame, region: dict, mag_min: float) -> pd.DataFrame:
    work = quakes.loc[quakes["magnitude"] >= mag_min].copy()
    if region.get("kind") == "bbox":
        from eqpred.regions import in_bbox

        work = work.loc[in_bbox(work, region)]
    return work


def is_aftershock_swarm(quakes: pd.DataFrame, region: dict, mag: float, issue: pd.Timestamp) -> bool:
    """True when most recent quakes are aftershocks of a shock already reported."""
    work = _box_events(quakes, region, SWARM_FLOOR)
    if len(work) < SWARM_MIN_EVENTS:
        return False
    kept = decluster_causal(work)
    aftershock = ~work.index.isin(kept.index)
    when = _issue_ns(issue)
    start = when - np.timedelta64(SWARM_DAYS, "D")
    obs = parse_utc(work["observed_at"]).to_numpy(dtype="datetime64[ns]")
    if "available_at" in work.columns:
        avail = parse_utc(work["available_at"]).to_numpy(dtype="datetime64[ns]")
    else:
        avail = obs
    recent = (avail <= when) & (obs <= when) & (obs >= start)
    if int(recent.sum()) < SWARM_MIN_EVENTS:
        return False
    parent = work["magnitude"].to_numpy(dtype=float) >= mag
    if not np.any(recent & parent):
        return False
    return bool(np.mean(np.asarray(aftershock)[recent]) >= SWARM_SHARE)


def swarm_mornings(
    quakes: pd.DataFrame,
    region: dict,
    mag: float,
    days: pd.DatetimeIndex,
) -> np.ndarray:
    """One flag per morning. Aftershocks of a shock already reported do not count as a new cycle."""
    flags = np.zeros(len(days), dtype=bool)
    work = _box_events(quakes, region, SWARM_FLOOR)
    if work.empty:
        return flags
    kept = decluster_causal(work)
    aftershock = ~work.index.isin(kept.index)
    obs = parse_utc(work["observed_at"]).to_numpy(dtype="datetime64[ns]")
    if "available_at" in work.columns:
        avail = parse_utc(work["available_at"]).to_numpy(dtype="datetime64[ns]")
    else:
        avail = obs
    parent = work["magnitude"].to_numpy(dtype=float) >= mag
    order = np.argsort(obs, kind="mergesort")
    obs_s = obs[order]
    avail_s = avail[order]
    after_s = np.asarray(aftershock, dtype=bool)[order]
    parent_s = parent[order]
    issue = pd.DatetimeIndex(days).tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    left = np.searchsorted(obs_s, issue - np.timedelta64(SWARM_DAYS, "D"), side="left")
    right = np.searchsorted(obs_s, issue, side="right")
    for i, when in enumerate(issue):
        sl = slice(int(left[i]), int(right[i]))
        if right[i] - left[i] < SWARM_MIN_EVENTS:
            continue
        reported = avail_s[sl] <= when
        if int(reported.sum()) < SWARM_MIN_EVENTS:
            continue
        if not np.any(reported & parent_s[sl]):
            continue
        if float(np.mean(after_s[sl][reported])) >= SWARM_SHARE:
            flags[i] = True
    return flags


def odd_box(quakes: pd.DataFrame, issue: pd.Timestamp) -> dict:
    """Which issued box is furthest from its own usual gap between first shocks."""
    from eqpred.eval.mainshock import ISSUED_TARGETS, _place
    from eqpred.regions import REGIONS

    when = _issue_ns(issue)
    rows = []
    for region_id, mag in ISSUED_TARGETS:
        origins, available = _fresh_times(quakes, REGIONS[region_id], mag)
        known = available <= when
        if not np.any(known):
            continue
        origins = origins[known]
        origins = origins[origins <= when]
        if origins.size < 3:
            continue
        gaps = (np.diff(origins) / NS_PER_DAY).astype(float)
        gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
        if gaps.size < 2:
            continue
        usual = float(np.median(gaps))
        since = float((when - origins[-1]) / NS_PER_DAY)
        if usual <= 0 or not math.isfinite(since):
            continue
        ratio = since / usual
        rows.append(
            {
                "region": region_id,
                "label": _place(region_id),
                "since_days": since,
                "usual_days": usual,
                "ratio": ratio,
                "oddness": abs(math.log(ratio)),
            }
        )
    if not rows:
        return {"plain": None, "writes_issued_p": False}
    winner = max(rows, key=lambda row: row["oddness"])
    how = "longer than usual" if winner["ratio"] >= 1 else "shorter than usual"
    plain = (
        f"{winner['label']} is the odd one out among first shocks. "
        f"{int(round(winner['since_days']))} days since the last one, "
        f"against a usual gap of about {int(round(winner['usual_days']))} days. "
        f"That wait is {how}. Aftershocks do not count. This does not change the 30-day chance."
    )
    return {**winner, "plain": plain, "writes_issued_p": False}
