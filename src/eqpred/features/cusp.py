from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.cusp_geom import cusp_weight
from eqpred.regions import REGIONS, region_centroid_lon
from eqpred.stamp import parse_utc

# Weight lasts 48h after the onset is on the wire. A window from origin
# would be empty: OMNI latency is already a day or more.
PERSIST = pd.Timedelta(hours=48)


def cusp_frame(issue_times: pd.Series, onsets: pd.DataFrame) -> pd.DataFrame:
    """As-of alignment of each issued box with the latest available storm-onset cusp."""
    out = pd.DataFrame({"issue_time": parse_utc(issue_times)})
    boxes = {rid: region_centroid_lon(region) for rid, region in REGIONS.items() if region.get("kind") == "bbox"}
    for rid in boxes:
        out[f"cusp_align_{rid}"] = 0.0
    if onsets is None or onsets.empty or "available_at" not in onsets.columns:
        return out
    work = onsets.copy()
    work["available_at"] = parse_utc(work["available_at"])
    work["observed_at"] = parse_utc(work["observed_at"])
    work = work.sort_values("available_at")
    left = out[["issue_time"]].sort_values("issue_time")
    merged = pd.merge_asof(
        left,
        work,
        left_on="issue_time",
        right_on="available_at",
        direction="backward",
    )
    fresh = merged["issue_time"] <= merged["available_at"] + PERSIST
    valid = merged["cusp_lon_deg"].notna() & fresh.fillna(False)
    cusp = merged["cusp_lon_deg"].to_numpy(dtype=float)
    aligned = merged[["issue_time"]].copy()
    for rid, lon in boxes.items():
        aligned[f"cusp_align_{rid}"] = np.where(valid.to_numpy(), cusp_weight(lon, cusp), 0.0)
    return out.drop(columns=[c for c in out.columns if c.startswith("cusp_align_")]).merge(
        aligned, on="issue_time", how="left"
    )
