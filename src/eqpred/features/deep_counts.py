"""Depth-band and mechanism counts a morning could already have known.

Interface is 8–60 km. Slab is 70 km and deeper. The count stays below the
card's own magnitude, so the card is not predicting itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.regions import REGIONS, in_bbox
from eqpred.stamp import parse_utc

IFACE_MIN_KM = 8.0
IFACE_MAX_KM = 60.0
SLAB_MIN_KM = 70.0
WINDOW_DAYS = 30


def window_counts(
    observed: pd.Series,
    available: pd.Series,
    issue: pd.Series,
    window_days: int = WINDOW_DAYS,
) -> np.ndarray:
    """How many events fall in the 30 days before each morning, once they are public."""
    out = np.zeros(len(issue), dtype=float)
    if observed is None or len(observed) == 0 or len(issue) == 0:
        return out
    obs = parse_utc(observed).to_numpy(dtype="datetime64[ns]")
    avail = parse_utc(available).to_numpy(dtype="datetime64[ns]")
    days = parse_utc(issue).to_numpy(dtype="datetime64[ns]")
    order = np.argsort(days)
    ordered = days[order]
    one = np.timedelta64(1, "D")
    span = np.timedelta64(int(window_days), "D")
    diff = np.zeros(len(ordered) + 1, dtype=float)
    for origin, public in zip(obs, avail, strict=True):
        if pd.isna(origin) or pd.isna(public):
            continue
        start = max(origin.astype("datetime64[D]") + one, public.astype("datetime64[D]"))
        stop = origin.astype("datetime64[D]") + one + span
        lo = int(np.searchsorted(ordered, np.datetime64(start, "ns"), side="left"))
        hi = int(np.searchsorted(ordered, np.datetime64(stop, "ns"), side="left"))
        if lo < hi:
            diff[lo] += 1.0
            diff[hi] -= 1.0
    counted = np.cumsum(diff[:-1])
    out[order] = counted
    return out


def _band(frame: pd.DataFrame, region: dict | None, mag_min: float, mag_max: float, deep: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.iloc[0:0] if frame is not None else pd.DataFrame()
    work = frame
    if region is not None and region.get("kind") == "bbox":
        work = work.loc[in_bbox(work, region)]
    magnitude = pd.to_numeric(work["magnitude"], errors="coerce")
    depth = pd.to_numeric(work["depth_km"], errors="coerce")
    keep = (magnitude >= mag_min) & (magnitude < mag_max)
    if deep:
        keep = keep & (depth >= SLAB_MIN_KM)
    else:
        keep = keep & depth.between(IFACE_MIN_KM, IFACE_MAX_KM)
    return work.loc[keep]


def catalog_depth_counts(quakes: pd.DataFrame, issue: pd.Series, *, source: pd.DataFrame, boxes: list[tuple[str, float, float]]) -> dict[str, np.ndarray]:
    """`boxes` is (region id, mag floor, mag ceiling). Global is handled by the caller."""
    out: dict[str, np.ndarray] = {}
    for region_id, mag_min, mag_max in boxes:
        region = REGIONS[region_id]
        base = source if region_id in {"hikurangi", "alpine"} and source is not None and not source.empty else quakes
        for deep, label in ((False, "iface"), (True, "slab")):
            band = _band(base, region, mag_min, mag_max, deep)
            out[f"n_{label}_{region_id}_30d"] = window_counts(
                band.get("observed_at", pd.Series(dtype=object)),
                band.get("available_at", pd.Series(dtype=object)),
                issue,
            )
    return out


def slab_global_counts(quakes: pd.DataFrame, issue: pd.Series) -> np.ndarray:
    band = _band(quakes, None, 6.0, 7.0, True)
    return window_counts(band.get("observed_at", pd.Series(dtype=object)), band.get("available_at", pd.Series(dtype=object)), issue)


def thrust_mask(rake: pd.Series) -> pd.Series:
    values = pd.to_numeric(rake, errors="coerce")
    return values.between(45.0, 135.0)


def _released_median(frame: pd.DataFrame, issue: pd.Series, public: str, column: str) -> np.ndarray:
    out = np.full(len(issue), np.nan)
    if frame is None or frame.empty or column not in frame.columns:
        return out
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return out
    opened = parse_utc(issue) >= pd.Timestamp(public)
    out[opened.to_numpy()] = float(np.median(finite))
    return out


def attach_deep_features(feat: pd.DataFrame, quakes: pd.DataFrame, root) -> pd.DataFrame:
    """Add depth, mechanism, Japan tremor, and offshore campaign columns. Not issued."""
    from eqpred.ingest.hobitss import OBS_PUBLIC, SWS_PUBLIC
    from eqpred.ingest.run import try_latest_raw

    issue = feat["issue_time"]
    geonet = try_latest_raw("geonet.fdsn", root)
    boxes = (
        ("hikurangi", 4.5, 5.0),
        ("alpine", 4.5, 5.0),
        ("kermadec", 4.5, 6.0),
    )
    for name, values in catalog_depth_counts(quakes, issue, source=geonet, boxes=list(boxes)).items():
        feat[name] = values
    feat["n_slab_global_30d"] = slab_global_counts(quakes, issue)

    cmt = try_latest_raw("gcmt.cmt", root)
    if not cmt.empty and "mw" in cmt.columns:
        work = cmt.copy()
        work["magnitude"] = pd.to_numeric(work["mw"], errors="coerce")
        work = work.loc[work["magnitude"] >= 5.5]
        targets = (
            ("hikurangi", REGIONS["hikurangi"]),
            ("alpine", REGIONS["alpine"]),
            ("kermadec", REGIONS["kermadec"]),
            ("japan", REGIONS["japan"]),
            ("global", None),
        )
        for region_id, region in targets:
            local = work if region is None else work.loc[in_bbox(work, region)]
            feat[f"cmt_{region_id}_n90"] = window_counts(
                local["observed_at"], local["available_at"], issue, window_days=90
            )
            thrust = local.loc[thrust_mask(local["rake"])] if "rake" in local.columns else local.iloc[0:0]
            feat[f"cmt_{region_id}_thrust90"] = window_counts(
                thrust.get("observed_at", pd.Series(dtype=object)),
                thrust.get("available_at", pd.Series(dtype=object)),
                issue,
                window_days=90,
            )
            deep = local.loc[pd.to_numeric(local["depth_km"], errors="coerce") >= SLAB_MIN_KM]
            feat[f"cmt_{region_id}_slab90"] = window_counts(
                deep["observed_at"], deep["available_at"], issue, window_days=90
            )

    local_cmt = try_latest_raw("geonet.cmt", root)
    if not local_cmt.empty and "mw" in local_cmt.columns:
        work = local_cmt.copy()
        work["magnitude"] = pd.to_numeric(work["mw"], errors="coerce")
        work = work.loc[work["magnitude"] >= 4.0]
        for region_id in ("hikurangi", "alpine", "kermadec"):
            local = work.loc[in_bbox(work, REGIONS[region_id])]
            feat[f"geonet_cmt_{region_id}_n90"] = window_counts(
                local["observed_at"], local["available_at"], issue, window_days=90
            )
            thrust = local.loc[thrust_mask(local["rake"])] if "rake" in local.columns else local.iloc[0:0]
            feat[f"geonet_cmt_{region_id}_thrust90"] = window_counts(
                thrust.get("observed_at", pd.Series(dtype=object)),
                thrust.get("available_at", pd.Series(dtype=object)),
                issue,
                window_days=90,
            )

    nied = try_latest_raw("eqdb.nied", root)
    if not nied.empty:
        feat["nied_japan_n90"] = window_counts(
            nied["observed_at"], nied["available_at"], issue, window_days=90
        )

    sws = try_latest_raw("hobitss.sws", root)
    if not sws.empty and "network" in sws.columns:
        feat["sws_obs_tlag"] = _released_median(
            sws.loc[sws["network"] == "obs"], issue, iso_public(SWS_PUBLIC), "tlag_s"
        )
        feat["sws_land_tlag"] = _released_median(
            sws.loc[sws["network"] == "land"], issue, iso_public(SWS_PUBLIC), "tlag_s"
        )
    obs = try_latest_raw("hobitss.obs", root)
    if not obs.empty:
        opened = parse_utc(issue) >= pd.Timestamp(iso_public(OBS_PUBLIC))
        count = np.full(len(issue), np.nan)
        depth = np.full(len(issue), np.nan)
        count[opened.to_numpy()] = float(len(obs))
        depths = pd.to_numeric(obs["depth_km"], errors="coerce")
        if depths.notna().any():
            depth[opened.to_numpy()] = float(depths.median())
        feat["obs_hikurangi_n"] = count
        feat["obs_hikurangi_median_km"] = depth
    sse = try_latest_raw("hikurangi.sse2019", root)
    if not sse.empty:
        from eqpred.ingest.sse2019 import PUBLIC as SSE_PUBLIC

        opened = parse_utc(issue) >= pd.Timestamp(iso_public(SSE_PUBLIC))
        count = np.full(len(issue), np.nan)
        count[opened.to_numpy()] = float(len(sse))
        feat["obs_sse2019_n"] = count
    rupture = try_latest_raw("geonet.rupture", root)
    if not rupture.empty and "available_at" in rupture.columns:
        counts, _medians = published_summary(rupture["available_at"], rupture.get("mw", pd.Series(dtype=float)), issue)
        feat["rupture_n"] = counts
    swsdb = try_latest_raw("sws.db", root)
    if not swsdb.empty and "available_at" in swsdb.columns:
        counts, medians = published_summary(swsdb["available_at"], swsdb.get("dt_s", pd.Series(dtype=float)), issue)
        feat["swsdb_nz_n"] = counts
        feat["swsdb_nz_dt"] = medians
    return feat


def published_summary(available: pd.Series, values: pd.Series, issue: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Count and median of values already public on each morning."""
    counts = np.zeros(len(issue), dtype=float)
    medians = np.full(len(issue), np.nan)
    if available is None or len(available) == 0 or len(issue) == 0:
        return counts, medians
    days = parse_utc(issue).to_numpy(dtype="datetime64[ns]")
    avail = parse_utc(available).to_numpy(dtype="datetime64[ns]")
    nums = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    usable = ~pd.isna(avail)
    avail = avail[usable]
    nums = nums[usable.to_numpy()] if hasattr(usable, "to_numpy") else nums[usable]
    if len(avail) == 0:
        return counts, medians
    order = np.argsort(days)
    ordered = days[order]
    counts[order] = np.searchsorted(np.sort(avail), ordered, side="right")
    pairs = sorted(zip(avail, nums, strict=True), key=lambda item: item[0])
    held: list[float] = []
    pointer = 0
    running = np.full(len(ordered), np.nan)
    for index, day in enumerate(ordered):
        while pointer < len(pairs) and pairs[pointer][0] <= day:
            value = pairs[pointer][1]
            if np.isfinite(value):
                held.append(float(value))
            pointer += 1
        if held:
            running[index] = float(np.median(held))
    medians[order] = running
    return counts, medians


def iso_public(when) -> str:
    return pd.Timestamp(when).tz_convert("UTC").strftime("%Y-%m-%dT00:00:00Z")
