from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.stamp import parse_utc, utc_ts

# First-report delay when we have no product history. Hits still use the
# final bulletin. Predictors use versions + Mc-at-origin when present.
REPORT_DELAY_HOURS = (
    (7.5, 3.0),
    (7.0, 6.0),
    (6.0, 12.0),
    (5.0, 24.0),
    (0.0, 48.0),
)

# Completeness at origin time, not at as-of time. A 1995 M4.6 that was
# added in a later reprocess stays hidden even in 2020.
COMPLETENESS_MAG = (
    (1980, 5.5),
    (1990, 5.0),
    (2000, 4.8),
    (2010, 4.5),
)

VERSION_SOURCE = "usgs.comcat.versions"


def reporting_delay_hours(magnitude: float) -> float:
    mag = float(magnitude) if np.isfinite(magnitude) else 4.5
    for floor, hours in REPORT_DELAY_HOURS:
        if mag >= floor:
            return hours
    return 48.0


def reporting_delay(magnitude: float) -> timedelta:
    return timedelta(hours=reporting_delay_hours(magnitude))


def completeness_mag(origin_time: datetime | pd.Timestamp | str) -> float:
    year = int(utc_ts(origin_time).year)
    mc = float(COMPLETENESS_MAG[0][1])
    for start_year, mag in COMPLETENESS_MAG:
        if year >= start_year:
            mc = float(mag)
        else:
            break
    return mc


def completeness_mag_series(stamps: pd.Series) -> pd.Series:
    years = parse_utc(stamps).dt.year
    mc = np.full(len(years), float(COMPLETENESS_MAG[0][1]), dtype=float)
    for start_year, mag in COMPLETENESS_MAG:
        mc = np.where(years.to_numpy() >= start_year, float(mag), mc)
    return pd.Series(mc, index=getattr(stamps, "index", None))


def apply_completeness(quakes: pd.DataFrame) -> pd.DataFrame:
    """Drop events below the Mc of their origin era (final or as-of mag)."""
    if quakes is None or quakes.empty:
        return quakes.copy() if quakes is not None else pd.DataFrame()
    out = quakes.copy()
    out["observed_at"] = parse_utc(out["observed_at"])
    mag = pd.to_numeric(out["magnitude"], errors="coerce")
    mc = completeness_mag_series(out["observed_at"])
    return out.loc[mag >= mc].copy()


def _first_version_at(versions: pd.DataFrame) -> pd.Series:
    work = versions.copy()
    work["event_id"] = work["event_id"].astype(str)
    work["version_at"] = parse_utc(work["version_at"])
    return work.groupby("event_id")["version_at"].min()


# ComCat's version feed does not reach back to 1990. For an older event its
# earliest record is a bulk re-review (thousands share 2014-11-07), not the
# first report, which would make a 1990 M6.5 look unknown until 2014. Trust a
# version stamp only when it lands within a normal review window of the origin.
VERSION_TRUST_DAYS = 365.0


def _untrusted_version_ids(versions: pd.DataFrame) -> set[str]:
    """Events whose earliest version record is a late bulk re-catalog stamp."""
    if (
        versions is None
        or versions.empty
        or "event_id" not in versions.columns
        or "observed_at" not in versions.columns
    ):
        return set()
    work = versions.copy()
    work["event_id"] = work["event_id"].astype(str)
    work["version_at"] = parse_utc(work["version_at"])
    work["observed_at"] = parse_utc(work["observed_at"])
    grouped = work.groupby("event_id").agg(
        first_version=("version_at", "min"),
        observed=("observed_at", "min"),
    )
    gap_days = (grouped["first_version"] - grouped["observed"]).dt.total_seconds() / 86400.0
    return set(grouped.index[gap_days > VERSION_TRUST_DAYS])


def _trusted_first_version(
    frame: pd.DataFrame,
    versions: pd.DataFrame,
    first: pd.Series | None = None,
) -> pd.Series:
    """First `version_at` per row, blanked when it is a late bulk re-catalog stamp."""
    if first is None:
        first = _first_version_at(versions)
    first = first.copy()
    first.index = first.index.astype(str)
    mapped = parse_utc(frame["event_id"].astype(str).map(first))
    observed = parse_utc(frame["observed_at"])
    gap_days = (mapped - observed).dt.total_seconds() / 86400.0
    return mapped.where(gap_days <= VERSION_TRUST_DAYS)


def apply_available_at(
    quakes: pd.DataFrame,
    versions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Overwrite available_at from origin time + magnitude-dependent delay.

    When version history exists for an event, first `version_at` is the
    first report — not delay-from-final-mag (which lets a later M7 appear
    too early).
    """
    if quakes is None or quakes.empty:
        return quakes.copy() if quakes is not None else pd.DataFrame()
    out = quakes.copy()
    out["observed_at"] = parse_utc(out["observed_at"])
    mag = pd.to_numeric(out["magnitude"], errors="coerce").to_numpy(dtype=float)
    hours = np.array([reporting_delay_hours(m) for m in mag], dtype=float)
    out["available_at"] = out["observed_at"] + pd.to_timedelta(hours, unit="h")
    if (
        versions is not None
        and not versions.empty
        and "event_id" in out.columns
        and "event_id" in versions.columns
    ):
        mapped = _trusted_first_version(out, versions)
        out["available_at"] = mapped.fillna(out["available_at"])
    return out


def versions_asof(
    versions: pd.DataFrame,
    asof: datetime | pd.Timestamp | str,
) -> pd.DataFrame:
    """Latest product of each event at `asof`. Deleted events disappear."""
    if versions is None or versions.empty:
        return pd.DataFrame()
    cut = utc_ts(asof)
    work = versions.copy()
    work["version_at"] = parse_utc(work["version_at"])
    if "observed_at" in work.columns:
        work["observed_at"] = parse_utc(work["observed_at"])
    work = work.loc[work["version_at"] <= cut]
    if work.empty:
        return work
    work = work.sort_values(["event_id", "version_at"]).groupby("event_id", as_index=False).tail(1)
    status = work["status"].astype(str).str.lower() if "status" in work.columns else "reviewed"
    return work.loc[status != "deleted"].reset_index(drop=True)


def catalog_asof(
    quakes: pd.DataFrame,
    asof: datetime | pd.Timestamp | str,
    *,
    minmag: float | None = None,
    versions: pd.DataFrame | None = None,
    apply_mc: bool = True,
) -> pd.DataFrame:
    """Events a predictor issued at `asof` could have seen.

    Versioned ids use the latest product <= T (not the final bulletin).
    Unversioned ids keep delay visibility + final mag. Mc is the
    completeness of the origin era, not of T.
    """
    cut = utc_ts(asof)
    if quakes is None or quakes.empty:
        return pd.DataFrame()

    has_versions = (
        versions is not None
        and not versions.empty
        and "event_id" in quakes.columns
        and "event_id" in versions.columns
    )
    versioned_ids: set[str] = set()
    asof_ver = pd.DataFrame()
    if has_versions:
        # Events the feed only re-catalogued years later take the delay path
        # below, otherwise a 1990 quake stays invisible until the re-review.
        untrusted = _untrusted_version_ids(versions)
        versioned_ids = set(versions["event_id"].dropna().astype(str)) - untrusted
        asof_ver = versions_asof(versions, cut)
        if not asof_ver.empty:
            asof_ver = asof_ver.copy()
            asof_ver["event_id"] = asof_ver["event_id"].astype(str)
            asof_ver = asof_ver.loc[asof_ver["event_id"].isin(versioned_ids)]
        if not asof_ver.empty:
            asof_ver["observed_at"] = parse_utc(asof_ver["observed_at"])
            asof_ver["magnitude"] = pd.to_numeric(asof_ver["magnitude"], errors="coerce")
            asof_ver["available_at"] = _trusted_first_version(asof_ver, versions)

    if "event_id" in quakes.columns and versioned_ids:
        eids = quakes["event_id"].astype(str)
        unversioned = quakes.loc[~eids.isin(versioned_ids)].copy()
    else:
        unversioned = quakes.copy()

    if unversioned.empty:
        visible = asof_ver
    else:
        unversioned = apply_available_at(unversioned)
        unversioned["available_at"] = parse_utc(unversioned["available_at"])
        unversioned["observed_at"] = parse_utc(unversioned["observed_at"])
        unversioned = unversioned.loc[unversioned["available_at"] <= cut]
        visible = (
            pd.concat([unversioned, asof_ver], ignore_index=True, sort=False)
            if not asof_ver.empty
            else unversioned
        )

    if visible is None or visible.empty:
        return pd.DataFrame()

    if apply_mc:
        visible = apply_completeness(visible)
    if minmag is not None and not visible.empty:
        visible = visible.loc[pd.to_numeric(visible["magnitude"], errors="coerce") >= minmag]
    return visible.reset_index(drop=True)


def event_available_as(
    quakes: pd.DataFrame,
    minmag: float,
    versions: pd.DataFrame | None = None,
) -> pd.Series:
    """When each event first counted as >= minmag (first crossing / delay)."""
    if quakes is None or quakes.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    work = apply_available_at(quakes, versions=versions)
    mag = pd.to_numeric(work["magnitude"], errors="coerce")
    out = parse_utc(work["available_at"]).where(mag >= float(minmag))
    if (
        versions is None
        or versions.empty
        or "event_id" not in work.columns
        or "event_id" not in versions.columns
    ):
        return out
    v = versions.copy()
    v["version_at"] = parse_utc(v["version_at"])
    v["magnitude"] = pd.to_numeric(v["magnitude"], errors="coerce")
    status = v["status"].astype(str).str.lower() if "status" in v.columns else "reviewed"
    ok = (v["magnitude"] >= float(minmag)) & (status != "deleted")
    if not bool(ok.any()):
        versioned = work["event_id"].astype(str).isin(versions["event_id"].astype(str))
        return out.where(~versioned)
    crossed = v.loc[ok].copy()
    crossed["event_id"] = crossed["event_id"].astype(str)
    first = crossed.sort_values("version_at").groupby("event_id")["version_at"].min()
    mapped = _trusted_first_version(work, versions, first=first)
    versioned = work["event_id"].astype(str).isin(versions["event_id"].astype(str))
    # A stamp we do not trust falls back to the delay model, same as an
    # unversioned event, instead of hiding the quake for years.
    return mapped.where(versioned & mapped.notna(), out)


def load_comcat_versions(root: Path | None = None) -> pd.DataFrame:
    from eqpred.ingest.run import try_latest_raw

    frame = try_latest_raw(VERSION_SOURCE, root)
    if frame.empty:
        return frame
    out = frame.copy()
    if "version_at" in out.columns:
        out["version_at"] = parse_utc(out["version_at"])
    if "observed_at" in out.columns:
        out["observed_at"] = parse_utc(out["observed_at"])
    if "magnitude" in out.columns:
        out["magnitude"] = pd.to_numeric(out["magnitude"], errors="coerce")
    return out


def load_comcat(root: Path | None = None) -> pd.DataFrame:
    from eqpred.ingest.run import latest_raw

    return apply_available_at(latest_raw("usgs.comcat", root), versions=load_comcat_versions(root))
