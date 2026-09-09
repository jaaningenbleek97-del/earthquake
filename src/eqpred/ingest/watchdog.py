"""Dead-man's switch on source available_at age.

A morning is issuable only if every required source has a row whose
available_at is not older than that morning by more than the source slack.
Holdout replay stays valid when the store covers those days. A live card
on a dead OMNI or ComCat feed is refused. Wilcox is optional: Polar.html
ended, so a stale SPF overlay must not block fusion cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from eqpred.catalog import reporting_delay
from eqpred.ingest.era5 import LATENCY as ERA5_LATENCY
from eqpred.ingest.geonet_gnss import LATENCY as GEONET_GNSS_LATENCY
from eqpred.ingest.gnss import LATENCY as NGL_LATENCY
from eqpred.ingest.grace import LATENCY as GRACE_LATENCY
from eqpred.ingest.intermagnet import LATENCY as IMAG_LATENCY
from eqpred.ingest.cme import LATENCY as CME_LATENCY
from eqpred.ingest.eop import LATENCY as EOP_LATENCY
from eqpred.ingest.fireball import LATENCY as FIREBALL_LATENCY
from eqpred.ingest.gle import LATENCY as GLE_LATENCY
from eqpred.ingest.lisird import LATENCY as LISIRD_LATENCY
from eqpred.ingest.donki import LATENCY as DONKI_LATENCY
from eqpred.ingest.hpo import LATENCY as HPO_LATENCY
from eqpred.ingest.mjo import LATENCY as MJO_LATENCY
from eqpred.ingest.neutron import LATENCY as NEUTRON_LATENCY
from eqpred.ingest.omni import LATENCY as OMNI_LATENCY
from eqpred.ingest.electrons import LATENCY as ELECTRON_LATENCY
from eqpred.ingest.protons import LATENCY as PROTON_LATENCY
from eqpred.ingest.silso import LATENCY as SILSO_LATENCY
from eqpred.ingest.so2 import LATENCY as SO2_LATENCY
from eqpred.ingest.sse import LATENCY as SSE_LATENCY
from eqpred.ingest.storms import LATENCY as STORM_LATENCY
from eqpred.ingest.swpc import LATENCY as SWPC_LATENCY
from eqpred.ingest.tec import LATENCY as TEC_LATENCY
from eqpred.ingest.wilcox import LATENCY as WILCOX_LATENCY
from eqpred.records import iso
from eqpred.stamp import parse_utc, utc_ts

# ComCat latency is magnitude-dependent; M≥4.5 (the pull floor) is 48h.
# A week without a newly available global M≥4.5 means the catalog pull is dead.
COMCAT_LATENCY = reporting_delay(4.5)
COMCAT_SLACK = timedelta(days=7)
GEONET_LATENCY = reporting_delay(3.0)
GEONET_SLACK = timedelta(days=7)


@dataclass(frozen=True)
class SourceSpec:
    source: str
    latency: timedelta
    slack: timedelta
    required: bool
    role: str


SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("usgs.comcat", COMCAT_LATENCY, COMCAT_SLACK, True, "score catalog + ETAS"),
    SourceSpec("nasa.omni", OMNI_LATENCY, timedelta(days=5), True, "solar-wind / Kp / IMF"),
    SourceSpec("wilcox.spf", WILCOX_LATENCY, timedelta(days=21), False, "polar field / Davidson window"),
    SourceSpec("sidc.silso", SILSO_LATENCY, timedelta(days=7), False, "sunspot cycle"),
    SourceSpec("noaa.swpc", SWPC_LATENCY, timedelta(days=2), False, "live GOES / Kp JSON"),
    SourceSpec("computed.geometry", timedelta(0), timedelta(days=1), False, "tides / lunar geometry"),
    SourceSpec("nasa.omni_storm", STORM_LATENCY, timedelta(days=3), False, "hourly Dst storm onsets"),
    SourceSpec("geonet.fdsn", GEONET_LATENCY, GEONET_SLACK, False, "NZ catalog overlay"),
    SourceSpec("geonet.gnss", GEONET_GNSS_LATENCY, timedelta(days=5), False, "GeoNet GNSS"),
    SourceSpec("ngl.gnss", NGL_LATENCY, timedelta(days=7), False, "NGL GNSS"),
    SourceSpec("igs.tec", TEC_LATENCY, timedelta(days=14), False, "weekly IGS TEC"),
    SourceSpec("copernicus.era5", ERA5_LATENCY, timedelta(days=7), False, "NZ loading"),
    SourceSpec("intermagnet.gin", IMAG_LATENCY, timedelta(days=3), False, "ground magnetics"),
    SourceSpec("fdsn.japan", COMCAT_LATENCY, GEONET_SLACK, False, "Japan FDSN overlay"),
    SourceSpec("fdsn.chile", COMCAT_LATENCY, GEONET_SLACK, False, "Chile FDSN overlay"),
    SourceSpec("noaa.goes_protons", PROTON_LATENCY, timedelta(days=2), False, "GOES protons / SEP"),
    SourceSpec("noaa.goes_electrons", ELECTRON_LATENCY, timedelta(days=2), False, "GOES electrons"),
    SourceSpec("csr.grace", GRACE_LATENCY, timedelta(days=90), False, "GRACE-FO mass"),
    SourceSpec("cams.so2", SO2_LATENCY, timedelta(days=14), False, "Kermadec SO2"),
    SourceSpec("slowslip.sse", SSE_LATENCY, timedelta(days=90), False, "Hikurangi / eqdb SSE"),
    SourceSpec("oulu.neutron", NEUTRON_LATENCY, timedelta(days=7), False, "Oulu / DOMC neutron monitors"),
    SourceSpec("oulu.gle", GLE_LATENCY, timedelta(days=30), False, "ground-level enhancement onsets"),
    SourceSpec("iers.eop", EOP_LATENCY, timedelta(days=14), False, "Earth orientation / LOD"),
    SourceSpec("lasp.lisird", LISIRD_LATENCY, timedelta(days=14), False, "TSI / Mg II / Lyman-alpha"),
    SourceSpec("cneos.fireball", FIREBALL_LATENCY, timedelta(days=14), False, "CNEOS fireballs"),
    SourceSpec("cdaw.cme", CME_LATENCY, timedelta(days=21), False, "CDAW CME catalog"),
    SourceSpec("bom.mjo", MJO_LATENCY, timedelta(days=14), False, "Madden–Julian Oscillation"),
    SourceSpec("gfz.hpo", HPO_LATENCY, timedelta(days=7), False, "GFZ Hp30 magnetosphere index"),
    SourceSpec("nasa.donki", DONKI_LATENCY, timedelta(days=14), False, "DONKI flares / HSS / SEP"),
)

REQUIRED = tuple(spec.source for spec in SPECS if spec.required)


class StaleSourcesError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        blockers = ", ".join(report.get("blockers") or [])
        as_of = report.get("as_of") or "?"
        super().__init__(f"required sources stale as of {as_of}: {blockers or 'unknown'}")


def _iso_ts(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return iso(ts.to_pydatetime())


def _latest_frame(source: str, root: Path | None = None) -> pd.DataFrame:
    from eqpred.ingest.run import try_latest_raw

    return try_latest_raw(source, root)


def source_clocks(root: Path | None = None) -> dict[str, dict[str, pd.Timestamp | None]]:
    """Latest observed_at / available_at per registered source."""
    clocks: dict[str, dict[str, pd.Timestamp | None]] = {}
    for spec in SPECS:
        frame = _latest_frame(spec.source, root)
        observed = available = None
        if not frame.empty and "observed_at" in frame.columns:
            observed = parse_utc(frame["observed_at"]).max()
        if not frame.empty and "available_at" in frame.columns:
            available = parse_utc(frame["available_at"]).max()
        elif observed is not None:
            available = observed + spec.latency
        clocks[spec.source] = {"observed_at": observed, "available_at": available}
    return clocks


def _inspect(
    spec: SourceSpec,
    as_of: datetime | pd.Timestamp,
    clocks: dict[str, dict[str, pd.Timestamp | None]],
) -> dict[str, Any]:
    clock = clocks.get(spec.source) or {}
    available = clock.get("available_at")
    observed = clock.get("observed_at")
    as_of_ts = utc_ts(as_of)
    if available is None or pd.isna(available):
        status = "missing"
        lag_days = None
        age = None
    else:
        age = as_of_ts - pd.Timestamp(available)
        lag_days = max(0.0, float(age / pd.Timedelta(days=1)))
        status = "fresh" if age <= spec.slack else "stale"
    blocking = spec.required and status != "fresh"
    return {
        "source": spec.source,
        "status": status,
        "required": spec.required,
        "blocking": blocking,
        "role": spec.role,
        "latest_observed_at": _iso_ts(observed),
        "latest_available_at": _iso_ts(available),
        "lag_days": None if lag_days is None else round(lag_days, 3),
        "latency_days": spec.latency / timedelta(days=1),
        "slack_days": spec.slack / timedelta(days=1),
    }


def freshness_report(
    as_of: datetime | pd.Timestamp | None = None,
    root: Path | None = None,
    clocks: dict[str, dict[str, pd.Timestamp | None]] | None = None,
) -> dict[str, Any]:
    when = as_of or datetime.now(timezone.utc)
    clocks = clocks if clocks is not None else source_clocks(root)
    sources = [_inspect(spec, when, clocks) for spec in SPECS]
    blockers = [row["source"] for row in sources if row["blocking"]]
    return {
        "as_of": _iso_ts(utc_ts(when)),
        "ok": not blockers,
        "blockers": blockers,
        "sources": sources,
    }


def issuable_times(
    times: Iterable[datetime | pd.Timestamp],
    root: Path | None = None,
    clocks: dict[str, dict[str, pd.Timestamp | None]] | None = None,
) -> list[pd.Timestamp]:
    clocks = clocks if clocks is not None else source_clocks(root)
    kept: list[pd.Timestamp] = []
    for value in times:
        ts = utc_ts(value)
        if freshness_report(ts, root, clocks)["ok"]:
            kept.append(ts)
    return kept


def ensure_can_issue(
    times: Iterable[datetime | pd.Timestamp],
    root: Path | None = None,
) -> list[pd.Timestamp]:
    """Return the issue mornings that pass the required-source gate.

    Raises StaleSourcesError when every requested morning is blocked, so the
    caller can leave existing prediction batches on disk.
    """
    series = [utc_ts(value) for value in times]
    if not series:
        return []
    clocks = source_clocks(root)
    kept = issuable_times(series, root, clocks)
    if not kept:
        raise StaleSourcesError(freshness_report(series[-1], root, clocks))
    return kept
