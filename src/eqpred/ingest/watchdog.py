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
from eqpred.ingest.swarm import LATENCY as SWARM_LATENCY
from eqpred.ingest.strain import LATENCY as STRAIN_LATENCY
from eqpred.ingest.cses import LATENCY as CSES_LATENCY
from eqpred.ingest.radon import LATENCY as RADON_LATENCY
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
from eqpred.ingest.tide import LATENCY as TIDE_LATENCY
from eqpred.ingest.insar import LATENCY as INSAR_LATENCY
from eqpred.ingest.wilcox import LATENCY as WILCOX_LATENCY
from eqpred.ingest.climate import LATENCY as PSL_LATENCY
from eqpred.ingest.dart import LATENCY as DART_LATENCY
from eqpred.ingest.volcano import LATENCY as VOLCANO_LATENCY
from eqpred.ingest.tremor import LATENCY as TREMOR_LATENCY
from eqpred.ingest.wells import LATENCY as WELL_LATENCY
from eqpred.ingest.cosmic2 import LATENCY as COSMIC2_LATENCY
from eqpred.ingest.haarp import LATENCY as HAARP_LATENCY
from eqpred.ingest.dvv import LATENCY as DVV_LATENCY
from eqpred.ingest.wspr import LATENCY as WSPR_LATENCY
from eqpred.ingest.wiki import LATENCY as WIKI_LATENCY
from eqpred.ingest.hydro import LATENCY as HYDRO_SURFACE_LATENCY
from eqpred.ingest.gnssa import LATENCY as GNSSA_LATENCY
from eqpred.ingest.roti import LATENCY as ROTI_LATENCY
from eqpred.ingest.smap import LATENCY as SMAP_LATENCY
from eqpred.ingest.superdarn import LATENCY as SDARN_LATENCY
from eqpred.ingest.vlf import LATENCY as VLF_LATENCY
from eqpred.ingest.tir import LATENCY as TIR_LATENCY
from eqpred.ingest.das import LATENCY as DAS_LATENCY
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
    SourceSpec("sidc.silso", SILSO_LATENCY, timedelta(days=21), False, "sunspot cycle"),
    SourceSpec("noaa.swpc", SWPC_LATENCY, timedelta(days=5), False, "live GOES / Kp JSON"),
    SourceSpec("computed.geometry", timedelta(0), timedelta(days=1), False, "tides / lunar geometry"),
    SourceSpec("nasa.omni_storm", STORM_LATENCY, timedelta(days=3), False, "hourly Dst storm onsets"),
    SourceSpec("geonet.fdsn", GEONET_LATENCY, GEONET_SLACK, False, "NZ catalog overlay"),
    SourceSpec("geonet.gnss", GEONET_GNSS_LATENCY, timedelta(days=5), False, "GeoNet GNSS"),
    SourceSpec("ngl.gnss", NGL_LATENCY, timedelta(days=7), False, "NGL GNSS"),
    SourceSpec("igs.tec", TEC_LATENCY, timedelta(days=14), False, "weekly IGS TEC"),
    SourceSpec("copernicus.era5", ERA5_LATENCY, timedelta(days=7), False, "NZ loading"),
    SourceSpec("intermagnet.gin", IMAG_LATENCY, timedelta(days=7), False, "ground magnetics"),
    SourceSpec("fdsn.japan", COMCAT_LATENCY, GEONET_SLACK, False, "Japan FDSN overlay"),
    SourceSpec("fdsn.chile", COMCAT_LATENCY, GEONET_SLACK, False, "Chile FDSN overlay"),
    SourceSpec("noaa.goes_protons", PROTON_LATENCY, timedelta(days=5), False, "GOES protons / SEP"),
    SourceSpec("noaa.goes_electrons", ELECTRON_LATENCY, timedelta(days=5), False, "GOES electrons"),
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
    SourceSpec("esa.swarm", SWARM_LATENCY, timedelta(days=14), False, "Swarm MAG residual"),
    SourceSpec("earthscope.strain", STRAIN_LATENCY, timedelta(days=21), False, "PBO borehole strain"),
    SourceSpec("esa.cses", CSES_LATENCY, timedelta(days=21), False, "CSES HEPD via Limadou"),
    SourceSpec("radon.public", RADON_LATENCY, timedelta(days=21), False, "IRON soil radon (Italy, ends 2021)"),
    SourceSpec("uhslc.tide", TIDE_LATENCY, timedelta(days=45), False, "UHSLC sea level"),
    SourceSpec("comet.insar", INSAR_LATENCY, timedelta(days=45), False, "COMET LiCSAR LOS"),
    SourceSpec("noaa.psl", PSL_LATENCY, timedelta(days=60), False, "climate indices"),
    SourceSpec("noaa.dart", DART_LATENCY, timedelta(days=5), False, "ocean-bottom pressure"),
    SourceSpec("si.gvp", VOLCANO_LATENCY, timedelta(days=14), False, "volcanic eruptions"),
    SourceSpec("eqdb.tremor", TREMOR_LATENCY, timedelta(days=90), False, "Hikurangi / Alpine tremor LFE"),
    SourceSpec("hydro.wells", WELL_LATENCY, timedelta(days=14), False, "NZ / USGS groundwater"),
    SourceSpec("ucar.cosmic2", COSMIC2_LATENCY, timedelta(days=14), False, "COSMIC-2 scintillation S4"),
    SourceSpec("gi.haarp", HAARP_LATENCY, timedelta(days=30), False, "HAARP campaign on/off"),
    SourceSpec("geonet.dvv", DVV_LATENCY, timedelta(days=14), False, "GeoNet ambient-noise dv/v"),
    SourceSpec("ham.wspr", WSPR_LATENCY, timedelta(days=7), False, "WSPR NZ path SNR"),
    SourceSpec("wiki.pageviews", WIKI_LATENCY, timedelta(days=14), False, "Wikipedia earthquake pageviews"),
    SourceSpec("hydro.surface", HYDRO_SURFACE_LATENCY, timedelta(days=14), False, "NZ lakes / rivers / wet weeks"),
    SourceSpec("mgds.gnssa", GNSSA_LATENCY, timedelta(days=90), False, "Hikurangi GNSS-A seafloor"),
    SourceSpec("geonet.roti", ROTI_LATENCY, timedelta(days=14), False, "GeoNet RINEX ROTI / slips"),
    SourceSpec("nasa.smap", SMAP_LATENCY, timedelta(days=21), False, "SMAP NZ soil moisture"),
    SourceSpec("superdarn.unwin", SDARN_LATENCY, timedelta(days=21), False, "SuperDARN Unwin HF radar"),
    SourceSpec("vlf.public", VLF_LATENCY, timedelta(days=21), False, "VLF foil (NWC / northern trap)"),
    SourceSpec("nasa.tir", TIR_LATENCY, timedelta(days=14), False, "thermal IR / skin-temp foil"),
    SourceSpec("das.haast", DAS_LATENCY, timedelta(days=90), False, "Haast DAS partnership stub"),
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


# Known finite archives. A last-day older than slack is not a dead feed.
ARCHIVE_ENDED = {
    "radon.public": "IRON dump ends 2021",
    "wilcox.spf": "Wilcox Polar.html ended",
    "earthscope.strain": "PBO L2 public dump ends 2023",
}

# We pulled the latest public file; the publisher has not posted a newer day.
PUBLISHER_LAG = {
    "cams.so2": "CAMS EAC4 2026 not on ADS yet",
    "bom.mjo": "BOM RMM realtime file last posted 2026-06-24",
}

# Event lists and monthly products: a multi-week gap is normal.
SPARSE_OK = {
    "nasa.omni_storm",
    "csr.grace",
    "cneos.fireball",
    "oulu.gle",
    "si.gvp",
    "noaa.swpc",
    "noaa.psl",
    "igs.tec",
}


def coverage_report(
    as_of: datetime | pd.Timestamp | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Date span, largest gap, and freshness for every registered source."""
    when = as_of or datetime.now(timezone.utc)
    fresh = freshness_report(when, root)
    by_fresh = {row["source"]: row for row in fresh["sources"]}
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        frame = _latest_frame(spec.source, root)
        first = last = None
        n_rows = 0
        n_days = 0
        max_gap = None
        coverage_pct = None
        if not frame.empty and "observed_at" in frame.columns:
            days = parse_utc(frame["observed_at"]).dt.floor("D").dropna().drop_duplicates().sort_values()
            n_rows = int(len(frame))
            n_days = int(len(days))
            if n_days:
                first = _iso_ts(days.min())
                last = _iso_ts(days.max())
                diffs = days.diff().dt.days.dropna()
                if len(diffs):
                    max_gap = int(diffs.max())
                span = int((days.max() - days.min()).days) + 1
                coverage_pct = round(100.0 * n_days / span, 1) if span else None
        clock = by_fresh.get(spec.source) or {}
        status = clock.get("status") or "missing"
        note = ARCHIVE_ENDED.get(spec.source) or PUBLISHER_LAG.get(spec.source)
        if status == "missing":
            verdict = "missing"
        elif spec.source in ARCHIVE_ENDED:
            verdict = "archive_ended"
        elif spec.source in PUBLISHER_LAG:
            verdict = "publisher_lag"
        elif spec.source in SPARSE_OK:
            # Event lists and monthly products: weeks without a new event is normal.
            verdict = "complete"
        elif status == "stale":
            verdict = "stale"
        elif max_gap is not None and max_gap > 21:
            verdict = "gappy"
        else:
            verdict = "complete"
        rows.append(
            {
                "source": spec.source,
                "role": spec.role,
                "required": spec.required,
                "verdict": verdict,
                "status": status,
                "note": note,
                "n_rows": n_rows,
                "n_days": n_days,
                "first_observed_at": first,
                "last_observed_at": last,
                "max_gap_days": max_gap,
                "coverage_pct": coverage_pct,
                "lag_days": clock.get("lag_days"),
            }
        )
    missing = [row["source"] for row in rows if row["verdict"] == "missing"]
    stale = [row["source"] for row in rows if row["verdict"] == "stale"]
    gappy = [row["source"] for row in rows if row["verdict"] == "gappy"]
    publisher_lag = [row["source"] for row in rows if row["verdict"] == "publisher_lag"]
    return {
        "as_of": _iso_ts(utc_ts(when)),
        "ok": fresh["ok"],
        "blockers": fresh["blockers"],
        "missing": missing,
        "stale": stale,
        "gappy": gappy,
        "publisher_lag": publisher_lag,
        "sources": rows,
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
