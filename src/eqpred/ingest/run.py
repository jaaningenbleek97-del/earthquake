from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from eqpred.ingest.comcat import pull_comcat
from eqpred.ingest.comcat_versions import (
    BOX_VERSION_MINMAG,
    VERSION_MINMAG,
    ids_in_window,
    normalize_version_frame,
    pull_versions,
)
from eqpred.ingest.fdsn import pull_fdsn_arc
from eqpred.ingest.geometry import pull_geometry
from eqpred.ingest.geonet import pull_geonet
from eqpred.ingest.geonet_gnss import pull_geonet_gnss
from eqpred.ingest.gnss import pull_gnss
from eqpred.ingest.insar import pull_insar
from eqpred.ingest.tide import pull_tide
from eqpred.ingest.grace import pull_grace
from eqpred.ingest.omni import pull_omni
from eqpred.ingest.protons import pull_protons
from eqpred.ingest.so2 import pull_so2
from eqpred.ingest.sse import pull_sse
from eqpred.ingest.storms import pull_storm_onsets
from eqpred.ingest.silso import pull_silso
from eqpred.ingest.cses import pull_cses
from eqpred.ingest.radon import pull_radon
from eqpred.ingest.strain import pull_strain
from eqpred.ingest.swarm import pull_swarm
from eqpred.ingest.swpc import pull_swpc
from eqpred.ingest.era5 import pull_era5
from eqpred.ingest.intermagnet import pull_intermagnet
from eqpred.ingest.tec import pull_tec
from eqpred.ingest.wilcox import pull_wilcox
from eqpred.paths import raw_dir

WAVE1_SOURCES = (
    "usgs.comcat",
    "usgs.comcat.versions",
    "nasa.omni",
    "nasa.omni_storm",
    "wilcox.spf",
    "sidc.silso",
    "computed.geometry",
    "noaa.swpc",
    "noaa.psl",
    "oulu.neutron",
    "oulu.gle",
    "iers.eop",
    "lasp.lisird",
    "cneos.fireball",
    "cdaw.cme",
    "bom.mjo",
    "gfz.hpo",
    "nasa.donki",
)
WAVE2_SOURCES = (
    "geonet.fdsn",
    "geonet.gnss",
    "ngl.gnss",
    "ngl.midas",
    "igs.tec",
    "copernicus.era5",
    "intermagnet.gin",
    "uhslc.tide",
    "comet.insar",
)
WAVE3_SOURCES = (
    "fdsn.japan",
    "fdsn.chile",
    "noaa.goes_protons",
    "noaa.goes_electrons",
    "csr.grace",
    "cams.so2",
    "slowslip.sse",
    "esa.swarm",
    "esa.cses",
    "radon.public",
    "earthscope.strain",
)
WAVE4_SOURCES = (
    "noaa.dart",
    "si.gvp",
    "eqdb.tremor",
    "hydro.wells",
    "ucar.cosmic2",
    "gi.haarp",
    "geonet.dvv",
    "ham.wspr",
    "wiki.pageviews",
    "hydro.surface",
    "mgds.gnssa",
    "geonet.roti",
    "nasa.smap",
    "superdarn.unwin",
    "vlf.public",
    "nasa.tir",
    "das.haast",
)


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _wanted(source: str, only: str | None) -> bool:
    return only is None or only == source


def merge_event_frames(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key: str = "event_id",
) -> pd.DataFrame:
    """Keep prior catalog rows when a later pull covers only a tail window."""
    if incoming is None or incoming.empty:
        return existing if existing is not None else pd.DataFrame()
    if existing is None or existing.empty or key not in incoming.columns:
        return incoming
    if key not in existing.columns:
        return incoming
    out = pd.concat([existing, incoming], ignore_index=True)
    return out.drop_duplicates(subset=[key], keep="last").sort_values("observed_at").reset_index(drop=True)


def merge_daily_frames(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key: str = "observed_at",
) -> pd.DataFrame:
    """Keep prior daily rows when a later pull covers only a tail window.

    Incoming wins on non-null cells so a partial-column refresh (Mg II
    without TSI) does not wipe values the previous file already had.
    """
    if incoming is None or incoming.empty:
        return existing if existing is not None else pd.DataFrame()
    if existing is None or existing.empty or key not in incoming.columns:
        return incoming
    if key not in existing.columns:
        return incoming
    left = incoming.copy()
    right = existing.copy()
    left["_merge_key"] = pd.to_datetime(left[key], utc=True)
    right["_merge_key"] = pd.to_datetime(right[key], utc=True)
    left = left.drop_duplicates(subset=["_merge_key"], keep="last").set_index("_merge_key")
    right = right.drop_duplicates(subset=["_merge_key"], keep="last").set_index("_merge_key")
    out = left.combine_first(right).reset_index()
    out[key] = pd.to_datetime(out["_merge_key"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.drop(columns=["_merge_key"]).sort_values(key).reset_index(drop=True)


def merge_keyed_frames(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    keys: tuple[str, ...] = ("observed_at", "site"),
) -> pd.DataFrame:
    """Keep prior station-day rows when a later pull covers only a tail window."""
    if incoming is None or incoming.empty:
        return existing if existing is not None else pd.DataFrame()
    if existing is None or existing.empty:
        return incoming
    missing = [k for k in keys if k not in incoming.columns or k not in existing.columns]
    if missing:
        return incoming
    out = pd.concat([existing, incoming], ignore_index=True)
    work = out.copy()
    if "observed_at" in keys:
        work["_obs"] = pd.to_datetime(work["observed_at"], utc=True)
        subset = ["_obs", *[k for k in keys if k != "observed_at"]]
        work = work.drop_duplicates(subset=subset, keep="last").drop(columns=["_obs"])
    else:
        work = work.drop_duplicates(subset=list(keys), keep="last")
    sort_cols = [k for k in keys if k in work.columns]
    return work.sort_values(sort_cols).reset_index(drop=True)


def ingest_wave1(
    start: datetime,
    end: datetime,
    minmag: float = 4.5,
    root: Path | None = None,
    only: str | None = None,
    issued_boxes: bool = False,
) -> dict[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    dest = raw_dir(root)
    written: dict[str, Path] = {}

    if _wanted("usgs.comcat", only):
        print(f"ingest {run_id} ComCat M>={minmag} {start.date()} -> {end.date()}")
        comcat = pull_comcat(start, end, minmag, run_id)
        comcat = merge_event_frames(try_latest_raw("usgs.comcat", root), comcat)
        written["usgs.comcat"] = _write(comcat, dest / "usgs.comcat" / f"{run_id}.parquet")
        print(f"  usgs.comcat {len(comcat)} events")

    # Opt-in: full wave1 must not fire thousands of ComCat detail fetches.
    if only == "usgs.comcat.versions":
        from eqpred.regions import issued_boxes as nz_boxes

        boxes = nz_boxes() if issued_boxes else None
        floor = max(float(minmag), BOX_VERSION_MINMAG if boxes else VERSION_MINMAG)
        where = " issued boxes" if boxes else ""
        print(f"ingest {run_id} ComCat versions M>={floor}{where} {start.date()} -> {end.date()}")
        catalog = try_latest_raw("usgs.comcat", root)
        if catalog.empty:
            print("  usgs.comcat.versions skipped: ingest usgs.comcat first")
        else:
            existing = normalize_version_frame(try_latest_raw("usgs.comcat.versions", root))
            event_ids = ids_in_window(catalog, start, end, floor, regions=boxes)
            have = set(existing["event_id"].astype(str)) if not existing.empty and "event_id" in existing.columns else set()
            todo = [eid for eid in event_ids if eid not in have]
            print(f"  {len(event_ids)} catalog events, {len(todo)} missing versions")
            incoming = normalize_version_frame(pull_versions(todo, run_id))
            versions = merge_keyed_frames(existing, incoming, ("event_id", "version_at"))
            if not versions.empty:
                written["usgs.comcat.versions"] = _write(
                    versions, dest / "usgs.comcat.versions" / f"{run_id}.parquet"
                )
                print(f"  usgs.comcat.versions {len(versions)} rows")
            else:
                print("  usgs.comcat.versions empty")

    if _wanted("nasa.omni", only):
        print("ingest NASA OMNI daily")
        try:
            incoming = pull_omni(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — keep the archive if NASA times out
            print(f"  nasa.omni skipped: {exc}")
            incoming = pd.DataFrame()
        omni = merge_daily_frames(try_latest_raw("nasa.omni", root), incoming)
        if not omni.empty:
            written["nasa.omni"] = _write(omni, dest / "nasa.omni" / f"{run_id}.parquet")
            print(f"  nasa.omni {len(omni)} days")

    if _wanted("nasa.omni_storm", only):
        print("ingest OMNI hourly storm onsets (cusp longitude)")
        try:
            storms = pull_storm_onsets(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — overlay; daily OMNI still lands
            print(f"  nasa.omni_storm skipped: {exc}")
            storms = pd.DataFrame()
        storms = merge_daily_frames(try_latest_raw("nasa.omni_storm", root), storms)
        if not storms.empty:
            written["nasa.omni_storm"] = _write(storms, dest / "nasa.omni_storm" / f"{run_id}.parquet")
            print(f"  nasa.omni_storm {len(storms)} onsets")

    if _wanted("wilcox.spf", only):
        print("ingest Wilcox SPF")
        wilcox = merge_daily_frames(try_latest_raw("wilcox.spf", root), pull_wilcox(run_id))
        written["wilcox.spf"] = _write(wilcox, dest / "wilcox.spf" / f"{run_id}.parquet")
        print(f"  wilcox.spf {len(wilcox)} rows")

    if _wanted("sidc.silso", only):
        print("ingest SILSO sunspots")
        silso = merge_daily_frames(try_latest_raw("sidc.silso", root), pull_silso(start, end, run_id))
        written["sidc.silso"] = _write(silso, dest / "sidc.silso" / f"{run_id}.parquet")
        print(f"  sidc.silso {len(silso)} days")

    if _wanted("computed.geometry", only):
        print("ingest computed geometry")
        geometry = merge_daily_frames(
            try_latest_raw("computed.geometry", root), pull_geometry(start, end, run_id)
        )
        written["computed.geometry"] = _write(
            geometry, dest / "computed.geometry" / f"{run_id}.parquet"
        )
        print(f"  computed.geometry {len(geometry)} days")

    if _wanted("noaa.swpc", only):
        print("ingest NOAA SWPC (short operational JSON)")
        try:
            swpc = pull_swpc(run_id)
        except Exception as exc:  # noqa: BLE001 — optional live feed
            print(f"  noaa.swpc skipped: {exc}")
            swpc = pd.DataFrame()
        if not swpc.empty:
            written["noaa.swpc"] = _write(swpc, dest / "noaa.swpc" / f"{run_id}.parquet")
            print(f"  noaa.swpc {len(swpc)} days")
        else:
            print("  noaa.swpc empty (OMNI still covers Kp/F10.7)")

    if _wanted("noaa.psl", only):
        from eqpred.ingest.climate import pull_climate

        print("ingest NOAA PSL climate indices")
        try:
            climate = pull_climate(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  noaa.psl skipped: {exc}")
            climate = pd.DataFrame()
        climate = merge_daily_frames(try_latest_raw("noaa.psl", root), climate)
        if not climate.empty:
            written["noaa.psl"] = _write(climate, dest / "noaa.psl" / f"{run_id}.parquet")
            print(f"  noaa.psl {len(climate)} months")
        else:
            print("  noaa.psl empty")

    if _wanted("oulu.neutron", only):
        from eqpred.ingest.neutron import pull_neutron

        print("ingest Oulu / DOMC neutron monitors")
        try:
            neutron = pull_neutron(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  oulu.neutron skipped: {exc}")
            neutron = pd.DataFrame()
        neutron = merge_daily_frames(try_latest_raw("oulu.neutron", root), neutron)
        if not neutron.empty:
            written["oulu.neutron"] = _write(neutron, dest / "oulu.neutron" / f"{run_id}.parquet")
            print(f"  oulu.neutron {len(neutron)} days")
        else:
            print("  oulu.neutron empty")

    if _wanted("oulu.gle", only):
        from eqpred.ingest.gle import pull_gle

        print("ingest Oulu GLE onsets")
        try:
            gle = pull_gle(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  oulu.gle skipped: {exc}")
            gle = pd.DataFrame()
        gle = merge_daily_frames(try_latest_raw("oulu.gle", root), gle)
        if not gle.empty:
            written["oulu.gle"] = _write(gle, dest / "oulu.gle" / f"{run_id}.parquet")
            print(f"  oulu.gle {int((gle.get('cosmic_gle', 0) > 0).sum())} flagged days")
        else:
            print("  oulu.gle empty")

    if _wanted("iers.eop", only):
        from eqpred.ingest.eop import pull_eop

        print("ingest IERS Earth orientation")
        try:
            eop = pull_eop(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  iers.eop skipped: {exc}")
            eop = pd.DataFrame()
        eop = merge_daily_frames(try_latest_raw("iers.eop", root), eop)
        if not eop.empty:
            written["iers.eop"] = _write(eop, dest / "iers.eop" / f"{run_id}.parquet")
            print(f"  iers.eop {len(eop)} days")
        else:
            print("  iers.eop empty")

    if _wanted("lasp.lisird", only):
        from eqpred.ingest.lisird import pull_lisird

        print("ingest LASP TSI / Mg II / Lyman-alpha")
        try:
            lisird = pull_lisird(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  lasp.lisird skipped: {exc}")
            lisird = pd.DataFrame()
        lisird = merge_daily_frames(try_latest_raw("lasp.lisird", root), lisird)
        if not lisird.empty:
            written["lasp.lisird"] = _write(lisird, dest / "lasp.lisird" / f"{run_id}.parquet")
            print(f"  lasp.lisird {len(lisird)} days")
        else:
            print("  lasp.lisird empty")

    if _wanted("cneos.fireball", only):
        from eqpred.ingest.fireball import pull_fireball

        print("ingest CNEOS fireballs")
        try:
            fire = pull_fireball(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  cneos.fireball skipped: {exc}")
            fire = pd.DataFrame()
        fire = merge_daily_frames(try_latest_raw("cneos.fireball", root), fire)
        if not fire.empty:
            written["cneos.fireball"] = _write(fire, dest / "cneos.fireball" / f"{run_id}.parquet")
            print(f"  cneos.fireball {int((fire.get('dust_fireball', 0) > 0).sum())} flagged days")
        else:
            print("  cneos.fireball empty")

    if _wanted("cdaw.cme", only):
        from eqpred.ingest.cme import pull_cme

        print("ingest CDAW CME catalog")
        try:
            cme = pull_cme(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  cdaw.cme skipped: {exc}")
            cme = pd.DataFrame()
        cme = merge_daily_frames(try_latest_raw("cdaw.cme", root), cme)
        if not cme.empty:
            written["cdaw.cme"] = _write(cme, dest / "cdaw.cme" / f"{run_id}.parquet")
            print(f"  cdaw.cme {int((cme.get('cme_count', 0) > 0).sum())} CME days")
        else:
            print("  cdaw.cme empty")

    if _wanted("bom.mjo", only):
        from eqpred.ingest.mjo import pull_mjo

        print("ingest BOM MJO RMM")
        try:
            mjo = pull_mjo(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  bom.mjo skipped: {exc}")
            mjo = pd.DataFrame()
        mjo = merge_daily_frames(try_latest_raw("bom.mjo", root), mjo)
        if not mjo.empty:
            written["bom.mjo"] = _write(mjo, dest / "bom.mjo" / f"{run_id}.parquet")
            print(f"  bom.mjo {len(mjo)} days")
        else:
            print("  bom.mjo empty")

    if _wanted("gfz.hpo", only):
        from eqpred.ingest.hpo import pull_hpo

        print("ingest GFZ Hp30")
        try:
            hpo = pull_hpo(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  gfz.hpo skipped: {exc}")
            hpo = pd.DataFrame()
        hpo = merge_daily_frames(try_latest_raw("gfz.hpo", root), hpo)
        if not hpo.empty:
            written["gfz.hpo"] = _write(hpo, dest / "gfz.hpo" / f"{run_id}.parquet")
            print(f"  gfz.hpo {len(hpo)} days")
        else:
            print("  gfz.hpo empty")

    if _wanted("nasa.donki", only):
        from eqpred.ingest.donki import pull_donki

        print("ingest NASA DONKI flares / HSS / SEP / shocks")
        try:
            donki = pull_donki(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  nasa.donki skipped: {exc}")
            donki = pd.DataFrame()
        donki = merge_daily_frames(try_latest_raw("nasa.donki", root), donki)
        if not donki.empty:
            written["nasa.donki"] = _write(donki, dest / "nasa.donki" / f"{run_id}.parquet")
            print(f"  nasa.donki {len(donki)} days")
        else:
            print("  nasa.donki empty")
    return written


def ingest_wave2(
    start: datetime,
    end: datetime,
    root: Path | None = None,
    only: str | None = None,
) -> dict[str, Path]:
    """GeoNet, GNSS, IGS TEC, ERA5 NZ loading, INTERMAGNET GIN (all stations)."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    dest = raw_dir(root)
    written: dict[str, Path] = {}

    print(f"ingest wave2 {run_id} {start.date()} -> {end.date()}")
    if _wanted("geonet.fdsn", only):
        print("ingest GeoNet FDSN M>=3")
        try:
            geonet = pull_geonet(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — overlay, not the score catalog
            print(f"  geonet.fdsn skipped: {exc}")
            geonet = pd.DataFrame()
        geonet = merge_event_frames(try_latest_raw("geonet.fdsn", root), geonet)
        if not geonet.empty:
            written["geonet.fdsn"] = _write(geonet, dest / "geonet.fdsn" / f"{run_id}.parquet")
            print(f"  geonet.fdsn {len(geonet)} events")

    if _wanted("geonet.gnss", only):
        print("ingest GeoNet Tilde GNSS (NZ sites)")
        existing = try_latest_raw("geonet.gnss", root)
        if existing is not None and not existing.empty and "site" not in existing.columns:
            existing = existing.copy()
            existing["site"] = "nzmean"
        from eqpred.ingest.gnss import CHILE_SITES, JAPAN_SITES

        ngl = try_latest_raw("ngl.gnss", root)
        sites = None
        if ngl is not None and not ngl.empty and "site" in ngl.columns:
            abroad = {code.upper() for code in (*JAPAN_SITES, *CHILE_SITES)}
            sites = tuple(
                code
                for code in sorted(ngl["site"].astype(str).str.upper().unique())
                if code not in abroad
            )
        try:
            fits = pull_geonet_gnss(start, end, run_id, sites=sites)
        except Exception as exc:  # noqa: BLE001
            print(f"  geonet.gnss skipped: {exc}")
            fits = pd.DataFrame()
        fits = merge_keyed_frames(existing, fits, ("observed_at", "site"))
        if not fits.empty:
            written["geonet.gnss"] = _write(fits, dest / "geonet.gnss" / f"{run_id}.parquet")
            print(f"  geonet.gnss {len(fits)} days")

    if _wanted("ngl.gnss", only):
        print("ingest NGL GNSS (NZ + Japan + Chile sites)")
        try:
            gnss = pull_gnss(start, end, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  ngl.gnss skipped: {exc}")
            gnss = pd.DataFrame()
        gnss = merge_keyed_frames(try_latest_raw("ngl.gnss", root), gnss, ("observed_at", "site"))
        if not gnss.empty:
            written["ngl.gnss"] = _write(gnss, dest / "ngl.gnss" / f"{run_id}.parquet")
            print(f"  ngl.gnss {len(gnss)} site-days")

    if _wanted("ngl.midas", only):
        print("ingest NGL MIDAS station speeds")
        from eqpred.ingest.gnss import pull_midas

        try:
            midas = pull_midas(run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  ngl.midas skipped: {exc}")
            midas = pd.DataFrame()
        if not midas.empty:
            written["ngl.midas"] = _write(midas, dest / "ngl.midas" / f"{run_id}.parquet")
            print(f"  ngl.midas {len(midas)} stations")

    if _wanted("igs.tec", only):
        print("ingest GIM TEC (JPL public weekly, IGS fallback, per-box means)")
        try:
            tec = pull_tec(start, end, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  igs.tec skipped: {exc}")
            tec = pd.DataFrame()
        tec = merge_daily_frames(try_latest_raw("igs.tec", root), tec)
        if not tec.empty:
            written["igs.tec"] = _write(tec, dest / "igs.tec" / f"{run_id}.parquet")
            print(f"  igs.tec {len(tec)} maps")

    if _wanted("copernicus.era5", only):
        print("ingest Copernicus ERA5 (NZ loading)")
        try:
            era5 = pull_era5(start, end, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  copernicus.era5 skipped: {exc}")
            era5 = pd.DataFrame()
        era5 = merge_daily_frames(try_latest_raw("copernicus.era5", root), era5)
        if not era5.empty:
            written["copernicus.era5"] = _write(era5, dest / "copernicus.era5" / f"{run_id}.parquet")
            print(f"  copernicus.era5 {len(era5)} days")

    if _wanted("intermagnet.gin", only):
        print("ingest INTERMAGNET GIN (NZ / Pacific / Japan / Chile stations)")
        try:
            imag = pull_intermagnet(start, end, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  intermagnet.gin skipped: {exc}")
            imag = pd.DataFrame()
        imag = merge_keyed_frames(
            try_latest_raw("intermagnet.gin", root), imag, ("observed_at", "mag_station")
        )
        if not imag.empty:
            written["intermagnet.gin"] = _write(imag, dest / "intermagnet.gin" / f"{run_id}.parquet")
            print(f"  intermagnet.gin {len(imag)} station-days")

    if _wanted("uhslc.tide", only):
        print("ingest UHSLC daily sea level")
        try:
            tide = pull_tide(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — overlay
            print(f"  uhslc.tide skipped: {exc}")
            tide = pd.DataFrame()
        tide = merge_daily_frames(try_latest_raw("uhslc.tide", root), tide)
        drop = [c for c in ("tide_auckland_mm", "tide_lyttelton_mm") if c in tide.columns]
        if drop:
            tide = tide.drop(columns=drop)
        if not tide.empty:
            written["uhslc.tide"] = _write(tide, dest / "uhslc.tide" / f"{run_id}.parquet")
            print(f"  uhslc.tide {len(tide)} days")
        else:
            print("  uhslc.tide empty")

    if _wanted("comet.insar", only):
        print("ingest COMET LiCSAR frame time series (if a small text product exists)")
        try:
            insar = pull_insar(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — overlay
            print(f"  comet.insar skipped: {exc}")
            insar = pd.DataFrame()
        insar = merge_daily_frames(try_latest_raw("comet.insar", root), insar)
        if not insar.empty:
            written["comet.insar"] = _write(insar, dest / "comet.insar" / f"{run_id}.parquet")
            print(f"  comet.insar {len(insar)} days")
        else:
            print("  comet.insar empty (no lightweight LiCSAR series on the public store)")
    return written


def ingest_wave3(
    start: datetime,
    end: datetime,
    root: Path | None = None,
    only: str | None = None,
) -> dict[str, Path]:
    """Japan/Chile FDSN slices, GOES protons/electrons, GRACE, CAMS SO2, slow slip. Swarm is skipped."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    dest = raw_dir(root)
    written: dict[str, Path] = {}

    print(f"ingest wave3 {run_id} {start.date()} -> {end.date()}")
    for source in ("fdsn.japan", "fdsn.chile"):
        if not _wanted(source, only):
            continue
        print(f"ingest USGS FDSN {source}")
        try:
            frame = pull_fdsn_arc(start, end, run_id, source)
        except Exception as exc:  # noqa: BLE001
            print(f"  {source} skipped: {exc}")
            frame = pd.DataFrame()
        frame = merge_event_frames(try_latest_raw(source, root), frame)
        if not frame.empty:
            written[source] = _write(frame, dest / source / f"{run_id}.parquet")
            print(f"  {source} {len(frame)} events")

    if _wanted("noaa.goes_protons", only):
        print("ingest NOAA GOES integral protons (DPD + SEISS SGPS + SWPC JSON)")
        try:
            protons = pull_protons(run_id, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  noaa.goes_protons skipped: {exc}")
            protons = pd.DataFrame()
        protons = merge_daily_frames(try_latest_raw("noaa.goes_protons", root), protons)
        if not protons.empty:
            written["noaa.goes_protons"] = _write(
                protons, dest / "noaa.goes_protons" / f"{run_id}.parquet"
            )
            print(f"  noaa.goes_protons {len(protons)} days")
        else:
            print("  noaa.goes_protons empty")

    if _wanted("noaa.goes_electrons", only):
        from eqpred.ingest.electrons import pull_electrons

        print("ingest NOAA GOES integral electrons (DPD + SEISS MPS-HI + SWPC JSON)")
        try:
            electrons = pull_electrons(run_id, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  noaa.goes_electrons skipped: {exc}")
            electrons = pd.DataFrame()
        electrons = merge_daily_frames(try_latest_raw("noaa.goes_electrons", root), electrons)
        if not electrons.empty:
            written["noaa.goes_electrons"] = _write(
                electrons, dest / "noaa.goes_electrons" / f"{run_id}.parquet"
            )
            print(f"  noaa.goes_electrons {len(electrons)} days")
        else:
            print("  noaa.goes_electrons empty")

    if _wanted("csr.grace", only):
        print("ingest GRACE-FO mascons")
        try:
            grace = pull_grace(start, end, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  csr.grace skipped: {exc}")
            grace = pd.DataFrame()
        grace = merge_daily_frames(try_latest_raw("csr.grace", root), grace)
        if not grace.empty:
            written["csr.grace"] = _write(grace, dest / "csr.grace" / f"{run_id}.parquet")
            print(f"  csr.grace {len(grace)} months")
        else:
            print("  csr.grace empty")

    if _wanted("cams.so2", only):
        print("ingest CAMS SO2 (Kermadec)")
        try:
            so2 = pull_so2(start, end, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  cams.so2 skipped: {exc}")
            so2 = pd.DataFrame()
        so2 = merge_daily_frames(try_latest_raw("cams.so2", root), so2)
        if not so2.empty:
            written["cams.so2"] = _write(so2, dest / "cams.so2" / f"{run_id}.parquet")
            print(f"  cams.so2 {len(so2)} days")
        else:
            print("  cams.so2 empty")

    if _wanted("slowslip.sse", only):
        print("ingest Hikurangi / Slow Earthquake Database SSE")
        try:
            sse = pull_sse(start, end, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  slowslip.sse skipped: {exc}")
            sse = pd.DataFrame()
        sse = merge_daily_frames(try_latest_raw("slowslip.sse", root), sse)
        if not sse.empty:
            written["slowslip.sse"] = _write(sse, dest / "slowslip.sse" / f"{run_id}.parquet")
            print(f"  slowslip.sse {len(sse)} days")
        else:
            print("  slowslip.sse empty")

    if _wanted("esa.swarm", only):
        print("ingest ESA Swarm MAG residual via VirES")
        try:
            swarm = pull_swarm(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  esa.swarm skipped: {exc}")
            swarm = pd.DataFrame()
        swarm = merge_daily_frames(try_latest_raw("esa.swarm", root), swarm)
        if not swarm.empty:
            written["esa.swarm"] = _write(swarm, dest / "esa.swarm" / f"{run_id}.parquet")
            print(f"  esa.swarm {len(swarm)} days")
        else:
            print("  esa.swarm empty")
    if _wanted("esa.cses", only):
        print("ingest CSES HEPD via Limadou")
        try:
            cses = pull_cses(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  esa.cses skipped: {exc}")
            cses = pd.DataFrame()
        cses = merge_daily_frames(try_latest_raw("esa.cses", root), cses)
        if not cses.empty:
            written["esa.cses"] = _write(cses, dest / "esa.cses" / f"{run_id}.parquet")
            print(f"  esa.cses {len(cses)} days")
        else:
            print("  esa.cses empty")
    if _wanted("radon.public", only):
        print("ingest IRON soil radon via Figshare")
        try:
            radon = pull_radon(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  radon.public skipped: {exc}")
            radon = pd.DataFrame()
        radon = merge_daily_frames(try_latest_raw("radon.public", root), radon)
        if not radon.empty:
            written["radon.public"] = _write(radon, dest / "radon.public" / f"{run_id}.parquet")
            print(f"  radon.public {len(radon)} days")
        else:
            print("  radon.public empty")
    if _wanted("earthscope.strain", only):
        print("ingest EarthScope / UNAVCO borehole strain L2")
        try:
            strain = pull_strain(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  earthscope.strain skipped: {exc}")
            strain = pd.DataFrame()
        strain = merge_keyed_frames(
            try_latest_raw("earthscope.strain", root), strain, ("observed_at", "strain_station")
        )
        if not strain.empty:
            written["earthscope.strain"] = _write(
                strain, dest / "earthscope.strain" / f"{run_id}.parquet"
            )
            print(f"  earthscope.strain {len(strain)} station-days")
        else:
            print("  earthscope.strain empty")
    return written


def ingest_wave4(
    start: datetime,
    end: datetime,
    root: Path | None = None,
    only: str | None = None,
) -> dict[str, Path]:
    """DART, volcanoes, tremor/LFE, wells, COSMIC-2 S4, HAARP, dv/v, WSPR, wiki, hydro, GNSS-A, ROTI, SMAP, Unwin, VLF/TIR foils, DAS stub."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    dest = raw_dir(root)
    written: dict[str, Path] = {}

    print(f"ingest wave4 {run_id} {start.date()} -> {end.date()}")
    if _wanted("noaa.dart", only):
        from eqpred.ingest.dart import pull_dart

        print("ingest NOAA / GeoNet DART ocean-bottom pressure")
        try:
            dart = pull_dart(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  noaa.dart skipped: {exc}")
            dart = pd.DataFrame()
        dart = merge_keyed_frames(try_latest_raw("noaa.dart", root), dart, ("observed_at", "dart_station"))
        if not dart.empty:
            written["noaa.dart"] = _write(dart, dest / "noaa.dart" / f"{run_id}.parquet")
            print(f"  noaa.dart {len(dart)} station-days")
        else:
            print("  noaa.dart empty")

    if _wanted("si.gvp", only):
        from eqpred.ingest.volcano import pull_volcano

        print("ingest Smithsonian / NCEI Holocene eruptions")
        try:
            volcano = pull_volcano(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  si.gvp skipped: {exc}")
            volcano = pd.DataFrame()
        volcano = merge_daily_frames(try_latest_raw("si.gvp", root), volcano)
        if not volcano.empty:
            written["si.gvp"] = _write(volcano, dest / "si.gvp" / f"{run_id}.parquet")
            print(f"  si.gvp {len(volcano)} days")
        else:
            print("  si.gvp empty")

    if _wanted("eqdb.tremor", only):
        from eqpred.ingest.tremor import pull_tremor

        print("ingest Slow Earthquake Database NZ tremor / LFE")
        try:
            tremor = pull_tremor(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  eqdb.tremor skipped: {exc}")
            tremor = pd.DataFrame()
        tremor = merge_daily_frames(try_latest_raw("eqdb.tremor", root), tremor)
        if not tremor.empty:
            written["eqdb.tremor"] = _write(tremor, dest / "eqdb.tremor" / f"{run_id}.parquet")
            print(f"  eqdb.tremor {len(tremor)} days")
        else:
            print("  eqdb.tremor empty")

    if _wanted("hydro.wells", only):
        from eqpred.ingest.wells import pull_wells

        print("ingest NZ Hilltop / USGS NWIS groundwater")
        try:
            wells = pull_wells(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  hydro.wells skipped: {exc}")
            wells = pd.DataFrame()
        wells = merge_keyed_frames(
            try_latest_raw("hydro.wells", root), wells, ("observed_at", "well_station")
        )
        if not wells.empty:
            written["hydro.wells"] = _write(wells, dest / "hydro.wells" / f"{run_id}.parquet")
            print(f"  hydro.wells {len(wells)} station-days")
        else:
            print("  hydro.wells empty")

    if _wanted("ucar.cosmic2", only):
        from eqpred.ingest.cosmic2 import pull_cosmic2

        print("ingest UCAR COSMIC-2 scintillation S4")
        try:
            cosmic = pull_cosmic2(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  ucar.cosmic2 skipped: {exc}")
            cosmic = pd.DataFrame()
        cosmic = merge_daily_frames(try_latest_raw("ucar.cosmic2", root), cosmic)
        if not cosmic.empty:
            written["ucar.cosmic2"] = _write(cosmic, dest / "ucar.cosmic2" / f"{run_id}.parquet")
            print(f"  ucar.cosmic2 {len(cosmic)} days")
        else:
            print("  ucar.cosmic2 empty")

    if _wanted("gi.haarp", only):
        from eqpred.ingest.haarp import pull_haarp

        print("ingest HAARP campaign on/off")
        try:
            haarp = pull_haarp(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  gi.haarp skipped: {exc}")
            haarp = pd.DataFrame()
        haarp = merge_daily_frames(try_latest_raw("gi.haarp", root), haarp)
        if not haarp.empty:
            written["gi.haarp"] = _write(haarp, dest / "gi.haarp" / f"{run_id}.parquet")
            print(f"  gi.haarp {len(haarp)} days")
        else:
            print("  gi.haarp empty")

    if _wanted("geonet.dvv", only):
        from eqpred.ingest.dvv import pull_dvv

        print("ingest GeoNet ambient-noise dv/v")
        try:
            dvv = pull_dvv(start, end, run_id, root=root)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  geonet.dvv skipped: {exc}")
            dvv = pd.DataFrame()
        dvv = merge_daily_frames(try_latest_raw("geonet.dvv", root), dvv)
        if not dvv.empty:
            written["geonet.dvv"] = _write(dvv, dest / "geonet.dvv" / f"{run_id}.parquet")
            print(f"  geonet.dvv {len(dvv)} days")
        else:
            print("  geonet.dvv empty")

    if _wanted("ham.wspr", only):
        from eqpred.ingest.wspr import pull_wspr

        print("ingest WSPR NZ path SNR")
        try:
            wspr = pull_wspr(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  ham.wspr skipped: {exc}")
            wspr = pd.DataFrame()
        wspr = merge_daily_frames(try_latest_raw("ham.wspr", root), wspr)
        if not wspr.empty:
            written["ham.wspr"] = _write(wspr, dest / "ham.wspr" / f"{run_id}.parquet")
            print(f"  ham.wspr {len(wspr)} days")
        else:
            print("  ham.wspr empty")

    if _wanted("wiki.pageviews", only):
        from eqpred.ingest.wiki import pull_wiki

        print("ingest Wikipedia earthquake pageviews")
        try:
            wiki = pull_wiki(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional foil
            print(f"  wiki.pageviews skipped: {exc}")
            wiki = pd.DataFrame()
        wiki = merge_daily_frames(try_latest_raw("wiki.pageviews", root), wiki)
        if not wiki.empty:
            written["wiki.pageviews"] = _write(wiki, dest / "wiki.pageviews" / f"{run_id}.parquet")
            print(f"  wiki.pageviews {len(wiki)} days")
        else:
            print("  wiki.pageviews empty")

    if _wanted("hydro.surface", only):
        from eqpred.ingest.hydro import pull_hydro

        print("ingest NZ lake / river stage + POWER wetness")
        try:
            hydro = pull_hydro(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  hydro.surface skipped: {exc}")
            hydro = pd.DataFrame()
        hydro = merge_daily_frames(try_latest_raw("hydro.surface", root), hydro)
        if not hydro.empty:
            written["hydro.surface"] = _write(hydro, dest / "hydro.surface" / f"{run_id}.parquet")
            print(f"  hydro.surface {len(hydro)} days")
        else:
            print("  hydro.surface empty")

    if _wanted("mgds.gnssa", only):
        from eqpred.ingest.gnssa import pull_gnssa

        print("ingest Hikurangi GNSS-A seafloor arrays")
        try:
            gnssa = pull_gnssa(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  mgds.gnssa skipped: {exc}")
            gnssa = pd.DataFrame()
        gnssa = merge_daily_frames(try_latest_raw("mgds.gnssa", root), gnssa)
        if not gnssa.empty:
            written["mgds.gnssa"] = _write(gnssa, dest / "mgds.gnssa" / f"{run_id}.parquet")
            print(f"  mgds.gnssa {len(gnssa)} days")
        else:
            print("  mgds.gnssa empty")

    if _wanted("geonet.roti", only):
        from eqpred.ingest.roti import pull_roti

        print("ingest GeoNet RINEX ROTI / cycle slips")
        try:
            roti = pull_roti(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  geonet.roti skipped: {exc}")
            roti = pd.DataFrame()
        roti = merge_daily_frames(try_latest_raw("geonet.roti", root), roti)
        if not roti.empty:
            written["geonet.roti"] = _write(roti, dest / "geonet.roti" / f"{run_id}.parquet")
            print(f"  geonet.roti {len(roti)} days")
        else:
            print("  geonet.roti empty")

    if _wanted("nasa.smap", only):
        from eqpred.ingest.smap import pull_smap

        print("ingest SMAP NZ soil moisture (Earthdata, never global daily)")
        try:
            smap = pull_smap(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  nasa.smap skipped: {exc}")
            smap = pd.DataFrame()
        smap = merge_daily_frames(try_latest_raw("nasa.smap", root), smap)
        if not smap.empty:
            written["nasa.smap"] = _write(smap, dest / "nasa.smap" / f"{run_id}.parquet")
            print(f"  nasa.smap {len(smap)} days")
        else:
            print("  nasa.smap empty")

    if _wanted("superdarn.unwin", only):
        from eqpred.ingest.superdarn import pull_superdarn

        print("ingest SuperDARN Unwin (Invercargill HF radar)")
        try:
            sdarn = pull_superdarn(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional overlay
            print(f"  superdarn.unwin skipped: {exc}")
            sdarn = pd.DataFrame()
        sdarn = merge_daily_frames(try_latest_raw("superdarn.unwin", root), sdarn)
        if not sdarn.empty:
            written["superdarn.unwin"] = _write(sdarn, dest / "superdarn.unwin" / f"{run_id}.parquet")
            print(f"  superdarn.unwin {len(sdarn)} days")
        else:
            print("  superdarn.unwin empty (CDAWeb has no Unwin; Globus/login if no HTTP listing)")

    if _wanted("vlf.public", only):
        from eqpred.ingest.vlf import pull_vlf

        print("ingest VLF foil (NWC if public, else northern trap)")
        try:
            vlf = pull_vlf(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional foil
            print(f"  vlf.public skipped: {exc}")
            vlf = pd.DataFrame()
        vlf = merge_daily_frames(try_latest_raw("vlf.public", root), vlf)
        if not vlf.empty:
            written["vlf.public"] = _write(vlf, dest / "vlf.public" / f"{run_id}.parquet")
            print(f"  vlf.public {len(vlf)} days")
        else:
            print("  vlf.public empty")

    if _wanted("nasa.tir", only):
        from eqpred.ingest.tir import pull_tir

        print("ingest thermal IR / skin-temperature foil")
        try:
            tir = pull_tir(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — optional foil
            print(f"  nasa.tir skipped: {exc}")
            tir = pd.DataFrame()
        tir = merge_daily_frames(try_latest_raw("nasa.tir", root), tir)
        if not tir.empty:
            written["nasa.tir"] = _write(tir, dest / "nasa.tir" / f"{run_id}.parquet")
            print(f"  nasa.tir {len(tir)} days")
        else:
            print("  nasa.tir empty")

    if _wanted("das.haast", only):
        from eqpred.ingest.das import pull_das

        print("DAS Haast / Southern Cross — partnership stub, no wget")
        try:
            das = pull_das(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — stub
            print(f"  das.haast skipped: {exc}")
            das = pd.DataFrame()
        if not das.empty:
            written["das.haast"] = _write(das, dest / "das.haast" / f"{run_id}.parquet")
            print(f"  das.haast {len(das)} days")
        else:
            print("  das.haast empty (partnership, not a 20-year wget)")
    return written


def latest_raw_path(source: str, root: Path | None = None) -> Path:
    folder = raw_dir(root) / source
    files = list(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw parquet for {source} in {folder}")
    return max(files, key=lambda path: path.stat().st_mtime)


def latest_raw(source: str, root: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(latest_raw_path(source, root))


def try_latest_raw(source: str, root: Path | None = None) -> pd.DataFrame:
    try:
        return latest_raw(source, root)
    except FileNotFoundError:
        return pd.DataFrame()
