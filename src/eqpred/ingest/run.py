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
    "igs.tec",
    "copernicus.era5",
    "intermagnet.gin",
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
    """Keep prior daily rows when a later pull covers only a tail window."""
    if incoming is None or incoming.empty:
        return existing if existing is not None else pd.DataFrame()
    if existing is None or existing.empty or key not in incoming.columns:
        return incoming
    if key not in existing.columns:
        return incoming
    out = pd.concat([existing, incoming], ignore_index=True)
    stamp = pd.to_datetime(out[key], utc=True)
    return (
        out.assign(_merge_key=stamp)
        .drop_duplicates(subset=["_merge_key"], keep="last")
        .drop(columns=["_merge_key"])
        .sort_values(key)
        .reset_index(drop=True)
    )


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
        omni = merge_daily_frames(try_latest_raw("nasa.omni", root), pull_omni(start, end, run_id))
        written["nasa.omni"] = _write(omni, dest / "nasa.omni" / f"{run_id}.parquet")
        print(f"  nasa.omni {len(omni)} days")

    if _wanted("nasa.omni_storm", only):
        print("ingest OMNI hourly storm onsets (cusp longitude)")
        try:
            storms = pull_storm_onsets(start, end, run_id)
        except Exception as exc:  # noqa: BLE001 — overlay; daily OMNI still lands
            print(f"  nasa.omni_storm skipped: {exc}")
            storms = pd.DataFrame()
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        except Exception as exc:  # noqa: BLE001 — leftover overlay
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
        try:
            fits = pull_geonet_gnss(start, end, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  geonet.gnss skipped: {exc}")
            fits = pd.DataFrame()
        fits = merge_daily_frames(try_latest_raw("geonet.gnss", root), fits)
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
        print("ingest NOAA GOES integral protons (DPD archive + SWPC JSON)")
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

        print("ingest NOAA GOES integral electrons (DPD extras + SWPC JSON)")
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
        pull_swarm(start, end, run_id)
    if _wanted("esa.cses", only):
        pull_cses(start, end, run_id)
    if _wanted("radon.public", only):
        pull_radon(start, end, run_id)
    if _wanted("earthscope.strain", only):
        pull_strain(start, end, run_id)
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
