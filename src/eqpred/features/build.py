from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.features.bvalue import bvalue_frame
from eqpred.features.coulomb import coulomb_frame
from eqpred.features.cusp import cusp_frame
from eqpred.features.davidson import apply_train_budget, field_strength, significant_windows
from eqpred.features.fresh import fresh_derived, planetary_frame
from eqpred.features.gnss_strain import gnss_strain_columns
from eqpred.features.nowcast import catalog_nowcast_frame
from eqpred.features.migration import migration_frame
from eqpred.catalog import apply_completeness, event_available_as, load_comcat, load_comcat_versions
from eqpred.ingest.run import latest_raw, try_latest_raw
from eqpred.paths import feature_dir
from eqpred.stamp import merge_daily, parse_utc, utc_ts


def _asof_available(frame: pd.DataFrame, days: pd.DatetimeIndex, value_cols: list[str]) -> pd.DataFrame:
    """Point-in-time: on day T use the latest row with available_at <= T."""
    cols = [c for c in value_cols if c in frame.columns]
    empty = pd.DataFrame({"issue_time": days, **{c: np.nan for c in value_cols}})
    if frame.empty or not cols:
        return empty
    work = frame.copy()
    work["available_at"] = parse_utc(work["available_at"])
    work = work.dropna(subset=["available_at"]).sort_values("available_at")
    if work.empty:
        return empty
    left = pd.DataFrame({"issue_time": days}).sort_values("issue_time")
    merged = pd.merge_asof(
        left,
        work.rename(columns={"available_at": "issue_time"}),
        on="issue_time",
        direction="backward",
    )
    keep = ["issue_time", *cols]
    out = merged[keep]
    for missing in value_cols:
        if missing not in out.columns:
            out[missing] = np.nan
    return out


def _cycle_phase(sunspots: pd.Series) -> pd.Series:
    smooth = sunspots.rolling(365, min_periods=60).mean()
    # 0=min, 0.5=max, 1=next min. Rank within a rolling 11y window.
    roll_min = smooth.rolling(4000, min_periods=200).min()
    roll_max = smooth.rolling(4000, min_periods=200).max()
    span = (roll_max - roll_min).replace(0, np.nan)
    return (smooth - roll_min) / span


def _proton_spike(density: pd.Series, train_end: pd.Timestamp) -> pd.Series:
    train = density.loc[density.index <= train_end].dropna()
    if train.empty:
        thresh = density.median()
    else:
        q75, q25 = train.quantile(0.75), train.quantile(0.25)
        thresh = train.median() + 1.5 * (q75 - q25)
    return (density > thresh).astype(float)


def build_features(
    train_end: str = "2018-12-31",
    root: Path | None = None,
) -> pd.DataFrame:
    quakes = load_comcat(root)
    versions = load_comcat_versions(root)
    omni = latest_raw("nasa.omni", root)
    spf = latest_raw("wilcox.spf", root)
    silso = latest_raw("sidc.silso", root)
    geom = latest_raw("computed.geometry", root)

    quakes["observed_at"] = parse_utc(quakes["observed_at"])
    # Calendar follows the ingested OMNI window, not the full Wilcox archive.
    start = parse_utc(omni["observed_at"]).min().floor("D")
    end = parse_utc(quakes["observed_at"]).max().floor("D")
    days = pd.date_range(start, end, freq="D", tz="UTC")

    windows = significant_windows(spf, quakes, version="v2", train_end=train_end)
    strength = field_strength(spf)
    # Polar.html posts ~40 days late — do not alarm on a peak we have not seen.
    windows.index = pd.DatetimeIndex(windows.index) + pd.Timedelta(days=40)
    strength.index = pd.DatetimeIndex(strength.index) + pd.Timedelta(days=40)
    windows = windows.reindex(days).fillna(False)
    strength = strength.reindex(days)
    # Budget uses the ingest calendar's training era, not 1976–2018 cycle peaks.
    windows, budget_cut = apply_train_budget(windows, strength, train_end)
    print(
        f"  SPF v2 train budget cutoff={budget_cut:.1f} "
        f"window_frac={float(windows.mean()):.3f}"
    )

    feat = _asof_available(
        omni,
        days,
        [
            "proton_density_cm3",
            "solar_wind_speed_kms",
            "imf_bz_nt",
            "imf_bx_nt",
            "imf_by_nt",
            "imf_b_nt",
            "proton_temp_k",
            "flow_pressure_npa",
            "electric_field_mvm",
            "plasma_beta",
            "kp",
            "dst_nt",
            "ae_nt",
            "al_nt",
            "au_nt",
            "pc_index",
            "ap",
            "f107",
        ],
    )
    silso_asof = _asof_available(silso, days, ["sunspot_number", "sunspot_north", "sunspot_south"])
    geom_asof = _asof_available(
        geom,
        days,
        [
            "lunar_phase_deg",
            "lunar_distance_re",
            "lunar_declination_deg",
            "tidal_strain",
            "near_new_or_full",
            "solar_eclipse_window",
            "lunar_eclipse_geom",
        ],
    )
    feat = feat.merge(silso_asof, on="issue_time", how="left")
    feat = feat.merge(geom_asof, on="issue_time", how="left")
    feat = feat.merge(coulomb_frame(feat), on="issue_time", how="left")
    storms = try_latest_raw("nasa.omni_storm", root)
    feat = feat.merge(cusp_frame(feat["issue_time"], storms), on="issue_time", how="left")
    feat["spf_significant_window"] = windows.astype(float).to_numpy()
    feat["spf_strength"] = strength.reindex(feat["issue_time"]).to_numpy()

    swpc = try_latest_raw("noaa.swpc", root)
    if not swpc.empty:
        xray_cols = [c for c in ("xray_peak_wm2", "swpc_kp", "swpc_f107") if c in swpc.columns]
        if xray_cols:
            swpc_asof = _asof_available(swpc, days, xray_cols)
            feat = feat.merge(swpc_asof, on="issue_time", how="left")
            if "xray_peak_wm2" in feat.columns:
                feat["xray_peak_7d"] = feat["xray_peak_wm2"].rolling(7, min_periods=1).max()

    feat = feat.sort_values("issue_time")

    for col in (
        "proton_density_cm3",
        "solar_wind_speed_kms",
        "imf_bz_nt",
        "kp",
        "dst_nt",
        "f107",
        "sunspot_number",
        "tidal_strain",
        "lunar_distance_re",
    ):
        if col in feat.columns:
            feat[col] = feat[col].ffill(limit=7)

    feat["proton_density_lag1d"] = feat["proton_density_cm3"].shift(1)
    cut = utc_ts(train_end)
    feat["proton_spike"] = _proton_spike(
        feat.set_index("issue_time")["proton_density_cm3"], cut
    ).to_numpy()
    feat["solar_cycle_phase"] = _cycle_phase(feat["sunspot_number"]).to_numpy()
    feat["f107_trend_90d"] = feat["f107"].diff(90)
    feat["kp_high"] = (feat["kp"] >= 5).astype(float)
    feat["dst_storm"] = (feat["dst_nt"] <= -50).astype(float)

    # Same-day catalog rate features (only events available by issue_time).
    quakes = quakes.copy()
    quakes["available_at"] = parse_utc(quakes["available_at"])
    feat["n_m45_7d"] = _count_available(quakes, feat["issue_time"], 4.5, 7, versions)
    feat["n_m6_7d"] = _count_available(quakes, feat["issue_time"], 6.0, 7, versions)
    feat["n_m7_30d"] = _count_available(quakes, feat["issue_time"], 7.0, 30, versions)
    feat["days_since_m7"] = _days_since(quakes, feat["issue_time"], 7.0, versions)
    # Independent events only: strip aftershock swarms, keep first shocks / background.
    # Causal decluster: an event is only an aftershock of something already reported.
    from eqpred.eval.decluster import decluster_causal

    fresh_quakes = decluster_causal(apply_completeness(quakes), space_km=200.0, time_days=10.0)
    feat["n_fresh_m45_7d"] = _count_available(fresh_quakes, feat["issue_time"], 4.5, 7, versions)
    feat["n_fresh_m45_30d"] = _count_available(fresh_quakes, feat["issue_time"], 4.5, 30, versions)
    feat["n_fresh_m6_7d"] = _count_available(fresh_quakes, feat["issue_time"], 6.0, 7, versions)
    feat["n_fresh_m6_30d"] = _count_available(fresh_quakes, feat["issue_time"], 6.0, 30, versions)
    feat["n_fresh_m7_30d"] = _count_available(fresh_quakes, feat["issue_time"], 7.0, 30, versions)
    feat["days_since_fresh_m7"] = _days_since(fresh_quakes, feat["issue_time"], 7.0, versions)
    feat = feat.merge(bvalue_frame(fresh_quakes, feat["issue_time"]), on="issue_time", how="left")
    # Origin-time twin: same counts without the reporting delay. Hunt-only leakage check.
    origin = quakes.copy()
    origin["available_at"] = origin["observed_at"]
    feat["n_m45_7d_origin"] = _count_available(origin, feat["issue_time"], 4.5, 7)
    nowcast = catalog_nowcast_frame(quakes, feat["issue_time"], versions=versions)
    feat = feat.merge(nowcast, on="issue_time", how="left")
    feat = feat.merge(migration_frame(feat["issue_time"], root=root), on="issue_time", how="left")
    feat = _join_wave2(feat, days, root)
    feat = _join_pore(feat, days, root)
    feat = _join_infrasound(feat, days, root)
    feat = _join_schumann(feat, days, root)
    from eqpred.eval.earth_state import attach_anomaly_features
    from eqpred.features.graph import attach_graph_features

    feat = attach_graph_features(feat, quakes, train_end=train_end)
    feat = attach_anomaly_features(feat, train_end=train_end)
    from eqpred.features.deep_counts import attach_deep_features

    feat = attach_deep_features(feat, quakes, root)

    dest = feature_dir(root) / "global_daily.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    feat = feat.copy()
    feat.to_parquet(dest, index=False)
    return feat


def _count_available(
    quakes: pd.DataFrame,
    issue: pd.Series,
    minmag: float,
    days: int,
    versions: pd.DataFrame | None = None,
) -> np.ndarray:
    work = apply_completeness(quakes)
    if versions is not None and not versions.empty:
        stamps = event_available_as(work, minmag, versions)
    else:
        stamps = work.loc[work["magnitude"] >= minmag, "available_at"]
    times = np.sort(parse_utc(stamps.dropna()).to_numpy(dtype="datetime64[ns]"))
    if times.size == 0:
        return np.zeros(len(issue), dtype=int)
    right = np.searchsorted(times, issue.to_numpy(dtype="datetime64[ns]"), side="right")
    left = np.searchsorted(
        times,
        (issue - pd.Timedelta(days=days)).to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    return (right - left).astype(int)


def _join_wave2(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    """Align Wave 2/3 raw if present. Unused on issued p until a head beats ETAS."""
    geonet = try_latest_raw("geonet.fdsn", root)
    if not geonet.empty and "available_at" in geonet.columns:
        geonet = geonet.copy()
        geonet["available_at"] = parse_utc(geonet["available_at"])
        feat["n_geonet_m3_7d"] = _count_available(geonet, feat["issue_time"], 3.0, 7)
        from eqpred.features.repeaters import days_since_hikurangi_repeater

        # Clock only. Catalog family, so the issued stack does not read it.
        feat["days_since_hikurangi_repeater"] = days_since_hikurangi_repeater(
            geonet, feat["issue_time"]
        )

    for source, slug in (("fdsn.japan", "japan"), ("fdsn.chile", "chile")):
        overlay = try_latest_raw(source, root)
        if overlay.empty or "available_at" not in overlay.columns:
            continue
        overlay = overlay.copy()
        overlay["available_at"] = parse_utc(overlay["available_at"])
        feat[f"n_fdsn_{slug}_m45_7d"] = _count_available(overlay, feat["issue_time"], 4.5, 7)

    feat = _join_gnss(feat, days, root)

    tec = try_latest_raw("igs.tec", root)
    if not tec.empty:
        cols = [c for c in tec.columns if c.startswith("tec_") and c.endswith("_tecu")]
        if cols:
            aligned = _asof_available(tec, days, cols)
            feat = feat.merge(aligned, on="issue_time", how="left")

    era5 = try_latest_raw("copernicus.era5", root)
    if not era5.empty:
        cols = [c for c in ("era5_sp_pa", "era5_swvl1") if c in era5.columns]
        if cols:
            aligned = _asof_available(era5, days, cols)
            feat = feat.merge(aligned, on="issue_time", how="left")

    feat = _join_magnet(feat, days, root)
    feat = _join_dart(feat, days, root)
    feat = _join_volcano(feat, days, root)
    feat = _join_swarm(feat, days, root)
    feat = _join_cses(feat, days, root)
    feat = _join_strain(feat, days, root)
    feat = _join_radon(feat, days, root)
    feat = _join_tremor(feat, days, root)
    feat = _join_wells(feat, days, root)
    feat = _join_cosmic2(feat, days, root)
    feat = _join_haarp(feat, days, root)
    feat = _join_dvv(feat, days, root)
    feat = _join_wspr(feat, days, root)
    feat = _join_wiki(feat, days, root)
    feat = _join_hydro_surface(feat, days, root)
    feat = _join_gnssa(feat, days, root)
    feat = _join_roti(feat, days, root)
    feat = _join_smap(feat, days, root)
    feat = _join_superdarn(feat, days, root)
    feat = _join_vlf(feat, days, root)
    feat = _join_tir(feat, days, root)
    feat = _join_tide(feat, days, root)
    feat = _join_insar(feat, days, root)
    feat = feat.copy()

    protons = try_latest_raw("noaa.goes_protons", root)
    if not protons.empty:
        cols = [c for c in protons.columns if c.startswith("goes_proton_")]
        if cols:
            aligned = _asof_available(protons, days, cols)
            feat = feat.merge(aligned, on="issue_time", how="left")
            if "goes_proton_10mev" in feat.columns:
                feat["goes_proton_10mev_7d"] = feat["goes_proton_10mev"].rolling(7, min_periods=1).max()
            if "goes_spe" in protons.columns:
                spe = _asof_available(protons, days, ["goes_spe"])
                feat = feat.merge(spe, on="issue_time", how="left")

    electrons = try_latest_raw("noaa.goes_electrons", root)
    if not electrons.empty:
        cols = [c for c in electrons.columns if str(c).startswith("electron_")]
        if cols:
            aligned = _asof_available(electrons, days, cols)
            feat = feat.merge(aligned, on="issue_time", how="left")

    grace = try_latest_raw("csr.grace", root)
    if not grace.empty:
        cols = [c for c in grace.columns if c.startswith("grace_")]
        if cols:
            aligned = _asof_available(grace, days, cols)
            feat = feat.merge(aligned, on="issue_time", how="left")

    so2 = try_latest_raw("cams.so2", root)
    if not so2.empty:
        cols = [c for c in so2.columns if c.startswith("so2_")]
        if cols:
            aligned = _asof_available(so2, days, cols)
            feat = feat.merge(aligned, on="issue_time", how="left")

    sse = try_latest_raw("slowslip.sse", root)
    if not sse.empty:
        cols = [c for c in sse.columns if c.startswith("sse_")]
        if cols:
            aligned = _asof_available(sse, days, cols)
            feat = feat.merge(aligned, on="issue_time", how="left")
            if "sse_hikurangi" in feat.columns:
                feat["sse_hikurangi_7d"] = feat["sse_hikurangi"].rolling(7, min_periods=1).max()
    climate = try_latest_raw("noaa.psl", root)
    if not climate.empty:
        cols = [c for c in climate.columns if str(c).startswith("climate_")]
        if cols:
            feat = feat.merge(_asof_available(climate, days, cols), on="issue_time", how="left")
    neutron = try_latest_raw("oulu.neutron", root)
    if not neutron.empty:
        cols = [c for c in neutron.columns if str(c).startswith("neutron_")]
        if cols:
            feat = feat.merge(_asof_available(neutron, days, cols), on="issue_time", how="left")
    gle = try_latest_raw("oulu.gle", root)
    if not gle.empty:
        cols = [c for c in gle.columns if str(c).startswith("cosmic_gle")]
        if cols:
            feat = feat.merge(_asof_available(gle, days, cols), on="issue_time", how="left")
    eop = try_latest_raw("iers.eop", root)
    if not eop.empty:
        cols = [c for c in eop.columns if str(c).startswith("eop_")]
        if cols:
            feat = feat.merge(_asof_available(eop, days, cols), on="issue_time", how="left")
    lisird = try_latest_raw("lasp.lisird", root)
    if not lisird.empty:
        cols = [c for c in lisird.columns if str(c).startswith("solar_")]
        if cols:
            feat = feat.merge(_asof_available(lisird, days, cols), on="issue_time", how="left")
    fire = try_latest_raw("cneos.fireball", root)
    if not fire.empty:
        cols = [c for c in fire.columns if str(c).startswith("dust_")]
        if cols:
            feat = feat.merge(_asof_available(fire, days, cols), on="issue_time", how="left")
    cme = try_latest_raw("cdaw.cme", root)
    if not cme.empty:
        cols = [c for c in cme.columns if str(c).startswith("cme_")]
        if cols:
            feat = feat.merge(_asof_available(cme, days, cols), on="issue_time", how="left")
    mjo = try_latest_raw("bom.mjo", root)
    if not mjo.empty:
        cols = [c for c in mjo.columns if str(c).startswith("climate_mjo")]
        if cols:
            feat = feat.merge(_asof_available(mjo, days, cols), on="issue_time", how="left")
    hpo = try_latest_raw("gfz.hpo", root)
    if not hpo.empty:
        cols = [c for c in hpo.columns if str(c).startswith(("hp_", "ap30"))]
        if cols:
            feat = feat.merge(_asof_available(hpo, days, cols), on="issue_time", how="left")
    donki = try_latest_raw("nasa.donki", root)
    if not donki.empty:
        cols = [
            c
            for c in donki.columns
            if str(c).startswith(("flare_", "hss_", "sep_", "ips_"))
        ]
        if cols:
            feat = feat.merge(_asof_available(donki, days, cols), on="issue_time", how="left")
    planets = planetary_frame(pd.DatetimeIndex(feat["issue_time"]))
    feat = feat.merge(planets, on="issue_time", how="left")
    from eqpred.features.quiet_residual import quiet_residual_columns

    return quiet_residual_columns(fresh_derived(feat))


def _nz_gnss_mean(frame: pd.DataFrame) -> pd.DataFrame:
    from eqpred.ingest.gnss import CHILE_SITES, JAPAN_SITES, NZ_SITES

    work = frame
    if "site" in frame.columns:
        abroad = {code.upper() for code in (*JAPAN_SITES, *CHILE_SITES)}
        sites = frame["site"].astype(str)
        work = frame.loc[~sites.str.upper().isin(abroad)].copy()
        if work.empty:
            return frame
        labels = work["site"].astype(str)
        per = work.loc[~labels.str.lower().eq("nzmean")]
        legacy = work.loc[labels.str.lower().eq("nzmean")]
        if not per.empty:
            short = per.loc[per["site"].astype(str).str.upper().isin(NZ_SITES)]
            chosen = short if not short.empty else per
            if not legacy.empty:
                chosen_days = set(parse_utc(chosen["observed_at"]).dt.floor("D"))
                legacy_days = parse_utc(legacy["observed_at"]).dt.floor("D")
                legacy = legacy.loc[~legacy_days.isin(chosen_days)]
                work = pd.concat([legacy, chosen], ignore_index=True)
            else:
                work = chosen
        elif not legacy.empty:
            work = legacy
    if "site" not in frame.columns:
        return work
    cols = {
        "east_mm": ("east_mm", "mean"),
        "north_mm": ("north_mm", "mean"),
        "vert_mm": ("vert_mm", "mean"),
        "available_at": ("available_at", "min"),
    }
    agg = {name: spec for name, spec in cols.items() if spec[0] in work.columns}
    daily = (
        work.assign(_day=parse_utc(work["observed_at"]).dt.floor("D"))
        .groupby("_day", as_index=False)
        .agg(**agg)
        .rename(columns={"_day": "observed_at"})
    )
    daily["observed_at"] = pd.to_datetime(daily["observed_at"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
    return daily


def _join_gnss(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    from eqpred.features.gnss_slip import gnss_slip_columns
    from eqpred.ingest.gnss import SITES, SLOW_SLIP_SITES

    geonet = try_latest_raw("geonet.gnss", root)
    ngl = try_latest_raw("ngl.gnss", root)
    mean_src = geonet if not geonet.empty else ngl
    if not mean_src.empty:
        mean = _nz_gnss_mean(mean_src) if "site" in mean_src.columns else mean_src
        for col in ("east_mm", "north_mm", "vert_mm"):
            if col not in mean.columns:
                continue
            aligned = _asof_available(mean, days, [col])
            feat = feat.merge(aligned.rename(columns={col: f"gnss_{col}"}), on="issue_time", how="left")
        if "gnss_east_mm" in feat.columns:
            feat["gnss_east_7d"] = feat["gnss_east_mm"].diff(7)

    if ngl.empty or "site" not in ngl.columns:
        return gnss_slip_columns(gnss_strain_columns(feat), root)
    keep_sites = {code.upper() for code in (*SITES, *SLOW_SLIP_SITES)}
    for site, part in ngl.groupby("site"):
        if str(site).upper() not in keep_sites:
            continue
        slug = str(site).lower()
        cols = [c for c in ("east_mm", "north_mm", "vert_mm") if c in part.columns]
        if not cols:
            continue
        aligned = _asof_available(part, days, cols)
        rename = {c: f"gnss_{slug}_{c}" for c in cols}
        feat = feat.merge(aligned.rename(columns=rename), on="issue_time", how="left")
        if slug.upper() in SLOW_SLIP_SITES and f"gnss_{slug}_east_mm" in feat.columns:
            feat[f"gnss_{slug}_slowslip_30d"] = feat[f"gnss_{slug}_east_mm"].diff(30)
    feat = feat.copy()
    return gnss_slip_columns(gnss_strain_columns(feat), root)


def _join_magnet(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    imag = try_latest_raw("intermagnet.gin", root)
    if imag.empty:
        return feat
    value_cols = [c for c in ("mag_h_nt", "mag_h_range_nt", "mag_dbdt_nt") if c in imag.columns]
    if not value_cols:
        return feat
    preferred = imag
    if "mag_station" in imag.columns:
        eyr = imag.loc[imag["mag_station"].astype(str).str.upper() == "EYR"]
        preferred = eyr if not eyr.empty else imag.drop_duplicates("observed_at")
        for station, part in imag.groupby("mag_station"):
            slug = str(station).lower()
            aligned = _asof_available(part, days, value_cols)
            rename = {c: f"{c.replace('mag_', f'mag_{slug}_')}" for c in value_cols}
            feat = feat.merge(aligned.rename(columns=rename), on="issue_time", how="left")
    cols = value_cols + (["mag_station"] if "mag_station" in preferred.columns else [])
    aligned = _asof_available(preferred, days, cols)
    return feat.merge(aligned, on="issue_time", how="left")


def _join_swarm(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    swarm = try_latest_raw("esa.swarm", root)
    if swarm.empty:
        return feat
    cols = [c for c in swarm.columns if str(c).startswith("swarm_")]
    if not cols:
        return feat
    aligned = _asof_available(swarm, days, cols)
    return feat.merge(aligned, on="issue_time", how="left")


def _join_cses(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    cses = try_latest_raw("esa.cses", root)
    if cses.empty:
        return feat
    cols = [c for c in cses.columns if str(c).startswith("cses_")]
    if not cols:
        return feat
    aligned = _asof_available(cses, days, cols)
    return feat.merge(aligned, on="issue_time", how="left")


def _join_radon(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    radon = try_latest_raw("radon.public", root)
    if radon.empty:
        return feat
    cols = [c for c in radon.columns if str(c).startswith("radon_")]
    if not cols:
        return feat
    aligned = _asof_available(radon, days, cols)
    return feat.merge(aligned, on="issue_time", how="left")


def _join_tremor(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    tremor = try_latest_raw("eqdb.tremor", root)
    if tremor.empty:
        return feat
    cols = [c for c in tremor.columns if str(c).startswith(("tremor_", "lfe_"))]
    if not cols:
        return feat
    return feat.merge(_asof_available(tremor, days, cols), on="issue_time", how="left")


def _join_wells(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    wells = try_latest_raw("hydro.wells", root)
    if wells.empty:
        return feat
    value_cols = [c for c in ("well_level_m", "well_range_m", "well_d1_m") if c in wells.columns]
    if not value_cols:
        return feat
    work = wells.copy()
    work["available_at"] = (
        parse_utc(work["available_at"]) if "available_at" in work.columns else parse_utc(work["observed_at"])
    )
    if "well_region" in work.columns:
        nz = work.loc[work["well_region"].astype(str).isin(("hikurangi", "alpine", "kermadec"))]
        if not nz.empty and "well_d1_m" in nz.columns:
            nz_daily = (
                nz.assign(_day=parse_utc(nz["observed_at"]).dt.floor("D"))
                .groupby("_day", as_index=False)
                .agg(well_nz_d1_m=("well_d1_m", "median"), available_at=("available_at", "min"))
                .rename(columns={"_day": "observed_at"})
            )
            nz_daily["observed_at"] = pd.to_datetime(nz_daily["observed_at"], utc=True).dt.strftime(
                "%Y-%m-%dT00:00:00Z"
            )
            feat = feat.merge(_asof_available(nz_daily, days, ["well_nz_d1_m"]), on="issue_time", how="left")
    named = {col: (col, "median") for col in value_cols}
    named["available_at"] = ("available_at", "min")
    mean = (
        work.assign(_day=parse_utc(work["observed_at"]).dt.floor("D"))
        .groupby("_day", as_index=False)
        .agg(**named)
        .rename(columns={"_day": "observed_at"})
    )
    mean["observed_at"] = pd.to_datetime(mean["observed_at"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
    return feat.merge(_asof_available(mean, days, value_cols), on="issue_time", how="left")


def _join_cosmic2(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    cosmic = try_latest_raw("ucar.cosmic2", root)
    if cosmic.empty:
        return feat
    cols = [c for c in cosmic.columns if str(c).startswith("scint_")]
    if not cols:
        return feat
    return feat.merge(_asof_available(cosmic, days, cols), on="issue_time", how="left")


def _join_haarp(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    haarp = try_latest_raw("gi.haarp", root)
    if haarp.empty:
        return feat
    cols = [c for c in haarp.columns if str(c).startswith("haarp_")]
    if not cols:
        return feat
    return feat.merge(_asof_available(haarp, days, cols), on="issue_time", how="left")


def _join_dvv(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    dvv = try_latest_raw("geonet.dvv", root)
    if dvv.empty:
        return feat
    cols = [c for c in dvv.columns if str(c).startswith("dvv_")]
    if not cols:
        return feat
    return feat.merge(_asof_available(dvv, days, cols), on="issue_time", how="left")


def _join_wspr(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    wspr = try_latest_raw("ham.wspr", root)
    if wspr.empty:
        return feat
    cols = [c for c in wspr.columns if str(c).startswith("wspr_")]
    if not cols:
        return feat
    return feat.merge(_asof_available(wspr, days, cols), on="issue_time", how="left")


def _join_wiki(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    wiki = try_latest_raw("wiki.pageviews", root)
    if wiki.empty:
        return feat
    cols = [
        c
        for c in wiki.columns
        if str(c).startswith("wiki_") and pd.api.types.is_numeric_dtype(wiki[c])
    ]
    if not cols:
        return feat
    return feat.merge(_asof_available(wiki, days, cols), on="issue_time", how="left")


def _join_prefix(feat: pd.DataFrame, days: pd.DatetimeIndex, source: str, prefix: str, root) -> pd.DataFrame:
    raw = try_latest_raw(source, root)
    if raw.empty:
        return feat
    cols = [c for c in raw.columns if str(c).startswith(prefix) and pd.api.types.is_numeric_dtype(raw[c])]
    if not cols:
        return feat
    return feat.merge(_asof_available(raw, days, cols), on="issue_time", how="left")


def _join_hydro_surface(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    return _join_prefix(feat, days, "hydro.surface", "hydro_", root)


def _join_gnssa(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    return _join_prefix(feat, days, "mgds.gnssa", "gnssa_", root)


def _join_roti(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    return _join_prefix(feat, days, "geonet.roti", "roti_", root)


def _join_smap(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    return _join_prefix(feat, days, "nasa.smap", "smap_", root)


def _join_superdarn(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    """Yesterday's Unwin gate share and speed. A later morning does not keep the last file."""
    from eqpred.ingest.unwin import COLUMNS, unwin_daily

    _ = days
    return _attach_yesterday(feat, unwin_daily(root), list(COLUMNS))


def _join_vlf(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    feat = _join_prefix(feat, days, "vlf.public", "vlf_", root)
    from eqpred.ingest.aardvark import read_local_aardvark

    local = read_local_aardvark(root)
    if local.empty:
        return feat
    cols = [c for c in ("aardvark_nwc_amp", "aardvark_nwc_phase") if c in local.columns]
    return feat.merge(_asof_available(local, days, cols), on="issue_time", how="left")


def _join_tir(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    return _join_prefix(feat, days, "nasa.tir", "tir_", root)


def _join_tide(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    tide = try_latest_raw("uhslc.tide", root)
    if tide.empty:
        return feat
    cols = [c for c in tide.columns if str(c).startswith("tide_")]
    if not cols:
        return feat
    aligned = _asof_available(tide, days, cols)
    feat = feat.merge(aligned, on="issue_time", how="left")
    if "tide_nz_mm" in feat.columns:
        feat["tide_nz_30d"] = feat["tide_nz_mm"].diff(30)
    return feat


def _join_insar(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    insar = try_latest_raw("comet.insar", root)
    if insar.empty:
        return feat
    cols = [c for c in insar.columns if str(c).startswith("insar_")]
    if not cols:
        return feat
    aligned = _asof_available(insar, days, cols)
    feat = feat.merge(aligned, on="issue_time", how="left")
    if "insar_los_mm" in feat.columns:
        feat["insar_los_30d"] = feat["insar_los_mm"].diff(30)
    return feat


def _join_dart(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    dart = try_latest_raw("noaa.dart", root)
    if dart.empty:
        return feat
    value_cols = [c for c in ("dart_height_m", "dart_range_m", "dart_d1_m") if c in dart.columns]
    if not value_cols:
        return feat
    work = dart.copy()
    work["available_at"] = parse_utc(work["available_at"]) if "available_at" in work.columns else parse_utc(work["observed_at"])
    if "dart_station" in work.columns:
        nz = work.loc[work["dart_station"].astype(str).str.upper().str.startswith("NZ")]
        if not nz.empty and "dart_height_m" in nz.columns:
            nz_daily = (
                nz.assign(_day=parse_utc(nz["observed_at"]).dt.floor("D"))
                .groupby("_day", as_index=False)
                .agg(dart_nz_m=("dart_height_m", "mean"), available_at=("available_at", "min"))
                .rename(columns={"_day": "observed_at"})
            )
            nz_daily["observed_at"] = pd.to_datetime(nz_daily["observed_at"], utc=True).dt.strftime(
                "%Y-%m-%dT00:00:00Z"
            )
            feat = feat.merge(_asof_available(nz_daily, days, ["dart_nz_m"]), on="issue_time", how="left")
        named = {}
        for col in value_cols:
            named[col] = (col, "min" if col == "dart_d1_m" else ("max" if col == "dart_range_m" else "mean"))
        named["available_at"] = ("available_at", "min")
        mean = (
            work.assign(_day=parse_utc(work["observed_at"]).dt.floor("D"))
            .groupby("_day", as_index=False)
            .agg(**named)
            .rename(columns={"_day": "observed_at"})
        )
        mean["observed_at"] = pd.to_datetime(mean["observed_at"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
        work = mean
    return feat.merge(_asof_available(work, days, value_cols), on="issue_time", how="left")


def _join_volcano(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    volcano = try_latest_raw("si.gvp", root)
    if volcano.empty:
        return feat
    cols = [c for c in volcano.columns if str(c).startswith("volcano_")]
    if not cols:
        return feat
    return feat.merge(_asof_available(volcano, days, cols), on="issue_time", how="left")


def _utc_day(stamps: pd.Series) -> pd.Series:
    return parse_utc(stamps).dt.floor("D").astype("datetime64[ns, UTC]")


def yesterday_frame(frame: pd.DataFrame, issue: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Yesterday's reading on this morning. A later morning does not keep the last value.

    Only readings with available_at <= issue morning are used. Delayed bulk
    releases stay hidden until the morning they could have been known.
    """
    issue_time = parse_utc(issue)
    empty = pd.DataFrame({"issue_time": issue_time, **{c: np.nan for c in columns}})
    if frame is None or frame.empty or "observed_at" not in getattr(frame, "columns", []):
        return empty
    work = frame.copy()
    work["observed_at"] = parse_utc(work["observed_at"])
    if "available_at" in work.columns:
        work["available_at"] = parse_utc(work["available_at"])
    else:
        work["available_at"] = work["observed_at"]
    work = work.dropna(subset=["observed_at", "available_at"])
    work["issue_time"] = _utc_day(work["observed_at"]) + pd.Timedelta(days=1)
    # Embargo: the issue morning must already have the reading in hand.
    work = work[work["available_at"] <= work["issue_time"]]
    cols = [c for c in columns if c in work.columns]
    if not cols or work.empty:
        return empty
    work = work[["issue_time", *cols]].dropna(subset=["issue_time"])
    work = work.drop_duplicates("issue_time", keep="last")
    left = pd.DataFrame({"issue_time": _utc_day(issue_time), "_row": np.arange(len(issue_time))})
    merged = left.merge(work, on="issue_time", how="left").sort_values("_row")
    out = pd.DataFrame({"issue_time": issue_time.to_numpy()})
    for col in columns:
        out[col] = pd.to_numeric(merged[col], errors="coerce").to_numpy() if col in merged.columns else np.nan
    return out


def _attach_yesterday(feat: pd.DataFrame, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in getattr(frame, "columns", [])]
    if not cols:
        return feat
    aligned = yesterday_frame(frame, feat["issue_time"], cols)
    feat = feat.drop(columns=[c for c in cols if c in feat.columns])
    return feat.merge(aligned, on="issue_time", how="left")


def _join_infrasound(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    """Previous day's station loudness. Eligible for the mainshock stack when the training years vary."""
    from eqpred.ingest.infrasound import infrasound_daily

    _ = days
    from eqpred.ingest.geonet_infra import etvz_daily

    frame = infrasound_daily(root)
    extra = etvz_daily(root)
    if not extra.empty:
        cols = [c for c in extra.columns if str(c).startswith("infra_") and c != "infra_etvz_location"]
        if frame.empty:
            frame = extra
        else:
            frame = merge_daily(frame, extra, cols)
    cols = [c for c in frame.columns if str(c).startswith("infra_")]
    return _attach_yesterday(feat, frame, cols)


def _join_pore(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    """Borehole and seafloor pressure from the day before. Eligible when the training years have a reading."""
    from eqpred.ingest.pore import pore_daily
    from eqpred.ingest.seafloor import apg_daily, rbr_daily

    _ = days
    frame = pore_daily(root)
    for extra in (apg_daily(root), rbr_daily(root)):
        if extra is None or extra.empty:
            continue
        cols = [c for c in extra.columns if str(c).startswith("pore_")]
        if frame.empty:
            frame = extra
        else:
            frame = merge_daily(frame, extra, cols)
    cols = [c for c in frame.columns if str(c).startswith("pore_")]
    return _attach_yesterday(feat, frame, cols)


def _join_schumann(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    """Previous day's Schumann band ratio. Eligible when the training years overlap the archive."""
    from eqpred.ingest.schumann import schumann_daily

    _ = days
    frame = schumann_daily(root)
    return _attach_yesterday(feat, frame, ["schumann_mode1"])


def _join_strain(feat: pd.DataFrame, days: pd.DatetimeIndex, root) -> pd.DataFrame:
    strain = try_latest_raw("earthscope.strain", root)
    if strain.empty:
        return feat
    value_cols = [
        c
        for c in ("strain_ee_nstrain", "strain_ee_range_nstrain", "strain_ee_dbdt_nstrain")
        if c in strain.columns
    ]
    if not value_cols:
        return feat
    preferred = strain
    if "strain_station" in strain.columns:
        first = strain.loc[strain["strain_station"].astype(str).str.upper() == "B004"]
        preferred = first if not first.empty else strain.drop_duplicates("observed_at")
        for station, part in strain.groupby("strain_station"):
            slug = str(station).lower()
            aligned = _asof_available(part, days, value_cols)
            rename = {c: c.replace("strain_", f"strain_{slug}_") for c in value_cols}
            feat = feat.merge(aligned.rename(columns=rename), on="issue_time", how="left")
    aligned = _asof_available(preferred, days, value_cols)
    return feat.merge(aligned, on="issue_time", how="left")


def _days_since(
    quakes: pd.DataFrame,
    issue: pd.Series,
    minmag: float,
    versions: pd.DataFrame | None = None,
) -> np.ndarray:
    work = apply_completeness(quakes)
    if versions is not None and not versions.empty:
        stamps = event_available_as(work, minmag, versions)
    else:
        stamps = work.loc[work["magnitude"] >= minmag, "available_at"]
    times = np.sort(parse_utc(stamps.dropna()).to_numpy(dtype="datetime64[ns]"))
    if times.size == 0:
        return np.full(len(issue), np.nan)
    idx = np.searchsorted(times, issue.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    out = np.full(len(issue), np.nan)
    ok = idx >= 0
    delta = issue.to_numpy(dtype="datetime64[ns]")[ok] - times[idx[ok]]
    out[ok] = delta / np.timedelta64(1, "D")
    return out
