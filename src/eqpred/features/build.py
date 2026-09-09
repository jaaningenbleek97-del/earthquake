from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.features.coulomb import coulomb_frame
from eqpred.features.cusp import cusp_frame
from eqpred.features.davidson import apply_train_budget, field_strength, significant_windows
from eqpred.features.leftover import leftover_derived, planetary_frame
from eqpred.features.nowcast import catalog_nowcast_frame
from eqpred.features.migration import migration_frame
from eqpred.catalog import apply_completeness, event_available_as, load_comcat, load_comcat_versions
from eqpred.ingest.run import latest_raw, try_latest_raw
from eqpred.paths import feature_dir
from eqpred.stamp import parse_utc, utc_ts


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

    windows = significant_windows(spf, quakes, version="v2")
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
    silso_asof = _asof_available(silso, days, ["sunspot_number"])
    geom_asof = _asof_available(
        geom,
        days,
        [
            "lunar_phase_deg",
            "lunar_distance_re",
            "lunar_declination_deg",
            "tidal_strain",
            "near_new_or_full",
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
    # Origin-time twin: same counts without the reporting delay. Hunt-only leakage check.
    origin = quakes.copy()
    origin["available_at"] = origin["observed_at"]
    feat["n_m45_7d_origin"] = _count_available(origin, feat["issue_time"], 4.5, 7)
    nowcast = catalog_nowcast_frame(quakes, feat["issue_time"], versions=versions)
    feat = feat.merge(nowcast, on="issue_time", how="left")
    feat = feat.merge(migration_frame(feat["issue_time"], root=root), on="issue_time", how="left")
    feat = _join_wave2(feat, days, root)

    dest = feature_dir(root) / "global_daily.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
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
    return leftover_derived(feat)


def _nz_gnss_mean(frame: pd.DataFrame) -> pd.DataFrame:
    from eqpred.ingest.gnss import NZ_SITES

    work = frame
    if "site" in frame.columns:
        work = frame.loc[frame["site"].isin(NZ_SITES)]
        if work.empty:
            return frame
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
    from eqpred.ingest.gnss import SLOW_SLIP_SITES

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
        return feat
    for site, part in ngl.groupby("site"):
        slug = str(site).lower()
        cols = [c for c in ("east_mm", "north_mm", "vert_mm") if c in part.columns]
        if not cols:
            continue
        aligned = _asof_available(part, days, cols)
        rename = {c: f"gnss_{slug}_{c}" for c in cols}
        feat = feat.merge(aligned.rename(columns=rename), on="issue_time", how="left")
        if slug.upper() in SLOW_SLIP_SITES and f"gnss_{slug}_east_mm" in feat.columns:
            feat[f"gnss_{slug}_slowslip_30d"] = feat[f"gnss_{slug}_east_mm"].diff(30)
    return feat


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
