from __future__ import annotations

"""Non-catalog derived series. No catalog counts.

Turns sources we already have into anomalies, streaks, and cheap planet
geometry so the hunter and the fresh-shock model see more than raw daily levels.
"""

import numpy as np
import pandas as pd

DERIVE_COLS = (
    "proton_density_cm3",
    "solar_wind_speed_kms",
    "imf_bz_nt",
    "imf_bx_nt",
    "imf_by_nt",
    "imf_b_nt",
    "kp",
    "dst_nt",
    "ae_nt",
    "al_nt",
    "ap",
    "f107",
    "flow_pressure_npa",
    "electric_field_mvm",
    "plasma_beta",
    "pc_index",
    "tec_tecu",
    "tec_hikurangi_tecu",
    "tec_alpine_tecu",
    "tec_kermadec_tecu",
    "tidal_strain",
    "lunar_distance_re",
    "gnss_east_mm",
    "gnss_east_1d",
    "gnss_east_3d",
    "gnss_east_7d",
    "gnss_strain_nz_dilatation_30d",
    "gnss_strain_nz_shear_30d",
    "tide_nz_mm",
    "insar_los_mm",
    "goes_proton_10mev",
    "xray_peak_wm2",
    "mag_h_nt",
    "mag_h_range_nt",
    "mag_dbdt_nt",
    "ulf_pc5_nt",
    "ulf_dbdt_nt",
    "swarm_f_res_nt",
    "swarm_f_res_range_nt",
    "swarm_f_res_dbdt_nt",
    "swarm_hikurangi_f_res_nt",
    "swarm_alpine_f_res_nt",
    "swarm_kermadec_f_res_nt",
    "cses_hepd_e",
    "cses_hepd_p",
    "strain_ee_nstrain",
    "strain_ee_range_nstrain",
    "strain_ee_dbdt_nstrain",
    "radon_bq_m3",
    "radon_range_bq_m3",
    "radon_dbdt_bq_m3",
    "grace_hikurangi_lwe_cm",
    "grace_alpine_lwe_cm",
    "so2_kermadec_kgm2",
    "sse_hikurangi",
    "spf_strength",
    "climate_soi",
    "climate_nao",
    "climate_ao",
    "climate_qbo",
    "climate_pdo",
    "climate_amo",
    "climate_pna",
    "climate_aao",
    "climate_nina34",
    "climate_olr",
    "era5_swvl1",
    "era5_sp_pa",
    "climate_mjo_amp",
    "neutron_oulu_count",
    "neutron_domc_count",
    "neutron_mosc_count",
    "neutron_newk_count",
    "neutron_sopo_count",
    "neutron_thul_count",
    "proton_temp_k",
    "eop_lod_s",
    "eop_pmx_arcsec",
    "eop_pmy_arcsec",
    "solar_tsi_wm2",
    "solar_mgii",
    "solar_lya",
    "cme_speed_kms",
    "cme_count",
    "dust_fireball_kt",
    "electron_2mev",
    "hp_max",
    "hp_mean",
    "flare_score",
    "flare_m_count",
    "sunspot_north",
    "sunspot_south",
    "dart_height_m",
    "dart_range_m",
    "dart_d1_m",
    "dart_nz_m",
    "volcano_vei_max",
    "tremor_n",
    "lfe_n",
    "well_level_m",
    "well_d1_m",
    "well_nz_d1_m",
    "scint_s4",
    "scint_s4_nz",
    "scint_s4_max",
    "haarp_on",
    "dvv_nz",
    "dvv_hikurangi",
    "dvv_alpine",
    "wspr_snr_med",
    "wspr_nz_snr",
    "wiki_eq_views",
    "hydro_nz",
    "hydro_gwet",
    "hydro_precip",
    "gnssa_hikurangi",
    "roti_nz",
    "roti_slips",
    "smap_sm",
    "sdarn_unwin",
    "vlf_amp",
    "tir_ts",
)

CROSS_PAIRS = (
    ("kp", "tidal_strain"),
    ("dst_nt", "lunar_distance_re"),
    ("imf_bz_nt", "tidal_strain"),
    ("proton_density_cm3", "near_new_or_full"),
    ("tec_tecu", "f107"),
    ("f107", "tidal_strain"),
    ("spf_strength", "kp"),
    ("ae_nt", "tidal_strain"),
    ("solar_wind_speed_kms", "imf_bz_nt"),
    ("lunar_declination_deg", "kp"),
    ("climate_soi", "tidal_strain"),
    ("flow_pressure_npa", "imf_bz_nt"),
    ("pc_index", "tidal_strain"),
    ("neutron_oulu_count", "kp"),
    ("eop_lod_s", "kp"),
    ("solar_tsi_wm2", "tidal_strain"),
    ("cme_speed_kms", "dst_nt"),
    ("hp_max", "tidal_strain"),
    ("flare_score", "kp"),
    ("dust_fireball_kt", "kp"),
    ("climate_mjo_amp", "tidal_strain"),
    ("solar_lya", "tec_tecu"),
    ("climate_aao", "eop_lod_s"),
    ("electron_2mev", "kp"),
    ("neutron_sopo_count", "imf_bz_nt"),
    ("climate_olr", "tidal_strain"),
    ("swarm_f_res_nt", "kp"),
)


def _mean_lon(t_century: float, epoch_deg: float, rate_deg: float) -> float:
    return (epoch_deg + rate_deg * t_century) % 360.0


def _angle_to_cardinal(delta_deg: float) -> float:
    """Smallest distance to 0 / 90 / 180 / 270 degrees."""
    wrapped = abs(delta_deg) % 360.0
    wrapped = min(wrapped, 360.0 - wrapped)
    return float(min(wrapped, abs(wrapped - 90.0), abs(wrapped - 180.0), abs(wrapped - 270.0)))


def planetary_frame(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Cheap mean longitudes. Replayable, no catalog, no download."""
    rows: list[dict[str, float]] = []
    for ts in days:
        utc = ts.to_pydatetime()
        if utc.tzinfo is None:
            from datetime import timezone

            utc = utc.replace(tzinfo=timezone.utc)
        # Julian centuries from J2000
        y, m, d = utc.year, utc.month, utc.day + utc.hour / 24.0
        if m <= 2:
            y -= 1
            m += 12
        a = y // 100
        b = 2 - a + a // 4
        jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
        t = (jd - 2451545.0) / 36525.0
        sun = _mean_lon(t, 280.46646, 36000.76983)
        mercury = _mean_lon(t, 252.250906, 149472.674636)
        venus = _mean_lon(t, 181.979801, 58517.815387)
        earth = (sun + 180.0) % 360.0
        mars = _mean_lon(t, 355.433, 19140.2993)
        jupiter = _mean_lon(t, 34.351519, 3034.905662)
        saturn = _mean_lon(t, 50.077444, 1222.113794)
        uranus = _mean_lon(t, 314.055005, 428.466998)
        neptune = _mean_lon(t, 304.348665, 218.486200)
        hard = 0.0
        pairs = (
            (sun, jupiter),
            (sun, saturn),
            (venus, jupiter),
            (jupiter, saturn),
            (mars, jupiter),
            (mercury, sun),
            (sun, uranus),
            (jupiter, uranus),
        )
        for left, right in pairs:
            if _angle_to_cardinal(left - right) <= 8.0:
                hard += 1.0
        rows.append(
            {
                "issue_time": ts,
                "planet_sun_lon_deg": sun,
                "planet_mercury_lon_deg": mercury,
                "planet_venus_lon_deg": venus,
                "planet_earth_lon_deg": earth,
                "planet_mars_lon_deg": mars,
                "planet_jupiter_lon_deg": jupiter,
                "planet_saturn_lon_deg": saturn,
                "planet_uranus_lon_deg": uranus,
                "planet_neptune_lon_deg": neptune,
                "planet_hard_aspects": hard,
                "planet_sun_jupiter_sep": _angle_to_cardinal(sun - jupiter),
                "planet_sun_saturn_sep": _angle_to_cardinal(sun - saturn),
                "planet_mars_jupiter_sep": _angle_to_cardinal(mars - jupiter),
                "planet_mercury_sun_sep": _angle_to_cardinal(mercury - sun),
                "planet_sun_uranus_sep": _angle_to_cardinal(sun - uranus),
            }
        )
    return pd.DataFrame(rows)


def _igrf_dipole(year_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IGRF-13 2020 dipole geomagnetic pole. Slow wander, not a measurement."""
    dt = np.asarray(year_frac, dtype=float) - 2020.0
    g10 = -29404.8 + 5.7 * dt
    g11 = -1450.9 + 7.4 * dt
    h11 = 4652.5 - 25.9 * dt
    colat = np.degrees(np.arctan2(np.hypot(g11, h11), np.abs(g10)))
    lon = np.degrees(np.arctan2(h11, g11))
    return 90.0 - colat, lon


def _igrf_dipole_lat(year_frac: np.ndarray) -> np.ndarray:
    lat, _lon = _igrf_dipole(year_frac)
    return lat


def _igrf_dipole_lon(year_frac: np.ndarray) -> np.ndarray:
    _lat, lon = _igrf_dipole(year_frac)
    return lon


def fresh_derived(feat: pd.DataFrame) -> pd.DataFrame:
    """Anomalies and storm memory on fresh-shock columns already in the daily stack."""
    work = feat.copy()
    if "gnss_east_mm" in work.columns:
        east = pd.to_numeric(work["gnss_east_mm"], errors="coerce")
        if "gnss_east_1d" not in work.columns:
            work["gnss_east_1d"] = east.diff(1)
        if "gnss_east_3d" not in work.columns:
            work["gnss_east_3d"] = east.diff(3)
        if "gnss_east_7d" not in work.columns:
            work["gnss_east_7d"] = east.diff(7)
    if "mag_h_range_nt" in work.columns and "ulf_pc5_nt" not in work.columns:
        work["ulf_pc5_nt"] = pd.to_numeric(work["mag_h_range_nt"], errors="coerce")
    if "mag_dbdt_nt" in work.columns and "ulf_dbdt_nt" not in work.columns:
        work["ulf_dbdt_nt"] = pd.to_numeric(work["mag_dbdt_nt"], errors="coerce")
    extra: dict[str, pd.Series] = {}
    days = None
    if "issue_time" in work.columns:
        days = pd.DatetimeIndex(pd.to_datetime(work["issue_time"], utc=True))
        doy = days.dayofyear.to_numpy()
        extra["cal_doy_sin"] = pd.Series(np.sin(2.0 * np.pi * doy / 365.25), index=work.index)
        extra["cal_doy_cos"] = pd.Series(np.cos(2.0 * np.pi * doy / 365.25), index=work.index)
        extra["cal_weekday"] = pd.Series(days.dayofweek.to_numpy(dtype=float), index=work.index)
        extra["cal_weekend"] = pd.Series((days.dayofweek.to_numpy() >= 5).astype(float), index=work.index)
        extra["cal_payday"] = pd.Series(((days.day >= 14) & (days.day <= 16)).astype(float), index=work.index)
    for col in DERIVE_COLS:
        if col not in work.columns:
            continue
        series = pd.to_numeric(work[col], errors="coerce")
        mean30 = series.rolling(30, min_periods=10).mean()
        std30 = series.rolling(30, min_periods=10).std()
        extra[f"{col}_z30"] = (series - mean30) / std30.replace(0, np.nan)
        extra[f"{col}_d1"] = series.diff(1)
        extra[f"{col}_d7"] = series.diff(7)
        extra[f"{col}_d14"] = series.diff(14)
        extra[f"{col}_d27"] = series.diff(27)
        extra[f"{col}_lag14"] = series.shift(14)
        extra[f"{col}_lag27"] = series.shift(27)
        extra[f"{col}_lag54"] = series.shift(54)
        extra[f"{col}_rec27"] = series - series.shift(27)
        extra[f"{col}_max7"] = series.rolling(7, min_periods=3).max()
        extra[f"{col}_max14"] = series.rolling(14, min_periods=5).max()
        extra[f"{col}_max27"] = series.rolling(27, min_periods=8).max()
        extra[f"{col}_min7"] = series.rolling(7, min_periods=3).min()
        extra[f"{col}_range7"] = extra[f"{col}_max7"] - extra[f"{col}_min7"]
        mean90 = series.rolling(90, min_periods=20).mean()
        std90 = series.rolling(90, min_periods=20).std()
        extra[f"{col}_z90"] = (series - mean90) / std90.replace(0, np.nan)
    if "lunar_phase_deg" in work.columns:
        phase = np.radians(pd.to_numeric(work["lunar_phase_deg"], errors="coerce"))
        extra["lunar_phase_sin"] = np.sin(phase)
        extra["lunar_phase_cos"] = np.cos(phase)
        extra["lunar_spring"] = np.cos(2.0 * phase)
    if "imf_by_nt" in work.columns and "imf_bz_nt" in work.columns:
        by = pd.to_numeric(work["imf_by_nt"], errors="coerce")
        bz = pd.to_numeric(work["imf_bz_nt"], errors="coerce")
        extra["imf_clock_deg"] = np.degrees(np.arctan2(by, bz))
    for left, right in CROSS_PAIRS:
        if left not in work.columns or right not in work.columns:
            continue
        a = pd.to_numeric(work[left], errors="coerce")
        b = pd.to_numeric(work[right], errors="coerce")
        az = (a - a.rolling(90, min_periods=20).mean()) / a.rolling(90, min_periods=20).std().replace(0, np.nan)
        bz = (b - b.rolling(90, min_periods=20).mean()) / b.rolling(90, min_periods=20).std().replace(0, np.nan)
        extra[f"x_{left}_{right}"] = az * bz
    if "imf_bz_nt" in work.columns:
        south = (pd.to_numeric(work["imf_bz_nt"], errors="coerce") < -5).astype(float)
        extra["bz_south_7d"] = south.rolling(7, min_periods=1).sum()
    if "dst_storm" in work.columns:
        extra["dst_storm_3d"] = pd.to_numeric(work["dst_storm"], errors="coerce").rolling(3, min_periods=1).max()
    if "kp_high" in work.columns:
        extra["kp_high_3d"] = pd.to_numeric(work["kp_high"], errors="coerce").rolling(3, min_periods=1).max()
    if "tec_tecu" in work.columns:
        tec = pd.to_numeric(work["tec_tecu"], errors="coerce")
        extra["tec_tecu_anom90"] = tec - tec.rolling(90, min_periods=20).mean()
    if "goes_spe" in work.columns:
        extra["goes_spe_7d"] = pd.to_numeric(work["goes_spe"], errors="coerce").rolling(7, min_periods=1).max()
    if "electron_2mev" in work.columns:
        extra["electron_2mev_7d"] = pd.to_numeric(work["electron_2mev"], errors="coerce").rolling(7, min_periods=1).max()
    z30 = extra.get("neutron_oulu_count_z30")
    if z30 is not None:
        extra["neutron_forbush"] = (pd.to_numeric(z30, errors="coerce") < -2.0).astype(float)
    if "neutron_oulu_count" in work.columns:
        drop = pd.to_numeric(work["neutron_oulu_count"], errors="coerce").pct_change(1)
        extra["neutron_forbush_pct"] = (drop < -0.03).astype(float)
    if "cosmic_gle" in work.columns:
        extra["cosmic_gle_7d"] = pd.to_numeric(work["cosmic_gle"], errors="coerce").rolling(7, min_periods=1).max()
        extra["cosmic_gle_14d"] = pd.to_numeric(work["cosmic_gle"], errors="coerce").rolling(14, min_periods=1).max()
    if "cme_halo" in work.columns:
        extra["cme_halo_7d"] = pd.to_numeric(work["cme_halo"], errors="coerce").rolling(7, min_periods=1).max()
        extra["cme_halo_14d"] = pd.to_numeric(work["cme_halo"], errors="coerce").rolling(14, min_periods=1).max()
    if "dust_fireball" in work.columns:
        extra["dust_fireball_7d"] = pd.to_numeric(work["dust_fireball"], errors="coerce").rolling(7, min_periods=1).max()
        extra["dust_fireball_14d"] = pd.to_numeric(work["dust_fireball"], errors="coerce").rolling(14, min_periods=1).max()
    speed = work.get("solar_wind_speed_kms")
    bt = work.get("imf_b_nt")
    bz = work.get("imf_bz_nt")
    if speed is not None and bt is not None and bz is not None:
        v = pd.to_numeric(speed, errors="coerce").clip(lower=1.0)
        btot = pd.to_numeric(bt, errors="coerce").abs()
        bz_n = pd.to_numeric(bz, errors="coerce")
        by_n = pd.to_numeric(work["imf_by_nt"], errors="coerce") if "imf_by_nt" in work.columns else bz_n
        clock = np.arctan2(by_n, bz_n)
        half = np.clip(np.abs(clock) / 2.0, 0.0, np.pi)
        extra["solar_newell"] = (v ** (4.0 / 3.0)) * (btot ** (2.0 / 3.0)) * (np.sin(half) ** 8)
        extra["solar_epsilon"] = v * (btot ** 2) * (np.sin(half) ** 4)
    if "imf_bx_nt" in work.columns:
        extra["imf_toward"] = (pd.to_numeric(work["imf_bx_nt"], errors="coerce") > 0).astype(float)
    if "lunar_phase_deg" in work.columns and "lunar_declination_deg" in work.columns:
        phase = pd.to_numeric(work["lunar_phase_deg"], errors="coerce")
        decl = pd.to_numeric(work["lunar_declination_deg"], errors="coerce").abs()
        syzygy = np.minimum(np.minimum(phase, (phase - 180.0).abs()), (360.0 - phase))
        extra["lunar_eclipse_window"] = ((syzygy < 18.0) & (decl < 1.8)).astype(float)
    if days is not None:
        doy = days.dayofyear.to_numpy()
        peaks = np.array([4, 22, 112, 125, 225, 294, 317, 348, 356], dtype=float)
        nearest = np.min(np.abs(doy[:, None] - peaks[None, :]), axis=1)
        extra["dust_meteor_shower"] = pd.Series((nearest <= 2.0).astype(float), index=work.index)
        epoch = pd.Timestamp("1900-01-01", tz="UTC")
        extra["solar_bartels"] = pd.Series(((days - epoch).days.to_numpy() % 27).astype(float), index=work.index)
        year = days.year.to_numpy() + (days.dayofyear.to_numpy() - 1) / 365.25
        extra["pole_dipole_lat"] = pd.Series(_igrf_dipole_lat(year), index=work.index)
        extra["pole_dipole_lon"] = pd.Series(_igrf_dipole_lon(year), index=work.index)
    if "hss_flag" in work.columns:
        extra["hss_7d"] = pd.to_numeric(work["hss_flag"], errors="coerce").rolling(7, min_periods=1).max()
    if "flare_x_count" in work.columns:
        extra["flare_x_7d"] = pd.to_numeric(work["flare_x_count"], errors="coerce").rolling(7, min_periods=1).max()
    speed = work.get("solar_wind_speed_kms")
    dens = work.get("proton_density_cm3")
    if speed is not None and dens is not None:
        extra["coronal_hole_stream"] = (
            (pd.to_numeric(speed, errors="coerce") >= 500.0)
            & (pd.to_numeric(dens, errors="coerce") <= 5.0)
        ).astype(float)
    if "imf_bx_nt" in work.columns:
        bx = pd.to_numeric(work["imf_bx_nt"], errors="coerce")
        extra["imf_sector_boundary"] = ((np.sign(bx) * np.sign(bx.shift(1))) < 0).astype(float)
    if "solar_cycle_phase" in work.columns:
        phase = pd.to_numeric(work["solar_cycle_phase"], errors="coerce")
        extra["solar_max_near"] = ((phase > 0.40) & (phase < 0.65)).astype(float)
        extra["solar_min_near"] = ((phase < 0.15) | (phase > 0.85)).astype(float)
        extra["solar_descending"] = (phase > 0.55).astype(float)
        crossed = (phase.shift(1) >= 0.50) & (phase < 0.50)
        last_max = pd.Series(np.where(crossed, np.arange(len(phase)), np.nan), index=work.index)
        last_max = last_max.ffill()
        extra["solar_days_since_max"] = pd.Series(np.arange(len(phase)), index=work.index) - last_max
    if "climate_ao" in work.columns:
        ao = pd.to_numeric(work["climate_ao"], errors="coerce")
        extra["polar_vortex_weak"] = (ao <= -1.5).astype(float)
    if "ap" in work.columns:
        extra["ap_storm"] = (pd.to_numeric(work["ap"], errors="coerce") >= 50.0).astype(float)
    flare_x = extra.get("flare_x_7d")
    if flare_x is None and "flare_x_count" in work.columns:
        flare_x = pd.to_numeric(work["flare_x_count"], errors="coerce")
    if flare_x is not None:
        extra["flare_x_on"] = (pd.to_numeric(flare_x, errors="coerce") > 0).astype(float)
    def _on(value) -> pd.Series:
        series = pd.to_numeric(value, errors="coerce")
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=work.index)
        return series.reindex(work.index).fillna(0) > 0.5

    if "hss_flag" in work.columns or "hss_7d" in extra or "coronal_hole_stream" in extra:
        extra["hss_on"] = (
            _on(work["hss_flag"] if "hss_flag" in work.columns else 0.0)
            | _on(extra.get("coronal_hole_stream", 0.0))
            | _on(extra.get("hss_7d", 0.0))
        ).astype(float)
    if "sep_flag" in work.columns or "goes_spe" in work.columns:
        extra["spe_on"] = (
            _on(work["sep_flag"] if "sep_flag" in work.columns else 0.0)
            | _on(work["goes_spe"] if "goes_spe" in work.columns else 0.0)
        ).astype(float)
    if "planet_hard_aspects" in work.columns:
        extra["planet_hard_on"] = (pd.to_numeric(work["planet_hard_aspects"], errors="coerce") >= 2.0).astype(float)
    if "dst_nt" in work.columns:
        dst = pd.to_numeric(work["dst_nt"], errors="coerce")
        extra["ssc_dst"] = (dst.diff(1) <= -30.0).astype(float)
    if "sunspot_north" in work.columns and "sunspot_south" in work.columns:
        north = pd.to_numeric(work["sunspot_north"], errors="coerce")
        south = pd.to_numeric(work["sunspot_south"], errors="coerce")
        total = (north + south).replace(0, np.nan)
        extra["sunspot_asymmetry"] = (north - south).abs() / total
        extra["sunspot_hemisphere_skew"] = ((north - south).abs() >= 20.0).astype(float)
    if "dart_d1_m" in work.columns:
        drop = pd.to_numeric(work["dart_d1_m"], errors="coerce")
        extra["dart_drop"] = (drop <= -0.04).astype(float)
    z_ulf = extra.get("ulf_pc5_nt_z30")
    if z_ulf is not None:
        extra["ulf_high"] = (pd.to_numeric(z_ulf, errors="coerce").abs() >= 2.0).astype(float)
    z_tec = extra.get("tec_tecu_z30")
    if z_tec is not None:
        extra["tec_storm"] = (pd.to_numeric(z_tec, errors="coerce").abs() >= 2.0).astype(float)
    z_radon = extra.get("radon_bq_m3_z30")
    if z_radon is not None:
        extra["radon_high"] = (pd.to_numeric(z_radon, errors="coerce").abs() >= 2.0).astype(float)
    z_grace = extra.get("grace_hikurangi_lwe_cm_z30")
    if z_grace is not None:
        extra["grace_wet"] = (pd.to_numeric(z_grace, errors="coerce") >= 2.0).astype(float)
    z_so2 = extra.get("so2_kermadec_kgm2_z30")
    if z_so2 is not None:
        extra["so2_high"] = (pd.to_numeric(z_so2, errors="coerce") >= 2.0).astype(float)
    z_strain = extra.get("strain_ee_dbdt_nstrain_z30")
    if z_strain is None:
        z_strain = extra.get("strain_ee_nstrain_z30")
    if z_strain is not None:
        extra["strain_pulse"] = (pd.to_numeric(z_strain, errors="coerce").abs() >= 2.0).astype(float)
    z_soil = extra.get("era5_swvl1_z30")
    if z_soil is not None:
        extra["era5_wet"] = (pd.to_numeric(z_soil, errors="coerce") >= 2.0).astype(float)
    z_olr = extra.get("climate_olr_z30")
    if z_olr is not None:
        extra["olr_anomaly"] = (pd.to_numeric(z_olr, errors="coerce").abs() >= 2.0).astype(float)
    z_gnss = extra.get("gnss_strain_nz_dilatation_30d_z30")
    if z_gnss is not None:
        extra["gnss_dilatation"] = (pd.to_numeric(z_gnss, errors="coerce").abs() >= 2.0).astype(float)
    if "sse_hikurangi" in work.columns:
        extra["sse_on"] = (pd.to_numeric(work["sse_hikurangi"], errors="coerce") > 0.5).astype(float)
    if "tremor_on" not in work.columns and ("tremor_n" in work.columns or "lfe_n" in work.columns):
        extra["tremor_on"] = (
            (pd.to_numeric(work["tremor_n"], errors="coerce").fillna(0) if "tremor_n" in work.columns else 0)
            + (pd.to_numeric(work["lfe_n"], errors="coerce").fillna(0) if "lfe_n" in work.columns else 0)
            > 0
        ).astype(float)
    z_well = extra.get("well_d1_m_z30")
    if z_well is None:
        z_well = extra.get("well_nz_d1_m_z30")
    if z_well is not None:
        extra["well_drop"] = (pd.to_numeric(z_well, errors="coerce") <= -2.0).astype(float)
    elif "well_d1_m" in work.columns:
        extra["well_drop"] = (pd.to_numeric(work["well_d1_m"], errors="coerce") <= -0.05).astype(float)
    z_scint = extra.get("scint_s4_z30")
    if z_scint is None:
        z_scint = extra.get("scint_s4_nz_z30")
    if z_scint is not None:
        extra["scint_high"] = (pd.to_numeric(z_scint, errors="coerce") >= 2.0).astype(float)
    if "haarp_on" in work.columns:
        extra["haarp_on"] = _on(work["haarp_on"])
    z_dvv = extra.get("dvv_nz_z30")
    if z_dvv is None:
        z_dvv = extra.get("dvv_hikurangi_z30")
    if z_dvv is not None:
        extra["dvv_drop"] = (pd.to_numeric(z_dvv, errors="coerce") <= -2.0).astype(float)
    z_wspr = extra.get("wspr_snr_med_z30")
    if z_wspr is None:
        z_wspr = extra.get("wspr_nz_snr_z30")
    if z_wspr is not None:
        extra["wspr_quiet"] = (pd.to_numeric(z_wspr, errors="coerce") <= -2.0).astype(float)
    z_wiki = extra.get("wiki_eq_views_z30")
    if z_wiki is not None:
        extra["wiki_hot"] = (pd.to_numeric(z_wiki, errors="coerce") >= 2.0).astype(float)
    z_wet = extra.get("hydro_gwet_z30")
    if z_wet is None:
        z_wet = extra.get("smap_sm_z30")
    if z_wet is not None:
        extra["hydro_wet"] = (pd.to_numeric(z_wet, errors="coerce") >= 2.0).astype(float)
    z_roti = extra.get("roti_nz_z30")
    if z_roti is not None:
        extra["roti_high"] = (pd.to_numeric(z_roti, errors="coerce") >= 2.0).astype(float)
    if "volcano_vei4" in work.columns or "volcano_nz" in work.columns:
        extra["volcano_unrest"] = (
            _on(work["volcano_vei4"] if "volcano_vei4" in work.columns else 0.0)
            | _on(work["volcano_nz"] if "volcano_nz" in work.columns else 0.0)
        ).astype(float)
    if "solar_eclipse_window" not in work.columns and "lunar_phase_deg" in work.columns and "lunar_declination_deg" in work.columns:
        phase = pd.to_numeric(work["lunar_phase_deg"], errors="coerce")
        decl = pd.to_numeric(work["lunar_declination_deg"], errors="coerce").abs()
        extra["solar_eclipse_window"] = ((np.minimum(phase, 360.0 - phase) < 18.0) & (decl < 1.8)).astype(float)
    if "lunar_eclipse_geom" in work.columns and "lunar_eclipse_window" not in work.columns:
        extra["lunar_eclipse_window"] = pd.to_numeric(work["lunar_eclipse_geom"], errors="coerce")
    new = {k: v for k, v in extra.items() if k not in work.columns}
    if not new:
        return work
    return pd.concat([work, pd.DataFrame(new, index=work.index)], axis=1)
