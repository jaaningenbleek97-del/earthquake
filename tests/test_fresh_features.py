import pandas as pd

from eqpred.eval.hunt import driver_family
from eqpred.features.fresh import fresh_derived, planetary_frame


def test_planets_and_anomalies_are_non_catalog():
    days = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    planets = planetary_frame(days)
    assert planets["planet_hard_aspects"].between(0, 8).all()
    assert "planet_uranus_lon_deg" in planets.columns
    assert "planet_mars_lon_deg" in planets.columns
    assert driver_family("planet_jupiter_lon_deg") == "geometry"
    assert driver_family("x_kp_tidal_strain") == "combo"
    assert driver_family("climate_soi") == "climate"
    feat = pd.DataFrame(
        {
            "issue_time": days,
            "proton_density_cm3": range(120),
            "kp": [2.0] * 120,
            "tidal_strain": [1.6] * 60 + [2.2] * 60,
            "dst_storm": [0] * 100 + [1] * 20,
        }
    )
    out = fresh_derived(feat)
    assert "proton_density_cm3_z30" in out.columns
    assert "proton_density_cm3_d7" in out.columns
    assert "proton_density_cm3_lag14" in out.columns
    assert "proton_density_cm3_lag27" in out.columns
    assert "proton_density_cm3_lag54" in out.columns
    assert "proton_density_cm3_rec27" in out.columns
    assert "proton_density_cm3_d27" in out.columns
    assert "dust_meteor_shower" in out.columns
    assert "proton_density_cm3_z90" in out.columns
    assert "x_kp_tidal_strain" in out.columns
    assert "dst_storm_3d" in out.columns
    assert "cal_doy_sin" in out.columns
    assert driver_family("cal_doy_sin") == "geometry"
    assert driver_family("proton_density_cm3_z30") == "solar"
    assert driver_family("neutron_oulu_count") == "cosmic"
    assert driver_family("neutron_forbush") == "cosmic"
    counts = [6800.0 + (i % 5) for i in range(50)] + [5000.0] * 70
    neutrons = fresh_derived(
        pd.DataFrame({"issue_time": days, "neutron_oulu_count": counts, "kp": [2.0] * 120})
    )
    assert "neutron_oulu_count_z30" in neutrons.columns
    assert "x_neutron_oulu_count_kp" in neutrons.columns
    assert "neutron_forbush_pct" in neutrons.columns
    assert float(neutrons["neutron_forbush"].iloc[50]) == 1.0
    gle = fresh_derived(
        pd.DataFrame({"issue_time": days, "cosmic_gle": [0.0] * 119 + [1.0]})
    )
    assert float(gle["cosmic_gle_7d"].iloc[-1]) == 1.0
    assert driver_family("cosmic_gle") == "cosmic"
    poles = fresh_derived(pd.DataFrame({"issue_time": days}))
    assert "pole_dipole_lat" in poles.columns
    assert poles["pole_dipole_lat"].between(70.0, 90.0).all()
    assert driver_family("pole_dipole_lat") == "earth"
    assert driver_family("hp_max") == "solar"
    assert driver_family("hss_flag") == "solar"
    gnss = fresh_derived(
        pd.DataFrame(
            {
                "issue_time": days,
                "gnss_east_mm": [float(i) for i in range(120)],
                "mag_h_range_nt": [12.0] * 120,
                "mag_dbdt_nt": [1.5] * 120,
            }
        )
    )
    assert "gnss_east_1d" in gnss.columns
    assert "gnss_east_3d" in gnss.columns
    assert "ulf_pc5_nt" in gnss.columns
    assert "ulf_dbdt_nt" in gnss.columns
    assert driver_family("ulf_pc5_nt") == "wave2"
    assert driver_family("swarm_f_res_nt") == "wave2"
    assert driver_family("cses_hepd_e") == "wave2"
    assert driver_family("gnss_east_1d") == "wave2"
    swarm_feat = fresh_derived(
        pd.DataFrame({"issue_time": days, "swarm_f_res_nt": [float(i % 7) for i in range(120)]})
    )
    assert "swarm_f_res_nt_z30" in swarm_feat.columns
    assert driver_family("strain_ee_nstrain") == "wave2"
    assert driver_family("radon_bq_m3") == "wave2"
    again = fresh_derived(gnss)
    assert list(again.columns).count("ulf_pc5_nt") == 1
    watch = fresh_derived(
        pd.DataFrame(
            {
                "issue_time": days,
                "solar_wind_speed_kms": [400.0] * 60 + [620.0] * 60,
                "proton_density_cm3": [8.0] * 60 + [3.0] * 60,
                "solar_cycle_phase": [0.1] * 40 + [0.52] * 40 + [0.7] * 40,
                "imf_bx_nt": [2.0] * 80 + [-2.0] * 40,
                "climate_ao": [0.2] * 100 + [-2.0] * 20,
                "dst_nt": [-10.0] * 119 + [-80.0],
                "planet_hard_aspects": [0.0] * 100 + [3.0] * 20,
                "sse_hikurangi": [0.0] * 110 + [1.0] * 10,
                "tremor_n": [0.0] * 119 + [4.0],
                "well_d1_m": [0.0] * 119 + [-0.2],
                "scint_s4": [0.08 + 0.001 * ((i % 5) - 2) for i in range(119)] + [0.45],
                "haarp_on": [0.0] * 119 + [1.0],
                "volcano_vei4": [0.0] * 119 + [1.0],
                "lunar_phase_deg": [5.0] * 120,
                "lunar_declination_deg": [0.4] * 120,
                "radon_bq_m3": [80.0 + (i % 5) for i in range(119)] + [400.0],
                "grace_hikurangi_lwe_cm": [0.1 * ((i % 5) - 2) for i in range(119)] + [8.0],
                "so2_kermadec_kgm2": [1e-7 * (1 + 0.05 * (i % 5)) for i in range(119)] + [2e-6],
                "era5_swvl1": [0.02 + 0.0004 * ((i % 5) - 2) for i in range(119)] + [0.033],
                "climate_olr": [0.1 * ((i % 5) - 2) for i in range(119)] + [-2.5],
                "gnss_strain_nz_dilatation_30d": [0.05 * ((i % 5) - 2) for i in range(119)] + [4.0],
            }
        )
    )
    assert float(watch["coronal_hole_stream"].iloc[-1]) == 1.0
    assert float(watch["solar_max_near"].iloc[50]) == 1.0
    assert float(watch["solar_descending"].iloc[-1]) == 1.0
    assert float(watch["imf_sector_boundary"].iloc[80]) == 1.0
    assert float(watch["polar_vortex_weak"].iloc[-1]) == 1.0
    assert float(watch["ssc_dst"].iloc[-1]) == 1.0
    assert float(watch["planet_hard_on"].iloc[-1]) == 1.0
    assert float(watch["sse_on"].iloc[-1]) == 1.0
    assert float(watch["tremor_on"].iloc[-1]) == 1.0
    assert float(watch["well_drop"].iloc[-1]) == 1.0
    assert float(watch["scint_high"].iloc[-1]) == 1.0
    assert float(watch["haarp_on"].iloc[-1]) == 1.0
    assert float(watch["volcano_unrest"].iloc[-1]) == 1.0
    assert float(watch["solar_eclipse_window"].iloc[0]) == 1.0
    assert float(watch["radon_high"].iloc[-1]) == 1.0
    assert float(watch["grace_wet"].iloc[-1]) == 1.0
    assert float(watch["so2_high"].iloc[-1]) == 1.0
    assert float(watch["era5_wet"].iloc[-1]) == 1.0
    assert float(watch["olr_anomaly"].iloc[-1]) == 1.0
    assert float(watch["gnss_dilatation"].iloc[-1]) == 1.0
    assert driver_family("coronal_hole_stream") == "solar"
    assert driver_family("dart_drop") == "wave2"
    assert driver_family("volcano_unrest") == "volcano"
    assert driver_family("polar_vortex_weak") == "climate"
    assert driver_family("solar_days_since_max") == "solar"
    assert driver_family("radon_high") == "wave2"
    assert driver_family("tremor_on") == "slowslip"
    assert driver_family("well_drop") == "wave2"
    assert driver_family("scint_s4") == "wave2"
    assert driver_family("dvv_nz") == "wave2"
    assert driver_family("wspr_snr_med") == "wave2"
    assert driver_family("haarp_on") == "solar"
    assert driver_family("grace_wet") == "mass"
    assert driver_family("so2_high") == "volcano"
    assert driver_family("era5_wet") == "wave2"
    assert driver_family("olr_anomaly") == "climate"
    assert driver_family("gnss_dilatation") == "wave2"
    assert driver_family("earth_state_z") == "earth"
    assert driver_family("graph_inflow_hikurangi") == "graph"
    assert driver_family("analog_p_global") == "analog"
