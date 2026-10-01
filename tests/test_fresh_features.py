import pandas as pd

from eqpred.eval.hunt import driver_family
from eqpred.features.leftover import leftover_derived, planetary_frame


def test_planets_and_anomalies_are_leftover_not_catalog():
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
    out = leftover_derived(feat)
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
    neutrons = leftover_derived(
        pd.DataFrame({"issue_time": days, "neutron_oulu_count": counts, "kp": [2.0] * 120})
    )
    assert "neutron_oulu_count_z30" in neutrons.columns
    assert "x_neutron_oulu_count_kp" in neutrons.columns
    assert "neutron_forbush_pct" in neutrons.columns
    assert float(neutrons["neutron_forbush"].iloc[50]) == 1.0
    gle = leftover_derived(
        pd.DataFrame({"issue_time": days, "cosmic_gle": [0.0] * 119 + [1.0]})
    )
    assert float(gle["cosmic_gle_7d"].iloc[-1]) == 1.0
    assert driver_family("cosmic_gle") == "cosmic"
    poles = leftover_derived(pd.DataFrame({"issue_time": days}))
    assert "pole_dipole_lat" in poles.columns
    assert poles["pole_dipole_lat"].between(70.0, 90.0).all()
    assert driver_family("pole_dipole_lat") == "earth"
    assert driver_family("hp_max") == "solar"
    assert driver_family("hss_flag") == "solar"
    gnss = leftover_derived(
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
    assert driver_family("gnss_east_1d") == "wave2"
    assert driver_family("strain_ee_nstrain") == "wave2"
    assert driver_family("radon_bq_m3") == "wave2"
    again = leftover_derived(gnss)
    assert list(again.columns).count("ulf_pc5_nt") == 1
