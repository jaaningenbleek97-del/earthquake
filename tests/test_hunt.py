from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.honest import junk_knob
from eqpred.eval.hunt import LAGS_DAYS, _driver_columns, benjamini_hochberg, driver_family, run_hunt


def test_family_and_fdr():
    assert driver_family("proton_density_cm3") == "solar"
    assert driver_family("n_m6_7d") == "catalog"
    assert driver_family("mig_swarm_alpine") == "catalog"
    assert driver_family("mig_along_hikurangi") == "migration"
    assert driver_family("tec_nz_tecu") == "wave2"
    assert driver_family("ulf_pc5_nt") == "wave2"
    assert driver_family("gnss_east_1d") == "wave2"
    assert driver_family("goes_proton_10mev") == "solar"
    assert driver_family("grace_lwe_cm") == "mass"
    assert driver_family("so2_kermadec_kgm2") == "volcano"
    assert driver_family("n_fdsn_japan_m45_7d") == "catalog"
    assert driver_family("sse_hikurangi") == "slowslip"
    assert driver_family("planet_jupiter_lon_deg") == "geometry"
    assert driver_family("x_kp_tidal_strain") == "combo"
    assert driver_family("climate_soi") == "climate"
    assert driver_family("ae_nt") == "solar"
    assert driver_family("neutron_oulu_count") == "cosmic"
    assert driver_family("eop_lod_s") == "earth"
    assert driver_family("lod_delta") == "earth"
    assert driver_family("earth_wobble") == "earth"
    assert driver_family("dust_fireball") == "dust"
    assert driver_family("fireball_energy") == "dust"
    assert driver_family("meteor_shower") == "dust"
    assert driver_family("cme_halo") == "solar"
    assert driver_family("flare_class") == "solar"
    assert driver_family("sep_flux") == "solar"
    assert driver_family("electron_2mev") == "solar"
    assert driver_family("tsi_wm2") == "solar"
    assert driver_family("mgii_index") == "solar"
    assert driver_family("lya_flux") == "solar"
    assert driver_family("mjo_amp") == "climate"
    assert driver_family("olr_anom") == "climate"
    assert driver_family("pdo_index") == "climate"
    assert driver_family("amo_index") == "climate"
    assert junk_knob("cosmic_gle_id")
    assert not junk_knob("neutron_oulu_uncorrected")
    cols = _driver_columns(
        pd.DataFrame(
            {
                "issue_time": pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"),
                "neutron_oulu_count": [1.0, 2.0, 3.0],
                "neutron_oulu_uncorrected": [1.0, 2.0, 3.0],
                "neutron_oulu_pressure_mbar": [1000.0, 999.0, 998.0],
            }
        )
    )
    assert "neutron_oulu_count" in cols
    assert "neutron_oulu_uncorrected" in cols
    assert "neutron_oulu_pressure_mbar" in cols
    q = benjamini_hochberg(np.array([0.001, 0.02, 0.8]))
    assert q[0] < q[2]
    assert q[0] < 0.05


def test_planted_solar_flag_survives_holdout(tmp_path):
    start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    days = [start + timedelta(days=i) for i in range(900)]
    issue = pd.to_datetime(days, utc=True)
    signal = np.zeros(len(days))
    quake_rows = []
    for i in range(20, len(days) - 10, 25):
        signal[i] = 1.0
        when = days[i] + timedelta(days=3)
        quake_rows.append(
            {
                "observed_at": when,
                "available_at": when + timedelta(hours=6),
                "magnitude": 7.2,
                "lat": 0.0,
                "lon": 0.0,
                "event_id": f"ev{i}",
            }
        )
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "issue_time": issue,
            "proton_spike": signal,
            "noise_knob": rng.normal(size=len(days)),
            "n_m6_7d": rng.integers(0, 4, size=len(days)),
        }
    )
    quakes = pd.DataFrame(quake_rows)
    report = run_hunt(root=tmp_path, features=features, quakes=quakes)
    assert report["n_tests"] == 3 * len(LAGS_DAYS) * 6
    assert report["n_knobs"] == 3
    planted = [
        row
        for row in report["confirmed"]
        if row["driver"] == "proton_spike" and row["region"] == "global" and row["lag_days"] == 0
    ]
    assert planted
    assert planted[0]["family"] == "solar"
    noise = [
        row
        for row in report["confirmed"]
        if row["driver"] == "noise_knob"
    ]
    assert not noise
    assert (tmp_path / "data" / "eval" / "hunt.json").exists()
    assert not (tmp_path / "data" / "models").exists() or not any(
        (tmp_path / "data" / "models").glob("*.json")
    )


def test_fresh_hunt_skips_catalog_and_japan(tmp_path):
    start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    days = [start + timedelta(days=i) for i in range(400)]
    issue = pd.to_datetime(days, utc=True)
    features = pd.DataFrame(
        {
            "issue_time": issue,
            "proton_spike": 0.0,
            "n_m6_7d": 1,
        }
    )
    quakes = pd.DataFrame(
        {
            "observed_at": [days[20]],
            "available_at": [days[20] + timedelta(hours=6)],
            "magnitude": [7.2],
            "lat": [0.0],
            "lon": [0.0],
            "event_id": ["ev"],
        }
    )
    report = run_hunt(root=tmp_path, features=features, quakes=quakes, fresh=True)
    assert report["fresh"] is True
    assert "n_m6_7d" not in {t["driver"] for t in report["loudest_train"]}
    assert all(t["region"] != "japan" for t in report.get("confirmed") or [])
    assert (tmp_path / "data" / "eval" / "hunt_fresh.json").exists()
