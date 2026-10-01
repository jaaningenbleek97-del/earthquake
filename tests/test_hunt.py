from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from eqpred.eval.honest import junk_knob
from eqpred.eval.hunt import (
    FRESH_LAGS,
    FRESH_TARGETS,
    LAGS_DAYS,
    SEQUENCE_LAGS,
    _driver_columns,
    benjamini_hochberg,
    driver_family,
    hunt_family,
    run_hunt,
    sequence_cache_matches,
)


def test_family_and_fdr():
    assert driver_family("proton_density_cm3") == "solar"
    assert driver_family("n_m6_7d") == "catalog"
    assert driver_family("n_fresh_m6_7d") == "fresh_catalog"
    assert driver_family("n_fresh_m45_7d") == "fresh_catalog"
    assert driver_family("days_since_fresh_m7") == "fresh_catalog"
    assert driver_family("b_fresh_90d") == "fresh_catalog"
    assert driver_family("rate_roll_5y_kermadec") == "fresh_catalog"
    assert driver_family("tide_wellington_mm") == "wave2"
    assert driver_family("insar_los_mm") == "wave2"
    assert driver_family("gnss_strain_nz_dilatation_30d") == "wave2"
    assert driver_family("mig_swarm_alpine") == "catalog"
    assert driver_family("seq_n_hikurangi") == "catalog"
    assert driver_family("seq_b_pos_delta_alpine") == "catalog"
    assert hunt_family("seq_n_hikurangi") == "sequence"
    assert hunt_family("n_m6_7d") == "catalog"
    assert driver_family("mig_along_hikurangi") == "migration"
    assert driver_family("tec_nz_tecu") == "wave2"
    assert driver_family("ulf_pc5_nt") == "wave2"
    assert driver_family("gnss_east_1d") == "wave2"
    assert driver_family("goes_proton_10mev") == "solar"
    assert driver_family("grace_lwe_cm") == "mass"
    assert driver_family("so2_kermadec_kgm2") == "volcano"
    assert driver_family("n_fdsn_japan_m45_7d") == "catalog"
    assert driver_family("sse_hikurangi") == "slowslip"
    assert driver_family("tremor_hikurangi") == "slowslip"
    assert driver_family("well_d1_m") == "wave2"
    assert driver_family("scint_s4") == "wave2"
    assert driver_family("dvv_hikurangi") == "wave2"
    assert driver_family("wspr_snr_med") == "wave2"
    assert driver_family("wiki_eq_views") == "foil"
    assert driver_family("vlf_amp") == "foil"
    assert driver_family("tir_ts") == "foil"
    assert driver_family("roti_nz") == "wave2"
    assert driver_family("gnssa_hikurangi") == "wave2"
    assert driver_family("hydro_gwet") == "wave2"
    assert driver_family("smap_sm") == "wave2"
    assert driver_family("sdarn_unwin") == "radio"
    assert driver_family("haarp_on") == "solar"
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
                "wiki_eq_views": [10.0, 11.0, 12.0],
            }
        )
    )
    assert "neutron_oulu_count" in cols
    assert "wiki_eq_views" not in cols
    assert "neutron_oulu_uncorrected" in cols
    assert "neutron_oulu_pressure_mbar" in cols
    q = benjamini_hochberg(np.array([0.001, 0.02, 0.8]))
    assert q[0] < q[2]
    assert q[0] < 0.05


def test_planted_solar_flag_survives_holdout(tmp_path):
    start = datetime(2014, 1, 1, tzinfo=timezone.utc)
    days = [start + timedelta(days=i) for i in range(3300)]
    issue = pd.to_datetime(days, utc=True)
    signal = np.zeros(len(days))
    quake_rows = []
    for i in range(50, len(days) - 40, 80):
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
    assert report["fresh"] is True
    assert report["n_tests"] == 2 * len(FRESH_LAGS) * len(FRESH_TARGETS)
    assert report["n_knobs"] == 2
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
    assert (tmp_path / "data" / "eval" / "hunt_fresh.json").exists()
    assert not (tmp_path / "data" / "models").exists() or not any(
        (tmp_path / "data" / "models").glob("*.json")
    )
    honest = report["honest"]
    assert honest["looks_at_all_sources"] is True
    assert honest["n_knobs"] == 2
    assert honest["writes_issued_p"] is False
    assert honest["n_families"] >= 1
    planted_honest = [
        row
        for row in honest["confirmed"]
        if row["driver"] == "proton_spike" and row["region"] == "global" and row["lag_days"] == 0
    ]
    assert planted_honest
    assert planted_honest[0]["question"] in {"fresh", "quiet_start"}
    assert "noise_knob" not in {row["driver"] for row in honest["confirmed"]}
    assert any(row["driver"] == "family:solar" for row in honest["families"] + honest["confirmed"])


def test_etas_residual_hunt_keeps_raw_catalog(tmp_path):
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
    report = run_hunt(root=tmp_path, features=features, quakes=quakes, fresh=False)
    assert report["fresh"] is False
    assert report["n_tests"] == 2 * len(LAGS_DAYS) * 6
    assert (tmp_path / "data" / "eval" / "hunt.json").exists()


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


def test_sequence_hunt_plants_and_skips_solar(tmp_path):
    start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    days = [start + timedelta(days=i) for i in range(2200)]
    issue = pd.to_datetime(days, utc=True)
    signal = np.zeros(len(days))
    quake_rows = []
    for i in range(50, len(days) - 40, 80):
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
            "seq_active_any": signal,
            "proton_spike": signal,
            "noise_knob": rng.normal(size=len(days)),
            "n_m6_7d": rng.integers(0, 4, size=len(days)),
        }
    )
    quakes = pd.DataFrame(quake_rows)
    report = run_hunt(root=tmp_path, features=features, quakes=quakes, sequence=True)
    assert report["sequence"] is True
    assert report["fresh"] is True
    assert report["writes_issued_p"] is False
    assert report["n_knobs"] == 2
    assert report["n_tests"] == 2 * len(SEQUENCE_LAGS) * len(FRESH_TARGETS)
    planted = [
        row
        for row in report["confirmed"]
        if row["driver"] == "seq_active_any" and row["region"] == "global" and row["lag_days"] == 0
    ]
    assert planted
    assert planted[0]["family"] == "sequence"
    assert planted[0]["aftershock_shaped"] is False
    assert "proton_spike" not in {t["driver"] for t in report["loudest_train"]}
    assert "noise_knob" not in {t["driver"] for t in report["loudest_train"]}
    assert (tmp_path / "data" / "eval" / "hunt_sequence.json").exists()
    assert not (tmp_path / "data" / "eval" / "hunt_fresh.json").exists()
    assert not (tmp_path / "data" / "models").exists() or not any(
        (tmp_path / "data" / "models").glob("*.json")
    )


def test_sequence_cache_matches_requires_same_calendar():
    days = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    feat = pd.DataFrame({"issue_time": days})
    ok = pd.DataFrame({"issue_time": days, "seq_alpine_n": [1.0] * 6})
    short = pd.DataFrame(
        {
            "issue_time": pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"),
            "seq_alpine_n": [1.0, 2.0, 3.0],
        }
    )
    shifted = pd.DataFrame(
        {
            "issue_time": pd.date_range("2019-01-01", periods=6, freq="D", tz="UTC"),
            "seq_alpine_n": [1.0] * 6,
        }
    )
    no_seq = pd.DataFrame({"issue_time": days, "n_m4_alpine": [1.0] * 6})
    assert sequence_cache_matches(ok, feat) is True
    assert sequence_cache_matches(short, feat) is False
    assert sequence_cache_matches(shifted, feat) is False
    assert sequence_cache_matches(no_seq, feat) is False
    assert sequence_cache_matches(pd.DataFrame(), feat) is False
