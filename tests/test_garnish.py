from eqpred.models.box_gain import apply_residual_gain, apply_trained_gain
from eqpred.models.garnish import WATCH_FLAGS, garnish_gains, watch_list


def test_trained_gain_applies_when_held_off_issued():
    meta = {"used_on_issued": False, "beta": 1.0, "mean": 0.0, "scale": 1.0}
    assert apply_residual_gain(meta, 1.0) == 1.0
    assert apply_trained_gain(meta, 1.0) != 1.0


def test_garnish_gains_include_space_weather():
    row = {
        "spf_significant_window": 1,
        "proton_spike": 0,
        "proton_density_lag1d": 8.0,
        "kp_high": 0,
        "dst_storm": 0,
        "tidal_strain": 1.6,
        "near_new_or_full": 0,
        "solar_cycle_phase": 0.4,
        "b_90d": 1.2,
        "log_energy_7d": 8.0,
        "log_energy_30d": 10.0,
        "energy_accel": 1.0,
        "tec_tecu": 7.0,
    }
    parts = garnish_gains(row, "global")
    assert "space_weather" in parts
    assert parts["space_weather"] > 0


def test_watch_list_covers_davidson_burns_and_high_value_flags():
    ids = [key for key, _label in WATCH_FLAGS]
    for required in (
        "spf_significant_window",
        "proton_spike",
        "hss_on",
        "flare_x_on",
        "spe_on",
        "solar_descending",
        "solar_eclipse_window",
        "volcano_unrest",
        "dart_drop",
        "sse_on",
        "tremor_on",
        "well_drop",
        "scint_high",
        "haarp_on",
        "dvv_drop",
        "wspr_quiet",
        "wiki_hot",
        "hydro_wet",
        "roti_high",
        "cal_payday",
        "ulf_high",
        "radon_high",
        "grace_wet",
        "so2_high",
        "strain_pulse",
        "era5_wet",
        "olr_anomaly",
        "gnss_dilatation",
        "sunspot_hemisphere_skew",
        "earth_state_unusual",
        "graph_transfer_on",
    ):
        assert required in ids
    flags = watch_list({"spf_significant_window": 1, "hss_on": 0, "volcano_unrest": 1})
    on = {item["id"]: item["on"] for item in flags}
    assert on["spf_significant_window"] is True
    assert on["hss_on"] is False
    assert on["volcano_unrest"] is True
