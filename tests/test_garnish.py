from eqpred.models.box_gain import apply_residual_gain, apply_trained_gain
from eqpred.models.leftover import leftover_gains


def test_trained_gain_applies_when_held_off_issued():
    meta = {"used_on_issued": False, "beta": 1.0, "mean": 0.0, "scale": 1.0}
    assert apply_residual_gain(meta, 1.0) == 1.0
    assert apply_trained_gain(meta, 1.0) != 1.0


def test_leftover_gains_include_space_weather():
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
    parts = leftover_gains(row, "global")
    assert "space_weather" in parts
    assert parts["space_weather"] > 0
