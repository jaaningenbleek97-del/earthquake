import numpy as np
import pandas as pd
import pytest

from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.honest import (
    classify_card_kind,
    derived_model_column,
    era_masks,
    family_mean_z,
    fit_val_split,
    holm_adjust,
    junk_knob,
    purged_train_hold,
    rolling_climate_series,
    signed_mix,
    week_mask,
)


def test_junk_knob_skips_gle_id_only():
    assert junk_knob("cosmic_gle_id")
    assert not junk_knob("neutron_oulu_uncorrected")
    assert not junk_knob("neutron_domc_pressure_mbar")
    assert not junk_knob("neutron_oulu_count")
    assert not junk_knob("neutron_forbush")
    assert not junk_knob("cosmic_gle")


def test_week_mask_keeps_one_morning_per_week():
    mask = np.ones(70, dtype=bool)
    weekly = week_mask(mask, stride=7, min_keep=8)
    assert int(weekly.sum()) == 10
    daily = week_mask(np.ones(20, dtype=bool), stride=7, min_keep=80)
    assert int(daily.sum()) == 20


def test_week_mask_strict_does_not_fall_back_to_overlapping_days():
    strict = week_mask(np.ones(20, dtype=bool), stride=7, min_keep=80, fallback=False)
    assert int(strict.sum()) == 3


def test_era_masks_keep_later_years_off_confirm():
    days = pd.date_range("2010-01-01", periods=4000, freq="D", tz="UTC")
    closed = np.ones(len(days), dtype=bool)
    discover, confirm, replay, start = era_masks(days, closed, "2018-12-31", "2019-01-01")
    assert start.startswith("2015")
    assert not (discover & confirm).any()
    assert not (confirm & replay).any()
    assert replay[days >= "2019-01-01"].all()
    assert not confirm[days >= "2019-01-01"].any()
    assert not discover[days >= "2019-01-01"].any()


def test_era_masks_purge_horizon_around_boundaries():
    days = pd.date_range("2010-01-01", periods=4000, freq="D", tz="UTC")
    closed = np.ones(len(days), dtype=bool)
    discover, confirm, replay, _start = era_masks(days, closed, "2018-12-31", "2019-01-01")
    gap = pd.Timedelta(days=HORIZON_DAYS)
    assert days[discover].max() + gap <= days[confirm].min()
    assert days[confirm].max() + gap <= days[replay].min()
    assert days[confirm].max() + gap <= pd.Timestamp("2019-01-01", tz="UTC")


def test_purged_train_hold_drops_overlapping_labels():
    days = pd.date_range("2018-12-01", periods=60, freq="D", tz="UTC")
    closed = np.ones(len(days), dtype=bool)
    train, hold = purged_train_hold(days, closed, "2018-12-31", "2019-01-01")
    assert days[train].max() + pd.Timedelta(days=HORIZON_DAYS) <= pd.Timestamp("2019-01-01", tz="UTC")
    assert days[hold].min() >= pd.Timestamp("2019-01-01", tz="UTC")
    assert not (train & hold).any()


def test_fit_val_split_uses_late_train_not_one_year_on_long_span():
    days = pd.date_range("2010-01-01", periods=4000, freq="D", tz="UTC")
    train = days <= "2018-12-31"
    fit, val = fit_val_split(days, np.asarray(train))
    assert days[fit].max().year <= 2016
    assert days[val].min().year >= 2016
    assert days[val].max().year == 2018
    assert days[fit].max() + pd.Timedelta(days=HORIZON_DAYS) <= days[val].min()


def test_holm_adjust_is_monotone_and_skips_none():
    out = holm_adjust([0.04, 0.02, None, 0.20])
    assert out[2] is None
    assert out[1] == pytest.approx(0.06)
    assert out[0] == pytest.approx(0.08)
    assert out[3] == pytest.approx(0.20)
    # Holm enforces monotone adjusted p on the sorted raw-p sequence.
    assert out[1] <= out[0] <= out[3]


def test_classify_card_kind_needs_a_same_morning_write():
    assert classify_card_kind("2026-09-19", "2026-09-19") == "prospective"
    assert classify_card_kind("2019-01-01", "2026-09-19") == "retrospective"
    assert classify_card_kind("2019-01-01", None) == "retrospective"


def test_derived_model_and_family_z_keep_every_source():
    assert derived_model_column("analog_p_global")
    assert derived_model_column("earth_state_z")
    assert derived_model_column("graph_degree_hikurangi")
    assert not derived_model_column("proton_density_cm3")
    days = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    feat = pd.DataFrame(
        {
            "issue_time": days,
            "proton_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "proton_b": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            "tide_wellington_mm": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )
    families = family_mean_z(feat, ["proton_a", "proton_b", "tide_wellington_mm"], lambda c: "solar" if c.startswith("proton") else "wave2")
    assert set(families) == {"solar", "wave2"}
    assert families["solar"].shape == (6,)


def test_signed_mix_recovers_a_climate_residual():
    days = pd.date_range("2010-01-01", periods=400, freq="D", tz="UTC")
    climate = np.full(400, 0.2)
    driver = np.zeros(400)
    hits = np.zeros(400)
    driver[50:80] = 1.0
    hits[50:80] = 1.0
    p, scale = signed_mix(driver, climate, hits, np.ones(400, dtype=bool))
    assert p is not None
    assert scale is not None and scale > 0
    mix = float(np.mean((p[50:80] - 1.0) ** 2))
    base = float(np.mean((climate[50:80] - 1.0) ** 2))
    assert mix < base
    roll = rolling_climate_series(hits, days)
    assert roll.shape == (400,)
    assert np.isfinite(roll[-1])
