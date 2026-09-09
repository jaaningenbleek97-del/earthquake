import numpy as np
import pandas as pd

from eqpred.eval.honest import era_masks, fit_val_split, junk_knob, week_mask


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


def test_fit_val_split_uses_late_train_not_one_year_on_long_span():
    days = pd.date_range("2010-01-01", periods=4000, freq="D", tz="UTC")
    train = days <= "2018-12-31"
    fit, val = fit_val_split(days, np.asarray(train))
    assert days[fit].max().year <= 2016
    assert days[val].min().year >= 2016
    assert days[val].max().year == 2018
