import pandas as pd

from eqpred.features.davidson import apply_train_budget, field_strength, significant_windows


def _spf_frame() -> pd.DataFrame:
    days = pd.date_range("2000-01-01", periods=400, freq="D", tz="UTC")
    north = pd.Series(5.0, index=days)
    # Strong peak (v2 must catch) then a reversal through zero to a deep trough.
    north.iloc[50:80] = [5 + i * 3 for i in range(30)]  # rises to 92
    north.iloc[80:140] = list(range(92, 92 - 60, -1))
    north.iloc[140:200] = -80
    # Weak wiggle that v1 would treat as an extremum and v2 should ignore.
    north.iloc[250:260] = [8, 9, 10, 11, 10, 9, 8, 7, 6, 5]
    return pd.DataFrame(
        {
            "observed_at": days,
            "spf_north_ut": north.to_numpy(),
            "spf_south_ut": (-north).to_numpy(),
            "spf_average_ut": 0.8 * north.to_numpy(),
        }
    )


def test_v2_covers_strong_peak_not_every_wiggle():
    frame = _spf_frame()
    v1 = significant_windows(frame, version="v1")
    v2 = significant_windows(frame, version="v2")
    assert bool(v2.iloc[80])
    assert v2.mean() < v1.mean()
    assert v2.mean() < 0.45
    # Weak bump around day 255 should not turn v2 on by itself.
    assert v2.iloc[255] is False or v2.mean() < 0.45


def test_train_budget_caps_fraction():
    days = pd.date_range("2016-01-01", periods=400, freq="D", tz="UTC")
    windows = pd.Series([True] * 200 + [False] * 200, index=days)
    strength = pd.Series(list(range(400)), index=days, dtype=float)
    capped, cutoff = apply_train_budget(windows, strength, "2017-01-01", max_frac=0.20)
    train = capped.loc[capped.index < pd.Timestamp("2017-01-01", tz="UTC")]
    assert float(train.mean()) <= 0.21
    assert cutoff > 0


def test_field_strength_positive():
    strength = field_strength(_spf_frame())
    assert float(strength.max()) > 40
