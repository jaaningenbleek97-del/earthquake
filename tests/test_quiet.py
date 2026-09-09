import numpy as np
import pandas as pd

from eqpred.models.quiet import (
    is_quiet,
    issued_train_mask,
    quiet_threshold,
    ready_train_mask,
    unconventional_drivers,
)


def test_ready_train_mask_drops_spinup_year():
    days = pd.date_range("2016-01-01", periods=800, freq="D", tz="UTC")
    mask = ready_train_mask(pd.Series(days), "2018-12-31", spinup_days=365)
    assert not mask[0]
    assert not mask[364]
    assert mask[365]
    assert mask[729]


def test_hold_issued_garnish_keeps_train_gate_and_clears_issued():
    from eqpred.models.quiet import hold_issued_garnish

    held = hold_issued_garnish(
        {
            "used_on_issued": True,
            "regions": {
                "hikurangi": {
                    "used_on_issued": True,
                    "columns": [{"used_on_issued": True, "column": "tec_tecu"}],
                }
            },
        }
    )
    assert held["used_on_issued"] is False
    assert held["train_gate_pass"] is True
    assert held["regions"]["hikurangi"]["used_on_issued"] is False
    assert held["regions"]["hikurangi"]["columns"][0]["used_on_issued"] is False
    assert held["regions"]["hikurangi"]["columns"][0]["train_gate_pass"] is True


def test_issued_fusion_p_is_scale_only():
    import pandas as pd

    from eqpred.models.gr import rate_to_p
    from eqpred.models.predict import issued_fusion_p

    row = pd.Series({"tec_tecu": 80.0, "coulomb_hikurangi": 2.0})
    cal = {"scale": 2.0, "scales": {"global": 2.0, "hikurangi": 3.0}}
    wave2 = {
        "regions": {
            "global": {
                "used_on_issued": True,
                "columns": [
                    {
                        "used_on_issued": True,
                        "column": "tec_tecu",
                        "beta": 2.0,
                        "mean": 0.0,
                        "scale": 1.0,
                    }
                ],
            }
        }
    }
    p, etas = issued_fusion_p(
        "global",
        row,
        1.0,
        cal=cal,
        nowcast_on=True,
        nowcast_scale=2.0,
        g_now_global=2.0,
        coulomb={},
        cusp={},
        wave2=wave2,
        migration={},
        mag=7.0,
        m0=4.5,
        b=1.0,
        horizon_days=7,
    )
    assert p == rate_to_p(2.0, 7.0, 4.5, 1.0, 7)
    assert etas == rate_to_p(1.0, 7.0, 4.5, 1.0, 7)


def test_issued_train_mask_starts_in_2016():
    days = pd.date_range("1990-01-01", "2019-01-05", freq="D", tz="UTC")
    mask = issued_train_mask(pd.Series(days), "2018-12-31", spinup_days=365)
    frame = pd.DataFrame({"day": days, "ok": mask})
    assert not bool(frame.loc[frame["day"] == "1995-06-01", "ok"].iloc[0])
    assert bool(frame.loc[frame["day"] == "2016-01-01", "ok"].iloc[0])
    assert bool(frame.loc[frame["day"] == "2018-12-30", "ok"].iloc[0])
    assert not bool(frame.loc[frame["day"] == "2018-12-31", "ok"].iloc[0])


def test_quiet_threshold_is_lower_tail():
    lam = np.linspace(1.0, 20.0, 100)
    mask = np.ones(100, dtype=bool)
    cut = quiet_threshold(lam, mask, quantile=0.35)
    assert 6 < cut < 10
    assert is_quiet(cut - 0.1, cut)
    assert not is_quiet(cut + 0.1, cut)


def test_unconventional_drivers_reads_flags():
    import pandas as pd

    row = pd.Series(
        {
            "spf_significant_window": 1.0,
            "proton_spike": 0.0,
            "kp_high": 1.0,
            "dst_storm": 0.0,
        }
    )
    assert unconventional_drivers(row) == ["spf_significant_window", "kp_high"]
