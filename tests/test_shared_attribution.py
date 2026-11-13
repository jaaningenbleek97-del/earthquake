from types import SimpleNamespace

import numpy as np

from eqpred.eval import mainshock


def test_shared_connections_rank_fitted_weights_not_input_order():
    names = ["cme_halo_7d", "cme_halo_7d__lag1", "tidal_strain__lag2", "kp"]
    model = SimpleNamespace(named_steps={"lr": SimpleNamespace(coef_=np.array([[0.0, 0.01, -2.0, 0.5]]))})
    before = model.named_steps["lr"].coef_.copy()
    rows = mainshock._encoder_connections(model, names)
    assert [r["feature"] for r in rows] == ["tidal_strain__lag2", "kp", "cme_halo_7d__lag1"]
    assert rows[0]["importance"] == -2.0
    assert rows[0]["lag_days"] == 2
    assert rows[0]["driver"] == "tidal_strain"
    assert rows[0]["attribution_kind"] == "shared_encoder_coefficient"
    np.testing.assert_array_equal(before, model.named_steps["lr"].coef_)


def test_shared_connections_name_missingness_flags_separately():
    names = ["kp", "tidal_strain__lag2"]
    model = SimpleNamespace(named_steps={"lr": SimpleNamespace(coef_=np.array([[0.2, 0.1, -3.0, 0.0]]))})
    rows = mainshock._encoder_connections(model, names)
    assert rows[0]["feature"] == "missing__kp"
    assert rows[0]["driver"] == "missing__kp"
    assert rows[0]["importance"] == -3.0
