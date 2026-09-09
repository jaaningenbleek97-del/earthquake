import numpy as np

from eqpred.eval.issued_stack import _reliability_bins, _wave2_global_tec_only


def test_prior_recipe_keeps_global_tec_and_drops_box_wave2():
    wave2 = {
        "regions": {
            "global": {
                "used_on_issued": True,
                "columns": [{"used_on_issued": True, "column": "tec_tecu"}],
            },
            "hikurangi": {
                "used_on_issued": True,
                "columns": [{"used_on_issued": True, "column": "era5_swvl1"}],
            },
        }
    }
    prior = _wave2_global_tec_only(wave2)
    assert prior["regions"]["global"]["columns"][0]["used_on_issued"] is True
    assert prior["regions"]["hikurangi"]["used_on_issued"] is False
    assert prior["regions"]["hikurangi"]["columns"][0]["used_on_issued"] is False
    assert wave2["regions"]["hikurangi"]["used_on_issued"] is True


def test_reliability_bins_split_ranked_p():
    p = np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    bins = _reliability_bins(p, y, n_bins=2)
    assert len(bins) == 2
    assert bins[0]["mean_p"] < bins[1]["mean_p"]
    assert bins[0]["hit_rate"] < bins[1]["hit_rate"]
