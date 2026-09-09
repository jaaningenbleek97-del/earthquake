import pandas as pd

from eqpred.eval.strategies import BOOK, Strategy, _entries, _window_hits


def test_kermadec_strategy_is_m6():
    rule = next(r for r in BOOK if r.region_id == "kermadec")
    assert rule.strategy_id == "spf_kermadec_m6"
    assert rule.magnitude_min == 6.0


def test_window_hits_only_inside_horizon_and_box():
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                ["2019-01-03T00:00:00Z", "2019-01-20T00:00:00Z"], utc=True
            ),
            "magnitude": [5.2, 5.4],
            "lon": [176.0, 170.0],
            "lat": [-39.0, -43.0],
        }
    )
    region = {
        "kind": "bbox",
        "min_lon": 174.6,
        "max_lon": 179.9,
        "min_lat": -42.4,
        "max_lat": -37.0,
    }
    starts = pd.DatetimeIndex(["2019-01-01", "2019-01-10"], tz="UTC")
    hits = _window_hits(quakes, region, 5.0, starts, 7)
    assert bool(hits[0])
    assert not bool(hits[1])


def test_entries_need_every_flag():
    frame = pd.DataFrame(
        {
            "proton_spike": [1.0, 1.0, 0.0],
            "dst_storm": [1.0, 0.0, 1.0],
            "lambda_etas": [1.0, 1.0, 1.0],
        }
    )
    rule = Strategy("x", "x", "hikurangi", 5.0, 7, flags=("proton_spike", "dst_storm"))
    on = _entries(frame, rule, quiet_cut=9.0)
    assert list(on) == [True, False, False]


def test_paper_quiet_gates_are_unlocked_and_score_fresh():
    ids = {r.strategy_id: r for r in BOOK}
    assert ids["quiet_unconv_m7"].locked is True
    assert ids["quiet_unconv_m7"].score_fresh is False
    for sid in ("quiet_etas_fresh_m7", "quiet_wide_fresh_m7", "quiet_start_fresh_m7"):
        assert ids[sid].locked is False
        assert ids[sid].score_fresh is True
    assert ids["quiet_wide_fresh_m7"].quiet_quantile == 0.50
    assert ids["quiet_start_fresh_m7"].require_prior_quiet is True


def test_quiet_entry_uses_threshold():
    frame = pd.DataFrame(
        {
            "spf_significant_window": [1.0, 1.0],
            "lambda_etas": [4.0, 20.0],
        }
    )
    rule = Strategy("q", "q", "global", 7.0, 7, flags=("spf_significant_window",), require_quiet=True)
    on = _entries(frame, rule, quiet_cut=9.0)
    assert list(on) == [True, False]
