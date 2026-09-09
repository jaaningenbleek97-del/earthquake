from datetime import datetime, timezone

import pandas as pd

from eqpred.eval.decluster import decluster
from eqpred.models.etas import EtasParams, daily_background_and_cluster
from eqpred.records import prediction


def test_etas_rate_rises_after_large_event():
    t0 = datetime(2010, 1, 1, tzinfo=timezone.utc)
    events = pd.DataFrame(
        {
            "observed_at": pd.to_datetime([t0], utc=True),
            "magnitude": [7.2],
        }
    )
    days = pd.date_range("2010-01-01", periods=20, freq="D", tz="UTC")
    rate = daily_background_and_cluster(events, days, EtasParams(mu=0.05, k=0.2, alpha=1.0, c=0.05, p=1.1))
    assert rate[2] > rate[0]
    assert rate[15] < rate[2]


def test_decluster_drops_nearby_smaller():
    events = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                ["2011-03-11T05:46:00Z", "2011-03-11T06:15:00Z"], utc=True
            ),
            "magnitude": [9.0, 6.4],
            "lon": [142.4, 142.5],
            "lat": [38.3, 38.2],
        }
    )
    kept = decluster(events)
    assert len(kept) == 1
    assert kept.iloc[0]["magnitude"] == 9.0


def test_prediction_record_shape():
    rec = prediction(
        prediction_id="x",
        issued_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2020, 1, 8, tzinfo=timezone.utc),
        horizon_days=7,
        region={"kind": "named", "name": "global"},
        magnitude_min=6.0,
        kind="probability",
        value=0.12,
        model_id="fusion.etas_x_mod.v1",
        drivers=["spf_significant_window"],
    )
    assert rec["value"] == 0.12
    assert rec["issued_at"].endswith("Z")
    assert rec["drivers"][0]["driver"] == "spf_significant_window"


def test_etas_mle_start_does_not_stretch_when_features_begin_in_1990():
    from eqpred.models.fit import etas_mle_start

    long = etas_mle_start(datetime(1990, 1, 1, tzinfo=timezone.utc), "2018-12-31")
    short = etas_mle_start(datetime(2017, 6, 1, tzinfo=timezone.utc), "2018-12-31")
    assert long.date().isoformat() == "2016-01-01"
    assert short.date().isoformat() == "2017-06-01"


def test_etas_mu_clip_allows_global_m45_rate():
    from eqpred.models.etas import EtasParams

    assert EtasParams(mu=23.5).clip().mu == 23.5
    assert EtasParams(mu=90.0).clip().mu == 80.0


def test_fit_etas_moves_cluster_params_off_the_seed():
    from datetime import timedelta

    from eqpred.models.etas import _etas_nll, fit_etas

    t0 = datetime(2016, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(days=i * 8) for i in range(40)]
    mags = [4.6] * 40
    main = t0 + timedelta(days=80)
    times.append(main)
    mags.append(7.4)
    for i in range(80):
        times.append(main + timedelta(hours=2 + i * 6))
        mags.append(4.7 + (i % 5) * 0.15)
    events = pd.DataFrame({"observed_at": pd.to_datetime(times, utc=True), "magnitude": mags})
    end = datetime(2017, 6, 1, tzinfo=timezone.utc)
    fitted = fit_etas(events, t0, end, m0=4.5)
    origin = events["observed_at"].min()
    t = ((events["observed_at"] - origin) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    mag = events["magnitude"].to_numpy(dtype=float)
    duration = (pd.Timestamp(end) - origin) / pd.Timedelta(days=1)
    seed_nll = _etas_nll(t, mag, float(duration), len(events) / float(duration), 0.04, 1.1, 0.02, 1.12, 4.5)
    fit_nll = _etas_nll(t, mag, float(duration), fitted.mu, fitted.k, fitted.alpha, fitted.c, fitted.p, 4.5)
    assert fit_nll <= seed_nll + 1e-6
    moved = abs(fitted.k - 0.04) > 1e-4 or abs(fitted.c - 0.02) > 1e-4 or abs(fitted.p - 1.12) > 1e-3
    assert moved


def test_even_subsample_keeps_full_span_and_reports_pi():
    from eqpred.models.etas import _even_subsample

    t0 = datetime(2016, 1, 1, tzinfo=timezone.utc)
    ev = pd.DataFrame(
        {
            "observed_at": pd.date_range(t0, periods=5000, freq="D", tz="UTC"),
            "magnitude": [4.6] * 5000,
        }
    )
    keep, pi = _even_subsample(ev, 4000)
    assert len(keep) == 4000
    assert abs(pi - 0.8) < 1e-6
    assert keep["observed_at"].min() == ev["observed_at"].min()
    assert keep["observed_at"].max() == ev["observed_at"].max()


def test_quiet_day_mu_uses_full_window_not_a_busy_tail():
    from datetime import timedelta

    from eqpred.models.etas import _quiet_day_mu

    t0 = datetime(2016, 1, 1, tzinfo=timezone.utc)
    end = t0 + timedelta(days=100)
    quiet = pd.date_range(t0, periods=80, freq="D", tz="UTC")
    burst = pd.date_range(t0 + timedelta(days=90), periods=400, freq="36min", tz="UTC")
    ev = pd.DataFrame(
        {
            "observed_at": list(quiet) + list(burst),
            "magnitude": [4.6] * (80 + 400),
        }
    )
    mu = _quiet_day_mu(ev, t0, end)
    assert 0.5 < mu < 2.5


def test_fit_etas_locks_mu_to_quiet_day_rate():
    from datetime import timedelta

    from eqpred.models.etas import fit_etas

    t0 = datetime(2016, 1, 1, tzinfo=timezone.utc)
    end = t0 + timedelta(days=1000)
    times = pd.date_range(t0, periods=5000, freq="4h48min", tz="UTC")
    events = pd.DataFrame({"observed_at": times, "magnitude": [4.6] * 5000})
    fitted = fit_etas(events, t0, end, m0=4.5)
    # Uniform 5/day. Quiet-day lock must keep μ at that count, not a 4000-event tail.
    assert 4.5 < fitted.mu < 5.5


def test_box_etas_weights_nearby_triggers():
    from eqpred.models.etas import EtasParams
    from eqpred.models.predict import _region_etas
    from eqpred.regions import REGIONS

    t0 = datetime(2016, 6, 1, tzinfo=timezone.utc)
    quakes = pd.DataFrame(
        {
            "observed_at": pd.to_datetime([t0, t0, t0], utc=True),
            "magnitude": [6.2, 6.4, 6.1],
            "lat": [-39.7, -39.7, 38.0],
            "lon": [177.25, 174.5, 142.0],
        }
    )
    hist, _params = _region_etas(quakes, EtasParams(), REGIONS["hikurangi"])
    assert "_etas_w" in hist.columns
    inside = hist.loc[(hist["lon"] - 177.25).abs() < 0.1, "_etas_w"]
    near = hist.loc[(hist["lon"] - 174.5).abs() < 0.1, "_etas_w"]
    assert float(inside.iloc[0]) == 1.0
    assert 0.0 < float(near.iloc[0]) < 1.0
    assert 142.0 not in set(hist["lon"])
