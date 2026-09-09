from datetime import datetime, timezone

import pandas as pd

from eqpred.eval.promote import evaluate_promote, run_promote
from eqpred.ingest.fdsn import parse_fdsn_geojson
from eqpred.ingest.protons import parse_dpd_text, parse_goes_fluence, parse_goes_protons, parse_spe_text
from eqpred.ingest.sse import events_to_daily, parse_hikurangi_rda_frame, parse_sloweq_csv
from eqpred.ingest.so2 import _year_request


def test_so2_request_uses_date_range_not_ymd():
    req = _year_request(
        2016,
        datetime(2016, 1, 1, tzinfo=timezone.utc),
        datetime(2016, 2, 1, tzinfo=timezone.utc),
    )
    assert req is not None
    assert req["date"] == ["2016-01-01/2016-01-31"]
    assert "year" not in req
    assert "month" not in req
    assert "day" not in req
    assert req["variable"] == ["total_column_sulphur_dioxide"]


def test_parse_fdsn_geojson_uses_reporting_delay():
    payload = {
        "features": [
            {
                "id": "us1000japan",
                "properties": {
                    "time": 1451606400000,
                    "mag": 6.2,
                    "magType": "mw",
                    "place": "near Honshu",
                    "status": "reviewed",
                },
                "geometry": {"coordinates": [141.0, 38.2, 25.0]},
            }
        ]
    }
    frame = parse_fdsn_geojson(payload, "fdsn.japan", "run1")
    assert len(frame) == 1
    assert frame.iloc[0]["event_id"] == "us1000japan"
    assert frame.iloc[0]["magnitude"] == 6.2
    assert frame.iloc[0]["lat"] == 38.2
    assert frame.iloc[0]["source"] == "fdsn.japan"
    assert frame.iloc[0]["available_at"] > frame.iloc[0]["observed_at"]


def test_parse_goes_protons_daily_max():
    rows = [
        {"time_tag": "2016-01-01T00:00:00Z", "energy": ">=10 MeV", "flux": 0.4},
        {"time_tag": "2016-01-01T12:00:00Z", "energy": ">=10 MeV", "flux": 1.2},
        {"time_tag": "2016-01-01T12:00:00Z", "energy": ">=100 MeV", "flux": 0.05},
        {"time_tag": "2016-01-02T00:00:00Z", "energy": ">=10 MeV", "flux": 0.3},
    ]
    frame = parse_goes_protons(rows)
    assert len(frame) == 2
    day1 = frame.loc[frame["observed_at"].str.startswith("2016-01-01")].iloc[0]
    assert float(day1["goes_proton_10mev"]) == 1.2
    assert float(day1["goes_proton_100mev"]) == 0.05


def test_parse_dpd_text_converts_fluence_to_pfu():
    text = """
:Product: Daily Particle Data
# Date >1 MeV >10 MeV >100 MeV
2016 01 01 8.640e+04 1.728e+04 8.640e+03
2016 01 02 -1.0e+05 8.640e+03 8.640e+02
"""
    frame = parse_dpd_text(text)
    assert len(frame) == 2
    day1 = frame.loc[frame["observed_at"].str.startswith("2016-01-01")].iloc[0]
    assert float(day1["goes_proton_1mev"]) == 1.0
    assert float(day1["goes_proton_10mev"]) == 0.2
    day2 = frame.loc[frame["observed_at"].str.startswith("2016-01-02")].iloc[0]
    assert pd.isna(day2.get("goes_proton_1mev"))
    assert float(day2["goes_proton_10mev"]) == 0.1


def test_parse_goes_fluence_divides_by_day():
    frame = parse_goes_fluence(
        [{"time_tag": "2026-09-01T00:00:00Z", "energy": ">=10 MeV", "fluence": 86400.0}]
    )
    assert float(frame.iloc[0]["goes_proton_10mev"]) == 1.0


def test_parse_spe_text_peak_on_start_day():
    text = """
:Product: SPE
  2017
  Sep 10/1645       Sep 11/1145     1490
"""
    frame = parse_spe_text(text)
    assert len(frame) == 1
    assert frame.iloc[0]["observed_at"].startswith("2017-09-10")
    assert float(frame.iloc[0]["goes_proton_10mev"]) == 1490


def test_parse_sloweq_csv_and_daily_flags():
    text = """year,month,day,hour,min,sec,lat,lon,dep,mag,catalog,ref,udate
2016,9,1,0,0,0,-39.2,178.1,12,6.5,Wallace2016-SSE,Wallace2016,2017-01-01
2016,9,1,0,0,0,38.2,142.0,25,6.2,Sekine2010-SSE,Sekine2010,2011-01-01
"""
    events = parse_sloweq_csv(text)
    assert len(events) == 2
    assert set(events["box"]) == {"hikurangi", "japan"}
    daily = events_to_daily(
        events,
        datetime(2016, 8, 31, tzinfo=timezone.utc),
        datetime(2016, 9, 4, tzinfo=timezone.utc),
    )
    on = daily.loc[daily["observed_at"].str.startswith("2016-09-01")].iloc[0]
    assert float(on["sse_hikurangi"]) == 1.0
    assert float(on["sse_japan"]) == 1.0
    off = daily.loc[daily["observed_at"].str.startswith("2016-08-31")].iloc[0]
    assert float(off["sse_hikurangi"]) == 0.0


def test_sse_duration_fills_the_window():
    events = pd.DataFrame(
        [
            {
                "start": datetime(2018, 1, 1, tzinfo=timezone.utc),
                "end": datetime(2018, 1, 3, tzinfo=timezone.utc),
                "box": "chile",
                "mag": 6.8,
            }
        ]
    )
    daily = events_to_daily(
        events,
        datetime(2018, 1, 1, tzinfo=timezone.utc),
        datetime(2018, 1, 5, tzinfo=timezone.utc),
    )
    assert list(daily["sse_chile"]) == [1.0, 1.0, 1.0, 0.0]
    assert float(daily.loc[0, "sse_chile_mag"]) == 6.8


def test_hikurangi_rda_uses_unix_start_end():
    frame = pd.DataFrame(
        {
            "date.start": [1_140_220_800.0],
            "date.end": [1_141_862_400.0],
            "nsta": [3],
        }
    )
    events = parse_hikurangi_rda_frame(frame)
    assert len(events) == 1
    assert events.iloc[0]["box"] == "hikurangi"
    assert events.iloc[0]["start"].year == 2006


def test_promote_gate_holds_without_skill(tmp_path):
    brain = {
        "walk_forward": {"mean_lift": -0.004},
        "locked_brier": 0.15,
        "locked_etas_brier": 0.10,
        "locked_lift": -0.01,
    }
    gate = evaluate_promote(brain)
    assert gate["pass"] is False
    assert gate["writes_issued_p"] is False
    written = run_promote(root=tmp_path, brain=brain)
    assert written["pass"] is False
    assert (tmp_path / "data" / "eval" / "promote.json").exists()
    assert not (tmp_path / "data" / "models").exists() or not any(
        (tmp_path / "data" / "models").glob("*.json")
    )


def test_promote_gate_passes_only_when_both_bars_clear():
    ok = evaluate_promote(
        {
            "walk_forward": {"mean_lift": 0.02},
            "locked_brier": 0.08,
            "locked_etas_brier": 0.10,
        }
    )
    assert ok["pass"] is True
    assert ok["writes_issued_p"] is False
    half = evaluate_promote(
        {
            "walk_forward": {"mean_lift": 0.02},
            "locked_brier": 0.12,
            "locked_etas_brier": 0.10,
        }
    )
    assert half["pass"] is False
