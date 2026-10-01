from eqpred.ingest.tide import STATIONS, _parse_daily_csv, _parse_erddap


def test_fast_delivery_ids_are_not_honolulu():
    # UHSLC 57 is Honolulu; Auckland is not on the fast-delivery feed.
    assert 57 not in STATIONS.values()
    assert STATIONS["wellington"] == 71
    assert STATIONS["valparaiso"] == 81
    assert STATIONS["kushiro"] == 350


def test_parse_uhslc_daily_csv_skips_missing():
    text = """year month day sl
2020 1 1 1450
2020 1 2 9999
2020,1,3,1462
not a row
"""
    frame = _parse_daily_csv(text)
    assert len(frame) == 2
    assert float(frame["tide_mm"].iloc[0]) == 1450


def test_parse_erddap_skips_unit_row():
    text = """time,sea_level,uhslc_id,station_name
UTC,mm,,
2020-01-01T00:00:00Z,1501.0,57,Auckland
2020-01-02T00:00:00Z,NaN,57,Auckland
"""
    frame = _parse_erddap(text)
    assert len(frame) == 1
    assert float(frame["tide_mm"].iloc[0]) == 1501.0
