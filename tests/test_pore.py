from eqpred.ingest.pore import PUBLIC_AT, parse_pressure_csv
from eqpred.stamp import parse_utc


def test_daily_formation_pressure_is_released_on_the_zenodo_date():
    text = "\n".join(
        [
            "date,wellhead_pressure,formation_pressure",
            "01-Jan-2021 00:00:00,100.0,10.0",
            "01-Jan-2021 00:01:00,100.0,12.0",
            "02-Jan-2021 00:00:00,100.0,20.0",
        ]
    )
    daily = parse_pressure_csv(text, "pore_u1518_kpa")
    assert list(daily["pore_u1518_kpa"]) == [11.0, 20.0]
    assert parse_utc(daily["available_at"]).iloc[0] == parse_utc([PUBLIC_AT]).iloc[0]
