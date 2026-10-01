from eqpred.eval.hunt import driver_family
from eqpred.ingest.radon import daily_radon_features, parse_radon_csv


def test_parse_iron_csv_prefers_corrected_and_drops_dupes():
    text = """Date;Rn (Bq/m^3);Rn corrected (Bq/m^3);Rn Instrument;Source
2018-11-07 02:00:00.0;100;200;AERC-1;tool
2018-11-07 02:00:00.0;100;200;AERC-1;website
2018-11-07 06:00:00.0;300;400;AERC-1;tool
"""
    samples = parse_radon_csv(text, "ACATE")
    assert len(samples) == 2
    assert samples.iloc[0]["radon_station"] == "ACATE"
    assert float(samples.iloc[0]["radon_bq_m3"]) == 200.0
    assert float(samples.iloc[1]["radon_bq_m3"]) == 400.0
    daily = daily_radon_features(samples)
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["source"] == "radon.public"
    assert abs(float(row["radon_bq_m3"]) - 300.0) < 1e-9
    assert abs(float(row["radon_range_bq_m3"]) - 200.0) < 1e-9
    assert abs(float(row["radon_dbdt_bq_m3"]) - 200.0) < 1e-9
    assert row["available_at"] > row["observed_at"]
    assert driver_family("radon_bq_m3") == "wave2"
    assert driver_family("radon_range_bq_m3") == "wave2"
