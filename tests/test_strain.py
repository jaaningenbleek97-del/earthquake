from eqpred.eval.hunt import driver_family
from eqpred.ingest.strain import daily_strain_features, parse_strain_geocsv


def test_parse_strain_geocsv_eee_nanostrain():
    text = """# dataset: GeoCSV 2.0
DateTime, Gauge 0 Uncorrected, Gauge 0 Corrected, Gauge 1 Uncorrected, Gauge 1 Corrected, Gauge 2 Uncorrected, Gauge 2 Corrected, Gauge 3 Uncorrected, Gauge 3 Corrected, Eee+Enn Uncorrected, Eee+Enn Corrected, Eee-Enn Uncorrected, Eee-Enn Corrected, 2Ene Uncorrected, 2Ene Corrected
2016-01-01T00:00:00Z, -1, -1, -1, -1, -1, -1, -1, -1, -20.0, -20.0, 10.0, 10.0, 1, 1
2016-01-01T12:00:00Z, -1, -1, -1, -1, -1, -1, -1, -1, -22.0, -22.0, 12.0, 12.0, 1, 1
"""
    samples = parse_strain_geocsv(text, "B004")
    assert len(samples) == 2
    # Eee = 0.5 * ((Eee+Enn) + (Eee-Enn)) microstrain → nanostrain
    assert abs(float(samples.iloc[0]["strain_ee_nstrain"]) - (-5.0 * 1000.0)) < 1e-6
    daily = daily_strain_features(samples)
    assert len(daily) == 1
    assert daily.iloc[0]["strain_station"] == "B004"
    assert daily.iloc[0]["available_at"] > daily.iloc[0]["observed_at"]
    assert driver_family("strain_ee_nstrain") == "wave2"
