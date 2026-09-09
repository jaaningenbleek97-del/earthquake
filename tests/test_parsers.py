from eqpred.ingest.climate import parse_psl_monthly
from eqpred.ingest.cme import parse_cdaw_cme
from eqpred.ingest.donki import parse_donki_flags, parse_donki_flares
from eqpred.ingest.electrons import parse_dpd_electrons, parse_goes_electrons
from eqpred.ingest.eop import parse_eop_c04
from eqpred.ingest.fireball import parse_fireball_api
from eqpred.ingest.gle import parse_gle_grid
from eqpred.ingest.hpo import parse_hpo_text
from eqpred.ingest.lisird import parse_lisird_csv
from eqpred.ingest.mjo import parse_rmm
from eqpred.ingest.neutron import parse_oulu_csv
from eqpred.ingest.omni import parse_omni_daily
from eqpred.ingest.silso import parse_silso
from eqpred.ingest.swpc import parse_xray_flares, parse_kp
from eqpred.ingest.wilcox import parse_wilcox


def test_parse_wilcox_line():
    text = """
Last updated
1976:05:31_21h:07m:13s    89N -126S  108Avg   20nhz filt:  111Nf  -92Sf  101Avgf
1976:06:10_21h:07m:13s    99N -121S  110Avg   20nhz filt:  109Nf  -92Sf  101Avgf
"""
    frame = parse_wilcox(text)
    assert len(frame) == 2
    assert frame.iloc[0]["spf_north_ut"] == 89
    assert frame.iloc[0]["spf_south_ut"] == -126


def test_parse_omni_daily_skips_hourly_and_fills():
    def row(doy: int, hour: int, proton: str, kp: str) -> str:
        fields = ["999.9"] * 55
        fields[0] = "2020"
        fields[1] = str(doy)
        fields[2] = str(hour)
        fields[8] = "6.1"
        fields[12] = "2.0"
        fields[15] = "-1.5"
        fields[16] = "-3.2"
        fields[23] = proton
        fields[24] = "400"
        fields[38] = kp
        fields[40] = "-20"
        fields[41] = "120"
        fields[49] = "7"
        fields[50] = "70.1"
        return " ".join(fields)

    text = "\n".join([row(10, 0, "4.5", "33"), row(10, 1, "9.9", "33"), row(11, 0, "999.9", "99")])
    frame = parse_omni_daily(text)
    assert len(frame) == 2
    assert frame.iloc[0]["proton_density_cm3"] == 4.5
    assert frame.iloc[0]["kp"] == 3.3
    assert frame.iloc[0]["imf_b_nt"] == 6.1
    assert frame.iloc[0]["ae_nt"] == 120
    assert _isna(frame.iloc[1]["proton_density_cm3"])


def test_parse_psl_monthly_skips_fill():
    text = """
 1866  1867
 1866  -1.10   0.20   0.10   0.00  -0.20  -0.30  -0.40  -0.10   0.05   0.15   0.25   0.35
 1867 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99
"""
    frame = parse_psl_monthly(text, "climate_soi")
    assert len(frame) == 12
    assert frame.iloc[0]["climate_soi"] == -1.10


def test_parse_oulu_csv_daily():
    text = """
Timestamp,FractionalDate,UncorrectedCountRate[cts/min],CorrectedCountRate[cts/min],Pressure[mbar]
2020-01-01T00:00:00Z,1.0000000,7113,6751,992.76
2020-01-02T00:00:00Z,2.0000000,7509,0,985.33
2020-01-03T00:00:00Z,3.0000000,8171,6787,974.69
"""
    frame = parse_oulu_csv(text, "neutron_oulu")
    assert len(frame) == 2
    assert frame.iloc[0]["neutron_oulu_count"] == 6751
    assert frame.iloc[0]["neutron_oulu_pressure_mbar"] == 992.76
    assert frame.iloc[1]["observed_at"].startswith("2020-01-03")


def test_parse_gle_grid():
    text = '{"gles":[{"gle":"72","subgle":null,"date":"2017-09-10","gle_type":"0"},{"gle":"x","date":"bad"}]}'
    frame = parse_gle_grid(text)
    assert len(frame) == 1
    assert frame.iloc[0]["cosmic_gle"] == 1.0
    assert frame.iloc[0]["cosmic_gle_id"] == 72
    assert frame.iloc[0]["observed_at"].startswith("2017-09-10")


def test_parse_silso_skips_missing():
    text = "2020;01;01;2020.001;  10.0; 1.0;  12;1\n2020;01;02;2020.004;  -1.0; 1.0;   0;1\n"
    frame = parse_silso(text)
    assert len(frame) == 1
    assert frame.iloc[0]["sunspot_number"] == 10.0


def test_parse_swpc_xray_and_kp():
    flares = [
        {
            "max_time": "2024-01-02T15:00:00Z",
            "max_class": "M1.2",
            "max_xrlong": 1.2e-5,
        },
        {
            "max_time": "2024-01-02T18:00:00Z",
            "max_class": "X1.0",
            "max_xrlong": 1.0e-4,
        },
    ]
    xray = parse_xray_flares(flares)
    assert len(xray) == 1
    assert xray.iloc[0]["xray_peak_wm2"] == 1.0e-4
    kp = parse_kp(
        [
            {"time_tag": "2024-01-02T00:00:00Z", "kp_index": 2.0},
            {"time_tag": "2024-01-02T03:00:00Z", "kp_index": 5.3},
        ]
    )
    assert len(kp) == 1
    assert kp.iloc[0]["swpc_kp"] == 5.3


def test_parse_eop_c04():
    text = """
# YEAR MONTH DAY MJD x y UT1-UTC LOD
1962 01 01 37665 0.012700 0.136900 0.0326338 0.0017230
1962 01 02 37666 0.013100 0.137200 0.0321000 0.0018000
"""
    frame = parse_eop_c04(text)
    assert len(frame) == 2
    assert frame.iloc[0]["eop_pmx_arcsec"] == 0.012700
    assert frame.iloc[0]["eop_lod_s"] == 0.0017230


def test_parse_lisird_csv():
    text = "time,irradiance\n1990-01-01T00:00:00.000Z,1361.1\n1990-01-02T00:00:00.000Z,1361.2\n"
    frame = parse_lisird_csv(text, "solar_tsi_wm2")
    assert len(frame) == 2
    assert frame.iloc[0]["solar_tsi_wm2"] == 1361.1
    days = "time (days since 1610-01-01),irradiance (W/m^2)\n138801.0,1361.1\n"
    frame = parse_lisird_csv(days, "solar_tsi_wm2")
    assert len(frame) == 1
    assert frame.iloc[0]["observed_at"].startswith("1990-01")


def test_parse_fireball_api():
    pack = {
        "fields": ["date", "energy"],
        "data": [["2019-06-22 21:25:48", "33"], ["bad", "1"]],
    }
    frame = parse_fireball_api(pack)
    assert len(frame) == 1
    assert frame.iloc[0]["dust_fireball"] == 1.0
    assert frame.iloc[0]["dust_fireball_kt"] == 33.0


def test_parse_cdaw_cme():
    text = """
Date       Time   CPA  Width  ...
1996/01/11  00:14   108  Halo   173    173   499
1996/01/12  01:00   200   40    80     80    310
"""
    frame = parse_cdaw_cme(text)
    assert len(frame) == 2
    assert float(frame.iloc[0]["cme_halo"]) == 1.0
    assert float(frame.iloc[0]["cme_speed_kms"]) == 173


def test_parse_rmm():
    text = """
Prelim RMM
1974 6 1 1.63 1.20 5 2.04
1974 6 2 1.50 1.10 5 1.86
"""
    frame = parse_rmm(text)
    assert len(frame) == 2
    assert frame.iloc[0]["climate_mjo_phase"] == 5.0


def test_parse_goes_electrons():
    rows = [
        {"time_tag": "2024-01-02T00:00:00Z", "energy": ">=2 MeV", "flux": 120.0},
        {"time_tag": "2024-01-02T12:00:00Z", "energy": ">=2 MeV", "flux": 180.0},
        {"time_tag": "2024-01-02T00:00:00Z", "energy": ">=0.8 MeV", "flux": 400.0},
    ]
    frame = parse_goes_electrons(rows)
    assert len(frame) == 1
    assert frame.iloc[0]["electron_2mev"] == 180.0
    assert frame.iloc[0]["electron_08mev"] == 400.0
    dpd = parse_dpd_electrons("2018 01 01 1e6 1e5 1e3 2.0e8 3.0e7\n")
    assert len(dpd) == 1
    assert "electron_08mev" in dpd.columns


def test_parse_hpo_daily_max():
    text = """
# header
2024 05 10 00.0 00.25 1 1 2.333 9 0
2024 05 10 12.0 12.25 1 1 4.667 27 0
2024 05 11 00.0 00.25 1 1 1.000 4 0
"""
    frame = parse_hpo_text(text)
    assert len(frame) == 2
    assert float(frame.iloc[0]["hp_max"]) == 4.667


def test_parse_donki_flares_and_hss():
    flares = parse_donki_flares(
        [
            {"peakTime": "2024-05-10T06:54Z", "classType": "X3.9"},
            {"peakTime": "2024-05-10T14:11Z", "classType": "M5.9"},
        ]
    )
    assert len(flares) == 1
    assert flares.iloc[0]["flare_x_count"] == 1.0
    assert flares.iloc[0]["flare_m_count"] == 1.0
    hss = parse_donki_flags(
        [{"eventTime": "2024-05-11T00:00Z"}], "hss_flag", ("eventTime", "startTime")
    )
    assert float(hss.iloc[0]["hss_flag"]) == 1.0


def _isna(value) -> bool:
    import pandas as pd

    return bool(pd.isna(value))
