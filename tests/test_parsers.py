from eqpred.ingest.climate import parse_psl_monthly
from eqpred.ingest.cme import parse_cdaw_cme
from eqpred.ingest.donki import parse_donki_flags, parse_donki_flares
from eqpred.ingest.electrons import parse_dpd_electrons, parse_goes_electrons
from eqpred.ingest.eop import parse_eop_c04, parse_eop_finals
from eqpred.ingest.fireball import parse_fireball_api
from eqpred.ingest.gle import parse_gle_grid
from eqpred.ingest.hpo import parse_hpo_text
from eqpred.ingest.lisird import parse_bremen_mgii, parse_lisird_csv
from eqpred.ingest.mjo import parse_omi, parse_rmm, rmm_phase
from eqpred.ingest.neutron import parse_oulu_csv
from eqpred.ingest.omni import parse_omni_daily
from eqpred.ingest.silso import parse_silso, parse_silso_hem
from eqpred.ingest.dart import daily_dart_features, parse_ndbc_dart, parse_tilde_height
from eqpred.ingest.geometry import pull_geometry
from eqpred.ingest.volcano import parse_volcano_events
from eqpred.ingest.tremor import events_to_daily, parse_eqdb_events
from eqpred.ingest.wells import daily_well_features, parse_hilltop_xml, parse_nwis_dv
from eqpred.ingest.haarp import EXTRA_WINDOWS, parse_campaign_windows, windows_to_daily
from eqpred.ingest.cosmic2 import (
    MISSION_START,
    _frontier_first,
    on_day_grid,
    parse_scn_tar,
    profiles_to_daily,
    thin_pending,
    worker_count,
)
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


def test_parse_silso_hem_skips_missing():
    text = (
        "2020;01;01;2020.001; 20.0; 12.0; 8.0; 1.0; 1.0; 1.0; 10;-1;-1;1\n"
        "2020;00;02;2020.004; 0.0; -1; -1; 1.0; 1.0; 1.0; 0;-1;-1;1\n"
    )
    frame = parse_silso_hem(text)
    assert len(frame) == 1
    assert frame.iloc[0]["sunspot_north"] == 12.0
    assert frame.iloc[0]["sunspot_south"] == 8.0


def test_parse_ndbc_dart_and_daily():
    text = """
#YY  MM DD hh mm ss T   HEIGHT
#yr  mo dy hr mn  s -      m
2020 01 01 00 15 00 1 5432.183
2020 01 01 00 30 00 1 5432.201
2020 01 02 00 15 00 1 5432.050
"""
    frame = parse_ndbc_dart(text, "55012")
    assert len(frame) == 3
    daily = daily_dart_features(frame)
    assert len(daily) == 2
    assert daily.iloc[0]["dart_station"] == "55012"
    assert daily.iloc[0]["dart_height_m"] > 5400


def test_parse_tilde_height_keeps_metres():
    payload = [{"data": [{"ts": "2021-03-05T00:00:00Z", "val": 5431.2}]}]
    rows = parse_tilde_height(payload)
    assert len(rows) == 1
    assert abs(rows[0][1] - 5431.2) < 1e-6


def test_geometry_splits_solar_and_lunar_eclipse_windows():
    from datetime import datetime, timezone

    frame = pull_geometry(
        datetime(2017, 8, 20, tzinfo=timezone.utc),
        datetime(2017, 8, 23, tzinfo=timezone.utc),
        "test",
    )
    assert "solar_eclipse_window" in frame.columns
    assert "lunar_eclipse_geom" in frame.columns
    assert set(frame["solar_eclipse_window"].unique()).issubset({0.0, 1.0})


def test_parse_volcano_events_marks_nz():
    rows = [
        {"year": 2012, "month": 8, "day": 6, "vei": 5, "latitude": -37.52, "longitude": 177.18},
        {
            "properties": {
                "StartDateYear": 1991,
                "StartDateMonth": 6,
                "StartDateDay": 15,
                "ExplosivityIndexMax": 6,
            },
            "geometry": {"coordinates": [120.35, 15.13]},
        },
        {"year": 1700, "month": 1, "day": 1, "vei": 4, "latitude": -38.0, "longitude": 176.0},
    ]
    frame = parse_volcano_events(rows)
    assert len(frame) == 2
    assert float(frame.loc[frame["volcano_vei"] == 5.0, "volcano_nz"].iloc[0]) == 1.0
    assert float(frame.loc[frame["volcano_vei"] == 6.0, "volcano_nz"].iloc[0]) == 0.0


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


def test_parse_eop_finals():
    text = (
        "26 1 5 61045.00 I  0.105577 0.000031  0.335516 0.000019  I 0.0002471 0.0000076  0.000491 0.0000076\n"
        "26 930 61313.00 P  0.179427 0.003369  0.324278 0.003553  P-0.0201464 0.0021877\n"
        "271031 61709.00\n"
    )
    frame = parse_eop_finals(text)
    assert list(frame["observed_at"].str[:10]) == ["2026-01-05", "2026-09-30"]
    assert abs(frame.iloc[0]["eop_pmx_arcsec"] - 0.105577) < 1e-6
    assert abs(frame.iloc[1]["eop_pmy_arcsec"] - 0.324278) < 1e-6


def test_parse_lisird_csv():
    text = "time,irradiance\n1990-01-01T00:00:00.000Z,1361.1\n1990-01-02T00:00:00.000Z,1361.2\n"
    frame = parse_lisird_csv(text, "solar_tsi_wm2")
    assert len(frame) == 2
    assert frame.iloc[0]["solar_tsi_wm2"] == 1361.1
    days = "time (days since 1610-01-01),irradiance (W/m^2)\n138801.0,1361.1\n"
    frame = parse_lisird_csv(days, "solar_tsi_wm2")
    assert len(frame) == 1
    assert frame.iloc[0]["observed_at"].startswith("1990-01")
    julian = "time,irradiance\n2461288.0,1362.2458\n"
    frame = parse_lisird_csv(julian, "solar_tsi_wm2")
    assert len(frame) == 1
    assert frame.iloc[0]["observed_at"].startswith("2026-09")
    lya = "time,irradiance\n29109.0,0.006776\n"
    frame = parse_lisird_csv(lya, "solar_lya")
    assert len(frame) == 1
    assert frame.iloc[0]["observed_at"].startswith("2026-09")


def test_parse_bremen_mgii():
    text = "  2026.70000 09 13 0.15951  4.85e-04  19\n"
    frame = parse_bremen_mgii(text)
    assert len(frame) == 1
    assert frame.iloc[0]["observed_at"].startswith("2026-09-13")
    assert abs(frame.iloc[0]["solar_mgii"] - 0.15951) < 1e-8


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
    assert rmm_phase(1.0, 0.0) == 5.0


def test_parse_omi():
    text = "2026 6 24 -0.24596 -0.31677 0.40105\n"
    frame = parse_omi(text)
    assert len(frame) == 1
    assert frame.iloc[0]["observed_at"].startswith("2026-06-24")
    assert abs(frame.iloc[0]["climate_mjo_amp"] - 0.40105) < 1e-6


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


def test_parse_eqdb_tremor_and_daily():
    from datetime import datetime, timezone

    payload = [
        {
            "date": "2010-01-20",
            "time": "12:46:46",
            "lat": "-39.19120",
            "lon": "176.03999",
            "dep": "17.705",
            "duration": "28.00",
            "catalog": "Romanet2019a-Tremor",
        },
        {
            "date": "2010-01-20",
            "time": "13:00:00",
            "lat": "-43.53000",
            "lon": "169.98000",
            "catalog": "Baratin2018-LFE",
        },
    ]
    events = parse_eqdb_events(payload)
    assert len(events) == 2
    assert set(events["kind"]) == {"tremor", "lfe"}
    assert "hikurangi" in set(events["box"])
    assert "alpine" in set(events["box"])
    daily = events_to_daily(
        events,
        datetime(2010, 1, 20, tzinfo=timezone.utc),
        datetime(2010, 1, 22, tzinfo=timezone.utc),
    )
    assert len(daily) == 2
    assert float(daily.iloc[0]["tremor_on"]) == 1.0
    assert float(daily.iloc[0]["tremor_hikurangi"]) == 1.0
    assert float(daily.iloc[0]["lfe_alpine"]) == 1.0
    assert float(daily.iloc[1]["tremor_on"]) == 0.0


def test_parse_hilltop_and_nwis_wells():
    xml = """
    <Hilltop>
      <Measurement>
        <Data>
          <E><T>2020-01-01T00:00:00</T><I1>12.4</I1></E>
          <E><T>2020-01-02T00:00:00</T><I1>12.1</I1></E>
        </Data>
      </Measurement>
    </Hilltop>
    """
    frame = parse_hilltop_xml(xml, "Gear Island")
    assert len(frame) == 2
    assert frame.iloc[0]["well_station"] == "Gear Island"
    daily = daily_well_features(frame)
    assert len(daily) == 2
    assert abs(daily.iloc[1]["well_d1_m"] + 0.3) < 1e-9
    nwis = parse_nwis_dv(
        {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {"siteCode": [{"value": "343204118480601"}]},
                        "variable": {
                            "variableCode": [{"value": "72019"}],
                            "unit": {"unitCode": "ft"},
                        },
                        "values": [
                            {
                                "value": [
                                    {"dateTime": "2020-01-01T00:00:00", "value": "10.0"}
                                ]
                            }
                        ],
                    }
                ]
            }
        }
    )
    assert len(nwis) == 1
    assert nwis.iloc[0]["well_level_m"] == -10.0 * 0.3048


def test_parse_haarp_campaign_windows():
    from datetime import datetime, timezone

    text = (
        "PARS 2024 (August 6-21, 2024) May 8-10, 2024 Campaign "
        "February 27-March 2, 2024 Campaign September 21-25, 2017 Campaign "
        "October 2022 Campaign December 27, 2022 Campaign"
    )
    windows = parse_campaign_windows(text)
    labels = " ".join(label for _a, _b, label in windows)
    assert "August 6-21, 2024" in labels
    assert "February 27-March 2, 2024" in labels
    daily = windows_to_daily(
        windows,
        datetime(2024, 8, 5, tzinfo=timezone.utc),
        datetime(2024, 8, 8, tzinfo=timezone.utc),
    )
    assert float(daily.iloc[0]["haarp_on"]) == 0.0
    assert float(daily.iloc[1]["haarp_on"]) == 1.0
    extra_labels = " ".join(label for _a, _b, label in EXTRA_WINDOWS)
    assert "January 27-31, 2025" in extra_labels
    assert "November 17-22, 2025" in extra_labels


def test_cosmic2_profiles_to_daily():
    from datetime import datetime, timezone

    import pandas as pd

    profiles = pd.DataFrame(
        {
            "s4": [0.08, 0.40, 0.12],
            "lat": [-39.2, 10.0, -43.5],
            "lon": [176.0, 20.0, 170.0],
        }
    )
    daily = profiles_to_daily(profiles, datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert float(daily.iloc[0]["scint_n"]) == 3.0
    assert float(daily.iloc[0]["scint_n_nz"]) == 2.0
    assert daily.iloc[0]["scint_s4_max"] == 0.40
    assert daily.iloc[0]["scint_hikurangi"] == 0.08


def test_cosmic2_worker_count_is_capped():
    assert 1 <= worker_count() <= 6


def test_cosmic2_day_stride_keeps_every_fourth_morning():
    from datetime import timedelta

    start = MISSION_START
    assert on_day_grid(start, 4)
    assert not on_day_grid(start + timedelta(days=1), 4)
    assert not on_day_grid(start + timedelta(days=2), 4)
    assert not on_day_grid(start + timedelta(days=3), 4)
    assert on_day_grid(start + timedelta(days=4), 4)
    days = [start + timedelta(days=i) for i in range(12)]
    kept = thin_pending(days, 4)
    assert [d.date() for d in kept] == [start.date(), (start + timedelta(days=4)).date(), (start + timedelta(days=8)).date()]
    assert thin_pending(days, 1) == days


def test_cosmic2_profile_stride_skips_unread_members():
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for i in range(4):
            payload = b"not-a-netcdf"
            info = tarfile.TarInfo(name=f"p{i}.nc")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    parsed = parse_scn_tar(buf.getvalue(), stride=4)
    assert parsed.empty


def test_cosmic2_frontier_first_defers_old_holes():
    from datetime import datetime, timedelta, timezone

    start = datetime(2022, 3, 25, tzinfo=timezone.utc)
    skip = {datetime(2022, 11, 1).date(), datetime(2022, 11, 2).date()}
    cached = []
    cursor = start
    while cursor <= datetime(2023, 3, 6, tzinfo=timezone.utc):
        if cursor.date() not in skip:
            cached.append(cursor)
        cursor += timedelta(days=1)
    cached.append(datetime(2024, 1, 1, tzinfo=timezone.utc))
    pending = [
        datetime(2022, 11, 1, tzinfo=timezone.utc),
        datetime(2022, 11, 2, tzinfo=timezone.utc),
        datetime(2023, 3, 7, tzinfo=timezone.utc),
        datetime(2023, 3, 8, tzinfo=timezone.utc),
    ]
    ordered = _frontier_first(pending, cached)
    assert ordered[0].date().isoformat() == "2023-03-07"
    assert ordered[-1].date().isoformat() == "2022-11-02"
