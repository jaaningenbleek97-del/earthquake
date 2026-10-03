import gzip
from datetime import datetime, timezone
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from eqpred.ingest.geonet import _parse_geonet_text
from eqpred.ingest.geonet_gnss import parse_fits_csv, parse_tilde_json
from eqpred.ingest.gnss import _from_decimal_year, _parse_tenv3
from eqpred.ingest.http import _looks_like_html, earthdata_credentials
from eqpred.ingest.era5 import parse_era5_netcdf
from eqpred.ingest.intermagnet import LATENCY, daily_magnet_features, parse_hapi_csv, parse_iaga2002
from eqpred.ingest.tec import _cddis_unusable, _decompress, ionex_candidate_urls, parse_ionex


def test_parse_geonet_text():
    text = (
        "#EventID|Time|Latitude|Longitude|Depth/km|MagnitudeType|Magnitude\n"
        "2016p858000|2016-11-13T11:02:56.353Z|-42.737|173.054|15.1|M|7.8\n"
    )
    rows = _parse_geonet_text(text)
    assert len(rows) == 1
    assert rows[0]["magnitude"] == 7.8
    assert rows[0]["lat"] == -42.737


def test_parse_tenv3_decimal_year():
    stamp = _from_decimal_year(2020.0)
    assert stamp == datetime(2020, 1, 1, tzinfo=timezone.utc)
    text = "AUCK 20JAN01 2020.0000 58849 2086 3 0.0 0.0 0.012 0.0 0.034 0.0 0.056\n"
    frame = _parse_tenv3(text, "AUCK")
    assert len(frame) == 1
    assert abs(frame.iloc[0]["east_m"] - 0.012) < 1e-9


def test_parse_midas_speeds_are_millimetres_per_year():
    from eqpred.ingest.gnss import parse_midas

    text = (
        "AUCK MIDAS5 2010.0000 2020.0000 10.0 100 90 80 "
        "0.030 0.040 -0.001 0.0002 0.0002 0.0008 0 0 0\n"
    )
    frame = parse_midas(text)
    assert list(frame["site"]) == ["AUCK"]
    assert abs(frame.iloc[0]["east_mm_yr"] - 30.0) < 1e-6
    assert abs(frame.iloc[0]["north_mm_yr"] - 40.0) < 1e-6
    assert frame.iloc[0]["observed_at"].year == 2020


def test_parse_fits_csv():
    text = (
        "siteID,typeID,date,value,error\n"
        "AUCK,e,2016-01-02T12:00:00Z,12.5,0.4\n"
        "# comment\n"
        "AUCK,e,2016-01-03T12:00:00Z,13.0,0.4\n"
    )
    rows = parse_fits_csv(text)
    assert len(rows) == 2
    assert rows[0][1] == 12.5
    assert rows[0][0].year == 2016


def test_parse_tilde_json_metres_to_mm():
    payload = [
        {
            "data": [
                {"val": 0.012, "ts": "2016-01-02T11:59:00Z"},
                {"val": 0.013, "ts": "2016-01-03T11:59:00Z"},
            ]
        }
    ]
    rows = parse_tilde_json(payload)
    assert len(rows) == 2
    assert abs(rows[0][1] - 12.0) < 1e-9
    assert rows[0][0].year == 2016


def test_earthdata_login_html_is_rejected():
    assert _looks_like_html(b"<!DOCTYPE html><html><title>Earthdata Login</title>")
    assert _looks_like_html(b"<html><body>authorize the application</body></html>")
    assert not _looks_like_html(b"     1.0            TEC map")


def test_decompress_gzip_ionex():
    raw = gzip.compress(b"     1.0            IONOSPHERE MAPS     MIX")
    assert "IONOSPHERE MAPS" in _decompress(raw)


def test_earthdata_credentials_from_netrc_or_env():
    creds = earthdata_credentials()
    if creds is not None:
        assert creds[0]
        assert creds[1]


def test_parse_era5_netcdf(tmp_path):
    import netCDF4

    path = tmp_path / "tiny.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 2)
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 2)
        t = ds.createVariable("valid_time", "f8", ("time",))
        t.units = "hours since 2016-01-01 00:00:00"
        t[:] = [0, 24]
        sp = ds.createVariable("sp", "f4", ("time", "lat", "lon"))
        sp[:] = [[[100000, 100200], [100100, 100300]], [[101000, 101000], [101000, 101000]]]
        soil = ds.createVariable("swvl1", "f4", ("time", "lat", "lon"))
        soil[:] = 0.25
    frame = parse_era5_netcdf(path)
    assert len(frame) == 2
    assert abs(frame.iloc[0]["era5_sp_pa"] - 100150) < 1
    assert abs(frame.iloc[0]["era5_swvl1"] - 0.25) < 1e-6
    assert frame.iloc[0]["observed_at"].year == 2016


def test_ionex_urls_prefer_public_jpl():
    old = ionex_candidate_urls(2017, 15, 17)
    assert old[0].endswith("JPLG0150.17I.gz")
    assert "sideshow.jpl.nasa.gov" in old[0]
    assert any("uqrg0150.17i.Z" in url for url in old)
    recent = ionex_candidate_urls(2026, 15, 26)
    assert "JPL0OPSFIN_20260150000_01D_02H_GIM.INX.gz" in recent[0]
    assert not any("ftp.aiub.unibe.ch" in url for url in old)


def test_cddis_login_html_disables_further_tries():
    assert _cddis_unusable(RuntimeError("Earthdata returned login HTML — authorize CDDIS"))
    assert _cddis_unusable(RuntimeError("curl: (56) Recv failure: Connection was reset"))
    assert not _cddis_unusable(RuntimeError("GET failed: HTTP 404"))


def test_parse_ionex_means():
    text = dedent(
        """\
         0.0                                                          EXPONENT
      -40.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
      20 21 22 23 24 25
        """
    )
    means = parse_ionex(text)
    assert means["tec_tecu"] == 22.5
    assert means["tec_nz_tecu"] is not None
    assert means["tec_nz_tecu"] > 0
    assert "tec_japan_tecu" in means
    assert "tec_chile_tecu" in means


def test_parse_ionex_fills_japan_chile_on_0360_grid():
    """CODE/IGS maps sometimes run longitude 0→360. Chile is west of Greenwich."""
    header = "     38.0   0.0 360.0   5.0 450.0                            LAT/LON1/LON2/DLON/H"
    chile = "    -30.0   0.0 360.0   5.0 450.0                            LAT/LON1/LON2/DLON/H"
    # 73 cells, 0..360 inclusive. Put a spike at 140°E (japan) and 285°E (chile).
    cells = ["1"] * 73
    cells[28] = "40"  # 0 + 28*5 = 140
    cells[57] = "80"  # 0 + 57*5 = 285 → -75
    row = " " + " ".join(cells)
    text = dedent(
        f"""\
         0.0                                                          EXPONENT
{header}
{row}
{chile}
{row}
        """
    )
    means = parse_ionex(text)
    # Japan 128–146 hits 130/135/140/145; Chile −76–−66 hits 285/290 → −75/−70.
    assert means["tec_japan_tecu"] == pytest.approx((1 + 1 + 40 + 1) / 4)
    assert means["tec_chile_tecu"] == pytest.approx((80 + 1) / 2)


def test_parse_ionex_reads_continuation_lines():
    """IONEX writes one latitude as several 16-value lines. Do not reset lon."""
    header = "    -40.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H"
    # 73 cells, -180→180. Spike at 175°E (hikurangi) = index 71, on a later line.
    cells = ["1"] * 73
    cells[71] = "50"
    chunks = [" " + " ".join(cells[i : i + 16]) for i in range(0, 73, 16)]
    text = "         0.0                                                          EXPONENT\n" + header + "\n" + "\n".join(chunks)
    means = parse_ionex(text)
    assert means["tec_hikurangi_tecu"] == pytest.approx(50)


IAGA_HDZF = dedent(
    """\
 Format                 IAGA-2002                                    |
 Reported               HDZF                                         |
DATE       TIME         DOY     EYRH      EYRD      EYRZ      EYRF   |
2016-01-01 00:00:00.000 001     21000.00   1400.00 -53500.00  57500.00
2016-01-01 00:01:00.000 001     21010.00   1400.00 -53500.00  57500.00
2016-01-01 00:02:00.000 001     99999.00  99999.00  99999.00  99999.00
2016-01-02 12:00:00.000 002     22000.00   1400.00 -53500.00  57500.00
"""
)


def test_parse_iaga2002_hdzf_skips_fill():
    minutes = parse_iaga2002(IAGA_HDZF)
    assert len(minutes) == 3
    assert minutes.iloc[0]["h_nt"] == 21000.0
    assert minutes.iloc[1]["h_nt"] == 21010.0
    assert minutes.iloc[2]["h_nt"] == 22000.0
    daily = daily_magnet_features(minutes, "EYR")
    assert len(daily) == 2
    assert daily.iloc[0]["mag_station"] == "EYR"
    assert abs(daily.iloc[0]["mag_h_nt"] - 21005.0) < 1e-6
    assert abs(daily.iloc[0]["mag_h_range_nt"] - 10.0) < 1e-6
    assert abs(daily.iloc[0]["mag_dbdt_nt"] - 10.0) < 1e-6
    assert daily.iloc[0]["observed_at"].startswith("2016-01-01")
    assert daily.iloc[0]["available_at"].startswith("2016-01-02")
    assert daily.iloc[1]["mag_h_nt"] == 22000.0
    assert pd.isna(daily.iloc[1]["mag_dbdt_nt"])


def test_parse_iaga2002_xyz_uses_horizontal():
    text = dedent(
        """\
 Reported               XYZF                                         |
2016-03-01 00:00:00.000 061      3000.00   4000.00  50000.00  51000.00
2016-03-01 00:01:00.000 061      99999.00   4000.00  50000.00  51000.00
"""
    )
    minutes = parse_iaga2002(text)
    assert len(minutes) == 1
    assert abs(minutes.iloc[0]["h_nt"] - 5000.0) < 1e-6


def test_intermagnet_next_run_skips_dead_codes_and_asks_minute():
    from eqpred.ingest.intermagnet import SKIP_STATIONS, STATIONS, _gin_data_url, _hapi_url

    assert "AMS" not in STATIONS and "IPM" not in STATIONS
    assert {"AMS", "IPM"} <= set(SKIP_STATIONS)
    start = datetime(2016, 1, 1, tzinfo=timezone.utc)
    gin = _gin_data_url("EYR", start, 31)
    assert "samplesPerDay=Minute" in gin
    hapi = _hapi_url("EYR", start, datetime(2016, 2, 1, tzinfo=timezone.utc))
    assert "PT1M" in hapi
    assert "PT1H" not in hapi


def test_eac4_end_stops_inside_ads_lag():
    from eqpred.ingest.so2 import eac4_end

    now = datetime(2026, 9, 16, tzinfo=timezone.utc)
    end = datetime(2026, 9, 17, tzinfo=timezone.utc)
    assert eac4_end(end, now=now) == datetime(2026, 6, 28, tzinfo=timezone.utc)


def test_parse_iaga2002_empty_garbage():
    assert parse_iaga2002("not a magnetometer file").empty
    assert daily_magnet_features(parse_iaga2002(""), "EYR").empty
    assert LATENCY.days == 1


def test_parse_hapi_csv_skips_fill_and_header():
    text = (
        "Time,H,D,Z,F\n"
        "2016-01-01T00:00Z,21000.00,1400.00,-53500.00,57500.00\n"
        "2016-01-01T00:01:00.000Z,99999.00,99999.00,99999.00,99999.00\n"
        "# comment\n"
    )
    minutes = parse_hapi_csv(text)
    assert len(minutes) == 1
    assert minutes.iloc[0]["h_nt"] == 21000.0


def test_wave2_family_keeps_better_train_column():
    from eqpred.models.wave2 import select_family_winners

    trials = select_family_winners(
        [
            {
                "column": "era5_sp_pa",
                "family": "era5",
                "used_on_issued": True,
                "train_brier_with_gain": 0.080,
                "train_rank_with_gain": 0.12,
            },
            {
                "column": "era5_swvl1",
                "family": "era5",
                "used_on_issued": True,
                "train_brier_with_gain": 0.070,
                "train_rank_with_gain": 0.10,
            },
            {
                "column": "tec_nz_tecu",
                "family": "tec",
                "used_on_issued": True,
                "train_brier_with_gain": 0.075,
                "train_rank_with_gain": 0.08,
            },
        ]
    )
    issued = {row["column"]: row["used_on_issued"] for row in trials}
    assert issued["era5_swvl1"] is True
    assert issued["era5_sp_pa"] is False
    assert issued["tec_nz_tecu"] is True
    assert trials[0]["held_out_reason"] == "same_family_worse_on_train"


def test_wave2_gain_is_one_when_held_out_or_missing():
    from eqpred.models.wave2 import wave2_gain

    row = pd.Series({"tec_nz_tecu": 12.0, "era5_sp_pa": np.nan})
    assert wave2_gain({"regions": {}}, "hikurangi", row) == 1.0
    payload = {
        "regions": {
            "hikurangi": {
                "used_on_issued": True,
                "columns": [
                    {
                        "used_on_issued": True,
                        "column": "era5_sp_pa",
                        "beta": 1.0,
                        "mean": 101000.0,
                        "scale": 100.0,
                    }
                ],
            }
        }
    }
    assert wave2_gain(payload, "hikurangi", row) == 1.0


def test_wave2_gain_clips_and_multiplies():
    from eqpred.models.wave2 import wave2_gain

    row = pd.Series({"tec_nz_tecu": 2.0, "n_geonet_m3_7d": 0.0})
    payload = {
        "regions": {
            "hikurangi": {
                "used_on_issued": True,
                "columns": [
                    {
                        "used_on_issued": True,
                        "column": "tec_nz_tecu",
                        "beta": 1.0,
                        "mean": 0.0,
                        "scale": 1.0,
                    },
                    {
                        "used_on_issued": True,
                        "column": "n_geonet_m3_7d",
                        "beta": 1.0,
                        "mean": 0.0,
                        "scale": 1.0,
                    },
                ],
            }
        }
    }
    g = wave2_gain(payload, "hikurangi", row)
    assert abs(g - float(np.exp(1.5) * 1.0)) < 1e-12
