from datetime import datetime, timezone

import pandas as pd
import pytest

from eqpred.eval.hunt import driver_family
from eqpred.ingest.das import pull_das
from eqpred.ingest.gnssa import parse_acoustics_csv, parse_bcsonardyne
from eqpred.ingest.hydro import daily_hydro
from eqpred.ingest.roti import _wanted_days, parse_rinex_roti
from eqpred.ingest.superdarn import discover_unwin
from eqpred.ingest.tir import parse_power_daily
from eqpred.ingest.vlf import parse_vlf_csv


def test_gnssa_acoustics_csv_daily_median():
    text = "time,twtt_sec\n2019-03-01T00:00:00Z,1.20\n2019-03-01T12:00:00Z,1.40\n2019-03-02T00:00:00Z,1.50\n"
    frame = parse_acoustics_csv(text, "HIK1")
    assert len(frame) == 2
    assert float(frame.iloc[0]["gnssa_range"]) == pytest.approx(1.3)
    assert frame.iloc[0]["gnssa_site"] == "HIK1"


def test_gnssa_acoustics_unix_common_time():
    text = (
        "Common Time, Range(s)\n"
        "1676998936.926711,4.05\n"
        "1676998951.919643,4.07\n"
    )
    frame = parse_acoustics_csv(text, "HAW1")
    assert len(frame) == 1
    assert str(frame.iloc[0]["observed_at"])[:10] == "2023-02-21"


def test_gnssa_bcsonardyne_daily_median():
    text = (
        "2038,511520,1549058720,2019/02/01 22:05:20.174 >CS:5209,R3766365[XC81,SNR16]\n"
        "2038,511520,1549058720,2019/02/01 22:06:20.174 >CS:5209,R3766405[XC80,SNR15]\n"
        "2038,511520,1549145120,2019/02/02 22:05:20.174 >CS:5209,R3767000[XC79,SNR14]\n"
    )
    frame = parse_bcsonardyne(text, "HAW1")
    assert len(frame) == 2
    assert frame.iloc[0]["gnssa_site"] == "HAW1"
    assert float(frame.iloc[0]["gnssa_range"]) == 3766385.0


def test_roti_wanted_days_stride_does_not_densify_historical_years():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2021, 1, 1, tzinfo=timezone.utc)
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    days, step = _wanted_days(start, end, now=now, stride=4)
    assert step == 4
    assert len(days) == 92
    assert days[0] == start
    assert days[1] == datetime(2020, 1, 5, tzinfo=timezone.utc)
    assert all((day - start).days % 4 == 0 for day in days)


def test_roti_rinex2_wrapped_gps_phase():
    header = (
        "     2.11           OBSERVATION DATA    G (GPS)             RINEX VERSION / TYPE\n"
        "     8    C1    L1    S1    P1    C2    L2    S2    P2          # / TYPES OF OBSERV\n"
        "                                                            END OF HEADER\n"
    )
    lines = [header]
    l1, l2 = 112938565.0, 87841184.0
    for i in range(16):
        mm, ss = divmod(i * 30, 60)
        lines.append(f" 15  1  1  0 {mm:2d} {ss:10.7f}  0  2R10G26\n")
        lines.append("  21186989.117   112938565.695 7        47.500    21186989.430\n")
        lines.append("  87841184.264 6        40.100    21186998.305\n")
        lines.append(
            f"  25075368.477   {l1 + i * 0.4:14.3f} 7        42.200    25075368.500\n"
        )
        lines.append(f"  {l2 + i * 0.3:14.3f} 6        38.900    25075374.133\n")
    parsed = parse_rinex_roti("".join(lines), epoch_stride=1)
    assert parsed["roti_n"] == 16
    assert parsed["roti"] is not None
    thinned = parse_rinex_roti("".join(lines), epoch_stride=4)
    assert thinned["roti_n"] == 4


def test_roti_rinex3_parser_sees_phase():
    header = (
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n"
        "G     2 L1C L2W                                               SYS / # / OBS TYPES\n"
        "                                                            END OF HEADER\n"
    )
    lines = [header]
    l1, l2 = 100000000.0, 78000000.0
    for i in range(16):
        lines.append(f"> 2024 01 01 00 {i:02d}  0.0000000  0  1\n")
        lines.append(f"G01  {l1 + i * 0.4:14.3f}  {l2 + i * 0.3:14.3f}\n")
    parsed = parse_rinex_roti("".join(lines))
    assert parsed["roti_n"] >= 16
    assert parsed["roti"] is not None or parsed["roti_slips"] is not None


def test_hydro_daily_from_station_samples():
    samples = pd.DataFrame(
        {
            "observed_at": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
            "hydro_station": ["Lake X", "Lake X"],
            "hydro_level": [10.0, 10.5],
            "hydro_region": ["alpine", "alpine"],
        }
    )
    daily = daily_hydro(samples)
    assert "hydro_nz" in daily.columns
    assert "hydro_alpine" in daily.columns
    assert len(daily) == 2


def test_vlf_and_tir_are_foils_not_leftover():
    assert driver_family("vlf_amp") == "foil"
    assert driver_family("tir_ts") == "foil"
    vlf = parse_vlf_csv("time,amp\n2020-01-01,12.0\n2020-01-02,13.0\n", "nwc")
    assert float(vlf.iloc[0]["vlf_nwc"]) == 1.0
    tir = parse_power_daily(
        {"properties": {"parameter": {"TS": {"20200101": 12.5, "20200102": 13.0}}}},
        "hikurangi",
    )
    assert len(tir) == 2


def test_das_stub_downloads_nothing():
    frame = pull_das(datetime(2019, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc), "test")
    assert frame.empty


def test_superdarn_cdaweb_without_unwin_is_blocked():
    html = '<a href="gbay/">gbay</a><a href="sask/">sask</a>'

    def fake_page(url, timeout=25):
        if "darn" in url:
            return html
        raise RuntimeError("no listing")

    found = discover_unwin(fetch_page=fake_page)
    assert found["blocked"] is True
    assert "no Unwin" in found["cdaweb_note"]
