from datetime import datetime, timezone

import pandas as pd

from eqpred.eval.hunt import driver_family
from eqpred.eval.mainshock import is_issued_driver
from eqpred.features.deep_counts import thrust_mask, window_counts
from eqpred.ingest.cmt import parse_ndk
from eqpred.ingest.geonet_mt import parse_geonet_cmt
from eqpred.ingest.sse2019 import parse_sse2019_catalog
from eqpred.ingest.hobitss import parse_obs_catalog, parse_sws_summary
from eqpred.ingest.nied import in_japan


NDK = """\
PDEQ 2026/01/28 12:00:00.0 -36.00 178.00  40.0 0.0 6.2 KERMADEC
C202601281200A   B: 0 0 0 S: 0 0 0 M: 0 0 0 CMT: 1 TRIHD: 1.0
CENTROID:      1.0 0.1 -36.00 0.01 178.00 0.01  80.0  0.0 FIX
25 1.000 0.010 -0.500 0.010 -0.500 0.010 0.100 0.010 0.200 0.010 0.300 0.010
V10   1.0 45 90   0.0 0 0  -1.0 45 270   1.0 10 20  90  200 80  90
MLI  1976/01/02 01:00:00.0 -38.00 178.50  20.0 5.5 0.0 HIKURANGI
M010276A         B: 0 0 0 S: 0 0 0 M: 0 0 0 CMT: 1 BOXHD: 1.0
CENTROID:      1.0 0.0 -38.00 0.00 178.50 0.00  15.0 0.0 FREE
24 0.500 0.010 -0.200 0.010 -0.300 0.010 0.100 0.010 0.100 0.010 0.100 0.010
V10   0.5 10 0  0.0 0 0  -0.5 10 180   0.4 20 40  10  100 70  20
"""


def test_ndk_hold_and_thrust():
    frame = parse_ndk(NDK)
    assert len(frame) == 2
    quick = frame.loc[frame["cmt_id"] == "C202601281200A"].iloc[0]
    assert bool(quick["quick"]) is True
    assert str(quick["available_at"]).startswith("2026-02-04")
    assert quick["depth_km"] == 80.0
    assert bool(thrust_mask(pd.Series([quick["rake"]])).iloc[0]) is True
    archived = frame.loc[frame["cmt_id"] == "M010276A"].iloc[0]
    assert str(archived["available_at"]).startswith("1976-02-01")
    assert bool(thrust_mask(pd.Series([archived["rake"]])).iloc[0]) is False


def test_depth_count_waits_until_public():
    issue = pd.Series(pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC"))
    observed = pd.Series([datetime(2020, 1, 5, 12, tzinfo=timezone.utc)])
    available = pd.Series([datetime(2020, 1, 20, tzinfo=timezone.utc)])
    counts = window_counts(observed, available, issue, window_days=30)
    assert counts[10] == 0
    assert counts[20] == 1
    assert counts[39] == 0


def test_new_deep_columns_can_vote_on_the_working_card():
    for name in ("n_slab_kermadec_30d", "cmt_global_n90", "sws_obs_tlag", "nied_japan_n90", "obs_hikurangi_n", "rupture_n", "swsdb_nz_dt"):
        assert is_issued_driver(name) is True
    assert driver_family("cmt_kermadec_thrust90") == "mechanism"
    assert driver_family("sws_land_tlag") == "mantle"
    assert driver_family("n_iface_hikurangi_30d") == "interface"
    assert driver_family("n_slab_kermadec_30d") == "slab"


def test_sws_keeps_good_grades_and_obs_catalog_parses():
    text = (
        "1event,stat,tlag,gradeABCNR\n"
        "2015.1.15.13.30.9-x,EBS1,0.09,BCl\n"
        "2015.2.1.1.0.0-x,EBS1,0.40,C\n"
    )
    splits = parse_sws_summary(text, "obs")
    assert len(splits) == 1
    assert splits.iloc[0]["tlag_s"] == 0.09
    assert str(splits.iloc[0]["available_at"]).startswith("2019-09-12")
    catalog = parse_obs_catalog(
        "evid lat lon depth epoch_time Ml np ns nps date time\n"
        "49570 -38.0780 178.4746  26.4119 1400405338.3387  1.66  2  5  7 05/18/2014 09:28:58.339\n"
    )
    assert len(catalog) == 1
    assert catalog.iloc[0]["depth_km"] == 26.4119
    assert str(catalog.iloc[0]["available_at"]).startswith("2018-12-07")


def test_splitting_db_keeps_new_zealand_and_the_paper_year():
    from eqpred.ingest.sws_db import parse_splitting_db

    references = "id|Tag|Authors|Year|Title|Journal|No|Pages|DOI|Link|Comment\n9|X|A|2014|T|J|1|1| | | \n"
    measurements = (
        "id|Station|Latitude|Longitude|phi|dt|refID|Phase|pmax|pmin|dtmax|dtmin|remark\n"
        "1|WEL|-41.3|174.8|40|1.2|9|SKS| | | | | \n"
        "2|LON|46.75|-121.81|84|1|9|SKS| | | | | \n"
    )
    frame = parse_splitting_db(measurements, references)
    assert len(frame) == 1
    assert frame.iloc[0]["station"] == "WEL"
    assert str(frame.iloc[0]["available_at"]).startswith("2015-01-01")


def test_rupture_model_waits_for_the_paper_year():
    from eqpred.ingest.rupture import parse_model_meta

    text = """
[event]
 public_ids = '2021p169083'
 evdate    = [5, 3, 2021]
[source]
 Mw        = 7.3
 evlat     = -37.466
 evlon     = 179.774
 evdep     = 72.00
"""
    parsed = parse_model_meta(text, "Okuwaki_EtAl2021")
    assert parsed is not None
    assert parsed["event_id"] == "2021p169083"
    assert str(parsed["available_at"]).startswith("2021-12-31")
    assert is_issued_driver("rupture_n") is True
    assert driver_family("swsdb_nz_dt") == "mantle"


def test_sse2019_catalog_is_hidden_until_publication():
    text = "time,lat,lon,dep,mag\n2018-11-02T03:07:48.814000Z,-38.65,178.45,17.7,1.8\n"
    frame = parse_sse2019_catalog(text)
    assert len(frame) == 1
    assert str(frame.iloc[0]["available_at"]).startswith("2025-07-31")
    assert is_issued_driver("obs_sse2019_n") is True


def test_geonet_cmt_holds_seven_days():
    text = (
        "PublicID,Date,Latitude,Longitude,strike1,dip1,rake1,strike2,dip2,rake2,ML,Mw,Mo\n"
        "2103645,20030821121200,-45.1929,166.8300,213,56,98,20,35,79,7.0,7.1,5.61e+26\n"
    )
    frame = parse_geonet_cmt(text)
    assert len(frame) == 1
    assert str(frame.iloc[0]["observed_at"]).startswith("2003-08-21")
    assert str(frame.iloc[0]["available_at"]).startswith("2003-08-28")
    assert float(frame.iloc[0]["mw"]) == 7.1


def test_nied_japan_box_drops_new_zealand():
    events = pd.DataFrame(
        {
            "lat": [35.0, -38.0],
            "lon": [142.0, 178.0],
            "observed_at": ["2018-06-01T00:00:00Z", "2018-06-01T00:00:00Z"],
        }
    )
    kept = in_japan(events)
    assert len(kept) == 1
    assert float(kept.iloc[0]["lat"]) == 35.0
