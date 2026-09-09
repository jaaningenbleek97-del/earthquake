import pandas as pd

from eqpred.eval.score import murphy_decompose, reliability_table
from eqpred.regions import REGIONS, in_bbox, is_issued_box, leaflet_rectangles, lon_spans, region_centroid_lon


def test_kermadec_wraps_dateline():
    assert lon_spans(177.0, -175.0) == [(177.0, 180.0), (-180.0, -175.0)]
    box = REGIONS["kermadec"]
    events = pd.DataFrame(
        {
            "lon": [178.5, -178.0, 174.0],
            "lat": [-32.0, -29.3, -32.0],
        }
    )
    inside = in_bbox(events, box)
    assert bool(inside.iloc[0])
    assert bool(inside.iloc[1])
    assert not bool(inside.iloc[2])
    assert len(leaflet_rectangles(box)) == 2


def test_kermadec_centroid_wraps():
    hik = region_centroid_lon(REGIONS["hikurangi"])
    ker = region_centroid_lon(REGIONS["kermadec"])
    assert 174 < hik < 180
    assert ker < -170 or ker > 176


def test_alpine_excludes_hikurangi_coast():
    events = pd.DataFrame(
        {
            "lon": [172.0, 178.0],
            "lat": [-43.5, -39.5],
        }
    )
    assert bool(in_bbox(events, REGIONS["alpine"]).iloc[0])
    assert not bool(in_bbox(events, REGIONS["alpine"]).iloc[1])
    assert bool(in_bbox(events, REGIONS["hikurangi"]).iloc[1])
    assert not bool(in_bbox(events, REGIONS["hikurangi"]).iloc[0])


def test_japan_chile_are_transfer_boxes():
    assert is_issued_box(REGIONS["hikurangi"]) is True
    assert is_issued_box(REGIONS["japan"]) is False
    assert is_issued_box(REGIONS["chile"]) is False
    events = pd.DataFrame(
        {
            "lon": [140.0, -72.0, 172.0],
            "lat": [38.0, -33.0, -43.5],
        }
    )
    assert bool(in_bbox(events, REGIONS["japan"]).iloc[0])
    assert bool(in_bbox(events, REGIONS["chile"]).iloc[1])
    assert not bool(in_bbox(events, REGIONS["japan"]).iloc[2])


def test_reliability_table_quantile_bins():
    ledger = pd.DataFrame(
        {
            "model_id": ["fusion.etas_x_mod.v1"] * 40,
            "region": ["hikurangi"] * 40,
            "kind": ["probability"] * 40,
            "value": [0.04] * 10 + [0.08] * 10 + [0.16] * 10 + [0.32] * 10,
            "hit": [0] * 40,
        }
    )
    table = reliability_table(ledger)
    assert len(table) == 1
    assert table[0]["region"] == "hikurangi"
    assert len(table[0]["bins"]) == 4
    assert all(b["n"] == 10 for b in table[0]["bins"])


def test_murphy_flat_has_no_resolution():
    y = [1, 0, 1, 0, 1, 0, 1, 0]
    p = [0.5] * 8
    out = murphy_decompose(p, y)
    assert out["resolution"] == 0.0
    assert out["uncertainty"] == 0.25


def test_murphy_ranked_has_resolution():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
    out = murphy_decompose(p, y)
    assert out["resolution"] > 0.1
    assert out["reliability"] < 0.02
