from datetime import datetime, timezone

from eqpred.ingest.wiki import parse_pageviews
from eqpred.ingest.wspr import _chunk_bounds, parse_wspr


def test_parse_wspr_clickhouse_json():
    payload = {
        "data": [
            {"d": "2024-01-01", "wspr_snr_med": -14.5, "wspr_n": 120},
            {"d": "2024-01-02", "wspr_snr_med": -11.0, "wspr_n": 90},
        ]
    }
    frame = parse_wspr(payload)
    assert len(frame) == 2
    assert float(frame.iloc[0]["wspr_snr_med"]) == -14.5
    assert "available_at" in frame.columns
    assert str(frame.iloc[0]["observed_at"]).startswith("2024-01-01")


def test_parse_wiki_pageviews():
    payload = {
        "items": [
            {"timestamp": "2024010100", "views": 12345, "article": "Earthquake"},
            {"timestamp": "2024010200", "views": 13000, "article": "Earthquake"},
        ]
    }
    frame = parse_pageviews(payload)
    assert len(frame) == 2
    assert float(frame.iloc[0]["wiki_eq_views"]) == 12345
    assert str(frame.iloc[1]["observed_at"]).startswith("2024-01-02")


def test_wspr_chunks_by_year():
    start = datetime(2018, 6, 1, tzinfo=timezone.utc)
    end = datetime(2020, 3, 1, tzinfo=timezone.utc)
    chunks = _chunk_bounds(start, end)
    assert len(chunks) == 3
    assert chunks[0][0].year == 2018
    assert chunks[-1][1] == end
