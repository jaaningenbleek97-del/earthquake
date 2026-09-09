import json
from pathlib import Path

from eqpred.ingest.comcat_versions import parse_origin_versions
from eqpred.ingest.run import merge_keyed_frames
from eqpred.stamp import parse_utc


FIXTURE = Path(__file__).parent / "fixtures" / "comcat_detail_versions.json"


def test_parse_origin_versions_keeps_revision_order():
    detail = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = parse_origin_versions(detail)
    assert [r["magnitude"] for r in rows] == [6.8, 7.4]
    assert rows[0]["status"] == "automatic"
    assert rows[1]["status"] == "reviewed"
    assert rows[0]["event_id"] == "us0fakeeq01"
    first, second = parse_utc([rows[0]["version_at"], rows[1]["version_at"]])
    assert first < second


def test_parse_origin_versions_records_deleted_event():
    detail = {
        "id": "us0gone",
        "properties": {
            "time": 1577836800000,
            "place": "nowhere",
            "products": {
                "origin": [
                    {
                        "updateTime": 1577844000000,
                        "status": "UPDATE",
                        "properties": {
                            "magnitude": "6.2",
                            "latitude": "1.0",
                            "longitude": "2.0",
                            "depth": "10",
                            "eventtime": "2020-01-01T00:00:00.000Z",
                            "evaluation-status": "automatic",
                        },
                    },
                    {
                        "updateTime": 1577930400000,
                        "status": "DELETE",
                        "properties": {
                            "magnitude": "6.2",
                            "latitude": "1.0",
                            "longitude": "2.0",
                            "depth": "10",
                            "eventtime": "2020-01-01T00:00:00.000Z",
                            "evaluation-status": "reviewed",
                        },
                    },
                ],
                "deleted-event": [
                    {
                        "updateTime": 1577930500000,
                        "properties": {
                            "eventtime": "2020-01-01T00:00:00.000Z",
                        },
                    }
                ],
            },
        },
    }
    rows = parse_origin_versions(detail)
    assert rows[0]["status"] == "automatic"
    assert any(r["status"] == "deleted" for r in rows)


def test_merge_version_rows_uses_event_and_version_at():
    import pandas as pd

    existing = pd.DataFrame(
        {
            "event_id": ["a", "a"],
            "version_at": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
            "magnitude": [6.1, 6.3],
        }
    )
    incoming = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "version_at": ["2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"],
            "magnitude": [6.4, 7.0],
        }
    )
    merged = merge_keyed_frames(existing, incoming, ("event_id", "version_at"))
    a_rows = merged.loc[merged["event_id"] == "a"].sort_values("version_at")
    assert list(a_rows["magnitude"]) == [6.1, 6.4]
    assert len(merged) == 3


def test_ids_in_window_can_stay_inside_issued_boxes():
    from datetime import datetime, timezone

    import pandas as pd

    from eqpred.ingest.comcat_versions import ids_in_window
    from eqpred.regions import issued_boxes

    catalog = pd.DataFrame(
        {
            "event_id": ["nz-in", "jp-out", "nz-small"],
            "observed_at": pd.to_datetime(
                ["2012-03-01T00:00:00Z", "2012-03-02T00:00:00Z", "2012-03-03T00:00:00Z"],
                utc=True,
            ),
            "magnitude": [5.6, 6.2, 5.0],
            "lat": [-39.5, 38.0, -39.5],
            "lon": [177.5, 142.0, 177.5],
        }
    )
    start = datetime(2012, 1, 1, tzinfo=timezone.utc)
    end = datetime(2013, 1, 1, tzinfo=timezone.utc)
    all_ids = ids_in_window(catalog, start, end, 5.5)
    box_ids = ids_in_window(catalog, start, end, 5.5, regions=issued_boxes())
    assert set(all_ids) == {"nz-in", "jp-out"}
    assert box_ids == ["nz-in"]
