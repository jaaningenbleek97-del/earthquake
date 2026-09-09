import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from eqpred.ingest.run import latest_raw, merge_daily_frames, merge_event_frames
from eqpred.ingest.watchdog import (
    REQUIRED,
    StaleSourcesError,
    ensure_can_issue,
    freshness_report,
    issuable_times,
)
from eqpred.models.predict import begin_issue
from eqpred.paths import raw_dir
from eqpred.records import iso


def _stamp(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc)


def _write_source(root: Path, source: str, available: datetime, observed: datetime | None = None):
    folder = raw_dir(root) / source
    folder.mkdir(parents=True, exist_ok=True)
    obs = observed or (available - timedelta(days=1))
    pd.DataFrame(
        {
            "observed_at": [iso(obs)],
            "available_at": [iso(available)],
        }
    ).to_parquet(folder / "run.parquet", index=False)


def _write_required(root: Path, available: datetime):
    for source in REQUIRED:
        _write_source(root, source, available)


def test_fresh_required_sources_pass(tmp_path):
    as_of = _stamp(2025, 12, 31)
    _write_required(tmp_path, as_of)
    report = freshness_report(as_of, tmp_path)
    assert report["ok"] is True
    assert report["blockers"] == []
    by_name = {row["source"]: row for row in report["sources"]}
    assert by_name["nasa.omni"]["status"] == "fresh"
    assert by_name["usgs.comcat"]["status"] == "fresh"
    assert by_name["wilcox.spf"]["required"] is False
    assert by_name["wilcox.spf"]["blocking"] is False


def test_stale_omni_blocks_even_if_catalog_is_current(tmp_path):
    as_of = _stamp(2026, 9, 6)
    _write_source(tmp_path, "usgs.comcat", as_of)
    _write_source(tmp_path, "wilcox.spf", as_of)
    _write_source(tmp_path, "nasa.omni", _stamp(2025, 12, 28))
    report = freshness_report(as_of, tmp_path)
    assert report["ok"] is False
    assert "nasa.omni" in report["blockers"]
    omni = next(row for row in report["sources"] if row["source"] == "nasa.omni")
    assert omni["status"] == "stale"
    assert omni["lag_days"] > 200


def test_missing_wilcox_does_not_block(tmp_path):
    as_of = _stamp(2025, 6, 1)
    _write_source(tmp_path, "usgs.comcat", as_of)
    _write_source(tmp_path, "nasa.omni", as_of)
    report = freshness_report(as_of, tmp_path)
    assert report["ok"] is True
    assert "wilcox.spf" not in report["blockers"]
    wilcox = next(row for row in report["sources"] if row["source"] == "wilcox.spf")
    assert wilcox["status"] == "missing"
    assert wilcox["required"] is False
    assert wilcox["blocking"] is False


def test_optional_stale_does_not_block(tmp_path):
    as_of = _stamp(2025, 12, 31)
    _write_required(tmp_path, as_of)
    _write_source(tmp_path, "igs.tec", _stamp(2024, 1, 1))
    report = freshness_report(as_of, tmp_path)
    assert report["ok"] is True
    tec = next(row for row in report["sources"] if row["source"] == "igs.tec")
    assert tec["status"] == "stale"
    assert tec["blocking"] is False


def test_holdout_morning_is_fresh_when_store_runs_past_it(tmp_path):
    _write_required(tmp_path, _stamp(2025, 12, 31))
    report = freshness_report(_stamp(2020, 6, 1), tmp_path)
    assert report["ok"] is True


def test_issuable_times_drops_only_the_stale_tail(tmp_path):
    _write_required(tmp_path, _stamp(2025, 12, 20))
    days = [_stamp(2025, 12, 15), _stamp(2025, 12, 31)]
    kept = issuable_times(days, tmp_path)
    assert [ts.date() for ts in kept] == [_stamp(2025, 12, 15).date()]


def test_ensure_can_issue_raises_when_every_morning_is_stale(tmp_path):
    _write_required(tmp_path, _stamp(2025, 1, 1))
    with pytest.raises(StaleSourcesError) as exc:
        ensure_can_issue([_stamp(2026, 9, 6)], tmp_path)
    assert "nasa.omni" in exc.value.report["blockers"]


def test_begin_issue_leaves_batches_when_stale(tmp_path):
    dest = tmp_path / "predictions"
    dest.mkdir()
    leftover = dest / "batch-holdout.json"
    leftover.write_text("[]", encoding="utf-8")
    _write_required(tmp_path, _stamp(2025, 1, 1))
    issue = pd.Timestamp("2026-09-06", tz="UTC")
    rates = pd.DataFrame({"issue_time": [issue]})
    window = pd.Series([True])
    with pytest.raises(StaleSourcesError):
        begin_issue(rates, window, tmp_path, dest)
    assert leftover.exists()


def test_begin_issue_leaves_batches_when_window_empty(tmp_path):
    dest = tmp_path / "predictions"
    dest.mkdir()
    leftover = dest / "batch-holdout.json"
    leftover.write_text("[]", encoding="utf-8")
    rates = pd.DataFrame({"issue_time": [pd.Timestamp("2025-06-01", tz="UTC")]})
    window = pd.Series([False])
    with pytest.raises(RuntimeError, match="existing cards left on disk"):
        begin_issue(rates, window, tmp_path, dest)
    assert leftover.exists()


def test_begin_issue_keeps_cards_outside_the_window(tmp_path):
    dest = tmp_path / "predictions"
    dest.mkdir()
    leftover = dest / "batch-holdout.json"
    leftover.write_text(
        json.dumps(
            [
                {"issued_at": "2019-01-01T00:00:00Z", "prediction_id": "holdout"},
                {"issued_at": "2025-06-01T00:00:00Z", "prediction_id": "overlap"},
            ]
        ),
        encoding="utf-8",
    )
    as_of = _stamp(2025, 6, 1)
    _write_required(tmp_path, as_of)
    issue = pd.Timestamp(as_of)
    rates = pd.DataFrame({"issue_time": [issue]})
    window = pd.Series([True])
    gated = begin_issue(rates, window, tmp_path, dest)
    assert bool(gated.iloc[0]) is True
    kept = json.loads(leftover.read_text(encoding="utf-8"))
    assert [card["prediction_id"] for card in kept] == ["holdout"]


def test_latest_raw_picks_newest_mtime(tmp_path):
    folder = tmp_path / "data" / "raw" / "igs.tec"
    folder.mkdir(parents=True)
    older = folder / "zzz.parquet"
    newer = folder / "aaa.parquet"
    pd.DataFrame({"tec_tecu": [1.0]}).to_parquet(older)
    pd.DataFrame({"tec_tecu": [2.0], "tec_japan_tecu": [9.0]}).to_parquet(newer)
    frame = latest_raw("igs.tec", root=tmp_path)
    assert float(frame.iloc[0]["tec_tecu"]) == 2.0
    assert "tec_japan_tecu" in frame.columns


def test_merge_daily_frames_keeps_history():
    old = pd.DataFrame(
        {
            "observed_at": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
            "kp": [2.0, 3.0],
        }
    )
    new = pd.DataFrame(
        {
            "observed_at": ["2020-01-02T00:00:00Z", "2026-02-01T00:00:00Z"],
            "kp": [3.5, 1.0],
        }
    )
    merged = merge_daily_frames(old, new)
    assert list(pd.to_datetime(merged["observed_at"], utc=True).dt.date.astype(str)) == [
        "2020-01-01",
        "2020-01-02",
        "2026-02-01",
    ]
    assert float(merged.loc[merged["kp"] == 3.5, "kp"].iloc[0]) == 3.5


def test_merge_event_frames_keeps_history():
    old = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "observed_at": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
            "magnitude": [5.0, 6.0],
        }
    )
    new = pd.DataFrame(
        {
            "event_id": ["b", "c"],
            "observed_at": ["2020-01-02T00:00:00Z", "2026-02-01T00:00:00Z"],
            "magnitude": [6.1, 5.5],
        }
    )
    merged = merge_event_frames(old, new)
    assert list(merged["event_id"]) == ["a", "b", "c"]
    assert float(merged.loc[merged["event_id"] == "b", "magnitude"].iloc[0]) == 6.1


def test_merge_keyed_frames_keeps_station_days():
    from eqpred.ingest.run import merge_keyed_frames

    old = pd.DataFrame(
        {
            "observed_at": ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            "mag_station": ["EYR", "KAK"],
            "mag_h_nt": [21000.0, 30000.0],
        }
    )
    new = pd.DataFrame(
        {
            "observed_at": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
            "mag_station": ["KAK", "HON"],
            "mag_h_nt": [30100.0, 28000.0],
        }
    )
    merged = merge_keyed_frames(old, new, ("observed_at", "mag_station"))
    assert len(merged) == 3
    kak = merged.loc[merged["mag_station"] == "KAK"]
    assert float(kak.iloc[0]["mag_h_nt"]) == 30100.0
