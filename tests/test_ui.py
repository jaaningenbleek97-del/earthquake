from pathlib import Path

import pytest

from eqpred.paths import prediction_dir, repo_root
from eqpred.stamp import parse_utc
from eqpred.ui.data import day_view, feature_series, globe_view, query_ledger, snapshot

HAS_LEDGER = (prediction_dir() / "ledger.parquet").exists()


@pytest.mark.skipif(not HAS_LEDGER, reason="no scored ledger on disk")
def test_snapshot_has_lab_fields():
    snap = snapshot()
    assert snap["n_cards"] > 0
    assert snap.get("n_scored", snap["n_cards"]) <= snap["n_cards"]
    assert snap.get("n_open_windows", 0) == snap["n_cards"] - snap.get("n_scored", snap["n_cards"])
    assert snap["catalog"]["n_events"] > 0
    assert "published chance" in snap["what"]
    assert "typical year" in snap["what"]
    assert snap["features"].get("tec_boxes") in {"joined", "not joined"}
    assert snap["catalog"]["etas"] == "as-of available_at"
    assert snap["catalog"]["hits"] == "final bulletin"
    assert snap["battery"]
    assert any(b.get("name") == "fusion_timeshift_180d" for b in snap["battery"])
    assert any(b.get("name") == "fusion_climatology_m7" for b in snap["battery"])
    assert any(b.get("name") == "fusion_modulator_m7" for b in snap["battery"])
    assert any(b.get("name") == "fusion_issued_m7" for b in snap["battery"])
    assert snap["models"].get("calibration", {}).get("issued") == "calibrated_etas"
    assert "used_on_issued" in snap["models"].get("nowcast", {})
    coul = snap["models"].get("coulomb", {}).get("regions", {})
    assert "hikurangi" in coul
    assert coul["hikurangi"]["used_on_issued"] is False
    assert coul.get("alpine", {}).get("used_on_issued") is False
    if "cusp" in snap["models"]:
        assert "regions" in snap["models"]["cusp"]
        assert snap["models"]["cusp"]["regions"]["hikurangi"]["used_on_issued"] is False
        assert snap["models"]["cusp"]["regions"]["alpine"]["used_on_issued"] is False
    assert snap["models"]["calibration"]["scale"] > 1.0
    assert snap["models"]["calibration"]["scales"]["hikurangi"] > 0
    assert snap["models"]["calibration"]["scales"]["kermadec"] > 0
    assert 0 < snap["models"]["calibration"]["scales"]["alpine"] < 1.0
    global_rel = next(
        (
            r
            for r in snap["reliability"]
            if r.get("region") == "global" and r.get("model_id") == "fresh.mainshock.v1"
        ),
        None,
    )
    assert global_rel and len(global_rel.get("bins") or []) >= 3
    hik = next(
        (h for h in snap["heads"] if h.get("region") == "hikurangi" and h.get("model_id") == "fresh.mainshock.v1"),
        None,
    )
    assert hik and hik.get("p_hit_corr") is not None
    assert hik.get("mean_near_miss") is not None
    assert hik["mean_near_miss"] >= hik["hit_rate"]
    assert snap["regions"]["hikurangi"]["kind"] == "bbox"
    assert snap["regions"]["kermadec"]["min_lon"] > snap["regions"]["kermadec"]["max_lon"]
    assert any(
        h["model_id"] == "fresh.mainshock.v1"
        and h.get("region") == "global"
        and h.get("magnitude_min") == 7.0
        for h in snap["heads"]
    )
    assert any(
        h["model_id"] == "fresh.mainshock.v1"
        and h.get("region") == "kermadec"
        and h.get("magnitude_min") == 6.0
        for h in snap["heads"]
    )
    assert snap["models"].get("budget", {}).get("issued") == "monthly_cap_walk_forward"
    assert snap["models"]["budget"]["days_per_month"]["alarm.spf.v2"] == 6
    spf = next((h for h in snap["heads"] if h.get("model_id") == "alarm.spf.v2"), None)
    assert spf and spf.get("budget_days_per_month") == 6
    if any(h.get("model_id") == "alarm.quiet.v1" for h in snap["heads"]):
        assert snap["episodes"]
        assert all(ep.get("verdict") for ep in snap["episodes"])
    else:
        assert snap["episodes"] == []
    assert snap["regions"]["kermadec"]["magnitude_min"] == 6.0
    assert "migration" in snap["features"]
    if "migration" in snap.get("models", {}):
        assert "regions" in snap["models"]["migration"]
    watch = snap.get("watchdog") or {}
    registry = snap.get("registry") or {}
    if registry.get("id"):
        assert registry.get("brain_hash")
        assert registry.get("code_hash")
        assert registry.get("n_versions", 0) >= 1
    hunt = snap.get("hunt") or {}
    if hunt.get("n_tests"):
        assert hunt["n_tests"] > 200
        assert "n_confirmed_unconventional" in hunt
    seq_hunt = snap.get("hunt_sequence") or {}
    if seq_hunt.get("n_tests"):
        assert seq_hunt.get("writes_issued_p") is False
        assert seq_hunt.get("sequence") is True
    bt = snap.get("backtest") or {}
    if bt.get("n_trades"):
        assert bt["trades"]
        first = bt["trades"][0]
        assert first.get("driver")
        assert "lift" in first
        assert first.get("n_days", 0) > 0
    brain = snap.get("brain") or {}
    if brain.get("n_features"):
        assert brain.get("writes_issued_p") is False
        assert "locked_lift" in brain
    assert watch.get("sources")
    names = {row.get("source") for row in watch["sources"]}
    assert {"usgs.comcat", "nasa.omni", "wilcox.spf"} <= names
    assert "ok" in watch
    assert "blockers" in watch
    if "anti" in snap.get("models", {}):
        assert snap["models"]["anti"].get("model_id") == "anti.quiet.v1"
        assert snap["models"]["anti"].get("quantile") == 0.15
        assert snap["models"]["budget"]["days_per_month"]["anti.quiet.v1"] == 4
        anti_heads = [h for h in snap["heads"] if h.get("model_id") == "anti.quiet.v1"]
        assert all(h.get("quiet_base_rate") is not None for h in anti_heads)
        assert all(h.get("budget_days_per_month") == 4 for h in anti_heads)


@pytest.mark.skipif(not HAS_LEDGER, reason="no scored ledger on disk")
def test_ledger_fresh_drivers_are_structured():
    page = query_ledger(model="fresh.mainshock.v1", limit=8)
    assert page["rows"]
    for row in page["rows"]:
        assert row["drivers"]
        first = row["drivers"][0]
        assert isinstance(first, dict)
        assert {"driver", "lag_hours", "contribution"} <= set(first)


@pytest.mark.skipif(not HAS_LEDGER, reason="no scored ledger on disk")
def test_ledger_filters_hits():
    page = query_ledger(hit="1", limit=20)
    assert page["total"] >= 1
    assert all(row["hit"] == 1 for row in page["rows"])
    assert Path(repo_root(), "src", "eqpred", "ui", "static", "index.html").exists()


@pytest.mark.skipif(not HAS_LEDGER, reason="no scored ledger on disk")
def test_ledger_filters_open_windows():
    page = query_ledger(hit="open", limit=20)
    assert "offset" in page
    for row in page["rows"]:
        closed = row.get("window_closed")
        assert closed is False or row.get("hit") is None


@pytest.mark.skipif(not (repo_root() / "data" / "features" / "global_daily.parquet").exists(), reason="no features")
def test_feature_series_includes_tec_boxes():
    pack = feature_series(start="2024-01-01", end="2024-02-01")
    assert pack["days"]
    day = pack["days"][0]
    assert "tec_tecu" in day
    assert "tec_japan_tecu" in day
    assert "tec_chile_tecu" in day
    assert "tec_hikurangi_tecu" in day


@pytest.mark.skipif(not HAS_LEDGER, reason="no scored ledger on disk")
@pytest.mark.skipif(not HAS_LEDGER, reason="no scored ledger on disk")
def test_attach_mainshock_keeps_other_mornings():
    from eqpred.ui.data import _attach_mainshock

    card = {
        "region": "global",
        "model_id": "fresh.mainshock.v1",
        "value": 0.11,
        "fresh_p": 0.11,
    }
    pack = {
        "week": [
            {
                "region": "global",
                "issue_day": "2026-09-16",
                "fresh_raw_p": 0.51,
                "fresh_raw_chance_plain": "51 in 100",
                "climatology_p": 0.70,
                "held_to_climatology": True,
                "sentence": "today's call",
                "why": [],
            }
        ]
    }
    old = _attach_mainshock(card, pack, issue_day="2024-08-15")
    assert old.get("fresh_p") == 0.11
    assert old.get("fresh_sentence") != "today's call"
    today = _attach_mainshock(card, pack, issue_day="2026-09-16")
    assert today.get("fresh_p") == 0.51
    assert today.get("fresh_raw_p") == 0.51
    assert today.get("climatology_p") == 0.70
    globe = _attach_mainshock(card, pack)
    assert globe.get("fresh_p") == 0.51
    assert globe.get("fresh_raw_p") == 0.51
    assert globe.get("climatology_p") == 0.70


def test_day_view_cards_are_dicts():
    from eqpred.ui.data import _joined_ledger

    cards = _joined_ledger()
    assert not cards.empty
    day = parse_utc(cards["issued_at"]).max().date().isoformat()
    pack = day_view(day)
    assert pack["cards"]
    assert all(isinstance(card, dict) and card.get("model_id") for card in pack["cards"])


def test_globe_view_is_issued_boxes_only():
    pack = globe_view()
    ids = {box["id"] for box in pack["boxes"]}
    assert ids == {"hikurangi", "alpine", "kermadec"}
    assert all(box.get("issued") for box in pack["boxes"])
    names = {box["name"] for box in pack["boxes"]}
    assert "japan_trench" not in names
    assert "chile_trench" not in names
    if pack.get("global"):
        assert pack["global"]["region_name"] == "global"
        assert pack["global"]["chance_plain"]
        assert pack["global"].get("fresh_p") is not None
        assert pack["global"].get("fresh_chance_plain")
        assert pack["global"].get("fresh_p") == pack["global"].get("fresh_raw_p")
        if pack["global"].get("held_to_climatology"):
            assert pack["global"].get("climatology_p") is not None
            assert "fresh-shock" in (pack["global"].get("fresh_sentence") or "")
        assert "Japan" not in pack["global"]["sentence"]
        assert "Japan" not in (pack["global"].get("fresh_sentence") or "")
    copy = pack.get("copy") or {}
    if copy.get("held"):
        assert "forecast" in copy.get("title", "")
        assert "typical year" in copy.get("lede", "")
    for card in pack.get("week") or []:
        assert card.get("region_name") not in {"japan", "japan_trench", "chile", "chile_trench"}
