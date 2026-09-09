from pathlib import Path

import pytest

from eqpred.paths import prediction_dir, repo_root
from eqpred.ui.data import feature_series, globe_view, query_ledger, snapshot

HAS_LEDGER = (prediction_dir() / "ledger.parquet").exists()


@pytest.mark.skipif(not HAS_LEDGER, reason="no scored ledger on disk")
def test_snapshot_has_lab_fields():
    snap = snapshot()
    assert snap["n_cards"] > 0
    assert snap.get("n_scored", snap["n_cards"]) <= snap["n_cards"]
    assert snap.get("n_open_windows", 0) == snap["n_cards"] - snap.get("n_scored", snap["n_cards"])
    assert snap["catalog"]["n_events"] > 0
    assert "Promote has not rewritten issued p" in snap["what"]
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
        (r for r in snap["reliability"] if r.get("region") == "global" and "fusion" in str(r.get("model_id"))),
        None,
    )
    assert global_rel and len(global_rel.get("bins") or []) >= 3
    hik = next(
        (h for h in snap["heads"] if h.get("region") == "hikurangi" and h.get("model_id") == "fusion.etas_x_mod.v1"),
        None,
    )
    assert hik and hik.get("p_hit_corr") is not None
    assert hik.get("mean_near_miss") is not None
    assert hik["mean_near_miss"] >= hik["hit_rate"]
    assert snap["regions"]["hikurangi"]["kind"] == "bbox"
    assert snap["regions"]["kermadec"]["min_lon"] > snap["regions"]["kermadec"]["max_lon"]
    assert any(
        h["model_id"] == "fusion.etas_x_mod.v1"
        and h.get("region") == "global"
        and h.get("magnitude_min") == 7.0
        for h in snap["heads"]
    )
    assert any(
        h["model_id"] == "fusion.etas_x_mod.v1"
        and h.get("region") == "kermadec"
        and h.get("magnitude_min") == 6.0
        for h in snap["heads"]
    )
    assert snap["models"].get("budget", {}).get("issued") == "monthly_cap_walk_forward"
    assert snap["models"]["budget"]["days_per_month"]["alarm.spf.v2"] == 6
    spf = next((h for h in snap["heads"] if h.get("model_id") == "alarm.spf.v2"), None)
    assert spf and spf.get("budget_days_per_month") == 6
    assert snap["episodes"]
    assert all(ep.get("verdict") for ep in snap["episodes"])
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
        assert anti_heads
        assert all(h.get("quiet_base_rate") is not None for h in anti_heads)
        assert all(h.get("budget_days_per_month") == 4 for h in anti_heads)


@pytest.mark.skipif(not HAS_LEDGER, reason="no scored ledger on disk")
def test_ledger_fusion_drivers_are_structured():
    page = query_ledger(model="fusion.etas_x_mod.v1", limit=8)
    assert page["rows"]
    for row in page["rows"]:
        assert row["drivers"]
        first = row["drivers"][0]
        assert isinstance(first, dict)
        assert {"driver", "lag_hours", "contribution"} <= set(first)
        names = [d["driver"] if isinstance(d, dict) else d for d in row["drivers"]]
        assert "etas_calibrated" in names
        assert "spf_significant_window" not in names
        assert "proton_spike" not in names


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
        assert pack["global"].get("leftover_p") is not None
        assert pack["global"].get("leftover_chance_plain")
        assert pack["global"].get("leftover_p") == pack["global"].get("leftover_raw_p")
        if pack["global"].get("held_to_climatology"):
            assert "attempt" in (pack["global"].get("leftover_sentence") or "")
        assert "Japan" not in pack["global"]["sentence"]
        assert "Japan" not in (pack["global"].get("leftover_sentence") or "")
    copy = pack.get("copy") or {}
    if copy.get("held"):
        assert "leftover attempt" in copy.get("title", "")
        assert "historical rate" in copy.get("lede", "")
    for card in pack.get("week") or []:
        assert card.get("region_name") not in {"japan", "japan_trench", "chile", "chile_trench"}
