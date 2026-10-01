from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eqpred.eval.episodes import quiet_episodes
from eqpred.eval.score import reliability_table, summarize_ledger
from eqpred.features.migration import BOX_IDS, cluster_asof, load_migration_catalogs
from eqpred.ingest.run import try_latest_raw
from eqpred.ingest.watchdog import freshness_report
from eqpred.eval.backtest import load_backtest
from eqpred.eval.brain import load_brain
from eqpred.eval.aftershock import load_aftershock
from eqpred.eval.compat import saved_fresh
from eqpred.eval.csep import load_csep
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.hunt import load_hunt, load_hunt_fresh, load_hunt_sequence
from eqpred.eval.foreshock import foreshock_week_copy, load_foreshock
from eqpred.eval.mainshock import FRESH_MODEL_ID, ISSUED_TARGETS, fresh_week_copy, load_mainshock
from eqpred.eval.waiting import catalog_wait, odd_box
from eqpred.eval.probe import load_probe
from eqpred.eval.promote import load_promote
from eqpred.eval.transfer import load_transfer
from eqpred.models.registry import load_current, load_index
from eqpred.models.gr import load_gr
from eqpred.eval.earth_state import load_earth_state
from eqpred.eval.circus import load_circus
from eqpred.eval.leftover import load_leftover
from eqpred.eval.leftover_look import load_leftover_look
from eqpred.eval.eepas import load_eepas
from eqpred.eval.widenet import load_widenet
from eqpred.eval.das_foil import load_das_foil
from eqpred.models.garnish import garnish_p, lam0_from_etas_p, watch_list
from eqpred.models.predict import BOX_MAG, GLOBAL_MAG
from eqpred.regions import (
    REGIONS,
    in_any_box,
    is_issued_box,
    leaflet_rectangles,
    region_centroid_lat,
    region_centroid_lon,
)
from eqpred.paths import data_dir, feature_dir, model_dir, prediction_batches, prediction_dir
from eqpred.stamp import parse_utc, utc_ts

ISSUED_P_MODELS = frozenset({FRESH_MODEL_ID, "fusion.etas_x_mod.v1"})

MODEL_LABELS = {
    FRESH_MODEL_ID: "This week's chance",
    "fusion.etas_x_mod.v1": "This week's chance (older aftershock ledger)",
    "alarm.spf.v2": "Rare huge-quake alert",
    "alarm.spf.v1": "Rare huge-quake alert (older)",
    "alarm.quiet.v1": "Quiet-week M7 alert",
    "alarm.quiet_spf.v1": "Quiet-week huge-quake alert",
    "anti.quiet.v1": "Stays quiet",
    "human": "Human call",
}

PLACE_SHORT = {
    "global": "Earth-wide",
    "hikurangi": "Hikurangi",
    "alpine": "Alpine Fault",
    "alpine_fault": "Alpine Fault",
    "kermadec": "Kermadec",
    "japan_trench": "Japan (practice)",
    "chile_trench": "Chile (practice)",
}

PLACE_PLAIN = {
    "global": "anywhere on Earth",
    "hikurangi": "along the Hikurangi trench, east of New Zealand’s North Island",
    "alpine": "on the Alpine Fault, on New Zealand’s South Island",
    "alpine_fault": "on the Alpine Fault, on New Zealand’s South Island",
    "kermadec": "in the Kermadec trench, northeast of New Zealand",
    "japan_trench": "along the Japan trench (practice only)",
    "chile_trench": "along the Chile trench (practice only)",
}

HEAD_PLAIN = {
    FRESH_MODEL_ID: "Chance from the fresh-shock model",
    "fusion.etas_x_mod.v1": "Chance from the older aftershock ledger",
    "alarm.spf.v2": "A rare alert for a magnitude 8 or larger quake",
    "alarm.spf.v1": "A rare alert for a magnitude 8 or larger quake",
    "alarm.quiet.v1": "An alert for a magnitude 7+ quake in an unusually quiet week",
    "alarm.quiet_spf.v1": "An alert for a magnitude 8+ quake in an unusually quiet week",
    "anti.quiet.v1": "A call that this place stays quiet",
    "human": "A person wrote this call by hand",
}

BATTERY_BLURB = {
    "davidson_spf_m8_v1": "Old SPF windows. High time-in-alarm, so hits are cheap.",
    "davidson_spf_m8_v2": "Tightened SPF recipe, raw windows, no 40-day delay.",
    "davidson_spf_m8_v2_timeshift_180d": "v2 windows with solar series shifted 180 days. If this still 'works', it is leakage.",
    "davidson_spf_m8_v2_budgeted": "Window recipe we issue from: v2 + train budget + 40-day Wilcox delay. Issued cards also sit under a monthly alarm cap.",
    "quiet_spf_m8": "M8 only when ETAS is in the quiet tail and the SPF window is on.",
    "marchitelli_proton_lag1_m6": "Proton density the day before M6 vs background.",
    "marchitelli_proton_lag1_m6_declustered": "Same proton test after dropping aftershocks.",
    "love_thomas_indices_vs_m5": "Correlation of Kp / Dst / sunspots with daily M5 counts. Near zero is the skeptic null.",
    "fusion_issued_m7": "Lab foil: train-Brier calibrated ETAS. Not the issued product. Nowcast, TEC, Coulomb, cusp, wave2, and migration stay off the foil too — they lost to scale-only on holdout.",
    "fusion_nowcast_m7": "Calibrated ETAS × trailing b / energy nowcast. Residual catalog physics, not solar.",
    "fusion_modulator_m7": "Abandoned recipe: locked solar g × ETAS. Kept so we can see it still is not timing skill.",
    "fusion_timeshift_180d": "Abandoned g with solar drivers shifted 180 days. If this still beats ETAS, timing is not the skill.",
    "fusion_climatology_m7": "Constant p = train-era 7-day M≥7 rate. A mean-p move toward this rate can beat ETAS without ranking weeks.",
    "fusion_etas_world": "Refit g on a Poisson ETAS catalog with no solar physics, then score on real holdout quakes. If this beats ETAS, g is a rate knob.",
}


def _load_strategies(root: Path | None = None) -> list[dict]:
    path = data_dir(root) / "eval" / "strategies.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _registry_snapshot(root: Path | None = None) -> dict:
    current = load_current(root)
    index = load_index(root)
    if not current:
        return {"current": None, "n_versions": 0}
    recipe = current.get("feature_recipe") or {}
    return {
        "id": current.get("id"),
        "created_at": current.get("created_at"),
        "code_hash": current.get("code_hash"),
        "brain_hash": current.get("brain_hash"),
        "train_end": recipe.get("train_end"),
        "n_columns": len(recipe.get("columns") or []),
        "n_versions": len(index.get("entries") or []),
        "files": current.get("files") or [],
    }


def _load_cards(root: Path | None = None) -> pd.DataFrame:
    files = prediction_batches(root)
    rows: list[dict] = []
    for path in files:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["drivers"] = frame["drivers"].apply(lambda x: x if isinstance(x, list) else [])
    return frame


def _load_ledger(root: Path | None = None) -> pd.DataFrame:
    path = prediction_dir(root) / "ledger.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _joined_ledger(root: Path | None = None) -> pd.DataFrame:
    ledger = _load_ledger(root)
    cards = _load_cards(root)
    if ledger.empty:
        return cards
    if cards.empty:
        return ledger
    extra = cards[
        [
            c
            for c in (
                "prediction_id",
                "drivers",
                "rationale",
                "valid_from",
                "valid_to",
                "horizon_days",
                "region",
            )
            if c in cards.columns
        ]
    ]
    extra = extra.rename(columns={"region": "region_obj"})
    merged = ledger.merge(extra, on="prediction_id", how="left")
    if "region_obj" in merged.columns:
        merged["region_name"] = merged["region"]
        merged["region"] = merged["region_obj"].where(merged["region_obj"].notna(), merged["region"])
        merged = merged.drop(columns=["region_obj"])
    return merged


def _trim_hunt(pack: dict | None, n: int = 40, *, keep_catalog: bool = False) -> dict | None:
    if not pack:
        return pack
    out = dict(pack)

    def _first(rows: list) -> list:
        rows = list(rows or [])
        if not keep_catalog:
            rows = [r for r in rows if r.get("family") not in {"catalog", "migration"}]
        rows.sort(key=lambda r: abs(float(r.get("holdout_rho") or r.get("train_rho") or 0)), reverse=True)
        return rows[:n]

    out["confirmed"] = _first(list(pack.get("confirmed") or []))
    out["loudest_train"] = _first(list(pack.get("loudest_train") or []))
    honest = pack.get("honest")
    if isinstance(honest, dict):
        trimmed = dict(honest)
        for key in ("confirmed", "beats_climate", "families", "quiet", "m8", "loudest_train"):
            if key in trimmed:
                trimmed[key] = _first(list(trimmed.get(key) or []))
        out["honest"] = trimmed
    return out


def snapshot(root: Path | None = None) -> dict:
    ledger = _load_ledger(root)
    features_path = feature_dir(root) / "global_daily.parquet"
    features = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
    quakes = try_latest_raw("usgs.comcat", root)
    battery_path = data_dir(root) / "eval" / "battery.json"
    battery = json.loads(battery_path.read_text(encoding="utf-8")) if battery_path.exists() else []
    for row in battery:
        row["blurb"] = BATTERY_BLURB.get(row.get("name"), "")

    models = {}
    folder = model_dir(root)
    for name in (
        "etas.json",
        "modulator.json",
        "quiet.json",
        "gr.json",
        "calibration.json",
        "nowcast.json",
        "coulomb.json",
        "cusp.json",
        "wave2.json",
        "migration.json",
        "anti.json",
        "budget.json",
    ):
        path = folder / name
        if path.exists():
            models[name.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))

    feat_start = feat_end = None
    if not features.empty:
        stamp = parse_utc(features["issue_time"])
        feat_start = stamp.min().date().isoformat()
        feat_end = stamp.max().date().isoformat()

    q_start = q_end = None
    n_quakes = 0
    n_m8 = 0
    if not quakes.empty:
        n_quakes = int(len(quakes))
        n_m8 = int((quakes["magnitude"] >= 8).sum())
        q_stamp = parse_utc(quakes["observed_at"])
        q_start = q_stamp.min().date().isoformat()
        q_end = q_stamp.max().date().isoformat()

    summary = summarize_ledger(ledger) if not ledger.empty else {"n_cards": 0, "heads": []}
    heads = []
    for stats in summary.get("heads") or []:
        model_id = stats.get("model_id")
        label = MODEL_LABELS.get(model_id, model_id)
        region = stats.get("region")
        mag = stats.get("magnitude_min")
        if region:
            label = f"{label} · {_place_short(region)}"
        if mag is not None:
            label = f"{label} · M≥{mag:g}"
        caps = (models.get("budget") or {}).get("days_per_month") or {}
        if model_id in caps:
            stats = {**stats, "budget_days_per_month": int(caps[model_id])}
        heads.append({"label": label, **stats})

    mainshock = _with_catalog_wait(load_mainshock(root), root)
    foreshock = load_foreshock(root)
    honesty = fresh_week_copy(mainshock)
    precursor = foreshock_week_copy(foreshock)
    manifesto = (
        "This lab publishes a dated chance of a first large quake, not the aftershocks that follow. "
        "A typical five-year rate sits beside each call. The aftershock model is a second opinion."
    )
    return {
        "title": "Earthquake prediction lab",
        "what": (
            "Dated calls, scored against the earthquake catalog. "
            "The published chance is the first-large-quake model. "
            "A typical year sits beside it. The aftershock model is a second opinion."
        ),
        "what_plain": honesty.get("lede") or manifesto,
        "catalog": {
            "n_events": n_quakes,
            "n_m8": n_m8,
            "start": q_start,
            "end": q_end,
            "etas": "as-of available_at",
            "hits": "final bulletin",
        },
        "features": {
            "start": feat_start,
            "end": feat_end,
            "n_days": int(len(features)),
            "gnss": (
                "joined"
                if not features.empty
                and "gnss_east_mm" in features.columns
                and bool(features["gnss_east_mm"].notna().any())
                else "not joined"
            ),
            "tec": (
                "joined"
                if not features.empty
                and "tec_tecu" in features.columns
                and bool(features["tec_tecu"].notna().any())
                else "not joined"
            ),
            "tec_boxes": (
                "joined"
                if not features.empty
                and "tec_japan_tecu" in features.columns
                and bool(features["tec_japan_tecu"].notna().any())
                else "not joined"
            ),
            "era5": (
                "joined"
                if not features.empty
                and "era5_sp_pa" in features.columns
                and bool(features["era5_sp_pa"].notna().any())
                else "not joined"
            ),
            "intermagnet": (
                "joined"
                if not features.empty
                and "mag_h_nt" in features.columns
                and bool(features["mag_h_nt"].notna().any())
                else "not joined"
            ),
            "geonet": (
                "joined"
                if not features.empty
                and "n_geonet_m3_7d" in features.columns
                else "not joined"
            ),
            "migration": (
                "joined"
                if not features.empty
                and "mig_along_hikurangi" in features.columns
                else "not joined"
            ),
        },
        "train_end": "2018-12-31",
        "holdout_start": "2019-01-01",
        "n_cards": int(summary.get("n_cards") or 0),
        "n_scored": int(summary.get("n_scored") or 0),
        "n_open_windows": int(summary.get("n_open_windows") or 0),
        "n_prospective": int(summary.get("n_prospective") or 0),
        "n_retrospective": int(summary.get("n_retrospective") or 0),
        "prospective_epoch_start": ((mainshock or {}).get("evaluation") or {}).get(
            "prospective_epoch_start"
        ),
        "watchdog": freshness_report(datetime.now(timezone.utc), root),
        "registry": _registry_snapshot(root),
        "heads": heads,
        "reliability": reliability_table(ledger) if not ledger.empty else [],
        "battery": battery,
        "hunt": _trim_hunt(load_hunt(root)),
        "hunt_fresh": _trim_hunt(load_hunt_fresh(root)),
        "hunt_sequence": _trim_hunt(load_hunt_sequence(root), keep_catalog=True),
        "backtest": load_backtest(root),
        "brain": load_brain(root),
        "mainshock": mainshock,
        "aftershock": load_aftershock(root),
        "csep": load_csep(root),
        "foreshock": foreshock,
        "foreshock_copy": precursor,
        "probe": load_probe(root),
        "transfer": load_transfer(root),
        "promote": load_promote(root),
        "strategies": _load_strategies(root),
        "earth_state": load_earth_state(root),
        "leftover": load_leftover(root),
        "leftover_look": load_leftover_look(root),
        "eepas": load_eepas(root),
        "widenet": load_widenet(root),
        "das_foil": load_das_foil(root),
        "circus": load_circus(root),
        "episodes": quiet_episodes(ledger, quakes, _load_cards(root)) if not ledger.empty and not quakes.empty else [],
        "models": models,
        "regions": {
            rid: {**region, "magnitude_min": GLOBAL_MAG if rid == "global" else BOX_MAG.get(rid)}
            for rid, region in REGIONS.items()
        },
        "model_labels": MODEL_LABELS,
    }


def query_ledger(
    root: Path | None = None,
    *,
    model: str | None = None,
    region: str | None = None,
    kind: str | None = None,
    hit: str | None = None,
    limit: int = 150,
    offset: int = 0,
) -> dict:
    frame = _joined_ledger(root)
    if frame.empty:
        return {"total": 0, "rows": [], "offset": offset, "limit": limit}
    work = frame.copy()
    if model:
        work = work.loc[work["model_id"] == model]
    if region:
        if "region_name" in work.columns:
            names = work["region_name"].astype(str)
        else:
            names = work["region"].apply(lambda r: r.get("name") if isinstance(r, dict) else str(r))
        work = work.loc[names == region]
    if kind:
        work = work.loc[work["kind"] == kind]
    if hit in {"0", "1"}:
        work = work.loc[work["hit"] == int(hit)]
    elif hit == "open":
        if "window_closed" in work.columns:
            work = work.loc[work["window_closed"] == False]
        else:
            work = work.loc[work["hit"].isna()]
    elif hit == "near" and "near_miss" in work.columns:
        near = pd.to_numeric(work["near_miss"], errors="coerce").fillna(0.0)
        anti = work["model_id"].astype(str).str.startswith("anti.")
        close = (near > 0) & (near < 1)
        graded = work["window_closed"].fillna(True) if "window_closed" in work.columns else True
        work = work.loc[graded & (((~anti) & (work["hit"] == 0) & (near > 0)) | (anti & close))]
    work = work.sort_values("issued_at", ascending=False)
    total = int(len(work))
    page = work.iloc[offset : offset + limit]
    rows = []
    for rec in _records(page):
        region_val = rec.get("region")
        if isinstance(region_val, dict):
            rec["region_name"] = region_val.get("name") or region_val.get("kind")
        else:
            rec["region_name"] = rec.get("region_name") or region_val
        rec["label"] = MODEL_LABELS.get(rec.get("model_id"), rec.get("model_id"))
        rows.append(_card_plain(rec))
    return {"total": total, "offset": offset, "limit": limit, "rows": rows}


def day_view(day: str, root: Path | None = None) -> dict:
    issued = utc_ts(day)
    cards = _joined_ledger(root)
    if cards.empty:
        return {"day": day, "horizon_days": HORIZON_DAYS, "cards": [], "features": None, "quakes": []}
    stamp = parse_utc(cards["issued_at"]) if "issued_at" in cards.columns else pd.Series(dtype="datetime64[ns, UTC]")
    day_cards = cards.loc[stamp.dt.floor("D") == issued]
    features_path = feature_dir(root) / "global_daily.parquet"
    feat_row = None
    feat_series = None
    if features_path.exists():
        features = pd.read_parquet(features_path)
        features["issue_time"] = parse_utc(features["issue_time"])
        match = features.loc[features["issue_time"] == issued]
        if not match.empty:
            feat_series = match.iloc[0]
            feat_row = {k: _jsonable(v) for k, v in feat_series.to_dict().items()}

    quakes = try_latest_raw("usgs.comcat", root)
    q_rows: list[dict] = []
    if not quakes.empty:
        quakes = quakes.copy()
        quakes["observed_at"] = parse_utc(quakes["observed_at"])
        window_end = issued + pd.Timedelta(days=HORIZON_DAYS)
        near = quakes.loc[
            (quakes["observed_at"] >= issued)
            & (quakes["observed_at"] < window_end)
            & (quakes["magnitude"] >= 5.0)
        ]
        q_rows = _records(near[["observed_at", "magnitude", "lon", "lat", "place", "event_id"]])

    catalogs = load_migration_catalogs(root)
    migration = {rid: cluster_asof(catalogs[rid], rid, issued) for rid in BOX_IDS}

    mainshock = _with_catalog_wait(load_mainshock(root), root)
    cards = [
        _attach_mainshock(card, mainshock, issue_day=day)
        if card.get("model_id") in ISSUED_P_MODELS
        else card
        for card in _day_cards(day_cards)
    ]
    return {
        "day": day,
        "horizon_days": HORIZON_DAYS,
        "cards": cards,
        "features": feat_row,
        "watch": watch_list(feat_row),
        "quakes": q_rows,
        "migration": migration,
    }


def _day_cards(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in _records(frame.sort_values("model_id")):
        region_val = rec.get("region")
        rec["region_name"] = (
            region_val.get("name") if isinstance(region_val, dict) else rec.get("region_name") or region_val
        )
        rec["label"] = MODEL_LABELS.get(rec.get("model_id"), rec.get("model_id"))
        rows.append(_card_plain(rec))
    return rows


def feature_series(root: Path | None = None, start: str | None = None, end: str | None = None) -> dict:
    path = feature_dir(root) / "global_daily.parquet"
    if not path.exists():
        return {"days": []}
    frame = pd.read_parquet(path)
    frame["issue_time"] = parse_utc(frame["issue_time"])
    if start:
        frame = frame.loc[frame["issue_time"] >= utc_ts(start)]
    if end:
        frame = frame.loc[frame["issue_time"] < utc_ts(end)]
    keep = [
        "issue_time",
        "spf_significant_window",
        "proton_density_cm3",
        "proton_spike",
        "kp",
        "kp_high",
        "dst_nt",
        "dst_storm",
        "f107",
        "sunspot_number",
        "tidal_strain",
        "n_m6_7d",
        "n_m7_30d",
        "n_fresh_m45_7d",
        "n_fresh_m6_7d",
        "n_fresh_m7_30d",
        "days_since_fresh_m7",
        "coulomb_hikurangi",
        "coulomb_alpine",
        "coulomb_kermadec",
        "cusp_align_hikurangi",
        "cusp_align_alpine",
        "cusp_align_kermadec",
        "gnss_east_mm",
        "gnss_east_7d",
        "n_geonet_m3_7d",
        "tec_tecu",
        "tec_nz_tecu",
        "tec_hikurangi_tecu",
        "tec_alpine_tecu",
        "tec_kermadec_tecu",
        "tec_japan_tecu",
        "tec_chile_tecu",
        "era5_sp_pa",
        "era5_swvl1",
        "mag_h_nt",
        "mag_h_range_nt",
        "mag_dbdt_nt",
        "mag_station",
        "sunspot_north",
        "sunspot_south",
        "solar_eclipse_window",
        "coronal_hole_stream",
        "hss_on",
        "flare_x_on",
        "spe_on",
        "solar_descending",
        "polar_vortex_weak",
        "volcano_unrest",
        "dart_height_m",
        "dart_drop",
        "sse_on",
        "tremor_on",
        "well_drop",
        "scint_high",
        "haarp_on",
        "dvv_drop",
        "dvv_nz",
        "wspr_quiet",
        "wspr_snr_med",
        "wiki_hot",
        "wiki_eq_views",
        "cal_payday",
        "ulf_high",
        "tec_storm",
        "radon_high",
        "radon_bq_m3",
        "grace_wet",
        "grace_hikurangi_lwe_cm",
        "so2_high",
        "so2_kermadec_kgm2",
        "strain_pulse",
        "era5_wet",
        "olr_anomaly",
        "gnss_dilatation",
        "ssc_dst",
        "sunspot_hemisphere_skew",
        "earth_state_z",
        "earth_state_unusual",
        "graph_inflow_hikurangi",
        "graph_inflow_alpine",
        "graph_inflow_kermadec",
        "graph_transfer_on",
        "analog_p_global",
        "mig_along_hikurangi",
        "mig_along_alpine",
        "mig_along_kermadec",
        "mig_updip_hikurangi",
        "mig_updip_alpine",
        "mig_updip_kermadec",
        "mig_n_hikurangi",
        "mig_n_alpine",
        "mig_n_kermadec",
    ]
    cols = [c for c in keep if c in frame.columns]
    return {"days": _records(frame[cols])}


def _region_key(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("kind") or "")
    return str(value or "")


def _place_short(region_name: str) -> str:
    key = str(region_name or "")
    return PLACE_SHORT.get(key, key.replace("_", " ") or "this place")


def _place_plain(region_name: str) -> str:
    return PLACE_PLAIN.get(str(region_name or ""), str(region_name or "this region"))


def _chance_copy(value) -> str | None:
    if value is None:
        return None
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p):
        return None
    n = int(round(max(0.0, min(1.0, p)) * 100))
    if n <= 0:
        return "less than 1 in 100"
    if n >= 100:
        return "about certain this week"
    return f"{n} in 100"


def _question_plain(region_name: str, magnitude_min, horizon_days: int) -> str:
    mag = f"{float(magnitude_min):g}" if magnitude_min is not None else "?"
    days = int(horizon_days or HORIZON_DAYS)
    window = "week" if days == 7 else f"{days} days"
    if str(region_name or "") == "global":
        return f"a magnitude {mag} or larger earthquake anywhere on Earth in the next {window}"
    place = _place_plain(region_name)
    return f"a magnitude {mag} or larger earthquake {place} in the next {window}"


def _card_plain(rec: dict) -> dict:
    region_name = rec.get("region_name") or _region_key(rec.get("region"))
    model_id = rec.get("model_id")
    horizon = int(rec.get("horizon_days") or HORIZON_DAYS)
    mag = rec.get("magnitude_min")
    kind = rec.get("kind")
    anti = str(model_id or "").startswith("anti.")
    question = _question_plain(region_name, mag, horizon)
    if anti:
        sentence = f"We think {_place_plain(region_name)} stays quiet this week."
    elif kind == "alarm":
        sentence = f"Alert: watch for {question}."
    else:
        chance = _chance_copy(rec.get("value")) or "an unknown chance"
        sentence = f"We put the chance of {question} at {chance}."
    climate = rec.get("competitor_climatology_p")
    if climate is None:
        climate = rec.get("climatology_p")
    rec = {
        **rec,
        "region_name": region_name,
        "place_short": _place_short(region_name),
        "place_plain": _place_plain(region_name),
        "head_plain": HEAD_PLAIN.get(model_id, MODEL_LABELS.get(model_id, model_id)),
        "question_plain": question,
        "chance_plain": _chance_copy(rec.get("value")),
        "etas_chance_plain": _chance_copy(rec.get("competitor_etas_p")),
        "climatology_p": climate,
        "climatology_chance_plain": rec.get("climatology_chance_plain") or _chance_copy(climate),
        "sentence": sentence,
    }
    return rec


def _attach_garnish(card: dict, feat_row, root: Path | None = None) -> dict:
    if not card or feat_row is None:
        return card
    etas_p = card.get("competitor_etas_p")
    mag = card.get("magnitude_min")
    if etas_p is None or mag is None:
        return card
    gr = load_gr(root)
    region_id = _region_key(card.get("region")) or card.get("region_name") or "global"
    if region_id == "alpine_fault":
        region_id = "alpine"
    try:
        lam0 = lam0_from_etas_p(
            etas_p,
            float(mag),
            float(gr.get("m0") or 4.5),
            float(gr.get("b") or 1.0),
            float(card.get("horizon_days") or HORIZON_DAYS),
        )
        pack = garnish_p(
            lam0,
            feat_row,
            region_id,
            mag=float(mag),
            horizon_days=int(card.get("horizon_days") or HORIZON_DAYS),
            root=root,
        )
    except Exception:
        return card
    why = pack.get("why") or []
    why_txt = ", ".join(why) if why else "the garnish mix"
    question = card.get("question_plain") or "a large earthquake this week"
    garnish_chance = _chance_copy(pack["garnish_p"])
    card = {
        **card,
        "garnish_p": pack["garnish_p"],
        "garnish_chance_plain": garnish_chance,
        "garnish_gain": pack["garnish_gain"],
        "garnish_parts": pack["parts"],
        "garnish_why": why,
        "garnish_sentence": (
            f"An unused garnish mix would put the chance of {question} at {garnish_chance}, "
            f"using {why_txt}."
        ),
    }
    return card


def _attach_mainshock(card: dict, pack: dict | None, *, issue_day: str | None = None) -> dict:
    """Hero number is the fresh-shock model. Climatology sits beside it."""
    if not card:
        return card
    rid = _region_key(card.get("region")) or card.get("region_name") or ""
    if rid == "alpine_fault":
        rid = "alpine"
    if rid in {"", "global"}:
        rid = "global"
    match = next((row for row in (pack or {}).get("week") or [] if row.get("region") == rid), None)
    if issue_day and match:
        tw_day = str(match.get("issue_day") or "")[:10]
        viewed = str(issue_day)[:10]
        if tw_day and viewed and tw_day != viewed:
            return card
    if not match:
        # No mainshock row for this box. Fall back to the card's own issued p —
        # never to garnish_p, which is the ETAS foil and not a fresh-shock call.
        value = card.get("value")
        climate = card.get("climatology_p")
        if climate is None:
            climate = card.get("competitor_climatology_p")
        card = {
            **card,
            "fresh_p": value,
            "fresh_chance_plain": card.get("fresh_chance_plain") or _chance_copy(value),
            "fresh_raw_p": value,
            "fresh_raw_chance_plain": card.get("fresh_raw_chance_plain") or _chance_copy(value),
            "climatology_p": climate,
            "climatology_chance_plain": card.get("climatology_chance_plain") or _chance_copy(climate),
            "mainshock": card.get("model_id") in ISSUED_P_MODELS,
        }
        return _attach_sidecars(card, pack)
    held = bool(match.get("held_to_climatology"))
    why = list(match.get("why") or [])
    fresh_raw = saved_fresh(match, "fresh_raw_p")
    fresh_raw_chance = saved_fresh(match, "fresh_raw_chance_plain") or _chance_copy(fresh_raw)
    climate = match.get("climatology_p")
    if climate is None:
        climate = card.get("competitor_climatology_p")
    hero = fresh_raw if fresh_raw is not None else match.get("p")
    hero_chance = match.get("fresh_raw_chance_plain") or match.get("chance_plain") or _chance_copy(hero)
    card = {
        **card,
        "fresh_p": hero,
        "fresh_chance_plain": hero_chance,
        "fresh_raw_p": fresh_raw,
        "fresh_raw_chance_plain": fresh_raw_chance,
        "fresh_why": why,
        "held_to_climatology": held,
        "fresh_parts": [{"name": name, "label": name, "gain": None} for name in why],
        "fresh_sentence": match.get("sentence"),
        "week_state": match.get("week_state") or ("rate" if held else "beat"),
        "traffic_light": match.get("traffic_light"),
        "traffic_sentence": match.get("traffic_sentence"),
        "climatology_p": climate,
        "climatology_chance_plain": match.get("climatology_chance_plain") or _chance_copy(climate),
        "wait_days": match.get("wait_days"),
        "wait_since_days": match.get("wait_since_days"),
        "wait_plain": match.get("wait_plain"),
        "neighborhood_plain": match.get("neighborhood_plain"),
        "mainshock": True,
    }
    return _attach_sidecars(card, pack)


def _attach_sidecars(card: dict, pack: dict | None) -> dict:
    return _attach_alpine_sequence(_attach_shrunk(_attach_last_week(card, pack), pack), pack)


def _annotate_explain(pack: dict, root: Path | None) -> dict:
    """Add the neighborhood, agreement, and three-story lines. Chances stay as saved."""
    from eqpred.eval.explain import read_explain

    explain = read_explain(root) or {}
    out = dict(pack)
    out["story"] = explain.get("story")
    out["agreement_exam"] = explain.get("agreement_exam")
    neighborhoods = explain.get("neighborhoods") or {}
    week = []
    for row in pack.get("week") or []:
        copied = dict(row)
        extra = neighborhoods.get(str(copied.get("region") or ""))
        if extra and extra.get("plain"):
            copied["neighborhood_plain"] = extra["plain"]
        week.append(copied)
    out["week"] = week
    agreements = explain.get("agreements") or {}
    short = []
    for row in pack.get("short_horizons") or []:
        copied = dict(row)
        if int(copied.get("horizon_days") or 0) == 3:
            extra = agreements.get(str(copied.get("region") or ""))
            if extra and extra.get("plain"):
                copied["agreement_plain"] = extra["plain"]
        short.append(copied)
    if pack.get("short_horizons") is not None:
        out["short_horizons"] = short
    return out


def _with_catalog_wait(pack: dict | None, root: Path | None = None) -> dict | None:
    """Add the catalog waiting time onto the 30-day rows. The chance itself is unchanged."""
    if not pack:
        return pack
    from eqpred.catalog import load_comcat
    from eqpred.eval.vote_freeze import score_frozen_ledger

    week = [dict(row) for row in pack.get("week") or []]
    issue = next(
        (
            row.get("issue_day")
            for row in week
            if row.get("issue_day") and int(row.get("horizon_days") or 30) == 30
        ),
        None,
    )
    if issue is None:
        pack = dict(pack)
        pack["frozen_score"] = score_frozen_ledger(load_comcat(root), root)
        return _annotate_explain(pack, root)
    quakes = load_comcat(root)
    when = utc_ts(str(issue)[:10])
    waits = {
        rid: catalog_wait(quakes, REGIONS[rid], mag, when)
        for rid, mag in ISSUED_TARGETS
    }
    for row in week:
        if int(row.get("horizon_days") or 30) != 30:
            continue
        extra = waits.get(str(row.get("region") or ""))
        if extra:
            row.update(extra)
    out = dict(pack)
    out["week"] = week
    out["odd_box"] = odd_box(quakes, when)
    out["frozen_score"] = score_frozen_ledger(quakes, root)
    return _annotate_explain(out, root)


def _attach_last_week(card: dict, pack: dict | None) -> dict:
    if not card or not pack:
        return card
    lw = pack.get("last_week") or {}
    rid = _region_key(card.get("region")) or card.get("region_name") or ""
    if rid == "alpine_fault":
        rid = "alpine"
    if rid in {"", "global"}:
        rid = "global"
    match = next((row for row in lw.get("week") or [] if row.get("region") == rid), None)
    if not match:
        return card
    held = bool(match.get("held_to_climatology"))
    fresh_raw = saved_fresh(match, "fresh_raw_p")
    fresh_raw_chance = saved_fresh(match, "fresh_raw_chance_plain") or _chance_copy(fresh_raw)
    hero = fresh_raw if fresh_raw is not None else match.get("p")
    hero_chance = match.get("fresh_raw_chance_plain") or match.get("chance_plain") or _chance_copy(hero)
    return {
        **card,
        "last_week_p": hero,
        "last_week_chance_plain": hero_chance,
        "last_week_raw_p": fresh_raw,
        "last_week_raw_chance_plain": fresh_raw_chance,
        "last_week_why": list(match.get("why") or []),
        "last_week_held": held,
        "last_week_sentence": match.get("sentence"),
        "last_week_climatology_p": match.get("climatology_p"),
        "last_week_climatology_chance_plain": match.get("climatology_chance_plain"),
    }


def _attach_shrunk(card: dict, pack: dict | None) -> dict:
    if not card or not pack:
        return card
    sh = pack.get("shrunk") or {}
    rid = _region_key(card.get("region")) or card.get("region_name") or ""
    if rid == "alpine_fault":
        rid = "alpine"
    if rid in {"", "global"}:
        rid = "global"
    match = next((row for row in sh.get("week") or [] if row.get("region") == rid), None)
    if not match:
        return card
    raw = saved_fresh(match, "fresh_raw_p")
    return {
        **card,
        "shrunk_p": raw if raw is not None else match.get("p"),
        "shrunk_chance_plain": match.get("fresh_raw_chance_plain") or match.get("chance_plain") or _chance_copy(raw),
        "shrunk_held": bool(match.get("held_to_climatology")),
        "shrunk_sentence": match.get("sentence"),
        "shrunk_climatology_chance_plain": match.get("climatology_chance_plain"),
        "shrunk_why": list(match.get("why") or []),
    }


def _attach_alpine_sequence(card: dict, pack: dict | None) -> dict:
    if not card or not pack:
        return card
    al = pack.get("alpine_sequence") or {}
    rid = _region_key(card.get("region")) or card.get("region_name") or ""
    if rid == "alpine_fault":
        rid = "alpine"
    if rid != "alpine":
        return {**card, "alpine_sequence_applied": bool(al.get("applied"))}
    match = next((row for row in al.get("week") or [] if row.get("region") == "alpine"), None)
    if not match:
        return {**card, "alpine_sequence_applied": bool(al.get("applied"))}
    raw = saved_fresh(match, "fresh_raw_p")
    return {
        **card,
        "alpine_sequence_p": raw if raw is not None else match.get("p"),
        "alpine_sequence_chance_plain": match.get("fresh_raw_chance_plain") or match.get("chance_plain") or _chance_copy(raw),
        "alpine_sequence_held": bool(match.get("held_to_climatology")),
        "alpine_sequence_sentence": match.get("sentence"),
        "alpine_sequence_applied": bool(al.get("applied")),
        "alpine_sequence_climatology_chance_plain": match.get("climatology_chance_plain"),
    }


def _latest_issue_day(frame: pd.DataFrame):
    if frame.empty or "issued_at" not in frame.columns:
        return None
    stamp = parse_utc(frame["issued_at"])
    if stamp.empty:
        return None
    return stamp.max().floor("D")


def _feature_row(day, root: Path | None = None):
    path = feature_dir(root) / "global_daily.parquet"
    if day is None or not path.exists():
        return None
    features = pd.read_parquet(path)
    features["issue_time"] = parse_utc(features["issue_time"])
    issued = day if hasattr(day, "tzinfo") else utc_ts(str(day)[:10])
    match = features.loc[features["issue_time"] == issued]
    if match.empty:
        match = features.loc[features["issue_time"].dt.floor("D") == pd.Timestamp(issued).floor("D")]
    if match.empty:
        return None
    return match.iloc[0]


def globe_view(root: Path | None = None) -> dict:
    """This week's issued calls plus recent quakes for the 3D globe. Transfer boxes stay off."""
    cards = _joined_ledger(root)
    latest = _latest_issue_day(cards)
    week_rows: list[dict] = []
    mainshock = _with_catalog_wait(load_mainshock(root), root)
    if latest is not None and not cards.empty:
        stamp = parse_utc(cards["issued_at"])
        day_cards = cards.loc[stamp.dt.floor("D") == latest]
        week_rows = [_card_plain(rec) for rec in _day_cards(day_cards)]
        week_rows = [
            _attach_mainshock(row, mainshock)
            if row.get("model_id") in ISSUED_P_MODELS
            else row
            for row in week_rows
        ]

    fusion = [c for c in week_rows if c.get("model_id") == FRESH_MODEL_ID]
    if not fusion:
        fusion = [c for c in week_rows if c.get("model_id") == "fusion.etas_x_mod.v1"]
    global_card = next((c for c in fusion if _region_key(c.get("region")) in {"global", ""} and c.get("region_name") == "global"), None)
    if global_card is None:
        global_card = next((c for c in fusion if c.get("region_name") == "global"), None)

    boxes = []
    for rid, region in REGIONS.items():
        if not is_issued_box(region):
            continue
        mag = BOX_MAG.get(rid)
        match = next(
            (c for c in fusion if c.get("region_name") in {region.get("name"), rid, region.get("label")}),
            None,
        )
        anti = next(
            (
                c
                for c in week_rows
                if str(c.get("model_id") or "").startswith("anti.")
                and c.get("region_name") in {region.get("name"), rid, region.get("label")}
            ),
            None,
        )
        boxes.append(
            {
                "id": rid,
                "name": region.get("name"),
                "label": region.get("label") or rid,
                "place_short": _place_short(region.get("name") or rid),
                "place_plain": _place_plain(region.get("name") or rid),
                "magnitude_min": mag,
                "lat": region_centroid_lat(region),
                "lon": region_centroid_lon(region),
                "polygons": leaflet_rectangles(region),
                "card": match,
                "anti": anti,
                "question_plain": _question_plain(region.get("name") or rid, mag, HORIZON_DAYS),
                "issued": True,
            }
        )

    end = None
    start = None
    if latest is not None:
        end = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start = (latest - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    quakes = map_quakes(root, start=start, end=end, minmag=6.0)
    feat_day = latest
    if feat_day is None:
        path = feature_dir(root) / "global_daily.parquet"
        if path.exists():
            stamp = parse_utc(pd.read_parquet(path, columns=["issue_time"])["issue_time"])
            if not stamp.empty:
                feat_day = stamp.max().floor("D")
    feat_row = _feature_row(feat_day, root) if feat_day is not None else None
    feat_dict = None if feat_row is None else {k: _jsonable(v) for k, v in feat_row.to_dict().items()}

    return {
        "issue_day": None if latest is None else latest.date().isoformat(),
        "global": global_card,
        "boxes": boxes,
        "week": week_rows,
        "quakes": quakes.get("quakes") or [],
        "quake_range": {"start": start, "end": end},
        "watch": watch_list(feat_dict),
        "features": feat_dict,
        "copy": fresh_week_copy(mainshock),
        "foreshock": {
            "week": (load_foreshock(root) or {}).get("week") or [],
            "copy": foreshock_week_copy(load_foreshock(root)),
            "note": (load_foreshock(root) or {}).get("note"),
            "n_features": (load_foreshock(root) or {}).get("n_features"),
            "n_beats_climatology": (load_foreshock(root) or {}).get("n_beats_climatology"),
        },
        "last_week": {
            "week": ((mainshock or {}).get("last_week") or {}).get("week") or [],
            "copy": fresh_week_copy((mainshock or {}).get("last_week"), mode="last_week")
            if (mainshock or {}).get("last_week")
            else None,
            "note": ((mainshock or {}).get("last_week") or {}).get("note"),
            "n_features": ((mainshock or {}).get("last_week") or {}).get("n_features"),
            "n_beats_climatology": ((mainshock or {}).get("last_week") or {}).get("n_beats_climatology"),
        },
        "short_horizons": (mainshock or {}).get("short_horizons") or [],
        "story": (mainshock or {}).get("story"),
        "agreement_exam": (mainshock or {}).get("agreement_exam"),
        "draft": (mainshock or {}).get("draft"),
        "vote_draft": (mainshock or {}).get("vote_draft"),
        "frozen_score": (mainshock or {}).get("frozen_score"),
        "frozen": (mainshock or {}).get("frozen"),
        "odd_box": (mainshock or {}).get("odd_box"),
        "shrunk": (mainshock or {}).get("shrunk"),
        "locked_thin": (mainshock or {}).get("locked_thin"),
        "alpine_sequence": (mainshock or {}).get("alpine_sequence"),
        "method_eval": (mainshock or {}).get("method_eval"),
        "kermadec_attribution": (mainshock or {}).get("kermadec_attribution"),
        "leftover": (mainshock or {}).get("leftover") or load_leftover(root),
        "leftover_look": (mainshock or {}).get("leftover_look") or load_leftover_look(root),
        "circus": (mainshock or {}).get("circus") or load_circus(root),
        "experiments": (mainshock or {}).get("experiments") or [],
        "evaluation": (mainshock or {}).get("evaluation"),
    }


def map_quakes(
    root: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    minmag: float = 6.0,
) -> dict:
    quakes = try_latest_raw("usgs.comcat", root)
    if quakes.empty:
        return {"quakes": []}
    frame = quakes.copy()
    frame["observed_at"] = parse_utc(frame["observed_at"])
    if start:
        frame = frame.loc[frame["observed_at"] >= utc_ts(start)]
    if end:
        frame = frame.loc[frame["observed_at"] < utc_ts(end)]
    global_big = frame.loc[frame["magnitude"] >= minmag]
    local = frame.loc[(frame["magnitude"] >= 5.0) & in_any_box(frame)]
    combined = (
        pd.concat([global_big, local], ignore_index=True)
        .drop_duplicates("event_id")
        .sort_values("observed_at")
    )
    cols = [c for c in ("event_id", "observed_at", "magnitude", "lon", "lat", "place") if c in combined.columns]
    drawn = []
    for rid, region in REGIONS.items():
        if not is_issued_box(region):
            continue
        for rect in leaflet_rectangles(region):
            drawn.append({**rect, "magnitude_min": BOX_MAG.get(rid)})
    return {"quakes": _records(combined[cols]), "regions": REGIONS, "boxes": drawn}
