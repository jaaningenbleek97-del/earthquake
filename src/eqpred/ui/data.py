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
from eqpred.eval.hunt import load_hunt, load_hunt_fresh
from eqpred.eval.foreshock import foreshock_week_copy, load_foreshock
from eqpred.eval.mainshock import leftover_week_copy, load_mainshock
from eqpred.eval.probe import load_probe
from eqpred.eval.promote import load_promote
from eqpred.eval.transfer import load_transfer
from eqpred.models.registry import load_current, load_index
from eqpred.models.gr import load_gr
from eqpred.models.leftover import lam0_from_etas_p, leftover_p, watch_list
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

MODEL_LABELS = {
    "fusion.etas_x_mod.v1": "This week's chance",
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
    "fusion.etas_x_mod.v1": "Chance from the aftershock model",
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
    "fusion_issued_m7": "What we issue: train-Brier calibrated ETAS only. Nowcast, TEC, Coulomb, cusp, wave2, and migration are fit for paper scores and stay off issued p — they lost to scale-only on holdout.",
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


def _trim_hunt(pack: dict | None, n: int = 40) -> dict | None:
    if not pack:
        return pack
    out = dict(pack)

    def _leftover_first(rows: list) -> list:
        leftover = [r for r in rows if r.get("family") not in {"catalog", "migration"}]
        leftover.sort(key=lambda r: abs(float(r.get("holdout_rho") or r.get("train_rho") or 0)), reverse=True)
        return leftover[:n]

    out["confirmed"] = _leftover_first(list(pack.get("confirmed") or []))
    out["loudest_train"] = _leftover_first(list(pack.get("loudest_train") or []))
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

    mainshock = load_mainshock(root)
    foreshock = load_foreshock(root)
    honesty = leftover_week_copy(mainshock)
    precursor = foreshock_week_copy(foreshock)
    manifesto = (
        "This lab tries to predict a fresh large earthquake — the first big shock, "
        "not the aftershocks that follow. It looks at as many leftover sources as "
        "we can timestamp (space weather, tides, ionosphere, GPS, and the rest), "
        "then backtests the call on later years. Aftershock rates are the thing "
        "we refuse to ship as the product."
    )
    return {
        "title": "Earthquake prediction lab",
        "what": (
            "Dated calls, scored against ComCat. Issued p is calibrated ETAS. "
            "Solar g is a diagnostic, not the number. Promote has not rewritten issued p."
        ),
        "what_plain": honesty["lede"] if honesty["held"] else manifesto,
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
        "watchdog": freshness_report(datetime.now(timezone.utc), root),
        "registry": _registry_snapshot(root),
        "heads": heads,
        "reliability": reliability_table(ledger) if not ledger.empty else [],
        "battery": battery,
        "hunt": _trim_hunt(load_hunt(root)),
        "hunt_fresh": _trim_hunt(load_hunt_fresh(root)),
        "backtest": load_backtest(root),
        "brain": load_brain(root),
        "mainshock": mainshock,
        "foreshock": foreshock,
        "foreshock_copy": precursor,
        "probe": load_probe(root),
        "transfer": load_transfer(root),
        "promote": load_promote(root),
        "strategies": _load_strategies(root),
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
        return {"day": day, "cards": [], "features": None, "quakes": []}
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
        window_end = issued + pd.Timedelta(days=20)
        near = quakes.loc[
            (quakes["observed_at"] >= issued)
            & (quakes["observed_at"] < window_end)
            & (quakes["magnitude"] >= 5.0)
        ]
        q_rows = _records(near[["observed_at", "magnitude", "lon", "lat", "place", "event_id"]])

    catalogs = load_migration_catalogs(root)
    migration = {rid: cluster_asof(catalogs[rid], rid, issued) for rid in BOX_IDS}

    mainshock = load_mainshock(root)
    cards = [
        _attach_leftover(card, feat_series, root)
        if card.get("model_id") == "fusion.etas_x_mod.v1" and feat_series is not None
        else card
        for card in _day_cards(day_cards)
    ]
    issue_days = {w.get("issue_day") for w in (mainshock or {}).get("week") or []}
    if day in issue_days:
        cards = [
            _attach_mainshock(card, mainshock)
            if card.get("model_id") == "fusion.etas_x_mod.v1"
            else card
            for card in cards
        ]
    return {
        "day": day,
        "cards": cards,
        "features": feat_row,
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
    days = int(horizon_days or 7)
    window = "week" if days == 7 else f"{days} days"
    if str(region_name or "") == "global":
        return f"a magnitude {mag} or larger earthquake anywhere on Earth in the next {window}"
    place = _place_plain(region_name)
    return f"a magnitude {mag} or larger earthquake {place} in the next {window}"


def _card_plain(rec: dict) -> dict:
    region_name = rec.get("region_name") or _region_key(rec.get("region"))
    model_id = rec.get("model_id")
    horizon = int(rec.get("horizon_days") or 7)
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
    rec = {
        **rec,
        "region_name": region_name,
        "place_short": _place_short(region_name),
        "place_plain": _place_plain(region_name),
        "head_plain": HEAD_PLAIN.get(model_id, MODEL_LABELS.get(model_id, model_id)),
        "question_plain": question,
        "chance_plain": _chance_copy(rec.get("value")),
        "etas_chance_plain": _chance_copy(rec.get("competitor_etas_p")),
        "sentence": sentence,
    }
    return rec


def _attach_leftover(card: dict, feat_row, root: Path | None = None) -> dict:
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
            float(card.get("horizon_days") or 7),
        )
        pack = leftover_p(
            lam0,
            feat_row,
            region_id,
            mag=float(mag),
            horizon_days=int(card.get("horizon_days") or 7),
            root=root,
        )
    except Exception:
        return card
    why = pack.get("why") or []
    why_txt = ", ".join(why) if why else "the leftover mix"
    question = card.get("question_plain") or "a large earthquake this week"
    leftover_chance = _chance_copy(pack["leftover_p"])
    card = {
        **card,
        "leftover_p": pack["leftover_p"],
        "leftover_chance_plain": leftover_chance,
        "leftover_gain": pack["leftover_gain"],
        "leftover_parts": pack["parts"],
        "leftover_why": why,
        "leftover_sentence": (
            f"An unused leftover mix would put the chance of {question} at {leftover_chance}, "
            f"using {why_txt}."
        ),
    }
    return card


def _attach_mainshock(card: dict, pack: dict | None) -> dict:
    """Our call is the leftover AI for a fresh big quake, not ETAS × leftover."""
    if not card or not pack:
        return card
    rid = _region_key(card.get("region")) or card.get("region_name") or ""
    if rid == "alpine_fault":
        rid = "alpine"
    if rid in {"", "global"}:
        rid = "global"
    match = next((row for row in pack.get("week") or [] if row.get("region") == rid), None)
    if not match:
        return _attach_last_week(card, pack)
    held = bool(match.get("held_to_climatology"))
    why = list(match.get("why") or [])
    leftover_raw = match.get("leftover_raw_p")
    leftover_raw_chance = match.get("leftover_raw_chance_plain") or _chance_copy(leftover_raw)
    card = {
        **card,
        "leftover_p": leftover_raw,
        "leftover_chance_plain": leftover_raw_chance,
        "leftover_raw_p": leftover_raw,
        "leftover_raw_chance_plain": leftover_raw_chance,
        "leftover_why": why,
        "held_to_climatology": held,
        "leftover_parts": [{"name": name, "label": name, "gain": None} for name in why],
        "leftover_sentence": match.get("sentence"),
        "climatology_p": match.get("climatology_p"),
        "climatology_chance_plain": match.get("climatology_chance_plain"),
        "mainshock": True,
    }
    return _attach_last_week(card, pack)


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
    leftover_raw = match.get("leftover_raw_p")
    leftover_raw_chance = match.get("leftover_raw_chance_plain") or _chance_copy(leftover_raw)
    return {
        **card,
        "last_week_p": leftover_raw,
        "last_week_chance_plain": leftover_raw_chance,
        "last_week_why": list(match.get("why") or []),
        "last_week_held": bool(match.get("held_to_climatology")),
        "last_week_sentence": match.get("sentence"),
        "last_week_climatology_p": match.get("climatology_p"),
        "last_week_climatology_chance_plain": match.get("climatology_chance_plain"),
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
    mainshock = load_mainshock(root)
    if latest is not None and not cards.empty:
        stamp = parse_utc(cards["issued_at"])
        day_cards = cards.loc[stamp.dt.floor("D") == latest]
        week_rows = [_card_plain(rec) for rec in _day_cards(day_cards)]
        feat_row = _feature_row(latest, root)
        if feat_row is not None:
            week_rows = [
                _attach_leftover(row, feat_row, root)
                if row.get("model_id") == "fusion.etas_x_mod.v1"
                else row
                for row in week_rows
            ]
        if mainshock:
            week_rows = [
                _attach_mainshock(row, mainshock)
                if row.get("model_id") == "fusion.etas_x_mod.v1"
                else row
                for row in week_rows
            ]

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
                "question_plain": _question_plain(region.get("name") or rid, mag, 7),
                "issued": True,
            }
        )

    end = None
    start = None
    if latest is not None:
        end = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start = (latest - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    quakes = map_quakes(root, start=start, end=end, minmag=6.0)
    feat_row = _feature_row(latest, root) if latest is not None else None
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
        "copy": leftover_week_copy(mainshock),
        "foreshock": {
            "week": (load_foreshock(root) or {}).get("week") or [],
            "copy": foreshock_week_copy(load_foreshock(root)),
            "note": (load_foreshock(root) or {}).get("note"),
            "n_features": (load_foreshock(root) or {}).get("n_features"),
            "n_beats_climatology": (load_foreshock(root) or {}).get("n_beats_climatology"),
        },
        "last_week": {
            "week": ((mainshock or {}).get("last_week") or {}).get("week") or [],
            "copy": leftover_week_copy((mainshock or {}).get("last_week"), mode="last_week")
            if (mainshock or {}).get("last_week")
            else None,
            "note": ((mainshock or {}).get("last_week") or {}).get("note"),
            "n_features": ((mainshock or {}).get("last_week") or {}).get("n_features"),
            "n_beats_climatology": ((mainshock or {}).get("last_week") or {}).get("n_beats_climatology"),
        },
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
