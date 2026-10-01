from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat
from eqpred.ingest.watchdog import ensure_can_issue
from eqpred.models.etas import EtasParams, daily_background_and_cluster, params_from_dict
from eqpred.models.calibrate import issued_scale
from eqpred.models.attribution import alarm_attrs, fresh_attrs
from eqpred.models.budget import AlertBudget, load_budget
from eqpred.models.registry import ensure_registered, stamp_cards, write_batch_meta
from eqpred.models.gr import rate_to_p
from eqpred.models.modulator import Modulator, modulator_from_dict
from eqpred.paths import feature_dir, model_dir, prediction_batches, prediction_dir
from eqpred.eval.honest import classify_card_kind
from eqpred.records import prediction
from eqpred.regions import REGIONS, in_bbox, is_issued_box, region_centroid_lon
from eqpred.stamp import parse_utc, utc_ts

# Global M≥6 / 7d hits on ~90% of days. Fusion and the quiet alarm issue M≥7.
GLOBAL_MAG = 7.0
NZ_MAG = 5.0
KERMADEC_MAG = 6.0
QUIET_MAG = 7.0
BOX_MAG = {
    "hikurangi": NZ_MAG,
    "alpine": NZ_MAG,
    "kermadec": KERMADEC_MAG,
    "japan": KERMADEC_MAG,
    "chile": KERMADEC_MAG,
}


def load_models(root: Path | None = None) -> tuple[EtasParams, Modulator]:
    folder = model_dir(root)
    etas = params_from_dict(json.loads((folder / "etas.json").read_text(encoding="utf-8")))
    mod = modulator_from_dict(json.loads((folder / "modulator.json").read_text(encoding="utf-8")))
    return etas, mod


def save_models(etas: EtasParams, mod: Modulator, root: Path | None = None) -> None:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "etas.json").write_text(json.dumps(etas.__dict__, indent=2), encoding="utf-8")
    from eqpred.models.modulator import modulator_to_dict

    (folder / "modulator.json").write_text(
        json.dumps(modulator_to_dict(mod), indent=2), encoding="utf-8"
    )


def _in_region(quakes: pd.DataFrame, region: dict) -> pd.Series:
    return in_bbox(quakes, region)


BOX_TRIGGER_KM = 250.0
BOX_TRIGGER_LENGTH_KM = 120.0


def _haversine_km(lat1, lon1, lat2: float, lon2: float):
    earth = 6371.0
    p1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - pd.to_numeric(lat1, errors="coerce"))
    dlmb = np.radians(lon2 - pd.to_numeric(lon1, errors="coerce"))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * earth * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _region_etas(quakes: pd.DataFrame, etas: EtasParams, region: dict) -> tuple[pd.DataFrame, EtasParams]:
    hist = quakes.loc[quakes["magnitude"] >= etas.m0].copy()
    hist["observed_at"] = parse_utc(hist["observed_at"])
    if region.get("kind") != "bbox":
        return hist, etas
    inside = _in_region(hist, region)
    local = hist.loc[inside]
    frac = max(len(local) / max(len(hist), 1), 1e-4)
    params = EtasParams(
        mu=etas.mu * frac,
        k=etas.k,
        alpha=etas.alpha,
        c=etas.c,
        p=etas.p,
        m0=etas.m0,
    )
    if local.empty or "lat" not in hist.columns or "lon" not in hist.columns:
        return local, params
    lat0 = 0.5 * (float(region["min_lat"]) + float(region["max_lat"]))
    lon0 = region_centroid_lon(region)
    dist = _haversine_km(hist["lat"], hist["lon"], lat0, lon0)
    near = inside | (dist <= BOX_TRIGGER_KM)
    out = hist.loc[near].copy()
    w = np.where(inside.loc[out.index], 1.0, np.exp(-dist.loc[out.index] / BOX_TRIGGER_LENGTH_KM))
    out["_etas_w"] = w
    return out, params


def issued_fusion_p(
    region_id: str,
    feat_row: pd.Series,
    lam0: float,
    *,
    cal: dict,
    nowcast_on: bool,
    nowcast_scale: float,
    g_now_global: float,
    coulomb: dict,
    cusp: dict,
    wave2: dict,
    migration: dict,
    mag: float,
    m0: float,
    b: float,
    horizon_days: int,
) -> tuple[float, float]:
    """Issued fusion p and raw ETAS p for one morning / box. No cards."""
    scale = issued_scale(cal, "global" if region_id == "global" else region_id, horizon_days)
    _ = (feat_row, nowcast_on, nowcast_scale, g_now_global, coulomb, cusp, wave2, migration)
    value = rate_to_p(lam0 * scale, mag, m0, b, horizon_days)
    etas_p = rate_to_p(lam0, mag, m0, b, horizon_days)
    return float(value), float(etas_p)


def expected_counts(
    features: pd.DataFrame,
    quakes: pd.DataFrame,
    etas: EtasParams,
    mod: Modulator,
    versions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    days = pd.DatetimeIndex(parse_utc(features["issue_time"]))
    hist = quakes.loc[quakes["magnitude"] >= etas.m0].copy()
    hist["observed_at"] = parse_utc(hist["observed_at"])
    lam = daily_background_and_cluster(hist, days, etas, versions=versions)
    gain = mod.gain(features)
    out = features[["issue_time"]].copy()
    out["lambda_etas"] = lam
    out["gain"] = gain
    out["lambda_mod"] = lam * gain
    return out


def load_batch_cards(dest: Path) -> list[dict]:
    rows: list[dict] = []
    if not dest.exists():
        return rows
    for path in prediction_batches(dest=dest):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)
    return rows


def _issued_day(card: dict) -> pd.Timestamp:
    return utc_ts(card["issued_at"]).tz_convert("UTC").normalize()


def replace_window_batches(dest: Path, issue_days: set[pd.Timestamp]) -> int:
    """Drop cards on the mornings we are about to rewrite. Keep the rest."""
    dest.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in prediction_batches(dest=dest):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cards = payload if isinstance(payload, list) else []
        kept = []
        for card in cards:
            if _issued_day(card) in issue_days:
                removed += 1
            else:
                kept.append(card)
        if kept:
            path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
        else:
            path.unlink()
    return removed


def begin_issue(
    rates: pd.DataFrame,
    window: pd.Series,
    root: Path | None,
    dest: Path,
) -> pd.Series:
    """Gate on source freshness, then replace only the requested issue mornings.

    Stale or empty windows raise before any batch file is rewritten. Cards
    outside the window stay on disk so a 2026 catch-up cannot wipe the holdout.
    """
    if not bool(window.any()):
        raise RuntimeError("no feature days in the predict window; existing cards left on disk")
    unique = pd.DatetimeIndex(pd.unique(parse_utc(rates.loc[window, "issue_time"])))
    kept = ensure_can_issue(unique, root)
    if len(kept) < len(unique):
        print(
            f"watchdog skipped {len(unique) - len(kept)} issue morning(s) "
            "with stale required sources"
        )
    kept_days = {pd.Timestamp(ts).tz_convert("UTC").normalize() for ts in kept}
    issue_days = parse_utc(rates["issue_time"]).dt.tz_convert("UTC").dt.normalize()
    gated = window & issue_days.isin(kept_days)
    if not bool(gated.any()):
        raise RuntimeError("no fresh issue mornings in the predict window; existing cards left on disk")
    dest.mkdir(parents=True, exist_ok=True)
    dropped = replace_window_batches(dest, kept_days)
    if dropped:
        print(f"replaced {dropped} existing card(s) in the issue window")
    return gated


def _lookup_daily_p(series: pd.Series, issued) -> float | None:
    ts = pd.Timestamp(issued)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    idx = pd.DatetimeIndex(series.index)
    if ts in idx:
        val = series.loc[ts]
        return float(val.iloc[0] if isinstance(val, pd.Series) else val)
    norms = {pd.Timestamp(i).tz_convert("UTC").normalize(): float(v) for i, v in series.items()}
    return norms.get(ts.normalize())


def issue_predictions(
    start: datetime,
    end: datetime,
    horizon_days: int = 30,
    root: Path | None = None,
) -> list[dict]:
    from eqpred.eval.mainshock import FRESH_MODEL_ID, issued_fresh_daily

    features = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    features["issue_time"] = parse_utc(features["issue_time"])
    quakes = load_comcat(root)
    dest = prediction_dir(root)
    registry = ensure_registered(root)
    budget = AlertBudget.from_config(load_budget(root))
    _feat, daily = issued_fresh_daily(root=root, features=features, quakes=quakes)

    window = (
        (features["issue_time"] >= utc_ts(start))
        & (features["issue_time"] < utc_ts(end))
    )
    cards: list[dict] = []
    written = datetime.now(timezone.utc)
    prior = load_batch_cards(dest)
    window = begin_issue(features, window, root, dest)
    issued_days = {
        pd.Timestamp(ts).tz_convert("UTC").normalize()
        for ts in parse_utc(features.loc[window, "issue_time"])
    }
    seeded = budget.seed_from_cards(prior, exclude_days=issued_days)
    if seeded:
        print(f"alarm budget replayed {seeded} existing alarm card(s)")

    fusion_targets = [("global", REGIONS["global"], GLOBAL_MAG)]
    fusion_targets.extend(
        (rid, region, BOX_MAG.get(rid, NZ_MAG))
        for rid, region in REGIONS.items()
        if is_issued_box(region)
    )

    for _, row in features.loc[window].iterrows():
        issued = row["issue_time"].to_pydatetime()
        valid_to = issued + timedelta(days=horizon_days)
        feat_row = features.loc[features["issue_time"] == row["issue_time"]].iloc[0]
        spf_on = float(feat_row.get("spf_significant_window") or 0) > 0.5

        for region_id, region, mag in fusion_targets:
            pack = daily.get(region_id)
            if not pack:
                continue
            value = _lookup_daily_p(pack["p"], row["issue_time"])
            if value is None:
                continue
            climate = float(pack["climatology_p"])
            why = [str(c.get("label") or c.get("driver") or "") for c in (pack.get("connections") or []) if c]
            why_txt = ", ".join(bit for bit in why[:4] if bit) or "the fresh-shock mix"
            cards.append(
                prediction(
                    prediction_id=f"{issued.date()}-{region_id}-m{mag:g}-{uuid4().hex[:8]}",
                    issued_at=issued,
                    valid_from=issued,
                    valid_to=valid_to,
                    horizon_days=horizon_days,
                    region=region,
                    magnitude_min=mag,
                    kind="probability",
                    value=value,
                    model_id=FRESH_MODEL_ID,
                    drivers=fresh_attrs(pack.get("connections")),
                    rationale=(
                        f"fresh-shock p={value:.4f} climatology={climate:.4f} "
                        f"uses {why_txt}"
                    ),
                    competitor_climatology_p=climate,
                    written_at=written,
                    card_kind=classify_card_kind(issued, written),
                )
            )
            if region_id == "global" and spf_on and budget.spend("alarm.spf.v2", region_id, issued):
                cap = budget.cap("alarm.spf.v2")
                used = budget.used("alarm.spf.v2", region_id, issued)
                cards.append(
                    prediction(
                        prediction_id=f"{issued.date()}-global-m8-alarm-{uuid4().hex[:8]}",
                        issued_at=issued,
                        valid_from=issued,
                        valid_to=issued + timedelta(days=20),
                        horizon_days=20,
                        region=REGIONS["global"],
                        magnitude_min=8.0,
                        kind="alarm",
                        value=1.0,
                        model_id="alarm.spf.v2",
                        drivers=alarm_attrs("spf_significant_window"),
                        rationale=(
                            "Tightened SPF window (v2 + 20% train budget); "
                            f"monthly budget {used}/{cap}"
                        ),
                        written_at=written,
                        card_kind=classify_card_kind(issued, written),
                    )
                )

    stamp_cards(cards, registry)
    out = dest / f"batch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    write_batch_meta(out, registry, n_cards=len(cards))
    print(f"registry {registry['id']}")
    return cards
