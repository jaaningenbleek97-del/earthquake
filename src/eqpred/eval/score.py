from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from eqpred.eval.nearmiss import grade_card
from eqpred.ingest.run import latest_raw
from eqpred.models.anti import invert_grade, is_anti_model
from eqpred.paths import prediction_batches, prediction_dir
from eqpred.stamp import parse_utc, utc_ts


def _ungraded() -> dict:
    return {
        "hit": None,
        "n_events": None,
        "near_miss": None,
        "miss_km": None,
        "miss_hours": None,
        "miss_mag": None,
        "near_magnitude": None,
        "near_place": None,
    }


def _spearman(pred, obs) -> float | None:
    """Same rank metric the Wave 2 / box residual train gate uses."""
    x = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    raw = spearmanr(x, y).correlation
    if raw is None or not np.isfinite(raw):
        return None
    return float(raw)


def _scored_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    if "window_closed" not in ledger.columns:
        return ledger
    return ledger.loc[ledger["window_closed"].fillna(True)]


def score_predictions(root: Path | None = None) -> pd.DataFrame:
    files = prediction_batches(root)
    if not files:
        raise FileNotFoundError("no prediction batches")
    cards: list[dict] = []
    for path in files:
        cards.extend(json.loads(path.read_text(encoding="utf-8")))
    # Ground truth: the final bulletin. λ at issue time is as-of; hits are not.
    quakes = latest_raw("usgs.comcat", root)
    quakes["observed_at"] = parse_utc(quakes["observed_at"])
    catalog_end = quakes["observed_at"].max() if not quakes.empty else None

    rows = []
    for card in cards:
        closed = catalog_end is not None and utc_ts(card["valid_to"]) <= catalog_end
        if closed:
            graded = grade_card(card, quakes)
            if is_anti_model(card.get("model_id")):
                graded = invert_grade(graded)
        else:
            graded = _ungraded()
        region = card["region"]
        rows.append(
            {
                "prediction_id": card["prediction_id"],
                "model_id": card["model_id"],
                "issued_at": card["issued_at"],
                "written_at": card.get("written_at"),
                "card_kind": card.get("card_kind") or "retrospective",
                "kind": card["kind"],
                "value": card["value"],
                "competitor_etas_p": card.get("competitor_etas_p"),
                "competitor_climatology_p": card.get("competitor_climatology_p"),
                "magnitude_min": card["magnitude_min"],
                "region": region.get("name", region.get("kind")),
                "window_closed": bool(closed),
                "hit": graded["hit"],
                "n_events": graded["n_events"],
                "near_miss": graded["near_miss"],
                "miss_km": graded["miss_km"],
                "miss_hours": graded["miss_hours"],
                "miss_mag": graded["miss_mag"],
                "near_magnitude": graded["near_magnitude"],
                "near_place": graded["near_place"],
            }
        )
    ledger = pd.DataFrame(rows)
    dest = prediction_dir(root) / "ledger.parquet"
    ledger.to_parquet(dest, index=False)
    return ledger


def summarize_ledger(ledger: pd.DataFrame) -> dict:
    out: dict = {"n_cards": int(len(ledger)), "n_scored": 0, "n_open_windows": 0, "heads": []}
    if ledger.empty:
        return out
    scored = _scored_rows(ledger)
    out["n_scored"] = int(len(scored))
    out["n_open_windows"] = int(len(ledger) - len(scored))
    if "card_kind" in ledger.columns:
        kinds = ledger["card_kind"].fillna("retrospective")
        out["n_prospective"] = int((kinds == "prospective").sum())
        out["n_retrospective"] = int((kinds == "retrospective").sum())
    else:
        out["n_prospective"] = 0
        out["n_retrospective"] = int(len(ledger))
    if scored.empty:
        return out
    calendar_days = max(int(ledger["issued_at"].nunique()), 1)
    keys = [c for c in ("model_id", "region", "magnitude_min") if c in scored.columns]
    for key, part in scored.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        fields = dict(zip(keys, key))
        is_alarm = bool(part["kind"].eq("alarm").any())
        etas = part["competitor_etas_p"] if "competitor_etas_p" in part.columns else None
        brier = float(((part["value"] - part["hit"]) ** 2).mean()) if not is_alarm else None
        brier_etas = None
        if etas is not None and etas.notna().any() and not is_alarm:
            aligned = part.loc[etas.notna()]
            brier_etas = float(((aligned["competitor_etas_p"] - aligned["hit"]) ** 2).mean())
        murphy = murphy_decompose(part["value"], part["hit"]) if not is_alarm else {}
        corr = None if is_alarm else _spearman(part["value"], part["hit"])
        near_mean = near_corr = None
        if "near_miss" in part.columns:
            near = pd.to_numeric(part["near_miss"], errors="coerce")
            if near.notna().any():
                near_mean = float(near.mean())
                if not is_alarm:
                    near_corr = _spearman(part["value"], near)
        row = {
            "model_id": fields.get("model_id"),
            "region": fields.get("region"),
            "magnitude_min": float(fields["magnitude_min"]) if fields.get("magnitude_min") is not None else None,
            "n": int(len(part)),
            "hit_rate": float(part["hit"].mean()),
            "mean_p": float(part["value"].mean()),
            "mean_etas_p": float(etas.mean()) if etas is not None and etas.notna().any() else None,
            "brier": brier,
            "brier_etas": brier_etas,
            "p_hit_corr": corr,
            "mean_near_miss": near_mean,
            "p_near_corr": near_corr,
            "resolution": murphy.get("resolution"),
            "reliability_term": murphy.get("reliability"),
            "uncertainty": murphy.get("uncertainty"),
            "time_in_alarm": float(len(part) / calendar_days) if is_alarm else None,
            "alarm_months": None,
            "max_days_in_month": None,
            "mean_days_per_month": None,
        }
        if is_alarm:
            stamps = parse_utc(part["issued_at"])
            month_counts = stamps.dt.strftime("%Y-%m").value_counts()
            row["alarm_months"] = int(len(month_counts))
            row["max_days_in_month"] = int(month_counts.max()) if len(month_counts) else 0
            row["mean_days_per_month"] = float(month_counts.mean()) if len(month_counts) else 0.0
        out["heads"].append(row)
        # Keep a model-level rollup for anything that still keys on model_id.
        model = row["model_id"]
        if model not in out:
            out[model] = {
                "n": 0,
                "hit_rate": 0.0,
                "mean_p": 0.0,
                "mean_etas_p": None,
                "time_in_alarm": None,
            }
        prev = out[model]
        n0, n1 = prev["n"], row["n"]
        total = n0 + n1
        prev["hit_rate"] = (prev["hit_rate"] * n0 + row["hit_rate"] * n1) / total
        prev["mean_p"] = (prev["mean_p"] * n0 + row["mean_p"] * n1) / total
        if row["mean_etas_p"] is not None:
            if prev["mean_etas_p"] is None:
                prev["mean_etas_p"] = row["mean_etas_p"]
            else:
                prev["mean_etas_p"] = (prev["mean_etas_p"] * n0 + row["mean_etas_p"] * n1) / total
        if is_alarm:
            prev["time_in_alarm"] = (prev.get("time_in_alarm") or 0.0) + (row["time_in_alarm"] or 0.0)
        prev["n"] = total
    fusion_quiet: dict[tuple, float] = {}
    for head in out["heads"]:
        if head.get("model_id") == "fusion.etas_x_mod.v1" and head.get("hit_rate") is not None:
            fusion_quiet[(head.get("region"), head.get("magnitude_min"))] = 1.0 - float(head["hit_rate"])
    for head in out["heads"]:
        if not is_anti_model(head.get("model_id")):
            continue
        base = fusion_quiet.get((head.get("region"), head.get("magnitude_min")))
        head["quiet_base_rate"] = base
        head["quiet_lift"] = None if base is None else float(head["hit_rate"]) - base
    return out


RELIABILITY_QUANTILES = 4


def murphy_decompose(p, y, n_quantiles: int = RELIABILITY_QUANTILES) -> dict:
    """Brier = reliability − resolution + uncertainty. Resolution is ranking."""
    pred = np.asarray(p, dtype=float)
    obs = np.asarray(y, dtype=float)
    n = int(obs.size)
    if n == 0:
        return {"uncertainty": None, "reliability": None, "resolution": None}
    ybar = float(obs.mean())
    unc = float(ybar * (1.0 - ybar))
    if np.unique(pred).size < 2:
        return {
            "uncertainty": unc,
            "reliability": float((float(pred.mean()) - ybar) ** 2),
            "resolution": 0.0,
        }
    frame = pd.DataFrame({"p": pred, "y": obs})
    frame["bin"] = pd.qcut(frame["p"], n_quantiles, duplicates="drop")
    rel = res = 0.0
    for _, grp in frame.groupby("bin", observed=False):
        if grp.empty:
            continue
        weight = len(grp) / n
        pk = float(grp["p"].mean())
        yk = float(grp["y"].mean())
        rel += weight * (pk - yk) ** 2
        res += weight * (yk - ybar) ** 2
    return {"uncertainty": unc, "reliability": float(rel), "resolution": float(res)}


def reliability_table(ledger: pd.DataFrame) -> list[dict]:
    """Hit rate vs mean p in issued-p quantiles. Flat hit rates mean p is not ranking weeks."""
    if ledger.empty or "kind" not in ledger.columns:
        return []
    work = _scored_rows(ledger)
    work = work.loc[work["kind"].eq("probability")].copy()
    if work.empty:
        return []
    keys = [c for c in ("model_id", "region") if c in work.columns]
    rows: list[dict] = []
    for key, part in work.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        fields = dict(zip(keys, key))
        part = part.copy()
        if part["value"].nunique() < 2:
            bins = [
                {
                    "lo": float(part["value"].min()),
                    "hi": float(part["value"].max()),
                    "n": int(len(part)),
                    "hit_rate": float(part["hit"].mean()),
                    "mean_p": float(part["value"].mean()),
                }
            ]
            rows.append({**fields, "bins": bins})
            continue
        part["bin"] = pd.qcut(part["value"], RELIABILITY_QUANTILES, duplicates="drop")
        bins = []
        for interval, grp in part.groupby("bin", observed=True):
            if grp.empty:
                continue
            bins.append(
                {
                    "lo": float(interval.left),
                    "hi": float(interval.right),
                    "n": int(len(grp)),
                    "hit_rate": float(grp["hit"].mean()),
                    "mean_p": float(grp["value"].mean()),
                }
            )
        rows.append({**fields, "bins": bins})
    return rows
