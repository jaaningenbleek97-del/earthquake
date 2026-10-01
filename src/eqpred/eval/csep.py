"""CSEP-style tests for the issued box forecasts.

The cards are P(at least one fresh shock in the horizon), not a spatial grid.
N / L / T use non-overlapping mornings (one per horizon). S compares the four
issued boxes. Molchan and the area skill score rank days by p.
Nothing here writes issued p.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from eqpred.catalog import load_comcat
from eqpred.eval.compat import saved_fresh
from eqpred.eval.decluster import fresh_hits
from eqpred.eval.honest import week_mask
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.mainshock import ISSUED_TARGETS, _chance, _place, load_mainshock
from eqpred.paths import data_dir, feature_dir
from eqpred.records import iso
from eqpred.regions import REGIONS
from eqpred.stamp import parse_utc, utc_ts

N_SIM = 2000
ALPHA = 0.025
RNG = 7


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Area under (x, y). Avoids numpy.trapz / trapezoid version splits."""
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


def _expected_count(p: np.ndarray) -> np.ndarray:
    """Expected hit-window count for a binary P(at least one) forecast.

    Cards are Bernoulli, not a Poisson catalog. E[N] is sum(p), not
    sum(-log(1-p)).
    """
    return np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)


def n_test(p: np.ndarray, hits: np.ndarray, n_sim: int = N_SIM, seed: int = RNG) -> dict[str, Any]:
    """Number test: is the count of hit windows typical of independent Bernoullis?"""
    ok = np.isfinite(p) & np.isfinite(hits)
    if int(ok.sum()) < 8:
        return {"name": "N", "error": "too few windows"}
    prob = _expected_count(p[ok])
    n_fore = float(prob.sum())
    n_obs = float(np.asarray(hits[ok], dtype=float).sum())
    if not np.isfinite(n_fore) or n_fore <= 0:
        return {"name": "N", "error": "empty forecast"}
    rng = np.random.default_rng(seed)
    sim = (rng.random((int(n_sim), prob.size)) < prob).sum(axis=1).astype(float)
    delta1 = float(np.mean(sim >= n_obs))
    delta2 = float(np.mean(sim <= n_obs))
    return {
        "name": "N",
        "n_forecast": n_fore,
        "n_observed": n_obs,
        "delta1": delta1,
        "delta2": delta2,
        "kind": "bernoulli_windows",
        "pass": bool(delta1 > ALPHA and delta2 > ALPHA),
        "plain": (
            f"expected {n_fore:.1f} hit windows, saw {n_obs:.0f} "
            f"({'consistent' if delta1 > ALPHA and delta2 > ALPHA else 'inconsistent'})"
        ),
    }


def l_test(p: np.ndarray, hits: np.ndarray, n_sim: int = N_SIM, seed: int = RNG) -> dict[str, Any]:
    """Likelihood test: is the joint Bernoulli likelihood typical of the forecast?"""
    ok = np.isfinite(p) & np.isfinite(hits)
    if int(ok.sum()) < 8:
        return {"name": "L", "error": "too few windows"}
    prob = np.clip(np.asarray(p[ok], dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(hits[ok], dtype=float)
    s_obs = float(np.sum(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)))
    rng = np.random.default_rng(seed)
    sim = np.empty(n_sim)
    for i in range(n_sim):
        draw = rng.random(prob.size) < prob
        sim[i] = float(np.sum(draw * np.log(prob) + (1.0 - draw) * np.log(1.0 - prob)))
    gamma = float(np.mean(sim <= s_obs))
    return {
        "name": "L",
        "log_likelihood": s_obs,
        "gamma": gamma,
        "pass": bool(gamma > 2 * ALPHA),
        "plain": (
            f"likelihood rank {gamma:.2f} "
            f"({'typical of the forecast' if gamma > 2 * ALPHA else 'unusually poor'})"
        ),
    }


def t_test(p: np.ndarray, hits: np.ndarray, n_sim: int = N_SIM, seed: int = RNG) -> dict[str, Any]:
    """Temporal test: given the observed count, are the hit mornings where the forecast said?"""
    ok = np.isfinite(p) & np.isfinite(hits)
    if int(ok.sum()) < 8:
        return {"name": "T", "error": "too few windows"}
    lam = _expected_count(p[ok])
    y = np.asarray(hits[ok], dtype=float)
    n_obs = int(y.sum())
    total = float(lam.sum())
    if n_obs < 2 or total <= 0:
        return {"name": "T", "error": "too few hits"}
    weights = lam / total
    s_obs = float(np.sum(y * np.log(np.clip(weights, 1e-12, 1.0))))
    rng = np.random.default_rng(seed)
    sim = np.empty(n_sim)
    idx = np.arange(weights.size)
    for i in range(n_sim):
        draw = rng.choice(idx, size=n_obs, replace=True, p=weights)
        counts = np.bincount(draw, minlength=weights.size).astype(float)
        sim[i] = float(np.sum(counts * np.log(np.clip(weights, 1e-12, 1.0))))
    gamma = float(np.mean(sim <= s_obs))
    return {
        "name": "T",
        "n_observed": n_obs,
        "gamma": gamma,
        "pass": bool(gamma > 2 * ALPHA),
        "plain": (
            f"timing rank {gamma:.2f} "
            f"({'hits landed where the forecast was loud' if gamma > 2 * ALPHA else 'hits missed the loud windows'})"
        ),
    }


def s_test(
    expected: dict[str, float],
    observed: dict[str, float],
    n_sim: int = N_SIM,
    seed: int = RNG,
) -> dict[str, Any]:
    """Spatial test across boxes, conditioned on the total number of hit windows."""
    keys = [k for k in expected if np.isfinite(expected[k]) and expected[k] > 0]
    if len(keys) < 2:
        return {"name": "S", "error": "need two boxes"}
    lam = np.array([expected[k] for k in keys], dtype=float)
    obs = np.array([float(observed.get(k) or 0.0) for k in keys], dtype=float)
    n_obs = int(obs.sum())
    total = float(lam.sum())
    if n_obs < 2 or total <= 0:
        return {"name": "S", "error": "too few hits"}
    weights = lam / total
    s_obs = float(np.sum(obs * np.log(np.clip(weights, 1e-12, 1.0))))
    rng = np.random.default_rng(seed)
    sim = np.empty(n_sim)
    idx = np.arange(weights.size)
    for i in range(n_sim):
        draw = rng.choice(idx, size=n_obs, replace=True, p=weights)
        counts = np.bincount(draw, minlength=weights.size).astype(float)
        sim[i] = float(np.sum(counts * np.log(np.clip(weights, 1e-12, 1.0))))
    gamma = float(np.mean(sim <= s_obs))
    by_box = {
        key: {"n_forecast": float(lam[i]), "n_observed": float(obs[i])} for i, key in enumerate(keys)
    }
    return {
        "name": "S",
        "gamma": gamma,
        "pass": bool(gamma > 2 * ALPHA),
        "boxes": by_box,
        "plain": (
            f"spatial rank {gamma:.2f} "
            f"({'the boxes that fired match the forecast' if gamma > 2 * ALPHA else 'the hits sat in the wrong boxes'})"
        ),
    }


def molchan(p: np.ndarray, hits: np.ndarray) -> dict[str, Any]:
    """Molchan diagram and area skill score. Random is 0; perfect is 1."""
    ok = np.isfinite(p) & np.isfinite(hits)
    if int(ok.sum()) < 8:
        return {"name": "Molchan", "error": "too few windows"}
    prob = np.asarray(p[ok], dtype=float)
    y = np.asarray(hits[ok], dtype=float)
    n_hit = float(y.sum())
    n = float(y.size)
    if n_hit < 1:
        return {"name": "Molchan", "error": "no hits"}
    order = np.argsort(-prob, kind="mergesort")
    y_ord = y[order]
    tau = np.arange(1, y.size + 1, dtype=float) / n
    caught = np.cumsum(y_ord)
    nu = 1.0 - caught / n_hit
    # Close the curve: (0, 1) → first alarm → (1, 0)
    tau_c = np.concatenate(([0.0], tau, [1.0]))
    nu_c = np.concatenate(([1.0], nu, [nu[-1] if nu.size else 0.0]))
    area = _trapz(nu_c, tau_c)
    ass = float(1.0 - 2.0 * area)
    # A few points for the UI, not the full 2k-row curve.
    picks = np.unique(np.linspace(0, max(tau.size - 1, 0), 12, dtype=int))
    curve = [{"tau": float(tau[i]), "nu": float(nu[i])} for i in picks]
    return {
        "name": "Molchan",
        "area": area,
        "area_skill_score": ass,
        "pass": bool(ass > 0.0),
        "curve": curve,
        "plain": (
            f"area skill {ass:+.3f} "
            f"({'better than random ranking' if ass > 0 else 'no better than ranking at random'})"
        ),
    }


def score_forecast(p: np.ndarray, hits: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Run N, L, T, and Molchan on one box."""
    if mask is None:
        mask = np.ones(len(p), dtype=bool)
    use = np.asarray(mask, dtype=bool) & np.isfinite(p)
    tests = {
        "n": n_test(p[use], hits[use]),
        "l": l_test(p[use], hits[use]),
        "t": t_test(p[use], hits[use]),
        "molchan": molchan(p[use], hits[use]),
    }
    passed = [row["name"] for row in tests.values() if row.get("pass")]
    failed = [row["name"] for row in tests.values() if row.get("pass") is False]
    return {
        "n_windows": int(use.sum()),
        "window_kind": "nonoverlapping_bernoulli",
        "n_hits": float(np.asarray(hits[use], dtype=float).sum()) if use.any() else 0.0,
        "tests": tests,
        "n_pass": len(passed),
        "n_fail": len(failed),
        "passed": passed,
        "failed": failed,
    }


def load_csep(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "csep.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_csep(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features: pd.DataFrame | None = None,
    quakes: pd.DataFrame | None = None,
    daily: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the issued fresh-shock daily p with the CSEP battery."""
    from eqpred.eval.mainshock import issued_fresh_daily

    feat = features if features is not None else pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    feat = feat.copy()
    feat["issue_time"] = parse_utc(feat["issue_time"])
    catalog = quakes if quakes is not None else load_comcat(root)
    catalog = catalog.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    if daily is None:
        _feat, daily = issued_fresh_daily(
            train_end=train_end,
            holdout_start=holdout_start,
            root=root,
            features=feat,
            quakes=catalog,
        )
    days = pd.DatetimeIndex(feat["issue_time"])
    catalog_end = catalog["observed_at"].max()
    closed = days + pd.Timedelta(days=HORIZON_DAYS) <= catalog_end
    hold = closed & (days >= utc_ts(holdout_start))
    hold_week = week_mask(hold, fallback=False)

    boxes: list[dict[str, Any]] = []
    expected: dict[str, float] = {}
    observed: dict[str, float] = {}
    for region_id, mag in ISSUED_TARGETS:
        pack = daily.get(region_id) or {}
        series = pack.get("p")
        if series is None:
            continue
        p = np.asarray(series.reindex(days).to_numpy(dtype=float), dtype=float)
        hits = fresh_hits(catalog, REGIONS[region_id], mag, days, HORIZON_DAYS).astype(float)
        scored = score_forecast(p, hits, hold_week)
        expected[region_id] = float(_expected_count(p[hold_week]).sum()) if hold_week.any() else 0.0
        observed[region_id] = float(hits[hold_week].sum()) if hold_week.any() else 0.0
        boxes.append(
            {
                "region": region_id,
                "label": _place(region_id),
                "magnitude_min": float(mag),
                **scored,
            }
        )

    spatial = s_test(expected, observed)
    n_pass = sum(int(b.get("n_pass") or 0) for b in boxes) + (1 if spatial.get("pass") else 0)
    n_fail = sum(int(b.get("n_fail") or 0) for b in boxes) + (0 if spatial.get("pass") else 1)
    main = load_mainshock(root) or {}
    report = {
        "writes_issued_p": False,
        "train_end": train_end,
        "holdout_start": holdout_start,
        "horizon_days": HORIZON_DAYS,
        "n_windows": int(hold_week.sum()),
        "window_kind": "nonoverlapping_bernoulli",
        "boxes": boxes,
        "spatial": spatial,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "week": [
            {
                "region": row.get("region"),
                "label": row.get("label"),
                "chance_plain": saved_fresh(next((w for w in (main.get("week") or []) if w.get("region") == row.get("region")), {}), "fresh_raw_chance_plain")
                or _chance(None),
            }
            for row in boxes
        ],
        "scanned_at": iso(datetime.now(timezone.utc)),
        "note": _english(boxes, spatial),
    }
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "csep.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def attach_region_csep(p_hat: np.ndarray, hits: np.ndarray, hold_mask: np.ndarray) -> dict[str, Any]:
    """Compact CSEP block stored on a mainshock region row.

    Always one morning per horizon. A short holdout does not fall back to
    overlapping daily windows.
    """
    return score_forecast(
        np.asarray(p_hat, dtype=float),
        np.asarray(hits, dtype=float),
        week_mask(hold_mask, fallback=False),
    )


def _english(boxes: list[dict[str, Any]], spatial: dict[str, Any]) -> str:
    bits = [
        f"CSEP tests on non-overlapping {HORIZON_DAYS}-day Bernoulli windows in the later years. "
        "N asks whether the count of hits matches sum(p), L whether the joint "
        "likelihood is typical, T whether the hits landed on loud mornings, S whether "
        "they landed in the loud boxes, and Molchan whether ranking days by p beats chance. "
        "These are not spatial-grid catalog tests."
    ]
    if not boxes:
        bits.append("No issued boxes had a daily p to score.")
        return " ".join(bits)
    wins = [b for b in boxes if (b.get("n_pass") or 0) >= 3]
    if wins:
        bits.append(
            "Mostly consistent on "
            + ", ".join(b["label"] for b in wins)
            + "."
        )
    else:
        bits.append("No box passed three of the four single-box tests.")
    if spatial.get("plain"):
        bits.append(spatial["plain"].capitalize() + ".")
    best = max(
        (b for b in boxes if (b.get("tests") or {}).get("molchan", {}).get("area_skill_score") is not None),
        key=lambda b: b["tests"]["molchan"]["area_skill_score"],
        default=None,
    )
    if best:
        ass = best["tests"]["molchan"]["area_skill_score"]
        bits.append(f"Best ranking is {best['label']} (area skill {ass:+.3f}).")
    bits.append("Nothing here changes the issued card.")
    return " ".join(bits)
