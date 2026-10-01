from __future__ import annotations

"""Label-free Earth-state tools: anomaly, historical analogs, fault-network stress.

The anomaly model never sees earthquake labels. Analog search only uses a
past morning after its 30-day window has closed. Neither writes issued p.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eqpred.catalog import load_comcat
from eqpred.eval.decluster import fresh_hits
from eqpred.eval.honest import EVAL_PROTOCOL, purged_train_hold, week_mask
from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.hunt import TARGETS, driver_family
from eqpred.faults import GRAPH_EDGES, SEGMENTS, segment_region
from eqpred.features.graph import attach_graph_features
from eqpred.paths import data_dir, feature_dir
from eqpred.regions import REGIONS, leaflet_rectangles
from eqpred.stamp import parse_utc, utc_ts

STATE_COLS = (
    "proton_density_cm3",
    "solar_wind_speed_kms",
    "imf_bz_nt",
    "kp",
    "dst_nt",
    "f107",
    "neutron_oulu_count",
    "tec_tecu",
    "gnss_east_7d",
    "gnss_strain_nz_dilatation_30d",
    "dart_d1_m",
    "tide_nz_mm",
    "era5_swvl1",
    "grace_hikurangi_lwe_cm",
    "strain_ee_nstrain",
    "tidal_strain",
    "mag_h_range_nt",
    "climate_ao",
    "eop_lod_s",
    "cme_speed_kms",
    "graph_inflow_hikurangi",
    "graph_inflow_alpine",
    "graph_inflow_kermadec",
)
K_NEIGHBORS = 8
UNUSUAL_Q = 0.95
MIN_STATE = 4


def _present_state_cols(feat: pd.DataFrame) -> list[str]:
    return [c for c in STATE_COLS if c in feat.columns]


def _state_matrix(feat: pd.DataFrame, cols: list[str], train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    raw = feat[cols].apply(pd.to_numeric, errors="coerce")
    med = raw.loc[train_mask].median(numeric_only=True)
    filled = raw.fillna(med)
    mu = filled.loc[train_mask].mean(numeric_only=True)
    sd = filled.loc[train_mask].std(numeric_only=True).replace(0.0, np.nan)
    z = (filled - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.to_numpy(dtype=float), mu.to_dict(), sd.to_dict()


def attach_anomaly_features(feat: pd.DataFrame, train_end: str = "2018-12-31") -> pd.DataFrame:
    """Fit a normal-Earth model on the training era only. No earthquake labels."""
    work = feat.copy()
    work["issue_time"] = parse_utc(work["issue_time"])
    cols = _present_state_cols(work)
    if len(cols) < MIN_STATE:
        return work
    train_mask = (work["issue_time"] <= utc_ts(train_end)).to_numpy()
    if int(train_mask.sum()) < 80:
        return work
    X, _mu, _sd = _state_matrix(work, cols, train_mask)
    x_train = X[train_mask]
    recon = _pca_recon(x_train, X)
    mahal = _mahalanobis(x_train, X)
    forest = _iforest(x_train, X)
    parts = [recon, mahal]
    if forest is not None:
        parts.append(forest)
    stacked = np.vstack(parts)
    # Each score → train-era rank so they share a scale, then average.
    ranks = []
    for row in stacked:
        train_vals = row[train_mask]
        ranks.append(_rank01(row, train_vals))
    unusual = np.mean(np.vstack(ranks), axis=0)
    cut = float(np.nanquantile(unusual[train_mask], UNUSUAL_Q))
    work["earth_state_recon"] = recon
    work["earth_state_mahal"] = mahal
    if forest is not None:
        work["earth_state_iforest"] = forest
    work["earth_state_z"] = unusual
    work["earth_state_unusual"] = (unusual >= cut).astype(float)
    return work


def _rank01(values: np.ndarray, train_vals: np.ndarray) -> np.ndarray:
    ref = np.sort(train_vals[np.isfinite(train_vals)])
    if ref.size == 0:
        return np.zeros_like(values, dtype=float)
    return np.searchsorted(ref, np.nan_to_num(values, nan=ref[0]), side="right") / float(ref.size)


def _pca_recon(train: np.ndarray, all_x: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    k = int(min(8, train.shape[1], max(1, train.shape[0] // 20)))
    model = PCA(n_components=k, random_state=0)
    model.fit(train)
    hat = model.inverse_transform(model.transform(all_x))
    return np.mean((all_x - hat) ** 2, axis=1)


def _mahalanobis(train: np.ndarray, all_x: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    cov = LedoitWolf().fit(train)
    prec = cov.precision_
    delta = all_x - cov.location_
    return np.einsum("ij,jk,ik->i", delta, prec, delta)


def _iforest(train: np.ndarray, all_x: np.ndarray) -> np.ndarray | None:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return None
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=0,
        n_jobs=1,
    )
    model.fit(train)
    return -model.score_samples(all_x)


def _closed_mask(days: pd.DatetimeIndex, last_quake: pd.Timestamp, horizon: int) -> np.ndarray:
    end = pd.DatetimeIndex(days) + pd.Timedelta(days=horizon)
    return np.asarray(end <= last_quake, dtype=bool)


def _diverse_pick(
    indices: np.ndarray,
    distances: np.ndarray,
    days: pd.DatetimeIndex,
    k: int,
    min_gap: int = HORIZON_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the closest mornings that are not the same storm twice."""
    if indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    order = np.argsort(distances)
    picked: list[int] = []
    picked_ts: list[pd.Timestamp] = []
    picked_dist: list[float] = []
    gap = pd.Timedelta(days=int(min_gap))
    for pos in order:
        idx = int(indices[pos])
        ts = pd.Timestamp(days[idx])
        if any(abs(ts - prev) < gap for prev in picked_ts):
            continue
        picked.append(idx)
        picked_ts.append(ts)
        picked_dist.append(float(distances[pos]))
        if len(picked) >= k:
            break
    return np.asarray(picked, dtype=int), np.asarray(picked_dist, dtype=float)


def analog_neighbors(
    X: np.ndarray,
    query_idx: int,
    library: np.ndarray,
    k: int = K_NEIGHBORS,
    days: pd.DatetimeIndex | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest library rows to one query. Library holds integer indexes."""
    if library.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    delta = X[library] - X[query_idx]
    dist = np.sqrt(np.sum(delta * delta, axis=1))
    if days is None:
        take = min(int(k), int(library.size))
        order = np.argpartition(dist, take - 1)[:take]
        order = order[np.argsort(dist[order])]
        return library[order], dist[order]
    return _diverse_pick(library, dist, days, k=k)


def analog_forecasts(
    X: np.ndarray,
    days: pd.DatetimeIndex,
    hits: dict[str, np.ndarray],
    train_mask: np.ndarray,
    k: int = K_NEIGHBORS,
    horizon: int = HORIZON_DAYS,
) -> dict[str, np.ndarray]:
    """Frozen train-era analog library. Each analog's window has already closed."""
    lib = np.flatnonzero(train_mask)
    out = {rid: np.full(len(days), np.nan, dtype=float) for rid in hits}
    if lib.size < k:
        return out
    # One sklearn index over the frozen library.
    from sklearn.neighbors import NearestNeighbors

    pool = min(max(k * 8, 24), int(lib.size))
    nn = NearestNeighbors(n_neighbors=pool, algorithm="auto")
    nn.fit(X[lib])
    dist, idx = nn.kneighbors(X)
    picked = np.full((len(days), k), -1, dtype=int)
    for i in range(len(days)):
        chosen, _d = _diverse_pick(lib[idx[i]], dist[i], days, k=k)
        if chosen.size:
            picked[i, : chosen.size] = chosen
    valid = picked >= 0
    for rid, y in hits.items():
        scores = np.full(len(days), np.nan, dtype=float)
        for i in range(len(days)):
            sel = picked[i][valid[i]]
            if sel.size:
                scores[i] = float(y[sel].mean())
        out[rid] = scores
    return out


def _this_week_analogs(
    X: np.ndarray,
    days: pd.DatetimeIndex,
    hits: dict[str, np.ndarray],
    train_mask: np.ndarray,
    query_idx: int,
    k: int = K_NEIGHBORS,
) -> list[dict[str, Any]]:
    lib = np.flatnonzero(train_mask)
    neigh, dist = analog_neighbors(X, query_idx, lib, k=k, days=days)
    rows = []
    for j, d in zip(neigh, dist):
        rec = {
            "issue_day": days[int(j)].date().isoformat(),
            "distance": float(d),
        }
        for rid, y in hits.items():
            rec[f"followed_{rid}"] = bool(y[int(j)])
        rows.append(rec)
    return rows


def _brier(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    keep = mask & np.isfinite(p)
    if int(keep.sum()) < 20:
        return None
    err = np.asarray(p[keep], dtype=float) - np.asarray(y[keep], dtype=float)
    return float(np.mean(err * err))


def run_earth_state(
    root: Path | None = None,
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    write_features: bool = True,
) -> dict[str, Any]:
    path = feature_dir(root) / "global_daily.parquet"
    feat = pd.read_parquet(path)
    feat["issue_time"] = parse_utc(feat["issue_time"])
    quakes = load_comcat(root)
    if not quakes.empty:
        quakes = quakes.copy()
        quakes["observed_at"] = parse_utc(quakes["observed_at"])
        if "available_at" in quakes.columns:
            quakes["available_at"] = parse_utc(quakes["available_at"])
        else:
            quakes["available_at"] = quakes["observed_at"]
    feat = attach_graph_features(feat, quakes, train_end=train_end)
    feat = attach_anomaly_features(feat, train_end=train_end)

    cols = _present_state_cols(feat)
    days = pd.DatetimeIndex(feat["issue_time"])
    last_quake = parse_utc(quakes["observed_at"]).max() if not quakes.empty else days.max()
    closed = _closed_mask(days, last_quake, HORIZON_DAYS)
    train_mask, hold_mask = purged_train_hold(days, closed, train_end, holdout_start)
    X, _mu, _sd = _state_matrix(feat, cols, train_mask) if len(cols) >= MIN_STATE else (np.zeros((len(feat), 1)), {}, {})

    hits = {}
    for rid, mag in TARGETS:
        hits[rid] = fresh_hits(quakes, REGIONS[rid], mag, days, HORIZON_DAYS).astype(float)

    analog_p = analog_forecasts(X, days, hits, train_mask) if len(cols) >= MIN_STATE else {}
    for rid, series in analog_p.items():
        feat[f"analog_p_{rid}"] = series

    if write_features:
        dest = feature_dir(root) / "global_daily.parquet"
        feat.to_parquet(dest, index=False)

    hold_week = week_mask(hold_mask)
    train_week = week_mask(train_mask)
    regions = []
    for rid, mag in TARGETS:
        y = hits[rid]
        p = analog_p.get(rid, np.full(len(feat), np.nan))
        climate = float(y[train_mask].mean()) if train_mask.any() else None
        climate_p = np.full(len(feat), climate if climate is not None else np.nan)
        analog_hold = _brier(p, y, hold_week)
        climate_hold = _brier(climate_p, y, hold_week)
        unusual = pd.to_numeric(feat.get("earth_state_z"), errors="coerce").to_numpy(dtype=float)
        regions.append(
            {
                "region": rid,
                "label": REGIONS[rid].get("label") or rid,
                "magnitude_min": mag,
                "analog_brier": analog_hold,
                "climatology_brier": climate_hold,
                "beats_climatology": bool(
                    analog_hold is not None
                    and climate_hold is not None
                    and analog_hold < climate_hold - 1e-6
                ),
                "analog_rate": float(p[hold_week].mean()) if hold_week.any() and np.isfinite(p[hold_week]).any() else None,
                "fresh_rate_holdout": float(y[hold_week].mean()) if hold_week.any() else None,
                "anomaly_mean_on_hits": float(np.nanmean(unusual[hold_week & (y > 0.5)])) if np.any(hold_week & (y > 0.5)) else None,
                "anomaly_mean_on_quiet": float(np.nanmean(unusual[hold_week & (y < 0.5)])) if np.any(hold_week & (y < 0.5)) else None,
                "this_week_analog_p": None if not np.isfinite(p[-1]) else float(p[-1]),
            }
        )

    query_idx = int(len(feat) - 1)
    analogs = (
        _this_week_analogs(X, days, hits, train_mask, query_idx)
        if len(cols) >= MIN_STATE
        else []
    )
    latest = feat.iloc[-1]
    nodes = []
    for sid, rec in SEGMENTS.items():
        nodes.append(
            {
                "id": sid,
                "label": rec["label"],
                "parent": rec["parent"],
                "energy": _json_float(latest.get(f"graph_n_{sid}")),
                "inflow": _json_float(latest.get(f"graph_inflow_{sid}")),
                "stress": _json_float(latest.get(f"graph_stress_{sid}")),
                "polygons": leaflet_rectangles(segment_region(sid)),
            }
        )
    report = {
        "writes_issued_p": False,
        "eval_protocol": EVAL_PROTOCOL,
        "note": (
            "Earth-state tools. The anomaly model is fit on training-era mornings "
            "with no earthquake labels. Analogs are earlier training mornings whose "
            "next 30 days have already closed. The fault graph moves seismic energy "
            "between neighboring segments instead of treating each box as an island. "
            "None of this rewrites issued p."
        ),
        "issue_day": days[-1].date().isoformat(),
        "n_state_columns": len(cols),
        "state_columns": cols,
        "k_neighbors": K_NEIGHBORS,
        "anomaly": {
            "earth_state_z": _json_float(latest.get("earth_state_z")),
            "earth_state_unusual": bool(float(latest.get("earth_state_unusual") or 0) > 0.5),
            "earth_state_recon": _json_float(latest.get("earth_state_recon")),
            "earth_state_mahal": _json_float(latest.get("earth_state_mahal")),
            "earth_state_iforest": _json_float(latest.get("earth_state_iforest")),
            "n_unusual_days": int(pd.to_numeric(feat.get("earth_state_unusual"), errors="coerce").fillna(0).sum())
            if "earth_state_unusual" in feat.columns
            else 0,
        },
        "analogs": analogs,
        "graph": {
            "nodes": nodes,
            "edges": [
                {"source": a, "target": b, "weight": w, "kind": kind} for a, b, w, kind in GRAPH_EDGES
            ],
            "transfer_on": bool(float(latest.get("graph_transfer_on") or 0) > 0.5),
            "max_inflow": _json_float(latest.get("graph_max_inflow")),
        },
        "regions": regions,
        "n_beats_climatology": int(sum(1 for r in regions if r.get("beats_climatology"))),
        "driver_families": {
            "earth_state_z": driver_family("earth_state_z"),
            "graph_inflow_hikurangi": driver_family("graph_inflow_hikurangi"),
            "analog_p_global": driver_family("analog_p_global"),
        },
        "n_train_analogs": int(train_mask.sum()),
        "n_holdout_weeks": int(hold_week.sum()),
        "n_train_weeks": int(train_week.sum()),
    }
    dest = data_dir(root) / "eval" / "earth_state.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def load_earth_state(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "earth_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
