from __future__ import annotations

import numpy as np
import pandas as pd

from eqpred.eval.decluster import haversine_km
from eqpred.models.attribution import flatten_driver_field
from eqpred.stamp import parse_utc

QUIET_MODEL = "alarm.quiet.v1"
GAP_DAYS = 10
PRIOR_DAYS = 14
NEIGHBOR_KM = 250.0
NEIGHBOR_DAYS = 10.0


def _cluster_dates(stamps: list[pd.Timestamp], gap_days: int = GAP_DAYS) -> list[list[pd.Timestamp]]:
    ordered = sorted(stamps)
    if not ordered:
        return []
    groups: list[list[pd.Timestamp]] = [[ordered[0]]]
    for stamp in ordered[1:]:
        if (stamp - groups[-1][-1]) / pd.Timedelta(days=1) <= gap_days:
            groups[-1].append(stamp)
        else:
            groups.append([stamp])
    return groups


def _has_prior_neighbor(event: pd.Series, quakes: pd.DataFrame) -> bool:
    t0 = event["observed_at"]
    prior = quakes.loc[
        (quakes["observed_at"] < t0)
        & (quakes["observed_at"] >= t0 - pd.Timedelta(days=NEIGHBOR_DAYS))
        & (quakes["magnitude"] >= float(event["magnitude"]) - 0.2)
    ]
    if prior.empty:
        return False
    dist = haversine_km(
        float(event["lon"]),
        float(event["lat"]),
        prior["lon"].to_numpy(dtype=float),
        prior["lat"].to_numpy(dtype=float),
    )
    return bool(np.any(dist <= NEIGHBOR_KM))


def quiet_episodes(ledger: pd.DataFrame, quakes: pd.DataFrame, cards: pd.DataFrame | None = None) -> list[dict]:
    """Group overlapping quiet-M7 alarms and say whether hits look like aftershocks."""
    if ledger.empty or "model_id" not in ledger.columns:
        return []
    quiet = ledger.loc[ledger["model_id"] == QUIET_MODEL].copy()
    if quiet.empty:
        return []
    quiet["issued_at"] = parse_utc(quiet["issued_at"])
    catalog = quakes.copy()
    catalog["observed_at"] = parse_utc(catalog["observed_at"])
    if cards is not None and not cards.empty and "prediction_id" in cards.columns:
        extra = cards[["prediction_id", "drivers", "rationale"]].copy() if "drivers" in cards.columns else pd.DataFrame()
        if not extra.empty:
            quiet = quiet.merge(extra, on="prediction_id", how="left")

    m7 = catalog.loc[catalog["magnitude"] >= 7.0]
    groups = _cluster_dates(list(quiet["issued_at"]))
    out: list[dict] = []
    for i, stamps in enumerate(groups, start=1):
        part = quiet.loc[quiet["issued_at"].isin(stamps)].sort_values("issued_at")
        start = stamps[0]
        end = stamps[-1] + pd.Timedelta(days=7)
        window = m7.loc[(m7["observed_at"] >= start) & (m7["observed_at"] < end)]
        prior = m7.loc[
            (m7["observed_at"] >= start - pd.Timedelta(days=PRIOR_DAYS))
            & (m7["observed_at"] < start)
        ]
        events = []
        fresh = 0
        echo = 0
        neighbors = catalog.loc[catalog["magnitude"] >= 6.5]
        for rec in window.itertuples(index=False):
            series = pd.Series(
                {
                    "observed_at": rec.observed_at,
                    "magnitude": rec.magnitude,
                    "lon": rec.lon,
                    "lat": rec.lat,
                }
            )
            after = _has_prior_neighbor(series, neighbors)
            events.append(
                {
                    "observed_at": pd.Timestamp(rec.observed_at).isoformat().replace("+00:00", "Z"),
                    "magnitude": float(rec.magnitude),
                    "place": getattr(rec, "place", "") or "",
                    "lon": float(rec.lon),
                    "lat": float(rec.lat),
                    "aftershock_like": after,
                }
            )
            if after:
                echo += 1
            else:
                fresh += 1
        if not events:
            verdict = "miss"
        elif echo and not fresh:
            verdict = "hit_echo"
        elif fresh and not echo:
            verdict = "hit_fresh" if prior.empty else "hit_after_prior_m7"
        else:
            verdict = "hit_mixed"
        drivers: list[str] = []
        if "drivers" in part.columns:
            for raw in part["drivers"]:
                drivers.extend(flatten_driver_field(raw))
        drivers = sorted({d for d in drivers if d and d != "etas_quiet"})
        out.append(
            {
                "episode_id": i,
                "start": start.date().isoformat(),
                "end": stamps[-1].date().isoformat(),
                "n_cards": int(len(part)),
                "n_hits": int(part["hit"].sum()) if "hit" in part.columns else 0,
                "issue_dates": [s.date().isoformat() for s in stamps],
                "drivers": drivers,
                "etas_p": float(part["competitor_etas_p"].mean()) if "competitor_etas_p" in part.columns else None,
                "prior_m7_14d": int(len(prior)),
                "window_m7": events,
                "n_fresh": fresh,
                "n_echo": echo,
                "verdict": verdict,
            }
        )
    return out
