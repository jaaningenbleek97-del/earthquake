# What we issue

A prediction is a dated claim that can be wrong.

```
issued_at:  2026-09-05T00:00:00Z
window:     next 7 days
region:     cells covering the Hikurangi / Kermadec box
magnitude:  M >= 6.0
p:          0.18
drivers:    [{driver: catalog_nowcast, lag_hours: 168, contribution: 0.041}, …]
```

That is the product. Not a copy of USGS aftershock text. Not a research poster about ETAS.

## Record shape

Every issued call validates against [`schemas/prediction.schema.json`](../schemas/prediction.schema.json).

Required fields:

| Field | Meaning |
| --- | --- |
| `issued_at` | When we made the call. Only data with `available_at <= issued_at` may be used. |
| `valid_from` / `valid_to` | The time box. If a qualifying quake falls outside it, the call is a miss. |
| `region` | Polygon, cell list, or named box. “Somewhere on Earth” is not allowed unless the magnitude floor is extreme and we say so. |
| `magnitude_min` | Floor. Optional `magnitude_max` for a band. |
| `kind` | `alarm` (binary high-risk window) or `probability` (calibrated p) or `ranked_cells` (ordered list with per-cell p). |
| `value` | Probability in `[0, 1]` or alarm `0` / `1`. |
| `model_id` | Which head wrote this. |
| `drivers` | Structured list: `driver`, `lag_hours`, `contribution`. Fusion contribution is `ln(gain)`. Alarm contribution is 1 for a gate that fired, or tail-depth for anti-prediction. Only terms that entered the call. |
| `horizon_days` | Convenience copy of the window length. |

Optional: `rationale` (human text), `competitor_etas_p` (what ETAS said for the same box — so we can see if we are just echoing aftershocks).

## Three prediction heads

These are the things we are trying to make work. Catalog-only models sit beside them as competition, not as the roadmap.

1. **Fusion predictor** — neural (TFT / Transformer) on catalog history **plus** solar, solar wind, Kp/Dst, SPF, tides, lunar geometry, later TEC. Outputs a probability map for 1d and 7d, then 90d.
2. **Solar-magnetic alarm** — Davidson-style SPF windows, Cataldi/Marchitelli proton-density spikes, storm indices, cycle phase (Burns). Outputs a hard alarm for M≥6 globally and M8+ as a rare separate card.
3. **Geometry alarm** — tides + lunar distance/declination + optional planetary longitudes. Cheap to compute. The question is whether it moves the needle on already-loaded faults.

A daily job writes a batch of prediction records. A later scorer matches ComCat events into those boxes and updates a ledger.

## How a call is judged

- **Hit** — at least one event with M ≥ `magnitude_min` inside the region and window. This is the honest ledger label. Brier and Spearman rank corr use it.
- **Miss** — no such event.
- **Open window** — `valid_to` is after the last catalog event. The card stays on the ledger as issued; it is excluded from the scoreboard until the window can close.
- **Ingest watchdog** — OMNI and ComCat must be fresh as-of that morning (`available_at` age within slack). Wilcox is optional (Polar.html ended). Otherwise we do not write cards, and we do not delete the last good batch. Overlays (TEC, GNSS, ERA5, INTERMAGNET, Japan/Chile FDSN) can be late; they do not block.
- **Append a later window** — `eqpred predict --start 2026-01-01 --end 2026-09-07` rewrites only those mornings. The 2019–2025 holdout stays on disk. Alarm spend from the kept cards is replayed so a catch-up cannot double-use a month.
- **Model registry** — every issued batch is pinned to a frozen brain (`registry_id`, `code_hash`, feature-column recipe). `eqpred fit` may write a new version; it does not delete the one that already issued cards.
- **Near-miss** (learning only) — a graded score in `[0, 1]` for an event that was close: inside 150 km of the box, 48 hours of the window, and 0.5 magnitude of the floor. A M5.9 one cell over, 8 hours late is not a total miss. Geometric mean of space / time / magnitude credits. Never replaces binary hit.
- **Time-in-alarm** — fraction of issue-days the alarm head was on. A head that is on 40% of days must beat that base rate.
- **Alert budget** — each alarm head may spend at most N days per region per calendar month (SPF 6, quiet specialists 2, anti-prediction 4). Caps are locked constants, same philosophy as the 20% train-era SPF cutoff. Spent in date order, which is what a live issuer can do. A long SPF blob uses the month on its first days. Probability cards are not budgeted.
- **Anti-prediction** — “this box stays quiet for 7 days.” Issued when fusion *p* is at or below the train-era 15th percentile for that box. Hit = no qualifying quake. Score against always-quiet (1 − fusion hit rate). Do not treat a 98% Alpine quiet rate as skill if the head is on every day.
- **Vs ETAS** — same boxes, same days. If our p is just ETAS with extra noise, we failed.
- **Attribution** — every card lists the terms that entered the number. Fusion rows are log-rate contributions plus the lag the feature was built at. Solar flags that did not multiply issued *p* are not listed. Alarm rows name the gates that spent the budget.
- **Migration** — a cluster visible that morning, projected onto the locked fault plane. Along-strike and up-dip km/day are diagnostics. One of them may multiply box *p* only if it beat the current issued baseline on train. Events deeper than 70 km are dropped.
- **Rare events** — M8+ cards are scored separately. Small n. Still issue them. Do not pretend 3 hits prove a theory.

We use CSEP-style tests because they are a decent scoreboard, not because we are trying to join a testing center.

## What we will not call a prediction

- “Elevated global risk” with no region and no magnitude
- A model that only outputs aftershock rates after an M7
- A YouTube timestamp with no frozen rule
- An in-sample window fitted after seeing the quake
