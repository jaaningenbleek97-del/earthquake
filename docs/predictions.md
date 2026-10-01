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

Optional: `rationale` (human text), `competitor_climatology_p` (train-era fresh-shock rate shown beside issued p), `competitor_etas_p` (historical aftershock foil on old ETAS cards only).

## Issued heads

The product is a **fresh-shock** call. Catalog-only models sit beside it as competition, not as the roadmap.

1. **Fresh-shock classifier** (`fresh.mainshock.v1`) — every timestamped non-catalog source we can replay (space weather, tides, ionosphere, GPS, and the rest). Target is a declustered first shock. Catalog counts and ETAS λ stay out. Issued `value` is that model's p. Climatology sits in `competitor_climatology_p`.
2. **Solar-magnetic alarm** — Davidson-style SPF windows. A second head, not aftershock physics. Outputs a rare M8+ alarm.
3. **Aftershock ETAS** — lab foil only. Old `fusion.etas_x_mod.v1` batches stay on disk. New predict runs do not write ETAS, quiet-ETAS, or anti cards.

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
- **Anti-prediction** — historical “this box stays quiet” cards on the old ETAS ledger. New predict runs do not issue them.
- **Vs climatology** — same boxes, same days. If issued p is just the train-era rate, the scoreboard says so.
- **Vs ETAS** — lab foil. If we only rediscovered aftershocks, the foil says so.
- **Attribution** — every card lists the terms that entered the number. Fresh-shock rows name the classifier connections. Alarm rows name the gates that spent the budget.
- **Migration** — a cluster visible that morning, projected onto the locked fault plane. Along-strike and up-dip km/day are diagnostics. One of them may multiply box *p* only if it beat the current issued baseline on train. Events deeper than 70 km are dropped.
- **Rare events** — M8+ cards are scored separately. Small n. Still issue them. Do not pretend 3 hits prove a theory.

We use CSEP-style tests because they are a decent scoreboard, not because we are trying to join a testing center.

## What we will not call a prediction

- “Elevated global risk” with no region and no magnitude
- A model that only outputs aftershock rates after an M7
- A YouTube timestamp with no frozen rule
- An in-sample window fitted after seeing the quake
