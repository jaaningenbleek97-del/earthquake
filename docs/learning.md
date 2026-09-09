# Correlation mining and learning as we go

Yes. This is a normal data system. Earthquake catalogs plus solar and magnetic archives are large enough to be interesting and small enough to live on one machine for Wave 1.

Two loops share the same feature store:

1. **Hunt** — scan many sources and lags for relationships.
2. **Learn** — issue predictions, see what hit, update the model on the next day without peeking at that day’s quakes.

```
new observations (available_at <= T)
        ↓
   feature store at issue time T
        ↓
   ┌────┴────┐
   hunt      predict
   (correlate) (fusion / alarms)
        ↓         ↓
   candidate   prediction records
   drivers          ↓
        └──── scoreboard (hits after valid_to)
                    ↓
              update model for T+1
```

## Scale (honest)

| Source | Rough volume | Fine |
| --- | --- | --- |
| ComCat decades, M≥2.5 | millions of rows | Trivial |
| OMNI / SWPC, hourly, decades | hundreds of thousands | Trivial |
| Wilcox SPF, 10-day | thousands | Trivial |
| skyfield geometry, daily | cheap to compute | Trivial |
| TEC / GNSS / magnetometers | much bigger | Wave 2; still ordinary columnar data |

We do not need a cluster to start. Parquet + a daily job is enough. Scale out when TEC/GNSS lands.

## Hunt: look for correlations

At each `issue_time` (or in a batch over history), compute relationships between:

- **Global drivers:** SPF, proton density, Kp/Dst, F10.7, tides, lunar distance, planetary flags, TEC
- **Targets:** did M≥4 / M≥5 / M≥6 / M≥8 happen in a cell or region in the next 1d / 7d / 90d

Tools that belong here:

- Lagged cross-correlation and mutual information (protons today vs M6 tomorrow, etc.)
- Conditional rates: P(M≥6 | SPF window) vs P(M≥6 | not)
- Partial correlation / residual after ETAS, so we do not “discover” aftershocks again
- Region slices: maybe a driver only shows up on the Pacific ring or on thrust faults
- Multiple-testing control (FDR). If you test 500 lags you will get junk. Candidates that survive get a locked recipe and a later-year replay.

Output of hunt is not a press release. It is a **candidate driver list** that the prediction heads are allowed to use.

## Learn as it goes

Three levels, all possible. We start at 1 and 2.

### 1. Walk-forward refit (default)

Every day (or week):

1. Build features using only `available_at <= T`.
2. Issue prediction records for the next 1d / 7d.
3. After each window closes, score hits.
4. Refit fusion weights / alarm thresholds on all closed history through `T`.
5. Repeat at `T+1`.

This is how weather models and most honest quake papers should work. It *is* learning as it goes. No shuffle.

### 2. Bayesian / running skill

Alarm heads keep a live posterior:

- How often did `spf_significant_window` precede an M8 on the days since we started?
- Same for proton spikes, geometry peaks, cycle phase.

The next day’s alarm probability is the updated rate, not a frozen 2015 p-value. New events change the call.

### 3. Incremental / online models (later)

Neural fusion can take a replay buffer of recent days plus a slow pass over older years so it does not forget 2011 while chasing last month. Gradient-boosted trees can do the same with incremental updates.

Rules that keep this from cheating:

- The label for a window that has not closed yet is unknown. Do not train on it. The scorer leaves those cards ungraded and drops them from the scoreboard (`valid_to` after the last catalog event).
- A card issued on a dead feed is not a card. If OMNI or ComCat is past its `available_at` slack that morning, do not write and do not train as if you did. Wilcox is optional.
- Aftershocks after a hit are not extra independent wins unless we say we are predicting aftershocks.
- If the live model starts only firing after big quakes, that is ETAS in a trench coat. The scoreboard vs ETAS exists to catch that.

## What can go wrong (and how we handle it)

| Failure | What we do |
| --- | --- |
| Spurious “sun did it” from hunting 500 series | FDR + locked holdout years |
| Learning aftershocks and calling it solar | Residualize vs ETAS; declustered twin |
| The world changes (catalog density, solar cycle) | Walk-forward; recency weights; do not freeze 1990 forever |
| Tiny M8 sample | Separate ledger; still update; do not overclaim |
| Binary miss hides a close call | Keep binary hit; add a graded near-miss (km / hours / magnitude) for learning |
| Alarm on 40% of days | Locked monthly cap per head and region; first-come walk-forward |
| Aftershock blob called a walking chain | Cluster only `available_at` events; residual-gate vs ETAS; drop slab >70 km |
| Always-quiet looks like skill | Anti-prediction only in the train low-*p* tail, 4 days/month, vs always-quiet base rate |
| Hunter and predictor drift apart | Every card stores `{driver, lag_hours, contribution}` for the terms that entered the call |
| Stale OMNI / ComCat issued as if live | Watchdog refuses the morning; existing batches stay on disk |
| Fit overwrites last month’s brain | Registry snapshots the old JSON first; cards pin `registry_id` |

None of that makes the project impossible. It makes the learning loop honest.

## Build order for this loop

1. Feature store (already specified)
2. Batch hunter: `eqpred hunt` — every numeric feature × lags 0/1/2/3/7/14/30 × issued box, residual Spearman after ETAS, FDR on train, same-sign confirm on holdout. Catalog leftovers stay labeled. Nothing writes issued *p*.
3. Instant bench: `eqpred backtest` / the Backtest tab. Same card question, scored on weeks that already closed. Refine a knob without waiting for next week.
4. Residual brain: `eqpred brain` / the Brain tab. One tree sees every unconventional source together (including combinations), walk-forwards as weeks close, and keeps a journal of what failed. Does not write issued *p*.
5. Transfer + promote: `eqpred transfer` trains one arc and scores the other. `eqpred learn` runs brain + transfer + the promote gate. Issued *p* changes only if walk-forward beats ETAS and always-on — until then `data/eval/promote.json` says held.
6. Daily predict → wait → score → refit
7. Live posteriors on alarm drivers
8. Incremental fusion when the daily refit is too slow or too stale
