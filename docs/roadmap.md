# What would actually make this better

Prioritized upgrades for a lab that hunts correlations, issues predictions, and learns. Not a wishlist of every sensor on Earth.

## Do these first (they change whether calls are real)

1. **ETAS × modulator, not ETAS vs Sun**  
   Best predictor is usually `λ_cluster(x,t) × g(solar, storms, tides, moon)`. Clustering handles aftershocks. Unconventional drivers get to raise or lower the rate *before* a mainshock. If `g` is always ~1, we learned something. If `g` only spikes after M7, we failed.

2. **Quiet-regime specialist**  
   A second head that is only allowed to fire when ETAS is boring. That is the only place Davidson/Burns-style calls can prove they are not aftershock echo.

3. **Fake-realtime catalog**  
   Replay ComCat as it looked at issue time: missing early aftershocks, magnitude revisions, Mc jumps when a network improves. If we train on today’s cleaned bulletin, live predictions will lie.

4. **Time-shift and ETAS-world nulls**  
   Shuffle or shift solar series by months; also run the hunter on synthetic catalogs with *no* solar physics. Anything that still “works” is leakage or aftershock theft.

5. **Paper battery**  
   Auto-replay Davidson 2015 windows, Marchitelli protons, Love–Thomas null, Cochran tides, on the same store. If we cannot reproduce (or fairly refute) the claims we encode, the hunter is theater.

## Make the calls sharper

6. **Coulomb stress on faults, not scalar tide**  
   Tide + lunar load projected onto nearby fault orientation. Same data, much less mush. Needs a simple fault file (NZ / global GEM), not a full NSHM.

7. **Polar-cusp / storm longitude**  
   Some solar-wind papers say the hit should prefer longitudes under the cusp at storm onset. Encode that as a spatial weight, not a global siren.

8. **Near-miss score**  
   A M5.9 one cell over, 8 hours late, is not the same as a total miss. Keep binary hit for honesty; add a graded score for learning.

9. **Alert budget**  
   Cap alarms per region per month. A head that is on 40% of days is a weather channel. Force the model to spend scarce alarms.

10. **Anti-prediction**  
    “This box stays quiet for 7 days” is a call and often easier to score. Useful for learning and for not crying wolf.

11. **Attribution on every card**  
    Already in `drivers`; make it mandatory and structured (`driver`, `lag_hours`, `contribution`) so the hunter and the predictor stay aligned.

## Learning loop upgrades

12. **Meta-learner over specialists**  
    The 200-knob scan is `eqpred hunt`. The residual tree is `eqpred brain`: every unconventional source at once, scored on closed weeks, journaled so the next run can drop losers. Still not a soup in issued *p*. Keep SPF, protons, geometry, TEC as separate heads until a confirmed specialist earns a slot.

13. **Transfer test**  
    Train on Japan / Chile, issue on NZ (or the reverse). If skill dies at the border, we memorized a catalog, not a mechanism.

14. **b-value and energy-release nowcast**  
    Cheap catalog physics: falling b, accelerating log-energy, swarm vs mainshock-aftershock shape. Goes into fusion even if solar is quiet. Along-strike / up-dip cluster velocities on the locked NZ planes are the spatial half of the same idea.

15. **Model registry** — done  
    `eqpred registry` / `eqpred predict` pin the active brain (`data/models/*.json`) plus a hash of the feature/predict source plus the feature-column recipe. `eqpred fit` snapshots the current version before it writes; old `data/models/registry/{id}/` copies stay. New cards carry `registry_id` and `code_hash`. Existing batches get a `.meta.json` sidecar.

16. **Ingest watchdog** — done  
    `eqpred watchdog` / `eqpred predict` refuse a morning when OMNI or ComCat is past its `available_at` slack. Wilcox is optional. Overlays warn in the lab UI but do not block. `eqpred ingest --only usgs.comcat` merges a catalog tail so open windows can close without dropping history.

## Data worth adding (after Wave 1 works)

17. **GNSS transients / slow slip** — real loading, especially NZ. Hikurangi windows from Perez-Silva et al. 2025 are ingested as `slowslip.sse` (on/off days, not issued *p*). GNSS 30-day east diffs stay as the live proxy.  
18. **TEC** — already planned; treat as a map, not one global number.  
19. **INTERMAGNET local dB/dt** — storms are not uniform on the ground.  
20. **Schumann / VLF** — only with a public timestamped archive.  
21. **Volcanic SO2 / tremor** — separate target, coupled in some arcs.  
22. **Human discretionary card** — if you want to type a call after watching solar streams, log it as `model_id=human` and score it on the same ledger. Do not mix it into training.

## Product surface

23. **Ledger first, maps second**  
    A table of issued cards, outcomes, ETAS’s p, and which drivers fired. That is how you get better. A globe is decoration until the ledger exists.

24. **NZ overlay as the first “named region” pack**  
    GeoNet + Hikurangi / Alpine / Kermadec boxes. Global hunter, local issuance. Matches where you are and a dense catalog.

## Explicit non-goals (they make it worse)

- Scraping video “predictions” as training labels  
- Optimizing ROC on shuffled events  
- One giant unregularized correlation dump published as findings  
- Replacing the scoreboard with vibes after a single M7 coincidence
