# Unconventional drivers we actually encode

Agencies do not use these. We do. Each row becomes columns on `features_global` and can fire a [prediction](predictions.md).

The scoreboard is how we find out who was onto something. It is not a reason to leave the data on the table.

## Communicators we encode

### Ben Davidson (Suspicious Observers / Space Weather News)

Most concrete public algorithm in this scene.

- 2015 NCGT paper with Kongpop U-Yen and Christopher Holloman: Wilcox solar polar fields vs USGS M8+, 1976–Jan 2014.
- Significant Windows: SPF peaks/troughs (longer window on stronger peaks), pole reversals and average-field reversals, delay after an M8+ or a 4×M7.5 cluster in 100 days.
- In-sample: 41.6% of days on; 26/33 M8+ inside (78.8%). They said the p-value is in-sample and the write-up is not a live predictor. Fine. We freeze the rules and **run them forward**.
- Data: http://suspicious0bservers.org/wp-content/uploads/2016/04/Paper-1.pdf and https://spaceweathernews.com/data-repository/
- Features: `spf_north_ut`, `spf_south_ut`, `spf_average_ut`, `spf_significant_window`
- Live use: solar-magnetic alarm head issues M8+ cards on those windows. Holdout = post-2014 and post-2018. Do not retune window lengths after seeing new M8s.
- **v2 (shipped):** only N/S extrema with `|SPF| ≥ max(40, 80th percentile)`, ±8–14 day windows, average-field reversals ±12 days, pole reversals only if post-crossing intensity ≥ 25. Then a **20% train-era alert budget** keeps the strongest days (cutoff locked at `train_end`). Issued SPF cards also sit under a **6 day / month** cap, spent in date order — not retuned on holdout M8s. v1 is still scored in the battery so we can see the TIA drop. Polar.html is delayed 40 days before issue time.

### Stefan Burns

Geologist / geophysicist by training; the earthquake work is public commentary, not a paper.

- Story: the planet loads through solar maximum; descending phase and minimum get noisier (quakes, volcanoes). Flares, storms, eclipses, sometimes planetary geometry.
- We do not need a paper to test a cycle-phase feature.
- Features: `solar_cycle_phase`, `days_since_solar_max`, `f107_trend_90d`, plus geometry flags from skyfield (lunar, optional planetary aspects).
- Live use: those columns go into fusion and the geometry alarm. Eclipses are computable; include them as a binary if we want that Burns-style card.

### Cataldi / Straser (Radio Emissions / LTPA)

EGU abstracts: ACE/DSCOVR proton-density jumps before global M6+.

- Features: `omni_proton_density`, `omni_proton_density_lag1d`, `omni_imf_bz`, `proton_spike` (threshold locked on the training era).
- Live use: solar-magnetic alarm. Score M≥6. Always run a declustered twin so a storm sitting on an aftershock sequence does not fake a win.

### Dutchsinse and similar

No frozen public rule. We do not scrape the channel. If someone writes a replayable map rule, it can become a head.

## Papers we steal features from

| Source | What we take | Feature |
| --- | --- | --- |
| Marchitelli et al. 2020 Sci Rep | Proton density, ~1-day lag, large events | Same OMNI series; we re-run with declustering instead of arguing on Twitter |
| Love and Thomas 2013 GRL | “Nothing there” on sunspots / wind / Kp | We reproduce their null as the skeptic score. If our alarm beats it on a later window, that is news. |
| Anagnostopoulos et al. | Decay-phase / high-speed streams | Cycle phase + solar wind speed |
| Cochran et al. 2004 Science | Earth tides can trigger some thrust faults | `tidal_strain` — Wave 1, not controversial as a weak load |
| Pulinets / Ouzounov / DEMETER | Ionosphere before events | TEC in Wave 1 if the pull is easy, else 1.5. We do not care that the causal cartoon is “quake → sky” in some papers. |
| Freund | Stressed rock → currents / IR / air ions | Not Wave 1 (needs different sensors). Do not block solar work on this. |
| Novikov / Sobolev | Currents can modulate local seismicity | Motivation for caring about storms; not a global feature by itself |

USGS FAQ says space weather does not cause earthquakes and there is no 11-year seismicity cycle. That is their operating position. Ours is: put the indices in the model and issue calls.

## Locked rules (so we cannot cheat)

1. Write the feature recipe here and in constants before the test years.
2. Default cuts: SPF v2 + 20% alert budget and proton-spike thresholds locked on data available at `train_end` (default 2018-12-31). Quiet-regime threshold = 35th percentile of training-era ETAS λ.
3. Walk forward by day. Each day writes prediction records.
4. Score hits, time-in-alarm, and the same boxes vs ETAS.
5. Full catalog and declustered catalog.
6. If a driver adds nothing, it stays in the store and drops out of that head. We do not delete Wilcox because an agency paper was null.

## What we will not do

- Wait until this is mainstream
- Fit a new window after an M8 and call it a prediction
- Treat a storm that happened near a famous quake as a trophy
- Scrape video descriptions and call that a model
