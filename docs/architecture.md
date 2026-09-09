# Data platform architecture

The store exists so we can **issue predictions** from mixed, often unconventional drivers. Catalog clustering is in the mix. It is not the product.

```
connectors  →  raw parquet  →  point-in-time aligner  →  feature store
                    ├─ hunter (correlations, lags, conditional rates)
                    ├─ fusion predictor
                    ├─ solar-magnetic alarm
                    ├─ geometry alarm
                    └─ Poisson / ETAS (skeptic)
                              ↓
                    prediction records → scoreboard → refit for tomorrow
```

Correlation mining and walk-forward learning: [learning.md](learning.md).

See [predictions.md](predictions.md) for the issued-call format.

## Connector contract

Every connector writes records that validate against
[`schemas/observation.schema.json`](../schemas/observation.schema.json).

| Field | Type | Rule |
| --- | --- | --- |
| `source` | string | Stable id, e.g. `usgs.comcat`, `wilcox.spf`, `nasa.omni` |
| `observed_at` | UTC datetime | When the physical quantity refers to |
| `available_at` | UTC datetime | When a predictor issued at this time could have used it |
| `geom` | object | `kind`: `point` \| `grid` \| `global` |
| `payload` | object | Source-native fields plus normalized aliases |
| `license` | string | SPDX or URL |
| `ingest_run_id` | string | Idempotency key |

Normalized aliases:

- Catalog: `magnitude`, `magnitude_type`, `depth_km`, `event_id`
- Solar / wind: `f107`, `sunspot_number`, `proton_density_cm3`, `solar_wind_speed_kms`, `imf_bz_nt`, `xray_flux`
- Magnetic: `kp`, `ap`, `dst_nt`, `ae_nt`, `sym_h_nt`
- SPF: `spf_north_ut`, `spf_south_ut`, `spf_average_ut`
- Tides / sky: `tidal_strain`, `lunar_phase_deg`, `lunar_distance_re`, `lunar_declination_deg`
- Ionosphere: `tec_tecu`

`available_at = observed_at + declared_latency` unless the archive has an issuance time. ComCat uses a magnitude-dependent first-report delay (hours for M7, up to two days for M4.5) when we have no product history. `usgs.comcat.versions` stores ComCat origin revisions for M≥6 (`includesuperseded`). Predictors at issue `T` see `catalog_asof`: latest version ≤ T (deleted products disappear), delay visibility for unversioned events, and Mc at origin time — not Mc at T — so a 1995 M4.6 stays hidden in 2020. ETAS λ uses piecewise magnitude on versioned events. Scoring hits still uses the final bulletin. Pull history with `eqpred ingest --only usgs.comcat.versions` (opt-in; not part of a default wave-1 ingest).

`eqpred watchdog` is a dead-man's switch on that clock. As-of issue morning `T`, OMNI and ComCat must have a latest `available_at` no older than `T` plus a small slack (OMNI 5d, ComCat 7d — global M≥4.5 is frequent enough that a week of silence means the pull is dead, not that Earth was quiet). Wilcox is optional: Polar.html ended in January 2026, so a stale SPF overlay must not block fusion cards. If a required source is stale or missing, `eqpred predict` refuses and leaves existing cards on disk. Wave 2/3 overlays can be stale; they do not block. Catch up one source with `eqpred ingest --only usgs.comcat` — catalog and daily pulls merge into the latest parquet so a 2026 tail does not drop 2016–2025. `eqpred predict` rewrites only the requested mornings; earlier batches stay, and their alarm spend is replayed into the monthly budget.

Issued brains live in `data/models/*.json` and are copied into `data/models/registry/{id}/` the first time they issue, and again whenever `fit` would replace them. The pin is `model_id` + `code_hash` (feature/predict source) + the feature-column recipe. `eqpred registry` lists versions. A later refit without that copy is amnesia.

Replayable archive or ephemeris. No social-media scrapes.

## Raw store

```
data/raw/{source}/dt={YYYY-MM-DD}/{ingest_run_id}.parquet
data/predictions/{model_id}/dt={YYYY-MM-DD}/{issued_at}.json
```

Raw observations are append-only. Predictions are the daily product.

## Point-in-time aligner

Issue time `T`, region, grid, source allow-list.

Keep the latest record with `available_at <= T`. Never gate on `observed_at` alone (revised catalogs and late OMNI fills leak the future).

Estimate Mc on a window that ends at `T`. Do not treat missing events below Mc as “no earthquake.”

## Feature store

### `features_cell`

Keys: `issue_time`, `cell_id`, `horizon_days`.

Catalog side (cell + neighbors): counts, log-energy, GR a/b, time since last M≥5/6/7, interevent quantiles, depth mix, optional fault distance, `declustered_*` twins. Issued boxes also get point-in-time migration: along-strike and up-dip km/day of the cluster visible that morning (`mig_along_*`, `mig_updip_*`).

### `features_global`

Keys: `issue_time`. This is where the unconventional stack lives.

- Solar cycle: F10.7, sunspots, phase, days since solar max
- Flares: GOES X-ray peaks 24h / 7d
- Solar wind: proton density, speed, IMF Bz (raw + 1-day lag + spike flags)
- Storms: Kp, ap, Dst, AE, SYM-H
- Wilcox SPF + Davidson Significant Window flag
- Tidal strain
- Lunar phase, distance, declination
- Optional planetary longitudes / hard-aspect flags (Burns-adjacent, from ephemerides)
- Wave 1.5+: TEC maps, GNSS strain, local magnetometers, Schumann if a public archive exists

Derived flags that encode someone else’s rule (Davidson windows, proton spikes) get a frozen definition in [solar-terrestrial.md](solar-terrestrial.md). Freeze before the test window. That is so we cannot cheat ourselves, not so we can dismiss the claim.

## Source waves

Unconventional drivers start in Wave 1. We do not wait for an agency blessing.

### Wave 1 — build first

| Source | Endpoint | Role |
| --- | --- | --- |
| USGS ComCat | https://earthquake.usgs.gov/fdsnws/event/1/ | What we score against, and catalog features |
| FDSN / GeoNet | FDSN event | Regional overlay |
| NOAA SWPC | https://services.swpc.noaa.gov/json/ | Live GOES X-ray / Kp / F10.7 (short JSON). Historical Kp/F10.7 stay on OMNI. |
| NASA OMNIWeb | https://omniweb.gsfc.nasa.gov/ | Proton density, IMF, speed — Cataldi / Marchitelli |
| Wilcox SPF | http://wso.stanford.edu/ | Davidson polar-field windows |
| SILSO | https://www.sidc.be/silso/datafiles | Cycle phase (Burns) |
| Tides + lunar + planets | local `skyfield` | Geometry alarm; no API |
| TEC (if the pull is clean) | NOAA GloTEC / IGS GIM | Ionosphere; pull early, label if flaky |

### Wave 2

`eqpred ingest --wave 2` pulls GeoNet FDSN, Tilde GNSS, weekly IGS TEC (per-box means), ERA5 NZ loading, and INTERMAGNET (all stations). `eqpred ingest --wave 3` pulls Japan/Chile USGS FDSN slices, GOES protons, GRACE-FO if a public mascon answers, and CAMS SO2 if CDS is configured. Swarm/CSES are skipped (no lightweight public daily file). Wave 2/3 columns join the feature store and may multiply issued *p* only if a one-coefficient residual gain beats the current issued baseline on train (greedy by family).

### Wave 3

Swarm / CSES, InSAR, volcano SO2. Raw waveforms are early warning, a different machine.

## Model heads

All heads read `features_cell` ⋈ `features_global` at `issue_time` and write [`prediction` records](predictions.md).

**Product**

1. **Fusion predictor** — TFT / Transformer. Catalog series + the full exogenous stack. 1d and 7d probability maps first, then 90d.
2. **Solar-magnetic alarm** — SPF windows, proton-density spikes, Kp/Dst, cycle phase. M≥6 and a separate M8+ card.
3. **Geometry alarm** — tides, lunar distance/declination, optional planetary aspects.

**Skeptic (required, not the roadmap)**

4. Poisson + GR + smoothed seismicity
5. ETAS / Reasenberg–Jones via [lmizrahi/etas](https://github.com/lmizrahi/etas)

If (1)–(3) cannot beat (4)–(5) on a later window, we say so and change the predictor. We do not delete the solar connectors.

## Build order

1. Observation + prediction schemas
2. ComCat + SWPC + OMNI + Wilcox + SILSO + skyfield geometry
3. Aligner and feature tables
4. Fusion + solar-magnetic + geometry heads writing daily prediction files
5. Scoreboard vs ComCat (binary hits, learning-only near-miss, time-in-alarm, monthly alarm budget, vs ETAS)
6. ETAS competitor wired into the same boxes so the comparison is fair
