# Unconventional earthquake prediction lab

This repo tries to **predict** earthquakes.

Agencies already publish clustering forecasts (ETAS, aftershock decay, long-term hazard maps). We are not here to rebuild those and stop. Catalog clustering is one input. The product is a model that also eats solar polar fields, solar wind, geomagnetic indices, tides, lunar geometry, ionosphere, and anything else we can timestamp and replay — then **issues a call**: where, how big, and in what window.

USGS will tell you this is not possible. That is the point of the experiment. If the unconventional stack cannot beat a dumb catalog baseline on a later time window, we keep the data and change the model. We do not pretend a Poisson map is the deliverable.

## Read this first

| Doc | What it is |
| --- | --- |
| [docs/predictions.md](docs/predictions.md) | What a prediction is, and how we issue one |
| [docs/learning.md](docs/learning.md) | Correlation mining and learning as it goes |
| [docs/roadmap.md](docs/roadmap.md) | Upgrades that make predictions sharper |
| [docs/architecture.md](docs/architecture.md) | Connectors, feature store, source waves |
| [docs/models.md](docs/models.md) | Prediction heads vs skeptic baselines |
| [docs/solar-terrestrial.md](docs/solar-terrestrial.md) | Davidson, Burns, Cataldi/Straser → features |
| [docs/research.md](docs/research.md) | Papers and public APIs |
| [schemas/prediction.schema.json](schemas/prediction.schema.json) | Issued prediction record |
| [schemas/observation.schema.json](schemas/observation.schema.json) | Ingested observation envelope |

Briefing canvas (open beside chat):
[earthquake-forecast-briefing](C:/Users/User/.cursor/projects/c-Users-User-earthquake/canvases/earthquake-forecast-briefing.canvas.tsx).

## Product rules

1. **Issue predictions.** Every run writes records with time window, region, magnitude floor, and a probability or alarm. Vague “elevated risk” with no box is not a prediction.
2. **Unconventional drivers are first-class.** Solar, magnetic, tidal, lunar, and ionospheric series are not an optional appendix. They still have to beat ETAS on a later window. Issued global *p* is calibrated ETAS until *g* shows timing skill; solar stays on the quiet specialist and on the abandoned-recipe nulls. Raw ETAS is the skeptic.
3. **No future leakage.** Features have `observed_at` and `available_at`. A prediction issued at `T` may only see data available at `T`. ETAS λ uses the same rule: ComCat events enter the history after a magnitude-dependent first-report delay. Hits still use the final bulletin. We do not have snapshot history, so magnitude revisions are not replayed.
4. **Hunt, then learn.** Scan sources and lags for correlations. Issue calls. When a window closes, update rates and refit. The model is allowed to change as new data arrives.
5. **Keep score.** Hit rate, time-in-alarm, and vs ETAS on later windows. If we are just rediscovering aftershocks, the scoreboard says so.
6. **Replay or it does not ship.** Timestamped public archive, or compute it from ephemerides. No YouTube scrapes.

## What we will not do

- Ship ETAS and call it done
- Wait for agency permission to try solar or magnetic drivers
- Train on shuffled catalogs
- Ingest sources we cannot replay

## Wave 1 is runnable

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
pytest
eqpred ingest --start 2016-01-01 --end 2026-01-01 --minmag 4.5
eqpred ingest --start 2016-01-01 --end 2026-01-01 --wave 2
eqpred ingest --start 2016-01-01 --end 2026-09-07 --wave 3
eqpred ingest --start 2025-12-01 --end 2026-09-07 --only usgs.comcat
eqpred watchdog
eqpred features --train-end 2018-12-31
eqpred fit --train-end 2018-12-31
eqpred predict --start 2019-01-01 --end 2026-01-01
eqpred battery --holdout-start 2019-01-01
eqpred score
eqpred strategies --holdout-start 2019-01-01
eqpred hunt
eqpred backtest --from-hunt
eqpred backtest --driver f107 --lag 30 --region alpine
eqpred brain
eqpred transfer
eqpred learn
```

Or `eqpred pipeline --start 2016-01-01 --end 2026-01-01 --minmag 4.5`.

What that writes:

| Path | What |
| --- | --- |
| `data/raw/{source}/{run}.parquet` | Wave 1: ComCat, OMNI, Wilcox, SILSO, geometry, SWPC. Wave 2: GeoNet, NGL GNSS (site stack), IGS TEC (per-box), ERA5, INTERMAGNET stations. Wave 3: Japan/Chile FDSN, GOES protons, GRACE, CAMS SO2. ComCat `available_at` is origin + first-report delay. |
| `data/features/global_daily.parquet` | Point-in-time daily stack (`available_at <= issue_time`) |
| `data/models/etas.json`, `modulator.json`, `quiet.json`, `gr.json`, `calibration.json`, `budget.json`, `migration.json` | Temporal ETAS, unused-on-issue solar g, quiet λ cut, locked GR b, issued scale per region (calibrated ETAS; λ is M≥4.5), monthly alarm caps, box migration residual |
| `data/predictions/batch-*.json` | Fusion global M≥7; Hikurangi / Alpine M≥5; Kermadec M≥6; SPF M8; quiet-regime M7 and M8; anti-prediction (box stays quiet) |
| `data/eval/battery.json` | Davidson v1 vs v2, quiet SPF, Marchitelli, Love–Thomas, issued calibrated ETAS vs abandoned g / time-shift / ETAS-world / climatology |
| `data/eval/strategies.json` | Walk-forward condition → box backtests (paper trades vs always-in base rate) |
| `data/eval/hunt.json` | Every feature column × lag × box vs next-week hits; train FDR then holdout confirm |
| `data/eval/backtest.json` | Paper trades on closed holdout weeks — no waiting for a live forecast |
| `data/eval/brain.json` | Residual tree on every unconventional source at once; journal under `data/eval/brain/` |
| `data/eval/transfer.json` | Japan ↔ Chile residual transfer; not issued *p* |
| `data/eval/promote.json` | Gate: walk-forward lift and leftover error vs ETAS. Never writes issued models. |
| `data/predictions/ledger.parquet` | Binary hits vs ComCat, plus a learning-only near-miss grade |

SWPC JSON is a short operational feed (GOES X-ray / live Kp). The long replay for wind, IMF, Kp, F10.7 is OMNI. Wilcox Polar.html is ingested in full, then the Significant Window is delayed 40 days so we do not alarm on a peak that had not been posted.

`eqpred pipeline --skip-ingest` reruns features → score on the latest raw dump.

`eqpred watchdog` prints source freshness as-of now (or `--as-of`). OMNI and ComCat must be fresh or `eqpred predict` refuses and leaves the last batch on disk. Wilcox is optional (Polar.html ended). `eqpred ingest --only usgs.comcat` (or `nasa.omni`, `wilcox.spf`, `computed.geometry`) pulls a tail and merges it into the latest parquet. `eqpred predict --start 2026-01-01 --end 2026-09-07` appends that window; it does not wipe the holdout.

`eqpred registry` pins the active model JSON + code hash + feature recipe, and writes `.meta.json` next to unpinned batches. `eqpred fit` snapshots the current version before it overwrites.

```powershell
eqpred serve
```

opens a local UI at http://127.0.0.1:8765 — scoreboard, filterable ledger, day inspector, driver charts, and a 2D map of the New Zealand box plus ComCat events. It reads `data/`. It is not a 3D globe; global cards are Earth-wide and have no polygon yet.
