# Prediction heads vs skeptic baselines

The thing we are building is a **predictor that uses unconventional data**. Catalog-only models are in the repo so we can tell if we are just rediscovering aftershocks.

## Product heads

| Head | Horizon | Output | Drivers | Build |
| --- | --- | --- | --- | --- |
| Fresh-shock classifier (`fresh.mainshock.v1`) | 7d | Issued probability + climatology beside it | Every timestamped non-catalog source. Catalog counts and ETAS λ stay out. Target is a declustered first shock. | Issued product |
| Solar-magnetic alarm | Days; M8+ as its own card | Binary alarm | Wilcox SPF windows | Second head |
| Geometry / non-catalog internals | Hours–days | Diagnostic only | Tidal strain, lunar geometry, planetary aspects | Features, not a separate issued card |

These write [prediction records](predictions.md). Fresh shock is the product.

## Skeptic baselines (competition, not the deliverable)

| Model | Why it is here | Build |
| --- | --- | --- |
| Poisson + GR + smoothed seismicity | Dumb spatial rate. If we cannot beat this, the neural net is lost. | Yes, as a foil |
| Reasenberg–Jones | Aftershock decay after a large event | Yes, as a foil |
| Space–time ETAS | Best operational clustering model. Italy, NZ, USGS aftershocks. | Lab foil only — [lmizrahi/etas](https://github.com/lmizrahi/etas). Not written as issued cards. |
| EEPAS | Medium-term catalog “precursory scale increase” (NZ) | Optional foil for 90d |
| STEP / UCERF / NSHM | Already exist in the world | Do not rebuild |

ETAS is strong **after** a sequence starts and weak **before** the first large event. That gap is where solar, magnetic, and geometry drivers have to earn their keep. If our “predictions” only light up after an M7, we failed.

## How the fresh-shock head should think

Not “predict the next aftershock from recent catalog counts.” That is ETAS.

Do this instead:

- Daily non-catalog stack known that morning. Catalog counts stay out.
- Target: declustered first shock in the box and window.
- Train earlier years, test later years. No shuffle.
- Issued p is the model call even when it has not beaten climatology yet. The historical rate sits beside it.
- ETAS stays as the skeptic we refuse to ship.

Published cousins (none of these are the product, they are architecture hints):

- TFT-style multi-horizon series (used for induced seismicity with a known injection schedule; we have solar/geometry instead)
- Spatial nowcasts on log-energy grids (arXiv:2408.11990)
- Hybrid NN + Markov (arXiv:2509.14661) — warning: looks good until the next decade

GNN-on-cells is allowed later if the grid fusion is already issuing calls.

## Scoreboard

We keep score because we are trying to be right, not because we are applying for a CSEP badge.

| Number | Question |
| --- | --- |
| Hit / miss | Did an M≥floor event land in the box and window? |
| Time-in-alarm | How often was the alarm on? Davidson’s 2015 windows covered ~42% of days — that is the bar for a “window” method. Issued alarms also sit under a locked monthly cap. |
| Migration | Along-strike / up-dip km/day of the cluster visible that morning. Diagnostic on every box; issued residual only if it beats the current baseline on train. |
| Anti-prediction | Dated “box stays quiet.” Train low-*p* tail + monthly cap. Hit = no qualifying quake. Compare to always-quiet. |
| Attribution | Every card: `driver`, `lag_hours`, `contribution`. Fusion uses `ln(gain)`. Solar flags stay off fusion cards. |
| Reliability | When we said 0.15, did ~15% of those cards hit? |
| vs ETAS (T/W or paired p) | Did we add anything beyond clustering? |
| Full vs declustered | Did a storm just overlap an aftershock sequence? |

Use [pyCSEP](https://docs.cseptesting.org/) where it fits (N/M/S/L, information gain). Also keep a simple prediction ledger. ROC on a shuffled catalog is worthless here.

M8+ cards stay on a separate ledger. Small n. Still issue them.

## Pass / iterate

A head stays in the daily job if it is issuing real records. It gets to claim skill only if the later-window scoreboard beats ETAS or clearly beats Poisson when ETAS is quiet (pre-mainshock). Negative results stay in the log. Connectors stay either way.
