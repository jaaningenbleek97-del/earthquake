# Research citations and public APIs

Sources for connectors and for the skeptic stack. This lab still issues predictions from unconventional drivers; the papers below are inputs, not a permission slip.

Prefer these URLs when implementing connectors.

## Definitions

- USGS — What is the difference between earthquake early warning, forecasts, probabilities, and prediction?  
  https://www.usgs.gov/faqs/what-difference-between-earthquake-early-warning-earthquake-forecasts-earthquake-probabilities
- USGS — Do solar flares or magnetic storms cause earthquakes?  
  https://www.usgs.gov/faqs/do-solar-flares-or-magnetic-storms-space-weather-cause-earthquakes
- Jordan et al. (2011). Operational Earthquake Forecasting: State of Knowledge and Guidelines for Utilization. *Annals of Geophysics*.  
  https://doi.org/10.4401/ag-5350

## Operational forecast models

- Mizrahi et al. (2024). Developing, testing, and communicating earthquake forecasts: current practices and future directions. *Reviews of Geophysics*, 62, e2023RG000823.  
  https://doi.org/10.1029/2023RG000823
- Ogata, Y. (1988/1998). Epidemic-type aftershock sequence (ETAS) models.
- Reasenberg, P. A., & Jones, L. M. (1989). Earthquake hazard after a mainshock in California. *Science*.
- Gerstenberger, M. C., et al. (2005). Real-time forecasts of tomorrow's earthquakes in California (STEP).
- Rhoades, D. A., & Evison, F. F. (2004). EEPAS: every earthquake a precursor according to scale.
- Jackson, D. D., & Kagan, Y. Y. (1999). Proximity to past earthquakes (PPE) smoothed seismicity.
- Field, E. H., et al. UCERF3-ETAS and later improvements. *The Seismic Record* / BSSA.
- Field, E. H., et al. (2025). A scientific vision and roadmap for earthquake rupture forecast developments, a USGS perspective. *BSSA*, 115(6).  
  https://doi.org/10.1785/0120240217  
  https://pubs.usgs.gov/publication/70273218
- USGS Operational Aftershock Forecasting software.  
  https://www.usgs.gov/software/operational-aftershock-forecasting
- Christophersen, A., et al. (2017). Progress and challenges in operational earthquake forecasting in New Zealand. NZSEE.  
  https://db.nzsee.org.nz/2017/O3C.4_Christophersen.pdf
- Rastin, S. J., et al. (2020). Catalogue lead time and EEPAS on New Zealand data. *Entropy*.  
  https://doi.org/10.3390/e22111264
- EEPAS open-source revisit (2025 preprint).  
  https://arxiv.org/html/2512.13064

## Evaluation software

- Savran, W. H., et al. pyCSEP: a Python toolkit for earthquake forecast developers.  
  https://github.com/SCECCode/pycsep  
  https://docs.cseptesting.org/  
  https://doi.org/10.1785/0220220033
- Mizrahi, L., et al. ETAS parameter inversion and catalog simulation (Swiss Seismological Service / ETH).  
  https://github.com/lmizrahi/etas

## AI / nowcast papers

- Kianifar, M. A., & Hamann, H. (2024). Time series foundation models and deep learning architectures for earthquake temporal and spatial nowcasting.  
  https://arxiv.org/abs/2408.11990
- Neural earthquake forecasting with minimal information: limits, interpretability, and Markov structure (2026 / arXiv:2509.14661).  
  https://arxiv.org/html/2509.14661v1
- SeismoQuakeGNN (2025). Hybrid GNN + Transformer. *Frontiers in Artificial Intelligence*.  
  https://www.frontiersin.org/articles/10.3389/frai.2025.1690476
- Scalable intermediate-term earthquake forecasting with multimodal fusion neural networks. *Scientific Reports* (2025).  
  https://www.nature.com/articles/s41598-025-93877-7
- Xie, Y., et al. (2024). Deep learning in earthquake engineering: a comprehensive review.  
  https://arxiv.org/abs/2405.09021

## Solar-terrestrial: communicators

- Davidson, B., U-Yen, K., & Holloman, C. (2015). Relationship between M8+ earthquake occurrences and the solar polar magnetic fields. *New Concepts in Global Tectonics Journal*, 3(3), 311–323.  
  http://suspicious0bservers.org/wp-content/uploads/2016/04/Paper-1.pdf
- Space Weather News data repository (SPF + M8+ through 2018, Python 2.7).  
  https://spaceweathernews.com/data-repository/
- Science Feedback review of Suspicious0bservers climate claims (not seismology, documents the broader channel).  
  https://science.feedback.org/review/the-sun-isnt-responsible-for-current-climate-change-contrary-claims-suspicious0bservers-ben-davidson/
- Stefan Burns — professional background (geology / Geometrics / Science Channel).  
  https://jason.org/stefan-burns/
- Cataldi, G., Cataldi, D., & Straser, V. EGU abstracts on solar-wind proton density and M6+ seismicity, e.g.  
  https://meetingorganizer.copernicus.org/EGU2019/EGU2019-3067.pdf

## Solar-terrestrial: peer-reviewed tests

- Love, J. J., & Thomas, J. N. (2013). Insignificant solar-terrestrial triggering of earthquakes. *Geophysical Research Letters*, 40, 1165–1170.  
  https://earthweb.ess.washington.edu/jnt/Thomas_Love_GRL_2013.pdf
- Marchitelli, V., Harabaglia, P., Troise, C., & De Natale, G. (2020). On the correlation between solar activity and large earthquakes worldwide. *Scientific Reports*, 10, 11495.  
  https://doi.org/10.1038/s41598-020-67860-3
- Khazaei, B., et al. (2022). Is the apparent correlation between solar-geomagnetic activity and powerful earthquakes a casual artifact? *Atmosphere*, 13, 1131.  
  https://doi.org/10.3390/atmos13071131
- Cochran, E. S., Vidale, J. E., & Tanaka, S. (2004). Earth tides can trigger shallow thrust fault earthquakes. *Science*.  
  https://www.science.org/doi/10.1126/science.1103961
- TEC enhancement before seismic events argued as artifact. *JGR Space Physics* (2020).  
  https://doi.org/10.1029/2020JA028733
- Freund / peroxy-defect review: earthquake precursors in the light of peroxy defects theory. *EPJ Special Topics* (2021).  
  https://doi.org/10.1140/epjst/e2020-000243-x
- Novikov et al. Space weather and earthquakes: possible triggering by strong solar flares. *Annals of Geophysics*.  
  https://doi.org/10.4401/ag-7975

## Tried drivers — negative results (do not re-run under new names)

- **Southward IMF (Bz) family — swept 2026-11-09, negative.** Built 68 derived variants from the OMNI IMF columns (strength: min/mean/peak Bz, southward integral −ΣBz; duration: days-below / fraction / longest-streak at thresholds −2/−4/−5/−8/−10 nT; onset delay; dawn/dusk By coupling; strong-condition gating Bz × Kp / wind / Dst) and scored 1,904 trials against first-shock (fresh) hits on the standard discover→confirm→replay split. One family (Bz<−8 nT duration/count/streak, Alpine M5) "confirmed" on early/late train (ρ≈+0.15→+0.12), but 0/1904 replayed: the 2019+ replay ρ is negative (−0.03), and Bz<−8 nT occurs on only 1 day in 2,867 modern days (0.14%), so the confirmation is an overfit to a handful of deep-storm days. All other strength/duration/gating variants were null-to-negative everywhere. Artifacts: `data/eval/research/bz_south_sweep.{json,csv}` and `scripts/research/bz_south_sweep.py`. Bz-south as a standalone forward predictor shows no durable skill and should not be re-scored as a new idea.

- **Southward-Bz storm-state onset — swept 2026-11-09, negative.** Second pass testing the physically-motivated version: southward Bz as the *switch* whose energy deposit shows up as the geomagnetic response, not as a raw level. 24 variants, 672 trials scored on the same split: rate-of-change (dBz/dt, southward-swing size), storm-state gating (Bz south × Dst drop / Dst≤−50), sudden commencement (Dst day-over-day drop ≥30 nT), substorm onset (AE jump / AE≥400), coupling-function gating (Newell/epsilon × southward), and a lead-lag chain (Bz south at t → Dst drop at t+1..2, marked retrospective-only/never-issuable). **Nothing discovered after FDR correction (q<0.10): 0/672.** Raw top-|ρ| ≈0.11–0.14 with p≈0.001–0.005 does not survive the multiple-comparison gate over 672 tests, and sign is roughly balanced (299+/373−) — pure noise. The largest confirm-era ρ (0.15–0.21 on `bz_south_kp_7d`, `newell_x_south_7d`, `dst_storm_flag` at 14/27-day lags) all have train ρ ≈0 or opposite sign, so they never passed discover, and their replay ρ is near-zero or negative. Artifacts: `data/eval/research/bz_south_storm_sweep.{json,csv}` and `scripts/research/bz_south_storm_sweep.py`. Southward-Bz onset as a storm-state trigger shows no durable first-shock skill.

## Public APIs for Wave 1 connectors

| Source | Base URL | Use |
| --- | --- | --- |
| USGS ComCat (FDSN event) | https://earthquake.usgs.gov/fdsnws/event/1/ | Global earthquake catalog |
| GeoNet FDSN | https://service.geonet.org.nz/fdsnws/event/1/ | New Zealand overlay |
| NOAA SWPC data service | https://services.swpc.noaa.gov/json/ | F10.7, GOES X-ray, Kp, solar wind. Docs: https://www.swpc.noaa.gov/content/data-access |
| NOAA NCEI space-weather archive | https://www.ncei.noaa.gov/ | Historical SWPC products |
| NASA OMNIWeb | https://omniweb.gsfc.nasa.gov/ | ACE/DSCOVR IMF, plasma, proton density; also Kp, Dst, F10.7 |
| Wilcox Solar Observatory | http://wso.stanford.edu/ | Solar polar magnetic field 10-day averages |
| SILSO sunspot number | https://www.sidc.be/silso/datafiles | Solar cycle phase |
| Kyoto WDC geomagnetic indices | https://wdc.kugi.kyoto-u.ac.jp/ | Dst, AE, SYM-H (also via OMNI) |
| GFZ Potsdam Kp | https://kp.gfz.de/ | Official Kp / ap |

Wave 2 / 3 pointers:

- NOAA GloTEC / US-TEC: https://www.ncei.noaa.gov/products/space-weather/ionospheric-program/total-electron-content
- Nevada Geodetic Laboratory GNSS: http://geodesy.unr.edu/
- INTERMAGNET: https://intermagnet.org/
- NCEI Daily Particle Data (GOES proton fluence, through 2020 Q1): https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/annual_reports/daily_solar_indices_summaries/daily_particle_data/
- SWPC GOES proton JSON (short live tail): https://services.swpc.noaa.gov/json/goes/primary/
- Perez-Silva et al. (2025) Hikurangi SSE catalog: https://doi.org/10.5281/zenodo.14848128
- Slow Earthquake Database (no redistribution of their CSV): https://eqdb.sakura.ne.jp/

## How to cite this briefing

The canvas and these docs summarize published work. They are not a new statistical result. Any later experiment in this repo should cite the original papers above and state the holdout cut date.