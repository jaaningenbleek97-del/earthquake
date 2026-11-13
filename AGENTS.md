# Earthquake prediction lab — coordination rules

Canonical working directory: `C:/Users/User/earthquake`
Private remote: `https://github.com/jaaningenbleek97-del/earthquake`
Working branch: `laptop-backup-2026-10-27` (GitHub default is `cursor/fresh-mainshock-tracks`; do NOT silently switch branches or assume default == checkout).

## What this project is
An experimental earthquake **prediction** lab. It issues timed, boxed calls
(region + magnitude floor + window + probability) for **first/mainshocks**, driven
by unconventional timestamped sources — solar polar fields, solar wind, geomagnetic
indices, tides/lunar geometry, ionosphere, GNSS, and more — stacked on catalog
clustering. It keeps score against a dumb catalog baseline. **Experimental, not a
validated public safety warning.** A large displayed probability is not proof of
calibrated prediction.

## The three agents and who touches what
- **Cursor** — edits code/research in this checkout (local).
- **Hermes** — runs the scheduled cards/ops and reads/writes this checkout (local).
- **Grok Bot "earthquake"** — cloud-side, CANNOT read these local `C:\` files. It is
  briefed via its chat; anything it needs to read/write must be shared online (Drive)
  or pasted to it.

## Canonical files (all local)
- `README.md` + `docs/*.md` — product rules, architecture, models, research.
- `data/` — raw, features, models, predictions, eval; the UI reads this.
- `scripts/daily_cards.ps1`, `scripts/weekly_refresh.ps1` — scheduled ops.
- `.venv/Scripts/python.exe` / `eqpred.exe` — use THIS interpreter, not global Python.

## Hard rules (all agents)
1. Predict **first/mainshocks**, never relabel aftershock skill as mainshock skill.
2. **No future leakage** — features carry `observed_at`/`available_at`; a call issued
   at `T` may only use data available at `T`.
3. Issued cards, recipes, models and the ledger are **frozen**; never rewrite history
   to repair explanations.
4. Keep 30-day / short-horizon / research / catalog-aftershock tracks distinct.
5. Don't re-tread already-tried strategies (99 ranked strategies + 30 research families
   are already on record).

## Do not
- Ship ETAS and call it done; claim aftershock skill as mainshock prediction.
- Ingest sources that can't be replayed from a timestamped archive.
- Train on shuffled catalogs.
