from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.catalog import catalog_asof, load_comcat, load_comcat_versions
from eqpred.models.etas import daily_background_and_cluster, fit_etas
from eqpred.models.gr import fit_b_value, save_gr, scale_rate
from eqpred.models.calibrate import fit_issued_scale
from eqpred.models.modulator import fit_modulator
from eqpred.models.coulomb import fit_issued_coulomb
from eqpred.models.cusp import fit_issued_cusp
from eqpred.models.nowcast import fit_issued_nowcast
from eqpred.models.wave2 import fit_issued_wave2
from eqpred.models.migration import fit_issued_migration
from eqpred.models.anti import fit_issued_anti
from eqpred.models.budget import save_budget
from eqpred.models.predict import GLOBAL_MAG, save_models
from eqpred.models.quiet import ready_train_mask, quiet_threshold, save_quiet
from eqpred.paths import feature_dir
from eqpred.stamp import parse_utc, utc_ts

# Still fit g on daily M≥7 counts for diagnostics / abandoned-recipe nulls. Issued p does not use it.
MODULATOR_MAG = GLOBAL_MAG
ETAS_MLE_YEARS = 3


def etas_mle_start(feature_start, train_end: str):
    """Keep the MLE window ~3y. A 1990 calendar must not stretch duration."""
    start = pd.Timestamp(feature_start)
    cut = utc_ts(train_end)
    return max(start, cut - pd.Timedelta(days=365 * ETAS_MLE_YEARS)).to_pydatetime()


def fit_all(train_end: str = "2018-12-31", root: Path | None = None):
    from eqpred.models.registry import preserve_current, register_current

    prior = preserve_current(root)
    if prior:
        print(f"registry kept prior brain {prior['id']}")
    features = pd.read_parquet(feature_dir(root) / "global_daily.parquet")
    features["issue_time"] = parse_utc(features["issue_time"])
    quakes = load_comcat(root)
    versions = load_comcat_versions(root)

    start = features["issue_time"].min().to_pydatetime()
    cut = utc_ts(train_end).to_pydatetime()
    etas_start = etas_mle_start(start, train_end)
    # Fit on the catalog as it looked at train_end — not today's revisions.
    asof_train = catalog_asof(quakes, cut, minmag=4.5, versions=versions)
    etas = fit_etas(asof_train, etas_start, cut, m0=4.5)

    train_quakes = asof_train.loc[
        (asof_train["magnitude"] >= etas.m0)
        & (asof_train["observed_at"] >= etas_start)
        & (asof_train["observed_at"] < cut)
    ]
    b = fit_b_value(train_quakes["magnitude"], etas.m0)
    save_gr(b, etas.m0, n=int(len(train_quakes)), root=root)

    days = pd.DatetimeIndex(features["issue_time"])
    hist = quakes.loc[quakes["magnitude"] >= etas.m0]
    lam = daily_background_and_cluster(hist, days, etas, versions=versions)
    lam_card = np.asarray(scale_rate(lam, MODULATOR_MAG, etas.m0, b), dtype=float)
    counts = (
        quakes.assign(day=quakes["observed_at"].dt.floor("D"))
        .loc[lambda d: d["magnitude"] >= MODULATOR_MAG]
        .groupby("day")
        .size()
    )
    daily_n = features["issue_time"].map(counts).fillna(0).to_numpy()
    train_mask = ready_train_mask(features["issue_time"], train_end)
    mod = fit_modulator(features, lam_card, daily_n, train_mask)
    threshold = quiet_threshold(lam, train_mask)
    save_models(etas, mod, root)
    save_quiet(threshold, root=root)
    scale = 1.0
    for horizon in (1, 7, 90):
        fitted = fit_issued_scale(train_end=train_end, horizon_days=horizon, root=root)
        if horizon == 7:
            scale = fitted
    fit_issued_nowcast(train_end=train_end, root=root)
    fit_issued_coulomb(train_end=train_end, root=root)
    fit_issued_cusp(train_end=train_end, root=root)
    fit_issued_wave2(train_end=train_end, root=root)
    fit_issued_migration(train_end=train_end, root=root)
    fit_issued_anti(train_end=train_end, root=root)
    save_budget(root=root)
    entry = register_current(root)
    if prior and prior["id"] != entry["id"]:
        print(f"registry new brain {entry['id']} (old {prior['id']} still on disk)")
    else:
        print(f"registry current {entry['id']}")
    return etas, mod, threshold, b, scale
