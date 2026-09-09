from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eqpred.paths import model_dir
from eqpred.stamp import parse_utc, utc_ts

# Bottom third of training-era ETAS λ counts as "boring".
QUIET_QUANTILE = 0.35
# Drop the first year: λ can sit at μ before Omori tails exist. That is not a quiet Earth.
SPINUP_DAYS = 365
# Issued scale / garnish train on this slice so a 1990 calendar cannot retune mean p.
ISSUED_SCALE_START = "2016-01-01"
# Holdout: nowcast / wave2 / coulomb / cusp / migration lost to scale-only.
# Fit still records the train gate; issued p does not multiply them.
ISSUED_GARNISH = False


def hold_issued_garnish(meta: dict) -> dict:
    """Keep train-gate diagnostics; never put garnish on issued p."""
    out = dict(meta)
    if "used_on_issued" in out:
        out["train_gate_pass"] = bool(out.get("train_gate_pass", out.get("used_on_issued")))
        if not ISSUED_GARNISH:
            out["used_on_issued"] = False
    regions = out.get("regions")
    if isinstance(regions, dict):
        out["regions"] = {rid: hold_issued_garnish(val) if isinstance(val, dict) else val for rid, val in regions.items()}
    columns = out.get("columns")
    if isinstance(columns, list):
        out["columns"] = [hold_issued_garnish(col) if isinstance(col, dict) else col for col in columns]
    return out
UNCONVENTIONAL = (
    "spf_significant_window",
    "proton_spike",
    "kp_high",
    "dst_storm",
)


def ready_train_mask(
    issue_time: pd.Series,
    train_end: str,
    spinup_days: int = SPINUP_DAYS,
) -> np.ndarray:
    start = parse_utc(issue_time).min()
    ready = parse_utc(issue_time) >= (start + pd.Timedelta(days=spinup_days))
    in_train = parse_utc(issue_time) < utc_ts(train_end)
    return (ready & in_train).to_numpy()


def issued_train_mask(
    issue_time: pd.Series,
    train_end: str,
    scale_start: str = ISSUED_SCALE_START,
    spinup_days: int = SPINUP_DAYS,
) -> np.ndarray:
    """Train mask for issued scale and garnish. Starts 2016, not the OMNI min."""
    base = ready_train_mask(issue_time, train_end, spinup_days=spinup_days)
    recent = parse_utc(issue_time) >= utc_ts(scale_start)
    return base & recent.to_numpy()


def quiet_threshold(lambda_etas: np.ndarray, train_mask: np.ndarray, quantile: float = QUIET_QUANTILE) -> float:
    sample = np.asarray(lambda_etas, dtype=float)[np.asarray(train_mask, dtype=bool)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("inf")
    return float(np.quantile(sample, quantile))


def is_quiet(lambda_etas: float, threshold: float) -> bool:
    return bool(np.isfinite(lambda_etas) and lambda_etas <= threshold)


def unconventional_drivers(row: pd.Series) -> list[str]:
    names: list[str] = []
    for name in UNCONVENTIONAL:
        if name in row.index and float(row[name] or 0) > 0.5:
            names.append(name)
    return names


def save_quiet(
    threshold: float,
    quantile: float = QUIET_QUANTILE,
    spinup_days: int = SPINUP_DAYS,
    root: Path | None = None,
) -> None:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"lambda_etas_max": threshold, "quantile": quantile, "spinup_days": spinup_days}
    (folder / "quiet.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_quiet(root: Path | None = None) -> dict:
    path = model_dir(root) / "quiet.json"
    if not path.exists():
        return {"lambda_etas_max": 0.0, "quantile": QUIET_QUANTILE}
    return json.loads(path.read_text(encoding="utf-8"))
